"""
NeuroCore Konfigürasyon Sistemi
================================
Donanım profillerine göre otomatik ayarlanan, dinamik yeniden yapılandırma
destekleyen kapsamlı konfigürasyon yönetimi.

Donanım profilleri:
    MINIMAL : <4GB RAM, GPU yok
    LOW     : 4-8GB RAM, <4GB VRAM
    MEDIUM  : 8-16GB RAM, 4-8GB VRAM
    HIGH    : 16-32GB RAM, 8-12GB VRAM
    ULTRA   : 32GB+ RAM, 12GB+ VRAM
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ─── Donanım Profil Tanımları ───────────────────────────────────────────────

class HardwareProfile(Enum):
    """Donanım profilleri — sistem kaynakları otomatik algılanır."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

    @staticmethod
    def detect(total_ram_gb: float, gpu_vram_gb: float = 0.0) -> "HardwareProfile":
        """RAM ve GPU VRAM miktarına göre donanım profilini belirler."""
        if total_ram_gb >= 32 and gpu_vram_gb >= 12:
            return HardwareProfile.ULTRA
        elif total_ram_gb >= 16 and gpu_vram_gb >= 8:
            return HardwareProfile.HIGH
        elif total_ram_gb >= 8 and gpu_vram_gb >= 4:
            return HardwareProfile.MEDIUM
        elif total_ram_gb >= 4:
            return HardwareProfile.LOW
        else:
            return HardwareProfile.MINIMAL


# ─── Profil Bazlı Varsayılan Değerler ──────────────────────────────────────

PROFILE_DEFAULTS: Dict[HardwareProfile, Dict[str, Any]] = {
    HardwareProfile.MINIMAL: {
        "embedding_dim": 128,
        "max_batch_size": 8,
        "model_size": "tiny",
        "num_workers": 2,
        "cache_size_mb": 64,
        "num_transformer_layers": 2,
        "num_attention_heads": 2,
        "d_ff_multiplier": 2,
        "num_initial_experts": 2,
        "max_experts": 4,
        "memory_bank_size": 256,
    },
    HardwareProfile.LOW: {
        "embedding_dim": 256,
        "max_batch_size": 16,
        "model_size": "small",
        "num_workers": 4,
        "cache_size_mb": 256,
        "num_transformer_layers": 4,
        "num_attention_heads": 4,
        "d_ff_multiplier": 4,
        "num_initial_experts": 3,
        "max_experts": 6,
        "memory_bank_size": 512,
    },
    HardwareProfile.MEDIUM: {
        "embedding_dim": 512,
        "max_batch_size": 32,
        "model_size": "medium",
        "num_workers": 6,
        "cache_size_mb": 512,
        "num_transformer_layers": 6,
        "num_attention_heads": 8,
        "d_ff_multiplier": 4,
        "num_initial_experts": 3,
        "max_experts": 8,
        "memory_bank_size": 1024,
    },
    HardwareProfile.HIGH: {
        "embedding_dim": 768,
        "max_batch_size": 64,
        "model_size": "large",
        "num_workers": 12,
        "cache_size_mb": 1024,
        "num_transformer_layers": 8,
        "num_attention_heads": 12,
        "d_ff_multiplier": 4,
        "num_initial_experts": 4,
        "max_experts": 12,
        "memory_bank_size": 2048,
    },
    HardwareProfile.ULTRA: {
        "embedding_dim": 1024,
        "max_batch_size": 128,
        "model_size": "xlarge",
        "num_workers": 16,
        "cache_size_mb": 2048,
        "num_transformer_layers": 12,
        "num_attention_heads": 16,
        "d_ff_multiplier": 4,
        "num_initial_experts": 6,
        "max_experts": 24,
        "memory_bank_size": 4096,
    },
}


# ─── Kaynak Nezaket Politikası ("boştakinin %90'ı") ─────────────────────────

@dataclass
class HeadroomPolicy:
    """
    Sistem kaynaklarını **paylaşma** politikası.

    Temel kural::

        harcanabilir = toplam × (1 − mevcut_kullanım) × nezaket_çarpanı

    Örnekler (nezaket_çarpanı = 0.90):
        * CPU %50 kullanımda → ajan toplamın %45'ine kadar kullanır
        * Sistem %90 boşta   → ajan toplamın %81'ine kadar kullanır
        * RAM %85 kullanımda → ajan toplamın %13.5'ine kadar kullanır

    Neden gerekli?
        Kullanıcı tarayıcıda video izlerken ajanın RAM/GPU'yu doyurması
        sistemi kilitler. Bu politika ajanı **her zaman ikinci sıraya** koyar:
        kullanıcı ne kadar çok kullanıyorsa ajan o kadar küçülür.

    Attributes:
        cpu / ram / gpu: Kaynak başına nezaket çarpanı
        cpu_floor_cores: Sistem tamamen dolu olsa bile ayrılan minimum çekirdek
            (sıfır ilerleme yerine yavaş ilerleme tercih edilir)
        ram_floor_mb / gpu_floor_mb: Ayrılan minimum bellek
        reserve_system_mb: İşletim sistemi için her zaman boş bırakılan RAM
        max_*_percent: Ne olursa olsun aşılmaması gereken kullanım oranları
        stall_*_percent: Bu kullanımın üstünde **yeni iş başlatılmaz**
        recover_*_percent: Bu kullanımın altına inince normal çalışmaya dönülür
        cooldown_seconds: Histerezis penceresi — bütçe bu süre içinde büyümez
    """
    cpu: float = 0.90
    ram: float = 0.90
    gpu: float = 0.90

    cpu_floor_cores: int = 1
    ram_floor_mb: float = 128.0
    gpu_floor_mb: float = 256.0
    reserve_system_mb: float = 512.0

    max_ram_percent: float = 0.85
    max_gpu_percent: float = 0.85
    max_cpu_percent: float = 0.85

    stall_cpu_percent: float = 0.85
    stall_ram_percent: float = 0.85
    stall_gpu_percent: float = 0.90

    recover_cpu_percent: float = 0.70
    recover_ram_percent: float = 0.70
    recover_gpu_percent: float = 0.75

    cooldown_seconds: float = 20.0

    # ─── Hesaplamalar ───────────────────────────────────────────────────

    @staticmethod
    def free_fraction(used_fraction: float) -> float:
        """Boşta olan oranı hesaplar (1 − kullanım, 0..1'e kırpılır)."""
        return max(0.0, min(1.0, 1.0 - float(used_fraction)))

    def cpu_budget(self, total_cores: int, used_fraction: float) -> int:
        """
        Kullanılabilecek çekirdek sayısı.

        Örnek: 24 çekirdek, %50 kullanımda → 24 × 0.50 × 0.90 = 10.8 → 10
        """
        total_cores = max(1, int(total_cores))
        capacity = total_cores * self.free_fraction(used_fraction)
        fraction_budget = int(capacity * self.cpu)
        percent_cap = int(total_cores * self.max_cpu_percent)
        return max(int(self.cpu_floor_cores),
                   min(fraction_budget, percent_cap, total_cores))

    def ram_budget_mb(self, total_mb: float, free_mb: float) -> float:
        """
        Kullanılabilecek RAM (MB).

        Örnek: 32 GB toplam, 20 GB boş → min(20 × 0.90, 32 × 0.70) = 18 GB;
        ardından işletim sistemi payı (`reserve_system_mb`) düşülür.
        """
        if total_mb <= 0:
            return float(self.ram_floor_mb)
        used_fraction = 1.0 - (max(0.0, free_mb) / total_mb)
        capacity = total_mb * self.free_fraction(used_fraction)
        budget = min(capacity * self.ram, total_mb * self.max_ram_percent)
        budget -= self.reserve_system_mb
        return max(float(self.ram_floor_mb), budget)

    def gpu_budget_mb(self, total_mb: float, free_mb: float) -> float:
        """
        Kullanılabilecek VRAM (MB).

        VRAM sistem RAM'inden daha kritiktir (taşma sürücüyü kilitler), bu
        yüzden üst sınır daha korumacıdır ve taban pay büyüktür.
        """
        if total_mb <= 0:
            return 0.0
        used_fraction = 1.0 - (max(0.0, free_mb) / total_mb)
        capacity = total_mb * self.free_fraction(used_fraction)
        budget = min(capacity * self.gpu, total_mb * self.max_gpu_percent)
        return max(float(self.gpu_floor_mb), budget)

    # ─── Duraklama Kararları ────────────────────────────────────────────

    def should_stall(self, cpu_percent: float = 0.0, ram_percent: float = 0.0,
                     gpu_percent: float = 0.0) -> Tuple[bool, str]:
        """
        Yeni iş başlatılmamalı mı?

        Args:
            cpu_percent / ram_percent / gpu_percent: 0..100 arası kullanım

        Returns:
            (duraklat, gerekçe)
        """
        cpu = float(cpu_percent) / 100.0
        ram = float(ram_percent) / 100.0
        gpu = float(gpu_percent) / 100.0

        reasons: List[str] = []
        if ram >= self.stall_ram_percent:
            reasons.append(f"RAM %{ram * 100:.0f} ≥ %{self.stall_ram_percent * 100:.0f}")
        if cpu >= self.stall_cpu_percent:
            reasons.append(f"CPU %{cpu * 100:.0f} ≥ %{self.stall_cpu_percent * 100:.0f}")
        if gpu >= self.stall_gpu_percent:
            reasons.append(f"VRAM %{gpu * 100:.0f} ≥ %{self.stall_gpu_percent * 100:.0f}")
        if reasons:
            return True, "sistem kaynakları dolu: " + ", ".join(reasons)
        return False, ""

    def can_recover(self, cpu_percent: float = 0.0, ram_percent: float = 0.0,
                    gpu_percent: float = 0.0) -> bool:
        """Normal çalışmaya dönülebilir mi? (histerezis üst sınırı)"""
        return (
            float(cpu_percent) / 100.0 < self.recover_cpu_percent
            and float(ram_percent) / 100.0 < self.recover_ram_percent
            and float(gpu_percent) / 100.0 < self.recover_gpu_percent
        )

    def pressure_level(self, cpu_percent: float = 0.0, ram_percent: float = 0.0,
                       gpu_percent: float = 0.0) -> str:
        """
        Kaynak baskı seviyesi: `NORMAL` | `ELEVATED` | `HIGH` | `CRITICAL`.

        `HIGH`/`CRITICAL` seviyelerinde ajan iş yükünü küçültür; `CRITICAL`
        seviyesinde yeni iş başlatmaz.
        """
        worst = max(
            float(cpu_percent) / 100.0,
            float(ram_percent) / 100.0,
            float(gpu_percent) / 100.0,
        )
        if worst >= max(self.stall_cpu_percent, self.stall_ram_percent,
                        self.stall_gpu_percent):
            return "CRITICAL"
        if worst >= self.recover_cpu_percent + 0.10:
            return "HIGH"
        if worst >= self.recover_cpu_percent:
            return "ELEVATED"
        return "NORMAL"

    def recovery_delay(self, pressure_level: str) -> float:
        """Bu baskı seviyesinde yeni iş öncesi beklenecek süre (saniye)."""
        return {
            "CRITICAL": 8.0,
            "HIGH": 3.0,
            "ELEVATED": 0.5,
            "NORMAL": 0.0,
        }.get(pressure_level, 0.0)

    def describe(self) -> str:
        """Politikanın insan okur özeti."""
        return (
            f"Kaynak nezaketi: CPU/RAM/GPU için boştakinin %{self.cpu * 100:.0f}'ı. "
            f"Üst sınır: RAM %{self.max_ram_percent * 100:.0f}, "
            f"VRAM %{self.max_gpu_percent * 100:.0f}, CPU %{self.max_cpu_percent * 100:.0f}. "
            f"Duraklama eşiği: RAM/CPU %{self.stall_ram_percent * 100:.0f}, "
            f"VRAM %{self.stall_gpu_percent * 100:.0f}."
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "cpu_factor": self.cpu, "ram_factor": self.ram, "gpu_factor": self.gpu,
            "cpu_floor_cores": self.cpu_floor_cores,
            "ram_floor_mb": self.ram_floor_mb, "gpu_floor_mb": self.gpu_floor_mb,
            "reserve_system_mb": self.reserve_system_mb,
            "max_ram_percent": self.max_ram_percent,
            "max_gpu_percent": self.max_gpu_percent,
            "max_cpu_percent": self.max_cpu_percent,
            "stall_ram_percent": self.stall_ram_percent,
            "stall_cpu_percent": self.stall_cpu_percent,
            "stall_gpu_percent": self.stall_gpu_percent,
            "cooldown_seconds": self.cooldown_seconds,
            "summary": self.describe(),
        }


# ─── Alt-Sistem Konfigürasyonları ───────────────────────────────────────────

@dataclass
class HardwareConfig:
    """Donanım izleme ve kaynak paylaşımı ayarları."""
    monitor_interval_sec: float = 2.0
    ram_warning_threshold: float = 0.80
    ram_critical_threshold: float = 0.90
    gpu_warning_threshold: float = 0.85
    gpu_critical_threshold: float = 0.95
    max_ram_usage_percent: float = 0.70
    max_gpu_usage_percent: float = 0.80
    enable_gpu: bool = True
    #: Sistem kaynaklarını paylaşma politikası ("boştakinin %90'ı")
    headroom: HeadroomPolicy = field(default_factory=HeadroomPolicy)
    #: Uzun işlerde kaynak denetimi uygulansın mı? (eğitim, araştırma, indeksleme)
    enforce_headroom: bool = True


@dataclass
class KnowledgeConfig:
    """Bilgi motoru ayarları."""
    vault_path: str = ""
    max_versions: int = 10
    auto_backup: bool = True
    default_confidence: float = 0.5
    categories: list = field(default_factory=lambda: [
        "system", "learned", "research", "meta", "user"
    ])
    embedding_dim: int = 768
    search_cache_size: int = 1000
    max_search_results: int = 50
    #: Yanıt bağlamı oluşturulurken hariç tutulacak not türleri.
    #: * `qa`  — sistemin kendi ürettiği yanıt notları; denetim izi olarak
    #:   değerlidir ama bağlama girerlerse kendi kendini besleyen bir döngü
    #:   yaratır (model kendi cevabını "kaynak" sanar).
    #: * `meta` — sistem envanteri/mimari notları; soruyla ilgisiz oldukları
    #:   halde anahtar kelime örtüşmesi yüzünden bağlama sızarlar.
    context_exclude_kinds: list = field(default_factory=lambda: ["qa", "meta"])


@dataclass
class NeuralConfig:
    """Sinir ağı ayarları."""
    embedding_dim: int = 768
    num_transformer_layers: int = 8
    num_attention_heads: int = 12
    d_ff_multiplier: int = 4
    dropout: float = 0.1
    num_initial_experts: int = 4
    max_experts: int = 12
    top_k_experts: int = 2
    memory_bank_size: int = 2048
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 0.01


@dataclass
class MultimodalConfig:
    """Çok-modlu işlem hattı ayarları."""
    image_size: tuple = (224, 224)
    audio_sample_rate: int = 16000
    max_text_length: int = 2048
    vocab_size: int = 32000
    fusion_strategy: str = "attention"  # early, late, attention


@dataclass
class LanguageConfig:
    """Dil ve çeviri ayarları."""
    default_language: str = "tr"
    supported_languages: list = field(default_factory=lambda: ["tr", "en", "de", "fr", "es"])
    use_local_model: bool = True
    api_fallback: bool = True
    translation_cache_size: int = 5000
    api_providers: list = field(default_factory=lambda: ["openai", "anthropic"])


@dataclass
class PlannerConfig:
    """Görev planlama ayarları."""
    max_task_depth: int = 10
    max_reasoning_steps: int = 50
    enable_parallel_tasks: bool = True
    checkpoint_interval: int = 5


@dataclass
class EvolutionConfig:
    """Evrim motoru ayarları."""
    learning_buffer_size: int = 10000
    min_samples_for_training: int = 100
    auto_expand_threshold: float = 0.8
    prune_threshold: float = 0.05
    expansion_strategy: str = "curiosity_driven"  # depth_first, breadth_first, curiosity_driven


# ─── YAML Güvenli Dönüşüm ───────────────────────────────────────────────────

def _yaml_safe(value: Any) -> Any:
    """
    Değeri `yaml.safe_load` ile geri okunabilecek biçime çevirir.

    Neden gerekli: `dataclasses.asdict` demetleri (`tuple`) olduğu gibi
    bırakır; PyYAML bunları `tag:yaml.org,2002:python/tuple` etiketiyle yazar
    ve `safe_load` bu etiketi **reddeder** (`ConstructorError`). Ayrıca
    Enum ve Path nesneleri de güvenli değildir. Bu fonksiyon yalnızca
    YAML/JSON'un doğal tiplerini bırakır.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_yaml_safe(item) for item in value]
    # Bilinmeyen tipler: metne çevir (YAML safe_load her zaman okur)
    return str(value)


# ─── Ana Sistem Konfigürasyonu ──────────────────────────────────────────────

@dataclass
class SystemConfig:
    """
    NeuroCore ana sistem konfigürasyonu.
    
    Tüm alt-sistem ayarlarını barındırır ve donanım profiline göre
    otomatik ayarlanabilir. Thread-safe singleton pattern kullanır.
    """
    # Genel ayarlar
    project_name: str = "NeuroCore AI"
    version: str = "0.1.0"
    base_path: str = ""
    hardware_profile: str = "high"  # HardwareProfile enum value
    device: str = "auto"  # auto, cpu, cuda, cuda:0
    log_level: str = "INFO"
    debug_mode: bool = False
    
    # Alt-sistem konfigürasyonları
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    
    # Performans ayarları
    max_batch_size: int = 64
    num_workers: int = 12
    cache_size_mb: int = 1024

    # Veri dizini (bellek, deneyim, checkpoint, günlükler)
    data_path: str = ""
    models_path: str = ""

    # ─── Singleton ──────────────────────────────────────────────────────

    _instance: Optional["SystemConfig"] = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        """Varsayılan yolları ayarla."""
        if not self.base_path:
            self.base_path = str(Path(__file__).parent.parent.parent)
        if not self.knowledge.vault_path:
            self.knowledge.vault_path = str(Path(self.base_path) / "knowledge_base")
        if not self.data_path:
            self.data_path = str(Path(self.base_path) / "data")
        if not self.models_path:
            self.models_path = str(Path(self.base_path) / "models")
        # Dizinlerin varlığını garanti et (motor ilk açılışta güvenle yazsın)
        for directory in (self.data_path, self.models_path, self.knowledge.vault_path):
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    @classmethod
    def get_instance(cls) -> "SystemConfig":
        """Thread-safe singleton erişimi."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Singleton instance'ı sıfırla (test amaçlı)."""
        cls._instance = None

    # ─── Profil Bazlı Konfigürasyon ─────────────────────────────────────

    def apply_hardware_profile(self, profile: HardwareProfile) -> None:
        """
        Donanım profiline göre tüm ayarları günceller.
        
        Args:
            profile: Uygulanacak donanım profili
        """
        self.hardware_profile = profile.value
        defaults = PROFILE_DEFAULTS[profile]
        
        # Genel ayarlar
        self.max_batch_size = defaults["max_batch_size"]
        self.num_workers = defaults["num_workers"]
        self.cache_size_mb = defaults["cache_size_mb"]
        
        # Sinir ağı ayarları
        self.neural.embedding_dim = defaults["embedding_dim"]
        self.neural.num_transformer_layers = defaults["num_transformer_layers"]
        self.neural.num_attention_heads = defaults["num_attention_heads"]
        self.neural.d_ff_multiplier = defaults["d_ff_multiplier"]
        self.neural.num_initial_experts = defaults["num_initial_experts"]
        self.neural.max_experts = defaults["max_experts"]
        self.neural.memory_bank_size = defaults["memory_bank_size"]
        
        # Bilgi motoru gömme boyutu
        self.knowledge.embedding_dim = defaults["embedding_dim"]
        
        # Düşük profillerde bellek tasarrufu
        if profile in (HardwareProfile.MINIMAL, HardwareProfile.LOW):
            self.neural.use_mixed_precision = False
            self.neural.gradient_checkpointing = True
            self.neural.dropout = 0.2

    def auto_configure(self, total_ram_gb: float, gpu_vram_gb: float = 0.0) -> HardwareProfile:
        """
        Donanım kaynaklarını algılayıp profili otomatik ayarlar.
        
        Args:
            total_ram_gb: Toplam RAM (GB)
            gpu_vram_gb: GPU VRAM (GB), yoksa 0
            
        Returns:
            Tespit edilen HardwareProfile
        """
        profile = HardwareProfile.detect(total_ram_gb, gpu_vram_gb)
        self.apply_hardware_profile(profile)
        
        # GPU yoksa CPU moduna geç
        if gpu_vram_gb <= 0:
            self.device = "cpu"
            self.hardware.enable_gpu = False
        elif self.device == "auto":
            self.device = "cuda"
            
        return profile

    # ─── Dosya İşlemleri ────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """
        Konfigürasyonu YAML dosyasına kaydeder.
        
        Args:
            path: Dosya yolu (None ise varsayılan konum)
            
        Returns:
            Kaydedilen dosya yolu
        """
        if path is None:
            path = str(Path(self.base_path) / "config.yaml")
        
        config_dict = self._to_serializable_dict()
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        return path

    @classmethod
    def load(cls, path: str) -> "SystemConfig":
        """
        YAML dosyasından konfigürasyon yükler.
        
        Args:
            path: YAML dosya yolu
            
        Returns:
            Yüklenen SystemConfig
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        config._from_dict(data)
        return config

    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Konfigürasyonu serileştirilebilir dict'e çevirir."""
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            if hasattr(value, '__dataclass_fields__'):
                result[key] = _yaml_safe(asdict(value))
            else:
                result[key] = _yaml_safe(value)
        return result

    def _from_dict(self, data: Dict[str, Any]) -> None:
        """
        Dict'ten konfigürasyonu günceller.

        İç içe dataclass'lar **değiştirilmez**; yalnızca bilinen alanları
        güncellenir. Böylece YAML'dan gelen düz sözlükler dataclass yerine
        geçip API'yi bozmaz.
        """
        for key, value in data.items():
            if key.startswith('_') or not hasattr(self, key):
                continue
            current = getattr(self, key)
            if hasattr(current, '__dataclass_fields__') and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if not hasattr(current, sub_key):
                        continue
                    existing = getattr(current, sub_key)
                    # İç içe dataclass (ör. HardwareConfig.headroom): alanları
                    # tek tek aktar, nesneyi değiştirme.
                    if hasattr(existing, '__dataclass_fields__') and isinstance(sub_value, dict):
                        for deep_key, deep_value in sub_value.items():
                            if hasattr(existing, deep_key):
                                setattr(existing, deep_key, deep_value)
                    else:
                        setattr(current, sub_key, sub_value)
            else:
                setattr(self, key, value)

    def to_json(self) -> str:
        """Konfigürasyonu JSON string'e çevirir."""
        return json.dumps(self._to_serializable_dict(), indent=2, ensure_ascii=False)

    def get_summary(self) -> Dict[str, Any]:
        """Konfigürasyon özeti döndürür."""
        return {
            "project": self.project_name,
            "version": self.version,
            "profile": self.hardware_profile,
            "device": self.device,
            "embedding_dim": self.neural.embedding_dim,
            "batch_size": self.max_batch_size,
            "workers": self.num_workers,
            "experts": f"{self.neural.num_initial_experts}/{self.neural.max_experts}",
            "transformer_layers": self.neural.num_transformer_layers,
            "cache_mb": self.cache_size_mb,
            "vault": self.knowledge.vault_path,
            "data": self.data_path,
        }
