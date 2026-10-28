"""
NeuroCore Dinamik Kaynak Tahsis Modülü
=======================================
Donanım durumuna göre model parametrelerini, batch boyutunu ve
bellek kullanımını otomatik ayarlar. Bellek baskısı altında
graceful degradation sağlar.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("hardware.allocator")


# ─── Bellek Baskı Seviyeleri ───────────────────────────────────────────────

class MemoryPressure(Enum):
    """Bellek baskı seviyeleri."""
    NORMAL = auto()    # Kaynaklar yeterli
    WARNING = auto()   # Kaynak azalıyor, önlem al
    CRITICAL = auto()  # Kaynak kritik, acil müdahale


# ─── Kaynak Bütçesi ────────────────────────────────────────────────────────

@dataclass
class ResourceBudget:
    """
    Sistem kaynak tahsis kararlarının tamamı.

    Bu yapı, mevcut donanım durumuna göre hesaplanan tüm çalışma
    parametrelerini barındırır. **Nezaket politikası** ("boştakinin %90'ı")
    uygulanmış nihai değerler buradadır; ham sistem verileri `snapshot`
    alanlarında ayrıca raporlanır.
    """
    # Cihaz
    device: str = "cpu"                    # 'cpu' veya 'cuda'

    # Model boyutlandırma
    embedding_dim: int = 768
    max_batch_size: int = 64
    num_workers: int = 12

    # Bellek (nezaket politikası uygulanmış bütçeler)
    cache_size_mb: int = 1024
    max_ram_mb: float = 0.0                # Ajanın kullanabileceği maks RAM
    max_gpu_mb: float = 0.0                # Ajanın kullanabileceği maks VRAM

    # Sinir ağı
    num_transformer_layers: int = 8
    num_attention_heads: int = 12
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False

    # Durum
    memory_pressure: MemoryPressure = MemoryPressure.NORMAL
    profile: str = "high"

    # ─── Nezaket (headroom) raporu ──────────────────────────────────────
    cpu_percent: float = 0.0               # Anlık CPU kullanımı (0..100)
    ram_percent: float = 0.0               # Anlık RAM kullanımı (0..100)
    gpu_percent: float = 0.0               # Anlık VRAM kullanımı (0..100)
    cpu_free_percent: float = 100.0        # Boşta olan CPU oranı
    ram_free_mb: float = 0.0               # Boşta olan RAM (MB)
    gpu_free_mb: float = 0.0               # Boşta olan VRAM (MB)
    #: Ajanın toplam sistemin yüzde kaçına kadar kullanabileceği
    cpu_budget_percent: float = 0.0
    ram_budget_percent: float = 0.0
    gpu_budget_percent: float = 0.0
    #: Yeni iş başlatılmalı mı? (kaynaklar doluysa False)
    should_stall: bool = False
    stall_reason: str = ""
    #: Baskı seviyesi: NORMAL | ELEVATED | HIGH | CRITICAL
    headroom_pressure: str = "NORMAL"

    def to_dict(self) -> Dict[str, Any]:
        """Bütçeyi dict olarak döndür."""
        return {
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "max_batch_size": self.max_batch_size,
            "num_workers": self.num_workers,
            "cache_size_mb": self.cache_size_mb,
            "max_ram_mb": round(self.max_ram_mb, 1),
            "max_gpu_mb": round(self.max_gpu_mb, 1),
            "num_transformer_layers": self.num_transformer_layers,
            "num_attention_heads": self.num_attention_heads,
            "use_mixed_precision": self.use_mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "memory_pressure": self.memory_pressure.name,
            "profile": self.profile,
            "headroom": {
                "cpu_percent": round(self.cpu_percent, 1),
                "ram_percent": round(self.ram_percent, 1),
                "gpu_percent": round(self.gpu_percent, 1),
                "cpu_free_percent": round(self.cpu_free_percent, 1),
                "ram_free_mb": round(self.ram_free_mb, 0),
                "gpu_free_mb": round(self.gpu_free_mb, 0),
                "cpu_budget_percent": round(self.cpu_budget_percent, 1),
                "ram_budget_percent": round(self.ram_budget_percent, 1),
                "gpu_budget_percent": round(self.gpu_budget_percent, 1),
                "pressure": self.headroom_pressure,
                "should_stall": self.should_stall,
                "stall_reason": self.stall_reason,
            },
        }

    def describe_headroom(self) -> str:
        """Nezaket durumunun insan okur tek satırlık özeti."""
        return (
            f"CPU %{self.cpu_percent:.0f} dolu → bütçe %{self.cpu_budget_percent:.0f} · "
            f"RAM %{self.ram_percent:.0f} dolu → bütçe %{self.ram_budget_percent:.0f} "
            f"({self.max_ram_mb:.0f}MB) · "
            f"VRAM %{self.gpu_percent:.0f} dolu → bütçe %{self.gpu_budget_percent:.0f} "
            f"({self.max_gpu_mb:.0f}MB) · {self.headroom_pressure}"
        )


# ─── Ana Tahsis Sınıfı ─────────────────────────────────────────────────────

class ResourceAllocator:
    """
    Dinamik kaynak tahsis yöneticisi.
    
    Donanım izleme verilerine göre model parametrelerini ve
    bellek kullanımını otomatik ayarlar.
    
    Kullanım:
        allocator = ResourceAllocator()
        allocator.initialize(config, event_bus)
        budget = allocator.allocate(monitor.get_snapshot())
        
        print(f"Device: {budget.device}")
        print(f"Embedding: {budget.embedding_dim}")
        print(f"Batch: {budget.max_batch_size}")
    """
    
    def __init__(self):
        self._config = None
        self._event_bus = None
        self._current_budget: Optional[ResourceBudget] = None
        self._lock = threading.Lock()
        self._last_pressure = MemoryPressure.NORMAL
        self._last_stall = False
        self._policy_override: Any = None
        self._last_reduction: float = 0.0

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Tahsis sistemini yapılandır."""
        self._config = config
        self._event_bus = event_bus
        try:
            logger.info(f"Kaynak tahsis sistemi başlatıldı — {self._policy().describe()}")
        except Exception:
            logger.info("Kaynak tahsis sistemi başlatıldı")

    # ─── Ana Tahsis Mantığı ─────────────────────────────────────────────

    def allocate(self, snapshot: Any = None) -> ResourceBudget:
        """
        Mevcut donanım durumuna göre kaynak bütçesi hesapla.
        
        Args:
            snapshot: SystemSnapshot (None ise varsayılanlar kullanılır)
            
        Returns:
            Hesaplanan ResourceBudget
        """
        budget = ResourceBudget()
        
        if snapshot is None:
            # Varsayılan tahsis (config bazlı)
            if self._config:
                budget.embedding_dim = self._config.neural.embedding_dim
                budget.max_batch_size = self._config.max_batch_size
                budget.num_workers = self._config.num_workers
                budget.cache_size_mb = self._config.cache_size_mb
                budget.device = self._config.device
            
            with self._lock:
                self._current_budget = budget
            return budget
        
        # ── GPU Kontrolü ────────────────────────────────────────────────
        #
        # NEZAKET: VRAM bütçesi "boştakinin %90'ı" kuralıyla hesaplanır.
        # Kullanıcı oyun/video ile GPU'yu kullanıyorsa ajan küçülür.
        policy = self._policy()
        gpu_total_mb = float(getattr(snapshot, "gpu_vram_total_mb", 0.0) or 0.0)
        gpu_free_mb = float(getattr(snapshot, "gpu_vram_free_mb", 0.0) or 0.0)
        gpu_used_percent = float(getattr(snapshot, "gpu_vram_percent", 0.0) or 0.0)

        budget.cpu_percent = float(getattr(snapshot, "cpu_percent", 0.0) or 0.0)
        budget.ram_percent = float(getattr(snapshot, "ram_percent", 0.0) or 0.0)
        budget.gpu_percent = gpu_used_percent
        budget.cpu_free_percent = max(0.0, 100.0 - budget.cpu_percent)
        budget.gpu_free_mb = gpu_free_mb

        # ── Duraklama kararı: kaynaklar doluysa yeni iş başlatma ──
        stall, stall_reason = policy.should_stall(
            cpu_percent=budget.cpu_percent,
            ram_percent=budget.ram_percent,
            gpu_percent=gpu_used_percent if snapshot.gpu_available else 0.0,
        )
        budget.should_stall = stall
        budget.stall_reason = stall_reason
        budget.headroom_pressure = policy.pressure_level(
            cpu_percent=budget.cpu_percent,
            ram_percent=budget.ram_percent,
            gpu_percent=gpu_used_percent if snapshot.gpu_available else 0.0,
        )

        if snapshot.gpu_available:
            budget.device = "cuda"
            budget.max_gpu_mb = policy.gpu_budget_mb(gpu_total_mb, gpu_free_mb)
            budget.gpu_budget_percent = (
                round(budget.max_gpu_mb / gpu_total_mb * 100.0, 1) if gpu_total_mb else 0.0
            )

            # Mimari boyutlandırma: **ajanın kullanabileceği** VRAM'e göre
            budget_gb = budget.max_gpu_mb / 1024.0
            if budget_gb >= 8:
                budget.embedding_dim = 1024
                budget.num_transformer_layers = 12
                budget.num_attention_heads = 16
                budget.max_batch_size = 128
            elif budget_gb >= 5:
                budget.embedding_dim = 768
                budget.num_transformer_layers = 8
                budget.num_attention_heads = 12
                budget.max_batch_size = 64
            elif budget_gb >= 2.5:
                budget.embedding_dim = 512
                budget.num_transformer_layers = 6
                budget.num_attention_heads = 8
                budget.max_batch_size = 32
            elif budget_gb >= 1.2:
                budget.embedding_dim = 256
                budget.num_transformer_layers = 4
                budget.num_attention_heads = 4
                budget.max_batch_size = 16
            else:
                budget.embedding_dim = 128
                budget.num_transformer_layers = 2
                budget.num_attention_heads = 2
                budget.max_batch_size = 8

            # Mixed precision (Volta ve üstü için)
            budget.use_mixed_precision = True

        else:
            budget.device = "cpu"
            budget.use_mixed_precision = False
            budget.embedding_dim = 256
            budget.num_transformer_layers = 4
            budget.num_attention_heads = 4
            budget.max_batch_size = 16
            budget.max_gpu_mb = 0.0
            budget.gpu_budget_percent = 0.0

        # ── RAM Kontrolü (nezaket) ──────────────────────────────────────
        ram_total_mb = float(getattr(snapshot, "ram_total_mb", 0.0) or 0.0)
        ram_free_mb = float(getattr(snapshot, "ram_free_mb", 0.0) or 0.0)
        budget.ram_free_mb = ram_free_mb
        budget.max_ram_mb = policy.ram_budget_mb(ram_total_mb, ram_free_mb)
        budget.ram_budget_percent = (
            round(budget.max_ram_mb / ram_total_mb * 100.0, 1) if ram_total_mb else 0.0
        )

        # ── Worker sayısı: boş CPU çekirdeklerinin nezaketli payı ───────
        logical_cores = int(getattr(snapshot, "cpu_count_logical", 0) or 1)
        budget.num_workers = policy.cpu_budget(
            logical_cores, budget.cpu_percent / 100.0
        )
        budget.cpu_budget_percent = round(budget.num_workers / logical_cores * 100.0, 1)

        # ── Önbellek: ajanın RAM bütçesinin %35'i ──────────────────────
        cache_cap = int(getattr(self._config, "cache_size_mb", 1024) or 1024) if self._config else 1024
        budget.cache_size_mb = max(32, min(int(budget.max_ram_mb * 0.35), cache_cap))

        # ── Toplu iş boyutu: bellek bütçesine göre sınırla ─────────────
        # Kaba model: her örnek ~2MB aktivasyon + gradyan (profil bağımsız üst sınır)
        if budget.max_ram_mb > 0:
            ram_cap = int(budget.max_ram_mb / 32)      # muhafazakâr: 32MB/örnek
            budget.max_batch_size = max(1, min(budget.max_batch_size, ram_cap or 1))
        if budget.device == "cuda" and budget.max_gpu_mb > 0:
            gpu_cap = int(budget.max_gpu_mb / 48)      # 48MB/örnek (KV + aktivasyon)
            budget.max_batch_size = max(1, min(budget.max_batch_size, gpu_cap or 1))

        # ── Bellek Baskısı Değerlendirmesi ──────────────────────────────
        #
        # Not: Bu, nezaket politikasından **bağımsız** ikinci bir güvenlik
        # katmanıdır. Baskı seviyeleri kullanıcının sistemini korumak için
        # ajanın kendi iş yükünü agresif biçimde küçültür.
        gpu_pressure = gpu_used_percent if snapshot.gpu_available else 0.0

        if budget.should_stall or budget.ram_percent >= 90 or gpu_pressure >= 95:
            budget.memory_pressure = MemoryPressure.CRITICAL
            budget.max_batch_size = max(1, budget.max_batch_size // 4)
            budget.cache_size_mb = max(16, budget.cache_size_mb // 4)
            budget.gradient_checkpointing = True
            budget.embedding_dim = min(budget.embedding_dim, 256)
            budget.num_transformer_layers = min(budget.num_transformer_layers, 4)
            budget.num_workers = max(1, budget.num_workers // 2)

        elif budget.ram_percent >= 75 or gpu_pressure >= 80:
            budget.memory_pressure = MemoryPressure.WARNING
            budget.max_batch_size = max(1, budget.max_batch_size // 2)
            budget.cache_size_mb = max(32, budget.cache_size_mb // 2)
            budget.gradient_checkpointing = True

        else:
            budget.memory_pressure = MemoryPressure.NORMAL

        # Baskı ayarlamalarından SONRAKİ hâl "tavan"dır: anlık durumdan
        # hesaplanan en geniş bütçe. Histerezis bu tavanı asla aşamaz.
        ceiling = ResourceBudget(**{
            key: getattr(budget, key)
            for key in ResourceBudget.__dataclass_fields__
        })

        # ── Histerezis: bütçe ani büyümesin ────────────────────────────
        budget = self._apply_hysteresis(budget, ceiling)

        # ── Profil Belirleme ────────────────────────────────────────────

        if budget.embedding_dim >= 1024:
            budget.profile = "ultra"
        elif budget.embedding_dim >= 768:
            budget.profile = "high"
        elif budget.embedding_dim >= 512:
            budget.profile = "medium"
        elif budget.embedding_dim >= 256:
            budget.profile = "low"
        else:
            budget.profile = "minimal"

        # ── Olay Yayınlama ─────────────────────────────────────────────

        if self._event_bus:
            if budget.memory_pressure != self._last_pressure:
                self._event_bus.emit(
                    "allocator.memory_pressure",
                    {"pressure": budget.memory_pressure.name, "budget": budget.to_dict()},
                    source="hardware.allocator",
                )
                self._last_pressure = budget.memory_pressure
            if budget.should_stall and not self._last_stall:
                self._event_bus.emit_warning(
                    "allocator.headroom_stall",
                    {"reason": budget.stall_reason, "budget": budget.to_dict()},
                    source="hardware.allocator",
                )
                logger.info(f"Kaynak nezaketi: yeni iş duraklatıldı — {budget.stall_reason}")
            self._last_stall = budget.should_stall

        with self._lock:
            self._current_budget = budget

        return budget

    # ─── Nezaket Yardımcıları ───────────────────────────────────────────

    def _policy(self) -> Any:
        """
        Yapılandırmadaki nezaket politikasını döndürür.

        Config yoksa güvenli varsayılanlar kullanılır (boştakinin %90'ı).
        """
        if self._policy_override is not None:
            return self._policy_override
        hardware = getattr(self._config, "hardware", None)
        policy = getattr(hardware, "headroom", None)
        if policy is not None:
            return policy
        from neurocore.core.config import HeadroomPolicy
        return HeadroomPolicy()

    def set_policy(self, policy: Any) -> None:
        """
        Nezaket politikasını çalışma zamanında değiştirir.

        Kullanıcı "daha agresif ol" veya "ben yokken tam güç kullan"
        diyebilsin diye açıktır.
        """
        self._policy_override = policy
        logger.info(f"Kaynak politikası güncellendi: {policy.describe()}")

    def _apply_hysteresis(self, budget: ResourceBudget, ceiling: ResourceBudget) -> ResourceBudget:
        """
        Bütçenin **büyümesini** yavaşlatır; küçülmesi anında olur.

        Kaynaklar bir an boşaldı diye ajanın hemen tam güce çıkması, ardından
        kullanıcı geri döndüğünde tekrar kilitlenmeye yol açar. Bu yüzden
        büyüme, `cooldown_seconds` boyunca bekletilir ve adım adım yapılır.

        ÖNEMLİ: Kademeli büyüme, anlık durumdan hesaplanan **tavanı asla
        aşmaz**. Aksi halde kaynaklar kısmen boşken ajan tavandan fazlasını
        kullanmaya çalışırdı (histerezis tavanı delerdi).

        Args:
            budget: Kısıtlanacak (döndürülecek) bütçe
            ceiling: Anlık durumdan hesaplanan üst sınır (kısıtsız bütçe)
        """
        policy = self._policy()
        previous = self._current_budget
        now = time.time()
        if previous is None:
            # İlk tahsis: cooldown'u **şimdi** başlat. Aksi halde süreç yeni
            # başladığında `now - 0` çok büyük olur, cooldown hiç uygulanmaz
            # ve bütçe ilk fırsatta tavana fırlar.
            self._last_reduction = now
            return budget

        # Küçülme tespiti: **uygulanmış** önceki değer ile anlık tavan
        # karşılaştırılır. (Ham tavanı `previous` ile karşılaştırmak yanlış
        # olurdu; tavan zaten sınırsız değerdir.)
        shrinking = (
            ceiling.max_ram_mb < previous.max_ram_mb * 0.98
            or (ceiling.max_gpu_mb > 0
                and ceiling.max_gpu_mb < previous.max_gpu_mb * 0.98)
            or ceiling.embedding_dim < previous.embedding_dim
            or ceiling.max_batch_size < previous.max_batch_size
            or ceiling.num_workers < previous.num_workers
        )
        if shrinking:
            self._last_reduction = now
            # Küçülmede anlık tavan geçerlidir (daha kötüyse ona uyulur)
            budget.max_ram_mb = min(ceiling.max_ram_mb, previous.max_ram_mb)
            budget.max_gpu_mb = min(ceiling.max_gpu_mb, previous.max_gpu_mb)
            budget.max_batch_size = min(ceiling.max_batch_size, previous.max_batch_size)
            budget.num_workers = min(ceiling.num_workers, previous.num_workers)
            budget.cache_size_mb = min(ceiling.cache_size_mb, previous.cache_size_mb)
            budget.embedding_dim = min(ceiling.embedding_dim, previous.embedding_dim)
            budget.num_transformer_layers = min(ceiling.num_transformer_layers,
                                               previous.num_transformer_layers)
            budget.num_attention_heads = min(ceiling.num_attention_heads,
                                            previous.num_attention_heads)
            return budget

        # Cooldown dolmadıysa: önceki (küçük) değerlerde kal
        if (now - self._last_reduction) < max(0.0, policy.cooldown_seconds):
            budget.embedding_dim = min(ceiling.embedding_dim, previous.embedding_dim)
            budget.num_transformer_layers = min(ceiling.num_transformer_layers,
                                               previous.num_transformer_layers)
            budget.num_attention_heads = min(ceiling.num_attention_heads,
                                            previous.num_attention_heads)
            budget.max_batch_size = min(ceiling.max_batch_size, previous.max_batch_size)
            budget.num_workers = min(ceiling.num_workers, previous.num_workers)
            budget.cache_size_mb = min(ceiling.cache_size_mb, previous.cache_size_mb)
            budget.max_ram_mb = min(ceiling.max_ram_mb, previous.max_ram_mb)
            budget.max_gpu_mb = min(ceiling.max_gpu_mb, previous.max_gpu_mb)
            return budget

        # Cooldown doldu → kademeli büyüme, ama tavanı aşmadan
        budget.max_ram_mb = min(
            ceiling.max_ram_mb, max(previous.max_ram_mb, 1.0) * 1.5 + 64.0
        )
        budget.max_gpu_mb = (
            min(ceiling.max_gpu_mb, max(previous.max_gpu_mb, 1.0) * 1.5 + 64.0)
            if ceiling.max_gpu_mb > 0 else 0.0
        )
        budget.max_batch_size = min(ceiling.max_batch_size,
                                    max(previous.max_batch_size, 1) * 2)
        budget.num_workers = min(ceiling.num_workers, max(previous.num_workers, 1) + 2)
        budget.cache_size_mb = min(ceiling.cache_size_mb,
                                   max(previous.cache_size_mb, 32) * 2)
        budget.embedding_dim = min(ceiling.embedding_dim,
                                   max(previous.embedding_dim, 128))
        return budget

    def reallocate(self, snapshot: Any) -> ResourceBudget:
        """Kaynakları yeniden tahsis et (snapshot değiştiğinde)."""
        old_budget = self._current_budget
        new_budget = self.allocate(snapshot)
        
        if old_budget and self._event_bus:
            # Önemli değişiklikleri raporla
            changes = {}
            for attr in ["embedding_dim", "max_batch_size", "device", "memory_pressure"]:
                old_val = getattr(old_budget, attr, None)
                new_val = getattr(new_budget, attr, None)
                if old_val != new_val:
                    changes[attr] = {"old": str(old_val), "new": str(new_val)}
            
            if changes:
                self._event_bus.emit(
                    "allocator.budget_updated",
                    {"changes": changes},
                    source="hardware.allocator"
                )
                logger.info(f"Kaynak bütçesi güncellendi: {changes}")
        
        return new_budget

    # ─── Yardımcı Metodlar ──────────────────────────────────────────────

    def get_budget(self) -> ResourceBudget:
        """Mevcut kaynak bütçesini döndür."""
        with self._lock:
            if self._current_budget:
                return self._current_budget
        return ResourceBudget()

    def get_device(self) -> str:
        """Kullanılacak cihazı döndür ('cpu' veya 'cuda')."""
        budget = self.get_budget()
        return budget.device

    def estimate_model_memory(self, param_count: int, precision: str = "fp32") -> float:
        """
        Model bellek kullanımını tahmin et (MB).
        
        Args:
            param_count: Model parametre sayısı
            precision: 'fp32', 'fp16', veya 'int8'
        """
        bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1}.get(precision, 4)
        # Model ağırlıkları + gradyanlar + optimizer durumu (yaklaşık 4x)
        model_mb = (param_count * bytes_per_param) / (1024 ** 2)
        total_mb = model_mb * 4  # Training için yaklaşık 4 kat
        return total_mb

    def can_fit_model(self, size_mb: float) -> bool:
        """Bu boyuttaki model mevcut kaynaklara sığar mı?"""
        budget = self.get_budget()
        if budget.device == "cuda":
            return size_mb < budget.max_gpu_mb
        return size_mb < budget.max_ram_mb

    def shutdown(self) -> None:
        """Tahsis sistemini kapat."""
        logger.info("Kaynak tahsis sistemi kapatıldı")
