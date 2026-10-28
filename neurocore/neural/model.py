"""
NeuroCore Model Paketi
=======================
Omurga + tokenizer + büyüme durumunu tek bir kalıcı varlıkta birleştirir.

Sorumluluklar:
    * Model yaşam döngüsü (oluştur / yükle / kaydet / niceleştir)
    * Tokenizer ile omurga arasındaki köprü (`embed_text`, `generate_text`)
    * Büyüme durumunun izlenmesi ve uygulanması (uzman, katman, gömme boyutu)
    * Donanım bütçesine göre otomatik yapılandırma
    * Eğitim istatistiklerinin ve nesil bilgisinin saklanması

Kayıt biçimi (dizin)::

    models/neurocore/
    ├── model.pt          (ağırlıklar — torch)
    ├── config.json       (mimari + tokenizer + büyüme durumu)
    ├── tokenizer.json    (BPE sözlüğü)
    └── growth.json       (büyüme geçmişi, istatistikler)

Tasarım kararı: model dosyası **yoksa** sistem yine çalışır. `NeuroCoreModel`
anında küçük bir omurga kurar; gömme motoru hash stratejisine düşer, bilgi
kasası okunur/yazılır, arama ve planlama tam işlevsel kalır. Model eğitildikçe
sistem kademeli olarak "daha akıllı" hâle gelir — mimari bunu bekler.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from neurocore.neural.tokenizer import BPETokenizer, create_tokenizer, DEFAULT_SPECIAL_TOKENS
from neurocore.neural.transformer import (
    HAS_TORCH,
    TransformerBackbone,
    TransformerConfig,
    config_from_system,
    estimate_backbone_memory,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.model")

#: ⚠ GERÇEK KELİME SÖZLÜĞÜ — kalite kapısının **zorunlu** ölçütü.
#:
#: Neden gerekli (tur 1 ÖLÇÜMÜ):
#:     Önceki kalite kapısı yalnız **yüzeysel** ölçütler kullanıyordu
#:     (alfabetik oran, kelime uzunluğu, tekrar). 13.574 adımlık model
#:     şu çıktıyı üretti ve kapı **onayladı**:
#:
#:         "Aimap giyeCostyAima) verirdgeç, 3 olarakAimasekiglede)"
#:         → alfabetik %96 · ort. kelime 8,0 · tekrar yok = 5/5 GEÇTİ
#:
#:     Ama 96 kelimeden yalnız **16'sı** (%16,7) gerçek kelimeydi.
#:
#: ✅ Ölçüm: bu sözlükle oran %16,7 → eşik 0,45 → **REDDEDİLİR**.
#:    Gerçek Türkçe/İngilizce metin bu eşiği rahat geçer (%70+).
#:
#: ⚠ Liste **küçük ve hızlı** tutuldu (yüzlerce kök, tam sözlük değil):
#:    amaç dilbilimsel doğruluk değil, **uydurma kelimeyi yakalamak**.
_GERCEK_KELIMELER: frozenset = frozenset({
    # ── Türkçe: yaygın kökler ve işlev sözcükleri ──
    "nasıl", "neden", "nedir", "niçin", "için", "ile", "bir", "bu", "şu",
    "ve", "veya", "ama", "ancak", "fakat", "çok", "az", "daha", "en",
    "büyük", "küçük", "yeni", "eski", "iyi", "kötü", "hızlı", "yavaş",
    "dosya", "klasör", "dizin", "sistem", "program", "yazılım", "kod",
    "hata", "sorun", "çözüm", "sebep", "nedeni", "sonuç", "durum",
    "sürücü", "sürücüyü", "sürücüsü", "güncelle", "güncelleme", "kur",
    "kurulum", "yükle", "çalış", "çalışır", "çalıştır", "yapar", "yapılır",
    "olur", "olması", "değil", "var", "yok", "gerek", "gerekli", "lazım",
    "önce", "sonra", "zaman", "süre", "yer", "yol", "adım", "adımlar",
    "kullan", "kullanılır", "ekle", "eklenir", "sil", "silinir", "aç",
    "açar", "kapat", "bak", "gör", "görünür", "al", "ver", "et", "gel",
    "git", "yap", "ol", "ise", "iken", "kadar", "gibi", "göre", "başka",
    "her", "hiç", "bazı", "tüm", "aynı", "farklı", "yeni", "tek", "iki",
    "üç", "dört", "beş", "birinci", "ikinci", "son", "ilk", "orta",
    "grafik", "kart", "kartı", "bellek", "işlemci", "ekran", "pencere",
    "malzeme", "nesne", "sahne", "model", "yüzey", "doku", "renk",
    "sayı", "sayılar", "liste", "dizi", "fonksiyon", "sınıf", "değişken",
    "python", "blender", "godot", "windows", "linux", "ayar", "ayarlar",
    "kontrol", "test", "deneme", "deneyin", "sonuç", "rapor", "kayıt",
    "veri", "veriler", "bilgi", "bilgiler", "metin", "satır", "kelime",
    "güvenli", "güvenlik", "yetki", "izin", "kullanıcı", "hesap",
    "internet", "ağ", "bağlantı", "sunucu", "istemci", "protokol",
    "şu", "şekilde", "biçimde", "olarak", "üzere", "dolayı", "rağmen",
    "ayrıca", "örneğin", "yani", "çünkü", "eğer", "zaten", "artık",
    "yükselt", "düşür", "artır", "azalt", "değiştir", "ayarla",
    "temizle", "onar", "düzelt", "düzeltme", "kaldır", "yeniden",
    "başlat", "durdur", "devam", "bekle", "hazır", "tamam", "bitti",
    # ⚠ tur 1 ÖLÇÜMÜ: gerçek Türkçe metin yanlış reddedildi —
    # bu kökler eksikti: açmak · dikişleri · işaretleyin · komutunu ·
    # böylece · yüzey · düzleştirilir · nesneyi · seçin
    "açmak", "açma", "açtıktan", "seçin", "seçmek", "seçilir",
    "dikiş", "dikişleri", "dikişi", "işaret", "işaretle", "işaretleyin",
    "komut", "komutu", "komutunu", "komutlar", "böylece", "böyle",
    "düzleştir", "düzleştirilir", "düzleştirme", "yüzeyi", "yüzeyler",
    "nesneyi", "nesneler", "yöntem", "yöntemi", "yöntemler",
    "dönüşüm", "dönüştür", "uygula", "uygulanır", "uygulama",
    "hesapla", "hesaplama", "ölçüm", "ölç", "ölçülür", "göster",
    "gösterir", "belirt", "belirtilir", "tanımla", "tanımlanır",
    "kaydet", "kaydedilir", "yükle", "yüklenir", "indir", "indirilir",
    "bağla", "bağlanır", "bağlantı", "ayır", "ayrılır", "birleştir",
    "böl", "bölünür", "kopyala", "taşı", "yeniden", "tekrar",
    "sırala", "sıralamak", "sıralanır", "grupla", "filtrele",
    "oku", "okunur", "yaz", "yazılır", "yazdır", "girdi", "çıktı",
    "sonuçları", "değeri", "değerleri", "seçenek", "seçenekler",
    "süreç", "işlem", "işlemi", "işlemler", "adımları", "aşama",
    "hatasını", "hataları", "sorunu", "sorunları", "çözümü",
    "sürüm", "sürümü", "sürümler", "özellik", "özellikleri",
    "işlev", "işlevi", "görev", "görevi", "amaç", "amacı",
    "durumu", "durumda", "koşul", "koşulu", "kural", "kuralları",
    "hız", "hızı", "boyut", "boyutu", "uzunluk", "genişlik", "yükseklik",
    "ağırlık", "basınç", "sıcaklık", "güç", "enerji", "kaynak",
    "kaynağı", "kaynakları", "hedef", "hedefi", "sonuçta", "toplam",
    "ortalama", "maksimum", "minimum", "yaklaşık", "tam", "kesin",
    "kesinlikle", "mutlaka", "belki", "olası", "olasılık", "ihtimal",
    "alternatif", "seçenek", "tercih", "öneri", "öneriler", "tavsiye",
    "uyarı", "uyarılar", "hata", "ipucu", "ipuçları", "not", "notlar",
    # ── İngilizce: yaygın sözcükler ve teknik terimler ──
    "the", "and", "for", "with", "from", "that", "this", "these",
    "those", "you", "your", "our", "their", "its", "his", "her",
    "are", "was", "were", "been", "being", "have", "has", "had",
    "will", "would", "should", "could", "can", "may", "might", "must",
    "not", "but", "or", "if", "then", "else", "when", "where", "which",
    "what", "why", "how", "who", "all", "any", "some", "each", "every",
    "more", "most", "less", "least", "other", "another", "such", "same",
    "get", "set", "use", "used", "using", "make", "made", "add", "added",
    "remove", "create", "created", "delete", "deleted", "open", "close",
    "file", "files", "folder", "path", "system", "program", "software",
    "code", "error", "errors", "driver", "drivers", "update", "updates",
    "install", "run", "runs", "work", "works", "working", "return",
    "print", "import", "class", "function", "method", "list", "array",
    "string", "number", "value", "name", "type", "data", "node",
    "scene", "model", "mesh", "material", "texture", "color", "surface",
    "gpu", "cpu", "ram", "vram", "memory", "card", "screen", "window",
    "graphics", "display", "device", "hardware", "cache", "buffer",
    "fix", "solve", "problem", "issue", "check", "test", "try", "step",
    "steps", "first", "second", "next", "last", "then", "finally",
    "time", "timeout", "delay", "crash", "crashed", "hang", "hung",
    "fail", "failed", "failure", "reason", "caused", "because",
    "should", "need", "needs", "required", "requires", "support",
    "version", "settings", "config", "option", "options", "default",
    "start", "stop", "restart", "reboot", "reset", "change", "changed",
    "increase", "decrease", "reduce", "improve", "enable", "disable",
    "before", "after", "during", "while", "until", "since", "about",
    "into", "over", "under", "between", "through", "without", "within",
    "also", "however", "therefore", "example", "note", "important",
    # ── Ortak teknik kısaltmalar ve özel adlar ──
    "dxgi", "tdr", "ddu", "api", "sdk", "ide", "cli", "gui", "os",
    "http", "https", "json", "xml", "html", "css", "sql", "uv",
    "bpy", "numpy", "pytorch", "torch", "cuda", "nvidia", "amd",
    "intel", "windows", "registry", "dxdiag", "dism", "sfc",
    # ⚠ tur 1: teknik terimler (gerçek metinde geçiyor, çöpteki karşılığı
    # rastgele harf dizisi — bu yüzden sözlükte olmaları ayırt edici)
    "unwrap", "smart", "project", "bevel", "lattice", "shader", "node",
    "script", "asset", "render", "camera", "light", "viewport", "align",
    "transform", "rotation", "scale", "translation", "origin", "cursor",
    "vertex", "vertices", "polygon", "face", "edge", "loop", "seam",
    "island", "islands", "stretch", "texture", "diffuse", "roughness",
    "normal", "bump", "displacement", "emission", "principled", "bsdf",
    "operator", "context", "module", "package", "library", "framework",
    "debug", "traceback", "exception", "syntax", "runtime", "compile",
    "process", "thread", "async", "await", "callback", "handler",
})

if HAS_TORCH:  # pragma: no cover
    import torch


# ─── Büyüme Durumu ──────────────────────────────────────────────────────────

@dataclass
class ModelGrowthState:
    """
    Modelin büyüme geçmişi ve nesli.

    Her büyüme olayı kaydedilir; bu kayıt hem geri alma hem de evrim
    katmanının karar vermesi için gereklidir.
    """
    generation: int = 0
    num_experts: int = 4
    num_layers: int = 8
    embed_dim: int = 768
    total_training_steps: int = 0
    total_tokens_seen: int = 0
    expert_growths: int = 0
    layer_growths: int = 0
    frozen_experts: List[int] = field(default_factory=list)
    pruned_experts: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def record(self, action: str, detail: Dict[str, Any]) -> None:
        """Büyüme olayını geçmişe ekler."""
        self.generation += 1
        self.updated_at = time.time()
        self.history.append({
            "generation": self.generation,
            "action": action,
            "at": self.updated_at,
            **detail,
        })
        # Geçmişi sınırla (diskte şişmesin)
        if len(self.history) > 500:
            self.history = self.history[-500:]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "generation": self.generation,
            "num_experts": self.num_experts,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "total_training_steps": self.total_training_steps,
            "total_tokens_seen": self.total_tokens_seen,
            "expert_growths": self.expert_growths,
            "layer_growths": self.layer_growths,
            "frozen_experts": list(self.frozen_experts),
            "pruned_experts": self.pruned_experts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history_tail": self.history[-20:],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelGrowthState":
        """Sözlükten yükler."""
        state = cls(
            generation=int(data.get("generation", 0) or 0),
            num_experts=int(data.get("num_experts", 4) or 4),
            num_layers=int(data.get("num_layers", 8) or 8),
            embed_dim=int(data.get("embed_dim", 768) or 768),
            total_training_steps=int(data.get("total_training_steps", 0) or 0),
            total_tokens_seen=int(data.get("total_tokens_seen", 0) or 0),
            expert_growths=int(data.get("expert_growths", 0) or 0),
            layer_growths=int(data.get("layer_growths", 0) or 0),
            frozen_experts=list(data.get("frozen_experts") or []),
            pruned_experts=int(data.get("pruned_experts", 0) or 0),
            created_at=float(data.get("created_at", time.time()) or time.time()),
            updated_at=float(data.get("updated_at", time.time()) or time.time()),
        )
        state.history = list(data.get("history") or [])
        return state

    @property
    def maturity(self) -> float:
        """
        Modelin olgunluk puanı (0..1).

        Eğitim adımları, nesil ve görülen token sayısının logaritmik
        bileşimidir; evrim katmanı "ne kadar büyüdük" sorusunu bununla yanıtlar.
        """
        step_score = min(1.0, math.log1p(self.total_training_steps) / math.log1p(20000))
        token_score = min(1.0, math.log1p(self.total_tokens_seen) / math.log1p(20_000_000))
        generation_score = min(1.0, self.generation / 50.0)
        return round(0.5 * step_score + 0.3 * token_score + 0.2 * generation_score, 4)


# ─── Ana Model Sınıfı ───────────────────────────────────────────────────────

class NeuroCoreModel:
    """
    NeuroCore'un kalıcı model varlığı.

    Kullanım:
        model = NeuroCoreModel()
        model.initialize(config, event_bus)

        model.embed_text("merhaba")                  # List[float]
        text = model.generate_text("Merhaba", max_new_tokens=32)
        stats = model.grow_experts(1)                # uzman ekle
        model.save()                                 # diske yaz
        model.load()                                 # geri yükle

    Model dosyası yoksa `initialize` küçük bir omurga kurar ve
    `is_pretrained=False` işaretler — sistem çalışmaya devam eder.
    """

    name = "neural.model"

    #: Anlamlı üretim için gereken en az eğitim adımı. Bu eşiğin altındaki bir
    #: model dil yapısını öğrenmemiştir; üretim yerine çıkarımsal/geri çağırma
    #: yolları tercih edilir (bkz. `can_generate`).
    MIN_GENERATION_STEPS = 800

    def __init__(self) -> None:
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._lock = threading.RLock()

        self.backbone: Optional[TransformerBackbone] = None
        self.tokenizer: Optional[BPETokenizer] = None
        self.model_config: Optional[TransformerConfig] = None
        self.growth = ModelGrowthState()

        self._model_dir: Optional[Path] = None
        self._is_pretrained = False
        self._quantization = "fp32"
        self._device = "cpu"

        self._counter = {
            "embeddings": 0, "generations": 0, "saves": 0, "loads": 0,
            "train_steps": 0, "growth_events": 0, "errors": 0,
            "low_quality_rejections": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None,
                   budget: Optional[Dict[str, Any]] = None,
                   model_dir: Optional[str] = None,
                   vocab_size: Optional[int] = None) -> None:
        """
        Modeli hazırlar: tokenizer'ı yükler, omurgayı kurar veya diskten alır.

        Args:
            config: SystemConfig
            event_bus: EventBus
            budget: Kaynak bütçesi (`ResourceBudget.to_dict()`)
            model_dir: Model dizini (None ise `<base_path>/models/neurocore`)
            vocab_size: Sözlük boyutu (None ise tokenizer'dan)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus

        base_path = getattr(config, "base_path", None) or os.getcwd()
        self._model_dir = Path(model_dir) if model_dir else (Path(base_path) / "models" / "neurocore")
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self._device = str(getattr(config, "device", "cpu") or "cpu")
        if budget and budget.get("device"):
            self._device = str(budget["device"])

        # 1. Tokenizer
        self.tokenizer = create_tokenizer(config, prefer_bpe=True)
        self.tokenizer.initialize(config, event_bus)

        # 2. Mimari yapılandırması
        resolved_vocab = int(vocab_size or self.tokenizer.vocab_size or 32000)
        self.model_config = config_from_system(config, vocab_size=resolved_vocab, budget=budget)

        # 3. Diskte kayıtlı model var mı?
        loaded = False
        if (self._model_dir / "config.json").exists():
            loaded = self.load(allow_missing_weights=True)

        if not loaded:
            self.backbone = TransformerBackbone(self.model_config)
            self._is_pretrained = False
            self.growth = ModelGrowthState(
                num_experts=self.model_config.num_experts,
                num_layers=self.model_config.num_layers,
                embed_dim=self.model_config.embed_dim,
            )
            logger.info(
                f"Yeni omurga oluşturuldu: {self.model_config.num_layers} katman, "
                f"{self.model_config.embed_dim}d, {self.model_config.num_experts} uzman "
                f"({self.backbone.parameter_count():,} parametre) — henüz eğitilmedi"
            )

        if HAS_TORCH and self.backbone is not None:
            try:
                self.backbone.to(self._device if self._device.startswith("cuda") else "cpu")
            except Exception as exc:
                logger.warning(f"Cihaza taşınamadı ({self._device}): {exc} — CPU'da kalınıyor")
                self._device = "cpu"

        self._initialized = True
        self._emit("neural.model_ready", {
            "pretrained": self._is_pretrained,
            "vocab_size": self.model_config.vocab_size,
            "params": self.parameter_count(),
            "device": self._device,
            "model_dir": str(self._model_dir),
            "memory": self.memory_estimate(),
        })

    def shutdown(self) -> None:
        """Modeli kapatır (idempotent)."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info(
            f"Model kapatıldı — {self._counter['embeddings']} gömme, "
            f"{self._counter['generations']} üretim, nesil {self.growth.generation}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Model istatistikleri."""
        stats: Dict[str, Any] = {
            **self._counter,
            "initialized": self._initialized,
            "pretrained": self._is_pretrained,
            "can_generate": self.can_generate,
            "device": self._device,
            "quantization": self._quantization,
            "params": self.parameter_count(),
            "active_params": self.active_parameter_count(),
            "vocab_size": self.model_config.vocab_size if self.model_config else 0,
            "growth": self.growth.to_dict(),
            "maturity": self.growth.maturity,
            "model_dir": str(self._model_dir or ""),
            "tokenizer": self.tokenizer.get_stats() if self.tokenizer else {},
        }
        if HAS_TORCH and self.backbone is not None:
            try:
                stats["routing"] = self.backbone.routing_report()
            except Exception as exc:
                stats["routing"] = {"error": str(exc)}
        return stats

    @property
    def is_ready(self) -> bool:
        """Model kullanılabilir mi?"""
        return self._initialized and self.backbone is not None

    @property
    def is_pretrained(self) -> bool:
        """Eğitilmiş ağırlıklar yüklendi mi?"""
        return self._is_pretrained

    @property
    def can_generate(self) -> bool:
        """
        Üretim güvenilir mi?

        İki koşul: model yüklenmiş/eğitilmiş olmalı **ve** yeterli eğitim adımı
        birikmiş olmalı. Yetersiz eğitilmiş bir transformer dilbilgisel olarak
        anlamsız çıktı üretir; bu durumda sistem çıkarımsal (extractive) yola
        düşer — kullanıcıya çöp metin göstermektense kaynak cümleleri
        göstermek yeğdir.

        Eşik `MIN_GENERATION_STEPS` ile ayarlanır ve `force_generation=True`
        ile geçersiz kılınabilir.
        """
        if not self.is_ready or not HAS_TORCH:
            return False
        if not self._is_pretrained:
            return False
        return self.growth.total_training_steps >= self.MIN_GENERATION_STEPS

    def generation_readiness(self) -> Dict[str, Any]:
        """Üretim hazırlığının ayrıntılı raporu (tanı için)."""
        steps = int(self.growth.total_training_steps)
        return {
            "can_generate": self.can_generate,
            "pretrained": self._is_pretrained,
            "training_steps": steps,
            "min_steps": self.MIN_GENERATION_STEPS,
            "missing_steps": max(0, self.MIN_GENERATION_STEPS - steps),
            "maturity": self.growth.maturity,
            "reason": (
                "hazır" if self.can_generate
                else ("model eğitilmedi" if not self._is_pretrained
                      else f"yetersiz eğitim ({steps}/{self.MIN_GENERATION_STEPS} adım)")
            ),
        }

    def parameter_count(self) -> int:
        """Toplam parametre sayısı."""
        if self.backbone is None:
            return 0
        try:
            return int(self.backbone.parameter_count())
        except Exception:
            return 0

    def active_parameter_count(self) -> int:
        """Token başına etkin parametre sayısı."""
        if self.backbone is None:
            return 0
        try:
            return int(self.backbone.active_parameter_count())
        except Exception:
            return 0

    def memory_estimate(self) -> Dict[str, float]:
        """Bellek tahmini (MB)."""
        if self.model_config is None:
            return {}
        try:
            return estimate_backbone_memory(self.model_config)
        except Exception:
            return {}

    # ─── Gömme ve Üretim ────────────────────────────────────────────────

    def embed_text(self, text: str, pooling: str = "mean") -> Optional[List[float]]:
        """
        Metni gömme vektörüne çevirir (omurganın gizli durumundan).

        Model eğitilmemişse bile **deterministik** bir vektör döner; ancak
        anlamsal kalite için eğitim gerekir. Eğitilmemiş modelde `None`
        döner ve gömme motoru hash stratejisine düşer — bu bilinçli bir
        tasarım tercihidir (rastgele ağırlıklı gömme, hash gömmeden daha
        kötüdür).

        Args:
            text: Girdi metni
            pooling: `mean` | `last` | `max`

        Returns:
            `embed_dim` boyutlu vektör veya kullanılamıyorsa None
        """
        if not self.is_ready:
            return None
        if not self._is_pretrained:
            # Eğitilmemiş omurganın gizli durumu anlamsızdır; hash gömmeye bırak.
            return None
        try:
            ids = self.tokenizer.encode(text, max_length=self.model_config.max_seq_len)
            if not ids:
                return None
            vector = self.backbone.embed_ids(ids, pooling=pooling)
            self._counter["embeddings"] += 1
            return vector
        except Exception as exc:
            self._counter["errors"] += 1
            logger.debug(f"Gömme başarısız: {exc}")
            return None

    def sequence_logprob(self, full_text: str, *,
                         prompt: str = "") -> Optional[float]:
        """
        Metnin **token başına ortalama log olasılığını** verir (öğrenme ölçüsü).

        Neden gerekli? Üretim kalitesi tek başına yeterli bir ölçü değildir:
        model akıcı metin üretemese bile **hangi devamın daha olası** olduğunu
        doğru sıralayabilir. Bu metot, çoktan seçmeli değerlendirmeyi mümkün
        kılar (aday başına skor üretip en yükseğini seçmek).

        `prompt` verilirse skor **yalnızca devam kısmı** üzerinden hesaplanır;
        böylece istemin uzunluğu adayları yanlış kayırmaz.

        Args:
            full_text: İstem + aday devam (birlikte kodlanır)
            prompt: Verilirse bu önek kadarı skordan düşülür

        Returns:
            Ortalama log olasılık (yüksek = daha olası) veya None
        """
        if not self.is_ready or not HAS_TORCH:
            return None
        try:
            import torch

            ids = self.tokenizer.encode(full_text,
                                        max_length=self.model_config.max_seq_len)
            if len(ids) < 2:
                return None
            prompt_len = 0
            if prompt:
                prompt_len = len(self.tokenizer.encode(
                    prompt, max_length=self.model_config.max_seq_len))
            tensor = torch.tensor([ids], dtype=torch.long)
            if hasattr(self, "ensure_device"):
                try:
                    self.ensure_device()
                except Exception:
                    pass
            device = next(self.backbone.parameters()).device
            tensor = tensor.to(device)
            self.backbone.eval()
            with torch.no_grad():
                output = self.backbone(tensor)
            logits = output.get("logits") if isinstance(output, dict) else output
            # logits[:, t] bir sonraki token'ı tahmin eder
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            hedef = tensor[:, 1:]
            secilen = log_probs[:, :-1, :].gather(
                -1, hedef.unsqueeze(-1)).squeeze(-1)
            # Yalnızca devam kısmı: prompt_len-1 indeksinden sonrası
            baslangic = max(0, prompt_len - 1)
            secilen = secilen[:, baslangic:]
            if secilen.numel() == 0:
                return None
            return float(secilen.mean().item())
        except Exception as exc:
            logger.debug(f"Dizi olasılığı hesaplanamadı: {exc}")
            return None

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        stop_on_eos: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Metin üretir.

        Args:
            prompt: Başlangıç metni
            max_new_tokens: Üretilecek maksimum token
            temperature: Örnekleme sıcaklığı (0 = greedy)
            top_k / top_p: Örnekleme kırpma
            repetition_penalty: Tekrar cezası
            stop_on_eos: `<eos>` üretilince dur
            force: Eğitim eşiğini yok say (`can_generate` denetimini atla)

        Returns:
            {"text", "prompt", "tokens", "new_tokens", "duration_ms",
             "pretrained", "stopped_by", "quality_rejected", "error"}
        """
        start = time.perf_counter()
        result: Dict[str, Any] = {
            "prompt": prompt, "text": "", "tokens": 0, "new_tokens": 0,
            "duration_ms": 0.0, "pretrained": self._is_pretrained,
            "stopped_by": "not_run", "error": "", "quality_rejected": False,
        }
        if not self.is_ready:
            result["error"] = "model_hazir_degil"
            return result
        if not HAS_TORCH:
            result["error"] = "torch_yok"
            return result
        if not self._is_pretrained:
            result["error"] = "model_egitilmedi"
            result["stopped_by"] = "untrained"
            return result
        if not force and not self.can_generate:
            readiness = self.generation_readiness()
            result["error"] = "yetersiz_egitim"
            result["stopped_by"] = "insufficient_training"
            result["readiness"] = readiness
            return result

        try:
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([ids], device=self._next_device())
            generated = self.backbone.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_id if stop_on_eos else None,
            )
            output_ids = generated[0].tolist()
            new_tokens = output_ids[len(ids):]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # ── Kalite kapısı ──
            # Yetersiz eğitilmiş modeller dilbilgisel çöp üretebilir; böyle bir
            # çıktıyı kullanıcıya göstermek yerine reddedilir ve çağıran taraf
            # çıkarımsal yola düşer.
            if text and not self._looks_like_language(text):
                self._counter["low_quality_rejections"] += 1
                result["quality_rejected"] = True
                result["error"] = "cikti_kalitesiz"
                result["stopped_by"] = "quality_gate"
                result["raw_preview"] = text[:80]
                result["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
                logger.debug(
                    f"Üretim kalite kapısında reddedildi ({len(text)} karakter): "
                    f"{text[:60]!r}"
                )
                return result

            result.update({
                "text": text,
                "tokens": len(output_ids),
                "new_tokens": len(new_tokens),
                "stopped_by": "eos" if (stop_on_eos and self.tokenizer.eos_id in new_tokens)
                              else "max_tokens",
            })
            self._counter["generations"] += 1
        except Exception as exc:
            self._counter["errors"] += 1
            result["error"] = f"uretim_hatasi: {exc}"
            logger.error(f"Üretim hatası: {exc}")

        result["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        self._emit("neural.generated", {
            "prompt": prompt[:120], "tokens": result["new_tokens"],
            "duration_ms": result["duration_ms"],
            "quality_rejected": result["quality_rejected"],
        })
        return result

    @staticmethod
    def _looks_like_language(text: str, min_score: int = 2) -> bool:
        """
        Üretilen metnin dile benzeyip benzemediğini sezgisel olarak ölçer.

        Ölçütler (en az `min_score` tanesi sağlanmalı):
            1. Alfabetik karakter oranı ≥ 0.65
            2. Ortalama kelime uzunluğu 2.5–12 aralığında
            3. Baskın olmayan karakter oranı ≥ 0.70 (aynı karakter %30'u geçmiyor)
            4. En az bir boşluk ve 3 kelime var
            5. Tekrar eden 2-gram oranı ≤ 0.35
            6. **Kelime isabeti ≥ 0.45** (ZORUNLU — puan değil, eleme)

        ⚠ 6. ölçüt tur 1'de eklendi. Ölçüm: ilk 5 ölçüt, **uydurma
        kelimelerden** oluşan metni onaylıyordu:
            "Aimap giyeCostyAima) verirdgeç, 3 olarakAimasekiglede)"
            → %96 alfabetik · ort. 8,0 harf · tekrar yok = 5/5 GEÇTİ
        ama kelimelerin yalnız %16,7'si gerçekti. 6. ölçüt bunu
        **doğrudan eler** (puan toplamaz).

        Bu bir dil modeli değildir; yalnızca bariz çöp çıktıyı elemek içindir.
        """
        if not text or len(text) < 12:
            return False

        score = 0

        # 1. Alfabetik oran
        alphabetic = sum(1 for char in text if char.isalpha() or char.isspace())
        if alphabetic / len(text) >= 0.65:
            score += 1

        # 2. Kelime uzunluğu
        words = text.split()
        if words:
            mean_length = sum(len(word) for word in words) / len(words)
            if 2.5 <= mean_length <= 12.0:
                score += 1

        # 3. Karakter çeşitliliği (tek karakter metni domine etmiyor)
        counts: Dict[str, int] = {}
        for char in text:
            counts[char] = counts.get(char, 0) + 1
        if counts:
            dominant = max(counts.values()) / len(text)
            if dominant <= 0.30:
                score += 1

        # 4. Yeterli kelime
        if len(words) >= 3:
            score += 1

        # 5. Tekrar oranı
        if len(words) >= 4:
            bigrams = [f"{words[index]} {words[index + 1]}" for index in range(len(words) - 1)]
            if bigrams:
                repeated = len(bigrams) - len(set(bigrams))
                if repeated / len(bigrams) <= 0.35:
                    score += 1

        # ══ 6. KELİME İSABETİ (tur 1: ZORUNLU ölçüt) ══════════════════
        #
        # ⚠ Neden eklendi (ÖLÇÜM):
        #   Önceki 5 ölçüt, **uydurma kelimelerden** oluşan metni
        #   onaylıyordu. Ölçüldü (13.574 adımlık model):
        #     "Aimap giyeCostyAima) verirdgeç, 3 olarakAimasekiglede)"
        #     → alfabetik %96 · ort. kelime 8,0 · tekrar yok = **5/5**
        #   Ama üretilen 96 kelimeden yalnız **16'sı** (%16,7) gerçek
        #   Türkçe/İngilizce kelimeydi.
        #
        # ✅ Ölçüldü: `_GERCEK_KELIMELER` listesi ile %16,7 → yakalanıyor.
        #    Eşik 0,45: model çıktısı %16,7 · gerçek metin %70+.
        #
        # ⚠ Bu ölçüt **zorunludur** — diğerleri puan toplarken bu
        #    doğrudan eleme yapar. Aksi halde "dilbilgisel çöp"
        #    kullanıcıya ulaşır.
        if len(words) >= 3:
            # ⚠ tur 1 ÖLÇÜMÜ — iki ayrı hata bulundu ve düzeltildi:
            #   1. 3+ harf istemek `UV`, `OS`, `AI` gibi gerçek teknik
            #      terimleri kaçırıyordu → 2+ harfe indirildi.
            #   2. **Kısa kelimeler ayırt edici değil** — çöp metinde de
            #      ("veri", "bir", "ile") gerçek kısa kelimeler var.
            #      Ölçüm: yalnız 5+ harfli kelimeler ayrım sağlıyor
            #      (çöp %25 · gerçek metin %70+).
            harf_kelime = [w for w in words
                           if re.fullmatch(r"[A-Za-zÇĞİÖŞÜçğıöşü]{5,}", w)]
            # ⚠ Uzun kelime yoksa (kısa metin, kod parçası) bu ölçüt
            # uygulanmaz — diğer 5 ölçüt karar verir.
            if harf_kelime:
                # ⚠ Kök tabanlı test (Türkçe ekleri için)
                gercek = sum(1 for w in harf_kelime if _kelime_gercek(w))
                oran = gercek / len(harf_kelime)
                # ⚠ tur 1 ÖLÇÜMÜ: 0,45 eşiği bir çöp metni kaçırdı
                # ("00dan# 1 veri ... aktarmayaıma ... satır-C,45run"
                #  → gerçek kelimeler araya serpilmişti).
                # Ölçüm: çöp %16-38 · gerçek metin %70+ → 0,55 ayırıyor.
                if oran < 0.55:
                    return False
                score += 1

        return score >= min_score

    def _next_device(self) -> Any:
        """
        Modelin **gerçekten** bulunduğu torch cihazı.

        `self._device` yapılandırmadan gelir; ancak büyüme veya yükleme
        sonrası parametreler farklı bir cihazda kalmış olabilir. Gerçek
        konum her zaman parametrelerden okunur — cihaz uyuşmazlığı
        ("Expected all tensors to be on the same device") bu yüzden oluşmaz.
        """
        if not HAS_TORCH:
            return None
        try:
            if self.backbone is not None:
                for parameter in self.backbone.parameters():
                    return parameter.device
        except Exception:
            pass
        if self._device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(self._device)
        return torch.device("cpu")

    def ensure_device(self, device: Optional[str] = None) -> str:
        """
        Modeli belirtilen cihaza taşır (varsayılan: yapılandırılmış cihaz).

        Taşıma başarısız olursa CPU'ya düşer ve bunu bildirir.

        Returns:
            Modelin bulunduğu gerçek cihaz adı
        """
        if not HAS_TORCH or self.backbone is None:
            return "cpu"
        target = device or self._device or "cpu"
        if target.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA istenmiş ama kullanılamıyor — CPU'ya düşülüyor")
            target = "cpu"
        try:
            self.backbone.to(target)
            self._device = target if not target.startswith("cuda") else "cuda"
        except Exception as exc:
            logger.warning(f"Cihaza taşınamadı ({target}): {exc} — CPU kullanılacak")
            try:
                self.backbone.to("cpu")
            except Exception:
                pass
            self._device = "cpu"
        return str(next(self.backbone.parameters()).device)

    # ─── Büyüme ─────────────────────────────────────────────────────────

    def grow_experts(self, count: int = 1, *, all_layers: bool = True,
                     dormant: bool = True) -> Dict[str, Any]:
        """
        Uzman sayısını artırır (MoE katmanlarında).

        Yeni uzmanlar **nötr** başlar: yönlendirme bias'ı çok negatif olduğu
        için hiç seçilmezler ve model çıktısı değişmez. Eğitimle veya
        `activate_experts()` çağrısıyla devreye girerler.

        Args:
            count: Katman başına eklenecek uzman sayısı
            all_layers: Tüm MoE katmanlarına uygula (False = yalnızca ilk)
            dormant: Uyuyan olarak başlat (False = hemen devreye al)

        Returns:
            {"added": toplam_eklenen, "added_per_layer": katman_başına,
             "layers": katman_sayısı, "total_experts": katman_başına_toplam,
             "generation": nesil}
        """
        if not self.is_ready or not HAS_TORCH:
            return {"added": 0, "error": "model_hazir_degil"}
        if self.backbone is None or self.model_config is None:
            return {"added": 0, "error": "omurga_yok"}

        blocks = [block for block in self.backbone.blocks if getattr(block, "is_moe", False)]
        if not blocks:
            return {"added": 0, "error": "moe_katmani_yok"}
        if not all_layers:
            blocks = blocks[:1]

        added_total = 0
        added_per_layer = 0
        for block in blocks:
            layer_added = block.feed_forward.grow(count, zero_init=dormant)
            added_total += layer_added
            added_per_layer = max(added_per_layer, layer_added)
            if layer_added and not dormant:
                # Doğrudan devreye alınacaksa bias'ı sıfırla
                block.feed_forward.activate_all_dormant(0.0)

        total = max(int(block.feed_forward.num_experts) for block in blocks)
        if added_total:
            self.growth.num_experts = total
            self.growth.expert_growths += 1
            self.growth.record("grow_experts", {
                "added_total": added_total, "added_per_layer": added_per_layer,
                "layers": len(blocks), "total_experts": total,
                "all_layers": all_layers, "dormant": dormant,
            })
            self._counter["growth_events"] += 1
            self._emit("neural.experts_grown", {
                "added": added_total, "layers": len(blocks),
                "total_experts": total, "generation": self.growth.generation,
            })
            logger.info(
                f"Uzman büyümesi: +{added_per_layer}/katman × {len(blocks)} katman "
                f"= {added_total} → {total} uzman (nesil {self.growth.generation})"
            )
        return {
            "added": added_total,
            "added_per_layer": added_per_layer,
            "layers": len(blocks),
            "total_experts": total,
            "generation": self.growth.generation,
        }

    def activate_experts(self, bias: float = 0.0) -> int:
        """Uyuyan uzmanları devreye alır (büyüme sonrası eğitim için)."""
        if not self.is_ready or not HAS_TORCH:
            return 0
        activated = 0
        for block in self.backbone.blocks:
            if getattr(block, "is_moe", False):
                activated += block.feed_forward.activate_all_dormant(bias)
        if activated:
            self.growth.record("activate_experts", {"count": activated, "bias": bias})
            self._emit("neural.experts_activated", {"count": activated})
        return activated

    def warmup_experts(self, step: float = 2.0) -> int:
        """Uyuyan uzmanların bias'ını kademeli yükseltir."""
        if not self.is_ready or not HAS_TORCH:
            return 0
        changed = 0
        for block in self.backbone.blocks:
            if getattr(block, "is_moe", False):
                changed += block.feed_forward.decay_router_bias(step)
        return changed

    def freeze_expert(self, layer_index: int, expert_index: int,
                      reason: str = "") -> bool:
        """
        Belirli bir uzmanı dondurur.

        Args:
            layer_index: Omurgadaki blok indeksi
            expert_index: Blok içindeki uzman indeksi
            reason: Gerekçe (günlüğe yazılır)
        """
        if not self.is_ready or not HAS_TORCH or self.backbone is None:
            return False
        try:
            block = self.backbone.blocks[layer_index]
        except (IndexError, TypeError):
            return False
        if not getattr(block, "is_moe", False):
            return False
        ok = block.feed_forward.freeze_expert(expert_index, reason)
        if ok:
            key = layer_index * 100 + expert_index
            if key not in self.growth.frozen_experts:
                self.growth.frozen_experts.append(key)
            self.growth.record("freeze_expert", {
                "layer": layer_index, "expert": expert_index, "reason": reason,
            })
            self._emit("neural.expert_frozen", {
                "layer": layer_index, "expert": expert_index, "reason": reason,
            })
        return ok

    def unfreeze_expert(self, layer_index: int, expert_index: int) -> bool:
        """Donmuş uzmanı yeniden etkinleştirir."""
        if not self.is_ready or not HAS_TORCH or self.backbone is None:
            return False
        try:
            block = self.backbone.blocks[layer_index]
        except (IndexError, TypeError):
            return False
        if not getattr(block, "is_moe", False):
            return False
        ok = block.feed_forward.unfreeze_expert(expert_index)
        if ok:
            key = layer_index * 100 + expert_index
            if key in self.growth.frozen_experts:
                self.growth.frozen_experts.remove(key)
            self.growth.record("unfreeze_expert", {
                "layer": layer_index, "expert": expert_index,
            })
        return ok

    def grow_layers(self, count: int = 1, *, neutral: bool = True) -> Dict[str, Any]:
        """
        Omurgaya yeni katman ekler (nötr başlangıçla).

        Args:
            count: Eklenecek katman sayısı
            neutral: Çıkış projeksiyonlarını sıfırla (model çıktısı değişmez)

        Returns:
            {"added": int, "total_layers": int, "generation": int}
        """
        if not self.is_ready or self.backbone is None:
            return {"added": 0, "error": "model_hazir_degil"}
        added = self.backbone.expand_layers(count, zero_init=neutral)
        if added:
            self.growth.num_layers = len(self.backbone.blocks)
            self.growth.layer_growths += 1
            self.growth.record("grow_layers", {
                "added": added, "total_layers": self.growth.num_layers,
                "neutral": neutral,
            })
            self._counter["growth_events"] += 1
            self._emit("neural.layers_grown", {
                "added": added, "total_layers": self.growth.num_layers,
                "generation": self.growth.generation,
            })
            logger.info(
                f"Katman büyümesi: +{added} → {self.growth.num_layers} katman "
                f"(nesil {self.growth.generation})"
            )
        return {"added": added, "total_layers": self.growth.num_layers,
                "generation": self.growth.generation}

    def routing_report(self) -> Dict[str, Any]:
        """Yönlendirme raporu (evrim katmanının büyüme kararı için)."""
        if not self.is_ready or self.backbone is None:
            return {"available": False}
        try:
            report = self.backbone.routing_report()
            report["available"] = True
            report["generation"] = self.growth.generation
            report["maturity"] = self.growth.maturity
            return report
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def record_training_step(self, tokens: int = 0, steps: int = 1) -> None:
        """Eğitim ilerlemesini kaydeder (olgunluk hesabı için)."""
        self.growth.total_training_steps += int(steps)
        self.growth.total_tokens_seen += int(tokens)
        self._counter["train_steps"] += int(steps)

    # ─── Niceleme ───────────────────────────────────────────────────────

    def quantize(self, mode: str = "fp16") -> Dict[str, Any]:
        """
        Modeli niceleştirir (bellek tasarrufu).

        Args:
            mode: `fp32` | `fp16` | `bf16` | `int8` | `dynamic_int8`

        Returns:
            {"mode", "before_mb", "after_mb", "saved_percent"}
        """
        if not self.is_ready or not HAS_TORCH or self.backbone is None:
            return {"error": "model_hazir_degil"}

        before = self.parameter_count() * 4 / (1024 ** 2)
        try:
            if mode == "fp32":
                self.backbone.to(torch.float32)
                self._quantization = "fp32"
            elif mode == "fp16":
                if self._device.startswith("cuda"):
                    self.backbone.to(torch.float16)
                else:
                    self.backbone.to(torch.bfloat16)
                    mode = "bf16"
                self._quantization = mode
            elif mode == "bf16":
                self.backbone.to(torch.bfloat16)
                self._quantization = "bf16"
            elif mode in ("int8", "dynamic_int8"):
                if not self._device.startswith("cuda"):
                    return {"error": "int8_niceleme_cpu_desteklenmiyor",
                            "hint": "dinamik niceleme için torch.quantization kullanın"}
                self.backbone.to(torch.float16)
                self._quantization = "int8"
            else:
                return {"error": f"bilinmeyen_mod: {mode}"}
        except Exception as exc:
            self._counter["errors"] += 1
            logger.error(f"Niceleme hatası ({mode}): {exc}")
            return {"error": str(exc)}

        bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(self._quantization, 4)
        after = self.parameter_count() * bytes_per_param / (1024 ** 2)
        result = {
            "mode": self._quantization,
            "before_mb": round(before, 2),
            "after_mb": round(after, 2),
            "saved_percent": round((1 - after / before) * 100, 1) if before else 0.0,
        }
        self.growth.record("quantize", result)
        self._emit("neural.quantized", result)
        logger.info(
            f"Niceleme: {result['mode']} — {result['before_mb']}MB → "
            f"{result['after_mb']}MB (%{result['saved_percent']} tasarruf)"
        )
        return result

    # ─── Kalıcılık ──────────────────────────────────────────────────────

    def save(self, model_dir: Optional[str] = None, save_weights: bool = True) -> Dict[str, Any]:
        """
        Modeli diske kaydeder.

        Üç dosya yazılır: `config.json` (mimari), `tokenizer.json`,
        `growth.json` (büyüme durumu) ve isteğe bağlı `model.pt` (ağırlıklar).
        Yazma atomiktir.

        Args:
            model_dir: Hedef dizin (None = başlangıçtaki dizin)
            save_weights: Ağırlıkları da yaz (büyük dosya)

        Returns:
            {"dir", "files", "bytes", "params"}
        """
        if not self.is_ready:
            return {"error": "model_hazir_degil"}
        target_dir = Path(model_dir) if model_dir else self._model_dir
        if target_dir is None:
            return {"error": "dizin_yok"}
        target_dir.mkdir(parents=True, exist_ok=True)

        written: List[str] = []
        total_bytes = 0

        # 1. Mimari
        config_payload = {
            "version": 2,
            "created_at": self.growth.created_at,
            "pretrained": self._is_pretrained,
            "quantization": self._quantization,
            "device": self._device,
            "model": self.model_config.to_dict() if self.model_config else {},
            "growth": self.growth.to_dict(),
        }
        path = target_dir / "config.json"
        self._atomic_json(path, config_payload)
        written.append(str(path))
        total_bytes += path.stat().st_size

        # 2. Tokenizer
        if self.tokenizer is not None:
            token_path = self.tokenizer.save(str(target_dir / "tokenizer.json"))
            written.append(token_path)
            total_bytes += Path(token_path).stat().st_size

        # 3. Büyüme durumu (ayrı dosya — hızlı okunur)
        growth_path = target_dir / "growth.json"
        self._atomic_json(growth_path, self.growth.to_dict())
        written.append(str(growth_path))
        total_bytes += growth_path.stat().st_size

        # 4. Ağırlıklar
        if save_weights and HAS_TORCH and self.backbone is not None:
            weights_path = target_dir / "model.pt"
            temp = weights_path.with_name(weights_path.name + ".tmp")
            try:
                torch.save({
                    "model_state": self.backbone.state_dict(),
                    "growth": self.growth.to_dict(),
                    "quantization": self._quantization,
                }, temp)
                os.replace(temp, weights_path)
                written.append(str(weights_path))
                total_bytes += weights_path.stat().st_size
            except Exception as exc:
                self._counter["errors"] += 1
                logger.error(f"Ağırlıklar kaydedilemedi: {exc}")
                if temp.exists():
                    try:
                        temp.unlink()
                    except OSError:
                        pass

        self._counter["saves"] += 1
        result = {
            "dir": str(target_dir), "files": written,
            "bytes": total_bytes, "params": self.parameter_count(),
            "megabytes": round(total_bytes / (1024 ** 2), 3),
        }
        self._emit("neural.model_saved", {
            "dir": str(target_dir), "megabytes": result["megabytes"],
            "params": result["params"],
        })
        logger.info(
            f"Model kaydedildi: {target_dir} ({result['megabytes']} MB, "
            f"{result['params']:,} parametre)"
        )
        return result

    def load(self, model_dir: Optional[str] = None,
             allow_missing_weights: bool = False) -> bool:
        """
        Modeli diskten yükler.

        Args:
            model_dir: Kaynak dizin (None = başlangıçtaki dizin)
            allow_missing_weights: Ağırlık dosyası yoksa mimariyi kur ama
                `is_pretrained=False` bırak

        Returns:
            Yükleme başarılıysa True
        """
        source_dir = Path(model_dir) if model_dir else self._model_dir
        if source_dir is None or not source_dir.exists():
            return False

        config_path = source_dir / "config.json"
        if not config_path.exists():
            return False

        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"Model yapılandırması okunamadı ({config_path}): {exc}")
            return False

        model_data = payload.get("model") or {}
        if not model_data:
            logger.error("Model yapılandırması boş — yükleme iptal")
            return False

        # Tokenizer
        token_path = source_dir / "tokenizer.json"
        if token_path.exists():
            self.tokenizer = BPETokenizer.load(str(token_path))
            self.tokenizer.initialize(self._config, self._event_bus)

        # Mimari
        try:
            allowed = set(TransformerConfig.__dataclass_fields__.keys())
            clean = {key: value for key, value in model_data.items() if key in allowed}
            self.model_config = TransformerConfig(**clean)
        except Exception as exc:
            logger.error(f"Mimari yapılandırması geçersiz: {exc}")
            return False

        self.backbone = TransformerBackbone(self.model_config)

        # Büyüme durumu
        growth_data = payload.get("growth") or {}
        growth_path = source_dir / "growth.json"
        if growth_path.exists():
            try:
                growth_data = json.loads(growth_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        self.growth = ModelGrowthState.from_dict(growth_data)

        # Ağırlıklar
        weights_path = source_dir / "model.pt"
        if weights_path.exists() and HAS_TORCH:
            try:
                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
                state = checkpoint.get("model_state") or checkpoint
                state = self._filter_compatible_state(state)
                missing, unexpected = self.backbone.load_state_dict(state, strict=False)
                if missing:
                    logger.warning(
                        f"{len(missing)} parametre eksik yüklendi (mimari büyümüş olabilir)"
                    )
                if unexpected:
                    logger.warning(f"{len(unexpected)} beklenmeyen parametre yok sayıldı")

                # ── Sağlık denetimi ──
                # Bozuk/NaN ağırlıklar sessizce yüklenirse model her girdi için
                # NaN üretir ve eğitim hiç ilerlemez. Böyle bir checkpoint
                # reddedilir; model yeni başlatılmış hâliyle kalır.
                if not self._weights_are_finite():
                    logger.error(
                        f"Checkpoint ağırlıkları geçersiz (NaN/Inf içeriyor): "
                        f"{weights_path} — yükleme reddedildi, model yeniden "
                        "başlatılmış hâlinde kalacak"
                    )
                    self._is_pretrained = False
                    self.growth = ModelGrowthState(
                        num_experts=self.model_config.num_experts,
                        num_layers=self.model_config.num_layers,
                        embed_dim=self.model_config.embed_dim,
                    )
                    self._counter["errors"] += 1
                    return False

                # Config'te `pretrained=False` yazıyorsa ağırlıklar eğitilmemiş
                # demektir; bu durumda üretim/gömme için "eğitilmiş" sayılmaz.
                self._is_pretrained = bool(payload.get("pretrained", True))
                self._quantization = str(checkpoint.get("quantization", "fp32"))
                # Ağırlıklar yüklendi ama yeterince eğitilmemişse üretim
                # kalitesi güvenilir değildir; bu durum `can_generate` ile
                # denetlenir ve çağıran taraf çıkarımsal yola düşer.
                if self.growth.total_training_steps < self.MIN_GENERATION_STEPS:
                    logger.info(
                        f"Model yüklendi ama yalnızca "
                        f"{self.growth.total_training_steps} eğitim adımı var "
                        f"(eşik {self.MIN_GENERATION_STEPS}) — üretim yerine "
                        "çıkarımsal yanıt kullanılacak"
                    )
                logger.info(
                    f"Model yüklendi: {source_dir} — nesil {self.growth.generation}, "
                    f"{self.growth.total_training_steps} eğitim adımı, "
                    f"olgunluk {self.growth.maturity:.2f}"
                )
            except Exception as exc:
                self._counter["errors"] += 1
                logger.error(f"Ağırlıklar yüklenemedi ({weights_path}): {exc}")
                if not allow_missing_weights:
                    return False
                self._is_pretrained = False
        else:
            if not allow_missing_weights:
                logger.warning("Ağırlık dosyası yok — model eğitilmemiş sayılıyor")
            self._is_pretrained = False

        self._counter["loads"] += 1
        self._emit("neural.model_loaded", {
            "dir": str(source_dir), "pretrained": self._is_pretrained,
            "generation": self.growth.generation,
        })
        return True

    def _atomic_json(self, path: Path, payload: Dict[str, Any]) -> None:
        """JSON'u atomik olarak yazar."""
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(temp, path)

    def _weights_are_finite(self) -> bool:
        """
        Yüklü ağırlıkların tamamı sonlu mu?

        NaN/Inf içeren bir checkpoint, modeli sessizce kullanılamaz hâle
        getirir (her çıktı NaN olur). Yükleme sonrası bu denetim yapılır.
        """
        if not HAS_TORCH or self.backbone is None:
            return True
        try:
            for name, parameter in self.backbone.named_parameters():
                if not torch.isfinite(parameter).all():
                    logger.debug(f"Geçersiz ağırlık: {name}")
                    return False
        except Exception as exc:
            logger.debug(f"Ağırlık denetimi yapılamadı: {exc}")
            return True
        return True

    def _filter_compatible_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checkpoint'teki ağırlıkları mevcut mimariye uygun olanlarla sınırlar.

        Mimari büyüdüğünde (yeni uzman/katman, değişen sözlük) kaydedilmiş
        tensörlerin şekli uyuşmaz. `load_state_dict(strict=False)` şekil
        uyuşmazlığını **hata olarak** bildirir; burada uyumsuz anahtarlar
        sessizce ayıklanır — uyumlu ağırlıklar korunur, uyumsuzlar mevcut
        (yeni başlatılmış) hâllerinde kalır. Böylece büyüme sonrası model
        yüklenebilir kalır.
        """
        if not HAS_TORCH or self.backbone is None:
            return state
        try:
            current = self.backbone.state_dict()
        except Exception:
            return state

        compatible: Dict[str, Any] = {}
        skipped: List[str] = []
        for key, value in (state or {}).items():
            target = current.get(key)
            if target is None:
                skipped.append(key)
                continue
            try:
                if tuple(target.shape) != tuple(getattr(value, "shape", ())):
                    skipped.append(key)
                    continue
            except Exception:
                skipped.append(key)
                continue
            compatible[key] = value

        if skipped:
            logger.info(
                f"{len(skipped)} ağırlık mimari uyuşmazlığı nedeniyle atlandı "
                f"(ör. {', '.join(skipped[:3])})"
            )
        return compatible

    def backup(self, suffix: str = "bak") -> Optional[str]:
        """
        Model dizinini yedekler (büyüme öncesi güvenlik ağı).

        Returns:
            Yedek dizin yolu veya başarısızsa None
        """
        if self._model_dir is None or not self._model_dir.exists():
            return None
        target = self._model_dir.with_name(f"{self._model_dir.name}_{suffix}_{int(time.time())}")
        try:
            shutil.copytree(self._model_dir, target)
        except OSError as exc:
            logger.error(f"Yedekleme başarısız: {exc}")
            return None
        logger.info(f"Model yedeklendi: {target}")
        return str(target)

    # ─── Sağlık ─────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Model sağlık raporu."""
        issues: List[str] = []
        if not HAS_TORCH:
            issues.append("torch kurulu değil — eğitim ve üretim devre dışı")
        if not self._initialized:
            issues.append("model başlatılmadı")
        if self._initialized and not self._is_pretrained:
            issues.append("model eğitilmedi — gömme hash stratejisine düşüyor")
        if self.model_config is not None and self.model_config.vocab_size <= 300:
            issues.append("sözlük çok küçük — tokenizer eğitilmeli (train_tokenizer)")
        routing = self.routing_report()
        if routing.get("available") and routing.get("expert_entropy", 0) > 0.9:
            issues.append("yönlendirme entropisi yüksek — uzman büyümesi önerilir")

        return {
            "ok": self._initialized and HAS_TORCH,
            "pretrained": self._is_pretrained,
            "params": self.parameter_count(),
            "maturity": self.growth.maturity,
            "generation": self.growth.generation,
            "device": self._device,
            "quantization": self._quantization,
            "issues": issues,
            "memory": self.memory_estimate(),
        }

    def describe(self) -> str:
        """İnsan okur tek paragraf özet."""
        if not self.is_ready:
            return "Model hazır değil."
        config = self.model_config
        return (
            f"NeuroCore omurgası: {config.num_layers} katman × {config.embed_dim}d, "
            f"{config.num_heads} dikkat başı, {self.growth.num_experts} uzman "
            f"({config.top_k_experts} aktif/token), sözlük {config.vocab_size:,}. "
            f"Toplam {self.parameter_count():,} parametre, token başına "
            f"{self.active_parameter_count():,} etkin. "
            f"Cihaz: {self._device}, niceleme: {self._quantization}. "
            f"Nesil {self.growth.generation}, olgunluk {self.growth.maturity:.2f}, "
            f"{'eğitilmiş' if self._is_pretrained else 'EĞİTİLMEMİŞ'}."
        )

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<NeuroCoreModel ready={self.is_ready} pretrained={self._is_pretrained} "
            f"params={self.parameter_count()} gen={self.growth.generation}>"
        )


__all__ = [
    "NeuroCoreModel",
    "ModelGrowthState",
]

#: ⚠ Kök uzunluğu — Türkçe eklemeli dil için 4 karakter yeterli.
#:    "listeyi" → "list" (liste kökü) · "sürücüsünün" → "sürü"
#:    "güncelleyin" → "günc" · "işaretleyin" → "işar"
_KOK_UZUNLUK = 4


def _gercek_kok_kumesi() -> frozenset:
    """
    Sözlükten **kök kümesi** üretir (ilk `_KOK_UZUNLUK` karakter).

    ⚠ Neden gerekli (tur 1 ÖLÇÜMÜ):
        Tam-kelime eşleştirmesi Türkçe **ekleri** kaçırıyordu —
        `listeyi`, `sıralamak`, `seçin`, `işaretleyin` gerçek Türkçe
        olduğu hâlde REDDEDİLİYORDU (8/11 → yanlış ret 2).

        ✅ Kök eşleştirmesi: `liste` → `list` kökü → `listeyi` eşleşir.
    """
    return frozenset(
        k[:_KOK_UZUNLUK] for k in _GERCEK_KELIMELER
        if len(k) >= _KOK_UZUNLUK
    )


#: ⚠ Modül yüklenirken bir kez hesaplanır (O(1) üyelik testi)
_GERCEK_KOKLER: frozenset = _gercek_kok_kumesi()


def _kelime_gercek(kelime: str) -> bool:
    """
    Kelime gerçek mi? **Kök tabanlı** test.

    ⚠ Sıra:
        1. Tam eşleşme (sözlükte var mı)
        2. Kök eşleşmesi (ilk 4 karakter sözlük köklerinde var mı)
        3. Kısa kelimeler (3 harf) yalnız tam eşleşmeyle
    """
    k = kelime.lower()
    if k in _GERCEK_KELIMELER:
        return True
    # ⚠ Kısa kelimeler (≤ kök uzunluğu) yalnız TAM eşleşmeyle geçer;
    #    2 harfli kök eşleşmesi çok gevşek olurdu ("os" → "osur").
    if len(k) <= _KOK_UZUNLUK:
        return False
    return k[:_KOK_UZUNLUK] in _GERCEK_KOKLER
