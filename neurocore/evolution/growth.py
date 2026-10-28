"""
NeuroCore Büyüme Denetleyicisi (L8 · Evolution)
================================================
Modelin **ne zaman ve nasıl büyüyeceğine** karar verir: uzman ekle, uzman
dondur/buda, embedding genişlet, yeniden eğit ya da hiçbir şey yapma.

Karar kuralları (`decide`):

    Uzman ekle   : yönlendirme entropisi ≥ auto_expand_threshold
                   VE num_experts < max_experts
                   VE tampon ≥ min_samples_for_training
                   VE ortalama kalite zaten yüksek değil (≤ 0.85)
    Dondur/Buda  : bir uzmanın kullanım payı < prune_threshold VE uzman > 3
                   (önce FREEZE; kullanım üst üste ≥ 3 kayıtta ~0 ise PRUNE)
    Yeniden eğit : tamponda `total_training_steps × batch_size`'dan fazla yeni
                   örnek var VE ortalama kalite < 0.7
    Embedding    : yalnızca donanım bütçesi izin veriyorsa
                   (embedding_dim² × 4 bayt × 12) VE belirsizlik > 0.6

Durum `<data_path>/growth_state.json` altında kalıcıdır ve `initialize()`
sırasında geri yüklenir. Bu modül yalnızca stdlib + numpy kullanır.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("evolution.growth")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: Büyüme durumu dosyası
STATE_FILE = "growth_state.json"

#: `data_path` verilmediğinde kullanılacak göreli dizin
DEFAULT_DATA_PATH = "data/evolution"

#: Büyüme alt dizini (kök veri yolu verildiğinde dosya buraya yazılır)
GROWTH_DIR = "growth"

#: Geçmiş kayıt üst sınırı
MAX_HISTORY = 200

#: Embedding genişletme için gereken bayt çarpanı (embedding_dim² × 4 × 12)
EMBED_BYTES_FACTOR = 12

#: Embedding genişletme için gereken ek bellek payı
EMBED_MEMORY_MARGIN = 1.25

#: Uzman ekleme için uzman başına asgari örnek
MIN_SAMPLES_PER_EXPERT = 40

#: Ortalama kalite bu değerin üzerindeyse uzman eklemeye gerek yok
QUALITY_SATURATION = 0.85

#: Yeniden eğitim için kalite eşiği
RETRAIN_QUALITY = 0.7

#: Yeni uzmanlar için başlangıç sayısı
DEFAULT_NUM_EXPERTS = 4

#: Uzman dondurma kararını tetikleyen ardışık "sıfır kullanım" kaydı
ZERO_USAGE_STREAK_FOR_PRUNE = 3


# ─── Yardımcılar ───────────────────────────────────────────────────────────

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Değeri [low, high] aralığına sıkıştır."""
    return max(low, min(high, float(value)))


# ─── Büyüme Eylemleri ──────────────────────────────────────────────────────

class GrowthAction(str, Enum):
    """Modelin büyüme/küçülme eylemleri."""
    EXPAND_EXPERTS = "expand_experts"       # Yeni uzman ekle
    FREEZE_EXPERT = "freeze_expert"         # Uzmanı dondur (silme, arşivle)
    PRUNE_EXPERT = "prune_expert"           # Uzmanı buda (arşive taşı)
    EXPAND_EMBEDDING = "expand_embedding"   # Gömme boyutunu büyüt
    ADD_LAYER = "add_layer"                 # Yeni transformer katmanı
    RETRAIN = "retrain"                     # Yeniden eğit
    MERGE_NOTES = "merge_notes"             # Bilgi tarafında not birleştir
    NOOP = "noop"                           # Hiçbir şey yapma


# ─── Büyüme Kararı ─────────────────────────────────────────────────────────

@dataclass
class GrowthDecision:
    """
    Tek bir büyüme kararı.

    Attributes:
        action: Seçilen eylem
        reason: Türkçe, insan okunur gerekçe
        magnitude: Eylemin büyüklüğü (ör. eklenecek uzman sayısı)
        details: Eyleme özgü ek bilgiler (ör. `expert_index`)
        confidence: Karar güveni (0..1)
        dry_run: Yalnızca öneri mi, uygulanabilir mi
    """

    action: GrowthAction
    reason: str
    magnitude: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dict."""
        return {
            "action": self.action.value if isinstance(self.action, GrowthAction) else str(self.action),
            "reason": self.reason,
            "magnitude": round(float(self.magnitude), 4),
            "details": dict(self.details),
            "confidence": round(float(self.confidence), 4),
            "dry_run": bool(self.dry_run),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GrowthDecision":
        """Dict'ten karar kurar (bilinmeyen eylem → NOOP)."""
        data = d if isinstance(d, dict) else {}
        raw_action = str(data.get("action") or GrowthAction.NOOP.value)
        try:
            action = GrowthAction(raw_action)
        except ValueError:
            action = GrowthAction.NOOP
        return cls(
            action=action,
            reason=str(data.get("reason") or ""),
            magnitude=float(data.get("magnitude", 0.0) or 0.0),
            details=dict(data.get("details") or {}),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            dry_run=bool(data.get("dry_run", False)),
        )


# ─── Büyüme Durumu ─────────────────────────────────────────────────────────

@dataclass
class GrowthState:
    """
    Modelin büyüme defteri — kalıcı durum.

    Attributes:
        num_experts: Mevcut uzman sayısı
        max_experts: Donanım bütçesine göre izin verilen en fazla uzman
        num_layers: Transformer katman sayısı
        embedding_dim: Gömme boyutu
        generation: Kaçıncı nesil (her uygulanan büyüme kararı +1)
        total_training_steps: Toplam eğitim adımı
        frozen_experts: Dondurulmuş uzman indeksleri
        history: Son kararların kaydı
    """

    num_experts: int = DEFAULT_NUM_EXPERTS
    max_experts: int = 12
    num_layers: int = 8
    embedding_dim: int = 768
    generation: int = 0
    total_training_steps: int = 0
    frozen_experts: List[int] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dict."""
        return {
            "num_experts": int(self.num_experts),
            "max_experts": int(self.max_experts),
            "num_layers": int(self.num_layers),
            "embedding_dim": int(self.embedding_dim),
            "generation": int(self.generation),
            "total_training_steps": int(self.total_training_steps),
            "frozen_experts": [int(i) for i in self.frozen_experts],
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GrowthState":
        """Dict'ten durum kurar; bozuk alanlar varsayılana düşer."""
        data = d if isinstance(d, dict) else {}

        def _int(key: str, default: int, minimum: int = 0) -> int:
            try:
                return max(minimum, int(data.get(key, default)))
            except (TypeError, ValueError):
                return default

        frozen_raw = data.get("frozen_experts") or []
        frozen: List[int] = []
        if isinstance(frozen_raw, (list, tuple)):
            for value in frozen_raw:
                try:
                    frozen.append(int(value))
                except (TypeError, ValueError):
                    continue

        history_raw = data.get("history") or []
        history = [h for h in history_raw if isinstance(h, dict)] if isinstance(history_raw, list) else []

        return cls(
            num_experts=_int("num_experts", DEFAULT_NUM_EXPERTS, 1),
            max_experts=_int("max_experts", 12, 1),
            num_layers=_int("num_layers", 8, 1),
            embedding_dim=_int("embedding_dim", 768, 16),
            generation=_int("generation", 0),
            total_training_steps=_int("total_training_steps", 0),
            frozen_experts=frozen,
            history=history,
        )


# ─── Büyüme Denetleyicisi ──────────────────────────────────────────────────

class GrowthController:
    """
    Model büyümesinin tek karar mercii.

    Kullanım:
        gc = GrowthController()
        gc.initialize(config, event_bus)
        gc.update_from_budget(budget.to_dict())
        decision = gc.decide(metrics, buffer_stats)
        if decision.action is not GrowthAction.NOOP:
            gc.apply(decision, apply_fn=lambda d: model.grow(d))
    """

    name = "evolution.growth"

    def __init__(self) -> None:
        self._config: Any = None
        self._event_bus: Any = None

        # Ayarlar (initialize ile doldurulur)
        self._data_path: str = ""
        self._state_file: Optional[Path] = None
        self._min_samples: int = 100
        self._auto_expand_threshold: float = 0.8
        self._prune_threshold: float = 0.05

        # Durum
        self._state_obj: GrowthState = GrowthState()
        self._budget: Dict[str, Any] = {}
        self._zero_usage_streak: Dict[int, int] = {}
        self._counters: Dict[str, float] = {
            "decisions": 0.0,
            "applied": 0.0,
            "failed": 0.0,
        }
        self._last_decision: Dict[str, Any] = {}
        self._initialized: bool = False

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Denetleyiciyi bağla, ayarları oku ve durumu diskten geri yükle.

        İdempotenttir.
        """
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus

        evolution = getattr(config, "evolution", None)
        neural = getattr(config, "neural", None)

        try:
            self._min_samples = max(1, int(getattr(evolution, "min_samples_for_training", 100)))
        except (TypeError, ValueError):
            self._min_samples = 100
        try:
            self._auto_expand_threshold = float(getattr(evolution, "auto_expand_threshold", 0.8))
        except (TypeError, ValueError):
            self._auto_expand_threshold = 0.8
        try:
            self._prune_threshold = float(getattr(evolution, "prune_threshold", 0.05))
        except (TypeError, ValueError):
            self._prune_threshold = 0.05

        # Yollar
        self._data_path = self._resolve_data_path(config)
        directory = self._resolve_absolute(self._data_path)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Büyüme dizini oluşturulamadı: {exc}")
        self._state_file = directory / STATE_FILE

        # Başlangıç durumu (konfigürasyondan)
        initial = GrowthState()
        try:
            initial.num_experts = max(1, int(getattr(neural, "num_initial_experts", DEFAULT_NUM_EXPERTS)))
        except (TypeError, ValueError):
            initial.num_experts = DEFAULT_NUM_EXPERTS
        try:
            initial.max_experts = max(initial.num_experts, int(getattr(neural, "max_experts", 12)))
        except (TypeError, ValueError):
            initial.max_experts = 12
        try:
            initial.num_layers = max(1, int(getattr(neural, "num_transformer_layers", 8)))
        except (TypeError, ValueError):
            initial.num_layers = 8
        try:
            initial.embedding_dim = max(16, int(getattr(neural, "embedding_dim", 768)))
        except (TypeError, ValueError):
            initial.embedding_dim = 768
        self._state_obj = initial

        self._load_state()
        self._initialized = True
        # Durum dosyasını hemen oluştur: yeniden başlatmada tutarlılık garantisi
        self._save_state()
        logger.info(
            f"Büyüme denetleyicisi hazır: uzman={self._state_obj.num_experts}/"
            f"{self._state_obj.max_experts} · nesil={self._state_obj.generation} · "
            f"embedding={self._state_obj.embedding_dim}"
        )

    def shutdown(self) -> None:
        """Durumu diske yaz ve kaynakları bırak. İdempotenttir."""
        if not self._initialized:
            return
        self._save_state()
        self._initialized = False
        logger.info("Büyüme denetleyicisi kapatıldı (durum kaydedildi)")

    # ─── Yol Çözümleme ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_absolute(spec: str) -> Path:
        """Yol tanımını mutlak `Path`'e çevirir (proje köküne göreli)."""
        project_root = Path(__file__).resolve().parents[2]
        candidate = Path(str(spec or DEFAULT_DATA_PATH))
        if candidate.is_absolute():
            return candidate
        return project_root / candidate

    @staticmethod
    def _resolve_data_path(config: Any) -> str:
        """
        Büyüme dizinini çöz.

        Sıra: `config.data_path` → `<base_path>/data/evolution` →
        proje kökündeki `config.yaml` → `data/evolution`.
        """
        explicit = getattr(config, "data_path", None) if config is not None else None
        if explicit:
            return str(Path(str(explicit)) / GROWTH_DIR)
        base = getattr(config, "base_path", None) if config is not None else None
        if base:
            return str(Path(str(base)) / "data" / "evolution" / GROWTH_DIR)
        try:
            project_root = Path(__file__).resolve().parents[2]
            cfg_file = project_root / "config.yaml"
            if cfg_file.exists():
                import yaml

                data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
                if data.get("data_path"):
                    return str(Path(str(data["data_path"])) / GROWTH_DIR)
                if data.get("base_path"):
                    return str(Path(str(data["base_path"])) / "data" / "evolution" / GROWTH_DIR)
        except Exception:
            pass
        return DEFAULT_DATA_PATH

    # ─── Kalıcı Durum ──────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Durumu diskten yükler; bozuk dosya sessizce yok sayılır."""
        if not self._state_file or not self._state_file.exists():
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
            loaded = GrowthState.from_dict(raw)
            self._state_obj = loaded
            logger.info(
                f"Büyüme durumu geri yüklendi: uzman={loaded.num_experts}/"
                f"{loaded.max_experts} · nesil={loaded.generation}"
            )
        except Exception as exc:
            logger.warning(f"Büyüme durumu okunamadı, sıfırdan başlanıyor: {exc}")

    def _save_state(self) -> None:
        """Durumu atomik olarak diske yazar."""
        if not self._state_file:
            return
        tmp = self._state_file.with_suffix(".json.tmp")
        try:
            payload = self._state_obj.to_dict()
            payload["updated_at"] = time.time()
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self._state_file)
        except Exception as exc:
            logger.warning(f"Büyüme durumu yazılamadı: {exc}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    # ─── Olay ve Durum Erişimi ─────────────────────────────────────────

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Olay yayınla (bus yoksa atlanır)."""
        if self._event_bus:
            try:
                self._event_bus.emit(event_type, data, source=self.name)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Olay yayınlanamadı ({event_type}): {exc}")

    def state(self) -> GrowthState:
        """Mevcut büyüme durumunu döndürür."""
        return self._state_obj

    def update_from_budget(self, budget: Dict[str, Any]) -> None:
        """
        Donanım bütçesine göre beklentileri uyarlar.

        `ResourceBudget.to_dict()` çıktısını kabul eder. Bütçe uzman eklemeye
        izin vermiyorsa `max_experts` düşürülür (mevcut uzman sayısının altına
        inmez); embedding için önerilen boyut `_budget` altında saklanır.
        """
        if not isinstance(budget, dict):
            return
        self._budget = dict(budget)

        try:
            max_ram_mb = float(budget.get("max_ram_mb") or 0.0)
            max_gpu_mb = float(budget.get("max_gpu_mb") or 0.0)
            device = str(budget.get("device") or "cpu")
            profile = str(budget.get("profile") or "")
            pressure = str(budget.get("memory_pressure") or "NORMAL").upper()
            suggested_embedding = int(budget.get("embedding_dim") or self._state_obj.embedding_dim)
        except (TypeError, ValueError):
            return

        available_mb = max_gpu_mb if device == "cuda" and max_gpu_mb > 0 else max_ram_mb

        # Bellek baskısı varsa büyümeyi kıs
        if pressure == "CRITICAL":
            allowed = max(self._state_obj.num_experts, 3)
        elif pressure == "WARNING":
            allowed = max(self._state_obj.num_experts, min(self._state_obj.max_experts, 8))
        elif available_mb > 0:
            allowed = max(self._state_obj.num_experts, int(max(3, min(24, available_mb / 512.0))))
        else:
            allowed = self._state_obj.max_experts

        previous = self._state_obj.max_experts
        self._state_obj.max_experts = max(self._state_obj.num_experts, int(allowed))

        if self._state_obj.max_experts != previous:
            logger.info(
                f"Bütçeye göre uzman üst sınırı güncellendi: {previous} → "
                f"{self._state_obj.max_experts} (profil={profile or 'bilinmiyor'}, "
                f"cihaz={device}, boş bellek≈{available_mb:.0f}MB)"
            )

        self._emit("evolution.budget_applied", {
            "max_experts": self._state_obj.max_experts,
            "suggested_embedding_dim": suggested_embedding,
            "available_mb": round(available_mb, 1),
            "device": device,
        })

    # ─── Karar Mantığı ─────────────────────────────────────────────────

    def decide(
        self,
        metrics: Dict[str, Any],
        buffer_stats: Optional[Dict[str, Any]] = None,
    ) -> GrowthDecision:
        """
        Metriklere bakarak tek bir büyüme kararı verir.

        Öncelik sırası: uzman ekle → dondur/buda → yeniden eğit → embedding
        genişlet → hiçbir şey yapma. Her karar Türkçe gerekçe ve sayısal güven
        taşır.

        Args:
            metrics: `expert_entropy`, `expert_usage`, `routing_confidence`,
                `mean_quality`, `mean_uncertainty`, `loss`, `buffer_size`, `novelty`
            buffer_stats: `ExperienceBuffer.stats()` çıktısı (örneklem sayısı için)

        Returns:
            `GrowthDecision`
        """
        self._counters["decisions"] += 1.0
        data = dict(metrics or {})
        stats = dict(buffer_stats or {})

        try:
            online = self._as_float(data.get("buffer_size"))
            if stats.get("count") is not None:
                online = max(online, self._as_float(stats.get("count")))

            should_grow, grow_reason = self.should_expand_experts(data, stats)
            if should_grow:
                decision = self._expand_decision(data, grow_reason)
                self._last_decision = decision.to_dict()
                return decision

            should_shrink, shrink_reason = self.should_prune(data)
            if should_shrink:
                decision = self._prune_decision(data, shrink_reason)
                self._last_decision = decision.to_dict()
                return decision

            # Yeniden eğitim
            if self._needs_retrain(data, online):
                batch_size = self._batch_size(data)
                new_items = max(0, int(online) - self._state_obj.total_training_steps * batch_size)
                quality = self._as_float(data.get("mean_quality"), 0.5)
                steps = max(1, min(2000, new_items // max(1, batch_size)))
                decision = GrowthDecision(
                    action=GrowthAction.RETRAIN,
                    reason=(
                        f"Tamponda {new_items} yeni örnek var (eğitim adımı "
                        f"{self._state_obj.total_training_steps} × batch {batch_size}) ve "
                        f"ortalama kalite {quality:.2f} < {RETRAIN_QUALITY:.2f}"
                    ),
                    magnitude=float(steps),
                    details={"steps": steps, "batch_size": batch_size, "new_items": new_items},
                    confidence=round(min(0.95, 0.5 + 0.45 * (1.0 - quality)), 4),
                )
                self._last_decision = decision.to_dict()
                return decision

            # Embedding genişletme
            allowed, budget_reason = self._embedding_allowed()
            uncertainty = self._as_float(data.get("mean_uncertainty"), 0.5)
            if allowed and uncertainty > 0.6:
                target = max(self._state_obj.embedding_dim * 2, 1024)
                decision = GrowthDecision(
                    action=GrowthAction.EXPAND_EMBEDDING,
                    reason=(
                        f"Ortalama belirsizlik {uncertainty:.2f} > 0.60 ve donanım bütçesi "
                        f"uygun ({budget_reason}); gömmeyi {self._state_obj.embedding_dim} → "
                        f"{target} büyütmek ayrım gücünü artırır"
                    ),
                    magnitude=float(target),
                    details={
                        "current_dim": self._state_obj.embedding_dim,
                        "target_dim": target,
                        "estimated_mb": round((target ** 2) * 4 * EMBED_BYTES_FACTOR / (1024 ** 2), 1),
                    },
                    confidence=round(min(0.9, 0.45 + (uncertainty - 0.6) * 1.5), 4),
                )
                self._last_decision = decision.to_dict()
                return decision

            # Hiçbir kural tetiklenmedi
            if not allowed and uncertainty > 0.6:
                reason = (
                    f"Belirsizlik yüksek ({uncertainty:.2f}) ama donanım bütçesi yetersiz: "
                    f"{budget_reason}"
                )
                confidence = 0.3
            else:
                reason = (
                    f"Büyüme ölçütleri sağlanmadı (entropi="
                    f"{self._as_float(data.get('expert_entropy')):.2f}, kalite="
                    f"{self._as_float(data.get('mean_quality'), 0.5):.2f}, "
                    f"tampon={int(online)}); mevcut yapı korunuyor"
                )
                confidence = 0.6

            decision = GrowthDecision(
                action=GrowthAction.NOOP,
                reason=reason,
                magnitude=0.0,
                details={"buffer_size": int(online), "budget_ok": bool(allowed)},
                confidence=confidence,
            )
            self._last_decision = decision.to_dict()
            return decision

        except Exception as exc:
            logger.warning(f"Büyüme kararı verilemedi: {exc}")
            decision = GrowthDecision(
                action=GrowthAction.NOOP,
                reason=f"Karar hesaplanamadı: {exc}",
                confidence=0.0,
            )
            self._last_decision = decision.to_dict()
            return decision

    # ─── Kural 1: Uzman Ekleme ─────────────────────────────────────────

    def should_expand_experts(
        self,
        metrics: Dict[str, Any],
        buffer_stats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Uzman eklenmeli mi?

        Returns:
            `(karar, Türkçe gerekçe)`
        """
        data = dict(metrics or {})
        stats = dict(buffer_stats or {})

        entropy = self._as_float(data.get("expert_entropy"))
        quality = self._as_float(data.get("mean_quality"), 0.5)
        online = self._as_float(data.get("buffer_size"))
        if stats.get("count") is not None:
            online = max(online, self._as_float(stats.get("count")))

        if self._state_obj.num_experts >= self._state_obj.max_experts:
            return False, (
                f"Uzman sayısı üst sınırda ({self._state_obj.num_experts}/"
                f"{self._state_obj.max_experts}); yeni uzman için bütçe yok"
            )
        if entropy < self._auto_expand_threshold:
            return False, (
                f"Yönlendirme entropisi {entropy:.2f} < eşik "
                f"{self._auto_expand_threshold:.2f}; uzmanlar yeterince ayrışmış"
            )
        if online < self._min_samples:
            return False, (
                f"Tamponda {int(online)} örnek var, eğitim için en az {self._min_samples} gerekli"
            )
        if quality > QUALITY_SATURATION:
            return False, (
                f"Ortalama kalite zaten yüksek ({quality:.2f} > {QUALITY_SATURATION:.2f}); "
                "yeni uzmana gerek yok"
            )
        return True, (
            f"Yönlendirme entropisi {entropy:.2f} ≥ eşik {self._auto_expand_threshold:.2f}, "
            f"tampon {int(online)} ≥ {self._min_samples}, kalite {quality:.2f} doygun değil → "
            "yeni uzman uzmanlaşmayı artırır"
        )

    def _expand_decision(self, metrics: Dict[str, Any], reason: str) -> GrowthDecision:
        """Uzman ekleme kararını kurar (güven entropi fazlalığıyla ölçeklenir)."""
        entropy = self._as_float(metrics.get("expert_entropy"))
        excess = max(0.0, entropy - self._auto_expand_threshold)
        share = excess / max(1e-6, 1.0 - self._auto_expand_threshold)
        confidence = round(min(0.98, 0.55 + 0.43 * share), 4)
        return GrowthDecision(
            action=GrowthAction.EXPAND_EXPERTS,
            reason=reason,
            magnitude=1.0,
            details={
                "expert_entropy": round(entropy, 4),
                "threshold": self._auto_expand_threshold,
                "num_experts": self._state_obj.num_experts,
                "max_experts": self._state_obj.max_experts,
                "excess": round(excess, 4),
            },
            confidence=confidence,
        )

    # ─── Kural 2: Dondurma / Budama ────────────────────────────────────

    def should_prune(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Bir uzman dondurulmalı/budanmalı mı?

        Kullanım payı `prune_threshold` altındaki en az kullanılan uzman
        seçilir; bu uzman üst üste `ZERO_USAGE_STREAK_FOR_PRUNE` kez ~0
        kullanıldıysa PRUNE, aksi hâlde FREEZE önerilir.
        """
        data = dict(metrics or {})
        usage = data.get("expert_usage") or []
        if not isinstance(usage, (list, tuple)) or len(usage) == 0:
            return False, "Uzman kullanım verisi yok; dondurma kararı ertelendi"

        # Ardışık ~0 kullanım sayacı (budama kararının tek dayanağı)
        self._update_zero_usage(usage)

        if self._state_obj.num_experts <= 3:
            return False, (
                f"Uzman sayısı {self._state_obj.num_experts} ≤ 3; daha fazla küçülme "
                "kapasiteyi riske atar"
            )

        try:
            shares = [max(0.0, float(value)) for value in usage]
        except (TypeError, ValueError):
            return False, "Uzman kullanım verisi bozuk"

        total = sum(shares)
        if total <= 0.0:
            return False, "Uzman kullanım toplamı sıfır; dağılım bilgisi güvenilmez"

        normalized = [value / total for value in shares]
        excluded = set(int(i) for i in (data.get("excluded_experts") or []))
        frozen = set(int(i) for i in self._state_obj.frozen_experts)

        candidates = [
            (index, share) for index, share in enumerate(normalized)
            if index not in excluded and index not in frozen
        ]
        if not candidates:
            return False, "Dondurulabilecek aday uzman yok"

        index, share = min(candidates, key=lambda pair: (pair[1], pair[0]))
        if share >= self._prune_threshold:
            return False, (
                f"En az kullanılan uzman #{index} payı {share:.3f} ≥ eşik "
                f"{self._prune_threshold:.3f}; dondurma gerekmez"
            )

        streak = self._zero_usage_streak.get(index, 0)
        if share <= 1e-3 and streak >= ZERO_USAGE_STREAK_FOR_PRUNE:
            return True, (
                f"Uzman #{index} payı {share:.4f} ve {streak} ardışık kayıtta ~0 kullanım → "
                "budama (arşive taşıma) güvenli"
            )
        return True, (
            f"Uzman #{index} payı {share:.3f} < eşik {self._prune_threshold:.3f} → "
            "dondurma (silme yok, geri alınabilir)"
        )

    def _prune_decision(self, metrics: Dict[str, Any], reason: str) -> GrowthDecision:
        """Dondurma/budama kararını kurar."""
        usage = list(metrics.get("expert_usage") or [])
        try:
            shares = [max(0.0, float(v)) for v in usage]
            total = sum(shares) or 1.0
            normalized = [v / total for v in shares]
            frozen = set(int(i) for i in self._state_obj.frozen_experts)
            excluded = set(int(i) for i in (metrics.get("excluded_experts") or []))
            candidates = [(i, s) for i, s in enumerate(normalized) if i not in frozen and i not in excluded]
            index, share = min(candidates, key=lambda pair: (pair[1], pair[0])) if candidates else (0, 0.0)
        except (TypeError, ValueError):
            index, share = 0, 0.0

        streak = self._zero_usage_streak.get(int(index), 0)
        action = (
            GrowthAction.PRUNE_EXPERT
            if share <= 1e-3 and streak >= ZERO_USAGE_STREAK_FOR_PRUNE
            else GrowthAction.FREEZE_EXPERT
        )
        confidence = round(min(0.95, 0.5 + (self._prune_threshold - min(share, self._prune_threshold))
                               / max(1e-6, self._prune_threshold) * 0.45), 4)
        return GrowthDecision(
            action=action,
            reason=reason,
            magnitude=round(float(share), 6),
            details={
                "expert_index": int(index),
                "usage_share": round(float(share), 4),
                "prune_threshold": self._prune_threshold,
                "zero_streak": int(streak),
                "num_experts": self._state_obj.num_experts,
            },
            confidence=confidence,
        )

    def _update_zero_usage(self, usage: Any) -> None:
        """Ardışık ~0 kullanım sayacını günceller (budama kararı için)."""
        if not isinstance(usage, (list, tuple)) or not usage:
            return
        try:
            shares = [max(0.0, float(v)) for v in usage]
        except (TypeError, ValueError):
            return
        total = sum(shares) or 1.0
        for index, value in enumerate(shares):
            if value / total <= 1e-3:
                self._zero_usage_streak[index] = self._zero_usage_streak.get(index, 0) + 1
            else:
                self._zero_usage_streak[index] = 0

    # ─── Kural 3: Yeniden Eğitim ───────────────────────────────────────

    def _needs_retrain(self, metrics: Dict[str, Any], online: float) -> bool:
        """Tamponda yeterli yeni örnek var ve kalite düşük mü?"""
        batch_size = self._batch_size(metrics)
        new_items = online - (self._state_obj.total_training_steps * batch_size)
        return new_items >= self._min_samples and self._as_float(
            metrics.get("mean_quality"), 0.5
        ) < RETRAIN_QUALITY

    def _batch_size(self, metrics: Optional[Dict[str, Any]] = None) -> int:
        """Eğitim batch boyutunu bütçe/metriklerden çıkarır."""
        data = dict(metrics or {})
        for source in (data.get("batch_size"), self._budget.get("max_batch_size")):
            try:
                value = int(source)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                continue
        try:
            return max(1, int(getattr(self._config, "max_batch_size", 32)))
        except (TypeError, ValueError):
            return 32

    # ─── Kural 4: Embedding Genişletme Bütçesi ─────────────────────────

    def _embedding_allowed(self) -> Tuple[bool, str]:
        """
        Donanım bütçesi embedding genişletmeye izin veriyor mu?

        Gereksinim: `embedding_dim² × 4 bayt × 12` + %25 pay, mevcut boş
        RAM/VRAM içinde kalmalı.
        """
        budget = self._budget or {}
        if not budget:
            return False, "Donanım bütçesi bilinmiyor (resource_budget kancası yok)"

        device = str(budget.get("device") or "cpu")
        try:
            max_ram_mb = float(budget.get("max_ram_mb") or 0.0)
            max_gpu_mb = float(budget.get("max_gpu_mb") or 0.0)
        except (TypeError, ValueError):
            return False, "Donanım bütçesi okunamadı"

        available_mb = max_gpu_mb if device == "cuda" and max_gpu_mb > 0 else max_ram_mb
        if available_mb <= 0:
            return False, "Boş RAM/VRAM bilgisi yok"

        required_mb = (self._state_obj.embedding_dim ** 2) * 4 * EMBED_BYTES_FACTOR / (1024 ** 2)
        required_mb *= EMBED_MEMORY_MARGIN
        if available_mb < required_mb:
            return False, (
                f"Gereken ≈{required_mb:.0f}MB > kullanılabilir {available_mb:.0f}MB "
                f"(embedding_dim={self._state_obj.embedding_dim}, cihaz={device})"
            )
        return True, (
            f"Kullanılabilir {available_mb:.0f}MB ≥ gereken ≈{required_mb:.0f}MB "
            f"(cihaz={device})"
        )

    # ─── Uygulama ve Kayıt ─────────────────────────────────────────────

    def apply(
        self,
        decision: GrowthDecision,
        apply_fn: Optional[Callable[[GrowthDecision], bool]] = None,
    ) -> bool:
        """
        Kararı uygular.

        `apply_fn` verilmezse (nöral katman bağlı değilse) yalnızca defter
        güncellenir ve `False` döner — yani karar "öneri" olarak kalır.
        `NOOP` kararları her zaman `False` döner.

        Args:
            decision: Uygulanacak karar
            apply_fn: Gerçek model mutasyonunu yapan geri çağrı

        Returns:
            Uygulandıysa True
        """
        if not isinstance(decision, GrowthDecision):
            return False
        if decision.action == GrowthAction.NOOP:
            self.record(decision, applied=False)
            return False

        if apply_fn is None:
            logger.info(f"Büyüme kararı uygulanamadı (model kancası yok): {decision.reason}")
            self.record(decision, applied=False)
            return False

        applied = False
        try:
            applied = bool(apply_fn(decision))
        except Exception as exc:
            self._counters["failed"] += 1.0
            logger.error(f"Büyüme kararı uygulanırken hata: {exc}")
            applied = False

        if applied:
            self._counters["applied"] += 1.0
            self._mutate_state(decision)
        else:
            logger.warning(f"Model büyüme kararını reddetti: {decision.action.value}")

        self.record(decision, applied=applied)
        return applied

    def _mutate_state(self, decision: GrowthDecision) -> None:
        """Uygulanan karara göre defteri günceller."""
        state = self._state_obj
        action = decision.action

        if action == GrowthAction.EXPAND_EXPERTS:
            increment = max(1, int(round(decision.magnitude or 1.0)))
            state.num_experts = min(state.max_experts, state.num_experts + increment)
        elif action == GrowthAction.FREEZE_EXPERT:
            index = self._expert_index(decision)
            if index is not None and index not in state.frozen_experts:
                state.frozen_experts.append(index)
        elif action == GrowthAction.PRUNE_EXPERT:
            index = self._expert_index(decision)
            if index is not None:
                if index not in state.frozen_experts:
                    state.frozen_experts.append(index)
                state.num_experts = max(1, state.num_experts - 1)
        elif action == GrowthAction.EXPAND_EMBEDDING:
            target = int(decision.magnitude or 0)
            if target > state.embedding_dim:
                state.embedding_dim = target
        elif action == GrowthAction.ADD_LAYER:
            state.num_layers = max(1, state.num_layers + max(1, int(round(decision.magnitude or 1.0))))
        elif action == GrowthAction.RETRAIN:
            state.total_training_steps += max(1, int(decision.details.get("steps", 1) or 1))

        state.generation += 1

    @staticmethod
    def _expert_index(decision: GrowthDecision) -> Optional[int]:
        """
        Karardan uzman indeksini çıkarır.

        Dondurma/budama kararlarında indeks `details["expert_index"]` içindedir;
        eski/elde kurulmuş kararlar için `magnitude` yedek olarak denenir.
        """
        raw = decision.details.get("expert_index")
        if raw is None and decision.action in (GrowthAction.FREEZE_EXPERT, GrowthAction.PRUNE_EXPERT):
            raw = int(decision.magnitude) if decision.magnitude else None
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def record(
        self,
        decision: GrowthDecision,
        applied: bool,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Kararı geçmişe yazar ve olay yayınlar."""
        if not isinstance(decision, GrowthDecision):
            return

        entry = {
            "at": time.time(),
            "action": decision.action.value,
            "reason": decision.reason,
            "magnitude": round(float(decision.magnitude), 4),
            "confidence": round(float(decision.confidence), 4),
            "applied": bool(applied),
            "dry_run": bool(decision.dry_run),
            "generation": self._state_obj.generation,
        }
        if metrics:
            entry["metrics"] = {
                key: metrics.get(key)
                for key in ("expert_entropy", "mean_quality", "mean_uncertainty", "loss", "novelty")
                if metrics.get(key) is not None
            }

        self._state_obj.history.append(entry)
        if len(self._state_obj.history) > MAX_HISTORY:
            self._state_obj.history = self._state_obj.history[-MAX_HISTORY:]

        self._emit("evolution.growth_decision", {
            "action": decision.action.value,
            "applied": bool(applied),
            "confidence": entry["confidence"],
            "reason": decision.reason,
            "generation": self._state_obj.generation,
        })

        if applied:
            logger.info(f"Büyüme uygulandı: {decision.action.value} · {decision.reason}")

    # ─── Olgunluk ──────────────────────────────────────────────────────

    def maturity(self) -> float:
        """
        Modelin ne kadar "olgunlaştığını" ölçer (0..1).

        Bileşenler: uzman kullanımı (max'a oran), eğitim adımları (doygunluk
        10.000 adım), nesil (doygunluk 20) ve katman sayısı (doygunluk 16).
        """
        state = self._state_obj

        expert_component = _clamp(state.num_experts / max(1, state.max_experts))
        steps_component = _clamp(state.total_training_steps / 10000.0)
        generation_component = _clamp(state.generation / 20.0)
        layer_component = _clamp(state.num_layers / 16.0)

        score = (
            0.40 * expert_component
            + 0.30 * steps_component
            + 0.20 * generation_component
            + 0.10 * layer_component
        )
        return round(_clamp(score), 4)

    # ─── Eğitim Planı ──────────────────────────────────────────────────

    def next_training_plan(
        self,
        buffer_stats: Dict[str, Any],
        budget: Dict[str, Any],
        quality_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Bir sonraki eğitim turunun planını üretir.

        Args:
            buffer_stats: `ExperienceBuffer.stats()` çıktısı
            budget: `ResourceBudget.to_dict()` çıktısı
            quality_report: `QualityReport.to_dict()` çıktısı (isteğe bağlı)

        Returns:
            `{"steps", "batch_size", "lr", "priority_tags", "rationale", "should_train"}`
        """
        stats = dict(buffer_stats or {})
        plan_budget = dict(budget or {})
        report = dict(quality_report or {})

        try:
            count = int(stats.get("count") or 0)
        except (TypeError, ValueError):
            count = 0

        batch_size = self._batch_size({"batch_size": plan_budget.get("max_batch_size")})
        quality = self._as_float(report.get("score"), self._as_float(stats.get("mean_quality"), 0.5))

        # Adım sayısı: tampondaki eğitilmemiş örnek / batch (sınırlı)
        new_items = max(0, count - self._state_obj.total_training_steps * batch_size)
        if new_items >= self._min_samples:
            steps = int(min(2000, max(1, new_items // max(1, batch_size))))
        else:
            steps = int(min(200, max(1, count // max(1, batch_size * 4)))) if count else 0

        # Öğrenme oranı: kalite düştükçe hafif artır, bellek baskısında düşür
        try:
            base_lr = float(getattr(getattr(self._config, "neural", None), "learning_rate", 1e-4))
        except (TypeError, ValueError):
            base_lr = 1e-4
        lr = base_lr
        if quality < 0.5:
            lr *= 1.5
        elif quality > 0.8:
            lr *= 0.6
        pressure = str(plan_budget.get("memory_pressure") or "NORMAL").upper()
        if pressure == "CRITICAL":
            lr *= 0.5
            batch_size = max(4, batch_size // 4)
        elif pressure == "WARNING":
            lr *= 0.8
            batch_size = max(8, batch_size // 2)

        priority_tags = self._priority_tags(stats)

        should_train = count >= self._min_samples and steps > 0
        if count < self._min_samples:
            rationale = (
                f"Tamponda {count} örnek var; eğitim için en az {self._min_samples} gerekli"
            )
        elif quality < RETRAIN_QUALITY:
            rationale = (
                f"Ortalama kalite {quality:.2f} < {RETRAIN_QUALITY:.2f}; "
                f"{steps} adım, batch={batch_size}, lr={lr:.2e}"
            )
        else:
            rationale = (
                f"Kalite {quality:.2f} yeterli; yine de {steps} adım hafif ince ayar "
                f"planlandı (batch={batch_size}, lr={lr:.2e})"
            )
        if pressure != "NORMAL":
            rationale += f" · bellek baskısı {pressure} nedeniyle ölçek küçültüldü"

        plan = {
            "steps": int(steps),
            "batch_size": int(batch_size),
            "lr": round(float(lr), 8),
            "priority_tags": priority_tags,
            "rationale": rationale,
            "should_train": bool(should_train),
            "quality": round(float(quality), 4),
            "buffer_size": count,
            "memory_pressure": pressure,
        }
        self._emit("evolution.training_planned", {
            "steps": plan["steps"],
            "batch_size": plan["batch_size"],
            "should_train": plan["should_train"],
        })
        return plan

    @staticmethod
    def _priority_tags(stats: Dict[str, Any], limit: int = 5) -> List[str]:
        """
        Eğitimde önceliklendirilecek etiketler.

        Etiket histogramındaki en sık etiketler öncelik kazanır; `mean_quality`
        düşükse (zayıf alanlar) ilk sıradakiler öne alınır.
        """
        histogram = stats.get("tag_histogram") or {}
        if not isinstance(histogram, dict) or not histogram:
            return []
        try:
            ordered = sorted(histogram.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        except (TypeError, ValueError):
            return []
        return [str(tag) for tag, _ in ordered[:limit]]

    # ─── Yardımcılar ───────────────────────────────────────────────────

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        """Güvenli float dönüşümü."""
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        if out != out:  # NaN
            return float(default)
        return out

    # ─── İstatistikler ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        state = self._state_obj
        return {
            "name": self.name,
            "initialized": self._initialized,
            "state": state.to_dict(),
            "maturity": self.maturity(),
            "decisions": int(self._counters["decisions"]),
            "applied": int(self._counters["applied"]),
            "failed": int(self._counters["failed"]),
            "auto_expand_threshold": self._auto_expand_threshold,
            "prune_threshold": self._prune_threshold,
            "min_samples_for_training": self._min_samples,
            "last_decision": dict(self._last_decision),
            "budget": {
                "device": self._budget.get("device"),
                "max_ram_mb": self._budget.get("max_ram_mb"),
                "max_gpu_mb": self._budget.get("max_gpu_mb"),
                "profile": self._budget.get("profile"),
                "memory_pressure": self._budget.get("memory_pressure"),
            },
            "zero_usage_streak": {str(k): int(v) for k, v in self._zero_usage_streak.items()},
            "history_size": len(state.history),
            "state_file": str(self._state_file) if self._state_file else "",
        }
