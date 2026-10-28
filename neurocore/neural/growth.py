"""
NeuroCore Nöral Büyüme Yöneticisi
==================================
Modelin ne zaman ve nasıl büyüyeceğine karar veren katman.

Bu modül **karar** verir; uygulama `NeuroCoreModel` üzerinden yapılır. Böylece
politika (hangi ölçütle büyümeli) ile mekanizma (ağırlıkları değiştirme) ayrılır
ve ikisi bağımsız test edilebilir.

Büyüme eksenleri:
    1. **Uzman büyümesi** (MoE genişliği) — kapasite artışı, maliyet sabit
    2. **Katman büyümesi** (derinlik) — soyutlama artışı, maliyet doğrusal
    3. **Gömme genişletmesi** — temsil gücü, maliyet karesel (en pahalı)
    4. **Dondurma/budama** — kaynak tasarrufu, kapasite kaybı yok

Karar ölçütleri (hepsi donanım bütçesiyle sınırlanır):
    * yönlendirme entropisi (model ayırt edemiyor mu?)
    * yönlendirme güveni (uzmanlar işe yarıyor mu?)
    * ortalama yanıt kalitesi (öğreniyor muyuz?)
    * ortalama belirsizlik (model kendinden emin değil mi?)
    * tampon doluluğu (öğrenecek veri var mı?)
    * yenilik oranı (yeni şeyler görüyor muyuz?)
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.growth")


# ─── Eylem Türleri ──────────────────────────────────────────────────────────

class GrowthAction(str, Enum):
    """Uygulanabilir büyüme eylemleri."""
    GROW_EXPERTS = "grow_experts"
    ACTIVATE_EXPERTS = "activate_experts"
    WARMUP_EXPERTS = "warmup_experts"
    FREEZE_EXPERT = "freeze_expert"
    UNFREEZE_EXPERT = "unfreeze_expert"
    PRUNE_EXPERT = "prune_expert"
    GROW_LAYERS = "grow_layers"
    EXPAND_EMBEDDING = "expand_embedding"
    QUANTIZE = "quantize"
    RETRAIN = "retrain"
    NOOP = "noop"


#: Eylemlerin göreli maliyeti (kaynak bütçesi karşılaştırması için)
ACTION_COST: Dict[str, float] = {
    "grow_experts": 0.35,
    "activate_experts": 0.01,
    "warmup_experts": 0.01,
    "freeze_expert": -0.05,
    "unfreeze_expert": 0.2,
    "prune_expert": -0.3,
    "grow_layers": 0.6,
    "expand_embedding": 1.0,
    "quantize": -0.4,
    "retrain": 0.5,
    "noop": 0.0,
}


@dataclass
class GrowthDecision:
    """Tek bir büyüme kararı — gerekçesi ve güveniyle."""
    action: GrowthAction = GrowthAction.NOOP
    reason: str = ""
    confidence: float = 0.0
    magnitude: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    predicted_gain: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "action": self.action.value if isinstance(self.action, GrowthAction)
                      else str(self.action),
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "magnitude": round(self.magnitude, 4),
            "cost": round(self.cost, 4),
            "predicted_gain": round(self.predicted_gain, 4),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GrowthDecision":
        """Sözlükten karar üretir."""
        try:
            action = GrowthAction(str(data.get("action", "noop")))
        except ValueError:
            action = GrowthAction.NOOP
        return cls(
            action=action,
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            magnitude=float(data.get("magnitude", 0.0) or 0.0),
            details=dict(data.get("details") or {}),
            cost=float(data.get("cost", 0.0) or 0.0),
            predicted_gain=float(data.get("predicted_gain", 0.0) or 0.0),
        )


# ─── Metrik Toplayıcı ───────────────────────────────────────────────────────

@dataclass
class GrowthMetrics:
    """
    Büyüme kararının girdisi.

    Tüm alanlar isteğe bağlıdır; eksik metrikler kararı engellemez, yalnızca
    güveni düşürür (ilk çalıştırmalarda metrik birikmemiş olabilir).
    """
    expert_entropy: float = 0.0
    routing_confidence: float = 0.0
    expert_usage: List[float] = field(default_factory=list)
    num_experts: int = 0
    max_experts: int = 12
    num_layers: int = 0
    embed_dim: int = 0
    mean_quality: float = 0.5
    mean_uncertainty: float = 0.5
    buffer_size: int = 0
    min_samples: int = 100
    novelty: float = 0.5
    loss: float = 0.0
    loss_trend: float = 0.0        # negatif = iyileşiyor
    maturity: float = 0.0
    generation: int = 0
    frozen_experts: int = 0
    memory_pressure: str = "NORMAL"
    available_mb: float = 0.0
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        data = {key: getattr(self, key) for key in self.__dataclass_fields__}
        data["expert_usage"] = [round(value, 5) for value in self.expert_usage]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GrowthMetrics":
        """Sözlükten metrik üretir (bilinmeyen alanlar yok sayılır)."""
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in (data or {}).items() if key in allowed}
        return cls(**clean)

    @classmethod
    def from_sources(cls, model: Any = None, budget: Optional[Dict[str, Any]] = None,
                     evolution: Optional[Dict[str, Any]] = None,
                     routing: Optional[Dict[str, Any]] = None,
                     quality: float = 0.5, uncertainty: float = 0.5,
                     buffer_size: int = 0, novelty: float = 0.5,
                     min_samples: int = 100, loss: float = 0.0,
                     loss_trend: float = 0.0) -> "GrowthMetrics":
        """
        Farklı kaynaklardan tek bir metrik nesnesi üretir.

        Args:
            model: `NeuroCoreModel` (yönlendirme + büyüme durumu)
            budget: Kaynak bütçesi
            evolution: Evrim katmanı metrikleri (`growth_metrics()`)
            routing: Doğrudan yönlendirme raporu (model'i geçersiz kılar)
        """
        metrics = cls()
        metrics.mean_quality = float(quality)
        metrics.mean_uncertainty = float(uncertainty)
        metrics.buffer_size = int(buffer_size)
        metrics.min_samples = int(min_samples)
        metrics.novelty = float(novelty)
        metrics.loss = float(loss)
        metrics.loss_trend = float(loss_trend)

        if routing is None and model is not None and hasattr(model, "routing_report"):
            try:
                routing = model.routing_report()
            except Exception as exc:
                logger.debug(f"Yönlendirme raporu alınamadı: {exc}")
        if routing:
            metrics.expert_entropy = float(routing.get("expert_entropy", 0.0) or 0.0)
            metrics.routing_confidence = float(routing.get("routing_confidence", 0.0) or 0.0)
            metrics.expert_usage = [float(value) for value in (routing.get("usage_ema") or [])]
            metrics.num_experts = int(routing.get("num_experts", 0) or 0)

        if model is not None:
            growth = getattr(model, "growth", None)
            if growth is not None:
                metrics.num_layers = int(getattr(growth, "num_layers", 0) or 0)
                metrics.embed_dim = int(getattr(growth, "embed_dim", 0) or 0)
                metrics.maturity = float(getattr(growth, "maturity", 0.0) or 0.0)
                metrics.generation = int(getattr(growth, "generation", 0) or 0)
                metrics.frozen_experts = len(getattr(growth, "frozen_experts", []) or [])
                if not metrics.num_experts:
                    metrics.num_experts = int(getattr(growth, "num_experts", 0) or 0)
            model_config = getattr(model, "model_config", None)
            if model_config is not None:
                metrics.max_experts = int(getattr(model_config, "max_experts", 12) or 12)

        if budget:
            metrics.memory_pressure = str(budget.get("memory_pressure", "NORMAL"))
            metrics.device = str(budget.get("device", "cpu"))
            metrics.available_mb = float(
                budget.get("max_gpu_mb" if metrics.device.startswith("cuda") else "max_ram_mb", 0.0)
                or 0.0
            )

        if evolution:
            metrics.mean_quality = float(evolution.get("mean_quality", metrics.mean_quality) or 0.0)
            metrics.mean_uncertainty = float(
                evolution.get("mean_uncertainty", metrics.mean_uncertainty) or 0.0
            )
            metrics.buffer_size = int(evolution.get("buffer_size", metrics.buffer_size) or 0)
            metrics.novelty = float(evolution.get("novelty", metrics.novelty) or 0.0)
        return metrics


# ─── Büyüme Yöneticisi ──────────────────────────────────────────────────────

class GrowthController:
    """
    Büyüme kararlarını üreten ve uygulayan yönetici.

    Kullanım:
        controller = GrowthController()
        controller.initialize(config, event_bus)

        metrics = GrowthMetrics.from_sources(model=model, budget=budget)
        decision = controller.decide(metrics)
        applied = controller.apply(decision, model)

        plan = controller.next_training_plan(metrics)
    """

    name = "neural.growth"

    def __init__(self) -> None:
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._lock = threading.RLock()

        # Politika eşikleri (config'ten güncellenir)
        self.entropy_threshold = 0.80
        self.confidence_threshold = 0.30
        self.quality_threshold = 0.62
        self.prune_threshold = 0.05
        self.min_experts = 3
        self.auto_expand = True
        self.cooldown_seconds = 60.0

        self._last_action_at: Dict[str, float] = {}
        self._counter = {
            "decisions": 0, "applied": 0, "rejected": 0, "noop": 0,
            "expert_growths": 0, "layer_growths": 0, "freezes": 0, "prunes": 0,
        }
        self._history: List[Dict[str, Any]] = []

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Yöneticiyi başlatır (idempotent)."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus

        evolution = getattr(config, "evolution", None)
        neural = getattr(config, "neural", None)
        self.entropy_threshold = float(
            getattr(evolution, "auto_expand_threshold", 0.8) or 0.8
        )
        self.prune_threshold = float(getattr(evolution, "prune_threshold", 0.05) or 0.05)
        self.min_experts = max(2, int(getattr(neural, "num_initial_experts", 3) or 3))
        self.confidence_threshold = 0.30
        self.quality_threshold = 0.62
        self.auto_expand = True

        self._initialized = True
        logger.info(
            f"Büyüme yöneticisi hazır — entropi eşiği {self.entropy_threshold}, "
            f"kalite eşiği {self.quality_threshold}"
        )

    def shutdown(self) -> None:
        """Yöneticiyi kapatır."""
        self._initialized = False
        logger.info(
            f"Büyüme yöneticisi kapatıldı ({self._counter['applied']} uygulanan karar, "
            f"{self._counter['expert_growths']} uzman, {self._counter['layer_growths']} katman büyümesi)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Yönetici istatistikleri."""
        return {
            **self._counter,
            "initialized": self._initialized,
            "entropy_threshold": self.entropy_threshold,
            "quality_threshold": self.quality_threshold,
            "prune_threshold": self.prune_threshold,
            "min_experts": self.min_experts,
            "cooldown_seconds": self.cooldown_seconds,
            "history_size": len(self._history),
            "last_decisions": self._history[-10:],
        }

    def configure(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Politika eşiklerini günceller.

        Kabul edilen anahtarlar: `entropy_threshold`, `confidence_threshold`,
        `quality_threshold`, `prune_threshold`, `min_experts`, `auto_expand`,
        `cooldown_seconds`.
        """
        changed: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if hasattr(self, key) and not key.startswith("_"):
                old = getattr(self, key)
                setattr(self, key, value)
                changed[key] = {"old": old, "new": value}
        if changed:
            logger.info(f"Büyüme politikası güncellendi: {list(changed.keys())}")
        return changed

    # ─── Karar ──────────────────────────────────────────────────────────

    def decide(self, metrics: GrowthMetrics) -> GrowthDecision:
        """
        Metriklerden büyüme kararı üretir.

        Öncelik sırası (ilk uyan kazanır):
            1. **Kaynak baskısı** → nicelleştir/budama (hayatta kalma)
            2. **Yüksek entropi + düşük güven + veri var** → uzman büyüt
            3. **Kullanılmayan uzman** → dondur/buda
            4. **Kalite düşük + veri birikmiş** → yeniden eğit
            5. **Belirsizlik yüksek + kaynak var + olgun** → katman/gömme büyüt
            6. Aksi hâlde → NOOP

        Args:
            metrics: `GrowthMetrics`

        Returns:
            GrowthDecision
        """
        self._counter["decisions"] += 1
        pressure = metrics.memory_pressure

        # ── 0. Soğuma süresi: aynı eylemi üst üste tekrarlama ──
        def cooling(action: str) -> bool:
            last = self._last_action_at.get(action, 0.0)
            return (time.time() - last) < self.cooldown_seconds

        # ── 1. Kaynak baskısı ──
        if pressure == "CRITICAL":
            if metrics.num_experts > self.min_experts + 1:
                idle = self._most_idle(metrics)
                if idle >= 0:
                    return self._decision(
                        GrowthAction.PRUNE_EXPERT,
                        f"KRİTİK bellek baskısı — en az kullanılan uzman "
                        f"({idle}, pay {self._usage_share(metrics, idle):.3f}) budanıyor",
                        confidence=0.85, magnitude=self._usage_share(metrics, idle),
                        details={"expert_index": idle, "pressure": pressure},
                    )
            if metrics.embed_dim >= 512:
                return self._decision(
                    GrowthAction.QUANTIZE,
                    "KRİTİK bellek baskısı — niceleme ile bellek tasarrufu",
                    confidence=0.8, magnitude=0.5,
                    details={"mode": "fp16" if metrics.device.startswith("cuda") else "bf16"},
                )
            return self._decision(
                GrowthAction.NOOP,
                "KRİTİK bellek baskısı ama budanacak uzman yok",
                confidence=0.6,
            )

        # ── 2. Uzman büyümesi: kapasite yetersiz ──
        can_grow_experts = (
            self.auto_expand
            and metrics.num_experts > 0
            and metrics.num_experts < metrics.max_experts
            and metrics.buffer_size >= metrics.min_samples
            and metrics.expert_entropy >= self.entropy_threshold
            and metrics.routing_confidence <= self.confidence_threshold
            and pressure != "WARNING"
            and not cooling("grow_experts")
        )
        if can_grow_experts:
            urgency = min(1.0, (metrics.expert_entropy - self.entropy_threshold) /
                          max(1e-6, 1.0 - self.entropy_threshold))
            quality_room = max(0.0, self.quality_threshold - metrics.mean_quality)
            confidence = min(0.95, 0.45 + 0.35 * urgency + 0.2 * (1.0 if quality_room > 0 else 0.0))
            return self._decision(
                GrowthAction.GROW_EXPERTS,
                f"yönlendirme entropisi {metrics.expert_entropy:.2f} ≥ {self.entropy_threshold} "
                f"ve güven {metrics.routing_confidence:.2f} düşük — kapasite yetersiz "
                f"({metrics.num_experts}/{metrics.max_experts} uzman, "
                f"{metrics.buffer_size} örnek hazır)",
                confidence=confidence, magnitude=urgency,
                details={"add": 1, "current": metrics.num_experts,
                         "max": metrics.max_experts},
                predicted_gain=round(0.15 * urgency, 4),
            )

        # ── 2b. Uyuyan uzmanları devreye al ──
        if metrics.frozen_experts > 0 and metrics.mean_quality < self.quality_threshold:
            return self._decision(
                GrowthAction.UNFREEZE_EXPERT,
                f"{metrics.frozen_experts} donmuş uzman var ve kalite düşük "
                f"({metrics.mean_quality:.2f}) — yeniden etkinleştirme değerlendirilmeli",
                confidence=0.5, magnitude=float(metrics.frozen_experts),
                details={"frozen": metrics.frozen_experts},
            )

        # ── 3. Kullanılmayan uzman: dondur ──
        if metrics.expert_usage and metrics.num_experts > self.min_experts:
            idle = self._most_idle(metrics)
            if idle >= 0 and self._usage_share(metrics, idle) < self.prune_threshold:
                share = self._usage_share(metrics, idle)
                action = (GrowthAction.PRUNE_EXPERT if share < self.prune_threshold / 4
                          and metrics.num_experts > self.min_experts + 1
                          else GrowthAction.FREEZE_EXPERT)
                return self._decision(
                    action,
                    f"uzman {idle} kullanım payı {share:.4f} < {self.prune_threshold} — "
                    f"{'budanıyor' if action == GrowthAction.PRUNE_EXPERT else 'donduruluyor'}",
                    confidence=0.7, magnitude=share,
                    details={"expert_index": idle, "usage_share": share},
                )

        # ── 4. Yeniden eğitim ──
        if (metrics.buffer_size >= metrics.min_samples
                and metrics.mean_quality < self.quality_threshold
                and not cooling("retrain")):
            deficit = self.quality_threshold - metrics.mean_quality
            return self._decision(
                GrowthAction.RETRAIN,
                f"kalite {metrics.mean_quality:.2f} < eşik {self.quality_threshold} ve "
                f"{metrics.buffer_size} örnek hazır — yeniden eğitim gerekli",
                confidence=min(0.9, 0.5 + deficit),
                magnitude=min(1.0, deficit * 2),
                details={"quality": metrics.mean_quality, "samples": metrics.buffer_size},
                predicted_gain=round(min(0.3, deficit), 4),
            )

        # ── 5. Derinlik / gömme genişletmesi ──
        if (metrics.mean_uncertainty > 0.6
                and metrics.mean_quality >= self.quality_threshold
                and metrics.maturity > 0.4
                and self._has_memory_room(metrics)
                and not cooling("grow_layers")):
            return self._decision(
                GrowthAction.GROW_LAYERS,
                f"kalite iyi ({metrics.mean_quality:.2f}) ama belirsizlik yüksek "
                f"({metrics.mean_uncertainty:.2f}) ve bellek yeterli — derinlik artırılabilir",
                confidence=0.55, magnitude=metrics.mean_uncertainty,
                details={"add": 1, "current_layers": metrics.num_layers},
                predicted_gain=round(0.12 * metrics.mean_uncertainty, 4),
            )

        # ── 6. Hiçbiri ──
        self._counter["noop"] += 1
        reasons: List[str] = []
        if metrics.num_experts >= metrics.max_experts:
            reasons.append(f"uzman sınırı ({metrics.max_experts})")
        if metrics.expert_entropy < self.entropy_threshold:
            reasons.append(f"entropi düşük ({metrics.expert_entropy:.2f})")
        if metrics.buffer_size < metrics.min_samples:
            reasons.append(f"veri yetersiz ({metrics.buffer_size}/{metrics.min_samples})")
        if metrics.mean_quality >= self.quality_threshold:
            reasons.append(f"kalite yeterli ({metrics.mean_quality:.2f})")
        if pressure == "WARNING":
            reasons.append("bellek uyarısı")
        return self._decision(
            GrowthAction.NOOP,
            "büyüme gerekmiyor: " + (", ".join(reasons) if reasons else "ölçütler sağlanmadı"),
            confidence=0.6,
        )

    def _decision(self, action: GrowthAction, reason: str, confidence: float = 0.5,
                  magnitude: float = 0.0, details: Optional[Dict[str, Any]] = None,
                  predicted_gain: float = 0.0) -> GrowthDecision:
        """Karar nesnesi üretir ve maliyeti hesaplar."""
        cost = ACTION_COST.get(action.value, 0.0)
        decision = GrowthDecision(
            action=action, reason=reason, confidence=float(confidence),
            magnitude=float(magnitude), details=dict(details or {}),
            cost=cost, predicted_gain=float(predicted_gain),
        )
        self._emit("neural.growth_decided", decision.to_dict())
        return decision

    @staticmethod
    def _usage_share(metrics: GrowthMetrics, index: int) -> float:
        """Uzmanın kullanım payı."""
        if 0 <= index < len(metrics.expert_usage):
            return float(metrics.expert_usage[index])
        return 0.0

    @staticmethod
    def _most_idle(metrics: GrowthMetrics) -> int:
        """En az kullanılan uzmanın indeksi (-1 = yok)."""
        if not metrics.expert_usage:
            return -1
        return min(range(len(metrics.expert_usage)),
                   key=lambda index: metrics.expert_usage[index])

    @staticmethod
    def _has_memory_room(metrics: GrowthMetrics) -> bool:
        """
        Katman büyümesi için bellek var mı?

        Kaba ölçüt: kullanılabilir belleğin, bir katmanın eğitim maliyetini
        (yaklaşık 12 × d² × 4 bayt × 4 Adam faktörü) karşılaması.
        """
        if metrics.available_mb <= 0:
            return False
        if metrics.embed_dim <= 0:
            return True
        layer_cost_mb = 12 * (metrics.embed_dim ** 2) * 4 * 4 / (1024 ** 2)
        return metrics.available_mb > layer_cost_mb * 1.5

    # ─── Uygulama ───────────────────────────────────────────────────────

    def apply(self, decision: GrowthDecision, model: Any,
              dry_run: bool = False) -> Dict[str, Any]:
        """
        Kararı modele uygular.

        Args:
            decision: `decide()` çıktısı
            model: `NeuroCoreModel`
            dry_run: Yalnızca raporla, değiştirme

        Returns:
            {"applied": bool, "action": str, "result": {...}, "reason": str}
        """
        result: Dict[str, Any] = {
            "applied": False,
            "action": decision.action.value if isinstance(decision.action, GrowthAction)
                      else str(decision.action),
            "reason": decision.reason,
            "dry_run": dry_run,
            "result": {},
        }

        if model is None or not getattr(model, "is_ready", False):
            result["reason"] = "model_hazır_değil"
            self._counter["rejected"] += 1
            return result

        action = decision.action if isinstance(decision.action, GrowthAction) \
            else GrowthAction.NOOP

        if action == GrowthAction.NOOP:
            self._counter["noop"] += 1
            result["reason"] = decision.reason
            return result

        if dry_run:
            result["applied"] = False
            result["reason"] = f"[kuru çalıştırma] {decision.reason}"
            return result

        try:
            if action == GrowthAction.GROW_EXPERTS:
                count = int(decision.details.get("add", 1) or 1)
                outcome = model.grow_experts(count, all_layers=True, dormant=True)
                result["result"] = outcome
                result["applied"] = bool(outcome.get("added"))
                if result["applied"]:
                    self._counter["expert_growths"] += 1
                    self._last_action_at["grow_experts"] = time.time()

            elif action == GrowthAction.ACTIVATE_EXPERTS:
                activated = model.activate_experts(float(decision.details.get("bias", 0.0)))
                result["result"] = {"activated": activated}
                result["applied"] = activated > 0
                if result["applied"]:
                    self._last_action_at["activate_experts"] = time.time()

            elif action == GrowthAction.WARMUP_EXPERTS:
                changed = model.warmup_experts(float(decision.magnitude or 2.0))
                result["result"] = {"warmed": changed}
                result["applied"] = changed > 0

            elif action == GrowthAction.FREEZE_EXPERT:
                layer = int(decision.details.get("layer", 0) or 0)
                expert = int(decision.details.get("expert_index", -1))
                if expert < 0:
                    result["reason"] = "uzman indeksi belirtilmedi"
                    self._counter["rejected"] += 1
                    return result
                ok = model.freeze_expert(layer, expert, decision.reason)
                result["result"] = {"layer": layer, "expert": expert, "frozen": ok}
                result["applied"] = ok
                if ok:
                    self._counter["freezes"] += 1
                    self._last_action_at["freeze_expert"] = time.time()

            elif action == GrowthAction.UNFREEZE_EXPERT:
                layer = int(decision.details.get("layer", 0) or 0)
                expert = int(decision.details.get("expert_index", -1))
                if expert < 0:
                    # Tüm donmuş uzmanları çöz
                    activated = model.activate_experts(0.0)
                    result["result"] = {"activated": activated}
                    result["applied"] = activated > 0
                else:
                    ok = model.unfreeze_expert(layer, expert)
                    result["result"] = {"layer": layer, "expert": expert, "unfrozen": ok}
                    result["applied"] = ok

            elif action == GrowthAction.PRUNE_EXPERT:
                # `NeuroCoreModel` doğrudan budama sunmaz (geri alınamaz işlem);
                # güvenli karşılığı dondurmadır.
                layer = int(decision.details.get("layer", 0) or 0)
                expert = int(decision.details.get("expert_index", -1))
                if expert < 0:
                    result["reason"] = "uzman indeksi belirtilmedi"
                    self._counter["rejected"] += 1
                    return result
                ok = model.freeze_expert(layer, expert,
                                         f"budama yerine dondurma: {decision.reason}")
                result["result"] = {"layer": layer, "expert": expert, "frozen": ok,
                                    "note": "geri alınabilirlik için dondurma tercih edildi"}
                result["applied"] = ok
                if ok:
                    self._counter["prunes"] += 1
                    self._last_action_at["prune_expert"] = time.time()

            elif action == GrowthAction.GROW_LAYERS:
                count = int(decision.details.get("add", 1) or 1)
                outcome = model.grow_layers(count, neutral=True)
                result["result"] = outcome
                result["applied"] = bool(outcome.get("added"))
                if result["applied"]:
                    self._counter["layer_growths"] += 1
                    self._last_action_at["grow_layers"] = time.time()

            elif action == GrowthAction.QUANTIZE:
                mode = str(decision.details.get("mode", "fp16"))
                outcome = model.quantize(mode)
                result["result"] = outcome
                result["applied"] = "error" not in outcome
                if result["applied"]:
                    self._last_action_at["quantize"] = time.time()

            elif action in (GrowthAction.RETRAIN, GrowthAction.EXPAND_EMBEDDING):
                # Bu eylemler model dışı koordinasyon gerektirir (eğitim turu /
                # yeni mimari); karar kaydedilir, uygulama motorda yapılır.
                result["applied"] = False
                result["reason"] = (
                    f"{action.value} eylemi eğitim/mimari koordinasyonu gerektirir — "
                    f"motor tarafından ele alınacak"
                )
                result["deferred"] = True

            else:
                result["reason"] = f"bilinmeyen eylem: {action}"

        except Exception as exc:
            self._counter["rejected"] += 1
            result["reason"] = f"uygulama hatası: {exc}"
            logger.error(f"Büyüme kararı uygulanamadı ({action}): {exc}", exc_info=True)

        if result["applied"] and not result.get("deferred"):
            self._counter["applied"] += 1

        self._history.append({
            "at": time.time(),
            "action": result["action"],
            "applied": result["applied"],
            "reason": decision.reason,
            "confidence": decision.confidence,
        })
        if len(self._history) > 200:
            self._history = self._history[-200:]

        self._emit("neural.growth_applied", {
            "action": result["action"], "applied": result["applied"],
            "reason": decision.reason,
        })
        if result["applied"]:
            logger.info(f"Büyüme uygulandı: {result['action']} — {decision.reason}")
        return result

    # ─── Eğitim Planı ───────────────────────────────────────────────────

    def next_training_plan(self, metrics: GrowthMetrics,
                           quality_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Bir sonraki eğitim turu için öneri üretir.

        `Trainer.plan_training()` ile aynı sözleşmeyi paylaşır ama burada
        yalnızca **öneri** üretilir (kaynak tahsisi eğiticide yapılır).

        Args:
            metrics: `GrowthMetrics`
            quality_report: `SelfEvaluator` çıktısı (isteğe bağlı)

        Returns:
            {"should_train": bool, "steps", "batch_size", "lr", "priority_tags",
             "rationale", "focus"}
        """
        health = float((quality_report or {}).get("score", metrics.mean_quality))
        deficit = max(0.0, self.quality_threshold - health)
        should_train = (
            metrics.buffer_size >= max(10, metrics.min_samples // 4)
            and (deficit > 0.05 or metrics.loss_trend > 0)
        )

        # Adım sayısı: eksik veri varsa örnek sayısına göre, yoksa sabit ince ayar
        if metrics.buffer_size > 0:
            steps = int(min(1000, max(10, metrics.buffer_size / 4)))
        else:
            steps = 20

        # Öğrenme oranı: açık büyükse daha güçlü öğrenme
        base_lr = float(getattr(getattr(self._config, "neural", None), "learning_rate", 1e-4) or 1e-4)
        lr = base_lr * (2.0 if deficit > 0.25 else 1.0 if deficit > 0.08 else 0.5)

        focus: List[str] = []
        if metrics.expert_entropy > self.entropy_threshold:
            focus.append("uzman_cesitlendirme")
        if metrics.mean_uncertainty > 0.6:
            focus.append("belirsizlik_azaltma")
        if metrics.novelty > 0.6:
            focus.append("yeni_bilgi_pekitirme")
        if health < 0.4:
            focus.append("temel_kalite")

        return {
            "should_train": should_train,
            "steps": steps,
            "batch_size": None,     # eğitici donanıma göre belirler
            "lr": round(lr, 8),
            "priority_tags": list((quality_report or {}).get("tags") or []),
            "focus": focus,
            "rationale": (
                f"kalite {health:.2f} (eşik {self.quality_threshold}), "
                f"tampon {metrics.buffer_size} örnek, "
                f"kalite açığı {deficit:.2f}, "
                f"eğilim {'kötüleşiyor' if metrics.loss_trend > 0 else 'iyi/kayıtsız'}"
            ),
        }

    # ─── Raporlama ──────────────────────────────────────────────────────

    def maturity(self, model: Any) -> float:
        """Modelin olgunluk puanı."""
        growth = getattr(model, "growth", None)
        if growth is None:
            return 0.0
        try:
            return float(growth.maturity)
        except Exception:
            return 0.0

    def summary(self) -> Dict[str, Any]:
        """İnsan okur büyüme özeti."""
        applied = [entry for entry in self._history if entry.get("applied")]
        return {
            "decisions": self._counter["decisions"],
            "applied": len(applied),
            "expert_growths": self._counter["expert_growths"],
            "layer_growths": self._counter["layer_growths"],
            "freezes": self._counter["freezes"],
            "prunes": self._counter["prunes"],
            "last_applied": [entry["action"] for entry in applied[-5:]],
        }

    def health(self) -> Dict[str, Any]:
        """Yönetici sağlık raporu."""
        issues: List[str] = []
        if not self._initialized:
            issues.append("yönetici başlatılmadı")
        if self._counter["rejected"] > self._counter["applied"]:
            issues.append("reddedilen karar sayısı uygulanandan fazla")
        return {
            "ok": self._initialized,
            "decisions": self._counter["decisions"],
            "applied": self._counter["applied"],
            "issues": issues,
        }

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
            f"<GrowthController decisions={self._counter['decisions']} "
            f"applied={self._counter['applied']}>"
        )


__all__ = [
    "GrowthController",
    "GrowthAction",
    "GrowthDecision",
    "GrowthMetrics",
    "ACTION_COST",
]
