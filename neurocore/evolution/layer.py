"""
NeuroCore Evrim Katmanı (L8 · Evolution) — Orkestrasyon
=========================================================
Sekiz alt sistemi tek bir **öğrenme döngüsünde** birleştirir:

    gözlemle → değerlendir → denetle → merak et → sentezle → büyü → doğrula → günlüğe yaz

**Gevşek bağlılık sözleşmesi**: Bu katman diğer NeuroCore katmanlarını import
etmez. Dış dünyaya erişimin TEK yolu `EvolutionHooks` üzerinden enjekte edilen
geri çağrılardır. Bir kanca yoksa ilgili yetenek atlanır ve gerekçesi Türkçe
olarak `EvolutionReport.skipped` içine yazılır. Böylece katman hem testlerde
tek başına hem de motorda tam bağlı çalışır.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from neurocore.utils.logger import NeuroCoreLogger

from neurocore.evolution.curiosity import CuriosityEngine, CuriosityItem
from neurocore.evolution.curriculum import CurriculumBuilder, CurriculumItem
from neurocore.evolution.evaluator import QualityReport, SelfEvaluator
from neurocore.evolution.experience import (
    Experience,
    ExperienceBuffer,
    FeedbackSignal,
    resolve_data_path,
    to_relative_display,
)
from neurocore.evolution.growth import GrowthAction, GrowthController, GrowthDecision
from neurocore.evolution.integrity import IntegrityGuard, IntegrityIssue
from neurocore.evolution.synthesizer import KnowledgeSynthesizer, SynthesisResult

logger = NeuroCoreLogger.get_logger("evolution.layer")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: Varsayılan veri yolu (konfigürasyon çözülemezse)
DEFAULT_DATA_PATH = "data/evolution"

#: Günlük dosyası adı
JOURNAL_FILE = "journal.md"

#: Bir turda en fazla kaç deneyim değerlendirilir
DEFAULT_EVAL_LIMIT = 50

#: Karar verirken kullanılan varsayılan not sayısı sınırı
DEFAULT_MAX_NOTES = 500

#: Kalite "iyi" sayılan eşik
GOOD_QUALITY = 0.6

#: Birleştirmeye değer eşiği (synthesizer ile aynı)
MERGE_WORTHY = 0.75

#: `EvolutionHooks` alan adları (kanca envanteri)
HOOK_NAMES: Tuple[str, ...] = (
    "list_notes", "get_note", "upsert_note", "link_notes", "archive_note",
    "graph_edges", "embed_text", "train_step", "model_metrics",
    "resource_budget", "web_research", "now",
)


# ─── Enjeksiyon Sözleşmesi ─────────────────────────────────────────────────

@dataclass
class EvolutionHooks:
    """
    Evrim katmanının dış dünyaya erişmesinin TEK yolu (gevşek bağlılık).

    Tüm alanlar `None` olabilir; her kullanım `hooks.has("ad")` ile korunur.
    Bir kanca yoksa ilgili yetenek atlanır ve Türkçe gerekçesiyle raporlanır.

    Attributes:
        list_notes: `(limit) -> not yükleri`
        get_note: `(note_id) -> not yükü ya da None`
        upsert_note: `(title, body, tags, confidence, sources, kind) -> not yükü`
        link_notes: `(src_id, dst_id, reason) -> bool`
        archive_note: `(note_id, reason) -> bool`
        graph_edges: `() -> {kaynak_id: [hedef_id, ...]}`
        embed_text: `(text) -> vektör ya da None`
        train_step: `(batch) -> {"loss": ..., "steps": ...}`
        model_metrics: `() -> {"expert_entropy": ..., ...}`
        resource_budget: `() -> ResourceBudget.to_dict()`
        web_research: `(query, depth) -> [bulgu, ...]`
        now: `() -> unix zaman damgası`
    """

    list_notes: Optional[Callable[[int], List[dict]]] = None
    get_note: Optional[Callable[[str], Optional[dict]]] = None
    upsert_note: Optional[Callable[..., dict]] = None
    link_notes: Optional[Callable[[str, str, str], bool]] = None
    archive_note: Optional[Callable[[str, str], bool]] = None
    graph_edges: Optional[Callable[[], Dict[str, List[str]]]] = None
    embed_text: Optional[Callable[[str], Optional[List[float]]]] = None
    train_step: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    model_metrics: Optional[Callable[[], Dict[str, Any]]] = None
    resource_budget: Optional[Callable[[], Dict[str, Any]]] = None
    web_research: Optional[Callable[[str, int], List[dict]]] = None
    now: Optional[Callable[[], float]] = None

    def available(self) -> List[str]:
        """Sağlanmış kanca adlarını döndürür."""
        return [name for name in HOOK_NAMES if getattr(self, name, None) is not None]

    def has(self, name: str) -> bool:
        """Belirtilen kanca sağlanmış mı?"""
        return getattr(self, name, None) is not None


# ─── Evrim Raporu ──────────────────────────────────────────────────────────

@dataclass
class EvolutionReport:
    """
    Tek bir evrim turunun raporu.

    Attributes:
        cycle: Tur numarası
        started_at / finished_at: Unix zaman damgaları
        evaluated: Değerlendirilen deneyim sayısı
        synthesized: Üretilen sentez notu sayısı
        curated: Müfredata alınan öğe sayısı
        grow_decisions: Verilen büyüme kararı sayısı
        lessons: Turdan çıkarılan dersler
        curiosity: Merak öğeleri (dict)
        applied: Uygulanan işlemler
        skipped: Atlanan yetenekler (Türkçe gerekçeleriyle)
        quality_before / quality_after: Tur öncesi/sonrası ortalama kalite
        integrity_score: Bilgi bütünlüğü sağlık puanı
        blocked: Tur ilerleyemedi mi
        block_reason: Engellenme gerekçesi
    """

    cycle: int
    started_at: float
    finished_at: float = 0.0
    evaluated: int = 0
    synthesized: int = 0
    curated: int = 0
    grow_decisions: int = 0
    lessons: List[str] = field(default_factory=list)
    curiosity: List[dict] = field(default_factory=list)
    applied: List[dict] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    quality_before: float = 0.0
    quality_after: float = 0.0
    integrity_score: float = 1.0
    blocked: bool = False
    block_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dict."""
        return {
            "cycle": int(self.cycle),
            "started_at": round(float(self.started_at), 3),
            "finished_at": round(float(self.finished_at), 3),
            "duration_ms": round(max(0.0, float(self.finished_at) - float(self.started_at)) * 1000.0, 1),
            "evaluated": int(self.evaluated),
            "synthesized": int(self.synthesized),
            "curated": int(self.curated),
            "grow_decisions": int(self.grow_decisions),
            "lessons": list(self.lessons),
            "curiosity": list(self.curiosity),
            "applied": list(self.applied),
            "skipped": list(self.skipped),
            "quality_before": round(float(self.quality_before), 4),
            "quality_after": round(float(self.quality_after), 4),
            "integrity_score": round(float(self.integrity_score), 4),
            "blocked": bool(self.blocked),
            "block_reason": self.block_reason,
        }

    def to_markdown(self) -> str:
        """Obsidian uyumlu günlük girdisi üretir."""
        status = "ENGELLENDİ" if self.blocked else "TAMAM"
        lines: List[str] = []
        lines.append(f"## Tur {self.cycle} — {status}")
        lines.append("")
        lines.append(
            f"- Süre: {max(0.0, float(self.finished_at) - float(self.started_at)) * 1000.0:.0f} ms"
        )
        lines.append(f"- Değerlendirilen deneyim: {self.evaluated}")
        lines.append(f"- Sentezlenen not: {self.synthesized}")
        lines.append(f"- Müfredat öğesi: {self.curated}")
        lines.append(f"- Büyüme kararı: {self.grow_decisions}")
        lines.append(
            f"- Kalite: {self.quality_before:.3f} → {self.quality_after:.3f} "
            f"(bütünlük: {self.integrity_score:.3f})"
        )
        if self.blocked and self.block_reason:
            lines.append(f"- ⚠ Engellenme: {self.block_reason}")

        if self.curiosity:
            lines.append("")
            lines.append("### Merak Konuları")
            for item in self.curiosity[:8]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- **{item.get('topic', '?')}** ({item.get('kind', '?')}, "
                    f"öncelik {float(item.get('priority', 0.0)):.2f}) — {item.get('reason', '')}"
                )

        if self.applied:
            lines.append("")
            lines.append("### Uygulananlar")
            for entry in self.applied[:12]:
                if not isinstance(entry, dict):
                    continue
                label = entry.get("kind") or entry.get("op") or entry.get("action") or "işlem"
                target = entry.get("target") or entry.get("title") or entry.get("subject") or ""
                note = entry.get("reason") or entry.get("detail") or ""
                lines.append(f"- `{label}` {target} {('— ' + str(note)) if note else ''}".rstrip())

        if self.lessons:
            lines.append("")
            lines.append("### Dersler")
            for lesson in self.lessons[:10]:
                lines.append(f"- {lesson}")

        if self.skipped:
            lines.append("")
            lines.append("### Atlananlar")
            for entry in self.skipped[:12]:
                lines.append(f"- {entry}")

        lines.append("")
        return "\n".join(lines)


# ─── Evrim Katmanı ─────────────────────────────────────────────────────────

class EvolutionLayer:
    """
    Öz-büyüme motorunun cephesi (L8).

    Kullanım (tek başına):
        layer = EvolutionLayer()
        layer.initialize(None, None)
        layer.observe("GPU neden ısınır?", "…", tags=["gpu"])
        report = layer.cycle()

    Kullanım (motorda):
        layer.initialize(config, bus, hooks=EvolutionHooks(list_notes=…, upsert_note=…))
        layer.evolve(cycles=3)
    """

    name = "evolution"

    def __init__(self) -> None:
        self._config: Any = None
        self._event_bus: Any = None
        self._hooks: EvolutionHooks = EvolutionHooks()

        # Yollar
        self._data_path: str = DEFAULT_DATA_PATH
        self._dir: Optional[Path] = None
        self._journal_path: Optional[Path] = None

        # Ayarlar
        self._buffer_size: int = 10000
        self._min_samples: int = 100
        self._auto_expand_threshold: float = 0.8
        self._prune_threshold: float = 0.05
        self._expansion_strategy: str = "curiosity_driven"
        self._embedding_dim: int = 768
        self._max_experts: int = 12
        self._base_path: str = ""
        self._hardware_profile: str = "high"

        # Alt sistemler
        self._experiences: ExperienceBuffer = ExperienceBuffer()
        self._evaluator: SelfEvaluator = SelfEvaluator()
        self._curriculum: CurriculumBuilder = CurriculumBuilder()
        self._synthesizer: KnowledgeSynthesizer = KnowledgeSynthesizer()
        self._curiosity: CuriosityEngine = CuriosityEngine()
        self._growth: GrowthController = GrowthController()
        self._integrity: IntegrityGuard = IntegrityGuard()

        # Döngü durumu
        self._cycle: int = 0
        self._reports: List[dict] = []
        self._lesson_pool: List[str] = []
        self._counters: Dict[str, float] = {
            "observations": 0.0,
            "feedbacks": 0.0,
            "cycles": 0.0,
            "blocked_cycles": 0.0,
            "step_failures": 0.0,
        }
        self._last_audit: Optional[Dict[str, Any]] = None
        self._pending_contradictions: List[dict] = []
        self._pending_curiosity: List[CuriosityItem] = []
        self._last_synthesis_note: str = ""
        self._last_plan: Dict[str, Any] = {}
        self._last_health: Dict[str, Any] = {}
        self._initialized: bool = False

    # ═══ Yaşam Döngüsü ═════════════════════════════════════════════════

    def initialize(
        self,
        config: Any = None,
        event_bus: Any = None,
        hooks: Optional[EvolutionHooks] = None,
    ) -> None:
        """
        Katmanı bağla ve tüm alt sistemleri başlat. İdempotenttir.

        Args:
            config: `SystemConfig` (None olabilir — varsayılanlar kullanılır)
            event_bus: `EventBus` (None olabilir — olaylar sessizce atlanır)
            hooks: `EvolutionHooks` (None → katman tek başına çalışır)
        """
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus
        self._hooks = hooks if isinstance(hooks, EvolutionHooks) else EvolutionHooks()

        self._read_config(config)
        self._data_path = self._resolve_data_path(config)
        self._dir = resolve_data_path(self._data_path, DEFAULT_DATA_PATH)
        self._journal_path = self._dir / JOURNAL_FILE

        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Evrim dizini oluşturulamadı: {exc}")

        # Alt sistemleri başlat (her biri kendi hatasını yönetir)
        for subsystem in (
            self._experiences, self._evaluator, self._curriculum,
            self._synthesizer, self._curiosity, self._growth, self._integrity,
        ):
            try:
                subsystem.initialize(config, event_bus)
            except Exception as exc:
                logger.error(f"Alt sistem başlatılamadı ({subsystem.name}): {exc}")

        # Donanım bütçesi varsa büyüme beklentilerini uyarla
        if self._hooks.has("resource_budget"):
            self._refresh_budget()

        self._initialized = True
        logger.info(
            f"Evrim katmanı hazır · tampon={self._buffer_size} · min_örnek={self._min_samples} · "
            f"strateji={self._expansion_strategy} · kancalar="
            f"{', '.join(self._hooks.available()) or 'yok'}"
        )
        self._emit("evolution.ready", {
            "buffer_size": self._buffer_size,
            "embedding_dim": self._embedding_dim,
            "hooks": self._hooks.available(),
            "data_path": str(self._dir),
        })

    def shutdown(self) -> None:
        """Tüm alt sistemleri kapat. İdempotenttir."""
        if not self._initialized:
            return
        for subsystem in (
            self._experiences, self._evaluator, self._curriculum,
            self._synthesizer, self._curiosity, self._growth, self._integrity,
        ):
            try:
                subsystem.shutdown()
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Alt sistem kapatılamadı ({subsystem.name}): {exc}")
        self._initialized = False
        logger.info(f"Evrim katmanı kapatıldı ({self._cycle} tur tamamlandı)")

    # ─── Konfigürasyon ─────────────────────────────────────────────────

    def _read_config(self, config: Any) -> None:
        """Konfigürasyon değerlerini güvenli biçimde okur (None toleranslı)."""
        evolution = getattr(config, "evolution", None)
        neural = getattr(config, "neural", None)

        def _int(source: Any, key: str, default: int) -> int:
            try:
                return int(getattr(source, key, default))
            except (TypeError, ValueError):
                return default

        def _float(source: Any, key: str, default: float) -> float:
            try:
                return float(getattr(source, key, default))
            except (TypeError, ValueError):
                return default

        self._buffer_size = max(1, _int(evolution, "learning_buffer_size", 10000))
        self._min_samples = max(1, _int(evolution, "min_samples_for_training", 100))
        self._auto_expand_threshold = _float(evolution, "auto_expand_threshold", 0.8)
        self._prune_threshold = _float(evolution, "prune_threshold", 0.05)
        strategy = str(getattr(evolution, "expansion_strategy", "curiosity_driven") or "").strip().lower()
        self._expansion_strategy = strategy or "curiosity_driven"

        self._embedding_dim = max(16, _int(neural, "embedding_dim", 768))
        self._max_experts = max(1, _int(neural, "max_experts", 12))

        profile = getattr(config, "hardware_profile", "high") if config is not None else "high"
        self._hardware_profile = str(getattr(profile, "value", profile) or "high")
        self._base_path = str(getattr(config, "base_path", "") or "") if config is not None else ""

    @staticmethod
    def _resolve_data_path(config: Any) -> str:
        """
        Katman veri kökünü çöz: `config.data_path` → `<base_path>/data/evolution`
        → proje kökündeki `config.yaml` → `data/evolution`.

        Günlük dosyası bu yolun altında (`journal.md`) tutulur; deneyim tamponu
        ve büyüme durumu kendi alt dizinlerini (`experience/`, `growth/`) açar.
        """
        explicit = getattr(config, "data_path", None) if config is not None else None
        if explicit:
            return str(explicit)
        base = getattr(config, "base_path", None) if config is not None else None
        if base:
            return str(Path(str(base)) / "data" / "evolution")
        try:
            project_root = Path(__file__).resolve().parents[2]
            cfg_file = project_root / "config.yaml"
            if cfg_file.exists():
                import yaml

                data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
                if data.get("data_path"):
                    return str(data["data_path"])
                if data.get("base_path"):
                    return str(Path(str(data["base_path"])) / "data" / "evolution")
        except Exception:
            pass
        return DEFAULT_DATA_PATH

    # ─── Yardımcılar ───────────────────────────────────────────────────

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Olay yayınla (bus yoksa atlanır)."""
        if self._event_bus:
            try:
                self._event_bus.emit(event_type, data, source=self.name)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Olay yayınlanamadı ({event_type}): {exc}")

    def _now(self) -> float:
        """Zaman kaynağı (kanca varsa ondan, yoksa sistem saati)."""
        if self._hooks.has("now"):
            try:
                return float(self._hooks.now())
            except Exception:
                pass
        return time.time()

    @staticmethod
    def _skip(report: "EvolutionReport", reason: str) -> None:
        """Atlanan yeteneği rapora yazar (tekrarsız)."""
        if reason and reason not in report.skipped:
            report.skipped.append(reason)

    def _call_hook(self, name: str, *args: Any, default: Any = None) -> Any:
        """
        Kancayı güvenle çağırır.

        Args:
            name: Kanca adı
            *args: Kancaya geçilecek argümanlar
            default: Kanca yoksa ya da hata verirse dönecek değer

        Returns:
            Kanca çıktısı ya da `default`
        """
        hook = getattr(self._hooks, name, None)
        if hook is None:
            return default
        try:
            return hook(*args)
        except Exception as exc:
            logger.warning(f"Kanca hatası ({name}): {exc}")
            return default

    @staticmethod
    def _context_of(experience: Experience, limit: int = 5) -> List[str]:
        """Deneyimin bağlam referanslarını metne çevirir."""
        context: List[str] = []
        for entry in (experience.context_used or [])[:limit]:
            if isinstance(entry, str):
                context.append(entry)
            elif hasattr(entry, "body"):
                context.append(str(getattr(entry, "body")))
            elif isinstance(entry, dict):
                context.append(str(entry.get("body") or entry.get("text") or entry.get("id") or ""))
            else:
                context.append(str(entry))
        return [c for c in context if c]

    def _refresh_budget(self) -> None:
        """Donanım bütçesini kancadan çekip büyüme denetleyicisine bildirir."""
        if not self._hooks.has("resource_budget"):
            return
        try:
            budget = self._hooks.resource_budget()
            if isinstance(budget, dict):
                self._growth.update_from_budget(budget)
        except Exception as exc:
            logger.warning(f"Donanım bütçesi alınamadı: {exc}")

    # ═══ Erişimciler ═══════════════════════════════════════════════════

    @property
    def experiences(self) -> ExperienceBuffer:
        """Deneyim tamponu."""
        return self._experiences

    @property
    def evaluator(self) -> SelfEvaluator:
        """Öz-değerlendirici."""
        return self._evaluator

    @property
    def curriculum(self) -> CurriculumBuilder:
        """Müfredat oluşturucu."""
        return self._curriculum

    @property
    def synthesizer(self) -> KnowledgeSynthesizer:
        """Bilgi sentezleyici."""
        return self._synthesizer

    @property
    def curiosity(self) -> CuriosityEngine:
        """Merak motoru."""
        return self._curiosity

    @property
    def growth(self) -> GrowthController:
        """Büyüme denetleyicisi."""
        return self._growth

    @property
    def integrity(self) -> IntegrityGuard:
        """Bütünlük bekçisi."""
        return self._integrity

    @property
    def hooks(self) -> EvolutionHooks:
        """Enjekte edilmiş kancalar."""
        return self._hooks

    # ═══ Gözlem ════════════════════════════════════════════════════════

    def observe(
        self,
        query: str,
        answer: str,
        *,
        context: Optional[List[str]] = None,
        latency_ms: float = 0.0,
        uncertainty: float = 0.5,
        strategy: str = "",
        tags: Optional[List[str]] = None,
        hardware: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Bir etkileşimi deneyim olarak kaydeder ve hemen değerlendirir.

        Args:
            query: Kullanıcı sorgusu
            answer: Üretilen yanıt
            context: Kullanılan bağlam referansları
            latency_ms: Yanıt gecikmesi (ms)
            uncertainty: Model belirsizliği (0..1)
            strategy: İzlenen yol (planlayıcı şablonu / rota)
            tags: Etiketler
            hardware: Donanım anlık görüntüsü (yoksa profil/cihaz doldurulur)

        Returns:
            Deneyim ve değerlendirme özeti içeren dict
        """
        hardware_info = dict(hardware or {})
        hardware_info.setdefault("profile", self._hardware_profile)
        hardware_info.setdefault("device", self._device())

        experience = self._experiences.log(
            query,
            answer,
            context_used=list(context or []),
            uncertainty=float(uncertainty or 0.0),
            latency_ms=float(latency_ms or 0.0),
            strategy=str(strategy or ""),
            tags=list(tags or []),
            hardware=hardware_info,
        )
        self._counters["observations"] += 1.0

        report = self._evaluate_experience(experience, list(context or []))

        self._emit("evolution.observed", {
            "id": experience.id,
            "quality": experience.quality,
            "verdict": report.verdict if report else "unknown",
            "tags": list(experience.tags),
        })
        return {
            "id": experience.id,
            "quality": experience.quality,
            "verdict": report.verdict if report else "unknown",
            "components": report.to_dict().get("components", {}) if report else {},
            "issues": list(report.issues) if report else [],
            "tags": list(experience.tags),
            "buffer_size": len(self._experiences),
        }

    def _device(self) -> str:
        """Bütçeden cihaz bilgisini çıkarır."""
        if self._hooks.has("resource_budget"):
            try:
                budget = self._hooks.resource_budget() or {}
                return str(budget.get("device") or "cpu")
            except Exception:
                return "cpu"
        return "cpu"

    def _evaluate_experience(
        self,
        experience: Experience,
        context: Optional[List[str]] = None,
    ) -> Optional[QualityReport]:
        """
        Deneyimi değerlendirir ve `quality` alanını günceller.

        Bağlam metni kancadan (`get_note`) zenginleştirilir; kanca yoksa
        yalnızca deneyimdeki referanslar kullanılır.
        """
        context_items = list(context) if context else list(experience.context_used or [])
        if self._hooks.has("get_note"):
            enriched: List[str] = []
            for reference in (experience.context_used or [])[:5]:
                try:
                    note = self._hooks.get_note(str(reference))
                except Exception:
                    note = None
                if isinstance(note, dict):
                    enriched.append(str(note.get("body") or note.get("title") or ""))
            if enriched:
                context_items = enriched + [str(c) for c in context_items]

        try:
            report = self._evaluator.evaluate(
                query=experience.query,
                answer=experience.answer,
                context=[str(c) for c in context_items],
                uncertainty=experience.uncertainty,
                latency_ms=experience.latency_ms,
            )
        except Exception as exc:
            logger.warning(f"Deneyim değerlendirilemedi ({experience.id}): {exc}")
            return None

        experience.quality = float(report.score)
        experience.metadata["quality_verdict"] = report.verdict
        experience.metadata["quality_issues"] = list(report.issues)
        return report

    def feedback(
        self,
        experience_id: str,
        feedback: Any,
        reward: Optional[float] = None,
        note: str = "",
    ) -> dict:
        """
        Bir deneyime geri bildirim yazar.

        Args:
            experience_id: Hedef deneyim kimliği
            feedback: `FeedbackSignal` ya da metin karşılığı
            reward: Açık ödül (None → sinyalin varsayılanı)
            note: Serbest not

        Returns:
            `{"ok": bool, "id": ..., "feedback": ..., "reward": ...}`
        """
        if isinstance(feedback, str):
            try:
                signal = FeedbackSignal(feedback)
            except ValueError:
                logger.warning(f"Bilinmeyen geri bildirim: {feedback}")
                return {"ok": False, "id": experience_id, "error": "bilinmeyen_sinyal"}
        elif isinstance(feedback, FeedbackSignal):
            signal = feedback
        else:
            return {"ok": False, "id": experience_id, "error": "gecersiz_sinyal"}

        ok = self._experiences.record_feedback(experience_id, signal, reward, note)
        if ok:
            self._counters["feedbacks"] += 1.0

        experience = self._experiences.get(experience_id)
        self._emit("evolution.feedback", {
            "id": experience_id,
            "feedback": signal.value,
            "reward": experience.reward if experience else 0.0,
            "ok": bool(ok),
        })
        return {
            "ok": bool(ok),
            "id": experience_id,
            "feedback": signal.value,
            "reward": experience.reward if experience else 0.0,
            "priority": experience.priority if experience else 0.0,
        }

    # ═══ Ana Döngü ═════════════════════════════════════════════════════

    def cycle(
        self,
        cycles: int = 1,
        *,
        dry_run: bool = False,
        max_notes: int = DEFAULT_MAX_NOTES,
        auto_apply: bool = True,
    ) -> EvolutionReport:
        """
        Evrim turunu çalıştırır.

        Adımlar: (1) değerlendir → (2) bütünlük denetimi → (3) merak taraması →
        (4) sentez/bağlama → (5) müfredat + büyüme kararı → (6) yeniden denetim →
        (7) günlük. Her adım kendi try/except'i içinde çalışır; bir adımın
        çökmesi turu durdurmaz. Yalnızca 2. ve 5. adımlar **birlikte** başarısız
        olursa tur `blocked` işaretlenir.

        Args:
            cycles: Kaç tur çalıştırılacak (≥1)
            dry_run: True → hiçbir yazma işlemi uygulanmaz
            max_notes: Turda işlenecek en fazla not sayısı
            auto_apply: True → büyüme kararları ve sentezler uygulanır

        Returns:
            Son turun `EvolutionReport` nesnesi
        """
        total = max(1, int(cycles or 1))
        limit = max(1, int(max_notes or DEFAULT_MAX_NOTES))
        report: Optional[EvolutionReport] = None

        for _ in range(total):
            self._cycle += 1
            self._counters["cycles"] += 1.0
            report = EvolutionReport(cycle=self._cycle, started_at=self._now())
            report.quality_before = self._mean_quality()
            failures: List[str] = []

            steps: List[Tuple[str, str, Callable[[], None]]] = [
                ("evaluate", "deneyim değerlendirme", lambda: self._step_evaluate(report)),
                ("audit", "bilgi denetimi", lambda: self._step_audit(report, limit)),
                ("curiosity", "merak taraması", lambda: self._step_curiosity(report)),
                ("synthesize", "bilgi sentezi", lambda: self._step_synthesize(report, limit, auto_apply, dry_run)),
                ("growth", "büyüme kararı", lambda: self._step_growth(report, auto_apply, dry_run)),
                ("verify", "yeniden denetim", lambda: self._step_verify(report)),
            ]

            for step_key, step_label, step in steps:
                try:
                    step()
                except Exception as exc:
                    self._counters["step_failures"] += 1.0
                    failures.append(step_key)
                    message = f"{step_label} başarısız: {exc}"
                    report.lessons.append(message)
                    logger.warning(message)
                    logger.debug(traceback.format_exc())

            # Yalnızca denetim VE büyüme birlikte çökerse tur engellenmiş sayılır
            if "audit" in failures and "growth" in failures:
                report.blocked = True
                report.block_reason = (
                    "Bilgi denetimi ve büyüme kararı birlikte başarısız oldu; "
                    "tur ilerleyemedi"
                )
                self._counters["blocked_cycles"] += 1.0

            report.quality_after = self._mean_quality()
            report.finished_at = self._now()

            self._collect_lessons(report)
            try:
                self.journal(report)
            except Exception as exc:
                logger.warning(f"Günlük yazılamadı: {exc}")

            self._reports.append(report.to_dict())
            if len(self._reports) > 100:
                self._reports = self._reports[-100:]

            self._emit("evolution.cycle_finished", {
                "cycle": report.cycle,
                "evaluated": report.evaluated,
                "synthesized": report.synthesized,
                "curated": report.curated,
                "grow_decisions": report.grow_decisions,
                "quality_after": report.quality_after,
                "integrity_score": report.integrity_score,
                "blocked": report.blocked,
                "skipped": report.skipped[:5],
            })
            logger.info(
                f"Evrim turu {report.cycle} tamam: değerlendirilen={report.evaluated} · "
                f"sentez={report.synthesized} · müfredat={report.curated} · "
                f"kalite={report.quality_after:.3f} · bütünlük={report.integrity_score:.3f}"
                + (" · ENGELLENDİ" if report.blocked else "")
            )

        assert report is not None  # total ≥ 1
        return report

    def evolve(self, cycles: int = 1, **kw: Any) -> dict:
        """
        `cycle()` çalıştırır ve raporu dict olarak döndürür.

        Args:
            cycles: Tur sayısı
            **kw: `cycle()` anahtar sözcük argümanları (dry_run, max_notes, auto_apply)
        """
        report = self.cycle(cycles, **kw)
        return report.to_dict()

    # ─── Adım 1: Değerlendirme ─────────────────────────────────────────

    def _step_evaluate(self, report: EvolutionReport) -> None:
        """Değerlendirilmemiş son deneyimleri puanlar."""
        pending = [
            experience for experience in self._experiences.recent(limit=DEFAULT_EVAL_LIMIT)
            if float(experience.quality) <= 0.0
        ]
        if not pending and not self._experiences.recent(1):
            self._skip(report, "Deneyim tamponu boş; değerlendirme atlandı")

        for experience in pending:
            if self._evaluate_experience(experience) is not None:
                report.evaluated += 1

    # ─── Adım 2: Bilgi Denetimi ────────────────────────────────────────

    def _step_audit(self, report: EvolutionReport, max_notes: int) -> None:
        """Kasa notlarını bütünlük ve çelişki açısından denetler."""
        if not self._hooks.has("list_notes"):
            self._skip(report, "Bilgi denetimi atlandı: 'list_notes' kancası yok")
            return

        notes = self._call_hook("list_notes", max_notes, default=[])
        if not isinstance(notes, list):
            notes = []
        graph = self._graph()
        issues, contradictions = self._audit_core(notes, graph)
        self._last_audit = {"notes": notes, "graph": graph}

        report.integrity_score = self._integrity.health_score(issues, len(notes))

        for issue in issues:
            if issue.kind == "contradiction" and issue.detail not in report.lessons:
                report.lessons.append(issue.detail)
        self._pending_contradictions = contradictions

    def _audit_core(self, notes: List[dict], graph: Dict[str, List[str]]) -> Tuple[List[IntegrityIssue], List[dict]]:
        """Denetimi ve çelişki tespitini birlikte çalıştırır."""
        issues: List[IntegrityIssue] = []
        contradictions: List[dict] = []
        try:
            contradictions = self._synthesizer.find_contradictions(notes)
        except Exception as exc:
            logger.warning(f"Çelişki tespiti başarısız: {exc}")
        try:
            issues = self._integrity.audit(notes, graph, contradictions)
        except Exception as exc:
            logger.warning(f"Bütünlük denetimi başarısız: {exc}")
        return issues, contradictions

    def _graph(self) -> Dict[str, List[str]]:
        """Graf kenarlarını kancadan alır (yoksa boş)."""
        if not self._hooks.has("graph_edges"):
            return {}
        edges = self._call_hook("graph_edges", default={})
        return edges if isinstance(edges, dict) else {}

    # ─── Adım 3: Merak Taraması ────────────────────────────────────────

    def _step_curiosity(self, report: EvolutionReport) -> None:
        """Kasadaki boşlukları/çelişkileri tarar."""
        notes = list((self._last_audit or {}).get("notes") or [])
        graph = dict((self._last_audit or {}).get("graph") or {})
        contradictions = list(getattr(self, "_pending_contradictions", []) or [])

        if not notes:
            self._skip(report, "Merak taraması atlandı: not listesi yok ('list_notes' kancası)")

        items: List[CuriosityItem] = []
        try:
            items = self._curiosity.scan(notes, graph, max_items=25)
        except Exception as exc:
            logger.warning(f"Merak taraması başarısız: {exc}")

        if contradictions:
            try:
                items = items + self._curiosity.contradictions(notes, contradictions)
            except Exception as exc:
                logger.debug(f"Çelişki öğeleri üretilemedi: {exc}")

        items.sort(key=lambda item: (-float(item.priority), item.topic))
        report.curiosity = [item.to_dict() for item in items[:25]]
        self._pending_curiosity = items

    # ─── Adım 4: Sentez ────────────────────────────────────────────────

    def _step_synthesize(
        self,
        report: EvolutionReport,
        max_notes: int,
        auto_apply: bool,
        dry_run: bool,
    ) -> None:
        """Birleştirmeye değer çiftleri ve merak konularını sentezler."""
        notes = list((self._last_audit or {}).get("notes") or [])
        if not notes:
            self._skip(report, "Sentez atlandı: not listesi yok ('list_notes' kancası)")
            return

        budget = max(0, int(max_notes))

        # (a) Birleştirmeye değer çiftler
        candidates: List[Tuple[float, dict, dict]] = []
        sample = notes[:300]
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                try:
                    score = self._synthesizer.score_merge_worthiness(sample[i], sample[j])
                except Exception:
                    continue
                if score > MERGE_WORTHY:
                    candidates.append((score, sample[i], sample[j]))
        candidates.sort(key=lambda entry: -entry[0])

        for score, note_a, note_b in candidates[: max(0, budget - report.synthesized)]:
            if dry_run:
                report.applied.append({
                    "kind": "merge_preview",
                    "target": f"{note_a.get('id')}+{note_b.get('id')}",
                    "score": round(score, 4),
                    "reason": "kuru çalıştırma: sentez uygulanmadı",
                })
                continue
            applied = self._merge_notes(note_a, note_b, score, report)
            if applied:
                report.synthesized += 1

        # (b) Merak konuları
        for item in list(getattr(self, "_pending_curiosity", []) or []):
            if report.synthesized >= budget:
                break
            if item.kind not in ("gap", "unexplored_cluster"):
                continue
            if dry_run:
                report.applied.append({
                    "kind": "topic_preview",
                    "target": item.topic,
                    "reason": "kuru çalıştırma: konu sentezi uygulanmadı",
                })
                continue
            result = self._synthesize_topic(item.topic, None, notes, auto_apply)
            if result:
                report.synthesized += 1

        if dry_run:
            self._skip(report, "Sentez kuru çalıştırma modunda; yalnızca önizleme üretildi")

    def _merge_notes(
        self,
        note_a: dict,
        note_b: dict,
        score: float,
        report: EvolutionReport,
    ) -> bool:
        """İki notu sentezleyip yeni not yazar ve bağlantıları kurar."""
        if not self._hooks.has("upsert_note"):
            self._skip(report, "Not birleştirme atlandı: 'upsert_note' kancası yok")
            return False

        result = self._synthesizer.synthesize([note_a, note_b], kind="synthesis")
        if result is None:
            return False

        note_id = self._upsert(result)
        if not note_id:
            return False

        if self._hooks.has("link_notes"):
            for source in (note_a, note_b):
                try:
                    self._hooks.link_notes(str(note_id), str(source.get("id") or ""), "sentez")
                except Exception as exc:
                    logger.debug(f"Bağlantı kurulamadı: {exc}")
        else:
            self._skip(report, "Sentez bağlantıları atlandı: 'link_notes' kancası yok")

        report.applied.append({
            "kind": "merge",
            "target": str(note_id),
            "sources": [str(note_a.get("id") or ""), str(note_b.get("id") or "")],
            "score": round(float(score), 4),
            "confidence": round(float(result.confidence), 4),
            "reason": result.rationale,
        })
        self._emit("evolution.merge_applied", {
            "note": str(note_id),
            "sources": [note_a.get("id"), note_b.get("id")],
            "score": round(float(score), 4),
        })
        return True

    def _synthesize_topic(
        self,
        topic: str,
        note_ids: Optional[List[str]],
        notes: List[dict],
        auto_apply: bool,
    ) -> Optional[SynthesisResult]:
        """Konuya ilişkin notları sentezler ve (izin veriliyorsa) kasaya yazar."""
        selected: List[dict] = []

        if note_ids:
            wanted = {str(i) for i in note_ids}
            selected = [n for n in notes if str(n.get("id")) in wanted]
            if not selected and self._hooks.has("get_note"):
                for identifier in note_ids:
                    note = self._call_hook("get_note", str(identifier), default=None)
                    if isinstance(note, dict):
                        selected.append(note)
        else:
            needle = str(topic or "").strip().lower()
            if needle:
                selected = [
                    n for n in notes
                    if needle in str(n.get("title") or "").lower()
                    or needle in " ".join(str(t) for t in (n.get("tags") or [])).lower()
                ]
            if len(selected) < 2:
                terms = {t for t in needle.replace("-", " ").split() if len(t) > 2}
                if terms:
                    scored: List[Tuple[int, dict]] = []
                    for note in notes:
                        haystack = f"{note.get('title') or ''} {note.get('body') or ''} " \
                                   f"{' '.join(str(t) for t in (note.get('tags') or []))}".lower()
                        hits = sum(1 for term in terms if term in haystack)
                        if hits:
                            scored.append((hits, note))
                    scored.sort(key=lambda entry: -entry[0])
                    selected = [note for _, note in scored[:4]]

        if len(selected) < 2:
            return None
        if not auto_apply:
            return None
        if not self._hooks.has("upsert_note"):
            return None

        result = self._synthesizer.synthesize(selected, topic=topic, kind="synthesis")
        if result is None:
            return None

        note_id = self._upsert(result)
        self._last_synthesis_note = str(note_id or "")
        if note_id and self._hooks.has("link_notes"):
            for source in selected:
                try:
                    self._hooks.link_notes(str(note_id), str(source.get("id") or ""), "sentez")
                except Exception:
                    pass

        if note_id:
            self._emit("evolution.topic_synthesized", {
                "topic": topic, "note": str(note_id), "sources": len(selected),
            })
        return result

    def _upsert(self, result: SynthesisResult) -> Optional[str]:
        """Sentez sonucunu kancayla kasaya yazar; not kimliğini döndürür."""
        if not self._hooks.has("upsert_note"):
            return None
        payload = result.to_note_payload()
        try:
            stored = self._hooks.upsert_note(
                payload["title"],
                payload["body"],
                payload["tags"],
                payload["confidence"],
                payload["sources"],
                payload["kind"],
            )
        except Exception as exc:
            logger.warning(f"Sentez notu yazılamadı: {exc}")
            return None
        if isinstance(stored, dict):
            return str(stored.get("id") or stored.get("note_id") or "")
        return str(stored) if stored else None

    # ─── Adım 5: Büyüme ────────────────────────────────────────────────

    def _step_growth(self, report: EvolutionReport, auto_apply: bool, dry_run: bool) -> None:
        """Müfredatı kurar, büyüme kararını verir ve (izin veriliyorsa) uygular."""
        metrics = self.growth_metrics()
        buffer_stats = self._experiences.stats()

        items = self._curriculum.build(self._experiences, max_items=500)
        report.curated = len(items)

        decision = self._growth.decide(metrics, buffer_stats)
        report.grow_decisions = 1

        if decision.action == GrowthAction.NOOP:
            report.applied.append(decision.to_dict())
            report.lessons.append(f"Büyüme kararı yok: {decision.reason}")
            return

        if dry_run:
            preview = decision.to_dict()
            preview["dry_run"] = True
            preview["reason"] = f"kuru çalıştırma: {decision.reason}"
            report.applied.append(preview)
            return

        if not auto_apply:
            report.applied.append(decision.to_dict())
            self._skip(report, "Büyüme kararı uygulanmadı: auto_apply=False")
            return

        applied = self._growth.apply(decision, self._growth_apply_fn(metrics))
        entry = decision.to_dict()
        entry["applied"] = bool(applied)
        report.applied.append(entry)

        if not applied:
            if not self._hooks.has("train_step") and decision.action in (
                GrowthAction.RETRAIN, GrowthAction.EXPAND_EXPERTS,
            ):
                self._skip(
                    report,
                    "Büyüme uygulanamadı: 'train_step' kancası yok (yalnızca karar kaydedildi)",
                )
            else:
                self._skip(report, "Model büyüme kararını uygulamadı")

        # Eğitim adımı ayrıca yürütülebilir (kanca varsa)
        if applied and decision.action == GrowthAction.RETRAIN and self._hooks.has("train_step"):
            self._run_training_step(report, metrics)

    def _growth_apply_fn(self, metrics: Dict[str, Any]) -> Optional[Callable[[GrowthDecision], bool]]:
        """Büyüme kararını modele uygulayan geri çağrıyı üretir (kanca varsa)."""
        if not self._hooks.has("train_step"):
            return None

        def _apply(decision: GrowthDecision) -> bool:
            batch = {
                "action": decision.action.value,
                "magnitude": decision.magnitude,
                "details": dict(decision.details),
                "metrics": dict(metrics),
            }
            outcome = self._call_hook("train_step", batch, default=None)
            return bool(outcome)

        return _apply

    def _run_training_step(self, report: EvolutionReport, metrics: Dict[str, Any]) -> None:
        """`train_step` kancasıyla gerçek bir eğitim adımı yürütür."""
        batch = self._experiences.sample_batch(
            batch_size=max(1, int(self._last_plan.get("batch_size", 32))),
            embedding_dim=self._embedding_dim,
        )
        if not batch.get("inputs"):
            self._skip(report, "Eğitim adımı atlandı: tamponda örnek yok")
            return
        result = self._call_hook("train_step", batch, default={})
        if isinstance(result, dict) and result:
            report.applied.append({
                "kind": "train_step",
                "target": "neural",
                "loss": result.get("loss"),
                "steps": result.get("steps"),
                "reason": "öncelikli örneklenmiş batch ile eğitim adımı",
            })
            self._growth.record(
                GrowthDecision(
                    action=GrowthAction.RETRAIN,
                    reason="Eğitim adımı yürütüldü",
                    details={"steps": int(result.get("steps", 1) or 1)},
                    confidence=0.6,
                ),
                applied=True,
                metrics=metrics,
            )
        else:
            self._skip(report, "Eğitim adımı sonuç döndürmedi ('train_step' kancası)")

    # ─── Adım 6: Doğrulama ─────────────────────────────────────────────

    def _step_verify(self, report: EvolutionReport) -> None:
        """Sentez sonrası durumu yeniden denetler."""
        if not self._hooks.has("list_notes"):
            self._skip(report, "Yeniden denetim atlandı: 'list_notes' kancası yok")
            return
        notes = self._call_hook("list_notes", DEFAULT_MAX_NOTES, default=[])
        if not isinstance(notes, list):
            notes = []
        graph = self._graph()
        issues, _ = self._audit_core(notes, graph)
        report.integrity_score = self._integrity.health_score(issues, len(notes))
        self._last_audit = {"notes": notes, "graph": graph}

    # ─── Dersler ───────────────────────────────────────────────────────

    def _collect_lessons(self, report: EvolutionReport) -> None:
        """Turdan dersler çıkarır ve kalıcı havuza ekler."""
        if report.quality_after and report.quality_after < GOOD_QUALITY:
            report.lessons.append(
                f"Ortalama kalite {report.quality_after:.2f} < {GOOD_QUALITY:.2f}; "
                "müfredatta düşük kaliteli örneklere ağırlık verilmeli"
            )
        if report.integrity_score < 0.8:
            report.lessons.append(
                f"Bilgi bütünlüğü {report.integrity_score:.2f}; çelişki ve yinelenmeler temizlenmeli"
            )
        if report.evaluated == 0 and len(self._experiences) == 0:
            report.lessons.append("Henüz deneyim yok; gözlem toplanmadan evrim ilerleyemez")
        if report.synthesized == 0 and report.curiosity:
            report.lessons.append(
                f"{len(report.curiosity)} merak konusu bulundu ama sentez üretilemedi"
            )

        seen: set = set()
        unique: List[str] = []
        for lesson in report.lessons:
            text = str(lesson).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique.append(text)
        report.lessons = unique

        self._lesson_pool.extend(unique)
        if len(self._lesson_pool) > 200:
            self._lesson_pool = self._lesson_pool[-200:]

    def lessons(self, limit: int = 10) -> List[str]:
        """Son turlardan çıkarılan dersleri döndürür (en yeniden eskiye)."""
        count = max(0, int(limit or 0))
        if count == 0:
            return []
        return list(reversed(self._lesson_pool[-count:]))

    # ─── Metrikler ─────────────────────────────────────────────────────

    def _mean_quality(self) -> float:
        """Son deneyimlerin ortalama kalitesi."""
        try:
            recent = self._experiences.recent(limit=DEFAULT_EVAL_LIMIT)
        except Exception:
            return 0.0
        scores = [float(e.quality) for e in recent if float(e.quality) > 0.0]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def growth_metrics(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Büyüme kararı için metrikleri toplar.

        `model_metrics` kancası yoksa yalnızca deneyim tamponundan türetilen
        metrikler döner (uzman/entropi bilgisi olmadan büyüme kararı NOOP'a
        kayar — bu beklenen davranıştır).
        """
        metrics: Dict[str, Any] = {}

        if self._hooks.has("model_metrics"):
            try:
                live = self._hooks.model_metrics()
                if isinstance(live, dict):
                    metrics.update(live)
            except Exception as exc:
                logger.warning(f"Model metrikleri alınamadı: {exc}")
        else:
            metrics.setdefault("model_metrics_available", False)

        stats = self._experiences.stats()
        metrics.setdefault("buffer_size", int(stats.get("count") or 0))
        metrics.setdefault("mean_quality", float(stats.get("mean_quality") or 0.0))

        recent = self._experiences.recent(limit=50)
        if recent:
            uncertainties = [float(e.uncertainty) for e in recent]
            metrics.setdefault(
                "mean_uncertainty", round(sum(uncertainties) / len(uncertainties), 4)
            )
            rewards = [abs(float(e.reward)) for e in recent]
            metrics.setdefault("mean_abs_reward", round(sum(rewards) / len(rewards), 4))
            metrics.setdefault("feedback_rate", round(
                sum(1 for e in recent if e.feedback != FeedbackSignal.NONE) / len(recent), 4
            ))

        metrics.setdefault("maturity", self._growth.maturity())
        metrics.setdefault("num_experts", self._growth.state().num_experts)
        metrics.setdefault("max_experts", self._growth.state().max_experts)
        metrics.setdefault("generation", self._growth.state().generation)
        metrics.setdefault("strategy", self._expansion_strategy)

        if extra:
            metrics.update(extra)
        return metrics

    # ─── Büyüme Arayüzü ────────────────────────────────────────────────

    def propose_growth(self, metrics: Optional[Dict[str, Any]] = None) -> dict:
        """
        Büyüme kararını üretir (uygulamaz).

        Args:
            metrics: Metrikler (None → `growth_metrics()`)

        Returns:
            `GrowthDecision.to_dict()` — `reason` her zaman Türkçedir
        """
        payload = dict(metrics) if isinstance(metrics, dict) else self.growth_metrics()
        if self._hooks.has("resource_budget"):
            self._refresh_budget()
        try:
            decision = self._growth.decide(payload, self._experiences.stats())
        except Exception as exc:
            logger.warning(f"Büyüme önerisi üretilemedi: {exc}")
            decision = GrowthDecision(
                action=GrowthAction.NOOP,
                reason=f"Karar üretilemedi: {exc}",
                confidence=0.0,
            )
        return decision.to_dict()

    def apply_growth(self, decision: dict, dry_run: bool = False) -> dict:
        """
        Verilen büyüme kararını uygular.

        Args:
            decision: `GrowthDecision.to_dict()` çıktısı
            dry_run: True → yalnızca doğrular, uygulamaz

        Returns:
            `{"applied": bool, "action": ..., "reason": ..., "state": {...}}`
        """
        parsed = GrowthDecision.from_dict(decision)
        if dry_run:
            parsed.dry_run = True
            return {
                "applied": False,
                "dry_run": True,
                "action": parsed.action.value,
                "reason": f"kuru çalıştırma: {parsed.reason}",
                "state": self._growth.state().to_dict(),
            }

        metrics = self.growth_metrics()
        applied = self._growth.apply(parsed, self._growth_apply_fn(metrics))
        return {
            "applied": bool(applied),
            "dry_run": False,
            "action": parsed.action.value,
            "reason": parsed.reason,
            "state": self._growth.state().to_dict(),
        }

    def training_plan(self, quality_report: Optional[Dict[str, Any]] = None) -> dict:
        """
        Eğitim planını üretir.

        Args:
            quality_report: `QualityReport.to_dict()` (None → güncel ortalama kalite)

        Returns:
            `{"steps", "batch_size", "lr", "priority_tags", "rationale", "should_train"}`
        """
        if self._hooks.has("resource_budget"):
            self._refresh_budget()

        report = dict(quality_report or {})
        if not report:
            report = {"score": self._mean_quality()}

        budget: Dict[str, Any] = {}
        if self._hooks.has("resource_budget"):
            try:
                budget = self._hooks.resource_budget() or {}
            except Exception as exc:
                logger.warning(f"Bütçe alınamadı: {exc}")
        if not isinstance(budget, dict):
            budget = {}

        try:
            plan = self._growth.next_training_plan(self._experiences.stats(), budget, report)
        except Exception as exc:
            logger.warning(f"Eğitim planı üretilemedi: {exc}")
            plan = {
                "steps": 0,
                "batch_size": 1,
                "lr": 1e-4,
                "priority_tags": [],
                "rationale": f"Plan üretilemedi: {exc}",
                "should_train": False,
            }
        self._last_plan = dict(plan)
        return plan

    # ─── Bilgi Bakımı ──────────────────────────────────────────────────

    def audit_knowledge(self, max_notes: int = DEFAULT_MAX_NOTES) -> dict:
        """
        Bilgi kasasını denetler.

        Returns:
            `{"notes", "issues", "contradictions", "health", "plan", "kinds"}`
        """
        limit = max(1, int(max_notes or DEFAULT_MAX_NOTES))
        if not self._hooks.has("list_notes"):
            return {
                "notes": 0,
                "issues": [],
                "contradictions": [],
                "health": 0.0,
                "plan": {"to_archive": [], "to_merge": [], "to_fix": [], "to_review": []},
                "kinds": {},
                "skipped": "'list_notes' kancası yok; kasa denetlenemedi",
            }

        notes = self._call_hook("list_notes", limit, default=[])
        if not isinstance(notes, list):
            notes = []
        graph = self._graph()
        issues, contradictions = self._audit_core(notes, graph)
        self._last_audit = {"notes": notes, "graph": graph}
        self._pending_contradictions = contradictions

        kinds: Dict[str, int] = {}
        for issue in issues:
            kinds[issue.kind] = kinds.get(issue.kind, 0) + 1

        plan = self._integrity.quarantine_plan(issues)
        health = self._integrity.health_score(issues, len(notes))

        return {
            "notes": len(notes),
            "issues": [issue.to_dict() for issue in issues],
            "contradictions": list(contradictions),
            "health": health,
            "plan": {
                "to_archive": list(plan.get("to_archive") or []),
                "to_merge": [list(pair) for pair in (plan.get("to_merge") or [])],
                "to_fix": list(plan.get("to_fix") or []),
                "to_review": list(plan.get("to_review") or []),
            },
            "kinds": kinds,
        }

    def synthesize_topic(self, topic: str, note_ids: Optional[List[str]] = None) -> Optional[dict]:
        """
        Belirli bir konuyu (ya da not kümesini) sentezler.

        Args:
            topic: Sentez konusu
            note_ids: Kullanılacak not kimlikleri (None → konuya göre arama)

        Returns:
            `SynthesisResult.to_dict()` ya da yeterli not yoksa None
        """
        notes: List[dict] = []
        if self._hooks.has("list_notes"):
            fetched = self._call_hook("list_notes", DEFAULT_MAX_NOTES, default=[])
            if isinstance(fetched, list):
                notes = fetched
        if not notes and isinstance(self._last_audit, dict):
            notes = list(self._last_audit.get("notes") or [])

        if not self._hooks.has("upsert_note"):
            logger.info("Sentez yalnızca üretilecek: 'upsert_note' kancası yok")

        result = self._synthesize_topic(topic, note_ids, notes, auto_apply=True)
        if result is None:
            return None

        payload = result.to_dict()
        # Not kimliği `_synthesize_topic` içinde yazma sırasında belirlenir
        payload["note_id"] = self._last_synthesis_note or None
        return payload

    def research_plan(self, top_k: int = 5) -> List[dict]:
        """
        Merak öğelerinden sıralı araştırma planı üretir.

        Args:
            top_k: Görev sayısı
        """
        items = list(getattr(self, "_pending_curiosity", []) or [])
        if not items:
            notes = list((self._last_audit or {}).get("notes") or [])
            if self._hooks.has("list_notes") and not notes:
                fetched = self._call_hook("list_notes", DEFAULT_MAX_NOTES, default=[])
                if isinstance(fetched, list):
                    notes = fetched
            try:
                items = self._curiosity.scan(notes, self._graph(), max_items=25)
                self._pending_curiosity = items
            except Exception as exc:
                logger.warning(f"Araştırma planı için tarama başarısız: {exc}")
                items = []

        try:
            return self._curiosity.next_research_plan(items, top_k=top_k)
        except Exception as exc:
            logger.warning(f"Araştırma planı üretilemedi: {exc}")
            return []

    # ─── Günlük ────────────────────────────────────────────────────────

    def journal(self, report: EvolutionReport, path: Optional[str] = None) -> str:
        """
        Tur raporunu markdown günlüğüne ekler.

        Args:
            report: Yazılacak rapor
            path: Hedef dosya (None → `<data_path>/evolution/journal.md`)

        Returns:
            Günlük dosyasının yolu (yazılamazsa boş dize)
        """
        target = Path(str(path)) if path else self._journal_path
        if target is None:
            return ""

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            header = ""
            if not target.exists() or target.stat().st_size == 0:
                header = (
                    "# NeuroCore Evrim Günlüğü\n\n"
                    "> Bu dosya `EvolutionLayer.cycle()` tarafından otomatik yazılır. "
                    "Her tur bir başlık açar.\n\n"
                )
            with open(target, "a", encoding="utf-8") as handle:
                if header:
                    handle.write(header)
                handle.write(report.to_markdown())
                handle.write("\n---\n\n")
            return str(target)
        except Exception as exc:
            logger.warning(f"Günlük yazılamadı ({target}): {exc}")
            return ""

    # ─── Sağlık ve İstatistik ──────────────────────────────────────────

    def health(self) -> dict:
        """
        Katmanın sağlık özeti (JSON-serileştirilebilir).

        Returns:
            Bileşen durumları, tampon/kasa metrikleri ve kanca envanteri
        """
        buffer_stats = self._experiences.stats()
        state = self._growth.state()
        health = {
            "layer": self.name,
            "initialized": self._initialized,
            "cycles": int(self._cycle),
            "blocked_cycles": int(self._counters["blocked_cycles"]),
            "step_failures": int(self._counters["step_failures"]),
            "buffer": {
                "count": buffer_stats.get("count"),
                "capacity": buffer_stats.get("capacity"),
                "mean_quality": buffer_stats.get("mean_quality"),
                "mean_reward": buffer_stats.get("mean_reward"),
            },
            "quality": {
                "mean_recent": self._mean_quality(),
                "evaluations": self._evaluator.get_stats().get("evaluations"),
                "errors": self._evaluator.get_stats().get("errors"),
            },
            "integrity": {
                "last_health": self._integrity.get_stats().get("last_health"),
                "issues_found": self._integrity.get_stats().get("issues_found"),
                "kinds": self._integrity.get_stats().get("last_kinds"),
            },
            "growth": {
                "maturity": self._growth.maturity(),
                "num_experts": state.num_experts,
                "max_experts": state.max_experts,
                "generation": state.generation,
                "last_decision": self._growth.get_stats().get("last_decision"),
            },
            "curiosity": {
                "pending": len(getattr(self, "_pending_curiosity", []) or []),
                "kinds": self._curiosity.get_stats().get("last_kinds"),
            },
            "hooks": {
                "available": self._hooks.available(),
                "missing": [name for name in HOOK_NAMES if not self._hooks.has(name)],
            },
            "lessons": self.lessons(5),
            "last_report": self._reports[-1] if self._reports else {},
        }
        self._last_health = health
        return health

    def get_stats(self) -> dict:
        """Katman ve tüm alt sistemler için JSON-serileştirilebilir metrikler."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "cycles": int(self._cycle),
            "observations": int(self._counters["observations"]),
            "feedbacks": int(self._counters["feedbacks"]),
            "blocked_cycles": int(self._counters["blocked_cycles"]),
            "step_failures": int(self._counters["step_failures"]),
            "config": {
                "buffer_size": self._buffer_size,
                "min_samples_for_training": self._min_samples,
                "auto_expand_threshold": self._auto_expand_threshold,
                "prune_threshold": self._prune_threshold,
                "expansion_strategy": self._expansion_strategy,
                "embedding_dim": self._embedding_dim,
                "max_experts": self._max_experts,
                "hardware_profile": self._hardware_profile,
                "data_path": str(self._dir) if self._dir else self._data_path,
            },
            "hooks": {
                "available": self._hooks.available(),
                "missing": [name for name in HOOK_NAMES if not self._hooks.has(name)],
            },
            "subsystems": {
                "experience": self._experiences.get_stats(),
                "evaluator": self._evaluator.get_stats(),
                "curriculum": self._curriculum.get_stats(),
                "synthesizer": self._synthesizer.get_stats(),
                "curiosity": self._curiosity.get_stats(),
                "growth": self._growth.get_stats(),
                "integrity": self._integrity.get_stats(),
            },
            "last_plan": dict(self._last_plan),
            "last_report": self._reports[-1] if self._reports else {},
        }
