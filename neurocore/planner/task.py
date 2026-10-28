"""
NeuroCore Görev Modeli ve Görev Grafiği
========================================
Planlayıcı katmanının (L7) temel veri yapıları:

    Task           : Tek bir yürütülebilir plan adımı
    ReasoningStep  : Akıl yürütme zincirinin tek bir halkası
    ReasoningTrace : Adım adım gerekçe kaydı (düşünce/eylem/gözlem)
    TaskGraph      : Gerçek DAG — sıralama, görselleştirme, doğrulama

Tasarım notu:
    ``TaskGraph`` sıralama/gezinme için ``networkx`` kullanır; ancak
    **tek doğruluk kaynağı** düz sözlük verisidir (``self.tasks``).
    Graf yalnızca bir *indeks*'tir; bozulursa sözlükten yeniden kurulabilir.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

import networkx as nx

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("planner.task")


# ─── Hatalar ────────────────────────────────────────────────────────────────

class PlanError(Exception):
    """
    Plan yapısı hataları (döngü, bilinmeyen bağımlılık, tekrarlı kimlik, …).

    Planlayıcı katmanının tüm yapısal hataları bu tipte toplanır; katman
    asla ham ``networkx``/``KeyError`` istisnası sızdırmaz.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: Dict[str, Any] = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Hata bilgisini JSON-serileştirilebilir sözlüğe çevirir."""
        return {"error": type(self).__name__, "message": self.message, "details": self.details}


# ─── Görev Durumu ve Türü ───────────────────────────────────────────────────

class TaskState(str, Enum):
    """Bir görevin yaşam döngüsü durumu."""

    PENDING = "pending"      # bağımlılıklarını bekliyor
    READY = "ready"          # bağımlılıkları tamam, yürütülebilir
    RUNNING = "running"      # şu anda yürütülüyor
    DONE = "done"            # başarıyla tamamlandı
    FAILED = "failed"        # tüm denemeler tükendi
    SKIPPED = "skipped"      # bağımlılığı başarısız / çevrimdışı engeli
    BLOCKED = "blocked"      # kritik kaynak yok (yetki, bütçe, kilit)
    CANCELLED = "cancelled"  # plan iptal edildi


class TaskKind(str, Enum):
    """Görevin bilişsel/araçsal türü."""

    REASON = "reason"            # akıl yürütme, çıkarım, tasarım
    RETRIEVE = "retrieve"        # kasadan bilgi getirme
    RESEARCH = "research"        # dış/derin araştırma
    COMPUTE = "compute"          # matematik/istatistik hesabı
    TOOL = "tool"                # işletim sistemi / harici araç çağrısı
    SYNTHESIZE = "synthesize"    # nihai yanıtı üretme
    VERIFY = "verify"            # doğrulama, çapraz kontrol
    DELEGATE = "delegate"        # alt planlayıcıya devretme
    MEMORIZE = "memorize"        # kasaya yazma (öğrenme)


#: Görevin sonlanmış (bir daha çalışmayacak) durumları.
TERMINAL_STATES: Set[TaskState] = {
    TaskState.DONE,
    TaskState.FAILED,
    TaskState.SKIPPED,
    TaskState.CANCELLED,
}

#: Başarısız sayılan durumlar.
FAILURE_STATES: Set[TaskState] = {
    TaskState.FAILED,
    TaskState.SKIPPED,
    TaskState.CANCELLED,
    TaskState.BLOCKED,
}


# ─── Serileştirme Yardımcıları ──────────────────────────────────────────────

def make_id_factory(prefix: str = "t", start: int = 1):
    """
    Sıralı kimlik üretici fabrikası — her çağrı **bağımsız** bir sayaç verir.

    Kullanım:
        factory = make_id_factory()
        factory()   # "t1"
        factory()   # "t2"
    """
    counter = {"n": start}

    def _next_id() -> str:
        value = f"{prefix}{counter['n']}"
        counter["n"] += 1
        return value

    return _next_id


def _stable_id_factory() -> str:
    """Kimliği dışarıdan atanmış görevler için boş üretici (asla çakışmaz)."""
    return ""


def json_safe(value: Any, depth: int = 0) -> Any:
    """
    Herhangi bir değeri JSON-serileştirilebilir biçime indirger.

    Dataclass, Enum, set, Path, bytes ve numpy-benzeri tipler güvenle
    dönüştürülür; döngüsel yapılarda derinlik sınırı uygulanır.
    """
    if depth > 6:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bayt>"
    if isinstance(value, (set, frozenset)):
        return sorted(json_safe(v, depth + 1) for v in value)
    if isinstance(value, (list, tuple)):
        return [json_safe(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v, depth + 1) for k, v in value.items()}
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return json_safe(value.to_dict(), depth + 1)
        except Exception:  # pragma: no cover - savunmacı
            return str(value)
    if hasattr(value, "tolist"):  # numpy dizileri
        try:
            return json_safe(value.tolist(), depth + 1)
        except Exception:  # pragma: no cover - savunmacı
            return str(value)
    return str(value)


def _short(text: Any, limit: int = 160) -> str:
    """Metni tek satıra indirip kısaltır."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# ─── Görev ──────────────────────────────────────────────────────────────────

#: Durum → görsel simge (konsol ve markdown aynı simgeyi paylaşır).
STATE_ICONS: Dict[TaskState, str] = {
    TaskState.PENDING: "○",
    TaskState.READY: "◔",
    TaskState.RUNNING: "⋯",
    TaskState.DONE: "✓",
    TaskState.FAILED: "✗",
    TaskState.SKIPPED: "⊘",
    TaskState.BLOCKED: "⊘",
    TaskState.CANCELLED: "⊘",
}


@dataclass
class Task:
    """
    Plan içindeki tek bir yürütülebilir adım.

    Attributes:
        id: Benzersiz kimlik ("t1", "t1.1", …)
        name: Kısa insan-okur ad
        kind: Görev türü (``TaskKind``)
        description: Ayrıntılı açıklama / gerekçe
        tool: Executor'a kayıtlı yetenek adı (boş = yetenek çağrılmaz)
        args: Yeteneğe geçirilecek argümanlar ("${t1.result}" şablonu destekler)
        depends_on: Önce tamamlanması gereken görev kimlikleri
        state: Yaşam döngüsü durumu
        attempts: Yapılan deneme sayısı
        max_attempts: İzin verilen en fazla deneme
        result: Yetenek dönüş değeri
        error: Son hata mesajı (Türkçe)
        confidence: 0..1 güven skoru
        priority: 0 = en yüksek öncelik
        depth: Ağaçtaki derinlik
        parent_id: Üst görev kimliği
        children: Alt görev kimlikleri
        duration_ms: Ölçülen yürütme süresi
        tags: Serbest etiketler
        critical: True ise başarısızlık planı kurtarmaya zorlar
    """

    name: str
    kind: TaskKind = TaskKind.REASON
    description: str = ""
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    state: TaskState = TaskState.PENDING
    attempts: int = 0
    max_attempts: int = 2
    result: Any = None
    error: str = ""
    confidence: float = 0.0
    priority: int = 5
    depth: int = 0
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    tags: Set[str] = field(default_factory=set)
    critical: bool = False
    id: str = field(default_factory=lambda: f"t{uuid4().hex[:6]}")

    # ─── Türetilmiş Özellikler ──────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        """Görev sonlanmış durumda mı?"""
        return self.state in TERMINAL_STATES

    @property
    def duration_seconds(self) -> float:
        """Yürütme süresi (saniye)."""
        return self.duration_ms / 1000.0

    @property
    def icon(self) -> str:
        """Duruma karşılık gelen görsel simge."""
        return STATE_ICONS.get(self.state, "?")

    def touch_started(self) -> None:
        """Görevi RUNNING durumuna alır ve başlangıç damgasını vurur."""
        self.started_at = time.time()
        self.state = TaskState.RUNNING
        self.attempts += 1

    def touch_finished(self) -> None:
        """Bitiş damgasını ve süreyi hesaplar."""
        self.finished_at = time.time()
        if self.started_at:
            self.duration_ms = max(0.0, (self.finished_at - self.started_at) * 1000.0)

    # ─── Serileştirme ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Görevi JSON-serileştirilebilir sözlüğe çevirir."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind.value,
            "description": self.description,
            "tool": self.tool,
            "args": json_safe(self.args),
            "depends_on": list(self.depends_on),
            "state": self.state.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "result": json_safe(self.result),
            "error": self.error,
            "confidence": round(float(self.confidence), 4),
            "priority": self.priority,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": round(self.duration_ms, 2),
            "tags": sorted(self.tags),
            "critical": self.critical,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        """
        Sözlükten görev kurar (eksik alanlara toleranslı).

        Bilinmeyen anahtarlar sessizce yok sayılır; bozuk enum değerleri
        varsayılana düşer — plan asla yükleme yüzünden çökmez.
        """
        data = dict(d or {})
        valid = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in valid}

        for key, enum_cls, default in (
            ("kind", TaskKind, TaskKind.REASON),
            ("state", TaskState, TaskState.PENDING),
        ):
            raw = data.get(key, default)
            try:
                data[key] = raw if isinstance(raw, enum_cls) else enum_cls(str(raw))
            except (ValueError, TypeError):
                data[key] = default

        if "tags" in data and not isinstance(data["tags"], set):
            tags = data["tags"] or []
            data["tags"] = set(tags) if isinstance(tags, (list, tuple, set)) else {str(tags)}
        if "depends_on" in data and not isinstance(data["depends_on"], list):
            data["depends_on"] = list(data["depends_on"] or [])
        if "children" in data and not isinstance(data["children"], list):
            data["children"] = list(data["children"] or [])
        if "args" in data and not isinstance(data["args"], dict):
            data["args"] = {}
        data.setdefault("name", data.get("id", "görev"))

        try:
            return cls(**data)
        except TypeError as exc:  # pragma: no cover - savunmacı
            raise PlanError(f"görev sözlükten kurulamadı: {exc}", {"data": json_safe(data)}) from exc

    # ─── Görselleştirme ─────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Tek satırlık Türkçe özet.

        Örnek: ``✓ [t1] Araştır (research) 231ms``
        """
        parts = [f"{self.icon} [{self.id}]", _short(self.name, 70), f"({self.kind.value})"]
        if self.duration_ms > 0:
            parts.append(f"{self.duration_ms:.0f}ms")
        if self.attempts > 1:
            parts.append(f"· {self.attempts} deneme")
        if self.state == TaskState.FAILED and self.error:
            parts.append(f"· {_short(self.error, 80)}")
        return " ".join(parts)

    def __str__(self) -> str:  # pragma: no cover - kolaylık
        return self.summary()


# ─── Akıl Yürütme Zinciri ───────────────────────────────────────────────────

#: Akıl yürütme adımı türleri.
REASONING_KINDS: Tuple[str, ...] = ("thought", "action", "observation", "critique", "reflection")


@dataclass
class ReasoningStep:
    """
    Akıl yürütme zincirinin tek bir halkası.

    Attributes:
        index: 1-tabanlı sıra numarası
        kind: thought | action | observation | critique | reflection
        content: Adımın metni
        task_id: İlişkili görev kimliği (varsa)
        confidence: 0..1 güven skoru
        timestamp: Oluşturulma zamanı
    """

    index: int
    kind: str
    content: str
    task_id: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Adımı JSON-serileştirilebilir sözlüğe çevirir."""
        return {
            "index": self.index,
            "kind": self.kind,
            "content": self.content,
            "task_id": self.task_id,
            "confidence": round(float(self.confidence), 4),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReasoningStep":
        """Sözlükten adım kurar."""
        data = dict(d or {})
        return cls(
            index=int(data.get("index", 0)),
            kind=str(data.get("kind", "thought")),
            content=str(data.get("content", "")),
            task_id=str(data.get("task_id", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            timestamp=float(data.get("timestamp", 0.0) or time.time()),
        )

    def render(self) -> str:
        """Tek satırlık gösterim: ``  1. [thought] …``"""
        flat = " ".join(str(self.content).split())
        prefix = f"{self.index:>3}. [{self.kind}]"
        if self.task_id:
            prefix += f" {self.task_id}"
        if self.confidence:
            prefix += f" ({self.confidence:.2f})"
        return f"{prefix} {flat}"


@dataclass
class ReasoningTrace:
    """
    Bir planın adım adım gerekçe kaydı.

    ``max_steps`` aşıldığında yeni adım **eklenmez** (uyarı loglanır);
    yürütme devam eder — izleme kaybı planı durdurmaz.
    """

    trace_id: str
    goal: str
    steps: List[ReasoningStep] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    max_steps: int = 50

    # ─── Kayıt ──────────────────────────────────────────────────────────

    def add(self, kind: str, content: str, task_id: str = "",
            confidence: float = 0.0) -> Optional[ReasoningStep]:
        """
        Zincire yeni adım ekler.

        Args:
            kind: Adım türü (``REASONING_KINDS``)
            content: Adım metni
            task_id: İlişkili görev kimliği
            confidence: 0..1 güven skoru

        Returns:
            Eklenen ``ReasoningStep`` ya da sınır aşıldıysa ``None``.
        """
        if len(self.steps) >= self.max_steps:
            logger.warning(
                f"⚠ Akıl yürütme adım sınırı aşıldı ({self.max_steps}) — "
                f"'{kind}' adımı kaydedilmedi"
            )
            return None

        step = ReasoningStep(
            index=len(self.steps) + 1,
            kind=str(kind),
            content=" ".join(str(content).split()),
            task_id=task_id,
            confidence=float(confidence or 0.0),
        )
        self.steps.append(step)
        return step

    @property
    def is_full(self) -> bool:
        """Adım sınırı doldu mu?"""
        return len(self.steps) >= self.max_steps

    def finish(self) -> None:
        """Zinciri kapatır (bitiş damgası)."""
        self.finished_at = time.time()

    def by_kind(self, kind: str) -> List[ReasoningStep]:
        """Belirli türdeki adımları döndürür."""
        return [s for s in self.steps if s.kind == kind]

    # ─── Serileştirme / Görselleştirme ──────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Zinciri JSON-serileştirilebilir sözlüğe çevirir."""
        return {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "max_steps": self.max_steps,
            "step_count": len(self.steps),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReasoningTrace":
        """Sözlükten zincir kurar."""
        data = dict(d or {})
        trace = cls(
            trace_id=str(data.get("trace_id", f"tr_{uuid4().hex[:8]}")),
            goal=str(data.get("goal", "")),
            started_at=float(data.get("started_at", 0.0) or time.time()),
            finished_at=float(data.get("finished_at", 0.0) or 0.0),
            max_steps=int(data.get("max_steps", 50) or 50),
        )
        for raw in data.get("steps", []) or []:
            if isinstance(raw, dict):
                trace.steps.append(ReasoningStep.from_dict(raw))
        return trace

    def render(self, limit: int = 0) -> str:
        """
        Zincirin tam çok satırlı gösterimi.

        Args:
            limit: >0 ise yalnızca son ``limit`` adım gösterilir
        """
        if not self.steps:
            return f"🧠 Akıl yürütme zinciri boş (hedef: {_short(self.goal, 80)})"

        shown = self.steps if limit <= 0 else self.steps[-limit:]
        lines = [
            f"🧠 Akıl yürütme zinciri [{self.trace_id}] — {len(self.steps)} adım",
            f"   hedef: {_short(self.goal, 120)}",
        ]
        if limit > 0 and len(shown) < len(self.steps):
            lines.append(f"   … ilk {len(self.steps) - len(shown)} adım gizlendi")
        lines.extend(s.render() for s in shown)
        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        """
        Zincir özeti: tür bazlı sayımlar, tahmini token ve süre.

        Token tahmini: ``len(metin) // 4`` (kaba ama deterministik).
        """
        counts: Dict[str, int] = {k: 0 for k in REASONING_KINDS}
        chars = 0
        for step in self.steps:
            counts[step.kind] = counts.get(step.kind, 0) + 1
            chars += len(step.content)

        duration_ms = 0.0
        if self.finished_at and self.started_at:
            duration_ms = max(0.0, (self.finished_at - self.started_at) * 1000.0)

        return {
            "trace_id": self.trace_id,
            "goal": _short(self.goal, 160),
            "total_steps": len(self.steps),
            "counts": counts,
            "tokens_estimate": chars // 4,
            "chars": chars,
            "duration_ms": round(duration_ms, 2),
            "max_steps": self.max_steps,
            "full": self.is_full,
            "mean_confidence": round(
                sum(s.confidence for s in self.steps) / len(self.steps), 4
            ) if self.steps else 0.0,
        }


# ─── Görev Grafiği (DAG) ────────────────────────────────────────────────────

class TaskGraph:
    """
    Görev bağımlılık grafiği — gerçek DAG (networkx) + düz sözlük doğruluk kaynağı.

    ``self.tasks`` (sıralı sözlük) tek doğruluk kaynağıdır; ``self._g`` yalnızca
    sıralama, gezinme ve görselleştirme için tutulan bir indekstir.

    Kullanım:
        g = TaskGraph(goal="GPU sıcaklığım neden yüksek?")
        t1 = g.add(Task(id="t1", name="Kasa araması", kind=TaskKind.RETRIEVE))
        t2 = g.add(Task(id="t2", name="Yanıt", kind=TaskKind.SYNTHESIZE, depends_on=["t1"]))
        for batch in g.parallel_batches():
            print(batch)
    """

    def __init__(self, goal: str = "") -> None:
        self.goal: str = goal
        self.tasks: Dict[str, Task] = {}
        self._g = nx.DiGraph()
        self.created_at: float = time.time()

    # ─── Yapı Kurma ─────────────────────────────────────────────────────

    def add(self, task: Task) -> Task:
        """
        Grafiğe görev ekler ve bağımlılık kenarlarını kurar.

        Raises:
            PlanError: Kimlik zaten varsa.
        """
        if task.id in self.tasks:
            raise PlanError(
                f"tekrarlı görev kimliği: '{task.id}'",
                {"task_id": task.id, "existing": self.tasks[task.id].name},
            )

        self.tasks[task.id] = task
        self._g.add_node(task.id)

        for dep in task.depends_on:
            if dep in self.tasks:
                self._g.add_edge(dep, task.id)
            else:
                # İleri bağımlılık: hedef sonradan eklendiğinde bağlanacak
                logger.debug(f"Görev '{task.id}' bilinmeyen bağımlılığa atıfta bulundu: '{dep}'")
                self._g.add_node(dep)

        # Sonradan eklenen hedeflerin bekleyen kenarlarını bağla
        for other in self.tasks.values():
            if task.id in other.depends_on and not self._g.has_edge(task.id, other.id):
                self._g.add_edge(task.id, other.id)

        return task

    def get(self, task_id: str) -> Task:
        """
        Görevi döndürür.

        Raises:
            PlanError: Görev bulunamazsa.
        """
        task = self.tasks.get(task_id)
        if task is None:
            raise PlanError(f"görev bulunamadı: '{task_id}'", {"task_id": task_id})
        return task

    def has(self, task_id: str) -> bool:
        """Görev var mı?"""
        return task_id in self.tasks

    def set_dependencies(self, task_id: str, depends_on: Sequence[str]) -> Task:
        """
        Görevin bağımlılıklarını **tek doğruluk kaynağından** yeniden kurar.

        Graf indeksi (``self._g``) her zaman ``Task.depends_on`` ile eşitlenir;
        böylece kenarlar elle değiştirildiğinde bile sıralama/gezinme tutarlı kalır.

        Args:
            task_id: Hedef görev
            depends_on: Yeni bağımlılık kimlikleri

        Raises:
            PlanError: Görev bulunamazsa.
        """
        task = self.get(task_id)
        task.depends_on = [d for d in dict.fromkeys(depends_on) if d and d != task_id]
        self.resync()
        return task
    def resync(self) -> None:
        """
        Graf indeksini ve ``children`` listelerini düz sözlük verisinden yeniden kurar.

        Kimlik doğruluk kaynağı ``self.tasks``; ``self._g`` ve ``Task.children``
        yalnızca türev verilerdir. Bu metot, görev/kenar elle değiştirildikten
        sonra çağrılır ve görselleştirmenin her zaman tutarlı kalmasını sağlar.
        """
        graph = nx.DiGraph()
        for tid in self.tasks:
            graph.add_node(tid)
        for tid, task in self.tasks.items():
            for dep in task.depends_on:
                if dep in self.tasks and dep != tid:
                    graph.add_edge(dep, tid)
        self._g = graph
        self._sync_children()

    def _sync_children(self) -> None:
        """Her görevin ``children`` listesini ``depends_on`` alanlarından türetir."""
        children: Dict[str, List[str]] = {tid: [] for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.depends_on:
                if dep in children and dep != task.id and task.id not in children[dep]:
                    children[dep].append(task.id)
        for tid, task in self.tasks.items():
            task.children = children[tid]

    def remove(self, task_id: str) -> bool:
        """
        Görevi ve tüm bağlantılarını grafdan çıkarır.

        Bağımlı görevlerin ``depends_on`` listeleri de temizlenir.

        Returns:
            Görev gerçekten silindiyse True.
        """
        if task_id not in self.tasks:
            return False

        self.tasks.pop(task_id)
        if self._g.has_node(task_id):
            self._g.remove_node(task_id)

        for task in self.tasks.values():
            if task_id in task.depends_on:
                task.depends_on = [d for d in task.depends_on if d != task_id]
            if task.parent_id == task_id:
                task.parent_id = None
            if task_id in task.children:
                task.children = [c for c in task.children if c != task_id]
        self.resync()
        return True
    def __len__(self) -> int:
        return len(self.tasks)

    def __contains__(self, task_id: object) -> bool:
        return task_id in self.tasks

    def __iter__(self):
        return iter(self.tasks.values())

    # ─── Gezinme ────────────────────────────────────────────────────────

    def edges(self) -> List[Tuple[str, str]]:
        """
        Tüm bağımlılık kenarları (kaynak → hedef).

        Kenarlar, kimlik doğruluk kaynağı olan ``Task.depends_on`` alanlarından
        türetilir; bu yüzden görev listesi elle değiştirilse bile tutarlıdır.
        """
        edges: List[Tuple[str, str]] = []
        for task in self.tasks.values():
            for dep in task.depends_on:
                if dep in self.tasks and dep != task.id and (dep, task.id) not in edges:
                    edges.append((dep, task.id))
        return edges

    def dependencies_of(self, task_id: str) -> List[str]:
        """Görevin doğrudan bağımlılıkları (giriş kenarları)."""
        if task_id not in self.tasks:
            return []
        return [d for d in self.tasks[task_id].depends_on if d in self.tasks]

    def dependents_of(self, task_id: str) -> List[str]:
        """Göreve doğrudan bağımlı görevler (çıkış kenarları)."""
        if task_id not in self.tasks:
            return []
        return [v for v in self._g.successors(task_id) if v in self.tasks]

    def ancestors(self, task_id: str) -> List[str]:
        """Tüm dolaylı bağımlılıklar."""
        if task_id not in self.tasks:
            return []
        return [n for n in nx.ancestors(self._g, task_id) if n in self.tasks]

    def descendants(self, task_id: str) -> List[str]:
        """Tüm dolaylı bağımlı görevler."""
        if task_id not in self.tasks:
            return []
        return [n for n in nx.descendants(self._g, task_id) if n in self.tasks]

    def roots(self) -> List[str]:
        """Bağımlılığı olmayan (giriş) görevler."""
        return [t.id for t in self.tasks.values() if not self.dependencies_of(t.id)]

    def leaves(self) -> List[str]:
        """Kendisine bağımlı görev olmayan (çıkış) görevler."""
        return [t.id for t in self.tasks.values() if not self.dependents_of(t.id)]

    def topological_order(self) -> List[str]:
        """
        Bağımlılık sırasına göre doğrusal sıralama.

        Raises:
            PlanError: Graf döngü içeriyorsa.
        """
        try:
            order = list(nx.lexicographical_topological_sort(self._g, key=self._sort_key))
        except nx.NetworkXUnfeasible:
            cycle = self._find_cycle()
            raise PlanError(
                f"döngü tespit edildi: {' → '.join(cycle)}",
                {"cycle": cycle},
            ) from None
        except Exception as exc:  # pragma: no cover - savunmacı
            raise PlanError(f"topolojik sıralama başarısız: {exc}") from exc

        return [n for n in order if n in self.tasks]

    def _sort_key(self, node_id: str) -> Tuple[int, int, str]:
        """Eşit hazır görevler arasında öncelik → ekleme sırası → kimlik."""
        task = self.tasks.get(node_id)
        if task is None:
            return (99, 10 ** 6, str(node_id))
        try:
            insertion = list(self.tasks.keys()).index(node_id)
        except ValueError:  # pragma: no cover - savunmacı
            insertion = 10 ** 6
        return (int(task.priority), insertion, node_id)

    def _find_cycle(self) -> List[str]:
        """Döngüyü düğüm listesi olarak döndürür (yoksa boş liste)."""
        try:
            cycle_edges = nx.find_cycle(self._g)
        except Exception:
            return []
        nodes: List[str] = []
        for edge in cycle_edges:
            u, v = edge[0], edge[1]
            if not nodes:
                nodes.append(u)
            if v == nodes[0]:
                break
            nodes.append(v)
        return nodes

    # ─── Hazır Görevler ve Dalgalar ─────────────────────────────────────

    def ready_tasks(self) -> List[str]:
        """
        Yürütülmeye hazır görevler.

        Bir görev hazırdır ancak ve ancak:
            * durumu ``PENDING`` ve
            * tüm gerçek bağımlılıkları ``DONE`` ise.
        """
        ready: List[str] = []
        for task in self.tasks.values():
            if task.state != TaskState.PENDING:
                continue
            deps = self.dependencies_of(task.id)
            if all(self.tasks[d].state == TaskState.DONE for d in deps):
                ready.append(task.id)
        return sorted(ready, key=self._sort_key)

    def parallel_batches(self) -> List[List[str]]:
        """
        Hazır-küme soyma (ready-set peeling) ile paralel dalgalar.

        Her dalga, o anda birlikte yürütülebilecek görevleri içerir —
        yürütme sırasından **bağımsız** olarak hesaplanır.

        Returns:
            Dalga listesi; her dalga görev kimliklerinden oluşur.
        """
        completed: Set[str] = set()
        batches: List[List[str]] = []
        remaining = set(self.tasks.keys())

        while remaining:
            wave = [
                tid for tid in remaining
                if all(
                    (dep not in remaining) or (dep in completed)
                    for dep in self.dependencies_of(tid)
                )
            ]
            if not wave:
                # Döngü ya da bozuk bağımlılık: kalanları tek dalgada ver
                logger.warning(
                    f"⚠ Paralel dalga hesaplanamadı, {len(remaining)} görev tek dalgada toplandı"
                )
                batches.append(sorted(remaining, key=self._sort_key))
                break

            wave.sort(key=self._sort_key)
            batches.append(wave)
            completed.update(wave)
            remaining.difference_update(wave)
            if len(batches) > len(self.tasks) + 1:  # pragma: no cover - güvenlik
                break

        return batches

    def max_parallelism(self) -> int:
        """En geniş dalgadaki görev sayısı."""
        return max((len(b) for b in self.parallel_batches()), default=0)

    # ─── Kritik Yol ─────────────────────────────────────────────────────

    def task_weight(self, task: Task) -> float:
        """
        Kritik yol için görev ağırlığı.

        Ağırlık arttıkça görev daha "uzun" sayılır:
        ``(1.0 + priority/10) * (1.0 + depth/10) * kind_factor``
        """
        kind_factor = {
            TaskKind.RESEARCH: 2.0,
            TaskKind.SYNTHESIZE: 1.8,
            TaskKind.COMPUTE: 1.5,
            TaskKind.TOOL: 1.4,
            TaskKind.VERIFY: 1.3,
            TaskKind.REASON: 1.2,
            TaskKind.RETRIEVE: 1.0,
            TaskKind.MEMORIZE: 0.6,
            TaskKind.DELEGATE: 1.5,
        }.get(task.kind, 1.0)
        return (1.0 + task.priority / 10.0) * (1.0 + task.depth / 10.0) * kind_factor

    def critical_path(self) -> List[str]:
        """
        En uzun (ağırlıklı) yolu döndürür — planın darboğaz zinciri.

        Eşit ağırlıkta lexicographic sıra tercih edilir (deterministik).
        """
        if not self.tasks:
            return []

        weights = {tid: self.task_weight(t) for tid, t in self.tasks.items()}
        try:
            path = nx.dag_longest_path(self._g, weight="weight")
        except nx.NetworkXUnfeasible:
            return []
        except Exception:  # pragma: no cover - savunmacı
            return []

        # Ağırlık özniteliği graf kenarlarında olmadığı için düğüm ağırlığıyla
        # ikinci bir doğrulama yapılır: en ağır düğümü içeren yol tercih edilir.
        best = [n for n in path if n in self.tasks]
        if not best:
            return []

        # Deterministik iyileştirme: aynı uzunlukta alternatif yol varsa
        # daha yüksek toplam ağırlığa sahip olanı seç.
        try:
            candidate = max(
                nx.all_simple_paths(self._g, source=best[0], target=best[-1]),
                key=lambda p: sum(weights.get(n, 0.0) for n in p),
            )
            if sum(weights.get(n, 0.0) for n in candidate) > sum(weights.get(n, 0.0) for n in best):
                best = [n for n in candidate if n in self.tasks]
        except Exception:
            pass

        return best

    # ─── Doğrulama ──────────────────────────────────────────────────────

    def validate(self, max_depth: int = 10) -> List[str]:
        """
        Planı denetler ve Türkçe sorun metinleri döndürür.

        Denetlenenler:
            * bilinmeyen bağımlılıklar
            * döngüler (kendi kendine bağımlılık dahil)
            * sahipsiz alt görevler (``parent_id`` grafikte yok)
            * izin verilen derinliği aşan görevler

        Returns:
            Sorun listesi (boş liste = sağlıklı plan).
        """
        problems: List[str] = []

        if not self.tasks:
            return ["plan boş: hiç görev yok"]

        for task in self.tasks.values():
            for dep in task.depends_on:
                if dep not in self.tasks:
                    problems.append(
                        f"bilinmeyen bağımlılık: '{task.id}' → '{dep}' (görev grafikte yok)"
                    )
                elif dep == task.id:
                    problems.append(f"kendi kendine bağımlılık: '{task.id}'")

            if task.parent_id and task.parent_id not in self.tasks:
                problems.append(
                    f"sahipsiz alt görev: '{task.id}' → üst '{task.parent_id}' grafikte yok"
                )

            if task.depth > max_depth:
                problems.append(
                    f"derinlik aşımı: '{task.id}' derinlik={task.depth} > max_depth={max_depth}"
                )

        cycle = self._find_cycle()
        if cycle:
            problems.append(f"döngü tespit edildi: {' → '.join(cycle)}")

        if not self.leaves():
            problems.append("plan çıkışsız: hiç yaprak görev yok")

        return problems

    # ─── İstatistik / İlerleme ──────────────────────────────────────────

    def counts_by_state(self) -> Dict[str, int]:
        """Durum bazlı görev sayıları."""
        counts = {state.value: 0 for state in TaskState}
        for task in self.tasks.values():
            counts[task.state.value] = counts.get(task.state.value, 0) + 1
        return counts

    def counts_by_kind(self) -> Dict[str, int]:
        """Tür bazlı görev sayıları."""
        counts = {kind.value: 0 for kind in TaskKind}
        for task in self.tasks.values():
            counts[task.kind.value] = counts.get(task.kind.value, 0) + 1
        return counts

    def depth_histogram(self) -> Dict[str, int]:
        """Derinlik bazlı görev dağılımı."""
        histogram: Dict[str, int] = {}
        for task in self.tasks.values():
            key = str(task.depth)
            histogram[key] = histogram.get(key, 0) + 1
        return dict(sorted(histogram.items(), key=lambda kv: int(kv[0])))

    def progress(self) -> float:
        """0..1 arası tamamlanma oranı (sonlanmış görevler / tümü)."""
        if not self.tasks:
            return 0.0
        done = sum(1 for t in self.tasks.values() if t.is_terminal)
        return round(done / len(self.tasks), 4)

    def stats(self) -> Dict[str, Any]:
        """Grafın JSON-serileştirilebilir özet istatistikleri."""
        states = self.counts_by_state()
        batches = self.parallel_batches()
        total_ms = sum(t.duration_ms for t in self.tasks.values())

        return {
            "goal": _short(self.goal, 200),
            "total_tasks": len(self.tasks),
            "by_state": states,
            "by_kind": self.counts_by_kind(),
            "depth_histogram": self.depth_histogram(),
            "max_depth": max((t.depth for t in self.tasks.values()), default=0),
            "edges": len(self.edges()),
            "roots": self.roots(),
            "leaves": self.leaves(),
            "waves": len(batches),
            "max_parallel_batch": max((len(b) for b in batches), default=0),
            "critical_path": self.critical_path(),
            "critical_path_length": len(self.critical_path()),
            "progress": self.progress(),
            "completed": sum(1 for t in self.tasks.values() if t.state == TaskState.DONE),
            "failed": sum(1 for t in self.tasks.values() if t.state == TaskState.FAILED),
            "skipped": sum(1 for t in self.tasks.values() if t.state == TaskState.SKIPPED),
            "cancelled": sum(1 for t in self.tasks.values() if t.state == TaskState.CANCELLED),
            "total_duration_ms": round(total_ms, 2),
            "mean_confidence": round(
                sum(t.confidence for t in self.tasks.values()) / len(self.tasks), 4
            ) if self.tasks else 0.0,
        }

    # ─── Serileştirme ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Grafı JSON-serileştirilebilir sözlüğe çevirir (düz veri = doğruluk kaynağı)."""
        return {
            "goal": self.goal,
            "created_at": self.created_at,
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "edges": [list(e) for e in self.edges()],
            "order": self.topological_order() if self._is_acyclic() else [],
            "batches": self.parallel_batches(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskGraph":
        """
        Sözlükten graf kurar (bozuk veriye toleranslı).

        Graf, düz ``tasks`` sözlüğünden **yeniden** inşa edilir; kenarlar
        ``depends_on`` alanlarından türetilir (dosyadaki ``edges`` yok sayılır —
        tek doğruluk kaynağı görev kayıtlarıdır).
        """
        data = dict(d or {})
        graph = cls(goal=str(data.get("goal", "")))
        graph.created_at = float(data.get("created_at", 0.0) or time.time())

        raw_tasks = data.get("tasks", {}) or {}
        if isinstance(raw_tasks, list):  # liste biçimine de tolerans
            raw_tasks = {str(t.get("id", f"t{i}")): t for i, t in enumerate(raw_tasks)}

        for tid, raw in raw_tasks.items():
            try:
                payload = dict(raw or {})
                payload["id"] = str(payload.get("id", tid))
                graph.add(Task.from_dict(payload))
            except PlanError:
                logger.warning(f"⚠ Yüklenen grafta tekrarlı kimlik atlandı: '{tid}'")
            except Exception as exc:
                logger.warning(f"⚠ Yüklenen grafta bozuk görev atlandı ('{tid}'): {exc}")

        return graph

    def _is_acyclic(self) -> bool:
        """Graf döngüsüz mü?"""
        try:
            return nx.is_directed_acyclic_graph(self._g)
        except Exception:  # pragma: no cover - savunmacı
            return False

    def copy(self) -> "TaskGraph":
        """Grafın derin kopyasını üretir (JSON üzerinden)."""
        return TaskGraph.from_dict(self.to_dict())

    # ─── Görselleştirme ─────────────────────────────────────────────────

    def to_markdown(self, title: str = "Görev Planı") -> str:
        """
        Girintili ağaç gösterimi (Obsidian uyumlu) + dalga/özet bölümleri.

        Örnek çıktı::

            ## Görev Planı — GPU sıcaklığım neden yüksek?

            1. ✓ **Kasa araması** · `retrieve` · 231ms
               └─ 1.1 ○ **Sorgu için kasa araması** · `retrieve`
        """
        lines: List[str] = [f"## {title}"]
        if self.goal:
            lines.append("")
            lines.append(f"> **Hedef:** {_short(self.goal, 400)}")
        lines.append("")
        lines.append(
            f"`{len(self.tasks)} görev` · `{len(self.edges())} bağımlılık` · "
            f"`ilerleme %{self.progress() * 100:.0f}`"
        )
        lines.append("")

        if not self.tasks:
            lines.append("_(boş plan)_")
            return "\n".join(lines)

        # Kökler: üst görevi olmayan giriş görevleri
        graph_roots = set(self.roots())
        roots = [
            t.id for t in self.tasks.values()
            if t.parent_id is None and t.id in graph_roots
        ]
        if not roots:
            roots = [t.id for t in self.tasks.values() if t.parent_id is None] or list(self.tasks)
        visited: Set[str] = set()
        claimed: Set[str] = set()

        def walk(task_id: str, indent: str, counter: str, is_last: bool) -> None:
            task = self.tasks.get(task_id)
            if task is None or task.id in visited:
                return
            visited.add(task.id)

            connector = "" if indent == "" else ("└─ " if is_last else "├─ ")
            label = f"{counter}." if counter else "-"
            parts = [f"**{_short(task.name, 90)}**"]
            if task.tool:
                parts.append(f"`{task.tool}`")
            elif task.kind != TaskKind.REASON:
                parts.append(f"`{task.kind.value}`")
            if task.duration_ms > 0:
                parts.append(f"{task.duration_ms:.0f}ms")
            if task.confidence:
                parts.append(f"güven {task.confidence:.2f}")
            if task.state == TaskState.FAILED and task.error:
                parts.append(f"⚠ {_short(task.error, 120)}")

            lines.append(f"{indent}{connector}{label} {task.icon} " + " · ".join(parts))

            kids = [c for c in task.children if c in self.tasks and c not in claimed]
            # Bağımlı ama başka bir görevin çocuğu olmayan görevler de dala dahil.
            # Her düğüm **tek bir** üstün altında gösterilir (``claimed``).
            extra = [
                d for d in self.dependents_of(task.id)
                if d not in kids
                and d not in claimed
                and self.tasks[d].parent_id in (None, task.id)
            ]
            branch = kids + [e for e in extra if e not in kids]

            for child in branch:
                claimed.add(child)

            for idx, child in enumerate(branch):
                child_counter = f"{counter}.{idx + 1}" if counter else f"{idx + 1}"
                child_indent = indent + ("   " if is_last else "│  ")
                walk(child, child_indent, child_counter, idx == len(branch) - 1)

        for root_index, root_id in enumerate(roots):
            walk(root_id, "", str(root_index + 1), root_index == len(roots) - 1)

        # Ağaca girmeyen görevler (kopuk bileşenler)
        orphans = [tid for tid in self.tasks if tid not in visited]
        if orphans:
            lines.append("")
            lines.append("**Bağlantısız görevler:**")
            for tid in orphans:
                lines.append(f"- {self.tasks[tid].summary()}")

        # Paralel dalgalar
        batches = self.parallel_batches()
        if len(batches) > 1 or (batches and len(batches[0]) > 1):
            lines.append("")
            lines.append("### Paralel Dalgalar")
            for i, batch in enumerate(batches, 1):
                lines.append(f"{i}. `{' | '.join(batch)}`")

        return "\n".join(lines)

    def to_mermaid(self, direction: str = "TD") -> str:
        """
        Mermaid ``graph`` gösterimi — düğümler duruma göre renklendirilir.

        Obsidian, GitHub ve mermaid-cli ile doğrudan çizilebilir.
        """
        lines = [f"graph {direction}"]

        for task in self.tasks.values():
            label = _short(task.name, 46).replace('"', "'")
            if task.tool:
                label += f"<br/><i>{task.tool}</i>"
            lines.append(f'    {self._mermaid_id(task.id)}["{task.icon} {label}"]')

        for src, dst in self.edges():
            lines.append(f"    {self._mermaid_id(src)} --> {self._mermaid_id(dst)}")

        style_by_state = {
            TaskState.DONE: "fill:#1f6f43,stroke:#39d98a,color:#ffffff",
            TaskState.FAILED: "fill:#7f1d1d,stroke:#f87171,color:#ffffff",
            TaskState.RUNNING: "fill:#1e3a8a,stroke:#60a5fa,color:#ffffff",
            TaskState.READY: "fill:#4c1d95,stroke:#a78bfa,color:#ffffff",
            TaskState.PENDING: "fill:#334155,stroke:#94a3b8,color:#ffffff",
            TaskState.SKIPPED: "fill:#78350f,stroke:#fbbf24,color:#ffffff",
            TaskState.BLOCKED: "fill:#78350f,stroke:#fbbf24,color:#ffffff",
            TaskState.CANCELLED: "fill:#3f3f46,stroke:#a1a1aa,color:#ffffff",
        }
        for state, style in style_by_state.items():
            ids = [self._mermaid_id(t.id) for t in self.tasks.values() if t.state == state]
            if ids:
                lines.append(f"    style {','.join(ids)} {style}")

        return "\n".join(lines)

    @staticmethod
    def _mermaid_id(task_id: str) -> str:
        """Mermaid için güvenli düğüm kimliği (nokta içeren kimlikler sorun çıkarır)."""
        return "n_" + "".join(ch if ch.isalnum() else "_" for ch in str(task_id))

    # ─── Alt Graf ───────────────────────────────────────────────────────

    def subgraph(self, task_id: str) -> "TaskGraph":
        """
        Görev + tüm alt görevlerinden (descendants) oluşan yeni bir graf döndürür.

        Raises:
            PlanError: Görev bulunamazsa.
        """
        root = self.get(task_id)
        keep = [task_id] + self.descendants(task_id)

        sub = TaskGraph(goal=root.name or self.goal)
        sub.created_at = self.created_at
        for tid in keep:
            task = self.tasks[tid]
            clone = Task.from_dict(task.to_dict())
            clone.depends_on = [d for d in task.depends_on if d in keep]
            clone.children = [c for c in task.children if c in keep]
            if clone.parent_id not in keep:
                clone.parent_id = None if tid == task_id else clone.parent_id
            sub.add(clone)
        return sub

    def __repr__(self) -> str:  # pragma: no cover - kolaylık
        return f"TaskGraph(goal={_short(self.goal, 40)!r}, tasks={len(self.tasks)})"


# ─── Modül Yardımcıları ─────────────────────────────────────────────────────

def tasks_to_json(graph: TaskGraph) -> str:
    """Grafı okunabilir JSON metnine çevirir (Türkçe karakterler korunur)."""
    return json.dumps(graph.to_dict(), ensure_ascii=False, indent=2)
