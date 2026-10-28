"""
NeuroCore Görev Yürütücü (Executor)
====================================
``TaskGraph``'ı yürütür: yetenek (capability) çağrıları, yeniden deneme,
paralel dalgalar, yeniden planlama ve checkpoint entegrasyonu.

Tasarım gereği bu modül diğer NeuroCore katmanlarını **import etmez**.
Bunun yerine katmanlar kendilerini birer **yetenek** olarak kaydeder:

    executor.capabilities.register("retrieve", vault_search, description="kasa arama")

Böylece planlayıcı tek başına (ve testlerde sahte yeteneklerle) çalışabilir.

Not:
    Paralel yürütme **yalnızca** ``execute_async`` içinde ``asyncio.gather`` ile
    yapılır. Yetenekler torch/CUDA'ya dokunabildiği için ``ThreadPoolExecutor``
    asla kullanılmaz.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from neurocore.planner.task import (
    PlanError,
    ReasoningTrace,
    Task,
    TaskGraph,
    TaskKind,
    TaskState,
    json_safe,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("planner.executor")


# ─── Hatalar ────────────────────────────────────────────────────────────────

class CapabilityError(Exception):
    """
    Yetenek çağrısı başarısız ya da yetenek kayıtlı değil.

    Attributes:
        name: Yetenek adı
        reason: Kısa Türkçe gerekçe
        original: Yakalanan özgün istisna (varsa)
    """

    def __init__(self, name: str, reason: str = "", original: Optional[BaseException] = None) -> None:
        self.name = name
        self.reason = reason or "yetenek çağrısı başarısız"
        self.original = original
        super().__init__(f"yetenek hatası [{name}]: {self.reason}")

    def to_dict(self) -> Dict[str, Any]:
        """Hata bilgisini JSON-serileştirilebilir sözlüğe çevirir."""
        return {
            "error": type(self).__name__,
            "capability": self.name,
            "reason": self.reason,
            "original": type(self.original).__name__ if self.original else "",
        }


class ExecutionAborted(Exception):
    """Yürütme iptal edildi (kullanıcı iptali, bütçe aşımı ya da kurtarılamaz hata)."""

    def __init__(self, reason: str = "yürütme iptal edildi",
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details: Dict[str, Any] = details or {}


# ─── Yetenek Kayıt Defteri ──────────────────────────────────────────────────

@dataclass
class Capability:
    """Kayıtlı tek bir yetenek tanımı."""

    name: str
    fn: Callable[..., Any]
    description: str = ""
    is_async: bool = False
    timeout: Optional[float] = None
    cost: float = 1.0
    calls: int = 0
    errors: int = 0
    total_ms: float = 0.0
    registered_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Yetenek tanımını JSON-serileştirilebilir sözlüğe çevirir."""
        return {
            "name": self.name,
            "description": self.description,
            "is_async": self.is_async,
            "timeout": self.timeout,
            "cost": self.cost,
            "calls": self.calls,
            "errors": self.errors,
            "mean_ms": round(self.total_ms / self.calls, 2) if self.calls else 0.0,
        }


class CapabilityRegistry:
    """
    Yetenek adı → çağrılabilir eşlemesi.

    Kullanım:
        reg = CapabilityRegistry()
        reg.register("retrieve", vault.search, description="kasa arama")
        notes = reg.call("retrieve", query="GPU termal")
    """

    def __init__(self) -> None:
        self._items: Dict[str, Capability] = {}
        self._stats: Dict[str, int] = {"calls": 0, "errors": 0}

    # ─── Kayıt ──────────────────────────────────────────────────────────

    def register(self, name: str, fn: Callable[..., Any], description: str = "",
                 is_async: bool = False, timeout: Optional[float] = None,
                 cost: float = 1.0) -> None:
        """
        Yetenek kaydeder (aynı ad varsa üzerine yazar).

        Args:
            name: Yetenek adı (görevlerin ``tool`` alanı bu adı kullanır)
            fn: Çağrılabilir
            description: İnsan-okur açıklama
            is_async: True ise ``fn`` coroutine fonksiyonudur
            timeout: Saniye cinsinden üst sınır (None = sınırsız)
            cost: Göreli maliyet (planlama sezgiselleri için)
        """
        if not name:
            raise CapabilityError("", "yetenek adı boş olamaz")
        if not callable(fn):
            raise CapabilityError(str(name), f"yetenek çağrılabilir değil ({type(fn).__name__})")

        if is_async or inspect.iscoroutinefunction(fn):
            is_async = True

        self._items[str(name)] = Capability(
            name=str(name),
            fn=fn,
            description=description or getattr(fn, "__doc__", "") or "",
            is_async=is_async,
            timeout=timeout,
            cost=float(cost),
        )
        logger.debug(f"Yetenek kaydedildi: '{name}' (async={is_async})")

    def unregister(self, name: str) -> bool:
        """Yeteneği kaldırır."""
        removed = self._items.pop(str(name), None) is not None
        if removed:
            logger.debug(f"Yetenek kaldırıldı: '{name}'")
        return removed

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        """Yeteneğin çağrılabilirini döndürür (yoksa None)."""
        item = self._items.get(str(name))
        return item.fn if item else None

    def get_info(self, name: str) -> Optional[Capability]:
        """Yeteneğin tam tanımını döndürür."""
        return self._items.get(str(name))

    def has(self, name: str) -> bool:
        """Yetenek kayıtlı mı?"""
        return str(name) in self._items

    def names(self) -> List[str]:
        """Kayıtlı yetenek adları."""
        return list(self._items.keys())

    def list(self) -> List[Dict[str, Any]]:
        """Kayıtlı yeteneklerin JSON-serileştirilebilir listesi."""
        return [item.to_dict() for item in self._items.values()]

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: object) -> bool:
        return str(name) in self._items

    # ─── Çağrı ──────────────────────────────────────────────────────────

    def call(self, name: str, **kwargs: Any) -> Any:
        """
        Yeteneği senkron çağırır.

        Args:
            name: Yetenek adı
            **kwargs: Yeteneğe geçirilecek argümanlar

        Raises:
            CapabilityError: Yetenek yoksa ya da çağrı hata verirse.
        """
        item = self._items.get(str(name))
        if item is None:
            self._stats["errors"] += 1
            raise CapabilityError(str(name), "yetenek kayıtlı değil")

        started = time.perf_counter()
        try:
            result = item.fn(**kwargs)
            if inspect.isawaitable(result):
                # Senkron yolda coroutine döndüyse güvenli biçimde çalıştır
                result = self._run_coroutine(result)
        except CapabilityError:
            item.errors += 1
            self._stats["errors"] += 1
            raise
        except Exception as exc:
            item.errors += 1
            self._stats["errors"] += 1
            raise CapabilityError(str(name), f"{type(exc).__name__}: {exc}", exc) from exc
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            item.calls += 1
            item.total_ms += elapsed
            self._stats["calls"] += 1

        return result

    async def acall(self, name: str, **kwargs: Any) -> Any:
        """
        Yeteneği asenkron çağırır (senkron yetenekler de desteklenir).

        Raises:
            CapabilityError: Yetenek yoksa, zaman aşımına uğrarsa ya da hata verirse.
        """
        item = self._items.get(str(name))
        if item is None:
            self._stats["errors"] += 1
            raise CapabilityError(str(name), "yetenek kayıtlı değil")

        started = time.perf_counter()
        try:
            if item.is_async:
                coro = item.fn(**kwargs)
                if item.timeout:
                    result = await asyncio.wait_for(coro, timeout=item.timeout)
                else:
                    result = await coro
            else:
                result = item.fn(**kwargs)
                if inspect.isawaitable(result):
                    if item.timeout:
                        result = await asyncio.wait_for(result, timeout=item.timeout)
                    else:
                        result = await result
        except CapabilityError:
            item.errors += 1
            self._stats["errors"] += 1
            raise
        except asyncio.TimeoutError as exc:
            item.errors += 1
            self._stats["errors"] += 1
            raise CapabilityError(
                str(name), f"zaman aşımı ({item.timeout}s)", exc
            ) from exc
        except Exception as exc:
            item.errors += 1
            self._stats["errors"] += 1
            raise CapabilityError(str(name), f"{type(exc).__name__}: {exc}", exc) from exc
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            item.calls += 1
            item.total_ms += elapsed
            self._stats["calls"] += 1

        return result

    @staticmethod
    def _run_coroutine(coro: Any) -> Any:
        """Çalışan bir event loop yoksa coroutine'i güvenle çalıştırır."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Zaten bir loop var: ayrı bir thread'de tamamla (nadir durum)
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    # ─── İstatistik ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir kayıt defteri metrikleri."""
        return {
            "capabilities": len(self._items),
            "names": list(self._items.keys()),
            "calls": self._stats["calls"],
            "errors": self._stats["errors"],
            "error_rate": round(
                self._stats["errors"] / self._stats["calls"], 4
            ) if self._stats["calls"] else 0.0,
            "details": {name: item.to_dict() for name, item in self._items.items()},
        }

    def reset_stats(self) -> None:
        """Çağrı sayaçlarını sıfırlar."""
        self._stats = {"calls": 0, "errors": 0}
        for item in self._items.values():
            item.calls = 0
            item.errors = 0
            item.total_ms = 0.0


# ─── Yürütme Raporu ─────────────────────────────────────────────────────────

@dataclass
class ExecutionReport:
    """
    Bir plan yürütmesinin sonucu.

    Attributes:
        state: success | partial | failed | aborted | cancelled
        answer: Son başarılı ``SYNTHESIZE`` görevinin sonucu (yoksa son DONE sonucu)
        summary: Tek satırlık Türkçe özet
    """

    goal: str
    graph: TaskGraph
    trace: Optional[ReasoningTrace] = None
    state: str = "pending"
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    answered: bool = False
    answer: Any = None
    summary: str = ""
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    cancelled: int = 0
    replans: int = 0
    checkpoints: int = 0
    error: str = ""
    #: Yürütme boyunca paylaşılan bağlam (görev sonuçları burada birikir)
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Yürütme başarılı mı?"""
        return self.state == "success"

    def to_dict(self) -> Dict[str, Any]:
        """Raporu JSON-serileştirilebilir sözlüğe çevirir (graf dahil)."""
        return {
            "goal": self.goal,
            "state": self.state,
            "answered": self.answered,
            "answer": json_safe(self.answer),
            "summary": self.summary,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": round(self.duration_ms, 2),
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "cancelled": self.cancelled,
            "replans": self.replans,
            "checkpoints": self.checkpoints,
            "error": self.error,
            "graph": self.graph.to_dict(),
            "graph_stats": self.graph.stats(),
            "trace": self.trace.to_dict() if self.trace else None,
            "trace_summary": self.trace.summary() if self.trace else None,
            "progress": self.graph.progress(),
        }


# ─── Yürütücü ───────────────────────────────────────────────────────────────

class Executor:
    """
    Görev grafiğini yürütür.

    Kullanım:
        ex = Executor()
        ex.initialize(config, event_bus, capabilities)
        report = ex.execute(graph, trace, context={"goal": "..."})
        print(report.summary)
    """

    name = "planner.executor"

    #: Yeniden deneme gecikmeleri (saniye) — üstel geri çekilme.
    RETRY_BACKOFF: Tuple[float, ...] = (0.2, 0.5)

    def __init__(self) -> None:
        self._config: Any = None
        self._event_bus: Any = None
        self._capabilities: CapabilityRegistry = CapabilityRegistry()
        self._initialized: bool = False
        self._cancelled: bool = False
        self._offline: bool = False
        self._max_reasoning_steps: int = 50
        self._checkpoint_interval: int = 5
        self._enable_parallel: bool = True
        self._max_depth: int = 10
        self._strict: bool = False
        self._stats: Dict[str, Any] = {
            "executions": 0,
            "tasks_executed": 0,
            "tasks_failed": 0,
            "tasks_skipped": 0,
            "replans": 0,
            "retries": 0,
            "aborts": 0,
            "offline_blocks": 0,
            "last_state": "",
            "last_duration_ms": 0.0,
        }
        self._started_at: float = 0.0

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None,
                   capabilities: Optional[CapabilityRegistry] = None) -> None:
        """
        Yürütücüyü yapılandırır (idempotent).

        Args:
            config: ``SystemConfig`` benzeri nesne (None olabilir)
            event_bus: Olay yolu
            capabilities: Dışarıdan verilen yetenek kayıt defteri
        """
        if self._initialized:
            if config is not None:
                self._config = config
                self._read_config()
            if event_bus is not None:
                self._event_bus = event_bus
            if capabilities is not None:
                self._capabilities = capabilities
            return

        self._config = config
        self._event_bus = event_bus
        if capabilities is not None:
            self._capabilities = capabilities
        self._read_config()
        self._initialized = True
        self._started_at = time.time()
        self._cancelled = False

        logger.info(
            f"Görev yürütücü hazır — {len(self._capabilities)} yetenek, "
            f"paralel={'açık' if self._enable_parallel else 'kapalı'}, "
            f"checkpoint aralığı={self._checkpoint_interval}"
        )

    def shutdown(self) -> None:
        """Yürütücüyü kapatır (idempotent)."""
        if not self._initialized:
            return
        self._initialized = False
        self._cancelled = True
        logger.info(
            f"Görev yürütücü kapatıldı — {self._stats['executions']} yürütme, "
            f"{self._stats['tasks_executed']} görev"
        )

    def _read_config(self) -> None:
        """Konfigürasyondan planlayıcı ayarlarını okur."""
        planner = getattr(self._config, "planner", None)
        self._max_reasoning_steps = int(getattr(planner, "max_reasoning_steps", 50) or 50)
        self._checkpoint_interval = max(1, int(getattr(planner, "checkpoint_interval", 5) or 5))
        self._enable_parallel = bool(getattr(planner, "enable_parallel_tasks", True))
        self._max_depth = int(getattr(planner, "max_task_depth", 10) or 10)

    # ─── Özellikler ─────────────────────────────────────────────────────

    @property
    def capabilities(self) -> CapabilityRegistry:
        """Yetenek kayıt defteri."""
        return self._capabilities

    @property
    def cancelled(self) -> bool:
        """İptal bayrağı."""
        return self._cancelled

    @property
    def offline(self) -> bool:
        """Çevrimdışı kip: ``net.*`` ve ``research`` yetenekleri engellenir."""
        return self._offline

    @offline.setter
    def offline(self, value: bool) -> None:
        self._offline = bool(value)

    def register_capability(self, name: str, fn: Callable[..., Any], **kwargs: Any) -> None:
        """Kısayol: yeteneği kayıt defterine ekler."""
        self._capabilities.register(name, fn, **kwargs)

    # ─── İptal ──────────────────────────────────────────────────────────

    def cancel(self) -> None:
        """
        Yürütmeyi iptal eder.

        Senkron yolda bir sonraki görev sınırında, asenkron yolda bir sonraki
        bekleme noktasında etkili olur.
        """
        self._cancelled = True
        logger.warning("Yürütme iptal istendi")
        self._emit("cancelled", {"reason": "kullanıcı iptali"})

    # ─── Olay Yayınlama ─────────────────────────────────────────────────

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Olay yayınlar (event_bus yoksa sessizce atlar)."""
        if self._event_bus:
            try:
                self._event_bus.emit(f"planner.{event}", data, source=self.name)
            except Exception as exc:  # pragma: no cover - savunmacı
                logger.debug(f"Olay yayınlanamadı ({event}): {exc}")

    # ─── Senkron Yürütme ────────────────────────────────────────────────

    def execute(self, graph: TaskGraph, trace: Optional[ReasoningTrace] = None,
                context: Optional[Dict[str, Any]] = None,
                checkpoint: Optional[Any] = None,
                on_task_done: Optional[Callable[..., Any]] = None,
                budget_seconds: Optional[float] = None,
                offline: Optional[bool] = None) -> ExecutionReport:
        """
        Görev grafiğini **sırayla** (dalga dalga, görevler tek tek) yürütür.

        Args:
            graph: Yürütülecek görev grafiği
            trace: Akıl yürütme zinciri (None ise otomatik oluşturulur)
            context: Paylaşılan bağlam (görev sonuçları buraya yazılır)
            checkpoint: ``CheckpointManager`` benzeri nesne (``save`` metodu beklenir)
            on_task_done: Her tamamlanan görevden sonra çağrılır (task, trace)
            budget_seconds: Duvarsaati bütçesi (None = sınırsız)
            offline: True ise ``net.*`` ve ``research`` yetenekleri engellenir

        Returns:
            ``ExecutionReport`` — asla istisna fırlatmaz, hata durumu raporlanır.
        """
        if not self._initialized:
            self.initialize()
        self._cancelled = False
        if offline is not None:
            self._offline = bool(offline)

        report = self._prepare(graph, trace, context)
        self._emit("plan_started", {
            "goal": report.goal[:200],
            "tasks": len(graph.tasks),
            "trace_id": report.trace.trace_id if report.trace else "",
            "parallel": False,
            "offline": self._offline,
        })
        logger.info(
            f"Plan yürütülüyor (senkron): '{report.goal[:70]}' — "
            f"{len(graph.tasks)} görev, {len(graph.parallel_batches())} dalga"
        )

        try:
            self._run_graph_sync(report, graph, report.trace, report.context,
                                 checkpoint, on_task_done, budget_seconds)
        except ExecutionAborted as exc:
            report.state = "aborted"
            report.error = exc.reason
            self._cancel_remaining(graph, reason=exc.reason)
        except Exception as exc:  # pragma: no cover - savunmacı
            report.state = "failed"
            report.error = f"{type(exc).__name__}: {exc}"
            logger.error(f"Plan yürütme hatası: {report.error}")

        return self._finalize(report, graph)

    async def execute_async(self, graph: TaskGraph, trace: Optional[ReasoningTrace] = None,
                            context: Optional[Dict[str, Any]] = None,
                            checkpoint: Optional[Any] = None,
                            on_task_done: Optional[Callable[..., Any]] = None,
                            budget_seconds: Optional[float] = None,
                            offline: Optional[bool] = None) -> ExecutionReport:
        """
        Görev grafiğini **asenkron** yürütür; paralel dalgalar eşzamanlı koşar.

        Paralellik yalnızca ``enable_parallel_tasks`` açıkken uygulanır ve
        yetenekler ``asyncio.gather`` ile aynı event loop üzerinde çağrılır —
        asla thread havuzu kullanılmaz.
        """
        if not self._initialized:
            self.initialize()
        self._cancelled = False
        if offline is not None:
            self._offline = bool(offline)

        report = self._prepare(graph, trace, context)
        self._emit("plan_started", {
            "goal": report.goal[:200],
            "tasks": len(graph.tasks),
            "trace_id": report.trace.trace_id if report.trace else "",
            "parallel": self._enable_parallel,
            "offline": self._offline,
        })
        logger.info(
            f"Plan yürütülüyor (asenkron): '{report.goal[:70]}' — "
            f"{len(graph.tasks)} görev, paralel={'açık' if self._enable_parallel else 'kapalı'}"
        )

        try:
            await self._run_graph_async(report, graph, report.trace, report.context,
                                        checkpoint, on_task_done, budget_seconds)
        except ExecutionAborted as exc:
            report.state = "aborted"
            report.error = exc.reason
            self._cancel_remaining(graph, reason=exc.reason)
        except asyncio.CancelledError:
            report.state = "cancelled"
            report.error = "görev iptal edildi (asyncio)"
            self._cancel_remaining(graph, reason="asyncio iptali")
            raise
        except Exception as exc:  # pragma: no cover - savunmacı
            report.state = "failed"
            report.error = f"{type(exc).__name__}: {exc}"
            logger.error(f"Async plan yürütme hatası: {report.error}")

        return self._finalize(report, graph)

    # ─── Hazırlık / Sonuçlandırma ───────────────────────────────────────

    def _prepare(self, graph: TaskGraph, trace: Optional[ReasoningTrace],
                 context: Optional[Dict[str, Any]]) -> ExecutionReport:
        """Rapor ve akıl yürütme zincirini hazırlar."""
        if trace is None:
            trace = ReasoningTrace(
                trace_id=f"tr_{uuid4().hex[:8]}",
                goal=graph.goal,
                max_steps=self._max_reasoning_steps,
            )
        trace.max_steps = int(self._max_reasoning_steps)

        ctx: Dict[str, Any] = dict(context or {})
        ctx.setdefault("goal", graph.goal)

        report = ExecutionReport(
            goal=graph.goal,
            graph=graph,
            trace=trace,
            state="pending",
            started_at=time.time(),
            context=ctx,
        )

        problems = graph.validate(max_depth=self._max_depth)
        if problems:
            logger.warning(f"⚠ Plan doğrulama uyarıları: {'; '.join(problems[:4])}")
            self._emit("plan_warnings", {"goal": graph.goal[:160], "problems": problems[:10]})
        return report

    def _finalize(self, report: ExecutionReport, graph: TaskGraph) -> ExecutionReport:
        """Raporu kapatır: durum, yanıt, özet ve istatistikler."""
        trace = report.trace
        if trace is not None:
            trace.finish()

        states = graph.counts_by_state()
        report.completed = states.get(TaskState.DONE.value, 0)
        report.failed = states.get(TaskState.FAILED.value, 0)
        report.skipped = states.get(TaskState.SKIPPED.value, 0)
        report.cancelled = states.get(TaskState.CANCELLED.value, 0)
        report.finished_at = time.time()
        report.duration_ms = max(0.0, (report.finished_at - report.started_at) * 1000.0)

        answer, answered = self._extract_answer(graph)
        report.answer = answer
        report.answered = answered

        # ─── Durum Belirleme ────────────────────────────────────────────
        # "aborted"/"cancelled": bütçe/iptal kararı korunur.
        # "failed": kritik görev kurtarılamadı → kısmi yanıtla yumuşatılmaz.
        # "partial": bir görev kurtarılamadı ama yanıt üretildi (bkz.
        #   `_handle_failure` — sessiz yutma yok).
        if report.state == "partial":
            # Kurtarılamayan başarısızlık kaydedildi; yanıt varsa kısmi kalır,
            # yoksa başarısız sayılır.
            if not answered and report.completed == 0:
                report.state = "failed"
        elif report.state not in ("aborted", "cancelled", "failed"):
            if answered and report.failed == 0 and report.skipped == 0 and report.cancelled == 0:
                report.state = "success"
            elif answered:
                report.state = "partial"
            elif report.completed > 0:
                report.state = "partial"
            else:
                report.state = "failed"

        report.summary = (
            f"{report.state} · {report.completed}/{len(graph.tasks)} görev tamam · "
            f"{report.duration_ms:.0f}ms"
            + (f" · {report.failed} hata" if report.failed else "")
            + (f" · {report.skipped} atlandı" if report.skipped else "")
            + (f" · {report.cancelled} iptal" if report.cancelled else "")
            + (f" · {report.replans} yeniden plan" if report.replans else "")
        )

        self._stats["executions"] += 1
        self._stats["last_state"] = report.state
        self._stats["last_duration_ms"] = round(report.duration_ms, 2)
        self._stats["tasks_executed"] += report.completed
        self._stats["tasks_failed"] += report.failed
        self._stats["tasks_skipped"] += report.skipped

        self._emit("plan_finished", {
            "goal": report.goal[:200],
            "state": report.state,
            "answered": report.answered,
            "completed": report.completed,
            "failed": report.failed,
            "duration_ms": round(report.duration_ms, 2),
            "replans": report.replans,
        })
        logger.info(f"Plan tamamlandı — {report.summary}")
        return report

    @staticmethod
    def _extract_answer(graph: TaskGraph) -> Tuple[Any, bool]:
        """
        Nihai yanıtı seçer.

        Öncelik: son başarılı ``SYNTHESIZE`` görevi; yoksa son başarılı görev.
        Sentez sonucu ``{"answer": ...}`` gibi bilinen bir sarmalayıcı ise
        iç değer açılır (kullanıcıya düz yanıt verilir).
        """
        synth: Optional[Task] = None
        for task in graph.tasks.values():
            if task.kind == TaskKind.SYNTHESIZE and task.state == TaskState.DONE:
                synth = task
        if synth is not None:
            return _unwrap_answer(synth.result), True

        last: Optional[Task] = None
        for task in graph.tasks.values():
            if task.state == TaskState.DONE:
                last = task
        if last is not None:
            return _unwrap_answer(last.result), False
        return None, False

    # ─── Dalga Yürütme (Senkron) ────────────────────────────────────────

    def _run_graph_sync(self, report: ExecutionReport, graph: TaskGraph,
                        trace: Optional[ReasoningTrace], context: Dict[str, Any],
                        checkpoint: Optional[Any], on_task_done: Optional[Callable[..., Any]],
                        budget_seconds: Optional[float]) -> None:
        """Dalgaları sırayla, dalga içindeki görevleri tek tek yürütür."""
        deadline = (time.time() + budget_seconds) if budget_seconds else None
        safety = len(graph.tasks) * 3 + 5

        for _ in range(safety):
            if self._check_stop(report, graph, deadline, context):
                return
            ready = self._ready_ids(graph)
            if not ready:
                break

            for task_id in ready:
                if self._check_stop(report, graph, deadline, context):
                    return
                task = graph.tasks.get(task_id)
                if task is None or task.state != TaskState.PENDING:
                    continue
                outcome = self._execute_task_sync(task, graph, trace, context)
                self._after_task(report, graph, task, trace, context, checkpoint,
                                 on_task_done, outcome)
                if report.state == "aborted":
                    return

    async def _run_graph_async(self, report: ExecutionReport, graph: TaskGraph,
                               trace: Optional[ReasoningTrace], context: Dict[str, Any],
                               checkpoint: Optional[Any], on_task_done: Optional[Callable[..., Any]],
                               budget_seconds: Optional[float]) -> None:
        """Dalgaları yürütür; paralel açıkken dalga içi görevler eşzamanlı koşar."""
        deadline = (time.time() + budget_seconds) if budget_seconds else None
        safety = len(graph.tasks) * 3 + 5

        for _ in range(safety):
            if self._check_stop(report, graph, deadline, context):
                return
            ready = self._ready_ids(graph)
            if not ready:
                break

            if self._enable_parallel and len(ready) > 1:
                results = await asyncio.gather(
                    *[self._execute_task_async(graph.tasks[tid], graph, trace, context)
                      for tid in ready],
                    return_exceptions=True,
                )
                for task_id, outcome in zip(ready, results):
                    task = graph.tasks.get(task_id)
                    if task is None:
                        continue
                    if isinstance(outcome, BaseException):
                        if isinstance(outcome, asyncio.CancelledError):
                            raise outcome
                        logger.error(f"Paralel görev hatası ('{task_id}'): {outcome}")
                        outcome = _StepOutcome(task_id=task_id, state=TaskState.FAILED,
                                               error=f"{type(outcome).__name__}: {outcome}")
                    self._after_task(report, graph, task, trace, context, checkpoint,
                                     on_task_done, outcome)
                    if report.state == "aborted":
                        return
            else:
                for task_id in ready:
                    if self._check_stop(report, graph, deadline, context):
                        return
                    task = graph.tasks.get(task_id)
                    if task is None or task.state != TaskState.PENDING:
                        continue
                    outcome = await self._execute_task_async(task, graph, trace, context)
                    self._after_task(report, graph, task, trace, context, checkpoint,
                                     on_task_done, outcome)
                    if report.state == "aborted":
                        return

    def _ready_ids(self, graph: TaskGraph) -> List[str]:
        """
        Şu anda yürütülebilecek görevler.

        ``TaskGraph.ready_tasks`` ile aynıdır; ek olarak **yumuşak bağımlılık**
        (``args._optional_deps``) desteği sunar: yumuşak bağımlılıklar
        sonlanmışsa (DONE/FAILED/SKIPPED/CANCELLED) görev yine hazır sayılır.
        Bu sayede çevrimdışı engelli bir kanat, planın kalanını kilitlemez.
        """
        ready: List[str] = []
        for task in graph.tasks.values():
            if task.state != TaskState.PENDING:
                continue
            deps = graph.dependencies_of(task.id)
            if not deps:
                ready.append(task.id)
                continue
            optional = set(task.args.get("_optional_deps") or [])
            ok = True
            for dep in deps:
                dep_state = graph.tasks[dep].state
                if dep in optional:
                    if dep_state not in (TaskState.DONE, TaskState.FAILED,
                                         TaskState.SKIPPED, TaskState.CANCELLED):
                        ok = False
                        break
                elif dep_state != TaskState.DONE:
                    ok = False
                    break
            if ok:
                ready.append(task.id)
        return sorted(ready, key=graph._sort_key)  # noqa: SLF001 - aynı paket içi kullanım

    def _check_stop(self, report: ExecutionReport, graph: TaskGraph,
                    deadline: Optional[float], context: Dict[str, Any]) -> bool:
        """
        İptal / bütçe denetimi.

        Returns:
            True ise yürütme durdurulmalı.
        """
        if self._cancelled:
            self._cancel_remaining(graph, reason="kullanıcı iptali")
            report.state = "cancelled"
            report.error = "kullanıcı iptali"
            return True

        if deadline is not None and time.time() > deadline:
            self._stats["aborts"] += 1
            self._cancel_remaining(graph, reason="bütçe aşıldı")
            report.state = "aborted"
            report.error = "duvarsaati bütçesi aşıldı"
            self._emit("budget_exceeded", {
                "goal": report.goal[:160],
                "elapsed_ms": round((time.time() - report.started_at) * 1000.0, 2),
            })
            logger.warning("⚠ Duvarsaati bütçesi aşıldı — plan iptal ediliyor")
            return True

        if not self._ready_ids(graph) and any(
            t.state not in (TaskState.DONE, TaskState.FAILED, TaskState.SKIPPED, TaskState.CANCELLED)
            for t in graph.tasks.values()
        ):
            # Kilitlenme: hazır görev yok ama sonlanmamış görev var → kalanları iptal et
            self._cancel_remaining(graph, reason="plan kilitlendi")
            logger.warning("⚠ Plan kilitlendi (hazır görev yok) — kalan görevler iptal edildi")
            return True

        return False

    # ─── Görev Yürütme ──────────────────────────────────────────────────

    def _execute_task_sync(self, task: Task, graph: TaskGraph,
                           trace: Optional[ReasoningTrace],
                           context: Dict[str, Any]) -> "_StepOutcome":
        """Görevi senkron yürütür (denemeler + geri çekilme dahil)."""
        self._trace_action(trace, task)
        self._emit("step_started", self._task_event_payload(task, graph))

        attempts = max(1, int(task.max_attempts or 1))
        last_error = ""
        result: Any = None

        for attempt in range(1, attempts + 1):
            task.touch_started()
            if attempt > 1:
                self._stats["retries"] += 1
                delay = self.RETRY_BACKOFF[min(attempt - 2, len(self.RETRY_BACKOFF) - 1)]
                logger.info(f"↻ '{task.id}' yeniden deneniyor ({attempt}/{attempts}) — {delay}s bekleme")
                time.sleep(delay)
                if self._cancelled:
                    break
            try:
                result = self._run_task(task, context)
                return self._mark_done(task, graph, trace, context, result)
            except _OfflineBlocked as exc:
                return self._mark_skipped(task, graph, trace, context, str(exc))
            except Exception as exc:
                last_error = self._error_message(task, exc)
                task.error = last_error
                task.touch_finished()
                self._trace_critique(trace, task, last_error, attempt, attempts)
                self._emit("step_failed", {
                    **self._task_event_payload(task, graph),
                    "error": last_error,
                    "attempt": attempt,
                })
                logger.warning(
                    f"✗ '{task.id}' ({task.kind.value}) denemesi başarısız: {last_error}"
                )

        task.state = TaskState.FAILED
        if not task.finished_at:
            task.touch_finished()
        return _StepOutcome(task_id=task.id, state=TaskState.FAILED, error=last_error)

    async def _execute_task_async(self, task: Task, graph: TaskGraph,
                                  trace: Optional[ReasoningTrace],
                                  context: Dict[str, Any]) -> "_StepOutcome":
        """Görevi asenkron yürütür (bekleme ``asyncio.sleep`` ile — asla bloklamaz)."""
        self._trace_action(trace, task)
        self._emit("step_started", self._task_event_payload(task, graph))

        attempts = max(1, int(task.max_attempts or 1))
        last_error = ""
        result: Any = None

        for attempt in range(1, attempts + 1):
            task.touch_started()
            if attempt > 1:
                self._stats["retries"] += 1
                delay = self.RETRY_BACKOFF[min(attempt - 2, len(self.RETRY_BACKOFF) - 1)]
                logger.info(f"↻ '{task.id}' yeniden deneniyor ({attempt}/{attempts}) — {delay}s bekleme")
                await asyncio.sleep(delay)
                if self._cancelled:
                    break
            try:
                result = await self._run_task_async(task, context)
                return self._mark_done(task, graph, trace, context, result)
            except _OfflineBlocked as exc:
                return self._mark_skipped(task, graph, trace, context, str(exc))
            except Exception as exc:
                last_error = self._error_message(task, exc)
                task.error = last_error
                task.touch_finished()
                self._trace_critique(trace, task, last_error, attempt, attempts)
                self._emit("step_failed", {
                    **self._task_event_payload(task, graph),
                    "error": last_error,
                    "attempt": attempt,
                })
                logger.warning(
                    f"✗ '{task.id}' ({task.kind.value}) denemesi başarısız: {last_error}"
                )

        task.state = TaskState.FAILED
        if not task.finished_at:
            task.touch_finished()
        return _StepOutcome(task_id=task.id, state=TaskState.FAILED, error=last_error)

    def _run_task(self, task: Task, context: Dict[str, Any]) -> Any:
        """
        Tek bir yetenek çağrısını yapar (senkron).

        Yeteneği olmayan görevler (``reason`` gibi) bağlamdan türetilmiş bir
        yer tutucu sonuç üretir — plan yine de ilerler.
        """
        args = self._resolve_args(task, context)
        if not task.tool:
            return self._synthetic_result(task, args, context)

        self._guard_offline(task)
        return self._capabilities.call(task.tool, **args)

    async def _run_task_async(self, task: Task, context: Dict[str, Any]) -> Any:
        """Tek bir yetenek çağrısını yapar (asenkron)."""
        args = self._resolve_args(task, context)
        if not task.tool:
            return self._synthetic_result(task, args, context)

        self._guard_offline(task)
        return await self._capabilities.acall(task.tool, **args)

    def _guard_offline(self, task: Task) -> None:
        """
        Çevrimdışı kipte ağ gerektiren yetenekleri engeller.

        Raises:
            _OfflineBlocked: Yetenek ``net.`` ile başlıyorsa veya ``research`` ise.
        """
        if not self._offline:
            return
        name = (task.tool or "").lower()
        if name.startswith("net.") or name == "research" or name.startswith("research."):
            self._stats["offline_blocks"] += 1
            raise _OfflineBlocked(
                f"çevrimdışı kip: '{task.tool}' yeteneği devre dışı"
            )

    def _synthetic_result(self, task: Task, args: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Yeteneksiz görevler için sentetik sonuç üretir.

        Böylece yalnızca birkaç yetenek kayıtlı olsa bile tüm şablonlar
        uçtan uca çalışır (zarif bozulma).
        """
        upstream = {
            dep: json_safe(context.get(dep))
            for dep in task.depends_on
            if dep in context
        }
        payload: Dict[str, Any] = {
            "kind": task.kind.value,
            "task": task.id,
            "name": task.name,
            "goal": context.get("goal", ""),
            "synthetic": True,
            "note": f"'{task.tool or task.kind.value}' yeteneği çağrılmadı; bağlamdan türetildi",
        }
        if upstream:
            payload["upstream"] = upstream
        if args:
            payload["args"] = json_safe(args)

        if task.kind == TaskKind.SYNTHESIZE:
            payload["answer"] = self._synthetic_answer(task, context)
        return payload

    def _synthetic_answer(self, task: Task, context: Dict[str, Any]) -> str:
        """Yetenek yokken üretilebilecek en iyi Türkçe yanıt taslağı."""
        goal = str(context.get("goal", "")).strip()
        parts = [f"Hedef: {goal}"] if goal else []
        for dep in task.depends_on:
            value = context.get(dep)
            if value is None:
                continue
            parts.append(f"- {dep}: {json_safe(value)}")
        if not parts:
            return "Yanıt üretilemedi: sentez yeteneği kayıtlı değil."
        return "\n".join(parts)

    # ─── Görev Sonu İşlemleri ───────────────────────────────────────────

    def _mark_done(self, task: Task, graph: TaskGraph, trace: Optional[ReasoningTrace],
                   context: Dict[str, Any], result: Any) -> "_StepOutcome":
        """Görevi DONE olarak işaretler, bağlamı ve izi günceller."""
        task.state = TaskState.DONE
        task.result = result
        task.error = ""
        task.touch_finished()
        task.confidence = self._confidence(result, task)

        safe = json_safe(result)
        context[task.id] = safe
        context.setdefault("results", {})[task.id] = safe

        self._trace_observation(trace, task, safe)
        self._emit("step_finished", {
            **self._task_event_payload(task, graph),
            "confidence": task.confidence,
            "result_preview": _preview(safe),
        })
        return _StepOutcome(task_id=task.id, state=TaskState.DONE, result=safe,
                            confidence=task.confidence)

    def _mark_skipped(self, task: Task, graph: TaskGraph, trace: Optional[ReasoningTrace],
                      context: Dict[str, Any], reason: str) -> "_StepOutcome":
        """Görevi SKIPPED olarak işaretler (engel nedeniyle)."""
        task.state = TaskState.SKIPPED
        task.error = reason
        task.touch_finished()
        if trace is not None:
            trace.add("observation", f"{task.id} atlandı: {reason}", task_id=task.id)
        self._emit("step_skipped", {**self._task_event_payload(task, graph), "reason": reason})
        logger.warning(f"⊘ '{task.id}' atlandı: {reason}")
        return _StepOutcome(task_id=task.id, state=TaskState.SKIPPED, error=reason)

    def _after_task(self, report: ExecutionReport, graph: TaskGraph, task: Task,
                    trace: Optional[ReasoningTrace], context: Dict[str, Any],
                    checkpoint: Optional[Any], on_task_done: Optional[Callable[..., Any]],
                    outcome: "_StepOutcome") -> None:
        """
        Görev sonrası ortak akış: geri bildirim, checkpoint, yeniden planlama.
        """
        if on_task_done is not None:
            try:
                on_task_done(task, trace)
            except Exception as exc:  # pragma: no cover - savunmacı
                logger.debug(f"on_task_done geri çağrısı hata verdi: {exc}")

        if outcome.state == TaskState.FAILED:
            self._handle_failure(report, graph, task, trace, context, checkpoint)
            return

        if outcome.state == TaskState.SKIPPED:
            # Çevrimdışı engeli bilinçli bir kısıtlamadır: plan kurtarmaya girmez,
            # yalnızca atlanan göreve bağlı dallar atlanır.
            self._skip_dependents(graph, task, reason=f"bağımlılık atlandı: {task.id}")

        done_count = sum(1 for t in graph.tasks.values() if t.state == TaskState.DONE)
        if (checkpoint is not None and done_count > 0
                and done_count % self._checkpoint_interval == 0):
            saved = self._save_checkpoint(checkpoint, graph, trace, context,
                                          note=f"{done_count} görev tamamlandı")
            if saved:
                report.checkpoints += 1

    def _handle_failure(self, report: ExecutionReport, graph: TaskGraph, task: Task,
                        trace: Optional[ReasoningTrace], context: Dict[str, Any],
                        checkpoint: Optional[Any]) -> None:
        """
        Görev başarısızlığı politikası.

        * ``critical`` görev ya da ``SYNTHESIZE``: yeniden planlama zorunlu;
          başarısız olursa plan iptal edilir.
        * Diğer görevler: bağımlıları atlanır; bağımlı varsa yeniden planlanır.
        """
        dependents = graph.dependents_of(task.id)
        must_replan = task.critical or task.kind == TaskKind.SYNTHESIZE

        # Kritik görev: kurtarma planı eklenmez — spec gereği plan iptal edilir.
        if must_replan:
            reason = f"kritik görev başarısız ve kurtarılamadı: {task.id}"
            logger.error(f"✗ {reason} — plan iptal ediliyor")
            self._cancel_remaining(graph, reason=reason)
            report.state = "failed"
            report.error = reason
            self._emit("plan_aborted", {"goal": report.goal[:160], "reason": reason})
            return

        if not dependents:
            logger.info(f"'{task.id}' başarısız ama bağımlısı yok — plan devam ediyor (kısmi)")
            # Şeffaflık: kurtarılamayan bir başarısızlık sessizce yutulmaz.
            # Plan yanıt üretmeye devam eder ama durum "partial"a düşer,
            # böylece motor/evrim katmanı kaliteyi doğru değerlendirebilir.
            if report.state not in ("failed", "aborted", "cancelled"):
                report.state = "partial"
            return

        try:
            modified = self.replan(graph, task, trace)
        except Exception as exc:  # pragma: no cover - savunmacı
            modified = False
            logger.error(f"Yeniden planlama hatası: {exc}")

        if modified:
            report.replans += 1
            self._stats["replans"] += 1
            if checkpoint is not None:
                if self._save_checkpoint(checkpoint, graph, trace, context,
                                         note=f"yeniden planlama: {task.id}"):
                    report.checkpoints += 1
            logger.info(f"↺ Plan yeniden düzenlendi ('{task.id}' sonrası)")
            return

        # Kurtarma planı kurulamadı: yalnızca bağımlılar atlanır (rapor kısmi kalır).
        self._skip_dependents(graph, task, reason=f"başarısız bağımlılık: {task.id}")
        self._stats["tasks_skipped"] += len(dependents)

    def _skip_dependents(self, graph: TaskGraph, task: Task, reason: str) -> List[str]:
        """Başarısız/atlanmış bir göreve bağlı, henüz başlamamış görevleri atlar."""
        skipped: List[str] = []
        for dependent_id in graph.dependents_of(task.id):
            dependent = graph.tasks.get(dependent_id)
            if dependent is None or dependent.state != TaskState.PENDING:
                continue
            dependent.state = TaskState.SKIPPED
            dependent.error = reason
            dependent.touch_finished()
            skipped.append(dependent_id)
            self._emit("step_skipped", {**self._task_event_payload(dependent, graph),
                                        "reason": reason})
            logger.warning(f"⊘ '{dependent_id}' atlandı: {reason}")
            skipped.extend(self._skip_dependents(graph, dependent, reason=reason))
        return skipped

    def _cancel_remaining(self, graph: TaskGraph, reason: str = "iptal edildi") -> int:
        """Sonlanmamış tüm görevleri CANCELLED yapar."""
        count = 0
        for task in graph.tasks.values():
            if task.state in (TaskState.PENDING, TaskState.RUNNING, TaskState.BLOCKED):
                task.state = TaskState.CANCELLED
                task.error = reason
                count += 1
        if count:
            logger.warning(f"⊘ {count} görev iptal edildi ({reason})")
        return count

    def _save_checkpoint(self, checkpoint: Any, graph: TaskGraph,
                         trace: Optional[ReasoningTrace], context: Dict[str, Any],
                         note: str = "") -> Optional[str]:
        """Checkpoint yazar (hata olsa bile yürütmeyi durdurmaz)."""
        saver = getattr(checkpoint, "save", None)
        if not callable(saver):
            return None
        try:
            checkpoint_id = saver(graph, trace, context=context, note=note)
            logger.debug(f"Checkpoint kaydedildi: {checkpoint_id} ({note})")
            return str(checkpoint_id)
        except Exception as exc:
            logger.warning(f"⚠ Checkpoint kaydedilemedi: {exc}")
            return None

    # ─── Akıl Yürütme İzi ───────────────────────────────────────────────

    def _trace_action(self, trace: Optional[ReasoningTrace], task: Task) -> None:
        """Görev öncesi ``action`` adımı ekler."""
        if trace is None:
            return
        detail = task.tool or task.kind.value
        trace.add("action", f"{task.name} ({detail})", task_id=task.id)

    def _trace_observation(self, trace: Optional[ReasoningTrace], task: Task,
                           result: Any) -> None:
        """Görev sonrası ``observation`` adımı ekler (≤500 karakter)."""
        if trace is None:
            return
        trace.add(
            "observation",
            _preview(result, limit=500),
            task_id=task.id,
            confidence=task.confidence,
        )

    def _trace_critique(self, trace: Optional[ReasoningTrace], task: Task, error: str,
                        attempt: int, attempts: int) -> None:
        """Başarısız denemeden sonra ``critique`` adımı ekler."""
        if trace is None:
            return
        trace.add(
            "critique",
            f"{task.id} başarısız (deneme {attempt}/{attempts}): {error}",
            task_id=task.id,
        )

    # ─── Argüman Çözümleme ──────────────────────────────────────────────

    def _resolve_args(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Görev argümanlarındaki şablon ifadelerini çözer.

        Desteklenen biçimler (dize içinde de geçebilir):
            ``${t1.result}``         → t1 görevinin sonucu
            ``${t1.result.field}``   → sonucun ``field`` alanı
            ``${t1.result.a.b}``     → iç içe alan
            ``${context.x}``         → paylaşılan bağlamdan ``x``
            ``${context}``           → bağlamın tamamı
            ``${goal}``              → hedef kısayolu

        Çözülemeyen ifadeler olduğu gibi bırakılır (plan çökmez).
        """
        resolved: Dict[str, Any] = {}
        for key, value in (task.args or {}).items():
            resolved[key] = self._resolve_value(value, context)
        return resolved

    def _resolve_value(self, value: Any, context: Dict[str, Any]) -> Any:
        """Tek bir argüman değerini (gerekiyorsa özyinelemeli) çözer."""
        if isinstance(value, str):
            return self._resolve_string(value, context)
        if isinstance(value, dict):
            return {k: self._resolve_value(v, context) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_value(v, context) for v in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_value(v, context) for v in value)
        return value

    def _resolve_string(self, text: str, context: Dict[str, Any]) -> Any:
        """Dize içindeki ``${...}`` ifadelerini çözer (tam eşleşmede tip korunur)."""
        if "${" not in text:
            return text

        # Tüm dize tek bir ifadeyse özgün tipi koru (int/dict/liste)
        stripped = text.strip()
        if stripped.startswith("${") and stripped.endswith("}") and stripped.count("${") == 1:
            sentinel = object()
            resolved = self._lookup(stripped[2:-1], context, sentinel)
            if resolved is not sentinel:
                return resolved
            return text

        def replace(match: "Any") -> str:
            expression = match.group(1)
            sentinel = object()
            resolved = self._lookup(expression, context, sentinel)
            if resolved is sentinel:
                return match.group(0)
            return _stringify(resolved)

        return re.sub(r"\$\{([^}]+)\}", replace, text)

    def _lookup(self, expression: str, context: Dict[str, Any], default: Any) -> Any:
        """
        ``task_id.result.field`` / ``context.x`` biçimindeki ifadeyi çözer.

        Returns:
            Bulunan değer ya da ``default``.
        """
        expr = (expression or "").strip()
        if not expr:
            return default

        parts = [p for p in expr.split(".") if p != ""]
        if not parts:
            return default

        head = parts[0]
        if head in ("context", "ctx"):
            if len(parts) == 1:
                return context
            current: Any = context
            rest = parts[1:]
        elif head in ("goal", "hedef"):
            return context.get("goal", default)
        else:
            if head not in context:
                return default
            current = context[head]
            rest = parts[1:]
            if rest and rest[0] == "result":
                rest = rest[1:]

        for key in rest:
            if isinstance(current, dict):
                if key not in current:
                    return default
                current = current[key]
            elif isinstance(current, (list, tuple)):
                try:
                    current = current[int(key)]
                except (ValueError, IndexError):
                    return default
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return default
        return current

    # ─── Yeniden Planlama ───────────────────────────────────────────────

    def replan(self, graph: TaskGraph, failed_task: Task,
               trace: Optional[ReasoningTrace] = None) -> bool:
        """
        Başarısız görev için kurtarma planı ekler.

        Yapılanlar:
            1. Başarısız görev ``FAILED``, başlamamış bağımlıları ``SKIPPED`` işaretlenir.
            2. Bir ``reason`` görevi eklenir — hatanın nedenini açıklar.
            3. Bir ``retrieve`` görevi eklenir — alternatif bilgi yolu dener.
            4. Atlanan bağımlılar kurtarma görevlerine yeniden bağlanır (``PENDING``).

        Args:
            graph: Görev grafiği (yerinde değiştirilir)
            failed_task: Başarısız görev
            trace: Akıl yürütme zinciri (opsiyonel)

        Returns:
            True ise graf değiştirildi.
        """
        if not isinstance(graph, TaskGraph):
            raise PlanError("replan bir TaskGraph bekler")
        if failed_task is None or failed_task.id not in graph.tasks:
            return False

        task = graph.tasks[failed_task.id]
        task.state = TaskState.FAILED
        if not task.error:
            task.error = "görev başarısız (neden belirtilmedi)"
        if not task.finished_at:
            task.touch_finished()

        dependents = [
            graph.tasks[d] for d in graph.dependents_of(task.id)
            if graph.tasks[d].state in (TaskState.PENDING, TaskState.SKIPPED, TaskState.BLOCKED)
        ]
        for dependent in dependents:
            if dependent.state == TaskState.PENDING:
                dependent.state = TaskState.SKIPPED

        base_id = f"{task.id}_r{self._stats['replans'] + 1}"
        reason_id = f"{base_id}a"
        retrieve_id = f"{base_id}b"

        # Kimlik çakışmasını önle
        suffix = 1
        while reason_id in graph.tasks or retrieve_id in graph.tasks:
            reason_id = f"{base_id}a{suffix}"
            retrieve_id = f"{base_id}b{suffix}"
            suffix += 1

        goal = graph.goal or task.name
        reason_task = Task(
            id=reason_id,
            name=f"Hatayı çözümle: {task.name[:50]}",
            kind=TaskKind.REASON,
            tool="reason",
            description=f"'{task.id}' görevinin başarısızlığını açıkla ve çözüm yolu öner",
            args={
                "goal": "${context.goal}",
                "question": f"'{task.id}' görevi neden başarısız oldu ve nasıl çözülür?",
                "failure": {
                    "task_id": task.id,
                    "tool": task.tool,
                    "kind": task.kind.value,
                    "error": task.error,
                    "attempts": task.attempts,
                },
                "instruction": "Kök nedeni ve uygulanabilir alternatifi Türkçe açıkla",
            },
            priority=1,
            depth=task.depth + 1,
            max_attempts=1,
            tags={TaskKind.REASON.value, "recovery", "planner"},
        )
        retrieve_task = Task(
            id=retrieve_id,
            name=f"Alternatif bilgi yolu: {task.name[:44]}",
            kind=TaskKind.RETRIEVE,
            tool="retrieve",
            description=f"'{task.id}' başarısız olduğu için farklı sorgu/araçla bilgi ara",
            args={
                "query": f"{goal} {task.name}",
                "exclude_tool": task.tool,
                "limit": 8,
                "recovery": True,
            },
            priority=2,
            depth=task.depth + 1,
            max_attempts=1,
            tags={TaskKind.RETRIEVE.value, "recovery", "planner"},
        )

        try:
            graph.add(reason_task)
            graph.add(retrieve_task)
        except PlanError as exc:
            logger.warning(f"⚠ Yeniden planlama eklenemedi: {exc}")
            return False

        # Kurtarma görevleri, başarısız görevin bağımlılıklarını devralır
        if task.depends_on:
            graph.set_dependencies(reason_id, list(task.depends_on))

        # Atlanan bağımlıları kurtarma görevlerine bağla ve yeniden hazırla
        for dependent in dependents:
            new_deps = list(dependent.depends_on)
            if task.id in new_deps:
                new_deps = [d for d in new_deps if d != task.id]
            for recovery in (reason_id, retrieve_id):
                if recovery not in new_deps:
                    new_deps.append(recovery)
            graph.set_dependencies(dependent.id, new_deps)
            if dependent.state == TaskState.SKIPPED:
                dependent.state = TaskState.PENDING
                dependent.error = ""
            dependent.tags.add("recovered")

        if trace is not None:
            trace.add(
                "reflection",
                f"'{task.id}' başarısız → kurtarma planı eklendi "
                f"({reason_id}, {retrieve_id}); {len(dependents)} bağımlı yeniden bağlandı",
                task_id=task.id,
            )

        self._emit("replanned", {
            "goal": str(goal)[:160],
            "failed_task": task.id,
            "reason_task": reason_id,
            "retrieve_task": retrieve_id,
            "affected": [d.id for d in dependents],
            "total_tasks": len(graph.tasks),
        })
        logger.info(
            f"↺ Yeniden planlama: '{task.id}' başarısız → '{reason_id}' + '{retrieve_id}', "
            f"{len(dependents)} bağımlı yeniden bağlandı"
        )
        return True

    # ─── Yardımcılar ────────────────────────────────────────────────────

    @staticmethod
    def _error_message(task: Task, exc: BaseException) -> str:
        """İstisnayı Türkçe hata mesajına çevirir (sınıf adı korunur)."""
        if isinstance(exc, CapabilityError):
            return f"yetenek hatası [{exc.name}]: {exc.reason}"
        if isinstance(exc, PlanError):
            return f"plan hatası: {exc.message}"
        return f"beklenmeyen hata ({type(exc).__name__}): {exc}"

    @staticmethod
    def _confidence(result: Any, task: Task) -> float:
        """Sonuçtan 0..1 güven skoru çıkarır."""
        if isinstance(result, dict):
            for key in ("confidence", "score", "güven", "guven"):
                value = result.get(key)
                if isinstance(value, (int, float)):
                    return max(0.0, min(1.0, float(value)))
            if result.get("ok") is True:
                return 0.8
            if result.get("stored") is True:
                return 0.7
            if result.get("answer"):
                return 0.6
        if isinstance(result, str) and result.strip():
            return 0.5
        return 0.4 if result else 0.2

    def _task_event_payload(self, task: Task, graph: TaskGraph) -> Dict[str, Any]:
        """Olay verisi için ortak görev yükü."""
        return {
            "goal": str(graph.goal)[:200],
            "task_id": task.id,
            "name": task.name,
            "kind": task.kind.value,
            "tool": task.tool,
            "state": task.state.value,
            "depth": task.depth,
            "progress": graph.progress(),
            "duration_ms": round(task.duration_ms, 2),
        }

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir yürütücü metrikleri."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "executions": self._stats["executions"],
            "tasks_executed": self._stats["tasks_executed"],
            "tasks_failed": self._stats["tasks_failed"],
            "tasks_skipped": self._stats["tasks_skipped"],
            "replans": self._stats["replans"],
            "retries": self._stats["retries"],
            "aborts": self._stats["aborts"],
            "offline_blocks": self._stats["offline_blocks"],
            "last_state": self._stats["last_state"],
            "last_duration_ms": self._stats["last_duration_ms"],
            "success_rate": round(
                1.0 - (self._stats["tasks_failed"] / self._stats["tasks_executed"]
                       if self._stats["tasks_executed"] else 0.0),
                4,
            ),
            "enable_parallel_tasks": self._enable_parallel,
            "checkpoint_interval": self._checkpoint_interval,
            "max_reasoning_steps": self._max_reasoning_steps,
            "offline": self._offline,
            "cancelled": self._cancelled,
            "capabilities": len(self._capabilities),
            "uptime_s": round(time.time() - self._started_at, 2) if self._started_at else 0.0,
        }


# ─── Dahili Veri Yapıları ───────────────────────────────────────────────────

@dataclass
class _StepOutcome:
    """Tek bir görev yürütmesinin iç sonucu."""

    task_id: str
    state: TaskState
    result: Any = None
    error: str = ""
    confidence: float = 0.0


class _OfflineBlocked(Exception):
    """Çevrimdışı kipte engellenen yetenek (görev atlanır, plan çökmez)."""


# ─── Modül Yardımcıları ─────────────────────────────────────────────────────

def _unwrap_answer(value: Any) -> Any:
    """
    Sentez sonucundaki bilinen sarmalayıcıyı açar.

    ``{"answer": "..."}`` / ``{"response": "..."}`` / ``{"text": "..."}``
    biçimlerindeki tek anahtarlı sözlükler iç değere indirgenir; diğer
    yapılar olduğu gibi döner.
    """
    if not isinstance(value, dict):
        return value
    for key in ("answer", "response", "text", "yanit", "yanıt"):
        if key in value:
            inner = value[key]
            if isinstance(inner, (str, int, float, bool, list, dict)) or inner is None:
                return inner
    return value


def _preview(value: Any, limit: int = 500) -> str:
    """Sonucu tek satırlık, kısaltılmış önizlemeye çevirir."""
    try:
        text = json.dumps(json_safe(value), ensure_ascii=False)
    except Exception:  # pragma: no cover - savunmacı
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _stringify(value: Any) -> str:
    """Şablon yerine geçirilecek değeri dizeye çevirir."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(json_safe(value), ensure_ascii=False)
    except Exception:  # pragma: no cover - savunmacı
        return str(value)
