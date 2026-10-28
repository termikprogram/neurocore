"""
NeuroCore Planlayıcı Katmanı (L7)
==================================
Görev ağacı · akıl yürütme zinciri · checkpoint · yeniden planlama ·
paralel yürütme (ARCHITECTURE §10, §13).

Katman, diğer NeuroCore katmanlarını **import etmez**. Bunun yerine
katmanlar kendilerini yetenek (capability) olarak kaydeder:

    planner = PlannerLayer()
    planner.initialize(config, event_bus)
    planner.register_capability("retrieve", vault_search, description="kasa arama")
    planner.register_capability("synthesize", neural_answer)

    rapor = planner.run("GPU sıcaklığım neden yüksek?")
    print(rapor["answer"])
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from neurocore.planner.checkpoint import Checkpoint, CheckpointManager
from neurocore.planner.decomposer import MAX_TOTAL_TASKS, Decomposer
from neurocore.planner.executor import (
    CapabilityError,
    CapabilityRegistry,
    ExecutionAborted,
    ExecutionReport,
    Executor,
)
from neurocore.planner.task import (
    PlanError,
    ReasoningStep,
    ReasoningTrace,
    Task,
    TaskGraph,
    TaskKind,
    TaskState,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("planner.layer")


# ─── Planlayıcı Katmanı ─────────────────────────────────────────────────────

class PlannerLayer:
    """
    Planlayıcı katmanının dış yüzü (ARCHITECTURE §3 katman sözleşmesi).

    Kullanım:
        layer = PlannerLayer()
        layer.initialize(config, event_bus)

        # Yalnızca plan (yürütme yok)
        graf = layer.plan("Obsidian vault yapısını nasıl kurarım?")
        print(graf.to_markdown())

        # Plan + yürütme
        sonuc = layer.run("sistem raporunu göster")
        print(sonuc["answer"])

        # Açıklama modu
        print(layer.explain("3+5 hesapla")["mermaid"])
    """

    name = "planner"

    def __init__(self) -> None:
        self._config: Any = None
        self._event_bus: Any = None
        self._initialized: bool = False
        self._offline: bool = False
        self._max_depth_override: Optional[int] = None

        self._decomposer: Optional[Decomposer] = None
        self._executor: Optional[Executor] = None
        self._checkpoints: Optional[CheckpointManager] = None

        self._last_report: Optional[ExecutionReport] = None
        self._plans: int = 0
        self._runs: int = 0
        self._started_at: float = 0.0

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None,
                   capabilities: Any = None) -> None:
        """
        Katmanı başlatır (idempotent).

        Args:
            config: ``SystemConfig`` benzeri nesne (None olabilir)
            event_bus: Olay yolu
            capabilities: ``CapabilityRegistry`` ya da ``{ad: çağrılabilir}``
        """
        if self._initialized:
            if config is not None:
                self._config = config
            if event_bus is not None:
                self._event_bus = event_bus
            if capabilities is not None:
                self._bind_capabilities(capabilities)
            return

        self._config = config
        self._event_bus = event_bus

        self._decomposer = Decomposer()
        self._decomposer.initialize(config, event_bus)

        self._executor = Executor()
        self._executor.initialize(config, event_bus)
        self._executor.offline = self._offline

        self._checkpoints = CheckpointManager()
        self._checkpoints.initialize(config, event_bus)

        if capabilities is not None:
            self._bind_capabilities(capabilities)

        self._initialized = True
        self._started_at = time.time()

        logger.info(
            f"Planlayıcı katmanı başlatıldı — {len(self._executor.capabilities)} yetenek, "
            f"checkpoint dizini: {self._checkpoints.directory}, "
            f"çevrimdışı={'evet' if self._offline else 'hayır'}"
        )
        self._emit("layer_ready", {
            "capabilities": len(self._executor.capabilities),
            "checkpoint_dir": str(self._checkpoints.directory or ""),
            "max_task_depth": self._config_max_depth(),
            "offline": self._offline,
        })

    def shutdown(self) -> None:
        """Katmanı kapatır (idempotent)."""
        if not self._initialized:
            return
        for component in (self._executor, self._decomposer, self._checkpoints):
            shutdown = getattr(component, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:  # pragma: no cover - savunmacı
                    logger.warning(f"⚠ Alt bileşen kapatılamadı: {exc}")
        self._initialized = False
        logger.info(
            f"Planlayıcı katmanı kapatıldı — {self._plans} plan, {self._runs} yürütme"
        )

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Olay yayınlar (event_bus yoksa sessizce atlar)."""
        if self._event_bus:
            try:
                self._event_bus.emit(f"planner.{event}", data, source=self.name)
            except Exception as exc:  # pragma: no cover - savunmacı
                logger.debug(f"Olay yayınlanamadı ({event}): {exc}")

    def _bind_capabilities(self, capabilities: Any) -> None:
        """Yetenekleri kayıt defterine bağlar (registry ya da sözlük)."""
        if self._executor is None:
            return
        if isinstance(capabilities, CapabilityRegistry):
            self._executor._capabilities = capabilities  # noqa: SLF001 - bilinçli devir
            logger.info(f"Yetenek kayıt defteri devralındı ({len(capabilities)} yetenek)")
            return
        if isinstance(capabilities, dict):
            for name, value in capabilities.items():
                if callable(value):
                    self._executor.register_capability(str(name), value)
                elif isinstance(value, (tuple, list)) and value and callable(value[0]):
                    kwargs = value[1] if len(value) > 1 and isinstance(value[1], dict) else {}
                    self._executor.register_capability(str(name), value[0], **kwargs)
            logger.info(f"{len(capabilities)} yetenek sözlükten yüklendi")
            return
        logger.warning(f"⚠ Tanınmayan capabilities türü: {type(capabilities).__name__}")

    def _ensure(self) -> None:
        """Katman başlatılmadıysa başlatır (tembel güvence)."""
        if not self._initialized:
            self.initialize(self._config, self._event_bus)

    # ─── Özellikler ─────────────────────────────────────────────────────

    @property
    def decomposer(self) -> Decomposer:
        """Kural tabanlı ayrıştırıcı."""
        self._ensure()
        assert self._decomposer is not None
        return self._decomposer

    @property
    def executor(self) -> Executor:
        """Görev yürütücü."""
        self._ensure()
        assert self._executor is not None
        return self._executor

    @property
    def checkpoints(self) -> CheckpointManager:
        """Checkpoint yöneticisi."""
        self._ensure()
        assert self._checkpoints is not None
        return self._checkpoints

    @property
    def offline(self) -> bool:
        """Çevrimdışı kip bayrağı."""
        return self._offline

    @offline.setter
    def offline(self, value: bool) -> None:
        """Çevrimdışı kip: ``net.*`` ve ``research`` yetenekleri engellenir."""
        self._offline = bool(value)
        if self._executor is not None:
            self._executor.offline = self._offline
        logger.info(f"Çevrimdışı kip: {'açık' if self._offline else 'kapalı'}")

    @property
    def initialized(self) -> bool:
        """Katman başlatıldı mı?"""
        return self._initialized

    # ─── Planlama ───────────────────────────────────────────────────────

    def plan(self, goal: str, context: Optional[Dict[str, Any]] = None,
             max_depth_override: Optional[int] = None) -> TaskGraph:
        """
        Hedefi görev grafiğine çevirir (**yürütmez**).

        Args:
            goal: Doğal dil hedefi
            context: Ek bağlam (``budget_seconds``, ``path`` gibi)
            max_depth_override: Özyineleme derinliği üst sınırı (None → konfig)

        Returns:
            ``TaskGraph``
        """
        self._ensure()
        depth = self._resolve_depth(max_depth_override)

        started = time.perf_counter()
        graph = self.decomposer.decompose(goal, context=context, max_depth=depth)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._plans += 1

        logger.info(
            f"Plan hazır: '{str(goal)[:70]}' → {len(graph)} görev, "
            f"{len(graph.parallel_batches())} dalga, {elapsed_ms:.1f}ms"
        )
        self._emit("plan_ready", {
            "goal": str(goal)[:200],
            "tasks": len(graph),
            "waves": len(graph.parallel_batches()),
            "max_depth": depth,
            "duration_ms": round(elapsed_ms, 2),
        })
        return graph

    def run(self, goal: str, context: Optional[Dict[str, Any]] = None,
            budget_seconds: Optional[float] = None,
            max_depth_override: Optional[int] = None,
            offline: Optional[bool] = None) -> Dict[str, Any]:
        """
        Hedefi planlar ve **senkron** yürütür.

        Args:
            goal: Doğal dil hedefi
            context: Paylaşılan bağlam
            budget_seconds: Duvarsaati bütçesi (None = sınırsız)
            max_depth_override: Ayrıştırma derinlik sınırı
            offline: True ise ağ gerektiren yetenekler engellenir

        Returns:
            ``ExecutionReport.to_dict()``
        """
        self._ensure()
        if offline is not None:
            self.offline = offline
        report = self._execute(goal, context, budget_seconds, max_depth_override)
        return report.to_dict()

    async def arun(self, goal: str, context: Optional[Dict[str, Any]] = None,
                   budget_seconds: Optional[float] = None,
                   max_depth_override: Optional[int] = None,
                   offline: Optional[bool] = None) -> Dict[str, Any]:
        """
        Hedefi planlar ve **asenkron** yürütür (paralel dalgalar desteklenir).

        Returns:
            ``ExecutionReport.to_dict()``
        """
        self._ensure()
        if offline is not None:
            self.offline = offline
        report = await self._execute_async(goal, context, budget_seconds, max_depth_override)
        return report.to_dict()

    def _execute(self, goal: str, context: Optional[Dict[str, Any]],
                 budget_seconds: Optional[float],
                 max_depth_override: Optional[int]) -> ExecutionReport:
        """İç yürütme akışı (senkron): plan + execute + istatistik."""
        started = time.perf_counter()
        graph = self.plan(goal, context=context, max_depth_override=max_depth_override)
        report = self.executor.execute(
            graph,
            trace=None,
            context=dict(context or {}),
            checkpoint=self.checkpoints,
            budget_seconds=budget_seconds,
        )
        report.context.setdefault("goal", goal)
        self._last_report = report
        self._runs += 1

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            f"Yürütme tamam: '{str(goal)[:60]}' → {report.state} "
            f"({report.completed}/{len(graph)} görev, {elapsed_ms:.0f}ms)"
        )
        return report

    async def _execute_async(self, goal: str, context: Optional[Dict[str, Any]],
                             budget_seconds: Optional[float],
                             max_depth_override: Optional[int]) -> ExecutionReport:
        """İç yürütme akışı (asenkron): plan + execute_async + istatistik."""
        started = time.perf_counter()
        graph = self.plan(goal, context=context, max_depth_override=max_depth_override)
        report = await self.executor.execute_async(
            graph,
            trace=None,
            context=dict(context or {}),
            checkpoint=self.checkpoints,
            budget_seconds=budget_seconds,
        )
        report.context.setdefault("goal", goal)
        self._last_report = report
        self._runs += 1

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            f"Async yürütme tamam: '{str(goal)[:60]}' → {report.state} "
            f"({report.completed}/{len(graph)} görev, {elapsed_ms:.0f}ms)"
        )
        return report

    def step(self, goal: str, context: Optional[Dict[str, Any]] = None,
             budget_seconds: Optional[float] = None) -> ExecutionReport:
        """
        ``run`` ile aynı işi yapar ama tam ``ExecutionReport`` nesnesi döndürür.

        (Grafik/iz üzerinde çalışmak isteyen çağıranlar için.)
        """
        self._ensure()
        return self._execute(goal, context, budget_seconds, None)

    # ─── Devam (Resume) ─────────────────────────────────────────────────

    def resume(self, budget_seconds: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        En yeni checkpoint'ten devam eder ve planı sürdürür.

        Returns:
            ``ExecutionReport.to_dict()`` ya da devam edilecek checkpoint yoksa None.
        """
        self._ensure()
        resumed = self.checkpoints.resume_latest()
        if resumed is None:
            logger.info("Devam edilecek checkpoint bulunamadı")
            return None

        graph, trace, context = resumed
        report = self.executor.execute(
            graph,
            trace=trace,
            context=context,
            checkpoint=self.checkpoints,
            budget_seconds=budget_seconds,
        )
        if not report.context.get("goal"):
            report.context["goal"] = graph.goal
        self._last_report = report
        self._runs += 1

        logger.info(
            f"Checkpoint'ten devam edildi: '{str(graph.goal)[:60]}' → {report.state}"
        )
        return report.to_dict()

    # ─── Yetenekler ─────────────────────────────────────────────────────

    def register_capability(self, name: str, fn: Callable[..., Any], **kwargs: Any) -> None:
        """
        Yetenek kaydeder.

        Args:
            name: Yetenek adı (görevlerin ``tool`` alanı bu adı kullanır)
            fn: Çağrılabilir (senkron ya da asenkron)
            **kwargs: ``description``, ``is_async``, ``timeout``, ``cost``
        """
        self._ensure()
        self.executor.register_capability(name, fn, **kwargs)
        self._emit("capability_registered", {
            "capability": name,
            "total": len(self.executor.capabilities),
        })

    def unregister_capability(self, name: str) -> bool:
        """Yeteneği kaldırır."""
        self._ensure()
        removed = self.executor.capabilities.unregister(name)
        if removed:
            self._emit("capability_unregistered", {"capability": name})
        return removed

    def capabilities(self) -> List[Dict[str, Any]]:
        """Kayıtlı yeteneklerin listesi."""
        self._ensure()
        return self.executor.capabilities.list()

    def missing_capabilities(self, graph: Optional[TaskGraph] = None) -> List[str]:
        """
        Planda gerekli olup kayıtlı olmayan yetenekleri döndürür.

        Args:
            graph: Denetlenecek graf (None ise tüm yerleşik şablon yetenekleri)
        """
        self._ensure()
        registry = self.executor.capabilities
        if graph is None:
            names = {
                "retrieve", "reason", "synthesize", "verify", "memorize",
                "compute", "research", "os.system_report", "os.read", "os.list", "os.write",
            }
        else:
            names = {t.tool for t in graph.tasks.values() if t.tool}
        return sorted(n for n in names if not registry.has(n))

    # ─── Açıklama Modu ──────────────────────────────────────────────────

    def explain(self, goal: str, context: Optional[Dict[str, Any]] = None,
                max_depth_override: Optional[int] = None) -> Dict[str, Any]:
        """
        ``./plan only`` — sınıflandırma + plan görselleri; **yürütme yapmaz**.

        Returns:
            ``{"goal", "classification", "markdown", "mermaid", "stats",
            "batches", "critical_path", "missing_capabilities", ...}``
        """
        self._ensure()
        classification = self.decomposer.classify(goal)
        graph = self.plan(goal, context=context, max_depth_override=max_depth_override)
        stats = graph.stats()

        payload = {
            "goal": goal,
            "classification": classification,
            "markdown": graph.to_markdown(),
            "mermaid": graph.to_mermaid(),
            "stats": stats,
            "batches": graph.parallel_batches(),
            "critical_path": graph.critical_path(),
            "problems": graph.validate(max_depth=self._resolve_depth(max_depth_override)),
            "missing_capabilities": self.missing_capabilities(graph),
            "capabilities": [c["name"] for c in self.capabilities()],
            "offline": self._offline,
            "enable_parallel_tasks": self.executor.get_stats()["enable_parallel_tasks"],
            "max_depth": self._resolve_depth(max_depth_override),
            "task_budget": MAX_TOTAL_TASKS,
        }
        logger.info(
            f"Plan açıklaması: '{str(goal)[:60]}' → niyet={classification.get('intent')}, "
            f"karmaşıklık={classification.get('complexity')}, {len(graph)} görev"
        )
        self._emit("plan_explained", {
            "goal": str(goal)[:200],
            "intent": classification.get("intent"),
            "complexity": classification.get("complexity"),
            "tasks": len(graph),
        })
        return payload

    # ─── Sağlık ve İstatistik ───────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """
        Katman sağlık raporu.

        Returns:
            ``{"ok": bool, "capabilities": n, "checkpoints": n, "issues": [...]}``
        """
        self._ensure()
        issues: List[str] = []

        cap_stats = self.executor.capabilities.get_stats()
        checkpoint_stats = self.checkpoints.get_stats()
        exec_stats = self.executor.get_stats()

        if not self._initialized:
            issues.append("katman başlatılmadı")
        if cap_stats["capabilities"] == 0:
            issues.append("hiç yetenek kayıtlı değil — planlar sentetik sonuç üretir")
        if self._last_report is not None and self._last_report.state in ("failed", "aborted"):
            issues.append(f"son plan '{self._last_report.state}': {self._last_report.error}")
        if checkpoint_stats["directory"] == "":
            issues.append("checkpoint dizini tanımsız")
        if checkpoint_stats["corrupt"]:
            issues.append(f"{checkpoint_stats['corrupt']} bozuk checkpoint karantinaya alındı")
        if self._offline:
            issues.append("çevrimdışı kip açık — net.* ve research yetenekleri engelli")
        if self._last_report is not None and self._last_report.replans:
            issues.append(f"son planda {self._last_report.replans} yeniden planlama yapıldı")

        return {
            "ok": len(issues) == 0,
            "layer": self.name,
            "initialized": self._initialized,
            "capabilities": cap_stats["capabilities"],
            "checkpoints": checkpoint_stats["files"],
            "issues": issues,
            "offline": self._offline,
            "last_state": exec_stats["last_state"],
            "capability_names": cap_stats["names"][:20],
            "checkpoint_dir": checkpoint_stats["directory"],
            "missing_capabilities": self.missing_capabilities(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir katman metrikleri."""
        if not self._initialized:
            return {
                "name": self.name,
                "initialized": False,
                "plans": self._plans,
                "runs": self._runs,
                "offline": self._offline,
            }

        last = None
        if self._last_report is not None:
            last = {
                "goal": self._last_report.goal[:200],
                "state": self._last_report.state,
                "answered": self._last_report.answered,
                "duration_ms": round(self._last_report.duration_ms, 2),
                "completed": self._last_report.completed,
                "failed": self._last_report.failed,
                "replans": self._last_report.replans,
            }

        return {
            "name": self.name,
            "initialized": self._initialized,
            "offline": self._offline,
            "plans": self._plans,
            "runs": self._runs,
            "max_depth": self._config_max_depth(),
            "uptime_s": round(time.time() - self._started_at, 2) if self._started_at else 0.0,
            "decomposer": self._decomposer.get_stats() if self._decomposer else {},
            "executor": self._executor.get_stats() if self._executor else {},
            "checkpoint": self._checkpoints.get_stats() if self._checkpoints else {},
            "capabilities": self.executor.capabilities.get_stats()["capabilities"],
            "last_report": last,
        }

    # ─── Dahili Yardımcılar ─────────────────────────────────────────────

    def _config_max_depth(self) -> int:
        """Konfigürasyondaki en fazla görev derinliği."""
        planner = getattr(self._config, "planner", None) if self._config else None
        try:
            return max(1, int(getattr(planner, "max_task_depth", 10) or 10))
        except (TypeError, ValueError):  # pragma: no cover - savunmacı
            return 10

    def _resolve_depth(self, override: Optional[int]) -> int:
        """Geçerli derinlik sınırını çözer (anlık override > kalıcı override > konfig)."""
        candidate = override if override is not None else self._max_depth_override
        if candidate is None:
            return self._config_max_depth()
        try:
            return max(1, int(candidate))
        except (TypeError, ValueError):  # pragma: no cover - savunmacı
            return self._config_max_depth()

    def set_max_depth(self, value: Optional[int]) -> None:
        """Kalıcı derinlik sınırını ayarlar (None → konfigürasyon değeri)."""
        self._max_depth_override = None if value is None else max(1, int(value))
        logger.info(f"Azami görev derinliği: {self._resolve_depth(None)}")

    # ─── Grafik Yardımcıları ────────────────────────────────────────────

    def visualize(self, goal: str, fmt: str = "markdown",
                  context: Optional[Dict[str, Any]] = None) -> str:
        """
        Planı metin biçiminde görselleştirir (yürütme yok).

        Args:
            goal: Hedef
            fmt: ``markdown`` | ``mermaid``
            context: Ek bağlam
        """
        graph = self.plan(goal, context=context)
        if fmt == "mermaid":
            return graph.to_mermaid()
        return graph.to_markdown()

    def restore(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Belirli bir checkpoint'in içeriğini döndürür (yürütmez).

        Returns:
            ``{"checkpoint_id", "goal", "markdown", "mermaid", "progress", "tasks"}``
        """
        self._ensure()
        checkpoint: Optional[Checkpoint] = self.checkpoints.load(checkpoint_id)
        if checkpoint is None:
            return None
        graph, trace, context = self.checkpoints.restore(checkpoint_id)
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "goal": checkpoint.goal,
            "note": checkpoint.note,
            "markdown": graph.to_markdown(),
            "mermaid": graph.to_mermaid(),
            "progress": graph.progress(),
            "tasks": len(graph.tasks),
            "trace_steps": len(trace.steps) if trace else 0,
            "context_keys": sorted(context.keys())[:40],
        }


__all__ = ["PlannerLayer", "ExecutionReport", "CapabilityRegistry", "CapabilityError"]
