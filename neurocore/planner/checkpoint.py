"""
NeuroCore Checkpoint Yöneticisi
================================
Plan durumunu diske yazar ve gerekirse geri yükler. Çökme sonrası
``resume_latest()`` ile kesintisiz devam sağlar (ARCHITECTURE §13).

Dosya biçimi::

    <dizin>/<iso_zaman>_<kısa_id>.json      UTF-8, ensure_ascii=False, indent 2

Dayanıklılık kuralları:
    * Yazma **atomiktir**: önce ``.tmp`` dosyasına yazılır, sonra ``os.replace``.
    * Bozuk dosyalar silinmez; ``<dizin>/_corrupt/`` altına karantinaya alınır.
    * ``load``/``list``/``restore`` hiçbir zaman istisna fırlatmaz.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from neurocore.planner.task import (
    ReasoningTrace,
    TaskGraph,
    TaskState,
    json_safe,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("planner.checkpoint")


# ─── Checkpoint Veri Yapısı ─────────────────────────────────────────────────

@dataclass
class Checkpoint:
    """
    Diske yazılmış tek bir plan anlık görüntüsü.

    Attributes:
        checkpoint_id: Benzersiz kimlik (``<zaman>_<kısa_id>``)
        goal: Plan hedefi
        created_at: Oluşturulma zamanı (epoch)
        graph: ``TaskGraph.to_dict()`` çıktısı
        trace: ``ReasoningTrace.to_dict()`` çıktısı (opsiyonel)
        context: Paylaşılan bağlam sözlüğü
        note: İnsan-okur kısa not ("5 görev tamamlandı" gibi)
    """

    checkpoint_id: str
    goal: str
    created_at: float
    graph: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    note: str = ""
    version: int = 1
    task_count: int = 0
    progress: float = 0.0
    state_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Checkpoint'i JSON-serileştirilebilir sözlüğe çevirir."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "goal": self.goal,
            "created_at": self.created_at,
            "created_iso": datetime.fromtimestamp(self.created_at).isoformat(timespec="seconds"),
            "graph": self.graph,
            "trace": self.trace,
            "context": self.context,
            "note": self.note,
            "version": self.version,
            "task_count": self.task_count,
            "progress": self.progress,
            "state_counts": self.state_counts,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        """Sözlükten checkpoint kurar (eksik alanlara toleranslı)."""
        data = dict(d or {})
        return cls(
            checkpoint_id=str(data.get("checkpoint_id", "")),
            goal=str(data.get("goal", "")),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            graph=data.get("graph") or {},
            trace=data.get("trace"),
            context=data.get("context") or {},
            note=str(data.get("note", "")),
            version=int(data.get("version", 1) or 1),
            task_count=int(data.get("task_count", 0) or 0),
            progress=float(data.get("progress", 0.0) or 0.0),
            state_counts=data.get("state_counts") or {},
        )

    def summary(self) -> str:
        """Tek satırlık Türkçe özet."""
        created = datetime.fromtimestamp(self.created_at).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"{self.checkpoint_id} · {created} · {self.task_count} görev · "
            f"%{self.progress * 100:.0f} · {self.note or 'not yok'}"
        )


# ─── Checkpoint Yöneticisi ──────────────────────────────────────────────────

class CheckpointManager:
    """
    Plan checkpoint'lerini yönetir.

    Kullanım:
        cm = CheckpointManager()
        cm.initialize(config)
        cid = cm.save(graph, trace, context={"goal": "..."})
        graph, trace, context = cm.restore(cid)
    """

    name = "planner.checkpoint"

    #: Saklanacak en fazla checkpoint sayısı.
    MAX_CHECKPOINTS: int = 50

    def __init__(self) -> None:
        self._config: Any = None
        self._event_bus: Any = None
        self._dir: Optional[Path] = None
        self._initialized: bool = False
        self._max_checkpoints: int = self.MAX_CHECKPOINTS
        self._counter: int = 0
        self._stats: Dict[str, Any] = {
            "saved": 0,
            "loaded": 0,
            "restored": 0,
            "deleted": 0,
            "pruned": 0,
            "resumed": 0,
            "corrupt": 0,
            "errors": 0,
            "last_checkpoint_id": "",
        }
        self._started_at: float = 0.0

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Yöneticiyi yapılandırır ve dizini hazırlar (idempotent).

        Dizin çözümü: ``config.data_path`` varsa ``<data_path>/checkpoints/planner``,
        yoksa ``<base_path veya çalışma dizini>/data/checkpoints/planner``.
        """
        if self._initialized:
            if config is not None:
                self._config = config
                self._dir = self._resolve_dir()
                self._ensure_dir()
            if event_bus is not None:
                self._event_bus = event_bus
            return

        self._config = config
        self._event_bus = event_bus
        self._dir = self._resolve_dir()
        self._ensure_dir()
        self._initialized = True
        self._started_at = time.time()

        logger.info(
            f"Checkpoint yöneticisi hazır — dizin: {self._dir} "
            f"(en fazla {self._max_checkpoints} checkpoint)"
        )

    def shutdown(self) -> None:
        """Yöneticiyi kapatır (idempotent)."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info(
            f"Checkpoint yöneticisi kapatıldı — {self._stats['saved']} kayıt, "
            f"{len(self._file_list())} dosya"
        )

    def _resolve_dir(self) -> Path:
        """Checkpoint dizinini konfigürasyondan çözer."""
        data_path = getattr(self._config, "data_path", None) if self._config else None
        base_path = getattr(self._config, "base_path", None) if self._config else None

        if data_path:
            root = Path(str(data_path))
            return root / "checkpoints" / "planner"

        if base_path:
            return Path(str(base_path)) / "data" / "checkpoints" / "planner"

        return Path(os.getcwd()) / "data" / "checkpoints" / "planner"

    def _ensure_dir(self) -> None:
        """Dizini (ve karantina alt dizinini) oluşturur."""
        if self._dir is None:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error(f"Checkpoint dizini oluşturulamadı ({self._dir}): {exc}")

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Olay yayınlar (event_bus yoksa sessizce atlar)."""
        if self._event_bus:
            try:
                self._event_bus.emit(f"planner.{event}", data, source=self.name)
            except Exception as exc:  # pragma: no cover - savunmacı
                logger.debug(f"Olay yayınlanamadı ({event}): {exc}")

    @property
    def directory(self) -> Optional[Path]:
        """Checkpoint dizini (initialize edilmediyse None)."""
        return self._dir

    # ─── Kaydetme ───────────────────────────────────────────────────────

    def save(self, graph: TaskGraph, trace: Optional[ReasoningTrace] = None,
             context: Optional[Dict[str, Any]] = None, note: str = "") -> str:
        """
        Plan durumunu diske **atomik** olarak yazar.

        Args:
            graph: Görev grafiği
            trace: Akıl yürütme zinciri (opsiyonel)
            context: Paylaşılan bağlam (opsiyonel)
            note: Kısa açıklama

        Returns:
            Checkpoint kimliği (``"<iso_zaman>_<kısa_id>"``). Yazma başarısız
            olursa yine bir kimlik döner ama dosya oluşmamış olabilir.
        """
        if not self._initialized:
            self.initialize()

        now = time.time()
        self._counter += 1
        short_id = f"{uuid4().hex[:6]}{self._counter % 1000:03d}"
        stamp = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
        checkpoint_id = f"{stamp.replace(':', '-').replace('.', '-')}_{short_id}"

        stats = graph.stats() if isinstance(graph, TaskGraph) else {}
        payload = Checkpoint(
            checkpoint_id=checkpoint_id,
            goal=graph.goal if isinstance(graph, TaskGraph) else "",
            created_at=now,
            graph=graph.to_dict() if isinstance(graph, TaskGraph) else {},
            trace=trace.to_dict() if trace is not None else None,
            context=json_safe(context or {}),
            note=note,
            task_count=int(stats.get("total_tasks", 0)),
            progress=float(stats.get("progress", 0.0)),
            state_counts=stats.get("by_state", {}),
        ).to_dict()

        if self._dir is None:
            self._ensure_dir()

        if self._dir is None:
            logger.error("Checkpoint kaydedilemedi: dizin yok")
            return checkpoint_id

        target = self._dir / f"{checkpoint_id}.json"
        temp = self._dir / f"{checkpoint_id}.json.tmp"
        try:
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
        except Exception as exc:
            self._stats["errors"] += 1
            logger.error(f"Checkpoint yazılamadı ({target.name}): {exc}")
            try:
                if temp.exists():
                    temp.unlink()
            except Exception:  # pragma: no cover - savunmacı
                pass
            return checkpoint_id

        self._stats["saved"] += 1
        self._stats["last_checkpoint_id"] = checkpoint_id
        self._emit("checkpoint_saved", {
            "checkpoint_id": checkpoint_id,
            "goal": str(payload.get("goal", ""))[:160],
            "tasks": payload["task_count"],
            "progress": payload["progress"],
            "note": note,
        })
        logger.debug(
            f"Checkpoint kaydedildi: {checkpoint_id} ({payload['task_count']} görev, "
            f"%{payload['progress'] * 100:.0f})"
        )
        self.prune(self._max_checkpoints)
        return checkpoint_id

    # ─── Yükleme ────────────────────────────────────────────────────────

    def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Checkpoint'i diskten okur.

        Bozuk dosya karantinaya alınır (``_corrupt/``) ve ``None`` döner —
        bu metot asla istisna fırlatmaz.
        """
        if not self._initialized:
            self.initialize()
        if self._dir is None or not checkpoint_id:
            return None

        path = self._dir / f"{checkpoint_id}.json"
        if not path.exists():
            logger.warning(f"Checkpoint bulunamadı: {checkpoint_id}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("kök nesne sözlük değil")
            checkpoint = Checkpoint.from_dict(data)
            if not checkpoint.checkpoint_id:
                checkpoint.checkpoint_id = checkpoint_id
            self._stats["loaded"] += 1
            return checkpoint
        except Exception as exc:
            self._quarantine(path, exc)
            return None

    def latest(self) -> Optional[Checkpoint]:
        """En yeni checkpoint'i döndürür (yoksa None)."""
        files = self._file_list()
        if not files:
            return None
        newest = max(files, key=lambda p: p[1])
        checkpoint = self.load(newest[0])
        if checkpoint is None:
            # En yeni dosya bozuksa bir öncekini dene
            remaining = [f for f in files if f[0] != newest[0]]
            if remaining:
                return self.load(max(remaining, key=lambda p: p[1])[0])
        return checkpoint

    def list(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Checkpoint'lerin **hafif** listesi (en yeniden en eskiye).

        Yalnızca üst bilgi alanları döner; graf içeriği okunmaz.
        """
        if not self._initialized:
            self.initialize()

        entries: List[Dict[str, Any]] = []
        for checkpoint_id, mtime, path in self._file_list():
            meta: Dict[str, Any] = {
                "checkpoint_id": checkpoint_id,
                "file": path.name,
                "size_bytes": 0,
                "mtime": mtime,
                "created_iso": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
                "goal": "",
                "task_count": 0,
                "progress": 0.0,
                "note": "",
                "valid": True,
            }
            try:
                meta["size_bytes"] = path.stat().st_size
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                meta["goal"] = str(data.get("goal", ""))[:200]
                meta["task_count"] = int(data.get("task_count", 0) or 0)
                meta["progress"] = float(data.get("progress", 0.0) or 0.0)
                meta["note"] = str(data.get("note", ""))
                meta["created_iso"] = str(
                    data.get("created_iso", meta["created_iso"])
                )
            except Exception as exc:
                meta["valid"] = False
                meta["error"] = f"{type(exc).__name__}: {exc}"
            entries.append(meta)

        return entries[: max(0, int(limit))] if limit else entries

    def restore(self, checkpoint_id: str) -> Tuple[TaskGraph, Optional[ReasoningTrace], Dict[str, Any]]:
        """
        Checkpoint'i canlı nesnelere dönüştürür.

        Returns:
            ``(TaskGraph, ReasoningTrace | None, context)`` — checkpoint yoksa
            boş bir graf ve boş bağlam döner (istisna fırlatmaz).
        """
        checkpoint = self.load(checkpoint_id)
        if checkpoint is None:
            return TaskGraph(goal=""), None, {}

        try:
            graph = TaskGraph.from_dict(checkpoint.graph or {})
            trace: Optional[ReasoningTrace] = None
            if checkpoint.trace:
                trace = ReasoningTrace.from_dict(checkpoint.trace)
            context: Dict[str, Any] = dict(checkpoint.context or {})
            self._stats["restored"] += 1
            self._emit("checkpoint_restored", {
                "checkpoint_id": checkpoint_id,
                "goal": checkpoint.goal[:160],
                "tasks": len(graph.tasks),
            })
            logger.info(
                f"Checkpoint geri yüklendi: {checkpoint_id} "
                f"({len(graph.tasks)} görev, ilerleme %{graph.progress() * 100:.0f})"
            )
            return graph, trace, context
        except Exception as exc:
            self._stats["errors"] += 1
            logger.error(f"Checkpoint geri yüklenemedi ({checkpoint_id}): {exc}")
            return TaskGraph(goal=""), None, {}

    def resume_latest(self) -> Optional[Tuple[TaskGraph, Optional[ReasoningTrace], Dict[str, Any]]]:
        """
        En yeni checkpoint'ten devam eder.

        ``RUNNING`` durumunda kalmış görevler (çökme anı) ``PENDING``'e
        döndürülür; böylece plan baştan güvenle sürdürülebilir.

        Returns:
            ``(graph, trace, context)`` ya da checkpoint yoksa ``None``.
        """
        checkpoint = self.latest()
        if checkpoint is None:
            logger.info("Devam edilecek checkpoint yok")
            return None

        graph, trace, context = self.restore(checkpoint.checkpoint_id)
        if not graph.tasks:
            logger.warning(f"⚠ Checkpoint boş graf içeriyor: {checkpoint.checkpoint_id}")
            return None

        reset = 0
        for task in graph.tasks.values():
            if task.state in (TaskState.RUNNING, TaskState.READY):
                task.state = TaskState.PENDING
                task.started_at = 0.0
                task.attempts = 0
                reset += 1

        self._stats["resumed"] += 1
        self._emit("checkpoint_resumed", {
            "checkpoint_id": checkpoint.checkpoint_id,
            "goal": checkpoint.goal[:160],
            "tasks": len(graph.tasks),
            "reset_running": reset,
        })
        logger.info(
            f"Checkpoint'ten devam ediliyor: {checkpoint.checkpoint_id} — "
            f"{len(graph.tasks)} görev, {reset} yarıda kalan görev sıfırlandı"
        )
        return graph, trace, context

    # ─── Silme / Budama ─────────────────────────────────────────────────

    def delete(self, checkpoint_id: str) -> bool:
        """Checkpoint dosyasını siler."""
        if not self._initialized:
            self.initialize()
        if self._dir is None or not checkpoint_id:
            return False

        path = self._dir / f"{checkpoint_id}.json"
        if not path.exists():
            return False
        try:
            path.unlink()
            self._stats["deleted"] += 1
            logger.debug(f"Checkpoint silindi: {checkpoint_id}")
            return True
        except Exception as exc:
            self._stats["errors"] += 1
            logger.error(f"Checkpoint silinemedi ({checkpoint_id}): {exc}")
            return False

    def prune(self, keep: int = 50) -> int:
        """
        En yeni ``keep`` checkpoint'i tutar, kalanını siler.

        Returns:
            Silinen dosya sayısı.
        """
        if not self._initialized:
            self.initialize()
        if self._dir is None:
            return 0

        files = self._file_list()
        if len(files) <= keep:
            return 0

        removed = 0
        for checkpoint_id, _mtime, path in files[keep:]:
            try:
                path.unlink()
                removed += 1
            except Exception as exc:  # pragma: no cover - savunmacı
                logger.warning(f"⚠ Checkpoint budanamadı ({checkpoint_id}): {exc}")

        if removed:
            self._stats["pruned"] += removed
            logger.debug(f"Checkpoint budama: {removed} dosya silindi (tutulan: {keep})")
        return removed

    def clear(self) -> int:
        """Tüm checkpoint'leri siler (karantina hariç)."""
        if not self._initialized:
            self.initialize()
        if self._dir is None:
            return 0
        files = self._file_list()
        removed = 0
        for checkpoint_id, _mtime, path in files:
            try:
                path.unlink()
                removed += 1
            except Exception:  # pragma: no cover - savunmacı
                pass
        logger.info(f"Tüm checkpoint'ler silindi ({removed} dosya)")
        return removed

    # ─── Dahili Yardımcılar ─────────────────────────────────────────────

    def _file_list(self) -> List[Tuple[str, float, Path]]:
        """
        Checkpoint dosyalarını **en yeniden en eskiye** listeler.

        Returns:
            ``[(checkpoint_id, mtime, path), …]``
        """
        return self._file_list_ascending()[::-1]

    def _file_list_ascending(self) -> List[Tuple[str, float, Path]]:
        """Checkpoint dosyalarını eskiden yeniye listeler."""
        if self._dir is None or not self._dir.exists():
            return []

        items: List[Tuple[str, float, Path]] = []
        try:
            for path in self._dir.glob("*.json"):
                if not path.is_file():
                    continue
                try:
                    mtime = path.stat().st_mtime
                except OSError:  # pragma: no cover - savunmacı
                    mtime = 0.0
                items.append((path.stem, mtime, path))
        except Exception as exc:  # pragma: no cover - savunmacı
            logger.warning(f"⚠ Checkpoint dizini okunamadı: {exc}")
            return []

        items.sort(key=lambda item: (item[1], item[0]))
        return items

    def _quarantine(self, path: Path, error: Exception) -> None:
        """Bozuk checkpoint dosyasını ``_corrupt/`` altına taşır."""
        self._stats["corrupt"] += 1
        target_dir = (self._dir or path.parent) / "_corrupt"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / path.name
            if target.exists():
                target = target_dir / f"{path.stem}_{uuid4().hex[:4]}.json"
            shutil.move(str(path), str(target))
        except Exception as exc:  # pragma: no cover - savunmacı
            logger.error(f"Bozuk checkpoint karantinaya alınamadı ({path.name}): {exc}")
            return
        logger.warning(
            f"⚠ Bozuk checkpoint karantinaya alındı: {path.name} "
            f"({type(error).__name__}: {error}) → {target_dir.name}/"
        )
        self._emit("checkpoint_corrupt", {
            "checkpoint_id": path.stem,
            "error": f"{type(error).__name__}: {error}",
            "quarantined_to": target_dir.name,
        })

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir checkpoint metrikleri."""
        files = self._file_list()
        total_bytes = 0
        for _cid, _mtime, path in files:
            try:
                total_bytes += path.stat().st_size
            except OSError:  # pragma: no cover - savunmacı
                continue

        newest = files[0] if files else None
        corrupt_dir = (self._dir / "_corrupt") if self._dir else None
        quarantined = 0
        if corrupt_dir and corrupt_dir.exists():
            try:
                quarantined = sum(1 for _ in corrupt_dir.glob("*.json"))
            except Exception:  # pragma: no cover - savunmacı
                quarantined = 0

        return {
            "name": self.name,
            "initialized": self._initialized,
            "directory": str(self._dir) if self._dir else "",
            "files": len(files),
            "max_checkpoints": self._max_checkpoints,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 3),
            "quarantined": quarantined,
            "latest": newest[0] if newest else "",
            "latest_age_s": round(time.time() - newest[1], 2) if newest else 0.0,
            "saved": self._stats["saved"],
            "loaded": self._stats["loaded"],
            "restored": self._stats["restored"],
            "resumed": self._stats["resumed"],
            "deleted": self._stats["deleted"],
            "pruned": self._stats["pruned"],
            "corrupt": self._stats["corrupt"],
            "errors": self._stats["errors"],
            "last_checkpoint_id": self._stats["last_checkpoint_id"],
            "uptime_s": round(time.time() - self._started_at, 2) if self._started_at else 0.0,
        }
