"""
NeuroCore Not Sürümleme
========================
Notların zaman içindeki değişimini izleyen, anlık görüntü alan ve geri alma
sağlayan sürüm yönetimi.

Depolama düzeni::

    knowledge_base/_index/versions/
    ├── kn_7f3a91c2/
    │   ├── v1.md          (tam not anlık görüntüsü, frontmatter dahil)
    │   ├── v2.md
    │   └── _history.json  (sürüm üst verisi: zaman, özet, fark)
    └── ...

Tasarım kararları:
    * Anlık görüntüler **tam markdown** olarak saklanır — insan tarafından
      okunabilir ve gerektiğinde elle geri yüklenebilir.
    * Sürüm başına yalnızca üst veri JSON'da tutulur; gövde kopyalanmaz
      (disk israfını önler).
    * Maksimum sürüm sayısı aşılırsa en eskiler budanır (`max_versions`),
      ancak `pinned` işaretli sürümler asla silinmez.
    * Geri alma **yeni bir sürüm** oluşturur (tarih yeniden yazılmaz);
      böylece geri alma da geri alınabilir.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neurocore.knowledge.note import (
    Note,
    content_hash,
    note_diff,
    now_iso,
    parse_timestamp,
    strip_markdown,
    utc_timestamp,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("knowledge.versioning")


# ─── Veri Yapıları ──────────────────────────────────────────────────────────

@dataclass
class VersionRecord:
    """Tek bir sürümün üst verisi (gövde saklanmaz)."""
    note_id: str
    version: int
    created: str
    checksum: str
    words: int = 0
    chars: int = 0
    title: str = ""
    reason: str = ""
    author: str = "system"
    changed_fields: List[str] = field(default_factory=list)
    summary: str = ""
    pinned: bool = False
    file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "note_id": self.note_id,
            "version": self.version,
            "created": self.created,
            "checksum": self.checksum,
            "words": self.words,
            "chars": self.chars,
            "title": self.title,
            "reason": self.reason,
            "author": self.author,
            "changed_fields": list(self.changed_fields),
            "summary": self.summary,
            "pinned": self.pinned,
            "file": self.file,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionRecord":
        return cls(
            note_id=str(data.get("note_id", "")),
            version=int(data.get("version", 0) or 0),
            created=str(data.get("created", "")),
            checksum=str(data.get("checksum", "")),
            words=int(data.get("words", 0) or 0),
            chars=int(data.get("chars", 0) or 0),
            title=str(data.get("title", "")),
            reason=str(data.get("reason", "")),
            author=str(data.get("author", "system")),
            changed_fields=list(data.get("changed_fields") or []),
            summary=str(data.get("summary", "")),
            pinned=bool(data.get("pinned", False)),
            file=str(data.get("file", "")),
        )

    @property
    def timestamp(self) -> float:
        """Sürüm zamanı (float)."""
        return parse_timestamp(self.created)

    def to_markdown_row(self) -> str:
        """Markdown tablo satırı (rapor için)."""
        pin = "📌" if self.pinned else ""
        return (
            f"| v{self.version} {pin} | {self.created[:19]} | {self.words} | "
            f"{', '.join(self.changed_fields[:4]) or '-'} | {self.reason[:40]} |"
        )


@dataclass
class DiffSummary:
    """İki sürüm arasındaki farkın insan okur özeti."""
    note_id: str
    from_version: int
    to_version: int
    changed_fields: List[str] = field(default_factory=list)
    words_delta: int = 0
    added_lines: int = 0
    removed_lines: int = 0
    preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "note_id": self.note_id,
            "from": self.from_version,
            "to": self.to_version,
            "changed_fields": list(self.changed_fields),
            "words_delta": self.words_delta,
            "added_lines": self.added_lines,
            "removed_lines": self.removed_lines,
            "preview": self.preview,
        }


class VersioningError(Exception):
    """Sürümleme hatası."""


# ─── Ana Sürüm Yöneticisi ───────────────────────────────────────────────────

class NoteVersioner:
    """
    Not sürümlerini yönetir.

    Kullanım:
        versioner = NoteVersioner()
        versioner.initialize(config, event_bus)

        versioner.snapshot(note, reason="kullanıcı düzenlemesi")
        history = versioner.history(note.id)
        versioner.restore(note.id, version=2)          # eski hâle dön
        diff = versioner.diff(note.id, 1, 3)
        versioner.prune(note.id)
    """

    name = "knowledge.versioning"

    def __init__(self, root: Optional[str] = None) -> None:
        """
        Args:
            root: Sürüm dizini (None ise initialize'da kasadan türetilir)
        """
        self._root: Optional[Path] = root and Path(root)
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._lock = threading.RLock()
        self._max_versions = 10
        self._auto_snapshot = True
        self._counter = {
            "snapshots": 0, "restores": 0, "prunes": 0,
            "errors": 0, "skipped": 0, "bytes_written": 0,
        }
        self._history_cache: Dict[str, List[VersionRecord]] = {}

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None,
                   root: Optional[str] = None) -> None:
        """Sürümleme sistemini başlat (idempotent)."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus

        if root:
            self._root = Path(root)
        if self._root is None:
            base: Optional[Path] = None
            if config is not None:
                vault_path = getattr(getattr(config, "knowledge", None), "vault_path", None)
                if vault_path:
                    base = Path(vault_path) / "_index" / "versions"
                else:
                    base = Path(getattr(config, "base_path", os.getcwd())) / \
                        "knowledge_base" / "_index" / "versions"
            else:
                base = Path(os.getcwd()) / "knowledge_base" / "_index" / "versions"
            self._root = base

        knowledge_config = getattr(config, "knowledge", None)
        self._max_versions = int(getattr(knowledge_config, "max_versions", 10) or 10)
        self._auto_snapshot = bool(getattr(knowledge_config, "auto_backup", True))

        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VersioningError(f"Sürüm dizini oluşturulamadı ({self._root}): {exc}") from exc

        self._initialized = True
        logger.info(f"Sürümleme hazır: {self._root} (maks. {self._max_versions} sürüm/not)")

    def shutdown(self) -> None:
        """Kaynakları serbest bırak ve önbelleği boşalt."""
        self._history_cache.clear()
        self._initialized = False
        logger.info(
            f"Sürümleme kapatıldı ({self._counter['snapshots']} anlık görüntü, "
            f"{self._counter['restores']} geri alma)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Sürümleme istatistikleri."""
        return {
            **self._counter,
            "root": str(self._root or ""),
            "initialized": self._initialized,
            "tracked_notes": len(self._history_cache) or self._count_tracked(),
            "max_versions": self._max_versions,
            "auto_snapshot": self._auto_snapshot,
        }

    def _count_tracked(self) -> int:
        """Diskte kaç notun sürüm dizini var?"""
        if not self._root or not self._root.exists():
            return 0
        try:
            return sum(1 for path in self._root.iterdir() if path.is_dir())
        except OSError:
            return 0

    # ─── Anlık Görüntü ──────────────────────────────────────────────────

    def snapshot(self, note: Note, reason: str = "", author: str = "system",
                 force: bool = False) -> Optional[VersionRecord]:
        """
        Notun anlık görüntüsünü alır.

        İçerik parmak izi son sürümle aynıysa **yazma yapılmaz**
        (`force=True` ile zorlanabilir).

        Args:
            note: Anlık görüntüsü alınacak not
            reason: Sürüm gerekçesi (insan okur)
            author: Değişikliği yapan (`user`, `system`, `evolution`, …)
            force: Değişiklik yoksa da yaz

        Returns:
            Oluşturulan sürüm kaydı; atlandıysa None
        """
        self._require_initialized()
        if not self._auto_snapshot and not force:
            self._counter["skipped"] += 1
            return None
        if not note.id:
            return None

        checksum = content_hash(note.title, note.body or "")
        # Geçmişi önbellekten/diskten al. ÖNEMLİ: `history()` **kopya** döndürür;
        # mutasyonun kalıcı olması için `_cache_records` ile önbellekteki
        # referans üzerinde çalışılır. Aksi halde sonraki okuma diskten eski
        # hâli yükler ve sürüm numarası ilerlemez.
        history = self._cache_records(note.id)

        if history and not force:
            last = history[-1]
            if last.checksum == checksum:
                self._counter["skipped"] += 1
                return None

        version = (history[-1].version + 1) if history else max(1, int(note.version or 1))
        directory = self._note_dir(note.id)
        filename = f"v{version}.md"
        target = directory / filename

        try:
            directory.mkdir(parents=True, exist_ok=True)
            markdown = note.to_markdown()
            temp = target.with_name(target.name + ".tmp")
            temp.write_text(markdown, encoding="utf-8", newline="\n")
            os.replace(temp, target)
        except OSError as exc:
            self._counter["errors"] += 1
            logger.error(f"Sürüm kaydedilemedi ({note.id} v{version}): {exc}")
            return None

        previous = None
        if history:
            # Önceki sürümün **Not** nesnesi gerekir (alan karşılaştırması için);
            # dosyadan okunan ham markdown bu yüzden ayrıştırılır.
            previous_text = self._read_snapshot_file(note.id, history[-1].version)
            if previous_text:
                try:
                    previous = Note.from_markdown(previous_text)
                except Exception as exc:
                    logger.debug(f"Önceki sürüm ayrıştırılamadı ({note.id}): {exc}")
                    previous = None

        record = VersionRecord(
            note_id=note.id,
            version=version,
            created=now_iso(),
            checksum=checksum,
            words=note.word_count,
            chars=note.char_count,
            title=note.title,
            reason=reason or "otomatik",
            author=author,
            changed_fields=self._changed_fields(previous, note) if previous else ["initial"],
            summary=note.excerpt[:200],
            file=filename,
        )

        history.append(record)
        self._write_history(note.id, history)
        self._counter["snapshots"] += 1
        self._counter["bytes_written"] += len(markdown.encode("utf-8"))

        self._prune(note.id, history)

        self._emit("knowledge.version_created", {
            "id": note.id, "version": version, "reason": record.reason,
            "words": record.words, "author": author,
        })
        logger.debug(f"Sürüm oluşturuldu: {note.id} v{version} ({record.reason})")
        return record

    def snapshot_many(self, notes: Sequence[Note], reason: str = "") -> int:
        """Birden çok notun anlık görüntüsünü alır; yazılan sayısını döndürür."""
        written = 0
        for note in notes:
            if self.snapshot(note, reason=reason):
                written += 1
        return written

    # ─── Sorgulama ──────────────────────────────────────────────────────

    def history(self, note_id: str, limit: Optional[int] = None) -> List[VersionRecord]:
        """
        Notun sürüm geçmişini döndürür (eskiden yeniye).

        Döndürülen liste bir **kopyadır**; çağıran tarafından değiştirilmesi
        önbelleği bozmaz. Yazma işlemleri `_cache_records` kullanır.

        Args:
            note_id: Not kimliği
            limit: Son kaç sürüm (None = tümü)
        """
        self._require_initialized()
        records = self._cache_records(note_id)
        if limit is not None:
            return list(records[-limit:])
        return list(records)

    def _cache_records(self, note_id: str) -> List[VersionRecord]:
        """
        Notun sürüm kayıtlarını **önbellekteki referans** olarak döndürür.

        İlk erişimde diskten okur ve önbelleğe koyar. Sonraki çağrılar aynı
        liste nesnesini döndürür; bu sayede mutasyonlar (ekleme, budama,
        sabitleme) kalıcı olur.
        """
        with self._lock:
            records = self._history_cache.get(note_id)
            if records is None:
                records = self._read_history(note_id)
                self._history_cache[note_id] = records
            return records

    def latest(self, note_id: str) -> Optional[VersionRecord]:
        """Notun en son sürümü."""
        history = self.history(note_id)
        return history[-1] if history else None

    def version_count(self, note_id: str) -> int:
        """Notun sürüm sayısı."""
        return len(self.history(note_id))

    def load_snapshot(self, note_id: str, version: int) -> Optional[Note]:
        """
        Belirli bir sürümü Not olarak yükler.

        Returns:
            Note veya dosya yoksa/bozuksa None
        """
        markdown = self._read_snapshot_file(note_id, version)
        if markdown is None:
            return None
        note = Note.from_markdown(markdown)
        note.id = note_id
        return note

    def read_snapshot_text(self, note_id: str, version: int) -> Optional[str]:
        """Sürümün ham markdown metni."""
        return self._read_snapshot_file(note_id, version)

    def stats(self, note_id: str) -> Dict[str, Any]:
        """Tek notun sürüm istatistikleri."""
        history = self.history(note_id)
        if not history:
            return {"note_id": note_id, "versions": 0}
        first, last = history[0], history[-1]
        growth = last.words - first.words
        return {
            "note_id": note_id,
            "title": last.title,
            "versions": len(history),
            "first": first.created,
            "last": last.created,
            "words_now": last.words,
            "words_first": first.words,
            "growth": growth,
            "growth_ratio": round(last.words / first.words, 3) if first.words else 0.0,
            "authors": sorted({record.author for record in history}),
            "pinned": sum(1 for record in history if record.pinned),
            "avg_words_delta": round(growth / max(1, len(history) - 1), 2),
        }

    def all_tracked(self) -> List[str]:
        """Sürümü izlenen tüm not kimlikleri."""
        if not self._root or not self._root.exists():
            return []
        try:
            return sorted(path.name for path in self._root.iterdir() if path.is_dir())
        except OSError:
            return []

    def global_stats(self) -> Dict[str, Any]:
        """Tüm kasa için sürüm istatistikleri."""
        tracked = self.all_tracked()
        total_versions = 0
        total_bytes = 0
        busiest: List[Tuple[str, int]] = []

        for note_id in tracked:
            records = self.history(note_id)
            total_versions += len(records)
            busiest.append((note_id, len(records)))
            try:
                directory = self._note_dir(note_id)
                total_bytes += sum(
                    path.stat().st_size for path in directory.glob("*.md") if path.is_file()
                )
            except OSError:
                pass

        busiest.sort(key=lambda item: -item[1])
        return {
            "tracked_notes": len(tracked),
            "total_versions": total_versions,
            "total_bytes": total_bytes,
            "mean_versions_per_note": round(total_versions / len(tracked), 2) if tracked else 0.0,
            "most_versioned": busiest[:10],
        }

    # ─── Fark ───────────────────────────────────────────────────────────

    def diff(self, note_id: str, from_version: int, to_version: Optional[int] = None) -> Optional[DiffSummary]:
        """
        İki sürüm arasındaki farkı çıkarır.

        Args:
            note_id: Not kimliği
            from_version: Karşılaştırılacak eski sürüm
            to_version: Yeni sürüm (None = en son)

        Returns:
            DiffSummary veya sürümlerden biri yoksa None
        """
        history = self.history(note_id)
        if not history:
            return None
        if to_version is None:
            to_version = history[-1].version

        old_note = self.load_snapshot(note_id, from_version)
        new_note = self.load_snapshot(note_id, to_version)
        if old_note is None or new_note is None:
            return None

        raw = note_diff(old_note, new_note)
        old_lines = [line for line in (old_note.body or "").splitlines() if line.strip()]
        new_lines = [line for line in (new_note.body or "").splitlines() if line.strip()]
        old_set, new_set = set(old_lines), set(new_lines)

        return DiffSummary(
            note_id=note_id,
            from_version=from_version,
            to_version=to_version,
            changed_fields=list(raw.get("changed", [])),
            words_delta=int(raw.get("body_delta_words", 0)),
            added_lines=len(new_set - old_set),
            removed_lines=len(old_set - new_set),
            preview=self._unified_preview(old_lines, new_lines),
        )

    @staticmethod
    def _unified_preview(old_lines: List[str], new_lines: List[str],
                         max_lines: int = 30) -> str:
        """Basit birleşik fark önizlemesi üretir."""
        import difflib
        diff = list(difflib.unified_diff(
            old_lines, new_lines, lineterm="", n=1,
            fromfile="eski", tofile="yeni",
        ))
        if len(diff) > max_lines:
            diff = diff[:max_lines] + [f"… (+{len(diff) - max_lines} satır daha)"]
        return "\n".join(diff)

    def timeline(self, note_id: str) -> str:
        """Notun sürüm zaman çizelgesini markdown olarak döndürür."""
        history = self.history(note_id)
        if not history:
            return f"_`{note_id}` için sürüm kaydı yok._"
        lines = [
            f"### Sürüm Geçmişi — {history[-1].title or note_id}",
            "",
            "| Sürüm | Zaman | Kelime | Değişen Alanlar | Gerekçe |",
            "|---|---|---|---|---|",
        ]
        lines += [record.to_markdown_row() for record in reversed(history)]
        return "\n".join(lines)

    # ─── Geri Alma ──────────────────────────────────────────────────────

    def restore(self, note_id: str, version: int, vault: Any = None,
                reason: str = "") -> Optional[Note]:
        """
        Notu eski bir sürüme döndürür.

        Geri alma **yeni bir sürüm** oluşturur (v_eski'nin kopyası v_yeni olur);
        böylece geri alma işlemi de geri alınabilir.

        Args:
            note_id: Not kimliği
            version: Geri dönülecek sürüm numarası
            vault: `KnowledgeVault` — verilirse diskteki not da güncellenir
            reason: Geri alma gerekçesi

        Returns:
            Geri yüklenen not (vault verilmediyse de Note döner)
        """
        self._require_initialized()
        restored = self.load_snapshot(note_id, version)
        if restored is None:
            logger.warning(f"Geri alınacak sürüm bulunamadı: {note_id} v{version}")
            return None

        restored.version = int(restored.version or 1)
        if vault is not None:
            current = vault.read_note(note_id)
            if current is not None:
                # Sürüm numarasını mevcut notun ilerisine taşı
                restored.version = max(current.version, version) + 1
                restored.path = current.path
            restored.touch()
            vault.save_note(restored, emit=False)

        self.snapshot(restored, reason=reason or f"v{version} sürümüne geri dönüş",
                      author="user", force=True)
        self._counter["restores"] += 1

        self._emit("knowledge.version_restored", {
            "id": note_id, "from_version": version, "new_version": restored.version,
        })
        logger.info(f"Not geri yüklendi: {note_id} → v{version} (yeni v{restored.version})")
        return restored

    def rollback(self, note_id: str, steps: int = 1, vault: Any = None) -> Optional[Note]:
        """
        N adım geriye döner.

        Args:
            note_id: Not kimliği
            steps: Kaç sürüm geriye
            vault: Diske de yaz
        """
        history = self.history(note_id)
        if len(history) <= steps:
            return None
        target = history[-(steps + 1)].version
        return self.restore(note_id, target, vault=vault,
                           reason=f"{steps} adım geri alma")

    def delete_history(self, note_id: str, keep_latest: bool = True) -> int:
        """
        Notun sürüm geçmişini siler.

        Args:
            note_id: Not kimliği
            keep_latest: En son sürümü koru

        Returns:
            Silinen sürüm sayısı
        """
        history = self._cache_records(note_id)
        if not history:
            return 0
        keep = history[-1].version if keep_latest else None
        removed = 0

        for record in history:
            if keep is not None and record.version == keep:
                continue
            path = self._note_dir(note_id) / record.file
            try:
                if path.exists():
                    path.unlink()
                    removed += 1
            except OSError:
                continue

        kept = [record for record in history if keep is not None and record.version == keep]
        with self._lock:
            self._history_cache[note_id] = kept
        self._write_history(note_id, kept)
        self._emit("knowledge.version_history_deleted", {"id": note_id, "removed": removed})
        return removed

    def pin(self, note_id: str, version: int, pinned: bool = True) -> bool:
        """Bir sürümü budamadan korur."""
        history = self._cache_records(note_id)
        for record in history:
            if record.version == version:
                record.pinned = pinned
                self._write_history(note_id, history)
                return True
        return False

    # ─── Budama ─────────────────────────────────────────────────────────

    def _prune(self, note_id: str, history: Optional[List[VersionRecord]] = None) -> int:
        """
        Maksimum sürüm sayısını aşan en eski sürümleri siler.

        `pinned` sürümler korunur.
        """
        records = history if history is not None else self._cache_records(note_id)
        if len(records) <= self._max_versions:
            return 0

        removable = [record for record in records if not record.pinned]
        excess = len(records) - self._max_versions
        if excess <= 0 or not removable:
            return 0

        to_remove = removable[:excess]
        remove_ids = {record.version for record in to_remove}
        for record in to_remove:
            try:
                path = self._note_dir(note_id) / record.file
                if path.exists():
                    path.unlink()
            except OSError:
                continue

        kept = [record for record in records if record.version not in remove_ids]
        with self._lock:
            self._history_cache[note_id] = kept
        self._write_history(note_id, kept)
        self._counter["prunes"] += len(to_remove)
        logger.debug(f"Sürüm budandı: {note_id} ({len(to_remove)} eski sürüm silindi)")
        return len(to_remove)

    def prune(self, note_id: Optional[str] = None, max_versions: Optional[int] = None) -> int:
        """
        Sürümleri budar.

        Args:
            note_id: Yalnızca bu notu buda (None = tüm kasa)
            max_versions: Geçici maksimum sürüm sayısı

        Returns:
            Silinen sürüm sayısı
        """
        original = self._max_versions
        if max_versions is not None:
            self._max_versions = int(max_versions)
        try:
            if note_id:
                return self._prune(note_id)
            total = 0
            for tracked in self.all_tracked():
                total += self._prune(tracked)
            return total
        finally:
            self._max_versions = original

    def compact(self, note_id: str) -> Dict[str, Any]:
        """
        Bir notun sürüm dizinini sıkıştırır.

        Eksik dosyaları (üst veride var, diskte yok) temizler, fazla
        dosyaları (diskte var, üst veride yok) karantinaya taşır.
        """
        history = self.history(note_id)
        known = {record.file for record in history}
        directory = self._note_dir(note_id)
        removed_records: List[int] = []
        orphaned: List[str] = []

        for record in list(history):
            if record.file and not (directory / record.file).exists():
                removed_records.append(record.version)
        for path in directory.glob("v*.md"):
            if path.name not in known:
                target = directory / "_orphan"
                target.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(path), str(target / path.name))
                    orphaned.append(path.name)
                except OSError:
                    continue

        if removed_records:
            kept = [record for record in history if record.version not in set(removed_records)]
            with self._lock:
                self._history_cache[note_id] = kept
            self._write_history(note_id, kept)

        result = {"missing_files": removed_records, "orphaned": orphaned}
        if removed_records or orphaned:
            logger.info(f"Sürüm dizini sıkıştırıldı: {note_id} {result}")
        return result

    # ─── Dışa Aktarma ───────────────────────────────────────────────────

    def export_markdown(self, note_id: str, path: Optional[str] = None) -> str:
        """Notun tüm sürüm geçmişini markdown raporu olarak yazar."""
        content = self.timeline(note_id)
        history = self.history(note_id)
        if history:
            latest = self.load_snapshot(note_id, history[-1].version)
            if latest:
                content += "\n\n---\n\n## Güncel İçerik\n\n" + latest.body

        target = Path(path) if path else (self._note_dir(note_id) / "HISTORY.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temp, target)
        return str(target)

    def export_all(self, path: Optional[str] = None) -> str:
        """Tüm kasanın sürüm raporunu tek dosyada toplar."""
        lines = [
            "# Sürüm Geçmişi Raporu",
            "",
            f"_Oluşturma: {now_iso()}_",
            "",
            f"İzlenen not sayısı: **{len(self.all_tracked())}**",
            "",
        ]
        for note_id in self.all_tracked():
            lines.append(self.timeline(note_id))
            lines.append("")
        content = "\n".join(lines)
        target = Path(path) if path else (self._root / "VERSION_REPORT.md")  # type: ignore[operator]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return str(target)

    # ─── İç Yardımcılar ─────────────────────────────────────────────────

    def _require_initialized(self) -> None:
        """Başlatılmadıysa hata ver."""
        if not self._initialized or self._root is None:
            raise VersioningError("Sürümleme başlatılmadı — initialize() çağırın")

    def _note_dir(self, note_id: str) -> Path:
        """Notun sürüm dizini."""
        assert self._root is not None
        safe = re.sub(r"[^A-Za-z0-9_\-]", "", str(note_id))[:64] or "bilinmeyen"
        return self._root / safe

    def _history_file(self, note_id: str) -> Path:
        """Notun sürüm üst veri dosyası."""
        return self._note_dir(note_id) / "_history.json"

    def _read_history(self, note_id: str) -> List[VersionRecord]:
        """Üst veri dosyasını okur (bozuksa boş liste)."""
        path = self._history_file(note_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Sürüm geçmişi okunamadı ({path}): {exc}")
            return []

        records_raw = data.get("versions") if isinstance(data, dict) else data
        if not isinstance(records_raw, list):
            return []
        records = [VersionRecord.from_dict(item) for item in records_raw if isinstance(item, dict)]
        records.sort(key=lambda record: record.version)
        return records

    def _write_history(self, note_id: str, records: List[VersionRecord]) -> None:
        """Üst veri dosyasını atomik yazar."""
        directory = self._note_dir(note_id)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._counter["errors"] += 1
            logger.error(f"Sürüm dizini oluşturulamadı ({directory}): {exc}")
            return

        payload = {
            "note_id": note_id,
            "updated": now_iso(),
            "count": len(records),
            "versions": [record.to_dict() for record in records],
        }
        target = self._history_file(note_id)
        temp = target.with_name(target.name + ".tmp")
        try:
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8", newline="\n",
            )
            os.replace(temp, target)
        except OSError as exc:
            self._counter["errors"] += 1
            logger.error(f"Sürüm geçmişi yazılamadı ({target}): {exc}")

    def _read_snapshot_file(self, note_id: str, version: int) -> Optional[str]:
        """Sürüm dosyasının metnini okur."""
        path = self._note_dir(note_id) / f"v{int(version)}.md"
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(f"Sürüm dosyası okunamadı ({path}): {exc}")
            return None

    @staticmethod
    def _changed_fields(previous: Optional[Note], current: Note) -> List[str]:
        """İki not arasında değişen alanların listesi."""
        if previous is None:
            return ["initial"]
        changed: List[str] = []
        if strip_markdown(previous.body) != strip_markdown(current.body):
            changed.append("body")
        if previous.title != current.title:
            changed.append("title")
        if set(previous.tags) != set(current.tags):
            changed.append("tags")
        if abs(previous.confidence - current.confidence) > 1e-6:
            changed.append("confidence")
        if previous.sources != current.sources:
            changed.append("sources")
        if set(previous.links) != set(current.links):
            changed.append("links")
        if previous.kind != current.kind or previous.category != current.category:
            changed.append("classification")
        return changed or ["metadata"]

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<NoteVersioner root={self._root} snapshots={self._counter['snapshots']}>"


__all__ = [
    "NoteVersioner",
    "VersionRecord",
    "DiffSummary",
    "VersioningError",
]
