"""
NeuroCore Bilgi Kasası (Vault)
===============================
Obsidian uyumlu markdown kasasının dosya sistemi katmanı.

Sorumluluklar:
    * Kasa dizin düzenini kurmak ve doğrulamak
    * Notları diske atomik yazmak / diskten okumak
    * Not kimliği ↔ dosya yolu eşlemesini yönetmek
    * Bütünlük taraması yapmak, bozuk notları karantinaya almak
    * Ekleri (görüntü, ses) saklamak ve çözmek

Kasa düzeni::

    knowledge_base/
    ├── 00_meta/  10_concepts/  20_entities/  30_sources/
    ├── 40_projects/  50_sessions/  60_skills/  70_journal/  90_archive/
    ├── _index/    (SQLite + graf + sürümler)
    └── _attachments/

Hiçbir metot kasa dışına yazmaz; yol çözümlemesi `_resolve_within_vault`
üzerinden yapılır ve kasa kökünün dışına çıkan yollar reddedilir.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from neurocore.knowledge.note import (
    Note,
    NOTE_CATEGORIES,
    content_hash,
    make_note,
    new_note_id,
    now_iso,
    slugify,
    body_similarity,
    title_similarity,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("knowledge.vault")


# ─── Sabitler ───────────────────────────────────────────────────────────────

#: Kategori → dizin eşlemesi
#:
#: Hem **üst kategoriler** (`learned`, `research`, `user`, `meta`) hem de
#: **not türleri** (`skill`, `journal`, `concept`, …) burada eşlenir; böylece
#: `create_note(kind="skill")` çağrısı doğrudan doğru dizine yazılır.
CATEGORY_DIRS: Dict[str, str] = {
    # Üst kategoriler
    "system": "00_meta",
    "meta": "00_meta",
    "learned": "10_concepts",
    "research": "30_sources",
    "user": "20_entities",
    # Not türleri
    "concept": "10_concepts",
    "summary": "10_concepts",
    "skill": "60_skills",
    "entity": "20_entities",
    "source": "30_sources",
    "project": "40_projects",
    "session": "50_sessions",
    "journal": "70_journal",
    "qa": "10_concepts",
}

ARCHIVE_DIR = "90_archive"
INDEX_DIR = "_index"
ATTACHMENT_DIR = "_attachments"
CORRUPT_DIR = "90_archive/_corrupt"

#: Tüm kasada taranan uzantılar
NOTE_EXTENSIONS: Tuple[str, ...] = (".md", ".markdown")

#: Sistem dosyaları (not olarak sayılmaz)
SYSTEM_FILES: Tuple[str, ...] = ("README.md", "index.md", "KASA.md")

#: Dosya adında yasak karakterler
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTI_DASH_RE = re.compile(r"-{2,}")


class VaultError(Exception):
    """Kasa işlem hatası."""


class VaultNotInitialized(VaultError):
    """Kasa henüz başlatılmadı."""


# ─── Kasa İstatistikleri ────────────────────────────────────────────────────

@dataclass
class VaultStats:
    """Kasa içeriğinin özet istatistikleri."""
    total_notes: int = 0
    total_words: int = 0
    total_bytes: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    by_kind: Dict[str, int] = field(default_factory=dict)
    tag_histogram: Dict[str, int] = field(default_factory=dict)
    mean_confidence: float = 0.0
    mean_importance: float = 0.0
    low_confidence: int = 0
    shallow: int = 0
    with_todos: int = 0
    orphan_links: int = 0
    attachments: int = 0
    oldest: float = 0.0
    newest: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "total_notes": self.total_notes,
            "total_words": self.total_words,
            "total_bytes": self.total_bytes,
            "by_category": dict(sorted(self.by_category.items(), key=lambda kv: -kv[1])),
            "by_kind": dict(sorted(self.by_kind.items(), key=lambda kv: -kv[1])),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_importance": round(self.mean_importance, 4),
            "low_confidence": self.low_confidence,
            "shallow": self.shallow,
            "with_todos": self.with_todos,
            "orphan_links": self.orphan_links,
            "attachments": self.attachments,
        }
        top_tags = sorted(self.tag_histogram.items(), key=lambda kv: -kv[1])[:15]
        data["top_tags"] = dict(top_tags)
        return data


# ─── Ana Kasa Sınıfı ────────────────────────────────────────────────────────

class KnowledgeVault:
    """
    Obsidian uyumlu bilgi kasası.

    Kullanım:
        vault = KnowledgeVault()
        vault.initialize(config, event_bus)

        note = vault.create_note("GPU Termal Yönetimi", "## Özet\\n...", tags=["gpu"])
        vault.save_note(note)

        found = vault.search_by_title("termal")
        all_notes = list(vault.iter_notes())

    Not: Arama, graf ve indeksleme `KnowledgeIndex` / `HybridSearch` /
    `KnowledgeGraph` sınıflarında; bu sınıf yalnızca dosya sisteminden sorumludur.
    """

    #: Katman kimliği (olay kaynağı)
    name = "knowledge.vault"

    def __init__(self, root: Optional[str] = None) -> None:
        """
        Args:
            root: Kasa kök dizini (None ise initialize'da config'ten alınır)
        """
        self._root: Optional[Path] = None
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._lock = threading.RLock()
        self._path_index: Dict[str, str] = {}      # note_id -> relative path
        self._title_index: Dict[str, List[str]] = {}  # title_slug -> [note_id]
        self._counter = {"reads": 0, "writes": 0, "errors": 0, "quarantined": 0}
        if root:
            self._root = Path(root).expanduser().resolve()

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Kasayı kur: dizinleri oluştur, dizinleri tara, kimlik haritasını çıkar."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus

        if self._root is None:
            root = None
            if config is not None:
                root = getattr(getattr(config, "knowledge", None), "vault_path", None)
                if not root:
                    base = getattr(config, "base_path", None) or os.getcwd()
                    root = os.path.join(base, "knowledge_base")
            else:
                root = os.path.join(os.getcwd(), "knowledge_base")
            self._root = Path(root).expanduser().resolve()

        self._ensure_layout()
        self._initialized = True

        count = self.rebuild_path_index()
        logger.info(
            f"Kasa hazır: {self._root} ({count} not dizine alındı)"
        )
        self._emit("knowledge.vault_ready", {
            "root": str(self._root),
            "notes": count,
        })

    def shutdown(self) -> None:
        """Kaynakları serbest bırak (idempotent)."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info(
            f"Kasa kapatıldı (okuma: {self._counter['reads']}, "
            f"yazma: {self._counter['writes']})"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Kasa sağlık metrikleri."""
        return {
            **self._counter,
            "root": str(self._root) if self._root else "",
            "initialized": self._initialized,
            "indexed_notes": len(self._path_index),
            "exists": bool(self._root and self._root.exists()),
        }

    # ─── Dizin Düzeni ───────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        """Kasa kök dizini."""
        self._require_initialized()
        assert self._root is not None
        return self._root

    @property
    def index_dir(self) -> Path:
        """İndeks dizini."""
        return self.root / INDEX_DIR

    @property
    def attachment_dir(self) -> Path:
        """Ek dosyalar dizini."""
        return self.root / ATTACHMENT_DIR

    @property
    def archive_dir(self) -> Path:
        """Arşiv dizini."""
        return self.root / ARCHIVE_DIR

    def _all_dirs(self) -> List[str]:
        """Oluşturulacak dizinlerin tam listesi."""
        dirs = sorted(set(CATEGORY_DIRS.values())) + [ARCHIVE_DIR, INDEX_DIR, ATTACHMENT_DIR, CORRUPT_DIR]
        dirs += [str(Path(INDEX_DIR) / "versions"), str(Path(INDEX_DIR) / "snapshots")]
        return dirs

    def _ensure_layout(self) -> None:
        """Kasa dizin düzenini kur (eksikleri tamamla)."""
        assert self._root is not None
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VaultError(f"Kasa kökü oluşturulamadı ({self._root}): {exc}") from exc

        for rel in self._all_dirs():
            target = self._root / rel
            try:
                target.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise VaultError(f"Kasa dizini oluşturulamadı ({target}): {exc}") from exc

        readme = self._root / "README.md"
        if not readme.exists():
            self._write_text(readme, self._readme_text())

    def _readme_text(self) -> str:
        """Kasa köküne yazılan tanıtım dosyası."""
        return (
            "---\n"
            "id: kn_readme\n"
            "title: \"NeuroCore Bilgi Kasası\"\n"
            "kind: meta\n"
            "category: meta\n"
            "tags: [meta, kasa, neurocore]\n"
            "confidence: 1.0\n"
            f"created: {now_iso()}\n"
            f"updated: {now_iso()}\n"
            "version: 1\n"
            "---\n\n"
            "# NeuroCore Bilgi Kasası\n\n"
            "Bu dizin Obsidian ile doğrudan açılabilir. Her not bir markdown\n"
            "dosyasıdır; frontmatter alanları NeuroCore tarafından yönetilir.\n\n"
            "## Dizinler\n\n"
            "| Dizin | İçerik |\n"
            "|---|---|\n"
            "| `00_meta` | Sistem ve şema notları |\n"
            "| `10_concepts` | Kavramlar, öğrenilen bilgiler |\n"
            "| `20_entities` | Kişi, kurum, ürün, kullanıcı |\n"
            "| `30_sources` | Araştırma kaynakları ve alıntılar |\n"
            "| `40_projects` | Proje ve görev bağlamları |\n"
            "| `50_sessions` | Konuşma özetleri |\n"
            "| `60_skills` | Öğrenilmiş yordamlar |\n"
            "| `70_journal` | Sistem günlüğü ve evrim kaydı |\n"
            "| `90_archive` | Arşivlenmiş ve karantinaya alınmış notlar |\n"
            "| `_index` | SQLite indeksi, graf, sürümler (elle düzenlemeyin) |\n"
            "| `_attachments` | Görüntü ve ses ekleri |\n\n"
            "## Frontmatter Alanları\n\n"
            "`id` · `title` · `kind` · `category` · `tags` · `confidence` ·\n"
            "`importance` · `created` · `updated` · `version` · `sources` · `links`\n"
        )

    # ─── Yol Çözümleme ─────────────────────────────────────────────────

    def _require_initialized(self) -> None:
        """Kasa başlatılmadıysa hata ver."""
        if not self._initialized or self._root is None:
            raise VaultNotInitialized(
                "Kasa başlatılmadı — önce initialize(config, event_bus) çağırın"
            )

    def _resolve_within_vault(self, relative: str) -> Path:
        """
        Kasa köküne göreli bir yolu mutlak yola çevirir.

        Kasa dışına çıkan yollar (`..`, mutlak yol) reddedilir.
        """
        self._require_initialized()
        assert self._root is not None
        candidate = (self._root / relative).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise VaultError(f"Kasa dışına çıkan yol reddedildi: {relative}") from exc
        return candidate

    def category_dir(self, category: str) -> str:
        """Kategoriye karşılık gelen dizin adını döndürür."""
        key = (category or "learned").strip().lower()
        return CATEGORY_DIRS.get(key, CATEGORY_DIRS["learned"])

    def note_relative_path(self, note: Note) -> str:
        """
        Notun kasa içindeki göreli yolunu belirler.

        Dosya adı `slugified-başlık--id.md` biçimindedir: Obsidian'da okunabilir
        kalır ve kimlik çakışması yaşanmaz.
        """
        directory = self.category_dir(note.category)
        slug = slugify(note.title, max_length=70) or note.title_slug or "not"
        short_id = note.id.replace("kn_", "")[:8]
        filename = f"{slug}--{short_id}.md"
        return f"{directory}/{filename}"

    def note_path(self, note: Note) -> Path:
        """Notun mutlak dosya yolu."""
        if note.path:
            existing = Path(note.path)
            if existing.is_absolute():
                return existing
        return self._resolve_within_vault(self.note_relative_path(note))

    # ─── Okuma ──────────────────────────────────────────────────────────

    def read_note(self, relative_or_id: str) -> Optional[Note]:
        """
        Tek bir notu okur.

        Args:
            relative_or_id: Kasa göreli yol, mutlak yol veya not kimliği

        Returns:
            Note veya bulunamazsa None
        """
        path = self.find_path(relative_or_id)
        if path is None:
            return None
        return self._load_path(path)

    def _load_path(self, path: Path) -> Optional[Note]:
        """Dosyayı okuyup Note'a çevirir; bozuksa karantinaya alır."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            self._counter["reads"] += 1
        except (OSError, UnicodeDecodeError) as exc:
            self._counter["errors"] += 1
            logger.error(f"Not okunamadı ({path}): {exc}")
            return None

        try:
            relative = str(path.relative_to(self._root)) if self._root else str(path)
        except ValueError:
            relative = str(path)

        note = Note.from_markdown(text, path=relative)
        note.clamp_scores()

        # Kimlik indeksini güncelle
        with self._lock:
            self._path_index[note.id] = relative
            slug = note.title_slug
            bucket = self._title_index.setdefault(slug, [])
            if note.id not in bucket:
                bucket.append(note.id)
        return note

    def load_file(self, path: Path) -> Optional[Note]:
        """Belirtilen dosyayı not olarak yükler (yol doğrulaması yapılır)."""
        if not path.exists() or not path.is_file():
            return None
        return self._load_path(path)

    def find_path(self, relative_or_id: str) -> Optional[Path]:
        """
        Bir kimliği/yolu gerçek dosya yoluna çözer.

        Sıra: doğrudan yol → kimlik indeksi → kasa içi tarama.
        """
        self._require_initialized()
        if not relative_or_id:
            return None
        assert self._root is not None

        raw = str(relative_or_id).strip().strip('"')

        # 1) Doğrudan yol
        try:
            direct = Path(raw)
            if direct.is_absolute() and direct.is_file():
                return direct
            candidate = (self._root / raw).resolve()
            if candidate.is_file() and self._is_within_vault(candidate):
                return candidate
        except (OSError, ValueError):
            pass

        # 2) Kimlik indeksi
        key = raw[:-3] if raw.endswith(".md") else raw
        with self._lock:
            relative = self._path_index.get(key)
        if relative:
            candidate = self._root / relative
            if candidate.is_file():
                return candidate

        # 3) Dosya adıyla arama
        for path in self.iter_note_files():
            if path.stem == key or path.name == raw:
                return path
        return None

    def _is_within_vault(self, path: Path) -> bool:
        """Yol kasa içinde mi?"""
        if self._root is None:
            return False
        try:
            path.resolve().relative_to(self._root)
            return True
        except ValueError:
            return False

    def iter_note_files(self, include_archive: bool = False) -> Iterator[Path]:
        """
        Kasadaki tüm not dosyalarını dolaşır.

        Args:
            include_archive: `90_archive` altını da dahil et
        """
        self._require_initialized()
        assert self._root is not None
        for path in sorted(self._root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in NOTE_EXTENSIONS:
                continue
            relative = path.relative_to(self._root)
            parts = relative.parts
            if not include_archive and ARCHIVE_DIR in parts:
                continue
            if INDEX_DIR in parts or ATTACHMENT_DIR in parts:
                continue
            yield path

    def iter_notes(
        self,
        category: Optional[str] = None,
        kind: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        include_archive: bool = False,
        limit: Optional[int] = None,
        exclude_kinds: Optional[Iterable[str]] = None,
    ) -> Iterator[Note]:
        """
        Kasadaki notları dolaşır (filtrelerle).

        Args:
            category: Kategori filtresi
            kind: Not türü filtresi
            tags: Bu etiketlerden **en az birini** taşıyan notlar
            include_archive: Arşivi dahil et
            limit: Maksimum not sayısı
            exclude_kinds: Bu türdeki notları atla
        """
        self._require_initialized()
        wanted_tags = {str(t).strip().lower() for t in (tags or []) if str(t).strip()}
        excluded = {str(k).strip() for k in (exclude_kinds or []) if str(k).strip()}
        directory = self.category_dir(category) if category else None
        count = 0

        for path in self.iter_note_files(include_archive=include_archive):
            if directory:
                try:
                    if path.relative_to(self._root).parts[0] != directory:
                        continue
                except (ValueError, IndexError):
                    continue
            note = self._load_path(path)
            if note is None:
                continue
            if kind and note.kind != kind:
                continue
            if excluded and note.kind in excluded:
                continue
            if wanted_tags and not wanted_tags.intersection(note.tags):
                continue
            yield note
            count += 1
            if limit is not None and count >= limit:
                return

    def load_all(self, limit: Optional[int] = None, include_archive: bool = False) -> List[Note]:
        """Tüm notları listeye yükler (indeksleme/analiz için)."""
        return list(self.iter_notes(limit=limit, include_archive=include_archive))

    # ─── Yazma ──────────────────────────────────────────────────────────

    def write_note(
        self, note: Note, *, touch: bool = True, emit: bool = True
    ) -> Tuple[str, Optional[str]]:
        """
        Notu diske yazar (atomik).

        Args:
            note: Yazılacak not
            touch: `updated` zamanını ve sürümü güncelle
            emit: Olay yayınla

        Returns:
            (kasa_göreli_yol, önceki_dosya_metni_veya_None)
        """
        self._require_initialized()
        assert self._root is not None

        is_new = not note.path
        if touch and not is_new:
            note.touch()
        note.clamp_scores()

        relative = self.note_relative_path(note)
        target = self._resolve_within_vault(relative)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Yazmadan önce eski dosyayı yedekle (sürümleme katmanı kullanır)
        old_text: Optional[str] = None
        if target.exists() and not is_new:
            try:
                old_text = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                old_text = None

        markdown = note.to_markdown()
        self._atomic_write_text(target, markdown)

        # Eski yol farklıysa temizle (başlık/kategori değişmişse)
        if note.path and note.path != relative:
            old_path = self._resolve_within_vault(note.path)
            if old_path.exists() and old_path != target:
                try:
                    old_path.unlink()
                except OSError as exc:
                    logger.warning(f"Eski not dosyası silinemedi ({old_path}): {exc}")

        note.path = relative
        with self._lock:
            self._path_index[note.id] = relative
            slug = note.title_slug
            bucket = self._title_index.setdefault(slug, [])
            if note.id not in bucket:
                bucket.append(note.id)

        self._counter["writes"] += 1

        if emit:
            self._emit(
                "knowledge.note_written" if not is_new else "knowledge.note_created",
                {
                    "id": note.id,
                    "title": note.title,
                    "path": relative,
                    "version": note.version,
                    "words": note.word_count,
                    "tags": list(note.tags),
                },
            )
        logger.debug(f"Not yazıldı: {relative} (v{note.version}, {note.word_count} kelime)")
        return relative, old_text

    def save_note(self, note: Note, **kwargs: Any) -> str:
        """
        `write_note` sarmalayıcısı — yalnızca kasa göreli yolunu döndürür.
        """
        relative, _ = self.write_note(note, **kwargs)
        return relative

    def create_note(
        self,
        title: str,
        body: str = "",
        *,
        kind: str = "concept",
        category: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        confidence: float = 0.5,
        importance: float = 0.5,
        sources: Optional[Iterable[str]] = None,
        links: Optional[Iterable[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Note:
        """
        Yeni not oluşturur ve diske yazar.

        Kategori verilmezse **`kind`'dan türetilir**; eşleme `CATEGORY_DIRS`
        üzerinden yapılır (ör. `skill` → `60_skills`, `journal` → `70_journal`).
        """
        resolved_kind = (kind or "concept").strip() or "concept"
        resolved_category = category or resolved_kind

        note = make_note(
            title=title,
            body=body,
            kind=resolved_kind,
            category=resolved_category,
            tags=tags,
            confidence=confidence,
            importance=importance,
            sources=sources,
            links=links,
            extra=extra,
        )
        self.save_note(note, touch=False)
        return note

    def update_note(
        self,
        note_id: str,
        body: Optional[str] = None,
        *,
        title: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
        append: bool = False,
        sources: Optional[Iterable[str]] = None,
    ) -> Optional[Note]:
        """
        Var olan notu günceller.

        Args:
            note_id: Not kimliği
            body: Yeni gövde; `append=True` ise gövde sonuna eklenir
            title: Yeni başlık
            tags: Eklenecek etiketler
            confidence: Yeni güven değeri
            importance: Yeni önem değeri
            append: Gövdeyi değiştirmek yerine ekle
            sources: Eklenecek kaynaklar

        Returns:
            Güncellenmiş not veya bulunamazsa None
        """
        note = self.read_note(note_id)
        if note is None:
            logger.warning(f"Güncellenecek not bulunamadı: {note_id}")
            return None

        if title:
            note.title = title.strip()
        if body is not None:
            note.body = f"{note.body.rstrip()}\n\n{body.strip()}" if append else body.strip()
        if tags:
            note.merge_tags(tags)
        if confidence is not None:
            note.confidence = float(confidence)
        if importance is not None:
            note.importance = float(importance)
        if sources:
            note.sources = list(dict.fromkeys(note.sources + [str(s) for s in sources]))

        self.save_note(note, touch=True)
        self._emit("knowledge.note_updated", {
            "id": note.id, "title": note.title, "version": note.version,
        })
        return note

    def delete_note(self, note_id: str, *, hard: bool = False, reason: str = "") -> bool:
        """
        Notu arşive taşır (varsayılan) veya kalıcı olarak siler.

        Args:
            note_id: Not kimliği
            hard: True ise dosyayı tamamen sil
            reason: Arşiv notu

        Returns:
            Başarılı ise True
        """
        path = self.find_path(note_id)
        if path is None:
            return False

        if hard:
            try:
                path.unlink()
            except OSError as exc:
                logger.error(f"Not silinemedi ({path}): {exc}")
                return False
            with self._lock:
                self._path_index.pop(note_id, None)
            self._emit("knowledge.note_deleted", {"id": note_id, "path": str(path), "hard": True})
            return True

        return self.archive_note(note_id, reason=reason) is not None

    def archive_note(self, note_id: str, reason: str = "") -> Optional[str]:
        """
        Notu `90_archive` altına taşır ve `archived` alanını işaretler.

        Returns:
            Yeni göreli yol veya başarısızsa None
        """
        self._require_initialized()
        assert self._root is not None

        note = self.read_note(note_id)
        if note is None:
            return None

        note.extra["archived"] = True
        if reason:
            note.extra["archive_reason"] = reason
        note.extra["archived_at"] = now_iso()

        source_path = self.find_path(note_id)
        relative = f"{ARCHIVE_DIR}/{self.note_relative_path(note).split('/', 1)[-1]}"
        target = self._resolve_within_vault(relative)
        target.parent.mkdir(parents=True, exist_ok=True)

        note.path = relative
        self._atomic_write_text(target, note.to_markdown())

        if source_path and source_path.exists() and source_path != target:
            try:
                source_path.unlink()
            except OSError as exc:
                logger.warning(f"Arşivleme sonrası kaynak silinemedi: {exc}")

        with self._lock:
            self._path_index[note.id] = relative

        self._emit("knowledge.note_archived", {
            "id": note.id, "title": note.title, "path": relative, "reason": reason,
        })
        logger.info(f"Not arşivlendi: {note.title} → {relative}")
        return relative

    # ─── Ekler ──────────────────────────────────────────────────────────

    def save_attachment(
        self,
        data: bytes,
        filename: str,
        subdir: str = "",
        note_id: str = "",
    ) -> str:
        """
        Ek dosyasını kasaya yazar.

        Args:
            data: Ham baytlar
            filename: Önerilen dosya adı
            subdir: `_attachments` altındaki alt dizin
            note_id: İlişkili not kimliği (dosya adına eklenir)

        Returns:
            Kasa göreli yolu
        """
        self._require_initialized()
        safe_name = _UNSAFE_FILENAME_RE.sub("_", Path(filename).name)
        stem = Path(safe_name).stem or "ek"
        suffix = Path(safe_name).suffix
        digest = hashlib.blake2b(data, digest_size=4).hexdigest()
        prefix = f"{note_id.replace('kn_', '')[:8]}-" if note_id else ""
        final_name = f"{prefix}{slugify(stem, 40)}-{digest}{suffix}"

        relative = f"{ATTACHMENT_DIR}/{(subdir.strip('/') + '/') if subdir else ''}{final_name}"
        target = self._resolve_within_vault(relative)
        target.parent.mkdir(parents=True, exist_ok=True)

        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_bytes(data)
        os.replace(temp, target)

        with self._lock:
            self._counter["writes"] += 1
        self._emit("knowledge.attachment_saved", {
            "path": relative, "bytes": len(data), "note_id": note_id,
        })
        return relative

    def read_attachment(self, relative: str) -> Optional[bytes]:
        """Ek dosyasını okur."""
        try:
            path = self._resolve_within_vault(relative)
        except VaultError:
            return None
        if not path.is_file():
            return None
        try:
            return path.read_bytes()
        except OSError as exc:
            logger.error(f"Ek okunamadı ({relative}): {exc}")
            return None

    def attachment_path(self, relative: str) -> Optional[Path]:
        """Ek dosyasının mutlak yolu."""
        try:
            path = self._resolve_within_vault(relative)
        except VaultError:
            return None
        return path if path.is_file() else None

    # ─── Arama Yardımcıları (dosya sistemi düzeyi) ──────────────────────

    def search_by_title(self, query: str, limit: int = 20,
                        exclude_kinds: Optional[Iterable[str]] = None
                        ) -> List[Tuple[Note, float]]:
        """
        Başlığa göre bulanık arama yapar.

        Args:
            query: Aranan başlık
            limit: Maksimum sonuç
            exclude_kinds: Bu türdeki notları atla (ör. sistemin ürettiği
                `qa` yanıt notları)

        Returns:
            (Not, benzerlik) çiftleri — benzerliğe göre azalan
        """
        if not query.strip():
            return []
        needle = query.strip().lower()
        excluded = {str(k).strip() for k in (exclude_kinds or []) if str(k).strip()}
        scored: List[Tuple[Note, float]] = []
        for note in self.iter_notes(exclude_kinds=excluded or None):
            title_lower = note.title.lower()
            if needle in title_lower:
                score = 0.75 + 0.25 * (len(needle) / max(1, len(title_lower)))
            else:
                score = title_similarity(query, note.title)
                if score < 0.35:
                    continue
            scored.append((note, round(score, 4)))
        scored.sort(key=lambda item: -item[1])
        return scored[:limit]

    def search_by_content(
        self,
        query: str,
        limit: int = 20,
        case_sensitive: bool = False,
    ) -> List[Tuple[Note, int]]:
        """
        Gövde içinde basit metin araması (indeks yoksa yedek yol).

        Returns:
            (Not, eşleşme_sayısı) çiftleri
        """
        if not query.strip():
            return []
        needle = query if case_sensitive else query.lower()
        results: List[Tuple[Note, int]] = []
        for note in self.iter_notes():
            haystack = note.body if case_sensitive else note.body.lower()
            hits = haystack.count(needle)
            if hits:
                results.append((note, hits))
        results.sort(key=lambda item: -item[1])
        return results[:limit]

    def find_duplicates(self, threshold: float = 0.85, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Birbirine çok benzeyen notları bulur.

        Returns:
            [{"a": id, "b": id, "score": float, "title_a":..., "title_b":...}]
        """
        notes = self.load_all()
        pairs: List[Dict[str, Any]] = []
        # Başlık bloklaması: aynı ilk 3 karakterle başlayanlar karşılaştırılır
        buckets: Dict[str, List[Note]] = {}
        for note in notes:
            key = note.title_slug[:3]
            buckets.setdefault(key, []).append(note)

        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a, b = bucket[i], bucket[j]
                    title_score = title_similarity(a.title, b.title)
                    if title_score < 0.4:
                        continue
                    body_score = body_similarity(a.body, b.body)
                    score = round(0.5 * title_score + 0.5 * body_score, 4)
                    if score >= threshold:
                        pairs.append({
                            "a": a.id, "b": b.id, "score": score,
                            "title_a": a.title, "title_b": b.title,
                        })
        pairs.sort(key=lambda item: -item["score"])
        return pairs[:limit]

    # ─── Bütünlük ───────────────────────────────────────────────────────

    def rebuild_path_index(self) -> int:
        """
        Kimlik → yol haritasını diskten yeniden kurar.

        Returns:
            Dizine alınan not sayısı
        """
        self._require_initialized()
        path_index: Dict[str, str] = {}
        title_index: Dict[str, List[str]] = {}
        counter = 0
        for path in self.iter_note_files(include_archive=True):
            note = self._load_path(path)
            if note is None:
                continue
            try:
                relative = str(path.relative_to(self._root)) if self._root else str(path)
            except ValueError:
                relative = str(path)
            path_index[note.id] = relative
            title_index.setdefault(note.title_slug, []).append(note.id)
            counter += 1
        with self._lock:
            self._path_index = path_index
            self._title_index = title_index
        return counter

    def integrity_check(self, fix: bool = False, max_notes: int = 2000) -> Dict[str, Any]:
        """
        Kasa bütünlüğünü denetler.

        Denetimler: bozuk frontmatter, eksik başlık, kimlik çakışması,
        kasa dışı `links`, geçersiz güven değeri, boş gövde.

        Args:
            fix: Basit sorunları düzelt ve yeniden yaz
            max_notes: Taranacak maksimum not sayısı

        Returns:
            {"ok": bool, "issues": [...], "checked": int, "fixed": int}
        """
        issues: List[Dict[str, Any]] = []
        seen_ids: Dict[str, str] = {}
        checked = 0
        fixed = 0

        for path in self.iter_note_files(include_archive=True):
            if checked >= max_notes:
                break
            checked += 1
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                issues.append({"kind": "unreadable", "path": str(path), "detail": str(exc)})
                continue

            relative = str(path.relative_to(self._root)) if self._root else str(path)
            note = Note.from_markdown(text, path=relative)
            needs_rewrite = False

            if not text.lstrip().startswith("---"):
                issues.append({"kind": "missing_frontmatter", "path": relative, "detail": "Frontmatter yok"})
                needs_rewrite = True
            if not note.title or note.title == note.id:
                issues.append({"kind": "missing_title", "path": relative, "detail": "Başlık yok/eksik"})
                needs_rewrite = True
            if not note.body.strip():
                issues.append({"kind": "empty_body", "path": relative, "detail": "Gövde boş"})
            if not (0.0 <= note.confidence <= 1.0):
                issues.append({"kind": "invalid_confidence", "path": relative,
                               "detail": f"confidence={note.confidence}"})
                note.clamp_scores()
                needs_rewrite = True

            previous = seen_ids.get(note.id)
            if previous:
                issues.append({
                    "kind": "duplicate_id", "path": relative,
                    "detail": f"Kimlik {note.id} ayrıca {previous} içinde",
                })
                new_id = new_note_id()
                note.id = new_id
                issues.append({"kind": "duplicate_id_fixed", "path": relative,
                               "detail": f"Yeni kimlik: {new_id}"})
                needs_rewrite = True
            seen_ids[note.id] = relative

            for link in note.links:
                if link not in self._path_index and not self.find_path(link):
                    issues.append({"kind": "unknown_link", "path": relative,
                                   "detail": f"Bilinmeyen bağlantı: {link}"})

            if fix and needs_rewrite:
                note.path = relative
                self._atomic_write_text(path, note.to_markdown())
                fixed += 1

        if fix:
            self.rebuild_path_index()

        ok = not any(issue["kind"] in {"unreadable", "missing_frontmatter", "duplicate_id"} for issue in issues)
        result = {"ok": ok, "issues": issues, "checked": checked, "fixed": fixed}
        self._emit("knowledge.integrity_checked", {
            "ok": ok, "issues": len(issues), "checked": checked, "fixed": fixed,
        })
        return result

    def quarantine(self, path: str, reason: str = "") -> Optional[str]:
        """
        Bozuk dosyayı `90_archive/_corrupt` altına taşır.

        Returns:
            Yeni göreli yol veya başarısızsa None
        """
        source = self.find_path(path)
        if source is None or not source.is_file():
            return None
        try:
            relative = source.relative_to(self._root) if self._root else Path(source.name)
        except ValueError:
            relative = Path(source.name)

        target = self._resolve_within_vault(f"{CORRUPT_DIR}/{relative.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(source), str(target))
        except OSError as exc:
            logger.error(f"Karantinaya alma başarısız ({source}): {exc}")
            return None

        if reason:
            try:
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n\n<!-- KARANTİNA: {now_iso()} — {reason} -->\n")
            except OSError:
                pass

        with self._lock:
            self._counter["quarantined"] += 1
        self._emit("knowledge.note_quarantined", {
            "from": str(relative), "to": f"{CORRUPT_DIR}/{relative.name}", "reason": reason,
        })
        logger.warning(f"Not karantinaya alındı: {relative} ({reason})")
        return f"{CORRUPT_DIR}/{relative.name}"

    # ─── İstatistik ─────────────────────────────────────────────────────

    def vault_stats(self, max_notes: int = 5000) -> VaultStats:
        """
        Kasa geneli istatistikleri hesaplar.

        Args:
            max_notes: Taranacak maksimum not sayısı
        """
        stats = VaultStats()
        confidences: List[float] = []
        importances: List[float] = []
        known_ids: set = set()

        notes = self.load_all(limit=max_notes, include_archive=False)
        for note in notes:
            known_ids.add(note.id)

        for note in notes:
            stats.total_notes += 1
            stats.total_words += note.word_count
            try:
                stats.total_bytes += self.note_path(note).stat().st_size
            except OSError:
                pass
            stats.by_category[note.category] = stats.by_category.get(note.category, 0) + 1
            stats.by_kind[note.kind] = stats.by_kind.get(note.kind, 0) + 1
            for tag in note.tags:
                stats.tag_histogram[tag] = stats.tag_histogram.get(tag, 0) + 1
            confidences.append(note.confidence)
            importances.append(note.importance)
            if note.is_low_confidence:
                stats.low_confidence += 1
            if note.is_shallow:
                stats.shallow += 1
            if note.has_todos:
                stats.with_todos += 1
            for link in note.links:
                if link not in known_ids:
                    stats.orphan_links += 1
            stamp = note.updated_ts
            stats.oldest = stamp if stats.oldest == 0 else min(stats.oldest, stamp)
            stats.newest = max(stats.newest, stamp)

        if confidences:
            stats.mean_confidence = sum(confidences) / len(confidences)
        if importances:
            stats.mean_importance = sum(importances) / len(importances)
        try:
            stats.attachments = sum(1 for _ in self.attachment_dir.rglob("*") if _.is_file())
        except OSError:
            stats.attachments = 0
        return stats

    # ─── Bakım ──────────────────────────────────────────────────────────

    def export_graph_json(self, path: Optional[str] = None) -> str:
        """
        Bağlantı grafını JSON olarak dışa aktarır (görselleştirme için).

        Returns:
            Yazılan dosyanın yolu
        """
        import json

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, str]] = []
        ids: set = set()
        for note in self.iter_notes():
            ids.add(note.id)

        for note in self.iter_notes():
            nodes.append({
                "id": note.id,
                "title": note.title,
                "kind": note.kind,
                "category": note.category,
                "tags": list(note.tags),
                "confidence": note.confidence,
                "words": note.word_count,
                "path": note.path,
            })
            for link in note.links:
                edges.append({"source": note.id, "target": link,
                              "resolved": "true" if link in ids else "false"})

        payload = {
            "generated_at": now_iso(),
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }
        target = Path(path) if path else (self.index_dir / "graph.json")
        self._atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info(f"Graf dışa aktarıldı: {target} ({len(nodes)} düğüm, {len(edges)} kenar)")
        return str(target)

    def rebuild_index_hint(self) -> str:
        """İndeksin yeniden kurulması gerektiğini bildirir ve yolunu döndürür."""
        self._emit("knowledge.index_stale", {"root": str(self._root)})
        return str(self.index_dir / "neurocore.db")

    def vacuum(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Kasayı temizler: boş dizinler, `.tmp` artıkları, eski karantinalar.

        Args:
            dry_run: True ise yalnızca raporla, silme

        Returns:
            {"empty_dirs": [...], "temp_files": [...], "removed": int}
        """
        empty_dirs: List[str] = []
        temp_files: List[str] = []
        removed = 0

        for path in sorted(self.root.rglob("*"), reverse=True):
            try:
                if path.is_dir() and path not in (self.root,) and not any(path.iterdir()):
                    empty_dirs.append(str(path.relative_to(self.root)))
                    if not dry_run:
                        path.rmdir()
                        removed += 1
                elif path.is_file() and path.suffix in (".tmp", ".bak", ".old"):
                    temp_files.append(str(path.relative_to(self.root)))
                    if not dry_run:
                        path.unlink()
                        removed += 1
            except OSError:
                continue

        result = {"empty_dirs": empty_dirs, "temp_files": temp_files,
                  "removed": removed, "dry_run": dry_run}
        if not dry_run:
            self._emit("knowledge.vacuumed", result)
        return result

    # ─── İç Yardımcılar ─────────────────────────────────────────────────

    def _atomic_write_text(self, path: Path, text: str) -> None:
        """Metni geçici dosyaya yazıp atomik olarak yerine koyar."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        try:
            with open(temp, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        except OSError as exc:
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:
                pass
            self._counter["errors"] += 1
            raise VaultError(f"Dosya yazılamadı ({path}): {exc}") from exc

    def _write_text(self, path: Path, text: str) -> None:
        """Basit metin yazımı (dizin düzeni kurulumu sırasında)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla (EventBus yoksa sessizce geç)."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover - olay hataları kritik değil
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __repr__(self) -> str:  # pragma: no cover
        root = self._root if self._root else "<kurulmadı>"
        return f"<KnowledgeVault root={root} notes={len(self._path_index)}>"


__all__ = [
    "KnowledgeVault",
    "VaultError",
    "VaultNotInitialized",
    "VaultStats",
    "CATEGORY_DIRS",
    "NOTE_EXTENSIONS",
    "ARCHIVE_DIR",
    "INDEX_DIR",
    "ATTACHMENT_DIR",
    "CORRUPT_DIR",
]
