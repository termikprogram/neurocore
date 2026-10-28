"""
NeuroCore Bilgi İndeksi
========================
Kasayı hızlı aranabilir hale getiren SQLite tabanlı indeks.

Sağlanan yetenekler:
    * **Meta tablo** — not üst verisi, sıralama ve filtreleme için
    * **FTS5 tam metin indeksi** — BM25 sıralaması (yoksa LIKE yedeği)
    * **Vektör saklama** — gömme vektörleri BLOB (float32) olarak
    * **Kenar tablosu** — `[[wiki]]` bağlantıları ve yazarlık ilişkileri
    * **Artımlı güncelleme** — içerik parmak izi değişmediyse yeniden indeksleme yok
    * **Kurtarma** — şema sürümü uyuşmazlığında indeks güvenle yeniden kurulur

Tasarım notları:
    * Tek yazar (single-writer) varsayımı: SQLite bağlantısı kilit ile korunur.
    * WAL kipi açılır — okuma ve yazma birbirini bloklamaz.
    * `check_same_thread=False` ile motorun thread'lerinden erişim güvenlidir.
    * FTS5 yoksa sistem çalışmaya devam eder, yalnızca sıralama kalitesi düşer.
"""

from __future__ import annotations

import array
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from neurocore.knowledge.note import (
    Note,
    content_hash,
    fold_turkish,
    now_iso,
    strip_markdown,
    utc_timestamp,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("knowledge.index")


# ─── Sabitler ───────────────────────────────────────────────────────────────

SCHEMA_VERSION = 4
DEFAULT_DB_NAME = "neurocore.db"

#: FTS5 sorgularında anlam taşıyan karakterler (kullanıcı girdisi temizlenir)
_FTS_UNSAFE_RE = re.compile(r'["\'^*:()\[\]{}~\\/]')
_WORD_RE = re.compile(r"[\wÇĞİÖŞÜçğıöşü]+", re.UNICODE)


def fold_for_search(text: str) -> str:
    """
    Arama için metni Türkçe-farkındalıklı biçimde sadeleştirir.

    FTS5 `unicode61` belirteci yalnızca Latin-1 aksanlarını katlar; Türkçe
    `ı/İ/ş/ğ/ç/ö/ü` harflerini katlamaz. Bu yüzden indekslenen metin ve sorgu
    **aynı** fonksiyondan geçirilir: `sıcaklık` ↔ `sicaklik` eşleşir.
    """
    if not text:
        return ""
    return fold_turkish(str(text).lower())


class IndexError_(Exception):
    """İndeks işlem hatası."""


class DatabaseUnavailable(IndexError_):
    """SQLite kullanılamıyor."""


# ─── Vektör Yardımcıları ────────────────────────────────────────────────────

def vector_to_blob(vector: Sequence[float]) -> bytes:
    """Float listesini float32 BLOB'a çevirir (küçük ve hızlı)."""
    return array.array("f", [float(v) for v in vector]).tobytes()


def blob_to_vector(blob: Optional[bytes]) -> List[float]:
    """BLOB'u float listesine çevirir."""
    if not blob:
        return []
    arr = array.array("f")
    try:
        arr.frombytes(blob)
    except (ValueError, TypeError):
        return []
    return arr.tolist()


# ─── İndeks Sınıfı ──────────────────────────────────────────────────────────

class KnowledgeIndex:
    """
    Kasanın SQLite indeksi.

    Kullanım:
        index = KnowledgeIndex()
        index.initialize(config, event_bus)

        index.rebuild(vault)                      # ilk kurulum / tam yenileme
        index.upsert_note(note)                   # artımlı güncelleme
        rows = index.search_fts("gpu termal", 10)
        rows = index.similar_notes(vector, 5)

        index.close()
    """

    name = "knowledge.index"

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Args:
            db_path: Veritabanı yolu (None ise initialize'da kasadan alınır)
        """
        self._db_path: Optional[Path] = db_path and Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._fts5 = False
        self._vec_cache: Optional[Tuple[List[str], List[List[float]]]] = None
        self._vec_cache_stamp = 0.0
        self._counter = {
            "upserts": 0, "deletes": 0, "searches": 0,
            "rebuilds": 0, "errors": 0, "cache_hits": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None, db_path: Optional[str] = None) -> None:
        """
        Veritabanını aç ve şemayı kur.

        Args:
            config: SystemConfig
            event_bus: EventBus
            db_path: Açık veritabanı yolu (kasadan bağımsız kullanım için)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus

        if db_path:
            self._db_path = Path(db_path)
        if self._db_path is None:
            base = None
            if config is not None:
                vault_path = getattr(getattr(config, "knowledge", None), "vault_path", None)
                if vault_path:
                    base = Path(vault_path) / "_index"
                else:
                    base = Path(getattr(config, "base_path", os.getcwd())) / "knowledge_base" / "_index"
            else:
                base = Path(os.getcwd()) / "knowledge_base" / "_index"
            self._db_path = base / DEFAULT_DB_NAME

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._open()
        self._initialized = True
        logger.info(f"Bilgi indeksi açıldı: {self._db_path} (FTS5: {'var' if self._fts5 else 'yok'})")
        self._emit("knowledge.index_ready", {
            "path": str(self._db_path), "fts5": self._fts5, **self.counts(),
        })

    def _open(self) -> None:
        """SQLite bağlantısını aç ve şemayı doğrula/kur."""
        assert self._db_path is not None
        try:
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30.0,
                isolation_level=None,   # otomatik commit; işlemleri elle yönetiyoruz
            )
        except sqlite3.Error as exc:
            raise DatabaseUnavailable(f"SQLite açılamadı ({self._db_path}): {exc}") from exc

        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-16000")   # ~16 MB
        except sqlite3.Error as exc:
            logger.warning(f"SQLite PRAGMA ayarları uygulanamadı: {exc}")

        self._conn = conn
        self._fts5 = self._detect_fts5()
        existing = self._schema_version()

        if existing != SCHEMA_VERSION:
            if existing == 0:
                self._create_schema()
            else:
                logger.warning(
                    f"İndeks şema sürümü uyuşmuyor (bulunan: {existing}, beklenen: {SCHEMA_VERSION}) "
                    "— indeks yeniden kurulacak"
                )
                self._drop_schema()
                self._create_schema()
        else:
            self._create_schema(if_not_exists=True)

    def _detect_fts5(self) -> bool:
        """FTS5 modülü kullanılabilir mi?"""
        if self._conn is None:
            return False
        try:
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
            self._conn.execute("DROP TABLE IF EXISTS _fts5_probe")
            return True
        except sqlite3.Error:
            return False

    def _schema_version(self) -> int:
        """Mevcut şema sürümünü okur (yoksa 0)."""
        if self._conn is None:
            return 0
        try:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            return int(row["value"]) if row else 0
        except sqlite3.Error:
            return 0

    def _create_schema(self, if_not_exists: bool = False) -> None:
        """Tabloları, indeksleri ve FTS tablosunu oluşturur."""
        if self._conn is None:
            raise DatabaseUnavailable("Veritabanı bağlantısı yok")
        ine = "IF NOT EXISTS " if if_not_exists else ""
        statements = [
            f"""CREATE TABLE {ine}meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )""",
            f"""CREATE TABLE {ine}notes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL DEFAULT '',
                    plain TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL DEFAULT 'concept',
                    category TEXT NOT NULL DEFAULT 'learned',
                    tags TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    importance REAL NOT NULL DEFAULT 0.5,
                    created REAL NOT NULL DEFAULT 0,
                    updated REAL NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    word_count INTEGER NOT NULL DEFAULT 0,
                    path TEXT NOT NULL DEFAULT '',
                    sources TEXT NOT NULL DEFAULT '',
                    checksum TEXT NOT NULL DEFAULT '',
                    indexed_at REAL NOT NULL DEFAULT 0,
                    embedding_model TEXT NOT NULL DEFAULT '',
                    embedding BLOB
                )""",
            "CREATE INDEX IF NOT EXISTS idx_notes_kind ON notes(kind)",
            "CREATE INDEX IF NOT EXISTS idx_notes_category ON notes(category)",
            "CREATE INDEX IF NOT EXISTS idx_notes_updated ON notes(updated DESC)",
            "CREATE INDEX IF NOT EXISTS idx_notes_confidence ON notes(confidence DESC)",
            "CREATE INDEX IF NOT EXISTS idx_notes_checksum ON notes(checksum)",
            f"""CREATE TABLE {ine}tags (
                    note_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (note_id, tag)
                )""",
            "CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)",
            f"""CREATE TABLE {ine}edges (
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    relation TEXT NOT NULL DEFAULT 'links',
                    weight REAL NOT NULL DEFAULT 1.0,
                    reason TEXT NOT NULL DEFAULT '',
                    unresolved INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (src, dst, relation)
                )""",
            "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)",
            "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)",
            f"""CREATE TABLE {ine}terms (
                    term TEXT NOT NULL,
                    note_id TEXT NOT NULL,
                    tf INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (term, note_id)
                )""",
            "CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term)",
            f"""CREATE TABLE {ine}history (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0,
                    checksum TEXT NOT NULL DEFAULT '',
                    detail TEXT NOT NULL DEFAULT '',
                    at REAL NOT NULL DEFAULT 0
                )""",
            "CREATE INDEX IF NOT EXISTS idx_history_note ON history(note_id, at DESC)",
        ]

        if self._fts5:
            # Not: content='' (contentless) DEĞİL — contentless FTS5 tablolarında
            # DELETE/UPDATE yapılamaz, bu da not güncellemeyi imkânsız kılar.
            # Depolama maliyeti karşılığında tam güncellenebilirlik tercih edilir.
            statements.append(
                f"""CREATE VIRTUAL TABLE {ine}notes_fts USING fts5(
                        title, body, tags,
                        tokenize='unicode61 remove_diacritics 0'
                    )"""
            )

        with self._lock:
            try:
                for statement in statements:
                    self._conn.execute(statement)
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION)),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    ("created_at", now_iso()),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    ("fts5", "1" if self._fts5 else "0"),
                )
            except sqlite3.Error as exc:
                self._counter["errors"] += 1
                raise IndexError_(f"Şema oluşturulamadı: {exc}") from exc

    def _drop_schema(self) -> None:
        """Tüm indeks tablolarını siler (yeniden kurulum için)."""
        if self._conn is None:
            return
        tables = ["notes_fts", "terms", "edges", "tags", "history", "notes", "meta"]
        with self._lock:
            for table in tables:
                try:
                    self._conn.execute(f"DROP TABLE IF EXISTS {table}")
                except sqlite3.Error:
                    pass

    def close(self) -> None:
        """Bağlantıyı kapat (idempotent)."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.execute("PRAGMA optimize")
                except sqlite3.Error:
                    pass
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None
        self._initialized = False
        logger.info("Bilgi indeksi kapatıldı")

    def shutdown(self) -> None:
        """`close` takma adı (katman sözleşmesi)."""
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Ham bağlantı (ileri düzey kullanım)."""
        if self._conn is None:
            raise DatabaseUnavailable("İndeks açık değil — initialize() çağırın")
        return self._conn

    @property
    def path(self) -> Optional[Path]:
        """Veritabanı dosya yolu."""
        return self._db_path

    @property
    def has_fts5(self) -> bool:
        """FTS5 kullanılabilir mi?"""
        return self._fts5

    # ─── Not İndeksleme ─────────────────────────────────────────────────

    def upsert_note(self, note: Note, *, embedding: Optional[Sequence[float]] = None,
                    emit: bool = True) -> bool:
        """
        Notu indekse ekler veya günceller.

        İçerik parmak izi değişmediyse ve gömme verilmediyse **yazma yapılmaz**.

        Args:
            note: İndekslenecek not
            embedding: Gömme vektörü (None ise notun kendi gömme alanı kullanılır)
            emit: Olay yayınla

        Returns:
            Gerçekten yazıldıysa True (değişiklik yoksa False)
        """
        self._require_conn()
        assert self._conn is not None

        checksum = note.content_fingerprint()
        vector = embedding if embedding is not None else note.embedding
        blob = vector_to_blob(vector) if vector else None

        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT checksum, embedding IS NOT NULL AS has_vec FROM notes WHERE id = ?",
                    (note.id,),
                ).fetchone()
            except sqlite3.Error as exc:
                self._counter["errors"] += 1
                logger.error(f"Not sorgulanamadı ({note.id}): {exc}")
                return False

            if row and row["checksum"] == checksum:
                if not (vector and not row["has_vec"]):
                    return False

            plain = strip_markdown(note.body)
            tags_text = " ".join(note.tags)
            sources_text = "|".join(note.sources)
            # FTS'e Türkçe katlanmış metin yazılır (bkz. fold_for_search)
            fts_title = fold_for_search(note.title)
            fts_body = fold_for_search(plain)
            fts_tags = fold_for_search(tags_text)

            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    """
                    INSERT INTO notes (
                        id, title, body, plain, kind, category, tags, confidence,
                        importance, created, updated, version, word_count, path,
                        sources, checksum, indexed_at, embedding_model, embedding
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET
                        title=excluded.title, body=excluded.body, plain=excluded.plain,
                        kind=excluded.kind, category=excluded.category, tags=excluded.tags,
                        confidence=excluded.confidence, importance=excluded.importance,
                        created=excluded.created, updated=excluded.updated,
                        version=excluded.version, word_count=excluded.word_count,
                        path=excluded.path, sources=excluded.sources,
                        checksum=excluded.checksum, indexed_at=excluded.indexed_at,
                        embedding_model=excluded.embedding_model,
                        embedding=COALESCE(excluded.embedding, notes.embedding)
                    """,
                    (
                        note.id, note.title, note.body, plain, note.kind, note.category,
                        tags_text, float(note.confidence), float(note.importance),
                        note.created_ts, note.updated_ts, int(note.version),
                        note.word_count, note.path, sources_text, checksum,
                        utc_timestamp(), note.embedding_model,
                        blob if blob is not None else None,
                    ),
                )

                # Etiketler
                self._conn.execute("DELETE FROM tags WHERE note_id = ?", (note.id,))
                if note.tags:
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO tags(note_id, tag) VALUES (?, ?)",
                        [(note.id, tag) for tag in note.tags],
                    )

                # Terim frekansları (basit anahtar kelime indeksi)
                self._conn.execute("DELETE FROM terms WHERE note_id = ?", (note.id,))
                term_counts: Dict[str, int] = {}
                for word in _WORD_RE.findall(plain.lower()):
                    if len(word) < 3:
                        continue
                    term_counts[word] = term_counts.get(word, 0) + 1
                if term_counts:
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO terms(term, note_id, tf) VALUES (?, ?, ?)",
                        [(term, note.id, count) for term, count in term_counts.items()],
                    )

                # Bağlantı kenarları (bu notun giden bağlantıları)
                self._conn.execute(
                    "DELETE FROM edges WHERE src = ? AND relation = 'links'", (note.id,)
                )
                if note.links:
                    self._conn.executemany(
                        """INSERT OR REPLACE INTO edges
                           (src, dst, relation, weight, reason, unresolved, updated_at)
                           VALUES (?, ?, 'links', 1.0, '', 0, ?)""",
                        [(note.id, link, utc_timestamp()) for link in note.links],
                    )

                # FTS (contentless olmadığı için UPDATE/INSERT ayrımı gerekir)
                if self._fts5:
                    existing_fts = self._conn.execute(
                        "SELECT rowid FROM notes_fts WHERE rowid = ("
                        "SELECT rowid FROM notes WHERE id = ?)", (note.id,)
                    ).fetchone()
                    if existing_fts:
                        self._conn.execute(
                            "UPDATE notes_fts SET title = ?, body = ?, tags = ? WHERE rowid = ?",
                            (fts_title, fts_body, fts_tags, existing_fts["rowid"]),
                        )
                    else:
                        row = self._conn.execute(
                            "SELECT rowid FROM notes WHERE id = ?", (note.id,)
                        ).fetchone()
                        if row:
                            self._conn.execute(
                                "INSERT INTO notes_fts(rowid, title, body, tags) VALUES (?, ?, ?, ?)",
                                (row["rowid"], fts_title, fts_body, fts_tags),
                            )

                self._conn.execute(
                    """INSERT INTO history(note_id, action, version, checksum, detail, at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (note.id, "upsert", int(note.version), checksum,
                     f"words={note.word_count}", utc_timestamp()),
                )
                self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                self._counter["errors"] += 1
                logger.error(f"Not indekslenemedi ({note.id}): {exc}")
                return False

        self._counter["upserts"] += 1
        self._vec_cache = None
        if emit:
            self._emit("knowledge.note_indexed", {
                "id": note.id, "title": note.title, "words": note.word_count,
            })
        return True

    def upsert_many(self, notes: Iterable[Note], batch_size: int = 200) -> int:
        """
        Çok sayıda notu indeksler.

        Args:
            notes: Not yineleyicisi
            batch_size: Kaç notta bir ilerleme loglanacak

        Returns:
            Yazılan not sayısı
        """
        written = 0
        for index, note in enumerate(notes, start=1):
            if self.upsert_note(note, emit=False):
                written += 1
            if index % batch_size == 0:
                logger.info(f"İndeksleniyor… {index} not işlendi ({written} yazıldı)")
        return written

    def delete_note(self, note_id: str) -> bool:
        """Notu indeksten siler."""
        self._require_conn()
        assert self._conn is not None
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if self._fts5:
                    row = self._conn.execute("SELECT rowid FROM notes WHERE id = ?", (note_id,)).fetchone()
                    if row:
                        self._conn.execute("DELETE FROM notes_fts WHERE rowid = ?", (row["rowid"],))
                self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
                self._conn.execute("DELETE FROM tags WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM terms WHERE note_id = ?", (note_id,))
                self._conn.execute("DELETE FROM edges WHERE src = ? OR dst = ?", (note_id, note_id))
                self._conn.execute(
                    "INSERT INTO history(note_id, action, detail, at) VALUES (?, 'delete', '', ?)",
                    (note_id, utc_timestamp()),
                )
                self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                self._counter["errors"] += 1
                logger.error(f"Not indeksten silinemedi ({note_id}): {exc}")
                return False
        self._counter["deletes"] += 1
        self._vec_cache = None
        self._emit("knowledge.note_deindexed", {"id": note_id})
        return True

    def rebuild(self, vault: Any, *, embeddings: Optional[Dict[str, Sequence[float]]] = None,
                prune_missing: bool = True) -> Dict[str, Any]:
        """
        İndeksi kasadan tamamen yeniden kurar.

        Args:
            vault: `KnowledgeVault` örneği (veya `iter_notes`/`load_all` sunan nesne)
            embeddings: {note_id: vektör} — verilirse vektörler de yazılır
            prune_missing: Diskte olmayan notları indeksten sil

        Returns:
            {"scanned": int, "written": int, "removed": int, "duration_ms": float}
        """
        self._require_conn()
        start = time.perf_counter()
        scanned = written = 0
        seen: Set[str] = set()

        if hasattr(vault, "iter_notes"):
            iterator = vault.iter_notes()
        elif hasattr(vault, "load_all"):
            iterator = iter(vault.load_all())
        else:
            raise IndexError_("rebuild için iter_notes/load_all sunan bir kasa gerekir")

        for note in iterator:
            scanned += 1
            seen.add(note.id)
            vector = (embeddings or {}).get(note.id)
            if self.upsert_note(note, embedding=vector, emit=False):
                written += 1

        removed = 0
        if prune_missing:
            for note_id in self.all_note_ids():
                if note_id not in seen:
                    if self.delete_note(note_id):
                        removed += 1

        duration = (time.perf_counter() - start) * 1000
        self._counter["rebuilds"] += 1
        self._vec_cache = None
        result = {
            "scanned": scanned, "written": written, "removed": removed,
            "duration_ms": round(duration, 1),
        }
        self._emit("knowledge.index_rebuilt", result)
        logger.info(
            f"İndeks yeniden kuruldu: {scanned} tarandı, {written} yazıldı, "
            f"{removed} silindi ({duration:.0f}ms)"
        )
        return result

    # ─── Arama ──────────────────────────────────────────────────────────

    @staticmethod
    def sanitize_query(query: str, use_or: bool = True) -> str:
        """
        Kullanıcı sorgusunu FTS5 güvenli biçime getirir.

        Operatörler (`* : ^ " ( )`) temizlenir, metin Türkçe katlanır ve
        kelimeler önek aramasına çevrilir: `GPU Sıcaklık` → `gpu* sicaklik*`

        Args:
            use_or: True ise kelimeler OR ile bağlanır (geri çağırma odaklı;
                kesinlik `_rerank` aşamasında kapsam puanıyla geri kazanılır).
                False ise FTS5 örtük AND semantiği kullanılır.
        """
        if not query:
            return ""
        cleaned = _FTS_UNSAFE_RE.sub(" ", str(query))
        folded = fold_for_search(cleaned)
        words = [w for w in _WORD_RE.findall(folded) if len(w) >= 2]
        if not words:
            return ""
        terms = [f"{word}*" for word in words[:16]]
        if use_or and len(terms) > 1:
            return "(" + " OR ".join(terms) + ")"
        return " ".join(terms)

    def search_fts(self, query: str, limit: int = 20, **filters: Any) -> List[Dict[str, Any]]:
        """
        Tam metin araması (FTS5 + kapsam yeniden sıralaması).

        Akış:
            1. Sorgu Türkçe katlanır ve OR-prefix FTS ifadesine çevrilir.
            2. FTS5 geniş bir aday kümesi döndürür (geri çağırma).
            3. `_rerank` adayları kelime kapsamı, başlık isabeti, BM25 ve
               not güvenine göre yeniden puanlar (kesinlik).

        Args:
            query: Arama sorgusu
            limit: Maksimum sonuç
            **filters: `kind`, `category`, `tags`, `min_confidence`,
                `max_confidence`, `exclude_ids` filtreleri

        Returns:
            [{"id","title","kind","category","tags","confidence","updated",
              "score","excerpt","path","coverage","rank"}]
        """
        self._require_conn()
        self._counter["searches"] += 1
        match = self.sanitize_query(query)
        if not match:
            return []

        candidate_limit = max(int(limit) * 8, 60)
        if self._fts5:
            hits = self._search_fts5(match, query, candidate_limit, filters)
        else:
            return self._search_like(query, limit, filters)

        return self._rerank(hits, query, limit)

    @staticmethod
    def _rerank(hits: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Arama adaylarını gerçek kelime kapsamına göre yeniden puanlar.

        Skor bileşenleri:
            * ``coverage``  — sorgu kelimelerinin kaçta kaçı notta geçiyor (0..1, ağırlık 0.55)
            * ``phrase``    — sorgu, gövdede bitişik ifade olarak geçiyor mu (bonus 0.15)
            * ``title``     — başlıkta geçen kelime oranı (ağırlık 0.20)
            * ``rank``      — BM25 sıralamasından gelen sinyal (ağırlık 0.10)
            * ``confidence``— notun kendi güven değeri (bonus 0..0.10)
        """
        if not hits:
            return []

        words = [w for w in _WORD_RE.findall(fold_for_search(query)) if len(w) >= 2]
        if not words:
            return hits[:limit]
        phrase = " ".join(words)
        total = len(words)
        max_rank = max((float(hit.get("rank", 0.0)) for hit in hits), default=0.0) or 1.0

        for hit in hits:
            full_text = fold_for_search(
                f"{hit.get('title', '')} {hit.get('_fulltext', '') or hit.get('excerpt', '')}"
            )
            title_text = fold_for_search(hit.get("title", ""))

            matched = sum(1 for word in words if word in full_text)
            coverage = matched / total
            title_hits = sum(1 for word in words if word in title_text)
            title_score = title_hits / total
            phrase_bonus = 0.15 if len(words) > 1 and phrase in full_text else 0.0
            rank_score = min(1.0, float(hit.get("rank", 0.0)) / max_rank)
            confidence = float(hit.get("confidence", 0.5))

            score = (
                0.55 * coverage
                + 0.20 * title_score
                + 0.10 * rank_score
                + phrase_bonus
                + 0.10 * confidence
            )
            hit["coverage"] = round(coverage, 4)
            hit["score"] = round(min(1.0, score), 4)

        hits.sort(key=lambda item: (-item["score"], -float(item.get("rank", 0.0))))
        return [_clean_hit(hit) for hit in hits[:limit]]

    def _search_fts5(self, match: str, raw_query: str, limit: int,
                     filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        FTS5 + BM25 araması.

        `prefix` uzantı işlevi Türkçe çekim eklerini yakalar
        (`sıcaklık` → `sicakligi`), bu yüzden sorgudan önce yüklenir.
        """
        assert self._conn is not None
        where, params = self._filter_clause(filters, alias="n")
        sql = f"""
            SELECT n.id, n.title, n.plain, n.kind, n.category, n.tags,
                   n.confidence, n.importance, n.updated, n.word_count, n.path,
                   bm25(notes_fts, 5.0, 1.0, 2.0) AS bm
            FROM notes_fts
            JOIN notes n ON n.rowid = notes_fts.rowid
            WHERE notes_fts MATCH ? {where}
            ORDER BY bm
            LIMIT ?
        """
        try:
            try:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS temp.prefix USING prefix='2 3 4 5 6 7 8'"
                )
            except sqlite3.Error as exc:
                logger.debug(f"prefix uzantısı yüklenemedi (önemsiz): {exc}")
            rows = self._conn.execute(sql, (match, *params, int(limit))).fetchall()
        except sqlite3.Error as exc:
            logger.warning(f"FTS5 araması başarısız ({match}): {exc} — LIKE yedeğine geçiliyor")
            return self._search_like(raw_query, limit, filters)

        results: List[Dict[str, Any]] = []
        bm_values = [float(row["bm"]) for row in rows if row["bm"] is not None]
        best_bm = min(bm_values) if bm_values else 0.0
        for row in rows:
            bm = float(row["bm"]) if row["bm"] is not None else 0.0
            # bm25 negatif döner; en iyi (en negatif) değeri 1.0'a normalize et.
            # İşaret farkını tolere etmek için mutlak değer oranı kullanılır.
            if bm != 0.0 and best_bm != 0.0:
                rank_score = min(abs(best_bm), abs(bm)) / max(abs(best_bm), abs(bm))
            else:
                rank_score = 0.0
            hit = self._row_to_hit(row, 0.0, raw_query)
            hit["rank"] = round(max(0.0, min(1.0, rank_score)), 4)
            hit["bm25"] = round(bm, 4)
            # Yeniden sıralama için tam metin (yanıtta taşınmaz)
            hit["_fulltext"] = row["plain"] or ""
            results.append(hit)
        return results

    def _search_like(self, query: str, limit: int, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """FTS5 yoksa kelime frekansına dayalı yedek arama."""
        assert self._conn is not None
        words = [w for w in _WORD_RE.findall(fold_for_search(query)) if len(w) >= 2][:10]
        if not words:
            return []
        where, params = self._filter_clause(filters, alias="n")

        # Terim tablosundan adayları bul (hızlı)
        placeholders = ",".join("?" for _ in words)
        candidate_sql = f"""
            SELECT note_id, SUM(tf) AS hits FROM terms
            WHERE term IN ({placeholders}) GROUP BY note_id
        """
        try:
            candidates = {
                row["note_id"]: row["hits"]
                for row in self._conn.execute(candidate_sql, words).fetchall()
            }
        except sqlite3.Error:
            candidates = {}

        if candidates:
            ids = list(candidates.keys())[: max(limit * 5, 50)]
            id_placeholders = ",".join("?" for _ in ids)
            sql = f"""
                SELECT n.id, n.title, n.plain, n.kind, n.category, n.tags,
                       n.confidence, n.importance, n.updated, n.word_count, n.path
                FROM notes n
                WHERE n.id IN ({id_placeholders}) {where}
            """
            try:
                rows = self._conn.execute(sql, (*ids, *params)).fetchall()
            except sqlite3.Error as exc:
                logger.error(f"Yedek arama başarısız: {exc}")
                return []
        else:
            sql = f"""
                SELECT n.id, n.title, n.plain, n.kind, n.category, n.tags,
                       n.confidence, n.importance, n.updated, n.word_count, n.path
                FROM notes n WHERE 1=1 {where} LIMIT 2000
            """
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.Error as exc:
                logger.error(f"Yedek arama başarısız: {exc}")
                return []

        hits: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            text = fold_for_search(f"{row['title']} {row['plain']} {row['tags']}")
            matched = sum(1 for word in words if word in text)
            if matched == 0:
                continue
            density = sum(text.count(word) for word in words) / max(1, len(text.split()))
            score = min(1.0, 0.35 + 0.5 * (matched / len(words)) + min(0.15, density * 5))
            hit = self._row_to_hit(row, score, query)
            hit["rank"] = round(min(1.0, matched / len(words)), 4)
            hit["coverage"] = round(matched / len(words), 4)
            hit["_fulltext"] = row["plain"] or ""
            hits.append((score, hit))

        hits.sort(key=lambda item: -item[0])
        return [_clean_hit(hit) for _, hit in hits[:limit]]

    def _filter_clause(self, filters: Dict[str, Any], alias: str = "n") -> Tuple[str, List[Any]]:
        """Filtre sözlüğünü SQL WHERE parçasına çevirir."""
        clauses: List[str] = []
        params: List[Any] = []
        if not filters:
            return "", params

        if filters.get("kind"):
            clauses.append(f"{alias}.kind = ?")
            params.append(str(filters["kind"]))
        if filters.get("exclude_kinds"):
            excluded = [str(k) for k in filters["exclude_kinds"]]
            placeholders = ",".join("?" for _ in excluded)
            clauses.append(f"{alias}.kind NOT IN ({placeholders})")
            params.extend(excluded)
        if filters.get("category"):
            clauses.append(f"{alias}.category = ?")
            params.append(str(filters["category"]))
        if filters.get("min_confidence") is not None:
            clauses.append(f"{alias}.confidence >= ?")
            params.append(float(filters["min_confidence"]))
        if filters.get("max_confidence") is not None:
            clauses.append(f"{alias}.confidence <= ?")
            params.append(float(filters["max_confidence"]))
        if filters.get("tags"):
            tags = [str(t).lower() for t in filters["tags"]]
            placeholders = ",".join("?" for _ in tags)
            clauses.append(
                f"EXISTS (SELECT 1 FROM tags t WHERE t.note_id = {alias}.id "
                f"AND t.tag IN ({placeholders}))"
            )
            params.extend(tags)
        if filters.get("exclude_ids"):
            excluded = [str(i) for i in filters["exclude_ids"]]
            placeholders = ",".join("?" for _ in excluded)
            clauses.append(f"{alias}.id NOT IN ({placeholders})")
            params.extend(excluded)

        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def _row_to_hit(row: sqlite3.Row, score: float, query: str) -> Dict[str, Any]:
        """SQL satırını arama sonucu sözlüğüne çevirir."""
        plain = row["plain"] or ""
        return {
            "id": row["id"],
            "title": row["title"],
            "kind": row["kind"],
            "category": row["category"],
            "tags": [t for t in (row["tags"] or "").split() if t],
            "confidence": float(row["confidence"]),
            "importance": float(row["importance"]),
            "updated": float(row["updated"]),
            "word_count": int(row["word_count"]),
            "path": row["path"],
            "score": round(float(score), 4),
            "excerpt": _make_excerpt(plain, query),
        }

    # ─── Vektör Benzerliği ──────────────────────────────────────────────

    def similar_notes(self, vector: Sequence[float], limit: int = 10,
                      min_score: float = 0.0, **filters: Any) -> List[Dict[str, Any]]:
        """
        Gömme vektörüne en benzer notları bulur (kosinüs).

        Vektörler bellekte önbelleğe alınır; yeni yazma olduğunda tazelenir.

        Args:
            vector: Sorgu vektörü
            limit: Maksimum sonuç
            min_score: Minimum kosinüs benzerliği
            **filters: `kind`, `category`, `tags` filtreleri

        Returns:
            [{"id","title",...,"similarity"}]
        """
        self._require_conn()
        ids, matrix = self._vector_cache()
        if not ids:
            return []

        import math
        query = [float(v) for v in vector]
        q_norm = math.sqrt(sum(v * v for v in query)) or 1.0

        scored: List[Tuple[float, int]] = []
        for index, vec in enumerate(matrix):
            if not vec:
                continue
            length = min(len(vec), len(query))
            dot = 0.0
            v_norm = 0.0
            for i in range(length):
                dot += vec[i] * query[i]
                v_norm += vec[i] * vec[i]
            if v_norm <= 0.0:
                continue
            score = dot / (q_norm * math.sqrt(v_norm))
            if score >= min_score:
                scored.append((score, index))

        scored.sort(key=lambda item: -item[0])
        selected = scored[: max(limit * 3, limit)]
        if not selected:
            return []

        notes = self.get_many([ids[i] for _, i in selected])
        by_id = {note["id"]: note for note in notes}
        results: List[Dict[str, Any]] = []
        for score, index in selected:
            note = by_id.get(ids[index])
            if not note:
                continue
            if filters.get("kind") and note["kind"] != filters["kind"]:
                continue
            if filters.get("category") and note["category"] != filters["category"]:
                continue
            if filters.get("tags") and not set(filters["tags"]).intersection(note["tags"]):
                continue
            note["similarity"] = round(score, 4)
            results.append(note)
            if len(results) >= limit:
                break
        return results

    def _vector_cache(self) -> Tuple[List[str], List[List[float]]]:
        """Vektör matrisini önbelleğe alır (yazma sonrası otomatik tazelenir)."""
        if self._vec_cache is not None:
            self._counter["cache_hits"] += 1
            return self._vec_cache
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                "SELECT id, embedding FROM notes WHERE embedding IS NOT NULL"
            ).fetchall()
        except sqlite3.Error as exc:
            logger.error(f"Vektörler okunamadı: {exc}")
            return [], []
        ids = [row["id"] for row in rows]
        matrix = [blob_to_vector(row["embedding"]) for row in rows]
        self._vec_cache = (ids, matrix)
        return self._vec_cache

    def store_embedding(self, note_id: str, vector: Sequence[float],
                        model: str = "") -> bool:
        """Tek bir notun gömme vektörünü yazar."""
        self._require_conn()
        assert self._conn is not None
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "UPDATE notes SET embedding = ?, embedding_model = ? WHERE id = ?",
                    (vector_to_blob(vector), model, note_id),
                )
            except sqlite3.Error as exc:
                logger.error(f"Gömme yazılamadı ({note_id}): {exc}")
                return False
        self._vec_cache = None
        return cursor.rowcount > 0

    def notes_without_embedding(self, limit: int = 500) -> List[Tuple[str, str]]:
        """Gömme vektörü olmayan notların (id, gövde) listesi."""
        self._require_conn()
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                """SELECT id, title, plain FROM notes
                   WHERE embedding IS NULL ORDER BY importance DESC, updated DESC LIMIT ?""",
                (int(limit),),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [(row["id"], f"{row['title']}\n{row['plain'][:2000]}") for row in rows]

    # ─── Sorgular ───────────────────────────────────────────────────────

    def get_note(self, note_id: str, include_body: bool = True) -> Optional[Dict[str, Any]]:
        """Tek notun indeks kaydını döndürür."""
        notes = self.get_many([note_id], include_body=include_body)
        return notes[0] if notes else None

    def get_many(self, note_ids: Sequence[str], include_body: bool = False) -> List[Dict[str, Any]]:
        """
        Birden çok notu toplu getirir (kimlik sırası korunmaz).

        Args:
            note_ids: Not kimlikleri
            include_body: Gövdeler dahil edilsin mi?
        """
        self._require_conn()
        assert self._conn is not None
        ids = [str(i) for i in note_ids if i]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        columns = ("id, title, kind, category, tags, confidence, importance, "
                   "created, updated, version, word_count, path, sources, checksum, "
                   "embedding_model" + (", body" if include_body else ""))
        try:
            rows = self._conn.execute(
                f"SELECT {columns} FROM notes WHERE id IN ({placeholders})", ids
            ).fetchall()
        except sqlite3.Error as exc:
            logger.error(f"Notlar getirilemedi: {exc}")
            return []
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """SQL satırını tam sözlüğe çevirir."""
        keys = row.keys()
        data: Dict[str, Any] = {
            "id": row["id"],
            "title": row["title"],
            "kind": row["kind"],
            "category": row["category"],
            "tags": [t for t in (row["tags"] or "").split() if t],
            "confidence": float(row["confidence"]),
            "importance": float(row["importance"]),
            "created": float(row["created"]),
            "updated": float(row["updated"]),
            "version": int(row["version"]),
            "word_count": int(row["word_count"]),
            "path": row["path"],
            "sources": [s for s in (row["sources"] or "").split("|") if s],
            "checksum": row["checksum"],
            "embedding_model": row["embedding_model"],
        }
        if "body" in keys:
            data["body"] = row["body"]
        return data

    def all_note_ids(self) -> List[str]:
        """İndeksteki tüm not kimlikleri."""
        self._require_conn()
        assert self._conn is not None
        try:
            return [row["id"] for row in self._conn.execute("SELECT id FROM notes").fetchall()]
        except sqlite3.Error:
            return []

    def by_tag(self, tag: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Etikete göre notlar (güven ve öneme göre sıralı)."""
        self._require_conn()
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                """SELECT n.id, n.title, n.kind, n.category, n.tags, n.confidence,
                          n.importance, n.created, n.updated, n.version, n.word_count,
                          n.path, n.sources, n.checksum, n.embedding_model
                   FROM notes n JOIN tags t ON t.note_id = n.id
                   WHERE t.tag = ?
                   ORDER BY n.importance DESC, n.confidence DESC, n.updated DESC
                   LIMIT ?""",
                (str(tag).lower(), int(limit)),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.error(f"Etiket sorgusu başarısız: {exc}")
            return []
        return [self._row_to_dict(row) for row in rows]

    def by_category(self, category: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Kategoriye göre notlar."""
        self._require_conn()
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                """SELECT id, title, kind, category, tags, confidence, importance,
                          created, updated, version, word_count, path, sources,
                          checksum, embedding_model
                   FROM notes WHERE category = ?
                   ORDER BY updated DESC LIMIT ?""",
                (str(category), int(limit)),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [self._row_to_dict(row) for row in rows]

    def top_tags(self, limit: int = 30) -> List[Tuple[str, int]]:
        """En sık kullanılan etiketler."""
        self._require_conn()
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                "SELECT tag, COUNT(*) AS n FROM tags GROUP BY tag ORDER BY n DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [(row["tag"], int(row["n"])) for row in rows]

    def edges(self, src: Optional[str] = None, dst: Optional[str] = None,
              relation: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        """Kenarları sorgular."""
        self._require_conn()
        assert self._conn is not None
        clauses: List[str] = []
        params: List[Any] = []
        if src:
            clauses.append("src = ?")
            params.append(src)
        if dst:
            clauses.append("dst = ?")
            params.append(dst)
        if relation:
            clauses.append("relation = ?")
            params.append(relation)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        try:
            rows = self._conn.execute(
                f"SELECT src, dst, relation, weight, reason, unresolved FROM edges{where} LIMIT ?",
                (*params, int(limit)),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [
            {
                "src": row["src"], "dst": row["dst"], "relation": row["relation"],
                "weight": float(row["weight"]), "reason": row["reason"],
                "unresolved": bool(row["unresolved"]),
            }
            for row in rows
        ]

    def add_edge(self, src: str, dst: str, relation: str = "links",
                 weight: float = 1.0, reason: str = "", unresolved: bool = False) -> bool:
        """Kenar ekler/günceller (graf katmanı çelişki ve benzerlik kenarları için kullanır)."""
        self._require_conn()
        assert self._conn is not None
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO edges
                       (src, dst, relation, weight, reason, unresolved, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (src, dst, relation, float(weight), reason,
                     1 if unresolved else 0, utc_timestamp()),
                )
            except sqlite3.Error as exc:
                logger.error(f"Kenar eklenemedi ({src}→{dst}): {exc}")
                return False
        return True

    def remove_edge(self, src: str, dst: str, relation: Optional[str] = None) -> int:
        """Kenar(lar)ı siler."""
        self._require_conn()
        assert self._conn is not None
        with self._lock:
            try:
                if relation:
                    cursor = self._conn.execute(
                        "DELETE FROM edges WHERE src = ? AND dst = ? AND relation = ?",
                        (src, dst, relation),
                    )
                else:
                    cursor = self._conn.execute(
                        "DELETE FROM edges WHERE src = ? AND dst = ?", (src, dst)
                    )
                return cursor.rowcount
            except sqlite3.Error:
                return 0

    def history(self, note_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Bir notun indeks geçmişi."""
        self._require_conn()
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                """SELECT action, version, checksum, detail, at FROM history
                   WHERE note_id = ? ORDER BY at DESC LIMIT ?""",
                (note_id, int(limit)),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [dict(row) for row in rows]

    # ─── İstatistik ve Bakım ────────────────────────────────────────────

    def counts(self) -> Dict[str, int]:
        """Tablo satır sayıları."""
        if self._conn is None:
            return {"notes": 0, "tags": 0, "edges": 0, "terms": 0, "history": 0}
        result: Dict[str, int] = {}
        for table in ("notes", "tags", "edges", "terms", "history"):
            try:
                row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
                result[table] = int(row["n"]) if row else 0
            except sqlite3.Error:
                result[table] = 0
        return result

    def index_stats(self) -> Dict[str, Any]:
        """İndeks içeriğinin ayrıntılı istatistiği."""
        counts = self.counts()
        stats: Dict[str, Any] = {**counts, "fts5": self._fts5, "path": str(self._db_path or "")}
        if self._conn is not None:
            try:
                row = self._conn.execute(
                    """SELECT AVG(confidence) AS c, AVG(importance) AS i,
                              SUM(word_count) AS w, COUNT(embedding) AS e,
                              MIN(updated) AS oldest, MAX(updated) AS newest
                       FROM notes"""
                ).fetchone()
                if row:
                    stats.update({
                        "mean_confidence": round(float(row["c"] or 0.0), 4),
                        "mean_importance": round(float(row["i"] or 0.0), 4),
                        "total_words": int(row["w"] or 0),
                        "embedded": int(row["e"] or 0),
                        "oldest": float(row["oldest"] or 0.0),
                        "newest": float(row["newest"] or 0.0),
                    })
            except sqlite3.Error:
                pass
            try:
                stats["db_bytes"] = (
                    self._db_path.stat().st_size if self._db_path and self._db_path.exists() else 0
                )
            except OSError:
                stats["db_bytes"] = 0
        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Katman sözleşmesi: sayaçlar + içerik istatistiği."""
        return {**self._counter, **self.index_stats(), "initialized": self._initialized}

    def optimize(self) -> Dict[str, Any]:
        """VACUUM + ANALYZE + FTS birleştirme."""
        self._require_conn()
        assert self._conn is not None
        start = time.perf_counter()
        actions: List[str] = []
        with self._lock:
            if self._fts5:
                try:
                    self._conn.execute("INSERT INTO notes_fts(notes_fts) VALUES('optimize')")
                    actions.append("fts_optimize")
                except sqlite3.Error as exc:
                    logger.debug(f"FTS optimize atlandı: {exc}")
            try:
                self._conn.execute("ANALYZE")
                actions.append("analyze")
            except sqlite3.Error:
                pass
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                actions.append("wal_checkpoint")
            except sqlite3.Error:
                pass
            try:
                self._conn.execute("VACUUM")
                actions.append("vacuum")
            except sqlite3.Error as exc:
                logger.debug(f"VACUUM atlandı (kilitli olabilir): {exc}")
        duration = (time.perf_counter() - start) * 1000
        return {"actions": actions, "duration_ms": round(duration, 1)}

    def reset(self) -> None:
        """İndeksi tamamen boşaltır ve yeniden kurar."""
        self._require_conn()
        self._drop_schema()
        self._create_schema()
        self._vec_cache = None
        logger.info("İndeks sıfırlandı")

    def generate_embeddings(self, embed_fn: Any, limit: int = 500,
                            model_name: str = "") -> int:
        """
        Gömme vektörü olmayan notlar için vektör üretir.

        Args:
            embed_fn: `Callable[[str], Sequence[float]]` — metni vektöre çevirir
            limit: Tek seferde işlenecek not sayısı
            model_name: Vektörleri üreten modelin adı (izlenebilirlik)

        Returns:
            Yazılan vektör sayısı
        """
        pending = self.notes_without_embedding(limit=limit)
        written = 0
        for note_id, text in pending:
            try:
                vector = embed_fn(text)
            except Exception as exc:
                logger.warning(f"Gömme üretilemedi ({note_id}): {exc}")
                continue
            if vector and self.store_embedding(note_id, vector, model=model_name):
                written += 1
        if written:
            self._emit("knowledge.embeddings_generated", {
                "count": written, "model": model_name, "pending": max(0, len(pending) - written),
            })
            logger.info(f"{written} gömme vektörü üretildi (model: {model_name or 'belirsiz'})")
        return written

    # ─── İç Yardımcılar ─────────────────────────────────────────────────

    def _require_conn(self) -> None:
        """Bağlantı yoksa hata ver."""
        if self._conn is None:
            raise DatabaseUnavailable("İndeks açık değil — initialize() çağırın")

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __enter__(self) -> "KnowledgeIndex":
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<KnowledgeIndex path={self._db_path} fts5={self._fts5}>"


# ─── Modül Yardımcıları ─────────────────────────────────────────────────────

def _clean_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Arama sonucundan iç kullanım alanlarını temizler."""
    return {key: value for key, value in hit.items() if not key.startswith("_")}


def _make_excerpt(plain: str, query: str, radius: int = 140) -> str:
    """
    Sorgu kelimesinin geçtiği yerden kısa bir alıntı üretir.

    Karşılaştırma Türkçe katlanmış metin üzerinde yapılır; ancak alıntı
    **orijinal** metinden kesilir (katlama karakter sayısını değiştirmez).
    Kelime bulunamazsa metnin başını döndürür.
    """
    if not plain:
        return ""
    words = [w for w in _WORD_RE.findall(fold_for_search(query or "")) if len(w) >= 3]
    lowered = fold_for_search(plain)
    position = -1
    for word in words:
        position = lowered.find(word)
        if position >= 0:
            break
    if position < 0:
        text = plain[: radius * 2].strip()
        return text + ("…" if len(plain) > radius * 2 else "")

    start = max(0, position - radius // 2)
    end = min(len(plain), position + radius)
    excerpt = plain[start:end].strip()
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(plain):
        excerpt = excerpt + "…"
    return excerpt.replace("\n", " ")


def build_index_from_notes(index: KnowledgeIndex, notes: Iterable[Note]) -> int:
    """Yardımcı: bir not listesini doğrudan indeksler."""
    return index.upsert_many(notes)


__all__ = [
    "KnowledgeIndex",
    "IndexError_",
    "DatabaseUnavailable",
    "SCHEMA_VERSION",
    "vector_to_blob",
    "blob_to_vector",
    "fold_for_search",
    "build_index_from_notes",
]
