"""
NeuroCore Bilgi Katmanı (L2) — Cephe
=====================================
Obsidian-tarzı bilgi motorunun tek giriş noktası. Kasa (dosya sistemi),
indeks (SQLite), graf (ilişkiler), arama (hibrit) ve sürümleme alt
sistemlerini orkestre eder.

Dış API (motor ve evrim katmanı bunu kullanır)::

    layer = KnowledgeLayer()
    layer.initialize(config, event_bus, embed_fn=neural.embed_text)

    # Yazma
    note   = layer.write(title="GPU Termal", body="...", tags=["gpu"])
    result = layer.ingest("uzun bir metin...", title="Kaynak A", kind="source")
    layer.link(note_a, note_b, relation="supports", reason="aynı ölçüm")

    # Okuma
    hits    = layer.search("GPU sıcaklığı neden yükselir?", limit=8)
    ctx     = layer.context_for("GPU sıcaklığı neden yükselir?")
    payload = layer.get(note_id)
    nb      = layer.neighborhood(note_id, depth=1)

    # Bakım
    layer.reindex()
    layer.maintain()
    report  = layer.health()

Bağımlılık yönü: Bu katman **yalnızca** `neurocore.knowledge.*` ve
`neurocore.core.*` altındaki modülleri kullanır. Nöral katman ters yönde
bağımlılık yaratmamak için `embed_fn` olarak **enjekte edilir**.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from neurocore.knowledge.graph import (
    GraphEdge,
    GraphError,
    GraphMetrics,
    GraphNode,
    KnowledgeGraph,
)
from neurocore.knowledge.index import DatabaseUnavailable, KnowledgeIndex
from neurocore.knowledge.note import (
    Note,
    body_similarity,
    content_hash,
    make_note,
    merge_notes,
    new_note_id,
    normalize_tags,
    now_iso,
    parse_timestamp,
    strip_markdown,
    title_similarity,
)
from neurocore.knowledge.search import (
    ContextBuilder,
    HybridSearch,
    QueryAnalyzer,
    SearchResult,
)
from neurocore.knowledge.vault import (
    ATTACHMENT_DIR,
    INDEX_DIR,
    KnowledgeVault,
    VaultError,
    VaultStats,
)
from neurocore.knowledge.versioning import DiffSummary, NoteVersioner, VersionRecord
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("knowledge.layer")


# ─── Sonuç Yapıları ─────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    """Bir metnin kasaya alınma sonucu."""
    note_id: str = ""
    title: str = ""
    created: bool = False
    updated: bool = False
    duplicated: bool = False
    duplicate_of: str = ""
    chunks: int = 0
    words: int = 0
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    derived_notes: List[str] = field(default_factory=list)
    linked: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.note_id,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            "duplicated": self.duplicated,
            "duplicate_of": self.duplicate_of,
            "chunks": self.chunks,
            "words": self.words,
            "tags": list(self.tags),
            "confidence": round(self.confidence, 4),
            "derived_notes": list(self.derived_notes),
            "linked": self.linked,
            "errors": list(self.errors),
        }


@dataclass
class MaintenanceReport:
    """Bakım turunun sonucu."""
    started_at: float = 0.0
    duration_ms: float = 0.0
    integrity: Dict[str, Any] = field(default_factory=dict)
    reindexed: Dict[str, Any] = field(default_factory=dict)
    graph: Dict[str, Any] = field(default_factory=dict)
    embeddings: int = 0
    duplicates: List[Dict[str, Any]] = field(default_factory=list)
    isolated: List[str] = field(default_factory=list)
    broken_links: List[Tuple[str, str]] = field(default_factory=list)
    low_confidence: List[str] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_ms": round(self.duration_ms, 1),
            "integrity_ok": self.integrity.get("ok"),
            "integrity_issues": len(self.integrity.get("issues") or []),
            "reindexed": self.reindexed,
            "graph": self.graph,
            "embeddings": self.embeddings,
            "duplicates": self.duplicates[:20],
            "isolated": self.isolated[:30],
            "broken_links": self.broken_links[:30],
            "low_confidence": self.low_confidence[:30],
            "link_suggestions": self.suggestions[:20],
            "actions": self.actions,
            "errors": self.errors,
        }


# ─── Ana Katman Sınıfı ──────────────────────────────────────────────────────

class KnowledgeLayer:
    """
    Bilgi katmanı cephesi.

    Tüm alt sistemleri tek bir tutarlı API arkasında toplar ve aralarındaki
    tutarlılığı (yazma → indeksleme → graf → sürüm) otomatik yönetir.

    Yazma işlemleri şu boru hattını izler::

        write() → dedupe kontrolü → vault.save_note() → versioner.snapshot()
                → index.upsert_note() (+gömme) → graph.add_edge() → olay
    """

    name = "knowledge"

    def __init__(self) -> None:
        self.vault = KnowledgeVault()
        self.index = KnowledgeIndex()
        self.graph = KnowledgeGraph()
        self.search_engine = HybridSearch()
        self.versioner = NoteVersioner()
        self.context_builder = ContextBuilder()
        self.analyzer = QueryAnalyzer()

        self._config = None
        self._event_bus = None
        self._embed_fn: Optional[Callable[[str], Sequence[float]]] = None
        self._embed_model: str = ""
        self._initialized = False
        self._lock = threading.RLock()

        # Politika bayrakları
        self.auto_version = True
        self.auto_index = True
        self.auto_embed = True
        self.auto_link = True
        self.auto_dedupe = True
        self.dedupe_threshold = 0.9

        self._counter = {
            "writes": 0, "reads": 0, "searches": 0, "links": 0,
            "duplicates_blocked": 0, "embeddings_stored": 0, "errors": 0,
            "maintenance_runs": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None,
                   embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
                   embed_model: str = "") -> None:
        """
        Bilgi katmanını başlatır.

        Args:
            config: SystemConfig
            event_bus: EventBus
            embed_fn: (metin) -> vektör; nöral katman tarafından enjekte edilir
            embed_model: gömme modeli adı (izlenebilirlik)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        if embed_fn is not None:
            self._embed_fn = embed_fn
        if embed_model:
            self._embed_model = embed_model

        self.vault.initialize(config, event_bus)

        try:
            self.index.initialize(config, event_bus)
        except DatabaseUnavailable as exc:
            logger.error(f"İndeks açılamadı, yalnızca dosya sistemi kullanılacak: {exc}")
            self.auto_index = False

        self.graph.initialize(config, event_bus)
        self.versioner.initialize(config, event_bus)

        self.search_engine.initialize(
            config, event_bus,
            index=self.index if self.auto_index else None,
            graph=self.graph,
            vault=self.vault,
            embed_fn=self._embed_fn,
            embed_model=self._embed_model,
        )

        # İndeks boşsa kasayı tara ve kur
        if self.auto_index:
            try:
                counts = self.index.counts()
                vault_notes = len(self.vault._path_index)  # hızlı yol sayımı
                if counts.get("notes", 0) == 0 and vault_notes > 0:
                    logger.info("İndeks boş — kasadan kuruluyor…")
                    self.index.rebuild(self.vault, prune_missing=False)
            except Exception as exc:
                logger.warning(f"Başlangıç indekslemesi başarısız: {exc}")

        self._initialized = True
        logger.info(
            f"Bilgi katmanı hazır — {len(self.vault._path_index)} not, "
            f"indeks: {'açık' if self.auto_index else 'kapalı'}, "
            f"gömme: {'var' if self._embed_fn else 'yok'}"
        )
        self._emit("knowledge.ready", {
            "vault": str(self.vault.root),
            "notes": len(self.vault._path_index),
            "fts5": self.index.has_fts5 if self.auto_index else False,
            "embedding": bool(self._embed_fn),
        })

    def shutdown(self) -> None:
        """Tüm alt sistemleri kapatır (idempotent)."""
        if not self._initialized:
            return
        for component, name in (
            (self.search_engine, "arama"),
            (self.versioner, "sürümleme"),
            (self.graph, "graf"),
            (self.index, "indeks"),
            (self.vault, "kasa"),
        ):
            try:
                component.shutdown()
            except Exception as exc:
                logger.warning(f"{name} kapatılırken hata: {exc}")
        self._initialized = False
        logger.info(
            f"Bilgi katmanı kapatıldı ({self._counter['writes']} yazma, "
            f"{self._counter['searches']} arama)"
        )

    def attach_embeddings(self, embed_fn: Callable[[str], Sequence[float]],
                          embed_model: str = "") -> None:
        """
        Gömme fonksiyonunu bağlar (nöral katman hazır olduğunda).

        Arama motoru ve indeks bu andan itibaren anlamsal kanalı kullanır.
        """
        self._embed_fn = embed_fn
        if embed_model:
            self._embed_model = embed_model
        self.search_engine.attach(embed_fn=embed_fn, embed_model=embed_model)
        logger.info(f"Gömme fonksiyonu bağlandı: {embed_model or 'isimsiz'}")

    def get_stats(self) -> Dict[str, Any]:
        """Katman sağlık metrikleri."""
        return {
            **self._counter,
            "initialized": self._initialized,
            "embedding_available": self._embed_fn is not None,
            "embed_model": self._embed_model,
            "auto_index": self.auto_index,
            "auto_version": self.auto_version,
            "auto_link": self.auto_link,
            "vault": self.vault.get_stats(),
            "index": self.index.get_stats() if self.auto_index else {"disabled": True},
            "graph": self.graph.get_stats(),
            "search": self.search_engine.get_stats(),
            "versioning": self.versioner.get_stats(),
        }

    @property
    def vault_path(self) -> str:
        """Kasa kök yolu."""
        return str(self.vault.root) if self._initialized else ""

    # ─── Yazma ──────────────────────────────────────────────────────────

    def write(
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
        dedupe: Optional[bool] = None,
        reason: str = "",
        author: str = "system",
    ) -> Note:
        """
        Yeni not yazar veya çok benzeyen bir notu günceller.

        Args:
            title: Başlık
            body: Markdown gövde
            kind: Not türü
            category: Kategori (None ise türden türetilir)
            tags: Etiketler
            confidence: Güven (0..1)
            importance: Önem (0..1)
            sources: Kaynaklar (`os://`, `http://`, `file://`, not id)
            links: Bağlantılı not kimlikleri
            extra: Ek frontmatter alanları
            dedupe: Aynı başlıklı not varsa güncelle (None = politika)
            reason: Sürüm gerekçesi
            author: Değişikliği yapan

        Returns:
            Yazılan/güncellenen Note
        """
        self._require_initialized()
        clean_title = (title or "").strip()
        if not clean_title:
            raise ValueError("Not başlığı zorunludur")

        use_dedupe = self.auto_dedupe if dedupe is None else bool(dedupe)

        if use_dedupe:
            existing = self._find_duplicate(clean_title, body)
            if existing is not None:
                merged = self._merge_into(existing, body, tags, confidence,
                                          sources, links, reason, author)
                self._counter["duplicates_blocked"] += 1
                self._emit("knowledge.duplicate_merged", {
                    "id": merged.id, "title": merged.title,
                    "version": merged.version,
                })
                return merged

        note = make_note(
            title=clean_title,
            body=body,
            kind=(kind or "concept").strip() or "concept",
            category=category or self._category_for((kind or "concept").strip() or "concept"),
            tags=tags,
            confidence=confidence,
            importance=importance,
            sources=sources,
            links=links,
            extra=extra,
        )
        if self._embed_fn is not None and self.auto_embed:
            self._attach_embedding(note, f"{note.title}\n{strip_markdown(note.body)[:1500]}")

        self.vault.save_note(note, touch=False)
        self._counter["writes"] += 1
        self._post_write(note, reason=reason or "yeni not", author=author)
        return note

    def update(
        self,
        note_id: str,
        *,
        body: Optional[str] = None,
        append: bool = False,
        title: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
        sources: Optional[Iterable[str]] = None,
        reason: str = "",
        author: str = "system",
    ) -> Optional[Note]:
        """
        Var olan notu günceller ve tüm alt sistemleri tutarlı tutar.

        Args:
            note_id: Not kimliği
            body: Yeni gövde (`append=True` ise sona eklenir)
            append: Gövdeyi değiştirmek yerine ekle
            title: Yeni başlık
            tags: Eklenecek etiketler
            confidence: Yeni güven değeri
            importance: Yeni önem değeri
            sources: Eklenecek kaynaklar
            reason: Sürüm gerekçesi
            author: Değişikliği yapan

        Returns:
            Güncellenmiş not veya bulunamazsa None
        """
        self._require_initialized()
        # Güncellemeden önce mevcut hâli sürümle
        current = self.vault.read_note(note_id)
        if current is None:
            logger.warning(f"Güncellenecek not bulunamadı: {note_id}")
            return None

        if self.auto_version:
            self.versioner.snapshot(current, reason=reason or "güncelleme öncesi",
                                    author=author)

        updated = self.vault.update_note(
            note_id, body, title=title, tags=tags,
            confidence=confidence, importance=importance,
            append=append, sources=sources,
        )
        if updated is None:
            return None

        if self._embed_fn is not None and self.auto_embed and body is not None:
            self._attach_embedding(updated, f"{updated.title}\n{strip_markdown(updated.body)[:1500]}")

        self._counter["writes"] += 1
        self._post_write(updated, reason=reason or "güncelleme", author=author,
                         snapshot=False)
        return updated

    def ingest(
        self,
        text: str,
        *,
        title: Optional[str] = None,
        kind: str = "source",
        category: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        confidence: float = 0.55,
        importance: float = 0.5,
        source: str = "",
        split: bool = True,
        max_chunk_words: int = 400,
        derive_summary: bool = True,
        reason: str = "içerik alımı",
        author: str = "system",
    ) -> IngestResult:
        """
        Uzun bir metni kasaya alır; gerekirse parçalara böler.

        Boru hattı:
            1. Metin normalize edilir, başlık yoksa ilk satırdan türetilir.
            2. Tekrar kontrolü yapılır (aynı içerik → birleştirme).
            3. `split=True` ve metin uzunsa başlıklara/paragraflara göre bölünür.
            4. Her parça not olarak yazılır; özet not ana nottur.
            5. Parçalar ana nota `part_of`, benzer notlar `related` ile bağlanır.

        Args:
            text: Ham içerik (markdown veya düz metin)
            title: Ana başlık (None ise türetilir)
            kind: Not türü
            category: Kategori
            tags: Etiketler
            confidence: Güven
            importance: Önem
            source: Kaynak adresi/etiketi
            split: Uzun metni parçalara böl
            max_chunk_words: Parça başına maksimum kelime
            derive_summary: Ana nota özet ekle
            reason: Sürüm gerekçesi
            author: Değişikliği yapan

        Returns:
            IngestResult
        """
        self._require_initialized()
        result = IngestResult()
        raw = (text or "").strip()
        if not raw:
            result.errors.append("boş içerik")
            return result

        resolved_title = (title or self._derive_title(raw)).strip()[:200]
        result.title = resolved_title
        resolved_tags = normalize_tags(list(tags or []) + self._derive_tags(raw))
        word_count = len(raw.split())
        result.words = word_count

        try:
            note = self.write(
                title=resolved_title,
                body=raw,
                kind=kind,
                category=category,
                tags=resolved_tags,
                confidence=confidence,
                importance=importance,
                sources=[source] if source else None,
                dedupe=True,
                reason=reason,
                author=author,
            )
        except Exception as exc:
            self._counter["errors"] += 1
            result.errors.append(f"yazma hatası: {exc}")
            return result

        result.note_id = note.id
        result.confidence = note.confidence
        result.tags = list(note.tags)

        # Ana not gövdesi çok uzunsa ve split isteniyorsa parçalara böl
        if split and word_count > max_chunk_words * 1.5:
            chunks = self._split_content(raw, max_chunk_words)
            header_note = note
            for index, chunk in enumerate(chunks, start=1):
                chunk_title = f"{resolved_title} — Bölüm {index}"
                try:
                    part = self.write(
                        title=chunk_title,
                        body=chunk,
                        kind=kind,
                        category=category,
                        tags=resolved_tags,
                        confidence=confidence,
                        importance=max(0.2, importance * 0.8),
                        links=[header_note.id],
                        sources=[source] if source else None,
                        dedupe=True,
                        reason=f"{reason} (parça {index})",
                        author=author,
                    )
                except Exception as exc:
                    result.errors.append(f"parça {index}: {exc}")
                    continue
                result.derived_notes.append(part.id)
                if self.auto_link:
                    self._safe_add_edge(
                        header_note.id, part.id, "part_of",
                        reason=f"içerik bölümü {index}", weight=0.8,
                    )
                    result.linked += 1

            # Ana nota parça listesi ekle
            if result.derived_notes and derive_summary:
                index_lines = "\n".join(
                    f"- [[{nid}]]" for nid in result.derived_notes
                )
                try:
                    self.update(header_note.id,
                                body=f"## Bölümler\n{index_lines}\n",
                                append=True, reason="bölüm listesi eklendi",
                                author=author)
                except Exception as exc:
                    result.errors.append(f"bölüm listesi: {exc}")

        result.chunks = len(result.derived_notes) + 1
        result.created = True
        result.linked += self._auto_link_note(result.note_id) if self.auto_link else 0

        self._emit("knowledge.ingested", result.to_dict())
        logger.info(
            f"İçerik alındı: '{resolved_title}' ({result.words} kelime, "
            f"{result.chunks} parça, {result.linked} bağlantı)"
        )
        return result

    def link(self, src: str, dst: str, relation: str = "links",
             reason: str = "", weight: Optional[float] = None,
             both_ways: bool = False) -> bool:
        """
        İki not arasında ilişki kurar (hem grafta hem notun `links` alanında).

        Args:
            src: Kaynak not kimliği
            dst: Hedef not kimliği
            relation: İlişki türü (`links`, `supports`, `contradicts`, …)
            reason: İnsan okur gerekçe
            weight: Kenar ağırlığı
            both_ways: Ters yönü de ekle

        Returns:
            Başarılı ise True
        """
        self._require_initialized()
        if not src or not dst or src == dst:
            return False

        try:
            self.graph.add_edge(src, dst, relation, weight, reason)
        except GraphError as exc:
            logger.warning(f"Kenar eklenemedi ({src}→{dst}): {exc}")
            return False

        if relation == "links":
            note = self.vault.read_note(src)
            if note is not None and dst not in note.links:
                note.set_links(list(note.links) + [dst])
                note.touch()
                self.vault.save_note(note, touch=False, emit=False)
                if self.auto_index:
                    self.index.upsert_note(note, emit=False)

        if both_ways:
            try:
                self.graph.add_edge(dst, src, relation, weight, reason)
            except GraphError:
                pass
            if relation == "links":
                reverse = self.vault.read_note(dst)
                if reverse is not None and src not in reverse.links:
                    reverse.set_links(list(reverse.links) + [src])
                    reverse.touch()
                    self.vault.save_note(reverse, touch=False, emit=False)
                    if self.auto_index:
                        self.index.upsert_note(reverse, emit=False)

        if self.auto_index:
            self.index.add_edge(src, dst, relation, weight or 1.0, reason)

        self._counter["links"] += 1
        self._emit("knowledge.linked", {
            "src": src, "dst": dst, "relation": relation, "reason": reason,
        })
        return True

    def unlink(self, src: str, dst: str, relation: Optional[str] = None) -> int:
        """İlişkiyi kaldırır."""
        removed = self.graph.remove_edge(src, dst, relation)
        if self.auto_index:
            self.index.remove_edge(src, dst, relation)
        return removed

    def archive(self, note_id: str, reason: str = "") -> bool:
        """Notu arşive taşır."""
        self._require_initialized()
        current = self.vault.read_note(note_id)
        if current is not None and self.auto_version:
            self.versioner.snapshot(current, reason=reason or "arşivleme", author="user")
        moved = self.vault.archive_note(note_id, reason=reason)
        if moved and self.auto_index:
            self.index.delete_note(note_id)
        return bool(moved)

    def delete(self, note_id: str, hard: bool = False, reason: str = "") -> bool:
        """Notu arşivler veya kalıcı siler."""
        self._require_initialized()
        current = self.vault.read_note(note_id)
        if current is not None and self.auto_version:
            self.versioner.snapshot(current, reason=reason or "silme öncesi", author="user")
        ok = self.vault.delete_note(note_id, hard=hard, reason=reason)
        if ok:
            if self.auto_index:
                self.index.delete_note(note_id)
            self.graph.remove_node(note_id)
        return ok

    def attach_file(self, path: str, data: Optional[bytes] = None,
                    note_id: str = "") -> str:
        """Dosyayı kasaya ek olarak kaydeder; göreli yolu döndürür."""
        self._require_initialized()
        file_path = Path(path)
        payload = data if data is not None else file_path.read_bytes()
        suffix = file_path.suffix.lower().lstrip(".") or "other"
        return self.vault.save_attachment(
            payload, file_path.name, subdir=suffix, note_id=note_id
        )

    # ─── Okuma ──────────────────────────────────────────────────────────

    def get(self, note_id: str, include_body: bool = True) -> Optional[Dict[str, Any]]:
        """
        Notu döndürür.

        Önce indeks (hızlı), sonra kasa denenir.
        """
        self._counter["reads"] += 1
        payload: Optional[Dict[str, Any]] = None
        if self.auto_index:
            try:
                payload = self.index.get_note(note_id, include_body=include_body)
            except Exception:
                payload = None
        if payload is None:
            note = self.vault.read_note(note_id)
            if note is None:
                return None
            payload = note.to_dict(include_body=include_body)
            payload["updated"] = note.updated_ts
            payload["created"] = note.created_ts
        return payload

    def get_note(self, note_id: str) -> Optional[Note]:
        """Notu `Note` nesnesi olarak döndürür."""
        self._counter["reads"] += 1
        return self.vault.read_note(note_id)

    def exists(self, note_id: str) -> bool:
        """Not var mı?"""
        return self.vault.find_path(note_id) is not None

    def find_by_title(self, title: str) -> Optional[str]:
        """Başlığa en yakın notun kimliğini döndürür."""
        matches = self.vault.search_by_title(title, limit=1)
        if matches and matches[0][1] >= 0.55:
            return matches[0][0].id
        return None

    def list_notes(self, limit: int = 200, category: Optional[str] = None,
                   kind: Optional[str] = None, tags: Optional[Iterable[str]] = None,
                   order_by: str = "updated") -> List[Dict[str, Any]]:
        """
        Notları listeler.

        Args:
            limit: Maksimum sonuç
            category: Kategori filtresi
            kind: Tür filtresi
            tags: Etiket filtresi (en az biri)
            order_by: `updated`, `created`, `confidence`, `importance`, `title`

        Returns:
            Not yükleri listesi (gövdesiz)
        """
        if self.auto_index and not category and not kind and not tags:
            try:
                payloads = self.index.get_many(self.index.all_note_ids()[: limit * 3])
                return self._sort_payloads(payloads, order_by)[:limit]
            except Exception:
                pass

        payloads: List[Dict[str, Any]] = []
        for note in self.vault.iter_notes(category=category, kind=kind,
                                          tags=tags, limit=limit * 2):
            data = note.to_dict(include_body=False)
            data["updated"] = note.updated_ts
            data["created"] = note.created_ts
            payloads.append(data)
        return self._sort_payloads(payloads, order_by)[:limit]

    @staticmethod
    def _sort_payloads(payloads: List[Dict[str, Any]], order_by: str) -> List[Dict[str, Any]]:
        """Not yüklerini belirtilen alana göre sıralar."""
        keys = {
            "updated": lambda p: -float(p.get("updated", 0) or 0),
            "created": lambda p: -float(p.get("created", 0) or 0),
            "confidence": lambda p: -float(p.get("confidence", 0) or 0),
            "importance": lambda p: -float(p.get("importance", 0) or 0),
            "title": lambda p: str(p.get("title", "")),
        }
        key = keys.get(order_by, keys["updated"])
        return sorted(payloads, key=key)

    def search(self, query: str, limit: int = 8, **kwargs: Any) -> List[SearchResult]:
        """
        Hibrit arama yapar.

        Args:
            query: Arama sorgusu
            limit: Maksimum sonuç
            **kwargs: `expand_graph`, `diversify`, `use_vector`, `min_score`,
                `include_body`, `recency_weight`, `filters`

        Returns:
            SearchResult listesi
        """
        self._counter["searches"] += 1
        kwargs.setdefault("use_vector", self._embed_fn is not None)
        try:
            return self.search_engine.search(query, limit=limit, **kwargs)
        except Exception as exc:
            self._counter["errors"] += 1
            logger.error(f"Arama hatası ('{query}'): {exc}")
            return []

    def search_payloads(self, query: str, limit: int = 8, **kwargs: Any) -> List[Dict[str, Any]]:
        """Aramayı JSON-serileştirilebilir sözlükler olarak döndürür."""
        include_body = bool(kwargs.pop("include_body", False))
        return [result.to_dict(include_body=include_body)
                for result in self.search(query, limit=limit, **kwargs)]

    def context_for(self, query: str, limit: int = 8, max_chars: int = 6000,
                    **kwargs: Any) -> Dict[str, Any]:
        """
        Sorgu için modele verilecek bağlamı üretir.

        Returns:
            {"text", "used", "dropped", "tokens_estimate", "confidence",
             "sources", "results"}
        """
        results = self.search(query, limit=limit, include_body=True, **kwargs)
        builder = ContextBuilder(max_chars=max_chars)
        context = builder.build(results, query=query)
        context["results"] = [result.to_dict() for result in results]
        context["query"] = query
        return context

    def answer_material(self, query: str, limit: int = 8,
                        exclude_kinds: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        Yanıt üretimi için gereken her şeyi tek pakette toplar.

        Bu, motorun nöral katmana beslediği yapıdır: bağlam metni + kaynaklar +
        güven + ilgili not önerileri + eksik bilgi sinyali.

        Args:
            query: Sorgu
            limit: Kullanılacak not sayısı
            exclude_kinds: Bağlamdan çıkarılacak not türleri (None = config'teki
                `context_exclude_kinds`; varsayılan olarak sistemin ürettiği
                yanıt notları hariç tutulur)
        """
        if exclude_kinds is None:
            exclude_kinds = list(getattr(self._config.knowledge, "context_exclude_kinds",
                                         ["qa"]) or [])
        filters: Dict[str, Any] = {}
        if exclude_kinds:
            filters["exclude_kinds"] = list(exclude_kinds)

        context = self.context_for(query, limit=limit, filters=filters or None)
        used: List[SearchResult] = context.get("used") or []
        related: List[Dict[str, Any]] = []
        if used and self.auto_index:
            try:
                for neighbor in self.graph.neighbors(used[0].note_id, depth=1)[:5]:
                    payload = self.get(neighbor, include_body=False)
                    if payload:
                        related.append({"id": payload.get("id"), "title": payload.get("title")})
            except Exception:
                pass

        gaps: List[str] = []
        if not used:
            gaps.append("kasada ilgili not bulunamadı")
        if used and max(result.confidence for result in used) < 0.45:
            gaps.append("ilgili notların güveni düşük")
        if len(used) < 2:
            gaps.append("tek kaynak — çapraz doğrulama yok")
        low_coverage = [result.title for result in used if result.score < 0.35]
        if low_coverage:
            gaps.append(f"zayıf eşleşme: {', '.join(low_coverage[:3])}")

        return {
            "query": query,
            "context": context.get("text", ""),
            "sources": context.get("sources", []),
            "confidence": context.get("confidence", 0.0),
            "tokens_estimate": context.get("tokens_estimate", 0),
            "results": context.get("results", []),
            "related": related,
            "gaps": gaps,
            "needs_research": bool(gaps) and (not used or context.get("confidence", 0.0) < 0.5),
        }

    def neighborhood(self, note_id: str, depth: int = 1) -> Dict[str, Any]:
        """Bir notun komşuluk raporu (graf üzerinden)."""
        return self.graph.neighborhood_report(note_id, depth=depth)

    def related(self, note_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        """İlişkili notlar (graf + vektör)."""
        return [result.to_dict() for result in self.search_engine.related_notes(note_id, limit)]

    def timeline(self, note_id: str) -> str:
        """Notun sürüm zaman çizelgesi (markdown)."""
        return self.versioner.timeline(note_id)

    def history(self, note_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Notun sürüm kayıtları."""
        return [record.to_dict() for record in self.versioner.history(note_id, limit)]

    def restore_version(self, note_id: str, version: int) -> Optional[Dict[str, Any]]:
        """Bir notu eski sürümüne döndürür ve indeksi günceller."""
        restored = self.versioner.restore(note_id, version, vault=self.vault)
        if restored is None:
            return None
        if self.auto_index:
            self.index.upsert_note(restored, emit=False)
        self._emit("knowledge.version_restored", {"id": note_id, "version": version})
        return restored.to_dict()

    def diff_versions(self, note_id: str, from_version: int,
                      to_version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """İki sürüm arasındaki fark."""
        summary: Optional[DiffSummary] = self.versioner.diff(note_id, from_version, to_version)
        return summary.to_dict() if summary else None

    # ─── İndeks ve Graf Bakımı ──────────────────────────────────────────

    def reindex(self, embeddings: bool = True, derive_similarity: bool = True) -> Dict[str, Any]:
        """
        İndeksi ve grafı kasadan yeniden kurar.

        Args:
            embeddings: Eksik gömme vektörlerini üret
            derive_similarity: Benzerlik kenarlarını türet

        Returns:
            {"index": {...}, "graph": {...}, "embeddings": int}
        """
        self._require_initialized()
        start = time.perf_counter()
        result: Dict[str, Any] = {"index": {}, "graph": {}, "embeddings": 0}

        if self.auto_index:
            try:
                result["index"] = self.index.rebuild(self.vault, prune_missing=True)
            except Exception as exc:
                self._counter["errors"] += 1
                result["index"] = {"error": str(exc)}
            if embeddings and self._embed_fn is not None:
                try:
                    result["embeddings"] = self.index.generate_embeddings(
                        self._embed_fn, limit=2000, model_name=self._embed_model
                    )
                except Exception as exc:
                    result["index"]["embedding_error"] = str(exc)

        try:
            result["graph"] = self.graph.rebuild(
                self.vault,
                derive_similarity=derive_similarity,
                embed_fn=self._embed_fn if self.graph and self._embed_fn else None,
            )
        except Exception as exc:
            self._counter["errors"] += 1
            result["graph"] = {"error": str(exc)}

        self.search_engine.refresh_centrality()
        self.search_engine.clear_cache()
        result["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        self._emit("knowledge.reindexed", result)
        logger.info(f"Yeniden indeksleme tamamlandı ({result['duration_ms']:.0f}ms)")
        return result

    def maintain(self, *, fix: bool = False, embeddings: bool = True,
                 reindex: bool = False, derive_similarity: bool = True,
                 limit: int = 2000) -> MaintenanceReport:
        """
        Bakım turu: bütünlük denetimi, indeks/gömme tazeleme, graf analizi.

        Args:
            fix: Basit sorunları düzelt (bozuk frontmatter, kimlik çakışması)
            embeddings: Eksik gömmeleri üret
            reindex: Tam yeniden indeksleme yap
            derive_similarity: Benzerlik kenarlarını türet
            limit: İşlenecek maksimum not sayısı

        Returns:
            MaintenanceReport
        """
        self._require_initialized()
        report = MaintenanceReport(started_at=time.time())
        self._counter["maintenance_runs"] += 1

        if reindex:
            try:
                outcome = self.reindex(embeddings=embeddings,
                                       derive_similarity=derive_similarity)
                report.reindexed = outcome.get("index", {})
                report.graph = outcome.get("graph", {})
                report.embeddings = int(outcome.get("embeddings", 0) or 0)
                report.actions.append("reindex")
            except Exception as exc:
                report.errors.append(f"reindex: {exc}")

        # 1. Bütünlük
        try:
            report.integrity = self.vault.integrity_check(fix=fix, max_notes=limit)
        except Exception as exc:
            report.errors.append(f"integrity: {exc}")

        # 2. Gömme tazeleme
        if embeddings and self._embed_fn is not None and self.auto_index:
            try:
                report.embeddings += self.index.generate_embeddings(
                    self._embed_fn, limit=limit, model_name=self._embed_model
                )
                if report.embeddings:
                    report.actions.append(f"{report.embeddings} gömme üretildi")
            except Exception as exc:
                report.errors.append(f"embeddings: {exc}")

        # 3. Graf analizi
        try:
            if self.graph.edge_count == 0 and self.graph.node_count > 0:
                report.graph = self.graph.rebuild(
                    self.vault, derive_similarity=derive_similarity,
                    embed_fn=self._embed_fn,
                )
                report.actions.append("graf yeniden kuruldu")
            elif derive_similarity:
                added = self.graph.derive_similarity_edges(
                    threshold=0.7, embed_fn=self._embed_fn
                )
                if added:
                    report.actions.append(f"{added} benzerlik kenarı")
            metrics: GraphMetrics = self.graph.metrics()
            report.graph = metrics.to_dict()
            report.isolated = self.graph.find_isolated()
            report.broken_links = self.graph.find_broken_links()
        except Exception as exc:
            report.errors.append(f"graph: {exc}")

        # 4. Tekrar ve öneri analizi
        try:
            report.duplicates = self.vault.find_duplicates(threshold=0.82, limit=50)
            report.suggestions = self.graph.suggest_links(threshold=0.6, limit=30)
        except Exception as exc:
            report.errors.append(f"duplicates: {exc}")

        # 5. Düşük güvenli notlar
        try:
            report.low_confidence = [
                note.id for note in self.vault.iter_notes(limit=limit)
                if note.confidence < 0.4
            ]
        except Exception as exc:
            report.errors.append(f"low_confidence: {exc}")

        report.duration_ms = (time.time() - report.started_at) * 1000
        self.search_engine.refresh_centrality()
        self._emit("knowledge.maintained", {
            "duration_ms": round(report.duration_ms, 1),
            "actions": report.actions,
            "errors": report.errors,
            "issues": len(report.integrity.get("issues") or []),
        })
        logger.info(
            f"Bakım tamamlandı ({report.duration_ms:.0f}ms): "
            f"{len(report.integrity.get('issues') or [])} bütünlük sorunu, "
            f"{len(report.duplicates)} tekrar, {len(report.suggestions)} bağlantı önerisi"
        )
        return report

    def metrics(self) -> Dict[str, Any]:
        """Kasa, indeks ve graf metriklerinin birleşik özeti."""
        result: Dict[str, Any] = {}
        try:
            result["vault"] = self.vault.vault_stats().to_dict()
        except Exception as exc:
            result["vault"] = {"error": str(exc)}
        try:
            result["index"] = self.index.index_stats() if self.auto_index else {"disabled": True}
        except Exception as exc:
            result["index"] = {"error": str(exc)}
        try:
            result["graph"] = self.graph.metrics().to_dict()
        except Exception as exc:
            result["graph"] = {"error": str(exc)}
        try:
            result["versioning"] = self.versioner.global_stats()
        except Exception as exc:
            result["versioning"] = {"error": str(exc)}
        return result

    def hub_notes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """En merkezi notlar (bilgi haritasının ana düğümleri)."""
        return self.graph.hub_notes(limit)

    def export_graph(self, path: Optional[str] = None) -> str:
        """Grafı JSON olarak dışa aktarır."""
        target = path or str(Path(self.vault.index_dir) / "graph.json")
        if self.graph.node_count == 0:
            try:
                self.graph.rebuild(self.vault)
            except Exception as exc:
                logger.warning(f"Graf kurulamadı: {exc}")
        return self.graph.save(target)

    def export_graph_note(self, title: str = "Bilgi Grafı") -> Dict[str, Any]:
        """Graf raporunu Obsidian notu olarak kasaya yazar."""
        content = self.graph.to_markdown(title)
        note = self.write(
            title=title,
            body=content,
            kind="meta",
            category="meta",
            tags=["meta", "graf", "rapor"],
            confidence=0.9,
            importance=0.4,
            reason="graf raporu",
            author="system",
        )
        return {"id": note.id, "title": note.title, "path": note.path}

    # ─── Sağlık ─────────────────────────────────────────────────────────

    def health(self, deep: bool = False) -> Dict[str, Any]:
        """
        Katman sağlık raporu.

        Args:
            deep: Ayrıntılı denetim (bütünlük taraması yapar — yavaş)

        Returns:
            {"ok": bool, "issues": [...], "details": {...}}
        """
        issues: List[str] = []
        details: Dict[str, Any] = {}

        if not self._initialized:
            return {"ok": False, "issues": ["katman başlatılmadı"], "details": {}}

        try:
            details["notes"] = len(self.vault._path_index)
            if details["notes"] == 0:
                issues.append("kasa boş")
        except Exception as exc:
            issues.append(f"kasa okunamadı: {exc}")
            details["notes"] = 0

        if self.auto_index:
            try:
                counts = self.index.counts()
                details["indexed"] = counts.get("notes", 0)
                if details.get("notes", 0) > 0 and counts.get("notes", 0) == 0:
                    issues.append("indeks boş — reindex() gerekli")
                details["fts5"] = self.index.has_fts5
                if not self.index.has_fts5:
                    issues.append("FTS5 yok — arama kalitesi düşük")
            except Exception as exc:
                issues.append(f"indeks hatası: {exc}")
        else:
            issues.append("indeks devre dışı")

        try:
            details["graph_nodes"] = self.graph.node_count
            details["graph_edges"] = self.graph.edge_count
            if self.graph.node_count > 0 and self.graph.edge_count == 0:
                issues.append("graf kurulmamış (hiç ilişki yok)")
        except Exception as exc:
            issues.append(f"graf hatası: {exc}")

        if self._embed_fn is None:
            issues.append("gömme fonksiyonu bağlı değil — anlamsal arama kapalı")

        try:
            details["tracked_versions"] = len(self.versioner.all_tracked())
        except Exception:
            details["tracked_versions"] = 0

        if deep:
            try:
                integrity = self.vault.integrity_check(fix=False, max_notes=500)
                details["integrity"] = {
                    "ok": integrity.get("ok"),
                    "checked": integrity.get("checked"),
                    "issues": len(integrity.get("issues") or []),
                }
                if not integrity.get("ok"):
                    issues.append("kasa bütünlük sorunları var")
            except Exception as exc:
                issues.append(f"bütünlük denetimi hatası: {exc}")

        return {"ok": not issues or issues == ["gömme fonksiyonu bağlı değil — anlamsal arama kapalı"],
                "issues": issues, "details": details}

    # ─── İç Yardımcılar ─────────────────────────────────────────────────

    def _require_initialized(self) -> None:
        """Başlatılmadıysa hata ver."""
        if not self._initialized:
            raise VaultError("Bilgi katmanı başlatılmadı — initialize() çağırın")

    @staticmethod
    def _category_for(kind: str) -> str:
        """Not türünden kategori türetir."""
        return {
            "concept": "learned", "summary": "learned", "skill": "learned",
            "research": "research", "source": "research",
            "entity": "user", "session": "user", "project": "user",
            "journal": "meta", "meta": "meta",
        }.get(kind, "learned")

    def _attach_embedding(self, note: Note, text: str) -> None:
        """Nota gömme vektörü hesaplar (hata durumunda sessizce geçer)."""
        if self._embed_fn is None:
            return
        try:
            vector = self._embed_fn(text)
        except Exception as exc:
            logger.debug(f"Gömme hesaplanamadı ({note.id}): {exc}")
            return
        if not vector:
            return
        note.embedding = [float(v) for v in vector]
        note.embedding_model = self._embed_model
        self._counter["embeddings_stored"] += 1

    def _post_write(self, note: Note, reason: str = "", author: str = "system",
                    snapshot: bool = True) -> None:
        """
        Yazma sonrası tutarlılık boru hattı.

        Sürümleme → indeksleme → graf kenarları → olay.
        Her adım ayrı ayrı korunur: biri başarısız olsa da diğerleri sürer.
        """
        if snapshot and self.auto_version:
            try:
                self.versioner.snapshot(note, reason=reason, author=author)
            except Exception as exc:
                logger.warning(f"Sürüm oluşturulamadı ({note.id}): {exc}")

        if self.auto_index:
            try:
                self.index.upsert_note(note)
            except Exception as exc:
                logger.warning(f"İndekslenemedi ({note.id}): {exc}")

        try:
            self.graph.add_node(GraphNode(
                id=note.id, title=note.title, kind=note.kind,
                category=note.category, tags=list(note.tags),
                confidence=note.confidence, importance=note.importance,
                word_count=note.word_count, updated=note.updated_ts,
                path=note.path,
            ))
            for link in note.links:
                self._safe_add_edge(note.id, link, "links", reason="not bağlantısı")
        except Exception as exc:
            logger.warning(f"Graf güncellenemedi ({note.id}): {exc}")

        self._emit("knowledge.note_saved", {
            "id": note.id, "title": note.title, "kind": note.kind,
            "tags": list(note.tags), "version": note.version,
            "words": note.word_count, "reason": reason,
        })

    def _safe_add_edge(self, src: str, dst: str, relation: str,
                       reason: str = "", weight: Optional[float] = None) -> bool:
        """Kenar ekler; hata durumunda False döner (asla yükseltmez)."""
        try:
            self.graph.add_edge(src, dst, relation, weight, reason)
            return True
        except GraphError:
            return False
        except Exception as exc:
            logger.debug(f"Kenar eklenemedi ({src}→{dst}): {exc}")
            return False

    def _find_duplicate(self, title: str, body: str) -> Optional[Note]:
        """
        Aynı başlıklı veya neredeyse aynı içerikli notu bulur.

        İki aşamalı: hızlı başlık taraması, ardından içerik benzerliği.
        """
        threshold = self.dedupe_threshold
        matches = self.vault.search_by_title(title, limit=5)
        for note, score in matches:
            if score >= 0.995:
                return note
            if score >= 0.85 and body:
                similarity = body_similarity(note.body, body)
                if similarity >= threshold * 0.85:
                    return note
            if score >= 0.95 and not body:
                return note
        return None

    def _merge_into(self, existing: Note, body: str, tags: Optional[Iterable[str]],
                    confidence: float, sources: Optional[Iterable[str]],
                    links: Optional[Iterable[str]], reason: str,
                    author: str) -> Note:
        """Yeni içeriği var olan nota birleştirir (tekrar notu oluşturmaz)."""
        merged_body = existing.body.rstrip()
        new_content = strip_markdown(body or "").strip()
        if new_content and new_content not in strip_markdown(merged_body):
            merged_body += f"\n\n### Güncelleme ({now_iso()[:16]})\n{body.strip()}\n"

        # Güven: yeni kanıt geldikçe artar, ama asla 1.0'ı aşmaz
        blended = min(0.99, max(existing.confidence, float(confidence)) + 0.03)

        updates: Dict[str, Any] = {
            "body": merged_body,
            "confidence": blended,
        }
        if tags:
            combined_tags = list(dict.fromkeys(list(existing.tags) + normalize_tags(list(tags))))
            updates["tags"] = combined_tags
        if sources:
            updates["sources"] = list(dict.fromkeys(
                list(existing.sources) + [str(s) for s in sources]
            ))
        if links:
            updates["links"] = list(dict.fromkeys(list(existing.links) + [str(l) for l in links]))

        return self.update(
            existing.id,
            body=updates.pop("body"),
            tags=updates.pop("tags", None),
            confidence=updates.pop("confidence", None),
            sources=updates.pop("sources", None),
            reason=reason or "tekrar birleştirme",
            author=author,
        ) or existing

    def _auto_link_note(self, note_id: str, max_links: int = 4) -> int:
        """
        Yeni nota otomatik ilişki önerir.

        Benzerlik eşiğini geçen notlara `related` kenarı ekler; böylece kasa
        kendiliğinden örülür.
        """
        payload = self.get(note_id, include_body=True)
        if not payload:
            return 0
        text = f"{payload.get('title', '')} {payload.get('body', '')[:800]}"
        candidates: List[Tuple[str, float]] = []

        # Graf üzerinden yapısal adaylar
        try:
            for suggestion in self.graph.suggest_links(threshold=0.62, limit=20):
                if suggestion["a"] == note_id:
                    candidates.append((suggestion["b"], float(suggestion["score"])))
                elif suggestion["b"] == note_id:
                    candidates.append((suggestion["a"], float(suggestion["score"])))
        except Exception:
            pass

        # Vektör üzerinden anlamsal adaylar
        if self._embed_fn is not None and self.auto_index and payload.get("embedding"):
            try:
                vector = payload["embedding"]
                for hit in self.index.similar_notes(vector, limit=10, min_score=0.45):
                    if hit["id"] != note_id:
                        candidates.append((hit["id"], float(hit.get("similarity", 0.5))))
            except Exception:
                pass

        added = 0
        seen: Set[str] = set()
        for other_id, score in sorted(candidates, key=lambda item: -item[1]):
            if added >= max_links or other_id in seen or other_id == note_id:
                continue
            seen.add(other_id)
            if self._safe_add_edge(note_id, other_id, "related",
                                   reason=f"otomatik benzerlik {score:.2f}",
                                   weight=round(score, 4)):
                if self.auto_index:
                    self.index.add_edge(note_id, other_id, "related", score,
                                        "otomatik benzerlik")
                added += 1
        return added

    @staticmethod
    def _derive_title(text: str, max_length: int = 120) -> str:
        """Metinden başlık türetir: ilk başlık satırı, yoksa ilk cümle."""
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                candidate = stripped.lstrip("#").strip()
                if candidate:
                    return candidate[:max_length]
            # İlk anlamlı satır
            if len(stripped) > 3:
                candidate = stripped.split(".")[0].strip()
                return (candidate or stripped)[:max_length]
        return f"Alınan içerik {now_iso()[:16]}"

    @staticmethod
    def _derive_tags(text: str, max_tags: int = 6) -> List[str]:
        """
        Metinden etiket adayları türetir (frekans tabanlı, durak kelimesiz).

        Gerçek anahtar kelime çıkarımı dil katmanında yapılır; bu yalnızca
        kasayı ilk yazımda makul biçimde etiketler.
        """
        import re
        from collections import Counter

        words = re.findall(r"[\wÇĞİÖŞÜçğıöşü]{4,}", (text or "").lower())
        if not words:
            return []
        try:
            from neurocore.knowledge.search import STOPWORDS
            stop = STOPWORDS
        except Exception:
            stop = set()
        counts = Counter(word for word in words if word not in stop)
        return [word for word, _ in counts.most_common(max_tags)]

    @staticmethod
    def _split_content(text: str, max_words: int = 400) -> List[str]:
        """
        İçeriği anlamlı parçalara böler.

        Önce markdown başlıklarından, yoksa paragraflardan, gerekirse
        cümlelerden böler; hiçbir parça `max_words * 1.5`'i geçmez.
        """
        import re

        stripped = (text or "").strip()
        if not stripped:
            return []

        # 1) Markdown başlıklarından böl
        sections = re.split(r"(?m)^(?=#{2,3}\s)", stripped)
        sections = [section.strip() for section in sections if section.strip()]

        # 2) Paragraflardan böl (başlık bölmesi yetersizse)
        if len(sections) <= 1:
            sections = [part.strip() for part in re.split(r"\n\s*\n", stripped) if part.strip()]

        # 3) Cümlelerden böl (paragraf hâlâ çok uzunsa)
        chunks: List[str] = []
        for section in sections:
            if len(section.split()) <= max_words * 1.5:
                chunks.append(section)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", section)
            buffer: List[str] = []
            count = 0
            for sentence in sentences:
                words = len(sentence.split())
                if count + words > max_words and buffer:
                    chunks.append(" ".join(buffer))
                    buffer, count = [], 0
                buffer.append(sentence)
                count += words
            if buffer:
                chunks.append(" ".join(buffer))

        return [chunk for chunk in chunks if chunk.strip()][:40]

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
            f"<KnowledgeLayer initialized={self._initialized} "
            f"notes={len(self.vault._path_index)} graph={self.graph.node_count}>"
        )


__all__ = [
    "KnowledgeLayer",
    "IngestResult",
    "MaintenanceReport",
]
