"""
NeuroCore Hibrit Arama
=======================
Kasadan bilgi getiren çok aşamalı arama motoru.

Boru hattı::

    sorgu
      ├─ 1. Sorgu analizi      → anahtar kelimeler, etiket ipuçları, niyet
      ├─ 2. Paralel geri çağırma
      │      ├─ FTS5/BM25     (kelime eşleşmesi)
      │      ├─ Vektör        (anlamsal benzerlik, gömme varsa)
      │      └─ Etiket/başlık (yapısal sinyal)
      ├─ 3. Kaynaştırma        → Reciprocal Rank Fusion (RRF)
      ├─ 4. Graf genişletme    → komşu notlar (isteğe bağlı)
      ├─ 5. Yeniden sıralama   → kapsam + güven + önem + tazelik + merkezilik
      └─ 6. Çeşitlilik (MMR)   → tekrar eden notları ele

Tasarım kararı: her aşama ayrı ayrı başarısız olabilir ve boru hattı devam
eder. Gömme yoksa vektör aşaması atlanır, graf yoksa genişletme atlanır —
arama her koşulda sonuç döndürür.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from neurocore.knowledge.note import (
    Note,
    cosine_similarity,
    fold_turkish,
    normalize_tags,
    strip_markdown,
    title_similarity,
    utc_timestamp,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("knowledge.search")


# ─── Sabitler ───────────────────────────────────────────────────────────────

#: Sorgu kelimelerinde anlam taşımayan Türkçe/İngilizce durak kelimeler
STOPWORDS: Set[str] = {
    # Türkçe
    "ve", "ile", "için", "bir", "bu", "şu", "o", "da", "de", "ki", "mi", "mı",
    "mu", "mü", "ne", "nasıl", "neden", "niçin", "nedir", "kim", "hangi",
    "çok", "daha", "en", "gibi", "kadar", "sonra", "önce", "ama", "fakat",
    "ancak", "veya", "ya", "eğer", "ise", "olarak", "olan", "var", "yok",
    "ben", "sen", "biz", "siz", "onlar", "benim", "senin", "bizim", "sizin",
    "her", "bazı", "tüm", "bütün", "şey", "şeyler", "zaman", "gün", "acaba",
    # İngilizce
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "at", "for", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "should", "may", "might", "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their", "what", "why", "how",
    "when", "where", "which", "who", "this", "that", "these", "those", "not",
}

#: Niyet sinyalleri
INTENT_SIGNALS: Dict[str, Tuple[str, ...]] = {
    "definition": ("nedir", "ne demek", "tanım", "what is", "definition", "meaning"),
    "howto": ("nasıl", "how to", "how do", "adımlar", "steps", "yöntem", "rehber"),
    "causal": ("neden", "niçin", "why", "sebep", "cause", "yüzünden", "dolayı"),
    "comparison": ("karşılaştır", "fark", "vs", "compare", "difference", "hangisi"),
    "list": ("listele", "say", "örnekler", "list", "examples", "kaç tane"),
    "system": ("sistem", "cpu", "gpu", "ram", "disk", "donanım", "işlemci", "bellek"),
    "code": ("kod", "code", "fonksiyon", "function", "script", "hata", "bug", "debug"),
    "research": ("araştır", "research", "incele", "kaynak", "makale", "paper"),
}

_WORD_RE = re.compile(r"[\wÇĞİÖŞÜçğıöşü]+", re.UNICODE)


class SearchError(Exception):
    """Arama hatası."""


# ─── Sorgu Analizi ──────────────────────────────────────────────────────────

@dataclass
class QueryAnalysis:
    """Ayrıştırılmış sorgu."""
    original: str
    normalized: str
    keywords: List[str] = field(default_factory=list)
    all_words: List[str] = field(default_factory=list)
    tag_hints: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)
    is_question: bool = False
    is_short: bool = False
    language_hint: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "keywords": self.keywords,
            "tag_hints": self.tag_hints,
            "intents": self.intents,
            "is_question": self.is_question,
            "is_short": self.is_short,
            "language_hint": self.language_hint,
        }

    def fts_query(self) -> str:
        """FTS için sorgu metni (anahtar kelimeler öncelikli)."""
        words = self.keywords or self.all_words
        return " ".join(words) if words else self.original


class QueryAnalyzer:
    """
    Sorguyu arama için hazırlar.

    Türkçe ekleri kabaca kırpar, durak kelimeleri atar, etiket ipuçlarını
    ve niyeti çıkarır. LLM kullanmaz — deterministik ve hızlıdır.
    """

    #: Türkçe çekim ekleri (uzundan kısaya — en uzun ek önce denenir)
    SUFFIXES: Tuple[str, ...] = (
        "larindan", "lerinden", "larımız", "lerimiz", "sından", "sinden",
        "lardan", "lerden", "ların", "lerin", "sıyla", "siyle", "ları", "leri",
        "dan", "den", "tan", "ten", "nin", "nın", "nun", "nün", "lar", "ler",
        "sı", "si", "su", "sü", "ın", "in", "un", "ün", "da", "de", "ta", "te",
        "ya", "ye", "ca", "ce", "lı", "li", "lu", "lü",
    )

    def __init__(self, min_word_length: int = 3, stem: bool = True) -> None:
        self.min_word_length = min_word_length
        self.stem = stem

    def analyze(self, query: str) -> QueryAnalysis:
        """Sorguyu ayrıştırır."""
        original = (query or "").strip()
        normalized = self.normalize(original)
        words = _WORD_RE.findall(normalized.lower())

        analysis = QueryAnalysis(
            original=original,
            normalized=normalized,
            all_words=words,
        )

        # Anahtar kelimeler: durak kelimeler atılır, ekler kırpılır
        keywords: List[str] = []
        for word in words:
            if len(word) < self.min_word_length or word in STOPWORDS:
                continue
            stemmed = self.stem_word(word) if self.stem else word
            if stemmed and stemmed not in keywords:
                keywords.append(stemmed)
        analysis.keywords = keywords[:12]

        # Etiket ipuçları: kısa, tek kelimelik, durak olmayan terimler
        analysis.tag_hints = [
            word for word in words
            if 3 <= len(word) <= 24 and word not in STOPWORDS
        ][:8]

        # Niyet
        lowered = normalized.lower()
        analysis.intents = [
            intent for intent, signals in INTENT_SIGNALS.items()
            if any(signal in lowered for signal in signals)
        ]
        analysis.is_question = (
            "?" in original
            or any(lowered.startswith(prefix) for prefix in
                   ("ne ", "neden", "nasıl", "niçin", "kim", "hangi", "kaç",
                    "what", "why", "how", "when", "where", "which", "who"))
        )
        analysis.is_short = len(words) <= 2
        analysis.language_hint = self._guess_language(words)
        return analysis

    @staticmethod
    def normalize(text: str) -> str:
        """Sorgu metnini sadeleştirir: NFC, boşluk, tırnak temizliği."""
        import unicodedata
        normalized = unicodedata.normalize("NFC", text or "")
        normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def stem_word(self, word: str) -> str:
        """
        Türkçe ekleri kabaca kırpar (gerçek morfoloji değil, geri çağırma için).

        Kök en az 3 karakter kalacak şekilde en uzun eşleşen ek atılır.
        """
        if len(word) <= 4:
            return word
        for suffix in self.SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    @staticmethod
    def _guess_language(words: Sequence[str]) -> str:
        """Kaba dil tahmini (yalnızca ipucu)."""
        if not words:
            return "unknown"
        text = " ".join(words)
        tr_markers = ("ı", "ş", "ğ", "ç", "ö", "ü")
        if any(marker in text for marker in tr_markers):
            return "tr"
        tr_words = {"ve", "ile", "için", "bir", "nedir", "nasıl", "neden"}
        if tr_words & set(words):
            return "tr"
        en_words = {"the", "and", "what", "how", "why", "is", "are"}
        if en_words & set(words):
            return "en"
        return "unknown"


# ─── Arama Sonucu ───────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Tek bir arama sonucu."""
    note_id: str
    title: str
    score: float
    kind: str = "concept"
    category: str = "learned"
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.5
    importance: float = 0.5
    updated: float = 0.0
    path: str = ""
    excerpt: str = ""
    body: str = ""
    source: str = ""            # hangi kanaldan geldi: fts | vector | graph | tag
    channel_scores: Dict[str, float] = field(default_factory=dict)
    via: str = ""               # graf üzerinden geldiyse çıkış notu
    hop: int = 0

    def to_dict(self, include_body: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.note_id,
            "title": self.title,
            "score": round(self.score, 5),
            "kind": self.kind,
            "category": self.category,
            "tags": list(self.tags),
            "confidence": round(self.confidence, 4),
            "importance": round(self.importance, 4),
            "updated": self.updated,
            "path": self.path,
            "excerpt": self.excerpt,
            "source": self.source,
            "channel_scores": {k: round(v, 5) for k, v in self.channel_scores.items()},
            "via": self.via,
            "hop": self.hop,
        }
        if include_body:
            data["body"] = self.body
        return data

    @property
    def citation(self) -> str:
        """Obsidian bağlantısı biçiminde atıf."""
        return f"[[{self.note_id}\\|{self.title}]]"


# ─── Yeniden Sıralama Ağırlıkları ───────────────────────────────────────────

@dataclass
class RerankWeights:
    """
    Yeniden sıralama bileşen ağırlıkları.

    Toplam 1.0 olmak zorunda değildir; skor en sonda 0..1 aralığına kırpılır.
    """
    relevance: float = 0.38      # kaynaştırılmış geri çağırma skoru
    coverage: float = 0.20       # sorgu kelimelerinin kapsamı
    confidence: float = 0.16     # notun güven değeri
    importance: float = 0.10     # notun önem değeri
    recency: float = 0.08        # tazelik (yarı ömür 180 gün)
    centrality: float = 0.08     # graf merkeziliği
    recency_half_life_days: float = 180.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "relevance": self.relevance, "coverage": self.coverage,
            "confidence": self.confidence, "importance": self.importance,
            "recency": self.recency, "centrality": self.centrality,
            "recency_half_life_days": self.recency_half_life_days,
        }


# ─── Ana Hibrit Arama Sınıfı ────────────────────────────────────────────────

class HybridSearch:
    """
    Çok kanallı hibrit arama motoru.

    Kullanım:
        search = HybridSearch()
        search.initialize(config, event_bus)
        search.attach(index=knowledge_index, graph=knowledge_graph,
                      vault=knowledge_vault, embed_fn=neural.embed_text)

        results = search.search("GPU sıcaklığı neden yükselir?", limit=8)
        for r in results:
            print(r.score, r.title, r.excerpt)

        stats = search.get_stats()
    """

    name = "knowledge.search"

    def __init__(self, analyzer: Optional[QueryAnalyzer] = None) -> None:
        self._index: Any = None
        self._graph: Any = None
        self._vault: Any = None
        self._embed_fn: Optional[Callable[[str], Sequence[float]]] = None
        self._embed_model: str = ""
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._lock = threading.RLock()

        self.analyzer = analyzer or QueryAnalyzer()
        self.weights = RerankWeights()
        self._pagerank: Dict[str, float] = {}

        self._counter = {
            "queries": 0, "fts_calls": 0, "vector_calls": 0,
            "graph_expansions": 0, "cache_hits": 0, "errors": 0,
            "total_results": 0, "total_duration_ms": 0.0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None, **components: Any) -> None:
        """
        Arama motorunu başlat.

        Args:
            config: SystemConfig
            event_bus: EventBus
            **components: `index`, `graph`, `vault`, `embed_fn`, `embed_model`
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self.attach(**components)

        search_cache = getattr(getattr(config, "knowledge", None), "search_cache_size", 1000)
        self._cache: Dict[str, Tuple[float, List[SearchResult]]] = {}
        self._cache_size = int(search_cache or 1000)
        self._cache_ttl = 30.0

        self._initialized = True
        logger.info(
            f"Hibrit arama hazır (indeks: {'var' if self._index else 'yok'}, "
            f"graf: {'var' if self._graph else 'yok'}, "
            f"gömme: {'var' if self._embed_fn else 'yok'})"
        )

    def attach(self, index: Any = None, graph: Any = None, vault: Any = None,
               embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
               embed_model: str = "") -> None:
        """
        Bileşenleri bağlar (initialize sonrası da çağrılabilir).

        Args:
            index: KnowledgeIndex
            graph: KnowledgeGraph
            vault: KnowledgeVault
            embed_fn: (metin) -> vektör
            embed_model: gömme modeli adı (izlenebilirlik)
        """
        with self._lock:
            if index is not None:
                self._index = index
            if graph is not None:
                self._graph = graph
                self.refresh_centrality()
            if vault is not None:
                self._vault = vault
            if embed_fn is not None:
                self._embed_fn = embed_fn
            if embed_model:
                self._embed_model = embed_model
            if hasattr(self, "_cache"):
                self._cache.clear()

    def shutdown(self) -> None:
        """Kaynakları serbest bırak."""
        self._initialized = False
        self._cache = {}
        logger.info(
            f"Hibrit arama kapatıldı ({self._counter['queries']} sorgu, "
            f"ort. {self.avg_duration_ms:.0f}ms)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Arama istatistikleri."""
        return {
            **self._counter,
            "avg_duration_ms": self.avg_duration_ms,
            "index_attached": self._index is not None,
            "graph_attached": self._graph is not None,
            "embeddings_available": self._embed_fn is not None,
            "embed_model": self._embed_model,
            "cache_size": len(getattr(self, "_cache", {}) or {}),
            "weights": self.weights.to_dict(),
        }

    @property
    def avg_duration_ms(self) -> float:
        """Sorgu başına ortalama süre."""
        queries = self._counter["queries"]
        return (self._counter["total_duration_ms"] / queries) if queries else 0.0

    # ─── Ana Arama ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 8,
        *,
        expand_graph: bool = True,
        graph_depth: int = 1,
        diversify: bool = True,
        use_vector: bool = True,
        min_score: float = 0.0,
        include_body: bool = False,
        recency_weight: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> List[SearchResult]:
        """
        Hibrit arama yapar.

        Args:
            query: Arama sorgusu
            limit: Maksimum sonuç sayısı
            expand_graph: Graf komşuluğunu sonuçlara ekle
            graph_depth: Graf genişletme derinliği
            diversify: MMR ile çeşitlilik uygula
            use_vector: Vektör kanalını kullan (gömme varsa)
            min_score: Bu skorun altındaki sonuçları at
            include_body: Sonuçlara gövde metnini de ekle
            recency_weight: Tazelik ağırlığını geçici olarak değiştir
            filters: `kind`, `category`, `tags`, `min_confidence` filtreleri
            use_cache: Önbelleği kullan

        Returns:
            Skora göre azalan sırada SearchResult listesi
        """
        start = time.perf_counter()
        self._counter["queries"] += 1
        query = (query or "").strip()
        if not query:
            return []

        analysis = self.analyzer.analyze(query)
        cache_key = self._cache_key(analysis, limit, expand_graph, diversify, use_vector, filters)

        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return self._finalize(cached, include_body, min_score, limit)

        channels: Dict[str, Dict[str, float]] = {}
        payloads: Dict[str, Dict[str, Any]] = {}

        try:
            self._channel_fts(analysis, channels, payloads, limit, filters)
        except Exception as exc:
            self._counter["errors"] += 1
            logger.warning(f"FTS kanalı başarısız: {exc}")

        if use_vector and self._embed_fn is not None:
            try:
                self._channel_vector(analysis, channels, payloads, limit, filters)
            except Exception as exc:
                self._counter["errors"] += 1
                logger.warning(f"Vektör kanalı başarısız: {exc}")

        try:
            self._channel_tag_title(analysis, channels, payloads, limit, filters)
        except Exception as exc:
            self._counter["errors"] += 1
            logger.warning(f"Etiket kanalı başarısız: {exc}")

        if not channels:
            self._counter["total_duration_ms"] += (time.perf_counter() - start) * 1000
            return []

        fused = self._fuse(channels)
        results = self._build_results(fused, payloads, channels)

        if expand_graph and self._graph is not None and results:
            try:
                results = self._expand_with_graph(results, graph_depth)
            except Exception as exc:
                self._counter["errors"] += 1
                logger.warning(f"Graf genişletme başarısız: {exc}")

        results = self._rerank(results, analysis, recency_weight)

        if diversify and len(results) > 2:
            results = self._mmr(results, analysis, limit=max(limit, 4))

        results = results[: max(limit * 2, limit)]
        if use_cache:
            self._cache_put(cache_key, results)

        self._counter["total_results"] += len(results)
        self._counter["total_duration_ms"] += (time.perf_counter() - start) * 1000

        final = self._finalize(results, include_body, min_score, limit)
        self._emit("knowledge.searched", {
            "query": query,
            "keywords": analysis.keywords,
            "intents": analysis.intents,
            "results": len(final),
            "top": [{"id": r.note_id, "title": r.title, "score": round(r.score, 4)}
                    for r in final[:5]],
        })
        return final

    def _finalize(self, results: List[SearchResult], include_body: bool,
                  min_score: float, limit: int) -> List[SearchResult]:
        """Sonuçları kırpar, filtreler ve gövde ekler."""
        selected = [r for r in results if r.score >= min_score][:limit]
        if include_body:
            for result in selected:
                if not result.body:
                    result.body = self._load_body(result.note_id)
        return selected

    # ─── Kanallar ───────────────────────────────────────────────────────

    def _channel_fts(self, analysis: QueryAnalysis,
                     channels: Dict[str, Dict[str, float]],
                     payloads: Dict[str, Dict[str, Any]],
                     limit: int,
                     filters: Optional[Dict[str, Any]] = None) -> None:
        """
        BM25/FTS kanalı.

        `filters` (kind/category/tags/exclude_kinds/…) SQL katmanına iletilir;
        filtreler burada uygulanmazsa bağlam dışlaması (ör. `qa`/`meta` notları)
        sessizce etkisiz kalır.
        """
        if self._index is None:
            return
        self._counter["fts_calls"] += 1
        hits = self._index.search_fts(
            analysis.fts_query(),
            limit=max(limit * 4, 20),
            **(filters or {}),
        )
        if not hits:
            return
        channel = channels.setdefault("fts", {})
        for rank, hit in enumerate(hits, start=1):
            note_id = hit["id"]
            channel[note_id] = max(channel.get(note_id, 0.0), float(hit.get("score", 0.0)))
            payloads.setdefault(note_id, hit)

    def _channel_vector(self, analysis: QueryAnalysis,
                        channels: Dict[str, Dict[str, float]],
                        payloads: Dict[str, Dict[str, Any]],
                        limit: int,
                        filters: Optional[Dict[str, Any]] = None) -> None:
        """Anlamsal (vektör) kanal."""
        if self._index is None or self._embed_fn is None:
            return
        self._counter["vector_calls"] += 1
        vector = self._embed_fn(analysis.original) or self._embed_fn(analysis.normalized)
        if not vector:
            return
        hits = self._index.similar_notes(vector, limit=max(limit * 3, 15), min_score=0.05)
        if not hits:
            return

        # Vektör kanalı tür filtresini SQL'de uygulamaz; burada eliyoruz.
        excluded = {str(k) for k in ((filters or {}).get("exclude_kinds") or [])}
        wanted_kind = (filters or {}).get("kind")
        channel = channels.setdefault("vector", {})
        for hit in hits:
            if excluded and str(hit.get("kind", "")) in excluded:
                continue
            if wanted_kind and str(hit.get("kind", "")) != str(wanted_kind):
                continue
            note_id = hit["id"]
            similarity = float(hit.get("similarity", 0.0))
            channel[note_id] = max(channel.get(note_id, 0.0), similarity)
            payloads.setdefault(note_id, hit)

    def _channel_tag_title(self, analysis: QueryAnalysis,
                           channels: Dict[str, Dict[str, float]],
                           payloads: Dict[str, Dict[str, Any]],
                           limit: int,
                           filters: Optional[Dict[str, Any]] = None) -> None:
        """Etiket ve başlık kanalı (yapısal sinyal)."""
        if self._index is None:
            return
        channel = channels.setdefault("tag", {})

        # Etiket eşleşmesi
        for hint in analysis.tag_hints:
            for tag in normalize_tags([hint]):
                try:
                    notes = self._index.by_tag(tag, limit=10)
                except Exception:
                    continue
                for note in notes:
                    note_id = note["id"]
                    channel[note_id] = max(channel.get(note_id, 0.0), 0.7)
                    payloads.setdefault(note_id, note)

        # Başlık benzerliği
        if self._vault is not None and analysis.keywords:
            try:
                matches = self._vault.search_by_title(
                    analysis.original, limit=8,
                    exclude_kinds=(filters or {}).get("exclude_kinds"),
                )
            except Exception:
                matches = []
            for note, score in matches:
                note_id = note.id
                channel[note_id] = max(channel.get(note_id, 0.0), float(score))
                payloads.setdefault(note_id, {
                    "id": note.id, "title": note.title, "kind": note.kind,
                    "category": note.category, "tags": list(note.tags),
                    "confidence": note.confidence, "importance": note.importance,
                    "updated": note.updated_ts, "path": note.path,
                    "excerpt": note.excerpt, "word_count": note.word_count,
                })

    # ─── Kaynaştırma ────────────────────────────────────────────────────

    @staticmethod
    def _fuse(channels: Dict[str, Dict[str, float]], k: int = 60) -> Dict[str, Tuple[float, Dict[str, float]]]:
        """
        Reciprocal Rank Fusion ile kanalları birleştirir.

        RRF her kanalı kendi sıralamasına göre ağırlıklandırır; bu yüzden
        farklı ölçeklerdeki skorlar (BM25 vs kosinüs) uyumlu biçimde
        birleştirilebilir.

        Args:
            channels: {kanal_adı: {not_id: skor}}
            k: RRF yumuşatma sabiti (60 literatür değeri)

        Returns:
            {not_id: (birleşik_skor, {kanal: skor})}
        """
        scores: Dict[str, float] = {}
        per_channel: Dict[str, Dict[str, float]] = {}

        # Kanal ağırlıkları — vektör anlamsal olarak daha değerli
        channel_weights = {"fts": 1.0, "vector": 1.15, "tag": 0.8}

        for name, channel in channels.items():
            if not channel:
                continue
            weight = channel_weights.get(name, 0.7)
            ordered = sorted(channel.items(), key=lambda item: -item[1])
            for rank, (note_id, raw_score) in enumerate(ordered, start=1):
                rrf = weight / (k + rank)
                scores[note_id] = scores.get(note_id, 0.0) + rrf
                per_channel.setdefault(note_id, {})[name] = float(raw_score)

        if not scores:
            return {}
        # 0..1 aralığına normalize et
        best = max(scores.values()) or 1.0
        return {
            note_id: (score / best, per_channel.get(note_id, {}))
            for note_id, score in scores.items()
        }

    def _build_results(self, fused: Dict[str, Tuple[float, Dict[str, float]]],
                       payloads: Dict[str, Dict[str, Any]],
                       channels: Dict[str, Dict[str, float]]) -> List[SearchResult]:
        """Kaynaştırılmış skorları SearchResult nesnelerine çevirir."""
        results: List[SearchResult] = []
        for note_id, (score, channel_scores) in fused.items():
            payload = payloads.get(note_id)
            if payload is None:
                payload = self._load_payload(note_id)
            if payload is None:
                continue
            updated = payload.get("updated", 0.0)
            if isinstance(updated, str):
                from neurocore.knowledge.note import parse_timestamp
                updated = parse_timestamp(updated)
            primary = max(channel_scores.items(), key=lambda item: item[1])[0] if channel_scores else ""
            results.append(SearchResult(
                note_id=note_id,
                title=str(payload.get("title") or note_id),
                score=float(score),
                kind=str(payload.get("kind") or "concept"),
                category=str(payload.get("category") or "learned"),
                tags=[str(t) for t in (payload.get("tags") or [])],
                confidence=float(payload.get("confidence", 0.5) or 0.5),
                importance=float(payload.get("importance", 0.5) or 0.5),
                updated=float(updated or 0.0),
                path=str(payload.get("path") or ""),
                excerpt=str(payload.get("excerpt") or ""),
                source=primary,
                channel_scores=dict(channel_scores),
            ))
        results.sort(key=lambda item: -item.score)
        return results

    # ─── Graf Genişletme ────────────────────────────────────────────────

    def _expand_with_graph(self, results: List[SearchResult], depth: int) -> List[SearchResult]:
        """
        Sonuç kümesini graf komşularıyla genişletir.

        Komşular `graph` kanalıyla işaretlenir ve daha düşük başlangıç skoru
        alır (doğrudan eşleşmeler her zaman önde kalır).
        """
        if self._graph is None or depth <= 0:
            return results
        self._counter["graph_expansions"] += 1

        seeds = [result.note_id for result in results[:6]]
        existing = {result.note_id for result in results}
        additions: List[SearchResult] = []
        decay = 0.45

        for seed in seeds:
            seed_result = next((r for r in results if r.note_id == seed), None)
            if seed_result is None:
                continue
            try:
                neighbors = self._graph.neighbors(seed, depth=depth)
            except Exception:
                continue
            for hop, neighbor in enumerate(neighbors, start=1):
                if neighbor in existing:
                    continue
                payload = self._load_payload(neighbor)
                if payload is None:
                    continue
                score = seed_result.score * (decay / hop)
                additions.append(SearchResult(
                    note_id=neighbor,
                    title=str(payload.get("title") or neighbor),
                    score=score,
                    kind=str(payload.get("kind") or "concept"),
                    category=str(payload.get("category") or "learned"),
                    tags=[str(t) for t in (payload.get("tags") or [])],
                    confidence=float(payload.get("confidence", 0.5) or 0.5),
                    importance=float(payload.get("importance", 0.5) or 0.5),
                    updated=float(payload.get("updated", 0.0) or 0.0),
                    path=str(payload.get("path") or ""),
                    excerpt=str(payload.get("excerpt") or ""),
                    source="graph",
                    channel_scores={"graph": score},
                    via=seed_result.title,
                    hop=hop,
                ))
                existing.add(neighbor)

        combined = results + additions
        combined.sort(key=lambda item: -item.score)
        return combined

    # ─── Yeniden Sıralama ───────────────────────────────────────────────

    def _rerank(self, results: List[SearchResult], analysis: QueryAnalysis,
                recency_weight: Optional[float] = None) -> List[SearchResult]:
        """
        Sonuçları çok kriterli skorla yeniden sıralar.

        Bileşenler: kaynaştırılmış alaka, sorgu kapsamı, not güveni, önem,
        tazelik ve graf merkeziliği (PageRank).
        """
        if not results:
            return results
        weights = RerankWeights(**self.weights.to_dict())
        if recency_weight is not None:
            weights.recency = float(recency_weight)

        query_terms = set(analysis.keywords) or set(analysis.all_words)
        now = utc_timestamp()
        max_centrality = max(self._pagerank.values(), default=0.0) or 1.0

        for result in results:
            haystack = fold_turkish(f"{result.title} {result.excerpt} {result.body}").lower()
            if query_terms:
                matched = sum(1 for term in query_terms if term in haystack)
                coverage = matched / len(query_terms)
            else:
                coverage = 0.5

            age_days = max(0.0, (now - result.updated) / 86400.0) if result.updated else 365.0
            recency = 0.5 ** (age_days / max(1.0, weights.recency_half_life_days))
            centrality = min(1.0, self._pagerank.get(result.note_id, 0.0) / max_centrality)

            score = (
                weights.relevance * result.score
                + weights.coverage * coverage
                + weights.confidence * result.confidence
                + weights.importance * result.importance
                + weights.recency * recency
                + weights.centrality * centrality
            )
            # Graf üzerinden gelen sonuçlar hafif ceza alır (dolaylı kanıt)
            if result.source == "graph":
                score *= 0.9
            result.score = min(1.0, score)

        results.sort(key=lambda item: -item.score)
        return results

    def _mmr(self, results: List[SearchResult], analysis: QueryAnalysis,
             limit: int, lambda_param: float = 0.72) -> List[SearchResult]:
        """
        Maximal Marginal Relevance ile çeşitlilik uygular.

        Benzer notların (aynı konu, düşük bilgi kazancı) art arda gelmesini
        engeller. Benzerlik başlık + etiket örtüşmesiyle yaklaşık hesaplanır.

        Args:
            limit: Kaç sonuç seçilecek
            lambda_param: 1.0 = yalnızca alaka, 0.0 = yalnızca çeşitlilik
        """
        if len(results) <= 2:
            return results

        selected: List[SearchResult] = [results[0]]
        remaining = results[1:]

        while remaining and len(selected) < limit:
            best_index = 0
            best_value = -1e9
            for index, candidate in enumerate(remaining):
                relevance = candidate.score
                redundancy = max(
                    self._pairwise_similarity(candidate, chosen) for chosen in selected
                )
                value = lambda_param * relevance - (1.0 - lambda_param) * redundancy
                if value > best_value:
                    best_value = value
                    best_index = index
            selected.append(remaining.pop(best_index))
        return selected

    @staticmethod
    def _pairwise_similarity(a: SearchResult, b: SearchResult) -> float:
        """İki sonucun yaklaşık benzerliği (başlık + etiket)."""
        title_score = title_similarity(a.title, b.title)
        tags_a, tags_b = set(a.tags), set(b.tags)
        tag_score = (len(tags_a & tags_b) / len(tags_a | tags_b)) if (tags_a | tags_b) else 0.0
        same_kind = 0.15 if a.kind == b.kind else 0.0
        return min(1.0, 0.5 * title_score + 0.4 * tag_score + same_kind)

    # ─── Yardımcı Sorgular ──────────────────────────────────────────────

    def related_notes(self, note_id: str, limit: int = 8) -> List[SearchResult]:
        """Bir notla ilişkili notları bulur (graf + vektör)."""
        results: List[SearchResult] = []
        seen: Set[str] = {note_id}

        if self._graph is not None:
            try:
                for neighbor in self._graph.neighbors(note_id, depth=1)[:limit]:
                    payload = self._load_payload(neighbor)
                    if not payload or neighbor in seen:
                        continue
                    seen.add(neighbor)
                    results.append(self._payload_to_result(payload, 1.0, "graph"))
            except Exception as exc:
                logger.debug(f"İlişkili not sorgusu başarısız: {exc}")

        if len(results) < limit and self._index is not None and self._embed_fn is not None:
            try:
                payload = self._load_payload(note_id, include_body=True)
                body = str((payload or {}).get("body") or (payload or {}).get("excerpt") or "")
                if body:
                    vector = self._embed_fn(body)
                    if vector:
                        for hit in self._index.similar_notes(vector, limit=limit * 2, min_score=0.2):
                            if hit["id"] in seen:
                                continue
                            seen.add(hit["id"])
                            results.append(self._payload_to_result(hit, float(hit.get("similarity", 0.5)), "vector"))
            except Exception as exc:
                logger.debug(f"Vektör komşuluğu başarısız: {exc}")

        results.sort(key=lambda item: -item.score)
        return results[:limit]

    def suggest_queries(self, note_id: str, limit: int = 5) -> List[str]:
        """Bir nottan yola çıkarak araştırılabilecek soru önerileri üretir."""
        payload = self._load_payload(note_id, include_body=True)
        if not payload:
            return []
        title = str(payload.get("title") or "")
        suggestions: List[str] = []
        if payload.get("has_todos") or "TODO" in str(payload.get("body") or ""):
            suggestions.append(f"{title} konusundaki eksik bilgiler nelerdir?")
        suggestions += [
            f"{title} nedir ve neden önemlidir?",
            f"{title} nasıl uygulanır?",
            f"{title} ile ilgili yaygın hatalar nelerdir?",
            f"{title} hangi durumlarda işe yaramaz?",
            f"{title} konusunda güncel yaklaşımlar nelerdir?",
        ]
        return suggestions[:limit]

    def explain_ranking(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Sorgu için sıralama gerekçelerini döndürür (hata ayıklama/şeffaflık).
        """
        analysis = self.analyzer.analyze(query)
        results = self.search(query, limit=limit, include_body=False)
        return [
            {
                "rank": index,
                "title": result.title,
                "score": round(result.score, 5),
                "channels": result.channel_scores,
                "source": result.source,
                "via": result.via,
                "analysis": analysis.to_dict() if index == 1 else None,
            }
            for index, result in enumerate(results, start=1)
        ]

    # ─── İç Yardımcılar ─────────────────────────────────────────────────

    def refresh_centrality(self) -> Dict[str, float]:
        """PageRank önbelleğini graf üzerinden tazeler."""
        if self._graph is None:
            self._pagerank = {}
            return {}
        try:
            self._pagerank = dict(self._graph.pagerank())
        except Exception as exc:
            logger.debug(f"PageRank hesaplanamadı: {exc}")
            self._pagerank = {}
        return self._pagerank

    def _load_payload(self, note_id: str, include_body: bool = False) -> Optional[Dict[str, Any]]:
        """Not yükünü indeksten, yoksa kasadan getirir."""
        if self._index is not None:
            try:
                payload = self._index.get_note(note_id, include_body=include_body)
                if payload:
                    return payload
            except Exception:
                pass
        if self._vault is not None:
            try:
                note = self._vault.read_note(note_id)
                if note:
                    data = note.to_dict(include_body=include_body)
                    data["updated"] = note.updated_ts
                    return data
            except Exception:
                pass
        return None

    def _load_body(self, note_id: str) -> str:
        """Notun tam gövdesini getirir."""
        if self._vault is not None:
            try:
                note = self._vault.read_note(note_id)
                if note:
                    return note.body
            except Exception:
                pass
        payload = self._load_payload(note_id, include_body=True)
        return str((payload or {}).get("body") or "")

    @staticmethod
    def _payload_to_result(payload: Dict[str, Any], score: float, source: str) -> SearchResult:
        """Yük sözlüğünü SearchResult'a çevirir."""
        updated = payload.get("updated", 0.0)
        if isinstance(updated, str):
            from neurocore.knowledge.note import parse_timestamp
            updated = parse_timestamp(updated)
        return SearchResult(
            note_id=str(payload.get("id") or ""),
            title=str(payload.get("title") or ""),
            score=float(score),
            kind=str(payload.get("kind") or "concept"),
            category=str(payload.get("category") or "learned"),
            tags=[str(t) for t in (payload.get("tags") or [])],
            confidence=float(payload.get("confidence", 0.5) or 0.5),
            importance=float(payload.get("importance", 0.5) or 0.5),
            updated=float(updated or 0.0),
            path=str(payload.get("path") or ""),
            excerpt=str(payload.get("excerpt") or ""),
            source=source,
            channel_scores={source: float(score)},
        )

    # ─── Önbellek ───────────────────────────────────────────────────────

    def _cache_key(self, analysis: QueryAnalysis, limit: int, expand: bool,
                   diversify: bool, use_vector: bool,
                   filters: Optional[Dict[str, Any]]) -> str:
        """Önbellek anahtarı üretir."""
        import hashlib
        filter_part = ""
        if filters:
            filter_part = "|".join(f"{k}={sorted(v) if isinstance(v, (list, set, tuple)) else v}"
                                   for k, v in sorted(filters.items()))
        raw = f"{analysis.normalized.lower()}|{limit}|{expand}|{diversify}|{use_vector}|{filter_part}"
        return hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()

    def _cache_get(self, key: str) -> Optional[List[SearchResult]]:
        """Önbellekten sonuç okur (TTL kontrollü)."""
        cache = getattr(self, "_cache", None)
        if not cache:
            return None
        entry = cache.get(key)
        if not entry:
            return None
        stamp, results = entry
        if (time.time() - stamp) > self._cache_ttl:
            cache.pop(key, None)
            return None
        self._counter["cache_hits"] += 1
        return results

    def _cache_put(self, key: str, results: List[SearchResult]) -> None:
        """Önbelleğe yazar (LRU benzeri budama)."""
        cache = getattr(self, "_cache", None)
        if cache is None:
            return
        if len(cache) >= self._cache_size:
            # En eski %10'u at
            oldest = sorted(cache.items(), key=lambda item: item[1][0])[: max(1, self._cache_size // 10)]
            for old_key, _ in oldest:
                cache.pop(old_key, None)
        cache[key] = (time.time(), results)

    def clear_cache(self) -> int:
        """Önbelleği temizler; silinen kayıt sayısını döndürür."""
        cache = getattr(self, "_cache", None) or {}
        count = len(cache)
        cache.clear()
        return count

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
            f"<HybridSearch index={self._index is not None} "
            f"graph={self._graph is not None} queries={self._counter['queries']}>"
        )


# ─── Bağlam Oluşturucu ──────────────────────────────────────────────────────

class ContextBuilder:
    """
    Arama sonuçlarından model için bağlam metni üretir.

    Token bütçesine sığacak biçimde sonuçları seçer, her notu başlık +
    özet + atıf ile biçimlendirir. Ayrıca kaynak listesi ve güven özeti
    çıkarır — modelin atıf yapabilmesi için gereklidir.
    """

    def __init__(self, max_chars: int = 6000, chars_per_token: float = 3.6) -> None:
        """
        Args:
            max_chars: Bağlamın maksimum karakter sayısı
            chars_per_token: Kaba token→karakter oranı (Türkçe için ~3.6)
        """
        self.max_chars = max_chars
        self.chars_per_token = chars_per_token

    def build(self, results: Sequence[SearchResult], query: str = "",
              include_sources: bool = True) -> Dict[str, Any]:
        """
        Bağlam bloğunu oluşturur.

        Returns:
            {"text": str, "used": [SearchResult...], "dropped": [str...],
             "tokens_estimate": int, "confidence": float, "sources": [str...]}
        """
        if not results:
            return {
                "text": "", "used": [], "dropped": [],
                "tokens_estimate": 0, "confidence": 0.0, "sources": [],
            }

        header = f"## Bağlam (sorgu: {query})\n" if query else "## Bağlam\n"
        blocks: List[str] = [header]
        used: List[SearchResult] = []
        dropped: List[str] = []
        total = len(header)

        for result in results:
            body = result.body or result.excerpt
            snippet = body[: max(400, self.max_chars // max(1, len(results)))]
            block = (
                f"### {result.title}\n"
                f"- id: `{result.note_id}` · güven: {result.confidence:.2f} · "
                f"etiketler: {', '.join(result.tags[:6]) or '-'}"
                + (f" · kaynak: graf ({result.via})" if result.via else "")
                + f"\n\n{snippet}\n"
            )
            if total + len(block) > self.max_chars and used:
                dropped.append(result.title)
                continue
            blocks.append(block)
            used.append(result)
            total += len(block)

        sources: List[str] = []
        if include_sources and used:
            # Not: kaynaklar düz metin olarak yazılır (markdown liste işareti yok);
            # tüketici katman kendi biçimlendirmesini uygular.
            sources = [f"[[{r.note_id}\\|{r.title}]] (skor {r.score:.2f})" for r in used]
            blocks.append("## Kaynaklar\n" + "\n".join(sources) + "\n")

        # Ağırlıklı güven: yüksek skorlu sonuçların güveni daha çok sayılır
        weight_total = sum(max(0.001, r.score) for r in used) or 1.0
        confidence = sum(r.confidence * max(0.001, r.score) for r in used) / weight_total

        text = "\n".join(blocks)
        return {
            "text": text,
            "used": used,
            "dropped": dropped,
            "tokens_estimate": int(len(text) / self.chars_per_token),
            "confidence": round(confidence, 4),
            "sources": sources,
        }

    def build_compact(self, results: Sequence[SearchResult], max_items: int = 5) -> str:
        """Tek satırlık özet liste (log ve istem başlıkları için)."""
        lines = []
        for result in list(results)[:max_items]:
            lines.append(f"- {result.title} ({result.score:.2f})")
        return "\n".join(lines)


__all__ = [
    "HybridSearch",
    "SearchResult",
    "QueryAnalyzer",
    "QueryAnalysis",
    "RerankWeights",
    "ContextBuilder",
    "SearchError",
    "STOPWORDS",
]
