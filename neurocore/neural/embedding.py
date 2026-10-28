"""
NeuroCore Gömme Motoru (Embedding)
===================================
Metni sabit boyutlu vektörlere çeviren çok stratejili gömme motoru.

Stratejiler (otomatik seçilir, en iyiden başlanır):
    1. ``learned``  — eğitilmiş omurga modelinin havuzlanmış gizli durumu
    2. ``local``    — diskteki bir Sentence-Transformers modeli (varsa)
    3. ``hash``     — deterministik hash tabanlı gömme (her zaman çalışır)

Neden hash yedeği?
    Anlamsal arama, model eğitilmeden de çalışmalıdır. Hash gömme
    kelime/karakter n-gram örtüşmesini yakalar; BPE tokenizasyonu ile
    birleştiğinde Türkçe eklerine karşı beklenmedik ölçüde dayanıklıdır:
    "sıcaklık" ↔ "sıcaklığı" yüksek benzerlik verir.

Vektörler **L2-normalize** edilir; böylece kosinüs benzerliği tek bir
iç çarpıma indirgenir ve önbellek karşılaştırmaları hızlanır.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.embedding")


# ─── Sabitler ───────────────────────────────────────────────────────────────

#: Gömme stratejileri
STRATEGY_LEARNED = "learned"
STRATEGY_LOCAL = "local"
STRATEGY_HASH = "hash"

#: Hash gömmede kaydırma (projeksiyon) için sabit tohum
HASH_PROJECTION_SEED = 20261005

_WORD_RE = re.compile(r"[\wÇĞİÖŞÜçğıöşü]+", re.UNICODE)

#: Türkçe gömme için ek kırpma listesi (hash çakışmasını artırır)
_TR_SUFFIXES: Tuple[str, ...] = (
    "larindan", "lerinden", "larımız", "lerimiz", "lardan", "lerden",
    "ların", "lerin", "sıyla", "siyle", "ları", "leri", "dan", "den",
    "tan", "ten", "nin", "nın", "nun", "nün", "lar", "ler", "sı", "si",
    "ın", "in", "un", "ün", "da", "de", "ta", "te", "ya", "ye", "lı", "li",
)

_STOPWORDS_TR_EN: frozenset = frozenset({
    "ve", "ile", "için", "bir", "bu", "şu", "da", "de", "ki", "mi", "ne",
    "çok", "daha", "en", "gibi", "kadar", "ama", "veya", "ise", "olan",
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "is", "are", "was", "be", "with", "by", "from", "as", "it", "this",
})


class EmbeddingError(Exception):
    """Gömme hatası."""


# ─── Yardımcı Fonksiyonlar ──────────────────────────────────────────────────

def l2_normalize(vector: Sequence[float]) -> List[float]:
    """Vektörü L2-normalize eder (sıfır vektör olduğu gibi döner)."""
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 1e-12:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """İki vektör arasındaki kosinüs benzerliği."""
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = norm_a = norm_b = 0.0
    for index in range(length):
        va, vb = float(a[index]), float(b[index])
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / math.sqrt(norm_a * norm_b)))


def _stable_hash_int(text: str, salt: str = "") -> int:
    """Metnin deterministik 64-bit hash'i (Python `hash()` KULLANILMAZ)."""
    digest = hashlib.blake2b(f"{salt}\x1f{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def stem_turkish(word: str) -> str:
    """Türkçe ekleri kabaca kırpar (en az 3 karakter kök kalır)."""
    if len(word) <= 4:
        return word
    for suffix in _TR_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str, stem: bool = True, keep_stopwords: bool = False) -> List[str]:
    """
    Metni normalize edilmiş token'lara ayırır.

    Args:
        text: Girdi
        stem: Türkçe ekleri kırp
        keep_stopwords: Durak kelimeleri koru
    """
    if not text:
        return []
    words = _WORD_RE.findall(text.lower())
    result: List[str] = []
    for word in words:
        if not keep_stopwords and word in _STOPWORDS_TR_EN:
            continue
        result.append(stem_turkish(word) if stem else word)
    return result


def hash_embed(
    text: str,
    dim: int = 768,
    *,
    salt: str = "neurocore-embed",
    use_bigrams: bool = True,
    use_char_ngrams: bool = True,
    idf: Optional[Dict[str, float]] = None,
) -> List[float]:
    """
    Deterministik hash tabanlı gömme (bağımlılıksız, model gerektirmez).

    Yöntem: "hashing trick" — her token (ve bigram/trigram) iki hash ile
    (işaret ve indeks) vektöre katkı verir. Katkı ağırlığı ``1 + log(tf)``
    ile ölçeklenir; IDF verilirse kelime önemine göre ağırlıklandırılır.

    Args:
        text: Girdi metni
        dim: Vektör boyutu
        salt: Hash tuzu (farklı uzaylar için farklı değer)
        use_bigrams: Kelime ikililerini ekle
        use_char_ngrams: Karakter 3-gram'larını ekle (yazım hatalarına dayanıklılık)
        idf: {token: idf} — verilirse TF-IDF ağırlıklandırması

    Returns:
        L2-normalize edilmiş `dim` boyutlu vektör
    """
    if dim <= 0:
        raise EmbeddingError(f"Geçersiz vektör boyutu: {dim}")

    vector = [0.0] * dim
    if not text or not text.strip():
        return vector

    tokens = tokenize(text, stem=True)
    if not tokens:
        tokens = tokenize(text, stem=False, keep_stopwords=True)
    if not tokens:
        tokens = [text.strip().lower()[:32]]

    # ── Tekil kelimeler (TF ağırlıklı) ──
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    def add_feature(feature: str, weight: float) -> None:
        """Bir özelliği vektöre işaretli olarak ekler."""
        index = _stable_hash_int(feature, f"{salt}:idx") % dim
        sign = 1.0 if (_stable_hash_int(feature, f"{salt}:sign") & 1) == 0 else -1.0
        vector[index] += sign * weight

    for token, count in counts.items():
        weight = 1.0 + math.log(count)
        if idf and token in idf:
            weight *= idf[token]
        add_feature(token, weight)

    # ── Kelime ikilileri (bağlam sinyali) ──
    if use_bigrams and len(tokens) > 1:
        for index in range(len(tokens) - 1):
            add_feature(f"{tokens[index]}_{tokens[index + 1]}", 0.6)

    # ── Karakter 3-gram'ları (yazım/morfoloji toleransı) ──
    if use_char_ngrams:
        joined = " ".join(tokens)[:2000]
        for index in range(len(joined) - 2):
            add_feature(f"c3:{joined[index:index + 3]}", 0.25)

    # ── Yapısal skalerler (3 boyuta sıkıştırılmış) ──
    if dim >= 3:
        length_bucket = min(1.0, len(text) / 2000.0)
        digit_ratio = (sum(1 for char in text if char.isdigit()) / len(text)) if text else 0.0
        upper_ratio = (sum(1 for char in text if char.isupper()) / len(text)) if text else 0.0
        vector[dim - 3] += length_bucket * 0.4
        vector[dim - 2] += digit_ratio * 0.4
        vector[dim - 1] += upper_ratio * 0.4

    return l2_normalize(vector)


# ─── Önbellek ───────────────────────────────────────────────────────────────

class EmbeddingCache:
    """LRU gömme önbelleği (metin → vektör)."""

    def __init__(self, max_size: int = 4096) -> None:
        self._store: "OrderedDict[str, List[float]]" = OrderedDict()
        self.max_size = max_size
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def key(self, text: str, strategy: str, dim: int) -> str:
        """Önbellek anahtarı."""
        digest = hashlib.blake2b(
            f"{strategy}|{dim}|{text}".encode("utf-8"), digest_size=16
        ).hexdigest()
        return digest

    def get(self, key: str) -> Optional[List[float]]:
        """Önbellekten okur (LRU sırasını günceller)."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return list(self._store[key])
            self.misses += 1
            return None

    def put(self, key: str, vector: Sequence[float]) -> None:
        """Önbelleğe yazar."""
        with self._lock:
            self._store[key] = [float(value) for value in vector]
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> int:
        """Önbelleği boşaltır."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def stats(self) -> Dict[str, Any]:
        """Önbellek istatistikleri."""
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }


# ─── Öğrenilmiş Ağırlık Matrisi ─────────────────────────────────────────────

@dataclass
class ProjectionMatrix:
    """
    Deterministik rastgele projeksiyon matrisi (hash vektörünü hedef boyuta taşır).

    Sabit tohumla üretilir; kaydedilip yüklenebilir. Yalnızca boyut
    uyuşmazlığında (ör. profil değiştiğinde) yeniden üretilir.
    """
    in_dim: int
    out_dim: int
    seed: int = HASH_PROJECTION_SEED
    rows: List[List[float]] = field(default_factory=list)

    def ensure(self) -> None:
        """Matrisi (gerekiyorsa) üretir."""
        if self.rows and len(self.rows) == self.out_dim and self.rows[0] and len(self.rows[0]) == self.in_dim:
            return
        self.rows = self._generate()

    def _generate(self) -> List[List[float]]:
        """Ortogonal-ish rastgele matris üretir (Johnson-Lindenstrauss)."""
        try:
            import numpy as np
            rng = np.random.default_rng(self.seed)
            matrix = rng.normal(0.0, 1.0 / math.sqrt(self.out_dim), size=(self.out_dim, self.in_dim))
            self.rows = matrix.tolist()
            self._load_numpy = True
            return self.rows
        except ImportError:
            pass

        # numpy yoksa: basit doğrusal eşlemeli sözde-rastgele matris
        rows: List[List[float]] = []
        scale = 1.0 / math.sqrt(self.out_dim)
        for row_index in range(self.out_dim):
            row: List[float] = []
            base = _stable_hash_int(f"proj:{self.seed}:{row_index}", "matrix")
            for column in range(self.in_dim):
                mixed = _stable_hash_int(f"{base}:{column}", "cell")
                value = ((mixed % 2000001) / 1000000.0) - 1.0
                row.append(value * scale)
            rows.append(row)
        return rows

    def project(self, vector: Sequence[float]) -> List[float]:
        """Vektörü hedef boyuta yansıtır."""
        self.ensure()
        if not self.rows:
            return list(vector)
        if len(vector) == self.out_dim:
            return [float(value) for value in vector]
        source = list(vector) + [0.0] * max(0, self.in_dim - len(vector))
        result: List[float] = []
        for row in self.rows:
            total = 0.0
            for index in range(min(len(row), len(source))):
                total += row[index] * source[index]
            result.append(total)
        return l2_normalize(result)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        self.ensure()
        return {"in_dim": self.in_dim, "out_dim": self.out_dim,
                "seed": self.seed, "rows": self.rows}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectionMatrix":
        """JSON'dan yükler."""
        matrix = cls(
            in_dim=int(data.get("in_dim", 0) or 0),
            out_dim=int(data.get("out_dim", 1) or 1),
            seed=int(data.get("seed", HASH_PROJECTION_SEED) or HASH_PROJECTION_SEED),
        )
        rows = data.get("rows") or []
        if rows:
            matrix.rows = [[float(value) for value in row] for row in rows]
        return matrix


# ─── Ana Gömme Motoru ───────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Çok stratejili gömme motoru.

    Kullanım:
        embed = EmbeddingEngine()
        embed.initialize(config, event_bus)

        vector = embed.embed("GPU sıcaklığı neden yükselir?")
        embed.attach_model(neural_model, tokenizer)      # öğrenilmiş gömmeyi açar
        score = embed.similarity("termal macun", "ısı iletken macun")
        embed.fit_idf(kasa_metinleri)                    # IDF ağırlıklandırması

    Motor her zaman çalışır: model yoksa hash gömme, varsa öğrenilmiş gömme.
    """

    name = "neural.embedding"

    def __init__(self) -> None:
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._lock = threading.RLock()

        self._dim = 768
        self._hash_dim = 768
        self._strategy = STRATEGY_HASH
        self._cache = EmbeddingCache()
        self._projection: Optional[ProjectionMatrix] = None
        self._idf: Dict[str, float] = {}
        self._idf_ready = False

        # Bağlanan bileşenler (nöral katman tarafından enjekte edilir)
        self._model: Any = None
        self._tokenizer: Any = None
        self._local_model: Any = None       # sentence-transformers (opsiyonel)

        self._counter = {
            "embeddings": 0, "cache_hits": 0, "fallbacks": 0,
            "errors": 0, "total_chars": 0, "total_ms": 0.0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Gömme motorunu başlatır (idempotent)."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus

        neural = getattr(config, "neural", None)
        self._dim = int(getattr(neural, "embedding_dim", 768) or 768)
        # Hash uzayı her zaman 768 boyutlu üretilir; hedef boyut farklıysa
        # sabit projeksiyonla taşınır (deterministik, kaydedilebilir).
        self._hash_dim = 768
        if self._dim != self._hash_dim:
            self._projection = ProjectionMatrix(in_dim=self._hash_dim, out_dim=self._dim)
            self._projection.ensure()

        cache_size = int(getattr(neural, "memory_bank_size", 2048) or 2048)
        self._cache.max_size = max(512, cache_size)

        self._initialized = True
        logger.info(
            f"Gömme motoru hazır — boyut {self._dim}, strateji '{self._strategy}'"
        )

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır."""
        self._cache.clear()
        self._local_model = None
        self._initialized = False
        logger.info(
            f"Gömme motoru kapatıldı ({self._counter['embeddings']} gömme, "
            f"önbellek isabeti {self._cache.stats()['hit_rate']:.0%})"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Gömme istatistikleri."""
        avg_ms = (self._counter["total_ms"] / self._counter["embeddings"]
                  if self._counter["embeddings"] else 0.0)
        return {
            **self._counter,
            "dim": self._dim,
            "strategy": self._strategy,
            "avg_ms": round(avg_ms, 3),
            "idf_terms": len(self._idf),
            "model_attached": self._model is not None,
            "local_model_attached": self._local_model is not None,
            "cache": self._cache.stats(),
            "initialized": self._initialized,
        }

    @property
    def dim(self) -> int:
        """Vektör boyutu."""
        return self._dim

    @property
    def strategy(self) -> str:
        """Etkin gömme stratejisi."""
        return self._strategy

    # ─── Bileşen Bağlama ────────────────────────────────────────────────

    def attach_model(self, model: Any, tokenizer: Any = None) -> None:
        """
        Öğrenilmiş gömmeyi açar: omurga modeli + tokenizer.

        Args:
            model: `NeuroCoreModel` benzeri, `embed_tokens(ids)` sunan nesne
            tokenizer: `BPETokenizer` benzeri
        """
        self._model = model
        self._tokenizer = tokenizer or self._tokenizer
        self._strategy = STRATEGY_LEARNED
        self._cache.clear()
        logger.info("Öğrenilmiş gömme etkinleştirildi")

    def detach_model(self) -> None:
        """Öğrenilmiş gömmeyi kapatır (hash gömmeye döner)."""
        self._model = None
        self._strategy = STRATEGY_LOCAL if self._local_model is not None else STRATEGY_HASH
        self._cache.clear()

    def attach_tokenizer(self, tokenizer: Any) -> None:
        """Tokenizer'ı bağlar (hash gömmede kelime bölütlemesi için kullanılır)."""
        self._tokenizer = tokenizer

    def load_local_model(self, path: Optional[str] = None, model_name: str = "") -> bool:
        """
        Yerel bir Sentence-Transformers modeli yükler (opsiyonel).

        Model yoksa/kurulu değilse sessizce False döner — sistem hash
        gömmeye devam eder. İnternetten indirme **yapılmaz** (çevrimdışı
        öncelikli); yalnızca diskteki model yüklenir.

        Args:
            path: Model dizini (None ise `<base_path>/models/embeddings/`)
            model_name: Alt dizin adı

        Returns:
            Yüklendiyse True
        """
        base = path
        if base is None:
            base_path = getattr(self._config, "base_path", None) or os.getcwd()
            base = str(Path(base_path) / "models" / "embeddings")
            if model_name:
                base = str(Path(base) / model_name)

        target = Path(base)
        if not target.exists():
            logger.debug(f"Yerel gömme modeli bulunamadı: {target}")
            return False

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            logger.info("sentence-transformers kurulu değil — hash gömme kullanılacak")
            return False

        try:
            self._local_model = SentenceTransformer(str(target), device="cpu")
            self._dim = int(self._local_model.get_sentence_embedding_dimension())
            self._strategy = STRATEGY_LOCAL
            self._cache.clear()
            logger.info(f"Yerel gömme modeli yüklendi: {target} (boyut {self._dim})")
            return True
        except Exception as exc:
            logger.warning(f"Yerel gömme modeli yüklenemedi: {exc}")
            return False

    def fit_idf(self, documents: Iterable[str], max_documents: int = 5000) -> int:
        """
        Hash gömme için IDF ağırlıklarını öğrenir.

        Nadir kelimeler daha yüksek ağırlık alır; bu, arama kalitesini
        belirgin biçimde artırır (özellikle Türkçe teknik metinlerde).

        Args:
            documents: Metin koleksiyonu
            max_documents: İşlenecek maksimum belge

        Returns:
            Sözlükteki terim sayısı
        """
        document_count = 0
        df: Dict[str, int] = {}
        for index, document in enumerate(documents):
            if index >= max_documents:
                break
            document_count += 1
            for term in set(tokenize(str(document))):
                df[term] = df.get(term, 0) + 1

        if document_count == 0 or not df:
            return 0

        self._idf = {
            term: math.log((1 + document_count) / (1 + count)) + 1.0
            for term, count in df.items()
        }
        self._idf_ready = True
        self._cache.clear()
        logger.info(
            f"IDF öğrenildi: {len(self._idf)} terim / {document_count} belge"
        )
        return len(self._idf)

    def save_state(self, path: str) -> str:
        """IDF ve projeksiyon durumunu kaydeder."""
        import json
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "dim": self._dim,
            "hash_dim": self._hash_dim,
            "idf": self._idf,
            "projection": self._projection.to_dict() if self._projection else None,
        }
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, target)
        return str(target)

    def load_state(self, path: str) -> bool:
        """IDF ve projeksiyon durumunu yükler."""
        import json
        target = Path(path)
        if not target.exists():
            return False
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Gömme durumu yüklenemedi: {exc}")
            return False
        self._idf = {str(k): float(v) for k, v in (payload.get("idf") or {}).items()}
        self._idf_ready = bool(self._idf)
        projection = payload.get("projection")
        if projection:
            self._projection = ProjectionMatrix.from_dict(projection)
        self._cache.clear()
        return True

    # ─── Gömme ──────────────────────────────────────────────────────────

    def embed(self, text: str, *, use_cache: bool = True,
              strategy: Optional[str] = None) -> List[float]:
        """
        Metni vektöre çevirir.

        Args:
            text: Girdi metni
            use_cache: Önbelleği kullan
            strategy: Belirli bir strateji zorla (`hash`, `learned`, `local`)

        Returns:
            L2-normalize edilmiş vektör (`dim` boyutlu)
        """
        start = time.perf_counter()
        chosen = strategy or self._strategy
        payload = text or ""

        cache_key = self._cache.key(payload, chosen, self._dim)
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._counter["cache_hits"] += 1
                return cached

        vector: Optional[List[float]] = None

        if chosen == STRATEGY_LEARNED and self._model is not None:
            vector = self._embed_learned(payload)
            if vector is None:
                self._counter["fallbacks"] += 1

        if vector is None and chosen == STRATEGY_LOCAL and self._local_model is not None:
            vector = self._embed_local(payload)
            if vector is None:
                self._counter["fallbacks"] += 1

        if vector is None:
            vector = self._embed_hash(payload)
            if chosen != STRATEGY_HASH:
                self._counter["fallbacks"] += 1

        if len(vector) != self._dim:
            vector = self._fit_dim(vector)

        if use_cache:
            self._cache.put(cache_key, vector)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._counter["embeddings"] += 1
        self._counter["total_chars"] += len(payload)
        self._counter["total_ms"] += elapsed_ms
        return vector

    def _embed_learned(self, text: str) -> Optional[List[float]]:
        """Omurga modelinin gizli durumunu havuzlar."""
        if self._model is None:
            return None
        try:
            if self._tokenizer is not None:
                ids = self._tokenizer.encode(text, max_length=512)
            else:
                ids = [ord(char) % 32000 for char in text[:512]]
            if not ids:
                return None
            vector = self._model.embed_tokens(ids)
            if vector is None:
                return None
            return l2_normalize([float(value) for value in vector])
        except Exception as exc:
            self._counter["errors"] += 1
            logger.debug(f"Öğrenilmiş gömme başarısız, hash'e düşülüyor: {exc}")
            return None

    def _embed_local(self, text: str) -> Optional[List[float]]:
        """Sentence-Transformers modeliyle gömer."""
        if self._local_model is None:
            return None
        try:
            vector = self._local_model.encode(text, normalize_embeddings=True)
            return [float(value) for value in vector]
        except Exception as exc:
            self._counter["errors"] += 1
            logger.debug(f"Yerel model gömme hatası: {exc}")
            return None

    def _embed_hash(self, text: str) -> List[float]:
        """Hash tabanlı gömme (her zaman başarılı)."""
        vector = hash_embed(
            text,
            dim=self._hash_dim,
            salt="neurocore-embed",
            idf=self._idf if self._idf_ready else None,
        )
        if self._projection is not None:
            return self._projection.project(vector)
        return vector

    def _fit_dim(self, vector: Sequence[float]) -> List[float]:
        """Vektörü hedef boyuta sığdırır (kırp veya projeksiyonla genişlet)."""
        if len(vector) == self._dim:
            return [float(value) for value in vector]
        if len(vector) > self._dim:
            return l2_normalize(list(vector)[: self._dim])
        if self._projection is None or self._projection.in_dim != len(vector):
            self._projection = ProjectionMatrix(in_dim=len(vector), out_dim=self._dim)
            self._projection.ensure()
        return self._projection.project(vector)

    def embed_batch(self, texts: Sequence[str], *, use_cache: bool = True,
                    normalize: bool = True) -> List[List[float]]:
        """
        Toplu gömme.

        Args:
            texts: Metin listesi
            use_cache: Önbelleği kullan
            normalize: Vektörleri L2-normalize et (zaten normalize edilir)

        Returns:
            Vektör listesi (girdi sırası korunur)
        """
        return [self.embed(text, use_cache=use_cache) for text in texts]

    def embed_tokens(self, ids: Sequence[int]) -> List[float]:
        """Token id dizisini gömer (tokenizer'sız doğrudan kullanım)."""
        if self._tokenizer is not None:
            text = self._tokenizer.decode(ids)
            return self.embed(text)
        return self.embed(" ".join(str(i) for i in ids))

    # ─── Benzerlik ──────────────────────────────────────────────────────

    def similarity(self, text_a: str, text_b: str) -> float:
        """İki metin arasındaki kosinüs benzerliği (0..1'e kırpılır)."""
        return max(0.0, cosine_similarity(self.embed(text_a), self.embed(text_b)))

    def similarity_matrix(self, texts: Sequence[str]) -> List[List[float]]:
        """Metinler arası benzerlik matrisi (küçük kümeler için)."""
        vectors = self.embed_batch(texts)
        size = len(vectors)
        matrix = [[0.0] * size for _ in range(size)]
        for i in range(size):
            matrix[i][i] = 1.0
            for j in range(i + 1, size):
                score = max(0.0, cosine_similarity(vectors[i], vectors[j]))
                matrix[i][j] = score
                matrix[j][i] = score
        return matrix

    def most_similar(self, query: str, candidates: Sequence[str],
                     top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Aday metinler arasından en benzerlerini bulur.

        Returns:
            [(aday_indeksi, benzerlik), ...] — azalan sırada
        """
        query_vector = self.embed(query)
        scored = [
            (index, max(0.0, cosine_similarity(query_vector, self.embed(candidate))))
            for index, candidate in enumerate(candidates)
        ]
        scored.sort(key=lambda item: -item[1])
        return scored[:top_k]

    def centroid(self, texts: Sequence[str]) -> List[float]:
        """Metin kümesinin ağırlık merkezi vektörü."""
        if not texts:
            return [0.0] * self._dim
        accumulator = [0.0] * self._dim
        for text in texts:
            vector = self.embed(text)
            for index in range(min(len(vector), self._dim)):
                accumulator[index] += vector[index]
        return l2_normalize(accumulator)

    def deduplicate(self, texts: Sequence[str], threshold: float = 0.92
                    ) -> List[int]:
        """
        Neredeyse aynı metinlerin indekslerini döndürür (elenecekler).

        Args:
            texts: Metin listesi
            threshold: Benzerlik eşiği

        Returns:
            Elenmesi gereken indeksler (ilk görülen korunur)
        """
        vectors = self.embed_batch(texts)
        keep: List[int] = []
        drop: List[int] = []
        for index, vector in enumerate(vectors):
            duplicate = False
            for kept_index in keep:
                if cosine_similarity(vector, vectors[kept_index]) >= threshold:
                    duplicate = True
                    break
            if duplicate:
                drop.append(index)
            else:
                keep.append(index)
        return drop

    # ─── Yardımcılar ────────────────────────────────────────────────────

    def keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Metinden anahtar kelimeler çıkarır (TF × IDF ağırlıklı).

        IDF öğrenilmemişse yalnızca TF kullanılır ve uzunluk cezası uygulanır.
        """
        tokens = tokenize(text)
        if not tokens:
            return []
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        scored: List[Tuple[float, str]] = []
        for token, count in counts.items():
            weight = count * (1.0 + math.log(count))
            if self._idf:
                weight *= self._idf.get(token, 1.0)
            # Çok kısa token'lar gürültülüdür
            if len(token) < 3:
                weight *= 0.4
            scored.append((weight, token))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [token for _, token in scored[:top_k]]

    def clear_cache(self) -> int:
        """Gömme önbelleğini temizler."""
        return self._cache.clear()

    def warmup(self, texts: Optional[Sequence[str]] = None) -> int:
        """
        Önbelleği önceden ısıtır (ilk sorgu gecikmesini düşürür).

        Args:
            texts: Isıtılacak metinler (None ise kısa bir varsayılan küme)

        Returns:
            Üretilen gömme sayısı
        """
        samples = list(texts or [
            "merhaba", "sistem durumu", "gpu sıcaklığı", "bilgi kasası",
            "yapay sinir ağı", "cpu kullanımı", "ram boş alan",
        ])
        for sample in samples:
            self.embed(sample, use_cache=True)
        return len(samples)

    def health(self) -> Dict[str, Any]:
        """Gömme motoru sağlık raporu."""
        issues: List[str] = []
        if self._model is None and self._local_model is None:
            issues.append("öğrenilmiş model yok — hash gömme kullanılıyor")
        if not self._idf_ready:
            issues.append("IDF öğrenilmedi — arama kalitesi düşük olabilir")
        if not self._initialized:
            issues.append("motor başlatılmadı")
        return {
            "ok": self._initialized,
            "strategy": self._strategy,
            "dim": self._dim,
            "issues": issues,
            "stats": self.get_stats(),
        }

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
            f"<EmbeddingEngine dim={self._dim} strategy={self._strategy} "
            f"cache={self._cache.stats()['size']}>"
        )


__all__ = [
    "EmbeddingEngine",
    "EmbeddingCache",
    "EmbeddingError",
    "ProjectionMatrix",
    "hash_embed",
    "l2_normalize",
    "cosine_similarity",
    "tokenize",
    "stem_turkish",
    "STRATEGY_HASH",
    "STRATEGY_LEARNED",
    "STRATEGY_LOCAL",
]
