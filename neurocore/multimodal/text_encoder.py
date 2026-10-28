"""
NeuroCore Metin Kodlayıcı (L4)
===============================
Deterministik, eğitim gerektirmeyen metin gömlemcisi.

Gömleme hattı:
    normalize → unigram+bigram hash (işaretli blake2b) → z-score
    → 3 konu özelliği (uzunluk kovası, rakam oranı, büyük harf oranı)
    → L2 normalize

Ağır model (transformers vb.) **gerekmez**; `tiktoken` varsa yalnızca token
sayımı/parçalama için kullanılır, yoksa kelime+noktalama regex'i devreye girer.

Örnek:
    enc = TextEncoder()
    enc.initialize(config, bus)
    chunk = enc.encode("GPU sıcaklığı neden yükselir?")
    print(chunk.meta["language_guess"], len(chunk.vector))
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from neurocore.multimodal.types import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_TEXT_LENGTH,
    EncodedChunk,
    Modality,
    attach_native,
    cosine_similarity,
    l2_normalize,
    signed_hash_accumulate,
    zscore_l2_normalize,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("multimodal.text")


# ─── Sabitler ve Desenler ──────────────────────────────────────────────────

#: Kelime + noktalama tokenizasyonu (tiktoken yoksa kullanılır).
_WORD_RE = re.compile(r"\S+\s*", re.UNICODE)

#: Gömleme için normalizasyon: harf/rakam kalır, aksanlar korunur.
_NORM_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Cümle sonu deseni.
_SENTENCE_RE = re.compile(r"[.!?…]+(?:\s|$)")

#: Kod izleri.
_CODE_RE = re.compile(
    r"(```|`[^`\n]+`|\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[\(:]|\bimport\s+\w+"
    r"|\bfunction\s+\w+\s*\(|\breturn\s+|\b#include\b|\bconsole\.log\b"
    r"|\bSELECT\b[\s\S]{0,80}\bFROM\b|=>|\bpublic\s+static\b|\bprint\()",
    re.IGNORECASE,
)

#: Matematik izleri.
_MATH_RE = re.compile(
    r"(\d+\s*[+\-*/^]\s*\d+|[\d)]\s*[=<>]\s*[\d(]|\\frac|\\sum|\\int|∫|∑|√|π|≈|≠|≤|≥)"
)

#: URL izleri.
_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\bfile://\S+)", re.IGNORECASE)

#: Soru kelimeleri (tr + en).
_QUESTION_WORDS = {
    "ne", "neden", "niçin", "nasıl", "niye", "mi", "mı", "mu", "mü", "kim",
    "nerede", "kaç", "hangi", "what", "why", "how", "who", "when", "where",
    "which", "is", "are", "do", "does", "can", "could", "should",
}

#: Durak kelime profilleri (dil tahmini — her dil için < 25 kelime).
_STOPWORDS: Dict[str, Tuple[str, ...]] = {
    "tr": ("bir", "bu", "ve", "için", "ile", "de", "da", "ne", "çok", "daha",
           "ama", "gibi", "olarak", "en", "mi", "mı", "mu", "mü", "her", "var",
           "yok", "ki", "ise", "nasıl", "neden"),
    "en": ("the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was",
           "were", "be", "for", "on", "with", "that", "this", "it", "as", "by",
           "at", "from", "not", "but"),
    "de": ("der", "die", "das", "und", "oder", "ist", "sind", "war", "ein",
           "eine", "einen", "mit", "für", "auf", "von", "zu", "im", "in", "den",
           "dem", "des", "nicht", "auch", "als", "wie"),
    "fr": ("le", "la", "les", "un", "une", "des", "et", "ou", "est", "sont",
           "était", "pour", "dans", "sur", "avec", "que", "qui", "ne", "pas",
           "plus", "au", "aux", "en", "du", "ce"),
    "es": ("el", "la", "los", "las", "un", "una", "unos", "y", "o", "es", "son",
           "era", "para", "en", "con", "que", "por", "no", "más", "como", "al",
           "del", "se", "este", "esta"),
}

#: Türkçe'ye özgü harfler (dil tahmininde ek puan).
_TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")


# ─── Ana Kodlayıcı Sınıfı ──────────────────────────────────────────────────

class TextEncoder:
    """
    Determinist hash tabanlı metin kodlayıcı.

    `embedding_dim` uzunluğunda L2-normalize vektör üretir; vektörün yönü
    metnin kelime dağılımını (unigram + bigram) ve kaba konu sinyallerini taşır.
    """

    name = "multimodal.text"

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — G/Ç veya ağır import yok."""
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._embedding_dim = DEFAULT_EMBEDDING_DIM
        self._max_text_length = DEFAULT_MAX_TEXT_LENGTH
        self._vocab_size = 32000
        self._tokenizer_name = "regex"
        self._tiktoken_available = False
        self._encoding = None          # tiktoken Encoding (tembel yüklenir)
        self._tiktoken_failed = False
        self._stats: Dict[str, Any] = {
            "encoded": 0,
            "batches": 0,
            "encoded_total": 0,
            "errors": 0,
            "truncated": 0,
            "total_tokens": 0,
        }
        self._language_counts: Dict[str, int] = {}

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Kodlayıcıyı yapılandırır (idempotent).

        Args:
            config: `SystemConfig` (None ise varsayılanlar: dim=768, max_len=2048)
            event_bus: Olay yolu (opsiyonel)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._embedding_dim = self._read_embedding_dim(config)
        if config is not None:
            self._max_text_length = int(getattr(config.multimodal, "max_text_length",
                                                 DEFAULT_MAX_TEXT_LENGTH))
            self._vocab_size = int(getattr(config.multimodal, "vocab_size", 32000))
        self._tiktoken_available = self._probe_tiktoken()
        self._tokenizer_name = "tiktoken:cl100k_base" if self._tiktoken_available else "regex"
        self._initialized = True
        logger.info("Metin kodlayıcı hazır (dim=%d, tokenizer=%s, max_len=%d)",
                    self._embedding_dim, self._tokenizer_name, self._max_text_length)
        if self._event_bus:
            self._event_bus.emit(
                "multimodal.text_encoder_ready",
                {"dim": self._embedding_dim, "tokenizer": self._tokenizer_name},
                source=self.name,
            )

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır (idempotent)."""
        if not self._initialized:
            return
        self._encoding = None
        self._initialized = False
        logger.info("Metin kodlayıcı kapatıldı")

    # ─── Tokenizasyon ───────────────────────────────────────────────────

    def tokenize(self, text: str) -> List[str]:
        """
        Metni token listesine ayırır.

        `tiktoken` (cl100k_base) varsa gerçek alt-kelime tokenları, yoksa
        kelime + noktalama regex tokenları döner.

        Args:
            text: Ham metin

        Returns:
            Token listesi
        """
        if text is None:
            return []
        raw = str(text)
        if not raw:
            return []

        encoding = self._get_encoding()
        if encoding is not None:
            try:
                ids = encoding.encode(raw)
                return [self._decode_token(encoding, token_id) for token_id in ids]
            except Exception as exc:  # pragma: no cover - tiktoken sürüm farkı
                logger.debug("tiktoken tokenizasyonu başarısız: %s", exc)
        return _WORD_RE.findall(raw)

    def _decode_token(self, encoding: Any, token_id: int) -> str:
        """Tek bir token kimliğini string'e çevirir (başarısızlıkta '<id>' döner)."""
        try:
            return encoding.decode([token_id])
        except Exception:  # pragma: no cover - savunma
            return f"<{token_id}>"

    def _get_encoding(self) -> Any:
        """tiktoken kodlayıcısını tembel yükler (yoksa None)."""
        if self._encoding is not None or self._tiktoken_failed:
            return self._encoding
        if not self._tiktoken_available:
            return None
        try:
            import tiktoken  # ağır/opsiyonel import — fonksiyon içinde

            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            logger.warning("tiktoken yüklenemedi, regex tokenizer kullanılacak: %s", exc)
            self._tiktoken_failed = True
            self._encoding = None
        return self._encoding

    def _count_tokens(self, text: str) -> int:
        """Metnin token sayısını (yaklaşık veya kesin) döndürür."""
        if not text:
            return 0
        encoding = self._get_encoding()
        if encoding is not None:
            try:
                return len(encoding.encode(text))
            except Exception:  # pragma: no cover - savunma
                pass
        # Yaklaşık: ~4 karakter/token
        return max(1, math.ceil(len(text) / 4))

    def ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Metnin kelime n-gram'larını (normalize edilmiş) döndürür.

        Args:
            text: Ham metin
            n: Gram uzunluğu (1 = unigram)

        Returns:
            Normalize kelime n-gram listesi
        """
        tokens = self._normalized_tokens(text)
        if n <= 1:
            return list(tokens)
        if len(tokens) < n:
            return []
        return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    # ─── Parçalama ──────────────────────────────────────────────────────

    def chunk(self, text: str, chunk_size: int = 512,
              overlap: int = 64) -> List[str]:
        """
        Uzun metni örtüşmeli parçalara böler (kelime ortasından **bölmez**).

        Ölçü birimi "token-ish": tiktoken varsa gerçek token, yoksa kelime
        başına ~1 token + karakter/4 yaklaşımı. Çok uzun tek bir kelime
        (chunk_size'ı aşan) zorunlu olarak karakter bazında bölünür.

        Args:
            text: Ham metin
            chunk_size: Parça başına en büyük token sayısı
            overlap: Ardışık parçalar arasındaki token örtüşmesi

        Returns:
            Parça listesi (metin boşsa boş liste)
        """
        if text is None:
            return []
        raw = str(text)
        if not raw.strip():
            return []
        chunk_size = max(1, int(chunk_size))
        overlap = max(0, min(int(overlap), chunk_size - 1))

        pieces = self._pieces(raw, chunk_size)
        if not pieces:
            return [raw]

        total_tokens = sum(count for _piece, count in pieces)
        if total_tokens <= chunk_size:
            return [raw.strip()]

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0
        for piece, count in pieces:
            if current_tokens + count > chunk_size and current:
                chunks.append("".join(current).strip())
                # Örtüşme için sondan geriye doğru `overlap` token topla
                tail: List[str] = []
                tail_tokens = 0
                for previous in reversed(current):
                    tail.insert(0, previous)
                    tail_tokens += self._piece_tokens(previous)
                    if tail_tokens >= overlap:
                        break
                current = tail if overlap > 0 else []
                current_tokens = sum(self._piece_tokens(p) for p in current)
            current.append(piece)
            current_tokens += count
        if current:
            chunks.append("".join(current).strip())
        return [c for c in chunks if c]

    def _pieces(self, text: str, chunk_size: int) -> List[Tuple[str, int]]:
        """
        Metni (parça, token_sayısı) çiftlerine ayırır; kelime bütünlüğünü korur.
        """
        pieces: List[Tuple[str, int]] = []
        for match in _WORD_RE.finditer(text):
            piece = match.group(0)
            core = piece.strip()
            if not core:
                continue
            tokens = self._piece_tokens(piece)
            if tokens <= chunk_size:
                pieces.append((piece, tokens))
                continue
            # Aşırı uzun tek kelime → karakter bazında böl (zorunlu istisna)
            step = max(1, chunk_size * 4)
            for start in range(0, len(piece), step):
                sub = piece[start:start + step]
                pieces.append((sub, max(1, math.ceil(len(sub) / 4))))
        return pieces

    def _piece_tokens(self, piece: str) -> int:
        """Tek bir parçanın token sayısı (yaklaşık veya kesin)."""
        core = piece.strip()
        if not core:
            return 0
        if self._get_encoding() is not None:
            return max(1, self._count_tokens(piece))
        return max(1, math.ceil(len(core) / 4))

    # ─── Kodlama ────────────────────────────────────────────────────────

    def encode(self, text: str, max_length: Optional[int] = None) -> EncodedChunk:
        """
        Metni `EncodedChunk`'a kodlar.

        Args:
            text: Metin (None/boş → düşük güvenli sıfır vektör)
            max_length: Bu çağrıya özel token sınırı (None → config değeri)

        Returns:
            `Modality.TEXT` tipli `EncodedChunk`
        """
        limit = int(max_length) if max_length else self._max_text_length
        raw = "" if text is None else str(text)
        try:
            truncated_text, n_tokens, was_truncated = self._truncate(raw, limit)
            signals = self._signals(truncated_text)
            vector, native_raw = self._embed(truncated_text)

            confidence = 0.55
            if signals["n_tokens"] >= 3:
                confidence += 0.15
            if signals["language_guess"] != "unknown":
                confidence += 0.10
            if was_truncated:
                confidence -= 0.15
            if signals["n_tokens"] == 0:
                confidence = 0.05
            confidence = max(0.05, min(0.95, confidence))

            meta: Dict[str, Any] = dict(signals)
            meta.update({
                "n_tokens": signals["n_tokens"] if signals["n_tokens"] else n_tokens,
                "max_length": limit,
                "truncated": was_truncated,
                "tokenizer": self._tokenizer_name,
                "vocab_size": self._vocab_size,
                "embedding_dim": self._embedding_dim,
                "hash_salt": "neurocore-text",
                "char_count": len(truncated_text),
            })
            attach_native(meta, native_raw)

            chunk = EncodedChunk(
                modality=Modality.TEXT,
                vector=vector,
                confidence=round(confidence, 4),
                meta=meta,
            )
            self._stats["encoded"] += 1
            self._stats["encoded_total"] += 1
            self._stats["total_tokens"] += meta["n_tokens"]
            if was_truncated:
                self._stats["truncated"] += 1
            language = meta["language_guess"]
            self._language_counts[language] = self._language_counts.get(language, 0) + 1

            if self._event_bus:
                self._event_bus.emit(
                    "multimodal.text_encoded",
                    {"dim": len(vector), "n_tokens": meta["n_tokens"],
                     "language": language, "confidence": chunk.confidence},
                    source=self.name,
                )
            return chunk
        except Exception as exc:
            self._stats["errors"] += 1
            logger.error("Metin kodlama hatası: %s", exc)
            raise

    def encode_batch(self, texts: List[str]) -> List[EncodedChunk]:
        """
        Metin listesini sırayla kodlar.

        Args:
            texts: Metin listesi

        Returns:
            `EncodedChunk` listesi (hatalı öğeler için düşük güvenli boş chunk)
        """
        self._stats["batches"] += 1
        results: List[EncodedChunk] = []
        for item in (texts or []):
            try:
                results.append(self.encode(item))
            except Exception as exc:
                logger.warning("Batch öğesi kodlanamadı: %s", exc)
                results.append(self._empty_chunk())
        return results

    def similarity(self, a: EncodedChunk, b: EncodedChunk) -> float:
        """
        İki metin chunk'ı arasında kosinüs benzerliği hesaplar.

        Args:
            a: Birinci chunk
            b: İkinci chunk

        Returns:
            [-1, 1] arası benzerlik (boyut uyuşmazlığında 0.0)
        """
        if a is None or b is None:
            return 0.0
        if len(a.vector) != len(b.vector):
            logger.warning("Benzerlik için boyutlar farklı: %d vs %d",
                           len(a.vector), len(b.vector))
            return 0.0
        return float(cosine_similarity(a.vector, b.vector))

    # ─── Gömleme İçi Yardımcılar ────────────────────────────────────────

    def _embed(self, text: str) -> Tuple[List[float], List[float]]:
        """
        Metni gömleme vektörüne çevirir.

        Args:
            text: (kısaltılmış) metin

        Returns:
            (L2-normalize gömme, ham z-score vektörü)
        """
        dim = self._embedding_dim
        tokens = self._normalized_tokens(text)

        counts: Dict[str, float] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + 1.0
        bigrams: Dict[str, float] = {}
        for gram in self._normalized_bigrams(tokens):
            bigrams[gram] = bigrams.get(gram, 0.0) + 1.0

        weighted: List[Tuple[str, float]] = []
        for token, count in counts.items():
            weighted.append((token, 1.0 * (1.0 + math.log(count))))
        for gram, count in bigrams.items():
            weighted.append((gram, 0.5 * (1.0 + math.log(count))))

        raw = signed_hash_accumulate(weighted, dim, salt="neurocore-text")
        standardized = zscore_l2_normalize(raw)
        native = list(standardized)

        # 3 konu özelliği → ilk 3 boyut
        if dim >= 3:
            standardized[0] += self._length_bucket(text)
            standardized[1] += self._digit_ratio(text)
            standardized[2] += self._uppercase_ratio(text)
        return l2_normalize(standardized), native

    def _truncate(self, text: str, limit: int) -> Tuple[str, int, bool]:
        """
        Metni token sınırına göre kısaltır.

        Args:
            text: Ham metin
            limit: En büyük token sayısı

        Returns:
            (kısaltılmış metin, token sayısı, kısaltıldı mı)
        """
        if not text:
            return "", 0, False
        if limit <= 0 or len(text) <= limit * 2:
            return text, self._count_tokens(text), False

        encoding = self._get_encoding()
        if encoding is not None:
            ids = encoding.encode(text)
            if len(ids) <= limit:
                return text, len(ids), False
            return encoding.decode(ids[:limit]), limit, True

        pieces = self._pieces(text, limit)
        kept: List[str] = []
        total = 0
        for piece, count in pieces:
            if total + count > limit:
                break
            kept.append(piece)
            total += count
        if total == 0:  # tek parça bile sığmadı
            head = text[:max(1, limit * 4)]
            return head, self._count_tokens(head), True
        return "".join(kept), total, True

    @staticmethod
    def _normalized_tokens(text: str) -> List[str]:
        """Küçük harfe indirger, noktalama atar, aksanları korur."""
        if not text:
            return []
        return _NORM_TOKEN_RE.findall(str(text).lower())

    @staticmethod
    def _normalized_bigrams(tokens: List[str]) -> List[str]:
        """Ardışık token çiftlerinden bigram üretir."""
        return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

    @staticmethod
    def _length_bucket(text: str) -> float:
        """Uzunluk kovası: log ölçekli, 0..1 arası."""
        n_chars = len(text or "")
        return float(min(1.0, math.log1p(n_chars) / math.log1p(DEFAULT_MAX_TEXT_LENGTH * 4)))

    @staticmethod
    def _digit_ratio(text: str) -> float:
        """Rakam karakterlerinin oranı (0..1)."""
        if not text:
            return 0.0
        digits = sum(1 for ch in text if ch.isdigit())
        return float(digits / len(text))

    @staticmethod
    def _uppercase_ratio(text: str) -> float:
        """Büyük harflerin harf sayısına oranı (0..1)."""
        if not text:
            return 0.0
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return 0.0
        upper = sum(1 for ch in letters if ch.isupper())
        return float(upper / len(letters))

    def signals(self, text: str) -> Dict[str, Any]:
        """
        Metnin dil-bağımsız sinyallerini döndürür (genel API).

        Args:
            text: Ham metin

        Returns:
            n_tokens, n_words, n_sentences, has_code, has_math, has_url,
            is_question, language_guess
        """
        return self._signals(text)

    def _signals(self, text: str) -> Dict[str, Any]:
        """
        Dile bağımsız sinyalleri çıkarır.

        Returns:
            n_tokens, n_words, n_sentences, has_code, has_math, has_url,
            is_question, language_guess
        """
        raw = text or ""
        words = self._normalized_tokens(raw)
        sentences = [s for s in _SENTENCE_RE.split(raw) if s.strip()]
        lowered = raw.lower()
        is_question = ("?" in raw) or any(
            w in _QUESTION_WORDS for w in words[:6])
        return {
            "n_tokens": self._count_tokens(raw) if raw else 0,
            "n_words": len(words),
            "n_sentences": max(1, len(sentences)) if raw.strip() else 0,
            "has_code": bool(_CODE_RE.search(raw)),
            "has_math": bool(_MATH_RE.search(raw)),
            "has_url": bool(_URL_RE.search(raw)),
            "is_question": bool(is_question),
            "language_guess": self._guess_language(lowered, words, raw),
        }

    @staticmethod
    def _guess_language(lowered: str, words: List[str], raw: str) -> str:
        """
        Durak kelime profiliyle dil tahmini (tr/en/de/fr/es).

        Args:
            lowered: Küçük harfli metin
            words: Normalize token listesi
            raw: Ham metin (Türkçe karakter bonusu için)

        Returns:
            Dil kodu veya "unknown"
        """
        if not words:
            return "unknown"
        word_set = set(words)
        best_language = "unknown"
        best_score = 0.0
        for language, stopwords in _STOPWORDS.items():
            hits = sum(1 for word in stopwords if word in word_set)
            score = hits / len(stopwords)
            if language == "tr" and any(ch in _TURKISH_CHARS for ch in raw):
                score += 0.15
            if score > best_score:
                best_score = score
                best_language = language
        # Çok zayıf sinyal → bilinmiyor
        if best_score < 0.04:
            return "unknown"
        return best_language

    def _empty_chunk(self) -> EncodedChunk:
        """Hata durumunda döndürülen düşük güvenli boş chunk."""
        return EncodedChunk(
            modality=Modality.TEXT,
            vector=[0.0] * self._embedding_dim,
            confidence=0.0,
            meta={"empty": True, "embedding_dim": self._embedding_dim},
        )

    # ─── Yapılandırma Yardımcıları ──────────────────────────────────────

    @staticmethod
    def _read_embedding_dim(config: Any) -> int:
        """`config.neural.embedding_dim` değerini güvenle okur."""
        if config is None:
            return DEFAULT_EMBEDDING_DIM
        try:
            return int(config.neural.embedding_dim)
        except Exception:
            return DEFAULT_EMBEDDING_DIM

    @staticmethod
    def _probe_tiktoken() -> bool:
        """tiktoken kurulu mu (import etmeden, ucuz kontrol)?"""
        try:
            import importlib.util

            return importlib.util.find_spec("tiktoken") is not None
        except Exception:  # pragma: no cover - savunma
            return False

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        JSON-serileştirilebilir istatistikleri döndürür.

        Returns:
            Kodlama sayaçları, tokenizer bilgisi ve dil dağılımı
        """
        encoded = max(1, self._stats["encoded"])
        return {
            "name": self.name,
            "initialized": self._initialized,
            "embedding_dim": self._embedding_dim,
            "max_text_length": self._max_text_length,
            "vocab_size": self._vocab_size,
            "tokenizer": self._tokenizer_name,
            "tiktoken_available": self._tiktoken_available,
            "tiktoken_loaded": self._encoding is not None,
            "counters": dict(self._stats),
            "avg_tokens_per_text": round(self._stats["total_tokens"] / encoded, 2),
            "language_distribution": dict(self._language_counts),
        }


__all__ = ["TextEncoder"]
