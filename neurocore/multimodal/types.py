"""
NeuroCore Çok-Modlu Çekirdek Tipleri (L4)
==========================================
Metin / görüntü / ses / tablo kodlayıcılarının paylaştığı ortak veri yapıları,
hata sınıfları ve deterministik yardımcılar burada tanımlanır.

Tasarım kuralları
-----------------
- **Gömme boyutu politikası**: Her kodlayıcı `embedding_dim` (varsayılan 768)
  uzunluğunda bir `vector` döndürür. Kodlayıcının kendi doğal (native) vektör
  boyutu `meta["native_dim"]` içinde raporlanır.
- Ham vektör yalnızca `native_dim <= 256` ise `meta["native"]` içine float
  listesi olarak konur; aksi halde konmaz ve `meta["native_truncated"] = True`
  işaretlenir (bellek/JSON şişmesini engellemek için).
- `stable_hash_vector` süreçten bağımsız **deterministiktir** (blake2b tabanlı);
  Python'un `hash()` fonksiyonu **kullanılmaz** (PYTHONHASHSEED'e bağlıdır).
- Tüm yardımcılar saf Python + `math` ile yazılmıştır; numpy yalnızca projeksiyon
  matrisi üretiminde ve dizi dönüşümlerinde kullanılır.

Kullanım:
    from neurocore.multimodal.types import Modality, EncodedChunk, stable_hash_vector

    vec = stable_hash_vector("GPU sıcaklığı", dim=768, salt="neurocore")
    chunk = EncodedChunk(modality=Modality.TEXT, vector=vec, confidence=0.9)
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("multimodal.types")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: `meta["native"]` içine ham vektörün yazılabileceği en büyük boyut.
NATIVE_VECTOR_MAX_DIM: int = 256

#: Varsayılan gömme boyutu (config yoksa kullanılır).
DEFAULT_EMBEDDING_DIM: int = 768

#: Varsayılan metin uzunluğu sınırı (token).
DEFAULT_MAX_TEXT_LENGTH: int = 2048

#: Hash tabanlı token ayrıştırma deseni (aksansızlaştırma YAPMAZ, `_` ayırıcıdır).
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


# ─── Hata Sınıfları ────────────────────────────────────────────────────────

class MultimodalError(Exception):
    """Çok-modlu katmanın tüm hatalarının ortak ata sınıfı."""


class ImageDecodeError(MultimodalError):
    """Görüntü çözülemedi (bozuk dosya, desteklenmeyen biçim, eksik bağımlılık)."""


class AudioDecodeError(MultimodalError):
    """Ses çözülemedi (bozuk dosya, desteklenmeyen biçim, eksik bağımlılık)."""


class TableDecodeError(MultimodalError):
    """Tablo çözülemedi (ayrıştırılamayan biçim, eksik bağımlılık)."""


class EncoderUnavailable(MultimodalError):
    """İstenen kodlayıcı/omurga bu makinede kullanılamıyor (paket veya ağırlık yok)."""


# ─── Modalite Numaralandırması ─────────────────────────────────────────────

class Modality(str, Enum):
    """
    Desteklenen veri modaliteleri.

    `str` tabanlı olduğu için JSON'a doğrudan `value` olarak yazılabilir ve
    `Modality.TEXT == "text"` karşılaştırması geçerlidir.
    """
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TABLE = "table"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: Any) -> "Modality":
        """
        Herhangi bir değeri güvenli biçimde `Modality`'ye çevirir.

        Args:
            value: Modality, string veya tanınmayan herhangi bir nesne

        Returns:
            Eşleşen modalite, bulunamazsa `Modality.UNKNOWN`
        """
        if isinstance(value, Modality):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                return cls.UNKNOWN
        return cls.UNKNOWN

    def __str__(self) -> str:  # pragma: no cover - kozmetik
        return self.value


# ─── Uzantı → Modalite Haritası ────────────────────────────────────────────

#: Dosya uzantılarına göre modalite eşlemesi (router ve kodlayıcılar kullanır).
MODALITY_BY_EXTENSION: Dict[str, Modality] = {
    # Görüntü
    ".png": Modality.IMAGE,
    ".jpg": Modality.IMAGE,
    ".jpeg": Modality.IMAGE,
    ".webp": Modality.IMAGE,
    ".gif": Modality.IMAGE,
    ".bmp": Modality.IMAGE,
    ".tiff": Modality.IMAGE,
    ".tif": Modality.IMAGE,
    ".ico": Modality.IMAGE,
    # Ses
    ".wav": Modality.AUDIO,
    ".mp3": Modality.AUDIO,
    ".flac": Modality.AUDIO,
    ".ogg": Modality.AUDIO,
    ".oga": Modality.AUDIO,
    ".m4a": Modality.AUDIO,
    ".aac": Modality.AUDIO,
    ".opus": Modality.AUDIO,
    ".wma": Modality.AUDIO,
    # Tablo / yapılandırılmış veri
    ".csv": Modality.TABLE,
    ".tsv": Modality.TABLE,
    ".xlsx": Modality.TABLE,
    ".xls": Modality.TABLE,
    ".xlsm": Modality.TABLE,
    ".json": Modality.TABLE,
    ".jsonl": Modality.TABLE,
    ".ndjson": Modality.TABLE,
    ".md": Modality.TEXT,      # markdown normalde metindir; tablo içeriği router'da sezilir
    ".markdown": Modality.TEXT,
    # Metin / kod / yapılandırma
    ".txt": Modality.TEXT,
    ".text": Modality.TEXT,
    ".log": Modality.TEXT,
    ".py": Modality.TEXT,
    ".js": Modality.TEXT,
    ".mjs": Modality.TEXT,
    ".cjs": Modality.TEXT,
    ".ts": Modality.TEXT,
    ".tsx": Modality.TEXT,
    ".jsx": Modality.TEXT,
    ".yaml": Modality.TEXT,
    ".yml": Modality.TEXT,
    ".html": Modality.TEXT,
    ".htm": Modality.TEXT,
    ".css": Modality.TEXT,
    ".scss": Modality.TEXT,
    ".sql": Modality.TEXT,
    ".ini": Modality.TEXT,
    ".toml": Modality.TEXT,
    ".cfg": Modality.TEXT,
    ".conf": Modality.TEXT,
    ".ps1": Modality.TEXT,
    ".bat": Modality.TEXT,
    ".cmd": Modality.TEXT,
    ".sh": Modality.TEXT,
    ".c": Modality.TEXT,
    ".h": Modality.TEXT,
    ".cpp": Modality.TEXT,
    ".hpp": Modality.TEXT,
    ".java": Modality.TEXT,
    ".rs": Modality.TEXT,
    ".go": Modality.TEXT,
    ".rb": Modality.TEXT,
    ".php": Modality.TEXT,
    ".xml": Modality.TEXT,
    ".rst": Modality.TEXT,
    ".tex": Modality.TEXT,
}


# ─── Gömme Vektörü Veri Yapısı ─────────────────────────────────────────────

@dataclass
class EncodedChunk:
    """
    Tek bir modalite parçasının kodlanmış hali.

    Attributes:
        modality: Parçanın modalitesi
        vector: `embedding_dim` uzunluğunda L2-normalize gömme vektörü
        confidence: 0..1 arası güven skoru (kodlayıcıya göre değişir)
        meta: Kodlayıcıya özgü ek bilgiler (native_dim, süre, çözünürlük, ...)
    """
    modality: Modality = Modality.UNKNOWN
    vector: List[float] = field(default_factory=list)
    confidence: float = 0.5
    meta: Dict[str, Any] = field(default_factory=dict)

    # ─── Serileştirme ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        JSON-serileştirilebilir sözlüğe çevirir.

        Returns:
            {"modality", "vector", "confidence", "meta", "dim"}
        """
        return {
            "modality": self.modality.value,
            "vector": [float(v) for v in self.vector],
            "confidence": round(float(self.confidence), 6),
            "meta": to_jsonable(self.meta),
            "dim": len(self.vector),
        }

    def to_numpy(self) -> "np.ndarray":
        """
        Vektörü `float32` numpy dizisine çevirir.

        Returns:
            Şekil (embedding_dim,) olan float32 dizi (boşsa sıfır uzunluklu dizi)
        """
        if not self.vector:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(self.vector, dtype=np.float32)

    @classmethod
    def from_dict(cls, d: dict) -> "EncodedChunk":
        """
        `to_dict()` çıktısından (veya uyumlu bir sözlükten) chunk üretir.

        Args:
            d: En az `vector` anahtarı içeren sözlük

        Returns:
            Yeniden kurulmuş `EncodedChunk`
        """
        data = dict(d or {})
        return cls(
            modality=Modality.from_value(data.get("modality")),
            vector=[float(v) for v in (data.get("vector") or [])],
            confidence=float(data.get("confidence", 0.5)),
            meta=dict(data.get("meta") or {}),
        )

    # ─── Yardımcılar ────────────────────────────────────────────────────

    def __len__(self) -> int:  # pragma: no cover - kozmetik
        return len(self.vector)

    def summary(self) -> str:
        """Log için kısa tek satırlık özet."""
        return (f"{self.modality.value} dim={len(self.vector)} "
                f"conf={self.confidence:.2f}")


# ─── Çok-Modlu Girdi ───────────────────────────────────────────────────────

@dataclass
class MultiModalInput:
    """
    Tek bir istekte birlikte gelen çok-modlu ham girdi paketi.

    Attributes:
        text: Metin (veya metne çevrilebilir yapılandırılmış veri)
        image: Yol (str) | bytes | PIL.Image | np.ndarray
        audio: Yol (str) | bytes | np.ndarray | (samples, sample_rate)
        table: csv/tsv/markdown/json metni | yol | list[dict] | pandas-benzeri
        extra: Hat dışı ek alanlar (kaynak, kullanıcı, oturum, ...)
    """
    text: Optional[str] = None
    image: Any = None
    audio: Any = None
    table: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def provided(self) -> List[Modality]:
        """
        Dolu olan modaliteleri sabit sırayla döndürür.

        Returns:
            [TEXT, IMAGE, AUDIO, TABLE] listesinin dolu olanlarının alt kümesi
        """
        found: List[Modality] = []
        if self.text is not None:
            found.append(Modality.TEXT)
        if self.image is not None:
            found.append(Modality.IMAGE)
        if self.audio is not None:
            found.append(Modality.AUDIO)
        if self.table is not None:
            found.append(Modality.TABLE)
        return found

    def is_empty(self) -> bool:
        """Hiçbir modalite sağlanmadıysa True."""
        return not self.provided()

    def to_dict(self, include_payload: bool = False) -> dict:
        """
        Sözlüğe çevirir.

        Args:
            include_payload: True ise ham verilerin tipleri de raporlanır
                (JSON için ham veri yerine tip adı yazılır).

        Returns:
            {"modalities": [...], "extra": {...}} (+ "payload_types")
        """
        out: Dict[str, Any] = {
            "modalities": [m.value for m in self.provided()],
            "extra": to_jsonable(self.extra),
        }
        if include_payload:
            out["payload_types"] = {
                key: type(value).__name__
                for key, value in (("text", self.text), ("image", self.image),
                                   ("audio", self.audio), ("table", self.table))
                if value is not None
            }
        return out


# ─── Füzyon Çıktısı ────────────────────────────────────────────────────────

@dataclass
class FusedRepresentation:
    """
    Füzyon katmanının ürettiği ortak temsil.

    Attributes:
        vector: `embedding_dim` uzunluğunda L2-normalize füzyon vektörü
        modalities: Füzyona katılan modaliteler (sırayla)
        weights: Modalite adı → ağırlık (toplamı 1.0)
        strategy: Kullanılan strateji ("attention" | "early" | "late" | "concat")
        parts: Modalite adı → o modalitenin katkı vektörü
        meta: Ek bilgiler (chunk sayısı, başarısızlar, güvenler, ...)
    """
    vector: List[float] = field(default_factory=list)
    modalities: List[Modality] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    strategy: str = "attention"
    parts: Dict[str, List[float]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        JSON-serileştirilebilir sözlüğe çevirir.

        Returns:
            {"vector", "modalities", "weights", "strategy", "parts", "meta", "dim"}
        """
        return {
            "vector": [float(v) for v in self.vector],
            "modalities": [m.value for m in self.modalities],
            "weights": {k: round(float(v), 6) for k, v in self.weights.items()},
            "strategy": self.strategy,
            "parts": {k: [float(x) for x in v] for k, v in self.parts.items()},
            "meta": to_jsonable(self.meta),
            "dim": len(self.vector),
        }

    def to_numpy(self) -> "np.ndarray":
        """Füzyon vektörünü `float32` numpy dizisine çevirir."""
        if not self.vector:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(self.vector, dtype=np.float32)

    def is_empty(self) -> bool:
        """Füzyona hiç modalite katılmadıysa True."""
        return not self.modalities


# ─── Tablo Veri Yapısı ─────────────────────────────────────────────────────

@dataclass
class TableData:
    """
    Ayrıştırılmış tablo verisi (TableEncoder'ın ortak ara gösterimi).

    Attributes:
        headers: Sütun adları
        rows: Satır listesi (her satır hücre listesi)
        n_rows: Satır sayısı
        n_cols: Sütun sayısı
        dtypes: Sütun adı → çıkarılan tip ("int","float","bool","date","str","category","id")
        source: Verinin kaynağı ("csv", "markdown", "json", "xlsx", ...)
    """
    headers: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    dtypes: Dict[str, str] = field(default_factory=dict)
    source: str = "unknown"

    def __post_init__(self) -> None:
        """Satır/sütun sayılarını ve eksik sütun adlarını tutarlı hale getirir."""
        self.rows = [list(r) for r in (self.rows or [])]
        if not self.n_rows:
            self.n_rows = len(self.rows)
        if self.headers:
            width = len(self.headers)
        else:
            width = max((len(r) for r in self.rows), default=0)
        if not self.n_cols:
            self.n_cols = width
        # Eksik/boş başlıkları deterministik biçimde doldur
        if len(self.headers) < self.n_cols:
            self.headers = list(self.headers) + [
                f"col_{i}" for i in range(len(self.headers), self.n_cols)
            ]
        # Satırları sütun sayısına hizala (kısa satırlar None ile doldurulur)
        for row in self.rows:
            if len(row) < self.n_cols:
                row.extend([None] * (self.n_cols - len(row)))
            elif len(row) > self.n_cols:
                del row[self.n_cols:]
        if not self.dtypes:
            self.dtypes = {h: "str" for h in self.headers}

    def to_dict(self) -> dict:
        """JSON-serileştirilebilir sözlüğe çevirir (ilk 50 satır ile)."""
        return {
            "headers": list(self.headers),
            "rows": to_jsonable(self.rows[:50]),
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "dtypes": dict(self.dtypes),
            "source": self.source,
        }

    @classmethod
    def from_rows(cls, headers: Sequence[Any], rows: Sequence[Sequence[Any]],
                  source: str = "unknown",
                  dtypes: Optional[Dict[str, str]] = None) -> "TableData":
        """
        Başlık + satırlardan `TableData` üretir.

        Args:
            headers: Sütun adları (None/boş olanlar `col_N` olur)
            rows: Satır verisi
            source: Kaynak etiketi
            dtypes: Hazır tip haritası (verilmezse `__post_init__` doldurur)

        Returns:
            Yeni `TableData`
        """
        clean_headers = [
            str(h).strip() if h is not None and str(h).strip() else f"col_{i}"
            for i, h in enumerate(headers)
        ]
        return cls(
            headers=clean_headers,
            rows=[list(r) for r in rows],
            dtypes=dict(dtypes or {}),
            source=source,
        )


# ─── Deterministik Hash Gömmesi ─────────────────────────────────────────────

def _tokenize_for_hash(text: str) -> List[str]:
    """
    Hash gömlesi için metni tokenlara ayırır (küçük harf, noktalama atılır,
    aksanlar korunur).

    Args:
        text: Ham metin

    Returns:
        Token listesi (boş olabilir)
    """
    if not text:
        return []
    return _TOKEN_RE.findall(str(text).lower())


def signed_hash_accumulate(weighted_grams: Sequence[Tuple[str, float]], dim: int,
                           salt: str = "neurocore") -> List[float]:
    """
    İşaretli hash biriktirmesi (hash gömlemesinin ortak düşük seviye ilkel öğesi).

    Her gram için `blake2b(salt|gram)` hesaplanır; ilk 8 bayt boyut içinde
    indekse, 9. baytın en düşük biti ise işarete (±1) dönüştürülür. Bu yöntem
    Python'un `hash()` fonksiyonundan farklı olarak **süreçler arası
    deterministiktir**.

    Args:
        weighted_grams: (gram, ağırlık) çiftleri
        dim: Vektör boyutu
        salt: Tuz (farklı uzaylar için farklı gömleme üretir)

    Returns:
        Ham (normalize edilmemiş) vektör
    """
    if dim <= 0:
        raise ValueError("dim pozitif olmalı")
    vec = [0.0] * dim
    for gram, weight in weighted_grams:
        if not gram:
            continue
        digest = hashlib.blake2b(f"{salt}|{gram}".encode("utf-8"),
                                 digest_size=16).digest()
        idx = int.from_bytes(digest[:8], "big") % dim
        sign = 1.0 if (digest[8] & 1) else -1.0
        vec[idx] += sign * float(weight)
    return vec


def l2_normalize(vector: Sequence[float]) -> List[float]:
    """
    Vektörü L2-normalize eder (saf Python).

    Sıfır vektör verilirse değiştirilmeden döndürülür.

    Args:
        vector: Sayı dizisi

    Returns:
        Birim normlu liste (veya sıfır vektör)
    """
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm <= 1e-12:
        return [float(v) for v in vector]
    return [float(v) / norm for v in vector]


def zscore_l2_normalize(vector: Sequence[float]) -> List[float]:
    """
    Önce sıfır-ortalama/birim-varyans (z-score) standartlaştırır, sonra
    L2-normalize eder.

    Hash gömlemelerinde ortak bileşenin kosinüs benzerliğini domine etmesini
    engeller; dağılım "birim varyans" özelliği kazanır.

    Args:
        vector: Ham sayı dizisi

    Returns:
        Standartlaştırılmış ve birim normlu liste
    """
    values = [float(v) for v in vector]
    if not values:
        return []
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    if std <= 1e-12:
        return l2_normalize(values)
    return l2_normalize([(v - mean) / std for v in values])


def stable_hash_vector(text: str, dim: int, salt: str = "neurocore") -> List[float]:
    """
    Deterministik, işaretli hash tabanlı bag-of-ngrams gömlesi (unigram + bigram).

    Özellikler:
        - Python `hash()` **kullanılmaz** → süreçten/oturumdan bağımsız aynı çıktı.
        - Unigram ağırlığı 1.0, bigram ağırlığı 0.5 (bağlam sinyali).
        - z-score standartlaştırma (birim varyans) + L2 normalizasyon (birim norm).
        - Boş metin için `salt`'a bağlı deterministik birim vektör döner
          (böylece kosinüs benzerliği tanımlı kalır).

    Args:
        text: Gömlemesi üretilecek metin
        dim: Vektör boyutu (gömme boyutu)
        salt: Tuz (varsayılan "neurocore")

    Returns:
        Uzunluğu `dim` olan, L2-normalize edilmiş float listesi
    """
    if dim <= 0:
        raise ValueError("dim pozitif olmalı")
    tokens = _tokenize_for_hash(text)
    if not tokens:
        digest = hashlib.blake2b(f"{salt}|<empty>".encode("utf-8"),
                                 digest_size=8).digest()
        empty_vec = [0.0] * dim
        empty_vec[int.from_bytes(digest, "big") % dim] = 1.0
        return empty_vec

    grams: List[Tuple[str, float]] = [(t, 1.0) for t in tokens]
    grams += [(f"{a}_{b}", 0.5) for a, b in zip(tokens, tokens[1:])]

    return zscore_l2_normalize(signed_hash_accumulate(grams, dim, salt))


# ─── Deterministik Projeksiyon ──────────────────────────────────────────────

def derive_seed(seed: int, salt: str, *parts: Any) -> int:
    """
    `(seed, salt, parts...)` üçlüsünden deterministik bir tohum türetir.

    Aynı girdi her zaman aynı tohumu üretir; böylece projeksiyon matrisleri
    çağrı sırasından bağımsız olarak sabit kalır.

    Args:
        seed: Temel tohum
        salt: Tuz metni
        *parts: Ek ayırt edici bileşenler (boyutlar gibi)

    Returns:
        64-bit pozitif tohum
    """
    payload = "|".join([salt, str(int(seed))] + [str(p) for p in parts])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 63 - 1)


def projection_matrix(native_dim: int, out_dim: int, seed: int,
                      salt: str) -> "np.ndarray":
    """
    Sabit tohumlu rastgele projeksiyon matrisi üretir (Johnson-Lindenstrauss
    tarzı; hem boyut artırma hem azaltma için kullanılır).

    Matris `(native_dim, out_dim)` şeklindedir ve `1/sqrt(out_dim)` ile
    ölçeklenir; böylece çıktının bileşen varyansı ~1 kalır.

    Args:
        native_dim: Giriş vektör boyutu
        out_dim: Çıkış (gömme) boyutu
        seed: Temel tohum
        salt: Tuz (modüle özgü; aynı tohum farklı uzaylar üretir)

    Returns:
        Şekil (native_dim, out_dim) float64 numpy matrisi
    """
    if native_dim <= 0 or out_dim <= 0:
        raise ValueError("native_dim ve out_dim pozitif olmalı")
    rng = np.random.default_rng(derive_seed(seed, salt, native_dim, out_dim))
    matrix = rng.standard_normal((native_dim, out_dim), dtype=np.float64)
    matrix /= math.sqrt(out_dim)
    return matrix


def attach_native(meta: Dict[str, Any], native: Sequence[float]) -> Dict[str, Any]:
    """
    Gömme boyutu politikasını uygular: ham (native) vektörü `meta`'ya işler.

    - `native_dim` her zaman yazılır.
    - `native_dim <= NATIVE_VECTOR_MAX_DIM` ise `meta["native"]` float listesi olarak eklenir.
    - Aksi halde `meta["native_truncated"] = True` işaretlenir ve vektör eklenmez.

    Args:
        meta: Hedef meta sözlüğü (yerinde güncellenir)
        native: Ham vektör (numpy dizi veya liste)

    Returns:
        Güncellenmiş `meta` sözlüğü
    """
    values = [float(v) for v in np.asarray(native, dtype=np.float64).ravel()]
    meta["native_dim"] = len(values)
    if len(values) <= NATIVE_VECTOR_MAX_DIM:
        meta["native"] = values
    else:
        meta["native_truncated"] = True
    return meta


def to_jsonable(obj: Any) -> Any:
    """
    Nesneyi JSON-serileştirilebilir hale getirir (numpy tipleri, Enum, Path...).

    Args:
        obj: Herhangi bir nesne

    Returns:
        JSON'a yazılabilir karşılığı
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return to_jsonable(obj.item())
        except Exception:  # pragma: no cover - savunma
            pass
    return str(obj)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    İki vektör arasında kosinüs benzerliği hesaplar (saf Python, güvenli).

    Args:
        a: Birinci vektör
        b: İkinci vektör

    Returns:
        [-1, 1] arası benzerlik; boyut uyuşmazlığı veya sıfır norm durumunda 0.0
    """
    if a is None or b is None:
        return 0.0
    va = [float(x) for x in a]
    vb = [float(x) for x in b]
    if not va or not vb or len(va) != len(vb):
        return 0.0
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


__all__ = [
    "Modality",
    "EncodedChunk",
    "MultiModalInput",
    "FusedRepresentation",
    "TableData",
    "MultimodalError",
    "ImageDecodeError",
    "AudioDecodeError",
    "TableDecodeError",
    "EncoderUnavailable",
    "MODALITY_BY_EXTENSION",
    "NATIVE_VECTOR_MAX_DIM",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_MAX_TEXT_LENGTH",
    "stable_hash_vector",
    "signed_hash_accumulate",
    "l2_normalize",
    "zscore_l2_normalize",
    "projection_matrix",
    "derive_seed",
    "attach_native",
    "to_jsonable",
    "cosine_similarity",
]
