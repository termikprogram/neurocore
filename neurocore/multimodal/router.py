"""
NeuroCore Modalite Yönlendirici (L4)
=====================================
Gelen ham yükün (payload) hangi modaliteye ait olduğunu sezer ve onu
`MultiModalInput` yapısına ayrıştırır.

Sezme stratejisi (ucuzdan pahalıya):
    1. Tip kontrolü (numpy, PIL, pandas-benzeri, dict/list/tuple)
    2. Sihirli bayt (magic bytes) koklaması — PNG/JPEG/GIF/BMP/WEBP/WAV/MP3/FLAC/OGG/ZIP/PDF
    3. Dosya uzantısı eşlemesi (`MODALITY_BY_EXTENSION`)
    4. Metin sezgileri (markdown tablo, CSV/TSV, JSON yapıları)

Örnek:
    router = ModalityRouter()
    router.initialize(config, bus)
    payload = router.split(open("veri.csv", "rb").read())
    print(router.describe("a,b\\n1,2"))
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

from neurocore.multimodal.types import (
    MODALITY_BY_EXTENSION,
    EncodedChunk,
    Modality,
    MultiModalInput,
    to_jsonable,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("multimodal.router")


# ─── Sabitler ve Desenler ──────────────────────────────────────────────────

#: `MultiModalInput` alanlarına karşılık gelen yapısal anahtarlar.
STRUCTURED_KEYS: Tuple[str, ...] = ("text", "image", "audio", "table")

#: Markdown tablo ayırıcı satırı (`|---|:--:|`).
_MD_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:\-]*\|[\s:|\-]*$")

#: Sihirli bayt imzaları → (modalite, mime, açıklama).
_MAGIC_SIGNATURES: Tuple[Tuple[bytes, Modality, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", Modality.IMAGE, "image/png", "png"),
    (b"\xff\xd8\xff", Modality.IMAGE, "image/jpeg", "jpeg"),
    (b"GIF8", Modality.IMAGE, "image/gif", "gif"),
    (b"BM", Modality.IMAGE, "image/bmp", "bmp"),
    (b"II*\x00", Modality.IMAGE, "image/tiff", "tiff"),
    (b"MM\x00*", Modality.IMAGE, "image/tiff", "tiff"),
    (b"\x00\x00\x01\x00", Modality.IMAGE, "image/x-icon", "ico"),
    (b"fLaC", Modality.AUDIO, "audio/flac", "flac"),
    (b"OggS", Modality.AUDIO, "audio/ogg", "ogg"),
    (b"ID3", Modality.AUDIO, "audio/mpeg", "mp3"),
    (b"\xff\xfb", Modality.AUDIO, "audio/mpeg", "mp3"),
    (b"\xff\xf3", Modality.AUDIO, "audio/mpeg", "mp3"),
    (b"\xff\xf2", Modality.AUDIO, "audio/mpeg", "mp3"),
    (b"\xff\xfa", Modality.AUDIO, "audio/mpeg", "mp3"),
    (b"PK\x03\x04", Modality.TABLE, "application/zip", "zip"),
    (b"%PDF", Modality.UNKNOWN, "application/pdf", "pdf"),
)

#: MIME öneki → modalite.
_MIME_BY_PREFIX: Tuple[Tuple[str, Modality], ...] = (
    ("image/", Modality.IMAGE),
    ("audio/", Modality.AUDIO),
    ("video/", Modality.UNKNOWN),
    ("text/csv", Modality.TABLE),
    ("text/tab-separated-values", Modality.TABLE),
    ("text/markdown", Modality.TEXT),
    ("text/", Modality.TEXT),
    ("application/json", Modality.TABLE),
    ("application/x-ndjson", Modality.TABLE),
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", Modality.TABLE),
    ("application/vnd.ms-excel", Modality.TABLE),
    ("application/pdf", Modality.UNKNOWN),
)


# ─── Ana Yönlendirici Sınıfı ───────────────────────────────────────────────

class ModalityRouter:
    """
    Ham yükleri modalitelere ayıran yönlendirici.

    Ağır bağımlılık (PIL, numpy, pandas) **kullanmaz**; yalnızca tip/uzantı/bayt
    sezgileriyle çalışır, böylece her ortamda ayağa kalkar.
    """

    name = "multimodal.router"

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — G/Ç yapılmaz."""
        self._config = None
        self._event_bus = None
        self._initialized = False
        # İstatistik sayaçları
        self._stats: Dict[str, int] = {
            "detect_calls": 0,
            "split_calls": 0,
            "describe_calls": 0,
            "detect_errors": 0,
            "file_missing": 0,
        }
        self._modality_counts: Dict[str, int] = {m.value: 0 for m in Modality}

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Yönlendiriciyi bağla (idempotent — ikinci çağrı no-op).

        Args:
            config: `SystemConfig` (None ise varsayılanlar kullanılır)
            event_bus: Olay yolu (opsiyonel)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._initialized = True
        logger.info("Modalite yönlendirici başlatıldı (%d uzantı eşlemesi)",
                    len(MODALITY_BY_EXTENSION))
        if self._event_bus:
            self._event_bus.emit(
                "multimodal.router_ready",
                {"extension_map_size": len(MODALITY_BY_EXTENSION),
                 "magic_signatures": len(_MAGIC_SIGNATURES)},
                source=self.name,
            )

    def shutdown(self) -> None:
        """Kaynakları serbest bırak (idempotent)."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info("Modalite yönlendirici kapatıldı")

    # ─── Modalite Sezme ─────────────────────────────────────────────────

    def detect(self, payload: Any) -> Modality:
        """
        Ham yükün modalitesini sezer.

        Args:
            payload: Metin, bytes, yol, dizi, PIL görüntüsü, tablo benzeri yapı...

        Returns:
            Sezilen `Modality` (anlaşılamazsa `Modality.UNKNOWN`)
        """
        self._stats["detect_calls"] += 1
        try:
            modality = self._detect_inner(payload)
        except Exception as exc:  # pragma: no cover - savunma
            self._stats["detect_errors"] += 1
            logger.warning("Modalite sezme hatası: %s", exc)
            modality = Modality.UNKNOWN
        self._modality_counts[modality.value] = \
            self._modality_counts.get(modality.value, 0) + 1
        # Uyarı: burada `describe()` ÇAĞRILMAZ — o da `detect()` çağırdığı için
        # sonsuz özyineleme yaratır. Yalnızca ucuz bilgi yayınlanır.
        if modality == Modality.UNKNOWN and self._event_bus:
            try:
                size: Any = len(payload) if hasattr(payload, "__len__") else None
            except Exception:  # pragma: no cover - savunma
                size = None
            self._event_bus.emit(
                "multimodal.modality_unknown",
                {"payload_type": type(payload).__name__, "size": size},
                source=self.name,
            )
        return modality

    def _detect_inner(self, payload: Any) -> Modality:
        """`detect()` gövdesi (sayaç/istisna sarmalayıcısı olmadan)."""
        if payload is None:
            return Modality.UNKNOWN
        if isinstance(payload, Modality):
            return payload
        if isinstance(payload, EncodedChunk):
            return payload.modality
        if isinstance(payload, MultiModalInput):
            provided = payload.provided()
            return provided[0] if provided else Modality.UNKNOWN

        # ── Bayt dizileri: sihirli bayt koklaması ───────────────────────
        if isinstance(payload, (bytes, bytearray, memoryview)):
            modality, _mime, _kind = self.sniff_bytes(bytes(payload))
            return modality

        # ── Metin: yol / tablo / düz metin ──────────────────────────────
        if isinstance(payload, str):
            return self._detect_str(payload)

        # ── Sayısal diziler ─────────────────────────────────────────────
        if self._is_numpy_array(payload):
            return self._detect_ndarray(payload)

        # ── Yapısal sözlükler ───────────────────────────────────────────
        if isinstance(payload, dict):
            return self._detect_dict(payload)

        # ── Listeler / demetler ─────────────────────────────────────────
        if isinstance(payload, (list, tuple)):
            return self._detect_sequence(payload)

        # ── Görüntü nesneleri (PIL duck-typing) ─────────────────────────
        if self._is_pil_like(payload):
            return Modality.IMAGE

        # ── Tablo nesneleri (pandas duck-typing) ────────────────────────
        if self._is_dataframe_like(payload):
            return Modality.TABLE

        return Modality.UNKNOWN

    def _detect_str(self, payload: str) -> Modality:
        """Metin yükünü sınıflandırır (dosya yolu → tablo → JSON → CSV → metin)."""
        text = payload.strip()
        if not text:
            return Modality.TEXT

        # Kısa ve tek satırlık bir yol mu?
        if len(text) < 4096 and "\n" not in text and self._looks_like_path(text):
            path_modality = self.detect_file(text)
            if path_modality != Modality.UNKNOWN:
                return path_modality

        if self._looks_like_markdown_table(payload):
            return Modality.TABLE
        if self._looks_like_json_table(text):
            return Modality.TABLE
        if self._looks_like_csv(payload):
            return Modality.TABLE
        return Modality.TEXT

    def _detect_ndarray(self, payload: Any) -> Modality:
        """numpy dizisinin şekline göre modalite sezer."""
        ndim = int(getattr(payload, "ndim", 0))
        shape = tuple(int(s) for s in getattr(payload, "shape", ()))
        dtype = getattr(payload, "dtype", None)
        is_uint8 = str(dtype) == "uint8"
        if ndim >= 3:
            return Modality.IMAGE
        if ndim == 2:
            if len(shape) == 2 and shape[1] in (1, 2, 3, 4) and is_uint8:
                return Modality.IMAGE
            if is_uint8 and min(shape) >= 8:
                return Modality.IMAGE
            return Modality.TABLE
        if ndim == 1:
            return Modality.AUDIO
        return Modality.UNKNOWN

    def _detect_dict(self, payload: dict) -> Modality:
        """Sözlük yükünü sınıflandırır."""
        if not payload:
            return Modality.UNKNOWN
        keys = {str(k).lower() for k in payload.keys()}
        if keys & set(STRUCTURED_KEYS):
            for key in STRUCTURED_KEYS:
                if key in payload and payload.get(key) is not None:
                    return self.detect(payload[key])
            return Modality.UNKNOWN
        # Sütun → değer listesi biçimi (dict-of-lists)
        values = list(payload.values())
        if values and all(isinstance(v, (list, tuple)) for v in values):
            return Modality.TABLE
        # Satır sözlüğü (tek kayıt) veya iç içe sözlük → tablo
        if all(isinstance(v, dict) for v in values):
            return Modality.TABLE
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in values):
            return Modality.TABLE
        return Modality.UNKNOWN

    def _detect_sequence(self, payload: Any) -> Modality:
        """Liste/demet yükünü sınıflandırır."""
        items = list(payload)
        if not items:
            return Modality.UNKNOWN
        # (samples, sample_rate) demeti → ses
        if (isinstance(payload, tuple) and len(payload) == 2
                and self._is_numpy_array(payload[0])
                and isinstance(payload[1], (int, float))):
            return Modality.AUDIO
        if all(isinstance(i, dict) for i in items):
            return Modality.TABLE
        if all(isinstance(i, (list, tuple)) for i in items):
            return Modality.TABLE
        if all(isinstance(i, (str, int, float, bool, type(None))) for i in items) \
                and len(items) > 2:
            return Modality.TABLE
        return Modality.UNKNOWN

    # ─── Dosya ve MIME Sezme ────────────────────────────────────────────

    def detect_file(self, path: str) -> Modality:
        """
        Dosya yolundan modalite sezer: önce uzantı, sonra sihirli baytlar.

        Args:
            path: Dosya yolu

        Returns:
            Sezilen modalite; dosya yoksa/okunamazsa `Modality.UNKNOWN`
        """
        try:
            candidate = Path(str(path))
        except (TypeError, ValueError):
            return Modality.UNKNOWN

        modality = MODALITY_BY_EXTENSION.get(candidate.suffix.lower(), Modality.UNKNOWN)
        if not candidate.exists():
            self._stats["file_missing"] += 1
            return modality
        if modality != Modality.UNKNOWN:
            return modality

        # Uzantı tanınmadı → ilk 64 KB'ı kokla
        try:
            with open(candidate, "rb") as handle:
                head = handle.read(65536)
            sniffed, _mime, _kind = self.sniff_bytes(head)
            return sniffed
        except OSError as exc:
            logger.debug("Dosya koklanamadı (%s): %s", candidate, exc)
            return Modality.UNKNOWN

    def detect_mime(self, mime: str) -> Modality:
        """
        MIME tipinden modalite sezer.

        Args:
            mime: Örn. "image/png", "audio/wav", "text/csv"

        Returns:
            Sezilen modalite
        """
        if not mime:
            return Modality.UNKNOWN
        value = str(mime).split(";")[0].strip().lower()
        for prefix, modality in _MIME_BY_PREFIX:
            if value.startswith(prefix):
                return modality
        return Modality.UNKNOWN

    def sniff_bytes(self, data: bytes) -> Tuple[Modality, str, str]:
        """
        Sihirli baytlardan (modalite, mime, kısa_ad) üçlüsü çıkarır.

        Desteklenen imzalar: PNG, JPEG, GIF, BMP, TIFF, ICO, WEBP, WAV, MP3, FLAC,
        OGG, ZIP (xlsx), PDF.

        Args:
            data: Ham baytlar (en az ilk 16 bayt önerilir)

        Returns:
            (Modality, mime tipi, kısa biçim adı)
        """
        if not data:
            return Modality.UNKNOWN, "", "empty"

        # RIFF tabanlı kaplar: WEBP (görüntü) ve WAVE (ses)
        if len(data) >= 12 and data[:4] == b"RIFF":
            container = data[8:12]
            if container == b"WEBP":
                return Modality.IMAGE, "image/webp", "webp"
            if container == b"WAVE":
                return Modality.AUDIO, "audio/wav", "wav"
            return Modality.UNKNOWN, "application/riff", "riff"

        for signature, modality, mime, kind in _MAGIC_SIGNATURES:
            if data.startswith(signature):
                if kind == "zip":
                    # ZIP kabı: yalnızca xlsx ise tablo sayılır
                    return self._classify_zip(data)
                return modality, mime, kind
        return Modality.UNKNOWN, "", "unknown"

    def _classify_zip(self, data: bytes) -> Tuple[Modality, str, str]:
        """ZIP kabının içine bakıp xlsx mi yoksa başka bir arşiv mi olduğunu sezer."""
        try:
            import io
            import zipfile  # ağır değil; yalnızca koklama anında gerekir

            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = archive.namelist()[:64]
            if any(name.startswith("xl/") for name in names):
                return (Modality.TABLE,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "xlsx")
            return Modality.UNKNOWN, "application/zip", "zip"
        except Exception:
            return Modality.TABLE, "application/zip", "zip?"

    # ─── Ayrıştırma (split) ─────────────────────────────────────────────

    def split(self, payload: Any) -> MultiModalInput:
        """
        Ham yükü yapısal `MultiModalInput`'a ayrıştırır.

        - `dict` → yapısal anahtarlar (text/image/audio/table) + `extra`
        - `str` → metin (veya markdown tablo / CSV görünümlüyse tablo)
        - `bytes` → sihirli baytlarla sezilen modalite
        - diğerleri → tip sezgisiyle ilgili alana yerleştirilir

        Args:
            payload: Ham yük

        Returns:
            `MultiModalInput`
        """
        self._stats["split_calls"] += 1

        if isinstance(payload, MultiModalInput):
            return payload

        # ── Yapısal sözlük ──────────────────────────────────────────────
        if isinstance(payload, dict):
            keys = {str(k).lower() for k in payload.keys()}
            if keys & set(STRUCTURED_KEYS):
                extra = {k: v for k, v in payload.items()
                         if str(k).lower() not in STRUCTURED_KEYS}
                return MultiModalInput(
                    text=payload.get("text"),
                    image=payload.get("image"),
                    audio=payload.get("audio"),
                    table=payload.get("table"),
                    extra=extra,
                )

        modality = self.detect(payload)
        if modality == Modality.TEXT:
            return MultiModalInput(text=payload if isinstance(payload, str)
                                   else str(payload))
        if modality == Modality.IMAGE:
            return MultiModalInput(image=payload)
        if modality == Modality.AUDIO:
            return MultiModalInput(audio=payload)
        if modality == Modality.TABLE:
            return MultiModalInput(table=payload)

        logger.warning("Tanınmayan yük tipi (%s) — 'extra' içine konuldu",
                       type(payload).__name__)
        return MultiModalInput(extra={
            "unrecognized": to_jsonable(str(payload)[:512]),
            "unrecognized_type": type(payload).__name__,
        })

    # ─── Açıklama (describe) ────────────────────────────────────────────

    def describe(self, payload: Any) -> dict:
        """
        Log/teşhis için yükü özetler.

        Args:
            payload: Ham yük

        Returns:
            {"modality": str, "size": int|list|str, "detail": str}
        """
        self._stats["describe_calls"] += 1
        modality = self.detect(payload)
        size: Any = "?"
        detail = ""

        if payload is None:
            size, detail = 0, "boş yük"
        elif isinstance(payload, str):
            size = len(payload)
            if modality == Modality.TABLE:
                detail = "markdown tablo" if self._looks_like_markdown_table(payload) \
                    else ("JSON yapısı" if self._looks_like_json_table(payload.strip())
                          else "CSV/TSV benzeri")
                detail += f", {payload.count(chr(10)) + 1} satır"
            elif self._looks_like_path(payload.strip()) and len(payload) < 4096:
                suffix = Path(payload.strip()).suffix.lower()
                detail = f"dosya yolu ({suffix or 'uzantısız'})"
            else:
                detail = f"{len(payload.split())} kelime"
        elif isinstance(payload, (bytes, bytearray, memoryview)):
            data = bytes(payload)
            size = len(data)
            sniffed, mime, kind = self.sniff_bytes(data)
            detail = f"magic={kind}" + (f", mime={mime}" if mime else "")
            modality = sniffed if sniffed != Modality.UNKNOWN else modality
        elif self._is_numpy_array(payload):
            size = [int(s) for s in getattr(payload, "shape", ())]
            detail = f"numpy dtype={getattr(payload, 'dtype', '?')}"
        elif self._is_pil_like(payload):
            try:
                width, height = payload.size
                size = [int(width), int(height)]
            except Exception:
                size = "?"
            detail = f"PIL mode={getattr(payload, 'mode', '?')}"
        elif isinstance(payload, dict):
            size = len(payload)
            detail = f"{len(payload)} anahtar"
        elif isinstance(payload, (list, tuple)):
            size = len(payload)
            detail = "demet (samples, sr)" if modality == Modality.AUDIO \
                else f"{len(payload)} öğe"
        elif self._is_dataframe_like(payload):
            try:
                rows, cols = payload.shape
                size = [int(rows), int(cols)]
            except Exception:
                size = "?"
            detail = "dataframe benzeri"
        else:
            detail = type(payload).__name__

        return {
            "modality": modality.value,
            "size": to_jsonable(size),
            "detail": detail,
        }

    # ─── Sezgi Yardımcıları ─────────────────────────────────────────────

    @staticmethod
    def _looks_like_path(text: str) -> bool:
        """Metnin var olan bir dosya yoluna benzediğini kontrol eder."""
        if not text or len(text) > 4096 or "\n" in text:
            return False
        if any(ch in text for ch in '<>|*?"'):
            # Windows'ta geçersiz karakterler; sürücü harfi ayırıcısı hariç
            if not re.match(r"^[A-Za-z]:[\\/]", text):
                return False
        try:
            return Path(text).exists()
        except (OSError, ValueError):
            return False

    @staticmethod
    def _looks_like_markdown_table(text: str) -> bool:
        """
        Markdown tablo sezgisi: `|` içeren bir satırı, ayırıcı satır takip ediyorsa.

        Ayırıcı satır `^\\s*\\|?[\\s:-]*\\|[\\s:|-]*$` desenine uymalı ve en az bir
        `-` içermelidir.
        """
        if not text or "|" not in text:
            return False
        lines = text.splitlines()
        for index in range(len(lines) - 1):
            current, following = lines[index], lines[index + 1]
            if "|" not in current or "-" not in following:
                continue
            if _MD_SEPARATOR_RE.match(following) and "|" in following:
                return True
        return False

    @staticmethod
    def _looks_like_json_table(text: str) -> bool:
        """Metnin tablo üretebilen bir JSON (liste/sözlük) olup olmadığını sezer."""
        stripped = text.strip()
        if not stripped or stripped[0] not in "[{":
            return False
        try:
            parsed = json.loads(stripped)
        except (ValueError, TypeError):
            return False
        if isinstance(parsed, list):
            return bool(parsed) and all(isinstance(i, (dict, list)) for i in parsed)
        if isinstance(parsed, dict):
            values = list(parsed.values())
            return bool(values) and all(
                isinstance(v, (list, tuple, dict)) for v in values)
        return False

    @staticmethod
    def _looks_like_csv(text: str, max_lines: int = 5) -> bool:
        """
        CSV/TSV sezgisi.

        Koşullar: en az 2 dolu satır, ilk satırlarda **aynı** ayırıcı sayısı (>=1),
        ilk satırda cümle noktalaması (`.?!`) olmaması, alan sayısı <= 64.
        """
        if not text:
            return False
        lines = [line for line in text.splitlines() if line.strip()][:max_lines]
        if len(lines) < 2:
            return False
        if any(len(line) > 4096 for line in lines):
            return False
        if any(ch in lines[0] for ch in ".?!"):
            return False
        for delimiter in (",", ";", "\t", "|"):
            counts = [line.count(delimiter) for line in lines]
            if counts[0] >= 1 and len(set(counts)) == 1 and counts[0] <= 63:
                return True
        return False

    @staticmethod
    def _is_numpy_array(payload: Any) -> bool:
        """Nesnenin numpy dizisi olup olmadığını (numpy import etmeden) sezer."""
        return (type(payload).__module__.split(".")[0] == "numpy"
                and hasattr(payload, "shape") and hasattr(payload, "dtype"))

    @staticmethod
    def _is_pil_like(payload: Any) -> bool:
        """PIL görüntüsü benzeri nesne sezgisi."""
        return (hasattr(payload, "size") and isinstance(getattr(payload, "size", None), tuple)
                and hasattr(payload, "mode") and hasattr(payload, "tobytes"))

    @staticmethod
    def _is_dataframe_like(payload: Any) -> bool:
        """pandas DataFrame benzeri nesne sezgisi."""
        return (hasattr(payload, "columns") and hasattr(payload, "shape")
                and (hasattr(payload, "to_dict") or hasattr(payload, "values")))

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        JSON-serileştirilebilir istatistikleri döndürür.

        Returns:
            Sayaçlar, modalite dağılımı ve bağlantı durumu
        """
        return {
            "name": self.name,
            "initialized": self._initialized,
            "counters": dict(self._stats),
            "modality_distribution": dict(self._modality_counts),
            "extension_map_size": len(MODALITY_BY_EXTENSION),
            "magic_signatures": len(_MAGIC_SIGNATURES) + 2,
        }


__all__ = ["ModalityRouter", "STRUCTURED_KEYS"]
