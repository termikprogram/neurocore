"""
NeuroCore Çok-Modlu Katman Cephesi (L4)
========================================
L4 katmanının motor tarafından sürülen tek giriş noktası. Yönlendirici,
dört kodlayıcı ve füzyon katmanını tek bir tutarlı API arkasında toplar.

Katman sözleşmesi:
    layer = MultimodalLayer()
    layer.initialize(config, event_bus)      # idempotent
    chunk = layer.encode("Merhaba dünya")    # tek en iyi chunk
    fused = layer.encode_input(mmi)          # çok-modlu füzyon
    print(layer.health(), layer.get_stats())
    layer.shutdown()

Yayınlanan olaylar:
    multimodal.layer_ready, multimodal.encoded, multimodal.fused,
    multimodal.encode_failed
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neurocore.multimodal.audio_encoder import AudioEncoder
from neurocore.multimodal.fusion import FusionLayer
from neurocore.multimodal.image_encoder import ImageEncoder
from neurocore.multimodal.router import ModalityRouter
from neurocore.multimodal.table_encoder import TableEncoder
from neurocore.multimodal.text_encoder import TextEncoder
from neurocore.multimodal.types import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_TEXT_LENGTH,
    EncodedChunk,
    FusedRepresentation,
    Modality,
    MultiModalInput,
    MultimodalError,
    cosine_similarity,
    to_jsonable,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("multimodal.layer")


# ─── Ana Katman Sınıfı ─────────────────────────────────────────────────────

class MultimodalLayer:
    """
    Çok-modlu katman cephesi (L4).

    Alt bileşenler:
        router : ham yük → modalite
        text / image / audio / table : kodlayıcılar
        fusion : `FusedRepresentation` üretici

    Tüm alt bileşenler `initialize()` ile bağlanır; `encode()` gibi iş
    metodları katman henüz başlatılmadıysa sessizce varsayılanlarla başlatır.
    """

    name = "multimodal"

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — G/Ç veya ağır import yok."""
        self._config = None
        self._event_bus = None
        self._initialized = False

        # Alt bileşenler (kurucuları hafiftir; bağlama initialize()'da yapılır)
        self._router = ModalityRouter()
        self._text = TextEncoder()
        self._image = ImageEncoder()
        self._audio = AudioEncoder()
        self._table = TableEncoder()
        self._fusion = FusionLayer()

        self._embedding_dim = DEFAULT_EMBEDDING_DIM
        self._max_text_length = DEFAULT_MAX_TEXT_LENGTH
        self._stats: Dict[str, Any] = {
            "encode_calls": 0,
            "encode_all_calls": 0,
            "encode_input_calls": 0,
            "encode_file_calls": 0,
            "encode_paths_calls": 0,
            "similarity_calls": 0,
            "describe_calls": 0,
            "fused": 0,
            "failures": 0,
            "chunks": 0,
            "auto_initialized": 0,
        }
        self._modality_counts: Dict[str, int] = {m.value: 0 for m in Modality}
        self._failure_counts: Dict[str, int] = {}

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Katmanı ve tüm alt bileşenlerini başlatır (idempotent).

        Args:
            config: `SystemConfig` (None → tüm varsayılanlar)
            event_bus: Olay yolu (opsiyonel)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._embedding_dim = self._read_embedding_dim(config)
        if config is not None:
            try:
                self._max_text_length = int(config.multimodal.max_text_length)
            except Exception:
                self._max_text_length = DEFAULT_MAX_TEXT_LENGTH

        self._router.initialize(config, event_bus)
        self._text.initialize(config, event_bus)
        self._image.initialize(config, event_bus)
        self._audio.initialize(config, event_bus)
        self._table.initialize(config, event_bus)
        self._fusion.initialize(config, event_bus)

        self._initialized = True
        logger.info("Çok-modlu katman hazır (dim=%d, füzyon=%s)",
                    self._embedding_dim, self._fusion.get_stats()["default_strategy"])
        if self._event_bus:
            self._event_bus.emit(
                "multimodal.layer_ready",
                {"dim": self._embedding_dim, "encoders":
                    ["text", "image", "audio", "table", "fusion"]},
                source=self.name,
            )

    def shutdown(self) -> None:
        """Tüm alt bileşenleri kapatır (idempotent)."""
        if not self._initialized:
            return
        for component in (self._text, self._image, self._audio, self._table,
                          self._fusion, self._router):
            try:
                component.shutdown()
            except Exception as exc:  # pragma: no cover - savunma
                logger.warning("Alt bileşen kapatılamadı (%s): %s",
                               getattr(component, "name", "?"), exc)
        self._initialized = False
        logger.info("Çok-modlu katman kapatıldı")

    def _ensure_initialized(self) -> None:
        """İş metodu çağrıldığında katman hazır değilse varsayılanlarla başlatır."""
        if not self._initialized:
            self._stats["auto_initialized"] += 1
            logger.debug("Katman başlatılmamıştı — varsayılanlarla başlatılıyor")
            self.initialize(self._config, self._event_bus)

    # ─── Alt Bileşen Erişimi ────────────────────────────────────────────

    @property
    def router(self) -> ModalityRouter:
        """Modalite yönlendirici."""
        return self._router

    @property
    def text(self) -> TextEncoder:
        """Metin kodlayıcı."""
        return self._text

    @property
    def image(self) -> ImageEncoder:
        """Görüntü kodlayıcı."""
        return self._image

    @property
    def audio(self) -> AudioEncoder:
        """Ses kodlayıcı."""
        return self._audio

    @property
    def table(self) -> TableEncoder:
        """Tablo kodlayıcı."""
        return self._table

    @property
    def fusion(self) -> FusionLayer:
        """Füzyon katmanı."""
        return self._fusion

    # ─── Kodlama ────────────────────────────────────────────────────────

    def encode(self, payload: Any, modality: Optional[Modality] = None) -> EncodedChunk:
        """
        Tek bir yükü kodlar ve **en iyi** chunk'ı döndürür.

        `dict`/`MultiModalInput` gibi çok-modlu yüklerde tüm modaliteler kodlanır
        ve güveni en yüksek chunk döndürülür (füzyon için `encode_input` kullanın).

        Args:
            payload: Herhangi bir ham yük
            modality: Zorlanacak modalite (None → otomatik sezim)

        Returns:
            `EncodedChunk`

        Raises:
            MultimodalError: Yük kodlanamazsa
        """
        self._ensure_initialized()
        self._stats["encode_calls"] += 1

        # ── Çok-modlu yapı → parçala, hepsini kodla, en iyisini seç ─────
        if isinstance(payload, MultiModalInput) or self._is_structured_dict(payload):
            chunks = self.encode_all(payload)
            if not chunks:
                self._publish_failure(Modality.UNKNOWN, payload,
                                      "Çok-modlu yükten hiçbir modalite kodlanamadı")
                raise MultimodalError("Çok-modlu yükten hiçbir modalite kodlanamadı")
            best = max(chunks, key=lambda chunk: chunk.confidence)
            self._count_modality(best.modality)
            return best

        target = modality if modality is not None else self._router.detect(payload)
        try:
            chunk = self._dispatch(payload, target)
        except Exception as exc:
            self._stats["failures"] += 1
            self._failure_counts[type(exc).__name__] = \
                self._failure_counts.get(type(exc).__name__, 0) + 1
            self._publish_failure(target, payload, str(exc))
            raise

        self._stats["chunks"] += 1
        self._count_modality(chunk.modality)
        if self._event_bus:
            self._event_bus.emit(
                "multimodal.encoded",
                {"modality": chunk.modality.value, "dim": len(chunk.vector),
                 "confidence": chunk.confidence,
                 "detail": self._router.describe(payload) if isinstance(payload, str)
                 else {"modality": chunk.modality.value}},
                source=self.name,
            )
        return chunk

    def _dispatch(self, payload: Any, modality: Modality) -> EncodedChunk:
        """
        Yükü ilgili kodlayıcıya yönlendirir.

        Args:
            payload: Ham yük
            modality: Hedef modalite

        Returns:
            `EncodedChunk`

        Raises:
            MultimodalError: Modalite belirlenemezse
        """
        if modality == Modality.TEXT:
            return self._text.encode(self._resolve_text_payload(payload))
        if modality == Modality.IMAGE:
            return self._image.encode(payload)
        if modality == Modality.AUDIO:
            return self._audio.encode(payload)
        if modality == Modality.TABLE:
            return self._table.encode(payload)

        # ── Bilinmeyen: baytları metne çevirmeyi dene ───────────────────
        if isinstance(payload, (bytes, bytearray, memoryview)):
            text = self._try_decode_text(bytes(payload))
            if text is not None:
                logger.warning("Tanınmayan bayt yükü metin olarak kodlanıyor "
                               "(%d bayt)", len(payload))
                return self._text.encode(text)
        raise MultimodalError(
            f"Yükün modalitesi belirlenemedi: {type(payload).__name__}")

    def encode_all(self, payload: Any) -> List[EncodedChunk]:
        """
        Çok-modlu yükün sağlanan tüm modalitelerini kodlar.

        Başarısız modaliteler atlanır (uyarı loglanır, hata `encode_input` meta'sına
        yazılır); tek bir modalite hatası diğerlerini engellemez.

        Args:
            payload: Ham yük, `dict` veya `MultiModalInput`

        Returns:
            `EncodedChunk` listesi (hiçbiri kodlanamazsa boş liste)
        """
        self._ensure_initialized()
        self._stats["encode_all_calls"] += 1
        mmi = payload if isinstance(payload, MultiModalInput) \
            else self._router.split(payload)
        chunks, _failures = self._encode_mmi(mmi)
        return chunks

    def encode_input(self, mmi: MultiModalInput) -> FusedRepresentation:
        """
        Çok-modlu girdiyi kodlayıp füzyonlar.

        Args:
            mmi: `MultiModalInput`

        Returns:
            `FusedRepresentation` (meta["failures"] içinde başarısız modaliteler)
        """
        self._ensure_initialized()
        self._stats["encode_input_calls"] += 1
        if not isinstance(mmi, MultiModalInput):
            mmi = self._router.split(mmi)

        chunks, failures = self._encode_mmi(mmi)
        fused = self._fusion.fuse(chunks)
        fused.meta["failures"] = failures
        fused.meta["extra"] = to_jsonable(mmi.extra)
        fused.meta["provided"] = [m.value for m in mmi.provided()]
        self._stats["fused"] += 1
        self._stats["failures"] += len(failures)

        if self._event_bus:
            self._event_bus.emit(
                "multimodal.fused",
                {"strategy": fused.strategy,
                 "modalities": [m.value for m in fused.modalities],
                 "weights": fused.weights, "n_chunks": len(chunks),
                 "failures": len(failures)},
                source=self.name,
            )
        return fused

    def encode_file(self, path: str) -> EncodedChunk:
        """
        Dosyayı uzantısına/sihirli baytlarına göre kodlar.

        Args:
            path: Dosya yolu

        Returns:
            `EncodedChunk`

        Raises:
            MultimodalError: Dosya yoksa veya türü desteklenmiyorsa
        """
        self._ensure_initialized()
        self._stats["encode_file_calls"] += 1
        candidate = Path(str(path))
        if not candidate.exists():
            self._publish_failure(Modality.UNKNOWN, path, "Dosya bulunamadı")
            raise MultimodalError(f"Dosya bulunamadı: {path}")

        modality = self._router.detect_file(str(candidate))
        if modality == Modality.UNKNOWN:
            self._publish_failure(Modality.UNKNOWN, path,
                                  "Desteklenmeyen dosya türü")
            raise MultimodalError(
                f"Desteklenmeyen dosya türü: {candidate.suffix or 'uzantısız'} ({path})")

        try:
            if modality == Modality.TEXT:
                text = self._read_text_file(candidate)
                return self.encode(text, Modality.TEXT)
            return self.encode(str(candidate), modality)
        except Exception as exc:
            self._stats["failures"] += 1
            self._publish_failure(modality, str(candidate), str(exc))
            raise

    def encode_paths(self, paths: List[str]) -> FusedRepresentation:
        """
        Birden çok dosyayı kodlayıp füzyonlar (hatalı dosyalar atlanır).

        Args:
            paths: Dosya yolları

        Returns:
            `FusedRepresentation` (meta["failures"] içinde atlananlar)
        """
        self._ensure_initialized()
        self._stats["encode_paths_calls"] += 1
        chunks: List[EncodedChunk] = []
        failures: List[Dict[str, Any]] = []
        for path in (paths or []):
            try:
                chunks.append(self.encode_file(path))
            except Exception as exc:
                logger.warning("Dosya kodlanamadı (%s): %s", path, exc)
                failures.append({"path": str(path), "error": str(exc)})

        fused = self._fusion.fuse(chunks)
        fused.meta["failures"] = failures
        fused.meta["n_paths"] = len(paths or [])
        self._stats["fused"] += 1
        self._stats["failures"] += len(failures)
        if self._event_bus:
            self._event_bus.emit(
                "multimodal.fused",
                {"strategy": fused.strategy, "n_chunks": len(chunks),
                 "n_paths": len(paths or []), "failures": len(failures)},
                source=self.name,
            )
        return fused

    # ─── Benzerlik ve Betimleme ─────────────────────────────────────────

    def similarity(self, a: Any, b: Any) -> float:
        """
        İki ham yük arasında kosinüs benzerliği hesaplar.

        Yükler önce kodlanır; birden çok modalite içeriyorsa füzyon vektörü
        karşılaştırılır.

        Args:
            a: Birinci yük (chunk, füzyon temsili veya ham yük)
            b: İkinci yük

        Returns:
            [-1, 1] arası benzerlik
        """
        self._ensure_initialized()
        self._stats["similarity_calls"] += 1
        vector_a = self._vector_of(a)
        vector_b = self._vector_of(b)
        if not vector_a or not vector_b:
            logger.warning("Benzerlik için vektör üretilemedi")
            return 0.0
        if len(vector_a) != len(vector_b):
            logger.warning("Benzerlik için boyutlar farklı: %d vs %d",
                           len(vector_a), len(vector_b))
            return 0.0
        return float(cosine_similarity(vector_a, vector_b))

    def _vector_of(self, payload: Any) -> List[float]:
        """Bir yükü temsil eden vektörü (kodlayarak) üretir."""
        if isinstance(payload, EncodedChunk):
            return list(payload.vector)
        if isinstance(payload, FusedRepresentation):
            return list(payload.vector)
        try:
            chunks = self.encode_all(payload)
        except Exception as exc:
            logger.warning("Benzerlik için kodlama başarısız: %s", exc)
            return []
        if not chunks:
            return []
        if len(chunks) == 1:
            return list(chunks[0].vector)
        return list(self._fusion.fuse(chunks).vector)

    def describe(self, payload: Any) -> dict:
        """
        Yükü ayrıntılı biçimde betimler (yönlendirici + kodlayıcı detayları).

        Args:
            payload: Ham yük

        Returns:
            JSON-serileştirilebilir betimleme sözlüğü
        """
        self._ensure_initialized()
        self._stats["describe_calls"] += 1
        base = self._router.describe(payload)
        result: Dict[str, Any] = dict(base)
        result["embedding_dim"] = self._embedding_dim
        modality = Modality.from_value(base.get("modality"))

        try:
            if modality == Modality.IMAGE:
                result.update(self._image.describe(payload))
            elif modality == Modality.AUDIO:
                result.update(self._audio.describe(payload))
            elif modality == Modality.TABLE:
                profile = self._table.profile(payload)
                result.update({
                    "n_rows": profile["n_rows"],
                    "n_cols": profile["n_cols"],
                    "headers": profile["headers"][:16],
                    "column_types": {c["name"]: c["inferred_type"]
                                     for c in profile["columns"][:16]},
                    "numeric_columns": profile["numeric_columns"],
                    "source": profile["source"],
                })
            elif modality == Modality.TEXT:
                text = self._resolve_text_payload(payload)
                signals = self._text.signals(text)
                result.update(signals)
                result["char_count"] = len(text or "")
        except Exception as exc:
            logger.warning("Betimleme sırasında hata (%s): %s", modality.value, exc)
            result["describe_error"] = str(exc)

        return to_jsonable(result)

    # ─── Sağlık ─────────────────────────────────────────────────────────

    def health(self) -> dict:
        """
        Katmanın sağlık durumunu ve yeteneklerini raporlar.

        Returns:
            {"ok": bool, "capabilities": {...}, "issues": [...]}
        """
        image_stats = self._image.get_stats()
        audio_stats = self._audio.get_stats()
        table_stats = self._table.get_stats()
        text_stats = self._text.get_stats()

        capabilities = {
            "initialized": self._initialized,
            "text": self._text is not None,
            "image": self._image is not None,
            "audio": self._audio is not None,
            "table": self._table is not None,
            "fusion": self._fusion is not None,
            "router": self._router is not None,
            "embedding_dim": self._embedding_dim,
            "fusion_strategies": ["attention", "early", "late", "concat"],
            "default_fusion_strategy": self._fusion.get_stats()["default_strategy"],
            "pillow": bool(image_stats.get("pillow_available")),
            "torch": bool(image_stats.get("torch_available")),
            "torchvision": bool(image_stats.get("torch_available")),
            "soundfile": bool(audio_stats.get("soundfile_available")),
            "scipy": bool(audio_stats.get("scipy_available")),
            "tiktoken": bool(text_stats.get("tiktoken_available")),
            "openpyxl": bool(table_stats.get("openpyxl_available")),
            "pandas": bool(table_stats.get("pandas_available")),
            "deep_image": bool(image_stats.get("backbone_loaded")),
            "deep_audio": bool(audio_stats.get("backbone_loaded")),
            "deep_audio_runtime": bool(audio_stats.get("torch_available")),
            "native_feature_dims": {
                "image": image_stats.get("native_feature_dim"),
                "audio": audio_stats.get("native_feature_dim"),
                "table": table_stats.get("native_feature_dim"),
            },
        }

        issues: List[str] = []
        if not self._initialized:
            issues.append("Katman başlatılmadı — initialize() çağrılmadı")
        if not capabilities["pillow"]:
            issues.append("Pillow yok: görüntü kodlama devre dışı")
        if not capabilities["soundfile"]:
            issues.append("soundfile yok: wav dışı ses biçimleri (mp3/flac/ogg) çözülemez")
        if not capabilities["scipy"]:
            issues.append("scipy yok: ses yeniden örnekleme doğrusal interpolasyona düşer")
        if not capabilities["tiktoken"]:
            issues.append("tiktoken yok: token sayımı yaklaşık (~4 karakter/token) yapılır")
        if not capabilities["torch"]:
            issues.append("torch/torchvision yok: derin görüntü kodlama (encode_deep) kapalı")
        if not capabilities["deep_audio_runtime"]:
            issues.append("torchaudio yok: derin ses kodlama (encode_deep) kapalı")
        if not capabilities["openpyxl"] and not capabilities["pandas"]:
            issues.append("openpyxl/pandas yok: xlsx tablolar okunamaz")
        for name, stats in (("text", text_stats), ("image", image_stats),
                            ("audio", audio_stats), ("table", table_stats)):
            if stats.get("counters", {}).get("errors", 0) > 0:
                issues.append(f"{name} kodlayıcıda "
                              f"{stats['counters']['errors']} hata kaydedildi")

        ok = bool(self._initialized)
        return {
            "ok": ok,
            "capabilities": to_jsonable(capabilities),
            "issues": issues,
        }

    # ─── İç Yardımcılar ─────────────────────────────────────────────────

    def _encode_mmi(self, mmi: MultiModalInput
                    ) -> Tuple[List[EncodedChunk], List[Dict[str, Any]]]:
        """
        `MultiModalInput`'un sağlanan modalitelerini kodlar.

        Args:
            mmi: Çok-modlu girdi

        Returns:
            (chunks, failures) — hatalı modaliteler `failures` listesine yazılır
        """
        chunks: List[EncodedChunk] = []
        failures: List[Dict[str, Any]] = []
        mapping = (
            (Modality.TEXT, mmi.text, self._text),
            (Modality.IMAGE, mmi.image, self._image),
            (Modality.AUDIO, mmi.audio, self._audio),
            (Modality.TABLE, mmi.table, self._table),
        )
        for modality, payload, encoder in mapping:
            if payload is None:
                continue
            try:
                if modality == Modality.TEXT:
                    chunk = encoder.encode(self._resolve_text_payload(payload))
                else:
                    chunk = encoder.encode(payload)
                chunks.append(chunk)
                self._stats["chunks"] += 1
                self._count_modality(modality)
            except Exception as exc:
                logger.warning("%s modalitesi kodlanamadı: %s", modality.value, exc)
                failures.append({"modality": modality.value, "error": str(exc),
                                 "error_type": type(exc).__name__})
                self._stats["failures"] += 1
                self._failure_counts[type(exc).__name__] = \
                    self._failure_counts.get(type(exc).__name__, 0) + 1
                self._publish_failure(modality, payload, str(exc))
        return chunks, failures

    def _resolve_text_payload(self, payload: Any) -> str:
        """
        Metin modalitesi için yükü gerçek metne çevirir (dosya yolu okunur).

        Args:
            payload: str | Path | bytes | diğer

        Returns:
            Metin içeriği
        """
        if isinstance(payload, str):
            stripped = payload.strip()
            if stripped and len(stripped) < 4096 and "\n" not in stripped:
                try:
                    candidate = Path(stripped)
                    if candidate.exists() and candidate.is_file():
                        logger.debug("Metin yükü dosya yolu olarak okunuyor: %s", stripped)
                        return self._read_text_file(candidate)
                except (OSError, ValueError):
                    # Geçersiz yol (boş bayt, izin vb.) → düz metin olarak kullan
                    pass
            return payload
        if isinstance(payload, Path):
            return self._read_text_file(payload)
        if isinstance(payload, (bytes, bytearray, memoryview)):
            text = self._try_decode_text(bytes(payload))
            if text is not None:
                return text
        return str(payload)

    @staticmethod
    def _read_text_file(path: Path, max_chars: int = 4_000_000) -> str:
        """
        Metin dosyasını okur (utf-8 → cp1254 → latin-1).

        Args:
            path: Dosya yolu
            max_chars: Okunacak en büyük karakter sayısı (güvenlik sınırı)

        Returns:
            Dosya içeriği (kesilmiş olabilir)

        Raises:
            MultimodalError: Dosya okunamazsa
        """
        for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
            try:
                text = path.read_text(encoding=encoding)
                return text[:max_chars]
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                raise MultimodalError(f"Metin dosyası okunamadı ({path}): {exc}") from exc
        raise MultimodalError(f"Metin dosyası hiçbir kodlamayla okunamadı: {path}")

    @staticmethod
    def _try_decode_text(data: bytes) -> Optional[str]:
        """
        Baytları metne çevirmeyi dener; çoğu yazdırılamayan baytsa None döner.

        Args:
            data: Ham baytlar

        Returns:
            Metin veya None
        """
        if not data:
            return None
        for encoding in ("utf-8", "cp1254", "latin-1"):
            try:
                text = data.decode(encoding)
            except UnicodeDecodeError:
                continue
            printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
            if text and printable / len(text) >= 0.9:
                return text
            return None
        return None

    @staticmethod
    def _is_structured_dict(payload: Any) -> bool:
        """Sözlüğün yapısal çok-modlu anahtarlar içerip içermediğini sezer."""
        if not isinstance(payload, dict):
            return False
        keys = {str(k).lower() for k in payload.keys()}
        return bool(keys & {"text", "image", "audio", "table"})

    def _count_modality(self, modality: Modality) -> None:
        """Modalite sayaçlarını günceller."""
        key = Modality.from_value(modality).value
        self._modality_counts[key] = self._modality_counts.get(key, 0) + 1

    def _publish_failure(self, modality: Modality, payload: Any, error: str) -> None:
        """`multimodal.encode_failed` olayını yayınlar."""
        logger.debug("Kodlama hatası (%s): %s", Modality.from_value(modality).value, error)
        if self._event_bus:
            self._event_bus.emit(
                "multimodal.encode_failed",
                {"modality": Modality.from_value(modality).value,
                 "payload_type": type(payload).__name__, "error": str(error)[:512]},
                source=self.name,
            )

    @staticmethod
    def _read_embedding_dim(config: Any) -> int:
        """`config.neural.embedding_dim` değerini güvenle okur."""
        if config is None:
            return DEFAULT_EMBEDDING_DIM
        try:
            return int(config.neural.embedding_dim)
        except Exception:
            return DEFAULT_EMBEDDING_DIM

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Katmanın ve alt bileşenlerinin JSON-serileştirilebilir istatistikleri.

        Returns:
            Sayaçlar, modalite dağılımı, alt bileşen istatistikleri
        """
        return {
            "name": self.name,
            "initialized": self._initialized,
            "embedding_dim": self._embedding_dim,
            "max_text_length": self._max_text_length,
            "device": str(getattr(self._config, "device", "cpu")) if self._config else "cpu",
            "counters": dict(self._stats),
            "modality_distribution": dict(self._modality_counts),
            "failure_types": dict(self._failure_counts),
            "components": {
                "router": self._router.get_stats(),
                "text": self._text.get_stats(),
                "image": self._image.get_stats(),
                "audio": self._audio.get_stats(),
                "table": self._table.get_stats(),
                "fusion": self._fusion.get_stats(),
            },
        }


__all__ = ["MultimodalLayer"]
