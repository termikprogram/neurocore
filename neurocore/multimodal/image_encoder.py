"""
NeuroCore Görüntü Kodlayıcı (L4)
=================================
Ağır model (CLIP/ResNet) **gerektirmeyen**, tamamen numpy + Pillow ile çalışan
deterministik görüntü kodlayıcı.

Özellik vektörü (ham/native boyut = 338):
    (a) 8×8×3 ortalama-havuzlanmış RGB ızgara      → 192
    (b) 8×8 gri ızgara                             →  64
    (c) kanal başına 16 kutulu histogram           →  48
    (d) 4×4 gradyan büyüklüğü ızgarası             →  16
    (e) 8 kutulu ton (hue) histogramı              →   8
    (f) skaler blok (en-boy, parlaklık, ...)       →  10

Ardından sabit tohumlu rastgele projeksiyon (`seed=1337`,
`salt="neurocore-image-proj"`) ile `embedding_dim` boyutuna taşınır ve
L2-normalize edilir. `torch`+`torchvision` varsa **isteğe bağlı** `encode_deep()`
kullanılabilir; omurga ağırlıkları `load_backbone()` ile yüklenmedikçe
`encode()` derin modeli **kullanmaz**.

Örnek:
    enc = ImageEncoder()
    enc.initialize(config, bus)
    chunk = enc.encode("ekran.png")
    print(enc.describe("ekran.png")["dominant_colors"])
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neurocore.multimodal.types import (
    DEFAULT_EMBEDDING_DIM,
    EncodedChunk,
    EncoderUnavailable,
    ImageDecodeError,
    Modality,
    attach_native,
    l2_normalize,
    projection_matrix,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("multimodal.image")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: Projeksiyon tohumu ve tuzu (deterministik — değiştirilmemeli).
IMAGE_PROJ_SEED = 1337
IMAGE_PROJ_SALT = "neurocore-image-proj"

#: Ham özellik vektörünün boyutu (a+b+c+d+e+f).
NATIVE_FEATURE_DIM = 192 + 64 + 48 + 16 + 8 + 10

#: HSV ton histogramı kutu sayısı.
_HUE_BINS = 8

#: Varsayılan görüntü boyutu.
_DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)

#: ImageNet normalizasyon sabitleri (derin omurga için).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ─── Ana Kodlayıcı Sınıfı ──────────────────────────────────────────────────

class ImageEncoder:
    """
    Hafif, deterministik görüntü kodlayıcı (el yapımı özellikler + projeksiyon).

    Hiçbir eğitilmiş ağırlık gerektirmez; bu yüzden sistem her ortamda çalışır.
    """

    name = "multimodal.image"

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — G/Ç veya ağır import yok."""
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._embedding_dim = DEFAULT_EMBEDDING_DIM
        self._image_size: Tuple[int, int] = _DEFAULT_IMAGE_SIZE
        self._pillow_available = False
        self._torch_available = False
        self._backbone = None
        self._backbone_name: Optional[str] = None
        self._torch_device = "cpu"
        # (native_dim, out_dim) → projeksiyon matrisi önbelleği
        self._proj_cache: Dict[Tuple[int, int], "np.ndarray"] = {}
        self._stats: Dict[str, Any] = {
            "encoded": 0,
            "batches": 0,
            "errors": 0,
            "decode_failures": 0,
            "padded": 0,
            "total_pixels": 0,
            "deep_encoded": 0,
        }
        self._format_counts: Dict[str, int] = {}

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Kodlayıcıyı yapılandırır (idempotent).

        Args:
            config: `SystemConfig`; `config.multimodal.image_size` kullanılır
            event_bus: Olay yolu (opsiyonel)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._embedding_dim = self._read_embedding_dim(config)
        self._image_size = self._read_image_size(config)
        self._torch_device = str(getattr(config, "device", "cpu") or "cpu") \
            if config is not None else "cpu"
        if self._torch_device == "auto":
            self._torch_device = "cuda" if self._probe_module("torch") else "cpu"

        # Ucuz yetenek keşfi (import etmeden)
        self._pillow_available = self._probe_module("PIL")
        self._torch_available = (self._probe_module("torch")
                                 and self._probe_module("torchvision"))
        self._initialized = True
        logger.info("Görüntü kodlayıcı hazır (dim=%d, boyut=%s, torch=%s, PIL=%s)",
                    self._embedding_dim, self._image_size,
                    self._torch_available, self._pillow_available)
        if self._event_bus:
            self._event_bus.emit(
                "multimodal.image_encoder_ready",
                {"dim": self._embedding_dim, "image_size": list(self._image_size),
                 "torch_available": self._torch_available},
                source=self.name,
            )

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır (idempotent)."""
        if not self._initialized:
            return
        self._backbone = None
        self._backbone_name = None
        self._proj_cache.clear()
        self._initialized = False
        logger.info("Görüntü kodlayıcı kapatıldı")

    # ─── Yükleme / Dönüştürme ───────────────────────────────────────────

    def to_pil(self, image: Any) -> Any:
        """
        Girdiyi `PIL.Image` nesnesine çevirir.

        Args:
            image: Yol (str) | bytes | PIL.Image | np.ndarray | iç içe liste

        Returns:
            Yüklenmiş `PIL.Image` (kopyası)

        Raises:
            ImageDecodeError: Girdi çözülemezse (Türkçe mesajla)
        """
        if image is None:
            raise ImageDecodeError("Görüntü verisi boş (None)")

        Image = self._import_pil()

        # ── Zaten PIL görüntüsü ─────────────────────────────────────────
        if self._is_pil(image):
            try:
                image.load()
                return image.copy()
            except Exception as exc:
                raise ImageDecodeError(f"PIL görüntüsü yüklenemedi: {exc}") from exc

        # ── Dosya yolu ──────────────────────────────────────────────────
        if isinstance(image, str):
            try:
                from pathlib import Path

                path = Path(image)
                if not path.exists():
                    raise ImageDecodeError(f"Görüntü dosyası bulunamadı: {image}")
                with Image.open(path) as handle:
                    handle.load()
                    return handle.copy()
            except ImageDecodeError:
                raise
            except Exception as exc:
                raise ImageDecodeError(f"Görüntü dosyası açılamadı ({image}): {exc}") from exc

        # ── Baytlar ─────────────────────────────────────────────────────
        if isinstance(image, (bytes, bytearray, memoryview)):
            data = bytes(image)
            if not data:
                raise ImageDecodeError("Görüntü bayt dizisi boş")
            try:
                with Image.open(io.BytesIO(data)) as handle:
                    handle.load()
                    return handle.copy()
            except Exception as exc:
                raise ImageDecodeError(
                    f"Görüntü baytları çözülemedi ({len(data)} bayt): {exc}") from exc

        # ── numpy dizisi / liste ────────────────────────────────────────
        try:
            array = np.asarray(image)
            return self._array_to_pil(array)
        except ImageDecodeError:
            raise
        except Exception as exc:
            raise ImageDecodeError(
                f"Desteklenmeyen görüntü tipi ({type(image).__name__}): {exc}") from exc

    def to_array(self, image: Any, size: Optional[Tuple[int, int]] = None) -> "np.ndarray":
        """
        Girdiyi `float32`, RGB, [0,1] aralığında numpy dizisine çevirir.

        Args:
            image: Yol | bytes | PIL.Image | np.ndarray
            size: (genişlik, yükseklik) hedef boyut (None → yeniden boyutlandırma yok)

        Returns:
            Şekil (H, W, 3) float32 dizi

        Raises:
            ImageDecodeError: Girdi çözülemezse
        """
        pil_image = self.to_pil(image)
        try:
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            if size:
                target = (int(size[0]), int(size[1]))
                if pil_image.size != target:
                    pil_image = pil_image.resize(target, self._resample_filter())
            array = np.asarray(pil_image, dtype=np.float32)
        except ImageDecodeError:
            raise
        except Exception as exc:
            raise ImageDecodeError(f"Görüntü diziye çevrilemedi: {exc}") from exc
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        return np.clip(array / 255.0, 0.0, 1.0).astype(np.float32)

    def _array_to_pil(self, array: "np.ndarray") -> Any:
        """numpy dizisini güvenli biçimde PIL görüntüsüne çevirir."""
        Image = self._import_pil()
        data = np.asarray(array)

        if data.ndim == 2:
            pass
        elif data.ndim == 3:
            channels = data.shape[2]
            if channels == 1:
                data = data[:, :, 0]
            elif channels == 2:
                data = data[:, :, 0]
            elif channels == 4:
                pass  # RGBA olduğu gibi bırakılır (şeffaflık korunur)
            elif channels != 3:
                raise ImageDecodeError(f"Desteklenmeyen kanal sayısı: {channels}")
        else:
            raise ImageDecodeError(
                f"Görüntü dizisi 2 veya 3 boyutlu olmalı (gelen: {data.ndim})")

        if data.size == 0:
            raise ImageDecodeError("Görüntü dizisi boş")

        if np.issubdtype(data.dtype, np.floating):
            finite = data[np.isfinite(data)]
            peak = float(finite.max()) if finite.size else 0.0
            scaled = data / 255.0 if peak > 1.5 else data
            data = np.clip(np.nan_to_num(scaled, nan=0.0) * 255.0, 0, 255).astype(np.uint8)
        elif data.dtype == np.bool_:
            data = (data.astype(np.uint8) * 255)
        elif data.dtype != np.uint8:
            if np.issubdtype(data.dtype, np.integer):
                info = np.iinfo(data.dtype)
                if info.max > 255:
                    data = (data.astype(np.float64) / max(1, info.max) * 255.0).astype(np.uint8)
                else:
                    data = np.clip(data, 0, 255).astype(np.uint8)
            else:
                data = np.clip(data, 0, 255).astype(np.uint8)

        try:
            return Image.fromarray(np.ascontiguousarray(data))
        except Exception as exc:
            raise ImageDecodeError(f"numpy dizisinden görüntü üretilemedi: {exc}") from exc

    # ─── Kodlama ────────────────────────────────────────────────────────

    def encode(self, image: Any) -> EncodedChunk:
        """
        Görüntüyü `EncodedChunk`'a kodlar.

        Args:
            image: Yol | bytes | PIL.Image | np.ndarray

        Returns:
            `Modality.IMAGE` tipli `EncodedChunk` (vektör uzunluğu `embedding_dim`)

        Raises:
            ImageDecodeError: Görüntü çözülemezse
        """
        try:
            rgb, info = self._load_rgb_array(image)
            features, scalars = self._feature_vector(rgb)
            vector = self._project(features, self._embedding_dim)

            confidence = 0.55
            if info["decoded_natively"]:
                confidence += 0.2
            if info["padded"]:
                confidence -= 0.15
            confidence = max(0.05, min(0.95, confidence))

            meta: Dict[str, Any] = {
                "width": info["width"],
                "height": info["height"],
                "mode": info["mode"],
                "aspect": round(info["width"] / max(1, info["height"]), 4),
                "resized_to": list(self._image_size),
                "decoded_natively": info["decoded_natively"],
                "padded": info["padded"],
                "source_type": info["source_type"],
                "embedding_dim": self._embedding_dim,
                "torch_available": self._torch_available,
                "backbone": self._backbone_name,
                "proj_seed": IMAGE_PROJ_SEED,
                "proj_salt": IMAGE_PROJ_SALT,
                "is_grayscale": scalars["is_grayscale"],
                "is_dark": scalars["is_dark"],
                "brightness": scalars["brightness"],
                "contrast": scalars["contrast"],
                "edge_density": scalars["edge_density"],
                "colorfulness": scalars["colorfulness"],
            }
            attach_native(meta, features)

            chunk = EncodedChunk(
                modality=Modality.IMAGE,
                vector=vector,
                confidence=round(confidence, 4),
                meta=meta,
            )
            self._stats["encoded"] += 1
            self._stats["total_pixels"] += int(info["width"]) * int(info["height"])
            if info["padded"]:
                self._stats["padded"] += 1
            self._format_counts[info["mode"]] = self._format_counts.get(info["mode"], 0) + 1

            if self._event_bus:
                self._event_bus.emit(
                    "multimodal.image_encoded",
                    {"dim": len(vector), "width": info["width"],
                     "height": info["height"], "confidence": chunk.confidence},
                    source=self.name,
                )
            return chunk
        except ImageDecodeError as exc:
            self._stats["errors"] += 1
            self._stats["decode_failures"] += 1
            logger.warning("Görüntü kodlama başarısız: %s", exc)
            raise
        except Exception as exc:
            self._stats["errors"] += 1
            logger.error("Görüntü kodlama hatası: %s", exc)
            raise ImageDecodeError(f"Görüntü kodlanamadı: {exc}") from exc

    def encode_batch(self, images: List[Any]) -> List[EncodedChunk]:
        """
        Görüntü listesini sırayla kodlar.

        Args:
            images: Görüntü listesi

        Returns:
            `EncodedChunk` listesi; hatalı öğeler için `None` yerine boş chunk
            (`confidence=0.0`, `meta["error"]` dolu)
        """
        self._stats["batches"] += 1
        results: List[EncodedChunk] = []
        for item in (images or []):
            try:
                results.append(self.encode(item))
            except Exception as exc:
                logger.warning("Batch görüntüsü kodlanamadı: %s", exc)
                results.append(self._empty_chunk(str(exc)))
        return results

    # ─── Özellik Çıkarımı ───────────────────────────────────────────────

    def _load_rgb_array(self, image: Any) -> Tuple["np.ndarray", Dict[str, Any]]:
        """
        Girdiyi (H, W, 3) float32 [0,1] dizisine indirger.

        Önce PIL ile **yerel** çözme denenir; başarısız olursa numpy zorlaması
        (coercion) devreye girer ve `padded=True` işaretlenir.

        Returns:
            (rgb dizisi, bilgi sözlüğü)
        """
        info: Dict[str, Any] = {
            "decoded_natively": False,
            "padded": False,
            "mode": "?",
            "width": 0,
            "height": 0,
            "source_type": type(image).__name__,
        }

        # ── 1) Yerel çözme yolu (PIL) ───────────────────────────────────
        native_error: Optional[ImageDecodeError] = None
        try:
            pil_image = self.to_pil(image)
            info["mode"] = str(getattr(pil_image, "mode", "?"))
            width, height = pil_image.size
            info["width"], info["height"] = int(width), int(height)
            working = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
            if working.size != self._image_size:
                working = working.resize(self._image_size, self._resample_filter())
            array = np.asarray(working, dtype=np.float32)
            info["decoded_natively"] = True
            if info["width"] < 8 or info["height"] < 8:
                info["padded"] = True
                array = self._pad_small(array)
            return np.clip(array / 255.0, 0.0, 1.0).astype(np.float32), info
        except ImageDecodeError as exc:
            native_error = exc
            # Yol/bayt girdilerinde numpy zorlaması anlamsızdır → hatayı koru
            if isinstance(image, (str, bytes, bytearray, memoryview)):
                raise
            logger.debug("Yerel çözme başarısız, numpy zorlaması denenecek: %s", exc)

        # ── 2) numpy zorlaması ──────────────────────────────────────────
        try:
            raw = np.asarray(image)
        except Exception as exc:
            raise ImageDecodeError(
                f"Görüntü ne PIL ne de numpy ile okunabildi: {exc}") from exc

        if raw.size == 0:
            raise ImageDecodeError(
                f"Görüntü verisi boş (0 öğe) — yerel çözme hatası: {native_error}")
        if raw.ndim not in (2, 3):
            raise ImageDecodeError(
                f"Görüntü dizisi 2/3 boyutlu olmalı (gelen: {raw.ndim}) — "
                f"yerel çözme hatası: {native_error}")

        info["padded"] = True
        info["decoded_natively"] = bool(raw.ndim == 3 and raw.shape[2] in (3, 4)
                                        and min(raw.shape[0], raw.shape[1]) >= 8)
        array = self._coerce_to_rgb(raw)
        info["height"], info["width"] = int(array.shape[0]), int(array.shape[1])
        info["mode"] = "numpy"
        if array.shape[0] < 8 or array.shape[1] < 8:
            array = self._pad_small(array)
            info["height"], info["width"] = int(array.shape[0]), int(array.shape[1])
        array = self._resize_nearest(array, self._image_size)
        return np.clip(array, 0.0, 1.0).astype(np.float32), info

    def _coerce_to_rgb(self, raw: "np.ndarray") -> "np.ndarray":
        """Herhangi bir numpy dizisini (H, W, 3) float32 [0,1] biçimine getirir."""
        data = raw
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        else:
            channels = data.shape[2]
            if channels == 1:
                data = np.repeat(data, 3, axis=2)
            elif channels == 2:
                data = np.repeat(data[:, :, :1], 3, axis=2)
            elif channels == 4:
                rgb = data[:, :, :3].astype(np.float32)
                alpha = data[:, :, 3:4].astype(np.float32)
                if np.issubdtype(data.dtype, np.floating) and float(np.nanmax(alpha)) <= 1.5:
                    alpha = alpha * 255.0
                # Şeffaf alanlar beyaz zeminle birleştirilir
                data = rgb * (alpha / 255.0) + 255.0 * (1.0 - alpha / 255.0)
            elif channels > 4:
                data = data[:, :, :3]

        if data.dtype == np.bool_:
            data = data.astype(np.float32)
            return data
        values = data.astype(np.float32)
        if np.issubdtype(data.dtype, np.integer):
            if data.dtype == np.uint8:
                return values / 255.0
            info = np.iinfo(data.dtype)
            return np.clip(values / max(1, info.max), 0.0, 1.0)
        finite = values[np.isfinite(values)]
        peak = float(finite.max()) if finite.size else 0.0
        values = np.nan_to_num(values, nan=0.0)
        if peak > 1.5:
            values = values / 255.0 if peak <= 255.0 else values / max(1.0, peak)
        return np.clip(values, 0.0, 1.0)

    @staticmethod
    def _pad_small(array: "np.ndarray", minimum: int = 8) -> "np.ndarray":
        """Çok küçük görüntüleri kenar tekrarıyla en az `minimum` boyuta genişletir."""
        height, width = array.shape[0], array.shape[1]
        pad_y = max(0, minimum - height)
        pad_x = max(0, minimum - width)
        if pad_y == 0 and pad_x == 0:
            return array
        return np.pad(array, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")

    @staticmethod
    def _resize_nearest(array: "np.ndarray", size: Tuple[int, int]) -> "np.ndarray":
        """PIL yoksa/yetmezse en yakın komşu yeniden boyutlandırma."""
        target_w, target_h = int(size[0]), int(size[1])
        height, width = array.shape[0], array.shape[1]
        if (height, width) == (target_h, target_w):
            return array
        ys = np.clip((np.arange(target_h) * height / target_h).astype(int), 0, height - 1)
        xs = np.clip((np.arange(target_w) * width / target_w).astype(int), 0, width - 1)
        return array[ys][:, xs]

    @staticmethod
    def _pool(matrix: "np.ndarray", out_h: int, out_w: int) -> "np.ndarray":
        """
        Bir 2B matrisi ortalama havuzlayarak (out_h, out_w) boyutuna indirir.

        Bölen olmayan boyutlarda da çalışır (kenar kutuları kısmi ortalama alır).
        """
        height, width = matrix.shape[0], matrix.shape[1]
        ys = np.linspace(0, height, out_h + 1).round().astype(int)
        xs = np.linspace(0, width, out_w + 1).round().astype(int)
        result = np.zeros((out_h, out_w), dtype=np.float64)
        for i in range(out_h):
            y0 = min(ys[i], height - 1)
            y1 = max(ys[i + 1], y0 + 1)
            for j in range(out_w):
                x0 = min(xs[j], width - 1)
                x1 = max(xs[j + 1], x0 + 1)
                result[i, j] = float(matrix[y0:y1, x0:x1].mean())
        return result

    def _feature_vector(self, rgb: "np.ndarray") -> Tuple["np.ndarray", Dict[str, float]]:
        """
        Ham 338 boyutlu özellik vektörünü ve skaler bloğu üretir.

        Args:
            rgb: (H, W, 3) float32 [0,1] dizi (hedef boyuta getirilmiş)

        Returns:
            (features (338,), scalars sözlüğü)
        """
        red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.299 * red + 0.587 * green + 0.114 * blue

        # (a) 8×8×3 ortalama-havuzlanmış RGB ızgara → 192
        rgb_grid = np.stack([self._pool(red, 8, 8),
                             self._pool(green, 8, 8),
                             self._pool(blue, 8, 8)], axis=-1).astype(np.float64).ravel()

        # (b) 8×8 gri ızgara → 64
        gray_grid = self._pool(gray, 8, 8).ravel()

        # (c) kanal başına 16 kutulu histogram (normalize) → 48
        histograms = []
        for channel in (red, green, blue):
            counts, _edges = np.histogram(channel, bins=16, range=(0.0, 1.0))
            total = float(counts.sum()) or 1.0
            histograms.append(counts.astype(np.float64) / total)
        histogram_block = np.concatenate(histograms)

        # (d) 4×4 gradyan büyüklüğü ızgarası → 16
        grad_y, grad_x = np.gradient(gray)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        magnitude_norm = np.clip(magnitude / 0.5, 0.0, 1.0)
        gradient_block = self._pool(magnitude_norm, 4, 4).ravel()

        # (e) 8 kutulu ton (hue) histogramı → 8
        hue_block = self._hue_histogram(rgb)

        # (f) skaler blok → 10
        scalars = self._scalars(rgb, gray, magnitude)
        scalar_block = np.asarray([
            min(1.0, (rgb.shape[1] / max(1, rgb.shape[0])) / 4.0),
            scalars["brightness"],
            scalars["contrast"],
            scalars["saturation"],
            scalars["edge_density"],
            scalars["entropy"],
            scalars["center_border_delta"],
            scalars["colorfulness"],
            1.0 if scalars["is_grayscale"] else 0.0,
            1.0 if scalars["is_dark"] else 0.0,
        ], dtype=np.float64)

        features = np.concatenate([
            rgb_grid, gray_grid, histogram_block,
            gradient_block, hue_block, scalar_block,
        ]).astype(np.float64)

        if features.shape[0] != NATIVE_FEATURE_DIM:  # pragma: no cover - tutarlılık kontrolü
            logger.debug("Özellik boyutu beklenenden farklı: %d != %d",
                         features.shape[0], NATIVE_FEATURE_DIM)
        return features, scalars

    def _hue_histogram(self, rgb: "np.ndarray") -> "np.ndarray":
        """8 kutulu, normalize edilmiş ton (hue) histogramı."""
        red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        maxc = np.max(rgb, axis=2)
        minc = np.min(rgb, axis=2)
        delta = maxc - minc
        mask = delta > 1e-6
        if not bool(mask.any()):
            return np.zeros((_HUE_BINS,), dtype=np.float64)

        hue = np.zeros_like(maxc)
        safe_delta = np.where(mask, delta, 1.0)
        red_is_max = mask & (maxc == red)
        green_is_max = mask & (maxc == green) & ~red_is_max
        blue_is_max = mask & ~red_is_max & ~green_is_max

        hue[red_is_max] = (((green - blue) / safe_delta)[red_is_max]) % 6.0
        hue[green_is_max] = ((blue - red) / safe_delta)[green_is_max] + 2.0
        hue[blue_is_max] = ((red - green) / safe_delta)[blue_is_max] + 4.0
        hue = (hue * 60.0)[mask]

        counts, _edges = np.histogram(hue, bins=_HUE_BINS, range=(0.0, 360.0))
        total = float(counts.sum()) or 1.0
        return counts.astype(np.float64) / total

    def _scalars(self, rgb: "np.ndarray", gray: "np.ndarray",
                 magnitude: "np.ndarray") -> Dict[str, float]:
        """
        Skaler görüntü istatistikleri (10 adet).

        Returns:
            brightness, contrast, saturation, edge_density, entropy,
            center_border_delta, colorfulness, is_grayscale, is_dark, aspect
        """
        height, width = gray.shape[0], gray.shape[1]
        brightness = float(gray.mean())
        contrast = float(gray.std())

        maxc = np.max(rgb, axis=2)
        minc = np.min(rgb, axis=2)
        saturation = float(np.mean((maxc - minc) / np.maximum(maxc, 1e-6)))

        edge_density = float(np.mean(magnitude > 0.1))

        counts, _edges = np.histogram(gray, bins=32, range=(0.0, 1.0))
        probabilities = counts.astype(np.float64) / max(1.0, float(counts.sum()))
        probabilities = probabilities[probabilities > 0]
        entropy = float(-np.sum(probabilities * np.log2(probabilities)) / 5.0)  # 0..1

        center_y0, center_y1 = height // 4, max(height // 4 + 1, (3 * height) // 4)
        center_x0, center_x1 = width // 4, max(width // 4 + 1, (3 * width) // 4)
        center_mean = float(gray[center_y0:center_y1, center_x0:center_x1].mean())
        border_mask = np.ones_like(gray, dtype=bool)
        border_mask[center_y0:center_y1, center_x0:center_x1] = False
        border_mean = float(gray[border_mask].mean()) if bool(border_mask.any()) else center_mean
        center_border_delta = float(np.clip((center_mean - border_mean + 1.0) / 2.0, 0.0, 1.0))

        # Hasler-Süsstrunk renklilik ölçütü (0..255 uzayında hesaplanır)
        rg = (rgb[:, :, 0] - rgb[:, :, 1]) * 255.0
        yb = (0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]) * 255.0
        colorfulness_raw = (math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2)
                            + 0.3 * math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2))
        colorfulness = float(np.clip(colorfulness_raw / 110.0, 0.0, 1.0))

        is_grayscale = bool(saturation < 0.06
                            and float(np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 1]))) < 0.02
                            and float(np.mean(np.abs(rgb[:, :, 1] - rgb[:, :, 2]))) < 0.02)
        is_dark = bool(brightness < 0.28)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "edge_density": edge_density,
            "entropy": entropy,
            "center_border_delta": center_border_delta,
            "colorfulness": colorfulness,
            "is_grayscale": is_grayscale,
            "is_dark": is_dark,
            "aspect": float(width / max(1, height)),
        }

    def _project(self, features: "np.ndarray", out_dim: int) -> List[float]:
        """
        Ham özellik vektörünü sabit tohumlu rastgele projeksiyonla `out_dim`'e taşır.

        Matris örnek üzerinde önbelleklenir (`self._proj_cache`), böylece tekrar
        kodlamalarda yeniden üretilmez.

        Args:
            features: Ham özellik vektörü
            out_dim: Hedef gömme boyutu

        Returns:
            L2-normalize edilmiş gömme vektörü
        """
        native_dim = int(features.shape[0])
        key = (native_dim, int(out_dim))
        matrix = self._proj_cache.get(key)
        if matrix is None:
            matrix = projection_matrix(native_dim, int(out_dim),
                                       IMAGE_PROJ_SEED, IMAGE_PROJ_SALT)
            self._proj_cache[key] = matrix
        projected = features.astype(np.float64) @ matrix
        return l2_normalize(projected.tolist())

    # ─── Tanımlama (describe) ───────────────────────────────────────────

    def describe(self, image: Any) -> dict:
        """
        Görüntüyü ayrıntılı biçimde betimler (log/teşhis amaçlı).

        Args:
            image: Yol | bytes | PIL.Image | np.ndarray

        Returns:
            width, height, mode, aspect, mean_colors, brightness, contrast,
            edge_density, dominant_colors (5 hex), is_grayscale, is_dark,
            has_transparency

        Raises:
            ImageDecodeError: Görüntü çözülemezse
        """
        pil_image = self.to_pil(image)
        mode = str(getattr(pil_image, "mode", "?"))
        width, height = (int(v) for v in pil_image.size)
        has_transparency = mode in ("RGBA", "LA", "PA") or "transparency" in pil_image.info

        rgb = self.to_array(pil_image)
        red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.299 * red + 0.587 * green + 0.114 * blue
        grad_y, grad_x = np.gradient(gray)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        scalars = self._scalars(rgb, gray, magnitude)

        mean_colors = {
            "r": round(float(red.mean()), 4),
            "g": round(float(green.mean()), 4),
            "b": round(float(blue.mean()), 4),
        }
        return {
            "width": width,
            "height": height,
            "mode": mode,
            "aspect": round(width / max(1, height), 4),
            "mean_colors": mean_colors,
            "mean_color_hex": self._rgb_to_hex(int(red.mean() * 255),
                                               int(green.mean() * 255),
                                               int(blue.mean() * 255)),
            "brightness": round(scalars["brightness"], 4),
            "contrast": round(scalars["contrast"], 4),
            "saturation": round(scalars["saturation"], 4),
            "edge_density": round(scalars["edge_density"], 4),
            "entropy": round(scalars["entropy"], 4),
            "colorfulness": round(scalars["colorfulness"], 4),
            "dominant_colors": self._dominant_colors(rgb),
            "is_grayscale": scalars["is_grayscale"],
            "is_dark": scalars["is_dark"],
            "has_transparency": bool(has_transparency),
            "perceptual_hash": self.perceptual_hash(pil_image),
        }

    def _dominant_colors(self, rgb: "np.ndarray", top: int = 5) -> List[str]:
        """
        En sık görülen renkleri (kaba 4-bit/kanal kuantalaması ile) hex olarak döndürür.

        Görüntüde `top`'tan az ayrı renk varsa liste daha kısa olur (dolgu yapılmaz).

        Args:
            rgb: (H, W, 3) float32 [0,1] dizi
            top: En büyük renk sayısı

        Returns:
            "#rrggbb" listesi (en sık → en seyrek)
        """
        quantized = np.clip((rgb * 15.0).astype(np.int32), 0, 15)
        codes = (quantized[:, :, 0] << 8) | (quantized[:, :, 1] << 4) | quantized[:, :, 2]
        values, counts = np.unique(codes.ravel(), return_counts=True)
        order = np.argsort(-counts)[:top]
        colors: List[str] = []
        for index in order:
            code = int(values[index])
            red = ((code >> 8) & 0xF) * 17
            green = ((code >> 4) & 0xF) * 17
            blue = (code & 0xF) * 17
            colors.append(self._rgb_to_hex(red, green, blue))
        return colors

    @staticmethod
    def _rgb_to_hex(red: int, green: int, blue: int) -> str:
        """RGB bileşenlerini `#rrggbb` biçimine çevirir."""
        return "#{:02x}{:02x}{:02x}".format(
            int(np.clip(red, 0, 255)), int(np.clip(green, 0, 255)), int(np.clip(blue, 0, 255)))

    # ─── Hash / Küçük Görsel ────────────────────────────────────────────

    def perceptual_hash(self, image: Any, hash_size: int = 8) -> str:
        """
        dHash (difference hash) üretir — algısal olarak benzer görüntüler
        benzer hash üretir.

        Args:
            image: Yol | bytes | PIL.Image | np.ndarray
            hash_size: Izgara boyutu (varsayılan 8 → 64 bit / 16 hex karakter)

        Returns:
            Hex string (küçük harf)
        """
        hash_size = max(2, int(hash_size))
        pil_image = self.to_pil(image)
        try:
            gray = pil_image.convert("L").resize((hash_size + 1, hash_size),
                                                 self._resample_filter())
        except Exception as exc:  # pragma: no cover - savunma
            raise ImageDecodeError(f"dHash için görüntü hazırlanamadı: {exc}") from exc
        pixels = np.asarray(gray, dtype=np.int16)
        bits = (pixels[:, 1:] > pixels[:, :-1]).astype(np.uint8).ravel()
        try:
            return np.packbits(bits).tobytes().hex()
        except Exception:  # pragma: no cover - savunma
            return "".join(str(int(b)) for b in bits)

    def thumbnail_b64(self, image: Any, max_side: int = 256) -> str:
        """
        Küçük bir PNG küçük resmini base64 veri URI'si olarak döndürür.

        Args:
            image: Yol | bytes | PIL.Image | np.ndarray
            max_side: En büyük kenarın piksel sınırı

        Returns:
            "data:image/png;base64,..." biçiminde string
        """
        max_side = max(8, int(max_side))
        pil_image = self.to_pil(image)
        if pil_image.mode not in ("RGB", "RGBA", "L"):
            pil_image = pil_image.convert("RGB")
        width, height = pil_image.size
        scale = min(1.0, max_side / max(1, max(width, height)))
        target = (max(1, int(width * scale)), max(1, int(height * scale)))
        thumb = pil_image.resize(target, self._resample_filter())
        buffer = io.BytesIO()
        thumb.save(buffer, format="PNG", optimize=False)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    # ─── Opsiyonel Derin Omurga ─────────────────────────────────────────

    def load_backbone(self, path: str) -> bool:
        """
        Opsiyonel ResNet-18 omurgasını bir kontrol noktasından yükler.

        Ağırlık yüklenmeden `encode_deep()` **kullanılmaz**; rastgele başlatılmış
        omurga anlamsız özellik üretir (yalnızca boru hattı testi için anlamlıdır).

        Args:
            path: `torch.save` ile kaydedilmiş state_dict dosyası

        Returns:
            True → omurga hazır, False → yüklenemedi (torch yok / dosya hatalı)
        """
        if not self._torch_available:
            logger.warning("torch/torchvision yok — derin omurga yüklenemez")
            return False
        try:
            import torch  # ağır import — fonksiyon içinde
            import torchvision

            model = torchvision.models.resnet18(weights=None)
            model.fc = torch.nn.Identity()
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.warning("Omurga eksik anahtarlarla yüklendi: %d adet", len(missing))
            if unexpected:
                logger.debug("Omurgada fazladan anahtar: %d adet", len(unexpected))
            model.eval()
            self._backbone = model
            self._backbone_name = f"resnet18:{path}"
            logger.info("Görüntü omurgası yüklendi: %s", path)
            if self._event_bus:
                self._event_bus.emit("multimodal.image_backbone_loaded",
                                     {"path": str(path)}, source=self.name)
            return True
        except Exception as exc:
            logger.error("Omurga yüklenemedi (%s): %s", path, exc)
            return False

    def encode_deep(self, image: Any) -> EncodedChunk:
        """
        Opsiyonel derin kodlama (ResNet-18, 512 boyutlu özellik).

        `load_backbone()` çağrılmadıysa `EncoderUnavailable` yükseltir; `encode()`
        bu metodu **asla** kendiliğinden çağırmaz.

        Args:
            image: Yol | bytes | PIL.Image | np.ndarray

        Returns:
            `Modality.IMAGE` tipli `EncodedChunk` (meta["deep"] = True)

        Raises:
            EncoderUnavailable: Omurga yüklü değilse veya torch yoksa
        """
        if self._backbone is None:
            raise EncoderUnavailable(
                "Derin omurga yüklü değil — önce load_backbone(path) çağırın "
                "(eğitilmemiş ResNet anlamsız özellik üretir)")
        try:
            import torch

            rgb = self.to_array(image, size=self._image_size)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
            mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
            backbone = self._backbone.to(self._torch_device)
            with torch.no_grad():
                features = backbone(tensor.to(self._torch_device)).squeeze(0).cpu().numpy()
            vector = self._project(np.asarray(features, dtype=np.float64), self._embedding_dim)
            meta: Dict[str, Any] = {
                "deep": True,
                "backbone": self._backbone_name or "resnet18",
                "embedding_dim": self._embedding_dim,
                "torch_available": True,
                "trained": True,
            }
            attach_native(meta, features)
            self._stats["deep_encoded"] += 1
            if self._event_bus:
                self._event_bus.emit("multimodal.image_deep_encoded",
                                     {"dim": len(vector)}, source=self.name)
            return EncodedChunk(modality=Modality.IMAGE, vector=vector,
                                confidence=0.85, meta=meta)
        except EncoderUnavailable:
            raise
        except Exception as exc:
            logger.error("Derin görüntü kodlama hatası: %s", exc)
            raise EncoderUnavailable(f"Derin görüntü kodlama başarısız: {exc}") from exc

    # ─── İç Yardımcılar ─────────────────────────────────────────────────

    def _empty_chunk(self, error: str) -> EncodedChunk:
        """Hata durumunda döndürülen boş chunk."""
        return EncodedChunk(
            modality=Modality.IMAGE,
            vector=[0.0] * self._embedding_dim,
            confidence=0.0,
            meta={"error": error, "embedding_dim": self._embedding_dim},
        )

    def _import_pil(self) -> Any:
        """Pillow'u tembel yükler; yoksa Türkçe mesajlı hata yükseltir."""
        try:
            from PIL import Image  # ağır/opsiyonel import — fonksiyon içinde

            return Image
        except Exception as exc:  # pragma: no cover - Pillow kurulu değilse
            raise ImageDecodeError(
                f"Pillow (PIL) kurulu değil, görüntü çözülemez: {exc}") from exc

    def _resample_filter(self) -> Any:
        """Yeniden boyutlandırma filtresi (PIL varsa LANCZOS, yoksa None)."""
        try:
            from PIL import Image

            return Image.LANCZOS
        except Exception:  # pragma: no cover - savunma
            return None

    @staticmethod
    def _is_pil(image: Any) -> bool:
        """Nesnenin PIL görüntüsü olup olmadığını (import etmeden) sezer."""
        module = type(image).__module__.split(".")[0]
        return module == "PIL" and hasattr(image, "mode") and hasattr(image, "size")

    @staticmethod
    def _probe_module(name: str) -> bool:
        """Modül kurulu mu (import etmeden, ucuz kontrol)?"""
        try:
            import importlib.util

            return importlib.util.find_spec(name) is not None
        except Exception:  # pragma: no cover - savunma
            return False

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
    def _read_image_size(config: Any) -> Tuple[int, int]:
        """`config.multimodal.image_size` değerini (genişlik, yükseklik) olarak okur."""
        if config is None:
            return _DEFAULT_IMAGE_SIZE
        try:
            size = config.multimodal.image_size
            width, height = int(size[0]), int(size[1])
            if width > 0 and height > 0:
                return (width, height)
        except Exception:
            pass
        return _DEFAULT_IMAGE_SIZE

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        JSON-serileştirilebilir istatistikleri döndürür.

        Returns:
            Kodlama sayaçları, yetenek bayrakları ve önbellek durumu
        """
        encoded = max(1, self._stats["encoded"])
        return {
            "name": self.name,
            "initialized": self._initialized,
            "embedding_dim": self._embedding_dim,
            "image_size": list(self._image_size),
            "native_feature_dim": NATIVE_FEATURE_DIM,
            "pillow_available": self._pillow_available,
            "torch_available": self._torch_available,
            "backbone_loaded": self._backbone is not None,
            "backbone_name": self._backbone_name,
            "counters": dict(self._stats),
            "avg_pixels_per_image": round(self._stats["total_pixels"] / encoded, 1),
            "mode_distribution": dict(self._format_counts),
            "projection_cache_entries": len(self._proj_cache),
            "proj_seed": IMAGE_PROJ_SEED,
            "proj_salt": IMAGE_PROJ_SALT,
        }


__all__ = ["ImageEncoder", "NATIVE_FEATURE_DIM", "IMAGE_PROJ_SEED", "IMAGE_PROJ_SALT"]
