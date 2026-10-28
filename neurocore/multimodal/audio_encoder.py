"""
NeuroCore Ses Kodlayıcı (L4)
=============================
`librosa`/`soundfile` **gerektirmeyen**, saf numpy ile FFT tabanlı mel
spektrogram ve spektral istatistikler çıkaran ses kodlayıcı.

Yükleme:
    - `.wav` → stdlib `wave` modülü (yerel, bağımlılıksız)
    - diğer biçimler → `soundfile` varsa; yoksa Türkçe mesajlı `AudioDecodeError`
    - bytes / np.ndarray / (samples, sample_rate) demeti doğrudan kabul edilir

Yeniden örnekleme:
    - `scipy.signal.resample_poly` varsa yüksek kaliteli yol kullanılır
    - yoksa numpy doğrusal interpolasyon uygulanır (zorunlu bağımlılık yok)

Özellik vektörü (ham/native boyut = 317):
    26 bant ortalaması + 26 bant std + 13 bant delta-ortalaması   →  65
    26×8 zaman-havuzlanmış log-mel ızgara                         → 208
    ZCR istatistikleri                                            →   4
    RMS enerji istatistikleri                                     →   4
    Spektral centroid/rolloff/flatness/bandwidth istatistikleri   →  16
    Chroma-benzeri 12 bant enerjisi                               →  12
    Tempo kestirimi (otokorelasyon)                               →   2
    Skaler blok                                                   →   6

Ardından sabit tohumlu projeksiyon (`seed=2024`, `salt="neurocore-audio-proj"`)
ile `embedding_dim` boyutuna taşınır ve L2-normalize edilir.

Örnek:
    enc = AudioEncoder()
    enc.initialize(config, bus)
    chunk = enc.encode("kayit.wav")
    print(enc.describe("kayit.wav")["tempo_bpm"])
"""

from __future__ import annotations

import io
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neurocore.multimodal.types import (
    DEFAULT_EMBEDDING_DIM,
    AudioDecodeError,
    EncodedChunk,
    EncoderUnavailable,
    Modality,
    attach_native,
    l2_normalize,
    projection_matrix,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("multimodal.audio")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: Projeksiyon tohumu ve tuzu (deterministik — değiştirilmemeli).
AUDIO_PROJ_SEED = 2024
AUDIO_PROJ_SALT = "neurocore-audio-proj"

#: Mel filtre sayısı ve ham özellik boyutu.
N_MELS = 26
NATIVE_FEATURE_DIM = 65 + 208 + 4 + 4 + 16 + 12 + 2 + 6

#: Varsayılan hedef örnekleme hızı.
DEFAULT_SAMPLE_RATE = 16000

#: Mel filtrebank üst sınırı (Hz).
MEL_FMAX = 8000.0

#: Özellik çıkarımında işlenecek en büyük kare sayısı (~60 sn @ 10 ms hop).
MAX_FRAMES = 6000

#: Sessizlik eşiği (RMS).
SILENCE_RMS = 1e-4

#: Kırpılma (clipping) eşiği.
CLIPPING_LEVEL = 0.99

#: ses dosyası sihirli bayt imzaları → biçim adı.
_AUDIO_MAGIC: Tuple[Tuple[bytes, str], ...] = (
    (b"RIFF", "wav"),
    (b"fLaC", "flac"),
    (b"OggS", "ogg"),
    (b"ID3", "mp3"),
    (b"\xff\xfb", "mp3"),
    (b"\xff\xf3", "mp3"),
    (b"\xff\xf2", "mp3"),
    (b"\xff\xfa", "mp3"),
    (b"\xff\xf1", "aac"),
)


# ─── Ana Kodlayıcı Sınıfı ──────────────────────────────────────────────────

class AudioEncoder:
    """
    Hafif, deterministik ses kodlayıcı (mel spektrogram + spektral istatistikler).

    Hiçbir eğitilmiş model gerektirmez; `torch`+`torchaudio` varsa isteğe bağlı
    `encode_deep()` (ve `load_backbone()`) kullanılabilir.
    """

    name = "multimodal.audio"

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — G/Ç veya ağır import yok."""
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._embedding_dim = DEFAULT_EMBEDDING_DIM
        self._target_sr = DEFAULT_SAMPLE_RATE
        self._soundfile_available = False
        self._scipy_available = False
        self._torch_available = False
        self._backbone = None
        self._backbone_name: Optional[str] = None
        self._proj_cache: Dict[Tuple[int, int], "np.ndarray"] = {}
        self._mel_cache: Dict[Tuple[int, int, float], "np.ndarray"] = {}
        self._stats: Dict[str, Any] = {
            "encoded": 0,
            "batches": 0,
            "errors": 0,
            "decode_failures": 0,
            "silent": 0,
            "resampled": 0,
            "deep_encoded": 0,
        }
        self._format_counts: Dict[str, int] = {}

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Kodlayıcıyı yapılandırır (idempotent).

        Args:
            config: `SystemConfig`; `config.multimodal.audio_sample_rate` kullanılır
            event_bus: Olay yolu (opsiyonel)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._embedding_dim = self._read_embedding_dim(config)
        self._target_sr = self._read_sample_rate(config)
        self._soundfile_available = self._probe_module("soundfile")
        self._scipy_available = self._probe_module("scipy")
        self._torch_available = (self._probe_module("torch")
                                 and self._probe_module("torchaudio"))
        self._initialized = True
        logger.info("Ses kodlayıcı hazır (dim=%d, sr=%d, soundfile=%s, scipy=%s)",
                    self._embedding_dim, self._target_sr,
                    self._soundfile_available, self._scipy_available)
        if self._event_bus:
            self._event_bus.emit(
                "multimodal.audio_encoder_ready",
                {"dim": self._embedding_dim, "sample_rate": self._target_sr,
                 "soundfile_available": self._soundfile_available},
                source=self.name,
            )

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır (idempotent)."""
        if not self._initialized:
            return
        self._backbone = None
        self._backbone_name = None
        self._proj_cache.clear()
        self._mel_cache.clear()
        self._initialized = False
        logger.info("Ses kodlayıcı kapatıldı")

    # ─── Yükleme ────────────────────────────────────────────────────────

    def load(self, audio: Any, target_sr: Optional[int] = None) -> Tuple["np.ndarray", int]:
        """
        Herhangi bir ses girdisini mono float32 [-1,1] dalga formuna çevirir.

        Args:
            audio: Yol (str) | bytes | np.ndarray | (samples, sample_rate) demeti
            target_sr: Hedef örnekleme hızı (None → `config.multimodal.audio_sample_rate`)

        Returns:
            (mono float32 dizi, hedef örnekleme hızı)

        Raises:
            AudioDecodeError: Girdi çözülemezse (Türkçe mesajla)
        """
        samples, rate, _info = self._load_with_info(audio, target_sr)
        return samples, rate

    def _load_with_info(self, audio: Any, target_sr: Optional[int] = None
                        ) -> Tuple["np.ndarray", int, Dict[str, Any]]:
        """
        `load()` ile aynı işi yapar, ayrıca çözme/örnekleme meta bilgisini döndürür.

        Args:
            audio: Ses girdisi
            target_sr: Hedef örnekleme hızı (None → config değeri)

        Returns:
            (mono float32 dizi, hedef hız, bilgi sözlüğü)
        """
        target = int(target_sr) if target_sr else self._target_sr
        samples, source_sr, info = self._decode(audio)

        if samples.size == 0:
            raise AudioDecodeError("Ses verisi boş — kodlanacak örnek yok")

        samples = self._to_mono(samples)
        if source_sr and int(source_sr) != int(target):
            samples = self._resample(samples, int(source_sr), int(target))
            info["resampled"] = True
        else:
            info["resampled"] = False

        samples = np.clip(np.nan_to_num(samples, nan=0.0), -1.0, 1.0).astype(np.float32)
        info["source_sample_rate"] = int(source_sr) if source_sr else int(target)
        if info["resampled"]:
            self._stats["resampled"] += 1
        return samples, int(target), info

    def _decode(self, audio: Any) -> Tuple["np.ndarray", int, Dict[str, Any]]:
        """
        Ham girdiyi çözer ve (samples, sample_rate, bilgi) üçlüsü döndürür.

        Örnekleme hızı bilinmiyorsa (çıplak dizi) hedef hız varsayılır ve
        `info["assumed_sample_rate"] = True` işaretlenir.
        """
        info: Dict[str, Any] = {
            "format": "unknown",
            "assumed_sample_rate": False,
            "channels": 1,
            "resampled": False,
        }

        # ── (samples, sample_rate) demeti ───────────────────────────────
        if isinstance(audio, (tuple, list)) and len(audio) == 2 \
                and isinstance(audio[1], (int, float)) and not isinstance(audio[1], bool):
            samples = np.asarray(audio[0])
            rate = int(audio[1])
            if rate <= 0:
                raise AudioDecodeError(f"Geçersiz örnekleme hızı: {rate}")
            info["format"] = "tuple"
            info["channels"] = self._channel_count(samples)
            return self._to_float(samples), rate, info

        # ── numpy dizisi ────────────────────────────────────────────────
        if self._is_ndarray(audio):
            info["format"] = "ndarray"
            info["channels"] = self._channel_count(audio)
            info["assumed_sample_rate"] = True
            return self._to_float(audio), int(self._target_sr), info

        # ── Baytlar ─────────────────────────────────────────────────────
        if isinstance(audio, (bytes, bytearray, memoryview)):
            data = bytes(audio)
            if not data:
                raise AudioDecodeError("Ses bayt dizisi boş")
            return self._decode_bytes(data, info)

        # ── Dosya yolu ──────────────────────────────────────────────────
        if isinstance(audio, str):
            path = audio
            if not os.path.exists(path):
                raise AudioDecodeError(f"Ses dosyası bulunamadı: {path}")
            extension = os.path.splitext(path)[1].lower()
            try:
                with open(path, "rb") as handle:
                    head = handle.read(16)
            except OSError as exc:
                raise AudioDecodeError(f"Ses dosyası okunamadı ({path}): {exc}") from exc

            if extension == ".wav" or head[:4] == b"RIFF":
                try:
                    with open(path, "rb") as handle:
                        return self._decode_bytes(handle.read(), info, kind="wav")
                except AudioDecodeError:
                    raise
                except Exception as exc:
                    raise AudioDecodeError(f"WAV dosyası çözülemedi ({path}): {exc}") from exc

            samples, rate = self._read_with_soundfile(path)
            info["format"] = extension.lstrip(".") or "soundfile"
            info["channels"] = self._channel_count(samples)
            return samples, rate, info

        raise AudioDecodeError(
            f"Desteklenmeyen ses girdisi tipi: {type(audio).__name__}")

    def _decode_bytes(self, data: bytes, info: Dict[str, Any],
                      kind: Optional[str] = None) -> Tuple["np.ndarray", int, Dict[str, Any]]:
        """Bayt dizisini (WAV → stdlib, diğerleri → soundfile) çözer."""
        detected = kind or self._sniff_format(data)
        info["format"] = detected
        if detected == "wav":
            samples, rate = self._read_wav_bytes(data)
            info["channels"] = self._channel_count(samples)
            return samples, rate, info

        if self._soundfile_available:
            samples, rate = self._read_with_soundfile(io.BytesIO(data))
            info["channels"] = self._channel_count(samples)
            return samples, rate, info

        raise AudioDecodeError(
            f"'{detected}' biçimi çözülemiyor: 'soundfile' paketi kurulu değil "
            f"(wav dışı biçimler için `pip install soundfile` gerekir)")

    def _read_wav_bytes(self, data: bytes) -> Tuple["np.ndarray", int]:
        """
        WAV verisini stdlib `wave` modülü ile okur (yerel, bağımlılıksız).

        Args:
            data: WAV dosyasının tam bayt içeriği

        Returns:
            (float32 dizi, örnekleme hızı)

        Raises:
            AudioDecodeError: WAV başlığı/verisi bozuksa
        """
        import wave  # stdlib — fonksiyon içinde

        try:
            with wave.open(io.BytesIO(data), "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                rate = handle.getframerate()
                frames = handle.getnframes()
                raw = handle.readframes(frames)
        except Exception as exc:
            raise AudioDecodeError(f"WAV verisi ayrıştırılamadı: {exc}") from exc

        if not raw:
            raise AudioDecodeError("WAV dosyası ses çerçevesi içermiyor")
        if rate <= 0:
            raise AudioDecodeError(f"WAV örnekleme hızı geçersiz: {rate}")

        samples = self._pcm_to_float(raw, sample_width)
        if channels > 1:
            usable = (samples.size // channels) * channels
            samples = samples[:usable].reshape(-1, channels)
        return samples, int(rate)

    @staticmethod
    def _pcm_to_float(raw: bytes, sample_width: int) -> "np.ndarray":
        """
        PCM baytlarını float32 [-1,1] dizisine çevirir (8/16/24/32 bit).

        Args:
            raw: Ham PCM baytları
            sample_width: Örnek başına bayt (1, 2, 3, 4)

        Returns:
            float32 dizi (çok kanallıysa düz)
        """
        if sample_width == 1:
            values = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            return (values - 128.0) / 128.0
        if sample_width == 2:
            return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if sample_width == 3:
            usable = (len(raw) // 3) * 3
            triplets = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(-1, 3)
            as_int = (triplets[:, 0].astype(np.int32)
                      | (triplets[:, 1].astype(np.int32) << 8)
                      | (triplets[:, 2].astype(np.int32) << 16))
            as_int = np.where(as_int >= (1 << 23), as_int - (1 << 24), as_int)
            return as_int.astype(np.float32) / 8388608.0
        if sample_width == 4:
            return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        raise AudioDecodeError(f"Desteklenmeyen PCM örnek genişliği: {sample_width} bayt")

    def _read_with_soundfile(self, source: Any) -> Tuple["np.ndarray", int]:
        """
        `soundfile` ile ses okur (yol veya dosya-benzeri nesne).

        Args:
            source: Dosya yolu veya `io.BytesIO`

        Returns:
            (float32 dizi, örnekleme hızı)

        Raises:
            AudioDecodeError: soundfile yoksa veya okuma başarısızsa
        """
        try:
            import soundfile  # opsiyonel ağır import — fonksiyon içinde
        except Exception as exc:
            raise AudioDecodeError(
                "Bu biçim için 'soundfile' paketi gerekli (kurulu değil): "
                f"{exc}") from exc
        try:
            data, rate = soundfile.read(source, dtype="float32", always_2d=False)
        except Exception as exc:
            raise AudioDecodeError(f"Ses verisi 'soundfile' ile okunamadı: {exc}") from exc
        return np.asarray(data, dtype=np.float32), int(rate)

    # ─── Dönüştürme Yardımcıları ────────────────────────────────────────

    @staticmethod
    def _to_float(samples: Any) -> "np.ndarray":
        """Çeşitli PCM tiplerini float32 [-1,1] biçimine getirir."""
        data = np.asarray(samples)
        if data.dtype == np.uint8:
            return (data.astype(np.float32) - 128.0) / 128.0
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            scale = float(max(abs(info.min), info.max)) or 1.0
            return data.astype(np.float32) / scale
        return np.nan_to_num(data.astype(np.float32), nan=0.0)

    @staticmethod
    def _channel_count(samples: Any) -> int:
        """Örnek dizisinin kanal sayısını (kaba) tahmin eder."""
        data = np.asarray(samples)
        if data.ndim == 1:
            return 1
        if data.ndim == 2:
            return int(min(data.shape))
        return 1

    @staticmethod
    def _to_mono(samples: "np.ndarray") -> "np.ndarray":
        """Çok kanallı veriyi kanal ortalamasıyla mono'ya indirger."""
        if samples.ndim == 1:
            return samples
        if samples.ndim == 2:
            # (n_samples, n_channels) varsayımı; kanallar ilk boyutta ve
            # örnek sayısından küçükse transpoze edilir.
            if samples.shape[0] <= 8 and samples.shape[1] > samples.shape[0]:
                return samples.mean(axis=0)
            return samples.mean(axis=1)
        return samples.reshape(-1)

    def _resample(self, samples: "np.ndarray", source_sr: int,
                  target_sr: int) -> "np.ndarray":
        """
        Doğrusal (veya scipy varsa polyphase) yeniden örnekleme.

        Args:
            samples: mono float32 dizi
            source_sr: Kaynak hız
            target_sr: Hedef hız

        Returns:
            Yeniden örneklenmiş float32 dizi
        """
        if source_sr == target_sr or samples.size == 0:
            return samples

        if self._scipy_available:
            try:
                from math import gcd

                from scipy.signal import resample_poly  # opsiyonel — fonksiyon içinde

                divisor = gcd(int(source_sr), int(target_sr))
                up = int(target_sr) // divisor
                down = int(source_sr) // divisor
                if 0 < up <= 512 and 0 < down <= 512:
                    return np.asarray(resample_poly(samples, up, down),
                                      dtype=np.float32)
            except Exception as exc:
                logger.debug("scipy resample_poly kullanılamadı: %s", exc)

        # numpy doğrusal interpolasyon
        target_count = max(1, int(round(samples.size * float(target_sr) / float(source_sr))))
        source_axis = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        target_axis = np.linspace(0.0, 1.0, target_count, endpoint=False)
        return np.interp(target_axis, source_axis, samples).astype(np.float32)

    @staticmethod
    def _sniff_format(data: bytes) -> str:
        """Sihirli baytlardan ses biçimini sezer."""
        for signature, name in _AUDIO_MAGIC:
            if data.startswith(signature):
                return name
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return "wav"
        return "unknown"

    # ─── Kodlama ────────────────────────────────────────────────────────

    def encode(self, audio: Any, sample_rate: Optional[int] = None) -> EncodedChunk:
        """
        Sesi `EncodedChunk`'a kodlar.

        Args:
            audio: Yol | bytes | np.ndarray | (samples, sample_rate)
            sample_rate: Çıplak dizi/bayt girdiler için **kaynak** örnekleme hızı
                (verilmezse hedef hız varsayılır). Sinyal her durumda
                `config.multimodal.audio_sample_rate` hedefine yeniden örneklenir.

        Returns:
            `Modality.AUDIO` tipli `EncodedChunk`

        Raises:
            AudioDecodeError: Ses çözülemezse veya boşsa
        """
        try:
            payload = audio
            if sample_rate is not None and self._is_bare_samples(audio):
                payload = (audio, int(sample_rate))

            samples, sr, info = self._load_with_info(payload)
            features, detail = self._compute_features(samples, sr)
            detail.update({
                "format": info.get("format", "unknown"),
                "channels": int(info.get("channels", 1)),
                "source_sample_rate": int(info.get("source_sample_rate", sr)),
                "info_resampled": bool(info.get("resampled", False)),
                "info_assumed_sample_rate": bool(info.get("assumed_sample_rate", False)),
            })
            vector = self._project(features, self._embedding_dim)

            confidence = 0.5
            if not detail["info_assumed_sample_rate"]:
                confidence += 0.2
            if not detail["is_silent"]:
                confidence += 0.1
            if detail["info_assumed_sample_rate"]:
                confidence -= 0.15
            if detail["duration_sec"] < 0.05:
                confidence -= 0.1
            confidence = max(0.05, min(0.95, confidence))

            meta: Dict[str, Any] = {
                "duration_sec": round(detail["duration_sec"], 4),
                "n_samples": int(samples.size),
                "sample_rate": int(sr),
                "source_sample_rate": detail["source_sample_rate"],
                "resampled": detail["info_resampled"],
                "assumed_sample_rate": detail["info_assumed_sample_rate"],
                "format": detail["format"],
                "channels": detail["channels"],
                "rms": round(detail["rms_mean"], 6),
                "peak": round(detail["peak"], 6),
                "is_silent": detail["is_silent"],
                "clipping_ratio": round(detail["clipping_ratio"], 6),
                "dynamic_range_db": round(detail["dynamic_range_db"], 3),
                "zcr": round(detail["zcr_mean"], 6),
                "spectral_centroid": round(detail["centroid_mean"], 3),
                "spectral_flatness": round(detail["flatness_mean"], 6),
                "tempo_bpm": round(detail["tempo_bpm"], 2),
                "speech_likeness": round(detail["speech_likeness"], 4),
                "n_frames": detail["n_frames"],
                "n_mels": N_MELS,
                "embedding_dim": self._embedding_dim,
                "torch_available": self._torch_available,
                "proj_seed": AUDIO_PROJ_SEED,
                "proj_salt": AUDIO_PROJ_SALT,
            }
            attach_native(meta, features)

            chunk = EncodedChunk(
                modality=Modality.AUDIO,
                vector=vector,
                confidence=round(confidence, 4),
                meta=meta,
            )
            self._stats["encoded"] += 1
            if detail["is_silent"]:
                self._stats["silent"] += 1
            self._format_counts[detail["format"]] = \
                self._format_counts.get(detail["format"], 0) + 1

            if self._event_bus:
                self._event_bus.emit(
                    "multimodal.audio_encoded",
                    {"dim": len(vector), "duration_sec": meta["duration_sec"],
                     "tempo_bpm": meta["tempo_bpm"], "confidence": chunk.confidence},
                    source=self.name,
                )
            return chunk
        except AudioDecodeError as exc:
            self._stats["errors"] += 1
            self._stats["decode_failures"] += 1
            logger.warning("Ses kodlama başarısız: %s", exc)
            raise
        except Exception as exc:
            self._stats["errors"] += 1
            logger.error("Ses kodlama hatası: %s", exc)
            raise AudioDecodeError(f"Ses kodlanamadı: {exc}") from exc

    def encode_batch(self, audios: List[Any]) -> List[EncodedChunk]:
        """
        Ses listesini sırayla kodlar.

        Args:
            audios: Ses girdileri listesi

        Returns:
            `EncodedChunk` listesi (hatalı öğeler için boş chunk)
        """
        self._stats["batches"] += 1
        results: List[EncodedChunk] = []
        for item in (audios or []):
            try:
                results.append(self.encode(item))
            except Exception as exc:
                logger.warning("Batch sesi kodlanamadı: %s", exc)
                results.append(self._empty_chunk(str(exc)))
        return results

    # ─── Özellik Çıkarımı ───────────────────────────────────────────────

    def _compute_features(self, samples: "np.ndarray",
                          sr: int) -> Tuple["np.ndarray", Dict[str, Any]]:
        """
        Ham 317 boyutlu özellik vektörünü ve okunabilir istatistikleri üretir.

        Args:
            samples: mono float32 [-1,1] dizi
            sr: örnekleme hızı

        Returns:
            (features, detay sözlüğü)
        """
        frame_length = max(16, int(0.025 * sr))
        hop = max(1, int(0.010 * sr))
        n_fft = 1 << int(math.ceil(math.log2(frame_length)))

        frames, frame_indices = self._frame(samples, frame_length, hop)
        window = np.hanning(frame_length).astype(np.float32)
        windowed = frames * window
        spectrum = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1)).astype(np.float64)
        power = spectrum ** 2

        mel_filters = self._mel_filterbank(sr, n_fft)
        mel_energy = power @ mel_filters.T                      # (n_frames, N_MELS)
        log_mel = np.log(mel_energy + 1e-6)

        n_frames = int(log_mel.shape[0])
        duration = float(samples.size) / float(sr) if sr else 0.0

        # ── (1) Bant istatistikleri: 26 ort + 26 std + 13 delta → 65 ────
        band_mean = log_mel.mean(axis=0)
        band_std = log_mel.std(axis=0)
        if n_frames >= 2:
            delta = np.diff(log_mel, axis=0).mean(axis=0)[:13]
        else:
            delta = np.zeros((13,), dtype=np.float64)

        # ── (2) 26×8 zaman-havuzlanmış log-mel ızgara → 208 ─────────────
        pooled = self._pool_time(log_mel, 8)                    # (8, N_MELS)
        time_grid = pooled.T.ravel()                            # bant-öncelikli

        # ── (3) ZCR istatistikleri → 4 ──────────────────────────────────
        signs = np.signbit(frames)
        zcr_per_frame = np.mean(signs[:, 1:] != signs[:, :-1], axis=1) \
            if frame_length > 1 else np.zeros((n_frames,))
        zcr_stats = self._stats4(zcr_per_frame)

        # ── (4) RMS enerji istatistikleri → 4 ───────────────────────────
        rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        rms_stats = self._stats4(rms_per_frame)

        # ── (5) Spektral istatistikler → 16 ─────────────────────────────
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
        total_power = np.sum(power, axis=1) + 1e-12
        centroid = (power @ freqs) / total_power
        cumulative = np.cumsum(power, axis=1)
        rolloff_index = np.argmax(cumulative >= 0.85 * total_power[:, None], axis=1)
        rolloff = freqs[np.clip(rolloff_index, 0, freqs.size - 1)]
        flatness = np.exp(np.mean(np.log(power + 1e-12), axis=1)) / \
            (np.mean(power, axis=1) + 1e-12)
        bandwidth = np.sqrt(
            (power @ (freqs ** 2)) / total_power - centroid ** 2)
        bandwidth = np.nan_to_num(bandwidth, nan=0.0)
        spectral_block = np.concatenate([
            self._stats4(centroid), self._stats4(rolloff),
            self._stats4(flatness), self._stats4(bandwidth),
        ])

        # ── (6) Chroma-benzeri 12 bant enerjisi → 12 ────────────────────
        chroma = self._chroma(power, freqs)

        # ── (7) Tempo kestirimi → 2 ─────────────────────────────────────
        tempo_bpm, tempo_strength = self._estimate_tempo(rms_per_frame, sr, hop)

        # ── (8) Skaler blok → 6 ─────────────────────────────────────────
        rms_mean = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2) + 1e-12))
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        is_silent = bool(rms_mean < SILENCE_RMS)
        clipping_ratio = float(np.mean(np.abs(samples) > CLIPPING_LEVEL)) \
            if samples.size else 0.0
        quiet = float(np.percentile(rms_per_frame, 10)) if n_frames else 0.0
        dynamic_range_db = float(np.clip(
            20.0 * math.log10((rms_mean + 1e-9) / (quiet + 1e-9)), 0.0, 60.0))
        sample_rate_ratio = float(min(1.0, float(sr) / 16000.0))
        voice_ratio = self._voice_band_ratio(power, freqs)
        speech_likeness = self._speech_likeness(
            voice_ratio, float(np.mean(zcr_per_frame)) if n_frames else 0.0,
            float(np.mean(flatness)) if n_frames else 0.0)

        scalar_block = np.asarray([
            min(1.0, duration / 60.0),
            1.0 if is_silent else 0.0,
            clipping_ratio,
            dynamic_range_db / 60.0,
            sample_rate_ratio,
            speech_likeness,
        ], dtype=np.float64)

        features = np.concatenate([
            band_mean, band_std, delta, time_grid, zcr_stats, rms_stats,
            spectral_block, chroma,
            np.asarray([min(1.0, tempo_bpm / 240.0), tempo_strength], dtype=np.float64),
            scalar_block,
        ]).astype(np.float64)

        detail: Dict[str, Any] = {
            "duration_sec": duration,
            "n_frames": n_frames,
            "frame_length": frame_length,
            "hop": hop,
            "n_fft": n_fft,
            "rms_mean": rms_mean,
            "rms_std": float(np.std(rms_per_frame)) if n_frames else 0.0,
            "rms_max": float(np.max(rms_per_frame)) if n_frames else 0.0,
            "peak": peak,
            "is_silent": is_silent,
            "clipping_ratio": clipping_ratio,
            "dynamic_range_db": dynamic_range_db,
            "zcr_mean": float(np.mean(zcr_per_frame)) if n_frames else 0.0,
            "zcr_std": float(np.std(zcr_per_frame)) if n_frames else 0.0,
            "centroid_mean": float(np.mean(centroid)) if n_frames else 0.0,
            "centroid_std": float(np.std(centroid)) if n_frames else 0.0,
            "rolloff_mean": float(np.mean(rolloff)) if n_frames else 0.0,
            "flatness_mean": float(np.mean(flatness)) if n_frames else 0.0,
            "bandwidth_mean": float(np.mean(bandwidth)) if n_frames else 0.0,
            "chroma": [round(float(v), 6) for v in chroma],
            "tempo_bpm": tempo_bpm,
            "tempo_strength": tempo_strength,
            "voice_band_ratio": voice_ratio,
            "speech_likeness": speech_likeness,
            "mel_energy_mean": float(np.mean(mel_energy)) if n_frames else 0.0,
            "feature_dim": int(features.shape[0]),
            "frame_indices": frame_indices,
            "format": "unknown",
            "channels": 1,
            "source_sample_rate": int(sr),
            "info_resampled": False,
            "info_assumed_sample_rate": False,
        }
        return features, detail

    def waveform_features(self, samples: Any, sr: int) -> dict:
        """
        Dalga formundan okunabilir akustik istatistikler çıkarır.

        Args:
            samples: mono (veya çok kanallı) dalga formu
            sr: örnekleme hızı

        Returns:
            duration_sec, rms_*, zcr_*, spectral_*, tempo_bpm, is_silent,
            clipping_ratio, dynamic_range_db, speech_likeness, n_frames ...
        """
        if sr is None or int(sr) <= 0:
            raise AudioDecodeError(f"Geçersiz örnekleme hızı: {sr}")
        data = self._to_mono(np.asarray(self._to_float(samples), dtype=np.float32))
        if data.size == 0:
            raise AudioDecodeError("Ses verisi boş — özellik çıkarılamaz")
        _features, detail = self._compute_features(data, int(sr))
        detail.pop("frame_indices", None)
        detail["sample_rate"] = int(sr)
        return detail

    def describe(self, audio: Any) -> dict:
        """
        Sesi ayrıntılı biçimde betimler (log/teşhis amaçlı).

        Args:
            audio: Yol | bytes | np.ndarray | (samples, sample_rate)

        Returns:
            Biçim, süre, RMS, tepe, ZCR, spektral özet, tempo, konuşma benzerliği

        Raises:
            AudioDecodeError: Ses çözülemezse
        """
        samples, sr, info = self._load_with_info(audio)
        features, detail = self._compute_features(samples, sr)
        detail.pop("frame_indices", None)
        features = dict(detail)
        features.update({
            "sample_rate": int(sr),
            "n_samples": int(samples.size),
            "feature_dim": int(NATIVE_FEATURE_DIM),
            "native_feature_dim": NATIVE_FEATURE_DIM,
            "format": info.get("format", "unknown"),
            "channels": int(info.get("channels", 1)),
            "source_sample_rate": int(info.get("source_sample_rate", sr)),
            "resampled": bool(info.get("resampled", False)),
            "assumed_sample_rate": bool(info.get("assumed_sample_rate", False)),
            "is_silent": bool(detail.get("is_silent", False)),
            "target_sample_rate": int(self._target_sr),
        })
        features.pop("chroma", None)
        features["tempo_bpm"] = round(float(features.get("tempo_bpm", 0.0)), 2)
        return features

    # ─── Sinyal İşleme Yardımcıları ─────────────────────────────────────

    def _frame(self, samples: "np.ndarray", frame_length: int,
               hop: int) -> Tuple["np.ndarray", "np.ndarray"]:
        """
        Dalga formunu örtüşen karelere böler (çok uzun seslerde kare seyreltir).

        Args:
            samples: mono dizi
            frame_length: kare uzunluğu (örnek)
            hop: kaydırma (örnek)

        Returns:
            (kareler (n_frames, frame_length), kullanılan kare indeksleri)
        """
        data = samples
        if data.size < frame_length:
            data = np.pad(data, (0, frame_length - data.size), mode="constant")
        n_frames = 1 + (data.size - frame_length) // hop
        indices = np.arange(n_frames)
        if n_frames > MAX_FRAMES:
            stride = int(math.ceil(n_frames / MAX_FRAMES))
            indices = indices[::stride]
            n_frames = indices.size
        offsets = (indices * hop)[:, None] + np.arange(frame_length)[None, :]
        return data[offsets].astype(np.float64), indices

    def _mel_filterbank(self, sr: int, n_fft: int, n_mels: int = N_MELS) -> "np.ndarray":
        """
        26 üçgen mel filtre matrisi üretir (0–8000 Hz, sr/2 ile sınırlı).

        Args:
            sr: örnekleme hızı
            n_fft: FFT boyutu
            n_mels: filtre sayısı

        Returns:
            Şekil (n_mels, n_fft//2+1) matris (her filtre alan-normalize)
        """
        fmax = min(MEL_FMAX, float(sr) / 2.0)
        key = (int(sr), int(n_fft), float(n_mels))
        cached = self._mel_cache.get(key)
        if cached is not None:
            return cached

        def hz_to_mel(frequency: float) -> float:
            """Hz → mel dönüşümü (O'Shaughnessy formülü)."""
            return 2595.0 * math.log10(1.0 + frequency / 700.0)

        def mel_to_hz(mel: float) -> float:
            """Mel → Hz dönüşümü (ters formül)."""
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(fmax), n_mels + 2)
        hz_points = np.asarray([mel_to_hz(m) for m in mel_points])
        bins = np.clip(np.floor((n_fft + 1) * hz_points / float(sr)).astype(int),
                       0, n_fft // 2)

        filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
        for index in range(n_mels):
            left, center, right = int(bins[index]), int(bins[index + 1]), int(bins[index + 2])
            if center <= left:
                center = min(left + 1, n_fft // 2)
            if right <= center:
                right = min(center + 1, n_fft // 2)
            if center > left:
                filters[index, left:center] = \
                    (np.arange(left, center) - left) / float(center - left)
            if right > center:
                filters[index, center:right] = \
                    (right - np.arange(center, right)) / float(right - center)
        filters /= np.maximum(filters.sum(axis=1, keepdims=True), 1e-9)

        self._mel_cache[key] = filters
        return filters

    @staticmethod
    def _pool_time(matrix: "np.ndarray", segments: int) -> "np.ndarray":
        """
        Zaman eksenini `segments` parçaya bölüp ortalamasını alır.

        Args:
            matrix: (n_frames, n_features) matris
            segments: hedef parça sayısı

        Returns:
            (segments, n_features) matris (kısa seslerde kenar tekrarı)
        """
        frames = matrix.shape[0]
        if frames == 0:
            return np.zeros((segments, matrix.shape[1]), dtype=np.float64)
        edges = np.linspace(0, frames, segments + 1).round().astype(int)
        out = np.zeros((segments, matrix.shape[1]), dtype=np.float64)
        for index in range(segments):
            start = min(edges[index], frames - 1)
            end = max(edges[index + 1], start + 1)
            out[index] = matrix[start:end].mean(axis=0)
        return out

    @staticmethod
    def _stats4(values: "np.ndarray") -> "np.ndarray":
        """Bir dizinin (ortalama, std, max, min) istatistiklerini döndürür."""
        data = np.asarray(values, dtype=np.float64).ravel()
        if data.size == 0:
            return np.zeros((4,), dtype=np.float64)
        return np.asarray([data.mean(), data.std(), data.max(), data.min()],
                          dtype=np.float64)

    @staticmethod
    def _chroma(power: "np.ndarray", freqs: "np.ndarray") -> "np.ndarray":
        """
        12 bantlı chroma-benzeri enerji dağılımı (55 Hz – 2 kHz aralığı).

        Args:
            power: (n_frames, n_bins) güç spektrumu
            freqs: bin frekansları

        Returns:
            12 boyutlu, toplamı 1 olan dağılım
        """
        chroma = np.zeros((12,), dtype=np.float64)
        valid = (freqs >= 55.0) & (freqs <= 2000.0)
        if not bool(valid.any()):
            return chroma
        valid_freqs = freqs[valid]
        with np.errstate(divide="ignore", invalid="ignore"):
            midi = 69.0 + 12.0 * np.log2(np.maximum(valid_freqs, 1e-6) / 440.0)
        pitch_classes = np.mod(np.round(midi).astype(int), 12)
        valid_power = power[:, valid]
        for pitch_class in range(12):
            mask = pitch_classes == pitch_class
            if bool(mask.any()):
                chroma[pitch_class] = float(np.sum(valid_power[:, mask]))
        total = float(chroma.sum())
        return chroma / total if total > 0 else chroma

    @staticmethod
    def _voice_band_ratio(power: "np.ndarray", freqs: "np.ndarray") -> float:
        """300–3400 Hz (konuşma bandı) enerjisinin toplam enerjiye oranı."""
        total = float(np.sum(power))
        if total <= 0:
            return 0.0
        band = (freqs >= 300.0) & (freqs <= 3400.0)
        return float(np.sum(power[:, band]) / total) if bool(band.any()) else 0.0

    @staticmethod
    def _speech_likeness(voice_ratio: float, zcr_mean: float,
                         flatness_mean: float) -> float:
        """
        Konuşma benzerliği sezgisi (0..1).

        Bant enerjisi oranı, ZCR aralığı ve spektral düzlük birleştirilir.
        """
        zcr_score = 1.0 - min(1.0, abs(zcr_mean - 0.08) / 0.15)
        flat_score = 1.0 - min(1.0, flatness_mean / 0.5)
        return float(np.clip(0.5 * voice_ratio + 0.3 * zcr_score + 0.2 * flat_score,
                             0.0, 1.0))

    @staticmethod
    def _estimate_tempo(rms_per_frame: "np.ndarray", sr: int,
                        hop: int) -> Tuple[float, float]:
        """
        Onset zarfının otokorelasyonuyla tempo kestirir.

        Args:
            rms_per_frame: kare başına RMS enerjisi
            sr: örnekleme hızı
            hop: kare kaydırması (örnek)

        Returns:
            (tempo_bpm, güç) — sessiz/kısa sinyallerde (0.0, 0.0)
        """
        envelope = np.asarray(rms_per_frame, dtype=np.float64)
        if envelope.size < 8 or float(envelope.std()) < 1e-6:
            return 0.0, 0.0
        onset = np.maximum(0.0, np.diff(envelope))
        onset = onset - onset.mean()
        if float(np.abs(onset).sum()) < 1e-9:
            return 0.0, 0.0

        autocorrelation = np.correlate(onset, onset, mode="full")[onset.size - 1:]
        if autocorrelation.size < 4 or autocorrelation[0] <= 0:
            return 0.0, 0.0
        autocorrelation = autocorrelation / autocorrelation[0]

        frames_per_second = float(sr) / float(hop)
        min_lag = max(1, int(round(frames_per_second * 60.0 / 240.0)))   # 240 BPM
        max_lag = min(autocorrelation.size - 1,
                      int(round(frames_per_second * 60.0 / 40.0)))    # 40 BPM
        if max_lag <= min_lag:
            return 0.0, 0.0

        segment = autocorrelation[min_lag:max_lag + 1]
        best_offset = int(np.argmax(segment))
        best_lag = best_offset + min_lag
        tempo_bpm = 60.0 * frames_per_second / float(max(1, best_lag))
        strength = float(np.clip(segment[best_offset], 0.0, 1.0))
        return float(tempo_bpm), strength

    def _project(self, features: "np.ndarray", out_dim: int) -> List[float]:
        """
        Ham özellik vektörünü sabit tohumlu rastgele projeksiyonla `out_dim`'e taşır.

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
                                       AUDIO_PROJ_SEED, AUDIO_PROJ_SALT)
            self._proj_cache[key] = matrix
        projected = features.astype(np.float64) @ matrix
        return l2_normalize(projected.tolist())

    # ─── Opsiyonel Derin Omurga ─────────────────────────────────────────

    def load_backbone(self, path: str) -> bool:
        """
        Opsiyonel derin ses omurgasını yükler (torch + torchaudio gerekir).

        Ağırlık yüklenmeden `encode_deep()` kullanılmaz.

        Args:
            path: `torch.save` ile kaydedilmiş state_dict dosyası

        Returns:
            True → omurga hazır, False → yüklenemedi
        """
        if not self._torch_available:
            logger.warning("torch/torchaudio yok — derin ses omurgası yüklenemez")
            return False
        try:
            import torch  # ağır import — fonksiyon içinde
            from torchaudio.models import Wav2Vec2Model

            model = Wav2Vec2Model(
                extractor_mode="default",
                extractor_conv_layer_config=None,
                extractor_conv_bias=False,
                encoder_embed_dim=128,
                encoder_projection_dropout=0.0,
                encoder_pos_conv_kernel=128,
                encoder_pos_conv_groups=16,
                encoder_num_layers=2,
                encoder_num_heads=4,
                encoder_attention_dropout=0.0,
                encoder_ff_interm_features=256,
                encoder_ff_interm_dropout=0.0,
                encoder_dropout=0.0,
                encoder_layer_norm_first=False,
                encoder_layer_drop=0.0,
                aux_num_out=None,
            )
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval()
            self._backbone = model
            self._backbone_name = f"wav2vec2:{path}"
            logger.info("Ses omurgası yüklendi: %s", path)
            if self._event_bus:
                self._event_bus.emit("multimodal.audio_backbone_loaded",
                                     {"path": str(path)}, source=self.name)
            return True
        except Exception as exc:
            logger.error("Ses omurgası yüklenemedi (%s): %s", path, exc)
            return False

    def encode_deep(self, audio: Any) -> EncodedChunk:
        """
        Opsiyonel derin ses kodlaması (torch + torchaudio gerekir).

        `load_backbone()` çağrılmadıysa `EncoderUnavailable` yükseltir.

        Args:
            audio: Yol | bytes | np.ndarray | (samples, sample_rate)

        Returns:
            `Modality.AUDIO` tipli `EncodedChunk` (meta["deep"] = True)

        Raises:
            EncoderUnavailable: Omurga yüklü değilse veya torch yoksa
        """
        if self._backbone is None:
            raise EncoderUnavailable(
                "Derin ses omurgası yüklü değil — önce load_backbone(path) çağırın")
        try:
            import torch

            samples, sr = self.load(audio)
            tensor = torch.from_numpy(samples).unsqueeze(0).float()
            with torch.no_grad():
                output = self._backbone(tensor)
            hidden = output[0] if isinstance(output, tuple) else output
            features = hidden.mean(dim=1).squeeze(0).cpu().numpy()
            vector = self._project(np.asarray(features, dtype=np.float64), self._embedding_dim)
            meta: Dict[str, Any] = {
                "deep": True,
                "backbone": self._backbone_name or "wav2vec2",
                "embedding_dim": self._embedding_dim,
                "sample_rate": int(sr),
                "torch_available": True,
            }
            attach_native(meta, features)
            self._stats["deep_encoded"] += 1
            if self._event_bus:
                self._event_bus.emit("multimodal.audio_deep_encoded",
                                     {"dim": len(vector)}, source=self.name)
            return EncodedChunk(modality=Modality.AUDIO, vector=vector,
                                confidence=0.8, meta=meta)
        except EncoderUnavailable:
            raise
        except Exception as exc:
            logger.error("Derin ses kodlama hatası: %s", exc)
            raise EncoderUnavailable(f"Derin ses kodlama başarısız: {exc}") from exc

    # ─── İç Yardımcılar ─────────────────────────────────────────────────

    @staticmethod
    def _is_bare_samples(audio: Any) -> bool:
        """Girdinin örnekleme hızı bilgisi taşımayan çıplak dizi olup olmadığı."""
        if isinstance(audio, (list, tuple)):
            # (samples, sr) demeti değilse çıplak örnek dizisi sayılır
            return not (len(audio) == 2 and isinstance(audio[1], (int, float))
                        and not isinstance(audio[1], bool))
        return type(audio).__module__.split(".")[0] == "numpy" \
            and hasattr(audio, "shape")

    @staticmethod
    def _is_ndarray(payload: Any) -> bool:
        """Nesnenin numpy dizisi olup olmadığını (import etmeden) sezer."""
        return (type(payload).__module__.split(".")[0] == "numpy"
                and hasattr(payload, "shape") and hasattr(payload, "dtype"))

    def _empty_chunk(self, error: str) -> EncodedChunk:
        """Hata durumunda döndürülen boş chunk."""
        return EncodedChunk(
            modality=Modality.AUDIO,
            vector=[0.0] * self._embedding_dim,
            confidence=0.0,
            meta={"error": error, "embedding_dim": self._embedding_dim},
        )

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
    def _read_sample_rate(config: Any) -> int:
        """`config.multimodal.audio_sample_rate` değerini güvenle okur."""
        if config is None:
            return DEFAULT_SAMPLE_RATE
        try:
            rate = int(config.multimodal.audio_sample_rate)
            return rate if rate > 0 else DEFAULT_SAMPLE_RATE
        except Exception:
            return DEFAULT_SAMPLE_RATE

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        JSON-serileştirilebilir istatistikleri döndürür.

        Returns:
            Kodlama sayaçları, yetenek bayrakları ve önbellek durumu
        """
        return {
            "name": self.name,
            "initialized": self._initialized,
            "embedding_dim": self._embedding_dim,
            "target_sample_rate": self._target_sr,
            "native_feature_dim": NATIVE_FEATURE_DIM,
            "n_mels": N_MELS,
            "soundfile_available": self._soundfile_available,
            "scipy_available": self._scipy_available,
            "torch_available": self._torch_available,
            "backbone_loaded": self._backbone is not None,
            "counters": dict(self._stats),
            "format_distribution": dict(self._format_counts),
            "projection_cache_entries": len(self._proj_cache),
            "mel_cache_entries": len(self._mel_cache),
            "proj_seed": AUDIO_PROJ_SEED,
            "proj_salt": AUDIO_PROJ_SALT,
        }


__all__ = ["AudioEncoder", "NATIVE_FEATURE_DIM", "N_MELS",
           "AUDIO_PROJ_SEED", "AUDIO_PROJ_SALT", "SILENCE_RMS"]
