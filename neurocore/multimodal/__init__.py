"""
NeuroCore Çok-Modlu Katman (L4)
================================
Metin, görüntü, ses ve tablo girdilerini ortak bir gömme uzayına taşıyan,
ardından füzyonlayan katman.

Bileşenler:
    ModalityRouter  : ham yük → modalite (tip, sihirli bayt, uzantı, sezgi)
    TextEncoder     : deterministik hash gömlesi (unigram+bigram + konu özellikleri)
    ImageEncoder    : el yapımı görsel özellikler + sabit projeksiyon (torch gerekmez)
    AudioEncoder    : numpy mel spektrogram + spektral istatistikler
    TableEncoder    : tip/dağılım profili + başlık hash'leri
    FusionLayer     : attention | early | late | concat
    MultimodalLayer : motorun sürdüğü katman cephesi

Tüm kodlayıcılar `embedding_dim` (varsayılan 768) uzunluğunda L2-normalize
vektör üretir ve süreçten bağımsız **deterministiktir**.

Kullanım:
    from neurocore.multimodal import MultimodalLayer, MultiModalInput

    layer = MultimodalLayer()
    layer.initialize(config, event_bus)

    chunk = layer.encode("GPU sıcaklığı neden yükselir?")
    fused = layer.encode_input(MultiModalInput(text="grafik", image="ekran.png"))
"""

from __future__ import annotations

from neurocore.multimodal.audio_encoder import AudioEncoder
from neurocore.multimodal.fusion import FusionLayer
from neurocore.multimodal.image_encoder import ImageEncoder
from neurocore.multimodal.layer import MultimodalLayer
from neurocore.multimodal.router import ModalityRouter
from neurocore.multimodal.table_encoder import TableEncoder
from neurocore.multimodal.text_encoder import TextEncoder
from neurocore.multimodal.types import (
    AudioDecodeError,
    EncodedChunk,
    EncoderUnavailable,
    FusedRepresentation,
    ImageDecodeError,
    Modality,
    MultiModalInput,
    MultimodalError,
    TableData,
    TableDecodeError,
    stable_hash_vector,
)

__all__ = [
    "MultimodalLayer",
    "Modality",
    "ModalityRouter",
    "EncodedChunk",
    "MultiModalInput",
    "FusedRepresentation",
    "TextEncoder",
    "ImageEncoder",
    "AudioEncoder",
    "TableEncoder",
    "FusionLayer",
    "TableData",
    "stable_hash_vector",
    "MultimodalError",
    "ImageDecodeError",
    "AudioDecodeError",
    "TableDecodeError",
    "EncoderUnavailable",
]
