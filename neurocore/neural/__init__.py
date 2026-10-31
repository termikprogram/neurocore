"""
NeuroCore Nöral Katman (L3)
============================
Adaptif sinir ağı: tokenizer, dikkat, transformer omurgası, uzman karışımı,
bellek bankası, gömme motoru, büyüme yöneticisi ve eğitim hattı.

Alt modüller:
    tokenizer    — Byte-level BPE (sıfırdan, bağımlılıksız)
    attention    — MHA + RoPE + KV önbellek (GQA, kayan pencere)
    transformer  — Blok, SwiGLU, RMSNorm, omurga montajı, üretim
    moe          — Seyrek uzman karışımı, nötr büyüme, yönlendirme
    memory       — İlişkisel bellek bankası (fayda tabanlı tahliye)
    embedding    — Çok stratejili gömme (öğrenilmiş / yerel / hash)
    growth       — Büyüme kararları (uzman, katman, nicelik)
    trainer      — Donanım-farkında eğitim döngüsü
    model        — Kalıcı model varlığı (kaydet/yükle/nicele)
    engine       — Katman cephesi (dış API)
"""

from neurocore.neural.tokenizer import (
    BPETokenizer,
    DEFAULT_SPECIAL_TOKENS,
    ROLE_TOKENS,
    SimpleTokenizer,
    TokenizerError,
    create_tokenizer,
)
from neurocore.neural.attention import (
    HAS_NUMPY,
    HAS_TORCH,
    AttentionConfig,
    CrossAttention,
    KVCache,
    MultiHeadAttention,
    RMSNorm,
    SwiGLU,
    apply_rope,
    build_rope_cache,
    estimate_attention_memory,
    recommended_heads,
)
from neurocore.neural.transformer import (
    TransformerBackbone,
    TransformerBlock,
    TransformerConfig,
    config_from_system,
    estimate_backbone_memory,
)
from neurocore.neural.moe import (
    GrowthSignal,
    MixtureOfExperts,
    MoEConfig,
    RouterStats,
    decide_growth,
    estimate_moe_memory,
)
from neurocore.neural.memory import (
    MemoryBank,
    MemoryRecord,
    RecallResult,
    WriteDecision,
)
from neurocore.neural.embedding import (
    STRATEGY_HASH,
    STRATEGY_LEARNED,
    STRATEGY_LOCAL,
    EmbeddingCache,
    EmbeddingEngine,
    EmbeddingError,
    ProjectionMatrix,
    cosine_similarity,
    hash_embed,
    l2_normalize,
    stem_turkish,
    tokenize,
)
from neurocore.neural.growth import (
    ACTION_COST,
    GrowthAction,
    GrowthController,
    GrowthDecision,
    GrowthMetrics,
)
from neurocore.neural.trainer import (
    Trainer,
    TrainingDataset,
    TrainingMetrics,
    TrainingPlan,
    TrainingResult,
    TrainingSample,
)
from neurocore.neural.model import (
    ModelGrowthState,
    NeuroCoreModel,
)
from neurocore.neural.engine import NeuralLayer, NeuralLayerError

__all__ = [
    # Cephe
    "NeuralLayer",
    "NeuralLayerError",
    # Model
    "NeuroCoreModel",
    "ModelGrowthState",
    # Tokenizer
    "BPETokenizer",
    "SimpleTokenizer",
    "TokenizerError",
    "create_tokenizer",
    "DEFAULT_SPECIAL_TOKENS",
    "ROLE_TOKENS",
    # Dikkat / omurga
    "AttentionConfig",
    "MultiHeadAttention",
    "CrossAttention",
    "KVCache",
    "RMSNorm",
    "SwiGLU",
    "build_rope_cache",
    "apply_rope",
    "estimate_attention_memory",
    "recommended_heads",
    "TransformerConfig",
    "TransformerBlock",
    "TransformerBackbone",
    "config_from_system",
    "estimate_backbone_memory",
    # MoE
    "MoEConfig",
    "MixtureOfExperts",
    "RouterStats",
    "GrowthSignal",
    "decide_growth",
    "estimate_moe_memory",
    # Bellek
    "MemoryBank",
    "MemoryRecord",
    "RecallResult",
    "WriteDecision",
    # Gömme
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
    # Büyüme
    "GrowthController",
    "GrowthAction",
    "GrowthDecision",
    "GrowthMetrics",
    "ACTION_COST",
    # Eğitim
    "Trainer",
    "TrainingPlan",
    "TrainingMetrics",
    "TrainingResult",
    "TrainingSample",
    "TrainingDataset",
    # Ortam
    "HAS_TORCH",
    "HAS_NUMPY",
]
