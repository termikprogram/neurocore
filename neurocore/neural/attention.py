"""
NeuroCore Dikkat Mekanizmaları
===============================
Çok başlı öz-dikkat, RoPE konum kodlaması ve KV önbelleği.

Tasarım kararları:
    * **PyTorch zorunlu değildir.** `torch` kuruluysa gerçek tensör
      uygulaması kullanılır; değilse aynı arayüz NumPy ile çalışır
      (yalnızca çıkarım; gradyan yok). Böylece sistem torch olmadan da
      ayağa kalkar.
    * **RoPE (Rotary Position Embedding)** — öğrenilmiş konum gömlemeleri
      yerine döndürmeli konum kodlaması. Uzunluk genellemesi daha iyidir ve
      bağlam penceresi büyütüldüğünde yeniden eğitim gerekmez.
    * **KV önbelleği** — otoregresif üretimde her adımda tüm bağlamı yeniden
      hesaplamamak için anahtar/değer tensörleri saklanır; üretim O(n²) → O(n).
    * **GQA/MQA desteği** (`num_kv_heads`) — bellek tasarrufu; küçük
      donanımlarda KV önbelleğini `num_heads / num_kv_heads` oranında küçültür.
    * **Kaydırıcı pencere** (`sliding_window`) — uzun bağlamda bellek sınırı.

Backend soyutlaması: `HAS_TORCH` bayrağı. Her sınıf `torch` varsa
`nn.Module` olarak, yoksa düz Python sınıfı olarak davranır.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.attention")


# ─── Backend Tespiti ────────────────────────────────────────────────────────

try:  # pragma: no cover - ortama bağlı
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    HAS_TORCH = False

try:  # pragma: no cover
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False


# ─── Yapılandırma ───────────────────────────────────────────────────────────

@dataclass
class AttentionConfig:
    """Dikkat katmanı yapılandırması."""
    embed_dim: int = 768
    num_heads: int = 12
    num_kv_heads: Optional[int] = None     # None = num_heads (MHA); < ise GQA/MQA
    dropout: float = 0.1
    bias: bool = False
    rope_base: float = 10000.0
    max_seq_len: int = 2048
    sliding_window: Optional[int] = None
    scale_qk: bool = True
    use_flash: bool = True                 # torch varsa SDPA/Flash dene
    qk_norm: bool = False                  # RMSNorm ile Q/K normalizasyonu

    def __post_init__(self) -> None:
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        self.num_kv_heads = max(1, min(int(self.num_kv_heads), int(self.num_heads)))
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) num_heads ({self.num_heads}) ile bölünmelidir"
            )

    @property
    def head_dim(self) -> int:
        """Baş başına boyut."""
        return self.embed_dim // self.num_heads

    @property
    def kv_repeat(self) -> int:
        """KV başlarının kaç kez tekrarlanacağı (GQA için)."""
        return self.num_heads // max(1, int(self.num_kv_heads or self.num_heads))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embed_dim": self.embed_dim, "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads, "head_dim": self.head_dim,
            "dropout": self.dropout, "rope_base": self.rope_base,
            "max_seq_len": self.max_seq_len, "sliding_window": self.sliding_window,
            "qk_norm": self.qk_norm, "kv_repeat": self.kv_repeat,
        }


# ─── KV Önbelleği ───────────────────────────────────────────────────────────

@dataclass
class KVCache:
    """
    Otoregresif üretim için anahtar/değer önbelleği.

    Tensör backend'i esnektir: torch tensörleri veya NumPy dizileri.
    `max_len` aşılırsa en eski girdiler düşürülür (kayan pencere).
    """

    max_len: Optional[int] = None
    keys: List[Any] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    _length: int = 0

    @property
    def length(self) -> int:
        """Önbellekteki token sayısı."""
        return self._length

    @property
    def is_empty(self) -> bool:
        """Önbellek boş mu?"""
        return self._length == 0

    def append(self, key: Any, value: Any) -> None:
        """Yeni anahtar/değer ekler."""
        self.keys.append(key)
        self.values.append(value)
        self._length += int(key.shape[-2]) if hasattr(key, "shape") else 1
        self._trim()

    def _trim(self) -> None:
        """Pencere sınırını aşan girdileri atar."""
        if self.max_len is None:
            return
        while self._length > self.max_len and len(self.keys) > 1:
            dropped = self.keys.pop(0)
            self.values.pop(0)
            self._length -= int(dropped.shape[-2]) if hasattr(dropped, "shape") else 1

    def clear(self) -> None:
        """Önbelleği boşaltır."""
        self.keys.clear()
        self.values.clear()
        self._length = 0

    def stats(self) -> Dict[str, Any]:
        """Önbellek istatistikleri."""
        return {"length": self._length, "chunks": len(self.keys),
                "max_len": self.max_len}


# ─── RoPE Yardımcıları ──────────────────────────────────────────────────────

def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                     device: Any = None, dtype: Any = None) -> Tuple[Any, Any]:
    """
    RoPE için kosinüs/sinüs tablolarını üretir.

    Args:
        seq_len: Maksimum dizi uzunluğu
        head_dim: Baş boyutu (çift olmalı)
        base: Teta tabanı (10 000 standart; uzun bağlamda büyütülebilir)
        device: torch cihazı (torch varsa)
        dtype: Veri tipi

    Returns:
        (cos, sin) — her biri (seq_len, head_dim) biçiminde
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE için head_dim çift olmalı, alınan: {head_dim}")

    if HAS_TORCH:
        half = head_dim // 2
        frequencies = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, frequencies)          # (seq_len, half)
        angles = torch.cat([angles, angles], dim=-1)          # (seq_len, head_dim)
        return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    if HAS_NUMPY:
        half = head_dim // 2
        frequencies = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half))
        positions = np.arange(seq_len, dtype=np.float32)
        angles = np.outer(positions, frequencies)
        angles = np.concatenate([angles, angles], axis=-1)
        return np.cos(angles), np.sin(angles)

    raise RuntimeError("RoPE için torch veya numpy gerekir")


def _rotate_half(x: Any) -> Any:
    """RoPE yarım döndürme işlemi: [-x2, x1]."""
    half = x.shape[-1] // 2
    if HAS_TORCH and isinstance(x, torch.Tensor):
        first, second = x[..., :half], x[..., half:]
        return torch.cat([-second, first], dim=-1)
    if HAS_NUMPY:
        first, second = x[..., :half], x[..., half:]
        return np.concatenate([-second, first], axis=-1)
    raise RuntimeError("Desteklenmeyen tensör tipi")


def apply_rope(x: Any, cos: Any, sin: Any) -> Any:
    """
    RoPE döndürmesini uygular.

    Args:
        x: (..., seq_len, head_dim) biçiminde sorgu veya anahtar
        cos: (seq_len, head_dim) kosinüs tablosu
        sin: (seq_len, head_dim) sinüs tablosu

    Returns:
        Döndürülmüş tensör (x ile aynı biçim)
    """
    seq_len = x.shape[-2]
    cos_part = cos[:seq_len]
    sin_part = sin[:seq_len]
    # (seq_len, head_dim) → yayınlanabilir biçim
    while cos_part.ndim < x.ndim:
        cos_part = cos_part.unsqueeze(0) if HAS_TORCH and isinstance(x, torch.Tensor) \
            else cos_part[None, ...]
        sin_part = sin_part.unsqueeze(0) if HAS_TORCH and isinstance(x, torch.Tensor) \
            else sin_part[None, ...]
    return (x * cos_part) + (_rotate_half(x) * sin_part)


# ─── RMSNorm ────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class RMSNorm(nn.Module):  # type: ignore[misc]
        """
        Kök ortalama kare normalizasyonu (LayerNorm'un hafif ve hızlı alternatifi).

        Ortalamayı çıkarmaz; bu, küçük modellerde daha kararlı eğitim sağlar
        ve daha az işlem gerektirir.
        """

        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.dim = int(dim)
            self.eps = float(eps)
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Normalize edip ölçekler."""
            dtype = x.dtype
            x32 = x.float()
            variance = x32.pow(2).mean(-1, keepdim=True)
            normalized = x32 * torch.rsqrt(variance + self.eps)
            return (self.weight * normalized.to(dtype))

        def extra_repr(self) -> str:
            return f"dim={self.dim}, eps={self.eps}"
else:
    class RMSNorm:  # type: ignore[no-redef]
        """RMSNorm (NumPy yedeği — yalnızca çıkarım)."""

        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            self.dim = int(dim)
            self.eps = float(eps)
            self.weight = [1.0] * int(dim)

        def __call__(self, x: Any) -> Any:
            """NumPy girdisi üzerinde normalize eder."""
            if not HAS_NUMPY:
                return x
            variance = (x.astype(np.float32) ** 2).mean(-1, keepdims=True)
            normalized = x / np.sqrt(variance + self.eps)
            return normalized * np.asarray(self.weight, dtype=normalized.dtype)

        def forward(self, x: Any) -> Any:
            """`__call__` takma adı."""
            return self(x)


# ─── SwiGLU İleri Besleme ───────────────────────────────────────────────────

if HAS_TORCH:
    class SwiGLU(nn.Module):  # type: ignore[misc]
        """
        SwiGLU ileri besleme ağı: `W2(silu(W1 x) * W3 x)`.

        Llama/PaLM ailesinde standarttır; aynı parametre bütçesinde GELU'lu
        MLP'den belirgin biçimde daha iyi sonuç verir. Gizli boyut genelde
        `4 * d_model` yerine `8/3 * d_model` seçilir (parametre eşitlemesi).
        """

        def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                     dropout: float = 0.0, bias: bool = False) -> None:
            super().__init__()
            self.dim = int(dim)
            self.hidden_dim = int(hidden_dim or int(8 * dim / 3))
            # 64'ün katına yuvarla (donanım verimliliği)
            self.hidden_dim = max(64, ((self.hidden_dim + 63) // 64) * 64)
            self.w1 = nn.Linear(dim, self.hidden_dim, bias=bias)   # kapı
            self.w3 = nn.Linear(dim, self.hidden_dim, bias=bias)   # değer
            self.w2 = nn.Linear(self.hidden_dim, dim, bias=bias)   # çıkış
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """SwiGLU ileri geçişi."""
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

        def extra_repr(self) -> str:
            return f"dim={self.dim}, hidden={self.hidden_dim}"
else:
    class SwiGLU:  # type: ignore[no-redef]
        """SwiGLU (NumPy yedeği, rastgele başlatılmış ağırlıklarla)."""

        def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                     dropout: float = 0.0, bias: bool = False) -> None:
            self.dim = int(dim)
            self.hidden_dim = max(64, int(hidden_dim or int(8 * dim / 3)))
            self._initialized = False

        def _ensure(self) -> None:
            """Ağırlıkları tembel başlatır."""
            if self._initialized or not HAS_NUMPY:
                return
            rng = np.random.default_rng(1234)
            scale = 1.0 / math.sqrt(self.dim)
            self.w1 = rng.normal(0, scale, (self.dim, self.hidden_dim)).astype(np.float32)
            self.w3 = rng.normal(0, scale, (self.dim, self.hidden_dim)).astype(np.float32)
            self.w2 = rng.normal(0, scale, (self.hidden_dim, self.dim)).astype(np.float32)
            self._initialized = True

        def forward(self, x: Any) -> Any:
            """NumPy SwiGLU."""
            self._ensure()
            if not HAS_NUMPY or not self._initialized:
                return x
            gate = x @ self.w1
            silu = gate / (1.0 + np.exp(-gate))
            return (silu * (x @ self.w3)) @ self.w2

        def __call__(self, x: Any) -> Any:
            """`forward` takma adı."""
            return self.forward(x)


# ─── Çok Başlı Dikkat ───────────────────────────────────────────────────────

if HAS_TORCH:
    class MultiHeadAttention(nn.Module):  # type: ignore[misc]
        """
        Çok başlı öz-dikkat; RoPE + KV önbelleği + GQA desteğiyle.

        Kullanım:
            attn = MultiHeadAttention(AttentionConfig(embed_dim=768, num_heads=12))
            out, cache = attn(x, cache=KVCache(), use_cache=True)
            out, cache = attn(next_token, cache=cache, use_cache=True)   # üretim

        Args:
            config: AttentionConfig
        """

        def __init__(self, config: AttentionConfig) -> None:
            super().__init__()
            self.config = config
            self.embed_dim = config.embed_dim
            self.num_heads = config.num_heads
            self.num_kv_heads = int(config.num_kv_heads or config.num_heads)
            self.head_dim = config.head_dim
            self.kv_repeat = config.kv_repeat
            self.dropout_rate = config.dropout

            self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
            self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
            self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
            self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=config.bias)
            self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

            self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else None
            self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else None

            self._rope_cache: Optional[Tuple[Any, Any]] = None
            self._rope_len = 0
            self.register_buffer("_causal_mask", torch.empty(0), persistent=False)

        def _get_rope(self, seq_len: int, device: Any, dtype: Any) -> Tuple[Any, Any]:
            """RoPE tablosunu (gerekiyorsa büyüterek) döndürür."""
            needed = max(seq_len, self.config.max_seq_len)
            if self._rope_cache is None or self._rope_len < needed or \
                    self._rope_cache[0].device != device:
                cos, sin = build_rope_cache(needed, self.head_dim,
                                            self.config.rope_base, device, torch.float32)
                self._rope_cache = (cos, sin)
                self._rope_len = needed
            return self._rope_cache  # type: ignore[return-value]

        def forward(
            self,
            x: "torch.Tensor",
            attention_mask: Optional["torch.Tensor"] = None,
            cache: Optional[KVCache] = None,
            use_cache: bool = False,
        ) -> Tuple["torch.Tensor", Optional[KVCache]]:
            """
            İleri geçiş.

            Args:
                x: (batch, seq_len, embed_dim)
                attention_mask: (batch, seq_len) — 1 geçerli, 0 dolgu
                cache: KVCache (üretimde artımlı)
                use_cache: Önbelleği güncelle ve döndür

            Returns:
                (çıktı, güncellenmiş önbellek veya None)
            """
            batch, seq_len, _ = x.shape
            past_len = cache.length if (cache is not None and use_cache) else 0

            query = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            if self.q_norm is not None:
                query = self.q_norm(query)
            if self.k_norm is not None:
                key = self.k_norm(key)

            # RoPE — yalnızca yeni token'lara uygulanır, konum `past_len` kadar kaydırılır
            cos, sin = self._get_rope(past_len + seq_len, x.device, x.dtype)
            cos_slice = cos[past_len:past_len + seq_len]
            sin_slice = sin[past_len:past_len + seq_len]
            query = apply_rope(query, cos_slice, sin_slice)
            key = apply_rope(key, cos_slice, sin_slice)

            # KV önbelleği
            if cache is not None and use_cache:
                cache.append(key, value)
                if cache.keys:
                    key = torch.cat(cache.keys, dim=2)
                    value = torch.cat(cache.values, dim=2)

            # GQA: KV başlarını sorgu baş sayısına genişlet
            if self.kv_repeat > 1:
                key = key.repeat_interleave(self.kv_repeat, dim=1)
                value = value.repeat_interleave(self.kv_repeat, dim=1)

            total_len = key.shape[2]
            scale = 1.0 / math.sqrt(self.head_dim) if self.config.scale_qk else 1.0

            # Sayısal güvenlik: `-inf` ile tamamen maskelenmiş bir sorgu satırı
            # softmax'ta 0/0 üretip NaN yayar. Bunun yerine çok büyük ama
            # **sonlu** bir negatif değer kullanılır (-1e4 ≈ exp(-1e4) = 0).
            # Bu, dolgu token'larının gizli durumlarını NaN yapmaz.
            masked_fill_value = -1.0e4

            attn_mask: Optional["torch.Tensor"] = None
            if total_len > 1:
                # Nedensel maske + kayan pencere; kayan pencerede sorgunun
                # kendi konumu her zaman görünür kalır (değilse satır tamamen
                # maskelenir).
                offset = total_len - seq_len
                rows = torch.arange(seq_len, device=x.device).unsqueeze(1) + offset
                cols = torch.arange(total_len, device=x.device).unsqueeze(0)
                allowed = cols <= rows
                if self.config.sliding_window:
                    allowed = allowed & ((cols > rows - int(self.config.sliding_window))
                                         | (cols == rows))
                attn_mask = torch.zeros(seq_len, total_len, device=x.device, dtype=x.dtype)
                attn_mask.masked_fill_(~allowed, masked_fill_value)
                if attention_mask is not None and past_len == 0:
                    # Dolgu token'ları anahtar olarak maskelenir
                    pad = attention_mask[:, None, None, :].to(dtype=x.dtype)
                    attn_mask = attn_mask[None, None, :, :] + (1.0 - pad) * masked_fill_value
                    attn_mask = attn_mask.expand(batch, self.num_heads, seq_len, total_len)

            output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                scale=scale,
                is_causal=(attn_mask is None and seq_len > 1),
            )
            output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
            return self.resid_dropout(self.out_proj(output)), (cache if use_cache else None)

        def extra_repr(self) -> str:
            return (
                f"embed_dim={self.embed_dim}, heads={self.num_heads}, "
                f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
                f"rope_base={self.config.rope_base}"
            )
else:
    class MultiHeadAttention:  # type: ignore[no-redef]
        """
        Çok başlı dikkat — NumPy yedeği (yalnızca çıkarım, rastgele ağırlıklar).

        Torch kurulu olmadığında sistemin çalışmaya devam etmesini sağlar.
        Ağırlıklar deterministik tohumla üretilir; gerçek eğitim için torch
        gerekir (`attention.HAS_TORCH` False iken trainer devre dışı kalır).
        """

        def __init__(self, config: AttentionConfig) -> None:
            self.config = config
            self.embed_dim = config.embed_dim
            self.num_heads = config.num_heads
            self.head_dim = config.head_dim
            self._initialized = False

        def _ensure(self) -> None:
            """Ağırlıkları tembel başlatır."""
            if self._initialized or not HAS_NUMPY:
                return
            rng = np.random.default_rng(4321)
            scale = 1.0 / math.sqrt(self.embed_dim)
            self.wq = rng.normal(0, scale, (self.embed_dim, self.embed_dim)).astype(np.float32)
            self.wk = rng.normal(0, scale, (self.embed_dim, self.embed_dim)).astype(np.float32)
            self.wv = rng.normal(0, scale, (self.embed_dim, self.embed_dim)).astype(np.float32)
            self.wo = rng.normal(0, scale, (self.embed_dim, self.embed_dim)).astype(np.float32)
            self._initialized = True

        def forward(self, x: Any, attention_mask: Any = None,
                    cache: Optional[KVCache] = None, use_cache: bool = False
                    ) -> Tuple[Any, Optional[KVCache]]:
            """NumPy öz-dikkat (nedensel)."""
            self._ensure()
            if not HAS_NUMPY or not self._initialized:
                return x, (cache if use_cache else None)

            single = x.ndim == 2
            if single:
                x = x[None, ...]
            batch, seq_len, _ = x.shape
            past_len = cache.length if (cache is not None and use_cache) else 0

            query = (x @ self.wq).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            key = (x @ self.wk).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            value = (x @ self.wv).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

            cos, sin = build_rope_cache(past_len + seq_len, self.head_dim, self.config.rope_base)
            cos_s, sin_s = cos[past_len:past_len + seq_len], sin[past_len:past_len + seq_len]
            query = apply_rope(query, cos_s, sin_s)
            key = apply_rope(key, cos_s, sin_s)

            if cache is not None and use_cache:
                cache.append(key, value)
                key = np.concatenate(cache.keys, axis=2)
                value = np.concatenate(cache.values, axis=2)

            total = key.shape[2]
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (query @ key.transpose(0, 1, 3, 2)) * scale

            offset = total - seq_len
            rows = np.arange(seq_len)[:, None] + offset
            cols = np.arange(total)[None, :]
            allowed = cols <= rows
            if self.config.sliding_window:
                allowed = allowed & ((cols > rows - int(self.config.sliding_window))
                                     | (cols == rows))
            # -1e9 yerine sonlu büyük negatif: tamamen maskelenmiş satır NaN vermez
            scores = np.where(allowed[None, None, :, :], scores, -1.0e9)

            scores = scores - scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-9)
            output = (weights @ value).transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)
            output = output @ self.wo
            return (output[0] if single else output), (cache if use_cache else None)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """`forward` takma adı."""
            return self.forward(*args, **kwargs)


# ─── Çapraz Dikkat (Modalite Füzyonu) ───────────────────────────────────────

if HAS_TORCH:
    class CrossAttention(nn.Module):  # type: ignore[misc]
        """
        Çapraz dikkat: bir modalitenin sorguları diğerinin anahtar/değerleriyle.

        Çok-modlu füzyonda kullanılır: metin sorguları görüntü/ses bağlamına
        "bakar". Çok-modlu katman torch olmadan da çalıştığı için bu sınıf
        yalnızca torch varken tanımlanır ve kullanılır.
        """

        def __init__(self, query_dim: int, context_dim: Optional[int] = None,
                     num_heads: int = 8, dropout: float = 0.0) -> None:
            super().__init__()
            context_dim = int(context_dim or query_dim)
            self.num_heads = int(num_heads)
            self.head_dim = int(query_dim) // self.num_heads
            if self.head_dim * self.num_heads != int(query_dim):
                raise ValueError("query_dim num_heads ile bölünmelidir")
            self.q_proj = nn.Linear(query_dim, query_dim)
            self.k_proj = nn.Linear(context_dim, query_dim)
            self.v_proj = nn.Linear(context_dim, query_dim)
            self.out_proj = nn.Linear(query_dim, query_dim)
            # Not: `nn.Dropout(0)` yerine Identity kullanıldığı için dropout
            # oranı ayrı bir skalarda tutulur (`Identity.p` yoktur).
            self._dropout_p = float(dropout)
            self.dropout = nn.Dropout(self._dropout_p) if self._dropout_p > 0 else nn.Identity()

        def forward(self, query: "torch.Tensor", context: "torch.Tensor",
                    context_mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
            """
            Args:
                query: (batch, q_len, query_dim)
                context: (batch, c_len, context_dim)
                context_mask: (batch, c_len) — 1 geçerli, 0 dolgu

            Returns:
                (batch, q_len, query_dim)
            """
            batch, q_len, _ = query.shape
            c_len = context.shape[1]

            q = self.q_proj(query).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(context).view(batch, c_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(context).view(batch, c_len, self.num_heads, self.head_dim).transpose(1, 2)

            mask = None
            if context_mask is not None:
                # Sonlu maske değeri (bkz. MultiHeadAttention'daki not):
                # tamamen maskelenmiş satırlarda NaN oluşmasını engeller.
                mask = context_mask[:, None, None, :].to(dtype=query.dtype)
                mask = (1.0 - mask) * -1.0e4

            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self._dropout_p if self.training else 0.0,
            )
            output = output.transpose(1, 2).contiguous().view(batch, q_len, -1)
            return self.out_proj(output)

        def extra_repr(self) -> str:
            return f"heads={self.num_heads}, head_dim={self.head_dim}"
else:
    CrossAttention = None  # type: ignore[assignment,misc]


# ─── Yardımcılar ────────────────────────────────────────────────────────────

def estimate_attention_memory(config: AttentionConfig, batch_size: int,
                              seq_len: int, bytes_per_param: int = 4) -> Dict[str, float]:
    """
    Dikkat katmanının bellek kullanımını tahmin eder (MB).

    Kaynak tahsis kararlarında kullanılır: KV önbelleği ve aktivasyon
    tensörleri ayrı ayrı raporlanır.
    """
    head_dim = config.head_dim
    kv_heads = int(config.num_kv_heads or config.num_heads)

    params = (
        config.embed_dim * config.num_heads * head_dim   # q
        + 2 * config.embed_dim * kv_heads * head_dim     # k, v
        + config.num_heads * head_dim * config.embed_dim  # out
    )
    weights_mb = params * bytes_per_param / (1024 ** 2)

    kv_cache_mb = (
        2 * batch_size * kv_heads * seq_len * head_dim * bytes_per_param / (1024 ** 2)
    )
    # Dikkat skorları: batch × heads × seq × seq
    scores_mb = batch_size * config.num_heads * seq_len * seq_len * bytes_per_param / (1024 ** 2)
    if config.sliding_window:
        window = min(seq_len, int(config.sliding_window))
        scores_mb *= window / max(1, seq_len)

    return {
        "weights_mb": round(weights_mb, 3),
        "kv_cache_mb": round(kv_cache_mb, 3),
        "attention_scores_mb": round(scores_mb, 3),
        "total_mb": round(weights_mb + kv_cache_mb + scores_mb, 3),
        "estimated_with_flash": round(weights_mb + kv_cache_mb, 3),
    }


def recommended_heads(embed_dim: int, vram_mb: float = 0.0) -> int:
    """
    Boyut ve bellek bütçesine göre önerilen baş sayısı.

    Baş boyutu 64 civarında tutulur (donanım verimliliği); VRAM azsa baş
    sayısı düşürülerek dikkat skorları matrisi küçültülür.
    """
    base = max(2, int(embed_dim // 64))
    if vram_mb and vram_mb < 2048:
        return max(2, base // 2)
    return base


__all__ = [
    "HAS_TORCH",
    "HAS_NUMPY",
    "AttentionConfig",
    "KVCache",
    "MultiHeadAttention",
    "CrossAttention",
    "RMSNorm",
    "SwiGLU",
    "build_rope_cache",
    "apply_rope",
    "estimate_attention_memory",
    "recommended_heads",
]
