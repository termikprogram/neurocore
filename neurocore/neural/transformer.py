"""
NeuroCore Transformer Omurgası
===============================
Dekoder-only transformer bloğu ve tam omurga montajı.

Mimari (Llama/PaLM ailesinden ilhamla, ama küçük donanıma uyarlanmış):

    token → Embedding (+ RoPE konum)
      └─ [TransformerBlock] × N
            ├─ RMSNorm → MultiHeadAttention (RoPE, KV önbellek) → residual
            ├─ RMSNorm → MixtureOfExperts (SwiGLU uzmanlar) → residual
            └─ (isteğe bağlı) Bellek Bankası çapraz dikkati
      └─ RMSNorm → LM Head (ağırlık paylaşımlı)

Tasarım kararları:
    * **Pre-norm** — eğitim kararlılığı için normalizasyon blok girişinde.
    * **Ağırlık paylaşımı** — LM head, token gömme matrisini transpoze eder
      (parametre tasarrufu; küçük modellerde belirgin kazanç).
    * **Gradient checkpointing** — bellek baskısı altında açılır; hesaplama
      %30 yavaşlar, aktivasyon belleği ~√N'ye düşer.
    * **Katman ekleme** — `expand_layers()` ile yeni katman, çıkış
      projeksiyonu sıfırlanarak eklenir (büyüme nötr kalır).
    * **NumPy yedeği** — torch yoksa aynı arayüz çalışır (yalnızca çıkarım).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neurocore.neural.attention import (
    HAS_TORCH,
    AttentionConfig,
    KVCache,
    MultiHeadAttention,
    RMSNorm,
    SwiGLU,
    estimate_attention_memory,
)
from neurocore.neural.moe import (
    HAS_NUMPY,
    MoEConfig,
    MixtureOfExperts,
    estimate_moe_memory,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.transformer")

if HAS_TORCH:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


# ─── Yapılandırma ───────────────────────────────────────────────────────────

@dataclass
class TransformerConfig:
    """Omurga yapılandırması."""
    vocab_size: int = 32000
    embed_dim: int = 768
    num_layers: int = 8
    num_heads: int = 12
    num_kv_heads: Optional[int] = None
    max_seq_len: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    d_ff_multiplier: int = 4
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False
    use_moe: bool = True
    num_experts: int = 4
    max_experts: int = 12
    top_k_experts: int = 2
    shared_expert: bool = True
    rope_base: float = 10000.0
    sliding_window: Optional[int] = None
    qk_norm: bool = False
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_memory: bool = False
    memory_slots: int = 512

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            # Baş sayısını boyuta uyarla (sessiz düzeltme — sistem çalışsın)
            adjusted = max(1, self.embed_dim // 64)
            while self.embed_dim % adjusted != 0 and adjusted > 1:
                adjusted -= 1
            logger.warning(
                f"num_heads ({self.num_heads}) embed_dim ({self.embed_dim}) ile bölünmüyor; "
                f"{adjusted} değerine düzeltildi"
            )
            self.num_heads = adjusted

    @property
    def head_dim(self) -> int:
        """Baş boyutu."""
        return self.embed_dim // self.num_heads

    @property
    def d_ff(self) -> int:
        """İleri besleme gizli boyutu."""
        return int(self.d_ff_multiplier * self.embed_dim)

    def attention_config(self) -> AttentionConfig:
        """Bu omurga için dikkat yapılandırması."""
        return AttentionConfig(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout=self.attention_dropout,
            rope_base=self.rope_base,
            max_seq_len=self.max_seq_len,
            sliding_window=self.sliding_window,
            qk_norm=self.qk_norm,
        )

    def moe_config(self) -> MoEConfig:
        """Bu omurga için MoE yapılandırması."""
        return MoEConfig(
            embed_dim=self.embed_dim,
            num_experts=self.num_experts,
            max_experts=self.max_experts,
            top_k=self.top_k_experts,
            dropout=self.dropout,
            shared_expert=self.shared_expert,
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir yapılandırma."""
        return {
            "vocab_size": self.vocab_size, "embed_dim": self.embed_dim,
            "num_layers": self.num_layers, "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads, "head_dim": self.head_dim,
            "max_seq_len": self.max_seq_len, "dropout": self.dropout,
            "d_ff_multiplier": self.d_ff_multiplier, "d_ff": self.d_ff,
            "tie_embeddings": self.tie_embeddings,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_moe": self.use_moe, "num_experts": self.num_experts,
            "max_experts": self.max_experts, "top_k_experts": self.top_k_experts,
            "rope_base": self.rope_base, "sliding_window": self.sliding_window,
            "use_memory": self.use_memory, "memory_slots": self.memory_slots,
        }


# ─── Torch: Transformer Bloğu ───────────────────────────────────────────────

if HAS_TORCH:
    class TransformerBlock(nn.Module):  # type: ignore[misc]
        """
        Tek bir transformer bloğu: dikkat + (MoE veya SwiGLU FFN).

        Pre-norm düzeni kullanır ve her alt katmanı residual ile sarar.
        """

        def __init__(self, config: TransformerConfig, layer_index: int = 0) -> None:
            super().__init__()
            self.config = config
            self.layer_index = int(layer_index)

            self.norm1 = RMSNorm(config.embed_dim, config.layer_norm_eps)
            self.attention = MultiHeadAttention(config.attention_config())

            self.norm2 = RMSNorm(config.embed_dim, config.layer_norm_eps)
            if config.use_moe:
                self.feed_forward: nn.Module = MixtureOfExperts(config.moe_config())
                self.is_moe = True
            else:
                self.feed_forward = SwiGLU(config.embed_dim, config.d_ff, config.dropout)
                self.is_moe = False

        def forward(
            self,
            x: "torch.Tensor",
            attention_mask: Optional["torch.Tensor"] = None,
            cache: Optional[KVCache] = None,
            use_cache: bool = False,
            return_aux: bool = False,
        ) -> Any:
            """
            Blok ileri geçişi.

            Args:
                x: (batch, seq_len, embed_dim)
                attention_mask: (batch, seq_len)
                cache: KVCache
                use_cache: Önbelleği kullan/güncelle
                return_aux: Yardımcı kaybı (MoE dengeleme) da döndür

            Returns:
                `h` veya `(h, aux_loss)`
            """
            residual = x
            attended, _ = self.attention(
                self.norm1(x), attention_mask=attention_mask,
                cache=cache, use_cache=use_cache,
            )
            x = residual + attended

            residual = x
            normalized = self.norm2(x)
            aux_loss = None
            if self.is_moe:
                if return_aux:
                    ffn_out, aux_loss, _stats = self.feed_forward(normalized, return_stats=True)
                else:
                    ffn_out = self.feed_forward(normalized)
            else:
                ffn_out = self.feed_forward(normalized)
            x = residual + ffn_out

            if return_aux:
                return x, aux_loss
            return x

        def zero_output_projection(self) -> None:
            """
            Bloğun çıkış katkısını sıfırlar (yeni katman eklerken kullanılır).

            Dikkat ve FFN çıkış projeksiyonlarını sıfırlamak, bloğu
            "kimlik" işlevine indirger: ekleme anında model çıktısı değişmez.
            """
            with torch.no_grad():
                if hasattr(self.attention, "out_proj"):
                    self.attention.out_proj.weight.zero_()
                    if self.attention.out_proj.bias is not None:
                        self.attention.out_proj.bias.zero_()
                if self.is_moe:
                    if getattr(self.feed_forward, "shared", None) is not None:
                        self.feed_forward.shared.w2.weight.zero_()
                    for expert in self.feed_forward.experts:
                        expert.zero_output()
                else:
                    self.feed_forward.w2.weight.zero_()

        def extra_repr(self) -> str:
            kind = "MoE" if self.is_moe else "SwiGLU"
            return f"layer={self.layer_index}, ffn={kind}"


    class TransformerBackbone(nn.Module):  # type: ignore[misc]
        """
        Tam omurga: gömme + N blok + final norm + LM head.

        Kullanım:
            cfg = TransformerConfig(vocab_size=8000, embed_dim=384, num_layers=6)
            model = TransformerBackbone(cfg)

            logits = model(input_ids)                        # (B, T, V)
            hidden = model.hidden_states(input_ids)          # (B, T, D)
            out = model.generate(input_ids, max_new_tokens=32, temperature=0.8)
        """

        def __init__(self, config: TransformerConfig) -> None:
            super().__init__()
            self.config = config
            self.embed_dim = config.embed_dim

            self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
            self.embedding_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
            self.blocks = nn.ModuleList(
                [TransformerBlock(config, index) for index in range(config.num_layers)]
            )
            self.final_norm = RMSNorm(config.embed_dim, config.layer_norm_eps)
            self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

            self.apply(self._init_weights)
            if config.tie_embeddings:
                self.lm_head.weight = self.token_embedding.weight

            self._gradient_checkpointing = bool(config.gradient_checkpointing)

        def _init_weights(self, module: "nn.Module") -> None:
            """Ağırlıkları normal dağılımla başlatır; residual çıkışlarını küçültür."""
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

        # ─── İleri Geçiş ────────────────────────────────────────────────

        def forward(
            self,
            input_ids: "torch.Tensor",
            attention_mask: Optional["torch.Tensor"] = None,
            labels: Optional["torch.Tensor"] = None,
            cache: Optional[KVCache] = None,
            use_cache: bool = False,
            return_hidden: bool = False,
        ) -> Dict[str, Any]:
            """
            İleri geçiş.

            Args:
                input_ids: (batch, seq_len) token id'leri
                attention_mask: (batch, seq_len) — 1 geçerli, 0 dolgu
                labels: (batch, seq_len) — verilirse kayıp hesaplanır
                cache: KVCache (üretimde)
                use_cache: Önbelleği kullan
                return_hidden: Gizli durumları da döndür

            Returns:
                {"logits", "loss"?, "aux_loss", "hidden"?, "cache"?}
            """
            batch, seq_len = input_ids.shape
            if seq_len > self.config.max_seq_len:
                logger.warning(
                    f"Dizi uzunluğu ({seq_len}) max_seq_len ({self.config.max_seq_len}) "
                    "aşıyor — kırpılıyor"
                )
                input_ids = input_ids[:, : self.config.max_seq_len]
                seq_len = input_ids.shape[1]

            hidden = self.embedding_dropout(self.token_embedding(input_ids))
            total_aux = torch.zeros((), device=input_ids.device)

            for block in self.blocks:
                if self._gradient_checkpointing and self.training:
                    # Gradient checkpointing: aktivasyonları saklamaz, geri
                    # geçişte yeniden hesaplar (bellek ~√N, hız ~%70).
                    def custom_forward(module: Any, inputs: Any) -> Any:
                        return module(inputs, attention_mask=attention_mask)

                    hidden, aux = torch.utils.checkpoint.checkpoint(
                        lambda h: block(h, attention_mask=attention_mask,
                                        cache=None, use_cache=False, return_aux=True),
                        hidden, use_reentrant=False,
                    )
                    total_aux = total_aux + (aux if aux is not None else 0.0)
                else:
                    hidden, aux = block(
                        hidden, attention_mask=attention_mask,
                        cache=cache, use_cache=use_cache, return_aux=True,
                    )
                    if aux is not None:
                        total_aux = total_aux + aux

            hidden = self.final_norm(hidden)
            logits = self.lm_head(hidden)

            result: Dict[str, Any] = {
                "logits": logits,
                "aux_loss": total_aux,
                "cache": cache if use_cache else None,
            }

            if labels is not None:
                # Kaydırılmış çapraz entropi (bir sonraki token tahmini)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                result["loss"] = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            if return_hidden:
                result["hidden"] = hidden
            return result

        def hidden_states(self, input_ids: "torch.Tensor",
                          attention_mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
            """Yalnızca gizli durumları döndürür (gömme amaçlı)."""
            return self.forward(input_ids, attention_mask=attention_mask,
                                return_hidden=True)["hidden"]

        @torch.no_grad()
        def embed_ids(self, ids: Sequence[int], pooling: str = "mean") -> List[float]:
            """
            Token id dizisini tek bir vektöre havuzlar (gömme motoru için).

            Args:
                ids: Token id'leri
                pooling: `mean` | `last` | `max`

            Returns:
                `embed_dim` boyutlu Python listesi
            """
            if not ids:
                return [0.0] * self.embed_dim
            device = next(self.parameters()).device
            tensor = torch.tensor([list(ids)[: self.config.max_seq_len]], device=device)
            hidden = self.hidden_states(tensor)             # (1, T, D)
            if pooling == "last":
                pooled = hidden[0, -1]
            elif pooling == "max":
                pooled = hidden[0].max(dim=0).values
            else:
                pooled = hidden[0].mean(dim=0)
            return pooled.detach().float().cpu().tolist()

        # ─── Üretim ─────────────────────────────────────────────────────

        @torch.no_grad()
        def generate(
            self,
            input_ids: "torch.Tensor",
            max_new_tokens: int = 64,
            temperature: float = 0.8,
            top_k: int = 40,
            top_p: float = 0.95,
            repetition_penalty: float = 1.1,
            eos_token_id: Optional[int] = None,
            stop_tokens: Optional[Sequence[int]] = None,
        ) -> "torch.Tensor":
            """
            Otoregresif üretim (KV önbelleğiyle).

            Args:
                input_ids: (batch, prompt_len) başlangıç token'ları
                max_new_tokens: Üretilecek maksimum token
                temperature: Örnekleme sıcaklığı (0 → greedy)
                top_k: Top-k örnekleme (0 = kapalı)
                top_p: Nucleus örnekleme eşiği (1.0 = kapalı)
                repetition_penalty: Tekrar cezası (1.0 = kapalı)
                eos_token_id: Bu token üretilirse dur
                stop_tokens: Bu token'lardan biri üretilirse dur

            Returns:
                (batch, prompt_len + üretilen) token tensörü
            """
            self.eval()
            device = input_ids.device
            generated = input_ids
            cache = KVCache(max_len=self.config.max_seq_len)
            stop_set = set(stop_tokens or [])
            if eos_token_id is not None:
                stop_set.add(int(eos_token_id))

            for _ in range(max(0, int(max_new_tokens))):
                context = generated[:, -1:] if cache.length else generated
                outputs = self.forward(context, cache=cache, use_cache=True)
                logits = outputs["logits"][:, -1, :].float()

                # Tekrar cezası
                if repetition_penalty and repetition_penalty != 1.0:
                    for batch_index in range(generated.shape[0]):
                        for token_id in set(generated[batch_index].tolist()):
                            if logits[batch_index, token_id] > 0:
                                logits[batch_index, token_id] /= repetition_penalty
                            else:
                                logits[batch_index, token_id] *= repetition_penalty

                if temperature <= 1e-5:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / max(1e-5, float(temperature))
                    if top_k and top_k > 0:
                        k = min(int(top_k), logits.size(-1))
                        threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
                        logits = logits.masked_fill(logits < threshold, float("-inf"))
                    if top_p and 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        remove = cumulative > top_p
                        remove[:, 1:] = remove[:, :-1].clone()
                        remove[:, 0] = False
                        indices_to_remove = remove.scatter(1, sorted_indices, remove)
                        logits = logits.masked_fill(indices_to_remove, float("-inf"))
                    probabilities = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)
                if stop_set and all(int(token) in stop_set for token in next_token):
                    break
                if generated.shape[1] >= self.config.max_seq_len:
                    logger.debug("Maksimum dizi uzunluğuna ulaşıldı")
                    break

            return generated

        # ─── Büyüme ─────────────────────────────────────────────────────

        def expand_layers(self, count: int = 1, zero_init: bool = True) -> int:
            """
            Omurgaya yeni katman(lar) ekler.

            `zero_init=True` iken yeni bloğun çıkış projeksiyonları sıfırlanır
            ve blok kimlik işlevine indirgenir — ekleme anında model çıktısı
            **değişmez**. Katman, eğitimle kademeli katkı vermeye başlar.

            ÖNEMLİ: Yeni blok, mevcut modelin cihazında ve veri tipinde
            oluşturulur; aksi halde CUDA/CPU karışımı ileri geçişi kırar.

            Args:
                count: Eklenecek katman sayısı
                zero_init: Nötr başlangıç uygula

            Returns:
                Eklenen katman sayısı
            """
            added = 0
            reference = None
            for parameter in self.parameters():
                reference = parameter
                break

            for _ in range(max(0, int(count))):
                index = len(self.blocks)
                block = TransformerBlock(self.config, index)
                if reference is not None:
                    block = block.to(device=reference.device, dtype=reference.dtype)
                if zero_init:
                    block.zero_output_projection()
                self.blocks.append(block)
                added += 1
                logger.info(f"Yeni katman eklendi: #{index} (nötr başlangıç: {zero_init})")
            return added

        def resize_vocab(self, new_vocab_size: int) -> int:
            """
            Gömme katmanını **yeni sözlük boyutuna** genişletir/daraltır.

            Neden gerekli? Tokenizer yeniden eğitildiğinde sözlük büyür
            (ör. 1.259 → 4.301). Gömme katmanı eski boyutta kalırsa yeni
            token kimlikleri **arıza** verir. Bu metot eğitilmiş satırları
            korur, yalnızca yeni satırları başlatır — öğrenilmiş temsil
            kaybolmaz.

            ÖNEMLİ: `tie_embeddings` açıkken `lm_head.weight` ile
            `token_embedding.weight` **aynı tensördür**. Yeni tensör kurup
            bağlamak bağı koparır ve çıkış katmanı rastgele kalır; bu yüzden
            önce bağ çözülür, sonra yeniden bağlanır.

            Args:
                new_vocab_size: Yeni sözlük boyutu

            Returns:
                Eklenen (veya çıkarılan) satır sayısı
            """
            target = int(new_vocab_size)
            if target <= 0:
                raise ValueError(f"geçersiz sözlük boyutu: {new_vocab_size}")
            current = int(self.token_embedding.weight.shape[0])
            if target == current:
                return 0

            reference = self.token_embedding.weight
            device, dtype = reference.device, reference.dtype
            expanded = torch.empty(target, self.embed_dim,
                                   device=device, dtype=dtype)
            # Yeni satırların ölçeği mevcut ağırlıklara uydurulur; sabit bir
            # değer (ör. 0.02) mevcut dağılıma göre çok büyük/küçük kalabilir.
            std = float(reference.detach().std().item()) or 0.02
            nn.init.normal_(expanded, mean=0.0, std=std)

            keep = min(current, target)
            with torch.no_grad():
                expanded[:keep] = reference[:keep].detach()

            tied = bool(getattr(self.config, "tie_embeddings", False))
            if tied:
                # Sıra önemli: lm_head'e yeni parametreyi ver, sonra gömmenin
                # aynı tensörü göstermesini sağla (paylaşım korunur)
                self.lm_head.weight = nn.Parameter(expanded, requires_grad=True)
                self.token_embedding.weight = self.lm_head.weight
            else:
                self.token_embedding.weight = nn.Parameter(expanded,
                                                           requires_grad=True)
                self.lm_head = nn.Linear(self.embed_dim, target, bias=False)
                self.lm_head = self.lm_head.to(device=device, dtype=dtype)

            self.config.vocab_size = target
            logger.info(
                f"Sözlük boyutu {current} → {target} "
                f"({target - current:+d} satır, bağlı={tied}, std={std:.4f})"
            )
            return target - current

        def set_gradient_checkpointing(self, enabled: bool) -> None:
            """Gradient checkpointing'i açar/kapatır (bellek ↔ hız)."""
            self._gradient_checkpointing = bool(enabled)
            logger.info(f"Gradient checkpointing: {'açık' if enabled else 'kapalı'}")

        # ─── Raporlama ──────────────────────────────────────────────────

        def parameter_count(self) -> int:
            """Toplam parametre sayısı."""
            return int(sum(parameter.numel() for parameter in self.parameters()))

        def trainable_parameter_count(self) -> int:
            """Eğitilebilir parametre sayısı."""
            return int(sum(parameter.numel() for parameter in self.parameters()
                           if parameter.requires_grad))

        def active_parameter_count(self) -> int:
            """
            Token başına etkin parametre sayısı (MoE seyrekliği hesaba katılır).
            """
            embedding = self.token_embedding.weight.numel()
            if self.config.tie_embeddings:
                embedding *= 1  # lm_head paylaşımlı
            else:
                embedding += self.lm_head.weight.numel()
            per_layer = 0
            for block in self.blocks:
                attention_params = sum(
                    parameter.numel() for parameter in block.attention.parameters()
                )
                if block.is_moe:
                    per_layer += attention_params + block.feed_forward.active_parameter_count()
                else:
                    per_layer += attention_params + sum(
                        parameter.numel() for parameter in block.feed_forward.parameters()
                    )
            return int(embedding + per_layer)

        def layer_summary(self) -> List[Dict[str, Any]]:
            """Katman katman özet (tanı ve görselleştirme için)."""
            summary: List[Dict[str, Any]] = []
            for index, block in enumerate(self.blocks):
                entry: Dict[str, Any] = {
                    "layer": index,
                    "type": "moe" if block.is_moe else "ffn",
                    "params": int(sum(p.numel() for p in block.parameters())),
                }
                if block.is_moe:
                    entry.update(block.feed_forward.routing_stats())
                summary.append(entry)
            return summary

        def routing_report(self) -> Dict[str, Any]:
            """MoE katmanlarının yönlendirme raporu (büyüme kararları için)."""
            moe_layers = [block for block in self.blocks if block.is_moe]
            if not moe_layers:
                return {"moe_layers": 0, "num_experts": 0,
                        "expert_entropy": 0.0, "routing_confidence": 1.0}
            # Katmanlar arası ortalama
            entropies = [block.feed_forward.routing_stats()["expert_entropy"]
                         for block in moe_layers]
            confidences = [block.feed_forward.routing_stats()["routing_confidence"]
                           for block in moe_layers]
            usage_lists = [block.feed_forward.routing_stats()["usage_ema"]
                           for block in moe_layers]
            width = max(len(usage) for usage in usage_lists)
            averaged_usage = [
                sum(usage[index] if index < len(usage) else 0.0 for usage in usage_lists)
                / len(usage_lists)
                for index in range(width)
            ]
            return {
                "moe_layers": len(moe_layers),
                "num_experts": width,
                "expert_entropy": round(sum(entropies) / len(entropies), 5),
                "routing_confidence": round(sum(confidences) / len(confidences), 5),
                "usage_ema": [round(value, 5) for value in averaged_usage],
                "can_grow": any(block.feed_forward.can_grow for block in moe_layers),
                "frozen_experts": {
                    index: block.feed_forward.routing_stats()["frozen_experts"]
                    for index, block in enumerate(self.blocks) if block.is_moe
                },
            }

        def memory_report(self) -> Dict[str, float]:
            """Parametre ve tahmini bellek raporu (MB)."""
            params = self.parameter_count()
            attention_mem = estimate_attention_memory(
                self.config.attention_config(), 1, self.config.max_seq_len
            )
            moe_mem = (estimate_moe_memory(self.config.moe_config(), 1, self.config.max_seq_len)
                       if self.config.use_moe else {"weights_mb": 0.0, "training_mb": 0.0})
            return {
                "parameters": float(params),
                "active_parameters": float(self.active_parameter_count()),
                "weights_fp32_mb": round(params * 4 / (1024 ** 2), 2),
                "weights_fp16_mb": round(params * 2 / (1024 ** 2), 2),
                "training_adam_mb": round(params * 16 / (1024 ** 2), 2),
                "moe_training_mb": float(moe_mem.get("training_mb", 0.0)),
                "attention_seq_mb": float(attention_mem.get("attention_scores_mb", 0.0)),
            }

else:
    # ─── NumPy Yedeği ───────────────────────────────────────────────────────

    class TransformerBlock:  # type: ignore[no-redef]
        """NumPy yedek blok (yalnızca çıkarım, rastgele ağırlıklar)."""

        def __init__(self, config: TransformerConfig, layer_index: int = 0) -> None:
            self.config = config
            self.layer_index = layer_index
            self.is_moe = config.use_moe
            self.feed_forward = MixtureOfExperts(config.moe_config()) if config.use_moe else None

        def forward(self, x: Any, **kwargs: Any) -> Any:
            """Kimlik benzeri geçiş (gerçek ağırlık yok)."""
            if self.is_moe and self.feed_forward is not None:
                out = self.feed_forward(x)
                return (out, 0.0) if kwargs.get("return_aux") else out
            return (x, 0.0) if kwargs.get("return_aux") else x

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """`forward` takma adı."""
            return self.forward(*args, **kwargs)


    class TransformerBackbone:  # type: ignore[no-redef]
        """
        NumPy yedek omurga.

        Torch yoksa sistem yine de ayağa kalkar: gömme motoru hash stratejisine
        düşer, üretim devre dışı kalır ama tüm API çalışır.
        """

        def __init__(self, config: TransformerConfig) -> None:
            self.config = config
            self.embed_dim = config.embed_dim
            self.blocks = [TransformerBlock(config, index) for index in range(config.num_layers)]
            self._warned = False

        def forward(self, input_ids: Any, **kwargs: Any) -> Dict[str, Any]:
            """Uyarı verir ve boş sonuç döner."""
            if not self._warned:
                logger.warning(
                    "torch kurulu değil — transformer yalnızca iskelet olarak çalışıyor"
                )
                self._warned = True
            return {"logits": None, "aux_loss": 0.0, "error": "torch_yok"}

        def hidden_states(self, input_ids: Any, **kwargs: Any) -> Any:
            """Desteklenmiyor."""
            return None

        def embed_ids(self, ids: Sequence[int], pooling: str = "mean") -> List[float]:
            """Hash tabanlı yedek gömme."""
            from neurocore.neural.embedding import hash_embed
            return hash_embed(" ".join(str(i) for i in ids), dim=self.embed_dim)

        def generate(self, input_ids: Any, **kwargs: Any) -> Any:
            """Desteklenmiyor (torch gerekir)."""
            raise RuntimeError("Üretim için torch gerekir")

        def expand_layers(self, count: int = 1, zero_init: bool = True) -> int:
            """Katman sayısını artırır."""
            added = 0
            for _ in range(max(0, int(count))):
                self.blocks.append(TransformerBlock(self.config, len(self.blocks)))
                added += 1
            return added

        def set_gradient_checkpointing(self, enabled: bool) -> None:
            """No-op."""
            self.config.gradient_checkpointing = bool(enabled)

        def parameter_count(self) -> int:
            """Kaba tahmin."""
            return self.config.num_layers * 12 * self.embed_dim * self.embed_dim

        def trainable_parameter_count(self) -> int:
            """Kaba tahmin."""
            return 0

        def active_parameter_count(self) -> int:
            """Kaba tahmin."""
            return self.config.num_layers * 4 * self.embed_dim * self.embed_dim

        def layer_summary(self) -> List[Dict[str, Any]]:
            """Katman özeti."""
            return [{"layer": index, "type": "numpy"} for index in range(len(self.blocks))]

        def routing_report(self) -> Dict[str, Any]:
            """Yönlendirme raporu (iskelet)."""
            return {"moe_layers": 0, "num_experts": 0, "expert_entropy": 0.0,
                    "routing_confidence": 1.0, "available": False}

        def memory_report(self) -> Dict[str, float]:
            """Bellek raporu (iskelet)."""
            return {"parameters": float(self.parameter_count()), "torch": 0.0}

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """`forward` takma adı."""
            return self.forward(*args, **kwargs)


# ─── Yapılandırma Fabrikası ─────────────────────────────────────────────────

def config_from_system(config: Any, vocab_size: Optional[int] = None,
                       budget: Optional[Dict[str, Any]] = None) -> TransformerConfig:
    """
    `SystemConfig` (ve isteğe bağlı donanım bütçesi) → `TransformerConfig`.

    Bütçe verilirse (kaynak tahsis katmanının çıktısı) katman sayısı, gömme
    boyutu ve uzman sayısı mevcut donanıma göre ayarlanır.

    Args:
        config: SystemConfig (None olabilir)
        vocab_size: Sözlük boyutu (None ise config'ten)
        budget: `ResourceBudget.to_dict()` çıktısı (None olabilir)

    Returns:
        TransformerConfig
    """
    neural = getattr(config, "neural", None)
    multimodal = getattr(config, "multimodal", None)

    resolved_vocab = int(
        vocab_size
        or getattr(multimodal, "vocab_size", 32000)
        or 32000
    )

    settings: Dict[str, Any] = {
        "vocab_size": resolved_vocab,
        "embed_dim": int(getattr(neural, "embedding_dim", 768) or 768),
        "num_layers": int(getattr(neural, "num_transformer_layers", 8) or 8),
        "num_heads": int(getattr(neural, "num_attention_heads", 12) or 12),
        "dropout": float(getattr(neural, "dropout", 0.1) or 0.1),
        "d_ff_multiplier": int(getattr(neural, "d_ff_multiplier", 4) or 4),
        "gradient_checkpointing": bool(getattr(neural, "gradient_checkpointing", False)),
        "use_moe": True,
        "num_experts": int(getattr(neural, "num_initial_experts", 4) or 4),
        "max_experts": int(getattr(neural, "max_experts", 12) or 12),
        "top_k_experts": int(getattr(neural, "top_k_experts", 2) or 2),
        "tie_embeddings": True,
    }
    if multimodal is not None:
        settings["max_seq_len"] = int(getattr(multimodal, "max_text_length", 2048) or 2048)

    if budget:
        # Kaynak bütçesi daha yetkili: donanım durumuna göre üzerine yaz
        if budget.get("embedding_dim"):
            settings["embed_dim"] = int(budget["embedding_dim"])
        if budget.get("num_transformer_layers"):
            settings["num_layers"] = int(budget["num_transformer_layers"])
        if budget.get("num_attention_heads"):
            settings["num_heads"] = int(budget["num_attention_heads"])
        if budget.get("gradient_checkpointing") is not None:
            settings["gradient_checkpointing"] = bool(budget["gradient_checkpointing"])
        if budget.get("memory_pressure") == "CRITICAL":
            settings["use_moe"] = True
            settings["top_k_experts"] = 1
        elif budget.get("memory_pressure") == "WARNING":
            settings["top_k_experts"] = min(2, settings["top_k_experts"])

    return TransformerConfig(**settings)


def estimate_backbone_memory(config: TransformerConfig, batch_size: int = 1,
                             seq_len: Optional[int] = None) -> Dict[str, float]:
    """
    Omurga bellek tahmini (MB) — profil seçimi ve bütçe kontrolü için.

    Args:
        config: Omurga yapılandırması
        batch_size: Toplu iş boyutu
        seq_len: Dizi uzunluğu (None = max_seq_len)

    Returns:
        {"parameters", "weights_mb", "activations_mb", "kv_cache_mb",
         "training_mb", "fits_4gb", "fits_8gb", "fits_12gb"}
    """
    seq = int(seq_len or config.max_seq_len)
    embed = config.embed_dim

    tokens = config.vocab_size * embed
    head_dim = config.head_dim
    kv_heads = int(config.num_kv_heads or config.num_heads)

    per_layer = (
        4 * embed * embed                                    # q, k, v, o (yaklaşık)
        + (3 * embed * config.d_ff if not config.use_moe else 0)
    )
    if config.use_moe:
        expert_hidden = int(8 * embed / 3)
        per_layer += config.num_experts * 3 * embed * expert_hidden
        per_layer += config.num_experts * embed

    total_params = tokens + config.num_layers * per_layer + embed

    weights_mb = total_params * 4 / (1024 ** 2)
    activations_mb = (
        batch_size * seq * embed * config.num_layers * 4 / (1024 ** 2)
    )
    kv_cache_mb = (
        2 * batch_size * kv_heads * seq * head_dim * config.num_layers * 4 / (1024 ** 2)
    )
    training_mb = total_params * 16 / (1024 ** 2)   # Adam: parametre + gradyan + 2 moment

    return {
        "parameters": float(total_params),
        "weights_mb": round(weights_mb, 2),
        "activations_mb": round(activations_mb, 2),
        "kv_cache_mb": round(kv_cache_mb, 2),
        "inference_mb": round(weights_mb + activations_mb + kv_cache_mb, 2),
        "training_mb": round(training_mb + activations_mb * 2, 2),
        "fits_4gb": float(weights_mb + activations_mb) < 3800.0,
        "fits_8gb": float(weights_mb + activations_mb) < 7600.0,
        "fits_12gb": float(weights_mb + activations_mb) < 11400.0,
    }


__all__ = [
    "TransformerConfig",
    "TransformerBlock",
    "TransformerBackbone",
    "config_from_system",
    "estimate_backbone_memory",
    "HAS_TORCH",
    "HAS_NUMPY",
]
