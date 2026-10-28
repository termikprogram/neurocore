"""
NeuroCore Uzman Karışımı (MoE)
==============================
Seyrek uzman yönlendirmesi — modelin **büyüyebilmesinin** temel mekanizması.

Neden MoE?
    Kapasiteyi parametre sayısıyla değil **uzman sayısıyla** ölçekler. Her
    token yalnızca `top_k` uzmanı çalıştırdığı için toplam parametre artsa da
    işlem maliyeti sabit kalır. Bu, "küçük bilgilerden kendini büyütme"
    hedefinin doğrudan karşılığıdır: yeni bir konu öğrenildiğinde yeni bir
    uzman doğar, mevcut bilgi bozulmaz.

Yönlendirme
    ``router(x) = softmax(W_r · x)`` → en yüksek `top_k` uzman seçilir.
    Yük dengeleme kaybı (Switch Transformer tarzı) belirli uzmanların
    tekeline girmesini engeller: ``L_bal = α · n · Σ(f_i · P_i)``.

Büyüme (residual sıfır ile ekleme)
    Yeni uzman **sıfır** çıkış verecek biçimde başlatılır (çıkış projeksiyonu
    sıfır). Böylece ekleme anında model fonksiyonu hiç değişmez; uzman
    eğitimle yavaşça katkı vermeye başlar. Bu, sürekli öğrenmede
    "catastrophic forgetting"i önleyen kritik bir tekniktir.

Soğuk başlangıç: uzmanlar tembel (lazy) oluşturulur; `max_experts` sınırına
kadar bellek yalnızca gerçekten eklenen uzmanlar için harcanır.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.moe")


# ─── Backend ────────────────────────────────────────────────────────────────

try:  # pragma: no cover
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

from neurocore.neural.attention import RMSNorm, SwiGLU


# ─── Yapılandırma ───────────────────────────────────────────────────────────

@dataclass
class MoEConfig:
    """Uzman karışımı yapılandırması."""
    embed_dim: int = 768
    num_experts: int = 4                  # başlangıç uzman sayısı
    max_experts: int = 12                 # büyüme sınırı
    top_k: int = 2                        # token başına etkin uzman
    expert_hidden_dim: Optional[int] = None
    balance_loss_weight: float = 0.01     # yük dengeleme katsayısı
    router_jitter: float = 0.0            # eğitimde yönlendirici gürültüsü
    router_z_loss_weight: float = 0.001   # logit büyüklüğü cezası
    dropout: float = 0.0
    capacity_factor: Optional[float] = None   # uzman başına token sınırı
    shared_expert: bool = True            # her zaman aktif paylaşılan uzman
    normalize_gates: bool = True
    #: Yeni uzmanın yönlendirme bias'ı. Çok negatif başlar → yeni uzman
    #: top-k'ya **girmez**, dolayısıyla büyüme model çıktısını değiştirmez.
    #: Eğitimle (dengeleme kaybı gradyanı) kademeli olarak yükselir.
    new_expert_bias: float = -12.0
    #: Bias'ın düşebileceği en düşük değer (sayısal kararlılık sınırı).
    router_bias_floor: float = -15.0

    def __post_init__(self) -> None:
        self.num_experts = max(1, int(self.num_experts))
        self.max_experts = max(self.num_experts, int(self.max_experts))
        self.top_k = max(1, min(int(self.top_k), self.num_experts))
        self.new_expert_bias = min(-1.0, float(self.new_expert_bias))
        self.router_bias_floor = min(float(self.new_expert_bias), float(self.router_bias_floor))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embed_dim": self.embed_dim, "num_experts": self.num_experts,
            "max_experts": self.max_experts, "top_k": self.top_k,
            "expert_hidden_dim": self.expert_hidden_dim,
            "balance_loss_weight": self.balance_loss_weight,
            "shared_expert": self.shared_expert,
            "capacity_factor": self.capacity_factor,
            "dropout": self.dropout,
            "new_expert_bias": self.new_expert_bias,
        }


@dataclass
class RouterStats:
    """
    Bir ileri geçişte toplanan yönlendirme istatistikleri.

    Attributes:
        expert_usage: Uzman başına **atama oranı** (`top_k > 1` iken toplam
            `top_k` olur, çünkü her belirteç birden çok uzmana atanır).
            Tek uzman başına göreli yük için `expert_share()` kullanın.
        expert_entropy: Normalize edilmiş yönlendirme entropisi (0..1)
        routing_confidence: Seçilen kapı ağırlıklarının ortalaması (0..1)
        balance_loss: Switch-tarzı yük dengeleme kaybı
        z_loss: Logit büyüklüğü cezası
        dropped_tokens: Kapasite nedeniyle düşürülen belirteç sayısı
        active_experts: Bu geçişte en az bir belirteç alan uzman sayısı
    """
    expert_usage: List[float] = field(default_factory=list)
    expert_entropy: float = 0.0
    routing_confidence: float = 0.0
    balance_loss: float = 0.0
    z_loss: float = 0.0
    dropped_tokens: int = 0
    active_experts: int = 0

    def expert_share(self) -> List[float]:
        """
        Uzman başına **göreli** yük payı (toplamı 1.0).

        `expert_usage` atama oranıdır ve `top_k > 1` iken toplamı `top_k`
        olur; budama/dondurma kararları bu göreli paya göre verilir.
        """
        total = sum(self.expert_usage)
        if total <= 0:
            return [0.0] * len(self.expert_usage)
        return [value / total for value in self.expert_usage]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl (göreli pay dahil)."""
        return {
            "expert_usage": [round(value, 5) for value in self.expert_usage],
            "expert_share": [round(value, 5) for value in self.expert_share()],
            "expert_entropy": round(self.expert_entropy, 5),
            "routing_confidence": round(self.routing_confidence, 5),
            "balance_loss": round(self.balance_loss, 5),
            "z_loss": round(self.z_loss, 5),
            "dropped_tokens": self.dropped_tokens,
            "active_experts": self.active_experts,
        }


# ─── Torch Uygulaması ───────────────────────────────────────────────────────

if HAS_TORCH:
    class Expert(nn.Module):  # type: ignore[misc]
        """
        Tek bir uzman: SwiGLU ileri besleme ağı.

        `zero_init=True` ile çıkış projeksiyonu sıfırlanır — yeni uzman
        modelin davranışını bozmadan eklenir (residual sıfır tekniği).
        """

        def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                     dropout: float = 0.0, zero_init: bool = False) -> None:
            super().__init__()
            self.net = SwiGLU(dim, hidden_dim, dropout)
            self.expert_id: int = -1
            self.usage: float = 0.0
            self.frozen: bool = False
            self.created_at: float = 0.0
            if zero_init:
                with torch.no_grad():
                    self.net.w2.weight.zero_()
                    if self.net.w2.bias is not None:
                        self.net.w2.bias.zero_()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Uzman ileri geçişi (donmuşsa kimlik döner)."""
            if self.frozen:
                return torch.zeros_like(x)
            return self.net(x)

        def freeze(self) -> None:
            """Uzmanı dondurur (çıktısı sıfırlanır, budama yerine)."""
            self.frozen = True
            for parameter in self.parameters():
                parameter.requires_grad_(False)

        def unfreeze(self) -> None:
            """Uzmanı yeniden etkinleştirir."""
            self.frozen = False
            for parameter in self.parameters():
                parameter.requires_grad_(True)

        def zero_output(self) -> None:
            """Çıkış projeksiyonunu sıfırlar (yeni ekleme/reset için)."""
            with torch.no_grad():
                self.net.w2.weight.zero_()
                if self.net.w2.bias is not None:
                    self.net.w2.bias.zero_()


    class Router(nn.Module):  # type: ignore[misc]
        """
        Gürültülü top-k yönlendirici (Switch Transformer / Shazeer tarzı).

        Uzman başına **bias** içerir. Bias, büyümenin nötr olmasını sağlayan
        kritik parçadır: yeni bir uzmanın bias'ı çok negatif başlar, dolayısıyla
        top-k seçimine girmez ve mevcut uzmanların kapı ağırlıkları değişmez.
        Eğitimde dengeleme kaybının gradyanı bias'ı yukarı çeker ve uzman
        kademeli olarak devreye girer (DeepSeek-V3 / Switch tarzı yumuşak
        başlangıç).
        """

        def __init__(self, dim: int, num_experts: int, jitter: float = 0.0,
                     bias_init: float = 0.0) -> None:
            super().__init__()
            self.dim = int(dim)
            self.num_experts = int(num_experts)
            self.jitter = float(jitter)
            self.weight = nn.Parameter(torch.empty(self.num_experts, self.dim))
            self.bias = nn.Parameter(torch.full((self.num_experts,), float(bias_init)))
            nn.init.normal_(self.weight, std=0.02)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (..., dim)

            Returns:
                (..., num_experts) logitleri
            """
            logits = F.linear(x, self.weight, self.bias)
            if self.training and self.jitter > 0:
                noise = torch.empty_like(logits).uniform_(1.0 - self.jitter, 1.0 + self.jitter)
                logits = logits * noise
            return logits


    class MixtureOfExperts(nn.Module):  # type: ignore[misc]
        """
        Seyrek uzman karışımı katmanı.

        Kullanım:
            moe = MixtureOfExperts(MoEConfig(embed_dim=768, num_experts=4, max_experts=12))
            output, aux_loss, stats = moe(x)

            moe.grow(1)                  # bir uzman ekle (residual sıfır)
            moe.freeze_expert(2)         # kullanılmayan uzmanı dondur
            moe.prune_expert(3)          # uzmanı tamamen kaldır

        Args:
            config: MoEConfig
        """

        def __init__(self, config: MoEConfig) -> None:
            super().__init__()
            self.config = config
            self.embed_dim = config.embed_dim
            self.experts = nn.ModuleList()
            self.router = Router(config.embed_dim, config.num_experts, config.router_jitter)
            self.shared = SwiGLU(config.embed_dim, config.expert_hidden_dim,
                                 config.dropout) if config.shared_expert else None
            self.norm = RMSNorm(config.embed_dim)

            self._stats = RouterStats()
            self._usage_ema: List[float] = []
            self._generation = 0

            for index in range(config.num_experts):
                self._add_expert(index, zero_init=False)
            self._refresh_usage()

        # ─── Uzman Yönetimi ─────────────────────────────────────────────

        def _add_expert(self, expert_id: int, zero_init: bool = True) -> None:
            """
            Yeni uzman oluşturur ve listeye ekler.

            ÖNEMLİ: Yeni uzman, mevcut modelin **cihazında ve veri tipinde**
            oluşturulmalıdır. Aksi halde `nn.ModuleList`'e eklenen uzman CPU'da
            kalır, model CUDA'da çalışırken ileri geçiş "Expected all tensors to
            be on the same device" hatası verir.
            """
            import time
            expert = Expert(self.embed_dim, self.config.expert_hidden_dim,
                            self.config.dropout, zero_init=zero_init)
            reference = None
            if self.experts:
                reference = next(self.experts[0].parameters(), None)
            elif self.shared is not None:
                reference = next(self.shared.parameters(), None)

            if reference is not None:
                expert = expert.to(device=reference.device, dtype=reference.dtype)

            expert.expert_id = expert_id
            expert.created_at = time.time()
            self.experts.append(expert)

        @property
        def num_experts(self) -> int:
            """Etkin uzman sayısı (donmuşlar dahil)."""
            return len(self.experts)

        @property
        def active_experts(self) -> int:
            """Donmamış uzman sayısı."""
            return sum(1 for expert in self.experts if not expert.frozen)

        @property
        def can_grow(self) -> bool:
            """Büyüme sınırına gelindi mi?"""
            return len(self.experts) < self.config.max_experts

        def grow(self, count: int = 1, zero_init: bool = True) -> int:
            """
            Yeni uzman(lar) ekler.

            Args:
                count: Kaç uzman eklenecek
                zero_init: Çıkışı sıfır başlat (davranışı bozmaz)

            Returns:
                Gerçekten eklenen uzman sayısı
            """
            added = 0
            for _ in range(max(0, int(count))):
                if not self.can_grow:
                    logger.info(
                        f"Uzman sınırına ulaşıldı ({self.config.max_experts}), büyüme durduruldu"
                    )
                    break
                expert_id = len(self.experts)
                self._add_expert(expert_id, zero_init=zero_init)
                added += 1

            if added:
                self._rebuild_router()
                self._generation += 1
                logger.info(
                    f"{added} uzman eklendi → toplam {len(self.experts)} "
                    f"(sınır {self.config.max_experts}, nesil {self._generation})"
                )
            return added

        def _rebuild_router(self) -> None:
            """
            Yönlendiriciyi yeni uzman sayısına göre yeniden kurar.

            Mevcut uzmanların ağırlıkları **korunur**; yeni uzmanların
            ağırlığı küçük rastgele değerlerle başlatılır. Bu, sürekli
            öğrenmede önceki bilgiyi bozmaz.

            ÖNEMLİ: `self.router.weight` doğrudan atanamaz — mevcut Adam
            optimizer durumu eski boyutu hatırlar ve güncelleme anında
            "expanded size ... must match" hatası verir. Bu yüzden parametre
            **yeni bir tensörle** değiştirilir ve `router.num_experts`
            güncellenir; optimizer yeni parametreyi ilk adımda sıfırdan kurar.
            """
            with torch.no_grad():
                old_weight = self.router.weight.detach()
                old_count = int(old_weight.shape[0])
                new_count = len(self.experts)
                if new_count == old_count:
                    return

                new_weight = torch.empty(
                    new_count, self.embed_dim,
                    device=old_weight.device, dtype=old_weight.dtype,
                )
                nn.init.normal_(new_weight, std=0.02)
                keep = min(old_count, new_count)
                new_weight[:keep] = old_weight[:keep]

                # ── Bias: büyümenin nötr olmasını sağlayan asıl mekanizma ──
                old_bias = self.router.bias.detach()
                new_bias = torch.empty(
                    new_count, device=old_bias.device, dtype=old_bias.dtype
                )
                new_bias[:keep] = old_bias[:keep]
                if new_count > old_count:
                    new_bias[old_count:] = float(self.config.new_expert_bias)

                self.router.num_experts = new_count
                self.router.weight = nn.Parameter(new_weight)
                self.router.bias = nn.Parameter(new_bias)
            self._refresh_usage()

        def freeze_expert(self, expert_id: int, reason: str = "") -> bool:
            """
            Uzmanı dondurur — silmez, çıktısını sıfırlar.

            `prune_expert`'ten farkı: geri alınabilir olmasıdır. Yönlendirme
            ağırlığı çok düşük olan uzmanlar için tercih edilir.
            """
            if 0 <= expert_id < len(self.experts):
                self.experts[expert_id].freeze()
                logger.info(f"Uzman {expert_id} donduruldu {f'({reason})' if reason else ''}")
                return True
            return False

        def unfreeze_expert(self, expert_id: int) -> bool:
            """Donmuş uzmanı yeniden etkinleştirir."""
            if 0 <= expert_id < len(self.experts):
                self.experts[expert_id].unfreeze()
                return True
            return False

        def prune_expert(self, expert_id: int) -> bool:
            """
            Uzmanı tamamen kaldırır (geri alınamaz).

            Kalıcı silme öncesi `freeze_expert` tercih edilmelidir; bu metot
            bellek baskısı altında kullanılır.
            """
            if not (0 <= expert_id < len(self.experts)):
                return False
            if len(self.experts) <= 1:
                logger.warning("Son uzman budanamaz")
                return False

            remaining = [expert for index, expert in enumerate(self.experts)
                         if index != expert_id]
            self.experts = nn.ModuleList(remaining)
            for index, expert in enumerate(self.experts):
                expert.expert_id = index
            self._rebuild_router()
            logger.info(f"Uzman {expert_id} budandı → {len(self.experts)} uzman kaldı")
            return True

        def reset_expert(self, expert_id: int) -> bool:
            """Uzmanın çıkışını sıfırlar (yeniden öğrenmeye başlatır)."""
            if 0 <= expert_id < len(self.experts):
                self.experts[expert_id].zero_output()
                self.experts[expert_id].unfreeze()
                return True
            return False

        # ─── Uyandırma (yeni uzmanı devreye alma) ───────────────────────

        def dormant_experts(self, threshold: float = -6.0) -> List[int]:
            """
            Yönlendirme bias'ı çok negatif olduğu için hiç seçilmeyen uzmanlar.

            Yeni eklenen uzmanlar bu durumdadır: modelde yer alırlar, gradyan
            alırlar, ama fonksiyona katkı vermezler.
            """
            with torch.no_grad():
                bias = self.router.bias.detach().cpu()
            return [index for index, value in enumerate(bias.tolist())
                    if float(value) < threshold]

        def activate_expert(self, expert_id: int, bias: float = 0.0) -> bool:
            """
            Uyuyan uzmanı devreye alır (bias'ı yükseltir).

            `grow(..., dormant=True)` ile eklenen uzmanlar bu çağrıyla
            (veya eğitimdeki dengeleme gradyanıyla) aktifleşir.

            Args:
                expert_id: Uzman indeksi
                bias: Yeni bias değeri (0.0 = diğerleriyle eşit şans)

            Returns:
                Başarılı ise True
            """
            if not (0 <= expert_id < len(self.experts)):
                return False
            with torch.no_grad():
                self.router.bias[expert_id] = float(bias)
            logger.info(f"Uzman {expert_id} devreye alındı (bias={bias})")
            return True

        def activate_all_dormant(self, bias: float = 0.0) -> int:
            """Tüm uyuyan uzmanları devreye alır; sayısını döndürür."""
            activated = 0
            for expert_id in self.dormant_experts():
                if self.activate_expert(expert_id, bias):
                    activated += 1
            return activated

        def decay_router_bias(self, amount: float = 2.0, floor: float = -1.0) -> int:
            """
            Çok negatif bias'ları yumuşatır (uyuyan uzmanı kademeli uyandırma).

            Uzun eğitimde bir uzman dengeleme gradyanıyla kendiliğinden
            devreye girmediyse bias'ını yükseltmek gerekir. Bias, `floor`
            değerinin altına inmez; `floor` varsayılanı -1.0'dır ve bu seviyede
            uzman top-k yarışına gerçekten katılır.

            Args:
                amount: Bu çağrıda eklenecek miktar (pozitif sayı)
                floor: Ulaşılacak en yüksek alt sınır (bias bu değeri aşmaz)

            Returns:
                Değiştirilen uzman sayısı
            """
            step = abs(float(amount))
            ceiling = float(floor)
            changed = 0
            with torch.no_grad():
                bias = self.router.bias
                for index in range(int(bias.shape[0])):
                    current = float(bias[index])
                    if current < ceiling:
                        bias[index] = min(ceiling, current + step)
                        changed += 1
            if changed:
                logger.info(f"{changed} uzmanın yönlendirme bias'ı yükseltildi (+{step})")
            return changed

        # ─── İleri Geçiş ────────────────────────────────────────────────

        def forward(
            self,
            x: "torch.Tensor",
            return_stats: bool = False,
        ) -> Any:
            """
            Seyrek uzman karışımı ileri geçişi.

            Args:
                x: (batch, seq_len, embed_dim) veya (batch, embed_dim)
                return_stats: Yönlendirme istatistiklerini de döndür

            Returns:
                `output` veya `(output, aux_loss, stats)`
            """
            original_shape = x.shape
            flat = x.reshape(-1, self.embed_dim)            # (N, D)
            num_tokens = flat.shape[0]
            num_experts = len(self.experts)

            logits = self.router(flat)                       # (N, E)
            probabilities = F.softmax(logits.float(), dim=-1)
            top_k = min(self.config.top_k, num_experts)
            top_values, top_indices = torch.topk(probabilities, top_k, dim=-1)

            if self.config.normalize_gates:
                gate_sum = top_values.sum(dim=-1, keepdim=True)
                # Sayısal güvence: toplam sıfıra çok yakınsa bölmeyi atla.
                top_values = torch.where(
                    gate_sum > 1e-9, top_values / (gate_sum + 1e-30), top_values
                )
            # Uyuyan (yeni eklenmiş, bias'ı çok negatif) uzmanlar top-k'ya hiç
            # girmediği için mevcut uzmanların kapı ağırlıkları **birebir**
            # korunur — büyüme model fonksiyonunu değiştirmez.

            # Uzman çıktılarını topla
            output = torch.zeros_like(flat)
            expert_token_counts = torch.zeros(num_experts, device=flat.device)

            for slot in range(top_k):
                expert_ids = top_indices[:, slot]            # (N,)
                gate = top_values[:, slot].to(flat.dtype)    # (N,)
                for expert_index in range(num_experts):
                    mask = (expert_ids == expert_index)
                    count = int(mask.sum().item())
                    if count == 0:
                        continue
                    expert_token_counts[expert_index] += count
                    expert_output = self.experts[expert_index](flat[mask])
                    output[mask] += gate[mask].unsqueeze(-1) * expert_output

            # Paylaşılan uzman (her token için her zaman aktif)
            if self.shared is not None:
                output = output + self.shared(flat)

            output = self.norm(output).reshape(original_shape)

            # ── Yardımcı kayıplar ──
            mean_probabilities = probabilities.mean(dim=0)                    # (E,)
            fraction_tokens = expert_token_counts / max(1, num_tokens)        # (E,)
            balance_loss = num_experts * torch.sum(fraction_tokens * mean_probabilities)
            z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()

            aux_loss = (
                self.config.balance_loss_weight * balance_loss
                + self.config.router_z_loss_weight * z_loss
            )

            # ── İstatistikler ──
            usage = fraction_tokens.detach().cpu().tolist()
            entropy = float(
                -(mean_probabilities * torch.log(mean_probabilities + 1e-9)).sum().item()
            )
            max_entropy = math.log(max(2, num_experts))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            confidence = float(top_values.mean().item())

            self._stats = RouterStats(
                expert_usage=usage,
                expert_entropy=normalized_entropy,
                routing_confidence=confidence,
                balance_loss=float(balance_loss.item()),
                z_loss=float(z_loss.item()),
                active_experts=int((expert_token_counts > 0).sum().item()),
            )
            self._update_usage_ema(usage)

            if return_stats:
                return output, aux_loss, self._stats
            return output

        def _update_usage_ema(self, usage: Sequence[float], alpha: float = 0.2) -> None:
            """Kullanım istatistiklerini üstel hareketli ortalamayla günceller."""
            if not self._usage_ema:
                self._usage_ema = [float(value) for value in usage]
                return
            length = min(len(self._usage_ema), len(usage))
            for index in range(length):
                self._usage_ema[index] = (
                    (1 - alpha) * self._usage_ema[index] + alpha * float(usage[index])
                )

        def _refresh_usage(self) -> None:
            """Uzman kullanım alanlarını EMA ile eşitler."""
            if len(self._usage_ema) != len(self.experts):
                self._usage_ema = [1.0 / max(1, len(self.experts))] * len(self.experts)
            for index, expert in enumerate(self.experts):
                expert.usage = self._usage_ema[index] if index < len(self._usage_ema) else 0.0

        # ─── Raporlama ──────────────────────────────────────────────────

        def routing_stats(self) -> Dict[str, Any]:
            """En son ileri geçişin yönlendirme istatistikleri."""
            return {
                **self._stats.to_dict(),
                "num_experts": len(self.experts),
                "active_experts": self.active_experts,
                "frozen_experts": [expert.expert_id for expert in self.experts if expert.frozen],
                "usage_ema": [round(value, 5) for value in self._usage_ema],
                "can_grow": self.can_grow,
                "generation": self._generation,
                "params": self.parameter_count(),
            }

        def parameter_count(self) -> int:
            """Toplam etkin parametre sayısı (paylaşılan uzman dahil)."""
            total = sum(parameter.numel() for parameter in self.parameters())
            return int(total)

        def active_parameter_count(self) -> int:
            """
            Token başına **gerçekten çalışan** parametre sayısı.

            MoE'nin ölçekleme avantajı burada görünür: toplam parametre çok
            olsa da her token yalnızca birkaç uzmanı çalıştırır.
            """
            expert_params = sum(
                parameter.numel() for parameter in self.experts[0].parameters()
            ) if self.experts else 0
            router_params = self.router.weight.numel()
            shared_params = (
                sum(parameter.numel() for parameter in self.shared.parameters())
                if self.shared is not None else 0
            )
            return int(expert_params * min(self.config.top_k, len(self.experts))
                       + router_params + shared_params)

        def load_balance_report(self) -> Dict[str, Any]:
            """
            Uzman yük dengesi raporu — büyüme/budama kararlarının girdisi.

            Returns:
                {"usage": [...], "imbalance": float, "idle": [id...],
                 "overloaded": [id...], "recommendation": str}
            """
            usage = list(self._usage_ema) or [0.0] * len(self.experts)
            if not usage:
                return {"usage": [], "imbalance": 0.0, "idle": [],
                        "overloaded": [], "recommendation": "uzman yok"}

            mean = sum(usage) / len(usage)
            variance = sum((value - mean) ** 2 for value in usage) / len(usage)
            imbalance = math.sqrt(variance) / mean if mean > 0 else 0.0

            idle = [index for index, value in enumerate(usage) if value < 0.02]
            overloaded = [index for index, value in enumerate(usage) if value > 2.5 * mean]

            if idle and not self.can_grow:
                recommendation = "bosta uzmanlar var — dondur veya sifirla"
            elif overloaded and self.can_grow:
                recommendation = "asiri yuklenen uzmanlar var — buyume onerilir"
            elif imbalance > 0.6:
                recommendation = "yuk dengesiz — dengeleme katsayisini artir"
            else:
                recommendation = "denge saglikli"

            return {
                "usage": [round(value, 5) for value in usage],
                "imbalance": round(imbalance, 5),
                "idle": idle,
                "overloaded": overloaded,
                "recommendation": recommendation,
            }

        def extra_repr(self) -> str:
            return (
                f"experts={len(self.experts)}/{self.config.max_experts}, "
                f"top_k={self.config.top_k}, dim={self.embed_dim}"
            )

else:
    # ─── NumPy Yedeği ───────────────────────────────────────────────────────

    class MixtureOfExperts:  # type: ignore[no-redef]
        """
        Uzman karışımı — NumPy yedeği (torch yoksa).

        Ağırlıklar deterministik tohumla üretilir; gerçek eğitim desteklenmez
        ama sistem ayağa kalkar ve yönlendirme istatistikleri üretilir.
        """

        def __init__(self, config: MoEConfig) -> None:
            self.config = config
            self.embed_dim = config.embed_dim
            self._initialized = False
            self._generation = 0
            self._usage_ema: List[float] = [1.0 / config.num_experts] * config.num_experts
            self._frozen: List[int] = []
            self._stats = RouterStats()

        @property
        def num_experts(self) -> int:
            """Etkin uzman sayısı."""
            return len(self._usage_ema)

        @property
        def active_experts(self) -> int:
            """Donmamış uzman sayısı."""
            return self.num_experts - len(self._frozen)

        @property
        def can_grow(self) -> bool:
            """Büyüme sınırına gelindi mi?"""
            return self.num_experts < self.config.max_experts

        def _ensure(self) -> None:
            """Ağırlıkları tembel başlatır."""
            if self._initialized or not HAS_NUMPY:
                return
            rng = np.random.default_rng(99)
            scale = 1.0 / math.sqrt(self.embed_dim)
            self.router_weight = rng.normal(0, scale, (self.num_experts, self.embed_dim)).astype(np.float32)
            self._initialized = True

        def forward(self, x: Any, return_stats: bool = False) -> Any:
            """NumPy yönlendirme + ağırlıklı birleştirme (kimlik tabanlı)."""
            self._ensure()
            if not HAS_NUMPY or not self._initialized:
                return (x, 0.0, self._stats) if return_stats else x

            single = x.ndim == 2
            flat = x.reshape(-1, self.embed_dim) if not single else x
            logits = flat @ self.router_weight.T
            logits = logits - logits.max(axis=-1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities = probabilities / (probabilities.sum(axis=-1, keepdims=True) + 1e-9)

            usage = probabilities.mean(axis=0).tolist()
            self._stats = RouterStats(
                expert_usage=usage,
                expert_entropy=float(
                    -(probabilities.mean(axis=0) *
                      np.log(probabilities.mean(axis=0) + 1e-9)).sum() /
                    math.log(max(2, self.num_experts))
                ),
                routing_confidence=float(probabilities.max(axis=-1).mean()),
                active_experts=self.num_experts,
            )
            output = flat
            if single:
                output = output[0]
            return (output, 0.0, self._stats) if return_stats else output

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """`forward` takma adı."""
            return self.forward(*args, **kwargs)

        def grow(self, count: int = 1, zero_init: bool = True) -> int:
            """Uzman sayısını artırır."""
            added = 0
            for _ in range(max(0, int(count))):
                if not self.can_grow:
                    break
                self._usage_ema.append(0.0)
                added += 1
            if added:
                self._initialized = False
                self._generation += 1
            return added

        def freeze_expert(self, expert_id: int, reason: str = "") -> bool:
            """Uzmanı dondurur."""
            if 0 <= expert_id < self.num_experts and expert_id not in self._frozen:
                self._frozen.append(expert_id)
                return True
            return False

        def unfreeze_expert(self, expert_id: int) -> bool:
            """Donmuş uzmanı etkinleştirir."""
            if expert_id in self._frozen:
                self._frozen.remove(expert_id)
                return True
            return False

        def prune_expert(self, expert_id: int) -> bool:
            """Uzmanı kaldırır."""
            if 0 <= expert_id < self.num_experts and self.num_experts > 1:
                self._usage_ema.pop(expert_id)
                if expert_id in self._frozen:
                    self._frozen.remove(expert_id)
                self._initialized = False
                return True
            return False

        def reset_expert(self, expert_id: int) -> bool:
            """Uzmanı sıfırlar."""
            return 0 <= expert_id < self.num_experts

        def routing_stats(self) -> Dict[str, Any]:
            """Yönlendirme istatistikleri."""
            return {**self._stats.to_dict(), "num_experts": self.num_experts,
                    "active_experts": self.active_experts,
                    "frozen_experts": list(self._frozen),
                    "usage_ema": [round(v, 5) for v in self._usage_ema],
                    "can_grow": self.can_grow, "generation": self._generation}

        def parameter_count(self) -> int:
            """Kaba parametre tahmini."""
            return self.num_experts * self.embed_dim * self.embed_dim

        def active_parameter_count(self) -> int:
            """Token başına etkin parametre tahmini."""
            return min(self.config.top_k, self.num_experts) * self.embed_dim * self.embed_dim

        def load_balance_report(self) -> Dict[str, Any]:
            """Yük dengesi raporu."""
            usage = list(self._usage_ema)
            mean = sum(usage) / len(usage) if usage else 0.0
            variance = sum((v - mean) ** 2 for v in usage) / len(usage) if usage else 0.0
            return {
                "usage": [round(v, 5) for v in usage],
                "imbalance": round(math.sqrt(variance) / mean, 5) if mean > 0 else 0.0,
                "idle": [i for i, v in enumerate(usage) if v < 0.02],
                "overloaded": [],
                "recommendation": "numpy yedegi - sinirli",
            }


# Not: `Router` ve `Expert` sınıfları yukarıda yalnızca torch varken tanımlanır.
# NumPy yedeğinde bu isimler modül düzeyinde bulunmaz; bu yüzden burada
# koşullu olarak boş bırakılır (bkz. aşağıdaki `if not HAS_TORCH` bloğu).


# ─── Büyüme Politikası ──────────────────────────────────────────────────────

@dataclass
class GrowthSignal:
    """Uzman büyütme/dondurma karar sinyali."""
    action: str = "noop"          # grow | freeze | unfreeze | prune | reset | noop
    expert_id: int = -1
    count: int = 0
    reason: str = ""
    urgency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "expert_id": self.expert_id,
                "count": self.count, "reason": self.reason,
                "urgency": round(self.urgency, 4)}


def decide_growth(
    stats: Dict[str, Any],
    *,
    entropy_threshold: float = 0.8,
    prune_threshold: float = 0.05,
    min_experts: int = 3,
    max_experts: int = 12,
    imbalance_threshold: float = 0.55,
    confidence_threshold: float = 0.25,
) -> GrowthSignal:
    """
    Yönlendirme istatistiklerinden büyüme kararı üretir.

    Karar mantığı:
        1. **Yüksek entropi + düşük güven** → uzmanlar yetmiyor, **büyü**.
           (Model her token'ı eşit olasılıkla dağıtıyor = ayırt edemiyor.)
        2. **Düşük entropi + dengesizlik** → bazı uzmanlar boşta, **dondur**.
        3. **Çok düşük kullanım** → **sıfırla** (yeniden öğrenmeye uygun).
        4. Aksi hâlde **noop**.

    Args:
        stats: `MixtureOfExperts.routing_stats()` çıktısı
        entropy_threshold: Büyüme için entropi eşiği
        prune_threshold: Dondurma için kullanım eşiği
        min_experts: Altında budama yapılmayacak uzman sayısı
        max_experts: Büyüme sınırı
        imbalance_threshold: Dengesizlik eşiği
        confidence_threshold: Yönlendirme güveni alt sınırı

    Returns:
        GrowthSignal
    """
    num_experts = int(stats.get("num_experts", 0) or 0)
    usage = list(stats.get("usage_ema") or stats.get("expert_usage") or [])
    entropy = float(stats.get("expert_entropy", 0.0) or 0.0)
    confidence = float(stats.get("routing_confidence", 1.0) or 1.0)
    can_grow = bool(stats.get("can_grow", num_experts < max_experts))

    if num_experts == 0:
        return GrowthSignal("noop", reason="uzman yok", urgency=0.0)

    # 1. Büyüme: model ayırt edemiyor
    if can_grow and entropy >= entropy_threshold and confidence <= confidence_threshold:
        urgency = min(1.0, (entropy - entropy_threshold) / max(1e-6, 1.0 - entropy_threshold))
        return GrowthSignal(
            "grow", count=1,
            reason=(f"yönlendirme entropisi yüksek ({entropy:.2f} ≥ {entropy_threshold}) "
                    f"ve güven düşük ({confidence:.2f}) — kapasite yetersiz"),
            urgency=urgency,
        )

    if not usage:
        return GrowthSignal("noop", reason="kullanım verisi yok")

    mean_usage = sum(usage) / len(usage)

    # 2. Budama: neredeyse hiç kullanılmayan uzman
    idle = [index for index, value in enumerate(usage) if value < prune_threshold]
    if idle and num_experts > min_experts:
        target = min(idle, key=lambda index: usage[index])
        # Neredeyse sıfır kullanım → kalıcı budama; düşük ama sıfır değil → dondur
        if usage[target] < prune_threshold / 4 and num_experts > min_experts + 1:
            return GrowthSignal(
                "prune", expert_id=target,
                reason=(f"uzman {target} neredeyse hiç kullanılmıyor "
                        f"({usage[target]:.4f} < {prune_threshold / 4:.4f})"),
                urgency=0.5,
            )
        return GrowthSignal(
            "freeze", expert_id=target,
            reason=(f"uzman {target} kullanımı eşiğin altında "
                    f"({usage[target]:.4f} < {prune_threshold})"),
            urgency=0.4,
        )

    # 3. Dengesizlik + düşük entropi → boştaki uzmanı sıfırla
    imbalance = float(stats.get("imbalance", 0.0) or 0.0)
    if not imbalance and mean_usage > 0:
        variance = sum((value - mean_usage) ** 2 for value in usage) / len(usage)
        imbalance = math.sqrt(variance) / mean_usage
    if imbalance >= imbalance_threshold and idle:
        target = min(idle, key=lambda index: usage[index])
        return GrowthSignal(
            "reset", expert_id=target,
            reason=f"yük dengesiz ({imbalance:.2f}) — uzman {target} sıfırlanacak",
            urgency=0.3,
        )

    return GrowthSignal("noop", reason="denge sağlıklı", urgency=0.0)


def estimate_moe_memory(config: MoEConfig, batch_size: int, seq_len: int,
                        bytes_per_param: int = 4) -> Dict[str, float]:
    """
    MoE katmanının bellek kullanımını tahmin eder (MB).

    Hem toplam hem **etkin** (token başına) parametre maliyeti raporlanır —
    büyüme kararlarında kaynak bütçesiyle karşılaştırmak için.
    """
    hidden = int(config.expert_hidden_dim or int(8 * config.embed_dim / 3))
    per_expert = 3 * config.embed_dim * hidden          # SwiGLU: w1, w2, w3
    router = config.num_experts * config.embed_dim
    shared = per_expert if config.shared_expert else 0

    total_params = per_expert * config.num_experts + router + shared
    active_params = per_expert * min(config.top_k, config.num_experts) + router + shared

    # Eğitimde gradyan + optimizer durumu ≈ 4× parametre (Adam)
    training_factor = 4
    return {
        "total_params": float(total_params),
        "active_params": float(active_params),
        "sparsity_ratio": round(active_params / total_params, 4) if total_params else 1.0,
        "weights_mb": round(total_params * bytes_per_param / (1024 ** 2), 3),
        "training_mb": round(total_params * bytes_per_param * training_factor / (1024 ** 2), 3),
        "max_growth_weights_mb": round(
            (per_expert * config.max_experts + config.max_experts * config.embed_dim + shared)
            * bytes_per_param / (1024 ** 2), 3
        ),
    }


__all__ = [
    "MoEConfig",
    "RouterStats",
    "MixtureOfExperts",
    "Expert",
    "Router",
    "GrowthSignal",
    "decide_growth",
    "estimate_moe_memory",
]
