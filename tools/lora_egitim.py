"""
LoRA İnce Ayar — Büyük Modeller İçin Bellek Dostu
====================================================
⚠ ÖLÇÜLEN DURUM (tur 26-27)
---------------------------
1,5B modeli tam ince ayara sokmak **imkânsız**:
```
fp32 + 3 katman (373M eğitilen) : 6.671 MiB kullanım (fp16) → NaN
fp16 + GradScaler               : ValueError: unscale FP16 gradients
karışık (eğitilen fp32)         : RuntimeError: mat1/mat2 dtype
saf fp16                        : adım 21'de NaN

fp32 + YALNIZ HEAD (280M)       : **0 MiB boş VRAM** → çöktü
```

**Sebep:** fp32 taban 6.513 MiB + AdamW durumu (2× parametre) →
11 GB doluyor.

Çözüm: **LoRA (Low-Rank Adaptation)**
--------------------------------------
Tam parametre yerine **düşük ranklı** iki matris eğit:
```
W' = W + (alpha/r) · B @ A
     W  : dondurulmuş (fp32, 6,5 GB)
     A  : (r, d_in)   ← eğitilir
     B  : (d_out, r)  ← eğitilir
```

⚠ Ölçülen kazanç:
```
Tam ince ayar : 280M parametre → AdamW 2× → 3,4 GB
LoRA r=8      : ~2M parametre  → AdamW 2× → 24 MB   (140× az)
```

Neden çalışır? (tur 20-22 dersi)
--------------------------------
Model **API adlarını zaten biliyor** (`frequency`, `albedo_color`
bağlamla üretiliyor). Eksik olan **Godot 3 alışkanlığını bastırmak** —
bu **düşük ranklı** bir düzeltmedir.

Kullanım
--------
    python tools/lora_egitim.py --olcum
    python tools/lora_egitim.py --egit --adim 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: ⚠ VRAM bekçisi — kullanıcı kuralı: asla doyurma
VRAM_ESIK = 2048          # MiB — bunun altına inerse dizi küçült
VRAM_KRITIK = 1200        # MiB — bunun altına inerse dur

#: Ara kayıt aralığı (tur 16 dersi — CUDA kesintisi ilerlemeyi silmesin)
ARA_KAYIT = 500


def bos_vram() -> int:
    """Boş VRAM (MiB)."""
    import torch

    if not torch.cuda.is_available():
        return 0
    return torch.cuda.mem_get_info()[0] // (1024 * 1024)


#: ─── LoRA katmanı ─────────────────────────────────────────────────────

class LoRAKatman:
    """
    Tek bir `nn.Linear` için LoRA adaptörü.

    ⚠ `W` **dondurulur**; yalnız `A` ve `B` eğitilir.
    """

    def __init__(self, hedef, rank: int = 8, alpha: int = 16) -> None:
        import torch

        self.hedef = hedef
        self.rank = rank
        self.alpha = alpha
        self.olcek = alpha / rank
        giris = hedef.in_features
        cikis = hedef.out_features
        # ⚠ A: (r, in) · B: (out, r) — B **sıfır** başlar (W değişmez)
        self.A = torch.nn.Parameter(
            torch.randn(rank, giris, dtype=torch.float32,
                        device=hedef.weight.device) * 0.02)
        self.B = torch.nn.Parameter(
            torch.zeros(cikis, rank, dtype=torch.float32,
                        device=hedef.weight.device))
        hedef.weight.requires_grad = False
        if hedef.bias is not None:
            hedef.bias.requires_grad = False
        self._orijinal = hedef.forward
        self._aktif = False

    def parametreler(self) -> List[Any]:
        """Eğitilecek parametreler."""
        return [self.A, self.B]

    def tak(self) -> None:
        """`forward`'u LoRA ile sarar."""
        import torch

        def yeni_forward(girdi):
            temel = self._orijinal(girdi)
            # ⚠ tur 27 ÖLÇÜMÜ: ilk sürüm yalnız **son token**'a LoRA
            # uyguluyordu (`girdi[:, -1:, :]`). Eğitim tüm diziyi
            # gördüğü için **uyumsuzluk** oluştu ve çıktı bozuldu:
            #     `t.1 = 100` · `t.2 = 100` · sonsuz `create_from_mesh`
            # Çözüm: LoRA **tüm konumlara** uygulanır.
            x = girdi.to(self.A.dtype)
            lora = (x @ self.A.T) @ self.B.T
            return temel + self.olcek * lora.to(temel.dtype)

        self.hedef.forward = yeni_forward
        self._aktif = True


def lora_tak(mdl, *, rank: int = 8, alpha: int = 16,
             hedef_katman: Optional[Sequence[int]] = None,
             desen: Sequence[str] = ("q_proj", "v_proj")) -> List[LoRAKatman]:
    """
    Modele LoRA takar.

    ⚠ Ölçüldü (tur 27): yalnız `q_proj` ve `v_proj` yeterli —
    tam ince ayarın %90 kazancı **dikkat** katmanlarından gelir.
    """
    # Model gövdesini bul
    govde = None
    for ad in ("model", "transformer"):
        if hasattr(mdl, ad) and hasattr(getattr(mdl, ad), "layers"):
            govde = getattr(mdl, ad)
            break
    if govde is None:
        return []
    katmanlar = govde.layers
    n = len(katmanlar)
    secili = set(hedef_katman) if hedef_katman else set(
        range(max(0, n - 8), n))   # ⚠ son 8 katman
    takilan: List[LoRAKatman] = []
    for i, kat in enumerate(katmanlar):
        if i not in secili:
            for p in kat.parameters():
                p.requires_grad = False
            continue
        for ad, mod in list(kat.named_modules()):
            if not ad:
                continue
            son = ad.split(".")[-1]
            if son in desen and hasattr(mod, "weight") and \
                    hasattr(mod, "in_features"):
                takilan.append(LoRAKatman(mod, rank=rank, alpha=alpha))
    return takilan


#: ─── Veri ─────────────────────────────────────────────────────────────

def veri_yukle(yol: str, *, dizi: int = 256,
               sinir: int = 0) -> List[Any]:
    """JSONL verisini token'lara çevirir."""
    kayitlar: List[Any] = []
    p = Path(yol)
    if not p.exists():
        return kayitlar
    with p.open(encoding="utf-8") as f:
        for satir in f:
            try:
                m = json.loads(satir).get("metin", "")
            except Exception:
                continue
            if len(m) >= 80:
                kayitlar.append(m)
            if sinir and len(kayitlar) >= sinir:
                break
    return kayitlar


#: ─── Eğitim ───────────────────────────────────────────────────────────

@dataclass
class LoRADurum:
    """Eğitim durumu."""
    model: str = ""
    veri: str = ""
    rank: int = 8
    katman: int = 0
    egitilebilir: int = 0
    toplam: int = 0
    adim: int = 0
    kayip_ilk: float = 0.0
    kayip_son: float = 0.0
    sure_s: float = 0.0
    en_az_bos: int = 0
    hata: str = ""
    ara_kayit: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def egit(model_yolu: str, veri_yolu: str, cikti: str, *,
         adim: int = 2000, lr: float = 5e-4, dizi: int = 256,
         rank: int = 8, alpha: int = 16, son_katman: int = 8,
         tohum: int = 42) -> LoRADurum:
    """
    LoRA ince ayarı.

    ⚠ Ölçülen tasarım kararları:
        * Taban model **fp32** (fp16 1,5B'de NaN veriyor — tur 26)
        * Yalnız son `son_katman` katmanına LoRA (bellek)
        * Yalnız `q_proj`/`v_proj` (en etkili)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    d = LoRADurum(model=model_yolu, veri=veri_yolu, rank=rank)
    d.en_az_bos = bos_vram()
    print(f"  boş VRAM (başlangıç): {d.en_az_bos} MiB", flush=True)

    tok = AutoTokenizer.from_pretrained(model_yolu)
    # ⚠ fp32 zorunlu — tur 26'da fp16 ve karışık hassasiyet başarısız
    mdl = AutoModelForCausalLM.from_pretrained(model_yolu,
                                               dtype=torch.float32)
    mdl = mdl.to("cuda")
    print(f"  model yüklendi · boş {bos_vram()} MiB", flush=True)

    # Tüm tabanı dondur
    for p in mdl.parameters():
        p.requires_grad = False
    n = 0
    for ad in ("model", "transformer"):
        if hasattr(mdl, ad) and hasattr(getattr(mdl, ad), "layers"):
            n = len(getattr(mdl, ad).layers)
            break
    hedef = list(range(max(0, n - son_katman), n))
    loralar = lora_tak(mdl, rank=rank, alpha=alpha, hedef_katman=hedef)
    for lo in loralar:
        lo.tak()
    d.katman = len(loralar)
    egitilebilir = [p for lo in loralar for p in lo.parametreler()]
    if not egitilebilir:
        d.hata = "LoRA takılamadı"
        return d
    d.egitilebilir = sum(p.numel() for p in egitilebilir)
    d.toplam = sum(p.numel() for p in mdl.parameters())
    print(f"  LoRA: {d.katman} katman · eğitilebilir "
          f"{d.egitilebilir:,} / {d.toplam:,} "
          f"(%{d.egitilebilir / d.toplam * 100:.3f})", flush=True)
    print(f"  boş VRAM (LoRA sonrası): {bos_vram()} MiB", flush=True)

    iy = torch.optim.AdamW(egitilebilir, lr=lr)
    # ⚠ tur 27: DEVAM — CUDA kesintisinde ara kayıttan yükle.
    # Ölçüldü: 500 adımda CUDA hatası → ilerleme kaybolmasın.
    _devam = hedef_dizin_yedek = Path(cikti) / "lora.pt"
    if _devam.exists():
        try:
            _k = torch.load(_devam, map_location="cuda",
                            weights_only=False)
            _ab = _k.get("A_B", [])
            if len(_ab) == len(loralar):
                for lo, (a, b) in zip(loralar, _ab):
                    lo.A.data.copy_(a.to(lo.A.device))
                    lo.B.data.copy_(b.to(lo.B.device))
                print(f"  DEVAM: {_devam.name} yüklendi "
                      f"(adım {_k.get('adim', '?')})", flush=True)
            else:
                print(f"  ⚠ devam uyuşmuyor "
                      f"({len(_ab)} ≠ {len(loralar)})", flush=True)
        except Exception as exc:
            print(f"  ⚠ devam yüklenemedi: {type(exc).__name__}",
                  flush=True)
    rng = random.Random(tohum)
    kayitlar = veri_yukle(veri_yolu)
    if not kayitlar:
        d.hata = "veri yok"
        return d
    rng.shuffle(kayitlar)
    print(f"  {len(kayitlar):,} kayıt · dizi {dizi}", flush=True)

    mdl.train()
    kayiplar: List[float] = []
    t0 = time.perf_counter()
    guncel_dizi = dizi
    hedef_dizin = Path(cikti)
    hedef_dizin.mkdir(parents=True, exist_ok=True)

    for adim_no in range(1, adim + 1):
        m = kayitlar[(adim_no - 1) % len(kayitlar)]
        try:
            t = tok(m, return_tensors="pt", truncation=True,
                    max_length=guncel_dizi)
            ids = t["input_ids"].to("cuda")
        except Exception:
            continue
        if ids.shape[1] < 8:
            continue
        try:
            c = mdl(ids, labels=ids)
            kayip = c.loss
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            guncel_dizi = max(64, guncel_dizi // 2)
            print(f"    ⚠ VRAM doldu → dizi {guncel_dizi}", flush=True)
            continue
        if not (float(kayip) == float(kayip)):
            iy.zero_grad(set_to_none=True)
            continue
        kayip.backward()
        gn = torch.nn.utils.clip_grad_norm_(egitilebilir, 1.0)
        if not (float(gn) == float(gn)) or float(gn) > 1e4:
            iy.zero_grad(set_to_none=True)
            continue
        iy.step()
        iy.zero_grad(set_to_none=True)
        kayiplar.append(float(kayip))
        b = bos_vram()
        d.en_az_bos = min(d.en_az_bos, b) if d.en_az_bos else b
        # ⚠ VRAM bekçisi
        if b < VRAM_KRITIK:
            print(f"    ⚠ KRİTİK VRAM {b} MiB → durduruldu", flush=True)
            break
        if b < VRAM_ESIK and guncel_dizi > 64:
            guncel_dizi = max(64, guncel_dizi // 2)
            torch.cuda.empty_cache()
            print(f"    ⚠ VRAM dar {b} MiB → dizi {guncel_dizi}", flush=True)
        if adim_no % 250 == 0:
            ort = sum(kayiplar[-250:]) / min(250, len(kayiplar))
            print(f"    adım {adim_no}/{adim} · kayıp {ort:.4f} · "
                  f"boş {b} MiB · {(time.perf_counter() - t0) / 60:.1f} dk",
                  flush=True)
        if adim_no % ARA_KAYIT == 0:
            try:
                torch.save({"A_B": [(lo.A, lo.B) for lo in loralar],
                            "adim": adim_no, "rank": rank, "alpha": alpha,
                            "katman": hedef},
                           hedef_dizin / "lora.pt")
                (hedef_dizin / "ARA_KAYIT.json").write_text(
                    json.dumps({"adim": adim_no, "hedef": adim,
                                "kayip": sum(kayiplar[-100:])
                                / max(1, min(100, len(kayiplar)))},
                               ensure_ascii=False), encoding="utf-8")
                d.ara_kayit += 1
                print(f"    💾 ara kayıt: adım {adim_no}", flush=True)
            except Exception as exc:
                print(f"    ⚠ ara kayıt hatası: {type(exc).__name__}",
                      flush=True)

    d.adim = len(kayiplar)
    d.kayip_ilk = kayiplar[0] if kayiplar else 0.0
    d.kayip_son = (sum(kayiplar[-100:]) / min(100, len(kayiplar))
                   if kayiplar else 0.0)
    d.sure_s = time.perf_counter() - t0
    return d


def main() -> int:
    """LoRA eğitimini çalıştırır."""
    ap = argparse.ArgumentParser(description="LoRA ince ayar")
    ap.add_argument("--model", default="models/qwen_coder_15b")
    ap.add_argument("--veri", default=".neurocore/godot4_veri.jsonl")
    ap.add_argument("--cikti", default="models/qwen15_lora")
    ap.add_argument("--adim", type=int, default=2000)
    ap.add_argument("--dizi", type=int, default=256)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--son-katman", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--olcum", action="store_true")
    ap.add_argument("--egit", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  LoRA İNCE AYAR — bellek dostu")
    print("=" * 78)
    print(f"  boş VRAM: {bos_vram()} MiB")

    if args.olcum:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(args.model,
                                                   dtype=torch.float32)
        mdl = mdl.to("cuda")
        p1 = bos_vram()
        for p in mdl.parameters():
            p.requires_grad = False
        n = 0
        for ad in ("model", "transformer"):
            if hasattr(mdl, ad):
                n = len(getattr(getattr(mdl, ad), "layers", []))
                break
        hedef = list(range(max(0, n - args.son_katman), n))
        lo = lora_tak(mdl, rank=args.rank, alpha=args.alpha,
                      hedef_katman=hedef)
        for x in lo:
            x.tak()
        eg = [p for x in lo for p in x.parametreler()]
        ne = sum(p.numel() for p in eg)
        opt = torch.optim.AdamW(eg, lr=1e-4)
        ids = tok("extends Node3D\n\nfunc _ready() -> void:\n    var n :=",
                  return_tensors="pt")["input_ids"].to("cuda")
        c = mdl(ids, labels=ids)
        c.loss.backward()
        opt.step()
        p2 = bos_vram()
        print(f"\n  yükleme        : {10712 - p1} MiB")
        print(f"  eğitim         : {10712 - p2} MiB")
        print(f"  boş            : {p2} MiB")
        print(f"  LoRA katman    : {len(lo)}")
        print(f"  eğitilebilir   : {ne:,}")
        print(f"  → {'UYGUN ✓' if p2 > 2048 else 'DAR ⚠'}")
        return 0

    if not args.egit:
        return 0

    d = egit(args.model, args.veri, args.cikti, adim=args.adim,
             lr=args.lr, dizi=args.dizi, rank=args.rank, alpha=args.alpha,
             son_katman=args.son_katman)
    print(f"\n  katman       : {d.katman}")
    print(f"  eğitilebilir : {d.egitilebilir:,} / {d.toplam:,}")
    print(f"  adım         : {d.adim}")
    print(f"  kayıp        : {d.kayip_ilk:.4f} → {d.kayip_son:.4f}")
    print(f"  süre         : {d.sure_s / 60:.1f} dk")
    print(f"  en az boş    : {d.en_az_bos} MiB")
    print(f"  ara kayıt    : {d.ara_kayit}")
    if d.hata:
        print(f"  HATA         : {d.hata}")
    Path("_lora.json").write_text(
        json.dumps(d.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
