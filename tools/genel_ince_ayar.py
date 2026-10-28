"""
Genel Kod Modeli İnce Ayarı — Qwen-Coder Tabanlı
==================================================
⚠ Neden bu araç (tur 41)
------------------------
`kod_iyilestir.py` **SmolLM2-135M**'e özeldi. Ölçüm gösterdi ki
**Qwen2.5-Coder-0.5B** (kod için 92 dilde eğitilmiş) daha iyi:

```
Qwen-Coder-0.5B  : 26/40 (%65) · prose 1/40 · orta %75 · zor %70
SmolLM2-135M(v5) : 25/40 (%62) · prose 4/40 · orta %58 · zor %70
```

**Kritik:** `orta` katman **%75 vs %58** — Qwen'de **kodlama mantığı**
belirgin şekilde güçlü. Ama `kolay` katmanda **%60 vs %60** (temel API
bilgisi eşit).

**Plan:** Qwen tabanını **kendi verimizle** ince ayar → kodlama mantığı
(Qwen'den) + kendi alan bilgimiz (kasadan) birleşir.

Kullanım
--------
    python tools/genel_ince_ayar.py --model models/qwen_coder_05b \\
        --veri .neurocore/v5_veri.jsonl --adim 8000
    python tools/genel_ince_ayar.py --kiyasla
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)


@dataclass
class EgitimOzeti:
    """İnce ayar özeti."""
    model: str = ""
    veri: str = ""
    adim: int = 0
    kayip_ilk: float = 0.0
    kayip_son: float = 0.0
    token: int = 0
    sure_s: float = 0.0
    cihaz: str = ""
    parametre: int = 0
    hata: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model, "veri": self.veri, "adim": self.adim,
            "kayip_ilk": round(self.kayip_ilk, 4),
            "kayip_son": round(self.kayip_son, 4),
            "token": self.token, "sure_dk": round(self.sure_s / 60, 1),
            "cihaz": self.cihaz, "parametre": self.parametre,
            "hata": self.hata[:120],
        }


def ince_ayar(model_yolu: str, veri_yolu: str, cikti: str, *,
              adim: int = 8000, lr: float = 3e-5,
              dizi: int = 512) -> EgitimOzeti:
    """
    Bir kod modelini kendi verimizle ince ayar yapar.

    ⚠ Ölçülen ayarlar:
        * `lr=3e-5` — ön-eğitimli ağırlıkları bozmaz
        * `dizi=512` — VRAM dengesi (0.5B model + 512 token ≈ 2,5 GB)
        * Parti=1 + rastgele örnek — basit ve ölçüldü
    """
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    s = EgitimOzeti(model=model_yolu, veri=veri_yolu)
    t0 = time.perf_counter()

    if not Path(model_yolu).exists():
        s.hata = f"model yok: {model_yolu}"
        return s
    if not Path(veri_yolu).exists():
        s.hata = f"veri yok: {veri_yolu}"
        return s

    print(f"  model yükleniyor: {model_yolu}")
    tok = AutoTokenizer.from_pretrained(model_yolu)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_yolu, torch_dtype=torch.float32)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(cihaz)
    mdl.train()
    s.cihaz = cihaz
    s.parametre = sum(p.numel() for p in mdl.parameters())
    print(f"  cihaz: {cihaz} · parametre: {s.parametre:,}")

    # ── Veri ──
    metinler: List[str] = []
    with Path(veri_yolu).open(encoding="utf-8") as f:
        for satir in f:
            try:
                metinler.append(json.loads(satir)["metin"])
            except Exception:
                continue
    if not metinler:
        s.hata = "veri boş"
        return s

    tokenler: List[Any] = []
    toplam = 0
    for m in metinler:
        t = tok(m, return_tensors="pt", truncation=True, max_length=dizi)
        if t["input_ids"].shape[1] < 16:
            continue
        tokenler.append(t["input_ids"][0])
        toplam += t["input_ids"].shape[1]
    s.token = toplam
    print(f"  {len(tokenler)} örnek · {toplam:,} token")

    iy = torch.optim.AdamW(mdl.parameters(), lr=lr)
    rng = random.Random(42)
    kayiplar: List[float] = []
    for adim_no in range(1, adim + 1):
        idx = rng.randrange(len(tokenler))
        ids = tokenler[idx][:dizi].unsqueeze(0).to(cihaz)
        if ids.shape[1] < 8:
            continue
        cikti_t = mdl(input_ids=ids, labels=ids)
        cikti_t.loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        iy.step()
        iy.zero_grad(set_to_none=True)
        kayiplar.append(float(cikti_t.loss))
        if adim_no % 500 == 0:
            ort = sum(kayiplar[-500:]) / min(500, len(kayiplar))
            print(f"    adım {adim_no}/{adim} · kayıp {ort:.4f} · "
                  f"{(time.perf_counter() - t0) / 60:.1f} dk", flush=True)

    s.adim = len(kayiplar)
    s.kayip_ilk = kayiplar[0] if kayiplar else 0.0
    s.kayip_son = (sum(kayiplar[-200:]) / min(200, len(kayiplar))
                   if kayiplar else 0.0)
    s.sure_s = time.perf_counter() - t0

    hedef = Path(cikti)
    hedef.mkdir(parents=True, exist_ok=True)
    mdl.save_pretrained(str(hedef))
    tok.save_pretrained(str(hedef))
    print(f"  kaydedildi: {hedef}")
    return s


def main() -> int:
    """İnce ayarı çalıştırır."""
    ap = argparse.ArgumentParser(description="Genel kod modeli ince ayarı")
    ap.add_argument("--model", default="models/qwen_coder_05b")
    ap.add_argument("--veri", default=".neurocore/v5_veri.jsonl")
    ap.add_argument("--cikti", default="models/qwen_coder_myai")
    ap.add_argument("--adim", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--dizi", type=int, default=512)
    args = ap.parse_args()

    print("=" * 78)
    print("  GENEL KOD MODELİ İNCE AYARI — Qwen-Coder tabanlı")
    print("=" * 78)

    s = ince_ayar(args.model, args.veri, args.cikti, adim=args.adim,
                  lr=args.lr, dizi=args.dizi)
    print()
    print(f"  SONUÇ: {s.adim} adım · {s.kayip_ilk:.4f} → {s.kayip_son:.4f} · "
          f"{s.sure_s / 60:.1f} dk")
    if s.hata:
        print(f"  HATA: {s.hata}")

    Path("_genel_ince_ayar.json").write_text(
        json.dumps(s.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0 if not s.hata else 1


if __name__ == "__main__":
    raise SystemExit(main())
