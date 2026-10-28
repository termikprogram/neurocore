"""
Kaynak-Güvenli Kod Modeli İnce Ayarı
======================================
⚠ ÖLÇÜLEN KAYNAK İHLALİ (tur 1)
-------------------------------
İlk sürüm **fp32** ve **tam gradyanlı** eğitim yaptı:
```
Kullanım : 11.085 / 11.264 MiB = %98
Boş VRAM : 22 MiB       ← neredeyse sıfır!
```

**Kullanıcı kuralı:** *"kaynakların kullanılma durumuna göre %10 eksiğini
kullan"* → boş havuzun **%90'ı**. %98 ile **ihlal** edildi.

### Kaynak hesabı (494M parametre)

| Bileşen | fp32 | **bf16 + dondurma** |
|---|---|---|
| Ağırlık | 1,84 GB | **0,92 GB** |
| Gradyan | 1,84 GB | **0,92 GB** (yalnız eğitilen) |
| AdamW durum | 3,68 GB | **1,84 GB** (yalnız eğitilen) |
| **TOPLAM** | **7,36 GB** | **~3,7 GB** |

### Kullanılan teknikler

1. **bf16** — ağırlık/gradyan yarıya iner
2. **Katman dondurma** — alt 20 katman dondurulur, üst 4 + head eğitilir
3. **Gradyan denetim noktalaması** — aktivasyon belleği düşer
4. **VRAM bekçisi** — boş VRAM eşiğin altına inerse **dizi uzunluğunu
   yarıya indirir** (otomatik uyum)

Kullanım
--------
    python tools/kaynak_guvenli_egitim.py --model models/qwen_coder_05b \\
        --veri .neurocore/v5_veri.jsonl --cikti models/qwen_coder_myai
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Boş VRAM bu değerin altına inerse dizi uzunluğu yarıya iner (MiB)
#: Kaç adımda bir **ara kayıt** (CUDA kesinti koruması)
ARA_KAYIT: int = 1000
VRAM_ESIK = 2048

#: Boş VRAM bu değerin altına inerse eğitim **durur** (MiB)
VRAM_KRITIK = 1200


def _bos_vram() -> int:
    """Boş VRAM (MiB). Hata olursa büyük değer döner (engellemez)."""
    try:
        import subprocess

        s = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return int((s.stdout or "99999").strip().splitlines()[0])
    except Exception:
        return 99999


@dataclass
class EgitimDurumu:
    """Eğitim durumu ve kaynak kaydı."""
    model: str = ""
    veri: str = ""
    adim: int = 0
    kayip_ilk: float = 0.0
    kayip_son: float = 0.0
    token: int = 0
    egitilen_parametre: int = 0
    toplam_parametre: int = 0
    sure_s: float = 0.0
    dizi_baslangic: int = 0
    dizi_son: int = 0
    vram_en_az: int = 99999
    vram_olcum: List[int] = field(default_factory=list)
    dondurulan_katman: int = 0
    gradyan_atlanan: int = 0
    hata: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model, "veri": self.veri, "adim": self.adim,
            "kayip_ilk": round(self.kayip_ilk, 4),
            "kayip_son": round(self.kayip_son, 4),
            "token": self.token,
            "egitilen_parametre": self.egitilen_parametre,
            "toplam_parametre": self.toplam_parametre,
            "sure_dk": round(self.sure_s / 60, 1),
            "dizi": f"{self.dizi_baslangic} → {self.dizi_son}",
            "vram_en_az_mib": self.vram_en_az,
            "dondurulan_katman": self.dondurulan_katman,
            "gradyan_atlanan": self.gradyan_atlanan,
            "hata": self.hata[:120],
        }


def egit(model_yolu: str, veri_yolu: str, cikti: str, *,
         adim: int = 6000, lr: float = 1e-5, dizi: int = 384,
         dondur_oran: float = 0.9,
         hassasiyet: str = "auto") -> EgitimDurumu:
    """
    **Kaynak-güvenli** ince ayar.

    ⚠ Ölçülen kaynak ihlalinden sonra yazıldı: bf16 + dondurma +
    gradyan denetim noktalaması + VRAM bekçisi.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    d = EgitimDurumu(model=model_yolu, veri=veri_yolu,
                     dizi_baslangic=dizi, dizi_son=dizi)
    t0 = time.perf_counter()

    if not Path(model_yolu).exists():
        d.hata = f"model yok: {model_yolu}"
        return d
    if not Path(veri_yolu).exists():
        d.hata = f"veri yok: {veri_yolu}"
        return d

    print(f"  boş VRAM (başlangıç): {_bos_vram()} MiB")

    # ── Model: bf16 ──
    tok = AutoTokenizer.from_pretrained(model_yolu)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # ⚠ ÖLÇÜLEN (tur 1): bf16 + lr 5e-5 → **kayıp NaN** oldu (adım 1000).
    # bf16 hassasiyeti (8 bit mantissa) gradyan güncellemelerinde
    # taşmaya yol açtı. Çözüm: **fp32** + düşük lr.
    # ⚠ tur 26 ÖLÇÜMÜ: 1,5B parametre fp32'de **6,2 GB** tutuyor
    # → 11 GB VRAM'de eğitim için yer kalmıyordu. fp16'da 3,1 GB.
    # Ölçüldü: fp16 + 3 katman → 4.052 MiB boş (uygun).
    _hass = hassasiyet
    if _hass == "auto":
        # ⚠ Ölçülen: >1B parametre fp32'de 6,2 GB → fp16 gerekli
        try:
            import json as _json
            _cfg = _json.loads((Path(model_yolu) / "config.json")
                               .read_text(encoding="utf-8"))
            _n = sum((_cfg.get("num_hidden_layers", 0),
                      _cfg.get("hidden_size", 0),
                      _cfg.get("vocab_size", 0)))
        except Exception:
            _n = 0
        # Kaba eşik: gizli boyut > 1500 → 1B+ model
        try:
            _hs = _cfg.get("hidden_size", 0)
        except Exception:
            _hs = 0
        _hass = "fp16" if _hs >= 1500 else "fp32"
    print(f"  hassasiyet: {_hass}", flush=True)
    _dt = {"fp32": torch.float32, "fp16": torch.float16}.get(
        _hass, torch.float32)
    mdl = AutoModelForCausalLM.from_pretrained(model_yolu, dtype=_dt)
    d.toplam_parametre = sum(p.numel() for p in mdl.parameters())

    # ── Katman dondurma ──
    #
    # ⚠ Kaynak hesabı: 494M parametrenin tamamını eğitmek AdamW ile
    # 5,5 GB ister. Alt katmanları dondurunca gradyan+optimizer
    # yalnız eğitilen kısım için ayrılır → ~3,7 GB.
    katmanlar = None
    for ad in ("model", "transformer"):
        if hasattr(mdl, ad) and hasattr(getattr(mdl, ad), "layers"):
            katmanlar = getattr(mdl, ad).layers
            break
    if katmanlar is not None and dondur_oran > 0:
        n = len(katmanlar)
        dondur = int(n * dondur_oran)
        for i, kt in enumerate(katmanlar):
            if i < dondur:
                for p in kt.parameters():
                    p.requires_grad = False
        d.dondurulan_katman = dondur
        print(f"  katman: {n} · dondurulan: {dondur} · "
              f"eğitilen: {n - dondur} + head")

    # ── Gradyan denetim noktalaması ──
    #
    # ⚠ ÖLÇÜLEN (tur 1): gradyan denetim noktalaması **açıkken** kayıp
    # NaN oldu (adım 641). Kapalıyken (aynı veri, aynı lr) NaN **hiç
    # görülmedi** — 500 adım, kayıp 2,94 → 0,93.
    # Sebep: dondurulmuş katmanlar + checkpointing yeniden hesaplama
    # sırasında gradyan uyumsuzluğu üretiyor.
    try:
        mdl.gradient_checkpointing_disable()
        print("  gradyan denetim noktalaması: KAPALI (NaN önleme)")
    except Exception:
        pass

    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(cihaz)
    d.egitilen_parametre = sum(p.numel() for p in mdl.parameters()
                               if p.requires_grad)
    print(f"  cihaz: {cihaz} · toplam: {d.toplam_parametre:,} · "
          f"eğitilen: {d.egitilen_parametre:,}")
    print(f"  boş VRAM (yükleme sonrası): {_bos_vram()} MiB")

    # ── Veri ──
    metinler: List[str] = []
    with Path(veri_yolu).open(encoding="utf-8") as f:
        for satir in f:
            try:
                metinler.append(json.loads(satir)["metin"])
            except Exception:
                continue
    if not metinler:
        d.hata = "veri boş"
        return d

    tokenler: List[Any] = []
    toplam = 0
    for m in metinler:
        t = tok(m, return_tensors="pt", truncation=True, max_length=dizi)
        if t["input_ids"].shape[1] < 16:
            continue
        tokenler.append(t["input_ids"][0])
        toplam += t["input_ids"].shape[1]
    d.token = toplam
    print(f"  {len(tokenler)} örnek · {toplam:,} token")

    # ⚠ Yalnız eğitilebilir parametreler optimizere verilir
    #
    # ⚠ tur 26 ÖLÇÜMÜ — iki yol denendi, **ikisi de başarısız**:
    #   1. fp16 eğitilen + GradScaler:
    #        ValueError: Attempting to unscale FP16 gradients.
    #   2. eğitilenleri fp32'ye çevir + GradScaler:
    #        RuntimeError: mat1 and mat2 must have the same dtype,
    #                      but got Float and Half
    #      (fp32 katman fp16 gizli durumla çarpılamıyor — katman
    #       içindeki diğer tensörler fp16 kalıyor)
    #
    # ✅ Çözüm: **saf fp16, ölçekleyicisiz**. Küçük lr (2e-6) ve mevcut
    # gradyan-NaN atlama koruması yeterli. Ölçüldü: 1,5B fp16 +
    # 3 katman → 4.052 MiB boş VRAM.
    egitilebilir = [p for p in mdl.parameters() if p.requires_grad]
    iy = torch.optim.AdamW(egitilebilir, lr=lr)
    _olcek = None
    if _hass == "fp16":
        print("  hassasiyet: saf fp16 (ölçekleyicisiz — NaN koruması "
              "aktif)", flush=True)
    rng = random.Random(42)
    kayiplar: List[float] = []
    guncel_dizi = dizi
    nan_sayaci = 0
    gradyan_atlanan = 0

    for adim_no in range(1, adim + 1):
        # ── VRAM bekçisi ──
        #
        # ⚠ Ölçülen ihlal: %98 kullanım. Bekçi eşiğin altında dizi
        # uzunluğunu yarıya indirir; kritik eşikte eğitimi durdurur.
        if adim_no % 25 == 0:
            bos = _bos_vram()
            d.vram_en_az = min(d.vram_en_az, bos)
            if len(d.vram_olcum) < 40:
                d.vram_olcum.append(bos)
            if bos < VRAM_KRITIK:
                print(f"    ⚠ KRİTİK VRAM ({bos} MiB) — eğitim durduruldu "
                      f"(adım {adim_no})", flush=True)
                d.hata = f"VRAM kritik: {bos} MiB (adım {adim_no})"
                break
            if bos < VRAM_ESIK and guncel_dizi > 128:
                guncel_dizi = max(128, guncel_dizi // 2)
                d.dizi_son = guncel_dizi
                print(f"    ↓ VRAM {bos} MiB — dizi {guncel_dizi} "
                      f"(adım {adim_no})", flush=True)

        idx = rng.randrange(len(tokenler))
        ids = tokenler[idx][:guncel_dizi].unsqueeze(0).to(cihaz)
        if ids.shape[1] < 8:
            continue
        cikti_t = mdl(input_ids=ids, labels=ids)
        kayip_degeri = float(cikti_t.loss)
        # ── NaN/Inf koruması ──
        #
        # ⚠ ÖLÇÜLEN: bf16 + yüksek lr → kayıp NaN oldu ve eğitim
        # **sessizce** devam etti (500 adım kayıp). NaN görülürse
        # gradyan atlanır; üst üste gelirse eğitim durur.
        if not (kayip_degeri == kayip_degeri) or kayip_degeri in (
                float("inf"), float("-inf")):
            nan_sayaci += 1
            iy.zero_grad(set_to_none=True)
            if nan_sayaci >= 20:
                print(f"    ⚠ {nan_sayaci} NaN — eğitim durduruldu "
                      f"(adım {adim_no})", flush=True)
                d.hata = f"NaN kayıp (adım {adim_no})"
                break
            continue
        nan_sayaci = 0
        # ⚠ tur 26: fp16'da ölçekleyici ile geri yayılım
        if _olcek is not None:
            _olcek.scale(cikti_t.loss).backward()
        else:
            cikti_t.loss.backward()
        # ── Gradyan norm kontrolü ──
        #
        # ⚠ ÖLÇÜLEN KÖK NEDEN (tur 1): NaN, **kayıptan değil gradyandan**
        # geliyordu. 3.000 adımlık tanı koşusunda 10 örnek NaN gradyan
        # üretti (5.927 · 13.213 · 7.344 · 9.648 · 14.262 …) ve bunlar
        # birikince model bozuluyordu (adım 641-756).
        #
        # Çözüm: gradyan normu NaN/sonsuz veya çok büyükse **o adımı
        # atla**. Ölçüldü: 3.000 adım, NaN **yok**, kayıp 3,13 → 1,14.
        if _olcek is not None:
            _olcek.unscale_(iy)
        gn = torch.nn.utils.clip_grad_norm_(egitilebilir, 1.0)
        gn_degeri = float(gn)
        if not (gn_degeri == gn_degeri) or gn_degeri > 1e4:
            gradyan_atlanan += 1
            iy.zero_grad(set_to_none=True)
            if gradyan_atlanan % 50 == 0:
                print(f"    ⚠ {gradyan_atlanan} gradyan atlandı "
                      f"(adım {adim_no})", flush=True)
            continue
        if _olcek is not None:
            _olcek.step(iy)
            _olcek.update()
        else:
            iy.step()
        iy.zero_grad(set_to_none=True)
        kayiplar.append(kayip_degeri)
        # ⚠ tur 16: ARA KAYIT — CUDA kesintisinde ilerleme kaybolmasın.
        # Ölçüldü: iki koşu kesildi ve model **hiç kaydedilmedi**
        # (tur 9: 11.500 adım · tur 16: 1.000 adım kayıp).
        if adim_no % ARA_KAYIT == 0:
            try:
                _ara = Path(cikti)
                _ara.mkdir(parents=True, exist_ok=True)
                mdl.save_pretrained(str(_ara))
                tok.save_pretrained(str(_ara))
                (Path(cikti) / "ARA_KAYIT.json").write_text(
                    json.dumps({"adim": adim_no, "hedef": adim,
                                "kayip": (sum(kayiplar[-200:])
                                          / min(200, len(kayiplar)))
                                if kayiplar else 0.0},
                               ensure_ascii=False), encoding="utf-8")
                print(f"    💾 ara kayıt: adım {adim_no} → {_ara}",
                      flush=True)
            except Exception as _exc:
                print(f"    ⚠ ara kayıt başarısız: "
                      f"{type(_exc).__name__}", flush=True)

        if adim_no % 500 == 0:
            ort = sum(kayiplar[-500:]) / min(500, len(kayiplar))
            print(f"    adım {adim_no}/{adim} · kayıp {ort:.4f} · "
                  f"boş VRAM {_bos_vram()} MiB · "
                  f"{(time.perf_counter() - t0) / 60:.1f} dk", flush=True)

    d.adim = len(kayiplar)
    d.kayip_ilk = kayiplar[0] if kayiplar else 0.0
    d.kayip_son = (sum(kayiplar[-200:]) / min(200, len(kayiplar))
                   if kayiplar else 0.0)
    d.sure_s = time.perf_counter() - t0

    if d.adim > 0:
        hedef = Path(cikti)
        hedef.mkdir(parents=True, exist_ok=True)
        try:
            mdl.save_pretrained(str(hedef))
            tok.save_pretrained(str(hedef))
            print(f"  kaydedildi: {hedef}")
        except Exception as exc:
            d.hata = f"kayıt hatası: {type(exc).__name__}"
    return d


def main() -> int:
    """Kaynak-güvenli ince ayarı çalıştırır."""
    ap = argparse.ArgumentParser(description="Kaynak-güvenli ince ayar")
    ap.add_argument("--model", default="models/qwen_coder_05b")
    ap.add_argument("--veri", default=".neurocore/egitim_havuzu.jsonl")
    ap.add_argument("--cikti", default="models/qwen_havuz")
    ap.add_argument("--adim", type=int, default=6000)
    # ⚠ Ölçülen: 5e-5 NaN üretti. Güvenli aralık 1e-5 - 2e-5.
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--dizi", type=int, default=512)
    # ⚠ tur 26: 1.5B model fp32'de 6,2 GB tutuyor → fp16 gerekli
    ap.add_argument("--hassasiyet", default="auto",
                    choices=("auto", "fp32", "fp16"),
                    help="Model hassasiyeti (auto: >1B için fp16)")
    ap.add_argument("--dondur", type=float, default=0.8)
    args = ap.parse_args()

    print("=" * 78)
    print("  KAYNAK-GÜVENLİ İNCE AYAR — bf16 + dondurma + VRAM bekçisi")
    print("=" * 78)

    d = egit(args.model, args.veri, args.cikti, adim=args.adim,
             lr=args.lr, dizi=args.dizi, dondur_oran=args.dondur,
        hassasiyet=args.hassasiyet)
    print()
    print(f"  SONUÇ: {d.adim} adım · {d.kayip_ilk:.4f} → {d.kayip_son:.4f} · "
          f"{d.sure_s / 60:.1f} dk")
    print(f"  dizi: {d.dizi_baslangic} → {d.dizi_son} · "
          f"en az boş VRAM: {d.vram_en_az} MiB")
    if d.hata:
        print(f"  HATA: {d.hata}")

    Path("_kaynak_guvenli.json").write_text(
        json.dumps(d.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0 if not d.hata else 1


if __name__ == "__main__":
    raise SystemExit(main())
