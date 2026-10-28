"""
Aşama 0 — Güvenilir ölçek: doğrulama kayıbı + ezber testi
===========================================================
⚠ Neden bu betik var? (ÖLÇÜLEN EKSİK)
--------------------------------------
```
trainer.py:508  split(validation_ratio=0.1)        ✅ var
trainer.py:891  train(validation=...)              ✅ var
trainer.py:72   eval_every: int = 0                ⚠️ KAPALI
eğitim döngüsünde val_loss hesabı                  ❌ YOK
tools/*.py içinde `validation=` geçen çağrı        ❌ YOK
```

Yani bugüne kadar görülen **tek** kayıp `0,083` = **eğitim kayıbı**,
modelin **3,9 kez gördüğü** veride. Model hiç **sınanmadı**.

Bu betik üç şeyi ölçer
----------------------
```
1. GÖRÜLMÜŞ veri kayıbı     (havuzdan örnek — model 3,9× gördü)
2. DONMUŞ doğrulama kayıbı  (havuzdan ayrılmış dilim — ⚠ HÂLÂ GÖRÜLMÜŞ)
3. YENİ metin kayıbı        (elle yazılmış, hiçbir havuzda YOK)
   ↑ **tek dürüst genelleme ölçüsü**
```

⚠ Kritik ayrım
---------------
Model **tüm havuzu** gördüğü için havuzdan ayrılan dilim **held-out
değildir**. Gerçek genelleme yalnız (3) ile ölçülebilir.
(2)'yi **Aşama 4 için** donduruyoruz: bundan sonraki eğitimde bu dilim
**dışlanacak**, böylece gerçek held-out ölçeği oluşacak.

Ezber testi
-----------
Eğitim örneğinin ilk N token'ı verilir, model **açgözlü** (greedy)
30 token üretir. Gerçek devamla **birebir** eşleşme oranı ölçülür.
Yüksek eşleşme → ezber.

Kullanım
--------
    python tools/dogrulama_kur.py --olc
    python tools/dogrulama_kur.py --olc --ornek 40
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

HAVUZ = Path(".neurocore/egitim_havuzu.jsonl")
DOGRULAMA = Path(".neurocore/dogrulama_kumesi.jsonl")
EGITIM_DILIMI = Path(".neurocore/egitim_dilimi.jsonl")
RAPOR = Path("_dogrulama_olcumu.json")

#: ⚠ Doğrulama oranı — Aşama 4'te bu dilim eğitimden **dışlanacak**
VAL_ORAN = 0.08

#: ⚠ YENİ metinler — hiçbir havuzda YOK (elle yazıldı).
#:    Dürüst genelleme ölçüsü yalnız bunlarla alınır.
YENI_METINLER: List[Dict[str, str]] = [
    {"kategori": "dxgi", "metin": (
        "DXGI_ERROR_DEVICE_HUNG hatası, ekran kartı sürücüsünün belirli "
        "bir süre içinde komutları tamamlayamaması sonucu oluşur. Windows "
        "kurtarma mekanizması TDR, sürücüyü sıfırlar ve uygulama kapanır. "
        "İlk çözüm sürücüyü DDU ile tamamen kaldırıp yeniden kurmaktır. "
        "Overlay yazılımları kapatılmalı, fabrika hız aşırtması geri "
        "alınmalıdır. Kalıcı sorunda TdrDelay değeri artırılabilir."
    )},
    {"kategori": "dxgi", "metin": (
        "The device removed error means the graphics driver stopped "
        "responding. Windows resets the driver after a timeout. To fix "
        "it, remove the old driver with DDU and install a clean version. "
        "Disable overlays and any factory overclock. If the problem "
        "continues, increase the TdrDelay registry value."
    )},
    {"kategori": "blender_uv", "metin": (
        "Blender'da UV açmak için önce dikişleri işaretlemek gerekir. "
        "Nesne seçilir, kenar moduna geçilir ve kesim çizgileri işaretlenir. "
        "Sonra unwrap komutu çalıştırılır. Yüzey parçaları düzleştirilir ve "
        "gerilme azalır. Dikişler yanlış yerden geçerse doku bozulur."
    )},
    {"kategori": "python_liste", "metin": (
        "Python'da bir listeyi sıralamak için iki yol vardır. sorted "
        "fonksiyonu yeni bir liste döndürür, listenin kendisini değiştirmez. "
        "Liste metodunun sort fonksiyonu ise listeyi yerinde sıralar ve None "
        "döndürür. Anahtar fonksiyonu ile sıralama ölçütü belirlenebilir."
    )},
    {"kategori": "godot_sinyal", "metin": (
        "Godot 4 sürümünde sinyal bağlamak için connect metodu kullanılır. "
        "Düğümün sinyali emit_signal ile yayılır. Dört sürümünde ise hedef "
        "nesne ilk parametre olarak verilirdi. Sürüm değişince parametre "
        "sırası değişti ve eski kod bozuldu."
    )},
    {"kategori": "gpu_surucu", "metin": (
        "Ekran kartı sürücüsü neden çöker? En sık sebep kararsız bellek "
        "hızıdır. İkinci sebep güç kaynağının yetersiz kalmasıdır. Üçüncü "
        "sebep aşırı ısınma ve termal kısıtlamadır. Ayrıca üst üste kurulan "
        "sürücü kalıntıları da çökmeye yol açar."
    )},
]


# ══ Veri hazırlığı ═══════════════════════════════════════════════════════

def _kayitlar(yol: Path) -> List[Dict[str, Any]]:
    """JSONL okur."""
    kayit: List[Dict[str, Any]] = []
    if not yol.exists():
        return kayit
    with yol.open(encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir:
                continue
            try:
                d = json.loads(satir)
            except json.JSONDecodeError:
                continue
            metin = d.get("metin") or d.get("text") or ""
            if metin:
                kayit.append(d)
    return kayit


def _belge_anahtari(metin: str) -> str:
    """
    Belge düzeyinde gruplama anahtarı.

    ⚠ Neden? Rastgele örnek bölmesi **sızıntı** yapar: aynı belgenin
    parçaları hem eğitimde hem doğrulamada kalır. İlk satır (çoğu kayıtta
    `# ...` başlığı) belge kimliği olarak kullanılır.
    """
    ilk = metin.strip().split("\n", 1)[0][:120]
    return ilk


def bol(kayitlar: List[Dict[str, Any]],
        oran: float = VAL_ORAN, tohum: int = 42
        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Belge düzeyinde eğitim/doğrulama bölmesi."""
    gruplar: Dict[str, List[Dict[str, Any]]] = {}
    for d in kayitlar:
        gruplar.setdefault(_belge_anahtari(d["metin"]), []).append(d)
    anahtarlar = sorted(gruplar)
    random.Random(tohum).shuffle(anahtarlar)
    kes = max(1, int(len(anahtarlar) * oran))
    val_anahtar = set(anahtarlar[:kes])
    egitim: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for k in anahtarlar:
        (val if k in val_anahtar else egitim).extend(gruplar[k])
    return egitim, val


# ══ Kayıp ölçümü ════════════════════════════════════════════════════════

def kayip_olc(model: Any, metinler: List[str], *, seq_len: int = 256,
              batch: int = 4) -> Dict[str, Any]:
    """
    Ortalama **token başına** çapraz entropi kaybını ölçer.

    ⚠ `torch.no_grad()` + fp32 — Pascal mimarisinde fp16 taşma yapar.
    """
    import torch

    backbone = model.backbone
    tok = model.tokenizer
    cihaz = model._next_device() if hasattr(model, "_next_device") else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eski = backbone.training
    backbone.eval()

    toplam_kayip = 0.0
    toplam_token = 0
    parti = 0
    try:
        with torch.no_grad():
            for i in range(0, len(metinler), batch):
                grup = metinler[i:i + batch]
                kodlu = []
                for m in grup:
                    ids = tok.encode(m, max_length=seq_len)
                    if len(ids) >= 8:
                        kodlu.append(ids[:seq_len])
                if not kodlu:
                    continue
                en_uzun = max(len(x) for x in kodlu)
                dolgu = getattr(tok, "pad_id", 0)
                girdi, etiket = [], []
                for ids in kodlu:
                    satir = list(ids) + [dolgu] * (en_uzun - len(ids))
                    girdi.append(satir)
                    etiket.append([t if t != dolgu else -100 for t in satir])
                it = torch.tensor(girdi, dtype=torch.long).to(cihaz)
                lb = torch.tensor(etiket, dtype=torch.long).to(cihaz)
                mt = (it != dolgu).long().to(cihaz)
                cikti = backbone(it, attention_mask=mt, labels=lb)
                n = int((lb != -100).sum().item())
                if n <= 0:
                    continue
                # ⚠ Kayıp ortalamadır → token sayısıyla geri çarp
                toplam_kayip += float(cikti["loss"].item()) * n
                toplam_token += n
                parti += 1
    finally:
        backbone.train(eski)

    return {
        "kayip": round(toplam_kayip / max(1, toplam_token), 4),
        "token": toplam_token,
        "parti": parti,
        "metin": len(metinler),
    }


# ══ Ezber testi ═════════════════════════════════════════════════════════

def ezber_testi(model: Any, metinler: List[str], *, adet: int = 24,
                onek_token: int = 40, uret_token: int = 24
                ) -> Dict[str, Any]:
    """
    Eğitim metninin ön eki verilip **açgözlü** devam üretilir; gerçek
    devamla birebir eşleşme ölçülür.

    ⚠ Yüksek eşleşme = ezber. Düşük = genelleme/öğrenmemiş.
    """
    import torch

    backbone = model.backbone
    tok = model.tokenizer
    cihaz = model._next_device() if hasattr(model, "_next_device") else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eski = backbone.training
    backbone.eval()

    tam = 0
    kismi = 0
    denenen = 0
    ornekler: List[Dict[str, Any]] = []
    try:
        with torch.no_grad():
            for metin in metinler:
                if denenen >= adet:
                    break
                ids = tok.encode(metin, max_length=256)
                if len(ids) < onek_token + uret_token + 4:
                    continue
                denenen += 1
                onek = ids[:onek_token]
                gercek = ids[onek_token:onek_token + uret_token]
                it = torch.tensor([onek], dtype=torch.long).to(cihaz)
                # ⚠ `generate()` imzasında `do_sample` YOK — greedy için
                # `temperature=0` (transformer.py:451 → argmax). Tekrar
                # cezası da KAPATILMALI, aksi halde ezber ölçümü bozulur.
                uretilen = backbone.generate(
                    it, max_new_tokens=uret_token,
                    temperature=0.0, top_k=0, top_p=1.0,
                    repetition_penalty=1.0,
                )
                yeni = uretilen[0].tolist()[len(onek):]
                # ⚠ Birebir eşleşme (ilk N token)
                esit = sum(1 for a, b in zip(gercek, yeni) if a == b)
                if esit == len(gercek):
                    tam += 1
                if esit >= max(4, len(gercek) // 2):
                    kismi += 1
                if len(ornekler) < 5:
                    ornekler.append({
                        "gercek": tok.decode(gercek)[:70],
                        "uretilen": tok.decode(yeni)[:70],
                        "esit": esit, "toplam": len(gercek),
                    })
    finally:
        backbone.train(eski)

    return {
        "denenen": denenen,
        "birebir": tam,
        "kismi": kismi,
        "birebir_oran": round(tam / max(1, denenen), 3),
        "kismi_oran": round(kismi / max(1, denenen), 3),
        "ornek": ornekler,
    }


# ══ Ana akış ════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="Aşama 0 — doğrulama ölçümü")
    ap.add_argument("--olc", action="store_true", help="Ölçümü çalıştır")
    ap.add_argument("--ornek", type=int, default=60,
                    help="Değerlendirilecek metin sayısı (küme başına)")
    ap.add_argument("--yeniden-bol", action="store_true",
                    help="Doğrulama dilimini yeniden oluştur")
    args = ap.parse_args()

    print("=" * 74)
    print("  AŞAMA 0 — GÜVENİLİR ÖLÇEK")
    print("=" * 74)

    kayitlar = _kayitlar(HAVUZ)
    print(f"\n  havuz       : {len(kayitlar):,} kayıt")

    # ── 1. Donmuş doğrulama dilimi (Aşama 4 için) ──────────────────────
    if DOGRULAMA.exists() and not args.yeniden_bol:
        val = _kayitlar(DOGRULAMA)
        egitim = _kayitlar(EGITIM_DILIMI)
        print(f"  doğrulama   : {len(val):,} kayıt (mevcut dosyadan)")
        print(f"  eğitim      : {len(egitim):,} kayıt (mevcut dosyadan)")
    else:
        egitim, val = bol(kayitlar)
        for yol, veri in ((DOGRULAMA, val), (EGITIM_DILIMI, egitim)):
            with yol.open("w", encoding="utf-8") as f:
                for d in veri:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  doğrulama   : {len(val):,} kayıt → {DOGRULAMA}")
        print(f"  eğitim      : {len(egitim):,} kayıt → {EGITIM_DILIMI}")

    # ── 2. Model ───────────────────────────────────────────────────────
    from neurocore import NeuroCoreEngine

    t0 = time.perf_counter()
    e = NeuroCoreEngine()
    e.initialize()
    m = e.model
    print(f"\n  model       : {m.growth.total_training_steps:,} adım · "
          f"{m.growth.total_tokens_seen:,} token")
    print(f"  yükleme     : {time.perf_counter() - t0:.1f} sn")

    if not args.olc:
        print("\n  (yalnız bölme yapıldı — ölçüm için --olc)")
        e.shutdown()
        return 0

    # ── 3. Üç kümede kayıp ─────────────────────────────────────────────
    rnd = random.Random(7)
    gorulmus = [d["metin"] for d in rnd.sample(
        kayitlar, min(args.ornek, len(kayitlar)))]
    held = [d["metin"] for d in rnd.sample(
        val, min(args.ornek, len(val)))] if val else []
    yeni = [d["metin"] for d in YENI_METINLER]

    print("\n── KAYIP ÖLÇÜMÜ (token başına çapraz entropi) ──")
    sonuc: Dict[str, Any] = {}
    for ad, kume, not_ in (
        ("gorulmus", gorulmus, "havuzdan — model 3,9× gördü"),
        ("dogrulama", held, "havuz dilimi — ⚠ HÂLÂ GÖRÜLMÜŞ"),
        ("yeni", yeni, "elle yazıldı — hiçbir havuzda YOK"),
    ):
        if not kume:
            continue
        t1 = time.perf_counter()
        r = kayip_olc(m, kume)
        r["sure_s"] = round(time.perf_counter() - t1, 1)
        r["not"] = not_
        sonuc[ad] = r
        print(f"  {ad:11} kayıp={r['kayip']:>7} · token={r['token']:>6} "
              f"· metin={r['metin']:>3} · {r['sure_s']:>5} sn")
        print(f"              ({not_})")

    # ── 4. Ezber testi ─────────────────────────────────────────────────
    print("\n── EZBER TESTİ (açgözlü devam · birebir eşleşme) ──")
    ez: Dict[str, Any] = {}
    for ad, kume in (("gorulmus", gorulmus), ("yeni", yeni)):
        if not kume:
            continue
        z = ezber_testi(m, kume, adet=min(16, len(kume)))
        ez[ad] = z
        print(f"  {ad:11} birebir={z['birebir']}/{z['denenen']} "
              f"(%{z['birebir_oran'] * 100:.0f}) · "
              f"kısmi=%{z['kismi_oran'] * 100:.0f}")
        for o in z["ornek"][:2]:
            print(f"      gerçek : {o['gercek']!r}")
            print(f"      üretilen: {o['uretilen']!r}  ({o['esit']}/"
                  f"{o['toplam']} eşit)")

    # ── 5. Yorum ───────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    g = sonuc.get("gorulmus", {}).get("kayip")
    y = sonuc.get("yeni", {}).get("kayip")
    if g is not None and y is not None:
        fark = y - g
        print(f"  görülmüş kayıp : {g}")
        print(f"  YENİ metin     : {y}")
        print(f"  FARK           : {fark:+.4f}")
        if g < 0.5 and fark > 1.0:
            karar = "EZBER — görülmüş veride ~0, yeni metinde yüksek"
        elif fark < 0.5:
            karar = "GENELLİYOR — fark küçük"
        elif g >= 1.0:
            karar = "ÖĞRENMEMİŞ — görülmüş veride bile kayıp yüksek"
        else:
            karar = "KISMİ EZBER"
        print(f"  KARAR          : {karar}")
        sonuc["karar"] = karar
    sonuc["ezber"] = ez
    RAPOR.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    print(f"\n  rapor → {RAPOR}")
    print("=" * 74)
    e.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
