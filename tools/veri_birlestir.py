"""
Eğitim Verisi Birleştirici — Dengeli ve Sürüm Etiketli
========================================================
⚠ Amaç (tur 8)
--------------
Hedef A tamamlandı: **19,9M token** veri toplandı. Şimdi bu veriyi
**tek eğitim dosyasına** dönüştürmek ve modeli eğitmek (Hedef D).

Neden birleştirici gerekli?
---------------------------
Veri **12 ayrı dosyada** ve oranları çok dengesiz:

    yerel_kod          14,3M token  (%72)  ← baskın
    v6_veri             3,1M token  (%16)
    dokumanlar          2,0M token  (%10)
    python_api          0,6M token  (% 3)

**Sorun:** Dengeli karışım olmazsa model yalnız python öğrenir
(yerel kodun %60'ı python).

Çözüm: **Ağırlıklı örnekleme**
------------------------------
    1. Her kaynağa **hedef oran** ver
    2. Kaynaktan o orana göre **örnekle** (tümünü al veya kısalt)
    3. Karıştır (shuffle)
    4. Tek JSONL yaz

⚠ Neden "tümünü almıyoruz"? 14,3M token yerel kodun **tamamı** eğitimde
4,5 saat sürer (ölçüldü: 500 adım ≈ 1,5 dk → 10.000 adım ≈ 25 dk,
ancak 20M token için ~60.000 adım gerekir). Hedefe uygun **dengeli
örnekleme** daha verimli.

Kullanım
--------
    python tools/veri_birlestir.py --istatistik
    python tools/veri_birlestir.py --birlestir --hedef-token 12000000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Kaynak tanımları — (ad, dosya, hedef oran, kategori)
#:
#: ⚠ Oran seçimi: kod **ağırlıklı** ama doküman ve mantık da yeterli.
KAYNAKLAR: Sequence[Tuple[str, str, float, str]] = (
    # ad                 dosya                            oran   kategori
    ("yerel_kod",        ".neurocore/yerel_kod.jsonl",      0.45, "kod"),
    ("v6_veri",          ".neurocore/v6_veri.jsonl",        0.25, "temel"),
    ("dokuman_cpp",      ".neurocore/dokuman_cpp.jsonl",    0.05, "api"),
    ("dokuman_python313", ".neurocore/dokuman_python313.jsonl", 0.05, "api"),
    ("dokuman_js_mdn",   ".neurocore/dokuman_js_mdn.jsonl", 0.04, "api"),
    ("dokuman_rust",     ".neurocore/dokuman_rust.jsonl",   0.03, "api"),
    ("dokuman_godot",    ".neurocore/dokuman_godot.jsonl",  0.03, "api"),
    ("python_api",       ".neurocore/python_api.jsonl",     0.05, "api"),
    ("kod_mantigi",      ".neurocore/kod_mantigi.jsonl",    0.03, "mantik"),
    ("kod_mantigi_dil",  ".neurocore/kod_mantigi_dil.jsonl", 0.02, "mantik"),
    # ⚠ tur 16: Blender mantık müfredatı — **tamamlama çifti** biçimi.
    # Ölçüm (tur 13): belge biçimi −%9, tamamlama çifti +%9.
    # Kaynak: 43 ders → 58 çift · ~2.746 token.
    ("blender_mantik",   ".neurocore/blender_mantik_tamamlama.jsonl",
     0.03, "mantik"),
)

#: Çıktı dosyası
CIKTI = Path(".neurocore/egitim_havuzu.jsonl")


@dataclass
class KaynakDurum:
    """Bir kaynağın durumu."""
    ad: str = ""
    dosya: str = ""
    oran: float = 0.0
    kategori: str = ""
    mevcut: int = 0
    alinan: int = 0
    token: int = 0
    var: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"ad": self.ad, "oran": self.oran, "kategori": self.kategori,
                "mevcut": self.mevcut, "alinan": self.alinan,
                "token": self.token, "var": self.var}


def _satirlar(yol: Path) -> List[Dict[str, Any]]:
    """JSONL dosyasını okur."""
    out: List[Dict[str, Any]] = []
    if not yol.exists():
        return out
    with yol.open(encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir:
                continue
            try:
                out.append(json.loads(satir))
            except Exception:
                continue
    return out


def istatistik() -> List[KaynakDurum]:
    """Kaynakların mevcut durumunu ölçer."""
    durumlar: List[KaynakDurum] = []
    for ad, dosya, oran, kat in KAYNAKLAR:
        d = KaynakDurum(ad=ad, dosya=dosya, oran=oran, kategori=kat)
        p = Path(dosya)
        if not p.exists():
            d.var = False
            durumlar.append(d)
            continue
        kayitlar = _satirlar(p)
        d.mevcut = len(kayitlar)
        d.token = sum(len(k.get("metin", "")) for k in kayitlar) // 3
        durumlar.append(d)
    return durumlar


def birlestir(hedef_token: int = 12_000_000, *,
              tohum: int = 42) -> Dict[str, Any]:
    """
    Kaynakları **oranlı** biçimde birleştirir.

    ⚠ Yöntem: her kaynaktan `hedef_token × oran` kadar token alınır.
    Kaynakta o kadarı yoksa **tamamı** alınır (oran yeniden dağıtılmaz
    — basitlik ve öngörülebilirlik).
    """
    rng = random.Random(tohum)
    durumlar = istatistik()
    havuz: List[Dict[str, Any]] = []

    for d in durumlar:
        if not d.var:
            continue
        kayitlar = _satirlar(Path(d.dosya))
        if not kayitlar:
            continue
        hedef = int(hedef_token * d.oran)
        # Kaydı karıştır (ilk N almak yanlı olur)
        rng.shuffle(kayitlar)
        alinan = 0
        for k in kayitlar:
            if alinan >= hedef:
                break
            metin = k.get("metin", "")
            if len(metin) < 40:
                continue
            havuz.append({
                "metin": metin if metin.endswith("\n") else metin + "\n",
                "kategori": k.get("kategori", d.kategori),
                "kaynak_havuz": d.ad,
            })
            alinan += len(metin) // 3
        d.alinan = alinan

    rng.shuffle(havuz)
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    karakter = 0
    with CIKTI.open("w", encoding="utf-8") as f:
        for k in havuz:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
            karakter += len(k["metin"])

    return {
        "hedef_token": hedef_token,
        "kayit": len(havuz),
        "karakter": karakter,
        "token": karakter // 3,
        "dosya": str(CIKTI),
        "kaynaklar": [d.to_dict() for d in durumlar],
    }


def main() -> int:
    """Birleştirme işlemlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="Eğitim verisi birleştirici")
    ap.add_argument("--istatistik", action="store_true")
    ap.add_argument("--birlestir", action="store_true")
    ap.add_argument("--hedef-token", type=int, default=12_000_000)
    args = ap.parse_args()

    print("=" * 78)
    print("  EĞİTİM VERİSİ BİRLEŞTİRİCİ — dengeli örnekleme")
    print("=" * 78)

    durumlar = istatistik()
    print(f"\n  {'KAYNAK':22} {'ORAN':>6} {'MEVCUT':>9} {'TOKEN':>12}")
    print(f"  {'-' * 22} {'-' * 6} {'-' * 9} {'-' * 12}")
    toplam_mevcut = 0
    for d in durumlar:
        im = " " if d.var else "✗"
        print(f"  {im}{d.ad:21} {d.oran:>6.2f} {d.mevcut:>9,} "
              f"{d.token:>12,}")
        toplam_mevcut += d.token
    print(f"  {'-' * 22} {'-' * 6} {'-' * 9} {'-' * 12}")
    print(f"  {'TOPLAM':22} {'':>6} {'':>9} {toplam_mevcut:>12,}")

    sonuc: Dict[str, Any] = {"mevcut": [d.to_dict() for d in durumlar]}
    if args.birlestir:
        print(f"\n  Birleştiriliyor (hedef {args.hedef_token:,} token)...")
        r = birlestir(hedef_token=args.hedef_token)
        print(f"\n  {'KAYNAK':22} {'ALINAN':>12}")
        print(f"  {'-' * 22} {'-' * 12}")
        for k in r["kaynaklar"]:
            if k["alinan"]:
                print(f"  {k['ad']:22} {k['alinan']:>12,}")
        print(f"  {'-' * 22} {'-' * 12}")
        print(f"  SONUÇ: {r['kayit']:,} kayıt · {r['token']:,} token")
        print(f"  dosya: {r['dosya']}")
        sonuc["birlestirme"] = r

    Path("_veri_birlestir.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
