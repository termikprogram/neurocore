"""
Varlıkları Toplu Kataloglar
=============================
⚠ Kullanıcı hedefi (G)
----------------------
*"oluşturulan varlıklar için kategorilere ayırıp bilgilerini de kayıt
altında tutabilir"*

⚠ Neden toplu?
--------------
Her varlık **diskten yeniden açılıp doğrulanır** (Blender ~0,9 sn).
Onlarca varlık tek tek kataloglanamaz — bu betik `Varliklar/` ağacını
tarar, doğrular, kataloğa yazar.

Kullanım
--------
    python tools/varliklari_katalogla.py --tara
    python tools/varliklari_katalogla.py --tara --kaydet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

VARLIK_KOK = Path("Varliklar")


def _varliklari_bul() -> List[Path]:
    """`varlik.blend` içeren klasörleri bulur."""
    out: List[Path] = []
    if not VARLIK_KOK.exists():
        return out
    for f in VARLIK_KOK.rglob("varlik.blend"):
        out.append(f.parent)
    return sorted(out)


def _kalite_bul(ad: str) -> str:
    a = ad.lower()
    if "lowpoly" in a or "low_poly" in a:
        return "lowpoly"
    if "kaliteli" in a or "yuksek" in a:
        return "kaliteli"
    return "orta"


def _ad_bul(klasor: Path) -> str:
    """Klasör adından **varlık adı** üretir (`_01` sayacını temizler)."""
    ad = klasor.name
    for k in ("lowpoly", "kaliteli", "orta"):
        ad = ad.replace(f"_{k}", "")
    # Sondaki `_01` sayacını temizle
    parcalar = ad.split("_")
    if len(parcalar) > 1 and parcalar[-1].isdigit() and len(parcalar[-1]) == 2:
        parcalar = parcalar[:-1]
    return "_".join(parcalar) or klasor.name


def main() -> int:
    """Toplu kataloglama."""
    ap = argparse.ArgumentParser(description="Toplu varlık kataloglama")
    ap.add_argument("--tara", action="store_true")
    ap.add_argument("--kaydet", action="store_true")
    # ⚠ tur 17: mevcut kayıtların meta verisini **yenile**
    # (önizleme, doku, animasyon sonradan eklenmiş olabilir)
    ap.add_argument("--guncelle", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  TOPLU VARLIK KATALOGLAMA")
    print("=" * 78)

    klasorler = _varliklari_bul()
    print(f"  bulunan varlık: {len(klasorler)}")
    if not klasorler:
        return 0

    # Zaten katalogda olanları atla
    from tools.varlik_katalogu import listele

    mevcut = {(x.get("blend") or "") for x in listele()}

    from tools.varlik_kaydet import dogrula, kataloga_yaz

    sonuc: Dict[str, Any] = {"toplam": len(klasorler), "yeni": 0,
                             "atlanan": 0, "detay": []}
    for k in klasorler:
        blend = str((k / "varlik.blend").resolve())
        # ⚠ Aynı dosya yolu zaten katalogda → ATLA (idempotent)
        # ⚠ `--guncelle` verilmişse ATLAMA, meta veriyi yenile
        if not args.guncelle and (
                blend in mevcut or any(
                    (x.get("blend") or "").endswith(
                        blend.split("Varliklar")[-1])
                    for x in listele())):
            print(f"  · {k.name:34} zaten katalogda")
            sonuc["atlanan"] += 1
            continue
        if blend in mevcut:
            print(f"  · {k.name:34} zaten katalogda")
            sonuc["atlanan"] += 1
            continue
        # ⚠ Kategori = klasörün üst adı (Varliklar/<kategori>/<ad>)
        kategori = k.parent.name if k.parent != VARLIK_KOK else "esya"
        ad = _ad_bul(k)
        kalite = _kalite_bul(k.name)
        d = dogrula(k)
        if "hata" in d:
            print(f"  ✗ {k.name:34} {d['hata'][:44]}")
            sonuc["detay"].append({"ad": ad, "hata": d["hata"]})
            continue
        me = d.get("mesh", {})
        ar = d.get("armature", {})
        an = d.get("animasyon", {})
        bayrak = ("A" if ar else "·") + ("N" if an else "·") + \
                 ("U" if me.get("uv_katman") else "·")
        print(f"  ✓ [{bayrak}] {k.name:28} v={me.get('vertex',0):>6} "
              f"y={me.get('yuz',0):>6} kemik={ar.get('kemik_sayisi',0):>2}")
        if args.kaydet:
            # ⚠ Güncelleme modunda eski kaydı sil (id artmasın)
            if args.guncelle:
                from tools.varlik_katalogu import (
                    _katalog_oku, _katalog_yaz)
                _kk = _katalog_oku()
                _sil = [x for x, v2 in _kk.get("varliklar", {}).items()
                        if (v2.get("blend") or "").endswith(
                            blend.split("Varliklar")[-1])]
                for _s in _sil:
                    _kk["varliklar"].pop(_s, None)
                _katalog_yaz(_kk)
            vid = kataloga_yaz(k, kategori=kategori, ad=ad, kalite=kalite)
            print(f"        → {vid}")
        sonuc["yeni"] += 1
        sonuc["detay"].append({"ad": ad, "kategori": kategori,
                               "kalite": kalite,
                               "vertex": me.get("vertex", 0),
                               "yuz": me.get("yuz", 0),
                               "kemik": ar.get("kemik_sayisi", 0),
                               "animasyon": bool(an)})

    print(f"\n  yeni: {sonuc['yeni']} · atlanan: {sonuc['atlanan']}")
    if not args.kaydet:
        print("  ⚠ kaydetmek için --kaydet ekleyin")

    Path("_kataloglama.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
