"""
Aşama 0d — Not başına kayıp dağılımı (ezber kanıtı)
====================================================
⚠ Soru: Model kasa notlarının bir kısmını **ezberlemiş** mi?

Yöntem: her not için ayrı kayıp ölçülür, dağılım raporlanır.
    · Bir grup notta kayıp ≈ 0,08 → EZBER (0,083 buradan geliyor)
    · Hepsi ≈ 1,4               → eğitim verisi bugünküyle aynı değil
    · Hepsi yüksek              → hiç öğrenmemiş

Kullanım
--------
    python tools/not_kaybi.py --adet 240
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def main() -> int:
    ap = argparse.ArgumentParser(description="Not başına kayıp dağılımı")
    ap.add_argument("--adet", type=int, default=240)
    args = ap.parse_args()

    print("=" * 74)
    print("  AŞAMA 0d — NOT BAŞINA KAYIP DAĞILIMI")
    print("=" * 74)

    from neurocore import NeuroCoreEngine

    e = NeuroCoreEngine()
    e.initialize()
    m = e.model

    katman = getattr(e, "knowledge", None)
    vault = getattr(katman, "vault", None) if katman is not None else None
    if vault is None:
        print("  ✗ kasa yok")
        e.shutdown()
        return 1

    sys.path.insert(0, str(_KOK / "tools"))
    from tools.dogrulama_kur import kayip_olc

    notlar: List[Any] = []
    for n in vault.iter_notes(limit=args.adet):
        if n.body and n.word_count >= 10:
            notlar.append(n)
    print(f"\n  not sayısı: {len(notlar)}")

    # ⚠ Her notu AYRI ölç (tek metinlik küme)
    kayitlar: List[Dict[str, Any]] = []
    for i, n in enumerate(notlar):
        metin = f"# {n.title}\n{n.body}"
        r = kayip_olc(m, [metin])
        if r["token"] < 8:
            continue
        kayitlar.append({"baslik": n.title, "kayip": r["kayip"],
                         "token": r["token"],
                         "kategori": str(getattr(n, "category", "") or "")})
        if (i + 1) % 40 == 0:
            print(f"    … {i + 1}/{len(notlar)}")

    if not kayitlar:
        print("  ✗ ölçüm yok")
        e.shutdown()
        return 1

    degerler = [k["kayip"] for k in kayitlar]
    degerler.sort()
    n = len(degerler)
    print("\n── DAĞILIM ──")
    print(f"  ölçülen    : {n} not")
    print(f"  en düşük   : {degerler[0]:.4f}")
    print(f"  %10 dilim  : {degerler[max(0, n // 10)]:.4f}")
    print(f"  medyan     : {statistics.median(degerler):.4f}")
    print(f"  %90 dilim  : {degerler[min(n - 1, 9 * n // 10)]:.4f}")
    print(f"  en yüksek  : {degerler[-1]:.4f}")
    print(f"  ortalama   : {statistics.mean(degerler):.4f}")

    ezber = [k for k in kayitlar if k["kayip"] < 0.15]
    print(f"\n  kayıp < 0,15 (EZBER) : {len(ezber)}/{n} "
          f"(%{100 * len(ezber) // n})")
    if ezber:
        print("  ezberlenmiş örnekler:")
        for k in sorted(ezber, key=lambda x: x["kayip"])[:6]:
            print(f"    {k['kayip']:.4f} · {k['baslik'][:52]}")

    print("\n" + "=" * 74)
    if len(ezber) >= max(3, n // 20):
        karar = "EZBER VAR — bir grup not 0,08 civarında"
    elif statistics.median(degerler) > 1.0:
        karar = "EZBER YOK — medyan yüksek, eğitim verisi değişmiş olabilir"
    else:
        karar = "KISMİ — dağılım geniş"
    print(f"  KARAR: {karar}")
    print("=" * 74)
    Path("_not_kaybi.json").write_text(
        json.dumps({"karar": karar, "medyan": statistics.median(degerler),
                    "ezber": len(ezber), "toplam": n,
                    "kayitlar": kayitlar}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("  rapor → _not_kaybi.json")
    e.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
