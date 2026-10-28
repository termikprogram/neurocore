"""Kare dizininden GIF üretir (Pillow'lu herhangi bir Python)."""
from __future__ import annotations
import sys
from pathlib import Path
from PIL import Image

def main() -> int:
    if len(sys.argv) < 3:
        print("kullanım: python tools/gif_yap.py <kare_dizini> <cikti.gif> [ms]")
        return 2
    kaynak = Path(sys.argv[1]); gif = Path(sys.argv[2])
    sure = int(sys.argv[3]) if len(sys.argv) > 3 else 90
    kareler = sorted(kaynak.glob("*.png"))
    if not kareler:
        print(f"  ✗ kare yok: {kaynak}"); return 1
    # ⚠ ÖLÇÜLDÜ: her kareye ayrı `ADAPTIVE` palet vermek GIF'te renkleri
    # bozuyor (pembe/yeşil karışımı). Çözüm: **ilk kareden** tek palet üret,
    # tüm kareleri o palete eşle.
    ilk = Image.open(kareler[0]).convert("RGB")
    palet = ilk.quantize(colors=128, method=Image.MEDIANCUT)
    goruntuler = []
    for f in kareler:
        kare = Image.open(f).convert("RGB").quantize(palette=palet,
                                                     dither=Image.NONE)
        goruntuler.append(kare)
    gif.parent.mkdir(parents=True, exist_ok=True)
    goruntuler[0].save(gif, save_all=True, append_images=goruntuler[1:],
                       duration=sure, loop=0, optimize=True)
    print(f"  ✓ {gif} · {len(goruntuler)} kare · {gif.stat().st_size // 1024} KB")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
