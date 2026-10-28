"""Temizlenmiş (Godot 3'süz) eğitim verisi kurar (tur 16).

⚠ ÖLÇÜLEN SORUN (tur 15-16)
---------------------------
Uçtan uca testte model **Godot 3 API'lerini uydurdu** →
`Parse Error` → derleme 0/3.

Tarama sonucu (tur 16):
    titowar_kod  : %35,5 kirli   ← Godot 3 projesi
    v6_veri      : %12,9 kirli
    karisim      : %11,9 kirli
    mantik_agr   : % 9,2 kirli
    mantik_tm    : % 7,4 kirli
    yerel_kod    : % 6,2 kirli

En sık desenler:
    getter çağrısı (`.get_x()`)  : 10.975
    setter çağrısı (`.set_x()`)  :  5.460
    Camera → Camera3D            :    268
    export/onready → @export     :    254

Bu betik: kirli örnekleri **atarak** temiz veri seti kurar.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.godot3_temizle import G3_DESEN  # noqa: E402

rng = random.Random(42)

HEDEF_TOKEN = 5_000_000
CIKTI = Path(".neurocore/temiz_veri.jsonl")

# (dosya, oran, ad, tekrar)
KAYNAKLAR = [
    (".neurocore/v6_veri.jsonl",           0.40, "alan_v6",        1),
    (".neurocore/titowar_kod_g4.jsonl",    0.08, "alan_titowar",   1),
    (".neurocore/mantik_tm_verisi.jsonl",  0.00, "alan_mantik_tm", 0),
    (".neurocore/yerel_kod.jsonl",         0.32, "havuz_yerel",    1),
    (".neurocore/dokuman_cpp.jsonl",       0.06, "havuz_cpp",      1),
    (".neurocore/dokuman_python313.jsonl", 0.05, "havuz_py",       1),
    (".neurocore/dokuman_godot.jsonl",     0.05, "havuz_godot",    1),
    (".neurocore/dokuman_js_mdn.jsonl",    0.04, "havuz_js",       1),
]


def kirli(metin: str) -> bool:
    """Godot 3 kalıntısı var mı?"""
    return any(re.search(d, metin) for d, _ in G3_DESEN)


def main() -> int:
    """Temiz veri setini kurar."""
    sayi = karakter = 0
    atilan = 0
    dagilim = {}
    with CIKTI.open("w", encoding="utf-8") as f:
        for dosya, oran, ad, tekrar in KAYNAKLAR:
            p = Path(dosya)
            # ⚠ `_g4` sürümü varsa onu kullan (zaten temizlenmiş)
            # ⚠ tur 17-18: temizlenmiş (`_g4`) ve dönüştürülmüş (`_g34`)
            # sürümleri tercih et — kaynak dosyaya hiç düşme.
            for ek in ("_g4_g34", "_g34", "_g4"):
                aday = Path(dosya.replace(".jsonl", ek + ".jsonl"))
                if aday.exists():
                    p = aday
                    break
            if not p.exists():
                print(f"  {ad:18} YOK")
                continue
            kayitlar = []
            with p.open(encoding="utf-8") as g:
                for s in g:
                    try:
                        d = json.loads(s)
                        m = d.get("metin", "")
                        if len(m) < 40:
                            continue
                        # ⚠ Çift denetim: kirli örnek **asla** girmez
                        if kirli(m):
                            atilan += 1
                            continue
                        kayitlar.append(d)
                    except Exception:
                        pass
            rng.shuffle(kayitlar)
            if oran > 0:
                hedef = int(HEDEF_TOKEN * oran)
                alinan = n = 0
                for d in kayitlar:
                    if alinan >= hedef:
                        break
                    m = d["metin"]
                    f.write(json.dumps(
                        {"metin": m if m.endswith("\n") else m + "\n",
                         "kategori": d.get("kategori", "kod"),
                         "kaynak": ad}, ensure_ascii=False) + "\n")
                    alinan += len(m) // 3
                    n += 1
                sayi += n
                karakter += alinan * 3
                dagilim[ad] = alinan
                print(f"  {ad:18} {n:6,} kayit · ~{alinan:9,} token")
            else:
                n = tok = 0
                # ⚠ `tekrar=0` → 1× (dosya zaten çoklu tekrar içeriyor)
                for _ in range(max(1, tekrar)):
                    for d in kayitlar:
                        m = d["metin"]
                        f.write(json.dumps(
                            {"metin": m if m.endswith("\n") else m + "\n",
                             "kategori": "mantik_tamamlama",
                             "kaynak": ad}, ensure_ascii=False) + "\n")
                        n += 1
                        tok += len(m) // 3
                sayi += n
                karakter += tok * 3
                dagilim[ad] = tok
                print(f"  {ad:18} {n:6,} kayit · ~{tok:9,} token ({tekrar}×)")

    tt = int(karakter / 3.5)
    print()
    print(f"  TOPLAM: {sayi:,} kayit · ~{tt:,} token")
    print(f"  ATILAN (Godot 3): {atilan:,} örnek")
    print(f"  dosya: {CIKTI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
