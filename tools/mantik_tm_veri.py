"""Mantık tamamlama çiftli eğitim verisi (tur 13).

⚠ ÖLÇÜLEN DERS (tur 12): mantık notları `**Kategori:**` biçiminde
"belge" idi ve 6× tekrar performansı **düşürdü** (%67 → %58).

Tur 13 çözümü: `tools/mantik_tamamlama.py` notları **önek/devam
çiftlerine** çevirdi (450 çift, 99k token).

Bu betik: çiftleri **10× tekrarla** (toplamda %15 pay) ve alan
verisiyle karıştır.
"""
import json
import random
from pathlib import Path

rng = random.Random(42)

KAYNAKLAR = [
    (".neurocore/v6_veri.jsonl",            0.35, "alan_v6",          1),
    (".neurocore/titowar_kod.jsonl",        0.08, "alan_titowar",     1),
    (".neurocore/mantik_tamamlama.jsonl",   0.00, "alan_mantik_tm",  10),  # ⚠ 10×
    (".neurocore/yerel_kod.jsonl",          0.30, "havuz_yerel",      1),
    (".neurocore/dokuman_cpp.jsonl",        0.06, "havuz_cpp",        1),
    (".neurocore/dokuman_python313.jsonl",  0.05, "havuz_py",         1),
    (".neurocore/dokuman_godot.jsonl",      0.05, "havuz_godot",      1),
    (".neurocore/dokuman_js_mdn.jsonl",     0.04, "havuz_js",         1),
    (".neurocore/dokuman_rust.jsonl",       0.03, "havuz_rust",       1),
    (".neurocore/kod_ornek.jsonl",          0.02, "havuz_dokuman_kod", 1),
]

HEDEF_TOKEN = 5_000_000
cikti = Path(".neurocore/mantik_tm_verisi.jsonl")

sayi = karakter = 0
dagilim = {}

with cikti.open("w", encoding="utf-8") as f:
    for dosya, oran, ad, tekrar in KAYNAKLAR:
        p = Path(dosya)
        if not p.exists():
            print(f"  {ad:20} YOK")
            continue
        kayitlar = []
        with p.open(encoding="utf-8") as g:
            for s in g:
                try:
                    d = json.loads(s)
                    if len(d.get("metin", "")) >= 40:
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
            print(f"  {ad:20} {n:6,} kayit · ~{alinan:9,} token")
        else:
            n = tok = 0
            for _ in range(tekrar):
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
            print(f"  {ad:20} {n:6,} kayit · ~{tok:9,} token ({tekrar}×)")

tt = int(karakter / 3.5)
print()
print(f"  TOPLAM: {sayi:,} kayit · ~{tt:,} token")
print(f"  MANTIK payi: %{dagilim.get('alan_mantik_tm', 0) / max(1, tt) * 100:.1f}")
print(f"  dosya: {cikti}")
