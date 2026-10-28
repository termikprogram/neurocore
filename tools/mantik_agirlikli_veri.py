"""Mantık ağırlıklı eğitim verisi kurar (tur 12).

⚠ Ölçülen durum (tur 10): bağımsız benchmark'ta `mantik` kategorisi
**%50** — en zayıf halka. Mantık notları 211 adet ama toplam havuzda
yalnız **36k token** (%0,2). Model bunları **göremiyor**.

Çözüm: mantık notlarını **6× tekrarla** → toplam havuzda %1,2 pay.
Ayrıca alan verisi (Godot/Blender) baskın tutulur.
"""
import json
import random
from pathlib import Path

rng = random.Random(42)

# (dosya, oran, ad, tekrar)
KAYNAKLAR = [
    (".neurocore/v6_veri.jsonl",       0.40, "alan_v6",       1),
    (".neurocore/titowar_kod.jsonl",   0.10, "alan_titowar",  1),
    (".neurocore/mantik_tum.jsonl",    0.00, "alan_mantik",   6),   # ⚠ 6×
    (".neurocore/yerel_kod.jsonl",     0.30, "havuz_yerel",   1),
    (".neurocore/dokuman_cpp.jsonl",   0.06, "havuz_cpp",     1),
    (".neurocore/dokuman_python313.jsonl", 0.05, "havuz_py",  1),
    (".neurocore/dokuman_godot.jsonl", 0.04, "havuz_godot",   1),
    (".neurocore/dokuman_js_mdn.jsonl", 0.03, "havuz_js",     1),
    (".neurocore/dokuman_rust.jsonl",  0.02, "havuz_rust",    1),
]

HEDEF_TOKEN = 5_000_000
cikti = Path(".neurocore/mantik_agirlikli.jsonl")

sayi = 0
karakter = 0
dagilim = {}

with cikti.open("w", encoding="utf-8") as f:
    for dosya, oran, ad, tekrar in KAYNAKLAR:
        p = Path(dosya)
        if not p.exists():
            print(f"  {ad:18} YOK")
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
            hedef_token = int(HEDEF_TOKEN * oran)
            alinan = 0
            n = 0
            for d in kayitlar:
                if alinan >= hedef_token:
                    break
                m = d["metin"]
                f.write(json.dumps({"metin": m if m.endswith("\n") else m + "\n",
                                    "kategori": d.get("kategori", "kod"),
                                    "kaynak": ad}, ensure_ascii=False) + "\n")
                alinan += len(m) // 3
                n += 1
            sayi += n
            karakter += alinan * 3
            dagilim[ad] = alinan
            print(f"  {ad:18} {n:6,} kayit · ~{alinan:9,} token")
        else:
            # ⚠ Tekrarlı kaynak (mantık) — tümü alınır, `tekrar` kez
            n = 0
            tok = 0
            for _ in range(tekrar):
                for d in kayitlar:
                    m = d["metin"]
                    f.write(json.dumps({"metin": m if m.endswith("\n") else m + "\n",
                                        "kategori": "mantik",
                                        "kaynak": ad}, ensure_ascii=False) + "\n")
                    n += 1
                    tok += len(m) // 3
            sayi += n
            karakter += tok * 3
            dagilim[ad] = tok
            print(f"  {ad:18} {n:6,} kayit · ~{tok:9,} token ({tekrar}×)")

toplam_token = int(karakter / 3.5)
print()
print(f"  TOPLAM: {sayi:,} kayit · ~{toplam_token:,} token")
print(f"  MANTIK payi: %{dagilim.get('alan_mantik', 0) / max(1, toplam_token) * 100:.1f}")
print(f"  dosya: {cikti}")
