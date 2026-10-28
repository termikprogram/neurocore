"""Godot 4 API ağırlıklı eğitim verisi (tur 19).

⚠ ÖLÇÜLEN SORUN (tur 15-18)
---------------------------
Model **Godot 4 API'lerini bilmiyor**: `ambientColor`, `Color3`,
`SetNoiseScale()` uyduruyor → derleme 0-1/4.

Tur 18'de dokümandan **3.089 doğru Godot 4 örneği** (226k token)
üretildi. Bu betik onları **yüksek tekrarla** eğitime sokar.

⚠ Neden yüksek tekrar?
    Godot 4 API verisi toplamın **%4'ü** olur (226k / 5M).
    Model onu **göremez**. 10× tekrar → **%25** pay.
"""
import json
import random
from pathlib import Path

rng = random.Random(42)
HEDEF_TOKEN = 5_000_000
CIKTI = Path(".neurocore/godot4_veri.jsonl")

KAYNAKLAR = [
    (".neurocore/godot4_tamamlama.jsonl", 0.00, "godot4_tm",     3),  # ⚠ 1× → aşırı uyum testi
    (".neurocore/temiz_veri.jsonl",        0.70, "alan_temiz",    1),
    (".neurocore/mantik_tm_verisi.jsonl",  0.20, "alan_mantik_tm", 1),
    (".neurocore/dokuman_godot_g34.jsonl", 0.00, "godot_dokuman", 1),  # ⚠ 1×
    (".neurocore/yerel_kod_g34.jsonl",     0.15, "havuz_yerel",   1),
    (".neurocore/dokuman_cpp.jsonl",       0.05, "havuz_cpp",     1),
]

sayi = karakter = 0
dagilim = {}

with CIKTI.open("w", encoding="utf-8") as f:
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
            for _ in range(max(1, tekrar)):
                for d in kayitlar:
                    m = d["metin"]
                    f.write(json.dumps(
                        {"metin": m if m.endswith("\n") else m + "\n",
                         "kategori": d.get("kategori", "api"),
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
print(f"  GODOT4 payi: %{dagilim.get('godot4_tm', 0) / max(1, tt) * 100:.1f}")
print(f"  dosya: {CIKTI}")
