"""
Üretim Kalitesi Ölçümü — eğitim öncesi/sonrası kıyas
======================================================
⚠ Ne ölçer?
-----------
Üç bağımsız ölçüt (tek başına hiçbiri yeterli değil):

```
1. KALİTE KAPISI   : kaç çıktı reddedildi (uydurma kelime oranı)
2. KELİME İSABETİ  : üretilen kelimelerin kaçı GERÇEK Türkçe/İngilizce
3. ANAHTAR İSABETİ : sorunun beklediği teknik terim çıktıda var mı
```

⚠ Ölçüldü (eğitim öncesi, 13.574 adım):
```
kelime isabeti : 16/96 = %16,7
anahtar isabeti: 1/8  (o da tesadüf)
kapı reddi     : 0/8
```

Kullanım
--------
    python tools/uretim_kalitesi.py
    python tools/uretim_kalitesi.py --etiket egitim_sonrasi
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

#: ⚠ Sabit soru seti — her koşuda AYNI (kıyas için)
SORULAR: List[Dict[str, Any]] = [
    {"soru": "DXGI_ERROR_DEVICE_HUNG hatası nasıl düzeltilir?",
     "anahtar": ["sürücü", "driver", "tdr", "gpu", "güncelle"]},
    {"soru": "GPU sürücüsü neden çöker?",
     "anahtar": ["sürücü", "bellek", "ısı", "güç", "vram"]},
    {"soru": "Blender'da UV nasıl açılır?",
     "anahtar": ["unwrap", "dikiş", "seam", "uv", "yüzey"]},
    {"soru": "Python'da liste nasıl sıralanır?",
     "anahtar": ["sorted", "sort", "key", "liste"]},
    {"soru": "TDR nedir?",
     "anahtar": ["timeout", "zaman", "sürücü", "kurtarma", "recovery"]},
    {"soru": "Godot 4'te sinyal nasıl bağlanır?",
     "anahtar": ["connect", "signal", "sinyal", "emit"]},
    {"soru": "Blender'da küp nasıl eklenir?",
     "anahtar": ["cube", "mesh", "primitive", "add", "küp"]},
    {"soru": "Malzeme nasıl oluşturulur?",
     "anahtar": ["material", "node", "bsdf", "malzeme"]},
]

#: ⚠ Gerçek kelime sözlüğü (model tarafındakiyle aynı mantık)
try:
    from neurocore.neural.model import (
        _GERCEK_KELIMELER, _kelime_gercek,
    )
except Exception:                                          # pragma: no cover
    _GERCEK_KELIMELER = frozenset()
    def _kelime_gercek(k: str) -> bool:                    # type: ignore
        return False


def kelime_isabeti(metin: str) -> Dict[str, Any]:
    """Üretilen metindeki gerçek kelime oranı."""
    kelimeler = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]{4,}", metin)
    if not kelimeler:
        return {"kelime": 0, "gercek": 0, "oran": 0.0}
    gercek = sum(1 for k in kelimeler if _kelime_gercek(k))
    return {"kelime": len(kelimeler), "gercek": gercek,
            "oran": round(gercek / len(kelimeler), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Üretim kalitesi ölçümü")
    ap.add_argument("--etiket", default="olcum", help="Rapor etiketi")
    ap.add_argument("--token", type=int, default=110)
    args = ap.parse_args()

    print("=" * 74)
    print(f"  ÜRETİM KALİTESİ — {args.etiket}")
    print("=" * 74)

    from neurocore import NeuroCoreEngine

    e = NeuroCoreEngine()
    e.initialize()
    m = e.model
    print(f"  model: {m.growth.total_training_steps:,} adım · "
          f"{m.growth.total_tokens_seen:,} token")

    sonuc: List[Dict[str, Any]] = []
    toplam_k = 0
    toplam_g = 0
    red = 0
    anahtar_ok = 0

    for s in SORULAR:
        soru = s["soru"]
        try:
            r = m.generate_text(soru, max_new_tokens=args.token,
                                temperature=0.7)
        except Exception as exc:
            print(f"\n  ✗ {soru[:40]}: {type(exc).__name__}")
            continue
        metin = (r.get("text") or "").strip()
        yeni = metin[len(soru):].strip() if metin.startswith(soru) else metin
        k = kelime_isabeti(yeni)
        toplam_k += k["kelime"]
        toplam_g += k["gercek"]
        _red = bool(r.get("quality_rejected"))
        red += _red
        dusuk = yeni.lower()
        bulunan = [a for a in s["anahtar"] if a.lower() in dusuk]
        anahtar_ok += bool(bulunan)
        print(f"\n  {soru}")
        print(f"     çıktı   : {yeni[:120]!r}")
        print(f"     kelime  : {k['gercek']}/{k['kelime']} "
              f"(%{k['oran'] * 100:.0f}) · red={_red}")
        if bulunan:
            print(f"     anahtar : {bulunan}")
        sonuc.append({"soru": soru, "cikti": yeni, **k,
                      "red": _red, "anahtar": bulunan})

    n = len(sonuc)
    kelime_oran = toplam_g / max(1, toplam_k)
    print("\n" + "=" * 74)
    print(f"  KELİME İSABETİ : {toplam_g}/{toplam_k} = "
          f"%{kelime_oran * 100:.1f}")
    print(f"  ANAHTAR İSABETİ: {anahtar_ok}/{n}")
    print(f"  KAPI REDDİ     : {red}/{n}")
    print("=" * 74)

    yol = Path(f"_uretim_kalitesi_{args.etiket}.json")
    yol.write_text(json.dumps({
        "etiket": args.etiket,
        "model_adim": m.growth.total_training_steps,
        "kelime_isabeti": round(kelime_oran, 3),
        "anahtar_isabeti": anahtar_ok, "kapi_reddi": red,
        "toplam": n, "kayit": sonuc,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  rapor → {yol}")
    e.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
