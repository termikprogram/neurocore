"""
Uçtan Uca Araştırma → Kod Üretimi Testi (tur 15, Hedef C kanıtı)
==================================================================
⚠ Amaç
-------
Kullanıcı hedefi (C): *"istediği dilin/kütüphanenin güncel sürümünü ve
API'sini internetten bulup okuyabilsin"*

Tur 14'te araştırma araçları myAi'ye **bağlandı** (20 araç).
Bu test **kanıtlar** ki zincir çalışıyor:

```
1. HEDEF belirle        (dil + kütüphane + fonksiyon)
2. ARAŞTIR              (doc.api_imzasi_al → internetten imza)
3. KOD ÜRET             (myAi modeli araştırılan API ile kod yazar)
4. DERLE                (gerçek Godot/Python doğrular)
5. RAPORLA              (hangi API kullanıldı, derlendi mi)
```

⚠ Neden bu test önemli?
-----------------------
Araç kayıtlı olması **yetmez** — modelin o API'yi **kullanabilmesi**
gerekir. "Araç var" ile "işe yarıyor" arasındaki farkı ölçer.

Kullanım
--------
    python tools/arastirma_kod_testi.py --liste
    python tools/arastirma_kod_testi.py --calistir
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Godot konsol yolu (derleme doğrulaması için)
GODOT = Path(r"E:\AppsTools\Godot_v4.8-dev7_win64"
             r"\Godot_v4.8-dev7_win64_console.exe")

#: Test görevleri — (dil, kütüphane, aranacak API, görev açıklaması)
GOREVLER: Sequence[Tuple[str, str, str, str]] = (
    ("gdscript", "FastNoiseLite", "frequency",
     "FastNoiseLite ile arazi yüksekliği üret"),
    ("gdscript", "CharacterBody3D", "move_and_slide",
     "CharacterBody3D ile yerçekimi ve hareket"),
    ("gdscript", "StandardMaterial3D", "albedo_color",
     "StandardMaterial3D ile renkli malzeme"),
    ("gdscript", "Tween", "tween_property",
     "Tween ile yumuşak geçiş animasyonu"),
    ("python", "pathlib", "read_text",
     "pathlib ile dosya okuma"),
)


@dataclass
class TestSonuc:
    """Tek test sonucu."""
    dil: str = ""
    kutuphane: str = ""
    api: str = ""
    gorev: str = ""
    arastirildi: bool = False
    imza_sayisi: int = 0
    aranan_api_goruldu: bool = False
    kod_uretildi: bool = False
    kod: str = ""
    derlendi: bool = False
    hata: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "kod"}


def _arastir(dil: str, kutuphane: str, api: str) -> Tuple[bool, int, bool]:
    """`doc.api_imzasi_al` aracını kullanır."""
    try:
        from neurocore.agent.coding import CodingAgent

        a = CodingAgent()
        r = a.api_imzasi_al(dil=dil, fonksiyon=f"{kutuphane}.{api}")
        imzalar = r.get("imzalar", [])
        metin = json.dumps(r, ensure_ascii=False)
        return (bool(r.get("bulundu")), len(imzalar),
                api.lower() in metin.lower())
    except Exception as exc:
        print(f"    araştırma hatası: {type(exc).__name__}: {str(exc)[:60]}")
        return (False, 0, False)


def _kod_uret(dil: str, kutuphane: str, api: str, gorev: str,
              imzalar: Sequence[str]) -> str:
    """
    myAi modeliyle kod üretir — **araştırılan API bağlamıyla**.

    ⚠ Kritik: imzalar prompt'a **enjekte edilir** → model
    araştırma sonucunu **kullanır**.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    yol = "models/qwen_havuz2"
    if not Path(yol).exists():
        yol = "models/qwen_coder_myai"
    tok = AutoTokenizer.from_pretrained(yol)
    mdl = AutoModelForCausalLM.from_pretrained(yol)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(cihaz)
    mdl.eval()

    baglam = "\n".join(str(s)[:150] for s in list(imzalar)[:6])
    if dil == "gdscript":
        onek = (f"extends Node3D\n\n# Görev: {gorev}\n"
                f"# API: {kutuphane}.{api}\n"
                f"# {baglam[:200]}\n\n"
                f"func _ready() -> void:\n    var n := {kutuphane}.new()\n"
                f"    n.")
    else:
        onek = (f"# Görev: {gorev}\n# API: {kutuphane}.{api}\n"
                f"# {baglam[:200]}\n\nimport {kutuphane}\n\n")
    ids = tok(onek, return_tensors="pt")["input_ids"].to(cihaz)
    with torch.no_grad():
        c = mdl.generate(ids, max_new_tokens=90, do_sample=False,
                         pad_token_id=tok.eos_token_id,
                         no_repeat_ngram_size=3)
    u = tok.decode(c[0][ids.shape[1]:], skip_special_tokens=True)
    return onek + u


def _godot_derle(kod: str) -> Tuple[bool, str]:
    """Godot'un gerçek ayrıştırıcısıyla doğrular (`--check-only`)."""
    if not GODOT.exists():
        return (False, "godot yok")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "project.godot").write_text(
            '[application]\nconfig/name="t"\n'
            '[rendering]\nrenderer/rendering_method="mobile"\n',
            encoding="utf-8")
        hedef = d / "test.gd"
        hedef.write_text(kod, encoding="utf-8")
        try:
            p = subprocess.run(
                [str(GODOT), "--headless", "--path", str(d),
                 "--check-only", "--script", str(hedef)],
                capture_output=True, text=True, timeout=90,
                encoding="utf-8", errors="replace")
            birlesik = (p.stdout or "") + (p.stderr or "")
            hata = [s for s in birlesik.splitlines()
                    if "error" in s.lower() or "Parse Error" in s]
            return (not hata, "; ".join(hata[:2])[:160])
        except Exception as exc:
            return (False, f"{type(exc).__name__}: {str(exc)[:60]}")


def _python_derle(kod: str) -> Tuple[bool, str]:
    """`ast.parse` ile sözdizimi doğrular."""
    import ast

    try:
        ast.parse(kod)
        return (True, "")
    except SyntaxError as exc:
        return (False, f"satır {exc.lineno}: {exc.msg}")


def calistir(*, maks: int = 5) -> List[TestSonuc]:
    """Testleri çalıştırır."""
    sonuclar: List[TestSonuc] = []
    for dil, kut, api, gorev in GOREVLER[:maks]:
        s = TestSonuc(dil=dil, kutuphane=kut, api=api, gorev=gorev)
        print(f"\n  [{dil}] {kut}.{api}")
        bulundu, imza_s, goruldu = _arastir(dil, kut, api)
        s.arastirildi = bulundu
        s.imza_sayisi = imza_s
        s.aranan_api_goruldu = goruldu
        print(f"    araştırma: bulundu={bulundu} imza={imza_s} "
              f"api_goruldu={goruldu}")
        if not bulundu:
            s.hata = "araştırma başarısız"
            sonuclar.append(s)
            continue
        try:
            from neurocore.agent.coding import CodingAgent

            r = CodingAgent().api_imzasi_al(dil=dil,
                                            fonksiyon=f"{kut}.{api}")
            kod = _kod_uret(dil, kut, api, gorev, r.get("imzalar", []))
        except Exception as exc:
            s.hata = f"kod üretimi: {type(exc).__name__}"
            sonuclar.append(s)
            continue
        s.kod_uretildi = bool(kod)
        s.kod = kod
        if dil == "gdscript":
            ok, hata = _godot_derle(kod)
        else:
            ok, hata = _python_derle(kod)
        s.derlendi = ok
        s.hata = hata
        print(f"    kod: {len(kod)} karakter · derlendi={ok}"
              + (f" · {hata[:50]}" if hata else ""))
        sonuclar.append(s)
    return sonuclar


def main() -> int:
    """Testi çalıştırır."""
    ap = argparse.ArgumentParser(description="Araştırma → kod testi")
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--calistir", action="store_true")
    ap.add_argument("--maks", type=int, default=5)
    args = ap.parse_args()

    print("=" * 78)
    print("  UÇTAN UCA ARAŞTIRMA → KOD ÜRETİMİ TESTİ")
    print("=" * 78)
    print(f"  {len(GOREVLER)} görev · 2 dil")
    print(f"  Godot: {'VAR' if GODOT.exists() else 'YOK'}")

    if args.liste:
        for d, k, a, g in GOREVLER:
            print(f"  [{d:9}] {k:20}.{a:16} → {g}")
        return 0

    if not args.calistir:
        return 0

    sonuclar = calistir(maks=args.maks)
    print("\n" + "=" * 78)
    print("  SONUÇ")
    print("=" * 78)
    arastirilan = sum(1 for s in sonuclar if s.arastirildi)
    api_gorulen = sum(1 for s in sonuclar if s.aranan_api_goruldu)
    uretilen = sum(1 for s in sonuclar if s.kod_uretildi)
    derlenen = sum(1 for s in sonuclar if s.derlendi)
    n = len(sonuclar)
    print(f"  araştırma başarılı : {arastirilan}/{n}")
    print(f"  API imzada görüldü : {api_gorulen}/{n}")
    print(f"  kod üretildi       : {uretilen}/{n}")
    print(f"  DERLENDİ           : {derlenen}/{n}")
    Path("_arastirma_kod_testi.json").write_text(
        json.dumps({"sonuclar": [s.to_dict() for s in sonuclar],
                    "arastirilan": arastirilan, "uretilen": uretilen,
                    "derlenen": derlenen, "toplam": n},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if derlenen == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
