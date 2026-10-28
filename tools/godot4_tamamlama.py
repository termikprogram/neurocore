"""
Godot 4 API Verisini DOĞAL Kod Tamamlama Çiftine Çevirir
==========================================================
⚠ ÖLÇÜLEN SORUN (tur 19)
------------------------
Tur 18'de dokümandan **3.089 doğru Godot 4 örneği** üretildi
(Godot 3 kirliliği %0, kendi dağılımında derleme %62). Ama eğitim
**doğal kod tamamlamayı iyileştirmedi**: 0/4 → 0/4.

**Kök neden — biçim uyuşmazlığı (ölçüldü):**

| Boyut | Tur 18 verisi | Test |
|---|---|---|
| Fonksiyon adı | `_hazirla` | `_ready` |
| Değişken adı | `d` | `n` / `m` / `t` |
| Önek konumu | satır **başı** | `n.` **ortasında** |
| Görev | "bloğu tekrarla" | "kodu **bitir**" |

**tur 12 → 13'te aynı hatâ ölçülmüştü:**
```
Belge biçimi  : mantık %67 → %58  (−%9)
Çift biçimi   : mantık %58 → %67  (+%9)
```

Çözüm: `_ready` + **doğal değişken adı** + **satır ortasında biten önek**
-----------------------------------------------------------------------
```gdscript
extends Node3D

func _ready() -> void:
    var n := FastNoiseLite.new()
    n.          ← model buradan devam eder
```

Her API üyesinden **3 varyant** üretilir:
    1. Önek `X.new()` sonrası `x.` ile biter → özellik/metot adı
    2. Önek `x.<üye> = ` ile biter → değer
    3. Önek `return x.` ile biter → özellik okuma

Kullanım
--------
    python tools/godot4_tamamlama.py --uret
    python tools/godot4_tamamlama.py --ornek 10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

KAYNAK = Path(".neurocore/godot4_api.jsonl")
CIKTI = Path(".neurocore/godot4_tamamlama.jsonl")

#: ⚠ Doğal değişken adları — teste benzer (`n`, `m`, `t`, `d`)
DEGISKEN: Sequence[str] = ("n", "m", "t", "d", "s", "o", "k", "v")

#: ⚠ Doğal fonksiyon adları — `_ready` birincil (test böyle)
FONKSIYON: Sequence[str] = ("_ready", "_baslat", "_kur")

#: Türe göre örnek değer
TIP_DEGER: Dict[str, str] = {
    "int": "1", "float": "0.5", "bool": "true", "String": '"metin"',
    "Vector2": "Vector2(1, 2)", "Vector3": "Vector3(1, 2, 3)",
    "Vector2i": "Vector2i(1, 2)", "Vector3i": "Vector3i(1, 2, 3)",
    "Color": "Color(1, 0, 0)", "Array": "[]", "Dictionary": "{}",
}

#: ⚠ Soyut sınıflar — `new()` edilemez (tur 18'de ölçüldü)
SOYUT: frozenset = frozenset({
    "MultiMesh", "Mesh", "Material", "Shader", "Texture", "Texture2D",
    "Texture3D", "Shape2D", "Shape3D", "Curve", "Font", "Skin",
    "AnimationRootNode", "AudioStream", "PackedDataContainer",
    "PhysicsMaterial", "Environment", "World3D", "Scenario",
    "ImageTexture", "ViewportTexture", "Noise", "NoiseTexture2D",
    "VideoStream", "BitMap", "Crypto", "Resource", "ResourceFormatLoader",
    "Node", "Object", "RefCounted", "Control", "CanvasItem", "Node2D",
    "Node3D", "Spatial",
})


@dataclass
class Uretim:
    """Üretim istatistiği."""
    kaynak: int = 0
    sinif: int = 0
    soyut_atlanan: int = 0
    cift: int = 0
    karakter: int = 0
    varyant: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _ornekler(sinif: str, uye: str, tip: str, metot_mu: bool,
              rng: random.Random) -> List[Tuple[str, str, str]]:
    """
    Doğal tamamlama çiftleri üretir — (önek, devam, varyant).

    ⚠ Önek **satır ortasında** biter; model kalanı üretir.
    """
    c: List[Tuple[str, str, str]] = []
    d = rng.choice(DEGISKEN)
    fn = FONKSIYON[0]
    if metot_mu:
        # Varyant: `x.<üye>(` ile biten önek
        c.append((
            f"extends Node3D\n\nfunc {fn}() -> void:\n"
            f"    var {d} := {sinif}.new()\n    {d}.{uye}(",
            ")\n", "metot"))
        # Varyant: `x.<üye>` ile biten önek (metot adı bulma)
        c.append((
            f"extends Node3D\n\nfunc {fn}() -> void:\n"
            f"    var {d} := {sinif}.new()\n    {d}.",
            f"{uye}()\n", "metot-ad"))
        return c
    # Özellik
    deger = TIP_DEGER.get(tip, "")
    if deger:
        c.append((
            f"extends Node3D\n\nfunc {fn}() -> void:\n"
            f"    var {d} := {sinif}.new()\n    {d}.{uye} = ",
            f"{deger}\n", "ozellik-deger"))
    # Özellik adı bulma — önek tam `x.` ile biter
    c.append((
        f"extends Node3D\n\nfunc {fn}() -> void:\n"
        f"    var {d} := {sinif}.new()\n    {d}.",
        f"{uye} = {deger or 'null'}\n", "ozellik-ad"))
    # Okuma
    if tip:
        c.append((
            f"extends Node3D\n\nfunc {fn}() -> {tip}:\n"
            f"    var {d} := {sinif}.new()\n    return {d}.",
            f"{uye}\n", "okuma"))
    return c


def _metin(onek: str, devam: str, sinif: str, uye: str,
           varyant: str) -> str:
    """Eğitim kaydı metni."""
    return (f"# Godot 4 kod tamamlama — {sinif}.{uye} ({varyant})\n\n"
            f"**Önek:**\n```gdscript\n{onek}\n```\n\n"
            f"**Devam:**\n```gdscript\n{devam}```\n\n"
            f"**Tam kod:**\n```gdscript\n{onek}{devam}```\n")


def uret(*, tohum: int = 42) -> Uretim:
    """Tamamlama çiftlerini üretir."""
    rng = random.Random(tohum)
    u = Uretim()
    if not KAYNAK.exists():
        print(f"  kaynak yok: {KAYNAK}")
        return u

    siniflar: Dict[str, List[Tuple[str, str, bool]]] = {}
    with KAYNAK.open(encoding="utf-8") as f:
        for satir in f:
            try:
                d = json.loads(satir)
            except Exception:
                continue
            u.kaynak += 1
            sinif = d.get("sinif", "")
            uye = d.get("uye", "")
            if not sinif or not uye:
                continue
            # ⚠ Tipi `metin`ten çıkar (`**Tür:** özellik → float`)
            tip = ""
            mt = re.search(r"\*\*Tür:\*\*\s*\w+\s*→\s*(\w+)", d.get("metin", ""))
            if mt:
                tip = mt.group(1)
            metot_mu = "**Tür:** metot" in d.get("metin", "")
            siniflar.setdefault(sinif, []).append((uye, tip, metot_mu))

    u.sinif = len(siniflar)
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    gorulen: set = set()
    with CIKTI.open("w", encoding="utf-8") as f:
        for sinif in sorted(siniflar):
            if sinif in SOYUT or sinif.startswith("@"):
                u.soyut_atlanan += len(siniflar[sinif])
                continue
            for uye, tip, metot_mu in siniflar[sinif]:
                for onek, devam, varyant in _ornekler(
                        sinif, uye, tip, metot_mu, rng):
                    anahtar = onek[-56:] + "|" + devam[:24]
                    if anahtar in gorulen:
                        continue
                    gorulen.add(anahtar)
                    m = _metin(onek, devam, sinif, uye, varyant)
                    f.write(json.dumps(
                        {"metin": m, "kategori": "godot4_tamamlama",
                         "sinif": sinif, "uye": uye, "varyant": varyant},
                        ensure_ascii=False) + "\n")
                    u.cift += 1
                    u.karakter += len(m)
                    u.varyant[varyant] = u.varyant.get(varyant, 0) + 1
    return u


def main() -> int:
    """Üretimi çalıştırır."""
    ap = argparse.ArgumentParser(description="Godot 4 tamamlama çiftleri")
    ap.add_argument("--uret", action="store_true")
    ap.add_argument("--ornek", type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print("  GODOT 4 API → DOĞAL KOD TAMAMLAMA ÇİFTLERİ")
    print("=" * 78)
    print(f"  kaynak: {KAYNAK}")

    u = uret()
    print(f"\n  kaynak kayıt   : {u.kaynak:,}")
    print(f"  sınıf          : {u.sinif:,}")
    print(f"  soyut atlanan  : {u.soyut_atlanan:,} üye")
    print(f"  ÜRETİLEN ÇİFT  : {u.cift:,}")
    print(f"  karakter       : {u.karakter:,}")
    print(f"  token          : ~{u.karakter // 3:,}")
    print(f"  varyant        : {u.varyant}")
    print(f"  dosya          : {CIKTI}")

    if args.ornek and CIKTI.exists():
        print("\n  ÖRNEKLER:")
        with CIKTI.open(encoding="utf-8") as f:
            for i, satir in enumerate(f):
                if i >= args.ornek:
                    break
                d = json.loads(satir)
                bolum = d["metin"].split("**Önek:**\n```gdscript\n")[-1]
                onek, _, kalan = bolum.partition("```")
                print(f"    [{d['sinif']}.{d['uye']} · {d['varyant']}]")
                for ln in onek.rstrip("\n").splitlines()[-3:]:
                    print(f"      {ln}")
                print(f"      ↑ devam: "
                      f"{d['metin'].split('**Devam:**')[1][:46]!r}")
                print()

    Path("_godot4_tamamlama.json").write_text(
        json.dumps(u.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
