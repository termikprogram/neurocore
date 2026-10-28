"""
Godot 3 → 4 API Dönüştürücü (atmak yerine düzeltmek)
======================================================
⚠ ÖLÇÜLEN DURUM (tur 16)
------------------------
`godot3_temizle.py` kirli örnekleri **attı** (8.700 örnek). Sızıntı
%50 → %25'e indi ama kalan desenler **duruyor**:
```gdscript
var c := Color3(0.5, 0.2, 1.0)   # Godot 3 → Color(...)
m.ambientColor = ...             # Godot 3 → albedo_color
m.diffuseColor = ...             # Godot 3 → albedo_color
```

**Neden atmak yetmez?** Bu desenler **kesin** ve **bağlamsız**
düzeltilebilir — atmak veri kaybı.

Neden bazıları atılmalı?
------------------------
`.set_x()` / `.get_x()` dönüşümü **bağlam** gerektirir:
```gdscript
obj.set_position(v)    → obj.position = v      # doğru
obj.set_position(v)    # ama `set_position` Godot 4'te de var (dinamik)
```
Yanlış düzeltme **atmaktan zararlı** (ölçüldü: tur 22-23). Bu yüzden
yalnız **kesin** desenler düzeltilir.

Kullanım
--------
    python tools/godot34_donustur.py --tara
    python tools/godot34_donustur.py --donustur
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: ═══════════════════════════════════════════════════════════════════
#: KESİN dönüşümler — regex → değiştirme (bağlam gerekmez)
#: ═══════════════════════════════════════════════════════════════════
#:
#: ⚠ Kural: dönüşüm **her bağlamda** doğru olmalı. Şüpheli olan
#: `SUPHELI` listesinde (uyarı verilir, değiştirilmez).
DONUSUM: Sequence[Tuple[str, str, str]] = (
    # ── Renk (Godot 4'te `Color`) ──
    (r"\bColor3\s*\(", "Color(", "Color3 → Color"),
    (r"\bColor8\s*\(", "Color8(", "Color8 (Godot 4'te de var)"),
    # ── Malzeme özellikleri ──
    (r"\.ambientColor\b", ".albedo_color", "ambientColor → albedo_color"),
    (r"\.diffuseColor\b", ".albedo_color", "diffuseColor → albedo_color"),
    (r"\.albedoColor\b", ".albedo_color", "albedoColor → albedo_color"),
    (r"\.specularColor\b", ".metallic_specular",
     "specularColor → metallic_specular"),
    # ── Sınıf adları ──
    (r"\bSpatialMaterial\b", "StandardMaterial3D",
     "SpatialMaterial → StandardMaterial3D"),
    # ⚠ tur 17 ÖLÇÜMÜ: kısa sınıf adları (`Area`, `Camera`, `Spatial`)
    # **metin içinde** yanlış eşleşiyordu:
    #     "Ground Area"  → "Ground Area3D"
    #     "Camera-ray"   → "Camera3D-ray"
    # Yalnız **kod bağlamında** güvenli oldukları için NOKTALI
    # desenler kullanılır (aşağıda `SINIF_BAGLAM`).
    (r"\bKinematicBody2D\b", "CharacterBody2D",
     "KinematicBody2D → CharacterBody2D"),
    (r"\bKinematicBody\b", "CharacterBody3D",
     "KinematicBody → CharacterBody3D"),
    (r"\bPosition3D\b", "Marker3D", "Position3D → Marker3D"),
    (r"\bOpenSimplexNoise\b", "FastNoiseLite",
     "OpenSimplexNoise → FastNoiseLite"),
    (r"\bNoiseTexture\b(?!2D|3D)", "NoiseTexture2D",
     "NoiseTexture → NoiseTexture2D"),
    # ── Packed diziler ──
    (r"\bPoolByteArray\b", "PackedByteArray", "PoolByteArray → PackedByteArray"),
    (r"\bPoolIntArray\b", "PackedInt32Array", "PoolIntArray → PackedInt32Array"),
    (r"\bPoolRealArray\b", "PackedFloat32Array",
     "PoolRealArray → PackedFloat32Array"),
    (r"\bPoolStringArray\b", "PackedStringArray",
     "PoolStringArray → PackedStringArray"),
    (r"\bPoolVector2Array\b", "PackedVector2Array",
     "PoolVector2Array → PackedVector2Array"),
    (r"\bPoolVector3Array\b", "PackedVector3Array",
     "PoolVector3Array → PackedVector3Array"),
    (r"\bPoolColorArray\b", "PackedColorArray",
     "PoolColorArray → PackedColorArray"),
    # ── Metotlar (kesin) ──
    (r"\.instance\s*\(\s*\)", ".instantiate()", "instance() → instantiate()"),
    # ⚠ tur 17 ÖLÇÜMÜ: `yield(` → `await ` dönüşümü **açılış
    # parantezini siliyordu**:
    #     `yield(get_tree(), "idle_frame")`
    #   → `await get_tree(), "idle_frame")`   ← BOZUK
    # Doğru: parantezi **koru**.
    (r"\byield\s*\(", "await (", "yield → await"),
    (r"\bOS\.get_ticks_msec\b", "Time.get_ticks_msec",
     "OS.get_ticks_msec → Time.get_ticks_msec"),
    (r"\bOS\.get_system_time_msecs\b", "Time.get_ticks_msec",
     "OS.get_system_time → Time.get_ticks_msec"),
    (r"\bFile\.new\s*\(\s*\)", "FileAccess", "File.new() → FileAccess"),
    (r"\bDirectory\.new\s*\(\s*\)", "DirAccess", "Directory.new() → DirAccess"),
    # ── Ek açıklamalar ──
    (r"^(\s*)export\s+var\b", r"\1@export var", "export var → @export var"),
    (r"^(\s*)onready\s+var\b", r"\1@onready var",
     "onready var → @onready var"),
    (r"^(\s*)export\s*\(", r"\1@export(", "export( → @export("),
    (r"^(\s*)remote\s+func\b", r"\1@rpc func", "remote func → @rpc func"),
    (r"^(\s*)master\s+func\b", r"\1@rpc func", "master func → @rpc func"),
    (r"^(\s*)puppet\s+func\b", r"\1@rpc func", "puppet func → @rpc func"),
)

#: ⚠ Kısa sınıf adları — yalnız **kod bağlamında** dönüştürülür.
#:
#: Neden? Ölçüldü (tur 17): `\bArea\b` deseni "Ground Area" gibi
#: **yorum metnini** bozuyordu. Bu desenler yalnız şu bağlamlarda
#: eşleşir: `Area.new()` · `: Area` · `<Area>` · `[Area]`
SINIF_BAGLAM: Sequence[Tuple[str, str, str]] = (
    # ⚠ tur 17 ÖLÇÜMÜ: `\s*[.(\[:]` deseni **yorum metnini** bozuyordu:
    #     "# calling Camera.method"  → "# calling Camera3D.method"
    #     "Spatial reduction factor" → "Node3D reduction factor"
    # Çözüm: `(` ve `[` **bitişik** olmalı; `(` öncesi boşluk yalnız
    # `var c := Camera (` gibi geçerli çağrıda bulunur.
    # ⚠ tur 17 ÖLÇÜMÜ: `(?=\s*\(|\[|:)` deseni `.new()` gibi **noktalı**
    # çağrıyı kapsamıyordu → `Camera.new()` değişmiyordu. Ayrıca
    # `SpatialMethod(...)` metnini yanlış eşliyordu (`.Method(` içindeki
    # `(` lookahead'i tetikliyordu).
    # Çözüm: nokta + üye adı **açıkça** eklenir ve `\b` sınırı korunur.
    (r"\bMeshInstance\b(?!3D)(?=\.\w+\s*\(|\s*\(|\[|:)", "MeshInstance3D",
     "MeshInstance → MeshInstance3D (kod)"),
    (r"\bCamera\b(?!3D)(?=\.\w+\s*\(|\s*\(|\[|:)", "Camera3D",
     "Camera → Camera3D (kod)"),
    (r"\bRigidBody\b(?!3D)(?=\.\w+\s*\(|\s*\(|\[|:)", "RigidBody3D",
     "RigidBody → RigidBody3D (kod)"),
    (r"\bStaticBody\b(?!3D)(?=\.\w+\s*\(|\s*\(|\[|:)", "StaticBody3D",
     "StaticBody → StaticBody3D (kod)"),
    (r"\bArea\b(?!3D)(?=\.\w+\s*\(|\s*\(|\[|:)", "Area3D",
     "Area → Area3D (kod)"),
    (r"\bSpatial\b(?=\.\w+\s*\(|\s*\(|\[|:)", "Node3D",
     "Spatial → Node3D (kod)"),
)

#: ⚠ Şüpheli — bağlam gerektirir; **dokunulmaz**, yalnız uyarılır
SUPHELI: Sequence[Tuple[str, str]] = (
    (r"\.set_[a-z_]+\s*\(", "setter çağrısı (özellik ataması olabilir)"),
    (r"\.get_[a-z_]+\s*\(\s*\)", "getter çağrısı (özellik okuma olabilir)"),
)

#: Kaynak dosyalar
VERI: Sequence[str] = (
    ".neurocore/temiz_veri.jsonl",
    ".neurocore/v6_veri.jsonl",
    ".neurocore/titowar_kod_g4.jsonl",
    ".neurocore/yerel_kod.jsonl",
    ".neurocore/dokuman_godot.jsonl",
)


@dataclass
class DonusumSonuc:
    """Dönüşüm sonucu."""
    dosya: str = ""
    kayit: int = 0
    degisen_kayit: int = 0
    degisim: int = 0
    sayac: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"dosya": self.dosya, "kayit": self.kayit,
                "degisen_kayit": self.degisen_kayit,
                "degisim": self.degisim, "sayac": self.sayac}


#: ⚠ Yalnız bu dildeki kod blokları dönüştürülür.
#:
#: Neden? Ölçüldü (tur 17): dönüşüm **tüm metne** uygulanınca gerçek
#: Python kodu bozuldu:
#:     `class Camera(object):`  →  `class Camera3D(object):`
#: (`nodes_camera_trajectory.py` içindeki gerçek bir sınıf.)
#: Godot API adları yalnız **GDScript**'te geçerlidir.
GECERLI_DIL: str = "gdscript"

#: Kod bloğu sınırları
_BLOK = re.compile(r"```(\w*)\s*\n(.*?)```", re.DOTALL)
_BLOK_DIL = re.compile(r"```(\w*)\s*\n")


def _bloklari_donustur(metin: str) -> Tuple[str, List[str]]:
    """Yalnız **GDScript** kod bloklarını dönüştürür."""
    uygulanan: List[str] = []
    parcalar: List[str] = []
    son = 0
    for m in _BLOK.finditer(metin):
        dil = (m.group(1) or "").strip().lower()
        parcalar.append(metin[son:m.start()])
        if dil in (GECERLI_DIL, "gd", "godot"):
            yeni_blok, uyg = _ham_donustur(m.group(0))
            parcalar.append(yeni_blok)
            uygulanan.extend(uyg)
        else:
            parcalar.append(m.group(0))
        son = m.end()
    parcalar.append(metin[son:])
    return ("".join(parcalar), uygulanan)


def _ham_donustur(metin: str) -> Tuple[str, List[str]]:
    """Desenleri uygular (dil denetimi yapmaz)."""
    yeni = metin
    uygulanan: List[str] = []
    for desen, hedef, aciklama in (*DONUSUM, *SINIF_BAGLAM):
        yeni2, n = re.subn(desen, hedef, yeni, flags=re.MULTILINE)
        if n:
            uygulanan.append(f"{aciklama} ({n})")
            yeni = yeni2
    return (yeni, uygulanan)


def donustur_metin(metin: str, *, dil: str = "") -> Tuple[str, List[str]]:
    """
    Metni dönüştürür — (yeni metin, uygulanan dönüşümler).

    ⚠ `dil` verilirse yalnız o dildeki kod blokları dönüştürülür.
    Verilmezse metin **kod bloğu içeriyorsa** yalnız GDScript blokları;
    içermiyorsa (çıplak kod örneği) tamamı işlenir.
    """
    if dil:
        if dil.strip().lower() not in (GECERLI_DIL, "gd", "godot"):
            return (metin, [])
        return _ham_donustur(metin)
    if "```" in metin:
        return _bloklari_donustur(metin)
    # Çıplak kod: GDScript göstergesi ara
    if re.search(r"\b(func\s+\w+|extends\s+\w+|var\s+\w+\s*:?=|"
                 r"@export|@onready|@rpc)\b", metin):
        return _ham_donustur(metin)
    return (metin, [])


def tara(yol: str, *, ornek: int = 0) -> DonusumSonuc:
    """Dönüşümleri sayar (yazmaz)."""
    s = DonusumSonuc(dosya=yol)
    p = Path(yol)
    if not p.exists():
        return s
    gosterilen = 0
    with p.open(encoding="utf-8") as f:
        for satir in f:
            try:
                d = json.loads(satir)
            except Exception:
                continue
            m = d.get("metin", "")
            if not m:
                continue
            s.kayit += 1
            _, uyg = donustur_metin(m)
            if uyg:
                s.degisen_kayit += 1
                s.degisim += sum(int(x.rsplit("(", 1)[1].rstrip(")"))
                                 for x in uyg)
                for x in uyg:
                    anahtar = x.rsplit(" (", 1)[0]
                    s.sayac[anahtar] = s.sayac.get(anahtar, 0) + 1
                if gosterilen < ornek:
                    yeni, _ = donustur_metin(m)
                    for a, b in zip(m.splitlines(), yeni.splitlines()):
                        if a != b:
                            print(f"      - {a.strip()[:52]}")
                            print(f"      + {b.strip()[:52]}")
                            gosterilen += 1
                            break
    return s


def uygula(yol: str, cikti: str) -> Tuple[int, int]:
    """Dönüşümü uygular ve yazar."""
    s = DonusumSonuc(dosya=yol)
    with Path(yol).open(encoding="utf-8") as f, \
            Path(cikti).open("w", encoding="utf-8") as g:
        for satir in f:
            try:
                d = json.loads(satir)
            except Exception:
                continue
            m = d.get("metin", "")
            yeni, uyg = donustur_metin(m)
            if uyg:
                d["metin"] = yeni
                d["g34_donusum"] = len(uyg)
                s.degisen_kayit += 1
            g.write(json.dumps(d, ensure_ascii=False) + "\n")
            s.kayit += 1
    return (s.kayit, s.degisen_kayit)


def main() -> int:
    """Dönüşümü çalıştırır."""
    ap = argparse.ArgumentParser(description="Godot 3→4 dönüştürücü")
    ap.add_argument("--tara", action="store_true")
    ap.add_argument("--donustur", action="store_true")
    ap.add_argument("--ornek", type=int, default=3)
    ap.add_argument("--dosya", default="")
    args = ap.parse_args()

    print("=" * 78)
    print("  GODOT 3 → 4 API DÖNÜŞTÜRÜCÜ (atmak yerine düzeltmek)")
    print("=" * 78)
    print(f"  kesin dönüşüm : {len(DONUSUM)}")
    print(f"  şüpheli (dokunulmaz): {len(SUPHELI)}")

    hedefler = [args.dosya] if args.dosya else list(VERI)
    sonuclar: List[DonusumSonuc] = []
    for y in hedefler:
        if not Path(y).exists():
            continue
        print(f"\n  {y}")
        s = tara(y, ornek=args.ornek)
        sonuclar.append(s)
        if s.kayit:
            print(f"    kayıt={s.kayit:,} değişen={s.degisen_kayit:,} "
                  f"(%{s.degisen_kayit / s.kayit * 100:.1f}) "
                  f"dönüşüm={s.degisim:,}")

    print("\n" + "=" * 78)
    print("  DÖNÜŞÜM DAĞILIMI")
    print("=" * 78)
    birlesik: Dict[str, int] = {}
    for s in sonuclar:
        for k, v in s.sayac.items():
            birlesik[k] = birlesik.get(k, 0) + v
    for k, v in sorted(birlesik.items(), key=lambda x: -x[1])[:16]:
        print(f"  {k:46} {v:7,}")

    if args.donustur:
        print("\n" + "=" * 78)
        print("  UYGULAMA")
        print("=" * 78)
        for y in hedefler:
            if not Path(y).exists():
                continue
            cikti = y.replace(".jsonl", "_g34.jsonl")
            kayit, degisen = uygula(y, cikti)
            print(f"  {y}")
            print(f"    {kayit:,} kayıt · {degisen:,} değişti → {cikti}")

    Path("_godot34_donusum.json").write_text(
        json.dumps({"sonuclar": [s.to_dict() for s in sonuclar],
                    "donusumler": birlesik}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
