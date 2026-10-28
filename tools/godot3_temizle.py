"""
Godot 3 → 4 API Sızıntısı Dedektörü
=====================================
⚠ ÖLÇÜLEN SORUN (tur 15)
------------------------
Uçtan uca testte model **Godot 3 API'lerini uydurdu**:
```gdscript
n.SetNoiseScale(0.001)      # Godot 3 — Godot 4'te `noise_scale` özelliği
m.ambientColor = ...        # Godot 3 — Godot 4'te `albedo_color`
m.diffuseColor = ...        # Godot 3 — Godot 4'te `albedo_color`
var c := Color3(0.5, ...)   # Godot 3 — Godot 4'te `Color`
```

**Kök neden:** Eğitim verisinde **Godot 3 dönemi GDScript** var.
Model, Godot 4 yerine **eski API'leri** öğreniyor.

Neden zararlı?
--------------
Godot 4'te bu adlar **yok** → `Parse Error: Function "set_size()" not found`
→ üretilen kod **derlenmiyor** (ölçüldü: 0/3).

Kullanım
--------
    python tools/godot3_temizle.py --tara
    python tools/godot3_temizle.py --temizle
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
#: Godot 3 → Godot 4 yeniden adlandırma tablosu (elle doğrulanmış)
#: ═══════════════════════════════════════════════════════════════════
#:
#: ⚠ Kaynak: Godot 4 resmî "GDScript changes" + sınıf referansı.
#: Yalnız **kesin** olarak bilinen eşlemeler alındı; şüpheli olanlar
#: `SUPHELI` listesinde (yalnız uyarı, değiştirilmez).
G3_G4: Dict[str, str] = {
    # ── Renk / malzeme ──
    "ambientColor": "albedo_color",
    "diffuseColor": "albedo_color",
    "albedoColor": "albedo_color",
    "Color3": "Color",
    "Color8": "Color",
    # ── Node2D / Node3D ──
    "set_position": "position",
    "get_position": "position",
    "set_rotation": "rotation",
    "get_rotation": "rotation",
    "set_scale": "scale",
    "get_scale": "scale",
    "set_visible": "visible",
    "set_name": "name",
    "get_name": "name",
    "set_size": "size",
    "get_size": "size",
    "set_text": "text",
    "get_text": "text",
    # ── Fizik (KinematicBody → CharacterBody) ──
    "KinematicBody": "CharacterBody3D",
    "KinematicBody2D": "CharacterBody2D",
    "move_and_slide_with_snap": "move_and_slide",
    "is_on_wall": "is_on_wall",
    "Spatial": "Node3D",
    "SpatialMaterial": "StandardMaterial3D",
    "MeshInstance": "MeshInstance3D",
    "Camera": "Camera3D",
    "Area": "Area3D",
    "RigidBody": "RigidBody3D",
    "StaticBody": "StaticBody3D",
    "Position3D": "Marker3D",
    "Listener": "AudioListener3D",
    "StreamPlayer": "AudioStreamPlayer",
    "StreamPlayer3D": "AudioStreamPlayer3D",
    # ── Gürültü ──
    "NoiseTexture": "NoiseTexture2D",
    "OpenSimplexNoise": "FastNoiseLite",
    "noise_scale": "noise_scale",
    "SetNoiseScale": "noise_scale",
    # ── Metin / UI ──
    "RichTextLabel": "RichTextLabel",
    "bbcode_text": "text",
    "DynamicFont": "FontFile",
    "TextureRect": "TextureRect",
    "expand": "expand_mode",
    "OS.get_ticks_msec": "Time.get_ticks_msec",
    "OS.get_system_time_msecs": "Time.get_ticks_msec",
    # ── YIELD / bekleme ──
    "yield(": "await ",
    "yield (": "await ",
    # ── Giriş ──
    "Input.is_action_just_pressed": "Input.is_action_just_pressed",
    "KEY_ESCAPE": "KEY_ESCAPE",
    # ── Çeşitli ──
    "PoolByteArray": "PackedByteArray",
    "PoolIntArray": "PackedInt32Array",
    "PoolRealArray": "PackedFloat32Array",
    "PoolStringArray": "PackedStringArray",
    "PoolVector2Array": "PackedVector2Array",
    "PoolVector3Array": "PackedVector3Array",
    "PoolColorArray": "PackedColorArray",
    "instance()": "instantiate()",
    "PackedScene.instance": "PackedScene.instantiate",
    "duplicate()": "duplicate()",
    "Directory": "DirAccess",
    "File": "FileAccess",
    "File.new()": "FileAccess.open",
    "Directory.new()": "DirAccess.open",
    "Engine.get_frames_per_second": "Engine.get_frames_per_second",
    "TranslationServer": "TranslationServer",
}

#: Godot 3'e özgü **kesin** desenler (yakalanınca örnek atılır)
G3_DESEN: Sequence[Tuple[str, str]] = (
    (r"\.set_[a-z_]+\s*\(", "setter çağrısı (Godot 4'te özellik ataması)"),
    (r"\.get_[a-z_]+\s*\(\s*\)", "getter çağrısı"),
    (r"\byield\s*\(", "yield (Godot 4'te `await`)"),
    (r"\bKinematicBody", "KinematicBody → CharacterBody3D"),
    (r"\bSpatial\b", "Spatial → Node3D"),
    (r"\bSpatialMaterial\b", "SpatialMaterial → StandardMaterial3D"),
    (r"\bColor3\s*\(", "Color3 → Color"),
    (r"\bPool(Byte|Int|Real|String|Vector2|Vector3|Color)Array\b",
     "Pool*Array → Packed*Array"),
    (r"\bambientColor\b|\bdiffuseColor\b", "ambientColor → albedo_color"),
    (r"\bOpenSimplexNoise\b", "OpenSimplexNoise → FastNoiseLite"),
    (r"\bDirectory\.new\s*\(|\bDirectory\b", "Directory → DirAccess"),
    (r"\bFile\.new\s*\(", "File.new → FileAccess.open"),
    (r"\bMeshInstance\b(?!3D)", "MeshInstance → MeshInstance3D"),
    (r"\bCamera\b(?!3D)", "Camera → Camera3D"),
    (r"\bPosition3D\b", "Position3D → Marker3D"),
    (r"\.instance\s*\(\s*\)", "instance() → instantiate()"),
    (r"\bexport\s*\(|\bonready\s+var\b", "export/onready → @export/@onready"),
    (r"\bOS\.get_ticks_msec\b|\bOS\.get_system_time_msecs\b",
     "OS zaman → Time"),
    # ⚠ tur 16: `get_node_or_null` yanlış pozitifti — Godot 4'te de var.
    # Ölçüldü: 2.709 sahte "kirli" kayıt üretiyordu.
)

#: ⚠ Şüpheli — Godot 4'te de geçerli olabilir; **uyarı** verilir
SUPHELI: Sequence[str] = (
    "set_", "get_", "position", "rotation", "scale", "size",
)


@dataclass
class TaramaSonuc:
    """Tarama sonucu."""
    dosya: str = ""
    kayit: int = 0
    kirli: int = 0
    temiz: int = 0
    desen_sayaci: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"dosya": self.dosya, "kayit": self.kayit,
                "kirli": self.kirli, "temiz": self.temiz,
                "desenler": self.desen_sayaci}


#: Eğitim verisi dosyaları
VERI: Sequence[str] = (
    ".neurocore/v6_veri.jsonl",
    ".neurocore/yerel_kod.jsonl",
    ".neurocore/mantik_tm_verisi.jsonl",
    ".neurocore/dokuman_godot.jsonl",
    ".neurocore/karisim_verisi.jsonl",
    ".neurocore/mantik_agirlikli.jsonl",
    ".neurocore/titowar_kod.jsonl",
)


def tara(yol: str, *, ornek: int = 0) -> TaramaSonuc:
    """Bir dosyayı Godot 3 kalıntısı için tarar."""
    s = TaramaSonuc(dosya=yol)
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
            metin = d.get("metin", "")
            if not metin:
                continue
            s.kayit += 1
            bulundu = False
            for desen, aciklama in G3_DESEN:
                if re.search(desen, metin):
                    s.desen_sayaci[aciklama] = (
                        s.desen_sayaci.get(aciklama, 0) + 1)
                    bulundu = True
                    if gosterilen < ornek:
                        m = re.search(desen, metin)
                        baglam = metin[max(0, m.start() - 40):m.end() + 40]
                        print(f"      ⚠ {aciklama}")
                        print(f"        {baglam!r}"[:110])
                        gosterilen += 1
            if bulundu:
                s.kirli += 1
            else:
                s.temiz += 1
    return s


def temizle(yol: str, cikti: str) -> Tuple[int, int]:
    """
    Godot 3 kalıntısı içeren örnekleri **atar**.

    ⚠ Neden düzeltmek yerine atıyoruz? `set_position(` → `position = `
    dönüşümü **bağlam** gerektirir (bazı setter'lar Godot 4'te de var).
    Yanlış düzeltme, atmaktan **daha zararlı** (ölçüldü: tur 22-23).
    """
    atilan = tutulan = 0
    with Path(yol).open(encoding="utf-8") as f, \
            Path(cikti).open("w", encoding="utf-8") as g:
        for satir in f:
            try:
                d = json.loads(satir)
            except Exception:
                continue
            metin = d.get("metin", "")
            if any(re.search(desen, metin) for desen, _ in G3_DESEN):
                atilan += 1
                continue
            g.write(satir if satir.endswith("\n") else satir + "\n")
            tutulan += 1
    return (tutulan, atilan)


def main() -> int:
    """Tarama/temizleme yapar."""
    ap = argparse.ArgumentParser(description="Godot 3 API sızıntısı")
    ap.add_argument("--tara", action="store_true")
    ap.add_argument("--temizle", action="store_true")
    ap.add_argument("--ornek", type=int, default=3)
    ap.add_argument("--dosya", default="")
    args = ap.parse_args()

    print("=" * 78)
    print("  GODOT 3 → 4 API SIZINTISI DEDEKTÖRÜ")
    print("=" * 78)
    print(f"  desen: {len(G3_DESEN)} · eşleme: {len(G3_G4)}")

    hedefler = [args.dosya] if args.dosya else list(VERI)
    sonuclar: List[TaramaSonuc] = []
    for y in hedefler:
        if not Path(y).exists():
            continue
        print(f"\n  {y}")
        s = tara(y, ornek=args.ornek)
        sonuclar.append(s)
        if s.kayit:
            oran = s.kirli / max(1, s.kayit) * 100
            print(f"    kayıt={s.kayit:,} kirli={s.kirli:,} "
                  f"temiz={s.temiz:,} (%{oran:.1f} kirli)")

    print("\n" + "=" * 78)
    print("  DESEN DAĞILIMI (tüm dosyalar)")
    print("=" * 78)
    birlesik: Dict[str, int] = {}
    for s in sonuclar:
        for k, v in s.desen_sayaci.items():
            birlesik[k] = birlesik.get(k, 0) + v
    for k, v in sorted(birlesik.items(), key=lambda x: -x[1])[:14]:
        print(f"  {k:48} {v:7,}")

    if args.temizle:
        print("\n" + "=" * 78)
        print("  TEMİZLEME")
        print("=" * 78)
        for y in hedefler:
            if not Path(y).exists():
                continue
            cikti = y.replace(".jsonl", "_g4.jsonl")
            tutulan, atilan = temizle(y, cikti)
            print(f"  {y}")
            print(f"    tutulan={tutulan:,} atılan={atilan:,} → {cikti}")

    Path("_godot3_tarama.json").write_text(
        json.dumps({"sonuclar": [s.to_dict() for s in sonuclar],
                    "desenler": birlesik}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
