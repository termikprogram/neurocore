"""
Palmiye önizleme — 4 açıdan render (doğrulama amaçlı)
=====================================================
`Varliklar/**/palmiye*/varlik.blend` dosyasını açar, kamera+ışık kurar ve
4 açıdan PNG üretir. Amaç: **yelpaze yapraklar gerçekten radyal mi?**

Kullanım (Blender içinde):
    blender -b -P tools/palmiye_onizleme.py -- <blend> <cikti_dizini>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# ─── Argümanlar ─────────────────────────────────────────────────────────────
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
BLEND = Path(argv[0]) if argv else Path(
    "Varliklar/esya/palmiye_orta_orta/varlik.blend")
CIKTI = Path(argv[1]) if len(argv) > 1 else Path("TestProjects/palmiye_preview")
CIKTI.mkdir(parents=True, exist_ok=True)

# ─── Sahne ──────────────────────────────────────────────────────────────────
bpy.ops.wm.open_mainfile(filepath=str(BLEND.resolve()))

# Ölçü: tüm nesnelerin birleşik sınırları
nesneler = [o for o in bpy.data.objects if o.type == "MESH"]
if not nesneler:
    raise SystemExit("sahne boş")
noktalar = []
for o in nesneler:
    for k in o.bound_box:
        noktalar.append(o.matrix_world @ Vector(k))
minx = min(p.x for p in noktalar); maxx = max(p.x for p in noktalar)
miny = min(p.y for p in noktalar); maxy = max(p.y for p in noktalar)
minz = min(p.z for p in noktalar); maxz = max(p.z for p in noktalar)
merkez = Vector(((minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2))
cap = max(maxx - minx, maxy - miny, maxz - minz) or 1.0
print(f"OLCUM: genişlik={maxx-minx:.2f} derinlik={maxy-miny:.2f} "
      f"yükseklik={maxz-minz:.2f} nesne={len(nesneler)}")

# ─── Render ayarları ────────────────────────────────────────────────────────
sahne = bpy.context.scene
for motor in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
    try:
        sahne.render.engine = motor
        break
    except TypeError:
        continue
sahne.render.resolution_x = 512
sahne.render.resolution_y = 512
sahne.render.film_transparent = False
try:
    sahne.view_settings.view_transform = "Standard"
except Exception:
    pass

# Dünya: açık gri
dunya = bpy.data.worlds.new("PreviewWorld") if not bpy.data.worlds else \
    bpy.data.worlds[0]
sahne.world = dunya
dunya.use_nodes = True
arka = dunya.node_tree.nodes.get("Background")
if arka:
    arka.inputs[0].default_value = (0.55, 0.57, 0.60, 1.0)
    arka.inputs[1].default_value = 1.0

# Işık
isik_veri = bpy.data.lights.new("Gunes", type="SUN")
isik_veri.energy = 3.0
isik = bpy.data.objects.new("Gunes", isik_veri)
isik.rotation_euler = (math.radians(55), math.radians(12), math.radians(35))
sahne.collection.objects.link(isik)

# Kamera
kamera_veri = bpy.data.cameras.new("Kamera")
kamera_veri.lens = 50
kamera = bpy.data.objects.new("Kamera", kamera_veri)
sahne.collection.objects.link(kamera)
sahne.camera = kamera

# ─── 4 açı ──────────────────────────────────────────────────────────────────
# ⚠ Mesafe: nesne boyutuna göre; sabit mesafe küçük/büyük varlıkta kadrajı bozar
mesafe = cap * 2.1
yukseklik = merkez.z + cap * 0.15

acilar = {"on": 0, "yan": 90, "arka": 180, "34": 45}
uretilen = []
for ad, derece in acilar.items():
    a = math.radians(derece)
    kamera.location = (
        merkez.x + mesafe * math.cos(a),
        merkez.y + mesafe * math.sin(a),
        yukseklik,
    )
    yon = (merkez - kamera.location)
    kamera.rotation_euler = yon.to_track_quat("-Z", "Y").to_euler()
    hedef = CIKTI / f"palmiye_{ad}.png"
    sahne.render.filepath = str(hedef.resolve())
    bpy.ops.render.render(write_still=True)
    if hedef.exists():
        uretilen.append(hedef.name)
        print(f"RENDER: {hedef.name} · {hedef.stat().st_size // 1024} KB")

# Üstten görünüm (yelpazenin radyal dağılımı en iyi buradan görülür)
kamera.location = (merkez.x, merkez.y, merkez.z + cap * 2.0)
kamera.rotation_euler = (0.0, 0.0, 0.0)
hedef = CIKTI / "palmiye_ust.png"
sahne.render.filepath = str(hedef.resolve())
bpy.ops.render.render(write_still=True)
if hedef.exists():
    uretilen.append(hedef.name)
    print(f"RENDER: {hedef.name} · {hedef.stat().st_size // 1024} KB")

print(f"TAMAM: {len(uretilen)} görsel · {CIKTI}")
