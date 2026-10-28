"""
Dönen varlık animasyonu — .blend → N kare → GIF
===============================================
README için kısa animasyon üretir: kamera modelin çevresinde döner, her
kare PNG olarak render edilir, sonra Pillow ile GIF'e çevrilir.

Kullanım (Blender):
    blender -b -P tools/donus_animasyon.py -- <blend> <cikti.gif> [kare] [boyut]

Örnek:
    blender -b -P tools/donus_animasyon.py -- Varliklar/esya/palmiye_orta_orta/varlik.blend docs/images/palm-rotate.gif 16 480
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector

# ─── Argümanlar ─────────────────────────────────────────────────────────────
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not argv:
    raise SystemExit("kullanım: blender -b -P donus_animasyon.py -- <blend> <gif>")

# ⚠ Blender'ın çalışma dizini farklı olabilir: yolları **mutlak** yap,
# aksi hâlde kareler beklenmeyen bir yere yazılır (ölçüldü).
BLEND = Path(argv[0]).resolve()
GIF = Path(argv[1]).resolve()
KARE = int(argv[2]) if len(argv) > 2 else 16
BOYUT = int(argv[3]) if len(argv) > 3 else 480

GIF.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(BLEND.resolve()))

# ─── Ölçü ───────────────────────────────────────────────────────────────────
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
print(f"OLCUM: {maxx-minx:.2f} x {maxy-miny:.2f} x {maxz-minz:.2f}")

# ─── Sahne ──────────────────────────────────────────────────────────────────
sahne = bpy.context.scene
for motor in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
    try:
        sahne.render.engine = motor
        break
    except TypeError:
        continue
sahne.render.resolution_x = BOYUT
sahne.render.resolution_y = BOYUT
sahne.render.film_transparent = False
try:
    sahne.view_settings.view_transform = "Standard"
except Exception:
    pass

dunya = bpy.data.worlds[0] if bpy.data.worlds else bpy.data.worlds.new("W")
sahne.world = dunya
dunya.use_nodes = True
arka = dunya.node_tree.nodes.get("Background")
if arka:
    arka.inputs[0].default_value = (0.55, 0.57, 0.60, 1.0)
    arka.inputs[1].default_value = 1.0

isik_veri = bpy.data.lights.new("Gunes", type="SUN")
isik_veri.energy = 3.0
isik = bpy.data.objects.new("Gunes", isik_veri)
isik.rotation_euler = (math.radians(55), math.radians(12), math.radians(35))
sahne.collection.objects.link(isik)

kamera_veri = bpy.data.cameras.new("Kamera")
kamera_veri.lens = 50
kamera = bpy.data.objects.new("Kamera", kamera_veri)
sahne.collection.objects.link(kamera)
sahne.camera = kamera

mesafe = cap * 2.0
yukseklik = merkez.z + cap * 0.12

# ─── Kareler ────────────────────────────────────────────────────────────────
# ⚠ Blender'ın Python'unda Pillow YOK: kareler kalıcı dizine yazılır,
# GIF ayrı bir adımda (Pillow'lu python) üretilir.
gecici = GIF.parent / f"_frames_{GIF.stem}"
gecici.mkdir(parents=True, exist_ok=True)
kareler = []
for i in range(KARE):
    a = 2 * math.pi * i / KARE
    kamera.location = (merkez.x + mesafe * math.cos(a),
                       merkez.y + mesafe * math.sin(a),
                       yukseklik)
    yon = merkez - kamera.location
    kamera.rotation_euler = yon.to_track_quat("-Z", "Y").to_euler()
    hedef = gecici / f"k{i:03d}.png"
    sahne.render.filepath = str(hedef)
    bpy.ops.render.render(write_still=True)
    if hedef.exists():
        kareler.append(hedef)
    print(f"KARE {i + 1}/{KARE}: {hedef.name}")

# ─── GIF (Pillow, Blender dışı python ile değil — burada sys.path'e ekle) ──
try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    print(f"GIF_ATLANDI: Pillow yok ({exc})")
    print(f"KARELER: {gecici}")
    print(f"SONRAKI ADIM: python -m tools.gif_yap \"{gecici}\" \"{GIF}\"")
    raise SystemExit(0)

goruntuler = [Image.open(f).convert("P", palette=Image.ADAPTIVE, colors=128)
              for f in kareler]
if goruntuler:
    goruntuler[0].save(GIF, save_all=True, append_images=goruntuler[1:],
                       duration=90, loop=0, optimize=True)
    print(f"GIF: {GIF} · {len(goruntuler)} kare · "
          f"{GIF.stat().st_size // 1024} KB")
