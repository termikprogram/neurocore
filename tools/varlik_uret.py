"""
Çok Kategorili Varlık Üretici — Bitki, Eşya, Mimari, Karakter
===============================================================
⚠ Kullanıcı hedefi
------------------
*"low-poly veya kaliteli kaya oluştur dediğimizde ona göre işlem
yapacak"* → **kalite kademesi** ✓ (kaya için ölçüldü)

*"mesh oluşturma, material oluşturma"* → **kategoriye göre** de
farklı üretim gerekir:
```
bitki    → ağaç (gövde + dallar + yaprak), çalı, çim
esya     → sandık, fıçı, masa, kılıç
mimari   → duvar, kapı, sütun, zemin karosu
karakter → insansı (2 kol, 2 bacak, omurga, kafa)
```

⚠ Ortak iskelet
---------------
Tüm kategoriler aynı **kalite kademesi** mantığını kullanır:
```python
KALITE == "lowpoly"  →  düşük bölme, düz gölge
KALITE == "orta"     →  orta bölme, yumuşak
KALITE == "kaliteli" →  yüksek bölme, bevel, yumuşak
```

Kullanım
--------
    python tools/varlik_uret.py --kategori bitki --tur agac --kalite orta
    python tools/varlik_uret.py --kategori mimari --tur duvar
    python tools/varlik_uret.py --liste
    python tools/varlik_uret.py --seri
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

BLENDER = Path(r"E:\AppsTools\Blender 5.2\blender.exe")


#: ⚠ Kategori/ürün kataloğu — her biri Blender betiği parçası
#:
#: Tasarım: her ürün **gövde parçalarından** kurulur (kemik/UV mantığı
#: için net sınırlar).
URUNLER: Dict[str, Dict[str, str]] = {

    # ══ KAYA ══════════════════════════════════════════════════════════
    "kaya": {
        "kategori": "kaya",
        "puruzluluk": 0.85,
        "aciklama": "Kaya — kaliteye göre ico sphere / küp + gürültü",
        "govde": '''
import random as _r
from mathutils import Vector as _V, noise as _n
_r.seed(SEED)
if KALITE == "lowpoly":
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
    k = bpy.context.active_object
    for _v in k.data.vertices:
        _v.co.x *= 1.0 + _r.uniform(-0.22, 0.22)
        _v.co.y *= 1.0 + _r.uniform(-0.22, 0.22)
        _v.co.z *= 0.65 + _r.uniform(-0.11, 0.11)
        _v.co += _v.co.normalized() * _r.uniform(-0.06, 0.06)
    for _p in k.data.polygons:
        _p.use_smooth = False
elif KALITE == "orta":
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0)
    k = bpy.context.active_object
    for _v in k.data.vertices:
        _v.co += _v.co.normalized() * _n.noise(_v.co * 3.0) * 0.16
        _v.co.z *= 0.70
    for _p in k.data.polygons:
        _p.use_smooth = True
else:
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    k = bpy.context.active_object
    _m = k.modifiers.new("AltBolme", "SUBSURF")
    _m.subdivision_type = "SIMPLE"; _m.levels = 3
    bpy.ops.object.modifier_apply(modifier=_m.name)
    for _v in k.data.vertices:
        _nn = _v.co.normalized() if _v.co.length > 0 else _V((0, 0, 1))
        _g = 0.0
        for _oc, _ge in ((2.0, 1.0), (4.0, 0.5), (8.0, 0.25)):
            _g += _n.noise(_v.co * _oc) * _ge
        _v.co += _nn * _g * 0.18
        _v.co.z *= 0.70
    for _p in k.data.polygons:
        _p.use_smooth = True
k.name = "Kaya"
parcalar.append(k)
''',
    },

    # ══ BİTKİ ═════════════════════════════════════════════════════════
    "agac": {
        "kategori": "bitki",
        "aciklama": "Ağaç — gövde + 3 dal + yaprak kümesi",
        "govde": '''
# ── Gövde (silindir, incelen) ──
bpy.ops.mesh.primitive_cone_add(vertices=8, radius1=0.22, radius2=0.13,
                                depth=1.8, location=(0, 0, 0.9))
govde = bpy.context.active_object
govde.name = "Agac_Govde"
parcalar.append(govde)
# ── Dallar (3 adet, farklı açı) ──
import math as _m
for i, (x, y, z, eg) in enumerate(((0.28, 0.0, 1.5, -0.7),
                                   (-0.22, 0.18, 1.75, 0.6),
                                   (0.0, -0.26, 1.62, 0.5))):
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.055,
                                        depth=0.72,
                                        location=(x, y, z))
    d = bpy.context.active_object
    d.name = f"Agac_Dal_{i+1}"
    d.rotation_euler = (_m.radians(eg * 30), _m.radians(-eg * 25), 0)
    bpy.ops.object.transform_apply(rotation=True)
    parcalar.append(d)
# ── Yaprak kümesi (ico sphere, basık) ──
for i, (x, y, z, r) in enumerate(((0.0, 0.0, 2.02, 0.62),
                                  (0.32, 0.14, 1.9, 0.4),
                                  (-0.28, -0.16, 1.94, 0.42))):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=_SUBD, radius=r,
                                          location=(x, y, z))
    y2 = bpy.context.active_object
    y2.name = f"Agac_Yaprak_{i+1}"
    y2.scale = (1.0, 1.0, 0.78)
    bpy.ops.object.transform_apply(scale=True)
    parcalar.append(y2)
''',
    },

    # ⚠ ÖLÇÜLDÜ: palmiye, "ağac" tarifiyle üretildiğinde yapraklar yoğun bir
    # küre yığını oluyordu (gerçek palmiye yelpazesi değil). Bu tarif
    # **radyal yelpaze** kullanır: 9 adet yatay düzleştirilmiş frond.
    "palmiye": {
        "kategori": "bitki",
        "aciklama": "Palmiye — ince gövde + 8 sarkık iki segmentli yelpaze",
        "renk": (0.22, 0.44, 0.20),
        "govde": '''
import math as _m
from mathutils import Matrix as _M4

# ── Gövde: 3 ince segment, hafif kıvrık ──
# ⚠ ÖLÇÜLDÜ: 3 segmentli gövde (3,2 m) taca göre çok uzundu; palmiye
# "kaktüs" gibi görünüyordu. Gövde 2 segmente (1,85 m) indirildi.
for i, (z, r1, r2) in enumerate(((0.0, 0.17, 0.14),
                                 (0.92, 0.14, 0.115))):
    bpy.ops.mesh.primitive_cone_add(
        vertices=8, radius1=r1, radius2=r2, depth=0.95,
        location=(0.05 * i, 0.03 * i, 0.48 + z))
    g = bpy.context.active_object
    g.name = f"Palmiye_Govde_{i+1}"
    parcalar.append(g)

# ── Taç: 8 yelpaze, her biri 2 segment (yukarı çıkış + aşağı sarkma) ──
TEPE = 1.92
ADET = 10
#       uzunluk  eğim(°)  r1     r2
# ⚠ ÖLÇÜLDÜ: ilk denemede (r1=0,075) frondlar **ince diken** gibi göründü.
# Genişlik artırıldı ve eğim yumuşatıldı (52→108 yerine 46→84).
SEG = ((0.78,     46,      0.22, 0.125),
       (1.12,     86,      0.125, 0.020))

for i in range(ADET):
    aci = _m.radians(360.0 / ADET * i)
    Rz = _M4.Rotation(aci, 4, 'Z')
    # ilk segmentin başlangıcı: gövde tepesi
    taban = _M4.Translation((0.0, 0.0, TEPE))
    for j, (uzunluk, egim, r1, r2) in enumerate(SEG):
        bpy.ops.mesh.primitive_cone_add(
            vertices=4, radius1=r1, radius2=r2, depth=uzunluk,
            location=(0, 0, 0))
        y = bpy.context.active_object
        y.name = f"Palmiye_Yaprak_{i+1}_{j+1}"
        # ⚠ Bıçak formu: **kalınlığı** incelt, genişliği koru.
        # Önceki (1, 0.35, 1) genişliği de kısıyordu → diken görünümü.
        y.scale = (1.0, 1.0, 0.26)
        bpy.ops.object.transform_apply(scale=True)
        t = _m.radians(egim)
        yon = (_m.sin(t), 0.0, _m.cos(t))
        # segmentin **orta noktası**: başlangıç + yön*uzunluk/2
        if j == 0:
            orta = (yon[0] * uzunluk / 2, 0.0, TEPE + yon[2] * uzunluk / 2)
        else:
            # önceki segmentin ucu
            t0 = _m.radians(SEG[0][1])
            ucu = (_m.sin(t0) * SEG[0][0], 0.0, TEPE + _m.cos(t0) * SEG[0][0])
            orta = (ucu[0] + yon[0] * uzunluk / 2, 0.0,
                    ucu[2] + yon[2] * uzunluk / 2)
        y.matrix_world = (
            Rz
            @ _M4.Translation(orta)
            @ _M4.Rotation(t, 4, 'Y')
        )
        parcalar.append(y)

# ── Hindistan cevizi (3 küçük küre) ──
for i in range(3):
    a = _m.radians(120.0 * i)
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=1, radius=0.085,
        location=(0.14 * _m.cos(a), 0.14 * _m.sin(a), TEPE - 0.18))
    c = bpy.context.active_object
    c.name = f"Palmiye_Ceviz_{i+1}"
    parcalar.append(c)
''',
    },

    "cali": {
        "kategori": "bitki",
        "aciklama": "Çalı — 4 basık küre",
        "govde": '''
import random as _r
_r.seed(SEED)
for i in range(4):
    x = _r.uniform(-0.25, 0.25)
    y = _r.uniform(-0.25, 0.25)
    r = _r.uniform(0.26, 0.42)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=_SUBD, radius=r,
                                          location=(x, y, 0.24 + i * 0.05))
    c = bpy.context.active_object
    c.name = f"Cali_{i+1}"
    c.scale = (1.0, 1.0, 0.72)
    bpy.ops.object.transform_apply(scale=True)
    parcalar.append(c)
''',
    },

    "cim": {
        "kategori": "bitki",
        "aciklama": "Çim tutamı — 9 ince koni",
        "govde": '''
import random as _r, math as _m
_r.seed(SEED)
for i in range(9):
    x = _r.uniform(-0.22, 0.22); y = _r.uniform(-0.22, 0.22)
    h = _r.uniform(0.22, 0.42)
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.035,
                                    radius2=0.004, depth=h,
                                    location=(x, y, h * 0.5))
    c = bpy.context.active_object
    c.name = f"Cim_{i+1}"
    c.rotation_euler = (_r.uniform(-0.22, 0.22), _r.uniform(-0.22, 0.22), 0)
    parcalar.append(c)
''',
    },

    # ══ EŞYA ══════════════════════════════════════════════════════════
    "sandik": {
        "kategori": "esya",
        "aciklama": "Sandık — gövde + kapak + 2 çember",
        "govde": '''
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.3))
g = bpy.context.active_object; g.name = "Sandik_Govde"
g.scale = (0.5, 0.34, 0.3)
bpy.ops.object.transform_apply(scale=True)
parcalar.append(g)
# Kapak (yarım silindir)
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.34, depth=1.0,
                                    location=(0, 0, 0.6),
                                    rotation=(0, 1.5707963, 0))
k = bpy.context.active_object; k.name = "Sandik_Kapak"
k.scale = (1.0, 1.0, 0.5)
bpy.ops.object.transform_apply(scale=True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.bisect(plane_co=(0, 0, 0.6), plane_no=(0, 0, 1),
                    clear_inner=True)
bpy.ops.object.mode_set(mode="OBJECT")
parcalar.append(k)
# 2 metal çember
for i, y in enumerate((-0.22, 0.22)):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.35, minor_radius=0.022,
                                     major_segments=16, minor_segments=6,
                                     location=(0, y, 0.3),
                                     rotation=(0, 1.5707963, 0))
    c = bpy.context.active_object; c.name = f"Sandik_Cember_{i+1}"
    parcalar.append(c)
''',
    },

    "fici": {
        "kategori": "esya",
        "aciklama": "Fıçı — silindir + 3 çember",
        "govde": '''
bpy.ops.mesh.primitive_cylinder_add(vertices=14, radius=0.34, depth=0.86,
                                    location=(0, 0, 0.43))
g = bpy.context.active_object; g.name = "Fici_Govde"
parcalar.append(g)
for i, z in enumerate((0.14, 0.43, 0.72)):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.355, minor_radius=0.022,
                                     major_segments=16, minor_segments=6,
                                     location=(0, 0, z))
    c = bpy.context.active_object; c.name = f"Fici_Cember_{i+1}"
    parcalar.append(c)
''',
    },

    "masa": {
        "kategori": "esya",
        "aciklama": "Masa — tabla + 4 bacak",
        "govde": '''
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.72))
t = bpy.context.active_object; t.name = "Masa_Tabla"
t.scale = (0.55, 0.36, 0.03)
bpy.ops.object.transform_apply(scale=True)
parcalar.append(t)
for i, (x, y) in enumerate(((-0.48, -0.29), (0.48, -0.29),
                            (-0.48, 0.29), (0.48, 0.29))):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, 0.35))
    b = bpy.context.active_object; b.name = f"Masa_Bacak_{i+1}"
    b.scale = (0.035, 0.035, 0.35)
    bpy.ops.object.transform_apply(scale=True)
    parcalar.append(b)
''',
    },

    "kilic": {
        "kategori": "esya",
        "aciklama": "Kılıç — bıçak + kabza + balçak",
        "govde": '''
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.85))
b = bpy.context.active_object; b.name = "Kilic_Bicak"
b.scale = (0.035, 0.012, 0.62)
bpy.ops.object.transform_apply(scale=True)
parcalar.append(b)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.06))
k = bpy.context.active_object; k.name = "Kilic_Kabza"
k.scale = (0.028, 0.028, 0.14)
bpy.ops.object.transform_apply(scale=True)
parcalar.append(k)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.22))
bc = bpy.context.active_object; bc.name = "Kilic_Balcak"
bc.scale = (0.13, 0.022, 0.018)
bpy.ops.object.transform_apply(scale=True)
parcalar.append(bc)
''',
    },

    # ══ MİMARİ ════════════════════════════════════════════════════════
    "duvar": {
        "kategori": "mimari",
        "aciklama": "Taş duvar — 3 sıra, kaydırmalı blok",
        "govde": '''
import random as _r
_r.seed(SEED)
for sira in range(3):
    kaydir = 0.5 if sira % 2 else 0.0
    for i in range(4):
        x = (i - 1.5) * 0.52 + kaydir * 0.26
        z = 0.16 + sira * 0.32
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, 0, z))
        bl = bpy.context.active_object
        bl.name = f"Duvar_{sira+1}_{i+1}"
        bl.scale = (0.25 + _r.uniform(-0.02, 0.02), 0.14,
                    0.15 + _r.uniform(-0.015, 0.015))
        bpy.ops.object.transform_apply(scale=True)
        bl.rotation_euler = (0, 0, _r.uniform(-0.03, 0.03))
        parcalar.append(bl)
''',
    },

    "kapi": {
        "kategori": "mimari",
        "aciklama": "Kapı — çerçeve + kanat + tokmak",
        "govde": '''
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 1.0))
k = bpy.context.active_object; k.name = "Kapi_Kanat"
k.scale = (0.46, 0.045, 1.0)
bpy.ops.object.transform_apply(scale=True)
parcalar.append(k)
for i, (x, z, sx, sz) in enumerate(((0.0, 2.06, 0.56, 0.06),
                                    (-0.53, 1.0, 0.07, 1.06),
                                    (0.53, 1.0, 0.07, 1.06))):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, 0, z))
    c = bpy.context.active_object; c.name = f"Kapi_Cerceve_{i+1}"
    c.scale = (sx, 0.075, sz)
    bpy.ops.object.transform_apply(scale=True)
    parcalar.append(c)
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.055,
                                      location=(0.36, 0.06, 0.98))
t = bpy.context.active_object; t.name = "Kapi_Tokmak"
parcalar.append(t)
''',
    },

    "sutun": {
        "kategori": "mimari",
        "aciklama": "Sütun — kaide + gövde + başlık",
        "govde": '''
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.2, depth=2.0,
                                    location=(0, 0, 1.0))
g = bpy.context.active_object; g.name = "Sutun_Govde"
parcalar.append(g)
for i, (z, r, h) in enumerate(((0.06, 0.28, 0.12), (1.96, 0.28, 0.12))):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, z))
    k = bpy.context.active_object; k.name = f"Sutun_Baslik_{i+1}"
    k.scale = (r, r, h)
    bpy.ops.object.transform_apply(scale=True)
    parcalar.append(k)
    # Yivli bilezik
    bpy.ops.mesh.primitive_torus_add(major_radius=0.215, minor_radius=0.022,
                                     major_segments=16, minor_segments=6,
                                     location=(0, 0, 0.15 + i * 1.76))
    t = bpy.context.active_object; t.name = f"Sutun_Bilezik_{i+1}"
    parcalar.append(t)
''',
    },

    "zemin": {
        "kategori": "mimari",
        "aciklama": "Zemin karosu — 4x4 çukur desenli",
        "govde": '''
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
g = bpy.context.active_object; g.name = "Zemin_Karo"
m = g.modifiers.new("AltBolme", "SUBSURF")
m.subdivision_type = "SIMPLE"
m.levels = 2
bpy.ops.object.modifier_apply(modifier=m.name)
# ⚠ Karo deseni: her 4. vertex'i Z'de kaydır
me_g = g.data
for i, v in enumerate(me_g.vertices):
    if i % 5 == 0:
        v.co.z -= 0.012
parcalar.append(g)
''',
    },
}


#: ⚠ Ortak Blender şablonu — tüm ürünler bunu kullanır
SABLON = r'''
import bpy, bmesh, math, random, json
from mathutils import Vector, Euler

bpy.ops.wm.read_factory_settings(use_empty=True)
random.seed(SEED)

KALITE = KALITE_ADI
# ⚠ Kalite kademesi → alt bölme seviyesi
_SUBD = {"lowpoly": 1, "orta": 2, "kaliteli": 3}[KALITE]

parcalar = []
olcum_parca = {}

# ══ ÜRÜN GÖVDESİ (kategoriye özgü) ══════════════════════════════════
GOVDE_KODU

# ══ BİRLEŞTİR ═══════════════════════════════════════════════════════
bpy.ops.object.select_all(action="DESELECT")
for ob in parcalar:
    ob.select_set(True)
bpy.context.view_layer.objects.active = parcalar[0]
if len(parcalar) > 1:
    bpy.ops.object.join()
govde = bpy.context.active_object
govde.name = URUN_AD

# ══ TEMİZLİK ════════════════════════════════════════════════════════
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=0.001)
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")
me = govde.data

# ══ KALİTEYE GÖRE GÖLGELENDİRME ═════════════════════════════════════
if KALITE == "lowpoly":
    for p in me.polygons:
        p.use_smooth = False
else:
    for p in me.polygons:
        p.use_smooth = True
    b = govde.modifiers.new("KenarYumusat", "BEVEL")
    b.width = 0.006 if KALITE == "orta" else 0.010
    b.segments = 1 if KALITE == "orta" else 2
    b.limit_method = "ANGLE"
    bpy.ops.object.modifier_apply(modifier=b.name)
    me = govde.data
    for p in me.polygons:
        p.use_smooth = True

# ══ UV ══════════════════════════════════════════════════════════════
uv_ok = True
try:
    bpy.ops.object.select_all(action="DESELECT")
    govde.select_set(True)
    bpy.context.view_layer.objects.active = govde
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
except Exception:
    uv_ok = False
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
me = govde.data

# ══ MALZEME ═════════════════════════════════════════════════════════
mat = bpy.data.materials.new(name=URUN_AD + "_Malzeme")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (RENK_R, RENK_G, RENK_B, 1.0)
    bsdf.inputs["Roughness"].default_value = PURUZLULUK
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.15
    # ⚠ Kaliteli/orta: prosedürel doku
    if KALITE != "lowpoly":
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.location = (-560, -200)
        tex.inputs["Scale"].default_value = (
            8.0 if KALITE == "kaliteli" else 5.0)
        tex.inputs["Detail"].default_value = (
            6.0 if KALITE == "kaliteli" else 3.0)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.location = (-280, -200)
        bump.inputs["Strength"].default_value = (
            0.25 if KALITE == "kaliteli" else 0.18)
        nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
me.materials.append(mat)

# ══ ÖLÇÜM ═══════════════════════════════════════════════════════════
olcum = {
    "nesne": govde.name, "kalite": KALITE, "urun": URUN_TURU,
    "parca": len(parcalar),
    "vertex": len(me.vertices), "kenar": len(me.edges),
    "yuz": len(me.polygons),
    "uv_katman": len(me.uv_layers), "uv_ok": uv_ok,
    "malzeme": [m.name for m in me.materials],
    "dugum_sayisi": len(mat.node_tree.nodes),
    "duz_shading": all(not p.use_smooth for p in me.polygons),
    "boyut": [round(float(x), 4) for x in govde.dimensions],
    "tohum": SEED,
}
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
bpy.ops.export_scene.gltf(filepath=str(OUT_GLB), export_format="GLB")
print("OLCUM:" + json.dumps(olcum, ensure_ascii=False))
'''


def uret(urun: str, kalite: str = "orta", *, tohum: int = 42,
         renk: Optional[tuple] = None, sira: int = 0,
         zaman: int = 240) -> Dict[str, Any]:
    """Tek ürün üretir."""
    from tools.blender_varlik_testi import _varlik_yolu

    if urun not in URUNLER:
        return {"hata": f"bilinmeyen ürün: {urun}"}
    if not BLENDER.exists():
        return {"hata": "blender yok"}
    bilgi = URUNLER[urun]
    kategori = bilgi["kategori"]
    ad = f"{urun}_{kalite}"
    temel = _varlik_yolu(ad)
    hedef = temel if sira == 0 else temel.parent / f"{temel.name}_v{sira:02d}"
    hedef.mkdir(parents=True, exist_ok=True)

    if renk is None:
        # ⚠ Ürün kendi rengini bildirebilir (ör. palmiye yeşili);
        # aksi hâlde eski varsayılan kahverengi korunur.
        renk = bilgi.get("renk") or (0.42, 0.30, 0.18)
    baslik = (
        f'SEED = {tohum}\nKALITE_ADI = "{kalite}"\n'
        f'URUN_AD = "{urun.capitalize()}"\nURUN_TURU = "{urun}"\n'
        f'RENK_R, RENK_G, RENK_B = {renk[0]!r}, {renk[1]!r}, {renk[2]!r}\n'
        f'PURUZLULUK = {bilgi.get("puruzluluk", 0.8)!r}\n'
        f'OUT_BLEND = r"{(hedef / "varlik.blend").resolve()}"\n'
        f'OUT_GLB = r"{(hedef / "varlik.glb").resolve()}"\n')
    kod = baslik + SABLON.replace("GOVDE_KODU", bilgi["govde"])

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "u.py"
        b.write_text(kod, encoding="utf-8")
        try:
            p = subprocess.run(
                [str(BLENDER), "--background", "--factory-startup",
                 "--python", str(b)],
                capture_output=True, text=True, timeout=zaman,
                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return {"hata": "zaman aşımı"}
    sure = time.perf_counter() - t0
    out = (p.stdout or "") + (p.stderr or "")
    for satir in out.splitlines():
        if satir.strip().startswith("OLCUM:"):
            d = json.loads(satir.strip()[6:])
            d["sure_s"] = round(sure, 2)
            d["kategori"] = kategori
            d["klasor"] = str(hedef)
            d["glb_var"] = (hedef / "varlik.glb").exists()
            if d["glb_var"]:
                d["glb_kb"] = round(
                    (hedef / "varlik.glb").stat().st_size / 1024, 1)
            return d
    iz = [x.strip() for x in out.splitlines()
          if "Error" in x or "error" in x.lower()]
    return {"hata": "; ".join(iz[:2])[:170] or "OLCUM yok",
            "urun": urun, "kategori": kategori}


def main() -> int:
    """Çok kategorili üretim."""
    ap = argparse.ArgumentParser(description="Çok kategorili varlık üretici")
    ap.add_argument("--kategori", default="")
    ap.add_argument("--tur", default="")
    ap.add_argument("--kalite", default="orta",
                    choices=("lowpoly", "orta", "kaliteli"))
    ap.add_argument("--tohum", type=int, default=42)
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--seri", action="store_true",
                    help="Tüm ürünleri üret")
    args = ap.parse_args()

    print("=" * 78)
    print("  ÇOK KATEGORİLİ VARLIK ÜRETİCİ")
    print("=" * 78)
    print(f"  {len(URUNLER)} ürün · "
          f"{len({v['kategori'] for v in URUNLER.values()})} kategori")

    if args.liste:
        for k in sorted({v["kategori"] for v in URUNLER.values()}):
            print(f"\n  [{k.upper()}]")
            for ad, b in URUNLER.items():
                if b["kategori"] == k:
                    print(f"    {ad:10} {b['aciklama']}")
        return 0

    # Üretilecek listesi
    if args.seri:
        hedefler = list(URUNLER.keys())
    elif args.tur:
        hedefler = [args.tur]
    elif args.kategori:
        hedefler = [a for a, b in URUNLER.items()
                    if b["kategori"] == args.kategori]
    else:
        ap.print_help()
        return 0

    sonuclar: List[Dict[str, Any]] = []
    for u in hedefler:
        print(f"\n  [{u} · {args.kalite}] üretiliyor…", flush=True)
        d = uret(u, args.kalite, tohum=args.tohum)
        sonuclar.append(d)
        if "hata" in d:
            print(f"    ✗ {d['hata'][:70]}")
            continue
        print(f"    ✓ {d['sure_s']} sn · parça={d['parca']} "
              f"vertex={d['vertex']} yüz={d['yuz']} "
              f"uv={d['uv_ok']} boyut={d['boyut']}")

    n = sum(1 for d in sonuclar if "hata" not in d)
    print(f"\n  {'=' * 74}")
    print(f"  BAŞARILI: {n}/{len(sonuclar)}")
    print(f"  {'=' * 74}")

    Path("_varlik_uretim.json").write_text(
        json.dumps({"kalite": args.kalite, "sonuclar": sonuclar,
                    "basarili": n}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0 if n == len(sonuclar) else 1


if __name__ == "__main__":
    raise SystemExit(main())
