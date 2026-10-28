"""
İnsansı Karakter Üretimi — Mesh → Armature → Weight → UV → Animasyon
======================================================================
⚠ Kullanıcı hedefi (C + D)
--------------------------
*"bir karakter ... oluştur dedik, şeklini oluşturma, mesh'e göre
armature oluşturma, kemik yapısını ve automatic weight yapabilme"*

⚠ Neden timsahtan **farklı**?
-----------------------------
```
Timsah (dört ayaklı)          İnsan (iki ayaklı, dik)
  kok → omurga                 kok → kalça → omurga → göğüs
   ├ 4 ayak (yatay)             ├ boyun → kafa
   └ kuyruk (3 parça)           ├ omuz_l → kol_l → dirsek_l → el_l
                                ├ omuz_r → ...
                                ├ bacak_l → diz_l → ayak_l → parmak_l
                                └ bacak_r → ...
```
⚠ Ölçülen fark: **dik duruş** kemik eksenlerini değiştirir. İnsanda
bacaklar **-Z** yönünde, timsahta **-Z + yatay** yayılım.

⚠ Kullanıcı: *"her armature'lü varlık için kemik yapısı farklı
olacağından ona uygun anim yapmak gerekecek"*
→ Animasyon ÜRETİMİ bu betikte **yapılmaz**; ortak boru hattı
(`tools/anim_uret.py`, 5 tarif) kullanılır. Böylece tek kaynak olur
ve farklı kemik yapıları **rol eşleştirmesiyle** desteklenir.

Kullanım
--------
    python tools/karakter_uret.py --olcum
    python tools/karakter_uret.py --kalite kaliteli
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
from typing import Any, Dict, List

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

BLENDER = Path(r"E:\AppsTools\Blender 5.2\blender.exe")


#: ═══════════════════════════════════════════════════════════════════
#: İNSAN KARAKTERİ — dik duruş, iki ayaklı
#: ═══════════════════════════════════════════════════════════════════
#:
#: ⚠ Kemik yerleşimi mesh parçalarına **denk** getirilir
#: (automatic weight için şart — tur 1 ölçümü)
KARAKTER = r'''
import bpy, bmesh, math, json
from mathutils import Vector, Euler

bpy.ops.wm.read_factory_settings(use_empty=True)
KALITE = KALITE_ADI
_SUBD = {"lowpoly": 1, "orta": 2, "kaliteli": 3}[KALITE]

parcalar = []
olcum_parca = {}

def parca(ad, tip, olcek, konum, *, alt_bolme=0, egim=(0, 0, 0)):
    """Gövde parçası ekler."""
    if tip == "kup":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=konum)
    elif tip == "kure":
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=_SUBD, radius=1.0,
                                              location=konum)
    elif tip == "silindir":
        bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=1.0,
                                            depth=1.0, location=konum)
    elif tip == "koni":
        bpy.ops.mesh.primitive_cone_add(vertices=8, radius1=1.0, radius2=0.4,
                                        depth=1.0, location=konum)
    ob = bpy.context.active_object
    ob.name = ad
    ob.scale = olcek
    ob.rotation_euler = Euler(egim, "XYZ")
    bpy.ops.object.transform_apply(rotation=True, scale=True)
    if alt_bolme:
        m = ob.modifiers.new("AltBolme", "SUBSURF")
        m.subdivision_type = "SIMPLE"
        m.levels = alt_bolme
        bpy.ops.object.modifier_apply(modifier=m.name)
    parcalar.append(ob)
    olcum_parca[ad] = len(ob.data.vertices)
    return ob

# ══ 1. GÖVDE PARÇALARI (dik insan, ~1.75 m) ═════════════════════════
# ⚠ Yükseklik dağılımı (gerçek oranlar):
#     ayak 0.00 · diz 0.46 · kalça 0.92 · bel 1.06 · göğüs 1.32
#     omuz 1.46 · boyun 1.52 · kafa 1.62 · tepe 1.75
parca("Karakter_Kalca", "kup", (0.30, 0.20, 0.16), (0, 0, 0.92), alt_bolme=1)
parca("Karakter_Karin", "kup", (0.28, 0.18, 0.18), (0, 0, 1.10), alt_bolme=1)
parca("Karakter_Gogus", "kup", (0.36, 0.22, 0.24), (0, 0, 1.34), alt_bolme=1)
parca("Karakter_Boyun", "silindir", (0.055, 0.055, 0.10), (0, 0, 1.52))
parca("Karakter_Kafa", "kure", (0.105, 0.115, 0.135), (0, 0, 1.63))
# ⚠ Kafa alt bölmesi (yüz şekli)
kh = parcalar[-1]
m = kh.modifiers.new("AltBolme", "SUBSURF")
m.subdivision_type = "SIMPLE"; m.levels = _SUBD
bpy.ops.object.modifier_apply(modifier=m.name)

# ══ 2. KOLLAR (omuz → dirsek → el) ══════════════════════════════════
for yan, x in (("l", -0.20), ("r", 0.20)):
    # Omuz
    parca(f"Karakter_Omuz_{yan}", "kure", (0.075, 0.075, 0.075),
          (x, 0, 1.44))
    # Üst kol (aşağı doğru)
    parca(f"Karakter_UstKol_{yan}", "silindir", (0.052, 0.052, 0.13),
          (x * 1.05, 0, 1.31))
    # Dirsek
    parca(f"Karakter_Dirsek_{yan}", "kure", (0.050, 0.050, 0.050),
          (x * 1.05, 0, 1.18))
    # Alt kol
    parca(f"Karakter_AltKol_{yan}", "silindir", (0.045, 0.045, 0.115),
          (x * 1.08, 0, 1.06))
    # El
    parca(f"Karakter_El_{yan}", "kup", (0.042, 0.028, 0.060),
          (x * 1.10, 0, 0.92))

# ══ 3. BACAKLAR (kalça → diz → ayak) ════════════════════════════════
for yan, x in (("l", -0.105), ("r", 0.105)):
    # Üst bacak
    parca(f"Karakter_UstBacak_{yan}", "silindir", (0.078, 0.078, 0.23),
          (x, 0, 0.69))
    # Diz
    parca(f"Karakter_Diz_{yan}", "kure", (0.072, 0.072, 0.072),
          (x, 0, 0.46))
    # Alt bacak
    parca(f"Karakter_AltBacak_{yan}", "silindir", (0.062, 0.062, 0.21),
          (x, 0, 0.25))
    # Ayak (öne uzun)
    parca(f"Karakter_Ayak_{yan}", "kup", (0.058, 0.12, 0.035),
          (x, 0.045, 0.035))

# ══ 4. BİRLEŞTİR ════════════════════════════════════════════════════
bpy.ops.object.select_all(action="DESELECT")
for ob in parcalar:
    ob.select_set(True)
bpy.context.view_layer.objects.active = parcalar[0]
bpy.ops.object.join()
govde = bpy.context.active_object
govde.name = "Karakter_Mesh"

# ══ 5. TEMİZLİK ═════════════════════════════════════════════════════
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=0.001)
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")
me = govde.data

# ══ 6. GÖLGELENDİRME (kalite kademesi) ══════════════════════════════
if KALITE == "lowpoly":
    for p in me.polygons:
        p.use_smooth = False
else:
    for p in me.polygons:
        p.use_smooth = True
    b2 = govde.modifiers.new("KenarYumusat", "BEVEL")
    b2.width = 0.005 if KALITE == "orta" else 0.008
    b2.segments = 1 if KALITE == "orta" else 2
    b2.limit_method = "ANGLE"
    bpy.ops.object.modifier_apply(modifier=b2.name)
    me = govde.data
    for p in me.polygons:
        p.use_smooth = True

# ══ 7. UV ═══════════════════════════════════════════════════════════
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

# ══ 8. ARMATURE — dik duruş, iki ayaklı ═════════════════════════════
bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
arm_ob = bpy.context.active_object
arm_ob.name = "Karakter_Armature"
arm = arm_ob.data
arm.name = "Karakter_Iskelet"
eb = arm.edit_bones
for b in list(eb):
    eb.remove(b)

#: (ad, baş, son, üst, bağla)
#: ⚠ Dik insan: omurga +Z, bacaklar -Z, kollar -Z (aşağı)
KEMIKLER = [
    ("kok",        (0, 0, 0.92), (0, 0, 1.00), None,       False),
    ("omurga",     (0, 0, 1.00), (0, 0, 1.22), "kok",      True),
    ("gogus",      (0, 0, 1.22), (0, 0, 1.46), "omurga",   True),
    ("boyun",      (0, 0, 1.46), (0, 0, 1.54), "gogus",    True),
    ("kafa",       (0, 0, 1.54), (0, 0, 1.76), "boyun",    True),
    # ── Sol kol ──
    ("omuz_l",     (-0.075, 0, 1.44), (-0.21, 0, 1.44), "gogus", False),
    ("ust_kol_l",  (-0.21, 0, 1.44), (-0.21, 0, 1.19), "omuz_l", True),
    ("alt_kol_l",  (-0.21, 0, 1.19), (-0.22, 0, 0.97), "ust_kol_l", True),
    ("el_l",       (-0.22, 0, 0.97), (-0.23, 0, 0.87), "alt_kol_l", True),
    # ── Sağ kol ──
    ("omuz_r",     (0.075, 0, 1.44), (0.21, 0, 1.44), "gogus", False),
    ("ust_kol_r",  (0.21, 0, 1.44), (0.21, 0, 1.19), "omuz_r", True),
    ("alt_kol_r",  (0.21, 0, 1.19), (0.22, 0, 0.97), "ust_kol_r", True),
    ("el_r",       (0.22, 0, 0.97), (0.23, 0, 0.87), "alt_kol_r", True),
    # ── Sol bacak ──
    ("ust_bacak_l", (-0.105, 0, 0.92), (-0.105, 0, 0.46), "kok", False),
    ("alt_bacak_l", (-0.105, 0, 0.46), (-0.105, 0, 0.05), "ust_bacak_l", True),
    ("ayak_l",      (-0.105, 0, 0.05), (-0.105, 0.14, 0.03), "alt_bacak_l", True),
    # ── Sağ bacak ──
    ("ust_bacak_r", (0.105, 0, 0.92), (0.105, 0, 0.46), "kok", False),
    ("alt_bacak_r", (0.105, 0, 0.46), (0.105, 0, 0.05), "ust_bacak_r", True),
    ("ayak_r",      (0.105, 0, 0.05), (0.105, 0.14, 0.03), "alt_bacak_r", True),
]
olusturulan = {}
for ad, bas, son, ust, bagla in KEMIKLER:
    b = eb.new(ad)
    b.head = Vector(bas)
    b.tail = Vector(son)
    if ust:
        b.parent = olusturulan[ust]
        b.use_connect = bagla
    olusturulan[ad] = b
bpy.ops.object.mode_set(mode="OBJECT")
kemik_adlari = [b.name for b in arm.bones]

# ══ 9. AUTOMATIC WEIGHT ═════════════════════════════════════════════
weight_ok = False
vg_sayisi = 0
try:
    bpy.ops.object.select_all(action="DESELECT")
    govde.select_set(True)
    arm_ob.select_set(True)
    bpy.context.view_layer.objects.active = arm_ob
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    weight_ok = True
    vg_sayisi = len(govde.vertex_groups)
except Exception as exc:
    print("WEIGHT_HATA:" + str(exc)[:120])

# ══ 10. MALZEME ═════════════════════════════════════════════════════
mat = bpy.data.materials.new(name="Karakter_Deri")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.72, 0.56, 0.44, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.2
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.location = (-560, -200)
    tex.inputs["Scale"].default_value = 42.0
    tex.inputs["Detail"].default_value = 5.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-280, -200)
    bump.inputs["Strength"].default_value = 0.12
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
me.materials.append(mat)

# ══ 11. ANİMASYON ═══════════════════════════════════════════════════
#: ⚠ tur 10: Karakter KENDİ animasyonunu üretmez. Animasyonlar ortak
#: boru hattından gelir (`tools/anim_uret.py`, 5 tarif).
#:
#: Ölçülen sorun: karakter `Insan_Yuruyus` üretiyordu ama
#: `anim_uret.py` tarifleri eklerken onu **siliyordu** (ölçüm: 5 action,
#: 6 beklenirken). Tek kaynak → `anim_uret.py`.
animasyonlar = []

# ══ 12. ÖLÇÜM ═══════════════════════════════════════════════════════
me = govde.data
toplam_kf = 0
olcum = {
    "nesne": govde.name, "kalite": KALITE,
    "vertex": len(me.vertices), "kenar": len(me.edges),
    "yuz": len(me.polygons),
    "uv_katman": len(me.uv_layers), "uv_ok": uv_ok,
    "malzeme": [m.name for m in me.materials],
    "dugum_sayisi": len(mat.node_tree.nodes),
    "duz_shading": all(not p.use_smooth for p in me.polygons),
    "armature_var": True,
    "kemik_sayisi": len(kemik_adlari),
    "kemik_adlari": kemik_adlari,
    "weight_ok": weight_ok,
    "vertex_group": vg_sayisi,
    "animasyonlar": [],
    "animasyonlu_kemik": 0,
    "fcurve": 0,
    "toplam_keyframe": 0,
    "parca": len(parcalar),
    "boyut": [round(float(x), 4) for x in govde.dimensions],
}
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
bpy.ops.export_scene.gltf(filepath=str(OUT_GLB), export_format="GLB")
print("OLCUM:" + json.dumps(olcum, ensure_ascii=False))
'''


def uret(kalite: str = "orta", *, zaman: int = 300) -> Dict[str, Any]:
    """İnsan karakteri üretir."""
    from tools.blender_varlik_testi import _varlik_yolu

    if not BLENDER.exists():
        return {"hata": "blender yok"}
    hedef = _varlik_yolu(f"insan_{kalite}")
    hedef.mkdir(parents=True, exist_ok=True)
    baslik = (f'KALITE_ADI = "{kalite}"\n'
              f'OUT_BLEND = r"{(hedef / "varlik.blend").resolve()}"\n'
              f'OUT_GLB = r"{(hedef / "varlik.glb").resolve()}"\n')
    kod = baslik + KARAKTER

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
    d: Dict[str, Any] = {}
    for satir in out.splitlines():
        if satir.strip().startswith("OLCUM:"):
            try:
                d = json.loads(satir.strip()[6:])
            except Exception:
                pass
        if satir.startswith("WEIGHT_HATA:"):
            d["weight_hata"] = satir[13:][:110]
    if not d:
        iz = [x.strip() for x in out.splitlines() if "Error" in x]
        return {"hata": "; ".join(iz[:2])[:170] or "OLCUM yok"}
    d["sure_s"] = round(sure, 2)
    d["klasor"] = str(hedef)
    d["glb_var"] = (hedef / "varlik.glb").exists()
    if d["glb_var"]:
        d["glb_kb"] = round(
            (hedef / "varlik.glb").stat().st_size / 1024, 1)
    return d


def main() -> int:
    """Karakter üretimi."""
    ap = argparse.ArgumentParser(description="İnsansı karakter")
    ap.add_argument("--olcum", action="store_true")
    ap.add_argument("--kalite", default="orta",
                    choices=("lowpoly", "orta", "kaliteli"))
    args = ap.parse_args()

    print("=" * 78)
    print("  İNSAN KARAKTERİ — dik duruş, iki ayaklı")
    print("=" * 78)
    print(f"  Blender: {BLENDER.name} · kalite: {args.kalite}")

    if not args.olcum:
        return 0

    print("\n  üretiliyor…", flush=True)
    d = uret(args.kalite)
    if "hata" in d:
        print(f"    ✗ {d['hata']}")
        return 1
    print(f"    ✓ {d['sure_s']} sn · parça={d['parca']} "
          f"vertex={d['vertex']} yüz={d['yuz']}")
    print(f"\n  ARMATURE:")
    print(f"    kemik         : {d['kemik_sayisi']}")
    print(f"    adlar         : {d['kemik_adlari']}")
    print(f"    vertex group  : {d['vertex_group']}")
    print(f"    weight_ok     : {d['weight_ok']}")
    if d.get("weight_hata"):
        print(f"    ⚠ {d['weight_hata']}")
    print(f"\n  ANİMASYON:")
    print(f"    action        : {d['animasyonlar']}")
    print(f"    animasyonlu   : {d['animasyonlu_kemik']} kemik")
    print(f"    fcurve        : {d['fcurve']}")
    print(f"    keyframe      : {d['toplam_keyframe']}")
    print(f"\n  DİĞER:")
    for k in ("uv_ok", "uv_katman", "duz_shading", "dugum_sayisi",
              "boyut", "glb_kb", "klasor"):
        print(f"    {k:14} {d.get(k)}")

    Path("_karakter.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
