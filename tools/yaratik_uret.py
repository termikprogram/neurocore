"""
Yaratık Üretimi — Mesh → Armature → Weight → UV → Animasyon
=============================================================
⚠ Kullanıcı hedefi (C + D + E)
------------------------------
*"bir karakter veya timsah monster oluştur dedik, şeklini oluşturma,
mesh'e göre armature oluşturma, kemik yapısını ve automatic weight
yapabilme"*

*"her armature'lü varlık için kemik yapısı farklı olacağından ona
uygun anim yapmak gerekecek"*

*"texture uv si ayarlamak da kolay değil"*

⚠ Neden bu en zor kısım?
------------------------
```
Mesh      : serbest şekil (kolay)
Armature  : kemik hiyerarşisi **mesh'e uygun** olmalı
Weight    : automatic weight = her vertex → kemik ağırlığı
            (Blender'ın `ARMATURE_AUTO` parent'ı)
UV        : smart_project + dikişler
Animasyon : kemik adları **armature'e özgü** → genel anim olmaz
```

Kemik sistemi (yaratık)
-----------------------
```
kok (root)
 └ omurga (spine)
    ├ boyun → kafa → cene
    ├ on_ayak_l → dirsek_l → pençe_l
    ├ on_ayak_r → dirsek_r → pençe_r
    ├ arka_ayak_l → diz_l → ayak_l
    ├ arka_ayak_r → diz_r → ayak_r
    └ kuyruk_1 → kuyruk_2 → kuyruk_3
```

Kullanım
--------
    python tools/yaratik_uret.py --olcum
    python tools/yaratik_uret.py --timsah
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

BLENDER = Path(r"E:\AppsTools\Blender 5.2\blender.exe")

#: ═══════════════════════════════════════════════════════════════════
#: TİMSAH ÜRETİMİ — mesh + armature + weight + UV + animasyon
#: ═══════════════════════════════════════════════════════════════════
#:
#: ⚠ Tasarım kararları (ölçülecek):
#:     * Mesh **gövde parçalarından** kurulur (gövde, kafa, çene,
#:       bacaklar, kuyruk) — böylece kemikler parçalara denk gelir
#:     * Kemikler mesh **eksenlerine** göre yerleştirilir
#:     * `ARMATURE_AUTO` = Blender'ın automatic weight'i
#:     * Animasyon kemik **adlarına** bağlı (armature'e özgü)
TİMSAH = r'''
import bpy, bmesh, math, json
from mathutils import Vector, Euler

bpy.ops.wm.read_factory_settings(use_empty=True)

# ══ 1. GÖVDE PARÇALARI ══════════════════════════════════════════════
# ⚠ Her parça ayrı mesh → sonra birleştirilir (kemik yerleşimi net olur)
parcalar = []
olcum_parca = {}

def parca_ekle(ad, olcek, konum, egim=0.0):
    """Bir gövde parçası ekler (küp tabanlı, alt bölmeli)."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=konum)
    ob = bpy.context.active_object
    ob.name = ad
    ob.scale = olcek
    ob.rotation_euler = Euler((0.0, egim, 0.0), "XYZ")
    bpy.ops.object.transform_apply(location=False, rotation=True,
                                   scale=True)
    # Alt bölme → yumuşak şekil
    m = ob.modifiers.new("AltBolme", "SUBSURF")
    m.subdivision_type = "SIMPLE"
    m.levels = 2
    bpy.ops.object.modifier_apply(modifier=m.name)
    parcalar.append(ob)
    olcum_parca[ad] = len(ob.data.vertices)
    return ob

# Gövde (uzun, basık)
parca_ekle("Govde", (0.42, 1.05, 0.30), (0.0, 0.0, 0.0))
# Boyun
parca_ekle("Boyun", (0.30, 0.30, 0.24), (0.0, 0.62, 0.04))
# Kafa
parca_ekle("Kafa", (0.30, 0.46, 0.22), (0.0, 0.94, 0.06))
# Çene (üst çene uzantısı)
parca_ekle("Cene", (0.24, 0.36, 0.11), (0.0, 1.02, -0.06))
# Kuyruk — 3 parça (incelen)
parca_ekle("Kuyruk1", (0.30, 0.36, 0.24), (0.0, -0.58, -0.02))
parca_ekle("Kuyruk2", (0.20, 0.32, 0.17), (0.0, -0.92, -0.04))
parca_ekle("Kuyruk3", (0.11, 0.28, 0.10), (0.0, -1.22, -0.06))
# Bacaklar — 4 adet (2 ön, 2 arka)
for yan, x in (("l", -0.30), ("r", 0.30)):
    parca_ekle(f"OnAyak_{yan}", (0.11, 0.11, 0.30), (x, 0.34, -0.20))
    parca_ekle(f"ArkaAyak_{yan}", (0.13, 0.13, 0.34), (x, -0.36, -0.22))

# ══ 2. PARÇALARI BİRLEŞTİR ══════════════════════════════════════════
bpy.ops.object.select_all(action="DESELECT")
for ob in parcalar:
    ob.select_set(True)
bpy.context.view_layer.objects.active = parcalar[0]
bpy.ops.object.join()
govde = bpy.context.active_object
govde.name = "Timsah_Mesh"

# ══ 3. MESH TEMİZLİĞİ ═══════════════════════════════════════════════
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=0.001)
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")
me = govde.data

# ══ 4. UV AÇMA ══════════════════════════════════════════════════════
uv_ok = True
try:
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.2, island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
except Exception:
    uv_ok = False
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

# ══ 5. ARMATURE — mesh'e UYGUN kemik yapısı ═════════════════════════
# ⚠ Kemik konumları **mesh parçalarının merkezlerine** denk gelir
bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
arm_ob = bpy.context.active_object
arm_ob.name = "Timsah_Armature"
arm = arm_ob.data
arm.name = "Timsah_Iskelet"

eb = arm.edit_bones
# Varsayılan kemiği temizle
for b in list(eb):
    eb.remove(b)

#: (ad, baş, son, üst)
KEMIKLER = [
    ("kok",        (0.0, -0.30, 0.00), (0.0,  0.10, 0.00), None),
    ("omurga",     (0.0,  0.10, 0.00), (0.0,  0.60, 0.02), "kok"),
    ("boyun",      (0.0,  0.60, 0.02), (0.0,  0.82, 0.05), "omurga"),
    ("kafa",       (0.0,  0.82, 0.05), (0.0,  1.16, 0.06), "boyun"),
    ("cene",       (0.0,  0.90, 0.02), (0.0,  1.20, -0.06), "kafa"),
    ("kuyruk_1",   (0.0, -0.30, 0.00), (0.0, -0.72, -0.02), "kok"),
    ("kuyruk_2",   (0.0, -0.72, -0.02), (0.0, -1.06, -0.04), "kuyruk_1"),
    ("kuyruk_3",   (0.0, -1.06, -0.04), (0.0, -1.38, -0.06), "kuyruk_2"),
    ("on_ayak_l",  (-0.30, 0.34, -0.02), (-0.30, 0.34, -0.36), "omurga"),
    ("on_ayak_r",  ( 0.30, 0.34, -0.02), ( 0.30, 0.34, -0.36), "omurga"),
    ("arka_ayak_l", (-0.30, -0.36, -0.02), (-0.30, -0.36, -0.40), "kok"),
    ("arka_ayak_r", ( 0.30, -0.36, -0.02), ( 0.30, -0.36, -0.40), "kok"),
]
olusturulan = {}
for ad, bas, son, ust in KEMIKLER:
    b = eb.new(ad)
    b.head = Vector(bas)
    b.tail = Vector(son)
    if ust:
        b.parent = olusturulan[ust]
        # ⚠ Bağlantı: kafa/kuyruk zincirinde sürekli, bacaklarda değil
        b.use_connect = ad in ("omurga", "boyun", "kafa",
                               "kuyruk_2", "kuyruk_3")
    olusturulan[ad] = b

bpy.ops.object.mode_set(mode="OBJECT")
kemik_adlari = [b.name for b in arm.bones]

# ══ 6. AUTOMATIC WEIGHT (Blender'ın ARMATURE_AUTO'su) ═══════════════
# ⚠ Ölçülecek: automatic weight başarılı mı, kaç vertex grup oluştu?
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

# ══ 7. MALZEME ══════════════════════════════════════════════════════
mat = bpy.data.materials.new(name="Timsah_Deri")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.16, 0.32, 0.16, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.65
    # Deri dokusu — gürültü + bump
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.location = (-560, -200)
    tex.inputs["Scale"].default_value = 26.0
    tex.inputs["Detail"].default_value = 8.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-280, -200)
    bump.inputs["Strength"].default_value = 0.35
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
me.materials.append(mat)

# ══ 8. ANİMASYON — bu armature'e ÖZGÜ ═══════════════════════════════
# ⚠ Kemik adları yukarıdaki `KEMIKLER` listesinden gelir
anim_adi = "Yuruyus"
bpy.context.view_layer.objects.active = arm_ob
bpy.ops.object.mode_set(mode="POSE")
if not arm_ob.animation_data:
    arm_ob.animation_data_create()
act = bpy.data.actions.new(anim_adi)
arm_ob.animation_data.action = act

pb = arm_ob.pose.bones
KARELER = 24
for kare in range(KARELER + 1):
    t = kare / KARELER
    aci = math.sin(t * 2.0 * math.pi)

    bpy.context.scene.frame_set(kare + 1)

    # Ön bacaklar (zıt faz)
    for yan, yon in (("l", 1.0), ("r", -1.0)):
        k = pb.get(f"on_ayak_{yan}")
        if k:
            k.rotation_mode = "XYZ"
            k.rotation_euler = (yon * aci * 0.55, 0.0, 0.0)
            k.keyframe_insert(data_path="rotation_euler", frame=kare + 1)
    # Arka bacaklar (ters faz)
    for yan, yon in (("l", -1.0), ("r", 1.0)):
        k = pb.get(f"arka_ayak_{yan}")
        if k:
            k.rotation_mode = "XYZ"
            k.rotation_euler = (yon * aci * 0.5, 0.0, 0.0)
            k.keyframe_insert(data_path="rotation_euler", frame=kare + 1)
    # Kuyruk salınımı
    for i, gen in ((1, 0.22), (2, 0.3), (3, 0.38)):
        k = pb.get(f"kuyruk_{i}")
        if k:
            k.rotation_mode = "XYZ"
            k.rotation_euler = (0.0, 0.0, aci * gen)
            k.keyframe_insert(data_path="rotation_euler", frame=kare + 1)
    # Kafa hafif sallanma
    k = pb.get("kafa")
    if k:
        k.rotation_mode = "XYZ"
        k.rotation_euler = (aci * 0.09, 0.0, 0.0)
        k.keyframe_insert(data_path="rotation_euler", frame=kare + 1)

bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.scene.frame_set(1)
animasyonlar = [a.name for a in bpy.data.actions]

# ══ 9. ÖLÇÜM + KAYIT ════════════════════════════════════════════════
me = govde.data
olcum = {
    "nesne": govde.name,
    "vertex": len(me.vertices),
    "kenar": len(me.edges),
    "yuz": len(me.polygons),
    "uv_katman": len(me.uv_layers),
    "uv_ok": uv_ok,
    "malzeme": [m.name for m in me.materials],
    "dugum_sayisi": len(mat.node_tree.nodes),
    "duz_shading": all(not p.use_smooth for p in me.polygons),
    "armature_var": True,
    "kemik_sayisi": len(kemik_adlari),
    "kemik_adlari": kemik_adlari,
    "weight_ok": weight_ok,
    "vertex_group": vg_sayisi,
    "animasyonlar": animasyonlar,
    "parca_vertex": olcum_parca,
    "boyut": [round(x, 4) for x in govde.dimensions],
}

bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
bpy.ops.export_scene.gltf(filepath=str(OUT_GLB), export_format="GLB")
print("OLCUM:" + json.dumps(olcum, ensure_ascii=False))
'''


@dataclass
class UretimSonuc:
    """Üretim sonucu."""
    ad: str = ""
    basarili: bool = False
    olcum: Dict[str, Any] = field(default_factory=dict)
    sure_s: float = 0.0
    hata: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def uret(ad: str = "timsah", *, zaman: int = 300) -> UretimSonuc:
    """Timsah üretir ve ölçer."""
    s = UretimSonuc(ad=ad)
    if not BLENDER.exists():
        s.hata = "blender yok"
        return s
    hedef = Path("Varliklar/yaratik") / ad
    hedef.mkdir(parents=True, exist_ok=True)
    s.blend = str(hedef / "varlik.blend")
    s.glb = str(hedef / "varlik.glb")

    kod = (f'OUT_BLEND = r"{(hedef / "varlik.blend").resolve()}"\n'
           f'OUT_GLB = r"{(hedef / "varlik.glb").resolve()}"\n') + TİMSAH

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "uret.py"
        b.write_text(kod, encoding="utf-8")
        try:
            p = subprocess.run(
                [str(BLENDER), "--background", "--factory-startup",
                 "--python", str(b)],
                capture_output=True, text=True, timeout=zaman,
                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            s.hata = "zaman aşımı"
            return s
        except Exception as exc:
            s.hata = f"{type(exc).__name__}"
            return s
    s.sure_s = time.perf_counter() - t0

    out = (p.stdout or "") + (p.stderr or "")
    for satir in out.splitlines():
        if satir.strip().startswith("OLCUM:"):
            try:
                s.olcum = json.loads(satir.strip()[6:])
                s.basarili = True
            except Exception:
                pass
        if satir.startswith("WEIGHT_HATA:"):
            s.olcum["weight_hata"] = satir[13:][:100]
    if not s.basarili:
        izler = [x.strip() for x in out.splitlines()
                 if "Error" in x or "Traceback" in x]
        s.hata = "; ".join(izler[:2])[:200] or "OLCUM yok"
    s.olcum["blend_var"] = Path(s.blend).exists()
    s.olcum["glb_var"] = Path(s.glb).exists()
    if s.olcum["blend_var"]:
        s.olcum["blend_kb"] = round(Path(s.blend).stat().st_size / 1024, 1)
    if s.olcum["glb_var"]:
        s.olcum["glb_kb"] = round(Path(s.glb).stat().st_size / 1024, 1)
    return s


def main() -> int:
    """Üretimi çalıştırır."""
    ap = argparse.ArgumentParser(description="Yaratık üretimi")
    ap.add_argument("--olcum", action="store_true")
    ap.add_argument("--timsah", action="store_true")
    ap.add_argument("--ad", default="timsah_0001")
    args = ap.parse_args()

    print("=" * 78)
    print("  YARATIK ÜRETİMİ — mesh + armature + weight + UV + animasyon")
    print("=" * 78)
    print(f"  Blender: {BLENDER.name} · var: {BLENDER.exists()}")

    if not (args.olcum or args.timsah):
        return 0

    print(f"\n  [{args.ad}] üretiliyor…", flush=True)
    s = uret(args.ad)
    print(f"    {'✓' if s.basarili else '✗'} {s.sure_s:.1f} sn"
          + (f" · {s.hata[:80]}" if s.hata else ""))
    for k, v in s.olcum.items():
        if isinstance(v, list) and len(v) > 8:
            print(f"      {k:16} [{len(v)} öğe] {v[:6]}…")
        else:
            print(f"      {k:16} {v}")

    Path("_yaratik.json").write_text(
        json.dumps(s.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0 if s.basarili else 1


if __name__ == "__main__":
    raise SystemExit(main())
