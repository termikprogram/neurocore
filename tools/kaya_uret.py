"""
Kaya Üretici — Kalite Kademesi + Parametrik Varyant
=====================================================
⚠ Kullanıcı hedefi (A)
----------------------
*"low-poly veya kaliteli kaya oluştur dediğimizde **ona göre işlem
yapacak**"*

⚠ Yani: aynı istek **kalite kademesine** göre **farklı** üretim
yapmalı:
```
"low-poly kaya"  → 20 yüz,    düz gölge,  oyun-içi
"kaliteli kaya"  → 960 yüz,   yumuşak,    render
```

Bu betik **tek arayüz** sunar: `--kalite lowpoly|orta|kaliteli`

⚠ Neden parametrik?
-------------------
```
Aynı kaya 10 kez üretilirse **aynı** çıktı → katalog anlamsız
Tohum (`SEED`) değişince **yeni varyant** → katalog anlamlı
```

Kullanım
--------
    python tools/kaya_uret.py --kalite lowpoly --adim 0.22
    python tools/kaya_uret.py --kalite kaliteli --tohum 7
    python tools/kaya_uret.py --seri 5
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


#: ⚠ Ortak kaya iskeleti — **kalite kademesine göre** dallanır.
#:
#: Tasarım:
#:     `KALITE` değişkeni Blender tarafında dallanma yapar:
#:         lowpoly  → ico_sphere(subdiv=1) + düz gölge
#:         orta     → ico_sphere(subdiv=2) + karışık
#:         kaliteli → küp + subsurf + çok oktavlı gürültü + bevel
KAYA = r'''
import bpy, bmesh, math, random, json
from mathutils import Vector, noise

bpy.ops.wm.read_factory_settings(use_empty=True)
random.seed(SEED)

KALITE = KALITE_ADI

# ══ KALİTE KADEMESİNE GÖRE ÜRETİM ═══════════════════════════════════
if KALITE == "lowpoly":
    # ── Düşük poligon: ico sphere, düz gölge ──
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
    ob = bpy.context.active_object
    me = ob.data
    for v in me.vertices:
        v.co.x *= 1.0 + random.uniform(-ADIM, ADIM)
        v.co.y *= 1.0 + random.uniform(-ADIM, ADIM)
        v.co.z *= 0.65 + random.uniform(-ADIM * 0.5, ADIM * 0.5)
        n = v.co.normalized()
        v.co += n * random.uniform(-0.06, 0.06)
    for p in me.polygons:
        p.use_smooth = False
    malz_ad = "Kaya_LowPoly_Malzeme"
    puruz = 0.9

elif KALITE == "orta":
    # ── Orta: ico sphere subdiv 2 + hafif alt bölme ──
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0)
    ob = bpy.context.active_object
    me = ob.data
    for v in me.vertices:
        n = v.co.normalized()
        g = noise.noise(v.co * 3.0)
        v.co += n * g * ADIM * 0.6
        v.co.z *= 0.7
    # Kenar yumuşatma
    b = ob.modifiers.new("KenarYumusat", "BEVEL")
    b.width = 0.008
    b.segments = 1
    b.limit_method = "ANGLE"
    bpy.ops.object.modifier_apply(modifier=b.name)
    for p in me.polygons:
        p.use_smooth = True
    malz_ad = "Kaya_Orta_Malzeme"
    puruz = 0.82

else:
    # ── Kaliteli: küp + subsurf + çok oktavlı gürültü + bevel ──
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    ob = bpy.context.active_object
    m = ob.modifiers.new("AltBolme", "SUBSURF")
    m.subdivision_type = "SIMPLE"
    m.levels = 3
    bpy.ops.object.modifier_apply(modifier=m.name)
    me = ob.data
    for v in me.vertices:
        n = v.co.normalized() if v.co.length > 0 else Vector((0, 0, 1))
        g = 0.0
        for octav, genlik in ((2.0, 1.0), (4.0, 0.5), (8.0, 0.25)):
            g += noise.noise(v.co * octav) * genlik
        v.co += n * g * ADIM
        v.co.z *= 0.7
    for p in me.polygons:
        p.use_smooth = True
    b = ob.modifiers.new("KenarYumusat", "BEVEL")
    b.width = 0.012
    b.segments = 2
    b.limit_method = "ANGLE"
    bpy.ops.object.modifier_apply(modifier=b.name)
    malz_ad = "Kaya_Kaliteli_Malzeme"
    puruz = 0.8

ob.name = f"Kaya_{KALITE.capitalize()}"
me = ob.data

# ══ UV ══════════════════════════════════════════════════════════════
uv_ok = True
try:
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
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

# ══ MALZEME ═════════════════════════════════════════════════════════
mat = bpy.data.materials.new(name=malz_ad)
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (RENK_R, RENK_G, RENK_B, 1.0)
    bsdf.inputs["Roughness"].default_value = puruz
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.15
    # ⚠ Kaliteli/orta: prosedürel gürültü + bump (doku dokusu)
    if KALITE != "lowpoly":
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.location = (-560, -200)
        tex.inputs["Scale"].default_value = (8.0 if KALITE == "kaliteli"
                                             else 5.0)
        tex.inputs["Detail"].default_value = (6.0 if KALITE == "kaliteli"
                                              else 3.0)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.location = (-280, -200)
        bump.inputs["Strength"].default_value = (0.25 if KALITE == "kaliteli"
                                                 else 0.18)
        nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
me.materials.append(mat)

# ══ ÖLÇÜM ═══════════════════════════════════════════════════════════
olcum = {
    "nesne": ob.name, "kalite": KALITE,
    "vertex": len(me.vertices), "kenar": len(me.edges),
    "yuz": len(me.polygons),
    "uv_katman": len(me.uv_layers), "uv_ok": uv_ok,
    "malzeme": [m.name for m in me.materials],
    "dugum_sayisi": len(mat.node_tree.nodes),
    "duz_shading": all(not p.use_smooth for p in me.polygons),
    "boyut": [round(float(x), 4) for x in ob.dimensions],
    "tohum": SEED, "adim": ADIM,
}
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
bpy.ops.export_scene.gltf(filepath=str(OUT_GLB), export_format="GLB")
print("OLCUM:" + json.dumps(olcum, ensure_ascii=False))
'''


def uret(kalite: str = "lowpoly", *, tohum: int = 42, adim: float = 0.22,
         renk: tuple = (0.32, 0.30, 0.28), sira: int = 0,
         zaman: int = 180) -> Dict[str, Any]:
    """Tek kaya üretir ve ölçümü döner."""
    from tools.blender_varlik_testi import _varlik_yolu

    if not BLENDER.exists():
        return {"hata": "blender yok"}
    ad = f"kaya_{kalite}"
    temel = _varlik_yolu(ad)
    # ⚠ Varyant: aynı ad + tohum → ayrı klasör
    hedef = temel if sira == 0 else temel.parent / f"{temel.name}_v{sira:02d}"
    hedef.mkdir(parents=True, exist_ok=True)

    baslik = (f'SEED = {tohum}\nADIM = {adim!r}\n'
              f'RENK_R, RENK_G, RENK_B = {renk[0]!r}, {renk[1]!r}, '
              f'{renk[2]!r}\nKALITE_ADI = "{kalite}"\n'
              f'OUT_BLEND = r"{(hedef / "varlik.blend").resolve()}"\n'
              f'OUT_GLB = r"{(hedef / "varlik.glb").resolve()}"\n')
    kod = baslik + KAYA

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
            d["klasor"] = str(hedef)
            d["blend_var"] = (hedef / "varlik.blend").exists()
            d["glb_var"] = (hedef / "varlik.glb").exists()
            if d["glb_var"]:
                d["glb_kb"] = round(
                    (hedef / "varlik.glb").stat().st_size / 1024, 1)
            return d
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:160] or "OLCUM yok"}


def main() -> int:
    """Kaya üretir."""
    ap = argparse.ArgumentParser(description="Kaya üretici")
    ap.add_argument("--kalite", default="lowpoly",
                    choices=("lowpoly", "orta", "kaliteli"))
    ap.add_argument("--tohum", type=int, default=42)
    ap.add_argument("--adim", type=float, default=0.0,
                    help="Bozulma miktarı (0 = kaliteye göre varsayılan)")
    ap.add_argument("--seri", type=int, default=0,
                    help="Kaç varyant üretilsin")
    ap.add_argument("--katalogla", action="store_true")
    args = ap.parse_args()

    # ⚠ Kaliteye göre **varsayılan** bozulma (kullanıcı isterse ezer)
    varsayilan = {"lowpoly": 0.22, "orta": 0.26, "kaliteli": 0.18}
    adim = args.adim or varsayilan[args.kalite]

    print("=" * 78)
    print("  KAYA ÜRETİCİ — kalite kademesine göre")
    print("=" * 78)
    print(f"  kalite: {args.kalite} · bozulma: {adim} · "
          f"tohum: {args.tohum}")

    adet = max(1, args.seri)
    sonuclar: List[Dict[str, Any]] = []
    for i in range(adet):
        t = args.tohum + i * 7
        print(f"\n  [{i + 1}/{adet}] tohum={t}", flush=True)
        d = uret(args.kalite, tohum=t, adim=adim, sira=(i if adet > 1 else 0))
        sonuclar.append(d)
        if "hata" in d:
            print(f"    ✗ {d['hata'][:70]}")
            continue
        print(f"    ✓ {d['sure_s']} sn · vertex={d['vertex']} "
              f"yüz={d['yuz']} uv={d['uv_ok']} "
              f"düz_gölge={d['duz_shading']}")
        print(f"      malzeme={d['malzeme']} düğüm={d['dugum_sayisi']} "
              f"boyut={d['boyut']}")

    n = sum(1 for d in sonuclar if "hata" not in d)
    print(f"\n  BAŞARILI: {n}/{len(sonuclar)}")

    if args.katalogla and n:
        import subprocess as _sp

        r = _sp.run([sys.executable, "-X", "utf8",
                     "tools/varliklari_katalogla.py", "--tara", "--kaydet"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace")
        for s in (r.stdout or "").splitlines():
            if "→" in s or "yeni:" in s:
                print(f"  {s.strip()}")

    Path("_kaya_uretim.json").write_text(
        json.dumps({"kalite": args.kalite, "adim": adim,
                    "sonuclar": sonuclar, "basarili": n},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if n == len(sonuclar) else 1


if __name__ == "__main__":
    raise SystemExit(main())
