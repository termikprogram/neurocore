"""
Prosedürel Doku Doğrulaması — Blender'da Derleme Testi
========================================================
⚠ Kullanıcı hedefi (B)
----------------------
*"mesh oluşturma, material oluşturma, texture ekleme"*

⚠ Neden gerekli?
----------------
`doku_bulucu.py` içindeki 7 prosedürel doku **tanımdır** — Blender'da
gerçekten çalışıp çalışmadıkları **ölçülmelidir**.

⚠ Ölçülen risk (tur 1-3 dersleri):
```
· Düğüm adları sürümler arası DEĞİŞİR
      `ShaderNodeTexNoise` ✓ ama girişler farklı olabilir
· Giriş adları değişir: `Specular` → `Specular IOR Level`
· `ShaderNodeTexWave` girişleri: Scale, Distortion, Detail...
· Düğüm BAĞLAMA sırası yanlışsa sessizce çalışmaz
```

Yöntem
------
    1. Her doku tanımı için Blender betiği üret
    2. Bir küpe uygula
    3. **Düğüm ağacını doğrula**: düğüm sayısı, bağlantı sayısı
    4. GLB'ye aktar → dosya oluştu mu
    5. Ölçümü raporla

Kullanım
--------
    python tools/doku_dogrula.py --olcum
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

#: ⚠ Düğüm tanımını Blender koduna çevirir
SABLON = r'''
import bpy, json

bpy.ops.wm.read_factory_settings(use_empty=True)

# ── Test nesnesi: küp + alt bölme (yüzey gerekli) ──
bpy.ops.mesh.primitive_cube_add(size=1.0)
ob = bpy.context.active_object
ob.name = "Doku_Test"
m = ob.modifiers.new("AltBolme", "SUBSURF")
m.subdivision_type = "SIMPLE"
m.levels = 2
bpy.ops.object.modifier_apply(modifier=m.name)
me = ob.data

# ── UV (doku için şart) ──
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
uv_ok = True
try:
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)
except Exception:
    uv_ok = False
bpy.ops.object.mode_set(mode="OBJECT")

# ── Malzeme + düğümler ──
mat = bpy.data.materials.new(name=MAT_AD)
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get("Principled BSDF")
uyari = []

if bsdf:
    bsdf.inputs["Base Color"].default_value = (RENK_R, RENK_G, RENK_B, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = PURUZLULUK
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = METALIK
    # ⚠ Sürüm farkı: `Specular` vs `Specular IOR Level`
    for ad in ("Specular IOR Level", "Specular"):
        if ad in bsdf.inputs:
            bsdf.inputs[ad].default_value = 0.15
            break

# ── Düğümleri ekle (düğüm tipi sürümde yoksa UYARI) ──
DUGUMLER = DUGUM_LISTESI
eklenen = []
for tip, params in DUGUMLER:
    try:
        d = nt.nodes.new(tip)
    except Exception as exc:
        uyari.append(f"{tip}: {type(exc).__name__}")
        continue
    for pk, pv in params.items():
        if pk in d.inputs:
            try:
                d.inputs[pk].default_value = pv
            except Exception:
                uyari.append(f"{tip}.{pk}: deger atanamadi")
        else:
            uyari.append(f"{tip}.{pk}: GIRIS YOK")
    eklenen.append(d)

# ── Bağla: dokuları BSDF'e bağla ──
baglanti = 0
if bsdf and eklenen:
    # Bump varsa: doku → Bump.Height → BSDF.Normal
    bump = next((d for d in eklenen if d.bl_idname == "ShaderNodeBump"), None)
    dokular = [d for d in eklenen if d != bump]
    if bump and dokular:
        try:
            nt.links.new(dokular[0].outputs["Fac" if "Fac" in
                                           dokular[0].outputs else "Color"],
                         bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
            baglanti += 2
        except Exception as exc:
            uyari.append(f"baglanti: {type(exc).__name__}")
    # İkinci doku varsa Roughness'e
    if len(dokular) > 1 and "Roughness" in bsdf.inputs:
        try:
            c = ("Fac" if "Fac" in dokular[1].outputs else "Color")
            nt.links.new(dokular[1].outputs[c], bsdf.inputs["Roughness"])
            baglanti += 1
        except Exception:
            pass
    # Bump yoksa: ilk doku → Base Color karışımı yerine Roughness
    if not bump and dokular and "Roughness" in bsdf.inputs:
        try:
            c = ("Fac" if "Fac" in dokular[0].outputs else "Color")
            nt.links.new(dokular[0].outputs[c], bsdf.inputs["Roughness"])
            baglanti += 1
        except Exception:
            pass

me.materials.append(mat)

# ── Ölçüm ──
olcum = {
    "doku": MAT_AD,
    "dugum_toplam": len(nt.nodes),
    "dugum_eklenen": len(eklenen),
    "baglanti": len(nt.links),
    "uyari": uyari,
    "uv_ok": uv_ok,
    "uv_katman": len(me.uv_layers),
    "vertex": len(me.vertices),
    "malzeme": [x.name for x in me.materials],
    "bsdf_var": bsdf is not None,
}

bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
glb_ok = True
try:
    bpy.ops.export_scene.gltf(filepath=str(OUT_GLB), export_format="GLB")
except Exception as exc:
    glb_ok = False
    olcum["glb_hata"] = str(exc)[:80]
olcum["glb_ok"] = glb_ok
print("OLCUM:" + json.dumps(olcum, ensure_ascii=False))
'''


def dogrula(ad: str, tanim: Dict[str, Any], *, zaman: int = 180
            ) -> Dict[str, Any]:
    """Tek doku tanımını Blender'da doğrular."""
    from tools.blender_varlik_testi import _varlik_yolu

    if not BLENDER.exists():
        return {"hata": "blender yok"}
    hedef = _varlik_yolu(f"doku_{ad}")
    hedef.mkdir(parents=True, exist_ok=True)
    renk = tanim.get("renk", (0.5, 0.5, 0.5))
    baslik = (
        f'MAT_AD = "{ad}"\n'
        f'RENK_R, RENK_G, RENK_B = {renk[0]!r}, {renk[1]!r}, {renk[2]!r}\n'
        f'PURUZLULUK = {tanim.get("puruzluluk", 0.8)!r}\n'
        f'METALIK = {tanim.get("metalik", 0.0)!r}\n'
        f'DUGUM_LISTESI = {json.dumps(tanim.get("dugumler", []))}\n'
        f'OUT_BLEND = r"{(hedef / "varlik.blend").resolve()}"\n'
        f'OUT_GLB = r"{(hedef / "varlik.glb").resolve()}"\n')
    kod = baslik + SABLON
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "d.py"
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
            d["hedef"] = str(hedef)
            d["glb_var"] = (hedef / "varlik.glb").exists()
            if d["glb_var"]:
                d["glb_kb"] = round(
                    (hedef / "varlik.glb").stat().st_size / 1024, 1)
            return d
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:150] or "OLCUM yok"}


def main() -> int:
    """Tüm prosedürel dokuları doğrular."""
    ap = argparse.ArgumentParser(description="Doku doğrulama")
    ap.add_argument("--olcum", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  PROSEDÜREL DOKU DOĞRULAMASI — Blender'da gerçek derleme")
    print("=" * 78)

    if not args.olcum:
        return 0

    from tools.doku_bulucu import PROSEDUREL

    sonuclar: List[Dict[str, Any]] = []
    for ad, tanim in PROSEDUREL.items():
        print(f"\n  [{ad}] doğrulanıyor…", flush=True)
        d = dogrula(ad, tanim)
        d["ad"] = ad
        sonuclar.append(d)
        if "hata" in d:
            print(f"    ✗ {d['hata'][:70]}")
            continue
        uy = d.get("uyari", [])
        print(f"    {'✓' if not uy else '~'} {d['sure_s']} sn · "
              f"düğüm={d['dugum_eklenen']}/{d['dugum_toplam'] - 1} "
              f"bağlantı={d['baglanti']} uv={d['uv_ok']} "
              f"glb={d.get('glb_kb', '?')}KB")
        if uy:
            for u in uy[:3]:
                print(f"        ⚠ {u}")

    n = sum(1 for d in sonuclar
            if "hata" not in d and not d.get("uyari"))
    kismi = sum(1 for d in sonuclar
                if "hata" not in d and d.get("uyari"))
    print(f"\n  {'=' * 74}")
    print(f"  TAM BAŞARILI : {n}/{len(sonuclar)}")
    print(f"  UYARILI      : {kismi}")
    print(f"  HATALI       : {len(sonuclar) - n - kismi}")
    print(f"  {'=' * 74}")

    Path("_doku_dogrulama.json").write_text(
        json.dumps({"sonuclar": sonuclar, "tam": n, "uyarili": kismi,
                    "toplam": len(sonuclar)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if n == len(sonuclar) else 1


if __name__ == "__main__":
    raise SystemExit(main())
