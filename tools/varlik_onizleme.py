"""
Varlık Önizleme — GLB/Blend'den Görüntü Üretimi
=================================================
⚠ Neden gerekli?
----------------
Üretilen varlıklar şu ana kadar yalnız **sayıyla** doğrulandı:
```
vertex=2012 · yüz=2270 · kemik=19 · weight=1.0 · UV=1
```
**Ama görsel olarak doğrulanmadı.** Sayılar doğru olsa da model
**bozuk görünebilir** (ters normal, yanlış ölçek, eksik parça).

⚠ Ölçülen risk
--------------
```
tur 1  : `Cannot open file` — klasör yok
tur 8  : DOLU=0 — değerler yazılmadı
tur 9  : 1/5 action — fake user yok
tur 12 : düğüm birikmesi 20
```
**Hepsi sayıyla yakalandı ama görsel doğrulama yoktu.**

Bu betik
--------
```
1. `.blend` aç
2. Kamera + 3 nokta ışık kur
3. 4 açıdan render (ön, yan, 3/4, üst)
4. PNG kaydet + ölçüm (boyut, süre)
```

⚠ Render motoru: **EEVEE** (hızlı) — Cycles yavaş, Workbench düz

Kullanım
--------
    python tools/varlik_onizleme.py --render kaya_0001
    python tools/varlik_onizleme.py --tumu
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

#: ⚠ Varlığı 4 açıdan render eder
RENDER = r'''
import bpy, json, math, os
from mathutils import Vector

bpy.ops.wm.open_mainfile(filepath=BLEND_YOLU)

CIKTI = CIKTI_V
COZUNURLUK = COZUNURLUK_V
KARE = KARE_V

os.makedirs(CIKTI, exist_ok=True)

olcum = {"kamera": 0, "isik": 0, "render": [], "uyari": [],
         "boyut": [0, 0, 0], "merkez": [0, 0, 0]}

# ══ 1. SAHNEYİ HAZIRLA ══════════════════════════════════════════════
# ⚠ Sahnedeki TÜM mesh + armature nesnelerinin sınırlarını bul
meshler = [o for o in bpy.data.objects if o.type == "MESH"]
if not meshler:
    print("P" + json.dumps({"hata": "mesh yok"}))
    raise SystemExit(0)

# ⚠ Dünya uzayında sınır kutusu
_min = Vector((1e9, 1e9, 1e9))
_max = Vector((-1e9, -1e9, -1e9))
for ob in meshler:
    for kose in ob.bound_box:
        d = ob.matrix_world @ Vector(kose)
        _min.x = min(_min.x, d.x); _min.y = min(_min.y, d.y)
        _min.z = min(_min.z, d.z)
        _max.x = max(_max.x, d.x); _max.y = max(_max.y, d.y)
        _max.z = max(_max.z, d.z)
merkez = (_min + _max) * 0.5
boyut = _max - _min
olcum["merkez"] = [round(v, 4) for v in merkez]
olcum["boyut"] = [round(v, 4) for v in boyut]
# ⚠ En büyük kenar (kamera mesafesi için)
_en_buyuk = max(boyut.x, boyut.y, boyut.z, 0.001)

# ══ 2. RENDER AYARLARI ══════════════════════════════════════════════
sahne = bpy.context.scene
# ⚠ Motor: EEVEE (Blender 5.x'te `BLENDER_EEVEE_NEXT`)
for motor in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
    try:
        sahne.render.engine = motor
        olcum["motor"] = motor
        break
    except Exception:
        continue
sahne.render.resolution_x = COZUNURLUK
sahne.render.resolution_y = COZUNURLUK
sahne.render.resolution_percentage = 100
sahne.render.image_settings.file_format = "PNG"
sahne.render.film_transparent = False
try:
    sahne.eevee.taa_render_samples = 16
except Exception:
    pass

# ⚠ Dünya arka planı (koyu gri — model görünsün)
dunya = bpy.data.worlds.new("Onizleme_Dunya") if not bpy.data.worlds else bpy.data.worlds[0]
sahne.world = dunya
dunya.use_nodes = True
_bg = dunya.node_tree.nodes.get("Background")
if _bg:
    _bg.inputs[0].default_value = (0.05, 0.05, 0.07, 1.0)
    _bg.inputs[1].default_value = 0.6

# ══ 3. IŞIKLAR (3 nokta) ════════════════════════════════════════════
def isik_ekle(ad, tip, konum, enerji, boyut=2.0):
    d = bpy.data.lights.new(ad, type=tip)
    d.energy = enerji
    if tip == "AREA":
        d.size = boyut
    ob = bpy.data.objects.new(ad, d)
    ob.location = konum
    bpy.context.collection.objects.link(ob)
    # ⚠ Hedefe yönlendir
    yon = merkez - Vector(konum)
    ob.rotation_euler = yon.to_track_quat("-Z", "Y").to_euler()
    olcum["isik"] += 1
    return ob

_m = _en_buyuk
# Anahtar (ön-sol-üst)
isik_ekle("Anahtar", "AREA",
          (merkez.x - _m * 1.6, merkez.y - _m * 1.6, merkez.z + _m * 1.8),
          _m * _m * 900, _m * 1.4)
# Dolgu (ön-sağ)
isik_ekle("Dolgu", "AREA",
          (merkez.x + _m * 2.0, merkez.y - _m * 1.2, merkez.z + _m * 0.6),
          _m * _m * 320, _m * 1.6)
# Arka (rim)
isik_ekle("Arka", "AREA",
          (merkez.x, merkez.y + _m * 2.0, merkez.z + _m * 1.4),
          _m * _m * 480, _m * 1.4)
# ⚠ Zemin (gölge düşsün)
try:
    bpy.ops.mesh.primitive_plane_add(
        size=_m * 8, location=(merkez.x, merkez.y, _min.z - 0.001))
    zemin = bpy.context.active_object
    zemin.name = "Onizleme_Zemin"
    _zm = bpy.data.materials.new("Zemin_Malzeme")
    _zm.use_nodes = True
    _b = _zm.node_tree.nodes.get("Principled BSDF")
    if _b:
        _b.inputs["Base Color"].default_value = (0.10, 0.10, 0.12, 1.0)
        _b.inputs["Roughness"].default_value = 0.9
    zemin.data.materials.append(_zm)
except Exception as exc:
    olcum["uyari"].append(f"zemin: {type(exc).__name__}")

# ══ 4. KAMERA ═══════════════════════════════════════════════════════
kamera_d = bpy.data.cameras.new("Onizleme_Kamera")
kamera_d.lens = 50.0
kamera = bpy.data.objects.new("Onizleme_Kamera", kamera_d)
bpy.context.collection.objects.link(kamera)
sahne.camera = kamera
olcum["kamera"] = 1

#: ⚠ 4 açı: (ad, yatay_derece, yükseklik_oranı)
ACILAR = [
    ("on",     -90.0, 0.10),
    ("yan",      0.0, 0.10),
    ("uc_dort", -45.0, 0.28),
    ("ust",    -45.0, 1.05),
]

mesafe = _en_buyuk * 2.35
for ad, yatay, yuk in ACILAR:
    a = math.radians(yatay)
    kamera.location = (
        merkez.x + math.cos(a) * mesafe,
        merkez.y + math.sin(a) * mesafe,
        merkez.z + _en_buyuk * yuk + _en_buyuk * 0.10,
    )
    yon = merkez - kamera.location
    kamera.rotation_euler = yon.to_track_quat("-Z", "Y").to_euler()
    yol = os.path.join(CIKTI, f"{ad}.png")
    sahne.render.filepath = yol
    try:
        bpy.ops.render.render(write_still=True)
        var = os.path.exists(yol)
        olcum["render"].append({"aci": ad, "dosya": yol, "var": var,
                                "kb": (round(os.path.getsize(yol) / 1024, 1)
                                       if var else 0)})
    except Exception as exc:
        olcum["render"].append({"aci": ad, "hata": type(exc).__name__})

# ══ 5. GLB de kaydet (önizleme sonrası) ═════════════════════════════
try:
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(CIKTI, "onizleme.glb"), export_format="GLB")
except Exception:
    pass

olcum["basarili"] = sum(1 for r in olcum["render"] if r.get("var"))
print("P" + json.dumps(olcum, ensure_ascii=False))
'''


# ══ ⚠ tur 25: ORTAK yol çözümleyici ═══════════════════════════════════
# Neden: 5 araçta **aynı** yol hatası vardı —
#   `Path("Varliklar") / hedef` → hedef zaten `Varliklar\...` içeriyorsa
#   **çift** birleşiyordu (`Varliklar\Varliklar\...`).
# Ve ID ile arama yapanlar bulamıyordu.
def _ortak_blend_bul(hedef: str):
    """Ortak çözümleyici (tam yol · klasör · göreli · ad · katalog)."""
    try:
        from tools.varlik_yolu import blend_bul

        return blend_bul(hedef)
    except Exception:
        return None


def _blend_bul(hedef: str):
    """
    Varlık `.blend` yolunu bulur.

    ⚠ tur 25: **ortak** çözümleyiciye delege edilir — önceki
    sürüm `Path("Varliklar") / hedef` ile **çift** birleştirme
    yapıyordu (`Varliklar\\Varliklar\\...`) ve ID ile
    bulamıyordu.
    """
    _o = _ortak_blend_bul(hedef)
    if _o is not None:
        return _o
    p = Path(hedef)
    if p.suffix == ".blend" and p.exists():
        return p
    aday = Path("Varliklar") / hedef / "varlik.blend"
    return aday if aday.exists() else None


def render(hedef: str, *, cozunurluk: int = 512, kare: int = 1,
           zaman: int = 600) -> Dict[str, Any]:
    """Varlığı 4 açıdan render eder."""
    if not BLENDER.exists():
        return {"hata": "blender yok"}
    blend = _blend_bul(hedef)
    if blend is None:
        return {"hata": f"varlık bulunamadı: {hedef}"}
    cikti = blend.parent / "onizleme"
    kod = (f'BLEND_YOLU = r"{blend.resolve()}"\n'
           f'CIKTI_V = r"{cikti.resolve()}"\n'
           f'COZUNURLUK_V = {int(cozunurluk)}\n'
           f'KARE_V = {int(kare)}\n') + RENDER
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "r.py"
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
        if "P{" in satir:
            try:
                d = json.loads(satir[satir.index("P{") + 1:])
                d["sure_s"] = round(sure, 2)
                d["hedef"] = hedef
                d["klasor"] = str(cikti)
                return d
            except Exception:
                pass
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:170] or "çıktı yok"}


def tumu(*, cozunurluk: int = 512) -> Dict[str, Any]:
    """Tüm varlıkları render eder."""
    from tools.varlik_katalogu import listele

    sonuc = {"varlik": [], "basarili": 0, "toplam": 0}
    for v in listele():
        b = Path(v.get("blend") or "")
        if not b.exists():
            continue
        sonuc["toplam"] += 1
        d = render(str(b), cozunurluk=cozunurluk)
        d["id"] = v.get("id")
        sonuc["varlik"].append(d)
        if not d.get("hata") and d.get("basarili", 0) >= 4:
            sonuc["basarili"] += 1
    return sonuc


def main() -> int:
    """Önizleme."""
    ap = argparse.ArgumentParser(description="Varlık önizleme")
    ap.add_argument("--render", default="", help="Varlık kimliği")
    ap.add_argument("--tumu", action="store_true")
    ap.add_argument("--cozunurluk", type=int, default=512)
    args = ap.parse_args()

    print("=" * 78)
    print("  VARLIK ÖNİZLEME — 4 açıdan render")
    print("=" * 78)

    if args.render:
        d = render(args.render, cozunurluk=args.cozunurluk)
        if "hata" in d:
            print(f"  ✗ {d['hata']}")
            return 1
        print(f"\n  ✓ {d['hedef']} · {d['sure_s']} sn · motor={d.get('motor')}")
        print(f"    boyut  : {d.get('boyut')}")
        print(f"    merkez : {d.get('merkez')}")
        print(f"    ışık   : {d.get('isik')} · kamera: {d.get('kamera')}")
        for r in d.get("render", []):
            if r.get("var"):
                print(f"    ✓ {r['aci']:8} {r['kb']:>7} KB")
            else:
                print(f"    ✗ {r['aci']:8} {r.get('hata', 'yok')}")
        if d.get("uyari"):
            print(f"    ⚠ {d['uyari'][:3]}")
        print(f"    klasör : {d.get('klasor')}")
        Path("_onizleme.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    if args.tumu:
        s = tumu(cozunurluk=args.cozunurluk)
        print(f"\n  {s['basarili']}/{s['toplam']} varlık render edildi")
        for d in s["varlik"]:
            if "hata" in d:
                print(f"    ✗ {d.get('id')}: {d['hata'][:50]}")
            else:
                print(f"    ✓ {d.get('id', '?'):30} "
                      f"{d.get('basarili', 0)}/4 açı · {d['sure_s']} sn")
        Path("_onizleme_tum.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
