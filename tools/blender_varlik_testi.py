"""
Blender Varlık Üretimi — Yetenek Ölçümü (temel)
=================================================
⚠ Amaç
------
Kullanıcı hedefi: *"low-poly veya kaliteli kaya oluştur dediğimizde ona
göre işlem yapacak"*

Bu betik **temel yeteneği ölçer**: Blender gerçekten çalışıyor mu,
mesh + material + UV üretilebiliyor mu, çıktı doğrulanabiliyor mu?

⚠ Ölçüm ilkeleri (40 turluk çalışmadan):
    1. **Gerçek Blender** — `bpy` betiği headless çalıştırılır
    2. **Çıktı doğrulanır** — dosya var mı, sahne içeriği doğru mu
    3. **Meta veri kaydedilir** — vertex/face/UV sayıları

Kullanım
--------
    python tools/blender_varlik_testi.py --olcum
    python tools/blender_varlik_testi.py --kaya
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

#: Çıktı kökü
CIKTI_KOK = Path("Varliklar")


#: ─── Blender betikleri ───────────────────────────────────────────────

#: ⚠ Low-poly kaya — `bpy` ile prosedürel
KAYA_LOWPOLY = '''
import bpy, bmesh, math, random, sys
from mathutils import Vector

# ── Sahneyi temizle ──
bpy.ops.wm.read_factory_settings(use_empty=True)

# ── Tohum ──
random.seed(SEED)

# ── Ico sphere → düşük poligon ──
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
ob = bpy.context.active_object
ob.name = "Kaya_LowPoly"

# ── Vertex'leri rastgele boz (kaya görünümü) ──
me = ob.data
for v in me.vertices:
    # Her eksende bağımsız bozulma + hafif ölçek farkı
    v.co.x *= 1.0 + random.uniform(-BOZULMA, BOZULMA)
    v.co.y *= 1.0 + random.uniform(-BOZULMA, BOZULMA)
    v.co.z *= 0.65 + random.uniform(-BOZULMA * 0.5, BOZULMA * 0.5)
    # Yüzeyi pürüzlendir
    n = v.co.normalized()
    v.co += n * random.uniform(-0.06, 0.06)

# ── Flat shading (low-poly görünüm) ──
for p in me.polygons:
    p.use_smooth = False

# ── Malzeme ──
mat = bpy.data.materials.new(name="Kaya_Malzeme")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (RENK_R, RENK_G, RENK_B, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.15
ob.data.materials.append(mat)

# ── UV aç (smart project) ──
bpy.ops.object.select_all(action="DESELECT")
ob.select_set(True)
bpy.context.view_layer.objects.active = ob
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
try:
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)
    UV_OK = True
except Exception:
    UV_OK = False
bpy.ops.object.mode_set(mode="OBJECT")

# ── Ölçümleri topla ──
olcum = {
    "nesne": ob.name,
    "vertex": len(me.vertices),
    "kenar": len(me.edges),
    "yuz": len(me.polygons),
    "uv_katman": len(me.uv_layers),
    "uv_ok": UV_OK,
    "malzeme": [m.name for m in me.materials],
    "boyut": [round(x, 4) for x in ob.dimensions],
    "duz_shading": all(not p.use_smooth for p in me.polygons),
}

# ── Kaydet ──
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
bpy.ops.export_scene.gltf(filepath=str(OUT_GLB), export_format="GLB")

print("OLCUM:" + __import__("json").dumps(olcum, ensure_ascii=False))
'''

#: ⚠ Kaliteli kaya — subdivision + bevel + pürüzlü yüzey
KAYA_KALITELI = '''
import bpy, bmesh, math, random, json
from mathutils import Vector, noise

bpy.ops.wm.read_factory_settings(use_empty=True)
random.seed(SEED)

# ── Küp → alt bölme → şekillendirme (kaliteli kaya) ──
bpy.ops.mesh.primitive_cube_add(size=1.0)
ob = bpy.context.active_object
ob.name = "Kaya_Kaliteli"

# ── Subdivision (yüzey yoğunluğu) ──
mod = ob.modifiers.new(name="AltBolme", type="SUBSURF")
mod.subdivision_type = "SIMPLE"
mod.levels = 3
bpy.ops.object.modifier_apply(modifier=mod.name)

# ── Prosedürel gürültü ile şekillendir ──
me = ob.data
for v in me.vertices:
    n = v.co.normalized() if v.co.length > 0 else Vector((0, 0, 1))
    # Çok oktavlı gürültü (kaya yüzeyi)
    g = 0.0
    for octav, genlik in ((2.0, 1.0), (4.0, 0.5), (8.0, 0.25)):
        g += noise.noise(v.co * octav) * genlik
    v.co += n * g * PURUZLULUK
    # Hafif basıklık
    v.co.z *= 0.7

# ── Yumuşak gölgeleme (kaliteli görünüm) ──
for p in me.polygons:
    p.use_smooth = True

# ── Bevel (kenar yumuşatma) ──
bev = ob.modifiers.new(name="KenarYumusat", type="BEVEL")
bev.width = 0.012
bev.segments = 2
bev.limit_method = "ANGLE"
bpy.ops.object.modifier_apply(modifier=bev.name)

# ── UV aç ──
bpy.ops.object.select_all(action="DESELECT")
ob.select_set(True)
bpy.context.view_layer.objects.active = ob
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
uv_ok = True
try:
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)
except Exception:
    uv_ok = False
bpy.ops.object.mode_set(mode="OBJECT")
me = ob.data

# ── Malzeme (düğüm tabanlı, gürültü dokulu) ──
mat = bpy.data.materials.new(name="Kaya_Kaliteli_Malzeme")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (RENK_R, RENK_G, RENK_B, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    # Gürültü dokusu → pürüzlülük (prosedürel texture)
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.location = (-500, -200)
    tex.inputs["Scale"].default_value = 8.0
    tex.inputs["Detail"].default_value = 6.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-250, -200)
    bump.inputs["Strength"].default_value = 0.25
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
ob.data.materials.append(mat)

olcum = {
    "nesne": ob.name,
    "vertex": len(me.vertices),
    "kenar": len(me.edges),
    "yuz": len(me.polygons),
    "uv_katman": len(me.uv_layers),
    "uv_ok": uv_ok,
    "malzeme": [m.name for m in me.materials],
    "dugum_sayisi": len(mat.node_tree.nodes),
    "boyut": [round(x, 4) for x in ob.dimensions],
    "duz_shading": all(not p.use_smooth for p in me.polygons),
}

bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
bpy.ops.export_scene.gltf(filepath=str(OUT_GLB), export_format="GLB")
print("OLCUM:" + json.dumps(olcum, ensure_ascii=False))
'''


def _kategori(ad: str) -> str:
    """
    Ad'dan kategori türetir.

    ⚠ tur 7 ÖLÇÜMÜ: eksik liste → `cali`, `cim` **esya**'ya düştü;
    `zemin`, `sutun`, `kapi` da. Liste `varlik_uret.URUNLER` ile
    **tutarlı** olmalı.
    """
    a = ad.lower().replace("-", "_")
    kelimeler = set(a.split("_"))

    def _var(*desenler: str) -> bool:
        return any(d in a for d in desenler)

    if kelimeler & {"kaya", "tas", "tasi", "mineral"} or _var("kaya"):
        return "kaya"
    if (kelimeler & {"agac", "cali", "cim", "bitki", "cicek", "ot",
                     "yaprak", "mantar", "sarmasik"}
            or _var("agac", "cali", "cim_", "bitki", "cicek")):
        return "bitki"
    if (kelimeler & {"timsah", "yaratik", "canavar", "ejder", "kurt",
                     "kus", "balik", "orumcek", "yilan", "dinozor"}
            or _var("timsah", "yarat", "canavar", "ejder")):
        return "yaratik"
    if kelimeler & {"karakter", "insan", "kahraman", "npc", "dusman"} \
            or _var("karakter", "kahraman"):
        return "karakter"
    if (kelimeler & {"duvar", "kapi", "sutun", "zemin", "tavan",
                     "merdiven", "kopru", "kule", "cati", "pencere",
                     "temel", "karo", "tugla"}
            or _var("duvar", "kapi", "sutun", "zemin", "merdiven",
                    "kule")):
        return "mimari"
    if kelimeler & {"arac", "araba", "tekne", "gemi", "ucak", "vagon"} \
            or _var("araba", "arac"):
        return "arac"
    return "esya"


def _kalite(ad: str) -> str:
    """Ad'dan kalite kademesi türetir."""
    a = ad.lower()
    if "lowpoly" in a or "low_poly" in a:
        return "lowpoly"
    if "kaliteli" in a or "yuksek" in a:
        return "kaliteli"
    return "orta"


def _varlik_yolu(ad: str) -> Path:
    """
    Varlığın **tek** klasör yolunu üretir.

    ⚠ Yapı: `Varliklar/<kategori>/<ad>_<kalite>/`
    Aynı ad + kalite tekrarında **sayaç** eklenir (çakışma olmasın).
    """
    # ⚠ tur 1 ÖLÇÜMÜ: eski sürüm var olan klasöre `_02`, `_03` ekliyordu
    # → aynı varlık her çağrıda **yeni klasör** → katalogda tekrarlı
    # kayıt (10 kayıt / 6 dosya). Artık **deterministik**: ad zaten
    # kaliteyi içeriyor, tekrar üretim **üzerine yazar**.
    return CIKTI_KOK / _kategori(ad) / f"{ad}_{_kalite(ad)}"


@dataclass
class UretimSonuc:
    """Tek üretim sonucu."""
    ad: str = ""
    tur: str = ""
    basarili: bool = False
    olcum: Dict[str, Any] = field(default_factory=dict)
    sure_s: float = 0.0
    blend: str = ""
    glb: str = ""
    hata: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _calistir(kod: str, ad: str, *, zaman: int = 180) -> UretimSonuc:
    """
    Blender betiğini **headless** çalıştırır ve ölçümü toplar.

    ⚠ Ölçüm: `OLCUM:` ön ekli satır JSON olarak ayrıştırılır.
    """
    s = UretimSonuc(ad=ad)
    if not BLENDER.exists():
        s.hata = "blender yok"
        return s
    # ⚠ tur 1 ÖLÇÜMÜ: iki farklı `hedef` hesabı vardı —
    # `mkdir` bir yolu, betiğe yazılan `OUT_BLEND` **başka** bir yolu
    # kullanıyordu → `Cannot open file ...` (klasör yoktu).
    # Çözüm: **tek** yol hesabı (`_varlik_yolu`).
    hedef = _varlik_yolu(ad)
    hedef.mkdir(parents=True, exist_ok=True)
    s.blend = str(hedef / "varlik.blend")
    s.glb = str(hedef / "varlik.glb")

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        betik = Path(td) / "uret.py"
        betik.write_text(kod, encoding="utf-8")
        try:
            p = subprocess.run(
                [str(BLENDER), "--background", "--factory-startup",
                 "--python", str(betik)],
                capture_output=True, text=True, timeout=zaman,
                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            s.hata = "zaman aşımı"
            return s
        except Exception as exc:
            s.hata = f"{type(exc).__name__}: {str(exc)[:60]}"
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
    if not s.basarili:
        # Hata izini sakla
        izler = [x.strip() for x in out.splitlines()
                 if "Error" in x or "Traceback" in x or "error" in x]
        s.hata = "; ".join(izler[:2])[:180] if izler else "OLCUM satırı yok"
        s.hata = s.hata or "bilinmeyen hata"
    # Dosya doğrulaması
    s.olcum["blend_var"] = Path(s.blend).exists()
    s.olcum["glb_var"] = Path(s.glb).exists()
    if s.olcum["blend_var"]:
        s.olcum["blend_kb"] = round(
            Path(s.blend).stat().st_size / 1024, 1)
    if s.olcum["glb_var"]:
        s.olcum["glb_kb"] = round(Path(s.glb).stat().st_size / 1024, 1)
    return s


def main() -> int:
    """Ölçümü çalıştırır."""
    ap = argparse.ArgumentParser(description="Blender varlık yetenek testi")
    ap.add_argument("--olcum", action="store_true")
    ap.add_argument("--kaya", action="store_true")
    ap.add_argument("--tohum", type=int, default=42)
    args = ap.parse_args()

    print("=" * 78)
    print("  BLENDER VARLIK ÜRETİMİ — YETENEK ÖLÇÜMÜ")
    print("=" * 78)
    print(f"  Blender: {BLENDER}")
    print(f"  var mı : {BLENDER.exists()}")

    if not (args.olcum or args.kaya):
        return 0

    testler: List[Tuple[str, str, Dict[str, Any]]] = [
        ("kaya_lowpoly", KAYA_LOWPOLY,
         {"SEED": args.tohum, "BOZULMA": 0.22,
          "RENK_R": 0.32, "RENK_G": 0.30, "RENK_B": 0.28}),
        ("kaya_kaliteli", KAYA_KALITELI,
         {"SEED": args.tohum, "PURUZLULUK": 0.18,
          "RENK_R": 0.34, "RENK_G": 0.31, "RENK_B": 0.29}),
    ]

    sonuclar: List[UretimSonuc] = []
    for ad, kalip, degisken in testler:
        kod = kalip
        # ⚠ Değişkenleri betiğe **başa** ekle (enjekte et)
        baslik = "\n".join(f"{k} = {v!r}" for k, v in degisken.items())
        kod = baslik + "\n" + kod
        # Çıktı yolları
        hedef = _varlik_yolu(ad).resolve()
        hedef.mkdir(parents=True, exist_ok=True)
        kod = (f'OUT_BLEND = r"{hedef / "varlik.blend"}"\n'
               f'OUT_GLB = r"{hedef / "varlik.glb"}"\n') + kod
        print(f"\n  [{ad}] çalıştırılıyor…", flush=True)
        s = _calistir(kod, ad)
        sonuclar.append(s)
        durum = "✓" if s.basarili else "✗"
        print(f"    {durum} {s.sure_s:.1f} sn"
              + (f" · {s.hata[:70]}" if s.hata else ""))
        for k, v in s.olcum.items():
            print(f"      {k:14} {v}")

    n = sum(1 for s in sonuclar if s.basarili)
    print(f"\n  {'=' * 74}")
    print(f"  BAŞARILI: {n}/{len(sonuclar)}")
    print(f"  {'=' * 74}")

    Path("_blender_varlik_testi.json").write_text(
        json.dumps({"sonuclar": [s.to_dict() for s in sonuclar],
                    "basarili": n, "toplam": len(sonuclar)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if n == len(sonuclar) else 1


if __name__ == "__main__":
    raise SystemExit(main())
