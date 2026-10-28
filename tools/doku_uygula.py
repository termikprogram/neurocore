"""
Doku Uygulayıcı — Bul, Seç, Uygula (Hedef B)
==============================================
⚠ Kullanıcı hedefi
------------------
*"mesh oluşturma, material oluşturma, **texture ekleme**, texture bulup
**seçime sunma**"*

⚠ Mevcut durum (tur 11 başı)
----------------------------
```
tools/doku_bulucu.py    : ara + puanla + SUN        ✓
tools/doku_dogrula.py   : 7 prosedürel dokuyu derle  ✓ (7/7)
YEREL KÜTÜPHANE         : BOŞ (0 dosya)             ✗
VARLIĞA UYGULAMA        : YOK                        ✗ ← bu betik
```

Bu betik
--------
```
1. `doku_bulucu` ile **ara + puanla**
2. Bulunanı varlığın malzemesine **uygula**
3. Doğrula: düğüm sayısı, bağlantı, UV
```

⚠ İki doku kaynağı
------------------
```
YEREL     : dosya (PNG/JPG) → `ShaderNodeTexImage` + UV ŞART
PROSEDÜREL: düğüm tarifi    → `ShaderNodeTexNoise/Wave/...` (UV gerekmez)
```

⚠ Ölçülen kısıt: çevrimiçi arama ikili dosya indiremez →
yerel kütüphane kullanıcı tarafından doldurulur; sistem **puanlayıp
sunmakla** yükümlüdür.

Kullanım
--------
    python tools/doku_uygula.py --ara "kaya pürüzlü"
    python tools/doku_uygula.py --uygula kaya_0001 --doku kaya
    python tools/doku_uygula.py --onizleme
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


#: ⚠ tur 12: Önceki doku düğümlerini TEMİZLE (birikimi önler).
#:
#: ÖLÇÜM (tur 12): aynı varlığa 4 kez doku uygulanınca düğüm sayısı
#: 14 → 16 → 18 → 20 oldu. Sebep: her uygulama **ekliyor**,
#: değiştirmiyor. Doğru davranış: **değiştir**.
TEMIZLE_KOD = """
# ⚠ Doku düğümlerini temizle (BSDF ve Output KORUNUR)
_KORU = {"ShaderNodeBsdfPrincipled", "ShaderNodeOutputMaterial",
         "ShaderNodeMixShader", "ShaderNodeBsdfDiffuse",
         "ShaderNodeBsdfGlossy", "ShaderNodeBsdfTransparent",
         "ShaderNodeBsdfEmission", "ShaderNodeBsdfHair",
         "ShaderNodeBsdfRefraction", "ShaderNodeBsdfToon",
         "ShaderNodeBsdfVelvet", "ShaderNodeBsdfSheen"}
_silinen = 0
for _n in list(nt.nodes):
    if _n.bl_idname in _KORU:
        continue
    try:
        nt.nodes.remove(_n)
        _silinen += 1
    except Exception:
        pass
olcum["temizlenen"] = _silinen
"""


#: ⚠ Prosedürel dokuyu varlığa uygular
#:
#: `PROSEDUREL` tarifindeki düğümleri kurar, BSDF'e bağlar ve kaydeder.
UYGULA_PROSEDUREL = r'''
import bpy, json

bpy.ops.wm.open_mainfile(filepath=BLEND_YOLU)

TARIF = TARIF_V          # {"dugumler": [[tip, {giris: deger}], ...],
                         #  "renk": [r,g,b], "puruzluluk": x, "metalik": y}
AD = AD_V

olcum = {"dugum": 0, "baglanti": 0, "uyari": [], "malzeme": "",
         "onceki_dugum": 0, "temizlenen": 0}

# ⚠ Hedef mesh (ilk mesh nesnesi)
hedef = next((o for o in bpy.data.objects if o.type == "MESH"), None)
if hedef is None:
    print("U" + json.dumps({"hata": "mesh yok"}))
    raise SystemExit(0)

me = hedef.data
# ⚠ Mevcut malzemeyi kullan ya da yenisini oluştur
if me.materials and me.materials[0]:
    mat = me.materials[0]
    olcum["onceki_dugum"] = len(mat.node_tree.nodes) if mat.use_nodes else 0
else:
    mat = bpy.data.materials.new(name=f"{AD}_Malzeme")
    mat.use_nodes = True
    me.materials.append(mat)
if not mat.use_nodes:
    mat.use_nodes = True
nt = mat.node_tree
olcum["malzeme"] = mat.name

bsdf = nt.nodes.get("Principled BSDF")
if bsdf is None:
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"),
                None)

if bsdf:
    renk = TARIF.get("renk") or [0.5, 0.5, 0.5]
    try:
        bsdf.inputs["Base Color"].default_value = (
            float(renk[0]), float(renk[1]), float(renk[2]), 1.0)
    except Exception as exc:
        olcum["uyari"].append(f"renk: {type(exc).__name__}")
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = float(
            TARIF.get("puruzluluk", 0.8))
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = float(
            TARIF.get("metalik", 0.0))
    for a in ("Specular IOR Level", "Specular"):
        if a in bsdf.inputs:
            bsdf.inputs[a].default_value = 0.15
            break

# ⚠ tur 12: ÖNCE temizle (birikimi önler)
if bsdf is not None:
    _KORU = {"ShaderNodeBsdfPrincipled", "ShaderNodeOutputMaterial",
             "ShaderNodeMixShader"}
    for _n in list(nt.nodes):
        if _n.bl_idname in _KORU:
            continue
        try:
            nt.nodes.remove(_n)
            olcum["temizlenen"] += 1
        except Exception:
            pass

# ⚠ Düğümleri kur
eklenen = []
for tip, params in TARIF.get("dugumler", []):
    try:
        d = nt.nodes.new(tip)
    except Exception as exc:
        olcum["uyari"].append(f"{tip}: {type(exc).__name__}")
        continue
    for pk, pv in (params or {}).items():
        if pk in d.inputs:
            try:
                d.inputs[pk].default_value = pv
            except Exception:
                olcum["uyari"].append(f"{tip}.{pk}: atanamadı")
        else:
            olcum["uyari"].append(f"{tip}.{pk}: giriş yok")
    eklenen.append(d)
olcum["dugum"] = len(eklenen)

# ⚠ Bağla: doku → Bump → Normal · ikinci doku → Roughness
if bsdf and eklenen:
    bump = next((d for d in eklenen if d.bl_idname == "ShaderNodeBump"), None)
    dokular = [d for d in eklenen if d is not bump]
    if bump and dokular:
        d0 = dokular[0]
        cikis = "Fac" if "Fac" in d0.outputs else (
            "Color" if "Color" in d0.outputs else "Val")
        try:
            nt.links.new(d0.outputs[cikis], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
            olcum["baglanti"] += 2
        except Exception as exc:
            olcum["uyari"].append(f"bump bağlantı: {type(exc).__name__}")
    if len(dokular) > 1 and "Roughness" in bsdf.inputs:
        d1 = dokular[1]
        cikis = "Fac" if "Fac" in d1.outputs else (
            "Color" if "Color" in d1.outputs else "Val")
        try:
            nt.links.new(d1.outputs[cikis], bsdf.inputs["Roughness"])
            olcum["baglanti"] += 1
        except Exception:
            pass
    if not bump and dokular and "Roughness" in bsdf.inputs:
        d0 = dokular[0]
        cikis = "Fac" if "Fac" in d0.outputs else (
            "Color" if "Color" in d0.outputs else "Val")
        try:
            nt.links.new(d0.outputs[cikis], bsdf.inputs["Roughness"])
            olcum["baglanti"] += 1
        except Exception:
            pass

olcum["dugum_toplam"] = len(nt.nodes)
olcum["baglanti_toplam"] = len(nt.links)
olcum["uv_katman"] = len(me.uv_layers)
olcum["vertex"] = len(me.vertices)

bpy.ops.wm.save_as_mainfile(filepath=BLEND_YOLU)
bpy.ops.export_scene.gltf(
    filepath=str(GLB_YOLU), export_format="GLB")
print("U" + json.dumps(olcum, ensure_ascii=False))
'''


#: ⚠ Görüntü dokusunu (dosya) uygular
UYGULA_GORUNTU = r'''
import bpy, json

bpy.ops.wm.open_mainfile(filepath=BLEND_YOLU)

DOSYA = DOSYA_V
TUR = TUR_V
AD = AD_V
KANAL = KANAL_V      # renk | puruzluluk | normal | ...

olcum = {"dugum": 0, "baglanti": 0, "uyari": [], "malzeme": "",
         "goruntu": "", "boyut": [], "temizlenen": 0}

hedef = next((o for o in bpy.data.objects if o.type == "MESH"), None)
if hedef is None:
    print("U" + json.dumps({"hata": "mesh yok"}))
    raise SystemExit(0)

me = hedef.data
# ⚠ Görüntü dokusu UV ŞART — yoksa uyarı
if not me.uv_layers:
    olcum["uyari"].append("UV katmanı YOK — görüntü doku görünmez")

if me.materials and me.materials[0]:
    mat = me.materials[0]
else:
    mat = bpy.data.materials.new(name=f"{AD}_Malzeme")
    me.materials.append(mat)
mat.use_nodes = True
nt = mat.node_tree
olcum["malzeme"] = mat.name

try:
    img = bpy.data.images.load(str(DOSYA))
    olcum["goruntu"] = img.name
    olcum["boyut"] = [img.size[0], img.size[1]]
except Exception as exc:
    print("U" + json.dumps({"hata": f"goruntu yuklenemedi: "
                                    f"{type(exc).__name__}"}))
    raise SystemExit(0)

# ⚠ tur 12: ÖNCE temizle (birikimi önler)
_KORU = {"ShaderNodeBsdfPrincipled", "ShaderNodeOutputMaterial",
         "ShaderNodeMixShader"}
for _n in list(nt.nodes):
    if _n.bl_idname in _KORU:
        continue
    try:
        nt.nodes.remove(_n)
        olcum["temizlenen"] += 1
    except Exception:
        pass

tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = img
tex.location = (-620, -180)
olcum["dugum"] = 1

bsdf = nt.nodes.get("Principled BSDF") or next(
    (n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
if bsdf is None:
    olcum["uyari"].append("Principled BSDF yok")

def _bagla(cikis_adi, giris_adi):
    global olcum
    if bsdf is None:
        return
    if giris_adi not in bsdf.inputs:
        olcum["uyari"].append(f"{giris_adi}: giriş yok")
        return
    try:
        nt.links.new(tex.outputs[cikis_adi], bsdf.inputs[giris_adi])
        olcum["baglanti"] += 1
    except Exception as exc:
        olcum["uyari"].append(f"{giris_adi}: {type(exc).__name__}")

if KANAL == "renk":
    _bagla("Color", "Base Color")
elif KANAL == "pürüzlülük":
    _bagla("Color", "Roughness")
elif KANAL == "metal":
    _bagla("Color", "Metallic")
elif KANAL == "normal":
    # ⚠ Normal map: ayrı düğüm gerekir
    try:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-320, -180)
        nt.links.new(tex.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
        olcum["dugum"] += 1
        olcum["baglanti"] += 2
    except Exception as exc:
        olcum["uyari"].append(f"normal map: {type(exc).__name__}")
elif KANAL == "yükseklik":
    try:
        bp = nt.nodes.new("ShaderNodeBump")
        bp.location = (-320, -180)
        nt.links.new(tex.outputs["Color"], bp.inputs["Height"])
        nt.links.new(bp.outputs["Normal"], bsdf.inputs["Normal"])
        olcum["dugum"] += 1
        olcum["baglanti"] += 2
    except Exception as exc:
        olcum["uyari"].append(f"bump: {type(exc).__name__}")
else:
    _bagla("Color", "Base Color")

olcum["dugum_toplam"] = len(nt.nodes)
olcum["baglanti_toplam"] = len(nt.links)
olcum["uv_katman"] = len(me.uv_layers)
olcum["vertex"] = len(me.vertices)

bpy.ops.wm.save_as_mainfile(filepath=BLEND_YOLU)
bpy.ops.export_scene.gltf(filepath=str(GLB_YOLU), export_format="GLB")
print("U" + json.dumps(olcum, ensure_ascii=False))
'''


# ══ ⚠ tur 25: ORTAK yol çözümleyici ═══════════════════════════════════
# Neden: 5 araçta **aynı** yol hatası vardı —
#   `Path("Varliklar") / hedef` → hedef zaten `Varliklar\...` içeriyorsa
#   **çift** birleşiyordu (`Varliklar\Varliklar\...`).
def _ortak_blend_bul(hedef: str):
    """Ortak çözümleyici (tam yol · klasör · göreli · ad · katalog)."""
    try:
        from tools.varlik_yolu import blend_bul

        return blend_bul(hedef)
    except Exception:
        return None


def _blend_bul(hedef: str) -> Optional[Path]:
    """
    Varlık `.blend` yolunu bulur.

    ⚠ tur 25: **ortak** çözümleyiciye delege edilir.
    """
    _o = _ortak_blend_bul(hedef)
    if _o is not None:
        return _o
    p = Path(hedef)
    if p.suffix == ".blend" and p.exists():
        return p
    aday = Path("Varliklar") / hedef / "varlik.blend"
    return aday if aday.exists() else None


def uygula_prosedurel(hedef: str, doku: str, *,
                      zaman: int = 300) -> Dict[str, Any]:
    """Prosedürel dokuyu varlığa uygular."""
    from tools.doku_bulucu import PROSEDUREL

    if doku not in PROSEDUREL:
        return {"hata": f"bilinmeyen doku: {doku}",
                "gecerli": list(PROSEDUREL.keys())}
    if not BLENDER.exists():
        return {"hata": "blender yok"}
    blend = _blend_bul(hedef)
    if blend is None:
        return {"hata": f"varlık bulunamadı: {hedef}"}
    tarif = PROSEDUREL[doku]
    glb = blend.parent / "varlik.glb"
    kod = (f'BLEND_YOLU = r"{blend.resolve()}"\n'
           f'GLB_YOLU = r"{glb.resolve()}"\n'
           f'AD_V = {doku!r}\n'
           f'TARIF_V = {json.dumps(tarif)}\n') + UYGULA_PROSEDUREL

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
        if "U{" in satir:
            try:
                d = json.loads(satir[satir.index("U{") + 1:])
                d["sure_s"] = round(sure, 2)
                d["doku"] = doku
                d["hedef"] = hedef
                d["blend"] = str(blend)
                return d
            except Exception:
                pass
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:170] or "çıktı yok"}


def uygula_goruntu(hedef: str, dosya: str, kanal: str = "renk",
                   *, zaman: int = 300) -> Dict[str, Any]:
    """Yerel görüntü dokusunu varlığa uygular."""
    if not BLENDER.exists():
        return {"hata": "blender yok"}
    blend = _blend_bul(hedef)
    if blend is None:
        return {"hata": f"varlık bulunamadı: {hedef}"}
    img = Path(dosya)
    if not img.exists():
        return {"hata": f"doku dosyası yok: {img}"}
    glb = blend.parent / "varlik.glb"
    kod = (f'BLEND_YOLU = r"{blend.resolve()}"\n'
           f'GLB_YOLU = r"{glb.resolve()}"\n'
           f'DOSYA_V = r"{img.resolve()}"\n'
           f'TUR_V = {kanal!r}\nKANAL_V = {kanal!r}\n'
           f'AD_V = {img.stem!r}\n') + UYGULA_GORUNTU
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
    out = (p.stdout or "") + (p.stderr or "")
    for satir in out.splitlines():
        if "U{" in satir:
            try:
                d = json.loads(satir[satir.index("U{") + 1:])
                d["doku"] = img.name
                d["kanal"] = kanal
                d["hedef"] = hedef
                return d
            except Exception:
                pass
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:170] or "çıktı yok"}


def onizleme() -> Dict[str, Any]:
    """
    Doku seçeneklerini **sunuma** hazırlar.

    ⚠ Kullanıcı: *"texture bulup **seçime sunma**"*
    """
    from tools.doku_bulucu import KAYNAKLAR, PROSEDUREL, ara, tara

    t = tara(kaydet=False)
    return {
        "yerel": {"bulunan": t["bulunan"]},
        "prosedurel": [
            {"ad": ad, "aciklama": b["aciklama"],
             "dugum": len(b["dugumler"]),
             "renk": b["renk"], "puruzluluk": b["puruzluluk"]}
            for ad, b in PROSEDUREL.items()],
        "kaynaklar": [{"ad": s["ad"], "lisans": s["lisans"],
                       "url": s["url"]} for s in KAYNAKLAR],
    }


def main() -> int:
    """Doku uygulama."""
    ap = argparse.ArgumentParser(description="Doku bul/seç/uygula")
    ap.add_argument("--ara", default="")
    ap.add_argument("--onizleme", action="store_true")
    ap.add_argument("--uygula", default="", help="Varlık kimliği")
    ap.add_argument("--doku", default="", help="Prosedürel doku adı")
    ap.add_argument("--dosya", default="", help="Görüntü doku yolu")
    ap.add_argument("--kanal", default="renk")
    args = ap.parse_args()

    print("=" * 78)
    print("  DOKU — bul, seç, UYGULA")
    print("=" * 78)

    if args.onizleme:
        d = onizleme()
        print(f"\n  YEREL KÜTÜPHANE: {d['yerel']['bulunan']} dosya")
        print(f"\n  PROSEDÜREL ({len(d['prosedurel'])}):")
        for x in d["prosedurel"]:
            print(f"    {x['ad']:10} {x['aciklama']:30} "
                  f"{x['dugum']} düğüm · pürüz={x['puruzluluk']}")
        print(f"\n  ÇEVRİMİÇİ KAYNAK ({len(d['kaynaklar'])}):")
        for s in d["kaynaklar"][:4]:
            print(f"    · {s['ad']:14} {s['lisans']:22} {s['url']}")
        return 0

    if args.ara:
        from tools.doku_bulucu import sun

        sun(args.ara)
        return 0

    if args.uygula and (args.doku or args.dosya):
        if args.dosya:
            r = uygula_goruntu(args.uygula, args.dosya, args.kanal)
        else:
            r = uygula_prosedurel(args.uygula, args.doku)
        if "hata" in r:
            print(f"  ✗ {r['hata']}")
            return 1
        print(f"\n  ✓ {r['hedef']} ← {r.get('doku')}")
        for k in ("malzeme", "dugum", "baglanti", "dugum_toplam",
                  "baglanti_toplam", "uv_katman", "vertex"):
            print(f"    {k:16} {r.get(k)}")
        if r.get("uyari"):
            print(f"    ⚠ uyarılar: {r['uyari'][:3]}")
        Path("_doku_uygula.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
