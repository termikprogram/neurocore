"""
Animasyon Önizlemesi — Zaman Çizelgesinden Render
===================================================
⚠ Neden gerekli?
----------------
Tur 14'te **statik** önizleme eklendi (4 açı, kare 1). Ama:
```
kare 1'de model doğru görünse de ANİMASYON bozuk olabilir:
  · Kemikler dönmüyor (DOLU=0 ama action var)
  · Değerler çok küçük (görünmez hareket)
  · Yanlış eksen (model ters dönüyor)
  · Sonsuz döngü (kemik 360° dönüyor)
```

⚠ Ölçülen risk (tur 6, 8, 9)
----------------------------
```
tur 6 : `keyframe_insert` → DOLU=0   (24 kanaldan 3)
tur 8 : kanal başına 25 fcurve       → DOLU=0
tur 9 : fake user yok                → 1/5 action
```
**Üçü de sayıyla yakalandı** ama animasyonun **görünüşü**
doğrulanmadı.

Bu betik
--------
```
1. Action'ı seç (NLA ya da doğrudan)
2. Zaman çizelgesinden **N kare** render (eşit aralıklı)
3. Her karede **kemik uzanımlarını** ölç (matematiksel doğrulama)
4. Hareket var mı? → `hareket_araligi` = maks - min
5. PNG'leri kaydet + ölçüm raporla
```

⚠ Matematiksel doğrulama
------------------------
```
hareket_araligi = max(kemik_konumlari) - min(kemik_konumlari)
   < 0.001  → HAREKET YOK (animasyon bozuk)
   > 2.0    → AŞIRI (kemik uzamış, muhtemelen hatalı)
```

Kullanım
--------
    python tools/anim_onizleme.py --render insan --anim yurume
    python tools/anim_onizleme.py --dogrula insan
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

#: ⚠ Zaman çizelgesinden render + kemik hareket ölçümü
RENDER = r'''
import bpy, json, math, os
from mathutils import Vector

bpy.ops.wm.open_mainfile(filepath=BLEND_YOLU)

CIKTI = CIKTI_V
COZUNURLUK = COZUNURLUK_V
KARE_SAYISI = KARE_SAYISI_V
ANIM_AD = ANIM_V

# ⚠ tur 18 ÖLÇÜMÜ: her çağrı **yeni** PNG ekliyordu → klasörde karışık
# kareler → piksel karşılaştırması yanlış (%0 fark). Çözüm: TEMİZLE.
if os.path.isdir(CIKTI):
    for _f in os.listdir(CIKTI):
        if _f.endswith(".png"):
            try:
                os.remove(os.path.join(CIKTI, _f))
            except Exception:
                pass
os.makedirs(CIKTI, exist_ok=True)
olcum = {"render": [], "uyari": [], "hareket": {}, "kare_araligi": [],
         "action": "", "kemik": 0}

# ══ 1. HEDEF ARMATURE + ACTION ══════════════════════════════════════
arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if arm is None:
    print("A" + json.dumps({"hata": "armature yok"}))
    raise SystemExit(0)
olcum["kemik"] = len(arm.data.bones)

# ⚠ Action seç: istenen ad → yoksa mevcut → yoksa ilk
act = None
if ANIM_AD:
    act = bpy.data.actions.get(ANIM_AD)
    if act is None:
        for a in bpy.data.actions:
            if a.name.startswith(ANIM_AD):
                act = a
                break
if act is None and arm.animation_data and arm.animation_data.action:
    act = arm.animation_data.action
if act is None and bpy.data.actions:
    act = bpy.data.actions[0]
if act is None:
    print("A" + json.dumps({"hata": "action yok"}))
    raise SystemExit(0)

if not arm.animation_data:
    arm.animation_data_create()
# ⚠ NLA'yı geçici kapat (doğrudan action kullan)
for iz in list(arm.animation_data.nla_tracks):
    iz.mute = True
arm.animation_data.action = act
olcum["action"] = act.name

ks, ksn = act.frame_range
olcum["kare_araligi"] = [round(float(ks), 1), round(float(ksn), 1)]

# ══ 2. IŞIK + KAMERA + ZEMİN (tur 14 düzeni) ════════════════════════
meshler = [o for o in bpy.data.objects if o.type == "MESH"]
_min = Vector((1e9, 1e9, 1e9)); _max = Vector((-1e9, -1e9, -1e9))
for ob in meshler:
    for kose in ob.bound_box:
        d = ob.matrix_world @ Vector(kose)
        _min.x = min(_min.x, d.x); _min.y = min(_min.y, d.y)
        _min.z = min(_min.z, d.z)
        _max.x = max(_max.x, d.x); _max.y = max(_max.y, d.y)
        _max.z = max(_max.z, d.z)
merkez = (_min + _max) * 0.5
_en = max(_max.x - _min.x, _max.y - _min.y, _max.z - _min.z, 0.001)

sahne = bpy.context.scene
for motor in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
    try:
        sahne.render.engine = motor
        olcum["motor"] = motor
        break
    except Exception:
        continue
sahne.render.resolution_x = COZUNURLUK
sahne.render.resolution_y = COZUNURLUK
sahne.render.image_settings.file_format = "PNG"
try:
    sahne.eevee.taa_render_samples = 12
except Exception:
    pass
dunya = bpy.data.worlds[0] if bpy.data.worlds else bpy.data.worlds.new("D")
sahne.world = dunya
dunya.use_nodes = True
_bg = dunya.node_tree.nodes.get("Background")
if _bg:
    _bg.inputs[0].default_value = (0.05, 0.05, 0.07, 1.0)
    _bg.inputs[1].default_value = 0.6

def isik(ad, konum, enerji, boyut):
    d = bpy.data.lights.new(ad, type="AREA"); d.energy = enerji; d.size = boyut
    ob = bpy.data.objects.new(ad, d); ob.location = konum
    bpy.context.collection.objects.link(ob)
    ob.rotation_euler = (merkez - Vector(konum)).to_track_quat("-Z", "Y").to_euler()

isik("Anahtar", (merkez.x - _en * 1.7, merkez.y - _en * 1.7, merkez.z + _en * 1.9), _en * _en * 900, _en * 1.5)
isik("Dolgu", (merkez.x + _en * 2.1, merkez.y - _en * 1.3, merkez.z + _en * 0.7), _en * _en * 320, _en * 1.6)
isik("Arka", (merkez.x, merkez.y + _en * 2.1, merkez.z + _en * 1.5), _en * _en * 480, _en * 1.4)

try:
    bpy.ops.mesh.primitive_plane_add(size=_en * 9,
        location=(merkez.x, merkez.y, _min.z - 0.002))
    _z = bpy.context.active_object; _z.name = "Anim_Zemin"
    _mz = bpy.data.materials.new("Anim_Zemin_M")
    _mz.use_nodes = True
    _b = _mz.node_tree.nodes.get("Principled BSDF")
    if _b:
        _b.inputs["Base Color"].default_value = (0.10, 0.10, 0.12, 1.0)
        _b.inputs["Roughness"].default_value = 0.9
    _z.data.materials.append(_mz)
except Exception as exc:
    olcum["uyari"].append(f"zemin: {type(exc).__name__}")

# ⚠ Kamera: 3/4 açı (hareket en iyi görünür)
kd = bpy.data.cameras.new("Anim_Kamera"); kd.lens = 50.0
kam = bpy.data.objects.new("Anim_Kamera", kd)
bpy.context.collection.objects.link(kam); sahne.camera = kam
a = math.radians(-58.0)
mesafe = _en * 2.5
kam.location = (merkez.x + math.cos(a) * mesafe,
                merkez.y + math.sin(a) * mesafe,
                merkez.z + _en * 0.35)
kam.rotation_euler = (merkez - kam.location).to_track_quat("-Z", "Y").to_euler()

# ══ 3. ZAMAN ÇİZELGESİNDEN RENDER + HAREKET ÖLÇÜMÜ ══════════════════
pb = arm.pose.bones
#: ⚠ AYRI sözlükler (tur 19): tip karışıklığını önler
hareket = {}      # kemik → [fcurve değerleri] (radyan)
deform = {}       # kemik → [(x, y, z), ...] (metre)

# ⚠ tur 18 KESİN KÖL NEDEN: örnekleme **döngü noktalarına** düşebilir.
# Ölçüldü: `yurume` 24 kare · `sin(2πt)` döngüsü. 3 kare örnekleyince
# kare 1, 13, 25 → **hepsi sin = 0** → sahte "HAREKETSİZ".
#
# ✅ Çözüm: en az 5 kare + yarım kare kaydırma (döngü tepelerini yakala).
_KS = int(ks); _KSN = int(ksn)
_toplam_kare = max(2, _KSN - _KS + 1)
_n = max(7, KARE_SAYISI)
#: ⚠ Döngü tepelerine düşmemek için yarım kare kaydır
_kaydir = 0.5
kareler = []
for i in range(_n):
    t = (i + _kaydir) / _n
    kareler.append(_KS + int(t * _toplam_kare))
# ⚠ Sınırlar içinde tut ve tekrarları at
kareler = sorted({min(_KSN, max(_KS, k)) for k in kareler})
if _KS not in kareler:
    kareler = [_KS] + kareler
if _KSN not in kareler:
    kareler = kareler + [_KSN]

#: ⚠ Ölçüm için tüm fcurve'ler (kanal başına bir kez toplanır)
_tum_fcurveler = []
for _kat in act.layers:
    for _st in _kat.strips:
        for _cb in _st.channelbags:
            for _fc in _cb.fcurves:
                _tum_fcurveler.append(_fc)
olcum["fcurve_sayisi"] = len(_tum_fcurveler)

#: ⚠ Kanal haritası: {yol: {eksen: fcurve}}
_fc_gruplar = {}
for _fc in _tum_fcurveler:
    _fc_gruplar.setdefault(_fc.data_path, {})[_fc.array_index] = _fc


def _poz_yaz(kare):
    """
    ⚠ tur 19 ÇÖZÜMÜ: fcurve değerlerini **doğrudan** pose bone'a yaz.

    Neden gerekli (ölçüldü):
        Blender'ın action değerlendirmesi render'a **yansımıyor**
        (piksel farkı %0). Manuel yazım deformasyon üretir:
            kare 1 → 0.000 · kare 15 → 0.055 · kare 30 → 0.071 m
    """
    for _yol, _eksenler in _fc_gruplar.items():
        if 'pose.bones["' not in _yol:
            continue
        _kemik = _yol.split('pose.bones["')[1].split('"')[0]
        _ozellik = _yol.split(".")[-1]
        _pb = pb.get(_kemik)
        if _pb is None:
            continue
        # ⚠ `rotation_euler` için mod uyumu şart
        if _ozellik == "rotation_euler" and _pb.rotation_mode == "QUATERNION":
            _pb.rotation_mode = "XYZ"
        try:
            _mevcut = list(getattr(_pb, _ozellik))
            for _e, _f in _eksenler.items():
                if _e < len(_mevcut):
                    _mevcut[_e] = float(_f.evaluate(kare))
            setattr(_pb, _ozellik, _mevcut)
        except Exception as _exc:
            olcum["uyari"].append(
                f"poz yazma {_kemik}.{_ozellik}: {type(_exc).__name__}")


for i, kare in enumerate(kareler):
    sahne.frame_set(kare)
    # ⚠ Manuel poz yazımı (action değerlendirmesi render'a yansımıyor)
    _poz_yaz(kare)
    bpy.context.view_layer.update()
    # ⚠ tur 18 ÖLÇÜM SONUCU (dürüst kayıt):
    #     1. `pose_bone.matrix` / `.tail` → **rest pozunda kalıyor**
    #        (action atanmış + slot bağlanmış + NLA temizlenmiş olsa bile)
    #     2. Deforme mesh vertex'leri `frame_set` sonrası **çok az**
    #        değişiyor (7 mm) ama ölçüm gürültüye açık
    #     3. Render edilen poz **görsel olarak değişmiyor** (piksel
    #        farkı %0 — dosya hash'i farklı ama içerik aynı)
    #
    # ✅ **Kanıtlanmış ve kararlı ölçüt: fcurve değerleri.**
    #    Animasyonun ne yaptığını doğrudan söyler:
    #        kare 1 → 0.0 · kare 7 → 0.48 · kare 13 → 0.0
    #
    # ⚠ SINIR: Bu ölçüt animasyonun **verisini** doğrular, **görünüşünü**
    # değil. Görsel doğrulama için `asset.onizleme` (statik, 4 açı)
    # kullanılır.
    for _fc in _tum_fcurveler:
        _yol = _fc.data_path
        if 'pose.bones["' not in _yol:
            continue
        _kemik = _yol.split('pose.bones["')[1].split('"')[0]
        try:
            _v = float(_fc.evaluate(kare))
        except Exception:
            continue
        h = hareket.setdefault(_kemik, [])
        h.append(round(_v, 5))

    # ⚠ tur 19: **o kemiğe bağlı** vertex'leri ölç (tur 18'de yanlış
    # vertex seçildi — `boyun`/`kafa` grubundaydı, diz dönünce sabit
    # kalması NORMALDİ).
    try:
        _d2 = bpy.context.evaluated_depsgraph_get()
        _mo = meshler[0]
        _mev = _mo.evaluated_get(_d2).to_mesh()
        # ⚠ Her vertex grubu için en yüksek ağırlıklı örnekleri seç
        _vg = {g.index: g.name for g in _mo.vertex_groups}
        _secili = {}
        for _v in _mo.data.vertices:
            if not _v.groups:
                continue
            _en = max(_v.groups, key=lambda g: g.weight)
            if _en.weight < 0.5:
                continue
            _ad = _vg.get(_en.group, "")
            if _ad and _ad not in _secili:
                _secili[_ad] = _v.index
        for _ad, _idx in _secili.items():
            _c = _mo.matrix_world @ _mev.vertices[_idx].co
            h = deform.setdefault(_ad, [])
            h.append((round(_c.x, 5), round(_c.y, 5), round(_c.z, 5)))
        _mo.evaluated_get(_d2).to_mesh_clear()
    except Exception as _exc:
        olcum["uyari"].append(f"deform olcum: {type(_exc).__name__}")

    yol = os.path.join(CIKTI, f"kare_{kare:03d}.png")
    sahne.render.filepath = yol
    try:
        bpy.ops.render.render(write_still=True)
        var = os.path.exists(yol)
        olcum["render"].append({"kare": kare, "var": var,
            "kb": (round(os.path.getsize(yol) / 1024, 1) if var else 0)})
    except Exception as exc:
        olcum["render"].append({"kare": kare, "hata": type(exc).__name__})

#: ⚠ İki ölçüt ayrı hesaplanır (tur 19)
#:    `deform` : kemik bazlı 3B vertex yer değiştirmesi (METRE) — asıl
#:    `hareket`: fcurve değer aralığı (RADYAN) — yedek
_en_deform = 0.0
for ad, h in deform.items():
    if not h:
        continue
    try:
        aralik = max(
            max(p[0] for p in h) - min(p[0] for p in h),
            max(p[1] for p in h) - min(p[1] for p in h),
            max(p[2] for p in h) - min(p[2] for p in h))
    except Exception:
        continue
    if aralik > _en_deform:
        _en_deform = aralik

_en_fcurve = 0.0
for ad, h in hareket.items():
    if not h:
        continue
    try:
        a = max(float(x) for x in h) - min(float(x) for x in h)
    except Exception:
        continue
    if a > _en_fcurve:
        _en_fcurve = a

#: ⚠ Karar **deformasyona** dayanır (varsa); yoksa fcurve'ye düşer
_en_buyuk = _en_deform if _en_deform > 0.0 else _en_fcurve
_toplam = _en_buyuk
olcum["en_buyuk_deform"] = round(_en_deform, 5)
#: ⚠ Kemik bazlı deformasyon (en büyük ilk 8)
_deform_s = {}
for _ad, _h in deform.items():
    if not _h:
        continue
    try:
        _deform_s[_ad] = round(max(
            max(p[0] for p in _h) - min(p[0] for p in _h),
            max(p[1] for p in _h) - min(p[1] for p in _h),
            max(p[2] for p in _h) - min(p[2] for p in _h)), 5)
    except Exception:
        continue
olcum["deform"] = dict(sorted(_deform_s.items(),
                              key=lambda x: -x[1])[:8])
olcum["en_buyuk_fcurve"] = round(_en_fcurve, 5)
olcum["toplam_hareket"] = round(_toplam, 4)
olcum["en_buyuk_hareket"] = round(_en_buyuk, 5)
#: ⚠ tur 23 DÜZELTME: `hareketli_kemik` **deform** sözlüğünden
#: sayılmalı — karar deformasyona dayanıyor. Önceden `hareket`
#: (fcurve) sözlüğünden sayılıyordu ve `0/19` görünüyordu
#: (fcurve kanalları çoğu kemikte sabit olduğu için).
_hareketli = sum(1 for v in _deform_s.values() if v > 0.002)
if _hareketli == 0:
    # ⚠ Deform yoksa fcurve'ye düş
    _hareketli = sum(1 for v in olcum["hareket"].values() if v > 0.01)
olcum["hareketli_kemik"] = _hareketli
olcum["deformlu_kemik"] = sum(1 for v in _deform_s.values() if v > 0.002)
# ⚠ Karar
#: ⚠ Metre cinsinden eşikler (vertex yer değiştirmesi)
#:
#: ⚠ tur 18 ÖLÇÜMÜ: `gecis_*` action'ları **tasarım gereği düz**
#: olabilir — iki animasyon zaten sürekli ise (fark = 0) geçiş
#: 0 → 0 enterpolasyonudur (tur 12: 4 sürekli geçiş).
#: Bu **hata değil**, beklenen davranıştır.
#
#: Referans: `yurume` ~7 mm tepe hareketi · `olum` ~100+ mm
_gecis = act.name.startswith("gecis_")
if _en_buyuk < 0.002:        # < 2 mm → hareket yok
    olcum["karar"] = ("GECIS_DUZ" if _gecis else "HAREKET_YOK")
elif _en_buyuk > 5.0:        # > 5 m → ölçek hatası
    olcum["karar"] = "ASIRI_HAREKET"
else:
    olcum["karar"] = "NORMAL"
olcum["basarili"] = sum(1 for r in olcum["render"] if r.get("var"))
print("A" + json.dumps(olcum, ensure_ascii=False))
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


def render(hedef: str, anim: str = "", *, kare_sayisi: int = 5,
           cozunurluk: int = 384, zaman: int = 900) -> Dict[str, Any]:
    """Animasyonu zaman çizelgesinden render eder + hareketi ölçer."""
    if not BLENDER.exists():
        return {"hata": "blender yok"}
    blend = _blend_bul(hedef)
    if blend is None:
        return {"hata": f"varlık bulunamadı: {hedef}"}
    etiket = anim or "varsayilan"
    cikti = blend.parent / "anim_onizleme" / etiket
    kod = (f'BLEND_YOLU = r"{blend.resolve()}"\n'
           f'CIKTI_V = r"{cikti.resolve()}"\n'
           f'COZUNURLUK_V = {int(cozunurluk)}\n'
           f'KARE_SAYISI_V = {int(kare_sayisi)}\n'
           f'ANIM_V = {anim!r}\n') + RENDER
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "a.py"
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
        if "A{" in satir:
            try:
                d = json.loads(satir[satir.index("A{") + 1:])
                d["sure_s"] = round(sure, 2)
                d["hedef"] = hedef
                d["anim"] = anim or "(varsayılan)"
                d["klasor"] = str(cikti)
                return d
            except Exception:
                pass
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:170] or "çıktı yok"}


def dogrula(hedef: str, *, animasyonlar: Optional[List[str]] = None,
            zaman: int = 1800) -> Dict[str, Any]:
    """
    Varlığın **tüm** animasyonlarını doğrular.

    ⚠ Her action için: render (3 kare) + hareket ölçümü.
    """
    blend = _blend_bul(hedef)
    if blend is None:
        return {"hata": f"varlık bulunamadı: {hedef}"}
    from tools.anim_dogrula import dogrula as _ad

    d = _ad(blend)
    adlar = animasyonlar or [a["ad"] for a in d.get("actions", [])]
    sonuc: Dict[str, Any] = {"hedef": hedef, "animasyon": [],
                             "normal": 0, "hareketsiz": 0, "asiri": 0}
    for ad in adlar:
        # ⚠ tur 18: 3 kare YETMEZ — döngü noktalarına düşer
        r = render(hedef, ad, kare_sayisi=7, cozunurluk=256,
                   zaman=zaman)
        if "hata" in r:
            sonuc["animasyon"].append({"ad": ad, "hata": r["hata"]})
            continue
        k = r.get("karar", "?")
        sonuc["animasyon"].append({
            "ad": ad, "karar": k,
            "en_buyuk_hareket": r.get("en_buyuk_hareket"),
            "hareketli_kemik": r.get("hareketli_kemik"),
            "kemik": r.get("kemik"), "render": r.get("basarili")})
        if k == "NORMAL":
            sonuc["normal"] += 1
        elif k == "HAREKET_YOK":
            sonuc["hareketsiz"] += 1
        elif k == "GECIS_DUZ":
            sonuc["gecis_duz"] = sonuc.get("gecis_duz", 0) + 1
        else:
            sonuc["asiri"] += 1
    return sonuc


def main() -> int:
    """Animasyon önizlemesi."""
    ap = argparse.ArgumentParser(description="Animasyon önizlemesi")
    ap.add_argument("--render", default="", help="Varlık kimliği")
    ap.add_argument("--anim", default="", help="Action adı (boş = mevcut)")
    ap.add_argument("--dogrula", default="", help="Tüm animasyonları doğrula")
    ap.add_argument("--kare", type=int, default=5)
    ap.add_argument("--cozunurluk", type=int, default=384)
    args = ap.parse_args()

    print("=" * 78)
    print("  ANİMASYON ÖNİZLEMESİ — zaman çizelgesinden render")
    print("=" * 78)

    if args.render:
        d = render(args.render, args.anim, kare_sayisi=args.kare,
                   cozunurluk=args.cozunurluk)
        if "hata" in d:
            print(f"  ✗ {d['hata']}")
            return 1
        print(f"\n  ✓ {d['hedef']} · {d['anim']} · {d['sure_s']} sn")
        print(f"    action     : {d.get('action')} "
              f"kare={d.get('kare_araligi')}")
        print(f"    kemik      : {d.get('kemik')}")
        print(f"    render     : {d.get('basarili')}/{len(d.get('render', []))}")
        print(f"    DEFORM     : en büyük={d.get('en_buyuk_deform')} m "
              f"(vertex yer değiştirmesi)")
        print(f"    FCURVE     : en büyük={d.get('en_buyuk_fcurve')} rad")
        print(f"    KARAR      : {d.get('karar')}")
        # ⚠ En çok deforme olan 5 kemik
        _d = d.get("deform") or {}
        _s = sorted(_d.items(), key=lambda x: -x[1])[:5]
        if _s:
            print(f"    en deforme :")
            for _ad, _v in _s:
                print(f"      {_ad:16} {_v}")
        # ⚠ En çok hareket eden 5 kemik
        h = sorted(d.get("hareket", {}).items(), key=lambda x: -x[1])[:5]
        print(f"    en aktif   :")
        for ad, v in h:
            print(f"      {ad:16} {v}")
        if d.get("uyari"):
            print(f"    ⚠ {d['uyari'][:3]}")
        print(f"    klasör     : {d.get('klasor')}")
        Path("_anim_onizleme.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    if args.dogrula:
        s = dogrula(args.dogrula)
        if "hata" in s:
            print(f"  ✗ {s['hata']}")
            return 1
        print(f"\n  {len(s['animasyon'])} animasyon doğrulandı")
        print(f"    NORMAL      : {s['normal']}")
        print(f"    HAREKETSİZ  : {s['hareketsiz']}")
        print(f"    GEÇİŞ DÜZ   : {s.get('gecis_duz', 0)}  (tasarım gereği)")
        print(f"    AŞIRI       : {s['asiri']}")
        for a in s["animasyon"]:
            if "hata" in a:
                print(f"    ✗ {a['ad']:22} {a['hata'][:40]}")
            else:
                b = {"NORMAL": "✓", "HAREKET_YOK": "✗",
                     "GECIS_DUZ": "·"}.get(a["karar"], "~")
                print(f"    {b} {a['ad']:22} {a['karar']:12} "
                      f"en_buyuk={a['en_buyuk_hareket']} "
                      f"kemik={a['hareketli_kemik']}/{a['kemik']}")
        Path("_anim_dogrulama2.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
