"""
Animasyon Yeniden Uygulama (Retarget)
=======================================
⚠ Kullanıcı hedefi (D)
----------------------
*"her armature'lü varlık için kemik yapısı farklı olacağından ona uygun
anim yapmak gerekecek"*

⚠ Neden gerekli?
----------------
`anim_kutuphanesi.py` **eşleştirme kararı** verir (hangi animasyon
uygulanabilir). Ama **uygulamak** ayrı bir iştir:

```
Kaynak animasyon : kemik adı `on_ayak_l`, dönme kanalı `rot_euler`
Hedef armature   : kemik adı `front_leg_l`
                   → kanalı `front_leg_l`'ye YAZ, ama:
                     · rotation_mode farklı olabilir (euler/quaternion)
                     · eksen yönü farklı olabilir
                     · kemik uzunluğu farklı → aynı açı farklı sonuç
```

⚠ Ölçülen tuzaklar
------------------
```
1. `rotation_mode`: kaynak `XYZ`, hedef `QUATERNION` olabilir
2. `data_path` kemik ADINI içerir → ad eşlemesi şart
3. Blender 5.x katmanlı Action → yeni action oluşturma yolu değişti
4. Farklı kemik uzunluğu → konum kanalı (`location`) ölçeklenmeli
```

Kullanım
--------
    python tools/anim_uygula.py --hedef yaratik/timsah_0001 --anim Yuruyus
    python tools/anim_uygula.py --test
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

BLENDER = Path(r"E:\AppsTools\Blender 5.2\blender.exe")


#: ⚠ Retarget betiği — kaynak animasyonu hedef armature'e yazar
#:
#: `ESLEME` : {kaynak_kemik: hedef_kemik} — rol eşleştirmesinden gelir
#: `VERI`   : {kaynak_kemik: {kanal: [(kare, deger), ...]}}
RETARGET = r'''
import bpy, json, math
from mathutils import Euler, Quaternion

bpy.ops.wm.open_mainfile(filepath=HEDEF_BLEND)

ESLEME = ESLEME_VERISI
VERI = ANIM_VERISI
ANIM_AD = ANIM_ADI
KARE_BAS = KARE_BASI

olcum = {"yazilan_kemik": 0, "yazilan_keyframe": 0,
         "atlanan": [], "mod_degisen": [], "action": "",
         "kanal_dagilim": {}}

# ⚠ Hedef armature'ü bul
arm_ob = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if arm_ob is None:
    print("OLCUM:" + json.dumps({"hata": "armature yok"}))
    raise SystemExit(0)

pb = arm_ob.pose.bones

# ⚠ Yeni action oluştur (Blender 5.x: animation_data + action)
if not arm_ob.animation_data:
    arm_ob.animation_data_create()
act = bpy.data.actions.new(ANIM_AD)
arm_ob.animation_data.action = act

bpy.context.view_layer.objects.active = arm_ob
bpy.ops.object.mode_set(mode="POSE")

# ⚠ Önce tüm kemiklerin rotation_mode'unu ayarla (kanal tutarlılığı)
for kaynak_kemik, kanallar in VERI.items():
    hedef_kemik = ESLEME.get(kaynak_kemik)
    if not hedef_kemik:
        olcum["atlanan"].append(kaynak_kemik)
        continue
    k = pb.get(hedef_kemik)
    if k is None:
        olcum["atlanan"].append(kaynak_kemik)
        continue

    for kanal, noktalar in kanallar.items():
        if not noktalar:
            continue
        # ⚠ Kanal → data_path
        if kanal == "rot_euler":
            # Hedef quaternion ise euler'e zorla (basit ve tutarlı)
            if k.rotation_mode == "QUATERNION":
                k.rotation_mode = "XYZ"
                olcum["mod_degisen"].append(hedef_kemik)
            yol = "rotation_euler"
            bilesenler = 3
        elif kanal == "rot_quaternion":
            if k.rotation_mode != "QUATERNION":
                k.rotation_mode = "QUATERNION"
                olcum["mod_degisen"].append(hedef_kemik)
            yol = "rotation_quaternion"
            bilesenler = 4
        elif kanal == "location":
            yol = "location"
            bilesenler = 3
        elif kanal == "scale":
            yol = "scale"
            bilesenler = 3
        else:
            continue

        # ⚠ Kaynak veri: [(kare, deger), ...] — tek eksen (fcurve başına)
        # Her fcurve ayrı geldiği için **indeks** bilgisi yok.
        # Bu yüzden veriyi dizi olarak toplayıp yeniden grupla.
        olcum["kanal_dagilim"][kanal] = olcum["kanal_dagilim"].get(
            kanal, 0) + len(noktalar)
        olcum["yazilan_kemik"] += 1
        olcum["yazilan_keyframe"] += len(noktalar)

olcum["action"] = act.name
olcum["hedef_kemik"] = [b.name for b in arm_ob.data.bones]
olcum["esleme"] = ESLEME
print("OLCUM:" + json.dumps(olcum, ensure_ascii=False))
'''


def rol_bul(kemik: str) -> str:
    """`anim_kutuphanesi` ile aynı rol mantığı."""
    from tools.anim_kutuphanesi import rol_bul as rb

    return rb(kemik)


def esleme_uret(kaynak_kemikler: Sequence[str],
                hedef_kemikler: Sequence[str]) -> Dict[str, str]:
    """
    Rol tabanlı kemik eşlemesi üretir.

    ⚠ İki aşama:
        1. **Tam ad** eşleşmesi (varsa en güvenli)
        2. **Rol** eşleşmesi (isim farklı olsa da)
    """
    e: Dict[str, str] = {}
    hedef_kume = set(hedef_kemikler)
    # 1) Tam ad
    for kk in kaynak_kemikler:
        if kk in hedef_kume:
            e[kk] = kk
    # 2) Rol
    hedef_rol: Dict[str, str] = {}
    for hk in hedef_kemikler:
        r = rol_bul(hk)
        hedef_rol.setdefault(r, hk)
    for kk in kaynak_kemikler:
        if kk in e:
            continue
        r = rol_bul(kk)
        if r in hedef_rol:
            e[kk] = hedef_rol[r]
    return e


def test(*, zaman: int = 240) -> Dict[str, Any]:
    """
    **Uçtan uca retarget testi**:
        1. Farklı kemik adlı bir yaratık üret (`timsah_esanlamli`)
        2. Kütüphanedeki `Yuruyus`'u **ona uygula**
        3. Kaydet → **diskten yeniden aç** → keyframe doğrula
    """
    if not BLENDER.exists():
        return {"hata": "blender yok"}

    # ⚠ Kaynak animasyon verisi
    from tools.anim_kutuphanesi import _kutuphane_oku

    k = _kutuphane_oku()
    anim = k.get("animasyonlar", {})
    if not anim:
        return {"hata": "kütüphane boş"}
    anahtar = next(iter(anim))
    a = anim[anahtar]
    kaynak_kemikler = a.get("kemikler", [])

    # ⚠ Hedef: eş anlamlı adlarla YENİ yaratık üret
    ES_ANLAMLI = {
        "kok": "root", "omurga": "spine", "boyun": "neck",
        "kafa": "head", "cene": "jaw",
        "on_ayak_l": "front_leg_l", "on_ayak_r": "front_leg_r",
        "arka_ayak_l": "back_leg_l", "arka_ayak_r": "back_leg_r",
        "kuyruk_1": "tail_1", "kuyruk_2": "tail_2", "kuyruk_3": "tail_3",
    }
    hedef_kemikler = [ES_ANLAMLI.get(x, x) for x in kaynak_kemikler]
    esleme = esleme_uret(kaynak_kemikler, hedef_kemikler)

    # ⚠ Hedef varlığı üret (aynı mesh + eş anlamlı kemikler)
    from tools.yaratik_uret import TİMSAH

    hedef_klasor = Path("Varliklar/yaratik/timsah_esanlamli")
    hedef_klasor.mkdir(parents=True, exist_ok=True)
    blend = hedef_klasor / "varlik.blend"
    glb = hedef_klasor / "varlik.glb"

    # Kemik adlarını değiştirerek betiği üret
    kod = TİMSAH
    for eski, yeni in ES_ANLAMLI.items():
        kod = kod.replace(f'("{eski}"', f'("{yeni}"')
        kod = kod.replace(f'"{eski}")', f'"{yeni}")')
        kod = kod.replace(f'("{eski}",', f'("{yeni}",')
        kod = kod.replace(f'"{eski}"', f'"{yeni}"')
    # ⚠ Animasyonu ÇIKAR (retarget ile gelecek)
    kod = kod.split("# ══ 8. ANİMASYON")[0]
    kod += '''
me2 = govde.data
olcum2 = {"vertex": len(me2.vertices), "kemik": [b.name for b in arm.bones],
          "kemik_sayisi": len(arm.bones), "vertex_group": len(govde.vertex_groups),
          "uv": len(me2.uv_layers)}
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
bpy.ops.export_scene.gltf(filepath=str(OUT_GLB), export_format="GLB")
import json as _j
print("OLCUM:" + _j.dumps(olcum2, ensure_ascii=False))
'''
    baslik = (f'OUT_BLEND = r"{blend.resolve()}"\n'
              f'OUT_GLB = r"{glb.resolve()}"\n')

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "u.py"
        b.write_text(baslik + kod, encoding="utf-8")
        try:
            p = subprocess.run(
                [str(BLENDER), "--background", "--factory-startup",
                 "--python", str(b)],
                capture_output=True, text=True, timeout=zaman,
                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return {"hata": "zaman aşımı (üretim)"}
    uretim_s = time.perf_counter() - t0
    out = (p.stdout or "") + (p.stderr or "")
    uretim: Dict[str, Any] = {}
    for satir in out.splitlines():
        if satir.strip().startswith("OLCUM:"):
            uretim = json.loads(satir[6:])
    if not uretim:
        iz = [x.strip() for x in out.splitlines() if "Error" in x]
        return {"hata": "üretim başarısız: "
                + ("; ".join(iz[:2])[:120] or "OLCUM yok")}

    # ⚠ Retarget: kaynak animasyonu hedefe YAZ + **diskten doğrula**
    yazim = _retarget_yaz(blend, esleme, a.get("veri", {}), a.get("ad", ""))
    dogr = _dogrula(blend)
    return {
        "uretim": uretim, "uretim_s": round(uretim_s, 2),
        "kaynak_anim": anahtar,
        "kaynak_kemikler": kaynak_kemikler,
        "hedef_kemikler": uretim.get("kemik", []),
        "esleme": esleme,
        "esleme_sayisi": len(esleme),
        "veri_kemik": list(a.get("veri", {}).keys()),
        "veri": a.get("veri", {}),
        "yazim": yazim, "dogrulama": dogr,
        "blend": str(blend),
    }


#: ⚠ Retarget yazımı — **fcurve'leri doğrudan kurar**.
#:
#: ⚠ ÖLÇÜLEN KÖK NEDEN (tur 4):
#:     `kb.keyframe_insert(data_path="rotation_euler", frame=N)` çağrısı
#:     Blender 5.x **katmanlı Action**'ında keyfi `array_index`leri
#:     güvenilir yazmıyor:
#:         idx2 (kuyruk) → 0.22 ✓
#:         idx0 (bacak)  → 0.0  ✗  (yazarken 0.55 OKUNUYOR!)
#:         idx1 (kafa)   → 0.0  ✗
#:     Ölçüm: 24 kanaldan yalnız 3'ü doldu.
#:
#: ✅ ÇÖZÜM — slot + channelbag + `fcurves.new()`:
#:     ```python
#:     act = bpy.data.actions.new(ad)
#:     kat = act.layers.new("K")
#:     st  = kat.strips.new(type="KEYFRAME")
#:     slot = act.slots.new(id_type="OBJECT", name=arm.name)
#:     cb  = st.channelbag(slot, ensure=True)
#:     fc  = cb.fcurves.new(data_path=yol, index=idx)
#:     fc.keyframe_points.add(n); fc.keyframe_points[i].co = (kare, deger)
#:     fc.update()
#:     ```
#:     Doğrulandı: **4/4 eksen** kaydedildi ve yeniden açıldı.
YAZIM = r'''
import bpy, json, re

bpy.ops.wm.open_mainfile(filepath=HEDEF_BLEND)
ESLEME = ESLEME_V
VERI = VERI_V
ANIM_AD = ANIM_V

o = {"yazilan_kemik": 0, "yazilan_nokta": 0, "atlanan": [],
     "kanal": {}, "eksen": {}, "mod_degisen": []}

arm = next((x for x in bpy.data.objects if x.type == "ARMATURE"), None)
if arm is None:
    print("Y" + json.dumps({"hata": "armature yok"}))
    raise SystemExit(0)

pb = arm.pose.bones
if not arm.animation_data:
    arm.animation_data_create()

# ⚠ Blender 5.x: katman + strip + slot + channelbag ZORUNLU
act = bpy.data.actions.new(ANIM_AD)
arm.animation_data.action = act
kat = act.layers.new("Katman")
st = kat.strips.new(type="KEYFRAME")
slot = act.slots.new(id_type="OBJECT", name=arm.name)
cb = st.channelbag(slot, ensure=True)
o["slot"] = getattr(slot, "identifier", "")

YOL = {"rot_euler": "rotation_euler",
       "rot_quaternion": "rotation_quaternion",
       "location": "location", "scale": "scale"}

for src in sorted(VERI.keys()):
    kanallar = VERI[src]
    hedef = ESLEME.get(src)
    kb = pb.get(hedef) if hedef else None
    if kb is None:
        o["atlanan"].append(src)
        continue
    # ⚠ rotation_mode kanaldan ÖNCE ayarlanır
    for kanal in kanallar:
        m0 = re.match(r"^(\w+)\[(\d+)\]$", kanal)
        tb0 = m0.group(1) if m0 else kanal
        if tb0 == "rot_euler" and kb.rotation_mode == "QUATERNION":
            kb.rotation_mode = "XYZ"
            o["mod_degisen"].append(hedef)
        if tb0 == "rot_quaternion" and kb.rotation_mode != "QUATERNION":
            kb.rotation_mode = "QUATERNION"
            o["mod_degisen"].append(hedef)

    for kanal in sorted(kanallar.keys()):
        noktalar = kanallar[kanal]
        if not noktalar:
            continue
        m = re.match(r"^(\w+)\[(\d+)\]$", kanal)
        if m:
            taban, idx = m.group(1), int(m.group(2))
        else:
            taban, idx = kanal, 0
        yol = YOL.get(taban)
        if yol is None:
            continue
        # ⚠ fcurve'u DOĞRUDAN kur (keyframe_insert GÜVENİLMEZ — ölçüldü)
        tam = f'pose.bones["{hedef}"].{yol}'
        try:
            fc = cb.fcurves.new(data_path=tam, index=idx)
        except Exception as exc:
            o.setdefault("fcurve_hata", []).append(
                f"{tam}[{idx}]: {type(exc).__name__}")
            continue
        fc.keyframe_points.add(len(noktalar))
        for i, (kare, deger) in enumerate(noktalar):
            kp = fc.keyframe_points[i]
            kp.co = (float(kare), float(deger))
            kp.interpolation = "BEZIER"
        fc.update()
        o["yazilan_nokta"] += len(noktalar)
        o["kanal"][kanal] = o["kanal"].get(kanal, 0) + len(noktalar)
        o["eksen"][f"idx{idx}"] = o["eksen"].get(f"idx{idx}", 0) + 1
    o["yazilan_kemik"] += 1

bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.wm.save_as_mainfile(filepath=HEDEF_BLEND)
o["action"] = act.name
print("Y" + json.dumps(o, ensure_ascii=False))
'''

#: ⚠ Diskten doğrulama — kanal doluluk ölçümü
DOGRULA = r'''
import bpy, json

bpy.ops.wm.open_mainfile(filepath=HEDEF_BLEND)
o = {"action": "", "kare": [], "fcurve": 0, "kf": 0,
     "kanal": {}, "dolu": 0, "sifir": 0}

for ob in bpy.data.objects:
    if ob.type != "ARMATURE":
        continue
    ad = ob.animation_data
    if not ad or not ad.action:
        continue
    a = ad.action
    o["action"] = a.name
    o["kare"] = [round(float(a.frame_range[0]), 1),
                 round(float(a.frame_range[1]), 1)]
    for kat in a.layers:
        for st in kat.strips:
            for cb in st.channelbags:
                for fc in cb.fcurves:
                    o["fcurve"] += 1
                    y = fc.data_path
                    if 'pose.bones["' not in y:
                        continue
                    kk = y.split('pose.bones["')[1].split('"')[0]
                    n = len(fc.keyframe_points)
                    o["kf"] += n
                    enb = max(abs(float(x.co[1])) for x in fc.keyframe_points)
                    anahtar = f"{kk}.{y.split('.')[-1]}[{fc.array_index}]"
                    o["kanal"][anahtar] = round(enb, 5)
                    if enb > 1e-6:
                        o["dolu"] += 1
                    else:
                        o["sifir"] += 1
o["animasyonlu_kemik"] = len({k.split(".")[0] for k in o["kanal"]})
print("G" + json.dumps(o, ensure_ascii=False))
'''


def _blender_kos(kod: str, *, zaman: int = 300) -> str:
    """Blender betiğini çalıştırır, çıktıyı döner."""
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "b.py"
        b.write_text(kod, encoding="utf-8")
        try:
            p = subprocess.run(
                [str(BLENDER), "--background", "--factory-startup",
                 "--python", str(b)],
                capture_output=True, text=True, timeout=zaman,
                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return ""
    return (p.stdout or "") + (p.stderr or "")


def _retarget_yaz(blend: Path, esleme: Dict[str, str],
                  veri: Dict[str, Any], anim_ad: str) -> Dict[str, Any]:
    """Animasyonu hedef armature'e yazar."""
    kod = (f'HEDEF_BLEND = r"{blend.resolve()}"\n'
           f'ESLEME_V = {json.dumps(esleme)}\n'
           f'VERI_V = {json.dumps(veri)}\n'
           f'ANIM_V = {json.dumps(anim_ad)}\n') + YAZIM
    out = _blender_kos(kod)
    for s in out.splitlines():
        if s.startswith("Y{"):
            try:
                return json.loads(s[1:])
            except Exception:
                pass
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:140] or "yazım çıktısı yok"}


def _dogrula(blend: Path) -> Dict[str, Any]:
    """Kaydedilmiş dosyayı yeniden açıp kanal doluluğunu ölçer."""
    kod = f'HEDEF_BLEND = r"{blend.resolve()}"\n' + DOGRULA
    out = _blender_kos(kod)
    for s in out.splitlines():
        if s.startswith("G{"):
            try:
                return json.loads(s[1:])
            except Exception:
                pass
    return {"hata": "doğrulama çıktısı yok"}


def main() -> int:
    """Retarget testi."""
    ap = argparse.ArgumentParser(description="Animasyon retarget")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--hedef", default="")
    ap.add_argument("--anim", default="")
    args = ap.parse_args()

    print("=" * 78)
    print("  ANİMASYON YENİDEN UYGULAMA (RETARGET)")
    print("=" * 78)

    if not args.test:
        ap.print_help()
        return 0

    print("\n  test ediliyor…", flush=True)
    r = test()
    if "hata" in r:
        print(f"    ✗ {r['hata']}")
        return 1

    print(f"\n  kaynak animasyon : {r['kaynak_anim']}")
    print(f"  üretim           : {r['uretim_s']} sn · "
          f"vertex={r['uretim'].get('vertex')} · "
          f"kemik={r['uretim'].get('kemik_sayisi')}")
    print(f"\n  KAYNAK kemikler  : {r['kaynak_kemikler'][:6]}…")
    print(f"  HEDEF kemikler   : {r['hedef_kemikler'][:6]}…")
    print(f"\n  EŞLEME ({r['esleme_sayisi']} kemik):")
    for kk, hk in list(r["esleme"].items())[:12]:
        im = "=" if kk == hk else "→"
        print(f"    {kk:14} {im} {hk}")

    # ⚠ Animasyonlu kemikler eşlendi mi?
    anim_kemik = set(r["veri_kemik"])
    eslenen = anim_kemik & set(r["esleme"].keys())
    y = r.get("yazim", {})
    d = r.get("dogrulama", {})
    print(f"\n  RETARGET YAZIMI:")
    print(f"    yazılan kemik : {y.get('yazilan_kemik', '?')}")
    print(f"    yazılan nokta : {y.get('yazilan_nokta', '?')}")
    print(f"    eksen dağılımı: {y.get('eksen', {})}")
    print(f"    atlanan       : {y.get('atlanan', [])}")
    print(f"\n  DISKTEN DOĞRULAMA:")
    print(f"    action        : {d.get('action', '?')}")
    print(f"    kare          : {d.get('kare')}")
    print(f"    fcurve        : {d.get('fcurve')} · kf={d.get('kf')}")
    print(f"    DOLU kanal    : {d.get('dolu')} · sıfır: {d.get('sifir')}")
    print(f"    animasyonlu   : {d.get('animasyonlu_kemik')} kemik")

    print(f"\n  ANİMASYONLU KEMİK: {len(anim_kemik)} · "
          f"eşlenen: {len(eslenen)}")
    eksik = anim_kemik - set(r["esleme"].keys())
    if eksik:
        print(f"    ✗ EŞLENMEYEN: {sorted(eksik)}")
    else:
        print(f"    ✓ TÜMÜ EŞLENDİ")

    Path("_retarget.json").write_text(
        json.dumps({k: v for k, v in r.items() if k != "veri"},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
