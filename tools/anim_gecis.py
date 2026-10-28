"""
Animasyon Geçişleri — NLA + Blend (Hedef D)
=============================================
⚠ Kullanıcı hedefi
------------------
*"her armature'lü varlık için kemik yapısı farklı olacağından ona uygun
anim yapmak gerekecek"*

Tur 9'da 5 animasyon üretildi ama aralarında **geçiş yoktu**:
```
bekleme → yurume  : ani sıçrama (kemik değerleri farklı)
yurume  → kosma   : ani sıçrama
```

⚠ Sorun
-------
```
Ayrı action'lar **bağımsız** çalışır:
    bekleme  kare 1:  omurga[1] = 0.000
    yurume   kare 1:  omurga[1] = 0.000   ← uyumlu (ikisi de sıfır)
    kosma    kare 1:  kok[0]    = 0.120   ← SIÇRAMA! (bekleme 0.000)
```

**Çözüm: geçiş (blend) action'ı** — iki animasyonun arasını yumuşatır.

Bu betik
--------
```
1. Her animasyon çifti için **ilk/son kare** değerlerini oku
2. Fark varsa **geçiş action'ı** üret (enterpolasyon)
3. NLA'ya yerleştir (sıralı, üst üste binmeyen)
```

⚠ Blender 5.x notu: NLA + fake user (tur 9 dersi)

Kullanım
--------
    python tools/anim_gecis.py --analiz insan_orta_orta
    python tools/anim_gecis.py --uret insan_orta_orta
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
from typing import Any, Dict, List, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

BLENDER = Path(r"E:\AppsTools\Blender 5.2\blender.exe")


#: ⚠ Animasyonların ilk/son kare değerlerini okur
ANALIZ = r'''
import bpy, json

bpy.ops.wm.open_mainfile(filepath=BLEND_YOLU)

o = {"actions": {}, "uyari": []}

for a in bpy.data.actions:
    kayit = {"kare": [round(float(a.frame_range[0]), 1),
                      round(float(a.frame_range[1]), 1)],
             "ilk": {}, "son": {}, "fcurve": 0}
    for kat in a.layers:
        for st in kat.strips:
            for cb in st.channelbags:
                for fc in cb.fcurves:
                    kayit["fcurve"] += 1
                    yol = fc.data_path
                    if 'pose.bones["' not in yol:
                        continue
                    kemik = yol.split('pose.bones["')[1].split('"')[0]
                    kanal = f"{yol.split('.')[-1]}[{fc.array_index}]"
                    anahtar = f"{kemik}.{kanal}"
                    nk = list(fc.keyframe_points)
                    if not nk:
                        continue
                    nk.sort(key=lambda x: x.co[0])
                    kayit["ilk"][anahtar] = round(float(nk[0].co[1]), 5)
                    kayit["son"][anahtar] = round(float(nk[-1].co[1]), 5)
    o["actions"][a.name] = kayit

print("A" + json.dumps(o, ensure_ascii=False))
'''


#: ⚠ Geçiş action'ı üretir (iki animasyon arasında enterpolasyon)
GECIS_URET = r'''
import bpy, json, math

bpy.ops.wm.open_mainfile(filepath=BLEND_YOLU)

CIFTLER = CIFTLER_V            # [[ad1, ad2, gecis_kare], ...]
olcum = {"uretilen": [], "atlanan": [], "hata": []}

arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if arm is None:
    print("G" + json.dumps({"hata": "armature yok"}))
    raise SystemExit(0)

def oku(act):
    """Action'ın kemik → kanal → [(kare, deger)] haritasını döner."""
    m = {}
    for kat in act.layers:
        for st in kat.strips:
            for cb in st.channelbags:
                for fc in cb.fcurves:
                    yol = fc.data_path
                    if 'pose.bones["' not in yol:
                        continue
                    kemik = yol.split('pose.bones["')[1].split('"')[0]
                    kanal = f"{yol.split('.')[-1]}[{fc.array_index}]"
                    nk = sorted(((float(x.co[0]), float(x.co[1]))
                                 for x in fc.keyframe_points),
                                key=lambda z: z[0])
                    if nk:
                        m.setdefault(kemik, {})[kanal] = nk
    return m

for cift in CIFTLER_V:
    a1, a2, gkare = cift[0], cift[1], int(cift[2])
    act1 = bpy.data.actions.get(a1)
    act2 = bpy.data.actions.get(a2)
    if act1 is None or act2 is None:
        olcum["atlanan"].append(f"{a1}->{a2}: action yok")
        continue

    m1 = oku(act1)
    m2 = oku(act2)
    # ⚠ Ortak kemik/kanallar
    ortak = []
    for kemik, kanallar in m1.items():
        if kemik not in m2:
            continue
        for kanal in kanallar:
            if kanal in m2[kemik]:
                ortak.append((kemik, kanal,
                              kanallar[kanal][-1][1],     # a1 SON
                              m2[kemik][kanal][0][1]))    # a2 İLK
    if not ortak:
        olcum["atlanan"].append(f"{a1}->{a2}: ortak kanal yok")
        continue

    # ⚠ Geçiş action'ı: farkı yumuşak kapat
    ad = f"gecis_{a1}_{a2}"
    # ⚠ Önce aynı adlı action varsa sil
    for _a in list(bpy.data.actions):
        if _a.name == ad or _a.name.startswith(ad + "."):
            _a.use_fake_user = False
            try:
                bpy.data.actions.remove(_a, do_unlink=True)
            except Exception:
                pass

    if not arm.animation_data:
        arm.animation_data_create()
    act = bpy.data.actions.new(ad)
    act.use_fake_user = True
    kat = act.layers.new("Katman")
    st = kat.strips.new(type="KEYFRAME")
    slot = act.slots.new(id_type="OBJECT", name=arm.name)
    cb = st.channelbag(slot, ensure=True)

    yazilan = 0
    azami_fark = 0.0
    for kemik, kanal, v1, v2 in ortak:
        idx = int(kanal.split("[")[1].rstrip("]"))
        taban = kanal.split("[")[0]
        yol = {"rotation_euler": "rotation_euler",
               "rotation_quaternion": "rotation_quaternion",
               "location": "location", "scale": "scale"}.get(taban)
        if yol is None:
            continue
        azami_fark = max(azami_fark, abs(v2 - v1))
        tam = f'pose.bones["{kemik}"].{yol}'
        try:
            fc = cb.fcurves.new(data_path=tam, index=idx)
        except Exception as exc:
            olcum["hata"].append(f"{ad}/{kemik}[{idx}]: {type(exc).__name__}")
            continue
        # ⚠ S yumuşatma (smoothstep): 3t² - 2t³
        noktalar = []
        for i in range(gkare + 1):
            t = i / gkare
            s = t * t * (3.0 - 2.0 * t)
            noktalar.append((i + 1, v1 + (v2 - v1) * s))
        fc.keyframe_points.add(len(noktalar))
        for j, (kare, deger) in enumerate(noktalar):
            kp = fc.keyframe_points[j]
            kp.co = (float(kare), float(deger))
            kp.interpolation = "BEZIER"
        fc.update()
        yazilan += len(noktalar)

    # ⚠ tur 12 ÖLÇÜMÜ: fark 0 ise iki animasyon **zaten sürekli**
    # (ör. bekleme ve yurume ikisi de `_sin(0) = 0` ile başlar).
    # Bu durumda geçiş action'ı **düz çizgi** olur — zararsız ama
    # gereksiz. Raporlamada ayrılır.
    olcum["uretilen"].append({
        "ad": ad, "kaynak": [a1, a2], "kare": gkare,
        "kanal": len(ortak), "keyframe": yazilan,
        "azami_fark": round(azami_fark, 5),
        "gercek_gecis": azami_fark > 0.001})

# ⚠ NLA'ya sıralı yerleştir (üst üste binmeyen)
try:
    if arm.animation_data:
        # Mevcut NLA izlerini temizle
        for iz in list(arm.animation_data.nla_tracks):
            arm.animation_data.nla_tracks.remove(iz)
        sira = 1
        for a in sorted(bpy.data.actions, key=lambda x: x.name):
            iz = arm.animation_data.nla_tracks.new()
            iz.name = a.name
            ks, ksn = a.frame_range
            serit = iz.strips.new(a.name, int(sira), a)
            serit.frame_start = float(sira)
            serit.frame_end = float(sira) + (float(ksn) - float(ks))
            sira = int(serit.frame_end) + 1
        olcum["nla"] = len(arm.animation_data.nla_tracks)
except Exception as exc:
    olcum["nla_hata"] = f"{type(exc).__name__}: {str(exc)[:70]}"

bpy.ops.wm.save_as_mainfile(filepath=BLEND_YOLU)
olcum["tum_action"] = sorted(a.name for a in bpy.data.actions)
print("G" + json.dumps(olcum, ensure_ascii=False))
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


def _kos(kod: str, *, zaman: int = 300) -> str:
    """Blender betiğini çalıştırır."""
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
            return ""
    return (p.stdout or "") + (p.stderr or "")


def analiz(hedef: str) -> Dict[str, Any]:
    """Animasyonların ilk/son kare değerlerini okur."""
    blend = _blend_bul(hedef)
    if blend is None:
        return {"hata": f"varlık bulunamadı: {hedef}"}
    out = _kos(f'BLEND_YOLU = r"{blend.resolve()}"\n' + ANALIZ)
    for s in out.splitlines():
        if "A{" in s:
            try:
                return json.loads(s[s.index("A{") + 1:])
            except Exception:
                pass
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:160] or "çıktı yok"}


def uret(hedef: str, *, gecis_kare: int = 8,
         zaman: int = 420) -> Dict[str, Any]:
    """
    Mantıklı animasyon çiftleri için geçiş action'ı üretir.

    ⚠ Çiftler: bekleme→yurume→kosma (döngü zinciri) +
    saldiri→bekleme, olum (tek yönlü)
    """
    blend = _blend_bul(hedef)
    if blend is None:
        return {"hata": f"varlık bulunamadı: {hedef}"}
    # ⚠ Anlamlı geçiş zinciri
    #: ⚠ tur 23: 13 tarif için anlamlı geçiş zinciri (24 çift)
    ciftler = [
        # ── Döngü zinciri ──
        ["bekleme", "yurume", gecis_kare],
        ["yurume", "kosma", gecis_kare],
        ["kosma", "yurume", gecis_kare],
        ["yurume", "bekleme", gecis_kare],
        # ── Eylemler ──
        ["bekleme", "saldiri", gecis_kare],
        ["saldiri", "bekleme", gecis_kare],
        ["bekleme", "blok", gecis_kare],
        ["blok", "bekleme", gecis_kare],
        # ── Dikey hareket ──
        ["bekleme", "ziplama", gecis_kare],
        ["ziplama", "dusme", gecis_kare],
        ["dusme", "bekleme", gecis_kare],
        # ── Durum değişimleri ──
        ["bekleme", "oturma", gecis_kare],
        ["oturma", "bekleme", gecis_kare],
        # ── Son ──
        ["bekleme", "olum", gecis_kare],
        # ══ ⚠ tur 23: YENİ 4 TARİF İÇİN GEÇİŞLER ══════════════════════
        # ── YÜZME: su kenarı mantığı ──
        #    bekleme ↔ yüzme (suya gir/çık)
        ["bekleme", "yuzme", gecis_kare],
        ["yuzme", "bekleme", gecis_kare],
        #    yüzme ↔ yurume (sudan çıkıp yürüme)
        ["yuzme", "yurume", gecis_kare],
        ["yurume", "yuzme", gecis_kare],
        # ── TIRMANMA: dikey geçiş mantığı ──
        #    bekleme → tirmanma (duvara tutun)
        ["bekleme", "tirmanma", gecis_kare],
        ["tirmanma", "bekleme", gecis_kare],
        #    tirmanma → dusme (bırakıp düşme)
        ["tirmanma", "dusme", gecis_kare],
        # ── HASAR: savaş döngüsü ──
        #    bekleme → hasar → bekleme (sarsılıp toparlan)
        ["bekleme", "hasar", gecis_kare],
        ["hasar", "bekleme", gecis_kare],
        #    hasar → olum (öldürücü darbe)
        ["hasar", "olum", gecis_kare],
        #    hasar → blok (savunmaya geç)
        ["hasar", "blok", gecis_kare],
        # ── DİRİLME: ölüm-diriliş döngüsü ──
        #    ⚠ olum → dirilme → bekleme (tam döngü)
        ["olum", "dirilme", gecis_kare],
        ["dirilme", "bekleme", gecis_kare],
    ]
    kod = (f'BLEND_YOLU = r"{blend.resolve()}"\n'
           f'CIFTLER_V = {json.dumps(ciftler)}\n') + GECIS_URET
    t0 = time.perf_counter()
    out = _kos(kod, zaman=zaman)
    sure = time.perf_counter() - t0
    for s in out.splitlines():
        if "G{" in s:
            try:
                d = json.loads(s[s.index("G{") + 1:])
                d["sure_s"] = round(sure, 2)
                d["hedef"] = hedef
                return d
            except Exception:
                pass
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:180] or "çıktı yok"}


def main() -> int:
    """Geçiş üretimi/analizi."""
    ap = argparse.ArgumentParser(description="Animasyon geçişleri")
    ap.add_argument("--analiz", default="")
    ap.add_argument("--uret", default="")
    ap.add_argument("--kare", type=int, default=8)
    args = ap.parse_args()

    print("=" * 78)
    print("  ANİMASYON GEÇİŞLERİ — NLA + blend")
    print("=" * 78)

    if args.analiz:
        d = analiz(args.analiz)
        # ⚠ tur 12: `if "hata" in d` YANLIŞ — ölçüm sözlüğünde
        # `"hata": []` her zaman var. Dolu mu diye bakılır.
        _h = d.get("hata")
        if isinstance(_h, str) and _h:
            print(f"  ✗ {_h}")
            return 1
        if isinstance(_h, list) and _h:
            print(f"  ✗ {len(_h)} hata: {_h[:2]}")
            return 1
        print(f"\n  {len(d['actions'])} action:")
        for ad, a in sorted(d["actions"].items()):
            print(f"\n    {ad:14} kare={a['kare']} fcurve={a['fcurve']}")
            # ⚠ Fark büyük olan kanallar
            farklar = []
            for k in a["ilk"]:
                if k in a["son"]:
                    f = abs(a["son"][k] - a["ilk"][k])
                    if f > 0.01:
                        farklar.append((f, k, a["ilk"][k], a["son"][k]))
            farklar.sort(reverse=True)
            for f, k, v1, v2 in farklar[:3]:
                print(f"      {k:24} {v1:>8} → {v2:>8}  (fark {f:.4f})")
        return 0

    if args.uret:
        d = uret(args.uret, gecis_kare=args.kare)
        # ⚠ Aynı tuzak: boş liste "hata" DEĞİLDİR
        _h = d.get("hata")
        if isinstance(_h, str) and _h:
            print(f"  ✗ {_h}")
            return 1
        if isinstance(_h, list) and _h:
            print(f"  ⚠ {len(_h)} hata: {_h[:2]}")
        print(f"\n  {d['sure_s']} sn · "
              f"{len(d.get('uretilen', []))} geçiş üretildi")
        for g in d.get("uretilen", []):
            tur = "GERÇEK" if g.get("gercek_gecis") else "sürekli"
            print(f"    {g['ad']:28} kare={g['kare']:>2} "
                  f"kanal={g['kanal']:>3} kf={g['keyframe']:>4} "
                  f"fark={g['azami_fark']:<8} [{tur}]")
        if d.get("atlanan"):
            print(f"    atlanan: {d['atlanan'][:3]}")
        if d.get("hata") and isinstance(d["hata"], list) and d["hata"]:
            print(f"    hata: {d['hata'][:2]}")
        print(f"\n  NLA iz: {d.get('nla', '?')}")
        if d.get("nla_hata"):
            print(f"    ⚠ {d['nla_hata']}")
        print(f"  toplam action: {len(d.get('tum_action', []))}")
        Path("_anim_gecis.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
