"""
Varlık Kaydedici — Blender Doğrulaması + Katalog
==================================================
⚠ Kullanıcı hedefleri (D, E, G)
-------------------------------
*"başarılı animler olursa onları o modele göre kayıt da tutma"*
*"her armature'lü varlık için kemik yapısı farklı olacağından ona
uygun anim yapmak gerekecek"*
*"oluşturulan varlıklar için kategorilere ayırıp bilgilerini de kayıt
altında tutabilir"*

⚠ Neden ayrı doğrulama katmanı?
-------------------------------
Blender betiği **kendi kendini** ölçemez — çünkü:
    * `print` çıktısı kaydedilmemiş olabilir
    * Kaydedilen `.blend` **yeniden açılıp** doğrulanmalı
    * Animasyon gerçekten **keyframe** içeriyor mu?

Bu yüzden: üret → **kaydet** → **yeniden aç** → doğrula → kataloğa yaz

⚠ Blender 5.2 API değişikliği (tur 1'de ölçüldü):
```
`Action.fcurves` KALDIRILDI → AttributeError
Yeni yol: action.layers → strips → channelbags → fcurves
```

Kullanım
--------
    python tools/varlik_kaydet.py --dogrula Varliklar/yaratik/timsah_0001
    python tools/varlik_kaydet.py --katalogla
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

BLENDER = Path(r"E:\AppsTools\Blender 5.2\blender.exe")


#: ⚠ Kaydedilmiş `.blend` dosyasını **yeniden açıp** doğrular.
#:
#: Ölçülenler:
#:     * mesh: vertex/kenar/yüz, UV katmanı, malzeme, gölgeleme
#:     * armature: kemik sayısı/adları, hiyerarşi
#:     * weight: vertex grubu sayısı + **ağırlıklı vertex oranı**
#:     * animasyon: action adı, kare aralığı, **fcurve sayısı**,
#:       kemik başına keyframe (Blender 5.2 katmanlı API)
DOGRULAMA = r'''
import bpy, json

bpy.ops.wm.open_mainfile(filepath=BLEND_YOLU)

o = {
    "mesh": {}, "armature": {}, "weight": {}, "animasyon": {},
    "malzeme": [], "duz_shading": False,
}

for ob in bpy.data.objects:
    # ── MESH ──
    if ob.type == "MESH" and not o["mesh"]:
        me = ob.data
        o["mesh"] = {
            "ad": ob.name,
            "vertex": len(me.vertices),
            "kenar": len(me.edges),
            "yuz": len(me.polygons),
            "uv_katman": len(me.uv_layers),
            "boyut": [round(float(x), 4) for x in ob.dimensions],
        }
        o["duz_shading"] = all(not p.use_smooth for p in me.polygons)
        o["malzeme"] = [m.name for m in me.materials if m]
        # ⚠ Ağırlık doğrulaması: her vertex en az bir gruba bağlı mı?
        bagli = sum(1 for v in me.vertices if len(v.groups) > 0)
        o["weight"] = {
            "vertex_group": len(ob.vertex_groups),
            "grup_adlari": [g.name for g in ob.vertex_groups],
            "agirlikli_vertex": bagli,
            "toplam_vertex": len(me.vertices),
            "oran": round(bagli / max(1, len(me.vertices)), 3),
        }
        # Malzeme düğümleri
        for m in me.materials:
            if m and m.use_nodes:
                o.setdefault("malzeme_dugum", {})[m.name] = len(
                    m.node_tree.nodes)

    # ── ARMATURE ──
    if ob.type == "ARMATURE":
        arm = ob.data
        o["armature"] = {
            "ad": ob.name,
            "kemik_sayisi": len(arm.bones),
            "kemik_adlari": [b.name for b in arm.bones],
            # ⚠ Hiyerarşi: ebeveyn ilişkileri
            "hiyerarsi": {b.name: (b.parent.name if b.parent else "")
                          for b in arm.bones},
            "kok_kemik": [b.name for b in arm.bones if not b.parent],
        }
        # ── ANİMASYON (Blender 5.2 katmanlı API) ──
        ad = ob.animation_data
        if ad and ad.action:
            a = ad.action
            kemik_kf = {}
            fcurve = 0
            # ⚠ `Action.fcurves` YOK → layers → strips → channelbags
            try:
                for kat in a.layers:
                    for strip in kat.strips:
                        for cb in strip.channelbags:
                            for fc in cb.fcurves:
                                fcurve += 1
                                yol = fc.data_path
                                if 'pose.bones["' in yol:
                                    k = yol.split('pose.bones["')[1].split('"')[0]
                                    kemik_kf[k] = kemik_kf.get(k, 0) + len(
                                        fc.keyframe_points)
            except Exception as exc:
                o["animasyon"]["api_hata"] = f"{type(exc).__name__}"
            o["animasyon"] = {
                "action": a.name,
                "kare_bas": round(float(a.frame_range[0]), 1),
                "kare_son": round(float(a.frame_range[1]), 1),
                "kare_sayisi": int(a.frame_range[1] - a.frame_range[0]) + 1,
                "fcurve": fcurve,
                "kemik_keyframe": kemik_kf,
                "toplam_keyframe": sum(kemik_kf.values()),
                "animasyonlu_kemik": len(kemik_kf),
                "dso_katman": len(a.layers),
                "dso_slot": len(a.slots),
            }
        # Tüm action'lar
        o["tum_action"] = [x.name for x in bpy.data.actions]

o["nesne_sayisi"] = len(bpy.data.objects)
print("D" + json.dumps(o, ensure_ascii=False))
'''


def dogrula(klasor: Path, *, zaman: int = 180) -> Dict[str, Any]:
    """
    Kaydedilmiş `.blend` dosyasını **yeniden açıp** doğrular.

    ⚠ Bu, üretim betiğinin `print`'ine güvenmekten **çok** daha güçlü:
    dosya gerçekten diskten okunur.
    """
    blend = klasor / "varlik.blend"
    if not blend.exists():
        return {"hata": "varlik.blend yok"}
    if not BLENDER.exists():
        return {"hata": "blender yok"}
    kod = f'BLEND_YOLU = r"{blend.resolve()}"\n' + DOGRULAMA
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
        except Exception as exc:
            return {"hata": f"{type(exc).__name__}"}
    out = (p.stdout or "") + (p.stderr or "")
    for satir in out.splitlines():
        if "D{" in satir:
            try:
                return json.loads(satir.strip()[1:])
            except Exception:
                pass
    izler = [x.strip() for x in out.splitlines()
             if "Error" in x or "Traceback" in x]
    return {"hata": "; ".join(izler[:2])[:180] or "doğrulama çıktısı yok"}


def kataloga_yaz(klasor: Path, *, kategori: str, ad: str, kalite: str,
                 parametreler: Optional[Dict[str, Any]] = None,
                 kod: str = "") -> str:
    """
    Doğrulanmış varlığı kataloğa yazar.

    ⚠ Meta veri **doğrulama çıktısından** doldurulur — üretim
    betiğinin iddiasından **değil**.
    """
    from tools.varlik_katalogu import VarlikMeta, kaydet

    d = dogrula(klasor)
    m = VarlikMeta(ad=ad, kategori=kategori, kalite=kalite)
    if "hata" in d:
        m.parametreler = {"dogrulama_hatasi": d["hata"]}
    else:
        me = d.get("mesh", {})
        ar = d.get("armature", {})
        w = d.get("weight", {})
        an = d.get("animasyon", {})
        m.vertex = me.get("vertex", 0)
        m.kenar = me.get("kenar", 0)
        m.yuz = me.get("yuz", 0)
        m.uv_katman = me.get("uv_katman", 0)
        m.uv_ok = m.uv_katman > 0
        m.malzeme = d.get("malzeme", [])
        m.duz_golge = d.get("duz_shading", False)
        m.boyut = me.get("boyut", [])
        m.armature_var = bool(ar)
        m.kemik_sayisi = ar.get("kemik_sayisi", 0)
        m.kemik_adlari = ar.get("kemik_adlari", [])
        m.automatic_weight = w.get("oran", 0) >= 0.9
        m.vertex_group = w.get("vertex_group", 0)
        if an:
            m.animasyonlar = [an.get("action", "")]
            m.parametreler = {
                "kare_sayisi": an.get("kare_sayisi"),
                "toplam_keyframe": an.get("toplam_keyframe"),
                "animasyonlu_kemik": an.get("animasyonlu_kemik"),
                "fcurve": an.get("fcurve"),
            }
        # ⚠ Ağırlık oranı ve hiyerarşi de saklanır (D hedefi:
        # her armature'e özgü animasyon)
        m.parametreler.setdefault("agirlik_orani", w.get("oran", 0))
        m.parametreler["hiyerarsi"] = ar.get("hiyerarsi", {})
    if parametreler:
        m.parametreler.update(parametreler)
    if kod:
        from tools.varlik_katalogu import _hash

        m.uret_KOD_hash = _hash(kod)
    m.blend = str(klasor / "varlik.blend")
    m.glb = str(klasor / "varlik.glb")
    m.blender_surum = _blender_surum()

    # ⚠ tur 17 (Hedef G): önizleme PNG'lerini meta veriye bağla
    oniz = klasor / "onizleme"
    if oniz.is_dir():
        m.onizleme_klasor = str(oniz)
        for png in sorted(oniz.glob("*.png")):
            m.onizleme[png.stem] = str(png)
        m.onizleme_aci = len(m.onizleme)

    # ⚠ tur 24 (Hedef D): animasyon önizlemelerini bağla
    #   `anim_onizleme/<animasyon>/kare_NNN.png`
    aniz = klasor / "anim_onizleme"
    if aniz.is_dir():
        m.anim_onizleme_klasor = str(aniz)
        for alt in sorted(x for x in aniz.iterdir() if x.is_dir()):
            pngler = sorted(alt.glob("*.png"))
            if not pngler:
                continue
            m.anim_onizleme[alt.name] = {
                "klasor": str(alt),
                "kare": len(pngler),
                "png": [str(x) for x in pngler],
            }
        m.anim_onizleme_sayisi = len(m.anim_onizleme)

    return kaydet(m, klasor=klasor)


def _blender_surum() -> str:
    """Blender sürümünü okur."""
    if not BLENDER.exists():
        return "?"
    try:
        p = subprocess.run([str(BLENDER), "--version"], capture_output=True,
                           text=True, timeout=40, encoding="utf-8",
                           errors="replace")
        return ((p.stdout or "").splitlines() or ["?"])[0].strip()[:40]
    except Exception:
        return "?"


def main() -> int:
    """Doğrulama/kataloglama."""
    ap = argparse.ArgumentParser(description="Varlık kaydedici")
    ap.add_argument("--dogrula", default="")
    ap.add_argument("--katalogla", action="store_true")
    ap.add_argument("--kategori", default="yaratik")
    ap.add_argument("--ad", default="timsah")
    ap.add_argument("--kalite", default="orta")
    args = ap.parse_args()

    print("=" * 78)
    print("  VARLIK KAYDEDİCİ — Blender doğrulaması + katalog")
    print("=" * 78)

    if args.dogrula:
        k = Path(args.dogrula)
        print(f"  doğrulanıyor: {k}")
        d = dogrula(k)
        print(json.dumps(d, ensure_ascii=False, indent=1)[:3000])
        return 0

    if args.katalogla:
        k = Path("Varliklar") / args.kategori / args.ad
        if not k.exists():
            print(f"  klasör yok: {k}")
            return 1
        vid = kataloga_yaz(k, kategori=args.kategori, ad=args.ad,
                           kalite=args.kalite)
        print(f"  kataloglandı: {vid}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
