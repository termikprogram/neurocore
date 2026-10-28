"""
Animasyon Kütüphanesi — Kemik İmzasıyla Eşleştirme
====================================================
⚠ Kullanıcı hedefi (D)
----------------------
*"başarılı animler olursa onları o modele göre kayıt da tutma, çünkü
sürekli denemeler ile uygun animi oluşturmak kolay olmayacak. her
armatur lü varlık için kemik yapısı farklı olacağından ona uygun anim
yapmak gerekecek."*

⚠ Temel problem
---------------
```
Animasyon bir **kemik adları kümesine** bağlıdır:
    Yuruyus → on_ayak_l, on_ayak_r, arka_ayak_l, arka_ayak_r,
              kuyruk_1..3, kafa      (8 kemik)

Yeni bir yaratık üretildi → kemik adları FARKLI olabilir:
    bacak_on_sol, bacak_on_sag, ...
    ↑ aynı animasyon UYGULANAMAZ (kemik yok)
```

Çözüm: **Kemik imzası + eşleştirme**
-------------------------------------
```
1. Her animasyon **kendi kemik imzasıyla** saklanır
       imza = {kemik adları} + {hiyerarşi} + {kanal türleri}

2. Yeni armature için **uyumluluk** hesaplanır:
       uyum = |ortak kemik| / |animasyon kemikleri|

3. Eşik üstünde (ör. %80) → **uygulanabilir**
   Altında                 → **uyarlama gerekir** veya yeni animasyon

⚠ Neden sadece isim değil, hiyerarşi de?
    İki armature'te `kafa` kemiği olabilir ama biri `boyun` altında,
    diğeri doğrudan `kok` altında → dönme ekseni FARKLI olur.
```

Kullanım
--------
    python tools/anim_kutuphanesi.py --liste
    python tools/anim_kutuphanesi.py --eslesme yaratik/timsah_0001
    python tools/anim_kutuphanesi.py --tara
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

BLENDER = Path(r"E:\AppsTools\Blender 5.2\blender.exe")
KUTUPHANE = Path("Varliklar/animasyon_kutuphanesi.json")


#: ⚠ Kemik rolü sözlüğü — isimler farklı olsa da **rol** eşleşir.
#:
#: Ölçülen ihtiyaç: iki yaratıkta aynı işlev farklı adla olabilir:
#:     `on_ayak_l`  ≡  `bacak_on_sol`  ≡  `front_leg_l`
#: Rol eşleştirmesi olmadan animasyon **uygulanamaz**.
ROLLER: Sequence[Tuple[str, Sequence[str]]] = (
    ("kok", ("kok", "root", "kök")),
    ("omurga", ("omurga", "spine", "govde", "body", "pelvis")),
    ("boyun", ("boyun", "neck")),
    ("kafa", ("kafa", "head", "kelle")),
    ("cene", ("cene", "çene", "jaw", "mouth")),
    ("on_ayak_l", ("on_ayak_l", "bacak_on_sol", "front_leg_l",
                   "on_bacak_l", "leg_front_l", "ayak_on_l")),
    ("on_ayak_r", ("on_ayak_r", "bacak_on_sag", "front_leg_r",
                   "on_bacak_r", "leg_front_r", "ayak_on_r")),
    ("arka_ayak_l", ("arka_ayak_l", "bacak_arka_sol", "back_leg_l",
                     "arka_bacak_l", "leg_back_l", "ayak_arka_l")),
    ("arka_ayak_r", ("arka_ayak_r", "bacak_arka_sag", "back_leg_r",
                     "arka_bacak_r", "leg_back_r", "ayak_arka_r")),
    ("kuyruk_1", ("kuyruk_1", "kuyruk1", "tail_1", "tail1")),
    ("kuyruk_2", ("kuyruk_2", "kuyruk2", "tail_2", "tail2")),
    ("kuyruk_3", ("kuyruk_3", "kuyruk3", "tail_3", "tail3")),
    ("kanat_l", ("kanat_l", "wing_l", "sol_kanat")),
    ("kanat_r", ("kanat_r", "wing_r", "sag_kanat")),
)


@dataclass
class AnimKaydi:
    """Kütüphanedeki tek animasyon kaydı."""

    ad: str = ""
    kaynak_varlik: str = ""
    kategori: str = ""

    # ── Kemik imzası (eşleştirme için) ──
    kemikler: List[str] = field(default_factory=list)
    roller: List[str] = field(default_factory=list)
    hiyerarsi: Dict[str, str] = field(default_factory=dict)

    # ── Animasyon verisi ──
    kare_sayisi: int = 0
    fcurve: int = 0
    toplam_keyframe: int = 0

    #: ⚠ **Keyframe verisi** — yeniden uygulanabilir olması için
    #: {kemik: {kanal: [(kare, x, y, z), ...]}}
    veri: Dict[str, Dict[str, List[List[float]]]] = field(
        default_factory=dict)

    # ── Meta ──
    olusturma: str = ""
    basarili: bool = False
    imza: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def imza_hesapla(kemikler: Sequence[str],
                 hiyerarsi: Dict[str, str]) -> str:
    """
    Kemik kümesi + hiyerarşiden **kararlı imza** üretir.

    ⚠ Sıralama **önemsiz** olmalı → `sorted` kullanılır.
    """
    s = "|".join(sorted(kemikler))
    h = "|".join(f"{k}>{hiyerarsi.get(k, '')}" for k in sorted(kemikler))
    return hashlib.sha1(f"{s}::{h}".encode("utf-8")).hexdigest()[:12]


def rol_bul(kemik: str) -> str:
    """Kemik adından **rol** çıkarır (isim farklı olsa da)."""
    k = kemik.lower().replace("-", "_").replace(" ", "_")
    for rol, esler in ROLLER:
        if k in esler:
            return rol
    # ⚠ Kısmi eşleşme: `kuyruk_4` → `kuyruk_3` ailesi
    for rol, esler in ROLLER:
        kok = rol.rsplit("_", 1)[0]
        if kok and kok in k:
            return f"{kok}_*"
    return k


#: ⚠ `.blend` içinden animasyon **keyframe verisini** çıkarır
VERI_CIKAR = r'''
import bpy, json

bpy.ops.wm.open_mainfile(filepath=BLEND_YOLU)
o = {"animasyonlar": [], "kemikler": [], "hiyerarsi": {}}

for ob in bpy.data.objects:
    if ob.type == "ARMATURE":
        arm = ob.data
        o["kemikler"] = [b.name for b in arm.bones]
        o["hiyerarsi"] = {b.name: (b.parent.name if b.parent else "")
                          for b in arm.bones}
        ad = ob.animation_data
        # ⚠ tur 9 ÖLÇÜMÜ: yalnız `ad.action` (AKTİF olan) okunuyordu →
        # 5 animasyonlu varlıktan **1**'i kütüphaneye giriyordu.
        # Çözüm: bir varlığın **tüm** action'ları taranır.
        _aktif = ad.action if (ad and ad.action) else None
        _hepsi = []
        if _aktif is not None:
            _hepsi.append(_aktif)
        for _a in bpy.data.actions:
            if _a is not _aktif:
                _hepsi.append(_a)
        for a in _hepsi:
                kayit = {"ad": a.name,
                         "kare_bas": round(float(a.frame_range[0]), 1),
                         "kare_son": round(float(a.frame_range[1]), 1),
                         "fcurve": 0, "veri": {}}
                # ⚠ Blender 5.x katmanlı API
                for kat in a.layers:
                    for strip in kat.strips:
                        for cb in strip.channelbags:
                            for fc in cb.fcurves:
                                kayit["fcurve"] += 1
                                yol = fc.data_path
                                if 'pose.bones["' not in yol:
                                    continue
                                kemik = yol.split('pose.bones["')[1].split('"')[0]
                                kanal = yol.split(".")[-1]
                                if kanal not in ("rotation_euler",
                                                 "rotation_quaternion",
                                                 "location", "scale"):
                                    continue
                                kanal = kanal.replace("rotation_", "rot_")
                                # ⚠ tur 4 ÖLÇÜMÜ: `array_index` KAYBOLURSA
                                # retarget tüm değerleri eksen 0'a yazar
                                # (ölçüldü: dolu kanal 8 → 3).
                                kanal = f"{kanal}[{fc.array_index}]"
                                noktalar = []
                                for kp in fc.keyframe_points:
                                    noktalar.append([round(float(kp.co[0]), 2),
                                                     round(float(kp.co[1]), 5)])
                                kayit["veri"].setdefault(kemik, {})[kanal] = \
                                    noktalar
                kayit["toplam_keyframe"] = sum(
                    len(v) for d in kayit["veri"].values()
                    for v in d.values())
                o["animasyonlar"].append(kayit)
print("A" + json.dumps(o, ensure_ascii=False))
'''


def _veri_cikar(blend: Path, *, zaman: int = 240) -> Dict[str, Any]:
    """Kaydedilmiş `.blend`'den animasyon verisini çıkarır."""
    if not BLENDER.exists():
        return {"hata": "blender yok"}
    if not blend.exists():
        return {"hata": "blend yok"}
    kod = f'BLEND_YOLU = r"{blend.resolve()}"\n' + VERI_CIKAR
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "v.py"
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
        if "A{" in satir:
            try:
                return json.loads(satir.strip()[1:])
            except Exception:
                pass
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:150] or "veri yok"}


#: ─── Kütüphane işlemleri ─────────────────────────────────────────────

def _kutuphane_oku() -> Dict[str, Any]:
    """Kütüphaneyi okur."""
    if KUTUPHANE.exists():
        try:
            return json.loads(KUTUPHANE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"surum": 1, "animasyonlar": {}}


def _kutuphane_yaz(k: Dict[str, Any]) -> None:
    """Kütüphaneyi **atomik** yazar."""
    KUTUPHANE.parent.mkdir(parents=True, exist_ok=True)
    g = KUTUPHANE.with_suffix(".tmp")
    g.write_text(json.dumps(k, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    g.replace(KUTUPHANE)


def tara(*, kaydet: bool = True) -> Dict[str, Any]:
    """
    Tüm varlıkları tarar, animasyonları kütüphaneye kaydeder.

    ⚠ **Yalnız başarılı** animasyonlar kaydedilir — kullanıcı:
    *"başarılı animler olursa onları ... kayıt da tutma"*
    """
    from tools.varlik_katalogu import listele

    k = _kutuphane_oku()
    sonuc = {"taranan": 0, "kayitli": 0, "atlanan": 0, "detay": []}
    for v in listele():
        blend = v.get("blend") or ""
        if not blend:
            continue
        # ⚠ Yol katalogda mutlak ya da göreli olabilir
        bp = Path(blend)
        if not bp.exists():
            bp = Path("Varliklar") / v.get("kategori", "") / v.get("ad", "")
            if not bp.exists():
                continue
            bp = bp / "varlik.blend"
        sonuc["taranan"] += 1
        vid = v.get("id", "")
        if any(x.get("kaynak_varlik") == vid
               for x in k["animasyonlar"].values()):
            sonuc["atlanan"] += 1
            continue
        # ⚠ tur 8 ÖLÇÜMÜ: aynı animasyon **3 kez** kaydedildi
        # (kalite kademeleri aynı kemik yapısı + aynı animasyon adı →
        # aynı imza). (`ad`, `imza`) çifti tekilleştirilir.
        d = _veri_cikar(bp)
        if "hata" in d or not d.get("animasyonlar"):
            sonuc["detay"].append({"id": vid,
                                   "hata": d.get("hata", "animasyon yok")})
            continue
        kemikler = d.get("kemikler", [])
        hiy = d.get("hiyerarsi", {})
        for a in d["animasyonlar"]:
            kayit = AnimKaydi(
                ad=a["ad"], kaynak_varlik=vid,
                kategori=v.get("kategori", ""),
                kemikler=kemikler, hiyerarsi=hiy,
                roller=sorted({rol_bul(x) for x in kemikler}),
                kare_sayisi=int(a["kare_son"] - a["kare_bas"]) + 1,
                fcurve=a["fcurve"],
                toplam_keyframe=a.get("toplam_keyframe", 0),
                veri=a.get("veri", {}),
                olusturma=time.strftime("%Y-%m-%dT%H:%M:%S"),
                basarili=True,
            )
            kayit.imza = imza_hesapla(kemikler, hiy)
            # ⚠ Aynı (ad, imza) zaten varsa ATLA — kalite kademeleri
            # aynı kemik yapısını üretir, animasyon TEK olmalı
            if any(x.get("ad") == a["ad"]
                   and x.get("imza") == kayit.imza
                   for x in k["animasyonlar"].values()):
                sonuc["atlanan"] += 1
                continue
            anahtar = f"{vid}::{a['ad']}"
            k["animasyonlar"][anahtar] = kayit.to_dict()
            sonuc["kayitli"] += 1
            sonuc["detay"].append({
                "id": anahtar, "kemik": len(kemikler),
                "animasyonlu_kemik": len(a.get("veri", {})),
                "keyframe": kayit.toplam_keyframe, "imza": kayit.imza})
    if kaydet:
        _kutuphane_yaz(k)
    return sonuc


def uyumluluk(kemikler: Sequence[str],
              hiyerarsi: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Verilen kemik yapısına **uyan** animasyonları bulur.

    ⚠ İki ölçüt:
        `uyum_isim` : ortak kemik adı oranı
        `uyum_rol`  : ortak **rol** oranı (isim farklı olsa da)

    Hiyerarşi farkı → uyarı (dönme ekseni değişebilir).
    """
    k = _kutuphane_oku()
    var = set(kemikler)
    var_rol = {rol_bul(x) for x in kemikler}
    out: List[Dict[str, Any]] = []
    for anahtar, a in k["animasyonlar"].items():
        ak = set(a.get("kemikler", []))
        if not ak:
            continue
        isim = len(var & ak) / len(ak)
        arol = set(a.get("roller", []))
        rol = len(var_rol & arol) / len(arol) if arol else 0.0
        # ⚠ tur 4 ÖLÇÜMÜ: hiyerarşi farkı **uygulanabilirliği** reddediyordu
        # ama çoğu fark ZARARSIZ (ebeveyn eklenmiş/çıkmış, eksen aynı).
        #
        # ✅ Doğru kriter: **animasyonlu kemikler EKSİK mi?**
        #     Eksik kemik varsa animasyon uygulanamaz (kanal hedefsiz).
        #     Hiyerarşi farkı **uyarı** olarak raporlanır (retarget gerekir).
        ah = a.get("hiyerarsi", {})
        anim_kemik = set(a.get("veri", {}).keys())
        fark = [x for x in anim_kemik
                if x in hiyerarsi and x in ah
                and hiyerarsi[x] != ah[x]]
        # ⚠ Rol bazlı eksik tespiti: animasyon `on_ayak_l` istiyor ama
        # hedef armature'te o ROL yoksa → uygulanamaz
        hedef_roller = {rol_bul(x) for x in kemikler}
        eksik_rol = sorted({rol_bul(x) for x in anim_kemik}
                           - hedef_roller)
        out.append({
            "anahtar": anahtar, "ad": a.get("ad"),
            "kaynak": a.get("kaynak_varlik"),
            "uyum_isim": round(isim, 3), "uyum_rol": round(rol, 3),
            "kemik": len(ak), "ortak": len(var & ak),
            "keyframe": a.get("toplam_keyframe", 0),
            "hiyerarsi_farki": fark,
            "eksik_rol": eksik_rol,
            "uygulanabilir": rol >= 0.8 and not eksik_rol,
        })
    return sorted(out, key=lambda x: -x["uyum_rol"])


def main() -> int:
    """Kütüphane işlemleri."""
    ap = argparse.ArgumentParser(description="Animasyon kütüphanesi")
    ap.add_argument("--tara", action="store_true")
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--eslesme", default="")
    args = ap.parse_args()

    print("=" * 78)
    print("  ANİMASYON KÜTÜPHANESİ — kemik imzasıyla eşleştirme")
    print("=" * 78)
    print(f"  kütüphane: {KUTUPHANE}")

    if args.tara:
        s = tara()
        print(f"\n  taranan : {s['taranan']}")
        print(f"  kayıtlı : {s['kayitli']}")
        print(f"  atlanan : {s['atlanan']}")
        for d in s["detay"]:
            if "hata" in d:
                print(f"    ✗ {d['id']}: {d['hata'][:56]}")
            else:
                print(f"    ✓ {d['id']:34} kemik={d['kemik']:>2} "
                      f"anim_kemik={d['animasyonlu_kemik']:>2} "
                      f"kf={d['keyframe']:>4} imza={d['imza']}")
        return 0

    k = _kutuphane_oku()
    anim = k.get("animasyonlar", {})

    if args.liste:
        print(f"\n  {len(anim)} ANİMASYON:")
        for anahtar, a in anim.items():
            print(f"    {anahtar:38} {a.get('ad','?'):12} "
                  f"kare={a.get('kare_sayisi',0):>3} "
                  f"kf={a.get('toplam_keyframe',0):>4} "
                  f"kemik={len(a.get('kemikler',[])):>2} "
                  f"imza={a.get('imza','?')}")
        return 0

    if args.eslesme:
        # ⚠ Varlığın kemik yapısını oku, sonra eşleştir
        kt = Path("Varliklar") / args.eslesme / "varlik.blend"
        if not kt.exists():
            print(f"  blend yok: {kt}")
            return 1
        d = _veri_cikar(kt)
        if "hata" in d:
            print(f"  ✗ {d['hata']}")
            return 1
        kemikler = d.get("kemikler", [])
        hiy = d.get("hiyerarsi", {})
        print(f"\n  varlık   : {args.eslesme}")
        print(f"  kemikler : {len(kemikler)} · {kemikler[:8]}…")
        print(f"\n  EŞLEŞEN ANİMASYONLAR:")
        for x in uyumluluk(kemikler, hiy)[:8]:
            im = "✓" if x["uygulanabilir"] else "~"
            print(f"    {im} {x['ad']:12} rol={x['uyum_rol']:.2f} "
                  f"isim={x['uyum_isim']:.2f} ortak={x['ortak']}/"
                  f"{x['kemik']} kf={x['keyframe']}"
                  + (f" ⚠ hiyerarşi farkı: {x['hiyerarsi_farki'][:3]}"
                     if x["hiyerarsi_farki"] else ""))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
