"""
Varlık Kataloğu — Kategorili, Meta Verili Kayıt
=================================================
⚠ Kullanıcı hedefi (G)
----------------------
*"oluşturulan varlıklar için kategorilere ayırıp bilgilerini de kayıt
altında tutabilir"*

⚠ Ayrıca (D)
------------
*"başarılı animleri olursa onları o model göre kayıt da tutma"*
→ Her varlık **kendi animasyonlarıyla** saklanır.

Neden katalog gerekli?
----------------------
```
Blender üretimi tek seferlik değil — **birikimli**:
    · Aynı kategori (kaya) 10 varyant → hangisi hangisi?
    · Her varlığın **armature'ü farklı** → animasyon uyumu?
    · Hangi varlık **hangi ayarlarla** üretildi → yeniden üretilebilir mi?
```

Katalog yapısı
--------------
```
Varliklar/
  katalog.json               ← tüm varlıkların dizini
  kaya/
    kaya_lowpoly_0001/
      varlik.blend · varlik.glb · meta.json · anim/ · texture/
```

Kullanım
--------
    python tools/varlik_katalogu.py --liste
    python tools/varlik_katalogu.py --istatistik
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

VARLIK_KOK = Path("Varliklar")
KATALOG = VARLIK_KOK / "katalog.json"

#: ⚠ Varlık kategorileri — kullanıcı hedefi: *"kategorilere ayırıp"*
KATEGORILER: Dict[str, Dict[str, Any]] = {
    "kaya": {
        "aciklama": "Kaya, taş, mineral",
        "armature_gerekli": False,
        "animasyon_gerekli": False,
        "uv_gerekli": True,
        "ornek_boyut": "0.5-3 m",
    },
    "bitki": {
        "aciklama": "Ağaç, çalı, çim, çiçek",
        "armature_gerekli": False,
        "animasyon_gerekli": False,
        "uv_gerekli": True,
        "ornek_boyut": "0.2-15 m",
    },
    "esya": {
        "aciklama": "Mobilya, kap, alet, silah",
        "armature_gerekli": False,
        "animasyon_gerekli": False,
        "uv_gerekli": True,
        "ornek_boyut": "0.1-3 m",
    },
    "mimari": {
        "aciklama": "Duvar, kapı, sütun, zemin",
        "armature_gerekli": False,
        "animasyon_gerekli": False,
        "uv_gerekli": True,
        "ornek_boyut": "0.5-20 m",
    },
    "karakter": {
        "aciklama": "İnsansı varlık (kahraman, NPC)",
        "armature_gerekli": True,
        "animasyon_gerekli": True,
        "uv_gerekli": True,
        "ornek_boyut": "1.5-2.2 m",
    },
    "yaratik": {
        "aciklama": "Canavar, hayvan, yaratık",
        "armature_gerekli": True,
        "animasyon_gerekli": True,
        "uv_gerekli": True,
        "ornek_boyut": "0.3-8 m",
    },
    "arac": {
        "aciklama": "Araba, tekne, uçan araç",
        "armature_gerekli": False,
        "animasyon_gerekli": False,
        "uv_gerekli": True,
        "ornek_boyut": "1-10 m",
    },
    "sahne": {
        "aciklama": "Kompozit sahne (birden çok nesne)",
        "armature_gerekli": False,
        "animasyon_gerekli": False,
        "uv_gerekli": False,
        "ornek_boyut": "değişken",
    },
}

#: ⚠ Kalite kademeleri — kullanıcı: *"low-poly veya kaliteli"*
KALITE: Dict[str, Dict[str, Any]] = {
    "lowpoly": {
        "aciklama": "Düşük poligon, düz gölge, oyun-içi",
        "azami_yuz": 500,
        "duz_golge": True,
    },
    "orta": {
        "aciklama": "Orta detay, karışık gölge",
        "azami_yuz": 5000,
        "duz_golge": False,
    },
    "kaliteli": {
        "aciklama": "Yüksek detay, yumuşak gölge, render",
        "azami_yuz": 200000,
        "duz_golge": False,
    },
}


@dataclass
class VarlikMeta:
    """Bir varlığın **tam meta verisi**."""

    # ── Kimlik ──
    id: str = ""
    ad: str = ""
    kategori: str = ""
    kalite: str = ""
    olusturma: str = ""

    # ── Geometri (Blender'dan ölçülür) ──
    vertex: int = 0
    kenar: int = 0
    yuz: int = 0

    # ── Yüzey ──
    uv_katman: int = 0
    uv_ok: bool = False
    malzeme: List[str] = field(default_factory=list)
    doku: List[str] = field(default_factory=list)
    duz_golge: bool = False

    # ── Armature (C hedefi) ──
    armature_var: bool = False
    kemik_sayisi: int = 0
    kemik_adlari: List[str] = field(default_factory=list)
    automatic_weight: bool = False
    vertex_group: int = 0

    # ── Animasyon (D hedefi) ──
    animasyonlar: List[str] = field(default_factory=list)

    # ── Üretim ──
    uret_KOD_hash: str = ""
    parametreler: Dict[str, Any] = field(default_factory=dict)
    blender_surum: str = ""

    # ── Dosyalar ──
    blend: str = ""
    glb: str = ""
    boyut: List[float] = field(default_factory=list)
    sure_s: float = 0.0

    # ── Önizleme (tur 17, Hedef G) ──
    #: ⚠ 4 açıdan render edilmiş PNG yolları
    #: {"on": "...", "yan": "...", "uc_dort": "...", "ust": "..."}
    onizleme: Dict[str, str] = field(default_factory=dict)
    #: Önizleme klasörü
    onizleme_klasor: str = ""
    #: Kaç açı render edildi (0-4)
    onizleme_aci: int = 0

    # ── Animasyon önizleme (tur 24, Hedef D) ──
    #: ⚠ {animasyon: {"klasor": ..., "kare": N, "png": [yollar]}}
    anim_onizleme: Dict[str, Any] = field(default_factory=dict)
    #: Kaç animasyon render edildi
    anim_onizleme_sayisi: int = 0
    #: Animasyon önizleme kök klasörü
    anim_onizleme_klasor: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _hash(kod: str) -> str:
    """Üretim kodunun kısa özeti (yeniden üretilebilirlik)."""
    return hashlib.sha1(kod.encode("utf-8")).hexdigest()[:12]


def _katalog_oku() -> Dict[str, Any]:
    """Kataloğu okur (yoksa boş iskelet)."""
    if KATALOG.exists():
        try:
            return json.loads(KATALOG.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"surum": 1, "varliklar": {}, "sayac": {}}


def _katalog_yaz(k: Dict[str, Any]) -> None:
    """Kataloğu **atomik** yazar."""
    KATALOG.parent.mkdir(parents=True, exist_ok=True)
    gecici = KATALOG.with_suffix(".tmp")
    gecici.write_text(json.dumps(k, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    gecici.replace(KATALOG)


def kaydet(meta: VarlikMeta, *, klasor: Path) -> str:
    """
    Varlığı kataloğa kaydeder.

    ⚠ Kimlik: `kategori_ad_000N` — sayaç kategoriye göre artar,
    böylece aynı addan birden çok varyant **çakışmaz**.
    """
    k = _katalog_oku()
    sayac = k.setdefault("sayac", {})
    anahtar = f"{meta.kategori}_{meta.ad}"
    sayac[anahtar] = sayac.get(anahtar, 0) + 1
    meta.id = f"{meta.kategori}_{meta.ad}_{sayac[anahtar]:04d}"
    if not meta.olusturma:
        meta.olusturma = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Meta dosyası varlık klasörüne
    klasor.mkdir(parents=True, exist_ok=True)
    (klasor / "meta.json").write_text(
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    meta.blend = meta.blend or str(klasor / "varlik.blend")
    meta.glb = meta.glb or str(klasor / "varlik.glb")

    k["varliklar"][meta.id] = meta.to_dict()
    _katalog_yaz(k)
    return meta.id


def varlik_klasoru(kategori: str, ad: str) -> Path:
    """Varlığın klasör yolunu üretir."""
    return VARLIK_KOK / kategori / ad


def listele(*, kategori: str = "") -> List[Dict[str, Any]]:
    """Katalogdaki varlıkları listeler."""
    k = _katalog_oku()
    v = list(k.get("varliklar", {}).values())
    if kategori:
        v = [x for x in v if x.get("kategori") == kategori]
    return sorted(v, key=lambda x: x.get("olusturma", ""))


def istatistik() -> Dict[str, Any]:
    """Katalog istatistiği — kategori/kalite kırılımı."""
    v = listele()
    kategori: Dict[str, int] = {}
    kalite: Dict[str, int] = {}
    armature = anim = uv = oniz = 0
    anim_oniz = 0
    anim_kare = 0
    for x in v:
        kategori[x.get("kategori", "?")] = (
            kategori.get(x.get("kategori", "?"), 0) + 1)
        kalite[x.get("kalite", "?")] = kalite.get(x.get("kalite", "?"), 0) + 1
        if x.get("armature_var"):
            armature += 1
        if x.get("animasyonlar"):
            anim += 1
        if x.get("uv_ok"):
            uv += 1
        if x.get("onizleme_aci", 0) >= 4:
            oniz += 1
        # ⚠ tur 24: animasyon önizlemesi
        if x.get("anim_onizleme_sayisi", 0) > 0:
            anim_oniz += 1
        anim_kare += x.get("anim_onizleme_sayisi", 0)
    return {"toplam": len(v), "kategori": kategori, "kalite": kalite,
            "armaturelu": armature, "animasyonlu": anim, "uvli": uv,
            "onizlemeli": oniz,
            # ⚠ tur 24: animasyon önizlemesi
            "anim_onizlemeli": anim_oniz,
            "anim_onizleme_toplam": anim_kare}


def main() -> int:
    """Katalog işlemleri."""
    ap = argparse.ArgumentParser(description="Varlık kataloğu")
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--istatistik", action="store_true")
    ap.add_argument("--kategori", default="")
    ap.add_argument("--kategoriler", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  VARLIK KATALOĞU")
    print("=" * 78)
    print(f"  kök: {VARLIK_KOK.resolve()}")

    if args.kategoriler:
        print(f"\n  {len(KATEGORILER)} KATEGORİ:")
        for ad, b in KATEGORILER.items():
            print(f"    {ad:10} armature={'✓' if b['armature_gerekli'] else '·'}"
                  f" anim={'✓' if b['animasyon_gerekli'] else '·'}"
                  f" uv={'✓' if b['uv_gerekli'] else '·'}"
                  f"  {b['aciklama']}")
        print(f"\n  {len(KALITE)} KALİTE KADEMESİ:")
        for ad, b in KALITE.items():
            print(f"    {ad:10} azami yüz {b['azami_yuz']:>7,}"
                  f" · {b['aciklama']}")
        return 0

    if args.istatistik:
        s = istatistik()
        print(f"\n  toplam varlık : {s['toplam']}")
        print(f"  armature'lü   : {s['armaturelu']}")
        print(f"  animasyonlu   : {s['animasyonlu']}")
        print(f"  UV'li         : {s['uvli']}")
        print(f"  Önizlemeli    : {s.get('onizlemeli', 0)}")
        print(f"  Anim önizleme : {s.get('anim_onizlemeli', 0)} varlık · "
              f"{s.get('anim_onizleme_toplam', 0)} animasyon")
        print(f"\n  KATEGORİ: {s['kategori']}")
        print(f"  KALİTE  : {s['kalite']}")
        return 0

    if args.liste:
        v = listele(kategori=args.kategori)
        print(f"\n  {len(v)} VARLIK:")
        for x in v:
            # ⚠ tur 17: 4. bayrak = önizleme (P)
            bayrak = ("A" if x.get("armature_var") else "·") + \
                     ("N" if x.get("animasyonlar") else "·") + \
                     ("U" if x.get("uv_ok") else "·") + \
                     ("P" if x.get("onizleme_aci", 0) >= 4 else "·") + \
                     ("M" if x.get("anim_onizleme_sayisi", 0) > 0 else "·")
            print(f"    [{bayrak}] {x.get('id','?'):34} "
                  f"v={x.get('vertex',0):>6} "
                  f"y={x.get('yuz',0):>6} · {x.get('kalite','?')}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
