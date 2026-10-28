"""
Doku Bulucu — Texture Arama, Seçim ve Uygulama
================================================
⚠ Kullanıcı hedefi (B)
----------------------
*"texture ekleme, texture bulup seçime sunma"*

⚠ Üç aşama
----------
```
1. BUL    → dokuyu nereden buluruz? (yerel kütüphane + çevrimiçi)
2. SEÇ    → hangisi uygun? (puanlama + SUNMA)
3. UYGULA → seçileni Blender'da kullan (UV + ShaderNodeTexImage)
```

⚠ Ölçülen kısıt
---------------
```
Çevrimiçi arama: `web_fetch` HTML döner, İKİLİ dosya (PNG/JPG) İNDİREMEZ
Çözüm          : (a) yerel doku kütüphanesi tara
                 (b) çevrimiçi KAYNAK LİSTESİ sun (kullanıcı indirsin)
                 (c) prosedürel doku üret (dosya gerekmez)
```

⚠ Neden "seçime sunma" önemli?
------------------------------
Kullanıcı: *"texture bulup **seçime sunma**"* — yani otomatik seçmek
yetmez, **seçenekleri göstermek** gerekir. Bu betik hem **puanlar**
hem **listeler**.

Kullanım
--------
    python tools/doku_bulucu.py --tara
    python tools/doku_bulucu.py --ara "tas pürüzlü"
    python tools/doku_bulucu.py --prosedurel kaya --sun
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

DOKU_KOK = Path("Varliklar/doku_kutuphanesi")
DOKU_INDEKS = Path("Varliklar/doku_indeksi.json")

#: ⚠ Desteklenen biçimler
BICIMLER = (".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".exr",
            ".hdr", ".bmp", ".webp")

#: ⚠ Çevrimiçi doku kaynakları — **lisans bilgisiyle**
#:
#: Kullanıcı: *"texture bulup seçime sunma"*
#: → Bu liste **sunulur**; indirme kullanıcıya bırakılır
#:   (ikili dosya indirme kısıtı ölçüldü)
KAYNAKLAR: Sequence[Dict[str, str]] = (
    {"ad": "Poly Haven", "url": "https://polyhaven.com/textures",
     "lisans": "CC0 (kamu malı)", "not": "En kaliteli PBR, ücretsiz"},
    {"ad": "ambientCG", "url": "https://ambientcg.com",
     "lisans": "CC0", "not": "PBR setleri, 2K/4K"},
    {"ad": "Textures.com", "url": "https://www.textures.com",
     "lisans": "Ücretsiz kota (15/gün)", "not": "Geniş arşiv"},
    {"ad": "OpenGameArt", "url": "https://opengameart.org",
     "lisans": "Karışık (kontrol et)", "not": "Oyun dokuları"},
    {"ad": "CC0 Textures", "url": "https://cc0textures.com",
     "lisans": "CC0", "not": "Poly Haven ile birleşti"},
    {"ad": "Blender Kit", "url": "https://www.blenderkit.com",
     "lisans": "Karışık", "not": "Blender eklentisi olarak"},
)

#: ⚠ Doku **türleri** — hangi kanal ne için
DOKU_TURLERI: Dict[str, str] = {
    "renk": "Base Color / Albedo — ana renk",
    "pürüzlülük": "Roughness — yüzey pürüzlülüğü (gri)",
    "normal": "Normal Map — yüzey kabartma (mor)",
    "yükseklik": "Height / Displacement — gerçek kabartma",
    "metal": "Metallic — metalik maske (siyah/beyaz)",
    "ortam_oklüzyon": "AO — gölge derinliği",
    "saydamlık": "Alpha — şeffaflık",
}

#: ⚠ Prosedürel doku **tarifleri** — dosya gerekmez
#:
#: Ölçülen avantaj: UV'ye bağımlı değil, sonsuz çözünürlük, dosya yok
PROSEDUREL: Dict[str, Dict[str, Any]] = {
    "kaya": {
        "aciklama": "Pürüzlü taş yüzeyi",
        "dugumler": [
            ("ShaderNodeTexNoise", {"Scale": 8.0, "Detail": 6.0,
                                    "Roughness": 0.6}),
            ("ShaderNodeBump", {"Strength": 0.25}),
        ],
        "renk": (0.32, 0.30, 0.28),
        "puruzluluk": 0.85,
    },
    "toprak": {
        "aciklama": "Yumuşak toprak / çamur",
        "dugumler": [
            ("ShaderNodeTexNoise", {"Scale": 5.0, "Detail": 4.0}),
            ("ShaderNodeBump", {"Strength": 0.35}),
        ],
        "renk": (0.22, 0.16, 0.10),
        "puruzluluk": 0.95,
    },
    "tahta": {
        "aciklama": "Ahşap damarı (dalga dokusu)",
        "dugumler": [
            ("ShaderNodeTexWave", {"Scale": 3.0, "Distortion": 6.0}),
            ("ShaderNodeBump", {"Strength": 0.2}),
        ],
        "renk": (0.35, 0.22, 0.11),
        "puruzluluk": 0.7,
    },
    "mermer": {
        "aciklama": "Mermer damarı",
        "dugumler": [
            ("ShaderNodeTexNoise", {"Scale": 3.0, "Detail": 8.0}),
            ("ShaderNodeTexWave", {"Scale": 2.0, "Distortion": 12.0}),
        ],
        "renk": (0.85, 0.84, 0.80),
        "puruzluluk": 0.25,
    },
    "deri": {
        "aciklama": "Hayvan derisi (pullu)",
        "dugumler": [
            ("ShaderNodeTexVoronoi", {"Scale": 28.0}),
            ("ShaderNodeBump", {"Strength": 0.35}),
        ],
        "renk": (0.16, 0.32, 0.16),
        "puruzluluk": 0.65,
    },
    "metal": {
        "aciklama": "İşlenmiş metal",
        "dugumler": [
            ("ShaderNodeTexNoise", {"Scale": 40.0, "Detail": 3.0}),
            ("ShaderNodeBump", {"Strength": 0.08}),
        ],
        "renk": (0.62, 0.64, 0.66),
        "puruzluluk": 0.3,
        "metalik": 1.0,
    },
    "kumas": {
        "aciklama": "Dokuma kumaş",
        "dugumler": [
            ("ShaderNodeTexChecker", {"Scale": 60.0}),
            ("ShaderNodeBump", {"Strength": 0.15}),
        ],
        "renk": (0.55, 0.20, 0.20),
        "puruzluluk": 0.9,
    },
}


@dataclass
class DokuKaydi:
    """Kütüphanedeki tek doku."""

    id: str = ""
    dosya: str = ""
    ad: str = ""
    tur: str = ""            # renk / pürüzlülük / normal / ...
    kategori: str = ""       # kaya / toprak / tahta / ...
    genislik: int = 0
    yukseklik: int = 0
    kb: float = 0.0
    kaynak: str = ""         # "yerel" / "polyhaven" / "prosedurel"
    lisans: str = ""
    etiketler: List[str] = field(default_factory=list)
    puan: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _indeks_oku() -> Dict[str, Any]:
    """Doku indeksini okur."""
    if DOKU_INDEKS.exists():
        try:
            return json.loads(DOKU_INDEKS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"surum": 1, "dokular": {}}


def _indeks_yaz(k: Dict[str, Any]) -> None:
    """İndeksi **atomik** yazar."""
    DOKU_INDEKS.parent.mkdir(parents=True, exist_ok=True)
    g = DOKU_INDEKS.with_suffix(".tmp")
    g.write_text(json.dumps(k, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    g.replace(DOKU_INDEKS)


#: ⚠ Dosya adından **tür** ve **kategori** çıkarır.
#:
#: Ölçülen desenler (Poly Haven / ambientCG adlandırması):
#:     `rock_04_diff_2k.png`     → kategori=rock, tür=renk
#:     `rock_04_rough_2k.png`    → tür=pürüzlülük
#:     `rock_04_nor_gl_2k.png`   → tür=normal
TUR_DESENLER: Sequence[Tuple[str, Sequence[str]]] = (
    ("normal", ("normal", "_nor_", "_nrm_", "normalgl", "nor_gl")),
    ("pürüzlülük", ("rough", "roughness", "pürüz", "puruz")),
    ("metal", ("metal", "metallic", "metalness")),
    ("yükseklik", ("height", "disp", "displacement", "bump")),
    ("ortam_oklüzyon", ("ao", "ambientocclusion", "occlusion")),
    ("saydamlık", ("alpha", "opacity", "mask")),
    ("renk", ("diff", "diffuse", "albedo", "col", "color", "basecolor",
              "base_color")),
)

KATEGORI_ANAHTAR: Sequence[Tuple[str, Sequence[str]]] = (
    ("kaya", ("rock", "kaya", "stone", "cliff", "granite", "boulder")),
    ("toprak", ("ground", "dirt", "soil", "mud", "sand", "toprak", "gravel")),
    ("tahta", ("wood", "tahta", "plank", "bark")),
    ("metal", ("metal", "steel", "iron", "rust", "metal")),
    ("mermer", ("marble", "mermer", "tile", "granite")),
    ("kumas", ("fabric", "cloth", "kumas", "leather", "deri")),
    ("bitki", ("grass", "leaf", "moss", "plant", "cicek", "forest")),
    ("duvar", ("brick", "wall", "concrete", "plaster", "tile")),
    ("su", ("water", "su", "ice", "buz", "lava")),
    ("gökyüzü", ("sky", "cloud", "hdri", "environment")),
)


def _siniflandir(dosya: Path) -> Tuple[str, str, List[str]]:
    """Dosya adından (kategori, tür, etiketler) çıkarır."""
    ad = dosya.stem.lower().replace("-", "_")
    tur = ""
    for t, desenler in TUR_DESENLER:
        if any(d in ad for d in desenler):
            tur = t
            break
    kat = ""
    for k, desenler in KATEGORI_ANAHTAR:
        if any(d in ad for d in desenler):
            kat = k
            break
    # Etiketler: adı parçala, sayıları at
    etiketler = [p for p in ad.split("_")
                 if p and not p.isdigit() and len(p) > 2
                 and p not in ("2k", "4k", "1k", "8k", "gl", "nor",
                               "diff", "rough", "map")]
    return kat or "genel", tur or "renk", sorted(set(etiketler))[:6]


def tara(*, kaydet: bool = True) -> Dict[str, Any]:
    """
    Yerel doku kütüphanesini tarar.

    ⚠ Boyut okuma: PNG/JPG başlığından **PIL olmadan**.
    """
    k = _indeks_oku()
    sonuc = {"bulunan": 0, "yeni": 0, "atlanan": 0, "tur": {},
             "kategori": {}}
    if not DOKU_KOK.exists():
        DOKU_KOK.mkdir(parents=True, exist_ok=True)
        return sonuc
    for f in sorted(DOKU_KOK.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in BICIMLER:
            continue
        sonuc["bulunan"] += 1
        anahtar = hashlib.sha1(
            str(f.relative_to(DOKU_KOK)).encode("utf-8")).hexdigest()[:12]
        if anahtar in k["dokular"]:
            sonuc["atlanan"] += 1
            continue
        kat, tur, etiket = _siniflandir(f)
        g, y = _boyut_oku(f)
        kayit = DokuKaydi(
            id=anahtar, dosya=str(f), ad=f.stem, tur=tur, kategori=kat,
            genislik=g, yukseklik=y,
            kb=round(f.stat().st_size / 1024, 1),
            kaynak="yerel", lisans="bilinmiyor", etiketler=etiket)
        k["dokular"][anahtar] = kayit.to_dict()
        sonuc["yeni"] += 1
        sonuc["tur"][tur] = sonuc["tur"].get(tur, 0) + 1
        sonuc["kategori"][kat] = sonuc["kategori"].get(kat, 0) + 1
    if kaydet:
        _indeks_yaz(k)
    return sonuc


def _boyut_oku(yol: Path) -> Tuple[int, int]:
    """
    Görüntü boyutunu **başlıktan** okur (PIL gerekmez).

    ⚠ PNG: IHDR (offset 16) · JPEG: SOF marker taraması
    """
    try:
        with yol.open("rb") as f:
            bas = f.read(32)
            if bas[:8] == b"\x89PNG\r\n\x1a\n":
                import struct

                g, y = struct.unpack(">II", bas[16:24])
                return int(g), int(y)
            if bas[:2] == b"\xff\xd8":       # JPEG
                f.seek(2)
                while True:
                    b = f.read(1)
                    if not b:
                        break
                    if b != b"\xff":
                        continue
                    m = f.read(1)
                    while m == b"\xff":
                        m = f.read(1)
                    if m in (b"\xc0", b"\xc1", b"\xc2", b"\xc3"):
                        f.read(3)
                        h, w = 0, 0
                        import struct

                        y, g = struct.unpack(">HH", f.read(4))
                        return int(g), int(y)
                    boy = f.read(2)
                    if len(boy) < 2:
                        break
                    import struct

                    (n,) = struct.unpack(">H", boy)
                    f.seek(n - 2, 1)
    except Exception:
        pass
    return 0, 0


def ara(sorgu: str, *, adet: int = 10) -> List[Dict[str, Any]]:
    """
    Doku **arar ve puanlar** — seçime sunmak için.

    ⚠ Puanlama:
        +3  kategori tam eşleşme
        +2  tür eşleşmesi (renk öncelikli)
        +1  etiket eşleşmesi (her biri)
        +0.5 çözünürlük ≥ 1024
    """
    k = _indeks_oku()
    q = sorgu.lower()
    kelimeler = [x for x in q.replace(",", " ").split() if len(x) > 2]
    # ⚠ Tür ipucu sorguda mı?
    istenen_tur = ""
    for t, desenler in TUR_DESENLER:
        if any(d in q for d in desenler):
            istenen_tur = t
            break
    out: List[Dict[str, Any]] = []
    for d in k["dokular"].values():
        puan = 0.0
        kat = (d.get("kategori") or "").lower()
        ad = (d.get("ad") or "").lower()
        etiket = [x.lower() for x in (d.get("etiketler") or [])]
        for w in kelimeler:
            if w == kat:
                puan += 3.0
            elif w in kat or kat in w:
                puan += 2.0
            if w in ad:
                puan += 2.0
            if w in etiket:
                puan += 1.0
        if istenen_tur and d.get("tur") == istenen_tur:
            puan += 2.5
        elif not istenen_tur and d.get("tur") == "renk":
            puan += 1.5
        if (d.get("genislik") or 0) >= 1024:
            puan += 0.5
        if puan > 0:
            x = dict(d)
            x["puan"] = round(puan, 2)
            out.append(x)
    return sorted(out, key=lambda z: -z["puan"])[:adet]


def sun(sorgu: str, *, adet: int = 6) -> None:
    """
    Bulunanları **seçime sunar**.

    ⚠ Kullanıcı: *"texture bulup **seçime sunma**"*
    → Yalnız seçmek yetmez; **seçenek listesi** gösterilir.
    """
    print(f"\n  ═══ YEREL KÜTÜPHANE: '{sorgu}' ═══")
    yerel = ara(sorgu, adet=adet)
    if yerel:
        for i, d in enumerate(yerel, 1):
            print(f"    {i}. [{d['puan']:>4}] {d['ad'][:38]:38} "
                  f"{d['tur']:12} {d['kategori']:10} "
                  f"{d['genislik']}x{d['yukseklik']} {d['kb']}KB")
    else:
        print("    (yerel kütüphanede bulunamadı)")

    # ⚠ Prosedürel alternatif
    print(f"\n  ═══ PROSEDÜREL ALTERNATİF (dosya gerekmez) ═══")
    for ad, b in PROSEDUREL.items():
        if any(w in ad or w in b["aciklama"].lower() for w in
               sorgu.lower().split()):
            print(f"    ✓ {ad:10} {b['aciklama']:32} "
                  f"{len(b['dugumler'])} düğüm")
    else:
        if not any(any(w in ad or w in b["aciklama"].lower()
                       for w in sorgu.lower().split())
                   for ad, b in PROSEDUREL.items()):
            print("    (ilgisiz — tümü: "
                  f"{', '.join(PROSEDUREL.keys())})")

    # ⚠ Çevrimiçi kaynaklar
    print(f"\n  ═══ ÇEVRİMİÇİ KAYNAKLAR ═══")
    for s in KAYNAKLAR[:5]:
        print(f"    · {s['ad']:16} {s['lisans']:26} {s['url']}")


def main() -> int:
    """Doku bulucu."""
    ap = argparse.ArgumentParser(description="Doku bulucu")
    ap.add_argument("--tara", action="store_true")
    ap.add_argument("--ara", default="")
    ap.add_argument("--sun", action="store_true")
    ap.add_argument("--kaynaklar", action="store_true")
    ap.add_argument("--prosedurel", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  DOKU BULUCU — ara, puanla, seçime sun")
    print("=" * 78)
    print(f"  kütüphane: {DOKU_KOK.resolve()}")

    if args.tara:
        s = tara()
        print(f"\n  bulunan : {s['bulunan']}")
        print(f"  yeni    : {s['yeni']}")
        print(f"  atlanan : {s['atlanan']}")
        if s["tur"]:
            print(f"  TÜR     : {s['tur']}")
            print(f"  KATEGORİ: {s['kategori']}")
        else:
            print("\n  ⚠ kütüphane BOŞ — doku eklemek için:")
            print(f"    {DOKU_KOK.resolve()}\\<kategori>\\ "
                  "<dosya>.png")
        return 0

    if args.kaynaklar:
        print(f"\n  {len(KAYNAKLAR)} ÇEVRİMİÇİ KAYNAK:")
        for s in KAYNAKLAR:
            print(f"    · {s['ad']:16} {s['lisans']:26}")
            print(f"      {s['url']}")
            print(f"      {s['not']}")
        return 0

    if args.prosedurel:
        print(f"\n  {len(PROSEDUREL)} PROSEDÜREL DOKU:")
        for ad, b in PROSEDUREL.items():
            print(f"    {ad:10} {b['aciklama']:34} "
                  f"renk={b['renk']} pürüzlülük={b['puruzluluk']}"
                  + (f" metalik={b['metalik']}" if "metalik" in b else ""))
            for d, p in b["dugumler"]:
                print(f"        {d}  {p}")
        return 0

    if args.ara:
        sun(args.ara)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
