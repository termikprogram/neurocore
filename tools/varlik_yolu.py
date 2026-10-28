"""
Varlık Yolu Çözümleyici — ORTAK yardımcı (tur 25)
====================================================
⚠ Neden ortak? (tur 25 ÖLÇÜMÜ)
-------------------------------
5+ araçta **aynı** yol çözümleme hatası vardı:

```
anim_uret    : Path("Varliklar") / hedef / "varlik.blend"
               hedef="Varliklar\\yaratik\\timsah_0007"
               → Varliklar\\Varliklar\\yaratik\\timsah_0007\\varlik.blend  ✗
```

Ve diğerleri **ID ile bulamıyordu**:
```
asset.gecis      : "varlık bulunamadı: Varliklar\\yaratik\\timsah_0007"
asset.onizleme   : aynı
asset.anim_onizleme : aynı
```

⚠ Kök neden: her araç kendi `_blend_bul`'ını yazmış, hiçbiri
**tam** değil.

Bu modül
--------
```
1. Tam `.blend` yolu
2. Tam klasör yolu (Varliklar içeren)
3. Göreli `kategori/ad`
4. Yalnız ad → katalog → kategori klasörleri
```

⚠ **Puanlı eşleştirme** (tur 22 dersi): TAM eşleşme önce, sonra
en uzun önek. Gevşek önek **yanlış dosya** seçer.

Kullanım
--------
    from tools.varlik_yolu import blend_bul, klasor_bul
    blend = blend_bul("timsah_0007")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logging.disable(logging.CRITICAL)

#: ⚠ Varlık kökü
KOK = Path("Varliklar")

#: ⚠ Kategori listesi (tur 1: 8 kategori, 6'sı üretiliyor)
KATEGORILER = ("kaya", "bitki", "esya", "mimari", "yaratik", "karakter",
               "arac", "sahne")


def _katalog() -> List[dict]:
    """Katalog kayıtları (hata olursa boş)."""
    try:
        from tools.varlik_katalogu import listele

        return list(listele())
    except Exception:
        return []


def blend_bul(hedef: str) -> Optional[Path]:
    """
    Varlık `.blend` yolunu bulur.

    ⚠ Kabul edilen biçimler:
        1. `Varliklar/yaratik/timsah_0007/varlik.blend`  (tam yol)
        2. `Varliklar/yaratik/timsah_0007`               (klasör)
        3. `yaratik/timsah_0007`                          (göreli)
        4. `timsah_0007`                                  (ad)
    """
    if not hedef:
        return None
    p = Path(hedef)

    # ⚠ 1. Tam `.blend` yolu
    if p.suffix == ".blend":
        if p.exists():
            return p
        # ⚠ Uzantı var ama yok → klasör olarak dene
        p = p.parent

    # ⚠ 2. Tam klasör yolu (Varliklar içeriyor)
    if "Varliklar" in p.parts:
        aday = p / "varlik.blend"
        if aday.exists():
            return aday
        # ⚠ `Varliklar` kökü göreli ise tekrar dene
        if p.is_absolute() and p.exists():
            aday = p / "varlik.blend"
            if aday.exists():
                return aday

    # ⚠ 3. `Varliklar` kökü (mutlak)
    aday = KOK / p / "varlik.blend"
    if aday.exists():
        return aday
    if p.is_absolute():
        aday = p / "varlik.blend"
        if aday.exists():
            return aday

    # ⚠ 4. Yalnız ad → kategori klasörleri
    kok = hedef.replace("\\", "/").split("/")[-1]
    for kat in KATEGORILER:
        aday = KOK / kat / kok / "varlik.blend"
        if aday.exists():
            return aday

    # ⚠ 5. Katalog → puanlı eşleştirme (tur 22 dersi)
    en_iyi = 0
    en_iyi_yol: Optional[Path] = None
    for v in _katalog():
        vid = str(v.get("id") or "")
        vad = str(v.get("ad") or "")
        b = Path(v.get("blend") or "")
        if not b.exists():
            continue
        if vid == kok or vad == kok:
            puan = 1000
        elif b.parent.name == kok:
            puan = 900
        elif vid.startswith(kok) or vad.startswith(kok):
            puan = 500 + len(kok)
        elif b.parent.name.startswith(kok):
            puan = 400 + len(kok)
        elif kok.startswith(vad) and vad:
            puan = 100 + len(vad)
        else:
            puan = 0
        if puan > en_iyi:
            en_iyi = puan
            en_iyi_yol = b
            if puan >= 1000:
                break
    if en_iyi_yol is not None:
        return en_iyi_yol

    # ⚠ 6. Derin arama (son çare — yavaş ama kesin)
    for kat in KATEGORILER:
        d = KOK / kat
        if not d.is_dir():
            continue
        for alt in d.iterdir():
            if alt.is_dir() and (alt.name == kok or kok in alt.name):
                aday = alt / "varlik.blend"
                if aday.exists():
                    return aday
    return None


def klasor_bul(hedef: str) -> Optional[Path]:
    """Varlık **klasörünü** bulur."""
    b = blend_bul(hedef)
    if b is not None:
        return b.parent
    p = Path(hedef)
    if "Varliklar" in p.parts and p.is_dir():
        return p
    aday = KOK / p
    if aday.is_dir():
        return aday
    return None


def kimlik(hedef: str) -> str:
    """Varlık kimliği (klasör adı)."""
    k = klasor_bul(hedef)
    if k is not None:
        return k.name
    return hedef.replace("\\", "/").split("/")[-1]


def dogrula(hedef: str) -> dict:
    """Yol çözümlemesini raporlar (hata ayıklama)."""
    b = blend_bul(hedef)
    return {
        "hedef": hedef,
        "blend": str(b) if b else None,
        "klasor": str(b.parent) if b else None,
        "kimlik": kimlik(hedef),
        "var": bool(b and b.exists()),
        "kb": round(b.stat().st_size / 1024, 1) if b and b.exists() else 0,
    }
