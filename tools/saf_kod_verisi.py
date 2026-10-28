"""
Saf Kod Verisi Üretici — Kod/Prose Ayrımı
==========================================
⚠ ÖLÇÜLEN KÖK NEDEN (tur 31)
-----------------------------
Model **kod yerine düz yazı** üretiyor:
```
'The first step is to create a surface tool...'
'Here, we defined the surface tool...'
'The result is a surface tool...'
```

Sebep — eğitim verisinde **yapılandırılmış prose**:
```
blender_api : 653/1375 not prose içeriyor (%47)
kod_ornek   :  72/135  not prose içeriyor (%53)

Örnek: '**Açıklama:** Sahneye Bézier eğrisinden oluşan bir daire ekler.'
```

`blender_api` notları `**Ad:** / **İmza:** / **Açıklama:**` alanları
taşıyor. Model bu **kalıbı** öğrenip kod yerine yazı üretiyor.

Çözüm: **Saf kod verisi**
--------------------------
    * Yalnız **kod blokları** (` ```python ... ``` `) alınır
    * Yapılandırılmış alanlar (`**Ad:**`) **atılır**
    * Yorum satırları **temizlenir** (model yorumu sürdürüyor)
    * Çıktı: yalnız çalıştırılabilir kod

Kullanım
--------
    python tools/saf_kod_verisi.py --analiz
    python tools/saf_kod_verisi.py --uret
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

SAF_VERI = Path(".neurocore/saf_kod_verisi.jsonl")

#: Kod kategorileri
KOD_KATEGORI = ("kod_uretim", "kod_ogrenme", "kod_ornek",
                "blender_api", "api_referans")

#: Kod bloklarını yakalayan desen
KOD_BLOK = re.compile(r"```(?:python|gdscript|py|gd)?\s*\n(.*?)```",
                      re.DOTALL)

#: Yapılandırılmış alan satırları — atılır
ALAN_DESEN = re.compile(
    r"^\s*\*\*(?:Ad|İmza|Imza|Parametreler|Açıklama|Aciklama|Konu|"
    r"Önek|Onek|Devam|Tam kod|Kaynak|Modül|Modul|Kullanım|Kullanim)"
    r":?\*\*", re.MULTILINE | re.IGNORECASE)


@dataclass
class SafKodRaporu:
    """Saf kod verisi raporu."""
    not_toplam: int = 0
    kod_blok: int = 0
    satir_toplam: int = 0
    prose_atilan: int = 0
    yorum_temizlenen: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "not_toplam": self.not_toplam, "kod_blok": self.kod_blok,
            "satir_toplam": self.satir_toplam,
            "prose_atilan": self.prose_atilan,
            "yorum_temizlenen": self.yorum_temizlenen,
        }


def kod_bloklari(metin: str) -> List[str]:
    """Metinden **kod bloklarını** çıkarır."""
    return [b.strip() for b in KOD_BLOK.findall(metin) if len(b.strip()) > 20]


def yorum_temizle(kod: str, *, godot: bool = False) -> str:
    """
    **Yorum satırlarını** temizler.

    ⚠ ÖLÇÜLEN SORUN: model yorum satırlarını **kalıp** olarak alıp
    sürdürüyor (`# Create a mesh` ×N). Yorumlar eğitimden çıkarılınca
    bu kalıp ortadan kalkar.
    """
    satirlar: List[str] = []
    for s in kod.splitlines():
        t = s.strip()
        if not t:
            satirlar.append(s)
            continue
        # Yalnız yorum satırı → atla
        if t.startswith("#"):
            continue
        # Satır sonu yorumu → temizle
        if "#" in s:
            # String içindeki # korunur (basit kural)
            if s.count('"') % 2 == 0 and s.count("'") % 2 == 0:
                s = s.split("#")[0].rstrip()
        satirlar.append(s)
    # Ardışık boş satırları sadeleştir
    temiz: List[str] = []
    for s in satirlar:
        if not s.strip() and temiz and not temiz[-1].strip():
            continue
        temiz.append(s)
    return "\n".join(temiz).strip()


def metinden_kod(metin: str) -> str:
    """
    Bir nottan **yalnız kod** çıkarır.

    Akış:
        1. Kod bloklarını bul
        2. Yoksa: yapılandırılmış alanları at, kalanı kod say
        3. Yorumları temizle
        4. Çok kısaysa boş döndür
    """
    # 1) Kod blokları
    bloklar = kod_bloklari(metin)
    if bloklar:
        kod = "\n\n".join(bloklar)
    else:
        # 2) Alanları at
        kod = ALAN_DESEN.sub("", metin)
        # Markdown başlıkları at
        kod = re.sub(r"^\s*#{1,6}\s+.*$", "", kod, flags=re.MULTILINE)
    # 3) Yorumları temizle
    kod = yorum_temizle(kod)
    # 4) Çok kısa / kod gibi değil
    if len(kod) < 40:
        return ""
    # Kod göstergesi ara
    if not any(k in kod for k in ("=", "(", "import ", "func ", "var ",
                                  "return", "bpy", "extends", "def ")):
        return ""
    return kod


def uret(maks_not: int = 8000, *, kod_tekrar: int = 3) -> SafKodRaporu:
    """Saf kod verisi üretir."""
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return SafKodRaporu()

    r = SafKodRaporu()
    with SAF_VERI.open("w", encoding="utf-8") as f:
        for n in k.vault.iter_notes(limit=maks_not):
            govde = (getattr(n, "body", "") or "").strip()
            kat = getattr(n, "category", "") or ""
            if len(govde) < 40:
                continue
            r.not_toplam += 1
            kod = metinden_kod(govde)
            if not kod:
                continue
            r.kod_blok += 1
            r.satir_toplam += len(kod.splitlines())
            # Kod ağırlığı
            tekrar = kod_tekrar if kat in KOD_KATEGORI else 1
            for _ in range(tekrar):
                f.write(json.dumps({"metin": kod + "\n\n", "kategori": kat},
                                   ensure_ascii=False) + "\n")
    k.shutdown()
    return r


def analiz(maks_not: int = 8000) -> SafKodRaporu:
    """Ne kadar kod çıkarılabileceğini ölçer."""
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return SafKodRaporu()

    r = SafKodRaporu()
    for n in k.vault.iter_notes(limit=maks_not):
        govde = (getattr(n, "body", "") or "").strip()
        if len(govde) < 40:
            continue
        r.not_toplam += 1
        bloklar = kod_bloklari(govde)
        if bloklar:
            r.kod_blok += 1
            r.satir_toplam += sum(len(b.splitlines()) for b in bloklar)
        # Prose var mı?
        if ALAN_DESEN.search(govde):
            r.prose_atilan += 1
    k.shutdown()
    return r


def main() -> int:
    """Saf kod verisi işlemlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="Saf kod verisi üretici")
    ap.add_argument("--analiz", action="store_true")
    ap.add_argument("--uret", action="store_true")
    ap.add_argument("--maks-not", type=int, default=8000)
    args = ap.parse_args()

    print("=" * 78)
    print("  SAF KOD VERİSİ — kod/prose ayrımı")
    print("=" * 78)

    sonuc: Dict[str, Any] = {}
    if args.analiz or not args.uret:
        a = analiz(maks_not=args.maks_not)
        print(f"  not: {a.not_toplam}")
        print(f"  kod bloğu içeren: {a.kod_blok}")
        print(f"  kod satırı: {a.satir_toplam}")
        print(f"  prose alanı olan: {a.prose_atilan}")
        sonuc["analiz"] = a.to_dict()
    if args.uret:
        r = uret(maks_not=args.maks_not)
        print(f"\n  ÜRETİLDİ: {r.kod_blok} nottan saf kod")
        print(f"  kod satırı: {r.satir_toplam}")
        print(f"  dosya: {SAF_VERI}")
        sonuc["uretim"] = r.to_dict()

    Path("_saf_kod.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
