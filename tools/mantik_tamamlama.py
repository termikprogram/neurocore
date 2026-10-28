"""
Mantık Notlarını Kod Tamamlama Çiftine Çevirir
================================================
⚠ ÖLÇÜLEN SORUN (tur 12)
------------------------
Mantık notları **6× tekrarlanarak** eğitildi ama performans **düştü**:
```
qwen_havuz2 (temel)     : mantik 8/12 (%67)
qwen_mantik (6× mantık) : mantik 7/12 (%58)   ← GERİLEME
```

**Kök neden (ölçüldü):** Notlar **belge biçiminde**:
```
# İkili arama — python (temel)

**Kategori:** arama
**Dil:** python
**Varyant:** temel
**Karmaşıklık:** O(log n)

```python
def ikili_arama(dizi, hedef):
    ...
```
```

Model bunları **tamamlama** olarak değil, **belge** olarak öğreniyor.
Test ise **tamamlama** ölçüyor:
```
'def ikili_arama(...):\n    while alt <= ust:\n        if dizi[orta] < hedef:\n            alt = '  →  'orta + 1'
```

**Dağılım uyuşmazlığı** — model belge üretmeyi öğrendi, kod değil.

Çözüm: **Notu kod tamamlama çiftine çevir**
--------------------------------------------
    1. Her nottaki kod bloğunu al
    2. **2-4 satırlık önek** + **sonraki 1-2 satır devam** üret
    3. Çıktı: `önek\n<FIM>\ndevam` biçimi (FIM = fill-in-middle)

⚠ Ölçülen ders (tur 22-23): önek/devam çiftleri **kod tamamlamayı**
doğrudan öğretir; belge biçimi öğretmez.

Kullanım
--------
    python tools/mantik_tamamlama.py --istatistik
    python tools/mantik_tamamlama.py --uret
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Kod bloğu deseni (```dil ... ```)
KOD_BLOK = re.compile(r"```(\w+)\s*\n(.*?)```", re.DOTALL)

#: Hedef dosya
CIKTI = Path(".neurocore/mantik_tamamlama.jsonl")

#: Önek uzunlukları (satır)
ONEK_UZUNLUK: Sequence[int] = (2, 3, 4)


@dataclass
class CiftIstatistik:
    """Üretim istatistiği."""
    not_sayisi: int = 0
    kod_blok: int = 0
    cift: int = 0
    atlanan_kisa: int = 0
    atlanan_gecersiz: int = 0
    karakter: int = 0
    dil_dagilim: Dict[str, int] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.dil_dagilim is None:
            self.dil_dagilim = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "not_sayisi": self.not_sayisi, "kod_blok": self.kod_blok,
            "cift": self.cift, "atlanan_kisa": self.atlanan_kisa,
            "atlanan_gecersiz": self.atlanan_gecersiz,
            "karakter": self.karakter,
            "token_tahmini": self.karakter // 3,
            "dil_dagilim": self.dil_dagilim,
        }


def _gecerli(onek: str, devam: str, dil: str) -> bool:
    """
    Önek + devam **sözdizimsel olarak** anlamlı mı?

    ⚠ Python için `ast.parse` kullanılır; diğer diller için
    parantez/blok dengesi denetlenir (derleyici yok).
    """
    tam = onek.rstrip("\n") + "\n" + devam
    if dil == "python":
        import ast

        try:
            ast.parse(tam)
            return True
        except SyntaxError:
            # Açık bloğu kapatıp dene
            acik = 0
            for s in tam.splitlines():
                t = s.split("#")[0]
                acik += t.count("(") - t.count(")")
                acik += t.count("[") - t.count("]")
                acik += t.count("{") - t.count("}")
            try:
                ast.parse(tam.rstrip() + ")" * max(0, acik))
                return True
            except SyntaxError:
                return False
    # Diğer diller: parantez dengesi + kod göstergesi
    for ac, kapa in (("(", ")"), ("[", "]"), ("{", "}")):
        if tam.count(ac) < tam.count(kapa):
            return False
    return any(k in tam for k in ("=", "(", "return", "var ", "let ",
                                  "fn ", "func ", "->"))


def cift_uret(kod: str, dil: str) -> List[Tuple[str, str]]:
    """
    Kod bloğundan **önek/devam çiftleri** üretir.

    ⚠ Ölçülen: 2-4 satırlık önek en iyi sonuç veriyor (tur 23'te
    `kod_ornekleri` aracında doğrulandı).
    """
    satirlar = [s for s in kod.splitlines() if s.strip()]
    if len(satirlar) < 4:
        return []
    ciftler: List[Tuple[str, str]] = []
    for n in ONEK_UZUNLUK:
        if len(satirlar) <= n:
            continue
        onek = "\n".join(satirlar[:n])
        devam = "\n".join(satirlar[n:n + 2])
        if not devam.strip():
            continue
        ciftler.append((onek, devam))
    return ciftler


def _notlari_al() -> List[Tuple[str, str]]:
    """Kasadan mantık notlarını alır — (kategori, gövde)."""
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return []
    out: List[Tuple[str, str]] = []
    try:
        for n in k.vault.iter_notes(limit=300000):
            kat = getattr(n, "category", "") or ""
            if kat in ("kod_mantigi", "kod_mantigi_dil",
                       "kod_mantigi_uretilen"):
                body = getattr(n, "body", "") or ""
                if len(body) >= 40:
                    out.append((kat, body))
    finally:
        k.shutdown()
    return out


def uret(*, kasaya: bool = False) -> CiftIstatistik:
    """Tamamlama çiftlerini üretir."""
    s = CiftIstatistik()
    notlar = _notlari_al()
    s.not_sayisi = len(notlar)
    print(f"  mantık notu: {s.not_sayisi}")

    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    gorulen: set = set()
    with CIKTI.open("w", encoding="utf-8") as f:
        for kat, govde in notlar:
            for m in KOD_BLOK.finditer(govde):
                dil = (m.group(1) or "python").strip().lower()
                if dil in ("", "py"):
                    dil = "python"
                kod = m.group(2).strip()
                if len(kod) < 40:
                    s.atlanan_kisa += 1
                    continue
                s.kod_blok += 1
                for onek, devam in cift_uret(kod, dil):
                    if not _gecerli(onek, devam, dil):
                        s.atlanan_gecersiz += 1
                        continue
                    anahtar = onek.strip()[-80:]
                    if anahtar in gorulen:
                        continue
                    gorulen.add(anahtar)
                    kayit = (f"# Kod tamamlama ({dil})\n\n"
                             f"**Önek:**\n```{dil}\n{onek}\n```\n\n"
                             f"**Devam:**\n```{dil}\n{devam}\n```\n\n"
                             f"**Tam kod:**\n```{dil}\n{kod[:900]}\n```\n")
                    f.write(json.dumps({"metin": kayit,
                                        "kategori": "mantik_tamamlama",
                                        "dil": dil,
                                        "onek": onek,
                                        "devam": devam},
                                       ensure_ascii=False) + "\n")
                    s.cift += 1
                    s.karakter += len(kayit)
                    s.dil_dagilim[dil] = s.dil_dagilim.get(dil, 0) + 1
    return s


def main() -> int:
    """Üretimi çalıştırır."""
    ap = argparse.ArgumentParser(description="Mantık tamamlama çiftleri")
    ap.add_argument("--istatistik", action="store_true")
    ap.add_argument("--uret", action="store_true")
    ap.add_argument("--liste", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  MANTIK → KOD TAMAMLAMA ÇİFTLERİ")
    print("=" * 78)

    if args.liste:
        notlar = _notlari_al()
        for kat, govde in notlar[:5]:
            print(f"\n  [{kat}] {govde.splitlines()[0][:60]}")
            for m in list(KOD_BLOK.finditer(govde))[:1]:
                kod = m.group(2).strip()
                for onek, devam in cift_uret(kod, "python")[:2]:
                    print(f"    ÖNEK : {onek[-50:]!r}")
                    print(f"    DEVAM: {devam[:50]!r}")
        return 0

    s = uret()
    print(f"\n  kod bloğu        : {s.kod_blok}")
    print(f"  ÜRETİLEN ÇİFT    : {s.cift}")
    print(f"  atlanan (kısa)   : {s.atlanan_kisa}")
    print(f"  atlanan (geçersiz): {s.atlanan_gecersiz}")
    print(f"  karakter         : {s.karakter:,}")
    print(f"  token            : ~{s.karakter // 3:,}")
    print(f"  dil dağılımı     : {s.dil_dagilim}")
    print(f"  dosya            : {CIKTI}")

    Path("_mantik_tamamlama.json").write_text(
        json.dumps(s.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
