"""
Yerel Kaynak Kod Madencisi — Çok Dilli Gerçek Proje Kodu
==========================================================
⚠ ÖLÇÜLEN POTANSİYEL (tur 6)
----------------------------
```
E:\\PROJELER : 164.000+ kaynak dosya · 1,3 GB
  .py       65.781 dosya   857 MB   python
  .js       42.054 dosya   188 MB   javascript
  .h        30.594 dosya   118 MB   c
  .ts       18.478 dosya  58,1 MB   typescript
  .gd        3.516 dosya    34 MB   gdscript
  .cpp       3.185 dosya  21,4 MB   cpp
  .c           605 dosya     8 MB   c
  .hpp         299 dosya   2,9 MB   cpp
  .lua          54 dosya   0,8 MB   lua
```

**Bu, hedefin (10×) çok üzerinde** — yalnız %1'i 13 MB / ~3,7M token.

Neden gerçek proje kodu değerli?
--------------------------------
    * **Çalışıyor** — gerçek projelerde kullanılıyor
    * **Çok dilli** — 9 dil
    * **Kalıp zengin** — tasarım kalıpları, mimari, test
    * **Uydurma riski yok** — insan yazmış

⚠ Ölçülen tuzaklar
------------------
    1. **Bağımlılıklar** (`site-packages`, `node_modules`) — çoğu
       üçüncü parti; **kendi kodumuz** değil. Ayırt edilmeli.
    2. **Tekrar** — aynı dosya kopyalanmış olabilir (TitoWar'da 3×)
       → MD5 ile tekil tut
    3. **Minify** — `.min.js` gibi tek satırlık dev dosyalar → **atla**
    4. **Üretilmiş** — `_pb2.py`, `.g.cs`, `dist/` → **atla**
    5. **Çok büyük** — 200 KB üstü tek dosya örnek olarak kullanılamaz

Kullanım
--------
    python tools/yerel_kod_madencisi.py --analiz
    python tools/yerel_kod_madencisi.py --topla --maks 40000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Taranacak kökler
#: ⚠ tur 7: ek kökler — Python kurulumu (17.412 dosya) ve E:\ kökü
KOKLER: Sequence[str] = (
    "E:\\PROJELER",
    "E:\\AppsTools\\PYTHON\\Python3.10",
    "E:\\",
    # ⚠ tur 7: hedefe 1,6M token kaldı — ek sistem kökleri
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "E:\\AppsTools",
)

#: Uzantı → dil
UZANTI: Dict[str, str] = {
    ".py": "python", ".gd": "gdscript", ".js": "javascript",
    ".ts": "typescript", ".rs": "rust", ".c": "c", ".cpp": "cpp",
    ".h": "c", ".hpp": "cpp", ".cs": "csharp", ".java": "java",
    ".lua": "lua", ".go": "go", ".rb": "ruby", ".kt": "kotlin",
    ".swift": "swift", ".php": "php",
}

#: Atlanacak dizin adları (bağımlılık/üretilmiş)
ATLA_DIZIN: Sequence[str] = (
    "site-packages", "node_modules", "dist-packages", ".git",
    "__pycache__", ".venv", "venv", "env", ".tox", ".mypy_cache",
    ".pytest_cache", "build", "dist", ".next", ".nuxt", "target",
    "obj", "bin", "Debug", "Release", ".idea", ".vs", "vendor",
    "pycache", "eggs", ".eggs", "wheels", "third_party",
)

#: Atlanacak dosya adı kalıpları (üretilmiş/minify)
ATLA_DOSYA = re.compile(
    r"(\.min\.|\.bundle\.|_pb2\.|\.g\.cs$|\.designer\.cs$|"
    r"\.generated\.|package-lock|yarn\.lock|\.d\.ts$)", re.IGNORECASE)

#: Tek dosya için azami boyut
MAKS_BAYT = 200_000

#: Tek örnek için azami satır
MAKS_SATIR = 120


@dataclass
class MadenRaporu:
    """Madencilik raporu."""
    taranan: int = 0
    atlanan_dizin: int = 0
    atlanan_dosya: int = 0
    atlanan_tekrar: int = 0
    atlanan_boyut: int = 0
    ornek: int = 0
    satir: int = 0
    karakter: int = 0
    dil_dagilim: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taranan": self.taranan, "atlanan_dizin": self.atlanan_dizin,
            "atlanan_dosya": self.atlanan_dosya,
            "atlanan_tekrar": self.atlanan_tekrar,
            "atlanan_boyut": self.atlanan_boyut,
            "ornek": self.ornek, "satir": self.satir,
            "karakter": self.karakter,
            "token_tahmini": int(self.karakter / 3.5),
            "dil_dagilim": dict(sorted(self.dil_dagilim.items(),
                                       key=lambda x: -x[1])),
        }


def _atla(yol: Path) -> bool:
    """Dosya atlanmalı mı?"""
    if ATLA_DOSYA.search(yol.name):
        return True
    if any(p in yol.parts for p in ATLA_DIZIN):
        return True
    return False


def _bloklar(icerik: str, dil: str) -> List[str]:
    """
    Kaynak dosyadan **fonksiyon/sınıf bloklarını** çıkarır.

    ⚠ Ölçülen: 200 KB'lık tek dosya örnek olarak kullanılamaz. Bloğa
    bölmek gerekir. Dil başına farklı başlangıç kalıbı:
        python     : `def `/`class ` (girintisiz)
        gdscript   : `func `/`class ` (girintisiz)
        javascript : `function ` / `const x = (` / `class `
        typescript : aynı + `interface`
        rust       : `fn `/`impl `/`struct `/`enum `
        c/cpp      : `void `/`int ` gibi satır başı tipler
        lua        : `function `
    """
    bas: Dict[str, str] = {
        "python": r"^(?:def|class)\s+\w+",
        "gdscript": r"^(?:func|class|static func)\s+\w+",
        "javascript": (r"^(?:export\s+)?(?:async\s+)?function\s+\w+|"
                       r"^(?:export\s+)?class\s+\w+"),
        "typescript": (r"^(?:export\s+)?(?:async\s+)?function\s+\w+|"
                       r"^(?:export\s+)?(?:abstract\s+)?class\s+\w+|"
                       r"^(?:export\s+)?interface\s+\w+"),
        "rust": (r"^(?:pub\s+)?(?:async\s+)?fn\s+\w+|"
                 r"^(?:pub\s+)?(?:struct|enum|impl|trait)\s+\w+"),
        "c": r"^[A-Za-z_][\w\s*]*\**\w+\s*\([^;]*\)\s*\{?$",
        "cpp": (r"^[A-Za-z_][\w\s*:<>,]*\**\w+\s*\([^;]*\)\s*\{?$|"
                r"^(?:class|struct|namespace)\s+\w+"),
        "csharp": (r"^(?:public|private|internal|protected|\s)*"
                   r"(?:static\s+)?(?:class|interface|void|int|string)\s"),
        "java": (r"^(?:public|private|protected|\s)*"
                 r"(?:static\s+)?(?:class|interface|void|int|String)\s"),
        "lua": r"^(?:local\s+)?function\s+[\w.:]+",
        "go": r"^func\s+",
        "ruby": r"^(?:def|class|module)\s+",
        "kotlin": r"^(?:fun|class|object)\s+",
        "swift": r"^(?:func|class|struct|enum)\s+",
        "php": r"^(?:function|class)\s+",
    }
    desen = bas.get(dil)
    if not desen:
        return []
    satirlar = icerik.splitlines()
    bloklar: List[str] = []
    baslangic: Optional[int] = None
    for i, s in enumerate(satirlar):
        if re.match(desen, s):
            if baslangic is not None:
                bloklar.append("\n".join(satirlar[baslangic:i]).rstrip())
            baslangic = i
        if baslangic is not None and i - baslangic >= MAKS_SATIR:
            bloklar.append("\n".join(satirlar[baslangic:i + 1]).rstrip())
            baslangic = None
    if baslangic is not None:
        bloklar.append("\n".join(satirlar[baslangic:]).rstrip())
    return [b for b in bloklar if b.strip()]


def _kod_gibi(metin: str) -> bool:
    """Metin gerçekten kod mu (yorum/boş değil)?"""
    # ⚠ ÖLÇÜLEN (tur 7): eşik 80 karakter, 165.286 dosyadan yalnız
    # %2 verim verdi (31.595 blok). Küçük ama gerçek fonksiyonlar
    # (`def f(): return x`) atlanıyordu. Eşik 45'e indirildi.
    if len(metin) < 45 or len(metin) > 4000:
        return False
    satirlar = [s for s in metin.splitlines() if s.strip()]
    if len(satirlar) < 2:
        return False
    # Yorum satırı oranı %70 üstündeyse kod değil
    yorum = sum(1 for s in satirlar
                if s.strip().startswith(("#", "//", "*", "/*")))
    if yorum / len(satirlar) > 0.7:
        return False
    # Kod göstergesi
    return any(k in metin for k in ("(", "=", "{", ":", "return",
                                    "import", "def ", "func ", "fn "))


#: Dil başına kota — bir dil diğerlerini ezmesin
#:
#: ⚠ ÖLÇÜLEN (tur 6): kotasız taramada **3.000 örneğin tamamı
#: python** oldu (ilk 363 dosya). Çok dilli denge için kota şart.
#: ⚠ tur 7: kotalar **2×** artırıldı — hedefe (10×) 4,1M token kaldı
DIL_KOTA: Dict[str, int] = {
    "python": 40000, "javascript": 30000, "gdscript": 30000,
    "cpp": 25000, "c": 25000, "typescript": 20000, "rust": 12000,
    "csharp": 8000, "java": 6000, "lua": 6000, "go": 4000,
    "ruby": 3000, "kotlin": 2000, "swift": 1500, "php": 1500,
}


def tara(*, maks: int = 40000, kokler: Sequence[str] = KOKLER,
         hedef: Path = Path(".neurocore/yerel_kod.jsonl"),
         analiz_modu: bool = False) -> MadenRaporu:
    """Yerel kaynak kodları tarar."""
    r = MadenRaporu()
    gorulen_dosya: Set[str] = set()
    gorulen_blok: Set[str] = set()
    hedef.parent.mkdir(parents=True, exist_ok=True)

    f = None if analiz_modu else hedef.open("w", encoding="utf-8")
    t0 = time.perf_counter()
    try:
        for kok in kokler:
            p = Path(kok)
            if not p.exists():
                continue
            for yol in p.rglob("*"):
                if r.ornek >= maks:
                    break
                if not yol.is_file():
                    continue
                dil = UZANTI.get(yol.suffix.lower())
                if not dil:
                    continue
                r.taranan += 1
                if _atla(yol):
                    r.atlanan_dosya += 1
                    continue
                try:
                    boyut = yol.stat().st_size
                except Exception:
                    continue
                if boyut < 200 or boyut > MAKS_BAYT:
                    r.atlanan_boyut += 1
                    continue
                try:
                    ham = yol.read_bytes()
                except Exception:
                    continue
                ozet = hashlib.md5(ham).hexdigest()
                if ozet in gorulen_dosya:
                    r.atlanan_tekrar += 1
                    continue
                gorulen_dosya.add(ozet)
                try:
                    icerik = ham.decode("utf-8", errors="replace")
                except Exception:
                    continue
                for b in _bloklar(icerik, dil):
                    if r.ornek >= maks:
                        break
                    if not _kod_gibi(b):
                        continue
                    # ── Dil kotası ──
                    if r.dil_dagilim.get(dil, 0) >= DIL_KOTA.get(dil, 500):
                        continue
                    anahtar = b.strip()[:120]
                    if anahtar in gorulen_blok:
                        continue
                    gorulen_blok.add(anahtar)
                    r.ornek += 1
                    r.satir += len(b.splitlines())
                    r.karakter += len(b)
                    r.dil_dagilim[dil] = r.dil_dagilim.get(dil, 0) + 1
                    if f is not None:
                        f.write(json.dumps({
                            "metin": (f"# {dil} · {yol.name}\n\n"
                                      f"```{dil}\n{b}\n```\n"),
                            "kategori": "yerel_kod",
                            "dil": dil,
                            "kaynak": str(yol)[:160],
                        }, ensure_ascii=False) + "\n")
                if r.taranan % 20000 == 0:
                    print(f"    taranan {r.taranan:,} · örnek {r.ornek:,} · "
                          f"~{r.karakter // 3500:,}k token · "
                          f"{time.perf_counter() - t0:.0f}s", flush=True)
    finally:
        if f is not None:
            f.close()
    return r


def main() -> int:
    """Madencilik işlemlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="Yerel kaynak kod madencisi")
    ap.add_argument("--analiz", action="store_true",
                    help="Yalnız say (yazma)")
    ap.add_argument("--topla", action="store_true")
    ap.add_argument("--maks", type=int, default=40000)
    ap.add_argument("--kok", default="", help="Virgülle ayrılmış kökler")
    args = ap.parse_args()

    kokler = ([k.strip() for k in args.kok.split(",") if k.strip()]
              if args.kok else list(KOKLER))

    print("=" * 78)
    print("  YEREL KAYNAK KOD MADENCİSİ — çok dilli gerçek proje kodu")
    print("=" * 78)
    print(f"  kökler: {kokler}")

    r = tara(maks=args.maks, kokler=kokler,
             analiz_modu=not args.topla)
    print()
    print(f"  taranan dosya : {r.taranan:,}")
    print(f"  atlanan       : dizin/kalıp={r.atlanan_dosya:,} "
          f"tekrar={r.atlanan_tekrar:,} boyut={r.atlanan_boyut:,}")
    print(f"  ÖRNEK         : {r.ornek:,}")
    print(f"  satır         : {r.satir:,}")
    print(f"  karakter      : {r.karakter:,}")
    print(f"  token         : ~{r.karakter // 3500:,}k")
    print("\n  DİL DAĞILIMI:")
    for d, n in list(r.dil_dagilim.items())[:12]:
        print(f"    {d:14} {n:7,}")
    if args.topla:
        print(f"\n  dosya: .neurocore/yerel_kod.jsonl")

    Path("_yerel_kod.json").write_text(
        json.dumps(r.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
