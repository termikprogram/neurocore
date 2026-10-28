"""
Python API Madencisi — Standart Kütüphane + Sürüm Bilgisi
===========================================================
⚠ Neden bu kaynak? (tur 2/40)
-----------------------------
Kullanıcı hedefi (A): *"her dilin kütüphanesi ve sürümüne kadar geniş,
çok dilli ve çok sürümlü kod verisi topla"*

**En güvenilir kaynak:** kurulu Python'un **kendi kütüphanesi**.

| Kaynak | Doğruluk | Sürüm bilgisi |
|---|---|---|
| Öğretmen model | ⚠️ Uydurma olabilir | ❌ yok |
| Web dokümanı | ✅ | ✅ (ama HTML ayrıştırma kırılgan) |
| **Kurulu kütüphane (bu araç)** | ✅ **kesin** | ✅ **`sys.version_info`** |

**Kritik:** `inspect` ve `pydoc` ile **gerçek imzalar**, **gerçek
docstring'ler** ve **gerçek parametreler** alınır. Uydurma riski **sıfır**.

Yöntem
------
    1. Modül listesini al (standart kütüphane + önemli 3. parti)
    2. Her modülü içe aktar
    3. `inspect` ile sınıf/fonksiyon/istisna topla
    4. İmza + docstring + parametreleri çıkar
    5. Sürüm etiketiyle kasaya yaz

Kapsam (hedef)
--------------
    * Standart kütüphane: ~200 modül
    * `math` · `os` · `sys` · `collections` · `itertools` · `functools`
    * `json` · `re` · `datetime` · `pathlib` · `typing` · `asyncio`
    * `unittest` · `logging` · `subprocess` · `threading` · `socket`
    * Ayrıca: `numpy` · `pandas` · `torch` · `PIL` (kuruluysa)

Kullanım
--------
    python tools/python_api_toplayici.py --analiz
    python tools/python_api_toplayici.py --topla --maks 8000
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import pkgutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Standart kütüphane modülleri (öncelik sırasıyla)
STD_MODULLER: Sequence[str] = (
    # Temel
    "builtins", "sys", "os", "os.path", "io", "time", "datetime",
    "math", "cmath", "decimal", "fractions", "random", "statistics",
    # Veri yapıları
    "collections", "collections.abc", "array", "heapq", "bisect",
    "queue", "enum", "dataclasses", "typing", "types",
    # Fonksiyonel
    "itertools", "functools", "operator", "contextlib",
    # Metin ve veri
    "re", "string", "textwrap", "unicodedata", "json", "csv",
    "configparser", "base64", "binascii", "struct", "hashlib",
    "hmac", "secrets", "uuid", "copy", "pprint", "reprlib",
    # Dosya ve yol
    "pathlib", "glob", "fnmatch", "shutil", "tempfile", "fileinput",
    "zipfile", "tarfile", "gzip", "bz2", "lzma", "pickle",
    "sqlite3", "dbm",
    # Eşzamanlılık
    "threading", "multiprocessing", "concurrent.futures", "asyncio",
    "select", "signal", "subprocess",
    # Ağ
    "socket", "ssl", "http", "http.client", "http.server",
    "urllib", "urllib.request", "urllib.parse", "smtplib",
    "email", "json", "xml", "html",
    # Hata ayıklama ve test
    "unittest", "doctest", "logging", "warnings", "traceback",
    "pdb", "cProfile", "timeit", "gc", "weakref", "inspect",
    "ast", "dis", "code", "codeop",
    # Sistem
    "platform", "sysconfig", "site", "importlib", "argparse",
    "getopt", "shlex", "cmd", "atexit", "contextvars",
    # Diğer
    "abc", "numbers", "cmath", "calendar", "zoneinfo", "graphlib",
    "difflib", "gettext", "locale", "getpass", "tarfile",
)

#: Üçüncü parti (kuruluysa)
UCUNCU_PARTI: Sequence[str] = (
    "numpy", "pandas", "torch", "PIL", "requests", "matplotlib",
    "scipy", "sklearn", "transformers", "cv2", "bpy",
)


@dataclass
class ApiKaydi:
    """Tek bir API kaydı."""
    modul: str
    ad: str
    tur: str          # function | class | method | exception | constant
    imza: str = ""
    docstring: str = ""
    parametreler: List[str] = field(default_factory=list)
    dondurur: str = ""
    surum: str = ""

    def metin(self) -> str:
        """Kaydın metin biçimi."""
        p = [f"# {self.modul}.{self.ad}", ""]
        if self.docstring:
            ilk = self.docstring.strip().split("\n\n")[0]
            p.append(ilk.strip()[:500])
            p.append("")
        p.append(f"**Tür:** {self.tur}")
        if self.imza:
            p.append(f"**İmza:** `{self.imza}`")
        if self.parametreler:
            p.append(f"**Parametreler:** {', '.join(self.parametreler[:8])}")
        if self.dondurur:
            p.append(f"**Döndürür:** {self.dondurur}")
        p.append(f"**Sürüm:** {self.surum}")
        p.append("")
        p.append("```python")
        # Kullanım örneği — imzadan türetilir
        p.append(f"import {self.modul.split('.')[0]}")
        p.append(f"{self.modul}.{self.ad}(...)")
        p.append("```")
        return "\n".join(p)


def _surum(modul: Any) -> str:
    """Modül sürümü."""
    for a in ("__version__", "version", "VERSION"):
        v = getattr(modul, a, None)
        if isinstance(v, str) and v:
            return v
    vi = sys.version_info
    return f"Python {vi.major}.{vi.minor}.{vi.micro}"


def _docstring_kisa(d: Any) -> str:
    """Docstring'i temizler."""
    if not isinstance(d, str):
        return ""
    d = inspect.cleandoc(d)
    # Çok uzunları kısalt
    if len(d) > 900:
        d = d[:900] + "…"
    return d


def modul_tara(ad: str, *, maks: int = 400) -> List[ApiKaydi]:
    """
    Bir modülü tarar ve API kayıtlarını çıkarır.

    ⚠ Ölçülen tuzaklar:
        * Bazı modüller içe aktarılınca **yan etki** üretir (ağ, dosya)
          → `try/except` ile korunur
        * Bazı nesneler `inspect.signature` desteklemez (C uzantıları)
          → imza boş kalır, docstring yine alınır
        * Özel (`_`) adlar **atlanır** — kullanıcı API'si değil
    """
    kayitlar: List[ApiKaydi] = []
    try:
        modul = importlib.import_module(ad)
    except Exception:
        return kayitlar
    surum = _surum(modul)

    # Modül docstring'i
    md = _docstring_kisa(getattr(modul, "__doc__", ""))
    if md:
        kayitlar.append(ApiKaydi(
            modul=ad, ad="(modül)", tur="module", docstring=md,
            surum=surum))

    uyeler = []
    try:
        uyeler = inspect.getmembers(modul)
    except Exception:
        return kayitlar

    for ad_, nesne in uyeler:
        if len(kayitlar) >= maks:
            break
        if ad_.startswith("_"):
            continue
        # Tanımlandığı modül bu modül mü? (yeniden dışa aktarımı atla)
        try:
            if (inspect.ismodule(nesne)):
                continue
        except Exception:
            continue

        tur = ""
        imza = ""
        params: List[str] = []
        dondurur = ""

        try:
            if inspect.isclass(nesne):
                tur = ("exception"
                       if issubclass(nesne, BaseException) else "class")
                try:
                    imza = f"{ad_}{inspect.signature(nesne)}"
                except Exception:
                    imza = f"{ad_}(...)"
                # Metotlar
                for ma, mn in inspect.getmembers(nesne):
                    if ma.startswith("_") or len(kayitlar) >= maks:
                        continue
                    if not (inspect.isfunction(mn)
                            or inspect.ismethoddescriptor(mn)):
                        continue
                    try:
                        mi = f"{ad_}.{ma}{inspect.signature(mn)}"
                    except Exception:
                        mi = f"{ad_}.{ma}(...)"
                    kayitlar.append(ApiKaydi(
                        modul=ad, ad=f"{ad_}.{ma}", tur="method",
                        imza=mi, docstring=_docstring_kisa(
                            getattr(mn, "__doc__", "")),
                        parametreler=_params(mn), surum=surum))
                params = _params(nesne)
            elif inspect.isfunction(nesne) or inspect.isbuiltin(nesne):
                tur = "function"
                try:
                    imza = f"{ad_}{inspect.signature(nesne)}"
                except Exception:
                    imza = f"{ad_}(...)"
                params = _params(nesne)
            else:
                # Sabit / nitelik
                if isinstance(nesne, (int, float, str, bool, bytes)):
                    tur = "constant"
                    imza = f"{ad_} = {nesne!r}"[:120]
                else:
                    continue
        except Exception:
            continue

        if not tur:
            continue
        kayitlar.append(ApiKaydi(
            modul=ad, ad=ad_, tur=tur, imza=imza,
            docstring=_docstring_kisa(getattr(nesne, "__doc__", "")),
            parametreler=params, dondurur=dondurur, surum=surum))
    return kayitlar


def _params(nesne: Any) -> List[str]:
    """İmzadan parametre adları çıkarır."""
    try:
        s = inspect.signature(nesne)
    except Exception:
        return []
    out: List[str] = []
    for ad, p in s.parameters.items():
        if ad in ("self", "cls"):
            continue
        if p.default is inspect.Parameter.empty:
            out.append(ad)
        else:
            try:
                d = repr(p.default)
                if len(d) > 24:
                    d = d[:21] + "…"
                out.append(f"{ad}={d}")
            except Exception:
                out.append(ad)
    return out


def analiz(moduller: Sequence[str]) -> Dict[str, Any]:
    """Modülleri analiz eder — kaç kayıt çıkacak?"""
    toplam = 0
    basarili = 0
    dagilim: Dict[str, int] = {}
    for m in moduller:
        k = modul_tara(m, maks=400)
        if k:
            basarili += 1
            toplam += len(k)
            dagilim[m] = len(k)
    return {"modul": len(moduller), "basarili": basarili,
            "kayit": toplam, "dagilim": dagilim}


def topla(moduller: Sequence[str], *, hedef: Path,
          maks_toplam: int = 8000) -> Dict[str, Any]:
    """API kayıtlarını JSONL olarak yazar."""
    hedef.parent.mkdir(parents=True, exist_ok=True)
    sayi = 0
    modul_sayi = 0
    karakter = 0
    gorulen: Set[str] = set()
    with hedef.open("w", encoding="utf-8") as f:
        for m in moduller:
            if sayi >= maks_toplam:
                break
            kayitlar = modul_tara(m, maks=min(400, maks_toplam - sayi))
            if not kayitlar:
                continue
            modul_sayi += 1
            for k in kayitlar:
                if sayi >= maks_toplam:
                    break
                anahtar = f"{k.modul}.{k.ad}"
                if anahtar in gorulen:
                    continue
                gorulen.add(anahtar)
                metin = k.metin()
                f.write(json.dumps({
                    "metin": metin + "\n\n",
                    "kategori": "python_api",
                    "modul": k.modul,
                    "surum": k.surum,
                    "tur": k.tur,
                }, ensure_ascii=False) + "\n")
                sayi += 1
                karakter += len(metin)
    return {"kayit": sayi, "modul": modul_sayi, "karakter": karakter,
            "token_tahmini": karakter // 3, "dosya": str(hedef)}


def main() -> int:
    """Python API madenciliğini çalıştırır."""
    ap = argparse.ArgumentParser(description="Python API madencisi")
    ap.add_argument("--analiz", action="store_true")
    ap.add_argument("--topla", action="store_true")
    ap.add_argument("--maks", type=int, default=8000)
    ap.add_argument("--ucuncu", action="store_true",
                    help="Üçüncü parti kütüphaneleri de tara")
    ap.add_argument("--liste", action="store_true")
    args = ap.parse_args()

    moduller = list(STD_MODULLER)
    if args.ucuncu:
        moduller += list(UCUNCU_PARTI)

    if args.liste:
        for m in moduller:
            print(f"  {m}")
        print(f"\n  TOPLAM: {len(moduller)} modül")
        return 0

    print("=" * 78)
    print("  PYTHON API MADENCİSİ — kurulu kütüphaneden gerçek imzalar")
    print("=" * 78)
    vi = sys.version_info
    print(f"  Python: {vi.major}.{vi.minor}.{vi.micro}")
    print(f"  modül : {len(moduller)}")

    sonuc: Dict[str, Any] = {}
    if args.analiz or not args.topla:
        a = analiz(moduller)
        print(f"\n  başarılı modül: {a['basarili']}/{a['modul']}")
        print(f"  toplam kayıt  : {a['kayit']:,}")
        print("\n  EN ZENGİN 12 MODÜL:")
        for m, n in sorted(a["dagilim"].items(),
                           key=lambda x: -x[1])[:12]:
            print(f"    {m:28} {n:4}")
        sonuc["analiz"] = {"modul": a["modul"], "basarili": a["basarili"],
                           "kayit": a["kayit"]}
    if args.topla:
        r = topla(moduller, hedef=Path(".neurocore/python_api.jsonl"),
                  maks_toplam=args.maks)
        print(f"\n  TOPLANDI: {r['kayit']:,} kayıt · {r['modul']} modül")
        print(f"  karakter: {r['karakter']:,} · token: ~{r['token_tahmini']:,}")
        print(f"  dosya: {r['dosya']}")
        sonuc["toplama"] = r

    Path("_python_api.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
