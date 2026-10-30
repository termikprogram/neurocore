"""
Büyük dosya okuma — 10 milyon satırlık dosyalar için katmanlı erişim
======================================================================
Sorun: `read_file` tüm dosyayı `text.splitlines()` ile belleğe alıyordu;
10 M satırlık bir dosya ~250 MB metin + ~1-2 GB Python nesnesi demek.
`search_text` ise **4 MB üstü dosyaları tamamen atlıyordu**.

Çözüm — üç katman
-----------------
```
1) SEYREK SATIR İNDEKSİ  : her N satırın **bayt ofseti** yan dosyada tutulur
   (<dosya>.ncidx.json). 10 M satır / 2000 = 5000 kayıt (~120 KB).
   → rastgele satıra seek ile atlama, sabit bellek.
2) PENCERE OKUMA         : yalnız istenen satır aralığı okunur (akış).
3) YAPISAL HARİTA        : dosya bir kez akıtılıp semboller/başlıklar çıkarılır
   (def/class/func/# başlık) → ajan önce haritayı görür, sonra pencere ister.
4) AKIŞLI ARAMA          : grep deseni satır satır, sonuç sınırlı, sabit bellek.
```

Kurallar
--------
```
· Asla `read()` + `splitlines()` ile tüm dosya yüklenmez (eşik üstünde)
· İndeks mtime+size ile doğrulanır; değişirse yeniden kurulur
· Tüm dönüşler `note` alanında "hangi aralık / toplam kaç satır" bildirir
```
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.bigfile")

#: Bu boyutun üstündeki dosyalar "büyük dosya" sayılır (bayt)
BUYUK_ESIK = 2 * 1024 * 1024

#: İndeks aralığı: her N satırda bir bayt ofseti
INDEX_ARALIK = 2000

#: Tek pencerede dönülecek en fazla satır
PENCERE_MAKS = 2000

#: Yapısal haritada en fazla kaç giriş
HARITA_MAKS = 400

#: Akışlı aramada en fazla sonuç
ARAMA_MAKS = 100

#: Uzantıya göre yapı kalıpları (satır başı eşleşmesi)
YAPI_KALIPLARI: Dict[str, re.Pattern] = {
    "python": re.compile(
        r"^(?:class |def |async def |@[A-Za-z_])"),
    "gdscript": re.compile(
        r"^(?:class_name |class |func |static func |@export|@onready|signal )"),
    "javascript": re.compile(
        r"^(?:export |class |function |const |let |var |module\.exports)"),
    "typescript": re.compile(
        r"^(?:export |interface |type |class |function |const )"),
    "rust": re.compile(r"^(?:pub |fn |struct |enum |impl |trait |mod )"),
    "go": re.compile(r"^(?:func |type |package )"),
    "csharp": re.compile(r"^(?:public |private |protected |internal |namespace |class )"),
    "markdown": re.compile(r"^#{1,6} "),
    "text": re.compile(r"^\S"),
}

UZANTI_DIL = {
    ".py": "python", ".gd": "gdscript", ".js": "javascript",
    ".mjs": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust", ".go": "go", ".cs": "csharp", ".md": "markdown",
}


# ─── yardımcılar ────────────────────────────────────────────────────────────

def dil(yol: Path) -> str:
    """Dosya uzantısından yapı dili."""
    return UZANTI_DIL.get(Path(yol).suffix.lower(), "text")


def indeks_yolu(yol: Path) -> Path:
    """Yan indeks dosyasının yolu."""
    yol = Path(yol)
    return yol.with_name(yol.name + ".ncidx.json")


def _satirlari_ak(aciK) -> Iterator[Tuple[int, bytes]]:
    """İkili akıştan satır satır (offset, satır) üretir."""
    ofset = 0
    for satir in aciK:
        yield ofset, satir
        ofset += len(satir)


def buyuk_mu(yol: Path, esik: int = BUYUK_ESIK) -> bool:
    """Dosya büyük-dosya modunu gerektiriyor mu?"""
    try:
        return Path(yol).stat().st_size > esik
    except OSError:
        return False


# ─── 1) seyrek satır indeksi ────────────────────────────────────────────────

def indeks_kur(yol: Path, aralik: int = INDEX_ARALIK) -> Dict[str, Any]:
    """
    Seyrek satır indeksi kurar (her `aralik` satırda bir bayt ofseti).

    Returns:
        {"aralik", "total_lines", "offsets": [bayt ofsetleri], "size", "mtime"}
    """
    yol = Path(yol)
    stat = yol.stat()
    offs: List[int] = [0]
    satir = 0
    with yol.open("rb") as f:
        for ofset, ham in _satirlari_ak(f):
            satir += 1
            if satir % aralik == 0:
                offs.append(ofset + len(ham))    # bir sonraki satırın başı
    veri = {"aralik": aralik, "total_lines": satir, "offsets": offs,
            "size": stat.st_size, "mtime": stat.st_mtime}
    try:
        indeks_yolu(yol).write_text(
            json.dumps(veri, separators=(",", ":")), encoding="utf-8")
    except OSError as exc:
        logger.debug(f"indeks yazılamadı: {exc}")
    return veri


def indeks_al(yol: Path, aralik: int = INDEX_ARALIK) -> Dict[str, Any]:
    """İndeksi okur; yoksa/bayatladıysa yeniden kurar."""
    yol = Path(yol)
    stat = yol.stat()
    ip = indeks_yolu(yol)
    if ip.exists():
        try:
            veri = json.loads(ip.read_text(encoding="utf-8"))
            if (veri.get("size") == stat.st_size
                    and abs(veri.get("mtime", 0) - stat.st_mtime) < 0.001
                    and veri.get("aralik") == aralik):
                return veri
        except (OSError, json.JSONDecodeError):
            pass
    return indeks_kur(yol, aralik)


# ─── 2) pencere okuma ───────────────────────────────────────────────────────

def pencere(yol: Path, bas: int = 1, adet: int = 200,
            numarali: bool = True) -> Dict[str, Any]:
    """
    Dosyanın **yalnız** istenen satır aralığını okur (sabit bellek).

    Args:
        yol: Dosya
        bas: Başlangıç satırı (1 tabanlı)
        adet: Satır sayısı (en fazla PENCERE_MAKS)
        numarali: Satır numarası ekle

    Returns:
        read_file ile aynı sözleşme: {"path", "language", "total_lines",
        "start_line", "end_line", "content", "truncated", "size", "note"}
    """
    yol = Path(yol)
    adet = max(1, min(int(adet), PENCERE_MAKS))
    bas = max(1, int(bas))
    idx = indeks_al(yol)
    total = int(idx["total_lines"])
    aralik = int(idx["aralik"])
    offs: List[int] = idx["offsets"]

    # en yakın indeks noktasından başla
    blok = (bas - 1) // aralik
    blok = max(0, min(blok, len(offs) - 1))
    baslangic_ofset = offs[blok]
    satir_no = blok * aralik          # bu ofsetten önce okunan satır sayısı

    satirlar: List[str] = []
    bayt = 0
    with yol.open("rb") as f:
        f.seek(baslangic_ofset)
        while satir_no < bas - 1:
            ham = f.readline()
            if not ham:
                break
            satir_no += 1
        while len(satirlar) < adet:
            ham = f.readline()
            if not ham:
                break
            satir_no += 1
            satirlar.append(ham.decode("utf-8", errors="replace").rstrip("\r\n"))
            bayt += len(ham)

    son = bas + len(satirlar) - 1 if satirlar else 0
    if numarali:
        genis = len(str(son or 1))
        icerik = "\n".join(
            f"{bas + i:>{genis}}| {s}" for i, s in enumerate(satirlar))
    else:
        icerik = "\n".join(satirlar)

    return {
        "path": str(yol),
        "language": dil(yol),
        "total_lines": total,
        "start_line": bas if satirlar else 0,
        "end_line": son,
        "content": icerik,
        "truncated": son < total,
        "size": idx["size"],
        "bytes_read": bayt,
        "indexed": True,
        "note": (f"[büyük dosya · indeks {aralik} satır] "
                 f"satır {bas}-{son} / {total}"
                 + (f" — devamı için start_line={son + 1}" if son < total else "")),
    }


# ─── 3) yapısal harita ──────────────────────────────────────────────────────

def harita(yol: Path, sinir: int = HARITA_MAKS) -> Dict[str, Any]:
    """
    Dosyayı **bir kez akıtarak** yapısal harita çıkarır (semboller/başlıklar).

    Returns:
        {"path", "language", "total_lines", "entries": [{line, text}],
         "truncated", "note"}
    """
    yol = Path(yol)
    kalip = YAPI_KALIPLARI.get(dil(yol), YAPI_KALIPLARI["text"])
    girisler: List[Dict[str, Any]] = []
    satir = 0
    with yol.open("rb") as f:
        for _, ham in _satirlari_ak(f):
            satir += 1
            s = ham.decode("utf-8", errors="replace").rstrip("\r\n")
            if not s.strip() or len(s) > 400:
                continue
            if kalip.match(s):
                girisler.append({"line": satir, "text": s.strip()[:160]})
                if len(girisler) >= sinir:
                    break
    return {
        "path": str(yol),
        "language": dil(yol),
        "total_lines": satir,
        "entries": girisler,
        "truncated": len(girisler) >= sinir,
        "note": (f"{len(girisler)} yapı girişi (ilk {satir} satır tarandı)"
                 + (" — sınır doldu" if len(girisler) >= sinir else "")),
    }


# ─── 4) akışlı arama ────────────────────────────────────────────────────────

def ara(yol: Path, desen: str, *, regex: bool = False,
        sinir: int = ARAMA_MAKS, baglam: int = 0) -> Dict[str, Any]:
    """
    Büyük dosyada **akışlı** arama (dosya belleğe alınmaz).

    Returns:
        {"path", "pattern", "hits": [{line, text}], "count", "truncated"}
    """
    yol = Path(yol)
    try:
        kalip = re.compile(desen if regex else re.escape(desen),
                           re.IGNORECASE)
    except re.error as exc:
        return {"path": str(yol), "pattern": desen, "hits": [], "count": 0,
                "error": f"geçersiz regex: {exc}"}
    hits: List[Dict[str, Any]] = []
    satir = 0
    tampon: List[Tuple[int, str]] = []
    with yol.open("rb") as f:
        for _, ham in _satirlari_ak(f):
            satir += 1
            s = ham.decode("utf-8", errors="replace").rstrip("\r\n")
            if baglam:
                tampon.append((satir, s))
                if len(tampon) > baglam * 2 + 1:
                    tampon.pop(0)
            if kalip.search(s):
                hits.append({"line": satir, "text": s.strip()[:200]})
                if len(hits) >= sinir:
                    break
    return {
        "path": str(yol),
        "pattern": desen,
        "hits": hits,
        "count": len(hits),
        "total_lines": satir,
        "truncated": len(hits) >= sinir,
        "note": f"{len(hits)} eşleşme (ilk {satir} satır tarandı)",
    }


# ─── okuma stratejisi (ajan için kısa rehber) ───────────────────────────────

def strateji(yol: Path) -> Dict[str, Any]:
    """Dosya için önerilen okuma planını döndürür (ajana ipucu)."""
    y = Path(yol)
    try:
        boyut = y.stat().st_size
    except OSError:
        return {"ok": False, "error": "dosya yok"}
    idx = indeks_al(y) if boyut > BUYUK_ESIK else {"total_lines": 0,
                                                   "aralik": INDEX_ARALIK}
    return {
        "ok": True,
        "path": str(y),
        "size": boyut,
        "buyuk": boyut > BUYUK_ESIK,
        "total_lines": idx["total_lines"],
        "index_araligi": idx["aralik"],
        "oneri": (
            "Önce `code.outline` ile yapı haritasını al, sonra ilgilenilen "
            "satır aralığını `code.read_file(start_line=..., max_lines=...)` "
            "ile oku. Arama için `code.search_text` (büyük dosyada akışlı)."
            if boyut > BUYUK_ESIK else
            "Küçük dosya: doğrudan read_file yeterli."),
    }
