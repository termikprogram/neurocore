"""
Proje kod indeksi — milyonlarca satır, binlerce dosya
====================================================
Soru: "Proje çok büyüdü, çok satırlı dosyalar oluştu, **birbiriyle ilişkili
kodları** da bulması gerekiyor. 10 milyon satırda takılı mı kalacağız?"

Cevap: dosya bazlı pencere okuma **tek dosya** için yeterli; proje ölçeğinde
gereken şey **kalıcı, artımlı bir kod indeksi**:

```text
┌──────────────────────────────────────────────────────────────┐
│ .neurocore/index/kod.db   (SQLite)                           │
│  dosya(path, mtime, size, satir, dil)                        │
│  sembol(dosya, satir, ad, tur)      ← def/class/func/const   │
│  iliski(kaynak, hedef, tur)         ← import/include/require │
│  sembol_fts / iliski_fts (FTS5)     ← hızlı arama            │
└──────────────────────────────────────────────────────────────┘
```

Kurallar
--------
```
· **Artımlı**: yalnız mtime/size değişen dosyalar yeniden taranır
· **Akışlı**: dosya belleğe alınmaz (satır satır regex)
· **Sınırlı**: dosya başına en fazla N sembol, indeks boyutu kontrollü
· **Sorgular ms düzeyi**: FTS5 + indeksli sütunlar
· İkili/üretilmiş dizinler atlanır (node_modules, .git, .godot, build…)
```

Sorgular
--------
```python
dizinle(kok)                 # kur/artımlı güncelle → özet
sembol_ara("can_eat")        # tanım satırları
referanslar("Config")        # geçen yerler (metin + ilişki)
iliskili_dosyalar(dosya)     # import eden / edilen dosyalar
proje_haritasi()             # dil dağılımı, en büyük dosyalar, giriş noktaları
```
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.codeindex")

#: İndeks sürümü (şema değişirse artır)
SEMVER = 3

#: Taranacak uzantılar → dil
DILLER: Dict[str, str] = {
    ".py": "python", ".gd": "gdscript", ".js": "javascript",
    ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".rs": "rust",
    ".go": "go", ".cs": "csharp", ".java": "java", ".kt": "kotlin",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".lua": "lua", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".sh": "shell", ".ps1": "powershell", ".sql": "sql",
    ".md": "markdown", ".json": "json", ".yml": "yaml", ".yaml": "yaml",
    ".toml": "toml", ".html": "html", ".css": "css",
}

#: Atlanacak dizinler
ATLA = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".godot", "build", "dist", "target", "bin", "obj", ".idea", ".vscode",
    "models", "data", "knowledge_base", ".neurocore", "raporlar",
}

#: Dosya başına en fazla sembol / ilişki
SEMBOL_MAKS = 400
ILISKI_MAKS = 200

#: İndekslenecek en büyük dosya (bayt) — üstündekiler yalnız sayılır
DOSYA_MAKS = 8 * 1024 * 1024

#: Sembol kalıpları (dil → [(tür, regex)])
SEMBOL_KALIPLARI: Dict[str, List[Tuple[str, re.Pattern]]] = {
    "python": [
        ("class", re.compile(r"^\s*class\s+([A-Za-z_]\w*)")),
        ("func", re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)")),
        ("const", re.compile(r"^([A-Z_][A-Z0-9_]{2,})\s*=")),
    ],
    "gdscript": [
        ("class", re.compile(r"^\s*class_name\s+([A-Za-z_]\w*)")),
        ("func", re.compile(r"^\s*(?:static\s+)?func\s+([A-Za-z_]\w*)")),
        ("signal", re.compile(r"^\s*signal\s+([A-Za-z_]\w*)")),
        ("const", re.compile(r"^\s*const\s+([A-Za-z_]\w*)")),
    ],
    "javascript": [
        ("class", re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)")),
        ("func", re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)")),
        ("func", re.compile(
            r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(")),
    ],
    "rust": [
        ("func", re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)")),
        ("struct", re.compile(r"^\s*(?:pub\s+)?struct\s+(\w+)")),
        ("enum", re.compile(r"^\s*(?:pub\s+)?enum\s+(\w+)")),
        ("trait", re.compile(r"^\s*(?:pub\s+)?trait\s+(\w+)")),
    ],
    "go": [
        ("func", re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)")),
        ("type", re.compile(r"^type\s+(\w+)")),
    ],
    "csharp": [
        ("class", re.compile(r"^\s*(?:public|internal|private)?\s*(?:static\s+)?class\s+(\w+)")),
        ("func", re.compile(
            r"^\s*(?:public|private|protected|internal)\s+[\w<>\[\],\s]+\s+(\w+)\s*\(")),
    ],
    "markdown": [
        ("baslik", re.compile(r"^(#{1,6})\s+(.+)$")),
    ],
}

#: Import kalıpları (dil → regex; 1. grup hedef)
IMPORT_KALIPLARI: Dict[str, List[re.Pattern]] = {
    "python": [
        re.compile(r"^\s*from\s+([\w\.]+)\s+import"),
        re.compile(r"^\s*import\s+([\w\.]+)"),
    ],
    "javascript": [
        re.compile(r"^\s*import\s+.*?from\s+['\"]([^'\"]+)['\"]"),
        re.compile(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)"),
    ],
    "typescript": [
        re.compile(r"^\s*import\s+.*?from\s+['\"]([^'\"]+)['\"]"),
    ],
    "gdscript": [
        re.compile(r"preload\(\s*['\"]([^'\"]+)['\"]\s*\)"),
        re.compile(r"^\s*const\s+\w+\s*=\s*preload"),
    ],
    "rust": [re.compile(r"^\s*use\s+([\w:]+)")],
    "go": [re.compile(r"^\s*\"([\w\./]+)\"")],
    "csharp": [re.compile(r"^\s*using\s+([\w\.]+)\s*;")],
    "c": [re.compile(r'^\s*#include\s*[<"]([^>"]+)[>"]')],
    "cpp": [re.compile(r'^\s*#include\s*[<"]([^>"]+)[>"]')],
}


def dil_bul(yol: Path) -> str:
    """Uzantıdan dil adı (yoksa boş)."""
    return DILLER.get(yol.suffix.lower(), "")


def _db_yolu(kok: Path) -> Path:
    """İndeks veritabanı yolu."""
    return Path(kok) / ".neurocore" / "index" / "kod.db"


def _baglan(yol: Path) -> sqlite3.Connection:
    """SQLite bağlantısı (WAL, FTS5 varsa kullanılır)."""
    yol.parent.mkdir(parents=True, exist_ok=True)
    bag = sqlite3.connect(str(yol), timeout=30)
    bag.row_factory = sqlite3.Row
    bag.execute("PRAGMA journal_mode=WAL")
    bag.execute("PRAGMA synchronous=NORMAL")
    return bag


def _sema(bag: sqlite3.Connection) -> None:
    """Tabloları kur (idempotent)."""
    bag.executescript("""
    CREATE TABLE IF NOT EXISTS dosya(
        path TEXT PRIMARY KEY, mtime REAL, size INTEGER,
        satir INTEGER, dil TEXT, sembol_sayi INTEGER);
    CREATE TABLE IF NOT EXISTS sembol(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT, satir INTEGER, ad TEXT, tur TEXT);
    CREATE INDEX IF NOT EXISTS ix_sembol_ad ON sembol(ad);
    CREATE INDEX IF NOT EXISTS ix_sembol_path ON sembol(path);
    CREATE TABLE IF NOT EXISTS iliski(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kaynak TEXT, hedef TEXT, tur TEXT);
    CREATE INDEX IF NOT EXISTS ix_iliski_hedef ON iliski(hedef);
    CREATE INDEX IF NOT EXISTS ix_iliski_kaynak ON iliski(kaynak);
    CREATE TABLE IF NOT EXISTS ustveri(anahtar TEXT PRIMARY KEY, deger TEXT);
    """)
    # FTS5 varsa sembol aramasını hızlandır
    try:
        bag.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS sembol_fts USING fts5(
            ad, path UNINDEXED, satir UNINDEXED, tur UNINDEXED)""")
        bag.execute("INSERT OR REPLACE INTO ustveri VALUES('fts','1')")
    except sqlite3.OperationalError:
        bag.execute("INSERT OR REPLACE INTO ustveri VALUES('fts','0')")
    bag.commit()


@dataclass
class DosyaTarama:
    """Bir dosyanın sembolleri ve ilişkileri."""
    path: str
    satir: int
    dil: str
    semboller: List[Tuple[int, str, str]]
    iliskiler: List[Tuple[str, str]]


def _tara(yol: Path, goreli: str) -> Optional[DosyaTarama]:
    """Dosyayı **akışlı** tarar: satır sayısı, semboller, import'lar."""
    d = dil_bul(yol)
    if not d:
        return None
    sembol_kaliplari = SEMBOL_KALIPLARI.get(d, [])
    import_kaliplari = IMPORT_KALIPLARI.get(d, [])
    semboller: List[Tuple[int, str, str]] = []
    iliskiler: List[Tuple[str, str]] = []
    satir = 0
    try:
        with yol.open("rb") as f:
            for ham in f:
                satir += 1
                if not sembol_kaliplari and not import_kaliplari:
                    continue
                s = ham.decode("utf-8", errors="replace")
                if len(s) > 600:
                    continue
                if len(semboller) < SEMBOL_MAKS:
                    for tur, kalip in sembol_kaliplari:
                        m = kalip.match(s)
                        if m:
                            ad = (m.group(2) if tur == "baslik" and m.lastindex
                                  else m.group(1))
                            semboller.append((satir, str(ad)[:120], tur))
                            break
                if len(iliskiler) < ILISKI_MAKS:
                    for kalip in import_kaliplari:
                        m = kalip.search(s)
                        if m:
                            hedef = str(m.group(1))[:200]
                            if hedef:
                                iliskiler.append((hedef, "import"))
                            break
    except (OSError, UnicodeError):
        return None
    return DosyaTarama(goreli, satir, d, semboller, iliskiler)


def _dosyalar(kok: Path) -> Iterable[Path]:
    """Taranacak dosyaları üretir (atlanan dizinler hariç)."""
    for p in Path(kok).rglob("*"):
        if not p.is_file():
            continue
        if any(parca in ATLA for parca in p.parts):
            continue
        if p.suffix.lower() not in DILLER:
            continue
        yield p


# ─── kurulum / güncelleme ───────────────────────────────────────────────────

def dizinle(kok: Path, zorla: bool = False) -> Dict[str, Any]:
    """
    Proje kod indeksini kurar veya **artımlı** günceller.

    Args:
        kok: Proje kökü
        zorla: True → tüm dosyalar yeniden taranır

    Returns:
        {"ok", "dosya", "sembol", "iliski", "yeni", "guncel", "silinen",
         "sure_ms", "db", "fts"}
    """
    kok = Path(kok).resolve()
    t0 = time.time()
    db = _db_yolu(kok)
    bag = _baglan(db)
    _sema(bag)
    var_olan = {r["path"]: (r["mtime"], r["size"])
                for r in bag.execute("SELECT path, mtime, size FROM dosya")}
    gorulen: set[str] = set()
    yeni = guncellenen = 0
    sembol_sayi = iliski_sayi = 0

    for yol in _dosyalar(kok):
        try:
            stat = yol.stat()
        except OSError:
            continue
        goreli = str(yol.relative_to(kok)).replace("\\", "/")
        gorulen.add(goreli)
        onceki = var_olan.get(goreli)
        if (not zorla and onceki
                and abs(onceki[0] - stat.st_mtime) < 0.001
                and onceki[1] == stat.st_size):
            continue

        if stat.st_size > DOSYA_MAKS:
            # ⚠ Çok büyük dosya: sembol çıkarma yerine yalnız satır say
            satir = _satir_say(yol)
            tarama = DosyaTarama(goreli, satir, dil_bul(yol), [], [])
        else:
            t = _tara(yol, goreli)
            if t is None:
                continue
            tarama = t

        bag.execute("DELETE FROM sembol WHERE path=?", (goreli,))
        bag.execute("DELETE FROM iliski WHERE kaynak=?", (goreli,))
        try:
            bag.execute("DELETE FROM sembol_fts WHERE path=?", (goreli,))
        except sqlite3.OperationalError:
            pass
        bag.execute(
            "INSERT OR REPLACE INTO dosya VALUES(?,?,?,?,?,?)",
            (goreli, stat.st_mtime, stat.st_size, tarama.satir,
             tarama.dil, len(tarama.semboller)))
        if tarama.semboller:
            bag.executemany(
                "INSERT INTO sembol(path, satir, ad, tur) VALUES(?,?,?,?)",
                [(goreli, s, ad, tur) for s, ad, tur in tarama.semboller])
            try:
                bag.executemany(
                    "INSERT INTO sembol_fts(ad, path, satir, tur) "
                    "VALUES(?,?,?,?)",
                    [(ad, goreli, s, tur) for s, ad, tur in tarama.semboller])
            except sqlite3.OperationalError:
                pass
        if tarama.iliskiler:
            bag.executemany(
                "INSERT INTO iliski(kaynak, hedef, tur) VALUES(?,?,?)",
                [(goreli, hedef, tur) for hedef, tur in tarama.iliskiler])
        sembol_sayi += len(tarama.semboller)
        iliski_sayi += len(tarama.iliskiler)
        if onceki:
            guncellenen += 1
        else:
            yeni += 1

    silinen = 0
    for eski in set(var_olan) - gorulen:
        bag.execute("DELETE FROM dosya WHERE path=?", (eski,))
        bag.execute("DELETE FROM sembol WHERE path=?", (eski,))
        bag.execute("DELETE FROM iliski WHERE kaynak=?", (eski,))
        try:
            bag.execute("DELETE FROM sembol_fts WHERE path=?", (eski,))
        except sqlite3.OperationalError:
            pass
        silinen += 1

    bag.execute("INSERT OR REPLACE INTO ustveri VALUES('semver',?)",
                (str(SEMVER),))
    bag.execute("INSERT OR REPLACE INTO ustveri VALUES('son',?)",
                (time.strftime("%Y-%m-%dT%H:%M:%S"),))
    bag.commit()

    toplam = bag.execute(
        "SELECT COUNT(*) c, COALESCE(SUM(satir),0) s FROM dosya").fetchone()
    fts = bag.execute(
        "SELECT deger FROM ustveri WHERE anahtar='fts'").fetchone()
    bag.close()
    sure = (time.time() - t0) * 1000
    return {
        "ok": True,
        "kok": str(kok),
        "db": str(db.relative_to(kok)) if str(db).startswith(str(kok)) else str(db),
        "dosya": toplam["c"],
        "toplam_satir": toplam["s"],
        "sembol": sembol_sayi,
        "iliski": iliski_sayi,
        "yeni": yeni,
        "guncel": guncellenen,
        "silinen": silinen,
        "sure_ms": round(sure, 1),
        "fts": bool(fts and fts["deger"] == "1"),
    }


def _satir_say(yol: Path) -> int:
    """Dosyayı belleğe almadan satır sayar."""
    n = 0
    try:
        with yol.open("rb") as f:
            for _ in f:
                n += 1
    except OSError:
        return 0
    return n


# ─── sorgular ───────────────────────────────────────────────────────────────

def sembol_ara(kok: Path, desen: str, sinir: int = 40) -> Dict[str, Any]:
    """Sembol adında arama (FTS5 varsa tam metin, yoksa LIKE)."""
    kok = Path(kok)
    db = _db_yolu(kok)
    if not db.exists():
        return {"ok": False, "error": "indeks yok — önce code.dizinle"}
    bag = _baglan(db)
    try:
        fts = bag.execute(
            "SELECT deger FROM ustveri WHERE anahtar='fts'").fetchone()
        if fts and fts["deger"] == "1":
            satirlar = bag.execute(
                "SELECT path, satir, ad, tur FROM sembol_fts "
                "WHERE ad MATCH ? LIMIT ?",
                (f"{desen}*", sinir)).fetchall()
        else:
            satirlar = bag.execute(
                "SELECT path, satir, ad, tur FROM sembol "
                "WHERE ad LIKE ? LIMIT ?", (f"%{desen}%", sinir)).fetchall()
        sonuc = [dict(r) for r in satirlar]
    finally:
        bag.close()
    return {"ok": True, "desen": desen, "sonuc": sonuc, "sayi": len(sonuc)}


def referanslar(kok: Path, ad: str, sinir: int = 40) -> Dict[str, Any]:
    """Bir adı içeren dosyalar (import grafiği + sembol tanımları)."""
    kok = Path(kok)
    db = _db_yolu(kok)
    if not db.exists():
        return {"ok": False, "error": "indeks yok"}
    bag = _baglan(db)
    try:
        tanim = [dict(r) for r in bag.execute(
            "SELECT path, satir, ad, tur FROM sembol WHERE ad=? LIMIT ?",
            (ad, sinir))]
        etkiler = [dict(r) for r in bag.execute(
            "SELECT kaynak, hedef, tur FROM iliski WHERE hedef LIKE ? LIMIT ?",
            (f"%{ad}%", sinir))]
    finally:
        bag.close()
    return {"ok": True, "ad": ad, "tanim": tanim, "etkiler": etkiler,
            "sayi": len(tanim) + len(etkiler)}


def iliskili_dosyalar(kok: Path, yol: str, sinir: int = 30) -> Dict[str, Any]:
    """Dosyanın import ettiği ve onu import eden dosyalar."""
    kok = Path(kok)
    db = _db_yolu(kok)
    if not db.exists():
        return {"ok": False, "error": "indeks yok"}
    goreli = yol.replace("\\", "/")
    bag = _baglan(db)
    try:
        giden = [dict(r) for r in bag.execute(
            "SELECT hedef, tur FROM iliski WHERE kaynak=? LIMIT ?",
            (goreli, sinir))]
        gelen = [dict(r) for r in bag.execute(
            "SELECT kaynak, tur FROM iliski WHERE hedef LIKE ? LIMIT ?",
            (f"%{Path(goreli).stem}%", sinir))]
        sembol = [dict(r) for r in bag.execute(
            "SELECT ad, tur, satir FROM sembol WHERE path=? LIMIT ?",
            (goreli, sinir))]
    finally:
        bag.close()
    return {"ok": True, "dosya": goreli, "import_ettigi": giden,
            "kullananlar": gelen, "semboller": sembol}


def proje_haritasi(kok: Path, sinir: int = 20) -> Dict[str, Any]:
    """Dil dağılımı, en büyük dosyalar, kök klasörler, giriş noktaları."""
    kok = Path(kok)
    db = _db_yolu(kok)
    if not db.exists():
        return {"ok": False, "error": "indeks yok"}
    bag = _baglan(db)
    try:
        diller = [dict(r) for r in bag.execute(
            "SELECT dil, COUNT(*) dosya, SUM(satir) satir FROM dosya "
            "GROUP BY dil ORDER BY satir DESC LIMIT 12")]
        buyuk = [dict(r) for r in bag.execute(
            "SELECT path, satir, dil FROM dosya ORDER BY satir DESC LIMIT ?",
            (sinir,))]
        with_symbols = [dict(r) for r in bag.execute(
            "SELECT path, COUNT(*) n FROM sembol GROUP BY path "
            "ORDER BY n DESC LIMIT ?", (sinir,))]
        toplam = bag.execute(
            "SELECT COUNT(*) c, COALESCE(SUM(satir),0) s FROM dosya").fetchone()
    finally:
        bag.close()
    return {
        "ok": True, "dosya": toplam["c"], "toplam_satir": toplam["s"],
        "diller": diller, "en_buyuk": buyuk, "sembol_yogun": with_symbols,
    }
