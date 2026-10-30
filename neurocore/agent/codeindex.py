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

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.codeindex")

#: İndeks sürümü (şema değişirse artır)
SEMVER = 4

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
    CREATE TABLE IF NOT EXISTS tip(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sembol TEXT, ad TEXT, tip TEXT, tur TEXT, dosya TEXT, satir INTEGER);
    CREATE INDEX IF NOT EXISTS ix_tip_sembol ON tip(sembol);
    CREATE INDEX IF NOT EXISTS ix_tip_tip ON tip(tip);
    CREATE TABLE IF NOT EXISTS kalitim(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alt TEXT, taban TEXT, dosya TEXT, satir INTEGER);
    CREATE INDEX IF NOT EXISTS ix_kalitim_alt ON kalitim(alt);
    CREATE INDEX IF NOT EXISTS ix_kalitim_taban ON kalitim(taban);
    CREATE TABLE IF NOT EXISTS cagri(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kaynak_sembol TEXT, hedef_ad TEXT, dosya TEXT, satir INTEGER);
    CREATE INDEX IF NOT EXISTS ix_cagri_hedef ON cagri(hedef_ad);
    CREATE INDEX IF NOT EXISTS ix_cagri_kaynak ON cagri(kaynak_sembol);
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
    cagrilar: List[Tuple[str, str, int]] = field(default_factory=list)
    kalitim: List[Tuple[str, str, int]] = field(default_factory=list)
    tipler: List[Tuple[str, str, str, str, int]] = field(default_factory=list)


# ─── AST tabanlı çıkarım (regex'ten daha isabetli) ──────────────────────────

def _python_ast(yol: Path, goreli: str) -> Optional[DosyaTarama]:
    """
    Python dosyasını **ast** ile ayrıştırır: sınıf/fonksiyon/sabit,
    import'lar ve **fonksiyon içi çağrılar** (çağrı grafiği için).

    Regex'e göre avantajı: yorumlar/string'ler yanlış eşleşmez, nitelikli
    adlar (`Sinif.metod`) ve iç içe tanımlar doğru çıkar.
    """
    import ast as _ast

    try:
        kaynak = yol.read_text(encoding="utf-8", errors="replace")
        agac = _ast.parse(kaynak)
    except (OSError, SyntaxError, ValueError):
        return None

    semboller: List[Tuple[int, str, str]] = []
    iliskiler: List[Tuple[str, str]] = []
    cagrilar: List[Tuple[str, str, int]] = []
    satir_sayisi = kaynak.count("\n") + (1 if kaynak else 0)
    kalitim: List[Tuple[str, str, int]] = []
    tipler: List[Tuple[str, str, str, str, int]] = []

    def _ekle(satir: int, ad: str, tur: str) -> None:
        if len(semboller) < SEMBOL_MAKS and ad:
            semboller.append((satir, ad[:120], tur))

    class Gez(_ast.NodeVisitor):
        def __init__(self) -> None:
            self.yigin: List[str] = []

        def visit_ClassDef(self, node: "_ast.ClassDef") -> None:
            _ekle(node.lineno, node.name, "class")
            # ⚠ (b) sınıf hiyerarşisi: taban sınıfları ilişki olarak sakla
            for taban in node.bases:
                ad_taban = _cagri_adi(taban)
                if ad_taban and len(kalitim) < 200:
                    # alt sınıf → taban sınıf (hiyerarşi sorgusu için)
                    kalitim.append((node.name[:120], ad_taban[:120],
                                    node.lineno))
            self.yigin.append(node.name)
            self.generic_visit(node)
            self.yigin.pop()

        def _fonksiyon(self, node: Any) -> None:
            ad = ".".join(self.yigin + [node.name])
            _ekle(node.lineno, node.name, "func")
            self.yigin.append(node.name)
            # (b) tip çıkarımı: parametre ve dönüş ek açıklamaları
            for arg in list(node.args.args) + list(getattr(node.args, "kwonlyargs", [])):
                if getattr(arg, "annotation", None) is not None:
                    tip_ad = _cagri_adi(arg.annotation) or str(
                        getattr(arg.annotation, "id", ""))
                    if tip_ad and len(tipler) < 300:
                        tipler.append((ad[:160], arg.arg[:60], tip_ad[:80],
                                       "parametre", node.lineno))
            if getattr(node, "returns", None) is not None:
                tip_ad = _cagri_adi(node.returns) or str(
                    getattr(node.returns, "id", ""))
                if tip_ad and len(tipler) < 300:
                    tipler.append((ad[:160], "dönüş", tip_ad[:80],
                                   "dönüş", node.lineno))
            for alt in _ast.walk(node):
                if isinstance(alt, _ast.Call):
                    hedef = _cagri_adi(alt.func)
                    if hedef and len(cagrilar) < 5000:
                        cagrilar.append((ad[:160], hedef[:120], alt.lineno))
                # (b) fonksiyon içi atama tipi: z = Sinif()
                if isinstance(alt, _ast.Assign):
                    for hedef in alt.targets:
                        if (isinstance(hedef, _ast.Name)
                                and isinstance(alt.value, _ast.Call)):
                            tip_ad = _cagri_adi(alt.value.func)
                            if tip_ad and len(tipler) < 300:
                                tipler.append((ad[:160], hedef.id[:60],
                                               tip_ad[:80], "atama",
                                               alt.lineno))
            self.yigin.pop()

        def visit_FunctionDef(self, node: "_ast.FunctionDef") -> None:
            self._fonksiyon(node)

        def visit_AsyncFunctionDef(self, node: "_ast.AsyncFunctionDef") -> None:
            self._fonksiyon(node)

        def visit_Import(self, node: "_ast.Import") -> None:
            for alias in node.names:
                iliskiler.append((alias.name, "import"))

        def visit_ImportFrom(self, node: "_ast.ImportFrom") -> None:
            modul = node.module or ""
            for alias in node.names:
                iliskiler.append((f"{modul}.{alias.name}" if modul else alias.name,
                                  "import"))

        def visit_Assign(self, node: "_ast.Assign") -> None:
            for hedef in node.targets:
                if isinstance(hedef, _ast.Name) and hedef.id.isupper():
                    _ekle(node.lineno, hedef.id, "const")
                # (b) x = Sinif() → x'in tipi Sinif (kaba çıkarım)
                if (isinstance(hedef, _ast.Name)
                        and isinstance(node.value, _ast.Call)):
                    tip_ad = _cagri_adi(node.value.func)
                    if tip_ad and len(tipler) < 300:
                        tipler.append(("", hedef.id[:60], tip_ad[:80],
                                       "atama", node.lineno))

    Gez().visit(agac)
    # sınırları uygula
    del iliskiler[ILISKI_MAKS:]
    del cagrilar[5000:]
    return DosyaTarama(goreli, satir_sayisi, "python", semboller, iliskiler,
                       cagrilar, kalitim, tipler)


def _cagri_adi(dugum: Any) -> str:
    """ast.Call func düğümünden okunur ad üretir (Name / Attribute zinciri)."""
    import ast as _ast

    if isinstance(dugum, _ast.Name):
        return dugum.id
    if isinstance(dugum, _ast.Attribute):
        taban = _cagri_adi(dugum.value)
        return f"{taban}.{dugum.attr}" if taban else dugum.attr
    if isinstance(dugum, _ast.Call):
        return _cagri_adi(dugum.func)
    return ""


def _gdscript_tara(yol: Path, goreli: str) -> Optional[DosyaTarama]:
    """
    GDScript dosyasını satır bazlı ayrıştırır: class_name/func/signal/const,
    preload ilişkileri ve **fonksiyon içi çağrılar**.

    GDScript'te Python `ast` muadili yok; girinti ile fonksiyon gövdesi
    izlenir (sekme/boşluk sayısı).
    """
    semboller: List[Tuple[int, str, str]] = []
    iliskiler: List[Tuple[str, str]] = []
    cagrilar: List[Tuple[str, str, int]] = []
    aktif_func: Optional[str] = None
    aktif_girinti = -1
    satir = 0
    try:
        with yol.open("r", encoding="utf-8", errors="replace") as f:
            for ham in f:
                satir += 1
                if not ham.strip() or ham.lstrip().startswith("#"):
                    continue
                girinti = len(ham) - len(ham.lstrip())
                s = ham.strip()
                if aktif_func and girinti <= aktif_girinti and not s.startswith("func "):
                    aktif_func = None
                if s.startswith("class_name "):
                    semboller.append((satir, s.split()[1][:120], "class"))
                elif s.startswith("func ") or s.startswith("static func "):
                    parca = s.split("func ", 1)[1]
                    ad = parca.split("(", 1)[0].strip()
                    if ad:
                        semboller.append((satir, ad[:120], "func"))
                        aktif_func = ad
                        aktif_girinti = girinti
                elif s.startswith("signal "):
                    semboller.append((satir, s.split()[1][:120], "signal"))
                elif s.startswith("const "):
                    ad = s[6:].split("=", 1)[0].strip().split(":")[0].strip()
                    if ad:
                        semboller.append((satir, ad[:120], "const"))
                # preload / load ilişkisi
                for isaret in ("preload(", "load("):
                    if isaret in s:
                        try:
                            hedef = s.split(isaret, 1)[1].split(")", 1)[0]
                            hedef = hedef.strip().strip('"').strip("'")
                            if hedef:
                                iliskiler.append((hedef[:200], "import"))
                        except IndexError:
                            pass
                # çağrı: ad( ... (tanım satırı hariç)
                if aktif_func and "(" in s and not s.startswith("func "):
                    for parca in s.replace("(", " ").split():
                        aday = parca.strip(",.:;)").strip()
                        if (aday and (aday[0].isalpha() or aday[0] == "_")
                                and len(aday) > 2
                                and aday not in ("if", "for", "while", "return",
                                                 "var", "and", "or", "not",
                                                 "elif", "else", "match", "func")
                                and len(cagrilar) < 5000):
                            cagrilar.append((aktif_func, aday[:120], satir))
    except OSError:
        return None
    del iliskiler[ILISKI_MAKS:]
    return DosyaTarama(goreli, satir, "gdscript", semboller, iliskiler, cagrilar)


def _tara(yol: Path, goreli: str) -> Optional[DosyaTarama]:
    """Dosyayı **akışlı** tarar: satır sayısı, semboller, import'lar."""
    d = dil_bul(yol)
    if not d:
        return None
    # ⚠ AST / yapısal ayrıştırıcı varsa onu kullan (regex'ten isabetli)
    if d == "python":
        return _python_ast(yol, goreli)
    if d == "gdscript":
        return _gdscript_tara(yol, goreli)
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


def _gdignore_dizinler(kok: Path) -> set:
    """
    `.gdignore` içeren dizinleri toplar (Godot kuralı: alt ağaç taranmaz).

    ⚠ ÖLÇÜLEN EKSİK: ajan TitoWarWeb'de `addons/.godot_ai_update/backup/`
    dizinine `.gdignore` koydu; Godot görmezden geldi ama **bizim indeks**
    hâlâ 140 yedek dosyayı tarıyordu. Bu yardımcı o boşluğu kapatır.
    """
    kok = Path(kok)
    bulunan: set = set()
    try:
        for p in kok.rglob(".gdignore"):
            bulunan.add(p.parent.resolve())
    except OSError:
        pass
    return bulunan


def _dosyalar(kok: Path) -> Iterable[Path]:
    """Taranacak dosyaları üretir (atlanan dizinler + `.gdignore` hariç)."""
    kok = Path(kok)
    yoksay = _gdignore_dizinler(kok)
    for p in kok.rglob("*"):
        if not p.is_file():
            continue
        if any(parca in ATLA for parca in p.parts):
            continue
        if p.suffix.lower() not in DILLER:
            continue
        if yoksay:
            try:
                if any(ust in yoksay for ust in p.parents):
                    continue
            except OSError:
                pass
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
    # ⚠ Şema sürümü değiştiyse (yeni tablo/alan) tam tarama yap
    onceki_semver = bag.execute(
        "SELECT deger FROM ustveri WHERE anahtar='semver'").fetchone()
    if onceki_semver and onceki_semver["deger"] != str(SEMVER):
        zorla = True
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
        bag.execute("DELETE FROM cagri WHERE dosya=?", (goreli,))
        bag.execute("DELETE FROM kalitim WHERE dosya=?", (goreli,))
        bag.execute("DELETE FROM tip WHERE dosya=?", (goreli,))
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
        if tarama.cagrilar:
            bag.executemany(
                "INSERT INTO cagri(kaynak_sembol, hedef_ad, dosya, satir) "
                "VALUES(?,?,?,?)",
                [(k, h, goreli, sat) for k, h, sat in tarama.cagrilar])
        if tarama.tipler:
            bag.executemany(
                "INSERT INTO tip(sembol, ad, tip, tur, dosya, satir) "
                "VALUES(?,?,?,?,?,?)",
                [(sem, ad, tip, tur, goreli, sat)
                 for sem, ad, tip, tur, sat in tarama.tipler])
        if tarama.kalitim:
            bag.executemany(
                "INSERT INTO kalitim(alt, taban, dosya, satir) "
                "VALUES(?,?,?,?)",
                [(a, t, goreli, sat) for a, t, sat in tarama.kalitim])
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
        bag.execute("DELETE FROM cagri WHERE dosya=?", (eski,))
        bag.execute("DELETE FROM kalitim WHERE dosya=?", (eski,))
        bag.execute("DELETE FROM tip WHERE dosya=?", (eski,))
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

    cagri_sayi = bag.execute("SELECT COUNT(*) c FROM cagri").fetchone()["c"]
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
        "cagri": cagri_sayi,
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

def cagri_grafigi(kok: Path, ad: str, sinir: int = 40) -> Dict[str, Any]:
    """
    Çağrı grafiği: `ad` fonksiyonunu **kim çağırıyor**, o **kimi çağırıyor**.

    Args:
        kok: Proje kökü (indeksli olmalı)
        ad: Fonksiyon/metot adı (kısmi eşleşme)
        sinir: En fazla kayıt

    Returns:
        {"ok", "ad", "cagiranlar": [...], "cagardiklari": [...]}
    """
    kok = Path(kok)
    db = _db_yolu(kok)
    if not db.exists():
        return {"ok": False, "error": "indeks yok — önce code.index"}
    bag = _baglan(db)
    try:
        cagiran = [dict(r) for r in bag.execute(
            "SELECT kaynak_sembol, dosya, satir FROM cagri "
            "WHERE hedef_ad LIKE ? LIMIT ?", (f"%{ad}%", sinir))]
        cagardigi = [dict(r) for r in bag.execute(
            "SELECT hedef_ad, dosya, satir FROM cagri "
            "WHERE kaynak_sembol LIKE ? LIMIT ?", (f"%{ad}%", sinir))]
    finally:
        bag.close()
    return {"ok": True, "ad": ad, "cagiranlar": cagiran,
            "cagardiklari": cagardigi,
            "sayi": len(cagiran) + len(cagardigi)}

def hiyerarsi(kok: Path, ad: str, sinir: int = 40) -> Dict[str, Any]:
    """
    Sınıf hiyerarşisi: verilen sınıfın **tabanları** ve **alt sınıfları**.

    Args:
        kok: Proje kökü (indeksli)
        ad: Sınıf adı (kısmi eşleşme)
        sinir: En fazla kayıt

    Returns:
        {"ok", "ad", "tanim": [...], "tabanlar": [...], "alt_siniflar": [...]}
    """
    kok = Path(kok)
    db = _db_yolu(kok)
    if not db.exists():
        return {"ok": False, "error": "indeks yok — önce code.index"}
    bag = _baglan(db)
    try:
        tanim = [dict(r) for r in bag.execute(
            "SELECT path, satir, ad FROM sembol WHERE ad LIKE ? AND tur='class' "
            "LIMIT ?", (f"%{ad}%", sinir))]
        tabanlar = [dict(r) for r in bag.execute(
            "SELECT alt, taban, dosya, satir FROM kalitim WHERE alt LIKE ? "
            "LIMIT ?", (f"%{ad}%", sinir))]
        alt = [dict(r) for r in bag.execute(
            "SELECT alt, taban, dosya, satir FROM kalitim WHERE taban LIKE ? "
            "LIMIT ?", (f"%{ad}%", sinir))]
    finally:
        bag.close()
    return {"ok": True, "ad": ad, "tanim": tanim,
            "tabanlar": tabanlar, "alt_siniflar": alt,
            "sayi": len(tanim) + len(tabanlar) + len(alt)}


def etki_haritasi(kok: Path, ad: str, derinlik: int = 1,
                  sinir: int = 25) -> Dict[str, Any]:
    """
    Etki haritası (etkileşimli HTML). `etki_haritasi_interaktif` sarmalayıcısı.

    Geriye dönük uyumluluk için aynı imza korunur; çıktı artık tıklanabilir
    grafik içerir.
    """
    return etki_haritasi_interaktif(kok, ad, derinlik=max(1, derinlik),
                                    sinir=max(1, sinir))


def etki_haritasi_interaktif(kok: Path, ad: str, derinlik: int = 2,
                             sinir: int = 20) -> Dict[str, Any]:
    """
    (c) **Etkileşimli** etki haritası: tek dosya HTML + gömülü JSON.

    Tıklanan düğüm seçilir; çağıranları/çağrılanları vurgulanır ve sağdaki
    panelde dosya:satır bilgisi görünür. Derinlik 2'de ikinci komşular da
    yüklenir. Dış kaynak/CDN yoktur (çevrimdışı açılır).

    Returns:
        {"ok", "ad", "dugum", "kenar", "html", "bytes"}
    """
    kok = Path(kok)
    db = _db_yolu(kok)
    if not db.exists():
        return {"ok": False, "error": "indeks yok — önce code.index"}
    bag = _baglan(db)
    dugumler: Dict[str, Dict[str, Any]] = {}
    kenarlar: List[Dict[str, Any]] = []

    def _ekle(ad_: str, taraf: str, dosya: str = "", satir: int = 0) -> None:
        anahtar = f"{taraf}:{ad_}"
        if anahtar not in dugumler and len(dugumler) < sinir * 4:
            dugumler[anahtar] = {"id": anahtar, "ad": ad_, "taraf": taraf,
                                 "dosya": dosya, "satir": satir}

    try:
        _ekle(ad, "odak")
        for r in bag.execute(
                "SELECT kaynak_sembol, dosya, satir FROM cagri "
                "WHERE hedef_ad LIKE ? LIMIT ?", (f"%{ad}%", sinir)):
            _ekle(r["kaynak_sembol"], "cagiran", r["dosya"], r["satir"])
            kenarlar.append({"kaynak": f"cagiran:{r['kaynak_sembol']}",
                             "hedef": f"odak:{ad}"})
        for r in bag.execute(
                "SELECT hedef_ad, dosya, satir FROM cagri "
                "WHERE kaynak_sembol LIKE ? LIMIT ?", (f"%{ad}%", sinir)):
            _ekle(r["hedef_ad"], "cagrilan", r["dosya"], r["satir"])
            kenarlar.append({"kaynak": f"odak:{ad}",
                             "hedef": f"cagrilan:{r['hedef_ad']}"})
        if derinlik >= 2:
            for d in [x for x in dugumler.values() if x["taraf"] == "cagiran"][:8]:
                for r in bag.execute(
                        "SELECT kaynak_sembol, dosya, satir FROM cagri "
                        "WHERE hedef_ad LIKE ? LIMIT 4", (f"%{d['ad']}%",)):
                    _ekle(r["kaynak_sembol"], "cagiran2", r["dosya"], r["satir"])
                    kenarlar.append({"kaynak": f"cagiran2:{r['kaynak_sembol']}",
                                     "hedef": f"cagiran:{d['ad']}"})
            for d in [x for x in dugumler.values() if x["taraf"] == "cagrilan"][:8]:
                for r in bag.execute(
                        "SELECT hedef_ad, dosya, satir FROM cagri "
                        "WHERE kaynak_sembol LIKE ? LIMIT 4", (f"%{d['ad']}%",)):
                    _ekle(r["hedef_ad"], "cagrilan2", r["dosya"], r["satir"])
                    kenarlar.append({"kaynak": f"cagrilan:{d['ad']}",
                                     "hedef": f"cagrilan2:{r['hedef_ad']}"})
    finally:
        bag.close()

    veri = json.dumps({"ad": ad, "dugumler": list(dugumler.values()),
                       "kenarlar": kenarlar}, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"><title>Etki haritası: {ad}</title>
<style>
 body{{margin:0;background:#0f1115;color:#e6e8ee;
      font:14px/1.5 system-ui,Segoe UI,sans-serif;display:flex}}
 #grafik{{flex:1;position:relative;min-height:100vh}}
 #panel{{width:300px;border-left:1px solid #2a2f3a;padding:16px;
        background:#171a21;overflow:auto}}
 h2{{font-size:18px;margin:0 0 4px}} .sub{{color:#9aa4b2;font-size:12px}}
 .dugum{{position:absolute;transform:translate(-50%,-50%);padding:6px 10px;
        border:1px solid #3a4250;border-radius:8px;background:#1d2836;
        font-family:Consolas,monospace;font-size:11px;cursor:pointer;
        white-space:nowrap;max-width:170px;overflow:hidden;text-overflow:ellipsis}}
 .dugum.odak{{background:#2f6b4f;border-color:#7ee787;font-size:12px}}
 .dugum.sec{{outline:2px solid #8ab4f8}} .dugum.soluk{{opacity:.25}}
 svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
 line{{stroke:#8ab4f8;stroke-width:1.2;opacity:.35}}
 line.vurgu{{stroke:#7ee787;opacity:.95;stroke-width:2}}
</style></head><body>
<div id="grafik"><svg id="kenarlar"></svg></div>
<div id="panel"><h2>{ad}</h2><div class="sub">etki haritası · tıkla</div>
 <div id="detay" style="margin-top:14px;color:#9aa4b2">Bir düğüme tıkla.</div>
</div>
<script>
const VERI = {veri};
const kutu = document.getElementById('grafik');
const svg = document.getElementById('kenarlar');
const detay = document.getElementById('detay');
const Y = {{cagiran2: 60, cagiran: 200, odak: 360, cagrilan: 520, cagrilan2: 660}};
const gruplar = {{}};
for (const d of VERI.dugumler) (gruplar[d.taraf] ||= []).push(d);
const konum = {{}};
for (const [taraf, liste] of Object.entries(gruplar)) {{
  const x = {{cagiran2: 90, cagiran: 250, odak: 480, cagrilan: 710, cagrilan2: 880}}[taraf] || 480;
  liste.forEach((d, i) => {{
    const y = Y[taraf] + (i - (liste.length - 1) / 2) * 46;
    konum[d.id] = [x, y];
    const el = document.createElement('div');
    el.className = 'dugum' + (taraf === 'odak' ? ' odak' : '');
    el.style.left = x + 'px'; el.style.top = y + 'px';
    el.textContent = d.ad; el.dataset.id = d.id;
    el.onclick = () => sec(d);
    kutu.appendChild(el);
        d.el = el;
  }});
}}
for (const k of VERI.kenarlar) {{
  const a = konum[k.kaynak], b = konum[k.hedef];
  if (!a || !b) continue;
  const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  l.setAttribute('x1', a[0]); l.setAttribute('y1', a[1]);
  l.setAttribute('x2', b[0]); l.setAttribute('y2', b[1]);
  l.dataset.kaynak = k.kaynak; l.dataset.hedef = k.hedef;
  svg.appendChild(l);
}}
function sec(d) {{
  for (const x of VERI.dugumler) x.el.classList.remove('sec', 'soluk');
  const komsu = new Set([d.id]);
  for (const k of VERI.kenarlar) {{
    if (k.kaynak === d.id) komsu.add(k.hedef);
    if (k.hedef === d.id) komsu.add(k.kaynak);
  }}
  for (const x of VERI.dugumler)
    if (!komsu.has(x.id)) x.el.classList.add('soluk');
  d.el.classList.add('sec');
  for (const l of svg.querySelectorAll('line'))
    l.classList.toggle('vurgu',
      l.dataset.kaynak === d.id || l.dataset.hedef === d.id);
  const cagiran = VERI.kenarlar.filter(k => k.hedef === d.id).length;
  const cagrilan = VERI.kenarlar.filter(k => k.kaynak === d.id).length;
  detay.innerHTML = '<b style="color:#e6e8ee">' + d.ad + '</b><br>' +
    (d.dosya ? d.dosya + ':' + d.satir + '<br>' : '') +
    'çağıran: ' + cagiran + ' · çağrılan: ' + cagrilan;
}}
</script></body></html>"""

    rapor = kok / ".neurocore" / "raporlar"
    rapor.mkdir(parents=True, exist_ok=True)
    guvenli = "".join(c if c.isalnum() or c in "-_" else "_" for c in ad)[:40]
    dosya = rapor / f"etki-{guvenli}.html"
    dosya.write_text(html, encoding="utf-8")
    return {"ok": True, "ad": ad, "dugum": len(dugumler),
            "kenar": len(kenarlar), "html": str(dosya),
            "bytes": len(html)}

def mro(kok: Path, ad: str, sinir: int = 12) -> Dict[str, Any]:
    """
    (b) Kalıtım zinciri (MRO benzeri): tabanlar, tabanların tabanları…

    Returns:
        {"ok", "ad", "zincir": [...], "derinlik": int}
    """
    kok = Path(kok)
    db = _db_yolu(kok)
    if not db.exists():
        return {"ok": False, "error": "indeks yok — önce code.index"}
    bag = _baglan(db)
    try:
        zincir: List[str] = [ad]
        gorulen = {ad}
        kuyruk = [ad]
        while kuyruk and len(zincir) < sinir:
            mevcut = kuyruk.pop(0)
            for r in bag.execute(
                    "SELECT taban FROM kalitim WHERE alt LIKE ? LIMIT 8",
                    (f"%{mevcut}%",)):
                taban = r["taban"]
                if taban and taban not in gorulen:
                    gorulen.add(taban)
                    zincir.append(taban)
                    kuyruk.append(taban)
    finally:
        bag.close()
    return {"ok": True, "ad": ad, "zincir": zincir,
            "derinlik": max(0, len(zincir) - 1)}


def tipler(kok: Path, ad: str, sinir: int = 40) -> Dict[str, Any]:
    """
    (b) Tip çıkarımı: parametre/dönüş tipleri + atama çıkarımları.

    Returns:
        {"ok", "ad", "kayitlar": [{sembol, ad, tip, tur, dosya, satir}]}
    """
    kok = Path(kok)
    db = _db_yolu(kok)
    if not db.exists():
        return {"ok": False, "error": "indeks yok — önce code.index"}
    bag = _baglan(db)
    try:
        kayitlar = [dict(r) for r in bag.execute(
            "SELECT sembol, ad, tip, tur, dosya, satir FROM tip "
            "WHERE sembol LIKE ? OR ad LIKE ? OR tip LIKE ? LIMIT ?",
            (f"%{ad}%", f"%{ad}%", f"%{ad}%", sinir))]
    finally:
        bag.close()
    return {"ok": True, "ad": ad, "kayitlar": kayitlar,
            "sayi": len(kayitlar)}
