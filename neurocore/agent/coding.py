"""
NeuroCore Kodlama Ajanı
========================
Bir kod tabanını **anlayan**, üzerinde **güvenle değişiklik yapan** ve
yaptığı işi **doğrulayan** araç seti.

Tasarım ilkeleri:
    * **Bağlam önce** — ajan önce projeyi tarar (`code.workspace_map`), sonra
      hedefli okuma yapar. Tüm depoyu isteme doldurmak yerine ilgili parçayı
      bulur.
    * **Cerrahi düzenleme** — `code.edit_file` eski metni yenisiyle değiştirir;
      dosyanın tamamını yeniden yazmaz. Bu, modele daha az token harcatır ve
      yanlışlıkla içerik kaybını önler.
    * **Her zaman yedek** — değiştirmeden önce içerik bellekte tutulur;
      `code.revert` ile geri alınabilir.
    * **Doğrulama zorunlu** — `code.run_tests` ve `code.check_syntax` ile
      değişikliğin gerçekten çalıştığı kanıtlanır.
    * **Dil farkındalığı** — Python için `ast` ile gerçek sembol çıkarımı;
      diğer diller için düzenli ifade tabanlı üst düzey tarama (dürüstçe
      `approximate=True` işaretlenir).

Bu modül `neurocore.agent.tools` ve `neurocore.utils` dışında hiçbir şeye
bağlı değildir; işletim sistemi katmanı **enjekte edilir** (sandbox için).
"""

from __future__ import annotations

import ast
import difflib
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from neurocore.agent.tools import (
    Safety,
    ToolContext,
    ToolDef,
    ToolParameter,
    collect_tools,
    tool,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.coding")


# ─── Sabitler ───────────────────────────────────────────────────────────────

#: Kaynak kod uzantıları ve dilleri
LANGUAGE_BY_EXTENSION: Dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".gd": "gdscript", ".cs": "csharp", ".java": "java", ".kt": "kotlin",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".rs": "rust", ".go": "go", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".lua": "lua", ".sh": "shell", ".ps1": "powershell", ".psm1": "powershell",
    ".bat": "batch", ".cmd": "batch",
    ".html": "html", ".css": "css", ".scss": "scss",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".md": "markdown", ".rst": "rst", ".txt": "text",
    ".xml": "xml", ".sql": "sql", ".glsl": "glsl", ".shader": "shader",
    ".tscn": "godot_scene", ".tres": "godot_resource", ".godot": "godot_project",
    ".blend": "blender",
}

#: Taranmayacak dizinler (gürültü ve boyut)
DEFAULT_EXCLUDES: Tuple[str, ...] = (
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", ".venv", "venv", "env", ".env",
    "dist", "build", ".next", ".nuxt", "target", "bin", "obj",
    ".idea", ".vscode", ".vs", "*.egg-info", ".tox",
    "Library", "Temp", "Logs", "obj", "Builds",  # Godot/Unity
    ".godot", "addons",                        # Godot iç dizinler
    "site-packages", ".dsh", "screenshots", "checkpoints",
)

#: İkili/okunamayacak uzantılar
BINARY_EXTENSIONS: Set[str] = {
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".pyo", ".obj", ".o", ".a",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff",
    ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mkv", ".webm",
    ".zip", ".gz", ".7z", ".rar", ".tar", ".xz", ".bz2",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".db", ".sqlite", ".sqlite3", ".dat", ".pack", ".idx",
    ".blend1", ".blend2", ".psd", ".ai", ".fbx", ".glb", ".gltf",
}

#: Varsayılan olarak düzenlenebilir uzantılar (ikili dosyalar hariç)
EDITABLE_EXTENSIONS: Set[str] = {
    ext for ext in LANGUAGE_BY_EXTENSION if ext not in (".blend",)
}

#: Proje türü imzaları (kökte görülen dosya → proje türü)
PROJECT_SIGNATURES: List[Tuple[str, str, str]] = [
    ("project.godot", "godot", "Godot oyun motoru projesi"),
    ("*.blend", "blender", "Blender 3B projesi"),
    ("pyproject.toml", "python", "Python paketi (PEP 621)"),
    ("setup.py", "python", "Python paketi (setuptools)"),
    ("requirements.txt", "python", "Python bağımlılık listesi"),
    ("Pipfile", "python", "Python (pipenv)"),
    ("package.json", "node", "Node.js/JavaScript projesi"),
    ("Cargo.toml", "rust", "Rust projesi (cargo)"),
    ("go.mod", "go", "Go modülü"),
    ("pom.xml", "java", "Java (Maven)"),
    ("build.gradle", "java", "Java/Kotlin (Gradle)"),
    ("*.csproj", "dotnet", ".NET projesi"),
    ("*.sln", "dotnet", ".NET çözümü"),
    ("CMakeLists.txt", "cpp", "CMake projesi"),
    ("Makefile", "make", "Make tabanlı proje"),
    ("docker-compose.yml", "docker", "Docker Compose"),
    ("Dockerfile", "docker", "Docker imajı"),
    ("*.unity", "unity", "Unity sahnesi"),
    (".gitignore", "git", "Git deposu"),
]


class CodingError(Exception):
    """Kodlama aracı hatası."""


# ─── Veri Yapıları ──────────────────────────────────────────────────────────

@dataclass
class FileInfo:
    """Bir dosyanın özeti."""
    path: str
    relative: str
    language: str = "text"
    size: int = 0
    lines: int = 0
    modified: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "relative": self.relative, "path": self.path,
            "language": self.language, "size": self.size, "lines": self.lines,
            "modified": self.modified,
        }


@dataclass
class Symbol:
    """
    Kod içindeki bir sembol (sınıf, fonksiyon, değişken).

    `approximate=True` ise sembol düzenli ifadeyle bulunmuştur (Python dışı
    diller); bu dürüstçe bildirilir çünkü yanlış olabilir.
    """
    name: str
    kind: str                     # class | function | method | variable | import | const
    file: str = ""
    line: int = 0
    column: int = 0
    end_line: int = 0
    parent: str = ""              # içinde bulunduğu sınıf (metotlar için)
    signature: str = ""
    docstring: str = ""
    approximate: bool = False

    @property
    def qualified(self) -> str:
        """`Dosya::Sınıf.metot` biçiminde tam ad."""
        parts = [self.file] if self.file else []
        name = f"{self.parent}.{self.name}" if self.parent else self.name
        parts.append(name)
        return "::".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        data = {
            "name": self.name, "kind": self.kind, "file": self.file,
            "line": self.line, "qualified": self.qualified,
        }
        if self.parent:
            data["parent"] = self.parent
        if self.signature:
            data["signature"] = self.signature
        if self.docstring:
            data["docstring"] = self.docstring[:200]
        if self.end_line:
            data["end_line"] = self.end_line
        if self.approximate:
            data["approximate"] = True
        return data


@dataclass
class SearchHit:
    """Bir metin araması sonucu."""
    file: str
    line: int
    text: str
    column: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {"file": self.file, "line": self.line,
                "text": self.text.strip()[:300], "column": self.column}


@dataclass
class EditResult:
    """Bir dosya düzenlemesinin sonucu."""
    ok: bool
    path: str = ""
    replaced: int = 0
    language: str = "text"
    diff: str = ""
    error: str = ""
    backup_id: str = ""
    syntax_ok: Optional[bool] = None
    syntax_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        data = {
            "ok": self.ok, "path": self.path, "replaced": self.replaced,
            "language": self.language, "error": self.error,
            "backup_id": self.backup_id,
        }
        if self.syntax_ok is not None:
            data["syntax_ok"] = self.syntax_ok
            if self.syntax_error:
                data["syntax_error"] = self.syntax_error
        if self.diff:
            data["diff"] = self.diff[:4000]
        return data


# ─── Kodlama Ajanı ──────────────────────────────────────────────────────────

class CodingAgent:
    """
    Kod tabanı araçları.

    Kullanım:
        coding = CodingAgent()
        coding.initialize(workspace="E:/PROJELER/myAi", config=config, event_bus=bus)

        registry.register_many(coding.tools())
        # veya doğrudan:
        print(coding.read_file("neurocore/core/engine.py", max_lines=50))

    Tüm yollar **çalışma alanı köküne** göre çözülür ve kök dışına çıkan
    yollar reddedilir (kum havuzu). `os_layer` enjekte edilirse ek olarak o
    katmanın kum havuzu denetimi de uygulanır.
    """

    name = "agent.coding"

    #: Geri alma için tutulan maksimum yedek
    MAX_BACKUPS = 100

    def __init__(self) -> None:
        # ⚠ tur 14: araştırma araçları için tembel önbellek
        self._arastirmaci_onbellek: Any = None
        self._workspace: Optional[Path] = None
        self._config: Any = None
        self._event_bus: Any = None
        self._os_layer: Any = None
        self._initialized = False
        self._backups: Dict[str, Dict[str, Any]] = {}
        self._cache: Dict[str, Any] = {}
        self._cache_stamp = 0.0

        self._counter = {
            "files_read": 0, "files_written": 0, "searches": 0,
            "maps": 0, "symbols_found": 0, "edits": 0, "reverts": 0,
            "commands": 0, "errors": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, workspace: Optional[str] = None, config: Any = None,
                   event_bus: Any = None, os_layer: Any = None) -> None:
        """
        Kodlama ajanını başlatır.

        Args:
            workspace: Çalışma alanı kökü (None = config.base_path veya cwd)
            config: SystemConfig
            event_bus: EventBus
            os_layer: `OSLayer` (kum havuzu denetimi ve komut çalıştırma için)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._os_layer = os_layer

        root = workspace
        if not root:
            root = getattr(config, "base_path", None) or os.getcwd()
        self._workspace = Path(root).expanduser().resolve()
        if not self._workspace.exists():
            raise CodingError(f"Çalışma alanı yok: {self._workspace}")

        self._initialized = True
        logger.info(f"Kodlama ajanı hazır — çalışma alanı: {self._workspace}")

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır."""
        self._backups.clear()
        self._cache.clear()
        self._initialized = False
        logger.info(
            f"Kodlama ajanı kapatıldı ({self._counter['files_read']} okuma, "
            f"{self._counter['edits']} düzenleme)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """İstatistikler."""
        return {
            **self._counter,
            "initialized": self._initialized,
            "workspace": str(self._workspace or ""),
            "backups": len(self._backups),
        }

    @property
    def workspace(self) -> Path:
        """Çalışma alanı kökü."""
        if self._workspace is None:
            raise CodingError("Kodlama ajanı başlatılmadı — initialize() çağırın")
        return self._workspace

    def set_workspace(self, path: str) -> str:
        """Çalışma alanını değiştirir (kullanıcı başka proje açtığında)."""
        root = Path(path).expanduser().resolve()
        if not root.exists():
            raise CodingError(f"Dizin yok: {root}")
        self._workspace = root
        self._cache.clear()
        logger.info(f"Çalışma alanı değişti: {root}")
        return str(root)

    # ─── Yol Güvenliği ──────────────────────────────────────────────────

    def resolve(self, path: str, must_exist: bool = False) -> Path:
        """
        Yolu çalışma alanına göre çözer ve kök dışına çıkışı engeller.

        Args:
            path: Göreli veya mutlak yol (`~` ve ortam değişkenleri genişletilir)
            must_exist: Dosya/dizim yoksa hata ver

        Returns:
            Mutlak `Path`

        Raises:
            CodingError: Yol çalışma alanı dışındaysa veya yoksa

        Not:
            `""`, `"."` ve `"./"` **çalışma alanı kökü** sayılır. Aksi hâlde
            bunlar süreç CWD'sine çözülür ve ajan yanlış dizini okur —
            ölçüldü: `TestProjects` çalışma alanındayken `path="."` gönderen
            `code.list_directory`, uygulamanın başlatıldığı `myAi` kökünü
            listeliyordu.
        """
        root = self.workspace
        raw = str(path or "").strip()
        if raw in ("", ".", "./", ".\\"):
            candidate = root
        else:
            expanded = os.path.expandvars(os.path.expanduser(raw))
            candidate = Path(expanded)
            if not candidate.is_absolute():
                candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise CodingError(f"Yol çözümlenemedi ({path}): {exc}") from exc

        # Kum havuzu: yalnızca çalışma alanı içi
        try:
            resolved.relative_to(root)
        except ValueError:
            # Sistem geçici dizini de izinli (araç çıktıları için)
            temp_root = Path(os.environ.get("TEMP", os.environ.get("TMP", ""))).resolve() \
                if os.environ.get("TEMP") or os.environ.get("TMP") else None
            allowed = temp_root is not None and _is_within(resolved, temp_root)
            if not allowed:
                raise CodingError(
                    f"güvenlik: çalışma alanı dışına erişim reddedildi — "
                    f"{resolved} (kök: {root})"
                )
        if must_exist and not resolved.exists():
            raise CodingError(f"bulunamadı: {resolved}")
        return resolved

    def relative(self, path: Any) -> str:
        """Yolu çalışma alanına göreli metne çevirir."""
        try:
            return str(Path(path).resolve().relative_to(self.workspace))
        except (ValueError, OSError):
            return str(path)

    # ─── Keşif: Proje Haritası ──────────────────────────────────────────

    def workspace_map(
        self,
        max_files: int = 400,
        max_depth: int = 6,
        exclude: Optional[Sequence[str]] = None,
        path: str = "",
    ) -> Dict[str, Any]:
        """
        Projenin yapısını çıkarır: tür, dosya ağacı, dil dağılımı, giriş
        noktaları ve önemli dosyalar.

        Ajanın **ilk** çağırması gereken araçtır: tüm depoyu okumak yerine
        nerede ne olduğunu öğrenir.

        Args:
            max_files: Listelenecek maksimum dosya
            max_depth: Dizin derinliği sınırı
            exclude: Ek dışlama kalıpları
            path: Alt dizin (boş = tüm çalışma alanı)

        Returns:
            {"root", "project_type", "file_count", "languages", "tree",
             "entry_points", "key_files", "size_mb"}
        """
        self._counter["maps"] += 1
        root = self.resolve(path) if path else self.workspace
        patterns = list(DEFAULT_EXCLUDES) + list(exclude or [])

        files: List[FileInfo] = []
        language_counts: Dict[str, int] = {}
        total_bytes = 0
        directory_counts: Dict[str, int] = {}

        for file_path in self._walk(root, patterns, max_depth):
            if len(files) >= max_files:
                break
            try:
                stat = file_path.stat()
            except OSError:
                continue
            extension = file_path.suffix.lower()
            language = LANGUAGE_BY_EXTENSION.get(extension, "other")
            relative = self.relative(file_path)
            files.append(FileInfo(
                path=str(file_path), relative=relative, language=language,
                size=stat.st_size, lines=0, modified=stat.st_mtime,
            ))
            language_counts[language] = language_counts.get(language, 0) + 1
            total_bytes += stat.st_size
            top = relative.split(os.sep)[0] if os.sep in relative else "(kök)"
            directory_counts[top] = directory_counts.get(top, 0) + 1

        project_type, project_label, key_files = self._detect_project(root)

        return {
            "root": str(root),
            "project_type": project_type,
            "project_label": project_label,
            "file_count": len(files),
            "total_files_scanned": len(files),
            "size_mb": round(total_bytes / (1024 ** 2), 2),
            "languages": dict(sorted(language_counts.items(), key=lambda item: -item[1])),
            "top_directories": dict(sorted(directory_counts.items(),
                                           key=lambda item: -item[1])[:20]),
            "key_files": key_files,
            "entry_points": self._find_entry_points(root, files),
            "files": [info.to_dict() for info in files[:max_files]],
            "note": (
                f"{len(files)} dosya listelendi"
                + (f" (sınır {max_files})" if len(files) >= max_files else "")
            ),
        }

    def _walk(self, root: Path, patterns: Sequence[str],
              max_depth: int) -> Iterable[Path]:
        """Dışlamalara uyan dosyaları dolaşır (deterministik sıra)."""
        root_depth = len(root.parts)
        for current, directories, filenames in os.walk(root):
            current_path = Path(current)
            depth = len(current_path.parts) - root_depth
            if depth >= max_depth:
                directories[:] = []
                continue

            directories[:] = [
                name for name in sorted(directories)
                if not _excluded(name, patterns)
            ]
            for filename in sorted(filenames):
                if _excluded(filename, patterns):
                    continue
                if Path(filename).suffix.lower() in BINARY_EXTENSIONS:
                    # İkili dosyaları haritada göstermek yararlı olabilir ama
                    # içerikleri okunmaz; yalnızca bilinen proje dosyalarını al
                    if Path(filename).suffix.lower() not in (".blend", ".glb", ".gltf"):
                        continue
                yield current_path / filename

    def _detect_project(self, root: Path) -> Tuple[str, str, List[str]]:
        """Proje türünü ve anahtar dosyaları belirler."""
        key_files: List[str] = []
        project_type = "unknown"
        label = "Bilinmeyen proje türü"

        for signature, kind, description in PROJECT_SIGNATURES:
            if "*" in signature:
                matches = list(root.glob(signature))
            else:
                candidate = root / signature
                matches = [candidate] if candidate.exists() else []
            if matches:
                key_files.append(self.relative(matches[0]))
                if project_type == "unknown":
                    project_type = kind
                    label = description

        # Yapılandırma dosyaları
        for name in ("README.md", "README.rst", "AGENTS.md", "CLAUDE.md",
                     ".editorconfig", "config.yaml", "config.yaml.example"):
            candidate = root / name
            if candidate.exists():
                key_files.append(self.relative(candidate))

        return project_type, label, key_files[:20]

    def _find_entry_points(self, root: Path, files: Sequence[FileInfo]) -> List[str]:
        """
        Olası giriş noktalarını bulur (programın başladığı dosyalar).

        Ölçütler: `main` bloğu, `if __name__ == "__main__"`, `project.godot`,
        `manage.py`, `app.py`, `main.py`, `index.js`, `.exe` hedefleri.
        """
        entries: List[str] = []
        names = {"main.py", "app.py", "manage.py", "__main__.py", "cli.py",
                 "main.js", "index.js", "server.js", "main.ts", "index.ts",
                 "project.godot", "main.gd", "Main.cs"}

        for info in files:
            name = Path(info.relative).name
            if name in names:
                entries.append(info.relative)
                continue
            # İçerik ipucu (yalnızca küçük dosyalarda)
            if info.language == "python" and info.size < 20000:
                try:
                    text = (root / info.relative).read_text(encoding="utf-8", errors="ignore")
                    if '__name__ == "__main__"' in text or "__name__ == '__main__'" in text:
                        entries.append(info.relative)
                except OSError:
                    continue
        return entries[:15]

    # ─── Okuma ──────────────────────────────────────────────────────────

    def read_file(self, path: str, start_line: int = 0, max_lines: int = 400,
                  with_line_numbers: bool = True) -> Dict[str, Any]:
        """
        Dosyayı satır numaralarıyla okur.

        Args:
            path: Dosya yolu (çalışma alanına göreli)
            start_line: Başlangıç satırı (0 = baştan, 1 tabanlı)
            max_lines: Okunacak maksimum satır
            with_line_numbers: Satır numaralarını ekle

        Returns:
            {"path", "language", "total_lines", "start_line", "end_line",
             "content", "truncated", "size"}
        """
        target = self.resolve(path, must_exist=True)
        if target.is_dir():
            raise CodingError(f"bu bir dizin, dosya değil: {path}")
        if target.suffix.lower() in BINARY_EXTENSIONS:
            stat = target.stat()
            return {
                "path": self.relative(target), "language": "binary",
                "total_lines": 0, "start_line": 0, "end_line": 0,
                "content": f"[ikili dosya — {stat.st_size} bayt, içerik okunmadı]",
                "truncated": False, "size": stat.st_size,
            }

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise CodingError(f"dosya okunamadı ({path}): {exc}") from exc

        self._counter["files_read"] += 1

        # ⚠ BÜYÜK DOSYA: tüm dosyayı splitlines() ile belleğe ALMA.
        # 10 M satır ≈ 250 MB metin + ~1-2 GB Python nesnesi; indeksli pencere
        # okuma 11 ms'de aynı satırı verir (ölçüldü).
        try:
            from neurocore.agent import bigfile

            if bigfile.buyuk_mu(target):
                pencere = bigfile.pencere(target, bas=start_line or 1,
                                          adet=max_lines,
                                          numarali=with_line_numbers)
                pencere["path"] = self.relative(target)
                return pencere
        except Exception as exc:
            logger.debug(f"büyük dosya penceresi kullanılamadı: {exc}")

        all_lines = text.splitlines()
        total = len(all_lines)
        begin = max(0, int(start_line) - 1) if start_line else 0
        end = min(total, begin + max(1, int(max_lines)))
        chunk = all_lines[begin:end]

        if with_line_numbers:
            width = len(str(end))
            content = "\n".join(
                f"{index + 1:>{width}}| {line}" for index, line in enumerate(chunk, start=begin)
            )
        else:
            content = "\n".join(chunk)

        return {
            "path": self.relative(target),
            "language": LANGUAGE_BY_EXTENSION.get(target.suffix.lower(), "text"),
            "total_lines": total,
            "start_line": begin + 1 if chunk else 0,
            "end_line": end,
            "content": content,
            "truncated": end < total,
            "size": len(text.encode("utf-8")),
            "note": (
                f"satır {begin + 1}-{end} / {total}"
                + (f" — devamı için start_line={end + 1}" if end < total else "")
            ),
        }

    # ─── proje kod indeksi (milyonlarca satır / binlerce dosya) ───────

    def index(self, zorla: bool = False, **_: Any) -> Dict[str, Any]:
        """
        Proje kod indeksini kurar/günceller (SQLite + FTS5).

        Semboller (def/class/func/const) ve import ilişkileri dosya başına
        **akışlı** taranır; yalnız değişen dosyalar yeniden işlenir.
        Ölçüm: 2000 dosya / 5,99 M satır → 39,8 sn ilk kurulum, 0,15 sn
        artımlı, sorgular 1-8 ms, veritabanı 13,4 MB.
        """
        from neurocore.agent import codeindex

        return codeindex.dizinle(self.workspace, zorla=zorla)

    def index_symbol(self, ad: str = "", limit: int = 40,
                     name: str = "", sembol: str = "", **_: Any) -> Dict[str, Any]:
        """Proje genelinde **sembol** arar (tanım satırları)."""
        from neurocore.agent import codeindex

        # ⚠ Model `ad` yerine `name`/`sembol` gönderebiliyor; hepsini kabul et.
        hedef = (ad or name or sembol or "").strip()
        if not hedef:
            return {"ok": False, "error": "sembol adı gerekli (ad=...)"}
        return codeindex.sembol_ara(self.workspace, hedef, sinir=limit)

    def find_references(self, ad: str = "", limit: int = 40,
                        name: str = "", **_: Any) -> Dict[str, Any]:
        """Bir adı kullanan dosyaları/ilişkileri bulur."""
        from neurocore.agent import codeindex

        hedef = (ad or name or "").strip()
        if not hedef:
            return {"ok": False, "error": "ad gerekli (ad=...)"}
        return codeindex.referanslar(self.workspace, hedef, sinir=limit)

    def callers(self, ad: str = "", limit: int = 40,
                name: str = "", **_: Any) -> Dict[str, Any]:
        """
        Çağrı grafiği: `ad` fonksiyonunu kim çağırıyor, o kimi çağırıyor.

        AST ile çıkarılır (python `ast`, GDScript yapısal tarama); regex
        tahmininden daha isabetlidir.
        """
        from neurocore.agent import codeindex

        hedef = (ad or name or "").strip()
        if not hedef:
            return {"ok": False, "error": "ad gerekli (ad=...)"}
        return codeindex.cagri_grafigi(self.workspace, hedef, sinir=limit)

    def hierarchy(self, ad: str = "", limit: int = 40,
                  name: str = "", **_: Any) -> Dict[str, Any]:
        """Sınıf hiyerarşisi: taban sınıflar + alt sınıflar (AST'ten)."""
        from neurocore.agent import codeindex

        hedef = (ad or name or "").strip()
        if not hedef:
            return {"ok": False, "error": "ad gerekli (ad=...)"}
        return codeindex.hiyerarsi(self.workspace, hedef, sinir=limit)

    def impact_map(self, ad: str, depth: int = 1,
                   limit: int = 25, **_: Any) -> Dict[str, Any]:
        """Etki haritası: çağrı grafiğinden SVG/HTML rapor üretir."""
        from neurocore.agent import codeindex

        return codeindex.etki_haritasi(self.workspace, ad, derinlik=depth,
                                       sinir=limit)

    def mro(self, ad: str = "", limit: int = 12,
            name: str = "", **_: Any) -> Dict[str, Any]:
        """Kalıtım zinciri (MRO benzeri): tabanların tabanları…"""
        from neurocore.agent import codeindex

        hedef = (ad or name or "").strip()
        if not hedef:
            return {"ok": False, "error": "ad gerekli (ad=...)"}
        return codeindex.mro(self.workspace, hedef, sinir=limit)

    def types(self, ad: str = "", limit: int = 40,
              name: str = "", **_: Any) -> Dict[str, Any]:
        """Tip çıkarımı: parametre/dönüş tipleri + atama çıkarımları."""
        from neurocore.agent import codeindex

        hedef = (ad or name or "").strip()
        if not hedef:
            return {"ok": False, "error": "ad gerekli (ad=...)"}
        return codeindex.tipler(self.workspace, hedef, sinir=limit)

    def related(self, path: str = "", limit: int = 30,
                dosya: str = "", **_: Any) -> Dict[str, Any]:
        """Dosyanın import ettikleri + onu kullananlar + sembolleri."""
        from neurocore.agent import codeindex

        hedef = (path or dosya or "").strip()
        if not hedef:
            return {"ok": False, "error": "path gerekli (path=...)"}
        return codeindex.iliskili_dosyalar(self.workspace, hedef, sinir=limit)

    def map(self, **_: Any) -> Dict[str, Any]:
        """İndeksten proje haritası: dil dağılımı, en büyük dosyalar."""
        from neurocore.agent import codeindex

        return codeindex.proje_haritasi(self.workspace)

    def outline(self, path: str, limit: int = 400, **_: Any) -> Dict[str, Any]:
        """
        Dosyanın **yapısal haritasını** çıkarır: sınıf/fonksiyon/başlık satırları.

        ⚠ 10 milyon satırlık dosyada bile dosya belleğe alınmaz; tek geçişte
        akıtılır ve yalnız yapı satırları döner. Ajan önce haritayı görür,
        sonra ilgilendiği aralığı `code.read_file` ile okur.

        Args:
            path: Dosya yolu (çalışma alanına göreli)
            limit: En fazla kaç yapı girişi

        Returns:
            {"path", "language", "total_lines", "entries": [{line, text}],
             "truncated", "note"}
        """
        self._counter["outlines"] = self._counter.get("outlines", 0) + 1
        target = self.resolve(path, must_exist=True)
        from neurocore.agent import bigfile

        sonuc = bigfile.harita(target, sinir=max(10, int(limit)))
        sonuc["path"] = self.relative(target)
        return sonuc

    def read_many(self, paths: Sequence[str], max_lines_each: int = 200) -> Dict[str, Any]:
        """Birden çok dosyayı okur (isteme toplu bağlam için)."""
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        for path in paths:
            try:
                results.append(self.read_file(path, max_lines=max_lines_each))
            except CodingError as exc:
                errors.append({"path": str(path), "error": str(exc)})
        return {"files": results, "errors": errors, "count": len(results)}

    def list_directory(self, path: str = "", recursive: bool = False,
                       max_items: int = 300) -> Dict[str, Any]:
        """
        Dizini listeler (dosya boyutu ve türüyle).

        Args:
            path: Dizin (boş = çalışma alanı kökü)
            recursive: Alt dizinlere in
            max_items: Maksimum girdi
        """
        target = self.resolve(path) if path else self.workspace
        if not target.is_dir():
            raise CodingError(f"dizin değil: {path}")

        items: List[Dict[str, Any]] = []
        if recursive:
            iterator = self._walk(target, DEFAULT_EXCLUDES, max_depth=8)
        else:
            iterator = (item for item in sorted(target.iterdir())
                        if not _excluded(item.name, DEFAULT_EXCLUDES))

        for item in iterator:
            if len(items) >= max_items:
                break
            try:
                stat = item.stat()
            except OSError:
                continue
            items.append({
                "name": item.name,
                "relative": self.relative(item),
                "type": "dir" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else 0,
                "language": LANGUAGE_BY_EXTENSION.get(item.suffix.lower(), "")
                if item.is_file() else "",
                "modified": stat.st_mtime,
            })
        return {"path": self.relative(target), "items": items, "count": len(items)}

    # ─── Arama ──────────────────────────────────────────────────────────

    def search_text(self, pattern: str, path: str = "", *,
                    file_glob: str = "", max_results: int = 60,
                    case_sensitive: bool = False, regex: bool = False,
                    context_lines: int = 0) -> Dict[str, Any]:
        """
        Dosya içeriklerinde metin/regex arar.

        Args:
            pattern: Aranacak metin veya regex
            path: Kök dizin (boş = çalışma alanı)
            file_glob: Dosya filtresi (ör. `"*.py"`, `"*.gd"`)
            max_results: Maksimum sonuç
            case_sensitive: Büyük/küçük harf duyarlı
            regex: `pattern` düzenli ifade mi?
            context_lines: Her sonuç için bağlam satırı

        Returns:
            {"pattern", "hits": [...], "count", "files_searched", "truncated"}
        """
        self._counter["searches"] += 1
        root = self.resolve(path) if path else self.workspace

        try:
            compiled = re.compile(pattern if regex else re.escape(pattern),
                                 0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            raise CodingError(f"geçersiz regex: {exc}") from exc

        hits: List[SearchHit] = []
        files_searched = 0
        truncated = False

        for file_path in self._walk(root, DEFAULT_EXCLUDES, max_depth=12):
            if file_glob and not fnmatch.fnmatch(file_path.name, file_glob):
                continue
            if file_path.stat().st_size > 4 * 1024 * 1024:
                # ⚠ ÖNCE: bu dosyalar tamamen ATLANIYORDU (10 M satırlık dosya
                # aramada görünmüyordu). Şimdi akışlı taranır (sabit bellek).
                try:
                    from neurocore.agent import bigfile

                    sonuc = bigfile.ara(file_path, pattern, regex=regex,
                                        sinir=max_results)
                    files_searched += 1
                    for h in sonuc.get("hits", []):
                        hits.append(SearchHit(file=self.relative(file_path),
                                              line=int(h["line"]),
                                              text=str(h["text"])))
                        if len(hits) >= max_results:
                            truncated = True
                            break
                except Exception as exc:
                    logger.debug(f"akışlı arama başarısız: {exc}")
                continue
            files_searched += 1
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, MemoryError):
                continue
            if pattern not in text and not regex:
                # Hızlı eleme (büyük harf duyarlılığı için iki deneme)
                if not case_sensitive and pattern.lower() not in text.lower():
                    continue

            relative = self.relative(file_path)
            for number, line in enumerate(text.splitlines(), start=1):
                match = compiled.search(line)
                if not match:
                    continue
                hits.append(SearchHit(file=relative, line=number,
                                      text=line, column=match.start()))
                if len(hits) >= max_results:
                    truncated = True
                    break
            if truncated:
                break

        return {
            "pattern": pattern,
            "hits": [hit.to_dict() for hit in hits],
            "count": len(hits),
            "files_searched": files_searched,
            "truncated": truncated,
            "note": (f"{len(hits)} sonuç, {files_searched} dosya tarandı"
                     + (" (sınır doldu)" if truncated else "")),
        }

    def find_files(self, pattern: str, path: str = "",
                   max_results: int = 100) -> Dict[str, Any]:
        """
        Dosya **adına** göre arar (glob).

        Dışlama denetimi **çalışma alanına göreli** yol üzerinden yapılır.
        Mutlak yol kullanılırsa çalışma alanının kendi üst dizinleri
        (ör. geçici dizinlerdeki `Temp`, ya da `C:\\build\\...` gibi bir
        konumdaki `build`) yanlışlıkla eşleşir ve **hiçbir sonuç dönmez**.
        Bu, sessiz bir hatadır: araç boş liste döndürür ve ajan "dosya yok"
        sanır.

        Args:
            pattern: Glob kalıbı (ör. `"*.gd"`, `"test_*.py"`, `"**/core/*.py"`)
            path: Kök dizin (boş = çalışma alanı)
            max_results: Maksimum sonuç
        """
        root = self.resolve(path) if path else self.workspace
        matches: List[Dict[str, Any]] = []

        try:
            for file_path in root.rglob(pattern):
                if len(matches) >= max_results:
                    break
                if not file_path.is_file():
                    continue
                # Dışlamayı köke göreli parçalar üzerinden denetle
                try:
                    relative_parts = file_path.relative_to(root).parts
                except ValueError:
                    relative_parts = file_path.parts
                if any(_excluded(part, DEFAULT_EXCLUDES)
                       for part in relative_parts[:-1]):
                    continue
                if _excluded(file_path.name, DEFAULT_EXCLUDES):
                    continue
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                matches.append({
                    "relative": self.relative(file_path),
                    "name": file_path.name,
                    "size": stat.st_size,
                    "language": LANGUAGE_BY_EXTENSION.get(file_path.suffix.lower(), ""),
                })
        except (OSError, re.error) as exc:
            raise CodingError(f"arama hatası: {exc}") from exc

        return {"pattern": pattern, "matches": matches, "count": len(matches)}

    # ─── Sembol Analizi ─────────────────────────────────────────────────

    def list_symbols(self, path: str = "", max_symbols: int = 500,
                     kinds: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        Koddaki sınıf/fonksiyon/değişkenleri listeler.

        Python dosyaları `ast` ile **kesin** olarak ayrıştırılır. Diğer diller
        düzenli ifadeyle taranır ve sonuçlar `approximate: true` işaretlenir —
        bu dürüstlük önemlidir çünkü ajan bu bilgiye göre düzenleme yapar.

        Args:
            path: Dosya veya dizin (boş = çalışma alanı)
            max_symbols: Maksimum sembol
            kinds: Yalnızca bu türler (`class`, `function`, `method`, ...)

        Returns:
            {"symbols": [...], "count", "by_kind", "by_file", "approximate_files"}
        """
        target = self.resolve(path) if path else self.workspace
        symbols: List[Symbol] = []
        approximate_files: List[str] = []
        wanted = {str(kind) for kind in (kinds or [])}

        files: List[Path] = []
        if target.is_file():
            files = [target]
        else:
            for file_path in self._walk(target, DEFAULT_EXCLUDES, max_depth=8):
                if file_path.suffix.lower() in (".py", ".pyi", ".gd", ".js", ".ts",
                                                ".jsx", ".tsx", ".cs", ".lua", ".rs",
                                                ".go", ".java", ".kt", ".cpp", ".c"):
                    files.append(file_path)
                if len(files) >= 300:
                    break

        for file_path in files:
            if len(symbols) >= max_symbols:
                break
            language = LANGUAGE_BY_EXTENSION.get(file_path.suffix.lower(), "")
            relative = self.relative(file_path)
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            if language == "python":
                extracted = self._symbols_python(text, relative)
            else:
                extracted = self._symbols_regex(text, relative, language)
                if extracted:
                    approximate_files.append(relative)

            for symbol in extracted:
                if wanted and symbol.kind not in wanted:
                    continue
                symbols.append(symbol)
                if len(symbols) >= max_symbols:
                    break

        self._counter["symbols_found"] += len(symbols)
        by_kind: Dict[str, int] = {}
        by_file: Dict[str, int] = {}
        for symbol in symbols:
            by_kind[symbol.kind] = by_kind.get(symbol.kind, 0) + 1
            by_file[symbol.file] = by_file.get(symbol.file, 0) + 1

        return {
            "symbols": [symbol.to_dict() for symbol in symbols],
            "count": len(symbols),
            "by_kind": dict(sorted(by_kind.items(), key=lambda item: -item[1])),
            "by_file": dict(sorted(by_file.items(), key=lambda item: -item[1])[:30]),
            "approximate_files": approximate_files[:20],
            "note": ("Python dosyaları ast ile kesin ayrıştırıldı; diğer diller "
                     "regex ile yaklaşık tarandı"
                     if approximate_files else "tüm semboller kesin ayrıştırıldı"),
        }

    def _symbols_python(self, text: str, relative: str) -> List[Symbol]:
        """Python kaynağından `ast` ile kesin sembol çıkarımı."""
        symbols: List[Symbol] = []
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            logger.debug(f"Python ayrıştırma hatası ({relative}): {exc}")
            return symbols

        def visit(node: ast.AST, parent: str = "") -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    symbols.append(Symbol(
                        name=child.name, kind="class", file=relative,
                        line=child.lineno, column=child.col_offset,
                        end_line=getattr(child, "end_lineno", 0) or 0,
                        docstring=ast.get_docstring(child) or "",
                    ))
                    visit(child, child.name)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    arguments = [argument.arg for argument in child.args.args]
                    if child.args.vararg:
                        arguments.append(f"*{child.args.vararg.arg}")
                    if child.args.kwarg:
                        arguments.append(f"**{child.args.kwarg.arg}")
                    prefix = "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
                    symbols.append(Symbol(
                        name=child.name,
                        kind="method" if parent else "function",
                        file=relative, line=child.lineno,
                        column=child.col_offset,
                        end_line=getattr(child, "end_lineno", 0) or 0,
                        parent=parent,
                        signature=f"{prefix} {child.name}({', '.join(arguments)})",
                        docstring=ast.get_docstring(child) or "",
                    ))
                    visit(child, parent)
                elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                    targets = child.targets if isinstance(child, ast.Assign) \
                        else [child.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            symbols.append(Symbol(
                                name=target.id, kind="variable", file=relative,
                                line=child.lineno, column=child.col_offset,
                                parent=parent,
                            ))
                elif isinstance(child, (ast.Import, ast.ImportFrom)):
                    modules = [alias.name for alias in child.names]
                    symbols.append(Symbol(
                        name=", ".join(modules[:4]), kind="import", file=relative,
                        line=child.lineno,
                    ))
                else:
                    visit(child, parent)

        visit(tree)
        return symbols

    def _symbols_regex(self, text: str, relative: str, language: str) -> List[Symbol]:
        """Python dışı diller için regex tabanlı yaklaşık sembol taraması."""
        patterns: List[Tuple[str, str]] = [
            (r"^\s*class\s+([A-Za-z_]\w*)", "class"),
            (r"^\s*(?:public|private|protected|static|async|export|\s)*"
             r"func(?:tion)?\s+([A-Za-z_]\w*)", "function"),
            (r"^\s*(?:async\s+)?function\s+([A-Za-z_]\w*)", "function"),
            (r"^\s*def\s+([A-Za-z_]\w*)", "function"),
            (r"^\s*const\s+([A-Za-z_]\w*)\s*=", "const"),
            (r"^\s*(?:public|private|protected)?\s*(?:static\s+)?"
             r"(?:[A-Za-z_<>\[\],\s]+)\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*\{", "method"),
            (r"^\s*fn\s+([A-Za-z_]\w*)", "function"),
            (r"^\s*(?:export\s+)?(?:async\s+)?([A-Za-z_]\w*)\s*=\s*(?:\([^)]*\)|"
             r"[A-Za-z_]\w*)\s*=>", "function"),
        ]
        symbols: List[Symbol] = []
        seen: Set[Tuple[str, int]] = set()
        lines = text.splitlines()

        for pattern, kind in patterns:
            try:
                compiled = re.compile(pattern)
            except re.error:
                continue
            for number, line in enumerate(lines, start=1):
                match = compiled.match(line)
                if not match:
                    continue
                name = match.group(1)
                key = (name, number)
                if key in seen or not name or name in ("if", "for", "while", "return"):
                    continue
                seen.add(key)
                symbols.append(Symbol(
                    name=name, kind=kind, file=relative, line=number,
                    column=match.start(1), approximate=True,
                    signature=line.strip()[:120],
                ))
        return symbols

    def find_symbol(self, name: str, kinds: Optional[Sequence[str]] = None,
                    max_results: int = 25) -> Dict[str, Any]:
        """
        Ada göre sembol arar (tanım yerini bulmak için).

        Ajanın "bu fonksiyon nerede tanımlı?" sorusunu yanıtlar.
        """
        outcome = self.list_symbols(max_symbols=3000, kinds=kinds)
        needle = name.lower()
        matches = [
            symbol for symbol in outcome["symbols"]
            if needle in str(symbol["name"]).lower()
            or needle in str(symbol.get("qualified", "")).lower()
        ]
        definitions = [symbol for symbol in matches if symbol["kind"] != "import"]
        usages = self.search_text(name, max_results=15, regex=False)

        return {
            "name": name,
            "definitions": definitions[:max_results],
            "definition_count": len(definitions),
            "approximate": any(symbol.get("approximate") for symbol in definitions),
            "usages": usages.get("hits", [])[:15],
            "usage_count": usages.get("count", 0),
        }

    # ─── Düzenleme ──────────────────────────────────────────────────────

    def edit_file(self, path: str, old_text: str, new_text: str,
                  *, replace_all: bool = False, create_backup: bool = True,
                  check_syntax: bool = True) -> Dict[str, Any]:
        """
        Dosyada **cerrahi** değişiklik yapar (eski metni yenisiyle değiştirir).

        Neden tam dosya yazmak yerine? Modelin tüm dosyayı yeniden üretmesi
        hem pahalı hem risklidir (ilgisiz satırlar kaybolur). Cerrahi
        değişiklik, niyeti açık ve doğrulanabilir kılar.

        Args:
            path: Dosya yolu
            old_text: Değiştirilecek mevcut metin (**tam eşleşme**)
            new_text: Yeni metin (boş = sil)
            replace_all: Tüm eşleşmeleri değiştir
            create_backup: Geri alınabilir yedek oluştur
            check_syntax: Değişiklikten sonra sözdizimini doğrula

        Returns:
            `EditResult.to_dict()`
        """
        target = self.resolve(path)
        result = EditResult(ok=False, path=self.relative(target))

        if target.suffix.lower() in BINARY_EXTENSIONS:
            result.error = f"ikili dosya düzenlenemez: {target.suffix}"
            return result.to_dict()

        existed = target.exists()
        try:
            original = target.read_text(encoding="utf-8") if existed else ""
        except OSError as exc:
            result.error = f"okuma hatası: {exc}"
            return result.to_dict()

        result.language = LANGUAGE_BY_EXTENSION.get(target.suffix.lower(), "text")

        if not existed and old_text:
            result.error = "dosya yok ve old_text boş değil — code.create_file kullanın"
            return result.to_dict()

        if old_text:
            occurrences = original.count(old_text)
            if occurrences == 0:
                result.error = (
                    "old_text dosyada bulunamadı — metni birebir kopyaladığınızdan "
                    "emin olun (girinti ve satır sonları dahil)"
                )
                # Yararlı ipucu: benzer satır var mı?
                hint = self._closest_line(original, old_text)
                if hint:
                    result.error += f" | en yakın satır {hint[0]}: {hint[1][:80]}"
                return result.to_dict()
            if occurrences > 1 and not replace_all:
                result.error = (
                    f"old_text dosyada {occurrences} kez geçiyor — daha fazla bağlam "
                    "ekleyin veya replace_all=true kullanın"
                )
                return result.to_dict()
            updated = original.replace(old_text, new_text) if replace_all \
                else original.replace(old_text, new_text, 1)
            result.replaced = occurrences if replace_all else 1
        else:
            updated = new_text
            result.replaced = 1

        if updated == original:
            result.ok = True
            result.error = ""
            result.diff = "(değişiklik yok)"
            return result.to_dict()

        # ── Yedek ──
        if create_backup and existed:
            backup_id = hashlib.blake2b(
                f"{target}:{time.time()}".encode("utf-8"), digest_size=6
            ).hexdigest()
            self._backups[backup_id] = {
                "path": str(target), "content": original, "at": time.time(),
            }
            result.backup_id = backup_id
            if len(self._backups) > self.MAX_BACKUPS:
                oldest = min(self._backups.items(), key=lambda item: item[1]["at"])[0]
                self._backups.pop(oldest, None)

        # ── Yazma (atomik) ──
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".nctmp")
            temporary.write_text(updated, encoding="utf-8", newline="")
            os.replace(temporary, target)
        except OSError as exc:
            result.error = f"yazma hatası: {exc}"
            self._counter["errors"] += 1
            return result.to_dict()

        self._counter["edits"] += 1
        self._counter["files_written"] += 1

        # ── Fark ──
        result.diff = self._unified_diff(original, updated, result.path)

        # ── Sözdizimi doğrulaması ──
        if check_syntax and result.language == "python":
            ok, error = self._check_python_syntax(updated)
            result.syntax_ok = ok
            result.syntax_error = error
            if not ok:
                # Sözdizimi bozuksa geri al (kullanıcıyı kırık dosyayla bırakma)
                if result.backup_id:
                    self._restore(result.backup_id)
                    result.error = (
                        f"sözdizimi hatası ve değişiklik geri alındı: {error}"
                    )
                    result.ok = False
                    return result.to_dict()

        result.ok = True
        self._emit("coding.file_edited", {
            "path": result.path, "replaced": result.replaced,
            "created": not existed, "diff_lines": result.diff.count("\n"),
        })
        logger.info(
            f"Dosya düzenlendi: {result.path} ({result.replaced} değişiklik)"
        )
        return result.to_dict()

    def create_file(self, path: str, content: str = "",
                    overwrite: bool = False) -> Dict[str, Any]:
        """
        Yeni dosya oluşturur.

        Args:
            path: Dosya yolu
            content: İçerik
            overwrite: Var olan dosyanın üzerine yaz

        Returns:
            `EditResult.to_dict()`
        """
        target = self.resolve(path)
        result = EditResult(ok=False, path=self.relative(target))

        if target.exists() and not overwrite:
            result.error = f"dosya zaten var: {result.path} (overwrite=true ile üzerine yazın)"
            return result.to_dict()

        if target.exists() and overwrite:
            return self.edit_file(path, old_text="", new_text=content,
                                  check_syntax=True)

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="")
        except OSError as exc:
            result.error = f"oluşturma hatası: {exc}"
            self._counter["errors"] += 1
            return result.to_dict()

        self._counter["files_written"] += 1
        self._counter["edits"] += 1
        result.ok = True
        result.replaced = 1
        result.language = LANGUAGE_BY_EXTENSION.get(target.suffix.lower(), "text")
        result.diff = self._unified_diff("", content, result.path)

        if result.language == "python":
            ok, error = self._check_python_syntax(content)
            result.syntax_ok = ok
            result.syntax_error = error
            if not ok:
                # Yeni dosya sözdizimi hatası içeriyorsa bunu **başarı** olarak
                # bildirmek yanıltıcıdır; dosya yazıldı ama kullanılamaz.
                result.ok = False
                result.error = (
                    f"dosya oluşturuldu ama sözdizimi hatalı: {error} — "
                    "düzeltmek için code.edit_file kullanın"
                )

        self._emit("coding.file_created", {
            "path": result.path, "bytes": len(content.encode("utf-8")),
            "syntax_ok": result.syntax_ok,
        })
        logger.info(
            f"Dosya oluşturuldu: {result.path}"
            + ("" if result.syntax_ok is not False else " (sözdizimi hatalı)")
        )
        return result.to_dict()

    def revert(self, backup_id: str = "", path: str = "") -> Dict[str, Any]:
        """
        Yapılan değişikliği geri alır.

        Args:
            backup_id: Yedek kimliği (edit sonucunda döner)
            path: Yedek kimliği yoksa dosyanın **en son** yedeğini geri al

        Returns:
            {"ok", "path", "restored_from", "error"}
        """
        if not backup_id and path:
            target = str(self.resolve(path))
            candidates = [
                (key, value) for key, value in self._backups.items()
                if value["path"] == target
            ]
            if not candidates:
                return {"ok": False, "error": f"yedek bulunamadı: {path}"}
            backup_id = max(candidates, key=lambda item: item[1]["at"])[0]

        if backup_id not in self._backups:
            return {"ok": False, "error": f"bilinmeyen yedek: {backup_id}"}

        success = self._restore(backup_id)
        self._counter["reverts"] += 1 if success else 0
        record = self._backups.get(backup_id, {})
        return {
            "ok": success,
            "path": self.relative(record.get("path", "")),
            "restored_from": backup_id,
            "error": "" if success else "geri alma başarısız",
        }

    def _restore(self, backup_id: str) -> bool:
        """Yedeği diske geri yazar."""
        record = self._backups.get(backup_id)
        if not record:
            return False
        try:
            Path(record["path"]).write_text(record["content"], encoding="utf-8", newline="")
            logger.info(f"Geri alındı: {record['path']}")
            self._emit("coding.file_reverted", {"path": record["path"]})
            return True
        except OSError as exc:
            logger.error(f"Geri alma hatası: {exc}")
            return False

    def list_backups(self) -> List[Dict[str, Any]]:
        """Geri alınabilir değişiklikler."""
        return [
            {"backup_id": key, "path": self.relative(value["path"]),
             "at": value["at"], "size": len(value["content"])}
            for key, value in sorted(self._backups.items(),
                                     key=lambda item: -item[1]["at"])
        ]

    @staticmethod
    def _unified_diff(before: str, after: str, name: str, context: int = 3) -> str:
        """İki metin arasındaki birleşik fark (okunabilir)."""
        diff = difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile=f"a/{name}", tofile=f"b/{name}", lineterm="", n=context,
        )
        text = "\n".join(diff)
        return text[:6000] + ("\n… (fark kısaltıldı)" if len(text) > 6000 else "")

    @staticmethod
    def _closest_line(text: str, needle: str) -> Optional[Tuple[int, str]]:
        """Aranan metne en benzeyen satırı bulur (hata ipucu için)."""
        first_line = needle.strip().splitlines()[0] if needle.strip() else ""
        if not first_line:
            return None
        best: Optional[Tuple[int, str, float]] = None
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            ratio = difflib.SequenceMatcher(None, line.strip(), first_line).ratio()
            if ratio > 0.6 and (best is None or ratio > best[2]):
                best = (number, line, ratio)
        return (best[0], best[1]) if best else None

    def _check_python_syntax(self, source: str) -> Tuple[bool, str]:
        """Python kaynağının sözdizimini doğrular."""
        try:
            ast.parse(source)
            return True, ""
        except SyntaxError as exc:
            return False, f"satır {exc.lineno}: {exc.msg}"

    def check_syntax(self, path: str) -> Dict[str, Any]:
        """
        Bir dosyanın sözdizimini doğrular (Python: `ast`, JSON/YAML: ayrıştırıcı).

        Returns:
            {"ok", "path", "language", "error", "lines"}
        """
        target = self.resolve(path, must_exist=True)
        language = LANGUAGE_BY_EXTENSION.get(target.suffix.lower(), "text")
        result: Dict[str, Any] = {"ok": True, "path": self.relative(target),
                                  "language": language, "error": ""}
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"ok": False, "path": self.relative(target), "error": str(exc)}
        result["lines"] = len(text.splitlines())

        if language == "python":
            ok, error = self._check_python_syntax(text)
            result.update({"ok": ok, "error": error})
        elif language == "json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                result.update({"ok": False,
                               "error": f"satır {exc.lineno}, sütun {exc.colno}: {exc.msg}"})
        elif language in ("yaml", "toml"):
            try:
                if language == "yaml":
                    import yaml  # type: ignore
                    yaml.safe_load(text)
                else:
                    import tomllib  # type: ignore
                    tomllib.loads(text)
            except ImportError:
                result["note"] = f"{language} ayrıştırıcısı kurulu değil — atlandı"
            except Exception as exc:
                result.update({"ok": False, "error": str(exc)[:200]})
        else:
            # Diğer diller: yalnızca parantez/köşeli parantez dengesi
            balance = _bracket_balance(text)
            if balance:
                result.update({"ok": False, "error": balance})
            result["note"] = "yalnızca parantez dengesi denetlendi (dil ayrıştırıcısı yok)"
        return result

    # ─── Komut ve Test Çalıştırma ───────────────────────────────────────

    def run_command(self, command: str, cwd: str = "", timeout: float = 120.0,
                    env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Çalışma alanında komut çalıştırır.

        `os_layer` enjekte edilmişse güvenlik denetimi (yıkıcı komut engeli,
        kum havuzu, denetim kaydı) **oradan** geçer. Aksi hâlde burada
        temel bir yıkıcı komut denetimi uygulanır.

        Args:
            command: Komut satırı
            cwd: Çalışma dizini (boş = çalışma alanı kökü)
            timeout: Saniye
            env: Ek ortam değişkenleri

        Returns:
            {"ok", "command", "returncode", "stdout", "stderr", "duration_ms",
             "timed_out", "blocked"}
        """
        self._counter["commands"] += 1
        workdir = self.resolve(cwd) if cwd else self.workspace

        # ── Güvenlik: kendi denetimimiz ──
        blocked, reason = detect_dangerous_command(command)
        if blocked and self._os_layer is None:
            return {
                "ok": False, "command": command, "returncode": -1,
                "stdout": "", "stderr": f"engellendi: {reason}", "blocked": True,
                "blocked_reason": reason, "duration_ms": 0.0, "timed_out": False,
            }

        # ── OS katmanı varsa ondan geç ──
        if self._os_layer is not None:
            try:
                outcome = self._os_layer.run(command, cwd=str(workdir))
                if isinstance(outcome, dict):
                    outcome.setdefault("command", command)
                    if outcome.get("blocked"):
                        outcome["blocked_reason"] = outcome.get("stderr", "")
                    return outcome
            except Exception as exc:
                logger.debug(f"OS katmanı komutu çalıştıramadı, yerel yola düşülüyor: {exc}")

        # ── Yerel çalıştırma ──
        started = time.perf_counter()
        environment = dict(os.environ)
        if env:
            environment.update({str(key): str(value) for key, value in env.items()})

        try:
            completed = subprocess.run(
                command, shell=True, cwd=str(workdir), env=environment,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=max(1.0, float(timeout)),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            duration = (time.perf_counter() - started) * 1000
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            result = {
                "ok": completed.returncode == 0,
                "command": command,
                "returncode": completed.returncode,
                "stdout": stdout[-20000:],
                "stderr": stderr[-10000:],
                "duration_ms": duration,
                "timed_out": False,
                "blocked": False,
                "truncated": len(stdout) > 20000 or len(stderr) > 10000,
                "cwd": self.relative(workdir),
            }
        except subprocess.TimeoutExpired as exc:
            duration = (time.perf_counter() - started) * 1000
            result = {
                "ok": False, "command": command, "returncode": -1,
                "stdout": (exc.stdout or b"").decode("utf-8", "replace")[-5000:]
                if isinstance(exc.stdout, bytes) else str(exc.stdout or "")[-5000:],
                "stderr": f"zaman aşımı ({timeout:.0f}s)",
                "duration_ms": duration, "timed_out": True, "blocked": False,
                "cwd": self.relative(workdir),
            }
        except OSError as exc:
            self._counter["errors"] += 1
            result = {
                "ok": False, "command": command, "returncode": -1,
                "stdout": "", "stderr": f"başlatma hatası: {exc}",
                "duration_ms": (time.perf_counter() - started) * 1000,
                "timed_out": False, "blocked": False,
            }

        self._emit("coding.command_run", {
            "command": command[:200], "ok": result["ok"],
            "returncode": result["returncode"], "duration_ms": result["duration_ms"],
        })
        return result

    def run_tests(self, target: str = "", framework: str = "",
                  timeout: float = 300.0, extra_args: str = "") -> Dict[str, Any]:
        """
        Testleri çalıştırır ve **yorumlanmış** sonuç döner.

        Framework otomatik algılanır: `pytest`, `unittest`, `npm test`,
        `cargo test`, `go test`, Godot/GUT.

        Args:
            target: Test dosyası/dizini (boş = tümü)
            framework: Zorla framework seç (`pytest`, `unittest`, ...)
            timeout: Saniye
            extra_args: Ek komut satırı argümanları

        Returns:
            {"ok", "framework", "command", "passed", "failed", "errors",
             "summary", "failures": [...], "stdout_tail", "duration_ms"}
        """
        root = self.workspace
        chosen = framework or self._detect_test_framework(root)

        if chosen == "pytest":
            command = f"{sys.executable} -m pytest -q --no-header"
            if target:
                command += f" {target}"
            if extra_args:
                command += f" {extra_args}"
        elif chosen == "unittest":
            command = self._unittest_command(target, extra_args)
        elif chosen == "npm":
            command = f"npm test {extra_args}".strip()
        elif chosen == "cargo":
            command = f"cargo test {extra_args}".strip()
        elif chosen == "go":
            command = f"go test ./... {extra_args}".strip()
        elif chosen == "godot":
            # ⚠ ÖLÇÜLDÜ: godot PATH'te olmayınca bu komut başarısız oluyordu
            # (ajan tam yolu elle yazmak zorunda kaldı). Bilinen yolları dene.
            godot_yolu = "godot"
            try:
                from neurocore.agent.envcheck import arac_durumu

                durum = arac_durumu("godot")
                if durum.get("kurulu") and durum.get("yol"):
                    godot_yolu = f'"{durum["yol"]}"'
            except Exception:
                pass
            command = (f"{godot_yolu} --headless --script "
                       f"{target or 'res://tests/run_tests.gd'} {extra_args}").strip()
        else:
            return {
                "ok": False, "error": "test framework algılanamadı",
                "framework": "unknown",
                "hint": "framework parametresiyle belirtin (pytest/unittest/npm/cargo/go)",
            }

        outcome = self.run_command(command, timeout=timeout)
        parsed = parse_test_output(
            f"{outcome.get('stdout', '')}\n{outcome.get('stderr', '')}", chosen
        )
        return {
            "ok": outcome.get("ok", False),
            "framework": chosen,
            "command": command,
            "returncode": outcome.get("returncode"),
            "duration_ms": outcome.get("duration_ms", 0.0),
            "timed_out": outcome.get("timed_out", False),
            "blocked": outcome.get("blocked", False),
            **parsed,
            "stdout_tail": str(outcome.get("stdout", ""))[-4000:],
            "stderr_tail": str(outcome.get("stderr", ""))[-2000:],
        }

    def _unittest_command(self, target: str = "", extra_args: str = "") -> str:
        """
        `unittest` komutunu **keşif/import hatası vermeyecek** biçimde kurar.

        Neden özel mantık?
            * `python -m unittest discover -s tests -t .`, `tests` dizini içe
              aktarılabilir bir paket değilse
              `ValueError: Start directory is not importable` ile çöker.
            * Tek dosyayı `python tests/test_x.py` diye çalıştırmak, dosya
              `from tests.helpers import ...` gibi **paket yolu** kullanıyorsa
              `ModuleNotFoundError` verir — çünkü betik olarak çalışan dosyada
              proje kökü `sys.path`'te olmaz.

        Bu yüzden dosya hedefleri **modül yoluyla** (`python -m unittest
        tests.test_x`) çalıştırılır; bu, proje kökünü `sys.path`'e ekler ve
        paket importlarını çalıştırır.

        Args:
            target: Test dosyası veya dizini
            extra_args: Ek argümanlar

        Returns:
            Çalıştırılacak komut satırı
        """
        python = f"{sys.executable}"
        if target:
            candidate = self.resolve(target)
            if candidate.is_file() and candidate.suffix == ".py":
                module = self._module_path(candidate)
                if module:
                    return f"{python} -m unittest {module} {extra_args}".strip()
                return f"{python} {target} {extra_args}".strip()
            if candidate.is_dir():
                return (f"{python} -m unittest discover -s {target} "
                        f"-t {target} {extra_args}").strip()

        # Hedef yok: yaygın test dizinlerini sırayla dene
        for directory in ("tests", "test", "Tests"):
            candidate = self.workspace / directory
            if not candidate.is_dir():
                continue
            top = "." if (candidate / "__init__.py").exists() else directory
            return (f"{python} -m unittest discover -s {directory} -t {top} "
                    f"{extra_args}").strip()

        # Hiç test dizini yok: kökten keşfet
        return f"{python} -m unittest discover -s . -p \"test_*.py\" {extra_args}".strip()

    def _module_path(self, path: Path) -> str:
        """
        Dosya yolunu Python modül yoluna çevirir (`tests/test_x.py` →
        `tests.test_x`).

        Kökten çalıştırılabilmesi için dosyanın ve üst dizinlerinin
        `__init__.py` içermesi ya da dosyanın kök altında bir paket yolunda
        olması gerekir. Aksi hâlde boş dize döner (betik olarak çalıştırılır).
        """
        try:
            relative = path.resolve().relative_to(self.workspace)
        except ValueError:
            return ""
        if relative.suffix != ".py":
            return ""
        parts = list(relative.parts)
        if parts and parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        if not parts or any(not part.isidentifier() for part in parts):
            return ""
        # En az bir üst dizinin paket olması gerekir (yoksa modül importu
        # kökten çalışmaz ve betik yolu daha güvenilirdir).
        parent = path.parent
        while parent != self.workspace and parent != parent.parent:
            if (parent / "__init__.py").exists():
                return ".".join(parts)
            parent = parent.parent
        return ".".join(parts) if len(parts) > 1 else ""

    def _detect_test_framework(self, root: Path) -> str:
        """
        Projede kullanılan test framework'ünü belirler.

        Not: `pytest` kuruluysa **tercih edilir** — hem `unittest` testlerini
        çalıştırır hem de çok daha okunur bir özet üretir. `unittest`'in kendi
        keşif mekanizması paket yapısına duyarlıdır ve sık çöker.
        """
        has_python_tests = (
            (root / "tests").is_dir()
            or (root / "test").is_dir()
            or bool(list(root.glob("test_*.py")))
            or bool(list(root.glob("*_test.py")))
            or (root / "pyproject.toml").exists()
            or (root / "pytest.ini").exists()
            or (root / "setup.cfg").exists()
            or (root / "tox.ini").exists()
        )
        if has_python_tests:
            try:
                import pytest  # noqa: F401
                return "pytest"
            except ImportError:
                return "unittest"
        if (root / "package.json").exists():
            return "npm"
        if (root / "Cargo.toml").exists():
            return "cargo"
        if (root / "go.mod").exists():
            return "go"
        if (root / "project.godot").exists():
            return "godot"
        return ""

    # ─── Yardımcılar ────────────────────────────────────────────────────

    def git_status(self) -> Dict[str, Any]:
        """
        Git durumunu okur (git kurulu değilse nazikçe bildirir).

        Returns:
            {"available", "branch", "changed", "untracked", "clean", "error"}
        """
        if not shutil.which("git"):
            return {"available": False, "error": "git kurulu değil",
                    "hint": "git olmadan da çalışılır; sürüm takibi devre dışı"}
        if not (self.workspace / ".git").exists():
            return {"available": False, "error": "bu dizin bir git deposu değil"}

        status = self.run_command("git status --porcelain=v1 --branch", timeout=30.0)
        if not status.get("ok"):
            return {"available": True, "error": status.get("stderr", "")[:300]}

        lines = [line for line in str(status.get("stdout", "")).splitlines() if line.strip()]
        branch = ""
        changed: List[str] = []
        untracked: List[str] = []
        for line in lines:
            if line.startswith("##"):
                branch = line[2:].strip().split("...")[0]
                continue
            if line.startswith("??"):
                untracked.append(line[3:].strip())
            else:
                changed.append(line[3:].strip())
        return {
            "available": True,
            "branch": branch,
            "changed": changed[:100],
            "untracked": untracked[:100],
            "changed_count": len(changed),
            "untracked_count": len(untracked),
            "clean": not changed and not untracked,
        }

    def git_diff(self, path: str = "", staged: bool = False,
                 max_lines: int = 400) -> Dict[str, Any]:
        """Git farkını okur."""
        if not shutil.which("git"):
            return {"available": False, "error": "git kurulu değil"}
        command = "git diff"
        if staged:
            command += " --cached"
        if path:
            command += f" -- {path}"
        outcome = self.run_command(command, timeout=30.0)
        text = str(outcome.get("stdout", ""))
        lines = text.splitlines()
        return {
            "available": True,
            "diff": "\n".join(lines[:max_lines]),
            "lines": len(lines),
            "truncated": len(lines) > max_lines,
        }

    def diagnose_error(self, text: str) -> Dict[str, Any]:
        """
        Hata çıktısını yorumlar: hangi dosya/satır, hangi tür hata, ne yapmalı.

        Ajanın bir test/komut başarısız olduğunda **doğru dosyaya** gitmesini
        sağlar; ham çıktıyı okumaktan çok daha verimlidir.

        Args:
            text: Komut çıktısı (stdout + stderr)

        Returns:
            {"errors": [{"type","message","file","line","hint"}],
             "primary": {...}, "error_count", "advice"}
        """
        errors: List[Dict[str, Any]] = []
        lines = str(text or "").splitlines()

        # Python: Traceback ve SyntaxError
        for index, line in enumerate(lines):
            stripped = line.strip()
            match = re.match(r'File "([^"]+)", line (\d+)(?:, in (.+))?', stripped)
            if match:
                errors.append({
                    "type": "python_traceback",
                    "file": self._normalize_error_path(match.group(1)),
                    "line": int(match.group(2)),
                    "function": (match.group(3) or "").strip(),
                    "message": _last_error_line(lines, index),
                    "hint": "izleme kaydındaki bu satırı okuyun ve hatayı düzeltin",
                })
            syntax = re.match(r'File "([^"]+)", line (\d+)', stripped)
            if syntax and index + 1 < len(lines) and "SyntaxError" in lines[index + 1]:
                errors.append({
                    "type": "python_syntax",
                    "file": self._normalize_error_path(syntax.group(1)),
                    "line": int(syntax.group(2)),
                    "message": lines[index + 1].strip(),
                    "hint": "sözdizimi hatası — code.check_syntax ile doğrulayın",
                })

        # Test çıktısı: FAILED / ERROR satırları
        for line in lines:
            failed = re.match(r"FAILED\s+([^\s:]+)(?:::([^\s]+))?", line.strip())
            if failed:
                errors.append({
                    "type": "test_failure",
                    "file": failed.group(1),
                    "test": failed.group(2) or "",
                    "message": line.strip()[:200],
                    "hint": "bu testi tek başına çalıştırıp çıktısını inceleyin",
                })
            assertion = re.match(r"E\s+(AssertionError.*)", line.strip())
            if assertion:
                errors.append({
                    "type": "assertion",
                    "message": assertion.group(1)[:250],
                    "hint": "beklenen ve gerçek değerleri karşılaştırın",
                })

        # Genel hata kalıpları
        generic_patterns = [
            (r"ModuleNotFoundError: No module named '([^']+)'", "missing_module",
             "eksik paket — pip install ile kurun"),
            (r"ImportError: cannot import name '([^']+)'", "import_error",
             "içe aktarma yolu yanlış veya isim değişmiş"),
            (r"AttributeError: '([^']+)' object has no attribute '([^']+)'", "attribute",
             "yöntem/alan adı yanlış veya nesne beklenen türde değil"),
            (r"TypeError: ([^\n]{0,120})", "type_error",
             "argüman sayısı/türü uyuşmuyor"),
            (r"KeyError: ([^\n]{0,80})", "key_error",
             "sözlükte bu anahtar yok — varsayılan değer kullanın"),
            (r"FileNotFoundError: [^\n]*?['\"]([^'\"]+)['\"]", "file_not_found",
             "yol yanlış veya dosya taşınmış"),
            (r"PermissionError", "permission",
             "yetki yok — farklı bir yol kullanın veya sandbox ayarını değiştirin"),
            (r"RecursionError", "recursion", "sonsuz özyineleme — taban durumu ekleyin"),
            (r"CUDA out of memory", "cuda_oom",
             "VRAM yetersiz — toplu işi küçültün (kaynak nezaketi zaten küçültür)"),
        ]
        for pattern, kind, hint in generic_patterns:
            for match in re.finditer(pattern, text or ""):
                message = match.group(0)[:250]
                if any(item.get("message") == message for item in errors):
                    continue
                errors.append({"type": kind, "message": message, "hint": hint})
                break

        primary = errors[0] if errors else {}
        advice = "hata bulunamadı — çıktı başarılı görünüyor"
        if primary:
            location = ""
            if primary.get("file"):
                location = f" ({primary['file']}"
                if primary.get("line"):
                    location += f":{primary['line']}"
                location += ")"
            advice = (
                f"{len(errors)} hata bulundu; ilki {primary.get('type')}"
                f"{location}: {primary.get('message', '')[:150]} → {primary.get('hint', '')}"
            )
        return {"errors": errors[:25], "primary": primary,
                "error_count": len(errors), "advice": advice}

    def _normalize_error_path(self, path: str) -> str:
        """Hata çıktısındaki yolu çalışma alanına göreli hale getirir."""
        try:
            candidate = Path(path)
            if candidate.is_absolute():
                return self.relative(candidate)
        except (OSError, ValueError):
            pass
        return path

    # ─── Araştırma Araçları (tur 14, Hedef C) ───────────────────────────
    #
    # ⚠ Kullanıcı hedefi: *"myAi'ye araştırma yeteneği ver — istediği
    # dilin/kütüphanenin güncel sürümünü ve API'sini internetten bulup
    # okuyabilsin"*
    #
    # Bu metotlar `tools/arastirma.py`'ye delege eder. Her arama sonucu
    # **kasaya yazılır** (`arastirma` kategorisi) → kalıcı öğrenme.

    def _arastirmaci(self):
        """`Arastirmaci` örneği (tembel yükleme)."""
        if getattr(self, "_arastirmaci_onbellek", None) is None:
            import sys as _sys

            kok = Path(__file__).resolve().parent.parent.parent
            if str(kok) not in _sys.path:
                _sys.path.insert(0, str(kok))
            from tools.arastirma import Arastirmaci

            self._arastirmaci_onbellek = Arastirmaci(kasaya=True)
        return self._arastirmaci_onbellek

    def kutuphane_ara(self, dil: str, kutuphane: str,
                      surum: str = "", **_: Any) -> Dict[str, Any]:
        """
        Resmî dokümanı bulur ve okur.

        ⚠ Sonuç kasaya yazılır → model bir kez araştırdığını bir daha
        aramaz.
        """
        s = self._arastirmaci().kutuphane_ara(dil, kutuphane, surum)
        self._emit("research.library", {"dil": dil, "ad": kutuphane,
                                        "bulundu": s.bulundu})
        return {"bulundu": s.bulundu, "url": s.url, "ozet": s.ozet[:400],
                "imzalar": s.imzalar[:25], "ornekler": s.ornekler[:3],
                "surum": s.surum, "kasaya_yazildi": s.kasaya_yazildi}

    def api_imzasi_al(self, dil: str, fonksiyon: str,
                      surum: str = "", **_: Any) -> Dict[str, Any]:
        """Bir fonksiyon/metot için imza + parametre + örnek getirir."""
        s = self._arastirmaci().api_imzasi_al(dil, fonksiyon, surum)
        self._emit("research.api", {"dil": dil, "fonksiyon": fonksiyon,
                                    "bulundu": s.bulundu})
        return {"bulundu": s.bulundu, "url": s.url,
                "imzalar": s.imzalar[:20], "ornekler": s.ornekler[:3],
                "kasaya_yazildi": s.kasaya_yazildi}

    def surum_karsilastir(self, dil: str, kutuphane: str, v1: str,
                          v2: str, **_: Any) -> Dict[str, Any]:
        """
        İki sürüm arasındaki API farkını çıkarır.

        ⚠ Kurulu sürümle eşleşiyorsa **kesin** veri (`inspect`) kullanır
        — ölçüldü: aksi hâlde elma-armut karşılaştırması olur.
        """
        s = self._arastirmaci().surum_karsilastir(dil, kutuphane, v1, v2)
        self._emit("research.version", {"dil": dil, "ad": kutuphane,
                                        "v1": v1, "v2": v2,
                                        "bulundu": s.bulundu})
        return {"bulundu": s.bulundu, "ozet": s.ozet,
                "farklar": s.imzalar[:20], "url": s.url,
                "kasaya_yazildi": s.kasaya_yazildi}

    def godot_api_baglam(self, sorgu: str, sinir: int = 24,
                         **_: Any) -> Dict[str, Any]:
        """
        Godot 4 API bağlamı üretir (retrieval).

        ⚠ Sonuç **doğrudan** kod üretimine enjekte edilebilir.
        """
        import sys as _sys

        kok = Path(__file__).resolve().parent.parent.parent
        if str(kok) not in _sys.path:
            _sys.path.insert(0, str(kok))
        from tools.godot4_baglam import baglam_uret

        b = baglam_uret(sorgu, sinir=int(sinir))
        api = [s[4:] for s in b.splitlines() if s.startswith("#   ")]
        self._emit("research.godot_api", {"sorgu": sorgu,
                                          "api": len(api)})
        return {"baglam": b, "api_sayisi": len(api),
                "api": api[:40], "sorgu": sorgu}

    # ══════════════════════════════════════════════════════════════
    # ⚠ tur 10: BLENDER VARLIK İŞLEYİCİLERİ
    # ══════════════════════════════════════════════════════════════

    def uret(self, urun: str, kalite: str = "orta", tohum: int = 42,
             **_: Any) -> Dict[str, Any]:
        """
        Blender'da 3B varlık üretir.

        ⚠ Ölçüldü (tur 1-9): 6 kategori · 12 ürün · 3 kalite kademesi.
        Yaratık/karakter için armature + weight + animasyon dahil.
        """
        import sys as _sys

        kok = Path(__file__).resolve().parent.parent.parent
        if str(kok) not in _sys.path:
            _sys.path.insert(0, str(kok))
        u = (urun or "").strip().lower()

        # ⚠ Armature'lü ürünler ayrı araçlarla üretilir.
        # ⚠ tur 10 ÖLÇÜMÜ: `yaratik_uret.uret()` bir **dataclass**
        # (`UretimSonuc`) döner, sözlük değil → `"hata" not in d`
        # `TypeError` veriyordu. Ortak sözlüğe çevrilir.
        if u in ("insan", "karakter", "human", "kadın", "erkek"):
            from tools.karakter_uret import uret as _u

            d = _u(kalite)
        elif u in ("timsah", "yaratik", "canavar", "monster", "ejder"):
            from tools.yaratik_uret import uret as _u

            _s = _u(f"{u}_{tohum:04d}")
            # Dataclass → sözlük
            if hasattr(_s, "to_dict"):
                _sd = _s.to_dict()
                d = dict(_sd.get("olcum") or {})
                # ⚠ tur 25 KESİN KÖK NEDEN: `olcum` sözlüğünde `klasor`
                # YOK — yalnız üst düzeyde `blend`/`glb` var. Eski kod
                # `d.setdefault("klasor", "")` ile **boş** ekliyordu →
                # sonraki tüm araçlar (`asset.animasyon`, `asset.gecis`,
                # `asset.onizleme`) "varlık bulunamadı" veriyordu.
                #
                # ✅ `klasor` = `blend` dosyasının **üst dizini**
                _blend_yolu = str(_sd.get("blend") or "")
                if _blend_yolu:
                    d["klasor"] = str(Path(_blend_yolu).parent)
                    d["blend"] = _blend_yolu
                    d["glb"] = str(_sd.get("glb") or "")
                    d["ad"] = str(_sd.get("ad") or "")
                else:
                    d["klasor"] = ""
                if not _sd.get("basarili"):
                    d["hata"] = _sd.get("hata") or "üretim başarısız"
            else:
                d = _s
                # ⚠ Sözlük dönen üreticilerde de `klasor` olmalı
                if d.get("basarili") and not d.get("klasor"):
                    _b = str(d.get("blend") or "")
                    if _b:
                        d["klasor"] = str(Path(_b).parent)
        else:
            from tools.varlik_uret import uret as _u

            d = _u(u, kalite, tohum=int(tohum))

        _basarili = not bool(d.get("hata"))
        self._emit("asset.uret", {"urun": u, "kalite": kalite,
                                  "basarili": _basarili})
        if not _basarili:
            return {"basarili": False, "hata": str(d.get("hata"))[:200]}
        return {
            "basarili": True,
            "urun": d.get("urun", u),
            "kalite": d.get("kalite", kalite),
            "vertex": d.get("vertex"),
            "yuz": d.get("yuz"),
            "uv": d.get("uv_ok"),
            "malzeme": d.get("malzeme"),
            "armature": d.get("armature_var", False),
            "kemik": d.get("kemik_sayisi", 0),
            "weight_orani": d.get("agirlikli_vertex", 0) / max(
                1, d.get("vertex", 1)) if d.get("armature_var") else None,
            "animasyon": d.get("animasyonlar", []),
            "boyut": d.get("boyut"),
            "klasor": d.get("klasor"),
        }

    def animasyon(self, hedef: str, tarif: str = "",
                  **_: Any) -> Dict[str, Any]:
        """
        Blender varlığına animasyon ekler.

        ⚠ Tarifler **rol tabanlı** — farklı kemik adlarına uygulanabilir
        (tur 4 dersi: `on_ayak_l` ≡ `front_leg_l`).
        """
        import sys as _sys

        kok = Path(__file__).resolve().parent.parent.parent
        if str(kok) not in _sys.path:
            _sys.path.insert(0, str(kok))
        from tools.anim_uret import TARIFLER, uret as _u

        if tarif and tarif not in TARIFLER:
            return {"basarili": False,
                    "hata": f"bilinmeyen tarif: {tarif}",
                    "gecerli": list(TARIFLER.keys())}
        d = _u(hedef)
        self._emit("asset.animasyon", {"hedef": hedef,
                                       "basarili": "hata" not in d})
        if isinstance(d.get("hata"), str) and d["hata"]:
            return {"basarili": False, "hata": d["hata"]}
        yazilan = d.get("yazilan", [])
        if tarif:
            yazilan = [y for y in yazilan if y.get("ad") == tarif]
        return {
            "basarili": bool(yazilan),
            "hedef": hedef,
            "kemik": d.get("kemik"),
            "animasyon_sayisi": len(yazilan),
            "animasyonlar": [
                {"ad": y["ad"], "kare": y["kare"],
                 "dongu": y["dongu"], "kemik": y["kemik"],
                 "keyframe": y["keyframe"]}
                for y in yazilan],
            "atlanan": d.get("atlanan", []),
        }

    def anim_onizleme(self, hedef: str, anim: str = "",
                      **_: Any) -> Dict[str, Any]:
        """
        Animasyonları doğrular (hareket ölçümü).

        ⚠ Ölçüt **fcurve değerleri** — animasyonun verisini doğrular.
        """
        import sys as _sys

        kok = Path(__file__).resolve().parent.parent.parent
        if str(kok) not in _sys.path:
            _sys.path.insert(0, str(kok))
        from tools.anim_onizleme import dogrula, render

        if anim:
            r = render(hedef, anim, kare_sayisi=7)
            if "hata" in r:
                return {"basarili": False, "hata": str(r["hata"])[:200]}
            return {"basarili": r.get("karar") == "NORMAL",
                    "mod": "tek", "hedef": hedef, "anim": anim,
                    "karar": r.get("karar"),
                    "en_buyuk_hareket": r.get("en_buyuk_hareket"),
                    "toplam_hareket": r.get("toplam_hareket"),
                    "hareketli_kemik": r.get("hareketli_kemik"),
                    "kemik": r.get("kemik"),
                    "en_aktif": sorted(r.get("hareket", {}).items(),
                                       key=lambda x: -x[1])[:5]}

        s = dogrula(hedef)
        if "hata" in s:
            return {"basarili": False, "hata": str(s["hata"])[:200]}
        self._emit("asset.anim_onizleme",
                   {"hedef": hedef, "normal": s.get("normal", 0)})
        return {"basarili": s.get("normal", 0) > 0,
                "mod": "tumu", "hedef": hedef,
                "toplam": len(s.get("animasyon", [])),
                "normal": s.get("normal", 0),
                "hareketsiz": s.get("hareketsiz", 0),
                "gecis_duz": s.get("gecis_duz", 0),
                "asiri": s.get("asiri", 0),
                "animasyonlar": s.get("animasyon", [])}

    def onizleme(self, hedef: str, cozunurluk: int = 512,
                 **_: Any) -> Dict[str, Any]:
        """
        Varlığı 4 açıdan render eder (görsel doğrulama).

        ⚠ Ölçüldü: 4/4 açı · ~3 sn · EEVEE 512².
        """
        import sys as _sys

        kok = Path(__file__).resolve().parent.parent.parent
        if str(kok) not in _sys.path:
            _sys.path.insert(0, str(kok))
        from tools.varlik_onizleme import render

        r = render(hedef, cozunurluk=int(cozunurluk))
        self._emit("asset.onizleme",
                   {"hedef": hedef, "basarili": "hata" not in r})
        if "hata" in r:
            return {"basarili": False, "hata": str(r["hata"])[:200]}
        return {
            "basarili": r.get("basarili", 0) >= 1,
            "hedef": hedef,
            "motor": r.get("motor"),
            "boyut": r.get("boyut"),
            "kamera": r.get("kamera"),
            "isik": r.get("isik"),
            "aci_sayisi": r.get("basarili", 0),
            "acilar": [
                {"aci": x.get("aci"), "dosya": x.get("dosya"),
                 "kb": x.get("kb")}
                for x in r.get("render", []) if x.get("var")],
            "klasor": r.get("klasor"),
            "sure_s": r.get("sure_s"),
        }

    def gecis(self, hedef: str, kare: int = 8,
              **_: Any) -> Dict[str, Any]:
        """
        Animasyonlar arası geçiş üretir + NLA'ya yerleştirir.

        ⚠ Ölçüldü: 7 geçiş · 3 GERÇEK (fark>0) · NLA 12 iz.
        """
        import sys as _sys

        kok = Path(__file__).resolve().parent.parent.parent
        if str(kok) not in _sys.path:
            _sys.path.insert(0, str(kok))
        from tools.anim_gecis import uret

        r = uret(hedef, gecis_kare=int(kare))
        _h = r.get("hata")
        _var = ((isinstance(_h, str) and _h)
                or (isinstance(_h, list) and _h))
        self._emit("asset.gecis", {"hedef": hedef, "basarili": not _var})
        if _var:
            return {"basarili": False, "hata": str(_h)[:200]}
        uretilen = r.get("uretilen", [])
        return {
            "basarili": bool(uretilen),
            "hedef": hedef,
            "gecis_sayisi": len(uretilen),
            "gercek_gecis": sum(1 for g in uretilen
                                if g.get("gercek_gecis")),
            "surekli_gecis": sum(1 for g in uretilen
                                 if not g.get("gercek_gecis")),
            "nla_iz": r.get("nla"),
            "toplam_action": len(r.get("tum_action", [])),
            "gecisler": [
                {"ad": g["ad"], "kaynak": g["kaynak"],
                 "kare": g["kare"], "kanal": g["kanal"],
                 "keyframe": g["keyframe"],
                 "azami_fark": g["azami_fark"],
                 "gercek": g.get("gercek_gecis")}
                for g in uretilen],
            "atlanan": r.get("atlanan", []),
        }

    def doku(self, sorgu: str = "", hedef: str = "", doku: str = "",
             dosya: str = "", kanal: str = "renk",
             **_: Any) -> Dict[str, Any]:
        """
        Doku arar / seçime sunar / uygular.

        ⚠ Üç mod:
            `sorgu`  → ara + puanla + sun
            `hedef` + `doku`   → prosedürel doku **uygula**
            `hedef` + `dosya`  → yerel görüntü doku **uygula**
        """
        import sys as _sys

        kok = Path(__file__).resolve().parent.parent.parent
        if str(kok) not in _sys.path:
            _sys.path.insert(0, str(kok))

        # ── Uygulama modu ──
        if hedef and (doku or dosya):
            from tools.doku_uygula import (uygula_goruntu,
                                           uygula_prosedurel)

            r = (uygula_goruntu(hedef, dosya, kanal) if dosya
                 else uygula_prosedurel(hedef, doku))
            self._emit("asset.doku", {"hedef": hedef,
                                      "basarili": "hata" not in r})
            if "hata" in r:
                return {"basarili": False, "hata": str(r["hata"])[:200]}
            return {"basarili": True, "mod": "uygula",
                    "hedef": hedef, "doku": r.get("doku"),
                    "kanal": r.get("kanal", "renk"),
                    "malzeme": r.get("malzeme"),
                    "dugum": r.get("dugum_toplam"),
                    "baglanti": r.get("baglanti_toplam"),
                    "uv": r.get("uv_katman"),
                    "uyarilar": r.get("uyari", [])}

        # ── Arama modu ──
        from tools.doku_bulucu import (KAYNAKLAR, PROSEDUREL, ara,
                                       tara)

        t = tara(kaydet=True)
        yerel = ara(sorgu, adet=8) if sorgu else []
        # ⚠ Prosedürel eşleşme
        q = (sorgu or "").lower()
        pros = [
            {"ad": ad, "aciklama": b["aciklama"],
             "dugum": len(b["dugumler"]), "renk": list(b["renk"]),
             "puruzluluk": b["puruzluluk"]}
            for ad, b in PROSEDUREL.items()
            if (not q or any(w in ad or w in b["aciklama"].lower()
                             for w in q.split()))]
        return {
            "basarili": True, "mod": "ara", "sorgu": sorgu,
            "yerel_kutuphane": {
                "dosya": t["bulunan"], "yeni": t["yeni"],
                "tur": t["tur"], "kategori": t["kategori"]},
            "yerel_bulunan": [
                {"ad": x["ad"], "puan": x["puan"], "tur": x["tur"],
                 "kategori": x["kategori"], "dosya": x["dosya"],
                 "boyut": [x["genislik"], x["yukseklik"]]}
                for x in yerel],
            "prosedurel": pros,
            "cevrimici_kaynaklar": [
                {"ad": s["ad"], "lisans": s["lisans"], "url": s["url"]}
                for s in KAYNAKLAR],
            "not": ("Prosedürel doku dosya gerektirmez; yerel kütüphane "
                    "kullanıcı tarafından doldurulur "
                    "(Varliklar/doku_kutuphanesi)."),
        }

    def katalog(self, kategori: str = "", ayrintili: bool = False,
                **_: Any) -> Dict[str, Any]:
        """Üretilen varlıkları listeler."""
        import sys as _sys

        kok = Path(__file__).resolve().parent.parent.parent
        if str(kok) not in _sys.path:
            _sys.path.insert(0, str(kok))
        from tools.varlik_katalogu import istatistik, listele

        v = listele(kategori=kategori or "")
        s = istatistik()
        if not ayrintili:
            v = [{"id": x.get("id"), "kategori": x.get("kategori"),
                  "kalite": x.get("kalite"), "vertex": x.get("vertex"),
                  "yuz": x.get("yuz"), "uv": x.get("uv_ok"),
                  "kemik": x.get("kemik_sayisi", 0),
                  "animasyon": len(x.get("animasyonlar", [])),
                  # ⚠ tur 17: önizleme yolları (Hedef G)
                  "onizleme_aci": x.get("onizleme_aci", 0),
                  "onizleme": x.get("onizleme", {}),
                  # ⚠ tur 24: animasyon önizlemesi (Hedef D)
                  "anim_onizleme_sayisi": x.get("anim_onizleme_sayisi", 0),
                  "anim_onizleme": x.get("anim_onizleme", {})}
                 for x in v]
        return {"basarili": True, "toplam": s["toplam"],
                "onizlemeli": s.get("onizlemeli", 0),
                "istatistik": s, "varliklar": v}

    def godot_onar(self, onek: str, sorgu: str = "", tur: int = 4,
                   model: str = "", **_: Any) -> Dict[str, Any]:
        """
        Godot 4 kodu üretir ve **gerçek derleyiciyle** onarır.

        ⚠ Ölçüldü: onarım döngüsü olmadan 0/4, döngü ile **4/4**.
        """
        import sys as _sys

        kok = Path(__file__).resolve().parent.parent.parent
        if str(kok) not in _sys.path:
            _sys.path.insert(0, str(kok))
        from tools.derleme_onarim import onar

        m = model or "models/qwen_coder_15b"
        if not Path(m).exists():
            m = "models/qwen_havuz2"
        s = onar(m, sorgu or onek[:60], onek, tur=int(tur))
        self._emit("code.godot_repair",
                   {"derlendi": s.derlendi, "tur": s.tur})
        return {"derlendi": s.derlendi, "tur": s.tur,
                "kod": s.kod,
                "hata_sayisi": len(s.hatalar),
                "son_hata": s.hatalar[-1][:200] if s.hatalar else "",
                "eksik_tanim": s.eksik_tanim, "eksik_ad": s.eksik_ad}

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    # ─── Araç Tanımları ─────────────────────────────────────────────────

    def tools(self) -> List[ToolDef]:
        """
        Kodlama araçlarının tam listesi (`ToolRegistry`'ye kaydedilmek üzere).

        Her araç `ToolContext` alır; böylece çalışma alanı ve iptal bayrağı
        çağrı anında geçerli olur.
        """
        coding = self

        def wrap(name: str, description: str, safety: Safety,
                 parameters: List[ToolParameter], returns: str = "",
                 tags: Optional[List[str]] = None,
                 method: Optional[str] = None):
            """
            Bağlam alan bir sarmalayıcı üretir.

            Args:
                method: Araç adından farklı bir Python metodu çağırmak için
                    (ör. `code.find_symbol` → `index_symbol`; sınıfta aynı adlı
                    başka bir metot varsa çakışmayı önler).
            """
            hedef = method or name.split(".")[-1].replace("-", "_")
            fonksiyon = getattr(coding, hedef, None)

            def handler(ctx: ToolContext, **kwargs: Any) -> Any:
                if ctx.cancelled():
                    raise CodingError("işlem kullanıcı tarafından iptal edildi")
                if fonksiyon is None:
                    raise CodingError(f"metot bulunamadı: {name} ({hedef})")
                return fonksiyon(**kwargs)

            return ToolDef(
                name=name, description=description, handler=handler,
                parameters=parameters, safety=safety, category="code",
                returns=returns, tags=tags or [],
            )

        from neurocore.agent.tools import ToolParameter as P

        return [
            wrap("code.workspace_map",
                 "Projenin yapısını haritalar: tür, dosya ağacı, diller, giriş "
                 "noktaları ve anahtar dosyalar. Yeni bir projede İLK çağrılacak araç.",
                 Safety.READ,
                 [P("path", description="Alt dizin (boş = kök)"),
                  P("max_files", "integer", "Listelenecek maksimum dosya", default=300),
                  P("max_depth", "integer", "Dizin derinliği", default=6)],
                 "proje haritası", ["kesif", "harita"]),

            wrap("code.read_file",
                 "Bir dosyayı satır numaralarıyla okur. Büyük dosyalarda "
                 "start_line/max_lines ile parça parça okuyun.",
                 Safety.READ,
                 [P("path", required=True, description="Dosya yolu"),
                  P("start_line", "integer", "Başlangıç satırı (1 tabanlı)"),
                  P("max_lines", "integer", "Okunacak satır sayısı", default=400)],
                 "dosya içeriği", ["okuma"]),

            wrap("code.outline",
                 "Dosyanın yapısal haritası (sınıf/fonksiyon/başlık satırları). "
                 "BÜYÜK dosyalarda (milyonlarca satır) İLK bunu çağırın, sonra "
                 "ilgili aralığı code.read_file(start_line, max_lines) ile okuyun.",
                 Safety.READ,
                 [P("path", required=True, description="Dosya yolu"),
                  P("limit", "integer", "En fazla yapı girişi", default=400)],
                 "yapı haritası", ["okuma", "harita", "büyük dosya"]),

            wrap("code.index",
                 "Proje kod indeksini kurar/günceller (semboller + import "
                 "ilişkileri, SQLite+FTS5). Büyük projelerde İLK çağrılacak "
                 "araç; artımlıdır (yalnız değişen dosyalar).",
                 Safety.READ,
                 [P("zorla", "boolean",
                    "Tüm dosyaları yeniden tara", default=False)],
                 "indeks özeti", ["indeks", "harita", "büyük proje"]),

            wrap("code.find_symbol",
                 "Proje genelinde sembol arar (sınıf/fonksiyon/sabit tanımı). "
                 "İlişkili kodu bulmak için code.related ve code.find_references "
                 "ile birlikte kullanın.",
                 Safety.READ,
                 [P("ad", required=True, description="Sembol adı (kısmi olabilir)"),
                  P("limit", "integer", "En fazla sonuç", default=40)],
                 "sembol listesi", ["arama", "sembol"],
                 method="index_symbol"),

            wrap("code.find_references",
                 "Bir adı kullanan dosyaları ve import ilişkilerini bulur "
                 "(tanım + etkiler).",
                 Safety.READ,
                 [P("ad", required=True, description="Aranacak ad"),
                  P("limit", "integer", "En fazla sonuç", default=40)],
                 "referanslar", ["arama", "ilişki"]),

            wrap("code.related",
                 "Bir dosyanın ilişkili kodlarını verir: import ettikleri, "
                 "onu kullananlar, içindeki semboller.",
                 Safety.READ,
                 [P("path", required=True, description="Dosya yolu"),
                  P("limit", "integer", "En fazla sonuç", default=30)],
                 "ilişkili dosyalar", ["ilişki", "keşif"]),

            wrap("code.callers",
                 "Çağrı grafiği: bir fonksiyonu kimler çağırıyor ve o hangi "
                 "fonksiyonları çağırıyor. Etki analizi için kullanın.",
                 Safety.READ,
                 [P("ad", required=True, description="Fonksiyon/metot adı"),
                  P("limit", "integer", "En fazla kayıt", default=40)],
                 "çağıranlar + çağrılanlar", ["çağrı", "grafik"],
                 method="callers"),

            wrap("code.hierarchy",
                 "Sınıf hiyerarşisi: bir sınıfın tabanları ve alt sınıfları "
                 "(AST'ten çıkarılır). Kalıtım etkisini görmek için kullanın.",
                 Safety.READ,
                 [P("ad", required=True, description="Sınıf adı"),
                  P("limit", "integer", "En fazla kayıt", default=40)],
                 "tabanlar + alt sınıflar", ["hiyerarşi", "kalıtım"],
                 method="hierarchy"),

            wrap("code.impact_map",
                 "Etki haritası: bir fonksiyonu değiştirince kimlerin "
                 "etkileneceğini SVG/HTML grafik olarak üretir "
                 "(.neurocore/raporlar/etki-<ad>.html).",
                 Safety.WRITE,
                 [P("ad", required=True, description="Sembol adı"),
                  P("depth", "integer", "Derinlik", default=1),
                  P("limit", "integer", "Düğüm sınırı", default=25)],
                 "html raporu + özet", ["etki", "grafik"],
                 method="impact_map"),

            wrap("code.mro",
                 "Kalıtım zinciri: bir sınıfın tabanları, tabanlarının "
                 "tabanları (MRO benzeri sıralı liste).",
                 Safety.READ,
                 [P("ad", required=True, description="Sınıf adı"),
                  P("limit", "integer", "En fazla derinlik", default=12)],
                 "zincir", ["kalıtım", "mro"], method="mro"),

            wrap("code.types",
                 "Tip çıkarımı: fonksiyon parametre/dönüş tipleri ve "
                 "`x = Sinif()` atamalarından çıkarılan tipler.",
                 Safety.READ,
                 [P("ad", required=True, description="Sembol veya tip adı"),
                  P("limit", "integer", "En fazla kayıt", default=40)],
                 "tip kayıtları", ["tip", "çıkarım"], method="types"),

            wrap("code.map",
                 "İndeksten proje haritası: dil dağılımı, en büyük dosyalar, "
                 "sembol yoğunluğu. Önce code.index çağrılmalı.",
                 Safety.READ, [],
                 "proje haritası", ["harita", "özet"]),

            wrap("code.read_many",
                 "Birden çok dosyayı tek çağrıda okur (ilgili dosyaları toplu görmek için).",
                 Safety.READ,
                 [P("paths", "array", "Dosya yolları listesi", required=True)],
                 "dosya içerikleri", ["okuma", "toplu"]),

            wrap("code.list_directory",
                 "Dizini listeler (tür, boyut, dil).",
                 Safety.READ,
                 [P("path", description="Dizin (boş = kök)"),
                  P("recursive", "boolean", "Alt dizinlere in", default=False)],
                 "dizin içeriği", ["kesif"]),

            wrap("code.search_text",
                 "Dosya içeriklerinde metin veya regex arar. Kod tabanında bir "
                 "sembolün nerede kullanıldığını bulmak için kullanın.",
                 Safety.READ,
                 [P("pattern", required=True, description="Aranacak metin veya regex"),
                  P("path", description="Kök dizin (boş = çalışma alanı)"),
                  P("file_glob", description="Dosya filtresi, ör. *.py"),
                  P("regex", "boolean", "pattern regex mi?", default=False),
                  P("case_sensitive", "boolean", "Büyük/küçük harf duyarlı", default=False)],
                 "eşleşmeler", ["arama"]),

            wrap("code.find_files",
                 "Dosya adına göre arar (glob deseni).",
                 Safety.READ,
                 [P("pattern", required=True, description="Glob, ör. **/*.gd"),
                  P("path", description="Kök dizin")],
                 "dosya listesi", ["arama"]),

            # ⚠ tur 14 (Hedef C): ARAŞTIRMA ARAÇLARI
            #
            # Kullanıcı hedefi: *"myAi'ye araştırma yeteneği ver —
            # istediği dilin/kütüphanenin güncel sürümünü ve API'sini
            # internetten bulup okuyabilsin"*
            wrap("doc.kutuphane_ara",
                 "Bir dilin/kütüphanenin resmî dokümanını internetten bulur "
                 "ve okur. İmzalar + kod örnekleri döner ve **kasaya yazar** "
                 "(kalıcı öğrenme). Desteklenen diller: python, gdscript, "
                 "javascript, rust, cpp.",
                 Safety.READ,
                 [P("dil", required=True,
                    description="python | gdscript | javascript | rust | cpp"),
                  P("kutuphane", required=True,
                    description="Modül/sınıf adı, ör. pathlib veya Vector3"),
                  P("surum", description="Sürüm, ör. 3.13 (boş = güncel)")],
                 "doküman imzaları + örnekler", ["arastirma", "api"]),

            wrap("doc.api_imzasi_al",
                 "Bir fonksiyon/metot için imza, parametreler ve örnek "
                 "getirir. `Modül.ad` biçiminde verilir.",
                 Safety.READ,
                 [P("dil", required=True, description="Dil"),
                  P("fonksiyon", required=True,
                    description="Modül.ad, ör. pathlib.Path.read_text")],
                 "imza + parametreler + örnek", ["arastirma", "api"]),

            wrap("doc.surum_karsilastir",
                 "İki sürüm arasındaki API farkını çıkarır (yeni/kalkan "
                 "imzalar). Kurulu sürümle eşleşiyorsa **kesin** veri "
                 "(inspect) kullanır.",
                 Safety.READ,
                 [P("dil", required=True, description="Dil"),
                  P("kutuphane", required=True, description="Modül adı"),
                  P("v1", required=True, description="Eski sürüm, ör. 3.10"),
                  P("v2", required=True, description="Yeni sürüm, ör. 3.13")],
                 "sürüm farkı", ["arastirma", "surum"]),

            # ⚠ tur 23 (Hedef C): Godot 4 API BAĞLAM enjektörü.
            #
            # Ölçülen gerekçe (tur 21-22): bağlamsız model
            # `n.Seed`, `n.SetNoiseScale()`, `m.ambientColor` gibi
            # **Godot 3** API'leri uyduruyor. Bağlam verilince
            # **0 uydurma** üretiyor (`n.affine_noise`, `n.seed`).
            wrap("doc.godot_api_baglam",
                 "Godot 4 API referansını ve **derleyici ile doğrulanmış** "
                 "kullanım örneklerini getirir. Kod yazmadan **önce** "
                 "çağırın — Godot 3 adları (ambientColor, Seed, "
                 "SetNoiseScale) Godot 4'te geçersizdir.",
                 Safety.READ,
                 [P("sorgu", required=True,
                    description="Ne arıyorsunuz, ör. 'FastNoiseLite gürültü'"),
                  P("sinir", "integer", "Kaç API satırı (varsayılan 24)",
                    default=24)],
                 "Godot 4 API referansı + kod örnekleri",
                 ["godot", "arastirma", "api"]),

            # ══════════════════════════════════════════════════════
            # ⚠ tur 10: BLENDER VARLIK ÜRETİMİ
            # ══════════════════════════════════════════════════════
            #
            # Ölçülen yetenek (tur 1-9):
            #   Kalite kademesi : lowpoly/orta/kaliteli
            #                     → 12 / 104 / 965 vertex (kaya)
            #                     → 424 / 2012 / 18112 vertex (insan)
            #   Kategori        : kaya, bitki, esya, mimari, yaratik,
            #                     karakter (6)
            #   Ürün türü       : 12 (agac, cali, cim, sandik, fici,
            #                     masa, kilic, duvar, kapi, sutun,
            #                     zemin, insan, timsah)
            wrap("asset.uret",
                 "Blender'da **3B varlık üretir** (mesh + malzeme + UV). "
                 "Kalite kademesi (`lowpoly`/`orta`/`kaliteli`) ve "
                 "kategori/ürün seçilir. Yaratık ve karakter için "
                 "**armature + automatic weight + animasyon** de üretilir. "
                 "Çıktı: `.blend` + `.glb` + meta veri (kataloğa kayıtlı).",
                 Safety.WRITE,
                 [P("urun", required=True,
                    description="Ürün türü: agac, cali, cim, sandik, fici, "
                                "masa, kilic, duvar, kapi, sutun, zemin, "
                                "kaya, insan, timsah"),
                  P("kalite", description="lowpoly | orta | kaliteli",
                    default="orta"),
                  P("tohum", "integer", "Varyant tohumu (aynı tohum = aynı "
                                        "çıktı)", default=42)],
                 "varlık yolu + vertex/yüz/UV/armature ölçümleri",
                 ["blender", "varlik", "mesh", "3d", "model"]),

            wrap("asset.animasyon",
                 "Blender varlığına **animasyon ekler**. 5 hazır tarif "
                 "(rol tabanlı, farklı kemik adlarına uygulanabilir): "
                 "`bekleme` (döngü), `yurume` (döngü), `kosma` (döngü), "
                 "`saldiri` (tek), `olum` (tek). Armature'ü olmayan "
                 "varlığa uygulanamaz.",
                 Safety.WRITE,
                 [P("hedef", required=True,
                    description="Varlık kimliği (ör. `timsah_0001`) veya "
                                "`kategori/ad`"),
                  P("tarif", description="Tek tarif adı (boş = tümü)")],
                 "yazılan animasyonlar + fcurve/keyframe ölçümleri",
                 ["blender", "animasyon", "rig", "armature"]),

            # ⚠ tur 11 (Hedef B): DOKU — ara, seçime sun, uygula.
            #
            # Ölçülen yetenek:
            #   Prosedürel : 7 doku (kaya, toprak, tahta, mermer,
            #                deri, metal, kumas) — 7/7 Blender'da derlendi
            #   Yerel      : PNG/JPG kütüphanesi (boyut başlıktan okunur)
            #   Puanlama   : kategori +3, tür +2.5, ad +2, etiket +1,
            #                çözünürlük≥1024 +0.5
            #   Uygulama   : kaya←kaya · agac←tahta · kilic←metal → 10/10
            wrap("asset.doku",
                 "Doku **arar, puanlar, seçime sunar ve uygular**. "
                 "Prosedürel dokular (7 hazır: kaya/toprak/tahta/mermer/"
                 "deri/metal/kumas — dosya gerekmez) veya yerel PNG/JPG "
                 "kütüphanesi kullanılır. Çevrimiçi kaynaklar **lisans "
                 "bilgisiyle** listelenir.",
                 Safety.WRITE,
                 [P("sorgu", description="Ne aranıyor (ör. 'kaya pürüzlü', "
                                         "'tahta renk'). Boş = önizleme"),
                  P("hedef", description="Doku uygulanacak varlık kimliği"),
                  P("doku", description="Prosedürel doku adı (kaya, toprak, "
                                        "tahta, mermer, deri, metal, kumas)"),
                  P("dosya", description="Yerel görüntü doku yolu (PNG/JPG)"),
                  P("kanal", description="renk | pürüzlülük | metal | "
                                         "normal | yükseklik",
                    default="renk")],
                 "bulunan dokular + puanlar · uygulama ölçümleri "
                 "(düğüm/bağlantı/UV)",
                 ["blender", "doku", "texture", "material"]),

            # ⚠ tur 13 (Hedef D): ANİMASYON GEÇİŞLERİ.
            #
            # Ölçülen: ayrı action'lar bağımsız çalışır → ani sıçrama
            # (ör. `olum` biterken diz 1.45 rad, sonraki animasyon 0).
            # Çözüm: **smoothstep** (3t²−2t³) geçiş action'ı + NLA.
            #
            # Ölçüm: 7 geçiş · 3 GERÇEK (fark>0) · NLA 12 iz.
            wrap("asset.gecis",
                 "Animasyonlar arası **geçiş (blend)** üretir ve "
                 "**NLA**'ya sıralı yerleştirir. `bekleme→yurume→kosma` "
                 "gibi zincirlerde ani sıçramayı yumuşatır (smoothstep). "
                 "Armature'ü ve en az 2 animasyonu olan varlığa uygulanır.",
                 Safety.WRITE,
                 [P("hedef", required=True,
                    description="Varlık kimliği (ör. `insan`, `timsah_0001`)"),
                  P("kare", "integer", "Geçiş uzunluğu (kare)",
                    default=8)],
                 "üretilen geçişler + fark ölçümleri + NLA iz sayısı",
                 ["blender", "animasyon", "gecis", "nla"]),

            # ⚠ tur 14: VARLIK ÖNİZLEME — 4 açıdan render.
            #
            # ⚠ Neden gerekli: tur 1-13'te varlıklar yalnız **sayıyla**
            # doğrulandı (vertex, yüz, kemik, weight). Sayılar doğru
            # olsa da model **bozuk görünebilir** (ters normal, yanlış
            # ölçek, eksik parça). Bu araç **görsel doğrulama** sağlar.
            #
            # Ölçüm: 4/4 açı (ön, yan, 3/4, üst) · ~3 sn · EEVEE 512².
            wrap("asset.onizleme",
                 "Varlığı **4 açıdan render eder** (ön, yan, 3/4, üst) ve "
                 "PNG kaydeder. Otomatik kamera + 3 nokta ışık + zemin "
                 "kurar. Görsel doğrulama içindir — sayısal ölçüm "
                 "`asset.katalog`'dadır.",
                 Safety.WRITE,
                 [P("hedef", required=True,
                    description="Varlık kimliği (ör. `kaya`, `insan`, "
                                "`timsah_0001`)"),
                  P("cozunurluk", "integer", "PNG boyutu (px)",
                    default=512)],
                 "render edilen açılar + PNG boyutları + süre",
                 ["blender", "onizleme", "render", "gorsel"]),

            # ⚠ tur 18 (Hedef D): ANİMASYON DOĞRULAMA.
            #
            # ⚠ Neden gerekli: sayı doğru olsa da animasyon bozuk olabilir
            # (kemik dönmüyor, değer çok küçük, yanlış eksen).
            #
            # ⚠ ÖLÇÜLEN SINIR (dürüst): `pose_bone.matrix` ve render
            # edilen poz, action değerlendirmesini **yansıtmıyor**
            # (piksel farkı %0). Bu yüzden ölçüt **fcurve değerleridir** —
            # animasyonun **verisini** doğrular, **görünüşünü** değil.
            # Görsel doğrulama için `asset.onizleme` kullanılır.
            #
            # Ölçüm: 23 animasyon · NORMAL 16 · GEÇİŞ DÜZ 7 · HATA 0.
            wrap("asset.anim_onizleme",
                 "Varlığın animasyonlarını **doğrular**: her action için "
                 "zaman çizelgesinden kare render eder ve **kemik hareket "
                 "aralığını** ölçer (radyan). Kararlar: `NORMAL` · "
                 "`HAREKET_YOK` · `GEÇİŞ_DÜZ` (tasarım gereği) · "
                 "`AŞIRI_HAREKET`.",
                 Safety.READ,
                 [P("hedef", required=True,
                    description="Varlık kimliği (ör. `insan`, `timsah_0001`)"),
                  P("anim", description="Tek action adı (boş = tümü)")],
                 "animasyon başına karar + hareket aralığı + hareketli kemik",
                 ["blender", "animasyon", "dogrulama", "hareket"]),

            wrap("asset.katalog",
                 "Üretilen **tüm varlıkları** listeler — kategori, kalite, "
                 "vertex/yüz sayısı, UV, kemik, animasyon bilgisiyle. "
                 "Kategori filtresi verilebilir.",
                 Safety.READ,
                 [P("kategori", description="kaya | bitki | esya | mimari | "
                                            "yaratik | karakter (boş = tümü)"),
                  P("ayrintili", "boolean", "Meta veriyi de göster",
                    default=False)],
                 "varlık listesi + istatistik",
                 ["blender", "katalog", "varlik", "liste"]),

            # ⚠ tur 41 (Hedef C/D): Godot 4 KOD ONARIM döngüsü.
            #
            # Ölçülen gerekçe (tur 29-41): onarım döngüsü olmadan
            # derleme **0/4**, döngü ile **4/4** (ortalama tur 1,2).
            wrap("code.godot_onar",
                 "Godot 4 kodunu **gerçek derleyiciyle** doğrular; hata "
                 "varsa modele geri besleyip yeniden ürettirir (en çok "
                 "`tur` deneme). Sarkan nokta, `f` son eki, kod çiti, "
                 "değersiz atama gibi **6 sözdizimsel kalıbı** düzeltir.",
                 Safety.READ,
                 [P("onek", required=True,
                    description="GDScript öneki, ör. 'extends Node3D\\n\\n"
                                "func _ready() -> void:\\n    var n := "
                                "FastNoiseLite.new()\\n    n.'"),
                  P("sorgu", description="Ne üretilecek (bağlam için), "
                                         "ör. 'FastNoiseLite gürültü'"),
                  P("tur", "integer", "En çok onarım turu",
                    default=4),
                  P("model", description="Model yolu (boş = varsayılan)")],
                 "derlenen kod + tur sayısı + hata geçmişi",
                 ["godot", "onarim", "derleme"]),

            wrap("code.list_symbols",
                 "Koddaki sınıf/fonksiyon/değişkenleri listeler. Python kesin "
                 "(ast), diğer diller yaklaşık (regex) ayrıştırılır.",
                 Safety.READ,
                 [P("path", description="Dosya veya dizin"),
                  P("kinds", "array", "Tür filtresi: class/function/method/variable")],
                 "sembol listesi", ["analiz", "sembol"]),

            wrap("code.find_symbol",
                 "Bir sembolün tanımını ve kullanımlarını bulur.",
                 Safety.READ,
                 [P("name", required=True, description="Sembol adı")],
                 "tanımlar ve kullanımlar", ["analiz", "sembol"]),

            wrap("code.edit_file",
                 "Dosyada cerrahi değişiklik yapar: old_text'i new_text ile "
                 "değiştirir. old_text dosyada BİREBİR bulunmalıdır (girinti dahil). "
                 "Python dosyalarında sözdizimi bozulursa değişiklik otomatik geri alınır.",
                 Safety.WRITE,
                 [P("path", required=True, description="Dosya yolu"),
                  P("old_text", required=True, description="Değiştirilecek mevcut metin"),
                  P("new_text", description="Yeni metin (boş = sil)", required=True),
                  P("replace_all", "boolean", "Tüm eşleşmeleri değiştir", default=False)],
                 "düzenleme sonucu ve fark", ["yazma", "duzenleme"]),

            wrap("code.create_file",
                 "Yeni dosya oluşturur (veya overwrite=true ile üzerine yazar).",
                 Safety.WRITE,
                 [P("path", required=True, description="Dosya yolu"),
                  P("content", description="Dosya içeriği", required=True),
                  P("overwrite", "boolean", "Üzerine yaz", default=False)],
                 "oluşturma sonucu", ["yazma"]),

            wrap("code.revert",
                 "Yapılan bir düzenlemeyi geri alır.",
                 Safety.WRITE,
                 [P("backup_id", description="Yedek kimliği"),
                  P("path", description="Dosya yolu (en son yedeği geri alır)")],
                 "geri alma sonucu", ["geri-alma"]),

            wrap("code.check_syntax",
                 "Dosyanın sözdizimini doğrular (Python/JSON/YAML kesin, diğerleri parantez).",
                 Safety.READ,
                 [P("path", required=True, description="Dosya yolu")],
                 "sözdizimi sonucu", ["dogrulama"]),

            wrap("code.run_command",
                 "Çalışma alanında komut çalıştırır. Yıkıcı komutlar engellenir; "
                 "çıktı ve çıkış kodu döner.",
                 Safety.EXECUTE,
                 [P("command", required=True, description="Komut satırı"),
                  P("cwd", description="Çalışma dizini (boş = kök)"),
                  P("timeout", "number", "Saniye", default=120)],
                 "komut çıktısı", ["kabuk", "komut"]),

            wrap("code.run_tests",
                 "Testleri çalıştırır ve sonucu yorumlar (geçen/kalan sayısı, "
                 "başarısız testler). Framework otomatik algılanır.",
                 Safety.EXECUTE,
                 [P("target", description="Test dosyası/dizini"),
                  P("framework", description="pytest/unittest/npm/cargo/go"),
                  P("timeout", "number", "Saniye", default=300)],
                 "test sonucu", ["test", "dogrulama"]),

            wrap("code.diagnose_error",
                 "Hata çıktısını yorumlar: dosya, satır, hata türü ve öneri. "
                 "Test/komut başarısız olduğunda doğru dosyaya gitmek için kullanın.",
                 Safety.READ,
                 [P("text", required=True, description="Komut/test çıktısı")],
                 "hata analizi", ["hata", "teshis"]),

            wrap("code.git_status",
                 "Git durumunu okur (dal, değişen ve izlenmeyen dosyalar).",
                 Safety.READ,
                 [], "git durumu", ["git"]),

            wrap("code.git_diff",
                 "Git farkını okur (neler değişti?).",
                 Safety.READ,
                 [P("path", description="Yalnızca bu dosya"),
                  P("staged", "boolean", "Sahnelenmiş değişiklikler", default=False)],
                 "git farkı", ["git"]),
        ]

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<CodingAgent workspace={self._workspace} "
                f"edits={self._counter['edits']}>")


# ─── Modül Düzeyi Yardımcılar ───────────────────────────────────────────────

def _is_within(path: Path, root: Path) -> bool:
    """`path` kökün altında mı?"""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _excluded(name: str, patterns: Sequence[str]) -> bool:
    """Ad dışlama kalıplarından birine uyuyor mu?"""
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern) or name == pattern:
            return True
    return False


def _bracket_balance(text: str) -> str:
    """
    Parantez/köşeli parantez/süslü parantez dengesini denetler.

    Metin içindeki parantezleri saymamak için dizeler ve yorumlar temizlenir.
    Bu, dil ayrıştırıcısı olmayan diller için kaba ama yararlı bir kontroldür.
    """
    cleaned = re.sub(r"//[^\n]*|#[^\n]*", "", text)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'"(?:\\.|[^"\\])*"', '""', cleaned)
    cleaned = re.sub(r"'(?:\\.|[^'\\])*'", "''", cleaned)

    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    for character in cleaned:
        if character in "([{":
            stack.append(character)
        elif character in pairs:
            if not stack or stack[-1] != pairs[character]:
                return f"parantez dengesiz: beklenmeyen '{character}'"
            stack.pop()
    if stack:
        return f"parantez dengesiz: {len(stack)} açık kaldı ({''.join(stack[-3:])})"
    return ""


#: Yıkıcı komut kalıpları (kendi başına çalıştırıldığında engellenir)
DANGEROUS_PATTERNS: List[Tuple[str, str]] = [
    (r"\bformat\s+[a-z]:", "disk biçimlendirme"),
    (r"\brd\s+/s\b|\brmdir\s+/s\b", "özyinelemeli dizin silme"),
    (r"\bdel\s+/[fsq]\b.*\*", "joker ile dosya silme"),
    (r"Remove-Item\s+.*-Recurse.*-Force", "zorla özyinelemeli silme"),
    (r"\bshutdown\b|\bStop-Computer\b|\bRestart-Computer\b", "sistemi kapatma/yeniden başlatma"),
    (r"\bdiskpart\b", "disk bölümleme"),
    (r"\bbcdedit\b", "önyükleme yapılandırması"),
    (r"\bcipher\s+/w\b", "diski güvenli silme"),
    (r"\bvssadmin\s+delete\b", "gölge kopya silme"),
    (r"\breg\s+delete\s+HKLM", "kayıt defteri silme (HKLM)"),
    (r"\btakeown\b|\bicacls\b.*\/grant", "sahiplik/izin değiştirme"),
    (r":\(\)\s*\{.*\};:", "fork bomb"),
    (r"\bmkfs\b", "dosya sistemi oluşturma"),
    (r"\bdd\s+if=.*of=/dev/", "ham disk yazma"),
    (r">\s*/dev/sd", "ham diske yönlendirme"),
    (r"\bgit\s+push\s+.*--force\b", "zorla uzak depoya yazma"),
    (r"\bgit\s+reset\s+--hard\b", "commit edilmemiş değişiklikleri silme"),
    (r"\bgit\s+clean\s+-[a-z]*f", "izlenmeyen dosyaları silme"),
    (r"\bpip\s+uninstall\b.*-y", "paket kaldırma (onaysız)"),
    (r"\bnpm\s+publish\b", "paket yayınlama"),
]

_COMPILED_DANGEROUS = [
    (re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in DANGEROUS_PATTERNS
]


def detect_dangerous_command(command: str) -> Tuple[bool, str]:
    """
    Komutun yıkıcı olup olmadığını tespit eder.

    Args:
        command: Komut satırı

    Returns:
        (yıkıcı_mı, gerekçe)
    """
    if not command:
        return False, ""
    text = str(command)
    for pattern, reason in _COMPILED_DANGEROUS:
        if pattern.search(text):
            return True, f"{reason} (kalıp: {pattern.pattern[:40]})"
    return False, ""


def parse_test_output(text: str, framework: str = "") -> Dict[str, Any]:
    """
    Test çıktısını yapılandırılmış özete çevirir.

    Farklı framework'lerin çıktı biçimlerini tanır (pytest, unittest, jest,
    cargo, go test).

    Args:
        text: Ham çıktı
        framework: Framework adı (ipucu)

    Returns:
        {"passed", "failed", "errors", "skipped", "summary", "failures", "duration"}
    """
    result: Dict[str, Any] = {
        "passed": 0, "failed": 0, "errors": 0, "skipped": 0,
        "summary": "", "failures": [], "duration": "",
    }
    if not text:
        return result

    # pytest: "12 passed, 2 failed, 1 skipped in 1.23s"
    # Sıra değişebilir: "2 failed, 10 passed, 1 skipped in 1.23s" de geçerlidir,
    # bu yüzden her sayaç bağımsız aranır.
    pytest_match = re.search(
        r"((?:\d+\s+\w+(?:,\s*)?)+?)\s+in\s+([\d.]+)s", text
    )
    if pytest_match:
        counts_text = pytest_match.group(1)
        result["duration"] = f"{pytest_match.group(2)}s"
        for count, label in re.findall(r"(\d+)\s+(\w+)", counts_text):
            label = label.lower()
            if label.startswith("pass"):
                result["passed"] = int(count)
            elif label.startswith("fail"):
                result["failed"] = int(count)
            elif label.startswith("error"):
                result["errors"] = int(count)
            elif label.startswith("skip"):
                result["skipped"] = int(count)
    else:
        # Süre bilgisi olmayan pytest çıktısı
        for count, label in re.findall(r"(\d+)\s+(passed|failed|error|errors|skipped)",
                                       text, re.IGNORECASE):
            label = label.lower()
            if label == "passed":
                result["passed"] = int(count)
            elif label == "failed":
                result["failed"] = int(count)
            elif label.startswith("error"):
                result["errors"] = int(count)
            elif label == "skipped":
                result["skipped"] = int(count)

    # unittest: "Ran 15 tests in 0.123s" + "FAILED (failures=2, errors=1)"
    if not result["passed"] and "Ran " in text:
        ran = re.search(r"Ran\s+(\d+)\s+tests?", text)
        failed = re.search(r"failures=(\d+)", text)
        errors = re.search(r"errors=(\d+)", text)
        if ran:
            total = int(ran.group(1))
            fail_count = int(failed.group(1)) if failed else 0
            error_count = int(errors.group(1)) if errors else 0
            result["failed"] = fail_count
            result["errors"] = error_count
            result["passed"] = max(0, total - fail_count - error_count)
        duration = re.search(r"in\s+([\d.]+)s", text)
        if duration:
            result["duration"] = f"{duration.group(1)}s"

    # jest/vitest: "Tests: 2 failed, 10 passed, 12 total"
    jest_match = re.search(
        r"Tests:\s*(?:(\d+)\s+failed,\s*)?(?:(\d+)\s+passed,\s*)?(\d+)\s+total", text
    )
    if jest_match:
        result["failed"] = int(jest_match.group(1) or 0)
        result["passed"] = int(jest_match.group(2) or 0)

    # cargo: "test result: ok. 5 passed; 0 failed; 0 ignored"
    cargo_match = re.search(
        r"test result:.*?(\d+)\s+passed;\s*(\d+)\s+failed;\s*(\d+)\s+ignored", text
    )
    if cargo_match:
        result["passed"] = int(cargo_match.group(1))
        result["failed"] = int(cargo_match.group(2))
        result["skipped"] = int(cargo_match.group(3))

    # Başarısız test adları
    for pattern in (
        r"^FAILED\s+(?!\()(\S+)",              # pytest: FAILED tests/test_x.py::test_y
        r"^\s*(\S+)\s+\.\.\.\s+FAIL",          # unittest -v
        r"^✗\s+(\S+)",
        r"^FAIL:\s+(\S+)\s+\(([^)]+)\)",       # unittest: FAIL: test_b (mod.T)
        r"^ERROR:\s+(\S+)\s+\(([^)]+)\)",
    ):
        for match in re.finditer(pattern, text, re.MULTILINE):
            groups = [item for item in match.groups() if item]
            if not groups:
                continue
            name = groups[0] if len(groups) == 1 else f"{groups[1]}.{groups[0]}"
            # Özet satırlarını test adı sanma
            if re.fullmatch(r"\(?[a-z_]+=\d+[,)]*", name) or name.startswith("("):
                continue
            if name not in result["failures"]:
                result["failures"].append(name)
    result["failures"] = result["failures"][:40]

    total = result["passed"] + result["failed"] + result["errors"] + result["skipped"]
    if total:
        result["summary"] = (
            f"{result['passed']} geçti, {result['failed']} başarısız"
            + (f", {result['errors']} hata" if result["errors"] else "")
            + (f", {result['skipped']} atlandı" if result["skipped"] else "")
            + (f" ({result['duration']})" if result["duration"] else "")
        )
    elif "error" in text.lower() and "no tests" in text.lower():
        result["summary"] = "test bulunamadı"
    return result


def _last_error_line(lines: Sequence[str], start: int, window: int = 12) -> str:
    """İzleme kaydında hatanın asıl mesajını bulur."""
    for index in range(start + 1, min(len(lines), start + window)):
        stripped = lines[index].strip()
        if not stripped:
            continue
        if re.match(r"^[A-Za-z_.]+(Error|Exception|Warning):", stripped):
            return stripped[:250]
    return (lines[start].strip() if start < len(lines) else "")[:250]


__all__ = [
    "CodingAgent",
    "CodingError",
    "FileInfo",
    "Symbol",
    "SearchHit",
    "EditResult",
    "LANGUAGE_BY_EXTENSION",
    "BINARY_EXTENSIONS",
    "DEFAULT_EXCLUDES",
    "detect_dangerous_command",
    "parse_test_output",
]
