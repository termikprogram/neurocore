"""
NeuroCore PowerShell / CMD Derin Entegrasyonu
==============================================
Ajanın kabukla çalışmasını "komut çalıştır" seviyesinden **oturum yönetimi**
seviyesine çıkaran katman.

Neden ayrı katman?
    Basit `subprocess.run("komut")` üç şeyi yapamaz:

    1. **Durum koruma** — `cd`, ortam değişkeni ve `$variable` bir sonraki
       komutta kaybolur. Bir projeyi kurarken dizin değiştirmek zorunludur.
    2. **Canlı akış** — uzun süren komutlar (`npm install`, `docker build`)
       bitene kadar hiçbir şey göstermez; ajan ilerlemeyi göremez.
    3. **Yapılandırılmış hata** — PowerShell'in hata biçimi Python'dan farklıdır;
       `ErrorRecord` alanları ayrıştırılmalıdır.

Bu modül üçünü de sağlar:

    * `ShellSession` — kalıcı PowerShell süreci; `cd` ve değişkenler korunur
    * `ShellRunner` — tek seferlik komutlar, geriye dönük uyumluluk
    * `PowerShellDiagnostics` — hata ayrıştırma ve düzeltme önerisi
    * `CommandClassifier` — komutu güvenlik kategorisine ayırma

Güvenlik: `Sandbox` (işletim sistemi katmanı) enjekte edilirse yıkıcı komutlar
oradan engellenir; ayrıca burada **bağımsız** bir yıkıcı kalıp denetimi vardır
(katman olmadan da güvenli).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from neurocore.agent.tools import Safety, ToolContext, ToolDef, ToolParameter
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.shell")


# ─── Komut Sınıflandırma ────────────────────────────────────────────────────

class CommandClass(str, Enum):
    """Komutun niyet kategorisi (onay kararı için)."""
    INFO = "info"          # bilgi toplama: ls, cat, git status, python --version
    BUILD = "build"        # derleme/çalıştırma: pip install, npm run, pytest, godot
    MODIFY = "modify"      # proje içi değişiklik: mkdir, touch, git add
    DELETE = "delete"      # silme: rm, del, Remove-Item
    SYSTEM = "system"      # sistem düzeyi: registry, servis, güç
    DANGEROUS = "dangerous"  # yıkıcı: format, diskpart, fork bomb


#: Sınıf → güvenlik sınıfı eşlemesi
CLASS_SAFETY: Dict[str, Safety] = {
    "info": Safety.READ,
    "build": Safety.EXECUTE,
    "modify": Safety.WRITE,
    "delete": Safety.DESTRUCTIVE,
    "system": Safety.SYSTEM,
    "dangerous": Safety.SYSTEM,
}

#: Sınıf → Türkçe etiket
CLASS_LABELS: Dict[str, str] = {
    "info": "bilgi toplama",
    "build": "çalıştırma/derleme",
    "modify": "proje değişikliği",
    "delete": "silme",
    "system": "sistem işlemi",
    "dangerous": "yıkıcı işlem",
}

#: Komut kalıpları → sınıf (ilk eşleşen kazanır, sıra önemli)
CLASSIFICATION_RULES: List[Tuple[str, str]] = [
    # Yıkıcı (en yüksek öncelik)
    (r"\bformat\s+[a-z]:", "dangerous"),
    (r"\bdiskpart\b|\bbcdedit\b|\bcipher\s+/w\b", "dangerous"),
    (r"\bmkfs\b|\bdd\s+if=.*of=/dev/", "dangerous"),
    (r":\(\)\s*\{.*\};:", "dangerous"),
    (r"\bvssadmin\s+delete\b", "dangerous"),
    (r"\brd\s+/s\b|\brmdir\s+/s\b", "dangerous"),
    (r"\bdel\s+/[fsq]\b", "dangerous"),
    (r"Remove-Item\b.*-Recurse.*-Force", "dangerous"),
    (r"\bshutdown\b|\bStop-Computer\b|\bRestart-Computer\b", "dangerous"),
    # Sistem
    (r"\breg\s+(add|delete|import)\b", "system"),
    (r"\bSet-ItemProperty\b.*HKLM", "system"),
    (r"\bNew-Service\b|\bSet-Service\b|\bStart-Service\b|\bStop-Service\b", "system"),
    (r"\bnet\s+(user|localgroup)\b", "system"),
    (r"\bschtasks\b", "system"),
    (r"\bicacls\b|\btakeown\b", "system"),
    (r"\bpip\s+install\b|\bpip\s+uninstall\b", "build"),
    (r"\bwinget\b|\bchoco\b|\bscoop\b", "system"),
    # Silme
    (r"\brm\b\s+-[rf]|\brm\s+-rf\b", "delete"),
    (r"\bRemove-Item\b|\bdel\b|\berase\b|\brmdir\b|\brd\b", "delete"),
    (r"\bgit\s+(clean|reset\s+--hard)\b", "delete"),
    # Proje değişikliği
    (r"\bmkdir\b|\bNew-Item\b|\btouch\b|\bcopy\b|\bcp\b|\bmove\b|\bmv\b", "modify"),
    (r"\bgit\s+(add|commit|checkout|switch|merge|stash|init|clone)\b", "modify"),
    (r"\b(Out-File|Set-Content|Add-Content)\b", "modify"),
    (r"\bnpm\s+(install|i|ci|run|build|test)\b", "build"),
    (r"\bpnpm\b|\byarn\b", "build"),
    (r"\bpython\b.*\.py\b|\bpython\s+-m\b", "build"),
    (r"\bpytest\b|\bunittest\b|\bcargo\s+(build|test|run)\b|\bgo\s+(build|test|run)\b", "build"),
    (r"\bgodot\b|\bblender\b", "build"),
    (r"\bdocker\b|\bkubectl\b", "build"),
    (r"\bcmake\b|\bmake\b|\bmsbuild\b", "build"),
    # Bilgi (en düşük öncelik)
    (r"^\s*(ls|dir|Get-ChildItem|gci)\b", "info"),
    (r"^\s*(cat|type|Get-Content|gc)\b", "info"),
    (r"^\s*(pwd|Get-Location|cd)\b", "info"),
    (r"^\s*(echo|Write-Output|Write-Host)\b", "info"),
    (r"\bgit\s+(status|log|diff|show|branch|remote)\b", "info"),
    (r"\b(Get-|Test-Path|Test-Connection|Resolve-Path|Select-String|Where-Object|Measure-Object)\b", "info"),
    (r"\b(where|which|Get-Command)\b", "info"),
    (r"\bpython\s+--version\b|\bnode\s+--version\b|--version\b", "info"),
    (r"\bnvidia-smi\b|\btasklist\b|\bGet-Process\b", "info"),
]

_COMPILED_RULES = [
    (re.compile(pattern, re.IGNORECASE), kind) for pattern, kind in CLASSIFICATION_RULES
]


#: Çalışma alanı dışına çıkabilen komut kalıpları
_KAPSAM_KALIPLARI = (
    r"\bcd\s+(?P<yol>[^\s;|&]+)",
    r"\bset-location\s+(?P<yol>[^\s;|&]+)",
    r"\bpushd\s+(?P<yol>[^\s;|&]+)",
    r"\bchdir\s+(?P<yol>[^\s;|&]+)",
)


def _kapsam_denetle(command: str, kok: str) -> str:
    """
    Komutun çalışma alanı **dışına** çıkıp çıkmadığını denetler.

    ⚠ ÖLÇÜLEN DAVRANIŞ: ajan seçilen klasörün üst dizinlerinde işlem
    yapabiliyordu (`cd ..`, `Set-Location C:\\...`). Dosya araçları
    (`code.*`) zaten kum havuzunda; kabuk bu açığı kapatır.

    Returns:
        Hata metni (boş = sorun yok).
    """
    import re as _re
    from pathlib import Path as _P

    kok_yol = _P(kok).expanduser().resolve()
    for kalip in _KAPSAM_KALIPLARI:
        for eslesme in _re.finditer(kalip, command or "", _re.IGNORECASE):
            hedef = (eslesme.group("yol") or "").strip().strip('"\'')
            if not hedef or hedef.startswith("-"):
                continue
            try:
                aday = _P(hedef)
                if not aday.is_absolute():
                    aday = kok_yol / aday
                aday = aday.resolve()
            except OSError:
                continue
            try:
                aday.relative_to(kok_yol)
            except ValueError:
                return (f"güvenlik: çalışma alanı dışına çıkılamaz — "
                        f"{aday} (kök: {kok_yol})")
    return ""


def classify_command(command: str) -> Tuple[CommandClass, str]:
    """
    Komutu güvenlik kategorisine ayırır.

    Args:
        command: Komut satırı

    Returns:
        (sınıf, gerekçe)
    """
    text = str(command or "").strip()
    if not text:
        return CommandClass.INFO, "boş komut"
    for pattern, kind in _COMPILED_RULES:
        match = pattern.search(text)
        if match:
            return CommandClass(kind), f"{CLASS_LABELS[kind]} (kalıp: {match.group(0)[:36]})"
    return CommandClass.INFO, "tanınmayan komut — bilgi toplama varsayıldı"


def command_safety(command: str) -> Safety:
    """Komutun güvenlik sınıfı (araç politikası için)."""
    kind, _ = classify_command(command)
    return CLASS_SAFETY.get(kind.value, Safety.EXECUTE)


# ─── Sonuç Yapıları ─────────────────────────────────────────────────────────

@dataclass
class CommandResult:
    """Tek bir komut çalıştırmasının sonucu."""
    command: str = ""
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False
    #: Kullanıcı "Durdur" ile iptal etti mi? (zaman aşımından farklıdır)
    cancelled: bool = False
    blocked: bool = False
    blocked_reason: str = ""
    cwd: str = ""
    shell: str = ""
    command_class: str = "info"
    truncated: bool = False
    session_id: str = ""

    @property
    def ok(self) -> bool:
        """Başarılı mı?"""
        return (self.returncode == 0 and not self.timed_out
                and not self.blocked and not self.cancelled)

    @property
    def output(self) -> str:
        """stdout + stderr birleşik çıktı."""
        parts = [part for part in (self.stdout, self.stderr) if part]
        return "\n".join(parts)

    def to_dict(self, max_chars: int = 20000) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        stdout, stderr, truncated = self.stdout, self.stderr, self.truncated
        if len(stdout) > max_chars:
            stdout = stdout[-max_chars:]
            truncated = True
        if len(stderr) > max_chars // 2:
            stderr = stderr[-max_chars // 2:]
            truncated = True
        return {
            "ok": self.ok, "command": self.command,
            "returncode": self.returncode,
            "stdout": stdout, "stderr": stderr,
            "duration_ms": round(self.duration_ms, 1),
            "timed_out": self.timed_out, "blocked": self.blocked,
            "cancelled": self.cancelled,
            "blocked_reason": self.blocked_reason,
            "cwd": self.cwd, "shell": self.shell,
            "command_class": self.command_class,
            "class_label": CLASS_LABELS.get(self.command_class, self.command_class),
            "safety": CLASS_SAFETY.get(self.command_class, Safety.EXECUTE).value,
            "truncated": truncated,
            "session_id": self.session_id,
        }


# ─── PowerShell Teşhis ──────────────────────────────────────────────────────

#: PowerShell hata kalıpları → (tür, öneri)
POWERSHELL_ERROR_HINTS: List[Tuple[str, str, str]] = [
    (r"is not recognized as the name of a cmdlet",
     "command_not_found",
     "komut yok — tam yolu kullanın veya shutil.which ile varlığını kontrol edin"),
    (r"Cannot find path '([^']+)'",
     "path_not_found",
     "yol yok — Test-Path ile önce varlığını doğrulayın"),
    (r"Access to the path .* is denied",
     "access_denied",
     "yetki yok — yönetici hakları gerekebilir veya sandbox yolu engelliyor"),
    (r"The term '([^']+)' is not recognized",
     "term_not_recognized",
     "araç kurulu değil veya PATH'te yok"),
    (r"Cannot overwrite|already exists",
     "conflict",
     "hedef zaten var — -Force kullanın veya farklı ad seçin"),
    (r"The process cannot access the file .* because it is being used by another process",
     "file_locked",
     "dosya başka bir süreç tarafından kilitli — önce o süreci kapatın"),
    (r"unauthorized|401|403",
     "auth",
     "kimlik doğrulama hatası — anahtar/token eksik veya geçersiz"),
    (r"could not resolve host|Name or service not known|getaddrinfo",
     "network",
     "ağ/DNS hatası — internet bağlantısını kontrol edin"),
    (r"timed out|timeout",
     "timeout",
     "zaman aşımı — süreyi artırın veya işi küçültün"),
    (r"out of memory|OutOfMemory",
     "memory",
     "bellek yetersiz — kaynak nezaketi toplu işi küçültür"),
    (r"Syntax error|ParserError|Unexpected token",
     "syntax",
     "PowerShell sözdizimi hatası — tırnak ve parantezleri kontrol edin"),
    (r"ExecutionPolicy|running scripts is disabled",
     "execution_policy",
     "betik çalıştırma politikası engelliyor: -ExecutionPolicy Bypass kullanın"),
    (r"ModuleNotFoundError|No module named",
     "python_module",
     "Python paketi eksik — pip install ile kurun"),
    (r"FileNotFoundError|No such file or directory",
     "file_not_found",
     "dosya yok — yolu kontrol edin"),
    (r"non-zero exit|exited with code [1-9]",
     "exit_code",
     "alt süreç başarısız — çıktısını inceleyin"),
]

_COMPILED_PS_HINTS = [
    (re.compile(pattern, re.IGNORECASE), kind, hint)
    for pattern, kind, hint in POWERSHELL_ERROR_HINTS
]


class PowerShellDiagnostics:
    """
    PowerShell/CMD çıktısını yorumlar: hata türü, konum ve öneri.

    Ajan ham çıktıyı okumak yerine bu özeti kullanır; bu, düzeltme döngüsünü
    kısaltır.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}

    def analyze(self, text: str, command: str = "") -> Dict[str, Any]:
        """
        Çıktıyı analiz eder.

        Args:
            text: Komut çıktısı (stdout + stderr)
            command: Çalıştırılan komut (bağlam için)

        Returns:
            {"ok", "errors": [{"kind","message","hint","line"}],
             "warnings": [...], "summary", "error_count"}
        """
        if not text:
            return {"ok": True, "errors": [], "warnings": [],
                    "summary": "çıktı yok", "error_count": 0}

        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        lines = str(text).splitlines()

        for kind_regex, kind, hint in _COMPILED_PS_HINTS:
            for index, line in enumerate(lines, start=1):
                if not kind_regex.search(line):
                    continue
                message = line.strip()[:250]
                entry = {"kind": kind, "message": message, "hint": hint, "line": index}
                if entry not in errors:
                    errors.append(entry)
                break

        # Uyarı satırları
        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if re.search(r"\bWARNING\b|\bUyarı\b|^W:", stripped, re.IGNORECASE):
                warnings.append({"message": stripped[:200], "line": index})
                if len(warnings) >= 10:
                    break

        # PowerShell hata kaydı bloğu (ErrorRecord)
        record = re.search(
            r"(?:At line:(\d+) char:(\d+))|(?:\+\s*CategoryInfo\s*:\s*([^\n]+))"
            r"|(?:\+\s*FullyQualifiedErrorId\s*:\s*([^\n]+))",
            text,
        )
        if record:
            details: Dict[str, Any] = {}
            if record.group(1):
                details["line"] = int(record.group(1))
            if record.group(3):
                details["category"] = record.group(3).strip()
            if record.group(4):
                details["error_id"] = record.group(4).strip()
            if details:
                errors.append({"kind": "powershell_record", "message": json.dumps(
                    details, ensure_ascii=False), "hint": "PowerShell hata kaydı",
                    "line": details.get("line", 0)})

        summary = "çıktı temiz"
        if errors:
            primary = errors[0]
            location = f" (satır {primary['line']})" if primary.get("line") else ""
            summary = (
                f"{len(errors)} sorun{location}: {primary['message'][:120]} "
                f"→ {primary['hint']}"
            )
        elif warnings:
            summary = f"{len(warnings)} uyarı (hata yok)"

        return {
            "ok": not errors,
            "errors": errors[:15],
            "warnings": warnings[:10],
            "summary": summary,
            "error_count": len(errors),
            "command": command,
        }

    def suggest_fix(self, text: str, command: str = "") -> Dict[str, Any]:
        """
        Hataya göre **somut** düzeltme önerisi üretir.

        Returns:
            {"kind", "suggestion", "alternative_command", "confidence"}
        """
        analysis = self.analyze(text, command)
        if analysis["ok"]:
            return {"kind": "none", "suggestion": "hata yok", "confidence": 1.0}

        primary = analysis["errors"][0]
        kind = primary["kind"]
        suggestion = primary["hint"]
        alternative = ""
        confidence = 0.6

        if kind == "command_not_found" or kind == "term_not_recognized":
            match = re.search(r"'([^']+)'", primary["message"])
            tool = match.group(1) if match else ""
            if tool:
                found = shutil.which(tool)
                alternative = f"Get-Command {tool} -ErrorAction SilentlyContinue"
                suggestion = (
                    f"'{tool}' PATH'te bulunamadı. "
                    + (f"Bulundu: {found}" if found else "Kurulu değil — capabilities.probe ile kontrol edin.")
                )
                confidence = 0.9
        elif kind == "path_not_found":
            match = re.search(r"'([^']+)'", primary["message"])
            if match:
                alternative = f"Test-Path -LiteralPath '{match.group(1)}'"
                confidence = 0.85
        elif kind == "execution_policy":
            alternative = "powershell -ExecutionPolicy Bypass -File <betik>"
            confidence = 0.9
        elif kind == "access_denied":
            alternative = "Start-Process powershell -Verb RunAs"
            confidence = 0.7
        elif kind == "python_module":
            match = re.search(r"No module named '([^']+)'", primary["message"])
            if match:
                alternative = f"{sys.executable} -m pip install {match.group(1)}"
                confidence = 0.9
        elif kind == "timeout":
            alternative = "# süreyi artırın veya işi parçalara bölün"
            confidence = 0.5

        return {
            "kind": kind,
            "suggestion": suggestion,
            "alternative_command": alternative,
            "confidence": confidence,
            "original_command": command,
        }


# ─── Kalıcı Kabuk Oturumu ───────────────────────────────────────────────────

class ShellSession:
    """
    Kalıcı PowerShell oturumu.

    `cd`, ortam değişkenleri ve PowerShell değişkenleri komutlar arasında
    **korunur**. Bu, bir projeyi kurarken veya sıralı komutlar çalıştırırken
    zorunludur.

    Uygulama biçimi: tek bir `powershell.exe` süreci `-NoLogo -NoProfile
    -NonInteractive -Command -` ile başlatılır ve stdin'e komut yazılır;
    her komutun sonuna bir **sentinel** yazılır. Sentinel görülene kadar
    okunan satırlar o komutun çıktısıdır. Bu yöntem harici bağımlılık
    gerektirmez ve Windows'ta güvenilir çalışır.

    Kullanım:
        session = ShellSession()
        session.start(cwd="E:/projelerim/x")
        session.run("cd src")
        session.run("$env:MY_VAR = 'deger'")
        session.run("python build.py")     # önceki cd ve değişken geçerli
        session.stop()
    """

    #: Her komutun sonuna yazılan ayırıcı (çıktıdan ayrıştırılır)
    SENTINEL = "<<<NC_SENTINEL_{token}>>>"

    def __init__(self, shell: str = "powershell", cwd: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None) -> None:
        self.shell = shell
        self._cwd = cwd or os.getcwd()
        self._env = dict(env or {})
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self._buffer: List[str] = []
        self._reader: Optional[threading.Thread] = None
        self._stop_reader = threading.Event()
        self._session_id = f"sh_{int(time.time() * 1000) % 1000000:06d}"
        self._counter = {"commands": 0, "errors": 0, "restarts": 0}
        self._history: List[Dict[str, Any]] = []
        self._last_cwd = self._cwd

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        """Oturum kimliği."""
        return self._session_id

    @property
    def alive(self) -> bool:
        """Süreç çalışıyor mu?"""
        return self._process is not None and self._process.poll() is None

    @property
    def cwd(self) -> str:
        """Oturumun güncel dizini (PowerShell'den okunur)."""
        return self._last_cwd

    def start(self) -> bool:
        """
        Kalıcı PowerShell sürecini başlatır.

        Returns:
            Başarılı ise True
        """
        if self.alive:
            return True
        executable = self._resolve_shell()
        if not executable:
            logger.error(f"Kabuk bulunamadı: {self.shell}")
            return False

        environment = dict(os.environ)
        environment.update(self._env)
        environment["PYTHONIOENCODING"] = "utf-8"

        arguments = [executable, "-NoLogo", "-NoProfile", "-NonInteractive",
                     "-ExecutionPolicy", "Bypass", "-Command", "-"]
        if os.path.basename(executable).lower().startswith("pwsh"):
            arguments = [executable, "-NoLogo", "-NoProfile", "-NonInteractive",
                         "-Command", "-"]

        try:
            self._process = subprocess.Popen(
                arguments,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=self._cwd if os.path.isdir(self._cwd) else None,
                env=environment, text=True, encoding="utf-8", errors="replace",
                bufsize=1, shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            logger.error(f"Kabuk başlatılamadı: {exc}")
            self._process = None
            return False

        # Çıktı okuyucu iş parçacığı (kilitlenmeyi önler)
        self._stop_reader.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name="ShellSessionReader")
        self._reader.start()
        time.sleep(0.4)     # kabuk hazır olsun
        logger.info(f"Kalıcı kabuk oturumu başladı: {self._session_id} ({executable})")
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Oturumu kapatır."""
        self._stop_reader.set()
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Oturum kapatma hatası: {exc}")
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
        logger.info(f"Kabuk oturumu kapatıldı: {self._session_id} "
                    f"({self._counter['commands']} komut)")

    def restart(self) -> bool:
        """Oturumu yeniden başlatır (süreç çöktüyse)."""
        self.stop()
        self._counter["restarts"] += 1
        return self.start()

    def _resolve_shell(self) -> Optional[str]:
        """Kabuk çalıştırılabilirinin yolunu bulur."""
        if self.shell in ("powershell", "pwsh"):
            # pwsh (PowerShell 7) varsa onu tercih et — daha hızlı ve UTF-8 dostu
            for candidate in ("pwsh", "powershell"):
                found = shutil.which(candidate)
                if found:
                    return found
            for candidate in (
                r"C:\Program Files\PowerShell\7\pwsh.exe",
                os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                             "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
            ):
                if os.path.exists(candidate):
                    return candidate
        elif self.shell == "cmd":
            return os.environ.get("COMSPEC") or shutil.which("cmd")
        return shutil.which(self.shell)

    def _read_loop(self) -> None:
        """Süreç çıktısını arka planda okur (buffer'a yazar)."""
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                if self._stop_reader.is_set():
                    break
                self._buffer.append(line.rstrip("\r\n"))
        except (ValueError, OSError):
            pass

    # ─── Komut Çalıştırma ───────────────────────────────────────────────

    def run(self, command: str, timeout: float = 120.0,
            capture_cwd: bool = False) -> CommandResult:
        """
        Oturumda komut çalıştırır (durum korunur).

        Args:
            command: Komut
            timeout: Saniye
            capture_cwd: Komut sonrası dizini oku (`$PWD`) — yavaştır

        Returns:
            `CommandResult`
        """
        result = CommandResult(command=command, session_id=self._session_id,
                               shell=self.shell)
        # ⚠ Kapsam denetimi: çalışma alanı dışına çıkan cd engellenir
        kok = str(getattr(self, "_workspace", "") or os.getcwd())
        kapsam_hatasi = _kapsam_denetle(command, kok)
        if kapsam_hatasi:
            result.ok = False
            result.error = kapsam_hatasi
            result.exit_code = 126
            return result

        kind, reason = classify_command(command)
        result.command_class = kind.value

        if not command.strip():
            result.blocked = True
            result.blocked_reason = "boş komut"
            return result

        if not self.alive and not self.start():
            result.returncode = -1
            result.stderr = "kabuk oturumu başlatılamadı"
            self._counter["errors"] += 1
            return result

        token = f"{int(time.time() * 1000) % 100000000:08d}"
        sentinel = self.SENTINEL.format(token=token)
        # Çıkış kodunu sentinel ile birlikte yazdır
        payload = (
            f"{command}\n"
            f"$__nc_exit = if ($LASTEXITCODE -ne $null) {{ $LASTEXITCODE }} "
            f"else {{ if ($?) {{ 0 }} else {{ 1 }} }}\n"
            f"if (${'?'} -eq $false -and $Error.Count -gt 0) {{ $__nc_exit = 1 }}\n"
        )
        if capture_cwd:
            payload += f'Write-Output "NC_CWD:$(Get-Location)"\n'
        payload += f'Write-Output "{sentinel}:$__nc_exit"\n'

        started = time.perf_counter()
        with self._lock:
            self._buffer.clear()
            process = self._process
            if process is None or process.stdin is None:
                result.returncode = -1
                result.stderr = "kabuk stdin kapalı"
                return result
            try:
                process.stdin.write(payload)
                process.stdin.flush()
            except (OSError, ValueError) as exc:
                self._counter["errors"] += 1
                result.returncode = -1
                result.stderr = f"komut yazılamadı: {exc}"
                # Süreç ölmüş olabilir: bir kez yeniden dene
                if self.restart():
                    result.stderr += " (oturum yeniden başlatıldı, komutu tekrar çalıştırın)"
                return result

            # Sentinel'i bekle
            exit_code: Optional[int] = None
            deadline = time.time() + max(1.0, float(timeout))
            collected: List[str] = []
            while time.time() < deadline:
                if self._buffer:
                    line = self._buffer.pop(0)
                    if line.startswith(sentinel):
                        try:
                            exit_code = int(line.split(":", 1)[1].strip())
                        except (IndexError, ValueError):
                            exit_code = 0
                        break
                    if line.startswith("NC_CWD:"):
                        self._last_cwd = line[7:].strip()
                        continue
                    collected.append(line)
                    continue
                if not self.alive:
                    break
                time.sleep(0.02)

            result.duration_ms = (time.perf_counter() - started) * 1000
            result.stdout = "\n".join(collected)

            if exit_code is None:
                result.timed_out = time.time() >= deadline
                result.returncode = -1
                if result.timed_out:
                    result.stderr = f"zaman aşımı ({timeout:.0f}s)"
                    # Takılı komutu iptal etmek için oturumu yenile
                    self.restart()
                    result.stderr += " (oturum yenilendi)"
                else:
                    result.stderr = "kabuk beklenmedik şekilde kapandı"
                    self._counter["errors"] += 1
            else:
                result.returncode = exit_code
                result.cwd = self._last_cwd

        result.truncated = len(result.stdout) > 20000
        self._counter["commands"] += 1
        self._history.append({
            "at": time.time(), "command": command[:200],
            "returncode": result.returncode, "class": kind.value,
            "duration_ms": result.duration_ms, "ok": result.ok,
        })
        if len(self._history) > 300:
            self._history = self._history[-300:]
        return result

    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Oturum komut geçmişi."""
        return self._history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Oturum istatistikleri."""
        return {
            **self._counter,
            "session_id": self._session_id,
            "shell": self.shell,
            "alive": self.alive,
            "cwd": self._last_cwd,
            "history_size": len(self._history),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<ShellSession {self._session_id} shell={self.shell} "
                f"alive={self.alive} commands={self._counter['commands']}>")


# ─── Ana Kabuk Katmanı ──────────────────────────────────────────────────────

class ShellIntegration:
    """
    Ajan için kabuk katmanı: kalıcı oturum + tek seferlik komutlar + teşhis.

    Kullanım:
        shell = ShellIntegration()
        shell.initialize(workspace="E:/proje", config=config, event_bus=bus)

        # Tek seferlik
        print(shell.run("git status").output)

        # Kalıcı oturum (durum korunur)
        shell.start_session()
        shell.session_run("cd src")
        shell.session_run("python -m pytest")

        # Teşhis
        print(shell.diagnose(output_text, command))

        registry.register_many(shell.tools())
    """

    name = "agent.shell"

    def __init__(self) -> None:
        self._workspace: Optional[Path] = None
        self._config: Any = None
        self._event_bus: Any = None
        self._sandbox: Any = None
        self._os_layer: Any = None
        self._initialized = False
        self._session: Optional[ShellSession] = None
        self.diagnostics = PowerShellDiagnostics()

        # ── Çalışan alt süreç kaydı ──
        # Neden gerekli? Kullanıcı "Durdur" düğmesine bastığında `cancel()`
        # yalnızca bir bayrak kaldırıyordu; **çalışan komut öldürülmüyordu**.
        # 600 saniyelik bir komut, iptalden sonra da çalışmaya devam ediyor ve
        # kullanıcı bilgisayarından nasıl keseceğini bilemiyordu. Bu kayıt,
        # iptal isteğinin gerçekten süreci sonlandırmasını sağlar.
        self._running: Dict[int, Dict[str, Any]] = {}
        self._running_lock = threading.RLock()
        #: İptal isteği geldiğinde set edilir; `_run_bounded` bunu izler.
        self._cancel_event = threading.Event()

        self._counter = {
            "commands": 0, "blocked": 0, "timeouts": 0, "errors": 0,
            "sessions_started": 0, "session_commands": 0, "killed": 0,
        }
        #: İptal isteğinde çağrılacak ek geri çağrımlar (ör. süreç öldürme)
        self._cancel_hooks: List[Callable[[], Any]] = []

    def add_cancel_hook(self, hook: Callable[[], Any]) -> None:
        """
        İptal edildiğinde çağrılacak bir geri çağrım ekler.

        Ajan döngüsü bunu kullanır: "Durdur" basıldığında yalnızca bayrak
        kalkmakla kalmaz, çalışan komutlar da sonlandırılır.
        """
        if hook not in self._cancel_hooks:
            self._cancel_hooks.append(hook)

    def on_cancel(self) -> int:
        """
        İptal isteğini uygular: **çalışan tüm süreçleri öldürür**.

        `AgentLoop.cancel()` bu metodu çağırır. Aksi hâlde 600 saniyelik bir
        komut, iptalden sonra da çalışmaya devam ederdi.

        Returns:
            Öldürülen süreç sayısı
        """
        killed = self.kill_all("kullanıcı iptali")
        for hook in list(self._cancel_hooks):
            try:
                hook()
            except Exception as exc:
                logger.debug(f"İptal geri çağrımı hata verdi: {exc}")
        return killed

    # ─── Çalışan Süreç Yönetimi ─────────────────────────────────────────

    def _register_process(self, process: Any, command: str, kind: str) -> int:
        """
        Çalışan süreci kayda alır (arayüzde gösterilir, iptalde öldürülür).

        Args:
            process: `subprocess.Popen` nesnesi
            command: Kullanıcıya gösterilecek komut (kısaltılmış)
            kind: "shell" | "session" | "python"

        Returns:
            Süreç kimliği (PID)
        """
        entry = {
            "pid": getattr(process, "pid", 0),
            "command": (command or "")[:200],
            "kind": kind,
            "started": time.time(),
        }
        with self._running_lock:
            self._running[entry["pid"]] = entry
        self._emit("shell.process_started", entry)
        return entry["pid"]

    def _unregister_process(self, pid: int) -> None:
        """Süreci kayıttan düşürür."""
        with self._running_lock:
            entry = self._running.pop(pid, None)
        if entry is not None:
            entry["finished"] = time.time()
            entry["duration_ms"] = round(
                (entry["finished"] - entry["started"]) * 1000, 1)
            self._emit("shell.process_finished", entry)

    def running_processes(self) -> List[Dict[str, Any]]:
        """
        Şu anda çalışan alt süreçler.

        Arayüz bunu kullanarak "şu komut çalışıyor" göstergesini çizer ve
        kullanıcı tek tek durdurabilir.
        """
        with self._running_lock:
            items = list(self._running.values())
        now = time.time()
        for item in items:
            item["elapsed_seconds"] = round(now - item["started"], 1)
        return items

    def kill_all(self, reason: str = "iptal") -> int:
        """
        Çalışan **tüm** alt süreçleri (alt süreç ağacıyla) öldürür.

        Args:
            reason: Günlüğe yazılacak gerekçe

        Returns:
            Öldürülen süreç sayısı
        """
        self._cancel_event.set()
        with self._running_lock:
            entries = list(self._running.values())
        killed = 0
        for entry in entries:
            pid = int(entry.get("pid") or 0)
            if pid <= 0:
                continue
            try:
                self._kill_pid_tree(pid)
                killed += 1
                logger.warning(
                    f"Alt süreç öldürüldü ({reason}): {entry.get('command', '')[:80]}"
                )
            except Exception as exc:
                logger.debug(f"Süreç öldürülemedi (pid={pid}): {exc}")
            finally:
                self._unregister_process(pid)
        if killed:
            self._counter["killed"] += killed
            self._emit("shell.processes_killed", {"count": killed, "reason": reason})
        return killed

    @staticmethod
    def _kill_pid_tree(pid: int) -> None:
        """
        Bir süreci **tüm alt süreçleriyle** öldürür.

        Windows'ta `terminate()` çocuk süreçleri öldürmez; `powershell.exe`
        başlatıp içinde `python` çalıştırıldığında torun süreç ayakta kalır ve
        boruyu tutmaya devam eder. `taskkill /T /F` ağacın tamamını kapatır.
        """
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                os.kill(pid, signal.SIGKILL)

    def reset_cancel(self) -> None:
        """İptal bayrağını temizler (yeni görev öncesi)."""
        self._cancel_event.clear()

    @property
    def cancel_requested(self) -> bool:
        """İptal istendi mi?"""
        return self._cancel_event.is_set()

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, workspace: Optional[str] = None, config: Any = None,
                   event_bus: Any = None, sandbox: Any = None,
                   os_layer: Any = None) -> None:
        """Katmanı başlatır."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._sandbox = sandbox
        self._os_layer = os_layer

        root = workspace or getattr(config, "base_path", None) or os.getcwd()
        self._workspace = Path(root).expanduser().resolve()
        self._initialized = True
        logger.info(
            f"Kabuk katmanı hazır — çalışma alanı: {self._workspace}, "
            f"kabuk: {self.default_shell}"
        )

    def shutdown(self) -> None:
        """Oturumu kapatır ve çalışan süreçleri sonlandırır."""
        # Çalışan alt süreçleri öldür: aksi hâlde uygulama kapansa bile
        # arka planda komutlar (sunucular, derlemeler) çalışmaya devam eder.
        self.kill_all("kapatma")
        if self._session is not None:
            self._session.stop()
            self._session = None
        self._initialized = False
        logger.info(
            f"Kabuk katmanı kapatıldı ({self._counter['commands']} komut, "
            f"{self._counter['blocked']} engel, {self._counter['killed']} öldürme)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """İstatistikler (çalışan süreçler dâhil)."""
        return {
            **self._counter,
            "initialized": self._initialized,
            "workspace": str(self._workspace or ""),
            "session": self._session.get_stats() if self._session else None,
            "session_alive": bool(self._session and self._session.alive),
            "default_shell": self.default_shell,
            # Arayüz "şu komut çalışıyor" göstergesini buradan çizer
            "running": self.running_processes(),
            "cancel_requested": self._cancel_event.is_set(),
        }

    @property
    def workspace(self) -> Path:
        """Çalışma alanı kökü."""
        if self._workspace is None:
            raise RuntimeError("Kabuk katmanı başlatılmadı")
        return self._workspace

    @property
    def default_shell(self) -> str:
        """Kullanılabilir en iyi kabuk: `pwsh` > `powershell` > `cmd`."""
        if shutil.which("pwsh"):
            return "pwsh"
        if shutil.which("powershell") or os.name == "nt":
            return "powershell"
        return os.environ.get("SHELL", "/bin/sh")

    # ─── Kalıcı Oturum ──────────────────────────────────────────────────

    def start_session(self, shell: str = "", cwd: str = "",
                      env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Kalıcı kabuk oturumu başlatır.

        Args:
            shell: `pwsh` | `powershell` | `cmd` (boş = otomatik)
            cwd: Başlangıç dizini (boş = çalışma alanı)
            env: Ek ortam değişkenleri

        Returns:
            {"ok", "session_id", "shell", "cwd"}
        """
        if self._session is not None and self._session.alive:
            return {"ok": True, "session_id": self._session.session_id,
                    "shell": self._session.shell, "cwd": self._session.cwd,
                    "note": "oturum zaten açık"}

        workdir = str(self._workspace) if not cwd else cwd
        chosen = shell or ("powershell" if os.name == "nt" else self.default_shell)
        session = ShellSession(shell=chosen, cwd=workdir, env=env)
        if not session.start():
            self._counter["errors"] += 1
            return {"ok": False, "error": f"oturum başlatılamadı (kabuk: {chosen})"}

        self._session = session
        self._counter["sessions_started"] += 1
        # Oturumun alt sürecini de kayda al: iptal edildiğinde kapanmalı,
        # aksi hâlde arka planda çalışmaya devam eder.
        process = getattr(session, "_process", None)
        if process is not None:
            self._register_process(process, f"kalıcı oturum ({chosen})",
                                   kind="session")
        self._emit("shell.session_started", {
            "session_id": session.session_id, "shell": chosen, "cwd": workdir,
        })
        return {"ok": True, "session_id": session.session_id, "shell": chosen,
                "cwd": workdir}

    def stop_session(self) -> Dict[str, Any]:
        """Kalıcı oturumu kapatır ve sürecini kayıttan düşürür."""
        if self._session is None:
            return {"ok": True, "note": "açık oturum yok"}
        session_id = self._session.session_id
        process = getattr(self._session, "_process", None)
        pid = getattr(process, "pid", 0) if process is not None else 0
        self._session.stop()
        self._session = None
        if pid:
            self._unregister_process(pid)
        return {"ok": True, "session_id": session_id}

    def session_run(self, command: str, timeout: float = 120.0,
                    *, check_dangerous: bool = True) -> Dict[str, Any]:
        """
        Kalıcı oturumda komut çalıştırır (durum korunur).

        Args:
            command: Komut
            timeout: Saniye
            check_dangerous: Yıkıcı komut denetimi

        Returns:
            `CommandResult.to_dict()`
        """
        if self._session is None or not self._session.alive:
            started = self.start_session()
            if not started.get("ok"):
                return {"ok": False, "error": started.get("error", "oturum yok")}

        if check_dangerous:
            blocked, reason = self._is_blocked(command)
            if blocked:
                self._counter["blocked"] += 1
                self._emit("shell.blocked", {"command": command[:200], "reason": reason})
                return CommandResult(command=command, returncode=-1, blocked=True,
                                     blocked_reason=reason,
                                     session_id=self._session.session_id).to_dict()

        assert self._session is not None
        result = self._session.run(command, timeout=timeout)
        self._counter["session_commands"] += 1
        if result.timed_out:
            self._counter["timeouts"] += 1
        self._emit("shell.session_command", {
            "command": command[:200], "ok": result.ok,
            "returncode": result.returncode, "duration_ms": result.duration_ms,
        })
        return result.to_dict()

    # ─── Tek Seferlik Komut ─────────────────────────────────────────────

    def run(self, command: str, cwd: str = "", timeout: float = 120.0,
            shell: str = "", *, check_dangerous: bool = True,
            capture_bytes: bool = False) -> CommandResult:
        """
        Tek seferlik komut çalıştırır (yeni süreç, durum korunmaz).

        Args:
            command: Komut satırı
            cwd: Çalışma dizini (boş = çalışma alanı)
            timeout: Saniye
            shell: `powershell` | `cmd` | `bash` (boş = otomatik)
            check_dangerous: Yıkıcı komut denetimi
            capture_bytes: Çıktıyı bayt olarak al (kodlama sorunları için)

        Returns:
            `CommandResult`
        """
        kind, reason = classify_command(command)
        result = CommandResult(command=command, command_class=kind.value)

        if check_dangerous:
            blocked, why = self._is_blocked(command)
            if blocked:
                self._counter["blocked"] += 1
                result.blocked = True
                result.blocked_reason = why
                result.returncode = -1
                self._emit("shell.blocked", {"command": command[:200], "reason": why})
                return result

        workdir = Path(cwd) if cwd else self.workspace
        if not workdir.exists():
            result.returncode = -1
            result.stderr = f"çalışma dizini yok: {workdir}"
            self._counter["errors"] += 1
            return result

        # ── OS katmanı varsa ondan geç (denetim kaydı için) ──
        if self._os_layer is not None:
            try:
                outcome = self._os_layer.run(command, cwd=str(workdir))
                if isinstance(outcome, dict):
                    result.returncode = int(outcome.get("returncode", -1))
                    result.stdout = str(outcome.get("stdout", ""))
                    result.stderr = str(outcome.get("stderr", ""))
                    result.duration_ms = float(outcome.get("duration_ms", 0.0) or 0.0)
                    result.timed_out = bool(outcome.get("timed_out"))
                    result.blocked = bool(outcome.get("blocked"))
                    result.blocked_reason = str(outcome.get("blocked_reason", ""))
                    result.cwd = str(workdir)
                    result.shell = "os_layer"
                    self._counter["commands"] += 1
                    return result
            except Exception as exc:
                logger.debug(f"OS katmanı yönlendirmesi başarısız: {exc}")

        chosen = shell or ("powershell" if os.name == "nt" else "bash")
        started = time.perf_counter()
        environment = dict(os.environ)
        environment["PYTHONIOENCODING"] = "utf-8"

        if chosen in ("powershell", "pwsh") and os.name == "nt":
            executable = shutil.which(chosen) or \
                os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32",
                             "WindowsPowerShell", "v1.0", "powershell.exe")
            arguments = [executable, "-NoLogo", "-NoProfile", "-NonInteractive",
                         "-ExecutionPolicy", "Bypass", "-Command", command]
        elif chosen == "cmd" and os.name == "nt":
            arguments = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", command]
        else:
            arguments = [chosen, "-c", command]

        try:
            bucket = self._run_bounded(
                arguments, workdir=str(workdir), environment=environment,
                timeout=max(1.0, float(timeout)), capture_bytes=capture_bytes,
                command=command, kind=chosen,
            )
            result.duration_ms = (time.perf_counter() - started) * 1000
            result.returncode = bucket["returncode"]
            result.stdout = bucket["stdout"]
            result.stderr = bucket["stderr"]
            result.timed_out = bucket["timed_out"]
            result.cancelled = bool(bucket.get("cancelled", False))
            result.truncated = len(result.stdout) > 20000 or len(result.stderr) > 20000
            result.cwd = str(workdir)
            result.shell = chosen
            if result.timed_out:
                self._counter["timeouts"] += 1
        except OSError as exc:
            result.duration_ms = (time.perf_counter() - started) * 1000
            result.returncode = -1
            result.stderr = f"başlatma hatası: {exc}"
            self._counter["errors"] += 1

        self._counter["commands"] += 1
        self._emit("shell.command", {
            "command": command[:200], "ok": result.ok,
            "returncode": result.returncode,
            "duration_ms": result.duration_ms, "class": kind.value,
        })
        return result

    def _run_bounded(self, arguments: List[str], *, workdir: str,
                     environment: Dict[str, str], timeout: float,
                     capture_bytes: bool = False,
                     command: str = "", kind: str = "shell") -> Dict[str, Any]:
        """
        Bir alt süreci **kesin** bir duvar saati sınırıyla çalıştırır.

        Neden `subprocess.run(timeout=...)` yetmiyor?
            Windows'ta alt süreç bir **torun** doğurduğunda (ör.
            `python → powershell → blender.exe` / `→ Godot.exe`) ve o torun
            ayakta kalırsa, alt süreç öldürülse bile **stdout borusu açık
            kalır**. `communicate()` boru kapanana kadar bekler, dolayısıyla
            `timeout` **aşılsa bile çağrı dönmez**. Ölçüldü: 8 saniyelik sınır
            59.6 saniyede tetiklendi; gerçekten takılan bir Blender'da ajan
            hiç dönmedi.

        Çözüm üç katmanlıdır:
            1. Çıktı **ayrı iş parçacıklarında** okunur (boru tıkanması olmaz).
            2. Süre dolduğunda **süreç ağacı** `taskkill /T /F` ile öldürülür
               (yalnızca `kill()` torunları sağ bırakır).
            3. Borular **kapatılır**, böylece `join` asla asılı kalmaz.

        Returns:
            {"returncode", "stdout", "stderr", "timed_out"}
        """
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                arguments, cwd=workdir, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, shell=False,
                creationflags=creation_flags,
            )
        except OSError as exc:
            return {"returncode": -1, "stdout": "",
                    "stderr": f"başlatma hatası: {exc}", "timed_out": False}

        chunks: Dict[str, List[bytes]] = {"stdout": [], "stderr": []}

        # Süreci kayda al: arayüzde "çalışıyor" göstergesi ve iptalde
        # gerçekten öldürme bu kayıt üzerinden yapılır.
        pid = self._register_process(process, command, kind=kind)
        self._emit("shell.command_started", {
            "command": (command or "")[:200], "kind": kind, "pid": pid,
        })

        def drain(stream: Any, key: str) -> None:
            """Boru boşalana kadar okur (ayrı iş parçacığı)."""
            try:
                for block in iter(lambda: stream.read(8192), b""):
                    chunks[key].append(block)
            except (OSError, ValueError):
                pass
            finally:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

        readers = [
            threading.Thread(target=drain, args=(process.stdout, "stdout"),
                             name="ShellOutReader", daemon=True),
            threading.Thread(target=drain, args=(process.stderr, "stderr"),
                             name="ShellErrReader", daemon=True),
        ]
        for reader in readers:
            reader.start()

        timed_out = False
        cancelled = False
        # ── İptal-farkında bekleme ──
        # `process.wait(timeout=...)` tek seferlik bekler ve iptal isteğini
        # **görmez**. Kısa dilimlerle bekleyip her dilimde iptal bayrağını
        # denetlemek, "Durdur" düğmesinin çalışan komutu gerçekten kesmesini
        # sağlar.
        wait_slice = 0.25
        waited = 0.0
        try:
            while True:
                try:
                    process.wait(timeout=wait_slice)
                    returncode = process.returncode
                    break
                except subprocess.TimeoutExpired:
                    waited += wait_slice
                    if self._cancel_event.is_set():
                        cancelled = True
                        self._kill_tree(process)
                        returncode = -1
                        break
                    if waited >= timeout:
                        timed_out = True
                        self._kill_tree(process)
                        returncode = -1
                        break
        finally:
            self._unregister_process(pid)

        # Okuyucuların bitmesini **sınırlı** süre bekle; torun süreç boruyu
        # açık tutuyorsa sonsuza kadar beklemeyiz.
        deadline = time.time() + 1.5
        for reader in readers:
            reader.join(timeout=max(0.0, deadline - time.time()))

        def decode(key: str) -> str:
            raw = b"".join(chunks[key])
            if capture_bytes:
                return raw.decode("utf-8", "replace")
            return raw.decode("utf-8", "replace")

        stdout = decode("stdout")
        stderr = decode("stderr")
        if cancelled:
            stderr = ("kullanıcı iptal etti — süreç ağacı sonlandırıldı"
                      + (f"\n{stderr}" if stderr else ""))
        elif timed_out:
            stderr = (f"zaman aşımı ({timeout:.0f}s) — süreç ağacı sonlandırıldı"
                      + (f"\n{stderr}" if stderr else ""))
        return {"returncode": returncode, "stdout": stdout, "stderr": stderr,
                "timed_out": timed_out, "cancelled": cancelled}

    @staticmethod
    def _kill_tree(process: Any) -> None:
        """
        Süreci **ve tüm torunlarını** sonlandırır.

        `process.kill()` yalnızca doğrudan çocuğu öldürür; Godot/Blender gibi
        torunlar ayakta kalır ve boruyu tutmaya devam eder. Windows'ta
        `taskkill /T` ağacın tamamını kapatır.
        """
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, timeout=15,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                process.kill()
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug(f"Süreç ağacı sonlandırılamadı: {exc}")
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass

    def run_powershell(self, command: str, **kwargs: Any) -> CommandResult:
        """PowerShell komutu (kısa yol)."""
        return self.run(command, shell="powershell", **kwargs)

    def run_cmd(self, command: str, **kwargs: Any) -> CommandResult:
        """CMD komutu (kısa yol)."""
        return self.run(command, shell="cmd", **kwargs)

    def run_python(self, code: str, timeout: float = 120.0,
                   cwd: str = "", args: Sequence[str] = ()) -> CommandResult:
        """
        Python kodu çalıştırır.

        Çok satırlı kod veya içinde tırnak bulunan kod, PowerShell'in `-Command`
        ayrıştırıcısını bozar (satır sonu yeni bir ifade sayılır). Bu yüzden kod
        **her zaman geçici bir dosyaya** yazılır ve `python <dosya>` ile
        çalıştırılır. Bu yaklaşım aynı zamanda gerçek satır numaralarıyla
        izleme kaydı (traceback) üretir — `-c` ile üretilmez.
        """
        import tempfile

        path = Path(tempfile.gettempdir()) / f"nc_snippet_{int(time.time() * 1000)}.py"
        try:
            path.write_text(code, encoding="utf-8")
            argument_text = " ".join(f"'{item}'" for item in args)
            command = (
                f"& '{sys.executable}' -X utf8 '{path}' {argument_text}".strip()
            )
            return self.run(command, timeout=timeout, cwd=cwd)
        finally:
            try:
                path.unlink()
            except OSError:
                pass

    # ─── Güvenlik ───────────────────────────────────────────────────────

    def _is_blocked(self, command: str) -> Tuple[bool, str]:
        """Komut engellenmeli mi? (sandbox + bağımsız denetim)"""
        kind, reason = classify_command(command)
        if kind == CommandClass.DANGEROUS:
            return True, f"yıkıcı komut engellendi: {reason}"

        if self._sandbox is not None:
            try:
                check = getattr(self._sandbox, "is_dangerous", None)
                if callable(check):
                    dangerous, why = check(command)
                    if dangerous:
                        return True, str(why)
                policy = getattr(self._sandbox, "policy", None)
                if policy is not None:
                    allowed, why = self._sandbox.check_command(command)
                    if not allowed:
                        return True, str(why)
            except Exception as exc:
                logger.debug(f"Sandbox denetimi atlandı: {exc}")

        return False, ""

    # ─── Teşhis ─────────────────────────────────────────────────────────

    def diagnose(self, text: str, command: str = "") -> Dict[str, Any]:
        """Komut çıktısını yorumlar (kısa yol)."""
        return self.diagnostics.analyze(text, command)

    def suggest_fix(self, text: str, command: str = "") -> Dict[str, Any]:
        """Düzeltme önerisi (kısa yol)."""
        return self.diagnostics.suggest_fix(text, command)

    def toolchain_versions(self) -> Dict[str, Any]:
        """
        Kurulu araç zincirinin sürümlerini toplar.

        Ajanın "hangi sürümle çalışıyorum?" sorusunu yanıtlar; komut seçimini
        (ör. `python` vs `py`, `pwsh` vs `powershell`) etkiler.

        Not: Sondaj komutları **argüman listesi** olarak değil, kabuğa yazılan
        metin olarak çalışır. Bu yüzden tırnak içindeki yollar tek tırnakla
        yazılır ve `shutil.which` başarısız olduğunda tam yol denenir.
        """
        python_exe = sys.executable
        powershell_exe = self._resolve_powershell()
        pwsh_exe = shutil.which("pwsh")
        godot_exe = self._find_executable("godot")
        blender_exe = self._find_executable("blender")

        def version_command(executable: str, block: str = "") -> str:
            """
            İç içe çalıştırma için güvenli komut üretir.

            ÖNEMLİ: `-Command` argümanı **betik bloğu** (`{...}`) olmalıdır.
            Düz bir ifade verilirse dış kabuk onu kendi ayrıştırıcısından
            geçirir ve `$PSVersionTable.PSVersion` genişletilip bozulur —
            komut çıkış kodu 0 ile **boş çıktı** verir, hata da vermez.
            """
            if not executable:
                return ""
            if block:
                return f"& '{executable}' -NoProfile -Command {{{block}}}"
            return f"& '{executable}' --version"

        probes: List[Tuple[str, Optional[str], str]] = [
            ("python", python_exe, f"& '{python_exe}' --version"),
            ("pip", python_exe, f"& '{python_exe}' -m pip --version"),
            ("powershell", powershell_exe,
             version_command(powershell_exe, "$PSVersionTable.PSVersion.ToString()")),
            ("pwsh", pwsh_exe,
             version_command(pwsh_exe, "$PSVersionTable.PSVersion.ToString()")),
            ("git", shutil.which("git"), "git --version"),
            ("node", shutil.which("node"), "node --version"),
            ("dotnet", shutil.which("dotnet"), "dotnet --version"),
            ("godot", godot_exe, version_command(godot_exe)),
            ("blender", blender_exe, version_command(blender_exe)),
            ("nvidia-smi", shutil.which("nvidia-smi"),
             "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
        ]

        versions: Dict[str, Any] = {}
        for name, executable, command in probes:
            if not executable or not command:
                versions[name] = {"available": False, "path": ""}
                continue
            outcome = self.run(command, timeout=20.0, check_dangerous=False)
            output = ((outcome.stdout or "") + (outcome.stderr or "")).strip()
            # PowerShell hata metni "sürüm" sayılmasın
            looks_like_error = bool(re.search(
                r"At line:|CategoryInfo|FullyQualifiedErrorId|is not recognized|"
                r"CommandNotFoundException|Cannot find",
                output, re.IGNORECASE,
            ))
            first_line = output.splitlines()[0][:140] if output else ""
            versions[name] = {
                "available": bool(output) and not outcome.timed_out
                and not looks_like_error and outcome.returncode == 0,
                "version": "" if looks_like_error else first_line,
                "path": str(executable)[:200],
            }
        return versions

    def _resolve_powershell(self) -> str:
        """
        PowerShell çalıştırılabilirinin tam yolunu bulur.

        `shutil.which` Windows'ta `powershell.exe`'i bulamayabilir çünkü
        Windows PowerShell PATH'te her zaman görünmez; bu yüzden bilinen
        sistem yolu da denenir.
        """
        found = shutil.which("pwsh") or shutil.which("powershell")
        if found:
            return found
        if os.name == "nt":
            system_root = os.environ.get("SystemRoot") or r"C:\Windows"
            candidates = [
                os.path.join(system_root, "System32", "WindowsPowerShell",
                             "v1.0", "powershell.exe"),
                os.path.join(system_root, "SysWOW64", "WindowsPowerShell",
                             "v1.0", "powershell.exe"),
                r"C:\Program Files\PowerShell\7\pwsh.exe",
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
        return ""

    def _find_executable(self, name: str) -> str:
        """
        Bir aracı PATH'te ve bilinen kurulum dizinlerinde arar.

        Godot ve Blender gibi araçlar genellikle PATH'e eklenmez; bu yüzden
        yaygın kurulum konumları taranır. Arama **yalnızca bilinen köklerde**
        ve sınırlı derinlikte yapılır (tüm diski taramak dakikalar sürer).
        """
        found = shutil.which(name)
        if found:
            return found

        search_roots = [
            r"E:\AppsTools", r"D:\AppsTools", r"C:\AppsTools",
            r"C:\Program Files", r"C:\Program Files (x86)",
            str(Path.home() / "Apps"), str(Path.home() / "Downloads"),
        ]
        patterns: List[Tuple[str, bool, bool]] = []
        if name == "godot":
            patterns = [("Godot*.exe", False, True), ("Godot*.exe", True, True)]
        elif name == "blender":
            patterns = [("blender.exe", True, False),
                        ("Blender*/blender.exe", False, False)]

        for root in search_roots:
            base = Path(root)
            if not base.is_dir():
                continue
            for pattern, recursive, exclude_console in patterns:
                for candidate in _glob_limited(base, pattern, recursive,
                                               exclude_console, limit=3):
                    if os.path.exists(candidate):
                        return candidate
        return ""

    # ─── Araç Tanımları ─────────────────────────────────────────────────

    def tools(self) -> List[ToolDef]:
        """Kabuk araçları (`ToolRegistry` için)."""
        shell = self

        def handler_for(method: Callable[..., Any]):
            def handler(ctx: ToolContext, **kwargs: Any) -> Any:
                if ctx.cancelled():
                    raise RuntimeError("komut kullanıcı tarafından iptal edildi")
                return method(**kwargs)
            return handler

        from neurocore.agent.tools import ToolParameter as P

        def run_dict(command: str, cwd: str = "", timeout: float = 120.0) -> Dict[str, Any]:
            """Tek seferlik komut → sözlük."""
            return shell.run(command, cwd=cwd, timeout=timeout).to_dict()

        def session_dict(command: str, timeout: float = 120.0) -> Dict[str, Any]:
            """Oturum komutu → sözlük (teşhis de eklenir)."""
            data = shell.session_run(command, timeout=timeout)
            if data.get("blocked"):
                return data
            if not data.get("ok"):
                diagnosis = shell.diagnose(
                    f"{data.get('stdout', '')}\n{data.get('stderr', '')}", command
                )
                if diagnosis.get("errors"):
                    data["diagnosis"] = diagnosis
                    data["fix"] = shell.suggest_fix(
                        f"{data.get('stdout', '')}\n{data.get('stderr', '')}", command
                    )
            return data

        definitions = [
            ("shell.run",
             "PowerShell komutu çalıştırır (yeni süreç). Çıktı, çıkış kodu ve "
             "hata analizi döner. `cd` kalıcı değildir — durum korumak için "
             "shell.session_run kullanın.",
             Safety.EXECUTE,
             [P("command", required=True, description="PowerShell komutu"),
              P("cwd", description="Çalışma dizini (boş = proje kökü)"),
              P("timeout", "number", "Saniye", default=120)],
             run_dict,
             "komut çıktısı"),

            ("shell.session_run",
             "KALICI PowerShell oturumunda komut çalıştırır: `cd`, ortam "
             "değişkenleri ve değişkenler sonraki komutlarda korunur. Sıralı "
             "işler (kurulum, derleme) için bunu kullanın. Başarısız olursa "
             "yanıta hata teşhisi ve düzeltme önerisi eklenir.",
             Safety.EXECUTE,
             [P("command", required=True, description="PowerShell komutu"),
              P("timeout", "number", "Saniye", default=120)],
             session_dict,
             "komut çıktısı (durum korunur)"),

            ("shell.cmd",
             "CMD (komut istemi) komutu çalıştırır. Yalnızca CMD'ye özgü "
             "komutlar (ör. bazı kurulum araçları) için gerekir.",
             Safety.EXECUTE,
             [P("command", required=True, description="CMD komutu"),
              P("cwd", description="Çalışma dizini"),
              P("timeout", "number", "Saniye", default=120)],
             lambda command, cwd="", timeout=120.0: shell.run_cmd(
                 command, cwd=cwd, timeout=timeout).to_dict(),
             "komut çıktısı"),

            ("shell.python",
             "Python kodu çalıştırır (çok satırlı kod desteklenir).",
             Safety.EXECUTE,
             [P("code", required=True, description="Python kodu"),
              P("timeout", "number", "Saniye", default=120)],
             lambda code, timeout=120.0: shell.run_python(code, timeout=timeout).to_dict(),
             "python çıktısı"),

            ("shell.diagnose",
             "Komut çıktısındaki hatayı yorumlar: tür, satır ve düzeltme önerisi.",
             Safety.READ,
             [P("text", required=True, description="Komut çıktısı"),
              P("command", description="Çalıştırılan komut (bağlam)")],
             lambda text, command="": {
                 "analysis": shell.diagnose(text, command),
                 "fix": shell.suggest_fix(text, command),
             },
             "hata analizi ve öneri"),

            ("shell.versions",
             "Kurulu araç zincirinin sürümlerini listeler (python, git, godot, "
             "blender, node, ...). Hangi komutun kullanılabilir olduğunu gösterir.",
             Safety.READ,
             [],
             shell.toolchain_versions,
             "araç sürümleri"),

            ("shell.start_session",
             "Kalıcı PowerShell oturumu başlatır (gerekirse otomatik başlar).",
             Safety.EXECUTE,
             [P("cwd", description="Başlangıç dizini")],
             lambda cwd="": shell.start_session(cwd=cwd),
             "oturum bilgisi"),
        ]

        tools: List[ToolDef] = []
        for name, description, safety, parameters, method, returns in definitions:
            tools.append(ToolDef(
                name=name, description=description,
                handler=handler_for(method), parameters=parameters,
                safety=safety, category="shell", returns=returns,
                tags=["kabuk", "powershell"],
            ))
        return tools

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<ShellIntegration shell={self.default_shell} "
                f"commands={self._counter['commands']} "
                f"session={'açık' if self._session else 'kapalı'}>")


__all__ = [
    "ShellIntegration",
    "ShellSession",
    "PowerShellDiagnostics",
    "CommandResult",
    "CommandClass",
    "CLASS_SAFETY",
    "CLASS_LABELS",
    "classify_command",
    "command_safety",
]


def _glob_limited(root: Path, pattern: str, recursive: bool,
                  exclude_console: bool = False, limit: int = 4) -> List[str]:
    """
    Bir kökte kalıba uyan çalıştırılabilirleri **sınırlı** biçimde bulur.

    Godot/Blender gibi araçlar PATH'e eklenmez ve kurulum dizinleri derin
    olabilir. Tüm diski taramak dakikalar sürer; bu yüzden derinlik ve sonuç
    sayısı sınırlanır. Godot'un `_console.exe` varyantı isteğe bağlı olarak
    dışlanır (GUI sürümü tercih edilir).
    """
    results: List[str] = []
    if not root.is_dir():
        return results
    target_name = Path(pattern).name

    def consider(candidate: Path) -> bool:
        if not candidate.is_file():
            return False
        if exclude_console and "_console" in candidate.name.lower():
            return False
        results.append(str(candidate))
        return len(results) >= limit

    try:
        if recursive:
            base_depth = len(root.parts)
            for current, directories, filenames in os.walk(root):
                if len(Path(current).parts) - base_depth > 3:
                    directories[:] = []
                    continue
                for filename in sorted(filenames):
                    if fnmatch.fnmatch(filename, target_name) and \
                            consider(Path(current) / filename):
                        return results
        else:
            for candidate in sorted(root.glob(pattern)):
                if consider(candidate):
                    return results
    except (OSError, ValueError):
        return results
    return results
