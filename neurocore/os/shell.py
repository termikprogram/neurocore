"""
NeuroCore İşletim Sistemi — Kabuk Çalıştırıcı (Shell)
=====================================================
L5 (OS) katmanının **komut yürütme** cephesi. Tüm komutlar sandbox
politikasından geçer, denetim günlüğüne yazılır ve **her zaman** zaman
aşımına tabidir.

Güvenlik kuralları:
    - `sandbox.allow_shell=False` ise `SandboxViolation` fırlatılır.
    - Yıkıcı komutlar (`format`, `rd /s`, `Remove-Item -Recurse`, …)
      `check_dangerous=True` iken **çalıştırılmaz**; `blocked=True` içeren bir
      `CommandResult` döner (istisna fırlatılmaz, karar çağırana bırakılır).
    - `sandbox.dry_run` (veya `dry_run=True`) ise komut hiç çalıştırılmaz.
    - Zaman aşımı: verilmezse `sandbox.max_command_seconds`, üst sınır 1800 sn.
    - Liste komutlar (`list[str]`) asla `shell=True` ile çalıştırılmaz.

Kullanım:
    sh = ShellRunner()
    sh.initialize(config, event_bus)
    sonuc = sh.run("echo merhaba")
    print(sonuc.ok, sonuc.stdout)
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from neurocore.os.sandbox import Sandbox, SandboxViolation
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("os.shell")

#: Hiçbir komut bu süreden uzun çalışamaz (saniye)
HARD_TIMEOUT_SECONDS = 1800.0


def _no_window_flags() -> int:
    """Alt süreçlerde konsol penceresi açılmasın."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def split_command(command: str) -> List[str]:
    """
    Komut metnini argüman listesine ayırır.

    Windows'ta `shlex` POSIX olmayan kipte tırnakları soymadığı için
    (``"C:\\yol\\program.exe" -c "kod"`` → tırnaklar kalır) fazladan bir
    tırnak temizliği uygulanır. Bu, `shell=False` ile çalıştırılan metin
    komutların Windows'ta da doğru ayrıştırılmasını sağlar.

    Args:
        command: Komut metni.

    Returns:
        Argüman listesi (boş parçalar atılır).
    """
    metin = str(command)
    try:
        parcalar: List[str] = shlex.split(metin, posix=(os.name != "nt"))
    except ValueError:
        parcalar = metin.split()

    if os.name != "nt":
        return parcalar

    temiz: List[str] = []
    for parca in parcalar:
        if len(parca) >= 2 and parca[0] == parca[-1] and parca[0] in ("'", '"'):
            parca = parca[1:-1]
        if parca != "":
            temiz.append(parca)
    return temiz


# ─── Komut Sonucu ──────────────────────────────────────────────────────────

@dataclass
class CommandResult:
    """
    Tek bir komut çalıştırmasının sonucu.

    Attributes:
        command: Çalıştırılan komut metni.
        returncode: Çıkış kodu (-1: engellendi/hata/zaman aşımı).
        stdout: Standart çıktı.
        stderr: Standart hata.
        duration_ms: Süre (ms).
        timed_out: Zaman aşımına uğradı mı?
        blocked: Sandbox tarafından engellendi mi?
        dry_run: Yalnızca simüle edildi mi?
        cwd: Çalışma dizini.
    """

    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    blocked: bool = False
    dry_run: bool = False
    cwd: str = ""

    @property
    def ok(self) -> bool:
        """Komut başarıyla mı tamamlandı? (kuru çalıştırma ve engel başarı sayılmaz)"""
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.blocked
            and not self.dry_run
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dict."""
        return {
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "blocked": self.blocked,
            "dry_run": self.dry_run,
            "cwd": self.cwd,
            "ok": self.ok,
        }


# ─── Ana Kabuk Çalıştırıcı ─────────────────────────────────────────────────

class ShellRunner:
    """
    Sandbox denetimli komut yürütücü.

    `sandbox` özniteliği dışarıdan atanabilir ya da `attach_sandbox()` ile
    bağlanabilir; yoksa tembel olarak varsayılan bir `Sandbox` kurulur.
    Her çağrı `audit` günlüğüne kaydedilir (audit bağlıysa).
    """

    name = "os.shell"

    def __init__(self) -> None:
        self.sandbox: Optional[Sandbox] = None
        self.audit: Any = None
        self._initialized: bool = False
        self._config: Any = None
        self._event_bus: Any = None
        self._counters: Dict[str, int] = defaultdict(int)
        self._last_error: str = ""

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Bağımlılıkları bağlar; sandbox yoksa varsayılanını kurar (idempotent)."""
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus

        if self.sandbox is None:
            kok = getattr(config, "base_path", None) if config is not None else None
            sandbox = Sandbox(str(kok) if kok else None)
            sandbox.initialize(config, event_bus)
            self.sandbox = sandbox

        self._initialized = True
        logger.info(
            "Kabuk çalıştırıcı hazır — kabuk izni: %s | üst süre: %.0f sn",
            self.sandbox.policy.allow_shell,
            self.sandbox.policy.max_command_seconds,
        )

    def shutdown(self) -> None:
        """Durumu serbest bırakır (idempotent). Çalışan alt süreçler etkilenmez."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info("Kabuk çalıştırıcı kapatıldı")

    def attach_sandbox(self, sandbox: Sandbox) -> Sandbox:
        """Paylaşılan sandbox örneğini bağlar."""
        self.sandbox = sandbox
        logger.debug("Kabuk çalıştırıcıya sandbox bağlandı: %s", sandbox.root)
        return sandbox

    def attach_audit(self, audit: Any) -> Any:
        """Denetim günlüğünü bağlar (opsiyonel)."""
        self.audit = audit
        return audit

    def _require_sandbox(self) -> Sandbox:
        """Sandbox'ı döndürür; yoksa tembel olarak oluşturur."""
        if self.sandbox is None:
            sandbox = Sandbox()
            sandbox.initialize(self._config, self._event_bus)
            self.sandbox = sandbox
        return self.sandbox

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Olay yayınlar (event_bus yoksa sessizce atlar)."""
        if not self._event_bus:
            return
        try:
            self._event_bus.emit(event, data, source=self.name)
        except Exception as exc:
            logger.debug("Olay yayınlanamadı (%s): %s", event, exc)

    def _record(self, target: str, ok: bool, detail: str = "", sure_ms: float = 0.0) -> None:
        """Denetim günlüğüne kayıt düşer (audit yoksa no-op)."""
        if not self.audit:
            return
        try:
            self.audit.record("shell.run", target, ok=ok, detail=detail, duration_ms=sure_ms, source=self.name)
        except Exception as exc:
            logger.debug("Denetim kaydı yazılamadı: %s", exc)

    # ─── Komut Çalıştırma ───────────────────────────────────────────────

    def run(
        self,
        command: Union[str, Sequence[str]],
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        shell: Optional[bool] = None,
        env: Optional[Dict[str, str]] = None,
        dry_run: bool = False,
        check_dangerous: bool = True,
    ) -> CommandResult:
        """
        Komut çalıştırır.

        Args:
            command: Komut metni ya da argüman listesi.
            cwd: Çalışma dizini (sandbox denetiminden geçer).
            timeout: Zaman aşımı (sn). None → `sandbox.max_command_seconds`.
            shell: True/False; None → metin komutlarda Windows'ta True, listede daima False.
            env: Ek ortam değişkenleri.
            dry_run: True ise komut çalıştırılmaz, simülasyon sonucu döner.
            check_dangerous: Yıkıcı komut denetimi yapılsın mı?

        Returns:
            `CommandResult`

        Raises:
            SandboxViolation: Kabuk izni kapalıysa (`allow_shell=False`).
        """
        baslangic = time.perf_counter()
        sandbox = self._require_sandbox()
        gosterim = command if isinstance(command, str) else " ".join(str(p) for p in command)
        gosterim = str(gosterim)

        # ─── Kabuk izni ───
        if not sandbox.policy.allow_shell:
            self._counters["denied"] += 1
            raise SandboxViolation("Kabuk çalıştırma izni kapalı (allow_shell=False)")

        # ─── Yıkıcı komut denetimi ───
        if check_dangerous:
            izinli, gerekce = sandbox.check_command(gosterim)
            if not izinli:
                self._counters["blocked"] += 1
                mesaj = f"Komut engellendi: {gerekce}. Bu işlem yıkıcı olduğu için çalıştırılmadı."
                logger.warning("Komut engellendi (%s): %s", gerekce, gosterim[:200])
                sonuc = CommandResult(
                    command=gosterim,
                    returncode=-1,
                    stdout="",
                    stderr=mesaj,
                    duration_ms=(time.perf_counter() - baslangic) * 1000,
                    blocked=True,
                    cwd=str(cwd or ""),
                )
                self._record(gosterim[:300], ok=False, detail=f"engellendi: {gerekce}")
                self._emit("os.shell_blocked", {"command": gosterim[:200], "reason": gerekce})
                return sonuc

        # ─── Kuru çalıştırma ───
        if dry_run or sandbox.policy.dry_run:
            sonuc = CommandResult(
                command=gosterim,
                returncode=0,
                stdout=f"[kuru çalıştırma] {gosterim}",
                stderr="",
                duration_ms=(time.perf_counter() - baslangic) * 1000,
                dry_run=True,
                cwd=str(cwd or ""),
            )
            self._counters["dry_run"] += 1
            logger.info("[kuru çalıştırma] Komut çalıştırılmadı: %s", gosterim[:200])
            self._record(gosterim[:300], ok=True, detail="kuru çalıştırma")
            return sonuc

        # ─── Komut biçimi ───
        if isinstance(command, (list, tuple)):
            argv: Union[str, List[str]] = [str(p) for p in command]
            kullan_shell = False  # liste komutlar kabuğa verilmez
        else:
            argv = str(command)
            kullan_shell = (os.name == "nt") if shell is None else bool(shell)
            if not kullan_shell:
                try:
                    argv = split_command(str(command))
                except Exception as exc:
                    self._fail(f"Komut ayrıştırılamadı: {exc}")
                    return CommandResult(
                        command=gosterim,
                        returncode=-1,
                        stdout="",
                        stderr=f"Komut ayrıştırılamadı: {exc}",
                        duration_ms=(time.perf_counter() - baslangic) * 1000,
                        cwd=str(cwd or ""),
                    )

        # ─── Çalışma dizini ───
        calisma_dizini: Optional[str] = None
        if cwd:
            try:
                calisma_dizini = str(sandbox.resolve_path(cwd, must_exist=True))
            except SandboxViolation:
                raise
            except Exception as exc:
                logger.warning("Çalışma dizini geçersiz (%s): %s", cwd, exc)
                calisma_dizini = None

        # ─── Zaman aşımı (her zaman uygulanır, üst sınır 1800 sn) ───
        sure = self._resolve_timeout(timeout, sandbox)

        # ─── Ortam ───
        ortam: Optional[Dict[str, str]] = None
        if env:
            try:
                ortam = dict(os.environ)
                ortam.update({str(k): str(v) for k, v in env.items()})
            except Exception as exc:
                logger.debug("Ortam değişkenleri hazırlanamadı: %s", exc)
                ortam = None

        try:
            tamamlanan = subprocess.run(
                argv,
                cwd=calisma_dizini,
                env=ortam,
                shell=kullan_shell,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=sure,
                creationflags=_no_window_flags(),
            )
        except subprocess.TimeoutExpired as exc:
            sure_ms = (time.perf_counter() - baslangic) * 1000
            self._counters["timeouts"] += 1
            self._fail(f"Komut zaman aşımı ({sure:.0f} sn): {gosterim[:200]}")
            sonuc = CommandResult(
                command=gosterim,
                returncode=-1,
                stdout=self._metin(exc.stdout),
                stderr=f"Zaman aşımı ({sure:.0f} sn) — süreç öldürüldü",
                duration_ms=sure_ms,
                timed_out=True,
                cwd=calisma_dizini or os.getcwd(),
            )
            self._record(gosterim[:300], ok=False, detail="zaman aşımı", sure_ms=sure_ms)
            self._emit("os.shell_timeout", {"command": gosterim[:200], "timeout": sure})
            return sonuc
        except SandboxViolation:
            raise
        except Exception as exc:
            sure_ms = (time.perf_counter() - baslangic) * 1000
            self._fail(f"Komut çalıştırılamadı: {exc}")
            sonuc = CommandResult(
                command=gosterim,
                returncode=-1,
                stdout="",
                stderr=str(exc),
                duration_ms=sure_ms,
                cwd=calisma_dizini or os.getcwd(),
            )
            self._record(gosterim[:300], ok=False, detail=str(exc), sure_ms=sure_ms)
            return sonuc

        sure_ms = (time.perf_counter() - baslangic) * 1000
        sonuc = CommandResult(
            command=gosterim,
            returncode=int(tamamlanan.returncode),
            stdout=tamamlanan.stdout or "",
            stderr=tamamlanan.stderr or "",
            duration_ms=sure_ms,
            cwd=calisma_dizini or os.getcwd(),
        )

        self._counters["run"] += 1
        if not sonuc.ok:
            self._counters["failed"] += 1

        self._record(
            gosterim[:300],
            ok=sonuc.ok,
            detail=f"rc={sonuc.returncode}",
            sure_ms=sure_ms,
        )
        self._emit(
            "os.shell_run",
            {
                "command": gosterim[:200],
                "returncode": sonuc.returncode,
                "duration_ms": sure_ms,
                "ok": sonuc.ok,
            },
        )

        if sure_ms > 5000:
            logger.warning("Yavaş komut (%.0f ms): %s", sure_ms, gosterim[:120])
        return sonuc

    def _resolve_timeout(self, timeout: Optional[float], sandbox: Sandbox) -> float:
        """Zaman aşımını belirler ve 1800 saniyelik sert üst sınırı uygular."""
        try:
            varsayilan = float(sandbox.policy.max_command_seconds)
        except Exception:
            varsayilan = 300.0
        sure = varsayilan if timeout is None else float(timeout)
        if sure <= 0:
            sure = varsayilan
        return max(1.0, min(sure, HARD_TIMEOUT_SECONDS))

    @staticmethod
    def _metin(veri: Any) -> str:
        """TimeoutExpired çıktısını güvenli biçimde metne çevirir."""
        if veri is None:
            return ""
        if isinstance(veri, bytes):
            return veri.decode("utf-8", errors="replace")
        return str(veri)

    # ─── Kısayollar ─────────────────────────────────────────────────────

    def run_powershell(self, script: str, **kw: Any) -> CommandResult:
        """
        PowerShell betiği çalıştırır (``-NoProfile -NonInteractive -Command``).

        Args:
            script: PowerShell betiği.
            **kw: `run()` ile aynı ek argümanlar.

        Returns:
            `CommandResult` — PowerShell bulunamazsa ``returncode=-1``.
        """
        kabuk = self.which("pwsh") or self.which("powershell")
        if not kabuk:
            self._fail("PowerShell bulunamadı (pwsh/powershell)")
            return CommandResult(
                command=f"powershell: {script}",
                returncode=-1,
                stdout="",
                stderr="PowerShell bulunamadı (pwsh/powershell)",
                duration_ms=0.0,
            )

        komut = [kabuk, "-NoProfile", "-NonInteractive", "-Command", str(script)]
        kw.setdefault("shell", False)
        return self.run(komut, **kw)

    def run_python(self, code: str, **kw: Any) -> CommandResult:
        """
        Geçerli yorumlayıcıyla Python kodu çalıştırır (``python -c``).

        Args:
            code: Çalıştırılacak Python kodu.
            **kw: `run()` ile aynı ek argümanlar.
        """
        komut = [sys.executable, "-c", str(code)]
        kw.setdefault("shell", False)
        return self.run(komut, **kw)

    def which(self, name: str) -> Optional[str]:
        """
        Çalıştırılabilir dosyanın tam yolunu bulur.

        Returns:
            Yol ya da None.
        """
        try:
            return shutil.which(str(name))
        except Exception as exc:
            logger.debug("which(%s) başarısız: %s", name, exc)
            return None

    # ─── İstatistikler ──────────────────────────────────────────────────

    def _fail(self, mesaj: str) -> None:
        """Hata sayacını artırır ve loglar."""
        self._counters["errors"] += 1
        self._last_error = mesaj
        logger.warning(mesaj)

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir kabuk metrikleri."""
        sandbox = self._require_sandbox()
        return {
            "name": self.name,
            "initialized": self._initialized,
            "operations": dict(self._counters),
            "allow_shell": bool(sandbox.policy.allow_shell),
            "dry_run": bool(sandbox.policy.dry_run),
            "max_command_seconds": sandbox.policy.max_command_seconds,
            "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
            "powershell": self.which("pwsh") or self.which("powershell") or "",
            "python": sys.executable,
            "last_error": self._last_error,
            "timestamp": time.time(),
        }


__all__ = ["ShellRunner", "CommandResult", "HARD_TIMEOUT_SECONDS", "split_command"]
