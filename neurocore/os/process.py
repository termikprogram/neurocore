"""
NeuroCore İşletim Sistemi — Süreç Yöneticisi
============================================
L5 (OS) katmanının **süreç** cephesi: listeleme, arama, başlatma ve
sonlandırma.

Tasarım kuralları:
    - `psutil` **fonksiyon içinde** import edilir → paket eksikken de çalışır.
    - Süreç sonlandırma, sandbox `allow_process_kill` bayrağı açık değilse
      `SandboxViolation` ile reddedilir.
    - Beklemeli çalıştırmalarda `subprocess.run(..., capture_output=...)`
      kullanılır; **asla** elle `PIPE` okuması yapılmaz (kilitlenme yok).
    - Beklemesiz çalıştırmalarda çıktı, arka plan boşaltma iş parçacıklarıyla
      toplanır → boru (pipe) tamponu taşmaz.

Kullanım:
    pm = ProcessManager()
    pm.initialize(config, event_bus)
    print(pm.top_consumers(3))
    sonuc = pm.start([sys.executable, "-c", "print('merhaba')"], wait=True)
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

from neurocore.os.sandbox import Sandbox, SandboxViolation
from neurocore.os.shell import split_command
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("os.process")


def _no_window_flags() -> int:
    """Alt süreçlerde konsol penceresi açılmasın."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


# ─── Ana Süreç Yöneticisi ──────────────────────────────────────────────────

class ProcessManager:
    """
    Sandbox denetimli süreç yönetimi.

    `sandbox` özniteliği dışarıdan atanabilir ya da `attach_sandbox()` ile
    bağlanabilir; hiçbiri yoksa tembel olarak varsayılan bir `Sandbox` kurulur.
    """

    name = "os.process"

    def __init__(self) -> None:
        self.sandbox: Optional[Sandbox] = None
        self.audit: Any = None
        self._initialized: bool = False
        self._config: Any = None
        self._event_bus: Any = None
        self._counters: Dict[str, int] = defaultdict(int)
        self._last_error: str = ""
        self._background: Dict[int, Dict[str, Any]] = {}
        self._cpu_sample_delay: float = 0.05

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
        logger.info("Süreç yöneticisi hazır (pid=%d)", os.getpid())

    def shutdown(self) -> None:
        """Arka plan kayıtlarını temizler (idempotent). Süreçleri öldürmez."""
        if not self._initialized:
            return
        self._background.clear()
        self._initialized = False
        logger.info("Süreç yöneticisi kapatıldı")

    def attach_sandbox(self, sandbox: Sandbox) -> Sandbox:
        """Paylaşılan sandbox örneğini bağlar."""
        self.sandbox = sandbox
        logger.debug("Süreç yöneticisine sandbox bağlandı: %s", sandbox.root)
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

    def _record(self, action: str, target: str, ok: bool = True, detail: str = "", sure_ms: float = 0.0) -> None:
        """Denetim günlüğüne kayıt düşer (audit yoksa no-op)."""
        if not self.audit:
            return
        try:
            self.audit.record(action, target, ok=ok, detail=detail, duration_ms=sure_ms, source=self.name)
        except Exception as exc:
            logger.debug("Denetim kaydı yazılamadı: %s", exc)

    # ─── Listeleme ──────────────────────────────────────────────────────

    def list_processes(
        self,
        sort_by: str = "cpu",
        limit: int = 50,
        name_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Çalışan süreçleri listeler.

        Args:
            sort_by: ``cpu`` | ``memory`` | ``name`` | ``pid`` | ``create_time``
            limit: Maksimum kayıt sayısı.
            name_filter: Ad içinde geçmesi gereken metin (küçük/büyük harf duyarsız).

        Returns:
            ``[{"pid", "name", "cpu", "memory_mb", "status", "username",
            "create_time", "cmdline"}, …]``

        Not:
            CPU yüzdesi iki örnek arasındaki farktan hesaplanır; bu nedenle
            çağrı kısa bir örnekleme gecikmesi (~50 ms) içerir.
        """
        psutil = self._import_psutil()
        if psutil is None:
            logger.warning("psutil yok — süreç listesi alınamadı")
            return []

        attrs = ["pid", "name", "cpu_percent", "memory_info", "status", "username", "create_time", "cmdline"]
        surecler: List[Any] = []
        try:
            surecler = list(psutil.process_iter(attrs=attrs, ad_value=None))
        except Exception as exc:
            self._fail(f"Süreç listesi alınamadı: {exc}")
            return []

        # CPU yüzdesi için ikinci örnek (ilk çağrı her zaman 0.0 döner)
        if self._cpu_sample_delay > 0:
            time.sleep(self._cpu_sample_delay)

        kayitlar: List[Dict[str, Any]] = []
        for surec in surecler:
            try:
                bilgi = surec.info
                ad = str(bilgi.get("name") or "")
                if name_filter and str(name_filter).lower() not in ad.lower():
                    continue

                bellek = bilgi.get("memory_info")
                bellek_mb = round(float(getattr(bellek, "rss", 0) or 0) / (1024 * 1024), 1)

                try:
                    cpu = float(surec.cpu_percent(None) or 0.0)
                except Exception:
                    cpu = float(bilgi.get("cpu_percent") or 0.0)

                cmdline = bilgi.get("cmdline") or []
                if isinstance(cmdline, (list, tuple)):
                    kisa = " ".join(str(p) for p in cmdline[:3])[:200]
                else:
                    kisa = str(cmdline)[:200]

                kayitlar.append(
                    {
                        "pid": int(bilgi.get("pid") or 0),
                        "name": ad,
                        "cpu": round(cpu, 1),
                        "memory_mb": bellek_mb,
                        "status": str(bilgi.get("status") or ""),
                        "username": str(bilgi.get("username") or ""),
                        "create_time": float(bilgi.get("create_time") or 0.0),
                        "cmdline": kisa,
                    }
                )
            except Exception:
                continue  # süreç bu arada ölmüş olabilir

        anahtarlar = {
            "cpu": lambda k: k["cpu"],
            "memory": lambda k: k["memory_mb"],
            "name": lambda k: k["name"].lower(),
            "pid": lambda k: k["pid"],
            "create_time": lambda k: k["create_time"],
        }
        anahtar = anahtarlar.get(str(sort_by), anahtarlar["cpu"])
        try:
            kayitlar.sort(key=anahtar, reverse=str(sort_by) in ("cpu", "memory", "create_time"))
        except Exception:
            pass

        self._counters["list_processes"] += 1
        return kayitlar[: max(0, int(limit))]

    def find(self, name_or_pid: Union[str, int]) -> List[Dict[str, Any]]:
        """
        Ada (parça eşleşmesi) veya PID'ye göre süreç arar.

        Args:
            name_or_pid: Süreç adı parçası ya da PID (int/str).

        Returns:
            Eşleşen süreç kayıtları.
        """
        hedef_pid: Optional[int] = None
        if isinstance(name_or_pid, int):
            hedef_pid = name_or_pid
        else:
            metin = str(name_or_pid).strip()
            if metin.isdigit():
                hedef_pid = int(metin)
            else:
                bulunan = self.list_processes(sort_by="pid", limit=100000, name_filter=metin)
                # cmdline içinde geçenleri de ekle
                if not bulunan:
                    for kayit in self.list_processes(sort_by="pid", limit=100000):
                        if metin.lower() in str(kayit.get("cmdline", "")).lower():
                            bulunan.append(kayit)
                return bulunan

        if hedef_pid is None:
            return []
        bilgi = self.info(hedef_pid)
        return [bilgi] if bilgi else []

    def info(self, pid: int) -> Optional[Dict[str, Any]]:
        """
        Tek bir süreç hakkında ayrıntılı bilgi döndürür.

        Returns:
            Bilgi sözlüğü; süreç yoksa/erişilemiyorsa None.
        """
        psutil = self._import_psutil()
        if psutil is None:
            return None

        try:
            surec = psutil.Process(int(pid))
            with surec.oneshot():
                bellek = surec.memory_info()
                try:
                    cpu = float(surec.cpu_percent(None) or 0.0)
                except Exception:
                    cpu = 0.0

                bilgi: Dict[str, Any] = {
                    "pid": int(pid),
                    "name": self._guvenli(surec.name),
                    "exe": self._guvenli(surec.exe),
                    "status": self._guvenli(surec.status),
                    "username": self._guvenli(surec.username),
                    "create_time": float(self._guvenli(surec.create_time) or 0.0),
                    "cpu": round(cpu, 1),
                    "memory_mb": round(float(getattr(bellek, "rss", 0) or 0) / (1024 * 1024), 1),
                    "vms_mb": round(float(getattr(bellek, "vms", 0) or 0) / (1024 * 1024), 1),
                    "num_threads": int(self._guvenli(surec.num_threads) or 0),
                    "parent_pid": int(self._guvenli(surec.ppid) or 0),
                    "cmdline": " ".join(str(p) for p in (self._guvenli(surec.cmdline) or []))[:500],
                    "cwd": self._guvenli(surec.cwd),
                    "nice": int(self._guvenli(surec.nice) or 0),
                }
            self._counters["info"] += 1
            return bilgi
        except Exception as exc:
            logger.debug("Süreç bilgisi alınamadı (%s): %s", pid, exc)
            return None

    @staticmethod
    def _guvenli(fn: Any) -> Any:
        """Erişilemeyen psutil alanlarını güvenli biçimde okur."""
        try:
            return fn() if callable(fn) else fn
        except Exception:
            return None

    def top_consumers(self, n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        En çok kaynak tüketen süreçler.

        Returns:
            ``{"cpu": [...], "memory": [...]}``
        """
        adet = max(1, int(n))
        return {
            "cpu": self.list_processes(sort_by="cpu", limit=adet),
            "memory": self.list_processes(sort_by="memory", limit=adet),
        }

    # ─── Başlatma ───────────────────────────────────────────────────────

    def start(
        self,
        command: Union[str, Sequence[str]],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        shell: bool = False,
        wait: bool = False,
        timeout: Optional[float] = None,
        capture: bool = True,
    ) -> Dict[str, Any]:
        """
        Yeni bir süreç başlatır.

        Args:
            command: Komut (``str`` veya argüman listesi). ``str`` + ``shell=False``
                verilirse `split_command` ile ayrıştırılır.
            cwd: Çalışma dizini (sandbox denetiminden geçer).
            env: Ek/geçersiz kılınan ortam değişkenleri.
            shell: True ise komut kabuğa verilir (**sandbox `allow_shell` ister**).
            wait: True ise süreç bitene kadar beklenir ve çıktı döndürülür.
            timeout: Bekleme zaman aşımı (saniye); aşılırsa süreç öldürülür.
            capture: Çıktıyı yakala.

        Returns:
            ``{"ok", "pid", "stdout", "stderr", "returncode", "duration_ms",
            "timed_out", "blocked", "dry_run", "command"}``

        Raises:
            SandboxViolation: ``shell=True`` istendiğinde `allow_shell=False` ise.
        """
        baslangic = time.perf_counter()
        sandbox = self._require_sandbox()
        gosterim = command if isinstance(command, str) else " ".join(str(p) for p in command)

        sonuc: Dict[str, Any] = {
            "ok": False,
            "pid": 0,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "duration_ms": 0.0,
            "timed_out": False,
            "blocked": False,
            "dry_run": False,
            "command": str(gosterim)[:500],
        }

        # ─── Politika denetimleri ───
        if shell and not sandbox.policy.allow_shell:
            raise SandboxViolation("Kabuk çalıştırma izni kapalı (allow_shell=False)")

        yikici, gerekce = sandbox.is_dangerous(str(gosterim))
        if yikici:
            sonuc.update({"blocked": True, "stderr": f"Engellendi: {gerekce}", "duration_ms": 0.0})
            logger.warning("Yıkıcı süreç başlatma engellendi: %s (%s)", gosterim, gerekce)
            self._counters["blocked"] += 1
            self._emit("os.process_blocked", {"command": str(gosterim)[:200], "reason": gerekce})
            self._record("process.start", str(gosterim)[:200], ok=False, detail=f"engellendi: {gerekce}")
            return sonuc

        # ─── Komut biçimini hazırla ───
        if isinstance(command, (list, tuple)):
            argv: Union[str, List[str]] = [str(p) for p in command]
            shell = False  # liste komutlar asla kabuktan geçmez
        else:
            argv = str(command)
            if not shell:
                try:
                    argv = split_command(str(command))
                except Exception as exc:
                    sonuc["stderr"] = f"Komut ayrıştırılamadı: {exc}"
                    return sonuc

        # ─── Çalışma dizini ───
        calisma_dizini: Optional[str] = None
        if cwd:
            try:
                cozulen = sandbox.resolve_path(cwd, must_exist=True)
                calisma_dizini = str(cozulen)
            except SandboxViolation:
                raise
            except Exception as exc:
                sonuc["stderr"] = f"Çalışma dizini geçersiz: {exc}"
                self._record("process.start", str(gosterim)[:200], ok=False, detail=str(exc))
                return sonuc

        # ─── Kuru çalıştırma ───
        if sandbox.policy.dry_run:
            sonuc.update(
                {
                    "ok": True,
                    "dry_run": True,
                    "stdout": f"[kuru çalıştırma] {gosterim}",
                    "duration_ms": (time.perf_counter() - baslangic) * 1000,
                }
            )
            logger.info("[kuru çalıştırma] Süreç başlatılmadı: %s", gosterim)
            return sonuc

        ortam: Optional[Dict[str, str]] = None
        if env:
            try:
                ortam = dict(os.environ)
                ortam.update({str(k): str(v) for k, v in env.items()})
            except Exception as exc:
                logger.debug("Ortam değişkenleri hazırlanamadı: %s", exc)
                ortam = None

        # ─── Beklemeli çalıştırma ───
        if wait:
            try:
                tamamlanan = subprocess.run(
                    argv,
                    cwd=calisma_dizini,
                    env=ortam,
                    shell=shell,
                    capture_output=bool(capture),
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=float(timeout) if timeout else None,
                    creationflags=_no_window_flags(),
                )
                sure_ms = (time.perf_counter() - baslangic) * 1000
                sonuc.update(
                    {
                        "ok": tamamlanan.returncode == 0,
                        "pid": int(getattr(tamamlanan, "pid", 0) or 0),
                        "stdout": (tamamlanan.stdout or "") if capture else "",
                        "stderr": (tamamlanan.stderr or "") if capture else "",
                        "returncode": int(tamamlanan.returncode),
                        "duration_ms": sure_ms,
                    }
                )
                self._counters["start_wait"] += 1
                self._record(
                    "process.start",
                    str(gosterim)[:200],
                    ok=bool(sonuc["ok"]),
                    detail=f"pid={sonuc['pid']} rc={sonuc['returncode']}",
                    sure_ms=sure_ms,
                )
                self._emit(
                    "os.process_finished",
                    {"pid": sonuc["pid"], "returncode": sonuc["returncode"], "duration_ms": sure_ms},
                )
                return sonuc
            except subprocess.TimeoutExpired as exc:
                sure_ms = (time.perf_counter() - baslangic) * 1000
                sonuc.update(
                    {
                        "timed_out": True,
                        "stderr": f"Zaman aşımı ({timeout} sn) — süreç öldürüldü",
                        "returncode": -1,
                        "duration_ms": sure_ms,
                    }
                )
                self._counters["timeouts"] += 1
                self._fail(f"Süreç zaman aşımı: {gosterim}")
                self._record("process.start", str(gosterim)[:200], ok=False, detail="zaman aşımı", sure_ms=sure_ms)
                logger.warning("Süreç zaman aşımına uğradı ve öldürüldü: %s", gosterim)
                return sonuc
            except SandboxViolation:
                raise
            except Exception as exc:
                sonuc["stderr"] = str(exc)
                sonuc["duration_ms"] = (time.perf_counter() - baslangic) * 1000
                self._fail(f"Süreç başlatılamadı: {exc}")
                self._record("process.start", str(gosterim)[:200], ok=False, detail=str(exc))
                return sonuc

        # ─── Beklemesiz çalıştırma (boru tamponu taşmasın diye boşaltma iş parçacığı) ───
        try:
            surec = subprocess.Popen(
                argv,
                cwd=calisma_dizini,
                env=ortam,
                shell=shell,
                stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_no_window_flags(),
            )
        except Exception as exc:
            sonuc["stderr"] = str(exc)
            sonuc["duration_ms"] = (time.perf_counter() - baslangic) * 1000
            self._fail(f"Süreç başlatılamadı: {exc}")
            self._record("process.start", str(gosterim)[:200], ok=False, detail=str(exc))
            return sonuc

        sonuc.update(
            {
                "ok": True,
                "pid": int(surec.pid),
                "returncode": None,
                "duration_ms": (time.perf_counter() - baslangic) * 1000,
            }
        )

        if capture:
            self._start_drain(surec)
        self._counters["start_async"] += 1
        self._record("process.start", str(gosterim)[:200], ok=True, detail=f"pid={surec.pid} (arka plan)")
        self._emit("os.process_started", {"pid": int(surec.pid), "command": str(gosterim)[:200]})
        return sonuc

    def _start_drain(self, surec: "subprocess.Popen[str]") -> None:
        """Arka plan sürecinin çıktısını boşaltan iş parçacıklarını başlatır."""
        kayit: Dict[str, Any] = {"stdout": [], "stderr": [], "process": surec, "done": False}
        self._background[int(surec.pid)] = kayit

        def bosalt(akis: Any, hedef: List[str]) -> None:
            try:
                if akis is None:
                    return
                for satir in akis:
                    hedef.append(satir)
            except Exception as exc:
                logger.debug("Çıktı boşaltma hatası: %s", exc)

        for akis, hedef in ((surec.stdout, kayit["stdout"]), (surec.stderr, kayit["stderr"])):
            ip = threading.Thread(target=bosalt, args=(akis, hedef), daemon=True)
            ip.start()

        def bekleyici() -> None:
            try:
                kayit["returncode"] = surec.wait()
            except Exception:
                kayit["returncode"] = None
            finally:
                kayit["done"] = True
                try:
                    surec.stdout and surec.stdout.close()
                    surec.stderr and surec.stderr.close()
                except Exception:
                    pass

        threading.Thread(target=bekleyici, daemon=True).start()

    def background_output(self, pid: int, join: bool = False, timeout: float = 0.2) -> Dict[str, Any]:
        """
        Beklemesiz başlatılan sürecin biriken çıktısını döndürür.

        Args:
            pid: Süreç kimliği.
            join: True ise çıktı boşalana kadar (kısa süre) beklenir.
            timeout: `join=True` için bekleme süresi.

        Returns:
            ``{"pid", "stdout", "stderr", "returncode", "done", "found"}``
        """
        kayit = self._background.get(int(pid))
        if not kayit:
            return {"pid": int(pid), "stdout": "", "stderr": "", "returncode": None, "done": True, "found": False}

        if join:
            bitis = time.time() + max(0.0, float(timeout))
            while time.time() < bitis and not kayit.get("done"):
                time.sleep(0.02)

        return {
            "pid": int(pid),
            "stdout": "".join(kayit.get("stdout") or []),
            "stderr": "".join(kayit.get("stderr") or []),
            "returncode": kayit.get("returncode"),
            "done": bool(kayit.get("done")),
            "found": True,
        }

    # ─── Sonlandırma ────────────────────────────────────────────────────

    def kill(self, pid: int, force: bool = False) -> Dict[str, Any]:
        """
        Tek bir süreci sonlandırır.

        Args:
            pid: Süreç kimliği.
            force: True ise ``kill`` (SIGKILL), değilse ``terminate``.

        Returns:
            ``{"ok", "pid", "name", "signal", "detail"}``

        Raises:
            SandboxViolation: `allow_process_kill=False` ise, kendi sürecimiz
                veya geçersiz bir PID hedeflenmişse.
        """
        sandbox = self._require_sandbox()
        if not sandbox.policy.allow_process_kill:
            raise SandboxViolation(
                "Süreç sonlandırma izni kapalı (allow_process_kill=False)"
            )

        hedef = int(pid)
        if hedef <= 0:
            raise SandboxViolation(f"Geçersiz PID: {hedef}")
        if hedef == os.getpid():
            raise SandboxViolation("Kendi sürecini sonlandırma reddedildi")
        if hedef == getattr(os, "getppid", lambda: -1)():
            raise SandboxViolation("Üst süreci sonlandırma reddedildi")

        sonuc: Dict[str, Any] = {"ok": False, "pid": hedef, "name": "", "signal": "", "detail": ""}
        sinyal = "kill" if force else "terminate"

        psutil = self._import_psutil()
        if psutil is not None:
            try:
                surec = psutil.Process(hedef)
                sonuc["name"] = self._guvenli(surec.name) or ""
                if force:
                    surec.kill()
                else:
                    surec.terminate()
                try:
                    surec.wait(timeout=5)
                except Exception:
                    pass
                sonuc.update({"ok": True, "signal": sinyal, "detail": "psutil"})
            except Exception as exc:
                sonuc["detail"] = str(exc)
                self._fail(f"Süreç sonlandırılamadı (pid={hedef}): {exc}")
        elif sys.platform == "win32":
            try:
                komut = ["taskkill", "/PID", str(hedef)] + (["/F"] if force else [])
                tamamlanan = subprocess.run(
                    komut,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                    creationflags=_no_window_flags(),
                )
                sonuc.update(
                    {
                        "ok": tamamlanan.returncode == 0,
                        "signal": sinyal,
                        "detail": (tamamlanan.stdout or tamamlanan.stderr or "").strip()[:200],
                    }
                )
            except Exception as exc:
                sonuc["detail"] = str(exc)
        else:
            try:
                os.kill(hedef, 9 if force else 15)
                sonuc.update({"ok": True, "signal": sinyal, "detail": "os.kill"})
            except Exception as exc:
                sonuc["detail"] = str(exc)

        self._counters["kill"] += 1
        self._record("process.kill", f"pid={hedef}", ok=bool(sonuc["ok"]), detail=sonuc["detail"] or sinyal)
        self._emit("os.process_killed", {"pid": hedef, "force": bool(force), "ok": bool(sonuc["ok"])})
        if sonuc["ok"]:
            logger.info("Süreç sonlandırıldı: pid=%d (%s)", hedef, sinyal)
        return sonuc

    def kill_tree(self, pid: int, force: bool = False) -> Dict[str, Any]:
        """
        Bir süreci ve tüm alt süreçlerini sonlandırır.

        Raises:
            SandboxViolation: `allow_process_kill=False` ise veya PID geçersizse.
        """
        sandbox = self._require_sandbox()
        if not sandbox.policy.allow_process_kill:
            raise SandboxViolation(
                "Süreç sonlandırma izni kapalı (allow_process_kill=False)"
            )

        hedef = int(pid)
        if hedef <= 0 or hedef == os.getpid():
            raise SandboxViolation(f"Geçersiz/ yasak PID: {hedef}")

        sonuc: Dict[str, Any] = {"ok": False, "pid": hedef, "killed": [], "failed": [], "detail": ""}
        psutil = self._import_psutil()
        if psutil is None:
            tek = self.kill(hedef, force=force)
            sonuc.update({"ok": tek["ok"], "killed": [hedef] if tek["ok"] else [], "detail": "psutil yok"})
            return sonuc

        try:
            kok = psutil.Process(hedef)
            cocuklar = []
            try:
                cocuklar = kok.children(recursive=True)
            except Exception as exc:
                logger.debug("Alt süreçler alınamadı: %s", exc)

            for cocuk in reversed(cocuklar):  # önce yapraklar
                try:
                    cocuk.kill() if force else cocuk.terminate()
                    sonuc["killed"].append(int(cocuk.pid))
                except Exception as exc:
                    sonuc["failed"].append({"pid": int(cocuk.pid), "error": str(exc)})

            try:
                kok.kill() if force else kok.terminate()
                sonuc["killed"].append(int(kok.pid))
                sonuc["ok"] = True
            except Exception as exc:
                sonuc["failed"].append({"pid": int(kok.pid), "error": str(exc)})
                sonuc["detail"] = str(exc)

            try:
                psutil.wait_procs(cocuklar + [kok], timeout=5)
            except Exception:
                pass
        except Exception as exc:
            sonuc["detail"] = str(exc)
            self._fail(f"Süreç ağacı sonlandırılamadı (pid={hedef}): {exc}")

        self._counters["kill_tree"] += 1
        self._record(
            "process.kill_tree",
            f"pid={hedef}",
            ok=bool(sonuc["ok"]),
            detail=f"{len(sonuc['killed'])} süreç",
        )
        self._emit("os.process_killed", {"pid": hedef, "tree": True, "count": len(sonuc["killed"])})
        return sonuc

    # ─── Yardımcılar ────────────────────────────────────────────────────

    @staticmethod
    def _import_psutil() -> Any:
        """psutil'i (varsa) döndürür — modül düzeyinde import edilmez."""
        try:
            import psutil

            return psutil
        except Exception as exc:
            logger.debug("psutil kullanılamıyor: %s", exc)
            return None

    def _fail(self, mesaj: str) -> None:
        """Hata sayacını artırır ve loglar."""
        self._counters["errors"] += 1
        self._last_error = mesaj
        logger.warning(mesaj)

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir süreç yöneticisi metrikleri."""
        psutil = self._import_psutil()
        return {
            "name": self.name,
            "initialized": self._initialized,
            "psutil_available": psutil is not None,
            "self_pid": os.getpid(),
            "operations": dict(self._counters),
            "background_processes": len(self._background),
            "allow_process_kill": bool(self._require_sandbox().policy.allow_process_kill),
            "last_error": self._last_error,
            "timestamp": time.time(),
        }


__all__ = ["ProcessManager"]
