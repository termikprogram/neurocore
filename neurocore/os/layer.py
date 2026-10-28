"""
NeuroCore İşletim Sistemi Katmanı (L5) — Yüz (Facade)
=====================================================
İşletim sistemi katmanının tek giriş noktası. Alt sistemleri (sandbox, denetim,
envanter, yetenek, dosya, süreç, kabuk) tek bir yaşam döngüsü altında toplar
ve **tek bir paylaşılan `Sandbox`** örneğini hepsine enjekte eder.

Katman Sözleşmesi (ARCHITECTURE.md §3):
    ``__init__`` → yalnızca durum · ``initialize`` → bağımlılıklar ·
    ``shutdown`` → idempotent serbest bırakma · ``get_stats`` → JSON metrikler.

Kullanım:
    os_layer = OSLayer()
    os_layer.initialize(config, event_bus)

    print(os_layer.system_report()["os"]["system"])
    print(os_layer.run("echo merhaba")["stdout"])
    os_layer.write("notlar/x.md", "# Merhaba")
    print(os_layer.health())

    os_layer.shutdown()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurocore.os.audit import AuditLog
from neurocore.os.capabilities import CapabilityProbe, ToolInfo
from neurocore.os.filesystem import FileSystemTools
from neurocore.os.inventory import SystemInventory, SystemInventoryReport
from neurocore.os.process import ProcessManager
from neurocore.os.sandbox import Sandbox
from neurocore.os.shell import CommandResult, ShellRunner
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("os.layer")

#: Sağlık denetiminde "olması beklenen" araçlar
REQUIRED_TOOLS = ("python",)
RECOMMENDED_TOOLS = ("git", "node", "ffmpeg", "rg", "pwsh")


# ─── OS Katmanı ────────────────────────────────────────────────────────────

class OSLayer:
    """
    L5 — İşletim Sistemi katmanı.

    Alt sistemler:
        - `sandbox`      : yol/komut güvenlik politikası (paylaşılan tek örnek)
        - `audit`        : markdown denetim günlüğü
        - `inventory`    : sistem envanteri
        - `capabilities` : araç/özellik keşfi
        - `filesystem`   : dosya G/Ç
        - `processes`    : süreç yönetimi
        - `shell`        : komut yürütme
    """

    name = "os"

    def __init__(self) -> None:
        # Yalnızca durum — ağır iş/bağımlılık initialize() içinde
        self._config: Any = None
        self._event_bus: Any = None
        self._initialized: bool = False
        self._initialized_at: float = 0.0
        self._init_errors: List[str] = []

        self._sandbox: Optional[Sandbox] = None
        self._audit = AuditLog()
        self._inventory = SystemInventory()
        self._capabilities = CapabilityProbe()
        self._filesystem = FileSystemTools()
        self._processes = ProcessManager()
        self._shell = ShellRunner()

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Katmanı başlatır: paylaşılan sandbox'ı kurar, alt sistemlere enjekte
        eder ve hepsini başlatır.

        İdempotenttir; bir alt sistem hata verse bile diğerleri başlatılır
        (zarif bozulma).
        """
        if self._initialized:
            return

        baslangic = time.perf_counter()
        self._config = config
        self._event_bus = event_bus
        self._init_errors = []

        kok = getattr(config, "base_path", None) if config is not None else None
        if not kok:
            kok = str(Path(__file__).resolve().parents[2])

        # ─── Paylaşılan sandbox ───
        try:
            sandbox = Sandbox(str(kok))
            sandbox.initialize(config, event_bus)
            self._sandbox = sandbox
        except Exception as exc:
            self._init_errors.append(f"sandbox: {exc}")
            logger.error("Sandbox başlatılamadı: %s", exc)
            self._sandbox = None

        # ─── Denetim günlüğü ───
        try:
            self._audit.initialize(config, event_bus)
        except Exception as exc:
            self._init_errors.append(f"audit: {exc}")
            logger.error("Denetim günlüğü başlatılamadı: %s", exc)

        # ─── Envanter ve yetenekler ───
        for ad, sistem in (("inventory", self._inventory), ("capabilities", self._capabilities)):
            try:
                sistem.initialize(config, event_bus)
            except Exception as exc:
                self._init_errors.append(f"{ad}: {exc}")
                logger.error("%s başlatılamadı: %s", ad, exc)

        # ─── Sandbox + audit enjeksiyonu (dosya, süreç, kabuk) ───
        for ad, arac in (
            ("filesystem", self._filesystem),
            ("processes", self._processes),
            ("shell", self._shell),
        ):
            try:
                if self._sandbox is not None:
                    arac.attach_sandbox(self._sandbox)
                arac.attach_audit(self._audit)
                arac.initialize(config, event_bus)
            except Exception as exc:
                self._init_errors.append(f"{ad}: {exc}")
                logger.error("%s başlatılamadı: %s", ad, exc)

        self._initialized = True
        self._initialized_at = time.time()
        sure_ms = (time.perf_counter() - baslangic) * 1000

        logger.info("-" * 58)
        logger.info("OS katmanı (L5) hazır — %.0f ms", sure_ms)
        logger.info("   Sandbox kökü : %s", self._sandbox.root if self._sandbox else "yok")
        logger.info("   Denetim       : %s", self._audit.path or "(bellek)")
        if self._init_errors:
            logger.warning("   Başlatma uyarıları: %s", "; ".join(self._init_errors))
        logger.info("-" * 58)

        try:
            self._audit.record("os.initialize", self.name, ok=True, detail=f"{sure_ms:.0f} ms", source=self.name)
        except Exception as exc:
            logger.debug("Başlatma kaydı yazılamadı: %s", exc)

        if self._event_bus:
            try:
                self._event_bus.emit(
                    "os.layer_ready",
                    {
                        "sandbox_root": str(self._sandbox.root) if self._sandbox else "",
                        "audit_path": self._audit.path,
                        "duration_ms": sure_ms,
                        "init_errors": list(self._init_errors),
                    },
                    source=self.name,
                )
            except Exception as exc:
                logger.debug("Katman olayı yayınlanamadı: %s", exc)

    def shutdown(self) -> None:
        """Tüm alt sistemleri ters sırada kapatır (idempotent)."""
        if not self._initialized:
            return

        for ad, sistem in (
            ("shell", self._shell),
            ("processes", self._processes),
            ("filesystem", self._filesystem),
            ("capabilities", self._capabilities),
            ("inventory", self._inventory),
        ):
            try:
                sistem.shutdown()
            except Exception as exc:
                logger.warning("%s kapatılamadı: %s", ad, exc)

        try:
            self._audit.record("os.shutdown", self.name, ok=True, source=self.name)
            self._audit.shutdown()
        except Exception as exc:
            logger.warning("Denetim günlüğü kapatılamadı: %s", exc)

        if self._sandbox is not None:
            try:
                self._sandbox.shutdown()
            except Exception as exc:
                logger.warning("Sandbox kapatılamadı: %s", exc)

        self._initialized = False
        logger.info("OS katmanı (L5) kapatıldı")

        if self._event_bus:
            try:
                self._event_bus.emit("os.layer_stopped", {"uptime_seconds": time.time() - self._initialized_at}, source=self.name)
            except Exception as exc:
                logger.debug("Kapanış olayı yayınlanamadı: %s", exc)

    # ─── Alt Sistem Erişimi ─────────────────────────────────────────────

    def _require_sandbox(self) -> Sandbox:
        """Paylaşılan sandbox'ı döndürür; yoksa tembel olarak kurar."""
        if self._sandbox is None:
            kok = getattr(self._config, "base_path", None) if self._config is not None else None
            sandbox = Sandbox(str(kok) if kok else None)
            sandbox.initialize(self._config, self._event_bus)
            self._sandbox = sandbox
            # Alt sistemlere de yay
            for arac in (self._filesystem, self._processes, self._shell):
                try:
                    arac.attach_sandbox(sandbox)
                except Exception:
                    pass
        return self._sandbox

    @property
    def sandbox(self) -> Sandbox:
        """Paylaşılan sandbox örneği."""
        return self._require_sandbox()

    @property
    def audit(self) -> AuditLog:
        """Denetim günlüğü."""
        return self._audit

    @property
    def inventory(self) -> SystemInventory:
        """Sistem envanteri."""
        return self._inventory

    @property
    def capabilities(self) -> CapabilityProbe:
        """Yetenek yoklayıcı."""
        return self._capabilities

    @property
    def filesystem(self) -> FileSystemTools:
        """Dosya sistemi araçları."""
        if self._filesystem.sandbox is None and self._sandbox is not None:
            self._filesystem.attach_sandbox(self._sandbox)
        return self._filesystem

    @property
    def processes(self) -> ProcessManager:
        """Süreç yöneticisi."""
        if self._processes.sandbox is None and self._sandbox is not None:
            self._processes.attach_sandbox(self._sandbox)
        return self._processes

    @property
    def shell(self) -> ShellRunner:
        """Kabuk çalıştırıcı."""
        if self._shell.sandbox is None and self._sandbox is not None:
            self._shell.attach_sandbox(self._sandbox)
        return self._shell

    @property
    def config(self) -> Any:
        """Bağlı `SystemConfig` (None olabilir)."""
        return self._config

    # ─── Kolaylık Metodları ─────────────────────────────────────────────

    def system_report(self, write_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Sistem envanteri anlık görüntüsünü sözlük olarak döndürür.

        Args:
            write_to: Verilirse markdown raporu bu yola yazılır.

        Returns:
            Envanter sözlüğü (``written_to`` alanı eklenebilir).
        """
        rapor: SystemInventoryReport = self._inventory.snapshot()
        veri = rapor.to_dict()

        if write_to:
            try:
                yazilan = self._inventory.write_report(str(write_to))
                veri["written_to"] = yazilan
                try:
                    self._audit.record("os.system_report", str(write_to), ok=bool(yazilan), detail=yazilan, source=self.name)
                except Exception:
                    pass
            except Exception as exc:
                veri["written_to"] = ""
                veri["write_error"] = str(exc)
                logger.warning("Envanter raporu yazılamadı: %s", exc)

        return veri

    def run(self, command: Any, **kw: Any) -> Dict[str, Any]:
        """
        Komut çalıştırır ve sonucu sözlük olarak döndürür.

        `SandboxViolation` (kabuk izni kapalı) yükselir; yıkıcı komutlar
        ``blocked=True`` içeren bir sözlükle döner.
        """
        sonuc: CommandResult = self._shell.run(command, **kw)
        return sonuc.to_dict()

    def read(self, path: Any, **kw: Any) -> str:
        """Dosya okur (sandbox denetimli)."""
        return self._filesystem.read_text(path, **kw)

    def write(self, path: Any, content: str, **kw: Any) -> str:
        """Dosya yazar (sandbox denetimli, atomik)."""
        return self._filesystem.write_text(path, content, **kw)

    def find(self, root: Any, pattern: str, **kw: Any) -> List[str]:
        """Ad desenine uyan dosya yollarını döndürür."""
        return self._filesystem.find_by_name(root, pattern, **kw)

    def list_dir(self, path: Any, **kw: Any) -> List[Dict[str, Any]]:
        """Dizin içeriğini listeler."""
        return self._filesystem.list_dir(path, **kw)

    def processes_top(self, n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """En çok kaynak tüketen süreçler."""
        return self._processes.top_consumers(n)

    def audit_tail(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Son denetim kayıtlarını sözlük listesi olarak döndürür."""
        return [kayit.to_dict() for kayit in self._audit.tail(limit)]

    # ─── Sağlık ─────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """
        Katman sağlık raporu.

        Returns:
            ``{"ok", "issues", "critical", "warnings", "checks", "timestamp"}``

            ``ok`` yalnızca **kritik** sorun yoksa True olur; yönetici yetkisi
            olmaması veya önerilen araçların eksikliği uyarıdır.
        """
        kritik: List[str] = []
        uyarilar: List[str] = []
        kontroller: Dict[str, Any] = {"initialized": self._initialized}

        if not self._initialized:
            kritik.append("KRİTİK: OS katmanı başlatılmadı (initialize() çağrılmadı)")

        # ─── Disk ───
        try:
            kullanim = self._filesystem.disk_usage(self._require_sandbox().root)
            kontroller["disk"] = kullanim
            bos_gb = float(kullanim.get("free_gb", 0.0))
            dolu = float(kullanim.get("percent", 0.0))
            if bos_gb < 5.0 or dolu >= 95.0:
                kritik.append(f"KRİTİK: Disk neredeyse dolu — boş {bos_gb:.1f} GB (%{dolu:.0f} dolu)")
            elif bos_gb < 15.0 or dolu >= 88.0:
                uyarilar.append(f"UYARI: Disk alanı azalıyor — boş {bos_gb:.1f} GB (%{dolu:.0f} dolu)")
        except Exception as exc:
            kritik.append(f"KRİTİK: Disk kullanımı okunamadı — {exc}")
            kontroller["disk"] = {"error": str(exc)}

        # ─── Yönetici yetkisi ───
        try:
            rapor = self._inventory.last_report or self._inventory.snapshot()
            kontroller["elevated"] = bool(rapor.elevated)
            if not rapor.elevated:
                uyarilar.append("UYARI: Yönetici yetkisi yok — bazı sistem işlemleri kısıtlı")
        except Exception as exc:
            kontroller["elevated"] = False
            uyarilar.append(f"UYARI: Yetki seviyesi belirlenemedi — {exc}")

        # ─── Araçlar ───
        try:
            araclar = self._capabilities.probe_all()
            eksik_zorunlu = [ad for ad in REQUIRED_TOOLS if not araclar.get(ad, ToolInfo(ad, False)).available]
            eksik_onerilen = [ad for ad in RECOMMENDED_TOOLS if not araclar.get(ad, ToolInfo(ad, False)).available]
            kontroller["tools_available"] = sorted(ad for ad, b in araclar.items() if b.available)
            kontroller["tools_missing"] = sorted(ad for ad, b in araclar.items() if not b.available)
            if eksik_zorunlu:
                kritik.append(f"KRİTİK: Zorunlu araçlar eksik — {', '.join(eksik_zorunlu)}")
            if eksik_onerilen:
                uyarilar.append(f"UYARI: Önerilen araçlar eksik — {', '.join(eksik_onerilen)}")
            kontroller["cuda"] = bool(araclar.get("cuda") and araclar["cuda"].available)
            kontroller["sqlite_fts5"] = bool(araclar.get("sqlite_fts5") and araclar["sqlite_fts5"].available)
        except Exception as exc:
            uyarilar.append(f"UYARI: Yetenek yoklaması başarısız — {exc}")

        # ─── Sandbox / denetim ───
        try:
            sandbox_stats = self._require_sandbox().get_stats()
            kontroller["sandbox"] = {
                "allow_shell": sandbox_stats.get("allow_shell"),
                "allow_write": sandbox_stats.get("allow_write"),
                "allow_process_kill": sandbox_stats.get("allow_process_kill"),
                "dry_run": sandbox_stats.get("dry_run"),
                "violations": sandbox_stats.get("violations", 0),
            }
            if sandbox_stats.get("dry_run"):
                uyarilar.append("UYARI: Sandbox kuru çalıştırma (dry_run) modunda — hiçbir eylem uygulanmıyor")
            if sandbox_stats.get("violations", 0) > 0:
                uyarilar.append(f"UYARI: {sandbox_stats['violations']} sandbox ihlali kaydedildi")
        except Exception as exc:
            uyarilar.append(f"UYARI: Sandbox istatistikleri alınamadı — {exc}")

        try:
            audit_stats = self._audit.get_stats()
            kontroller["audit"] = {
                "path": audit_stats.get("path"),
                "write_errors": audit_stats.get("write_errors", 0),
                "entries": audit_stats.get("total_recorded", 0),
            }
            if audit_stats.get("write_errors", 0) > 0:
                uyarilar.append(
                    f"UYARI: Denetim günlüğüne {audit_stats['write_errors']} kez yazılamadı"
                )
        except Exception as exc:
            uyarilar.append(f"UYARI: Denetim istatistikleri alınamadı — {exc}")

        if self._init_errors:
            uyarilar.extend(f"UYARI: Başlatma hatası — {hata}" for hata in self._init_errors)

        return {
            "ok": not kritik,
            "issues": kritik + uyarilar,
            "critical": kritik,
            "warnings": uyarilar,
            "checks": kontroller,
            "timestamp": time.time(),
        }

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Katmanın ve tüm alt sistemlerinin JSON metrikleri."""
        def _guvenli(fn: Any) -> Any:
            try:
                return fn()
            except Exception as exc:
                return {"error": str(exc)}

        return {
            "name": self.name,
            "initialized": self._initialized,
            "initialized_at": self._initialized_at,
            "uptime_seconds": (time.time() - self._initialized_at) if self._initialized else 0.0,
            "init_errors": list(self._init_errors),
            "sandbox": _guvenli(self._require_sandbox().get_stats),
            "audit": _guvenli(self._audit.get_stats),
            "inventory": _guvenli(self._inventory.get_stats),
            "capabilities": _guvenli(self._capabilities.get_stats),
            "filesystem": _guvenli(self._filesystem.get_stats),
            "processes": _guvenli(self._processes.get_stats),
            "shell": _guvenli(self._shell.get_stats),
        }

    def __repr__(self) -> str:  # pragma: no cover - tanı amaçlı
        return f"<OSLayer initialized={self._initialized} kök={self._sandbox.root if self._sandbox else '-'}>"


__all__ = ["OSLayer", "REQUIRED_TOOLS", "RECOMMENDED_TOOLS"]
