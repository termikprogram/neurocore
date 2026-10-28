"""
NeuroCore İşletim Sistemi — Sistem Envanteri
============================================
L5 (OS) katmanının **donanım/yazılım parmak izi** modülü. İşletim sistemi,
CPU, bellek, disk, GPU, ağ ve ortam değişkenlerini tek bir
`SystemInventoryReport` içinde toplar.

Tasarım kuralları:
    - Her alan **ayrı ayrı** try/except ile toplanır → tek bir başarısızlık
      anlık görüntüyü öldürmez, `errors` listesine yazılır.
    - Ağır modüller (`psutil`, `torch`) yalnızca fonksiyon içinde import edilir.
    - İnternet erişilebilirliği `snapshot()` içinde **denenmez**; pahalı ve
      yan etkili olduğu için ayrı bir `check_connectivity()` metodundadır.

Kullanım:
    env = SystemInventory()
    env.initialize(config, event_bus)
    rapor = env.snapshot()
    print(rapor.to_markdown())
"""

from __future__ import annotations

import getpass
import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("os.inventory")


# ─── Sabitler ───────────────────────────────────────────────────────────────

#: Envantere alınacak ortam değişkenleri (küratörlü izin listesi)
ENV_ALLOWLIST = (
    "PATH",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMFILES",
    "SYSTEMROOT",
    "COMPUTERNAME",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    "CUDA_PATH",
    "VIRTUAL_ENV",
)

#: Ek olarak toplanacak ortam değişkeni önekleri
ENV_PREFIXES = ("DSH_", "NEUROCORE_")

#: İçinde bu kelimeler geçen değişkenlerin **değeri** maskelenir
SENSITIVE_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")

MASK = "***"


def _no_window_flags() -> int:
    """Alt süreçlerde konsol penceresi açılmasını engelleyen bayraklar."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


# ─── Envanter Raporu ───────────────────────────────────────────────────────

@dataclass
class SystemInventoryReport:
    """
    Sistem envanterinin tam anlık görüntüsü.

    Tüm alt bölümler sözlük/liste tutulur; alan tipleri JSON uyumludur.
    """

    timestamp: float = 0.0
    hostname: str = ""
    user: str = ""
    elevated: bool = False

    os: Dict[str, Any] = field(default_factory=dict)
    python: Dict[str, Any] = field(default_factory=dict)
    cpu: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    disks: List[Dict[str, Any]] = field(default_factory=list)
    gpu: Dict[str, Any] = field(default_factory=dict)
    network: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)

    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Raporu JSON-serileştirilebilir dict'e çevirir."""
        return {
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "user": self.user,
            "elevated": self.elevated,
            "os": dict(self.os),
            "python": dict(self.python),
            "cpu": dict(self.cpu),
            "memory": dict(self.memory),
            "disks": [dict(d) for d in self.disks],
            "gpu": dict(self.gpu),
            "network": dict(self.network),
            "environment": dict(self.environment),
            "errors": list(self.errors),
            "duration_ms": self.duration_ms,
        }

    # ─── Markdown ───────────────────────────────────────────────────────

    def to_markdown(self) -> str:
        """Raporu Obsidian uyumlu markdown metnine çevirir."""
        zaman = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp or time.time()))
        satirlar: List[str] = [
            "# 🖥 Sistem Envanteri",
            "",
            f"> Oluşturma: **{zaman}** · Süre: **{self.duration_ms:.0f} ms**",
            "",
            "## Özet",
            "",
            "| Alan | Değer |",
            "|---|---|",
            f"| Ana makine | `{self.hostname}` |",
            f"| Kullanıcı | `{self.user}` |",
            f"| Yönetici yetkisi | {'✅ evet' if self.elevated else '❌ hayır'} |",
            f"| İşletim sistemi | {self.os.get('system', '?')} {self.os.get('release', '')} "
            f"({self.os.get('edition', '')}) |",
            f"| Python | {self.python.get('version', '?')} |",
            f"| CPU | {self.cpu.get('processor', '?')} · "
            f"{self.cpu.get('count_physical', 0)}P/{self.cpu.get('count_logical', 0)}L |",
            f"| RAM | {self.memory.get('total_mb', 0):.0f} MB "
            f"(boş {self.memory.get('available_mb', 0):.0f} MB) |",
            f"| GPU | {self.gpu.get('summary', 'yok')} |",
            f"| Çalışma süresi | {self._sure(self.os.get('uptime_seconds', 0))} |",
            "",
        ]

        # ─── İşletim sistemi ───
        satirlar += ["## İşletim Sistemi", ""]
        for anahtar, deger in self.os.items():
            if anahtar == "uptime_seconds":
                deger = self._sure(deger)
            satirlar.append(f"- **{anahtar}**: `{deger}`")
        satirlar.append("")

        # ─── Python ───
        satirlar += ["## Python", ""]
        for anahtar, deger in self.python.items():
            satirlar.append(f"- **{anahtar}**: `{deger}`")
        satirlar.append("")

        # ─── CPU ───
        satirlar += ["## CPU", ""]
        for anahtar, deger in self.cpu.items():
            if anahtar == "per_core_percent":
                yuzdeler = ", ".join(f"{v:.0f}%" for v in deger) if deger else "-"
                satirlar.append(f"- **çekirdek kullanımı**: {yuzdeler}")
            else:
                satirlar.append(f"- **{anahtar}**: `{deger}`")
        satirlar.append("")

        # ─── Bellek ───
        satirlar += ["## Bellek", ""]
        for anahtar, deger in self.memory.items():
            if isinstance(deger, float):
                satirlar.append(f"- **{anahtar}**: {deger:.1f}")
            else:
                satirlar.append(f"- **{anahtar}**: `{deger}`")
        satirlar.append("")

        # ─── Diskler ───
        satirlar += ["## Diskler", ""]
        if self.disks:
            satirlar += ["| Sürücü | Tür | Toplam (GB) | Boş (GB) | Doluluk |", "|---|---|---|---|---|"]
            for disk in self.disks:
                satirlar.append(
                    f"| `{disk.get('mountpoint', '?')}` | {disk.get('fstype', '?')} | "
                    f"{disk.get('total_gb', 0):.1f} | {disk.get('free_gb', 0):.1f} | "
                    f"{disk.get('percent', 0):.1f}% |"
                )
        else:
            satirlar.append("_Disk bilgisi alınamadı._")
        satirlar.append("")

        # ─── GPU ───
        satirlar += ["## GPU", ""]
        if self.gpu.get("available"):
            satirlar.append(f"- **kaynak**: `{self.gpu.get('source', '?')}`")
            satirlar.append(f"- **sürücü**: `{self.gpu.get('driver_version', '?')}`")
            if self.gpu.get("cuda_version"):
                satirlar.append(f"- **CUDA**: `{self.gpu['cuda_version']}`")
            for aygit in self.gpu.get("devices", []):
                satirlar.append(
                    f"- **{aygit.get('name', '?')}**: VRAM "
                    f"{aygit.get('vram_total_mb', 0):.0f} MB "
                    f"(boş {aygit.get('vram_free_mb', 0):.0f} MB)"
                )
        else:
            satirlar.append("_GPU bulunamadı (CUDA ve nvidia-smi yok)._")
        satirlar.append("")

        # ─── Ağ ───
        satirlar += ["## Ağ", ""]
        satirlar.append(f"- **ana makine**: `{self.network.get('hostname', '?')}`")
        satirlar.append(f"- **birincil IP**: `{self.network.get('primary_ip', '?')}`")
        for arayuz, adresler in (self.network.get("interfaces") or {}).items():
            satirlar.append(f"- **{arayuz}**: {', '.join(adresler) if adresler else '-'}")
        satirlar.append("")

        # ─── Ortam değişkenleri ───
        satirlar += ["## Ortam Değişkenleri (küratörlü)", ""]
        if self.environment:
            for anahtar, deger in self.environment.items():
                satirlar.append(f"- **{anahtar}** = `{deger}`")
        else:
            satirlar.append("_Ortam değişkeni toplanamadı._")
        satirlar.append("")

        # ─── Hatalar ───
        if self.errors:
            satirlar += ["## Toplama Uyarıları", ""]
            for hata in self.errors:
                satirlar.append(f"- ⚠️ {hata}")
            satirlar.append("")

        return "\n".join(satirlar)

    @staticmethod
    def _sure(saniye: Any) -> str:
        """Saniyeyi okunur süre metnine çevirir."""
        try:
            toplam = int(float(saniye))
        except Exception:
            return "bilinmiyor"
        gun, kalan = divmod(toplam, 86400)
        saat, kalan = divmod(kalan, 3600)
        dakika, saniye = divmod(kalan, 60)
        return f"{gun}g {saat}s {dakika}d {saniye}sn"


# ─── Ana Envanter Sınıfı ───────────────────────────────────────────────────

class SystemInventory:
    """
    Sistem envanteri toplayıcısı.

    Kullanım:
        env = SystemInventory()
        env.initialize(config, event_bus)
        rapor = env.snapshot()
        env.write_report("data/os_report.md")
    """

    name = "os.inventory"

    def __init__(self) -> None:
        self._initialized: bool = False
        self._config: Any = None
        self._event_bus: Any = None
        self._last_report: Optional[SystemInventoryReport] = None
        self._snapshot_count: int = 0
        self._error_count: int = 0
        self._last_duration_ms: float = 0.0
        self._connectivity_checks: int = 0

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Bağımlılıkları bağlar (idempotent). Ağır G/Ç yapılmaz."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._initialized = True
        logger.info("Sistem envanteri hazır (platform: %s)", platform.system())

    def shutdown(self) -> None:
        """Envanteri kapatır (idempotent)."""
        if not self._initialized:
            return
        self._initialized = False
        self._last_report = None
        logger.info("Sistem envanteri kapatıldı")

    @property
    def last_report(self) -> Optional[SystemInventoryReport]:
        """En son alınan anlık görüntü (henüz alınmadıysa None)."""
        return self._last_report

    # ─── Anlık Görüntü ──────────────────────────────────────────────────

    def snapshot(self) -> SystemInventoryReport:
        """
        Tam sistem anlık görüntüsü alır.

        Her bölüm bağımsız olarak toplanır; bir bölüm hata verirse
        `errors` listesine eklenir ve diğerleri toplanmaya devam eder.

        Returns:
            `SystemInventoryReport`
        """
        baslangic = time.perf_counter()
        rapor = SystemInventoryReport(timestamp=time.time())

        bolumler = (
            ("hostname", self._hostname),
            ("user", self._user),
            ("elevated", self._elevated),
            ("os", self._collect_os),
            ("python", self._collect_python),
            ("cpu", self._collect_cpu),
            ("memory", self._collect_memory),
            ("disks", self._collect_disks),
            ("gpu", self._collect_gpu),
            ("network", self._collect_network),
            ("environment", self._collect_environment),
        )

        for alan, toplayici in bolumler:
            try:
                setattr(rapor, alan, toplayici())
            except Exception as exc:
                rapor.errors.append(f"{alan}: {exc}")
                self._error_count += 1
                logger.debug("Envanter bölümü toplanamadı (%s): %s", alan, exc)

        rapor.duration_ms = (time.perf_counter() - baslangic) * 1000
        self._last_duration_ms = rapor.duration_ms
        self._snapshot_count += 1
        self._last_report = rapor

        if self._event_bus:
            try:
                self._event_bus.emit(
                    "os.inventory_snapshot",
                    {
                        "hostname": rapor.hostname,
                        "elevated": rapor.elevated,
                        "disk_count": len(rapor.disks),
                        "gpu_available": bool(rapor.gpu.get("available")),
                        "duration_ms": rapor.duration_ms,
                        "error_count": len(rapor.errors),
                    },
                    source=self.name,
                )
            except Exception as exc:
                logger.debug("Envanter olayı yayınlanamadı: %s", exc)

        logger.info(
            "Sistem envanteri alındı — %s | %d disk | GPU: %s | %.0f ms",
            rapor.hostname or "?",
            len(rapor.disks),
            "var" if rapor.gpu.get("available") else "yok",
            rapor.duration_ms,
        )
        return rapor

    # ─── Bölüm Toplayıcıları ────────────────────────────────────────────

    def _hostname(self) -> str:
        try:
            return socket.gethostname()
        except Exception:
            return platform.node() or ""

    def _user(self) -> str:
        try:
            return getpass.getuser()
        except Exception:
            return os.environ.get("USERNAME") or os.environ.get("USER") or ""

    def _elevated(self) -> bool:
        """Süreç yönetici yetkisiyle mi çalışıyor? (Windows dışında False)."""
        if sys.platform != "win32":
            try:
                return os.geteuid() == 0  # type: ignore[attr-defined]
            except Exception:
                return False
        try:
            import ctypes

            return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
        except Exception:
            return False

    def _collect_os(self) -> Dict[str, Any]:
        """İşletim sistemi + önyükleme bilgileri."""
        bilgi: Dict[str, Any] = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
        }

        try:
            win = platform.win32_ver()
            bilgi["win32_version"] = win[0]
            bilgi["win32_build"] = win[1]
            bilgi["win32_csd"] = win[2]
            bilgi["win32_ptype"] = win[3]
        except Exception as exc:
            logger.debug("win32_ver alınamadı: %s", exc)

        try:
            edition = platform.win32_edition()  # type: ignore[attr-defined]
            bilgi["edition"] = edition or ""
        except Exception:
            bilgi["edition"] = ""

        try:
            import psutil

            boot = psutil.boot_time()
            bilgi["boot_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(boot))
            bilgi["uptime_seconds"] = max(0.0, time.time() - boot)
        except Exception as exc:
            logger.debug("Önyükleme süresi alınamadı: %s", exc)

        return bilgi

    def _collect_python(self) -> Dict[str, Any]:
        """Python yorumlayıcı bilgileri."""
        bilgi: Dict[str, Any] = {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        }
        try:
            bilgi["virtual_env"] = os.environ.get("VIRTUAL_ENV", "")
            bilgi["in_venv"] = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        except Exception as exc:
            logger.debug("Sanal ortam bilgisi alınamadı: %s", exc)
        return bilgi

    def _collect_cpu(self) -> Dict[str, Any]:
        """CPU çekirdek, frekans ve kullanım bilgileri."""
        bilgi: Dict[str, Any] = {
            "processor": platform.processor() or platform.machine(),
            "architecture": platform.machine(),
        }
        import psutil

        try:
            bilgi["count_physical"] = psutil.cpu_count(logical=False) or 0
        except Exception:
            bilgi["count_physical"] = 0
        try:
            bilgi["count_logical"] = psutil.cpu_count(logical=True) or 0
        except Exception:
            bilgi["count_logical"] = 0
        try:
            frekans = psutil.cpu_freq()
            if frekans:
                bilgi["freq_current_mhz"] = round(float(frekans.current), 1)
                bilgi["freq_max_mhz"] = round(float(frekans.max or 0.0), 1)
                bilgi["freq_min_mhz"] = round(float(frekans.min or 0.0), 1)
        except Exception as exc:
            logger.debug("CPU frekansı alınamadı: %s", exc)
        try:
            bilgi["per_core_percent"] = [round(float(v), 1) for v in psutil.cpu_percent(interval=None, percpu=True)]
            bilgi["usage_percent"] = round(float(psutil.cpu_percent(interval=None)), 1)
        except Exception as exc:
            logger.debug("CPU kullanımı alınamadı: %s", exc)
        return bilgi

    def _collect_memory(self) -> Dict[str, Any]:
        """RAM + takas (swap) bilgileri (MB)."""
        import psutil

        mb = 1024 * 1024
        sanal = psutil.virtual_memory()
        bilgi: Dict[str, Any] = {
            "total_mb": round(sanal.total / mb, 1),
            "available_mb": round(sanal.available / mb, 1),
            "used_mb": round(sanal.used / mb, 1),
            "percent": round(float(sanal.percent), 1),
        }
        try:
            takas = psutil.swap_memory()
            bilgi["swap_total_mb"] = round(takas.total / mb, 1)
            bilgi["swap_used_mb"] = round(takas.used / mb, 1)
            bilgi["swap_percent"] = round(float(takas.percent), 1)
        except Exception as exc:
            logger.debug("Takas belleği alınamadı: %s", exc)
        return bilgi

    def _collect_disks(self) -> List[Dict[str, Any]]:
        """Tüm disk bölümleri ve doluluk oranları (hatalı sürücüler atlanır)."""
        import psutil

        gb = 1024 ** 3
        diskler: List[Dict[str, Any]] = []
        for bolum in psutil.disk_partitions(all=False):
            try:
                kullanim = psutil.disk_usage(bolum.mountpoint)
            except Exception:
                continue  # boş sürücü / izin yok → sessizce atla
            diskler.append(
                {
                    "device": bolum.device,
                    "mountpoint": bolum.mountpoint,
                    "fstype": bolum.fstype,
                    "opts": bolum.opts,
                    "total_gb": round(kullanim.total / gb, 2),
                    "used_gb": round(kullanim.used / gb, 2),
                    "free_gb": round(kullanim.free / gb, 2),
                    "percent": round(float(kullanim.percent), 1),
                }
            )
        return diskler

    def _collect_gpu(self) -> Dict[str, Any]:
        """
        GPU bilgisi: önce `torch.cuda`, sonra `nvidia-smi` alt süreci.

        Hiçbiri yoksa ``{"available": False, ...}`` döner.
        """
        torch_bilgi = self._gpu_from_torch()
        if torch_bilgi.get("available"):
            return torch_bilgi

        smi_bilgi = self._gpu_from_nvidia_smi()
        if smi_bilgi.get("available"):
            return smi_bilgi

        return {
            "available": False,
            "source": "",
            "devices": [],
            "summary": "yok",
            "note": torch_bilgi.get("note", "") or smi_bilgi.get("note", ""),
        }

    def _gpu_from_torch(self) -> Dict[str, Any]:
        """`torch.cuda` üzerinden GPU bilgisi."""
        try:
            import torch
        except Exception:
            return {"available": False, "note": "torch yok"}

        try:
            if not torch.cuda.is_available():
                return {"available": False, "note": "CUDA yok"}
        except Exception as exc:
            return {"available": False, "note": f"CUDA sorgusu başarısız: {exc}"}

        mb = 1024 * 1024
        aygitlar: List[Dict[str, Any]] = []
        try:
            adet = torch.cuda.device_count()
        except Exception:
            adet = 0

        for i in range(adet):
            try:
                ozellik = torch.cuda.get_device_properties(i)
                try:
                    serbest, toplam = torch.cuda.mem_get_info(i)
                except Exception:
                    serbest, toplam = 0, getattr(ozellik, "total_memory", 0)
                aygitlar.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "vram_total_mb": round(toplam / mb, 1),
                        "vram_free_mb": round(serbest / mb, 1),
                        "capability": f"{ozellik.major}.{ozellik.minor}",
                        "multi_processor_count": getattr(ozellik, "multi_processor_count", 0),
                    }
                )
            except Exception as exc:
                logger.debug("CUDA aygıtı %d okunamadı: %s", i, exc)

        cuda_surum = ""
        try:
            cuda_surum = str(torch.version.cuda or "")
        except Exception:
            pass

        return {
            "available": bool(aygitlar),
            "source": "torch.cuda",
            "devices": aygitlar,
            "cuda_version": cuda_surum,
            "torch_version": getattr(torch, "__version__", ""),
            "driver_version": "",
            "summary": ", ".join(a["name"] for a in aygitlar) if aygitlar else "yok",
        }

    def _gpu_from_nvidia_smi(self) -> Dict[str, Any]:
        """`nvidia-smi` çıktısını ayrıştırarak GPU bilgisi."""
        import shutil

        exe = shutil.which("nvidia-smi")
        if not exe:
            return {"available": False, "note": "nvidia-smi yok"}

        komut = [
            exe,
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
        try:
            sonuc = subprocess.run(
                komut,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
                creationflags=_no_window_flags(),
            )
        except Exception as exc:
            return {"available": False, "note": f"nvidia-smi çalıştırılamadı: {exc}"}

        if sonuc.returncode != 0:
            return {"available": False, "note": "nvidia-smi hata döndürdü"}

        aygitlar: List[Dict[str, Any]] = []
        surucu = ""
        for satir in (sonuc.stdout or "").splitlines():
            parcalar = [p.strip() for p in satir.split(",")]
            if len(parcalar) < 4:
                continue
            try:
                toplam = float(parcalar[1])
                serbest = float(parcalar[2])
            except Exception:
                toplam, serbest = 0.0, 0.0
            surucu = surucu or parcalar[3]
            aygitlar.append(
                {
                    "index": len(aygitlar),
                    "name": parcalar[0],
                    "vram_total_mb": toplam,
                    "vram_free_mb": serbest,
                    "driver_version": parcalar[3],
                }
            )

        return {
            "available": bool(aygitlar),
            "source": "nvidia-smi",
            "devices": aygitlar,
            "driver_version": surucu,
            "cuda_version": "",
            "summary": ", ".join(a["name"] for a in aygitlar) if aygitlar else "yok",
        }

    def _collect_network(self) -> Dict[str, Any]:
        """Ana makine adı, birincil IP ve arayüz adresleri."""
        bilgi: Dict[str, Any] = {"hostname": self._hostname()}

        try:
            bilgi["primary_ip"] = socket.gethostbyname(socket.gethostname())
        except Exception:
            bilgi["primary_ip"] = "127.0.0.1"

        arayuzler: Dict[str, List[str]] = {}
        try:
            import psutil

            for ad, adresler in psutil.net_if_addrs().items():
                toplu: List[str] = []
                for adres in adresler:
                    try:
                        toplu.append(str(adres.address))
                    except Exception:
                        continue
                arayuzler[ad] = toplu
        except Exception as exc:
            logger.debug("Ağ arayüzleri alınamadı: %s", exc)

        bilgi["interfaces"] = arayuzler
        bilgi["interface_count"] = len(arayuzler)
        return bilgi

    def _collect_environment(self) -> Dict[str, str]:
        """
        Küratörlü ortam değişkeni listesi.

        `KEY/TOKEN/SECRET/PASSWORD` içeren adların **değeri** maskelenir.
        """
        toplu: Dict[str, str] = {}
        for ad in ENV_ALLOWLIST:
            if ad in os.environ:
                toplu[ad] = self._maskele(ad, os.environ[ad])

        for ad, deger in os.environ.items():
            if ad.startswith(ENV_PREFIXES):
                toplu[ad] = self._maskele(ad, deger)

        return toplu

    @staticmethod
    def _maskele(ad: str, deger: str) -> str:
        """Hassas görünen değişkenlerin değerini maskeler."""
        ust = ad.upper()
        if any(isaret in ust for isaret in SENSITIVE_MARKERS):
            return MASK
        return deger

    # ─── Ek Sorgular ────────────────────────────────────────────────────

    def check_connectivity(self, host: str = "1.1.1.1", port: int = 443, timeout: float = 2.0) -> bool:
        """
        TCP bağlantısı ile internet erişimini denetler.

        `snapshot()` içinde **çağrılmaz**; pahalı ve yan etkili olduğu için
        yalnızca istendiğinde çalıştırılır.
        """
        self._connectivity_checks += 1
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception as exc:
            logger.debug("Bağlantı denemesi başarısız (%s:%s): %s", host, port, exc)
            return False

    def installed_python_packages(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Kurulu Python paketlerini listeler.

        Args:
            limit: Döndürülecek maksimum paket sayısı.

        Returns:
            ``[{"name": ..., "version": ...}, …]`` (ada göre sıralı).
        """
        paketler: List[Dict[str, Any]] = []
        try:
            for dagitim in importlib.metadata.distributions():
                try:
                    ad = dagitim.metadata["Name"] if dagitim.metadata else None
                    if not ad:
                        continue
                    paketler.append({"name": str(ad), "version": str(dagitim.version or "")})
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Paket listesi alınamadı: %s", exc)
            return []

        paketler.sort(key=lambda p: p["name"].lower())
        return paketler[: max(0, limit)]

    def write_report(self, path: str) -> str:
        """
        Markdown envanter raporunu diske yazar (üst dizinleri oluşturur).

        Returns:
            Yazılan dosyanın yolu.
        """
        rapor = self._last_report or self.snapshot()
        hedef = Path(os.path.expandvars(os.path.expanduser(str(path))))
        try:
            hedef.parent.mkdir(parents=True, exist_ok=True)
            hedef.write_text(rapor.to_markdown(), encoding="utf-8")
        except Exception as exc:
            logger.warning("Envanter raporu yazılamadı (%s): %s", hedef, exc)
            return ""
        logger.info("Envanter raporu yazıldı: %s", hedef)
        return str(hedef)

    def to_markdown(self, report: Optional[SystemInventoryReport] = None) -> str:
        """Markdown raporu üretir (rapor verilmezse yeni anlık görüntü alır)."""
        hedef = report or self._last_report or self.snapshot()
        return hedef.to_markdown()

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir envanter metrikleri."""
        son = self._last_report
        return {
            "name": self.name,
            "initialized": self._initialized,
            "snapshot_count": self._snapshot_count,
            "last_duration_ms": self._last_duration_ms,
            "error_count": self._error_count,
            "connectivity_checks": self._connectivity_checks,
            "last_snapshot": {
                "timestamp": son.timestamp if son else 0.0,
                "hostname": son.hostname if son else "",
                "elevated": son.elevated if son else False,
                "disk_count": len(son.disks) if son else 0,
                "gpu_available": bool(son.gpu.get("available")) if son else False,
                "errors": list(son.errors) if son else [],
            },
        }


__all__ = ["SystemInventory", "SystemInventoryReport"]
