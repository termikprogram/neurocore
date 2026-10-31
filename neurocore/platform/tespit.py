"""
Platform yetenek tespiti — Windows/Arm64 (RTX Spark) ve Linux/ROCm (AI Halo) dahil.

Tasarım kuralları:
  * Hiçbir platforma özel paket **zorunlu değil**; hepsi isteğe bağlı.
  * Tespit sonuçları önbelleklenir (her karede NVML/amdsmi çağırmayalım).
  * GPU bilgisi **tek sözlük** biçiminde döner; çağıran taraf arka ucu bilmez.
"""

from __future__ import annotations

import json
import os
import platform as _pyplatform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

__all__ = [
    "SistemBilgisi",
    "sistem",
    "kabuk",
    "kabuk_ad",
    "gpu_arka_ucu",
    "gpu_bilgi",
    "ui_otomasyon",
    "ozellikler",
]


# ── sistem ──────────────────────────────────────────────────────────────────
def _mimari_norm(ham: str) -> str:
    """`AMD64`/`x86_64`/`aarch64`/`arm64` → `x86_64` | `arm64` | diğer."""
    a = (ham or "").lower()
    if a in ("amd64", "x86_64", "x64", "i386", "i686", "x86"):
        return "x86_64"
    if a in ("arm64", "aarch64", "armv8", "armv8l"):
        return "arm64"
    return a or "unknown"


@dataclass(frozen=True)
class SistemBilgisi:
    """Çalıştığımız ortamın özeti."""

    os: str                    # windows | linux | macos | other
    arch: str                  # x86_64 | arm64
    python: str
    is_64bit: bool = True
    ek: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_windows(self) -> bool:
        return self.os == "windows"

    @property
    def is_linux(self) -> bool:
        return self.os == "linux"

    @property
    def is_macos(self) -> bool:
        return self.os == "macos"

    @property
    def is_arm(self) -> bool:
        return self.arch == "arm64"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "os": self.os,
            "arch": self.arch,
            "python": self.python,
            "is_64bit": self.is_64bit,
            "is_arm": self.is_arm,
            **self.ek,
        }


@lru_cache(maxsize=1)
def sistem() -> SistemBilgisi:
    """İşletim sistemi + mimariyi (tek sefer) tespit eder."""
    if sys.platform.startswith("win"):
        ad = "windows"
    elif sys.platform.startswith("linux"):
        ad = "linux"
    elif sys.platform == "darwin":
        ad = "macos"
    else:
        ad = "other"

    makine = _pyplatform.machine()
    arch = _mimari_norm(makine)
    # Windows'ta PROCESSOR_ARCHITECTURE daha güvenilir (Arm64 emülasyonu)
    if ad == "windows":
        env_arch = os.environ.get("PROCESSOR_ARCHITECTURE", "")
        if _mimari_norm(env_arch) == "arm64":
            arch = "arm64"

    return SistemBilgisi(
        os=ad,
        arch=arch,
        python=f"{sys.version_info.major}.{sys.version_info.minor}."
               f"{sys.version_info.micro}",
        is_64bit=sys.maxsize > 2 ** 32,
        ek={
            "machine": makine,
            "release": _pyplatform.release(),
            "emulated": bool(
                ad == "windows" and _mimari_norm(makine) == "x86_64"
                and _mimari_norm(os.environ.get("PROCESSOR_ARCHITECTURE", ""))
                == "arm64"
            ),
        },
    )


# ── kabuk ───────────────────────────────────────────────────────────────────
def kabuk(komut: str) -> List[str]:
    """
    Verilen komutu platforma uygun `subprocess` argv'sine çevirir.

    Windows : powershell -NoProfile -Command "<komut>"
    Linux   : bash -lc "<komut>"   (macOS dahil)
    """
    if sistem().is_windows:
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", komut]
    return ["bash", "-lc", komut]


def kabuk_ad() -> str:
    """Etkin kabuğun adı (log/UI için)."""
    return "powershell" if sistem().is_windows else "bash"


# ── UI otomasyonu ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def ui_otomasyon() -> bool:
    """
    `pywinauto` kullanılabilir mi?

    Windows + Arm64 için `comtypes` tekerleği yok → False.
    Linux/macOS → False (masaüstü okuma AT-SPI'ye düşer, sonraki faz).
    """
    if not sistem().is_windows:
        return False
    try:
        import pywinauto  # noqa: F401
        import comtypes  # noqa: F401
    except Exception:
        return False
    return True


# ── GPU arka uçları ─────────────────────────────────────────────────────────
def _nvml_bilgi() -> Optional[Dict[str, Any]]:
    """NVIDIA: NVML (Windows, Linux x86/arm64)."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        ad = pynvml.nvmlDeviceGetName(h)
        if isinstance(ad, bytes):
            ad = ad.decode("utf-8", "replace")
        try:
            sicaklik = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            sicaklik = -1
        try:
            guc = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            guc = -1.0
        return {
            "name": ad,
            "vram_total_mb": mem.total / (1024 ** 2),
            "vram_used_mb": mem.used / (1024 ** 2),
            "vram_free_mb": mem.free / (1024 ** 2),
            "vram_percent": (mem.used / mem.total) * 100 if mem.total else 0.0,
            "gpu_utilization": getattr(util, "gpu", 0),
            "memory_utilization": getattr(util, "memory", 0),
            "temperature_c": sicaklik,
            "power_w": guc,
            "available": True,
            "backend": "nvml",
        }
    except Exception:
        return None


def _amdsmi_bilgi() -> Optional[Dict[str, Any]]:
    """AMD: amdsmi (ROCm resmî Python bağlayıcısı)."""
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        isaretler = amdsmi.amdsmi_get_processor_handles()
        if not isaretler:
            return None
        g = isaretler[0]
        ad = "AMD GPU"
        try:
            ad = amdsmi.amdsmi_get_gpu_asic_info(g).get("market_name", ad)
        except Exception:
            pass
        vram = amdsmi.amdsmi_get_gpu_vram_usage(g)
        toplam = float(vram.get("vram_total", 0))
        kullanilan = float(vram.get("vram_used", 0))
        try:
            akt = amdsmi.amdsmi_get_gpu_activity(g)
            gpu_util = float(akt.get("gfx_activity", 0))
            bellek_util = float(akt.get("umc_activity", 0))
        except Exception:
            gpu_util = bellek_util = 0.0
        try:
            sicaklik = float(
                amdsmi.amdsmi_get_temp_metric(
                    g,
                    amdsmi.AmdSmiTemperatureType.EDGE,
                    amdsmi.AmdSmiTemperatureMetric.CURRENT,
                )
            )
        except Exception:
            sicaklik = -1.0
        try:
            guc = float(amdsmi.amdsmi_get_power_info(g).get("average_socket_power", -1))
        except Exception:
            guc = -1.0
        return {
            "name": ad,
            "vram_total_mb": toplam / (1024 ** 2),
            "vram_used_mb": kullanilan / (1024 ** 2),
            "vram_free_mb": (toplam - kullanilan) / (1024 ** 2),
            "vram_percent": (kullanilan / toplam * 100) if toplam else 0.0,
            "gpu_utilization": gpu_util,
            "memory_utilization": bellek_util,
            "temperature_c": sicaklik,
            "power_w": guc,
            "available": True,
            "backend": "amdsmi",
        }
    except Exception:
        return None


def _rocm_smi_bilgi() -> Optional[Dict[str, Any]]:
    """AMD: `rocm-smi --json` (amdsmi yoksa CLI yedeği)."""
    if not shutil.which("rocm-smi"):
        return None
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse",
             "--showtemp", "--showpower", "--json"],
            capture_output=True, text=True, timeout=8,
        )
        veri = json.loads(out.stdout or "{}")
        if not veri:
            return None
        kart = next(iter(veri.values()))
        toplam = float(kart.get("VRAM Total Memory (B)", 0))
        kullanilan = float(kart.get("VRAM Total Used Memory (B)", 0))
        gpu_util = float(str(kart.get("GPU use (%)", "0")).replace("%", "") or 0)
        try:
            sicaklik = float(str(kart.get("Temperature (Sensor edge) (C)", "-1")))
        except Exception:
            sicaklik = -1.0
        try:
            guc = float(kart.get("Average Graphics Package Power (W)", -1))
        except Exception:
            guc = -1.0
        return {
            "name": kart.get("Card series", "AMD GPU"),
            "vram_total_mb": toplam / (1024 ** 2),
            "vram_used_mb": kullanilan / (1024 ** 2),
            "vram_free_mb": (toplam - kullanilan) / (1024 ** 2),
            "vram_percent": (kullanilan / toplam * 100) if toplam else 0.0,
            "gpu_utilization": gpu_util,
            "memory_utilization": 0.0,
            "temperature_c": sicaklik,
            "power_w": guc,
            "available": True,
            "backend": "rocm-smi",
        }
    except Exception:
        return None


def _sysfs_bilgi() -> Optional[Dict[str, Any]]:
    """Linux amdgpu sysfs (rocm-smi de yoksa)."""
    if not sistem().is_linux:
        return None
    from glob import glob

    for kart in sorted(glob("/sys/class/drm/card*/device")):
        try:
            with open(os.path.join(kart, "mem_info_vram_total")) as f:
                toplam = float(f.read().strip())
            with open(os.path.join(kart, "mem_info_vram_used")) as f:
                kullanilan = float(f.read().strip())
        except Exception:
            continue
        gpu_util = 0.0
        try:
            with open(os.path.join(kart, "gpu_busy_percent")) as f:
                gpu_util = float(f.read().strip())
        except Exception:
            pass
        return {
            "name": "AMD GPU (sysfs)",
            "vram_total_mb": toplam / (1024 ** 2),
            "vram_used_mb": kullanilan / (1024 ** 2),
            "vram_free_mb": (toplam - kullanilan) / (1024 ** 2),
            "vram_percent": (kullanilan / toplam * 100) if toplam else 0.0,
            "gpu_utilization": gpu_util,
            "memory_utilization": 0.0,
            "temperature_c": -1.0,
            "power_w": -1.0,
            "available": True,
            "backend": "sysfs",
        }
    return None


def _torch_bilgi() -> Optional[Dict[str, Any]]:
    """Son yedek: torch.cuda (bellek var, kullanım yüzdesi yok)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        ad = torch.cuda.get_device_name(0)
        toplam = float(torch.cuda.get_device_properties(0).total_memory)
        kullanilan = float(torch.cuda.memory_allocated(0))
        return {
            "name": ad,
            "vram_total_mb": toplam / (1024 ** 2),
            "vram_used_mb": kullanilan / (1024 ** 2),
            "vram_free_mb": (toplam - kullanilan) / (1024 ** 2),
            "vram_percent": (kullanilan / toplam * 100) if toplam else 0.0,
            "gpu_utilization": 0.0,
            "memory_utilization": 0.0,
            "temperature_c": -1.0,
            "power_w": -1.0,
            "available": True,
            "backend": "torch",
        }
    except Exception:
        return None


_ARKA_UCLAR = (
    ("nvml", _nvml_bilgi),
    ("amdsmi", _amdsmi_bilgi),
    ("rocm-smi", _rocm_smi_bilgi),
    ("sysfs", _sysfs_bilgi),
    ("torch", _torch_bilgi),
)


@lru_cache(maxsize=1)
def gpu_arka_ucu() -> str:
    """İlk çalışan GPU arka ucunun adı; yoksa `cpu`."""
    for ad, islev in _ARKA_UCLAR:
        try:
            if islev():
                return ad
        except Exception:
            continue
    return "cpu"


def gpu_bilgi(yenile: bool = False) -> Optional[Dict[str, Any]]:
    """
    Tek biçimli GPU bilgisi (yoksa None).

    `yenile=True` önbelleği atlar; varsayılan olarak 2 saniyelik önbellek
    kullanılır (monitor her saniye çağırıyor).
    """
    if not yenile:
        onbellek = _GPU_ONBELLEK
        if onbellek and (time.monotonic() - onbellek[0]) < 2.0:
            return onbellek[1]

    sonuc: Optional[Dict[str, Any]] = None
    for ad, islev in _ARKA_UCLAR:
        try:
            sonuc = islev()
        except Exception:
            sonuc = None
        if sonuc:
            break

    _GPU_ONBELLEK[0] = time.monotonic()
    _GPU_ONBELLEK[1] = sonuc
    return sonuc


_GPU_ONBELLEK: List[Any] = [0.0, None]


# ── özellik bayrakları ──────────────────────────────────────────────────────
def ozellikler() -> Dict[str, Any]:
    """Tek bakışta platform yetenekleri (UI, kabuk, GPU, dağıtım)."""
    s = sistem()
    arka_uc = gpu_arka_ucu()
    try:
        import torch

        torch_var = True
        cuda = bool(torch.cuda.is_available())
        rocm = bool(getattr(torch.version, "hip", None))
    except Exception:
        torch_var = cuda = rocm = False

    return {
        "os": s.os,
        "arch": s.arch,
        "python": s.python,
        "is_arm": s.is_arm,
        "shell": kabuk_ad(),
        "ui_automation": ui_otomasyon(),
        "gpu_backend": arka_uc,
        "gpu": gpu_bilgi() or {},
        "torch": torch_var,
        "cuda": cuda,
        "rocm": rocm,
        "directml": _directml_var(),
        "notes": _notlar(s, arka_uc),
    }


def _directml_var() -> bool:
    try:
        import torch_directml  # noqa: F401
        return True
    except Exception:
        return False


def _notlar(s: SistemBilgisi, arka_uc: str) -> List[str]:
    """Kullanıcıya gösterilecek kısa uyarılar."""
    n: List[str] = []
    if s.is_windows and s.is_arm:
        n.append(
            "Windows on Arm: torch CUDA tekerleği yok — CPU/DirectML "
            "kullanılıyor; tam hız için WSL2 önerilir."
        )
    if arka_uc == "cpu" and s.is_linux:
        n.append(
            "GPU arka ucu bulunamadı — AMD için amdsmi veya rocm-smi kurun."
        )
    if not ui_otomasyon() and s.is_windows:
        n.append("pywinauto/comtypes yok — masaüstü UI okuma devre dışı.")
    if s.is_linux:
        n.append("Masaüstü UI okuma Linux'ta AT-SPI ile gelecek (P4).")
    return n
