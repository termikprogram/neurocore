"""
NeuroCore — platform capability layer (P1).

Tek yerden yanıtlar:
    sistem()        → işletim sistemi, mimari, Python sürümü
    kabuk(komut)    → platforma uygun `subprocess` argv'si
    gpu_arka_ucu()  → nvml | amdsmi | rocm-smi | sysfs | torch | cpu
    gpu_bilgi()     → tek biçimli GPU sözlüğü (NVIDIA + AMD + yok)
    ui_otomasyon()  → pywinauto kullanılabilir mi (Windows)
    ozellikler()    → özellik bayrakları (UI, PowerShell, CUDA, ROCm…)

Neden: RTX Spark (Windows + Arm64) ve AMD AI Halo (Linux + ROCm) hedeflerinde
kodun geri kalanı **değişmeden** çalışsın; platform farkları burada toplansın.
"""

from .tespit import (
    SistemBilgisi,
    gpu_arka_ucu,
    gpu_bilgi,
    kabuk,
    ozellikler,
    sistem,
    ui_otomasyon,
)

__all__ = [
    "SistemBilgisi",
    "gpu_arka_ucu",
    "gpu_bilgi",
    "kabuk",
    "ozellikler",
    "sistem",
    "ui_otomasyon",
    "kabuk_ad",
]
