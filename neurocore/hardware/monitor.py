"""
NeuroCore Donanım İzleme Modülü
================================
CPU, GPU, RAM ve disk kaynaklarını gerçek zamanlı izler.
Eşik değerleri aşıldığında EventBus üzerinden uyarı yayınlar.
"""

from __future__ import annotations

import platform
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import psutil

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("hardware.monitor")


# ─── GPU Bilgisi Yardımcıları ──────────────────────────────────────────────

def _get_gpu_info() -> Optional[Dict[str, Any]]:
    """
    Tek biçimli GPU bilgisi.

    Sıra: `neurocore.platform` (NVML → amdsmi → rocm-smi → sysfs → torch).
    Böylece AMD (ROCm) ve Linux da çalışır; eski NVML kodu yedek olarak kalır.
    """
    try:
        from neurocore.platform import gpu_bilgi

        bilgi = gpu_bilgi()
        if bilgi:
            return bilgi
    except Exception as exc:  # platform katmanı yoksa/bozuksa sessizce devam
        logger.debug(f"platform.gpu_bilgi başarısız: {exc}")

    # ── Yedek: doğrudan NVML (eski davranış) ──────────────────────────────
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = -1
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            power = -1.0
        return {
            "name": name,
            "vram_total_mb": info.total / (1024 ** 2),
            "vram_used_mb": info.used / (1024 ** 2),
            "vram_free_mb": info.free / (1024 ** 2),
            "vram_percent": (info.used / info.total) * 100 if info.total else 0.0,
            "gpu_utilization": util.gpu,
            "memory_utilization": util.memory,
            "temperature_c": temp,
            "power_w": power,
            "available": True,
            "backend": "nvml-direct",
        }
    except Exception:
        return None


# ─── Sistem Anlık Görüntüsü ────────────────────────────────────────────────

@dataclass
class SystemSnapshot:
    """Sistem kaynaklarının anlık görüntüsü."""
    timestamp: float = 0.0
    
    # CPU
    cpu_percent: float = 0.0
    cpu_percent_per_core: List[float] = field(default_factory=list)
    cpu_freq_mhz: float = 0.0
    cpu_count_physical: int = 0
    cpu_count_logical: int = 0
    
    # RAM
    ram_total_mb: float = 0.0
    ram_used_mb: float = 0.0
    ram_free_mb: float = 0.0
    ram_percent: float = 0.0
    swap_total_mb: float = 0.0
    swap_used_mb: float = 0.0
    swap_percent: float = 0.0
    
    # GPU
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_vram_total_mb: float = 0.0
    gpu_vram_used_mb: float = 0.0
    gpu_vram_free_mb: float = 0.0
    gpu_vram_percent: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature_c: int = -1
    gpu_power_w: float = -1.0
    
    # Disk
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_percent: float = 0.0
    
    # Sistem
    os_name: str = ""
    hostname: str = ""
    uptime_seconds: float = 0.0


# ─── Ana İzleme Sınıfı ────────────────────────────────────────────────────

class HardwareMonitor:
    """
    Gerçek zamanlı donanım izleme sistemi.
    
    CPU, RAM, GPU ve disk kaynaklarını periyodik olarak izler.
    Eşik değerleri aşıldığında EventBus üzerinden olay yayınlar.
    
    Kullanım:
        monitor = HardwareMonitor()
        monitor.initialize(config, event_bus)
        monitor.start_monitoring()
        
        snapshot = monitor.get_snapshot()
        print(f"RAM: {snapshot.ram_percent}%")
    """
    
    def __init__(self):
        self._event_bus = None
        self._config = None
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._interval: float = 2.0
        self._history: Deque[SystemSnapshot] = deque(maxlen=300)  # ~10 dk
        self._last_snapshot: Optional[SystemSnapshot] = None
        self._lock = threading.Lock()
        
        # Uyarı durumları (tekrar eden uyarıları önlemek için)
        self._ram_warning_sent = False
        self._ram_critical_sent = False
        self._gpu_warning_sent = False
    
    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        İzleme sistemini yapılandır.
        
        Args:
            config: SystemConfig instance
            event_bus: EventBus instance
        """
        self._config = config
        self._event_bus = event_bus
        
        if config:
            hw = config.hardware
            self._interval = hw.monitor_interval_sec
            self._ram_warn = hw.ram_warning_threshold
            self._ram_crit = hw.ram_critical_threshold
            self._gpu_warn = hw.gpu_warning_threshold
        else:
            self._ram_warn = 0.80
            self._ram_crit = 0.90
            self._gpu_warn = 0.85
        
        logger.info("Donanım izleme sistemi başlatıldı")

    # ─── İzleme Döngüsü ────────────────────────────────────────────────

    def start_monitoring(self) -> None:
        """Arka plan izleme thread'ini başlat."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="HardwareMonitor"
        )
        self._monitor_thread.start()
        logger.info(f"İzleme başlatıldı (aralık: {self._interval}s)")

    def stop_monitoring(self) -> None:
        """Arka plan izleme thread'ini durdur."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("İzleme durduruldu")

    def _monitor_loop(self) -> None:
        """Ana izleme döngüsü (arka plan thread)."""
        while self._monitoring:
            try:
                snapshot = self._collect_snapshot()
                
                with self._lock:
                    self._last_snapshot = snapshot
                    self._history.append(snapshot)
                
                self._check_thresholds(snapshot)
                
            except Exception as e:
                logger.error(f"İzleme hatası: {e}")
            
            time.sleep(self._interval)

    # ─── Veri Toplama ───────────────────────────────────────────────────

    def _collect_snapshot(self) -> SystemSnapshot:
        """Tüm sistem kaynaklarının anlık görüntüsünü topla."""
        snap = SystemSnapshot(timestamp=time.time())
        
        # CPU
        snap.cpu_percent = psutil.cpu_percent(interval=0)
        snap.cpu_percent_per_core = psutil.cpu_percent(interval=0, percpu=True)
        snap.cpu_count_physical = psutil.cpu_count(logical=False) or 0
        snap.cpu_count_logical = psutil.cpu_count(logical=True) or 0
        try:
            freq = psutil.cpu_freq()
            snap.cpu_freq_mhz = freq.current if freq else 0
        except Exception:
            pass
        
        # RAM
        ram = psutil.virtual_memory()
        snap.ram_total_mb = ram.total / (1024 ** 2)
        snap.ram_used_mb = ram.used / (1024 ** 2)
        snap.ram_free_mb = ram.available / (1024 ** 2)
        snap.ram_percent = ram.percent
        
        swap = psutil.swap_memory()
        snap.swap_total_mb = swap.total / (1024 ** 2)
        snap.swap_used_mb = swap.used / (1024 ** 2)
        snap.swap_percent = swap.percent
        
        # GPU
        gpu = _get_gpu_info()
        if gpu:
            snap.gpu_available = True
            snap.gpu_name = gpu["name"]
            snap.gpu_vram_total_mb = gpu["vram_total_mb"]
            snap.gpu_vram_used_mb = gpu["vram_used_mb"]
            snap.gpu_vram_free_mb = gpu["vram_free_mb"]
            snap.gpu_vram_percent = gpu["vram_percent"]
            snap.gpu_utilization = gpu["gpu_utilization"]
            snap.gpu_temperature_c = gpu["temperature_c"]
            snap.gpu_power_w = gpu["power_w"]
        
        # Disk (ana sürücü)
        try:
            disk = psutil.disk_usage("/")
            snap.disk_total_gb = disk.total / (1024 ** 3)
            snap.disk_used_gb = disk.used / (1024 ** 3)
            snap.disk_free_gb = disk.free / (1024 ** 3)
            snap.disk_percent = disk.percent
        except Exception:
            pass
        
        # Sistem bilgisi
        snap.os_name = f"{platform.system()} {platform.release()}"
        snap.hostname = platform.node()
        snap.uptime_seconds = time.time() - psutil.boot_time()
        
        return snap

    # ─── Eşik Kontrolleri ──────────────────────────────────────────────

    def _check_thresholds(self, snap: SystemSnapshot) -> None:
        """Kaynak eşiklerini kontrol et ve gerekirse olay yayınla."""
        if not self._event_bus:
            return
        
        # RAM kontrolleri
        ram_ratio = snap.ram_percent / 100.0
        
        if ram_ratio >= self._ram_crit and not self._ram_critical_sent:
            self._event_bus.emit_critical(
                "hardware.ram_critical",
                {"ram_percent": snap.ram_percent, "free_mb": snap.ram_free_mb},
                source="hardware.monitor"
            )
            self._ram_critical_sent = True
            logger.critical(f"RAM KRİTİK: %{snap.ram_percent:.1f} kullanımda!")
            
        elif ram_ratio >= self._ram_warn and not self._ram_warning_sent:
            self._event_bus.emit_warning(
                "hardware.ram_warning",
                {"ram_percent": snap.ram_percent, "free_mb": snap.ram_free_mb},
                source="hardware.monitor"
            )
            self._ram_warning_sent = True
            logger.warning(f"RAM UYARI: %{snap.ram_percent:.1f} kullanımda")
            
        elif ram_ratio < self._ram_warn * 0.9:  # %10 hysteresis
            self._ram_warning_sent = False
            self._ram_critical_sent = False
        
        # GPU kontrolleri
        if snap.gpu_available:
            gpu_ratio = snap.gpu_vram_percent / 100.0
            if gpu_ratio >= self._gpu_warn and not self._gpu_warning_sent:
                self._event_bus.emit_warning(
                    "hardware.gpu_warning",
                    {
                        "vram_percent": snap.gpu_vram_percent,
                        "vram_free_mb": snap.gpu_vram_free_mb,
                    },
                    source="hardware.monitor"
                )
                self._gpu_warning_sent = True
                logger.warning(f"GPU VRAM UYARI: %{snap.gpu_vram_percent:.1f}")
            elif gpu_ratio < self._gpu_warn * 0.9:
                self._gpu_warning_sent = False

    # ─── Genel Erişim Metodları ────────────────────────────────────────

    def get_snapshot(self) -> SystemSnapshot:
        """Mevcut veya yeni anlık görüntüyü döndür."""
        with self._lock:
            if self._last_snapshot:
                return self._last_snapshot
        return self._collect_snapshot()

    def get_history(self, limit: int = 100) -> List[SystemSnapshot]:
        """Son N anlık görüntüyü döndür."""
        with self._lock:
            return list(self._history)[-limit:]

    def get_cpu_usage(self) -> float:
        """Anlık CPU kullanım yüzdesini döndür."""
        return psutil.cpu_percent(interval=0)

    def get_ram_usage(self) -> Dict[str, float]:
        """RAM kullanım bilgisini döndür."""
        ram = psutil.virtual_memory()
        return {
            "total_mb": ram.total / (1024 ** 2),
            "used_mb": ram.used / (1024 ** 2),
            "free_mb": ram.available / (1024 ** 2),
            "percent": ram.percent,
        }

    def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """GPU bilgisini döndür (yoksa None)."""
        return _get_gpu_info()

    def is_gpu_available(self) -> bool:
        """GPU kullanılabilir mi?"""
        return _get_gpu_info() is not None

    def get_total_ram_gb(self) -> float:
        """Toplam RAM (GB)."""
        return psutil.virtual_memory().total / (1024 ** 3)

    def get_gpu_vram_gb(self) -> float:
        """GPU VRAM (GB). GPU yoksa 0."""
        gpu = _get_gpu_info()
        return (gpu["vram_total_mb"] / 1024) if gpu else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Kısa sistem özeti döndür."""
        snap = self.get_snapshot()
        return {
            "cpu": f"{snap.cpu_percent:.1f}%",
            "ram": f"{snap.ram_percent:.1f}% ({snap.ram_free_mb:.0f}MB boş)",
            "gpu": (
                f"{snap.gpu_name} - VRAM {snap.gpu_vram_percent:.1f}% "
                f"({snap.gpu_vram_free_mb:.0f}MB boş)"
                if snap.gpu_available else "Yok"
            ),
            "disk": f"{snap.disk_percent:.1f}% ({snap.disk_free_gb:.1f}GB boş)",
            "os": snap.os_name,
        }

    # ─── Yaşam Döngüsü ────────────────────────────────────────────────

    def shutdown(self) -> None:
        """İzleme sistemini kapat."""
        self.stop_monitoring()
        logger.info("Donanım izleme sistemi kapatıldı")
