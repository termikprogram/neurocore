"""
NeuroCore Performans Profilleme Modülü
=======================================
İşlem süreleri, bellek kullanımı ve throughput ölçümü.
Darboğaz tespiti ve istatistiksel analiz.
"""

from __future__ import annotations

import functools
import statistics
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("hardware.profiler")


# ─── Profil Sonucu ──────────────────────────────────────────────────────────

@dataclass
class ProfileResult:
    """Tek bir profilleme sonucu."""
    operation_name: str
    duration_ms: float
    memory_delta_mb: float = 0.0
    gpu_memory_delta_mb: float = 0.0
    throughput: float = 0.0          # items/sec
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProfileStats:
    """Bir işlem için istatistiksel özet."""
    operation_name: str
    count: int = 0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    total_ms: float = 0.0
    mean_memory_delta_mb: float = 0.0
    mean_throughput: float = 0.0


# ─── Ana Profilleme Sınıfı ──────────────────────────────────────────────────

class PerformanceProfiler:
    """
    Performans profilleme ve darboğaz tespiti.
    
    Context manager ve decorator olarak kullanılabilir.
    
    Kullanım:
        profiler = PerformanceProfiler()
        
        # Context manager
        with profiler.profile("model_inference"):
            result = model(input)
        
        # Decorator
        @profiler.profile_function
        def train_step(data):
            ...
        
        # İstatistikler
        stats = profiler.get_stats("model_inference")
        print(f"Ortalama: {stats.mean_ms:.1f}ms")
    """

    def __init__(self):
        self._results: Dict[str, List[ProfileResult]] = defaultdict(list)
        self._lock = threading.Lock()
        self._active: Dict[str, float] = {}  # operation -> start_time
        self._event_bus = None
        self._config = None

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Profilleme sistemini yapılandır."""
        self._config = config
        self._event_bus = event_bus
        logger.info("Performans profilleme sistemi başlatıldı")

    # ─── Profilleme ─────────────────────────────────────────────────────

    @contextmanager
    def profile(self, name: str, num_items: int = 1):
        """
        İşlem profilleme context manager'ı.
        
        Args:
            name: İşlem adı
            num_items: İşlenen öğe sayısı (throughput hesabı için)
            
        Yields:
            None
        """
        import psutil
        process = psutil.Process()
        
        # Başlangıç ölçümleri
        start_time = time.perf_counter()
        start_mem = process.memory_info().rss / (1024 ** 2)
        
        gpu_start = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_start = torch.cuda.memory_allocated() / (1024 ** 2)
        except ImportError:
            pass
        
        try:
            yield
        finally:
            # Bitiş ölçümleri
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            end_mem = process.memory_info().rss / (1024 ** 2)
            
            gpu_delta = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_delta = torch.cuda.memory_allocated() / (1024 ** 2) - gpu_start
            except ImportError:
                pass
            
            throughput = (num_items / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0
            
            result = ProfileResult(
                operation_name=name,
                duration_ms=elapsed_ms,
                memory_delta_mb=end_mem - start_mem,
                gpu_memory_delta_mb=gpu_delta,
                throughput=throughput,
            )
            
            with self._lock:
                self._results[name].append(result)
            
            # Yavaş işlem uyarıları
            if elapsed_ms > 5000:
                logger.warning(f"⚠ YAVAŞ İŞLEM: {name} = {elapsed_ms:.1f}ms")
            elif elapsed_ms > 1000:
                logger.info(f"⏱ {name} = {elapsed_ms:.1f}ms")

    def profile_function(self, func: Optional[Callable] = None, name: Optional[str] = None):
        """
        Fonksiyon profilleme decorator'ı.
        
        Kullanım:
            @profiler.profile_function
            def my_function():
                ...
            
            @profiler.profile_function(name="custom_name")
            def another_function():
                ...
        """
        def decorator(fn: Callable) -> Callable:
            op_name = name or f"{fn.__module__}.{fn.__qualname__}"
            
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.profile(op_name):
                    return fn(*args, **kwargs)
            
            return wrapper
        
        if func is not None:
            return decorator(func)
        return decorator

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self, name: str) -> Optional[ProfileStats]:
        """Belirli bir işlem için istatistikleri hesapla."""
        with self._lock:
            results = self._results.get(name, [])
        
        if not results:
            return None
        
        durations = [r.duration_ms for r in results]
        sorted_d = sorted(durations)
        n = len(sorted_d)
        
        stats = ProfileStats(
            operation_name=name,
            count=n,
            mean_ms=statistics.mean(durations),
            std_ms=statistics.stdev(durations) if n > 1 else 0.0,
            min_ms=sorted_d[0],
            max_ms=sorted_d[-1],
            p50_ms=sorted_d[n // 2],
            p95_ms=sorted_d[int(n * 0.95)] if n >= 20 else sorted_d[-1],
            p99_ms=sorted_d[int(n * 0.99)] if n >= 100 else sorted_d[-1],
            total_ms=sum(durations),
            mean_memory_delta_mb=statistics.mean([r.memory_delta_mb for r in results]),
            mean_throughput=statistics.mean([r.throughput for r in results]),
        )
        
        return stats

    def get_all_stats(self) -> Dict[str, ProfileStats]:
        """Tüm işlemler için istatistikleri döndür."""
        with self._lock:
            names = list(self._results.keys())
        return {name: self.get_stats(name) for name in names if self.get_stats(name)}

    def detect_bottlenecks(self, threshold_ms: float = 1000.0) -> List[Dict[str, Any]]:
        """
        Darboğazları tespit et.
        
        Args:
            threshold_ms: Darboğaz eşiği (ms)
            
        Returns:
            Darboğaz bilgilerinin listesi
        """
        bottlenecks = []
        all_stats = self.get_all_stats()
        
        for name, stats in all_stats.items():
            if stats.mean_ms > threshold_ms:
                bottlenecks.append({
                    "operation": name,
                    "mean_ms": round(stats.mean_ms, 1),
                    "max_ms": round(stats.max_ms, 1),
                    "p95_ms": round(stats.p95_ms, 1),
                    "count": stats.count,
                    "severity": "critical" if stats.mean_ms > 5000 else "warning",
                })
        
        bottlenecks.sort(key=lambda x: x["mean_ms"], reverse=True)
        return bottlenecks

    def export_report(self) -> Dict[str, Any]:
        """Kapsamlı profil raporu oluştur."""
        all_stats = self.get_all_stats()
        bottlenecks = self.detect_bottlenecks()
        
        return {
            "summary": {
                "total_operations": len(all_stats),
                "total_measurements": sum(s.count for s in all_stats.values()),
                "bottleneck_count": len(bottlenecks),
            },
            "operations": {
                name: {
                    "count": s.count,
                    "mean_ms": round(s.mean_ms, 2),
                    "p95_ms": round(s.p95_ms, 2),
                    "total_ms": round(s.total_ms, 2),
                }
                for name, s in all_stats.items()
            },
            "bottlenecks": bottlenecks,
        }

    def reset(self, name: Optional[str] = None) -> None:
        """Profil verilerini sıfırla."""
        with self._lock:
            if name:
                self._results.pop(name, None)
            else:
                self._results.clear()

    def shutdown(self) -> None:
        """Profilleme sistemini kapat."""
        report = self.export_report()
        if report["summary"]["total_measurements"] > 0:
            logger.info(f"Profil özeti: {report['summary']}")
        logger.info("Profilleme sistemi kapatıldı")
