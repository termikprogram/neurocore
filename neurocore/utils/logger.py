"""
NeuroCore Loglama Sistemi
==========================
Renkli konsol çıktısı, dosya rotasyonu, yapılandırılmış loglama ve
performans ölçüm context manager'ları sunan gelişmiş loglama altyapısı.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


# ─── Renk Kodları (Windows + ANSI uyumlu) ──────────────────────────────────

class _Colors:
    """ANSI renk kodları."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BG_RED = "\033[41m"

# Windows'da ANSI renk desteğini etkinleştir
if sys.platform == "win32":
    os.system("")  # ANSI escape kodlarını aktif eder


# ─── Renkli Formatter ──────────────────────────────────────────────────────

class ColoredFormatter(logging.Formatter):
    """Seviyeye göre renkli log formatı."""
    
    LEVEL_COLORS = {
        logging.DEBUG: _Colors.DIM + _Colors.CYAN,
        logging.INFO: _Colors.GREEN,
        logging.WARNING: _Colors.YELLOW,
        logging.ERROR: _Colors.RED,
        logging.CRITICAL: _Colors.BG_RED + _Colors.WHITE + _Colors.BOLD,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, _Colors.WHITE)
        
        # Modül adını kısalt
        module = record.name.replace("neurocore.", "")
        
        # Zaman damgası
        time_str = self.formatTime(record, "%H:%M:%S")
        
        # Seviye kısaltması
        level_short = {
            "DEBUG": "DBG", "INFO": "INF", "WARNING": "WRN",
            "ERROR": "ERR", "CRITICAL": "CRT"
        }.get(record.levelname, record.levelname[:3])
        
        msg = record.getMessage()
        
        formatted = (
            f"{_Colors.DIM}{time_str}{_Colors.RESET} "
            f"{color}{_Colors.BOLD}[{level_short}]{_Colors.RESET} "
            f"{_Colors.CYAN}{module}{_Colors.RESET} "
            f"{color}{msg}{_Colors.RESET}"
        )
        
        if record.exc_info and record.exc_info[1]:
            formatted += f"\n{_Colors.RED}{self.formatException(record.exc_info)}{_Colors.RESET}"
        
        return formatted


class FileFormatter(logging.Formatter):
    """Dosya için yapılandırılmış (JSON-benzeri) format."""
    
    def format(self, record: logging.LogRecord) -> str:
        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        module = record.name
        msg = record.getMessage()
        
        line = f"[{time_str}] [{record.levelname:8s}] [{module}] {msg}"
        
        if record.exc_info and record.exc_info[1]:
            line += f"\n{self.formatException(record.exc_info)}"
        
        return line


# ─── Ana Logger Sınıfı ─────────────────────────────────────────────────────

class NeuroCoreLogger:
    """
    NeuroCore loglama sistemi.
    
    Özellikler:
        - Renkli konsol çıktısı
        - Dosya loglama (rotasyonlu)
        - Modül bazlı log seviyeleri
        - Performans ölçüm context manager'ı
        - Singleton pattern
    
    Kullanım:
        logger = NeuroCoreLogger.get_logger("hardware.monitor")
        logger.info("CPU kullanımı: %45")
        
        with NeuroCoreLogger.timer("embedding_hesaplama"):
            # ... işlem ...
            pass
    """
    
    _initialized: bool = False
    _log_dir: Optional[Path] = None
    _base_level: int = logging.INFO
    _module_levels: dict = {}
    
    @classmethod
    def initialize(
        cls,
        log_dir: Optional[str] = None,
        level: str = "INFO",
        console: bool = True,
        file: bool = True,
        max_file_size_mb: int = 10,
        backup_count: int = 5,
        force: bool = False,
    ) -> None:
        """
        Loglama sistemini başlat.
        
        Args:
            log_dir: Log dosyaları dizini
            level: Varsayılan log seviyesi
            console: Konsol çıktısı etkin mi
            file: Dosya loglaması etkin mi
            max_file_size_mb: Maksimum log dosyası boyutu (MB)
            backup_count: Saklanacak eski log dosyası sayısı
            force: Zaten başlatılmışsa yeniden yapılandır (CLI seviye değişimi)
        """
        if cls._initialized and not force:
            return
        cls._initialized = False
        
        cls._base_level = getattr(logging, level.upper(), logging.INFO)
        
        # Kök logger'ı ayarla
        root_logger = logging.getLogger("neurocore")
        root_logger.setLevel(logging.DEBUG)  # Handler'lar filtreleyecek
        root_logger.handlers.clear()
        
        # Konsol handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(cls._base_level)
            console_handler.setFormatter(ColoredFormatter())
            root_logger.addHandler(console_handler)
        
        # Dosya handler
        if file and log_dir:
            cls._log_dir = Path(log_dir)
            cls._log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                cls._log_dir / "neurocore.log",
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(FileFormatter())
            root_logger.addHandler(file_handler)
        
        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Modül için logger al.
        
        Args:
            name: Modül adı (ör: 'hardware.monitor')
            
        Returns:
            Yapılandırılmış logger
        """
        if not cls._initialized:
            cls.initialize(console=True, file=False)
        
        full_name = f"neurocore.{name}" if not name.startswith("neurocore.") else name
        logger = logging.getLogger(full_name)
        
        # Modül bazlı seviye ayarı
        if name in cls._module_levels:
            logger.setLevel(cls._module_levels[name])
        
        return logger

    @classmethod
    def set_module_level(cls, module: str, level: str) -> None:
        """Belirli bir modülün log seviyesini ayarla."""
        level_int = getattr(logging, level.upper(), logging.INFO)
        cls._module_levels[module] = level_int
        logger = logging.getLogger(f"neurocore.{module}")
        logger.setLevel(level_int)

    @classmethod
    @contextmanager
    def timer(cls, operation: str, logger_name: str = "performance"):
        """
        İşlem süresini ölçen context manager.
        
        Kullanım:
            with NeuroCoreLogger.timer("model_inference"):
                result = model(input)
        """
        logger = cls.get_logger(logger_name)
        start = time.perf_counter()
        logger.debug(f"⏱ Başladı: {operation}")
        
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            if elapsed > 5000:
                logger.warning(f"⏱ {operation}: {elapsed:.1f}ms (YAVAŞ!)")
            elif elapsed > 1000:
                logger.info(f"⏱ {operation}: {elapsed:.1f}ms")
            else:
                logger.debug(f"⏱ {operation}: {elapsed:.1f}ms")

    @classmethod
    def log_separator(cls, title: str = "", char: str = "─", width: int = 60,
                      logger_name: str = "system") -> None:
        """Görsel ayırıcı log satırı yaz."""
        logger = cls.get_logger(logger_name)
        if title:
            side = (width - len(title) - 2) // 2
            line = f"{char * side} {title} {char * side}"
        else:
            line = char * width
        logger.info(line)
