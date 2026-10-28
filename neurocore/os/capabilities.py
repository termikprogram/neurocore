"""
NeuroCore İşletim Sistemi — Yetenek Keşfi (Capabilities)
========================================================
L5 (OS) katmanının **"neler yapabilirim?"** modülü. Sisteme kurulu
çalıştırılabilirleri ve Python kitaplıklarını keşfeder, önbelleğe alır ve
diğer katmanların (planner, evolution) yetenek sorgulamasına sunar.

Tasarım kuralları:
    - Araç yolları `shutil.which` ile; Windows'ta ek olarak `Get-Command`
      yedeği ile aranır.
    - Sürüm sorgusu **her zaman** zaman aşımlıdır → takılan araç sistemi
      kilitleyemez.
    - Ağır modüller (`torch`, `transformers`, `PIL`, …) yalnızca ilgili
      yoklama fonksiyonunun içinde import edilir.

Kullanım:
    cap = CapabilityProbe()
    cap.initialize(config, event_bus)
    if cap.has("git"):
        print(cap.probe_all()["git"].version)
    print(cap.available_tools(category="runtime"))
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("os.capabilities")

#: Araç adı doğrulaması (kabuk enjeksiyonunu önler)
_NAME_RE = re.compile(r"^[A-Za-z0-9_.+\-]{1,64}$")


def _no_window_flags() -> int:
    """Alt süreçlerde konsol penceresi açılmasın."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


# ─── Araç Tanımları ────────────────────────────────────────────────────────

#: Yoklanacak araçlar: ad → {category, args, timeout, note}
TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "git": {"category": "vcs", "args": ["--version"]},
    "node": {"category": "runtime", "args": ["--version"]},
    "npm": {"category": "package_manager", "args": ["--version"], "timeout": 12.0},
    "pnpm": {"category": "package_manager", "args": ["--version"], "timeout": 12.0},
    "python": {"category": "runtime", "args": ["--version"]},
    "pip": {"category": "package_manager", "args": ["--version"], "timeout": 15.0},
    "ffmpeg": {"category": "media", "args": ["-version"]},
    "docker": {"category": "container", "args": ["--version"], "timeout": 8.0},
    "nvidia-smi": {
        "category": "gpu",
        "args": ["--query-gpu=name,driver_version", "--format=csv,noheader"],
    },
    "curl": {"category": "network", "args": ["--version"]},
    "tar": {"category": "archive", "args": ["--version"]},
    "7z": {"category": "archive", "args": []},
    "winget": {"category": "package_manager", "args": ["--version"], "timeout": 12.0},
    "code": {"category": "editor", "args": ["--version"], "timeout": 12.0},
    "rg": {"category": "search", "args": ["--version"]},
    "sqlite3": {"category": "database", "args": ["--version"]},
    "powershell": {
        "category": "shell",
        "args": ["-NoProfile", "-NonInteractive", "-Command", "$PSVersionTable.PSVersion.ToString()"],
        "timeout": 12.0,
    },
    "pwsh": {
        "category": "shell",
        "args": ["-NoProfile", "-NonInteractive", "-Command", "$PSVersionTable.PSVersion.ToString()"],
        "timeout": 12.0,
    },
    "cargo": {"category": "runtime", "args": ["--version"]},
    "go": {"category": "runtime", "args": ["version"]},
    "java": {"category": "runtime", "args": ["-version"], "timeout": 8.0},
    "gcc": {"category": "build", "args": ["--version"]},
    "cmake": {"category": "build", "args": ["--version"]},
    "make": {"category": "build", "args": ["--version"]},
    "msbuild": {"category": "build", "args": ["-version"], "timeout": 12.0},
}

#: Yoklanacak özellik/kütüphane probları
FEATURE_SPECS: Dict[str, Dict[str, str]] = {
    "cuda": {"category": "feature", "note": "CUDA GPU erişilebilirliği"},
    "torch": {"category": "library", "note": "PyTorch"},
    "psutil": {"category": "library", "note": "Süreç/sistem izleme"},
    "soundfile": {"category": "library", "note": "Ses dosyası G/Ç"},
    "PIL": {"category": "library", "note": "Pillow görüntü işleme"},
    "tiktoken": {"category": "library", "note": "Hızlı tokenizer"},
    "transformers": {"category": "library", "note": "HuggingFace modelleri"},
    "sqlite_fts5": {"category": "feature", "note": "SQLite tam metin arama"},
    "long_paths_support": {"category": "feature", "note": "Windows uzun yol desteği"},
}


# ─── Araç Bilgisi ──────────────────────────────────────────────────────────

@dataclass
class ToolInfo:
    """
    Tek bir araç/yetenek hakkında bilgi.

    Attributes:
        name: Araç adı.
        available: Kullanılabilir mi?
        path: Çalıştırılabilir dosya yolu.
        version: Sürüm metni (ilk satır, en fazla 80 karakter).
        category: Kategori (vcs, runtime, library, feature, …).
        note: Ek açıklama / hata gerekçesi.
    """

    name: str
    available: bool
    path: str = ""
    version: str = ""
    category: str = "general"
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dict."""
        return {
            "name": self.name,
            "available": self.available,
            "path": self.path,
            "version": self.version,
            "category": self.category,
            "note": self.note,
        }


# ─── Ana Yoklama Sınıfı ────────────────────────────────────────────────────

class CapabilityProbe:
    """
    Sistem yeteneklerini keşfeden ve önbelleğe alan yoklayıcı.

    Kullanım:
        cap = CapabilityProbe()
        cap.initialize(config, event_bus)
        araclar = cap.probe_all()          # ilk çağrıda yoklar, sonra önbellek
        cap.probe_all(force=True)          # yeniden yokla
    """

    name = "os.capabilities"

    def __init__(self) -> None:
        self._initialized: bool = False
        self._config: Any = None
        self._event_bus: Any = None
        self._cache: Dict[str, ToolInfo] = {}
        self._batch_paths: Dict[str, str] = {}
        self._probed_all: bool = False
        self._probe_count: int = 0
        self._failure_count: int = 0
        self._last_probe_ts: float = 0.0
        self._default_timeout: float = 5.0

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Bağımlılıkları bağlar (idempotent). Yoklama tembeldir."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._initialized = True
        logger.info(
            "Yetenek yoklayıcı hazır — %d araç + %d özellik tanımlı",
            len(TOOL_SPECS),
            len(FEATURE_SPECS),
        )

    def shutdown(self) -> None:
        """Önbelleği boşaltır (idempotent)."""
        if not self._initialized:
            return
        self._initialized = False
        self._cache.clear()
        self._batch_paths.clear()
        self._probed_all = False
        logger.info("Yetenek yoklayıcı kapatıldı")

    # ─── Araç Yoklama ───────────────────────────────────────────────────

    def probe_tool(
        self,
        name: str,
        version_args: Optional[List[str]] = None,
        timeout: float = 5.0,
    ) -> ToolInfo:
        """
        Tek bir aracı yoklar.

        Args:
            name: Araç adı (ör: ``git``).
            version_args: Sürüm sorgu argümanları (None ise tanımdan/`--version`).
            timeout: Sürüm sorgusu zaman aşımı (saniye).

        Returns:
            `ToolInfo` — bulunamazsa ``available=False``.
        """
        self._probe_count += 1
        ad = str(name)
        spec = TOOL_SPECS.get(ad, {})
        kategori = str(spec.get("category", "general"))

        if not _NAME_RE.match(ad):
            self._failure_count += 1
            return ToolInfo(ad, False, "", "", kategori, "geçersiz araç adı")

        yol = ""
        try:
            yol = shutil.which(ad) or ""
        except Exception as exc:
            logger.debug("which(%s) başarısız: %s", ad, exc)

        if not yol and sys.platform == "win32":
            # Toplu çözümleme yapıldıysa oradaki sonuç (boş = bulunamadı) geçerlidir
            yol = self._batch_paths[ad] if ad in self._batch_paths else self._get_command_path(ad)

        if not yol:
            self._failure_count += 1
            bilgi = ToolInfo(ad, False, "", "", kategori, "bulunamadı")
            self._cache[ad] = bilgi
            return bilgi

        if version_args is None:
            version_args = list(spec.get("args", ["--version"]))
        sure = float(spec.get("timeout", timeout) or timeout)

        surum, not_metni = self._read_version(yol, list(version_args), sure)
        bilgi = ToolInfo(ad, True, yol, surum, kategori, not_metni)
        self._cache[ad] = bilgi
        return bilgi

    def _get_command_path(self, name: str) -> str:
        """Windows'ta `Get-Command` ile araç yolunu bulur (yedek yöntem)."""
        kabuk = ""
        try:
            kabuk = shutil.which("powershell") or shutil.which("pwsh") or ""
        except Exception:
            kabuk = ""
        if not kabuk:
            return ""

        betik = (
            f"(Get-Command -Name '{name}' -ErrorAction SilentlyContinue | "
            f"Select-Object -First 1).Source"
        )
        try:
            sonuc = subprocess.run(
                [kabuk, "-NoProfile", "-NonInteractive", "-Command", betik],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
                creationflags=_no_window_flags(),
            )
        except Exception as exc:
            logger.debug("Get-Command(%s) başarısız: %s", name, exc)
            return ""

        for satir in (sonuc.stdout or "").splitlines():
            temiz = satir.strip()
            if temiz and temiz.lower() != "none":
                return temiz
        return ""

    def _resolve_paths_batch(self, names: Sequence[str]) -> Dict[str, str]:
        """
        Windows'ta **tek** PowerShell çağrısıyla birden çok araç yolunu çözer.

        `shutil.which` başarısız olduğunda her araç için ayrı bir PowerShell
        süreci başlatmak yavaştır (~0.7 sn/araç); bu yöntem hepsini tek
        çağrıda çözer.

        Args:
            names: Çözümlenecek araç adları.

        Returns:
            ``{ad: yol}`` — bulunamayanlar yer almaz.
        """
        gecerli = [ad for ad in names if _NAME_RE.match(ad)]
        if not gecerli:
            return {}

        try:
            kabuk = shutil.which("powershell") or shutil.which("pwsh") or ""
        except Exception:
            kabuk = ""
        if not kabuk:
            return {}

        liste = ", ".join(f"'{ad}'" for ad in gecerli)
        betik = (
            f"foreach ($n in @({liste})) {{ "
            f"$c = Get-Command -Name $n -ErrorAction SilentlyContinue | Select-Object -First 1; "
            f"if ($c) {{ Write-Output ($n + '=' + $c.Source) }} }}"
        )
        try:
            sonuc = subprocess.run(
                [kabuk, "-NoProfile", "-NonInteractive", "-Command", betik],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                creationflags=_no_window_flags(),
            )
        except Exception as exc:
            logger.debug("Toplu Get-Command başarısız: %s", exc)
            return {}

        bulunan: Dict[str, str] = {ad: "" for ad in gecerli}
        for satir in (sonuc.stdout or "").splitlines():
            temiz = satir.strip()
            if "=" not in temiz:
                continue
            ad, _, yol = temiz.partition("=")
            ad, yol = ad.strip(), yol.strip()
            if ad and yol and yol.lower() != "none":
                bulunan[ad] = yol
        return bulunan

    def _read_version(self, yol: str, args: Sequence[str], timeout: float) -> tuple[str, str]:
        """
        Aracın sürüm çıktısını okur.

        Returns:
            ``(sürüm_metni, not)`` — hata durumunda not doldurulur.
        """
        komut = [yol, *args]
        try:
            sonuc = subprocess.run(
                komut,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(1.0, float(timeout)),
                creationflags=_no_window_flags(),
            )
        except subprocess.TimeoutExpired:
            self._failure_count += 1
            logger.debug("Sürüm sorgusu zaman aşımı: %s", yol)
            return "", "sürüm sorgusu zaman aşımı"
        except Exception as exc:
            self._failure_count += 1
            logger.debug("Sürüm sorgusu başarısız (%s): %s", yol, exc)
            return "", f"sürüm okunamadı: {exc}"

        metin = (sonuc.stdout or "").strip() or (sonuc.stderr or "").strip()
        for satir in metin.splitlines():
            temiz = satir.strip()
            if temiz:
                return temiz[:80], ""
        return "", "sürüm çıktısı boş"

    # ─── Toplu Yoklama ──────────────────────────────────────────────────

    def probe_all(self, force: bool = False) -> Dict[str, ToolInfo]:
        """
        Tüm araçları ve özellikleri yoklar (önbellekli).

        Args:
            force: True ise önbellek yok sayılır ve yeniden yoklanır.

        Returns:
            ``{ad: ToolInfo}``
        """
        if self._probed_all and not force:
            return dict(self._cache)

        baslangic = time.perf_counter()
        sonuc: Dict[str, ToolInfo] = {}

        # Windows'ta `which` ile bulunamayanları tek PowerShell çağrısında çöz
        self._batch_paths = {}
        if sys.platform == "win32":
            try:
                eksikler = [ad for ad in TOOL_SPECS if not _NAME_RE.match(ad) or not shutil.which(ad)]
                if eksikler:
                    self._batch_paths = self._resolve_paths_batch(eksikler)
            except Exception as exc:
                logger.debug("Toplu yol çözümlemesi başarısız: %s", exc)

        for ad in TOOL_SPECS:
            try:
                sonuc[ad] = self.probe_tool(ad)
            except Exception as exc:  # tek araç tüm yoklamayı düşürmesin
                self._failure_count += 1
                logger.debug("Araç yoklanamadı (%s): %s", ad, exc)
                sonuc[ad] = ToolInfo(ad, False, "", "", TOOL_SPECS[ad].get("category", "general"), str(exc))

        for ad in FEATURE_SPECS:
            try:
                sonuc[ad] = self.probe_feature(ad)
            except Exception as exc:
                self._failure_count += 1
                logger.debug("Özellik yoklanamadı (%s): %s", ad, exc)
                sonuc[ad] = ToolInfo(ad, False, "", "", FEATURE_SPECS[ad]["category"], str(exc))

        self._cache = sonuc
        self._probed_all = True
        self._last_probe_ts = time.time()
        sure_ms = (time.perf_counter() - baslangic) * 1000

        mevcut = [ad for ad, bilgi in sonuc.items() if bilgi.available]
        logger.info(
            "Yetenek yoklaması tamamlandı — %d/%d mevcut (%.0f ms)",
            len(mevcut),
            len(sonuc),
            sure_ms,
        )

        if self._event_bus:
            try:
                self._event_bus.emit(
                    "os.capabilities_probed",
                    {
                        "available": mevcut,
                        "missing": [ad for ad, b in sonuc.items() if not b.available],
                        "duration_ms": sure_ms,
                    },
                    source=self.name,
                )
            except Exception as exc:
                logger.debug("Yetenek olayı yayınlanamadı: %s", exc)

        return dict(sonuc)

    # ─── Özellik Yoklama ────────────────────────────────────────────────

    def probe_feature(self, name: str) -> ToolInfo:
        """
        Tek bir özellik/kütüphane probu çalıştırır.

        Args:
            name: Özellik adı (ör: ``cuda``, ``sqlite_fts5``).

        Returns:
            `ToolInfo`
        """
        self._probe_count += 1
        spec = FEATURE_SPECS.get(name, {"category": "feature", "note": ""})
        kategori = spec["category"]

        yoklayici = {
            "cuda": self._feature_cuda,
            "torch": self._feature_torch,
            "psutil": self._feature_psutil,
            "soundfile": self._feature_soundfile,
            "PIL": self._feature_pil,
            "tiktoken": self._feature_tiktoken,
            "transformers": self._feature_transformers,
            "sqlite_fts5": self._feature_sqlite_fts5,
            "long_paths_support": self._feature_long_paths,
        }.get(name)

        if yoklayici is None:
            return ToolInfo(name, False, "", "", kategori, "bilinmeyen özellik")

        bilgi = yoklayici()
        self._cache[name] = bilgi
        return bilgi

    def _feature_cuda(self) -> ToolInfo:
        """CUDA erişilebilirliği (torch üzerinden)."""
        try:
            import torch
        except Exception:
            return ToolInfo("cuda", False, "", "", "feature", "torch yok")
        try:
            if torch.cuda.is_available():
                ad = torch.cuda.get_device_name(0)
                surum = str(getattr(torch.version, "cuda", "") or "")
                return ToolInfo("cuda", True, "", surum, "feature", f"GPU: {ad}")
            return ToolInfo("cuda", False, "", "", "feature", "CUDA aygıtı yok")
        except Exception as exc:
            return ToolInfo("cuda", False, "", "", "feature", f"CUDA sorgusu başarısız: {exc}")

    def _feature_torch(self) -> ToolInfo:
        try:
            import torch

            return ToolInfo("torch", True, "", str(getattr(torch, "__version__", "")), "library", "")
        except Exception as exc:
            return ToolInfo("torch", False, "", "", "library", f"yok ({exc.__class__.__name__})")

    def _feature_psutil(self) -> ToolInfo:
        try:
            import psutil

            return ToolInfo("psutil", True, "", str(psutil.__version__), "library", "")
        except Exception as exc:
            return ToolInfo("psutil", False, "", "", "library", f"yok ({exc.__class__.__name__})")

    def _feature_soundfile(self) -> ToolInfo:
        try:
            import soundfile

            return ToolInfo("soundfile", True, "", str(getattr(soundfile, "__version__", "")), "library", "")
        except Exception as exc:
            return ToolInfo("soundfile", False, "", "", "library", f"yok ({exc.__class__.__name__})")

    def _feature_pil(self) -> ToolInfo:
        try:
            import PIL

            return ToolInfo("PIL", True, "", str(getattr(PIL, "__version__", "")), "library", "")
        except Exception as exc:
            return ToolInfo("PIL", False, "", "", "library", f"yok ({exc.__class__.__name__})")

    def _feature_tiktoken(self) -> ToolInfo:
        try:
            import tiktoken

            return ToolInfo("tiktoken", True, "", str(getattr(tiktoken, "__version__", "")), "library", "")
        except Exception as exc:
            return ToolInfo("tiktoken", False, "", "", "library", f"yok ({exc.__class__.__name__})")

    def _feature_transformers(self) -> ToolInfo:
        try:
            import transformers

            return ToolInfo(
                "transformers", True, "", str(getattr(transformers, "__version__", "")), "library", ""
            )
        except Exception as exc:
            return ToolInfo("transformers", False, "", "", "library", f"yok ({exc.__class__.__name__})")

    def _feature_sqlite_fts5(self) -> ToolInfo:
        """SQLite FTS5 sanal tablo desteğini gerçek bir denemeyle doğrular."""
        try:
            import sqlite3
        except Exception as exc:
            return ToolInfo("sqlite_fts5", False, "", "", "feature", f"sqlite3 yok: {exc}")

        baglanti = None
        try:
            baglanti = sqlite3.connect(":memory:")
            baglanti.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            baglanti.execute("INSERT INTO t(x) VALUES ('merhaba')")
            satir = baglanti.execute("SELECT count(*) FROM t").fetchone()
            if not satir or int(satir[0]) != 1:
                return ToolInfo("sqlite_fts5", False, "", sqlite3.sqlite_version, "feature", "FTS5 sorgusu beklenmedik sonuç")
            return ToolInfo("sqlite_fts5", True, "", str(sqlite3.sqlite_version), "feature", "")
        except Exception as exc:
            return ToolInfo("sqlite_fts5", False, "", "", "feature", f"FTS5 yok: {exc}")
        finally:
            if baglanti is not None:
                try:
                    baglanti.close()
                except Exception:
                    pass

    def _feature_long_paths(self) -> ToolInfo:
        """Windows `LongPathsEnabled` kayıt defteri değerini okur (korumalı)."""
        if sys.platform != "win32":
            return ToolInfo("long_paths_support", True, "", "", "feature", "Windows dışı — sınır yok")
        try:
            import winreg

            yol = r"SYSTEM\CurrentControlSet\Control\FileSystem"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, yol, 0, winreg.KEY_READ) as anahtar:
                deger, _ = winreg.QueryValueEx(anahtar, "LongPathsEnabled")
            aktif = bool(int(deger))
            return ToolInfo(
                "long_paths_support",
                aktif,
                "",
                f"LongPathsEnabled={int(deger)}",
                "feature",
                "" if aktif else "uzun yollar kapalı (260 karakter sınırı)",
            )
        except Exception as exc:
            return ToolInfo("long_paths_support", False, "", "", "feature", f"kayıt defteri okunamadı: {exc}")

    # ─── Sorgular ───────────────────────────────────────────────────────

    def has(self, name: str) -> bool:
        """
        Araç/özellik kullanılabilir mi?

        Bilinen araçlar önbellekten (gerekirse yoklanarak), bilinmeyen adlar
        canlı yoklanarak yanıtlanır.
        """
        bilinen = self._cache.get(name)
        if bilinen is None:
            bilinen = self.probe_tool(name) if name not in FEATURE_SPECS else self.probe_feature(name)
        return bool(bilinen.available)

    def get(self, name: str) -> Optional[ToolInfo]:
        """Önbellekten araç bilgisi döndürür (yoksa yoklar)."""
        if name not in self._cache:
            self.probe_all()
        return self._cache.get(name)

    def available_tools(self, category: Optional[str] = None) -> List[str]:
        """
        Kullanılabilir araç adlarını döndürür.

        Args:
            category: Yalnızca bu kategoriye ait araçlar (None = tümü).
        """
        araclar = self.probe_all()
        return sorted(
            ad
            for ad, bilgi in araclar.items()
            if bilgi.available and (category is None or bilgi.category == category)
        )

    def missing_tools(self, category: Optional[str] = None) -> List[str]:
        """Kullanılamayan araç adlarını döndürür."""
        araclar = self.probe_all()
        return sorted(
            ad
            for ad, bilgi in araclar.items()
            if not bilgi.available and (category is None or bilgi.category == category)
        )

    def categories(self) -> Dict[str, List[str]]:
        """Kategori → mevcut araç adları eşlemesi."""
        toplu: Dict[str, List[str]] = {}
        for ad, bilgi in self.probe_all().items():
            if bilgi.available:
                toplu.setdefault(bilgi.category, []).append(ad)
        for liste in toplu.values():
            liste.sort()
        return toplu

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir yoklama metrikleri."""
        toplam = len(TOOL_SPECS) + len(FEATURE_SPECS)
        mevcut = sum(1 for bilgi in self._cache.values() if bilgi.available)
        return {
            "name": self.name,
            "initialized": self._initialized,
            "cached": len(self._cache),
            "known_total": toplam,
            "available": mevcut,
            "missing": len(self._cache) - mevcut,
            "probe_count": self._probe_count,
            "failure_count": self._failure_count,
            "last_probe_ts": self._last_probe_ts,
            "categories": {k: len(v) for k, v in self.categories().items()},
            "tools": {ad: bilgi.to_dict() for ad, bilgi in sorted(self._cache.items())},
        }


__all__ = ["CapabilityProbe", "ToolInfo", "TOOL_SPECS", "FEATURE_SPECS"]
