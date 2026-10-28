"""
NeuroCore Görsel Doğrulama (L4 — Desktop Vision)
=================================================
Ekran görüntüsü alma, pencere/UI analizi ve "arayüz gerçekten doğru mu?"
sorusunu yanıtlayan katman.

Neden gerekli?
    Bir ajan kod yazıp çalıştırdığında, işin **gerçekten** olduğunu yalnızca
    çıktı metnine bakarak doğrulayamaz. Godot'da bir sahne yanlış konumda
    olabilir, Blender render'ı boş çıkabilir, kendi web arayüzümüz bozuk
    görünebilir. Bu modül ekranı görüntüler ve çok-modlu katmanla yorumlar.

Yetenekler:
    * **Ekran görüntüsü** — tam ekran, tek pencere, bölge (ctypes + GDI, harici
      bağımlılık yok; Pillow yalnızca kodlama için kullanılır)
    * **Pencere listesi** — başlık, sınıf, konum, boyut, görünürlük, PID
    * **Piksel analizi** — boş/tek renk ekran tespiti, parlaklık, kontrast,
      kenar yoğunluğu, baskın renkler (bir görüntünün "boş" olup olmadığını
      anlamak için)
    * **UI metni çıkarma** — Windows UI Automation ile denetim ağacından metin
      toplama (buton etiketleri, hata mesajları) — isteğe bağlı, `uiautomation`
      veya `pywinauto` yoksa atlanır
    * **OCR** — `tesseract` kuruluysa metin okuma (isteğe bağlı)
    * **Karşılaştırma** — iki ekran görüntüsü arasındaki farkı ölçme (regresyon
      kontrolü: "arayüz değişti mi?")
    * **Çok-modlu köprü** — görüntüyü `MultimodalLayer` üzerinden vektöre çevirip
      betimleme üretme

Tasarım kararı: Hiçbir ağır bağımlılık zorunlu değildir. `ctypes` + `Pillow`
ile ekran görüntüsü ve piksel analizi her koşulda çalışır; OCR ve UI Automation
varsa ek yetenek olarak devreye girer.

Bu modül **kendi başına** çalışabilir ve `neurocore.multimodal` dışında hiçbir
NeuroCore katmanını import etmez (çok-modlu köprü enjeksiyonla bağlanır).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("vision.desktop")

try:  # pragma: no cover
    from PIL import Image, ImageChops, ImageFilter, ImageGrab, ImageStat

    HAS_PIL = True
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    HAS_PIL = False

#: Windows API sabitleri (ctypes ile GDI ekran yakalama için)
_SRCCOPY = 0x00CC0020
_DIB_RGB_COLORS = 0
_BI_RGB = 0


class VisionError(Exception):
    """Görsel doğrulama hatası."""


# ─── Veri Yapıları ──────────────────────────────────────────────────────────

@dataclass
class WindowInfo:
    """Tek bir pencerenin bilgisi."""
    handle: int
    title: str
    class_name: str = ""
    pid: int = 0
    process: str = ""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    visible: bool = True
    minimized: bool = False
    foreground: bool = False

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """(sol, üst, sağ, alt) biçiminde sınır kutusu."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def area(self) -> int:
        """Piksel alanı."""
        return max(0, self.width) * max(0, self.height)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "handle": self.handle,
            "title": self.title,
            "class_name": self.class_name,
            "pid": self.pid,
            "process": self.process,
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "visible": self.visible, "minimized": self.minimized,
            "foreground": self.foreground,
        }

    def matches(self, needle: str) -> bool:
        """Başlık, sınıf veya süreç adında arama (büyük/küçük harf duyarsız)."""
        text = f"{self.title} {self.class_name} {self.process}".lower()
        return needle.lower() in text


@dataclass
class ScreenAnalysis:
    """
    Bir ekran görüntüsünün nicel analizi.

    "Boş ekran" tespiti ajan için kritiktir: bir uygulama çökerse veya sahne
    yüklenmezse ekran tek renk olur ve bu, metin çıktısında görünmez.
    """
    width: int = 0
    height: int = 0
    mean_brightness: float = 0.0        # 0..255
    contrast: float = 0.0               # standart sapma
    unique_colors: int = 0              # yaklaşık benzersiz renk sayısı
    edge_density: float = 0.0           # 0..1 (kenar pikselleri oranı)
    is_blank: bool = False              # tek renk / neredeyse tek renk
    is_dark: bool = False               # ortalama parlaklık < 30
    is_uniform: bool = False            # düşük kontrast
    dominant_colors: List[str] = field(default_factory=list)   # #RRGGBB
    window_title: str = ""
    captured_at: float = 0.0
    path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "width": self.width, "height": self.height,
            "mean_brightness": round(self.mean_brightness, 1),
            "contrast": round(self.contrast, 1),
            "unique_colors": self.unique_colors,
            "edge_density": round(self.edge_density, 4),
            "is_blank": self.is_blank,
            "is_dark": self.is_dark,
            "is_uniform": self.is_uniform,
            "dominant_colors": list(self.dominant_colors),
            "window_title": self.window_title,
            "captured_at": self.captured_at,
            "path": self.path,
        }

    def has_content(self) -> bool:
        """
        Ekranda **kullanılabilir içerik** var mı?

        Yalnızca parlaklığa bakmak yanıltıcıdır: koyu temalı uygulamalar
        (editörler, terminaller, bu projenin kendi arayüzü) düşük ortalama
        parlaklığa sahiptir ama tamamen doludur. Doğru ölçüt, yapısal
        zenginliktir: kenar yoğunluğu ve kontrast.
        """
        return self.edge_density >= 0.005 or self.contrast >= 20.0

    def verdict(self) -> str:
        """
        İnsan okur yorum: ekran sağlıklı mı?

        Sıralama önemlidir: önce "boş mu" (kesin hata), sonra "içerik var mı"
        (asıl soru), en son "karanlık mı" (yalnızca içerik de yoksa sorun).
        """
        if self.is_blank:
            return ("⚠ EKRAN BOŞ: tek renk tespit edildi — uygulama çökmüş, "
                    "pencere görünmüyor veya sahne yüklenmemiş olabilir")
        if not self.has_content:
            return ("⚠ EKRAN TEKDÜZE: kontrast ve kenar yoğunluğu çok düşük — "
                    "içerik yüklenmemiş olabilir")
        if self.is_dark:
            return (f"✓ EKRAN SAĞLIKLI (koyu tema): içerik var — "
                    f"parlaklık {self.mean_brightness:.0f}, "
                    f"kenar yoğunluğu {self.edge_density:.3f}")
        return "✓ EKRAN SAĞLIKLI: içerik var, kontrast yeterli"


@dataclass
class DiffResult:
    """İki ekran görüntüsü arasındaki fark."""
    changed_ratio: float = 0.0          # 0..1 değişen piksel oranı
    mean_delta: float = 0.0             # ortalama piksel farkı
    max_delta: float = 0.0
    identical: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None   # değişen bölge

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "changed_ratio": round(self.changed_ratio, 5),
            "mean_delta": round(self.mean_delta, 2),
            "max_delta": round(self.max_delta, 2),
            "identical": self.identical,
            "bbox": list(self.bbox) if self.bbox else None,
        }


# ─── Pencere Yönetimi (Win32) ───────────────────────────────────────────────

class WindowEnumerator:
    """
    Win32 API ile pencere listesi ve kontrolü.

    Windows dışında çalışmaz; bu durumda tüm metotlar boş liste döndürür ve
    `available` False olur.
    """

    def __init__(self) -> None:
        self._user32 = None
        self._kernel32 = None
        if os.name == "nt":
            try:
                self._user32 = ctypes.windll.user32
                self._kernel32 = ctypes.windll.kernel32
                self._user32.SetProcessDPIAware()
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Win32 API yüklenemedi: {exc}")
                self._user32 = None

    @property
    def available(self) -> bool:
        """Win32 API kullanılabilir mi?"""
        return self._user32 is not None

    def list_all(self, only_visible: bool = True, min_size: int = 0,
                 limit: int = 200) -> List[WindowInfo]:
        """
        Tüm üst düzey pencereleri listeler.

        Args:
            only_visible: Yalnızca görünür pencereler
            min_size: Minimum genişlik/yükseklik (gürültüyü eler)
            limit: Maksimum pencere sayısı

        Returns:
            Pencere listesi (öne gelen pencereler önce)
        """
        if not self.available:
            return []

        windows: List[WindowInfo] = []
        foreground = self._user32.GetForegroundWindow()
        dpi_scale = self._dpi_scale()

        # CALLBACK tipi ve referans tutma (GC'ye karşı)
        enum_proc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, wintypes.HWND, wintypes.LPARAM
        )

        def callback(hwnd: int, _lparam: int) -> bool:
            try:
                info = self._describe(hwnd, foreground, dpi_scale)
                if info is None:
                    return True
                if only_visible and (not info.visible or info.minimized):
                    return True
                if min_size and (info.width < min_size or info.height < min_size):
                    return True
                windows.append(info)
            except Exception:
                pass
            return len(windows) < limit

        try:
            self._user32.EnumWindows(enum_proc(callback), 0)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Pencere numaralandırma hatası: {exc}")
        windows.sort(key=lambda item: (not item.foreground, -item.area))
        return windows[:limit]

    def _dpi_scale(self) -> float:
        """
        Ekran ölçekleme oranını döndürür.

        `SetProcessDPIAware` sonrası `GetWindowRect` fiziksel piksel verir;
        DPI ölçekli sistemlerde yine de doğrulama için kullanılır.
        """
        try:
            hdc = self._user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            self._user32.ReleaseDC(0, hdc)
            return max(1.0, float(dpi) / 96.0)
        except Exception:
            return 1.0

    def _describe(self, hwnd: int, foreground: int, dpi_scale: float) -> Optional[WindowInfo]:
        """Tek pencereyi inceler; üst düzey değilse None."""
        user32 = self._user32
        if not user32.IsWindow(hwnd):
            return None
        # Yalnızca üst düzey pencereler (sahibi yok)
        if user32.GetWindow(hwnd, 4) != 0:  # GW_OWNER
            return None

        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value or ""

        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buffer, 256)
        class_name = class_buffer.value or ""

        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        width = int((rect.right - rect.left) / dpi_scale)
        height = int((rect.bottom - rect.top) / dpi_scale)

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process_name = ""
        try:
            import psutil
            process_name = psutil.Process(pid.value).name()
        except Exception:
            pass

        placement = None
        try:
            placement = wintypes.WINDOWPLACEMENT()
            placement.length = ctypes.sizeof(wintypes.WINDOWPLACEMENT)
            user32.GetWindowPlacement(hwnd, ctypes.byref(placement))
        except Exception:
            placement = None

        return WindowInfo(
            handle=int(hwnd),
            title=title,
            class_name=class_name,
            pid=int(pid.value),
            process=process_name,
            x=int(rect.left / dpi_scale),
            y=int(rect.top / dpi_scale),
            width=width,
            height=height,
            visible=bool(user32.IsWindowVisible(hwnd)),
            minimized=bool(placement and placement.showCmd == 2),
            foreground=(hwnd == foreground),
        )

    def find(self, needle: str, only_visible: bool = True,
             min_size: int = 100) -> List[WindowInfo]:
        """
        Başlık/sınıf/süreç adına göre pencere arar.

        Args:
            needle: Aranacak metin (ör. `"godot"`, `"Blender"`, `"NeuroCore"`)
            only_visible: Yalnızca görünür pencereler
            min_size: Minimum boyut

        Returns:
            Eşleşen pencereler (en büyük önce)
        """
        return [window for window in self.list_all(only_visible=only_visible, min_size=min_size)
                if window.matches(needle)]

    def foreground(self) -> Optional[WindowInfo]:
        """Ön plandaki pencere."""
        if not self.available:
            return None
        hwnd = self._user32.GetForegroundWindow()
        return self._describe(hwnd, hwnd, self._dpi_scale())

    def activate(self, handle: int) -> bool:
        """
        Pencereyi öne getirir.

        Not: Windows, başka bir sürecin penceresini öne getirmeyi kısıtlar;
        bu yüzden önce simge durumundan çıkarıp sonra `SetForegroundWindow`
        denenir. Başarısız olursa False döner (kritik değil).
        """
        if not self.available:
            return False
        try:
            user32 = self._user32
            user32.ShowWindow(handle, 9)      # SW_RESTORE
            user32.SetForegroundWindow(handle)
            return True
        except Exception as exc:
            logger.debug(f"Pencere öne getirilemedi ({handle}): {exc}")
            return False

    def window_rect(self, handle: int) -> Optional[Tuple[int, int, int, int]]:
        """Pencerenin ekran koordinatlarındaki sınır kutusu."""
        if not self.available:
            return None
        rect = wintypes.RECT()
        if not self._user32.GetWindowRect(handle, ctypes.byref(rect)):
            return None
        return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


# ─── Ekran Yakalama ─────────────────────────────────────────────────────────

class ScreenCapture:
    """
    Ekran görüntüsü alıcı.

    Öncelik sırası:
        1. `PIL.ImageGrab` (en hızlı ve güvenilir; Pillow kurulu)
        2. `ctypes` + GDI `BitBlt` (Pillow yoksa saf Win32)
        3. `nircmd`/PowerShell .NET yedeği (son çare)

    Yakalama **bellekte** tutulur; diske yazmak isteğe bağlıdır (`save=True`).
    """

    def __init__(self) -> None:
        self._windows = WindowEnumerator()

    @property
    def windows(self) -> WindowEnumerator:
        """Pencere numaralandırıcı."""
        return self._windows

    def virtual_screen_size(self) -> Tuple[int, int]:
        """
        Tüm ekranların kapsadığı sanal alanın boyutu.

        Çok monitörlü kurulumlarda negatif koordinatlar olabilir; bu yüzden
        yalnızca boyut döner.
        """
        if not self._windows.available:
            return (1920, 1080)
        user32 = self._windows._user32
        return (
            int(user32.GetSystemMetrics(78)),   # SM_CXVIRTUALSCREEN
            int(user32.GetSystemMetrics(79)),   # SM_CYVIRTUALSCREEN
        )

    def capture(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        window_title: Optional[str] = None,
        *,
        save: bool = False,
        path: Optional[str] = None,
        max_side: Optional[int] = None,
    ) -> Any:
        """
        Ekran görüntüsü alır.

        Args:
            region: (sol, üst, sağ, alt) — None ise tüm ekran
            window_title: Başlığa göre pencere yakala (region'ı geçersiz kılar)
            save: Diske yaz
            path: Kayıt yolu (None ve save=True ise geçici dosya)
            max_side: En uzun kenarı bu değere küçült (token tasarrufu)

        Returns:
            `PIL.Image` nesnesi

        Raises:
            VisionError: Yakalama başarısızsa
        """
        image = None

        if window_title:
            matches = self._windows.find(window_title)
            if not matches:
                raise VisionError(f"Pencere bulunamadı: {window_title}")
            target = matches[0]
            self._windows.activate(target.handle)
            time.sleep(0.3)     # pencere öne gelsin
            region = self._windows.window_rect(target.handle) or target.rect

        if HAS_PIL and ImageGrab is not None:
            try:
                # `all_screens=True` çok monitörlü kurulumu doğru yakalar
                image = ImageGrab.grab(bbox=region, all_screens=True)
            except Exception as exc:
                logger.debug(f"ImageGrab başarısız, GDI yedeğine geçiliyor: {exc}")
                image = None

        if image is None:
            image = self._capture_gdi(region)

        if image is None:
            raise VisionError(
                "Ekran görüntüsü alınamadı — Pillow kurulu değil ve GDI yakalama "
                "başarısız oldu"
            )

        if max_side and max(image.size) > max_side:
            ratio = max_side / max(image.size)
            new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
            image = image.resize(new_size, Image.LANCZOS)

        if save:
            target_path = Path(path) if path else Path(tempfile.gettempdir()) / \
                f"neurocore_shot_{int(time.time() * 1000)}.png"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(target_path, format="PNG")
            image.neurocore_path = str(target_path)  # type: ignore[attr-defined]
        return image

    def _capture_gdi(self, region: Optional[Tuple[int, int, int, int]]) -> Any:
        """
        Win32 GDI `BitBlt` ile ekran yakalama (Pillow'suz yedek).

        Not: Bu yol `PIL.Image` nesnesi üretmek için Pillow'a ihtiyaç duyar;
        Pillow yoksa ham BMP baytları döndürülür ve üst katman bunu diske
        yazabilir.
        """
        if not self._windows.available:
            return None
        user32 = self._windows._user32
        gdi32 = ctypes.windll.gdi32

        if region is None:
            width, height = self.virtual_screen_size()
            left, top = 0, 0
            # Sanal ekranın sol üstü negatif olabilir
            left = int(user32.GetSystemMetrics(76))    # SM_XVIRTUALSCREEN
            top = int(user32.GetSystemMetrics(77))     # SM_YVIRTUALSCREEN
        else:
            left, top, right, bottom = region
            width, height = right - left, bottom - top

        if width <= 0 or height <= 0:
            return None

        hdc_screen = user32.GetDC(0)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        bitmap = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
        gdi32.SelectObject(hdc_mem, bitmap)

        try:
            gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_screen, left, top, _SRCCOPY)

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD),
                    ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD),
                ]

            header = BITMAPINFOHEADER()
            header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            header.biWidth = width
            header.biHeight = -height        # negatif = yukarıdan aşağı
            header.biPlanes = 1
            header.biBitCount = 32
            header.biCompression = _BI_RGB

            buffer_size = width * height * 4
            buffer = ctypes.create_string_buffer(buffer_size)
            gdi32.GetDIBits(hdc_mem, bitmap, 0, height, buffer, ctypes.byref(header),
                            _DIB_RGB_COLORS)

            if HAS_PIL:
                image = Image.frombuffer("RGBA", (width, height), buffer, "raw", "BGRA", 0, 1)
                return image.convert("RGB")
            return buffer.raw
        finally:
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(0, hdc_screen)

    def capture_window(self, title: str, **kwargs: Any) -> Any:
        """Başlığa göre pencere yakalar (kısa yol)."""
        return self.capture(window_title=title, **kwargs)

    def capture_region(self, x: int, y: int, width: int, height: int,
                       **kwargs: Any) -> Any:
        """Bölge yakalar (kısa yol)."""
        return self.capture(region=(x, y, x + width, y + height), **kwargs)


# ─── Ana Görsel Doğrulama Katmanı ───────────────────────────────────────────

class DesktopVision:
    """
    Görsel doğrulama katmanı.

    Kullanım:
        vision = DesktopVision()
        vision.initialize(config, event_bus)

        # Ekran görüntüsü + analiz
        shot = vision.screenshot(save=True)
        analysis = vision.analyze(shot)
        print(analysis.verdict())

        # Pencere arama
        windows = vision.find_windows("godot")

        # Pencere görüntüsü ve betimleme
        result = vision.describe_window("Godot")

        # Regresyon: iki görüntü farkı
        diff = vision.compare(image_a, image_b)
        if not diff.identical:
            print(f"Arayüz değişti: %{diff.changed_ratio * 100:.1f}")

        # UI metni (Windows UI Automation)
        text = vision.window_text("MyApp")
    """

    name = "vision.desktop"

    def __init__(self) -> None:
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._lock = threading.RLock()
        self.capture = ScreenCapture()
        self.windows = WindowEnumerator()
        self._multimodal: Any = None          # enjekte edilir (çok-modlu katman)
        self._embed_fn: Optional[Callable[[str], Sequence[float]]] = None
        self._shot_dir: Optional[Path] = None
        self._last_analysis: Optional[ScreenAnalysis] = None
        self._baseline: Dict[str, Any] = {}   # ad → PIL görüntü (regresyon için)

        self._counter = {
            "screenshots": 0, "analyses": 0, "windows_listed": 0,
            "descriptions": 0, "comparisons": 0, "text_extractions": 0,
            "errors": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None,
                   multimodal: Any = None) -> None:
        """
        Katmanı başlatır.

        Args:
            config: SystemConfig
            event_bus: EventBus
            multimodal: `MultimodalLayer` (görüntü betimleme için, isteğe bağlı)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._multimodal = multimodal

        base = getattr(config, "data_path", None)
        if not base:
            base_path = getattr(config, "base_path", None) or os.getcwd()
            base = str(Path(base_path) / "data")
        self._shot_dir = Path(base) / "screenshots"
        try:
            self._shot_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"Ekran görüntüsü dizini oluşturulamadı: {exc}")
            self._shot_dir = None

        self._initialized = True
        logger.info(
            f"Görsel doğrulama hazır — yakalama: "
            f"{'Pillow' if HAS_PIL else 'GDI'}, "
            f"pencere listesi: {'var' if self.windows.available else 'yok'}, "
            f"OCR: {'var' if self._tesseract_path() else 'yok'}"
        )

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır ve önbelleği temizler."""
        self._baseline.clear()
        self._initialized = False
        logger.info(
            f"Görsel doğrulama kapatıldı ({self._counter['screenshots']} görüntü, "
            f"{self._counter['analyses']} analiz)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Katman istatistikleri."""
        return {
            **self._counter,
            "initialized": self._initialized,
            "pillow": HAS_PIL,
            "window_api": self.windows.available,
            "ocr": bool(self._tesseract_path()),
            "ui_automation": self._ui_automation_available(),
            "shot_dir": str(self._shot_dir or ""),
            "baselines": len(self._baseline),
            "screen": {
                "virtual_width": self.capture.virtual_screen_size()[0],
                "virtual_height": self.capture.virtual_screen_size()[1],
            },
        }

    def attach_multimodal(self, multimodal: Any) -> None:
        """Çok-modlu katmanı bağlar (görüntü betimleme için)."""
        self._multimodal = multimodal

    # ─── Ekran Görüntüsü ────────────────────────────────────────────────

    def screenshot(
        self,
        *,
        window: Optional[str] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        save: bool = True,
        tag: str = "",
        max_side: int = 1280,
    ) -> Dict[str, Any]:
        """
        Ekran görüntüsü alır ve analiz eder.

        Args:
            window: Pencere başlığı (verilirse o pencere yakalanır)
            region: (sol, üst, sağ, alt); None ise tüm ekran
            save: Diske kaydet
            tag: Dosya adına eklenecek etiket (ör. `"godot_test"`)
            max_side: En uzun kenar sınırı (token/bellek tasarrufu)

        Returns:
            {"ok", "path", "width", "height", "analysis", "verdict", "error"}
        """
        result: Dict[str, Any] = {"ok": False, "path": "", "analysis": {}}
        try:
            path = None
            if save and self._shot_dir is not None:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                suffix = f"_{tag}" if tag else ""
                path = str(self._shot_dir / f"{stamp}{suffix}.png")

            image = self.capture.capture(
                region=region, window_title=window, save=save, path=path,
                max_side=max_side,
            )
            self._counter["screenshots"] += 1
            analysis = self.analyze(image)
            self._last_analysis = analysis

            result.update({
                "ok": True,
                "path": getattr(image, "neurocore_path", path or ""),
                "width": image.width if hasattr(image, "width") else 0,
                "height": image.height if hasattr(image, "height") else 0,
                "analysis": analysis.to_dict(),
                "verdict": analysis.verdict(),
            })
            self._emit("vision.screenshot", {
                "path": result["path"], "window": window or "",
                "is_blank": analysis.is_blank,
                "brightness": round(analysis.mean_brightness, 1),
            })
        except VisionError as exc:
            self._counter["errors"] += 1
            result["error"] = str(exc)
        except Exception as exc:
            self._counter["errors"] += 1
            result["error"] = f"beklenmeyen hata: {exc}"
            logger.error(f"Ekran görüntüsü hatası: {exc}", exc_info=True)
        return result

    # ─── Analiz ─────────────────────────────────────────────────────────

    def analyze(self, image: Any = None, path: Optional[str] = None) -> ScreenAnalysis:
        """
        Görüntüyü nicel olarak analiz eder.

        "Boş ekran" tespiti ajanın en çok ihtiyaç duyduğu sinyaldir: bir
        uygulama çöktüğünde veya sahne yüklenmediğinde ekran tek renk olur ve
        bu, metin çıktısında görünmez.

        Args:
            image: `PIL.Image` nesnesi
            path: Görüntü dosyası (image None ise)

        Returns:
            `ScreenAnalysis`
        """
        self._counter["analyses"] += 1
        analysis = ScreenAnalysis(captured_at=time.time())

        if image is None and path:
            if not HAS_PIL:
                raise VisionError("Görüntü analizi için Pillow gerekir")
            image = Image.open(path)
            image.load()
        if image is None:
            raise VisionError("Analiz için görüntü gerekli")

        try:
            analysis.width, analysis.height = image.size
        except Exception:
            return analysis

        if not HAS_PIL:
            return analysis

        try:
            # ── Renk modu ve küçültme (hız için) ──
            working = image.convert("RGB")
            small = working.copy()
            small.thumbnail((256, 256), Image.LANCZOS)

            # ── Parlaklık ve kontrast ──
            gray = small.convert("L")
            stat = ImageStat.Stat(gray)
            analysis.mean_brightness = float(stat.mean[0]) if stat.mean else 0.0
            analysis.contrast = float(stat.stddev[0]) if stat.stddev else 0.0

            # ── Benzersiz renk sayısı (yaklaşık, 5-bit kuantalama) ──
            quantized = small.quantize(colors=64, method=Image.MEDIANCUT) \
                if hasattr(Image, "MEDIANCUT") else small.convert("P", palette=Image.ADAPTIVE, colors=64)
            colors = quantized.getcolors(maxcolors=70000) or []
            analysis.unique_colors = len(colors)

            # ── Baskın renkler ──
            colors.sort(key=lambda item: -item[0])
            palette = quantized.getpalette() or []
            dominant: List[str] = []
            for _count, index in colors[:5]:
                offset = index * 3
                if offset + 2 < len(palette):
                    rgb = palette[offset:offset + 3]
                    dominant.append("#%02x%02x%02x" % tuple(rgb))
            analysis.dominant_colors = dominant

            # ── Kenar yoğunluğu (içerik var mı?) ──
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            analysis.edge_density = min(1.0, (edge_stat.mean[0] if edge_stat.mean else 0.0) / 255.0)

            # ── Kararlar ──
            # Tek renk ekran: kontrast ~0 ve çok az benzersiz renk
            analysis.is_blank = analysis.contrast < 3.0 and analysis.unique_colors <= 3
            analysis.is_dark = analysis.mean_brightness < 30.0
            analysis.is_uniform = analysis.contrast < 12.0 and analysis.edge_density < 0.02

            if path:
                analysis.path = str(path)
            elif getattr(image, "neurocore_path", None):
                analysis.path = str(getattr(image, "neurocore_path"))
        except Exception as exc:
            logger.debug(f"Analiz hatası: {exc}")

        return analysis

    def analyze_blank(self, image: Any = None) -> Tuple[bool, str]:
        """
        Ekranın boş olup olmadığını hızlıca söyler (karar için kısa yol).

        Returns:
            (boş_mu, gerekçe)
        """
        analysis = self.analyze(image)
        return analysis.is_blank, analysis.verdict()

    def compare(self, first: Any, second: Any,
                threshold: float = 0.02) -> DiffResult:
        """
        İki görüntüyü karşılaştırır (regresyon kontrolü).

        Ajan bir arayüzü değiştirdikten sonra "görünüm bozuldu mu?" sorusunu
        ancak bu yöntemle yanıtlayabilir.

        Args:
            first: Önceki görüntü (`PIL.Image` veya yol)
            second: Yeni görüntü
            threshold: Bu orandan az fark "aynı" sayılır

        Returns:
            `DiffResult`
        """
        self._counter["comparisons"] += 1
        result = DiffResult()
        if not HAS_PIL:
            return result

        try:
            image_a = self._as_image(first)
            image_b = self._as_image(second)
            if image_a.size != image_b.size:
                image_b = image_b.resize(image_a.size, Image.LANCZOS)

            gray_a = image_a.convert("L")
            gray_b = image_b.convert("L")
            diff = ImageChops.difference(gray_a, gray_b)

            histogram = diff.histogram()
            total_pixels = max(1, sum(histogram))
            changed = sum(histogram[16:])          # 16/255 üstü farkı say
            result.changed_ratio = changed / total_pixels

            stat = ImageStat.Stat(diff)
            result.mean_delta = float(stat.mean[0]) if stat.mean else 0.0
            result.max_delta = float(max(diff.getextrema() or (0, 0)))

            bbox = diff.getbbox()
            result.bbox = tuple(int(value) for value in bbox) if bbox else None
            result.identical = result.changed_ratio <= threshold
        except Exception as exc:
            logger.debug(f"Karşılaştırma hatası: {exc}")
        return result

    def set_baseline(self, name: str, image: Any = None, **capture_kwargs: Any) -> bool:
        """
        Bir ad altında referans görüntü saklar (regresyon testi için).

        Args:
            name: Referans adı (ör. `"main_ui"`)
            image: Görüntü (None ise yeni ekran görüntüsü alınır)
            **capture_kwargs: `screenshot()` parametreleri

        Returns:
            Başarılı ise True
        """
        try:
            if image is None:
                image = self.capture.capture(save=False, **capture_kwargs)
            self._baseline[name] = image
            logger.info(f"Referans görüntü kaydedildi: {name} ({len(self._baseline)} adet)")
            return True
        except Exception as exc:
            logger.warning(f"Referans kaydedilemedi ({name}): {exc}")
            return False

    def check_baseline(self, name: str, **capture_kwargs: Any) -> Dict[str, Any]:
        """
        Mevcut ekranı referansla karşılaştırır.

        Returns:
            {"ok", "name", "diff", "changed", "message"}
        """
        if name not in self._baseline:
            return {"ok": False, "error": f"referans yok: {name}"}
        try:
            current = self.capture.capture(save=False, **capture_kwargs)
            diff = self.compare(self._baseline[name], current)
            changed = not diff.identical
            message = (
                f"arayüz değişti: %{diff.changed_ratio * 100:.2f} piksel farklı"
                if changed else "arayüz aynı"
            )
            return {"ok": True, "name": name, "diff": diff.to_dict(),
                    "changed": changed, "message": message}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _as_image(self, source: Any) -> Any:
        """Kaynağı `PIL.Image`'a çevirir (yol, bayt veya nesne)."""
        if HAS_PIL and isinstance(source, Image.Image):
            return source
        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")
        if isinstance(source, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(source))).convert("RGB")
        raise VisionError(f"Görüntü kaynağı anlaşılamadı: {type(source).__name__}")

    # ─── Pencere İşlemleri ──────────────────────────────────────────────

    def list_windows(self, only_visible: bool = True, min_size: int = 100,
                     limit: int = 100, query: str = "") -> List[Dict[str, Any]]:
        """
        Pencereleri listeler.

        Args:
            only_visible: Yalnızca görünür pencereler
            min_size: Minimum boyut (gürültü filtresi)
            limit: Maksimum pencere
            query: Başlık/sınıf/süreç filtresi

        Returns:
            Pencere sözlükleri listesi
        """
        windows = self.windows.list_all(only_visible=only_visible,
                                       min_size=min_size, limit=limit)
        if query:
            windows = [window for window in windows if window.matches(query)]
        self._counter["windows_listed"] += len(windows)
        return [window.to_dict() for window in windows]

    def find_windows(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Pencere arar (kısa yol)."""
        return self.list_windows(query=query, **kwargs)

    def foreground_window(self) -> Dict[str, Any]:
        """Ön plandaki pencerenin bilgisi."""
        window = self.windows.foreground()
        return window.to_dict() if window else {}

    def window_text(self, query: str, max_chars: int = 4000) -> Dict[str, Any]:
        """
        Bir pencerenin UI denetim ağacından **metinleri** toplar.

        Windows UI Automation (UIA) kullanılır. `uiautomation` veya
        `pywinauto` kurulu değilse boş sonuç döner (hata değil, eksik yetenek).

        Bu, "buton gerçekten 'Kaydet' mi yazıyor?", "hata mesajı ne?" gibi
        soruları ekran görüntüsü olmadan yanıtlar.

        Args:
            query: Pencere başlığı filtresi
            max_chars: Toplanacak maksimum metin

        Returns:
            {"ok", "window", "texts": [...], "joined": str, "method": str}
        """
        result: Dict[str, Any] = {"ok": False, "window": query, "texts": [],
                                  "joined": "", "method": ""}
        matches = self.windows.find(query)
        if not matches:
            result["error"] = f"pencere bulunamadı: {query}"
            return result
        target = matches[0]
        result["handle"] = target.handle
        result["title"] = target.title

        # 1. pywinauto (varsa en zengin çıktı)
        try:
            from pywinauto import Desktop  # type: ignore

            window = Desktop(backend="uia").window(handle=target.handle)
            texts: List[str] = []
            for control in window.descendants():
                try:
                    name = control.window_text()
                    if name and name.strip():
                        texts.append(name.strip())
                except Exception:
                    continue
                if sum(len(item) for item in texts) > max_chars:
                    break
            result.update({"ok": True, "texts": texts[:300],
                           "joined": " | ".join(texts)[:max_chars],
                           "method": "pywinauto"})
            self._counter["text_extractions"] += 1
            return result
        except ImportError:
            pass
        except Exception as exc:
            logger.debug(f"pywinauto okuma hatası: {exc}")

        # 2. uiautomation (varsa)
        try:
            import uiautomation as auto  # type: ignore

            window = auto.ControlFromHandle(target.handle)
            texts = []
            if window is not None:
                for control in window.GetChildren():
                    try:
                        name = control.Name
                        if name and name.strip():
                            texts.append(name.strip())
                    except Exception:
                        continue
            result.update({"ok": bool(texts), "texts": texts[:300],
                           "joined": " | ".join(texts)[:max_chars],
                           "method": "uiautomation"})
            if texts:
                self._counter["text_extractions"] += 1
            return result
        except ImportError:
            pass
        except Exception as exc:
            logger.debug(f"uiautomation okuma hatası: {exc}")

        # 3. OCR yedeği (tesseract kuruluysa)
        ocr = self.ocr_window(target.title)
        if ocr.get("ok"):
            result.update({"ok": True, "texts": [ocr["text"]],
                           "joined": ocr["text"][:max_chars], "method": "ocr"})
            self._counter["text_extractions"] += 1
            return result

        result["error"] = (
            "UI metni okunamadı — pywinauto/uiautomation/tesseract kurulu değil"
        )
        return result

    def _ui_automation_available(self) -> bool:
        """UI Automation kütüphanesi var mı?"""
        for module in ("pywinauto", "uiautomation"):
            try:
                __import__(module)
                return True
            except ImportError:
                continue
        return False

    # ─── OCR ────────────────────────────────────────────────────────────

    def _tesseract_path(self) -> Optional[str]:
        """`tesseract` çalıştırılabilirinin yolu (yoksa None)."""
        found = shutil.which("tesseract")
        if found:
            return found
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if os.path.exists(candidate):
                return candidate
        return None

    def ocr(self, image: Any = None, path: Optional[str] = None,
            language: str = "tur+eng", timeout: float = 30.0) -> Dict[str, Any]:
        """
        Görüntüden metin okur (Tesseract ile, isteğe bağlı).

        Args:
            image: `PIL.Image` (None ise `path` veya yeni ekran görüntüsü)
            path: Görüntü dosyası
            language: Tesseract dil paketi (`tur+eng`)
            timeout: Saniye

        Returns:
            {"ok", "text", "language", "error"}
        """
        binary = self._tesseract_path()
        if not binary:
            return {"ok": False, "error": "tesseract kurulu değil", "text": ""}

        temp_path = path
        created = False
        try:
            if image is not None and HAS_PIL:
                temp_path = str(Path(tempfile.gettempdir()) /
                                f"nc_ocr_{int(time.time() * 1000)}.png")
                image.save(temp_path)
                created = True
            elif temp_path is None:
                shot = self.capture.capture(save=False)
                temp_path = str(Path(tempfile.gettempdir()) /
                                f"nc_ocr_{int(time.time() * 1000)}.png")
                shot.save(temp_path)
                created = True

            completed = subprocess.run(
                [binary, str(temp_path), "stdout", "-l", language, "--psm", "6"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            text = (completed.stdout or "").strip()
            self._counter["text_extractions"] += 1
            return {
                "ok": bool(text),
                "text": text,
                "language": language,
                "returncode": completed.returncode,
                "error": "" if text else (completed.stderr or "metin bulunamadı")[:300],
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"OCR zaman aşımı ({timeout}s)", "text": ""}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "text": ""}
        finally:
            if created and temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def ocr_window(self, query: str, language: str = "tur+eng") -> Dict[str, Any]:
        """Pencere görüntüsünü alıp OCR uygular."""
        try:
            image = self.capture.capture(window_title=query, save=False)
            return self.ocr(image=image, language=language)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "text": ""}

    # ─── Çok-Modlu Köprü ────────────────────────────────────────────────

    def describe(self, image: Any = None, path: Optional[str] = None,
                 question: str = "", window: Optional[str] = None) -> Dict[str, Any]:
        """
        Görüntüyü çok-modlu katmanla betimler.

        Bu metot, "ekranda ne var?" sorusunu **sayısal analizin ötesine**
        taşır: görüntü çok-modlu kodlayıcıdan geçirilir ve (varsa) OCR metni
        ile analiz sonuçları tek bir yapıda birleştirilir.

        Args:
            image: `PIL.Image`
            path: Görüntü dosyası
            question: Görüntüyle ilgili soru (betimleme bağlamı için)
            window: Pencere başlığı (görüntü yerine)

        Returns:
            {"ok", "modality", "vector_dim", "analysis", "verdict",
             "ocr_text", "window", "summary"}
        """
        result: Dict[str, Any] = {"ok": False, "analysis": {}, "ocr_text": ""}
        try:
            if window:
                image = self.capture.capture(window_title=window, save=True)
            elif image is None and path:
                image = self._as_image(path)
            elif image is None:
                image = self.capture.capture(save=True)

            analysis = self.analyze(image)
            result["analysis"] = analysis.to_dict()
            result["verdict"] = analysis.verdict()
            result["path"] = analysis.path or getattr(image, "neurocore_path", "")

            # Çok-modlu kodlama (varsa)
            if self._multimodal is not None:
                try:
                    chunk = self._multimodal.encode(image)
                    result["modality"] = getattr(chunk.modality, "value", str(chunk.modality))
                    result["vector_dim"] = len(chunk.vector)
                    result["confidence"] = getattr(chunk, "confidence", 0.0)
                except Exception as exc:
                    logger.debug(f"Çok-modlu kodlama başarısız: {exc}")

            # OCR (varsa)
            ocr = self.ocr(image=image)
            if ocr.get("ok"):
                result["ocr_text"] = ocr["text"][:2000]

            # İnsan okur özet
            parts = [analysis.verdict()]
            if result.get("ocr_text"):
                preview = result["ocr_text"][:400].replace("\n", " ")
                parts.append(f"Ekrandaki metin: {preview}")
            if question:
                parts.append(f"(soru: {question})")
            result["summary"] = " ".join(parts)
            result["ok"] = True
            self._counter["descriptions"] += 1
            self._emit("vision.described", {
                "path": result.get("path", ""), "is_blank": analysis.is_blank,
                "has_ocr": bool(result.get("ocr_text")),
            })
        except Exception as exc:
            self._counter["errors"] += 1
            result["error"] = str(exc)
            logger.debug(f"Betimleme hatası: {exc}")
        return result

    # ─── Yüksek Seviye Doğrulama ────────────────────────────────────────

    def verify_application(self, expected_title: str, *,
                           expect_content: bool = True,
                           timeout: float = 15.0,
                           poll_interval: float = 0.8) -> Dict[str, Any]:
        """
        Bir uygulamanın açıldığını ve **içerik gösterdiğini** doğrular.

        Ajanın en sık ihtiyaç duyduğu kontrol: "uygulama gerçekten açıldı mı?"
        Süreç başlamış olsa bile pencere gelmemiş veya boş olabilir.

        Args:
            expected_title: Aranacak pencere başlığı parçası
            expect_content: Ekranın boş olmaması beklensin mi?
            timeout: Bu süre boyunca beklenir
            poll_interval: Denemeler arası bekleme

        Returns:
            {"ok", "window", "analysis", "waited_seconds", "message"}
        """
        start = time.time()
        last: Dict[str, Any] = {}
        deadline = start + max(1.0, timeout)

        while time.time() < deadline:
            matches = self.windows.find(expected_title)
            if matches:
                target = matches[0]
                try:
                    image = self.capture.capture(window_title=expected_title, save=True)
                    analysis = self.analyze(image)
                    last = {
                        "window": target.to_dict(),
                        "analysis": analysis.to_dict(),
                        "verdict": analysis.verdict(),
                    }
                    if not expect_content:
                        return {"ok": True, "waited_seconds": round(time.time() - start, 1),
                                "message": f"pencere bulundu: {target.title}", **last}
                    if not analysis.is_blank:
                        return {"ok": True, "waited_seconds": round(time.time() - start, 1),
                                "message": f"uygulama çalışıyor ve içerik gösteriyor: {target.title}",
                                **last}
                    last["message"] = "pencere var ama ekran boş — içerik bekleniyor"
                except Exception as exc:
                    last = {"error": str(exc)}
            else:
                last = {"message": f"pencere bulunamadı: {expected_title}"}
            time.sleep(max(0.1, poll_interval))

        waited = round(time.time() - start, 1)
        return {
            "ok": False,
            "waited_seconds": waited,
            "message": f"{timeout:.0f}s içinde doğrulanamadı — "
                       f"{last.get('message', last.get('error', 'bilinmiyor'))}",
            **last,
        }

    def health(self) -> Dict[str, Any]:
        """Katman sağlık raporu."""
        issues: List[str] = []
        if not HAS_PIL:
            issues.append("Pillow kurulu değil — analiz ve kayıt sınırlı")
        if not self.windows.available:
            issues.append("Win32 pencere API'si kullanılamıyor (Windows dışı?)")
        if not self._ui_automation_available():
            issues.append("UI Automation yok (pywinauto/uiautomation) — UI metni OCR'a bağlı")
        if not self._tesseract_path():
            issues.append("tesseract yok — OCR devre dışı")
        if self._multimodal is None:
            issues.append("çok-modlu katman bağlı değil — görüntü betimleme sınırlı")
        return {
            "ok": self._initialized and HAS_PIL,
            "screen": f"{self.capture.virtual_screen_size()[0]}x{self.capture.virtual_screen_size()[1]}",
            "capabilities": {
                "screenshot": HAS_PIL or self.windows.available,
                "window_list": self.windows.available,
                "ui_text": self._ui_automation_available(),
                "ocr": bool(self._tesseract_path()),
                "multimodal": self._multimodal is not None,
            },
            "issues": issues,
            "stats": self.get_stats(),
        }

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __repr__(self) -> str:  # pragma: no cover
        size = self.capture.virtual_screen_size()
        return f"<DesktopVision screen={size[0]}x{size[1]} shots={self._counter['screenshots']}>"


__all__ = [
    "DesktopVision",
    "ScreenCapture",
    "WindowEnumerator",
    "WindowInfo",
    "ScreenAnalysis",
    "DiffResult",
    "VisionError",
    "HAS_PIL",
]
