"""
NeuroCore Görsel Doğrulama Katmanı
===================================
Ekran görüntüsü, pencere analizi, UI metni ve görüntü karşılaştırma.

Alt modüller:
    desktop — Ekran yakalama, pencere numaralandırma, piksel analizi,
              UI Automation metni, OCR ve çok-modlu betimleme köprüsü

Neden ayrı katman?
    Ajan kod yazıp çalıştırdığında işin **gerçekten** olduğunu yalnızca metin
    çıktısına bakarak doğrulayamaz. Bir oyun sahnesi yanlış konumda olabilir,
    bir render boş çıkabilir, kendi arayüzümüz bozuk görünebilir. Bu katman
    "göz" işlevini görür.

Bağımlılık politikası:
    Zorunlu bağımlılık yoktur. `ctypes` + Win32 ile ekran yakalama ve pencere
    listesi her koşulda çalışır; Pillow, OCR (tesseract) ve UI Automation
    (pywinauto/uiautomation) varsa ek yetenek olarak devreye girer.
"""

from neurocore.vision.desktop import (
    HAS_PIL,
    DesktopVision,
    DiffResult,
    ScreenAnalysis,
    ScreenCapture,
    VisionError,
    WindowEnumerator,
    WindowInfo,
)

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
