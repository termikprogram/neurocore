"""
NeuroCore Masaüstü Uygulaması
==============================
Ajanı kullanıcıya açan masaüstü kabuğu.

Bileşenler::

    state.py    — Paylaşılan durum, onay broker'ı, olay akışı
    server.py   — FastAPI HTTP + WebSocket uç noktaları
    launcher.py — Katman kurulumu, sunucu başlatma, Edge --app penceresi
    web/        — Arayüz (HTML + CSS + JS, çerçevesiz)

Neden bu katman ayrı?
    `neurocore.agent` **ne yapılacağını** bilir (araçlar, döngü, politika).
    `neurocore.app` ise bunu **kullanıcıya nasıl sunacağını** bilir (arayüz,
    onay akışı, oturumlar, ayarlar). Ajan katmanı başsız (headless) da
    çalışabilir; uygulama katmanı ise insan etkileşimi içindir.

Çalıştırma::

    python -m neurocore.app
    python -m neurocore.app --workspace E:/PROJELER/oyunum --autonomy trusted
"""

from neurocore.app.launcher import DEFAULT_PORT, DesktopApp, build_parser, main
from neurocore.app.state import (
    AppState,
    ApprovalBroker,
    EventStream,
    PendingApproval,
)

__all__ = [
    "DesktopApp",
    "main",
    "build_parser",
    "DEFAULT_PORT",
    "AppState",
    "ApprovalBroker",
    "EventStream",
    "PendingApproval",
]
