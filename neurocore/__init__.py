"""
NeuroCore AI — Kendi kendini büyütebilen, donanım-farkında, çok-modlu yapay zeka sistemi.

Bu paket, 8 katmanlı bir mimari üzerine inşa edilmiş adaptif bir AI sisteminin
tüm bileşenlerini içerir.

Katmanlar:
    1. Hardware    — Donanım izleme ve kaynak tahsisi
    2. Knowledge   — Obsidian-tarzı bilgi motoru (kasa, indeks, graf, arama)
    3. Neural      — Adaptif sinir ağı (MoE transformer + bellek bankası)
    4. Multimodal  — Çok-modlu işlem hattı (metin/görüntü/ses/tablo)
    5. OS          — İşletim sistemi entegrasyonu (sandbox denetimli)
    6. Language    — Dil algılama ve çeviri (yerel + API köprüsü)
    7. Planner     — Görev planlama ve akıl yürütme
    8. Evolution   — Evrim ve öz-büyüme motoru

Hızlı başlangıç::

    from neurocore import NeuroCoreEngine

    with NeuroCoreEngine() as engine:
        result = engine.query("GPU sıcaklığı neden yükselir?")
        print(result.render())

        engine.learn("Termal macun iki yılda bir yenilenmelidir.",
                     title="Termal Macun", tags=["donanım"])

        engine.start()                     # arka plan bakım döngüleri
        engine.evolve(cycles=1)            # bir evrim turu
        print(engine.health_report().render())
"""

__version__ = "0.2.5"
__author__ = "NeuroCore Team"
__description__ = "Self-growing, hardware-aware, multimodal AI system"

from neurocore.core.config import HardwareProfile, SystemConfig
from neurocore.core.event_bus import Event, EventBus, EventPriority
from neurocore.core.engine import EngineState, HealthReport, NeuroCoreEngine, QueryResult

__all__ = [
    "NeuroCoreEngine",
    "EngineState",
    "QueryResult",
    "HealthReport",
    "SystemConfig",
    "HardwareProfile",
    "EventBus",
    "Event",
    "EventPriority",
    "__version__",
]
