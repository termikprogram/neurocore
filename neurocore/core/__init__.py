"""
NeuroCore Çekirdek Modülü (L0)
===============================
Konfigürasyon, olay yönetimi ve ana motor.

Alt modüller:
    config     — Donanım profillerine göre otomatik ayarlanan konfigürasyon
    event_bus  — Katmanlar arası publish/subscribe iletişimi
    engine     — Sekiz katmanı orkestre eden ana motor
"""

from neurocore.core.config import (
    EvolutionConfig,
    HardwareConfig,
    HardwareProfile,
    KnowledgeConfig,
    LanguageConfig,
    MultimodalConfig,
    NeuralConfig,
    PlannerConfig,
    PROFILE_DEFAULTS,
    SystemConfig,
)
from neurocore.core.event_bus import Event, EventBus, EventPriority, Subscription
from neurocore.core.engine import (
    EngineState,
    HealthReport,
    NeuroCoreEngine,
    QueryResult,
)

__all__ = [
    # Motor
    "NeuroCoreEngine",
    "EngineState",
    "QueryResult",
    "HealthReport",
    # Konfigürasyon
    "SystemConfig",
    "HardwareProfile",
    "HardwareConfig",
    "KnowledgeConfig",
    "NeuralConfig",
    "MultimodalConfig",
    "LanguageConfig",
    "PlannerConfig",
    "EvolutionConfig",
    "PROFILE_DEFAULTS",
    # Olaylar
    "EventBus",
    "Event",
    "EventPriority",
    "Subscription",
]
