"""
NeuroCore Evrim Katmanı (L8)
==============================
Kendi kendini büyüten sistemin beyni: deneyim tamponu, öz-değerlendirme,
müfredat üretimi, bilgi sentezi, merak güdümlü keşif, büyüme denetimi ve
bütünlük denetimi.

    from neurocore.evolution import EvolutionLayer, EvolutionHooks

    layer = EvolutionLayer()
    layer.initialize(config, event_bus, hooks=EvolutionHooks(list_notes=...))
    layer.observe("GPU neden ısınır?", "…", tags=["gpu"])
    report = layer.cycle()

Bu katman diğer NeuroCore katmanlarını **import etmez**; dış dünyaya yalnızca
`EvolutionHooks` üzerinden enjekte edilen geri çağrılarla erişir.
"""

from neurocore.evolution.curiosity import CuriosityEngine, CuriosityItem
from neurocore.evolution.curriculum import CurriculumBuilder, CurriculumItem
from neurocore.evolution.evaluator import QualityReport, SelfEvaluator
from neurocore.evolution.experience import (
    Experience,
    ExperienceBuffer,
    FeedbackSignal,
    hash_embed,
)
from neurocore.evolution.growth import (
    GrowthAction,
    GrowthController,
    GrowthDecision,
    GrowthState,
)
from neurocore.evolution.integrity import IntegrityGuard, IntegrityIssue
from neurocore.evolution.layer import EvolutionHooks, EvolutionLayer, EvolutionReport
from neurocore.evolution.synthesizer import KnowledgeSynthesizer, SynthesisResult

__all__ = [
    # Orkestrasyon
    "EvolutionLayer",
    "EvolutionHooks",
    "EvolutionReport",
    # Deneyim
    "Experience",
    "ExperienceBuffer",
    "FeedbackSignal",
    "hash_embed",
    # Değerlendirme
    "SelfEvaluator",
    "QualityReport",
    # Müfredat
    "CurriculumBuilder",
    "CurriculumItem",
    # Sentez
    "KnowledgeSynthesizer",
    "SynthesisResult",
    # Merak
    "CuriosityEngine",
    "CuriosityItem",
    # Büyüme
    "GrowthController",
    "GrowthAction",
    "GrowthDecision",
    "GrowthState",
    # Bütünlük
    "IntegrityGuard",
    "IntegrityIssue",
]
