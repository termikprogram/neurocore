"""
NeuroCore Planlayıcı Katmanı (L7)
==================================
Görev ağacı, kural tabanlı ayrıştırma, akıl yürütme zinciri, checkpoint ve
yeniden planlama. Diğer katmanlara **doğrudan bağımlı değildir**: yetenekler
(capability callables) dışarıdan enjekte edilir.

Kullanım:
    from neurocore.planner import PlannerLayer

    planner = PlannerLayer()
    planner.initialize(config, event_bus)
    planner.register_capability("retrieve", vault_search)
    planner.register_capability("synthesize", neural_answer)

    print(planner.run("GPU sıcaklığım neden yüksek?")["answer"])
"""

from neurocore.planner.task import (
    PlanError,
    ReasoningStep,
    ReasoningTrace,
    Task,
    TaskGraph,
    TaskKind,
    TaskState,
)
from neurocore.planner.decomposer import Decomposer
from neurocore.planner.executor import (
    CapabilityError,
    CapabilityRegistry,
    ExecutionAborted,
    ExecutionReport,
    Executor,
)
from neurocore.planner.checkpoint import Checkpoint, CheckpointManager
from neurocore.planner.layer import PlannerLayer

__all__ = [
    # Katman
    "PlannerLayer",
    # Görev modeli
    "Task",
    "TaskGraph",
    "TaskState",
    "TaskKind",
    "ReasoningStep",
    "ReasoningTrace",
    # Alt sistemler
    "Decomposer",
    "Executor",
    "ExecutionReport",
    "CapabilityRegistry",
    "CheckpointManager",
    "Checkpoint",
    # Hatalar
    "PlanError",
    "CapabilityError",
    "ExecutionAborted",
]
