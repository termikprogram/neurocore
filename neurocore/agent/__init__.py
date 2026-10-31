"""
NeuroCore Ajan Katmanı
=======================
Kodlama, kabuk, uygulama kontrolü ve görsel doğrulama yeteneklerini tek bir
araç protokolü altında birleştirir ve bir dil modeliyle yönetir.

Katman haritası::

    tools.py    — Araç protokolü, özerklik politikası, denetim kaydı
    llm.py      — Dil modeli sağlayıcıları (bulut API / yerel sunucu / kendi model)
    coding.py   — Kod tabanı araçları (okuma, arama, düzenleme, test, git)
    shell.py    — PowerShell/CMD derin entegrasyonu (kalıcı oturum, teşhis)
    godot.py    — Godot oyun motoru kontrolü (başsız sahne/script/export)
    blender.py  — Blender 3B kontrolü (başsız modelleme/render/export)
    loop.py     — Planla → uygula → gözlemle döngüsü

Neden bu katman ayrı?
    NeuroCore'un çekirdeği (L0-L8) **bilgi** ve **öğrenme** ile ilgilenir.
    Bu katman ise **eylem** ile ilgilenir: kullanıcının bilgisayarında iş
    yapmak. İkisi aynı araç protokolünü paylaşır ama farklı yaşam döngüleri
    vardır — ajan katmanı yalnızca masaüstü uygulaması çalışırken yüklenir.

Kullanım::

    from neurocore.agent import (AgentLoop, ToolRegistry, ToolContext,
                                 AutonomyPolicy, AutonomyLevel, LLMRouter,
                                 CodingAgent, ShellIntegration)

    registry = ToolRegistry()
    registry.initialize(event_bus=bus, policy=AutonomyPolicy(AutonomyLevel.TRUSTED))

    coding = CodingAgent();  coding.initialize(workspace=proje)
    shell = ShellIntegration(); shell.initialize(workspace=proje)

    registry.register_many(coding.tools())
    registry.register_many(shell.tools())

    router = LLMRouter(); router.initialize(data_path="data")
    loop = AgentLoop(); loop.initialize(registry, router, workspace=proje)
    print(loop.run("bu projedeki testleri çalıştır").render())

Görsel doğrulama araçları `neurocore.vision` altındadır ve ayrıca burada
`register_vision_tools()` ile kaydedilebilir.

Dayanıklılık notu:
    Alt modüller (özellikle `godot` ve `blender`) isteğe bağlıdır — ilgili
    uygulama kurulu değilse veya modül henüz yoksa paket **yine de** içe
    aktarılabilmelidir. Bu yüzden eksik modüller sessizce atlanır ve
    `available_modules()` ile raporlanır; bir simgeye erişilirse anlaşılır
    bir hata verilir.
"""

from typing import Any, Dict, List

from neurocore.agent.coding import CodingAgent, CodingError
from neurocore.agent.llm import (
    KNOWN_SERVICES,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMRouter,
    NeuroCoreProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
)
from neurocore.agent.loop import (
    PLAN_PROMPT,
    RULE_PATTERNS,
    SYSTEM_PROMPT,
    AgentLoop,
    AgentRun,
    AgentStep,
    RulePattern,
    RuleRouter,
)
from neurocore.agent.shell import (
    CLASS_LABELS,
    CLASS_SAFETY,
    CommandClass,
    CommandResult,
    PowerShellDiagnostics,
    ShellIntegration,
    ShellSession,
    classify_command,
    command_safety,
)
from neurocore.agent.tools import (
    SAFETY_LABELS,
    AutonomyLevel,
    AutonomyPolicy,
    Safety,
    ToolCallRecord,
    ToolContext,
    ToolDef,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    collect_tools,
    tool,
)

#: Alt modül → (sınıf adları, içe aktarma hatası)
_OPTIONAL_MODULES: Dict[str, tuple] = {
    "godot": (("GodotController", "GodotError"), None),
    "blender": (("BlenderController", "BlenderError"), None),
}
_OPTIONAL_SYMBOLS: Dict[str, str] = {}

for _module_name, (_names, _error) in _OPTIONAL_MODULES.items():
    try:
        _module = __import__(f"neurocore.agent.{_module_name}", fromlist=list(_names))
        for _name in _names:
            globals()[_name] = getattr(_module, _name)
            _OPTIONAL_SYMBOLS[_name] = _module_name
    except Exception as _exc:  # ImportError veya modül içi hata
        for _name in _names:
            _OPTIONAL_SYMBOLS[_name] = ""
        _OPTIONAL_MODULES[_module_name] = (_names, str(_exc))


def available_modules() -> Dict[str, Any]:
    """
    Hangi yetenek modüllerinin yüklendiğini bildirir.

    Returns:
        {"modules": {ad: hata_veya_None}, "symbols": {sınıf: modül}}
    """
    return {
        "modules": {
            name: (error if error else "hazır")
            for name, (_names, error) in _OPTIONAL_MODULES.items()
        },
        "symbols": dict(_OPTIONAL_SYMBOLS),
    }


def __getattr__(name: str) -> Any:
    """
    Eksik isteğe bağlı simgelere erişildiğinde anlaşılır hata verir.

    `from neurocore.agent import GodotController` çağrısı, modül yoksa
    `ModuleNotFoundError` yerine **neden** yok olduğunu söyleyen bir hata
    üretir.
    """
    if name in _OPTIONAL_SYMBOLS:
        module = _OPTIONAL_SYMBOLS[name]
        error = _OPTIONAL_MODULES.get(module, ((), ""))[1]
        raise ImportError(
            f"'{name}' kullanılamıyor: neurocore.agent.{module} yüklenemedi"
            + (f" ({error})" if error else " (modül eksik)")
        )
    raise AttributeError(f"module 'neurocore.agent' has no attribute '{name}'")

__all__ = [
    # Araç protokolü
    "ToolRegistry", "ToolDef", "ToolParameter", "ToolResult", "ToolContext",
    "ToolCallRecord", "Safety", "SAFETY_LABELS", "AutonomyPolicy",
    "AutonomyLevel", "tool", "collect_tools",
    # Dil modeli
    "LLMRouter", "LLMMessage", "LLMResponse", "LLMProvider", "LLMError",
    "OpenAICompatibleProvider", "OllamaProvider", "NeuroCoreProvider",
    "KNOWN_SERVICES",
    # Yetenekler
    "CodingAgent", "CodingError",
    "ShellIntegration", "ShellSession", "PowerShellDiagnostics",
    "CommandResult", "CommandClass", "CLASS_SAFETY", "CLASS_LABELS",
    "classify_command", "command_safety",
    "GodotController", "GodotError",
    "BlenderController", "BlenderError",
    # Döngü
    "AgentLoop", "AgentRun", "AgentStep", "RuleRouter", "RulePattern",
    "RULE_PATTERNS", "SYSTEM_PROMPT", "PLAN_PROMPT",
]
