"""
NeuroCore Araç Protokolü ve Özerklik Politikası
================================================
Ajanın dünyaya dokunduğu tek kapı.

Neden tek kapı?
    Bir ajanın yapabilecekleri (dosya yazma, komut çalıştırma, uygulama
    kontrolü) sınırsız olduğunda, güvenlik ve denetlenebilirlik kaybolur.
    Bu modül her eylemi **bildirilmiş bir araca** indirger; her aracın bir
    güvenlik sınıfı, bir şema ve bir denetim kaydı vardır.

Özerklik seviyeleri (uygulamadan seçilir)::

    readonly   → Yalnızca okuma/arama/analiz. Hiçbir değişiklik yapılmaz.
    supervised → Yazma ve komut çalıştırma **onay ister** (varsayılan).
    trusted    → Yazma, test, derleme serbest; yıkıcı işlemler onay ister.
    yolo       → Yalnızca kum havuzu (sandbox) sınırları geçerli.

Güvenlik sınıfları::

    read       → Bilgi okuma (dosya oku, ara, listele, ekran görüntüsü)
    write      → Kendi çalışma alanında değişiklik (dosya yaz, düzenle)
    execute    → Komut/uygulama çalıştırma (kabuk, test, derleme)
    destructive→ Geri alınamaz işlem (sil, taşı üzerine yaz, format)
    system     → Sistem düzeyi (registry, servis, kurulum, güç)

Onay akışı, kullanıcı arayüzüne bir **istek** yayınlar ve yanıtı bekler;
politika dışı çağrılar hiç çalıştırılmaz.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.tools")


# ─── Güvenlik Sınıfları ─────────────────────────────────────────────────────

class Safety(str, Enum):
    """
    Bir aracın sistem üzerindeki etki sınıfı.

    Sıralama önemlidir: `READ < WRITE < EXECUTE < DESTRUCTIVE < SYSTEM`.
    Özerklik politikası bu sıralamayı eşik olarak kullanır.
    """
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DESTRUCTIVE = "destructive"
    SYSTEM = "system"

    @property
    def rank(self) -> int:
        """Karşılaştırma için sıra numarası."""
        return _SAFETY_ORDER.index(self)

    def __lt__(self, other: "Safety") -> bool:  # type: ignore[override]
        return self.rank < other.rank

    def __le__(self, other: "Safety") -> bool:  # type: ignore[override]
        return self.rank <= other.rank

    def __gt__(self, other: "Safety") -> bool:  # type: ignore[override]
        return self.rank > other.rank

    def __ge__(self, other: "Safety") -> bool:  # type: ignore[override]
        return self.rank >= other.rank


_SAFETY_ORDER: List[Safety] = [
    Safety.READ, Safety.WRITE, Safety.EXECUTE, Safety.DESTRUCTIVE, Safety.SYSTEM,
]

#: Güvenlik sınıflarının Türkçe etiketleri (arayüz için)
SAFETY_LABELS: Dict[Safety, str] = {
    Safety.READ: "okuma",
    Safety.WRITE: "yazma",
    Safety.EXECUTE: "çalıştırma",
    Safety.DESTRUCTIVE: "yıkıcı",
    Safety.SYSTEM: "sistem",
}


# ─── Özerklik Politikası ────────────────────────────────────────────────────

class AutonomyLevel(str, Enum):
    """
    Ajanın ne kadarını kendi başına yapabileceği.

    Arayüzden canlı olarak değiştirilebilir; her seviye bir öncekinden daha
    geniş yetki verir.
    """
    READONLY = "readonly"
    SUPERVISED = "supervised"
    TRUSTED = "trusted"
    YOLO = "yolo"

    @property
    def label(self) -> str:
        """Türkçe etiket."""
        return {
            AutonomyLevel.READONLY: "Salt okunur",
            AutonomyLevel.SUPERVISED: "Denetimli",
            AutonomyLevel.TRUSTED: "Güvenilir",
            AutonomyLevel.YOLO: "Serbest",
        }[self]

    @property
    def description(self) -> str:
        """
        Arayüzde gösterilecek açıklama.

        Not: Bu metin arayüzde **düz metin** olarak gösterilir (`textContent`),
        bu yüzden markdown işaretleri kullanılmaz — kullanılsaydı kullanıcı
        ham `**` karakterlerini görürdü.
        """
        return {
            AutonomyLevel.READONLY:
                "Hiçbir değişiklik yapılmaz. Yalnızca okuma, arama, analiz ve "
                "ekran görüntüsü. En güvenli seçenek.",
            AutonomyLevel.SUPERVISED:
                "Yazma, çalıştırma ve yıkıcı işlemler onay ister. Okuma serbest. "
                "Önerilen varsayılan.",
            AutonomyLevel.TRUSTED:
                "Yazma, test ve derleme serbest; silme, sistem ayarı ve proje "
                "dışına yazma onay ister.",
            AutonomyLevel.YOLO:
                "Her şey serbest; yalnızca kum havuzu sınırları (proje dışı "
                "yollar, yıkıcı komut kalıpları) geçerlidir. Dikkatli kullanın.",
        }[self]


#: Seviye → otomatik onaylanan en yüksek güvenlik sınıfı
_AUTO_APPROVE: Dict[AutonomyLevel, Safety] = {
    AutonomyLevel.READONLY: Safety.READ,
    AutonomyLevel.SUPERVISED: Safety.READ,
    AutonomyLevel.TRUSTED: Safety.EXECUTE,
    AutonomyLevel.YOLO: Safety.SYSTEM,
}


@dataclass
class AutonomyPolicy:
    """
    Onay kararlarını veren politika.

    Kural:
        * Sınıf, seviyenin otomatik onay eşiğinin altındaysa → **serbest**
        * Değilse → **onay gerekir** (kullanıcı arayüzüne sorulur)
        * Kum havuzu ihlali her seviyede reddedilir

    Ayrıca araç adı bazlı **açık izinler** ve **yasaklar** desteklenir;
    böylece kullanıcı "şu aracı hep serbest bırak" diyebilir.

    Attributes:
        level: Aktif özerklik seviyesi
        allow: Her zaman izinli araç adları (seviyeden bağımsız)
        deny: Her zaman yasaklı araç adları
        require_approval_for: Ek onay gerektiren araç adları
        max_calls_per_minute: Dakikada en fazla araç çağrısı (kaza koruması)
        dry_run: Tüm yazma/çalıştırma araçlarını kuru çalıştırmaya çevir
    """
    level: AutonomyLevel = AutonomyLevel.SUPERVISED
    allow: List[str] = field(default_factory=list)
    deny: List[str] = field(default_factory=list)
    require_approval_for: List[str] = field(default_factory=list)
    max_calls_per_minute: int = 240
    dry_run: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.level, str):
            self.level = AutonomyLevel(self.level)

    # ─── Karar ──────────────────────────────────────────────────────────

    def decide(self, tool_name: str, safety: Safety) -> Tuple[bool, str]:
        """
        Bu araç çağrısı onay gerektiriyor mu?

        Args:
            tool_name: Araç adı
            safety: Aracın güvenlik sınıfı

        Returns:
            (izinli, gerekçe). `izinli=False` ise çağrı **onay beklemeli**;
            gerekçe kullanıcıya gösterilir.
        """
        if tool_name in self.deny:
            return False, f"'{tool_name}' aracı politika gereği yasaklı"

        if self.level == AutonomyLevel.READONLY and safety != Safety.READ:
            return False, (
                f"salt okunur modda '{tool_name}' ({SAFETY_LABELS[safety]}) "
                "çalıştırılamaz"
            )

        if tool_name in self.allow and tool_name not in self.require_approval_for:
            return True, "açık izin listesinde"

        if tool_name in self.require_approval_for:
            return False, f"'{tool_name}' için her zaman onay gerekir"

        threshold = _AUTO_APPROVE.get(self.level, Safety.READ)
        if safety.rank <= threshold.rank:
            return True, f"{self.level.label} modunda {SAFETY_LABELS[safety]} serbest"

        return False, (
            f"{self.level.label} modunda {SAFETY_LABELS[safety]} işlemleri "
            f"için onay gerekir ('{tool_name}')"
        )

    def set_level(self, level: Any) -> AutonomyLevel:
        """Seviyeyi değiştirir (arayüzden çağrılır)."""
        if not isinstance(level, AutonomyLevel):
            level = AutonomyLevel(str(level))
        old = self.level
        self.level = level
        logger.info(f"Özerklik seviyesi: {old.label} → {level.label}")
        return level

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "level": self.level.value,
            "label": self.level.label,
            "description": self.level.description,
            "allow": list(self.allow),
            "deny": list(self.deny),
            "require_approval_for": list(self.require_approval_for),
            "dry_run": self.dry_run,
            "max_calls_per_minute": self.max_calls_per_minute,
            "auto_approve_up_to": SAFETY_LABELS[_AUTO_APPROVE[self.level]],
        }

    @staticmethod
    def levels() -> List[Dict[str, Any]]:
        """Tüm seviyelerin açıklamalı listesi (arayüz menüsü için)."""
        return [
            {"value": level.value, "label": level.label,
             "description": level.description,
             "auto_approve_up_to": SAFETY_LABELS[_AUTO_APPROVE[level]]}
            for level in AutonomyLevel
        ]


# ─── Araç Tanımı ────────────────────────────────────────────────────────────

#: Sağlayıcıların araç adlarında kabul ettiği karakterler.
#:
#: Bazı sağlayıcılar katıdır ve şu kalıbı şart koşar: `^[a-zA-Z0-9_-]+$`.
#: DeepSeek bu kuralı uygular ve nokta içeren adı reddeder:
#:     `HTTP 400 Invalid 'tools[0].function.name': string does not match
#:      pattern. Expected a string that matches the pattern '^[a-zA-Z0-9_-]+$'`
#: Bu yüzden adlar API'ye gönderilirken **normalize edilir**
#: (`code.create_file` → `code_create_file`) ve gelen çağrı geri eşlenir.
WIRE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

#: Ayrıştırıcı olarak kullanılan karakter (nokta → alt çizgi)
_WIRE_SEPARATOR = "_"


def to_wire_name(name: str) -> str:
    """
    Araç adını API-güvenli biçime çevirir.

    Nokta ve diğer izin verilmeyen karakterler alt çizgiye dönüştürülür.
    Ad zaten güvenliyse **değiştirilmez** (gereksiz dönüşüm, kayıtlı adı
    arayan kodla uyumsuzluk yaratırdı).

    Args:
        name: Kayıtlı araç adı (ör. `code.create_file`)

    Returns:
        API-güvenli ad (ör. `code_create_file`)
    """
    if not name or WIRE_NAME_PATTERN.match(name):
        return name
    return re.sub(r"[^a-zA-Z0-9_-]", _WIRE_SEPARATOR, name)


def from_wire_name(name: str, registry: "ToolRegistry") -> Optional[str]:
    """
    API'den gelen araç adını kayıtlı **gerçek** adla eşleştirir.

    Sıra:
        1. Ad zaten kayıtlıysa doğrudan kabul
        2. Alt çizgi/nokta farkını gözeterek tam eşleşme
        3. Kategori ön ekini atlayarak kuyruk eşleşmesi

    Args:
        name: Modelden gelen ad (ör. `code_create_file`)
        registry: Araç kayıt defteri

    Returns:
        Kayıtlı gerçek ad veya None (eşleşme yok)
    """
    if not name:
        return None
    if registry.has(name):
        return name

    def normalise(value: str) -> str:
        """Nokta/alt çizgi/tire farkını siler."""
        return value.lower().replace(".", "_").replace("-", "_")

    target = normalise(name)
    matches = [item for item in registry.names() if normalise(item) == target]
    if len(matches) == 1:
        return matches[0]

    # Kuyruk eşleşmesi: `create_file` → `code.create_file`
    tails = [item for item in registry.names()
             if normalise(item).split("_", 1)[-1] == target
             or normalise(item).endswith("_" + target)]
    if len(tails) == 1:
        return tails[0]
    return None


@dataclass
class ToolParameter:
    """Tek bir araç parametresi (LLM şeması için)."""
    name: str
    type: str = "string"            # string | integer | number | boolean | array | object
    description: str = ""
    required: bool = False
    default: Any = None
    enum: Optional[List[str]] = None

    def to_schema(self) -> Dict[str, Any]:
        """JSON Schema parçası."""
        schema: Dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = list(self.enum)
        if self.default is not None:
            schema["default"] = self.default
        return schema


def _kirp_veri(veri: Any, butce: int, _derinlik: int = 0) -> Any:
    """
    Araç çıktısını **yapıyı bozmadan** kısaltır.

    ⚠ `json.dumps(...)[:max_chars]` yöntemi sözlüğü string'e çeviriyordu ve
    ajan dosya içeriğini göremiyordu. Bu yardımcı; sözlük/liste yapısını
    korur, yalnız uzun metin alanlarından keser.

    Args:
        veri: Araç çıktısı (sözlük, liste veya skaler)
        butce: Metin alanları için üst sınır (karakter)
        _derinlik: Özyineleme koruması

    Returns:
        Kısaltılmış ama aynı yapıda veri
    """
    if _derinlik > 6:
        return "…"
    if isinstance(veri, str):
        if len(veri) > butce:
            return veri[:butce] + f"\n… (+{len(veri) - butce} karakter)"
        return veri
    if isinstance(veri, dict):
        return {k: _kirp_veri(v, butce, _derinlik + 1) for k, v in veri.items()}
    if isinstance(veri, (list, tuple)):
        return [_kirp_veri(v, butce, _derinlik + 1) for v in list(veri)[:500]]
    return veri


@dataclass
class ToolResult:
    """
    Tek bir araç çağrısının sonucu.

    Ajan döngüsü bu yapıyı okur ve bir sonraki adımı buna göre seçer.
    """
    ok: bool
    tool: str = ""
    data: Any = None
    error: str = ""
    duration_ms: float = 0.0
    safety: str = ""
    blocked: bool = False
    needs_approval: bool = False
    reason: str = ""
    truncated: bool = False

    def to_dict(self, max_chars: int = 12000) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl (çıktı sınırlandırılır)."""
        payload = {
            "ok": self.ok, "tool": self.tool, "error": self.error,
            "duration_ms": round(self.duration_ms, 1), "safety": self.safety,
            "blocked": self.blocked, "needs_approval": self.needs_approval,
            "reason": self.reason, "truncated": self.truncated,
        }
        if self.data is not None:
            try:
                text = json.dumps(self.data, ensure_ascii=False, default=str)
                if len(text) > max_chars:
                    # ⚠ ÖLÇÜLEN HATA: burada data JSON **string**'e çevriliyordu;
                    # ajanın araç-mesajı biçimlendiricisi sözlük beklediği için
                    # dosya içeriği kayboluyor, model yalnızca meta veri görüyordu.
                    # Artık **yapı korunur**, uzun metin alanları tek tek kısaltılır.
                    payload["data"] = _kirp_veri(self.data, max_chars)
                    payload["truncated"] = True
                    payload["note"] = (f"çıktı {len(text)} karakterden "
                                       "kısaltıldı (alanlar korundu)")
                else:
                    payload["data"] = self.data
            except (TypeError, ValueError):
                text = str(self.data)
                payload["data"] = text[:max_chars]
                payload["truncated"] = len(text) > max_chars
        return payload

    def render(self, max_chars: int = 2000) -> str:
        """İnsan/LLM okur kısa metin."""
        if self.blocked:
            return f"[ENGELLENDİ] {self.tool}: {self.reason}"
        if self.needs_approval:
            return f"[ONAY BEKLİYOR] {self.tool}: {self.reason}"
        if not self.ok:
            return f"[HATA] {self.tool}: {self.error or 'bilinmeyen hata'}"
        try:
            text = json.dumps(self.data, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(self.data)
        if len(text) > max_chars:
            text = text[:max_chars] + f"… (+{len(text) - max_chars} karakter)"
        return f"[OK] {self.tool}: {text}"


@dataclass
class ToolDef:
    """
    Bir aracın tam tanımı.

    Attributes:
        name: Benzersiz ad (LLM bu adı çağırır; `kategori.islev` biçimi)
        description: Ne yaptığı (LLM'e verilir — net ve eylem odaklı olmalı)
        handler: Çağrılabilir; `(**kwargs) -> Any` veya `(ctx, **kwargs) -> Any`
        parameters: Parametre listesi
        safety: Güvenlik sınıfı
        category: Grup (`code`, `shell`, `app`, `vision`, `knowledge`, ...)
        returns: Dönüş değerinin açıklaması (LLM için)
        requires: Bu aracın çalışması için gereken yetenek adı
        timeout: Saniye (None = sınırsız)
        tags: Arama için etiketler
    """
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: List[ToolParameter] = field(default_factory=list)
    safety: Safety = Safety.READ
    category: str = "general"
    returns: str = ""
    requires: str = ""
    timeout: Optional[float] = None
    tags: List[str] = field(default_factory=list)

    @property
    def takes_context(self) -> bool:
        """Handler ilk argüman olarak bağlam alıyor mu?"""
        try:
            signature = inspect.signature(self.handler)
            parameters = list(signature.parameters.values())
            if not parameters:
                return False
            first = parameters[0]
            return first.name in ("ctx", "context") or first.annotation in ("ToolContext", "ctx")
        except (TypeError, ValueError):
            return False

    def schema(self) -> Dict[str, Any]:
        """
        LLM araç şeması (JSON Schema + OpenAI `function` biçimi).

        Hem Anthropic (`input_schema`) hem OpenAI (`parameters`) alanına
        doğrudan konabilecek ortak yapı üretilir.
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for parameter in self.parameters:
            properties[parameter.name] = parameter.to_schema()
            if parameter.required:
                required.append(parameter.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def brief(self) -> Dict[str, Any]:
        """Arayüz için kısa tanım."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "safety": self.safety.value,
            "safety_label": SAFETY_LABELS[self.safety],
            "parameters": [parameter.name for parameter in self.parameters],
            "returns": self.returns,
            "tags": list(self.tags),
        }


@dataclass
class ToolContext:
    """
    Araç çağrısı bağlamı.

    Araçların birbirini import etmeden ortak duruma erişmesini sağlar:
    çalışma alanı yolu, olay bus'ı, iptal bayrağı, ayar nesnesi.
    """
    workspace: str = ""
    event_bus: Any = None
    config: Any = None
    engine: Any = None
    cancel_event: Optional[threading.Event] = None
    autonomy: Optional[AutonomyPolicy] = None
    dry_run: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def cancelled(self) -> bool:
        """İptal istenmiş mi?"""
        return bool(self.cancel_event and self.cancel_event.is_set())

    def emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self.event_bus is None:
            return
        try:
            self.event_bus.emit(event_type, data, source="agent.tools")
        except Exception:
            pass


# ─── Araç Kayıt Defteri ─────────────────────────────────────────────────────

@dataclass
class ToolCallRecord:
    """Tek bir araç çağrısının denetim kaydı."""
    at: float
    tool: str
    safety: str
    ok: bool
    duration_ms: float
    approved: bool = False
    blocked: bool = False
    reason: str = ""
    args_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "at": self.at, "time": time.strftime("%H:%M:%S", time.localtime(self.at)),
            "tool": self.tool, "safety": self.safety, "ok": self.ok,
            "duration_ms": round(self.duration_ms, 1), "approved": self.approved,
            "blocked": self.blocked, "reason": self.reason,
            "args": self.args_preview,
        }


class ToolRegistry:
    """
    Araçların kayıt defteri ve çağrı yöneticisi.

    Kullanım:
        registry = ToolRegistry()
        registry.initialize(event_bus=bus, policy=AutonomyPolicy())

        registry.register(ToolDef(
            name="code.read_file",
            description="Bir dosyayı okur",
            handler=read_file,
            parameters=[ToolParameter("path", required=True)],
            safety=Safety.READ,
            category="code",
        ))

        result = registry.call("code.read_file", {"path": "a.py"})
        print(result.render())

    Politika dışı çağrılar `ToolResult(needs_approval=True)` döner; ajan
    döngüsü bunu kullanıcıya sorar ve onay sonrası `approve()` + `call()`
    ile yeniden çalıştırır.
    """

    name = "agent.tools"

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDef] = {}
        self._lock = threading.RLock()
        self._event_bus: Any = None
        self._policy = AutonomyPolicy()
        self._context = ToolContext()
        self._approvals: Dict[str, float] = {}      # tool adı → onay zamanı
        self._call_times: List[float] = []
        self._audit: List[ToolCallRecord] = []
        self._initialized = False
        self._counter = {
            "calls": 0, "blocked": 0, "failed": 0, "approved": 0,
            "denied": 0, "rate_limited": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, event_bus: Any = None, policy: Optional[AutonomyPolicy] = None,
                   context: Optional[ToolContext] = None) -> None:
        """Kayıt defterini başlatır (idempotent)."""
        if self._initialized:
            return
        self._event_bus = event_bus
        if policy is not None:
            self._policy = policy
        if context is not None:
            self._context = context
        if self._context.event_bus is None:
            self._context.event_bus = event_bus
        self._context.autonomy = self._policy
        self._initialized = True
        logger.info(
            f"Araç kayıt defteri hazır — özerklik: {self._policy.level.label}"
        )

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır."""
        self._initialized = False
        logger.info(
            f"Araç kayıt defteri kapatıldı ({self._counter['calls']} çağrı, "
            f"{self._counter['blocked']} engel, {self._counter['failed']} hata)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """İstatistikler."""
        by_category: Dict[str, int] = {}
        by_safety: Dict[str, int] = {}
        for tool in self._tools.values():
            by_category[tool.category] = by_category.get(tool.category, 0) + 1
            by_safety[tool.safety.value] = by_safety.get(tool.safety.value, 0) + 1
        return {
            **self._counter,
            "tools": len(self._tools),
            "by_category": by_category,
            "by_safety": by_safety,
            "policy": self._policy.to_dict(),
            "initialized": self._initialized,
        }

    # ─── Kayıt ──────────────────────────────────────────────────────────

    def register(self, tool: ToolDef, replace: bool = False) -> bool:
        """
        Araç kaydeder.

        Args:
            tool: Araç tanımı
            replace: Aynı ad varsa üzerine yaz

        Returns:
            Kaydedildiyse True
        """
        with self._lock:
            if tool.name in self._tools and not replace:
                logger.warning(f"Araç zaten kayıtlı: {tool.name} (replace=False)")
                return False
            self._tools[tool.name] = tool
        logger.debug(f"Araç kaydedildi: {tool.name} [{SAFETY_LABELS[tool.safety]}]")
        return True

    def register_many(self, tools: Sequence[ToolDef], replace: bool = False) -> int:
        """Çok sayıda araç kaydeder; kaydedilen sayısını döndürür."""
        return sum(1 for tool in tools if self.register(tool, replace=replace))

    def unregister(self, name: str) -> bool:
        """Araç kaldırır."""
        with self._lock:
            return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[ToolDef]:
        """Araç tanımını döndürür."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Araç kayıtlı mı?"""
        return name in self._tools

    def resolve(self, name: str) -> Optional[str]:
        """
        API'den gelen adı kayıtlı gerçek adla eşleştirir.

        Şemalar `to_wire_name()` ile normalize edilerek gönderildiği için
        modelin döndürdüğü ad (`code_create_file`) kayıtlı addan
        (`code.create_file`) farklıdır. Bu metot köprüyü kurar.
        """
        return from_wire_name(name, self)

    def names(self) -> List[str]:
        """Kayıtlı araç adları."""
        return sorted(self._tools.keys())

    def list_tools(self, category: Optional[str] = None,
                   safety: Optional[Safety] = None,
                   query: str = "") -> List[Dict[str, Any]]:
        """
        Araçları listeler (arayüz ve LLM istemi için).

        Args:
            category: Kategori filtresi
            safety: Güvenlik sınıfı filtresi
            query: Ad/açıklama araması
        """
        needle = query.lower().strip()
        result: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            if category and tool.category != category:
                continue
            if safety and tool.safety != safety:
                continue
            if needle and needle not in tool.name.lower() \
                    and needle not in tool.description.lower():
                continue
            result.append(tool.brief())
        return sorted(result, key=lambda item: (item["category"], item["name"]))

    def categories(self) -> Dict[str, int]:
        """Kategori → araç sayısı."""
        counts: Dict[str, int] = {}
        for tool in self._tools.values():
            counts[tool.category] = counts.get(tool.category, 0) + 1
        return dict(sorted(counts.items()))

    def schemas(self, category: Optional[str] = None,
                names: Optional[Sequence[str]] = None,
                max_safety: Optional[Safety] = None) -> List[Dict[str, Any]]:
        """
        LLM'e verilecek araç şemalarını üretir.

        Args:
            category: Yalnızca bu kategori
            names: Yalnızca bu araç adları
            max_safety: Bu güvenlik sınıfına kadar olan araçlar
                (salt okunur modda yıkıcı araçlar isteme)

        Returns:
            OpenAI/Anthropic uyumlu fonksiyon şemaları.

        Not:
            Araç adları **API-güvenli** biçime çevrilir (`code.create_file` →
            `code_create_file`): bazı sağlayıcılar (DeepSeek) `^[a-zA-Z0-9_-]+$`
            kalıbını şart koşar ve nokta içeren adı `HTTP 400` ile reddeder.
            Gelen çağrı `from_wire_name()` ile geri eşlenir.
        """
        selected: List[ToolDef] = []
        for tool in self._tools.values():
            if category and tool.category != category:
                continue
            if names is not None and tool.name not in names:
                continue
            if max_safety is not None and tool.safety.rank > max_safety.rank:
                continue
            selected.append(tool)
        selected.sort(key=lambda item: item.name)

        result: List[Dict[str, Any]] = []
        for tool in selected:
            schema = tool.schema()
            schema["function"]["name"] = to_wire_name(tool.name)
            result.append(schema)
        return result

    def describe_for_prompt(self, category: Optional[str] = None,
                            max_tools: int = 80) -> str:
        """
        Araç listesini LLM istemi için **metin** olarak biçimlendirir.

        Şema tabanlı çağrı desteklemeyen modeller için gereklidir: model
        yanıtında `{"tool": "ad", "args": {...}}` biçiminde bir JSON bloğu
        üretir.
        """
        lines: List[str] = ["Kullanılabilir araçlar:"]
        grouped: Dict[str, List[ToolDef]] = {}
        for tool in self._tools.values():
            if category and tool.category != category:
                continue
            grouped.setdefault(tool.category, []).append(tool)

        count = 0
        for group, tools in sorted(grouped.items()):
            lines.append(f"\n## {group}")
            for tool in sorted(tools, key=lambda item: item.name):
                if count >= max_tools:
                    lines.append("… (daha fazla araç var)")
                    break
                params = ", ".join(
                    f"{parameter.name}{'*' if parameter.required else ''}"
                    for parameter in tool.parameters
                )
                badge = "" if tool.safety == Safety.READ else \
                    f" [{SAFETY_LABELS[tool.safety]}]"
                lines.append(f"- `{tool.name}({params})`{badge}: {tool.description}")
                count += 1
        lines.append(
            "\nAraç çağırmak için yanıtına şu biçimde bir JSON bloğu ekle:\n"
            '```json\n{"tool": "araç.adı", "args": {"parametre": "değer"}}\n```\n'
            "Birden çok araç gerekiyorsa sırayla çağır; her çağrının sonucunu gör."
        )
        return "\n".join(lines)

    # ─── Politika ───────────────────────────────────────────────────────

    @property
    def policy(self) -> AutonomyPolicy:
        """Aktif politika."""
        return self._policy

    def set_policy(self, policy: AutonomyPolicy) -> None:
        """Politikayı değiştirir."""
        self._policy = policy
        self._context.autonomy = policy
        logger.info(f"Araç politikası güncellendi: {policy.level.label}")

    def set_level(self, level: Any) -> Dict[str, Any]:
        """
        Özerklik seviyesini değiştirir (arayüzden çağrılır).

        Seviye yükseltildiğinde bekleyen onaylar geçersiz sayılır.
        """
        previous = self._policy.level
        self._policy.set_level(level)
        self._approvals.clear()
        info = self._policy.to_dict()
        self._emit("agent.autonomy_changed", {
            "previous": previous.value, "current": self._policy.level.value,
        })
        return info

    def approve(self, tool_name: str, remember: bool = False,
                consume: bool = True) -> None:
        """
        Bir aracın çağrısını onaylar.

        Args:
            tool_name: Onaylanan araç
            remember: True ise bu araç **oturum boyunca** serbest bırakılır
                (politika `allow` listesine eklenir). Kullanıcının "hepsi"
                seçeneği bunu kullanır.
            consume: True (varsayılan) ise onay **tek kullanımlıktır** — bir
                sonraki çağrıda tüketilir ve tekrar onay istenir.

        Neden tek kullanımlık?
            Onayı süresiz saklamak ciddi bir güvenlik açığıdır: kullanıcı bir
            kez "izin ver" dediğinde, ajan aynı aracı sonraki adımlarda
            **sormadan** sonsuz kez çalıştırabilir. Onay, sorulduğu çağrıya
            aittir; kullanıcı açıkça "hatırla" demedikçe yeniden sorulur.
        """
        self._counter["approved"] += 1
        if remember:
            if tool_name not in self._policy.allow:
                self._policy.allow.append(tool_name)
            self._approvals.pop(tool_name, None)
            logger.info(f"Araç oturum boyunca onaylandı: {tool_name}")
            return
        self._approvals[tool_name] = time.time()
        logger.info(
            f"Araç onaylandı: {tool_name}"
            + (" (tek kullanımlık)" if consume else "")
        )

    def is_approved(self, tool_name: str, consume: bool = True,
                    max_age: float = 300.0) -> bool:
        """
        Araç için geçerli bir onay var mı?

        Args:
            tool_name: Araç adı
            consume: True ise onay tüketilir (tek kullanımlık)
            max_age: Bu süreden eski onaylar geçersiz sayılır (güvenlik ağı)

        Returns:
            Geçerli onay varsa True
        """
        if tool_name in self._policy.allow:
            return True
        stamp = self._approvals.get(tool_name)
        if stamp is None:
            return False
        if (time.time() - stamp) > max_age:
            self._approvals.pop(tool_name, None)
            return False
        if consume:
            self._approvals.pop(tool_name, None)
        return True

    def deny(self, tool_name: str, remember: bool = False) -> None:
        """Bir aracın çağrısını reddeder."""
        self._approvals.pop(tool_name, None)
        self._counter["denied"] += 1
        if remember:
            if tool_name not in self._policy.deny:
                self._policy.deny.append(tool_name)
        logger.info(f"Araç reddedildi: {tool_name}")

    def clear_approvals(self) -> int:
        """Tüm geçici onayları temizler."""
        count = len(self._approvals)
        self._approvals.clear()
        return count

    # ─── Çağrı ──────────────────────────────────────────────────────────

    def call(self, name: str, args: Optional[Dict[str, Any]] = None,
             *, context: Optional[ToolContext] = None,
             force: bool = False) -> ToolResult:
        """
        Aracı çağırır (politika denetimiyle).

        Args:
            name: Araç adı
            args: Parametreler
            context: Geçici bağlam (None = kayıtlı bağlam)
            force: Politika denetimini atla (yalnızca iç kullanım/onay sonrası)

        Returns:
            `ToolResult` — `needs_approval=True` ise çağrı **yapılmadı**.
        """
        started = time.time()
        arguments = dict(args or {})
        ctx = context or self._context

        tool = self._tools.get(name)
        if tool is None:
            self._counter["failed"] += 1
            return ToolResult(ok=False, tool=name, error=f"bilinmeyen araç: {name}",
                              reason="araç kayıtlı değil")

        # ── 1. Kuru çalıştırma ──
        # Politika denetiminden **önce** gelir: kullanıcı "hiçbir şey
        # değiştirme, sadece ne yapacağını göster" dediğinde onay sorulması
        # anlamsızdır.
        if (self._policy.dry_run or ctx.dry_run) and tool.safety != Safety.READ:
            self._emit("agent.tool_dry_run", {"tool": name, "args": arguments})
            return ToolResult(
                ok=True, tool=name, safety=tool.safety.value,
                data={"dry_run": True, "would_call": name, "args": arguments,
                      "note": "kuru çalıştırma: hiçbir değişiklik yapılmadı"},
                reason="kuru çalıştırma modu",
                duration_ms=(time.time() - started) * 1000,
            )

        # ── 2. Politika denetimi ──
        if not force:
            allowed, reason = self._policy.decide(name, tool.safety)
            if not allowed:
                # Kullanıcı bu çağrı için onay verdiyse geç (onay tek kullanımlık)
                if not self.is_approved(name, consume=True):
                    self._append_audit(ToolCallRecord(
                        at=started, tool=name, safety=tool.safety.value, ok=False,
                        duration_ms=0.0, blocked=True, reason=reason,
                        args_preview=_preview_args(arguments),
                    ))
                    self._emit("agent.tool_blocked", {
                        "tool": name, "reason": reason, "safety": tool.safety.value,
                        "args": arguments,
                    })
                    return ToolResult(
                        ok=False, tool=name, safety=tool.safety.value,
                        blocked=True, needs_approval=True, reason=reason,
                        error=reason, duration_ms=(time.time() - started) * 1000,
                    )

        # ── 3. Hız sınırı (kaza koruması) ──
        if not self._check_rate():
            self._counter["rate_limited"] += 1
            return ToolResult(
                ok=False, tool=name, safety=tool.safety.value, blocked=True,
                error="çağrı hızı sınırı aşıldı", reason="dakikada çok fazla araç çağrısı",
                duration_ms=(time.time() - started) * 1000,
            )

        # ── 4. Çalıştır ──
        self._counter["calls"] += 1
        self._call_times.append(time.time())
        self._emit("agent.tool_started", {
            "tool": name, "safety": tool.safety.value, "args": arguments,
        })

        try:
            if tool.takes_context:
                output = tool.handler(ctx, **arguments)
            else:
                output = tool.handler(**arguments)
            duration = (time.time() - started) * 1000
            result = ToolResult(ok=True, tool=name, data=output,
                                safety=tool.safety.value, duration_ms=duration)
            self._emit("agent.tool_finished", {
                "tool": name, "ok": True, "duration_ms": duration,
            })
        except TypeError as exc:
            # Parametre hatası: modele net geri bildirim ver
            duration = (time.time() - started) * 1000
            self._counter["failed"] += 1
            result = ToolResult(
                ok=False, tool=name, safety=tool.safety.value,
                error=f"parametre hatası: {exc}",
                reason=f"beklenen parametreler: {[p.name for p in tool.parameters]}",
                duration_ms=duration,
            )
            self._emit("agent.tool_finished", {
                "tool": name, "ok": False, "error": str(exc),
            })
        except Exception as exc:
            duration = (time.time() - started) * 1000
            self._counter["failed"] += 1
            result = ToolResult(
                ok=False, tool=name, safety=tool.safety.value,
                error=f"{type(exc).__name__}: {exc}", duration_ms=duration,
            )
            logger.warning(f"Araç hatası ({name}): {exc}")
            self._emit("agent.tool_finished", {
                "tool": name, "ok": False, "error": str(exc),
            })

        self._append_audit(ToolCallRecord(
            at=started, tool=name, safety=tool.safety.value, ok=result.ok,
            duration_ms=result.duration_ms,
            approved=name in self._approvals, blocked=False,
            reason=result.error, args_preview=_preview_args(arguments),
        ))
        return result

    def call_many(self, calls: Sequence[Dict[str, Any]],
                  stop_on_error: bool = False) -> List[ToolResult]:
        """
        Birden çok aracı sırayla çağırır.

        Args:
            calls: `[{"tool": "ad", "args": {...}}, ...]`
            stop_on_error: İlk hatada dur

        Returns:
            Sonuç listesi
        """
        results: List[ToolResult] = []
        for call in calls:
            name = str(call.get("tool") or call.get("name") or "")
            args = call.get("args") or call.get("arguments") or {}
            result = self.call(name, args if isinstance(args, dict) else {})
            results.append(result)
            if stop_on_error and not result.ok:
                break
        return results

    def _check_rate(self) -> bool:
        """Dakikalık çağrı sınırını denetler."""
        limit = max(1, int(self._policy.max_calls_per_minute))
        now = time.time()
        self._call_times = [stamp for stamp in self._call_times if now - stamp < 60.0]
        return len(self._call_times) < limit

    def _append_audit(self, record: ToolCallRecord) -> None:
        """Denetim kaydı ekler (sınırlı boyutta)."""
        with self._lock:
            self._audit.append(record)
            if len(self._audit) > 2000:
                self._audit = self._audit[-2000:]

    def audit(self, limit: int = 50, tool: Optional[str] = None) -> List[Dict[str, Any]]:
        """Son araç çağrılarının denetim kaydı."""
        with self._lock:
            records = list(self._audit)
        if tool:
            records = [record for record in records if record.tool == tool]
        return [record.to_dict() for record in records[-limit:]]

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<ToolRegistry tools={len(self._tools)} "
                f"level={self._policy.level.value} calls={self._counter['calls']}>")


def _preview_args(args: Dict[str, Any], max_length: int = 200) -> str:
    """Argümanların kısa, güvenli önizlemesi (denetim kaydı için)."""
    try:
        text = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(args)
    # Uzun içerikleri kısalt (dosya gövdeleri denetim kaydını şişirmesin)
    if len(text) > max_length:
        text = text[:max_length] + "…"
    return text


def tool(
    name: str,
    description: str,
    *,
    safety: Safety = Safety.READ,
    category: str = "general",
    parameters: Optional[List[ToolParameter]] = None,
    returns: str = "",
    requires: str = "",
    tags: Optional[List[str]] = None,
    timeout: Optional[float] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Fonksiyonu araç olarak işaretleyen dekoratör.

    Kullanım:
        @tool("code.read_file", "Dosya okur", safety=Safety.READ, category="code",
              parameters=[ToolParameter("path", required=True)])
        def read_file(path: str) -> dict:
            ...

        registry.register_many(collect_tools(globals()))

    Dekoratör, fonksiyona `__tool_def__` özniteliği ekler; `ToolDef`'e çevirmek
    için `to_tool_def()` kullanılır.
    """
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        function.__tool_def__ = ToolDef(  # type: ignore[attr-defined]
            name=name,
            description=description,
            handler=function,
            parameters=list(parameters or []),
            safety=safety,
            category=category,
            returns=returns,
            requires=requires,
            tags=list(tags or []),
            timeout=timeout,
        )
        return function
    return decorator


def collect_tools(namespace: Dict[str, Any]) -> List[ToolDef]:
    """
    Bir modül sözlüğündeki işaretli fonksiyonları toplar.

    Args:
        namespace: `globals()` veya benzeri

    Returns:
        `ToolDef` listesi
    """
    tools: List[ToolDef] = []
    seen: set = set()
    for value in namespace.values():
        definition = getattr(value, "__tool_def__", None)
        if definition is not None and definition.name not in seen:
            seen.add(definition.name)
            tools.append(definition)
    return tools


def describe_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Bir araç çağrısını **anlaşılır** tek satır olarak özetler.

    Onay isteğinde kullanıcıya ham JSON göstermek yanıltıcıdır; kullanıcı bu
    bilgiyle karar vereceği için ne yapılacağı açıkça yazılmalıdır. Bu
    fonksiyon ajan döngüsü ve arayüz tarafından **ortak** kullanılır ki
    kullanıcı her yerde aynı açıklamayı görsün.
    """
    arguments = arguments or {}
    if tool_name in ("code.edit_file",):
        old = str(arguments.get("old_text", ""))[:60].replace("\n", "⏎")
        new = str(arguments.get("new_text", ""))[:60].replace("\n", "⏎")
        return (f"`{arguments.get('path', '?')}` dosyasında değişiklik: "
                f"'{old}' → '{new}'")
    if tool_name in ("code.create_file",):
        return f"yeni dosya oluştur: `{arguments.get('path', '?')}`"
    if tool_name in ("code.run_command", "shell.run", "shell.session_run", "shell.cmd"):
        return f"komut çalıştır: `{str(arguments.get('command', ''))[:140]}`"
    if tool_name == "code.run_tests":
        return f"testleri çalıştır ({arguments.get('framework') or 'otomatik algılama'})"
    if tool_name == "code.revert":
        return (f"değişikliği geri al: "
                f"{arguments.get('path') or arguments.get('backup_id') or '?'}")
    if tool_name.startswith("godot."):
        return f"Godot: {tool_name.split('.', 1)[1]} {_brief_args(arguments, 50)}".strip()
    if tool_name.startswith("blender."):
        return f"Blender: {tool_name.split('.', 1)[1]} {_brief_args(arguments, 50)}".strip()
    if tool_name.startswith("vision."):
        return f"görsel doğrulama: {tool_name.split('.', 1)[1]} {_brief_args(arguments, 50)}".strip()
    return f"{tool_name} {_brief_args(arguments, 90)}".strip()


def _brief_args(args: Dict[str, Any], max_length: int = 70) -> str:
    """Argümanların kısa önizlemesi (özet satırları için)."""
    if not args:
        return ""
    parts: List[str] = []
    for key, value in list(args.items())[:3]:
        text = str(value).replace("\n", "⏎")
        if len(text) > 30:
            text = text[:30] + "…"
        parts.append(f"{key}={text}")
    return ", ".join(parts)[:max_length]


__all__ = [
    "Safety",
    "SAFETY_LABELS",
    "AutonomyLevel",
    "AutonomyPolicy",
    "ToolParameter",
    "ToolResult",
    "ToolDef",
    "ToolContext",
    "ToolCallRecord",
    "ToolRegistry",
    "tool",
    "collect_tools",
    "describe_tool_call",
]
