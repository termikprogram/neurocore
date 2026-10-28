"""
NeuroCore Kural Tabanlı Görev Ayrıştırıcı (Decomposer)
======================================================
Doğal dil hedefini **LLM olmadan** yapılandırılmış bir ``TaskGraph``'a çevirir.

Çalışma biçimi:
    1. ``classify()``  → niyet, karmaşıklık, gereksinimler, dil, anahtar kelimeler
    2. Şablon motoru  → niyete uyan ilk şablon ``TaskGraph``'ı üretir
    3. Özyineleme     → karmaşık hedefler alt hedeflere bölünüp grafta eklenir
    4. Bütçe          → toplam görev sayısı sınırlanır (MAX_TOTAL_TASKS)

Tasarım gereği bu modül diğer NeuroCore katmanlarını **import etmez**;
yalnızca ``Task``/``TaskGraph`` veri modeline ve enjekte edilen yetenek
adlarına dayanır. Böylece tek başına test edilebilir.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from neurocore.planner.task import (
    PlanError,
    Task,
    TaskGraph,
    TaskKind,
    TaskState,
    json_safe,
    make_id_factory,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("planner.decomposer")


# ─── Sabitler ───────────────────────────────────────────────────────────────

#: Şablon çalıştırıcılarının imzası: (goal, classification, context, depth) → TaskGraph
TemplateBuilder = Callable[[str, Dict[str, Any], Dict[str, Any], int], TaskGraph]

#: Bir matcher ya hedef metnini alır ya da daha zengin bağlam ister.
TemplateMatcher = Callable[[str], Any]

#: Özyinelemeli ayrıştırmada izin verilen en fazla toplam görev.
MAX_TOTAL_TASKS = 60

#: Alt hedef üretiminde en fazla parça sayısı.
MAX_SUBGOALS = 6

#: Alt hedef özyinelemesinde en fazla genişlik (her seviyede).
MAX_EXPANSION_BRANCHES = 3

#: Kasadan yazarken kabul edilen en düşük güven (memorize eşiği).
MEMORIZE_CONFIDENCE_THRESHOLD = 0.6


# ─── Niyet Sözlükleri (Türkçe + İngilizce) ──────────────────────────────────

#: Soru sözcükleri — hedef bilgi talebi mi?
INTERROGATIVES: Tuple[str, ...] = (
    "neden", "niçin", "nasıl", "nasil", "ne ", "ne?", "ne,", "nerede", "hangi",
    "kim", "kaç", "kac", "mi?", "mı?", "mü?", "mu?", "midir", "mıdır",
    "why", "how", "what", "which", "where", "who", "when",
)

#: Emir kipleri — hedef eylem talebi mi?
IMPERATIVES: Tuple[str, ...] = (
    "yap", "oluştur", "olustur", "araştır", "araştir", "hesapla", "yaz",
    "analiz et", "karşılaştır", "karsilastir", "bul", "listele", "düzelt",
    "duzelt", "kur", "göster", "goster", "incele", "özetle", "ozetle",
    "açıkla", "acikla", "temizle", "optimize et", "planla", "ara", "kontrol et",
    "build", "create", "write", "analyze", "compare", "find", "list", "fix",
    "install", "show", "research", "explain", "summarize", "check", "make",
)

#: Araç/donanım izleri — OS veya dosya sistemi erişimi gerekir.
TOOL_WORDS: Tuple[str, ...] = (
    "dosya", "klasör", "klasor", "dizin", "sistem", "gpu", "cpu", "ram",
    "disk", "kurulu", "donanım", "donanim", "işlemci", "islemci", "sürücü",
    "surucu", "internet", "web", "arama", "kod", "script", "betik", "terminal",
    "powershell", "kabuk", "shell", "işlem", "islem", "süreç", "surec", "registry",
    "file", "folder", "directory", "system", "driver", "process", "network",
)

#: Dosya işlemi izleri → "os.*" yetenekleri.
FILE_TOOL_WORDS: Tuple[str, ...] = (
    "dosya", "klasör", "klasor", "dizin", "oku", "okur", "yaz", "kaydet",
    "listele", "içeriğini", "icerigini", "file", "folder", "directory", "read",
    "write", "save", "list",
)

#: Sistem raporu izleri → "os.system_report" yeteneği.
SYSTEM_TOOL_WORDS: Tuple[str, ...] = (
    "sistem", "cpu", "gpu", "ram", "disk", "kurulu", "donanım", "donanim",
    "işlemci", "islemci", "sürücü", "surucu", "termal", "sıcaklık", "sicaklik",
    "bellek", "envanter", "system", "hardware", "memory", "inventory", "thermal",
)

#: Hesap izleri → "compute" türü.
COMPUTE_WORDS: Tuple[str, ...] = (
    "hesapla", "hesap", "matematik", "istatistik", "ortalama", "toplam", "yüzde",
    "yuzde", "çarp", "carp", "böl", "bol", "karekök", "karekok", "denklem",
    "formül", "formul", "olasılık", "olasilik", "varyans", "sapma", "medyan",
    "calculate", "compute", "math", "statistics", "average", "mean", "sum",
    "percent", "probability", "variance", "equation",
)

#: Araştırma izleri → "research" türü.
RESEARCH_WORDS: Tuple[str, ...] = (
    "araştır", "arastir", "research", "incele", "keşfet", "kesfet", "survey",
    "literatür", "literatur", "karşılaştırmalı", "derinlemesine", "explore",
    "investigate", "derin araştırma", "deep dive",
)

#: Kasa referansı izleri → önce getir (retrieve) gerekir.
VAULT_WORDS: Tuple[str, ...] = (
    "not", "notlar", "bilgi", "hatırla", "hatirla", "öğren", "ogren", "kasa",
    "vault", "obsidian", "kayıt", "kayit", "belge", "doküman", "dokuman",
    "remember", "recall", "knowledge", "note", "notes", "learn",
)

#: Kod görevi izleri.
CODE_WORDS: Tuple[str, ...] = (
    "kod", "script", "betik", "fonksiyon", "hata", "bug", "debug", "derle",
    "sınıf", "sinif", "modül", "modul", "api", "kodla", "refactor", "test",
    "code", "function", "error", "compile", "class", "module",
)

#: Karşılaştırma izleri.
COMPARISON_WORDS: Tuple[str, ...] = (
    "karşılaştır", "karsilastir", "kıyasla", "kiyasla", "fark", "farkı",
    "farki", "vs", "versus", "hangisi daha", "compare", "difference", "better",
)

#: Nasıl-yapılır izleri.
HOWTO_WORDS: Tuple[str, ...] = (
    "nasıl", "nasil", "how to", "adımlar", "adimlar", "yöntem", "yontem",
    "kılavuz", "kilavuz", "rehber", "tutorial", "setup", "kurulum",
)

#: Selamlama / sohbet izleri.
GREETING_WORDS: Tuple[str, ...] = (
    "merhaba", "selam", "selamlar", "günaydın", "gunaydin", "iyi akşamlar",
    "iyi aksamlar", "nasılsın", "nasilsin", "teşekkür", "tesekkur", "sağol",
    "sagol", "hello", "hi ", "hey", "thanks", "thank you", "how are you", "good morning",
)

#: Alt hedef ayırıcı bağlaçlar.
CONNECTOR_PATTERNS: Tuple[str, ...] = (
    r"\s+ve\s+", r"\s+sonra\s+", r"\s+ardından\s+", r"\s+ardindan\s+",
    r"\s+ayrıca\s+", r"\s+ayrica\s+", r"\s+then\s+", r"\s+and\s+",
    r"\s*;\s*", r"\s*,\s*",
)

#: Türkçe ayırt edici karakterler.
_TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
_TURKISH_WORDS = {
    "ve", "bir", "bu", "için", "icin", "ile", "neden", "nasıl", "nasil", "mi",
    "mı", "değil", "degil", "var", "yok", "çok", "cok", "daha", "olarak", "gibi",
    "ama", "ancak", "veya", "her", "şu", "su", "benim", "senin", "göster",
    "goster", "yap", "et", "olan", "olduğu", "oldugu",
}
_ENGLISH_WORDS = {
    "the", "and", "why", "how", "what", "is", "are", "of", "to", "in", "for",
    "with", "my", "does", "do", "can", "should", "compare", "please", "show",
    "system", "file",
}


# ─── Sınıflandırma Sonucu ───────────────────────────────────────────────────

@dataclass
class Classification:
    """
    ``Decomposer.classify()`` çıktısının yapılandırılmış hâli.

    ``to_dict()`` ile düz sözlüğe çevrilir; şablonlar bu sözlüğü kullanır.
    """

    intent: str = "unknown"
    complexity: str = "simple"
    needs_research: bool = False
    needs_compute: bool = False
    needs_tools: List[str] = field(default_factory=list)
    estimated_steps: int = 3
    language: str = "unknown"
    keywords: List[str] = field(default_factory=list)
    word_count: int = 0
    clause_count: int = 1
    connector_count: int = 0
    subgoal_count: int = 1
    template: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Sınıflandırmayı JSON-serileştirilebilir sözlüğe çevirir."""
        return {
            "intent": self.intent,
            "complexity": self.complexity,
            "needs_research": self.needs_research,
            "needs_compute": self.needs_compute,
            "needs_tools": list(self.needs_tools),
            "estimated_steps": self.estimated_steps,
            "language": self.language,
            "keywords": list(self.keywords),
            "word_count": self.word_count,
            "clause_count": self.clause_count,
            "connector_count": self.connector_count,
            "subgoal_count": self.subgoal_count,
            "template": self.template,
        }


# ─── Şablon Kaydı ───────────────────────────────────────────────────────────

@dataclass
class _Template:
    """Kayıtlı bir ayrıştırma şablonu."""

    name: str
    matcher: Callable[[str, Dict[str, Any]], bool]
    builder: TemplateBuilder
    description: str = ""
    #: Şablon alt hedefleri kendi içinde zaten böldüyse özyineleme atlanır.
    handles_subgoals: bool = False
    hits: int = 0


# ─── Ana Ayrıştırıcı ────────────────────────────────────────────────────────

class Decomposer:
    """
    Kural/tabanca tabanlı hedef ayrıştırıcı — **LLM kullanmaz**.

    Kullanım:
        d = Decomposer()
        d.initialize(config, event_bus)
        graph = d.decompose("GPU sıcaklığım neden yüksek?")
        print(graph.to_markdown())
    """

    name = "planner.decomposer"

    def __init__(self) -> None:
        self._config: Any = None
        self._event_bus: Any = None
        self._initialized: bool = False
        self._templates: Dict[str, _Template] = {}
        self._max_depth: int = 10
        self._max_reasoning_steps: int = 50
        self._enable_parallel: bool = True
        self._stats: Dict[str, Any] = {
            "decompositions": 0,
            "classifications": 0,
            "recursive_expansions": 0,
            "cap_hits": 0,
            "tasks_created": 0,
            "errors": 0,
            "template_hits": {},
        }
        self._last_classification: Dict[str, Any] = {}
        self._started_at: float = 0.0

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Ayrıştırıcıyı yapılandırır (idempotent).

        Args:
            config: ``SystemConfig`` benzeri nesne (None olabilir)
            event_bus: Olay yolu (``emit`` metodu beklenir)
        """
        if self._initialized:
            if config is not None:
                self._config = config
                self._read_config()
            if event_bus is not None:
                self._event_bus = event_bus
            return

        self._config = config
        self._event_bus = event_bus
        self._read_config()
        self._register_builtin_templates()
        self._initialized = True
        self._started_at = time.time()

        logger.info(
            f"Kural tabanlı ayrıştırıcı hazır — {len(self._templates)} şablon, "
            f"maks. derinlik {self._max_depth}"
        )

    def shutdown(self) -> None:
        """Ayrıştırıcıyı kapatır (idempotent)."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info(
            f"Ayrıştırıcı kapatıldı — toplam {self._stats['decompositions']} ayrıştırma"
        )

    def _read_config(self) -> None:
        """Konfigürasyondan planlayıcı ayarlarını okur (yoksa varsayılan)."""
        planner = getattr(self._config, "planner", None)
        self._max_depth = int(getattr(planner, "max_task_depth", 10) or 10)
        self._max_reasoning_steps = int(getattr(planner, "max_reasoning_steps", 50) or 50)
        self._enable_parallel = bool(getattr(planner, "enable_parallel_tasks", True))

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Olay yayınlar (event_bus yoksa sessizce atlar)."""
        if self._event_bus:
            try:
                self._event_bus.emit(f"planner.{event}", data, source=self.name)
            except Exception as exc:  # pragma: no cover - savunmacı
                logger.debug(f"Olay yayınlanamadı ({event}): {exc}")

    # ─── Sınıflandırma ──────────────────────────────────────────────────

    def classify(self, goal: str) -> Dict[str, Any]:
        """
        Hedefi niyet/karmaşıklık/gereksinim açısından çözümler.

        Returns:
            ``{"intent", "complexity", "needs_research", "needs_compute",
            "needs_tools", "estimated_steps", "language", "keywords", ...}``
        """
        text = (goal or "").strip()
        low = text.lower()
        result = Classification()

        if not text:
            result.intent = "empty"
            result.complexity = "trivial"
            result.template = "fallback"
            self._stats["classifications"] += 1
            self._last_classification = result.to_dict()
            return result.to_dict()

        result.language = self._detect_language(text)
        words = re.findall(r"\S+", low)
        result.word_count = len(words)
        result.clause_count = max(
            1,
            len(re.split(r"[.!?;]+|\s(?:ve|sonra|ardından|ayrıca|then|and)\s", low)),
        )
        result.connector_count = sum(
            1 for pattern in CONNECTOR_PATTERNS if re.search(pattern, low)
        )
        subgoals = self.subgoals(text)
        result.subgoal_count = len(subgoals)

        has_interrogative = any(token in low for token in INTERROGATIVES)
        has_imperative = any(token in low for token in IMPERATIVES)
        needs_tools = self._match_words(low, TOOL_WORDS)
        compute_hits = self._match_words(low, COMPUTE_WORDS)
        research_hits = self._match_words(low, RESEARCH_WORDS)
        vault_hits = self._match_words(low, VAULT_WORDS)
        code_hits = self._match_words(low, CODE_WORDS)
        compare_hits = self._match_words(low, COMPARISON_WORDS)
        howto_hits = self._match_words(low, HOWTO_WORDS)
        greeting_hits = self._match_words(low, GREETING_WORDS)

        result.needs_tools = needs_tools
        result.needs_compute = bool(compute_hits) or bool(re.search(r"\d+\s*[-+*/^]\s*\d+", low))
        result.needs_research = bool(research_hits)

        # ─── Niyet Belirleme (öncelik sırası önemli) ────────────────────
        if greeting_hits and result.word_count <= 6:
            result.intent = "smalltalk"
        elif compare_hits:
            result.intent = "comparison"
        elif code_hits:
            result.intent = "code_task"
        elif result.needs_compute and not has_interrogative:
            result.intent = "compute"
        elif needs_tools and any(w in low for w in SYSTEM_TOOL_WORDS):
            result.intent = "system"
        elif any(w in low for w in FILE_TOOL_WORDS) and has_imperative:
            result.intent = "file_operation"
        elif result.needs_research:
            result.intent = "research"
        elif howto_hits:
            result.intent = "howto"
        elif has_interrogative and (vault_hits or result.word_count >= 3):
            result.intent = "factual_question"
        elif result.needs_compute:
            result.intent = "compute"
        elif needs_tools:
            result.intent = "file_operation"
        elif has_imperative:
            result.intent = "howto"
        else:
            result.intent = "general"

        # ─── Karmaşıklık ────────────────────────────────────────────────
        score = 0
        if result.word_count > 6:
            score += 1
        if result.word_count > 14:
            score += 1
        if result.word_count > 26:
            score += 1
        if result.clause_count > 1:
            score += 1
        if result.clause_count > 2:
            score += 1
        if result.connector_count >= 2:
            score += 1
        if result.subgoal_count > 1:
            score += 1
        if result.needs_research:
            score += 1
        if result.needs_compute:
            score += 1
        if len(needs_tools) >= 2:
            score += 1
        if result.intent in ("research", "code_task", "comparison"):
            score += 1

        if result.word_count <= 3 and result.subgoal_count == 1 and score <= 1:
            result.complexity = "trivial"
        elif score <= 1:
            result.complexity = "trivial" if result.word_count <= 3 else "simple"
        elif score <= 3:
            result.complexity = "simple"
        elif score <= 6:
            result.complexity = "moderate"
        else:
            result.complexity = "complex"

        # ─── Anahtar Kelimeler ──────────────────────────────────────────
        result.keywords = self._keywords(low)

        # ─── Tahmini Adım Sayısı ────────────────────────────────────────
        base = {"trivial": 2, "simple": 4, "moderate": 6, "complex": 9}.get(
            result.complexity, 4
        )
        result.estimated_steps = int(
            min(
                self._max_reasoning_steps,
                base
                + (2 if result.needs_research else 0)
                + (1 if result.needs_compute else 0)
                + (1 if result.needs_tools else 0)
                + max(0, result.subgoal_count - 1),
            )
        )

        result.template = self._select_template(text, result.to_dict())

        payload = result.to_dict()
        self._stats["classifications"] += 1
        self._last_classification = payload

        logger.debug(
            f"Sınıflandırma: niyet={payload['intent']}, karmaşıklık={payload['complexity']}, "
            f"dil={payload['language']}, adım≈{payload['estimated_steps']}, "
            f"şablon={payload['template']}"
        )
        return payload

    # ─── Alt Hedefler ───────────────────────────────────────────────────

    def subgoals(self, goal: str) -> List[str]:
        """
        Hedefi bağlaçlara göre alt hedeflere böler.

        Ayırıcılar: " ve ", " sonra ", " ardından ", " ayrıca ", " then ",
        " and ", virgül, noktalı virgül. 3 karakterden kısa parçalar atılır,
        en fazla 6 parça döner.
        """
        text = (goal or "").strip()
        if not text:
            return []

        pieces = [text]
        for pattern in CONNECTOR_PATTERNS:
            new_pieces: List[str] = []
            for piece in pieces:
                new_pieces.extend(re.split(pattern, piece))
            pieces = new_pieces

        cleaned: List[str] = []
        for piece in pieces:
            fragment = piece.strip(" \t\n\r.,;:!?·-—")
            if len(fragment) < 3:
                continue
            if fragment.lower() in cleaned:
                continue
            cleaned.append(fragment)

        return cleaned[:MAX_SUBGOALS]

    # ─── Şablon Yönetimi ────────────────────────────────────────────────

    def register_template(self, name: str, matcher: Any, builder: Any) -> None:
        """
        Yeni bir ayrıştırma şablonu kaydeder.

        Args:
            name: Şablon adı (``templates()`` çıktısında görünür)
            matcher: ``(goal, classification) -> bool`` ya da derlenmiş regex
            builder: ``(goal, classification, context, depth) -> TaskGraph``

        Raises:
            PlanError: Ad veya çağrılabilirler geçersizse.
        """
        if not name or not isinstance(name, str):
            raise PlanError("şablon adı boş olamaz")

        if isinstance(matcher, re.Pattern):
            pattern = matcher

            def compiled_matcher(goal: str, _classification: Dict[str, Any]) -> bool:
                return bool(pattern.search((goal or "").lower()))
            match_fn: Callable[[str, Dict[str, Any]], bool] = compiled_matcher
        elif callable(matcher):
            def callable_matcher(goal: str, classification: Dict[str, Any]) -> bool:
                try:
                    return bool(matcher(goal, classification))
                except TypeError:
                    return bool(matcher(goal))
            match_fn = callable_matcher
        else:
            raise PlanError(f"geçersiz matcher türü: {type(matcher).__name__}")

        if not callable(builder):
            raise PlanError(f"geçersiz builder türü: {type(builder).__name__}")

        self._templates[name] = _Template(
            name=name,
            matcher=match_fn,
            builder=builder,
            description=f"kullanıcı şablonu ({getattr(builder, '__name__', 'anonim')})",
        )
        logger.info(f"Şablon kaydedildi: '{name}'")
        self._emit("template_registered", {"template": name})

    def templates(self) -> List[str]:
        """Kayıtlı şablon adlarını sırayla döndürür."""
        return list(self._templates.keys())

    def template_info(self) -> List[Dict[str, Any]]:
        """Şablonların ad/açıklama/istatistik listesi."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "handles_subgoals": t.handles_subgoals,
                "hits": t.hits,
            }
            for t in self._templates.values()
        ]

    # ─── Ayrıştırma ─────────────────────────────────────────────────────

    def decompose(self, goal: str, context: Optional[Dict[str, Any]] = None,
                  depth: int = 0, max_depth: Optional[int] = None) -> TaskGraph:
        """
        Hedefi görev grafiğine çevirir.

        Args:
            goal: Doğal dil hedefi
            context: Ek bağlam (``max_depth_override``, ``budget`` gibi anahtarlar)
            depth: Özyineleme derinliği (0 = kök)
            max_depth: İzin verilen en fazla derinlik (None → konfigürasyon)

        Returns:
            Yürütmeye hazır ``TaskGraph`` (asla None; boş hedefte tek görev üretir).
        """
        if not self._initialized:
            self.initialize()

        ctx: Dict[str, Any] = dict(context or {})
        limit = int(max_depth if max_depth is not None else self._max_depth)
        limit = max(1, limit)
        text = (goal or "").strip() or "boş hedef"

        classification = self.classify(text)
        template_name = classification.get("template") or "fallback"
        template = self._templates.get(template_name) or self._templates["fallback"]

        graph = TaskGraph(goal=text)
        ctx["_graph"] = graph
        ctx["_classification"] = classification
        ctx["_depth"] = depth
        ctx["_max_depth"] = limit
        # Bütçe: alt ayrıştırmalarla **paylaşılan** sayaç (ad alanı başına ayrı
        # sözlük verilirse o sözlük kullanılır — toplam yine sınırlıdır).
        if "_budget" not in ctx:
            ctx["_budget"] = {"used": int(ctx.get("_budget_used", 0)),
                              "limit": int(ctx.get("_budget_limit", MAX_TOTAL_TASKS))}
        ctx["_budget_limit"] = int(ctx["_budget"]["limit"])
        namespace = str(ctx.get("_ns", ""))
        if not callable(ctx.get("_id_factory")):
            ctx["_id_factory"] = make_id_factory(prefix=f"{namespace}t")

        try:
            template.builder(text, classification, ctx, depth)
        except PlanError:
            raise
        except Exception as exc:  # pragma: no cover - savunmacı
            self._stats["errors"] += 1
            logger.error(f"Şablon '{template_name}' başarısız: {exc} — yedek şablona düşülüyor")
            graph = TaskGraph(goal=text)
            ctx["_graph"] = graph
            ctx["_budget"]["used"] = max(0, ctx["_budget"]["used"] - len(graph.tasks))
            ctx["_id_factory"] = make_id_factory(prefix=f"{namespace}t")
            self._templates["fallback"].builder(text, classification, ctx, depth)

        template.hits += 1
        self._stats["template_hits"][template.name] = (
            self._stats["template_hits"].get(template.name, 0) + 1
        )

        # ─── Özyinelemeli Genişletme ────────────────────────────────────
        if (
            classification.get("complexity") == "complex"
            and depth < limit
            and not template.handles_subgoals
        ):
            self._expand_subgoals(graph, text, classification, ctx, depth, limit)

        # ─── Sentez Görevi Garantisi ────────────────────────────────────
        self._make_synthesizer(graph, text, ctx)

        self._stats["decompositions"] += 1
        self._stats["tasks_created"] += len(graph)
        self._emit(
            "plan_created",
            {
                "goal": text[:200],
                "template": template.name,
                "tasks": len(graph),
                "intent": classification.get("intent"),
                "complexity": classification.get("complexity"),
                "depth": depth,
            },
        )
        logger.info(
            f"Plan üretildi: '{text[:70]}' → {template.name} şablonu, "
            f"{len(graph)} görev, {len(graph.parallel_batches())} dalga"
        )
        return graph

    def estimate_weight(self, task: Task) -> float:
        """
        Kritik yol hesabı için görev ağırlığı (tahmini süre vekili).

        Tür bazlı taban maliyet × öncelik × derinlik × ücret çarpanı.
        """
        base_cost = {
            TaskKind.RETRIEVE: 1.0,
            TaskKind.REASON: 1.4,
            TaskKind.COMPUTE: 1.6,
            TaskKind.TOOL: 1.8,
            TaskKind.RESEARCH: 2.4,
            TaskKind.VERIFY: 1.3,
            TaskKind.SYNTHESIZE: 2.0,
            TaskKind.MEMORIZE: 0.5,
            TaskKind.DELEGATE: 2.2,
        }.get(task.kind, 1.0)

        weight = base_cost
        weight *= 1.0 + (task.priority / 10.0)
        weight *= 1.0 + (task.depth / 10.0)
        if task.tool:
            weight *= 1.15
        if task.critical:
            weight *= 1.2
        if task.max_attempts > 1:
            weight *= 1.0 + (task.max_attempts - 1) * 0.25
        return round(weight, 4)

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir ayrıştırıcı metrikleri."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "templates": len(self._templates),
            "template_hits": dict(self._stats["template_hits"]),
            "decompositions": self._stats["decompositions"],
            "classifications": self._stats["classifications"],
            "recursive_expansions": self._stats["recursive_expansions"],
            "cap_hits": self._stats["cap_hits"],
            "tasks_created": self._stats["tasks_created"],
            "errors": self._stats["errors"],
            "avg_tasks_per_plan": round(
                self._stats["tasks_created"] / self._stats["decompositions"], 2
            ) if self._stats["decompositions"] else 0.0,
            "max_depth": self._max_depth,
            "enable_parallel_tasks": self._enable_parallel,
            "last_classification": dict(self._last_classification),
            "uptime_s": round(time.time() - self._started_at, 2) if self._started_at else 0.0,
        }

    # ════════════════════════════════════════════════════════════════════
    #  ŞABLON MOTORU — Dahili
    # ════════════════════════════════════════════════════════════════════

    def _match_words(self, low: str, words: Sequence[str]) -> List[str]:
        """Metinde geçen sözcükleri döndürür."""
        return [w for w in words if w in low]

    def _detect_language(self, text: str) -> str:
        """Kaba dil algılama: tr | en | unknown."""
        low = text.lower()
        has_turkish_char = any(ch in _TURKISH_CHARS for ch in text)
        tokens = set(re.findall(r"[a-zA-Zçğıöşü]+", low))
        tr_score = len(tokens & _TURKISH_WORDS) + (2 if has_turkish_char else 0)
        en_score = len(tokens & _ENGLISH_WORDS)

        tr_markers = ("neden", "nasıl", "nasil", "için", "icin", "bir", "mı", "mi",
                      "değil", "degil", "yüksek", "yuksek", "göster", "goster")
        tr_score += sum(1 for m in tr_markers if m in low)
        en_markers = ("why", "how", "what", "the ", " is ", " are ", "my ")
        en_score += sum(1 for m in en_markers if m in low)

        if tr_score == 0 and en_score == 0:
            return "unknown"
        return "tr" if tr_score >= en_score else "en"

    def _keywords(self, low: str) -> List[str]:
        """Anlamlı anahtar kelimeleri (durak sözcükler ayıklanmış) döndürür."""
        stopwords = {
            "ve", "ile", "için", "icin", "bir", "bu", "şu", "su", "mi", "mı", "mu",
            "mü", "ne", "nasıl", "nasil", "neden", "the", "and", "for", "with",
            "that", "this", "from", "how", "why", "what", "are", "is", "my", "our",
            "bir", "olarak", "daha", "çok", "cok", "ama", "ancak", "veya", "gibi",
        }
        tokens = re.findall(r"[a-zA-Zçğıöşü0-9][a-zA-Zçğıöşü0-9+.\-]*", low)
        keywords: List[str] = []
        for token in tokens:
            word = token.strip(".-")
            if len(word) < 3 or word in stopwords:
                continue
            if word not in keywords:
                keywords.append(word)
        return keywords[:12]

    # ─── Dahili Görev Yardımcıları ──────────────────────────────────────

    def _spawn(self, ctx: Dict[str, Any], name: str, kind: TaskKind,
               tool: str = "", args: Optional[Dict[str, Any]] = None,
               critical: bool = False, priority: int = 5,
               description: str = "", force_id: str = "",
               depth_boost: int = 0) -> Task:
        """
        Grafiğe yeni görev ekler (bütçe denetimli, kimlik üretimi otomatik).

        Raises:
            PlanError: Görev bütçesi (``MAX_TOTAL_TASKS``) dolduysa.
        """
        graph: TaskGraph = ctx["_graph"]
        budget: Dict[str, int] = ctx["_budget"]
        if budget["used"] >= budget["limit"]:
            self._stats["cap_hits"] += 1
            logger.warning(
                f"⚠ Görev bütçesi doldu ({budget['limit']}) — kalan alt görevler eklenmedi "
                f"(hedef: '{str(graph.goal)[:60]}')"
            )
            raise PlanError(f"görev bütçesi doldu: {budget['limit']}", {"limit": budget["limit"]})

        task_id = force_id or (ctx["_id_factory"]() if callable(ctx.get("_id_factory")) else "")
        task = Task(
            id=task_id,
            name=name,
            kind=kind,
            description=description or name,
            tool=tool,
            args=dict(args or {}),
            critical=critical,
            priority=priority,
            depth=int(ctx.get("_depth", 0)) + int(depth_boost),
            tags={kind.value, "planner"},
        )
        graph.add(task)
        budget["used"] += 1
        ctx["_budget_used"] = budget["used"]
        return task

    @staticmethod
    def _attach(previous: Optional[Task], task: Task) -> Task:
        """``previous`` → ``task`` bağımlılığını kurar (kimlik vererek)."""
        if previous is not None and previous.id not in task.depends_on:
            task.depends_on.append(previous.id)
            if task.id not in previous.children:
                previous.children.append(task.id)
        return task

    @staticmethod
    def _bind(graph: TaskGraph, task: Task, dependencies: Sequence[Task]) -> Task:
        """
        Görevin bağımlılıklarını verilen görevlerle **eşitler** ve grafı eşitler.

        ``TaskGraph.set_dependencies`` kullanıldığı için ``self._g`` indeksi de
        güncellenir — sıralama, dalga ve kritik yol hesapları tutarlı kalır.
        """
        ids = list(task.depends_on)
        for dep in dependencies:
            if dep.id not in ids and dep.id != task.id:
                ids.append(dep.id)
        graph.set_dependencies(task.id, ids)
        return task

    @staticmethod
    def _mark_optional(graph: TaskGraph, task: Task, optional_ids: Sequence[str]) -> None:
        """
        Bağımlılıkları "yumuşak" işaretler (``args._optional_deps``).

        Yürütücü, yumuşak bağımlılıklar sonlanmış (DONE/FAILED/SKIPPED/CANCELLED)
        olduğunda görevi **hazır** sayar ve bağlamdaki eksik değerlerle çalıştırır.
        Böylece çevrimdışı/engelli kanatlar planın geri kalanını kilitlemez.
        """
        ids = [str(i) for i in optional_ids if i in graph.tasks]
        if ids:
            task.args["_optional_deps"] = ids

    def _chain(self, graph: TaskGraph, tasks: Sequence[Task]) -> Task:
        """Görevleri sıraya dizer (her biri öncekine bağlı) ve graf indeksini eşitler."""
        previous: Optional[Task] = None
        for task in tasks:
            if previous is not None:
                self._bind(graph, task, [previous])
            previous = task
        return tasks[-1] if tasks else None  # type: ignore[return-value]

    def _fan_out(self, graph: TaskGraph, sources: Sequence[Task], target: Task) -> Task:
        """Birden çok görevi tek hedefe bağlar (paralel kanatları birleştirir)."""
        self._bind(graph, target, list(sources))
        return target

    def _wire(self, graph: TaskGraph, source: Task, target: Task) -> Task:
        """Tek bir bağımlılık kenarı kurar (graf indeksi dahil)."""
        self._bind(graph, target, [source])
        return target

    @staticmethod
    def _leaf_tasks(graph: TaskGraph) -> List[Task]:
        """Grafın mevcut yaprak görevleri."""
        return [graph.tasks[tid] for tid in graph.leaves()]

    def _terminals(self, graph: TaskGraph) -> List[Task]:
        """Diğer görevlerin bağımlı olduğu uç görevler (sentez için bağlanacak)."""
        referenced: Set[str] = set()
        for task in graph.tasks.values():
            referenced.update(task.depends_on)
        terminals = [
            t for t in graph.tasks.values()
            if t.id not in referenced and t.kind != TaskKind.SYNTHESIZE
        ]
        if not terminals:
            terminals = self._leaf_tasks(graph)
        return [t for t in terminals if t.kind != TaskKind.SYNTHESIZE] or self._leaf_tasks(graph)

    def _spec(self, text: str) -> str:
        """Görev argümanları için hedef metnini kısaltır."""
        return text if len(text) <= 240 else text[:237] + "…"

    # ─── Dahili: Sentez ve Genişletme ───────────────────────────────────

    def _make_synthesizer(self, graph: TaskGraph, goal: str, ctx: Dict[str, Any]) -> Optional[Task]:
        """
        Grafda ``SYNTHESIZE`` görevi yoksa sonuna ekler.

        Sentez görevi her zaman ``critical``'dır: yanıt üretilemezse
        plan başarısız sayılır. Üst düzey grafta kimliği ``S``'tir; alt
        graflarda ad alanı öneki alır (``e1.t5``).
        """
        if any(t.kind == TaskKind.SYNTHESIZE for t in graph.tasks.values()):
            return None

        try:
            synth = self._spawn(
                ctx,
                name="Nihai yanıtı sentezle",
                kind=TaskKind.SYNTHESIZE,
                tool="synthesize",
                args={"goal": "${context.goal}", "context": "${context}"},
                critical=True,
                priority=0,
                description=f"Hedefe yönelik nihai yanıtı üret: {self._spec(goal)}",
                force_id="S",
                depth_boost=1,
            )
        except PlanError:
            logger.warning("⚠ Sentez görevi eklenemedi (görev bütçesi doldu)")
            return None

        terminals = [t for t in self._terminals(graph) if t.id != synth.id]
        if terminals:
            self._fan_out(graph, terminals, synth)
        return synth

    def _expand_subgoals(self, graph: TaskGraph, goal: str, classification: Dict[str, Any],
                         ctx: Dict[str, Any], depth: int, max_depth: int) -> None:
        """
        Karmaşık hedefi alt hedeflere böler ve her birini alt plan olarak ekler.

        Her alt hedef:
            * **kendi ad alanında** ayrıştırılır (``t1`` → ``t1.1``, ``t1.2``, …),
            * grafın mevcut yapraklarına bağlanır (paralel kanat olarak),
            * çıkış görevleri sentez görevine bağlanır.

        Toplam görev sayısı ``MAX_TOTAL_TASKS`` ile sınırlıdır; sınır aşılırsa
        Türkçe uyarı loglanır ve kalan alt hedefler eklenmez.
        """
        subgoals = self.subgoals(goal)
        if len(subgoals) < 2:
            return

        created = 0
        for index, subgoal in enumerate(subgoals[:MAX_EXPANSION_BRANCHES]):
            budget = ctx["_budget"]
            if budget["used"] >= budget["limit"]:
                self._stats["cap_hits"] += 1
                logger.warning(
                    f"⚠ Görev bütçesi doldu ({budget['limit']}) — "
                    f"{len(subgoals) - index} alt hedef eklenemedi"
                )
                break

            text = subgoal if len(subgoal) >= 3 else f"{subgoal} hakkında bilgi"
            try:
                sub_graph = self._expand_one(graph, text, ctx, depth, max_depth, index)
            except PlanError as exc:
                logger.warning(f"⚠ Alt hedef genişletilemedi ('{subgoal[:40]}'): {exc}")
                continue
            except Exception as exc:  # pragma: no cover - savunmacı
                self._stats["errors"] += 1
                logger.warning(f"⚠ Alt hedef genişletme hatası ('{subgoal[:40]}'): {exc}")
                continue

            if sub_graph is None or not sub_graph.tasks:
                continue

            added = self._splice(graph, sub_graph, depth_offset=1)
            outputs = self._wire_subgraph(graph, sub_graph, added, index)
            created += 1
            self._stats["recursive_expansions"] += 1
            logger.debug(f"Alt hedef genişletildi: '{subgoal[:50]}' → {len(added)} görev")

        if created:
            self._emit(
                "subgoals_expanded",
                {
                    "goal": goal[:160],
                    "subgoals": [s[:80] for s in subgoals],
                    "branches": created,
                    "total_tasks": len(graph),
                    "depth": depth,
                },
            )
            logger.info(
                f"Karmaşık hedef {created} alt plana bölündü — "
                f"toplam {len(graph)} görev (derinlik {depth} → {depth + 1})"
            )
        elif int(ctx["_budget"]["used"]) >= int(ctx["_budget"]["limit"]):
            logger.warning(
                f"⚠ Görev bütçesi doldu ({ctx['_budget']['limit']}): "
                f"'{goal[:50]}' hedefi alt planlara bölünemedi"
            )

    def _expand_one(self, graph: TaskGraph, subgoal: str, ctx: Dict[str, Any],
                    depth: int, max_depth: int, index: int) -> Optional[TaskGraph]:
        """
        Tek bir alt hedefi kendi ad alanında ayrıştırır.

        Alt graf kimlikleri ``e1``, ``e2``, … öneki alır (``t1`` → ``e1.t1``),
        böylece üst grafla çakışma olmaz ve ağaç görünümü okunabilir kalır.
        """
        child = self._child_context(ctx, index)
        return self.decompose(
            subgoal,
            context=child,
            depth=depth,
            max_depth=max_depth,
        )

    def _child_context(self, ctx: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Alt ayrıştırma için ad alanlı, **bütçesi paylaşılan** bağlam üretir.

        Bütçe sözlüğü üst bağlamla **aynı nesnedir**; böylece tüm alt planlar
        toplamda ``MAX_TOTAL_TASKS`` sınırını paylaşır. Kimlik üreticisi ad
        alanına göre ayrı sayılır (``e1.t1``, ``e1.t2``, …).
        """
        namespace = f"e{index + 1}."
        counter = {"n": 0}

        def _id_factory() -> str:
            counter["n"] += 1
            return f"{namespace}t{counter['n']}"

        return {
            "max_depth_override": ctx.get("_max_depth"),
            "_budget": ctx["_budget"],
            "_budget_limit": int(ctx["_budget"]["limit"]),
            "_ns": namespace,
            "_id_factory": _id_factory,
            "_parent_anchor": namespace.rstrip("."),
        }

    def _wire_subgraph(self, graph: TaskGraph, sub_graph: TaskGraph,
                       added: Sequence[str], index: int) -> List[str]:
        """
        Yeni eklenen alt grafı üst grafa bağlar.

        Yön **üst → alt**'tır: ana hattın bilgi toplayan görevleri, alt grafin
        girişlerinin bağımlılıkları yapılır — böylece alt plan toplanmış bağlamı
        devralır. Çıkışlar (uç görevler) nihai sentez görevine bağlanır.

        Returns:
            Alt grafin çıkış (uç) görev kimlikleri.
        """
        if not added:
            return []

        sub_ids = set(added)
        roots = [
            tid for tid in added
            if not any(dep in sub_ids for dep in graph.dependencies_of(tid))
        ]
        leaves = [
            tid for tid in added
            if not any(other in sub_ids for other in graph.dependents_of(tid))
        ]

        # Girişler: ana hattın bilgi toplayan görevlerine bağlanır
        feeders = self._parent_feeders(graph, sub_ids)
        for root_id in roots:
            root = graph.tasks[root_id]
            merged = [f for f in feeders if f not in root.depends_on] + list(root.depends_on)
            graph.set_dependencies(root_id, merged)

        # Çıkışlar: üst graftaki sentez görevi
        synthesizer = next(
            (t.id for t in graph.tasks.values()
             if t.kind == TaskKind.SYNTHESIZE and t.id not in sub_ids),
            None,
        )
        if synthesizer and leaves:
            synth = graph.tasks[synthesizer]
            deps = [d for d in synth.depends_on if d not in sub_ids]
            graph.set_dependencies(synthesizer, deps + leaves)

        logger.debug(
            f"Alt graf bağlandı [{index + 1}]: {len(roots)} giriş, "
            f"{len(leaves)} çıkış, {len(feeders)} besleyici"
        )
        return leaves

    @staticmethod
    def _parent_feeders(graph: TaskGraph, sub_ids: Set[str]) -> List[str]:
        """
        Alt grafı besleyecek **üst düzey** bilgi toplama görevlerini bulur.

        Yalnızca ana hattın görevleri döner (kimliğinde ``.`` bulunmayanlar);
        böylece genişletmeler birbirine çapraz bağlanmaz.
        """
        feeders: List[str] = []
        for task in graph.tasks.values():
            if task.id in sub_ids or "." in task.id:
                continue
            if task.kind in (TaskKind.RETRIEVE, TaskKind.RESEARCH):
                feeders.append(task.id)
        return feeders

    def _splice(self, target: TaskGraph, source: TaskGraph,
                depth_offset: int = 1) -> List[str]:
        """
        ``source`` grafiğini ``target`` grafına ekler.

        Kaynak kimlikler zaten ad alanlıdır (``e1.t1``); yalnızca çakışma
        durumunda sona sayaç eklenir. Görevler **sıfırlanmış** durumda kopyalanır
        (sonuç/hata/deneme temizlenir, ağaç derinliği ``depth_offset`` kadar
        artar, ``parent_id``/``children`` bağları yeni kimliklere eşlenir).

        Args:
            depth_offset: Kopyalanan görevlerin derinliğine eklenecek değer.

        Returns:
            Eklenen yeni görev kimliklerinin listesi.
        """
        mapping: Dict[str, str] = {}
        for tid in source.tasks:
            new_id = tid
            if new_id in target.tasks:
                suffix = 1
                while f"{tid}_{suffix}" in target.tasks:
                    suffix += 1
                new_id = f"{tid}_{suffix}"
            mapping[tid] = new_id

        added: List[str] = []
        for tid, task in source.tasks.items():
            clone = Task.from_dict(task.to_dict())
            clone.id = mapping[tid]
            clone.depth = int(task.depth) + int(depth_offset)
            clone.parent_id = mapping.get(task.parent_id) if task.parent_id else None
            clone.depends_on = [mapping[d] for d in task.depends_on if d in mapping]
            clone.children = [mapping[c] for c in task.children if c in mapping]
            clone.result = None
            clone.error = ""
            clone.attempts = 0
            clone.state = TaskState.PENDING
            clone.started_at = 0.0
            clone.finished_at = 0.0
            clone.duration_ms = 0.0
            target.add(clone)
            added.append(clone.id)

        return added



    # ════════════════════════════════════════════════════════════════════
    #  YERLEŞİK ŞABLONLAR
    # ════════════════════════════════════════════════════════════════════

    def _register_builtin_templates(self) -> None:
        """Yerleşik şablonları sırayla kaydeder (sıra = öncelik)."""
        self._templates = {}
        builtins: List[Tuple[str, Callable[[str, Dict[str, Any]], bool],
                             TemplateBuilder, str, bool]] = [
            (
                "greeting",
                lambda goal, c: c.get("intent") == "smalltalk",
                self._tpl_greeting,
                "Selamlama/sohbet → tek sentez görevi",
                True,
            ),
            (
                "system",
                lambda goal, c: c.get("intent") == "system",
                self._tpl_system,
                "Sistem/donanım sorgusu → os.system_report + yorum",
                False,
            ),
            (
                "file_operation",
                lambda goal, c: c.get("intent") == "file_operation",
                self._tpl_file_operation,
                "Dosya/klasör işlemi → os.* aracı + yorum",
                False,
            ),
            (
                "code_task",
                lambda goal, c: c.get("intent") == "code_task",
                self._tpl_code_task,
                "Kod/hata ayıklama → tasarım + hesap/araç + doğrulama",
                False,
            ),
            (
                "research",
                lambda goal, c: c.get("intent") == "research"
                or (c.get("needs_research") and c.get("complexity") in ("moderate", "complex")),
                self._tpl_research,
                "Araştırma → getir + N paralel araştırma + doğrula + sentezle + anımsa",
                True,
            ),
            (
                "compute",
                lambda goal, c: c.get("intent") == "compute" or c.get("needs_compute"),
                self._tpl_compute,
                "Hesaplama → formül getir + hesapla + doğrula + sentezle",
                False,
            ),
            (
                "comparison",
                lambda goal, c: c.get("intent") == "comparison",
                self._tpl_comparison,
                "Karşılaştırma → iki paralel getirme + fark çıkarımı + sentez",
                True,
            ),
            (
                "howto",
                lambda goal, c: c.get("intent") == "howto",
                self._tpl_howto,
                "Nasıl-yapılır → getir + adım çıkar + doğrula + sentezle",
                False,
            ),
            (
                "factual_question",
                lambda goal, c: c.get("intent") in ("factual_question", "general"),
                self._tpl_factual,
                "Bilgi sorusu → getir + doğrula + sentezle + anımsa",
                False,
            ),
            (
                "fallback",
                lambda goal, c: True,
                self._tpl_fallback,
                "Genel → getir + akıl yürüt + sentezle",
                False,
            ),
        ]

        for name, matcher, builder, description, handles_subgoals in builtins:
            self._templates[name] = _Template(
                name=name,
                matcher=matcher,
                builder=builder,
                description=description,
                handles_subgoals=handles_subgoals,
            )

    def _select_template(self, goal: str, classification: Dict[str, Any]) -> str:
        """Sınıflandırmaya uyan ilk şablonun adını döndürür."""
        for name, template in self._templates.items():
            try:
                if template.matcher(goal, classification):
                    return name
            except Exception as exc:  # pragma: no cover - savunmacı
                logger.warning(f"⚠ Şablon matcher hatası ('{name}'): {exc}")
        return "fallback"

    # ─── Şablon: Selamlama ──────────────────────────────────────────────

    def _tpl_greeting(self, goal: str, classification: Dict[str, Any],
                      ctx: Dict[str, Any], depth: int) -> TaskGraph:
        """Selamlama: tek sentez görevi (kasa/araç erişimi yok)."""
        graph: TaskGraph = ctx["_graph"]
        self._spawn(
            ctx,
            name="Kısa yanıt üret",
            kind=TaskKind.SYNTHESIZE,
            tool="synthesize",
            args={"goal": "${context.goal}", "context": "${context}"},
            critical=True,
            priority=0,
            description=f"Sohbet/selamlama yanıtı: {self._spec(goal)}",
            force_id="A",
        )
        return graph

    # ─── Şablon: Bilgi Sorusu ───────────────────────────────────────────

    def _tpl_factual(self, goal: str, classification: Dict[str, Any],
                     ctx: Dict[str, Any], depth: int) -> TaskGraph:
        """Bilgi sorusu: getir → doğrula → sentezle → (koşullu) anımsa."""
        graph: TaskGraph = ctx["_graph"]
        keywords = classification.get("keywords", [])[:5]
        retrieve = self._spawn(
            ctx,
            name="Kasadan ilgili notları getir",
            kind=TaskKind.RETRIEVE,
            tool="retrieve",
            args={"query": "${context.goal}", "keywords": keywords, "limit": 10},
            priority=1,
            description=f"Hedefle ilgili notları kasadan ara: {self._spec(goal)}",
        )
        verify = self._spawn(
            ctx,
            name="Getirilen notları çapraz doğrula",
            kind=TaskKind.VERIFY,
            tool="verify",
            args={
                "goal": "${context.goal}",
                "notes": "${%s.result}" % retrieve.id,
                "min_confidence": MEMORIZE_CONFIDENCE_THRESHOLD,
            },
            priority=2,
            description="Notların tutarlılığını ve güncelliğini denetle",
        )
        synthesize = self._spawn(
            ctx,
            name="Yanıtı sentezle",
            kind=TaskKind.SYNTHESIZE,
            tool="synthesize",
            args={"goal": "${context.goal}", "context": "${context}"},
            critical=True,
            priority=0,
            description="Doğrulanmış bilgiden nihai yanıtı üret",
        )
        memorize = self._spawn(
            ctx,
            name="Yanıtı kasaya kaydet",
            kind=TaskKind.MEMORIZE,
            tool="memorize",
            args={
                "goal": "${context.goal}",
                "answer": "${%s.result}" % synthesize.id,
                "min_confidence": MEMORIZE_CONFIDENCE_THRESHOLD,
            },
            priority=8,
            description="Güven eşiği aşılırsa yanıtı yeni not olarak sakla",
        )
        self._chain(graph, [retrieve, verify, synthesize, memorize])
        return graph

    # ─── Şablon: Araştırma ──────────────────────────────────────────────

    def _tpl_research(self, goal: str, classification: Dict[str, Any],
                      ctx: Dict[str, Any], depth: int) -> TaskGraph:
        """Araştırma: getir → N paralel araştırma → doğrula → sentezle → anımsa."""
        graph: TaskGraph = ctx["_graph"]
        keywords = classification.get("keywords", []) or self.subgoals(goal)
        probes = [k for k in keywords if len(k) >= 3][:4] or [goal[:40]]

        retrieve = self._spawn(
            ctx,
            name="Mevcut bilgiyi kasadan getir",
            kind=TaskKind.RETRIEVE,
            tool="retrieve",
            args={"query": "${context.goal}", "keywords": probes, "limit": 12},
            priority=1,
            description=f"Araştırma öncesi eldeki bilgiyi topla: {self._spec(goal)}",
        )

        research_tasks: List[Task] = []
        parallel = self._enable_parallel and len(probes) > 1
        for index, probe in enumerate(probes):
            try:
                research_tasks.append(self._spawn(
                    ctx,
                    name=f"Araştır: {probe[:60]}",
                    kind=TaskKind.RESEARCH,
                    tool="research",
                    args={
                        "query": probe,
                        "goal": "${context.goal}",
                        "depth": min(2, self._max_depth),
                    },
                    priority=2 if parallel else 3 + index,
                    description=f"'{probe}' alt konusunu derinlemesine araştır",
                ))
            except PlanError:
                break

        verify = self._spawn(
            ctx,
            name="Araştırma bulgularını doğrula",
            kind=TaskKind.VERIFY,
            tool="verify",
            args={
                "goal": "${context.goal}",
                "notes": "${%s.result}" % retrieve.id,
                "findings": "${%s.result}" % (research_tasks[0].id if research_tasks else retrieve.id),
                "min_confidence": MEMORIZE_CONFIDENCE_THRESHOLD,
            },
            priority=4,
            # Doğrulama yalnızca elde bulgular varken anlamlıdır; araştırma
            # kanatları çevrimdışı/engelli atlanırsa doğrulama atlanmaz ki
            # sentez yine de yanıt üretebilsin (zarafetli bozulma).
            description="Kaynakları çapraz kontrol et, çelişkileri işaretle",
        )
        self._mark_optional(graph, verify, [t.id for t in research_tasks])
        synthesize = self._spawn(
            ctx,
            name="Araştırmayı sentezle",
            kind=TaskKind.SYNTHESIZE,
            tool="synthesize",
            args={"goal": "${context.goal}", "context": "${context}"},
            critical=True,
            priority=0,
            description="Tüm bulguları tek, tutarlı bir yanıtta birleştir",
        )
        memorize = self._spawn(
            ctx,
            name="Araştırma özetini kasaya kaydet",
            kind=TaskKind.MEMORIZE,
            tool="memorize",
            args={
                "goal": "${context.goal}",
                "answer": "${%s.result}" % synthesize.id,
                "tags": ["arastirma", "planner"],
                "min_confidence": MEMORIZE_CONFIDENCE_THRESHOLD,
            },
            priority=8,
            description="Araştırma sonucunu bilgi kasasına yeni not olarak yaz",
        )

        for task in research_tasks:
            self._wire(graph, retrieve, task)
        self._fan_out(graph, research_tasks or [retrieve], verify)
        self._wire(graph, retrieve, verify)
        self._chain(graph, [verify, synthesize, memorize])
        return graph

    # ─── Şablon: Hesaplama ──────────────────────────────────────────────

    def _tpl_compute(self, goal: str, classification: Dict[str, Any],
                     ctx: Dict[str, Any], depth: int) -> TaskGraph:
        """Hesaplama: formül getir → hesapla → doğrula → sentezle."""
        graph: TaskGraph = ctx["_graph"]
        retrieve = self._spawn(
            ctx,
            name="Formül ve bağlamı getir",
            kind=TaskKind.RETRIEVE,
            tool="retrieve",
            args={"query": "${context.goal}", "limit": 8, "kind": "formula"},
            priority=1,
            description=f"Hesap için gerekli formül/sabitleri kasadan al: {self._spec(goal)}",
        )
        compute = self._spawn(
            ctx,
            name="Hesaplamayı yürüt",
            kind=TaskKind.COMPUTE,
            tool="compute",
            args={"expression": "${context.goal}", "goal": "${context.goal}"},
            priority=2,
            description="İfadeyi/durumu sayısal olarak çöz",
        )
        verify = self._spawn(
            ctx,
            name="Hesap sonucunu doğrula",
            kind=TaskKind.VERIFY,
            tool="verify",
            args={
                "goal": "${context.goal}",
                "result": "${%s.result}" % compute.id,
                "min_confidence": MEMORIZE_CONFIDENCE_THRESHOLD,
            },
            priority=3,
            description="Birim, yuvarlama ve mantık tutarlılığını denetle",
        )
        synthesize = self._spawn(
            ctx,
            name="Sonucu sentezle",
            kind=TaskKind.SYNTHESIZE,
            tool="synthesize",
            args={"goal": "${context.goal}", "context": "${context}"},
            critical=True,
            priority=0,
            description="Sayısal sonucu açıklamalı yanıta dönüştür",
        )
        self._chain(graph, [retrieve, compute, verify, synthesize])
        return graph

    # ─── Şablon: Sistem ─────────────────────────────────────────────────

    def _tpl_system(self, goal: str, classification: Dict[str, Any],
                    ctx: Dict[str, Any], depth: int) -> TaskGraph:
        """Sistem sorgusu: os.system_report → yorum → sentezle."""
        graph: TaskGraph = ctx["_graph"]
        report = self._spawn(
            ctx,
            name="Sistem raporunu al",
            kind=TaskKind.TOOL,
            tool="os.system_report",
            args={"goal": "${context.goal}", "sections": ["cpu", "gpu", "ram", "disk"]},
            priority=1,
            description="Canlı donanım/işletim sistemi envanterini çıkar",
        )
        reason = self._spawn(
            ctx,
            name="Raporu yorumla",
            kind=TaskKind.REASON,
            tool="reason",
            args={
                "goal": "${context.goal}",
                "system_report": "${%s.result}" % report.id,
                "question": "${context.goal}",
            },
            priority=2,
            description="Ölçümleri hedefe göre yorumla (eşikler, anormallikler)",
        )
        synthesize = self._spawn(
            ctx,
            name="Sistem yanıtını sentezle",
            kind=TaskKind.SYNTHESIZE,
            tool="synthesize",
            args={"goal": "${context.goal}", "context": "${context}"},
            critical=True,
            priority=0,
            description="Ölçüm + yorumu kullanıcıya dönük yanıta çevir",
        )
        self._chain(graph, [report, reason, synthesize])
        return graph

    # ─── Şablon: Dosya İşlemi ───────────────────────────────────────────

    def _tpl_file_operation(self, goal: str, classification: Dict[str, Any],
                            ctx: Dict[str, Any], depth: int) -> TaskGraph:
        """Dosya işlemi: os.<eylem> aracı → yorum → sentezle."""
        graph: TaskGraph = ctx["_graph"]
        low = (goal or "").lower()
        if any(w in low for w in ("oku", "read", "içeriğini", "icerigini")):
            tool, operation = "os.read", "oku"
        elif any(w in low for w in ("yaz", "kaydet", "write", "save")):
            tool, operation = "os.write", "yaz"
        else:
            tool, operation = "os.list", "listele"

        tool_task = self._spawn(
            ctx,
            name=f"Dosya işlemi: {operation}",
            kind=TaskKind.TOOL,
            tool=tool,
            args={"path": "${context.path}", "goal": "${context.goal}", "dry_run": True},
            priority=1,
            description=f"İşletim sistemi katmanı üzerinden dosya işlemi ({operation})",
        )
        reason = self._spawn(
            ctx,
            name="İşlem sonucunu yorumla",
            kind=TaskKind.REASON,
            tool="reason",
            args={
                "goal": "${context.goal}",
                "file_result": "${%s.result}" % tool_task.id,
                "question": "${context.goal}",
            },
            priority=2,
            description="Dosya içeriğini/yapısını hedefe göre değerlendir",
        )
        synthesize = self._spawn(
            ctx,
            name="Dosya yanıtını sentezle",
            kind=TaskKind.SYNTHESIZE,
            tool="synthesize",
            args={"goal": "${context.goal}", "context": "${context}"},
            critical=True,
            priority=0,
            description="İşlem sonucunu ve yorumu özetle",
        )
        self._chain(graph, [tool_task, reason, synthesize])
        return graph

    # ─── Şablon: Karşılaştırma ──────────────────────────────────────────

    def _tpl_comparison(self, goal: str, classification: Dict[str, Any],
                        ctx: Dict[str, Any], depth: int) -> TaskGraph:
        """Karşılaştırma: iki paralel getirme → fark çıkarımı → sentezle."""
        graph: TaskGraph = ctx["_graph"]
        sides = self._comparison_sides(goal)
        if len(sides) < 2:
            sides = [f"{goal} (birinci taraf)", f"{goal} (ikinci taraf)"]

        retrieve = self._spawn(
            ctx,
            name="Karşılaştırma bağlamını getir",
            kind=TaskKind.RETRIEVE,
            tool="retrieve",
            args={"query": "${context.goal}", "keywords": sides[:2], "limit": 12},
            priority=1,
            description=f"Her iki taraf için ortak bağlamı topla: {self._spec(goal)}",
        )

        side_tasks: List[Task] = []
        for index, side in enumerate(sides[:3]):
            try:
                side_tasks.append(self._spawn(
                    ctx,
                    name=f"Tarafı incele: {side[:60]}",
                    kind=TaskKind.RETRIEVE,
                    tool="retrieve",
                    args={"query": side, "limit": 8, "side": index + 1},
                    priority=2,
                    description=f"'{side}' tarafının özelliklerini getir",
                ))
            except PlanError:
                break

        if len(side_tasks) >= 2:
            reason = self._spawn(
                ctx,
                name="İki tarafı karşılaştır (fark analizi)",
                kind=TaskKind.REASON,
                tool="reason",
                args={
                    "goal": "${context.goal}",
                    "left": "${%s.result}" % side_tasks[0].id,
                    "right": "${%s.result}" % side_tasks[1].id,
                    "question": "${context.goal}",
                },
                priority=3,
                description="Ortak ölçütlerde farkları ve üstünlükleri çıkar",
            )
        else:
            reason = self._spawn(
                ctx,
                name="Fark analizi yap",
                kind=TaskKind.REASON,
                tool="reason",
                args={
                    "goal": "${context.goal}",
                    "context_notes": "${%s.result}" % retrieve.id,
                    "question": "${context.goal}",
                },
                priority=3,
                description="Elde edilen bilgiden farkları çıkar",
            )

        synthesize = self._spawn(
            ctx,
            name="Karşılaştırma sonucunu sentezle",
            kind=TaskKind.SYNTHESIZE,
            tool="synthesize",
            args={"goal": "${context.goal}", "context": "${context}"},
            critical=True,
            priority=0,
            description="Karşılaştırmayı tablo/çıkarım biçiminde özetle",
        )

        for task in side_tasks:
            self._wire(graph, retrieve, task)
            self._wire(graph, task, reason)
        if not side_tasks:
            self._wire(graph, retrieve, reason)
        self._wire(graph, reason, synthesize)
        return graph

    def _comparison_sides(self, goal: str) -> List[str]:
        """Karşılaştırılacak iki tarafı metinden çıkarır."""
        text = goal or ""
        pattern = r"|".join(
            [r"\s+vs\.?\s+", r"\s+versus\s+", r"\s+ile\s+", r"\s+ile\s+",
             r"\s+ve\s+", r"\s+karşı\s+", r"\s+karsı\s+", r"\s+compared to\s+"]
        )
        parts = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            left = re.sub(
                r"(karşılaştır|karsilastir|fark|farkı|farki|compare|vs\.?|hangisi|daha|iyi|"
                r"arasındaki|arasindaki|nedir|ne)\b",
                "", parts[0], flags=re.IGNORECASE,
            ).strip(" ?.,;:")
            right = re.sub(
                r"(karşılaştır|karsilastir|fark|farkı|farki|compare|hangisi|daha|iyi|"
                r"arasındaki|arasindaki|nedir|ne)\b",
                "", parts[1], flags=re.IGNORECASE,
            ).strip(" ?.,;:")
            sides = [s for s in (left, right) if len(s) >= 2]
            if len(sides) >= 2:
                return sides[:3]

        keywords = self._keywords((goal or "").lower())
        return keywords[:3] if len(keywords) >= 2 else []

    # ─── Şablon: Nasıl-Yapılır ──────────────────────────────────────────

    def _tpl_howto(self, goal: str, classification: Dict[str, Any],
                   ctx: Dict[str, Any], depth: int) -> TaskGraph:
        """Nasıl-yapılır: getir → adım çıkar → doğrula → sentezle."""
        graph: TaskGraph = ctx["_graph"]
        retrieve = self._spawn(
            ctx,
            name="Yöntem bilgisini getir",
            kind=TaskKind.RETRIEVE,
            tool="retrieve",
            args={"query": "${context.goal}", "limit": 10, "kind": "procedure"},
            priority=1,
            description=f"Kasadaki ilgili yordam/notları ara: {self._spec(goal)}",
        )
        reason = self._spawn(
            ctx,
            name="Adımları çıkar",
            kind=TaskKind.REASON,
            tool="reason",
            args={
                "goal": "${context.goal}",
                "notes": "${%s.result}" % retrieve.id,
                "instruction": "Adımları sıralı, uygulanabilir biçimde çıkar",
            },
            priority=2,
            description="Bilgiden sıralı ve uygulanabilir adımlar üret",
        )
        verify = self._spawn(
            ctx,
            name="Adımları doğrula",
            kind=TaskKind.VERIFY,
            tool="verify",
            args={
                "goal": "${context.goal}",
                "steps": "${%s.result}" % reason.id,
                "min_confidence": MEMORIZE_CONFIDENCE_THRESHOLD,
            },
            priority=3,
            description="Adımların eksiksizliğini ve sırasını denetle",
        )
        synthesize = self._spawn(
            ctx,
            name="Kılavuzu sentezle",
            kind=TaskKind.SYNTHESIZE,
            tool="synthesize",
            args={"goal": "${context.goal}", "context": "${context}"},
            critical=True,
            priority=0,
            description="Adımları okunabilir bir kılavuza dönüştür",
        )
        self._chain(graph, [retrieve, reason, verify, synthesize])
        return graph

    # ─── Şablon: Kod Görevi ─────────────────────────────────────────────

    def _tpl_code_task(self, goal: str, classification: Dict[str, Any],
                       ctx: Dict[str, Any], depth: int) -> TaskGraph:
        """Kod görevi: getir → tasarım → (hesap/araç) → doğrula → sentezle."""
        graph: TaskGraph = ctx["_graph"]
        low = (goal or "").lower()
        retrieve = self._spawn(
            ctx,
            name="İlgili kod/not bağlamını getir",
            kind=TaskKind.RETRIEVE,
            tool="retrieve",
            args={"query": "${context.goal}", "limit": 10, "kind": "code"},
            priority=1,
            description=f"Benzer kod parçaları ve notları getir: {self._spec(goal)}",
        )
        design = self._spawn(
            ctx,
            name="Çözümü tasarla",
            kind=TaskKind.REASON,
            tool="reason",
            args={
                "goal": "${context.goal}",
                "notes": "${%s.result}" % retrieve.id,
                "instruction": "Kod tasarımı: veri akışı, fonksiyonlar, hata yolları",
            },
            priority=2,
            description="Sorunun kök nedenini ve çözüm taslağını belirle",
        )

        previous: Task = design
        if classification.get("needs_compute"):
            compute = self._spawn(
                ctx,
                name="Karmaşıklık/mantık hesabı",
                kind=TaskKind.COMPUTE,
                tool="compute",
                args={"expression": "${context.goal}", "goal": "${context.goal}"},
                priority=3,
                description="Algoritmik maliyet veya parametre hesabı yap",
            )
            self._wire(graph, design, compute)
            previous = compute

        tools = [t for t in classification.get("needs_tools", [])
                 if t in ("sistem", "system", "dosya", "file", "kod", "code", "script")]
        if tools or "test" in low or "çalıştır" in low or "calistir" in low:
            tool_task = self._spawn(
                ctx,
                name="Kodu çalıştır/statik denetle",
                kind=TaskKind.TOOL,
                tool="os.read",
                args={"goal": "${context.goal}", "dry_run": True},
                priority=4,
                description="Yazılan/incelenen kodu güvenli biçimde çalıştır veya oku",
            )
            self._wire(graph, previous, tool_task)
            previous = tool_task

        verify = self._spawn(
            ctx,
            name="Çözümü doğrula",
            kind=TaskKind.VERIFY,
            tool="verify",
            args={
                "goal": "${context.goal}",
                "solution": "${%s.result}" % previous.id,
                "min_confidence": MEMORIZE_CONFIDENCE_THRESHOLD,
            },
            priority=5,
            description="Çözümün hedefi karşıladığını ve çalıştığını denetle",
        )
        synthesize = self._spawn(
            ctx,
            name="Kod yanıtını sentezle",
            kind=TaskKind.SYNTHESIZE,
            tool="synthesize",
            args={"goal": "${context.goal}", "context": "${context}"},
            critical=True,
            priority=0,
            description="Kodu/çözümü açıklamalı biçimde sun",
        )
        self._chain(graph, [previous, verify, synthesize])
        return graph

    # ─── Şablon: Yedek ──────────────────────────────────────────────────

    def _tpl_fallback(self, goal: str, classification: Dict[str, Any],
                      ctx: Dict[str, Any], depth: int) -> TaskGraph:
        """Yedek şablon: getir → akıl yürüt → sentezle."""
        graph: TaskGraph = ctx["_graph"]
        retrieve = self._spawn(
            ctx,
            name="İlgili bilgiyi getir",
            kind=TaskKind.RETRIEVE,
            tool="retrieve",
            args={"query": "${context.goal}", "limit": 10},
            priority=1,
            description=f"Kasadan hedefle ilgili bağlamı topla: {self._spec(goal)}",
        )
        reason = self._spawn(
            ctx,
            name="Hedefi akıl yürüterek çöz",
            kind=TaskKind.REASON,
            tool="reason",
            args={
                "goal": "${context.goal}",
                "notes": "${%s.result}" % retrieve.id,
                "question": "${context.goal}",
            },
            priority=2,
            description="Getirilen bağlam üzerinden çıkarım yap",
        )
        synthesize = self._spawn(
            ctx,
            name="Yanıtı sentezle",
            kind=TaskKind.SYNTHESIZE,
            tool="synthesize",
            args={"goal": "${context.goal}", "context": "${context}"},
            critical=True,
            priority=0,
            description="Çıkarımı nihai yanıta dönüştür",
        )
        self._chain(graph, [retrieve, reason, synthesize])
        return graph


# ─── Modül Yardımcıları ─────────────────────────────────────────────────────

def decomposition_summary(graph: TaskGraph) -> Dict[str, Any]:
    """Ayrıştırma sonucunun kısa, JSON-serileştirilebilir özeti."""
    stats = graph.stats()
    return json_safe({
        "goal": stats.get("goal"),
        "tasks": stats.get("total_tasks"),
        "waves": stats.get("waves"),
        "max_parallel_batch": stats.get("max_parallel_batch"),
        "by_kind": stats.get("by_kind"),
        "critical_path": stats.get("critical_path"),
    })
