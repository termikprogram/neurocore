"""
NeuroCore Ana Motoru (L0)
==========================
Sekiz katmanı tek bir bilişsel sistemde birleştiren orkestratör.

Yaşam döngüsü::

    CREATED → INITIALIZING → READY → RUNNING ⇄ DEGRADED → STOPPING → STOPPED

Sorumluluklar:
    1. **Donanım keşfi ve profil seçimi** — RAM/VRAM'e göre tüm parametreler
    2. **Katman kurulumu** — sırayla ve bağımlılık sırasına göre
    3. **Kablolama** — katmanlar birbirini import etmez; motor enjekte eder
    4. **Sorgu yönlendirme** — basit / derin / araştırma yolları
    5. **Bakım döngüsü** — bellek, indeks, graf, evrim turu
    6. **Dayanıklılık** — bellek baskısı, kısmi arıza, zarif bozulma

Kablolama şeması (tek yönlü enjeksiyon)::

    HardwareMonitor ──► ResourceAllocator ──► (bütçe) ──► Neural / Knowledge
                                              └─ olaylar ─► EventBus ─► herkes

    KnowledgeLayer ◄──embed_fn── EmbeddingEngine ◄──model── NeuroCoreModel
         │                                                       ▲
         └──hooks──► EvolutionLayer ──karar──► GrowthController ──┘

    PlannerLayer ◄──yetenek kaydı── (retrieve, reason, synthesize, os.*, …)
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from neurocore.core.config import HardwareProfile, SystemConfig
from neurocore.core.event_bus import Event, EventBus, EventPriority
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("core.engine")


# ─── Yaşam Döngüsü Durumları ────────────────────────────────────────────────

class EngineState(str, Enum):
    """Motorun yaşam döngüsü durumu."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


# ─── Sonuç Yapıları ─────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """
    `NeuroCoreEngine.query()` çıktısı.

    Attributes:
        query: Özgün sorgu
        answer: Üretilen yanıt (nöral üretim veya kasa sentezi)
        mode: `search` | `answer` | `deep` | `research` | `system`
        confidence: 0..1 — yanıtın güveni
        sources: Kullanılan notların Obsidian bağlantıları
        notes: Kullanılan notların kimlik ve başlıkları
        context: Modele verilen bağlam metni
        quality: Kendi kendini değerlendirme puanı (0..1)
        gaps: Bilgi boşlukları (araştırma gerektiren alanlar)
        plan: Derin modda görev planı özeti
        trace: Akıl yürütme izi (özet)
        learned_note_id: Yanıttan üretilen not (varsa)
        duration_ms: Toplam süre
        tokens: Kullanılan token tahmini
        language: Algılanan dil
        error: Hata mesajı (başarısızsa)
    """
    query: str
    answer: str = ""
    mode: str = "search"
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    notes: List[Dict[str, str]] = field(default_factory=list)
    context: str = ""
    quality: float = 0.0
    gaps: List[str] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
    learned_note_id: str = ""
    duration_ms: float = 0.0
    tokens: int = 0
    language: str = "unknown"
    error: str = ""

    @property
    def ok(self) -> bool:
        """Yanıt üretildi mi?"""
        return bool(self.answer) and not self.error

    def to_dict(self, include_context: bool = False) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        data = {
            "query": self.query,
            "answer": self.answer,
            "mode": self.mode,
            "confidence": round(self.confidence, 4),
            "quality": round(self.quality, 4),
            "sources": list(self.sources),
            "notes": list(self.notes),
            "gaps": list(self.gaps),
            "plan": self.plan,
            "trace": self.trace,
            "learned_note_id": self.learned_note_id,
            "duration_ms": round(self.duration_ms, 1),
            "tokens": self.tokens,
            "language": self.language,
            "error": self.error,
            "ok": self.ok,
        }
        if include_context:
            data["context"] = self.context
        return data

    def render(self, include_sources: bool = True) -> str:
        """
        Kullanıcıya gösterilecek metni üretir (kaynak atıflarıyla).
        """
        parts: List[str] = [self.answer or "(yanıt üretilemedi)"]
        if include_sources and self.sources:
            parts.append("\n**Kaynaklar:**")
            parts.extend(f"- {source}" for source in self.sources[:8])
        meta: List[str] = []
        if self.mode:
            meta.append(f"mod: {self.mode}")
        if self.confidence:
            meta.append(f"güven: {self.confidence:.2f}")
        if self.quality:
            meta.append(f"kalite: {self.quality:.2f}")
        if self.duration_ms:
            meta.append(f"{self.duration_ms:.0f}ms")
        if meta:
            parts.append("\n_" + " · ".join(meta) + "_")
        return "\n".join(parts)


@dataclass
class HealthReport:
    """Motorun bütünsel sağlık raporu."""
    state: str = "created"
    ok: bool = False
    uptime_seconds: float = 0.0
    profile: str = ""
    device: str = ""
    layers: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    critical: List[str] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "state": self.state, "ok": self.ok,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "profile": self.profile, "device": self.device,
            "layers": self.layers, "issues": self.issues,
            "critical": self.critical, "counters": self.counters,
        }

    def render(self) -> str:
        """İnsan okur özet."""
        icon = "✓" if self.ok else "✗"
        lines = [
            f"{icon} NeuroCore — durum: {self.state} · profil: {self.profile} · "
            f"cihaz: {self.device} · çalışma süresi: {self.uptime_seconds:.0f}s",
        ]
        for name, info in self.layers.items():
            healthy = info.get("ok", info.get("healthy", True)) if isinstance(info, dict) else True
            mark = "✓" if healthy else "!"
            lines.append(f"  {mark} {name}")
        if self.critical:
            lines.append("KRİTİK: " + "; ".join(self.critical))
        if self.issues:
            lines.append("Uyarılar: " + "; ".join(self.issues[:6]))
        return "\n".join(lines)


# ─── Ana Motor ──────────────────────────────────────────────────────────────

class NeuroCoreEngine:
    """
    NeuroCore'un tamamını yöneten motor.

    Kullanım:
        engine = NeuroCoreEngine()
        engine.initialize()                       # donanımı algıla, katmanları kur
        engine.start()                            # arka plan döngülerini başlat

        result = engine.query("GPU sıcaklığı neden yükselir?")
        print(result.render())

        deep = engine.query("Ryzen 9 ve i7 karşılaştır", deep=True)
        report = engine.research("termal macun", depth=2)
        engine.learn("Yeni bilgi...", title="Termal Yönetim", tags=["donanım"])

        engine.evolve(cycles=2)                   # evrim turu
        print(engine.health_report().render())
        engine.shutdown()
    """

    VERSION = "0.1.0"

    def __init__(self, config: Optional[SystemConfig] = None) -> None:
        """
        Args:
            config: Önceden hazırlanmış konfigürasyon (None = otomatik oluştur)
        """
        self.config = config or SystemConfig()
        self.event_bus = EventBus(history_size=2000)

        # Katmanlar (initialize'da kurulur)
        self.monitor: Any = None
        self.allocator: Any = None
        self.profiler: Any = None
        self.knowledge: Any = None
        self.model: Any = None
        self.embedding: Any = None
        self.memory: Any = None
        self.growth: Any = None
        self.trainer: Any = None
        self.multimodal: Any = None
        self.os_layer: Any = None
        self.language: Any = None
        self.planner: Any = None
        self.evolution: Any = None

        self.state = EngineState.CREATED
        self._lock = threading.RLock()
        self._started_at = 0.0
        self._init_errors: Dict[str, str] = {}
        self._layer_health: Dict[str, Any] = {}
        self._maintenance_thread: Optional[threading.Thread] = None
        self._maintenance_stop = threading.Event()
        self._budget: Dict[str, Any] = {}

        self._counter = {
            "queries": 0, "deep_queries": 0, "learnings": 0, "researches": 0,
            "evolutions": 0, "maintenance_rounds": 0, "degradations": 0,
            "errors": 0, "os_calls": 0, "translations": 0,
        }
        self._last_budget_at = 0.0
        self._query_history: List[Dict[str, Any]] = []

        # Olay kancaları: bellek baskısı değişince bütçeyi tazele
        self.event_bus.subscribe("hardware.*", self._on_hardware_event)
        self.event_bus.subscribe("allocator.*", self._on_allocator_event)

    # ═══════════════════════════════════════════════════════════════════
    #  YAŞAM DÖNGÜSÜ
    # ═══════════════════════════════════════════════════════════════════

    def initialize(self, auto_profile: bool = True, load_model: bool = True) -> "NeuroCoreEngine":
        """
        Motoru başlatır: donanımı algılar, profili seçer, katmanları kurar.

        Args:
            auto_profile: Donanıma göre profili otomatik ayarla
            load_model: Diskteki modeli yükle (yoksa yeni omurga kur)

        Returns:
            self (zincirleme kullanım için)

        Raises:
            RuntimeError: Zorunlu katmanlar (donanım, bilgi) kurulamazsa
        """
        with self._lock:
            if self.state in (EngineState.READY, EngineState.RUNNING):
                logger.debug("Motor zaten başlatılmış")
                return self
            self.state = EngineState.INITIALIZING
            start = time.perf_counter()

        logger.info("=" * 62)
        logger.info(f"NeuroCore {self.VERSION} başlatılıyor — {platform.system()} {platform.release()}")
        logger.info("=" * 62)

        # ── Aşama 1: Donanım ─────────────────────────────────────────
        if not self._init_hardware(auto_profile):
            self.state = EngineState.FAILED
            raise RuntimeError("Donanım katmanı kurulamadı — sistem başlatılamıyor")

        # ── Aşama 2: Bilgi (kasa + indeks + graf + arama) ────────────
        self._init_knowledge()

        # ── Aşama 3: Nöral (model + gömme + bellek) ──────────────────
        self._init_neural(load_model=load_model)

        # ── Aşama 4: Gömme fonksiyonunu bilgi katmanına bağla ────────
        self._wire_embedding()

        # ── Aşama 5: Çok-modlu ───────────────────────────────────────
        self._safe_init("multimodal", self._init_multimodal)

        # ── Aşama 6: İşletim sistemi ─────────────────────────────────
        self._safe_init("os", self._init_os)

        # ── Aşama 7: Dil ─────────────────────────────────────────────
        self._safe_init("language", self._init_language)

        # ── Aşama 8: Evrim ───────────────────────────────────────────
        self._safe_init("evolution", self._init_evolution)

        # ── Aşama 9: Planlayıcı (yetenekler bağlandıktan sonra) ──────
        self._safe_init("planner", self._init_planner)

        # ── Aşama 10: İlk bütçe + sağlık ─────────────────────────────
        self._refresh_budget(force=True)
        self.refresh_health()

        elapsed = (time.perf_counter() - start) * 1000
        with self._lock:
            self.state = EngineState.READY if not self._critical_failures() else EngineState.DEGRADED

        logger.info("=" * 62)
        logger.info(
            f"NeuroCore hazır — profil: {self.config.hardware_profile}, "
            f"cihaz: {self.config.device}, {elapsed:.0f}ms"
        )
        if self._init_errors:
            for layer, error in self._init_errors.items():
                logger.warning(f"  ! {layer} katmanı kurulamadı: {error}")
        logger.info("=" * 62)

        self.event_bus.emit(
            "engine.ready",
            {"state": self.state.value, "profile": self.config.hardware_profile,
             "device": self.config.device, "init_ms": round(elapsed, 1),
             "errors": dict(self._init_errors)},
            source="core.engine",
        )
        return self

    def start(self, maintenance_interval: float = 300.0) -> "NeuroCoreEngine":
        """
        Arka plan döngülerini başlatır (donanım izleme + periyodik bakım).

        Args:
            maintenance_interval: Bakım turu aralığı (saniye)

        Returns:
            self
        """
        if self.state not in (EngineState.READY, EngineState.DEGRADED, EngineState.RUNNING):
            self.initialize()

        if self.monitor is not None:
            try:
                self.monitor.start_monitoring()
            except Exception as exc:
                self._record_error("monitor", exc)

        if self._maintenance_thread is None or not self._maintenance_thread.is_alive():
            self._maintenance_stop.clear()
            self._maintenance_thread = threading.Thread(
                target=self._maintenance_loop,
                args=(maintenance_interval,),
                daemon=True,
                name="NeuroCoreMaintenance",
            )
            self._maintenance_thread.start()

        with self._lock:
            self._started_at = time.time()
            self.state = EngineState.RUNNING
        logger.info(f"Motor çalışıyor (bakım aralığı: {maintenance_interval:.0f}s)")
        self.event_bus.emit("engine.started", {"interval": maintenance_interval},
                            source="core.engine")
        return self

    def shutdown(self, save: bool = True) -> None:
        """
        Motoru güvenle kapatır.

        Kapanış sırası, kuruluş sırasının tersidir: önce dış katmanlar
        (planlayıcı, evrim, dil, OS, çok-modlu), sonra nöral, en son bilgi
        ve donanım. Her katman ayrı korunur — biri hata verse de diğerleri
        kapanır.

        Args:
            save: Model ve belleği diske yaz
        """
        with self._lock:
            if self.state == EngineState.STOPPED:
                return
            self.state = EngineState.STOPPING

        logger.info("NeuroCore kapatılıyor…")
        self._maintenance_stop.set()
        if self._maintenance_thread is not None:
            self._maintenance_thread.join(timeout=5.0)

        if self.monitor is not None:
            try:
                self.monitor.stop_monitoring()
            except Exception as exc:
                logger.debug(f"İzleme durdurulamadı: {exc}")

        # Nöral durumu kaydet (bilgi katmanından önce — gömme hâlâ açık)
        if save:
            try:
                if self.model is not None and self.model.is_ready:
                    self.model.save()
                if self.memory is not None:
                    self.memory.save()
                if self.embedding is not None:
                    self.embedding.save_state(
                        str(self._data_path() / "embedding_state.json")
                    )
            except Exception as exc:
                logger.warning(f"Durum kaydedilemedi: {exc}")

        order: List[Tuple[str, Any]] = [
            ("planner", self.planner), ("evolution", self.evolution),
            ("language", self.language), ("os", self.os_layer),
            ("multimodal", self.multimodal), ("trainer", self.trainer),
            ("growth", self.growth), ("memory", self.memory),
            ("model", self.model), ("embedding", self.embedding),
            ("knowledge", self.knowledge), ("profiler", self.profiler),
            ("allocator", self.allocator), ("monitor", self.monitor),
        ]
        for name, layer in order:
            if layer is None:
                continue
            try:
                layer.shutdown()
            except Exception as exc:
                logger.warning(f"{name} kapatılırken hata: {exc}")

        with self._lock:
            self.state = EngineState.STOPPED
        uptime = (time.time() - self._started_at) if self._started_at else 0.0
        logger.info(
            f"NeuroCore durduruldu — {self._counter['queries']} sorgu, "
            f"{self._counter['learnings']} öğrenme, {uptime:.0f}s çalışma"
        )
        self.event_bus.emit("engine.stopped", {
            "uptime": round(uptime, 1), "counters": dict(self._counter),
        }, source="core.engine")
        self.event_bus.pause()

    def __enter__(self) -> "NeuroCoreEngine":
        """Bağlam yöneticisi: `with NeuroCoreEngine() as engine:`"""
        return self.initialize().start()

    def __exit__(self, *exc_info: Any) -> None:
        """Bağlam yöneticisi çıkışı."""
        self.shutdown()

    # ═══════════════════════════════════════════════════════════════════
    #  KATMAN KURULUMU
    # ═══════════════════════════════════════════════════════════════════

    def _init_hardware(self, auto_profile: bool) -> bool:
        """
        Donanım katmanını kurar: izleme, tahsis, profilleme.

        Profil seçimi RAM ve VRAM'e göre yapılır; `SystemConfig.apply_hardware_profile`
        tüm alt sistem parametrelerini buna göre ayarlar.
        """
        try:
            from neurocore.hardware import HardwareMonitor, PerformanceProfiler, ResourceAllocator

            self.monitor = HardwareMonitor()
            self.monitor.initialize(self.config, self.event_bus)

            total_ram_gb = self.monitor.get_total_ram_gb()
            gpu_vram_gb = self.monitor.get_gpu_vram_gb()

            if auto_profile:
                profile = self.config.auto_configure(total_ram_gb, gpu_vram_gb)
                logger.info(
                    f"Donanım algılandı: {total_ram_gb:.1f}GB RAM, "
                    f"{gpu_vram_gb:.1f}GB VRAM → profil '{profile.value}'"
                )
            else:
                profile = HardwareProfile(self.config.hardware_profile)

            self.allocator = ResourceAllocator()
            self.allocator.initialize(self.config, self.event_bus)

            self.profiler = PerformanceProfiler()
            self.profiler.initialize(self.config, self.event_bus)

            self._layer_health["hardware"] = {
                "ok": True, "profile": profile.value,
                "ram_gb": round(total_ram_gb, 1), "vram_gb": round(gpu_vram_gb, 1),
                "device": self.config.device,
                "headroom": self.config.hardware.headroom.describe(),
            }
            logger.info(self.config.hardware.headroom.describe())
            return True
        except Exception as exc:
            self._record_error("hardware", exc, critical=True)
            return False

    def _init_knowledge(self) -> None:
        """Bilgi katmanını kurar (kasa, indeks, graf, arama, sürümleme)."""
        try:
            from neurocore.knowledge import KnowledgeLayer

            self.knowledge = KnowledgeLayer()
            self.knowledge.initialize(self.config, self.event_bus)
            stats = self.knowledge.get_stats()
            self._layer_health["knowledge"] = {
                "ok": True,
                "notes": stats.get("vault", {}).get("indexed_notes", 0),
                "indexed": stats.get("index", {}).get("notes", 0),
                "fts5": stats.get("index", {}).get("fts5", False),
                "vault": self.knowledge.vault_path,
            }
        except Exception as exc:
            self._record_error("knowledge", exc, critical=True)

    def _init_neural(self, load_model: bool = True) -> None:
        """Nöral katmanı kurar (tokenizer, omurga, gömme, bellek, eğitici)."""
        try:
            from neurocore.neural import (
                EmbeddingEngine, GrowthController, MemoryBank, NeuroCoreModel, Trainer,
            )

            self.embedding = EmbeddingEngine()
            self.embedding.initialize(self.config, self.event_bus)

            if load_model:
                self.model = NeuroCoreModel()
                self.model.initialize(self.config, self.event_bus, budget=self._budget)

                # Tokenizer'ı gömme motoruna bağla (hash gömmede bölütleme için)
                if self.model.tokenizer is not None:
                    self.embedding.attach_tokenizer(self.model.tokenizer)

                # Eğitilmiş model varsa öğrenilmiş gömmeyi aç
                if self.model.is_pretrained:
                    self.embedding.attach_model(self.model.backbone, self.model.tokenizer)
                    logger.info("Eğitilmiş model bulundu — öğrenilmiş gömme etkin")

            self.memory = MemoryBank()
            self.memory.initialize(self.config, self.event_bus)
            self.memory.attach_embedder(self.embedding.embed)

            self.growth = GrowthController()
            self.growth.initialize(self.config, self.event_bus)

            self.trainer = Trainer()
            self.trainer.initialize(self.config, self.event_bus, budget=self._budget)

            # IDF öğren: kasada metin varsa arama kalitesini artır
            if self.knowledge is not None:
                try:
                    texts = [
                        f"{note.title} {note.body[:1500]}"
                        for note in self.knowledge.vault.iter_notes(limit=500)
                    ]
                    if texts:
                        self.embedding.fit_idf(texts)
                except Exception as exc:
                    logger.debug(f"IDF öğrenilemedi: {exc}")

            self._layer_health["neural"] = {
                "ok": True,
                "pretrained": self.model.is_pretrained if self.model else False,
                "params": self.model.parameter_count() if self.model else 0,
                "vocab_size": self.model.model_config.vocab_size if self.model and self.model.model_config else 0,
                "embedding_strategy": self.embedding.strategy,
                "memory_capacity": self.memory.capacity,
            }
        except Exception as exc:
            self._record_error("neural", exc)

    def _init_multimodal(self) -> None:
        """Çok-modlu katmanı kurar."""
        from neurocore.multimodal import MultimodalLayer

        self.multimodal = MultimodalLayer()
        self.multimodal.initialize(self.config, self.event_bus)
        health = self.multimodal.health() if hasattr(self.multimodal, "health") else {}
        self._layer_health["multimodal"] = {
            "ok": bool(health.get("ok", True)),
            "capabilities": health.get("capabilities", {}),
            "issues": health.get("issues", []),
        }

    def _init_os(self) -> None:
        """İşletim sistemi katmanını kurar."""
        from neurocore.os import OSLayer

        self.os_layer = OSLayer()
        self.os_layer.initialize(self.config, self.event_bus)
        health = self.os_layer.health() if hasattr(self.os_layer, "health") else {}
        self._layer_health["os"] = {
            "ok": bool(health.get("ok", True)),
            "issues": health.get("issues", []),
            "critical": health.get("critical", []),
        }

    def _init_language(self) -> None:
        """Dil ve çeviri katmanını kurar."""
        from neurocore.language import LanguageLayer

        self.language = LanguageLayer()
        self.language.initialize(self.config, self.event_bus)
        health = self.language.health() if hasattr(self.language, "health") else {}
        self._layer_health["language"] = {
            "ok": bool(health.get("ok", True)),
            "glossary": health.get("glossary_size", 0),
            "issues": health.get("issues", []),
        }

    def _init_evolution(self) -> None:
        """Evrim katmanını kurar ve kancaları bağlar."""
        from neurocore.evolution import EvolutionHooks, EvolutionLayer

        hooks = self._build_evolution_hooks()
        self.evolution = EvolutionLayer()
        self.evolution.initialize(self.config, self.event_bus, hooks=hooks)
        available = hooks.available() if hasattr(hooks, "available") else []
        self._layer_health["evolution"] = {
            "ok": True, "hooks": list(available),
            "missing_hooks": 12 - len(available),
        }

    def _init_planner(self) -> None:
        """Planlayıcıyı kurar ve yetenekleri kaydeder."""
        from neurocore.planner import CapabilityRegistry, PlannerLayer

        registry = CapabilityRegistry()
        self.planner = PlannerLayer()
        self.planner.initialize(self.config, self.event_bus, capabilities=registry)
        registered = self._register_planner_capabilities()
        self._layer_health["planner"] = {
            "ok": True, "capabilities": registered,
        }

    def _safe_init(self, name: str, initializer: Callable[[], None]) -> None:
        """Bir katmanı hata toleranslı biçimde kurar."""
        try:
            initializer()
        except Exception as exc:
            self._record_error(name, exc)

    def _record_error(self, layer: str, exc: Exception, critical: bool = False) -> None:
        """Katman kurulum hatasını kaydeder."""
        self._init_errors[layer] = str(exc)
        self._counter["errors"] += 1
        self._layer_health[layer] = {"ok": False, "error": str(exc), "critical": critical}
        log = logger.error if critical else logger.warning
        log(f"{layer} katmanı kurulamadı: {exc}")

    def _critical_failures(self) -> List[str]:
        """Kritik katman hatalarını döndürür."""
        return [name for name, info in self._layer_health.items()
                if isinstance(info, dict) and info.get("critical") and not info.get("ok")]

    def _wire_embedding(self) -> None:
        """
        Gömme fonksiyonunu bilgi katmanına enjekte eder.

        Bu, iki katman arasındaki **tek** bağlantıdır ve tek yönlüdür:
        bilgi katmanı nöral katmanı import etmez.
        """
        if self.knowledge is None or self.embedding is None:
            return
        try:
            self.knowledge.attach_embeddings(
                self.embedding.embed,
                embed_model=f"neurocore-{self.embedding.strategy}-{self.embedding.dim}",
            )
        except Exception as exc:
            self._record_error("wiring.embedding", exc)

    # ═══════════════════════════════════════════════════════════════════
    #  KABLOLAMA: EVRİM KANCALARI
    # ═══════════════════════════════════════════════════════════════════

    def _build_evolution_hooks(self) -> Any:
        """
        Evrim katmanı için kanca setini üretir.

        Evrim katmanı diğer katmanları import etmez; yalnızca bu kancalarla
        konuşur. Böylece tek başına da test edilebilir kalır.
        """
        from neurocore.evolution import EvolutionHooks

        hooks = EvolutionHooks()

        # ── Bilgi kancaları ──
        if self.knowledge is not None:
            hooks.list_notes = lambda limit=200: self.knowledge.list_notes(limit=limit)

            def get_note(note_id: str) -> Optional[Dict[str, Any]]:
                payload = self.knowledge.get(note_id, include_body=True)
                if payload is None:
                    return None
                # Evrim katmanı `body` bekler; indeksten gelmiyorsa ekle
                if "body" not in payload:
                    note = self.knowledge.get_note(note_id)
                    if note is not None:
                        payload["body"] = note.body
                return payload

            hooks.get_note = get_note

            def upsert_note(title: str, body: str, tags: Any = None,
                            confidence: float = 0.6, sources: Any = None,
                            kind: str = "concept", **kwargs: Any) -> Dict[str, Any]:
                """Evrim katmanının not yazma kancası."""
                try:
                    note = self.knowledge.write(
                        title=title, body=body, tags=tags,
                        confidence=confidence, sources=sources, kind=kind,
                        reason="evrim sentezi", author="evolution",
                    )
                    payload = note.to_dict(include_body=True)
                    payload["id"] = note.id
                    return payload
                except Exception as exc:
                    logger.warning(f"Evrim notu yazılamadı: {exc}")
                    return {}

            hooks.upsert_note = upsert_note

            def link_notes(source_id: str, target_id: str, reason: str = "") -> bool:
                try:
                    return bool(self.knowledge.link(source_id, target_id,
                                                    relation="derived_from",
                                                    reason=reason))
                except Exception as exc:
                    logger.debug(f"Evrim bağlantısı kurulamadı: {exc}")
                    return False

            hooks.link_notes = link_notes

            def archive_note(note_id: str, reason: str = "") -> bool:
                try:
                    return bool(self.knowledge.archive(note_id, reason=reason))
                except Exception as exc:
                    logger.debug(f"Evrim arşivlemesi başarısız: {exc}")
                    return False

            hooks.archive_note = archive_note

            def graph_edges() -> Dict[str, List[str]]:
                try:
                    edges: Dict[str, List[str]] = {}
                    for edge in self.knowledge.graph.edges:
                        edges.setdefault(edge.src, []).append(edge.dst)
                    return edges
                except Exception:
                    return {}

            hooks.graph_edges = graph_edges

        # ── Gömme kancası ──
        if self.embedding is not None:
            hooks.embed_text = lambda text: self.embedding.embed(text)

        # ── Model kancaları ──
        if self.model is not None:
            hooks.model_metrics = lambda: {
                "pretrained": self.model.is_pretrained,
                "params": self.model.parameter_count(),
                "maturity": self.model.growth.maturity,
                "generation": self.model.growth.generation,
                "routing": self.model.routing_report(),
                "embed_dim": self.model.model_config.embed_dim if self.model.model_config else 0,
                "num_experts": self.model.growth.num_experts,
                "num_layers": self.model.growth.num_layers,
            }

        # ── Kaynak bütçesi kancası ──
        hooks.resource_budget = lambda: dict(self._budget)

        # ── Eğitim kancası ──
        if self.trainer is not None and self.model is not None:
            def train_step(batch: Dict[str, Any]) -> Dict[str, Any]:
                """
                Tek bir eğitim turu çalıştırır (evrim katmanı isteğiyle).

                Küçük ve güvenli tutulur: kaynak bütçesine göre plan üretilir,
                en fazla `max_minutes` kadar sürer.
                """
                try:
                    dataset = self.trainer.build_dataset_from_experiences(
                        batch, self.model.tokenizer,
                        seq_len=int(self._budget.get("max_batch_size", 32) or 32) * 8,
                    )
                    if dataset.is_empty:
                        return {"trained": False, "reason": "veri_yok"}
                    if self.model is not None and not self.model.is_pretrained and \
                            not any(True for _ in dataset.batches(1)):
                        return {"trained": False, "reason": "veri_kodlanamadi"}

                    plan = self.trainer.plan_training(
                        estimated_samples=len(dataset), model=self.model,
                        rationale="evrim katmanı isteği",
                    )
                    # Evrim turunda kısa tut: ilk turda daha uzun (model boş)
                    if not self.model.is_pretrained:
                        plan.steps = min(plan.steps, 200)
                    else:
                        plan.steps = min(plan.steps, 80)

                    result = self.trainer.train(
                        self.model, dataset, plan=plan, save_checkpoints=True,
                    )
                    # İlk eğitim tamamlandıysa öğrenilmiş gömmeyi etkinleştir
                    if result.steps_completed > 0 and not self.model.is_pretrained:
                        self.model._is_pretrained = True
                        self.embedding.attach_model(self.model.backbone, self.model.tokenizer)
                        self.model.save()

                    return {
                        "trained": result.steps_completed > 0,
                        "loss": result.final_loss,
                        "initial_loss": result.initial_loss,
                        "steps": result.steps_completed,
                        "tokens": result.tokens_seen,
                        "duration_ms": result.duration_ms,
                        "stop_reason": result.stop_reason,
                    }
                except Exception as exc:
                    logger.error(f"Evrim eğitim turu başarısız: {exc}")
                    return {"trained": False, "reason": str(exc)}

            hooks.train_step = train_step

        # ── Web araştırma kancası ──
        hooks.web_research = self._web_research_hook
        hooks.now = time.time
        return hooks

    def _web_research_hook(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Çevrimiçi araştırma kancası.

        Bu motor **ağ erişimi varsaymaz**. Dil katmanının API köprüsünde
        anahtar varsa oradan yararlanır; yoksa işletim sistemi katmanının
        yetenek yoklamasıyla durumu bildirir. Sistem çevrimdışıyken araştırma
        yalnızca kasadaki bilgiyle yapılır — bu bilinçli bir sınırlamadır
        (gizlilik ve öngörülebilirlik).
        """
        results: List[Dict[str, Any]] = []
        # 1. Kasadaki mevcut bilgiler
        if self.knowledge is not None:
            for hit in self.knowledge.search(query, limit=max_results):
                results.append({
                    "title": hit.title,
                    "url": f"vault://{hit.note_id}",
                    "snippet": hit.excerpt,
                    "score": hit.score,
                    "source": "vault",
                })
        # 2. Çevrimiçi sağlayıcı varsa (yalnızca anahtar tanımlıysa)
        if self.language is not None and len(results) < max_results:
            try:
                health = self.language.health()
                providers = health.get("providers") or []
                if providers:
                    prompt = (
                        f"Aşağıdaki konuda kısa, maddeli ve doğrulanabilir bilgiler ver: {query}\n"
                        "Her madde tek cümle olsun ve kaynak türü belirtilsin."
                    )
                    response = self.language.bridge.complete(prompt, max_tokens=512)
                    if response.get("ok") and response.get("text"):
                        results.append({
                            "title": f"{query} (çevrimiçi özet)",
                            "url": f"api://{response.get('provider', 'api')}",
                            "snippet": response["text"][:1200],
                            "score": 0.6,
                            "source": "api",
                        })
            except Exception as exc:
                logger.debug(f"Çevrimiçi araştırma atlandı: {exc}")
        return results[:max_results]

    # ═══════════════════════════════════════════════════════════════════
    #  KABLOLAMA: PLANLAYICI YETENEKLERİ
    # ═══════════════════════════════════════════════════════════════════

    def _register_planner_capabilities(self) -> int:
        """
        Planlayıcıya sistem yeteneklerini kaydeder.

        Kaydedilen her yetenek, planlayıcının görev grafiğinde bir düğüm
        türüne karşılık gelir (`retrieve`, `reason`, `synthesize`, …).

        Returns:
            Kaydedilen yetenek sayısı
        """
        if self.planner is None:
            return 0

        registered = 0
        capabilities: List[Tuple[str, Callable[..., Any], str]] = []

        # ── Bilgi ──
        if self.knowledge is not None:
            capabilities.append((
                "retrieve",
                lambda query="", limit=8, **_: {
                    "query": query,
                    "notes": [
                        {"id": hit.note_id, "title": hit.title,
                         "score": hit.score, "excerpt": hit.excerpt}
                        for hit in self.knowledge.search(query, limit=int(limit or 8))
                    ],
                },
                "Bilgi kasasında hibrit arama yapar",
            ))
            capabilities.append((
                "memorize",
                lambda title="", body="", tags=None, confidence=0.55, **_: self._capability_memorize(
                    title, body, tags, confidence
                ),
                "Yeni bilgiyi kasaya not olarak yazar",
            ))
            capabilities.append((
                "link",
                lambda source_id="", target_id="", reason="", **_: {
                    "linked": bool(self.knowledge.link(source_id, target_id,
                                                       reason=reason or "plan bağlantısı"))
                },
                "İki not arasında ilişki kurar",
            ))

        # ── Nöral ──
        capabilities.append((
            "reason",
            lambda thought="", context="", question="", **kwargs: self._capability_reason(
                thought or kwargs.get("goal", ""), context, question
            ),
            "Bağlam üzerinden akıl yürütür ve ara sonuç üretir",
        ))
        capabilities.append((
            "synthesize",
            lambda question="", context="", notes=None, **kwargs: self._capability_synthesize(
                question or kwargs.get("goal", ""), context, notes
            ),
            "Toplanan bilgiden nihai yanıtı üretir",
        ))
        capabilities.append((
            "verify",
            lambda answer="", context="", claims=None, **_: self._capability_verify(
                answer, context
            ),
            "Yanıtı kaynaklarla çapraz doğrular",
        ))
        capabilities.append((
            "compute",
            lambda expression="", **_: self._capability_compute(expression),
            "Matematiksel ifadeyi güvenli biçimde hesaplar",
        ))
        capabilities.append((
            "summarize",
            lambda text="", max_sentences=3, **_: {
                "summary": self._summarize(text, int(max_sentences or 3))
            },
            "Metni özetler",
        ))
        capabilities.append((
            "translate",
            lambda text="", target_lang="en", source_lang=None, **_: self._capability_translate(
                text, target_lang, source_lang
            ),
            "Metni hedef dile çevirir",
        ))
        capabilities.append((
            "research",
            lambda query="", depth=1, max_results=5, **_: {
                "query": query,
                "findings": self._web_research_hook(query, int(max_results or 5)),
            },
            "Kasa ve (varsa) çevrimiçi kaynaklarda araştırma yapar",
        ))

        # ── İşletim sistemi ──
        if self.os_layer is not None:
            capabilities.append((
                "os.system_report",
                lambda **_: self._capability_os_call("system_report"),
                "Sistem envanteri raporu üretir",
            ))
            capabilities.append((
                "os.read",
                lambda path="", **_: self._capability_os_call("read", path=path),
                "Dosya okur (sandbox denetimli)",
            ))
            capabilities.append((
                "os.list",
                lambda path=".", **_: self._capability_os_call("list_dir", path=path),
                "Dizin listeler (sandbox denetimli)",
            ))
            capabilities.append((
                "os.write",
                lambda path="", content="", **_: self._capability_os_call(
                    "write", path=path, content=content
                ),
                "Dosya yazar (sandbox denetimli)",
            ))
            capabilities.append((
                "os.find",
                lambda root=".", pattern="*", **_: self._capability_os_call(
                    "find", root=root, pattern=pattern
                ),
                "Dosya adına göre arar",
            ))
            capabilities.append((
                "os.run",
                lambda command="", **_: self._capability_os_call("run", command=command),
                "Komut çalıştırır (yıkıcı komutlar engellenir)",
            ))

        # ── Çok-modlu ──
        if self.multimodal is not None:
            capabilities.append((
                "multimodal.describe",
                lambda payload=None, path="", **_: self._capability_describe(path or payload),
                "Görüntü/ses/tablo içeriğini betimler",
            ))

        for name, function, description in capabilities:
            try:
                self.planner.register_capability(name, function, description=description)
                registered += 1
            except Exception as exc:
                logger.warning(f"Yetenek kaydedilemedi ({name}): {exc}")

        logger.info(f"Planlayıcıya {registered} yetenek kaydedildi")
        return registered

    # ─── Yetenek Uygulamaları ───────────────────────────────────────────

    def _capability_memorize(self, title: str, body: str, tags: Any,
                             confidence: float) -> Dict[str, Any]:
        """`memorize` yeteneği: planlayıcı sonucu kasaya yazar."""
        if not title and not body:
            return {"stored": False, "reason": "içerik_yok"}
        if self.knowledge is None:
            return {"stored": False, "reason": "bilgi_katmani_yok"}
        try:
            note = self.knowledge.write(
                title=title or self._derive_title(body),
                body=body, tags=tags, confidence=float(confidence or 0.55),
                kind="concept", reason="planlayıcı sonucu", author="planner",
            )
            self._counter["learnings"] += 1
            return {"stored": True, "id": note.id, "title": note.title}
        except Exception as exc:
            return {"stored": False, "reason": str(exc)}

    def _capability_reason(self, goal: str, context: str = "",
                           question: str = "") -> Dict[str, Any]:
        """
        `reason` yeteneği: mevcut bağlamı analiz eder.

        Ham bağlam verilmezse kasada arama yapar. Nöral model eğitilmişse
        kısa bir akıl yürütme üretir; değilse yapısal analiz (anahtar
        kavramlar, ilişkiler, eksikler) döndürür.
        """
        text = context or ""
        if not text and self.knowledge is not None:
            material = self.knowledge.answer_material(goal or question, limit=4)
            text = material.get("context", "")

        analysis: Dict[str, Any] = {
            "goal": goal or question,
            "context_chars": len(text),
        }
        if self.language is not None and text:
            try:
                analysis["keywords"] = self.language.keywords(text, top_k=10)
                analysis["summary"] = self.language.summarize_extractive(text, 3)
            except Exception:
                pass

        if self.model is not None and self.model.is_pretrained:
            generated = self.model.generate_text(
                f"{goal or question}\nBağlam:\n{text[:1200]}\nAnaliz:",
                max_new_tokens=96, temperature=0.4,
            )
            if generated.get("text"):
                analysis["thought"] = generated["text"]
        if "thought" not in analysis:
            analysis["thought"] = analysis.get("summary") or (
                f"'{goal or question}' için {len(text)} karakter bağlam incelendi."
            )
        return analysis

    def _capability_synthesize(self, question: str, context: str = "",
                               notes: Any = None) -> Dict[str, Any]:
        """
        `synthesize` yeteneği: nihai yanıtı üretir.

        Sıra:
            1. Eğitilmiş nöral model (bağlam koşullu üretim)
            2. Çıkarımsal özet (dil katmanı) — kaynak atıflarıyla
            3. Yapısal rapor (kasa boşsa)
        """
        text = context or ""
        if not text and self.knowledge is not None:
            material = self.knowledge.answer_material(question, limit=6)
            text = material.get("context", "")

        sources: List[str] = []
        if isinstance(notes, (list, tuple)):
            for item in notes:
                if isinstance(item, dict) and item.get("title"):
                    identifier = item.get("id", "")
                    sources.append(f"[[{identifier}\\|{item['title']}]]" if identifier
                                   else item["title"])
        if not sources and self.knowledge is not None:
            sources = self.knowledge.answer_material(question, limit=6).get("sources", [])

        # 1. Nöral üretim (yalnızca model yeterince eğitilmişse)
        if self.model is not None and self.model.can_generate and text:
            prompt = (
                f"Soru: {question}\n\n"
                f"Bağlam:\n{text[:2400]}\n\n"
                "Yukarıdaki bağlama dayanarak soruyu Türkçe, açık ve maddeli yanıtla. "
                "Bağlamda olmayan bilgi uydurma.\nYanıt:"
            )
            generated = self.model.generate_text(
                prompt, max_new_tokens=320, temperature=0.6, top_p=0.95,
            )
            if generated.get("text"):
                return {
                    "answer": generated["text"].strip(),
                    "sources": sources,
                    "generator": "neural",
                    "tokens": generated.get("new_tokens", 0),
                }
            if generated.get("quality_rejected"):
                # Kalite kapısı reddetti → çıkarımsal yola düş (zarif bozulma)
                logger.info(
                    "Nöral üretim kalite kapısında reddedildi — çıkarımsal yanıta geçiliyor"
                )

        # 2. Çıkarımsal özet
        if self.language is not None and text:
            try:
                # Bağlam bloğunun başlık/kaynak iskeleti özet için gürültüdür;
                # yalnızca bilgi taşıyan gövde parçaları özetlenir.
                body_text = self._context_bodies(text)
                summary = self.language.summarize_extractive(body_text, max_sentences=5)
                if summary:
                    body = summary.strip()
                    if sources:
                        body += "\n\n**Kaynaklar:**\n" + "\n".join(
                            f"- {source}" for source in sources[:6]
                        )
                    answer = f"**{question}**\n\n{body}" if question else body
                    return {
                        "answer": answer,
                        "sources": sources,
                        "generator": "extractive",
                    }
            except Exception as exc:
                logger.debug(f"Özet üretilemedi: {exc}")

        # 3. Yapısal rapor
        if text:
            answer = (
                f"**{question}**\n\n"
                f"Aşağıdaki bağlam derlendi ({len(text)} karakter):\n\n"
                f"{text[:1500]}"
            )
            if sources:
                answer += "\n\n**Kaynaklar:**\n" + "\n".join(
                    f"- {source}" for source in sources[:6]
                )
            return {"answer": answer, "sources": sources, "generator": "context"}

        return {
            "answer": (
                f"**{question}** hakkında kasada bilgi bulunamadı.\n\n"
                "Öneri: `learn()` ile bilgi ekleyin veya `research()` ile araştırma başlatın."
            ),
            "sources": [],
            "generator": "empty",
            "gaps": ["kasa_bolu"],
        }

    def _capability_verify(self, answer: str, context: str = "",
                           claims: Any = None) -> Dict[str, Any]:
        """
        `verify` yeteneği: yanıt iddialarını bağlamla karşılaştırır.

        Kaynak kapsamı, sayısal tutarlılık ve belirsizlik dilini denetler.
        """
        if not answer:
            return {"ok": False, "reason": "yanit_yok", "score": 0.0}

        report: Dict[str, Any] = {"ok": True, "score": 0.5, "issues": []}
        haystack = (context or "").lower()
        answer_lower = answer.lower()

        if not haystack:
            report["issues"].append("bağlam yok — doğrulama yapılamadı")
            report["score"] = 0.3
            return report

        # 1. Kaynak kapsamı: yanıt cümlelerinin bağlamla örtüşmesi
        sentences = [s.strip() for s in answer.replace("\n", " ").split(".")
                     if len(s.strip()) > 20]
        overlap_scores: List[float] = []
        for sentence in sentences[:20]:
            words = {word for word in sentence.lower().split() if len(word) > 4}
            if not words:
                continue
            hits = sum(1 for word in words if word in haystack)
            overlap_scores.append(hits / len(words))
        coverage = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0

        # 2. Sayısal tutarlılık: yanıttaki sayılar bağlamda geçiyor mu?
        import re
        numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", answer))
        context_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", context or ""))
        fabricated = numbers - context_numbers
        if numbers and len(fabricated) / len(numbers) > 0.5:
            report["issues"].append(
                f"{len(fabricated)} sayı bağlamda bulunamadı — uydurma riski"
            )

        report["coverage"] = round(coverage, 4)
        report["numbers_checked"] = len(numbers)
        report["numbers_unsupported"] = len(fabricated)
        report["score"] = round(
            max(0.0, min(1.0, 0.35 + 0.5 * coverage - 0.15 * (len(fabricated) > 0))), 4
        )
        report["ok"] = report["score"] >= 0.4 and not report["issues"]
        return report

    def _capability_compute(self, expression: str) -> Dict[str, Any]:
        """
        `compute` yeteneği: matematiksel ifadeyi **güvenli** biçimde hesaplar.

        `eval` kullanılmaz — yalnızca AST ile doğrulanmış aritmetik ifadeler
        (sayılar, `+ - * / ** % //`, parantez, `math` fonksiyonları) çalışır.
        """
        import ast
        import math as _math
        import operator

        if not expression or not str(expression).strip():
            return {"ok": False, "error": "ifade_bos"}

        text = str(expression).strip()
        # Türkçe yazım desteği: "3 artı 5" → "3 + 5"
        replacements = {
            "artı": "+", "eksi": "-", "çarpı": "*", "carpi": "*",
            "bölü": "/", "bolu": "/", "kare": "**2", "yüzde": "/100*",
        }
        lowered = text.lower()
        for word, symbol in replacements.items():
            lowered = lowered.replace(word, symbol)
        lowered = lowered.replace("^", "**").replace(",", ".")

        allowed_binary = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        allowed_unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}
        allowed_functions = {
            name: getattr(_math, name) for name in
            ("sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan",
             "asin", "acos", "atan", "floor", "ceil", "fabs", "factorial",
             "degrees", "radians", "hypot", "pow", "gcd")
            if hasattr(_math, name)
        }
        allowed_functions["abs"] = abs
        allowed_functions["round"] = round
        allowed_functions["min"] = min
        allowed_functions["max"] = max
        allowed_names = {"pi": _math.pi, "e": _math.e, "tau": _math.tau}

        def evaluate(node: Any) -> Any:
            """AST düğümünü güvenli biçimde değerlendirir."""
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("yalnızca sayısal sabitler")
            if isinstance(node, ast.BinOp):
                operation = allowed_binary.get(type(node.op))
                if operation is None:
                    raise ValueError(f"desteklenmeyen işlem: {type(node.op).__name__}")
                return operation(evaluate(node.left), evaluate(node.right))
            if isinstance(node, ast.UnaryOp):
                operation = allowed_unary.get(type(node.op))
                if operation is None:
                    raise ValueError("desteklenmeyen tekli işlem")
                return operation(evaluate(node.operand))
            if isinstance(node, ast.Name):
                if node.id in allowed_names:
                    return allowed_names[node.id]
                raise ValueError(f"bilinmeyen ad: {node.id}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in allowed_functions:
                    raise ValueError("izin verilmeyen fonksiyon")
                args = [evaluate(argument) for argument in node.args]
                return allowed_functions[node.func.id](*args)
            if isinstance(node, (ast.List, ast.Tuple)):
                return [evaluate(element) for element in node.elts]
            raise ValueError(f"desteklenmeyen ifade: {type(node).__name__}")

        try:
            tree = ast.parse(lowered, mode="eval")
            value = evaluate(tree)
            return {"ok": True, "expression": text, "value": value,
                    "normalized": lowered}
        except Exception as exc:
            return {"ok": False, "expression": text, "error": str(exc)}

    def _capability_translate(self, text: str, target_lang: str,
                              source_lang: Optional[str]) -> Dict[str, Any]:
        """`translate` yeteneği: dil katmanına devreder."""
        if self.language is None or not text:
            return {"ok": False, "error": "dil_katmani_yok"}
        try:
            result = self.language.translate(text, target_lang=target_lang,
                                             source_lang=source_lang)
            self._counter["translations"] += 1
            return result
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _capability_os_call(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        """
        `os.*` yetenekleri: işletim sistemi katmanına denetimli çağrı.

        Sonuç her zaman sözlüktür; hatalar `ok=False` ile döner (planlayıcı
        istisna görmez).
        """
        self._counter["os_calls"] += 1
        if self.os_layer is None:
            return {"ok": False, "error": "os_katmani_yok"}
        try:
            if method == "system_report":
                report = self.os_layer.system_report()
                return {"ok": True, "report": report}
            if method == "read":
                return {"ok": True, "path": kwargs.get("path", ""),
                        "content": self.os_layer.read(kwargs.get("path", ""))[:8000]}
            if method == "list_dir":
                items = self.os_layer.list_dir(kwargs.get("path", "."))
                return {"ok": True, "items": items[:200], "count": len(items)}
            if method == "write":
                path = self.os_layer.write(kwargs.get("path", ""),
                                           kwargs.get("content", ""))
                return {"ok": True, "path": path}
            if method == "find":
                matches = self.os_layer.find(kwargs.get("root", "."),
                                             kwargs.get("pattern", "*"))
                return {"ok": True, "matches": matches[:200], "count": len(matches)}
            if method == "run":
                result = self.os_layer.run(kwargs.get("command", ""))
                return {"ok": bool(result.get("ok")), **result}
            return {"ok": False, "error": f"bilinmeyen_yontem: {method}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "method": method}

    def _capability_describe(self, payload: Any) -> Dict[str, Any]:
        """`multimodal.describe` yeteneği."""
        if self.multimodal is None:
            return {"ok": False, "error": "multimodal_katmani_yok"}
        try:
            if isinstance(payload, str) and os.path.exists(payload):
                chunk = self.multimodal.encode_file(payload)
            else:
                chunk = self.multimodal.encode(payload)
            return {
                "ok": True,
                "modality": chunk.modality.value if hasattr(chunk.modality, "value")
                            else str(chunk.modality),
                "confidence": chunk.confidence,
                "meta": {key: value for key, value in (chunk.meta or {}).items()
                         if key != "native"},
                "vector_dim": len(chunk.vector),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ═══════════════════════════════════════════════════════════════════
    #  SORGU
    # ═══════════════════════════════════════════════════════════════════

    def query(
        self,
        text: str,
        *,
        deep: bool = False,
        limit: int = 8,
        learn: bool = True,
        use_memory: bool = True,
        max_minutes: Optional[float] = None,
        remember: bool = True,
    ) -> QueryResult:
        """
        Sorguyu yanıtlar.

        Yollar:
            * **search** — yalnızca bilgi kasasında arama (en hızlı)
            * **answer** — arama + yanıt sentezi (varsayılan)
            * **deep** — planlayıcı ile çok adımlı akıl yürütme
            * **system** — sistemle ilgili sorular canlı donanım verisiyle

        Args:
            text: Kullanıcı sorgusu
            deep: Planlayıcıyı kullan (çok adımlı)
            limit: Kullanılacak maksimum not
            learn: Yanıttan not üret (öğren)
            use_memory: Bellek bankasını kullan
            max_minutes: Derin modda zaman sınırı
            remember: Etkileşimi belleğe ve deneyim tamponuna yaz

        Returns:
            QueryResult
        """
        start = time.perf_counter()
        result = QueryResult(query=(text or "").strip())

        if not result.query:
            result.error = "boş_sorgu"
            return result
        if self.state in (EngineState.STOPPED, EngineState.FAILED):
            result.error = f"motor_durumu: {self.state.value}"
            return result

        self._counter["queries"] += 1
        self._refresh_budget()

        # ── 1. Dil algılama + iç dile çeviri ──
        if self.language is not None:
            try:
                detected = self.language.detect(result.query)
                result.language = detected.get("code", "unknown")
                if result.language not in ("tr", "unknown"):
                    translated = self.language.to_internal(result.query)
                    if translated.get("was_translated"):
                        result.query = translated.get("text") or result.query
                        self._counter["translations"] += 1
            except Exception as exc:
                logger.debug(f"Dil algılama atlandı: {exc}")

        # ── 2. Sistem sorgusu mu? ──
        if self._is_system_query(result.query):
            self._answer_system_query(result)
            result.duration_ms = (time.perf_counter() - start) * 1000
            self._finish_query(result, remember=remember, learn=learn)
            return result

        # ── 3. Derin mod: planlayıcı ──
        if deep and self.planner is not None:
            return self._answer_deep(result, limit=limit, learn=learn,
                                     remember=remember, max_minutes=max_minutes,
                                     start=start)

        # ── 4. Standart yol: bağlam + sentez ──
        self._answer_standard(result, limit=limit, use_memory=use_memory)
        result.duration_ms = (time.perf_counter() - start) * 1000
        self._finish_query(result, remember=remember, learn=learn)
        return result

    def _answer_standard(self, result: QueryResult, limit: int = 8,
                         use_memory: bool = True) -> None:
        """Standart sorgu yolu: kasa bağlamı + bellek + sentez."""
        result.mode = "answer"

        # 1. Kasa bağlamı
        material: Dict[str, Any] = {}
        if self.knowledge is not None:
            try:
                material = self.knowledge.answer_material(result.query, limit=limit)
            except Exception as exc:
                self._record_error("query.knowledge", exc)
                material = {}

        context = material.get("context", "")
        result.sources = list(material.get("sources", []))
        result.notes = [
            {"id": hit.get("id", ""), "title": hit.get("title", ""),
             "score": str(hit.get("score", 0.0))}
            for hit in (material.get("results") or [])
        ]
        result.gaps = list(material.get("gaps", []))
        result.context = context
        result.tokens = int(material.get("tokens_estimate", 0))

        # 2. Bellek bankası (geçmiş etkileşimler)
        #
        # ÖNEMLİ: Kasa aramasının aksine bellek, sistemin **kendi** ürettiği
        # yanıtları da içerir. Bunları bağlama katmak öz besleme döngüsü
        # yaratır (sistem kendi cevabını "bilgi" sanar ve aynı cümleleri
        # tekrarlar). Bu yüzden `query:` kaynaklı bellekler yalnızca kasa
        # sonuç vermediğinde devreye girer — o durumda konuşma sürekliliği
        # kasa tazeliğinden daha değerlidir.
        memory_text = ""
        if use_memory and self.memory is not None:
            try:
                if material.get("results"):
                    memory_text = self.memory.recall_text(
                        result.query, top_k=2, min_score=0.45,
                        exclude_source_prefixes=("query:",),
                    )
                else:
                    memory_text = self.memory.recall_text(
                        result.query, top_k=3, min_score=0.35
                    )
            except Exception as exc:
                logger.debug(f"Bellek geri çağırma atlandı: {exc}")

        if memory_text:
            context = (context + "\n\n" + memory_text) if context else memory_text

        # 3. Sentez
        synthesized = self._capability_synthesize(result.query, context)
        result.answer = synthesized.get("answer", "")
        if synthesized.get("sources") and not result.sources:
            result.sources = list(synthesized["sources"])
        result.tokens += int(synthesized.get("tokens", 0) or 0)

        # 4. Güven: kasa güveni ve yanıt uzunluğunun bileşimi
        base_confidence = float(material.get("confidence", 0.0) or 0.0)
        if not material.get("results"):
            base_confidence = 0.2
        length_factor = min(1.0, len(result.answer) / 400.0)
        result.confidence = round(min(0.95, 0.6 * base_confidence + 0.4 * length_factor), 4)

    def _answer_system_query(self, result: QueryResult) -> None:
        """Donanım/sistem sorgularını canlı verilerle yanıtlar."""
        result.mode = "system"
        blocks: List[str] = []

        try:
            if self.monitor is not None:
                summary = self.monitor.get_summary()
                blocks.append("**Canlı sistem durumu**")
                for key, value in summary.items():
                    blocks.append(f"- {key}: {value}")

            if self.os_layer is not None:
                try:
                    report = self.os_layer.system_report()
                    info = report.get("system") or report
                    cpu = info.get("cpu") or {}
                    memory = info.get("memory") or {}
                    gpu = info.get("gpu") or {}
                    blocks.append("\n**Envanter**")
                    if cpu:
                        model_name = (cpu.get("model") or cpu.get("brand")
                                      or cpu.get("name") or "bilinmiyor")
                        blocks.append(
                            f"- İşlemci: {model_name} "
                            f"({cpu.get('cores_physical', cpu.get('physical_cores', '?'))}C/"
                            f"{cpu.get('cores_logical', cpu.get('logical_cores', '?'))}T)"
                        )
                    if memory:
                        blocks.append(
                            f"- Bellek: {memory.get('total_mb', 0) / 1024:.1f} GB "
                            f"(boş {memory.get('available_mb', 0) / 1024:.1f} GB)"
                        )
                    if gpu:
                        blocks.append(f"- GPU: {gpu}")
                except Exception as exc:
                    logger.debug(f"Envanter alınamadı: {exc}")
        except Exception as exc:
            result.error = f"system_sorgu_hatasi: {exc}"

        if self.config is not None:
            blocks.append("\n**NeuroCore yapılandırması**")
            for key, value in self.config.get_summary().items():
                blocks.append(f"- {key}: {value}")

        if self._budget:
            blocks.append("\n**Kaynak bütçesi**")
            for key in ("device", "max_batch_size", "cache_size_mb", "num_workers",
                        "memory_pressure", "profile"):
                if key in self._budget:
                    blocks.append(f"- {key}: {self._budget[key]}")

        result.answer = "\n".join(blocks) if blocks else "Sistem bilgisi alınamadı."
        result.confidence = 0.95

    def _answer_deep(self, result: QueryResult, limit: int, learn: bool,
                     remember: bool, max_minutes: Optional[float],
                     start: float) -> QueryResult:
        """Derin mod: planlayıcı ile çok adımlı yürütme."""
        self._counter["deep_queries"] += 1
        result.mode = "deep"
        if max_minutes is None:
            max_minutes = float(getattr(self.config.planner, "max_reasoning_steps", 50)) / 25.0

        try:
            outcome = self.planner.run(
                result.query,
                context={
                    "limit": limit,
                    "engine": True,
                    "mode": "deep",
                },
                budget_seconds=max_minutes * 60.0,
            )
        except Exception as exc:
            self._record_error("query.deep", exc)
            result.error = f"planlayici_hatasi: {exc}"
            # Derin mod başarısız olursa standart yola düş (zarif bozulma)
            self._answer_standard(result, limit=limit)
            result.mode = "answer_fallback"
            result.duration_ms = (time.perf_counter() - start) * 1000
            self._finish_query(result, remember=remember, learn=learn)
            return result

        answer = outcome.get("answer")
        if isinstance(answer, dict):
            result.answer = str(answer.get("answer") or answer.get("thought") or "")
            sources = answer.get("sources")
            if isinstance(sources, list):
                result.sources = [str(source) for source in sources]
        else:
            result.answer = str(answer or "")

        result.confidence = float(outcome.get("confidence", 0.0) or 0.0)
        result.plan = {
            "state": outcome.get("state", ""),
            "summary": outcome.get("summary", ""),
            "tasks": len((outcome.get("graph") or {}).get("tasks") or {}),
            "replans": outcome.get("replans", 0),
            "completed": outcome.get("completed", 0),
            "failed": outcome.get("failed", 0),
            "progress": (outcome.get("graph_stats") or {}).get("progress", 0.0)
                        if isinstance(outcome.get("graph_stats"), dict) else 0.0,
        }
        trace_summary = outcome.get("trace_summary")
        if isinstance(trace_summary, dict):
            result.trace = trace_summary

        if not result.answer:
            # Planlayıcı yanıt üretemedi: standart yola düş
            self._answer_standard(result, limit=limit)
            result.mode = "deep_no_answer"
        if not result.confidence:
            result.confidence = 0.5 if result.answer else 0.15

        result.duration_ms = (time.perf_counter() - start) * 1000
        self._finish_query(result, remember=remember, learn=learn)
        return result

    def _finish_query(self, result: QueryResult, remember: bool = True,
                      learn: bool = True) -> None:
        """
        Sorgu sonrası boru hattı: değerlendirme → öğrenme → bellek → olay.

        Her adım ayrı korunur; biri başarısız olsa da yanıt kullanıcıya döner.
        """
        # ── 1. Kendi kendini değerlendirme ──
        quality = 0.0
        if self.evolution is not None:
            try:
                report = self.evolution.evaluator.evaluate(
                    result.query, result.answer,
                    context=[result.context] if result.context else None,
                    uncertainty=1.0 - result.confidence,
                    latency_ms=result.duration_ms,
                )
                quality = float(getattr(report, "score", 0.0) or 0.0)
                result.quality = quality
            except Exception as exc:
                logger.debug(f"Değerlendirme atlandı: {exc}")

        # ── 2. Öğrenme: yeterince iyiyse not üret ──
        if learn and result.mode in ("answer", "deep") and result.confidence >= 0.5 \
                and len(result.answer) > 200 and quality >= 0.45:
            try:
                title = self._derive_title(result.query)
                note = self.knowledge.write(
                    title=f"{title} — yanıt",
                    body=(
                        f"## Soru\n{result.query}\n\n"
                        f"## Yanıt\n{result.answer}\n\n"
                        f"## Kaynaklar\n"
                        + ("\n".join(f"- {s}" for s in result.sources) or "- (kaynak yok)")
                    ),
                    tags=self._tags_from_query(result.query),
                    confidence=round(min(0.9, result.confidence), 3),
                    importance=round(min(0.9, quality + 0.2), 3),
                    # `qa` türü, sistemin ürettiği yanıt notlarını işaretler;
                    # bunlar denetim izi olarak saklanır ama `context_exclude_kinds`
                    # sayesinde sonraki yanıtların bağlamına **girmez** (öz besleme
                    # döngüsünü önler).
                    kind="qa",
                    sources=[s.split(" (")[0].strip("[]\\") for s in result.sources[:5]],
                    reason="sorgu yanıtı", author="engine",
                )
                result.learned_note_id = note.id
                self._counter["learnings"] += 1
            except Exception as exc:
                logger.debug(f"Yanıt notu yazılamadı: {exc}")

        # ── 3. Bellek + deneyim ──
        if remember:
            if self.memory is not None and result.answer:
                try:
                    self.memory.remember(
                        f"Soru: {result.query}\nYanıt: {result.answer[:1200]}",
                        source=f"query:{result.mode}",
                        tags=self._tags_from_query(result.query),
                        importance=round(min(1.0, 0.3 + result.confidence * 0.5), 3),
                        meta={"note_id": result.learned_note_id,
                              "quality": quality, "mode": result.mode},
                    )
                except Exception as exc:
                    logger.debug(f"Bellek yazımı atlandı: {exc}")

            if self.evolution is not None:
                try:
                    self.evolution.observe(
                        result.query, result.answer,
                        context=[note["id"] for note in result.notes][:8],
                        latency_ms=result.duration_ms,
                        uncertainty=1.0 - result.confidence,
                        strategy=result.mode,
                        tags=self._tags_from_query(result.query),
                        hardware={
                            "profile": self.config.hardware_profile,
                            "device": self.config.device,
                            "pressure": self._budget.get("memory_pressure", "NORMAL"),
                        },
                    )
                except Exception as exc:
                    logger.debug(f"Deneyim kaydı atlandı: {exc}")

        self._query_history.append({
            "at": time.time(), "query": result.query[:200], "mode": result.mode,
            "confidence": result.confidence, "quality": quality,
            "duration_ms": result.duration_ms,
        })
        if len(self._query_history) > 500:
            self._query_history = self._query_history[-500:]

        self.event_bus.emit("engine.query_answered", {
            "query": result.query[:200], "mode": result.mode,
            "confidence": result.confidence, "quality": quality,
            "sources": len(result.sources), "duration_ms": result.duration_ms,
            "learned_note": result.learned_note_id,
        }, source="core.engine")

    # ═══════════════════════════════════════════════════════════════════
    #  ÖĞRENME / ARAŞTIRMA / EVRİM
    # ═══════════════════════════════════════════════════════════════════

    def learn(
        self,
        text: str,
        *,
        title: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        kind: str = "concept",
        confidence: float = 0.6,
        source: str = "",
        split: bool = True,
    ) -> Dict[str, Any]:
        """
        Sisteme bilgi öğretir (kasaya yazar, indeksler, grafı örer).

        Args:
            text: Öğretilecek içerik (markdown veya düz metin)
            title: Başlık (None = içerikten türetilir)
            tags: Etiketler
            kind: Not türü
            confidence: Güven (0..1)
            source: Kaynak adresi/etiketi
            split: Uzun içeriği parçalara böl

        Returns:
            `IngestResult.to_dict()`
        """
        if self.knowledge is None:
            return {"ok": False, "error": "bilgi_katmani_yok"}
        try:
            outcome = self.knowledge.ingest(
                text, title=title, kind=kind, tags=tags,
                confidence=confidence, source=source, split=split,
                reason="kullanıcı öğretmesi", author="user",
            )
            payload = outcome.to_dict()
            payload["ok"] = not outcome.errors
            self._counter["learnings"] += 1

            # Belleğe de yaz (hızlı geri çağırma için)
            if self.memory is not None and outcome.note_id:
                try:
                    self.memory.remember(
                        text[:4000], source=f"vault:{outcome.note_id}",
                        tags=list(tags or []), importance=min(1.0, confidence + 0.2),
                        meta={"note_id": outcome.note_id, "title": outcome.title},
                    )
                except Exception as exc:
                    logger.debug(f"Bellek yazımı atlandı: {exc}")

            return payload
        except Exception as exc:
            self._record_error("learn", exc)
            return {"ok": False, "error": str(exc)}

    def research(
        self,
        topic: str,
        *,
        depth: int = 2,
        max_notes: int = 5,
        learn: bool = True,
    ) -> Dict[str, Any]:
        """
        Bir konuyu çok adımlı olarak araştırır ve öğrenir.

        Akış:
            1. Konuyu dil katmanıyla anahtar kelimelere ayır
            2. Her anahtar için kasa + (varsa) çevrimiçi araştırma
            3. Bulguları sentezle ve çelişkileri işaretle
            4. Yeni bilgiyi kasaya not olarak yaz
            5. Evrim katmanına araştırma boşluklarını bildir

        Args:
            topic: Araştırma konusu
            depth: Kaç tur derinleşme (1 = tek tur)
            max_notes: Her turda yazılacak maksimum not
            learn: Bulguları kasaya yaz

        Returns:
            {"topic", "findings", "notes_written", "rounds", "gaps", "duration_ms"}
        """
        start = time.perf_counter()
        self._counter["researches"] += 1
        report: Dict[str, Any] = {
            "topic": topic, "rounds": 0, "findings": [],
            "notes_written": [], "gaps": [], "duration_ms": 0.0,
        }
        if not topic.strip():
            report["error"] = "boş_konu"
            return report

        # Evrim katmanından araştırma önerisi al
        if self.evolution is not None:
            try:
                plan = self.evolution.research_plan(top_k=depth * 3)
                if plan:
                    report["curiosity_plan"] = plan[:depth * 2]
            except Exception as exc:
                logger.debug(f"Merak planı alınamadı: {exc}")

        # Derinlik: konu → alt konular
        queue: List[Tuple[str, int]] = [(topic, 0)]
        visited: set = set()

        while queue and report["rounds"] < max(1, depth):
            current, level = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            report["rounds"] += 1

            # 1. Kasa taraması
            vault_hits: List[Dict[str, Any]] = []
            if self.knowledge is not None:
                try:
                    for hit in self.knowledge.search(current, limit=6):
                        vault_hits.append({
                            "title": hit.title, "id": hit.note_id,
                            "score": hit.score, "excerpt": hit.excerpt,
                            "origin": "vault",
                        })
                except Exception as exc:
                    logger.debug(f"Kasa taraması başarısız: {exc}")

            # 2. Harici araştırma (çevrimdışıysa kasayla sınırlı)
            external = self._web_research_hook(current, max_results=5)
            for item in external:
                if item.get("source") != "vault":
                    vault_hits.append({**item, "origin": item.get("source", "external")})

            report["findings"].extend(vault_hits[:8])

            # 3. Sentez
            context = "\n\n".join(
                f"[{item.get('origin')}] {item.get('title')}\n{item.get('snippet', item.get('excerpt', ''))}"
                for item in vault_hits[:8]
            )
            if context and learn and self.knowledge is not None:
                synthesized = self._capability_synthesize(current, context)
                answer = synthesized.get("answer", "")
                if len(answer) > 120:
                    try:
                        note = self.knowledge.write(
                            title=f"Araştırma: {current}",
                            body=(
                                f"## Özet\n{synthesized.get('answer', '')[:4000]}\n\n"
                                f"## Bulgular\n"
                                + "\n".join(
                                    f"- {item.get('title')} ({item.get('origin')}, "
                                    f"skor {item.get('score', 0):.2f})"
                                    for item in vault_hits[:10]
                                )
                                + f"\n\n## Kaynaklar\n"
                                + "\n".join(
                                    f"- {item.get('url', item.get('id', '?'))}"
                                    for item in vault_hits[:10]
                                )
                            ),
                            tags=self._tags_from_query(current),
                            confidence=0.55,
                            kind="research",
                            reason="araştırma turu", author="engine",
                        )
                        report["notes_written"].append({"id": note.id, "title": note.title})
                        self._counter["learnings"] += 1
                    except Exception as exc:
                        logger.debug(f"Araştırma notu yazılamadı: {exc}")

            # 4. Alt konular (derinleşme)
            if level + 1 < depth and self.knowledge is not None:
                for hit in vault_hits[:3]:
                    related = hit.get("title", "")
                    if related and related not in visited:
                        queue.append((related, level + 1))

        # 5. Boşluk analizi
        if self.evolution is not None:
            try:
                gaps = self.evolution.research_plan(top_k=5)
                report["gaps"] = [item.get("topic", "") for item in gaps if item.get("topic")]
            except Exception as exc:
                logger.debug(f"Boşluk analizi atlandı: {exc}")

        report["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        report["ok"] = bool(report["findings"]) or bool(report["notes_written"])
        self.event_bus.emit("engine.research_done", {
            "topic": topic, "rounds": report["rounds"],
            "findings": len(report["findings"]),
            "notes": len(report["notes_written"]),
        }, source="core.engine")
        logger.info(
            f"Araştırma tamamlandı: '{topic}' — {report['rounds']} tur, "
            f"{len(report['findings'])} bulgu, {len(report['notes_written'])} not"
        )
        return report

    def evolve(self, cycles: int = 1, *, dry_run: bool = False, train: bool = True) -> Dict[str, Any]:
        """
        Evrim turu çalıştırır ve gerekiyorsa eğitim yapar.

        Args:
            cycles: Kaç evrim turu
            dry_run: Yalnızca planla, uygulama
            train: Eğitim planı varsa eğitimi de çalıştır

        Returns:
            {"cycles", "reports", "training", "growth", "duration_ms"}
        """
        start = time.perf_counter()
        self._counter["evolutions"] += 1
        outcome: Dict[str, Any] = {
            "cycles": cycles, "reports": [], "training": None,
            "growth": None, "duration_ms": 0.0,
        }

        if self.evolution is None:
            outcome["error"] = "evrim_katmani_yok"
            return outcome

        # ── Nezaket kapısı ──────────────────────────────────────────────
        # Hafif evrim adımları (denetim, müfredat, sentez) her zaman çalışır;
        # **eğitim** yalnızca kaynaklar uygun olduğunda yapılır. Kullanıcı
        # sistemi kullanırken ağır eğitim başlatmak bilgisayarı kilitler.
        heavy_ok, heavy_reason = self.resources_available(for_heavy_work=True)
        if train and not heavy_ok:
            train = False
            outcome["training_skipped"] = heavy_reason
            logger.info(f"Evrim eğitimi ertelendi — {heavy_reason}")

        # 1. Evrim döngüsü
        try:
            report = self.evolution.cycle(cycles=max(1, int(cycles)), dry_run=dry_run,
                                          auto_apply=not dry_run)
            outcome["reports"].append(report.to_dict() if hasattr(report, "to_dict") else {})
        except Exception as exc:
            self._record_error("evolve.cycle", exc)
            outcome["error"] = str(exc)

        # 2. Büyüme kararı
        try:
            metrics = self._growth_metrics()
            decision = self.growth.decide(metrics) if self.growth is not None else None
            if decision is not None:
                applied = self.growth.apply(decision, self.model, dry_run=dry_run)
                outcome["growth"] = {"decision": decision.to_dict(), "applied": applied}
        except Exception as exc:
            self._record_error("evolve.growth", exc)

        # 3. Eğitim
        if train and not dry_run and self.trainer is not None and self.model is not None:
            try:
                batch = self.evolution.experiences.sample_batch(
                    batch_size=64, embedding_dim=self.embedding.dim if self.embedding else 768
                )
                curriculum = None
                try:
                    items = self.evolution.curriculum.build(
                        self.evolution.experiences, max_items=200
                    )
                    curriculum = self.evolution.curriculum.to_training_pairs(items)
                except Exception as exc:
                    logger.debug(f"Müfredat üretilemedi: {exc}")

                payload = curriculum if curriculum and curriculum.get("prompts") else batch
                if payload and payload.get("prompts"):
                    dataset = self.trainer.build_dataset_from_experiences(
                        payload, self.model.tokenizer,
                        seq_len=min(256, int(self._budget.get("max_batch_size", 32) or 32) * 8),
                    )
                    # Kasa bilgisini de karıştır (deneyim azsa model kasa üslubunu öğrensin)
                    if self.knowledge is not None and len(dataset) < 60:
                        vault_dataset = self.trainer.build_dataset_from_vault(
                            self.knowledge.vault, self.model.tokenizer,
                            seq_len=256, max_notes=200,
                        )
                        for sample in vault_dataset.samples:
                            dataset.add(sample.prompt, sample.target, sample.weight * 0.6,
                                        sample.tags, sample.difficulty)

                    if not dataset.is_empty:
                        plan = self.trainer.plan_training(
                            estimated_samples=len(dataset), model=self.model,
                            rationale="evrim turu",
                        )
                        plan.steps = min(plan.steps, 150 if not self.model.is_pretrained else 60)
                        result = self.trainer.train(self.model, dataset, plan=plan)
                        if result.steps_completed > 0 and not self.model.is_pretrained:
                            self.model._is_pretrained = True
                            if self.embedding is not None:
                                self.embedding.attach_model(self.model.backbone,
                                                            self.model.tokenizer)
                            self.model.save()
                        outcome["training"] = result.to_dict()
            except Exception as exc:
                self._record_error("evolve.train", exc)
                outcome["training"] = {"error": str(exc)}

        outcome["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            f"Evrim turu tamamlandı: {cycles} döngü, "
            f"eğitim: {'var' if outcome.get('training') else 'yok'} "
            f"({outcome['duration_ms']:.0f}ms)"
        )
        return outcome

    def feedback(self, positive: bool = True, note: str = "",
                 reward: Optional[float] = None) -> Dict[str, Any]:
        """
        Son sorguya geri bildirim verir (öğrenme sinyali).

        Args:
            positive: Olumlu mu?
            note: Açıklama
            reward: -1..1 arası açık ödül (None = otomatik)

        Returns:
            {"recorded": bool, "feedback": str, "reward": float}
        """
        if self.evolution is None or not self._query_history:
            return {"recorded": False, "error": "deneyim_yok"}

        try:
            experiences = self.evolution.experiences.recent(1)
            if not experiences:
                return {"recorded": False, "error": "deneyim_bulunamadi"}
            experience = experiences[0]
            signal = "explicit_positive" if positive else "explicit_negative"
            resolved_reward = reward
            if resolved_reward is None:
                resolved_reward = 0.8 if positive else -0.6
            self.evolution.feedback(experience.id, signal, reward=resolved_reward,
                                    note=note)
            # Bellekteki ilgili kayıtları da güncelle
            if self.memory is not None:
                try:
                    self.memory.feedback_by_source(
                        f"query:{experience.strategy or 'answer'}",
                        0.9 if positive else 0.2,
                    )
                except Exception as exc:
                    logger.debug(f"Bellek geri bildirimi atlandı: {exc}")
            return {"recorded": True, "feedback": signal,
                    "reward": float(resolved_reward), "experience_id": experience.id}
        except Exception as exc:
            self._record_error("feedback", exc)
            return {"recorded": False, "error": str(exc)}

    def train(
        self,
        *,
        steps: Optional[int] = None,
        from_vault: bool = True,
        from_experiences: bool = True,
        max_minutes: Optional[float] = None,
        user_initiated: bool = False,
    ) -> Dict[str, Any]:
        """
        Modeli eğitir (kasa bilgisi + deneyimler).

        Args:
            steps: Adım sayısı (None = otomatik plan)
            from_vault: Kasadaki notları kullan
            from_experiences: Deneyim tamponunu kullan
            max_minutes: Zaman sınırı

        Returns:
            Eğitim sonucu sözlüğü
        """
        if self.trainer is None or self.model is None:
            return {"ok": False, "error": "egitici_yok"}

        # ── Nezaket kapısı: kullanıcı sistemi kullanıyorsa hiç başlama ──
        available, reason = self.resources_available(for_heavy_work=True)
        if not available:
            logger.info(f"Eğitim ertelendi — {reason}")
            self.event_bus.emit("engine.work_deferred", {
                "work": "train", "reason": reason,
            }, source="core.engine")
            return {
                "ok": False, "error": "kaynaklar_dolu", "reason": reason,
                "hint": "Kullanıcı sistemi kullanıyor; kaynaklar boşaldığında "
                        "tekrar deneyin veya policy faktörünü düşürün.",
                "resources": self.resource_report()["headroom"],
            }

        seq_len = min(256, int(self._budget.get("max_batch_size", 32) or 32) * 8)
        dataset = None

        if from_vault and self.knowledge is not None:
            try:
                dataset = self.trainer.build_dataset_from_vault(
                    self.knowledge.vault, self.model.tokenizer,
                    seq_len=seq_len, max_notes=600,
                )
            except Exception as exc:
                logger.warning(f"Kasa veri kümesi üretilemedi: {exc}")

        if from_experiences and self.evolution is not None:
            try:
                items = self.evolution.curriculum.build(
                    self.evolution.experiences, max_items=300
                )
                pairs = self.evolution.curriculum.to_training_pairs(items) \
                    if items else {}
                if pairs.get("prompts"):
                    experience_dataset = self.trainer.build_dataset_from_experiences(
                        pairs, self.model.tokenizer, seq_len=seq_len
                    )
                    if dataset is None:
                        dataset = experience_dataset
                    else:
                        for sample in experience_dataset.samples:
                            dataset.add(sample.prompt, sample.target, sample.weight * 1.5,
                                        sample.tags, sample.difficulty)
            except Exception as exc:
                logger.debug(f"Deneyim veri kümesi üretilemedi: {exc}")

        if dataset is None or dataset.is_empty:
            return {"ok": False, "error": "veri_yok",
                    "hint": "learn() ile bilgi ekleyin veya query() ile etkileşim üretin"}

        plan = self.trainer.plan_training(
            estimated_samples=len(dataset), model=self.model,
            steps=steps, rationale="motor eğitim isteği",
        )
        train_set, valid_set = dataset.split(validation_ratio=0.1)
        result = self.trainer.train(
            self.model, train_set if not train_set.is_empty else dataset,
            plan=plan,
            validation=valid_set if not valid_set.is_empty else None,
            max_minutes=max_minutes,
            user_initiated=user_initiated,
        )

        if result.steps_completed > 0:
            self.model._is_pretrained = True
            if self.embedding is not None:
                self.embedding.attach_model(self.model.backbone, self.model.tokenizer)
            self.model.save()

        payload = result.to_dict()
        payload["ok"] = result.steps_completed > 0
        payload["dataset"] = dataset.stats()
        if not valid_set.is_empty:
            payload["validation"] = self.trainer.evaluate(self.model, valid_set)
        return payload

    # ═══════════════════════════════════════════════════════════════════
    #  BAKIM VE SAĞLIK
    # ═══════════════════════════════════════════════════════════════════

    def resource_report(self) -> Dict[str, Any]:
        """
        Kaynak nezaketi durumunun tam raporu.

        Kullanıcı "şu an sistemi ne kadar kullanıyorsun?" sorusunun yanıtı
        burada: anlık kullanım, ajanın bütçesi ve duraklama kararı.
        """
        self._refresh_budget(force=True)
        headroom = dict(self._budget.get("headroom", {}) or {})
        policy = getattr(self.config.hardware, "headroom", None)
        return {
            "headroom": headroom,
            "budget": {
                "device": self._budget.get("device"),
                "max_batch_size": self._budget.get("max_batch_size"),
                "num_workers": self._budget.get("num_workers"),
                "cache_size_mb": self._budget.get("cache_size_mb"),
                "max_ram_mb": self._budget.get("max_ram_mb"),
                "max_gpu_mb": self._budget.get("max_gpu_mb"),
                "profile": self._budget.get("profile"),
                "memory_pressure": self._budget.get("memory_pressure"),
            },
            "policy": policy.to_dict() if policy is not None else {},
            "summary": (
                f"CPU %{headroom.get('cpu_percent', 0):.0f} dolu → "
                f"bütçe %{headroom.get('cpu_budget_percent', 0):.0f} · "
                f"RAM %{headroom.get('ram_percent', 0):.0f} dolu → "
                f"bütçe %{headroom.get('ram_budget_percent', 0):.0f} "
                f"({self._budget.get('max_ram_mb', 0):.0f}MB) · "
                f"VRAM %{headroom.get('gpu_percent', 0):.0f} dolu → "
                f"bütçe %{headroom.get('gpu_budget_percent', 0):.0f} "
                f"({self._budget.get('max_gpu_mb', 0):.0f}MB) · "
                f"{headroom.get('pressure', 'NORMAL')}"
            ),
        }

    def _own_vram_percent(self) -> float:
        """
        **Kendi** VRAM kullanımımızı yüzde olarak döndürür.

        ⚠ Neden gerekli? Süreç modeli VRAM'e yükler. Nezaket kapısı ham
        VRAM kullanımına bakarsa **kendi** payımızı "kullanıcı sistemi
        kullanıyor" sanır ve ağır işi reddeder. Ölçüldü: uygulama açıkken
        taze süreçle eğitim **hiç başlamadı** —

            {"error": "kaynaklar_dolu",
             "reason": "sistem kaynakları dolu: VRAM %98 ≥ %90",
             "resources": {"gpu_free_mb": 215.0}}

        Oysa 215 MB boş VRAM'in ~2 GB'ı bizim modelimizdi. Eğiticinin
        içindeki koruma bu payı düşüyor (`own_gpu_mb`); ön kapı da
        düşmelidir, aksi hâlde iki katman çelişir.

        Returns:
            Kendi VRAM payımız (0–100)
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return 0.0
            _free, _total = torch.cuda.mem_get_info()
            toplam_mb = float(_total) / 1024**2
            if toplam_mb <= 0:
                return 0.0
            kendi_mb = float(torch.cuda.memory_allocated()) / 1024**2
            # Model henüz CUDA'ya taşınmadıysa **parametre sayısından**
            # tahmin et. Ölçüldü: eğitim öncesi `memory_allocated()` 0
            # döner; tahmin olmadan kapı kendi modelini yabancı yük sanıp
            # eğitimi reddeder.
            if kendi_mb <= 0.0 and self.model is not None:
                try:
                    # Eğitim belleği ≈ parametre × 4 bayt × 2,2
                    # (gradyan + Adam'ın iki momenti + etkinleştirmeler)
                    adet = float(self.model.parameter_count())
                    kendi_mb = adet * 4.0 * 2.2 / 1024**2
                except Exception:
                    kendi_mb = 0.0
            if kendi_mb <= 0.0:
                try:
                    mem = getattr(self, "_memory_profile", None) or {}
                    kendi_mb = float(mem.get("training_adam_mb", 0.0) or 0.0)
                except Exception:
                    kendi_mb = 0.0
            return min(100.0, kendi_mb / toplam_mb * 100.0)
        except Exception:
            return 0.0

    def resources_available(self, *, for_heavy_work: bool = True,
                            exclude_own_vram: bool = True) -> Tuple[bool, str]:
        """
        Ağır iş başlatılabilir mi?

        "Boştakinin %90'ı" politikası burada uygulanır: kullanıcı sistemi
        yoğun kullanıyorsa ajan yeni ağır iş **başlatmaz** (eğitim, toplu
        indeksleme, derin araştırma).

        Args:
            for_heavy_work: Ağır iş için mi kontrol ediliyor?
            exclude_own_vram: Kendi VRAM payımız düşülsün mü? (varsayılan
                açık — aksi hâlde süreç kendi modelini "yabancı yük" sanar)

        Returns:
            (uygun, gerekçe)
        """
        if not getattr(self.config.hardware, "enforce_headroom", True):
            return True, "nezaket politikası devre dışı"
        self._refresh_budget()
        headroom = dict(self._budget.get("headroom", {}) or {})

        # ── Kendi VRAM payımızı düş ──
        if exclude_own_vram:
            kendi = self._own_vram_percent()
            if kendi > 0.5:
                ham = float(headroom.get("gpu_percent", 0.0) or 0.0)
                duzeltilmis = max(0.0, ham - kendi)
                if duzeltilmis < ham:
                    logger.debug(
                        f"VRAM düzeltmesi: ham %{ham:.1f} − kendi %{kendi:.1f} "
                        f"= %{duzeltilmis:.1f}"
                    )
                    headroom["gpu_percent_raw"] = ham
                    headroom["gpu_percent"] = duzeltilmis
                    headroom["own_vram_percent"] = kendi
                    # Eşik yeniden değerlendirilsin
                    if duzeltilmis < 90.0:
                        headroom["should_stall"] = False
                        headroom["stall_reason"] = ""
                    if str(headroom.get("pressure")) == "CRITICAL" \
                            and duzeltilmis < 85.0:
                        headroom["pressure"] = "HIGH"

        if headroom.get("should_stall"):
            return False, str(headroom.get("stall_reason") or "kaynaklar dolu")
        if for_heavy_work:
            level = str(headroom.get("pressure", "NORMAL"))
            if level == "CRITICAL":
                return False, f"kaynak baskısı CRITICAL ({headroom.get('stall_reason', '')})"
            if level == "HIGH":
                return False, (
                    f"kaynak baskısı HIGH — CPU %{headroom.get('cpu_percent', 0):.0f}, "
                    f"RAM %{headroom.get('ram_percent', 0):.0f}, "
                    f"VRAM %{headroom.get('gpu_percent', 0):.0f}"
                )
        return True, ""

    def _maintenance_loop(self, interval: float) -> None:
        """Periyodik bakım döngüsü (arka plan thread'i)."""
        logger.debug(f"Bakım döngüsü başladı ({interval:.0f}s)")
        while not self._maintenance_stop.wait(timeout=max(30.0, interval)):
            try:
                if self.state not in (EngineState.RUNNING, EngineState.DEGRADED):
                    continue
                self.maintain(deep=False)
            except Exception as exc:
                logger.warning(f"Bakım turu hatası: {exc}")
        logger.debug("Bakım döngüsü durdu")

    def maintain(self, *, deep: bool = False, reindex: bool = False,
                 evolute: bool = False) -> Dict[str, Any]:
        """
        Bakım turu: kaynak tazeleme, bellek sıkıştırma, bütünlük denetimi.

        Args:
            deep: Ağır bakım (bütünlük taraması, yeniden indeksleme)
            reindex: İndeks ve grafı yeniden kur
            evolute: Evrim turu da çalıştır

        Returns:
            Bakım raporu sözlüğü
        """
        start = time.perf_counter()
        self._counter["maintenance_rounds"] += 1
        report: Dict[str, Any] = {"actions": [], "errors": []}

        # 1. Kaynak bütçesi tazele
        try:
            self._refresh_budget(force=True)
            report["budget"] = dict(self._budget)
            report["actions"].append("bütçe tazelendi")
        except Exception as exc:
            report["errors"].append(f"bütçe: {exc}")

        # 2. Bilgi katmanı bakımı
        if self.knowledge is not None:
            try:
                knowledge_report = self.knowledge.maintain(
                    fix=False, embeddings=True, reindex=reindex, limit=1000,
                )
                report["knowledge"] = knowledge_report.to_dict()
                report["actions"].append("bilgi katmanı bakımı")
            except Exception as exc:
                report["errors"].append(f"bilgi: {exc}")

        # 3. Bellek sıkıştırma + temizlik
        if self.memory is not None:
            try:
                merged = self.memory.consolidate(min_similarity=0.88)
                pruned = self.memory.prune(max_age_days=120.0, min_utility=0.05)
                report["memory"] = {"merged": merged.get("merged", 0), "pruned": pruned,
                                    "size": self.memory.size}
                if merged.get("merged") or pruned:
                    report["actions"].append(
                        f"bellek: {merged.get('merged', 0)} birleştirme, {pruned} temizlik"
                    )
            except Exception as exc:
                report["errors"].append(f"bellek: {exc}")

        # 4. Model durumu
        if self.model is not None:
            try:
                report["model"] = {
                    "pretrained": self.model.is_pretrained,
                    "maturity": self.model.growth.maturity,
                    "generation": self.model.growth.generation,
                    "routing": self.model.routing_report(),
                }
            except Exception as exc:
                report["errors"].append(f"model: {exc}")

        # 5. Donanım sağlığı
        if self.monitor is not None:
            try:
                snapshot = self.monitor.get_snapshot()
                report["hardware"] = {
                    "cpu_percent": round(snapshot.cpu_percent, 1),
                    "ram_percent": round(snapshot.ram_percent, 1),
                    "ram_free_mb": round(snapshot.ram_free_mb, 0),
                    "gpu_utilization": round(snapshot.gpu_utilization, 1),
                    "vram_percent": round(snapshot.gpu_vram_percent, 1),
                    "gpu_temperature_c": snapshot.gpu_temperature_c,
                }
            except Exception as exc:
                report["errors"].append(f"donanım: {exc}")

        # 6. Evrim turu (isteğe bağlı)
        if evolute and self.evolution is not None:
            try:
                report["evolution"] = self.evolve(cycles=1, train=False)
                report["actions"].append("evrim turu")
            except Exception as exc:
                report["errors"].append(f"evrim: {exc}")

        # 7. Sağlık tazeleme
        self.refresh_health()
        report["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        report["ok"] = not report["errors"]
        self._emit("engine.maintained", {
            "actions": report["actions"], "errors": report["errors"],
            "duration_ms": report["duration_ms"],
        })
        logger.info(
            f"Bakım tamamlandı ({report['duration_ms']:.0f}ms): "
            f"{', '.join(report['actions']) or 'işlem yok'}"
            + (f" | hatalar: {len(report['errors'])}" if report["errors"] else "")
        )
        return report

    def health_report(self, deep: bool = False) -> HealthReport:
        """
        Bütünsel sağlık raporu üretir.

        Args:
            deep: Katmanlardan ayrıntılı sağlık bilgisi iste (yavaş)
        """
        report = HealthReport(
            state=self.state.value,
            profile=self.config.hardware_profile,
            device=self.config.device,
            uptime_seconds=(time.time() - self._started_at) if self._started_at else 0.0,
            counters=dict(self._counter),
        )

        checks: List[Tuple[str, Any, bool]] = [
            ("hardware", self.monitor, False),
            ("knowledge", self.knowledge, True),
            ("neural", self.model, False),
            ("cache", self.memory, False),
        ]
        for name, layer, critical in checks:
            if layer is None:
                report.layers[name] = {"ok": False, "error": "kurulmadı",
                                       "critical": critical}
                if critical:
                    report.critical.append(f"{name} katmanı kurulmadı")
                continue
            try:
                if deep and hasattr(layer, "health"):
                    info = layer.health()
                    if hasattr(info, "to_dict"):
                        info = info.to_dict()
                else:
                    info = self._layer_health.get(name, {"ok": True})
                report.layers[name] = info
            except Exception as exc:
                report.layers[name] = {"ok": False, "error": str(exc)}

        # İsteğe bağlı katmanlar
        for name, layer in (("multimodal", self.multimodal), ("os", self.os_layer),
                            ("language", self.language), ("planner", self.planner),
                            ("evolution", self.evolution), ("trainer", self.trainer),
                            ("growth", self.growth), ("embedding", self.embedding)):
            if layer is None:
                report.layers[name] = {"ok": False, "optional": True,
                                       "error": "kurulmadı"}
                continue
            try:
                if deep and hasattr(layer, "health"):
                    info = layer.health()
                    report.layers[name] = info.to_dict() if hasattr(info, "to_dict") else info
                else:
                    report.layers[name] = self._layer_health.get(name, {"ok": True})
            except Exception as exc:
                report.layers[name] = {"ok": False, "error": str(exc)}

        # Sorun toplama
        for name, info in report.layers.items():
            if not isinstance(info, dict):
                continue
            if not info.get("ok", True):
                message = f"{name}: {info.get('error', 'sağlıksız')}"
                if info.get("critical"):
                    report.critical.append(message)
                else:
                    report.issues.append(message)
            for issue in (info.get("issues") or []):
                report.issues.append(f"{name}: {issue}")
            for problem in (info.get("critical") or []):
                report.critical.append(f"{name}: {problem}")

        report.issues.extend(f"{layer}: {error}" for layer, error in self._init_errors.items())
        report.ok = not report.critical and self.state not in (EngineState.FAILED,
                                                              EngineState.STOPPED)
        return report

    def refresh_health(self) -> None:
        """Katman sağlık önbelleğini tazeler."""
        self._layer_health["engine"] = {
            "ok": self.state not in (EngineState.FAILED, EngineState.STOPPED),
            "state": self.state.value,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Motorun tam istatistik paketi."""
        stats: Dict[str, Any] = {
            "version": self.VERSION,
            "state": self.state.value,
            "profile": self.config.hardware_profile,
            "device": self.config.device,
            "uptime_seconds": round((time.time() - self._started_at), 1)
                              if self._started_at else 0.0,
            "counters": dict(self._counter),
            "init_errors": dict(self._init_errors),
            "budget": dict(self._budget),
            "event_bus": self.event_bus.get_stats(),
            "recent_queries": self._query_history[-10:],
        }
        for name, layer in (
            ("hardware", self.monitor), ("allocator", self.allocator),
            ("profiler", self.profiler), ("knowledge", self.knowledge),
            ("model", self.model), ("embedding", self.embedding),
            ("memory", self.memory), ("growth", self.growth),
            ("trainer", self.trainer), ("multimodal", self.multimodal),
            ("os", self.os_layer), ("language", self.language),
            ("planner", self.planner), ("evolution", self.evolution),
        ):
            if layer is None:
                continue
            try:
                stats[name] = layer.get_stats()
            except Exception as exc:
                stats[name] = {"error": str(exc)}
        return stats

    # ═══════════════════════════════════════════════════════════════════
    #  İÇ YARDIMCILAR
    # ═══════════════════════════════════════════════════════════════════

    def bootstrap(
        self,
        *,
        seed_knowledge: bool = True,
        train_tokenizer: bool = True,
        target_vocab: int = 4000,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        İlk çalıştırma hazırlığı: tokenizer eğitimi + başlangıç bilgisi.

        Yeni bir kurulumda sözlük yalnızca 271 token'dır (256 bayt + özel
        token) — bu, modelin Türkçe metni çok fazla token'a bölmesine yol
        açar. Bu metot, kasadaki (veya tohum metinlerdeki) içerikle BPE
        tokenizer'ı eğitir ve sözlüğü hedef boyuta çıkarır.

        Args:
            seed_knowledge: Kasa boşsa sistem bilgisi tohum notlarını ekle
            train_tokenizer: Tokenizer'ı eğit (sözlük küçükse)
            target_vocab: Hedef sözlük boyutu
            force: Daha önce yapılmış olsa da tekrar çalıştır

        Returns:
            {"seeded_notes": int, "tokenizer": {...}, "vocab_size": int}
        """
        outcome: Dict[str, Any] = {"seeded_notes": 0, "tokenizer": {}, "ok": True}
        if self.state in (EngineState.STOPPED, EngineState.FAILED):
            return {"ok": False, "error": f"motor_durumu: {self.state.value}"}

        # ── 1. Tohum bilgi: sistem envanteri + yetenekler ──
        if seed_knowledge and self.knowledge is not None:
            try:
                existing = len(self.knowledge.vault._path_index)
                if existing <= 1 or force:
                    outcome["seeded_notes"] = self._seed_knowledge()
            except Exception as exc:
                outcome["seed_error"] = str(exc)

        # ── 2. Tokenizer eğitimi ──
        if train_tokenizer and self.model is not None and self.model.tokenizer is not None:
            try:
                current_vocab = self.model.tokenizer.vocab_size
                if current_vocab <= 300 or force:
                    texts = self._bootstrap_corpus()
                    if texts:
                        result = self.model.tokenizer.train(
                            texts, vocab_size=target_vocab, min_frequency=2,
                        )
                        self.model.tokenizer.save_default()

                        # Sözlük değişti → omurga **yeni sözlük boyutuyla**
                        # yeniden kurulur. (Eski sözlükle kurulmuş gömme
                        # matrisi yeni token id'lerini indeksleyemez.)
                        new_vocab = int(self.model.tokenizer.vocab_size)
                        self.model.model_config.vocab_size = new_vocab
                        from neurocore.neural.transformer import TransformerBackbone
                        self.model.backbone = TransformerBackbone(self.model.model_config)
                        self.model._is_pretrained = False
                        # Yeni omurga CPU'da doğar; yapılandırılmış cihaza geri taşı.
                        resolved_device = self.model.ensure_device()
                        if self.embedding is not None:
                            self.embedding.attach_tokenizer(self.model.tokenizer)
                        outcome["tokenizer"] = result
                        outcome["device"] = resolved_device
                        logger.info(
                            f"Tokenizer eğitildi: {new_vocab} token "
                            f"({len(texts)} metin) — omurga yeniden kuruldu "
                            f"({self.model.model_config.num_layers} katman, cihaz {resolved_device})"
                        )
            except Exception as exc:
                outcome["tokenizer_error"] = str(exc)
                logger.warning(f"Tokenizer eğitimi başarısız: {exc}")

        outcome["vocab_size"] = (
            self.model.tokenizer.vocab_size if self.model and self.model.tokenizer else 0
        )
        self._emit("engine.bootstrapped", outcome)
        return outcome

    def _seed_knowledge(self) -> int:
        """
        Kasa boşken sisteme öz-bilgi tohumlar.

        Amaç: sistemin kendi donanımını ve yeteneklerini **bilgi olarak**
        taşıması; böylece "sistemim ne yapabilir?" sorusu kasaya dayanarak
        yanıtlanabilir ve model bu üslupla eğitilebilir.
        """
        if self.knowledge is None:
            return 0
        written = 0
        try:
            summary = self.monitor.get_summary() if self.monitor else {}
            config_summary = self.config.get_summary()

            # 1. Donanım notu
            body_lines = ["## Özet", "Bu sistemin donanım envanteri NeuroCore tarafından "
                          "otomatik olarak çıkarılmıştır.", "", "## Canlı Durum"]
            for key, value in summary.items():
                body_lines.append(f"- **{key}**: {value}")
            body_lines += ["", "## Yapılandırma"]
            for key, value in config_summary.items():
                body_lines.append(f"- `{key}`: {value}")
            self.knowledge.write(
                title="Sistem Donanım Envanteri",
                body="\n".join(body_lines),
                kind="meta", category="meta",
                tags=["sistem", "donanim", "envanter"],
                confidence=0.9, importance=0.7,
                sources=[f"os://{platform.node()}"],
                reason="ilk kurulum tohumu", author="system",
            )
            written += 1

            # 2. Yetenek notu
            if self.os_layer is not None:
                try:
                    health = self.os_layer.health()
                    capabilities = self.os_layer.capabilities
                    tools = []
                    if hasattr(capabilities, "probe_all"):
                        tools = [
                            f"- `{name}`: {'var' if info.available else 'yok'}"
                            + (f" ({info.version[:40]})" if info.version else "")
                            for name, info in sorted(capabilities.probe_all().items())
                        ]
                    self.knowledge.write(
                        title="İşletim Sistemi Yetenekleri",
                        body=(
                            "## Özet\nBu makinede hangi araçların ve özelliklerin "
                            "kullanılabilir olduğu listesi.\n\n"
                            "## Araçlar\n" + "\n".join(tools[:50]) + "\n\n"
                            "## Uyarılar\n"
                            + "\n".join(f"- {item}" for item in (health.get("issues") or []))
                        ),
                        kind="meta", category="meta",
                        tags=["sistem", "yetenek", "araclar"],
                        confidence=0.85, importance=0.6,
                        reason="ilk kurulum tohumu", author="system",
                    )
                    written += 1
                except Exception as exc:
                    logger.debug(f"Yetenek notu yazılamadı: {exc}")

            # 3. Mimari notu
            self.knowledge.write(
                title="NeuroCore Mimarisi",
                body=(
                    "## Özet\nNeuroCore sekiz katmanlı, donanım-farkında ve "
                    "kendi kendini büyütebilen bir yapay zeka sistemidir.\n\n"
                    "## Katmanlar\n"
                    "1. **Donanım** — CPU/GPU/RAM izleme ve kaynak tahsisi\n"
                    "2. **Bilgi** — Obsidian uyumlu kasa, SQLite indeks, bilgi grafı\n"
                    "3. **Nöral** — uzman karışımlı transformer + bellek bankası\n"
                    "4. **Çok-modlu** — metin, görüntü, ses, tablo kodlayıcıları\n"
                    "5. **İşletim sistemi** — sandbox denetimli dosya/süreç/kabuk\n"
                    "6. **Dil** — algılama, çeviri, terim sözlüğü\n"
                    "7. **Planlayıcı** — görev grafiği, akıl yürütme, checkpoint\n"
                    "8. **Evrim** — deneyim, değerlendirme, sentez, merak, büyüme\n\n"
                    "## Büyüme\nModel, yönlendirme entropisi yükseldiğinde yeni "
                    "uzman ekler; ekleme **nötrdür** (yönlendirme bias'ı çok "
                    "negatif başlar), böylece mevcut bilgi bozulmaz.\n"
                ),
                kind="meta", category="meta",
                tags=["mimari", "neurocore", "sistem"],
                confidence=0.95, importance=0.8,
                reason="ilk kurulum tohumu", author="system",
            )
            written += 1

            logger.info(f"Kasa tohumlandı: {written} sistem notu eklendi")
        except Exception as exc:
            logger.warning(f"Tohumlama hatası: {exc}")
        return written

    def _bootstrap_corpus(self, limit: int = 1200) -> List[str]:
        """
        Tokenizer eğitimi için metin derlemi oluşturur.

        Kaynak sırası: kasadaki notlar → mimari/yapılandırma metinleri →
        yerleşik Türkçe teknik metin şablonları. Böylece kasa boşken de
        anlamlı bir BPE sözlüğü öğrenilir.
        """
        texts: List[str] = []
        # 1. Kasa
        if self.knowledge is not None:
            try:
                for note in self.knowledge.vault.iter_notes(limit=limit):
                    texts.append(f"# {note.title}\n{note.body}")
                    if len(texts) >= limit:
                        break
            except Exception as exc:
                logger.debug(f"Kasa derlemi alınamadı: {exc}")

        # 2. Sistem bilgisi
        try:
            if self.os_layer is not None:
                report = self.os_layer.system_report()
                texts.append(json.dumps(report, ensure_ascii=False, indent=1))
            if self.monitor is not None:
                texts.append(json.dumps(self.monitor.get_summary(), ensure_ascii=False))
            texts.append(json.dumps(self.config.get_summary(), ensure_ascii=False))
        except Exception as exc:
            logger.debug(f"Sistem metni eklenemedi: {exc}")

        # 3. Yerleşik Türkçe teknik şablonlar (dil yapısını öğrenmek için)
        texts.extend(_BOOTSTRAP_TEXTS)
        return texts

    def neural_status(self) -> Dict[str, Any]:
        """Nöral katmanın durum özeti (model + gömme + bellek)."""
        status: Dict[str, Any] = {"ready": self.model is not None and self.model.is_ready}
        if self.model is not None and self.model.is_ready:
            status.update({
                "pretrained": self.model.is_pretrained,
                "params": self.model.parameter_count(),
                "active_params": self.model.active_parameter_count(),
                "vocab_size": self.model.model_config.vocab_size if self.model.model_config else 0,
                "num_experts": self.model.growth.num_experts,
                "num_layers": self.model.growth.num_layers,
                "generation": self.model.growth.generation,
                "maturity": self.model.growth.maturity,
                "device": self.model._device,
                "quantization": self.model._quantization,
            })
        if self.embedding is not None:
            status["embedding_strategy"] = self.embedding.strategy
            status["embed_dim"] = self.embedding.dim
        if self.memory is not None:
            status["memory_size"] = self.memory.size
            status["memory_capacity"] = self.memory.capacity
        return status

    def _refresh_budget(self, force: bool = False) -> None:
        """
        Kaynak bütçesini tazeler (en fazla 5 saniyede bir).

        Bütçe değiştiğinde eğitici ve model güncellenir — sistem yük altında
        kendiliğinden küçülür, yük geçince toparlanır.
        """
        now = time.time()
        if not force and (now - self._last_budget_at) < 5.0:
            return
        if self.monitor is None or self.allocator is None:
            return
        try:
            snapshot = self.monitor.get_snapshot()
            old_pressure = self._budget.get("memory_pressure", "NORMAL")
            old_stall = bool(self._budget.get("headroom", {}).get("should_stall", False))
            budget = self.allocator.reallocate(snapshot)
            self._budget = budget.to_dict()
            self._last_budget_at = now

            # Nezaket bilgilerini eğiticiye ve model durumuna aktar
            headroom = self._budget.get("headroom", {}) or {}
            self._budget["enforce_headroom"] = bool(
                getattr(self.config.hardware, "enforce_headroom", True)
            )
            self._budget["headroom_factor"] = float(
                getattr(self.config.hardware.headroom, "cpu", 0.90) or 0.90
            )
            self._budget["should_stall"] = bool(headroom.get("should_stall", False))

            new_pressure = self._budget.get("memory_pressure", "NORMAL")
            new_stall = bool(headroom.get("should_stall", False))

            if new_stall != old_stall:
                if new_stall:
                    logger.info(
                        f"Kaynak nezaketi: yeni ağır iş duraklatıldı — "
                        f"{headroom.get('stall_reason', '')}"
                    )
                else:
                    logger.info("Kaynaklar boşaldı — ağır iş yeniden başlatılabilir")

            if new_pressure != old_pressure:
                logger.info(
                    f"Bellek baskısı: {old_pressure} → {new_pressure} "
                    f"(cihaz {self._budget.get('device')}, "
                    f"batch {self._budget.get('max_batch_size')})"
                )
                if new_pressure == "CRITICAL":
                    self._counter["degradations"] += 1
                    with self._lock:
                        if self.state == EngineState.RUNNING:
                            self.state = EngineState.DEGRADED
                elif new_pressure == "NORMAL" and self.state == EngineState.DEGRADED:
                    with self._lock:
                        self.state = EngineState.RUNNING

            # Eğitici ve nöral katman her durumda güncel bütçeyi bilmeli
            if self.trainer is not None:
                self.trainer.update_budget(self._budget)
        except Exception as exc:
            logger.debug(f"Bütçe tazelenemedi: {exc}")

    def _on_hardware_event(self, event: Event) -> None:
        """Donanım uyarılarını karşılar."""
        if event.event_type.endswith("ram_critical"):
            logger.warning("RAM kritik — bakım turu tetikleniyor")
            self._refresh_budget(force=True)
            if self.memory is not None:
                try:
                    self.memory.prune(max_age_days=30.0, min_utility=0.1)
                except Exception:
                    pass
        elif event.event_type.endswith("ram_warning"):
            self._refresh_budget(force=True)

    def _on_allocator_event(self, event: Event) -> None:
        """Kaynak tahsis olaylarını karşılar."""
        if event.event_type.endswith("budget_updated"):
            data = event.data if isinstance(event.data, dict) else {}
            logger.debug(f"Bütçe güncellendi: {data.get('changes')}")

    def _growth_metrics(self) -> Any:
        """Evrim + model + bütçe metriklerini birleştirir."""
        from neurocore.neural.growth import GrowthMetrics

        evolution_metrics: Dict[str, Any] = {}
        if self.evolution is not None:
            try:
                evolution_metrics = self.evolution.growth_metrics() or {}
            except Exception as exc:
                logger.debug(f"Evrim metrikleri alınamadı: {exc}")

        return GrowthMetrics.from_sources(
            model=self.model,
            budget=self._budget,
            evolution=evolution_metrics,
            buffer_size=int(evolution_metrics.get("buffer_size", 0) or 0),
            min_samples=int(getattr(self.config.evolution, "min_samples_for_training", 100) or 100),
        )

    def _is_system_query(self, text: str) -> bool:
        """
        Sorgu canlı sistem verisi gerektiriyor mu?

        Donanım/kaynak sorusu ise kasa araması yerine doğrudan canlı veri
        kullanılır (kasa eski bilgi içerebilir; sıcaklık anlıktır).
        """
        lowered = (text or "").lower()
        system_markers = (
            "cpu", "gpu", "ram", "bellek", "disk", "işlemci", "islemci",
            "sıcaklık", "sicaklik", "vram", "termal", "fan",
            "sistem durumu", "sistem raporu", "donanım", "donanim",
            "kaç çekirdek", "kac cekirdek", "boş alan", "bos alan",
            "kullanım", "kullanim", "yük", "yuk",
        )
        action_markers = (
            "göster", "goster", "rapor", "durum", "nedir", "kaç", "kac",
            "ne kadar", "listele", "kontrol",
        )
        has_system = any(marker in lowered for marker in system_markers)
        has_action = any(marker in lowered for marker in action_markers)
        # "GPU sıcaklığı neden yükselir" → sistem değil, bilgi sorusu
        is_why = any(marker in lowered for marker in ("neden", "niçin", "nicin", "why"))
        return has_system and has_action and not is_why

    @staticmethod
    def _derive_title(text: str, max_length: int = 80) -> str:
        """Sorgudan başlık üretir."""
        cleaned = (text or "").strip().replace("\n", " ")
        cleaned = cleaned.rstrip("?.!").strip()
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length].rsplit(" ", 1)[0] + "…"
        return cleaned or "Sorgu yanıtı"

    def _tags_from_query(self, text: str) -> List[str]:
        """Sorgudan etiket çıkarır (dil katmanı varsa onu kullanır)."""
        if self.language is not None:
            try:
                keywords = self.language.keywords(text, top_k=5)
                return [keyword for keyword in keywords if len(keyword) > 2][:5]
            except Exception:
                pass
        words = [word for word in (text or "").lower().split() if len(word) > 4]
        return words[:5]

    def _summarize(self, text: str, max_sentences: int = 3) -> str:
        """Metni özetler (dil katmanı varsa çıkarımsal, yoksa ilk cümleler)."""
        if not text:
            return ""
        if self.language is not None:
            try:
                return self.language.summarize_extractive(text, max_sentences)
            except Exception:
                pass
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return ". ".join(sentences[:max_sentences]) + ("." if sentences else "")

    @staticmethod
    def _context_bodies(context: str) -> str:
        """
        Bağlam bloğundan yalnızca bilgi taşıyan gövdeyi çıkarır.

        Bağlam biçimi `### Başlık` + `- id: … · güven: …` + gövde satırlarıdır.
        Özetleyiciye başlık, meta ve markdown iskelet satırlarını vermek,
        özetin yarısını yapısal gürültüye harcatır; bu yüzden ayıklanır.
        Ayrıca aynı satır birden çok notta geçiyorsa (tekrar eden kalıp)
        yalnızca ilk kez tutulur — böylece özet kendini tekrarlamaz.
        """
        if not context:
            return ""
        import re as _re

        # "Kaynaklar" blokları ve kaynak listesi satırları yapısal gürültüdür;
        # özete girmemeli. Not gövdeleri kendi kaynak listelerini içerebildiği
        # için (yanıt notları kaynaklarını gömerek saklar) temizlik üç biçimi
        # de kapsar ve kararlı hâle gelene kadar yinelenir:
        #   1. `## Kaynaklar` başlıklı blok
        #   2. `**Kaynaklar:**` veya `Kaynaklar:` satırı ve altındaki liste
        #   3. Tek başına duran kaynak satırları: `[[kn_x|Başlık]] (skor 0.77)`
        patterns = (
            _re.compile(r"(?ms)^#{0,6}\s*Kaynaklar\b.*?(?=^###\s|^##\s|\n\s*\n|\Z)"),
            _re.compile(r"(?ms)^\*\*Kaynaklar:\*\*.*?(?=\n\s*\n|\Z)"),
            _re.compile(r"(?m)^\s*-?\s*\[\[[^\]|]+\|[^\]]+\]\]\s*\(skor[^)]*\)\s*$"),
            _re.compile(r"(?m)^[^\n]{0,120}\(skor\s*[\d.,]+\)\s*$"),
            _re.compile(r"(?m)^\s*-?\s*\[\[[^\]|]+\|[^\]]+\]\]\s*$"),
        )
        previous = None
        while previous != context:
            previous = context
            for pattern in patterns:
                context = pattern.sub("", context)

        lines: List[str] = []
        seen: set = set()
        for line in context.splitlines():
            stripped = line.strip()

            # ── Yapısal satırları atla ──
            if not stripped:
                lines.append("")
                continue
            if stripped.startswith("#") or stripped.startswith("> "):
                continue
            if stripped.startswith("- id:") or stripped.startswith("id:"):
                continue
            if stripped.startswith("- ") and ("güven:" in stripped or "skor" in stripped):
                continue
            # Kaynak adı satırları: "Başlık (skor 0.77)" — yapısal gürültü
            if _re.search(r"\(skor\s*[\d.,]+\)\s*$", stripped) and len(stripped) < 160:
                continue
            # Markdown ayırıcıları (===, ---, ___) ve setext alt çizgileri
            if _re.fullmatch(r"[=\-_*~#\s]{3,}", stripped):
                continue
            # Yalnızca biçimlendirme artığı olan satırlar (ör. "``")
            if _re.fullmatch(r"[`*_>\-\s]+", stripped):
                continue
            # Yalnızca anahtar-değer satırları (canlı durum / yapılandırma).
            # Anahtar tarafında boşluk olmamalı; aksi halde "GPU sıcaklığı
            # yük altında: ..." gibi gerçek cümleler yanlışlıkla elenir.
            if _re.match(r"^[`*_]{0,2}[\w\.\-]{1,32}[`*_]{0,2}\s*[:=]\s*\S", stripped) \
                    and len(stripped) < 90:
                continue
            # Bölüm başlığı artığı (satır sonunda ## ...)
            stripped = _re.sub(r"\s*#{2,}\s*[\wÇĞİÖŞÜçğıöşü _\-]{0,40}$", "", stripped).strip()
            if not stripped or len(stripped) < 25:
                continue

            # ── Tekrar eden satırları bir kez tut ──
            key = _re.sub(r"\W+", "", stripped.lower())[:120]
            if key in seen:
                continue
            seen.add(key)
            lines.append(stripped)

        cleaned = "\n".join(lines)
        cleaned = _re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
        cleaned = _re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", cleaned)
        cleaned = _re.sub(r"^\s*[-*]\s+", "", cleaned, flags=_re.MULTILINE)
        cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned)
        # Paragrafları tek satıra indir (cümle bölütleme için)
        cleaned = _re.sub(r"(?<![.!?:;])\n(?=[a-zçğıöşü])", " ", cleaned)
        return cleaned.strip()

    def _data_path(self) -> Path:
        """Veri dizini."""
        base = getattr(self.config, "data_path", None)
        if not base:
            base = str(Path(self.config.base_path) / "data")
        path = Path(base)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        try:
            self.event_bus.emit(event_type, data, source="core.engine")
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    # ═══════════════════════════════════════════════════════════════════
    #  KISA YOLLAR (CLI ve API kolaylıkları)
    # ═══════════════════════════════════════════════════════════════════

    def ask(self, text: str, **kwargs: Any) -> str:
        """`query()` yapıp yalnızca yanıt metnini döndürür."""
        return self.query(text, **kwargs).render()

    def search(self, text: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Yalnızca kasa araması yapar (yanıt üretmeden)."""
        if self.knowledge is None:
            return []
        return self.knowledge.search_payloads(text, limit=limit)

    def graph_note(self, write: bool = False) -> Dict[str, Any]:
        """
        Bilgi grafı raporunu üretir; `write=True` ise kasaya not olarak ekler.
        """
        if self.knowledge is None:
            return {}
        if write:
            return self.knowledge.export_graph_note()
        try:
            return {"markdown": self.knowledge.graph.to_markdown()}
        except Exception as exc:
            return {"error": str(exc)}

    def status_line(self) -> str:
        """Tek satırlık durum özeti (istem başlığı için)."""
        try:
            stats = self.event_bus.get_stats()
            memory_size = self.memory.size if self.memory else 0
            notes = len(self.knowledge.vault._path_index) if self.knowledge else 0
            return (
                f"NeuroCore {self.VERSION} · {self.state.value} · "
                f"{self.config.hardware_profile}/{self.config.device} · "
                f"{notes} not · {memory_size} bellek · "
                f"{stats.get('total_events', 0)} olay"
            )
        except Exception:
            return f"NeuroCore {self.VERSION} · {self.state.value}"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<NeuroCoreEngine state={self.state.value} "
            f"profile={self.config.hardware_profile} device={self.config.device}>"
        )


__all__ = [
    "NeuroCoreEngine",
    "EngineState",
    "QueryResult",
    "HealthReport",
]


# ─── Bootstrap Derlemi ──────────────────────────────────────────────────────
#
# Bu metinler, kasa boşken BPE sözlüğünün Türkçe morfolojisini ve teknik
# sözcük dağarcığını öğrenmesi için kullanılır. Kod içinde tutulur (dosya
# bağımlılığı yok) ve yalnızca **tokenizer eğitimi** için kullanılır;
# kasaya not olarak yazılmaz.

_BOOTSTRAP_TEXTS: List[str] = [
    "Yapay sinir ağları, gradyan inişi ve geri yayılım algoritması ile eğitilir. "
    "Kayıp fonksiyonu, öğrenme oranı ve toplu iş boyutu eğitimin temel "
    "hiperparametreleridir. Aşırı öğrenme, doğrulama kaybı artarken eğitim "
    "kaybının düşmeye devam etmesiyle anlaşılır.",

    "Bilgi kasası, Obsidian uyumlu markdown dosyalarından oluşur. Her notun "
    "frontmatter bölümünde kimlik, başlık, etiketler, güven ve sürüm alanları "
    "bulunur. Notlar arasındaki bağlantılar çift köşeli parantez ile kurulur ve "
    "iki yönlü bir bilgi grafı oluşturur.",

    "Donanım izleme katmanı işlemci, bellek, ekran kartı ve disk kullanımını "
    "gerçek zamanlı olarak ölçer. Kaynak tahsis sistemi, boş bellek miktarına "
    "göre toplu iş boyutunu, önbellek boyutunu ve çalışan iş parçacığı sayısını "
    "otomatik olarak ayarlar.",

    "Uzman karışımı mimarisi, her belirteci yalnızca birkaç uzmana yönlendirir. "
    "Bu sayede toplam parametre sayısı artarken işlem maliyeti sabit kalır. Yeni "
    "bir uzman eklenirken yönlendirme ağırlığı sıfıra yakın başlatılır, böylece "
    "mevcut bilgi bozulmaz.",

    "Çok-modlu işlem hattı metin, görüntü, ses ve tablo girdilerini ayrı "
    "kodlayıcılarla işler ve tek bir ortak gösterim uzayında birleştirir. "
    "Füzyon katmanı, modaliteler arası dikkat mekanizması kullanır.",

    "Görev planlayıcı, karmaşık istekleri alt görevlere ayırır ve aralarındaki "
    "bağımlılıkları yönlü bir çizge olarak modeller. Her adım denetim noktası "
    "olarak diske yazılır, böylece kesinti sonrası plan kaldığı yerden sürdürülür.",

    "Evrim motoru deneyimleri değerlendirir, zayıf alanları belirler ve müfredat "
    "oluşturur. Merak motoru, bilgi grafındaki boşlukları ve çelişkileri tespit "
    "ederek yeni araştırma görevleri önerir.",

    "Dil katmanı, kaynak metnin dilini algılar ve gerekirse iç dile çevirir. "
    "Terim sözlüğü teknik kavramların tutarlı çevirisini sağlar. Çevrimdışı "
    "çalışma önceliklidir; uzak servisler yalnızca anahtar tanımlıysa kullanılır.",

    "Güvenlik katmanı, dosya sistemi ve kabuk işlemlerini kum havuzu kurallarıyla "
    "sınırlar. Yıkıcı komutlar engellenir, her eylem denetim günlüğüne yazılır ve "
    "hassas yollar reddedilir.",

    "Sorgu işleme hattı şu adımlardan oluşur: dil algılama, bilgi kasasında "
    "hibrit arama, bellek bankasından geri çağırma, bağlam derleme, yanıt sentezi "
    "ve kendi kendini değerlendirme. Yeterince iyi yanıtlar yeni not olarak saklanır.",
]
