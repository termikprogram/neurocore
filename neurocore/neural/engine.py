"""
NeuroCore Nöral Katman Cephesi
===============================
Nöral alt sistemlerin tek giriş noktası. Motor bu sınıfı kullanır; alt
modüller (tokenizer, omurga, MoE, bellek, gömme, büyüme, eğitici) doğrudan
dışarıya açılmaz.

Sorumluluklar:
    * Alt sistemlerin yaşam döngüsünü yönetmek
    * Aralarındaki bağımlılıkları kurmak (model → gömme, model → MoE)
    * Tek bir tutarlı API sunmak (`embed`, `generate`, `remember`, `recall`,
      `train`, `grow`, `health`)
    * Donanım bütçesi değiştiğinde kendini uyarlamak

Kullanım:
    neural = NeuralLayer()
    neural.initialize(config, event_bus, budget=budget)

    vector = neural.embed("merhaba")
    text   = neural.generate("Merhaba", max_new_tokens=32)["text"]
    neural.remember("GPU sıcaklığı tozdan yükselir.", source="vault:kn_1")
    hits   = neural.recall("GPU neden ısınır?")

    neural.train_from_vault(vault)
    decision = neural.propose_growth()
    neural.apply_growth(decision)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from neurocore.neural.embedding import EmbeddingEngine
from neurocore.neural.growth import GrowthController, GrowthDecision, GrowthMetrics
from neurocore.neural.memory import MemoryBank, RecallResult
from neurocore.neural.model import NeuroCoreModel
from neurocore.neural.tokenizer import BPETokenizer
from neurocore.neural.trainer import Trainer, TrainingDataset, TrainingPlan, TrainingResult
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.layer")


class NeuralLayerError(Exception):
    """Nöral katman hatası."""


@dataclass
class NeuralStatus:
    """Nöral katmanın tek bakışta durumu."""
    ready: bool = False
    pretrained: bool = False
    params: int = 0
    active_params: int = 0
    vocab_size: int = 0
    embed_dim: int = 768
    embedding_strategy: str = "hash"
    num_experts: int = 0
    num_layers: int = 0
    generation: int = 0
    maturity: float = 0.0
    memory_size: int = 0
    memory_capacity: int = 0
    device: str = "cpu"
    quantization: str = "fp32"
    training_steps: int = 0
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "ready": self.ready, "pretrained": self.pretrained,
            "params": self.params, "active_params": self.active_params,
            "vocab_size": self.vocab_size, "embed_dim": self.embed_dim,
            "embedding_strategy": self.embedding_strategy,
            "num_experts": self.num_experts, "num_layers": self.num_layers,
            "generation": self.generation, "maturity": self.maturity,
            "memory_size": self.memory_size, "memory_capacity": self.memory_capacity,
            "device": self.device, "quantization": self.quantization,
            "training_steps": self.training_steps, "issues": list(self.issues),
        }

    def render(self) -> str:
        """İnsan okur tek paragraf."""
        state = "hazır" if self.ready else "hazır değil"
        training = "eğitilmiş" if self.pretrained else "EĞİTİLMEMİŞ"
        return (
            f"Nöral katman {state}: {self.num_layers} katman × {self.embed_dim}d, "
            f"{self.num_experts} uzman, sözlük {self.vocab_size:,}. "
            f"{self.params:,} parametre ({self.active_params:,} etkin/token). "
            f"Gömme: {self.embedding_strategy}, bellek: {self.memory_size}/"
            f"{self.memory_capacity}, {training}, olgunluk {self.maturity:.2f}."
        )


class NeuralLayer:
    """
    Nöral katman cephesi.

    Alt sistemleri tek bir tutarlı API arkasında toplar ve aralarındaki
    bağımlılıkları (model ↔ gömme ↔ bellek ↔ eğitici) yönetir.
    """

    name = "neural"

    def __init__(self) -> None:
        self.model = NeuroCoreModel()
        self.embedding = EmbeddingEngine()
        self.memory = MemoryBank()
        self.growth = GrowthController()
        self.trainer = Trainer()

        self._config = None
        self._event_bus = None
        self._budget: Dict[str, Any] = {}
        self._initialized = False
        self._lock = threading.RLock()

        self._counter = {
            "embeddings": 0, "generations": 0, "memories_written": 0,
            "memories_recalled": 0, "trainings": 0, "growths": 0, "errors": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None,
                   budget: Optional[Dict[str, Any]] = None,
                   load_model: bool = True) -> None:
        """
        Nöral katmanı başlatır.

        Args:
            config: SystemConfig
            event_bus: EventBus
            budget: Kaynak bütçesi
            load_model: Diskteki modeli yükle
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._budget = dict(budget or {})

        self.embedding.initialize(config, event_bus)

        if load_model:
            self.model.initialize(config, event_bus, budget=self._budget)
            if self.model.tokenizer is not None:
                self.embedding.attach_tokenizer(self.model.tokenizer)
            if self.model.is_pretrained:
                self.embedding.attach_model(self.model.backbone, self.model.tokenizer)

        self.memory.initialize(config, event_bus)
        self.memory.attach_embedder(self.embedding.embed)
        self.growth.initialize(config, event_bus)
        self.trainer.initialize(config, event_bus, budget=self._budget)

        self._initialized = True
        logger.info(self.status().render())

    def shutdown(self) -> None:
        """Katmanı kapatır (idempotent)."""
        if not self._initialized:
            return
        for name, component in (
            ("eğitici", self.trainer), ("büyüme", self.growth),
            ("bellek", self.memory), ("model", self.model),
            ("gömme", self.embedding),
        ):
            try:
                component.shutdown()
            except Exception as exc:
                logger.warning(f"{name} kapatılırken hata: {exc}")
        self._initialized = False
        logger.info(
            f"Nöral katman kapatıldı ({self._counter['embeddings']} gömme, "
            f"{self._counter['generations']} üretim)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Katman istatistikleri."""
        return {
            **self._counter,
            "initialized": self._initialized,
            "status": self.status().to_dict(),
            "model": self.model.get_stats() if self.model.is_ready else {"ready": False},
            "embedding": self.embedding.get_stats(),
            "memory": self.memory.get_stats(),
            "growth": self.growth.get_stats(),
            "trainer": self.trainer.get_stats(),
            "budget": dict(self._budget),
        }

    def update_budget(self, budget: Dict[str, Any]) -> None:
        """Kaynak bütçesini günceller (bellek baskısı değiştiğinde)."""
        self._budget = dict(budget or {})
        try:
            self.trainer.update_budget(self._budget)
        except Exception as exc:
            logger.debug(f"Eğitici bütçesi güncellenemedi: {exc}")

    # ─── Durum ──────────────────────────────────────────────────────────

    def status(self) -> NeuralStatus:
        """Nöral katmanın durumu."""
        status = NeuralStatus(ready=self._initialized)
        try:
            if self.model.is_ready:
                config = self.model.model_config
                status.pretrained = self.model.is_pretrained
                status.params = self.model.parameter_count()
                status.active_params = self.model.active_parameter_count()
                status.vocab_size = config.vocab_size if config else 0
                status.embed_dim = config.embed_dim if config else self.embedding.dim
                status.num_experts = self.model.growth.num_experts
                status.num_layers = self.model.growth.num_layers
                status.generation = self.model.growth.generation
                status.maturity = self.model.growth.maturity
                status.device = self.model._device
                status.quantization = self.model._quantization
                status.training_steps = self.model.growth.total_training_steps
            status.embedding_strategy = self.embedding.strategy
            status.embed_dim = status.embed_dim or self.embedding.dim
            status.memory_size = self.memory.size
            status.memory_capacity = self.memory.capacity

            if not status.pretrained:
                status.issues.append("model eğitilmedi — gömme hash stratejisinde")
            elif not self.model.can_generate:
                readiness = self.model.generation_readiness()
                status.issues.append(
                    f"üretim için yetersiz eğitim: {readiness['training_steps']}/"
                    f"{readiness['min_steps']} adım — çıkarımsal yanıt kullanılıyor"
                )
            if self.memory.size == 0:
                status.issues.append("bellek bankası boş")
            if status.vocab_size and status.vocab_size <= 300:
                status.issues.append("tokenizer eğitilmedi (sözlük çok küçük)")
        except Exception as exc:
            status.issues.append(f"durum okunamadı: {exc}")
        return status

    def health(self) -> Dict[str, Any]:
        """Sağlık raporu."""
        status = self.status()
        issues = list(status.issues)
        for component_name, component in (("gömme", self.embedding),
                                          ("bellek", self.memory),
                                          ("eğitici", self.trainer)):
            try:
                if hasattr(component, "health"):
                    info = component.health()
                    if hasattr(info, "to_dict"):
                        info = info.to_dict()
                    issues.extend(f"{component_name}: {item}"
                                  for item in (info.get("issues") or []))
            except Exception as exc:
                issues.append(f"{component_name}: sağlık okunamadı ({exc})")
        return {
            "ok": self._initialized,
            "status": status.to_dict(),
            "issues": issues,
            "critical": [] if self._initialized else ["nöral katman başlatılmadı"],
        }

    # ─── Gömme ──────────────────────────────────────────────────────────

    def embed(self, text: str, *, use_cache: bool = True) -> List[float]:
        """
        Metni vektöre çevirir.

        Strateji otomatiktir: model eğitilmişse öğrenilmiş, değilse hash
        gömme. Her durumda `embed_dim` boyutlu, L2-normalize bir vektör döner.
        """
        self._counter["embeddings"] += 1
        try:
            return self.embedding.embed(text, use_cache=use_cache)
        except Exception as exc:
            self._counter["errors"] += 1
            logger.error(f"Gömme hatası: {exc}")
            from neurocore.neural.embedding import hash_embed
            return hash_embed(text, dim=self.embedding.dim)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Toplu gömme."""
        return self.embedding.embed_batch(texts)

    def similarity(self, a: str, b: str) -> float:
        """İki metin arasındaki kosinüs benzerliği."""
        return self.embedding.similarity(a, b)

    def attach_embedder(self, embed_fn: Callable[[str], Sequence[float]]) -> None:
        """
        Harici bir gömme fonksiyonu bağlar (ör. çok-modlu katman).

        Bağlandığında bellek bankası bu fonksiyonu kullanır. Omurga tabanlı
        gömme tercih edilir; bu yalnızca bellek bankasının anahtar üretimi
        için bir geçersiz kılma noktasıdır.
        """
        self.memory.attach_embedder(embed_fn)
        logger.info("Harici gömme fonksiyonu bağlandı (bellek bankası)")

    # ─── Üretim ─────────────────────────────────────────────────────────

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Metin üretir (`NeuroCoreModel.generate_text` sarmalayıcısı)."""
        self._counter["generations"] += 1
        try:
            return self.model.generate_text(prompt, **kwargs)
        except Exception as exc:
            self._counter["errors"] += 1
            return {"error": str(exc), "text": "", "pretrained": False}

    def tokenize(self, text: str, **kwargs: Any) -> List[int]:
        """Metni token id'lerine çevirir."""
        if self.model.tokenizer is None:
            return []
        return self.model.tokenizer.encode(text, **kwargs)

    def decode(self, ids: Sequence[int], **kwargs: Any) -> str:
        """Token id'lerini metne çevirir."""
        if self.model.tokenizer is None:
            return ""
        return self.model.tokenizer.decode(ids, **kwargs)

    # ─── Bellek ─────────────────────────────────────────────────────────

    def remember(self, text: str, **kwargs: Any) -> Optional[str]:
        """Belleğe bilgi yazar (yazma politikası uygulanır)."""
        record_id = self.memory.remember(text, **kwargs)
        if record_id:
            self._counter["memories_written"] += 1
        return record_id

    def recall(self, query: str, top_k: int = 5, **kwargs: Any) -> List[RecallResult]:
        """Bellekten geri çağırır."""
        results = self.memory.recall(query, top_k=top_k, **kwargs)
        if results:
            self._counter["memories_recalled"] += 1
        return results

    def recall_text(self, query: str, top_k: int = 5, **kwargs: Any) -> str:
        """Geri çağrılan kayıtları tek bağlam metnine birleştirir."""
        return self.memory.recall_text(query, top_k=top_k, **kwargs)

    # ─── Eğitim ─────────────────────────────────────────────────────────

    def train(
        self,
        dataset: TrainingDataset,
        plan: Optional[TrainingPlan] = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """
        Verilen veri kümesiyle modeli eğitir.

        Eğitim başarılıysa model `pretrained` işaretlenir ve gömme motoru
        öğrenilmiş stratejiye geçer.
        """
        self._counter["trainings"] += 1
        result = self.trainer.train(self.model, dataset, plan=plan, **kwargs)
        if result.steps_completed > 0:
            self.model._is_pretrained = True
            if self.model.backbone is not None:
                self.embedding.attach_model(self.model.backbone, self.model.tokenizer)
            try:
                self.model.save()
            except Exception as exc:
                logger.warning(f"Model kaydedilemedi: {exc}")
        return result

    def train_from_vault(self, vault: Any, max_notes: int = 500,
                         steps: Optional[int] = None, **kwargs: Any) -> TrainingResult:
        """Kasadaki notlarla modeli eğitir."""
        if self.model.tokenizer is None:
            raise NeuralLayerError("Tokenizer hazır değil")
        dataset = self.trainer.build_dataset_from_vault(
            vault, self.model.tokenizer, max_notes=max_notes,
        )
        if dataset.is_empty:
            logger.warning("Kasa boş — eğitim yapılmadı")
            return TrainingResult(stop_reason="veri_yok")
        plan = self.trainer.plan_training(
            estimated_samples=len(dataset), model=self.model, steps=steps,
            rationale="kasa eğitimi",
        )
        return self.train(plan=plan, dataset=dataset, **kwargs)

    def train_tokenizer(self, texts: Iterable[str], vocab_size: int = 8000) -> Dict[str, Any]:
        """
        BPE tokenizer'ı eğitir ve omurgayı yeni sözlüğe göre yeniden kurar.

        Sözlük değiştiğinde gömme matrisi boyutu değişir; bu yüzden omurga
        **yeniden oluşturulur** (mevcut ağırlıklar sözlükle uyumsuz kalır).
        """
        if self.model.tokenizer is None:
            return {"error": "tokenizer_yok"}
        outcome = self.model.tokenizer.train(texts, vocab_size=vocab_size)
        self.model.tokenizer.save_default()

        # Omurgayı yeni sözlükle yeniden kur
        if self._config is not None:
            try:
                self.model.model_config.vocab_size = self.model.tokenizer.vocab_size
                from neurocore.neural.transformer import TransformerBackbone
                self.model.backbone = TransformerBackbone(self.model.model_config)
                self.model._is_pretrained = False
                self.embedding.attach_tokenizer(self.model.tokenizer)
                logger.info(
                    f"Tokenizer yeniden eğitildi ({outcome.get('vocab_size')} token) — "
                    "omurga sıfırlandı, yeniden eğitim gerekli"
                )
            except Exception as exc:
                logger.error(f"Omurga yeniden kurulamadı: {exc}")
                outcome["rebuild_error"] = str(exc)
        return outcome

    def evaluate(self, dataset: TrainingDataset, **kwargs: Any) -> Dict[str, Any]:
        """Doğrulama kaybını ölçer."""
        return self.trainer.evaluate(self.model, dataset, **kwargs)

    # ─── Büyüme ─────────────────────────────────────────────────────────

    def growth_metrics(self, **kwargs: Any) -> GrowthMetrics:
        """Büyüme kararı için metrik toplar."""
        return GrowthMetrics.from_sources(
            model=self.model, budget=self._budget, **kwargs
        )

    def propose_growth(self, metrics: Optional[GrowthMetrics] = None) -> GrowthDecision:
        """Büyüme kararı üretir (uygulamaz)."""
        return self.growth.decide(metrics or self.growth_metrics())

    def apply_growth(self, decision: GrowthDecision, dry_run: bool = False) -> Dict[str, Any]:
        """Büyüme kararını uygular."""
        self._counter["growths"] += 1
        return self.growth.apply(decision, self.model, dry_run=dry_run)

    def grow(
        self,
        *,
        experts: int = 0,
        layers: int = 0,
        activate: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Doğrudan büyüme uygular (karar mantığını atlar).

        Args:
            experts: Eklenecek uzman sayısı (katman başına)
            layers: Eklenecek katman sayısı
            activate: Yeni uzmanları hemen devreye al
            dry_run: Yalnızca raporla

        Returns:
            {"experts": {...}, "layers": {...}, "activation": {...}}
        """
        outcome: Dict[str, Any] = {}
        if dry_run:
            return {
                "dry_run": True,
                "would_add_experts": experts,
                "would_add_layers": layers,
                "current_experts": self.model.growth.num_experts,
                "current_layers": self.model.growth.num_layers,
            }
        if experts:
            outcome["experts"] = self.model.grow_experts(experts, dormant=not activate)
        if layers:
            outcome["layers"] = self.model.grow_layers(layers, neutral=True)
        if activate:
            outcome["activation"] = {"activated": self.model.activate_experts(0.0)}
        self._counter["growths"] += 1
        return outcome

    def quantize(self, mode: str = "fp16") -> Dict[str, Any]:
        """Modeli niceleştirir."""
        return self.model.quantize(mode)

    # ─── Kalıcılık ──────────────────────────────────────────────────────

    def save(self) -> Dict[str, Any]:
        """Modeli, belleği ve gömme durumunu kaydeder."""
        outcome: Dict[str, Any] = {}
        try:
            outcome["model"] = self.model.save()
        except Exception as exc:
            outcome["model"] = {"error": str(exc)}
        try:
            outcome["memory"] = {"path": self.memory.save()}
        except Exception as exc:
            outcome["memory"] = {"error": str(exc)}
        return outcome

    def load(self) -> bool:
        """Modeli ve belleği diskten yükler."""
        loaded = False
        try:
            loaded = self.model.load(allow_missing_weights=True)
            if loaded and self.model.is_pretrained:
                self.embedding.attach_model(self.model.backbone, self.model.tokenizer)
        except Exception as exc:
            logger.error(f"Model yüklenemedi: {exc}")
        try:
            self.memory.load()
        except Exception as exc:
            logger.debug(f"Bellek yüklenemedi: {exc}")
        return loaded

    # ─── Sorgu Yardımcıları ─────────────────────────────────────────────

    def answer(self, question: str, context: str = "",
               max_new_tokens: int = 256) -> Dict[str, Any]:
        """
        Soruyu bağlamla yanıtlar (üretim veya geri çağırma).

        Model eğitilmemişse bellek bankasından geri çağırma ile yetinir;
        bu, sistemin ilk günden işe yaramasını sağlar.
        """
        # 1. Eğitilmiş model varsa üret
        if self.model.can_generate and context:
            prompt = (
                f"Soru: {question}\n\nBağlam:\n{context[:2400]}\n\n"
                "Yukarıdaki bağlama dayanarak Türkçe yanıtla.\nYanıt:"
            )
            generated = self.generate(prompt, max_new_tokens=max_new_tokens,
                                      temperature=0.6)
            if generated.get("text"):
                return {"answer": generated["text"].strip(),
                        "source": "neural", "tokens": generated.get("new_tokens", 0)}

        # 2. Bellekten geri çağır
        remembered = self.recall_text(question, top_k=3, min_score=0.3)
        if remembered:
            return {"answer": remembered, "source": "memory", "tokens": 0}

        return {"answer": "", "source": "none", "tokens": 0,
                "note": "model eğitilmedi ve bellekte ilgili kayıt yok"}

    def describe(self) -> str:
        """İnsan okur açıklama."""
        parts = [self.status().render()]
        if self.model.is_ready:
            parts.append(self.model.describe())
        return " ".join(parts)

    def __repr__(self) -> str:  # pragma: no cover
        status = self.status()
        return (
            f"<NeuralLayer ready={status.ready} pretrained={status.pretrained} "
            f"params={status.params} memory={status.memory_size}>"
        )


__all__ = [
    "NeuralLayer",
    "NeuralStatus",
    "NeuralLayerError",
]
