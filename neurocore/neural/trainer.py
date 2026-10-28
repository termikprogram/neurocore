"""
NeuroCore Eğitim Hattı
=======================
Donanım-farkında, checkpoint'lenebilir ve kesintiye dayanıklı eğitim döngüsü.

Tasarım ilkeleri:
    * **Donanıma uyum** — toplu iş boyutu, birikim adımı ve worker sayısı
      kaynak bütçesinden gelir; bellek baskısı artarsa eğitim otomatik
      olarak küçülür (`adaptive_batch`).
    * **Kesintisiz** — her `checkpoint_interval` adımda diske yazılır; süreç
      ölürse `resume()` ile kaldığı yerden devam eder.
    * **Kararlı** — AdamW + kosinüs/lineer ısınma planı, gradyan kırpma,
      NaN/Inf koruması ve kayıp ölçekleme (AMP).
    * **Şeffaf** — her adım metrik üretir; darboğazda erken uyarı verir.
    * **Bellek güvenli** — eğitim öncesi bellek tahmini yapılır ve bütçeyle
      karşılaştırılır; sığmıyorsa plan otomatik küçültülür.

Kullanım:
    trainer = Trainer()
    trainer.initialize(config, event_bus, budget=budget)
    plan = trainer.plan_training(estimated_samples=1200)
    history = trainer.train(model, dataset, plan=plan)
    trainer.resume(model)                     # kesintiden sonra
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from neurocore.neural.transformer import HAS_TORCH, estimate_backbone_memory
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.trainer")

if HAS_TORCH:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


# ─── Yapılandırma ve Plan ───────────────────────────────────────────────────

@dataclass
class TrainingPlan:
    """
    Tek bir eğitim turunun planı.

    Bu yapı, "ne kadar eğiteceğiz ve hangi kaynakla" sorusunun tam yanıtıdır;
    evrim katmanı bu planı üretip eğiticiye verir.
    """
    steps: int = 100
    batch_size: int = 8
    grad_accumulation: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    schedule: str = "cosine"           # cosine | linear | constant
    max_grad_norm: float = 1.0
    use_amp: bool = False
    gradient_checkpointing: bool = False
    seq_len: int = 256
    save_every: int = 50
    log_every: int = 10
    eval_every: int = 0
    seed: Optional[int] = None
    rationale: str = ""
    priority_tags: List[str] = field(default_factory=list)

    @property
    def effective_batch_size(self) -> int:
        """Birikim dahil gerçek toplu iş boyutu."""
        return int(self.batch_size) * max(1, int(self.grad_accumulation))

    @property
    def tokens_per_step(self) -> int:
        """Adım başına işlenen token sayısı."""
        return self.effective_batch_size * int(self.seq_len)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        data = {key: getattr(self, key) for key in self.__dataclass_fields__}
        data["effective_batch_size"] = self.effective_batch_size
        data["tokens_per_step"] = self.tokens_per_step
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingPlan":
        """Sözlükten plan üretir (bilinmeyen alanlar yok sayılır)."""
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class TrainingMetrics:
    """Tek bir eğitim adımının metrikleri."""
    step: int = 0
    loss: float = 0.0
    aux_loss: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0
    tokens: int = 0
    duration_ms: float = 0.0
    memory_mb: float = 0.0
    skipped: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "step": self.step, "loss": round(self.loss, 6),
            "aux_loss": round(self.aux_loss, 6), "lr": self.lr,
            "grad_norm": round(self.grad_norm, 4), "tokens": self.tokens,
            "duration_ms": round(self.duration_ms, 2),
            "memory_mb": round(self.memory_mb, 1),
            "skipped": self.skipped, "note": self.note,
        }


@dataclass
class ResourceGuard:
    """
    Uzun işler sırasında kaynak nezaketini uygulayan gözcü.

    Kural ("boştakinin %90'ı"):
        Ajan, boşta olan kaynağın %90'ından fazlasını kullanmaz. Kullanıcı
        tarayıcıda video izlerken veya oyun oynarken ajan kendini küçültür ve
        gerekirse **duraklar** — bilgisayarın kilitlenmesini önleyen katman
        budur.

    Eylemler:
        * `stop`   — iş yükü kritik eşiği aştı, tur sonlandırılır
        * `reduce` — baskı yükseldi, toplu iş küçültülür
        * `pause`  — kısa süre beklenir, sonra devam edilir
        * `none`   — kaynak yeterli
    """
    #: Kullanıcı payı: boştakinin bu kadarı kullanılır
    factor: float = 0.90
    max_ram_percent: float = 0.85
    max_gpu_percent: float = 0.90
    max_cpu_percent: float = 0.85
    #: Baskı altında kaç adım beklenecek (adım başına değil, ardışık)
    pause_seconds: float = 5.0
    #: Kaç baskı olayından sonra tur sonlandırılır
    max_pressure_events: int = 6
    enabled: bool = True

    pressure_events: int = 0
    reductions: int = 0
    last_pressure: str = "NORMAL"
    last_reason: str = ""
    total_paused_seconds: float = 0.0
    checks: int = 0

    def check(self, *,
              ram_percent: Optional[float] = None,
              cpu_percent: Optional[float] = None,
              gpu_percent: Optional[float] = None) -> str:
        """
        Kaynak durumunu değerlendirir.

        Args:
            ram_percent / cpu_percent / gpu_percent: 0..100 arası kullanım
                (None verilenler atlanır)

        Returns:
            `none` | `pause` | `reduce` | `stop`
        """
        if not self.enabled:
            return "none"
        self.checks += 1

        usage: Dict[str, float] = {}
        if ram_percent is not None:
            usage["RAM"] = float(ram_percent)
        if cpu_percent is not None:
            usage["CPU"] = float(cpu_percent)
        if gpu_percent is not None:
            usage["VRAM"] = float(gpu_percent)
        if not usage:
            return "none"

        limits = {
            "RAM": self.max_ram_percent * 100.0,
            "CPU": self.max_cpu_percent * 100.0,
            "VRAM": self.max_gpu_percent * 100.0,
        }
        blocking = [
            f"{name} %{value:.0f} ≥ %{limits[name]:.0f}"
            for name, value in usage.items() if value >= limits[name]
        ]
        worst = max(usage.values())
        worst_limit = max(limits[name] for name in usage)

        if blocking:
            self.pressure_events += 1
            self.last_pressure = "CRITICAL"
            self.last_reason = "sistem kaynakları dolu: " + ", ".join(blocking)
            # Çok sık baskı görülüyorsa turu bitir (sistemi yorma)
            if self.pressure_events >= self.max_pressure_events:
                return "stop"
            return "pause" if worst < 95.0 else "reduce"

        self.last_pressure = "NORMAL"
        self.last_reason = ""
        return "none"

    def reduce_batch(self, plan: TrainingPlan) -> bool:
        """
        Plan toplu işini küçültür (bellek baskısı altında).

        Returns:
            Gerçekten küçültüldüyse True (zaten minimumdaysa False)
        """
        changed = False
        if plan.batch_size > 1:
            plan.batch_size = max(1, plan.batch_size // 2)
            changed = True
        elif plan.seq_len > 64:
            plan.seq_len = max(64, plan.seq_len // 2)
            changed = True
        elif plan.grad_accumulation < 16:
            plan.grad_accumulation = min(16, plan.grad_accumulation * 2)
            changed = True
        if changed:
            self.reductions += 1
        return changed

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "enabled": self.enabled,
            "factor": self.factor,
            "pressure_events": self.pressure_events,
            "reductions": self.reductions,
            "last_pressure": self.last_pressure,
            "last_reason": self.last_reason,
            "total_paused_seconds": round(self.total_paused_seconds, 1),
            "checks": self.checks,
        }


@dataclass
class TrainingResult:
    """Eğitim turunun özeti."""
    steps_completed: int = 0
    steps_requested: int = 0
    initial_loss: float = 0.0
    final_loss: float = 0.0
    best_loss: float = 0.0
    mean_loss: float = 0.0
    #: ⚠ Doğrulama (held-out) kaybı — eğitim boyunca izlenir.
    #:
    #: Neden eklendi? Ölçüldü: doğrulama kaybı **hiç** ölçülmüyordu
    #: (`eval_every = 0`, döngüde `validation` kullanımı yok). Eğitim
    #: kaybı 0,083'e inerken gerçek genelleme 6,44'tü — yani tek başına
    #: eğitim kaybı **yanıltıcıydı**. Bu alanlar olmadan "öğrendi mi?"
    #: sorusu cevaplanamaz.
    initial_val_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = 0.0
    #: Her doğrulama ölçümü: {"step", "loss", "perplexity", "batches", "samples"}
    val_history: List[Dict[str, Any]] = field(default_factory=list)
    tokens_seen: int = 0
    duration_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    grad_norm_max: float = 0.0
    diverged: bool = False
    early_stopped: bool = False
    stop_reason: str = ""
    checkpoint_path: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    guard: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_history: bool = False) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        data = {
            "steps_completed": self.steps_completed,
            "steps_requested": self.steps_requested,
            "initial_loss": round(self.initial_loss, 6),
            "final_loss": round(self.final_loss, 6),
            "best_loss": round(self.best_loss, 6),
            "mean_loss": round(self.mean_loss, 6),
            # ⚠ Doğrulama kaybı — eğitim kaybından AYRI raporlanır
            "initial_val_loss": round(self.initial_val_loss, 6),
            "final_val_loss": round(self.final_val_loss, 6),
            "best_val_loss": round(self.best_val_loss, 6),
            "val_history": list(self.val_history or []),
            "loss_reduction_percent": round(
                (1 - self.final_loss / self.initial_loss) * 100, 2
            ) if self.initial_loss > 0 else 0.0,
            "tokens_seen": self.tokens_seen,
            "duration_ms": round(self.duration_ms, 1),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 1),
            "grad_norm_max": round(self.grad_norm_max, 3),
            "diverged": self.diverged,
            "early_stopped": self.early_stopped,
            "stop_reason": self.stop_reason,
            "checkpoint_path": self.checkpoint_path,
            "plan": self.plan,
            "guard": dict(self.guard or {}),
        }
        if include_history:
            data["history"] = self.history
        return data


# ─── Veri Kümesi ────────────────────────────────────────────────────────────

@dataclass
class TrainingSample:
    """Tek bir eğitim örneği (istem → hedef)."""
    prompt: str
    target: str = ""
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)
    difficulty: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {"prompt": self.prompt, "target": self.target,
                "weight": self.weight, "tags": list(self.tags),
                "difficulty": self.difficulty}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingSample":
        """Sözlükten örnek üretir."""
        return cls(
            prompt=str(data.get("prompt") or data.get("query") or ""),
            target=str(data.get("target") or data.get("answer") or ""),
            weight=float(data.get("weight", 1.0) or 1.0),
            tags=[str(tag) for tag in (data.get("tags") or [])],
            difficulty=float(data.get("difficulty", 0.5) or 0.5),
        )


class TrainingDataset:
    """
    Eğitim verisi tutucusu.

    Örnekleri token'lara çevirir, uzunluğa göre gruplar (length bucketing) ve
    ağırlıklı örnekleme yapar. Büyük veri kümeleri için bellekte tutulan
    token dizileri `max_samples` ile sınırlanır.
    """

    def __init__(self, tokenizer: Any, seq_len: int = 256, max_samples: int = 20000) -> None:
        """
        Args:
            tokenizer: `BPETokenizer` benzeri (`encode`/`pad_id`/`eos_id`)
            seq_len: Dizi uzunluğu (kırpma/dolgu hedefi)
            max_samples: Bellekte tutulacak maksimum örnek
        """
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.max_samples = int(max_samples)
        self.samples: List[TrainingSample] = []
        self._encoded: List[Tuple[List[int], float, int]] = []
        self._dirty = True
        self._stats = {"added": 0, "skipped": 0, "tokens": 0, "truncated": 0}

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def is_empty(self) -> bool:
        """Veri kümesi boş mu?"""
        return not self.samples

    def add(self, prompt: str, target: str = "", weight: float = 1.0,
            tags: Optional[Iterable[str]] = None, difficulty: float = 0.5) -> bool:
        """Tek örnek ekler."""
        if not prompt or not prompt.strip():
            self._stats["skipped"] += 1
            return False
        if len(self.samples) >= self.max_samples:
            # Yer açmak için en ağır (en az bilgilendirici) örneği at
            self.samples.sort(key=lambda sample: -sample.weight)
            self.samples.pop()
        self.samples.append(TrainingSample(
            prompt=prompt.strip(), target=(target or "").strip(),
            weight=float(weight), tags=[str(tag) for tag in (tags or [])],
            difficulty=float(difficulty),
        ))
        self._stats["added"] += 1
        self._dirty = True
        return True

    def add_many(self, items: Iterable[Any]) -> int:
        """Çok sayıda örnek ekler (sözlük, `TrainingSample` veya demet)."""
        added = 0
        for item in items:
            if isinstance(item, TrainingSample):
                if self.add(item.prompt, item.target, item.weight, item.tags, item.difficulty):
                    added += 1
            elif isinstance(item, dict):
                sample = TrainingSample.from_dict(item)
                if self.add(sample.prompt, sample.target, sample.weight,
                            sample.tags, sample.difficulty):
                    added += 1
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                if self.add(str(item[0]), str(item[1])):
                    added += 1
        return added

    def from_batch(self, batch: Dict[str, Any]) -> int:
        """
        Evrim katmanının ürettiği eğitim paketini yükler.

        Beklenen biçim: {"prompts": [...], "targets": [...], "weights": [...],
        "difficulties": [...]}
        """
        prompts = list(batch.get("prompts") or batch.get("inputs") or [])
        targets = list(batch.get("targets") or batch.get("outputs") or [])
        weights = list(batch.get("weights") or [])
        difficulties = list(batch.get("difficulties") or [])
        added = 0
        for index, prompt in enumerate(prompts):
            target = targets[index] if index < len(targets) else ""
            weight = float(weights[index]) if index < len(weights) else 1.0
            difficulty = float(difficulties[index]) if index < len(difficulties) else 0.5
            if self.add(str(prompt), str(target), weight, difficulty=difficulty):
                added += 1
        return added

    def _ensure_encoded(self) -> None:
        """Örnekleri token'lara çevirir (tembel, önbellekli)."""
        if not self._dirty:
            return
        self._encoded = []
        bos = getattr(self.tokenizer, "bos_id", 1)
        eos = getattr(self.tokenizer, "eos_id", 2)
        total_tokens = 0

        for index, sample in enumerate(self.samples):
            text = sample.prompt if not sample.target else f"{sample.prompt}\n{sample.target}"
            try:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
            except Exception as exc:
                logger.debug(f"Örnek kodlanamadı ({index}): {exc}")
                continue
            if not ids:
                continue
            ids = [bos] + ids + [eos]
            if len(ids) > self.seq_len:
                ids = ids[: self.seq_len]
                self._stats["truncated"] += 1
            total_tokens += len(ids)
            self._encoded.append((ids, sample.weight, index))

        self._stats["tokens"] = total_tokens
        self._dirty = False

    def batches(self, batch_size: int, shuffle: bool = True,
                seed: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Toplu iş üreteci.

        Args:
            batch_size: Toplu iş boyutu
            shuffle: Karıştır (uzunluk gruplaması içinde)
            seed: Rastgelelik tohumu (tekrarlanabilirlik)

        Yields:
            {"input_ids": tensor, "labels": tensor, "attention_mask": tensor,
             "weights": tensor, "meta": {...}}
        """
        self._ensure_encoded()
        if not self._encoded:
            return

        padded = getattr(self.tokenizer, "pad_id", 0)
        items = list(self._encoded)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(items)

        batch_size = max(1, int(batch_size))
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            if not chunk:
                continue
            max_len = max(len(ids) for ids, _weight, _index in chunk)
            max_len = max(8, min(max_len, self.seq_len))

            input_rows: List[List[int]] = []
            label_rows: List[List[int]] = []
            mask_rows: List[List[int]] = []
            weights: List[float] = []
            indices: List[int] = []

            for ids, weight, index in chunk:
                row = list(ids[:max_len])
                row += [padded] * (max_len - len(row))
                # Etiketler: dolgu konumları -100 (kayıptan hariç)
                labels = [token if token != padded else -100 for token in row]
                input_rows.append(row)
                label_rows.append(labels)
                mask_rows.append([1 if token != padded else 0 for token in row])
                weights.append(float(weight))
                indices.append(index)

            if HAS_TORCH:
                yield {
                    "input_ids": torch.tensor(input_rows, dtype=torch.long),
                    "labels": torch.tensor(label_rows, dtype=torch.long),
                    "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
                    "weights": torch.tensor(weights, dtype=torch.float32),
                    "meta": {"indices": indices, "length": max_len},
                }
            else:
                yield {
                    "input_ids": input_rows, "labels": label_rows,
                    "attention_mask": mask_rows, "weights": weights,
                    "meta": {"indices": indices, "length": max_len},
                }

    def split(self, validation_ratio: float = 0.1, seed: int = 42
              ) -> Tuple["TrainingDataset", "TrainingDataset"]:
        """Veriyi eğitim/doğrulama olarak böler."""
        rng = random.Random(seed)
        items = list(self.samples)
        rng.shuffle(items)
        cut = max(0, int(len(items) * (1.0 - max(0.0, min(0.5, validation_ratio)))))
        train = TrainingDataset(self.tokenizer, self.seq_len, self.max_samples)
        valid = TrainingDataset(self.tokenizer, self.seq_len, self.max_samples)
        for sample in items[:cut]:
            train.add(sample.prompt, sample.target, sample.weight,
                      sample.tags, sample.difficulty)
        for sample in items[cut:]:
            valid.add(sample.prompt, sample.target, sample.weight,
                      sample.tags, sample.difficulty)
        return train, valid

    def stats(self) -> Dict[str, Any]:
        """Veri kümesi istatistikleri."""
        self._ensure_encoded()
        lengths = [len(ids) for ids, _weight, _index in self._encoded]
        tags: Dict[str, int] = {}
        for sample in self.samples:
            for tag in sample.tags:
                tags[tag] = tags.get(tag, 0) + 1
        return {
            **self._stats,
            "samples": len(self.samples),
            "encoded": len(self._encoded),
            "seq_len": self.seq_len,
            "mean_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
            "max_length": max(lengths) if lengths else 0,
            "mean_weight": round(
                sum(sample.weight for sample in self.samples) / max(1, len(self.samples)), 3
            ),
            "mean_difficulty": round(
                sum(sample.difficulty for sample in self.samples) / max(1, len(self.samples)), 3
            ),
            "tags": dict(sorted(tags.items(), key=lambda item: -item[1])[:20]),
        }

    def save(self, path: str) -> str:
        """Veri kümesini JSONL olarak kaydeder."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        return str(target)

    @classmethod
    def load(cls, path: str, tokenizer: Any, seq_len: int = 256) -> "TrainingDataset":
        """JSONL'den veri kümesi yükler."""
        dataset = cls(tokenizer, seq_len)
        target = Path(path)
        if not target.exists():
            return dataset
        with open(target, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    dataset.add_many([json.loads(line)])
                except json.JSONDecodeError:
                    continue
        return dataset


# ─── Eğitici ────────────────────────────────────────────────────────────────

class Trainer:
    """
    Donanım-farkında eğitim döngüsü.

    Kullanım:
        trainer = Trainer()
        trainer.initialize(config, event_bus, budget=budget)

        plan = trainer.plan_training(estimated_samples=len(dataset))
        result = trainer.train(model, dataset, plan=plan)
        print(result.to_dict())

        # Kesintiden sonra
        trainer.resume(model)
    """

    name = "neural.trainer"

    def __init__(self) -> None:
        self._config = None
        self._event_bus = None
        self._budget: Dict[str, Any] = {}
        self._initialized = False
        self._lock = threading.RLock()

        self._checkpoint_dir: Optional[Path] = None
        self._cancel_requested = False
        self._optimizer: Optional[Any] = None
        self._scheduler: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._global_step = 0
        self._last_result: Optional[TrainingResult] = None
        #: En iyi kayıp anındaki ağırlıklar (ıraksama koruması)
        self._en_iyi_durum: Optional[Dict[str, Any]] = None
        self._en_iyi_kayip: float = 0.0
        self._en_iyi_adim: int = 0
        #: Mutlak kayıp eşiği — bu değerin üzeri "bozulma"
        #: sayılır ve göreli iyileşmeye bakılmaz (ölçülen hata:
        #: 0,083 → 2,28 → 0,51 "iyileşti" sanılıp commit edildi)
        self._mutlak_kayip_esigi: float = 0.5

        self._counter = {
            "runs": 0, "steps": 0, "tokens": 0, "checkpoints": 0,
            "resumes": 0, "divergences": 0, "errors": 0,
            "total_ms": 0.0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None,
                   budget: Optional[Dict[str, Any]] = None,
                   checkpoint_dir: Optional[str] = None) -> None:
        """
        Eğiticiyi hazırlar.

        Args:
            config: SystemConfig
            event_bus: EventBus
            budget: Kaynak bütçesi (`ResourceBudget.to_dict()`)
            checkpoint_dir: Checkpoint dizini (None = `<data>/checkpoints/neural`)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._budget = dict(budget or {})

        if checkpoint_dir:
            self._checkpoint_dir = Path(checkpoint_dir)
        else:
            base = getattr(config, "data_path", None)
            if not base:
                base_path = getattr(config, "base_path", None) or os.getcwd()
                base = str(Path(base_path) / "data")
            self._checkpoint_dir = Path(base) / "checkpoints" / "neural"

        try:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"Checkpoint dizini oluşturulamadı: {exc}")
            self._checkpoint_dir = None

        self._initialized = True
        logger.info(
            f"Eğitici hazır — cihaz {self._budget.get('device', 'cpu')}, "
            f"checkpoint: {self._checkpoint_dir}"
        )

    def shutdown(self) -> None:
        """Eğiticiyi kapatır."""
        self._initialized = False
        logger.info(
            f"Eğitici kapatıldı ({self._counter['runs']} tur, "
            f"{self._counter['steps']} adım, {self._counter['tokens']:,} token)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Eğitim istatistikleri."""
        return {
            **self._counter,
            "initialized": self._initialized,
            "global_step": self._global_step,
            "avg_ms_per_step": round(
                self._counter["total_ms"] / self._counter["steps"], 2
            ) if self._counter["steps"] else 0.0,
            "checkpoint_dir": str(self._checkpoint_dir or ""),
            "budget": dict(self._budget),
            "last_result": (
                self._last_result.to_dict() if self._last_result else None
            ),
        }

    def cancel(self) -> None:
        """Eğitimi iptal eder (sonraki adımda durur)."""
        self._cancel_requested = True
        logger.info("Eğitim iptali istendi")

    def update_budget(self, budget: Dict[str, Any]) -> None:
        """Kaynak bütçesini günceller (bellek baskısı değiştiğinde)."""
        self._budget = dict(budget or {})
        logger.debug(f"Eğitim bütçesi güncellendi: {self._budget.get('memory_pressure')}")

    # ─── Planlama ───────────────────────────────────────────────────────

    def plan_training(
        self,
        estimated_samples: int = 0,
        *,
        steps: Optional[int] = None,
        priority_tags: Optional[List[str]] = None,
        quality: float = 0.5,
        model: Any = None,
        rationale: str = "",
        data_tokens: int = 0,
    ) -> TrainingPlan:
        """
        Kaynak bütçesine göre eğitim planı üretir.

        Kurallar:
            * Toplu iş boyutu: VRAM/RAM'e göre; bellek baskısı yüksekse küçülür.
            * Birikim adımı: küçük toplu işte etkin boyutu korumak için artar.
            * Öğrenme oranı: kalite düşükse daha yüksek (daha çok öğrenmeli),
              kalite yüksekse daha düşük (ince ayar).
            * Adım sayısı: örnek sayısı ve kaliteye göre; küçük veri kümesinde
              aşırı öğrenmeyi önlemek için sınırlanır.
            * AMP: yalnızca CUDA'da; gradient checkpointing bellek baskısında.

        Args:
            estimated_samples: Eğitim örneği sayısı
            steps: Adım sayısını zorla
            priority_tags: Öncelikli etiketler (günlük/rapor için)
            quality: Mevcut ortalama yanıt kalitesi (0..1)
            model: `NeuroCoreModel` (bellek tahmini için)
            rationale: Plan gerekçesi (evrim katmanı doldurur)

        Returns:
            TrainingPlan
        """
        pressure = str(self._budget.get("memory_pressure", "NORMAL"))
        device = str(self._budget.get("device", "cpu"))
        max_batch = int(self._budget.get("max_batch_size", 8) or 8)
        max_ram_mb = float(self._budget.get("max_ram_mb", 0.0) or 0.0)
        max_gpu_mb = float(self._budget.get("max_gpu_mb", 0.0) or 0.0)

        # ── Toplu iş ──
        batch_size = max(1, min(max_batch, 32))
        if pressure == "CRITICAL":
            batch_size = max(1, batch_size // 4)
        elif pressure == "WARNING":
            batch_size = max(1, batch_size // 2)

        # ── Dizi uzunluğu ──
        seq_len = 256
        if pressure in ("WARNING", "CRITICAL"):
            seq_len = 128
        if max_gpu_mb and max_gpu_mb < 2048:
            seq_len = min(seq_len, 128)

        # ── Birikim ──
        target_effective = 16
        grad_accumulation = max(1, min(8, target_effective // max(1, batch_size)))

        # ── Öğrenme oranı ──
        base_lr = float(getattr(getattr(self._config, "neural", None), "learning_rate", 1e-4) or 1e-4)
        if quality < 0.4:
            learning_rate = base_lr * 3.0
            lr_reason = "kalite düşük — daha güçlü öğrenme"
        elif quality < 0.7:
            learning_rate = base_lr * 1.5
            lr_reason = "kalite orta — normal öğrenme"
        else:
            learning_rate = base_lr * 0.6
            lr_reason = "kalite yüksek — ince ayar"

        # ── Adım sayısı ──
        #
        # ⚠ Sıra önemli: adım sayısı, aşağıdaki **doygunluk** hesabından
        # ÖNCE çözülür. Doygunluk artık "bu turda veri kaç kez görülecek"
        # sorusunu sorar; bu da adım sayısına bağlıdır.
        if steps is not None:
            resolved_steps = max(1, int(steps))
        elif estimated_samples > 0:
            effective = batch_size * grad_accumulation
            epochs = 3.0 if quality < 0.5 else 1.5
            resolved_steps = max(10, min(2000, int(estimated_samples * epochs / effective)))
        else:
            resolved_steps = 50

        # ── Doygunluk koruması ──
        #
        # Amaç: aynı veriyi **çok kez** göreceksek öğrenme oranını küçült
        # (ezberlenmiş veride büyük adım modeli bozar; ölçüldü: kayıp
        # 0,718 → 1,030 çıktı).
        #
        # ⚠ AŞAMA 0'DA ÖLÇÜLEN HATA (düzeltildi):
        #     `gorulme` payında **ömür boyu** sayaç kullanılıyordu:
        #         gorulen_token = model.growth.total_tokens_seen   # 90,9M
        #         veri_token    = estimated_samples * 90           # 1,8M
        #         → gorulme = 50,5  →  lr = base × 0.05 = 5e-6
        #     Bu sayaç modelin **tüm hayatında** işlediği token'ı sayar;
        #     mevcut veri kümesiyle ilgisi yoktur. Sonuç: model yaşlandıkça
        #     oran şişiyor ve lr **kalıcı olarak 30× düşüyordu** — öğrenme
        #     duruyordu (ölçüldü: 300 adımda val kaybı düşmüyordu).
        #
        # ✅ Doğrusu: pay **bu turda** planlanan token hacmi (adım ×
        #     gerçek etkin toplu iş × dizi), payda ise **bu veri kümesinin**
        #     gerçek token sayısı (`data_tokens`). Yani oran = bu turda veri
        #     kaç kez görülecek (epoch).
        bu_tur_token = float(resolved_steps) * float(
            batch_size * grad_accumulation * seq_len)
        if data_tokens and data_tokens > 0:
            veri_token = float(data_tokens)
            veri_kaynak = "ölçülen (data_tokens)"
        else:
            # ⚠ Yedek: gerçek token sayısı verilmediyse örnek başına
            # ~90 token varsayılır (kasa karışık: bilgi + kod).
            veri_token = float(estimated_samples) * 90.0
            veri_kaynak = "tahmin (örnek × 90)"
        gorulme = (bu_tur_token / veri_token) if veri_token > 0 else 0.0
        if gorulme > 0:
            logger.info(
                f"Doygunluk: bu tur {bu_tur_token / 1e6:.2f}M token · "
                f"veri {veri_token / 1e6:.2f}M ({veri_kaynak}) → "
                f"{gorulme:.2f}× geçiş"
            )
        if gorulme >= 20.0:
            # ⚠ Ölçüldü: 88× görülmüş veride `0,2×base (2e-5)` bile
            # ıraksattı (kayıp 0,39 → 0,56). Ezberlenmiş veride tek doğru
            # hamle **çok küçük** adımdır.
            learning_rate = base_lr * 0.05
            lr_reason = (f"veri {gorulme:.0f}× görüldü (ağır ezber) — "
                         f"minimal adım")
            logger.info(f"Doygunluk koruması: lr={learning_rate:.2e} ({lr_reason})")
        elif gorulme >= 5.0:
            learning_rate = base_lr * 0.2
            lr_reason = (f"veri {gorulme:.0f}× görüldü (ezber) — çok ince ayar")
            logger.info(f"Doygunluk koruması: lr={learning_rate:.2e} ({lr_reason})")
        elif gorulme >= 2.0:
            learning_rate = base_lr * 0.5
            lr_reason = (f"veri {gorulme:.1f}× görüldü — ince ayar")
            logger.info(f"Doygunluk koruması: lr={learning_rate:.2e} ({lr_reason})")

        # ── Bellek doğrulaması ──
        amp = device.startswith("cuda") and HAS_TORCH
        gradient_checkpointing = pressure in ("WARNING", "CRITICAL")
        # `memory` her durumda tanımlı olmalı: aşağıdaki gözcü ayarı ve
        # eğitim döngüsü bu sözlüğü koşulsuz okur.
        memory: Dict[str, Any] = {}

        if model is not None and hasattr(model, "memory_estimate"):
            try:
                memory = model.memory_estimate()
                needed = float(memory.get("training_adam_mb", 0.0) or 0.0)
                budget_mb = max_gpu_mb if device.startswith("cuda") else max_ram_mb
                if budget_mb and needed > budget_mb * 1.2:
                    # Sığmıyor: önce birikimi artır, sonra toplu işi küçült
                    grad_accumulation = min(16, grad_accumulation * 2)
                    batch_size = max(1, batch_size // 2)
                    gradient_checkpointing = True
                    rationale = (
                        f"{rationale} | bellek düzeltmesi: tahmini {needed:.0f}MB > "
                        f"bütçe {budget_mb:.0f}MB → batch={batch_size}, accum={grad_accumulation}"
                    ).strip(" |")
            except Exception as exc:
                logger.debug(f"Bellek doğrulaması atlandı: {exc}")

        plan = TrainingPlan(
            steps=resolved_steps,
            batch_size=batch_size,
            grad_accumulation=grad_accumulation,
            learning_rate=learning_rate,
            weight_decay=float(getattr(getattr(self._config, "neural", None),
                                      "weight_decay", 0.01) or 0.01),
            warmup_steps=max(1, min(resolved_steps // 10, 50)),
            schedule="cosine" if resolved_steps > 100 else "linear",
            max_grad_norm=1.0,
            use_amp=amp,
            gradient_checkpointing=gradient_checkpointing,
            seq_len=seq_len,
            save_every=max(10, resolved_steps // 4),
            log_every=max(1, min(10, resolved_steps // 10)),
            seed=1337,
            rationale=(
                f"{rationale} | baskı={pressure}, cihaz={device}, {lr_reason}"
            ).strip(" |"),
            priority_tags=list(priority_tags or []),
        )
        logger.info(
            f"Eğitim planı: {plan.steps} adım × {plan.batch_size}×{plan.grad_accumulation} "
            f"({plan.tokens_per_step} token/adım), lr={plan.learning_rate:.2e}, "
            f"seq={plan.seq_len}, amp={plan.use_amp}"
        )
        return plan

    # ─── Eğitim ─────────────────────────────────────────────────────────

    def train(
        self,
        model: Any,
        dataset: TrainingDataset,
        plan: Optional[TrainingPlan] = None,
        *,
        validation: Optional[TrainingDataset] = None,
        max_minutes: Optional[float] = None,
        target_loss: Optional[float] = None,
        user_initiated: bool = False,
        on_step: Optional[Callable[[TrainingMetrics], None]] = None,
        save_checkpoints: bool = True,
    ) -> TrainingResult:
        """
        Eğitim döngüsünü çalıştırır.

        Args:
            model: `NeuroCoreModel` (backbone + tokenizer içerir)
            dataset: `TrainingDataset`
            plan: `TrainingPlan` (None ise otomatik üretilir)
            validation: Doğrulama kümesi (kayıp izleme)
            max_minutes: Süre sınırı (aşılırsa durur)
            target_loss: Bu kaybın altına inince erken dur
            on_step: Her adımda çağrılan geri bildirim
            save_checkpoints: Checkpoint yaz

        Returns:
            TrainingResult
        """
        result = TrainingResult()
        if not HAS_TORCH:
            result.stop_reason = "torch_yok"
            logger.error("Eğitim için torch gerekir")
            return result
        if model is None or not getattr(model, "is_ready", False):
            result.stop_reason = "model_hazir_degil"
            return result
        if dataset is None or dataset.is_empty:
            result.stop_reason = "veri_yok"
            logger.warning("Eğitim verisi boş — tur atlandı")
            return result

        if plan is None:
            plan = self.plan_training(estimated_samples=len(dataset), model=model)

        result.plan = plan.to_dict()
        result.steps_requested = plan.steps
        self._cancel_requested = False
        self._counter["runs"] += 1
        start_wall = time.perf_counter()

        # ── Kaynak gözcüsü ("boştakinin %90'ı") ─────────────────────────
        guard = ResourceGuard(
            factor=float(self._budget.get("headroom_factor", 0.90) or 0.90),
            enabled=bool(self._budget.get("enforce_headroom", True)),
        )

        # Kaynaklar zaten doluysa hiç başlama — kullanıcının sistemini yorma.
        initial_state = self._resource_state()
        initial_check = guard.check(**initial_state)
        if initial_check == "stop" or (
            initial_check in ("pause", "reduce")
            and self._budget.get("should_stall")
        ):
            result.stop_reason = "kaynaklar_dolu"
            result.guard = guard.to_dict()
            logger.warning(
                f"Eğitim başlatılmadı — {guard.last_reason or 'kaynaklar dolu'}. "
                "Sistem boşaldığında tekrar deneyin."
            )
            return result

        # ── Kendi belleğini düşman sanmayı önle ──
        #
        # ⚠ Ölçülen tuzak: eğitim kaçınılmaz olarak VRAM'i %94'e çıkarır
        # (ağırlıklar + AdamW durumları + etkinleştirmeler). Varsayılan
        # `max_gpu_percent=0.90` tavanı bunu "sistem dolu" sayar, gözcü
        # duraklar ve 600 saniye sonra **eğitimi iptal eder**. Sonuç: 400
        # adımlık istek 2 adımda biter.
        #
        # Doğru denge: plan **başlamadan** belleği doğrular
        # (`plan_training` → `memory_estimate`). Kullanıcı eğitimi açıkça
        # istediyse VRAM tavanı %98'e çıkarılır — %2 pay sistem için kalır
        # ve gerçekten taşarsa gözcü yine durdurur (aşağıdaki
        # `ceiling_hits` emniyeti).
        # ── Kendi VRAM payımızı ölç ──
        #
        # ⚠ Ölçülen tuzak: gözcü toplu işi küçülttükçe VRAM **düşmez**, çünkü
        # ağırlıklar + AdamW durumları toplu işten bağımsızdır. Gözcü bunu
        # görmezse 32→16→8→4→2→1 diye küçültür, sonra "küçültemiyorum" deyip
        # eğitimi **durdurur** (ölçüldü: 500 adımlık istek 26 adımda bitti).
        #
        # Doğru ölçü: gerçek GPU kullanımından **bizim modelimizin payını**
        # düşmek. Kalan kısım gerçekten başka uygulamalara aittir; kullanıcı
        # video açtıysa orada görünür.
        own_gpu_mb = 0.0
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _free, _total = _torch.cuda.mem_get_info()
                # Tahsis edilen (bizim) bellek — süreç başına
                own_gpu_mb = float(_torch.cuda.memory_allocated()) / 1024**2
        except Exception:
            own_gpu_mb = 0.0

        # ⚠ ÖLÇÜLEN KRİTİK HATA: burada `memory.get("training_adam_mb")`
        # çağrılıyordu ama `memory` **`plan_training()`'in yerel
        # değişkeniydi** — `train()` kapsamında tanımlı değil. Sonuç:
        #
        #     NameError: name 'memory' is not defined
        #
        # `except Exception` bloğu bunu yuttuğu için hata **sessizdi** ve
        # yalnız `user_initiated=True` yolunda (gözcü ayarı
        # `max_gpu_percent = 0.98`) ortaya çıkıyordu.
        #
        # Doğru davranış: tahsis edilen bellek okunamadıysa **modelden**
        # yeniden tahmin al (`plan_training` zaten aynı çağrıyı kullanır).
        if own_gpu_mb <= 0.0 and model is not None \
                and hasattr(model, "memory_estimate"):
            try:
                own_gpu_mb = float(
                    (model.memory_estimate() or {}).get(
                        "training_adam_mb", 0.0) or 0.0)
            except Exception:
                own_gpu_mb = 0.0

        if user_initiated:
            guard.max_gpu_percent = 0.98
            # Baskı olayı sınırı da yükseltilir: gözcü her 2 adımda ölçer ve
            # varsayılan 6 olay **12 adımda** turu bitirirdi. Kendi ayırdığı
            # VRAM yüzünden erken bitmesin; gerçek taşma yine yakalanır.
            guard.max_pressure_events = 200
            logger.info(
                "Kullanıcı istekli eğitim: VRAM tavanı %98, baskı sınırı 200 "
                "(plan belleği doğruladı, gözcü yine denetliyor)"
            )

        backbone = model.backbone
        if plan.gradient_checkpointing and hasattr(backbone, "set_gradient_checkpointing"):
            backbone.set_gradient_checkpointing(True)

        # Modeli ve tensörleri **aynı** cihaza hizala. Büyüme/yükleme sonrası
        # parametreler farklı bir cihazda kalmış olabilir; bu, ileri geçişte
        # "Expected all tensors to be on the same device" hatasına yol açar.
        if hasattr(model, "ensure_device"):
            try:
                resolved = model.ensure_device()
                logger.debug(f"Eğitim cihazı: {resolved}")
            except Exception as exc:
                logger.warning(f"Cihaz hizalanamadı: {exc}")

        # ── Optimizer + plan ──
        trainable = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
        if not trainable:
            result.stop_reason = "egitilebilir_parametre_yok"
            return result

        try:
            self._optimizer = torch.optim.AdamW(
                trainable, lr=plan.learning_rate, weight_decay=plan.weight_decay,
                betas=(0.9, 0.95), eps=1e-8,
            )
        except Exception as exc:
            self._counter["errors"] += 1
            result.stop_reason = f"optimizer_hatasi: {exc}"
            return result

        self._scheduler = self._build_scheduler(self._optimizer, plan)
        if plan.use_amp:
            self._scaler = self._make_scaler()
        else:
            self._scaler = None

        device = getattr(model, "_next_device", lambda: torch.device("cpu"))()
        backbone.train()
        losses: List[float] = []
        history: List[Dict[str, Any]] = []
        generator_seed = plan.seed
        step = 0
        diverged = False
        stop_reason = "completed"

        # ══ ⚠ DOĞRULAMA KAYBI — ölçüm altyapısı ═══════════════════════
        #
        # Neden gerekli? (ÖLÇÜM)
        #     `validation` parametresi kabul ediliyordu ama **döngüde hiç
        #     kullanılmıyordu**. Sonuç: eğitim kaybı 0,083'e inerken
        #     gerçek genelleme 6,44 (rastgele 8,45) kaldı ve bu **hiç
        #     fark edilmedi**. Tek başına eğitim kaybı, doymuş bir veri
        #     kümesinde her zaman iyi görünür.
        #
        # Davranış:
        #     · Başlangıçta bir kez ölçülür (kıyas tabanı)
        #     · `eval_every` adımda bir ölçülür (plan 0 ise 200)
        #     · Sonuç `val_history`'ye yazılır ve `TrainingResult`'a geçer
        eval_every = int(plan.eval_every or 0)
        if eval_every <= 0 and validation is not None:
            eval_every = 200
        #: ⚠ Son "en iyi ağırlık" anlık görüntüsünün adımı (performans
        #: kilidi: en fazla 20 adımda bir derin kopya alınır — 160MB).
        _anlik_adim = -10 ** 9
        if validation is not None:
            _v0 = self.evaluate(model, validation)
            if "loss" in _v0:
                result.initial_val_loss = float(_v0["loss"])
                result.best_val_loss = float(_v0["loss"])
                result.val_history.append({"step": 0, **_v0})
                logger.info(
                    f"Doğrulama (başlangıç): kayıp {_v0['loss']:.4f} · "
                    f"ppl {_v0.get('perplexity')}"
                )

        try:
            while step < plan.steps:
                if self._cancel_requested:
                    stop_reason = "iptal"
                    break
                if max_minutes is not None and \
                        (time.perf_counter() - start_wall) / 60.0 >= max_minutes:
                    stop_reason = "sure_siniri"
                    break

                # ── Kaynak nezaketi denetimi ────────────────────────────
                # Her adımda sistemin durumuna bakılır: kullanıcı tarayıcıda
                # video açtıysa ajan küçülür, gerekirse duraklar. Sistemin
                # kilitlenmesini önleyen asıl mekanizma budur.
                #
                # ⚠ Ölçülen tuzak: ham GPU yüzdesi **bizim** ağırlıklarımızı ve
                # AdamW durumlarını da içerir. Toplu işi küçültmek bunları
                # düşürmez, ama gözcü düşeceğini varsayıp 32→16→…→1 diye
                # küçültür ve sonunda "küçültemiyorum" deyip eğitimi
                # **durdurur** (ölçüldü: 500 adımlık istek 26 adımda bitti).
                #
                # Doğru ölçü: kendi payımızı düşmek. Kalan kısım gerçekten
                # başka uygulamalara aittir; kullanıcı video açtıysa orada
                # görünür ve nezaket politikası yine devreye girer.
                if guard.enabled and step % 2 == 0:
                    state = self._resource_state()
                    ham_gpu = state.get("gpu_percent")
                    if ham_gpu is not None and own_gpu_mb > 0:
                        try:
                            import torch as _t

                            _free, _total = _t.cuda.mem_get_info()
                            toplam_mb = float(_total) / 1024**2
                            if toplam_mb > 0:
                                # Bizim payımızı çıkar
                                disaridan = max(
                                    0.0,
                                    (toplam_mb - _free / 1024**2) - own_gpu_mb)
                                state["gpu_percent"] = min(
                                    100.0, disaridan / toplam_mb * 100.0)
                        except Exception:
                            pass
                    action = guard.check(**state)
                    if action == "stop":
                        stop_reason = "kaynak_baskisi"
                        logger.warning(f"Eğitim durduruldu — {guard.last_reason}")
                        break
                    if action == "reduce":
                        if guard.reduce_batch(plan):
                            logger.warning(
                                f"Kaynak baskısı — toplu iş küçültüldü: "
                                f"batch={plan.batch_size}, seq={plan.seq_len}, "
                                f"accum={plan.grad_accumulation}"
                            )
                        else:
                            stop_reason = "kaynak_baskisi"
                            break
                    elif action == "pause":
                        # Kullanıcıya yol ver: kaynaklar boşalana kadar bekle.
                        #
                        # ⚠ Bekleme süresi kritiktir. Kısa bir sınır (60 sn)
                        # eğitimi **kendi yarattığı** VRAM yüküyle iptal eder:
                        # PyTorch ayırdığı belleği süreç boyunca tutar, koruma
                        # bunu "sistem dolu" sanır ve eğitimi durdurur. Ölçüldü:
                        # 400 adımlık istek 2 adımda `kaynak_beklemesi_asildi`
                        # ile bitti. Kullanıcı eğitimi **açıkça** istediyse
                        # (user_initiated) daha uzun beklenir; kendi ayırdığı
                        # bellek yüzünden iş iptal edilmez.
                        wait_limit = 600.0 if user_initiated else 60.0
                        waited = 0.0
                        while waited < wait_limit and not self._cancel_requested:
                            time.sleep(guard.pause_seconds)
                            waited += guard.pause_seconds
                            if guard.check(**self._resource_state()) == "none":
                                break
                        guard.total_paused_seconds += waited
                        if waited >= wait_limit:
                            stop_reason = "kaynak_beklemesi_asildi"
                            logger.warning(
                                "Kaynaklar 60 saniye içinde boşalmadı — eğitim "
                                "durduruluyor (ilerleme checkpoint'lendi)"
                            )
                            break

                step_start = time.perf_counter()
                accumulated_loss = 0.0
                accumulated_aux = 0.0
                batches_done = 0
                tokens_this_step = 0
                self._optimizer.zero_grad(set_to_none=True)

                for micro in range(max(1, plan.grad_accumulation)):
                    batch = self._next_batch(dataset, plan.batch_size,
                                             shuffle=True,
                                             seed=None if generator_seed is None else generator_seed + step * 31 + micro)
                    if batch is None:
                        logger.warning(
                            "Toplu iş üretilemedi (veri kodlanamadı veya sözlük "
                            "uyumsuz) — eğitim durduruluyor"
                        )
                        if not history:
                            stop_reason = "veri_kodlanamadi"
                        break
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    tokens_this_step += int(input_ids.numel())

                    try:
                        if plan.use_amp and self._scaler is not None:
                            with torch.amp.autocast("cuda"):
                                output = backbone(input_ids, attention_mask=attention_mask,
                                                  labels=labels)
                                loss = (output["loss"] + output["aux_loss"]) / plan.grad_accumulation
                            self._scaler.scale(loss).backward()
                        else:
                            output = backbone(input_ids, attention_mask=attention_mask,
                                              labels=labels)
                            loss = (output["loss"] + output["aux_loss"]) / plan.grad_accumulation
                            loss.backward()
                    except RuntimeError as exc:
                        # Bellek yetersiz: toplu işi küçült ve devam et
                        if "out of memory" in str(exc).lower():
                            logger.warning("CUDA/CPU bellek yetersiz — toplu iş küçültülüyor")
                            self._optimizer.zero_grad(set_to_none=True)
                            plan.batch_size = max(1, plan.batch_size // 2)
                            if plan.batch_size == 1 and plan.grad_accumulation > 1:
                                plan.grad_accumulation = max(1, plan.grad_accumulation // 2)
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            break
                        raise

                    accumulated_loss += float(output["loss"].detach().item())
                    accumulated_aux += float(output["aux_loss"].detach().item())
                    batches_done += 1

                if batches_done == 0:
                    stop_reason = "veri_tukendi" if not history else stop_reason or "veri_tukendi"
                    if not history:
                        logger.warning("Hiç toplu iş üretilemedi — eğitim verisi kodlanamadı")
                    break

                # ── Kaynak kullanımını adım metriğine yaz (şeffaflık) ──
                memory_note = ""
                step_state = self._resource_state()
                step_ram = step_state.get("ram_percent")
                step_gpu = step_state.get("gpu_percent")
                if step_ram is not None or step_gpu is not None:
                    indicators = []
                    if step_ram is not None:
                        indicators.append(f"RAM %{step_ram:.0f}")
                    if step_gpu is not None:
                        indicators.append(f"VRAM %{step_gpu:.0f}")
                    memory_note = "sistem: " + ", ".join(indicators)

                # ── Gradyan kırpma + adım ──
                grad_norm = 0.0
                if self._scaler is not None:
                    self._scaler.unscale_(self._optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    trainable, plan.max_grad_norm
                ).item())
                if not math.isfinite(grad_norm):
                    diverged = True
                    self._counter["divergences"] += 1
                    logger.error(f"Adım {step}: gradyan NaN/Inf — adım atlandı")
                    self._optimizer.zero_grad(set_to_none=True)
                    if self._scheduler is not None:
                        self._scheduler.step()
                    # ⚠ GradScaler'ın `unscale_` bayrağı sıfırlanmalı.
                    # PyTorch bir optimizer için `unscale_` çağrıldıktan sonra
                    # `step()` çağrılmadan ikinci kez `unscale_` çağrılmasını
                    # reddeder:
                    #   "unscale_() has already been called on this optimizer
                    #    since the last update()"
                    # Bu dal `step()` çağırmadan `continue` ettiği için eğitim
                    # **kalıcı olarak** kırılıyordu (ölçüldü: 42 adımda
                    # `hata: unscale_() has already been called`).
                    # GradScaler'ı yeniden kurmak bayrağı temizler ve ölçeği
                    # makul bir değere döndürür.
                    self._scaler = self._make_scaler()
                    step += 1
                    continue

                if self._scaler is not None:
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    self._optimizer.step()
                if self._scheduler is not None:
                    self._scheduler.step()

                mean_loss = accumulated_loss / batches_done
                mean_aux = accumulated_aux / batches_done
                losses.append(mean_loss)
                self._global_step += 1
                step += 1

                current_lr = float(self._optimizer.param_groups[0]["lr"])
                metrics = TrainingMetrics(
                    step=step, loss=mean_loss, aux_loss=mean_aux, lr=current_lr,
                    grad_norm=grad_norm, tokens=tokens_this_step,
                    duration_ms=(time.perf_counter() - step_start) * 1000,
                    note=memory_note,
                )
                history.append(metrics.to_dict())

                # ⚠ Doğrulama kaybı — her `eval_every` adımda (Aşama 4).
                # `evaluate()` içinde `no_grad` var ve `backbone.eval()`
                # sonrası eğitim moduna geri döner; gradyan birikmez.
                if (validation is not None and eval_every > 0
                        and step % eval_every == 0):
                    _v = self.evaluate(model, validation)
                    if "loss" in _v:
                        result.val_history.append({"step": step, **_v})
                        _vl = float(_v["loss"])
                        if result.best_val_loss <= 0 or _vl < result.best_val_loss:
                            result.best_val_loss = _vl
                        logger.info(
                            f"adım {step} · DOĞRULAMA kaybı {_vl:.4f} · "
                            f"ppl {_v.get('perplexity')} · "
                            f"eğitim {mean_loss:.4f}"
                        )
                        try:
                            self._emit("neural.validation",
                                       {"step": step, **_v})
                        except Exception:
                            pass

                self._counter["steps"] += 1
                self._counter["tokens"] += tokens_this_step
                self._counter["total_ms"] += metrics.duration_ms
                if hasattr(model, "record_training_step"):
                    model.record_training_step(tokens=tokens_this_step, steps=1)

                if on_step is not None:
                    try:
                        on_step(metrics)
                    except Exception as exc:
                        logger.debug(f"on_step geri bildirimi hata verdi: {exc}")

                if step % max(1, plan.log_every) == 0:
                    logger.info(
                        f"adım {step}/{plan.steps} · kayıp {mean_loss:.4f} · "
                        f"aux {mean_aux:.4f} · lr {current_lr:.2e} · "
                        f"grad {grad_norm:.2f} · {metrics.duration_ms:.0f}ms"
                    )
                    self._emit("neural.training_step", metrics.to_dict())

                # ── Erken durma ──
                if target_loss is not None and mean_loss <= target_loss:
                    stop_reason = "hedef_kayip"
                    break
                if len(losses) >= 20:
                    recent = losses[-10:]
                    earlier = losses[-20:-10]
                    if sum(recent) / 10 > sum(earlier) / 10 * 1.5 and mean_loss > 1.0:
                        stop_reason = "iraksama_egilimi"
                        logger.warning("Kayıp artıyor — eğitim durduruldu")
                        break

                # ── En iyi ağırlıkları sakla (ıraksama koruması) ──
                #
                # ⚠ Ölçülen sorun: doymuş veride (92× görülmüş) eğitim
                # kaybı **artırabiliyor** — ölçüldü: 0,084 → 0,824.
                # `taze_egitim.py` yalnızca **tur sonuna** bakıp geri
                # alıyor, bu yüzden tur içindeki iyileşme kayboluyor.
                #
                # Çözüm: kayıp **en düşük** olduğu andaki ağırlıkları
                # bellekte tut. Böylece tur bittiğinde en iyi duruma
                # dönülebilir ve kısmi ilerleme **korunur**.
                if not losses or mean_loss < min(losses[:-1] or [mean_loss]):
                    # ⚠ Performans: her iyileşen adımda 160MB derin kopya
                    # almak eğitimi belirgin yavaşlatıyordu (ölçüldü:
                    # 40 adım 1,8 dk iken 300 adım 30,9 dk — adım başına
                    # 8× yavaşlama; iyileşme sık olduğu için kopya sık).
                    # Anlık görüntü en fazla 20 adımda bir alınır; kayıp
                    # yine **en iyi** değerde saklanır (granülerlik yeterli).
                    if (step - _anlik_adim) >= 20:
                        try:
                            import copy as _copy

                            self._en_iyi_durum = _copy.deepcopy(
                                {k: v.detach().cpu().clone() for k, v in
                                 model.backbone.state_dict().items()})
                            self._en_iyi_kayip = float(mean_loss)
                            self._en_iyi_adim = step
                            _anlik_adim = step
                        except Exception as exc:
                            logger.debug(
                                f"En iyi durum saklanamadı: {exc}")

                # ── Checkpoint ──
                if save_checkpoints and plan.save_every and step % plan.save_every == 0:
                    path = self.save_checkpoint(model, plan, step, float(mean_loss),
                                                history=history[-50:])
                    if path:
                        result.checkpoint_path = path

        except KeyboardInterrupt:  # pragma: no cover - etkileşimli kullanım
            stop_reason = "klavye_kesintisi"
            logger.warning("Eğitim kullanıcı tarafından kesildi")
        except Exception as exc:
            self._counter["errors"] += 1
            stop_reason = f"hata: {exc}"
            logger.error(f"Eğitim hatası: {exc}", exc_info=True)
            # ⚠ Hata ayıklama: `NC_DEBUG_TRAIN=1` iken istisna yutulmaz,
            # çağırana kadar çıkar. Aksi hâlde hata yalnız
            # `stop_reason = "hata: ..."` metnine dönüşür ve iz (traceback)
            # kaybolur — ölçüldü: `shape mismatch` hatasının kaynağı bu
            # yüzden bulunamadı.
            if os.environ.get("NC_DEBUG_TRAIN") == "1":
                raise

        # ── Sonuç ──
        result.steps_completed = len(losses)
        result.initial_loss = losses[0] if losses else 0.0

        # ── En iyi ağırlıklara dön ──
        #
        # ⚠ Neden gerekli: doymuş veride (92× görülmüş) kayıp tur içinde
        # düşüp sonra **artabiliyor** (ölçüldü: 0,084 → 0,824). Tur sonu
        # ağırlıklarını kaydetmek modeli **bozar**. En iyi kayıp anındaki
        # ağırlıklara dönmek kısmi ilerlemeyi korur.
        en_iyi = getattr(self, "_en_iyi_durum", None)
        en_iyi_kayip = float(getattr(self, "_en_iyi_kayip", 0.0) or 0.0)
        son_kayip = losses[-1] if losses else 0.0
        geri_donuldu = False

        # ── Mutlak kayıp eşiği ──
        #
        # ⚠ ÖLÇÜLEN KRİTİK HATA: koruma yalnız **göreli** iyileşmeye
        # bakıyordu (`son > en_iyi × 1,05`). Ölçüldü: eğitim 0,083'ten
        # 2,28'e sıçradı, sonra 0,51'e "düştü" — göreli olarak
        # **iyileşme** sayıldı ve model **commit edildi**. Sonuç: model
        # bozuldu (bağımsız ölçümde kayıp 4,99).
        #
        # Düzeltme: **mutlak eşik**. Sağlıklı kayıp 0,06-0,09
        # aralığındadır; bunun katları üzeri **bozulma** demektir ve
        # göreli iyileşme yeterli sayılmaz.
        mutlak_esik = max(self._mutlak_kayip_esigi,
                          en_iyi_kayip * 3.0 if en_iyi_kayip > 0 else 0.0)
        if losses and mutlak_esik > 0 and son_kayip > mutlak_esik:
            logger.error(
                f"Mutlak kayıp eşiği aşıldı: {son_kayip:.4f} > "
                f"{mutlak_esik:.4f} — model bozulmuş sayılıyor")
            stop_reason = "mutlak_kayip_asildi"
            if en_iyi and en_iyi_kayip > 0:
                try:
                    model.backbone.load_state_dict(
                        {k: v.to(next(model.backbone.parameters()).device)
                         for k, v in en_iyi.items()})
                    geri_donuldu = True
                    logger.info(
                        f"Bozulmadan önceki ağırlıklara dönüldü: "
                        f"{son_kayip:.4f} → {en_iyi_kayip:.4f}")
                except Exception as exc:
                    logger.debug(f"Geri dönülemedi: {exc}")

        if not geri_donuldu and en_iyi and en_iyi_kayip > 0 \
                and son_kayip > en_iyi_kayip * 1.05:
            try:
                model.backbone.load_state_dict(
                    {k: v.to(next(model.backbone.parameters()).device)
                     for k, v in en_iyi.items()})
                logger.info(
                    f"En iyi ağırlıklara dönüldü: kayıp {son_kayip:.4f} → "
                    f"{en_iyi_kayip:.4f} (adım {getattr(self, '_en_iyi_adim', 0)})")
                geri_donuldu = True
                result.stop_reason = "en_iyi_agirliklara_donuldu"
                self._counter["geri_donus"] = (
                    self._counter.get("geri_donus", 0) + 1)
            except Exception as exc:
                logger.debug(f"En iyi ağırlıklara dönülemedi: {exc}")
        # Saklanan durumu bırak (bellek)
        self._en_iyi_durum = None
        self._en_iyi_kayip = 0.0

        result.final_loss = en_iyi_kayip if geri_donuldu else son_kayip
        # ⚠ Doğrulama: en son ölçülen değer (Aşama 4)
        if result.val_history:
            _son_val = result.val_history[-1].get("loss")
            if _son_val is not None:
                result.final_val_loss = float(_son_val)
        result.best_loss = min(losses) if losses else 0.0
        result.mean_loss = sum(losses) / len(losses) if losses else 0.0
        result.tokens_seen = sum(entry.get("tokens", 0) for entry in history)
        result.duration_ms = (time.perf_counter() - start_wall) * 1000
        result.throughput_tokens_per_sec = (
            result.tokens_seen / (result.duration_ms / 1000) if result.duration_ms > 0 else 0.0
        )
        result.grad_norm_max = max((entry.get("grad_norm", 0.0) for entry in history), default=0.0)
        result.diverged = diverged
        result.early_stopped = stop_reason in ("hedef_kayip", "iraksama_egilimi", "sure_siniri")
        result.stop_reason = stop_reason
        result.history = history
        # Gözcü raporu her zaman doldurulur: kaynak kullanımı kullanıcıya
        # görünür olmalı (bir adım bile çalışmadıysa da).
        guard_state = self._resource_state()
        result.guard = guard.to_dict()
        result.guard["final_state"] = " | ".join(
            f"{key} %{value:.0f}" for key, value in (
                ("RAM", guard_state.get("ram_percent")),
                ("CPU", guard_state.get("cpu_percent")),
                ("VRAM", guard_state.get("gpu_percent")),
            ) if value is not None
        )
        self._last_result = result

        # Son checkpoint
        if save_checkpoints and result.steps_completed > 0:
            path = self.save_checkpoint(model, plan, result.steps_completed,
                                        result.final_loss, history=history[-50:])
            if path:
                result.checkpoint_path = path

        if plan.gradient_checkpointing and hasattr(backbone, "set_gradient_checkpointing"):
            backbone.set_gradient_checkpointing(False)
        backbone.eval()

        self._emit("neural.training_finished", result.to_dict())
        logger.info(
            f"Eğitim tamamlandı: {result.steps_completed}/{plan.steps} adım, "
            f"kayıp {result.initial_loss:.4f} → {result.final_loss:.4f}, "
            f"{result.tokens_seen:,} token, {result.duration_ms / 1000:.1f}s "
            f"({stop_reason})"
        )
        return result

    def _make_scaler(self) -> Any:
        """
        Yeni bir `GradScaler` üretir (AMP için).

        Yeniden kurmak, `unscale_` bayrağını temizler ve ölçek çarpanını
        makul bir değere döndürür. NaN gradyan görülen adımdan sonra eğitimin
        kalıcı olarak kırılmasını önler.

        Returns:
            GradScaler veya None (AMP desteklenmiyorsa)
        """
        try:
            return torch.amp.GradScaler("cuda")
        except Exception:
            try:
                return torch.cuda.amp.GradScaler()
            except Exception:
                return None

    def _resource_state(self) -> Dict[str, Optional[float]]:
        """
        Anlık sistem kaynak kullanımını okur (gözcü için).

        Bu okuma **hafiftir**: psutil `interval=None` ile çağrılır (önceki
        örnekten bu yana geçen süreye göre hesaplar, bloklamaz). GPU bilgisi
        NVML'den alınır; yoksa atlanır.

        Returns:
            {"ram_percent": float|None, "cpu_percent": float|None,
             "gpu_percent": float|None}
        """
        if not getattr(self, "_guard_enabled", True):
            return {"ram_percent": None, "cpu_percent": None, "gpu_percent": None}
        state: Dict[str, Optional[float]] = {
            "ram_percent": None, "cpu_percent": None, "gpu_percent": None,
        }
        try:
            import psutil
            state["ram_percent"] = float(psutil.virtual_memory().percent)
            state["cpu_percent"] = float(psutil.cpu_percent(interval=None))
        except Exception as exc:
            logger.debug(f"Kaynak durumu okunamadı: {exc}")
        try:
            from neurocore.hardware.monitor import _get_gpu_info
            gpu = _get_gpu_info()
            if gpu:
                state["gpu_percent"] = float(gpu.get("vram_percent", 0.0))
        except Exception:
            pass
        return state

    def _next_batch(self, dataset: TrainingDataset, batch_size: int,
                    shuffle: bool, seed: Optional[int]) -> Optional[Dict[str, Any]]:
        """Veri kümesinden bir sonraki toplu işi üretir (sonsuz döngü)."""
        if not hasattr(self, "_batch_iterator") or self._batch_iterator is None or \
                getattr(self, "_batch_source", None) is not dataset or \
                getattr(self, "_batch_size", None) != batch_size:
            self._batch_iterator = dataset.batches(batch_size, shuffle=shuffle, seed=seed)
            self._batch_source = dataset
            self._batch_size = batch_size
        try:
            return next(self._batch_iterator)
        except StopIteration:
            # Veri kümesi bitti: baştan başla (epoch döngüsü)
            self._batch_iterator = dataset.batches(batch_size, shuffle=shuffle, seed=seed)
            try:
                return next(self._batch_iterator)
            except StopIteration:
                return None

    @staticmethod
    def _build_scheduler(optimizer: Any, plan: TrainingPlan) -> Optional[Any]:
        """Öğrenme oranı planını kurar (ısınma + kosinüs/lineer)."""
        if not HAS_TORCH:
            return None
        try:
            warmup = max(1, int(plan.warmup_steps))

            def lr_lambda(step: int) -> float:
                if step < warmup:
                    return (step + 1) / warmup
                progress = (step - warmup) / max(1, plan.steps - warmup)
                progress = min(1.0, max(0.0, progress))
                if plan.schedule == "cosine":
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                if plan.schedule == "linear":
                    return max(0.0, 1.0 - progress)
                return 1.0

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        except Exception as exc:
            logger.debug(f"LR planı kurulamadı: {exc}")
            return None

    # ─── Checkpoint ─────────────────────────────────────────────────────

    def save_checkpoint(self, model: Any, plan: Optional[TrainingPlan] = None,
                        step: int = 0, loss: float = 0.0,
                        history: Optional[List[Dict[str, Any]]] = None,
                        note: str = "") -> str:
        """
        Eğitim checkpoint'i yazar.

        Args:
            model: `NeuroCoreModel`
            plan: Kullanılan plan
            step: Adım numarası
            loss: Kayıp değeri
            history: Son metrikler
            note: Ek açıklama

        Returns:
            Yazılan dosya yolu (başarısızsa "")
        """
        if not HAS_TORCH or self._checkpoint_dir is None:
            return ""
        if model is None or not getattr(model, "is_ready", False):
            return ""

        payload = {
            "version": 1,
            "step": int(step),
            "loss": float(loss),
            "created_at": time.time(),
            "note": note,
            "plan": plan.to_dict() if plan else {},
            "model_state": model.backbone.state_dict(),
            "growth": model.growth.to_dict() if hasattr(model, "growth") else {},
            "optimizer_state": self._optimizer.state_dict() if self._optimizer else None,
            "scheduler_state": self._scheduler.state_dict() if self._scheduler else None,
            "history": history or [],
        }

        path = self._checkpoint_dir / "latest.pt"
        temp = path.with_name(path.name + ".tmp")
        try:
            torch.save(payload, temp)
            os.replace(temp, path)
            # Adım bazlı arşiv (son 5)
            stamped = self._checkpoint_dir / f"step_{step:07d}.pt"
            try:
                torch.save({key: value for key, value in payload.items()
                            if key not in ("optimizer_state", "scheduler_state")}, stamped)
                self._prune_checkpoints(keep=5)
            except Exception:
                pass
        except Exception as exc:
            self._counter["errors"] += 1
            logger.error(f"Checkpoint yazılamadı: {exc}")
            return ""

        self._counter["checkpoints"] += 1
        logger.debug(f"Checkpoint kaydedildi: {path} (adım {step}, kayıp {loss:.4f})")
        return str(path)

    def resume(self, model: Any, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Checkpoint'ten devam eder.

        Args:
            model: `NeuroCoreModel` (ağırlıklar buraya yüklenir)
            path: Checkpoint dosyası (None = `latest.pt`)

        Returns:
            {"resumed": bool, "step": int, "loss": float, "reason": str}
        """
        if not HAS_TORCH:
            return {"resumed": False, "reason": "torch_yok"}
        if self._checkpoint_dir is None:
            return {"resumed": False, "reason": "checkpoint_dizini_yok"}

        target = Path(path) if path else (self._checkpoint_dir / "latest.pt")
        if not target.exists():
            return {"resumed": False, "reason": "checkpoint_yok"}

        try:
            payload = torch.load(target, map_location="cpu", weights_only=False)
        except Exception as exc:
            self._counter["errors"] += 1
            return {"resumed": False, "reason": f"okuma_hatasi: {exc}"}

        if model is None or not getattr(model, "is_ready", False):
            return {"resumed": False, "reason": "model_hazir_degil"}

        try:
            missing, unexpected = model.backbone.load_state_dict(
                payload.get("model_state") or {}, strict=False
            )
            model._is_pretrained = True
            self._global_step = int(payload.get("step", 0) or 0)
            self._counter["resumes"] += 1
            logger.info(
                f"Checkpoint'ten devam: adım {self._global_step}, "
                f"kayıp {payload.get('loss', 0.0):.4f} "
                f"({len(missing)} eksik / {len(unexpected)} fazla parametre)"
            )
            return {
                "resumed": True,
                "step": self._global_step,
                "loss": float(payload.get("loss", 0.0) or 0.0),
                "reason": "",
                "missing": len(missing),
                "unexpected": len(unexpected),
                "plan": payload.get("plan") or {},
            }
        except Exception as exc:
            self._counter["errors"] += 1
            return {"resumed": False, "reason": f"yukleme_hatasi: {exc}"}

    def _prune_checkpoints(self, keep: int = 5) -> int:
        """Eski adım checkpoint'lerini siler."""
        if self._checkpoint_dir is None:
            return 0
        stamped = sorted(self._checkpoint_dir.glob("step_*.pt"))
        removed = 0
        for path in stamped[:-keep] if len(stamped) > keep else []:
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed

    def list_checkpoints(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Checkpoint listesi."""
        if self._checkpoint_dir is None:
            return []
        entries: List[Dict[str, Any]] = []
        for path in sorted(self._checkpoint_dir.glob("*.pt"), reverse=True)[:limit]:
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append({
                "name": path.name,
                "path": str(path),
                "bytes": stat.st_size,
                "megabytes": round(stat.st_size / (1024 ** 2), 2),
                "modified": stat.st_mtime,
                "is_latest": path.name == "latest.pt",
            })
        return entries

    # ─── Değerlendirme ──────────────────────────────────────────────────

    def evaluate(self, model: Any, dataset: TrainingDataset,
                 batch_size: int = 8, max_batches: int = 50) -> Dict[str, Any]:
        """
        Doğrulama kaybını ölçer (gradyan hesaplamadan).

        Args:
            model: `NeuroCoreModel`
            dataset: Doğrulama kümesi
            batch_size: Toplu iş boyutu
            max_batches: Maksimum toplu iş sayısı

        Returns:
            {"loss", "perplexity", "batches", "samples"}
        """
        if not HAS_TORCH or model is None or not getattr(model, "is_ready", False):
            return {"error": "kullanilamaz"}
        if dataset is None or dataset.is_empty:
            return {"error": "veri_yok"}

        backbone = model.backbone
        was_training = backbone.training
        backbone.eval()
        device = model._next_device()

        total_loss = 0.0
        batches = 0
        samples = 0
        try:
            with torch.no_grad():
                for batch in dataset.batches(batch_size, shuffle=False):
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    mask = batch["attention_mask"].to(device)
                    output = backbone(input_ids, attention_mask=mask, labels=labels)
                    loss = float(output["loss"].detach().item())
                    if math.isfinite(loss):
                        total_loss += loss
                        batches += 1
                        samples += int(input_ids.shape[0])
                    if batches >= max_batches:
                        break
        except Exception as exc:
            logger.error(f"Doğrulama hatası: {exc}")
            return {"error": str(exc)}
        finally:
            if was_training:
                backbone.train()

        mean_loss = total_loss / batches if batches else 0.0
        return {
            "loss": round(mean_loss, 6),
            "perplexity": round(math.exp(min(20.0, mean_loss)), 4),
            "batches": batches,
            "samples": samples,
        }

    # ─── Veri Hazırlama ─────────────────────────────────────────────────

    #: ⚠ KURAL 4 — eğitimde kullanılacak kasa notlarının türü.
    #:
    #: Ölçüldü (2026-10-28): kasada 6.061 not var; **2.937'si (%48,5)
    #: `kind: skill`** — ajanın "Çözüm: onay veriyorum", "Çözüm: ekle"
    #: gibi kendi ürettiği gürültü notları. Bunlar eğitime girerse model
    #: gürültüyü öğrenir.
    #:
    #: Varsayılan yalnız `reference` + `qa`. `NC_EGITIM_NOTE_TURLERI`
    #: ortam değişkeni ile genişletilebilir (ör. "reference,qa,concept").
    EGITIM_NOTE_TURLERI: Tuple[str, ...] = tuple(
        t.strip() for t in (
            os.environ.get("NC_EGITIM_NOTE_TURLERI") or "reference,qa"
        ).split(",") if t.strip())

    #: Notun eğitime girmesi için gereken en düşük güven
    EGITIM_NOTE_MIN_GUVEN: float = 0.6

    def build_dataset_from_vault(self, vault: Any, tokenizer: Any,
                                 seq_len: int = 256, max_notes: int = 500,
                                 include_summaries: bool = True) -> TrainingDataset:
        """
        Kasadaki notlardan eğitim verisi üretir.

        Notlar "başlık → gövde" ve "özet → gövde" çiftlerine dönüştürülür;
        böylece model kasanın **üslubunu ve bilgisini** öğrenir.

        Args:
            vault: `KnowledgeVault`
            tokenizer: `BPETokenizer`
            seq_len: Dizi uzunluğu
            max_notes: Kullanılacak maksimum not sayısı
            include_summaries: Özet→gövde çiftleri de üretilsin mi?

        Returns:
            TrainingDataset
        """
        dataset = TrainingDataset(tokenizer, seq_len)
        if vault is None:
            return dataset

        count = 0
        try:
            iterator = vault.iter_notes(limit=max_notes)
        except Exception as exc:
            logger.error(f"Kasa okunamadı: {exc}")
            return dataset

        # ⚠ KURAL 4: tür ve güven süzgeci (sayaçlarla raporlanır)
        atlanan_tur = 0
        atlanan_guven = 0

        for note in iterator:
            if not note.body or note.word_count < 5:
                continue
            tur = str(getattr(note, "kind", "") or "")
            if tur and tur not in self.EGITIM_NOTE_TURLERI:
                atlanan_tur += 1
                continue
            try:
                guven = float(getattr(note, "confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                guven = 0.0
            if guven < self.EGITIM_NOTE_MIN_GUVEN:
                atlanan_guven += 1
                continue
            # ── Ağırlık: güven + **kategori çarpanı** ──
            #
            # ⚠ Ölçülen sorun: ağırlık yalnız `0.5 + confidence` idi ve
            # tüm kategoriler eşit sayılıyordu. Ama kasa dağılımı çok
            # dengesiz:
            #
            #     kod_uretim     2291  (%55)
            #     learned         964  (%23)
            #     tercih          395  (%10)   ← karar katmanı
            #     api_referans    292  (%7)    ← API bilgisi
            #     kod_ogrenme     196  (%5)
            #
            # Tercih ve API notları **karar verme** için kritik ama sayıca
            # az; eşit ağırlıkla model bunları ihmal eder (ölçüldü: 377
            # tercih örneği eklendi, karar benchmark'ı **değişmedi**).
            #
            # Çözüm: kategoriye göre çarpan. Böylece az ama değerli
            # örnekler eğitimde daha sık görünür.
            KATEGORI_AGIRLIK = {
                "tercih": 4.0,        # karar/estetik yargı
                "blender_api": 2.0,   # resmî doküman API'si (1375 kayıt)
                "api_referans": 2.5,  # öğretmen üretimi API
                "kod_ogrenme": 2.0,   # kod tamamlama çiftleri
                "kod_uretim": 1.5,    # sentetik kod
            }
            kat = str(getattr(note, "category", "") or "")
            weight = (0.5 + note.confidence) * KATEGORI_AGIRLIK.get(kat, 1.0)
            dataset.add(
                prompt=f"# {note.title}\n",
                target=note.body,
                weight=weight,
                tags=list(note.tags),
                difficulty=0.3 if note.word_count < 100 else 0.7,
            )
            if include_summaries:
                summary = note.summary
                if summary and len(summary) > 40 and summary not in note.body:
                    dataset.add(
                        prompt=f"{note.title} konusunu açıkla:\n",
                        target=summary,
                        weight=weight * 0.8,
                        tags=list(note.tags),
                        difficulty=0.2,
                    )
            count += 1
        logger.info(f"Kasadan {count} not ile eğitim verisi üretildi ({len(dataset)} örnek)")
        return dataset

    def build_dataset_from_experiences(self, batch: Dict[str, Any],
                                       tokenizer: Any, seq_len: int = 256) -> TrainingDataset:
        """
        Evrim katmanının deneyim paketinden veri kümesi üretir.

        Args:
            batch: `CurriculumBuilder.to_training_pairs()` çıktısı
            tokenizer: `BPETokenizer`
            seq_len: Dizi uzunluğu
        """
        dataset = TrainingDataset(tokenizer, seq_len)
        dataset.from_batch(batch or {})
        return dataset

    # ─── Sağlık ─────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Eğitici sağlık raporu."""
        issues: List[str] = []
        if not HAS_TORCH:
            issues.append("torch kurulu değil — eğitim devre dışı")
        if not self._initialized:
            issues.append("eğitici başlatılmadı")
        if self._checkpoint_dir is None:
            issues.append("checkpoint dizini yok")
        if self._counter["divergences"]:
            issues.append(f"{self._counter['divergences']} ıraksama olayı kaydedildi")
        if self._counter["errors"]:
            issues.append(f"{self._counter['errors']} eğitim hatası")
        return {
            "ok": self._initialized and HAS_TORCH,
            "runs": self._counter["runs"],
            "steps": self._counter["steps"],
            "tokens": self._counter["tokens"],
            "checkpoints": self._counter["checkpoints"],
            "issues": issues,
        }

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Trainer runs={self._counter['runs']} steps={self._counter['steps']} "
            f"tokens={self._counter['tokens']}>"
        )


__all__ = [
    "Trainer",
    "TrainingPlan",
    "TrainingMetrics",
    "TrainingResult",
    "TrainingSample",
    "TrainingDataset",
]
