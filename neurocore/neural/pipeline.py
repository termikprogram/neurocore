"""
Müfredat Eğitim Hattı
======================
Yerel modeli **yapılandırılmış müfredatla** eğiten bağımsız hat.

Neden ayrı modül? Ajan deneyimleri dar bir veri kümesi üretir (aynı tür
görevler, sınırlı sözcük dağarcığı). Temel kavramları öğretmek için
`neurocore.neural.curriculum` içeriği gerekir. Bu hat şu akışı yürütür:

    1. Müfredatı üret (katmanlı dersler + soru kalıpları)
    2. **Tokenizer'ı müfredat üzerinde yeniden eğit**
    3. Modelin **gömme katmanını** yeni sözlüğe göre yeniden boyutlandır
    4. Veri kümesini kur (müfredat çiftleri + kasa notları + ajan deneyimleri)
    5. Eğitimi çalıştır, ilerlemeyi diske yaz

Neden tokenizer yeniden eğitilir? Ölçüldü: mevcut sözlük (1.259) teknik
metinde **0,94 karakter/token** üretiyordu — neredeyse her karakter bir
token. Müfredatla yeniden eğitildiğinde **4,10 karakter/token** oluyor
(4,4× kazanç). Bu, aynı donanımda 4 kat fazla içerik öğrenmek demektir.

Kaynak nezaketi: Eğitim `NeuroCoreEngine.train` üzerinden yapılır; o da
`resources_available()` kapısını denetler. Sistem doluysa eğitim ertelenir
ve bu durum rapora yazılır — bilgisayar kilitlenmez.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from neurocore.neural.curriculum import (
    LAYER_LABELS,
    Lesson,
    build_curriculum,
    curriculum_stats,
    expand_curriculum,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.pipeline")

#: İlerleme dosyası (eğitim arka planda çalışırken durum buradan okunur)
PROGRESS_FILE = "curriculum_progress.json"

#: Sözlük hedefi. Ölçüm: müfredatla 8.000 hedefinde 4.301'e doyuyor
#: (daha fazla birleşim anlamlı token üretmiyor).
DEFAULT_VOCAB_SIZE = 8000


@dataclass
class PipelineProgress:
    """Eğitim hattının durumu (arayüz ve denetim için)."""
    stage: str = "bekliyor"
    started: float = 0.0
    updated: float = 0.0
    lessons: int = 0
    samples: int = 0
    tokens: int = 0
    vocab_before: int = 0
    vocab_after: int = 0
    steps_done: int = 0
    steps_target: int = 0
    loss_first: float = 0.0
    loss_last: float = 0.0
    total_model_steps: int = 0
    rounds: int = 0
    finished: bool = False
    error: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        data = {key: getattr(self, key) for key in self.__dataclass_fields__}
        data["elapsed_seconds"] = round(
            (self.updated or time.time()) - self.started, 1
        ) if self.started else 0.0
        return data


class CurriculumTrainer:
    """
    Müfredatla eğitim hattı.

    Kullanım:
        trainer = CurriculumTrainer()
        trainer.initialize(workspace="E:/proje", data_path=".neurocore")
        trainer.prepare()          # tokenizer + gömme boyutlandırma
        trainer.run_rounds(rounds=4, max_minutes=6)
        print(trainer.progress())
    """

    name = "neural.pipeline"

    def __init__(self, *, patterns: int = 8,
                 vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        self.patterns = max(1, int(patterns))
        self.vocab_size = max(512, int(vocab_size))

        self._workspace: Optional[Path] = None
        self._data_path: Optional[Path] = None
        self._engine: Any = None
        self._initialized = False
        self._lock = threading.RLock()
        self._progress = PipelineProgress()
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, *, workspace: str = "", data_path: str = "",
                   engine: Any = None) -> None:
        """Hattı başlatır."""
        if self._initialized:
            return
        self._workspace = Path(workspace).resolve() if workspace else Path.cwd()
        self._data_path = Path(data_path) if data_path else self._workspace / ".neurocore"
        self._data_path.mkdir(parents=True, exist_ok=True)
        self._engine = engine
        self._initialized = True
        stats = curriculum_stats(expand=True, patterns=self.patterns)
        logger.info(
            f"Müfredat hattı hazır — {stats['total_lessons']} ders, "
            f"{stats['total_chars']:,} karakter, hedef sözlük {self.vocab_size}"
        )

    def shutdown(self) -> None:
        """İşçiyi durdurur."""
        self._stop.set()
        worker = self._worker
        self._worker = None
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)
        self._initialized = False

    # ─── Yardımcılar ────────────────────────────────────────────────────

    def _release_vram_hint(self) -> None:
        """
        Eğitim öncesi VRAM'i boşaltmayı **dener**.

        Ağır iş sonrası referans döngüleri ve CUDA önbelleği belleği tutar;
        taahhüt edilen kaynak geri verilmezse kaynak nezaketi kapısı
        kilitlenir (ölçüldü: 1450 MiB kullanımda kalıyordu).
        """
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _engine_or_load(self) -> Any:
        """Motoru döndürür; yoksa yükler."""
        if self._engine is not None:
            return self._engine
        from neurocore import NeuroCoreEngine

        engine = NeuroCoreEngine()
        engine.initialize()
        self._engine = engine
        return engine

    @property
    def _progress_path(self) -> Path:
        """İlerleme dosyasının yolu."""
        assert self._data_path is not None
        return self._data_path / PROGRESS_FILE

    def _save_progress(self, stage: str = "") -> None:
        """İlerlemeyi diske yazar."""
        with self._lock:
            if stage:
                self._progress.stage = stage
            self._progress.updated = time.time()
            data = self._progress.to_dict()
        try:
            self._progress_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except OSError as exc:
            logger.debug(f"İlerleme yazılamadı: {exc}")

    def progress(self) -> Dict[str, Any]:
        """Güncel ilerleme (önce bellek, sonra disk)."""
        with self._lock:
            if self._progress.started:
                return self._progress.to_dict()
        try:
            if self._progress_path.exists():
                return json.loads(self._progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        return self._progress.to_dict()

    def lessons(self) -> List[Lesson]:
        """Genişletilmiş müfredat dersleri."""
        return expand_curriculum(patterns=self.patterns)

    # ─── Adım 1: Tokenizer ──────────────────────────────────────────────

    def prepare(self) -> Dict[str, Any]:
        """
        Tokenizer'ı müfredatla yeniden eğitir ve gömme katmanını boyutlandırır.

        Returns:
            {"ok", "vocab_before", "vocab_after", "efficiency", ...}
        """
        from neurocore.neural.tokenizer import BPETokenizer

        engine = self._engine_or_load()
        model = getattr(engine, "model", None)
        if model is None:
            return {"ok": False, "error": "model yüklenemedi"}

        lessons = self.lessons()
        documents = [lesson.as_document() for lesson in lessons]
        total_chars = sum(len(doc) for doc in documents)
        if not documents:
            return {"ok": False, "error": "müfredat boş"}

        with self._lock:
            self._progress.started = self._progress.started or time.time()
            self._progress.lessons = len(lessons)
            self._progress.stage = "tokenizer"
        self._save_progress()

        # ── Mevcut verimlilik (karşılaştırma için) ──
        old_tokenizer = getattr(model, "tokenizer", None)
        vocab_before = int(getattr(old_tokenizer, "vocab_size", 0) or 0)
        before_tokens = 0
        if old_tokenizer is not None:
            try:
                before_tokens = sum(len(old_tokenizer.encode(doc))
                                    for doc in documents)
            except Exception:
                before_tokens = 0

        # ── Yeni tokenizer ──
        fresh = BPETokenizer()
        try:
            fresh.initialize(getattr(model, "_config", None), None)
        except Exception as exc:
            logger.debug(f"Tokenizer initialize uyarısı: {exc}")
        stats = fresh.train(documents, vocab_size=self.vocab_size,
                            min_frequency=1, verbose=False)
        vocab_after = int(fresh.vocab_size)
        after_tokens = sum(len(fresh.encode(doc)) for doc in documents)

        # ── Modelin tokenizer'ını değiştir ──
        model.tokenizer = fresh
        try:
            fresh.initialize(getattr(model, "_config", None), None)
        except Exception:
            pass

        # ── Gömme katmanını boyutlandır ──
        backbone = model.backbone
        added = 0
        if hasattr(backbone, "resize_vocab"):
            added = int(backbone.resize_vocab(vocab_after))
        if model.model_config is not None:
            model.model_config.vocab_size = vocab_after

        result = {
            "ok": True,
            "lessons": len(lessons),
            "chars": total_chars,
            "vocab_before": vocab_before,
            "vocab_after": vocab_after,
            "rows_added": added,
            "tokens_before": before_tokens,
            "tokens_after": after_tokens,
            "chars_per_token_before": round(total_chars / max(1, before_tokens), 2),
            "chars_per_token_after": round(total_chars / max(1, after_tokens), 2),
            "gain": round(before_tokens / max(1, after_tokens), 2) if before_tokens else 0.0,
        }
        with self._lock:
            self._progress.vocab_before = vocab_before
            self._progress.vocab_after = vocab_after
            self._progress.tokens = after_tokens
            self._progress.notes.append(
                f"Tokenizer: {vocab_before} → {vocab_after} "
                f"({result['gain']}× verimli)"
            )
        self._save_progress("tokenizer_hazir")
        logger.info(
            f"Tokenizer hazır: {vocab_before} → {vocab_after} sözlük, "
            f"{result['gain']}× verimli, {added:+d} gömme satırı"
        )
        return result

    # ─── Adım 2: Eğitim ─────────────────────────────────────────────────

    def run_rounds(self, *, rounds: int = 4, max_minutes: float = 6.0,
                   steps_per_round: Optional[int] = None,
                   skip_prepare: bool = False,
                   fresh_engine_per_round: bool = True) -> Dict[str, Any]:
        """
        Eğitim turlarını çalıştırır.

        Args:
            rounds: Tur sayısı
            max_minutes: Tur başına süre sınırı
            steps_per_round: Tur başına adım (None = motor otomatik seçer)
            skip_prepare: Tokenizer adımını atla
            fresh_engine_per_round: Her turda motoru **kapat ve yeniden yükle**

        Not:
            `fresh_engine_per_round=True` varsayılandır çünkü **kaynak
            nezaketi** kapısı kendi yarattığı yükle kilitlenir: eğitim VRAM'i
            doldurur, sonraki turun ön kontrolü "%99 ≥ %90" görür ve eğitim
            `kaynaklar_dolu` ile ertelenir. Motoru turlar arasında bırakmak
            VRAM'i serbest bırakır ve kapı yeniden açılır.

            Ölçüldü: 8 turluk bir çalıştırmada, motor süreçte tutulduğunda
            yalnızca **1 tur** tamamlandı; kalan 7 tur `kaynaklar_dolu` ile
            atlandı.

        Returns:
            Özet rapor
        """
        engine = self._engine_or_load()
        model = getattr(engine, "model", None)
        if model is None:
            return {"ok": False, "error": "model yüklenemedi"}

        if not skip_prepare:
            prepared = self.prepare()
            if not prepared.get("ok"):
                return prepared

        # ── Müfredatı kasaya yaz: eğitim verisi kasadan üretilir ──
        written = self._seed_vault(engine)

        # ── VRAM'i eğitim için boşalt ──
        # Kasa notları gömme motoru üzerinden işlenir; o motor omurgayı
        # referansla tutabilir. Eğitim başlamadan önce serbest bırakılır.
        self._release_vram_hint()
        engine = self._engine_or_load()

        results: List[Dict[str, Any]] = []
        deferred = 0
        for index in range(max(1, rounds)):
            if self._stop.is_set():
                break
            self._save_progress(f"egitim_{index + 1}/{rounds}")
            logger.info(f"Müfredat eğitimi: tur {index + 1}/{rounds}")

            # ── Ön kontrol: kaynak gerçekten uygun mu? ──
            # Kapı kendi spike'ına takılmasın diye **iş başlamadan** ölçülür.
            if fresh_engine_per_round and index > 0:
                engine = self._reload_engine()

            ok, why = True, ""
            try:
                ok, why = engine.resources_available(for_heavy_work=True)
            except Exception as exc:
                logger.debug(f"Kaynak denetimi başarısız: {exc}")
            if not ok:
                deferred += 1
                with self._lock:
                    self._progress.notes.append(
                        f"Tur {index + 1} ertelendi: {why[:70]}"
                    )
                self._save_progress("kaynak_bekliyor")
                logger.info(f"Tur {index + 1} ertelendi — {why[:90]}")
                # Kaynak açılana kadar kısa bekleme (en fazla 3 deneme)
                for _ in range(3):
                    if self._stop.is_set():
                        break
                    time.sleep(20.0)
                    if fresh_engine_per_round:
                        engine = self._reload_engine()
                    ok, why = engine.resources_available(for_heavy_work=True)
                    if ok:
                        break
                if not ok:
                    with self._lock:
                        self._progress.error = f"kaynaklar_dolu: {why[:120]}"
                    break

            try:
                outcome = engine.train(
                    steps=steps_per_round,
                    from_vault=True,
                    from_experiences=True,
                    max_minutes=max_minutes,
                    # Kullanıcı eğitimi açıkça istedi: kaynak koruması kendi
                    # ayırdığı VRAM yüzünden işi iptal etmesin, beklesin.
                    user_initiated=True,
                )
            except Exception as exc:
                logger.error(f"Tur {index + 1} çöktü: {exc}", exc_info=True)
                outcome = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            if not isinstance(outcome, dict):
                outcome = {"ok": False, "error": "sonuç sözlük değil"}
            results.append(outcome)

            if not outcome.get("ok"):
                reason = str(outcome.get("error") or outcome.get("stop_reason") or "")
                with self._lock:
                    self._progress.error = reason[:200]
                    self._progress.notes.append(
                        f"Tur {index + 1} başarısız: {reason[:70]}")
                self._save_progress("hata")
                if "kaynak" in reason.lower() or "dolu" in reason.lower():
                    deferred += 1
                    continue
                continue

            with self._lock:
                self._progress.rounds += 1
                self._progress.steps_done += int(outcome.get("steps_completed", 0) or 0)
                self._progress.loss_first = self._progress.loss_first or float(
                    outcome.get("initial_loss", 0.0) or 0.0)
                self._progress.loss_last = float(outcome.get("final_loss", 0.0) or 0.0)
                self._progress.steps_target = int(outcome.get("steps_requested", 0) or 0)
            self._save_progress(f"tur_{index + 1}_bitti")

        # ── Modeli kaydet ──
        saved: Dict[str, Any] = {}
        try:
            current_model = getattr(engine, "model", None) or model
            saved = current_model.save() if hasattr(current_model, "save") else {}
        except Exception as exc:
            logger.warning(f"Model kaydedilemedi: {exc}")

        growth = getattr(getattr(engine, "model", None), "growth", None)
        with self._lock:
            self._progress.total_model_steps = int(
                getattr(growth, "total_training_steps", 0) or 0)
            self._progress.finished = True
            self._progress.stage = "bitti" if not self._progress.error else "kısmi"
        self._save_progress()

        summary = {
            "ok": any(item.get("ok") for item in results),
            "rounds": len(results),
            "rounds_ok": sum(1 for item in results if item.get("ok")),
            "deferred": deferred,
            "lessons": len(self.lessons()),
            "vault_notes": written,
            "steps_total": sum(int(item.get("steps_completed", 0) or 0)
                               for item in results),
            "loss_first": results[0].get("initial_loss") if results else None,
            "loss_last": results[-1].get("final_loss") if results else None,
            "model_steps": self._progress.total_model_steps,
            "saved": bool(saved),
            "progress": self._progress.to_dict(),
        }
        logger.info(
            f"Müfredat eğitimi bitti: {summary['rounds_ok']}/{summary['rounds']} tur "
            f"({deferred} ertelendi), {summary['steps_total']} adım, "
            f"model {summary['model_steps']} adımda"
        )
        return summary

    def _reload_engine(self) -> Any:
        """
        Motoru kapatıp yeniden yükler (VRAM'i **gerçekten** serbest bırakır).

        Turlar arasında motoru bırakmak, kaynak nezaketi kapısının kendi
        yarattığı doluluk yüzünden kilitlenmesine yol açıyordu.

        ⚠ Ölçüldü: `engine.shutdown()` tek başına VRAM'i **bırakmıyor**
        (1450 MiB'de kalıyor) — çünkü gömme motoru ve referans döngüleri
        omurgayı canlı tutuyor. `gc.collect()` + `torch.cuda.empty_cache()`
        ile 1450 → 554 MiB'e iniyor. Bu adım olmadan sonraki turun ön
        kontrolü `VRAM %98 ≥ %90` görüp eğitimi erteliyor.
        """
        import gc

        old = self._engine
        if old is not None:
            try:
                model = getattr(old, "model", None)
                if model is not None and hasattr(model, "save"):
                    model.save()
            except Exception as exc:
                logger.debug(f"Tur arası kayıt başarısız: {exc}")
            # Gömme motoru omurgayı referansla tutabiliyor; bağını kopar
            try:
                embeddings = getattr(old, "neural", None)
                if embeddings is not None and hasattr(embeddings, "_model"):
                    embeddings._model = None
            except Exception:
                pass
            try:
                old.shutdown()
            except Exception as exc:
                logger.debug(f"Motor kapatılamadı: {exc}")
        self._engine = None

        # ── VRAM'i gerçekten boşalt ──
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                free, total = torch.cuda.mem_get_info()
                logger.info(
                    f"VRAM boşaltıldı: {(total - free) / 1024**2:.0f} MiB kullanımda, "
                    f"{free / 1024**2:.0f} MiB boş"
                )
        except Exception as exc:
            logger.debug(f"CUDA temizliği yapılamadı: {exc}")
        time.sleep(1.0)
        return self._engine_or_load()

    def _seed_vault(self, engine: Any) -> int:
        """
        Müfredat derslerini **bilgi kasasına** yazar.

        Neden? Eğitim verisi kasadan üretilir (`build_dataset_from_vault`).
        Dersleri kasaya yazmak hem eğitime sokar hem ajan bağlamında
        kullanılabilir kılar.

        ⚠ **Genişletilmiş** müfredat yazılır (464 ders), yalnızca temel 58
        değil. Ölçüldü: 58 not ≈ 57 bin token üretiyor ve eğitim **tek
        epoch'ta** (≈7 adım) duruyor — 80 adımlık istek boşa gidiyor.
        Genişletilmiş içerikle veri kümesi ~8× büyür.

        Not: Aynı dersin farklı soru kalıpları **farklı başlık** taşır
        (başlık = soru), bu yüzden kasa tekrar saymaz.
        """
        knowledge = getattr(engine, "knowledge", None)
        if knowledge is None:
            return 0
        written = 0
        skipped = 0
        for lesson in self.lessons():
            try:
                knowledge.write(
                    lesson.prompt(), lesson.target(),
                    kind="concept",
                    tags=["müfredat", lesson.layer] + lesson.tags[:3],
                    confidence=0.9, importance=0.75,
                    sources=["curriculum://neurocore"],
                    dedupe=True,
                )
                written += 1
            except Exception as exc:
                skipped += 1
                logger.debug(f"Ders yazılamadı ({lesson.title[:40]}): {exc}")
        logger.info(
            f"Müfredat kasaya yazıldı: {written} ders "
            f"({skipped} atlandı) — eğitim verisi buradan üretilecek"
        )
        return written

    # ─── Arka Plan ──────────────────────────────────────────────────────

    def start_background(self, *, rounds: int = 6, max_minutes: float = 6.0,
                         interval: float = 5.0) -> bool:
        """
        Eğitimi **arka planda** başlatır (istek bloklanmasın).

        Turlar arasında `interval` saniye beklenir; bu, GPU'nun soğumasına ve
        kaynak kapısının yeniden açılmasına fırsat verir.
        """
        if self._worker is not None and self._worker.is_alive():
            return False
        self._stop.clear()

        def loop() -> None:
            """Eğitim döngüsü."""
            try:
                self.run_rounds(rounds=rounds, max_minutes=max_minutes)
            except Exception as exc:
                logger.error(f"Arka plan eğitimi çöktü: {exc}", exc_info=True)
                with self._lock:
                    self._progress.error = f"{type(exc).__name__}: {exc}"
                    self._progress.finished = True
                self._save_progress("hata")
            finally:
                time.sleep(interval)

        self._worker = threading.Thread(target=loop, name="NeuroCoreCurriculum",
                                        daemon=True)
        self._worker.start()
        logger.info(f"Müfredat eğitimi arka planda başladı ({rounds} tur)")
        return True

    @property
    def is_running(self) -> bool:
        """Eğitim sürüyor mu?"""
        return bool(self._worker and self._worker.is_alive())


__all__ = [
    "CurriculumTrainer", "PipelineProgress",
    "PROGRESS_FILE", "DEFAULT_VOCAB_SIZE",
]
