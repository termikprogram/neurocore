"""
Eğitim İş Akışı — GPU İşlerini Parçalı Motora Bağlar
=====================================================
Eğitim turlarını `WorkStep`'e çevirir: her tur bir **parça** olur, kaynağa
göre sıralanır, kesintiden sonra kaldığı yerden devam eder.

Neden gerekli?
    Eğitim döngüsü elle yazılmıştı: `for tur in range(...)` + `engine.train`.
    Bu yaklaşımın ölçülen sorunları:

        1. **Kesinti = kayıp** — süreç ölürse tur başından başlar
        2. **Kaynak görünmez** — VRAM yetmezse tur atlanır, neden kaydedilmez
        3. **İlerleme dağınık** — hangi turun bittiği ayrı dosyada tutulur
        4. **Kalibrasyon yok** — adım/dakika hızı ölçülmez

    Parçalı motor bu dördünü de çözer:

        * Her tur bir `WorkStep` → günlükte `bitti`/`başarısız`/`kaynak bekliyor`
        * VRAM ihtiyacı `needs` alanında → motor kuyruğa alır
        * `resume()` → kesintiden sonra yalnızca kalan turlar çalışır
        * `estimate_s` → ölçülen hızla **kendiliğinden kalibre edilir**

Ölçülen hız (kalibrasyon girdisi):

    41 adım / 18 dakika = **2,28 adım/dakika**  (208M model, GPU %99)

    Yani tur başına tahmin: `adim / 2,28 * 60` saniye.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from neurocore.core.governor import ResourceSnapshot
from neurocore.core.work import WorkEngine, WorkStep
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.training_work")

#: Ölçülen başlangıç hızı (adım/dakika) — 208M model, 1080 Ti, AMP açık.
#: `TrainingWorkflow` bu değeri gerçek ölçümlerle günceller.
DEFAULT_STEPS_PER_MINUTE = 2.28

#: Eğitim adımının VRAM ihtiyacı (MB). 208M model + AdamW + AMP.
#:
#: Ölçüldü: eğitim sırasında `nvidia-smi` ~10,9 GB gösteriyor; bunun
#: ~10,2 GB'ı eğitime aittir (ağırlıklar + optimizer + etkinleştirmeler).
TRAIN_VRAM_MB = 10_200.0

#: Eğitim adımının CPU/RAM ihtiyacı
TRAIN_CPU_PERCENT = 25.0
TRAIN_RAM_MB = 2_500.0


@dataclass
class TrainingRound:
    """Tek eğitim turunun sonucu."""
    round_index: int
    steps_requested: int = 0
    steps_done: int = 0
    loss_first: float = 0.0
    loss_last: float = 0.0
    tokens: int = 0
    duration_s: float = 0.0
    stop_reason: str = ""
    model_steps: int = 0

    @property
    def steps_per_minute(self) -> float:
        """Bu turun ölçülen hızı."""
        if self.duration_s <= 0 or self.steps_done <= 0:
            return 0.0
        return self.steps_done / (self.duration_s / 60.0)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "round": self.round_index, "steps_done": self.steps_done,
            "steps_requested": self.steps_requested,
            "loss_first": round(self.loss_first, 6),
            "loss_last": round(self.loss_last, 6),
            "tokens": self.tokens, "duration_s": round(self.duration_s, 1),
            "steps_per_minute": round(self.steps_per_minute, 3),
            "stop_reason": self.stop_reason, "model_steps": self.model_steps,
        }


class TrainingWorkflow:
    """
    Eğitimi parçalı motora bağlar.

    Kullanım:
        akis = TrainingWorkflow(workspace=".neurocore")
        akis.initialize(train_fn=engine.train, model_steps_fn=lambda: engine.model.growth.total_training_steps)
        akis.plan_rounds(rounds=12, steps_per_round=150, dataset=ds)
        rapor = akis.run()
        print(akis.speed_report())
    """

    name = "neural.training_work"

    def __init__(self, *, workspace: str = "",
                 chunk_ratio: float = 1.0) -> None:
        #: Parça payı 1,0 → tek eğitim parçası tüm bütçeyi kullanabilir.
        #: Eğitim **bölünemez** (aynı model, sıralı adımlar); paralellik
        #: burada anlamsızdır, kaynak denetimi anlamlıdır.
        self.engine = WorkEngine(workspace=workspace, chunk_ratio=chunk_ratio,
                                 chunk_steps=1)
        self._train_fn: Optional[Callable[..., Any]] = None
        self._model_steps_fn: Optional[Callable[[], int]] = None
        self._rounds: List[TrainingRound] = []
        self._steps_per_minute = DEFAULT_STEPS_PER_MINUTE
        self._initialized = False
        #: Eğitim işlevi `user_initiated` alıyor mu? (None = henüz bilinmiyor)
        self._supports_user_initiated: Optional[bool] = None

    def _detect_user_initiated(self) -> bool:
        """
        Eğitim işlevinin `user_initiated` parametresini kabul edip
        etmediğini **imzadan** anlar.

        Neden imza? `TypeError` yakalamak güvenilmez: eğitimin kendi
        içindeki bir `TypeError` de aynı hatayı fırlatır ve yanlış imzayla
        yeniden denenir (ölçüldü: gereksiz ikinci çağrı).
        """
        import inspect

        try:
            imza = inspect.signature(self._train_fn)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        if "user_initiated" in imza.parameters:
            return True
        # `**kwargs` alan işlevler de kabul eder
        return any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in imza.parameters.values())

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, *, train_fn: Callable[..., Any],
                   model_steps_fn: Callable[[], int],
                   dataset: Any = None) -> None:
        """
        İş akışını başlatır.

        Args:
            train_fn: Eğitim çağrısı (`engine.train` benzeri)
            model_steps_fn: Modelin toplam adımını döndüren çağrılabilir
            dataset: Veri kümesi (None = eğitimci kendi kurar)
        """
        self._train_fn = train_fn
        self._model_steps_fn = model_steps_fn
        self._dataset = dataset
        self.engine.register("train", self._run_training_step)
        self._initialized = True
        logger.info(
            f"Eğitim iş akışı hazır — başlangıç hızı "
            f"{self._steps_per_minute:.2f} adım/dk"
        )

    def shutdown(self) -> None:
        """Kaynakları bırakır."""
        self._initialized = False

    # ─── Planlama ───────────────────────────────────────────────────────

    def plan_rounds(self, *, rounds: int, steps_per_round: int,
                    max_minutes: float = 18.0,
                    objective: str = "") -> List[WorkStep]:
        """
        Eğitim turlarını planlar.

        Her tur **bağımsız** adımdır ama gerçekte sıralıdır (aynı model).
        Bu yüzden `depends_on` zinciri kurulur: tur N, tur N-1'e bağlıdır.
        Zincir olmadan motor turları paralel çalıştırmaya çalışır ve aynı
        modele eşzamanlı yazma **bozuk ağırlık** üretir.

        Args:
            rounds: Tur sayısı
            steps_per_round: Tur başına adım
            max_minutes: Tur başına süre sınırı
            objective: Amaç metni

        Returns:
            `WorkStep` listesi
        """
        adimlar: List[WorkStep] = []
        onceki: Optional[str] = None
        for index in range(max(1, rounds)):
            kimlik = f"tur{index:03d}"
            adimlar.append(WorkStep(
                id=kimlik,
                title=f"Eğitim turu {index + 1}/{rounds} ({steps_per_round} adım)",
                kind="train",
                needs={"cpu": TRAIN_CPU_PERCENT, "ram_mb": TRAIN_RAM_MB,
                       "vram_mb": TRAIN_VRAM_MB},
                depends_on=[onceki] if onceki else [],
                payload={"steps": steps_per_round,
                         "max_minutes": max_minutes,
                         "round_index": index},
                estimate_s=self.estimate_seconds(steps_per_round),
            ))
            onceki = kimlik
        amaç = objective or f"{rounds} turluk eğitim ({steps_per_round} adım/tur)"
        self.engine.plan(amaç, adimlar)
        logger.info(
            f"Eğitim planı: {len(adimlar)} tur, tur başına "
            f"{steps_per_round} adım, tahmini {self.estimate_seconds(steps_per_round):.0f} sn"
        )
        return adimlar

    def estimate_seconds(self, steps: int) -> float:
        """Ölçülen hıza göre süre tahmini (kalibre edilir)."""
        if self._steps_per_minute <= 0:
            return float(steps) * 26.0
        return float(steps) / self._steps_per_minute * 60.0

    # ─── Yürütme ────────────────────────────────────────────────────────

    def _run_training_step(self, step: WorkStep) -> str:
        """
        Bir eğitim turunu çalıştırır.

        `engine.train` çağrılır; sonuç `TrainingRound` olarak kaydedilir ve
        **ölçülen hız** kalibrasyon için güncellenir.
        """
        if self._train_fn is None:
            raise RuntimeError("eğitim işlevi bağlı değil")
        adim = int(step.payload.get("steps") or 100)
        sinir = float(step.payload.get("max_minutes") or 18.0)
        tur = int(step.payload.get("round_index") or 0)

        basladi = time.time()
        # `user_initiated` destekleniyor mu? **İmza bir kez denetlenir**;
        # her turda `TypeError` yakalamak, eğitimin içindeki gerçek bir
        # `TypeError`'ı yanlış yorumlamaya yol açardı.
        if self._supports_user_initiated is None:
            self._supports_user_initiated = self._detect_user_initiated()
        try:
            if self._supports_user_initiated:
                sonuc = self._train_fn(steps=adim, max_minutes=sinir,
                                       user_initiated=True)
            else:
                sonuc = self._train_fn(steps=adim, max_minutes=sinir)
        except Exception as exc:
            logger.warning(f"Tur {tur + 1} eğitimi çöktü: {exc}")
            raise
        sure = time.time() - basladi

        if not isinstance(sonuc, dict):
            sonuc = {}
        kayit = TrainingRound(
            round_index=tur,
            steps_requested=int(sonuc.get("steps_requested") or adim),
            steps_done=int(sonuc.get("steps_completed") or 0),
            loss_first=float(sonuc.get("initial_loss") or 0.0),
            loss_last=float(sonuc.get("final_loss") or 0.0),
            tokens=int(sonuc.get("tokens_seen") or 0),
            duration_s=sure,
            stop_reason=str(sonuc.get("stop_reason") or ""),
            model_steps=int(self._model_steps_fn() if self._model_steps_fn
                            else 0),
        )
        self._rounds.append(kayit)

        # ── Hız kalibrasyonu ──
        #
        # Ölçülen hızı **üstel hareketli ortalama** ile güncelle: tek bir
        # yavaş tur (ör. kaynak baskısı) tahmini bozmasın.
        if kayit.steps_per_minute > 0:
            self._steps_per_minute = (0.7 * self._steps_per_minute
                                      + 0.3 * kayit.steps_per_minute)
            logger.info(
                f"Tur {tur + 1}: {kayit.steps_done} adım, "
                f"{kayit.steps_per_minute:.2f} adım/dk "
                f"(kalibre: {self._steps_per_minute:.2f})"
            )
        if kayit.steps_done <= 0:
            raise RuntimeError(
                f"tur {tur + 1} ilerlemedi: {kayit.stop_reason or 'bilinmiyor'}")
        return (f"{kayit.steps_done} adım, kayıp "
                f"{kayit.loss_first:.4f}→{kayit.loss_last:.4f}, "
                f"{kayit.steps_per_minute:.2f} adım/dk")

    def run_fresh_rounds(self, *, rounds: int,
                         round_runner: Callable[[], Dict[str, Any]],
                         objective: str = "",
                         stop_timeout_s: float = 7200.0) -> Dict[str, Any]:
        """
        Her turu **ayrı bir süreçte** çalıştırır.

        ⚠ Neden gerekli? Ölçüldü: eğitim VRAM'i bırakmıyor. Aynı süreçte
        turlar tekrarlandığında her tur ~10,2 GB ile başlıyor, gözcü kaynak
        dolu görüyor ve toplu işi küçültüyor:

            tur 1 (temiz)        : 13,92 adım/dk
            tur 2 (VRAM dolu)    :  2,24 adım/dk
            tur 3                 : 10,26 adım/dk  (kısmi toparlanma)
            tur 4+                :  1,05 → 0,87 adım/dk

        Yani aynı süreçte devam etmek **13× yavaşlatıyor**. Her turu taze bir
        Python sürecinde çalıştırmak CUDA bağlamını sıfırlar ve tam hız verir.

        Bu metot süreç yönetimini **çağırana** bırakır (`round_runner`), çünkü
        bir kütüphane kendi kendini yeniden başlatamaz. Tipik kullanım:

            alt_süreç = subprocess.run([sys.executable, "tools/tur.py"])
            akis.run_fresh_rounds(rounds=12, round_runner=lambda: {...})

        Args:
            rounds: Kaç tur çalıştırılacak
            round_runner: Bir turu çalıştırıp sonuç sözlüğü döndüren çağrı
            objective: Amaç metni
            stop_timeout_s: Toplam süre sınırı

        Returns:
            Özet rapor (turlar ve hız)
        """
        basladi = time.time()
        for index in range(max(1, rounds)):
            if time.time() - basladi > stop_timeout_s:
                logger.warning("Süre sınırı aşıldı — kalan turlar atlandı")
                break
            t0 = time.time()
            try:
                sonuc = round_runner() or {}
            except Exception as exc:
                logger.error(f"Tur {index + 1} süreci çöktü: {exc}")
                sonuc = {}
            sure = time.time() - t0
            kayit = TrainingRound(
                round_index=index,
                steps_requested=int(sonuc.get("steps_requested") or 0),
                steps_done=int(sonuc.get("steps_completed") or 0),
                loss_first=float(sonuc.get("initial_loss") or 0.0),
                loss_last=float(sonuc.get("final_loss") or 0.0),
                tokens=int(sonuc.get("tokens_seen") or 0),
                duration_s=sure,
                stop_reason=str(sonuc.get("stop_reason") or ""),
                model_steps=int(sonuc.get("model_steps") or 0),
            )
            self._rounds.append(kayit)
            if kayit.steps_per_minute > 0:
                self._steps_per_minute = (0.7 * self._steps_per_minute
                                          + 0.3 * kayit.steps_per_minute)
            logger.info(
                f"Taze tur {index + 1}/{rounds}: {kayit.steps_done} adım, "
                f"{kayit.steps_per_minute:.2f} adım/dk"
            )
        return {"rounds": len(self._rounds),
                "speed": self.speed_report(),
                "duration_s": round(time.time() - basladi, 1)}

    def run(self, *, snapshot: Optional[ResourceSnapshot] = None,
            max_wait_s: float = 1800.0, poll_s: float = 30.0) -> Dict[str, Any]:
        """
        Planlanan turları çalıştırır.

        `run_until_done` kullanılır: VRAM yetmezse tur **bekler**, atlanmaz.
        Kaynak açılmazsa kalan turlar dürüstçe raporlanır.

        ⚠ `max_wait_s` **test edilebilir olmalı**: sınırsız bekleme, VRAM'i
        yetmeyen bir sistemde testi dakikalarca askıda bırakır (ölçüldü).
        Varsayılan 30 dakika üretim içindir; testler kısa değer verir.

        Args:
            snapshot: Kaynak profili (None = canlı ölç)
            max_wait_s: En fazla ne kadar beklenecek
            poll_s: İlerleme olmayınca bekleme aralığı

        Returns:
            Özet rapor (eğitim turları ve hız raporuyla)
        """
        rapor = self.engine.run_until_done(
            snapshot=snapshot, max_wait_s=max_wait_s, poll_s=poll_s)
        rapor["training"] = [t.to_dict() for t in self._rounds]
        rapor["speed"] = self.speed_report()
        return rapor

    # ─── Raporlama ──────────────────────────────────────────────────────

    def speed_report(self) -> Dict[str, Any]:
        """
        Ölçülen eğitim hızı.

        Kullanıcı "hızımızı görelim" dediğinde bakılacak yer burasıdır.
        """
        basarili = [t for t in self._rounds if t.steps_done > 0]
        toplam_adim = sum(t.steps_done for t in basarili)
        toplam_sure = sum(t.duration_s for t in basarili)
        return {
            "rounds_done": len(basarili),
            "rounds_total": len(self._rounds),
            "steps_total": toplam_adim,
            "duration_s": round(toplam_sure, 1),
            "steps_per_minute": round(
                toplam_adim / (toplam_sure / 60.0), 3) if toplam_sure else 0.0,
            "calibrated_steps_per_minute": round(self._steps_per_minute, 3),
            # 4 basamak: çok hızlı adımlarda (test/sahte eğitim) 2 basamak
            # 0,0'a yuvarlanıyordu ve ölçüm "sıfır" görünüyordu.
            "seconds_per_step": round(
                toplam_sure / toplam_adim, 4) if toplam_adim else 0.0,
            "tokens_total": sum(t.tokens for t in basarili),
            "loss_first": basarili[0].loss_first if basarili else None,
            "loss_last": basarili[-1].loss_last if basarili else None,
            "model_steps": basarili[-1].model_steps if basarili else 0,
        }

    def rounds(self) -> List[TrainingRound]:
        """Tamamlanan turlar."""
        return list(self._rounds)

    def get_stats(self) -> Dict[str, Any]:
        """Sayaçlar ve hız."""
        return {
            "initialized": self._initialized,
            "rounds": len(self._rounds),
            "speed": self.speed_report(),
            "engine": self.engine.get_stats(),
        }


__all__ = [
    "TrainingWorkflow", "TrainingRound",
    "DEFAULT_STEPS_PER_MINUTE", "TRAIN_VRAM_MB",
]
