"""
Eğitim ve Düşünme İş Akışı Testleri
=====================================
`neurocore.neural.training_work` ve `neurocore.agent.thinking_work` —
GPU işlerini ve düşünmeyi parçalı motora bağlar.

Neden bu testler önemli?
    İki iş akışı da **kaynak denetimi** ve **aşama aktarımı** üzerine kurulu.
    Bir hata şu sonuçları doğurur:

        * Eğitim turları **yanlış sırada** çalışır → bozuk ağırlık
        * VRAM yetmezse tur **sessizce atlanır** → kayıp ilerleme
        * Düşünme aşamaları arasında **aday kaybolur** → sessiz yanlış karar
        * Hız ölçümü yanlış olur → yanlış süre tahmini

    Sahte eğitim/sıralayıcı kullanılır; GPU ve model gerekmez.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from typing import Any, Dict, List, Sequence

from tests.helpers import assert_json_safe

from neurocore.agent.thinking import ThinkingLoop
from neurocore.agent.thinking_work import (
    STAGE_ESTIMATES,
    STAGE_RESOURCES,
    ThinkingTrace,
    ThinkingWorkflow,
)
from neurocore.core.governor import ResourceSnapshot
from neurocore.core.work import DONE, BLOCKED
from neurocore.neural.training_work import (
    DEFAULT_STEPS_PER_MINUTE,
    TRAIN_VRAM_MB,
    TrainingRound,
    TrainingWorkflow,
)


# ─── Eğitim İş Akışı ────────────────────────────────────────────────────────

class TestTrainingWorkflowPlanning(unittest.TestCase):
    """Eğitim turlarının planlanması."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_twp_")
        self.snap = ResourceSnapshot(cpu_percent=5, ram_percent=20,
                                     vram_percent=10, ram_total_mb=32768,
                                     vram_total_mb=11264, cpu_cores=24)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _akis(self) -> TrainingWorkflow:
        akis = TrainingWorkflow(workspace=self.root)
        akis.initialize(train_fn=lambda **_: {}, model_steps_fn=lambda: 0)
        return akis

    def test_rounds_chained(self) -> None:
        """
        Turlar **zincirlenmeli** (sıralı çalışmalı).

        Neden? Aynı model üzerinde paralel eğitim bozuk ağırlık üretir.
        Zincir olmazsa motor turları eşzamanlı başlatır.
        """
        akis = self._akis()
        adimlar = akis.plan_rounds(rounds=4, steps_per_round=100)
        self.assertEqual(adimlar[0].depends_on, [])
        for onceki, sonraki in zip(adimlar, adimlar[1:]):
            self.assertEqual(sonraki.depends_on, [onceki.id])

    def test_vram_declared(self) -> None:
        """Her tur VRAM ihtiyacını **bildirmeli** (kuyruk kararı için)."""
        akis = self._akis()
        adimlar = akis.plan_rounds(rounds=2, steps_per_round=100)
        for adim in adimlar:
            self.assertAlmostEqual(adim.needs["vram_mb"], TRAIN_VRAM_MB)

    def test_estimate_uses_calibrated_speed(self) -> None:
        """Süre tahmini ölçülen hızdan gelmeli."""
        akis = self._akis()
        akis._steps_per_minute = 2.0
        self.assertAlmostEqual(akis.estimate_seconds(120), 3600.0, places=1)

    def test_default_speed_matches_measurement(self) -> None:
        """Varsayılan hız ölçülen değere yakın olmalı (2,28 adım/dk)."""
        self.assertAlmostEqual(DEFAULT_STEPS_PER_MINUTE, 2.28, places=2)

    def test_plan_creates_chunks(self) -> None:
        akis = self._akis()
        akis.plan_rounds(rounds=5, steps_per_round=100)
        self.assertEqual(len(akis.engine.chunks), 5)


class TestTrainingWorkflowExecution(unittest.TestCase):
    """Eğitim turlarının yürütülmesi ve hız ölçümü."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_twe_")
        self.buyuk = ResourceSnapshot(cpu_percent=5, ram_percent=20,
                                      vram_percent=10, ram_total_mb=32768,
                                      vram_total_mb=11264, cpu_cores=24)
        self.kucuk = ResourceSnapshot(cpu_percent=10, ram_percent=10,
                                      vram_percent=10, ram_total_mb=4096,
                                      vram_total_mb=4096, cpu_cores=4)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _akis(self, *, gecikme: float = 0.02,
              adim: int = 50) -> TrainingWorkflow:
        akis = TrainingWorkflow(workspace=self.root)
        durum = {"adim": 0}

        def egitim(*, steps: int = 100, max_minutes: float = 18.0,
                   user_initiated: bool = True) -> Dict[str, Any]:
            time.sleep(gecikme)
            yapilan = min(steps, adim)
            durum["adim"] += yapilan
            return {
                "steps_completed": yapilan, "steps_requested": steps,
                "initial_loss": 0.10, "final_loss": 0.08,
                "tokens_seen": yapilan * 1200, "stop_reason": "completed",
            }

        akis.initialize(train_fn=egitim, model_steps_fn=lambda: durum["adim"])
        return akis

    def test_rounds_execute_in_order(self) -> None:
        akis = self._akis()
        akis.plan_rounds(rounds=3, steps_per_round=50)
        rapor = akis.run(snapshot=self.buyuk, max_wait_s=2.0, poll_s=0.5)
        self.assertEqual(rapor["steps_done"], 3)
        self.assertEqual(rapor["steps_failed"], 0)
        self.assertEqual(len(akis.rounds()), 3)

    def test_vram_too_small_defers_rounds(self) -> None:
        """VRAM yetmezse turlar **çalıştırılmamalı**, kuyruğa alınmalı."""
        akis = self._akis()
        akis.plan_rounds(rounds=3, steps_per_round=50)
        rapor = akis.run(snapshot=self.kucuk, max_wait_s=2.0, poll_s=0.5)
        self.assertEqual(rapor["steps_done"], 0)
        self.assertEqual(len(akis.rounds()), 0)
        self.assertGreater(rapor["chunks_blocked"], 0)
        self.assertIn(BLOCKED, {c.status for c in akis.engine.chunks})

    def test_speed_measured(self) -> None:
        """Hız ölçülmeli ve makul olmalı."""
        akis = self._akis(gecikme=0.5, adim=50)
        akis.plan_rounds(rounds=2, steps_per_round=50)
        akis.run(snapshot=self.buyuk, max_wait_s=2.0, poll_s=0.5)
        hiz = akis.speed_report()
        self.assertEqual(hiz["rounds_done"], 2)
        self.assertEqual(hiz["steps_total"], 100)
        self.assertGreater(hiz["steps_per_minute"], 0.0)
        self.assertGreater(hiz["seconds_per_step"], 0.0)

    def test_calibration_updates(self) -> None:
        """
        Ölçülen hız tahmini **güncellemeli**.

        Sahte eğitim gerçek hızdan çok daha hızlı olduğu için kalibre edilen
        değer başlangıçtan **yukarı** çıkmalı.
        """
        akis = self._akis(gecikme=0.01, adim=50)
        baslangic = akis._steps_per_minute
        akis.plan_rounds(rounds=2, steps_per_round=50)
        akis.run(snapshot=self.buyuk, max_wait_s=2.0, poll_s=0.5)
        self.assertGreater(akis._steps_per_minute, baslangic)

    def test_round_dict_json_safe(self) -> None:
        tur = TrainingRound(round_index=0, steps_done=10, duration_s=60.0,
                            loss_first=0.1, loss_last=0.05)
        assert_json_safe(self, tur.to_dict(), "training_round")
        self.assertAlmostEqual(tur.steps_per_minute, 10.0, places=3)

    def test_fresh_rounds_keep_speed(self) -> None:
        """
        `run_fresh_rounds` hızı **korumalı** (her tur ayrı süreç varsayımı).

        Ölçülen sorun: aynı süreçte turlar tekrarlandığında eğitim VRAM'i
        bırakmıyor ve hız 13,92 → 0,87 adım/dk'ya düşüyordu (13×). Taze
        süreç yönteminde hız sabit kalır: 10,15 / 10,17 / 10,18.
        """
        akis = TrainingWorkflow(workspace=self.root)
        akis.initialize(train_fn=lambda **_: {}, model_steps_fn=lambda: 0)
        sayac = {"n": 0}

        def tur() -> Dict[str, Any]:
            sayac["n"] += 1
            time.sleep(0.05)
            return {"steps_completed": 50, "steps_requested": 50,
                    "initial_loss": 0.1, "final_loss": 0.08,
                    "tokens_seen": 60000, "stop_reason": "completed",
                    "model_steps": sayac["n"] * 50}

        rapor = akis.run_fresh_rounds(rounds=3, round_runner=tur)
        self.assertEqual(sayac["n"], 3)
        self.assertEqual(rapor["rounds"], 3)
        self.assertEqual(len(akis.rounds()), 3)
        # Tüm turlar adım attı ve hız ölçüldü
        hiz = rapor["speed"]
        self.assertEqual(hiz["steps_total"], 150)
        self.assertGreater(hiz["steps_per_minute"], 0.0)

    def test_fresh_rounds_survive_round_crash(self) -> None:
        """Bir tur çökse kalanlar yine çalışmalı."""
        akis = TrainingWorkflow(workspace=self.root)
        akis.initialize(train_fn=lambda **_: {}, model_steps_fn=lambda: 0)
        sayac = {"n": 0}

        def tur() -> Dict[str, Any]:
            sayac["n"] += 1
            if sayac["n"] == 2:
                raise RuntimeError("kasıtlı tur çökmesi")
            return {"steps_completed": 20, "steps_requested": 20}

        rapor = akis.run_fresh_rounds(rounds=3, round_runner=tur)
        self.assertEqual(sayac["n"], 3, "çöken tur döngüyü bitirmemeli")
        basarili = [t for t in akis.rounds() if t.steps_done > 0]
        self.assertEqual(len(basarili), 2)

    def test_fresh_rounds_respect_timeout(self) -> None:
        """Süre sınırı aşılırsa kalan turlar atlanmalı."""
        akis = TrainingWorkflow(workspace=self.root)
        akis.initialize(train_fn=lambda **_: {}, model_steps_fn=lambda: 0)

        def tur() -> Dict[str, Any]:
            time.sleep(0.1)
            return {"steps_completed": 10, "steps_requested": 10}

        rapor = akis.run_fresh_rounds(rounds=10, round_runner=tur,
                                      stop_timeout_s=0.25)
        self.assertLess(rapor["rounds"], 10)
        self.assertGreaterEqual(rapor["rounds"], 1)

    def test_report_json_safe(self) -> None:
        akis = self._akis()
        akis.plan_rounds(rounds=2, steps_per_round=50)
        rapor = akis.run(snapshot=self.buyuk, max_wait_s=2.0, poll_s=0.5)
        assert_json_safe(self, rapor, "training_report")

    def test_stats_json_safe(self) -> None:
        akis = self._akis()
        assert_json_safe(self, akis.get_stats(), "training_stats")

    def test_zero_progress_is_failure(self) -> None:
        """Hiç adım atamayan tur **başarısız** sayılmalı (sessiz geçilmemeli)."""
        akis = TrainingWorkflow(workspace=self.root)
        akis.initialize(
            train_fn=lambda **_: {"steps_completed": 0, "steps_requested": 50,
                                  "stop_reason": "kaynaklar_dolu"},
            model_steps_fn=lambda: 0)
        akis.plan_rounds(rounds=1, steps_per_round=50)
        rapor = akis.run(snapshot=self.buyuk, max_wait_s=2.0, poll_s=0.5)
        self.assertEqual(rapor["steps_failed"], 1)
        self.assertIn("ilerlemedi", akis.engine.steps[0].error)

    def test_user_initiated_detected_from_signature(self) -> None:
        """
        `user_initiated` desteği **imzadan** anlaşılmalı.

        Neden imza, `try/except TypeError` değil? Eğitimin kendi içindeki bir
        `TypeError` de aynı istisnayı fırlatır; o zaman işlev yanlış imzayla
        yeniden çağrılır ve gerçek hata maskelenir.
        """
        akis = TrainingWorkflow(workspace=self.root)
        akis.initialize(
            train_fn=lambda *, steps=0, max_minutes=0.0, user_initiated=False: {
                "steps_completed": steps, "steps_requested": steps},
            model_steps_fn=lambda: 10)
        self.assertTrue(akis._detect_user_initiated())

        eski = TrainingWorkflow(workspace=self.root)
        eski.initialize(
            train_fn=lambda *, steps=0, max_minutes=0.0: {
                "steps_completed": steps, "steps_requested": steps},
            model_steps_fn=lambda: 10)
        self.assertFalse(eski._detect_user_initiated())

    def test_old_signature_called_without_flag(self) -> None:
        """`user_initiated` almayan işlev bayraksız çağrılmalı."""
        akis = TrainingWorkflow(workspace=self.root)
        cagri: List[Dict[str, Any]] = []

        def eski(*, steps: int = 0, max_minutes: float = 0.0) -> Dict[str, Any]:
            cagri.append({"steps": steps, "max_minutes": max_minutes})
            return {"steps_completed": steps, "steps_requested": steps}

        akis.initialize(train_fn=eski, model_steps_fn=lambda: 10)
        akis.plan_rounds(rounds=1, steps_per_round=10)
        akis.run(snapshot=self.buyuk, max_wait_s=2.0, poll_s=0.5)
        self.assertEqual(len(cagri), 1, "işlev gereksiz tekrar çağrılmamalı")
        self.assertNotIn("user_initiated", cagri[0])


# ─── Düşünme İş Akışı ───────────────────────────────────────────────────────

class FakeRanker:
    """
    Sahte yerel sıralayıcı.

    ⚠ `**kwargs` **şart**: gerçek `LocalRankController.rank_candidates`
    fazladan anahtar argümanları kabul eder ve `ThinkingLoop` bunları
    geçirir. Kabul etmeyen bir sahte, `TypeError` fırlatır; döngü bunu
    yakalayıp **sessizce buluta düşer** ve test "yerel seçim çalışmıyor"
    diye yanlış sonuç verir (ölçüldü: 20 dakika kaybettirdi).
    """

    def __init__(self, *, best: int = 1, uncertain: bool = False) -> None:
        self.best = best
        self.uncertain = uncertain
        self.calls = 0

    def rank_candidates(self, prompt: str, candidates: Sequence[str], *,
                        min_margin: float = 0.0, top_k: int = 0,
                        **kwargs: Any) -> Dict[str, Any]:
        self.calls += 1
        return {
            "ok": True, "best_index": self.best,
            "best_text": str(candidates[self.best]),
            "margin": 0.01 if self.uncertain else 1.4,
            "uncertain": self.uncertain,
            "confidence": "belirsiz" if self.uncertain else "yüksek",
            "candidates": len(candidates),
            "ranking": [{"index": i, "score": -i, "margin": -i,
                         "preview": str(candidates[i])[:50]}
                        for i in range(len(candidates))],
        }


class TestThinkingWorkflow(unittest.TestCase):
    """Üç aşamalı düşünme."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_thw_")
        self.snap = ResourceSnapshot(cpu_percent=10, ram_percent=20,
                                     vram_percent=15, ram_total_mb=32768,
                                     vram_total_mb=11264, cpu_cores=24)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _akis(self, *, uncertain: bool = False, thinking: bool = True,
              aday: int = 3, user_initiated: bool = False) -> ThinkingWorkflow:
        akis = ThinkingWorkflow(workspace=self.root)
        tl = None
        if thinking:
            tl = ThinkingLoop(min_margin=0.15)
            tl.initialize(ranker=FakeRanker(uncertain=uncertain),
                          data_path=self.root)
        akis.initialize(
            thinking=tl,
            propose=lambda problem, n: [f"{problem[:12]}-yol{i}" for i in range(n)],
            candidates=aday)
        return akis

    def test_three_stages_per_problem(self) -> None:
        """Her problem **üç** adıma bölünmeli."""
        akis = self._akis()
        adimlar = akis.plan_problems(["P1", "P2"])
        self.assertEqual(len(adimlar), 6)
        turler = [a.kind for a in adimlar]
        for tur in ("think_uret", "think_sec", "think_dogrula"):
            self.assertEqual(turler.count(tur), 2)

    def test_stages_chained_within_problem(self) -> None:
        """Aşamalar kendi içinde sıralı olmalı."""
        akis = self._akis()
        adimlar = akis.plan_problems(["P1"])
        self.assertEqual(adimlar[0].depends_on, [])
        self.assertEqual(adimlar[1].depends_on, [adimlar[0].id])
        self.assertEqual(adimlar[2].depends_on, [adimlar[1].id])

    def test_problems_independent(self) -> None:
        """Farklı problemler **bağımsız** olmalı (paralel düşünülebilsin)."""
        akis = self._akis()
        adimlar = akis.plan_problems(["P1", "P2"])
        ilk_asamalar = [a for a in adimlar if a.kind == "think_uret"]
        for adim in ilk_asamalar:
            self.assertEqual(adim.depends_on, [])

    def test_stage_resources_differ(self) -> None:
        """
        Aşamaların kaynak ihtiyacı **farklı** olmalı.

        Sıralama yerel modeli çalıştırır (GPU); üretim ve doğrulama ağ
        bekler. Aynı sayılsalardı motor yanlış parçayı kuyruğa alırdı.
        """
        self.assertGreater(STAGE_RESOURCES["sec"].get("vram_mb", 0), 0)
        self.assertEqual(STAGE_RESOURCES["uret"].get("vram_mb", 0), 0)
        self.assertGreater(STAGE_RESOURCES["sec"]["cpu"],
                           STAGE_RESOURCES["uret"]["cpu"])

    def test_local_selection_used(self) -> None:
        """Yerel seçim belirginse **kullanılmalı**."""
        akis = self._akis(uncertain=False)
        akis.plan_problems(["P1"])
        akis.run(snapshot=self.snap, max_wait_s=2.0, poll_s=0.5)
        izler = akis.traces()
        self.assertEqual(len(izler), 1)
        self.assertEqual(izler[0].source, "local")
        self.assertEqual(izler[0].confidence, "yüksek")
        self.assertIn("yol1", izler[0].chosen)

    def test_uncertain_falls_back_to_cloud(self) -> None:
        """Belirsizse bulut tercihi geçerli olmalı."""
        akis = self._akis(uncertain=True)
        akis.plan_problems(["P1"])
        akis.run(snapshot=self.snap, max_wait_s=2.0, poll_s=0.5)
        iz = akis.traces()[0]
        self.assertEqual(iz.source, "cloud")
        self.assertIn("yol0", iz.chosen)
        self.assertGreaterEqual(
            akis.stage_report()["counter"]["cloud_fallbacks"], 1)

    def test_no_thinking_loop_is_explicit(self) -> None:
        """Yerel model yoksa **sessizce** yanlış yapmamalı."""
        akis = self._akis(thinking=False)
        akis.plan_problems(["P1"])
        akis.run(snapshot=self.snap, max_wait_s=2.0, poll_s=0.5)
        iz = akis.traces()[0]
        self.assertEqual(iz.source, "cloud")
        self.assertIn("seçim atlandı", akis.engine.steps[1].result)

    def test_stage_times_recorded(self) -> None:
        """Her aşamanın süresi kaydedilmeli (nerede yavaş?)."""
        akis = self._akis()
        akis.plan_problems(["P1"])
        akis.run(snapshot=self.snap, max_wait_s=2.0, poll_s=0.5)
        iz = akis.traces()[0]
        for asama in ("uret", "sec", "dogrula"):
            self.assertIn(asama, iz.stage_times)
        self.assertGreaterEqual(iz.total_s, 0.0)

    def test_stage_report_shape(self) -> None:
        akis = self._akis()
        akis.plan_problems(["P1", "P2"])
        akis.run(snapshot=self.snap, max_wait_s=2.0, poll_s=0.5)
        rapor = akis.stage_report()
        for asama in ("uret", "sec", "dogrula"):
            self.assertIn(asama, rapor["stages"])
            self.assertEqual(rapor["stages"][asama]["calls"], 2)
        self.assertIn("local_share", rapor)
        assert_json_safe(self, rapor, "stage_report")

    def test_produce_failure_recorded(self) -> None:
        """Aday üretimi çökerse adım başarısız olmalı, motor ayakta kalmalı."""
        akis = ThinkingWorkflow(workspace=self.root)
        akis.initialize(thinking=None,
                        propose=lambda p, n: (_ for _ in ()).throw(
                            RuntimeError("bulut erişilemedi")))
        akis.plan_problems(["P1"])
        rapor = akis.run(snapshot=self.snap, max_wait_s=2.0, poll_s=0.5)
        self.assertEqual(rapor["steps_failed"], 1)
        self.assertIn("bulut", akis.engine.steps[0].error)

    def test_missing_propose_fails_cleanly(self) -> None:
        akis = ThinkingWorkflow(workspace=self.root)
        akis.initialize(thinking=None)
        akis.plan_problems(["P1"])
        rapor = akis.run(snapshot=self.snap, max_wait_s=2.0, poll_s=0.5)
        self.assertEqual(rapor["steps_failed"], 1)

    def test_trace_dict_json_safe(self) -> None:
        iz = ThinkingTrace(problem="p", candidates=["a", "b"], chosen="a",
                           source="local", confidence="yüksek", margin=1.2,
                           stage_times={"uret": 1.0, "sec": 0.5})
        assert_json_safe(self, iz.to_dict(), "thinking_trace")
        self.assertAlmostEqual(iz.total_s, 1.5, places=3)

    def test_stats_json_safe(self) -> None:
        akis = self._akis()
        assert_json_safe(self, akis.get_stats(), "thinking_work_stats")

    def test_estimates_positive(self) -> None:
        for asama, sure in STAGE_ESTIMATES.items():
            self.assertGreater(sure, 0.0, f"{asama} tahmini sıfır")

    def test_candidate_count_respected(self) -> None:
        akis = self._akis(aday=5)
        akis.plan_problems(["P1"])
        akis.run(snapshot=self.snap, max_wait_s=2.0, poll_s=0.5)
        self.assertEqual(len(akis.traces()[0].candidates), 5)

    def test_lifecycle(self) -> None:
        akis = self._akis()
        self.assertTrue(akis.get_stats()["initialized"])
        akis.shutdown()
        self.assertFalse(akis.get_stats()["initialized"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
