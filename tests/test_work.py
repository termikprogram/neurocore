"""
Parçalı İş Motoru Testleri
===========================
`neurocore.core.governor` ve `neurocore.core.work` — kaynak farkında
parçalama, kuyruklama, paralel yürütme ve günlükleme.

Neden bu testler kritik?
    Motor, **bilgisayarı kilitlememek** üzerine kurulu. Bir hata
    (yanlış bütçe, sığmayan parçayı çalıştırma, sonsuz bekleme) kullanıcının
    makinesini kilitler. Bu yüzden:

        * bütçe matematiği ayrı ayrı doğrulanır,
        * sığmayan işin **çalıştırılmadığı** kanıtlanır,
        * küçük sistem senaryosu (4 GB) açıkça test edilir,
        * kesintiden sonra devam edebilme sınanır,
        * paralelliğin **gerçekten** kullanıldığı ölçülür.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

from tests.helpers import assert_json_safe

from neurocore.core.governor import (
    ChunkBudget,
    GovernorPolicy,
    ResourceGovernor,
    ResourceSnapshot,
)
from neurocore.core.work import (
    BLOCKED,
    DONE,
    FAILED,
    JOURNAL_FILE,
    WorkChunk,
    WorkEngine,
    WorkJournal,
    WorkStep,
)


# ─── Kaynak Yönetişimi ──────────────────────────────────────────────────────

class TestGovernorBudget(unittest.TestCase):
    """Bütçe matematiği (kullanıcı kuralı: boştakinin %90'ı)."""

    def setUp(self) -> None:
        self.gov = ResourceGovernor()

    def test_uses_ninety_percent_of_free(self) -> None:
        """%50 dolu → %45 kullanılabilir (boştakinin %90'ı)."""
        self.assertAlmostEqual(self.gov.spendable_percent(50.0), 45.0, places=3)

    def test_idle_system_gets_eighty_one(self) -> None:
        """%10 dolu → %81 (kullanıcının verdiği örnek)."""
        self.assertAlmostEqual(self.gov.spendable_percent(10.0), 81.0, places=3)

    def test_hard_ceiling_applies(self) -> None:
        """Boş sistemde bile sert tavan (%85) aşılmaz."""
        self.assertLessEqual(self.gov.spendable_percent(0.0), 85.0)

    def test_fully_busy_gives_nothing(self) -> None:
        """Sistem tamamen doluysa kaynak verilmez."""
        self.assertAlmostEqual(self.gov.spendable_percent(100.0), 0.0, places=3)

    def test_negative_usage_clamped(self) -> None:
        """Anlamsız ölçüm çökmeye yol açmamalı."""
        self.assertGreaterEqual(self.gov.spendable_percent(-10.0), 0.0)


class TestGovernorPlan(unittest.TestCase):
    """Parça planı — küçük ve büyük sistemlerde."""

    def setUp(self) -> None:
        self.gov = ResourceGovernor()

    def test_tiny_system_parallelism(self) -> None:
        """
        **Kullanıcının senaryosu**: 4 GB VRAM/RAM, %10 dolu.

        Beklenen: bütçe ~%81 → 3,2 GB; parça %5 → 200 MB; ~16 paralel.
        """
        butce = self.gov.plan(ResourceSnapshot.tiny())
        self.assertGreaterEqual(butce.parallel, 10)
        self.assertLessEqual(butce.parallel, 16)
        self.assertGreater(butce.ram_mb, 100)

    def test_busy_system_reduces_parallelism(self) -> None:
        """Sistem doldukça paralellik **azalmalı**."""
        bos = self.gov.plan(ResourceSnapshot(
            cpu_percent=5, ram_percent=5, vram_percent=5,
            ram_total_mb=32768, vram_total_mb=11264, cpu_cores=24))
        yogun = self.gov.plan(ResourceSnapshot(
            cpu_percent=80, ram_percent=80, vram_percent=80,
            ram_total_mb=32768, vram_total_mb=11264, cpu_cores=24))
        self.assertGreater(bos.parallel, yogun.parallel)

    def test_limiting_resource_reported(self) -> None:
        """Hangi kaynağın kısıtladığı bildirilmeli (tanı için)."""
        butce = self.gov.plan(ResourceSnapshot.tiny())
        self.assertIn(butce.limiting, ("cpu", "ram", "vram"))
        self.assertTrue(butce.reason)

    def test_parallel_capped(self) -> None:
        """Aşırı paralellik fayda getirmez; tavan uygulanır."""
        politika = GovernorPolicy(max_parallel=4)
        gov = ResourceGovernor(politika)
        butce = gov.plan(ResourceSnapshot(
            cpu_percent=1, ram_percent=1, vram_percent=1,
            ram_total_mb=65536, vram_total_mb=24576, cpu_cores=64))
        self.assertLessEqual(butce.parallel, 4)

    def test_unknown_system_degrades_safely(self) -> None:
        """Sistem bilgisi okunamazsa tek parça ile ilerlenir (çökme yok)."""
        butce = self.gov.plan(ResourceSnapshot())   # tüm alanlar 0
        self.assertGreaterEqual(butce.parallel, 1)
        self.assertTrue(butce.reason)

    def test_budget_dict_json_safe(self) -> None:
        assert_json_safe(self, self.gov.plan(ResourceSnapshot.tiny()).to_dict(),
                         "chunk_budget")

    def test_snapshot_dict_json_safe(self) -> None:
        assert_json_safe(self, ResourceSnapshot.tiny().to_dict(), "snapshot")

    def test_snapshot_live_does_not_crash(self) -> None:
        anlik = ResourceSnapshot.live()
        self.assertGreaterEqual(anlik.cpu_cores, 1)


class TestGovernorFits(unittest.TestCase):
    """`fits`: parça bütçeye sığıyor mu?"""

    def setUp(self) -> None:
        self.gov = ResourceGovernor()

    def test_small_need_fits(self) -> None:
        butce = self.gov.plan(ResourceSnapshot.tiny())
        self.assertTrue(self.gov.fits({"ram_mb": 10.0}, butce))

    def test_huge_need_does_not_fit(self) -> None:
        """Bütçeyi aşan ihtiyaç **kabul edilmemeli** (kilitlenme koruması)."""
        butce = self.gov.plan(ResourceSnapshot.tiny())
        self.assertFalse(self.gov.fits({"ram_mb": 100_000.0}, butce))

    def test_cpu_limit_respected(self) -> None:
        butce = self.gov.plan(ResourceSnapshot.tiny())
        self.assertFalse(self.gov.fits({"cpu": 999.0}, butce))


class TestChunkSplitting(unittest.TestCase):
    """IDM tarzı iş bölme."""

    def setUp(self) -> None:
        self.gov = ResourceGovernor()

    def test_splits_evenly(self) -> None:
        parcalar = self.gov.split_into_chunks(100, 25)
        self.assertEqual(parcalar, [(0, 25), (25, 50), (50, 75), (75, 100)])

    def test_last_chunk_shorter(self) -> None:
        parcalar = self.gov.split_into_chunks(100, 30)
        self.assertEqual(parcalar[-1], (90, 100))

    def test_covers_everything_exactly_once(self) -> None:
        """Parçalar tüm işi **tam bir kez** kapsamalı (boşluk/çakışma yok)."""
        parcalar = self.gov.split_into_chunks(8_000_000, 300_000)
        kapsanan = sum(b - a for a, b in parcalar)
        self.assertEqual(kapsanan, 8_000_000)
        # Ardışık ve örtüşmesiz
        for (_, onceki_son), (sonraki_bas, _) in zip(parcalar, parcalar[1:]):
            self.assertEqual(onceki_son, sonraki_bas)

    def test_zero_units(self) -> None:
        self.assertEqual(self.gov.split_into_chunks(0, 10), [])

    def test_invalid_chunk_size_guarded(self) -> None:
        """Parça boyu 0/negatifse en az 1 kabul edilir (sonsuz döngü yok)."""
        parcalar = self.gov.split_into_chunks(5, 0)
        self.assertEqual(len(parcalar), 5)


# ─── Günlük ─────────────────────────────────────────────────────────────────

class TestWorkJournal(unittest.TestCase):
    """Günlük — kesintiye dayanıklı olmalı."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="nc_jr_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_append_and_load(self) -> None:
        j = WorkJournal(self.root / JOURNAL_FILE)
        j.append({"olay": "a", "değer": 1})
        j.append({"olay": "b", "değer": 2})
        kayitlar = WorkJournal(self.root / JOURNAL_FILE).load()
        self.assertEqual(len(kayitlar), 2)
        self.assertEqual(kayitlar[1]["değer"], 2)

    def test_corrupt_line_skipped(self) -> None:
        """
        Yarım yazılmış satır **atlanmalı**, kalan okunmalı.

        Neden? Kesinti anında dosyanın son satırı yarım kalabilir; tüm
        günlüğü kaybetmek "nerede kaldım" bilgisini yok eder.
        """
        yol = self.root / JOURNAL_FILE
        yol.write_text(
            '{"olay": "a"}\n'
            '{"olay": "bozuk"\n'          # yarım satır
            '{"olay": "c"}\n',
            encoding="utf-8",
        )
        kayitlar = WorkJournal(yol).load()
        self.assertEqual(len(kayitlar), 2)
        self.assertEqual(kayitlar[-1]["olay"], "c")

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(WorkJournal(self.root / "yok.jsonl").load(), [])

    def test_memory_only_journal(self) -> None:
        """Yolsuz günlük de çalışmalı (test ve geçici kullanım)."""
        j = WorkJournal(None)
        j.append({"olay": "x"})
        self.assertEqual(len(j.records()), 1)

    def test_concurrent_appends_safe(self) -> None:
        """Paralel yazımda satırlar bozulmamalı."""
        j = WorkJournal(self.root / JOURNAL_FILE)

        def yaz(index: int) -> None:
            for alt in range(10):
                j.append({"olay": "t", "i": index, "alt": alt})

        isler = [threading.Thread(target=yaz, args=(i,)) for i in range(4)]
        for is_ in isler:
            is_.start()
        for is_ in isler:
            is_.join()
        kayitlar = WorkJournal(self.root / JOURNAL_FILE).load()
        self.assertEqual(len(kayitlar), 40)


# ─── İş Motoru ──────────────────────────────────────────────────────────────

class TestWorkEnginePlanning(unittest.TestCase):
    """Planlama: katmanlar, parçalar, kaynak toplamı."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_we_")
        self.snap = ResourceSnapshot(cpu_percent=10, ram_percent=10,
                                     vram_percent=10, ram_total_mb=32768,
                                     vram_total_mb=11264, cpu_cores=24)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _steps(self, count: int, *, chain_every: int = 0,
               ram: float = 40.0) -> List[WorkStep]:
        adimlar = []
        for i in range(count):
            dep = [f"s{i-1:03d}"] if (chain_every and i % chain_every) else []
            adimlar.append(WorkStep(id=f"s{i:03d}", title=f"Adım {i}",
                                    kind="k", needs={"ram_mb": ram},
                                    depends_on=dep))
        return adimlar

    def test_independent_steps_are_parallelizable(self) -> None:
        """
        Bağımsız adımlar **paralel çalışabilir** olmalı.

        İki koşul birlikte aranır:

            1. Tek bir bağımlılık katmanında olmalılar (`_layers`)
            2. Parça içi bağımlılık taşımamalılar — yani bir parçanın
               diğerini beklemesi gerekmez

        Neden parça sayısı değil de bu? Planlayıcı bağımsız adımları
        **bilinçli olarak** ayrı parçalara böler; aksi hâlde hepsi tek
        parçaya düşer ve eşzamanlı çalışamaz (ölçüldü: 8 bağımsız adım tek
        parçada seri çalıştı, hızlanma 1,0×).
        """
        adimlar = self._steps(6)
        motor = WorkEngine(workspace=self.root)
        katmanlar = motor._layers(adimlar)
        self.assertEqual(len(katmanlar), 1,
                         "bağımsız adımlar tek katmanda olmalı")
        self.assertEqual(len(katmanlar[0]), 6)

        motor.plan("p", adimlar, snapshot=self.snap)
        ic_bagimlilik = 0
        for parca in motor.chunks:
            icerde = {a.id for a in parca.steps}
            for adim in parca.steps:
                ic_bagimlilik += sum(1 for d in adim.depends_on if d in icerde)
        self.assertEqual(ic_bagimlilik, 0,
                         "paralel parçalar birbirini beklememeli")
        # Ve gerçekten birden fazla parçaya bölünmüş olmalı
        self.assertGreater(len(motor.chunks), 1)

    def test_dependency_creates_layers(self) -> None:
        """Zincir bağımlılık katman sayısını artırmalı."""
        duz = WorkEngine(workspace=self.root)
        duz.plan("p", self._steps(6), snapshot=self.snap)
        zincir = WorkEngine(workspace=self.root)
        zincir.plan("p", self._steps(9, chain_every=3), snapshot=self.snap)
        self.assertGreater(len(zincir.chunks), len(duz.chunks))

    def test_all_steps_assigned_to_chunk(self) -> None:
        """Her adım **tam olarak bir** parçaya atanmalı."""
        adimlar = self._steps(20, chain_every=5)
        motor = WorkEngine(workspace=self.root)
        parcalar = motor.plan("p", adimlar, snapshot=self.snap)
        atanan = [a.id for p in parcalar for a in p.steps]
        self.assertEqual(sorted(atanan), sorted(a.id for a in adimlar))
        for adim in adimlar:
            self.assertTrue(adim.chunk_id)

    def test_chunk_needs_summed(self) -> None:
        """Parça kaynak ihtiyacı adımların toplamı olmalı."""
        motor = WorkEngine(workspace=self.root)
        parcalar = motor.plan("p", self._steps(4, ram=100.0),
                              snapshot=self.snap)
        for parca in parcalar:
            beklenen = sum(a.needs.get("ram_mb", 0.0) for a in parca.steps)
            self.assertAlmostEqual(parca.needs.get("ram_mb", 0.0), beklenen)

    def test_chunk_respects_step_cap(self) -> None:
        """Parça, `chunk_steps` sınırını aşmamalı."""
        motor = WorkEngine(workspace=self.root, chunk_steps=3)
        parcalar = motor.plan("p", self._steps(20), snapshot=self.snap)
        for parca in parcalar:
            self.assertLessEqual(len(parca.steps), 3)

    def test_cycles_do_not_hang(self) -> None:
        """Döngüsel bağımlılık sonsuz döngüye yol açmamalı."""
        adimlar = [
            WorkStep(id="a", title="A", depends_on=["b"]),
            WorkStep(id="b", title="B", depends_on=["a"]),
        ]
        motor = WorkEngine(workspace=self.root)
        parcalar = motor.plan("döngü", adimlar, snapshot=self.snap)
        self.assertEqual(len([s for p in parcalar for s in p.steps]), 2)

    def test_empty_plan(self) -> None:
        motor = WorkEngine(workspace=self.root)
        self.assertEqual(motor.plan("boş", [], snapshot=self.snap), [])


class TestWorkEngineExecution(unittest.TestCase):
    """Yürütme: çalıştırma, kuyruklama, kayıt."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_wx_")
        self.snap = ResourceSnapshot(cpu_percent=10, ram_percent=10,
                                     vram_percent=10, ram_total_mb=32768,
                                     vram_total_mb=11264, cpu_cores=24)
        self.yogun = ResourceSnapshot(cpu_percent=85, ram_percent=85,
                                      vram_percent=85, ram_total_mb=4096,
                                      vram_total_mb=4096, cpu_cores=4)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _engine(self, count: int = 6, ram: float = 40.0,
                **kwargs: Any) -> WorkEngine:
        motor = WorkEngine(workspace=self.root, **kwargs)
        adimlar = [WorkStep(id=f"s{i}", title=f"Adım {i}", kind="k",
                            needs={"ram_mb": ram}) for i in range(count)]
        motor.plan("test", adimlar, snapshot=self.snap)
        motor.register("k", lambda a: f"ok:{a.id}")
        return motor

    def test_runs_all_steps(self) -> None:
        motor = self._engine()
        rapor = motor.run(snapshot=self.snap)
        self.assertEqual(rapor["steps_done"], 6)
        self.assertEqual(rapor["steps_failed"], 0)

    def test_handler_receives_step(self) -> None:
        motor = self._engine(2)
        gorulen: List[str] = []
        motor.register("k", lambda a: gorulen.append(a.id))
        motor.run(snapshot=self.snap)
        self.assertEqual(len(gorulen), 2)

    def test_missing_handler_is_dry_run(self) -> None:
        """İşleyici yoksa adım **çökmez**, kuru çalıştırma sayılır."""
        motor = WorkEngine(workspace=self.root)
        motor.plan("p", [WorkStep(id="a", title="A", kind="yok")],
                   snapshot=self.snap)
        rapor = motor.run(snapshot=self.snap)
        self.assertEqual(rapor["steps_done"], 1)

    def test_handler_exception_recorded(self) -> None:
        """İşleyici hatası kaydedilmeli, motor ayakta kalmalı."""
        motor = self._engine(3)

        def patla(adim: WorkStep) -> None:
            if adim.id == "s1":
                raise ValueError("kasıtlı")

        motor.register("k", patla)
        rapor = motor.run(snapshot=self.snap)
        self.assertEqual(rapor["steps_failed"], 1)
        self.assertEqual(rapor["steps_done"], 2)
        hatali = [a for a in motor.steps if a.status == FAILED]
        self.assertIn("kasıtlı", hatali[0].error)

    def test_heavy_work_deferred_not_run(self) -> None:
        """
        Bütçeyi aşan iş **çalıştırılmamalı**, kuyruğa alınmalı.

        Bu, bilgisayarı kilitlemeye karşı asıl korumadır.
        """
        motor = WorkEngine(workspace=self.root)
        adimlar = [WorkStep(id="agir", title="Çok ağır", kind="k",
                            needs={"ram_mb": 500_000.0})]
        motor.plan("ağır", adimlar, snapshot=self.yogun)
        cagrildi = {"n": 0}
        motor.register("k", lambda a: cagrildi.__setitem__("n", cagrildi["n"] + 1))
        rapor = motor.run(snapshot=self.yogun)
        self.assertEqual(cagrildi["n"], 0, "ağır iş çalıştırılmamalı")
        self.assertEqual(rapor["steps_done"], 0)
        self.assertGreater(rapor["chunks_blocked"], 0)
        self.assertIn(BLOCKED, {c.status for c in motor.chunks})

    def test_deferral_logged_with_budget(self) -> None:
        """Erteleme gerekçesi ve bütçe günlüğe yazılmalı (tanı için)."""
        motor = WorkEngine(workspace=self.root)
        motor.plan("ağır",
                   [WorkStep(id="a", title="A", kind="k",
                             needs={"ram_mb": 900_000.0})],
                   snapshot=self.yogun)
        motor.register("k", lambda a: None)
        motor.run(snapshot=self.yogun)
        kayitlar = WorkJournal(Path(self.root) / JOURNAL_FILE).load()
        ertelenen = [k for k in kayitlar if k.get("olay") == "ertelendi"]
        self.assertTrue(ertelenen)
        self.assertIn("bütçe", ertelenen[0])

    def test_max_chunks_limits_work(self) -> None:
        motor = self._engine(6)
        rapor = motor.run(snapshot=self.snap, max_chunks=2)
        self.assertLessEqual(rapor["chunks_done"], 2)
        self.assertGreater(rapor["chunks_blocked"], 0)

    def test_report_json_safe(self) -> None:
        motor = self._engine(3)
        assert_json_safe(self, motor.run(snapshot=self.snap), "work_report")

    def test_stats_json_safe(self) -> None:
        motor = self._engine(3)
        assert_json_safe(self, motor.get_stats(), "work_stats")


class TestWorkEngineParallel(unittest.TestCase):
    """Gerçek paralellik — IDM'in özü."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_wp_")
        self.snap = ResourceSnapshot(cpu_percent=5, ram_percent=5,
                                     vram_percent=5, ram_total_mb=32768,
                                     vram_total_mb=11264, cpu_cores=24)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_independent_work_runs_in_parallel(self) -> None:
        """
        Bağımsız parçalar **eşzamanlı** çalışmalı.

        Ölçüt: 8 adım × 0,2 sn bekleme. Seri ≈ 1,6 sn; paralel belirgin
        biçimde kısa olmalı. Bu, boştaki kapasitenin kullanıldığının
        kanıtıdır.
        """
        motor = WorkEngine(workspace=self.root)
        adimlar = [WorkStep(id=f"p{i}", title=f"P{i}", kind="io",
                            needs={"ram_mb": 20.0}) for i in range(8)]
        motor.plan("paralel", adimlar, snapshot=self.snap)

        def bekle(adim: WorkStep) -> str:
            time.sleep(0.2)
            return f"ok:{adim.id}"

        motor.register("io", bekle)
        t0 = time.perf_counter()
        rapor = motor.run(snapshot=self.snap)
        sure = time.perf_counter() - t0
        self.assertEqual(rapor["steps_done"], 8)
        self.assertLess(sure, 1.6 * 0.75,
                        f"paralellik yok gibi: {sure:.2f}s (seri ≈1,6s)")

    def test_parallel_execution_is_thread_safe(self) -> None:
        """Paralel yürütmede sayaçlar ve günlük bozulmamalı."""
        motor = WorkEngine(workspace=self.root)
        adimlar = [WorkStep(id=f"t{i}", title=f"T{i}", kind="io",
                            needs={"ram_mb": 20.0}) for i in range(12)]
        motor.plan("iş parçacığı", adimlar, snapshot=self.snap)
        motor.register("io", lambda a: time.sleep(0.01))
        rapor = motor.run(snapshot=self.snap)
        self.assertEqual(rapor["steps_done"], 12)
        kayitlar = WorkJournal(Path(self.root) / JOURNAL_FILE).load()
        adim_kayitlari = [k for k in kayitlar if k.get("olay") == "adım"]
        self.assertEqual(len(adim_kayitlari), 12)

    def test_chunk_failure_isolated(self) -> None:
        """Bir parça çökse diğerleri tamamlanmalı."""
        motor = WorkEngine(workspace=self.root)
        adimlar = [WorkStep(id=f"f{i}", title=f"F{i}", kind="k",
                            needs={"ram_mb": 20.0}) for i in range(6)]
        motor.plan("hata", adimlar, snapshot=self.snap)

        def bazen_patla(adim: WorkStep) -> str:
            if adim.id == "f0":
                raise RuntimeError("kasıtlı çökme")
            return "ok"

        motor.register("k", bazen_patla)
        rapor = motor.run(snapshot=self.snap)
        self.assertEqual(rapor["steps_failed"], 1)
        self.assertEqual(rapor["steps_done"], 5)


class TestWorkEngineAlgorithm(unittest.TestCase):
    """Algoritma üretimi ve devam edebilme."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_wa_")
        self.snap = ResourceSnapshot(cpu_percent=10, ram_percent=10,
                                     vram_percent=10, ram_total_mb=32768,
                                     vram_total_mb=11264, cpu_cores=24)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_algorithm_contains_steps_and_results(self) -> None:
        """Algoritma, adımları ve **gerçekleşen** sonuçları içermeli."""
        motor = WorkEngine(workspace=self.root)
        adimlar = [WorkStep(id=f"a{i}", title=f"Modül {i}", kind="k")
                   for i in range(4)]
        motor.plan("algoritma testi", adimlar, snapshot=self.snap)
        motor.register("k", lambda a: "tamam")
        motor.run(snapshot=self.snap)
        metin = motor.algorithm()
        self.assertIn("algoritma testi", metin)
        self.assertIn("Modül 0", metin)
        self.assertIn("başarılı", metin)

    def test_algorithm_reports_failures(self) -> None:
        motor = WorkEngine(workspace=self.root)
        motor.plan("hata", [WorkStep(id="x", title="Bozuk", kind="k")],
                   snapshot=self.snap)
        motor.register("k", lambda a: (_ for _ in ()).throw(ValueError("olmadi")))
        motor.run(snapshot=self.snap)
        self.assertIn("HATA", motor.algorithm())

    def test_algorithm_limit(self) -> None:
        motor = WorkEngine(workspace=self.root)
        motor.plan("p", [WorkStep(id=f"s{i}", title=f"S{i}") for i in range(30)],
                   snapshot=self.snap)
        kisa = motor.algorithm(limit=5)
        self.assertIn("S0", kisa)
        self.assertNotIn("S25", kisa)

    def test_resume_state_from_journal(self) -> None:
        """
        Kesintiden sonra **nerede kalındığı** bilinmeli.

        Günlük taze bir örnekten okunur (süreç yeniden başlamış gibi).
        """
        motor = WorkEngine(workspace=self.root)
        adimlar = [WorkStep(id=f"r{i}", title=f"R{i}", kind="k")
                   for i in range(5)]
        motor.plan("devam", adimlar, snapshot=self.snap)
        motor.register("k", lambda a: "ok")
        motor.run(snapshot=self.snap)

        yeni = WorkEngine(workspace=self.root)
        durum = yeni.resume_state()
        self.assertEqual(durum["completed_count"], 5)
        self.assertIn("r0", durum["completed_steps"])
        self.assertIn("devam", durum["objective"])

    def test_resume_state_empty_without_journal(self) -> None:
        motor = WorkEngine(workspace=self.root)
        durum = motor.resume_state()
        self.assertEqual(durum["completed_count"], 0)

    def test_journal_records_start_and_end(self) -> None:
        """Günlük başlangıç ve bitiş olaylarını taşımalı."""
        motor = WorkEngine(workspace=self.root)
        motor.plan("p", [WorkStep(id="a", title="A")], snapshot=self.snap)
        motor.run(snapshot=self.snap)
        olaylar = [k.get("olay") for k in
                   WorkJournal(Path(self.root) / JOURNAL_FILE).load()]
        self.assertIn("başladı", olaylar)
        self.assertIn("bitti", olaylar)

    def test_chunk_dict_json_safe(self) -> None:
        motor = WorkEngine(workspace=self.root)
        parcalar = motor.plan("p", [WorkStep(id="a", title="A")],
                              snapshot=self.snap)
        assert_json_safe(self, parcalar[0].to_dict(), "chunk")
        assert_json_safe(self, motor.steps[0].to_dict(), "step")


class TestQueuePersistence(unittest.TestCase):
    """Kuyruk ısrarı: kaynak açılınca devam, açılmazsa dürüst rapor."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_qp_")
        self.bos = ResourceSnapshot(cpu_percent=5, ram_percent=5,
                                    vram_percent=5, ram_total_mb=32768,
                                    vram_total_mb=11264, cpu_cores=24)
        self.yogun = ResourceSnapshot(cpu_percent=88, ram_percent=88,
                                      vram_percent=88, ram_total_mb=4096,
                                      vram_total_mb=4096, cpu_cores=4)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _engine(self, *, ram: float = 20.0) -> WorkEngine:
        motor = WorkEngine(workspace=self.root)
        motor.plan("kuyruk", [
            WorkStep(id="hafif", title="Hafif", kind="k",
                     needs={"ram_mb": ram}),
            WorkStep(id="agir", title="Ağır", kind="k",
                     needs={"ram_mb": 900_000.0}),
        ], snapshot=self.bos)
        motor.register("k", lambda a: f"ok:{a.id}")
        return motor

    def test_waits_and_reports_when_resource_never_frees(self) -> None:
        """
        Kaynak açılmazsa **dürüstçe** raporlamalı.

        Sessizce pes etmek kabul edilemez: kullanıcı işin bittiğini sanır.
        Bu yüzden `still_blocked` alanı ve gerekçe döner.
        """
        motor = self._engine()
        rapor = motor.run_until_done(snapshot=self.yogun, max_wait_s=1.0,
                                     poll_s=0.2)
        # Hafif adım çalıştı, ağır kuyrukta kaldı
        self.assertEqual(rapor["steps_done"], 1)
        self.assertTrue(rapor["still_blocked"])
        self.assertIn("ram_mb", rapor["still_blocked"][0]["needs"])
        self.assertGreaterEqual(rapor["rounds"], 1)
        self.assertIn("waited_s", rapor)

    def test_finishes_when_resources_available(self) -> None:
        """Kaynak yeterliyse tek turda bitmeli, bekleme olmamalı."""
        motor = WorkEngine(workspace=self.root)
        motor.plan("yeterli", [
            WorkStep(id=f"a{i}", title=f"A{i}", kind="k",
                     needs={"ram_mb": 20.0}) for i in range(4)
        ], snapshot=self.bos)
        motor.register("k", lambda a: "ok")
        rapor = motor.run_until_done(snapshot=self.bos, max_wait_s=5.0,
                                     poll_s=0.2)
        self.assertEqual(rapor["steps_done"], 4)
        self.assertEqual(rapor["still_blocked"], [])
        self.assertEqual(rapor["waited_s"], 0.0)

    def test_stop_event_halts_waiting(self) -> None:
        """Dışarıdan durdurma beklemeyi kesmeli."""
        motor = self._engine()
        durdur = threading.Event()
        durdur.set()
        rapor = motor.run_until_done(snapshot=self.yogun, max_wait_s=30.0,
                                     poll_s=0.2, stop_event=durdur)
        self.assertLess(rapor["waited_s"], 5.0)
        self.assertLessEqual(rapor["rounds"], 2)

    def test_queue_summary_reports_blocking_reason(self) -> None:
        """`deferred_summary` neyin beklendiğini söylemeli."""
        motor = self._engine()
        motor.run(snapshot=self.yogun)
        ozet = motor.deferred_summary()
        self.assertEqual(ozet["count"], 1)
        self.assertEqual(ozet["steps"], 1)
        self.assertTrue(ozet["reason"])

    def test_queue_summary_empty_when_all_done(self) -> None:
        motor = WorkEngine(workspace=self.root)
        motor.plan("temiz", [WorkStep(id="a", title="A", kind="k",
                                      needs={"ram_mb": 10.0})],
                   snapshot=self.bos)
        motor.register("k", lambda a: "ok")
        motor.run(snapshot=self.bos)
        ozet = motor.deferred_summary()
        self.assertEqual(ozet["count"], 0)
        self.assertEqual(ozet["steps"], 0)

    def test_queue_events_logged(self) -> None:
        """Kuyruk bitişi günlüğe yazılmalı (izlenebilirlik)."""
        motor = self._engine()
        motor.run_until_done(snapshot=self.yogun, max_wait_s=1.0, poll_s=0.2)
        olaylar = [k.get("olay") for k in
                   WorkJournal(Path(self.root) / JOURNAL_FILE).load()]
        self.assertIn("kuyruk_bitti", olaylar)

    def test_run_until_done_report_json_safe(self) -> None:
        motor = self._engine()
        rapor = motor.run_until_done(snapshot=self.yogun, max_wait_s=1.0,
                                     poll_s=0.2)
        assert_json_safe(self, rapor, "queue_report")


class TestAppendPlan(unittest.TestCase):
    """Plana sonradan adım ekleme — tamamlananlar korunur."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_ap_")
        self.snap = ResourceSnapshot(cpu_percent=5, ram_percent=5,
                                     vram_percent=5, ram_total_mb=32768,
                                     vram_total_mb=11264, cpu_cores=24)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_completed_steps_preserved(self) -> None:
        """
        Tamamlanan adım **yeniden çalıştırılmamalı**.

        Neden? Uzun işler aşamalı planlanır; her yeni aşamada tamamlanmış
        işi tekrarlamak hem zaman hem yan etki (dosya yazma) kaybıdır.
        """
        motor = WorkEngine(workspace=self.root)
        motor.plan("aşamalı", [WorkStep(id="s1", title="Aşama 1", kind="k",
                                        needs={"ram_mb": 10.0})],
                   snapshot=self.snap)
        cagri: List[str] = []
        motor.register("k", lambda a: cagri.append(a.id) or "ok")
        motor.run(snapshot=self.snap)
        self.assertEqual(cagri, ["s1"])

        motor.append_plan("aşamalı", [
            WorkStep(id="s2", title="Aşama 2", kind="k",
                     needs={"ram_mb": 10.0})])
        self.assertEqual(motor.steps[0].status, DONE)
        motor.run(snapshot=self.snap)
        self.assertEqual(cagri, ["s1", "s2"], "tamamlanan adım tekrarlanmamalı")

    def test_duplicate_ids_ignored(self) -> None:
        motor = WorkEngine(workspace=self.root)
        motor.plan("p", [WorkStep(id="a", title="A", kind="k")],
                   snapshot=self.snap)
        motor.append_plan("p", [WorkStep(id="a", title="A tekrar", kind="k")])
        self.assertEqual(len(motor.steps), 1)

    def test_objective_kept_when_blank(self) -> None:
        motor = WorkEngine(workspace=self.root)
        motor.plan("ilk amaç", [WorkStep(id="a", title="A")],
                   snapshot=self.snap)
        motor.append_plan("", [WorkStep(id="b", title="B")])
        self.assertEqual(motor._objective, "ilk amaç")


if __name__ == "__main__":
    unittest.main(verbosity=2)
