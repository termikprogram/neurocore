"""
Kaynak nezaketi testleri ("boştakinin %90'ı" politikası).

Bu testler, ajanın kullanıcının bilgisayarını kilitlemesini önleyen
politikanın **sayısal doğruluğunu** doğrular. Gerçek olay sonrası eklendi:
ajan sistem kaynaklarını doyurup makineyi kilitledi.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict

from tests.helpers import assert_json_safe, isolated_config

from neurocore.core.config import HardwareConfig, HeadroomPolicy, SystemConfig
from neurocore.hardware.allocator import (
    MemoryPressure,
    ResourceAllocator,
    ResourceBudget,
)


class FakeSnapshot:
    """
    Test için yapay sistem anlık görüntüsü.

    Gerçek `psutil`/NVML çağrıları yapmadan politika matematiğini sınamayı
    sağlar; her alan istenen senaryoya göre ayarlanabilir.
    """

    def __init__(self, *, cpu_percent: float = 10.0, ram_percent: float = 20.0,
                 ram_total_mb: float = 32768.0, cores: int = 24,
                 gpu: bool = True, vram_total_mb: float = 11264.0,
                 gpu_vram_percent: float = 10.0) -> None:
        self.cpu_percent = cpu_percent
        self.cpu_count_logical = cores
        self.cpu_count_physical = max(1, cores // 2)
        self.ram_percent = ram_percent
        self.ram_total_mb = ram_total_mb
        self.ram_used_mb = ram_total_mb * ram_percent / 100.0
        self.ram_free_mb = ram_total_mb - self.ram_used_mb
        self.swap_percent = 0.0
        self.disk_percent = 40.0
        self.gpu_available = gpu
        self.gpu_name = "Test GPU" if gpu else ""
        self.gpu_vram_total_mb = vram_total_mb if gpu else 0.0
        self.gpu_vram_percent = gpu_vram_percent if gpu else 0.0
        self.gpu_vram_used_mb = vram_total_mb * gpu_vram_percent / 100.0 if gpu else 0.0
        self.gpu_vram_free_mb = (vram_total_mb - self.gpu_vram_used_mb) if gpu else 0.0
        self.gpu_utilization = 5.0 if gpu else 0.0
        self.gpu_temperature_c = 45 if gpu else -1
        self.os_name = "TestOS"
        self.hostname = "test-host"
        self.uptime_seconds = 1000.0


class TestHeadroomMath(unittest.TestCase):
    """Politika matematiği: kullanıcının istediği kuralın doğrulanması."""

    def setUp(self) -> None:
        self.policy = HeadroomPolicy()

    def test_rule_stated_by_user(self) -> None:
        """
        Kullanıcının kuralı: %50 dolu → %40 bütçe, %90 boş → %80 bütçe.

        Ajan boşta olan kaynağın %90'ını kullanır:
            %50 dolu → boş %50 → bütçe %45
            %10 dolu → boş %90 → bütçe %81
        """
        # 24 çekirdek üzerinden CPU
        self.assertEqual(self.policy.cpu_budget(24, 0.50), 10)   # %42 (≈%45)
        self.assertEqual(self.policy.cpu_budget(24, 0.10), 19)   # %79 (≈%81)

    def test_cpu_budget_shrinks_with_usage(self) -> None:
        """Kullanım arttıkça ayrılan çekirdek sayısı azalır."""
        budgets = [self.policy.cpu_budget(24, used / 100.0)
                   for used in (0, 25, 50, 75, 95)]
        self.assertEqual(budgets, sorted(budgets, reverse=True))
        self.assertLess(budgets[-1], budgets[0])

    def test_cpu_budget_never_zero(self) -> None:
        """Sistem tamamen dolu olsa bile asgari çekirdek ayrılır."""
        self.assertGreaterEqual(self.policy.cpu_budget(24, 1.0), 1)
        self.assertGreaterEqual(self.policy.cpu_budget(2, 0.99), 1)

    def test_cpu_respects_hard_cap(self) -> None:
        """Boş sistemde bile üst sınır aşılmaz."""
        budget = self.policy.cpu_budget(64, 0.0)
        self.assertLessEqual(budget, int(64 * self.policy.max_cpu_percent))

    def test_ram_budget_shrinks_with_usage(self) -> None:
        """RAM bütçesi kullanımla birlikte azalır."""
        total = 32768.0
        budgets = [
            self.policy.ram_budget_mb(total, total * (1 - used / 100.0))
            for used in (10, 30, 50, 70, 90)
        ]
        self.assertEqual(budgets, sorted(budgets, reverse=True))
        self.assertGreater(budgets[0], budgets[-1] * 2)

    def test_ram_budget_reserves_system_memory(self) -> None:
        """İşletim sistemi için her zaman pay bırakılır."""
        total = 8192.0
        budget = self.policy.ram_budget_mb(total, total)  # tamamen boş
        self.assertLessEqual(budget, total - self.policy.reserve_system_mb)
        self.assertGreaterEqual(budget, self.policy.ram_floor_mb)

    def test_ram_budget_follows_90_percent_rule(self) -> None:
        """
        Boşta olan RAM'in %90'ı kuralı: %50 dolu → %45 bütçe.

        Üst sınır (`max_ram_percent`) devreye girmemeli (%45 < %85).
        """
        total = 32768.0
        budget = self.policy.ram_budget_mb(total, total * 0.5)
        expected = total * 0.5 * 0.90 - self.policy.reserve_system_mb
        self.assertAlmostEqual(budget, expected, delta=1.0)
        self.assertAlmostEqual(budget / total, 0.434, delta=0.01)

    def test_gpu_budget_follows_90_percent_rule(self) -> None:
        """VRAM bütçesi de aynı kurala uyar."""
        total = 11264.0
        budget = self.policy.gpu_budget_mb(total, total * 0.5)
        self.assertAlmostEqual(budget, total * 0.5 * 0.90, delta=1.0)

    def test_gpu_budget_zero_without_gpu(self) -> None:
        """GPU yoksa VRAM bütçesi sıfırdır."""
        self.assertEqual(self.policy.gpu_budget_mb(0.0, 0.0), 0.0)

    def test_free_fraction_clamped(self) -> None:
        """Boş oran 0..1 aralığına kırpılır."""
        self.assertEqual(HeadroomPolicy.free_fraction(1.5), 0.0)
        self.assertEqual(HeadroomPolicy.free_fraction(-0.5), 1.0)
        self.assertAlmostEqual(HeadroomPolicy.free_fraction(0.3), 0.7)


class TestStallDecisions(unittest.TestCase):
    """Duraklama ve baskı seviyesi kararları."""

    def setUp(self) -> None:
        self.policy = HeadroomPolicy()

    def test_no_stall_when_idle(self) -> None:
        """Boş sistemde yeni iş başlatılır."""
        stall, reason = self.policy.should_stall(cpu_percent=10, ram_percent=20,
                                                 gpu_percent=5)
        self.assertFalse(stall)
        self.assertEqual(reason, "")

    def test_stall_on_high_ram(self) -> None:
        """RAM doyduğunda yeni iş başlatılmaz."""
        stall, reason = self.policy.should_stall(ram_percent=90)
        self.assertTrue(stall)
        self.assertIn("RAM", reason)

    def test_stall_on_high_cpu(self) -> None:
        """CPU doyduğunda yeni iş başlatılmaz."""
        stall, reason = self.policy.should_stall(cpu_percent=92)
        self.assertTrue(stall)
        self.assertIn("CPU", reason)

    def test_stall_on_high_vram(self) -> None:
        """VRAM doyduğunda yeni iş başlatılmaz."""
        stall, reason = self.policy.should_stall(gpu_percent=95)
        self.assertTrue(stall)
        self.assertIn("VRAM", reason)

    def test_pressure_levels(self) -> None:
        """Baskı seviyeleri eşiklere göre doğru sınıflandırılır."""
        self.assertEqual(self.policy.pressure_level(10, 10, 10), "NORMAL")
        self.assertEqual(self.policy.pressure_level(75, 20, 10), "ELEVATED")
        self.assertEqual(self.policy.pressure_level(82, 20, 10), "HIGH")
        self.assertEqual(self.policy.pressure_level(95, 95, 95), "CRITICAL")

    def test_recover_after_stall(self) -> None:
        """Kaynaklar boşalınca toparlanma bildirilir."""
        self.assertFalse(self.policy.can_recover(cpu_percent=90, ram_percent=90))
        self.assertTrue(self.policy.can_recover(cpu_percent=20, ram_percent=30,
                                               gpu_percent=10))

    def test_recovery_delay_ordering(self) -> None:
        """Kritik baskıda bekleme daha uzundur."""
        self.assertGreater(self.policy.recovery_delay("CRITICAL"),
                           self.policy.recovery_delay("HIGH"))
        self.assertGreater(self.policy.recovery_delay("HIGH"),
                           self.policy.recovery_delay("NORMAL"))
        self.assertEqual(self.policy.recovery_delay("NORMAL"), 0.0)

    def test_describe_and_json(self) -> None:
        """Politika açıklaması ve JSON çıktısı üretilir."""
        self.assertIn("boştakinin", self.policy.describe())
        assert_json_safe(self, self.policy.to_dict(), "headroom_policy")


class TestAllocatorHeadroom(unittest.TestCase):
    """Tahsis edicinin politikayı uygulaması."""

    def setUp(self) -> None:
        self.config = SystemConfig()
        self.allocator = ResourceAllocator()
        self.allocator.initialize(self.config, None)

    def tearDown(self) -> None:
        self.allocator.shutdown()

    def test_budget_reports_headroom(self) -> None:
        """Bütçe, nezaket bilgilerini raporlar."""
        budget = self.allocator.allocate(FakeSnapshot(cpu_percent=30, ram_percent=40,
                                                      gpu_vram_percent=20))
        self.assertAlmostEqual(budget.cpu_percent, 30.0)
        self.assertAlmostEqual(budget.ram_percent, 40.0)
        self.assertAlmostEqual(budget.gpu_percent, 20.0)
        self.assertGreater(budget.cpu_budget_percent, 0)
        self.assertGreater(budget.ram_budget_percent, 0)
        self.assertIn("headroom", budget.to_dict())

    def test_idle_system_gets_large_budget(self) -> None:
        """Boş sistemde ajan geniş bütçe alır."""
        budget = self.allocator.allocate(FakeSnapshot(cpu_percent=5, ram_percent=15,
                                                      gpu_vram_percent=5))
        self.assertEqual(budget.device, "cuda")
        self.assertGreater(budget.max_gpu_mb, 4000)
        self.assertGreater(budget.max_ram_mb, 15000)
        self.assertFalse(budget.should_stall)
        self.assertEqual(budget.headroom_pressure, "NORMAL")

    def test_busy_system_shrinks_budget(self) -> None:
        """Yük altındaki sistemde bütçe küçülür."""
        idle = self.allocator.allocate(FakeSnapshot(cpu_percent=5, ram_percent=15,
                                                    gpu_vram_percent=5))
        idle_ram = idle.max_ram_mb
        idle_batch = idle.max_batch_size

        busy = self.allocator.allocate(FakeSnapshot(cpu_percent=70, ram_percent=80,
                                                    gpu_vram_percent=70))
        self.assertLess(busy.max_ram_mb, idle_ram)
        self.assertLessEqual(busy.max_batch_size, idle_batch)
        self.assertIn(busy.headroom_pressure, ("ELEVATED", "HIGH", "CRITICAL"))

    def test_stall_flagged_when_full(self) -> None:
        """Kaynaklar dolduğunda `should_stall` işaretlenir."""
        budget = self.allocator.allocate(FakeSnapshot(cpu_percent=90, ram_percent=92,
                                                      gpu_vram_percent=95))
        self.assertTrue(budget.should_stall)
        self.assertTrue(budget.stall_reason)
        self.assertEqual(budget.memory_pressure, MemoryPressure.CRITICAL)

    def test_cpu_budget_bounds_num_workers(self) -> None:
        """Worker sayısı CPU bütçesini aşmaz."""
        budget = self.allocator.allocate(FakeSnapshot(cpu_percent=70, cores=24))
        self.assertLessEqual(budget.num_workers, 24)
        self.assertGreaterEqual(budget.num_workers, 1)

    def test_no_gpu_falls_back_to_cpu(self) -> None:
        """GPU yoksa CPU moduna düşülür."""
        budget = self.allocator.allocate(FakeSnapshot(gpu=False))
        self.assertEqual(budget.device, "cpu")
        self.assertEqual(budget.max_gpu_mb, 0.0)
        self.assertFalse(budget.use_mixed_precision)

    def test_hysteresis_prevents_rapid_growth(self) -> None:
        """Bütçe küçüldükten sonra hemen büyümez (histerezis)."""
        busy = self.allocator.allocate(FakeSnapshot(cpu_percent=80, ram_percent=85,
                                                    gpu_vram_percent=85))
        shrunk_ram = busy.max_ram_mb
        # Aynı anda sistem boşalsa bile bütçe cooldown boyunca küçük kalmalı
        recovered = self.allocator.allocate(FakeSnapshot(cpu_percent=2, ram_percent=5,
                                                         gpu_vram_percent=2))
        self.assertLessEqual(recovered.max_ram_mb, shrunk_ram + 1.0)
        self.assertLessEqual(recovered.max_batch_size, busy.max_batch_size)

    def test_policy_override(self) -> None:
        """Politika çalışma zamanında değiştirilebilir."""
        strict = HeadroomPolicy(cpu=0.25, ram=0.25, gpu=0.25,
                                max_ram_percent=0.30, max_gpu_percent=0.30)
        self.allocator.set_policy(strict)
        budget = self.allocator.allocate(FakeSnapshot(cpu_percent=10, ram_percent=10,
                                                      gpu_vram_percent=10))
        self.assertLess(budget.max_ram_mb, 32768 * 0.30 + 1)
        self.assertLess(budget.max_gpu_mb, 11264 * 0.30 + 1)

    def test_budget_json_safe(self) -> None:
        """Bütçe JSON-güvenli."""
        budget = self.allocator.allocate(FakeSnapshot())
        assert_json_safe(self, budget.to_dict(), "resource_budget")
        self.assertIn("bütçe", budget.describe_headroom())


class TestTrainerResourceGuard(unittest.TestCase):
    """Eğitim döngüsündeki kaynak gözcüsü."""

    def setUp(self) -> None:
        from neurocore.neural.trainer import ResourceGuard, TrainingPlan

        self.guard = ResourceGuard(enabled=True, pause_seconds=0.01,
                                   max_pressure_events=3)
        self.plan = TrainingPlan(batch_size=16, seq_len=256, grad_accumulation=1)

    def test_allows_when_idle(self) -> None:
        """Boş sistemde eylem yok."""
        self.assertEqual(self.guard.check(ram_percent=20, cpu_percent=10), "none")
        self.assertEqual(self.guard.pressure_events, 0)

    def test_pauses_under_pressure(self) -> None:
        """Baskı altında duraklama kararı verilir."""
        action = self.guard.check(ram_percent=90, cpu_percent=30)
        self.assertEqual(action, "pause")
        self.assertEqual(self.guard.pressure_events, 1)
        self.assertTrue(self.guard.last_reason)

    def test_stops_after_repeated_pressure(self) -> None:
        """Sürekli baskı eğitimi durdurur (sistemi yormamak için)."""
        actions = [self.guard.check(ram_percent=92) for _ in range(4)]
        self.assertIn("stop", actions, f"eylemler: {actions}")

    def test_reduces_batch_progressively(self) -> None:
        """Toplu iş kademeli küçülür, sonra dizi uzunluğu, sonra birikim."""
        self.assertTrue(self.guard.reduce_batch(self.plan))
        self.assertEqual(self.plan.batch_size, 8)
        self.assertTrue(self.guard.reduce_batch(self.plan))
        self.assertEqual(self.plan.batch_size, 4)
        while self.plan.batch_size > 1:
            self.guard.reduce_batch(self.plan)
        self.assertTrue(self.guard.reduce_batch(self.plan))
        self.assertLessEqual(self.plan.seq_len, 128)
        self.assertGreaterEqual(self.plan.grad_accumulation, 1)

    def test_disabled_guard_allows_everything(self) -> None:
        """Gözcü kapatılırsa hiçbir kısıt uygulanmaz."""
        guard = __import__("neurocore.neural.trainer",
                           fromlist=["ResourceGuard"]).ResourceGuard(enabled=False)
        self.assertEqual(guard.check(ram_percent=99, cpu_percent=99), "none")

    def test_guard_json_safe(self) -> None:
        """Gözcü raporu JSON-güvenli."""
        self.guard.check(ram_percent=50)
        assert_json_safe(self, self.guard.to_dict(), "resource_guard")


class TestEngineResourceGating(unittest.TestCase):
    """Motorun ağır işleri kaynak durumuna göre kapılaması."""

    def test_engine_reports_headroom(self) -> None:
        """Motor nezaket raporu üretir."""
        from tests.helpers import engine_context

        with engine_context() as engine:
            report = engine.resource_report()
            self.assertIn("headroom", report)
            self.assertIn("policy", report)
            self.assertIn("bütçe", report["summary"])
            assert_json_safe(self, report, "resource_report")

    def test_engine_gates_heavy_work(self) -> None:
        """Kaynaklar dolu olduğunda ağır iş reddedilir (gerekçesiyle)."""
        from tests.helpers import engine_context

        with engine_context() as engine:
            available, reason = engine.resources_available(for_heavy_work=True)
            if available:
                # Sistemi yapay olarak "dolu" göster
                engine._budget["headroom"] = {
                    "should_stall": True,
                    "stall_reason": "test: RAM %95",
                    "pressure": "CRITICAL",
                }
                available, reason = engine.resources_available(for_heavy_work=True)
            self.assertFalse(available)
            self.assertTrue(reason)

            outcome = engine.train(steps=5)
            self.assertFalse(outcome.get("ok"))
            self.assertEqual(outcome.get("error"), "kaynaklar_dolu")
            self.assertIn("resources", outcome)

    def test_policy_disable_allows_work(self) -> None:
        """Politika kapatılırsa ağır iş kapısı açılır."""
        from tests.helpers import engine_context

        with engine_context() as engine:
            engine.config.hardware.enforce_headroom = False
            engine._budget["headroom"] = {"should_stall": True,
                                          "stall_reason": "test"}
            available, reason = engine.resources_available(for_heavy_work=True)
            self.assertTrue(available)


if __name__ == "__main__":
    unittest.main(verbosity=2)
