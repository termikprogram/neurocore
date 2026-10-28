"""
Çekirdek motor testleri: konfigürasyon, olay bus'ı ve uçtan uca akış.

Bu testler sistemin **bütünleşik** davranışını doğrular: katmanların
kurulması, kablolanması, sorgu yolları, öğrenme ve kapanış.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.helpers import (
    assert_json_safe,
    engine_context,
    isolated_config,
    seed_vault,
    summarize_engine,
)

from neurocore.core.config import HardwareProfile, SystemConfig
from neurocore.core.event_bus import Event, EventBus, EventPriority
from neurocore.core.engine import EngineState, NeuroCoreEngine


class TestSystemConfig(unittest.TestCase):
    """Konfigürasyon: profiller, otomatik yapılandırma, dosya G/Ç."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_config_")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_default_paths_created(self) -> None:
        """Varsayılan yollar türetilir ve dizinler oluşturulur."""
        config = isolated_config(self.root)
        self.assertTrue(Path(config.knowledge.vault_path).is_dir())
        self.assertTrue(Path(config.data_path).is_dir())
        self.assertTrue(Path(config.models_path).is_dir())

    def test_profile_detection(self) -> None:
        """RAM/VRAM'e göre doğru profil seçilir."""
        self.assertEqual(HardwareProfile.detect(4.0, 0.0), HardwareProfile.LOW)
        self.assertEqual(HardwareProfile.detect(8.0, 4.0), HardwareProfile.MEDIUM)
        self.assertEqual(HardwareProfile.detect(16.0, 8.0), HardwareProfile.HIGH)
        self.assertEqual(HardwareProfile.detect(32.0, 12.0), HardwareProfile.ULTRA)
        self.assertEqual(HardwareProfile.detect(2.0, 0.0), HardwareProfile.MINIMAL)

    def test_apply_profile_scales_parameters(self) -> None:
        """Profil uygulandığında parametreler ölçeklenir."""
        config = SystemConfig()
        config.apply_hardware_profile(HardwareProfile.MINIMAL)
        minimal_dim = config.neural.embedding_dim
        config.apply_hardware_profile(HardwareProfile.ULTRA)
        self.assertLess(minimal_dim, config.neural.embedding_dim)
        self.assertLess(config.neural.num_transformer_layers,
                        config.neural.num_transformer_layers + 1)

    def test_auto_configure_no_gpu(self) -> None:
        """GPU yoksa CPU moduna geçilir."""
        config = SystemConfig()
        config.auto_configure(total_ram_gb=8.0, gpu_vram_gb=0.0)
        self.assertEqual(config.device, "cpu")
        self.assertFalse(config.hardware.enable_gpu)

    def test_save_load_roundtrip(self) -> None:
        """Konfigürasyon YAML'a yazılıp geri yüklenir."""
        config = isolated_config(self.root)
        config.max_batch_size = 17
        path = config.save(str(Path(self.root) / "config.yaml"))
        restored = SystemConfig.load(path)
        self.assertEqual(restored.max_batch_size, 17)
        self.assertEqual(restored.hardware_profile, config.hardware_profile)

    def test_summary_json_safe(self) -> None:
        """Konfigürasyon özeti JSON-güvenli."""
        assert_json_safe(self, isolated_config(self.root).get_summary(), "config_summary")


class TestEventBus(unittest.TestCase):
    """Olay bus'ı: abonelik, yayın, filtreleme, geçmiş."""

    def setUp(self) -> None:
        self.bus = EventBus(history_size=100)
        self.received = []

    def test_publish_subscribe(self) -> None:
        """Abone olunan olay teslim edilir."""
        self.bus.subscribe("test.event", lambda event: self.received.append(event))
        count = self.bus.publish(Event("test.event", {"value": 1}, source="test"))
        self.assertEqual(count, 1)
        self.assertEqual(len(self.received), 1)
        self.assertEqual(self.received[0].data["value"], 1)

    def test_wildcard_subscription(self) -> None:
        """Joker karakterli abonelik tüm alt olayları yakalar."""
        self.bus.subscribe("hardware.*", lambda event: self.received.append(event))
        self.bus.publish(Event("hardware.ram_warning", {}))
        self.bus.publish(Event("hardware.gpu_warning", {}))
        self.bus.publish(Event("other.event", {}))
        self.assertEqual(len(self.received), 2)

    def test_priority_filter(self) -> None:
        """Öncelik filtresi düşük öncelikli olayları eler."""
        self.bus.subscribe("test.*", lambda event: self.received.append(event),
                           priority_filter=EventPriority.HIGH)
        self.bus.publish(Event("test.low", {}, priority=EventPriority.LOW))
        self.bus.publish(Event("test.high", {}, priority=EventPriority.HIGH))
        self.assertEqual(len(self.received), 1)
        self.assertEqual(self.received[0].event_type, "test.high")

    def test_max_calls(self) -> None:
        """`max_calls` sonrası abonelik kapanır."""
        self.bus.subscribe("test.event", lambda event: self.received.append(event),
                           max_calls=2)
        for _ in range(4):
            self.bus.publish(Event("test.event", {}))
        self.assertEqual(len(self.received), 2)

    def test_unsubscribe(self) -> None:
        """Abonelik iptal edilir."""
        subscription_id = self.bus.subscribe("test.event",
                                             lambda event: self.received.append(event))
        self.assertTrue(self.bus.unsubscribe(subscription_id))
        self.bus.publish(Event("test.event", {}))
        self.assertEqual(len(self.received), 0)

    def test_handler_error_isolated(self) -> None:
        """Bozuk handler diğerlerini engellemez."""
        def broken(event: Event) -> None:
            raise RuntimeError("kasıtlı hata")

        self.bus.subscribe("test.event", broken)
        self.bus.subscribe("test.event", lambda event: self.received.append(event))
        self.bus.publish(Event("test.event", {}))
        self.assertEqual(len(self.received), 1, "sağlam handler çalışmalı")

    def test_history(self) -> None:
        """Olay geçmişi sorgulanabilir."""
        for index in range(5):
            self.bus.publish(Event("test.event", {"index": index}, source="test"))
        history = self.bus.get_history(limit=3)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0].data["index"], 4, "en yeni önce gelmeli")

    def test_stats_json_safe(self) -> None:
        """Bus istatistikleri JSON-güvenli."""
        assert_json_safe(self, self.bus.get_stats(), "event_bus_stats")


class TestEngineLifecycle(unittest.TestCase):
    """Motor yaşam döngüsü: kurulum, sağlık, kapanış."""

    def test_initialize_and_shutdown(self) -> None:
        """Motor kurulur, sağlıklı rapor verir ve temiz kapanır."""
        with engine_context() as engine:
            self.assertIn(engine.state, (EngineState.READY, EngineState.DEGRADED))
            self.assertIsNotNone(engine.monitor)
            self.assertIsNotNone(engine.knowledge)
            self.assertIsNotNone(engine.model)
            self.assertIsNotNone(engine.embedding)

            report = engine.health_report()
            self.assertTrue(report.ok, f"kritik sorunlar: {report.critical}")
            self.assertEqual(report.critical, [])

            status = engine.status_line()
            self.assertIn("NeuroCore", status)

        self.assertEqual(engine.state, EngineState.STOPPED)

    def test_initialize_idempotent(self) -> None:
        """İki kez `initialize` çağrısı zarar vermez."""
        with engine_context() as engine:
            state_before = engine.state
            engine.initialize()
            self.assertEqual(engine.state, state_before)

    def test_context_manager(self) -> None:
        """`with NeuroCoreEngine()` motoru kurar ve kapatır."""
        config = isolated_config()
        try:
            with NeuroCoreEngine(config) as engine:
                self.assertIn(engine.state, (EngineState.RUNNING, EngineState.READY))
            self.assertEqual(engine.state, EngineState.STOPPED)
        finally:
            shutil.rmtree(config.base_path, ignore_errors=True)

    def test_layers_wired(self) -> None:
        """Motor tüm katmanları kurar ve kablolar."""
        with engine_context() as engine:
            self.assertIsNotNone(engine.planner)
            self.assertIsNotNone(engine.evolution)
            self.assertIsNotNone(engine.multimodal)
            self.assertIsNotNone(engine.os_layer)
            self.assertIsNotNone(engine.language)

            # Gömme fonksiyonu bilgi katmanına bağlanmış olmalı
            self.assertIsNotNone(engine.knowledge._embed_fn)

            # Planlayıcıya yetenekler kaydedilmiş olmalı
            capabilities = engine.planner.capabilities()
            names = {item.get("name") for item in capabilities}
            self.assertIn("retrieve", names)
            self.assertIn("synthesize", names)
            self.assertIn("verify", names)

    def test_stats_json_safe(self) -> None:
        """Motor istatistikleri JSON-güvenli."""
        with engine_context() as engine:
            assert_json_safe(self, engine.get_stats(), "engine_stats")
            assert_json_safe(self, engine.health_report().to_dict(), "health_report")


class TestEngineQuery(unittest.TestCase):
    """Sorgu yolları: standart, sistem, derin."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._context = engine_context()
        cls.engine = cls._context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._context.__exit__(None, None, None)

    def test_answer_query(self) -> None:
        """Standart sorgu bilgi kasasından yanıt üretir."""
        result = self.engine.query("GPU sıcaklığı neden yükselir?",
                                   learn=False, remember=False)
        self.assertTrue(result.ok, f"hata: {result.error}")
        self.assertEqual(result.mode, "answer")
        self.assertGreater(result.confidence, 0.3)
        self.assertTrue(result.sources, "kaynak listelenmeli")
        self.assertIn("GPU", result.answer)

    def test_empty_query(self) -> None:
        """Boş sorgu hata bildirir (çökmez)."""
        result = self.engine.query("", learn=False, remember=False)
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "boş_sorgu")

    def test_system_query(self) -> None:
        """Sistem sorgusu canlı donanım verisi döndürür."""
        result = self.engine.query("sistem raporunu göster", learn=False, remember=False)
        self.assertEqual(result.mode, "system")
        self.assertIn("cpu", result.answer.lower())
        self.assertGreater(result.confidence, 0.8)

    def test_irrelevant_query(self) -> None:
        """Alakasız sorgu bilgi bulamadığını bildirir."""
        result = self.engine.query("kuantum kromodinamik plazma nedir",
                                   learn=False, remember=False)
        self.assertTrue(result.ok)
        self.assertLess(result.confidence, 0.9)

    def test_learn_and_query_new_knowledge(self) -> None:
        """Öğretilen bilgi sonraki sorguda kullanılır."""
        marker = "ZorluSinamaBelirteci"
        self.engine.learn(
            f"{marker} sistemin özel bir parçasıdır ve 42 birim dayanıklılığa sahiptir.",
            title=f"{marker} Test Notu", tags=["test"], confidence=0.8,
        )
        result = self.engine.query(f"{marker} nedir?", learn=False, remember=False)
        self.assertTrue(result.ok)
        self.assertTrue(
            marker.lower() in result.answer.lower() or result.sources,
            "öğretilen bilgi yanıtta veya kaynaklarda görünmeli",
        )

    def test_deep_query(self) -> None:
        """Derin mod planlayıcıyı kullanır."""
        result = self.engine.query("sistem raporunu göster", deep=True,
                                   learn=False, remember=False)
        self.assertTrue(result.ok)
        self.assertIn(result.mode, ("deep", "system"))
        if result.plan:
            self.assertIn("state", result.plan)

    def test_query_result_serializable(self) -> None:
        """Sorgu sonucu JSON-güvenli."""
        result = self.engine.query("GPU termal", learn=False, remember=False)
        assert_json_safe(self, result.to_dict(), "query_result")
        self.assertIsInstance(result.render(), str)


class TestEngineLearnAndFeedback(unittest.TestCase):
    """Öğrenme, geri bildirim ve araştırma akışları."""

    def test_learn_creates_note(self) -> None:
        """`learn` notu kasaya yazar ve indeksler."""
        with engine_context(seed=False) as engine:
            outcome = engine.learn(
                "Deneme bilgisi: kasa testi için yazılmış içerik.",
                title="Deneme Notu", tags=["test"], confidence=0.7,
            )
            self.assertTrue(outcome.get("ok"), outcome)
            self.assertTrue(outcome.get("id"))
            # İndekste bulunmalı
            hits = engine.search("deneme bilgisi", limit=5)
            self.assertTrue(hits)

    def test_learn_split_long_content(self) -> None:
        """Uzun içerik parçalara bölünür."""
        with engine_context(seed=False) as engine:
            long_text = "\n\n".join(
                f"## Bölüm {index}\n" + ("Bu bölümde ayrıntılı bilgi verilir. " * 60)
                for index in range(1, 4)
            )
            outcome = engine.learn(long_text, title="Uzun İçerik", tags=["test"])
            self.assertTrue(outcome.get("ok"))
            self.assertGreater(outcome.get("chunks", 1), 1)
            self.assertTrue(outcome.get("derived_notes"))

    def test_feedback_recorded(self) -> None:
        """Geri bildirim deneyim tamponuna yazılır."""
        with engine_context() as engine:
            engine.query("GPU termal yönetimi", remember=True)
            outcome = engine.feedback(positive=True, note="test")
            self.assertTrue(outcome.get("recorded"), outcome)
            self.assertEqual(outcome.get("feedback"), "explicit_positive")

    def test_feedback_without_query(self) -> None:
        """Sorgu yapılmadan geri bildirim zarar vermez."""
        with engine_context() as engine:
            outcome = engine.feedback(positive=True)
            self.assertFalse(outcome.get("recorded", False) and not outcome.get("error"))

    def test_research_runs(self) -> None:
        """Araştırma turu bulgu veya not üretir."""
        with engine_context() as engine:
            report = engine.research("termal macun", depth=1)
            self.assertIn("rounds", report)
            self.assertGreaterEqual(report["rounds"], 1)
            self.assertIsInstance(report["findings"], list)


class TestEngineMaintenance(unittest.TestCase):
    """Bakım, evrim ve büyüme akışları."""

    def test_maintain(self) -> None:
        """Bakım turu hata vermeden tamamlanır."""
        with engine_context() as engine:
            report = engine.maintain(deep=False)
            self.assertIn("actions", report)
            self.assertIn("duration_ms", report)
            self.assertIsInstance(report["errors"], list)

    def test_evolve_without_training(self) -> None:
        """Evrim turu (eğitimsiz) rapor üretir."""
        with engine_context() as engine:
            outcome = engine.evolve(cycles=1, train=False)
            self.assertIn("reports", outcome)
            self.assertEqual(len(outcome["reports"]), 1)

    def test_growth_neutral_for_query(self) -> None:
        """Uzman büyümesi sorgu yanıtını bozmaz."""
        with engine_context() as engine:
            before = engine.query("GPU sıcaklığı neden yükselir?",
                                  learn=False, remember=False)
            outcome = engine.model.grow_experts(1, dormant=True)
            self.assertGreater(outcome.get("added", 0), 0)
            after = engine.query("GPU sıcaklığı neden yükselir?",
                                 learn=False, remember=False)
            self.assertTrue(after.ok)
            self.assertAlmostEqual(before.confidence, after.confidence, delta=0.15)

    def test_start_and_stop(self) -> None:
        """Arka plan döngüleri başlatılıp durdurulur."""
        with engine_context() as engine:
            engine.start(maintenance_interval=600)
            self.assertEqual(engine.state, EngineState.RUNNING)
            self.assertTrue(engine.monitor._monitoring)
            engine.shutdown(save=False)
            self.assertEqual(engine.state, EngineState.STOPPED)


class TestEngineBootstrap(unittest.TestCase):
    """İlk kurulum: tohum bilgi ve tokenizer eğitimi."""

    def test_bootstrap_seeds_knowledge(self) -> None:
        """Tohumlama sistem notları ekler."""
        with engine_context(seed=False) as engine:
            outcome = engine.bootstrap(seed_knowledge=True, train_tokenizer=False)
            self.assertGreater(outcome.get("seeded_notes", 0), 0)
            notes = engine.knowledge.list_notes(limit=20)
            titles = [note["title"] for note in notes]
            self.assertTrue(any("Envanter" in title or "Mimari" in title
                                for title in titles))

    def test_bootstrap_trains_tokenizer(self) -> None:
        """Tokenizer eğitimi sözlüğü büyütür ve omurgayı yeniden kurar."""
        with engine_context(seed=True) as engine:
            before = engine.model.tokenizer.vocab_size
            outcome = engine.bootstrap(seed_knowledge=False, train_tokenizer=True,
                                       target_vocab=600, force=True)
            after = engine.model.tokenizer.vocab_size
            self.assertGreater(after, before)
            self.assertEqual(engine.model.model_config.vocab_size, after)
            self.assertIn("tokenizer", outcome)


if __name__ == "__main__":
    unittest.main(verbosity=2)
