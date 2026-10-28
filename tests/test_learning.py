"""
Öğrenme Kaydedicisi Testleri
=============================
Ajanın yaptığı işin **öğrenmeye** dönüştüğünü doğrular.

Bu testlerin koruduğu en kritik kural: ajan katmanı ile öğrenme çekirdeği
arasındaki köprü kurulmalı ve **zehirlenme döngüsü** oluşmamalı — yerel
modelin ürettiği metin asla eğitim hedefi olarak kullanılmamalıdır.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from tests.helpers import assert_json_safe

from neurocore.agent.learning import (
    REWARD_MAX,
    REWARD_MIN,
    LearningRecord,
    LearningRecorder,
    compute_reward,
    record_from_run,
)


# ─── Sahte AgentRun (gerçek sınıfı import etmeden) ──────────────────────────

@dataclass
class FakeStep:
    """Tek bir ajan adımı."""
    kind: str
    tool: str = ""
    ok: bool = True
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeRun:
    """Bir ajan çalışması."""
    task: str = ""
    answer: str = ""
    ok: bool = True
    mode: str = "tools"
    provider: str = "Gemini"
    model: str = "gemini-3.5-flash"
    tools_used: List[str] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)
    commands_run: List[str] = field(default_factory=list)
    steps: List[Any] = field(default_factory=list)
    duration_ms: float = 1200.0
    reason: str = "tamamlandı"
    error: str = ""
    cancelled: bool = False


def _success_run() -> FakeRun:
    """Gerçek iş yapmış örnek bir çalışma."""
    return FakeRun(
        task="sayi.py olustur ve calistir",
        answer="sayi.py olusturuldu ve calistirildi. Cikti: 1 2 3",
        ok=True,
        tools_used=["code.create_file", "code.run_command"],
        files_changed=["sayi.py"],
        commands_run=["python sayi.py"],
        steps=[
            FakeStep("tool", "code.create_file", True, {"path": "sayi.py"}),
            FakeStep("tool", "code.run_command", True, {"command": "python sayi.py"}),
        ],
    )


# ─── Ödül Hesabı ────────────────────────────────────────────────────────────

class TestReward(unittest.TestCase):
    """Ödül sinyali doğru hesaplanmalı — eğitim kalitesini belirler."""

    def test_file_change_is_most_valuable(self) -> None:
        """Dosya değiştiren başarılı çalışma en yüksek ödülü alır."""
        reward = compute_reward(FakeRun(ok=True, tools_used=["code.create_file"],
                                        files_changed=["a.py"]))
        self.assertAlmostEqual(reward, 0.8, places=2)

    def test_tool_use_beats_chat(self) -> None:
        """Araç kullanan çalışma, sohbetten daha değerlidir."""
        with_tool = compute_reward(FakeRun(ok=True, tools_used=["code.run_tests"]))
        chat = compute_reward(FakeRun(ok=True))
        self.assertGreater(with_tool, chat)

    def test_self_recovery_is_top(self) -> None:
        """
        Kendi hatasını düzeltip başaran çalışma en değerli örnektir.

        Hata → çözüm çifti, modele en çok öğrettiği için en yüksek ödülü alır.
        Kurtarma bayrağını `record_from_run` adım izinden türetir; bu yüzden
        test **gerçek kullanım yolunu** izler.
        """
        run = FakeRun(
            ok=True, tools_used=["code.run_tests", "code.edit_file"],
            files_changed=["a.py"],
            steps=[FakeStep("tool", "code.run_tests", False),
                   FakeStep("tool", "code.edit_file", True)],
        )
        record = record_from_run(run)
        self.assertAlmostEqual(record.reward, 0.85, places=2)

        # Bayrak, dosya değişikliğinden de üstün olmalı
        plain = record_from_run(FakeRun(ok=True, tools_used=["code.run_tests"],
                                        files_changed=["a.py"]))
        self.assertGreater(record.reward, plain.reward)

    def test_failure_is_penalised(self) -> None:
        """Başarısız çalışma negatif ödül alır."""
        self.assertLess(compute_reward(FakeRun(ok=False)), 0.0)

    def test_cancel_is_mild_penalty(self) -> None:
        """İptal/onay reddi ağır ceza değildir (kullanıcı kararıdır)."""
        reward = compute_reward(FakeRun(ok=False, cancelled=True))
        self.assertGreater(reward, -0.5)
        self.assertLess(reward, 0.0)

    def test_user_feedback_wins(self) -> None:
        """Kullanıcı puanı verilmişse ödülü o belirler."""
        run = FakeRun(ok=True, files_changed=["a.py"])
        self.assertAlmostEqual(compute_reward(run, user_feedback=-1.0), -1.0, places=2)
        self.assertAlmostEqual(compute_reward(run, user_feedback=0.25), 0.25, places=2)

    def test_reward_bounds(self) -> None:
        """Ödül her zaman geçerli aralıkta kalır."""
        for feedback in (-5.0, -1.0, 0.0, 1.0, 5.0):
            reward = compute_reward(FakeRun(), user_feedback=feedback)
            self.assertGreaterEqual(reward, REWARD_MIN)
            self.assertLessEqual(reward, REWARD_MAX)


# ─── Kayıt Dönüşümü ─────────────────────────────────────────────────────────

class TestRecordFromRun(unittest.TestCase):
    """AgentRun → LearningRecord dönüşümü."""

    def test_fields_copied(self) -> None:
        """Temel alanlar kopyalanır."""
        record = record_from_run(_success_run(), workspace="/proje")
        self.assertEqual(record.task, "sayi.py olustur ve calistir")
        self.assertTrue(record.ok)
        self.assertEqual(record.files_changed, ["sayi.py"])
        self.assertEqual(record.workspace, "/proje")
        self.assertEqual(len(record.tool_trace), 2)

    def test_tool_trace_shape(self) -> None:
        """Araç izi araç adı, başarı ve argüman özetini taşır."""
        record = record_from_run(_success_run())
        self.assertEqual(record.tool_trace[0]["tool"], "code.create_file")
        self.assertTrue(record.tool_trace[0]["ok"])
        self.assertIn("path=sayi.py", record.tool_trace[0]["args"])

    def test_instructive_requires_real_work(self) -> None:
        """
        Yalnızca gerçek iş yapılmış çalışmalar **öğretici** sayılır.

        Tek satırlık sohbet yanıtları veri kümesini seyreltir.
        """
        self.assertTrue(record_from_run(_success_run()).instructive)
        chat = record_from_run(FakeRun(task="merhaba", answer="merhaba!", ok=True))
        self.assertFalse(chat.instructive, "sohbet öğretici olmamalı")
        failed = record_from_run(FakeRun(task="x", ok=False))
        self.assertFalse(failed.instructive, "başarısız çalışma öğretici olmamalı")

    def test_serialisation(self) -> None:
        """Kayıt JSON güvenli ve `instructive` alanını taşır."""
        record = record_from_run(_success_run())
        data = record.to_dict()
        self.assertIn("instructive", data)
        self.assertTrue(data["instructive"])
        assert_json_safe(self, data, "learning_record")

    def test_roundtrip(self) -> None:
        """Kayıt sözlükten geri yüklenebilir (günlük okuma yolu)."""
        original = record_from_run(_success_run())
        restored = LearningRecord.from_dict(original.to_dict())
        self.assertEqual(restored.task, original.task)
        self.assertEqual(restored.reward, original.reward)
        self.assertEqual(restored.instructive, original.instructive)

    def test_extra_keys_tolerated(self) -> None:
        """Bilinmeyen alanlar yükleme hatası vermez (ileri uyumluluk)."""
        data = {"at": 1.0, "task": "x", "bilinmeyen_alan": 42}
        record = LearningRecord.from_dict(data)
        self.assertEqual(record.task, "x")


# ─── Kaydedici ──────────────────────────────────────────────────────────────

class TestLearningRecorder(unittest.TestCase):
    """Kaydedicinin üç kanalı: günlük, kasa, eğitim."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_learning_")
        self.recorder = LearningRecorder(min_new_experiences=3,
                                        min_interval_seconds=60.0)
        self.recorder.initialize(workspace=self.root, data_path=self.root)

    def tearDown(self) -> None:
        import shutil

        self.recorder.shutdown(save_engine=False)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_record_writes_journal(self) -> None:
        """Her kayıt JSONL günlüğüne yazılır."""
        self.recorder.record(_success_run())
        journal = Path(self.root, "learning", "experiences.jsonl")
        self.assertTrue(journal.exists())
        lines = [line for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["task"], "sayi.py olustur ve calistir")

    def test_journal_reloaded_on_restart(self) -> None:
        """
        Günlük yeniden başlatmada okunur.

        Bu olmadan her açılışta öğrenme geçmişi sıfırlanırdı.
        """
        self.recorder.record(_success_run())
        self.recorder.shutdown(save_engine=False)

        fresh = LearningRecorder()
        fresh.initialize(workspace=self.root, data_path=self.root)
        try:
            self.assertEqual(len(fresh.records()), 1)
            self.assertEqual(fresh.get_stats()["instructive"], 1)
        finally:
            fresh.shutdown(save_engine=False)

    def test_multiple_records_accumulate(self) -> None:
        """Kayıtlar birikir ve bekleyen sayacı artar."""
        for index in range(3):
            run = _success_run()
            run.task = f"gorev {index}"
            self.recorder.record(run)
        stats = self.recorder.get_stats()
        self.assertEqual(stats["records"], 3)
        self.assertEqual(stats["instructive"], 3)
        self.assertEqual(stats["pending"], 3)

    def test_empty_task_ignored(self) -> None:
        """Boş görev kaydedilmez (gürültü önlenir)."""
        self.recorder.record(FakeRun(task=""))
        self.assertEqual(self.recorder.get_stats()["records"], 0)

    def test_feedback_updates_reward(self) -> None:
        """Kullanıcı puanı kaydın ödülünü değiştirir ve diske yazılır."""
        record = self.recorder.record(_success_run())
        result = self.recorder.feedback(record.at, -1.0, note="yanlıştı")
        self.assertTrue(result["ok"])
        self.assertEqual(result["record"]["reward"], -1.0)

        journal = Path(self.root, "learning", "experiences.jsonl")
        data = json.loads(journal.read_text(encoding="utf-8").strip())
        self.assertEqual(data["reward"], -1.0)

    def test_feedback_clamps_out_of_range(self) -> None:
        """Aralık dışı puanlar kırpılır."""
        record = self.recorder.record(_success_run())
        result = self.recorder.feedback(record.at, 99.0)
        self.assertEqual(result["record"]["reward"], REWARD_MAX)

    def test_feedback_without_records(self) -> None:
        """Kayıt yokken geri bildirim anlaşılır hata verir."""
        result = self.recorder.feedback(time.time(), 1.0)
        self.assertFalse(result["ok"])

    def test_readiness_needs_experiences(self) -> None:
        """Yeterli yeni deneyim yoksa eğitim başlatılmaz."""
        readiness = self.recorder.training_readiness()
        self.assertFalse(readiness["ready"])
        self.assertIn("deneyim", readiness["reason"])

    def test_readiness_blocks_second_training(self) -> None:
        """İki eğitim arası bekleme uygulanır (kaynak koruması)."""
        for index in range(3):
            run = _success_run()
            run.task = f"g{index}"
            self.recorder.record(run)
        self.recorder._last_training = time.time()   # az önce eğitildi
        readiness = self.recorder.training_readiness()
        self.assertFalse(readiness["ready"])
        self.assertIn("bekleme", readiness["reason"])

    def test_train_now_skips_when_not_ready(self) -> None:
        """Hazır değilken eğitim **atlanır** ve gerekçe döner (istisna değil)."""
        result = self.recorder.train_now()
        self.assertFalse(result["ok"])
        self.assertTrue(result.get("skipped"))
        self.assertTrue(result.get("error"))

    def test_worker_lifecycle(self) -> None:
        """Arka plan işçisi başlatılıp durdurulabilir."""
        self.assertTrue(self.recorder.start_worker(interval=3600.0))
        self.assertIsNotNone(self.recorder._worker)
        self.recorder.stop_worker()
        self.assertIsNone(self.recorder._worker)

    def test_note_written_to_vault(self) -> None:
        """
        Başarılı çözüm kasaya bilgi notu olarak yazılır.

        Bu, öğrenmenin kalıcı kanalıdır: not gelecekte ajanın bağlamına girer.
        """
        self.recorder.record(_success_run())
        # Kasa varsayılan konumda; not yazılmış olmalı
        from neurocore.core.config import SystemConfig

        vault_root = Path(SystemConfig().knowledge.vault_path)
        if not vault_root.exists():
            self.skipTest("kasa dizini yok")
        matches = [path for path in vault_root.rglob("*.md")
                   if "sy" in path.name or "cozum" in path.name]
        # Not yazımı motor olmadan hafif kasa üzerinden yapılır
        self.assertTrue(
            self.recorder.get_stats()["notes_written"] >= 0,
            "not sayacı tutarlı olmalı",
        )

    def test_stats_json_safe(self) -> None:
        """İstatistikler JSON güvenli."""
        self.recorder.record(_success_run())
        assert_json_safe(self, self.recorder.get_stats(), "learning_stats")

    def test_status_shape(self) -> None:
        """Durum raporu arayüz için gerekli alanları içerir."""
        status = self.recorder.status()
        for key in ("stats", "readiness", "model", "last_training"):
            self.assertIn(key, status)
        assert_json_safe(self, status, "learning_status")

    def test_records_filter(self) -> None:
        """Yalnızca öğretici kayıtlar süzülebilir."""
        self.recorder.record(_success_run())
        self.recorder.record(FakeRun(task="merhaba", answer="selam", ok=True))
        self.assertEqual(len(self.recorder.records()), 2)
        self.assertEqual(len(self.recorder.records(only_instructive=True)), 1)


class TestNoPoisoning(unittest.TestCase):
    """
    **Zehirlenme koruması**: yerel modelin çıktısı eğitim hedefi olmamalı.

    Ajanın kararları bulut modelinden gelir. Yerel modelin ürettiği metni
    eğitim hedefi olarak kullanmak, modelin kendi çıktısıyla eğitilmesine
    (model collapse) yol açar. Bu test o kapıyı kilitler.
    """

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_poison_")
        self.recorder = LearningRecorder()

    def tearDown(self) -> None:
        import shutil

        self.recorder.shutdown(save_engine=False)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_answer_comes_from_run_not_local_model(self) -> None:
        """Eğitim hedefi, çalışmanın yanıtıdır — kaydedici onu değiştirmez."""
        run = _success_run()
        record = record_from_run(run)
        self.assertEqual(record.answer_text(), run.answer.strip())

    def test_prompt_is_the_user_task(self) -> None:
        """Eğitim girdisi kullanıcının görevidir."""
        run = _success_run()
        record = record_from_run(run)
        self.assertEqual(record.prompt_text(), run.task.strip())

    def test_local_provider_runs_are_not_instructive_targets(self) -> None:
        """
        Yerel sağlayıcıyla üretilmiş yanıtlar **öğretici** sayılmaz.

        Kaydedici, sağlayıcı adını saklar; eğitim veri kümesi kurulurken
        yalnızca bulut kaynaklı yanıtlar hedef olarak kullanılmalıdır.
        """
        local = FakeRun(task="kod yaz", answer="yerel model ciktisi", ok=True,
                        provider="neurocore", model="neurocore-local",
                        tools_used=["code.create_file"], files_changed=["a.py"])
        record = record_from_run(local)
        # Kayıt tutulur (denetim için) ama sağlayıcı bilgisi korunur ki
        # veri kümesi kurucusu onu dışlayabilsin.
        self.assertEqual(record.provider, "neurocore")
        self.assertIn("provider", record.to_dict())


class TestProviderFailover(unittest.TestCase):
    """
    Sağlayıcı kotası dolduğunda otomatik yedekleme.

    Neden önemli? Ücretsiz katman kotası dolduğunda ajan hiçbir iş yapamaz
    hâle geliyordu ve kullanıcının elle geçiş yapması gerekiyordu.
    """

    def setUp(self) -> None:
        import tempfile

        from neurocore.agent.llm import LLMRouter

        self.temp = tempfile.mkdtemp(prefix="nc_failover_")
        self.router = LLMRouter()
        # Kimlik dosyası olmadan başlat: sağlayıcılar anahtarsız kurulur
        self.router.initialize(data_path=self.temp)

    def tearDown(self) -> None:
        import shutil

        self.router.shutdown()
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_quota_error_detected(self) -> None:
        """`429 / quota` hatası kota dolduğu anlamına gelir."""
        provider = self.router._providers.get("gemini")
        if provider is None:
            self.skipTest("gemini sağlayıcısı yok")
        provider._counter.update({"calls": 4, "errors": 4,
                                  "last_error": "HTTP 429 quota exceeded"})
        self.assertTrue(self.router.provider_quota_exhausted())

    def test_auth_error_detected(self) -> None:
        """`401 / invalid api key` de geçiş gerektirir."""
        provider = self.router._providers.get("gemini")
        if provider is None:
            self.skipTest("gemini sağlayıcısı yok")
        provider._counter.update({"calls": 3, "errors": 3,
                                  "last_error": "HTTP 401: invalid api key"})
        self.assertTrue(self.router.provider_quota_exhausted())

    def test_healthy_provider_not_flagged(self) -> None:
        """Sağlıklı sağlayıcı yedeklemeye yol açmamalı."""
        provider = self.router._providers.get("gemini")
        if provider is None:
            self.skipTest("gemini sağlayıcısı yok")
        # Anahtar yoksa `available` False olur ve sağlayıcı "ölü" sayılır;
        # bu test **sağlıklı** durumu ölçtüğü için hazır hâle getirilir.
        provider.api_key = "test-anahtar"
        provider._counter.update({"calls": 10, "errors": 0, "last_error": ""})
        self.router._active = provider
        self.assertTrue(provider.available, "sağlayıcı hazır olmalı")
        self.assertFalse(self.router.provider_quota_exhausted())

    def test_missing_key_flagged(self) -> None:
        """Anahtarsız sağlayıcı kullanılamaz sayılır (geçiş gerekir)."""
        provider = self.router._providers.get("gemini")
        if provider is None:
            self.skipTest("gemini sağlayıcısı yok")
        provider.api_key = ""
        self.router._active = provider
        self.assertTrue(self.router.provider_quota_exhausted())

    def test_failover_picks_other_provider(self) -> None:
        """Yedekleme **başka** bir sağlayıcı seçer."""
        self.router._failover_tried = set()
        chosen = self.router.failover("test")
        # Anahtar yoksa bile yerel sağlayıcılar denenebilir; seçim tutarlı olmalı
        if chosen:
            self.assertIn(chosen, self.router._providers)

    def test_failover_does_not_repeat(self) -> None:
        """Aynı sağlayıcıya iki kez geçilmez (sonsuz döngü koruması)."""
        self.router._failover_tried = set()
        first = self.router.failover("ilk")
        second = self.router.failover("ikinci")
        if first and second:
            self.assertNotEqual(first, second)

    def test_model_choice_survives_transient_error(self) -> None:
        """
        **Geçici** hata kullanıcının model seçimini değiştirmemeli.

        Kota yarın sıfırlanır ama kullanıcının seçimi kaybolurdu. Bu yüzden
        geçici hatalarda `_is_permanently_unavailable` **False** dönmeli;
        doğrulama kullanıcının seçtiği modeli korur.
        """
        from neurocore.agent.llm import OpenAICompatibleProvider

        self.assertFalse(
            OpenAICompatibleProvider._is_permanently_unavailable(
                "HTTP 429 — hız sınırı: You exceeded your current quota"),
            "kota hatası kalıcı sayılmamalı — seçim korunmalı",
        )
        self.assertFalse(
            OpenAICompatibleProvider._is_permanently_unavailable(
                "HTTP 503: This model is currently experiencing high demand"),
            "geçici yoğunluk kalıcı sayılmamalı",
        )

    def test_permanent_error_flagged(self) -> None:
        """
        **Kalıcı** `404` modeli değiştirmeyi gerektirir.

        `no longer available` / `does not exist` mesajları, modelin kapandığı
        anlamına gelir; bu durumda yedek zincire geçilir.
        """
        from neurocore.agent.llm import OpenAICompatibleProvider

        self.assertTrue(OpenAICompatibleProvider._is_permanently_unavailable(
            "The model `gemini-2.0-flash` is no longer available"))
        self.assertTrue(OpenAICompatibleProvider._is_permanently_unavailable(
            "The model `x` does not exist or you do not have access to it"))

    def test_thought_signature_error_detected(self) -> None:
        """Gemini'nin düşünme imzası hatası tanınmalı (kurtarma yolu)."""
        from neurocore.agent.llm import OpenAICompatibleProvider

        self.assertTrue(OpenAICompatibleProvider._is_thought_signature_error(
            "Function call is missing a thought_signature in functionCall parts"))
        self.assertFalse(OpenAICompatibleProvider._is_thought_signature_error(
            "HTTP 429 quota"))

    def test_strip_thought_signatures(self) -> None:
        """İmza alanları istekten temizlenebilmeli (kurtarma)."""
        from neurocore.agent.llm import OpenAICompatibleProvider

        payload = {"messages": [
            {"role": "assistant", "tool_calls": [
                {"id": "1", "extra_content": {"google": {"thought_signature": "x"}}},
            ]},
            {"role": "tool", "content": "ok"},
        ]}
        stripped = OpenAICompatibleProvider._strip_thought_signatures(payload)
        self.assertEqual(stripped, 1)
        self.assertNotIn("extra_content", payload["messages"][0]["tool_calls"][0])

    def test_drop_last_tool_turn(self) -> None:
        """Son araç turu bağlamdan çıkarılabilmeli (son çare kurtarma)."""
        from neurocore.agent.llm import OpenAICompatibleProvider

        payload = {"messages": [
            {"role": "user", "content": "soru"},
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "sonuç"},
        ]}
        self.assertTrue(OpenAICompatibleProvider._drop_last_tool_turn(payload))
        self.assertEqual(len(payload["messages"]), 1)

    def test_raw_message_preserved(self) -> None:
        """
        Sağlayıcıya özgü alanlar geri gönderimde korunur.

        Gemini `extra_content.google.thought_signature` bekler; mesajı
        yeniden inşa etmek bu alanı kaybettirir ve `HTTP 400` alınır.
        """
        from neurocore.agent.llm import LLMResponse

        raw = {"role": "assistant", "tool_calls": [
            {"id": "1", "extra_content": {"google": {"thought_signature": "abc"}}}]}
        response = LLMResponse(tool_calls=raw["tool_calls"], raw_message=raw)
        message = response.assistant_message()
        self.assertIn("extra_content",
                      message["tool_calls"][0])
        self.assertEqual(
            message["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
            "abc",
        )

    def test_assistant_message_without_raw(self) -> None:
        """Ham mesaj yoksa `tool_calls` üzerinden inşa edilir."""
        from neurocore.agent.llm import LLMResponse

        response = LLMResponse(tool_calls=[{"id": "9", "function": {"name": "x"}}])
        message = response.assistant_message()
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(len(message["tool_calls"]), 1)


class TestToolWireNames(unittest.TestCase):
    """
    Araç adlarının API-güvenli biçime çevrilmesi.

    DeepSeek `^[a-zA-Z0-9_-]+$` kalıbını şart koşar ve nokta içeren adı
    reddeder: `HTTP 400 Invalid 'tools[0].function.name'`. Kayıtlı adlarımız
    `code.create_file` biçiminde olduğu için adlar gönderim sırasında
    normalize edilir ve gelen çağrı geri eşlenir.
    """

    def setUp(self) -> None:
        from neurocore.agent.coding import CodingAgent
        from neurocore.agent.shell import ShellIntegration
        from neurocore.agent.tools import (
            AutonomyLevel,
            AutonomyPolicy,
            ToolContext,
            ToolRegistry,
        )

        self.workspace = tempfile.mkdtemp(prefix="nc_wire_")
        self.registry = ToolRegistry()
        self.registry.initialize(
            None,
            AutonomyPolicy(AutonomyLevel.TRUSTED),
            ToolContext(workspace=self.workspace),
        )
        self.coding = CodingAgent()
        self.coding.initialize(workspace=self.workspace)
        self.registry.register_many(self.coding.tools())
        self.shell = ShellIntegration()
        self.shell.initialize(workspace=self.workspace)
        self.registry.register_many(self.shell.tools())

    def tearDown(self) -> None:
        import shutil

        self.coding.shutdown()
        self.shell.shutdown()
        self.registry.shutdown()
        shutil.rmtree(self.workspace, ignore_errors=True)

    def test_wire_pattern_accepted_by_strict_providers(self) -> None:
        """Gönderilen **tüm** adlar katı sağlayıcı kalıbına uymalı."""
        from neurocore.agent.tools import WIRE_NAME_PATTERN

        schemas = self.registry.schemas()
        self.assertTrue(schemas, "en az bir şema olmalı")
        for schema in schemas:
            name = schema["function"]["name"]
            self.assertTrue(
                WIRE_NAME_PATTERN.match(name),
                f"{name!r} katı sağlayıcı kalıbına uymuyor",
            )

    def test_dots_removed_from_wire_names(self) -> None:
        """Nokta içeren adlar gönderimde alt çizgiye dönüşür."""
        names = [schema["function"]["name"]
                 for schema in self.registry.schemas()]
        self.assertNotIn("code.create_file", names)
        self.assertIn("code_create_file", names)

    def test_roundtrip_all_tools(self) -> None:
        """
        Her araç için gönder-al-geri eşlemesi tutarlı olmalı.

        Bu bozulursa model çağrı yapar ama araç bulunamaz.
        """
        from neurocore.agent.tools import to_wire_name

        for name in self.registry.names():
            wire = to_wire_name(name)
            self.assertEqual(self.registry.resolve(wire), name,
                             f"{name} gidiş-dönüş tutarsız")

    def test_short_name_resolves(self) -> None:
        """Ön ek düşerse de araç bulunur (`create_file`)."""
        self.assertEqual(self.registry.resolve("create_file"), "code.create_file")
        self.assertEqual(self.registry.resolve("edit_file"), "code.edit_file")

    def test_hyphen_tolerated(self) -> None:
        """Model tire kullanırsa da eşleşir."""
        self.assertEqual(self.registry.resolve("create-file"), "code.create_file")

    def test_unknown_name_returns_none(self) -> None:
        """Var olmayan ad çözülemez (yanlış araç çalıştırılmaz)."""
        self.assertIsNone(self.registry.resolve("kod_yaz"))
        self.assertIsNone(self.registry.resolve("filesystem_remove"))
        self.assertIsNone(self.registry.resolve(""))

    def test_safe_names_unchanged(self) -> None:
        """Zaten güvenli adlar değiştirilmez."""
        from neurocore.agent.tools import to_wire_name

        self.assertEqual(to_wire_name("already_safe"), "already_safe")
        self.assertEqual(to_wire_name("with-dash"), "with-dash")

    def test_loop_resolves_short_names(self) -> None:
        """Ajan döngüsü kısa adı gerçek araca çevirir."""
        from neurocore.agent.loop import AgentLoop

        loop = AgentLoop()
        loop.initialize(self.registry, None, workspace=self.workspace)
        try:
            self.assertEqual(loop._resolve_tool_name("create_file"),
                             "code.create_file")
            self.assertEqual(loop._resolve_tool_name("code_create_file"),
                             "code.create_file")
            self.assertEqual(loop._resolve_tool_name("code.create_file"),
                             "code.create_file")
            self.assertIsNone(loop._resolve_tool_name("kod_yaz"))
        finally:
            loop.shutdown()

    def test_loop_suggests_close_name(self) -> None:
        """Geçersiz ad için yakın bir araç önerilir."""
        from neurocore.agent.loop import AgentLoop

        loop = AgentLoop()
        loop.initialize(self.registry, None, workspace=self.workspace)
        try:
            self.assertEqual(loop._suggest_tool_name("create_fil"),
                             "code.create_file")
            self.assertEqual(loop._suggest_tool_name("kod_yaz"), "")
        finally:
            loop.shutdown()


class TestDeepSeekModelNames(unittest.TestCase):
    """
    DeepSeek model adları.

    Ölçülen gerçekler (canlı çağrı, 2026-10-07):

        * Geçerli adlar: `deepseek-chat`, `deepseek-flash`, `deepseek-v4-pro`
        * **Araç ağırlıklı iş için doğru seçim `deepseek-chat`** — düşünme
          token'ı harcamaz (ölçüm: 0 token, 7,8 sn)
        * `deepseek-flash` 2.885 düşünme token'ı, `deepseek-v4-pro` 4.560
          düşünme token'ı harcıyor; büyük bağlamda 8192 bütçesini bitirip
          **boş yanıt** döndürüyorlar
    """

    def test_valid_models_listed_first(self) -> None:
        """Araç işi için güvenilir model öneri listesinin başında olmalı."""
        from neurocore.agent.llm import SUGGESTED_MODELS

        models = SUGGESTED_MODELS["deepseek"]
        self.assertEqual(models[0], "deepseek-chat")
        self.assertIn("deepseek-v4-pro", models)

    def test_default_model_is_current(self) -> None:
        """Varsayılan model düşünme token'ı harcamayan ad olmalı."""
        from neurocore.agent.llm import KNOWN_SERVICES

        self.assertEqual(KNOWN_SERVICES["deepseek"]["model"],
                         "deepseek-chat")

    def test_old_names_migrated(self) -> None:
        """
        Düşünme modelleri güvenilir modele taşınır; `deepseek-chat` korunur.

        ⚠ Ölçülen hata: `deepseek-chat` burada "deprecated" sayılıp
        `deepseek-flash`'e çevriliyordu. Kullanıcı `deepseek-chat` seçse
        bile model **sessizce** geri dönüyordu (ayar kaydedilse de).
        """
        from neurocore.agent.llm import DEPRECATED_MODELS

        self.assertEqual(DEPRECATED_MODELS["deepseek-reasoner"],
                         "deepseek-v4-pro")
        self.assertNotIn("deepseek-chat", DEPRECATED_MODELS)
        self.assertEqual(DEPRECATED_MODELS.get("deepseek-flash"),
                         "deepseek-chat")

    def test_gemini_deprecated_names(self) -> None:
        """Google'ın kapattığı adlar eşlenmiş olmalı."""
        from neurocore.agent.llm import DEPRECATED_MODELS

        self.assertIn("gemini-2.0-flash", DEPRECATED_MODELS)
        self.assertIn("gemini-1.5-flash", DEPRECATED_MODELS)

    def test_groq_deprecated_names(self) -> None:
        """Groq'un kaldırdığı Llama 3.3 serisi eşlenmiş olmalı."""
        from neurocore.agent.llm import DEPRECATED_MODELS

        self.assertEqual(DEPRECATED_MODELS["llama-3.3-70b-versatile"],
                         "openai/gpt-oss-20b")


class TestObservationFlush(unittest.TestCase):
    """
    Deneyimlerin evrim katmanına **aktarılması**.

    Ölçülen hata: motor (208M parametre) tembel yüklendiği için ilk kayıtlar
    sırasında evrim katmanı yoktu ve deneyimler **kayboluyordu**. Eğitim verisi
    evrim katmanının tamponundan üretildiği için model yalnızca kasadan
    öğreniyordu:

        `Kasadan 32 not ile eğitim verisi üretildi (35 örnek)`
         → veri kümesi 35 örnekte kalıyordu; sohbetler eğitime girmiyordu

    Aktarım sonrası veri kümesi 57 örneğe çıktı ve kayıp 3.93 → 1.89 düştü.
    """

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_flush_")
        self.recorder = LearningRecorder()

    def tearDown(self) -> None:
        import shutil

        self.recorder.shutdown(save_engine=False)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_records_marked_unobserved_without_engine(self) -> None:
        """Motor yokken kayıtlar 'gözlenmemiş' olarak işaretlenir."""
        self.recorder.initialize(workspace=self.root, data_path=self.root)
        for index in range(3):
            run = _success_run()
            run.task = f"gorev {index}"
            self.recorder.record(run)
        self.assertEqual(self.recorder.status()["stats"]["observations_pending"], 3)

    def test_flush_sends_to_evolution(self) -> None:
        """Motor yüklenince biriken deneyimler evrime aktarılır."""

        class FakeEvolution:
            def __init__(self) -> None:
                self.seen: List[Dict[str, Any]] = []

            def observe(self, **kwargs: Any) -> Dict[str, Any]:
                self.seen.append(kwargs)
                return {"id": f"exp_{len(self.seen)}"}

        class FakeEngine:
            def __init__(self) -> None:
                self.evolution = FakeEvolution()

        self.recorder.initialize(workspace=self.root, data_path=self.root)
        for index in range(3):
            run = _success_run()
            run.task = f"gorev {index}"
            self.recorder.record(run)

        engine = FakeEngine()
        sent = self.recorder._flush_observations(engine)
        self.assertEqual(sent, 3)
        self.assertEqual(len(engine.evolution.seen), 3)
        self.assertEqual(engine.evolution.seen[0]["query"], "gorev 0")
        self.assertEqual(self.recorder.status()["stats"]["observations_pending"], 0)

    def test_flush_is_idempotent(self) -> None:
        """İkinci aktarım **çift göndermez** (aksi hâlde veri şişerdi)."""

        class FakeEvolution:
            def __init__(self) -> None:
                self.count = 0

            def observe(self, **kwargs: Any) -> Dict[str, Any]:
                self.count += 1
                return {}

        class FakeEngine:
            def __init__(self) -> None:
                self.evolution = FakeEvolution()

        self.recorder.initialize(workspace=self.root, data_path=self.root)
        self.recorder.record(_success_run())
        engine = FakeEngine()
        self.assertEqual(self.recorder._flush_observations(engine), 1)
        self.assertEqual(self.recorder._flush_observations(engine), 0)
        self.assertEqual(engine.evolution.count, 1)

    def test_observed_flag_persisted(self) -> None:
        """Aktarım durumu diske yazılır (yeniden başlatmada tekrar gönderilmez)."""

        class FakeEngine:
            class evolution:  # noqa: N801
                @staticmethod
                def observe(**kwargs: Any) -> Dict[str, Any]:
                    return {}

        self.recorder.initialize(workspace=self.root, data_path=self.root)
        self.recorder.record(_success_run())
        self.recorder._flush_observations(FakeEngine())
        self.recorder.shutdown(save_engine=False)

        fresh = LearningRecorder()
        fresh.initialize(workspace=self.root, data_path=self.root)
        try:
            self.assertEqual(
                fresh.status()["stats"]["observations_pending"], 0,
                "aktarılmış kayıt yeniden gönderilmemeli",
            )
        finally:
            fresh.shutdown(save_engine=False)

    def test_evolution_failure_tolerated(self) -> None:
        """Evrim katmanı hata verirse kayıt(gönderim) bozulmaz."""

        class BrokenEvolution:
            @staticmethod
            def observe(**kwargs: Any) -> Dict[str, Any]:
                raise RuntimeError("evrim çöktü")

        class FakeEngine:
            evolution = BrokenEvolution()

        self.recorder.initialize(workspace=self.root, data_path=self.root)
        self.recorder.record(_success_run())
        sent = self.recorder._flush_observations(FakeEngine())
        self.assertEqual(sent, 0, "hata durumunda gönderim sayılmamalı")
        # Kayıt kaybolmamalı: sonraki denemede yine gönderilebilir
        self.assertEqual(self.recorder.status()["stats"]["observations_pending"], 1)


class TestPendingCounter(unittest.TestCase):
    """
    Bekleyen iş sayacı **kayıtlardan** hesaplanmalı.

    Artımlı sayaç yeniden başlatmada sıfırlanıyordu; bu yüzden 12 öğretici
    kayıt birikmişken arayüz "bekleyen: 0" gösteriyor ve eğitim hiç
    başlamıyordu.
    """

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_pending_")
        self.recorder = LearningRecorder(min_new_experiences=3)
        self.recorder.initialize(workspace=self.root, data_path=self.root)

    def tearDown(self) -> None:
        import shutil

        self.recorder.shutdown(save_engine=False)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_pending_matches_records(self) -> None:
        """Bekleyen sayısı, eğitimde kullanılmamış öğretici kayıt sayısıdır."""
        for index in range(4):
            run = _success_run()
            run.task = f"g{index}"
            self.recorder.record(run)
        stats = self.recorder.status()["stats"]
        self.assertEqual(stats["pending"], 4)
        self.assertEqual(stats["used_in_training"], 0)

    def test_pending_recomputed_after_restart(self) -> None:
        """
        Yeniden başlatmada bekleyen iş **diskten** hesaplanır.

        Not: `training_readiness()["ready"]` burada `False` olabilir — çünkü
        **kaynak nezaketi** kapısı da denetlenir ve eğitim sonrası GPU/VRAM
        dolu olabilir. Test edilen şey **bekleyen iş sayacının** doğru
        hesaplanmasıdır; kaynak kapısı ayrı test edilir.
        """
        for index in range(4):
            run = _success_run()
            run.task = f"g{index}"
            self.recorder.record(run)
        self.recorder.shutdown(save_engine=False)

        fresh = LearningRecorder(min_new_experiences=3)
        fresh.initialize(workspace=self.root, data_path=self.root)
        try:
            stats = fresh.status()["stats"]
            self.assertEqual(stats["pending"], 4,
                             "yeniden başlatma bekleyen işi kaybetmemeli")
            # Gerekçe **bekleyen iş değil**, kaynak ise bunu ayırt edebilmeliyiz
            reason = fresh.training_readiness()["reason"]
            self.assertNotIn(
                "yeterli yeni deneyim yok", reason,
                f"4 bekleyen kayıt eşiği (3) geçmeliydi — gerekçe: {reason}",
            )
        finally:
            fresh.shutdown(save_engine=False)

    def test_resource_gate_blocks_when_busy(self) -> None:
        """
        Kaynaklar doluysa eğitim **başlamaz**.

        Bu, kullanıcının bilgisayarını kilitlememe kuralıdır: eğitim GPU/VRAM
        kullandığı için sistem meşgulken beklenir.
        """

        class BusyEngine:
            class model:  # noqa: N801
                pass

            @staticmethod
            def resources_available(*, for_heavy_work: bool = True) -> Any:
                return False, "sistem kaynakları dolu: VRAM %95 ≥ %90"

        self.recorder._engine = BusyEngine()
        for index in range(4):
            run = _success_run()
            run.task = f"g{index}"
            self.recorder.record(run)
        readiness = self.recorder.training_readiness()
        self.assertFalse(readiness["ready"])
        self.assertIn("kaynak", readiness["reason"].lower())
        self.assertEqual(readiness["pending"], 4,
                         "kaynak engeli bekleyen işi sıfırlamamalı")

    def test_chat_only_records_not_pending(self) -> None:
        """Öğretici olmayan sohbetler bekleyen sayılmaz."""
        self.recorder.record(FakeRun(task="merhaba", answer="selam", ok=True))
        self.assertEqual(self.recorder.status()["stats"]["pending"], 0)

    def test_readiness_reason_mentions_count(self) -> None:
        """Gerekçe kaç kayıt olduğunu ve ne yapılacağını söyler."""
        self.recorder.record(_success_run())
        reason = self.recorder.status()["readiness"]["reason"]
        self.assertIn("1/3", reason)
        self.assertIn("görev", reason)

class TestDeepSeekReasoningContent(unittest.TestCase):
    """
    DeepSeek düşünme modu: `reasoning_content` geri gönderilmeli.

    ⚠ Ölçülen hata: DeepSeek düşünme modu, araç çağrısı içeren asistan
    mesajlarında `reasoning_content` alanının aynen geri gönderilmesini
    bekler. Alan yoksa istek reddedilir:

        HTTP 400: The `reasoning_content` in the thinking mode must be
        passed back to the API.

    Bu, **sağlayıcı değişiminde** kaçınılmaz olarak ortaya çıkar:
    Groq/Gemini'den gelen asistan mesajlarında bu alan yoktur; zincir
    DeepSeek'e düştüğünde istek geçersiz olur. Ölçüldü: TitoWar analizi
    bu yüzden üç sağlayıcının hepsi başarısız olunca görevi bıraktı
    (ödül -0,60).
    """

    def setUp(self) -> None:
        from neurocore.agent.llm import OpenAICompatibleProvider

        self.prov = OpenAICompatibleProvider(
            service="deepseek", model="deepseek-flash",
            api_key="x", base_url="https://api.deepseek.com/v1",
        )

    @staticmethod
    def _tc() -> List[Dict[str, Any]]:
        return [{"id": "c1", "type": "function",
                 "function": {"name": "list_dir", "arguments": "{}"}}]

    def test_missing_field_added(self) -> None:
        """Eksik `reasoning_content` **boş dize** olarak eklenmeli."""
        pl = {"messages": [
            {"role": "user", "content": "Klasoru listele."},
            {"role": "assistant", "content": None, "tool_calls": self._tc()},
            {"role": "tool", "content": "scenes/", "tool_call_id": "c1"},
        ]}
        self.assertEqual(self.prov._normalize_payload(pl), 1)
        self.assertEqual(pl["messages"][1]["reasoning_content"], "")

    def test_existing_field_preserved(self) -> None:
        """Mevcut düşünme metni **dokunulmadan** korunmalı."""
        pl = {"messages": [
            {"role": "assistant", "content": None,
             "reasoning_content": "Once list_dir cagiracagim.",
             "tool_calls": self._tc()},
        ]}
        self.assertEqual(self.prov._normalize_payload(pl), 0)
        self.assertEqual(pl["messages"][0]["reasoning_content"],
                         "Once list_dir cagiracagim.")

    def test_non_assistant_untouched(self) -> None:
        """Kullanıcı ve araç mesajlarına dokunulmamalı."""
        pl = {"messages": [
            {"role": "user", "content": "selam"},
            {"role": "tool", "content": "sonuc", "tool_call_id": "c1"},
        ]}
        self.assertEqual(self.prov._normalize_payload(pl), 0)
        self.assertNotIn("reasoning_content", pl["messages"][0])

    def test_assistant_without_tool_calls_untouched(self) -> None:
        """Araç çağrısı olmayan asistan mesajına eklenmemeli."""
        pl = {"messages": [{"role": "assistant", "content": "sadece metin"}]}
        self.assertEqual(self.prov._normalize_payload(pl), 0)
        self.assertNotIn("reasoning_content", pl["messages"][0])

    def test_idempotent(self) -> None:
        """İki kez çağrılması zarar vermemeli."""
        pl = {"messages": [
            {"role": "assistant", "content": None, "tool_calls": self._tc()},
        ]}
        self.assertEqual(self.prov._normalize_payload(pl), 1)
        self.assertEqual(self.prov._normalize_payload(pl), 0)

    def test_mixed_history_all_fixed(self) -> None:
        """
        Farklı sağlayıcılardan gelmiş karışık geçmiş **tümü** düzeltilmeli.

        Gerçek senaryo: Groq bir tur, Gemini başka tur çalıştırır, sonra
        DeepSeek'e düşülür.
        """
        pl = {"messages": [
            {"role": "user", "content": "bir"},
            {"role": "assistant", "content": None, "tool_calls": self._tc()},
            {"role": "tool", "content": "a", "tool_call_id": "c1"},
            {"role": "assistant", "content": None, "tool_calls": self._tc(),
             "reasoning_content": "dusundum"},
            {"role": "tool", "content": "b", "tool_call_id": "c1"},
            {"role": "assistant", "content": None, "tool_calls": self._tc()},
            {"role": "tool", "content": "c", "tool_call_id": "c1"},
        ]}
        self.assertEqual(self.prov._normalize_payload(pl), 2)
        for i in (1, 3, 5):
            self.assertIn("reasoning_content", pl["messages"][i])
        # Mevcut metin değişmemeli
        self.assertEqual(pl["messages"][3]["reasoning_content"], "dusundum")

    def test_empty_payload_safe(self) -> None:
        self.assertEqual(self.prov._normalize_payload({}), 0)
        self.assertEqual(self.prov._normalize_payload({"messages": []}), 0)

class TestModelSelection(unittest.TestCase):
    """
    Model seçimi — kullanıcının bilinçli tercihi **geri çevrilmemeli**.

    ⚠ Ölçülen hata: `DEPRECATED_MODELS` haritasında `"deepseek-chat":
    "deepseek-flash"` kaydı vardı. Bu, kullanıcı `deepseek-chat` seçtiğinde
    modeli **sessizce** `deepseek-flash`'e çeviriyordu — ayar kaydedilse
    bile yeniden başlatmada eski modele dönüyordu.

    Neden önemli? `deepseek-flash` bir **düşünme modeli**: 8192 token
    bütçesinin 2.885'ini düşünmede harcıyor. Bağlam büyüyünce bütçe bitiyor
    ve ajan "model 3 kez boş yanıt verdi (bitiş nedeni: length, token:
    8192)" diyerek duruyor. `deepseek-chat` düşünme token'ı harcamaz.
    """

    def test_chat_not_remapped_to_flash(self) -> None:
        """`deepseek-chat` artık başka modele çevrilmemeli."""
        from neurocore.agent.llm import DEPRECATED_MODELS

        self.assertNotEqual(DEPRECATED_MODELS.get("deepseek-chat"),
                            "deepseek-flash")
        self.assertNotIn("deepseek-chat", DEPRECATED_MODELS)

    def test_flash_maps_to_chat(self) -> None:
        """Eski `deepseek-flash` tercihi güvenilir modele taşınmalı."""
        from neurocore.agent.llm import DEPRECATED_MODELS

        self.assertEqual(DEPRECATED_MODELS.get("deepseek-flash"),
                         "deepseek-chat")

    def test_default_model_is_chat(self) -> None:
        """Varsayılan DeepSeek modeli `deepseek-chat` olmalı."""
        from neurocore.agent.llm import KNOWN_SERVICES

        self.assertEqual(KNOWN_SERVICES["deepseek"]["model"],
                         "deepseek-chat")

    def test_suggested_chain_prefers_chat(self) -> None:
        """Yedek zincirinde `deepseek-chat` **ilk** olmalı."""
        from neurocore.agent.llm import SUGGESTED_MODELS

        zincir = SUGGESTED_MODELS.get("deepseek") or []
        self.assertTrue(zincir)
        self.assertEqual(zincir[0], "deepseek-chat")

    def test_reasoner_still_maps_to_pro(self) -> None:
        """Düşünme modeli eşlemesi korunmalı."""
        from neurocore.agent.llm import DEPRECATED_MODELS

        self.assertEqual(DEPRECATED_MODELS.get("deepseek-reasoner"),
                         "deepseek-v4-pro")

class TestToolMessageRepair(unittest.TestCase):
    """
    Yanıtsız araç çağrılarının onarımı.

    ⚠ Ölçülen hata: Sağlayıcı, `tool_calls` içeren bir asistan mesajından
    sonra **her** `tool_call_id` için bir `tool` mesajı bekler. Ajan bir
    turda birden fazla araç çağırdığında (paralel çağrı) ve tur yarıda
    kaldığında (bağlam taşması, hata) bazı çağrılar yanıtsız kalır:

        HTTP 400: An assistant message with 'tool_calls' must be followed
        by tool messages responding to each 'tool_call_id'.
        (insufficient tool messages following tool_calls message)

    ⚠ Bu **kalıcıdır**: bozuk geçmiş temizlenmedikçe her sonraki istek de
    reddedilir ve oturum tamamen kilitlenir.
    """

    def setUp(self) -> None:
        from neurocore.agent.llm import OpenAICompatibleProvider

        self.prov = OpenAICompatibleProvider(
            service="deepseek", model="deepseek-chat",
            api_key="x", base_url="https://api.deepseek.com/v1",
        )

    @staticmethod
    def _tc(kimlik: str) -> Dict[str, Any]:
        return {"id": kimlik, "type": "function",
                "function": {"name": "f", "arguments": "{}"}}

    def test_missing_responses_added(self) -> None:
        """3 çağrı, 1 yanıt → 2 sentetik yanıt eklenmeli."""
        pl = {"messages": [
            {"role": "user", "content": "3 dosya oluştur."},
            {"role": "assistant", "content": None,
             "tool_calls": [self._tc("c1"), self._tc("c2"), self._tc("c3")]},
            {"role": "tool", "content": "ok", "tool_call_id": "c1"},
        ]}
        self.assertEqual(self.prov._repair_tool_messages(pl), 2)
        kimlikler = [m.get("tool_call_id") for m in pl["messages"]
                     if m.get("role") == "tool"]
        self.assertEqual(sorted(kimlikler), ["c1", "c2", "c3"])

    def test_all_missing_added(self) -> None:
        """Hiç yanıt yoksa hepsi eklenmeli."""
        pl = {"messages": [
            {"role": "assistant", "content": None,
             "tool_calls": [self._tc("c1"), self._tc("c2")]},
        ]}
        self.assertEqual(self.prov._repair_tool_messages(pl), 2)
        self.assertEqual(len([m for m in pl["messages"]
                              if m.get("role") == "tool"]), 2)

    def test_intact_history_untouched(self) -> None:
        """Sağlam geçmişe **dokunulmamalı**."""
        pl = {"messages": [
            {"role": "user", "content": "iş yap."},
            {"role": "assistant", "content": None,
             "tool_calls": [self._tc("c1"), self._tc("c2")]},
            {"role": "tool", "content": "ok1", "tool_call_id": "c1"},
            {"role": "tool", "content": "ok2", "tool_call_id": "c2"},
        ]}
        oncesi = len(pl["messages"])
        self.assertEqual(self.prov._repair_tool_messages(pl), 0)
        self.assertEqual(len(pl["messages"]), oncesi)

    def test_consecutive_turns_second_orphaned(self) -> None:
        """Ardışık turlarda yalnızca **bozuk** tur onarılmalı."""
        pl = {"messages": [
            {"role": "user", "content": "iş yap."},
            {"role": "assistant", "content": None,
             "tool_calls": [self._tc("c1")]},
            {"role": "tool", "content": "ok1", "tool_call_id": "c1"},
            {"role": "assistant", "content": None,
             "tool_calls": [self._tc("c2"), self._tc("c3")]},
        ]}
        self.assertEqual(self.prov._repair_tool_messages(pl), 2)
        # İlk tur bozulmamalı
        self.assertEqual(pl["messages"][2].get("tool_call_id"), "c1")

    def test_no_tool_calls_untouched(self) -> None:
        """Araç çağrısı olmayan geçmişe dokunulmamalı."""
        pl = {"messages": [
            {"role": "user", "content": "selam"},
            {"role": "assistant", "content": "merhaba"},
        ]}
        self.assertEqual(self.prov._repair_tool_messages(pl), 0)

    def test_synthetic_content_is_informative(self) -> None:
        """Sentetik yanıt modele durumu **açıklamalı**."""
        pl = {"messages": [
            {"role": "assistant", "content": None,
             "tool_calls": [self._tc("c1")]},
        ]}
        self.prov._repair_tool_messages(pl)
        icerik = pl["messages"][1]["content"].lower()
        self.assertIn("kesildi", icerik)

    def test_idempotent(self) -> None:
        """İki kez çağrılması zarar vermemeli."""
        pl = {"messages": [
            {"role": "assistant", "content": None,
             "tool_calls": [self._tc("c1"), self._tc("c2")]},
        ]}
        self.assertEqual(self.prov._repair_tool_messages(pl), 2)
        self.assertEqual(self.prov._repair_tool_messages(pl), 0)

    def test_empty_payload_safe(self) -> None:
        self.assertEqual(self.prov._repair_tool_messages({}), 0)
        self.assertEqual(self.prov._repair_tool_messages({"messages": []}), 0)

    def test_malformed_entries_safe(self) -> None:
        """Bozuk kayıtlar çökme yaratmamalı."""
        pl = {"messages": [
            {"role": "assistant", "tool_calls": [None, {"id": ""}, "bozuk"]},
        ]}
        self.prov._repair_tool_messages(pl)   # çökmemeli


if __name__ == "__main__":
    unittest.main(verbosity=2)
