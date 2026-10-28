"""
Yerel Sıralama Aracı Testleri
==============================
`neurocore.agent.local_rank` — yerel modelin **tanıma** yeteneğini kullanan
aday sıralama aracı.

Neden bu testler önemli?
    Aracın değeri **ölçülmüş** bir yeteneğe dayanır (Blender kod seçiminde
    %70 isabet, rastgele %25). Bu testler aracın:
    * gerçek model olasılığını kullandığını (kelime torbası sezgisi değil),
    * belirsiz durumları **dürüstçe** bildirdiğini,
    * bozuk girdilerde çökmediğini,
    * doğru güvenlik düzeyinde kaydedildiğini

    doğrular. Sahte model kullanılır; GPU gerekmez.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

from tests.helpers import assert_json_safe

from neurocore.agent.local_rank import (
    DEFAULT_MIN_MARGIN,
    MAX_CANDIDATE_CHARS,
    MIN_CANDIDATES,
    LocalRankController,
    RankedCandidate,
)
from neurocore.agent.tools import Safety


class FakeGrowth:
    """Sahte büyüme denetleyicisi."""

    total_training_steps = 809


class FakeModel:
    """
    Sahte yerel model.

    `sequence_logprob` gerçek modele benzer biçimde davranır: metin
    uzunluğuna ve içeriğe göre **belirleyici** bir skor üretir.
    """

    MIN_GENERATION_STEPS = 800

    def __init__(self, *, favor: str = "", ready: bool = True) -> None:
        self.growth = FakeGrowth()
        self.can_generate = ready
        self.favor = favor
        self.calls: List[str] = []
        self.raises = False

    def sequence_logprob(self, full_text: str, *, prompt: str = "") -> Optional[float]:
        """Tercih edilen metne yüksek skor verir."""
        if self.raises:
            raise RuntimeError("kasıtlı hata")
        self.calls.append(full_text)
        if self.favor and self.favor in full_text:
            return -1.0
        # Daha uzun devam = biraz daha düşük (gerçekçi davranış)
        return -5.0 - len(full_text) * 1e-5


class FakeEngine:
    """Sahte motor (model taşıyıcısı)."""

    def __init__(self, model: Any = None) -> None:
        self.model = model


class TestRankingBasics(unittest.TestCase):
    """Temel sıralama davranışı."""

    def _controller(self, **kwargs: Any) -> LocalRankController:
        model = FakeModel(**kwargs)
        controller = LocalRankController()
        controller.initialize(engine=FakeEngine(model))
        return controller

    def test_picks_favored_candidate(self) -> None:
        """Tercih edilen aday birinci olmalı."""
        controller = self._controller(favor="DOGRU_KOD")
        result = controller.rank_candidates(
            "Soru:", ["yanlis_bir", "DOGRU_KOD", "yanlis_iki"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["best_index"], 1)
        self.assertIn("DOGRU_KOD", result["best_text"])

    def test_ranking_is_sorted_descending(self) -> None:
        """Sıralama skora göre azalan olmalı."""
        controller = self._controller(favor="en_iyi")
        result = controller.rank_candidates("S:", ["a", "en_iyi", "b", "c"])
        skorlar = [item["score"] for item in result["ranking"]]
        self.assertEqual(skorlar, sorted(skorlar, reverse=True))

    def test_model_probability_used_not_heuristic(self) -> None:
        """
        Arac **modelin** olasılığını kullanmalı.

        Kanıt: sahte model her adayı kaydettiği için çağrı sayısı aday
        sayısına eşit olmalı ve istem+aday birlikte gönderilmeli.
        """
        model = FakeModel(favor="B")
        controller = LocalRankController()
        controller.initialize(engine=FakeEngine(model))
        controller.rank_candidates("ISTEM:", ["A", "B", "C"])
        self.assertEqual(len(model.calls), 3)
        for cagri in model.calls:
            self.assertTrue(cagri.startswith("ISTEM:"))

    def test_prompt_scored_separately(self) -> None:
        """İstem skordan düşülmeli (adaylar istem uzunluğundan etkilenmesin)."""
        model = FakeModel()
        controller = LocalRankController()
        controller.initialize(engine=FakeEngine(model))
        controller.rank_candidates("UZUN_ISTEM" * 20, ["kisa", "daha_uzun_aday"])
        self.assertEqual(len(model.calls), 2)
        for cagri in model.calls:
            self.assertIn("UZUN_ISTEM", cagri)

    def test_top_k_limits_results(self) -> None:
        controller = self._controller()
        result = controller.rank_candidates("S:", ["a", "b", "c", "d"], top_k=2)
        self.assertEqual(len(result["ranking"]), 2)

    def test_result_json_safe(self) -> None:
        controller = self._controller()
        result = controller.rank_candidates("S:", ["a", "b"])
        assert_json_safe(self, result, "rank_result")


class TestConfidence(unittest.TestCase):
    """Güven düzeyi dürüst bildirilmeli."""

    def test_high_confidence_on_clear_winner(self) -> None:
        """Belirgin kazanan 'yüksek' güven vermeli."""
        model = FakeModel(favor="kazanan")
        controller = LocalRankController()
        controller.initialize(engine=FakeEngine(model))
        result = controller.rank_candidates("S:", ["kazanan", "kaybeden"])
        self.assertFalse(result["uncertain"])
        self.assertEqual(result["confidence"], "yüksek")
        self.assertGreater(result["margin"], 0.5)

    def test_uncertain_when_scores_close(self) -> None:
        """
        Skorlar yakınsa **belirsiz** denmeli.

        Neden? Model %70 isabetli; küçük farklarda karar güvenilmez.
        Aracın bunu saklaması kullanıcıyı yanıltır.
        """
        model = FakeModel()          # tercih yok → skorlar çok yakın
        controller = LocalRankController()
        controller.initialize(engine=FakeEngine(model))
        result = controller.rank_candidates("S:", ["a", "b"], min_margin=10.0)
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["confidence"], "belirsiz")
        self.assertEqual(controller.get_stats()["uncertain"], 1)

    def test_margin_reported(self) -> None:
        model = FakeModel(favor="kazanan")
        controller = LocalRankController()
        controller.initialize(engine=FakeEngine(model))
        result = controller.rank_candidates("S:", ["kazanan", "x", "y"])
        self.assertIsNotNone(result["margin"])
        self.assertGreater(result["margin"], 0)

    def test_default_min_margin_positive(self) -> None:
        self.assertGreater(DEFAULT_MIN_MARGIN, 0.0)


class TestBadInput(unittest.TestCase):
    """Bozuk girdilerde çökmemeli, anlaşılır hata dönmeli."""

    def _controller(self, model: Any = None) -> LocalRankController:
        controller = LocalRankController()
        controller.initialize(engine=FakeEngine(model or FakeModel()))
        return controller

    def test_single_candidate_rejected(self) -> None:
        result = self._controller().rank_candidates("S:", ["tek"])
        self.assertFalse(result["ok"])
        self.assertIn("aday", result["error"])

    def test_empty_list_rejected(self) -> None:
        result = self._controller().rank_candidates("S:", [])
        self.assertFalse(result["ok"])

    def test_blank_candidates_filtered(self) -> None:
        """Boş/whitespace adaylar elenir; kalan 2 ise çalışır."""
        result = self._controller().rank_candidates("S:", ["a", "   ", "", "b"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["candidates"], 2)

    def test_missing_model_reports_clearly(self) -> None:
        """Model yoksa net hata (çökme değil)."""
        controller = LocalRankController()
        controller.initialize(engine=FakeEngine(None))
        controller._model = lambda: None      # type: ignore[assignment]
        result = controller.rank_candidates("S:", ["a", "b"])
        self.assertFalse(result["ok"])
        self.assertIn("model", result["error"].lower())
        self.assertEqual(controller.get_stats()["model_unavailable"], 1)

    def test_model_exception_handled(self) -> None:
        """Model istisna atarsa araç ayakta kalmalı."""
        model = FakeModel()
        model.raises = True
        controller = self._controller(model)
        result = controller.rank_candidates("S:", ["a", "b"])
        self.assertFalse(result["ok"])
        self.assertEqual(controller.get_stats()["errors"], 1)

    def test_long_candidates_truncated(self) -> None:
        """Aşırı uzun adaylar kırpılır (bağlam taşmasın)."""
        uzun = "x" * (MAX_CANDIDATE_CHARS * 2)
        result = self._controller().rank_candidates("S:", [uzun, "kisa"])
        self.assertTrue(result["ok"])
        self.assertLessEqual(len(result["best_text"]), MAX_CANDIDATE_CHARS)

    def test_min_candidates_constant(self) -> None:
        self.assertGreaterEqual(MIN_CANDIDATES, 2)


class TestToolDefinition(unittest.TestCase):
    """Ajan aracı olarak doğru tanımlanmalı."""

    def setUp(self) -> None:
        self.controller = LocalRankController()
        self.controller.initialize(engine=FakeEngine(FakeModel()))
        self.tools = {tool.name: tool for tool in self.controller.tools()}

    def test_tools_exposed(self) -> None:
        self.assertIn("local.rank_candidates", self.tools)
        self.assertIn("local.model_status", self.tools)

    def test_read_only_safety(self) -> None:
        """Sıralama **okuma** işidir; sistem değiştirmez."""
        for tool in self.tools.values():
            self.assertEqual(tool.safety, Safety.READ)

    def test_wire_name_compatible(self) -> None:
        """Araç adları sağlayıcı regex'ine uymalı (nokta yasak)."""
        from neurocore.agent.tools import to_wire_name

        for name in self.tools:
            wire = to_wire_name(name)
            self.assertNotIn(".", wire)
            self.assertRegex(wire, r"^[a-zA-Z0-9_-]+$")

    def test_handler_via_tool(self) -> None:
        """Araç işleyicisi doğrudan çağrılabilmeli."""
        tool = self.tools["local.rank_candidates"]
        result = tool.handler(prompt="S:", candidates=["a", "b"])
        self.assertTrue(result["ok"])

    def test_handler_accepts_json_string(self) -> None:
        """Model JSON metni verirse ayrıştırılmalı."""
        tool = self.tools["local.rank_candidates"]
        result = tool.handler(prompt="S:", candidates='["a", "b"]')
        self.assertTrue(result["ok"])
        self.assertEqual(result["candidates"], 2)

    def test_handler_accepts_delimited_string(self) -> None:
        """`---` ile ayrılmış metin de kabul edilmeli."""
        tool = self.tools["local.rank_candidates"]
        result = tool.handler(prompt="S:", candidates="birinci\n---\nikinci")
        self.assertTrue(result["ok"])
        self.assertEqual(result["candidates"], 2)

    def test_status_tool_reports_state(self) -> None:
        tool = self.tools["local.model_status"]
        result = tool.handler()
        self.assertTrue(result["initialized"])
        self.assertTrue(result["model_ready"])
        self.assertEqual(result["training_steps"], 809)

    def test_has_description(self) -> None:
        for tool in self.tools.values():
            self.assertGreater(len(tool.description), 30)

    def test_rank_tool_has_parameters(self) -> None:
        tool = self.tools["local.rank_candidates"]
        adlar = {p.name for p in tool.parameters}
        self.assertIn("prompt", adlar)
        self.assertIn("candidates", adlar)


class TestLifecycle(unittest.TestCase):
    """Yaşam döngüsü ve sayaçlar."""

    def test_initialize_idempotent(self) -> None:
        controller = LocalRankController()
        engine = FakeEngine(FakeModel())
        controller.initialize(engine=engine)
        controller.initialize(engine=engine)
        self.assertTrue(controller.get_stats()["initialized"])

    def test_attach_engine(self) -> None:
        controller = LocalRankController()
        controller.initialize()
        controller.attach_engine(FakeEngine(FakeModel()))
        result = controller.rank_candidates("S:", ["a", "b"])
        self.assertTrue(result["ok"])

    def test_counters_accumulate(self) -> None:
        controller = LocalRankController()
        controller.initialize(engine=FakeEngine(FakeModel()))
        controller.rank_candidates("S:", ["a", "b", "c"])
        controller.rank_candidates("S:", ["d", "e"])
        stats = controller.get_stats()
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["candidates_scored"], 5)

    def test_shutdown_clears_flag(self) -> None:
        controller = LocalRankController()
        controller.initialize(engine=FakeEngine(FakeModel()))
        controller.shutdown()
        self.assertFalse(controller.get_stats()["initialized"])

    def test_stats_json_safe(self) -> None:
        controller = LocalRankController()
        controller.initialize(engine=FakeEngine(FakeModel()))
        assert_json_safe(self, controller.get_stats(), "local_rank_stats")

    def test_ranked_candidate_dict(self) -> None:
        item = RankedCandidate(index=0, text="a" * 400, score=-1.5, margin=0.9)
        data = item.to_dict()
        assert_json_safe(self, data, "ranked_candidate")
        self.assertLessEqual(len(data["preview"]), 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
