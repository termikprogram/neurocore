"""
Düşünme Döngüsü Testleri
=========================
`neurocore.agent.thinking` — bulut üretir, yerel seçer, belirsizlikte
buluta düşülür.

Neden bu testler önemli?
    Döngünün değeri **dürüstlüğüne** bağlıdır: yerel seçim belirsizse
    kullanılmamalı ve bu durum gizlenmemelidir. Ayrıca her tur
    günlüğe yazılmalıdır ki "isabet ediyor" iddiası kanıtlanabilsin.

Sahte sıralayıcı kullanılır; GPU/model gerekmez.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tests.helpers import assert_json_safe

from neurocore.agent.thinking import (
    DEFAULT_CANDIDATES,
    DEFAULT_MIN_MARGIN,
    JOURNAL_FILE,
    ThinkingLoop,
    ThinkingResult,
    ThinkingStep,
)


class FakeRanker:
    """Sahte yerel sıralayıcı (LocalRankController benzeri)."""

    def __init__(self, *, best: int = 0, uncertain: bool = False,
                 margin: float = 1.0, fail: bool = False) -> None:
        self.best = best
        self.uncertain = uncertain
        self.margin = margin
        self.fail = fail
        self.calls: List[Dict[str, Any]] = []

    def rank_candidates(self, prompt: str, candidates: Sequence[str], *,
                        min_margin: float = 0.0, top_k: int = 0) -> Dict[str, Any]:
        """Sıralama taklidi."""
        self.calls.append({"prompt": prompt, "candidates": list(candidates),
                           "min_margin": min_margin})
        if self.fail:
            return {"ok": False, "error": "sıralama başarısız"}
        siralama = [
            {"index": i, "score": -1.0 * i, "margin": -1.0 * i,
             "preview": str(candidates[i])[:80]}
            for i in range(len(candidates))
        ]
        return {
            "ok": True, "best_index": self.best,
            "best_text": str(candidates[self.best]),
            "margin": self.margin, "uncertain": self.uncertain,
            "confidence": "belirsiz" if self.uncertain else "yüksek",
            "candidates": len(candidates), "ranking": siralama,
        }


class TestThinkingLoop(unittest.TestCase):
    """Döngünün temel davranışı."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_think_")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _loop(self, ranker: Any = None, **kwargs: Any) -> ThinkingLoop:
        loop = ThinkingLoop(**kwargs)
        loop.initialize(ranker=ranker or FakeRanker(), data_path=self.root)
        return loop

    def test_uses_local_choice_when_confident(self) -> None:
        """Yerel seçim belirginse **o kullanılmalı**."""
        ranker = FakeRanker(best=1, uncertain=False, margin=1.5)
        loop = self._loop(ranker)
        sonuc = loop.think("Problem",
                           lambda n: ["yanlis_a", "dogru_b", "yanlis_c"])
        self.assertTrue(sonuc.ok)
        self.assertEqual(sonuc.source, "local")
        self.assertEqual(sonuc.chosen, "dogru_b")
        self.assertEqual(sonuc.confidence, "yüksek")
        self.assertEqual(loop.get_stats()["local_chosen"], 1)

    def test_falls_back_to_cloud_when_uncertain(self) -> None:
        """
        Belirsizse **bulut tercihi** geçerli olmalı.

        Neden? Yerel isabet %50–70; küçük farklarda karar güvenilmez.
        Belirsizken yerel seçimi dayatmak kaliteyi düşürür.
        """
        ranker = FakeRanker(best=2, uncertain=True, margin=0.01)
        loop = self._loop(ranker)
        sonuc = loop.think("Problem", lambda n: ["bulut_tercihi", "b", "c"])
        self.assertTrue(sonuc.ok)
        self.assertEqual(sonuc.source, "cloud")
        self.assertEqual(sonuc.chosen, "bulut_tercihi")
        stats = loop.get_stats()
        self.assertEqual(stats["uncertain"], 1)
        self.assertEqual(stats["local_chosen"], 0)

    def test_propose_receives_candidate_count(self) -> None:
        """Üretici, istenen aday sayısını almalı."""
        loop = self._loop(candidates=4)
        alinan: List[int] = []

        def uret(n: int) -> List[str]:
            alinan.append(n)
            return [f"aday{i}" for i in range(n)]

        loop.think("P", uret)
        self.assertEqual(alinan, [4])

    def test_single_candidate_skips_selection(self) -> None:
        """Tek aday varsa seçim yapılmaz, doğrudan kullanılır."""
        ranker = FakeRanker()
        loop = self._loop(ranker)
        sonuc = loop.think("P", lambda n: ["tek_cozum"])
        self.assertTrue(sonuc.ok)
        self.assertEqual(sonuc.chosen, "tek_cozum")
        self.assertEqual(sonuc.source, "cloud")
        self.assertEqual(len(ranker.calls), 0, "tek adayda sıralama çağrılmamalı")

    def test_empty_candidates_handled(self) -> None:
        loop = self._loop()
        sonuc = loop.think("P", lambda n: [])
        self.assertFalse(sonuc.ok)
        self.assertEqual(sonuc.chosen, "")

    def test_propose_exception_handled(self) -> None:
        """Üretici çökerse döngü çökmemeli."""
        loop = self._loop()

        def patla(n: int) -> List[str]:
            raise RuntimeError("bulut erişilemedi")

        sonuc = loop.think("P", patla)
        self.assertFalse(sonuc.ok)
        self.assertIn("aday üretilemedi", sonuc.error)
        self.assertEqual(loop.get_stats()["errors"], 1)

    def test_ranker_failure_falls_back(self) -> None:
        """Sıralayıcı çökerse bulut tercihine düşülmeli."""
        loop = self._loop(FakeRanker(fail=True))
        sonuc = loop.think("P", lambda n: ["birinci", "ikinci"])
        self.assertTrue(sonuc.ok)
        self.assertEqual(sonuc.source, "cloud")
        self.assertEqual(sonuc.chosen, "birinci")

    def test_no_ranker_reports_error(self) -> None:
        loop = ThinkingLoop()
        loop.initialize(ranker=None, data_path=self.root)
        sonuc = loop.think("P", lambda n: ["a", "b"])
        self.assertFalse(sonuc.ok)
        self.assertIn("sıralayıcı", sonuc.error)

    def test_min_margin_passed_to_ranker(self) -> None:
        """Belirsizlik eşiği sıralayıcıya iletilmeli."""
        ranker = FakeRanker()
        loop = self._loop(ranker, min_margin=0.42)
        loop.think("P", lambda n: ["a", "b"])
        self.assertAlmostEqual(ranker.calls[0]["min_margin"], 0.42)

    def test_context_included_in_prompt(self) -> None:
        ranker = FakeRanker()
        loop = self._loop(ranker)
        loop.think("Problem", lambda n: ["a", "b"], context="KISIT: 5 saniye")
        self.assertIn("KISIT", ranker.calls[0]["prompt"])

    def test_stats_rates(self) -> None:
        loop = self._loop(FakeRanker(uncertain=True))
        loop.think("P", lambda n: ["a", "b"])
        stats = loop.get_stats()
        self.assertEqual(stats["rounds"], 1)
        self.assertAlmostEqual(stats["uncertain_rate"], 1.0)
        self.assertAlmostEqual(stats["local_share"], 0.0)

    def test_stats_json_safe(self) -> None:
        assert_json_safe(self, self._loop().get_stats(), "thinking_stats")


class TestThinkingJournal(unittest.TestCase):
    """Ölçüm günlüğü — isabet iddiası kanıtlanabilir olmalı."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_tj_")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _loop(self, ranker: Any = None, **kwargs: Any) -> ThinkingLoop:
        loop = ThinkingLoop(**kwargs)
        loop.initialize(ranker=ranker or FakeRanker(), data_path=self.root)
        return loop

    def test_journal_written(self) -> None:
        """Her tur günlüğe bir satır yazmalı."""
        loop = self._loop()
        loop.think("Problem A", lambda n: ["x", "y"])
        yol = Path(self.root) / JOURNAL_FILE
        self.assertTrue(yol.exists())
        satirlar = [s for s in yol.read_text(encoding="utf-8").splitlines()
                    if s.strip()]
        self.assertEqual(len(satirlar), 1)
        kayit = json.loads(satirlar[0])
        self.assertEqual(kayit["source"], "local")
        self.assertIn("Problem A", kayit["problem"])

    def test_journal_appends(self) -> None:
        """Günlük birikmeli (üzerine yazmamalı)."""
        loop = self._loop()
        for i in range(3):
            loop.think(f"Problem {i}", lambda n: ["a", "b"])
        yol = Path(self.root) / JOURNAL_FILE
        satirlar = [s for s in yol.read_text(encoding="utf-8").splitlines()
                    if s.strip()]
        self.assertEqual(len(satirlar), 3)

    def test_journal_records_source_and_margin(self) -> None:
        """Kaynak ve güven bilgisi kaydedilmeli (denetim için)."""
        loop = self._loop(FakeRanker(uncertain=True, margin=0.01))
        loop.think("P", lambda n: ["a", "b"])
        yol = Path(self.root) / JOURNAL_FILE
        kayit = json.loads(yol.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(kayit["source"], "cloud")
        # Belirsizlikte de fark kaydedilir (neden buluta düşüldüğü görünsün)
        self.assertIn("margin", kayit)
        self.assertAlmostEqual(kayit["margin"], 0.01)

    def test_journal_json_lines_valid(self) -> None:
        """Her satır geçerli JSON olmalı (bozuk satır = okunamaz günlük)."""
        loop = self._loop()
        for i in range(4):
            loop.think(f"P{i}", lambda n: ["a", "b", "c"])
        yol = Path(self.root) / JOURNAL_FILE
        for satir in yol.read_text(encoding="utf-8").splitlines():
            if satir.strip():
                json.loads(satir)   # istisna atarsa test başarısız


class TestThinkingResult(unittest.TestCase):
    """Sonuç nesnesi ve JSON güvenliği."""

    def test_result_dict(self) -> None:
        sonuc = ThinkingResult(ok=True, chosen="x", source="local",
                               confidence="yüksek", margin=1.2,
                               candidates=["x", "y"])
        veri = sonuc.to_dict()
        assert_json_safe(self, veri, "thinking_result")
        self.assertEqual(veri["source"], "local")

    def test_step_dict_truncates(self) -> None:
        adim = ThinkingStep(stage="uret", detail="a" * 900)
        veri = adim.to_dict()
        assert_json_safe(self, veri, "thinking_step")
        self.assertLessEqual(len(veri["detail"]), 300)

    def test_steps_recorded(self) -> None:
        """Döngü adımları denetim izi olarak kaydedilmeli."""
        with tempfile.TemporaryDirectory() as kok:
            loop = ThinkingLoop()
            loop.initialize(ranker=FakeRanker(), data_path=kok)
            sonuc = loop.think("P", lambda n: ["a", "b"])
        asamalar = [s.stage for s in sonuc.steps]
        self.assertIn("uret", asamalar)
        self.assertIn("sec", asamalar)
        self.assertIn("bitti", asamalar)

    def test_constants_sane(self) -> None:
        self.assertGreaterEqual(DEFAULT_CANDIDATES, 2)
        self.assertGreater(DEFAULT_MIN_MARGIN, 0.0)

    def test_lifecycle(self) -> None:
        loop = ThinkingLoop()
        loop.initialize(ranker=FakeRanker(), data_path=self.root_path())
        self.assertTrue(loop.get_stats()["initialized"])
        loop.shutdown()
        self.assertFalse(loop.get_stats()["initialized"])

    def root_path(self) -> str:
        """Test için geçici dizin."""
        return tempfile.mkdtemp(prefix="nc_lc_")


if __name__ == "__main__":
    unittest.main(verbosity=2)
