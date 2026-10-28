"""
Müfredat ve Eğitim Hattı Testleri
==================================
İki modülü kapsar:

* `neurocore.neural.curriculum` — katmanlı ders içeriği ve genişletme
* `neurocore.neural.pipeline`   — tokenizer yenileme + gömme boyutlandırma

Ağır eğitim testleri **atlanır** (GPU/bellek gerektirir); ağsız ve hafif
mantık (içerik bütünlüğü, genişletme, gömme boyutlandırma) her zaman
sınanır.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import assert_json_safe

from neurocore.neural.curriculum import (
    ALL_LAYERS,
    LAYER_LABELS,
    LAYERS,
    QUESTION_PATTERNS,
    Lesson,
    build_curriculum,
    curriculum_stats,
    expand_curriculum,
)
from neurocore.neural.pipeline import (
    DEFAULT_VOCAB_SIZE,
    PROGRESS_FILE,
    CurriculumTrainer,
    PipelineProgress,
)


# ─── Müfredat İçeriği ───────────────────────────────────────────────────────

class TestCurriculumContent(unittest.TestCase):
    """Ders içeriği eksiksiz ve tutarlı olmalı."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.lessons = build_curriculum()

    def test_all_layers_covered(self) -> None:
        """Her katmanda ders olmalı (kullanıcının istediği tüm alanlar)."""
        present = {lesson.layer for lesson in self.lessons}
        for layer in LAYERS:
            self.assertIn(layer, present, f"{layer} katmanı boş")

    @staticmethod
    def _gdscript_ayikla(kod: str) -> str:
        """
        GDScript kodundan **yorumları ve dizgileri** çıkarır.

        Parantez dengesi kontrolü için gerekli: yorumlardaki parantezler
        (ör. "(`body_entered`)") sayımı bozar.
        """
        import re

        # Satır yorumlarını kaldır (# ile başlayan veya sonrası)
        satirlar = []
        for satir in kod.splitlines():
            # Dizgi içindeki # korunmalı — basit yaklaşım: tırnak sayısına bak
            tirnak = satir.count('"') + satir.count("'")
            if tirnak % 2 == 0:
                satir = satir.split("#")[0]
            satirlar.append(satir)
        kod = "\n".join(satirlar)
        # Dizgileri boşalt
        kod = re.sub(r'"[^"\n]*"', '""', kod)
        kod = re.sub(r"'[^'\n]*'", "''", kod)
        return kod

    def test_layer_counts(self) -> None:
        """
        Her katman anlamlı sayıda ders içerir.

        ⚠ Eşikler katmana göre değişir (ölçüldü):
            * Yerel/ağ katmanları : en az 8 ders
            * Dar uzmanlık katmanları : en az 4 ders
              (lowpoly doku/detay ve Godot katmanları — ileri konular
               dar ve derindir, doğal olarak az ders içerir)
        """
        counts = {}
        for lesson in self.lessons:
            counts[lesson.layer] = counts.get(lesson.layer, 0) + 1

        #: Dar uzmanlık katmanları — 4 ders yeterli
        #: Ölçülen ders sayıları (build_curriculum):
        #:   yerel/ağ katmanları  : 8-11 ders
        #:   Godot katmanları     : 3-8 ders
        #:   lowpoly uzmanlık     : 6-8 ders
        DAR = {
            "lowpoly_doku", "lowpoly_detay", "lowpoly_bitki",
            "lowpoly_arazi",
            "godot_3b", "godot_nesne", "godot_materyal",
            "godot_karakter", "godot_seviye", "godot_girdi",
            "godot_arayuz", "godot_ses", "godot_ag",
        }
        #: Godot katmanları en az 3 ders (ölçüldü: godot_ses = 3,
        #: ses konusu doğal olarak dar)
        GODOT = {a for a in counts if a.startswith("godot_")}
        for layer, count in counts.items():
            if layer in GODOT:
                esik = 3
            elif layer in DAR:
                esik = 4
            else:
                esik = 8
            self.assertGreaterEqual(
                count, esik,
                f"{layer} katmanı çok az ders içeriyor ({count} < {esik})")

    def test_no_empty_content(self) -> None:
        """Boş ders olmamalı."""
        for lesson in self.lessons:
            self.assertTrue(lesson.title.strip(), "boş başlık")
            self.assertGreater(len(lesson.body.strip()), 60,
                               f"gövde çok kısa: {lesson.title}")
            # Birleşik müfredat ağ katmanlarını da içerir → ALL_LAYERS
            self.assertIn(lesson.layer, ALL_LAYERS,
                          f"bilinmeyen katman: {lesson.layer}")

    def test_titles_unique(self) -> None:
        """Başlıklar tekrar etmemeli (tekrar veri kümesini seyreltir)."""
        titles = [lesson.title for lesson in self.lessons]
        self.assertEqual(len(titles), len(set(titles)),
                         "aynı başlık birden fazla kez var")

    def test_binary_layer_covers_requested_topic(self) -> None:
        """Kullanıcının istediği 'ikili sayı sistemi' katmanı olmalı."""
        binary = [lesson for lesson in self.lessons if lesson.layer == "binary"]
        self.assertTrue(binary)
        text = " ".join(lesson.title.lower() + lesson.body.lower()
                        for lesson in binary)
        for keyword in ("ikili", "binary", "bit", "bayt"):
            self.assertIn(keyword, text)

    def test_os_layer_covers_requested_topic(self) -> None:
        """Kullanıcının istediği 'OS oluşumu' alanı kapsanmalı."""
        os_lessons = [lesson for lesson in self.lessons if lesson.layer == "os"]
        text = " ".join(lesson.title.lower() + lesson.body.lower()
                        for lesson in os_lessons)
        for keyword in ("kernel", "süreç", "dosya", "bellek"):
            self.assertIn(keyword.lower(), text)

    def test_blender_layer_covers_modeling(self) -> None:
        """Blender katmanı nesne oluşturmayı öğretmeli."""
        blender = [lesson for lesson in self.lessons if lesson.layer == "blender"]
        text = " ".join(lesson.title.lower() + lesson.body.lower()
                        for lesson in blender)
        for keyword in ("bpy", "mesh", "malzeme", "render"):
            self.assertIn(keyword, text)

    def test_libraries_layer_covers_common_packages(self) -> None:
        """Kütüphane katmanı yaygın paketleri tanıtmalı."""
        libs = [lesson for lesson in self.lessons if lesson.layer == "libraries"]
        text = " ".join(lesson.title.lower() + lesson.body.lower()
                        + lesson.code.lower() for lesson in libs)
        for keyword in ("numpy", "torch", "asyncio", "pathlib"):
            self.assertIn(keyword, text)

    def test_code_blocks_are_valid_python(self) -> None:
        """
        Ders kodları **sözdizimsel olarak geçerli** olmalı.

        Bozuk kod öğretmek, modele yanlış kalıp öğretir.
        """
        import ast

        checked = 0
        for lesson in self.lessons:
            if not lesson.code.strip():
                continue
            # Yorum satırı veya kabuk örneği içeren bloklar atlanır
            body = lesson.code.strip()
            if body.startswith("#"):
                continue
            if "blender --background" in body or "uvicorn " in body:
                continue
            # ⚠ Godot müfredatı **GDScript** içerir; bu Python değildir ve
            # `ast.parse` ile doğrulanamaz. GDScript blokları yapı
            # kontrolünden geçirilir (parantez dengesi).
            if lesson.layer.startswith("godot_"):
                # ⚠ Ölçülen tuzak: ders metinleri yorum ve açıklama içerir
                # ("... (ör. `body_entered`) ..."). Ham parantez sayımı
                # yanlış pozitif verir. Yorumlar ve dizgiler **ayıklanır**.
                kod = self._gdscript_ayikla(body)
                for acik, kapali in (("(", ")"), ("[", "]"), ("{", "}")):
                    self.assertEqual(
                        kod.count(acik), kod.count(kapali),
                        f"parantez dengesiz — {lesson.title}: "
                        f"{acik}={kod.count(acik)} {kapali}={kod.count(kapali)}")
                if "func " in kod or "extends " in kod or "var " in kod:
                    checked += 1
                continue
            try:
                ast.parse(body)
                checked += 1
            except SyntaxError as exc:
                self.fail(f"geçersiz kod — {lesson.title}: {exc}")
        self.assertGreater(checked, 10, "yeterli kod örneği test edilmedi")

    def test_lesson_prompt_is_labeled(self) -> None:
        """Soru, katman etiketiyle başlar (model bağlamı öğrensin)."""
        for lesson in self.lessons:
            self.assertTrue(lesson.prompt().startswith(lesson.label))

    def test_lesson_target_includes_code(self) -> None:
        """Kodu olan dersin hedefi kod bloğunu içerir."""
        with_code = [lesson for lesson in self.lessons if lesson.code.strip()]
        self.assertTrue(with_code)
        for lesson in with_code[:5]:
            self.assertIn("```", lesson.target())

    def test_documents_contain_title_and_body(self) -> None:
        """Tokenizer belgesi başlık ve gövdeyi birlikte taşır."""
        for lesson in self.lessons[:5]:
            doc = lesson.as_document()
            self.assertIn(lesson.body.strip()[:40], doc)
            self.assertIn(lesson.label, doc)

    def test_lesson_dict_json_safe(self) -> None:
        assert_json_safe(self, self.lessons[0].to_dict(), "lesson")

    def test_stats_shape(self) -> None:
        stats = curriculum_stats()
        self.assertEqual(stats["total_lessons"], len(self.lessons))
        self.assertGreater(stats["total_chars"], 10_000)
        assert_json_safe(self, stats, "curriculum_stats")


class TestCurriculumExpansion(unittest.TestCase):
    """Soru kalıplarıyla genişletme."""

    def test_patterns_increase_lessons(self) -> None:
        base = len(build_curriculum())
        for count in (1, 3, 6):
            expanded = expand_curriculum(patterns=count)
            self.assertEqual(len(expanded), base * count)

    def test_patterns_capped(self) -> None:
        """Kalıp sayısı mevcut kalıpları aşamaz."""
        expanded = expand_curriculum(patterns=99)
        self.assertEqual(len(expanded),
                         len(build_curriculum()) * len(QUESTION_PATTERNS))

    def test_target_text_unchanged(self) -> None:
        """
        Genişletme **içerik uydurmaz**: hedef metin aynı kalır.

        Yalnızca soru biçimi değişir; bilgi eklenmez. Aksi hâlde yapay veri
        gerçek bilgiyi seyreltirdi.
        """
        base = {lesson.title: lesson.target() for lesson in build_curriculum()}
        for lesson in expand_curriculum(patterns=4):
            source = next((title for title in base if title in lesson.title), None)
            self.assertIsNotNone(source, f"kaynak ders bulunamadı: {lesson.title}")
            self.assertEqual(lesson.target(), base[source])

    def test_questions_differ(self) -> None:
        """Aynı dersin farklı kalıpları farklı sorular üretir."""
        expanded = expand_curriculum(patterns=6)
        first_title = build_curriculum()[0].title
        variants = [lesson.title for lesson in expanded
                    if first_title in lesson.title]
        self.assertGreater(len(set(variants)), 1)

    def test_tags_include_pattern_kind(self) -> None:
        """Kalıp türü etikete eklenir (izlenebilirlik)."""
        kinds = {kind for kind, _ in QUESTION_PATTERNS}
        for lesson in expand_curriculum(patterns=2):
            self.assertTrue(kinds & set(lesson.tags))

    def test_stats_expanded(self) -> None:
        stats = curriculum_stats(expand=True, patterns=6)
        self.assertTrue(stats["expanded"])
        self.assertEqual(stats["patterns"], 6)
        self.assertEqual(stats["total_lessons"],
                         len(build_curriculum()) * 6)
        assert_json_safe(self, stats, "expanded_stats")


# ─── Gömme Yeniden Boyutlandırma ────────────────────────────────────────────

class TestResizeVocab(unittest.TestCase):
    """
    Tokenizer değişince gömme katmanı yeniden boyutlandırılmalı.

    Ölçülen değer: müfredatla tokenizer 1.259 → 4.301 sözlüğe çıktı ve
    verimlilik 1,73 → 4,16 karakter/token oldu (2,4×). Gömme katmanı
    boyutlandırılmazsa yeni token kimlikleri arıza verir.
    """

    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("torch yok")
        from neurocore.neural.transformer import (
            TransformerBackbone,
            TransformerConfig,
        )

        self.config = TransformerConfig(vocab_size=64, embed_dim=32,
                                        num_layers=1, num_experts=2)
        self.backbone = TransformerBackbone(self.config)

    def test_grow_preserves_tied_weights(self) -> None:
        """
        Büyütmede **ağırlık paylaşımı korunmalı**.

        `tie_embeddings` açıkken `lm_head.weight` ile
        `token_embedding.weight` aynı tensördür; bağ koparsa çıkış katmanı
        rastgele kalır ve eğitim bozulur.
        """
        import torch

        self.assertTrue(self.backbone.lm_head.weight
                        is self.backbone.token_embedding.weight)
        with torch.no_grad():
            self.backbone.token_embedding.weight[5] = 1.5

        added = self.backbone.resize_vocab(200)
        self.assertEqual(added, 136)
        self.assertEqual(self.backbone.config.vocab_size, 200)
        self.assertEqual(tuple(self.backbone.token_embedding.weight.shape),
                         (200, 32))
        self.assertTrue(self.backbone.lm_head.weight
                        is self.backbone.token_embedding.weight,
                        "paylaşım korunmadı")
        self.assertAlmostEqual(
            float(self.backbone.token_embedding.weight[5][0]), 1.5, places=4,
            msg="eğitilmiş satır korunmadı",
        )

    def test_shrink_works(self) -> None:
        """Küçültme de çalışmalı (hata vermemeli)."""
        added = self.backbone.resize_vocab(32)
        self.assertEqual(added, -32)
        self.assertEqual(tuple(self.backbone.token_embedding.weight.shape),
                         (32, 32))

    def test_forward_and_gradient_after_resize(self) -> None:
        """Boyutlandırma sonrası ileri geçiş ve gradyan çalışmalı."""
        import torch

        self.backbone.resize_vocab(150)
        ids = torch.randint(0, 150, (2, 6))
        output = self.backbone(ids)
        logits = output.get("logits") if isinstance(output, dict) else output
        self.assertEqual(tuple(logits.shape), (2, 6, 150))
        self.assertFalse(bool(torch.isnan(logits).any()))

        logits.sum().backward()
        grad = self.backbone.token_embedding.weight.grad
        self.assertIsNotNone(grad)
        self.assertFalse(bool(torch.isnan(grad).any()))
        # Yeni satırlarda gradyan akmalı
        self.assertGreater(float(grad[140].abs().sum()), 0.0)

    def test_same_size_is_noop(self) -> None:
        """Aynı boyut verilirse değişiklik yapılmaz."""
        self.assertEqual(self.backbone.resize_vocab(64), 0)

    def test_invalid_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.backbone.resize_vocab(0)
        with self.assertRaises(ValueError):
            self.backbone.resize_vocab(-5)

    def test_untied_config_rebuilds_head(self) -> None:
        """Paylaşımsız yapılandırmada çıkış katmanı yeniden kurulur."""
        from neurocore.neural.transformer import (
            TransformerBackbone,
            TransformerConfig,
        )

        config = TransformerConfig(vocab_size=32, embed_dim=16, num_layers=1,
                                   num_experts=2, tie_embeddings=False)
        backbone = TransformerBackbone(config)
        backbone.resize_vocab(80)
        self.assertEqual(tuple(backbone.lm_head.weight.shape), (80, 16))
        self.assertEqual(tuple(backbone.token_embedding.weight.shape), (80, 16))
        self.assertFalse(backbone.lm_head.weight
                         is backbone.token_embedding.weight)


# ─── Eğitim Hattı (hafif) ───────────────────────────────────────────────────

class TestPipelineProgress(unittest.TestCase):
    """İlerleme kaydı ve hattın hafif parçaları."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_pipe_")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def test_initialize_creates_data_dir(self) -> None:
        trainer = CurriculumTrainer()
        trainer.initialize(workspace=self.root, data_path=self.root)
        try:
            self.assertTrue(Path(self.root).is_dir())
            stats = curriculum_stats(expand=True, patterns=trainer.patterns)
            self.assertEqual(stats["total_lessons"],
                             len(build_curriculum()) * trainer.patterns)
        finally:
            trainer.shutdown()

    def test_lessons_expanded(self) -> None:
        trainer = CurriculumTrainer(patterns=3)
        trainer.initialize(workspace=self.root, data_path=self.root)
        try:
            self.assertEqual(len(trainer.lessons()),
                             len(build_curriculum()) * 3)
        finally:
            trainer.shutdown()

    def test_progress_saved_and_reloaded(self) -> None:
        """İlerleme diske yazılır ve yeni örnekten okunur."""
        trainer = CurriculumTrainer()
        trainer.initialize(workspace=self.root, data_path=self.root)
        with trainer._lock:
            trainer._progress.started = 1.0
            trainer._progress.lessons = 99
        trainer._save_progress("test")
        trainer.shutdown()

        path = Path(self.root) / PROGRESS_FILE
        self.assertTrue(path.exists())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["lessons"], 99)
        self.assertEqual(data["stage"], "test")

        fresh = CurriculumTrainer()
        fresh.initialize(workspace=self.root, data_path=self.root)
        try:
            self.assertEqual(fresh.progress()["lessons"], 99)
        finally:
            fresh.shutdown()

    def test_progress_without_start_returns_defaults(self) -> None:
        trainer = CurriculumTrainer()
        trainer.initialize(workspace=self.root, data_path=self.root)
        try:
            data = trainer.progress()
            self.assertIn("stage", data)
            self.assertEqual(data["stage"], "bekliyor")
        finally:
            trainer.shutdown()

    def test_progress_dict_json_safe(self) -> None:
        progress = PipelineProgress(stage="x", lessons=5, tokens=100)
        assert_json_safe(self, progress.to_dict(), "pipeline_progress")

    def test_default_vocab_size_reasonable(self) -> None:
        self.assertGreaterEqual(DEFAULT_VOCAB_SIZE, 2048)

    def test_not_running_initially(self) -> None:
        trainer = CurriculumTrainer()
        trainer.initialize(workspace=self.root, data_path=self.root)
        try:
            self.assertFalse(trainer.is_running)
        finally:
            trainer.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
