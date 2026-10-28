"""
Akıl Yürütme ve Ölçekli Veri Testleri
======================================
İki modülü kapsar:

* `neurocore.neural.reasoning` — düşünme biçimleri ve iz yapısı
* `neurocore.neural.scale`     — şablon tabanlı veri çoğaltma

Neden bu testler önemli?
    Ölçekli veri **şablonla** üretilir; şablonların bilgi uydurmadığını,
    yapıyı bozmadığını ve gerçekten ölçek sağladığını doğrulamak gerekir.
    Uydurma bilgi modele zehir olur; bu yüzden "hedef metin dersin kendi
    içeriğinden türemeli" kuralı test edilir.
"""

from __future__ import annotations

import re
import unittest
from typing import Set

from tests.helpers import assert_json_safe

from neurocore.neural.curriculum import build_curriculum
from neurocore.neural.reasoning import (
    MODE_LABELS,
    THINK_PREFIX,
    THINKING_MODES,
    ReasoningLesson,
    build_reasoning_curriculum,
    reasoning_stats,
)
from neurocore.neural.scale import (
    AUDIENCES,
    GeneratedSample,
    generate_samples,
    scale_report,
)


# ─── Akıl Yürütme Müfredatı ─────────────────────────────────────────────────

class TestReasoningCurriculum(unittest.TestCase):
    """Düşünme dersleri eksiksiz ve yapılı olmalı."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.lessons = build_reasoning_curriculum()

    def test_all_modes_covered(self) -> None:
        """Beş düşünme biçiminin hepsinde ders olmalı."""
        mevcut = {ders.mode for ders in self.lessons}
        for mod in THINKING_MODES:
            self.assertIn(mod, mevcut, f"{mod} biçimi boş")

    def test_each_mode_has_multiple(self) -> None:
        sayim = {}
        for ders in self.lessons:
            sayim[ders.mode] = sayim.get(ders.mode, 0) + 1
        for mod, adet in sayim.items():
            self.assertGreaterEqual(adet, 3, f"{mod} çok az ders içeriyor")

    def test_prompt_uses_think_prefix(self) -> None:
        """İstemler `DÜŞÜN:` ile başlamalı (ajan bu öneki kullanır)."""
        for ders in self.lessons:
            self.assertTrue(ders.prompt().startswith(THINK_PREFIX))

    def test_trace_is_structured(self) -> None:
        """
        Her iz **yapılı** olmalı: sıralı adımlar, karşılaştırma veya teşhis.

        Neden? Modelin öğrendiği şey yalnızca cevap değil, cevaba giden yol.
        Yapısız düz metin, "düşünme" öğretmez. Üç geçerli biçim vardır:

            * `ADIM 1, ADIM 2, …`         — sıralı akıl yürütme
            * `SEÇENEK A / SEÇENEK B`     — karşılaştırmalı akıl yürütme
            * `BELİRTİ / OLASI NEDEN`     — teşhis akıl yürütmesi
        """
        desenler = (r"ADIM\s*\d", r"SEÇENEK\s*[AB]", r"BELİRTİ",
                    r"OLASI NEDEN", r"VERİ\s*:", r"YORUM\s*:")
        for ders in self.lessons:
            iz = ders.target()
            self.assertGreater(len(iz), 120, f"iz çok kısa: {ders.problem[:40]}")
            self.assertTrue(
                any(re.search(desen, iz) for desen in desenler),
                f"yapı yok (adım/seçenek/teşhis/yorum): {ders.problem[:40]}",
            )

    def test_trace_has_decision_or_verdict(self) -> None:
        """İz bir karar/sonuç içermeli (yalnızca liste değil)."""
        for ders in self.lessons:
            iz = ders.target().upper()
            self.assertTrue(
                "KARAR" in iz or "ÇÖZÜM" in iz or "DERS" in iz
                or "NE YAPMALIYIM" in iz or "YORUM" in iz,
                f"karar/sonuç yok: {ders.problem[:40]}",
            )

    def test_modes_are_distinct(self) -> None:
        """Aynı dersin farklı biçimleri farklı izler üretmeli."""
        problemler: Set[str] = set()
        for ders in self.lessons:
            anahtar = f"{ders.mode}:{ders.problem}"
            self.assertNotIn(anahtar, problemler, "aynı biçim+problem tekrar")
            problemler.add(anahtar)

    def test_labels_are_turkish(self) -> None:
        """Etiketler Türkçe olmalı (kullanıcı arayüzü için)."""
        for mod in THINKING_MODES:
            etiket = MODE_LABELS[mod]
            self.assertTrue(etiket and etiket[0].isupper())
            self.assertNotEqual(etiket, mod)

    def test_document_contains_label_and_trace(self) -> None:
        for ders in self.lessons[:5]:
            belge = ders.as_document()
            self.assertIn(ders.label, belge)
            self.assertIn(ders.trace.strip()[:40], belge)

    def test_dict_json_safe(self) -> None:
        assert_json_safe(self, self.lessons[0].to_dict(), "reasoning_lesson")

    def test_stats_shape(self) -> None:
        stats = reasoning_stats()
        self.assertEqual(stats["total_lessons"], len(self.lessons))
        self.assertGreater(stats["total_chars"], 5_000)
        assert_json_safe(self, stats, "reasoning_stats")

    def test_mode_filter(self) -> None:
        """Belirli biçimler istenebilmeli."""
        sadece = build_reasoning_curriculum(["decide"])
        self.assertTrue(sadece)
        self.assertTrue(all(d.mode == "decide" for d in sadece))

    def test_covers_measured_topics(self) -> None:
        """
        İzler, ölçümde **gerçekten yaşanan** sorunları kapsamalı.

        Bu dersler oturum boyunca karşılaşılan gerçek hatalardan türetildi;
        kaybolurlarsa öğrenme değeri düşer.
        """
        metin = " ".join(d.problem + d.trace for d in self.lessons).lower()
        for anahtar in ("unscale_", "vram", "veri ölçeği", "403",
                        "bpy.data.objects", "24 adım"):
            self.assertIn(anahtar.lower(), metin,
                          f"ölçülen konu eksik: {anahtar}")


# ─── Ölçekli Veri Üretimi ───────────────────────────────────────────────────

class TestScaleGeneration(unittest.TestCase):
    """Şablon tabanlı çoğaltma."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temel = generate_samples(audiences=1, knowledge_patterns=4)

    def test_produces_many_samples(self) -> None:
        """Üretim gerçekten ölçek sağlamalı."""
        ornekler = generate_samples(audiences=3, knowledge_patterns=20)
        self.assertGreater(len(ornekler), 1500)

    def test_audience_multiplies_thinking(self) -> None:
        """Kitle sayısı yalnızca düşünme örneklerini çoğaltmalı."""
        bir = generate_samples(audiences=1, knowledge_patterns=20)
        uc = generate_samples(audiences=3, knowledge_patterns=20)
        self.assertGreater(len(uc), len(bir))
        bilgi_bir = sum(1 for o in bir if o.kind == "bilgi")
        bilgi_uc = sum(1 for o in uc if o.kind == "bilgi")
        self.assertEqual(bilgi_bir, bilgi_uc,
                         "bilgi örnekleri kitleyle çoğalmamalı")

    def test_all_layers_present(self) -> None:
        """Tüm müfredat katmanları temsil edilmeli."""
        katmanlar = {o.layer for o in generate_samples(audiences=2,
                                                       knowledge_patterns=8)}
        for beklenen in ("binary", "hardware", "os", "languages",
                         "libraries", "blender"):
            self.assertIn(beklenen, katmanlar)

    def test_thinking_kinds_present(self) -> None:
        """Her düşünme biçimi için örnek üretilmeli."""
        turler = {o.kind for o in generate_samples(audiences=1,
                                                   knowledge_patterns=4)}
        for mod in THINKING_MODES:
            self.assertIn(f"düşünme:{mod}", turler)

    def test_no_empty_fields(self) -> None:
        for ornek in self.temel:
            self.assertTrue(ornek.prompt.strip())
            self.assertGreater(len(ornek.target.strip()), 50)

    def test_target_derives_from_lesson(self) -> None:
        """
        **Uydurma yok**: düşünme izleri dersin kendi içeriğinden türemeli.

        Kanıt: izlerde ders gövdesinin cümleleri (en az 40 karakterlik
        parçalar) geçmeli. Şablonlar yalnızca **sunum biçimini** değiştirir,
        bilgi eklemez — aksi hâlde yapay veri modele zehir olurdu.
        """
        dersler = build_curriculum()
        kontrol = 0
        for ornek in self.temel:
            if not ornek.kind.startswith("düşünme:"):
                continue
            # İstemden başlığı çıkar: "DÜŞÜN: <başlık> — ..." veya "<başlık> ..."
            govde = ornek.prompt.replace(THINK_PREFIX, "", 1).strip()
            eslesen = None
            for ders in dersler:
                if govde.startswith(ders.title):
                    eslesen = ders
                    break
            if eslesen is None:
                continue
            cumleler = [c.strip() for c in
                        re.split(r"(?<=[.!?])\s+", " ".join(eslesen.body.split()))
                        if len(c.strip()) > 25]
            if not cumleler:
                continue
            kontrol += 1
            bulundu = any(c[:25] in ornek.target for c in cumleler)
            self.assertTrue(bulundu,
                            f"iz ders içeriğinden türememiş: {ornek.prompt[:50]}")
        self.assertGreater(kontrol, 0, "hiç düşünme örneği kontrol edilmedi")

    def test_thinking_disabled(self) -> None:
        """`thinking=False` yalnızca bilgi örnekleri vermeli."""
        ornekler = generate_samples(audiences=2, thinking=False,
                                    knowledge_patterns=4)
        self.assertTrue(all(o.kind == "bilgi" for o in ornekler))

    def test_sample_dict_json_safe(self) -> None:
        assert_json_safe(self, self.temel[0].to_dict(), "generated_sample")

    def test_deterministic(self) -> None:
        """Aynı ayarlarla aynı çıktı (tekrarlanabilirlik)."""
        a = generate_samples(audiences=2, knowledge_patterns=6)
        b = generate_samples(audiences=2, knowledge_patterns=6)
        self.assertEqual(len(a), len(b))
        self.assertEqual(a[0].prompt, b[0].prompt)
        self.assertEqual(a[-1].target, b[-1].target)

    def test_tags_present(self) -> None:
        for ornek in self.temel[:20]:
            self.assertTrue(ornek.tags, "etiketsiz örnek")


class TestScaleReport(unittest.TestCase):
    """Ölçek raporu doğru sayıları vermeli."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rapor = scale_report(audiences=2, knowledge_patterns=6)

    def test_report_fields(self) -> None:
        for alan in ("samples", "chars", "tokens", "vocab",
                     "chars_per_token", "per_kind"):
            self.assertIn(alan, self.rapor)

    def test_token_count_positive(self) -> None:
        self.assertGreater(self.rapor["tokens"], 10_000)

    def test_tokenizer_efficiency_reasonable(self) -> None:
        """Eğitilen tokenizer teknik metinde makul verim vermeli."""
        self.assertGreater(self.rapor["chars_per_token"], 2.0)

    def test_vocab_within_target(self) -> None:
        self.assertLessEqual(self.rapor["vocab"], 8000)

    def test_per_kind_sums_to_total(self) -> None:
        toplam = sum(self.rapor["per_kind"].values())
        self.assertEqual(toplam, self.rapor["samples"])

    def test_report_json_safe(self) -> None:
        assert_json_safe(self, self.rapor, "scale_report")

    def test_more_patterns_more_data(self) -> None:
        """Daha çok kalıp → daha çok örnek."""
        az = scale_report(audiences=1, knowledge_patterns=2)
        cok = scale_report(audiences=1, knowledge_patterns=8)
        self.assertGreater(cok["samples"], az["samples"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
