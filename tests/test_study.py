"""
Kendi Kendini Geliştirme ve Web Araçları Testleri
==================================================
İki modülü kapsar:

* `neurocore.agent.web`   — sayfa indirme, HTML temizleme, SSRF koruması
* `neurocore.agent.study` — öğretmen–öğrenci öğrenme döngüsü

Ağ erişimi gerektiren testler **atlanır** (ortam yoksa); ağsız mantık
(SSRF, HTML temizleme, soru teknikleri, kayıt biçimi, zehirlenme koruması)
her zaman sınanır.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List

from tests.helpers import assert_json_safe

from neurocore.agent.study import (
    ANSWER_PROMPT,
    CRITIQUE_PROMPT,
    QUESTION_PROMPT,
    TECHNIQUE_CYCLE,
    QuestionTechnique,
    SelfStudy,
    StudyRound,
)
from neurocore.agent.web import (
    WebPage,
    extract_links,
    extract_title,
    html_to_text,
    is_public_url,
    _decode_ddg_url,
    _parse_ddg_lite,
)


def _network_available() -> bool:
    """İnternet erişimi var mı? (testleri atlamak için)"""
    import socket

    try:
        socket.create_connection(("1.1.1.1", 53), timeout=3).close()
        return True
    except OSError:
        return False


# ─── SSRF Koruması ──────────────────────────────────────────────────────────

class TestUrlSafety(unittest.TestCase):
    """
    Ajan iç ağa istek atmamalı (SSRF).

    `http://localhost:8791/api/...` gibi bir adres, uygulamanın kendi API'sini
    çağırıp durum değiştirebilirdi.
    """

    def test_localhost_blocked(self) -> None:
        for url in ("http://localhost:8791/", "http://127.0.0.1/admin",
                    "http://0.0.0.0/", "http://[::1]/"):
            allowed, reason = is_public_url(url)
            self.assertFalse(allowed, f"{url} engellenmeliydi")
            self.assertTrue(reason)

    def test_private_ranges_blocked(self) -> None:
        for url in ("http://192.168.1.1/", "http://10.0.0.5/",
                    "http://172.16.0.1/", "http://169.254.1.1/"):
            allowed, _ = is_public_url(url)
            self.assertFalse(allowed, f"{url} özel ağ — engellenmeliydi")

    def test_local_domains_blocked(self) -> None:
        for url in ("http://example.local/", "http://server.internal/"):
            allowed, _ = is_public_url(url)
            self.assertFalse(allowed)

    def test_non_http_schemes_blocked(self) -> None:
        for url in ("file:///C:/Windows/win.ini", "ftp://example.com/x",
                    "javascript:alert(1)", "data:text/html,x"):
            allowed, _ = is_public_url(url)
            self.assertFalse(allowed, f"{url} şeması reddedilmeliydi")

    def test_public_urls_allowed(self) -> None:
        for url in ("https://example.com/", "https://docs.python.org/3/",
                    "http://realpython.com/article"):
            allowed, reason = is_public_url(url)
            self.assertTrue(allowed, f"{url} izinli olmalıydı ({reason})")

    def test_empty_and_malformed(self) -> None:
        for url in ("", "   ", "http://", "sadece-metin"):
            allowed, _ = is_public_url(url)
            self.assertFalse(allowed)


# ─── HTML Temizleme ─────────────────────────────────────────────────────────

class TestHtmlCleaning(unittest.TestCase):
    """Ham HTML, öğretmen modele verilecek **temiz metne** çevrilmeli."""

    MARKUP = """<html><head><title>Başlık</title>
    <style>.x{color:red}</style><script>var gizli=1;</script></head>
    <body><nav>Menü</nav>
    <article><h1>Ana Başlık</h1><p>Paragraf &amp; özel &lt;karakter&gt;.</p>
    <ul><li>bir</li><li>iki</li></ul>
    <a href="/devam">Devam</a></article>
    <footer>Alt bilgi</footer></body></html>"""

    def test_scripts_and_styles_removed(self) -> None:
        text = html_to_text(self.MARKUP)
        self.assertNotIn("gizli", text)
        self.assertNotIn("color:red", text)

    def test_nav_and_footer_removed(self) -> None:
        text = html_to_text(self.MARKUP)
        self.assertNotIn("Menü", text)
        self.assertNotIn("Alt bilgi", text)

    def test_main_content_kept(self) -> None:
        text = html_to_text(self.MARKUP)
        self.assertIn("Ana Başlık", text)
        self.assertIn("bir", text)
        self.assertIn("iki", text)

    def test_entities_decoded(self) -> None:
        text = html_to_text(self.MARKUP)
        self.assertIn("&", text)
        self.assertIn("<karakter>", text)

    def test_tags_stripped(self) -> None:
        text = html_to_text(self.MARKUP)
        self.assertNotIn("<p>", text)
        self.assertNotIn("</h1>", text)

    def test_title_extracted(self) -> None:
        self.assertEqual(extract_title(self.MARKUP), "Başlık")

    def test_links_made_absolute(self) -> None:
        links = extract_links(self.MARKUP, "https://site.example/sayfa")
        self.assertEqual(links, ["https://site.example/devam"])

    def test_empty_input(self) -> None:
        self.assertEqual(html_to_text(""), "")
        self.assertEqual(extract_title(""), "")
        self.assertEqual(extract_links("", "https://x.example"), [])

    def test_whitespace_collapsed(self) -> None:
        text = html_to_text("<p>a</p>\n\n\n\n<p>b</p>")
        self.assertNotIn("\n\n\n", text)


class TestDuckDuckGoParsing(unittest.TestCase):
    """DDG Lite sonuç ayrıştırması (yapı değişirse test uyarır)."""

    MARKUP = """
    <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&amp;rut=abc"
       class='result-link'>Python Docs</a>
    <td class='result-snippet'>Resmî <b>Python</b> dokümantasyonu.</td>
    <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpeps.python.org%2F"
       class='result-link'>PEP Index</a>
    <td class='result-snippet'>Python Enhancement Proposals</td>
    """

    def test_results_parsed(self) -> None:
        results = _parse_ddg_lite(self.MARKUP, limit=5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["url"], "https://docs.python.org/3/")
        self.assertEqual(results[0]["title"], "Python Docs")

    def test_snippets_cleaned(self) -> None:
        results = _parse_ddg_lite(self.MARKUP, limit=5)
        self.assertEqual(results[0]["snippet"], "Resmî Python dokümantasyonu.")
        self.assertNotIn("<b>", results[0]["snippet"])

    def test_limit_respected(self) -> None:
        self.assertEqual(len(_parse_ddg_lite(self.MARKUP, limit=1)), 1)

    def test_redirect_url_decoded(self) -> None:
        target = _decode_ddg_url(
            "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fyol&rut=x"
        )
        self.assertEqual(target, "https://example.com/yol")

    def test_empty_markup(self) -> None:
        self.assertEqual(_parse_ddg_lite("", limit=5), [])

    def test_webpage_dict(self) -> None:
        page = WebPage(url="https://x.example", status=200, text="içerik",
                       title="T")
        data = page.to_dict()
        self.assertTrue(data["ok"])
        assert_json_safe(self, data, "webpage")


# ─── Soru Sorma Teknikleri ──────────────────────────────────────────────────

class TestQuestionTechniques(unittest.TestCase):
    """Beş teknik tanımlı ve her birinin yönergesi olmalı."""

    def test_all_techniques_have_instruction(self) -> None:
        for technique in QuestionTechnique:
            self.assertTrue(technique.instruction, f"{technique} yönergesiz")
            self.assertTrue(technique.label, f"{technique} etiketsiz")

    def test_cycle_covers_multiple_techniques(self) -> None:
        self.assertGreaterEqual(len(set(TECHNIQUE_CYCLE)), 4)

    def test_techniques_distinct(self) -> None:
        instructions = {item.instruction for item in QuestionTechnique}
        self.assertEqual(len(instructions), len(list(QuestionTechnique)))

    def test_prompts_contain_placeholders(self) -> None:
        self.assertIn("{context}", QUESTION_PROMPT)
        self.assertIn("{topic}", QUESTION_PROMPT)
        self.assertIn("{question}", ANSWER_PROMPT)
        self.assertIn("{sources}", ANSWER_PROMPT)
        self.assertIn("{reference}", CRITIQUE_PROMPT)
        self.assertIn("{student}", CRITIQUE_PROMPT)

    def test_source_citation_required(self) -> None:
        """Cevap istemi atıf istemeli (uydurmayı önler)."""
        self.assertIn("[1]", ANSWER_PROMPT)


# ─── Öğrenme Döngüsü (ağsız) ────────────────────────────────────────────────

@dataclass
class FakeMessage:
    """Sahte LLM mesajı."""
    role: str
    content: str = ""


@dataclass
class FakeResponse:
    """Sahte LLM yanıtı."""
    ok: bool = True
    text: str = ""
    tool_calls: List[Any] = field(default_factory=list)
    total_tokens: int = 10
    error: str = ""
    finish_reason: str = "stop"


class FakeTeacher:
    """
    Sahte öğretmen model.

    Test edilen şey modelin zekâsı değil, **döngünün doğru kurulması**:
    kaynak gerçekten indiriliyor mu, not yazılıyor mu, hata yönetiliyor mu.
    """

    def __init__(self, *, question: str = "", answer: str = "",
                 critique: str = "", fail: bool = False) -> None:
        self.question = question
        self.answer = answer
        self.critique = critique
        self.fail = fail
        self.calls: List[str] = []

    def status(self) -> Dict[str, Any]:
        return {"active": "Sahte", "active_model": "fake-1"}

    def chat(self, messages: Any, **kwargs: Any) -> FakeResponse:
        prompt = messages[0].content if messages else ""
        self.calls.append(prompt)
        if self.fail:
            return FakeResponse(ok=False, error="sahte hata")
        if "tek bir soru" in prompt.lower():
            return FakeResponse(text=self.question)
        if "kaynaklara dayanarak" in prompt.lower():
            return FakeResponse(text=self.answer)
        if "değerlendiriyorsun" in prompt.lower():
            return FakeResponse(text=self.critique)
        return FakeResponse(text=self.answer)

    def chat_with_budget(self, messages: Any, **kwargs: Any) -> FakeResponse:
        return self.chat(messages, **kwargs)


class FakeKnowledge:
    """Sahte kasa katmanı (yazılan notları toplar)."""

    def __init__(self) -> None:
        self.notes: List[Dict[str, Any]] = []

    def write(self, title: str, body: str, **kwargs: Any) -> Dict[str, Any]:
        self.notes.append({"title": title, "body": body, **kwargs})
        return {"id": f"fake_{len(self.notes)}"}

    def get_stats(self) -> Dict[str, Any]:
        return {"notes": len(self.notes)}

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        return [{"title": note["title"]} for note in self.notes[:limit]]


class TestSelfStudy(unittest.TestCase):
    """Öğrenme döngüsünün kurulumu, akışı ve hata yönetimi."""

    def setUp(self) -> None:
        self.workspace = tempfile.mkdtemp(prefix="nc_study_")
        self.knowledge = FakeKnowledge()

    def _study(self, teacher: Any, **kwargs: Any) -> SelfStudy:
        study = SelfStudy(**kwargs)
        study.initialize(llm=teacher, knowledge=self.knowledge,
                         workspace=self.workspace)
        return study

    def test_no_teacher_returns_error(self) -> None:
        """Öğretmen yoksa tur anlaşılır hata döner (istisna değil)."""
        study = SelfStudy()
        study.initialize()
        round_ = study.study("konu")
        self.assertFalse(round_.ok)
        self.assertTrue(round_.error)

    def test_no_topic_returns_error(self) -> None:
        """Konu bulunamazsa tur başarısız olur."""
        study = self._study(FakeTeacher(question="Soru?"))
        round_ = study.study()
        self.assertFalse(round_.ok)
        self.assertIn("konu", round_.error)

    def test_round_uses_technique(self) -> None:
        """Her tur bir soru tekniği kullanır."""
        study = self._study(FakeTeacher(question="Örnek soru metni?"),
                            topics=["konu A"])
        round_ = study.study()
        self.assertTrue(round_.technique)
        self.assertIn(round_.technique, [item.value for item in QuestionTechnique])

    def test_technique_cycles(self) -> None:
        """Teknikler döngüsel olarak değişir (tek tip soru olmasın)."""
        study = self._study(FakeTeacher(question="Soru metni burada?"))
        study._topics = [f"konu {index}" for index in range(5)]
        techniques = [study.study().technique for _ in range(5)]
        self.assertEqual(len(set(techniques)), 5, "beş farklı teknik beklenirdi")

    def test_teacher_failure_handled(self) -> None:
        """Öğretmen çağrısı başarısız olursa tur çökmez."""
        study = self._study(FakeTeacher(fail=True), topics=["konu"])
        round_ = study.study()
        self.assertFalse(round_.ok)
        self.assertTrue(round_.error)

    def test_no_verified_sources_fails(self) -> None:
        """
        Doğrulanmış kaynak yoksa tur **başarısız** olur.

        Kaynaksız cevap, öğretmenin hafızasından uydurduğu bilgi olurdu;
        kasaya bu girmemeli.
        """
        study = self._study(FakeTeacher(question="Soru?"), topics=["konu"])
        # Ağ erişimini taklit et: kaynak toplama boş dönsün
        study._gather_sources = lambda question: []      # type: ignore[assignment]
        round_ = study.study()
        self.assertFalse(round_.ok)
        self.assertIn("kaynak", round_.error.lower())

    def test_note_written_when_successful(self) -> None:
        """Kaynak ve cevap varsa kasaya not yazılır."""
        study = self._study(
            FakeTeacher(question="Soru metni?", answer="Cevap [1]"),
            topics=["konu"],
        )
        study._gather_sources = lambda question: [   # type: ignore[assignment]
            {"url": "https://docs.example/x", "title": "Kaynak", "text": "metin"}
        ]
        round_ = study.study()
        self.assertTrue(round_.ok, round_.error)
        self.assertEqual(len(self.knowledge.notes), 1)
        note = self.knowledge.notes[0]
        self.assertTrue(note["title"].startswith("Öğrenildi:"))
        self.assertIn("Soru metni?", note["body"])
        self.assertIn("https://docs.example/x", note["body"])

    def test_critique_included_when_student_answers(self) -> None:
        """Yerel model cevap verdiyse eleştiri notu da yazılır."""
        study = self._study(
            FakeTeacher(question="Soru?", answer="Doğru cevap", critique="- EKSİK: x"),
            topics=["konu"],
        )
        study._gather_sources = lambda question: [   # type: ignore[assignment]
            {"url": "https://d.example/a", "title": "K", "text": "içerik"}
        ]
        study._student_reflection = lambda question: "yanlış cevap"   # type: ignore[assignment]
        round_ = study.study()
        self.assertTrue(round_.ok)
        self.assertIn("EKSİK", round_.critique)
        self.assertIn("eksikleri", self.knowledge.notes[0]["body"])

    def test_student_answer_skipped_without_model(self) -> None:
        """Yerel model yoksa öğrenci cevabı boş kalır (tur yine başarılı)."""
        study = self._study(FakeTeacher(question="S?", answer="C"),
                            topics=["konu"])
        study._gather_sources = lambda question: [   # type: ignore[assignment]
            {"url": "https://d.example/a", "title": "K", "text": "içerik"}
        ]
        round_ = study.study()
        self.assertTrue(round_.ok)
        self.assertEqual(round_.student_answer, "")

    def test_status_shape(self) -> None:
        """Durum raporu arayüz için gerekli alanları içerir."""
        study = self._study(FakeTeacher())
        status = study.status()
        for key in ("teacher", "topics", "studying", "stats", "recent"):
            self.assertIn(key, status)
        assert_json_safe(self, status, "study_status")

    def test_round_dict_json_safe(self) -> None:
        """Tur kaydı JSON güvenli."""
        round_ = StudyRound(at=1.0, topic="t", technique="uygulama",
                            question="s?", answer="c",
                            sources=[{"url": "https://x.example", "title": "k", "text": "m"}])
        data = round_.to_dict()
        self.assertEqual(data["sources"], ["https://x.example"])
        assert_json_safe(self, data, "study_round")

    def test_worker_requires_teacher(self) -> None:
        """Öğretmen yokken arka plan işçisi başlamaz."""
        study = SelfStudy()
        study.initialize()
        self.assertFalse(study.start_worker(interval=3600))

    def test_worker_lifecycle(self) -> None:
        """İşçi başlatılıp durdurulabilir."""
        study = self._study(FakeTeacher())
        self.assertTrue(study.start_worker(interval=3600))
        self.assertIsNotNone(study._worker)
        study.stop_worker()
        self.assertIsNone(study._worker)

    def test_topics_configured(self) -> None:
        """Konular yapıcıda ve sonradan ayarlanabilir."""
        study = self._study(FakeTeacher(), topics=["a", "b"])
        self.assertEqual(study.status()["topics"], ["a", "b"])
        study._topics = ["c"]
        self.assertEqual(study.status()["topics"], ["c"])

    def test_failed_topics_are_retried(self) -> None:
        """
        Başarısız konu **atlanmaz**.

        Kullanıcının verdiği konu öğrenilemediyse sonraki turda yeniden
        denenmeli; aksi hâlde konu sessizce kaybolur.
        """
        study = self._study(FakeTeacher(fail=True), topics=["zor konu"])
        study.study()
        self.assertEqual(study._next_topic(), "zor konu")


class TestStudyNoPoisoning(unittest.TestCase):
    """
    **Zehirlenme koruması**: yerel modelin cevabı eğitim hedefi olmamalı.

    Öğretmen cevabı kasaya yazılır; yerel modelin cevabı yalnızca
    karşılaştırma ve eleştiri için kullanılır.
    """

    def test_student_answer_not_written_as_target(self) -> None:
        knowledge = FakeKnowledge()
        study = SelfStudy()
        study.initialize(llm=FakeTeacher(question="S?", answer="ÖĞRETMEN CEVABI"),
                         knowledge=knowledge)
        study._topics = ["konu"]
        study._gather_sources = lambda question: [   # type: ignore[assignment]
            {"url": "https://d.example/a", "title": "K", "text": "içerik"}
        ]
        study._student_answer = lambda question: "YEREL MODEL CEVABI"   # type: ignore[assignment]
        round_ = study.study()

        self.assertTrue(round_.ok)
        body = knowledge.notes[0]["body"]
        # Öğretmen cevabı hedef olarak yazılır
        self.assertIn("ÖĞRETMEN CEVABI", body)
        # Yerel modelin cevabı **hedef bölümünde** yer almaz; yalnızca
        # eleştiri başlığı altında bağlam olarak geçebilir.
        self.assertNotIn("## Cevap\nYEREL MODEL CEVABI", body)

    def test_note_kind_is_reference(self) -> None:
        """Öğrenilen not kaynak türünde yazılır (çözüm notlarından ayrı)."""
        knowledge = FakeKnowledge()
        study = SelfStudy()
        study.initialize(llm=FakeTeacher(question="S?", answer="C"),
                         knowledge=knowledge)
        study._topics = ["konu"]
        study._gather_sources = lambda question: [   # type: ignore[assignment]
            {"url": "https://d.example/a", "title": "K", "text": "içerik"}
        ]
        study.study()
        self.assertEqual(knowledge.notes[0]["kind"], "reference")
        self.assertIn("öğrenildi", knowledge.notes[0]["tags"])


class TestNetworkFeatures(unittest.TestCase):
    """Ağ erişimi gerektiren kontroller (erişim yoksa atlanır)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _network_available():
            raise unittest.SkipTest("ağ erişimi yok")

    def test_fetch_real_page(self) -> None:
        """Gerçek bir sayfa indirilip temizlenebilmeli."""
        from neurocore.agent.web import fetch

        page = fetch("https://example.com", timeout=20)
        if not page.ok:
            self.skipTest(f"sayfa indirilemedi: {page.error}")
        self.assertIn("Example Domain", page.title + page.text)
        self.assertGreater(len(page.text), 40)

    def test_fetch_blocks_localhost_even_here(self) -> None:
        """Yerel adres ağ olsa da reddedilmeli."""
        from neurocore.agent.web import fetch

        page = fetch("http://127.0.0.1:8791/api/state", timeout=5)
        self.assertFalse(page.ok)
        self.assertTrue(page.error)


class TestLocalQuestioning(unittest.TestCase):
    """
    **Yerel modelin soru sorması.**

    Kullanıcının istediği davranış: yerel model önce *neyi bilmediğini*
    formüle etsin, öğretmen o soruyu kaynakla yanıtlasın. Öğretmenin her
    zaman soru üretmesi, modelin soru sormayı **hiç öğrenmemesi** demekti.
    """

    def setUp(self) -> None:
        self.workspace = tempfile.mkdtemp(prefix="nc_lq_")
        self.knowledge = FakeKnowledge()

    def _study(self, *, local_question: str = "", **kwargs: Any) -> SelfStudy:
        study = SelfStudy(**kwargs)
        study.initialize(llm=FakeTeacher(question="Öğretmen sorusu?",
                                         answer="Cevap [1]"),
                         knowledge=self.knowledge, workspace=self.workspace)
        study._ask_local = lambda prompt: local_question   # type: ignore[assignment]
        study._gather_sources = lambda question: [         # type: ignore[assignment]
            {"url": "https://d.example/a", "title": "K", "text": "içerik"}
        ]
        return study

    def test_local_question_used_when_valid(self) -> None:
        """Geçerli yerel soru varsa **o kullanılır**, öğretmen sorusu değil."""
        study = self._study(local_question="Godot'ta sahne ağacına nasıl düğüm eklerim?",
                            topics=["konu"])
        round_ = study.study()
        self.assertTrue(round_.ok, round_.error)
        self.assertEqual(round_.question_source, "local")
        self.assertIn("sahne ağacına", round_.question)
        self.assertEqual(study.status()["stats"]["local_questions"], 1)

    def test_falls_back_to_teacher_when_local_silent(self) -> None:
        """Yerel model üretemezse öğretmen sorar (öğrenme durmaz)."""
        study = self._study(local_question="", topics=["konu"])
        round_ = study.study()
        self.assertTrue(round_.ok, round_.error)
        self.assertEqual(round_.question_source, "teacher")
        self.assertEqual(round_.question, "Öğretmen sorusu?")
        self.assertEqual(study.status()["stats"]["local_questions"], 0)

    def test_too_short_local_output_rejected(self) -> None:
        """Çok kısa çıktı soru sayılmaz (yerel model henüz üretemiyor)."""
        study = self._study(local_question="ne?", topics=["konu"])
        round_ = study.study()
        self.assertEqual(round_.question_source, "teacher")

    def test_non_question_rejected(self) -> None:
        """Soru biçiminde olmayan uzun metin reddedilir."""
        study = self._study(
            local_question="Bu konuda bilgi sahibi değilim ve yardım istiyorum.",
            topics=["konu"],
        )
        round_ = study.study()
        self.assertEqual(round_.question_source, "teacher")

    def test_multi_line_output_extracted(self) -> None:
        """Model başlık eklese de soru satırı bulunur."""
        study = self._study(
            local_question="Soru: Python'da asyncio.gather nasıl çalışır?\nEk açıklama.",
            topics=["konu"],
        )
        round_ = study.study()
        self.assertEqual(round_.question_source, "local")
        self.assertNotIn("Soru:", round_.question)
        self.assertIn("asyncio.gather", round_.question)

    def test_question_source_recorded_in_dict(self) -> None:
        """Tur kaydı sorunun kaynağını taşır (ilerleme ölçülebilsin)."""
        round_ = StudyRound(at=1.0, topic="t", question="s?",
                            question_source="local")
        self.assertEqual(round_.to_dict()["question_source"], "local")


class TestTeacherFallback(unittest.TestCase):
    """
    Öğretmen **yanıt vermezse diğer sağlayıcılar denenir**.

    Kullanıcının istediği davranış: bir model cevap vermezse sıradaki
    denenir; tur çökmez.
    """

    class FlakyRouter:
        """İlk N çağrıda hata veren, sonra başarılı olan sahte yönlendirici."""

        def __init__(self, *, fail_count: int = 1, switchable: bool = True) -> None:
            self.fail_count = fail_count
            self.switchable = switchable
            self.calls = 0
            self.switches = 0
            self.name = "Sahte-1"

        def status(self) -> Dict[str, Any]:
            return {"active": self.name, "active_model": "m"}

        def chat(self, messages: Any, **kwargs: Any) -> "FakeResponse":
            self.calls += 1
            if self.calls <= self.fail_count:
                return FakeResponse(ok=False, error="HTTP 429 kota doldu")
            return FakeResponse(text="Kaynağa dayalı cevap")

        def chat_with_budget(self, messages: Any, **kwargs: Any) -> "FakeResponse":
            return self.chat(messages, **kwargs)

        def failover(self, reason: str = "") -> Any:
            if not self.switchable:
                return None
            self.switches += 1
            self.name = f"Sahte-{self.switches + 1}"
            return self.name

    def test_switches_provider_on_failure(self) -> None:
        """Başarısız çağrıdan sonra **sıradaki sağlayıcı** denenir."""
        router = self.FlakyRouter(fail_count=1)
        study = SelfStudy()
        study.initialize(llm=router)
        text = study._call_teacher("soru")
        self.assertEqual(text, "Kaynağa dayalı cevap")
        self.assertEqual(router.calls, 2, "iki deneme beklenirdi")
        self.assertEqual(study.status()["stats"]["teacher_switches"], 1)

    def test_no_fallback_available_reports_failure(self) -> None:
        """Yedek yoksa tur çökmez; hata sayılır ve boş döner."""
        router = self.FlakyRouter(fail_count=99, switchable=False)
        study = SelfStudy()
        study.initialize(llm=router)
        text = study._call_teacher("soru")
        self.assertEqual(text, "")
        self.assertEqual(study.status()["stats"]["teacher_failures"], 1)

    def test_teacher_failure_fails_round_cleanly(self) -> None:
        """Öğretmen hiç yanıt veremezse tur **anlaşılır** hata döner."""
        router = self.FlakyRouter(fail_count=99, switchable=False)
        study = SelfStudy(topics=["konu"])
        study.initialize(llm=router, knowledge=FakeKnowledge())
        study._student_question = lambda topic, technique: "Soru?"  # type: ignore[assignment]
        study._gather_sources = lambda question: [   # type: ignore[assignment]
            {"url": "https://d.example/a", "title": "K", "text": "içerik"}
        ]
        round_ = study.study()
        self.assertFalse(round_.ok)
        self.assertIn("öğretmen", round_.error.lower())

    def test_all_providers_exhausted_stops(self) -> None:
        """Sonsuz döngüye girmez: en fazla 4 deneme yapılır."""
        router = self.FlakyRouter(fail_count=99, switchable=True)
        study = SelfStudy()
        study.initialize(llm=router)
        study._call_teacher("soru")
        self.assertLessEqual(router.calls, 4, "deneme sayısı sınırlı olmalı")


if __name__ == "__main__":
    unittest.main(verbosity=2)
