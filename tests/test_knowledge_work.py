"""
Bilgi İş Akışı Testleri
========================
`neurocore.agent.knowledge_work` — ajan bilmediğinde **öğrenir**.

Neden bu testler kritik?
    Kullanıcının sorusu tam buydu: "bir iş verildiğinde bilmiyorsa öğrenerek
    yapacak mı?" Bu testler dört davranışı **kanıtlar**:

        1. Kasa doluysa **öğrenmeye kalkışmaz** (boşa API harcamaz)
        2. Kasa boşsa **öğrenir** (öğretmen çağrılır)
        3. Öğretmen yoksa **dürüstçe bildirir** (uydurmaz)
        4. Öğrenme başarısızsa **tahmin yürütmez**, kullanıcıya bildirir

    Ayrıca kaynak doğrulama kuralı: kaynaksız öğrenme kasaya **yazılmaz**.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from typing import Any, Dict, List

from tests.helpers import assert_json_safe

from neurocore.agent.knowledge_work import (
    ENOUGH_RESULTS,
    KnowledgeWorkflow,
    LearnedTopic,
)
from neurocore.agent.tools import Safety


class FakeVault:
    """
    Sahte bilgi kasası — **alakalı** sonuç döndürür.

    Önemli: dönen içerik sorgunun terimlerini **içerir**, aksi hâlde
    `KnowledgeWorkflow`'un alakalılık süzgeci sonucu eler ve test yanlış
    yere başarısız olur (süzgeç doğru çalışırken).
    """

    def __init__(self, hits: int = 0, *, fail: bool = False) -> None:
        self.hits = hits
        self.fail = fail
        self.queries: List[str] = []

    def search(self, query: str, limit: int = 8, **kwargs: Any) -> List[Dict[str, Any]]:
        """Arama taklidi — sorgu terimlerini içeren sonuçlar."""
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("kasa bozuk")
        return [
            {"title": f"{query} — kasa notu {i}",
             "body": f"{query} hakkında ayrıntılı bilgi. " * 10,
             "score": 0.85}
            for i in range(self.hits)
        ]


class FakeRound:
    """Sahte öğrenme turu."""

    def __init__(self, ok: bool = True, *, sources: int = 1,
                 error: str = "") -> None:
        self.ok = ok
        self.question = "Öğretmen sorusu?"
        # Kalite kapısından geçecek kadar **uzun ve öz** cevap
        self.answer = ("Kaynağa dayalı ayrıntılı cevap [1]. Bu konuda üç "
                       "temel nokta vardır: birincisi yapılandırma, ikincisi "
                       "kullanım sırası, üçüncüsü doğrulama. Ayrıntılar "
                       "resmî belgede açıklanmıştır ve örnek kod içerir.")
        self.sources = [{"url": f"https://docs.example/{i}"}
                        for i in range(sources)]
        self.note_title = "Öğrenildi: konu"
        self.error = error
        self.question_source = "teacher" if ok else ""


class FakeStudy:
    """Sahte öğretmen–öğrenci döngüsü."""

    def __init__(self, *, ok: bool = True, sources: int = 1,
                 crash: bool = False) -> None:
        self.ok = ok
        self.sources = sources
        self.crash = crash
        self.calls: List[str] = []

    def study(self, topic: str = "") -> FakeRound:
        """Öğrenme turu taklidi."""
        self.calls.append(topic)
        if self.crash:
            raise RuntimeError("öğretmen erişilemedi")
        return FakeRound(self.ok, sources=self.sources,
                         error="" if self.ok else "doğrulanmış kaynak yok")


# ─── Kasadan Sorma ──────────────────────────────────────────────────────────

class TestAskVaultFirst(unittest.TestCase):
    """Önce kasaya sor — boşa öğrenmeye kalkışma."""

    def test_vault_hit_skips_learning(self) -> None:
        """
        Kasa yeterliyse **öğretmen çağrılmamalı**.

        Neden? Her öğrenme turu API kotası ve zaman harcar. Bilgi zaten
        eldeyse tekrar öğrenmek israftır.
        """
        study = FakeStudy()
        akis = KnowledgeWorkflow()
        akis.initialize(study=study, knowledge=FakeVault(hits=ENOUGH_RESULTS))
        sonuc = akis.ask("Godot CharacterBody3D")
        self.assertTrue(sonuc["ok"])
        self.assertEqual(sonuc["source"], "vault")
        self.assertEqual(study.calls, [], "öğretmen çağrılmamalı")
        self.assertIn("zaten kasada", sonuc["note"])

    def test_empty_vault_triggers_learning(self) -> None:
        """Kasa boşsa **öğrenmeli** — sorunun cevabı budur."""
        study = FakeStudy()
        akis = KnowledgeWorkflow()
        akis.initialize(study=study, knowledge=FakeVault(hits=0))
        sonuc = akis.ask("Godot CharacterBody3D")
        self.assertTrue(sonuc["ok"])
        self.assertEqual(sonuc["source"], "learned")
        self.assertEqual(len(study.calls), 1, "öğretmen bir kez çağrılmalı")
        self.assertEqual(sonuc["learned"]["topic"], "Godot CharacterBody3D")

    def test_partial_vault_still_learns(self) -> None:
        """Kasa yetersizse (eşiğin altında) öğrenmeye devam etmeli."""
        study = FakeStudy()
        akis = KnowledgeWorkflow()
        akis.initialize(study=study, knowledge=FakeVault(hits=1))
        sonuc = akis.ask("konu")
        self.assertEqual(sonuc["source"], "learned")
        self.assertEqual(len(study.calls), 1)

    def test_allow_learn_false_respects_flag(self) -> None:
        """`allow_learn=False` iken öğrenmemeli."""
        study = FakeStudy()
        akis = KnowledgeWorkflow()
        akis.initialize(study=study, knowledge=FakeVault(hits=0))
        sonuc = akis.ask("konu", allow_learn=False)
        self.assertEqual(study.calls, [])
        self.assertEqual(sonuc["source"], "none")
        self.assertIn("kapalı", sonuc["note"])

    def test_vault_error_does_not_crash(self) -> None:
        """Kasa çökerse akış devam etmeli (öğrenmeye düşer)."""
        study = FakeStudy()
        akis = KnowledgeWorkflow()
        akis.initialize(study=study, knowledge=FakeVault(fail=True))
        sonuc = akis.ask("konu")
        self.assertEqual(sonuc["source"], "learned")

    def test_empty_question_rejected(self) -> None:
        akis = KnowledgeWorkflow()
        akis.initialize(study=FakeStudy(), knowledge=FakeVault())
        sonuc = akis.ask("   ")
        self.assertFalse(sonuc["ok"])
        self.assertEqual(sonuc["source"], "none")

    def test_missing_knowledge_layer(self) -> None:
        """Kasa yoksa da çalışmalı (yalnızca öğrenir)."""
        study = FakeStudy()
        akis = KnowledgeWorkflow()
        akis.initialize(study=study, knowledge=None)
        sonuc = akis.ask("konu")
        self.assertEqual(sonuc["source"], "learned")


# ─── Öğrenme ────────────────────────────────────────────────────────────────

class TestLearning(unittest.TestCase):
    """Öğrenme turu ve dürüstlük kuralları."""

    def test_learn_returns_answer_and_sources(self) -> None:
        akis = KnowledgeWorkflow()
        akis.initialize(study=FakeStudy(sources=3), knowledge=FakeVault())
        kayit = akis.learn("Godot sahne ağacı")
        self.assertTrue(kayit.ok)
        self.assertIn("Kaynağa dayalı", kayit.answer)
        self.assertEqual(len(kayit.sources), 3)
        self.assertGreater(kayit.duration_ms, 0.0)

    def test_no_teacher_reports_honestly(self) -> None:
        """
        Öğretmen yoksa **uydurmamalı**, açıkça bildirmeli.

        Bu, "bilmiyorsa öğrensin" akışının sınırıdır: öğretmen (API) yoksa
        öğrenme mümkün değildir ve kullanıcı bunu bilmelidir.
        """
        akis = KnowledgeWorkflow()
        akis.initialize(study=None, knowledge=FakeVault())
        kayit = akis.learn("konu")
        self.assertFalse(kayit.ok)
        self.assertIn("öğretmen bağlı değil", kayit.error)
        self.assertEqual(akis.get_stats()["no_knowledge_source"], 1)

    def test_failed_learning_reported(self) -> None:
        akis = KnowledgeWorkflow()
        akis.initialize(study=FakeStudy(ok=False), knowledge=FakeVault())
        kayit = akis.learn("konu")
        self.assertFalse(kayit.ok)
        self.assertIn("kaynak", kayit.error.lower())
        self.assertEqual(akis.get_stats()["learn_failures"], 1)

    def test_teacher_crash_handled(self) -> None:
        """Öğretmen çökerse akış ayakta kalmalı."""
        akis = KnowledgeWorkflow()
        akis.initialize(study=FakeStudy(crash=True), knowledge=FakeVault())
        kayit = akis.learn("konu")
        self.assertFalse(kayit.ok)
        self.assertIn("RuntimeError", kayit.error)

    def test_ask_with_missing_teacher_warns_against_guessing(self) -> None:
        """
        Öğrenme başarısızsa sonuç **tahmin yürütmeye karşı** uyarmalı.

        Ajan bu uyarıyı görürse uydurmak yerine kullanıcıya bildirir.
        """
        akis = KnowledgeWorkflow()
        akis.initialize(study=None, knowledge=FakeVault(hits=0))
        sonuc = akis.ask("bilinmeyen konu")
        self.assertFalse(sonuc["ok"])
        self.assertIn("tahmin", sonuc["note"].lower())

    def test_empty_topic_rejected(self) -> None:
        akis = KnowledgeWorkflow()
        akis.initialize(study=FakeStudy(), knowledge=FakeVault())
        kayit = akis.learn("  ")
        self.assertFalse(kayit.ok)
        self.assertIn("boş", kayit.error)

    def test_history_recorded(self) -> None:
        akis = KnowledgeWorkflow()
        akis.initialize(study=FakeStudy(), knowledge=FakeVault())
        akis.learn("birinci")
        akis.learn("ikinci")
        gecmis = akis.learned()
        self.assertEqual(len(gecmis), 2)
        self.assertEqual(gecmis[0].topic, "birinci")

    def test_history_limited(self) -> None:
        akis = KnowledgeWorkflow()
        akis.initialize(study=FakeStudy(), knowledge=FakeVault())
        for i in range(10):
            akis.learn(f"konu {i}")
        self.assertEqual(len(akis.learned(limit=3)), 3)

    def test_learned_topic_dict_json_safe(self) -> None:
        kayit = LearnedTopic(topic="k", answer="a" * 5000,
                             sources=["https://x"], ok=True)
        veri = kayit.to_dict()
        assert_json_safe(self, veri, "learned_topic")
        self.assertLessEqual(len(veri["answer"]), 1200)

    def test_stats_json_safe(self) -> None:
        akis = KnowledgeWorkflow()
        akis.initialize(study=FakeStudy(), knowledge=FakeVault())
        akis.ask("konu")
        assert_json_safe(self, akis.get_stats(), "knowledge_stats")


class TestRelevanceFilter(unittest.TestCase):
    """
    Alakalılık süzgeci — ajan **yanlışlıkla "biliyorum" dememeli**.

    Ölçülen sorun: kasa araması, sorgunun terimleri sonuçta **geçmese de**
    yüksek skorlu kayıt döndürüyordu:

        "zzz qqq bilinmeyen konu" → "İşletim Sistemi — IPC" (skor 0,729)

    Bu yüzden ajan kasanın bildiğini sanıp **öğrenmiyordu**. Süzgeç
    düzeltilmezse "bilmiyorsa öğrenir" akışı hiç çalışmaz.
    """

    def test_terms_extraction(self) -> None:
        from neurocore.agent.knowledge_work import _terms

        terimler = _terms("Godot 4.8'de CharacterBody3D ile hareket nasıl?")
        self.assertIn("characterbody3d", terimler)
        self.assertIn("godot", terimler)
        # Genel soru kelimeleri atılmalı
        self.assertNotIn("nasıl", terimler)
        self.assertNotIn("ile", terimler)

    def test_terms_drop_short_words(self) -> None:
        from neurocore.agent.knowledge_work import _terms

        self.assertEqual(_terms("bir ve bu abc"), [])

    def test_coverage_calculation(self) -> None:
        from neurocore.agent.knowledge_work import _coverage

        terimler = ["blender", "malzeme", "renk"]
        self.assertAlmostEqual(
            _coverage(terimler, "Blender malzeme rehberi"), 2 / 3, places=3)
        self.assertAlmostEqual(_coverage(terimler, "alakasız metin"), 0.0)

    def test_coverage_empty_terms(self) -> None:
        from neurocore.agent.knowledge_work import _coverage

        self.assertEqual(_coverage([], "herhangi"), 0.0)

    def test_irrelevant_results_filtered(self) -> None:
        """
        Sorgu terimleri geçmiyorsa sonuç **elenmeli**.

        Sahte kasa yüksek skorlu ama alakasız kayıt döndürüyor; süzgeç
        bunları geçirmemeli.
        """
        class AlakasizKasa:
            def search(self, query: str, limit: int = 8, **kw: Any) -> List[Any]:
                return [{"title": "İşletim Sistemi — IPC",
                         "body": "Süreçler arası iletişim anlatılır.",
                         "score": 0.73}]

        akis = KnowledgeWorkflow()
        akis.initialize(study=None, knowledge=AlakasizKasa())
        sonuc = akis._search_vault("zzz qqq bilinmeyen konu", limit=5)
        self.assertEqual(sonuc, [], "alakasız sonuç elenmeli")

    def test_relevant_results_kept(self) -> None:
        class AlakaliKasa:
            def search(self, query: str, limit: int = 8, **kw: Any) -> List[Any]:
                return [{"title": "Blender malzeme renklendirme",
                         "body": "Blender malzeme renk ayarları anlatılır. " * 10,
                         "score": 0.88}]

        akis = KnowledgeWorkflow()
        akis.initialize(study=None, knowledge=AlakaliKasa())
        sonuc = akis._search_vault("Blender malzeme renklendirme", limit=5)
        self.assertEqual(len(sonuc), 1, "alakalı sonuç korunmalı")

    def test_relevant_only_false_bypasses_filter(self) -> None:
        """Süzgeç kapatılabilmeli (tanı için)."""
        class AlakasizKasa:
            def search(self, query: str, limit: int = 8, **kw: Any) -> List[Any]:
                return [{"title": "Alakasız", "body": "Metin", "score": 0.7}]

        akis = KnowledgeWorkflow()
        akis.initialize(study=None, knowledge=AlakasizKasa())
        sonuc = akis._search_vault("zzz qqq", limit=5, relevant_only=False)
        self.assertEqual(len(sonuc), 1)

    def test_low_score_filtered(self) -> None:
        """Skor eşiğin altındaysa sonuç elenmeli."""
        class DusukSkorKasa:
            def search(self, query: str, limit: int = 8, **kw: Any) -> List[Any]:
                return [{"title": "Blender malzeme", "body": "Blender malzeme",
                         "score": 0.05}]

        akis = KnowledgeWorkflow()
        akis.initialize(study=None, knowledge=DusukSkorKasa())
        self.assertEqual(akis._search_vault("Blender malzeme", limit=5), [])

    def test_false_positive_triggers_learning(self) -> None:
        """
        **Asıl senaryo:** kasa alakasız sonuç döndürüyor → öğrenmeye
        gidilmeli.

        Süzgeç olmadan bu test başarısız olur ve ajan öğrenmezdi.
        """
        class AlakasizKasa:
            def search(self, query: str, limit: int = 8, **kw: Any) -> List[Any]:
                return [{"title": "Alakasız not", "body": "İlgisiz içerik",
                         "score": 0.73}]

        study = FakeStudy()
        akis = KnowledgeWorkflow()
        akis.initialize(study=study, knowledge=AlakasizKasa())
        sonuc = akis.ask("zzz qqq bilinmeyen konu")
        self.assertEqual(sonuc["source"], "learned",
                         "alakasız kasa sonucu öğrenmeyi engellememeli")
        self.assertEqual(len(study.calls), 1)

    def test_search_error_returns_empty(self) -> None:
        akis = KnowledgeWorkflow()
        akis.initialize(study=FakeStudy(), knowledge=FakeVault(fail=True))
        self.assertEqual(akis._search_vault("konu"), [])


class TestAnswerQualityGate(unittest.TestCase):
    """
    Kalite kapısı — değersiz cevap **kasaya yazılmamalı**.

    Ölçülen sorun: öğretmen yanlış kaynak önerdiğinde (PyTorch sorusuna
    NVIDIA CUDA kılavuzu) kendi cevabında "bu kaynakla cevaplanamaz"
    diyordu — ama tur **başarılı** sayılıp kasaya yazılıyordu. Böyle bir
    not kasa için zehirdir: model ondan öğrenir.
    """

    def test_admits_insufficiency_rejected(self) -> None:
        from neurocore.agent.knowledge_work import answer_is_useful

        cevap = ("Verilen kaynak [1] ile bu soru **tam olarak cevaplanamaz**. "
                 "Çünkü [1], NVIDIA CUDA Programming Guide metnidir ve "
                 "PyTorch DataLoader içermez.")
        yararli, gerekce = answer_is_useful(cevap)
        self.assertFalse(yararli)
        self.assertIn("cevaplanamaz", gerekce)

    def test_english_insufficiency_rejected(self) -> None:
        from neurocore.agent.knowledge_work import answer_is_useful

        yararli, _ = answer_is_useful(
            "The provided sources do not contain information about this API.")
        self.assertFalse(yararli)

    def test_substantive_answer_accepted(self) -> None:
        from neurocore.agent.knowledge_work import answer_is_useful

        cevap = ("PyTorch DataLoader'da num_workers, veri yüklemeyi paralel iş "
                 "parçacıklarına böler. pin_memory ise sayfalanmış belleği "
                 "sabitler; böylece GPU'ya kopyalama hızlanır. Tipik değerler: "
                 "num_workers=4-8, pin_memory=True.")
        yararli, gerekce = answer_is_useful(cevap)
        self.assertTrue(yararli, gerekce)

    def test_short_answer_rejected(self) -> None:
        from neurocore.agent.knowledge_work import answer_is_useful

        yararli, gerekce = answer_is_useful("Kısa.")
        self.assertFalse(yararli)
        self.assertIn("kısa", gerekce)

    def test_empty_answer_rejected(self) -> None:
        from neurocore.agent.knowledge_work import answer_is_useful

        self.assertFalse(answer_is_useful("")[0])
        self.assertFalse(answer_is_useful("   ")[0])

    def test_useless_learning_marked_failed(self) -> None:
        """
        Yetersiz cevap veren tur **başarısız** sayılmalı.

        Böylece `learn.topic` aracı ajanı uyarır ve ajan tahmin yürütmek
        yerine kullanıcıya bildirir.
        """
        class YetersizStudy:
            def study(self, topic: str = "") -> Any:
                tur = FakeRound(ok=True)
                tur.answer = ("Verilen kaynak [1] ile bu soru tam olarak "
                              "cevaplanamaz; kaynak bu konuyu içermiyor.")
                tur.note_title = ""
                return tur

        akis = KnowledgeWorkflow()
        akis.initialize(study=YetersizStudy(), knowledge=FakeVault())
        kayit = akis.learn("PyTorch DataLoader")
        self.assertFalse(kayit.ok, "yetersiz cevap başarılı sayılmamalı")
        self.assertIn("kullanılabilir değil", kayit.error)
        self.assertEqual(akis.get_stats()["learn_failures"], 1)

    def test_useful_learning_marked_ok(self) -> None:
        akis = KnowledgeWorkflow()
        akis.initialize(study=FakeStudy(), knowledge=FakeVault())
        kayit = akis.learn("konu")
        self.assertTrue(kayit.ok)
        self.assertEqual(akis.get_stats()["learns"], 1)


class TestWebTools(unittest.TestCase):
    """
    İnternet araçları — ajan **kendi kararıyla** araştırabilmeli.

    Ölçülen boşluk: arama altyapısı (`web.search_web`) vardı ve çalışıyordu
    ama **hiçbir yerden çağrılmıyordu** (ölü kod). Öğrenme akışı yalnızca
    öğretmenin hafızasından adres önerisi alıyordu; öğretmen yanlış adres
    verdiğinde yanlış bilgi geliyordu:

        Soru : "PyTorch DataLoader num_workers nasıl kullanılır?"
        Öneri: docs.nvidia.com/cuda/cuda-programming-guide/  ← ALAKASIZ

    `web.search` ve `web.fetch` araçları bu boşluğu kapatır.
    """

    def setUp(self) -> None:
        self.akis = KnowledgeWorkflow()
        self.akis.initialize(study=None, knowledge=None)
        self.tools = {t.name: t for t in self.akis.tools()}

    def test_web_tools_exposed(self) -> None:
        self.assertIn("web.search", self.tools)
        self.assertIn("web.fetch", self.tools)

    def test_web_tools_are_read_safety(self) -> None:
        """İnternet okuma **salt okuma** sayılır (yazma değil)."""
        for ad in ("web.search", "web.fetch"):
            self.assertEqual(self.tools[ad].safety, Safety.READ)

    def test_web_tools_wire_names_valid(self) -> None:
        from neurocore.agent.tools import to_wire_name

        for ad in ("web.search", "web.fetch"):
            wire = to_wire_name(ad)
            self.assertNotIn(".", wire)
            self.assertRegex(wire, r"^[a-zA-Z0-9_-]+$")

    def test_search_requires_query(self) -> None:
        """Boş sorgu hata döndürmeli (çökme değil)."""
        sonuc = self.tools["web.search"].handler(query="   ")
        self.assertFalse(sonuc["ok"])
        self.assertIn("boş", sonuc["error"])
        self.assertEqual(sonuc["results"], [])

    def test_fetch_requires_url(self) -> None:
        sonuc = self.tools["web.fetch"].handler(url="")
        self.assertFalse(sonuc["ok"])
        self.assertIn("boş", sonuc["error"])

    def test_search_description_guides_agent(self) -> None:
        """
        Açıklama ajanı **doğru davranışa** yönlendirmeli.

        `web.search` açıklaması "adres uydurma, önce ara" demeli; aksi
        hâlde model adres uydurur ve yanlış kaynak okur.
        """
        aciklama = self.tools["web.search"].description.lower()
        self.assertIn("uydurma", aciklama)
        self.assertIn("web.fetch", aciklama)

    def test_search_has_required_query_param(self) -> None:
        zorunlu = [p.name for p in self.tools["web.search"].parameters
                   if p.required]
        self.assertIn("query", zorunlu)

    def test_fetch_has_required_url_param(self) -> None:
        zorunlu = [p.name for p in self.tools["web.fetch"].parameters
                   if p.required]
        self.assertIn("url", zorunlu)

    def test_fetch_private_address_rejected(self) -> None:
        """
        **Güvenlik:** yerel ağ adresleri reddedilmeli (SSRF koruması).
        """
        for adres in ("http://127.0.0.1:8791/api/state",
                      "http://localhost/admin",
                      "http://192.168.1.1/"):
            sonuc = self.tools["web.fetch"].handler(url=adres)
            self.assertFalse(sonuc["ok"],
                             f"özel adres kabul edildi: {adres}")

    def test_fetch_invalid_scheme_rejected(self) -> None:
        sonuc = self.tools["web.fetch"].handler(url="file:///etc/passwd")
        self.assertFalse(sonuc["ok"])


class TestAutonomousTopicDiscovery(unittest.TestCase):
    """
    Özerk konu keşfi — öğrenme **kendi kendine** ilerlemeli.

    Ölçülen sorun: özerk öğrenci (SelfStudy) sabit konu listesi boşken
    kasada **zaten olan** başlıkları "X konusunu derinleştir" diye tekrar
    soruyordu. Yani kendi bildiğini tekrar öğreniyor, **yeni konu
    keşfetmiyordu**.
    """

    def _study(self, llm: Any, knowledge: Any = None) -> Any:
        from neurocore.agent.study import SelfStudy

        s = SelfStudy()
        s.initialize(llm=llm, knowledge=knowledge, recorder=None,
                     workspace=".")
        return s

    def test_generate_topic_uses_teacher(self) -> None:
        class Sahte:
            def __init__(self) -> None:
                self.cagri = 0

            def chat(self, messages: Any, **kwargs: Any) -> Any:
                self.cagri += 1

                class Y:
                    ok = True
                    text = "Raft konsensüs algoritmasında lider seçimi"
                return Y()

        yon = Sahte()
        s = self._study(yon)
        konu = s._generate_topic(["Blender malzeme"])
        self.assertEqual(konu, "Raft konsensüs algoritmasında lider seçimi")
        self.assertEqual(yon.cagri, 1)

    def test_generate_topic_strips_formatting(self) -> None:
        """Madde imi, numara ve tırnak temizlenmeli."""
        class Karmasik:
            def chat(self, messages: Any, **kwargs: Any) -> Any:
                class Y:
                    ok = True
                    text = '1) "Docker katman önbelleği" \n\n açıklama'
                return Y()

        s = self._study(Karmasik())
        self.assertEqual(s._generate_topic([]), "Docker katman önbelleği")

    def test_generate_topic_rejects_too_short(self) -> None:
        class Kisa:
            def chat(self, messages: Any, **kwargs: Any) -> Any:
                class Y:
                    ok = True
                    text = "kısa"
                return Y()

        self.assertEqual(self._study(Kisa())._generate_topic([]), "")

    def test_generate_topic_rejects_too_long(self) -> None:
        class Uzun:
            def chat(self, messages: Any, **kwargs: Any) -> Any:
                class Y:
                    ok = True
                    text = "x" * 300
                return Y()

        self.assertEqual(self._study(Uzun())._generate_topic([]), "")

    def test_generate_topic_handles_no_teacher(self) -> None:
        self.assertEqual(self._study(None)._generate_topic([]), "")

    def test_generate_topic_handles_failure(self) -> None:
        """Öğretmen hata dönerse boş dize (çökme yok)."""
        class Bozuk:
            def chat(self, messages: Any, **kwargs: Any) -> Any:
                class Y:
                    ok = False
                    text = ""
                    error = "sağlayıcı yok"
                return Y()

        self.assertEqual(self._study(Bozuk())._generate_topic([]), "")

    def test_generate_topic_handles_exception(self) -> None:
        class Patlayan:
            def chat(self, messages: Any, **kwargs: Any) -> Any:
                raise RuntimeError("ağ yok")

        self.assertEqual(self._study(Patlayan())._generate_topic([]), "")

    def test_counter_records_generation(self) -> None:
        class Iyi:
            def chat(self, messages: Any, **kwargs: Any) -> Any:
                class Y:
                    ok = True
                    text = "PostgreSQL sorgu planı okuma"
                return Y()

        s = self._study(Iyi())
        s._generate_topic([])
        self.assertEqual(s._counter.get("topics_generated"), 1)

    def test_next_topic_prefers_new_over_deepening(self) -> None:
        """
        Yeni konu üretilebiliyorsa **derinleştirme yerine** o seçilmeli.

        Derinleştirme son çaredir; ana yol yeni konu keşfidir.
        """
        class Iyi:
            def chat(self, messages: Any, **kwargs: Any) -> Any:
                class Y:
                    ok = True
                    text = "Yeni kesfedilmis konu"
                return Y()

        class Kasa:
            def recent(self, limit: int = 15) -> Any:
                return [{"title": "Zaten bilinen konu"}]

        s = self._study(Iyi(), Kasa())
        self.assertEqual(s._next_topic(), "Yeni kesfedilmis konu")

    def test_next_topic_falls_back_to_deepening(self) -> None:
        """Öğretmen konu üretemezse bilinen başlık derinleştirilmeli."""
        class Sessiz:
            def chat(self, messages: Any, **kwargs: Any) -> Any:
                class Y:
                    ok = False
                    text = ""
                return Y()

        class Kasa:
            def recent(self, limit: int = 15) -> Any:
                return [{"title": "Zaten bilinen konu"}]

        s = self._study(Sessiz(), Kasa())
        self.assertIn("derinleştir", s._next_topic())

    def test_next_topic_returns_empty_when_nothing(self) -> None:
        """Hiçbir kaynak yoksa boş dize (çökme yok)."""
        self.assertEqual(self._study(None)._next_topic(), "")


# ─── Araç Tanımları ─────────────────────────────────────────────────────────

class TestKnowledgeTools(unittest.TestCase):
    """Ajan araçları doğru tanımlanmalı."""

    def setUp(self) -> None:
        self.akis = KnowledgeWorkflow()
        self.akis.initialize(study=FakeStudy(), knowledge=FakeVault())
        self.tools = {t.name: t for t in self.akis.tools()}

    def test_three_tools_exposed(self) -> None:
        """Üç araç: sor, öğren, durum."""
        for ad in ("knowledge.ask", "learn.topic", "learn.status"):
            self.assertIn(ad, self.tools)

    def test_all_read_safety(self) -> None:
        """
        Hepsi **read** olmalı.

        Neden? Kasa yazımı araç içinde olur ve doğrulanmış kaynağa dayanır.
        `write` güvenliği istemek, Denetimli modda her öğrenme için onay
        gerektirirdi ve "bilmiyorsan öğren" akışı kullanılamaz hâle gelirdi.
        """
        for arac in self.tools.values():
            self.assertEqual(arac.safety, Safety.READ)

    def test_wire_names_valid(self) -> None:
        """Araç adları sağlayıcı regex'ine uymalı."""
        from neurocore.agent.tools import to_wire_name

        for ad in self.tools:
            wire = to_wire_name(ad)
            self.assertNotIn(".", wire)
            self.assertRegex(wire, r"^[a-zA-Z0-9_-]+$")

    def test_ask_handler_works(self) -> None:
        arac = self.tools["knowledge.ask"]
        sonuc = arac.handler(question="Godot düğüm sistemi")
        self.assertTrue(sonuc["ok"])
        self.assertEqual(sonuc["source"], "learned")

    def test_learn_handler_works(self) -> None:
        arac = self.tools["learn.topic"]
        sonuc = arac.handler(topic="Godot sinyaller")
        self.assertTrue(sonuc["ok"])
        self.assertIn("answer", sonuc)
        self.assertIn("sources", sonuc)

    def test_status_handler_works(self) -> None:
        arac = self.tools["learn.status"]
        sonuc = arac.handler()
        self.assertTrue(sonuc["initialized"])
        self.assertTrue(sonuc["study_ready"])
        self.assertTrue(sonuc["knowledge_ready"])

    def test_descriptions_are_guiding(self) -> None:
        """
        Açıklamalar ajanı **doğru davranışa** yönlendirmeli.

        `knowledge.ask` açıklaması "tahmin yürütmek yerine bunu kullan"
        demeli; aksi hâlde model uydurmayı seçebilir.
        """
        aciklama = self.tools["knowledge.ask"].description.lower()
        self.assertIn("tahmin", aciklama)
        self.assertIn("öğren", aciklama)

    def test_ask_has_required_parameter(self) -> None:
        arac = self.tools["knowledge.ask"]
        zorunlu = [p.name for p in arac.parameters if p.required]
        self.assertIn("question", zorunlu)

    def test_lifecycle(self) -> None:
        akis = KnowledgeWorkflow()
        akis.initialize(study=FakeStudy(), knowledge=FakeVault())
        self.assertTrue(akis.get_stats()["initialized"])
        akis.shutdown()
        self.assertFalse(akis.get_stats()["initialized"])

    def test_initialize_updates_dependencies(self) -> None:
        """
        Sonraki `initialize` çağrıları bağımlılıkları **güncellemeli**.

        Ölçülen hata (köprüde): erken dönüş, yeni `study`/`knowledge`
        bağlanmasını engelliyordu.
        """
        akis = KnowledgeWorkflow()
        akis.initialize(study=None, knowledge=None)
        yeni_study = FakeStudy()
        yeni_kasa = FakeVault(hits=5)
        akis.initialize(study=yeni_study, knowledge=yeni_kasa)
        self.assertIs(akis._study, yeni_study)
        self.assertIs(akis._knowledge, yeni_kasa)


if __name__ == "__main__":
    unittest.main(verbosity=2)
