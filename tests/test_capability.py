"""
Yetenek katmanı testleri — playbook · hiyerarşi · bağlam sıkıştırma
====================================================================
Ağ çağrısı YOK: router sahtelenir.

```text
1. BENZERLİK  : Türkçe ekler önek eşleşmesiyle yakalanır
2. PLAYBOOK   : eşik altı reddedilir · tekrar sayılır · benzer görev bulunur
3. HİYERARŞİ  : JSON çözümü · liste dışı araç atılır · hata → boş plan
4. SIKIŞTIRMA : uzun geçmiş özete indirilir, sistem + son mesajlar korunur
5. İPUCU      : benzer playbook plana eklenir, sayaç artar
```
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, List

from neurocore.agent.hierarchy import AltHedef, Adim, Hiyerarsi, planla
from neurocore.agent.playbooks import (
    MAX_ADIM,
    MIN_ODUL,
    Playbook,
    PlaybookStore,
    benzerlik,
)


class _Yanit:
    def __init__(self, text: str, ok: bool = True, error: str = "") -> None:
        self.text = text
        self.ok = ok
        self.error = error


class _Router:
    def __init__(self, text: str = "{}", ok: bool = True,
                 patlar: bool = False) -> None:
        self.text = text
        self.ok = ok
        self.patlar = patlar
        self.cagri = 0

    def chat(self, messages: Any, **kwargs: Any) -> _Yanit:
        self.cagri += 1
        if self.patlar:
            raise RuntimeError("ağ koptu")
        return _Yanit(self.text, ok=self.ok)


def _store() -> PlaybookStore:
    return PlaybookStore(Path(tempfile.mkdtemp(prefix="nc-pb-")) / "pb.jsonl")


# ─── 1) Benzerlik ───────────────────────────────────────────────────────────

class TestBenzerlik(unittest.TestCase):

    def test_turkce_ekler(self) -> None:
        """⚠ Ölçüldü: düz küme kesişimi 0.20 veriyordu (eşik 0.35)."""
        skor = benzerlik("lowpoly ağaç üret", "lowpoly ağaçları geliştir")
        self.assertGreaterEqual(skor, 0.35, f"skor {skor}")

    def test_alakasiz(self) -> None:
        self.assertLess(benzerlik("blog sitesi oluştur",
                                  "kılıç modeli üret"), 0.2)

    def test_bos(self) -> None:
        self.assertEqual(benzerlik("", "ağaç"), 0.0)

    def test_sirasiz_kelime(self) -> None:
        self.assertGreaterEqual(benzerlik("ağaç lowpoly üret",
                                          "lowpoly ağaç üret"), 0.9)


# ─── 2) Playbook ────────────────────────────────────────────────────────────

class TestPlaybook(unittest.TestCase):

    def test_dusuk_odul_reddedilir(self) -> None:
        st = _store()
        self.assertIsNone(st.kaydet("görev", [{"tool": "a"}, {"tool": "b"}],
                                   MIN_ODUL - 0.1))

    def test_tek_adim_reddedilir(self) -> None:
        st = _store()
        self.assertIsNone(st.kaydet("görev", [{"tool": "a"}], 0.9))

    def test_kaydedilir_ve_bulunur(self) -> None:
        st = _store()
        pb = st.kaydet("lowpoly ağaç üret",
                       [{"tool": "asset.uret", "args": "kategori=bitki"},
                        {"tool": "asset.onizleme"}], 0.85)
        self.assertIsNotNone(pb)
        bulunan = st.bul("lowpoly ağaçları geliştir")
        self.assertTrue(bulunan, "benzer görev bulunamadı")
        self.assertEqual(bulunan[0][0].id, pb.id)

    def test_ayni_akis_tekrar_sayilir(self) -> None:
        st = _store()
        adimlar = [{"tool": "a"}, {"tool": "b"}]
        st.kaydet("görev x", adimlar, 0.8)
        st.kaydet("görev x", adimlar, 0.9)
        kayitlar = st.hepsi(tazele=True)
        self.assertEqual(len(kayitlar), 1, "aynı akış ikinci kez yazıldı")
        self.assertEqual(kayitlar[0].kullanim, 1)

    def test_adim_siniri(self) -> None:
        st = _store()
        adimlar = [{"tool": f"t{i}"} for i in range(20)]
        pb = st.kaydet("uzun", adimlar, 0.9)
        self.assertLessEqual(len(pb.adimlar), MAX_ADIM)

    def test_ipucu_metni(self) -> None:
        pb = Playbook(goal="g", adimlar=[{"tool": "asset.uret"},
                                         {"tool": "asset.onizleme"}], odul=0.8)
        metin = pb.ipucu()
        self.assertIn("asset.uret", metin)
        self.assertIn("asset.onizleme", metin)

    def test_istatistik(self) -> None:
        st = _store()
        st.kaydet("a", [{"tool": "x"}, {"tool": "y"}], 0.8)
        st = st.istatistik()
        self.assertEqual(st["playbook"], 1)
        self.assertEqual(st["ortalama_adim"], 2.0)

    def test_bozuk_satir_atlanir(self) -> None:
        yol = Path(tempfile.mkdtemp(prefix="nc-pb2-")) / "pb.jsonl"
        yol.write_text('{"goal": "a", "adimlar": [{"tool": "x"},'
                       '{"tool": "y"}], "odul": 0.9}\nBOZUK SATIR\n',
                       encoding="utf-8")
        st = PlaybookStore(yol)
        self.assertEqual(len(st.hepsi()), 1)


# ─── 3) Hiyerarşi ───────────────────────────────────────────────────────────

class TestHiyerarsi(unittest.TestCase):

    def test_cozum(self) -> None:
        metin = json.dumps({
            "goal": "ağaç geliştir",
            "subgoals": [
                {"title": "incele",
                 "steps": [{"tool": "asset.katalog"}],
                 "check": "katalog var"},
                {"title": "üret",
                 "steps": [{"tool": "asset.uret", "args": "kategori=bitki"}],
                 "check": "dosya var"},
            ]}, ensure_ascii=False)
        h = Hiyerarsi.coz(metin)
        self.assertTrue(h.ok)
        self.assertEqual(len(h.alt_hedefler), 2)
        self.assertEqual(h.araclar(), ["asset.katalog", "asset.uret"])
        self.assertIn("ALT HEDEFLER", h.ozet())

    def test_bozuk_json(self) -> None:
        self.assertFalse(Hiyerarsi.coz("plan yok").ok)

    def test_string_alt_hedef(self) -> None:
        h = Hiyerarsi.coz('{"subgoals": ["birinci", "ikinci"]}')
        self.assertTrue(h.ok)
        self.assertEqual(len(h.alt_hedefler), 2)

    def test_liste_disi_arac_atilir(self) -> None:
        """⚠ Model uydurma araç yazarsa plana girmemeli."""
        metin = json.dumps({"subgoals": [
            {"title": "a", "steps": [{"tool": "uydurma.arac"},
                                     {"tool": "asset.uret"}]}]})
        r = _Router(metin)
        h = planla(r, "ağaç üret", araclar=["asset.uret"])
        self.assertTrue(h.ok)
        self.assertEqual(h.araclar(), ["asset.uret"])

    def test_router_hatasi(self) -> None:
        h = planla(_Router(patlar=True), "görev", araclar=["a"])
        self.assertFalse(h.ok)

    def test_kisa_gorev_icin_cagri_yok(self) -> None:
        """Planlama uzun görevlerde devreye girer (ajan tarafında)."""
        from neurocore.agent.loop import HIER_MIN_UZUNLUK

        self.assertGreater(HIER_MIN_UZUNLUK, 40)


# ─── 4-5) Döngü: sıkıştırma ve ipucu ───────────────────────────────────────

class TestMesajOnarimi(unittest.TestCase):
    """
    ⚠ ÖLÇÜLEN HATA: bağlam sıkıştırması kuyruğu bir `tool` mesajının
    ortasından başlatınca sağlayıcı **HTTP 400** verdi:
    "Messages with role 'tool' must be a response to a preceding message
    with 'tool_calls'". Bu testler o hatanın geri dönmesini engeller.
    """

    def _dongu(self) -> Any:
        from neurocore.agent.loop import AgentLoop

        return AgentLoop()

    def test_yetim_tool_atilir(self) -> None:
        from neurocore.agent.llm import LLMMessage

        loop = self._dongu()
        mesajlar = [
            LLMMessage("system", "s"),
            LLMMessage("tool", "yetim sonuç", tool_call_id="x1"),
            LLMMessage("user", "devam"),
        ]
        loop._mesajlari_onar(mesajlar)
        self.assertEqual([m.role for m in mesajlar], ["system", "user"])

    def test_eslesen_tool_kalir(self) -> None:
        from neurocore.agent.llm import LLMMessage

        loop = self._dongu()
        a = LLMMessage("assistant", "")
        a.tool_calls = [{"id": "c1", "function": {"name": "x"}}]
        mesajlar = [LLMMessage("system", "s"), a,
                    LLMMessage("tool", "sonuç", tool_call_id="c1"),
                    LLMMessage("user", "devam")]
        loop._mesajlari_onar(mesajlar)
        self.assertEqual([m.role for m in mesajlar],
                         ["system", "assistant", "tool", "user"])

    def test_sikistirma_tool_ile_baslamaz(self) -> None:
        from neurocore.agent.cognition import HedefDurumu
        from neurocore.agent.llm import LLMMessage

        loop = self._dongu()
        loop._hedef = HedefDurumu(hedef="g")
        loop._hedef.olgu_ekle("kanıt")
        mesajlar = [LLMMessage("system", "s")]
        # son 10 mesajın ilki `tool` olacak şekilde düzenle
        mesajlar += [LLMMessage("user", f"u{i}") for i in range(20)]
        for _ in range(6):
            mesajlar.append(LLMMessage("tool", "t", tool_call_id="yok"))
        loop._baglami_sikistir(mesajlar)
        self.assertNotEqual(mesajlar[1].role, "tool")
        self.assertTrue(all(m.role != "tool" for m in mesajlar))


class TestDonguYetenekleri(unittest.TestCase):

    def _dongu(self) -> Any:
        from neurocore.agent.loop import AgentLoop

        loop = AgentLoop()
        loop._workspace = "."
        loop._config = None
        loop._event_bus = None
        loop._cog = None
        return loop

    def test_sikistirma(self) -> None:
        from neurocore.agent.cognition import HedefDurumu
        from neurocore.agent.llm import LLMMessage

        loop = self._dongu()
        loop._hedef = HedefDurumu(hedef="uzun görev")
        loop._hedef.olgu_ekle("dosya 42 satır")
        mesajlar = [LLMMessage("system", "sistem")]
        mesajlar += [LLMMessage("user", f"adım {i}") for i in range(30)]
        loop._baglami_sikistir(mesajlar)
        self.assertLess(len(mesajlar), 20)
        self.assertEqual(mesajlar[0].content, "sistem")
        self.assertEqual(mesajlar[1].role, "user")
        self.assertIn("ÖZET", mesajlar[1].content)
        self.assertIn("42 satır", mesajlar[1].content)
        self.assertEqual(loop._counter["context_compressed"], 1)

    def test_kisa_gecmis_sikistirilmaz(self) -> None:
        from neurocore.agent.llm import LLMMessage

        loop = self._dongu()
        loop._hedef = None
        mesajlar = [LLMMessage("system", "s")] + \
                   [LLMMessage("user", f"{i}") for i in range(5)]
        loop._baglami_sikistir(mesajlar)
        self.assertEqual(len(mesajlar), 6)
        self.assertEqual(loop._counter["context_compressed"], 0)

    def test_playbook_ipucu_eklenir(self) -> None:
        from neurocore.agent.llm import LLMMessage
        from neurocore.agent.loop import AgentRun

        loop = self._dongu()
        st = _store()
        st.kaydet("lowpoly ağaç üret",
                  [{"tool": "asset.uret"}, {"tool": "asset.onizleme"}], 0.85)
        loop._playbooks = st
        run = AgentRun(task="t")
        mesajlar: List[Any] = []
        loop._playbook_ipucu("lowpoly ağaçları geliştir", run, mesajlar)
        self.assertEqual(len(mesajlar), 1)
        self.assertIn("İPUCU", mesajlar[0].content)
        self.assertEqual(loop._counter["playbook_hit"], 1)
        self.assertTrue(any(s.kind == "playbook" for s in run.steps))

    def test_alakasiz_gorev_ipucu_almaz(self) -> None:
        from neurocore.agent.loop import AgentRun

        loop = self._dongu()
        st = _store()
        st.kaydet("lowpoly ağaç üret",
                  [{"tool": "asset.uret"}, {"tool": "asset.onizleme"}], 0.85)
        loop._playbooks = st
        mesajlar: List[Any] = []
        loop._playbook_ipucu("blog sitesi oluştur", AgentRun(task="t"),
                             mesajlar)
        self.assertEqual(mesajlar, [])
        self.assertEqual(loop._counter["playbook_hit"], 0)

    def test_kaydet_akistan_uretir(self) -> None:
        from neurocore.agent.loop import AgentRun, AgentStep

        loop = self._dongu()
        loop._playbooks = _store()
        run = AgentRun(task="kılıç üret")
        run.ok = True
        run.steps = [
            AgentStep(index=0, kind="tool", tool="asset.uret",
                      args={"kategori": "silah"}, ok=True),
            AgentStep(index=1, kind="tool", tool="asset.onizleme", ok=True),
        ]
        loop._playbook_kaydet("kılıç üret", run)
        self.assertEqual(loop._counter["playbook_saved"], 1)
        self.assertEqual(loop._playbooks.istatistik()["playbook"], 1)

    def test_basarisiz_calisma_kaydedilmez(self) -> None:
        from neurocore.agent.loop import AgentRun, AgentStep

        loop = self._dongu()
        loop._playbooks = _store()
        run = AgentRun(task="x")
        run.ok = False
        run.steps = [AgentStep(index=0, kind="tool", tool="a", ok=True),
                     AgentStep(index=1, kind="tool", tool="b", ok=True)]
        loop._playbook_kaydet("x", run)
        self.assertEqual(loop._counter["playbook_saved"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
