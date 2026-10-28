"""
Düşünme entegrasyonu testleri (Aşama 5)
=========================================
Koruduğu davranışlar:

1. **Araştırma araçları çekirdekte** — `web.search`, `web.fetch`,
   `knowledge.ask` her görevde seçilebilir olmalı. Ölçülen eksik: bu üçü
   kayıt defterinde vardı ama `CORE_TOOLS`'ta değildi → ajan
   **kendiliğinden** araştırmıyordu.

2. **Düşünme döngüsü plana bağlı** — `ThinkingLoop` yalnızca köprüde
   kullanılıyordu; sohbet yolunda plan **tek** LLM çağrısıyla alınıyordu.
   Artık: bulut N aday üretir → yerel model seçer.

3. **Geri düşüş güvenli** — düşünme yoksa veya hata verirse davranış
   eskisi gibi kalır (tek çağrı / ilk aday), çökme yok.

Bu testler **ağ çağrısı yapmaz**: LLM ve düşünücü sahtedir.
"""

from __future__ import annotations

import unittest
from typing import Any, List, Sequence

from neurocore.agent.loop import CORE_TOOLS, AgentLoop


# ─── Sahteler ───────────────────────────────────────────────────────────────

class _Yanit:
    """`LLMResponse` yerine geçen asgari nesne."""

    def __init__(self, text: str, ok: bool = True) -> None:
        self.text = text
        self.ok = ok
        self.prompt_tokens = 0
        self.completion_tokens = 0


class _SahteRouter:
    """Her çağrıda sabit metin döner; kaç kez çağrıldığını sayar."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.cagri = 0

    def chat(self, messages: Sequence[Any], **kwargs: Any) -> _Yanit:
        self.cagri += 1
        return _Yanit(self.text)


class _SahteSonuc:
    """`ThinkingResult` yerine geçen asgari nesne."""

    def __init__(self, chosen: str, *, ok: bool = True,
                 source: str = "local", confidence: str = "yuksek",
                 margin: float = 0.4, error: str = "") -> None:
        self.ok = ok
        self.chosen = chosen
        self.source = source
        self.confidence = confidence
        self.margin = margin
        self.error = error
        self.ranking: List[dict] = [{"index": 0}] if ok else []


class _SahteDusunucu:
    """`think()` çağrısını kaydeder, sabit seçim döner."""

    def __init__(self, secilen: str) -> None:
        self.secilen = secilen
        self.cagri = 0
        self.gorulen: List[str] = []

    def think(self, problem: str, propose: Any, *, context: str = "") -> Any:
        self.cagri += 1
        self.gorulen = list(propose(3))
        return _SahteSonuc(self.secilen)


class _BozukDusunucu:
    """Her çağrıda hata verir (sıralayıcı yok senaryosu)."""

    def think(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("yerel sıralayıcı yok")


_UC_PLAN = (
    "### PLAN 1\nÖnce oku, sonra düzelt.\n\n"
    "### PLAN 2\nTest yaz, sonra düzelt.\n\n"
    "### PLAN 3\nYedek al, sonra düzelt.\n"
)


# ─── Testler ────────────────────────────────────────────────────────────────

class TestArastirmaCekirdekte(unittest.TestCase):
    """Araştırma araçları varsayılan araç setinde olmalı."""

    def test_arastirma_araclari(self) -> None:
        for ad in ("web.search", "web.fetch", "knowledge.ask"):
            self.assertIn(ad, CORE_TOOLS,
                          f"{ad} CORE_TOOLS'ta değil — ajan "
                          "kendiliğinden araştıramaz")

    def test_cekirdek_sayisi(self) -> None:
        """25 blok + 3 araştırma = 28."""
        self.assertGreaterEqual(len(CORE_TOOLS), 28)


class TestPlanAyristirma(unittest.TestCase):
    """`_parse_plans` adayları doğru bölmeli."""

    def setUp(self) -> None:
        self.loop = AgentLoop()

    def test_uc_aday(self) -> None:
        adaylar = self.loop._parse_plans(_UC_PLAN)
        self.assertEqual(len(adaylar), 3)
        self.assertIn("Test yaz", adaylar[1])

    def test_tek_aday_yedegi(self) -> None:
        """Başlık yoksa metnin tamamı tek aday olur (sessiz kayıp yok)."""
        adaylar = self.loop._parse_plans("Düz bir plan metni.")
        self.assertEqual(len(adaylar), 1)
        self.assertEqual(adaylar[0], "Düz bir plan metni.")


class TestDusunmesizPlan(unittest.TestCase):
    """Düşünme bağlı değilken davranış **eskisi gibi** olmalı."""

    def test_tek_cagri_ve_bos_meta(self) -> None:
        loop = AgentLoop()
        router = _SahteRouter("tek plan")
        loop._router = router
        plan, meta = loop._ask_plan("görev", [], "")
        self.assertEqual(plan, "tek plan")
        self.assertEqual(meta, {})
        self.assertEqual(router.cagri, 1)


class TestDusunmeliPlan(unittest.TestCase):
    """Düşünme bağlıyken: N aday → yerel seçim → seçilen plan."""

    def test_secim_uygulanir(self) -> None:
        loop = AgentLoop()
        router = _SahteRouter(_UC_PLAN)
        dusunucu = _SahteDusunucu("### PLAN 2 kazanır")
        loop._router = router
        loop.attach_thinker(dusunucu)

        plan, meta = loop._ask_plan("görev", [], "")

        self.assertEqual(plan, "### PLAN 2 kazanır")
        self.assertEqual(meta.get("source"), "local")
        self.assertAlmostEqual(meta.get("margin"), 0.4, places=3)
        # ⚠ Maliyet: tek LLM çağrısı (adaylar aynı yanıttan ayrıştırılır)
        self.assertEqual(router.cagri, 1)
        self.assertEqual(dusunucu.cagri, 1)
        # ⚠ Yerel modele **3 aday** gitmeli
        self.assertEqual(len(dusunucu.gorulen), 3)

    def test_dusunucu_hatasi_geriduser(self) -> None:
        """Düşünme hata verirse ilk aday kullanılır ve not yazılır."""
        loop = AgentLoop()
        loop._router = _SahteRouter(_UC_PLAN)
        loop.attach_thinker(_BozukDusunucu())

        plan, meta = loop._ask_plan("görev", [], "")

        self.assertIn("Önce oku", plan)
        self.assertIn("düşünme hatası", str(meta.get("note", "")))

    def test_ayristirilamayan_metin(self) -> None:
        """Başlıksız yanıt: düşünme çağrılmaz, metin olduğu gibi döner."""
        loop = AgentLoop()
        loop._router = _SahteRouter("tek parça plan")
        dusunucu = _SahteDusunucu("kullanılmamalı")
        loop.attach_thinker(dusunucu)

        plan, meta = loop._ask_plan("görev", [], "")

        self.assertEqual(plan, "tek parça plan")
        self.assertEqual(dusunucu.cagri, 0, "tek adayda seçim gereksiz")

    def test_attach_none_kapatir(self) -> None:
        loop = AgentLoop()
        loop.attach_thinker(_SahteDusunucu("x"))
        loop.attach_thinker(None)
        loop._router = _SahteRouter("tek plan")
        plan, meta = loop._ask_plan("görev", [], "")
        self.assertEqual(plan, "tek plan")
        self.assertEqual(meta, {})


class TestOtomatikPlan(unittest.TestCase):
    """Uzun görevde plan kendiliğinden devreye girer."""

    def test_esikler(self) -> None:
        self.assertGreaterEqual(AgentLoop.PLAN_CANDIDATES, 2)
        self.assertGreater(AgentLoop.AUTO_PLAN_CHARS, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
