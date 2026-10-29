"""
Entegrasyon testleri — kabuk kapsamı · MCP istemcisi · faz raporu · adım sınırı
==============================================================================
Ağ gerekmez; MCP için sahte stdio sunucusu (`tools/mcp_sahte.py`) kullanılır.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, List

from neurocore.agent.shell import _kapsam_denetle


class TestKabukKapsami(unittest.TestCase):
    """⚠ Ajan seçilen klasörün üst dizinlerine çıkabiliyordu."""

    KOK = str(Path.cwd())

    def test_serbest_komutlar(self) -> None:
        for komut in ("python build.py", "cd src && python main.py",
                      "git status", "cd sub;python x.py"):
            self.assertEqual(_kapsam_denetle(komut, self.KOK), "",
                             f"yanlışlıkla engellendi: {komut}")

    def test_ust_dizin_engellenir(self) -> None:
        for komut in ("cd ..", "cd ../..", "pushd ..\\other",
                      "Set-Location C:\\Windows", "chdir /etc"):
            self.assertTrue(_kapsam_denetle(komut, self.KOK),
                            f"engellenmedi: {komut}")

    def test_ic_abs_yol_serbest(self) -> None:
        ic = str(Path(self.KOK) / "sub")
        self.assertEqual(_kapsam_denetle(f"cd {ic}", self.KOK), "")

    def test_bos_komut(self) -> None:
        self.assertEqual(_kapsam_denetle("", self.KOK), "")


class TestAdimSiniri(unittest.TestCase):
    def test_varsayilan_sinirsiz(self) -> None:
        from neurocore.agent.loop import AgentLoop, MUTLAK_ADIM

        self.assertEqual(AgentLoop.VARSAYILAN_ADIM, 0,
                         "varsayılan sınırsız olmalı (0)")
        self.assertGreater(MUTLAK_ADIM, 100, "güvenlik tavanı çok küçük")

    def test_faz_raporu_acik(self) -> None:
        from neurocore.agent.loop import AgentLoop

        self.assertGreaterEqual(AgentLoop.FAZ_ADIM, 4)


class TestFazRaporu(unittest.TestCase):

    def _dongu(self) -> Any:
        from neurocore.agent.loop import AgentLoop

        loop = AgentLoop()
        loop._workspace = "."
        loop._config = None
        loop._event_bus = None
        return loop

    def test_faz_raporu_eklenir(self) -> None:
        from neurocore.agent.cognition import HedefDurumu
        from neurocore.agent.llm import LLMMessage
        from neurocore.agent.loop import AgentLoop, AgentRun, AgentStep

        loop = self._dongu()
        loop._hedef = HedefDurumu(hedef="uzun görev")
        loop._hedef.olgu_ekle("dosya 42 satır")
        run = AgentRun(task="t")
        run.steps = [AgentStep(index=i, kind="tool", tool=f"t{i}", ok=True)
                     for i in range(AgentLoop.FAZ_ADIM)]
        mesajlar: List[Any] = []
        loop._faz_raporu("görev", run, mesajlar)
        self.assertEqual(len(mesajlar), 1)
        self.assertIn("FAZ DURUMU", mesajlar[0].content)
        self.assertIn("42 satır", mesajlar[0].content)
        self.assertEqual(loop._counter["faz_raporu"], 1)

    def test_erken_adimda_rapor_yok(self) -> None:
        from neurocore.agent.cognition import HedefDurumu
        from neurocore.agent.llm import LLMMessage
        from neurocore.agent.loop import AgentRun, AgentStep

        loop = self._dongu()
        loop._hedef = HedefDurumu(hedef="g")
        run = AgentRun(task="t")
        run.steps = [AgentStep(index=0, kind="tool", tool="a", ok=True)]
        mesajlar: List[Any] = []
        loop._faz_raporu("görev", run, mesajlar)
        self.assertEqual(mesajlar, [])


class TestMCP(unittest.TestCase):
    """Sahte MCP sunucusuyla protokol testi (initialize/list/call)."""

    @classmethod
    def setUpClass(cls) -> None:
        from neurocore.agent.mcp import MCPIstemci

        cls.betik = str(Path("tools/mcp_sahte.py").resolve())
        cls.istemci = MCPIstemci("sahte", sys.executable, [cls.betik])
        cls.bagli = cls.istemci.baslat()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.istemci.kapat()

    def test_baglanir(self) -> None:
        self.assertTrue(self.bagli, f"bağlanamadı: {self.istemci.hata}")

    def test_arac_listesi(self) -> None:
        araclar = self.istemci.araclari_yukle()
        self.assertEqual([a.ad for a in araclar], ["echo", "topla", "patlat"])

    def test_arac_cagrisi(self) -> None:
        sonuc = self.istemci.cagir("echo", {"metin": "merhaba"})
        self.assertTrue(sonuc["ok"])
        self.assertEqual(sonuc["metin"], "echo: merhaba")

    def test_sayi_donusu(self) -> None:
        sonuc = self.istemci.cagir("topla", {"a": 19, "b": 23})
        self.assertTrue(sonuc["ok"])
        self.assertEqual(sonuc["metin"], "42.0")

    def test_hata_yolu(self) -> None:
        sonuc = self.istemci.cagir("patlat", {})
        self.assertFalse(sonuc["ok"])

    def test_bilinmeyen_arac(self) -> None:
        sonuc = self.istemci.cagir("yok", {})
        self.assertFalse(sonuc["ok"])
        self.assertIn("-32601", str(sonuc.get("error")))

    def test_sema_uretimi(self) -> None:
        araclar = self.istemci.araclari_yukle()
        sema = araclar[0].arac_tanimi("mcp.sahte")
        self.assertEqual(sema["function"]["name"], "mcp.sahte.echo")
        self.assertIn("metin", sema["function"]["parameters"]["required"])


class TestMCPYonetici(unittest.TestCase):

    def test_sunucu_ekle_sil(self) -> None:
        import tempfile
        from neurocore.agent.mcp import MCPYonetici

        yonetici = MCPYonetici(Path(tempfile.mkdtemp(prefix="nc-mcp-")) / "m.json")
        self.assertEqual(yonetici.durum()["sunucu_sayisi"], 0)
        self.assertTrue(yonetici.ekle("sahte", sys.executable,
                                      [str(Path("tools/mcp_sahte.py").resolve())])
                        ["ok"])
        self.assertEqual(yonetici.durum()["sunucu_sayisi"], 1)
        self.assertTrue(yonetici.dene("sahte")["ok"])
        # ad çözümleme: mcp.<sunucu>.<araç>
        sonuc = yonetici.cagir("mcp.sahte.echo", {"metin": "x"})
        self.assertTrue(sonuc["ok"])
        self.assertTrue(yonetici.sil("sahte")["ok"])
        yonetici.kapat()

    def test_gecersiz_ad(self) -> None:
        import tempfile
        from neurocore.agent.mcp import MCPYonetici

        yonetici = MCPYonetici(Path(tempfile.mkdtemp(prefix="nc-mcp2-")) / "m.json")
        self.assertFalse(yonetici.cagir("echo", {})["ok"])
        self.assertFalse(yonetici.cagir("mcp.yok.x", {})["ok"])
        yonetici.kapat()


if __name__ == "__main__":
    unittest.main(verbosity=2)
