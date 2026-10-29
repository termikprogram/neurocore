"""
Kontrol noktası testleri — kesilen işi kaldığı yerden sürdürme
==============================================================
Ağ yok.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from typing import Any
from pathlib import Path

from neurocore.agent.checkpoint import (
    MAX_ADIM,
    MIN_BENZERLIK,
    Kontrol,
    benzer,
    devam_edilebilir,
    devam_istegi,
)


def _kontrol(adim: int = 3, tamam: bool = False) -> Kontrol:
    return Kontrol(
        gorev="balon oyunu yap: WASD, büyük küçüğü yutar",
        workspace="E:/proje",
        adimlar=[{"tool": f"code.create_file{i}", "ozet": f"dosya {i}",
                  "ok": True} for i in range(adim)],
        dosyalar=["server.js", "src/oyun.js"],
        olgular=["node 24 kurulu", "ws paketi eksik"],
        acik_sorular=["kaç oyuncu?"],
        tamamlandi=tamam,
        sebep="tamamlandı" if tamam else "kullanıcı durdurdu",
    )


class TestKontrol(unittest.TestCase):

    def test_ozet_nerede_kaldik(self) -> None:
        metin = _kontrol().ozet()
        self.assertIn("KALDIĞIN YER", metin)
        self.assertIn("3 araç adımı", metin)
        self.assertIn("code.create_file2", metin)
        self.assertIn("server.js", metin)
        self.assertIn("Devam et", metin)

    def test_son_adim(self) -> None:
        self.assertIn("code.create_file2", _kontrol().son_adim())

    def test_bos_kontrol(self) -> None:
        self.assertEqual(Kontrol().ozet(), "")
        self.assertEqual(Kontrol().son_adim(), "")

    def test_tamamlanan_ozeti(self) -> None:
        metin = _kontrol(tamam=True).ozet()
        self.assertIn("tamamlandı", metin)
        self.assertNotIn("TEKRARLAMA", metin)

    def test_adim_siniri(self) -> None:
        k = _kontrol(adim=MAX_ADIM + 40)
        self.assertLessEqual(len(k.to_dict()["adimlar"]), MAX_ADIM)

    def test_kaydet_yukle(self) -> None:
        yol = Path(tempfile.mkdtemp(prefix="nc-ck-")) / "k.json"
        k = _kontrol()
        self.assertTrue(k.kaydet(yol))
        geri = Kontrol.yukle(yol)
        self.assertIsNotNone(geri)
        self.assertEqual(geri.gorev, k.gorev)
        self.assertEqual(len(geri.adimlar), 3)
        self.assertEqual(geri.dosyalar, ["server.js", "src/oyun.js"])
        self.assertFalse(geri.tamamlandi)

    def test_bozuk_dosya(self) -> None:
        yol = Path(tempfile.mkdtemp(prefix="nc-ck2-")) / "b.json"
        yol.write_text("{bozuk", encoding="utf-8")
        self.assertIsNone(Kontrol.yukle(yol))

    def test_arsiv(self) -> None:
        yol = Path(tempfile.mkdtemp(prefix="nc-ck3-")) / "a.jsonl"
        Kontrol.arsivle(_kontrol(), yol)
        Kontrol.arsivle(_kontrol(adim=5), yol)
        satirlar = [s for s in yol.read_text(encoding="utf-8").splitlines() if s]
        self.assertEqual(len(satirlar), 2)
        self.assertEqual(json.loads(satirlar[0])["gorev"][:5], "balon")


class TestDevamKurali(unittest.TestCase):

    def test_benzer_gorev_devam(self) -> None:
        k = _kontrol()
        self.assertTrue(devam_edilebilir(
            k, "balon oyunu yap: WASD, büyük küçüğü yutar, renkler farklı"))
        self.assertGreaterEqual(
            benzer(k.gorev, "balon oyunu yap: WASD"), 0.35)

    def test_devam_et_acik_istek(self) -> None:
        """⚠ "devam et" benzerliği 0 olsa da sürdürmeli."""
        k = _kontrol()
        self.assertLess(benzer(k.gorev, "devam et"), 0.2)
        self.assertTrue(devam_edilebilir(k, "devam et"))
        self.assertTrue(devam_edilebilir(k, "kaldığın yerden devam"))

    def test_alakasiz_uzun_gorev_devam_etmez(self) -> None:
        k = _kontrol()
        self.assertFalse(devam_edilebilir(
            k, "veritabanı yedeği al ve raporla, sonra disk alanını temizle"))

    def test_alakasiz_gorev_devam_etmez(self) -> None:
        self.assertFalse(devam_edilebilir(_kontrol(), "veritabanı yedeği al"))

    def test_tamamlanan_devam_etmez(self) -> None:
        k = _kontrol(tamam=True)
        self.assertFalse(devam_edilebilir(k, k.gorev))

    def test_bos_kontrol(self) -> None:
        self.assertFalse(devam_edilebilir(None, "görev"))
        self.assertFalse(devam_edilebilir(Kontrol(gorev=""), "görev"))

    def test_adimsiz_devam_etmez(self) -> None:
        k = _kontrol(adim=0)
        self.assertFalse(devam_edilebilir(k, k.gorev))


class TestDonguKontrolu(unittest.TestCase):

    def _dongu(self) -> Any:
        from neurocore.agent.loop import AgentLoop

        loop = AgentLoop()
        loop._workspace = "."
        loop._config = None
        loop._event_bus = None
        loop._kontrol_yol = Path(tempfile.mkdtemp(prefix="nc-ck4-")) / "k.json"
        loop._kontrol_arsiv = loop._kontrol_yol.parent / "a.jsonl"
        return loop

    def test_yeni_calisma_kaydi(self) -> None:
        from neurocore.agent.loop import AgentRun

        loop = self._dongu()
        loop._kontrol_hazirla("kısa görev", AgentRun(task="t"))
        self.assertIsNotNone(loop._kontrol)
        self.assertEqual(loop._kontrol.gorev, "kısa görev")
        self.assertEqual(loop._counter.get("kontrol_devam", 0), 0)

    def test_devam_geri_yukler(self) -> None:
        from neurocore.agent.loop import AgentRun

        loop = self._dongu()
        _kontrol().kaydet(loop._kontrol_yol)
        loop._kontrol_hazirla("balon oyunu yap: WASD, büyük küçüğü yutar",
                              AgentRun(task="t"))
        self.assertEqual(loop._counter.get("kontrol_devam", 0), 1)
        self.assertEqual(len(loop._kontrol.adimlar), 3)

    def test_devam_mesaji_adim_ekler(self) -> None:
        from neurocore.agent.cognition import HedefDurumu
        from neurocore.agent.loop import AgentRun

        loop = self._dongu()
        loop._hedef = HedefDurumu(hedef="g")
        loop._kontrol = _kontrol()
        run = AgentRun(task="t")
        mesajlar: list = []
        loop._kontrol_mesaji(mesajlar, run)
        self.assertEqual(len(mesajlar), 1)
        self.assertIn("KALDIĞIN YER", mesajlar[0].content)
        self.assertEqual(loop._counter["kontrol_devam_mesaji"], 1)
        self.assertEqual(run.steps[-1].kind, "resume")

    def test_guncelle_diske_yazar(self) -> None:
        from neurocore.agent.loop import AgentRun, AgentStep

        loop = self._dongu()
        loop._kontrol_hazirla("görev", AgentRun(task="t"))
        run = AgentRun(task="t")
        run.steps = [AgentStep(index=0, kind="tool", tool="code.read_file",
                               args={"path": "a.js"}, ok=True)]
        run.files_changed = ["a.js"]
        loop._kontrol_guncelle(run, "devam ediyor")
        self.assertTrue(loop._kontrol_yol.exists())
        veri = json.loads(loop._kontrol_yol.read_text(encoding="utf-8"))
        self.assertEqual(veri["sebep"], "devam ediyor")
        self.assertEqual(veri["adimlar"][0]["tool"], "code.read_file")
        self.assertIn("a.js", veri["dosyalar"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

class TestProjeKarti(unittest.TestCase):
    """⚠ Proje klasöründe .neurocore/proje.json — projenin künyesi."""

    def _dongu(self, kok: Path) -> Any:
        from neurocore.agent.loop import AgentLoop

        loop = AgentLoop()
        loop._config = None
        loop._event_bus = None
        loop.set_workspace(str(kok))
        return loop

    def test_proje_karti_icerigi(self) -> None:
        import tempfile

        kok = Path(tempfile.mkdtemp(prefix="nc-proje-"))
        (kok / "package.json").write_text(
            '{"name":"x","scripts":{"start":"node server.js","test":"node --test"}}',
            encoding="utf-8")
        loop = self._dongu(kok)
        kart = loop.proje_karti()
        self.assertEqual(kart["proje"], kok.name)
        self.assertIn("npm start", kart["komutlar"])
        self.assertTrue(any("npm" in k and "test" in k for k in kart["komutlar"]))

    def test_proje_dosyalari_yazilir(self) -> None:
        import tempfile

        kok = Path(tempfile.mkdtemp(prefix="nc-proje2-"))
        loop = self._dongu(kok)
        from neurocore.agent.loop import AgentRun

        run = AgentRun(task="t")
        loop._kontrol_hazirla("proje testi", run)
        # ⚠ kontrol.json ancak güncelleme ile diske yazılır (tasarım)
        loop._kontrol_guncelle(run, "devam ediyor")
        loop._proje_kaydet(run)
        for ad in ("proje.json", "kontrol.json"):
            self.assertTrue((kok / ".neurocore" / ad).exists(), f"{ad} yok")
        kart = json.loads((kok / ".neurocore" / "proje.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(kart["kok"], str(kok.resolve()))

    def test_proje_mesaji_eklenir(self) -> None:
        import tempfile
        from neurocore.agent.loop import AgentRun

        kok = Path(tempfile.mkdtemp(prefix="nc-proje3-"))
        loop = self._dongu(kok)
        loop._proje_yol.parent.mkdir(parents=True, exist_ok=True)
        loop._proje_yol.write_text(json.dumps({
            "proje": "demo", "gorev": "oyun yap", "durum": "devam ediyor",
            "adim": 5, "son_adim": "code.read_file", "komutlar": ["npm start"],
        }), encoding="utf-8")
        mesajlar: list = []
        loop._proje_mesaji(mesajlar, AgentRun(task="t"))
        self.assertEqual(len(mesajlar), 1)
        self.assertIn("PROJE KARTI", mesajlar[0].content)
        self.assertIn("npm start", mesajlar[0].content)

    def test_set_workspace_yollari_tasir(self) -> None:
        import tempfile

        kok = Path(tempfile.mkdtemp(prefix="nc-proje4-"))
        loop = self._dongu(kok)
        self.assertIn(".neurocore", str(loop._kontrol_yol))
        self.assertEqual(Path(loop._proje_yol).parent.name, ".neurocore")
        self.assertEqual(loop._workspace, str(kok.resolve()))

class TestMimariOzet(unittest.TestCase):
    """⚠ Proje kartındaki mimari: giriş noktaları, klasörler, yığın."""

    def test_node_projesi(self) -> None:
        import tempfile
        from neurocore.agent.loop import AgentLoop

        kok = Path(tempfile.mkdtemp(prefix="nc-mim-"))
        (kok / "package.json").write_text('{"name":"x"}', encoding="utf-8")
        (kok / "server.js").write_text("//", encoding="utf-8")
        (kok / "public").mkdir()
        (kok / "public" / "index.html").write_text("<h1>", encoding="utf-8")
        (kok / "test").mkdir()
        (kok / "node_modules").mkdir()
        dongu = AgentLoop()
        dongu._config = None
        dongu.set_workspace(str(kok))
        m = dongu._mimari_ozet(Path(kok).resolve())
        self.assertIn("node", m["yigin"])
        self.assertIn("server.js", m["giris_noktalari"])
        self.assertIn("test", m["test_klasoru"])
        adlar = [k["ad"] for k in m["klasorler"]]
        self.assertIn("public", adlar)
        self.assertNotIn("node_modules", adlar)

    def test_python_projesi(self) -> None:
        import tempfile
        from neurocore.agent.loop import AgentLoop

        kok = Path(tempfile.mkdtemp(prefix="nc-mim2-"))
        (kok / "requirements.txt").write_text("x", encoding="utf-8")
        (kok / "main.py").write_text("print(1)", encoding="utf-8")
        dongu = AgentLoop()
        dongu._config = None
        dongu.set_workspace(str(kok))
        m = dongu._mimari_ozet(Path(kok).resolve())
        self.assertIn("python", m["yigin"])
        self.assertIn("main.py", m["giris_noktalari"])
        self.assertTrue(m["readme"] is False)
