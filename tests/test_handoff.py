"""
Proje teslimi testleri — başlatma betikleri ve sunum
====================================================
Ağ yok; geçici dizinde proje iskeleti kurulur.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neurocore.agent.handoff import (
    KORUNAN,
    komutlari_bul,
    sunum_html,
    sunum_md,
    yaz,
)


def _node_projesi() -> Path:
    kok = Path(tempfile.mkdtemp(prefix="nc-hand-"))
    (kok / "package.json").write_text(json.dumps({
        "name": "demo",
        "scripts": {"start": "node server.js", "test": "node --test"},
        "dependencies": {"ws": "^8"},
    }), encoding="utf-8")
    (kok / "server.js").write_text("//", encoding="utf-8")
    (kok / "tests").mkdir()
    return kok


def _kart(kok: Path) -> dict:
    return {
        "proje": kok.name, "gorev": "balon oyunu yap", "durum": "tamamlandı",
        "adim": 40, "komutlar": ["npm start", "npm test"],
        "dosyalar": ["server.js", "src/game.js"],
        "mimari": {"yigin": ["node"], "giris_noktalari": ["server.js"],
                   "klasorler": [{"ad": "src", "dosya": 4}],
                   "toplam_dosya": 12},
    }


class TestKomutKesfi(unittest.TestCase):

    def test_node(self) -> None:
        k = komutlari_bul(_node_projesi())
        self.assertEqual(k["calistir"], ["npm start"])
        self.assertEqual(k["test"], ["npm test"])
        self.assertEqual(k["kurulum"], ["npm install"])
        self.assertEqual(k["dil"], "node")

    def test_duz_server(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-hand2-"))
        (kok / "server.js").write_text("//", encoding="utf-8")
        k = komutlari_bul(kok)
        self.assertEqual(k["calistir"], ["node server.js"])

    def test_python(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-hand3-"))
        (kok / "main.py").write_text("print(1)", encoding="utf-8")
        k = komutlari_bul(kok)
        self.assertEqual(k["calistir"], ["python main.py"])
        self.assertEqual(k["dil"], "python")

    def test_bos_proje(self) -> None:
        k = komutlari_bul(Path(tempfile.mkdtemp(prefix="nc-hand4-")))
        self.assertEqual(k["calistir"], [])


class TestYazma(unittest.TestCase):

    def test_dort_dosya(self) -> None:
        kok = _node_projesi()
        sonuc = yaz(kok, _kart(kok), {"duration": "142 sn"})
        self.assertTrue(sonuc["ok"])
        # ⚠ README de teslim parçası: yoksa üretilir (5 dosya)
        self.assertEqual(sorted(sonuc["yazilan"]),
                         sorted(["baslat.ps1", "baslat.bat", "SUNUM.md",
                                 "sunum.html", "README.md"]))
        for ad in sonuc["yazilan"]:
            self.assertTrue((kok / ad).exists(), f"{ad} yok")

    def test_var_olan_ezilmez(self) -> None:
        kok = _node_projesi()
        (kok / "README.md").write_text("# kendi readme", encoding="utf-8")
        (kok / "baslat.ps1").write_text("# elle yazıldı", encoding="utf-8")
        sonuc = yaz(kok, _kart(kok))
        self.assertIn("baslat.ps1", sonuc["atlanan"])
        self.assertEqual((kok / "baslat.ps1").read_text(encoding="utf-8"),
                         "# elle yazıldı")
        self.assertEqual((kok / "README.md").read_text(encoding="utf-8"),
                         "# kendi readme")

    def test_komut_yoksa_yazmaz(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-hand5-"))
        sonuc = yaz(kok, _kart(kok))
        self.assertFalse(sonuc["ok"])
        self.assertIn("çalıştırma komutu", sonuc["error"])

    def test_betik_icerigi(self) -> None:
        kok = _node_projesi()
        yaz(kok, _kart(kok))
        ps1 = (kok / "baslat.ps1").read_text(encoding="utf-8")
        bat = (kok / "baslat.bat").read_text(encoding="utf-8")
        self.assertIn("npm start", ps1)
        self.assertIn("Set-Location", ps1)
        self.assertIn("baslat.ps1", bat)
        # ⚠ Betikler veri silmez
        self.assertNotIn("Remove-Item", ps1)
        self.assertNotIn("del ", bat.lower())

    def test_korunan_listesi(self) -> None:
        self.assertEqual(set(KORUNAN),
                         {"baslat.ps1", "baslat.bat", "SUNUM.md", "sunum.html"})


class TestSunum(unittest.TestCase):

    def test_markdown_bolumleri(self) -> None:
        kok = _node_projesi()
        md = sunum_md(_kart(kok), {"duration": "142 sn"}, komutlari_bul(kok))
        for baslik in ("Ne yapıldı", "Mimari", "Nasıl çalıştırılır",
                       "Sıradaki adımlar"):
            self.assertIn(baslik, md)
        self.assertIn("npm start", md)
        self.assertIn(".\\baslat.bat", md)

    def test_html_tek_dosya(self) -> None:
        kok = _node_projesi()
        html = sunum_html(_kart(kok), "özet")
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn(kok.name, html)
        self.assertIn("NeuroCore", html)
        # dış kaynak yok (paylaşılabilir tek dosya)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)

class TestGodotProjesi(unittest.TestCase):
    """⚠ Godot projesi desteği (web build servisi)."""

    def _godot(self, web: bool = True) -> Path:
        kok = Path(tempfile.mkdtemp(prefix="nc-godot-"))
        (kok / "project.godot").write_text("[application]\nconfig/name=\"x\"\n",
                                           encoding="utf-8")
        if web:
            (kok / "build" / "web").mkdir(parents=True)
            (kok / "build" / "web" / "index.html").write_text("<canvas>",
                                                              encoding="utf-8")
        return kok

    def test_web_build_servis_komutu(self) -> None:
        k = komutlari_bul(self._godot(web=True))
        self.assertEqual(k["dil"], "godot")
        self.assertIn("http.server", k["calistir"][0])
        self.assertIn("build/web", k["calistir"][0])

    def test_export_komutu(self) -> None:
        k = komutlari_bul(self._godot(web=False))
        self.assertIn("--export-release", k["calistir"][0])
        self.assertIn("Web", k["calistir"][0])

    def test_baslat_ps1_port_8000(self) -> None:
        kok = self._godot(web=True)
        yaz(kok, {"proje": kok.name, "gorev": "godot oyun"})
        ps1 = (kok / "baslat.ps1").read_text(encoding="utf-8")
        self.assertIn("8000", ps1)
        self.assertIn("http.server", ps1)

    def test_readme_uretir(self) -> None:
        kok = self._godot(web=True)
        yaz(kok, {"proje": kok.name, "gorev": "godot oyun"})
        readme = (kok / "README.md").read_text(encoding="utf-8")
        self.assertIn("#", readme)
        self.assertIn("baslat", readme)
