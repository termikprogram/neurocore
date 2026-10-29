"""
Ortam ön denetimi testleri — plan sonrası "ne kurulu, ne eksik?"
===============================================================
Gerçek alt süreç çağrıları yapılmaz; `which`/sürüm çağrıları sahtelenir.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from neurocore.agent import envcheck


class TestSistemBilgisi(unittest.TestCase):

    def test_os_ve_paket_yoneticisi(self) -> None:
        bilgi = envcheck.sistem_bilgisi()
        self.assertIn(bilgi["os"], ("windows", "linux", "macos"))
        self.assertIsInstance(bilgi["paket_yoneticileri"], list)
        self.assertTrue(bilgi["python"])

    @mock.patch.object(envcheck, "platform")
    def test_windows_winget(self, sahte: Any) -> None:
        sahte.system.return_value = "Windows"
        sahte.version.return_value = "10.0.26200"
        sahte.python_version.return_value = "3.10.11"
        with mock.patch.object(envcheck.shutil, "which",
                               side_effect=lambda x: "winget.exe" if x == "winget" else None):
            bilgi = envcheck.sistem_bilgisi()
        self.assertEqual(bilgi["os"], "windows")
        self.assertIn("winget", bilgi["paket_yoneticileri"])


class TestAracDurumu(unittest.TestCase):

    def test_kurulu_arac(self) -> None:
        with mock.patch.object(envcheck.shutil, "which",
                               return_value=r"C:\tools\python.exe"), \
             mock.patch.object(envcheck, "_surum", return_value="Python 3.12.1"):
            d = envcheck.arac_durumu("python")
        self.assertTrue(d["kurulu"])
        self.assertIn("3.12", d["surum"])
        self.assertFalse(d["path_disi"])

    def test_eksik_arac(self) -> None:
        with mock.patch.object(envcheck.shutil, "which", return_value=None), \
             mock.patch.object(envcheck.Path, "exists", return_value=False):
            d = envcheck.arac_durumu("dotnet")
        self.assertFalse(d["kurulu"])

    def test_bilinen_yol_path_disi(self) -> None:
        """⚠ Ölçüldü: paketli node PATH'te değil ama kurulu."""
        gercek = r"C:\paketli\node.exe"
        with mock.patch.object(envcheck.shutil, "which", return_value=None), \
             mock.patch.dict(envcheck.BILINEN_YOLLAR,
                             {"node": (gercek,)}, clear=False), \
             mock.patch.object(envcheck.Path, "exists", return_value=True), \
             mock.patch.object(envcheck, "_surum", return_value="v24.21.0"):
            d = envcheck.arac_durumu("node")
        self.assertTrue(d["kurulu"])
        self.assertTrue(d["path_disi"])


class TestKurulumKomutu(unittest.TestCase):

    def test_winget(self) -> None:
        komut = envcheck._kurulum_komutu(
            "node", {"paket_yoneticileri": ["winget"]})
        self.assertIn("winget install", komut)
        self.assertIn("OpenJS.NodeJS.LTS", komut)

    def test_apt(self) -> None:
        komut = envcheck._kurulum_komutu(
            "git", {"paket_yoneticileri": ["apt"]})
        self.assertIn("sudo apt install git", komut)

    def test_brew(self) -> None:
        komut = envcheck._kurulum_komutu(
            "node", {"paket_yoneticileri": ["brew"]})
        self.assertIn("brew install node", komut)

    def test_paket_yoneticisi_yok(self) -> None:
        self.assertEqual(envcheck._kurulum_komutu(
            "godot", {"paket_yoneticileri": []}), "")


class TestDenetle(unittest.TestCase):

    def _node_projesi(self) -> Path:
        kok = Path(tempfile.mkdtemp(prefix="nc-env-"))
        (kok / "package.json").write_text(json.dumps({
            "name": "x", "dependencies": {"ws": "^8"}}), encoding="utf-8")
        return kok

    def test_node_bagimliligi_eksik(self) -> None:
        kok = self._node_projesi()
        with mock.patch.object(envcheck, "arac_durumu",
                               side_effect=lambda a: {"ad": a, "kurulu": True,
                                                      "surum": "v1", "yol": "x",
                                                      "path_disi": False}):
            r = envcheck.denetle(kok, "node oyun")
        adlar = [e["ad"] for e in r["eksik"]]
        self.assertTrue(any("npm paketi" in a for a in adlar),
                        f"eksik: {adlar}")
        self.assertIn("npm install", r["eksik"][0]["komut"])

    def test_node_bagimliligi_kurulu(self) -> None:
        kok = self._node_projesi()
        (kok / "node_modules").mkdir()
        with mock.patch.object(envcheck, "arac_durumu",
                               side_effect=lambda a: {"ad": a, "kurulu": True,
                                                      "surum": "v1", "yol": "x",
                                                      "path_disi": False}):
            r = envcheck.denetle(kok, "node oyun")
        self.assertTrue(r["hazir"], f"eksik: {r['eksik']}")

    def test_godot_projesi(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-env2-"))
        (kok / "project.godot").write_text("[application]", encoding="utf-8")
        with mock.patch.object(envcheck, "arac_durumu",
                               side_effect=lambda a: {"ad": a, "kurulu": True,
                                                      "surum": "4.8", "yol": "x",
                                                      "path_disi": True}), \
             mock.patch.object(envcheck, "_godot_sablonlari",
                               return_value={"kurulu": False, "surum": ""}):
            r = envcheck.denetle(kok, "godot oyun")
        adlar = [e["ad"] for e in r["eksik"]]
        self.assertIn("godot export şablonları", adlar)

    def test_python_bagimliligi(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-env3-"))
        (kok / "requirements.txt").write_text("olmayan-paket-abc==1.0\n",
                                               encoding="utf-8")
        with mock.patch.object(envcheck, "arac_durumu",
                               side_effect=lambda a: {"ad": a, "kurulu": True,
                                                      "surum": "3.12", "yol": "x",
                                                      "path_disi": False}):
            r = envcheck.denetle(kok, "python script")
        adlar = [e["ad"] for e in r["eksik"]]
        self.assertIn("olmayan-paket-abc", adlar)

    def test_bos_proje(self) -> None:
        r = envcheck.denetle(Path(tempfile.mkdtemp(prefix="nc-env4-")), "")
        self.assertTrue(r["hazir"])


class TestMetin(unittest.TestCase):

    def test_hazir_metni(self) -> None:
        rapor = {"sistem": {"os": "windows", "paket_yoneticileri": ["winget"]},
                 "araclar": [{"ad": "python", "kurulu": True, "surum": "3.10"}],
                 "eksik": [], "uyarilar": []}
        metin = envcheck.metin(rapor)
        self.assertIn("ORTAM ÖN DENETİMİ", metin)
        self.assertIn("hazır", metin)

    def test_eksik_metni(self) -> None:
        rapor = {"sistem": {"os": "windows", "paket_yoneticileri": ["winget"]},
                 "araclar": [],
                 "eksik": [{"ad": "node", "tur": "araç", "oneri": "yok",
                            "komut": "winget install X"}],
                 "uyarilar": ["not"]}
        metin = envcheck.metin(rapor)
        self.assertIn("EKSİK", metin)
        self.assertIn("winget install X", metin)
        self.assertIn("sor", metin.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
