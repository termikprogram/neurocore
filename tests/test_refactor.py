"""
Refactor parite testleri — "taşıdım" mı, "sildim" mi?
=====================================================
Gerçek olay (TitoWarWeb): ajan 706 satır sildi, 35 satır ekledi ve "taşıdım"
dedi. Bu testler denetimin o kaybı yakaladığını garanti eder.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurocore.agent import refactor


def _numstat(*satirlar: tuple) -> str:
    return "\n".join(f"{e}\t{s}\t{y}" for e, s, y in satirlar)


class TestNumstatDenetle(unittest.TestCase):

    def test_kayip_suphesi(self) -> None:
        """706 silinen / 35 eklenen → kayıp şüphesi (gerçek olay)."""
        rapor = refactor.numstat_denetle(
            _numstat((35, 706, "scripts/player/player_controller.gd")))
        self.assertTrue(rapor["kayip_suphesi"])
        self.assertEqual(rapor["silinen"], 706)
        self.assertEqual(rapor["eklenen"], 35)
        self.assertLess(rapor["oran"], 0.6)
        self.assertIn("parite", rapor["mesaj"].lower().replace("i̇", "i"))

    def test_saglikli_tasima(self) -> None:
        """Taşıma doğruysa (silinen ≈ eklenen) şüphe yok."""
        rapor = refactor.numstat_denetle(_numstat(
            (0, 400, "eski.gd"), (395, 2, "yeni.gd")))
        self.assertFalse(rapor["kayip_suphesi"])
        self.assertGreater(rapor["oran"], 0.9)

    def test_kucuk_degisiklik_denetlenmez(self) -> None:
        rapor = refactor.numstat_denetle(_numstat(
            (5, 20, "a.py")))
        self.assertFalse(rapor["kayip_suphesi"])

    def test_kod_disi_dosyalar_sayilmaz(self) -> None:
        """Sadece kod uzantıları dikkate alınır (md/txt gürültü yapmaz)."""
        rapor = refactor.numstat_denetle(_numstat(
            (0, 900, "README.md"), (0, 900, "data.json")))
        self.assertEqual(rapor["silinen"], 0)
        self.assertFalse(rapor["kayip_suphesi"])

    def test_tamamen_silinen_dosya(self) -> None:
        rapor = refactor.numstat_denetle(_numstat(
            (0, 300, "silinen.gd"), (10, 2, "kalan.gd")))
        self.assertIn("silinen.gd", rapor["silen_dosyalar"])

    def test_ikili_dosya_satirlari(self) -> None:
        """`-` içeren numstat satırları çökertmez."""
        rapor = refactor.numstat_denetle("-	-	resim.png\n")
        self.assertTrue(rapor["ok"])
        self.assertEqual(rapor["eklenen"], 0)

    def test_bos_cikti(self) -> None:
        rapor = refactor.numstat_denetle("")
        self.assertTrue(rapor["ok"])
        self.assertFalse(rapor["kayip_suphesi"])
        self.assertEqual(rapor["oran"], 1.0)


class TestPariteKontrol(unittest.TestCase):

    def test_eksik_fonksiyon_bulunur(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-parite-"))
        (kok / "eski.gd").write_text(
            "func a():\n\tpass\n\nfunc b():\n\tpass\n\nfunc c():\n\tpass\n",
            encoding="utf-8")
        (kok / "yeni.gd").write_text(
            "func a():\n\tpass\n\nfunc c():\n\tpass\n", encoding="utf-8")
        sonuc = refactor.parite_kontrol(kok, "eski.gd", "yeni.gd")
        self.assertTrue(sonuc["ok"])
        self.assertEqual(sonuc["eksik"], ["b"])
        self.assertIn("fonksiyon", sonuc["mesaj"])

    def test_tam_parite(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-parite2-"))
        icerik = "func a():\n\tpass\n\nfunc b():\n\tpass\n"
        (kok / "eski.gd").write_text(icerik, encoding="utf-8")
        (kok / "yeni.gd").write_text(icerik + "\nfunc c():\n\tpass\n",
                                     encoding="utf-8")
        sonuc = refactor.parite_kontrol(kok, "eski.gd", "yeni.gd")
        self.assertEqual(sonuc["eksik"], [])
        self.assertEqual(sonuc["fazla"], ["c"])

    def test_olmayan_dosya(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-parite3-"))
        sonuc = refactor.parite_kontrol(kok, "yok.gd", "yok2.gd")
        self.assertFalse(sonuc["ok"])


class TestGitNumstat(unittest.TestCase):

    def test_git_repo_degilse_bos(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-git-"))
        self.assertEqual(refactor.git_numstat(kok), "")

    def test_git_repo_varsa_cikti(self) -> None:
        import shutil
        import subprocess

        if not shutil.which("git"):
            self.skipTest("git yok")
        kok = Path(tempfile.mkdtemp(prefix="nc-git2-"))
        subprocess.run(["git", "init", "-q", str(kok)], check=False)
        subprocess.run(["git", "-C", str(kok), "config", "user.email",
                        "t@t"], check=False)
        subprocess.run(["git", "-C", str(kok), "config", "user.name", "t"],
                       check=False)
        (kok / "a.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(kok), "add", "."], check=False)
        subprocess.run(["git", "-C", str(kok), "commit", "-qm", "ilk"],
                       check=False)
        (kok / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        ham = refactor.git_numstat(kok)
        self.assertIn("a.py", ham)


if __name__ == "__main__":
    unittest.main(verbosity=2)

class TestCokDilliParite(unittest.TestCase):
    """(b) Parite kontrolü yalnız GDScript için değil."""

    def _kok(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="nc-cok-"))

    def test_python_parite(self) -> None:
        kok = self._kok()
        (kok / "eski.py").write_text(
            "def a():\n    pass\n\nclass K:\n    def m(self):\n        pass\n",
            encoding="utf-8")
        (kok / "yeni.py").write_text("def a():\n    pass\n", encoding="utf-8")
        r = refactor.parite_kontrol(kok, "eski.py", "yeni.py")
        self.assertIn("m", r["eksik"])
        self.assertNotIn("a", r["eksik"])

    def test_js_parite(self) -> None:
        kok = self._kok()
        (kok / "eski.js").write_text(
            "function a() {}\nexport function b() {}\nconst c = () => 1;\n",
            encoding="utf-8")
        (kok / "yeni.js").write_text("function a() {}\n", encoding="utf-8")
        r = refactor.parite_kontrol(kok, "eski.js", "yeni.js")
        self.assertIn("b", r["eksik"])
        self.assertIn("c", r["eksik"])

    def test_rust_go(self) -> None:
        kok = self._kok()
        (kok / "e.rs").write_text("pub fn a() {}\nfn b() {}\n", encoding="utf-8")
        (kok / "y.rs").write_text("pub fn a() {}\n", encoding="utf-8")
        r = refactor.parite_kontrol(kok, "e.rs", "y.rs")
        self.assertIn("b", r["eksik"])
        (kok / "e.go").write_text("func A() {}\nfunc B() {}\n", encoding="utf-8")
        (kok / "y.go").write_text("func A() {}\n", encoding="utf-8")
        r2 = refactor.parite_kontrol(kok, "e.go", "y.go")
        self.assertIn("B", r2["eksik"])
