"""
Proje kod indeksi testleri — büyük projede ilişkili kodu bulma
=============================================================
Ölçüm (2000 dosya / 5,99 M satır): ilk kurulum 39,8 sn · artımlı 0,15 sn ·
sorgular 3-8 ms · veritabanı 13,4 MB.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurocore.agent import codeindex


def _proje(dosya: int = 25) -> Path:
    """Semboller ve import ilişkileri olan küçük proje üretir."""
    kok = Path(tempfile.mkdtemp(prefix="nc-kodidx-"))
    for i in range(dosya):
        hedef = kok / f"paket{i % 3}"
        hedef.mkdir(exist_ok=True)
        icerik = [
            f"# modul {i}",
            f"from paket{(i + 1) % 3}.mod_{i + 1} import Sinif{i + 1}",
            "",
            f"class Sinif{i}:",
            f"    def metod_{i}(self):",
            "        return 1",
            "",
            f"def fonksiyon_{i}(a):",
            "    return a",
            "",
            f"AYAR_{i} = {i}",
        ]
        (hedef / f"mod_{i}.py").write_text("\n".join(icerik), encoding="utf-8")
    return kok


class TestIndeks(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.kok = _proje()
        cls.ozet = codeindex.dizinle(cls.kok)

    def test_dosya_sayisi(self) -> None:
        self.assertEqual(self.ozet["dosya"], 25)
        self.assertGreater(self.ozet["toplam_satir"], 200)

    def test_semboller(self) -> None:
        self.assertGreater(self.ozet["sembol"], 25 * 3)
        self.assertTrue(self.ozet["fts"])

    def test_iliski(self) -> None:
        self.assertGreater(self.ozet["iliski"], 0)

    def test_db_olustu(self) -> None:
        db = self.kok / ".neurocore" / "index" / "kod.db"
        self.assertTrue(db.exists())
        self.assertGreater(db.stat().st_size, 0)


class TestSorgular(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.kok = _proje()
        codeindex.dizinle(cls.kok)

    def test_sembol_bul(self) -> None:
        s = codeindex.sembol_ara(self.kok, "Sinif7")
        self.assertGreaterEqual(s["sayi"], 1)
        self.assertTrue(any("mod_7" in r["path"] for r in s["sonuc"]))

    def test_sembol_tur(self) -> None:
        s = codeindex.sembol_ara(self.kok, "fonksiyon_3")
        self.assertTrue(s["sonuc"])
        self.assertEqual(s["sonuc"][0]["tur"], "func")

    def test_referanslar(self) -> None:
        r = codeindex.referanslar(self.kok, "Sinif5")
        self.assertGreaterEqual(r["sayi"], 1)

    def test_iliskili_dosyalar(self) -> None:
        yol = "paket0/mod_0.py"
        il = codeindex.iliskili_dosyalar(self.kok, yol)
        self.assertTrue(il["ok"])
        self.assertTrue(il["import_ettigi"], "import ilişkisi bulunmalı")
        self.assertTrue(il["semboller"])

    def test_proje_haritasi(self) -> None:
        h = codeindex.proje_haritasi(self.kok)
        self.assertEqual(h["dosya"], 25)
        self.assertTrue(h["diller"])
        self.assertEqual(h["diller"][0]["dil"], "python")

    def test_indeks_yokken(self) -> None:
        bos = Path(tempfile.mkdtemp(prefix="nc-bos-"))
        self.assertFalse(codeindex.sembol_ara(bos, "x")["ok"])


class TestArtimli(unittest.TestCase):

    def test_degismeyen_dosya_atlanir(self) -> None:
        kok = _proje(10)
        codeindex.dizinle(kok)
        ikinci = codeindex.dizinle(kok)
        self.assertEqual(ikinci["yeni"], 0)
        self.assertEqual(ikinci["guncel"], 0)

    def test_degisen_dosya_islenir(self) -> None:
        kok = _proje(10)
        codeindex.dizinle(kok)
        hedef = kok / "paket0" / "mod_0.py"
        hedef.write_text(hedef.read_text(encoding="utf-8")
                         + "\ndef yeni_fonksiyon():\n    pass\n",
                         encoding="utf-8")
        sonuc = codeindex.dizinle(kok)
        self.assertEqual(sonuc["guncel"], 1)
        self.assertGreaterEqual(
            codeindex.sembol_ara(kok, "yeni_fonksiyon")["sayi"], 1)

    def test_silinen_dosya_temizlenir(self) -> None:
        kok = _proje(10)
        codeindex.dizinle(kok)
        (kok / "paket0" / "mod_0.py").unlink()
        sonuc = codeindex.dizinle(kok)
        self.assertEqual(sonuc["silinen"], 1)
        self.assertEqual(codeindex.sembol_ara(kok, "Sinif0")["sayi"], 0)


class TestSinirlar(unittest.TestCase):

    def test_binary_dosya_atlanir(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-bin-"))
        (kok / "resim.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        (kok / "kod.py").write_text("def x():\n    pass\n", encoding="utf-8")
        o = codeindex.dizinle(kok)
        self.assertEqual(o["dosya"], 1)

    def test_atlanan_dizinler(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-atla-"))
        (kok / "node_modules").mkdir()
        (kok / "node_modules" / "a.js").write_text("function f(){}",
                                                  encoding="utf-8")
        (kok / "ana.py").write_text("def a():\n    pass\n", encoding="utf-8")
        o = codeindex.dizinle(kok)
        self.assertEqual(o["dosya"], 1)

    def test_cok_buyuk_dosya_yalniz_sayilir(self) -> None:
        """DOSYA_MAKS üstü dosya sembol çıkarmadan satır sayılır."""
        kok = Path(tempfile.mkdtemp(prefix="nc-huge-"))
        dev = kok / "dev.py"
        with dev.open("w", encoding="utf-8") as f:
            for i in range(200_000):          # ~8 MB üstü
                f.write(f"# dolgu {i} xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
        o = codeindex.dizinle(kok)
        self.assertEqual(o["dosya"], 1)
        self.assertGreater(o["toplam_satir"], 100_000)
        self.assertEqual(o["sembol"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
