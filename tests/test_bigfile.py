"""
Büyük dosya okuma testleri — milyonlarca satırlık dosyalar
==========================================================
Gerçek ölçüm (10.000.000 satır · 427 MB):

```
seyrek indeks kurulumu : 8,2 sn · 0,6 MB bellek · 48 KB yan dosya
rastgele pencere       : 11 ms · 232 KB bellek   (satır 9.999.500)
derin pencere          :  1 ms                   (satır 5.000.000)
akışlı arama           : 6,7 sn · 19 eşleşme
```

Testler daha küçük (≈400 bin satır) dosyayla aynı yolları doğrular.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurocore.agent import bigfile


def _buyuk_dosya(satir: int = 400_000) -> Path:
    """Test için büyük dosya üretir (satır başına ~40 bayt → ~16 MB)."""
    yol = Path(tempfile.mkdtemp(prefix="nc-big-")) / "veri.log"
    with yol.open("w", encoding="utf-8", newline="\n") as f:
        for i in range(satir):
            if i % 50_000 == 0 and i:
                f.write(f"{i}| ISARET bulundu\n")
            else:
                f.write(f"{i}| kayit {i} veri veri\n")
    return yol


class TestEsik(unittest.TestCase):

    def test_buyuk_mu(self) -> None:
        yol = _buyuk_dosya(200_000)
        self.assertTrue(bigfile.buyuk_mu(yol))
        kucuk = yol.with_name("kucuk.txt")
        kucuk.write_text("a\nb\n", encoding="utf-8")
        self.assertFalse(bigfile.buyuk_mu(kucuk))


class TestIndeks(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.yol = _buyuk_dosya()

    def test_satir_sayisi(self) -> None:
        idx = bigfile.indeks_al(self.yol)
        self.assertEqual(idx["total_lines"], 400_000)
        self.assertEqual(idx["offsets"][0], 0)
        self.assertGreater(len(idx["offsets"]), 100)

    def test_yan_dosya_yazilir(self) -> None:
        bigfile.indeks_al(self.yol)
        self.assertTrue(bigfile.indeks_yolu(self.yol).exists())

    def test_bayat_indeks_yenilenir(self) -> None:
        # ⚠ Sınıf dosyasını DEĞİŞTİRME: diğer testlerin satır sayısını bozar.
        import shutil

        kopya = Path(tempfile.mkdtemp(prefix="nc-idx-")) / "kopya.log"
        shutil.copyfile(self.yol, kopya)
        onceki = bigfile.indeks_al(kopya)["total_lines"]
        with kopya.open("a", encoding="utf-8") as f:
            f.write("yeni satir\n")
        yeni = bigfile.indeks_al(kopya)
        self.assertEqual(yeni["total_lines"], onceki + 1)


class TestPencere(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.yol = _buyuk_dosya()

    def test_sona_yakin(self) -> None:
        p = bigfile.pencere(self.yol, bas=399_950, adet=20)
        self.assertEqual(p["start_line"], 399_950)
        self.assertEqual(p["end_line"], 399_969)
        self.assertTrue(p["indexed"])
        self.assertIn("399950", p["content"].splitlines()[0])

    def test_bastan(self) -> None:
        p = bigfile.pencere(self.yol, bas=1, adet=5)
        self.assertEqual(p["start_line"], 1)
        self.assertEqual(len(p["content"].splitlines()), 5)

    def test_adet_siniri(self) -> None:
        p = bigfile.pencere(self.yol, bas=1, adet=99_999)
        self.assertLessEqual(len(p["content"].splitlines()),
                             bigfile.PENCERE_MAKS)

    def test_dosya_sonu(self) -> None:
        p = bigfile.pencere(self.yol, bas=399_995, adet=100)
        self.assertEqual(p["end_line"], 400_000)
        self.assertFalse(p["truncated"])

    def test_bellek_sinirli(self) -> None:
        """Pencere okuma tüm dosyayı belleğe almamalı."""
        import tracemalloc

        tracemalloc.start()
        bigfile.pencere(self.yol, bas=399_000, adet=100)
        _, tepe = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(tepe, 8 * 1024 * 1024, f"tepe {tepe/1024/1024:.1f} MB")


class TestHarita(unittest.TestCase):

    def test_semboller(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-map-"))
        kod = kok / "modul.py"
        govde = "\n".join([
            "import os", "", "class Bir:", "    def x(self):", "        return 1",
            "", "def fonksiyon(a):", "    return a * 2", "",
            "async def uzak():", "    pass",
        ] + [f"# dolgu {i}" for i in range(5000)])
        kod.write_text(govde, encoding="utf-8")
        h = bigfile.harita(kod)
        metinler = " ".join(e["text"] for e in h["entries"])
        self.assertIn("class Bir", metinler)
        self.assertIn("def fonksiyon", metinler)
        self.assertIn("async def uzak", metinler)
        self.assertEqual(h["language"], "python")

    def test_markdown_basliklari(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-map2-"))
        md = kok / "not.md"
        md.write_text("# Baslik\n\ntext\n\n## Alt\n", encoding="utf-8")
        h = bigfile.harita(md)
        self.assertEqual(len(h["entries"]), 2)
        self.assertEqual(h["entries"][0]["line"], 1)


class TestArama(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.yol = _buyuk_dosya()

    def test_desen_bulunur(self) -> None:
        a = bigfile.ara(self.yol, "ISARET", sinir=10)
        self.assertGreaterEqual(a["count"], 5)
        self.assertTrue(all("ISARET" in h["text"] for h in a["hits"]))

    def test_sinir(self) -> None:
        a = bigfile.ara(self.yol, "kayit", sinir=7)
        self.assertEqual(a["count"], 7)
        self.assertTrue(a["truncated"])

    def test_gecersiz_regex(self) -> None:
        a = bigfile.ara(self.yol, "([", regex=True)
        self.assertIn("error", a)


class TestStrateji(unittest.TestCase):

    def test_buyuk_dosya_onerisi(self) -> None:
        yol = _buyuk_dosya(120_000)
        s = bigfile.strateji(yol)
        self.assertTrue(s["buyuk"])
        self.assertIn("outline", s["oneri"])
        self.assertGreater(s["total_lines"], 1000)

    def test_olmayan_dosya(self) -> None:
        s = bigfile.strateji(Path(tempfile.gettempdir()) / "yok-boyle-dosya")
        self.assertFalse(s["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
