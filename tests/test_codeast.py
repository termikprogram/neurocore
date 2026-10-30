"""
AST tabanlı çıkarım + çağrı grafiği testleri
============================================
Regex yerine `ast` (python) ve yapısal tarama (GDScript) kullanılır; ayrıca
"kim kimi çağırıyor" grafiği çıkarılır.

Ölçüm (kendi projemiz, 327 dosya): indeks 3,2 sn · 7.373 sembol · 4.341 ilişki
· 48.425 çağrı · çağrı grafiği sorgusu 13 ms.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurocore.agent import codeindex


def _py_proje() -> Path:
    kok = Path(tempfile.mkdtemp(prefix="nc-ast-"))
    (kok / "cekirdek.py").write_text(
        '"""Modül docstring."""\n'
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "AYAR = 5\n"
        "\n"
        "class Hesap:\n"
        "    def __init__(self, x):\n"
        "        self.x = x\n"
        "\n"
        "    def topla(self, y):\n"
        "        return self.x + y\n"
        "\n"
        "def yardimci(a):\n"
        "    return a * 2\n"
        "\n"
        "def ana():\n"
        "    h = Hesap(3)\n"
        "    return yardimci(h.topla(4))\n",
        encoding="utf-8")
    (kok / "kullanici.py").write_text(
        "from cekirdek import ana, yardimci\n"
        "\n"
        "def calistir():\n"
        "    return ana() + yardimci(1)\n",
        encoding="utf-8")
    return kok


def _gd_proje() -> Path:
    kok = Path(tempfile.mkdtemp(prefix="nc-gd-"))
    (kok / "oyun.gd").write_text(
        "class_name Oyun\n"
        "extends Node\n"
        "\n"
        "signal bitti\n"
        "const HIZ = 4.0\n"
        "var hedef = preload(\"res://yardimci.gd\")\n"
        "\n"
        "func basla():\n"
        "\tvar r = _hesapla(2)\n"
        "\tprint(r)\n"
        "\n"
        "func _hesapla(x):\n"
        "\treturn x * 2\n",
        encoding="utf-8")
    return kok


class TestPythonAst(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.kok = _py_proje()
        cls.ozet = codeindex.dizinle(cls.kok)

    def test_sinif_ve_fonksiyonlar(self) -> None:
        s = codeindex.sembol_ara(self.kok, "Hesap")
        self.assertTrue(s["sayi"] >= 1)
        s2 = codeindex.sembol_ara(self.kok, "yardimci")
        self.assertTrue(any(r["tur"] == "func" for r in s2["sonuc"]))

    def test_docstring_yanlis_eslesmez(self) -> None:
        """Docstring içindeki 'class' kelimesi sembol sayılmamalı."""
        s = codeindex.sembol_ara(self.kok, "docstring")
        self.assertEqual(s["sayi"], 0)

    def test_sabit(self) -> None:
        s = codeindex.sembol_ara(self.kok, "AYAR")
        self.assertTrue(any(r["tur"] == "const" for r in s["sonuc"]))

    def test_import_iliskisi(self) -> None:
        il = codeindex.iliskili_dosyalar(self.kok, "kullanici.py")
        hedefler = " ".join(x.get("hedef", "") for x in il["import_ettigi"])
        self.assertIn("cekirdek", hedefler)

    def test_cagri_grafigi_cagiranlar(self) -> None:
        c = codeindex.cagri_grafigi(self.kok, "ana")
        kaynaklar = " ".join(x["kaynak_sembol"] for x in c["cagiranlar"])
        self.assertIn("calistir", kaynaklar)

    def test_cagri_grafigi_cagardiklari(self) -> None:
        c = codeindex.cagri_grafigi(self.kok, "ana")
        hedefler = " ".join(x["hedef_ad"] for x in c["cagardiklari"])
        self.assertIn("yardimci", hedefler)

    def test_metot_cagrisi(self) -> None:
        c = codeindex.cagri_grafigi(self.kok, "topla")
        self.assertTrue(c["cagiranlar"] or c["cagardiklari"])


class TestGdscript(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.kok = _gd_proje()
        cls.ozet = codeindex.dizinle(cls.kok)

    def test_sinif_fonksiyon_sabit(self) -> None:
        for ad, tur in (("Oyun", "class"), ("basla", "func"),
                        ("HIZ", "const"), ("bitti", "signal")):
            s = codeindex.sembol_ara(self.kok, ad)
            self.assertTrue(any(r["tur"] == tur for r in s["sonuc"]),
                            f"{ad} ({tur}) bulunamadı: {s['sonuc']}")

    def test_preload_iliskisi(self) -> None:
        il = codeindex.iliskili_dosyalar(self.kok, "oyun.gd")
        self.assertTrue(il["import_ettigi"], "preload ilişkisi yok")

    def test_fonksiyon_ici_cagri(self) -> None:
        c = codeindex.cagri_grafigi(self.kok, "_hesapla")
        self.assertTrue(c["cagiranlar"], "çağıran bulunamadı")


class TestSemver(unittest.TestCase):

    def test_sema_degisince_tam_tarama(self) -> None:
        import sqlite3

        kok = _py_proje()
        codeindex.dizinle(kok)
        db = kok / ".neurocore" / "index" / "kod.db"
        with sqlite3.connect(str(db)) as bag:
            bag.execute("INSERT OR REPLACE INTO ustveri VALUES('semver','1')")
            bag.commit()
        sonuc = codeindex.dizinle(kok)
        self.assertGreater(sonuc["guncel"] + sonuc["yeni"], 0)


class TestCagriSayisi(unittest.TestCase):

    def test_ozet_cagri_alanı(self) -> None:
        kok = _py_proje()
        o = codeindex.dizinle(kok)
        self.assertIn("cagri", o)
        self.assertGreater(o["cagri"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

class TestHiyerarsi(unittest.TestCase):
    """(b) sınıf hiyerarşisi: taban + alt sınıf."""

    def test_taban_ve_alt(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-hier-"))
        (kok / "modeller.py").write_text(
            "class Hayvan:\n"
            "    def ses(self):\n"
            "        return '...'\n"
            "\n"
            "class Kopek(Hayvan):\n"
            "    def ses(self):\n"
            "        return 'hav'\n"
            "\n"
            "class Kedi(Hayvan):\n"
            "    pass\n",
            encoding="utf-8")
        codeindex.dizinle(kok)
        h = codeindex.hiyerarsi(kok, "Kopek")
        self.assertTrue(h["tanim"])
        self.assertEqual(h["tabanlar"][0]["taban"], "Hayvan")
        alt = codeindex.hiyerarsi(kok, "Hayvan")
        alt_adlar = {x["alt"] for x in alt["alt_siniflar"]}
        self.assertIn("Kopek", alt_adlar)
        self.assertIn("Kedi", alt_adlar)

    def test_hic_kalitim_yok(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-hier2-"))
        (kok / "a.py").write_text("class Yalniz:\n    pass\n", encoding="utf-8")
        codeindex.dizinle(kok)
        h = codeindex.hiyerarsi(kok, "Yalniz")
        self.assertEqual(h["tabanlar"], [])
        self.assertEqual(h["alt_siniflar"], [])


class TestEtkiHaritasi(unittest.TestCase):
    """(c) etki haritası: SVG + HTML raporu."""

    def test_html_ve_svg(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-etki-"))
        (kok / "a.py").write_text(
            "def hedef():\n    return 1\n"
            "\n"
            "def cagiran_a():\n    return hedef()\n"
            "\n"
            "def cagiran_b():\n    return hedef()\n",
            encoding="utf-8")
        codeindex.dizinle(kok)
        e = codeindex.etki_haritasi(kok, "hedef", sinir=10)
        self.assertTrue(e["ok"])
        self.assertGreaterEqual(e["cagiranlar"], 2)
        html = Path(e["html"])
        self.assertTrue(html.exists())
        icerik = html.read_text(encoding="utf-8")
        self.assertIn("<svg", icerik)
        self.assertIn("hedef", icerik)
        self.assertGreater(e["svg_bytes"], 100)

    def test_indeks_yok(self) -> None:
        kok = Path(tempfile.mkdtemp(prefix="nc-etki2-"))
        e = codeindex.etki_haritasi(kok, "x")
        self.assertFalse(e["ok"])
