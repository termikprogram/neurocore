"""
Araç sonucu serileştirme testleri — "ajan dosya içeriğini görebiliyor mu?"
==========================================================================
ÖLÇÜLEN HATA (2026-10-30): `ToolResult.to_dict()` çıktı 12.000 karakteri
aşınca `data`yı **JSON string**'e çeviriyordu. Ajan döngüsünün araç-mesajı
biçimlendiricisi sözlük beklediği için `code.read_file` sonucu yalnızca
meta veri (yol, satır sayısı) olarak görünüyor, **dosya içeriği kayboluyordu**.
Ajan da "araç çıktısında içerik yok, teşhis yapamam" diyerek işi bırakıyordu.

Bu testler yapının korunduğunu garanti eder.
"""

from __future__ import annotations

import json
import unittest

from neurocore.agent.tools import ToolResult, _kirp_veri


class TestYapiKorunur(unittest.TestCase):
    """Büyük çıktı kısaltılırken sözlük yapısı bozulmamalı."""

    def _buyuk(self, satir: int = 900) -> ToolResult:
        icerik = "\n".join(f"satir {i} değeri" for i in range(satir))
        return ToolResult(
            ok=True, tool="code.read_file",
            data={"path": "main.gd", "language": "gdscript",
                  "total_lines": satir, "content": icerik})

    def test_data_sozluk_kalir(self) -> None:
        d = self._buyuk().to_dict(max_chars=6000)
        self.assertIsInstance(d["data"], dict,
                              "data string'e dönüşmemeli (içerik kaybı)")
        self.assertIn("content", d["data"])
        self.assertIn("total_lines", d["data"])

    def test_icerik_korunur(self) -> None:
        d = self._buyuk().to_dict(max_chars=6000)
        icerik = d["data"]["content"]
        self.assertTrue(icerik.startswith("satir 0"),
                        "içerik baştan başlamalı (satırlar kaybolmamalı)")
        self.assertGreater(len(icerik), 1000)
        self.assertIn("karakter", icerik.splitlines()[-1])  # kırpma notu

    def test_kucuk_cikti_degismez(self) -> None:
        r = ToolResult(ok=True, tool="code.read_file",
                       data={"path": "a.py", "content": "x = 1\n"})
        d = r.to_dict()
        self.assertEqual(d["data"]["content"], "x = 1\n")
        self.assertFalse(d.get("truncated"))

    def test_truncated_bayragi(self) -> None:
        d = self._buyuk(2000).to_dict(max_chars=4000)
        self.assertTrue(d.get("truncated"))
        self.assertIn("korundu", d.get("note", ""))

    def test_stdout_korunur(self) -> None:
        """Komut çıktıları da aynı yoldan geçer."""
        cikti = "A" * 50000
        r = ToolResult(ok=True, tool="shell.run",
                       data={"command": "pytest", "returncode": 0,
                             "stdout": cikti, "stderr": ""})
        d = r.to_dict(max_chars=8000)
        self.assertIsInstance(d["data"], dict)
        self.assertIn("stdout", d["data"])
        self.assertTrue(d["data"]["stdout"].startswith("AAAA"))

    def test_json_string_gelirse_cozulur(self) -> None:
        """Savunma: eski üretici string verirse içerik yine bulunmalı."""
        import neurocore.agent.loop as dongu

        ham = json.dumps({"path": "m.gd", "content": "kod", "total_lines": 1})
        # Döngü biçimlendiricisi string'i çözmeli (json.loads)
        self.assertEqual(json.loads(ham)["content"], "kod")


class TestKirpVeri(unittest.TestCase):
    """Yardımcı fonksiyonun kendi davranışı."""

    def test_metin_kirpilir(self) -> None:
        self.assertIn("karakter", _kirp_veri("x" * 100, 10))

    def test_sozluk_yapisi(self) -> None:
        veri = {"a": "x" * 50, "b": 3, "c": {"d": "y" * 50}}
        k = _kirp_veri(veri, 20)
        self.assertIsInstance(k, dict)
        self.assertEqual(k["b"], 3)
        self.assertIsInstance(k["c"], dict)
        self.assertLess(len(k["a"]), 50)

    def test_liste_kirpilir(self) -> None:
        k = _kirp_veri(["z" * 100] * 3, 10)
        self.assertIsInstance(k, list)
        self.assertEqual(len(k), 3)

    def test_skaler_aynen(self) -> None:
        self.assertEqual(_kirp_veri(42, 10), 42)
        self.assertEqual(_kirp_veri(None, 10), None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
