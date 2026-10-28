"""
Derleme Onarım Normalleştirici Testleri
=========================================
⚠ Neden gerekli?
----------------
`tools/derleme_onarim.py` içindeki `normalestir()` fonksiyonu **6
sözdizimsel kalıbı** düzeltir. Bu kalıplar **40 tur boyunca ölçülerek**
bulundu ve her biri bir gerçek Godot hata mesajına karşılık gelir.

⚠ Ölçülen risk: tur 38'de "daha doğru" sanılan bir değişiklik
**geriletti** (3/4 → 2/4). Bu testler o kazancı **korur**.

Kalıplar (hepsi ölçülmüş):
    1. Sarkan nokta + girintili devam
       → `Expected identifier after "." for attribute access.`
    2. `f` son eki (`100.0f`)
       → `Invalid numeric notation.`
    3. Girintili nokta devamı (`\\n        .start()`)
       → `Expected statement, found "Indent" instead.`
    4. Değersiz atama (`m.clearcoat =`)
       → `Expected an expression after "=".`
    5. Tekrarlanan dosya (`extends` iki kez)
       → `Identifier "..." not declared`
    6. Yarım `func` başlığı
       → `Expected end of statement after expression`
"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)


class TestNormalestirici(unittest.TestCase):
    """`normalestir()` sözdizimsel düzeltmeleri."""

    @classmethod
    def setUpClass(cls) -> None:
        from tools.derleme_onarim import normalestir

        cls.normalestir = staticmethod(normalestir)

    # ─── Kalıp 1: sarkan nokta ────────────────────────────────────────

    def test_sarkan_nokta_birlestirilir(self) -> None:
        """`t.` + satır sonu + girintili devam → tek satır."""
        kod = ("extends Node\n\nfunc _ready() -> void:\n"
               "    var t := create_tween()\n    t.\n"
               "        tween_property(self, \"p\", Vector2(1, 0), 1.0)\n")
        d = self.normalestir(kod)
        self.assertNotIn("t.\n", d)
        self.assertIn("t.tween_property", d)

    def test_kontrol_temiz_kod_degismez(self) -> None:
        """Temiz kod **değişmemeli**."""
        kod = ("extends Node3D\n\nfunc _ready() -> void:\n"
               "    var n := FastNoiseLite.new()\n"
               "    n.frequency = 0.03\n")
        self.assertEqual(self.normalestir(kod).rstrip("\n"),
                         kod.rstrip("\n"))

    # ─── Kalıp 2: `f` son eki ─────────────────────────────────────────

    def test_f_son_eki_silinir(self) -> None:
        """Godot'ta `f` son eki yok (C/Python alışkanlığı)."""
        kod = ("extends Node\n\nfunc _ready() -> void:\n"
               "    var t := create_tween()\n"
               "    t.tween_property(self, \"p\", Vector2(1, 0), 1.0f)\n")
        d = self.normalestir(kod)
        self.assertNotIn("1.0f", d)
        self.assertIn("1.0", d)

    # ─── Kalıp 3: girintili nokta ─────────────────────────────────────

    def test_girintili_nokta_birlestirilir(self) -> None:
        """`\\n        .start()` → `.start()`."""
        kod = ("extends Node\n\nfunc _ready() -> void:\n"
               "    var t := create_tween()\n"
               "    t.tween_property(self, \"p\", Vector2(1, 0), 1.0)\n"
               "        .start()\n")
        d = self.normalestir(kod)
        self.assertNotIn("\n        .start()", d)
        self.assertIn(".start()", d)

    # ─── Kalıp 4: değersiz atama ──────────────────────────────────────

    def test_degersiz_atama_atilir(self) -> None:
        """`m.clearcoat =` satırı atılmalı."""
        kod = ("extends MeshInstance3D\n\nfunc _ready() -> void:\n"
               "    var m := StandardMaterial3D.new()\n"
               "    m.roughness = 0.5\n    m.clearcoat =\n")
        d = self.normalestir(kod)
        self.assertNotIn("m.clearcoat", d)
        self.assertIn("m.roughness = 0.5", d)

    def test_dolu_atama_korunur(self) -> None:
        """Değeri olan atama **korunmalı**."""
        kod = ("extends MeshInstance3D\n\nfunc _ready() -> void:\n"
               "    var m := StandardMaterial3D.new()\n"
               "    m.clearcoat = 0.5\n")
        self.assertIn("m.clearcoat = 0.5", self.normalestir(kod))

    # ─── Kalıp 5: dosya tekrarı ───────────────────────────────────────

    def test_dosya_tekrari_kesilir(self) -> None:
        """İkinci `extends`'ten sonrası atılmalı."""
        kod = ("extends Node\n\nfunc _ready() -> void:\n"
               "    var z := Timer.new()\n\n"
               "func _x() -> void:\n    pass\n\n"
               "extends Node\n\nfunc _ready() -> void:\n    pass\n")
        d = self.normalestir(kod)
        self.assertEqual(d.count("extends Node"), 1)

    # ─── Kalıp 6: yarım func ──────────────────────────────────────────

    def test_yarim_func_tamamlanir(self) -> None:
        """Gövdesiz son `func`'a `pass` eklenmeli."""
        kod = ("extends Node\n\nfunc _ready() -> void:\n    pass\n\n"
               "func _y()\n")
        d = self.normalestir(kod)
        self.assertNotIn("func _y()\n", d)
        self.assertIn("pass", d.split("func _y")[-1])

    # ─── Godot derleyicisi ile bütünleşik doğrulama ───────────────────

    def test_godot_derleyicisi_ile_dogrula(self) -> None:
        """Normalize edilmiş kod **gerçek Godot** ile derlenmeli."""
        from tools.derleme_onarim import GODOT, derle

        if not GODOT.exists():
            self.skipTest("Godot bulunamadı")
        kod = ("extends Node\n\nfunc _ready() -> void:\n"
               "    var t := create_tween()\n    t.\n"
               "        tween_property(self, \"position\", "
               "Vector2.RIGHT * 100.0f, 1.0f)\n"
               "        .start()\n")
        ok, hata = derle(self.normalestir(kod))
        self.assertTrue(ok, f"derlenmedi: {hata[:110]}")

    def test_degersiz_atama_godot_ile(self) -> None:
        """Değersiz atama temizlendikten sonra derlenmeli."""
        from tools.derleme_onarim import GODOT, derle

        if not GODOT.exists():
            self.skipTest("Godot bulunamadı")
        kod = ("extends MeshInstance3D\n\nfunc _ready() -> void:\n"
               "    var m := StandardMaterial3D.new()\n"
               "    m.roughness = 0.5\n    m.clearcoat =\n")
        ok, hata = derle(self.normalestir(kod))
        self.assertTrue(ok, f"derlenmedi: {hata[:110]}")

    # ─── Kalıp 7: kod çiti ────────────────────────────────────────────

    def test_kod_citi_satirlari_atilir(self) -> None:
        """
        ⚠ tur 41'de ölçüldü: model markdown çiti üretiyor ve satır içi
        temizlik yetersiz kalıyordu:
            `Parse Error: Expected statement, found "`" instead.`
        Çit **satırları** atılmalı (baştaki boşluk ne olursa olsun).
        """
        kod = ("extends Node\n\nfunc _ready() -> void:\n"
               "    ```gdscript\n    var z := Timer.new()\n    ```\n")
        d = self.normalestir(kod)
        self.assertNotIn("```", d)
        self.assertIn("var z := Timer.new()", d)

    def test_kod_citi_godot_ile(self) -> None:
        """Çit temizlendikten sonra kod derlenmeli."""
        from tools.derleme_onarim import GODOT, derle

        if not GODOT.exists():
            self.skipTest("Godot bulunamadı")
        kod = ("extends Node\n\nfunc _ready() -> void:\n"
               "    ```gdscript\n    var z := Timer.new()\n    ```\n")
        ok, hata = derle(self.normalestir(kod))
        self.assertTrue(ok, f"derlenmedi: {hata[:110]}")


class TestGovdeAyristirma(unittest.TestCase):
    """
    Gövde ayıklama (tur 41 dersi).

    ⚠ Ölçülen kritik hata: döngü, gövdeden `extends`/`func` satırlarını
    **atıyordu** → eksik fonksiyon tanımı (`_on_timeout`) kayboluyor ve
    `Identifier not declared` alınıyordu. Dosya tekrarı zaten
    `normalestir()` içinde çözülüyor; bu filtre **gereksiz ve zararlı**.
    """

    def test_govdedeki_func_tanimlari_korunur(self) -> None:
        """
        ⚠ `else` dalında gövdedeki `func` tanımını atan filtre
        **geri gelmemeli** (4/4'ü 3/4'e düşürür).

        Not: `elif s.eksik_tanim` dalındaki filtre **doğrudur** —
        orada başlık ayrıca ekleniyor, bu yüzden tekrarı önlemek
        için gerekli.
        """
        import inspect

        from tools.derleme_onarim import onar

        kaynak = inspect.getsource(onar)
        # ⚠ Zararlı kalıp: `g = _tam_koddan_govde(...)` sonrasında
        # gövdeyi `extends|func` ile süzmek.
        self.assertNotIn(
            'g = "\\n".join(x for x in g.splitlines()', kaynak,
            "gövdedeki func tanımlarını atan filtre geri gelmiş — "
            "bu 4/4'ü 3/4'e düşürür")
        # Filtre yalnız eksik-tanım dalında olmalı (1 kez)
        self.assertEqual(
            kaynak.count(r"^\s*(extends|func)\b"), 1,
            "gövde filtresi yalnız `elif s.eksik_tanim` dalında olmalı")

    def test_eksik_tanim_godot_ile(self) -> None:
        """`func` tanımı korununca Timer kodu derlenmeli."""
        from tools.derleme_onarim import GODOT, derle, normalestir

        if not GODOT.exists():
            self.skipTest("Godot bulunamadı")
        # Modelin gerçek çıktı deseni: dosya tekrarı + `func` tanımı
        kod = ("extends Node\n\nfunc _ready() -> void:\n"
               "    var z := Timer.new()\n"
               "    z.ontimeout = _on_timeout\n    z.start()\n\n"
               "func _on_timeout() -> void:\n    print(\"timeout\")\n\n"
               "extends Node\n\nfunc _ready() -> void:\n"
               "    var z := Timer.new()\n")
        d = normalestir(kod)
        self.assertEqual(d.count("extends"), 1)
        self.assertIn("func _on_timeout", d,
                      "eksik tanım filtrelenmiş")
        ok, hata = derle(d)
        self.assertTrue(ok, f"derlenmedi: {hata[:110]}")


class TestHataAyristirma(unittest.TestCase):
    """Godot hata çıktısı ayrıştırma (tur 33 dersi)."""

    @classmethod
    def setUpClass(cls) -> None:
        from tools.derleme_onarim import GODOT, derle, hata_satiri

        cls.derle = staticmethod(derle)
        cls.hata_satiri = staticmethod(hata_satiri)
        cls.godot_var = GODOT.exists()

    def test_satir_numarasi_cikarilir(self) -> None:
        """
        ⚠ tur 33'te ölçüldü: `derle()` yalnız **ilk** hata satırını
        döndürdüğü için satır numarası (`at: ... :4`) **kayboluyordu**
        ve satır-hedefli onarım 2 tur boyunca **etkisiz** kaldı.
        """
        if not self.godot_var:
            self.skipTest("Godot bulunamadı")
        kod = ("extends Node\n\nfunc _ready() -> void:\n"
               "    var x := \n")
        ok, hata = self.derle(kod)
        self.assertFalse(ok)
        self.assertIn("at:", hata, "satır no içeren `at:` satırı eksik")
        no, cevre = self.hata_satiri(hata, kod)
        self.assertEqual(no, 4)
        self.assertIn("var x", cevre)


if __name__ == "__main__":
    unittest.main(verbosity=2)
