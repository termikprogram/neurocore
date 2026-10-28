"""
Güvenlik Filtresi Testleri
============================
`tools/guvenlik_filtresi.py` — myAi'nin **ölçülmüş gücü** için test.

⚠ Ölçülen gerçek (tur 14): myAi **saçma seçenekleri eliyor** (4/7, hepsi
absürt yanlışlı) ama **ince ayrım yapamıyor** (~%50). Filtre bu güce
uygun tasarlandı: absürt değerleri eler, kararsızsa **varsayılana** düşer.
"""

from __future__ import annotations

import json
import logging
import sys
import unittest
from pathlib import Path

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)


class TestAralikKontrolu(unittest.TestCase):
    """Aralık kontrolü — myAi'siz, her zaman güvenilir."""

    def test_normal_deger_gecer(self) -> None:
        from tools.guvenlik_filtresi import aralik_kontrol

        for p, d in (("labirent_boyut", 21), ("karakter_hizi", 5.0),
                     ("agac_sayisi", 40), ("magara_tunel", 10)):  # ⚠ varsayılan 10 (Kararlar ile hizalı)
            ok, neden = aralik_kontrol(p, d)
            self.assertTrue(ok, f"{p}={d} reddedildi: {neden}")

    def test_sacma_deger_reddedilir(self) -> None:
        from tools.guvenlik_filtresi import aralik_kontrol

        vakalar = [
            ("labirent_boyut", 2),      # çok küçük
            ("labirent_boyut", 501),    # çok büyük
            ("karakter_hizi", 0.01),    # neredeyse duruyor
            ("karakter_hizi", 500),     # ışınlanma
            ("agac_sayisi", 0),         # hiç ağaç yok
            ("agac_sayisi", 100000),    # performans çöker
            ("magara_tunel", 1),        # tünel yok
            ("magara_tunel", 200),      # sonsuz tünel
        ]
        for p, d in vakalar:
            ok, neden = aralik_kontrol(p, d)
            self.assertFalse(ok, f"{p}={d} kabul edildi (saçma olmalı)")

    def test_sayisal_olmayan_reddedilir(self) -> None:
        from tools.guvenlik_filtresi import aralik_kontrol

        for d in ("abc", None, [1], {}):
            ok, _ = aralik_kontrol("agac_sayisi", d)
            self.assertFalse(ok, f"{d!r} kabul edildi")


class TestFiltre(unittest.TestCase):
    """Filtre mantığı — myAi olmadan da test edilebilir."""

    def test_tek_guvenli_aday(self) -> None:
        """2 saçma + 1 makul → makul seçilir (myAi'siz)."""
        from tools.guvenlik_filtresi import GuvenlikFiltresi

        g = GuvenlikFiltresi()          # model yüklenmez
        s = g.filtrele("labirent_boyut", [21, 2, 501])
        self.assertEqual(s.deger, 21)
        self.assertFalse(s.myAi_kullanildi)
        self.assertIn("aralık", s.neden)

    def test_hepsi_sacma_varsayilana_duser(self) -> None:
        """Tüm adaylar saçma → varsayılan kullanılır ve güvensiz işaretlenir."""
        from tools.guvenlik_filtresi import GuvenlikFiltresi

        g = GuvenlikFiltresi()
        s = g.filtrele("agac_sayisi", [0, 100000, -5])
        self.assertEqual(s.deger, 40)   # varsayılan
        self.assertFalse(s.guvenli)
        self.assertIn("aralık dışı", s.neden)

    def test_kararsizsa_varsayilan(self) -> None:
        """
        ⚠ Ölçülen gerçek: makul adaylar arasında myAi **rastgele**
        seçiyor (%50). Skorlar yakınsa varsayılan güvenlidir.
        """
        from tools.guvenlik_filtresi import GuvenlikFiltresi, PARAMETRELER

        g = GuvenlikFiltresi()
        # myAi olmadan: kararsızlık yolu → varsayılan
        s = g.filtrele("agac_sayisi", [20, 40, 70])
        self.assertIn(s.deger, (None, 40),
                      f"beklenen varsayılan, gelen {s.deger}")
        if s.deger is None:
            self.assertIn("myAi", s.neden)

    def test_parametre_tablosu_tutarli(self) -> None:
        """Her parametrenin varsayılanı aralık içinde olmalı."""
        from tools.guvenlik_filtresi import PARAMETRELER, aralik_kontrol

        for p, ayar in PARAMETRELER.items():
            v = ayar["varsayilan"]
            ok, neden = aralik_kontrol(p, v)
            self.assertTrue(ok, f"{p} varsayılanı aralık dışı: {neden}")
            alt, ust = ayar["aralik"]
            self.assertLess(alt, ust, f"{p} aralığı ters")
            self.assertTrue(ayar.get("soru"), f"{p} sorusu yok")


class TestAltSurec(unittest.TestCase):
    """Alt süreç arayüzü (VRAM çakışmasını önler)."""

    def test_alt_surec_tek_aday(self) -> None:
        """Tek güvenli aday varsa model yüklenmez (hızlı yol)."""
        from tools.guvenlik_filtresi import filtrele_alt_surec

        r = filtrele_alt_surec("karakter_hizi", [5.0, 0.01, 500])
        self.assertEqual(r.get("deger"), 5.0)
        self.assertIn("tek güvenli", r.get("neden", ""))

    def test_alt_surec_hepsi_sacma(self) -> None:
        from tools.guvenlik_filtresi import filtrele_alt_surec

        r = filtrele_alt_surec("agac_sayisi", [0, 100000])
        self.assertEqual(r.get("deger"), 40)
        self.assertFalse(r.get("guvenli", True))


class TestUretimEntegrasyonu(unittest.TestCase):
    """Filtre üretim parametreleriyle uyumlu mu?"""

    def test_kararlar_alanlari_kapsanmis(self) -> None:
        """`Kararlar` alanları filtre tablosunda olmalı."""
        from tools.godot_oyun_uret import Kararlar
        from tools.guvenlik_filtresi import PARAMETRELER

        k = Kararlar()
        for alan in ("labirent_boyut", "karakter_hizi", "agac_sayisi",
                     "kaya_sayisi", "arena_yaricap", "magara_tunel",
                     "yukseklik"):
            self.assertIn(alan, PARAMETRELER,
                          f"{alan} filtre tablosunda yok")
            self.assertTrue(hasattr(k, alan),
                            f"{alan} Kararlar'da yok")
            # Varsayılan değerler eşleşmeli (tutarlılık)
            self.assertEqual(
                getattr(k, alan), PARAMETRELER[alan]["varsayilan"],
                f"{alan} varsayılanı uyuşmuyor: "
                f"Kararlar={getattr(k, alan)} "
                f"filtre={PARAMETRELER[alan]['varsayilan']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
