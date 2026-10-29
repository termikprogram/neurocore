"""
Puzzle testleri — iç içe fazlar, walkthrough, revizyon
======================================================
Ağ yok: planlayıcıya sahte router verilir.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from neurocore.agent.puzzle import (
    AKTIF,
    BEKLIYOR,
    MAX_DERINLIK,
    MAX_SORU,
    REVIZE,
    TAMAM,
    Parca,
    Puzzle,
    PuzzlePlanner,
)


class _Yanit:
    def __init__(self, text: str, ok: bool = True) -> None:
        self.text = text
        self.ok = ok
        self.error = "" if ok else "fake hata"


class _Router:
    def __init__(self, text: str = "{}", ok: bool = True,
                 patlar: bool = False) -> None:
        self.text = text
        self.ok = ok
        self.patlar = patlar
        self.cagri = 0

    def chat(self, messages: Any, **kwargs: Any) -> _Yanit:
        self.cagri += 1
        if self.patlar:
            raise RuntimeError("ağ koptu")
        return _Yanit(self.text, ok=self.ok)


def _plan_json() -> str:
    return json.dumps({
        "hedef": "ağaç sahnesini geliştir",
        "parcalar": [
            {"baslik": "Faz 1 — keşif", "olcut": "sahne raporu",
             "walkthrough": ["kaç tür var?", "poligon bütçesi?"],
             "alt": [{"baslik": "1.1 katalog", "olcut": "liste"},
                     {"baslik": "1.2 ölçüm", "olcut": "boyutlar"}]},
            {"baslik": "Faz 2 — üretim", "olcut": "yeni .blend",
             "walkthrough": ["kaç yeni tür?"]},
            {"baslik": "Faz 3 — render", "olcut": "12 PNG"},
        ]}, ensure_ascii=False)


class TestParca(unittest.TestCase):

    def test_bul_ve_duz(self) -> None:
        kok = Parca(id="p0", baslik="kök",
                    cocuklar=[Parca(id="p1", baslik="a",
                                    cocuklar=[Parca(id="p1.1", baslik="b")])])
        self.assertEqual(len(kok.duz()), 3)
        self.assertIsNotNone(kok.bul("p1.1"))
        self.assertIsNone(kok.bul("yok"))

    def test_yaprak(self) -> None:
        kok = Parca(id="p0", baslik="kök",
                    cocuklar=[Parca(id="p1", baslik="a", durum=TAMAM),
                              Parca(id="p2", baslik="b")])
        yaprak = kok.yaprak()
        self.assertEqual(yaprak.id, "p2")

    def test_serilestirme(self) -> None:
        kok = Parca(id="p0", baslik="kök",
                    cocuklar=[Parca(id="p1", baslik="a", sorular=["s"])])
        geri = Parca.from_dict(kok.to_dict())
        self.assertEqual(geri.baslik, "kök")
        self.assertEqual(geri.cocuklar[0].sorular, ["s"])


class TestPuzzlePlanner(unittest.TestCase):

    def test_agac_kurulur(self) -> None:
        p = PuzzlePlanner(_Router(_plan_json())).planla("ağaçları geliştir")
        self.assertEqual(p.sayi(), 6)               # p0 + 2 faz + 2 alt + 1
        self.assertEqual(p.tamamlanan(), 0)
        self.assertEqual(len([x for x in p.parcalar() if x.sorular]), 2)

    def test_ilk_yaprak_aktif(self) -> None:
        p = PuzzlePlanner(_Router(_plan_json())).planla("g")
        self.assertEqual(p.aktif_id, "p1.1")
        self.assertEqual(p.aktif_yaprak().durum, AKTIF)

    def test_ilerleme_ust_fazi_kapatir(self) -> None:
        p = PuzzlePlanner(_Router(_plan_json())).planla("g")
        p.ilerle()                                   # p1.1
        p.ilerle()                                   # p1.2 → faz 1 kapanır
        faz1 = p.bul("p1")
        self.assertEqual(faz1.durum, TAMAM,
                         "tüm alt parçalar bitince faz tamam olmalı")
        self.assertEqual(p.aktif_yaprak().id, "p2")

    def test_tamamlanma(self) -> None:
        p = PuzzlePlanner(_Router(_plan_json())).planla("g")
        for _ in range(len([x for x in p.parcalar() if not x.cocuklar])):
            p.ilerle()
        self.assertTrue(p.tamamlandi())
        self.assertEqual(p.tamamlanan(), p.sayi())

    def test_revizyon(self) -> None:
        p = PuzzlePlanner(_Router(_plan_json())).planla("g")
        parca = p.revize("p2", "önce mevcutları düzelt")
        self.assertEqual(parca.durum, REVIZE)
        self.assertEqual(parca.not_, "önce mevcutları düzelt")
        self.assertEqual(p.aktif_id, "p2")
        self.assertEqual(p.revizyon, 1)
        self.assertIn("kullanıcı notu", p.ozet())

    def test_cevapla(self) -> None:
        p = PuzzlePlanner(_Router(_plan_json())).planla("g")
        parca = p.cevapla("p1", "3 tür var")
        self.assertEqual(parca.cevap, "3 tür var")
        self.assertEqual(parca.durum, AKTIF)

    def test_llm_yoksa_duz_plan(self) -> None:
        p = PuzzlePlanner(None).planla("basit görev")
        self.assertGreaterEqual(p.sayi(), 3)
        self.assertIsNotNone(p.aktif_yaprak())

    def test_router_patlarsa(self) -> None:
        p = PuzzlePlanner(_Router(patlar=True)).planla("g")
        self.assertGreaterEqual(p.sayi(), 3)         # düz plana düştü

    def test_bozuk_json(self) -> None:
        p = PuzzlePlanner(_Router("plan yok")).planla("g")
        self.assertGreaterEqual(p.sayi(), 3)

    def test_yanit_basarisiz(self) -> None:
        p = PuzzlePlanner(_Router("{}", ok=False)).planla("g")
        self.assertGreaterEqual(p.sayi(), 3)

    def test_derinlik_siniri(self) -> None:
        derin: Any = {"baslik": "yaprak"}
        for i in range(MAX_DERINLIK + 3):
            derin = {"baslik": f"d{i}", "alt": [derin]}
        metin = json.dumps({"hedef": "x", "parcalar": [derin]})
        p = PuzzlePlanner(_Router(metin)).planla("x")
        self.assertLessEqual(max(x.derinlik for x in p.parcalar()),
                             MAX_DERINLIK)

    def test_soru_siniri(self) -> None:
        metin = json.dumps({"hedef": "x", "parcalar": [
            {"baslik": "f", "walkthrough": [f"s{i}" for i in range(12)]}]})
        p = PuzzlePlanner(_Router(metin)).planla("x")
        sorular = [x for x in p.parcalar() if x.sorular][0].sorular
        self.assertLessEqual(len(sorular), MAX_SORU)

    def test_ozet_metni(self) -> None:
        p = PuzzlePlanner(_Router(_plan_json())).planla("ağaçları geliştir")
        ozet = p.ozet()
        self.assertIn("PUZZLE", ozet)
        self.assertIn("Faz 1", ozet)
        self.assertLessEqual(len(ozet), 1500)


class TestPuzzleKalici(unittest.TestCase):

    def test_kaydet_yukle(self) -> None:
        p = PuzzlePlanner(_Router(_plan_json())).planla("g")
        p.revize("p2", "not")
        yol = Path(tempfile.mkdtemp(prefix="nc-pz-")) / "p.json"
        self.assertTrue(p.kaydet(yol))
        geri = Puzzle.yukle(yol)
        self.assertIsNotNone(geri)
        self.assertEqual(geri.sayi(), p.sayi())
        self.assertEqual(geri.revizyon, 1)
        self.assertEqual(geri.aktif_id, "p2")

    def test_bozuk_dosya(self) -> None:
        yol = Path(tempfile.mkdtemp(prefix="nc-pz2-")) / "bozuk.json"
        yol.write_text("{bozuk", encoding="utf-8")
        self.assertIsNone(Puzzle.yukle(yol))


if __name__ == "__main__":
    unittest.main(verbosity=2)
