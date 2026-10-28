"""
Bilişsel katman testleri (ağ çağrısı YOK)
==========================================
Kapsam:

```
1. HATA TAKSONOMİSİ : her hata türü doğru sınıflanır; kurtarma talimatı var
2. HEDEF DURUMU     : olgu/deneme ekleme, tekrar eleme, sınırlama, özet
3. YANSIMA          : JSON çözümü (düz / kod bloklu / bozuk), karar kümesi
4. DOĞRULAMA        : destekli/desteksiz, puan kırpma, eksik listesi
5. ROBUSTLUK        : router hata verirse katman **çökmez**
6. KANCA            : ajan döngüsü kanıt metni + düzeltme yolu
```
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from neurocore.agent.cognition import (
    CognitionLayer,
    Dogrulama,
    HataTuru,
    HedefDurumu,
    Yansima,
    _json_cikar,
    dogrula,
    duzelt,
    kurtarma_talimati,
    siniflandir,
    yansit,
)


# ─── Sahteler ───────────────────────────────────────────────────────────────

class _Yanit:
    def __init__(self, text: str, ok: bool = True, error: str = "") -> None:
        self.text = text
        self.ok = ok
        self.error = error


class _Router:
    """Sabit yanıt döndüren sahte router."""

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


# ─── 1) Hata taksonomisi ────────────────────────────────────────────────────

class TestHataTaksonomisi(unittest.TestCase):

    def test_turler(self) -> None:
        ornekler = {
            HataTuru.IZIN: "Permission denied: /etc/hosts",
            HataTuru.AG: "HTTP 429 hız sınırı",
            HataTuru.KAYNAK: "CUDA out of memory",
            HataTuru.BULUNAMADI: "FileNotFoundError: not found",
            HataTuru.SOZDIZIMI: "SyntaxError: invalid syntax",
            HataTuru.ARGUMAN: "TypeError: missing required argument",
        }
        for beklenen, metin in ornekler.items():
            self.assertEqual(siniflandir(metin), beklenen, metin)

    def test_bos_metin(self) -> None:
        self.assertEqual(siniflandir(""), HataTuru.YOK)
        self.assertEqual(siniflandir("   "), HataTuru.YOK)

    def test_bilinmeyen(self) -> None:
        self.assertEqual(siniflandir("qwerty asdf zxcv"), HataTuru.BILINMEYEN)

    def test_her_tur_icin_talimat_var(self) -> None:
        for tur in HataTuru:
            if tur == HataTuru.YOK:
                continue
            talimat = kurtarma_talimati(tur)
            self.assertGreater(len(talimat), 20, tur.value)

    def test_talimatlar_farkli(self) -> None:
        """⚠ Aynı talimatı her türe vermek taksonomiyi anlamsız kılar."""
        metinler = {kurtarma_talimati(t) for t in HataTuru
                    if t != HataTuru.YOK}
        self.assertGreaterEqual(len(metinler), 7)


# ─── 2) Hedef durumu ────────────────────────────────────────────────────────

class TestHedefDurumu(unittest.TestCase):

    def test_olgu_ve_deneme(self) -> None:
        h = HedefDurumu(hedef="kılıç üret")
        h.olgu_ekle("asset.uret: kilic.blend")
        h.deneme_ekle("asset.uret → ok")
        ozet = h.ozet()
        self.assertIn("kılıç üret", ozet)
        self.assertIn("kilic.blend", ozet)
        self.assertIn("asset.uret → ok", ozet)

    def test_tekrar_elenir(self) -> None:
        h = HedefDurumu(hedef="g")
        h.olgu_ekle("aynı bilgi")
        h.olgu_ekle("aynı bilgi")
        self.assertEqual(len(h.facts), 1)

    def test_sinir(self) -> None:
        h = HedefDurumu(hedef="g")
        for i in range(30):
            h.olgu_ekle(f"olgu {i}")
            h.deneme_ekle(f"deneme {i}")
        self.assertLessEqual(len(h.facts), h.MAX_FACT)
        self.assertLessEqual(len(h.attempts), h.MAX_ATTEMPT)
        # ⚠ En YENİ olgular korunmalı
        self.assertIn("olgu 29", h.ozet())

    def test_ozet_siniri(self) -> None:
        h = HedefDurumu(hedef="x" * 500)
        for i in range(12):
            h.olgu_ekle("y" * 200)
        self.assertLessEqual(len(h.ozet(sinir=300)), 300)


# ─── 3) Yansıma ─────────────────────────────────────────────────────────────

class TestYansima(unittest.TestCase):

    def test_duz_json(self) -> None:
        y = Yansima.coz('{"verdict": "yeniden_planla", '
                        '"missing": ["dosya yolu"], "next": "listele"}')
        self.assertTrue(y.ok)
        self.assertEqual(y.verdict, "yeniden_planla")
        self.assertEqual(y.missing, ["dosya yolu"])

    def test_kod_bloklu_json(self) -> None:
        y = Yansima.coz('İşte:\n```json\n{"verdict": "cevapla"}\n```')
        self.assertTrue(y.ok)
        self.assertEqual(y.verdict, "cevapla")

    def test_gecersiz_karar_devam(self) -> None:
        y = Yansima.coz('{"verdict": "saçmalık"}')
        self.assertEqual(y.verdict, "devam")

    def test_bozuk_json(self) -> None:
        y = Yansima.coz("json yok burada")
        self.assertFalse(y.ok)

    def test_router_ile(self) -> None:
        r = _Router('{"verdict": "devam", "missing": []}')
        y = yansit(r, "hedef", "durum")
        self.assertTrue(y.ok)
        self.assertEqual(r.cagri, 1)

    def test_router_patlarsa(self) -> None:
        y = yansit(_Router(patlar=True), "h", "d")
        self.assertFalse(y.ok)
        self.assertIn("RuntimeError", y.error)


# ─── 4) Doğrulama ───────────────────────────────────────────────────────────

class TestDogrulama(unittest.TestCase):

    def test_destekli(self) -> None:
        d = Dogrulama.coz('{"supported": true, "score": 0.9, "unsupported": []}')
        self.assertTrue(d.supported)
        self.assertAlmostEqual(d.score, 0.9, places=2)

    def test_desteksiz(self) -> None:
        d = Dogrulama.coz('{"supported": false, "score": 0.2, '
                          '"unsupported": ["sürüm 5.2"]}')
        self.assertFalse(d.supported)
        self.assertEqual(d.unsupported, ["sürüm 5.2"])

    def test_string_true(self) -> None:
        d = Dogrulama.coz('{"supported": "true", "score": "0.8"}')
        self.assertTrue(d.supported)

    def test_puan_kirpilir(self) -> None:
        d = Dogrulama.coz('{"supported": true, "score": 7.5}')
        self.assertLessEqual(d.score, 1.0)

    def test_router_ile_desteksiz(self) -> None:
        r = _Router('{"supported": false, "score": 0.1, '
                    '"unsupported": ["iddia"]}')
        d = dogrula(r, "hedef", "yanıt", "kanıt")
        self.assertFalse(d.supported)

    def test_duzeltme(self) -> None:
        r = _Router("düzeltilmiş yanıt")
        yeni = duzelt(r, "hedef", "eski", ["iddia"], "kanıt")
        self.assertEqual(yeni, "düzeltilmiş yanıt")

    def test_duzeltme_bos_yanit(self) -> None:
        r = _Router("   ")
        self.assertIsNone(duzelt(r, "h", "e", ["x"], "k"))


# ─── 5) Katman sayaçları ────────────────────────────────────────────────────

class TestKatman(unittest.TestCase):

    def test_sayaclar(self) -> None:
        cog = CognitionLayer(max_reflect=1, verify=True, reflect_every=4)
        cog.initialize(router=_Router('{"verdict": "yeniden_planla"}'))
        cog.siniflandir("permission denied")
        cog.siniflandir("permission denied")
        cog.yansit("h", "d")
        st = cog.get_stats()
        self.assertEqual(st["classify"], 2)
        self.assertEqual(st["hata_turleri"].get("izin"), 2)
        self.assertEqual(st["reflect_calls"], 1)
        self.assertEqual(st["reflect_replan"], 1)
        self.assertEqual(st["max_reflect"], 1)

    def test_router_yoksa_cokmez(self) -> None:
        cog = CognitionLayer()
        cog.initialize()
        y = cog.yansit("h", "d")
        self.assertFalse(y.ok)
        self.assertEqual(cog.get_stats()["reflect_fail"], 1)


# ─── 6) Ajan döngüsü kancaları ──────────────────────────────────────────────

class TestDonguKancalari(unittest.TestCase):

    def _dongu(self) -> Any:
        from neurocore.agent.loop import AgentLoop

        loop = AgentLoop()
        loop._workspace = "."
        loop._config = None
        loop._event_bus = None
        return loop

    def _adimlar(self) -> Any:
        from neurocore.agent.loop import AgentRun, AgentStep

        run = AgentRun(task="test")
        run.steps = [
            AgentStep(index=0, kind="tool", tool="code.read_file",
                      result="dosya içeriği: 42 satır"),
            AgentStep(index=1, kind="tool", tool="shell.run",
                      result="çıktı: 1 2 3"),
            AgentStep(index=2, kind="plan", text="plan"),
        ]
        return run

    def test_kanit_metni(self) -> None:
        loop = self._dongu()
        kanit = loop._kanit_metni(self._adimlar())
        self.assertIn("code.read_file", kanit)
        self.assertIn("42 satır", kanit)
        self.assertNotIn("plan", kanit.split("[", 1)[0])   # plan kanıt değil

    def test_kisa_olgu(self) -> None:
        from neurocore.agent.loop import _kisa_olgu
        from neurocore.agent.tools import ToolResult

        self.assertIn("a.txt", _kisa_olgu(
            ToolResult(ok=True, tool="code.read_file", data={"path": "a.txt"})))
        self.assertIn("3 öğe", _kisa_olgu(
            ToolResult(ok=True, tool="x", data=[1, 2, 3])))
        self.assertIn("tamam", _kisa_olgu(
            ToolResult(ok=True, tool="x", data=None)))

    def test_cevap_duzeltilir(self) -> None:
        """Desteksiz yanıt → tek düzeltme çağrısı."""
        loop = self._dongu()

        class _SahteCog:
            verify = True

            def __init__(self) -> None:
                self.duzelt_cagri = 0

            def dogrula(self, hedef: str, cevap: str, kanit: str) -> Dogrulama:
                return Dogrulama(supported=False, score=0.2,
                                 unsupported=["uydurma sürüm"])

            def duzelt(self, hedef: str, cevap: str, eksik: Any,
                       kanit: str) -> str:
                self.duzelt_cagri += 1
                return "düzeltilmiş yanıt"

        loop._cog = _SahteCog()
        run = self._adimlar()
        sonuc = loop._cevabi_dogrula("görev", "uydurma yanıt", run)
        self.assertEqual(sonuc, "düzeltilmiş yanıt")
        self.assertEqual(loop._counter["cog_verify"], 1)
        self.assertEqual(loop._counter["cog_correct"], 1)
        # Doğrulama adımı kayda geçti
        self.assertTrue(any(s.kind == "verify" for s in run.steps))

    def test_ikinci_duzeltme_yok(self) -> None:
        loop = self._dongu()

        class _SahteCog:
            verify = True

            def dogrula(self, *a: Any) -> Dogrulama:
                return Dogrulama(supported=False, score=0.1, unsupported=["x"])

            def duzelt(self, *a: Any) -> str:
                return "yeni"

        loop._cog = _SahteCog()
        loop._counter["cog_correct"] = 1        # bütçe tükendi
        sonuc = loop._cevabi_dogrula("g", "yanıt", self._adimlar())
        self.assertEqual(sonuc, "yanıt")        # eldeki korunur

    def test_kanit_yoksa_dogrulama_atlanir(self) -> None:
        loop = self._dongu()

        class _SahteCog:
            verify = True

            def dogrula(self, *a: Any) -> Dogrulama:
                raise AssertionError("kanıt yokken çağrılmamalı")

        loop._cog = _SahteCog()
        from neurocore.agent.loop import AgentRun

        sonuc = loop._cevabi_dogrula("g", "yanıt", AgentRun(task="t"))
        self.assertEqual(sonuc, "yanıt")

    def test_yansima_siniri(self) -> None:
        """`reflect_every` ve `max_reflect` sınırları uygulanmalı."""
        loop = self._dongu()

        class _SahteCog:
            max_reflect = 1
            reflect_every = 2
            verify = False

            def __init__(self) -> None:
                self.cagri = 0

            def yansit(self, *a: Any) -> Yansima:
                self.cagri += 1
                return Yansima(verdict="devam")

        cog = _SahteCog()
        loop._cog = cog
        loop._hedef = HedefDurumu(hedef="g")
        run = self._adimlar()
        run.steps = [s for s in run.steps if s.kind == "tool"]   # 2 araç adımı
        mesajlar: List[Any] = []
        loop._yansima_kapisi("g", run, mesajlar)
        self.assertEqual(cog.cagri, 1, "ilk yansıma yapılmalı")
        loop._yansima_kapisi("g", run, mesajlar)
        self.assertEqual(cog.cagri, 1, "max_reflect=1 → ikinci çağrı yok")


class TestJsonCikar(unittest.TestCase):

    def test_duz(self) -> None:
        self.assertEqual(_json_cikar('{"a": 1}'), {"a": 1})

    def test_kod_bloklu(self) -> None:
        self.assertEqual(_json_cikar('```json\n{"a": 2}\n```'), {"a": 2})

    def test_metin_icinde(self) -> None:
        self.assertEqual(_json_cikar('cevap: {"a": 3} son'), {"a": 3})

    def test_bozuk(self) -> None:
        self.assertIsNone(_json_cikar("json yok"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
