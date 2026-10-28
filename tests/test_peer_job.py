"""
`/peer/job` — uzak düğümden iş alma testleri
==============================================
Korunan davranışlar:

```
1. TOKEN      : paylaşılan anahtar yanlışsa **çalışmaz** (403)
2. İZİN       : `shell.run` / `code.run_command` gibi araçlar uzaktan
                **asla** çalışmaz (yalnız okuma listesi)
3. ONAY       : onay gerektiren araç uzakta çalıştırılmaz
4. HATA       : araç hatası çağırana **açıkça** döner
5. BAŞARILI   : izinli araç çalışır, JSON-güvenli sonuç döner
6. HTTP       : gerçek POST /peer/job (FastAPI TestClient)
```

⚠ Neden token? Ölçüldü: ters tünel sunucunun `127.0.0.1`'ine bağlanır ve
relay isteği sunucu üzerinden gelir → **kaynak IP güvenilir değildir**.
"""

from __future__ import annotations

import types
import unittest
from typing import Any, Dict, Optional

from neurocore.agent.tools import Safety, ToolParameter, ToolRegistry, ToolResult


# ─── Sahte kayıt defteri ────────────────────────────────────────────────────

class _SahteKayit:
    """`ToolRegistry` yerine geçen asgari nesne."""

    def __init__(self, adlar: Optional[Dict[str, Dict[str, Any]]] = None):
        self.adlar = adlar or {}
        self.cagrilar: list = []

    def has(self, ad: str) -> bool:
        return ad in self.adlar

    def __len__(self) -> int:
        return len(self.adlar)

    def call(self, ad: str, args: Optional[Dict[str, Any]] = None,
             **kw: Any) -> ToolResult:
        self.cagrilar.append((ad, args))
        tanim = self.adlar.get(ad) or {}
        if tanim.get("onay"):
            return ToolResult(ok=False, tool=ad, needs_approval=True,
                              error="onay gerekir")
        if tanim.get("hata"):
            return ToolResult(ok=False, tool=ad, error=tanim["hata"])
        return ToolResult(ok=True, tool=ad,
                          data={"sonuc": tanim.get("veri", "tamam")})


def _durum(kayit: Any) -> Any:
    return types.SimpleNamespace(registry=kayit)


# ─── Testler ────────────────────────────────────────────────────────────────

class TestPeerJobMantik(unittest.TestCase):
    """`peer_job_isle` — HTTP olmadan."""

    def setUp(self) -> None:
        # ⚠ Küme anahtarı artık SSH anahtarından türetilmez; testte
        # ortam değişkeniyle verilir (sunucudan alınmış gibi).
        import os

        self._eski_token = os.environ.get("NC_PEER_TOKEN")
        os.environ["NC_PEER_TOKEN"] = "test-kume-anahtari-1234"

        from neurocore.app.server import PEER_IZINLI_ARACLAR, peer_job_isle
        from neurocore.net.peer_pool import yerel_token

        self.isle = peer_job_isle
        self.token = yerel_token()
        self.izinli = PEER_IZINLI_ARACLAR
        self.kayit = _SahteKayit({
            "web.search": {"veri": [{"baslik": "DXGI"}]},
            "knowledge.ask": {"veri": "kasa cevabı"},
            "shell.run": {"veri": "TEHLİKELİ"},
            "code.run_command": {"veri": "TEHLİKELİ"},
        })
        self.durum = _durum(self.kayit)

    def tearDown(self) -> None:
        import os

        if self._eski_token is None:
            os.environ.pop("NC_PEER_TOKEN", None)
        else:
            os.environ["NC_PEER_TOKEN"] = self._eski_token

    # 1) TOKEN
    def test_token_yanlissa_calismaz(self) -> None:
        r = self.isle(self.durum, {"token": "yanlis", "tool": "web.search",
                                   "args": {"q": "x"}})
        self.assertFalse(r["ok"])
        self.assertIn("yetkisiz", r["hata"])
        self.assertEqual(self.kayit.cagrilar, [], "araç çağrılmamalıydı")

    def test_token_dogruysa_calisir(self) -> None:
        r = self.isle(self.durum, {"token": self.token, "tool": "web.search",
                                   "args": {"q": "dxgi"}})
        self.assertTrue(r["ok"])
        self.assertEqual(self.kayit.cagrilar[0][0], "web.search")

    # 2) İZİN
    def test_tehlikeli_arac_reddedilir(self) -> None:
        for ad in ("shell.run", "code.run_command"):
            r = self.isle(self.durum, {"token": self.token, "tool": ad,
                                       "args": {}})
            self.assertFalse(r["ok"], f"{ad} geçmemeliydi")
            self.assertIn("uzaktan çalıştırılamaz", r["hata"])
        self.assertEqual(self.kayit.cagrilar, [])

    def test_izin_listesi_yalniz_okuma(self) -> None:
        """İzin listesinde tehlikeli araç olmamalı."""
        for ad in ("shell.run", "shell.session_run", "code.run_command",
                   "code.edit_file", "code.create_file"):
            self.assertNotIn(ad, self.izinli, f"{ad} izinli olmamalı")

    # 3) ONAY
    def test_onay_gerektiren_arac(self) -> None:
        kayit = _SahteKayit({"web.fetch": {"onay": True}})
        r = self.isle(_durum(kayit), {"token": self.token, "tool": "web.fetch",
                                      "args": {"url": "http://x"}})
        self.assertFalse(r["ok"])
        self.assertIn("onay", r["hata"])

    # 4) HATA
    def test_arac_hatasi_iletilir(self) -> None:
        kayit = _SahteKayit({"knowledge.ask": {"hata": "kasa boş"}})
        r = self.isle(_durum(kayit), {"token": self.token,
                                      "tool": "knowledge.ask", "args": {}})
        self.assertFalse(r["ok"])
        self.assertIn("kasa boş", r["hata"])

    # 5) BAŞARILI
    def test_basarili_json_guvenli(self) -> None:
        from tests.helpers import assert_json_safe

        r = self.isle(self.durum, {"token": self.token, "tool": "web.search",
                                   "args": {"q": "dxgi"}})
        self.assertTrue(r["ok"])
        self.assertIn("sonuc", r)
        assert_json_safe(self, r)

    def test_gecersiz_govde(self) -> None:
        r = self.isle(self.durum, ["liste"])          # type: ignore[arg-type]
        self.assertFalse(r["ok"])
        self.assertIn("sözlük", r["hata"])

    def test_token_yoksa_uc_kapali(self) -> None:
        """
        ⚠ Küme anahtarı yoksa istek **reddedilmeli** (uç açık kalmamalı).

        ⚠ Test belirleyici olmalı: gerçek `.neurocore/peer_token` dosyası
        varsa `yerel_token()` onu okur (ölçüm yaptıktan sonra bu dosya
        gerçekten var). Bu yüzden anahtar arama **sahtelenir**.
        """
        import os
        from unittest import mock

        os.environ.pop("NC_PEER_TOKEN", None)
        with mock.patch("neurocore.net.peer_pool.yerel_token",
                        return_value=""):
            r = self.isle(self.durum, {"token": "", "tool": "web.search",
                                       "args": {}})
        self.assertFalse(r["ok"])
        self.assertIn("küme anahtarı yok", r["hata"])
        # ⚠ Anahtar yokken araç **çağrılmamalı**
        self.assertEqual(self.kayit.cagrilar, [])

    def test_anahtar_tokeni_etkilemez(self) -> None:
        """
        ⚠ KULLANICI KURALI: SSH anahtarı istemcilerle paylaşılmaz.

        Bu yüzden token **anahtardan türetilmemelidir**: iki farklı anahtar
        yolu aynı sonucu vermeli (anahtar yoksa boş).
        """
        import os

        os.environ.pop("NC_PEER_TOKEN", None)
        from neurocore.net.peer_pool import ortak_token, yerel_token

        self.assertEqual(ortak_token(r"C:\a\anahtar.key"),
                         ortak_token(r"D:\b\baska.key"))
        self.assertEqual(yerel_token(), ortak_token())

    def test_kayit_defteri_yok(self) -> None:
        r = self.isle(types.SimpleNamespace(),
                      {"token": self.token, "tool": "web.search"})
        self.assertFalse(r["ok"])
        self.assertIn("kayıt defteri", r["hata"])


class TestPeerJobHTTP(unittest.TestCase):
    """6) Gerçek HTTP: FastAPI TestClient ile POST /peer/job."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import neurocore.app.server as S

            if not S.HAS_FASTAPI:
                raise unittest.SkipTest("fastapi yok")
            from fastapi.testclient import TestClient
        except Exception as exc:                    # pragma: no cover
            raise unittest.SkipTest(f"TestClient yok: {type(exc).__name__}")

        cls.S = S
        cls.istemci = None
        cls.hata = ""
        try:
            # ⚠ Asgari sahte durum: create_app'in tanım anında dokunduğu
            # alanlar sağlanır; rotalar çalıştırılmadan kurulur.
            import threading

            kayit = _SahteKayit({"web.search": {"veri": "HTTP'den geldi"},
                                 "shell.run": {"veri": "TEHLİKELİ"}})

            class _OtomatikSahte:
                """⚠ İstenen her metoda makul yanıt verir.

                `create_app` onay/olay katmanlarını **kurarken** dinleyici
                ekler (`add_listener`, `add_resolution_listener`…). Sahte
                durumda bunları tek tek yazmak yerine otomatik karşılanır.
                """

                def __getattr__(self, ad: str) -> Any:
                    if ad.startswith("add_"):
                        return lambda *a, **k: None
                    return lambda *a, **k: []

            durum = types.SimpleNamespace(
                registry=kayit,
                router=None, loop=None, bridge=None, thinking=None,
                local_rank=None, knowledge_work=None, coding=None,
                shell=None, vision=None, godot=None, blender=None,
                approvals=_OtomatikSahte(), events=_OtomatikSahte(),
                workspace=".", autonomy="supervised", config=None,
                lock=threading.RLock(),
            )
            app = S.create_app(durum, static_dir=None)
            cls.istemci = TestClient(app)
            cls.kayit = kayit
        except Exception as exc:                    # pragma: no cover
            cls.hata = f"{type(exc).__name__}: {exc}"

    def setUp(self) -> None:
        if self.istemci is None:
            self.skipTest(f"uygulama kurulamadı: {self.hata[:80]}")
        # ⚠ Endpoint küme anahtarını **yerelde** okur (SSH'sız)
        import os

        self._eski = os.environ.get("NC_PEER_TOKEN")
        os.environ["NC_PEER_TOKEN"] = "test-kume-anahtari-1234"

    def tearDown(self) -> None:
        import os

        if getattr(self, "_eski", None) is None:
            os.environ.pop("NC_PEER_TOKEN", None)
        else:
            os.environ["NC_PEER_TOKEN"] = self._eski

    def test_peer_info(self) -> None:
        y = self.istemci.get("/peer/info")
        self.assertEqual(y.status_code, 200)
        veri = y.json()
        self.assertTrue(veri.get("ok"))
        self.assertIn("izinli_araclar", veri)

    def test_job_dogru_token(self) -> None:
        from neurocore.net.peer_pool import yerel_token

        y = self.istemci.post("/peer/job", json={
            "token": yerel_token(), "tool": "web.search",
            "args": {"q": "dxgi"}})
        self.assertEqual(y.status_code, 200)
        self.assertTrue(y.json().get("ok"))

    def test_job_yanlis_token_403(self) -> None:
        y = self.istemci.post("/peer/job", json={
            "token": "yanlis", "tool": "web.search", "args": {}})
        self.assertEqual(y.status_code, 403)

    def test_job_izinli_olmayan_arac(self) -> None:
        from neurocore.net.peer_pool import yerel_token

        y = self.istemci.post("/peer/job", json={
            "token": yerel_token(), "tool": "shell.run",
            "args": {"command": "echo pwned"}})
        self.assertEqual(y.status_code, 200)
        self.assertFalse(y.json().get("ok"))
        self.assertIn("uzaktan çalıştırılamaz", y.json().get("hata", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
