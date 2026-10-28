"""
Adım 3 — Ajan döngüsünde iş paylaşımı testleri
===============================================
Korunan davranışlar:

```
1. ÖNCE KOMŞU : izinli okuma aracı önce komşuda denenir
2. YEREL DEVRALMA : komşu yanıt vermezse araç **yerelde** çalışır
3. İZİN       : izin listesinde olmayan araç (shell.run) komşuya GİTMEZ
4. HAVUZ YOK  : havuz bağlı değilse davranış eskisi gibi (yerel)
5. HATA       : komşu istisna atarsa çökmez, yerel devralır
6. İZ         : adım metninde kaynak görünür (`[uzak:…]` / `[yerel]`)
```

⚠ Ağ çağrısı yok: havuz ve kayıt defteri sahtedir.
"""

from __future__ import annotations

import types
import unittest
from typing import Any, Dict, List, Optional

from neurocore.agent.loop import AgentLoop
from neurocore.agent.tools import ToolResult


class _SahteKayit:
    """`ToolRegistry` yerine: çağrıları kaydeder, sabit sonuç döner."""

    def __init__(self) -> None:
        self.cagrilar: List[str] = []
        # ⚠ `_build_context` kayıt defterinin **politikasını** okur:
        # `autonomy=self._registry.policy`, `dry_run=policy.dry_run`.
        self.policy = types.SimpleNamespace(dry_run=False)

    def has(self, ad: str) -> bool:
        return True

    def __len__(self) -> int:
        return 1

    def get(self, ad: str) -> Any:
        return None

    def call(self, ad: str, args: Optional[Dict[str, Any]] = None,
             **kw: Any) -> ToolResult:
        self.cagrilar.append(ad)
        return ToolResult(ok=True, tool=ad, data={"kaynak": "yerel"})


class _SahteHavuz:
    """`PeerPool` yerine: `komsu_dene` davranışı ayarlanabilir."""

    def __init__(self, *, basarili: bool = True,
                 patlar: bool = False) -> None:
        self.basarili = basarili
        self.patlar = patlar
        self.cagrilar: List[Dict[str, Any]] = []

    def komsu_dene(self, istek: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
        self.cagrilar.append(istek)
        if self.patlar:
            raise RuntimeError("ağ koptu")
        if self.basarili:
            return {"ok": True, "kaynak": "relay:25949",
                    "sonuc": {"ok": True, "tool": istek.get("tool"),
                              "data": {"kaynak": "uzak"}, "error": ""}}
        return {"ok": False, "denenen": ["relay:25949"]}


def _run() -> Any:
    return types.SimpleNamespace(
        steps=[], tools_used=[], files_changed=[], commands_run=[],
        approvals=0, task="test", ok=True, reason="", provider="",
        model="", prompt_tokens=0, completion_tokens=0, duration_ms=0.0,
        cancelled=False, mode="model", error="",
    )


def _loop(havuz: Optional[Any] = None) -> AgentLoop:
    loop = AgentLoop()
    loop._registry = _SahteKayit()          # type: ignore[assignment]
    loop._workspace = "."
    loop._config = None
    loop._event_bus = None
    if havuz is not None:
        loop.attach_peer(havuz)
    return loop


class TestKomsuOnce(unittest.TestCase):
    """1-6) Komşu önce, yoksa yerel."""

    def test_komsu_basariliysa_yerel_cagrilmaz(self) -> None:
        havuz = _SahteHavuz(basarili=True)
        loop = _loop(havuz)
        run = _run()
        r = loop._execute("web.search", {"q": "dxgi"}, run)
        self.assertIsNotNone(r)
        self.assertTrue(r.ok)
        # ⚠ Sonuç komşudan geldi
        self.assertEqual((r.data or {}).get("kaynak"), "uzak")
        # ⚠ Yerel kayıt defteri HİÇ çağrılmamalı
        self.assertEqual(loop._registry.cagrilar, [], "yerel çağrıldı")
        self.assertEqual(len(havuz.cagrilar), 1)
        st = loop._counter
        self.assertEqual(st["peer_calls"], 1)
        self.assertEqual(st["peer_ok"], 1)
        self.assertEqual(st["peer_fallback"], 0)
        # ⚠ Adım metninde kaynak
        self.assertIn("[uzak:relay:25949]", run.steps[-1].text)

    def test_komsu_yanitsizsa_yerel_devralir(self) -> None:
        havuz = _SahteHavuz(basarili=False)
        loop = _loop(havuz)
        run = _run()
        r = loop._execute("knowledge.ask", {"q": "x"}, run)
        self.assertTrue(r.ok)
        self.assertEqual((r.data or {}).get("kaynak"), "yerel")
        self.assertEqual(loop._registry.cagrilar, ["knowledge.ask"])
        self.assertEqual(loop._counter["peer_fallback"], 1)
        self.assertEqual(loop._counter["peer_ok"], 0)
        self.assertIn("[yerel]", run.steps[-1].text)

    def test_izinli_olmayan_arac_komsuya_gitmez(self) -> None:
        havuz = _SahteHavuz(basarili=True)
        loop = _loop(havuz)
        run = _run()
        for ad in ("shell.run", "code.run_command", "code.edit_file"):
            loop._execute(ad, {"command": "echo x"}, run)
        self.assertEqual(havuz.cagrilar, [], "tehlikeli araç komşuya gitti")
        self.assertEqual(loop._counter["peer_calls"], 0)
        self.assertEqual(len(loop._registry.cagrilar), 3)

    def test_havuz_yoksa_yerel(self) -> None:
        loop = _loop(None)
        run = _run()
        r = loop._execute("web.search", {"q": "x"}, run)
        self.assertTrue(r.ok)
        self.assertEqual(loop._registry.cagrilar, ["web.search"])
        self.assertEqual(loop._counter["peer_calls"], 0)

    def test_komsu_istisna_atarsa_cokmez(self) -> None:
        havuz = _SahteHavuz(basarili=True, patlar=True)
        loop = _loop(havuz)
        run = _run()
        r = loop._execute("web.fetch", {"url": "http://x"}, run)
        self.assertTrue(r.ok, "yerel devralmalıydı")
        self.assertEqual((r.data or {}).get("kaynak"), "yerel")
        self.assertEqual(loop._counter["peer_fallback"], 1)

    def test_attach_none_kapatir(self) -> None:
        havuz = _SahteHavuz(basarili=True)
        loop = _loop(havuz)
        loop.attach_peer(None)
        run = _run()
        loop._execute("web.search", {"q": "x"}, run)
        self.assertEqual(havuz.cagrilar, [], "havuz kapalıyken çağrıldı")
        self.assertEqual(loop._registry.cagrilar, ["web.search"])


class TestIzinListesiOrtak(unittest.TestCase):
    """İzin listesi **tek** kaynaktan gelmeli (sunucu + döngü)."""

    def test_ayni_liste(self) -> None:
        from neurocore.app.server import PEER_IZINLI_ARACLAR as sunucu
        from neurocore.net.peer_pool import IZINLI_ARACLAR as dongu

        self.assertEqual(tuple(sunucu), tuple(dongu),
                         "iki taraf farklı izin listesi kullanıyor")

    def test_tehlikeli_araclar_yok(self) -> None:
        from neurocore.net.peer_pool import IZINLI_ARACLAR

        for ad in ("shell.run", "code.run_command", "code.edit_file",
                   "code.create_file", "shell.session_run"):
            self.assertNotIn(ad, IZINLI_ARACLAR)


if __name__ == "__main__":
    unittest.main(verbosity=2)
