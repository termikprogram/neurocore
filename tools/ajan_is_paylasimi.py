"""
Adım 3 Ölçümü — ajan döngüsü işi komşuya veriyor mu?
=====================================================
Senaryo (gerçek sunucu + gerçek tünel)
--------------------------------------
```
1. Yerel sahte düğüm HTTP sunucusu (127.0.0.1:8791) — /peer/job taklidi
2. PeerPool: ters tünel aç + kendini kaydet
3. **Sahte komşu** kaydı ekle (farklı node_id, aynı relay_port)
   → böylece `peers()` bir komşu görür (tek makinede bile)
4. AgentLoop._execute("web.search")  → komşuya gitmeli (kaynak: uzak)
5. AgentLoop._execute("shell.run")    → **komşuya GİTMEMELİ** (yerel)
6. Temizlik: sahte kayıt + kendi kaydı sil, tünel kapat
```

⚠ Bu bir "kendine relay" testidir: iki ayrı makine olmadan **zincirin
tamamını** (döngü → havuz → SSH → sunucu → tünel → HTTP → geri) ölçer.

Kullanım
--------
    python tools/ajan_is_paylasimi.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")

from neurocore.agent.loop import AgentLoop            # noqa: E402
from neurocore.agent.tools import ToolResult          # noqa: E402
from neurocore.net.peer_pool import PeerPool          # noqa: E402

PORT = 8791
SAHTE_DUGUM = "sahte-komsu-0001"


class _UzakDugum(BaseHTTPRequestHandler):
    """`/peer/job` taklidi — komşu düğüm gibi yanıt verir."""

    def do_POST(self) -> None:                        # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        ham = self.rfile.read(n) if n else b"{}"
        try:
            gelen = json.loads(ham.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            gelen = {}
        govde = {
            "ok": True, "ms": 1.0, "arac": gelen.get("tool"),
            "sonuc": {
                "ok": True, "tool": gelen.get("tool"),
                "data": {"nerede": "KOMSU DUGUM",
                         "arac": gelen.get("tool"),
                         "arg": (gelen.get("args") or {}).get("q")},
                "error": "", "safety": "read",
            },
        }
        veri = json.dumps(govde, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(veri)))
        self.end_headers()
        self.wfile.write(veri)

    def log_message(self, *a: Any) -> None:
        pass


class _YerelKayit:
    """Yerel araç kayıt defteri taklidi (çağrıları sayar)."""

    def __init__(self) -> None:
        self.cagrilar: List[str] = []
        self.policy = types.SimpleNamespace(dry_run=False)

    def has(self, ad: str) -> bool:
        return True

    def __len__(self) -> int:
        return 1

    def get(self, ad: str) -> Any:
        return None

    def call(self, ad: str, args: Any = None, **kw: Any) -> ToolResult:
        self.cagrilar.append(ad)
        return ToolResult(ok=True, tool=ad, data={"nerede": "YEREL"})


def _p(*parts: Any) -> None:
    print(" ".join(str(x) for x in parts), flush=True)


def main() -> int:
    srv = HTTPServer(("127.0.0.1", PORT), _UzakDugum)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    havuz = PeerPool(aralik=15.0)
    havuz.initialize(data_path=".neurocore")
    ok = 0
    toplam = 5

    _p("=" * 74)
    _p("  ADIM 3 ÖLÇÜMÜ — ajan döngüsü iş paylaşımı")
    _p("=" * 74)

    try:
        # ── 1) Tünel + kayıt ──
        _p("\n── 1) TÜNEL + KAYIT ──")
        r = havuz.baslat(port=PORT, caps=["code"], tunel=True)
        relay = int(r.get("tunel") or 0)
        _p(f"  ok={r.get('ok')} · relay_port={relay}")
        ok += 1 if relay else 0

        # ── 2) Sahte komşu kaydı (kendini hariç tutma kuralını aşmak için)
        _p("\n── 2) SAHTE KOMŞU KAYDI ──")
        sahte = {
            "node_id": SAHTE_DUGUM, "ad": "sahte-komsu",
            "ip": "10.9.9.9", "port": 0, "relay_port": relay,
            "caps": ["code"], "ts": time.time(), "surum": "test",
        }
        yz = havuz._ssh("python3 -", havuz._defter_betik("kayit", sahte))
        komsular = havuz.peers()
        _p(f"  kayıt ok={yz.get('ok')} · görünen komşu={len(komsular)}")
        for k in komsular:
            _p(f"    {k.node_id[:20]:22} {k.adres:18} canlı={k.canli}")
        ok += 1 if any(k.node_id == SAHTE_DUGUM for k in komsular) else 0

        # ── 3) Ajan döngüsü: izinli araç → KOMŞU ──
        _p("\n── 3) AJAN: web.search (izinli) ──")
        dongu = AgentLoop()
        dongu._registry = _YerelKayit()          # type: ignore[assignment]
        dongu._workspace = "."
        dongu._config = None
        dongu._event_bus = None
        dongu.attach_peer(havuz)
        kosu = types.SimpleNamespace(
            steps=[], tools_used=[], files_changed=[], commands_run=[],
            approvals=0, task="t", ok=True, reason="", provider="",
            model="", prompt_tokens=0, completion_tokens=0,
            duration_ms=0.0, cancelled=False, mode="model", error="")
        t0 = time.perf_counter()
        sonuc = dongu._execute("web.search", {"q": "dxgi"}, kosu)
        sure = time.perf_counter() - t0
        veri = sonuc.data if isinstance(sonuc.data, dict) else {}
        _p(f"  sonuç: {veri} · {sure:.1f} sn")
        _p(f"  yerel kayıt defteri çağrıları: {dongu._registry.cagrilar}")
        _p(f"  sayaçlar: {dict(dongu._counter)}")
        komsudan = veri.get("nerede") == "KOMSU DUGUM"
        if komsudan and not dongu._registry.cagrilar:
            ok += 1
            _p("  ✓ iş KOMŞUDA çalıştı, yerel hiç çağrılmadı")
        else:
            _p("  ✗ iş komşuya gitmedi")
        _p(f"  adım metni: {kosu.steps[-1].text!r}")

        # ── 4) İzinli olmayan araç → YEREL ──
        _p("\n── 4) AJAN: shell.run (izinli değil) ──")
        once = len(dongu._registry.cagrilar)
        s2 = dongu._execute("shell.run", {"command": "echo x"}, kosu)
        sonra = len(dongu._registry.cagrilar)
        v2 = s2.data if isinstance(s2.data, dict) else {}
        _p(f"  sonuç: {v2} · yerel çağrı {once} → {sonra}")
        if v2.get("nerede") == "YEREL" and sonra == once + 1:
            ok += 1
            _p("  ✓ tehlikeli araç YERELDE kaldı (komşuya gitmedi)")
        else:
            _p("  ✗ beklenmedik")

        # ── 5) Komşu erişilemezse yerel devralma ──
        _p("\n── 5) KOMŞU ERİŞİLEMEZ → YEREL DEVRALMA ──")
        # ⚠ Sahte komşunun relay portunu geçersiz yap
        bozuk = dict(sahte)
        bozuk["relay_port"] = 1
        havuz._ssh("python3 -", havuz._defter_betik("kayit", bozuk))
        once2 = len(dongu._registry.cagrilar)
        t0 = time.perf_counter()
        s3 = dongu._execute("web.search", {"q": "x"}, kosu)
        sure3 = time.perf_counter() - t0
        v3 = s3.data if isinstance(s3.data, dict) else {}
        _p(f"  sonuç: {v3} · {sure3:.1f} sn · "
           f"yerel çağrı: {len(dongu._registry.cagrilar) - once2}")
        if v3.get("nerede") == "YEREL":
            ok += 1
            _p("  ✓ komşu yanıtsız → yerel üstlendi (iş kaybolmadı)")
        else:
            _p("  ✗ yerel devralma olmadı")

    finally:
        _p("\n── 6) TEMİZLİK ──")
        try:
            havuz._ssh("python3 -",
                       havuz._defter_betik("birak",
                                           {"node_id": SAHTE_DUGUM}))
        except Exception:
            pass
        havuz.durdur()
        havuz.shutdown()
        srv.shutdown()
        kalan = havuz.peers(kendini_say=True) if havuz._initialized else []
        _p(f"  defterde kalan: {len(kalan)}")

    _p("\n" + "=" * 74)
    _p(f"  BAŞARILI: {ok}/{toplam}")
    _p("=" * 74)
    return 0 if ok >= 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
