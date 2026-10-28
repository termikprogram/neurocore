"""
Uçtan Uca Havuz Testi — gerçek sunucu, gerçek tünel
====================================================
Ne ölçer?
---------
```
1. TÜNEL    : ssh -R gerçekten açılıyor mu (sunucuda port görünüyor mu)
2. KAYIT    : defterde satır var mı, relay_port yazıldı mı
3. RELAY İŞ : komşu gibi davranıp **sunucu üzerinden** iş gönderiliyor mu
4. HEARTBEAT: periyodik kayıt tazeleniyor mu (ts güncelleniyor mu)
5. TEMİZLİK : durdur() sonrası defterden siliniyor + tünel kapanıyor
```

⚠ Test, `/peer/job` yerine **sahte bir yerel sunucu** kullanır (uygulamayı
başlatmadan taşımayı ölçmek için). Sahte sunucu JSON döner.

Kullanım
--------
    python tools/havuz_uctan_uca.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")

from neurocore.net.peer_pool import DEFTER, Komsu, PeerPool  # noqa: E402

PORT = 8791
ARALIK = 6.0

YANIT = {"ok": True, "sonuc": {"nerede": "UZAK DUGUM", "deger": 42}}


class _Isleyici(BaseHTTPRequestHandler):
    def do_POST(self) -> None:                    # noqa: N802
        uzunluk = int(self.headers.get("Content-Length") or 0)
        ham = self.rfile.read(uzunluk) if uzunluk else b"{}"
        try:
            gelen = json.loads(ham.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            gelen = {}
        # ⚠ Token doğrulaması burada yapılmıyor (bu sahte uç)
        govde = dict(YANIT)
        govde["gelen_arac"] = gelen.get("tool")
        govde["token_var"] = bool(gelen.get("token"))
        veri = json.dumps(govde, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(veri)))
        self.end_headers()
        self.wfile.write(veri)

    def log_message(self, *a: Any) -> None:
        pass


def _p(*parts: Any) -> None:
    print(" ".join(str(x) for x in parts), flush=True)


def main() -> int:
    srv = HTTPServer(("127.0.0.1", PORT), _Isleyici)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    havuz = PeerPool(aralik=ARALIK)
    havuz.initialize(data_path=".neurocore")
    sonuc: Dict[str, Any] = {"adim": {}}

    _p("=" * 74)
    _p("  UÇTAN UCA HAVUZ TESTİ — gerçek sunucu")
    _p("=" * 74)
    _p(f"  düğüm    : {havuz._kimlik}")
    _p(f"  yerel uç : 127.0.0.1:{PORT} (sahte /peer/job)")
    _p(f"  aralık   : {ARALIK} sn")

    ok_sayisi = 0
    try:
        # ── 1-2) Başlat: tünel + kayıt ─────────────────────────────────
        _p("\n── 1) BAŞLAT (tünel + kayıt) ──")
        t0 = time.perf_counter()
        r = havuz.baslat(port=PORT, caps=["code", "blender"], tunel=True)
        _p(f"  ok={r.get('ok')} · tünel_port={r.get('tunel')} · "
           f"{time.perf_counter() - t0:.1f} sn")
        kayit = r.get("kayit") or {}
        _p(f"  kayıt: ip={kayit.get('ip')} · ok={kayit.get('ok')}")
        sonuc["adim"]["baslat"] = {"ok": bool(r.get("ok")),
                                   "tunel": r.get("tunel")}
        ok_sayisi += 1 if r.get("ok") else 0

        # ── Sunucuda port görünüyor mu? ────────────────────────────────
        relay = int(r.get("tunel") or 0)
        if relay:
            _p("\n── 1b) SUNUCUDA TÜNEL PORTU ──")
            cikti = havuz._ssh(f"ss -ltn 2>/dev/null | grep {relay} || "
                               f"echo 'YOK'")
            satir = (cikti.get("cikti") or "").strip().splitlines()
            _p(f"  {satir[0].strip() if satir else '?'}")
            if satir and "YOK" not in satir[0]:
                ok_sayisi += 1

        # ── 3) Relay ile iş gönder ─────────────────────────────────────
        _p("\n── 2) RELAY İLE İŞ GÖNDERİMİ ──")
        komsu = Komsu(node_id="kendim-gibi", ad="test-peer",
                      ip=kayit.get("ip", ""), port=PORT,
                      relay_port=relay, caps=["code"],
                      ts=time.time())
        t0 = time.perf_counter()
        y = havuz.gonder(komsu, {"tool": "web.search",
                                 "args": {"q": "dxgi"}})
        _p(f"  ok={y.get('ok')} · {time.perf_counter() - t0:.1f} sn")
        _p(f"  yanıt: {json.dumps(y, ensure_ascii=False)[:150]}")
        if y.get("ok"):
            ok_sayisi += 1
            ic = y.get("sonuc") or {}
            if isinstance(ic, dict) and ic.get("token_var"):
                _p("  ✓ token relay üzerinden taşındı")

        # ── 4) Heartbeat ───────────────────────────────────────────────
        _p(f"\n── 3) HEARTBEAT ({ARALIK + 2:.0f} sn bekle) ──")
        onceki = havuz._son_kayit
        ilk_adet = havuz.get_stats()["kayit"]
        time.sleep(ARALIK + 2.0)
        st = havuz.get_stats()
        _p(f"  kayıt sayısı: {ilk_adet} → {st['kayit']} · "
           f"heartbeat={st['heartbeat']}")
        _p(f"  son kayıt yaşı: {st['son_kayit_yas']} sn")
        if st["kayit"] > ilk_adet and st["son_kayit_yas"] < ARALIK + 1:
            ok_sayisi += 1
            _p("  ✓ kayıt tazelendi")
        else:
            _p("  ✗ tazelenmedi")
        sonuc["adim"]["heartbeat"] = {
            "kayit": st["kayit"], "yas": st["son_kayit_yas"]}

        # ── Defterde bizim satır ───────────────────────────────────────
        _p("\n── 4) DEFTERDE SATIR ──")
        komsular = havuz.peers(kendini_say=True)
        benim = [k for k in komsular if k.node_id == havuz._kimlik]
        if benim:
            _p(f"  {benim[0].ad} · {benim[0].adres} · "
               f"yaş={benim[0].yas:.1f}s · canlı={benim[0].canli}")
            ok_sayisi += 1
        else:
            _p("  ✗ kendi satırım bulunamadı")

    finally:
        # ── 5) Temizlik ────────────────────────────────────────────────
        _p("\n── 5) DURDUR (temizlik) ──")
        d = havuz.durdur()
        _p(f"  ok={d.get('ok')} · kayıt silindi="
           f"{(d.get('birak') or {}).get('ok')} · "
           f"tünel kapandı={d.get('tunel_kapandi')}")
        # ⚠ Defterde gerçekten yok mu?
        kalan = [k for k in havuz.peers(kendini_say=True)
                 if k.node_id == havuz._kimlik]
        _p(f"  defterde kalan kendi satırı: {len(kalan)} "
           f"{'✓' if not kalan else '✗'}")
        if not kalan:
            ok_sayisi += 1
        srv.shutdown()
        havuz.shutdown()

    _p("\n" + "=" * 74)
    _p(f"  BAŞARILI ADIM: {ok_sayisi}/6")
    _p("=" * 74)
    return 0 if ok_sayisi >= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
