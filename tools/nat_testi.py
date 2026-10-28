"""
NAT aşma testi — ters SSH tüneli gerçekten çalışıyor mu?
=========================================================
⚠ ÖLÇÜLEN SORUN
---------------
```
yerel makine : 192.168.1.199/24
genel IP     : 95.70.193.163   (router)
```
Dışarıdan `95.70.193.163:port` **router'a** düşer. Bu yüzden komşular
birbirine doğrudan bağlanamaz.

Bu test
-------
```
1. Yerelde küçük bir HTTP sunucu aç (8799)
2. `ssh -R 18899:localhost:8799` ile sunucuya **ters tünel** aç
3. Sunucunun kendisinden `http://127.0.0.1:18899/` iste
4. Yanıt geldiyse → ters tünel NAT'ı AŞIYOR
```

⚠ İkinci ölçüm: `sshd` `GatewayPorts` kapalıysa tünel yalnız sunucunun
`127.0.0.1`'ine bağlanır → **başka bir komşu doğrudan erişemez**.
O durumda iş, sunucu üzerinden **relay** ile taşınır (ikinci test).
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

#: ⚠ Proje kökü (`tools/` değil) — yoksa `import neurocore` başarısız
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neurocore.net.peer_pool import VARSAYILAN_ANAHTAR, VARSAYILAN_SUNUCU

YEREL_PORT = 8799
UZAK_PORT = 18899
ISARET = b"NEUROCORE-TUNEL-TESTI"


class _Isleyici(BaseHTTPRequestHandler):
    def do_GET(self) -> None:            # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(ISARET)

    def log_message(self, *args: object) -> None:
        pass


def _yerel_sunucu() -> HTTPServer:
    s = HTTPServer(("127.0.0.1", YEREL_PORT), _Isleyici)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def _ssh(temel: list, komut: str, timeout: float = 20.0) -> str:
    try:
        p = subprocess.run(temel + [komut], text=True, capture_output=True,
                           timeout=timeout, encoding="utf-8",
                           errors="replace")
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return f"HATA {type(exc).__name__}: {exc}"


def main() -> int:
    ssh = shutil.which("ssh")
    if not ssh:
        print("  ✗ ssh yok")
        return 1
    temel = [ssh, "-i", VARSAYILAN_ANAHTAR, "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
             "-o", "ExitOnForwardFailure=yes", VARSAYILAN_SUNUCU]

    print("=" * 74)
    print("  NAT AŞMA TESTİ — ters SSH tüneli")
    print("=" * 74)

    # ── 0) Yerel port boş mu ───────────────────────────────────────────
    try:
        with socket.create_connection(("127.0.0.1", YEREL_PORT), timeout=1):
            print(f"  ⚠ yerel {YEREL_PORT} dolu — başka port gerekebilir")
    except OSError:
        pass

    srv = _yerel_sunucu()
    print(f"\n  1) yerel HTTP    : 127.0.0.1:{YEREL_PORT} (açıldı)")

    # ── 2) Ters tünel ──────────────────────────────────────────────────
    tunel = subprocess.Popen(
        temel + ["-N", "-R", f"{UZAK_PORT}:127.0.0.1:{YEREL_PORT}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace")
    print(f"  2) ters tünel    : sunucu:{UZAK_PORT} → yerel:{YEREL_PORT}")
    time.sleep(6.0)

    if tunel.poll() is not None:
        hata = (tunel.stderr.read() if tunel.stderr else "")[:200]
        print(f"  ✗ tünel KURULAMADI (çıktı): {hata}")
        srv.shutdown()
        return 1
    print("     tünel süreci    : çalışıyor")

    # ── 3) Sunucudan istek (relay yolu) ────────────────────────────────
    kod = (
        "import urllib.request;"
        f"d=urllib.request.urlopen('http://127.0.0.1:{UZAK_PORT}/',"
        "timeout=8).read();"
        f"print('ALINDI' if d=={ISARET!r} else 'FARKLI:'+repr(d[:30]))"
    )
    t0 = time.perf_counter()
    cikti = _ssh(temel, f"python3 -c {json.dumps(kod)}")
    sure = time.perf_counter() - t0
    satir = [s for s in cikti.splitlines() if s.strip()][:1]
    print(f"\n  3) sunucudan istek: {satir[0] if satir else '(boş)'} "
          f"· {sure:.1f} sn")
    basarili = bool(satir) and "ALINDI" in satir[0]
    print(f"     SONUÇ          : "
          f"{'✓ TERS TÜNEL ÇALIŞIYOR — NAT aşıldı' if basarili else '✗ tünel yanıt vermedi'}")

    # ── 4) GatewayPorts durumu (başka komşu doğrudan erişebilir mi?) ───
    gp = _ssh(temel, "grep -i '^GatewayPorts' /etc/ssh/sshd_config "
                     "2>/dev/null || echo 'TANIMSIZ(varsayılan=no)'")
    gp = (gp or "").strip().splitlines()[:1]
    print(f"\n  4) GatewayPorts   : {gp[0] if gp else '?'}")
    print("     (no ise tünel yalnız sunucunun 127.0.0.1'ine bağlanır;"
          " komşular işi **sunucu üzerinden relay** ile taşımalı)")

    # ── 5) Dinleme adresi ──────────────────────────────────────────────
    din = _ssh(temel, f"ss -ltn 2>/dev/null | grep {UZAK_PORT} || "
                      f"netstat -ltn 2>/dev/null | grep {UZAK_PORT} || "
                      f"echo 'görünmüyor'")
    din = (din or "").strip().splitlines()[:1]
    print(f"\n  5) dinleme        : {din[0].strip() if din else '?'}")

    tunel.terminate()
    try:
        tunel.wait(timeout=5)
    except subprocess.TimeoutExpired:
        tunel.kill()
    srv.shutdown()
    print("\n" + "=" * 74)
    print(f"  KARAR: {'NAT ters tünelle AŞILDI' if basarili else 'AŞILAMADI'}")
    print("  (iş paylaşımı: relay_port doluysa istek sunucu üzerinden gider)")
    print("=" * 74)
    return 0 if basarili else 1


if __name__ == "__main__":
    raise SystemExit(main())
