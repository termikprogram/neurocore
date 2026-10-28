"""
Ortak Havuz Ölçüm Aracı — gerçek sunucuya karşı
=================================================
Yaptığı ölçümler
----------------
```
1. KAYIT   : düğüm kendini sunucudaki deftere yazar (IP dahil)
2. KALICI  : defter dosyası sunucuda gerçekten var mı (cat ile)
3. KEŞİF   : defterden komşular okunur (kendisi hariç)
4. BAYAT   : TTL'i geçmiş kayıt düşürülür mü
5. DEVRALMA: komşu yanıt vermezse **yerel** iş çalışır mı
6. BIRAK   : temiz kapanışta kayıt silinir
```

Kullanım
--------
    python tools/ag_havuzu.py --dene
    python tools/ag_havuzu.py --kaydet --port 8791
    python tools/ag_havuzu.py --liste
    python tools/ag_havuzu.py --birak
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from neurocore.net.peer_pool import DEFTER, Komsu, PeerPool  # noqa: E402


def _p(*parts: object) -> None:
    print(" ".join(str(x) for x in parts), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ortak havuz ölçümü")
    ap.add_argument("--dene", action="store_true", help="Tam senaryo")
    ap.add_argument("--kaydet", action="store_true")
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--birak", action="store_true")
    ap.add_argument("--port", type=int, default=8791)
    args = ap.parse_args()

    havuz = PeerPool()
    havuz.initialize(data_path=".neurocore")

    _p("=" * 74)
    _p("  ORTAK HAVUZ — gerçek sunucu ölçümü")
    _p("=" * 74)
    _p(f"  düğüm kimliği : {havuz._kimlik}")
    _p(f"  sunucu        : {havuz._sunucu}")
    _p(f"  defter        : ~/{DEFTER}")

    if not args.dene and not (args.kaydet or args.liste or args.birak):
        args.dene = True

    # ── 1) KAYIT ───────────────────────────────────────────────────────
    if args.kaydet or args.dene:
        _p("\n── 1) KAYIT ──")
        t0 = time.perf_counter()
        r = havuz.register(port=args.port,
                           caps=["code", "blender", "arastirma", "egitim"])
        sure = time.perf_counter() - t0
        _p(f"  ok={r.get('ok')} · ip={r.get('ip')} · "
           f"düğüm={r.get('komsu')} · {sure:.1f} sn")
        if not r.get("ok"):
            _p(f"  ✗ {r.get('hata')}")

    # ── 2) SUNUCUDA DOSYA GERÇEKTEN VAR MI ─────────────────────────────
    if args.dene:
        _p("\n── 2) KALICILIK (sunucuda cat) ──")
        rr = havuz._ssh(f'cat "$HOME/{DEFTER}" 2>/dev/null | tail -5')
        if rr.get("ok"):
            satirlar = [s for s in (rr.get("cikti") or "").splitlines() if s]
            _p(f"  defterde {len(satirlar)} satır (son 5):")
            for s in satirlar[-5:]:
                try:
                    d = json.loads(s)
                    _p(f"    {d.get('ad'):16} {d.get('ip'):16} "
                       f"port={d.get('port')} caps={d.get('caps')}")
                except json.JSONDecodeError:
                    _p(f"    {s[:70]}")
        else:
            _p(f"  ✗ okunamadı: {rr.get('hata')}")

    # ── 3) KEŞİF ───────────────────────────────────────────────────────
    if args.liste or args.dene:
        _p("\n── 3) KEŞİF ──")
        komsular = havuz.peers()
        _p(f"  komşu sayısı : {len(komsular)} (kendisi hariç)")
        for k in komsular:
            _p(f"    {k.ad:16} {k.adres:22} yaş={k.yas:5.1f}s "
               f"canlı={k.canli} caps={k.caps}")
        if not komsular:
            _p("    (havuzda başka düğüm yok — tek makine)")

    # ── 4) BAYAT SATIR ─────────────────────────────────────────────────
    if args.dene:
        _p("\n── 4) BAYAT KAYIT ──")
        eski = Komsu(node_id="olu-dugum", ad="olu-makine",
                     ip="10.9.9.9", port=1234, caps=["code"],
                     ts=time.time() - (havuz._ttl + 60))
        _p(f"  TTL {havuz._ttl} sn · sahte kayıt yaşı "
           f"{eski.yas:.0f} sn → canlı={eski.canli} "
           f"{'✓ düşürülür' if not eski.canli else '✗ KABUL EDİLDİ'}")

    # ── 5) YEREL DEVRALMA ──────────────────────────────────────────────
    if args.dene:
        _p("\n── 5) YEREL DEVRALMA (komşu yok/yanıtsız) ──")

        def _yerel() -> dict:
            return {"hesap": "yerelde yapıldı", "deger": 42}

        r5 = havuz.dispatch({"tool": "web.search", "args": {"q": "dxgi"}},
                            yerel=_yerel, deneme=2)
        _p(f"  kaynak={r5.get('kaynak')} · ok={r5.get('ok')} · "
           f"denenen={r5.get('denenen')}")
        _p(f"  sonuç={r5.get('sonuc')}")
        _p(f"  {'✓ iş kaybolmadı, yerel üstlendi' if r5.get('ok') and r5.get('kaynak') == 'yerel' else '✗ beklenmedik'}")

        # ⚠ Ulaşılamayan sahte komşu ile de dene (gerçek ağ hatası)
        hayali = Komsu(node_id="hayali", ad="hayali",
                       ip="10.255.255.1", port=9, ts=time.time())
        t0 = time.perf_counter()
        y = havuz.gonder(hayali, {"tool": "x"})
        _p(f"  ulaşılamaz komşu → ok={y.get('ok')} · "
           f"{time.perf_counter() - t0:.1f} sn · hata={str(y.get('hata'))[:46]}")

    # ── 6) BIRAK ───────────────────────────────────────────────────────
    if args.birak and not args.dene:
        _p("\n── 6) BIRAK ──")
        r6 = havuz.birak()
        _p(f"  ok={r6.get('ok')}")

    # ── ÖZET ───────────────────────────────────────────────────────────
    if args.dene:
        _p("\n" + "=" * 74)
        st = havuz.get_stats()
        for k in ("kayit", "kesif", "gonderim", "komsu_basarili",
                  "yerel_devralma", "hata"):
            _p(f"  {k:18} {st.get(k)}")
        if st.get("son_hata"):
            _p(f"  son hata           {str(st['son_hata'])[:60]}")
        _p("=" * 74)
        havuz.birak()

    havuz.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
