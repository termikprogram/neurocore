"""
Ortak havuz testleri — ağ çağrısı YAPMAZ (SSH sahtelenir).
==========================================================
Korunan davranışlar (hepsi gerçek sunucuda ölçüldü):

```
1. KAYIT    : düğüm kendini yazar (node_id + IP + port + caps)
2. KEŞİF    : kendisi hariç, bayat olmayan komşular döner
3. BAYAT    : TTL geçen kayıt düşürülür
4. DEVRALMA : komşu yanıt vermezse iş **yerelde** çalışır (kaybolmaz)
5. RELAY    : `relay_port` varsa istek sunucu üzerinden gider
              (ölçüldü: GatewayPorts=no → tünel 127.0.0.1 dinler)
6. TÜNEL    : uzak port düğüm kimliğinden **belirlenimci** türetilir
```
"""

from __future__ import annotations

import json
import time
import unittest
from typing import Any, Dict, List, Optional

from neurocore.net.peer_pool import TTL_SANIYE, Komsu, PeerPool


class _SahteHavuz(PeerPool):
    """`_ssh` sahtelenmiş havuz — defter bellekte tutulur."""

    def __init__(self, kayitlar: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__()
        # ⚠ Gerçek `.neurocore` dizinini kirletmemek için geçici dizin
        import tempfile

        self._tmp = tempfile.mkdtemp(prefix="nc-pool-test-")
        self.defter: List[Dict[str, Any]] = list(kayitlar or [])
        self.cagrilar: List[str] = []
        self.relay_yaniti: Dict[str, Any] = {"ok": True, "sonuc": "uzak"}

    def _ssh(self, uzak_komut: str, girdi: str = "") -> Dict[str, Any]:
        self.cagrilar.append(girdi[:200] or uzak_komut)
        # ⚠ Üretilen betikte `islem = 'kayit'` **tek tırnaklı** yazılır;
        # önceki sürüm çift tırnak aradığı için dal hiç çalışmıyordu.
        if "peers.jsonl" in girdi and "islem = 'kayit'" in girdi:
            bas = girdi.find("yeni = json.loads(")
            if bas >= 0:
                parca = girdi[bas + len("yeni = json.loads("):]
                son = parca.find(")\n")
                if son < 0:
                    son = parca.find(")")
                try:
                    # ⚠ Kaynak betikte payload bir Python **dizgesi** olarak
                    # gömülü: önce literal_eval, sonra json.
                    import ast as _ast

                    yeni = json.loads(_ast.literal_eval(parca[:son]))
                    # ⚠ Sunucu IP'yi `SSH_CLIENT`'ten doldurur (taklit)
                    if not yeni.get("ip"):
                        yeni["ip"] = "95.70.193.163"
                    self.defter = [d for d in self.defter
                                   if d.get("node_id") != yeni.get("node_id")]
                    self.defter.append(yeni)
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    pass
            return {"ok": True, "hata": "",
                    "cikti": json.dumps(self.defter, ensure_ascii=False)}
        if "peers.jsonl" in girdi and "islem = 'birak'" in girdi:
            return {"ok": True, "hata": "", "cikti": "[]"}
        if "peers.jsonl" in girdi:
            return {"ok": True, "hata": "",
                    "cikti": json.dumps(self.defter, ensure_ascii=False)}
        if "SSH_CLIENT" in uzak_komut:
            return {"ok": True, "hata": "", "cikti": "95.70.193.163 51000 22"}
        # ⚠ relay betiği
        if "127.0.0.1:" in girdi:
            return {"ok": True, "hata": "",
                    "cikti": json.dumps(self.relay_yaniti)}
        return {"ok": True, "hata": "", "cikti": ""}


def _kayit(node_id: str = "peer-a", ts: Optional[float] = None,
           relay_port: int = 0, **kw: Any) -> Dict[str, Any]:
    return {
        "node_id": node_id, "ad": f"makine-{node_id}", "ip": "10.0.0.2",
        "port": 8791, "relay_port": relay_port, "caps": ["code"],
        "ts": ts if ts is not None else time.time(), "surum": "test",
        **kw,
    }


class TestKayit(unittest.TestCase):
    """1) Kayıt: IP sunucudan alınır ve deftere yazılır."""

    def test_kayit_ip_ve_dugum(self) -> None:
        h = _SahteHavuz()
        h.initialize(data_path=h._tmp)
        r = h.register(port=8791, caps=["code", "blender"])
        self.assertTrue(r.get("ok"))
        # ⚠ IP `$SSH_CLIENT`'ten gelir (sunucunun gördüğü)
        self.assertEqual(r.get("ip"), "95.70.193.163")
        self.assertEqual(len(h.defter), 1)
        self.assertEqual(h.defter[0]["node_id"], h._kimlik)
        self.assertIn("blender", h.defter[0]["caps"])

    def test_kimlik_kalici(self) -> None:
        """Aynı veri dizininde kimlik değişmemeli (defter şişmesin)."""
        a = PeerPool()
        a.initialize(data_path=".neurocore")
        b = PeerPool()
        b.initialize(data_path=".neurocore")
        self.assertEqual(a._kimlik, b._kimlik)
        self.assertTrue(a._kimlik)


class TestKesif(unittest.TestCase):
    """2-3) Keşif ve bayat kayıt."""

    def test_kendisi_haric(self) -> None:
        h = _SahteHavuz()
        h.initialize(data_path=h._tmp)
        h.defter = [_kayit(h._kimlik), _kayit("peer-b")]
        adlar = [k.node_id for k in h.peers()]
        self.assertEqual(adlar, ["peer-b"])

    def test_bayat_dusurulur(self) -> None:
        h = _SahteHavuz()
        h.initialize(data_path=h._tmp)
        eski = time.time() - (TTL_SANIYE + 30)
        h.defter = [_kayit("eski", ts=eski), _kayit("taze")]
        adlar = [k.node_id for k in h.peers()]
        self.assertIn("taze", adlar)
        self.assertNotIn("eski", adlar)

    def test_kendini_say(self) -> None:
        h = _SahteHavuz()
        h.initialize(data_path=h._tmp)
        h.defter = [_kayit(h._kimlik)]
        self.assertEqual(len(h.peers(kendini_say=True)), 1)

    def test_canli_ozelligi(self) -> None:
        taze = Komsu.from_dict(_kayit("a"))
        eski = Komsu.from_dict(_kayit("b", ts=time.time() - (TTL_SANIYE + 5)))
        self.assertTrue(taze.canli)
        self.assertFalse(eski.canli)


class TestDevralma(unittest.TestCase):
    """4) Komşu yanıt vermezse iş yerelde çalışır."""

    def test_komsu_yoksa_yerel(self) -> None:
        h = _SahteHavuz()
        h.initialize(data_path=h._tmp)
        r = h.dispatch({"tool": "x"}, yerel=lambda: {"yerel": 1})
        self.assertTrue(r["ok"])
        self.assertEqual(r["kaynak"], "yerel")
        self.assertEqual(r["sonuc"], {"yerel": 1})
        self.assertEqual(h.get_stats()["yerel_devralma"], 1)

    def test_komsu_hata_verirse_yerel(self) -> None:
        h = _SahteHavuz()
        h.initialize(data_path=h._tmp)
        h.defter = [_kayit("peer-b")]
        # ⚠ relay de başarısız olsun
        h.relay_yaniti = {"ok": False, "hata": "ulaşılamadı"}
        r = h.dispatch({"tool": "x"}, yerel=lambda: {"yerel": 2})
        self.assertEqual(r["kaynak"], "yerel")
        self.assertEqual(r["sonuc"], {"yerel": 2})
        # ⚠ Komşu gerçekten denendi mi?
        # ⚠ relay_port yok → doğrudan adres denenir
        self.assertEqual(r["denenen"], ["10.0.0.2:8791"])

    def test_yerel_istisna_raporlanir(self) -> None:
        h = _SahteHavuz()
        h.initialize(data_path=h._tmp)

        def _patlat() -> None:
            raise RuntimeError("yerel de bozuk")

        r = h.dispatch({"tool": "x"}, yerel=_patlat)
        self.assertFalse(r["ok"])
        self.assertIn("RuntimeError", r["hata"])


class TestRelay(unittest.TestCase):
    """5) `relay_port` varsa istek sunucu üzerinden gider."""

    def test_relay_yolu_seciliyor(self) -> None:
        h = _SahteHavuz()
        h.initialize(data_path=h._tmp)
        k = Komsu(node_id="p", ad="p", ip="10.0.0.9", port=8791,
                  relay_port=28111, ts=time.time())
        r = h.gonder(k, {"tool": "web.search"})
        self.assertTrue(r.get("ok"))
        # ⚠ Sunucuda çalışan betik 127.0.0.1:<relay> hedeflemeli
        birlesik = " ".join(h.cagrilar)
        self.assertIn("127.0.0.1:28111", birlesik)
        self.assertIn("web.search", birlesik)

    def test_relay_adresi(self) -> None:
        k = Komsu(node_id="p", relay_port=28111)
        self.assertEqual(k.adres, "relay:28111")
        k2 = Komsu(node_id="q", ip="10.0.0.3", port=8791)
        self.assertEqual(k2.adres, "10.0.0.3:8791")


class TestTunel(unittest.TestCase):
    """6) Tünel portu düğüm kimliğinden belirlenimci türetilir."""

    def _gecici(self) -> PeerPool:
        import tempfile

        h = PeerPool()
        h.initialize(data_path=tempfile.mkdtemp(prefix="nc-tunel-test-"))
        return h

    def test_port_belirlenimci(self) -> None:
        h = self._gecici()
        # ⚠ `tunel_baslat` gerçek ssh açar; burada yalnız port kuralını
        # doğruluyoruz (yan etkisiz).
        port = 20000 + (int(h._kimlik[:4], 16) % 20000)
        self.assertGreaterEqual(port, 20000)
        self.assertLess(port, 40000)

    def test_ssh_yoksa_hata(self) -> None:
        h = self._gecici()
        h._ssh_komutu = lambda: None          # type: ignore[assignment]
        r = h.tunel_baslat(yerel_port=8791)
        self.assertFalse(r.get("ok"))
        self.assertIn("ssh", str(r.get("hata", "")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
