"""
Havuz yaşam döngüsü testleri — kayıt · tünel · heartbeat · gözetim
===================================================================
Ağ çağrısı yapılmaz: `_ssh` ve `tunel_baslat` sahtelenir, aralık
0,15 sn'ye düşürülür (gerçek 30 sn beklenmez).

Korunan davranışlar
-------------------
```
1. baslat()  : tünel açılır + kayıt yazılır + thread başlar
2. heartbeat : periyodik olarak kaydı tazeler (defter bayatlamaz)
3. gözetim   : tünel süreci ölürse **yeniden açılır**
4. durdur()  : thread durur, kayıt silinir, tünel kapatılır
5. hata      : SSH hatasında çökmez, sayacı artırır
```
"""

from __future__ import annotations

import json
import threading
import time
import unittest
from typing import Any, Dict, List, Optional

from neurocore.net.peer_pool import Komsu, PeerPool


class _SahteSurec:
    """`subprocess.Popen` yerine geçen asgari süreç."""

    def __init__(self) -> None:
        self.olu = False
        self.sonlandi = False
        self.olduruldu = False

    def poll(self) -> Optional[int]:
        return 0 if self.olu else None

    def terminate(self) -> None:
        self.sonlandi = True
        self.olu = True

    def wait(self, timeout: float = 0) -> int:
        return 0

    def kill(self) -> None:
        self.olduruldu = True
        self.olu = True


class _SahteHavuz(PeerPool):
    """SSH ve tünel sahtelenmiş havuz."""

    def __init__(self, aralik: float = 0.15) -> None:
        super().__init__(aralik=aralik)
        # ⚠ Testler GERÇEK `.neurocore` dizinini kirletmemeli: sahte SSH
        # "sunucu token'ı" döndürüyor ve `_token_yaz` onu diske yazıyordu —
        # bu, gerçek küme anahtarını sahte değerle **eziyordu** (ölçüldü).
        import tempfile

        self._tmp = tempfile.mkdtemp(prefix="nc-peer-test-")
        # ⚠ Kaç ssh çağrısı yapıldığını sayar (pencere kırpışması testi)
        self._ssh_cagrilari: List[str] = []
        self.defter: List[Dict[str, Any]] = []
        self.kayit_sayisi = 0
        self.silme_sayisi = 0
        self.surecler: List[_SahteSurec] = []
        self.tunel_cagri = 0
        self.ssh_hatasi = False

    # ── SSH ──
    def _ssh(self, uzak_komut: str, girdi: str = "") -> Dict[str, Any]:
        self._ssh_cagrilari.append((girdi or uzak_komut)[:40])
        if self.ssh_hatasi:
            return {"ok": False, "hata": "ssh koptu", "cikti": ""}
        # ⚠ Küme anahtarı artık sunucudan alınır (`~/neurocore/cluster.token`)
        if "cluster.token" in girdi:
            return {"ok": True, "hata": "", "cikti": "sunucu-anahtari-abc123"}
        if "SSH_CLIENT" in uzak_komut:
            return {"ok": True, "hata": "", "cikti": "95.70.193.163 1 22"}
        if "islem = 'kayit'" in girdi:
            self.kayit_sayisi += 1
            bas = girdi.find("yeni = json.loads(")
            if bas >= 0:
                parca = girdi[bas + len("yeni = json.loads("):]
                son = parca.find(")\n")
                if son < 0:
                    son = parca.find(")")
                try:
                    import ast as _ast

                    yeni = json.loads(_ast.literal_eval(parca[:son]))
                    if not yeni.get("ip"):
                        yeni["ip"] = "95.70.193.163"
                    self.defter = [d for d in self.defter
                                   if d.get("node_id") != yeni.get("node_id")]
                    self.defter.append(yeni)
                except Exception:
                    pass
            return {"ok": True, "hata": "",
                    "cikti": json.dumps(self.defter, ensure_ascii=False)}
        if "islem = 'birak'" in girdi:
            self.silme_sayisi += 1
            return {"ok": True, "hata": "", "cikti": "[]"}
        if "peers.jsonl" in girdi:
            return {"ok": True, "hata": "",
                    "cikti": json.dumps(self.defter, ensure_ascii=False)}
        return {"ok": True, "hata": "", "cikti": ""}

    # ── Tünel ──
    def tunel_baslat(self, *, yerel_port: int, uzak_port: int = 0,
                     zaman_asimi: float = 20.0) -> Dict[str, Any]:
        self.tunel_cagri += 1
        s = _SahteSurec()
        self.surecler.append(s)
        return {"ok": True, "uzak_port": uzak_port or 25000,
                "surec": s, "hata": ""}


def _init(h: Any) -> None:
    """Havuzu **geçici** veri diziniyle başlatır (gerçek dizin korunur)."""
    h.initialize(data_path=h._tmp)


class TestBaslat(unittest.TestCase):
    """1) Açılış: tünel + kayıt + thread."""

    def test_baslat_kayit_ve_tunel(self) -> None:
        h = _SahteHavuz()
        _init(h)
        try:
            r = h.baslat(port=8791, caps=["code", "blender"])
            self.assertTrue(r.get("ok"))
            self.assertEqual(r.get("tunel"), 25000)
            # ⚠ Heartbeat ilk turda da kayıt atar → en az 1 beklenir
            self.assertGreaterEqual(h.kayit_sayisi, 1, "kayıt yapılmadı")
            self.assertEqual(h.tunel_cagri, 1)
            # ⚠ Kayıt satırında relay_port olmalı (NAT aşımı için)
            self.assertEqual(h.defter[0]["relay_port"], 25000)
            self.assertIn("blender", h.defter[0]["caps"])
            self.assertTrue(h._hb_thread is not None and h._hb_thread.is_alive())
        finally:
            h.durdur()
        self.assertFalse(h._calisiyor)


class TestHeartbeat(unittest.TestCase):
    """2) Periyodik kayıt tazeleme."""

    def test_kayit_tazelenir(self) -> None:
        h = _SahteHavuz(aralik=0.12)
        _init(h)
        try:
            h.baslat(port=8791, caps=["code"], tunel=False)
            ilk = h.kayit_sayisi
            time.sleep(0.45)
            self.assertGreater(h.kayit_sayisi, ilk,
                               "heartbeat kaydı tazelemedi")
            self.assertGreaterEqual(h.get_stats()["heartbeat"], 1)
        finally:
            h.durdur()


class TestTunelGozetimi(unittest.TestCase):
    """3) Tünel süreci ölürse yeniden açılır."""

    def test_olen_tunel_yeniden_acilir(self) -> None:
        h = _SahteHavuz(aralik=0.1)
        _init(h)
        try:
            h.baslat(port=8791, caps=["code"])
            self.assertEqual(h.tunel_cagri, 1)
            # ⚠ Tüneli "öldür" → heartbeat fark edip yeniden açmalı
            h.surecler[0].olu = True
            time.sleep(0.5)
            self.assertGreaterEqual(h.tunel_cagri, 2,
                                    "ölü tünel yeniden açılmadı")
            self.assertGreaterEqual(h.get_stats()["tunel_yeniden"], 1)
            self.assertGreaterEqual(len(h.surecler), 2)
        finally:
            h.durdur()


class TestDurdur(unittest.TestCase):
    """4) Kapanış: thread durur, kayıt silinir, tünel kapanır."""

    def test_durdur_temiz(self) -> None:
        h = _SahteHavuz(aralik=0.1)
        _init(h)
        r = h.baslat(port=8791, caps=["code"])
        self.assertTrue(r.get("ok"))
        surec = h.surecler[0]
        d = h.durdur()
        self.assertTrue(d.get("ok"))
        self.assertFalse(h._calisiyor)
        self.assertGreaterEqual(h.silme_sayisi, 1, "kayıt silinmedi")
        self.assertTrue(surec.sonlandi, "tünel kapatılmadı")
        # ⚠ Thread gerçekten durdu mu?
        th = h._hb_thread
        self.assertTrue(th is None or not th.is_alive())

    def test_sutdown_durdurur(self) -> None:
        h = _SahteHavuz(aralik=0.1)
        _init(h)
        h.baslat(port=8791, caps=["code"], tunel=False)
        h.shutdown()
        self.assertFalse(h._initialized)
        self.assertFalse(h._calisiyor)


class TestPencereVeCagriSayisi(unittest.TestCase):
    """
    ⚠ Kullanıcı bildirimi: "arada bir cmd ssh.exe çalışıp kapanıyor".

    İki koruma:
      1. Alt süreçler **gizli** başlatılır (pythonw altında pencere açmaz)
      2. Heartbeat'te **tek** ssh çağrısı yapılır (eskiden 2'ydi)
    """

    def test_pencere_gizleme_bayraklari(self) -> None:
        import os
        import subprocess

        from neurocore.net.peer_pool import _pencere_gizle

        p = _pencere_gizle()
        if os.name == "nt":
            self.assertIn("creationflags", p)
            self.assertEqual(p["creationflags"], subprocess.CREATE_NO_WINDOW)
            self.assertIn("startupinfo", p)
            self.assertTrue(p["startupinfo"].dwFlags
                            & subprocess.STARTF_USESHOWWINDOW)
        else:
            self.assertEqual(p, {})

    def test_kayit_tek_ssh_cagrisi(self) -> None:
        """⚠ `register()` tek ssh çağrısı yapmalı (IP sunucudan gelir)."""
        h = _SahteHavuz()
        _init(h)
        h._ssh_cagrilari.clear()
        r = h.register(port=8791, caps=["code"])
        self.assertTrue(r.get("ok"))
        # ⚠ IP sunucunun doldurduğu satırdan gelmeli
        self.assertEqual(r.get("ip"), "95.70.193.163")
        # ⚠ Kaç ssh çağrısı yapıldı?
        self.assertEqual(len(h._ssh_cagrilari), 1,
                         f"beklenen 1, görülen {len(h._ssh_cagrilari)}")

    def test_heartbeat_basina_tek_cagri(self) -> None:
        h = _SahteHavuz(aralik=0.1)
        _init(h)
        try:
            h.baslat(port=8791, caps=["code"], tunel=False)
            h._ssh_cagrilari.clear()
            time.sleep(0.32)
            # ⚠ aralık 0,1 → ~3 tur; tur başına 1 çağrı (+ küçük sapma)
            self.assertLessEqual(len(h._ssh_cagrilari), 5,
                                 f"çok fazla ssh: {len(h._ssh_cagrilari)}")
            self.assertGreaterEqual(len(h._ssh_cagrilari), 1)
        finally:
            h.durdur()


class TestHataToleransi(unittest.TestCase):
    """5) SSH hatasında çökmez, sayacı artırır."""

    def test_ssh_hatasi_cokertmez(self) -> None:
        h = _SahteHavuz(aralik=0.1)
        _init(h)
        h.ssh_hatasi = True
        try:
            r = h.baslat(port=8791, caps=["code"], tunel=False)
            self.assertFalse(r.get("ok"), "kayıt başarısız bildirilmeli")
            time.sleep(0.35)
            # ⚠ Thread yaşıyor ve hata sayacı artmış olmalı
            self.assertTrue(h._hb_thread is not None)
            self.assertGreaterEqual(h.get_stats()["hata"], 1)
        finally:
            h.ssh_hatasi = False
            h.durdur()


if __name__ == "__main__":
    unittest.main(verbosity=2)
