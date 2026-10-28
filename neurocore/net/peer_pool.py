"""
Ortak Havuz — makineler arası keşif ve iş paylaşımı
====================================================
⚠ ÖLÇÜLEN GERÇEK (bu makinede)
------------------------------
```
yerel adres   : 192.168.1.199/24      (router arkasında)
genel IP      : 95.70.193.163         (router çıkışı)
SSH sunucusu  : ör. `ubuntu@<sunucu>`  (adres **kodda tutulmaz**)
                `$SSH_CLIENT` → genel IP            ← sunucunun gördüğü
`who`         : BOŞ (betik oturumları tty açmaz) → IP'yi `who`'dan ALAMAYIZ
```

Yani:
```
1. IP'yi sunucu **kendisi** söyler ($SSH_CLIENT) → kayıt defterine yazarız.
2. Ama 95.70.193.163 bir **NAT** adresidir; dışarıdan doğrudan bağlantı
   router'a düşer. Bu yüzden her düğüm sunucuya **ters SSH tüneli**
   (ssh -R) açar; sunucu üzerinden erişilir. Router ayarı gerekmez.
```

Tasarım
-------
```
     düğüm A ──ssh──▶  sunucu  ◀──ssh──  düğüm B
        │                 │                 │
        │   ~/neurocore/peers.jsonl         │
        │   (node_id, ip, port, ts)         │
        └─────── sunucu üzerinden ──────────┘
                 (ters tünel / relay)
```

* **Kayıt** : her düğüm kendi satırını yazar (IP + port + yetenek + ts)
* **Keşif** : defteri oku → bayat satırları at → komşu listesi
* **İş**    : sırayla komşuya gönder; **yanıt gelmezse yerel üstlenir**
* **Kütüphane**: kayıt satırındaki `caps` ile hangi düğüm ne yapabiliyor

Sınırlar (dürüst)
-----------------
```
· Doğrudan (P2P) bağlantı NAT yüzünden ÇALIŞMAZ; ters tünel gerekir.
· Tünel tarayıcı/SSH süreci kadar yaşar; koparsa heartbeat durur,
  kayıt TTL ile bayatlar ve komşular onu düşürür.
· Paylaşılan "kütüphane" bu modülde yalnız **meta**dır (not sayısı,
  yetenekler). İçerik paylaşımı ayrı iş (vardiya senkronizasyonu).
```

Kullanım
--------
    from neurocore.net.peer_pool import PeerPool
    havuz = PeerPool()
    havuz.initialize(config=None)
    kayit = havuz.register(port=8791, caps=["code", "blender"])
    komsular = havuz.peers()
    sonuc = havuz.dispatch({"tool": "web.search", "args": {...}},
                           yerel=lambda: {"ok": True})
"""

from __future__ import annotations

import base64
import json
import logging
import os
import platform
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("net.peer_pool")

#: ⚠ Gömülü **sır yok**. Sunucu ve anahtar yolu şuradan gelir:
#:   1. `NC_SERVER` / `NC_SSH_KEY` ortam değişkenleri
#:   2. `.neurocore/peer.json`  {"server": "user@host", "ssh_key": "..."}
#: Depoya giren kodda adres/anahtar **bulunmaz** (git'e sızmaz).
VARSAYILAN_ANAHTAR = ""
VARSAYILAN_SUNUCU = ""

#: Yerel küme ayar dosyası (git'e girmez)
AYAR_DOSYASI = "peer.json"


def kume_ayarlari(data_path: str = ".neurocore") -> Dict[str, str]:
    """
    Küme bağlantı ayarlarını okur (ortam → yerel dosya).

    Returns:
        {"server": "user@host", "ssh_key": "C:\\...\\key"}
        Eksik alanlar boş dizge olur.
    """
    import os

    sonuc = {
        "server": (os.environ.get("NC_SERVER") or "").strip(),
        "ssh_key": (os.environ.get("NC_SSH_KEY") or "").strip(),
    }
    adaylar = [Path(data_path) / AYAR_DOSYASI,
               Path(".neurocore") / AYAR_DOSYASI,
               Path(__file__).resolve().parents[2] / ".neurocore" / AYAR_DOSYASI]
    for yol in adaylar:
        try:
            if not yol.exists():
                continue
            veri = json.loads(yol.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not sonuc["server"]:
            sonuc["server"] = str(veri.get("server") or "").strip()
        if not sonuc["ssh_key"]:
            sonuc["ssh_key"] = str(veri.get("ssh_key") or "").strip()
        if sonuc["server"] and sonuc["ssh_key"]:
            break
    return sonuc

#: Sunucudaki kayıt defteri (göreli → $HOME altında)
DEFTER = "neurocore/peers.jsonl"

#: ⚠ Kayıt TTL: bu süreden eski satırlar bayat sayılır
TTL_SANIYE = 90.0

#: SSH komut zaman aşımı
SSH_TIMEOUT = 25.0

#: Yerel düğüm kimliği dosyası
KIMLIK_DOSYASI = ".neurocore/node_id"

#: ⚠ Uzaktan (komşu düğümden) çalıştırılabilir araçlar — **yalnız okuma**.
#:
#: Tek doğru kaynak burasıdır: sunucu (`/peer/job`) ve ajan döngüsü
#: (`AgentLoop._execute`) aynı listeyi kullanır. Aksi hâlde bir tarafta
#: izin verilen araç diğer tarafta reddedilir.
IZINLI_ARACLAR: Tuple[str, ...] = (
    "web.search", "web.fetch", "knowledge.ask", "learn.status",
    "code.workspace_map", "code.read_file", "code.list_directory",
    "code.search_text", "code.find_files",
    "doc.kutuphane_ara", "doc.api_imzasi_al", "doc.surum_karsilastir",
    "doc.godot_api_baglam",
    "asset.katalog", "asset.onizleme", "asset.anim_onizleme",
)


def yerel_token(data_path: str = "") -> str:
    """
    **Sunucuya gitmeden** okunabilen küme anahtarı.

    Sıra: `NC_PEER_TOKEN` → `.neurocore/peer_token`.
    Yoksa boş dizge döner (o zaman sunucudan alınır — bkz. `PeerPool.token`).

    ⚠ Neden ayrı? `/peer/job` ucu **her istekte** doğrulama yapar; orada
    SSH çağrısı yapılamaz (yavaş + SSH düşünce uç ölür). Bu yüzden token
    bir kez alınır, yerelde saklanır, HTTP yolu yalnız dosyayı okur.
    """
    import os

    ortam = (os.environ.get("NC_PEER_TOKEN") or "").strip()
    if ortam:
        return ortam

    # ⚠ İki olası konum: verilen dizin ve proje kökü. Uygulama farklı bir
    # çalışma diziniyle açılırsa token bulunamaz ve `/peer/job` her isteği
    # reddeder — bu yüzden ikisi de denenir.
    adaylar = []
    if data_path:
        adaylar.append(Path(data_path) / "peer_token")
    adaylar.append(Path(".neurocore/peer_token"))
    adaylar.append(Path(__file__).resolve().parents[2] / ".neurocore"
                       / "peer_token")
    for yol in adaylar:
        try:
            if yol.exists():
                deger = yol.read_text(encoding="utf-8").strip()
                if deger:
                    return deger
        except OSError:
            continue
    return ""


def ortak_token(anahtar: str = "") -> str:
    """
    Geriye dönük ad: yerel token.

    ⚠ **ARTIK SSH ANAHTARINDAN TÜRETİLMEZ.** Gerekçe (kullanıcı kuralı):
    SSH anahtarı **makineye özeldir**, istemcilerle paylaşılmaz. Anahtar
    paylaşılmadığı sürece anahtardan türetilen token makinelerde farklı
    olur ve küme kimlik doğrulaması **hiç eşleşmez**.

    Yeni akış:
        1. `NC_PEER_TOKEN` (elle verilmiş)
        2. `.neurocore/peer_token` (sunucudan bir kez alınmış, yerelde saklı)
        3. yoksa `PeerPool.token()` **sunucudan çeker** (bkz. orada)
    """
    return yerel_token()


def _token_dosyasi(data_path: str = ".neurocore") -> Path:
    """Yerel token dosyasının yolu."""
    return Path(data_path) / "peer_token"


def _token_yaz(token: str, data_path: str = ".neurocore") -> bool:
    """Token'ı yerelde saklar (yalnız kullanıcı okuyabilir)."""
    if not token:
        return False
    yol = _token_dosyasi(data_path)
    try:
        yol.parent.mkdir(parents=True, exist_ok=True)
        yol.write_text(token, encoding="utf-8")
        # ⚠ Windows'ta ACL, POSIX'te mod 600: başka kullanıcı okumasın
        try:
            if os.name == "nt":
                import subprocess as _sp

                kullanici = os.environ.get("USERNAME", "")
                if kullanici:
                    _sp.run(["icacls", str(yol), "/inheritance:r",
                             "/grant:r", f"{kullanici}:F"],
                            capture_output=True, timeout=15)
            else:
                yol.chmod(0o600)
        except Exception:
            pass
        return True
    except OSError:
        return False


@dataclass
class Komsu:
    """Havuzdaki tek düğüm."""
    node_id: str = ""
    ad: str = ""
    ip: str = ""
    port: int = 0
    relay_port: int = 0
    caps: List[str] = field(default_factory=list)
    ts: float = 0.0
    surum: str = ""

    @property
    def yas(self) -> float:
        """Kaç saniye önce kaydedildi."""
        return max(0.0, time.time() - float(self.ts or 0))

    @property
    def canli(self) -> bool:
        return self.yas <= TTL_SANIYE

    @property
    def adres(self) -> str:
        """Erişim adresi: ters tünel varsa sunucu üzerinden."""
        if self.relay_port:
            return f"relay:{self.relay_port}"
        return f"{self.ip}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id, "ad": self.ad, "ip": self.ip,
            "port": self.port, "relay_port": self.relay_port,
            "caps": list(self.caps), "ts": self.ts, "surum": self.surum,
            "yas": round(self.yas, 1), "canli": self.canli,
            "adres": self.adres,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Komsu":
        return cls(
            node_id=str(d.get("node_id") or ""),
            ad=str(d.get("ad") or ""),
            ip=str(d.get("ip") or ""),
            port=int(d.get("port") or 0),
            relay_port=int(d.get("relay_port") or 0),
            caps=[str(c) for c in (d.get("caps") or [])],
            ts=float(d.get("ts") or 0.0),
            surum=str(d.get("surum") or ""),
        )


class PeerPool:
    """
    SSH tabanlı ortak havuz: kayıt · keşif · iş paylaşımı (yedekli).

    ⚠ Bu sınıf **ağ çağrısı** yapar (SSH). Testlerde `_ssh` sahtelenir.
    """

    name = "net.peer_pool"

    #: ⚠ Heartbeat aralığı: kayıt TTL'i (90 sn) 3 katı → güvenli pay
    HEARTBEAT_SANIYE = 30.0

    def __init__(self, *, anahtar: str = VARSAYILAN_ANAHTAR,
                 sunucu: str = VARSAYILAN_SUNUCU,
                 ttl: float = TTL_SANIYE,
                 aralik: float = HEARTBEAT_SANIYE) -> None:
        # ⚠ Başka makinede anahtar farklı yerde olabilir: `NC_SSH_KEY`
        # ortam değişkeni açıkça verilen yolu **ezer** (kurulum betiği
        # bunu kullanır). Sessizce yanlış anahtarla bağlanmak yerine
        # tek doğru yol bırakılır.
        import os as _os

        # ⚠ Sır kodda değil: ortam → yerel `peer.json`
        _ayar = kume_ayarlari()
        self._anahtar = ((_os.environ.get("NC_SSH_KEY") or "").strip()
                         or anahtar or _ayar.get("ssh_key", ""))
        self._sunucu = (sunucu or _ayar.get("server", ""))
        self._ttl = float(ttl)
        self._aralik = float(aralik)
        self._initialized = False
        self._kimlik: str = ""
        self._data_path = Path(".neurocore")
        self._counter: Dict[str, int] = {
            "kayit": 0, "kesif": 0, "gonderim": 0,
            "komsu_basarili": 0, "yerel_devralma": 0, "hata": 0,
            # ⚠ yaşam döngüsü sayaçları
            "heartbeat": 0, "tunel_acildi": 0, "tunel_yeniden": 0,
        }
        self._son_hata = ""
        # ── Yaşam döngüsü durumu ──
        self._calisiyor = False
        self._hb_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._port = 0
        self._caps: List[str] = []
        self._tunel: Any = None
        self._tunel_port = 0
        self._son_kayit = 0.0
        # ⚠ Küme anahtarı önbelleği (sunucuya her istekte gitmemek için)
        self._token = ""
        self._token_denendi = False
        self._token_deneme_ts = 0.0
        self._ayar_uyarisi = False

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None, *,
                   data_path: str = "") -> None:
        """Havuzu başlatır (idempotent). Kimlik kalıcıdır."""
        if self._initialized:
            return
        if data_path:
            self._data_path = Path(data_path)
        try:
            self._data_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._data_path = Path(".")
        self._kimlik = self._kimlik_yukle()
        self._initialized = True
        logger.info(f"Ortak havuz hazır — düğüm: {self._kimlik}")

    # ─── Yaşam Döngüsü: kayıt + tünel + heartbeat ──────────────────────

    def baslat(self, *, port: int, caps: Sequence[str] = (),
               tunel: bool = True, uzak_port: int = 0) -> Dict[str, Any]:
        """
        Havuzu **otomatik** çalıştırır: tünel → kayıt → heartbeat.

        Args:
            port: Yerel HTTP API portu (`/peer/job` ucu)
            caps: Bu düğümün yetenekleri
            tunel: Ters SSH tüneli açılsın mı
            uzak_port: Sunucudaki port (0 = kimlikten türetilir)

        Returns:
            {"ok", "tunel", "kayit", "hata"}
        """
        if not self._initialized:
            return {"ok": False, "hata": "havuz başlatılmadı"}
        self._port = int(port)
        self._caps = [str(c) for c in caps]

        # ⚠ Önce küme anahtarı: `/peer/job` doğrulaması bunu okur.
        # (Sunucudan bir kez alınır, yerelde saklanır.)
        self.token()

        # ── 1) Ters tünel (NAT aşımı) ──
        sonuc: Dict[str, Any] = {"ok": True, "tunel": None, "kayit": None,
                                 "hata": ""}
        if tunel:
            t = self.tunel_baslat(yerel_port=self._port, uzak_port=uzak_port)
            if t.get("ok"):
                self._tunel = t.get("surec")
                self._tunel_port = int(t.get("uzak_port") or 0)
                self._counter["tunel_acildi"] += 1
                sonuc["tunel"] = self._tunel_port
            else:
                sonuc["hata"] = f"tünel: {t.get('hata')}"
                self._son_hata = sonuc["hata"]
                logger.warning(f"Ters tünel açılamadı: {t.get('hata')}")

        # ── 2) Kayıt ──
        k = self.register(port=self._port, caps=self._caps,
                          relay_port=self._tunel_port)
        sonuc["kayit"] = k
        if not k.get("ok"):
            sonuc["ok"] = False
            sonuc["hata"] = sonuc["hata"] or f"kayıt: {k.get('hata')}"

        # ── 3) Heartbeat ──
        self._calisiyor = True
        self._stop.clear()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_dongu, name="nc-peer-heartbeat",
            daemon=True)
        self._hb_thread.start()
        logger.info(
            f"Havuz başladı — port {self._port} · tünel "
            f"{self._tunel_port or 'yok'} · aralık {self._aralik} sn")
        return sonuc

    def _heartbeat_dongu(self) -> None:
        """
        Her `aralik` saniyede: tüneli denetle → gerekirse yeniden aç →
        kaydı tazele.

        ⚠ Tünel süreci ölürse (SSH koptu) kayıt TTL ile bayatlar ve
        komşular bizi düşürür. Bu yüzden tünel **gözetlenir** ve
        yeniden açılır — aksi hâlde düğüm havuzda "ölü" görünür.
        """
        while self._calisiyor and not self._stop.is_set():
            try:
                # ── Tünel gözetimi ──
                if self._tunel is not None:
                    olu = False
                    try:
                        olu = self._tunel.poll() is not None
                    except Exception:
                        olu = True
                    if olu:
                        t = self.tunel_baslat(yerel_port=self._port,
                                              uzak_port=self._tunel_port)
                        if t.get("ok"):
                            self._tunel = t.get("surec")
                            self._tunel_port = int(
                                t.get("uzak_port") or self._tunel_port)
                            self._counter["tunel_yeniden"] += 1
                            logger.info("Ters tünel yeniden açıldı")
                        else:
                            self._son_hata = f"tünel: {t.get('hata')}"
                            logger.warning(
                                f"Tünel yeniden açılamadı: {t.get('hata')}")
                # ── Kayıt tazeleme ──
                self.register(port=self._port, caps=self._caps,
                              relay_port=self._tunel_port)
                self._counter["heartbeat"] += 1
            except Exception as exc:
                self._counter["hata"] += 1
                self._son_hata = f"{type(exc).__name__}: {exc}"
                logger.debug(f"Heartbeat hatası: {exc}")
            # ⚠ Alt sınır 0,05 sn: testlerde/kısa aralıkta
            # bekleme aralığı **ezilmemeli** (önce 1,0 sn idi →
            # heartbeat hiç çalışmıyor gibi görünüyordu).
            self._stop.wait(max(0.05, self._aralik))

    def durdur(self) -> Dict[str, Any]:
        """Heartbeat'i durdurur, kaydı siler, tüneli kapatır."""
        self._calisiyor = False
        self._stop.set()
        th = self._hb_thread
        if th is not None and th.is_alive():
            th.join(timeout=3.0)
        self._hb_thread = None
        birak = self.birak() if self._initialized else {"ok": False}
        tunel_kapandi = False
        if self._tunel is not None:
            try:
                self._tunel.terminate()
                self._tunel.wait(timeout=5)
            except Exception:
                try:
                    self._tunel.kill()
                except Exception:
                    pass
            self._tunel = None
            tunel_kapandi = True
        logger.info("Havuz durduruldu")
        return {"ok": True, "birak": birak, "tunel_kapandi": tunel_kapandi}

    def shutdown(self) -> None:
        """Kaynakları bırakır."""
        if self._calisiyor:
            try:
                self.durdur()
            except Exception:
                pass
        self._initialized = False

    def token(self, *, zorla: bool = False) -> str:
        """
        Küme anahtarını döndürür (gerekirse **sunucudan bir kez** alır).

        Sıra:
            1. `NC_PEER_TOKEN` / `.neurocore/peer_token`  (yerel)
            2. sunucudaki `~/neurocore/cluster.token`      (yoksa oluşturulur)
            3. alınan anahtar **yerelde saklanır** → `/peer/job` SSH'sız doğrular

        ⚠ SSH anahtarından **türetilmez**: anahtar makineye özeldir,
        istemcilerle paylaşılmaz. Küme sırrı sunucuda tek yerde durur.

        Args:
            zorla: yerel kopyayı yok say, sunucudan yeniden al
        """
        if self._token and not zorla:
            return self._token
        if not zorla:
            yerel = yerel_token(str(self._data_path))
            if yerel:
                self._token = yerel
                return self._token
        # ⚠ Sunucuya her istekte gitmemek için 60 sn bekleme
        simdi = time.time()
        if (not zorla and self._token_denendi
                and (simdi - self._token_deneme_ts) < 60.0):
            return self._token
        self._token_denendi = True
        self._token_deneme_ts = simdi
        sunucu = self._sunucudan_token()
        if sunucu:
            self._token = sunucu
            _token_yaz(sunucu, str(self._data_path))
            logger.info("Küme anahtarı sunucudan alındı ve yerelde saklandı")
        elif not self._token:
            logger.warning(
                "Küme anahtarı alınamadı — uzak istekler yetkisiz sayılacak")
        return self._token

    def _sunucudan_token(self) -> str:
        """
        Sunucudaki küme anahtarını okur; yoksa **oluşturur** (chmod 600).

        ⚠ Neden sunucuda? Her istemci kendi SSH anahtarıyla bağlanır ama
        **aynı** kümeye katılır; sır tek yerde durur ve istemciye
        kopyalanmaz. Sunucuya erişimi olan = kümeye üye.
        """
        betik = (
            "import os, secrets\n"
            "yol = os.path.expanduser('~/neurocore/cluster.token')\n"
            "if not os.path.exists(yol):\n"
            "    os.makedirs(os.path.dirname(yol), exist_ok=True)\n"
            "    with open(yol, 'w', encoding='utf-8') as f:\n"
            "        f.write(secrets.token_hex(16))\n"
            "    os.chmod(yol, 0o600)\n"
            "print(open(yol, encoding='utf-8').read().strip())\n"
        )
        r = self._ssh("python3 -", betik)
        if not r.get("ok"):
            self._son_hata = str(r.get("hata") or "token alınamadı")[:200]
            return ""
        return (r.get("cikti") or "").strip().splitlines()[-1].strip() \
            if (r.get("cikti") or "").strip() else ""

    def get_stats(self) -> Dict[str, Any]:
        """Sayaçlar."""
        return {
            **self._counter,
            "initialized": self._initialized,
            "node_id": self._kimlik,
            "sunucu": self._sunucu,
            "ttl": self._ttl,
            "son_hata": self._son_hata,
            "calisiyor": self._calisiyor,
            "port": self._port,
            "tunel_aktif": bool(self._tunel is not None
                                and getattr(self._tunel, "poll", lambda: 0)() is None),
            "tunel_port": self._tunel_port,
            "aralik": self._aralik,
            "son_kayit_yas": round(max(0.0, time.time() - self._son_kayit), 1),
        }

    # ─── Kimlik ─────────────────────────────────────────────────────────

    def _kimlik_yukle(self) -> str:
        """
        Kalıcı düğüm kimliği.

        ⚠ Neden kalıcı? Havuzda aynı makine her açılışta yeni kimlik
        alırsa defter şişer ve komşular onu **yeni düğüm** sanar.
        """
        yol = self._data_path / "node_id"
        try:
            if yol.exists():
                mevcut = yol.read_text(encoding="utf-8").strip()
                if mevcut:
                    return mevcut
        except OSError:
            pass
        # ⚠ Deterministik tohum: makine adı + platform (uuid yalnız yedek)
        tohum = f"{platform.node()}|{platform.machine()}|{uuid.getnode()}"
        yeni = uuid.uuid5(uuid.NAMESPACE_DNS, tohum).hex[:16]
        try:
            yol.write_text(yeni, encoding="utf-8")
        except OSError:
            pass
        return yeni

    # ─── SSH ────────────────────────────────────────────────────────────

    def _ssh_komutu(self) -> Optional[List[str]]:
        """SSH komut satırı (istemci/ayar yoksa None)."""
        ssh = shutil.which("ssh")
        if not ssh:
            return None
        if not self._sunucu or not self._anahtar:
            # ⚠ Sessiz başarısızlık yerine tek satır açıklama
            if not self._ayar_uyarisi:
                self._ayar_uyarisi = True
                logger.warning(
                    "Ortak havuz ayarsız: NC_SERVER/NC_SSH_KEY veya "
                    ".neurocore/peer.json gerekli — havuz devre dışı")
            return None
        return [
            ssh, "-i", self._anahtar,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "IdentitiesOnly=yes",
            "-o", f"ConnectTimeout={int(min(15, SSH_TIMEOUT))}",
            self._sunucu,
        ]

    def _ssh(self, uzak_komut: str,
             girdi: str = "") -> Dict[str, Any]:
        """
        Sunucuda komut çalıştırır.

        ⚠ `python3 -` + stdin kullanılır: tırnak/kaçış sorunu olmaz
        (Windows PowerShell → SSH → bash zincirinde tırnaklar bozuluyordu).
        """
        temel = self._ssh_komutu()
        if temel is None:
            return {"ok": False, "hata": "ssh istemcisi yok", "cikti": ""}
        try:
            p = subprocess.run(
                temel + [uzak_komut], input=girdi, text=True,
                capture_output=True, timeout=SSH_TIMEOUT,
                encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            self._counter["hata"] += 1
            self._son_hata = "ssh zaman aşımı"
            return {"ok": False, "hata": "zaman aşımı", "cikti": ""}
        except OSError as exc:
            self._counter["hata"] += 1
            self._son_hata = f"{type(exc).__name__}: {exc}"
            return {"ok": False, "hata": self._son_hata, "cikti": ""}
        if p.returncode != 0:
            self._counter["hata"] += 1
            self._son_hata = (p.stderr or p.stdout or "")[:200]
            return {"ok": False, "hata": self._son_hata,
                    "cikti": p.stdout or ""}
        return {"ok": True, "hata": "", "cikti": p.stdout or ""}

    def genel_ip(self) -> str:
        """
        Sunucunun **gördüğü** IP (`$SSH_CLIENT`).

        ⚠ `who` kullanılamaz: betik oturumları tty açmaz, `who` boş döner
        (ölçüldü). Bu yüzden IP'yi sunucu bize doğrudan söyler.
        """
        r = self._ssh('printf "%s" "$SSH_CLIENT"')
        if not r.get("ok"):
            return ""
        return (r.get("cikti") or "").split()[0].strip() if r.get("cikti") else ""

    # ─── Kayıt Defteri ──────────────────────────────────────────────────

    def _defter_betik(self, islem: str, kayit: Optional[Dict[str, Any]] = None
                      ) -> str:
        """
        Sunucuda çalışacak Python betiği (stdin ile gönderilir).

        ⚠ Neden sunucuda Python? Atomik (tek yazma), bayat temizliği ve
        JSON işleme tek yerde; bash'te tırnak cehennemi olurdu.
        """
        yuk = json.dumps(kayit or {}, ensure_ascii=False)
        return f'''
import json, os, time
DEFTER = os.path.expanduser("{DEFTER}")
TTL = {self._ttl}
SIMDI = time.time()
yol = os.path.dirname(DEFTER)
os.makedirs(yol, exist_ok=True)

satirlar = []
if os.path.exists(DEFTER):
    with open(DEFTER, encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except Exception:
                continue
            d["ts"] = float(d.get("ts") or 0)
            if SIMDI - d["ts"] <= TTL:
                satirlar.append(d)

islem = {islem!r}
if islem == "kayit":
    yeni = json.loads({yuk!r})
    nid = yeni.get("node_id")
    satirlar = [d for d in satirlar if d.get("node_id") != nid]
    satirlar.append(yeni)
elif islem == "birak":
    nid = json.loads({yuk!r}).get("node_id")
    satirlar = [d for d in satirlar if d.get("node_id") != nid]

with open(DEFTER, "w", encoding="utf-8") as f:
    for d in satirlar:
        f.write(json.dumps(d, ensure_ascii=False) + "\\n")

print(json.dumps(satirlar, ensure_ascii=False))
'''

    def register(self, *, port: int = 0, caps: Sequence[str] = (),
                 relay_port: int = 0, ad: str = "") -> Dict[str, Any]:
        """
        Bu düğümü havuza yazar (heartbeat).

        Returns:
            {"ok", "node_id", "ip", "komsu": N, "hata"}
        """
        if not self._initialized:
            return {"ok": False, "hata": "havuz başlatılmadı"}
        ip = self.genel_ip()
        kayit = {
            "node_id": self._kimlik,
            "ad": ad or platform.node(),
            "ip": ip,
            "port": int(port),
            "relay_port": int(relay_port),
            "caps": [str(c) for c in caps],
            "ts": time.time(),
            "surum": platform.platform()[:80],
        }
        r = self._ssh("python3 -", self._defter_betik("kayit", kayit))
        if not r.get("ok"):
            # ⚠ Sayaç artırılır: aksi hâlde "hata 0" görünüp sorun
            # sessizce yutulur (ölçüldü: testte hata sayacı 0 kaldı).
            self._counter["hata"] += 1
            self._son_hata = str(r.get("hata", "ssh"))[:200]
            return {"ok": False, "hata": r.get("hata", "ssh"), "ip": ip}
        self._counter["kayit"] += 1
        # ⚠ ÖLÇÜLDÜ: bu alan yazılmadığı için `son_kayit_yas` **epoch
        # kadar** (1,79 milyar sn) görünüyordu → heartbeat "çalışmıyor"
        # sanılıyordu. Başarılı kayıtta damga güncellenir.
        self._son_kayit = time.time()
        try:
            satirlar = json.loads(r.get("cikti") or "[]")
        except json.JSONDecodeError:
            satirlar = []
        logger.info(f"Havuza kaydedildi: {ip} · düğüm {len(satirlar)}")
        return {"ok": True, "node_id": self._kimlik, "ip": ip,
                "komsu": len(satirlar)}

    def birak(self) -> Dict[str, Any]:
        """Bu düğümü defterden çıkarır (temiz kapanış)."""
        r = self._ssh("python3 -", self._defter_betik(
            "birak", {"node_id": self._kimlik}))
        return {"ok": bool(r.get("ok")), "hata": r.get("hata", "")}

    def peers(self, *, kendini_say: bool = False,
              yalniz_canli: bool = True) -> List[Komsu]:
        """
        Havuzdaki komşuları okur.

        ⚠ Bayat satırlar sunucu tarafında da temizlenir; yine de istemci
        tarafında `canli` ile süzülür (saat farkı güvencesi).
        """
        r = self._ssh("python3 -", self._defter_betik("liste"))
        if not r.get("ok"):
            return []
        try:
            ham = json.loads(r.get("cikti") or "[]")
        except json.JSONDecodeError:
            return []
        self._counter["kesif"] += 1
        out: List[Komsu] = []
        for d in ham:
            k = Komsu.from_dict(d)
            if not kendini_say and k.node_id == self._kimlik:
                continue
            if yalniz_canli and not k.canli:
                continue
            out.append(k)
        out.sort(key=lambda x: x.yas)
        return out

    # ─── İş Paylaşımı ───────────────────────────────────────────────────

    def komsu_dene(self, is_istegi: Dict[str, Any], *,
                   uygun: Optional[Callable[[Komsu], bool]] = None,
                   deneme: int = 1) -> Dict[str, Any]:
        """
        Yalnız **komşuları** dener; yerel devralma YAPMAZ.

        ⚠ Neden ayrı? `dispatch` yerel devralmayı kendi yapar; ajan
        döngüsünde ise yerel çağrı **araç kayıt defterinden** yapılmalıdır
        (onay/politika dahil). Böylece sorumluluk karışmaz.

        Returns:
            {"ok": True, "kaynak": "ip:port|relay:N", "sonuc": ...}
            veya {"ok": False, "denenen": [...]}
        """
        komsular = [k for k in self.peers() if (uygun(k) if uygun else True)]
        denenen: List[str] = []
        for k in komsular[:max(0, int(deneme))]:
            denenen.append(k.adres)
            self._counter["gonderim"] += 1
            try:
                yanit = self.gonder(k, is_istegi)
            except Exception as exc:
                logger.debug(f"Komşu {k.adres}: {type(exc).__name__}")
                continue
            if yanit and yanit.get("ok"):
                self._counter["komsu_basarili"] += 1
                return {"ok": True, "kaynak": k.adres,
                        "sonuc": yanit.get("sonuc"), "denenen": denenen}
        return {"ok": False, "denenen": denenen}

    def dispatch(self, is_istegi: Dict[str, Any],
                 *, yerel: Callable[[], Any],
                 uygun: Optional[Callable[[Komsu], bool]] = None,
                 deneme: int = 2) -> Dict[str, Any]:
        """
        İşi önce komşulara verir; **yanıt gelmezse yerel üstlenir**.

        ⚠ Ölçülen gerçek: NAT yüzünden doğrudan bağlantı çoğu ağda
        kurulamaz. Bu yüzden `gonderici` çağrılabilir olarak dışarıdan
        verilebilir (ters tünel/relay uygulaması). Varsayılan gönderici
        `http` tabanlıdır ve başarısız olursa **yerel devralma** devreye
        girer — yani iş kaybolmaz.

        Args:
            is_istegi: Gönderilecek iş (sözlük)
            yerel: Yanıt alınamazsa çalıştırılacak yerel işlev
            uygun: Komşu süzgeci (yetenek eşleşmesi)
            deneme: Kaç komşu denenecek

        Returns:
            {"ok", "kaynak", "sonuc", "denenen", "hata"}
        """
        komsular = [k for k in self.peers() if (uygun(k) if uygun else True)]
        denenen: List[str] = []
        for k in komsular[:max(0, deneme)]:
            denenen.append(k.adres)
            self._counter["gonderim"] += 1
            try:
                yanit = self.gonder(k, is_istegi)
            except Exception as exc:
                logger.warning(f"Komşu {k.adres} hatası: {type(exc).__name__}")
                continue
            if yanit and yanit.get("ok"):
                self._counter["komsu_basarili"] += 1
                return {"ok": True, "kaynak": k.adres,
                        "sonuc": yanit.get("sonuc"),
                        "denenen": denenen, "hata": ""}
        # ── Hiçbiri yanıt vermedi: YEREL üstlen ──
        self._counter["yerel_devralma"] += 1
        try:
            sonuc = yerel()
        except Exception as exc:
            return {"ok": False, "kaynak": "yerel",
                    "sonuc": None, "denenen": denenen,
                    "hata": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "kaynak": "yerel", "sonuc": sonuc,
                "denenen": denenen, "hata": ""}

    def gonder(self, komsu: Komsu, is_istegi: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tek komşuya iş gönderir (HTTP).

        ⚠ Varsayılan uygulama: `http://<ip>:<port>/peer/job`.
        `relay_port` varsa sunucu adresi kullanılır (ters tünel).
        Bağlantı kurulamazsa `{"ok": False}` döner; çağıran yerel üstlenir.
        """
        import urllib.error
        import urllib.request

        # ⚠ ÖLÇÜLDÜ (bu makinede): `GatewayPorts` tanımsız (=no) →
        # ters tünel yalnız sunucunun **127.0.0.1**'ine bağlanır
        # (`ss -ltn`: LISTEN 127.0.0.1:18899). Bu yüzden komşunun
        # relay portuna **dışarıdan** bağlanılamaz; istek sunucunun
        # kendisinde çalıştırılır (SSH üzerinden relay).
        if komsu.relay_port:
            return self._relay_gonder(komsu, is_istegi)
        if komsu.port:
            url = f"http://{komsu.ip}:{komsu.port}/peer/job"
        else:
            return {"ok": False, "hata": "adres yok"}
        # ⚠ Paylaşılan anahtar her istekte gider (uzak taraf doğrular)
        _yuk = dict(is_istegi)
        _yuk.setdefault("token", self.token())
        veri = json.dumps(_yuk, ensure_ascii=False).encode("utf-8")
        istek = urllib.request.Request(
            url, data=veri, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(istek, timeout=6.0) as y:
                return json.loads(y.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as exc:
            return {"ok": False, "hata": f"{type(exc).__name__}: {exc}"}

    def _relay_gonder(self, komsu: Komsu,
                      is_istegi: Dict[str, Any]) -> Dict[str, Any]:
        """
        İşi **sunucu üzerinden** komşuya taşır (ters tünel relay).

        ⚠ Neden? Ölçüldü: `GatewayPorts` kapalı olduğu için tünel
        `127.0.0.1:<port>` dinler; dışarıdan erişilemez. İstek, sunucuda
        çalışan küçük bir Python betiğiyle `localhost:<port>`'a gönderilir.
        Böylece **router ayarı gerekmeden** iş paylaşımı çalışır.
        """
        _yuk = dict(is_istegi)
        _yuk.setdefault("token", self.token())
        yuk = json.dumps(_yuk, ensure_ascii=False)
        kod = (
            "import json, urllib.request\n"
            f"veri = json.loads({yuk!r})\n"
            "istek = urllib.request.Request(\n"
            f"    'http://127.0.0.1:{komsu.relay_port}/peer/job',\n"
            "    data=json.dumps(veri).encode('utf-8'), method='POST',\n"
            "    headers={'Content-Type': 'application/json'})\n"
            "print(urllib.request.urlopen(istek, timeout=8)\n"
            "      .read().decode('utf-8'))\n"
        )
        r = self._ssh("python3 -", kod)
        if not r.get("ok"):
            return {"ok": False, "hata": r.get("hata", "relay ssh")}
        try:
            return json.loads(r.get("cikti") or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "hata": "relay yanıtı çözülemedi"}

    def tunel_baslat(self, *, yerel_port: int, uzak_port: int = 0,
                     zaman_asimi: float = 20.0) -> Dict[str, Any]:
        """
        Kendi ters tünelini açar (`ssh -N -R`).

        ⚠ Tünel, düğüm çalıştığı sürece açık kalmalı; `ssh` süreci
        ölürse heartbeat durur ve kayıt TTL ile bayatlar (komşular düşürür).

        Args:
            yerel_port: Yerel HTTP API portu (peer/job ucu)
            uzak_port: Sunucudaki port (0 = ssh seçer)

        Returns:
            {"ok", "uzak_port", "surec", "hata"}
        """
        temel = self._ssh_komutu()
        if temel is None:
            return {"ok": False, "hata": "ssh yok"}
        # ⚠ `-R 0:…` ile ssh portu kendi seçer ama bunu **stderr**'den
        # okumak gerekir (kırılgan). Bu yüzden port **belirlenimci**:
        # düğüm kimliğinden türetilir → aynı düğüm her açılışta aynı port.
        if not uzak_port:
            try:
                uzak_port = 20000 + (int(self._kimlik[:4], 16) % 20000)
            except (ValueError, TypeError):
                uzak_port = 28000
        hedef = f"{int(uzak_port)}:127.0.0.1:{int(yerel_port)}"
        try:
            surec = subprocess.Popen(
                temel + ["-N", "-R", hedef],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"ok": False, "hata": f"{type(exc).__name__}: {exc}"}
        time.sleep(4.0)
        if surec.poll() is not None:
            hata = (surec.stderr.read() if surec.stderr else "")[:180]
            return {"ok": False, "hata": hata or "tünel kapandı"}
        return {"ok": True, "uzak_port": int(uzak_port), "surec": surec,
                "hata": ""}
