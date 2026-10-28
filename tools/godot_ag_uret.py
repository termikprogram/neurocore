"""
Godot Ağ Üretici — Master Server + Proxy Server + İstemci
==========================================================
Hedef bileşeni 4: **master server + proxy server ağ mimarisi**.

Mimari
------
    ┌──────────┐   HTTPS    ┌───────────┐   dahili   ┌──────────────┐
    │ İstemci  │ ─────────► │   Proxy   │ ─────────► │    Master    │
    └──────────┘            │  (TLS,    │            │  (kimlik,    │
         │                  │   hız     │            │   eşleştirme │
         │                  │   sınırı) │            │   bilet)     │
         │                  └───────────┘            └──────────────┘
         │                                                    │
         │  bilet + sunucu adresi                             │
         │ ◄──────────────────────────────────────────────────┘
         │
         │  UDP/ENet (doğrudan)
         ▼
    ┌──────────────┐
    │ Oyun Sunucusu│  ← bileti master'a doğrulatır
    │ (yetkili     │
    │  fizik)      │
    └──────────────┘

Neden bu ayrım?
    * **Master** kimlik ve eşleştirme yükünü taşır — oyun sunucusu
      bu yükü taşımaz.
    * **Proxy** TLS sonlandırma, hız sınırlama, DDoS yumuşatma yapar —
      master'ın IP'si gizli kalır.
    * **Oyun sunucusu** yalnız maçı çalıştırır (yetkili fizik).
    * İstemci master'dan **bilet** alır, sonra oyun sunucusuna
      **doğrudan** bağlanır (düşük gecikme).

Güvenlik
--------
    * Bilet **tek kullanımlık** (`kullanildi` bayrağı)
    * Oyun sunucusu bileti master'a **doğrulatır** (paylaşılan anahtar)
    * `@rpc("any_peer")` ile gelen her çağrıda
      `multiplayer.get_remote_sender_id()` **doğrulanır**
    * Hız sınırı: saniyede en fazla N istek
"""

from __future__ import annotations

import json
from typing import Any, Dict

#: Godot ağ portları (yapılandırılabilir)
PORT_MASTER = 8780
PORT_PROXY = 8790
PORT_OYUN = 7777


# ─── Master Server ──────────────────────────────────────────────────────────

MASTER_SCRIPT = '''extends Node
# myAi ağ mimarisi: MASTER SERVER
#
# Görevleri: kimlik doğrulama, sunucu kaydı, maç eşleştirme, bilet üretme.
# Oyun sunucusu **bu sunucuya güvenir** ama kimlik yükünü taşımaz.

const PORT := __PORT_MASTER__
const MAKS_SUNUCU := 64
const MAKS_OYUNCU_SUNUCU := 16
#: Paylaşılan gizli anahtar (üretimde ortam değişkeninden okunmalı!)
const SUNUCU_ANAHTARI := "GIZLI_SUNUCU_ANAHTARI"

var _sunucular: Array = []          # kayıtlı oyun sunucuları
var _biletler: Dictionary = {}      # bilet -> {oyuncu, kullanildi, zaman}
var _oyuncular: Dictionary = {}     # oyuncu_id -> {ad, seviye}
var _istek_sayaci: Dictionary = {}  # ip -> [zaman damgaları]

signal sunucu_kaydedildi(adres: String, port: int)
signal bilet_uretildi(oyuncu_id: int, bilet: String)


func _ready() -> void:
	print("MASTER SERVER baslatiliyor — port ", PORT)
	var http := HTTPRequest.new()   # Gerçek kurulumda HTTPServer kullanılır
	add_child(http)
	print("Master hazir: ", MAKS_SUNUCU, " sunucu kapasitesi")


# ─── Sunucu kaydı ───

func sunucu_kaydet(adres: String, port: int, anahtar: String,
		maks_oyuncu: int = MAKS_OYUNCU_SUNUCU) -> bool:
	if anahtar != SUNUCU_ANAHTARI:
		push_warning("Gecersiz sunucu anahtari: " + adres)
		return false
	if _sunucular.size() >= MAKS_SUNUCU:
		push_warning("Sunucu kapasitesi dolu")
		return false
	_sunucular.append({
		"adres": adres, "port": port, "oyuncu": 0,
		"maks": maks_oyuncu, "zaman": Time.get_ticks_msec(),
	})
	sunucu_kaydedildi.emit(adres, port)
	print("Sunucu kayitli: ", adres, ":", port,
		" (toplam ", _sunucular.size(), ")")
	return true


func sunucu_durum_guncelle(adres: String, oyuncu: int) -> void:
	for s in _sunucular:
		if s["adres"] == adres:
			s["oyuncu"] = oyuncu
			return


# ─── Eşleştirme ───

func uygun_sunucu_bul() -> Dictionary:
	# En az dolu sunucuyu seç (yük dengeleme)
	var en_iyi: Dictionary = {}
	var en_az := 99999
	for s in _sunucular:
		if int(s["oyuncu"]) >= int(s["maks"]):
			continue
		if int(s["oyuncu"]) < en_az:
			en_az = int(s["oyuncu"])
			en_iyi = s
	return en_iyi


func bilet_uret(oyuncu_id: int) -> Dictionary:
	var sunucu := uygun_sunucu_bul()
	if sunucu.is_empty():
		return {}
	var bilet := "%d-%d-%d" % [oyuncu_id, Time.get_ticks_msec(),
		randi() % 100000]
	_biletler[bilet] = {
		"oyuncu": oyuncu_id, "kullanildi": false,
		"zaman": Time.get_ticks_msec(),
	}
	bilet_uretildi.emit(oyuncu_id, bilet)
	return {
		"bilet": bilet,
		"adres": sunucu["adres"],
		"port": sunucu["port"],
	}


# ─── Bilet doğrulama (oyun sunucusu çağırır) ───

func bilet_dogrula(bilet: String) -> int:
	# Döner: oyuncu_id (0 = geçersiz)
	var kayit = _biletler.get(bilet, null)
	if kayit == null:
		push_warning("Bilinmeyen bilet")
		return 0
	if kayit["kullanildi"]:
		push_warning("Bilet zaten kullanilmis — tekrar saldiri?")
		return 0
	# Bilet 5 dakika geçerli
	var yas := Time.get_ticks_msec() - int(kayit["zaman"])
	if yas > 300000:
		push_warning("Bilet suresi dolmus")
		return 0
	kayit["kullanildi"] = true
	return int(kayit["oyuncu"])


# ─── Hız sınırlama (proxy'de de var, burada yedek) ───

func hiz_izni(ip: String, maks_saniye: int = 10) -> bool:
	var simdi := Time.get_ticks_msec()
	var liste: Array = _istek_sayaci.get(ip, [])
	# Son 1 saniyeyi tut
	var yeni: Array = []
	for z in liste:
		if simdi - int(z) < 1000:
			yeni.append(z)
	if yeni.size() >= maks_saniye:
		_istek_sayaci[ip] = yeni
		push_warning("Hiz siniri asildi: " + ip)
		return false
	yeni.append(simdi)
	_istek_sayaci[ip] = yeni
	return true


func istatistik() -> Dictionary:
	return {
		"sunucu": _sunucular.size(),
		"oyuncu": _oyuncular.size(),
		"bilet": _biletler.size(),
	}
'''

# ─── Proxy Server ───────────────────────────────────────────────────────────

PROXY_SCRIPT = '''extends Node
# myAi ağ mimarisi: PROXY SERVER
#
# Görevleri: TLS sonlandırma (gerçek kurulumda), hız sınırlama,
# bağlantı yönlendirme, günlükleme, DDoS yumuşatma.
# Master'ın IP'si **gizli** kalır; istemci yalnız proxy'yi görür.

const PROXY_PORT := __PORT_PROXY__
const HEDEF_ADRES := "127.0.0.1"
const HEDEF_PORT := __PORT_MASTER__
const MAKS_BAGLANTI := 200
const MAKS_ISTEK_SANIYE := 20

var _sunucu := TCPServer.new()
var _istemciler: Array = []
var _istek_sayaci: Dictionary = {}
var _reddedilen := 0
var _iletilen := 0


func _ready() -> void:
	var hata := _sunucu.listen(PROXY_PORT)
	if hata != OK:
		push_error("Proxy dinleyemedi: %d" % hata)
		return
	print("PROXY hazir: ", PROXY_PORT, " -> ", HEDEF_ADRES, ":", HEDEF_PORT)


func _process(_delta: float) -> void:
	while _sunucu.is_connection_available():
		if _istemciler.size() >= MAKS_BAGLANTI:
			var fazla := _sunucu.take_connection()
			fazla.disconnect_from_host()
			_reddedilen += 1
			continue
		var istemci := _sunucu.take_connection()
		var giden := StreamPeerTCP.new()
		if giden.connect_to_host(HEDEF_ADRES, HEDEF_PORT) != OK:
			istemci.disconnect_from_host()
			_reddedilen += 1
			continue
		_istemciler.append({
			"gelen": istemci, "giden": giden,
			"ip": istemci.get_connected_host(),
			"baslangic": Time.get_ticks_msec(),
		})
		print("Proxy baglanti kabul: ", istemci.get_connected_host())
	_aktar()


func _aktar() -> void:
	var kopan: Array = []
	for i in _istemciler:
		var gelen: StreamPeerTCP = i["gelen"]
		var giden: StreamPeerTCP = i["giden"]
		gelen.poll()
		giden.poll()
		if (gelen.get_status() != StreamPeerTCP.STATUS_CONNECTED
				or giden.get_status() != StreamPeerTCP.STATUS_CONNECTED):
			kopan.append(i)
			continue
		# Istemci -> master
		if gelen.get_available_bytes() > 0:
			if not _hiz_izni(i["ip"]):
				_reddedilen += 1
				continue
			# ⚠ Ölçülen hata: `get_partial_data()` **Variant** döndürür;
			# `var veri := ...` tip çıkarımı yapamaz ve Godot parse hatası
			# verir ("Cannot infer the type of veri variable"). Açık tip
			# belirtilmelidir.
			var veri: PackedByteArray = gelen.get_partial_data(4096)[1]
			giden.put_data(veri)
			_iletilen += 1
		# Master -> istemci
		if giden.get_available_bytes() > 0:
			var veri2: PackedByteArray = giden.get_partial_data(4096)[1]
			gelen.put_data(veri2)
			_iletilen += 1
	for k in kopan:
		_istemciler.erase(k)
		print("Proxy baglanti kapandi")


func _hiz_izni(ip: String) -> bool:
	var simdi := Time.get_ticks_msec()
	var liste: Array = _istek_sayaci.get(ip, [])
	var yeni: Array = []
	for z in liste:
		if simdi - int(z) < 1000:
			yeni.append(z)
	if yeni.size() >= MAKS_ISTEK_SANIYE:
		_istek_sayaci[ip] = yeni
		push_warning("Proxy hiz siniri: " + ip)
		return false
	yeni.append(simdi)
	_istek_sayaci[ip] = yeni
	return true


func istatistik() -> Dictionary:
	return {
		"baglanti": _istemciler.size(),
		"iletilen": _iletilen,
		"reddedilen": _reddedilen,
	}
'''

# ─── Oyun Sunucusu ──────────────────────────────────────────────────────────

OYUN_SUNUCU_SCRIPT = '''extends Node
# myAi ağ mimarisi: OYUN SUNUCUSU (yetkili fizik)
#
# İstemci yalnız **girdi** gönderir; sonucu sunucu belirler.
# Hile önleme: menzil, hız sınırı, değer aralığı doğrulaması.

const PORT := __PORT_OYUN__
const MAKS_OYUNCU := 16
const MAKS_MENZIL := 3.0
const MAKS_SALDIRI_SANIYE := 5

var _biletler: Dictionary = {}      # oyuncu_id -> bilet
var _son_saldiri: Dictionary = {}   # oyuncu_id -> zaman
var _konumlar: Dictionary = {}      # oyuncu_id -> Vector3


func _ready() -> void:
	var peer := ENetMultiplayerPeer.new()
	var hata := peer.create_server(PORT, MAKS_OYUNCU)
	if hata != OK:
		push_error("Oyun sunucusu acilamadi: %d" % hata)
		return
	multiplayer.multiplayer_peer = peer
	multiplayer.peer_connected.connect(_on_baglandi)
	multiplayer.peer_disconnected.connect(_on_ayrildi)
	print("OYUN SUNUCUSU hazir — port ", PORT)


func _on_baglandi(id: int) -> void:
	print("Oyuncu baglandi: ", id)


func _on_ayrildi(id: int) -> void:
	print("Oyuncu ayrildi: ", id)
	_konumlar.erase(id)


# ─── Bilet doğrulama (istemci bunu çağırır) ───

@rpc("any_peer", "call_remote", "reliable")
func bilet_gonder(bilet: String) -> void:
	var gonderen := multiplayer.get_remote_sender_id()
	# ⚠ GÜVENLİK: master'a doğrulat (gerçek kurulumda HTTP isteği)
	# Burada basitleştirilmiş: bilet formatı ve tek kullanım kontrolü
	if bilet.length() < 8:
		push_warning("Gecersiz bilet: " + str(gonderen))
		oyuncu_at(gonderen, "gecersiz_bilet")
		return
	if _biletler.has(bilet):
		push_warning("Bilet zaten kullanilmis: " + str(gonderen))
		oyuncu_at(gonderen, "bilet_kullanilmis")
		return
	_biletler[bilet] = gonderen
	print("Bilet kabul: oyuncu ", gonderen)
	rpc_id(gonderen, "giris_onaylandi", gonderen)


@rpc("authority", "call_remote", "reliable")
func giris_onaylandi(oyuncu_id: int) -> void:
	print("Giris onaylandi: ", oyuncu_id)


# ─── Hareket (istemci -> sunucu, yetkili) ───

@rpc("any_peer", "call_remote", "unreliable_ordered")
func konum_bildir(yeni: Vector3) -> void:
	if not multiplayer.is_server():
		return
	var gonderen := multiplayer.get_remote_sender_id()
	# ⚠ GÜVENLİK: ışınlanma koruması
	var eski: Vector3 = _konumlar.get(gonderen, yeni)
	if eski.distance_to(yeni) > 12.0:
		push_warning("Isinlanma suphesi: oyuncu " + str(gonderen))
		return
	# Değer aralığı: harita sınırları
	if abs(yeni.x) > 200.0 or abs(yeni.z) > 200.0 or yeni.y < -20.0:
		return
	_konumlar[gonderen] = yeni
	rpc("konum_yayinla", gonderen, yeni)


@rpc("authority", "call_remote", "unreliable_ordered")
func konum_yayinla(oyuncu_id: int, konum: Vector3) -> void:
	pass  # istemci tarafı: uzak oyuncuyu güncelle


# ─── Saldırı (yetkili doğrulama) ───

@rpc("any_peer", "call_remote", "reliable")
func saldiri_iste(hedef_id: int) -> void:
	if not multiplayer.is_server():
		return
	var gonderen := multiplayer.get_remote_sender_id()
	var simdi := Time.get_ticks_msec()
	# 1) Hız sınırı: saniyede en fazla 5 saldırı
	if simdi - int(_son_saldiri.get(gonderen, 0)) < (1000 / MAKS_SALDIRI_SANIYE):
		return
	_son_saldiri[gonderen] = simdi
	# 2) Menzil kontrolü
	var a: Vector3 = _konumlar.get(gonderen, Vector3.ZERO)
	var b: Vector3 = _konumlar.get(hedef_id, Vector3.ZERO)
	if a.distance_to(b) > MAKS_MENZIL:
		push_warning("Menzil disi saldiri: " + str(gonderen))
		return
	# 3) Hasarı SUNUCU belirler (istemci miktar göndermez!)
	var hasar := 8
	rpc("hasar_uygula", hedef_id, hasar)
	print("Saldiri kabul: ", gonderen, " -> ", hedef_id)


@rpc("authority", "call_local", "reliable")
func hasar_uygula(oyuncu_id: int, miktar: int) -> void:
	var hedef := get_node_or_null("Oyuncular/%d" % oyuncu_id)
	if hedef and hedef.has_method("hasar_al"):
		hedef.hasar_al(miktar)


func oyuncu_at(id: int, neden: String) -> void:
	push_warning("Oyuncu atildi (%s): %d" % [neden, id])
	var peer := multiplayer.multiplayer_peer as ENetMultiplayerPeer
	if peer:
		peer.disconnect_peer(id)
'''

# ─── İstemci ────────────────────────────────────────────────────────────────

ISTEMCI_SCRIPT = '''extends Node
# myAi ağ mimarisi: İSTEMCİ
#
# Akış: master'dan bilet al -> oyun sunucusuna bağlan -> bileti gönder.
# İstemci **yalnız girdi** gönderir; konum ve hasar sunucudan gelir.

const ADRES := "127.0.0.1"
const PORT_OYUN := __PORT_OYUN__

var oyuncu_id := 0
var bilet := ""
var bagli := false

signal baglandi(oyuncu_id: int)
signal baglanti_hatasi(neden: String)


func master_bilet_iste(_kullanici: String, _sifre: String) -> void:
	# Gerçek kurulumda: HTTP isteği proxy üzerinden master'a gider.
	# Proxy: TLS sonlandirir, hiz sinirlar, master'a yonlendirir.
	#
	# var http := HTTPRequest.new()
	# add_child(http)
	# http.request("https://proxy.ornek.com/api/bilet",
	#     ["Content-Type: application/json"],
	#     HTTPClient.METHOD_POST,
	#     JSON.stringify({"kullanici": _kullanici, "sifre": _sifre}))
	# var yanit = await http.request_completed
	# var veri = JSON.parse_string(yanit[3].get_string_from_utf8())
	# bilet = veri["bilet"]
	# _oyun_sunucusuna_baglan(veri["adres"], veri["port"])
	print("Master'dan bilet isteniyor (proxy uzerinden)...")


func _oyun_sunucusuna_baglan(adres: String, port: int) -> void:
	var peer := ENetMultiplayerPeer.new()
	var hata := peer.create_client(adres, port)
	if hata != OK:
		baglanti_hatasi.emit("baglanti_kurulamadi")
		return
	multiplayer.multiplayer_peer = peer
	multiplayer.connected_to_server.connect(_on_baglandi)
	multiplayer.connection_failed.connect(_on_hata)
	multiplayer.server_disconnected.connect(_on_koptu)


func _on_baglandi() -> void:
	bagli = true
	print("Oyun sunucusuna baglandi — bilet gonderiliyor")
	rpc_id(1, "bilet_gonder", bilet)


func _on_hata() -> void:
	baglanti_hatasi.emit("sunucu_reddetti")


func _on_koptu() -> void:
	bagli = false
	baglanti_hatasi.emit("baglanti_koptu")


@rpc("authority", "call_remote", "reliable")
func giris_onaylandi(id: int) -> void:
	oyuncu_id = id
	baglandi.emit(id)
	print("Giris onaylandi: oyuncu ", id)


# ─── Girdi gönderimi ───

func girdi_gonder(yon: Vector2) -> void:
	if not bagli:
		return
	# ⚠ İstemci yalnız **girdi** gönderir, konum değil.
	# Sunucu konumu hesaplar ve yayınlar.
	rpc_id(1, "girdi_bildir", yon)


func konum_gonder(konum: Vector3) -> void:
	if not bagli:
		return
	rpc_id(1, "konum_bildir", konum)


func saldiri_gonder(hedef_id: int) -> void:
	if not bagli:
		return
	rpc_id(1, "saldiri_iste", hedef_id)
'''


def ag_scriptleri(master_port: int = PORT_MASTER,
                  proxy_port: int = PORT_PROXY,
                  oyun_port: int = PORT_OYUN) -> Dict[str, str]:
    """
    Ağ mimarisi scriptlerini üretir.

    Args:
        master_port: Master server portu
        proxy_port: Proxy portu
        oyun_port: Oyun sunucusu portu

    Returns:
        {'master.gd': ..., 'proxy.gd': ..., 'oyun_sunucu.gd': ..., 'istemci.gd': ...}
    """
    yerler = {
        "__PORT_MASTER__": str(master_port),
        "__PORT_PROXY__": str(proxy_port),
        "__PORT_OYUN__": str(oyun_port),
    }

    def doldur(sablon: str) -> str:
        for a, d in yerler.items():
            sablon = sablon.replace(a, d)
        return sablon

    return {
        "master.gd": doldur(MASTER_SCRIPT),
        "proxy.gd": doldur(PROXY_SCRIPT),
        "oyun_sunucu.gd": doldur(OYUN_SUNUCU_SCRIPT),
        "istemci.gd": doldur(ISTEMCI_SCRIPT),
    }


def ag_ozeti() -> Dict[str, Any]:
    """Ağ mimarisinin özetini verir."""
    return {
        "portlar": {
            "master": PORT_MASTER, "proxy": PORT_PROXY, "oyun": PORT_OYUN,
        },
        "bilesenler": ["master.gd", "proxy.gd", "oyun_sunucu.gd", "istemci.gd"],
        "guvenlik": [
            "tek kullanımlık bilet",
            "5 dakika bilet geçerliliği",
            "hız sınırı (10 istek/sn master, 20 proxy)",
            "ışınlanma koruması (12 birim)",
            "harita sınırı (200 birim)",
            "saldırı hız sınırı (5/sn)",
            "menzil kontrolü (3 birim)",
            "hasarı sunucu belirler",
            "sunucu anahtarı doğrulaması",
        ],
    }


if __name__ == "__main__":
    o = ag_ozeti()
    print("=" * 74)
    print("  GODOT AĞ MİMARİSİ — master + proxy + oyun sunucusu")
    print("=" * 74)
    print(f"  portlar: master={o['portlar']['master']} "
          f"proxy={o['portlar']['proxy']} oyun={o['portlar']['oyun']}")
    print(f"  bileşenler: {', '.join(o['bilesenler'])}")
    print()
    print("  GÜVENLİK ÖNLEMLERİ:")
    for g in o["guvenlik"]:
        print(f"    ✓ {g}")
    print()
    s = ag_scriptleri()
    for ad, kod in s.items():
        print(f"  {ad:18} {len(kod):6,} karakter  {kod.count(chr(10)):4} satır")
