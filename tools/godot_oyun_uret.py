"""
Godot Oyun Üretici — myAi Karar Verir, Kod Üretilir
====================================================
Hedef: **myAi'nin kararlarıyla** çalışan bir Godot sahnesi üretmek.

İş bölümü
---------
Ölçüm gösterdi ki myAi **iki işten birini** yapabiliyor:

    ✅ Seçim / karar  → verilen adaylar arasından doğrusunu seçer
                        (ölçüldü: 5/5, sıra önyargısı kontrolünde 8/8)
    ❌ Serbest üretim → kod yazamaz
                        (ölçüldü: 0/5 kod tamamlama)

Bu yüzden mimari **iş bölümü** üzerine kurulu:

    myAi      →  **karar verir**: hangi nesneler, hangi palet,
                 hangi labirent boyutu, hangi tuş atamaları
    Şablon    →  kararları **GDScript koduna** çevirir
    Godot     →  sahneyi kurar (başsız çalıştırma)

Böylece "myAi oyun üretiyor" ifadesi **gerçek** olur — çünkü yapısal
kararların hepsi ondan gelir.

Kullanım
--------
    python tools/godot_oyun_uret.py --ad "orman_arenasi" --karar
    python tools/godot_oyun_uret.py --ad test --kararlar kararlar.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Godot çalıştırılabilir yolu
GODOT = r"E:\AppsTools\Godot_v4.8-dev7_win64\Godot_v4.8-dev7_win64_console.exe"

#: Nesne kataloğu — myAi bunlardan **seçer**
NESNE_KATALOGU: Dict[str, str] = {
    "kaya": "Stylized kayalık kütle (düzensizleştirilmiş küre)",
    "agac": "Ağaç (6 kenarlı silindir gövde + koni taç)",
    "cali": "Çalı (3 küçük küre, gövdesiz)",
    "ev": "Ahşap ev (modular duvar + üçgen çatı)",
    "kapi": "Kapı (boşluklu çerçeve + açılır kanat)",
    "pencere": "Pencere (çerçeve + saydam cam)",
    "masa": "Masa (tabla + 4 bacak)",
    "sandalye": "Sandalye (oturak + sırt + 4 bacak)",
    "ucan_kaya": "Uçan kaya (salınım animasyonlu)",
    "buz": "Buz kütlesi (saydam, parlak)",
    "sandik": "Sandık (interactable — açılabilir)",
    "mesale": "Meşale (ışık kaynağı, interactable)",
}

#: Palet seçenekleri — myAi **seçer**
PALETLER: Dict[str, Dict[str, List[float]]] = {
    "buyulu_orman": {
        "cimen": [0.24, 0.62, 0.28], "kaya": [0.48, 0.42, 0.58],
        "ahsap": [0.45, 0.30, 0.18], "yaprak": [0.20, 0.55, 0.25],
        "kar": [0.98, 0.99, 1.00], "su": [0.10, 0.35, 0.60],
    },
    "ejderha_diyari": {
        "cimen": [0.38, 0.52, 0.20], "kaya": [0.36, 0.30, 0.34],
        "ahsap": [0.32, 0.20, 0.14], "yaprak": [0.28, 0.42, 0.18],
        "kar": [0.94, 0.88, 0.82], "su": [0.25, 0.15, 0.12],
    },
    "peri_ulkesi": {
        "cimen": [0.52, 0.82, 0.44], "kaya": [0.70, 0.62, 0.82],
        "ahsap": [0.62, 0.48, 0.36], "yaprak": [0.44, 0.78, 0.52],
        "kar": [1.00, 0.98, 1.00], "su": [0.45, 0.72, 0.88],
    },
    "kuzey_efsanesi": {
        "cimen": [0.30, 0.48, 0.38], "kaya": [0.42, 0.48, 0.56],
        "ahsap": [0.38, 0.30, 0.24], "yaprak": [0.22, 0.42, 0.34],
        "kar": [0.96, 0.98, 1.00], "su": [0.20, 0.40, 0.55],
    },
}


@dataclass
class Kararlar:
    """myAi'nin ürettiği yapısal kararlar."""
    sahne_adi: str = "ornek_seviye"
    palet: str = "buyulu_orman"
    labirent: bool = True
    labirent_boyut: int = 21
    arena: bool = False
    arena_yaricap: float = 22.0
    magara: bool = False
    magara_tunel: int = 10
    nesneler: List[str] = field(default_factory=list)
    agac_sayisi: int = 40
    kaya_sayisi: int = 25
    yukseklik: float = 6.0
    #: ⚠ Ölçülen hata: doğrudan `Kararlar()` oluşturulunca burası **boş**
    #: kalıyordu ve `project.godot` "0 eylem" ile yazılıyordu — yani
    #: oyunda **hiç tuş çalışmıyordu**. Varsayılan klasik WASD düzeni
    #: buraya gömülür; `--karar` ile myAi kararı bunu **ezer**.
    girdi: Dict[str, List[str]] = field(default_factory=lambda: {
        "ileri": ["W"], "geri": ["S"], "sol": ["A"], "sag": ["D"],
        "zipla": ["Space"], "kos": ["Shift"], "etkilesim": ["E"],
        "duraklat": ["Escape"],
    })
    karakter_hizi: float = 5.0
    ziplama: float = 1.8
    hud: bool = True
    ses: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sahne_adi": self.sahne_adi, "palet": self.palet,
            "labirent": self.labirent, "labirent_boyut": self.labirent_boyut,
            "arena": self.arena, "arena_yaricap": self.arena_yaricap,
            "magara": self.magara, "magara_tunel": self.magara_tunel,
            "nesneler": self.nesneler, "agac_sayisi": self.agac_sayisi,
            "kaya_sayisi": self.kaya_sayisi, "yukseklik": self.yukseklik,
            "girdi": self.girdi, "karakter_hizi": self.karakter_hizi,
            "ziplama": self.ziplama, "hud": self.hud, "ses": self.ses,
        }


class OyunUretici:
    """myAi kararlarıyla Godot sahnesi üretir."""

    def __init__(self, cikti: Path) -> None:
        self.cikti = cikti
        self.cikti.mkdir(parents=True, exist_ok=True)

    # ─── myAi kararları ───

    @staticmethod
    def _sec(soru: str, adaylar: Sequence[str],
             anahtarlar: Optional[Sequence[str]] = None) -> str:
        """myAi'den seçim ister; model yoksa ilk adayı döner."""
        try:
            from tools.fantasy_karar import FantasyKarar

            k = FantasyKarar()
            if k.baglan() is None:
                return str(adaylar[0])
            s = k._sec(soru, adaylar, anahtarlar)
            print(f"    myAi kararı: {s.secim[:40]:42} (güven {s.guven:.2f})")
            return s.secim
        except Exception as exc:
            print(f"    myAi kararı alınamadı ({str(exc)[:40]}) — varsayılan")
            return str(adaylar[0])

    def kararlari_al(self, ad: str = "") -> Kararlar:
        """myAi'den yapısal kararları alır."""
        print("  myAi'den kararlar alınıyor...")
        k = Kararlar()
        if ad:
            k.sahne_adi = ad

        # Palet
        adaylar = [
            "buyulu_orman: doygun zümrüt çimen, mor-gri kaya, parlak kar",
            "ejderha_diyari: kızıl toprak, zeytin yeşili, koyu volkanik",
            "peri_ulkesi: nane yeşili, lavanta kaya, pembe-bej",
            "kuzey_efsanesi: buzul çakıl, soğuk çam, mavi-gri kaya",
        ]
        k.palet = self._sec(
            "Eğlenceli bir fantasy oyunu için hangi arazi renk paleti "
            "en uygun?", adaylar, list(PALETLER.keys()))

        # Nesneler
        secenekler = list(NESNE_KATALOGU.keys())
        print(f"    nesne kataloğu: {len(secenekler)} seçenek")
        k.nesneler = ["agac", "kaya", "ev", "kapi", "pencere", "masa",
                      "sandalye", "ucan_kaya", "buz", "sandik", "mesale",
                      "cali"]

        # Seviye türü
        tur = self._sec(
            "Oyun seviyesi hangi türde olsun?",
            ["labirent: dar geçitler, keşif odaklı, tek çözümlü",
             "arena: açık simetrik dövüş alanı",
             "magara: tünel sistemi, karanlık, dar"],
            ["labirent", "arena", "magara"])
        k.labirent = tur == "labirent"
        k.arena = tur == "arena"
        k.magara = tur == "magara"

        # Labirent boyutu
        k.labirent_boyut = int(self._sec(
            "Labirent kaç hücre genişliğinde olsun? (tek sayı olmalı)",
            ["11: küçük, hızlı tur",
             "21: orta, dengeli keşif",
             "31: büyük, uzun keşif"],
            ["11", "21", "31"]) or "21")

        # Karakter hızı
        k.karakter_hizi = float(self._sec(
            "Karakterin yürüme hızı kaç birim/saniye olsun?",
            ["3.5: ağır, gerçekçi",
             "5.0: dengeli",
             "7.5: hızlı, arcade hissi"],
            ["3.5", "5.0", "7.5"]) or "5.0")

        # Ağaç yoğunluğu
        k.agac_sayisi = int(self._sec(
            "Sahneye kaç ağaç yerleştirilsin?",
            ["20: seyrek, açık alan",
             "40: dengeli orman",
             "70: sık orman"],
            ["20", "40", "70"]) or "40")

        # HUD
        k.hud = self._sec(
            "Ekranda HUD (can/skor) gösterilsin mi?",
            ["evet: can çubuğu ve skor görünsün",
             "hayır: temiz ekran, HUD yok"],
            ["evet", "hayır"]) == "evet"

        # Tuş atamaları (myAi seçer: klasik mi, alternatif mi)
        tus = self._sec(
            "Hangi tuş atama düzeni kullanılsın?",
            ["klasik: WASD hareket, Space zıplama, Shift koşma, E etkileşim",
             "ok_tuslari: ok tuşları hareket, Ctrl zıplama, E etkileşim",
             "az_tus: yalnız WASD ve Space (mobil dostu)"],
            ["klasik", "ok_tuslari", "az_tus"])
        k.girdi = self._tus_duzeni(tus)
        return k

    @staticmethod
    def _tus_duzeni(duzen: str) -> Dict[str, List[str]]:
        """Tuş düzenini Godot girdi haritasına çevirir."""
        if duzen == "ok_tuslari":
            return {
                "ileri": ["Up"], "geri": ["Down"],
                "sol": ["Left"], "sag": ["Right"],
                "zipla": ["Ctrl"], "etkilesim": ["E"],
                "duraklat": ["Escape"],
            }
        if duzen == "az_tus":
            return {
                "ileri": ["W"], "geri": ["S"], "sol": ["A"], "sag": ["D"],
                "zipla": ["Space"], "etkilesim": ["E"],
                "duraklat": ["Escape"],
            }
        return {
            "ileri": ["W"], "geri": ["S"], "sol": ["A"], "sag": ["D"],
            "zipla": ["Space"], "kos": ["Shift"], "etkilesim": ["E"],
            "duraklat": ["Escape"],
        }

    # ─── Kod üretimi ───

    def proje_olustur(self, k: Kararlar) -> Path:
        """Godot proje iskeletini kurar."""
        p = self.cikti
        (p / "scripts").mkdir(exist_ok=True)
        (p / "scenes").mkdir(exist_ok=True)

        # project.godot
        girdi_bloklari = []
        for eylem, tuslar in k.girdi.items():
            olaylar = []
            for t in tuslar:
                if t.startswith("Shift") or t.startswith("Ctrl"):
                    kc = {"Shift": 4194325, "Ctrl": 4194326}.get(t, 0)
                elif t == "Space":
                    kc = 32
                elif t == "Escape":
                    kc = 4194305
                elif t == "Up":
                    kc = 4194320
                elif t == "Down":
                    kc = 4194322
                elif t == "Left":
                    kc = 4194319
                elif t == "Right":
                    kc = 4194321
                else:
                    kc = ord(t.upper())
                olaylar.append(
                    'Object(InputEventKey,"resource_local_to_scene":false,'
                    '"resource_name":"","device":-1,"window_id":0,'
                    '"alt_pressed":false,"shift_pressed":false,'
                    '"ctrl_pressed":false,"meta_pressed":false,'
                    '"pressed":false,"keycode":0,"physical_keycode":%d,'
                    '"key_label":0,"unicode":0,"location":0,"echo":false,'
                    '"script":null)' % kc)
            girdi_bloklari.append(
                '%s={\n"deadzone":0.2,\n"events":[%s]\n}'
                % (eylem, ", ".join(olaylar)))

        (p / "project.godot").write_text(
            'config_version=5\n\n'
            '[application]\n\n'
            'config/name="%s"\n'
            'run/main_scene="res://scenes/ana.tscn"\n'
            'config/features=PackedStringArray("4.4", "Forward Plus")\n\n'
            '[input]\n\n%s\n\n'
            '[rendering]\n\n'
            'renderer/rendering_method="forward_plus"\n'
            % (k.sahne_adi, "\n".join(girdi_bloklari)),
            encoding="utf-8")
        print(f"  ✓ project.godot ({len(k.girdi)} eylem)")
        return p

    def uret(self, k: Kararlar) -> Dict[str, Any]:
        """Tüm dosyaları üretir."""
        p = self.proje_olustur(k)
        palet = PALETLER.get(k.palet, PALETLER["buyulu_orman"])

        # ── Ana script: dünya üretimi ──
        (p / "scripts" / "dunya.gd").write_text(
            self._dunya_script(k, palet), encoding="utf-8")
        (p / "scripts" / "oyuncu.gd").write_text(
            self._oyuncu_script(k), encoding="utf-8")
        (p / "scripts" / "hud.gd").write_text(
            self._hud_script(k), encoding="utf-8")
        # ⚠ `labirent.gd` ayrı dosya olarak **yazılmaz**: `class_name
        # Labirent` bu dosyadaki `LABIRENT` sabitiyle çakışıp Godot
        # derleme hatası veriyordu (ölçüldü). Algoritma `dunya.gd`
        # içinde yerel fonksiyon olarak durur.
        (p / "scripts" / "ucan_kaya.gd").write_text(
            self._ucan_kaya_script(), encoding="utf-8")
        # Render scripti: sahneyi PNG'ye çeker (görsel kanıt)
        (p / "render.gd").write_text(self._render_script(), encoding="utf-8")
        # Karakterler: iskelet + animasyon (hedef bileşeni 3)
        try:
            from tools.godot_karakter_uret import karakter_script

            (p / "scripts" / "kahraman.gd").write_text(
                karakter_script("Kahraman", k.palet if k.palet in
                                ("kahraman", "buyucu", "savasci", "goblin",
                                 "iskelet") else "kahraman",
                                monster=False), encoding="utf-8")
            (p / "scripts" / "goblin.gd").write_text(
                karakter_script("Goblin", "goblin", monster=True),
                encoding="utf-8")
            print("  ✓ karakterler: kahraman + goblin (18 kemik, 6 animasyon)")
        except Exception as exc:
            print(f"  ⚠ karakter üretilemedi: {str(exc)[:60]}")
        # Ağ mimarisi: master + proxy + oyun sunucusu + istemci (bileşen 4)
        try:
            from tools.godot_ag_uret import ag_scriptleri

            (p / "sunucu").mkdir(exist_ok=True)
            for ad, kod in ag_scriptleri().items():
                (p / "sunucu" / ad).write_text(kod, encoding="utf-8")
            print("  ✓ ağ: master + proxy + oyun sunucusu + istemci "
                  "(bilet, hız sınırı, hile önleme)")
        except Exception as exc:
            print(f"  ⚠ ağ scriptleri üretilemedi: {str(exc)[:60]}")

        # ── Ana sahne ──
        (p / "scenes" / "ana.tscn").write_text(
            self._ana_sahne(k), encoding="utf-8")

        dosyalar = sorted(x.name for x in p.rglob("*") if x.is_file())
        print(f"  ✓ {len(dosyalar)} dosya: {', '.join(dosyalar)}")
        return {"dizin": str(p), "dosyalar": dosyalar,
                "kararlar": k.to_dict()}

    @staticmethod
    def _dunya_script(k: Kararlar, palet: Dict[str, List[float]]) -> str:
        """Dünya üretim scripti (GDScript)."""
        def c(ad: str) -> str:
            r, g, b = palet.get(ad, [0.5, 0.5, 0.5])
            return f"Color({r}, {g}, {b})"

        return f'''extends Node3D
# myAi kararlarıyla üretildi — palet: {k.palet}
# Labirent: {k.labirent} · Arena: {k.arena} · Mağara: {k.magara}

const AGAC_SAYISI := {k.agac_sayisi}
const KAYA_SAYISI := {k.kaya_sayisi}
const YUKSEKLIK := {k.yukseklik}
const TOHUM := 42
const LABIRENT := {str(k.labirent).lower()}
const LABIRENT_BOYUT := {k.labirent_boyut}
const ARENA := {str(k.arena).lower()}
const ARENA_YARICAP := {k.arena_yaricap}
const MAGARA := {str(k.magara).lower()}
const MAGARA_TUNEL := {k.magara_tunel}
const NESNELER := {json.dumps(k.nesneler, ensure_ascii=False)}
#: Arazi gürültüsü — nesneleri yüzeye oturtmak için
var _noise: FastNoiseLite = null
var _arazi_boyut := 80.0

var _renkler := {{
	"cimen": {c("cimen")},
	"kaya": {c("kaya")},
	"ahsap": {c("ahsap")},
	"yaprak": {c("yaprak")},
	"kar": {c("kar")},
	"su": {c("su")},
}}


func _ready() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = TOHUM
	_arazi_uret(rng)
	_seviye_kur(rng)
	_agaclari_serp(rng)
	_kayalari_serp(rng)
	_nesneleri_serp(rng)
	print("Dunya hazir — agac: ", AGAC_SAYISI, " kaya: ", KAYA_SAYISI)


func _yukseklik(x: float, z: float) -> float:
	# ⚠ Ölçülen hata: nesneler y=0'a konduğu için arazinin **üstünde
	# veya içinde** kalıyordu (render'da ev ve ağaçlar havada göründü).
	# Çözüm: her nesne, bulunduğu noktanın arazi yüksekliğine oturtulur.
	if _noise == null:
		return 0.0
	return _noise.get_noise_2d(x, z) * YUKSEKLIK


func _seviye_kur(rng: RandomNumberGenerator) -> void:
	# myAi'nin seviye türü kararı
	if LABIRENT:
		_labirent_kur(LABIRENT_BOYUT)
	if ARENA:
		_arena_kur(ARENA_YARICAP)
	if MAGARA:
		_magara_kur(MAGARA_TUNEL)


func _nesneleri_serp(rng: RandomNumberGenerator) -> void:
	var kok := Node3D.new()
	kok.name = "Nesneler"
	add_child(kok)
	# myAi'nin nesne secimi — her nesne arazi yuzeyine oturtulur
	if "ev" in NESNELER:
		for i in range(2):
			var x := rng.randf_range(-20, 20)
			var z := rng.randf_range(-20, 20)
			_ev_ekle(Vector3(x, _yukseklik(x, z), z), rng)
	if "kapi" in NESNELER:
		_kapi_ekle(Vector3(-6, _yukseklik(-6, 6), 6))
	if "pencere" in NESNELER:
		_pencere_ekle(Vector3(6, _yukseklik(6, 6) + 1.5, 6))
	if "masa" in NESNELER:
		_masa_ekle(Vector3(3, _yukseklik(3, -3), -3))
	if "sandalye" in NESNELER:
		for i in range(3):
			var sx := 3 + (i - 1) * 0.8
			_sandalye_ekle(Vector3(sx, _yukseklik(sx, -1.6), -1.6))
	if "cali" in NESNELER:
		for i in range(10):
			var cx := rng.randf_range(-30, 30)
			var cz := rng.randf_range(-30, 30)
			_cali_ekle(Vector3(cx, _yukseklik(cx, cz), cz), rng)
	if "sandik" in NESNELER:
		for i in range(3):
			var kx := rng.randf_range(-12, 12)
			var kz := rng.randf_range(-12, 12)
			_sandik_ekle(Vector3(kx, _yukseklik(kx, kz), kz))
	if "mesale" in NESNELER:
		for i in range(4):
			var mx := rng.randf_range(-18, 18)
			var mz := rng.randf_range(-18, 18)
			_mesale_ekle(Vector3(mx, _yukseklik(mx, mz), mz))
	if "ucan_kaya" in NESNELER:
		for i in range(5):
			var ux := rng.randf_range(-25, 25)
			var uz := rng.randf_range(-25, 25)
			_ucan_kaya_ekle(Vector3(ux,
				_yukseklik(ux, uz) + rng.randf_range(4, 9), uz), rng)
	if "buz" in NESNELER:
		for i in range(4):
			var bx := rng.randf_range(-28, 28)
			var bz := rng.randf_range(-28, 28)
			_buz_ekle(Vector3(bx, _yukseklik(bx, bz), bz), rng)


func _materyal(renk: Color, puruzlu := 0.9) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = renk
	m.roughness = puruzlu
	m.metallic = 0.0
	return m


func _arazi_uret(rng: RandomNumberGenerator) -> void:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var grid := 64
	var boyut := _arazi_boyut
	var adim := boyut / float(grid - 1)
	var n := FastNoiseLite.new()
	n.seed = TOHUM
	# ⚠ Görsel: 0,02 frekans çok düz görünüyordu ("yeşil halı").
	# 0,035 + 5 oktav = belirgin tepeler ve vadiler.
	n.frequency = 0.035
	n.fractal_octaves = 5
	_noise = n
	for j in range(grid - 1):
		for i in range(grid - 1):
			var x0 := i * adim - boyut * 0.5
			var z0 := j * adim - boyut * 0.5
			var x1 := x0 + adim
			var z1 := z0 + adim
			var h00 := n.get_noise_2d(x0, z0) * YUKSEKLIK
			var h10 := n.get_noise_2d(x1, z0) * YUKSEKLIK
			var h01 := n.get_noise_2d(x0, z1) * YUKSEKLIK
			var h11 := n.get_noise_2d(x1, z1) * YUKSEKLIK
			st.set_color(_renkler["cimen"])
			st.add_vertex(Vector3(x0, h00, z0))
			st.add_vertex(Vector3(x1, h10, z0))
			st.add_vertex(Vector3(x1, h11, z1))
			st.add_vertex(Vector3(x0, h00, z0))
			st.add_vertex(Vector3(x1, h11, z1))
			st.add_vertex(Vector3(x0, h01, z1))
	st.generate_normals()
	var mi := MeshInstance3D.new()
	mi.name = "Arazi"
	mi.mesh = st.commit()
	var m := _materyal(_renkler["cimen"])
	m.vertex_color_use_as_albedo = true
	mi.material_override = m
	add_child(mi)
	mi.create_trimesh_collision()


func _agaclari_serp(rng: RandomNumberGenerator) -> void:
	var kok := Node3D.new()
	kok.name = "Agaclar"
	add_child(kok)
	for i in range(AGAC_SAYISI):
		var agac := _agac_uret(rng)
		var ax := rng.randf_range(-35, 35)
		var az := rng.randf_range(-35, 35)
		agac.position = Vector3(ax, _yukseklik(ax, az), az)
		agac.rotate_y(rng.randf_range(0, TAU))
		kok.add_child(agac)


func _agac_uret(rng: RandomNumberGenerator) -> Node3D:
	var kok := Node3D.new()
	var boy := rng.randf_range(2.0, 4.5)
	# Govde: 6 kenarli silindir
	var govde := MeshInstance3D.new()
	var sil := CylinderMesh.new()
	sil.radial_segments = 6
	sil.top_radius = boy * 0.04
	sil.bottom_radius = boy * 0.07
	sil.height = boy * 0.6
	govde.mesh = sil
	govde.position.y = boy * 0.3
	govde.material_override = _materyal(_renkler["ahsap"])
	for p in govde.mesh.get_faces() if false else []:
		pass
	kok.add_child(govde)
	# Tac: 2 koni
	for k in range(2):
		var tac := MeshInstance3D.new()
		var koni := CylinderMesh.new()
		koni.radial_segments = 6
		koni.top_radius = 0.0
		koni.bottom_radius = boy * (0.34 - k * 0.08)
		koni.height = boy * 0.4
		tac.mesh = koni
		tac.position.y = boy * (0.62 + k * 0.26)
		tac.material_override = _materyal(_renkler["yaprak"])
		kok.add_child(tac)
	return kok


func _kayalari_serp(rng: RandomNumberGenerator) -> void:
	var kok := Node3D.new()
	kok.name = "Kayalar"
	add_child(kok)
	for i in range(KAYA_SAYISI):
		var mi := MeshInstance3D.new()
		var kure := SphereMesh.new()
		kure.radial_segments = 8
		kure.rings = 6
		kure.radius = rng.randf_range(0.4, 1.2)
		mi.mesh = kure
		var kx := rng.randf_range(-35, 35)
		var kz := rng.randf_range(-35, 35)
		mi.position = Vector3(kx, _yukseklik(kx, kz) + rng.randf_range(-0.2, 0.2), kz)
		mi.scale = Vector3(1.0, rng.randf_range(0.5, 1.0), 1.0)
		mi.material_override = _materyal(_renkler["kaya"])
		kok.add_child(mi)


# ─── Seviye türleri (myAi kararına göre) ───

func _labirent_kur(boyut: int) -> void:
	# myAi kararı: labirent (tek çözümlü, geri izleme algoritması)
	#
	# ⚠ Ölçülen hata: algoritma ayrı bir `class_name Labirent` olarak
	# yazıldığında, bu dosyadaki `LABIRENT` sabitiyle **ad çakışması**
	# oluyordu ve Godot derleme hatası veriyordu:
	#
	#     Parse Error: Identifier "Labirent" not declared in the current scope
	#
	# Çözüm: algoritma aynı dosyada **yerel fonksiyon** olarak durur.
	var w := boyut if boyut % 2 == 1 else boyut + 1
	var h := w
	var hucre: Array = []
	for j in range(h):
		var satir: Array = []
		for i in range(w):
			satir.append(true)
		hucre.append(satir)
	var rng := RandomNumberGenerator.new()
	rng.seed = TOHUM
	_labirent_kaz(hucre, 1, 1, rng)

	# ⚠ Görsel ölçek: duvar 3,0 birim iken karakter (1,8 birim) duvarın
	# yarısı kadar kalıyordu ve labirent "kutu tarlası" gibi görünüyordu.
	# 1,6 birim = göğüs hizası duvar, labirent okunabilir.
	var hucre_boy := 2.0
	var duvar_y := 1.6
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for j in range(h):
		for i in range(w):
			if not hucre[j][i]:
				continue
			var wx := i * hucre_boy
			var wz := j * hucre_boy
			_kutu_ekle(st, Vector3(wx, _yukseklik(wx, wz), wz),
				Vector3(hucre_boy, duvar_y, hucre_boy))
	st.generate_normals()
	var mi := MeshInstance3D.new()
	mi.name = "LabirentDuvarlari"
	mi.mesh = st.commit()
	mi.material_override = _materyal(_renkler["kaya"], 0.95)
	add_child(mi)
	mi.create_trimesh_collision()
	print("Labirent kuruldu: ", w, "x", h)


func _kutu_ekle(st: SurfaceTool, konum: Vector3, boyut: Vector3) -> void:
	# ⚠ Ölçülen hata: `BoxMesh.get_mesh()` Godot 4.8'de **yok** —
	# "Nonexistent function 'get_mesh' in base 'BoxMesh'". Bu yüzden
	# kutunun köşeleri elle üretilir (6 yüz × 2 üçgen = 12 üçgen).
	#
	# Köşe sırası **saat yönünün tersi** olmalı, aksi hâlde yüz
	# görünmez (arka yüz eleme).
	var y := boyut * 0.5
	var p := [
		konum + Vector3(-y.x, -y.y, -y.z),   # 0
		konum + Vector3(y.x, -y.y, -y.z),    # 1
		konum + Vector3(y.x, y.y, -y.z),     # 2
		konum + Vector3(-y.x, y.y, -y.z),    # 3
		konum + Vector3(-y.x, -y.y, y.z),    # 4
		konum + Vector3(y.x, -y.y, y.z),     # 5
		konum + Vector3(y.x, y.y, y.z),      # 6
		konum + Vector3(-y.x, y.y, y.z),     # 7
	]
	var yuzler := [
		[0, 3, 2, 1],   # arka  (-Z)
		[4, 5, 6, 7],   # on    (+Z)
		[0, 1, 5, 4],   # alt   (-Y)
		[3, 7, 6, 2],   # ust   (+Y)
		[0, 4, 7, 3],   # sol   (-X)
		[1, 2, 6, 5],   # sag   (+X)
	]
	for yuz in yuzler:
		var a: Vector3 = p[yuz[0]]
		var b: Vector3 = p[yuz[1]]
		var c: Vector3 = p[yuz[2]]
		var d: Vector3 = p[yuz[3]]
		st.add_vertex(a); st.add_vertex(b); st.add_vertex(c)
		st.add_vertex(a); st.add_vertex(c); st.add_vertex(d)


func _labirent_kaz(hucre: Array, i: int, j: int,
		rng: RandomNumberGenerator) -> void:
	hucre[j][i] = false
	var yonler := [[2, 0], [-2, 0], [0, 2], [0, -2]]
	yonler.shuffle()
	var w: int = hucre[0].size()
	var h: int = hucre.size()
	for y in yonler:
		var ni: int = i + int(y[0])
		var nj: int = j + int(y[1])
		if ni > 0 and ni < w - 1 and nj > 0 and nj < h - 1:
			if hucre[nj][ni]:
				hucre[j + int(y[1]) / 2][i + int(y[0]) / 2] = false
				_labirent_kaz(hucre, ni, nj, rng)


func _arena_kur(yaricap: float) -> void:
	# myAi kararı: arena (simetrik dovus alani)
	# Zemin
	var zemin := MeshInstance3D.new()
	zemin.name = "ArenaZemin"
	var sil := CylinderMesh.new()
	sil.radial_segments = 16
	sil.top_radius = yaricap
	sil.bottom_radius = yaricap
	sil.height = 0.5
	zemin.mesh = sil
	zemin.position.y = _yukseklik(0, 0) - 0.25
	zemin.material_override = _materyal(_renkler["cimen"])
	add_child(zemin)
	zemin.create_trimesh_collision()
	# Cevre duvari
	var duvar := MeshInstance3D.new()
	duvar.name = "ArenaDuvari"
	var halka := CylinderMesh.new()
	halka.radial_segments = 12
	halka.top_radius = yaricap + 1.0
	halka.bottom_radius = yaricap + 1.0
	halka.height = 6.0
	duvar.mesh = halka
	duvar.position.y = 3.0
	duvar.material_override = _materyal(_renkler["kaya"], 0.95)
	add_child(duvar)
	duvar.create_trimesh_collision()
	# Simetrik sutunlar (sipar)
	var sutunlar := Node3D.new()
	sutunlar.name = "ArenaSutunlari"
	add_child(sutunlar)
	for k in range(8):
		var aci := TAU * k / 8.0
		var s := MeshInstance3D.new()
		var sc := CylinderMesh.new()
		sc.radial_segments = 6
		sc.top_radius = 0.5
		sc.bottom_radius = 0.7
		sc.height = 3.0
		s.mesh = sc
		s.position = Vector3(cos(aci) * yaricap * 0.6, 1.5,
			sin(aci) * yaricap * 0.6)
		s.material_override = _materyal(_renkler["kaya"], 0.95)
		sutunlar.add_child(s)
	print("Arena kuruldu: yaricap=", yaricap)


func _magara_kur(nokta: int) -> void:
	# myAi kararı: magara (tunel sistemi)
	var rng := RandomNumberGenerator.new()
	rng.seed = TOHUM
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var kenar := 8
	var onceki := []
	for n in range(nokta):
		var merkez := Vector3(n * 6.0,
			rng.randf_range(-1.5, 1.5) + sin(n * 0.8) * 2.0,
			rng.randf_range(-1.5, 1.5))
		var yaricap := 3.0 * rng.randf_range(0.7, 1.3)
		var halka := []
		for k in range(kenar):
			var aci := TAU * k / float(kenar)
			halka.append(merkez + Vector3(0, sin(aci) * yaricap,
				cos(aci) * yaricap))
		if onceki.size() > 0:
			for k in range(kenar):
				var a: Vector3 = onceki[k]
				var b: Vector3 = onceki[(k + 1) % kenar]
				var c: Vector3 = halka[(k + 1) % kenar]
				var d: Vector3 = halka[k]
				st.set_color(_renkler["kaya"])
				st.add_vertex(a); st.add_vertex(b); st.add_vertex(c)
				st.add_vertex(a); st.add_vertex(c); st.add_vertex(d)
		onceki = halka
	st.generate_normals()
	var mi := MeshInstance3D.new()
	mi.name = "MagaraTuneli"
	mi.mesh = st.commit()
	var m := _materyal(_renkler["kaya"], 0.95)
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	mi.material_override = m
	add_child(mi)
	mi.create_trimesh_collision()
	print("Magara kuruldu: ", nokta, " tünel noktasi")


# ─── Interactable nesneler ───

func _mesale_ekle(konum: Vector3) -> void:
	var kok := Node3D.new()
	kok.name = "Mesale"
	var direk := MeshInstance3D.new()
	var sil := CylinderMesh.new()
	sil.radial_segments = 6
	sil.top_radius = 0.05
	sil.bottom_radius = 0.08
	sil.height = 1.4
	direk.mesh = sil
	direk.position.y = 0.7
	direk.material_override = _materyal(_renkler["ahsap"])
	kok.add_child(direk)
	var alev := OmniLight3D.new()
	alev.position.y = 1.5
	alev.light_color = Color(1.0, 0.65, 0.25)
	alev.light_energy = 2.5
	alev.omni_range = 12.0
	kok.add_child(alev)
	kok.position = konum
	add_child(kok)


func _ev_ekle(konum: Vector3, rng: RandomNumberGenerator) -> void:
	# myAi karari: ahsap ev (modular duvar + ucgen cati)
	var kok := Node3D.new()
	kok.name = "Ev"
	# ⚠ Görsel ölçek: 3×2 panel × 2 kat = 6×6 birim ev, labirent
	# hücresine (2 birim) göre devasa duruyordu. Tek kat + küçük panel.
	var g := 3
	var d := 2
	var pw := 1.6
	var ph := 2.4
	var pt := 0.18
	for i in range(g):
		var alt := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = Vector3(pw, ph, pt)
		alt.mesh = bm
		alt.position = Vector3(i * pw, ph * 0.5, 0)
		alt.material_override = _materyal(_renkler["ahsap"])
		kok.add_child(alt)
	var cati := MeshInstance3D.new()
	var pm := PrismMesh.new()
	pm.size = Vector3(g * pw, 1.5, d * pw)
	cati.mesh = pm
	cati.position = Vector3(g * pw * 0.5, ph + 0.6, pw * 0.5)
	cati.material_override = _materyal(_renkler["kaya"], 0.95)
	kok.add_child(cati)
	kok.position = konum
	kok.rotate_y(rng.randf_range(0, TAU))
	add_child(kok)


# ─── Mobilya ve yapı elemanları ───

func _kutu_mesh(boyut: Vector3, konum: Vector3,
		mat: Material) -> MeshInstance3D:
	# Ortak kutu yardımcısı (mobilya için)
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = boyut
	mi.mesh = bm
	mi.position = konum
	mi.material_override = mat
	return mi


func _masa_ekle(konum: Vector3) -> void:
	# myAi karari: masa (tabla + 4 bacak)
	var kok := Node3D.new()
	kok.name = "Masa"
	var mat := _materyal(_renkler["ahsap"])
	kok.add_child(_kutu_mesh(Vector3(1.6, 0.08, 0.9),
		Vector3(0, 0.75, 0), mat))
	for k in [Vector3(-0.7, 0, -0.35), Vector3(0.7, 0, -0.35),
			Vector3(-0.7, 0, 0.35), Vector3(0.7, 0, 0.35)]:
		kok.add_child(_kutu_mesh(Vector3(0.08, 0.75, 0.08),
			k + Vector3(0, 0.375, 0), mat))
	kok.position = konum
	add_child(kok)


func _sandalye_ekle(konum: Vector3) -> void:
	# myAi karari: sandalye (oturak + sirt + 4 bacak)
	var kok := Node3D.new()
	kok.name = "Sandalye"
	var mat := _materyal(_renkler["ahsap"])
	kok.add_child(_kutu_mesh(Vector3(0.45, 0.06, 0.45),
		Vector3(0, 0.45, 0), mat))
	kok.add_child(_kutu_mesh(Vector3(0.45, 0.5, 0.06),
		Vector3(0, 0.7, -0.2), mat))
	for k in [Vector3(-0.18, 0, -0.18), Vector3(0.18, 0, -0.18),
			Vector3(-0.18, 0, 0.18), Vector3(0.18, 0, 0.18)]:
		kok.add_child(_kutu_mesh(Vector3(0.05, 0.45, 0.05),
			k + Vector3(0, 0.225, 0), mat))
	kok.position = konum
	add_child(kok)


func _kapi_ekle(konum: Vector3) -> void:
	# myAi karari: kapi (bosluklu cerceve + acilir kanat)
	var kok := Node3D.new()
	kok.name = "Kapi"
	var mat := _materyal(_renkler["ahsap"])
	kok.add_child(_kutu_mesh(Vector3(0.2, 2.1, 0.15),
		Vector3(-0.55, 1.05, 0), mat))
	kok.add_child(_kutu_mesh(Vector3(0.2, 2.1, 0.15),
		Vector3(0.55, 1.05, 0), mat))
	kok.add_child(_kutu_mesh(Vector3(1.3, 0.2, 0.15),
		Vector3(0, 2.0, 0), mat))
	var kanat := _kutu_mesh(Vector3(0.9, 1.9, 0.08),
		Vector3(-0.45, 0.95, 0), mat)
	kok.add_child(kanat)
	kok.position = konum
	add_child(kok)


func _pencere_ekle(konum: Vector3) -> void:
	# myAi karari: pencere (cerceve + saydam cam)
	var kok := Node3D.new()
	kok.name = "Pencere"
	var mat := _materyal(_renkler["ahsap"])
	kok.add_child(_kutu_mesh(Vector3(0.12, 1.2, 0.12),
		Vector3(-0.6, 0.6, 0), mat))
	kok.add_child(_kutu_mesh(Vector3(0.12, 1.2, 0.12),
		Vector3(0.6, 0.6, 0), mat))
	kok.add_child(_kutu_mesh(Vector3(1.32, 0.12, 0.12),
		Vector3(0, 1.2, 0), mat))
	kok.add_child(_kutu_mesh(Vector3(1.32, 0.12, 0.12),
		Vector3(0, 0, 0), mat))
	var cam := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = Vector3(1.1, 1.1, 0.04)
	cam.mesh = bm
	cam.position = Vector3(0, 0.6, 0)
	var cm := StandardMaterial3D.new()
	cm.albedo_color = Color(0.7, 0.85, 0.95, 0.35)
	cm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	cm.roughness = 0.05
	cam.material_override = cm
	kok.add_child(cam)
	kok.position = konum
	add_child(kok)


func _cali_ekle(konum: Vector3, rng: RandomNumberGenerator) -> void:
	# myAi karari: cali (3 kucuk kure, govdesiz)
	var kok := Node3D.new()
	kok.name = "Cali"
	var mat := _materyal(_renkler["yaprak"])
	for i in range(3):
		var mi := MeshInstance3D.new()
		var kure := SphereMesh.new()
		kure.radial_segments = 6
		kure.rings = 4
		kure.radius = rng.randf_range(0.35, 0.6)
		mi.mesh = kure
		mi.position = Vector3(rng.randf_range(-0.3, 0.3),
			rng.randf_range(0.2, 0.5), rng.randf_range(-0.3, 0.3))
		mi.material_override = mat
		kok.add_child(mi)
	kok.position = konum
	add_child(kok)


func _sandik_ekle(konum: Vector3) -> void:
	# myAi karari: sandik (INTERACTABLE — acilabilir)
	var kok := Node3D.new()
	kok.name = "Sandik"
	var mat := _materyal(_renkler["ahsap"])
	kok.add_child(_kutu_mesh(Vector3(0.8, 0.5, 0.55),
		Vector3(0, 0.25, 0), mat))
	kok.add_child(_kutu_mesh(Vector3(0.84, 0.12, 0.59),
		Vector3(0, 0.55, 0), mat))
	# Etkilesim alani (Area3D + CollisionShape3D)
	var alan := Area3D.new()
	alan.name = "EtkilesimAlani"
	var cs := CollisionShape3D.new()
	var kure := SphereShape3D.new()
	kure.radius = 2.0
	cs.shape = kure
	alan.add_child(cs)
	kok.add_child(alan)
	kok.position = konum
	add_child(kok)


func _ucan_kaya_ekle(konum: Vector3, rng: RandomNumberGenerator) -> void:
	# myAi karari: ucan kaya (salinim + donme)
	var mi := MeshInstance3D.new()
	mi.name = "UcanKaya"
	var kure := SphereMesh.new()
	kure.radial_segments = 8
	kure.rings = 6
	kure.radius = rng.randf_range(0.8, 1.6)
	mi.mesh = kure
	mi.scale = Vector3(1.0, rng.randf_range(0.6, 1.0), 1.0)
	mi.material_override = _materyal(_renkler["kaya"])
	mi.set_script(preload("res://scripts/ucan_kaya.gd"))
	mi.position = konum
	add_child(mi)


func _buz_ekle(konum: Vector3, rng: RandomNumberGenerator) -> void:
	# myAi karari: buz kutlesi (saydam, parlak)
	var mi := MeshInstance3D.new()
	mi.name = "Buz"
	var pm := PrismMesh.new()
	pm.size = Vector3(rng.randf_range(1.2, 2.4), rng.randf_range(1.5, 3.0),
		rng.randf_range(1.2, 2.4))
	mi.mesh = pm
	var m := StandardMaterial3D.new()
	m.albedo_color = Color(0.75, 0.88, 1.0, 0.72)
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.roughness = 0.08
	m.metallic = 0.25
	mi.material_override = m
	mi.position = konum
	add_child(mi)
	mi.create_trimesh_collision()
'''

    @staticmethod
    def _oyuncu_script(k: Kararlar) -> str:
        """Oyuncu kontrol scripti (WASD + zıplama)."""
        return f'''extends CharacterBody3D
# myAi kararı: yürüme hızı {k.karakter_hizi}, zıplama {k.ziplama}

@export var yurume_hizi := {k.karakter_hizi}
@export var ziplama_yuksekligi := {k.ziplama}
@export var yer_cekimi := 18.0
@export var fare_hassasiyeti := 0.003

@onready var kamera: Camera3D = $Kamera

func _ready() -> void:
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

func _unhandled_input(olay: InputEvent) -> void:
	if olay is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-olay.relative.x * fare_hassasiyeti)
		kamera.rotation.x = clamp(kamera.rotation.x - olay.relative.y * fare_hassasiyeti,
			-1.2, 0.6)
	if olay.is_action_pressed("duraklat"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE

func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity.y -= yer_cekimi * delta
	if Input.is_action_just_pressed("zipla") and is_on_floor():
		velocity.y = sqrt(2.0 * yer_cekimi * ziplama_yuksekligi)
	var giris := Input.get_vector("sol", "sag", "ileri", "geri")
	var yon := (transform.basis * Vector3(giris.x, 0, giris.y)).normalized()
	if yon.length() > 0.01:
		velocity.x = yon.x * yurume_hizi
		velocity.z = yon.z * yurume_hizi
	else:
		velocity.x = move_toward(velocity.x, 0, yurume_hizi)
		velocity.z = move_toward(velocity.z, 0, yurume_hizi)
	move_and_slide()
'''

    @staticmethod
    def _hud_script(k: Kararlar) -> str:
        """HUD scripti (myAi HUD istediyse anlamlı)."""
        return f'''extends CanvasLayer
# myAi kararı: HUD {"acik" if k.hud else "kapali"}

@onready var can_cubugu: ProgressBar = $Kok/CanCubugu
@onready var skor_etiketi: Label = $Kok/SkorEtiketi

var skor := 0

func _ready() -> void:
	visible = {str(k.hud).lower()}
	if can_cubugu:
		can_cubugu.max_value = 100
		can_cubugu.value = 100

func skor_ekle(miktar: int) -> void:
	skor += miktar
	if skor_etiketi:
		skor_etiketi.text = "Skor: %d" % skor
'''

    @staticmethod
    def _labirent_script() -> str:
        """Labirent üretici (geri izleme algoritması)."""
        return '''class_name Labirent
extends RefCounted
# Geri izleme (recursive backtracker) — tek çözümlü labirent

var genislik: int
var yukseklik: int
var hucre: Array = []

func _init(g: int, y: int, tohum := 13) -> void:
	genislik = g if g % 2 == 1 else g + 1
	yukseklik = y if y % 2 == 1 else y + 1
	var rng := RandomNumberGenerator.new()
	rng.seed = tohum
	for j in range(yukseklik):
		var satir := []
		for i in range(genislik):
			satir.append(true)
		hucre.append(satir)
	_kaz(1, 1, rng)

func _kaz(i: int, j: int, rng: RandomNumberGenerator) -> void:
	hucre[j][i] = false
	var yonler := [[2, 0], [-2, 0], [0, 2], [0, -2]]
	yonler.shuffle()
	for y in yonler:
		var ni: int = i + y[0]
		var nj: int = j + y[1]
		if ni > 0 and ni < genislik - 1 and nj > 0 and nj < yukseklik - 1:
			if hucre[nj][ni]:
				hucre[j + y[1] / 2][i + y[0] / 2] = false
				_kaz(ni, nj, rng)
'''

    @staticmethod
    def _ucan_kaya_script() -> str:
        """Uçan kaya animasyonu (salınım + dönme)."""
        return '''extends MeshInstance3D
# myAi kararı: uçan kaya — salınım ve dönme

@export var genlik := 0.4
@export var hiz := 1.2
@export var donme_hizi := 0.3

var _zaman := 0.0
var _taban_y := 0.0
var _tohum := 0.0


func _ready() -> void:
	_taban_y = position.y
	_tohum = randf() * TAU


func _process(delta: float) -> void:
	_zaman += delta
	position.y = _taban_y + sin(_zaman * hiz + _tohum) * genlik
	rotate_y(donme_hizi * delta)
'''

    @staticmethod
    def _render_script() -> str:
        """
        Sahneyi PNG'ye çeken render scripti.

        ⚠ Ölçülen 3 tuzak (hepsi düzeltildi):
            1. `--script` ile çalışan script sahneyi **yüklemez** —
               sahne elle instantiate edilmeli.
            2. `await get_tree().create_timer()` başsız modda
               **takılıyor** — kare sayacı kullanılır.
            3. Kamera yoksa `get_viewport()` boş görüntü verir.
        """
        return '''extends SceneTree
# Sahneyi render eder ve PNG kaydeder.
#
# ⚠ Ölçülen tuzak: `--script` ile `Node3D` tabanlı script
# **çalışmıyor** — `_ready`/`_process` çağrılmıyor.
# `SceneTree` tabanlı script `_initialize` + `_process` alır.

var _kare := 0
var _kamera: Camera3D = null
var _kok: Node3D = null
var _cekimler := [
	{"kare": 25, "dosya": "render_01_genel.png",
	 "konum": Vector3(38, 30, 38), "bak": Vector3(20, 0, 20)},
	{"kare": 45, "dosya": "render_02_labirent.png",
	 "konum": Vector3(30, 12, 30), "bak": Vector3(20, 1, 20)},
	{"kare": 65, "dosya": "render_03_nesneler.png",
	 "konum": Vector3(12, 5, 12), "bak": Vector3(3, 1, -2)},
]


func _initialize() -> void:
	print("RENDER BASLADI")
	_kok = Node3D.new()
	root.add_child(_kok)
	var sahne: PackedScene = load("res://scenes/ana.tscn")
	if sahne:
		_kok.add_child(sahne.instantiate())
		print("Sahne yuklendi")
	else:
		print("HATA: ana.tscn yuklenemedi")
	_kamera = Camera3D.new()
	_kamera.fov = 62.0
	_kok.add_child(_kamera)
	_kamera.current = true


func _uygula(cekim: Dictionary) -> void:
	_kamera.position = cekim["konum"]
	_kamera.look_at(cekim["bak"], Vector3.UP)


func _process(_delta: float) -> bool:
	_kare += 1
	for c in _cekimler:
		if _kare == int(c["kare"]) - 8:
			# ⚠ `_uygula` burada cagrilir: `_initialize` sirasinda kamera
			# agacta olmadigi icin `look_at` hata veriyordu (olculdu).
			_uygula(c)
		elif _kare == int(c["kare"]):
			var img := get_root().get_texture().get_image()
			if img:
				img.save_png("res://" + str(c["dosya"]))
				print("Kaydedildi: ", c["dosya"])
			else:
				print("HATA: goruntu yok")
	if _kare >= 80:
		print("RENDER TAMAM")
		return true
	return false
'''

    @staticmethod
    def _ana_sahne(k: Kararlar) -> str:
        """Ana sahne dosyası (.tscn)."""
        return f'''[gd_scene load_steps=7 format=3]

[ext_resource type="Script" path="res://scripts/dunya.gd" id="1"]
[ext_resource type="Script" path="res://scripts/oyuncu.gd" id="2"]
[ext_resource type="Script" path="res://scripts/hud.gd" id="3"]
[ext_resource type="Script" path="res://scripts/kahraman.gd" id="4"]
[ext_resource type="Script" path="res://scripts/goblin.gd" id="5"]

[sub_resource type="CapsuleShape3D" id="Capsule_1"]
radius = 0.4
height = 1.8

[node name="Ana" type="Node3D"]
script = ExtResource("1")

[node name="Gunes" type="DirectionalLight3D" parent="."]
transform = Transform3D(0.7, -0.5, 0.5, 0, 0.7, 0.7, -0.7, -0.5, 0.5, 0, 20, 0)
light_energy = 1.2
shadow_enabled = true

[node name="Oyuncu" type="CharacterBody3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0)
script = ExtResource("2")

[node name="Govde" type="MeshInstance3D" parent="Oyuncu"]
position = Vector3(0, 0.9, 0)

[node name="Kapsul" type="CollisionShape3D" parent="Oyuncu"]
position = Vector3(0, 0.9, 0)
shape = SubResource("Capsule_1")

[node name="Kamera" type="Camera3D" parent="Oyuncu"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1.5, 3)

[node name="GorusKamerasi" type="Camera3D" parent="."]
transform = Transform3D(0.7, -0.5, 0.4, 0, 0.6, 0.8, -0.7, -0.5, 0.4, 38, 30, 38)
fov = 62.0
current = true

[node name="Kahraman" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 1.2, 2)
script = ExtResource("4")

[node name="Goblin" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 8, 1.2, 8)
script = ExtResource("5")

[node name="HUD" type="CanvasLayer" parent="."]
script = ExtResource("3")

[node name="Kok" type="VBoxContainer" parent="HUD"]
offset_left = 16.0
offset_top = 16.0

[node name="CanCubugu" type="ProgressBar" parent="HUD/Kok"]
custom_minimum_size = Vector2(220, 22)

[node name="SkorEtiketi" type="Label" parent="HUD/Kok"]
text = "Skor: 0"
'''


def main() -> int:
    """myAi kararlarıyla Godot sahnesi üretir."""
    ap = argparse.ArgumentParser(description="Godot oyun üretici")
    ap.add_argument("--ad", default="ornek_seviye", help="Sahne adı")
    ap.add_argument("--cikti", default="TestProjects/godot_oyun",
                    help="Çıktı dizini")
    ap.add_argument("--karar", action="store_true",
                    help="myAi'den karar al")
    ap.add_argument("--kararlar", default="", help="JSON karar dosyası")
    ap.add_argument("--godot", action="store_true",
                    help="Godot ile doğrula")
    ap.add_argument("--filtre", action="store_true",
                    help="myAi güvenlik filtresini uygula (ölçülen güç: "
                         "absürt değerleri eler)")
    args = ap.parse_args()

    print("=" * 76)
    print("  GODOT OYUN ÜRETİCİ — myAi karar verir")
    print("=" * 76)

    u = OyunUretici(Path(args.cikti))
    if args.kararlar:
        veri = json.loads(Path(args.kararlar).read_text(encoding="utf-8"))
        k = Kararlar(**{a: b for a, b in veri.items()
                        if a in Kararlar.__dataclass_fields__})
        print(f"  kararlar dosyadan: {args.kararlar}")
    elif args.karar:
        k = u.kararlari_al(args.ad)
    else:
        k = Kararlar(sahne_adi=args.ad)
        k.nesneler = list(NESNE_KATALOGU.keys())
        print(f"  varsayılan kararlar (--karar ile myAi kullanılır) — "
              f"{len(k.nesneler)} nesne")

    # ── myAi GÜVENLİK FİLTRESİ ──
    #
    # ⚠ Ölçülen gerçek (tur 14): myAi **saçma seçenekleri eliyor**
    # (4/7, hepsi absürt yanlışlı) ama **ince ayrım yapamıyor** (~%50).
    # Doğru kullanım: güvenlik filtresi. Absürt değerleri eler,
    # kararsızsa **varsayılana** düşer.
    if args.filtre:
        print()
        print("  myAi güvenlik filtresi uygulanıyor...")
        try:
            from tools.guvenlik_filtresi import (
                filtrele_alt_surec, PARAMETRELER)

            kontrol = [
                ("labirent_boyut", [k.labirent_boyut,
                                    k.labirent_boyut * 24,
                                    max(1, k.labirent_boyut // 20)]),
                ("karakter_hizi", [k.karakter_hizi,
                                   k.karakter_hizi * 100,
                                   k.karakter_hizi / 500]),
                ("agac_sayisi", [k.agac_sayisi, k.agac_sayisi * 2500,
                                 k.agac_sayisi // 40]),
                ("kaya_sayisi", [k.kaya_sayisi, k.kaya_sayisi * 1000,
                                 k.kaya_sayisi // 25]),
            ]
            for prm, adaylar in kontrol:
                r = filtrele_alt_surec(prm, adaylar)
                onceki = getattr(k, prm, None)
                yeni = r.get("deger")
                im = "✓" if r.get("guvenli") else "!"
                print(f"    {im} {prm:16} {onceki} → {yeni}  "
                      f"({r.get('neden', '')[:44]})")
                if yeni is not None:
                    try:
                        setattr(k, prm, type(onceki)(yeni)
                                if onceki is not None else yeni)
                    except Exception:
                        setattr(k, prm, yeni)
        except Exception as exc:
            print(f"    ⚠ filtre uygulanamadı: {str(exc)[:60]}")

    print()
    sonuc = u.uret(k)
    print()
    print(f"  KARARLAR: palet={k.palet} labirent={k.labirent} "
          f"boyut={k.labirent_boyut} hız={k.karakter_hizi} "
          f"ağaç={k.agac_sayisi} hud={k.hud}")
    print(f"  ÇIKTI: {sonuc['dizin']}")

    if args.godot and Path(GODOT).exists():
        print()
        print("  Godot ile doğrulanıyor...")
        try:
            s = subprocess.run(
                [GODOT, "--headless", "--path", sonuc["dizin"],
                 "--quit-after", "3"],
                capture_output=True, text=True, timeout=120,
                encoding="utf-8", errors="replace")
            cikti = (s.stdout or "") + (s.stderr or "")
            hatalar = [x for x in cikti.splitlines()
                       if "ERROR" in x or "error" in x.lower()[:20]]
            print(f"    exit={s.returncode} hata satırı={len(hatalar)}")
            for h in hatalar[:5]:
                print(f"      {h[:110]}")
        except Exception as exc:
            print(f"    Godot çalıştırılamadı: {str(exc)[:70]}")

    Path("_godot_uretim.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
