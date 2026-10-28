"""
Godot Karakter Üretici — Stylized Lowpoly + İskelet + Animasyon
================================================================
Hedef bileşeni 3: **karakter + monster modelleme, armature (iskelet)
ve animasyonlar**.

Neden kod ile iskelet?
    Blender'da iskelet kurmak ve GLTF aktarmak **zahmetli** ve dış
    bağımlılık gerektirir. Godot'ta iskelet **kod ile** kurulabilir:

        Skeleton3D.add_bone()          → kemik ekle
        Skeleton3D.set_bone_rest()     → dinlenme pozu
        Skeleton3D.set_bone_parent()   → hiyerarşi
        BoneAttachment3D               → mesh'i kemiğe bağla
        AnimationPlayer + Animation    → animasyon kaydet

    Bu, **dışa aktarma sorunu olmadan** çalışan bir iskelet verir.

Kemik yapısı (11 kemik — lowpoly için yeterli)
----------------------------------------------
    kok (root)
     └ govde (pelvis)           → kalça
        ├ omurga (spine)        → bel
        │  └ gogus (chest)      → göğüs
        │     ├ boyun → kafa    → baş
        │     ├ omuz_sol → kol_sol → on_kol_sol
        │     └ omuz_sag → kol_sag → on_kol_sag
        ├ bacak_sol → incik_sol → ayak_sol
        └ bacak_sag → incik_sag → ayak_sag

Animasyonlar
------------
    bekle (idle)   — hafif nefes salınımı
    yuru (walk)    — kol/bacak sallanması
    kos (run)      — hızlı, geniş salınım
    saldir (attack)— kol kaldırma + gövde dönüşü
    hasar (hurt)   — geriye savrulma
    ol (die)       — yere yığılma
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

#: Kemik tanımı: (ad, ebeveyn, dinlenme konumu, rol)
#:
#: Konumlar **ebeveyne göre yerel** koordinatlardır. Godot'ta ileri
#: yön `-Z`, yukarı `+Y`, sağ `+X`.
KEMIKLER: Sequence[Tuple[str, str, Tuple[float, float, float], str]] = (
    ("kok", "", (0.0, 0.0, 0.0), "root"),
    ("govde", "kok", (0.0, 0.95, 0.0), "pelvis"),
    ("omurga", "govde", (0.0, 0.25, 0.0), "spine"),
    ("gogus", "omurga", (0.0, 0.25, 0.0), "chest"),
    ("boyun", "gogus", (0.0, 0.22, 0.0), "neck"),
    ("kafa", "boyun", (0.0, 0.16, 0.0), "head"),
    ("omuz_sol", "gogus", (0.22, 0.12, 0.0), "shoulder_l"),
    ("kol_sol", "omuz_sol", (0.12, 0.0, 0.0), "upper_arm_l"),
    ("on_kol_sol", "kol_sol", (0.26, 0.0, 0.0), "forearm_l"),
    ("omuz_sag", "gogus", (-0.22, 0.12, 0.0), "shoulder_r"),
    ("kol_sag", "omuz_sag", (-0.12, 0.0, 0.0), "upper_arm_r"),
    ("on_kol_sag", "kol_sag", (-0.26, 0.0, 0.0), "forearm_r"),
    ("bacak_sol", "govde", (0.11, -0.05, 0.0), "thigh_l"),
    ("incik_sol", "bacak_sol", (0.0, -0.42, 0.0), "shin_l"),
    ("ayak_sol", "incik_sol", (0.0, -0.42, 0.0), "foot_l"),
    ("bacak_sag", "govde", (-0.11, -0.05, 0.0), "thigh_r"),
    ("incik_sag", "bacak_sag", (0.0, -0.42, 0.0), "shin_r"),
    ("ayak_sag", "incik_sag", (0.0, -0.42, 0.0), "foot_r"),
)

#: Parça tanımı: (ad, kemik, boyut, konum, renk anahtarı)
#:
#: Her parça bir kutu (`BoxMesh`) ve bir kemiğe `BoneAttachment3D` ile
#: bağlanır. Böylece kemik döndüğünde parça **birlikte** hareket eder.
PARÇALAR: Sequence[Tuple[str, str, Tuple[float, float, float],
                         Tuple[float, float, float], str]] = (
    ("leğen", "govde", (0.30, 0.18, 0.20), (0.0, 0.0, 0.0), "giysi"),
    ("göğüs_kutusu", "gogus", (0.36, 0.34, 0.22), (0.0, 0.05, 0.0), "giysi"),
    ("baş", "kafa", (0.24, 0.26, 0.24), (0.0, 0.06, 0.0), "ten"),
    ("üst_kol_sol", "kol_sol", (0.26, 0.11, 0.11), (0.13, 0.0, 0.0), "ten"),
    ("ön_kol_sol", "on_kol_sol", (0.24, 0.09, 0.09), (0.12, 0.0, 0.0), "ten"),
    ("üst_kol_sag", "kol_sag", (0.26, 0.11, 0.11), (-0.13, 0.0, 0.0), "ten"),
    ("ön_kol_sag", "on_kol_sag", (0.24, 0.09, 0.09), (-0.12, 0.0, 0.0), "ten"),
    ("uyluk_sol", "bacak_sol", (0.13, 0.42, 0.13), (0.0, -0.21, 0.0), "giysi"),
    ("incik_sol_u", "incik_sol", (0.11, 0.42, 0.11), (0.0, -0.21, 0.0), "giysi"),
    ("ayak_sol_u", "ayak_sol", (0.13, 0.08, 0.24), (0.0, -0.04, -0.06), "ayakkabi"),
    ("uyluk_sag", "bacak_sag", (0.13, 0.42, 0.13), (0.0, -0.21, 0.0), "giysi"),
    ("incik_sag_u", "incik_sag", (0.11, 0.42, 0.11), (0.0, -0.21, 0.0), "giysi"),
    ("ayak_sag_u", "ayak_sag", (0.13, 0.08, 0.24), (0.0, -0.04, -0.06), "ayakkabi"),
)

#: Karakter paletleri
KARAKTER_PALETLERI: Dict[str, Dict[str, List[float]]] = {
    "kahraman": {
        "ten": [0.85, 0.68, 0.55], "giysi": [0.25, 0.45, 0.75],
        "ayakkabi": [0.30, 0.22, 0.16], "sac": [0.35, 0.22, 0.14],
    },
    "buyucu": {
        "ten": [0.80, 0.65, 0.58], "giysi": [0.42, 0.25, 0.62],
        "ayakkabi": [0.25, 0.20, 0.30], "sac": [0.90, 0.88, 0.85],
    },
    "savasci": {
        "ten": [0.78, 0.60, 0.48], "giysi": [0.55, 0.30, 0.20],
        "ayakkabi": [0.28, 0.24, 0.20], "sac": [0.20, 0.16, 0.12],
    },
    "goblin": {
        "ten": [0.45, 0.62, 0.32], "giysi": [0.35, 0.30, 0.20],
        "ayakkabi": [0.25, 0.20, 0.15], "sac": [0.30, 0.35, 0.20],
    },
    "iskelet": {
        "ten": [0.88, 0.86, 0.80], "giysi": [0.55, 0.52, 0.48],
        "ayakkabi": [0.45, 0.42, 0.38], "sac": [0.80, 0.78, 0.72],
    },
}


def karakter_script(ad: str = "Karakter", palet: str = "kahraman",
                    monster: bool = False) -> str:
    """
    Godot karakter scripti üretir (iskelet + mesh + animasyon).

    ⚠ Ölçülen tuzak: Python 3.10'da **f-string ifadesi `\\` içeremez**
    (`f-string expression part cannot include a backslash`). GDScript
    kodu `\\` ve `{}` yoğun olduğu için şablon **yer tutucu + replace**
    yöntemiyle doldurulur.

    Args:
        ad: Karakter/script adı
        palet: `KARAKTER_PALETLERI` anahtarı
        monster: True ise düşman davranışı (kovalama, saldırı) eklenir

    Returns:
        GDScript kaynağı
    """
    renk = KARAKTER_PALETLERI.get(palet, KARAKTER_PALETLERI["kahraman"])
    kemik_json = json.dumps([k[0] for k in KEMIKLER], ensure_ascii=False)

    kemik_kodu = "\n".join(
        '\t_ekle("%s", "%s", Vector3(%s, %s, %s), "%s")'
        % (k_ad, ebeveyn, konum[0], konum[1], konum[2], rol)
        for k_ad, ebeveyn, konum, rol in KEMIKLER)

    parca_kodu = "\n".join(
        '\t_parca("%s", "%s", Vector3(%s, %s, %s), Vector3(%s, %s, %s), '
        '_renkler["%s"])'
        % (p_ad, kemik, boyut[0], boyut[1], boyut[2],
           konum[0], konum[1], konum[2], renk_a)
        for p_ad, kemik, boyut, konum, renk_a in PARÇALAR)

    renk_json = "{" + ", ".join(
        '"%s": Color(%s, %s, %s)' % (k, v[0], v[1], v[2])
        for k, v in renk.items()) + "}"

    yerler = {
        "__AD__": ad,
        "__PALET__": palet,
        "__MONSTER__": str(monster),
        "__KEMIK_JSON__": kemik_json,
        "__RENK_JSON__": renk_json,
        "__MAKS_SAGLIK__": "40" if monster else "100",
        "__GRUP__": '"dusman"' if monster else '"oyuncu"',
        "__KEMIK_KODU__": kemik_kodu,
        "__PARCA_KODU__": parca_kodu,
        # ⚠ Ölçülen hata: oyuncu karakterinde `_fizik` fonksiyonu yokken
        # `_fizik(delta)` çağrısı parse hatası veriyordu
        # ("Function _fizik() not found in base self"). Çağrı da
        # monster durumuna bağlanır.
        "__FIZIK_CAGRI__": "\t_fizik(delta)" if monster else "\tpass",
        "__MONSTER_KODU__": _MONSTER_KODU if monster else "",
    }
    kod = _KARAKTER_SABLONU
    for anahtar, deger in yerler.items():
        kod = kod.replace(anahtar, deger)
    return kod


#: Monster davranış kodu (durum makinesi) — şablona eklenir
_MONSTER_KODU = '''
# ─── Monster davranışı (durum makinesi) ───
enum Durum { BEKLE, KOVALA, SALDIR, OLU }
var durum: Durum = Durum.BEKLE
var oyuncu: Node3D = null
@export var gorus_menzili := 14.0
@export var saldiri_menzili := 1.8
@export var kovalama_hizi := 3.0
var _bekleme := 0.0


func _fizik(delta: float) -> void:
	if durum == Durum.OLU:
		return
	if not is_instance_valid(oyuncu):
		oyuncu = get_tree().get_first_node_in_group("oyuncu") as Node3D
	if not is_instance_valid(oyuncu):
		animasyon_oyna("bekle")
		return
	var mesafe := global_position.distance_to(oyuncu.global_position)
	if mesafe <= saldiri_menzili:
		durum = Durum.SALDIR
		animasyon_oyna("saldir")
		_bekleme += delta
		if _bekleme >= 1.2:
			_bekleme = 0.0
			if oyuncu.has_method("hasar_al"):
				oyuncu.hasar_al(8)
	elif mesafe <= gorus_menzili:
		durum = Durum.KOVALA
		animasyon_oyna("kos")
		var yon := (oyuncu.global_position - global_position)
		yon.y = 0.0
		yon = yon.normalized()
		position += yon * kovalama_hizi * delta
		look_at(global_position + yon, Vector3.UP)
	else:
		durum = Durum.BEKLE
		animasyon_oyna("bekle")
'''


_KARAKTER_SABLONU = '''extends Node3D
# myAi kararı: karakter "__AD__" — palet "__PALET__", monster=__MONSTER__
#
# İskelet **kod ile** kurulur: `Skeleton3D` + `BoneAttachment3D`.
# Bu, Blender'dan GLTF aktarma sorununu atlar ve doğrudan çalışır.
#
# ⚠ Ölçülen gerçek: `AnimationPlayer` ile `Skeleton3D` kemik pozunu
# sürmek için kemik başına node yolu gerekir ve bu **kırılgandır**.
# Güvenilir yol: kemik pozunu **kodla** sürmek
# (`set_bone_pose_rotation`). Lowpoly karakter için yeterlidir ve
# hiçbir animasyon kaynağı gerektirmez.

signal animasyon_degisti(ad: String)
signal saglik_degisti(yeni: int, maks: int)

@export var yurume_hizi := 4.0
@export var donme_hizi := 8.0

const KEMIKLER := __KEMIK_JSON__
const MAKS_SAGLIK := __MAKS_SAGLIK__

var iskelet: Skeleton3D = null
var saglik := MAKS_SAGLIK
var _parcalar: Array = []
var _renkler := __RENK_JSON__
var _kemik_idx := {}
var _hedef_animasyon := "bekle"
var _anim_zaman := 0.0


func _ready() -> void:
	add_to_group(__GRUP__)
	_iskelet_kur()
	_parcalari_kur()
	animasyon_oyna("bekle")
	saglik_degisti.emit(saglik, MAKS_SAGLIK)
	print("Karakter hazir: __AD__ — kemik: ", iskelet.get_bone_count())


# ─── İskelet ───

func _iskelet_kur() -> void:
	iskelet = Skeleton3D.new()
	iskelet.name = "Iskelet"
	add_child(iskelet)
	# Kemikler önce kurulur (ebeveyn önce gelmeli)
__KEMIK_KODU__


func _ekle(ad: String, ebeveyn: String, konum: Vector3, _rol: String) -> void:
	var idx := iskelet.add_bone(ad)
	var t := Transform3D(Basis(), konum)
	iskelet.set_bone_rest(idx, t)
	iskelet.set_bone_pose_position(idx, konum)
	if ebeveyn != "":
		var p: int = _kemik_idx.get(ebeveyn, -1)
		if p >= 0:
			iskelet.set_bone_parent(idx, p)
	_kemik_idx[ad] = idx


func _parcalari_kur() -> void:
__PARCA_KODU__


func _parca(ad: String, kemik: String, boyut: Vector3, konum: Vector3,
		renk: Color) -> void:
	var idx: int = _kemik_idx.get(kemik, -1)
	if idx < 0:
		push_warning("Kemik yok: " + kemik)
		return
	var bag := BoneAttachment3D.new()
	bag.name = ad
	bag.bone_idx = idx
	iskelet.add_child(bag)
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = boyut
	mi.mesh = bm
	mi.position = konum
	var m := StandardMaterial3D.new()
	m.albedo_color = renk
	m.roughness = 0.85
	m.metallic = 0.0
	mi.material_override = m
	bag.add_child(mi)
	_parcalar.append(bag)


# ─── Animasyon (kemik pozu kodla sürülür) ───

func animasyon_oyna(ad: String) -> void:
	if _hedef_animasyon != ad:
		_hedef_animasyon = ad
		animasyon_degisti.emit(ad)


func _yon_kemik(derece: Vector3) -> Quaternion:
	return Quaternion.from_euler(Vector3(
		deg_to_rad(derece.x), deg_to_rad(derece.y), deg_to_rad(derece.z)))


func _dondur(ad: String, derece: Vector3) -> void:
	var idx: int = _kemik_idx.get(ad, -1)
	if idx < 0:
		return
	iskelet.set_bone_pose_rotation(idx, _yon_kemik(derece))


func _process(delta: float) -> void:
	if iskelet == null:
		return
	_anim_zaman += delta
	_poz_guncelle()
__FIZIK_CAGRI__


func _poz_guncelle() -> void:
	var t := _anim_zaman
	match _hedef_animasyon:
		"bekle":
			_dondur("gogus", Vector3(sin(t * 1.5) * 1.5, 0, 0))
			_dondur("kol_sol", Vector3(0, 0, sin(t * 1.5) * 3.0))
			_dondur("kol_sag", Vector3(0, 0, -sin(t * 1.5) * 3.0))
		"yuru":
			var s := sin(t * 6.0) * 25.0
			_dondur("bacak_sol", Vector3(s, 0, 0))
			_dondur("bacak_sag", Vector3(-s, 0, 0))
			_dondur("incik_sol", Vector3(max(0.0, -s) * 0.8, 0, 0))
			_dondur("incik_sag", Vector3(max(0.0, s) * 0.8, 0, 0))
			_dondur("kol_sol", Vector3(-s * 0.7, 0, 0))
			_dondur("kol_sag", Vector3(s * 0.7, 0, 0))
		"kos":
			var s2 := sin(t * 11.0) * 45.0
			_dondur("bacak_sol", Vector3(s2, 0, 0))
			_dondur("bacak_sag", Vector3(-s2, 0, 0))
			_dondur("incik_sol", Vector3(max(0.0, -s2) * 1.1, 0, 0))
			_dondur("incik_sag", Vector3(max(0.0, s2) * 1.1, 0, 0))
			_dondur("kol_sol", Vector3(-s2 * 1.2, 0, 0))
			_dondur("kol_sag", Vector3(s2 * 1.2, 0, 0))
			_dondur("gogus", Vector3(12, 0, 0))
		"saldir":
			var v := sin(t * 8.0) * 0.5 + 0.5
			_dondur("kol_sag", Vector3(-110.0 * v, 0, 0))
			_dondur("on_kol_sag", Vector3(-40.0 * v, 0, 0))
			_dondur("gogus", Vector3(0, 20.0 * v, 0))
		"hasar":
			_dondur("gogus", Vector3(-18, 0, 0))
			_dondur("kol_sol", Vector3(0, 0, 35))
			_dondur("kol_sag", Vector3(0, 0, -35))
		"ol":
			_dondur("govde", Vector3(-80, 0, 0))


# ─── Hareket ───

func girdi_uygula(giris: Vector2, delta: float) -> void:
	var yon := Vector3(giris.x, 0, giris.y)
	if yon.length() < 0.01:
		animasyon_oyna("bekle")
		return
	yon = yon.normalized()
	position += yon * yurume_hizi * delta
	look_at(global_position + yon, Vector3.UP)
	animasyon_oyna("yuru")


func hasar_al(miktar: int) -> void:
	saglik = max(0, saglik - miktar)
	saglik_degisti.emit(saglik, MAKS_SAGLIK)
	animasyon_oyna("hasar" if saglik > 0 else "ol")

__MONSTER_KODU__'''


def karakter_sahnesi(ad: str = "Karakter", palet: str = "kahraman",
                     monster: bool = False) -> str:
    """Karakter için `.tscn` sahne metni üretir."""
    return f'''[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/{ad.lower()}.gd" id="1"]

[node name="{ad}" type="Node3D"]
script = ExtResource("1")
'''


def karakter_ozeti() -> Dict[str, Any]:
    """Karakter sisteminin özetini verir."""
    return {
        "kemik_sayisi": len(KEMIKLER),
        "parca_sayisi": len(PARÇALAR),
        "animasyonlar": ["bekle", "yuru", "kos", "saldir", "hasar", "ol"],
        "paletler": list(KARAKTER_PALETLERI.keys()),
        "kemikler": [k[0] for k in KEMIKLER],
        "roller": {k[0]: k[3] for k in KEMIKLER},
    }


if __name__ == "__main__":
    o = karakter_ozeti()
    print("=" * 74)
    print("  GODOT KARAKTER ÜRETİCİ")
    print("=" * 74)
    print(f"  kemik sayısı  : {o['kemik_sayisi']}")
    print(f"  parça sayısı  : {o['parca_sayisi']}")
    print(f"  animasyonlar  : {', '.join(o['animasyonlar'])}")
    print(f"  paletler      : {', '.join(o['paletler'])}")
    print()
    print("  KEMİK HİYERARŞİSİ:")
    for k_ad, ebeveyn, konum, rol in KEMIKLER:
        girinti = "  " if ebeveyn else ""
        print(f"    {girinti}{k_ad:14} ({rol:12}) ebeveyn={ebeveyn or '-'}")
    print()
    kod = karakter_script("Test", "kahraman")
    print(f"  üretilen script: {len(kod):,} karakter, "
          f"{kod.count(chr(10))} satır")
