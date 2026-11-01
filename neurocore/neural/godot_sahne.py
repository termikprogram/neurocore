"""
Godot — sahne kurulumu, Player sahnesi, terrain ve terrain'e göre spawn.

Bu katman, gerçek bir projede (Balon 3D) ölçülerek çıkarılmış dersleri içerir:
her dersin yanında **çalışan kod** ve **doğrulama kriteri** vardır.

Katman adı: `godot_sahne`
"""

from __future__ import annotations

from typing import List

from .curriculum import Lesson

LAYER = "godot_sahne"
LABEL = "Sahne Kurulumu, Player ve Terrain Spawn"


def _sahne_lessons() -> List[Lesson]:
    """Ana sahne, player sahnesi, terrain ve spawn dersleri."""

    return [
        Lesson(
            LAYER,
            "Godot'ta ana sahne (.tscn) nasıl kurulur?",
            "Bir .tscn dosyası üç bölümden oluşur: `[ext_resource]` (dış dosya "
            "referansları), `[sub_resource]` (sahne içi kaynaklar) ve `[node]` "
            "(düğüm ağacı). Dış bir sahneyi (ör. terrain.glb) ana sahneye eklemek "
            "için `ext_resource type=\"PackedScene\"` tanımlanır ve düğüm satırında "
            "`instance=ExtResource(...)` yazılır. `parent=\".\"` kökün altına ekler; "
            "iç içe düğümler için `parent=\"DugumAdi\"` kullanılır. "
            "Sahne dosyası elle de yazılabilir; Godot açılışta ayrıştırır ve "
            "hata varsa editörde kırmızı gösterir.",
            ["tscn", "ext_resource", "instance", "sahne"],
            """[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://main.gd" id="1_main"]
[ext_resource type="PackedScene" path="res://assets/terrain/terrain_lowpoly.glb" id="2_terrain"]

[node name="Main" type="Node3D"]
script = ExtResource("1_main")

[node name="Terrain" parent="." instance=ExtResource("2_terrain")]
""",
        ),
        Lesson(
            LAYER,
            "Player neden ayrı bir sahne (player.tscn) olur?",
            "Player'ı kodla üretmek yerine ayrı bir `.tscn` yapmak üç şey kazandırır: "
            "(1) yeniden kullanım — aynı sahne 4 oyuncu için 4 kez örneklenir; "
            "(2) editörde görsel düzenleme; (3) `@export` ile yarıçap/renk gibi "
            "değerlerin sahneden ayarlanması. Kök düğüm bir script taşır; script "
            "`_ready()` içinde görseli ölçekler ve `yuzeye_yerles()` gibi yardımcılar "
            "sunar. Sahne içinde `MeshInstance3D` alt düğümü hazır durur; script onu "
            "`get_node_or_null(\"Mesh\")` ile bulur, yoksa kendisi üretir.",
            ["player.tscn", "@export", "instantiate", "yeniden kullanım"],
            """# player.gd
extends Node3D

@export var radius: float = 0.6
@export var renk: Color = Color(1.0, 0.85, 0.15, 1.0)
@export var oyuncu_mu: bool = true

var mesh: MeshInstance3D = null

func _ready() -> void:
    mesh = get_node_or_null("Mesh") as MeshInstance3D
    if mesh == null:
        mesh = _mesh_olustur()
    mesh.scale = Vector3.ONE * radius

func yuzeye_yerles(yuzey_y: float) -> void:
    position.y = yuzey_y + radius
""",
        ),
        Lesson(
            LAYER,
            "Terrain'i ana sahneye ekleyip koddan kullanmak",
            "Terrain'i `main.tscn` içine `PackedScene` olarak instance etmek, "
            "yüklemeyi koddan sahneye taşır: editörde görünür, `get_node(\"Terrain\")` "
            "ile erişilir. Ancak oyun kodu mesh'i doğrudan bekliyorsa (ör. "
            "`terrain_mesh`) koddan mesh'i bulup ana sahneye taşımak gerekir: "
            "önce `_ilk_mesh()` ile ağaçta ilk `MeshInstance3D` bulunur, sonra "
            "`remove_child` + `add_child` ile köke alınır ve adı sabitlenir. Böylece "
            "hem sahne dosyası düzenlenebilir kalır hem de eski kod yolu bozulmaz.",
            ["terrain", "PackedScene", "_ilk_mesh", "mesh taşıma"],
            """func _load_blender_terrain() -> bool:
    var sahne_arazi := get_node_or_null("Terrain")
    if sahne_arazi != null:
        var mi: MeshInstance3D = _ilk_mesh(sahne_arazi)
        if mi != null and mi.mesh != null:
            var eski := mi.get_parent()
            if eski != null and eski != self:
                eski.remove_child(mi)
            mi.name = "Arazi"
            mi.owner = null
            add_child(mi)
            terrain_mesh = mi
            return _sample_terrain_heights(mi.mesh)
    return false

func _ilk_mesh(kok: Node) -> MeshInstance3D:
    if kok is MeshInstance3D:
        return kok
    for c in kok.get_children():
        var bulunan: MeshInstance3D = _ilk_mesh(c)
        if bulunan != null:
            return bulunan
    return null
""",
        ),
        Lesson(
            LAYER,
            "Player'ları terrain yüksekliğine göre spawn etmek",
            "Spawn, rastgele bir Y vermek değil **arazinin o noktadaki yüksekliğini "
            "okumak** demektir. Adımlar: (1) her oyuncu için (x, z) seç; (2) "
            "`_zemin_y(x, z)` ile yüksekliği al (Blender mesh'inden örneklenen "
            "ızgara ya da prosedürel formül); (3) merkez Y = yükseklik + yarıçap, "
            "böylece balon zemine oturur; (4) düğümü sahne ağacına ekle. "
            "Birden çok oyuncu için ofset listesi kullan ve hepsini bir konteyner "
            "düğümde topla (`Oyuncular`) — test ve temizlik kolaylaşır.",
            ["spawn", "zemin_y", "raycast", "ofset", "konteyner"],
            """const PLAYER_COUNT := 4
const SPAWN_OFFSETS := [
    Vector2(0.0, 0.0), Vector2(7.0, 5.0), Vector2(-8.0, 6.0), Vector2(4.0, -9.0),
]

func _spawn_players() -> void:
    var sahne := load("res://player.tscn") as PackedScene
    var konteyner := get_node_or_null("Oyuncular") as Node3D
    if konteyner == null:
        konteyner = Node3D.new()
        konteyner.name = "Oyuncular"
        add_child(konteyner)
    for i in PLAYER_COUNT:
        var ofs: Vector2 = SPAWN_OFFSETS[i]
        var yuzey := _zemin_y(ofs.x, ofs.y)
        var dugum := sahne.instantiate() as Node3D
        dugum.name = "Oyuncu%d" % (i + 1)
        dugum.position = Vector3(ofs.x, yuzey + player_radius, ofs.y)
        konteyner.add_child(dugum)
""",
        ),
        Lesson(
            LAYER,
            "Sahne ve spawn'ı headless test ile doğrulamak",
            "Godot'ta `SceneTree` türevi bir script, sahneyi yükleyip birkaç kare "
            "bekleyerek `_ready()` sonrası durumu ölçebilir. Önemli nokta: "
            "`root.add_child()` `_ready()`'yi **hemen** çağırmaz; bir kare beklemek "
            "gerekir. Test her kontrol için `PASS/FAIL` yazmalı ve hata varsa "
            "`quit(1)` ile çıkmalıdır; böylece CI ve betikler çıkış kodunu okuyabilir. "
            "Sayısal tolerans kullan (ör. `absf(fark) <= 0.05`), birebir eşitlik arama.",
            ["headless", "SceneTree", "process_frame", "PASS/FAIL", "exit code"],
            """extends SceneTree

var _hata := 0

func _ok(kosul: bool, mesaj: String) -> void:
    print(("  [PASS] " if kosul else "  [FAIL] ") + mesaj)
    if not kosul:
        _hata += 1

func _init() -> void:
    var sahne := load("res://main.tscn") as PackedScene
    var kok := sahne.instantiate()
    root.add_child(kok)
    await process_frame
    await process_frame
    var kont := kok.get_node_or_null("Oyuncular")
    _ok(kont != null, "Oyuncular konteyneri var")
    _ok(kont.get_child_count() == 4, "4 oyuncu üretildi")
    for c in kont.get_children():
        var beklenen: float = kok._zemin_y(c.position.x, c.position.z) + 1.0
        _ok(absf(c.position.y - beklenen) <= 0.05, "%s araziye oturmuş" % c.name)
    quit(0 if _hata == 0 else 1)
""",
        ),
        Lesson(
            LAYER,
            "GDScript: Variant'tan tip çıkarımı neden parse hatası verir?",
            "`var x := dugum.get(\"radius\")` yazınca `get()` `Variant` döndürdüğü "
            "için tip çıkarımı **Variant** olur. Proje ayarlarında "
            "`debug/gdscript/warnings/inference_on_variant` uyarısı **hata** olarak "
            "işaretliyse (Godot 4 varsayılanı) script derlenmez ve şu hata çıkar: "
            "\"The variable type is being inferred from a Variant value\". "
            "Çözüm: ya tipi açıkça yaz (`var x: float = float(dugum.get(...))`), "
            "ya `Variant` olarak işaretle (`var x: Variant = ...`), ya da `:=` "
            "yerine `=` kullan. Bu hata sahneyi değil **scripti** bozar; sahne "
            "yüklenmiş görünse de script çalışmaz.",
            ["Variant", "tip çıkarımı", "parse error", "uyarı=hata"],
            """# HATALI
var oyuncu_betik := dugum.get("radius")      # Variant -> parse error

# DOĞRU (açık tip)
var oyuncu_betik: Variant = dugum.get("radius")
if oyuncu_betik != null:
    r = float(oyuncu_betik)
""",
        ),
        Lesson(
            LAYER,
            "Ajan disiplini: okuma yapmak yazmak değildir",
            "Ölçülen bir başarısızlık: ajan 30 dakika boyunca yalnızca "
            "`read_file`, `find_files`, `search_text` çağırdı ve **hiç yazmadı**; "
            "sonunda \"doğrulanamadı\" dedi. Kural: her görev için (1) kabul "
            "kriterlerini yaz (dosya + ölçüm), (2) en fazla 5 okumadan sonra "
            "**ilk yazmayı** yap, (3) yazdıktan sonra derle/çalıştır, (4) ölçüm "
            "sonucunu raporla. Okuma sayısı 10'u geçtiyse dur ve planı yeniden kur; "
            "daha fazla okumak bilgi eklemez, süre tüketir.",
            ["ajan", "kabul kriteri", "yazma", "zaman yönetimi"],
            """Kabul kriteri örneği (Balon 3D):
1. player.tscn dosyası var ve MeshInstance3D taşıyor
2. main.tscn içinde Terrain düğümü var
3. Oyuncular konteynerinde 4 düğüm var
4. Her oyuncu |y - (zemin + yaricap)| <= 0.05
5. Headless test çıkış kodu 0
""",
        ),
    ]
