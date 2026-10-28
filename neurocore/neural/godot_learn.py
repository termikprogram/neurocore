"""
Godot Oyun Üretim Müfredatı
============================
Yerel modelin **Godot ile sıfırdan oyun üretmesi** için katmanlı içerik.

Kapsam (kullanıcı hedefi):
    1. Stylized lowpoly nesneler — kaya, masa, sandalye, ahşap ev, kapı,
       pencere, modular yapı, bitki, uçan kaya, buz kütlesi, interactable
    2. Texture / material sistemleri
    3. Karakter + monster, armature (iskelet), animasyon
    4. Master server + proxy server ağ mimarisi
    5. Seviye tasarımı — maze (labirent), arena, mağara
    6. Girdi sistemleri — tuş atama, WASD

Katmanlar:
    godot_temel     — Sahne, düğüm, script, sinyal, yaşam döngüsü
    godot_3b        — Node3D, MeshInstance3D, mesh üretimi, primitifler
    godot_nesne     — Stylized nesneler (mobilya, yapı, doğa, buz)
    godot_materyal  — Material, texture, StandardMaterial3D, shader
    godot_karakter  — Karakter, iskelet (armature), animasyon, monster
    godot_seviye    — Maze, arena, mağara, prosedürel seviye
    godot_girdi     — Input map, WASD, tuş atama, hareket
    godot_arayuz    — UI, HUD, envanter, menü
    godot_ses       — AudioStreamPlayer, müzik, efekt
    godot_ag        — Master/proxy server, RPC, multiplayer

⚠ Tüm kod **gerçek Godot 4 API'sidir** ve Python sözdizimi denetiminden
geçirilebilir olması için GDScript blokları string olarak tutulur.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from neurocore.neural.curriculum import Lesson

#: Godot müfredatı katmanları (sıra öğretim sırasıdır)
GODOT_LAYERS: Sequence[str] = (
    "godot_temel", "godot_3b", "godot_nesne", "godot_materyal",
    "godot_karakter", "godot_seviye", "godot_girdi", "godot_arayuz",
    "godot_ses", "godot_ag",
)

#: Katman etiketleri
GODOT_LABELS: Dict[str, str] = {
    "godot_temel": "Godot Temelleri",
    "godot_3b": "3B Dünya ve Mesh",
    "godot_nesne": "Stylized Lowpoly Nesneler",
    "godot_materyal": "Material ve Texture",
    "godot_karakter": "Karakter, İskelet ve Animasyon",
    "godot_seviye": "Seviye Tasarımı (Maze/Arena/Mağara)",
    "godot_girdi": "Girdi ve Tuş Atamaları",
    "godot_arayuz": "Arayüz (UI/HUD)",
    "godot_ses": "Ses ve Müzik",
    "godot_ag": "Ağ (Master/Proxy Server)",
}


# ─── Katman 1: Godot Temelleri ──────────────────────────────────────────────

def _temel_lessons() -> List[Lesson]:
    """Sahne, düğüm, script, sinyal."""
    return [
        Lesson("godot_temel", "Godot'ta sahne (scene) ve düğüm (node) nedir?",
               "Godot'ta her şey **düğüm**dür. Düğümler ağaç yapısında birleşir: "
               "bir **kök** düğüm ve altındaki çocuklar bir **sahne** oluşturur. "
               "Sahne kaydedilir (`.tscn`) ve başka sahnelerin içine "
               "yerleştirilebilir — bu **kompozisyon** Godot'un gücüdür. "
               "Örnek: `Player` sahnesi kendi içinde `MeshInstance3D`, "
               "`CollisionShape3D`, `Camera3D` taşır ve `Level` sahnesine "
               "yerleştirilir. Düğüm türleri: `Node3D` (konum), "
               "`MeshInstance3D` (görsel), `CollisionShape3D` (çarpışma), "
               "`CharacterBody3D` (fizikli karakter).",
               ["sahne", "düğüm", "kompozisyon"]),
        Lesson("godot_temel", "Script yaşam döngüsü: _ready, _process, _physics_process",
               "GDScript'te dört kritik fonksiyon vardır: **`_init()`** nesne "
               "oluşturulurken (constructor), **`_ready()`** sahne ağacına "
               "girdiğinde **bir kez**, **`_process(delta)`** her karede "
               "(görsel), **`_physics_process(delta)`** sabit fizik "
               "adımında (60 Hz). **Fizik işlemleri `_physics_process`'te "
               "yapılmalıdır** — `_process`'te yapmak kare hızına bağlı "
               "tutarsızlık yaratır. `delta` saniye cinsinden geçen süredir; "
               "hız hesaplarında çarpılır.",
               ["_ready", "_process", "delta"], code=(
                   "extends Node3D\n"
                   "\n"
                   "var hiz := 5.0\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    print(\"Sahne hazir\")\n"
                   "\n"
                   "func _process(delta: float) -> void:\n"
                   "    # Gorsel isler (kare hizina bagli)\n"
                   "    pass\n"
                   "\n"
                   "func _physics_process(delta: float) -> void:\n"
                   "    # Fizik: sabit adim, tutarli\n"
                   "    position.x += hiz * delta")),
        Lesson("godot_temel", "Sinyal (signal) ile olay haberleşmesi",
               "**Sinyal**, bir düğümün olay yaymasıdır. Godot'un yerleşik "
               "sinyalleri vardır: `body_entered`, `pressed`, `timeout`, "
               "`tree_exited`. Kendi sinyalinizi de tanımlayabilirsiniz: "
               "`signal hasar_alindi(miktar)`. Bağlama: "
               "`dugum.sinyal.connect(fonksiyon)`. Bu **gevşek bağlılık** "
               "sağlar — gönderen, alıcıyı bilmez. Alternatif "
               "`await sinyal` ile beklemektir.",
               ["signal", "connect", "olay"], code=(
                   "extends Node3D\n"
                   "\n"
                   "signal hasar_alindi(miktar: int)\n"
                   "signal oldu\n"
                   "\n"
                   "var saglik := 100\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    hasar_alindi.connect(_on_hasar)\n"
                   "    oldu.connect(_on_oldu)\n"
                   "\n"
                   "func hasar_ver(miktar: int) -> void:\n"
                   "    saglik -= miktar\n"
                   "    hasar_alindi.emit(miktar)\n"
                   "    if saglik <= 0:\n"
                   "        oldu.emit()\n"
                   "\n"
                   "func _on_hasar(miktar: int) -> void:\n"
                   "    print(\"Hasar: \", miktar, \" kalan: \", saglik)\n"
                   "\n"
                   "func _on_oldu() -> void:\n"
                   "    queue_free()")),
        Lesson("godot_temel", "Düğüm bulma: get_node ve @onready",
               "Düğümlere erişim yolları: `get_node(\"Yol/Dugum\")`, kısa "
               "yazım `$Yol/Dugum`, sahne köküne göre `%BenzersizAd` "
               "(**unique name** — editörde işaretlenir). **`@onready`** "
               "ile düğüm referansı `_ready`'den önce alınır: "
               "`@onready var kamera := $Camera3D`. Yol **bağıl** "
               "(kendi konumundan) veya **mutlak** (`/root/...`) olabilir. "
               "Yanlış yol **null** döner ve hata verir — `is_instance_valid` "
               "ile kontrol edin.",
               ["get_node", "@onready", "unique name"], code=(
                   "extends Node3D\n"
                   "\n"
                   "@onready var kamera: Camera3D = $Camera3D\n"
                   "@onready var govde: MeshInstance3D = $Govde\n"
                   "@onready var benzersiz: Node = %Oyuncu\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    # Uc erisim yolu\n"
                   "    print(get_node(\"Camera3D\").name)\n"
                   "    print($Camera3D.name)\n"
                   "    print(%Oyuncu.name)\n"
                   "    if is_instance_valid(govde):\n"
                   "        govde.visible = true")),
        Lesson("godot_temel", "Sınıf (class_name) ve kalıtım",
               "`class_name` bir scripti **global tür** yapar: başka "
               "dosyalardan doğrudan kullanılır. Kalıtım `extends` ile: "
               "`extends CharacterBody3D`. Kendi taban sınıfınızı yazmak "
               "**kod tekrarını önler**: ortak sağlık/hasar mantığı "
               "`EntityBase`'te, oyuncu ve canavar ondan türer. "
               "**Sanal fonksiyonlar** (`func _hasar_uygula() -> void:`) "
               "alt sınıfta ezilebilir — çok biçimlilik sağlar.",
               ["class_name", "extends", "kalıtım"], code=(
                   "class_name EntityBase\n"
                   "extends CharacterBody3D\n"
                   "\n"
                   "signal oldu\n"
                   "\n"
                   "@export var maks_saglik := 100\n"
                   "var saglik := 0\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    saglik = maks_saglik\n"
                   "\n"
                   "func hasar_uygula(miktar: int) -> void:\n"
                   "    saglik = max(0, saglik - miktar)\n"
                   "    if saglik == 0:\n"
                   "        oldu.emit()\n"
                   "\n"
                   "func is_oyuncu() -> bool:\n"
                   "    return false")),
        Lesson("godot_temel", "@export ile editörden ayarlanabilir değişken",
               "**`@export`** bir değişkeni editörde **görünür** yapar — "
               "kod yazmadan ayarlanır, sahne başına farklı değer alır. "
               "Türler: `int`, `float`, `bool`, `String`, `Vector3`, "
               "`Color`, `Array`, düğüm yolları, kaynaklar (`Mesh`, "
               "`Material`). Aralık: `@export_range(0, 100) var hiz := 50`. "
               "Gruplama: `@export_group(\"Hareket\")`. Bu, **tasarımcı "
               "dostu** kod yazmanın yoludur.",
               ["@export", "editör", "ayar"], code=(
                   "extends Node3D\n"
                   "\n"
                   "@export_group(\"Hareket\")\n"
                   "@export var yurume_hizi := 5.0\n"
                   "@export_range(0.0, 30.0) var kosma_hizi := 12.0\n"
                   "@export var ziplayacagi_yukseklik := 2.0\n"
                   "\n"
                   "@export_group(\"Gorsel\")\n"
                   "@export var govde_rengi := Color(0.3, 0.6, 0.3)\n"
                   "@export var kullan_materyal: Material\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    print(\"Hiz: \", yurume_hizi, \" Kosma: \", kosma_hizi)")),
        Lesson("godot_temel", "Sahne değiştirme ve otoload (singleton)",
               "**Sahne değiştirme:** `get_tree().change_scene_to_file(\"res://"
               "seviye_2.tscn\")` veya daha esnek "
               "`change_scene_to_packed(packed)`. **Otoload** (Proje → "
               "Proje Ayarları → Otoload): bir script/sahne **global** "
               "erişilebilir olur — oyun durumu, ağ yöneticisi, ses "
               "yöneticisi için idealdir. `Node` türünde olmalı ve sahne "
               "ağacının kökünde durur; `get_node(\"/root/Oyun\")` veya "
               "doğrudan adıyla erişilir.",
               ["change_scene", "otoload", "singleton"], code=(
                   "# otoload: OyunDurumu (Node)\n"
                   "extends Node\n"
                   "\n"
                   "var skor := 0\n"
                   "var seviye := 1\n"
                   "\n"
                   "func seviye_yukle(yeni: int) -> void:\n"
                   "    seviye = yeni\n"
                   "    var yol := \"res://seviyeler/seviye_%d.tscn\" % yeni\n"
                   "    get_tree().change_scene_to_file(yol)\n"
                   "\n"
                   "func skor_ekle(miktar: int) -> void:\n"
                   "    skor += miktar\n"
                   "    print(\"Skor: \", skor)")),
        Lesson("godot_temel", "Hata ayıklama: print, assert, breakpoint",
               "Araçlar: **`print()`** (`print_rich` renkli), "
               "**`push_error()`** / **`push_warning()`** (konsolda "
               "vurgulu, `print`'ten iyidir), **`assert(kosul, \"mesaj\")`** "
               "(yalnızca debug derlemesinde çalışır), **`breakpoint`** "
               "anahtar sözcüğü (hata ayıklayıcıyı durdurur). Uzak "
               "hata ayıklama: editörde **Hata Ayıklayıcı → Küçük "
               "Hata Ayıklayıcı**. **Performans:** `print` her karede "
               "çağrılırsa yavaşlatır — koşullu yazdırın.",
               ["push_error", "assert", "hata ayıklama"], code=(
                   "extends Node3D\n"
                   "\n"
                   "var sayac := 0\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    sayac += 1\n"
                   "    assert(sayac > 0, \"Sayac sifir olmamali\")\n"
                   "    if sayac > 100:\n"
                   "        push_warning(\"Sayac yuksek: %d\" % sayac)\n"
                   "    if sayac < 0:\n"
                   "        push_error(\"Sayac negatif olamaz!\")\n"
                   "    print_rich(\"[color=green]Hazir[/color] \", sayac)")),
    ]


# ─── Katman 2: 3B Dünya ve Mesh ─────────────────────────────────────────────

def _ucb_lessons() -> List[Lesson]:
    """Node3D, MeshInstance3D, prosedürel mesh."""
    return [
        Lesson("godot_3b", "Node3D dönüşümü: position, rotation, scale",
               "Her `Node3D` **yerel dönüşüm** taşır: `position` (Vector3, "
               "metre), `rotation` (radyan, derece için `rotation_degrees`), "
               "`scale` (çarpan). **Global** konum: `global_position`, "
               "`global_transform`. **Yerel ↔ global:** "
               "`to_global(yerel)`, `to_local(global)`. **Bakış:** "
               "`look_at(hedef)` nesneyi hedefe çevirir (yukarı vektörü "
               "varsayılan `Vector3.UP`). Ebeveyn döndüğünde çocuklar "
               "**birlikte** döner — hiyerarşi önemlidir.",
               ["position", "rotation", "transform"], code=(
                   "extends Node3D\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    position = Vector3(0, 2, 0)\n"
                   "    rotation_degrees = Vector3(0, 45, 0)\n"
                   "    scale = Vector3(1, 2, 1)\n"
                   "    # Global konum\n"
                   "    print(global_position)\n"
                   "    # Bir hedefe bak\n"
                   "    var hedef := Vector3(10, 0, 10)\n"
                   "    look_at(hedef, Vector3.UP)\n"
                   "    # Yerel <-> global cevirim\n"
                   "    var yerel := to_local(hedef)\n"
                   "    var glob := to_global(yerel)")),
        Lesson("godot_3b", "MeshInstance3D ve hazır primitifler",
               "**`MeshInstance3D`** bir **mesh kaynağını** (`Mesh`) "
               "görüntüler. Hazır primitifler: `BoxMesh`, `SphereMesh` "
               "(`radial_segments`, `rings`), `CylinderMesh` "
               "(`radial_segments` = kenar sayısı!), `CapsuleMesh`, "
               "`TorusMesh`, `PlaneMesh`, `PrismMesh`, `QuadMesh`. "
               "**Lowpoly için `radial_segments` düşük tutulur**: silindir "
               "6-8, küre 8×6. **`surface_material_override`** ile "
               "malzeme atanır. Mesh'i **kodla** üretmek prosedürel "
               "varlık demektir.",
               ["MeshInstance3D", "CylinderMesh", "radial_segments"], code=(
                   "extends Node3D\n"
                   "\n"
                   "func lowpoly_silindir(kenar: int, yaricap: float,\n"
                   "                      yukseklik: float) -> MeshInstance3D:\n"
                   "    var mi := MeshInstance3D.new()\n"
                   "    var m := CylinderMesh.new()\n"
                   "    m.radial_segments = kenar      # 6 = lowpoly\n"
                   "    m.top_radius = yaricap\n"
                   "    m.bottom_radius = yaricap * 1.6   # tabana kalin\n"
                   "    m.height = yukseklik\n"
                   "    mi.mesh = m\n"
                   "    return mi\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    var govde := lowpoly_silindir(6, 0.15, 2.0)\n"
                   "    add_child(govde)")),
        Lesson("godot_3b", "ArrayMesh ile sıfırdan mesh üretimi",
               "**`ArrayMesh`** ham köşe verisiyle mesh üretir — en esnek "
               "yol. `SurfaceTool` daha kolaydır: "
               "`begin(Mesh.PRIMITIVE_TRIANGLES)`, `set_normal`, "
               "`set_uv`, `add_vertex`, sonra `commit()`. "
               "**Dikdörtgen/üçgen** eklemek için köşe sırası **saat "
               "yönünün tersi** olmalıdır (aksi hâlde yüz görünmez). "
               "`generate_normals()` ve `generate_tangents()` otomatik "
               "normal üretir. Bu yöntem **prosedürel dünya** (arazi, "
               "mağara, labirent) için zorunludur.",
               ["ArrayMesh", "SurfaceTool", "add_vertex"], code=(
                   "extends Node3D\n"
                   "\n"
                   "func ucgen_mesh() -> ArrayMesh:\n"
                   "    var st := SurfaceTool.new()\n"
                   "    st.begin(Mesh.PRIMITIVE_TRIANGLES)\n"
                   "    # Saat yonunun TERSI sira (yuz gorunsun)\n"
                   "    st.set_normal(Vector3.UP)\n"
                   "    st.add_vertex(Vector3(0, 0, 0))\n"
                   "    st.add_vertex(Vector3(1, 0, 0))\n"
                   "    st.add_vertex(Vector3(0, 0, 1))\n"
                   "    st.generate_normals()\n"
                   "    return st.commit()\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    var mi := MeshInstance3D.new()\n"
                   "    mi.mesh = ucgen_mesh()\n"
                   "    add_child(mi)")),
        Lesson("godot_3b", "Grid tabanlı arazi (terrain) üretimi",
               "Arazi = düzenli köşe ızgarası + yükseklik. Adımlar: "
               "**(1)** `GRID×GRID` köşe üret, **(2)** her köşenin Y'sini "
               "gürültüden hesapla, **(3)** üçgenleri kur (iki üçgen = bir "
               "kare), **(4)** `SurfaceTool.generate_normals()`. "
               "**FastNoiseLite** Godot'un yerleşik gürültüsüdür: "
               "`noise_type`, `frequency`, `fractal_octaves` ayarlanır. "
               "**Ölçek:** grid 128×128 = 16.384 üçgen (kabul edilebilir); "
               "Chunk'lara bölmek LOD sağlar.",
               ["terrain", "FastNoiseLite", "grid"], code=(
                   "extends MeshInstance3D\n"
                   "\n"
                   "@export var grid := 64\n"
                   "@export var boyut := 40.0\n"
                   "@export var yukseklik := 6.0\n"
                   "\n"
                   "func arazi_uret() -> ArrayMesh:\n"
                   "    var n := FastNoiseLite.new()\n"
                   "    n.noise_type = FastNoiseLite.TYPE_SIMPLEX\n"
                   "    n.frequency = 0.03\n"
                   "    n.fractal_octaves = 4\n"
                   "    var st := SurfaceTool.new()\n"
                   "    st.begin(Mesh.PRIMITIVE_TRIANGLES)\n"
                   "    var adim := boyut / float(grid - 1)\n"
                   "    for j in range(grid - 1):\n"
                   "        for i in range(grid - 1):\n"
                   "            var x0 := i * adim - boyut * 0.5\n"
                   "            var z0 := j * adim - boyut * 0.5\n"
                   "            var x1 := x0 + adim\n"
                   "            var z1 := z0 + adim\n"
                   "            var h00 := n.get_noise_2d(x0, z0) * yukseklik\n"
                   "            var h10 := n.get_noise_2d(x1, z0) * yukseklik\n"
                   "            var h01 := n.get_noise_2d(x0, z1) * yukseklik\n"
                   "            var h11 := n.get_noise_2d(x1, z1) * yukseklik\n"
                   "            var a := Vector3(x0, h00, z0)\n"
                   "            var b := Vector3(x1, h10, z0)\n"
                   "            var c := Vector3(x1, h11, z1)\n"
                   "            var d := Vector3(x0, h01, z1)\n"
                   "            st.add_vertex(a); st.add_vertex(b); st.add_vertex(c)\n"
                   "            st.add_vertex(a); st.add_vertex(c); st.add_vertex(d)\n"
                   "    st.generate_normals()\n"
                   "    return st.commit()\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    mesh = arazi_uret()")),
        Lesson("godot_3b", "Mesh birleştirme ve yüz sayısı ölçümü",
               "Yüz sayısı: `mesh.get_surface_count()`, "
               "`mesh.surface_get_arrays(0)[Mesh.ARRAY_VERTEX].size()`. "
               "**Üçgen sayısı** = köşe sayısı / 3 (indekssiz). "
               "**Birleştirme:** birden fazla `MeshInstance3D` yerine tek "
               "mesh kullanmak **draw call** azaltır. `ArrayMesh`'leri "
               "elle birleştirmek için `SurfaceTool.append_from(mesh, 0, "
               "transform)`. Performans: mobilde **hedef 100.000 üçgen/kare**; "
               "500 taş = 500 draw call yerine tek mesh = 1 draw call.",
               ["yüz sayısı", "draw call", "append_from"], code=(
                   "extends Node3D\n"
                   "\n"
                   "func yuz_sayisi(mi: MeshInstance3D) -> int:\n"
                   "    if mi.mesh == null:\n"
                   "        return 0\n"
                   "    var toplam := 0\n"
                   "    for s in range(mi.mesh.get_surface_count()):\n"
                   "        var dizi: Array = mi.mesh.surface_get_arrays(s)\n"
                   "        toplam += dizi[Mesh.ARRAY_VERTEX].size() / 3\n"
                   "    return toplam\n"
                   "\n"
                   "func birlestir(meshlar: Array) -> ArrayMesh:\n"
                   "    var st := SurfaceTool.new()\n"
                   "    st.begin(Mesh.PRIMITIVE_TRIANGLES)\n"
                   "    for mi in meshlar:\n"
                   "        st.append_from(mi.mesh, 0, mi.transform)\n"
                   "    return st.commit()")),
    ]


# ─── Katman 3: Stylized Lowpoly Nesneler ────────────────────────────────────

def _nesne_lessons() -> List[Lesson]:
    """Kaya, mobilya, yapı, doğa, buz, interactable nesneler."""
    return [
        Lesson("godot_nesne", "Stylized lowpoly kaya nasıl üretilir?",
               "Kaya = **düzensizleştirilmiş küre**. Yöntem: "
               "`SphereMesh` (8×6) al, her köşeyi rastgele kaydır, "
               "sonra `flat shading` uygula. **Tohum** ile "
               "tekrarlanabilir çeşitlilik: `RandomNumberGenerator` "
               "(`seed` ayarlanır). Farklı kaya türleri **ölçek ve "
               "düzensizlik** ile: yassı (y=0.6), dik (y=1.5), sivri "
               "(tepe köşeleri dışa). **Yüz bütçesi 40-120** — kaya "
               "sahneye çok konur, ucuz olmalıdır.",
               ["kaya", "düzensizleştirme", "seed"], code=(
                   "extends MeshInstance3D\n"
                   "\n"
                   "@export var tohum := 42\n"
                   "@export var duzensizlik := 0.25\n"
                   "\n"
                   "func kaya_uret() -> ArrayMesh:\n"
                   "    var rng := RandomNumberGenerator.new()\n"
                   "    rng.seed = tohum\n"
                   "    var kure := SphereMesh.new()\n"
                   "    kure.radial_segments = 8\n"
                   "    kure.rings = 6\n"
                   "    kure.radius = 1.0\n"
                   "    var st := SurfaceTool.new()\n"
                   "    st.begin(Mesh.PRIMITIVE_TRIANGLES)\n"
                   "    var dizi: Array = kure.surface_get_arrays(0)\n"
                   "    var noktalar: PackedVector3Array = dizi[Mesh.ARRAY_VERTEX]\n"
                   "    var indeksler: PackedInt32Array = dizi[Mesh.ARRAY_INDEX]\n"
                   "    for i in range(0, indeksler.size(), 3):\n"
                   "        for k in 3:\n"
                   "            var v: Vector3 = noktalar[indeksler[i + k]]\n"
                   "            var d := 1.0 + rng.randf_range(-duzensizlik, duzensizlik)\n"
                   "            st.add_vertex(v * d)\n"
                   "    st.generate_normals()\n"
                   "    return st.commit()\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    mesh = kaya_uret()")),
        Lesson("godot_nesne", "Stylized masa ve sandalye (mobilya)",
               "Mobilya **dikdörtgen prizmalardan** kurulur — kutu en "
               "ucuz şekildir (12 üçgen). **Masa:** tabla (yassı kutu) + "
               "4 bacak (ince kutu). **Sandalye:** oturak + 4 bacak + "
               "sırt (dikey kutu). Yüz bütçesi: masa ~60, sandalye ~80. "
               "**Modular yaklaşım:** `BoxMesh` boyutlarını parametre "
               "yapıp aynı fonksiyondan farklı boyutlar üretmek. "
               "Ahşap görünümü **material** ile (kahverengi, yüksek "
               "roughness).",
               ["masa", "sandalye", "kutu"], code=(
                   "extends Node3D\n"
                   "\n"
                   "func kutu(boyut: Vector3, konum: Vector3,\n"
                   "          mat: Material) -> MeshInstance3D:\n"
                   "    var mi := MeshInstance3D.new()\n"
                   "    var m := BoxMesh.new()\n"
                   "    m.size = boyut\n"
                   "    mi.mesh = m\n"
                   "    mi.position = konum\n"
                   "    mi.material_override = mat\n"
                   "    return mi\n"
                   "\n"
                   "func masa(mat: Material) -> Node3D:\n"
                   "    var kok := Node3D.new()\n"
                   "    kok.add_child(kutu(Vector3(1.6, 0.08, 0.9),\n"
                   "                       Vector3(0, 0.75, 0), mat))   # tabla\n"
                   "    for k in [Vector3(-0.7, 0, -0.35), Vector3(0.7, 0, -0.35),\n"
                   "              Vector3(-0.7, 0, 0.35), Vector3(0.7, 0, 0.35)]:\n"
                   "        kok.add_child(kutu(Vector3(0.08, 0.75, 0.08),\n"
                   "                           k + Vector3(0, 0.375, 0), mat))\n"
                   "    return kok\n"
                   "\n"
                   "func sandalye(mat: Material) -> Node3D:\n"
                   "    var kok := Node3D.new()\n"
                   "    kok.add_child(kutu(Vector3(0.45, 0.06, 0.45),\n"
                   "                       Vector3(0, 0.45, 0), mat))   # oturak\n"
                   "    kok.add_child(kutu(Vector3(0.45, 0.5, 0.06),\n"
                   "                       Vector3(0, 0.7, -0.2), mat))  # sirt\n"
                   "    for k in [Vector3(-0.18, 0, -0.18), Vector3(0.18, 0, -0.18),\n"
                   "              Vector3(-0.18, 0, 0.18), Vector3(0.18, 0, 0.18)]:\n"
                   "        kok.add_child(kutu(Vector3(0.05, 0.45, 0.05),\n"
                   "                           k + Vector3(0, 0.225, 0), mat))\n"
                   "    return kok")),
        Lesson("godot_nesne", "Ahşap ev: modular duvar sistemi",
               "Ev **modular parçalardan** kurulur: duvar paneli (2×3×0.2), "
               "kapı boşluklu panel, pencere boşluklu panel, çatı "
               "(üçgen prizma), zemin. **Izgara (grid) yerleşimi:** "
               "her panel `boyut` aralığıyla yerleştirilir — hizalama "
               "otomatiktir. **Çatı** için `PrismMesh` (üçgen prizma) "
               "kullanılır. **Yüz bütçesi:** duvar 12, çatı 8, toplam "
               "küçük ev ~200. **Modular olması** aynı parçalardan "
               "farklı evler üretmeyi sağlar.",
               ["ev", "modular", "duvar paneli"], code=(
                   "extends Node3D\n"
                   "\n"
                   "@export var panel := Vector3(2.0, 3.0, 0.2)\n"
                   "@export var mat_ahsap: Material\n"
                   "@export var mat_cati: Material\n"
                   "\n"
                   "func panel_ekle(kok: Node3D, konum: Vector3) -> void:\n"
                   "    var mi := MeshInstance3D.new()\n"
                   "    var m := BoxMesh.new()\n"
                   "    m.size = panel\n"
                   "    mi.mesh = m\n"
                   "    mi.position = konum\n"
                   "    mi.material_override = mat_ahsap\n"
                   "    kok.add_child(mi)\n"
                   "\n"
                   "func ev(kok: Node3D, genislik: int, derinlik: int) -> void:\n"
                   "    for i in range(genislik):\n"
                   "        panel_ekle(kok, Vector3(i * panel.x, panel.y * 0.5, 0))\n"
                   "        panel_ekle(kok, Vector3(i * panel.x, panel.y * 0.5,\n"
                   "                                derinlik * panel.z))\n"
                   "    for j in range(derinlik):\n"
                   "        panel_ekle(kok, Vector3(0, panel.y * 0.5, j * panel.z))\n"
                   "        panel_ekle(kok, Vector3(genislik * panel.x,\n"
                   "                                panel.y * 0.5, j * panel.z))\n"
                   "    # Cati: ucgen prizma\n"
                   "    var cati := MeshInstance3D.new()\n"
                   "    var pm := PrismMesh.new()\n"
                   "    pm.size = Vector3(genislik * panel.x, 1.5, derinlik * panel.z)\n"
                   "    cati.mesh = pm\n"
                   "    cati.position = Vector3(genislik * panel.x * 0.5, panel.y + 0.75,\n"
                   "                            derinlik * panel.z * 0.5)\n"
                   "    cati.material_override = mat_cati\n"
                   "    kok.add_child(cati)")),
        Lesson("godot_nesne", "Kapı ve pencere: boşluklu panel",
               "**Kapı**, duvar panelinin **boşluklu** hâlidir: tam panel "
               "yerine 4 parça (üst lento, iki yan, alt eşik yok). "
               "**Pencere** aynı mantık: 4 parça + cam (saydam material). "
               "**Kapı açılması:** `AnimationPlayer` veya kod ile "
               "`rotation.y` değişimi; `Tween` ile yumuşak geçiş: "
               "`create_tween().tween_property(kapi, \"rotation:y\", "
               "PI/2, 0.4)`. **Etkileşim:** `Area3D` + `body_entered` "
               "sinyali ile oyuncu yaklaşınca aç.",
               ["kapı", "pencere", "Tween"], code=(
                   "extends Node3D\n"
                   "\n"
                   "@export var genislik := 1.0\n"
                   "@export var yukseklik := 2.1\n"
                   "@export var kalinlik := 0.15\n"
                   "@export var mat: Material\n"
                   "\n"
                   "func _parca(boyut: Vector3, konum: Vector3) -> MeshInstance3D:\n"
                   "    var mi := MeshInstance3D.new()\n"
                   "    var m := BoxMesh.new()\n"
                   "    m.size = boyut\n"
                   "    mi.mesh = m\n"
                   "    mi.position = konum\n"
                   "    mi.material_override = mat\n"
                   "    return mi\n"
                   "\n"
                   "func kapi_cercevesi() -> Node3D:\n"
                   "    var kok := Node3D.new()\n"
                   "    var yan := (genislik - 0.9) * 0.5\n"
                   "    kok.add_child(_parca(Vector3(yan, yukseklik, kalinlik),\n"
                   "                         Vector3(-0.45 - yan * 0.5, yukseklik * 0.5, 0)))\n"
                   "    kok.add_child(_parca(Vector3(yan, yukseklik, kalinlik),\n"
                   "                         Vector3(0.45 + yan * 0.5, yukseklik * 0.5, 0)))\n"
                   "    kok.add_child(_parca(Vector3(genislik, 0.3, kalinlik),\n"
                   "                         Vector3(0, yukseklik - 0.15, 0)))\n"
                   "    return kok\n"
                   "\n"
                   "func kapi_ac(kapi: Node3D) -> void:\n"
                   "    var tw := create_tween()\n"
                   "    tw.tween_property(kapi, \"rotation:y\", PI * 0.5, 0.4)")),
        Lesson("godot_nesne", "Bitkiler: çim, çalı, ağaç (Godot'ta)",
               "Godot'ta bitki üretimi Blender'dakiyle aynı mantık: "
               "**ağaç** = `CylinderMesh` (6 kenar, gövde) + `CylinderMesh` "
               "koniler (taç). **Çim** = çok sayıda ince üçgen/quad, "
               "`MultiMeshInstance3D` ile **tek draw call**'da binlerce "
               "kopya. **`MultiMesh`** kullanımı: `multimesh.mesh` ata, "
               "`instance_count` belirle, `set_instance_transform(i, "
               "transform)` ile yerleştir. Bu, **10.000 çimi tek çizimde** "
               "gösterir — mobil için kritik.",
               ["MultiMesh", "ağaç", "çim"], code=(
                   "extends MultiMeshInstance3D\n"
                   "\n"
                   "@export var adet := 500\n"
                   "@export var alan := 40.0\n"
                   "@export var tohum := 7\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    var rng := RandomNumberGenerator.new()\n"
                   "    rng.seed = tohum\n"
                   "    var tek := CylinderMesh.new()\n"
                   "    tek.radial_segments = 6\n"
                   "    tek.top_radius = 0.02\n"
                   "    tek.bottom_radius = 0.05\n"
                   "    tek.height = 0.6\n"
                   "    multimesh = MultiMesh.new()\n"
                   "    multimesh.transform_format = MultiMesh.TRANSFORM_3D\n"
                   "    multimesh.mesh = tek\n"
                   "    multimesh.instance_count = adet\n"
                   "    for i in range(adet):\n"
                   "        var t := Transform3D()\n"
                   "        t.origin = Vector3(rng.randf_range(-alan, alan), 0,\n"
                   "                           rng.randf_range(-alan, alan))\n"
                   "        t = t.rotated(Vector3.UP, rng.randf_range(0, TAU))\n"
                   "        t = t.scaled(Vector3.ONE * rng.randf_range(0.7, 1.4))\n"
                   "        multimesh.set_instance_transform(i, t)")),
        Lesson("godot_nesne", "Uçan kaya ve buz kütlesi (fantasy nesneler)",
               "**Uçan kaya**: kaya mesh'i + **animasyon**. Yöntemler: "
               "`_process`'te `sin` ile salınım "
               "(`position.y = taban + sin(zaman * hiz) * genlik`), "
               "`rotation` ile yavaş dönüş, `AnimationPlayer` ile "
               "karmaşık hareket. **Buz kütlesi**: `PrismMesh` veya "
               "düzensizleştirilmiş `BoxMesh`; material **saydam + "
               "yüksek parlaklık** (`transparency = TRANSPARENCY_ALPHA`, "
               "`albedo_color.a = 0.7`, `metallic`, `roughness` düşük). "
               "İçine `OmniLight3D` koymak **parlayan buz** etkisi verir.",
               ["uçan kaya", "buz", "animasyon"], code=(
                   "extends Node3D\n"
                   "\n"
                   "@export var genlik := 0.4\n"
                   "@export var hiz := 1.2\n"
                   "@export var donme_hizi := 0.3\n"
                   "\n"
                   "var _zaman := 0.0\n"
                   "var _taban_y := 0.0\n"
                   "var _tohum := 0.0\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    _taban_y = position.y\n"
                   "    _tohum = randf() * TAU\n"
                   "\n"
                   "func _process(delta: float) -> void:\n"
                   "    _zaman += delta\n"
                   "    position.y = _taban_y + sin(_zaman * hiz + _tohum) * genlik\n"
                   "    rotate_y(donme_hizi * delta)\n"
                   "\n"
                   "func buz_materyali() -> StandardMaterial3D:\n"
                   "    var m := StandardMaterial3D.new()\n"
                   "    m.albedo_color = Color(0.75, 0.88, 1.0, 0.72)\n"
                   "    m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA\n"
                   "    m.roughness = 0.08\n"
                   "    m.metallic = 0.25\n"
                   "    return m")),
        Lesson("godot_nesne", "Interactable (etkileşimli) nesne sistemi",
               "Etkileşim üç parçadan kurulur: **(1) algılama** — `Area3D` "
               "veya ışın (`RayCast3D`) ile oyuncunun yaklaşması/bakması; "
               "**(2) istem** — ekranda \"[E] Aç\" yazısı; **(3) eylem** — "
               "sinyal ile nesnenin tepki vermesi. **Ortak arayüz:** bir "
               "`Interactable` taban sınıfı (`Node3D`) yazın, tüm "
               "etkileşimli nesneler ondan türesin ve `etkilesim()` "
               "fonksiyonunu ezsin. **Sıralama:** aynı anda birden fazla "
               "nesne menzildeyse **en yakın** veya **en çok bakan** "
               "seçilir.",
               ["interactable", "RayCast3D", "Area3D"], code=(
                   "class_name Interactable\n"
                   "extends Node3D\n"
                   "\n"
                   "signal etkilesim_basladi(oyuncu: Node3D)\n"
                   "\n"
                   "@export var etiket := \"Ac\"\n"
                   "@export var menzil := 2.5\n"
                   "\n"
                   "func ipucu() -> String:\n"
                   "    return \"[E] \" + etiket\n"
                   "\n"
                   "func etkilesim(_oyuncu: Node3D) -> void:\n"
                   "    # Alt sinif ezer\n"
                   "    pass\n"
                   "\n"
                   "# Kullanim: Kapi.gd\n"
                   "# extends Interactable\n"
                   "#\n"
                   "# func etkilesim(oyuncu: Node3D) -> void:\n"
                   "#     etkilesim_basladi.emit(oyuncu)\n"
                   "#     kapi_ac()")),
    ]


# ─── Katman 4: Material ve Texture ──────────────────────────────────────────

def _materyal_lessons() -> List[Lesson]:
    """StandardMaterial3D, texture, shader."""
    return [
        Lesson("godot_materyal", "StandardMaterial3D temel ayarları",
               "**`StandardMaterial3D`** Godot'un varsayılan PBR "
               "malzemesidir. Kritik özellikler: **`albedo_color`** (temel "
               "renk), **`metallic`** (0 = yalıtkan, 1 = metal), "
               "**`roughness`** (0 = ayna, 1 = mat), "
               "**`emission`** (kendi ışıması — parlayan nesne), "
               "**`normal_enabled`** (normal haritası), "
               "**`cull_mode`** (yüz eleme — `CULL_DISABLED` çift taraflı). "
               "**Stylized için:** `roughness` 0,7-0,9, `metallic` 0, "
               "`specular` düşük. **`vertex_color_use_as_albedo`** ile "
               "köşe rengi kullanılır.",
               ["StandardMaterial3D", "albedo", "roughness"], code=(
                   "func stilli_materyal(renk: Color) -> StandardMaterial3D:\n"
                   "    var m := StandardMaterial3D.new()\n"
                   "    m.albedo_color = renk\n"
                   "    m.roughness = 0.85          # mat, stilize\n"
                   "    m.metallic = 0.0\n"
                   "    m.specular = 0.15           # az parlaklik\n"
                   "    m.vertex_color_use_as_albedo = true\n"
                   "    return m\n"
                   "\n"
                   "func parlayan_materyal(renk: Color) -> StandardMaterial3D:\n"
                   "    var m := StandardMaterial3D.new()\n"
                   "    m.albedo_color = renk\n"
                   "    m.emission_enabled = true\n"
                   "    m.emission = renk\n"
                   "    m.emission_energy_multiplier = 2.0\n"
                   "    return m")),
        Lesson("godot_materyal", "Doku (texture) yükleme ve ayarları",
               "Doku yükleme: `load(\"res://dokular/cimen.png\")` veya "
               "`preload()` (derleme zamanı, hızlı). Malzemeye atama: "
               "`albedo_texture`. **Ölçek/tekrar:** "
               "`uv1_scale = Vector3(8, 8, 1)` ile doku tekrarlanır. "
               "**Filtreleme:** `texture_filter = TEXTURE_FILTER_NEAREST` "
               "**pixel-art/stilize** görünüm verir (bulanıklaştırmaz); "
               "`LINEAR` yumuşak. **`texture_repeat = true`** ile doku "
               "döşenir. **Normal haritası** için "
               "`normal_enabled = true` + `normal_texture`.",
               ["texture", "uv1_scale", "TEXTURE_FILTER_NEAREST"], code=(
                   "func dokulu_materyal(yol: String, olcek := 8.0,\n"
                   "                     pixel_art := true) -> StandardMaterial3D:\n"
                   "    var m := StandardMaterial3D.new()\n"
                   "    var doku: Texture2D = load(yol)\n"
                   "    m.albedo_texture = doku\n"
                   "    m.uv1_scale = Vector3(olcek, olcek, 1.0)\n"
                   "    m.texture_repeat = true\n"
                   "    if pixel_art:\n"
                   "        m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST\n"
                   "    m.roughness = 0.9\n"
                   "    return m\n"
                   "\n"
                   "func normal_haritali(renk: Color, normal_yolu: String) -> StandardMaterial3D:\n"
                   "    var m := StandardMaterial3D.new()\n"
                   "    m.albedo_color = renk\n"
                   "    m.normal_enabled = true\n"
                   "    m.normal_texture = load(normal_yolu)\n"
                   "    m.normal_scale = 1.5\n"
                   "    m.roughness = 0.8\n"
                   "    return m")),
        Lesson("godot_materyal", "Vertex color ile katman karıştırma",
               "Arazi katmanları için **vertex color** en verimli yoldur: "
               "0 texture, 0 ek draw call. `StandardMaterial3D`'de "
               "`vertex_color_use_as_albedo = true` açılır; renkler "
               "**mesh köşelerinde** saklanır. **Katman ağırlıkları** "
               "shader'da karıştırılır: 4 ağırlık kanalı (RGBA) → 4 doku "
               "→ ağırlıklı toplam. **Godot'ta shader** yazmak için "
               "`ShaderMaterial` + `.gdshader` dosyası. Basit yol: "
               "**ağırlıkları CPU'da karıştırıp** sonucu vertex color'a "
               "yazmak (vertex color bake).",
               ["vertex color", "katman", "shader"], code=(
                   "# shader: terrain_karisim.gdshader\n"
                   "# shader_type spatial;\n"
                   "# uniform sampler2D kum : source_color;\n"
                   "# uniform sampler2D cimen : source_color;\n"
                   "# uniform sampler2D kaya : source_color;\n"
                   "# uniform sampler2D kar : source_color;\n"
                   "#\n"
                   "# void fragment() {\n"
                   "#     vec2 uv2 = UV * 10.0;\n"
                   "#     float wKum = COLOR.a;\n"
                   "#     float wCimen = COLOR.g;\n"
                   "#     float wKaya = COLOR.b;\n"
                   "#     float wKar = COLOR.r;\n"
                   "#     vec3 renk = texture(kum, uv2).rgb * wKum;\n"
                   "#     renk += texture(cimen, uv2).rgb * wCimen;\n"
                   "#     renk += texture(kaya, uv2).rgb * wKaya;\n"
                   "#     renk += texture(kar, uv2).rgb * wKar;\n"
                   "#     ALBEDO = renk;\n"
                   "#     ROUGHNESS = 0.9;\n"
                   "# }\n"
                   "\n"
                   "func shader_materyal(yol: String) -> ShaderMaterial:\n"
                   "    var sm := ShaderMaterial.new()\n"
                   "    sm.shader = load(yol)\n"
                   "    return sm")),
        Lesson("godot_materyal", "Saydamlık, kesme ve sıralama",
               "**Saydamlık türleri:** `TRANSPARENCY_DISABLED` (opak, en "
               "hızlı), `TRANSPARENCY_ALPHA` (yumuşak saydam — su, cam), "
               "`TRANSPARENCY_ALPHA_SCISSOR` (keskin kenar — yaprak, "
               "çim; **gölge verir**), `TRANSPARENCY_ALPHA_HASH` "
               "(titremeli, gölge verir). **Kesme** (`scissor`) "
               "**yapraklar için doğrudur** — `alpha_scissor_threshold` "
               "ile eşik ayarlanır. **Sıralama:** "
               "`render_priority` ve `depth_draw_mode` ile. "
               "**Çift taraflı:** `cull_mode = CULL_DISABLED` (yaprak "
               "düzlemi için şart).",
               ["saydamlık", "scissor", "yaprak"], code=(
                   "func yaprak_materyali(doku: String) -> StandardMaterial3D:\n"
                   "    var m := StandardMaterial3D.new()\n"
                   "    m.albedo_texture = load(doku)\n"
                   "    m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR\n"
                   "    m.alpha_scissor_threshold = 0.5\n"
                   "    m.cull_mode = BaseMaterial3D.CULL_DISABLED\n"
                   "    m.vertex_color_use_as_albedo = true\n"
                   "    return m\n"
                   "\n"
                   "func su_materyali() -> StandardMaterial3D:\n"
                   "    var m := StandardMaterial3D.new()\n"
                   "    m.albedo_color = Color(0.1, 0.35, 0.6, 0.75)\n"
                   "    m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA\n"
                   "    m.roughness = 0.05\n"
                   "    m.metallic = 0.3\n"
                   "    m.rim_enabled = true\n"
                   "    return m")),
    ]


# ─── Katman 5: Karakter, İskelet ve Animasyon ───────────────────────────────

def _karakter_lessons() -> List[Lesson]:
    """Karakter, armature (Skeleton3D), animasyon, monster."""
    return [
        Lesson("godot_karakter", "Karakter gövdesi: CharacterBody3D kurulumu",
               "**`CharacterBody3D`** fizik tabanlı karakterdir: "
               "`velocity` ayarlanır ve `move_and_slide()` çağrılır — "
               "çarpışma, eğim, zemin otomatik çözülür. **Zemin kontrolü:** "
               "`is_on_floor()`. **Yerçekimi:** "
               "`velocity.y -= yerçekimi * delta` (yerçekimi ~9,8–24). "
               "**Zıplama:** `if is_on_floor() and Input.is_action_just_"
               "pressed(\"zipla\"): velocity.y = sqrt(2 * yerçekimi * "
               "yukseklik)`. **`up_direction`** varsayılan `Vector3.UP`. "
               "Gövde **CollisionShape3D** (kapsül) gerektirir.",
               ["CharacterBody3D", "move_and_slide", "yerçekimi"], code=(
                   "extends CharacterBody3D\n"
                   "\n"
                   "@export var yurume := 5.0\n"
                   "@export var kosma := 9.0\n"
                   "@export var yer_cekimi := 18.0\n"
                   "@export var ziplama_yuksekligi := 1.8\n"
                   "\n"
                   "func _physics_process(delta: float) -> void:\n"
                   "    if not is_on_floor():\n"
                   "        velocity.y -= yer_cekimi * delta\n"
                   "    if Input.is_action_just_pressed(\"zipla\") and is_on_floor():\n"
                   "        velocity.y = sqrt(2.0 * yer_cekimi * ziplama_yuksekligi)\n"
                   "    var giris := Input.get_vector(\"sol\", \"sag\", \"ileri\", \"geri\")\n"
                   "    var yon := (transform.basis * Vector3(giris.x, 0, giris.y)).normalized()\n"
                   "    var hiz := kosma if Input.is_action_pressed(\"kos\") else yurume\n"
                   "    if yon:\n"
                   "        velocity.x = yon.x * hiz\n"
                   "        velocity.z = yon.z * hiz\n"
                   "    else:\n"
                   "        velocity.x = move_toward(velocity.x, 0, hiz)\n"
                   "        velocity.z = move_toward(velocity.z, 0, hiz)\n"
                   "    move_and_slide()")),
        Lesson("godot_karakter", "Kameraya göre hareket (üçüncü şahıs)",
               "Karakter hareketi **kameraya göre** olmalıdır, dünyaya göre "
               "değil. Yöntem: kamera yönünü al "
               "(`kamera.global_transform.basis`), giriş vektörünü bu "
               "tabana çevir. **Kamera döndürme:** fare hareketi ile "
               "`rotate_y` (karakter gövdesi) ve `rotate_x` (kamera "
               "yüksekliği, sınırlı `clamp(-1.2, 0.6)`). **Yumuşatma:** "
               "`lerp` ile kamerayı hedefe yaklaştırmak titremeyi önler. "
               "`Input.MOUSE_MODE_CAPTURED` fareyi kilitler.",
               ["kamera", "üçüncü şahıs", "fare"], code=(
                   "extends Node3D\n"
                   "\n"
                   "@export var fare_hassasiyeti := 0.003\n"
                   "@export var kamera_mesafesi := 4.0\n"
                   "\n"
                   "@onready var yay := $SpringArm3D\n"
                   "@onready var kamera := $SpringArm3D/Camera3D\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    Input.mouse_mode = Input.MOUSE_MODE_CAPTURED\n"
                   "\n"
                   "func _unhandled_input(olay: InputEvent) -> void:\n"
                   "    if olay is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:\n"
                   "        rotate_y(-olay.relative.x * fare_hassasiyeti)\n"
                   "        yay.rotation.x = clamp(\n"
                   "            yay.rotation.x - olay.relative.y * fare_hassasiyeti,\n"
                   "            -1.2, 0.6)\n"
                   "    if olay.is_action_pressed(\"kamera_serbest\"):\n"
                   "        Input.mouse_mode = Input.MOUSE_MODE_VISIBLE")),
        Lesson("godot_karakter", "İskelet (Skeleton3D) ve kemik yapısı",
               "**`Skeleton3D`** kemik hiyerarşisini tutar; "
               "**`MeshInstance3D`** ile **skin** üzerinden bağlanır. "
               "Kemik erişimi: `skeleton.find_bone(\"kol_ust\")` → indeks, "
               "`skeleton.get_bone_pose_position(idx)`, "
               "`set_bone_pose_rotation(idx, quat)`. **Kemik sayısı:** "
               "basit karakter 15-25, detaylı 40+. **Blender'dan "
               "aktarım:** GLTF/GLB formatı iskeleti ve ağırlıkları "
               "korur — `export_scene.gltf` ile `.glb` alınır. "
               "**Skin ağırlıkları** Blender'da atanır, Godot'ta hazır "
               "gelir.",
               ["Skeleton3D", "kemik", "GLTF"], code=(
                   "extends Skeleton3D\n"
                   "\n"
                   "func kemik_bul(ad: String) -> int:\n"
                   "    return find_bone(ad)\n"
                   "\n"
                   "func kemik_dondur(ad: String, derece: Vector3) -> void:\n"
                   "    var idx := find_bone(ad)\n"
                   "    if idx < 0:\n"
                   "        push_warning(\"Kemik yok: \" + ad)\n"
                   "        return\n"
                   "    var q := Quaternion.from_euler(Vector3(\n"
                   "        deg_to_rad(derece.x), deg_to_rad(derece.y),\n"
                   "        deg_to_rad(derece.z)))\n"
                   "    set_bone_pose_rotation(idx, q)\n"
                   "\n"
                   "func kemik_listesi() -> Array:\n"
                   "    var liste := []\n"
                   "    for i in range(get_bone_count()):\n"
                   "        liste.append(get_bone_name(i))\n"
                   "    return liste")),
        Lesson("godot_karakter", "Animasyon: AnimationPlayer ve karıştırma",
               "**`AnimationPlayer`** hazır animasyonları oynatır "
               "(`play(\"yuru\")`). **`AnimationTree`** ise "
               "**durum makinesi** ve **karıştırma** sağlar: yürüme ↔ "
               "koşma arası yumuşak geçiş (`BlendSpace1D`), alt gövde "
               "yürürken üst gövde ateş eder (`BlendSpace2D`, kemik "
               "maskeleri). **Geçiş:** `AnimationNodeStateMachine` "
               "düğümleri ve `travel(\"kos\")` çağrısı. Bu, **gerçek "
               "oyun** animasyon sistemidir; tek `AnimationPlayer` "
               "yeterli değildir.",
               ["AnimationPlayer", "AnimationTree", "state machine"], code=(
                   "extends CharacterBody3D\n"
                   "\n"
                   "@onready var anim: AnimationPlayer = $Model/AnimationPlayer\n"
                   "@onready var agac: AnimationTree = $Model/AnimationTree\n"
                   "\n"
                   "var _durum: AnimationNodeStateMachinePlayback\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    agac.active = true\n"
                   "    _durum = agac.get(\"parameters/StateMachine/playback\")\n"
                   "    _durum.travel(\"bekle\")\n"
                   "\n"
                   "func durum_degistir(yeni: String) -> void:\n"
                   "    if _durum.get_current_node() != yeni:\n"
                   "        _durum.travel(yeni)\n"
                   "\n"
                   "func hiz_ayarla(deger: float) -> void:\n"
                   "    # BlendSpace1D: hiza gore yuru<->kos karisimi\n"
                   "    agac.set(\"parameters/Hiz/blend_position\", deger)")),
        Lesson("godot_karakter", "Monster ve yapay zekâ (durum makinesi)",
               "Monster davranışı **durum makinesi** ile yazılır: "
               "`BEKLE`, `DEVGİNLE` (oyuncuyu ara), `KOVALA`, `SALDIR`, "
               "`OLU`. **Hedef bulma:** `Area3D` (görüş alanı) veya "
               "mesafe kontrolü (`global_position.distance_to(oyuncu)`). "
               "**Yol bulma:** `NavigationAgent3D` — hedefi verir, "
               "`get_next_path_position()` ile yön alınır; engellerden "
               "kaçınır. **Saldırı:** menzile girince `AnimationPlayer` "
               "ve `hasar_uygula`. **Optimizasyon:** uzak monsterlar "
               "`_physics_process` yerine zamanlayıcı ile seyrek çalışır.",
               ["monster", "AI", "NavigationAgent3D"], code=(
                   "extends CharacterBody3D\n"
                   "\n"
                   "enum Durum { BEKLE, KOVALA, SALDIR, OLU }\n"
                   "\n"
                   "@export var hiz := 3.0\n"
                   "@export var saldiri_menzili := 1.8\n"
                   "@export var gorus_menzili := 12.0\n"
                   "@export var hasar := 8\n"
                   "\n"
                   "@onready var ajan: NavigationAgent3D = $NavigationAgent3D\n"
                   "@onready var anim: AnimationPlayer = $Model/AnimationPlayer\n"
                   "\n"
                   "var durum: Durum = Durum.BEKLE\n"
                   "var oyuncu: Node3D = null\n"
                   "var saglik := 40\n"
                   "\n"
                   "func _physics_process(delta: float) -> void:\n"
                   "    if durum == Durum.OLU:\n"
                   "        return\n"
                   "    if not is_instance_valid(oyuncu):\n"
                   "        durum = Durum.BEKLE\n"
                   "        return\n"
                   "    var mesafe := global_position.distance_to(oyuncu.global_position)\n"
                   "    if mesafe <= saldiri_menzili:\n"
                   "        durum = Durum.SALDIR\n"
                   "        anim.play(\"saldir\")\n"
                   "    elif mesafe <= gorus_menzili:\n"
                   "        durum = Durum.KOVALA\n"
                   "        ajan.target_position = oyuncu.global_position\n"
                   "        var yon := (ajan.get_next_path_position() - global_position).normalized()\n"
                   "        velocity = yon * hiz\n"
                   "        move_and_slide()\n"
                   "        if yon.length() > 0.1:\n"
                   "            look_at(global_position + yon, Vector3.UP)")),
        Lesson("godot_karakter", "Blender'dan karakter aktarımı (GLTF)",
               "Akış: **Blender'da** model + iskelet + ağırlık + animasyon "
               "hazırlanır → **GLB olarak dışa aktarılır** "
               "(`bpy.ops.export_scene.gltf(filepath=..., "
               "export_format='GLB', export_animations=True)`) → Godot'ta "
               "`res://karakter.glb` içe aktarılır. **Godot tarafı:** "
               "kök düğüm `Node3D`, altında `Skeleton3D` + `MeshInstance3D` "
               "+ `AnimationPlayer`. **Kontrol listesi:** (1) iskelet "
               "kökü `Armature` olmalı, (2) mesh `ARMATURE` modifier "
               "taşımalı, (3) animasyonlar `NLA` veya eylem olarak "
               "işaretli, (4) ölçek 1,0 (Godot metre kullanır).",
               ["GLTF", "aktarım", "Blender"], code=(
                   "# Blender tarafi (Python):\n"
                   "# import bpy\n"
                   "# bpy.ops.export_scene.gltf(\n"
                   "#     filepath='karakter.glb',\n"
                   "#     export_format='GLB',\n"
                   "#     export_animations=True,\n"
                   "#     export_skins=True,\n"
                   "#     export_yup=True)      # Y yukari (Godot standardi)\n"
                   "\n"
                   "# Godot tarafi (GDScript):\n"
                   "extends Node3D\n"
                   "\n"
                   "func yukle(yol := \"res://karakter.glb\") -> void:\n"
                   "    var sahne: PackedScene = load(yol)\n"
                   "    var ornek := sahne.instantiate()\n"
                   "    add_child(ornek)\n"
                   "    var iskelet := ornek.find_child(\"Skeleton3D\", true, false)\n"
                   "    if iskelet:\n"
                   "        print(\"Kemik sayisi: \", iskelet.get_bone_count())\n"
                   "    var anim := ornek.find_child(\"AnimationPlayer\", true, false)\n"
                   "    if anim:\n"
                   "        print(\"Animasyonlar: \", anim.get_animation_list())")),
    ]


# ─── Katman 6: Seviye Tasarımı ──────────────────────────────────────────────

def _seviye_lessons() -> List[Lesson]:
    """Maze, arena, mağara, prosedürel seviye."""
    return [
        Lesson("godot_seviye", "Labirent (maze) üretimi: geri izleme algoritması",
               "**Geri izleme (recursive backtracker)** en yaygın labirent "
               "algoritmasıdır: **(1)** tüm hücreler duvarlı başlar, "
               "**(2)** rastgele bir hücreden başla, işaretle, "
               "**(3)** ziyaret edilmemiş komşu varsa duvarı yık ve oraya "
               "geç, **(4)** yoksa geri dön. Sonuç: **tek çözümlü, "
               "dallanan** labirent. Alternatifler: **Prim** (daha çok "
               "kısa çıkmaz), **Kruskal** (rastgele), **Wilson** (tam "
               "rastgele). Izgara `N×M` tek sayı olmalı (duvar hücreleri "
               "için).",
               ["maze", "geri izleme", "labirent"], code=(
                   "class_name LabirentUretici\n"
                   "extends RefCounted\n"
                   "\n"
                   "var genislik: int\n"
                   "var yukseklik: int\n"
                   "var hucre: Array = []      # true = duvar\n"
                   "\n"
                   "func _init(g: int, y: int, tohum := 0) -> void:\n"
                   "    genislik = g if g % 2 == 1 else g + 1\n"
                   "    yukseklik = y if y % 2 == 1 else y + 1\n"
                   "    var rng := RandomNumberGenerator.new()\n"
                   "    rng.seed = tohum\n"
                   "    for j in range(yukseklik):\n"
                   "        var satir := []\n"
                   "        for i in range(genislik):\n"
                   "            satir.append(true)\n"
                   "        hucre.append(satir)\n"
                   "    _kaz(1, 1, rng)\n"
                   "\n"
                   "func _kaz(i: int, j: int, rng: RandomNumberGenerator) -> void:\n"
                   "    hucre[j][i] = false\n"
                   "    var yonler := [[2, 0], [-2, 0], [0, 2], [0, -2]]\n"
                   "    yonler.shuffle()\n"
                   "    for y in yonler:\n"
                   "        var ni: int = i + y[0]\n"
                   "        var nj: int = j + y[1]\n"
                   "        if ni > 0 and ni < genislik - 1 and nj > 0 and nj < yukseklik - 1:\n"
                   "            if hucre[nj][ni]:\n"
                   "                hucre[j + y[1] / 2][i + y[0] / 2] = false\n"
                   "                _kaz(ni, nj, rng)")),
        Lesson("godot_seviye", "Labirenti 3B sahneye dökmek",
               "Izgara → 3B: her **duvar hücresi** için bir kutu "
               "(`BoxMesh`) üretilir. **Performans:** yüzlerce kutu yerine "
               "**tek mesh** üretmek (`SurfaceTool` ile tüm kutuları "
               "birleştirmek) draw call'u 1'e indirir. **Çarpışma:** "
               "`MeshInstance3D.create_trimesh_collision()` tek çağrıyla "
               "statik çarpışma üretir. **Zemin ve tavan** ayrı eklenir. "
               "**Ölçek:** hücre boyutu 2 birim ise 21×21 labirent 42×42 "
               "birim alan kaplar.",
               ["labirent 3B", "trimesh", "birleştirme"], code=(
                   "extends Node3D\n"
                   "\n"
                   "@export var hucre_boyutu := 2.0\n"
                   "@export var duvar_yuksekligi := 3.0\n"
                   "@export var tohum := 13\n"
                   "\n"
                   "func labirent_kur(g: int, y: int) -> void:\n"
                   "    var lab := LabirentUretici.new(g, y, tohum)\n"
                   "    var st := SurfaceTool.new()\n"
                   "    st.begin(Mesh.PRIMITIVE_TRIANGLES)\n"
                   "    var kutu := BoxMesh.new()\n"
                   "    kutu.size = Vector3(hucre_boyutu, duvar_yuksekligi, hucre_boyutu)\n"
                   "    var kutu_mesh: ArrayMesh = kutu.get_mesh()\n"
                   "    for j in range(lab.yukseklik):\n"
                   "        for i in range(lab.genislik):\n"
                   "            if not lab.hucre[j][i]:\n"
                   "                continue\n"
                   "            var t := Transform3D()\n"
                   "            t.origin = Vector3(i * hucre_boyutu, duvar_yuksekligi * 0.5,\n"
                   "                               j * hucre_boyutu)\n"
                   "            st.append_from(kutu_mesh, 0, t)\n"
                   "    st.generate_normals()\n"
                   "    var mi := MeshInstance3D.new()\n"
                   "    mi.mesh = st.commit()\n"
                   "    add_child(mi)\n"
                   "    mi.create_trimesh_collision()\n"
                   "    print(\"Labirent: \", lab.genislik, \"x\", lab.yukseklik)")),
        Lesson("godot_seviye", "Arena: simetrik dövüş alanı",
               "**Arena** kapalı, simetrik ve okunabilir olmalıdır. "
               "Bileşenler: **zemin** (yuvarlak veya kare), **çevre duvarı** "
               "(`CylinderMesh` ince halka veya 4 duvar), **kapaklar** "
               "(giriş noktaları), **örtü** (sütun, kaya — siper), "
               "**merkez öğesi** (yükselti veya tehlike). **Simetri** "
               "adaleti sağlar: 2/4/6 kapak eşit aralıkla. "
               "**Boyut:** 1v1 için 30×30, 4v4 için 60×60 birim. "
               "**Doğurma noktaları** `Marker3D` düğümleriyle işaretlenir.",
               ["arena", "simetri", "doğurma"], code=(
                   "extends Node3D\n"
                   "\n"
                   "@export var yaricap := 22.0\n"
                   "@export var kapak_sayisi := 4\n"
                   "@export var sutun_sayisi := 8\n"
                   "\n"
                   "func arena_kur() -> void:\n"
                   "    # Zemin\n"
                   "    var zemin := MeshInstance3D.new()\n"
                   "    var sil := CylinderMesh.new()\n"
                   "    sil.radial_segments = 16      # lowpoly\n"
                   "    sil.top_radius = yaricap\n"
                   "    sil.bottom_radius = yaricap\n"
                   "    sil.height = 0.5\n"
                   "    zemin.mesh = sil\n"
                   "    zemin.position.y = -0.25\n"
                   "    add_child(zemin)\n"
                   "    zemin.create_trimesh_collision()\n"
                   "    # Cevre duvari: 12 kenarli halka\n"
                   "    var duvar := MeshInstance3D.new()\n"
                   "    var halka := CylinderMesh.new()\n"
                   "    halka.radial_segments = 12\n"
                   "    halka.top_radius = yaricap + 1.0\n"
                   "    halka.bottom_radius = yaricap + 1.0\n"
                   "    halka.height = 6.0\n"
                   "    duvar.mesh = halka\n"
                   "    duvar.position.y = 3.0\n"
                   "    add_child(duvar)\n"
                   "    # Sutunlar (simerik)\n"
                   "    for k in range(sutun_sayisi):\n"
                   "        var aci := TAU * k / float(sutun_sayisi)\n"
                   "        var m := Marker3D.new()\n"
                   "        m.position = Vector3(cos(aci) * yaricap * 0.6, 0,\n"
                   "                             sin(aci) * yaricap * 0.6)\n"
                   "        add_child(m)")),
        Lesson("godot_seviye", "Mağara: 3B gürültü ile oyma",
               "**Mağara** iki yolla üretilir: **(1) voxel/3B gürültü** — "
               "3B uzayda gürültü eşiğin altındaysa boşluk (yürünebilir), "
               "üstündeyse kaya; **marching cubes** veya basit kutu "
               "yaklaşımı. **(2) tünel yolu** — bir eğri (Curve3D) "
               "boyunca silindir oyup birleştirmek; daha kontrollü ve "
               "ucuz. **Lowpoly mağara:** tünel yolu + düzensizleştirme. "
               "**Aydınlatma:** mağarada güneş yok — `OmniLight3D` "
               "(meşale) veya `WorldEnvironment` ile düşük ortam ışığı.",
               ["mağara", "voxel", "tünel"], code=(
                   "extends Node3D\n"
                   "\n"
                   "@export var nokta_sayisi := 12\n"
                   "@export var turel_yaricap := 3.0\n"
                   "@export var tohum := 5\n"
                   "\n"
                   "func tunel_uret() -> ArrayMesh:\n"
                   "    var rng := RandomNumberGenerator.new()\n"
                   "    rng.seed = tohum\n"
                   "    var st := SurfaceTool.new()\n"
                   "    st.begin(Mesh.PRIMITIVE_TRIANGLES)\n"
                   "    var kenar := 8\n"
                   "    var onceki := []\n"
                   "    for n in range(nokta_sayisi):\n"
                   "        var merkez := Vector3(n * 6.0,\n"
                   "            rng.randf_range(-1.5, 1.5) + sin(n * 0.8) * 2.0,\n"
                   "            rng.randf_range(-1.5, 1.5))\n"
                   "        var yaricap := turel_yaricap * rng.randf_range(0.7, 1.3)\n"
                   "        var halka := []\n"
                   "        for k in range(kenar):\n"
                   "            var aci := TAU * k / float(kenar)\n"
                   "            halka.append(merkez + Vector3(0,\n"
                   "                sin(aci) * yaricap, cos(aci) * yaricap))\n"
                   "        if onceki.size() > 0:\n"
                   "            for k in range(kenar):\n"
                   "                var a: Vector3 = onceki[k]\n"
                   "                var b: Vector3 = onceki[(k + 1) % kenar]\n"
                   "                var c: Vector3 = halka[(k + 1) % kenar]\n"
                   "                var d: Vector3 = halka[k]\n"
                   "                st.add_vertex(a); st.add_vertex(b); st.add_vertex(c)\n"
                   "                st.add_vertex(a); st.add_vertex(c); st.add_vertex(d)\n"
                   "        onceki = halka\n"
                   "    st.generate_normals()\n"
                   "    return st.commit()")),
        Lesson("godot_seviye", "NavigationMesh ile yürünebilir alan",
               "**`NavigationRegion3D`** + **`NavigationMesh`** yapay "
               "zekânın yürüyebileceği alanı tanımlar. Kurulum: "
               "`NavigationRegion3D` ekle, altına zemin mesh'ini koy, "
               "`bake_navigation_mesh()` çağır. **Ajanlar:** "
               "`NavigationAgent3D` (`target_position` ayarla, "
               "`get_next_path_position()` kullan) veya "
               "`NavigationAgent2D`. **Ayarlar:** `agent_radius` "
               "(karakter genişliği), `cell_size` (çözünürlük), "
               "`agent_max_climb` (çıkabildiği yükseklik). **Prosedürel "
               "seviyede** bake'i çalışma zamanında çağırmak gerekir.",
               ["NavigationRegion3D", "bake", "yol bulma"], code=(
                   "extends NavigationRegion3D\n"
                   "\n"
                   "@export var otomatik_piser := true\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    if otomatik_piser:\n"
                   "        piser()\n"
                   "\n"
                   "func piser() -> void:\n"
                   "    var nm := NavigationMesh.new()\n"
                   "    nm.agent_radius = 0.6\n"
                   "    nm.agent_height = 1.8\n"
                   "    nm.agent_max_climb = 0.4\n"
                   "    nm.cell_size = 0.25\n"
                   "    nm.cell_height = 0.2\n"
                   "    navigation_mesh = nm\n"
                   "    bake_navigation_mesh(false)\n"
                   "    print(\"Navigasyon pisirildi\")")),
    ]


# ─── Katman 7: Girdi ve Tuş Atamaları ───────────────────────────────────────

def _girdi_lessons() -> List[Lesson]:
    """Input map, WASD, tuş atama."""
    return [
        Lesson("godot_girdi", "Girdi haritası (Input Map) ve WASD",
               "**Girdi haritası** tuşları **eylemlere** bağlar — kod "
               "tuş adı bilmez, `ileri`/`geri`/`sol`/`sag` eylemlerini "
               "bilir. Böylece tuş değiştirmek **kod değiştirmez**. "
               "Proje → Proje Ayarları → Girdi Haritası: `ileri` (W, ↑), "
               "`geri` (S, ↓), `sol` (A, ←), `sag` (D, →), `zipla` "
               "(Space), `kos` (Shift), `etkilesim` (E), `envanter` (Tab). "
               "**Ölü bölge:** analog çubuklar için `deadzone` ~0,2. "
               "**Kod ile ekleme:** `InputMap.add_action(\"eylem\")` + "
               "`InputMap.action_add_event(...)`.",
               ["InputMap", "WASD", "eylem"], code=(
                   "extends CharacterBody3D\n"
                   "\n"
                   "@export var hiz := 5.0\n"
                   "\n"
                   "func _physics_process(delta: float) -> void:\n"
                   "    # WASD -> vektor (x: sol/sag, y: ileri/geri)\n"
                   "    var giris := Input.get_vector(\"sol\", \"sag\", \"ileri\", \"geri\")\n"
                   "    var yon := (transform.basis * Vector3(giris.x, 0, giris.y)).normalized()\n"
                   "    velocity.x = yon.x * hiz\n"
                   "    velocity.z = yon.z * hiz\n"
                   "    if Input.is_action_just_pressed(\"zipla\"):\n"
                   "        print(\"Zipla\")\n"
                   "    move_and_slide()")),
        Lesson("godot_girdi", "Tuş atamayı kod ile yapmak (yeniden atanabilir)",
               "Oyun içi **tuş atama ekranı** için: "
               "`InputMap.action_get_events(eylem)` ile mevcut tuşlar "
               "okunur, `action_erase_events` + `action_add_event` ile "
               "değiştirilir. **Yeni tuş yakalama:** "
               "`_input(olay)` içinde `olay is InputEventKey and "
               "olay.pressed` kontrolü. **Kaydetme:** atamalar "
               "`ConfigFile` ile `user://ayarlar.cfg`'ye yazılır — "
               "böylece oyuncu ayarı kalıcı olur.",
               ["tuş atama", "InputEventKey", "ConfigFile"], code=(
                   "extends Control\n"
                   "\n"
                   "var _bekleyen: String = \"\"\n"
                   "\n"
                   "func tus_degistir(eylem: String) -> void:\n"
                   "    _bekleyen = eylem\n"
                   "    $Etiket.text = \"Bir tus basin...\"\n"
                   "\n"
                   "func _input(olay: InputEvent) -> void:\n"
                   "    if _bekleyen == \"\":\n"
                   "        return\n"
                   "    if olay is InputEventKey and olay.pressed:\n"
                   "        InputMap.action_erase_events(_bekleyen)\n"
                   "        InputMap.action_add_event(_bekleyen, olay)\n"
                   "        print(\"Yeni tus: \", _bekleyen, \" -> \", olay.as_text())\n"
                   "        _bekleyen = \"\"\n"
                   "        get_viewport().set_input_as_handled()\n"
                   "\n"
                   "func kaydet(yol := \"user://ayarlar.cfg\") -> void:\n"
                   "    var cfg := ConfigFile.new()\n"
                   "    for eylem in InputMap.get_actions():\n"
                   "        var tuslar := []\n"
                   "        for o in InputMap.action_get_events(eylem):\n"
                   "            tuslar.append(o.as_text())\n"
                   "        cfg.set_value(\"girdi\", eylem, tuslar)\n"
                   "    cfg.save(yol)")),
        Lesson("godot_girdi", "Dokunmatik ve oyun kolu desteği",
               "**Dokunmatik:** `InputEventScreenTouch` (parmak basma) ve "
               "`InputEventScreenDrag` (sürükleme). **Sanal joystick** "
               "genelde `TouchScreenButton` veya özel `Control` ile "
               "yapılır. **Oyun kolu:** Godot kolları otomatik tanır; "
               "`Input.get_joy_name(0)`, `Input.get_joy_axis(0, "
               "JOY_AXIS_LEFT_X)`. **Eylem eşlemesi:** kol düğmelerini "
               "de girdi haritasına ekleyin — kod değişmez. "
               "**`Input.get_vector`** analog çubukla da çalışır; "
               "klavye+kolu aynı anda desteklenir.",
               ["dokunmatik", "oyun kolu", "analog"], code=(
                   "extends CharacterBody3D\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    # Kol bagli mi?\n"
                   "    var kollar := Input.get_connected_joypads()\n"
                   "    if kollar.size() > 0:\n"
                   "        print(\"Kol: \", Input.get_joy_name(kollar[0]))\n"
                   "\n"
                   "func _physics_process(delta: float) -> void:\n"
                   "    # get_vector hem klavye hem analog cubugu okur\n"
                   "    var giris := Input.get_vector(\"sol\", \"sag\", \"ileri\", \"geri\")\n"
                   "    # Dik eksen: tetik tuslari\n"
                   "    var gaz := Input.get_axis(\"geri_vites\", \"ileri_vites\")\n"
                   "    var yon := (transform.basis * Vector3(giris.x, 0, giris.y))\n"
                   "    velocity = yon * 6.0\n"
                   "    move_and_slide()")),
        Lesson("godot_girdi", "Girdi yakalama sırası ve çakışma",
               "Girdi üç katmanda işlenir: **`_input`** (en önce, tüm "
               "olaylar), **`_gui_input`** (UI düğümleri), "
               "**`_unhandled_input`** (UI tüketmediyse — **oyun "
               "girdisi için doğru yer**). `_input`'ta oyun girdisi "
               "işlemek, UI açıkken karakterin hareket etmesine yol açar. "
               "**Tüketme:** `get_viewport().set_input_as_handled()` ile "
               "olay işlendi olarak işaretlenir, alt katmanlara gitmez. "
               "**Pause:** `get_tree().paused = true` + düğümde "
               "`process_mode` ayarı.",
               ["_input", "_unhandled_input", "tüketme"], code=(
                   "extends Node3D\n"
                   "\n"
                   "var envanter_acik := false\n"
                   "\n"
                   "func _unhandled_input(olay: InputEvent) -> void:\n"
                   "    # UI acikken oyun girdisini yok say\n"
                   "    if envanter_acik:\n"
                   "        return\n"
                   "    if olay.is_action_pressed(\"etkilesim\"):\n"
                   "        _etkilesim_dene()\n"
                   "        get_viewport().set_input_as_handled()\n"
                   "    if olay.is_action_pressed(\"envanter\"):\n"
                   "        envanter_acik = true\n"
                   "        Input.mouse_mode = Input.MOUSE_MODE_VISIBLE\n"
                   "\n"
                   "func _etkilesim_dene() -> void:\n"
                   "    var isin: RayCast3D = $Kamera/RayCast3D\n"
                   "    if isin.is_colliding():\n"
                   "        var hedef := isin.get_collider()\n"
                   "        if hedef is Interactable:\n"
                   "            hedef.etkilesim(self)")),
    ]


# ─── Katman 8: Arayüz ───────────────────────────────────────────────────────

def _arayuz_lessons() -> List[Lesson]:
    """UI, HUD, envanter, menü."""
    return [
        Lesson("godot_arayuz", "Control düğümleri ve yerleşim",
               "UI **`Control`** tabanlıdır (2B). Temel düğümler: "
               "`Label` (yazı), `Button` (düğme), `TextureRect` (resim), "
               "`ProgressBar` (çubuk), `Panel` (arka plan). "
               "**Kapsayıcılar (container) yerleşimi otomatik yapar:** "
               "`VBoxContainer` (dikey), `HBoxContainer` (yatay), "
               "`GridContainer` (ızgara), `MarginContainer` (boşluk), "
               "`CenterContainer` (ortala). **Çapa (anchor):** "
               "`anchors_preset` ile köşeye yapıştırma "
               "(`PRESET_TOP_LEFT`, `PRESET_FULL_RECT`). Elle konum "
               "**kullanmayın** — farklı çözünürlükte bozulur.",
               ["Control", "container", "anchor"], code=(
                   "extends CanvasLayer\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    var kok := MarginContainer.new()\n"
                   "    kok.set_anchors_preset(Control.PRESET_TOP_LEFT)\n"
                   "    kok.add_theme_constant_override(\"margin_left\", 16)\n"
                   "    kok.add_theme_constant_override(\"margin_top\", 16)\n"
                   "    add_child(kok)\n"
                   "    var kutu := VBoxContainer.new()\n"
                   "    kok.add_child(kutu)\n"
                   "    var saglik := ProgressBar.new()\n"
                   "    saglik.max_value = 100\n"
                   "    saglik.value = 100\n"
                   "    saglik.custom_minimum_size = Vector2(220, 22)\n"
                   "    kutu.add_child(saglik)\n"
                   "    var etiket := Label.new()\n"
                   "    etiket.text = \"Can: 100\"\n"
                   "    kutu.add_child(etiket)")),
        Lesson("godot_arayuz", "HUD: sağlık, mana, skor bağlama",
               "HUD **sinyalle** güncellenir — her karede değil. "
               "Karakter `saglik_degisti(yeni, maks)` sinyali yayar, "
               "HUD bağlanır ve çubuğu günceller. Bu **gevşek bağlılık** "
               "sağlar: karakter UI'yı bilmez. **Ölçekleme:** "
               "`CanvasLayer` + `Control` kullanmak çözünürlükten "
               "bağımsız kılar. **Performans:** `_process`'te "
               "`label.text = str(...)` yazmak her karede metin "
               "oluşturur — yalnızca değer değiştiğinde güncelleyin.",
               ["HUD", "sinyal", "sağlık"], code=(
                   "extends CanvasLayer\n"
                   "\n"
                   "@onready var saglik_cubugu: ProgressBar = $Kok/SaglikCubugu\n"
                   "@onready var saglik_etiketi: Label = $Kok/SaglikEtiketi\n"
                   "@onready var skor_etiketi: Label = $Kok/SkorEtiketi\n"
                   "\n"
                   "func karakter_bagla(karakter: Node) -> void:\n"
                   "    if karakter.has_signal(\"saglik_degisti\"):\n"
                   "        karakter.saglik_degisti.connect(_on_saglik)\n"
                   "    if karakter.has_signal(\"skor_degisti\"):\n"
                   "        karakter.skor_degisti.connect(_on_skor)\n"
                   "\n"
                   "func _on_saglik(yeni: int, maks: int) -> void:\n"
                   "    saglik_cubugu.max_value = maks\n"
                   "    saglik_cubugu.value = yeni\n"
                   "    saglik_etiketi.text = \"%d / %d\" % [yeni, maks]\n"
                   "\n"
                   "func _on_skor(yeni: int) -> void:\n"
                   "    skor_etiketi.text = \"Skor: %d\" % yeni")),
        Lesson("godot_arayuz", "Envanter: ızgara ve eşya yerleştirme",
               "Envanter = **veri** (eşya listesi) + **görünüm** "
               "(`GridContainer` + `TextureRect`). **Sürükle-bırak:** "
               "`_get_drag_data`, `_can_drop_data`, `_drop_data` "
               "fonksiyonları `Control` üzerinde ezilir. **Yuva:** her "
               "yuva bir `Panel` + `TextureRect` + `Label` (adet). "
               "**Veri modeli:** `Dictionary` listesi "
               "(`{\"id\": \"kilic\", \"adet\": 1}`) — görünüm veriden "
               "**türetilir**, tersi değil.",
               ["envanter", "sürükle-bırak", "yuva"], code=(
                   "extends GridContainer\n"
                   "\n"
                   "signal esya_secildi(esya: Dictionary)\n"
                   "\n"
                   "@export var yuva_sayisi := 20\n"
                   "\n"
                   "var esyalar: Array = []\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    columns = 5\n"
                   "    for i in range(yuva_sayisi):\n"
                   "        var yuva := Panel.new()\n"
                   "        yuva.custom_minimum_size = Vector2(64, 64)\n"
                   "        yuva.name = \"Yuva%d\" % i\n"
                   "        var simge := TextureRect.new()\n"
                   "        simge.name = \"Simge\"\n"
                   "        simge.expand_mode = TextureRect.EXPAND_IGNORE_SIZE\n"
                   "        simge.set_anchors_preset(Control.PRESET_FULL_RECT)\n"
                   "        yuva.add_child(simge)\n"
                   "        var adet := Label.new()\n"
                   "        adet.name = \"Adet\"\n"
                   "        adet.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)\n"
                   "        yuva.add_child(adet)\n"
                   "        add_child(yuva)\n"
                   "\n"
                   "func yenile() -> void:\n"
                   "    for i in range(yuva_sayisi):\n"
                   "        var yuva := get_child(i)\n"
                   "        var simge: TextureRect = yuva.get_node(\"Simge\")\n"
                   "        var adet: Label = yuva.get_node(\"Adet\")\n"
                   "        if i < esyalar.size():\n"
                   "            simge.texture = load(esyalar[i].get(\"simge\", \"\"))\n"
                   "            adet.text = str(esyalar[i].get(\"adet\", 1))\n"
                   "        else:\n"
                   "            simge.texture = null\n"
                   "            adet.text = \"\"")),
        Lesson("godot_arayuz", "Menü, duraklatma ve sahne akışı",
               "**Ana menü:** `Button` + `pressed` sinyali → "
               "`get_tree().change_scene_to_file(...)`. **Duraklatma:** "
               "`get_tree().paused = true`; duraklatma menüsü "
               "`process_mode = PROCESS_MODE_WHEN_PAUSED` olmalı, aksi "
               "hâlde menü de durur. **Ayarlar:** ses seviyesi "
               "(`AudioServer.set_bus_volume_db`), çözünürlük "
               "(`DisplayServer.window_set_size`), tam ekran "
               "(`window_set_mode`). **Kaydetme:** `ConfigFile` veya "
               "`ResourceSaver` ile `user://` altına.",
               ["menü", "duraklatma", "sahne akışı"], code=(
                   "extends CanvasLayer\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    process_mode = Node.PROCESS_MODE_WHEN_PAUSED\n"
                   "    visible = false\n"
                   "\n"
                   "func _unhandled_input(olay: InputEvent) -> void:\n"
                   "    if olay.is_action_pressed(\"duraklat\"):\n"
                   "        _degistir()\n"
                   "\n"
                   "func _degistir() -> void:\n"
                   "    var yeni := not get_tree().paused\n"
                   "    get_tree().paused = yeni\n"
                   "    visible = yeni\n"
                   "    Input.mouse_mode = (Input.MOUSE_MODE_VISIBLE if yeni\n"
                   "                        else Input.MOUSE_MODE_CAPTURED)\n"
                   "\n"
                   "func _on_devam_pressed() -> void:\n"
                   "    get_tree().paused = false\n"
                   "    visible = false\n"
                   "\n"
                   "func _on_ana_menu_pressed() -> void:\n"
                   "    get_tree().paused = false\n"
                   "    get_tree().change_scene_to_file(\"res://menuler/ana_menu.tscn\")")),
    ]


# ─── Katman 9: Ses ──────────────────────────────────────────────────────────

def _ses_lessons() -> List[Lesson]:
    """AudioStreamPlayer, müzik, efekt."""
    return [
        Lesson("godot_ses", "AudioStreamPlayer: 2B, 3B ve genel",
               "Üç tür: **`AudioStreamPlayer`** (konumsuz — müzik, arayüz), "
               "**`AudioStreamPlayer2D`** (2B konumlu), "
               "**`AudioStreamPlayer3D`** (3B konumlu — **dünyada ses**, "
               "mesafeyle azalır). 3B ayarları: `unit_size` (duyulma "
               "mesafesi), `max_distance`, `attenuation_model`, "
               "`panning_strength`. **Çalma:** `play()`, `stop()`, "
               "`stream_paused`. **Döngü:** stream'in `loop` özelliği "
               "veya `finished` sinyalinde `play()` tekrarı.",
               ["AudioStreamPlayer", "3B ses", "döngü"], code=(
                   "extends Node3D\n"
                   "\n"
                   "@onready var ayak: AudioStreamPlayer3D = $AyakSesi\n"
                   "@onready var muzik: AudioStreamPlayer = $Muzik\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    ayak.unit_size = 6.0\n"
                   "    ayak.max_distance = 25.0\n"
                   "    ayak.attenuation_model = AudioStreamPlayer3D.ATTENUATION_INVERSE_DISTANCE\n"
                   "    muzik.stream = load(\"res://sesler/ambiyans.ogg\")\n"
                   "    muzik.volume_db = -12.0\n"
                   "    muzik.play()\n"
                   "\n"
                   "func adim_sesi() -> void:\n"
                   "    ayak.stream = load(\"res://sesler/adim.ogg\")\n"
                   "    ayak.pitch_scale = randf_range(0.9, 1.1)   # cesitlilik\n"
                   "    ayak.play()")),
        Lesson("godot_ses", "Ses veri yolu (bus) ve ses seviyesi",
               "**Ses veri yolu** sesleri gruplar: `Master` → `Muzik`, "
               "`Efektler`, `Arayuz`. Ayarlar → Ses Veri Yolu. Kod ile "
               "seviye: `AudioServer.get_bus_index(\"Muzik\")` + "
               "`set_bus_volume_db(idx, db)`. **Desibel:** 0 dB tam "
               "seviye, −6 dB yarı, −80 dB sessiz. Doğrusal kaydırıcı "
               "için: `linear_to_db(deger)` ve `db_to_linear(db)`. "
               "**Efekt:** `AudioServer.add_bus_effect` ile yankı, "
               "filtre eklenir.",
               ["bus", "desibel", "ses seviyesi"], code=(
                   "extends Node\n"
                   "\n"
                   "func ses_seviyesi_ayarla(bus_adi: String, oran: float) -> void:\n"
                   "    # oran: 0.0 (sessiz) .. 1.0 (tam)\n"
                   "    var idx := AudioServer.get_bus_index(bus_adi)\n"
                   "    if idx < 0:\n"
                   "        push_warning(\"Bus yok: \" + bus_adi)\n"
                   "        return\n"
                   "    var db := linear_to_db(clampf(oran, 0.0001, 1.0))\n"
                   "    AudioServer.set_bus_volume_db(idx, db)\n"
                   "    AudioServer.set_bus_mute(idx, oran <= 0.001)\n"
                   "\n"
                   "func butun_sesleri_kapat() -> void:\n"
                   "    AudioServer.set_bus_mute(AudioServer.get_bus_index(\"Master\"), true)\n"
                   "\n"
                   "func yankı_ekle(bus_adi: String) -> void:\n"
                   "    var idx := AudioServer.get_bus_index(bus_adi)\n"
                   "    var yanki := AudioEffectReverb.new()\n"
                   "    yanki.room_size = 0.7\n"
                   "    yanki.wet = 0.3\n"
                   "    AudioServer.add_bus_effect(idx, yanki)")),
        Lesson("godot_ses", "Ses havuzu (pooling) ve performans",
               "Aynı sesi **arka arkaya** çalmak (`ates`) tek "
               "`AudioStreamPlayer` ile çalışmaz — ilk ses kesilir. "
               "**Çözüm: havuz** (pool) — N adet player hazırlanır, "
               "boşta olan kullanılır. **`max_polyphony`** özelliği "
               "bunu otomatik yapar (Godot 4): `player.max_polyphony = 8`. "
               "**3B sahne:** onlarca 3B ses CPU yükü yaratır — "
               "duyulmayanları `stop()` ile kapatın.",
               ["ses havuzu", "max_polyphony", "performans"], code=(
                   "extends Node3D\n"
                   "\n"
                   "@export var havuz_boyu := 6\n"
                   "\n"
                   "var _havuz: Array = []\n"
                   "var _sira := 0\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    for i in range(havuz_boyu):\n"
                   "        var p := AudioStreamPlayer3D.new()\n"
                   "        p.max_distance = 30.0\n"
                   "        add_child(p)\n"
                   "        _havuz.append(p)\n"
                   "\n"
                   "func cal(stream: AudioStream, ses_db := 0.0) -> void:\n"
                   "    var p: AudioStreamPlayer3D = _havuz[_sira]\n"
                   "    _sira = (_sira + 1) % _havuz.size()\n"
                   "    p.stream = stream\n"
                   "    p.volume_db = ses_db\n"
                   "    p.play()")),
    ]


# ─── Katman 10: Ağ (Master/Proxy Server) ────────────────────────────────────

def _ag_lessons() -> List[Lesson]:
    """Master server, proxy, RPC, çok oyunculu."""
    return [
        Lesson("godot_ag", "Godot'ta sunucu/istemci kurulumu",
               "**Sunucu:** `ENetMultiplayerPeer.new()` → "
               "`create_server(port, max_clients)` → "
               "`multiplayer.multiplayer_peer = peer`. **İstemci:** "
               "`create_client(adres, port)`. **Rol kontrolü:** "
               "`multiplayer.is_server()`, `multiplayer.get_unique_id()` "
               "(1 = sunucu). **Olaylar:** `peer_connected`, "
               "`peer_disconnected`, `connected_to_server`, "
               "`connection_failed`. **Port:** 7777 yaygın; güvenlik "
               "duvarında açılmalı.",
               ["ENetMultiplayerPeer", "sunucu", "istemci"], code=(
                   "extends Node\n"
                   "\n"
                   "const PORT := 7777\n"
                   "const MAKS_OYUNCU := 16\n"
                   "\n"
                   "func sunucu_baslat() -> bool:\n"
                   "    var peer := ENetMultiplayerPeer.new()\n"
                   "    var hata := peer.create_server(PORT, MAKS_OYUNCU)\n"
                   "    if hata != OK:\n"
                   "        push_error(\"Sunucu acilamadi: %d\" % hata)\n"
                   "        return false\n"
                   "    multiplayer.multiplayer_peer = peer\n"
                   "    multiplayer.peer_connected.connect(_on_baglandi)\n"
                   "    multiplayer.peer_disconnected.connect(_on_ayrildi)\n"
                   "    print(\"Sunucu hazir, port \", PORT)\n"
                   "    return true\n"
                   "\n"
                   "func istemci_baglan(adres: String) -> bool:\n"
                   "    var peer := ENetMultiplayerPeer.new()\n"
                   "    var hata := peer.create_client(adres, PORT)\n"
                   "    if hata != OK:\n"
                   "        return false\n"
                   "    multiplayer.multiplayer_peer = peer\n"
                   "    multiplayer.connected_to_server.connect(_on_baglandi_ok)\n"
                   "    multiplayer.connection_failed.connect(_on_baglanti_hatasi)\n"
                   "    return true\n"
                   "\n"
                   "func _on_baglandi(id: int) -> void:\n"
                   "    print(\"Oyuncu katildi: \", id)\n"
                   "\n"
                   "func _on_ayrildi(id: int) -> void:\n"
                   "    print(\"Oyuncu ayrildi: \", id)")),
        Lesson("godot_ag", "RPC: sunucu-istemci fonksiyon çağrısı",
               "**`@rpc`** işaretleyicisi bir fonksiyonu ağ üzerinden "
               "çağrılabilir yapar. **Parametreler:** (1) **yetki** — "
               "`\"authority\"` (yalnız sunucu çağırır), `\"any_peer\"` "
               "(herkes); (2) **çağrı** — `\"call_remote\"` (yalnız "
               "uzakta), `\"call_local\"` (yerelde de); (3) **aktarım** — "
               "`\"reliable\"` (kayıp yok), `\"unreliable\"`, "
               "`\"unreliable_ordered\"` (hızlı, sıralı). "
               "**Hedefleme:** `rpc_id(id, ...)` tek oyuncuya, "
               "`rpc(...)` herkese. **⚠ Güvenlik:** `any_peer` ile "
               "gelen çağrıda **`multiplayer.get_remote_sender_id()` "
               "doğrulanmalıdır.**",
               ["@rpc", "RPC", "güvenlik"], code=(
                   "extends Node3D\n"
                   "\n"
                   "# Sunucu -> tum istemciler\n"
                   "@rpc(\"authority\", \"call_local\", \"reliable\")\n"
                   "func skor_guncelle(yeni: int) -> void:\n"
                   "    $UI/Skor.text = str(yeni)\n"
                   "\n"
                   "# Istemci -> sunucu (girdi, hizli)\n"
                   "@rpc(\"any_peer\", \"call_remote\", \"unreliable_ordered\")\n"
                   "func girdi_gonder(yon: Vector2) -> void:\n"
                   "    if not multiplayer.is_server():\n"
                   "        return\n"
                   "    var gonderen := multiplayer.get_remote_sender_id()\n"
                   "    # GUVENLIK: yalnizca kendi oyuncusunu kontrol edebilir\n"
                   "    var oyuncu := _oyuncu_bul(gonderen)\n"
                   "    if oyuncu == null:\n"
                   "        return\n"
                   "    oyuncu.girdi_uygula(yon)\n"
                   "\n"
                   "# Sunucu -> tek istemci\n"
                   "@rpc(\"authority\", \"call_remote\", \"reliable\")\n"
                   "func ozel_mesaj(metin: String) -> void:\n"
                   "    print(\"Sunucudan: \", metin)")),
        Lesson("godot_ag", "Master server / proxy server mimarisi",
               "**Master server** iş mantığını tutar: kimlik doğrulama, "
               "oyuncu verisi, maç eşleştirme, ilerleme kaydı. **Proxy "
               "server** önünde durur: TLS sonlandırma, hız sınırlama, "
               "yönlendirme, **kimlik ön denetimi**. **Oyun sunucusu** "
               "maçı çalıştırır (yetkili fizik). "
               "**Akış:** istemci → proxy (TLS) → master (kimlik + "
               "eşleştirme) → **sunucu adresi + bilet** → istemci "
               "**doğrudan** oyun sunucusuna bağlanır → oyun sunucusu "
               "bileti master'a doğrulatır. Bu ayrım, oyun sunucusunun "
               "kimlik yükünü taşımamasını sağlar.",
               ["master server", "proxy", "bilet"], code=(
                   "# MasterServer.gd - HTTP API (kimlik + eslestirme)\n"
                   "extends Node\n"
                   "\n"
                   "var _biletler: Dictionary = {}      # bilet -> oyuncu\n"
                   "var _sunucular: Array = []          # kayitli oyun sunuculari\n"
                   "\n"
                   "func sunucu_kaydet(adres: String, port: int, anahtar: String) -> bool:\n"
                   "    if anahtar != \"GIZLI_SUNUCU_ANAHTARI\":\n"
                   "        push_warning(\"Gecersiz sunucu anahtari\")\n"
                   "        return false\n"
                   "    _sunucular.append({\"adres\": adres, \"port\": port,\n"
                   "                       \"oyuncu\": 0, \"maks\": 16})\n"
                   "    print(\"Sunucu kayitli: \", adres, \":\", port)\n"
                   "    return true\n"
                   "\n"
                   "func bilet_uret(oyuncu_id: int, sunucu_idx: int) -> String:\n"
                   "    var bilet := \"%d-%d-%d\" % [oyuncu_id, sunucu_idx,\n"
                   "                                Time.get_ticks_msec()]\n"
                   "    _biletler[bilet] = {\"oyuncu\": oyuncu_id, \"kullanildi\": false}\n"
                   "    return bilet\n"
                   "\n"
                   "func bilet_dogrula(bilet: String) -> bool:\n"
                   "    var kayit = _biletler.get(bilet, null)\n"
                   "    if kayit == null or kayit[\"kullanildi\"]:\n"
                   "        return false\n"
                   "    kayit[\"kullanildi\"] = true\n"
                   "    return true")),
        Lesson("godot_ag", "Proxy server: istemci yönlendirme",
               "**Proxy** istemci ile oyun sunucusu arasında durur. "
               "Görevleri: **yönlendirme** (uygun sunucuya iletme), "
               "**hız sınırlama**, **bağlantı sayısı sınırı**, "
               "**günlükleme**, **DDoS yumuşatma**. Godot'ta proxy "
               "yazmak: `TCPServer` veya `WebSocketServer` ile bağlantı "
               "kabul et, hedef sunucuya **istemci olarak** bağlan, "
               "paketleri iki yöne aktar. **Alternatif:** nginx/HAProxy "
               "(UDP için `stream` modülü) — üretimde daha yaygın.",
               ["proxy", "yönlendirme", "TCPServer"], code=(
                   "# ProxyServer.gd - baglantilari oyun sunucusuna yonlendirir\n"
                   "extends Node\n"
                   "\n"
                   "@export var proxy_port := 9000\n"
                   "@export var hedef_adres := \"127.0.0.1\"\n"
                   "@export var hedef_port := 7777\n"
                   "@export var maks_baglanti := 200\n"
                   "\n"
                   "var _sunucu := TCPServer.new()\n"
                   "var _istemciler: Array = []\n"
                   "\n"
                   "func _ready() -> void:\n"
                   "    var hata := _sunucu.listen(proxy_port)\n"
                   "    if hata != OK:\n"
                   "        push_error(\"Proxy dinleyemedi: %d\" % hata)\n"
                   "        return\n"
                   "    print(\"Proxy hazir: \", proxy_port, \" -> \",\n"
                   "          hedef_adres, \":\", hedef_port)\n"
                   "\n"
                   "func _process(_delta: float) -> void:\n"
                   "    while _sunucu.is_connection_available():\n"
                   "        if _istemciler.size() >= maks_baglanti:\n"
                   "            var fazla := _sunucu.take_connection()\n"
                   "            fazla.disconnect_from_host()\n"
                   "            continue\n"
                   "        var istemci := _sunucu.take_connection()\n"
                   "        var giden := StreamPeerTCP.new()\n"
                   "        if giden.connect_to_host(hedef_adres, hedef_port) != OK:\n"
                   "            istemci.disconnect_from_host()\n"
                   "            continue\n"
                   "        _istemciler.append({\"gelen\": istemci, \"giden\": giden})\n"
                   "\n"
                   "func _aktar(ikili: Dictionary) -> void:\n"
                   "    var gelen: StreamPeerTCP = ikili[\"gelen\"]\n"
                   "    var giden: StreamPeerTCP = ikili[\"giden\"]\n"
                   "    if gelen.get_available_bytes() > 0:\n"
                   "        giden.put_data(gelen.get_partial_data(4096)[1])\n"
                   "    if giden.get_available_bytes() > 0:\n"
                   "        gelen.put_data(giden.get_partial_data(4096)[1])")),
        Lesson("godot_ag", "Yetkili sunucu: hile önleme",
               "**Kural: istemciye asla güvenme.** İstemci yalnız "
               "**girdi** gönderir ('ileri basılı', 'ateş ettim'), "
               "**sonuç** değil. Sunucu doğrular: **menzil** (hedef "
               "gerçekten yakın mı), **görüş** (görmediği veriyi "
               "gönderme), **hız sınırı** (saniyede en fazla N olay), "
               "**değer aralığı** (konum harita içinde mi). "
               "**`@rpc(\"any_peer\")`** ile gelen her çağrıda "
               "`get_remote_sender_id()` doğrulanır. Hasar, para, "
               "envanter **yalnız sunucuda** değişir.",
               ["hile önleme", "yetkili sunucu", "doğrulama"], code=(
                   "extends Node3D\n"
                   "\n"
                   "const MAKS_HIZ := 12.0\n"
                   "const MAKS_MENZIL := 3.0\n"
                   "\n"
                   "var _son_istek: Dictionary = {}      # oyuncu -> zaman\n"
                   "\n"
                   "# YANLIS: dogrulama yok\n"
                   "# @rpc(\"any_peer\")\n"
                   "# func hasar_ver(hedef: int, miktar: int) -> void:\n"
                   "#     saglik[hedef] -= miktar       # istemci istedigi hasari verir!\n"
                   "\n"
                   "# DOGRU: sunucu dogrular\n"
                   "@rpc(\"any_peer\", \"call_remote\", \"reliable\")\n"
                   "func saldiri_iste(hedef_id: int) -> void:\n"
                   "    if not multiplayer.is_server():\n"
                   "        return\n"
                   "    var gonderen := multiplayer.get_remote_sender_id()\n"
                   "    var simdi := Time.get_ticks_msec()\n"
                   "    # 1) Hiz siniri: saniyede en fazla 5 saldiri\n"
                   "    if simdi - int(_son_istek.get(gonderen, 0)) < 200:\n"
                   "        return\n"
                   "    _son_istek[gonderen] = simdi\n"
                   "    var saldiran := _oyuncu_bul(gonderen)\n"
                   "    var hedef := _oyuncu_bul(hedef_id)\n"
                   "    if saldiran == null or hedef == null:\n"
                   "        return\n"
                   "    # 2) Menzil kontrolu\n"
                   "    if saldiran.global_position.distance_to(hedef.global_position) > MAKS_MENZIL:\n"
                   "        return\n"
                   "    # 3) Hasari SUNUCU belirler\n"
                   "    hedef.hasar_uygula(saldiran.saldiri_gucu)\n"
                   "    rpc(\"saglik_guncelle\", hedef_id, hedef.saglik)")),
    ]


# ─── Müfredat ───────────────────────────────────────────────────────────────

def build_godot_curriculum(
        layers: Sequence[str] | None = None) -> List[Lesson]:
    """
    Godot oyun üretim müfredatını üretir.

    Args:
        layers: Yalnızca bu katmanlar (None = hepsi, sıralı)

    Returns:
        Ders listesi
    """
    builders = {
        "godot_temel": _temel_lessons,
        "godot_3b": _ucb_lessons,
        "godot_nesne": _nesne_lessons,
        "godot_materyal": _materyal_lessons,
        "godot_karakter": _karakter_lessons,
        "godot_seviye": _seviye_lessons,
        "godot_girdi": _girdi_lessons,
        "godot_arayuz": _arayuz_lessons,
        "godot_ses": _ses_lessons,
        "godot_ag": _ag_lessons,
    }
    istenen = list(layers) if layers else list(GODOT_LAYERS)
    dersler: List[Lesson] = []
    for ad in istenen:
        uretici = builders.get(ad)
        if uretici is not None:
            dersler.extend(uretici())
    return dersler


def godot_stats(layers: Sequence[str] | None = None) -> Dict[str, object]:
    """Godot müfredatı istatistikleri."""
    dersler = build_godot_curriculum(layers)
    per_layer: Dict[str, Dict[str, int]] = {}
    for ders in dersler:
        kayit = per_layer.setdefault(ders.layer,
                                     {"lessons": 0, "chars": 0, "code": 0})
        kayit["lessons"] += 1
        kayit["chars"] += len(ders.as_document())
        kayit["code"] += 1 if ders.code.strip() else 0
    return {
        "total_lessons": len(dersler),
        "total_chars": sum(item["chars"] for item in per_layer.values()),
        "per_layer": per_layer,
        "layers": [GODOT_LABELS.get(a, a)
                   for a in (layers or GODOT_LAYERS)],
    }


__all__ = [
    "GODOT_LAYERS", "GODOT_LABELS", "build_godot_curriculum", "godot_stats",
]
