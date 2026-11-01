"""
Godot — efektler (.tscn içinde) ve ses ekleme.

Bu katman gerçek bir projede (Balon 3D) ölçülen API ile yazıldı:
GPUParticles3D, ParticleProcessMaterial, AudioStreamPlayer, AudioStreamPlayer3D.

Katman adı: `godot_efekt_ses`
"""

from __future__ import annotations

from typing import List

from .curriculum import Lesson

LAYER = "godot_efekt_ses"
LABEL = "Efektler (.tscn) ve Ses"


def _efekt_ses_lessons() -> List[Lesson]:
    """Parçacık efektleri ve ses düğümleri."""

    return [
        Lesson(
            LAYER,
            "Efektler .tscn içinde nasıl tanımlanır?",
            "Bir parçacık efekti sahnede üç düğümden oluşur: (1) `GPUParticles3D` "
            "düğümü (amount, lifetime, emitting), (2) `process_material` olarak "
            "`ParticleProcessMaterial` (yön, hız, yerçekimi, ölçek), "
            "(3) `draw_pass_1` olarak bir `Mesh` (çoğu zaman `QuadMesh`). "
            "Malzeme ve mesh `[sub_resource]` olarak sahne içinde tanımlanır; "
            "böylece efekt başka sahneye kopyalanabilir. Sahne açıldığında "
            "`emitting = false` ise parçacık görünmez; koddan `emitting = true` "
            "ile başlatılır.",
            ["tscn", "GPUParticles3D", "process_material", "draw_pass_1"],
            """[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://main.gd" id="1_main"]

[sub_resource type="ParticleProcessMaterial" id="PPM_toz"]
emission_shape = 3            # EMISSION_SHAPE_SPHERE
emission_sphere_radius = 0.35
direction = Vector3(0, -0.35, 0)
spread = 75.0
flatness = 0.85
initial_velocity_min = 0.4
initial_velocity_max = 1.6
gravity = Vector3(0, 0.35, 0)
damping_min = 0.8
damping_max = 1.6

[sub_resource type="QuadMesh" id="Quad_toz"]
size = Vector2(0.4, 0.4)

[node name="Main" type="Node3D"]
script = ExtResource("1_main")

[node name="Toz" type="GPUParticles3D" parent="."]
amount = 32
lifetime = 1.1
emitting = false
local_coords = false
visibility_aabb = AABB(-6, -3, -6, 12, 8, 12)
process_material = SubResource("PPM_toz")
draw_pass_1 = SubResource("Quad_toz")
""",
        ),
        Lesson(
            LAYER,
            "Toz efekti: parçacıklar neden ilk karede patlar?",
            "`GPUParticles3D.amount_ratio` varsayılan **1.0**'dır; `emitting` "
            "açıldığı ilk karede tüm parçacık havuzu aynı anda doğar ve ekran "
            "toz bulutuyla dolar. Çözüm: `amount_ratio = 0.0` ile başla, "
            "yoğunluğu hıza/orana göre 0..1 arasında sür. Ayrıca "
            "`local_coords = false` parçacıkları dünyada bırakır (iz bırakır); "
            "`visibility_aabb` kamera dışına çıkınca kesilmeyi önler. "
            "`preprocess = 0.0` ile efekt ilk karede önceden ısıtılmaz.",
            ["GPUParticles3D", "amount_ratio", "local_coords", "visibility_aabb"],
            """func _build_dust() -> void:
    dust = GPUParticles3D.new()
    dust.amount = 32
    dust.lifetime = 1.1
    dust.preprocess = 0.0
    dust.explosiveness = 0.0
    dust.randomness = 0.6
    dust.fixed_fps = 0
    dust.local_coords = false              # dünyada kalır -> iz
    dust.emitting = false
    dust.amount_ratio = 0.0                # ilk karede patlama yok
    dust.visibility_aabb = AABB(Vector3(-6, -3, -6), Vector3(12, 8, 12))
    add_child(dust)

func _dust_guncelle(hiz: float) -> void:
    dust.amount_ratio = clampf(hiz / 6.0, 0.0, 1.0)
    dust.emitting = hiz > 0.5
""",
        ),
        Lesson(
            LAYER,
            "Tek seferlik efekt (patlama) nasıl yapılır?",
            "Patlama için `one_shot = true` ve `explosiveness = 1.0` kullanılır: "
            "tüm parçacıklar tek karede doğar. `emitting` bir kez `true` yapılır, "
            "efekt bitince Godot kendisi `false`'a çeker (`one_shot` modunda). "
            "Yeniden tetiklemek için `restart()` çağrılır. Sürekli efektte "
            "(toz, duman) `one_shot = false`, `explosiveness = 0.0` kullanılır.",
            ["one_shot", "explosiveness", "restart", "emitting"],
            """# Sürekli (toz)
toz.one_shot = false
toz.explosiveness = 0.0

# Tek seferlik (patlama)
patlama.one_shot = true
patlama.explosiveness = 1.0
patlama.emitting = true      # bir kez; bitince Godot kapatır

# Yeniden tetikle
patlama.restart()
patlama.emitting = true
""",
        ),
        Lesson(
            LAYER,
            "Parçacık malzemesi: yön, yayılma, yerçekimi, ölçek",
            "`ParticleProcessMaterial` parçacığın fiziğini belirler. "
            "`emission_shape` doğum yüzeyini seçer (0=nokta, 1=küre, 3=küre, "
            "4=kutu — sürüme göre sabitler değişir; editörde seçmek en güvenlisi). "
            "`direction` + `spread` (derece) fırlatma konisini, `flatness` "
            "yassılığı, `gravity` ivmeyi, `damping_min/max` sönümü, "
            "`scale_min/max` parçacık boyutunu belirler. "
            "`scale_curve` ile ömür boyunca küçülme/büyüme verilir; "
            "toz için yavaş büyüyüp sönme tipiktir.",
            ["ParticleProcessMaterial", "emission_shape", "spread", "gravity", "scale_curve"],
            """var pm := ParticleProcessMaterial.new()
pm.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_SPHERE
pm.emission_sphere_radius = 0.35
pm.direction = Vector3(0, -0.35, 0)
pm.spread = 75.0
pm.flatness = 0.85
pm.initial_velocity_min = 0.4
pm.initial_velocity_max = 1.6
pm.gravity = Vector3(0, 0.35, 0)
pm.damping_min = 0.8
pm.damping_max = 1.6
pm.scale_min = 0.35
pm.scale_max = 0.9
pm.scale_curve = _dust_scale_curve()     # büyüyüp sönen eğri
""",
        ),
        Lesson(
            LAYER,
            "Ses ekleme: AudioStreamPlayer (.tscn içinde)",
            "Arayüz/2D sesleri için `AudioStreamPlayer` kullanılır. Sahnede "
            "`[ext_resource type=\"AudioStream\" ...]` ile dosya referanslanır "
            "(.ogg veya .wav). Önemli alanlar: `stream`, `volume_db` (0 = tam, "
            "-6 = yarı ses), `autoplay` (sahne açılınca çalsın), "
            "`max_polyphony` (aynı anda kaç ses), `bus` (ses kanalı). "
            "Ses dosyası içe aktarılırken `.ogg` için `loop` ayarı import "
            "panelinden yapılır; döngülü müzik için şart.",
            ["AudioStreamPlayer", "stream", "volume_db", "autoplay", "bus"],
            """[gd_scene load_steps=2 format=3]

[ext_resource type="AudioStream" path="res://assets/ses/puan.ogg" id="1_puan"]

[node name="PuanSesi" type="AudioStreamPlayer"]
stream = ExtResource("1_puan")
volume_db = -6.0
autoplay = false
max_polyphony = 3
bus = &"Master"
""",
        ),
        Lesson(
            LAYER,
            "Ses ekleme: AudioStreamPlayer3D (dünyadaki ses)",
            "3B sahnede ses konumlu olmalıdır: `AudioStreamPlayer3D` düğümü "
            "kaynağın yerine konur. `max_distance` duyulma menzili (metre), "
            "`unit_size` sesin ne kadar uzağa taşındığı, `attenuation_model` "
            "düşüş eğrisi (0=ters kare, 1=ters, 2=logaritmik, 3=doğrusal). "
            "Ses kameraya göre sağdan/soldan gelir. `volume_db` kaba ayardır; "
            "ince ayar için `AudioStreamPlayer3D` + `AudioServer` bus volume "
            "kullanılır. Zıplama, patlama, adım sesleri için tipiktir.",
            ["AudioStreamPlayer3D", "max_distance", "unit_size", "attenuation_model"],
            """[gd_scene load_steps=2 format=3]

[ext_resource type="AudioStream" path="res://assets/ses/zoplama.ogg" id="1_z"]
[ext_resource type="Script" path="res://player.gd" id="2_p"]

[node name="Player" type="Node3D"]
script = ExtResource("2_p")

[node name="ZiplamaSesi" type="AudioStreamPlayer3D" parent="."]
stream = ExtResource("1_z")
volume_db = -4.0
max_distance = 40.0
unit_size = 8.0
attenuation_model = 0
autoplay = false
""",
        ),
        Lesson(
            LAYER,
            "Koddan ses ve efekt tetikleme",
            "Ses ve efekt düğümleri sahneye konur, koddan **tetiklenir**. "
            "Ses: `play()` (baştan çalar), `stop()`, `pitch_scale` (hız/ton), "
            "çoklu ses için `max_polyphony > 1` veya ayrı oyuncular. "
            "Efekt: `emitting = true/false`, tek seferlikte `restart()`. "
            "Değişkeni `@onready var ses := $ZiplamaSesi` ile bir kez bağla; "
            "her karede `get_node` çağırma. Kısa seslerde `play()` çağrısı "
            "önceki sesi kesmez, üst üste biner (polyphony).",
            ["play", "pitch_scale", "emitting", "@onready", "restart"],
            """@onready var ziplama_sesi: AudioStreamPlayer3D = $ZiplamaSesi
@onready var toz: GPUParticles3D = $Toz

func zipla() -> void:
    ziplama_sesi.pitch_scale = randf_range(0.95, 1.05)
    ziplama_sesi.play()
    toz.emitting = true

func yere_indi() -> void:
    toz.emitting = false

func patla() -> void:
    toz.restart()
    toz.emitting = true
""",
        ),
        Lesson(
            LAYER,
            "Efekt/ses düğümlerini headless testte doğrulamak",
            "Sahne testinde düğümlerin varlığı ve bağlantıları ölçülür: "
            "`get_node_or_null(\"Toz\")` `GPUParticles3D` mi, `process_material` "
            "atanmış mı, `draw_pass_1` boş mu; ses düğümünde `stream` **null "
            "değil** mi (dosya yolu yanlışsa null olur ve ses çıkmaz). "
            "Bu kontroller ses dosyası silindiğinde veya düğüm adı değiştiğinde "
            "sessiz bozulmayı yakalar. Test `quit(1)` ile hata döndürmeli.",
            ["headless", "AudioStream", "process_material", "doğrulama"],
            """extends SceneTree

func _init() -> void:
    var sahne := load("res://main.tscn") as PackedScene
    var kok := sahne.instantiate()
    root.add_child(kok)
    await process_frame

    var toz := kok.get_node_or_null("Toz")
    assert(toz is GPUParticles3D)
    assert(toz.process_material != null)
    assert(toz.draw_pass_1 != null)

    var ses := kok.get_node_or_null("ZiplamaSesi")
    assert(ses is AudioStreamPlayer3D)
    assert(ses.stream != null)          # yol yanlışsa null olur

    print("  [PASS] efekt ve ses düğümleri bağlı")
    quit(0)
""",
        ),
    ]
