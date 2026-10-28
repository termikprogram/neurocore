"""
NeuroCore Godot Oyun Motoru Kontrolü
=====================================
Ajanın bir Godot projesini **gerçek motorla** oluşturması, içini görmesi,
düzenlemesi, çalıştırması ve dışa aktarması için gereken katman.

Neden alt süreç, neden "import godot" değil?
    Godot'un Python bağlaması yoktur; motorla konuşmanın tek güvenilir yolu
    çalıştırılabilir dosyayı çağırmaktır. Bu modül **sahte bir API taklidi
    yapmaz**: gerçek motoru çalıştırır, çıktısını ayrıştırır ve sonucu
    yapılandırılmış sözlüğe çevirir. Böylece "çalıştı mı?" sorusunun yanıtı
    motorun kendisinden gelir.

Neden GDScript üretiyoruz?
    Sahne düzenlemek (düğüm ekleme, özellik atama, kaydetme) `.tscn` metnini
    elle yazmakla yapılmamalıdır: kaynak kimlikleri (`ext_resource`/`sub_resource`),
    düğüm sahipliği (`owner`) ve varsayılan değer eleme mantığı bozulur.
    Bunun yerine kısa bir GDScript dosyası üretilir, `--headless --script` ile
    çalıştırılır ve sonucu makine tarafından okunabilir biçimde geri alınır.
    Üretilen betikler `<proje>/nc_tools/` altında yaşar; bunlar geçici araç
    dosyalarıdır, projenin oyun koduna karışmaz.

Neden `--path` ve asla `cd` değil?
    Süreç dizinini değiştirmek ajanın geri kalanı için görünmez bir yan etki
    yaratır ve eşzamanlı çağrılarda yarış koşuluna yol açar. Godot'un kendi
    `--path` argümanı proje kökünü açıkça bildirir; bu modül her çağrıda onu
    kullanır.

Yazdırma protokolü (üretilen GDScript → Python)
-----------------------------------------------
Üretilen her betik sonucu **tek satırda** ve şu önekle yazdırır::

    NC_RESULT {"ok": true, "action": "create_scene", ...}

Kurallar:
    * `NC_RESULT ` önekinden sonraki kısım tek satırlık geçerli JSON'dur
      (`JSON.stringify` ile üretilir; satır sonu içermez).
    * Bir çalıştırmada **en fazla bir** `NC_RESULT` satırı olur; Python
      tarafı sonuncuyu esas alır, yoksa ham çıktıya ve çıkış koduna düşer.
    * Betik başarıda `quit(0)`, hatada `quit(1)` ile çıkar. Python hem çıkış
      kodunu hem JSON'un `ok` alanını hem de hata kalıplarını birlikte
      değerlendirir; üçü çelişirse **hata kazanır** (yanlış "başarılı"
      raporlamak, yanlış "başarısız" raporlamaktan daha zararlıdır).
    * `NC_RESULT` dışındaki satırlar insan içindir; `stdout`/`stderr`
      alanlarında ham olarak döner.
    * Motorun `... were leaked at exit`, `Pages in use exist at exit` gibi
      kapanış gürültüsü hata **sayılmaz** (bkz. `GD_NOISE_PATTERNS`).

Motorun doğrulanmış tuzakları (Godot 4.8-dev7)
----------------------------------------------
Bu maddeler gerçek motorda tek tek sınandı; her biri bir **sessiz yalan**
üretiyordu, bu yüzden modül özel önlemler alır:

1. **Çıkış kodu başarı göstergesi DEĞİLDİR.** GDScript'te bir *çalışma
   zamanı* hatası (`SCRIPT ERROR`) stderr'e yazılır, `quit()` hiç çağrılmaz ve
   süreç **sonsuza kadar asılı kalır** (çıkış kodu gelmez). Bu yüzden her
   `--script` çağrısına `--quit-after <N>` (varsayılan 500 iterasyon) eklenir:
   betik `quit()` çağırmasa bile motor N iterasyon sonra kapanır. `ok` kararı
   üç sinyalin birleşimidir: `exit_code == 0` **ve** çıktıda hata işareti
   olmaması **ve** (varsa) `NC_RESULT` içindeki `ok` alanı. Asla
   `ok = (exit_code == 0)` değildir.

2. **`PackedScene.pack()` sessizce boş sahne yazabilir.** `owner` yalnızca
   ağaçta bir atasöz konumu gösteriyorsa kabul edilir; `owner` `add_child()`
   çağrısından **önce** atanırsa motor `ERROR: Invalid owner...` yazar ama
   `pack()` ve `ResourceSaver.save()` yine de `OK` döner ve diske **boş** bir
   sahne yazılır. Bu yüzden üretilen betikler önce `add_child()` sonra
   `owner =` yapar ve kayıttan sonra sahneyi **yeniden yükleyip düğüm sayısını
   karşılaştırır** (`result["verified"]`). Sayı tutmazsa `ok=False` döner.

3. **Dışa aktarma, GDScript hataları varken de exit 0 verir** ve paketi yazar.
   Bu yüzden `export_project` sonucu yalnızca çıkış koduna değil, üretilen
   dosyanın varlığına ve çıktıdaki hata işaretlerine de bakar.

4. **`--script` yalnızca düzenleyici (editor) yapılarında ve
   `disable_path_overrides=false` ile derlenmiş şablonlarda çalışır.** Yalnızca
   bir dışa aktarma şablonu (`windows_release_x86_64.exe`) varsa `--script`
   desteklenmez; modül bunu ayırt edip anlaşılır hata döner.

5. **Bilinmeyen bayraklar sessizce yok sayılır.** Bu yüzden "bayrak reddedildi"
   hiçbir zaman kanıt sayılmaz; sonuç dosya sisteminden doğrulanır.

Graceful degradation
--------------------
Godot kurulu değilse hiçbir metot istisna fırlatmaz; her çağrı
`{"ok": False, "error": "...", "hint": "..."}` döner. Bu, ajanın "motor yok"
durumunu bir çökme değil, çözülebilir bir eksik olarak görmesini sağlar.

`.tscn` metin kuralları (elle yazılan sahneler için)
----------------------------------------------------
    * Godot 4: `[gd_scene format=3]` (3.x `format=2` idi).
    * Bölüm sırası anlamlıdır: `gd_scene` → `ext_resource` → `sub_resource`
      → `node` → `connection`.
    * `load_steps` 4.6'dan beri kullanımdan kaldırıldı; yazılmaz.
    * Kaynak kimlikleri ayrı ad alanlarındadır: `ExtResource("1_ab")` ile
      `SubResource("Box_ab")` aynı metni taşıyabilir.
    * Kök düğümde `parent=` yoktur; doğrudan çocuklar `parent="."` kullanır.
    * Varsayılana eşit özellikler yazılmaz; elle eklenen varsayılan değerler
      motor yeniden kaydettiğinde düşer.

Kullanım:
    godot = GodotController()
    godot.initialize(workspace="E:/PROJELER/Oyunum", config=config, shell=shell)
    registry.register_many(godot.tools())

    godot.create_project("E:/PROJELER/Oyunum", name="Oyunum", template="3d")
    godot.create_scene("res://Level1.tscn", [{"type": "Node3D", "name": "Root"}])
    print(godot.run_script("extends SceneTree\\nfunc _init():\\n\\tprint(1)\\n\\tquit(0)\\n"))
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neurocore.agent.shell import ShellIntegration
from neurocore.agent.tools import Safety, ToolContext, ToolDef, ToolParameter
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.godot")


# ─── Sabitler ───────────────────────────────────────────────────────────────

#: Godot çalıştırılabilirinin aranacağı ortam değişkenleri (öncelik sırasıyla)
GODOT_ENV_VARS: Tuple[str, ...] = (
    "NEUROCORE_GODOT", "GODOT_CONSOLE", "GODOT4", "GODOT_BIN", "GODOT",
)

#: Bu makinede doğrulanmış kurulum yolları (PATH'te olmadığı için gerekli)
GODOT_KNOWN_EXES: Tuple[str, ...] = (
    r"E:\AppsTools\Godot_v4.8-dev7_win64\Godot_v4.8-dev7_win64_console.exe",
    r"E:\AppsTools\Godot_v4.8-dev7_win64\Godot_v4.8-dev7_win64.exe",
    r"D:\AppsTools\Godot_v4.8-dev7_win64\Godot_v4.8-dev7_win64_console.exe",
)

#: Yeni kurulumların bulunabileceği kök dizinler (sınırlı derinlikte taranır)
GODOT_SCAN_ROOTS: Tuple[str, ...] = (
    r"E:\AppsTools", r"D:\AppsTools", r"C:\AppsTools",
    r"C:\Program Files", r"C:\Program Files (x86)",
)

#: Üretilen yardımcı GDScript'lerin proje içindeki dizini
NC_TOOLS_DIR = "nc_tools"

#: GDScript'in sonucu yazdırdığı önek (tek satır JSON izler)
NC_RESULT_PREFIX = "NC_RESULT "

#: Godot 4 proje dosyası sürümü
GODOT_CONFIG_VERSION = 5

#: Projede üretilen/aranan Godot dosya uzantıları
SCENE_SUFFIXES: Tuple[str, ...] = (".tscn", ".scn")
RESOURCE_SUFFIXES: Tuple[str, ...] = (".tres", ".res")

#: Motorun normal kapanışta ürettiği, **hata sayılmayan** gürültü kalıpları.
#: (Bunlar başarılı çalıştırmalarda da görünür; filtrelenmezse her çağrı
#: yanlışlıkla "başarısız" raporlanır.)
GD_NOISE_PATTERNS: Tuple[str, ...] = (
    r"were leaked at exit",
    r"instances were leaked at exit",
    r"resources still in use at exit",
    r"Pages in use exist at exit",
    r"Leaked instance dependency",
    r"ObjectDB instances were leaked",
    r"RID allocations of type",
    r"Scan thread aborted",
    r"^\s*at:\s+\S+\s*\(",                    # yığın konumu satırları
    r"^\s*GDScript backtrace",
    r"^\s*\[\d+\]\s+\w+\s+\(",                # GDScript geri izleme satırları
    r"^\s*\*\s*$",
)

#: Dışa aktarma şablonu dosya adları (platform → beklenen dosya önekleri)
EXPORT_TEMPLATE_FILES: Dict[str, Tuple[str, ...]] = {
    "windows": ("windows_release_x86_64.exe", "windows_debug_x86_64.exe"),
    "linux": ("linux_release.x86_64", "linux_debug.x86_64"),
    "macos": ("macos.zip",),
    "web": ("web_release.zip", "web_debug.zip"),
    "android": ("android_release.apk", "android_debug.apk"),
    "ios": ("ios.zip",),
}

#: Preset `platform` alanından şablon ailesine eşleme
PLATFORM_TEMPLATE_FAMILY: Tuple[Tuple[str, str], ...] = (
    ("windows", "windows"), ("linux", "linux"), ("macos", "macos"),
    ("mac os", "macos"), ("web", "web"), ("android", "android"), ("ios", "ios"),
)

#: Düğüm özelliklerini okurken kullanılan güvenli denetim sınırı
MAX_INSPECT_NODES = 4000


class GodotError(Exception):
    """Godot aracı hatası (yalnızca programcı hatası durumunda fırlatılır)."""


# ─── Hata Kalıpları ─────────────────────────────────────────────────────────

#: Godot çıktısındaki hata kalıpları → (tür, Türkçe öneri).
#:
#: Not: Godot 4.8'de **editör** mesajları yerelleştirilir (bu makinede Türkçe
#: görülebilir: "Şablon Hazırla: Belirtilen dışa aktarım yolu mevcut değil.")
#: ama motorun çalışma zamanı hataları (`SCRIPT ERROR`, `Parse Error`, `Node
#: not found`) İngilizce kalır. Bu yüzden yaygın editör mesajlarının hem
#: İngilizce hem Türkçe biçimleri aşağıda listelenmiştir.
GODOT_ERROR_HINTS: List[Tuple[str, str, str]] = [
    # ── GDScript sözdizimi ──
    (r"Parse Error:\s*(?P<msg>.+)", "parse_error",
     "GDScript sözdizimi hatası — dosya:satır konumundaki ifadeyi düzeltin; "
     "girinti tutarlı olmalı ve satır sonunda eksik ifade bırakılmamalı"),
    # ── SCRIPT ERROR alt türleri (genel SCRIPT ERROR kalıbından ÖNCE gelmeli) ──
    (r"Invalid access to (?:property|index) '[^']*' on a base object of type 'Nil'",
     "null_instance",
     "null bir nesnenin özelliğine erişildi — düğüm/nesne oluşturma başarısız "
     "olmuş (ClassDB.instantiate null döndü mü?) veya get_node yolu yanlış"),
    (r"Invalid assignment of property or key '[^']*'.*on a base object of type 'Nil'",
     "null_instance",
     "null bir nesneye değer atandı — `var x = ClassDB.instantiate(...)` null "
     "dönmüş olabilir; `if x != null` ile doğrulayın"),
    (r"on a base object of type 'Nil'", "null_instance",
     "İşlem null bir nesne üzerinde yapıldı — nesnenin gerçekten oluştuğunu "
     "doğrulayın (null dönen bir çağrı zinciri var)"),
    (r"Invalid (?:get|set) index '[^']*'", "invalid_index",
     "Dizi/sözlük indeksi geçersiz — boyutu ve anahtar tipini denetleyin"),
    (r"Attempt to call function '[^']*' in base 'null instance'", "null_instance",
     "null nesne üzerinde çağrı — `get_node()` yolu yanlış veya nesne "
     "oluşturulmamış; `get_node_or_null` + null denetimi kullanın"),
    (r"Trying to assign value of type '[^']*' to a variable of type", "type_mismatch",
     "Tür uyuşmazlığı — dizeyi sayıya/vektöre açıkça çevirin (int(), float(), Vector3())"),
    (r'Identifier "(?P<msg>[^"]+)" not declared in the current scope', "identifier_not_found",
     "Tanımsız ad: değişken/sınıf adı yanlış yazılmış ya da `class_name` "
     "bildirimi eksik; `preload`/`load` yolunu kontrol edin"),
    (r'Could not find type "(?P<msg>[^"]+)" in the current scope', "missing_class_name",
     "Sınıf bulunamadı — `class_name X` bildirimi eksik olabilir; dosyayı "
     "godot.import_project ile yeniden taratın ve adı birebir kullanın"),
    (r'Class "(?P<msg>[^"]+)" hides a global script class', "class_name_conflict",
     "Aynı `class_name` iki dosyada tanımlı — birini yeniden adlandırın"),
    (r"Invalid call\. Nonexistent function '(?P<msg>[^']+)' in base '(?P<base>[^']+)'",
     "invalid_call",
     "Olmayan fonksiyon çağrısı — Godot 4 API'si mi? (3.x `instance()` yerine "
     "`instantiate()`, `yield` yerine `await`, `OS.set_exit_code()` yoktur)"),
    (r"Invalid call\. Nonexistent property '(?P<msg>[^']+)'", "invalid_property",
     "Olmayan özellik — godot.inspect_scene ile düğümün gerçek özelliklerini "
     "okuyun; `godot.set_node_property` yanıtındaki `similar` listesine bakın"),
    # ── Betik yüklenemedi / yanlış taban sınıf ──
    (r"doesn't inherit from SceneTree or MainLoop", "wrong_base_class",
     "Yardımcı betik `extends SceneTree` olmalı ve `func _init()` kullanmalı; "
     "`quit(kod)` yalnızca SceneTree/MainLoop üzerinde çıkış kodunu belirler "
     "(`OS.set_exit_code()` Godot 4'te YOKTUR)"),
    (r"Failed to load script \"(?P<msg>[^\"]+)\" with error \"Parse error\"", "failed_load_script",
     "Betik yüklenemedi (parse error) — dosyadaki sözdizimi hatasını düzeltin; "
     "godot.diagnose aynı çıktıdaki Parse Error satırını dosya:satır ile verir"),
    (r"Can't load the script|Failed to load script", "failed_load_script",
     "Betik yüklenemedi — dosya yolu, taban sınıf (`extends SceneTree`) ve "
     "sözdizimi doğru mu? Ayrıca `--script` yalnızca düzenleyici yapılarında çalışır"),
    (r"Can't run project: no main scene defined", "no_main_scene",
     "Çalıştırılacak ana sahne yok — godot.run_scene ile sahneyi açıkça verin"),
    # ── Genel SCRIPT ERROR (spesifik kalıplar yukarıda) ──
    (r"SCRIPT ERROR:", "script_error",
     "Betik çalışma zamanı hatası — geri izlemedeki satıra bakın. DİKKAT: bu "
     "durumda Godot exit 0 verebilir ve `quit()` çağrılmadığı için süreç asılı "
     "kalır; modül `--quit-after` ile kapatır ve `ok=False` döner"),
    # ── Sahne, sahiplik, düğümler ──
    (r"Invalid owner\.|Owner must be an ancestor in the tree", "invalid_owner",
     "Düğüm sahipliği geçersiz — `owner` yalnızca `add_child()` SONRASINDA "
     "atanabilir. Yanlış sıra `pack()`un sessizce BOŞ sahne yazmasına yol açar"),
    (r'Node not found:\s*"(?P<msg>[^"]+)"', "node_not_found",
     "Düğüm yolu yanlış — sahne ağacını godot.inspect_scene ile doğrulayın; "
     "göreli yollar sahne köküne göredir"),
    (r"Parent node is busy setting up children", "parent_busy",
     "Üst düğüm henüz hazır değil — `add_child` çağrısını "
     "`await get_tree().process_frame` sonrasına taşıyın"),
    (r"Failed to load resource '(?P<msg>[^']+)'", "resource_load",
     "Kaynak yüklenemedi — yolu (`res://`) ve dosyanın varlığını doğrulayın"),
    (r"Failed loading resource:\s*(?P<msg>\S+)", "resource_load",
     "Kaynak yüklenemedi — .tscn/.tres bozuk olabilir veya bağımlılığı eksik"),
    (r"Cannot open file ['\"](?P<msg>[^'\"]+)['\"]", "cannot_open",
     "Dosya açılamadı — `res://` yolu proje köküne göreli mi ve dosya var mı?"),
    (r"Cannot open file \"?res://", "cannot_open",
     "Dosya açılamadı — `--path` ile verilen proje kökü ile `res://` yolunu "
     "karşılaştırın; dosya `nc_tools/` altında gerçekten oluştu mu?"),
    (r"Scene (?:file|format) .* is (?:invalid|corrupt)", "scene_corrupt",
     "Sahne dosyası bozuk — yedekten dönün veya godot.create_scene ile yeniden üretin"),
    # ── Motor koşulları ──
    (r'Condition "(?P<msg>[^"]+)" is true', "engine_condition",
     "Motor iç koşulu tetiklendi — genellikle geçersiz argüman anlamına gelir; "
     "iletideki koşul metni hangi denetimin düştüğünü söyler"),
    (r'Parameter "(?P<msg>[^"]+)" is null', "null_parameter",
     "Zorunlu parametre null — nesneyi oluşturduktan sonra geçirin"),
    (r"Can't change this state while flushing queries", "physics_flush",
     "Fizik sorgusu sürerken değişiklik yapılamaz — `call_deferred` kullanın"),
    # ── Dışa aktarma ──
    (r"No export template found|Export template not found|"
     r"Export templates for this version of Godot are not installed|"
     r"[Şş]ablon bulunamad[ıi]|[Şş]ablon Hazırla", "missing_templates",
     "Dışa aktarma şablonları kurulu değil — Godot Düzenleyici > Düzenleyici > "
     "Dışa Aktarma Şablonlarını Yönet ile bu sürümün şablonlarını indirin"),
    (r"doesn't have an `?export_presets\.cfg`?", "missing_export_presets",
     "export_presets.cfg yok — Godot Düzenleyici > Proje > Dışa Aktar ile bir "
     "preset oluşturup kaydedin"),
    (r"Invalid export preset name:\s*(?P<msg>.+?)\.?\s*$", "invalid_preset",
     "Preset adı yanlış — godot.list_export_presets ile geçerli adları okuyun "
     "(liste, aynı çıktının devamında yazdırılır)"),
    (r"export path does not exist|dışa aktarım yolu mevcut değil", "export_path_missing",
     "Hedef dizin yok — Godot çıktı yolunu **proje dizinine göre** çözer ve "
     "klasör önceden var olmalıdır; godot.export_project bunu kendisi oluşturur"),
    (r"Project export for preset .* failed", "export_failed",
     "Dışa aktarma başarısız — yukarıdaki ilk ERROR satırı asıl nedendir"),
    # ── Proje ──
    (r"Error: Couldn't load project data at path|Project file not found|"
     r"Unable to load project|proje dosyası bulunamadı", "project_missing",
     "project.godot bulunamadı — `--path` ile verilen dizinin Godot proje kökü "
     "olduğundan emin olun (godot.is_godot_project)"),
    (r"No main scene defined|main scene .* not found", "no_main_scene",
     "Ana sahne tanımlı değil — project.godot içindeki "
     "`application/run/main_scene` değerini ayarlayın"),
    # ── Gölgelendirici ──
    (r"SHADER ERROR|Shader compilation failed", "shader_error",
     "Gölgelendirici derleme hatası — satır numarası `.gdshader` dosyasına aittir"),
    # ── Genel yakalayıcı (EN SONDA): tanınmayan motor hataları ──
    (r"^\s*ERROR:\s*(?P<msg>.+)", "engine_error",
     "Motor hatası — ileti tam olarak neyin reddedildiğini söyler; ilgili "
     "godot.* aracının `hint` alanına ve godot.diagnose özetine bakın"),
    (r"^\s*FAILED:\s*(?P<msg>.+)", "engine_failure",
     "Motor işlemi başarısız bildirdi — çıktıdaki ilk hata satırı asıl nedendir"),
]

_COMPILED_HINTS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(pattern, re.IGNORECASE), kind, hint)
    for pattern, kind, hint in GODOT_ERROR_HINTS
]

_COMPILED_NOISE: List[re.Pattern] = [
    re.compile(pattern, re.IGNORECASE) for pattern in GD_NOISE_PATTERNS
]

#: `at: GDScript::reload (res://x.gd:12)` veya `[0] _init (res://x.gd:12)`
_LOCATION_RE = re.compile(
    r"\((res://[^:()]+|/root/[^:()]+|[A-Za-z]:[\\/][^:()]+):(\d+)\)"
)
_BARE_LOCATION_RE = re.compile(r"\b(res://[^\s:()\"']+):(\d+)\b")


# ─── GDScript Şablonları ────────────────────────────────────────────────────

#: Üretilen her betiğin başlığı. `extends` ilk ifade olmak zorundadır.
_GD_HEADER = (
    "# NeuroCore tarafından üretildi — elle düzenlemeyin.\n"
    "# Çalıştırma: <godot> --headless --path <proje> --script <bu dosya>\n"
    "extends SceneTree\n"
)

#: Tüm üretilen betiklerin paylaştığı yardımcı fonksiyonlar.
#: Girinti olarak **boşluk** kullanılır (GDScript bunu kabul eder ve Python
#: içinde sekmelerle uğraşmaktan daha güvenlidir).
_GD_PRELUDE = '''

const NC_RESULT_PREFIX: String = "NC_RESULT "


func nc_emit(payload: Dictionary) -> void:
    print(NC_RESULT_PREFIX + JSON.stringify(payload))


func nc_instantiate(type_name: String) -> Object:
    # Godot 4: ClassDB.instantiate (Godot 3'teki ClassDB.instance DEĞİL).
    # class_exists denetimi, motorun "Cannot get class" hatasını önler.
    if not ClassDB.class_exists(type_name):
        return null
    var created: Variant = ClassDB.instantiate(type_name)
    if created == null:
        return null
    return created


func nc_instantiate_error(type_name: String) -> String:
    if not ClassDB.class_exists(type_name):
        return "bilinmeyen sınıf: '" + type_name + "' — Godot 4 sınıf adı olduğundan emin olun (ör. Node3D, MeshInstance3D, Camera3D)"
    return "sınıf örneklenemedi (soyut veya oluşturulamaz): '" + type_name + "'"


func nc_numbers(text: String, count: int) -> Array:
    # "1,2,3" / "Vector3(1; 2; 3)" / "(1 2 3)" biçimlerini sayıya çevirir.
    var cleaned: String = text
    for junk in ["Vector4", "Vector3i", "Vector3", "Vector2i", "Vector2", "Color", "(", ")", "[", "]", "{", "}"]:
        cleaned = cleaned.replace(String(junk), "")
    cleaned = cleaned.replace(";", ",").replace(" ", ",").replace("\\t", ",")
    var numbers: Array = []
    for part in cleaned.split(",", false):
        var token: String = String(part).strip_edges()
        if token == "":
            continue
        numbers.append(float(token))
    while numbers.size() < count:
        numbers.append(0.0)
    return numbers


func nc_coerce(raw: Variant, current: Variant) -> Variant:
    # Dize girdilerini hedef özelliğin mevcut türüne göre dönüştürür.
    var kind: int = typeof(current)
    if typeof(raw) != TYPE_STRING:
        if kind == TYPE_INT and typeof(raw) == TYPE_FLOAT:
            return int(raw)
        if kind == TYPE_STRING or kind == TYPE_STRING_NAME:
            return str(raw)
        return raw
    var text: String = String(raw).strip_edges()
    if kind == TYPE_BOOL:
        return text.to_lower() in ["true", "1", "yes", "on", "evet", "doğru"]
    if kind == TYPE_INT:
        return int(float(text))
    if kind == TYPE_FLOAT:
        return float(text)
    if kind == TYPE_VECTOR2:
        var values2: Array = nc_numbers(text, 2)
        return Vector2(float(values2[0]), float(values2[1]))
    if kind == TYPE_VECTOR2I:
        var values2i: Array = nc_numbers(text, 2)
        return Vector2i(int(values2i[0]), int(values2i[1]))
    if kind == TYPE_VECTOR3:
        var values3: Array = nc_numbers(text, 3)
        return Vector3(float(values3[0]), float(values3[1]), float(values3[2]))
    if kind == TYPE_VECTOR3I:
        var values3i: Array = nc_numbers(text, 3)
        return Vector3i(int(values3i[0]), int(values3i[1]), int(values3i[2]))
    if kind == TYPE_VECTOR4:
        var values4: Array = nc_numbers(text, 4)
        return Vector4(float(values4[0]), float(values4[1]), float(values4[2]), float(values4[3]))
    if kind == TYPE_COLOR:
        return Color(text)
    if kind == TYPE_STRING_NAME:
        return StringName(text)
    if kind == TYPE_NODE_PATH:
        return NodePath(text)
    return text


func nc_property_info(node: Object, property_name: String) -> Dictionary:
    for item in node.get_property_list():
        if String(item.get("name", "")) == property_name:
            return item
    return {}


func nc_similar_properties(node: Object, property_name: String) -> Array:
    var needle: String = property_name.to_lower()
    var found: Array = []
    for item in node.get_property_list():
        var candidate: String = String(item.get("name", ""))
        if candidate == "" or candidate.begins_with("_") or candidate.begins_with("metadata/"):
            continue
        if candidate.to_lower().find(needle) >= 0 or needle.find(candidate.to_lower()) >= 0:
            found.append(candidate)
        if found.size() >= 8:
            break
    return found


func nc_text(value: Variant) -> String:
    if value == null:
        return "null"
    if value is Resource:
        var resource: Resource = value as Resource
        if resource != null and resource.resource_path != "":
            return resource.resource_path
    return str(value)


func nc_script_of(node: Object) -> String:
    var script: Variant = node.get_script()
    if script != null and script is Script:
        return (script as Script).resource_path
    return ""


func nc_apply(node: Object, property_name: String, raw: Variant) -> Dictionary:
    if node == null:
        return {"ok": false, "property": property_name, "error": "hedef düğüm yok"}
    var info: Dictionary = nc_property_info(node, property_name)
    if info.is_empty():
        return {"ok": false, "property": property_name,
            "error": "özellik bulunamadı: '" + property_name + "' (" + node.get_class() + ")",
            "similar": nc_similar_properties(node, property_name)}
    var value: Variant = nc_coerce(raw, node.get(property_name))
    if typeof(raw) == TYPE_STRING and int(info.get("type", 0)) == TYPE_OBJECT:
        # Kaynak (Resource) tipli özellikler: "res://..." dizesini yükle.
        var hint_string: String = String(info.get("hint_string", ""))
        var wanted: String = String(raw)
        if wanted.begins_with("res://") and hint_string != "":
            if not ResourceLoader.exists(wanted):
                return {"ok": false, "property": property_name,
                    "error": "kaynak yok: '" + wanted + "' (beklenen tür: " + hint_string + ")"}
            var loaded: Resource = load(wanted)
            if loaded == null:
                return {"ok": false, "property": property_name,
                    "error": "kaynak yüklenemedi: '" + wanted + "' (beklenen tür: " + hint_string + ")"}
            value = loaded
    node.set(property_name, value)
    var applied: Variant = node.get(property_name)
    var report: Dictionary = {"ok": true, "property": property_name,
        "value": nc_text(applied), "type": type_string(typeof(applied))}
    if applied == null and value != null:
        report["ok"] = false
        report["expected"] = nc_text(value)
        report["error"] = "özellik atanamadı (değer null kaldı): '" + property_name + "'"
    return report


func nc_find(root: Node, node_path: String) -> Node:
    var text: String = node_path.strip_edges()
    if text == "" or text == "." or text == "/":
        return root
    if text == String(root.name):
        return root
    var direct: Node = root.get_node_or_null(NodePath(text))
    if direct != null:
        return direct
    var prefix: String = String(root.name) + "/"
    if text.begins_with(prefix):
        var trimmed: Node = root.get_node_or_null(NodePath(text.substr(prefix.length())))
        if trimmed != null:
            return trimmed
    return root.get_node_or_null(NodePath("./" + text))


func nc_path_of(root: Node, node: Node) -> String:
    if node == root:
        return "."
    var parts: Array = []
    var current: Node = node
    while current != null and current != root:
        parts.push_front(String(current.name))
        current = current.get_parent()
    return ".".join(parts)


func nc_load_scene(scene_path: String) -> Node:
    if not ResourceLoader.exists(scene_path):
        return null
    var packed: Variant = ResourceLoader.load(scene_path, "PackedScene", ResourceLoader.CACHE_MODE_IGNORE)
    if packed == null or not (packed is PackedScene):
        return null
    return (packed as PackedScene).instantiate()


func nc_save(root: Node, scene_path: String) -> Dictionary:
    var packed := PackedScene.new()
    var pack_error: int = packed.pack(root)
    if pack_error != OK:
        return {"ok": false, "error": "sahne paketlenemedi: " + error_string(pack_error), "code": pack_error}
    var directory: String = ProjectSettings.globalize_path(scene_path).get_base_dir()
    if directory != "":
        DirAccess.make_dir_recursive_absolute(directory)
    var save_error: int = ResourceSaver.save(packed, scene_path)
    if save_error != OK:
        return {"ok": false, "error": "sahne kaydedilemedi: " + error_string(save_error), "code": save_error}
    if not FileAccess.file_exists(scene_path):
        return {"ok": false, "error": "kayıt sonrası dosya bulunamadı: " + scene_path}
    return {"ok": true, "path": scene_path}


func nc_count_tree(node: Node) -> int:
    # Ağaçtaki TÜM düğümler (sahiplikten bağımsız) — "niyet edilen" sayı.
    var total: int = 1
    for child in node.get_children():
        total += nc_count_tree(child)
    return total


func nc_verify_saved(scene_path: String, expected_nodes: int) -> Dictionary:
    # pack() sessizce BOŞ sahne yazabilir: `owner` yalnızca ata zincirinde
    # kabul edilir, `add_child()` ÖNCESİNDE atanırsa motor "Invalid owner"
    # uyarır ama pack()/save() yine OK döner ve diske eksik sahne yazar.
    # Bu yüzden sahneyi diskten geri okuyup düğüm sayısını karşılaştırıyoruz.
    var reloaded: Node = nc_load_scene(scene_path)
    if reloaded == null:
        return {"ok": false, "expected_nodes": expected_nodes, "actual_nodes": -1,
            "error": "kayıt sonrası sahne yeniden yüklenemedi: " + scene_path}
    var actual: int = nc_count_tree(reloaded)
    reloaded.free()
    if actual != expected_nodes:
        return {"ok": false, "expected_nodes": expected_nodes, "actual_nodes": actual,
            "error": "sahne kaydı doğrulanamadı: bellekte " + str(expected_nodes) +
                " düğüm vardı, dosyada " + str(actual) + " düğüm var — düğüm " +
                "sahipliği (owner) bozulmuş; `owner` add_child() SONRASINDA atanmalı"}
    return {"ok": true, "expected_nodes": expected_nodes, "actual_nodes": actual}


const NC_INSPECT_PROPS: Dictionary = {
    "Node": ["process_mode"],
    "Node3D": ["position", "rotation_degrees", "scale", "visible", "top_level"],
    "Node2D": ["position", "rotation_degrees", "scale", "visible", "z_index"],
    "CanvasItem": ["visible", "modulate", "self_modulate"],
    "Control": ["position", "size", "anchor_left", "anchor_top", "anchor_right", "anchor_bottom"],
    "Camera3D": ["fov", "near", "far", "current", "projection", "size", "cull_mask"],
    "Camera2D": ["zoom", "offset", "enabled", "anchor_mode"],
    "MeshInstance3D": ["mesh", "cast_shadow", "skeleton", "material_override"],
    "DirectionalLight3D": ["light_energy", "light_color", "shadow_enabled", "directional_shadow_mode"],
    "OmniLight3D": ["light_energy", "light_color", "omni_range", "shadow_enabled"],
    "SpotLight3D": ["light_energy", "light_color", "spot_range", "spot_angle", "shadow_enabled"],
    "WorldEnvironment": ["environment"],
    "CollisionShape3D": ["shape", "disabled"],
    "CollisionShape2D": ["shape", "disabled"],
    "CharacterBody3D": ["motion_mode", "velocity", "up_direction"],
    "CharacterBody2D": ["motion_mode", "velocity", "up_direction"],
    "RigidBody3D": ["mass", "gravity_scale", "freeze"],
    "RigidBody2D": ["mass", "gravity_scale", "freeze"],
    "Area3D": ["monitoring", "monitorable"],
    "Area2D": ["monitoring", "monitorable"],
    "Label": ["text", "horizontal_alignment", "vertical_alignment", "autowrap_mode"],
    "RichTextLabel": ["text", "bbcode_enabled", "fit_content"],
    "Button": ["text", "disabled", "toggle_mode"],
    "LineEdit": ["text", "placeholder_text", "editable"],
    "TextEdit": ["text", "editable"],
    "ColorRect": ["color"],
    "Sprite2D": ["texture", "flip_h", "flip_v"],
    "TextureRect": ["texture", "expand_mode", "stretch_mode"],
    "ProgressBar": ["value", "max_value", "min_value"],
    "Timer": ["wait_time", "autostart", "one_shot"],
    "AudioStreamPlayer": ["stream", "volume_db", "autoplay", "bus"],
    "AudioStreamPlayer3D": ["stream", "volume_db", "autoplay", "unit_size"],
    "AnimationPlayer": ["autoplay", "root_node"],
    "AnimationTree": ["tree_root", "active"],
    "GPUParticles3D": ["emitting", "amount", "lifetime"],
    "SubViewport": ["size", "render_target_update_mode"],
    "CanvasLayer": ["layer", "visible", "offset"],
    "TileMapLayer": ["tile_set", "enabled"],
}


func nc_inspect_props(node: Object) -> Array:
    var found: Array = []
    var klass: String = node.get_class()
    var guard: int = 0
    while klass != "" and guard < 16:
        if NC_INSPECT_PROPS.has(klass):
            for property_name in NC_INSPECT_PROPS[klass]:
                if not found.has(property_name):
                    found.append(property_name)
        klass = String(ClassDB.get_parent_class(klass))
        guard += 1
    var script: Variant = node.get_script()
    if script != null and script is Script:
        for item in (script as Script).get_script_property_list():
            if (int(item.get("usage", 0)) & PROPERTY_USAGE_SCRIPT_VARIABLE) == 0:
                continue
            var script_name: String = String(item.get("name", ""))
            if script_name != "" and not found.has(script_name):
                found.append(script_name)
    return found


func nc_walk(node: Node, root: Node, parent_path: String, out: Array) -> void:
    var node_path: String = String(node.name)
    if parent_path != "":
        node_path = parent_path + "/" + String(node.name)
    var entry: Dictionary = {
        "name": String(node.name),
        "type": node.get_class(),
        "path": node_path,
        "parent": parent_path,
        "children": node.get_child_count(),
    }
    var script_path: String = nc_script_of(node)
    if script_path != "":
        entry["script"] = script_path
    var properties: Dictionary = {}
    for property_name in nc_inspect_props(node):
        if nc_property_info(node, property_name).is_empty():
            continue
        properties[property_name] = nc_text(node.get(property_name))
    if not properties.is_empty():
        entry["properties"] = properties
    out.append(entry)
    for child in node.get_children():
        nc_walk(child, root, node_path, out)


func nc_collect(root: Node) -> Array:
    var nodes: Array = []
    nc_walk(root, root, "", nodes)
    return nodes
'''

#: Sahne düzenleme/inceleme betiğinin gövdesi. `NC_SCENE_PATH`,
#: `NC_ACTION` ve `NC_OPERATIONS_JSON` sabitleri Python tarafından eklenir.
#:
#: Kısmi hatalarda bile sahne **kaydedilir**: eklenen düğümü sessizce atmak,
#: ajanın yaptığı işi kaybettirir. Hatalar `errors` alanında açıkça döner.
_GD_SCENE_MAIN = '''

func _init() -> void:
    var result: Dictionary = {
        "ok": true,
        "action": NC_ACTION,
        "scene": NC_SCENE_PATH,
        "existed": false,
        "created_root": false,
        "saved": false,
        "applied": [],
        "errors": [],
    }
    var parsed: Variant = JSON.parse_string(NC_OPERATIONS_JSON)
    if parsed == null or not (parsed is Array):
        result["ok"] = false
        result["errors"].append("işlem listesi JSON olarak çözümlenemedi")
        nc_emit(result)
        quit(1)
        return

    var root: Node = nc_load_scene(NC_SCENE_PATH)
    result["existed"] = root != null
    var dirty: bool = false

    for item in (parsed as Array):
        if not (item is Dictionary):
            continue
        var spec: Dictionary = item as Dictionary
        var op: String = String(spec.get("op", ""))

        if op == "create_root":
            if root != null:
                continue
            var wanted_type: String = String(spec.get("type", "Node3D"))
            var created: Object = nc_instantiate(wanted_type)
            if created == null or not (created is Node):
                result["ok"] = false
                result["errors"].append(nc_instantiate_error(wanted_type))
                continue
            root = created as Node
            var wanted_name: String = String(spec.get("name", ""))
            if wanted_name != "":
                root.name = wanted_name
            dirty = true
            result["created_root"] = true

        elif op == "add":
            if root == null:
                result["ok"] = false
                result["errors"].append("sahne yok: " + NC_SCENE_PATH + " — önce godot.create_project/godot.create_scene")
                continue
            var parent_path: String = String(spec.get("parent", "."))
            var parent_node: Node = nc_find(root, parent_path)
            if parent_node == null:
                result["ok"] = false
                result["errors"].append("üst düğüm bulunamadı: '" + parent_path + "'")
                continue
            var node_type: String = String(spec.get("type", "Node"))
            var fresh: Object = nc_instantiate(node_type)
            if fresh == null or not (fresh is Node):
                result["ok"] = false
                result["errors"].append(nc_instantiate_error(node_type))
                continue
            var node: Node = fresh as Node
            var node_name: String = String(spec.get("name", ""))
            if node_name != "":
                node.name = node_name
            parent_node.add_child(node)
            node.owner = root
            if parent_node != root:
                parent_node.owner = root
            var property_reports: Array = []
            var table: Dictionary = spec.get("properties", {})
            for property_name in table.keys():
                var report: Dictionary = nc_apply(node, String(property_name), table[property_name])
                property_reports.append(report)
                if not report.get("ok", false):
                    result["ok"] = false
                    result["errors"].append(String(report.get("error", "özellik atanamadı")))
            result["applied"].append({
                "op": "add",
                "name": String(node.name),
                "type": node.get_class(),
                "path": nc_path_of(root, node),
                "properties": property_reports,
            })
            dirty = true

        elif op == "set":
            if root == null:
                result["ok"] = false
                result["errors"].append("sahne yok: " + NC_SCENE_PATH)
                continue
            var target_path: String = String(spec.get("path", "."))
            var target: Node = nc_find(root, target_path)
            if target == null:
                result["ok"] = false
                result["errors"].append("düğüm bulunamadı: '" + target_path + "'")
                continue
            var changed: Dictionary = nc_apply(target, String(spec.get("property", "")), spec.get("value"))
            changed["node"] = String(target.name)
            changed["path"] = nc_path_of(root, target)
            result["applied"].append({"op": "set", "result": changed})
            if changed.get("ok", false):
                dirty = true
            else:
                result["ok"] = false
                result["errors"].append(String(changed.get("error", "özellik atanamadı")))

        elif op == "remove":
            if root == null:
                result["ok"] = false
                result["errors"].append("sahne yok: " + NC_SCENE_PATH)
                continue
            var victim_path: String = String(spec.get("path", ""))
            var victim: Node = nc_find(root, victim_path)
            if victim == null:
                result["ok"] = false
                result["errors"].append("düğüm bulunamadı: '" + victim_path + "'")
                continue
            if victim == root:
                result["ok"] = false
                result["errors"].append("kök düğüm silinemez: '" + victim_path + "'")
                continue
            var removed_name: String = String(victim.name)
            var removed_path: String = nc_path_of(root, victim)
            var removed_parent: Node = victim.get_parent()
            if removed_parent != null:
                removed_parent.remove_child(victim)
            victim.free()
            result["applied"].append({"op": "remove", "name": removed_name, "path": removed_path})
            dirty = true

        elif op == "inspect":
            if root == null:
                result["ok"] = false
                result["errors"].append("sahne yok: " + NC_SCENE_PATH)
                continue
            var nodes: Array = nc_collect(root)
            result["nodes"] = nodes
            result["node_count"] = nodes.size()
            result["root"] = {"name": String(root.name), "type": root.get_class(),
                "script": nc_script_of(root), "path": "."}

        else:
            result["ok"] = false
            result["errors"].append("bilinmeyen işlem: '" + op + "'")

    if root != null and dirty:
        # DİKKAT: beklenen sayı, kaydetmeden ÖNCE bellek ağacından hesaplanır.
        var expected_nodes: int = nc_count_tree(root)
        var saved: Dictionary = nc_save(root, NC_SCENE_PATH)
        if saved.get("ok", false):
            result["saved"] = true
            var verified: Dictionary = nc_verify_saved(NC_SCENE_PATH, expected_nodes)
            result["verified"] = verified
            if not verified.get("ok", false):
                result["ok"] = false
                result["errors"].append(String(verified.get("error", "sahne doğrulanamadı")))
        else:
            result["ok"] = false
            result["errors"].append(String(saved.get("error", "sahne kaydedilemedi")))

    if root != null:
        root.free()
    nc_emit(result)
    quit(0 if result.get("ok", false) else 1)
'''


# ─── Yardımcılar ────────────────────────────────────────────────────────────

_SAFE_SHELL_ARG = re.compile(r"^[A-Za-z0-9_\-./:=+@\\]+$")


def _ps_quote(value: Any) -> str:
    """
    Bir argümanı PowerShell için güvenle tırnaklar.

    Boşluk içeren yollar (`E:\\AppsTools\\Blender 5.2`) bu makinede gerçekten
    vardır; tırnaklanmazsa komut iki argümana bölünür ve Godot yanlış proje
    açar. Tek tırnak içindeki tek tırnak `''` ile kaçırılır.
    """
    text = str(value)
    if text and _SAFE_SHELL_ARG.match(text):
        return text
    return "'" + text.replace("'", "''") + "'"


def _gd_string(value: Any) -> str:
    """
    Python değerini geçerli bir GDScript dize değişmezine çevirir.

    `json.dumps` çıktısı GDScript'in kabul ettiği kaçış dizilerini (`\\"`,
    `\\\\`, `\\n`, `\\uXXXX`) üretir; `ensure_ascii=False` sayesinde Türkçe
    karakterler olduğu gibi kalır.
    """
    return json.dumps(str(value), ensure_ascii=False)


def _is_godot_noise(line: str) -> bool:
    """Satır, motorun normal kapanış gürültüsü mü?"""
    stripped = str(line).strip()
    if not stripped:
        return True
    return any(pattern.search(stripped) for pattern in _COMPILED_NOISE)


def _locate_error(lines: Sequence[str], index: int, window: int = 4) -> Tuple[str, int]:
    """
    Hata satırının yakınından `dosya:satır` konumunu çıkarır.

    Godot konumu bir sonraki satırda `at: GDScript::reload (res://x.gd:12)`
    biçiminde ya da geri izlemede `[0] _init (res://x.gd:12)` biçiminde verir.
    """
    for offset in range(0, window):
        position = index + offset
        if position >= len(lines):
            break
        candidate = lines[position]
        match = _LOCATION_RE.search(candidate) or _BARE_LOCATION_RE.search(candidate)
        if match:
            return match.group(1), int(match.group(2))
    return "", 0


def _summarize_stdout(text: str, limit: int = 20000) -> str:
    """Uzun çıktıyı sondan kısaltır (ajan istemini şişirmemek için)."""
    value = str(text or "")
    if len(value) <= limit:
        return value
    return "…(kısaltıldı)…\n" + value[-limit:]


def _parse_ini(text: str) -> Dict[str, Dict[str, str]]:
    """
    Godot'un `project.godot` / `export_presets.cfg` biçimini ayrıştırır.

    `configparser` bu dosyalarda tökezler: anahtarlar `/` içerir (sorun değil)
    ama değerler `PackedStringArray("4.8")` gibi Python olmayan ifadelerdir ve
    yinelenen anahtarlar olabilir. Bu yüzden satır bazlı, hoşgörülü bir
    ayrıştırma yapılır ve değerler **ham metin** olarak bırakılır.
    """
    sections: Dict[str, Dict[str, str]] = {"": {}}
    current = ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, {})
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            sections.setdefault(current, {})[key.strip()] = value.strip()
    return sections


def _ini_list(raw: str) -> List[str]:
    """`PackedStringArray("a", "b")` veya `["a", "b"]` değerini listeye çevirir."""
    text = str(raw or "").strip()
    if not text:
        return []
    match = re.search(r"\((.*)\)", text, re.DOTALL)
    if match:
        text = match.group(1)
    elif text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [item for item in re.findall(r'"([^"]*)"', text) if item]


def _safe_file_stem(value: str, fallback: str = "nc_tool") -> str:
    """Dosya adına dönüştürülebilecek güvenli bir kök ad üretir."""
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value or "")).strip("_")
    return (cleaned or fallback)[:64]


def _scan_for_godot(max_depth: int = 3, limit: int = 12) -> List[Path]:
    """
    Bilinen kök dizinlerde Godot çalıştırılabilirini **sınırlı** biçimde arar.

    Godot PATH'e eklenmez ve kurulum dizinleri derin olabilir; tüm diski
    taramak dakikalar sürer. Bu yüzden yalnızca adı `Godot*` ile başlayan
    dizinlere inilir ve sonuç sayısı sınırlanır.
    """
    found: List[Path] = []
    for root in GODOT_SCAN_ROOTS:
        base = Path(root)
        if not base.is_dir():
            continue
        base_depth = len(base.parts)
        try:
            for current, directories, filenames in os.walk(base):
                depth = len(Path(current).parts) - base_depth
                if depth >= max_depth:
                    directories[:] = []
                else:
                    directories[:] = [
                        name for name in sorted(directories)
                        if name.lower().startswith("godot")
                    ]
                for name in sorted(filenames):
                    lowered = name.lower()
                    if lowered.startswith("godot") and lowered.endswith(".exe"):
                        found.append(Path(current) / name)
                if len(found) >= limit:
                    return found
        except OSError:
            continue
    return found


def _family_for_platform(platform: str) -> str:
    """Preset platform adından şablon ailesini (windows/linux/... ) çıkarır."""
    lowered = str(platform or "").lower()
    for needle, family in PLATFORM_TEMPLATE_FAMILY:
        if needle in lowered:
            return family
    return ""


# ─── Godot Denetleyicisi ────────────────────────────────────────────────────

class GodotController:
    """
    Godot oyun motoru denetleyicisi.

    Kullanım:
        godot = GodotController()
        godot.initialize(workspace="E:/PROJELER/Oyunum", config=config,
                         event_bus=bus, shell=shell)

        registry.register_many(godot.tools())
        # veya doğrudan:
        print(godot.health())
        print(godot.run_script("extends SceneTree\\nfunc _init():\\n\\tprint(1)\\n\\tquit(0)\\n"))

    Kabuk katmanı **enjekte edilir**: `initialize(shell=...)` verilmezse aynı
    çalışma alanıyla özel bir `ShellIntegration` oluşturulur. Böylece komut
    çalıştırma, güvenlik denetimi ve zaman aşımı mantığı tek yerde kalır;
    bu modül kendi `subprocess` tesisatını yazmaz.

    Her genel metot istisna fırlatmak yerine yapılandırılmış sözlük döner:
    `{"ok": False, "error": "...", "hint": "..."}`.
    """

    name = "agent.godot"

    #: Godot çıktısında hata sayılan satırlar için üst sınır
    MAX_ERRORS = 25

    def __init__(self) -> None:
        self._workspace: Optional[Path] = None
        self._config: Any = None
        self._event_bus: Any = None
        self._shell: Any = None
        self._own_shell = False
        self._initialized = False
        self._godot_cache: Optional[Tuple[str, str, str]] = None
        self._version_cache: Optional[str] = None
        self._script_unsupported = False
        self._counter = {
            "scripts_run": 0, "projects_created": 0, "imports": 0,
            "scenes_written": 0, "scenes_inspected": 0, "runs": 0,
            "exports": 0, "errors": 0, "timeouts": 0, "commands": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, workspace: Optional[str] = None, config: Any = None,
                   event_bus: Any = None, shell: Any = None) -> None:
        """
        Denetleyiciyi başlatır.

        Args:
            workspace: Çalışma alanı kökü (None = `config.base_path` veya cwd)
            config: SystemConfig
            event_bus: EventBus (`godot.*` olayları buraya yayınlanır)
            shell: `ShellIntegration` benzeri nesne; verilmezse özel bir
                örnek oluşturulur (aynı çalışma alanı ile)

        Raises:
            GodotError: Çalışma alanı yoksa
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus

        root = workspace
        if not root:
            root = getattr(config, "base_path", None) or os.getcwd()
        self._workspace = Path(root).expanduser().resolve()
        if not self._workspace.exists():
            raise GodotError(f"Çalışma alanı yok: {self._workspace}")

        if shell is not None:
            self._shell = shell
            self._own_shell = False
        else:
            own = ShellIntegration()
            own.initialize(workspace=str(self._workspace), config=config,
                           event_bus=event_bus)
            self._shell = own
            self._own_shell = True

        self._initialized = True
        console, _gui, source = self._find_godot()
        logger.info(
            f"Godot denetleyicisi hazır — çalışma alanı: {self._workspace}, "
            f"motor: {console or '(bulunamadı)'} [{source}]"
        )

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır (kendi oluşturduğu kabuğu kapatır)."""
        if self._own_shell and self._shell is not None:
            try:
                self._shell.shutdown()
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Kabuk kapatılamadı: {exc}")
        self._shell = None
        self._own_shell = False
        self._godot_cache = None
        self._version_cache = None
        self._initialized = False
        logger.info(
            f"Godot denetleyicisi kapatıldı ({self._counter['scripts_run']} betik, "
            f"{self._counter['scenes_written']} sahne yazımı, "
            f"{self._counter['errors']} hata)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """İstatistikler."""
        console, gui, source = self._find_godot()
        return {
            **self._counter,
            "initialized": self._initialized,
            "workspace": str(self._workspace or ""),
            "executable": console or gui,
            "executable_source": source,
            "own_shell": self._own_shell,
        }

    @property
    def workspace(self) -> Path:
        """Çalışma alanı kökü."""
        if self._workspace is None:
            raise GodotError("Godot denetleyicisi başlatılmadı — initialize() çağırın")
        return self._workspace

    # ─── Motor Keşfi ────────────────────────────────────────────────────

    def _find_godot(self) -> Tuple[str, str, str]:
        """
        Godot çalıştırılabilirlerini bulur.

        Returns:
            `(konsol_exe, gui_exe, kaynak)`. Konsol sürümü tercih edilir çünkü
            Windows'ta **yalnızca** `_console.exe` stdout'u ebeveyne iletir;
            GUI sürümü çıktıyı yutar ve ajan körleşir.
        """
        if self._godot_cache is not None:
            return self._godot_cache

        candidates: List[Tuple[Path, str]] = []
        for variable in GODOT_ENV_VARS:
            value = os.environ.get(variable, "").strip().strip('"')
            if value and Path(value).is_file():
                candidates.append((Path(value), f"ortam:{variable}"))
        for name in ("godot", "godot4", "godot.exe", "godot4.exe"):
            located = shutil.which(name)
            if located:
                candidates.append((Path(located), "PATH"))
        for known in GODOT_KNOWN_EXES:
            path = Path(known)
            if path.is_file():
                candidates.append((path, "bilinen yol"))
        if not candidates:
            for path in _scan_for_godot():
                candidates.append((path, "tarama"))

        console, gui, source = "", "", "yok"
        for path, origin in candidates:
            try:
                resolved = path.resolve()
            except OSError:  # pragma: no cover
                continue
            text = str(resolved)
            is_console = "_console" in resolved.name.lower()
            if is_console and not console:
                console, source = text, origin
            elif not is_console and not gui:
                gui = text
                if not source or source == "yok":
                    source = origin
            # GUI ikizi varsa onu da bul (konsol yoksa yedek olur)
            if not gui:
                sibling = resolved.with_name(
                    resolved.name.replace("_console.exe", ".exe")
                )
                if sibling.is_file() and sibling != resolved:
                    gui = str(sibling)
        if not console and gui:
            source = source or "bilinen yol"
        self._godot_cache = (console, gui, source if (console or gui) else "yok")
        return self._godot_cache

    def executable(self) -> str:
        """
        Kullanılacak Godot çalıştırılabilirinin tam yolu.

        Konsol sürümü (`*_console.exe`) varsa o döner; yoksa GUI sürümü.
        Hiçbiri yoksa boş dize döner (çağıran taraf yapılandırılmış hata üretir).
        """
        console, gui, _source = self._find_godot()
        return console or gui

    def version(self) -> str:
        """
        Motorun sürüm dizesi (ör. `4.8.dev7.official.c971f93e7`).

        Sonuç önbelleklenir; motor yoksa boş dize döner.
        """
        if self._version_cache is not None:
            return self._version_cache
        exe = self.executable()
        if not exe:
            return ""
        outcome = self._run_godot(["--version"], timeout=30.0, cwd=self._command_cwd())
        text = f"{outcome.get('stdout', '')}\n{outcome.get('stderr', '')}"
        match = re.search(r"^\s*(\d[\w.\-]*)", text, re.MULTILINE)
        self._version_cache = match.group(1) if match else ""
        return self._version_cache

    def _command_cwd(self) -> Path:
        """Komutların çalıştırılacağı dizin (proje yoksa çalışma alanı)."""
        root = self._workspace or Path(os.getcwd())
        return root if root.is_dir() else Path(os.getcwd())

    def _missing_executable(self) -> Dict[str, Any]:
        """Motor bulunamadığında dönen açıklayıcı hata."""
        self._counter["errors"] += 1
        return {
            "ok": False,
            "error": "Godot çalıştırılabilir dosyası bulunamadı",
            "hint": (
                "Godot kurun ve `NEUROCORE_GODOT` ortam değişkenini konsol "
                "sürümüne işaret ettirin (ör. "
                "E:\\AppsTools\\Godot_v4.8-dev7_win64\\Godot_v4.8-dev7_win64_console.exe). "
                "Windows'ta stdout için `_console.exe` gerekir."
            ),
            "searched_env": list(GODOT_ENV_VARS),
            "searched_paths": list(GODOT_KNOWN_EXES),
        }

    # ─── Komut Çalıştırma ───────────────────────────────────────────────

    def _build_command(self, exe: str, arguments: Sequence[Any]) -> str:
        """
        Kabuk için komut metnini üretir.

        Windows/PowerShell'de çağrı operatörü (`&`) kullanılır; boşluk içeren
        yollar tırnaklanır. `cd` **asla** kullanılmaz: proje kökü her zaman
        Godot'un kendi `--path` argümanıyla bildirilir.
        """
        parts = [_ps_quote(str(argument)) for argument in arguments]
        if os.name == "nt":
            return "& " + " ".join([_ps_quote(exe)] + parts)
        return " ".join([_ps_quote(exe)] + parts)

    def _as_dict(self, outcome: Any) -> Dict[str, Any]:
        """`CommandResult` (veya benzeri) nesnesini sözlüğe çevirir."""
        if outcome is None:
            return {"ok": False, "returncode": -1, "stdout": "", "stderr": "boş sonuç",
                    "timed_out": False, "blocked": False, "duration_ms": 0.0}
        if isinstance(outcome, dict):
            return dict(outcome)
        converter = getattr(outcome, "to_dict", None)
        if callable(converter):
            try:
                return dict(converter())
            except Exception:  # pragma: no cover
                pass
        return {
            "ok": bool(getattr(outcome, "ok", False)),
            "returncode": int(getattr(outcome, "returncode", -1)),
            "stdout": str(getattr(outcome, "stdout", "")),
            "stderr": str(getattr(outcome, "stderr", "")),
            "timed_out": bool(getattr(outcome, "timed_out", False)),
            "blocked": bool(getattr(outcome, "blocked", False)),
            "duration_ms": float(getattr(outcome, "duration_ms", 0.0) or 0.0),
        }

    def _run_godot(self, arguments: Sequence[Any], timeout: float = 120.0,
                   cwd: Optional[Path] = None) -> Dict[str, Any]:
        """
        Godot'yu çalıştırır ve ham sonucu döndürür.

        Args:
            arguments: Godot argümanları (`--path` dahil, çağıran ekler)
            timeout: Saniye — takılan bir motor ajanı kilitlememeli
            cwd: Çalışma dizini (None = proje/çalışma alanı)

        Returns:
            `{"ok", "returncode", "stdout", "stderr", "duration_ms",
             "timed_out", "blocked", "command"}`
        """
        exe = self.executable()
        if not exe:
            return self._missing_executable()
        if self._shell is None:
            return {"ok": False, "error": "kabuk katmanı yok — initialize() çağırın"}

        workdir = cwd or self._command_cwd()
        command = self._build_command(exe, arguments)
        started = time.perf_counter()
        self._counter["commands"] += 1
        try:
            outcome = self._shell.run(command, cwd=str(workdir),
                                      timeout=max(1.0, float(timeout)))
        except Exception as exc:
            self._counter["errors"] += 1
            return {"ok": False, "error": f"Godot çalıştırılamadı: {exc}",
                    "command": command, "returncode": -1, "stdout": "", "stderr": str(exc),
                    "timed_out": False, "blocked": False,
                    "duration_ms": (time.perf_counter() - started) * 1000}

        data = self._as_dict(outcome)
        data["command"] = command
        data.setdefault("duration_ms", (time.perf_counter() - started) * 1000)
        data["timeout_s"] = float(timeout)
        if data.get("timed_out"):
            self._counter["timeouts"] += 1
            data["ok"] = False
            data["error"] = f"Godot zaman aşımına uğradı ({timeout:.0f}s)"
            data["hint"] = ("işi küçültün veya timeout değerini artırın; "
                            "sahne açılışta sonsuz döngüye giriyor olabilir")
        elif data.get("blocked"):
            data["ok"] = False
            data["error"] = data.get("blocked_reason") or "komut güvenlik denetiminde engellendi"
        else:
            data["ok"] = bool(data.get("ok"))
        return data

    @staticmethod
    def _collect_errors(text: str) -> List[Dict[str, Any]]:
        """
        Çıktıdan yapılandırılmış hata listesi çıkarır.

        Motorun kapanış gürültüsü (`were leaked at exit` vb.) elenir; aksi
        hâlde **her** başarılı çalıştırma hatalı görünür.
        """
        lines = str(text or "").splitlines()
        entries: List[Dict[str, Any]] = []
        seen: set = set()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if _is_godot_noise(stripped):
                continue
            for pattern, kind, hint in _COMPILED_HINTS:
                if not pattern.search(stripped):
                    continue
                message = stripped[:300]
                key = (kind, message)
                if key in seen:
                    break
                seen.add(key)
                entry: Dict[str, Any] = {"kind": kind, "message": message, "hint": hint}
                file_path, line_number = _locate_error(lines, index)
                if file_path:
                    entry["file"] = file_path
                if line_number:
                    entry["line"] = line_number
                entries.append(entry)
                break
        return entries[:GodotController.MAX_ERRORS]

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    # ─── Yol Çözümleme ──────────────────────────────────────────────────

    def _resolve_dir(self, path: str = "") -> Path:
        """
        Dizini çalışma alanına göre çözer (kum havuzu denetimiyle).

        `coding.resolve` ile aynı politikadır: yalnızca çalışma alanı ve sistem
        geçici dizini kabul edilir. Yol henüz var olmayabilir (create_project).
        """
        root = self.workspace
        text = os.path.expandvars(os.path.expanduser(str(path or "")))
        candidate = Path(text) if text else root
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise GodotError(f"Yol çözümlenemedi ({path}): {exc}") from exc

        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            pass
        temp_root = os.environ.get("TEMP") or os.environ.get("TMP")
        if temp_root:
            try:
                resolved.relative_to(Path(temp_root).resolve())
                return resolved
            except (ValueError, OSError):
                pass
        raise GodotError(
            f"güvenlik: çalışma alanı dışına erişim reddedildi — {resolved} (kök: {root})"
        )

    def _project_root(self, path: str = "") -> Dict[str, Any]:
        """
        Proje kökünü doğrular.

        Returns:
            `{"ok": True, "root": Path}` ya da `{"ok": False, "error", "hint"}`
        """
        try:
            root = self._resolve_dir(path)
        except GodotError as exc:
            return {"ok": False, "error": str(exc),
                    "hint": "yolu çalışma alanı içinde verin"}
        if not root.is_dir():
            return {"ok": False, "error": f"dizin yok: {root}",
                    "hint": "önce godot.create_project ile proje oluşturun"}
        if not (root / "project.godot").is_file():
            return {"ok": False, "error": f"Godot projesi değil (project.godot yok): {root}",
                    "hint": "godot.create_project ile oluşturun veya doğru dizini verin",
                    "path": str(root)}
        return {"ok": True, "root": root}

    def _to_res_path(self, root: Path, value: str) -> str:
        """
        Bir yolu `res://` biçimine çevirir.

        GDScript'e yazılan **tüm** proje yolları ileri eğik çizgili ve
        `res://` göreli olmalıdır; Windows ters bölü karakteri Godot'ta
        kaynak yolu sayılmaz.
        """
        text = str(value or "").strip().strip('"')
        if not text:
            return ""
        if text.startswith("res://"):
            return text.replace("\\", "/")
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            relative = candidate.resolve().relative_to(root.resolve())
        except (ValueError, OSError):
            return ""
        return "res://" + relative.as_posix()

    def is_godot_project(self, path: str = "") -> bool:
        """
        Verilen yol bir Godot proje kökü mü?

        Args:
            path: Dizin (boş = çalışma alanı)

        Returns:
            `project.godot` varsa True
        """
        try:
            root = self._resolve_dir(path)
        except GodotError:
            return False
        return (root / "project.godot").is_file()

    # ─── Proje Yönetimi ─────────────────────────────────────────────────

    def create_project(self, path: str, name: str = "", template: str = "3d") -> Dict[str, Any]:
        """
        Yeni bir Godot 4 projesi oluşturur ve içe aktarır.

        Üretilen dosyalar minimal ama **gerçekten çalıştırılabilirdir**:
        `project.godot` (`config_version=5`, `config/features=PackedStringArray("4.8")`),
        bir ana sahne (`Main.tscn`) ve ona bağlı bir `main.gd`.

        Args:
            path: Proje dizini (yoksa oluşturulur)
            name: Proje adı (boş = dizin adı)
            template: `"3d"` (Node3D + Camera3D + DirectionalLight3D + BoxMesh),
                `"2d"` (Node2D + Camera2D + ColorRect) veya `"empty"`

        Returns:
            `{"ok", "path", "name", "template", "files", "import", "error"}`
        """
        chosen = str(template or "3d").strip().lower()
        if chosen not in ("3d", "2d", "empty"):
            return {"ok": False, "error": f"bilinmeyen şablon: {template}",
                    "hint": "geçerli şablonlar: 3d, 2d, empty"}
        try:
            root = self._resolve_dir(path)
        except GodotError as exc:
            return {"ok": False, "error": str(exc)}

        if (root / "project.godot").is_file():
            return {"ok": False, "error": f"bu dizinde zaten bir Godot projesi var: {root}",
                    "path": str(root), "existing": True,
                    "hint": "mevcut projeyi düzenlemek için godot.inspect_scene / "
                            "godot.project_map kullanın"}

        project_name = (str(name).strip() or root.name or "NeuroCoreProject")
        project_name = project_name.replace('"', "'").replace("\n", " ")[:120]
        version = self.version()
        files: List[str] = []
        try:
            root.mkdir(parents=True, exist_ok=True)
            main_scene_relative = "res://Main.tscn" if chosen != "empty" else ""
            (root / "project.godot").write_text(
                self._project_godot_text(project_name, main_scene_relative, chosen),
                encoding="utf-8", newline="\n",
            )
            files.append("project.godot")
            if chosen != "empty":
                (root / "main.gd").write_text(
                    self._main_script_text(chosen), encoding="utf-8", newline="\n")
                (root / "Main.tscn").write_text(
                    self._main_scene_text(chosen), encoding="utf-8", newline="\n")
                files.extend(["main.gd", "Main.tscn"])
        except OSError as exc:
            self._counter["errors"] += 1
            return {"ok": False, "error": f"proje dosyaları yazılamadı: {exc}",
                    "path": str(root)}

        self._counter["projects_created"] += 1
        imported = self.import_project(str(root), timeout=300.0)
        self._emit("godot.project_created", {
            "path": str(root), "name": project_name, "template": chosen,
            "engine_version": version, "import_ok": imported.get("ok", False),
        })
        logger.info(f"Godot projesi oluşturuldu: {root} ({chosen}, {project_name})")
        return {
            "ok": bool(imported.get("ok", False)),
            "path": str(root),
            "name": project_name,
            "template": chosen,
            "engine_version": version,
            "files": files,
            "main_scene": main_scene_relative,
            "import": imported,
            "error": "" if imported.get("ok") else str(imported.get("error", "")),
            "hint": "" if imported.get("ok") else str(imported.get("hint", "")),
        }

    def _project_godot_text(self, name: str, main_scene: str, template: str) -> str:
        """`project.godot` içeriğini üretir (Godot 4.8 biçimi)."""
        renderer = "gl_compatibility" if template == "2d" else "forward_plus"
        lines = [
            "; Engine configuration file.",
            "; NeuroCore tarafından üretildi — Godot 4.8 (config_version=5).",
            "",
            f"config_version={GODOT_CONFIG_VERSION}",
            "",
            "[application]",
            "",
            f'config/name="{name}"',
        ]
        if main_scene:
            lines.append(f'run/main_scene="{main_scene}"')
        lines += [
            'config/features=PackedStringArray("4.8")',
            "",
            "[display]",
            "",
            "window/size/viewport_width=1152",
            "window/size/viewport_height=648",
            "",
            "[rendering]",
            "",
            f'renderer/rendering_method="{renderer}"',
            "",
        ]
        return "\n".join(lines)

    def _main_script_text(self, template: str) -> str:
        """Ana sahneye bağlanan başlangıç betiğini üretir."""
        base = "Node2D" if template == "2d" else "Node3D"
        return (
            f"extends {base}\n"
            "\n"
            "## NeuroCore tarafından oluşturulan başlangıç betiği.\n"
            "## Oyun mantığını buraya yazın; sahneyi godot.* araçlarıyla düzenleyin.\n"
            "\n"
            "\n"
            "func _ready() -> void:\n"
            '\tprint("NeuroCore: \'", name, "\' sahnesi hazır.")\n'
        )

    def _main_scene_text(self, template: str) -> str:
        """
        `Main.tscn` içeriğini üretir.

        Sahne metni elle yazılır çünkü motor kurulu olmasa bile proje
        oluşturulabilmelidir; biçim Godot 4 (`.tscn format=3`) ile uyumludur
        ve bu makinede yüklenerek doğrulanmıştır.
        """
        if template == "2d":
            return (
                "[gd_scene format=3]\n"
                "\n"
                '[ext_resource type="Script" path="res://main.gd" id="1_main"]\n'
                "\n"
                '[node name="Main" type="Node2D"]\n'
                'script = ExtResource("1_main")\n'
                "\n"
                '[node name="Camera2D" type="Camera2D" parent="."]\n'
                "\n"
                '[node name="ArkaPlan" type="ColorRect" parent="."]\n'
                "offset_right = 1152.0\n"
                "offset_bottom = 648.0\n"
                "color = Color(0.101961, 0.121569, 0.180392, 1)\n"
            )
        return (
            "[gd_scene format=3]\n"
            "\n"
            '[ext_resource type="Script" path="res://main.gd" id="1_main"]\n'
            "\n"
            '[sub_resource type="BoxMesh" id="BoxMesh_main"]\n'
            "\n"
            '[node name="Main" type="Node3D"]\n'
            'script = ExtResource("1_main")\n'
            "\n"
            '[node name="Kamera" type="Camera3D" parent="."]\n'
            "transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 6)\n"
            "\n"
            '[node name="Isik" type="DirectionalLight3D" parent="."]\n'
            "transform = Transform3D(1, 0, 0, 0, 0.707107, -0.707107, 0, 0.707107, 0.707107, 0, 5, 0)\n"
            "shadow_enabled = true\n"
            "\n"
            '[node name="Kutu" type="MeshInstance3D" parent="."]\n'
            'mesh = SubResource("BoxMesh_main")\n'
        )

    def import_project(self, path: str = "", timeout: float = 300.0,
                       method: str = "auto") -> Dict[str, Any]:
        """
        Projeyi içe aktarır (kaynak taraması ve `.godot` önbelleğinin kurulması).

        Neden gerekli? Godot, ilk çalıştırmada kaynakları tarar ve `.godot/`
        altında içe aktarılmış biçimleri üretir. Bu yapılmadan çalıştırılan bir
        sahne "kaynak bulunamadı" hataları verebilir.

        Godot sürümleri arasında fark var:
            * `--headless --path <proje> --import` — **birincil** biçim.
              `--editor --quit` anlamına gelir ve taramayı sonuna kadar
              sürdürür; başarıda exit 0 verir. Yalnızca düzenleyici yapılarda
              çalışır (bu makinedeki dev derlemesi dâhil).
            * `--headless --editor --quit-after 2` — düzenleyiciyi açıp 2 kare
              sonra kapatır. Bazı yapılarda `--import` yok sayılırsa bu biçim
              gerekir; ancak tarama iş parçacığı "Scan thread aborted" uyarısıyla
              kesilebilir (uyarı, hata değildir) ve büyük projelerde tarama
              tamamlanmayabilir. Bu yüzden **yedek** biçimdir.

        Args:
            path: Proje dizini (boş = çalışma alanı)
            timeout: Saniye
            method: `"auto"` (import → editor), `"import"`, `"editor"`

        Returns:
            `{"ok", "form", "exit_code", "stdout", "stderr", "duration_ms",
             "tried", "error", "hint"}`
        """
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        if not self.executable():
            return self._missing_executable()

        editor_form = ("editor", ["--headless", "--editor", "--quit-after", "2",
                                  "--path", str(root)])
        import_form = ("import", ["--headless", "--import", "--path", str(root)])
        if method == "import":
            forms: List[Tuple[str, List[Any]]] = [import_form]
        elif method == "editor":
            forms = [editor_form]
        else:
            forms = [import_form, editor_form]

        tried: List[Dict[str, Any]] = []
        last: Dict[str, Any] = {}
        for form, arguments in forms:
            outcome = self._run_godot(arguments, timeout=timeout, cwd=root)
            errors = self._collect_errors(
                f"{outcome.get('stdout', '')}\n{outcome.get('stderr', '')}")
            # "Scan thread aborted" gürültüsü zaten elenir; geriye kalan
            # hatalar gerçek içe aktarma sorunlarıdır.
            ok = bool(outcome.get("ok")) and not errors
            tried.append({"form": form, "ok": ok,
                          "exit_code": outcome.get("returncode"),
                          "error_count": len(errors)})
            last = {**outcome, "form": form, "errors": errors}
            if ok:
                self._counter["imports"] += 1
                self._emit("godot.project_imported", {
                    "path": str(root), "form": form,
                    "duration_ms": outcome.get("duration_ms"),
                })
                logger.info(f"Godot projesi içe aktarıldı: {root} (biçim: {form})")
                return {
                    "ok": True, "path": str(root), "form": form,
                    "exit_code": outcome.get("returncode"),
                    "stdout": _summarize_stdout(outcome.get("stdout", "")),
                    "stderr": _summarize_stdout(outcome.get("stderr", ""), 8000),
                    "duration_ms": outcome.get("duration_ms", 0.0),
                    "errors": errors,
                    "tried": tried,
                    "godot_dir": str(root / ".godot"),
                    "imported": (root / ".godot").is_dir(),
                }

        self._counter["errors"] += 1
        return {
            "ok": False,
            "path": str(root),
            "form": last.get("form", ""),
            "error": (last.get("error") or "içe aktarma başarısız"),
            "hint": ("proje dosyası bozuk olabilir veya motor sürümü uyumsuz; "
                     "ham çıktıyı inceleyin"),
            "exit_code": last.get("returncode"),
            "stdout": _summarize_stdout(last.get("stdout", "")),
            "stderr": _summarize_stdout(last.get("stderr", ""), 8000),
            "errors": last.get("errors", []),
            "tried": tried,
        }

    # ─── Betik Çalıştırma ───────────────────────────────────────────────

    def run_script(self, code: str, path: str = "", timeout: float = 120.0,
                   script_name: str = "nc_tool", quit_after: int = 500) -> Dict[str, Any]:
        """
        Proje içinde bir GDScript betiği çalıştırır (bu modülün iş atı).

        Betik `<proje>/nc_tools/<script_name>.gd` dosyasına yazılır ve
        `--headless --path <proje> --script res://nc_tools/<ad>.gd` ile
        çalıştırılır. Çıktı ayrıştırılır: GDScript hataları (`SCRIPT ERROR`,
        `Parse Error`) yakalanır ve `ok=False` yapılır — motorun çıkış kodu 0
        olsa bile.

        Neden `--quit-after`? Bir GDScript **çalışma zamanı** hatası
        (`SCRIPT ERROR`) `quit()` satırına hiç ulaşmaz; Godot exit 0 verir ama
        süreç **sonsuza kadar asılı kalır**. `--quit-after 500` bu durumda
        motoru ~3 saniye sonra kapatır ve çıktıyı yine de teslim eder. Bu bir
        güvenlik ağıdır: `_init` içinde `await` kullanan uzun ömürlü betikler
        için `quit_after` artırılmalıdır (`timeout` duvar saati üst sınırıdır
        ve aşılırsa süreç sert şekilde öldürülür).

        `ok` kararı **üç** sinyalin birleşimidir: `exit_code == 0` **ve**
        çıktıda hata işareti olmaması **ve** (varsa) `NC_RESULT.ok`.
        `ok = (exit_code == 0)` **asla** kullanılmaz.

        Args:
            code: GDScript kaynağı (`extends SceneTree` ile başlamalı)
            path: Proje dizini (boş = çalışma alanı)
            timeout: Saniye (duvar saati üst sınırı)
            script_name: Üretilecek dosya adı (uzantısız)
            quit_after: Güvenlik ağı — bu kadar iterasyon sonra motor kapanır

        Returns:
            `{"ok", "exit_code", "stdout", "stderr", "errors", "duration_ms",
             "script_path", "result", "command"}`
            — `result`, betiğin yazdırdığı `NC_RESULT` JSON'udur (varsa).
        """
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        if not self.executable():
            return self._missing_executable()
        if not str(code or "").strip():
            return {"ok": False, "error": "boş GDScript kodu"}

        stem = _safe_file_stem(script_name)
        tools_dir = root / NC_TOOLS_DIR
        try:
            tools_dir.mkdir(parents=True, exist_ok=True)
            script_file = tools_dir / f"{stem}.gd"
            text = str(code)
            if not text.endswith("\n"):
                text += "\n"
            script_file.write_text(text, encoding="utf-8", newline="\n")
        except OSError as exc:
            self._counter["errors"] += 1
            return {"ok": False, "error": f"betik yazılamadı: {exc}",
                    "script_path": str(tools_dir / f"{stem}.gd")}

        relative = f"res://{NC_TOOLS_DIR}/{stem}.gd"
        arguments: List[Any] = ["--headless"]
        if int(quit_after) > 0:
            arguments += ["--quit-after", str(int(quit_after))]
        arguments += ["--path", str(root), "--script", relative]
        outcome = self._run_godot(arguments, timeout=timeout, cwd=root)
        stdout = str(outcome.get("stdout", ""))
        stderr = str(outcome.get("stderr", ""))
        self._counter["scripts_run"] += 1

        errors = self._collect_errors(f"{stdout}\n{stderr}")
        result = self._parse_result(stdout) or self._parse_result(stderr)
        exit_code = int(outcome.get("returncode", -1))
        ok = bool(outcome.get("ok")) and not errors and exit_code == 0
        if result is not None and not result.get("ok", False):
            ok = False
            if not errors:
                errors = [{"kind": "gdscript_result",
                           "message": "; ".join(str(item) for item in result.get("errors", []))
                           or "betik başarısız bildirdi",
                           "hint": "betiğin NC_RESULT satırındaki `errors` alanına bakın"}]
        if not ok and not errors:
            errors = [{
                "kind": "exit_code",
                "message": f"Godot {exit_code} çıkış koduyla bitti",
                "hint": ("çıktı boşsa motor GUI sürümüyle çalışıyor olabilir; "
                         "`_console.exe` kullanın (godot.executable)"),
            }]

        # `--script` desteği yalnızca taban sınıf hatasıyla kesinleşir: sıradan
        # bir parse hatası (`failed_load_script`) yeteneğin yokluğunu göstermez.
        unsupported = any(item.get("kind") == "wrong_base_class" for item in errors)
        summary: Dict[str, Any] = {
            "ok": ok,
            "exit_code": exit_code,
            "stdout": _summarize_stdout(stdout),
            "stderr": _summarize_stdout(stderr, 8000),
            "errors": errors,
            "duration_ms": outcome.get("duration_ms", 0.0),
            "script_path": relative,
            "script_file": str(script_file),
            "command": outcome.get("command", ""),
            "timed_out": bool(outcome.get("timed_out")),
            "timeout_s": float(timeout),
            "quit_after": int(quit_after),
        }
        if result is not None:
            summary["result"] = result
        if not ok:
            self._counter["errors"] += 1
            summary["error"] = (outcome.get("error") or errors[0]["message"]
                                if errors else "betik başarısız")
        if unsupported:
            self._script_unsupported = True
            summary["hint"] = (
                "`--script` yalnızca düzenleyici (editor) yapılarında ve "
                "`disable_path_overrides=false` ile derlenmiş şablonlarda çalışır. "
                "Şu anki yürütülebilir dosya bunu desteklemiyor olabilir; tam "
                "Godot düzenleyici ikilisi kullanın (godot.health)."
            )
        self._emit("godot.script_run", {
            "script": relative, "ok": ok, "exit_code": exit_code,
            "error_count": len(errors),
            "duration_ms": outcome.get("duration_ms", 0.0),
        })
        if ok:
            logger.debug(f"GDScript çalıştı: {relative} ({exit_code})")
        else:
            logger.warning(
                f"GDScript başarısız: {relative} — "
                f"{errors[0]['message'][:160] if errors else 'bilinmeyen hata'}"
            )
        return summary

    @staticmethod
    def _parse_result(stdout: str) -> Optional[Dict[str, Any]]:
        """
        Çıktıdaki `NC_RESULT {...}` satırını çözer.

        Birden çok satır varsa **sonuncusu** geçerlidir (betik ilerledikçe
        daha kesin bilgi yazar).
        """
        found: Optional[Dict[str, Any]] = None
        for line in str(stdout or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith(NC_RESULT_PREFIX):
                continue
            payload = stripped[len(NC_RESULT_PREFIX):]
            try:
                decoded = json.loads(payload)
            except (ValueError, TypeError):
                continue
            if isinstance(decoded, dict):
                found = decoded
        return found

    # ─── Sahne Düzenleme ────────────────────────────────────────────────

    def _scene_script(self, action: str, scene_path: str,
                      operations: Sequence[Dict[str, Any]]) -> str:
        """
        Sahne düzenleme betiğini üretir.

        İşlemler JSON olarak gömülür (GDScript sözlük değişmezi yazmak yerine):
        kaçış kuralları tek yerde kalır ve sayı/boolean türleri korunur.
        """
        payload = json.dumps(list(operations), ensure_ascii=False)
        return (
            _GD_HEADER
            + _GD_PRELUDE
            + "\nconst NC_SCENE_PATH: String = " + _gd_string(scene_path) + "\n"
            + "const NC_ACTION: String = " + _gd_string(action) + "\n"
            + "const NC_OPERATIONS_JSON: String = " + _gd_string(payload) + "\n"
            + _GD_SCENE_MAIN
        )

    def _scene_op(self, action: str, scene_path: str, path: str,
                  operations: Sequence[Dict[str, Any]],
                  timeout: float = 120.0) -> Dict[str, Any]:
        """Sahne işlemini üretilen betikle çalıştırır ve sonucu sadeleştirir."""
        if not str(scene_path or "").strip():
            return {"ok": False, "error": "sahne yolu boş",
                    "hint": "örnek: res://Level1.tscn"}
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        res_path = self._to_res_path(root, scene_path)
        if not res_path:
            return {"ok": False,
                    "error": f"sahne yolu proje dışında: {scene_path}",
                    "hint": "sahneler `res://` altında olmalı (proje köküne göreli)"}
        if not res_path.lower().endswith((".tscn", ".scn")):
            res_path += ".tscn"

        payload = json.dumps(list(operations), ensure_ascii=False)
        if len(payload) > 400_000:
            return {"ok": False,
                    "error": "işlem listesi çok büyük (400 KB üstü)",
                    "hint": "düğümleri partiler hâlinde ekleyin"}

        outcome = self.run_script(
            self._scene_script(action, res_path, operations),
            path=str(root), timeout=timeout,
            script_name=f"scene_{action}",
        )
        if not outcome.get("ok"):
            return {
                "ok": False, "scene": res_path,
                "error": outcome.get("error") or "sahne işlemi başarısız",
                "errors": outcome.get("errors", []),
                "stdout": outcome.get("stdout", ""),
                "stderr": outcome.get("stderr", ""),
                "script_path": outcome.get("script_path", ""),
                "hint": "godot.diagnose ile çıktıyı yorumlayın",
            }

        result = outcome.get("result") or {}
        absolute = root / res_path.replace("res://", "").replace("/", os.sep)
        inspecting = action == "inspect_scene"
        if inspecting:
            self._counter["scenes_inspected"] += 1
        else:
            self._counter["scenes_written"] += 1
        summary: Dict[str, Any] = {
            "ok": True,
            "scene": res_path,
            "file": str(absolute),
            "exists": absolute.is_file(),
            "existed_before": bool(result.get("existed")),
            "saved": bool(result.get("saved")),
            "applied": result.get("applied", []),
            "errors": [],
            "duration_ms": outcome.get("duration_ms", 0.0),
            "script_path": outcome.get("script_path", ""),
            "size": absolute.stat().st_size if absolute.is_file() else 0,
        }
        if result.get("verified"):
            # Kayıt sonrası diskten geri okunan düğüm sayısı — sessiz "boş
            # sahne" hatasının yakalandığı yer.
            summary["verified"] = result["verified"]
        if inspecting:
            summary["nodes"] = result.get("nodes", [])
            summary["node_count"] = int(result.get("node_count", 0) or 0)
            summary["root"] = result.get("root", {})
            self._emit("godot.scene_inspected", {
                "scene": res_path, "node_count": summary["node_count"],
            })
        else:
            self._emit("godot.scene_written", {
                "scene": res_path, "action": action,
                "applied": len(summary["applied"]),
            })
        return summary

    def create_scene(self, scene_path: str, nodes: Sequence[Dict[str, Any]],
                     path: str = "") -> Dict[str, Any]:
        """
        Sahne oluşturur (varsa üzerine düğüm ekler).

        `nodes` içindeki **parent alanı boş olan ilk öğe kök** olur; diğerleri
        `parent` ile belirtilen düğümün altına eklenir (varsayılan `"."` = kök).
        Kök belirtilmezse `Node3D` kök üretilir.

        Args:
            scene_path: `res://` yolu (ör. `res://Level1.tscn`)
            nodes: `[{"type": "Node3D", "name": "Root"},
                     {"type": "MeshInstance3D", "name": "Box", "parent": ".",
                      "properties": {"position": "0,1,0"}}]`
            path: Proje dizini (boş = çalışma alanı)

        Returns:
            `{"ok", "scene", "file", "exists", "applied", "error"}`
        """
        specs = list(nodes or [])
        if not specs:
            return {"ok": False, "error": "düğüm listesi boş",
                    "hint": "en az bir kök düğüm verin: "
                            '[{"type": "Node3D", "name": "Root"}]'}
        operations: List[Dict[str, Any]] = []
        root_spec: Optional[Dict[str, Any]] = None
        for spec in specs:
            if not isinstance(spec, dict):
                return {"ok": False, "error": f"düğüm tanımı sözlük olmalı: {spec!r}"}
            node_type = str(spec.get("type") or spec.get("node_type") or "").strip()
            if not node_type:
                return {"ok": False, "error": "düğüm tanımında 'type' eksik",
                        "hint": "örnek: {\"type\": \"Node3D\", \"name\": \"Root\"}"}
            parent = str(spec.get("parent") or "").strip()
            if root_spec is None and not parent:
                root_spec = spec
                continue
            operations.append({
                "op": "add",
                "type": node_type,
                "name": str(spec.get("name") or ""),
                "parent": parent or ".",
                "properties": spec.get("properties") or {},
            })
        if root_spec is None:
            root_spec = {"type": "Node3D", "name": "Root"}
        operations.insert(0, {
            "op": "create_root",
            "type": str(root_spec.get("type") or "Node3D"),
            "name": str(root_spec.get("name") or ""),
        })
        return self._scene_op("create_scene", scene_path, path, operations)

    def add_node(self, scene_path: str, node_type: str, node_name: str = "",
                 parent: str = "", properties: Optional[Dict[str, Any]] = None,
                 path: str = "") -> Dict[str, Any]:
        """
        Var olan sahneye düğüm ekler.

        Düğüm `ClassDB.instantiate(node_type)` ile üretilir (Godot 3.x'teki
        `ClassDB.instance` değil); sınıf yoksa veya soyutsa net bir hata döner.

        Args:
            scene_path: Sahne (`res://...`)
            node_type: Godot 4 sınıf adı (`Camera3D`, `MeshInstance3D`, ...)
            node_name: Düğüm adı (boş = sınıf adından türetilir)
            parent: Üst düğüm yolu (boş veya `"."` = kök)
            properties: Başlangıç özellikleri (dize değerler türe çevrilir)
            path: Proje dizini

        Returns:
            `{"ok", "scene", "applied", "error"}`
        """
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        res_path = self._to_res_path(root, scene_path)
        if not res_path:
            return {"ok": False, "error": f"sahne yolu proje dışında: {scene_path}"}
        if not res_path.lower().endswith((".tscn", ".scn")):
            res_path += ".tscn"
        absolute = root / res_path.replace("res://", "").replace("/", os.sep)
        if not absolute.is_file():
            return {"ok": False, "error": f"sahne yok: {res_path}",
                    "hint": "önce godot.create_scene ile sahneyi oluşturun",
                    "scene": res_path}
        clean_type = str(node_type or "").strip()
        if not clean_type:
            return {"ok": False, "error": "node_type boş",
                    "hint": "Godot 4 sınıf adı verin (ör. Camera3D)"}
        operations = [{
            "op": "add",
            "type": clean_type,
            "name": str(node_name or "").strip(),
            "parent": str(parent or ".").strip() or ".",
            "properties": dict(properties or {}),
        }]
        return self._scene_op("add_node", res_path, str(root), operations)

    def set_node_property(self, scene_path: str, node_path: str, property_name: str,
                          value: Any, path: str = "") -> Dict[str, Any]:
        """
        Sahnedeki bir düğümün özelliğini ayarlar ve **geri okuyarak doğrular**.

        Dize değerler hedef özelliğin mevcut türüne göre çevrilir: sayılar,
        boolean'lar, `Vector2/3/4` ("x,y,z"), `Color`, `NodePath` ve
        `res://` yollu kaynaklar (`mesh`, `texture`, `script`).

        Args:
            scene_path: Sahne (`res://...`)
            node_path: Düğüm yolu (`"."` = kök, `"Kamera"`, `"Root/Box"`)
            property_name: Özellik adı (ör. `fov`)
            value: Değer (dize veya sayı/bool)
            path: Proje dizini

        Returns:
            `{"ok", "scene", "applied": [{"property", "value", "type"}], "error"}`
        """
        if not str(property_name or "").strip():
            return {"ok": False, "error": "property_name boş"}
        operations = [{
            "op": "set",
            "path": str(node_path or ".").strip() or ".",
            "property": str(property_name).strip(),
            "value": value,
        }]
        return self._scene_op("set_node_property", scene_path, path, operations)

    def remove_node(self, scene_path: str, node_path: str,
                    path: str = "") -> Dict[str, Any]:
        """
        Sahneden düğümü (ve alt ağacını) siler.

        Kök düğüm silinemez: bir sahnenin kökü olmadan `.tscn` geçersiz olur.

        Args:
            scene_path: Sahne (`res://...`)
            node_path: Silinecek düğüm yolu
            path: Proje dizini

        Returns:
            `{"ok", "scene", "applied", "error"}`
        """
        if not str(node_path or "").strip():
            return {"ok": False, "error": "node_path boş",
                    "hint": "silmek için düğüm yolunu verin (ör. 'Kamera')"}
        operations = [{"op": "remove", "path": str(node_path).strip()}]
        return self._scene_op("remove_node", scene_path, path, operations)

    def inspect_scene(self, scene: str, path: str = "") -> Dict[str, Any]:
        """
        Sahneyi motorun içinde açar ve düğüm ağacını okur.

        Metin ayrıştırmak yerine gerçek `PackedScene` örneklenir; bu yüzden
        sonuç **kesindir**: düğüm adları, sınıflar, yollar, betikler ve seçilmiş
        özellikler (konum, kamera açısı, ışık enerjisi, ...) döner.

        Args:
            scene: Sahne (`res://...`)
            path: Proje dizini

        Returns:
            `{"ok", "scene", "node_count", "root", "nodes": [
             {"name", "type", "path", "parent", "children", "script",
              "properties"}], "error"}`
        """
        return self._scene_op("inspect_scene", scene, path, [{"op": "inspect"}])

    # ─── Proje Tarama ───────────────────────────────────────────────────

    @staticmethod
    def _walk_project(root: Path, suffixes: Sequence[str],
                      limit: int = 500) -> List[Path]:
        """Proje içindeki dosyaları (`.godot` hariç) sınırlı biçimde dolaşır."""
        results: List[Path] = []
        for current, directories, filenames in os.walk(root):
            directories[:] = [
                name for name in sorted(directories)
                if name not in (".godot", ".git", "addons", "__pycache__")
            ]
            for name in sorted(filenames):
                if Path(name).suffix.lower() in suffixes:
                    results.append(Path(current) / name)
                    if len(results) >= limit:
                        return results
        return results

    def _scene_summary(self, root: Path, file_path: Path) -> Dict[str, Any]:
        """Bir `.tscn` dosyasının kök düğümünü ve düğüm sayısını metinden okur."""
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        # ext_resource kimlikleri kendi ad alanındadır ve öznitelik sırası
        # Godot sürümüne göre değişir (uid= alanı 4.4+ ile eklendi); bu yüzden
        # öznitelikler sıradan bağımsız olarak sözlüğe çevrilir.
        resources: Dict[str, str] = {}
        for match in re.finditer(r"^\[ext_resource\s+([^\]]*)\]", text, re.MULTILINE):
            attributes = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
            if attributes.get("id") and attributes.get("path"):
                resources[attributes["id"]] = attributes["path"]
        root_name, root_type, root_script = "", "", ""
        for match in re.finditer(r"^\[node\s+([^\]]*)\]", text, re.MULTILINE):
            attributes = match.group(1)
            if "parent=" in attributes:
                continue
            name_match = re.search(r'name="([^"]*)"', attributes)
            type_match = re.search(r'type="([^"]*)"', attributes)
            root_name = name_match.group(1) if name_match else ""
            root_type = type_match.group(1) if type_match else ""
            break
        script_match = re.search(r'^script\s*=\s*ExtResource\("([^"]+)"\)',
                                 text, re.MULTILINE)
        if script_match:
            root_script = resources.get(script_match.group(1), "")
        try:
            relative = file_path.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover
            relative = file_path.name
        return {
            "path": "res://" + relative,
            "file": str(file_path),
            "root_name": root_name,
            "root_type": root_type,
            "root_script": root_script,
            "nodes": len(re.findall(r"^\[node\s", text, re.MULTILINE)),
            "size": file_path.stat().st_size if file_path.is_file() else 0,
        }

    def list_scenes(self, path: str = "") -> Dict[str, Any]:
        """
        Projedeki sahneleri listeler (kök düğüm türü ve düğüm sayısıyla).

        Returns:
            `{"ok", "root", "scenes": [{"path", "root_name", "root_type",
             "root_script", "nodes"}], "count", "error"}`
        """
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        scenes = [self._scene_summary(root, item)
                  for item in self._walk_project(root, SCENE_SUFFIXES)]
        scenes.sort(key=lambda item: item["path"])
        return {"ok": True, "root": str(root), "scenes": scenes,
                "count": len(scenes),
                "note": f"{len(scenes)} sahne bulundu"}

    def list_scripts(self, path: str = "") -> Dict[str, Any]:
        """
        Projedeki GDScript dosyalarını listeler (`extends` ve `class_name` ile).

        `class_name` bilgisi kritiktir: başka bir betiğin bu sınıfı
        kullanabilmesi için bildirimin **taranmış** olması gerekir
        (`godot.import_project`).

        Returns:
            `{"ok", "root", "scripts": [{"path", "extends", "class_name",
             "lines", "functions"}], "count", "classes"}`
        """
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        scripts: List[Dict[str, Any]] = []
        for item in self._walk_project(root, (".gd",)):
            try:
                text = item.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            extends_match = re.search(r"^extends\s+([^\n#]+)", text, re.MULTILINE)
            class_match = re.search(r"^class_name\s+([A-Za-z_]\w*)", text, re.MULTILINE)
            try:
                relative = item.relative_to(root).as_posix()
            except ValueError:  # pragma: no cover
                relative = item.name
            scripts.append({
                "path": "res://" + relative,
                "file": str(item),
                "extends": extends_match.group(1).strip() if extends_match else "",
                "class_name": class_match.group(1) if class_match else "",
                "lines": len(text.splitlines()),
                "functions": len(re.findall(r"^func\s+\w+", text, re.MULTILINE)),
            })
        scripts.sort(key=lambda entry: entry["path"])
        return {
            "ok": True, "root": str(root), "scripts": scripts,
            "count": len(scripts),
            "classes": sorted(entry["class_name"] for entry in scripts
                              if entry["class_name"]),
        }

    def read_project_settings(self, path: str = "") -> Dict[str, Any]:
        """
        `project.godot` ayarlarını okur (autoload, ana sahne, girdi eylemleri).

        Godot'un biçimi INI'ye benzer ama `configparser`'a değil: değerler
        `PackedStringArray("4.8")` gibi motor ifadeleridir ve anahtarlar `/`
        içerir. Bu yüzden satır bazlı, ham metin bırakan bir ayrıştırma yapılır.

        Returns:
            `{"ok", "path", "config_version", "name", "main_scene", "features",
             "autoload", "input_actions", "sections", "raw", "error"}`
        """
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        project_file = root / "project.godot"
        try:
            text = project_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"ok": False, "error": f"project.godot okunamadı: {exc}"}

        sections = _parse_ini(text)
        application = sections.get("application", {})
        display = sections.get("display", {})
        autoload_raw = sections.get("autoload", {})
        autoload = {key: value.strip('"') for key, value in autoload_raw.items()}
        raw_features = application.get("config/features", "")
        return {
            "ok": True,
            "path": str(root),
            "project_file": str(project_file),
            "config_version": int(sections.get("", {}).get("config_version", 0) or 0),
            "name": application.get("config/name", "").strip('"') or root.name,
            "main_scene": application.get("run/main_scene", "").strip('"'),
            "features": _ini_list(raw_features),
            "icon": application.get("config/icon", "").strip('"'),
            "autoload": autoload,
            "autoload_count": len(autoload),
            "input_actions": sorted(sections.get("input", {}).keys()),
            "input_action_count": len(sections.get("input", {})),
            "display": display,
            "rendering_method": sections.get("rendering", {}).get(
                "renderer/rendering_method", "").strip('"'),
            "sections": {name: len(values) for name, values in sections.items() if name},
            "raw": text[:20000],
        }

    def list_export_presets(self, path: str = "",
                            probe: bool = False) -> Dict[str, Any]:
        """
        Dışa aktarma presetlerini listeler.

        Godot'un **liste bayrağı yoktur**; motoru konuşturmanın tek yolu kasıtlı
        olarak geçersiz bir preset adıyla dışa aktarma denemektir: motor
        `Invalid export preset name: X.` yazıp ardından
        `The following presets were detected in this project's
        ``export_presets.cfg``:` bloğunda adları sıralar. Bu modül dosyayı
        kendi ayrıştırır (BOM'u temizleyerek) ve gerektiğinde motoru sondalar.

        Doğrulanmış tuzak: **UTF-8 BOM içeren `export_presets.cfg` motor
        tarafından tümden reddedilir** — dosya geçerli görünse bile preset
        listesi boş döner ve `--export-release` "Invalid export preset name"
        hatası verir. PowerShell'in `Set-Content -Encoding utf8` komutu
        (5.1'de) BOM ekler; Godot Düzenleyicisi eklemez. `bom=True` alanı bu
        durumu bildirir.

        Args:
            path: Proje dizini
            probe: Dosya ayrıştırması yeterli görünse bile motoru da sorgula

        Returns:
            `{"ok", "presets", "count", "file", "bom", "engine_accepted",
             "engine_presets", "probe_output", "error", "hint"}`
        """
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        preset_file = root / "export_presets.cfg"
        if not preset_file.is_file():
            return {
                "ok": False,
                "error": "export_presets.cfg yok — projede dışa aktarma preseti tanımlı değil",
                "hint": ("Godot Düzenleyici > Proje > Dışa Aktar menüsünden bir "
                         "preset ekleyip kaydedin (headless modda oluşturulamaz)"),
                "file": str(preset_file),
                "presets": [],
                "count": 0,
            }
        try:
            raw = preset_file.read_bytes()
        except OSError as exc:
            return {"ok": False, "error": f"export_presets.cfg okunamadı: {exc}"}
        has_bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8", errors="replace").lstrip("\ufeff")

        sections = _parse_ini(text)
        presets: List[Dict[str, Any]] = []
        for name, values in sections.items():
            if not name.startswith("preset.") or name.endswith(".options"):
                continue
            platform = values.get("platform", "").strip('"')
            presets.append({
                "name": values.get("name", "").strip('"'),
                "platform": platform,
                "family": _family_for_platform(platform),
                "export_path": values.get("export_path", "").strip('"'),
                "runnable": values.get("runnable", "false").strip().lower() == "true",
                "section": name,
                "has_options": any(
                    key.startswith(name + ".options") for key in sections),
                "custom_features": values.get("custom_features", "").strip('"'),
            })
        presets.sort(key=lambda entry: entry["name"])

        result: Dict[str, Any] = {
            "ok": True, "presets": presets, "count": len(presets),
            "file": str(preset_file), "root": str(root),
            "bom": has_bom,
            "engine_accepted": None,
            "engine_presets": [],
            "probe_output": "",
        }
        if has_bom:
            result["hint"] = (
                "export_presets.cfg UTF-8 BOM ile başlıyor; Godot bu dosyayı "
                "reddeder ve preset listesi boş görünür — dosyayı BOM'suz "
                "UTF-8 olarak kaydedin (PowerShell 5.1 `Set-Content -Encoding "
                "utf8` BOM ekler; Godot Düzenleyicisi eklemez)")

        # Motor sondası: dosya ayrıştırması şüpheliyse (BOM, options eksik,
        # hiç preset yok) veya açıkça istendiyse.
        needs_probe = bool(
            probe or not presets or has_bom
            or any(not entry["has_options"] for entry in presets)
        )
        if needs_probe:
            probed = self._probe_export_presets(root)
            names = probed.get("names", [])
            result["engine_presets"] = names
            result["probe_output"] = probed.get("text", "")
            result["engine_accepted"] = bool(names)
            if not names and presets:
                result["hint"] = (
                    "dosyada preset tanımı var ama motor hiçbirini kabul etmiyor "
                    "— `[preset.N.options]` bölümünü ekleyin ve dosyayı BOM'suz "
                    "UTF-8 olarak kaydedin")
            elif names and not presets:
                result["hint"] = ("presetler yalnızca motordan okunabildi; "
                                  "export_presets.cfg elle yazılmış olabilir")
            if not names and not presets:
                result["ok"] = False
                result["error"] = "dışa aktarma preseti bulunamadı"
                result["hint"] = ("export_presets.cfg var ama motor hiçbir preset "
                                  "tanımıyor — Godot Düzenleyicisi'nden kaydedin")
        return result

    def _probe_export_presets(self, root: Path) -> Dict[str, Any]:
        """
        Motoru geçersiz bir preset adıyla dışa aktarmaya zorlayıp listeyi okur.

        Godot'un preset listeleme bayrağı yoktur; bu kasıtlı hata, adları
        yazdıran **tek** güvenilir yoldur.
        """
        if not self.executable():
            return {"names": [], "text": ""}
        dummy = root / "nc_preset_probe.exe"
        outcome = self._run_godot(
            ["--headless", "--path", str(root), "--export-release",
             "__nc_probe__", str(dummy)],
            timeout=180.0, cwd=root,
        )
        text = f"{outcome.get('stdout', '')}\n{outcome.get('stderr', '')}"
        names: List[str] = []
        match = re.search(
            r"presets were detected in this project's `export_presets\.cfg`:\s*\n(.*)",
            text, re.DOTALL | re.IGNORECASE,
        )
        if match:
            for line in match.group(1).splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('"') and stripped.endswith('"'):
                    names.append(stripped.strip('"'))
                    continue
                if re.match(r"^(ERROR|WARNING|at:|\+|\s*CategoryInfo)", stripped):
                    break
        return {"names": names, "text": _summarize_stdout(text, 4000)}

    # ─── Çalıştırma ve Dışa Aktarma ─────────────────────────────────────

    def run_project(self, path: str = "", timeout: float = 60.0, scene: str = "",
                    headless: bool = True, quit_after: int = 180) -> Dict[str, Any]:
        """
        Projeyi (veya belirtilen sahneyi) çalıştırır — duman testi.

        `--quit-after` verilir (varsayılan 180 iterasyon ≈ 3 saniye): bir oyun
        döngüsü kendiliğinden kapanmaz, bu yüzden motor bu sayıda kareden sonra
        zorla kapatılır. Bu hem "açılışta hata veriyor mu?" sorusunu yanıtlar
        hem de asılı kalan bir sürecin ajanı kilitlemesini engeller.

        Args:
            path: Proje dizini
            timeout: Saniye (duvar saati üst sınırı)
            scene: Çalıştırılacak sahne (`res://...`, boş = ana sahne)
            headless: Görsel pencere açmadan çalıştır
            quit_after: Kaç iterasyon sonra kapanacağı (0 = sınırsız, önerilmez)

        Returns:
            `{"ok", "exit_code", "stdout", "stderr", "errors", "command"}`
        """
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        if not self.executable():
            return self._missing_executable()

        arguments: List[Any] = []
        if headless:
            arguments.append("--headless")
        if scene:
            res_path = self._to_res_path(root, scene)
            if not res_path:
                return {"ok": False, "error": f"sahne proje dışında: {scene}",
                        "hint": "sahneler `res://` altında olmalı"}
            arguments.append(res_path)
        arguments += ["--path", str(root)]
        if int(quit_after) > 0:
            arguments += ["--quit-after", str(int(quit_after))]

        outcome = self._run_godot(arguments, timeout=timeout, cwd=root)
        stdout = str(outcome.get("stdout", ""))
        stderr = str(outcome.get("stderr", ""))
        errors = self._collect_errors(f"{stdout}\n{stderr}")
        self._counter["runs"] += 1
        ok = bool(outcome.get("ok")) and not errors
        summary: Dict[str, Any] = {
            "ok": ok,
            "exit_code": outcome.get("returncode"),
            "scene": scene or "(ana sahne)",
            "headless": bool(headless),
            "stdout": _summarize_stdout(stdout),
            "stderr": _summarize_stdout(stderr, 8000),
            "errors": errors,
            "duration_ms": outcome.get("duration_ms", 0.0),
            "command": outcome.get("command", ""),
            "timed_out": bool(outcome.get("timed_out")),
            "quit_after": int(quit_after),
        }
        if not ok:
            self._counter["errors"] += 1
            summary["error"] = outcome.get("error") or (
                errors[0]["message"] if errors else "proje çalıştırılamadı")
        self._emit("godot.project_run", {
            "path": str(root), "scene": scene, "ok": ok,
            "exit_code": outcome.get("returncode"),
        })
        return summary

    def run_scene(self, scene: str, path: str = "", timeout: float = 60.0,
                  quit_after: int = 180) -> Dict[str, Any]:
        """
        Tek bir sahneyi başsız (headless) çalıştırır.

        Ajanın "sahne açılışta hata veriyor mu?" sorusunu yanıtlar; çıktıdaki
        `SCRIPT ERROR` ve `Node not found` satırları yapılandırılmış hâle gelir.
        `--quit-after` güvenlik ağı `run_project` ile aynıdır.
        """
        return self.run_project(path=path, timeout=timeout, scene=scene,
                                headless=True, quit_after=quit_after)

    def _templates_root(self) -> Path:
        """Dışa aktarma şablonlarının kök dizini (`%APPDATA%/Godot/export_templates`)."""
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / "Godot" / "export_templates"

    def export_templates(self) -> Dict[str, Any]:
        """
        Dışa aktarma şablonlarının kurulu olup olmadığını denetler.

        Bu, Godot'ta **en sık** karşılaşılan kurulum eksiğidir: motor projeyi
        dışa aktaramaz çünkü platform şablon dosyaları (`windows_release_x86_64.exe`
        vb.) indirilmemiştir. Motorun ham hatası bunu dolaylı anlatır; burada
        dosya sistemi doğrudan denetlenir ve eylem önerisi üretilir.

        Returns:
            `{"installed", "dir", "files", "version", "checked", "hint"}`
        """
        version = self.version()
        checked: List[str] = []
        directories: List[Path] = []

        exe = self.executable()
        if exe:
            sibling = Path(exe).parent / "templates"
            directories.append(sibling)
        root = self._templates_root()
        if root.is_dir():
            try:
                directories.extend(sorted(
                    entry for entry in root.iterdir() if entry.is_dir()))
            except OSError:  # pragma: no cover
                pass
        if version:
            # Motor sürümü "4.8.dev7.official.c971f93e7" biçiminde; şablon
            # dizini genellikle durum son ekini içermez ("4.8.dev7").
            for candidate in (version, ".".join(version.split(".")[:3]),
                              ".".join(version.split(".")[:2])):
                directories.append(root / candidate)

        for directory in directories:
            checked.append(str(directory))
            if not directory.is_dir():
                continue
            matched = sorted(
                entry.name for entry in directory.iterdir()
                if entry.is_file() and entry.suffix.lower() in (".exe", ".zip", ".apk")
                and (entry.name.startswith(("windows_", "linux_", "macos", "web_",
                                            "android_", "ios_")) or "template" in entry.name)
            )
            if matched:
                return {
                    "installed": True,
                    "dir": str(directory),
                    "files": matched[:40],
                    "file_count": len(matched),
                    "version": version,
                    "templates_root": str(root),
                    "checked": checked,
                    "hint": "",
                }

        return {
            "installed": False,
            "dir": "",
            "files": [],
            "file_count": 0,
            "version": version,
            "templates_root": str(root),
            "checked": checked,
            "hint": (
                f"Bu Godot sürümü ({version or 'bilinmiyor'}) için dışa aktarma "
                "şablonları kurulu değil. Godot Düzenleyici > Düzenleyici > "
                "Dışa Aktarma Şablonlarını Yönet menüsünden indirin; şablonlar "
                f"şuraya kurulur: {root}\\<sürüm>"
            ),
        }

    def export_project(self, preset: str = "Windows Desktop", output: str = "",
                       path: str = "", debug: bool = False,
                       timeout: float = 900.0,
                       import_first: bool = True) -> Dict[str, Any]:
        """
        Projeyi bir dışa aktarma presetiyle derler.

        Çağrıdan **önce** üç yaygın hata nedeni denetlenir ve anlaşılır bir
        öneriyle döndürülür:
            1. `export_presets.cfg` yok (preset tanımlı değil).
            2. Preset adı motor tarafından tanınmıyor (liste sondajla doğrulanır).
            3. Dışa aktarma şablonları kurulu değil (dosya sistemi denetlenir).
        Ayrıca **hedef dizin oluşturulur**: Godot 4.8 hedef klasör yoksa
        "Belirtilen dışa aktarım yolu mevcut değil" hatası verir (mesaj
        yerelleştirilmiş olabilir), bu da ham çıktıda anlaşılmaz görünür.

        `--export-release` **`--import` anlamına gelmez** (`--export-debug`,
        `--export-pack` ve `--export-patch` anlamına gelir). Bu yüzden sürüm
        derlemesinden önce `import_first=True` ile proje içe aktarılır.

        DİKKAT: **Dışa aktarma, GDScript hataları varken de exit 0 verir** ve
        paketi yazar. Bu yüzden `ok` yalnızca çıkış koduna değil, üretilen
        dosyanın varlığına ve çıktıdaki hata işaretlerine de bakar.

        Args:
            preset: Preset adı (`export_presets.cfg` içindeki `name`)
            output: Çıktı yolu (boş = preset'in `export_path` değeri veya
                `<proje>/build/<ad>`). Motor yolu **proje dizinine göre** çözer;
                bu modül her zaman mutlak yol geçer.
            path: Proje dizini
            debug: True ise `--export-debug`, değilse `--export-release`
            timeout: Saniye (derleme uzun sürebilir)
            import_first: Sürüm derlemesinden önce `--import` çalıştır

        Returns:
            `{"ok", "preset", "output", "exists", "size", "command",
             "stdout", "stderr", "error", "hint"}`
        """
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        if not self.executable():
            return self._missing_executable()

        preset_file = root / "export_presets.cfg"
        if not preset_file.is_file():
            return {
                "ok": False,
                "error": "export_presets.cfg yok — dışa aktarma preseti tanımlı değil",
                "hint": ("Godot Düzenleyici > Proje > Dışa Aktar menüsünden bir "
                         "preset ekleyip kaydedin (headless modda preset "
                         "oluşturulamaz, dosya elle de yazılabilir)"),
                "preset_file": str(preset_file),
            }

        listed = self.list_export_presets(str(root))
        available = listed.get("presets", []) if listed.get("ok") else []
        names = [entry["name"] for entry in available] or list(
            listed.get("engine_presets", []))
        chosen = next((entry for entry in available if entry["name"] == preset), None)
        if names and preset not in names:
            return {
                "ok": False,
                "error": f"preset bulunamadı: '{preset}'",
                "available": names,
                "hint": f"geçerli presetler: {', '.join(names)}",
            }
        if not names:
            return {
                "ok": False,
                "error": "motor hiçbir dışa aktarma preseti tanımıyor",
                "hint": listed.get("hint") or (
                    "export_presets.cfg dosyasını Godot düzenleyicisinden "
                    "kaydedin; dosya UTF-8 BOM içeriyorsa motor onu tümden "
                    "reddeder"),
                "probe_output": listed.get("probe_output", ""),
            }
        if listed.get("engine_accepted") is False:
            # Dosyada preset görünüyor ama motor reddediyor (en sık nedeni BOM).
            return {
                "ok": False,
                "error": "motor export_presets.cfg dosyasını reddediyor",
                "hint": listed.get("hint") or (
                    "dosyayı BOM'suz UTF-8 olarak yeniden kaydedin"),
                "bom": listed.get("bom", False),
                "engine_presets": listed.get("engine_presets", []),
                "file_presets": names,
                "probe_output": listed.get("probe_output", ""),
            }

        templates = self.export_templates()
        if not templates.get("installed"):
            return {
                "ok": False,
                "error": "dışa aktarma şablonları kurulu değil",
                "hint": templates.get("hint", ""),
                "templates": templates,
                "preset": preset,
            }

        target = str(output or "").strip()
        if not target and chosen:
            target = str(chosen.get("export_path") or "")
        if not target:
            extension = ".exe" if "windows" in str(
                (chosen or {}).get("platform", "")).lower() else ""
            target = f"build/{_safe_file_stem(preset, 'build')}{extension}"
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = root / target_path
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": f"hedef dizin oluşturulamadı: {exc}",
                    "output": str(target_path)}

        imported: Dict[str, Any] = {}
        if import_first and not debug:
            imported = self.import_project(str(root), timeout=min(float(timeout), 300.0))

        flag = "--export-debug" if debug else "--export-release"
        arguments = ["--headless", "--path", str(root), flag, preset, str(target_path)]
        outcome = self._run_godot(arguments, timeout=timeout, cwd=root)
        stdout = str(outcome.get("stdout", ""))
        stderr = str(outcome.get("stderr", ""))
        errors = self._collect_errors(f"{stdout}\n{stderr}")
        produced = target_path.is_file()
        self._counter["exports"] += 1
        ok = bool(outcome.get("ok")) and produced and not errors

        summary: Dict[str, Any] = {
            "ok": ok,
            "preset": preset,
            "output": str(target_path),
            "exists": produced,
            "size": target_path.stat().st_size if produced else 0,
            "debug": bool(debug),
            "exit_code": outcome.get("returncode"),
            "stdout": _summarize_stdout(stdout),
            "stderr": _summarize_stdout(stderr, 8000),
            "errors": errors,
            "duration_ms": outcome.get("duration_ms", 0.0),
            "command": outcome.get("command", ""),
            "templates_dir": templates.get("dir", ""),
            "import_form": imported.get("form", ""),
        }
        if not ok:
            self._counter["errors"] += 1
            summary["error"] = outcome.get("error") or (
                errors[0]["message"] if errors else "dışa aktarma başarısız")
            summary["hint"] = ("ham çıktıdaki ilk ERROR satırı asıl nedendir; "
                               "godot.diagnose ile yorumlayın")
            if produced and errors:
                summary["hint"] = (
                    "paket yazıldı ama çıktıda hatalar var — Godot dışa aktarmada "
                    "GDScript hataları varken de exit 0 verir; hataları düzeltip "
                    "yeniden derleyin")
            elif not produced:
                summary["hint"] = ("çıktı dosyası oluşmadı — preset adını ve "
                                   "şablon kurulumunu denetleyin")
        self._emit("godot.project_exported", {
            "preset": preset, "output": str(target_path), "ok": ok,
            "size": summary["size"],
        })
        return summary

    # ─── Teşhis ve Günlük ───────────────────────────────────────────────

    def error_log(self, path: str = "", lines: int = 80) -> Dict[str, Any]:
        """
        Godot'un günlük dosyasını okur (en yeni olanı tercih eder).

        Godot iki yere yazar:
            * Çalıştırılan oyun: `%APPDATA%\\Godot\\app_userdata\\<proje adı>\\logs\\godot.log`
            * Proje dizini: `<proje>/logs/godot.log` ve `<proje>/godot.log`
              (düzenleyici/başsız çalıştırmalar)

        Dosya yoksa istisna değil, açıklayıcı bir sözlük döner: günlük
        genellikle **ilk** çalıştırmadan sonra oluşur.

        Args:
            path: Proje dizini (boş = çalışma alanı)
            lines: Sondan kaç satır

        Returns:
            `{"ok", "path", "source", "lines": [...], "text", "size",
             "modified", "checked", "error", "hint"}`
        """
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        settings = self.read_project_settings(str(root))
        project_name = str(settings.get("name") or root.name) if settings.get("ok") else root.name

        checked: List[Path] = []
        candidates: List[Path] = []

        appdata = os.environ.get("APPDATA")
        if appdata:
            logs_dir = Path(appdata) / "Godot" / "app_userdata" / project_name / "logs"
            checked.append(logs_dir)
            if logs_dir.is_dir():
                try:
                    candidates.extend(sorted(logs_dir.glob("*.log")))
                except OSError:  # pragma: no cover
                    pass
        for direct in (root / "logs", root, root / ".godot" / "logs"):
            checked.append(direct)
            if direct.is_dir():
                for name in ("godot.log", "godot_editor.log"):
                    candidate = direct / name
                    if candidate.is_file():
                        candidates.append(candidate)
                try:
                    candidates.extend(sorted(direct.glob("godot*.log")))
                except OSError:  # pragma: no cover
                    pass

        unique: Dict[str, Path] = {}
        for candidate in candidates:
            try:
                unique[str(candidate.resolve())] = candidate
            except OSError:  # pragma: no cover
                continue
        existing = [item for item in unique.values() if item.is_file()]
        if not existing:
            return {
                "ok": False,
                "error": "Godot günlük dosyası bulunamadı",
                "hint": ("günlük ilk çalıştırmadan sonra oluşur — "
                         "godot.run_project veya godot.run_scene çağırın"),
                "project": project_name,
                "checked": [str(item) for item in checked],
                "lines": [],
                "text": "",
            }

        newest = max(existing, key=lambda item: item.stat().st_mtime)
        try:
            text = newest.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"ok": False, "error": f"günlük okunamadı: {exc}",
                    "path": str(newest), "checked": [str(item) for item in checked]}
        stat = newest.stat()
        tail = text.splitlines()[-max(1, int(lines)):]
        source = "app_userdata" if "app_userdata" in str(newest) else "proje"
        errors = self._collect_errors("\n".join(tail))
        return {
            "ok": True,
            "path": str(newest),
            "source": source,
            "project": project_name,
            "lines": tail,
            "line_count": len(tail),
            "text": "\n".join(tail),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "errors": errors,
            "checked": [str(item) for item in checked],
        }

    def diagnose(self, text: str, command: str = "") -> Dict[str, Any]:
        """
        Godot hata çıktısını yapılandırılmış hâle getirir.

        Kapsanan kalıplar: `Parse Error` (dosya/satır ile), `SCRIPT ERROR`,
        `Node not found`, `Invalid call`, `Cannot open file`,
        `Condition "..." is true`, null örnek üzerinde çağrı, eksik
        `class_name`, `Failed to load resource` ve eksik dışa aktarma şablonları.

        Args:
            text: Motor çıktısı (stdout + stderr)
            command: Çalıştırılan komut (bağlam için)

        Returns:
            `{"errors": [{"kind", "message", "hint", "file?", "line?"}],
             "summary", "error_count", "command", "warnings"}`
        """
        errors = self._collect_errors(text)
        warnings: List[str] = []
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if not stripped or _is_godot_noise(stripped):
                continue
            if re.search(r"\bWARNING\b|\bUyarı\b|^W\s", stripped, re.IGNORECASE):
                warnings.append(stripped[:200])
            if len(warnings) >= 10:
                break

        if not errors:
            summary = "hata bulunamadı — çıktı temiz görünüyor"
            if warnings:
                summary = f"hata yok, {len(warnings)} uyarı var"
        else:
            primary = errors[0]
            location = ""
            if primary.get("file"):
                location = f" ({primary['file']}"
                if primary.get("line"):
                    location += f":{primary['line']}"
                location += ")"
            summary = (f"{len(errors)} hata; ilki {primary['kind']}{location}: "
                       f"{primary['message'][:160]} → {primary['hint'][:160]}")
        return {
            "errors": errors,
            "error_count": len(errors),
            "warnings": warnings,
            "summary": summary,
            "command": command,
            "ok": not errors,
        }

    # ─── Proje Haritası ─────────────────────────────────────────────────

    def project_map(self, path: str = "") -> Dict[str, Any]:
        """
        Projenin tam haritasını çıkarır.

        Ajanın **ilk** çağırması gereken Godot aracıdır: sahne listesi ve kök
        düğüm türleri, betikler (`extends`/`class_name`), kaynaklar, autoload
        girdileri, girdi eylemleri ve dışa aktarma presetleri tek yanıtta gelir.
        Tüm projeyi okumak yerine nerede ne olduğunu söyler.

        Returns:
            `{"ok", "root", "name", "main_scene", "config_version", "features",
             "scenes", "scripts", "resources", "autoload", "input_actions",
             "export_presets", "counts", "export_templates", "note"}`
        """
        check = self._project_root(path)
        if not check["ok"]:
            return check
        root: Path = check["root"]
        settings = self.read_project_settings(str(root))
        scenes = self.list_scenes(str(root))
        scripts = self.list_scripts(str(root))
        presets = self.list_export_presets(str(root))

        resources: List[Dict[str, Any]] = []
        for item in self._walk_project(root, RESOURCE_SUFFIXES, limit=300):
            kind = ""
            try:
                head = item.read_text(encoding="utf-8", errors="replace")[:400]
                match = re.search(r"^\[(\w[\w\d]*)\s", head, re.MULTILINE)
                kind = match.group(1) if match else ""
            except OSError:
                pass
            try:
                relative = item.relative_to(root).as_posix()
            except ValueError:  # pragma: no cover
                relative = item.name
            resources.append({"path": "res://" + relative, "type": kind,
                              "size": item.stat().st_size if item.is_file() else 0})

        scene_list = scenes.get("scenes", []) if scenes.get("ok") else []
        script_list = scripts.get("scripts", []) if scripts.get("ok") else []
        preset_list = presets.get("presets", []) if presets.get("ok") else []
        templates = self.export_templates()
        counts = {
            "scenes": len(scene_list),
            "scripts": len(script_list),
            "resources": len(resources),
            "class_names": len(scripts.get("classes", [])) if scripts.get("ok") else 0,
            "autoload": len(settings.get("autoload", {})) if settings.get("ok") else 0,
            "input_actions": len(settings.get("input_actions", [])) if settings.get("ok") else 0,
            "export_presets": len(preset_list),
        }
        return {
            "ok": True,
            "root": str(root),
            "project_file": str(root / "project.godot"),
            "name": settings.get("name", root.name),
            "main_scene": settings.get("main_scene", ""),
            "config_version": settings.get("config_version", 0),
            "features": settings.get("features", []),
            "rendering_method": settings.get("rendering_method", ""),
            "autoload": settings.get("autoload", {}),
            "input_actions": settings.get("input_actions", []),
            "scenes": scene_list,
            "scripts": script_list,
            "resources": resources,
            "class_names": scripts.get("classes", []) if scripts.get("ok") else [],
            "export_presets": preset_list,
            "export_templates": {
                "installed": templates.get("installed", False),
                "dir": templates.get("dir", ""),
            },
            "counts": counts,
            "note": ("{scenes} sahne, {scripts} betik, {resources} kaynak, "
                     "{autoload} autoload, {input_actions} girdi eylemi, "
                     "{export_presets} dışa aktarma preseti").format(**counts),
        }

    # ─── Sağlık ─────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """
        Modülün ve motorun durumunu bildirir.

        Ajanın "Godot'yu kullanabilir miyim?" sorusunu yanıtlar: yürütülebilir
        dosya bulundu mu, sürüm ne, çalışma alanı bir Godot projesi mi ve
        eksik önkoşul (dışa aktarma şablonları) var mı.

        Returns:
            `{"ok", "godot_found", "executable", "version", "is_godot_project",
             "export_templates", "missing", "hints", "stats"}`
        """
        console, gui, source = self._find_godot()
        found = bool(console or gui)
        version = self.version() if found else ""
        is_project = self.is_godot_project()
        settings: Dict[str, Any] = {}
        if is_project:
            settings = self.read_project_settings()
        templates = self.export_templates() if found else {"installed": False, "dir": ""}

        missing: List[str] = []
        advisories: List[str] = []
        hints: List[str] = []
        if not found:
            missing.append("godot_calistirilabilir")
            hints.append(
                "Godot bulunamadı; `NEUROCORE_GODOT` ortam değişkenini "
                "`_console.exe` dosyasına işaret ettirin")
        if found and not console:
            advisories.append("godot_console_exe")
            hints.append(
                "Yalnızca GUI sürümü bulundu. GUI ikilisi de stdout/stderr'i "
                "Python'a iletir; `_console.exe` yalnızca etkileşimli terminal "
                "için gerekir, ama varsa tercih edilir")
        if not is_project:
            missing.append("godot_projesi")
            hints.append("Çalışma alanı bir Godot projesi değil — godot.create_project")
        if found and not templates.get("installed"):
            advisories.append("disa_aktarma_sablonlari")
            hints.append(str(templates.get("hint", "")))

        return {
            "ok": found and is_project,
            "module": self.name,
            "initialized": self._initialized,
            "godot_found": found,
            "executable": console or gui,
            "executable_console": console,
            "executable_gui": gui,
            "executable_source": source,
            "version": version,
            "workspace": str(self._workspace or ""),
            "is_godot_project": is_project,
            "project_name": settings.get("name", "") if settings else "",
            "main_scene": settings.get("main_scene", "") if settings else "",
            "export_templates": templates,
            "missing": missing,
            "advisories": advisories,
            "hints": [item for item in hints if item],
            "capabilities": {
                "script": found and not self._script_unsupported,
                "run": found and is_project,
                "export": found and is_project and bool(templates.get("installed")),
            },
            "stats": self.get_stats(),
        }

    # ─── Araç Tanımları ─────────────────────────────────────────────────

    def tools(self) -> List[ToolDef]:
        """
        Godot araçlarının tam listesi (`ToolRegistry`'ye kaydedilmek üzere).

        Her araç `ToolContext` alır; böylece iptal bayrağı çağrı anında
        geçerli olur ve kullanıcı uzun bir derlemeyi durdurabilir.
        """
        controller = self

        def wrap(name: str, description: str, safety: Safety,
                 parameters: List[ToolParameter], returns: str = "",
                 tags: Optional[List[str]] = None, timeout: Optional[float] = None):
            """Bağlam alan bir araç sarmalayıcısı üretir."""
            method = getattr(controller, name.split(".")[-1], None)

            def handler(ctx: ToolContext, **kwargs: Any) -> Any:
                if ctx.cancelled():
                    raise GodotError("işlem kullanıcı tarafından iptal edildi")
                if method is None:
                    raise GodotError(f"metot bulunamadı: {name}")
                return method(**kwargs)

            return ToolDef(
                name=name, description=description, handler=handler,
                parameters=parameters, safety=safety, category="godot",
                returns=returns, tags=tags or ["godot", "oyun"], timeout=timeout,
            )

        P = ToolParameter
        path_param = P("path", description="Proje dizini (boş = çalışma alanı)")

        return [
            wrap("godot.health",
                 "Godot motorunun durumunu bildirir: yürütülebilir dosya, sürüm, "
                 "çalışma alanının Godot projesi olup olmadığı ve eksik önkoşullar. "
                 "Godot işine başlamadan ÖNCE çağırın.",
                 Safety.READ, [], "motor durumu", ["durum", "teshis"]),

            wrap("godot.project_map",
                 "Projenin haritasını çıkarır: sahneler (kök düğüm türleriyle), "
                 "betikler (extends/class_name), kaynaklar, autoload girdileri, "
                 "girdi eylemleri ve dışa aktarma presetleri. Yeni bir Godot "
                 "projesinde İLK çağrılacak araçtır.",
                 Safety.READ, [path_param], "proje haritası", ["kesif", "harita"]),

            wrap("godot.create_project",
                 "Yeni bir Godot 4 projesi oluşturur: project.godot, Main.tscn ve "
                 "main.gd üretir, ardından içe aktarır. Şablonlar: 3d (kamera + "
                 "ışık + kutu), 2d (kamera + renk dikdörtgeni), empty.",
                 Safety.WRITE,
                 [P("path", required=True, description="Proje dizini"),
                  P("name", description="Proje adı (boş = dizin adı)"),
                  P("template", description="3d | 2d | empty", default="3d")],
                 "oluşturma sonucu", ["proje", "olustur"]),

            wrap("godot.import_project",
                 "Projeyi içe aktarır (kaynak taraması, .godot önbelleği). Yeni "
                 "dosya eklendikten veya class_name değiştikten sonra çağırın.",
                 Safety.EXECUTE,
                 [path_param, P("timeout", "number", "Saniye", default=300)],
                 "içe aktarma sonucu", ["import"]),

            wrap("godot.run_script",
                 "Proje içinde bir GDScript betiği başsız çalıştırır ve çıktısını "
                 "döner. `extends SceneTree` + `func _init()` ile başlamalı ve "
                 "sonunda `quit(0)` çağırmalıdır. Çalışma zamanı hataları exit 0 "
                 "verse bile yakalanır; `--quit-after` asılı kalmayı önler.",
                 Safety.EXECUTE,
                 [P("code", required=True, description="GDScript kaynağı"),
                  path_param,
                  P("timeout", "number", "Saniye", default=120),
                  P("script_name", description="Dosya adı (uzantısız)",
                    default="nc_tool"),
                  P("quit_after", "integer", "Güvenlik ağı iterasyon sayısı",
                    default=500)],
                 "betik çıktısı ve sonuç", ["gdscript", "calistir"]),

            wrap("godot.create_scene",
                 "Sahne oluşturur (varsa üzerine ekler). parent alanı boş olan ilk "
                 "düğüm kök olur; diğerleri parent ile belirtilen düğümün altına "
                 "girer. Özelliklerde dize değerler türe çevrilir.",
                 Safety.WRITE,
                 [P("scene_path", required=True, description="res://... .tscn"),
                  P("nodes", "array", "Düğüm tanımları (type/name/parent/properties)",
                    required=True),
                  path_param],
                 "sahne sonucu", ["sahne", "olustur"]),

            wrap("godot.add_node",
                 "Var olan sahneye düğüm ekler (ClassDB.instantiate ile üretilir; "
                 "sınıf yoksa net hata döner).",
                 Safety.WRITE,
                 [P("scene_path", required=True, description="res://... .tscn"),
                  P("node_type", required=True, description="Godot 4 sınıfı, ör. Camera3D"),
                  P("node_name", description="Düğüm adı"),
                  P("parent", description="Üst düğüm yolu (boş = kök)"),
                  P("properties", "object", "Başlangıç özellikleri"),
                  path_param],
                 "ekleme sonucu", ["sahne", "dugum"]),

            wrap("godot.set_node_property",
                 "Sahnedeki bir düğümün özelliğini ayarlar ve değeri geri okuyarak "
                 "doğrular. Dizeler türe çevrilir (sayı, bool, Vector3 'x,y,z', "
                 "Color, res:// kaynağı).",
                 Safety.WRITE,
                 [P("scene_path", required=True, description="res://... .tscn"),
                  P("node_path", required=True, description="Düğüm yolu ('.' = kök)"),
                  P("property_name", required=True, description="Özellik adı, ör. fov"),
                  P("value", required=True, description="Değer"),
                  path_param],
                 "atama sonucu", ["sahne", "ozellik"]),

            wrap("godot.remove_node",
                 "Sahneden düğümü ve alt ağacını siler (kök düğüm silinemez).",
                 Safety.DESTRUCTIVE,
                 [P("scene_path", required=True, description="res://... .tscn"),
                  P("node_path", required=True, description="Silinecek düğüm yolu"),
                  path_param],
                 "silme sonucu", ["sahne", "dugum", "sil"]),

            wrap("godot.inspect_scene",
                 "Sahneyi motorun içinde açar ve düğüm ağacını okur: adlar, "
                 "sınıflar, yollar, betikler ve seçilmiş özellikler. Metin "
                 "ayrıştırmaz, gerçek PackedScene örnekler — sonuç kesindir.",
                 Safety.READ,
                 [P("scene", required=True, description="res://... .tscn"),
                  path_param],
                 "düğüm ağacı", ["sahne", "incele"]),

            wrap("godot.list_scenes",
                 "Projedeki sahneleri kök düğüm türü ve düğüm sayısıyla listeler.",
                 Safety.READ, [path_param], "sahne listesi", ["kesif"]),

            wrap("godot.list_scripts",
                 "Projedeki GDScript dosyalarını extends/class_name bilgisiyle listeler.",
                 Safety.READ, [path_param], "betik listesi", ["kesif"]),

            wrap("godot.read_project_settings",
                 "project.godot ayarlarını okur: ad, ana sahne, autoload, girdi "
                 "eylemleri, motor özellikleri.",
                 Safety.READ, [path_param], "proje ayarları", ["ayar"]),

            wrap("godot.list_export_presets",
                 "Dışa aktarma presetlerini listeler. Gerekirse motoru geçersiz "
                 "bir adla sondalayıp motorun kabul ettiği listeyi de döner "
                 "(elle yazılmış export_presets.cfg reddedilebilir).",
                 Safety.READ,
                 [path_param,
                  P("probe", "boolean", "Motoru da sorgula", default=False)],
                 "preset listesi", ["export"]),

            wrap("godot.export_project",
                 "Projeyi bir preset ile derler. Önce export_presets.cfg, preset "
                 "adı ve dışa aktarma şablonlarının varlığını denetler, hedef "
                 "dizini oluşturur. Sürüm derlemesinden önce --import çalıştırır "
                 "(--export-release bunu kendisi yapmaz). DİKKAT: Godot dışa "
                 "aktarmada GDScript hataları varken de exit 0 verir; sonuç "
                 "dosya varlığı ve hata işaretleriyle doğrulanır.",
                 Safety.EXECUTE,
                 [P("preset", description="Preset adı", default="Windows Desktop"),
                  P("output", description="Çıktı yolu (boş = preset değeri)"),
                  path_param,
                  P("debug", "boolean", "Debug derlemesi", default=False),
                  P("timeout", "number", "Saniye", default=900),
                  P("import_first", "boolean", "Önce --import çalıştır",
                    default=True)],
                 "dışa aktarma sonucu", ["export", "derle"]),

            wrap("godot.run_project",
                 "Projeyi (veya bir sahneyi) başsız çalıştırır ve `--quit-after` "
                 "ile ~180 kare sonra kapatır — açılış hatalarını yakalamak ve "
                 "asılmayı önlemek için duman testi.",
                 Safety.EXECUTE,
                 [path_param,
                  P("timeout", "number", "Saniye", default=60),
                  P("scene", description="res://... (boş = ana sahne)"),
                  P("headless", "boolean", "Görsel pencere açma", default=True),
                  P("quit_after", "integer", "Kaç kare sonra kapanacağı",
                    default=180)],
                 "çalıştırma çıktısı", ["calistir", "test"]),

            wrap("godot.run_scene",
                 "Tek bir sahneyi başsız çalıştırır (açılışta hata veriyor mu?).",
                 Safety.EXECUTE,
                 [P("scene", required=True, description="res://... .tscn"),
                  path_param,
                  P("timeout", "number", "Saniye", default=60),
                  P("quit_after", "integer", "Kaç kare sonra kapanacağı",
                    default=180)],
                 "çalıştırma çıktısı", ["calistir", "test"]),

            wrap("godot.error_log",
                 "Godot'un günlük dosyasını okur (app_userdata ve proje dizini "
                 "denetlenir, en yenisi döner).",
                 Safety.READ,
                 [path_param, P("lines", "integer", "Sondan satır sayısı", default=80)],
                 "günlük son satırları", ["gunluk", "teshis"]),

            wrap("godot.diagnose",
                 "Godot hata çıktısını yorumlar: tür, dosya/satır ve Türkçe öneri. "
                 "Parse Error, SCRIPT ERROR, Node not found, Invalid call, "
                 "Condition is true, eksik şablon gibi kalıpları kapsar.",
                 Safety.READ,
                 [P("text", required=True, description="Motor çıktısı"),
                  P("command", description="Çalıştırılan komut (bağlam)")],
                 "hata analizi", ["teshis", "hata"]),
        ]

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<GodotController workspace={self._workspace} "
                f"exe={self.executable() or 'yok'} "
                f"scripts={self._counter['scripts_run']}>")


# ─── Modül Düzeyi Yardımcılar ───────────────────────────────────────────────

def parse_godot_errors(text: str) -> List[Dict[str, Any]]:
    """
    Godot çıktısındaki hataları (motor gürültüsünü eleyerek) çıkarır.

    Modül düzeyinde de kullanılabilir; `GodotController.diagnose` ile aynı
    kalıp tablosunu paylaşır.

    Args:
        text: Motor çıktısı

    Returns:
        `[{"kind", "message", "hint", "file"?, "line"?}]`
    """
    return GodotController._collect_errors(text)


__all__ = [
    "GodotController",
    "GodotError",
    "parse_godot_errors",
    "GODOT_ERROR_HINTS",
    "GD_NOISE_PATTERNS",
    "GODOT_ENV_VARS",
    "NC_TOOLS_DIR",
    "NC_RESULT_PREFIX",
]
