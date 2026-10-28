"""
NeuroCore Blender 3B Kontrol Modülü
====================================
Ajanın **Blender'ı başsız (headless) sürmesini** sağlayan katman: sahne kurar,
betik çalıştırır, render alır, sahneyi inceler ve dışa aktarır.

Neden ayrı bir katman?
    `bpy` API'si **yalnızca Blender'ın kendi Python yorumlayıcısının** içinden
    erişilebilir; sistem Python'undan `import bpy` çalışmaz. Bu yüzden her işlem
    şu zincire indirgenir::

        ajan kodu  →  geçici .py  →  blender --background ... --python <betik>
                   →  NC_RESULT / NC_ERROR işaret satırları  →  ayrıştır

Blender'ın iki tehlikeli davranışı burada kapatılır
---------------------------------------------------
1. **İstisnayı yutar.** Bu makinede doğrulandı: `--python` ile çalışan bir
   betikte yakalanmamış istisna oluştuğunda Blender izleme kaydını basar,
   "Blender quit" yazar ve **çıkış kodu 0** döner. Ajan "başarılı" sanıp
   yanlış karar verir. Bu modül **iki katmanlı** koruma uygular:

   * `--python-exit-code 1` — **`--python`'dan ÖNCE** gelmelidir; sonra
     verilirse hiçbir etkisi olmaz (doğrulandı: kod 1 → çıkış 1, kod sonra → 0).
   * Üretilen her betiğin gövdesi zorunlu ``try/except`` ile sarılır; hata
     hâlinde ``NC_ERROR <json>`` basılır ve ``sys.exit(1)`` çağrılır.

2. **GUI bağlamı ister.** `bpy.ops` operatörlerinin çoğu bir `area`/`region`
   bekler ve başsız modda `poll() failed` verir. Bu modül veri API'sini
   (`bpy.data.*` + `collection.objects.link`) tercih eder; operatör
   kaçınılmazsa (render, dışa aktarma, fabrika sıfırlaması) bağlam **bilinçli**
   kurulur.

Zaman sınırı neden Blender'ın içinden uygulanır?
------------------------------------------------
Windows'ta alt süreç zinciri `python → powershell → blender.exe` şeklindedir.
Takılan bir Blender süreci boruyu (pipe) açık tuttuğu için
`subprocess.run(..., timeout=N)` çağrısı N saniyede dönmez; `TimeoutExpired`
ancak torun süreç boruyu bıraktığında görülür. Bu makinede **ölçüldü**: 8
saniyelik sınır 59.6 saniyede doldu; sonsuz döngüye giren bir betikte ise ajan
hiç dönmedi. Bu yüzden üretilen her betiğe bir **watchdog iş parçacığı** gömülür
ve süre dolduğunda `os._exit(1)` ile Blender **kendini** sonlandırır (bu çağrı
ana iş parçacığı meşgul olsa bile çalışır). Kabuk katmanının zaman aşımı
yalnızca Blender Python'unun hiç başlatılamadığı durumlar için yedektir.

Argüman sırası yük taşır
------------------------
Blender argümanları **verildikleri sırada** işler; `.blend` dosyası göründüğü
noktada yüklenir ve yüklenirken o ana kadarki render ayarlarını **ezer**. İki
klasik tuzak:

* Dosyayı `-o`/`-F`/`-f`'den **sonra** vermek → eski dosya yolu kullanılır
  (``blender -b -f 1 -o /tmp/x file.blend`` yanlış yola render eder).
* Yeni sahne kurarken dosya argümanı vermek → boş sahne yerine dosya yüklenir
  ve kurulum üstüne yazılır.

Bu modül bu yüzden `.blend` yolunu **her zaman** `--python`'dan önce ve
`run_script` içinde tek bir yerde yerleştirir; sahne sıfırdan kuruluyorsa
dosya argümanı **hiç verilmez**. `-o`/`-F`/`-f` bayrakları hiç kullanılmaz:
çıktı yolu betik içinde `scene.render.filepath` ile sabitlenir.

Motor seçimi ve makine nezaketi
-------------------------------
Varsayılan **Cycles + CPU**'dur (`-E CYCLES`, `cycles.device = "CPU"`). Ölçüm
(bu makine, 64x64 / 1 örnek): Cycles CPU **0.22 s**, EEVEE başsız **10.92 s**
(GPU bağlamı kurulumu + gölgelendirici derlemesi). EEVEE burada çalışıyor
çünkü GPU var; ancak GPU'su olmayan, sanal makinede veya yalnızca RDP ile
erişilen bir makinede EEVEE **kırılgandır**. Bu yüzden varsayılan CPU'dur ve
motor `engine` parametresiyle değiştirilebilir.

Ayrıca iş parçacığı sayısı `threads` ile sınırlanabilir (`-t/--threads`); bu
projenin kuralı gereği ajan kullanıcının bilgisayarını doyuracak kadar yük
bindirmemelidir. `threads=0` (varsayılan) Blender'ın kendi seçimini bırakır;
kalabalık bir makinede `threads=2` gibi bir değer makineyi duyarlı tutar.

Sürüm notları (Blender 5.2.2, bu makinede doğrulandı)
-----------------------------------------------------
* Motor kimliği **`BLENDER_EEVEE`**'dir; `BLENDER_EEVEE_NEXT` 5.0'da geri
  alındı ve atanırsa ``TypeError: enum "BLENDER_EEVEE_NEXT" not found`` verir.
  `bl_rna.properties["engine"].enum_items` bu yapıda yalnızca
  `BLENDER_EEVEE` gösterir (eklenti motorları `CYCLES`/`BLENDER_WORKBENCH`
  listede görünmez ama **atanabilir**). Bu yüzden liste keşfedilir, atama
  denenir ve başarısızlık açıkça bildirilir — sessizce yanlış motorda render
  alınmaz.
* `Material.use_nodes` / `World.use_nodes` **kullanımdan kaldırıldı**
  ("expected to be removed in Blender 6.0"); 5.2'de zaten `True` ve düğüm
  ağacı `bpy.data.materials.new()` ile hazır geliyor. Bu modül bu bayrağı
  **okumaz ve yazmaz** — uyarı yalnızca niteliğe dokunulduğunda çıkar; yalnızca
  eski sürümler için (varsa ve `False` ise) uyarı bastırılarak yazılır.
* `scene.node_tree` **kaldırıldı** → `scene.compositing_node_group`.
* `image_settings` için **`media_type` önce**, `file_format` sonra
  ayarlanmalıdır (`IMAGE` → `PNG`).
* Principled BSDF giriş adları değişti: `Specular` → `Specular IOR Level`,
  `Subsurface` → `Subsurface Weight`, `Transmission` → `Transmission Weight`,
  `Clearcoat` → `Coat Weight`, `Emission` → `Emission Color`. `Emission` ve
  `Specular` adlı giriş **yoktur**. Bu modül dostu anahtarları eşler ve
  bulunamayan girişi **mevcut girişleri listeleyen** bir hatayla bildirir.
* `blender-launcher.exe` **kullanılmaz** (gerçek süreci ayrık başlatıp hemen
  döner; stdout ve anlamlı çıkış kodu vermez). Yalnızca `blender.exe`.

Bu modül `neurocore.agent.tools` ve `neurocore.utils` dışında hiçbir şeye bağlı
değildir; kabuk katmanı **enjekte edilir** (`ShellIntegration`), böylece komut
çalıştırma tek bir yerde denetlenir ve test edilebilir kalır.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neurocore.agent.tools import Safety, ToolContext, ToolDef, ToolParameter
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.blender")


# ─── Sabitler ───────────────────────────────────────────────────────────────

#: Bu makinede doğrulanmış Blender yolu (PATH'te DEĞİL)
DEFAULT_EXECUTABLE = r"E:\AppsTools\Blender 5.2\blender.exe"

#: Kurulumun aranacağı kökler (tüm diski taramak yerine bilinen yerler)
SEARCH_ROOTS: Tuple[str, ...] = (
    r"E:\AppsTools", r"D:\AppsTools", r"C:\AppsTools",
    r"C:\Program Files", r"C:\Program Files (x86)",
    str(Path.home() / "Apps"), str(Path.home() / "Downloads"),
)

#: Üretilen betiklerin işaret satırları
RESULT_MARKER = "NC_RESULT"
ERROR_MARKER = "NC_ERROR"

#: Alt sürece aktarılan ortam değişkenleri (yollar koda gömülmesin)
ENV_OUT_DIR = "NC_BLEND_OUT"
ENV_BLEND_FILE = "NC_BLEND_FILE"

#: Varsayılan render ayarları — CPU'da güvenli, ekran/GPU bağlamı gerektirmez
DEFAULT_ENGINE = "CYCLES"
DEFAULT_DEVICE = "CPU"
DEFAULT_SAMPLES = 32
DEFAULT_RESOLUTION: Tuple[int, int] = (960, 540)

#: Blender 5.x'te atanabilir motor kimlikleri (enum'da görünmeseler bile).
#: Sıra, geri düşme (fallback) sırasıdır: önce Cycles, sonra EEVEE.
KNOWN_ENGINES: Tuple[str, ...] = ("CYCLES", "BLENDER_EEVEE", "BLENDER_WORKBENCH")

#: Zaman aşımları — asılı Blender ajanı kilitlememeli
TIMEOUT_VERSION = 60.0
TIMEOUT_SCRIPT = 600.0
TIMEOUT_EXPRESSION = 300.0
TIMEOUT_SCENE = 900.0
TIMEOUT_INSPECT = 300.0
TIMEOUT_RENDER = 1800.0
TIMEOUT_EXPORT = 600.0

#: Kabuk zaman aşımına eklenen pay. Alt süreç önce kendi watchdog'u ile
#: sonlanmalı; kabuk zaman aşımı yalnızca **yedek** güvenlik ağıdır.
#:
#: ÖLÇÜLEN TUZAK (Windows): takılan bir Blender süreci boruyu (pipe) açık
#: tuttuğu için `subprocess.run(..., timeout=N)` N saniyede dönmez —
#: `TimeoutExpired` ancak torun süreç boruyu bıraktığında görülür (bu makinede
#: 8 s'lik sınır 59.6 s'de doldu). Bu yüzden zaman sınırı Blender'ın İÇİNDEN,
#: üretilen betiğe gömülen watchdog iş parçacığıyla uygulanır; kabuk zaman
#: aşımı yalnızca Blender Python'u hiç başlatamadığı durum için yedektir.
SHELL_TIMEOUT_MARGIN = 60.0

#: Yakalanan çıktının üst sınırı (ajan istemini şişirmemek için)
MAX_OUTPUT_CHARS = 20000

#: Blender'ın izole Python'unda bulunup bulunmadığı sınanan modüller
PROBE_MODULES: Tuple[str, ...] = (
    "numpy", "PIL", "requests", "scipy", "yaml", "cv2", "matplotlib",
)

#: Dostu giriş adı → Principled BSDF giriş adı (sürümler arasında değişti).
#: Değer bir demettir: ilk bulunan ad kullanılır (yeni → eski sırası).
SOCKET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "base color": ("Base Color", "Color"),
    "color": ("Base Color", "Color"),
    "metallic": ("Metallic",),
    "roughness": ("Roughness",),
    "specular": ("Specular IOR Level", "Specular"),
    "specular ior level": ("Specular IOR Level", "Specular"),
    "specular tint": ("Specular Tint",),
    "ior": ("IOR",),
    "alpha": ("Alpha",),
    "transmission": ("Transmission Weight", "Transmission"),
    "transmission weight": ("Transmission Weight", "Transmission"),
    "subsurface": ("Subsurface Weight", "Subsurface"),
    "subsurface weight": ("Subsurface Weight", "Subsurface"),
    "subsurface radius": ("Subsurface Radius",),
    "coat": ("Coat Weight", "Clearcoat"),
    "coat weight": ("Coat Weight", "Clearcoat"),
    "clearcoat": ("Coat Weight", "Clearcoat"),
    "coat roughness": ("Coat Roughness", "Clearcoat Roughness"),
    "clearcoat roughness": ("Coat Roughness", "Clearcoat Roughness"),
    "coat normal": ("Coat Normal", "Clearcoat Normal"),
    "sheen": ("Sheen Weight", "Sheen"),
    "sheen weight": ("Sheen Weight", "Sheen"),
    "sheen roughness": ("Sheen Roughness",),
    "emission": ("Emission Color", "Emission"),
    "emission color": ("Emission Color", "Emission"),
    "emission strength": ("Emission Strength",),
    "anisotropic": ("Anisotropic",),
    "normal": ("Normal",),
    "weight": ("Weight",),
    "thin wall": ("Thin Wall",),
    "diffuse roughness": ("Diffuse Roughness",),
}


class BlenderError(Exception):
    """Blender aracı hatası."""


# ─── Teşhis Kalıpları ───────────────────────────────────────────────────────

#: Ham Blender/`bpy` çıktısı → (tür, ipucu). Sıra önemlidir: özel → genel.
DIAGNOSE_RULES: List[Tuple[str, str, str]] = [
    (r"Operator bpy\.ops\.[\w.]+\.poll\(\) failed|"
     r"poll\(\) failed, context is incorrect",
     "poll_failed",
     "Operatör bu bağlamda çalışmıyor: başsız modda `area`/`region` yoktur. "
     "Veri API'sini kullanın (bpy.data.meshes.new + bpy.data.objects.new + "
     "collection.objects.link) veya operatörden önce bağlamı kurun: "
     "bpy.context.view_layer.objects.active = obj ve obj.select_set(True)"),
    (r"enum \"([A-Z_]+)\" not found in|enum \"([A-Z_]+)\" not found",
     "enum_invalid",
     "Geçersiz enum değeri (ör. BLENDER_EEVEE_NEXT 5.0'da geri alındı; doğrusu "
     "BLENDER_EEVEE). Motor listesi bl_rna'da eksik görünebilir: "
     "blender.health()['engines'] ile bu yapıda gerçekten atanabilen motorları "
     "görün ve sabit kodlamayın"),
    (r"'bpy\.ops\.[\w.]*' object has no attribute|"
     r"AttributeError:.*bpy\.ops|"
     r"AttributeError:.*has no attribute '([\w.]+)'",
     "ops_attribute",
     "Operatör bu sürümde yok veya adı değişti. Önce varlığını denetleyin: "
     "print([n for n in dir(bpy.ops.import_scene) if 'x' in n.lower()]); "
     "ayrıca FBX'te `export_selected` YOK (doğrusu `use_selection`), "
     "`use_animations` YOK (doğrusu `bake_anim`)"),
    (r"KeyError:\s*['\"]?([^'\"\n]{1,60})['\"]?",
     "socket_key_error",
     "Düğüm girişi (socket) adı sürümler arasında değişti: `Specular` → "
     "`Specular IOR Level`, `Subsurface` → `Subsurface Weight`, `Transmission` "
     "→ `Transmission Weight`, `Clearcoat` → `Coat Weight`, `Emission` → "
     "`Emission Color`. 5.2'de `Emission` veya `Specular` adlı giriş YOKTUR. "
     "Adı sabit kodlamayın: print([s.name for s in node.inputs]) ile "
     "numaralandırın; blender.build_scene malzeme tanımı bu eşlemeyi yapar ve "
     "bulunamayan girişi mevcut girişlerle birlikte bildirir"),
    (r"ModuleNotFoundError:\s*No module named ['\"]([^'\"]+)['\"]",
     "blender_module_missing",
     "Blender KENDİ izole Python'unu çalıştırır (5.2'de Python 3.13, "
     "…\\5.2\\python\\bin\\python.exe). Sistem Python'unuzun site-packages "
     "dizini Blender'dan GÖRÜNMEZ: `pip install` sistem Python'una kurar, "
     "Blender bulamaz. Önce blender.python_environment() ile hazır olanları "
     "görün (numpy ve requests gelir; PIL, scipy, yaml, cv2 GELMEZ). Eksik "
     "paketi Blender'ın kendi python'u ile kurun: "
     "& '<blender>\\5.2\\python\\bin\\python.exe' -m pip install <paket>"),
    (r"ImportError:\s*cannot import name ['\"]([^'\"]+)['\"]",
     "import_error",
     "Ad veya modül yolu bu sürümde değişmiş olabilir; `dir(modul)` ile gerçek "
     "adları listeleyin. Blender 4.x/5.x'te birçok bpy sınıfı taşındı"),
    (r"Cannot read file ['\"]?([^'\"\n]+)|Cannot open file ['\"]?([^'\"\n]+)|"
     r"FileNotFoundError:\s*\[[^\]]*\]\s*['\"]?([^'\"\n]+)",
     "file_missing",
     "Dosya bulunamadı. Yolu MUTLAK verin (bu modül asla `cd` yapmaz) ve "
     "varlığını önceden doğrulayın (.blend için blender.inspect_blend)"),
    (r"Python script failed|IndentationError|SyntaxError:|TabError:|"
     r"Cannot execute script file",
     "script_syntax",
     "Betiğin kendisi çalıştırılamadı (sözdizimi/girinti). Blender'ı hiç "
     "başlatmadan blender.validate_script ile sistem Python'unda `ast` denetimi "
     "yapın — saniyeler kazandırır"),
    (r"CUDA error|CUDA kernel compilation failed|OptiX error|HIP error|"
     r"oneAPI error|No CUDA-capable device|"
     r"GPU (?:device )?(?:not found|is not available)",
     "gpu_error",
     "GPU/CUDA yolu başarısız. Başsız makinede güvenli seçim CPU'dur: "
     "blender.render(..., device='CPU') (varsayılan) veya "
     "engine='CYCLES' + device='CPU'. GPU'su olmayan/VM/RDP makinede EEVEE de "
     "kırılgandır: `-E CYCLES` kullanın"),
    (r"Out of memory|OutOfMemory|MemoryError|Cannot allocate memory|"
     r"std::bad_alloc|bad_alloc",
     "out_of_memory",
     "Bellek yetersiz. Çözünürlüğü ve örnek sayısını düşürün "
     "(render(samples=16, resolution=(480,270))), alt bölme/parçacık "
     "seviyelerini azaltın veya sahneyi parçalara bölün"),
    (r"Segmentation fault|segfault|SIGSEGV|access violation|"
     r"Blender.*has stopped working",
     "segfault",
     "Blender çöktü. Tipik nedenler: kararsız GPU sürücüsü, bozuk .blend, "
     "yerel (C) eklenti. CPU'ya düşün, --factory-startup kullanın (varsayılan) "
     "ve son çalışan adımı geri alarak yeniden deneyin"),
    (r"Permission denied|Access is denied|WinError 5|Errno 13",
     "permission",
     "Yazma/okuma izni yok. Çıktıyı çalışma alanı içine verin ve dosyanın "
     "başka bir Blender süreci tarafından açık olmadığından emin olun"),
    (r"DeprecationWarning:\s*'([\w.]+)' is expected to be removed",
     "deprecation",
     "UYARI — hata değil, süreç başarılı sayılır. Blender 5.2'de "
     "Material.use_nodes / World.use_nodes için bu uyarı çıkar (6.0'da "
     "kaldırılacak). Düğüm ağacı artık her zaman vardır: bayrağa dokunmadan "
     "material.node_tree üzerinden ilerleyin"),
    (r"RuntimeError:\s*Error:\s*([^\n]{1,140})",
     "runtime",
     "Blender çalışma zamanı hatası; mesajdaki operatör/veri yolunu kontrol "
     "edin ve bağlamı (aktif nesne, seçim, mod) açıkça kurun"),
]

_COMPILED_RULES = [
    (re.compile(pattern, re.IGNORECASE | re.MULTILINE), kind, hint)
    for pattern, kind, hint in DIAGNOSE_RULES
]

#: İzleme kaydı göstergeleri — `NC_ERROR` olmasa bile hata sayılır
TRACEBACK_MARKERS: Tuple[str, ...] = (
    "Traceback (most recent call last)",
    "Python script failed",
    "Error: Python: Traceback",
)

#: Betik gövdesine enjekte edilen yardımcılar (üretilen her betikte bulunur)
_SCRIPT_HELPERS = '''
def nc_input(node, *names):
    """Giriş adlarını sırayla dener; hiçbiri yoksa None döner."""
    for candidate in names:
        socket = node.inputs.get(candidate)
        if socket is not None:
            return socket
    return None


def nc_sockets(node):
    """Düğümün giriş adlarını döndürür (hata mesajlarında kullanılır)."""
    return [item.name for item in node.inputs]


def nc_available_engines():
    """bl_rna'da görünen motorlar + atanabilir olduğu bilinenler."""
    try:
        listed = [item.identifier for item in
                  bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items]
    except Exception:
        listed = []
    for candidate in {KNOWN!r}:
        if candidate not in listed:
            listed.append(candidate)
    return listed


def nc_use_nodes(block):
    """Eski Blender sürümlerinde düğüm ağacını açar (5.x'te gerekmez).

    Blender 5.2'de Material.use_nodes/World.use_nodes kullanımdan kaldırıldı
    (6.0'da kaldırılacak) ve zaten True'dur; bu yüzden yalnızca var olup
    False ise ve yalnızca uyarı bastırılarak yazılır.
    """
    try:
        if hasattr(block, 'use_nodes') and not block.use_nodes:
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore', DeprecationWarning)
                block.use_nodes = True
    except Exception:
        pass


def nc_set_image_format(image_settings, file_format):
    """5.x'te media_type ÖNCE ayarlanmalıdır, sonra file_format."""
    try:
        if hasattr(image_settings, 'media_type'):
            image_settings.media_type = 'IMAGE'
    except Exception:
        pass
    image_settings.file_format = file_format
'''

#: Zorunlu hata sarmalayıcısı — Blender istisnayı yutsa bile hata görünür olur
_WRAPPER_HEADER = '''# -*- coding: utf-8 -*-
# NeuroCore tarafından üretildi. Elle düzenlemeyin.
# ZORUNLU try/except: Blender yakalanmamış istisnada "Blender quit" yazıp
# ÇIKIŞ KODU 0 döndürebilir; ajan yanlışlıkla başarılı sanmasın diye hata
# NC_ERROR olarak basılır ve sys.exit(1) çağrılır. `--python-exit-code 1`
# ikinci güvenlik katmanıdır.
import json
import os
import sys
import threading
import time
import traceback

NC_OUT_DIR = os.environ.get("{out_env}") or {out_repr}
NC_BLEND_IN = os.environ.get("{blend_env}", "")
NC_STARTED = time.time()
NC_COLLECT = {collect!r}
NC_BUDGET = {budget!r}

if NC_OUT_DIR:
    try:
        os.makedirs(NC_OUT_DIR, exist_ok=True)
    except OSError:
        pass

import bpy  # Blender içinde her zaman hazırdır


def nc_watchdog():
    """Katı zaman sınırı: betik takılırsa Blender KENDİ KENDİNİ sonlandırır.

    Neden gerekli? Kabuk katmanı (`subprocess.run(..., timeout=)`) süresi
    dolduğunda yalnızca PowerShell'i öldürür ve `wait()` içinde bloke olur;
    `blender.exe` torun süreç olarak yaşamaya devam eder, boruyu (pipe) açık
    tutar ve ajan da asılı kalır. Sonsuz döngüyü `os._exit` ile kesmek
    mümkündür: bu çağrı Python yığınını temizlemez ve ana iş parçacığı meşgul
    olsa bile süreci anında sonlandırır. Bu yüzden zaman aşımı Blender'ın
    İÇİNDEN uygulanır; kabuk zaman aşımı yalnızca yedek güvenlik ağıdır.
    """
    time.sleep(NC_BUDGET)
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        print("{error_marker} " + json.dumps({{
            "type": "TimeoutError",
            "message": "betik " + str(NC_BUDGET) + " saniyede bitmedi; "
                       "watchdog Blender'i sonlandirdi",
        }}))
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(1)


if NC_BUDGET > 0:
    _watchdog = threading.Thread(target=nc_watchdog, name="nc_watchdog", daemon=True)
    _watchdog.start()


def nc_artifacts():
    """Bu çalıştırmada oluşan/değişen dosyaları NC_ARTIFACTS olarak basar."""
    if not NC_COLLECT or not NC_OUT_DIR or not os.path.isdir(NC_OUT_DIR):
        return
    items = []
    try:
        for name in sorted(os.listdir(NC_OUT_DIR)):
            path = os.path.join(NC_OUT_DIR, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            if os.path.isfile(path) and stat.st_mtime >= NC_STARTED - 1.0:
                items.append({{"name": name, "path": path, "size": stat.st_size}})
    except OSError:
        return
    if items:
        print("NC_ARTIFACTS " + json.dumps(items))
        sys.stdout.flush()

{helpers}

try:
'''

_WRAPPER_FOOTER = '''
except SystemExit:
    raise
except BaseException as nc_exc:
    try:
        nc_artifacts()
    except Exception:
        pass
    print("{error_marker} " + json.dumps({{
        "type": type(nc_exc).__name__,
        "message": str(nc_exc)[:2000],
        "script": __file__,
        "blend": NC_BLEND_IN,
    }}, ensure_ascii=True))
    traceback.print_exc(file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(1)
else:
    try:
        nc_artifacts()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
'''


# ─── Blender Denetleyicisi ──────────────────────────────────────────────────

class BlenderController:
    """
    Blender'ı başsız süren araç katmanı.

    Kullanım:
        blender = BlenderController()
        blender.initialize(workspace="E:/PROJELER/myAi", config=config,
                           event_bus=bus, shell=shell)

        scene = blender.build_scene({
            "objects": [
                {"type": "cube", "name": "Kup", "size": 2.0,
                 "material": {"name": "Kirmizi", "base_color": [0.8, 0.2, 0.1, 1.0],
                              "metallic": 0.4, "roughness": 0.3}},
                {"type": "camera", "location": [6, -6, 4], "look_at": [0, 0, 1]},
                {"type": "sun", "energy": 3.0},
            ],
            "render": {"engine": "CYCLES", "samples": 16},
        }, out_dir="cikti", render=True)

    Tasarım kararları:
        * **Kabuk enjekte edilir** — `ShellIntegration` verilmezse özel bir
          örnek oluşturulur; komut çalıştırma mantığı burada tekrarlanmaz.
        * **Zaman sınırı Blender'ın içinden uygulanır.** Üretilen her betiğe bir
          watchdog iş parçacığı gömülür; süre dolduğunda `os._exit(1)` ile
          Blender **kendini** sonlandırır. Nedeni Windows'a özgüdür: takılan bir
          Blender süreci boruyu açık tuttuğu için kabuk katmanının
          `subprocess.run(timeout=)` çağrısı zamanında dönmez (ölçüldü: 8 s'lik
          sınır 59.6 s'de doldu) ve ajan asılı kalırdı.
        * **Nezaketli bozulma** — çalıştırılabilir yoksa hiçbir metot istisna
          atmaz, `{"ok": False, "error": ..., "hint": ...}` döner.
        * **`cd` yok** — tüm yollar mutlaktır ve alt sürece `cwd` verilir.
        * **Varsayılan `threads=0`** Blender'ın kendi seçimidir; kalabalık
          makinede `threads=2` gibi bir sınır koyun ki kullanıcı makinesi
          duyarlı kalsın.
    """

    name = "agent.blender"

    def __init__(self) -> None:
        self._workspace: Optional[Path] = None
        self._config: Any = None
        self._event_bus: Any = None
        self._shell: Any = None
        self._owns_shell = False
        self._initialized = False
        self._executable = ""
        self._version_cache = ""
        self._engine_cache: List[str] = []
        self._temp_root: Optional[Path] = None

        self._counter = {
            "scripts": 0, "expressions": 0, "scenes": 0, "renders": 0,
            "inspections": 0, "exports": 0, "validations": 0, "diagnoses": 0,
            "failures": 0, "timeouts": 0, "syntax_errors": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, workspace: Optional[str] = None, config: Any = None,
                   event_bus: Any = None, shell: Any = None) -> None:
        """
        Denetleyiciyi başlatır (idempotent).

        Args:
            workspace: Çalışma alanı kökü (None = config.base_path veya cwd)
            config: SystemConfig
            event_bus: EventBus
            shell: `ShellIntegration` örneği. Verilmezse aynı çalışma alanı için
                özel bir örnek oluşturulur ve `shutdown()` ile kapatılır.

        Raises:
            BlenderError: Çalışma alanı yoksa
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus

        root = workspace or getattr(config, "base_path", None) or os.getcwd()
        self._workspace = Path(root).expanduser().resolve()
        if not self._workspace.exists():
            raise BlenderError(f"Çalışma alanı yok: {self._workspace}")

        if shell is not None:
            self._shell = shell
            self._owns_shell = False
        else:
            from neurocore.agent.shell import ShellIntegration

            self._shell = ShellIntegration()
            self._shell.initialize(workspace=str(self._workspace), config=config,
                                   event_bus=event_bus)
            self._owns_shell = True

        self._executable = self._detect_executable()
        self._temp_root = Path(tempfile.gettempdir()) / "nc_blender"
        try:
            self._temp_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover
            logger.warning(f"Geçici dizin oluşturulamadı: {exc}")
            self._temp_root = Path(tempfile.gettempdir())

        self._initialized = True
        logger.info(
            f"Blender denetleyicisi hazır — çalışma alanı: {self._workspace}, "
            f"blender: {self._executable or '(bulunamadı)'}"
        )

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır (sahiplenilen kabuk oturumu kapatılır)."""
        if self._owns_shell and self._shell is not None:
            try:
                self._shell.shutdown()
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Kabuk kapatılamadı: {exc}")
        self._shell = None
        self._owns_shell = False
        self._initialized = False
        logger.info(
            f"Blender denetleyicisi kapatıldı ({self._counter['scripts']} betik, "
            f"{self._counter['renders']} render)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        İstatistikler.

        Returns:
            Sayaçlar + çalışma alanı, çalıştırılabilir yolu ve önbellekler
        """
        return {
            **self._counter,
            "initialized": self._initialized,
            "workspace": str(self._workspace or ""),
            "executable": self._executable,
            "version": self._version_cache,
            "engines": list(self._engine_cache),
            "owns_shell": self._owns_shell,
        }

    def health(self) -> Dict[str, Any]:
        """
        Kurulumun kullanılabilirliğini **tek çağrıda** bildirir.

        Ajanın işe başlamadan önce "Blender var mı, sürümü ne, hangi motorlar
        atanabiliyor, Blender Python'unda neler hazır?" sorusunu yanıtlar.

        Returns:
            {"ok", "executable", "version", "python", "engines", "modules",
             "threads", "hint"}
        """
        executable = self.executable()
        if not executable:
            return {
                "ok": False, "executable": "", "version": "", "engines": [],
                "error": "Blender çalıştırılabilir dosyası bulunamadı",
                "searched": list(SEARCH_ROOTS),
                "hint": "config.blender_path ayarlayın veya NC_BLENDER ortam "
                        "değişkenini tanımlayın",
            }

        started = time.perf_counter()
        version = self.version()
        python_info = self.python_environment() if version else {}
        engines = self._discover_engines() if version else []
        duration_ms = (time.perf_counter() - started) * 1000

        return {
            "ok": bool(version),
            "executable": executable,
            "version": version,
            "python": python_info.get("python_version", ""),
            "python_executable": python_info.get("executable", ""),
            "prefix": python_info.get("prefix", ""),
            "engines": engines or list(self._engine_cache),
            "default_engine": DEFAULT_ENGINE,
            "default_device": DEFAULT_DEVICE,
            "modules": python_info.get("modules", {}),
            "missing_modules": python_info.get("missing", []),
            "search_paths": python_info.get("path", []),
            "duration_ms": round(duration_ms, 1),
            "note": (
                "Varsayılan CYCLES + CPU: başsız makinede ekran/GPU bağlamı "
                "gerektirmez. EEVEE burada çalışır ama GPU'su olmayan/VM/RDP "
                "makinede kırılgandır ve ölçülen hızı daha düşüktür."
            ),
        }

    # ─── Temel Bilgiler ─────────────────────────────────────────────────

    def executable(self) -> str:
        """
        Kullanılacak `blender.exe` yolunu döndürür (yalnızca `initialize` ve
        ilk çağrıda bir kez aranır).

        `blender-launcher.exe` **asla** seçilmez: gerçek süreci ayrık başlatıp
        hemen döner, bu yüzden stdout ve anlamlı çıkış kodu vermez.

        Returns:
            Mutlak yol; bulunamazsa boş metin (istisna atılmaz)
        """
        if not self._executable:
            self._executable = self._detect_executable()
        return self._executable

    def version(self) -> str:
        """
        Blender sürüm metnini döndürür (ör. ``"5.2.2 LTS"``).

        Önce `--version` çıktısı okunur (tam sürüm + derleme bilgisi verir);
        olmazsa `bpy.app.version_string` sorulur. Sonuç önbelleklenir.

        Returns:
            Sürüm metni; Blender yoksa boş metin
        """
        if self._version_cache:
            return self._version_cache
        if not self.executable():
            return ""

        outcome = self._run_blender(["--version"], timeout=TIMEOUT_VERSION,
                                    stage="version", check_syntax=False)
        version = _first_version_line(outcome.get("stdout", ""))
        if not version:
            payload = self.run_expression("bpy.app.version_string",
                                          timeout=TIMEOUT_VERSION)
            if payload.get("ok"):
                value = payload.get("value") or payload.get("result")
                version = str(value) if not isinstance(value, dict) else ""
        self._version_cache = str(version).strip()
        if self._version_cache:
            self._engine_cache = self._discover_engines()
        return self._version_cache

    def python_environment(self) -> Dict[str, Any]:
        """
        Blender'ın **kendi izole Python'u** hakkında rapor verir.

        Neden gerekli? Blender `bpy`'yi yalnızca gömülü Python yorumlayıcısıyla
        sunar ve o yorumlayıcı sistemin `site-packages` dizinini **görmez**.
        Ajan `import PIL` yazıp başarısız olduğunda sebebini anlamak zor olur;
        bu metot hangi üçüncü parti modüllerin Blender içinden gerçekten içe
        aktarılabildiğini söyler.

        Returns:
            {"ok", "blender_version", "python_version", "executable", "prefix",
             "path", "modules", "missing", "hint"}
        """
        if not self.executable():
            return {"ok": False, "error": "Blender çalıştırılabilir dosyası yok",
                    "modules": {}, "missing": list(PROBE_MODULES)}

        outcome = self.run_script(
            _PYTHON_ENV_TEMPLATE.format(modules=repr(list(PROBE_MODULES))),
            timeout=TIMEOUT_EXPRESSION, script_name="nc_pyenv")
        payload = outcome.get("result") or {}
        if not isinstance(payload, dict) or not payload:
            return {
                "ok": False, "modules": {}, "missing": list(PROBE_MODULES),
                "error": outcome.get("error") or "Python ortamı okunamadı",
                "stdout_tail": str(outcome.get("stdout", ""))[-800:],
            }

        modules = {key: value for key, value in (payload.get("modules") or {}).items()
                   if value}
        missing = [key for key in PROBE_MODULES if key not in modules]
        self._version_cache = self._version_cache or str(
            payload.get("blender_version", ""))
        return {
            "ok": True,
            "blender_version": payload.get("blender_version", ""),
            "blender_version_tuple": payload.get("blender_version_tuple", []),
            "python_version": payload.get("python_version", ""),
            "python_version_tuple": payload.get("python_version_tuple", []),
            "executable": payload.get("executable", ""),
            "binary_path": payload.get("binary_path", ""),
            "prefix": payload.get("prefix", ""),
            "path": payload.get("path", []),
            "modules": modules,
            "missing": missing,
            "hint": (
                "Blender kendi Python'unu kullanır; sistem site-packages GÖRÜNMEZ. "
                "Eksik: " + (", ".join(missing) if missing else "yok")
            ),
            "duration_ms": outcome.get("duration_ms", 0.0),
        }

    # ─── Betik Çalıştırma (iş atı) ──────────────────────────────────────

    def run_script(self, code: str, timeout: float = TIMEOUT_SCRIPT,
                   blend_file: str = "", script_name: str = "nc_blender",
                   factory_startup: bool = True, out_dir: str = "",
                   threads: int = 0) -> Dict[str, Any]:
        """
        Verilen Python kodunu Blender içinde **başsız** çalıştırır.

        Çalıştırılan komut (sıra yük taşır!)::

            blender --background --factory-startup [--threads N] \\
                    --python-exit-code 1 [<blend_file>] --python '<script.py>'

        * `.blend` dosyası **`--python`'dan önce** verilir; aksi hâlde Blender
          dosyayı betikten sonra yükler ve betik boş sahne görür.
        * `--python-exit-code 1` **`--python`'dan önce** verilir; sonra
          verilirse etkisizdir (bu makinede doğrulandı). Betik gövdesi ayrıca
          `try/except` ile sarılır — iki katmanlı koruma.
        * `out_dir` alt sürece `NC_BLEND_OUT` ortam değişkeni olarak geçer;
          betik yolları **kod içine gömmek yerine**
          `os.environ["NC_BLEND_OUT"]` ile alır.

        Args:
            code: Blender içinde çalışacak Python kodu
            timeout: Saniye (varsayılan 600)
            blend_file: Önce yüklenecek `.blend` (boş = dosya argümanı verilmez)
            script_name: Geçici dosya adı öneki (hata ayıklamada iz sürmek için)
            factory_startup: Fabrika ayarlarıyla başlat (kullanıcı eklentileri ve
                bozuk başlangıç dosyaları devre dışı kalır — önerilir)
            out_dir: Çıktı dizini (`NC_BLEND_OUT`)
            threads: İş parçacığı sınırı (`--threads`); 0 = Blender seçsin

        Returns:
            {"ok", "exit_code", "stdout", "stderr", "result", "errors",
             "duration_ms", "script_path", "artifacts", "out_dir", "command"}
        """
        self._counter["scripts"] += 1
        if not self.executable():
            self._counter["failures"] += 1
            return self._failure("Blender çalıştırılabilir dosyası bulunamadı",
                                 hint="blender.initialize(...) çağrıldığından "
                                      "ve kurulum yolunun doğru olduğundan emin olun")

        resolved_blend = self._optional_path(blend_file)
        if blend_file and resolved_blend is None:
            self._counter["failures"] += 1
            return self._failure(f".blend dosyası yok: {blend_file}",
                                 hint="yolu mutlak verin veya önce build_scene ile üretin")

        work_dir = self._resolve_dir(out_dir)
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return self._failure(f"çıktı dizini oluşturulamadı: {exc}")

        script_path = self._write_temp_script(
            self._wrap_script(code, work_dir, collect=True,
                              budget=max(5.0, float(timeout))),
            script_name)
        if script_path is None:
            return self._failure("geçici betik yazılamadı")

        arguments = ["--background"]
        if factory_startup:
            arguments.append("--factory-startup")
        if int(threads or 0) > 0:
            arguments.extend(["--threads", str(int(threads))])
        arguments.append("--python-exit-code")
        arguments.append("1")
        if resolved_blend is not None:
            arguments.append(str(resolved_blend))
        arguments.extend(["--python", str(script_path)])

        environment = {ENV_OUT_DIR: str(work_dir)}
        if resolved_blend is not None:
            environment[ENV_BLEND_FILE] = str(resolved_blend)

        stamp = time.time() - 1.0
        outcome = self._run_blender(arguments, timeout=timeout, stage="run_script",
                                    environment=environment, cwd=work_dir)

        artifacts = self._collect_artifacts(work_dir, stamp)
        result = {
            "ok": outcome.get("ok", False),
            "exit_code": outcome.get("exit_code"),
            "stdout": outcome.get("stdout", ""),
            "stderr": outcome.get("stderr", ""),
            "result": outcome.get("result"),
            "errors": outcome.get("errors", []),
            "duration_ms": outcome.get("duration_ms", 0.0),
            "script_path": str(script_path),
            "artifacts": artifacts,
            "artifact_paths": [item["path"] for item in artifacts],
            "command": outcome.get("command", ""),
            "timed_out": outcome.get("timed_out", False),
            "out_dir": str(work_dir),
        }
        if not result["ok"]:
            self._counter["failures"] += 1
            if result["timed_out"]:
                self._counter["timeouts"] += 1
        self._emit("blender.script_finished", {
            "ok": result["ok"], "exit_code": result["exit_code"],
            "duration_ms": result["duration_ms"], "artifacts": len(artifacts),
            "errors": [item.get("message", "")[:160] for item in result["errors"][:3]],
        })
        return result

    def run_expression(self, expression: str,
                       timeout: float = TIMEOUT_EXPRESSION,
                       blend_file: str = "") -> Dict[str, Any]:
        """
        Kısa bir Python ifadesini Blender içinde çalıştırır.

        İfade `--python-expr` yerine **geçici betiğe** yazılır: `--python-expr`
        tırnak/kaçış yönetimini kırılgan yapar ve `--python-exit-code` ile
        birlikte kullanıldığında sıra sorunları doğurur. Tek satırlık ifadenin
        değeri otomatik olarak JSON'a çevrilir; çok satırlı kod olduğu gibi
        çalıştırılır (sonucu `print` ile bildirin).

        Args:
            expression: Python ifadesi veya kısa kod bloğu
            timeout: Saniye
            blend_file: Önce yüklenecek `.blend`

        Returns:
            `run_script` sözlüğü + `{"value"}` (tek satırlık ifadelerde)
        """
        self._counter["expressions"] += 1
        code = expression if "\n" in expression else (
            "value = (" + expression + ")\n"
            "print('" + RESULT_MARKER + "' + json.dumps({'value': value}, default=str))\n"
        )
        outcome = self.run_script(code, timeout=timeout, blend_file=blend_file,
                                  script_name="nc_expr")
        payload = outcome.get("result")
        value = None
        if isinstance(payload, dict):
            value = payload.get("value", payload)
        elif payload is not None:
            value = payload
        return {**outcome, "value": value}

    # ─── Sahne Kurulumu ─────────────────────────────────────────────────

    def build_scene(self, spec: Dict[str, Any], out_dir: str = "",
                    blend_name: str = "scene.blend",
                    render: bool = False) -> Dict[str, Any]:
        """
        JSON uyumlu **sahne tanımından** Blender Python'u üretip çalıştırır.

        Desteklenen nesne türleri: `cube`, `sphere`, `cylinder`, `plane`,
        `torus`, `text`, `empty`, `camera`, `sun`, `point_light`, `area_light`,
        `spot`. Kamera için `look_at` verilebilir: Blender'da "bak" komutu
        yoktur, yön vektörü `direction.to_track_quat('-Z', 'Y').to_euler()` ile
        döndürülür (yön `-Z`, yukarı `Y` — Blender kamera ekseniyle uyumlu).
        Işık nesneleri de `look_at` kabul eder.

        Malzeme tanımı dostu anahtarlar alır (`base_color`, `metallic`,
        `roughness`, `alpha`, `emission`, `emission_strength`) ve bunları 5.2'nin
        gerçek giriş adlarına eşler; bulunamayan giriş **mevcut girişleri
        listeleyen** bir hatayla bildirilir (bkz. `_material_code`).

        Args:
            spec: Sahne tanımı (sözlük):
                ``clear``, ``objects``, ``world``, ``render``, ``camera``,
                ``save`` anahtarlarını destekler.
            out_dir: Çıktı dizini (boş = çalışma alanı kökü)
            blend_name: Kaydedilecek dosya adı
            render: Kaydettikten sonra render al

        Returns:
            `run_script` sözlüğü + `{"blend_file", "blend_exists", "blend_size",
             "objects", "object_count", "render_file", "render_size",
             "spec_warnings"}`
        """
        self._counter["scenes"] += 1
        if not isinstance(spec, dict):
            return self._failure("spec bir sözlük olmalı")

        work_dir = self._resolve_dir(out_dir)
        warnings = self._validate_spec(spec)
        code = self._generate_scene_script(spec, blend_name, render)
        outcome = self.run_script(code, timeout=TIMEOUT_SCENE, out_dir=str(work_dir),
                                  script_name="nc_scene")
        payload = outcome.get("result") or {}

        raw_blend = payload.get("blend_file") if isinstance(payload, dict) else ""
        blend_path = Path(str(raw_blend)) if raw_blend else (work_dir / blend_name)
        raw_render = payload.get("render_file") if isinstance(payload, dict) else ""
        render_path = Path(str(raw_render)) if raw_render else None

        existence = blend_path.exists() and blend_path.stat().st_size > 0
        report = {
            **outcome,
            "blend_file": str(blend_path),
            "blend_exists": existence,
            "blend_size": blend_path.stat().st_size if blend_path.exists() else 0,
            "objects": payload.get("objects", []) if isinstance(payload, dict) else [],
            "object_count": len(payload.get("objects", []) or [])
            if isinstance(payload, dict) else 0,
            "render_file": str(render_path) if render_path else "",
            "render_size": render_path.stat().st_size
            if render_path and render_path.exists() else 0,
            "render_ms": payload.get("render_ms", 0.0) if isinstance(payload, dict) else 0.0,
            "spec_warnings": warnings,
        }
        if report["ok"] and not existence:
            report["ok"] = False
            report["errors"] = list(report.get("errors", [])) + [{
                "kind": "artifact_missing",
                "message": f".blend dosyası oluşmadı veya boş: {blend_path}",
                "hint": "kaydetme adımı sessizce atlanmış olabilir; çıktı "
                        "dizininin yazılabilir olduğunu doğrulayın",
            }]
        if report["ok"]:
            self._emit("blender.scene_built", {
                "blend_file": str(blend_path), "objects": report["object_count"],
                "size": report["blend_size"],
            })
        return report

    # ─── Render ─────────────────────────────────────────────────────────

    def render(self, blend_file: str = "", output: str = "", engine: str = "",
               samples: int = DEFAULT_SAMPLES,
               resolution: Sequence[int] = DEFAULT_RESOLUTION,
               frame: Optional[int] = None, timeout: float = TIMEOUT_RENDER,
               device: str = DEFAULT_DEVICE, threads: int = 0) -> Dict[str, Any]:
        """
        Bir `.blend` dosyasını başsız render eder.

        Neden `-f`/`-o` yerine **betik**? Betik yaklaşımı tam denetim verir:
        motor/örnek/çözünürlük çağrı anında ayarlanır, hedef yol
        `scene.render.filepath` ile sabitlenir (Blender kare numarasını adın
        sonuna eklemez) ve dönen `NC_RESULT` gerçek yolu bildirir. `-o`/`-f`
        ile `.blend` dosyasının sırası da kritiktir: dosya `-o`'dan sonra
        gelirse Blender eski çıktı yolunu kullanır. Bu modül dosyayı her zaman
        `--python`'dan önce, `-o`/`-F`/`-f` ise **hiç** kullanmadan çalışır.

        Motor: varsayılan **CYCLES + CPU**. Ölçüm (64x64/1 örnek): Cycles CPU
        ~0.22 s, EEVEE başsız ~10.92 s. EEVEE GPU bağlamı ister; GPU'su
        olmayan/VM/RDP makinede kırılgandır. `engine=""` → CYCLES.

        Args:
            blend_file: Render edilecek `.blend` (boş = boş sahne)
            output: Çıktı dosyası (`.png`); boş = `<çalışma alanı>/render.png`
            engine: `CYCLES` | `BLENDER_EEVEE` | `BLENDER_WORKBENCH` | boş
            samples: Cycles örnek sayısı
            resolution: (genişlik, yükseklik)
            frame: Tek kare numarası (None = sahnenin geçerli karesi)
            timeout: Saniye (varsayılan 1800)
            device: `CPU` | `GPU` (yalnızca Cycles)
            threads: İş parçacığı sınırı (0 = Blender seçsin)

        Returns:
            {"ok", "file", "exists", "size", "engine", "device", "samples",
             "resolution", "frame", "render_ms", "duration_ms", "errors", ...}
        """
        self._counter["renders"] += 1
        resolved_blend = self._optional_path(blend_file)
        if blend_file and resolved_blend is None:
            self._counter["failures"] += 1
            return self._failure(f".blend dosyası yok: {blend_file}")

        target = self._resolve_file(output, "render.png")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return self._failure(f"çıktı dizini oluşturulamadı: {exc}")

        width, height = _normalize_resolution(resolution)
        wanted_engine = (engine or DEFAULT_ENGINE).strip().upper()
        code = self._generate_render_script(str(target), wanted_engine, int(samples),
                                            width, height, frame, device)
        outcome = self.run_script(
            code, timeout=timeout,
            blend_file=str(resolved_blend) if resolved_blend else "",
            script_name="nc_render", out_dir=str(target.parent), threads=threads)
        payload = outcome.get("result") if isinstance(outcome.get("result"), dict) else {}

        reported = payload.get("file")
        if reported and Path(str(reported)).exists() and Path(str(reported)).stat().st_size > 0:
            target = Path(str(reported))
        exists = target.exists() and target.stat().st_size > 0

        errors = list(outcome.get("errors", []))
        ok = bool(outcome.get("ok")) and exists
        if outcome.get("ok") and not exists:
            errors.append({
                "kind": "artifact_missing",
                "message": f"render dosyası oluşmadı veya boş: {target}",
                "hint": "sahnede kamera var mı (scene.camera)? Ayarları "
                        "blender.render(samples=8, resolution=(320,240), "
                        "engine='CYCLES') ile küçültüp yeniden deneyin",
            })
        # CI/GPU hatası gibi durumlarda Cycles'a düşmeyi önerelim
        if not ok and any(item.get("kind") in ("gpu_error", "enum_invalid")
                          for item in errors) and wanted_engine != DEFAULT_ENGINE:
            errors.append({
                "kind": "advice",
                "message": f"{wanted_engine} başarısız oldu",
                "hint": "blender.render(..., engine='CYCLES', device='CPU') ile "
                        "yeniden deneyin — başsız makinede güvenli seçim budur",
            })
        if not ok:
            self._counter["failures"] += 1
            if outcome.get("timed_out"):
                self._counter["timeouts"] += 1
        else:
            self._emit("blender.rendered", {
                "file": str(target), "size": target.stat().st_size,
                "engine": payload.get("engine", wanted_engine),
                "render_ms": payload.get("render_ms", 0.0),
            })

        return {
            **outcome,
            "ok": ok,
            "file": str(target),
            "exists": exists,
            "size": target.stat().st_size if target.exists() else 0,
            "engine": payload.get("engine", wanted_engine),
            "engine_note": payload.get("engine_note", ""),
            "engines_available": payload.get("engines_available", self._engine_cache),
            "device": payload.get("device", device),
            "samples": payload.get("samples", samples),
            "resolution": payload.get("resolution", [width, height]),
            "frame": payload.get("frame"),
            "render_ms": payload.get("render_ms", 0.0),
            "threads": payload.get("threads", 0),
            "errors": errors,
        }

    # ─── İnceleme ───────────────────────────────────────────────────────

    def inspect_blend(self, blend_file: str) -> Dict[str, Any]:
        """
        Bir `.blend` dosyasının **JSON özetini** çıkarır.

        Ajanın dosyayı açmadan "içinde ne var?" sorusunu yanıtlar: dosyayı son
        kaydeden Blender sürümü, sahneler, nesneler (tür/konum/boyut), malzemeler
        ve temel renkleri, kameralar, ışıklar, koleksiyonlar, render ayarları,
        kare aralığı ve toplam tepe/yüz sayısı.

        Args:
            blend_file: İncelenecek `.blend` (zorunlu)

        Returns:
            `run_script` sözlüğü + `{"scene", "objects", "materials",
             "cameras", "lights", "collections", "counts"}`
        """
        self._counter["inspections"] += 1
        if not blend_file:
            return self._failure("blend_file zorunlu")
        resolved = self._optional_path(blend_file)
        if resolved is None:
            self._counter["failures"] += 1
            return self._failure(f".blend dosyası yok: {blend_file}")

        outcome = self.run_script(_INSPECT_SCRIPT, timeout=TIMEOUT_INSPECT,
                                  blend_file=str(resolved), script_name="nc_inspect",
                                  out_dir=str(resolved.parent))
        payload = outcome.get("result") if isinstance(outcome.get("result"), dict) else {}
        if outcome.get("ok") and not payload:
            outcome["ok"] = False
            outcome["errors"] = list(outcome.get("errors", [])) + [{
                "kind": "empty_report",
                "message": "inceleme betiği özet üretmedi",
                "hint": "dosya bozuk olabilir; dosyayı Blender'da açıp yeniden "
                        "kaydedin veya blender.list_objects ile deneyin",
            }]
        return {
            **outcome,
            "blend_file": str(resolved),
            "size": resolved.stat().st_size,
            "scene": payload.get("scene", {}),
            "scenes": payload.get("scenes", []),
            "objects": payload.get("objects", []),
            "object_count": len(payload.get("objects", []) or []),
            "object_names": [item.get("name") for item in payload.get("objects", [])],
            "materials": payload.get("materials", []),
            "material_count": len(payload.get("materials", []) or []),
            "cameras": payload.get("cameras", []),
            "lights": payload.get("lights", []),
            "collections": payload.get("collections", []),
            "counts": payload.get("counts", {}),
            "saved_by_version": payload.get("saved_by_version", []),
            "report": payload,
        }

    def list_objects(self, blend_file: str = "") -> Dict[str, Any]:
        """
        Bir `.blend` içindeki nesneleri listeler (hafif sürüm).

        `inspect_blend`'in aksine malzeme/düğüm ayrıntısı döndürmez; büyük
        sahnelerde ajan istemini şişirmemek için ad, tür, konum, boyut, ebeveyn
        ve görünürlük ile sınırlıdır.

        Args:
            blend_file: `.blend` yolu (boş = boş sahne)

        Returns:
            {"ok", "objects", "count", "types", "names"}
        """
        resolved = self._optional_path(blend_file)
        if blend_file and resolved is None:
            return self._failure(f".blend dosyası yok: {blend_file}")

        outcome = self.run_script(_LIST_OBJECTS_SCRIPT, timeout=TIMEOUT_INSPECT,
                                  blend_file=str(resolved) if resolved else "",
                                  script_name="nc_list_objects",
                                  out_dir=str(resolved.parent) if resolved else "")
        payload = outcome.get("result") if isinstance(outcome.get("result"), dict) else {}
        objects = payload.get("objects", []) if isinstance(payload, dict) else []
        types: Dict[str, int] = {}
        for item in objects:
            kind = str(item.get("type", "?"))
            types[kind] = types.get(kind, 0) + 1
        return {
            **outcome,
            "objects": objects,
            "count": len(objects),
            "types": dict(sorted(types.items(), key=lambda pair: -pair[1])),
            "names": [item.get("name") for item in objects],
        }

    def export_gltf(self, blend_file: str = "", output: str = "",
                    timeout: float = TIMEOUT_EXPORT) -> Dict[str, Any]:
        """
        Sahneyi glTF 2.0 (`.glb`) olarak dışa aktarır.

        Operatör `bpy.ops.export_scene.gltf`; argümanlar 5.2 RNA'sından
        doğrulandı: `export_format='GLB'`, `use_selection`, `export_apply`,
        `export_yup`, `export_materials`, `export_animations`. UI bağlamı
        gerekmez.

        Args:
            blend_file: Kaynak `.blend` (boş = boş sahne)
            output: Çıktı yolu (`.glb`); boş = `<çalışma alanı>/export.glb`
            timeout: Saniye

        Returns:
            {"ok", "file", "exists", "size", "format", "errors", ...}
        """
        return self._export(blend_file, output, "glb", timeout)

    def export_fbx(self, blend_file: str = "", output: str = "",
                   timeout: float = TIMEOUT_EXPORT) -> Dict[str, Any]:
        """
        Sahneyi FBX olarak dışa aktarır.

        Operatör `bpy.ops.export_scene.fbx`; argümanlar 5.2 RNA'sından
        doğrulandı: `use_selection`, `path_mode`, `embed_textures`,
        `bake_space_transform`, `apply_scale_options`, `axis_forward`,
        `axis_up`. Dikkat: FBX'te `export_format` **yoktur**, `export_selected`
        **yoktur** (doğrusu `use_selection`) ve `use_animations` **yoktur**
        (doğrusu `bake_anim`).

        Args:
            blend_file: Kaynak `.blend` (boş = boş sahne)
            output: Çıktı yolu (`.fbx`); boş = `<çalışma alanı>/export.fbx`
            timeout: Saniye

        Returns:
            `export_gltf` ile aynı yapı, `format="fbx"`
        """
        return self._export(blend_file, output, "fbx", timeout)

    # ─── Doğrulama ve Teşhis ────────────────────────────────────────────

    def validate_script(self, code: str) -> Dict[str, Any]:
        """
        Betiği **Blender'ı hiç başlatmadan** doğrular.

        İki şey yapılır:
            1. Sözdizimi **sistem Python'unun** `ast` modülüyle denetlenir
               (anında döner).
            2. Kodda `NC_ERROR` + `sys.exit(1)` sarmalayıcısı yoksa bu
               bildirilir (`has_error_handler=False`) ve `wrapped` alanında
               sarılmış sürüm verilir.

        Neden zorunlu? Blender yakalanmamış istisnada çıkış kodunu 0
        döndürebilir; sarmalayıcı hatayı görünür kılar.

        Args:
            code: Doğrulanacak Python kodu

        Returns:
            {"ok", "error", "has_error_handler", "wrapped", "lines", "imports",
             "advice"}
        """
        self._counter["validations"] += 1
        text = str(code or "")
        has_handler = bool(re.search(r"\bNC_ERROR\b", text)
                           and re.search(r"sys\.exit\s*\(\s*1\s*\)", text))
        info: Dict[str, Any] = {
            "ok": True, "error": "", "has_error_handler": has_handler,
            "lines": len(text.splitlines()), "imports": [], "wrapped": "",
            "advice": "",
        }
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            self._counter["syntax_errors"] += 1
            info.update({
                "ok": False,
                "error": f"satır {exc.lineno}, sütun {exc.offset}: {exc.msg}",
                "lineno": exc.lineno, "offset": exc.offset,
                "advice": "sözdizimi hatası — Blender'ı başlatmadan düzeltin "
                          "(saniyeler kazandırır)",
            })
            return info

        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        info["imports"] = sorted(set(imports))

        risky = sorted({name.split(".")[0] for name in info["imports"]
                        if name.split(".")[0] in set(PROBE_MODULES)})
        if risky:
            info["advice"] = (
                f"şu modüller Blender'ın izole Python'unda bulunmayabilir: "
                f"{', '.join(risky)} — blender.python_environment() ile doğrulayın "
                "(numpy/requests gelir; PIL, scipy, yaml, cv2 gelmez)"
            )
        if not has_handler:
            info["wrapped"] = self._wrap_script(text, None, collect=False)
            info["advice"] = ((info["advice"] + " | ") if info["advice"] else "") + (
                "kodda NC_ERROR + sys.exit(1) sarmalayıcısı yok; run_script zaten "
                "sarar, ancak betiği dışarıda çalıştıracaksanız `wrapped` alanını "
                "kullanın ve `--python-exit-code 1` bayrağını `--python`'dan ÖNCE verin"
            )
        return info

    def diagnose(self, text: str, command: str = "") -> Dict[str, Any]:
        """
        Blender hata çıktısını yorumlar: **ne oldu** ve **ne yapmalı**.

        Kapsanan başlıca durumlar: `bpy.ops` poll hatası, geçersiz enum (ör.
        `BLENDER_EEVEE_NEXT`), eksik operatör/yanlış argüman adı, düğüm girişi
        (`KeyError`) adı değişikliği, eksik dosya, Blender'ın izole Python'unda
        bulunmayan üçüncü parti modüller, GPU/CUDA, bellek yetersizliği, çökme
        (segmentation fault) ve `use_nodes` deprecation uyarısı (uyarı — hata
        değil).

        Args:
            text: stdout + stderr birleşik çıktı
            command: Çalıştırılan komut (bağlam için)

        Returns:
            {"ok", "errors": [{"kind","message","hint"}], "warnings",
             "error_count", "traceback", "summary"}
        """
        self._counter["diagnoses"] += 1
        raw = str(text or "")
        if not raw.strip():
            return {"ok": True, "errors": [], "warnings": [], "error_count": 0,
                    "traceback": False, "summary": "çıktı yok", "command": command}

        errors: List[Dict[str, Any]] = []
        seen: set = set()

        for pattern, kind, hint in _COMPILED_RULES:
            for match in pattern.finditer(raw):
                message = match.group(0).strip()[:300]
                key = (kind, message[:80])
                if key in seen:
                    continue
                seen.add(key)
                errors.append({"kind": kind, "message": message, "hint": hint})
                break

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped.startswith(ERROR_MARKER):
                continue
            payload = stripped[len(ERROR_MARKER):].strip()
            entry = _parse_json(payload)
            detail = ""
            if isinstance(entry, dict):
                detail = f"{entry.get('type', '')}: {entry.get('message', '')}"
            message = detail or payload[:280]
            key = ("script_error", message[:80])
            if key not in seen:
                seen.add(key)
                errors.insert(0, {
                    "kind": "script_error", "message": message,
                    "hint": "Blender betiği NC_ERROR ile sonlandı; yukarıdaki tür "
                            "ipucuna göre düzeltin, sonra blender.validate_script "
                            "ile hızlıca doğrulayın",
                })
            break

        traceback_seen = any(marker in raw for marker in TRACEBACK_MARKERS)
        if traceback_seen and not any(item["kind"] == "script_error" for item in errors):
            errors.insert(0, {
                "kind": "traceback",
                "message": "Python izleme kaydı (traceback) çıktıda görüldü",
                "hint": "izleme kaydındaki dosya ve satırı okuyun; betiği "
                        "parçalara bölüp ara değerleri print ile görün. Blender "
                        "istisnayı yutsa bile --python-exit-code 1 çıkış kodunu "
                        "1 yapar",
            })

        warnings: List[Dict[str, Any]] = []
        for index, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if re.search(r"\bWARNING\b|^\s*W:|\bDeprecationWarning\b", stripped,
                         re.IGNORECASE):
                warnings.append({"message": stripped[:220], "line": index})
                if len(warnings) >= 10:
                    break

        # Deprecation uyarıları hata değildir → ayrı listede tutulur
        errors = [item for item in errors if item["kind"] != "deprecation"]

        summary = "çıktı temiz — hata izi yok"
        if errors:
            primary = errors[0]
            summary = (f"{len(errors)} sorun: {primary['kind']} — "
                       f"{primary['message'][:140]} → {primary['hint'][:220]}")
        elif warnings:
            summary = f"{len(warnings)} uyarı (hata yok; Blender başarılı olabilir)"

        return {
            "ok": not errors,
            "errors": errors[:12],
            "warnings": warnings,
            "error_count": len(errors),
            "traceback": traceback_seen,
            "summary": summary,
            "command": command,
        }

    # ─── Dışa Aktarma Gövdesi ───────────────────────────────────────────

    def _export(self, blend_file: str, output: str, extension: str,
                timeout: float) -> Dict[str, Any]:
        """
        glTF/FBX dışa aktarmanın ortak gövdesi.

        Args:
            blend_file: Kaynak `.blend`
            output: Çıktı yolu
            extension: `glb` | `fbx`
            timeout: Saniye

        Returns:
            Ortak dışa aktarma sözlüğü
        """
        self._counter["exports"] += 1
        resolved_blend = self._optional_path(blend_file)
        if blend_file and resolved_blend is None:
            self._counter["failures"] += 1
            return self._failure(f".blend dosyası yok: {blend_file}")

        target = self._resolve_file(output, f"export.{extension}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return self._failure(f"çıktı dizini oluşturulamadı: {exc}")

        code = _EXPORT_GLTF_SCRIPT if extension == "glb" else _EXPORT_FBX_SCRIPT
        code = code.format(target=repr(str(target)))
        outcome = self.run_script(
            code, timeout=timeout,
            blend_file=str(resolved_blend) if resolved_blend else "",
            script_name=f"nc_export_{extension}", out_dir=str(target.parent))
        payload = outcome.get("result") if isinstance(outcome.get("result"), dict) else {}

        reported = payload.get("file")
        if reported and Path(str(reported)).exists() and Path(str(reported)).stat().st_size > 0:
            target = Path(str(reported))
        exists = target.exists() and target.stat().st_size > 0

        errors = list(outcome.get("errors", []))
        ok = bool(outcome.get("ok")) and exists
        if outcome.get("ok") and not exists:
            errors.append({
                "kind": "artifact_missing",
                "message": f"dışa aktarma dosyası oluşmadı veya boş: {target}",
                "hint": "operatörün varlığını denetleyin: "
                        "print([n for n in dir(bpy.ops.export_scene)]); ilgili "
                        "eklentinin (io_scene_gltf2 / io_scene_fbx) etkin olduğundan "
                        "emin olun",
            })
        if not ok:
            self._counter["failures"] += 1
            if outcome.get("timed_out"):
                self._counter["timeouts"] += 1
        else:
            self._emit("blender.exported", {
                "format": extension.upper(), "file": str(target),
                "size": target.stat().st_size,
            })

        return {
            **outcome,
            "ok": ok,
            "format": extension.upper(),
            "file": str(target),
            "exists": exists,
            "size": target.stat().st_size if target.exists() else 0,
            "objects": payload.get("objects", []),
            "errors": errors,
        }

    # ─── Blender Çağrı Altyapısı ────────────────────────────────────────

    def _run_blender(self, arguments: Sequence[str], timeout: float, stage: str,
                     environment: Optional[Dict[str, str]] = None,
                     cwd: Optional[Path] = None,
                     check_syntax: bool = True) -> Dict[str, Any]:
        """
        `blender.exe` sürecini çalıştırır ve çıktıyı ayrıştırır.

        Komut enjekte edilen kabuk katmanından geçer (denetim ve kum havuzu tek
        yerde kalsın diye). Çalışma dizini her zaman mutlak verilir; `cd`
        **asla** kullanılmaz. Ortam değişkenleri komut öneki olarak verilir
        (`ShellIntegration.run` imzası `env` kabul etmediği için) ve çağrıdan
        sonra ana sürecin ortamı eski hâline döndürülür.

        Args:
            arguments: Blender'a verilecek argümanlar
            timeout: Saniye
            stage: Günlük/olay kaydı için aşama adı
            environment: Alt sürece eklenecek ortam değişkenleri
            cwd: Çalışma dizini (None = çalışma alanı)
            check_syntax: Çıktıda izleme kaydı aransın mı?

        Returns:
            {"ok", "exit_code", "stdout", "stderr", "result", "errors",
             "duration_ms", "command", "timed_out"}
        """
        executable = self.executable()
        if not executable or not Path(executable).is_file():
            # Nezaketli bozulma: sahte/eksik yol kabuğa hiç verilmez. Yol
            # SIFIRLANMAZ; aksi hâlde sonraki çağrı yeniden arama yapıp gerçek
            # Blender'ı bulur ve "Blender yok" durumu sessizce maskelenir.
            return {
                "ok": False, "exit_code": -1, "stdout": "", "stderr": "",
                "result": None, "errors": [{
                    "kind": "no_executable",
                    "message": f"Blender çalıştırılabilir dosyası yok: "
                               f"{executable or '(bulunamadı)'}",
                    "hint": "config.blender_path veya NC_BLENDER ortam değişkenini "
                            "ayarlayın; blender.health() ile kurulumu denetleyin",
                }],
                "duration_ms": 0.0, "command": "", "timed_out": False,
                "error": "Blender yok",
            }

        work_dir = cwd or self._workspace or Path.cwd()
        command = _build_command(executable, arguments, environment)
        started = time.perf_counter()

        if self._shell is None:
            from neurocore.agent.shell import ShellIntegration

            self._shell = ShellIntegration()
            self._shell.initialize(workspace=str(work_dir), config=self._config,
                                   event_bus=self._event_bus)
            self._owns_shell = True

        saved_environment = {key: os.environ.get(key)
                             for key in (environment or {})}
        os.environ.update({key: str(value)
                           for key, value in (environment or {}).items()})
        try:
            outcome = self._shell.run(
                command, cwd=str(work_dir),
                timeout=float(timeout) + SHELL_TIMEOUT_MARGIN)
        except Exception as exc:
            logger.error(f"Blender çalıştırılamadı ({stage}): {exc}")
            return {
                "ok": False, "exit_code": -1, "stdout": "", "stderr": str(exc),
                "result": None, "errors": [{
                    "kind": "launch_failed", "message": str(exc),
                    "hint": "kabuk katmanının çalıştığını doğrulayın",
                }],
                "duration_ms": (time.perf_counter() - started) * 1000,
                "command": command, "timed_out": False, "error": str(exc),
            }
        finally:
            for key, previous in saved_environment.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

        stdout = str(getattr(outcome, "stdout", "") or "")
        stderr = str(getattr(outcome, "stderr", "") or "")
        exit_code = int(getattr(outcome, "returncode", -1))
        timed_out = bool(getattr(outcome, "timed_out", False))
        blocked = bool(getattr(outcome, "blocked", False))
        duration_ms = float(getattr(outcome, "duration_ms", 0.0) or
                            (time.perf_counter() - started) * 1000)

        result, error_lines = _parse_markers(stdout, stderr)
        errors: List[Dict[str, Any]] = []
        for line in error_lines:
            entry = _parse_json(line)
            if isinstance(entry, dict):
                errors.append({
                    "kind": "script_error",
                    "message": f"{entry.get('type', '')}: "
                               f"{entry.get('message', '')}".strip(": "),
                    "hint": "üretilen betik NC_ERROR ile bitti; "
                            "blender.diagnose(stdout) ayrıntılı ipucu verir",
                })
            else:
                errors.append({"kind": "script_error", "message": line[:400],
                               "hint": "üretilen betik NC_ERROR ile bitti"})

        if timed_out:
            # BURAYA NORMALDE GELİNMEZ: zaman sınırı üretilen betiğin içindeki
            # watchdog ile uygulanır, çünkü Windows'ta takılan bir torun süreç
            # boruyu açık tutar ve `subprocess.run(timeout=)` çok geç döner
            # (ölçüldü: 8 s'lik sınır 59.6 s'de doldu). Yine de Blender Python'u
            # hiç başlatılamadıysa (bozuk kurulum, anında çöken süreç) bu yol
            # çalışır: yetim kalmış olabilecek süreçler burada temizlenir.
            killed = _kill_lingering_blender(arguments)
            errors.append({
                "kind": "timeout",
                "message": f"Blender {timeout:.0f} saniyede bitmedi",
                "hint": "işi küçültün (daha az örnek/kare, daha düşük çözünürlük) "
                        "veya timeout'u artırın; threads=2 ile makineyi duyarlı "
                        "tutun. Süreç sonlandırıldı, ajan asılı kalmaz",
                "killed_processes": killed,
            })
        if blocked:
            errors.append({
                "kind": "blocked",
                "message": str(getattr(outcome, "blocked_reason", "") or "komut engellendi"),
                "hint": "kabuk/güvenlik katmanı bu komutu engelledi",
            })
        if check_syntax and not errors:
            haystack = f"{stdout}\n{stderr}"
            if any(marker in haystack for marker in TRACEBACK_MARKERS):
                errors.append({
                    "kind": "traceback",
                    "message": _last_traceback_line(haystack),
                    "hint": "Blender istisnayı yuttu ve NC_ERROR basılmadı; kodun "
                            "try/except ile sarıldığından emin olun "
                            "(blender.validate_script) ve --python-exit-code "
                            "bayrağının --python'dan ÖNCE geldiğini doğrulayın",
                })

        ok = exit_code == 0 and not timed_out and not blocked and not errors
        self._emit("blender.invocation", {
            "stage": stage, "ok": ok, "exit_code": exit_code,
            "duration_ms": duration_ms, "timed_out": timed_out,
        })
        return {
            "ok": ok,
            "exit_code": exit_code,
            "stdout": stdout[-MAX_OUTPUT_CHARS:],
            "stderr": stderr[-MAX_OUTPUT_CHARS // 2:],
            "result": result,
            "errors": errors,
            "duration_ms": duration_ms,
            "command": command,
            "timed_out": timed_out,
            "blocked": blocked,
        }

    def _discover_engines(self) -> List[str]:
        """
        Bu yapıda **gerçekten atanabilen** render motorlarını keşfeder.

        `bl_rna.properties["engine"].enum_items` bu makinede yalnızca
        `BLENDER_EEVEE` gösterir; eklenti motorları (`CYCLES`,
        `BLENDER_WORKBENCH`) listede görünmez ama atanabilir. Bu yüzden liste
        keşfedilir, sonra her aday **gerçekten atanarak** sınanır ve atanabilen
        motorun adı raporlanır (yan etki: sahnenin motoru döngü sonunda
        CYCLES'e döner).

        Returns:
            Atanabilen motor kimlikleri listesi (boş olabilir)
        """
        if self._engine_cache:
            return self._engine_cache
        outcome = self.run_script(_ENGINE_PROBE_SCRIPT, timeout=TIMEOUT_EXPRESSION,
                                  script_name="nc_engines")
        payload = outcome.get("result")
        if isinstance(payload, dict) and payload.get("settable"):
            self._engine_cache = [str(item) for item in payload["settable"]]
        return self._engine_cache

    # ─── Kod Üretimi ────────────────────────────────────────────────────

    def _wrap_script(self, code: str, out_dir: Optional[Path],
                     collect: bool, budget: float = 0.0) -> str:
        """
        Kullanıcı kodunu **zorunlu** hata sarmalayıcısıyla sarar.

        Modülün en kritik güvenlik ağı budur: Blender yakalanmamış istisnada
        çıkış kodunu 0 döndürebilir. Sarmalayıcı her durumda `NC_ERROR` basar,
        izleme kaydını stderr'e yazar, `sys.exit(1)` çağırır ve çıkmadan önce
        `sys.stdout.flush()` yapar — aksi hâlde tamponlanan `NC_RESULT` kaybolur.

        Ayrıca `budget` saniyelik bir **watchdog** iş parçacığı başlatılır.
        `budget` verilmezse sarmalayıcının kendi zaman aşımı devre dışı kalır ve
        yalnızca kabuk katmanının zaman aşımı geçerli olur; bu durumda takılan
        bir Blender torun süreç olarak yaşamaya devam edebilir (bkz.
        `nc_watchdog` açıklaması). Bu yüzden `run_script` her zaman bir bütçe
        geçirir.

        Args:
            code: Çalıştırılacak Blender kodu
            out_dir: Yapıt taraması yapılacak dizin (None = ortam değişkeni)
            collect: Yapıt taraması yapılsın mı
            budget: Katı zaman sınırı (saniye); 0 = sınır yok

        Returns:
            Çalıştırılabilir tam betik
        """
        header = _WRAPPER_HEADER.format(
            out_env=ENV_OUT_DIR, blend_env=ENV_BLEND_FILE,
            out_repr=repr(str(out_dir)) if out_dir is not None else "''",
            collect=collect, budget=float(budget),
            helpers=_SCRIPT_HELPERS.format(KNOWN=list(KNOWN_ENGINES)),
            error_marker=ERROR_MARKER,
        )
        body = "\n".join(("    " + line) if line.strip() else ""
                         for line in code.splitlines())
        footer = _WRAPPER_FOOTER.format(error_marker=ERROR_MARKER)
        return header + body + "\n" + footer

    def _generate_scene_script(self, spec: Dict[str, Any], blend_name: str,
                               render: bool) -> str:
        """
        Sahne tanımından Blender Python'u üretir.

        Üretilen kod veri API'sini tercih eder; `bpy.ops` yalnızca
        `read_factory_settings`, `save_as_mainfile` ve `render.render` için
        kullanılır — bunlar başsız modda bağlam gerektirmez.

        Args:
            spec: Sahne tanımı
            blend_name: Kaydedilecek dosya adı
            render: Render alınsın mı?

        Returns:
            Blender Python kaynağı
        """
        objects = spec.get("objects") or []
        world = spec.get("world") or {}
        render_spec = spec.get("render") or {}
        camera_name = str(spec.get("camera") or "")
        save_name = str(spec.get("save") or blend_name or "scene.blend")

        lines: List[str] = []
        add = lines.append
        add("# ── Sahne kurulumu (NeuroCore tarafından üretildi) ──")
        if bool(spec.get("clear", True)):
            add("# read_factory_settings mevcut Scene RNA'sını GEÇERSİZ kılar;")
            add("# bu yüzden sahneye sıfırlamadan SONRA erişilir.")
            add("bpy.ops.wm.read_factory_settings(use_empty=True)")
        add("scene = bpy.context.scene")

        if world:
            color = _color(world.get("background_color"), (0.05, 0.05, 0.05, 1.0))
            strength = float(world.get("strength", 1.0) or 1.0)
            add("world = bpy.data.worlds.get('NCWorld') or bpy.data.worlds.new('NCWorld')")
            add("scene.world = world")
            add("nc_use_nodes(world)")
            add("background = world.node_tree.nodes.get('Background')")
            add("if background is None:")
            add("    background = world.node_tree.nodes.new('ShaderNodeBackground')")
            add(f"background.inputs['Color'].default_value = {color!r}")
            add(f"background.inputs['Strength'].default_value = {strength!r}")

        add("created = []")
        for index, item in enumerate(objects):
            if isinstance(item, dict):
                lines.extend(self._object_code(item, index))

        if camera_name:
            add(f"camera = bpy.data.objects.get({camera_name!r})")
            add("if camera is not None and camera.type == 'CAMERA':")
            add("    scene.camera = camera")
        else:
            add("cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']")
            add("if scene.camera is None and cameras:")
            add("    scene.camera = cameras[0]")

        lines.extend(self._render_settings_code(render_spec))

        add("# Kaydetme: yol NC_BLEND_OUT üzerinden gelir (bu modül asla cd yapmaz)")
        add("if not NC_OUT_DIR:")
        add("    raise RuntimeError('NC_BLEND_OUT tanimsiz - ortam degiskeni gecmedi')")
        add(f"blend_path = os.path.join(NC_OUT_DIR, {save_name!r})")
        add("bpy.ops.wm.save_as_mainfile(filepath=blend_path, check_existing=False)")
        add("if not os.path.isfile(blend_path) or os.path.getsize(blend_path) == 0:")
        add("    raise RuntimeError('kaydetme basarisiz: ' + str(blend_path))")

        if render:
            width, height = _normalize_resolution(
                render_spec.get("resolution", DEFAULT_RESOLUTION))
            file_format = str(render_spec.get("format") or "PNG").upper()
            extension = {"JPEG": "jpg", "JPG": "jpg"}.get(file_format,
                                                          file_format.lower())
            add("# Render: dosya adı SABİT verilir → kare numarası eklenmez")
            add(f"scene.render.filepath = os.path.join(NC_OUT_DIR, '1_render.{extension}')")
            add("if scene.camera is None:")
            add("    raise RuntimeError('sahnede kamera yok - render alinamaz; "
                "spec nesnelerine {\"type\": \"camera\"} ekleyin')")
            add("render_started = time.time()")
            add("bpy.ops.render.render(write_still=True)")
            add("render_ms = (time.time() - render_started) * 1000.0")
            add("render_file = scene.render.filepath")
            add("if not os.path.isfile(render_file) and os.path.isfile("
                "bpy.path.abspath(render_file)):")
            add("    render_file = bpy.path.abspath(render_file)")
            add("if not os.path.isfile(render_file) or os.path.getsize(render_file) == 0:")
            add("    raise RuntimeError('render dosyasi olusmadi veya bos: ' "
                "+ str(render_file))")
        else:
            add("render_ms = 0.0")
            add("render_file = ''")

        add("print('" + RESULT_MARKER + "' + ' ' + json.dumps({")
        add("    'blend_file': blend_path,")
        add("    'blend_size': os.path.getsize(blend_path),")
        add("    'objects': created,")
        add("    'object_count': len(created),")
        add("    'render_file': render_file,")
        add("    'render_ms': render_ms,")
        add("    'engine': scene.render.engine,")
        add("    'resolution': [scene.render.resolution_x, scene.render.resolution_y],")
        add("    'camera': scene.camera.name if scene.camera else '',")
        add("    'blender_version': bpy.app.version_string,")
        add("}, default=str))")
        return "\n".join(lines) + "\n"

    def _object_code(self, item: Dict[str, Any], index: int) -> List[str]:
        """
        Tek bir nesne tanımı için Blender kodu üretir.

        Tüm nesneler veri API'siyle oluşturulur (veri bloğu → nesne → bağlama →
        ad → dönüşüm → malzeme); böylece `bpy.context` hiç gerekmez. Metin
        nesnesi `bpy.data.curves.new(type='FONT')` ile kurulur — `bpy.ops.
        object.text_add` bağlam gerektirdiği için kullanılmaz.

        Args:
            item: Nesne tanımı
            index: Sıra numarası (ad üretimi için)

        Returns:
            Kod satırları
        """
        kind = str(item.get("type") or "cube").strip().lower()
        name = str(item.get("name") or f"{kind.title()}.{index + 1:03d}")
        location = _vector(item.get("location"), (0.0, 0.0, 0.0))
        rotation = _vector(item.get("rotation"), (0.0, 0.0, 0.0))
        scale = _vector(item.get("scale"), (1.0, 1.0, 1.0))
        look_at = item.get("look_at")

        lines: List[str] = []
        add = lines.append
        add(f"# ── {name} ({kind}) ──")
        add(f"name = {name!r}")

        if kind == "cube":
            add("mesh = bpy.data.meshes.new(name + 'Mesh')")
            add(f"half = {float(item.get('size', 2.0) or 2.0)!r} / 2.0")
            add("verts = [(-half, -half, -half), (half, -half, -half),")
            add("         (half, half, -half), (-half, half, -half),")
            add("         (-half, -half, half), (half, -half, half),")
            add("         (half, half, half), (-half, half, half)]")
            add("faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),")
            add("         (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]")
            add("mesh.from_pydata(verts, [], faces)")
            add("mesh.update()")
            add("obj = bpy.data.objects.new(name, mesh)")
        elif kind == "sphere":
            radius = float(item.get("radius", 1.0) or 1.0)
            segments = max(3, int(item.get("segments", 32) or 32))
            rings = max(3, int(item.get("rings", 16) or 16))
            add("import math as _math")
            add("mesh = bpy.data.meshes.new(name + 'Mesh')")
            add("verts = []")
            add(f"for _ring in range(1, {rings}):")
            add(f"    _phi = _math.pi * _ring / {rings}")
            add(f"    for _segment in range({segments}):")
            add(f"        _theta = 2.0 * _math.pi * _segment / {segments}")
            add(f"        _s = {radius!r} * _math.sin(_phi)")
            add("        verts.append((_s * _math.cos(_theta), _s * _math.sin(_theta),")
            add(f"                     {radius!r} * _math.cos(_phi)))")
            add("_top = len(verts)")
            add(f"verts.append((0.0, 0.0, {radius!r}))")
            add("_bottom = len(verts)")
            add(f"verts.append((0.0, 0.0, {-radius!r}))")
            add("faces = []")
            add(f"for _ring in range({rings} - 2):")
            add(f"    for _segment in range({segments}):")
            add(f"        _next = (_segment + 1) % {segments}")
            add(f"        _a = _ring * {segments} + _segment")
            add(f"        _b = _ring * {segments} + _next")
            add(f"        _c = (_ring + 1) * {segments} + _next")
            add(f"        _d = (_ring + 1) * {segments} + _segment")
            add("        faces.append((_a, _b, _c, _d))")
            add(f"for _segment in range({segments}):")
            add(f"    _next = (_segment + 1) % {segments}")
            add("    faces.append((_top, _next, _segment))")
            add(f"    faces.append((_bottom, {rings - 1} * {segments} + _segment,")
            add(f"                  {rings - 1} * {segments} + _next))")
            add("mesh.from_pydata(verts, [], faces)")
            add("mesh.update()")
            add("obj = bpy.data.objects.new(name, mesh)")
        elif kind == "cylinder":
            radius = float(item.get("radius", 1.0) or 1.0)
            depth = float(item.get("depth", 2.0) or 2.0)
            segments = max(3, int(item.get("vertices", 32) or 32))
            add("import math as _math")
            add("mesh = bpy.data.meshes.new(name + 'Mesh')")
            add("verts = []")
            add(f"for _layer in (0.0, 1.0):")
            add(f"    _z = -{depth!r} / 2.0 + {depth!r} * _layer")
            add(f"    for _segment in range({segments}):")
            add(f"        _theta = 2.0 * _math.pi * _segment / {segments}")
            add(f"        verts.append(({radius!r} * _math.cos(_theta),")
            add(f"                     {radius!r} * _math.sin(_theta), _z))")
            add("faces = []")
            add(f"for _segment in range({segments}):")
            add(f"    _next = (_segment + 1) % {segments}")
            add(f"    faces.append((_segment, _next, {segments} + _next,"
                f" {segments} + _segment))")
            add(f"faces.append(tuple(range({segments} - 1, -1, -1)))")
            add(f"faces.append(tuple(range({segments}, {segments} * 2)))")
            add("mesh.from_pydata(verts, [], faces)")
            add("mesh.update()")
            add("obj = bpy.data.objects.new(name, mesh)")
        elif kind == "plane":
            add("mesh = bpy.data.meshes.new(name + 'Mesh')")
            add(f"half = {float(item.get('size', 2.0) or 2.0)!r} / 2.0")
            add("verts = [(-half, -half, 0.0), (half, -half, 0.0),")
            add("         (half, half, 0.0), (-half, half, 0.0)]")
            add("mesh.from_pydata(verts, [], [(0, 1, 2, 3)])")
            add("mesh.update()")
            add("obj = bpy.data.objects.new(name, mesh)")
        elif kind == "torus":
            major = float(item.get("major_radius", 1.0) or 1.0)
            minor = float(item.get("minor_radius", 0.25) or 0.25)
            major_segments = max(3, int(item.get("major_segments", 48) or 48))
            minor_segments = max(3, int(item.get("minor_segments", 12) or 12))
            add("import math as _math")
            add("mesh = bpy.data.meshes.new(name + 'Mesh')")
            add("verts = []")
            add(f"for _i in range({major_segments}):")
            add(f"    _u = 2.0 * _math.pi * _i / {major_segments}")
            add(f"    for _j in range({minor_segments}):")
            add(f"        _v = 2.0 * _math.pi * _j / {minor_segments}")
            add(f"        _r = {major!r} + {minor!r} * _math.cos(_v)")
            add("        verts.append((_r * _math.cos(_u), _r * _math.sin(_u),")
            add(f"                     {minor!r} * _math.sin(_v)))")
            add("faces = []")
            add(f"for _i in range({major_segments}):")
            add(f"    for _j in range({minor_segments}):")
            add(f"        _a = _i * {minor_segments} + _j")
            add(f"        _b = ((_i + 1) % {major_segments}) * {minor_segments} + _j")
            add(f"        _c = ((_i + 1) % {major_segments}) * {minor_segments}"
                f" + ((_j + 1) % {minor_segments})")
            add(f"        _d = _i * {minor_segments} + ((_j + 1) % {minor_segments})")
            add("        faces.append((_a, _b, _c, _d))")
            add("mesh.from_pydata(verts, [], faces)")
            add("mesh.update()")
            add("obj = bpy.data.objects.new(name, mesh)")
        elif kind == "text":
            body = str(item.get("body") or item.get("text") or "Metin")
            add("# FONT eğrisi veri API'siyle kurulur: bpy.ops.object.text_add")
            add("# bir aktif alan/bağlam bekler ve başsız modda poll() verir.")
            add("curve = bpy.data.curves.new(name + 'Curve', type='FONT')")
            add(f"curve.body = {body!r}")
            add(f"curve.size = {float(item.get('size', 1.0) or 1.0)!r}")
            add("curve.align_x = 'CENTER'")
            add("curve.align_y = 'CENTER'")
            add("obj = bpy.data.objects.new(name, curve)")
        elif kind == "empty":
            add("obj = bpy.data.objects.new(name, None)")
            add("obj.empty_display_size = 1.0")
        elif kind == "camera":
            add("data = bpy.data.cameras.new(name)")
            add(f"data.lens = {float(item.get('lens', 50.0) or 50.0)!r}")
            add("obj = bpy.data.objects.new(name, data)")
        elif kind in ("sun", "point_light", "area_light", "spot", "light"):
            light_type = {"sun": "SUN", "point_light": "POINT", "area_light": "AREA",
                          "spot": "SPOT", "light": "POINT"}[kind]
            default_energy = 1000.0 if light_type in ("POINT", "AREA", "SPOT") else 3.0
            add(f"data = bpy.data.lights.new(name, type={light_type!r})")
            add(f"data.energy = {float(item.get('energy', default_energy) or default_energy)!r}")
            if light_type in ("AREA", "POINT", "SPOT"):
                add(f"data.size = {float(item.get('size', 1.0) or 1.0)!r}")
            if item.get("color"):
                add(f"data.color = {_vector(item.get('color'), (1.0, 1.0, 1.0))!r}")
            add("obj = bpy.data.objects.new(name, data)")
        else:
            add(f"# Bilinmeyen tür {kind!r} → boş nesne olarak eklendi")
            add("obj = bpy.data.objects.new(name, None)")

        add("if scene.collection.objects.get(name) is not None:")
        add("    obj.name = name + '.001'")
        add("scene.collection.objects.link(obj)")
        add(f"obj.location = {location!r}")
        add(f"obj.rotation_euler = {rotation!r}")
        add(f"obj.scale = {scale!r}")

        if look_at and kind in ("camera", "sun", "point_light", "area_light",
                                "spot", "light"):
            target = _vector(look_at, (0.0, 0.0, 0.0))
            add("# Blender'da 'bak' komutu YOKTUR: yön vektörü izleme")
            add("# kuaterniyonuna çevrilir (kamera -Z'ye bakar, yukarısı +Y).")
            add("from mathutils import Vector")
            add(f"_target = Vector({target!r})")
            add("_direction = _target - obj.location")
            add("if _direction.length > 0.0:")
            add("    obj.rotation_euler = _direction.to_track_quat('-Z', 'Y').to_euler()")

        material = item.get("material")
        if isinstance(material, dict) and material:
            lines.extend(self._material_code(material, name))
        add("created.append({'name': obj.name, 'type': obj.type,")
        add("                'location': [round(float(v), 5) for v in obj.location]})")
        return lines

    def _material_code(self, material: Dict[str, Any], object_name: str) -> List[str]:
        """
        Malzeme kuran kod üretir (düğümlere **doğrudan** erişim).

        `Material.use_nodes` 5.2'de kullanımdan kaldırıldı ve zaten `True`'dur;
        `bpy.data.materials.new()` düğüm ağacını hazır oluşturur. Bu yüzden
        üretilen kod bayrağa **dokunmaz** (uyarı yalnızca niteliğe erişince
        çıkar); yalnızca eski sürümlerde (varsa ve `False` ise) `nc_use_nodes`
        yardımcısı uyarıyı bastırarak açar.

        Dostu anahtarlar (`base_color`, `metallic`, `specular`, `transmission`,
        `clearcoat`, `emission`, ...) 5.2'nin gerçek giriş adlarına eşlenir.
        Girişler bulunamazsa hata **mevcut giriş adlarını listeler** — böylece
        ajan `KeyError: 'Specular'` gibi anlamsız bir hata yerine ne
        yapacağını bilir. `inputs` sözlüğüyle ham giriş adları da verilebilir;
        bu adlar da mevcut girişlere karşı doğrulanır.

        Args:
            material: Malzeme tanımı
            object_name: Malzemenin atanacağı nesnenin adı

        Returns:
            Kod satırları
        """
        name = str(material.get("name") or f"{object_name}_Malzeme")
        base_color = _color(material.get("base_color", material.get("color")),
                            (0.8, 0.8, 0.8, 1.0))
        required = [("Base Color", "base_color", base_color),
                    ("Metallic", "metallic", float(material.get("metallic", 0.0) or 0.0)),
                    ("Roughness", "roughness", float(material.get("roughness", 0.5) or 0.5))]
        optional = [("Alpha", material.get("alpha")),
                    ("IOR", material.get("ior")),
                    ("Transmission Weight", material.get("transmission")),
                    ("Coat Weight", material.get("coat", material.get("clearcoat"))),
                    ("Coat Roughness", material.get("coat_roughness",
                                                    material.get("clearcoat_roughness"))),
                    ("Sheen Weight", material.get("sheen")),
                    ("Specular IOR Level", material.get("specular")),
                    ("Specular Tint", material.get("specular_tint")),
                    ("Subsurface Weight", material.get("subsurface")),
                    ("Anisotropic", material.get("anisotropic")),
                    ("Emission Strength", material.get("emission_strength"))]
        emission = material.get("emission")

        lines: List[str] = []
        add = lines.append
        add(f"# ── Malzeme: {name} ──")
        add(f"material = bpy.data.materials.get({name!r}) or bpy.data.materials.new({name!r})")
        add("# NOT: Material.use_nodes 5.2'de kullanımdan kaldırıldı (6.0'da")
        add("# kaldırılacak) ve zaten True; bu yüzden BURADA ATANMAZ. Yalnızca eski")
        add("# sürümler için nc_use_nodes yardımcısı gerekirse açar.")
        add("nc_use_nodes(material)")
        add("nodes = material.node_tree.nodes")
        add("bsdf = nodes.get('Principled BSDF')")
        add("if bsdf is None:")
        add("    bsdf = nodes.new('ShaderNodeBsdfPrincipled')")
        add("    output = nodes.get('Material Output')")
        add("    if output is None:")
        add("        output = nodes.new('ShaderNodeOutputMaterial')")
        add("    material.node_tree.links.new(bsdf.outputs[0], output.inputs[0])")
        add("# Zorunlu girişler: bulunamazsa mevcut girişler listelenir")
        for socket, key, value in required:
            add(f"_socket = bsdf.inputs.get({socket!r})")
            add("if _socket is None:")
            add("    raise RuntimeError(" + repr(
                f"Principled BSDF girisinde {socket!r} yok (istenen {key!r}). "
                "Mevcut girisler: ") + " + ', '.join(nc_sockets(bsdf)))")
            add(f"_socket.default_value = {value!r}")
        add(f"material.diffuse_color = {base_color!r}")
        for socket, value in optional:
            if value is None:
                continue
            add(f"_socket = nc_input(bsdf, {socket!r})")
            add("if _socket is not None:")
            add(f"    _socket.default_value = {float(value)!r}")
        if emission is not None:
            add("_socket = nc_input(bsdf, 'Emission Color', 'Emission')")
            add("if _socket is None:")
            add("    raise RuntimeError(" + repr(
                "Principled BSDF girisinde 'Emission Color' yok (istenen emission). "
                "Mevcut girisler: ") + " + ', '.join(nc_sockets(bsdf)))")
            add(f"_socket.default_value = {_color(emission, (1.0, 1.0, 1.0, 1.0))!r}")
        # Ham giriş adları: adı sabit kodlamak yerine doğrula ve bulunamazsa
        # mevcut girişleri listele (socket adları sürümler arasında değişiyor).
        for socket, value in (material.get("inputs") or {}).items():
            add(f"_socket = bsdf.inputs.get({str(socket)!r})")
            add("if _socket is None:")
            add("    raise RuntimeError(" + repr(
                f"Principled BSDF girisinde {str(socket)!r} yok. "
                "Mevcut girisler: ") + " + ', '.join(nc_sockets(bsdf)))")
            add("try:")
            add(f"    _socket.default_value = {value!r}")
            add("except (TypeError, ValueError):")
            add("    raise RuntimeError(" + repr(
                f"{str(socket)!r} girisi {value!r} degerini kabul etmedi; "
                "beklenen tur: sayi, renk (4'lü) veya vektor (3'lü). "
                "Mevcut girisler: ") + " + ', '.join(nc_sockets(bsdf)))")
        if material.get("alpha") is not None:
            add("try:")
            add("    material.blend_method = 'BLEND'")
            add("except Exception:")
            add("    pass")
        add("if obj.type in ('MESH', 'CURVE', 'FONT', 'SURFACE', 'META'):")
        add("    if obj.data.materials:")
        add("        obj.data.materials[0] = material")
        add("    else:")
        add("        obj.data.materials.append(material)")
        return lines

    def _render_settings_code(self, render_spec: Dict[str, Any]) -> List[str]:
        """
        Render ayarlarını uygulayan kod üretir.

        Motor kimlikleri **sabit kodlanmaz**: `nc_available_engines()` hem
        `bl_rna` enum'unu hem de atanabilir olduğu bilinen kimlikleri döndürür
        (5.2'de enum yalnızca `BLENDER_EEVEE` gösterir ama `CYCLES` atanabilir).
        Atama denenir; başarısız olursa geri düşülen motor ve **nedeni**
        `engine_note` ile bildirilir — sessizce yanlış motorda render alınmaz.

        Args:
            render_spec: `spec["render"]` sözlüğü

        Returns:
            Kod satırları
        """
        engine = str(render_spec.get("engine") or DEFAULT_ENGINE).strip().upper()
        device = str(render_spec.get("device") or DEFAULT_DEVICE).strip().upper()
        samples = int(render_spec.get("samples", DEFAULT_SAMPLES) or DEFAULT_SAMPLES)
        width, height = _normalize_resolution(
            render_spec.get("resolution", DEFAULT_RESOLUTION))
        file_format = str(render_spec.get("format") or "PNG").upper()
        frame = render_spec.get("frame")

        lines: List[str] = []
        add = lines.append
        add("# ── Render ayarları ──")
        add("engines = nc_available_engines()")
        add("engine_note = ''")
        add(f"wanted_engine = {engine!r}")
        add("if wanted_engine:")
        add("    try:")
        add("        scene.render.engine = wanted_engine")
        add("    except Exception as exc:")
        add("        fallback = engines[0]")
        add("        scene.render.engine = fallback")
        add("        engine_note = ('istenen motor ' + wanted_engine +")
        add("                       ' bu yapida atanamadi (' + type(exc).__name__ +")
        add("                       '); ' + fallback + ' kullanildi')")
        add("if scene.render.engine == 'CYCLES':")
        add("    cycles = getattr(scene, 'cycles', None)")
        add("    if cycles is not None:")
        add(f"        cycles.samples = {samples}")
        add(f"        cycles.device = {device!r}")
        add("        # CPU varsayılanı: başsız makinede ekran/GPU bağlamı gerekmez")
        add("    else:")
        add("        engine_note = (engine_note + ' | Cycles ayarlarina erisilemedi')")
        add(f"scene.render.resolution_x = {width}")
        add(f"scene.render.resolution_y = {height}")
        add("scene.render.resolution_percentage = 100")
        add("# 5.x: media_type ÖNCE, file_format SONRA ayarlanmalı")
        add(f"nc_set_image_format(scene.render.image_settings, {file_format!r})")
        if frame is not None:
            add(f"scene.frame_set({int(frame)})")
        add("resolution = [scene.render.resolution_x, scene.render.resolution_y]")
        add("engine = scene.render.engine")
        return lines

    def _generate_render_script(self, target: str, engine: str, samples: int,
                                width: int, height: int, frame: Optional[int],
                                device: str) -> str:
        """
        Render betiğini üretir.

        Dosya yolu `scene.render.filepath` ile **sabit** verilir; böylece
        Blender kare numarasını adın sonuna eklemez ve dönen yol kesin olur
        (kare `scene.frame_set` ile ayarlanır). `-o`/`-F`/`-f` komut satırı
        bayrakları hiç kullanılmaz.

        Args:
            target: Hedef dosya (mutlak)
            engine: İstenen motor kimliği
            samples: Cycles örnek sayısı
            width: Genişlik
            height: Yükseklik
            frame: Kare numarası (None = sahnenin karesi)
            device: `CPU` | `GPU`

        Returns:
            Betik kaynağı
        """
        lines: List[str] = []
        add = lines.append
        add("# ── Başsız render ──")
        add(f"target = {target!r}")
        add("scene = bpy.context.scene")
        add("engines = nc_available_engines()")
        add("engine_note = ''")
        add(f"wanted = {engine!r}")
        add("if wanted:")
        add("    try:")
        add("        scene.render.engine = wanted")
        add("    except Exception as exc:")
        add("        fallback = engines[0]")
        add("        scene.render.engine = fallback")
        add("        engine_note = ('istenen motor ' + wanted + ' atanamadi (' +")
        add("                       type(exc).__name__ + '); ' + fallback + ' kullanildi')")
        add("if scene.render.engine == 'CYCLES':")
        add("    cycles = getattr(scene, 'cycles', None)")
        add("    if cycles is not None:")
        add(f"        cycles.samples = {int(samples)}")
        add(f"        cycles.device = {device!r}")
        add(f"scene.render.resolution_x = {int(width)}")
        add(f"scene.render.resolution_y = {int(height)}")
        add("scene.render.resolution_percentage = 100")
        add("extension = os.path.splitext(target)[1].lower().lstrip('.') or 'png'")
        add("format_map = {'png': 'PNG', 'jpg': 'JPEG', 'jpeg': 'JPEG', 'bmp': 'BMP',")
        add("              'tif': 'TIFF', 'tiff': 'TIFF', 'exr': 'OPEN_EXR',")
        add("              'webp': 'WEBP'}")
        add("nc_set_image_format(scene.render.image_settings,")
        add("                     format_map.get(extension, 'PNG'))")
        add("if scene.render.image_settings.file_format == 'JPEG':")
        add("    scene.render.image_settings.quality = 90")
        if frame is not None:
            add(f"scene.frame_set({int(frame)})")
        add("camera = getattr(scene, 'camera', None)")
        add("if camera is None:")
        add("    candidates = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']")
        add("    if candidates:")
        add("        scene.camera = candidates[0]")
        add("        camera = candidates[0]")
        add("if camera is None:")
        add("    raise RuntimeError('sahnede kamera yok - render alinamaz')")
        add("scene.render.filepath = target")
        add("render_started = time.time()")
        add("bpy.ops.render.render(write_still=True)")
        add("render_ms = (time.time() - render_started) * 1000.0")
        add("written = scene.render.filepath")
        add("if not os.path.isfile(written) and os.path.isfile(bpy.path.abspath(written)):")
        add("    written = bpy.path.abspath(written)")
        add("if not os.path.isfile(written):")
        add("    raise RuntimeError('render dosyasi yazilmadi: ' + str(written))")
        add("size = os.path.getsize(written)")
        add("if size == 0:")
        add("    raise RuntimeError('render dosyasi bos: ' + str(written))")
        add("print('" + RESULT_MARKER + "' + ' ' + json.dumps({")
        add("    'file': written,")
        add("    'size': size,")
        add("    'engine': scene.render.engine,")
        add("    'engines_available': engines,")
        add("    'engine_note': engine_note,")
        add("    'device': getattr(getattr(scene, 'cycles', None), 'device', ''),")
        add("    'samples': getattr(getattr(scene, 'cycles', None), 'samples', 0),")
        add("    'resolution': [scene.render.resolution_x, scene.render.resolution_y],")
        add("    'frame': scene.frame_current,")
        add("    'render_ms': render_ms,")
        add("    'blender_version': bpy.app.version_string,")
        add("}, default=str))")
        return "\n".join(lines) + "\n"

    # ─── Yardımcılar ────────────────────────────────────────────────────

    def _validate_spec(self, spec: Dict[str, Any]) -> List[str]:
        """
        Sahne tanımındaki olası sorunları **Blender'ı başlatmadan** bildirir.

        Sessizce yanlış sahne kurmak yerine uyarı üretmek ajanın döngüsünü
        kısaltır (ör. hiç kamera yoksa render alınamaz).

        Args:
            spec: Sahne tanımı

        Returns:
            Uyarı metinleri
        """
        warnings: List[str] = []
        objects = spec.get("objects")
        if not isinstance(objects, list) or not objects:
            warnings.append("'objects' boş veya liste değil — sahne boş kalacak")
            objects = []

        known = {"cube", "sphere", "cylinder", "plane", "torus", "text", "empty",
                 "camera", "sun", "point_light", "area_light", "spot", "light"}
        types = [str((item or {}).get("type", "cube")).lower()
                 for item in objects if isinstance(item, dict)]
        unknown = sorted({kind for kind in types if kind not in known})
        if unknown:
            warnings.append("bilinmeyen nesne türleri boş nesne olarak eklenecek: "
                            + ", ".join(unknown))
        if "camera" not in types and not spec.get("camera"):
            warnings.append("tanımda kamera yok — render=True ise hata verir; "
                            "{\"type\": \"camera\"} ekleyin")
        if not any(kind in types for kind in ("sun", "point_light", "area_light",
                                              "spot", "light")):
            warnings.append("tanımda ışık yok — render karanlık çıkabilir")
        render_spec = spec.get("render") or {}
        if isinstance(render_spec, dict) and render_spec.get("engine"):
            engine = str(render_spec["engine"]).upper()
            if engine not in KNOWN_ENGINES:
                warnings.append(f"bilinmeyen motor: {engine} (bilinenler: "
                                f"{', '.join(KNOWN_ENGINES)})")
        for key in ("world", "render"):
            if key in spec and not isinstance(spec[key], dict):
                warnings.append(f"'{key}' sözlük olmalı")
        return warnings

    def _resolve_dir(self, path: str) -> Path:
        """Dizini mutlak yola çevirir (boş = çalışma alanı kökü)."""
        if not str(path or "").strip():
            return self._workspace or Path.cwd()
        return _absolute(str(path), self._workspace)

    def _resolve_file(self, path: str, default_name: str) -> Path:
        """Çıktı dosyası yolunu çözer (boş = çalışma alanı/varsayılan ad)."""
        if not str(path or "").strip():
            return (self._workspace or Path.cwd()) / default_name
        return _absolute(str(path), self._workspace)

    def _optional_path(self, path: Any) -> Optional[Path]:
        """
        Yolu mutlak `Path`'e çevirir; boş veya var olmayan yolda `None` döner.

        Boş değer "isteğe bağlı argüman verilmedi", dolu ama var olmayan değer
        "hata" anlamına gelir; çağıran taraf ikisini ayırt eder. Böylece hiçbir
        metot istisna atmaz.
        """
        text = str(path or "").strip()
        if not text:
            return None
        candidate = _absolute(text, self._workspace)
        return candidate if candidate.exists() else None

    def _write_temp_script(self, code: str, name: str) -> Optional[Path]:
        """Üretilen betiği geçici dosyaya yazar (hata ayıklama izi kalır)."""
        root = self._temp_root or Path(tempfile.gettempdir())
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            root = Path(tempfile.gettempdir())
        stamp = f"{int(time.time() * 1000) % 100000000:08d}"
        safe = re.sub(r"[^\w.-]", "_", str(name or "script"))[:48] or "script"
        path = root / f"{safe}_{stamp}.py"
        try:
            path.write_text(code, encoding="utf-8", newline="")
        except OSError as exc:
            logger.error(f"Betik yazılamadı ({path}): {exc}")
            return None
        return path

    def _collect_artifacts(self, out_dir: Path, since: float) -> List[Dict[str, Any]]:
        """
        Çıktı dizininde **gerçekten var olan** yeni/değişmiş dosyaları listeler.

        Args:
            out_dir: İzlenecek dizin
            since: Bu zaman damgasından sonra değişenler

        Returns:
            [{"path", "name", "size", "modified"}] — boyuta göre azalan
        """
        artifacts: List[Dict[str, Any]] = []
        try:
            for item in out_dir.iterdir():
                if not item.is_file():
                    continue
                try:
                    stat = item.stat()
                except OSError:
                    continue
                if stat.st_mtime < since or stat.st_size == 0:
                    continue
                artifacts.append({"path": str(item), "name": item.name,
                                  "size": stat.st_size, "modified": stat.st_mtime})
        except OSError as exc:
            logger.debug(f"Yapıtlar listelenemedi ({out_dir}): {exc}")
        artifacts.sort(key=lambda entry: -entry["size"])
        return artifacts

    def _detect_executable(self) -> str:
        """
        `blender.exe` yolunu bulur.

        Sıra: yapılandırma → ortam değişkeni → PATH → bilinen yol → sınırlı
        kurulum taraması. Blender bu makinede PATH'te **değildir**; bu yüzden
        doğrulanmış tam yol erken denenir. `blender-launcher.exe` **elenir**.
        """
        for attribute in ("blender_path", "blender_executable", "blender_exe"):
            candidate = getattr(self._config, attribute, "") if self._config else ""
            if candidate and Path(str(candidate)).exists():
                return str(Path(str(candidate)).resolve())

        for variable in ("NC_BLENDER", "BLENDER_EXE", "BLENDER_PATH"):
            candidate = os.environ.get(variable, "")
            if candidate and Path(candidate).exists():
                return str(Path(candidate).resolve())

        found = shutil.which("blender")
        if found and "launcher" not in os.path.basename(found).lower():
            return found

        if Path(DEFAULT_EXECUTABLE).exists():
            return DEFAULT_EXECUTABLE

        if os.name == "nt":
            for root in SEARCH_ROOTS:
                base = Path(root)
                if not base.is_dir():
                    continue
                try:
                    exact = base / "blender.exe"
                    if exact.exists():
                        return str(exact)
                    for candidate in sorted(base.glob("Blender*/blender.exe"))[:3]:
                        if "launcher" not in candidate.name.lower():
                            return str(candidate)
                except OSError:
                    continue
        return ""

    @staticmethod
    def _failure(message: str, hint: str = "") -> Dict[str, Any]:
        """
        Nezaketli bozulma için standart hata sözlüğü üretir.

        Args:
            message: Hata metni
            hint: Ne yapmalı

        Returns:
            {"ok": False, ...} — hiçbir metot istisna atmaz
        """
        return {
            "ok": False, "error": str(message), "hint": hint,
            "exit_code": -1, "stdout": "", "stderr": "", "result": None,
            "errors": ([{"kind": "blender_error", "message": str(message),
                         "hint": hint}] if message else []),
            "duration_ms": 0.0, "script_path": "", "artifacts": [],
            "timed_out": False, "command": "",
        }

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla (event bus yoksa sessizce yut)."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    # ─── Araç Tanımları ─────────────────────────────────────────────────

    def tools(self) -> List[ToolDef]:
        """
        Blender araçlarının tam listesi (`ToolRegistry`'ye kaydedilmek üzere).

        Her araç `ToolContext` alır ve iptal bayrağını denetler. Güvenlik
        sınıfları: okuma `READ`, `.blend` üretenler `WRITE`, Blender süreci
        başlatanlar `EXECUTE`.

        Returns:
            `ToolDef` listesi
        """
        controller = self

        def wrap(name: str, description: str, safety: Safety,
                 parameters: List[ToolParameter], returns: str = "",
                 tags: Optional[List[str]] = None) -> ToolDef:
            """Bağlam alan bir araç sarmalayıcısı üretir."""
            method = getattr(controller, name.split(".")[-1].replace("-", "_"), None)

            def handler(ctx: ToolContext, **kwargs: Any) -> Any:
                if ctx.cancelled():
                    raise BlenderError("işlem kullanıcı tarafından iptal edildi")
                if method is None:
                    raise BlenderError(f"metot bulunamadı: {name}")
                return method(**kwargs)

            return ToolDef(
                name=name, description=description, handler=handler,
                parameters=parameters, safety=safety, category="blender",
                returns=returns, tags=tags or [],
            )

        from neurocore.agent.tools import ToolParameter as P

        return [
            wrap("blender.health",
                 "Blender kurulumunu denetler: çalıştırılabilir yol, sürüm, "
                 "atanabilen render motorları, Blender Python'unda hazır modüller. "
                 "Blender işlerine başlamadan ÖNCE çağırın.",
                 Safety.READ, [], "kurulum durumu", ["kurulum", "saglik"]),

            wrap("blender.version",
                 "Blender sürüm metnini döndürür (ör. '5.2.2 LTS').",
                 Safety.READ, [], "sürüm", ["kurulum"]),

            wrap("blender.python_environment",
                 "Blender'ın izole Python ortamını raporlar (Python sürümü, "
                 "sys.executable, sys.path, numpy/PIL/requests/scipy). Sistem "
                 "Python'unuzun site-packages dizini Blender'dan GÖRÜNMEZ.",
                 Safety.READ, [], "python ortamı", ["kurulum", "python"]),

            wrap("blender.run_script",
                 "Blender içinde Python kodu çalıştırır (başsız, "
                 "--python-exit-code 1 + zorunlu try/except sarmalayıcısı). Çıktı "
                 "dizini alt sürece NC_BLEND_OUT olarak geçer; yolları koda "
                 "gömmek yerine os.environ['NC_BLEND_OUT'] kullanın. Varsayılan "
                 "threads=0 (Blender seçer); kalabalık makinede threads=2 verin.",
                 Safety.EXECUTE,
                 [P("code", required=True, description="Blender Python kodu"),
                  P("blend_file", description="Önce yüklenecek .blend"),
                  P("timeout", "number", "Saniye", default=600),
                  P("factory_startup", "boolean", "Fabrika ayarlarıyla başlat",
                    default=True),
                  P("out_dir", description="Çıktı dizini (NC_BLEND_OUT)"),
                  P("threads", "integer", "İş parçacığı sınırı (0 = Blender seçer)",
                    default=0)],
                 "stdout, NC_RESULT, yapıtlar", ["betik", "calistirma"]),

            wrap("blender.run_expression",
                 "Kısa bir Python ifadesini Blender içinde çalıştırır ve JSON "
                 "değerini döndürür (hızlı sorgular için).",
                 Safety.EXECUTE,
                 [P("expression", required=True, description="Python ifadesi"),
                  P("blend_file", description="Önce yüklenecek .blend"),
                  P("timeout", "number", "Saniye", default=300)],
                 "ifadenin değeri", ["sorgu"]),

            wrap("blender.build_scene",
                 "JSON sahne tanımından .blend üretir: küp/küre/silindir/düzlem/"
                 "halka/metin/boş nesne, kamera (look_at destekli), güneş/nokta/"
                 "alan ışığı, dünya arka planı, render ayarları, malzeme (dostu "
                 "giriş adları 5.x soket adlarına eşlenir). render=true ile aynı "
                 "çağrıda render alır.",
                 Safety.WRITE,
                 [P("spec", "object", "Sahne tanımı: objects, world, render, "
                    "camera, save, clear", required=True),
                  P("out_dir", description="Çıktı dizini"),
                  P("blend_name", description="Dosya adı", default="scene.blend"),
                  P("render", "boolean", "Kaydettikten sonra render al",
                    default=False)],
                 "blend_file, nesneler, render dosyası", ["sahne", "kurulum"]),

            wrap("blender.render",
                 "Bir .blend dosyasını başsız render eder. VARSAYILAN CYCLES + "
                 "CPU'dur: başsız makinede ekran/GPU bağlamı gerektirmez (ölçüm: "
                 "Cycles CPU ~0.2 s, EEVEE başsız ~11 s aynı kare için). EEVEE "
                 "GPU'su olmayan/VM/RDP makinede kırılgandır. Dosyanın gerçekten "
                 "oluştuğunu ve boş olmadığını bildirir. threads ile makineyi "
                 "duyarlı tutun.",
                 Safety.EXECUTE,
                 [P("blend_file", description="Render edilecek .blend"),
                  P("output", description="Çıktı dosyası (.png)"),
                  P("engine", description="CYCLES | BLENDER_EEVEE | "
                    "BLENDER_WORKBENCH | boş (=CYCLES)"),
                  P("samples", "integer", "Cycles örnek sayısı", default=32),
                  P("resolution", "array", "Çözünürlük [genişlik, yükseklik]"),
                  P("frame", "integer", "Kare numarası"),
                  P("device", description="CPU | GPU", default="CPU"),
                  P("threads", "integer", "İş parçacığı sınırı (0 = Blender seçer)",
                    default=0),
                  P("timeout", "number", "Saniye", default=1800)],
                 "render dosyası, boyut, süre", ["render"]),

            wrap("blender.inspect_blend",
                 "Bir .blend dosyasının JSON özetini çıkarır: dosyayı kaydeden "
                 "sürüm, sahneler, nesneler (konum/boyut), malzemeler ve temel "
                 "renkleri, kameralar, ışıklar, koleksiyonlar, render ayarları, "
                 "kare aralığı, tepe/yüz sayıları.",
                 Safety.READ,
                 [P("blend_file", required=True, description="İncelenecek .blend")],
                 "sahne özeti", ["inceleme", "analiz"]),

            wrap("blender.list_objects",
                 "Bir .blend içindeki nesneleri listeler (ad, tür, konum, boyut, "
                 "ebeveyn, görünürlük). inspect_blend'in hafif sürümü.",
                 Safety.READ,
                 [P("blend_file", description=".blend yolu (boş = boş sahne)")],
                 "nesne listesi", ["inceleme"]),

            wrap("blender.export_gltf",
                 "Sahneyi glTF 2.0 (.glb) olarak dışa aktarır "
                 "(export_format='GLB', export_yup, export_materials).",
                 Safety.WRITE,
                 [P("blend_file", description="Kaynak .blend"),
                  P("output", description="Çıktı .glb yolu"),
                  P("timeout", "number", "Saniye", default=600)],
                 "dışa aktarma sonucu", ["disa-aktarma", "gltf"]),

            wrap("blender.export_fbx",
                 "Sahneyi FBX olarak dışa aktarır (use_selection, path_mode, "
                 "bake_space_transform). Not: FBX'te export_format ve "
                 "export_selected argümanları YOKTUR.",
                 Safety.WRITE,
                 [P("blend_file", description="Kaynak .blend"),
                  P("output", description="Çıktı .fbx yolu"),
                  P("timeout", "number", "Saniye", default=600)],
                 "dışa aktarma sonucu", ["disa-aktarma", "fbx"]),

            wrap("blender.validate_script",
                 "Blender betiğini Blender'ı BAŞLATMADAN doğrular (sistem "
                 "Python'unun ast modülü) ve zorunlu NC_ERROR + sys.exit(1) "
                 "sarmalayıcısının eksik olup olmadığını bildirir.",
                 Safety.READ,
                 [P("code", required=True, description="Doğrulanacak kod")],
                 "sözdizimi sonucu", ["dogrulama"]),

            wrap("blender.diagnose",
                 "Blender hata çıktısını yorumlar: poll() hatası, geçersiz enum "
                 "(BLENDER_EEVEE_NEXT), eksik operatör, düğüm girişi (socket) adı "
                 "değişikliği, eksik dosya, Blender'ın izole Python'unda "
                 "bulunmayan modüller, GPU/CUDA, bellek, çökme.",
                 Safety.READ,
                 [P("text", required=True, description="Blender çıktısı"),
                  P("command", description="Çalıştırılan komut (bağlam)")],
                 "hata analizi", ["hata", "teshis"]),
        ]

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<BlenderController executable={self._executable or '-'} "
                f"scripts={self._counter['scripts']} "
                f"renders={self._counter['renders']}>")


# ─── Gömülü Betikler ────────────────────────────────────────────────────────

_ENGINE_PROBE_SCRIPT = '''
# ── Atanabilen motorları keşfet (enum eksik gösterebilir) ──
settable = []
errors = []
for candidate in nc_available_engines():
    try:
        bpy.context.scene.render.engine = candidate
        settable.append(candidate)
    except Exception as exc:
        errors.append({candidate: type(exc).__name__})
if settable:
    bpy.context.scene.render.engine = 'CYCLES' if 'CYCLES' in settable else settable[0]
print('NC_RESULT ' + json.dumps({
    'settable': settable,
    'errors': errors,
    'enum_listed': [item.identifier for item in
                    bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items],
    'current': bpy.context.scene.render.engine,
}))
'''

#: Blender'ın izole Python ortamını raporlayan betik.
#: `{modules}` yer tutucusu `PROBE_MODULES` ile doldurulur (bkz. `python_environment`).
_PYTHON_ENV_TEMPLATE = '''
# ── Blender'ın izole Python ortamı ──
mods = {{}}
for probe_name in {modules}:
    try:
        module = __import__(probe_name)
        mods[probe_name] = getattr(module, '__version__', 'var')
    except Exception:
        mods[probe_name] = ''
print('NC_RESULT ' + json.dumps({{
    'blender_version': bpy.app.version_string,
    'blender_version_tuple': list(bpy.app.version),
    'python_version': sys.version.split()[0],
    'python_version_tuple': list(sys.version_info[:3]),
    'executable': sys.executable,
    'binary_path': bpy.app.binary_path,
    'prefix': sys.prefix,
    'path': [item for item in sys.path if item],
    'modules': mods,
}}))
'''

_LIST_OBJECTS_SCRIPT = '''
# ── Nesne listesi (hafif) ──
items = []
for obj in bpy.data.objects:
    items.append({
        'name': obj.name,
        'type': obj.type,
        'location': [round(float(v), 5) for v in obj.location],
        'rotation': [round(float(v), 5) for v in obj.rotation_euler],
        'scale': [round(float(v), 5) for v in obj.scale],
        'dimensions': [round(float(v), 5) for v in obj.dimensions],
        'parent': obj.parent.name if obj.parent else '',
        'visible': bool(obj.visible_get()),
    })
print('NC_RESULT ' + json.dumps({
    'objects': items,
    'filepath': bpy.data.filepath,
    'blender_version': bpy.app.version_string,
}))
'''

_INSPECT_SCRIPT = '''
# ── .blend inceleme özeti ──
data = bpy.data
scene = bpy.context.scene

objects = []
vertex_total = 0
face_total = 0
for obj in data.objects:
    entry = {
        'name': obj.name,
        'type': obj.type,
        'location': [round(float(v), 5) for v in obj.location],
        'rotation': [round(float(v), 5) for v in obj.rotation_euler],
        'scale': [round(float(v), 5) for v in obj.scale],
        'dimensions': [round(float(v), 5) for v in obj.dimensions],
        'parent': obj.parent.name if obj.parent else '',
        'collections': [collection.name for collection in obj.users_collection],
        'materials': [slot.material.name for slot in obj.material_slots
                      if slot.material],
        'visible': bool(obj.visible_get()),
        'vertex_count': 0,
        'face_count': 0,
    }
    mesh = getattr(obj, 'data', None)
    if obj.type == 'MESH' and mesh is not None:
        entry['vertex_count'] = len(mesh.vertices)
        entry['face_count'] = len(mesh.polygons)
        vertex_total += entry['vertex_count']
        face_total += entry['face_count']
    objects.append(entry)

materials = []
for material in data.materials:
    # NOT: use_nodes okunmaz — 5.2'de okumak bile DeprecationWarning üretir.
    base_color = [round(float(v), 5) for v in material.diffuse_color]
    principled = None
    node_names = []
    tree = getattr(material, 'node_tree', None)
    if tree is not None:
        node_names = [node.name for node in tree.nodes]
        for node in tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                socket = node.inputs.get('Base Color')
                if socket is not None:
                    principled = [round(float(v), 5) for v in socket.default_value]
                break
    materials.append({
        'name': material.name,
        'base_color': base_color,
        'principled_base_color': principled,
        'nodes': node_names,
        'users': material.users,
    })

cameras = []
for obj in data.objects:
    if obj.type == 'CAMERA':
        cameras.append({
            'name': obj.name,
            'lens': round(float(obj.data.lens), 4),
            'location': [round(float(v), 5) for v in obj.location],
            'is_scene_camera': bool(scene.camera is obj),
        })

lights = []
for obj in data.objects:
    if obj.type == 'LIGHT':
        lights.append({
            'name': obj.name,
            'light_type': obj.data.type,
            'energy': round(float(obj.data.energy), 4),
            'color': [round(float(v), 4) for v in obj.data.color],
        })

report = {
    'filepath': data.filepath,
    'saved_by_version': list(data.version),
    'blender_version': bpy.app.version_string,
    'blender_version_tuple': list(bpy.app.version),
    'scenes': [{'name': item.name,
                'object_count': len(item.objects),
                'frame_start': item.frame_start,
                'frame_end': item.frame_end,
                'frame_current': item.frame_current} for item in data.scenes],
    'scene': {
        'name': scene.name,
        'engine': scene.render.engine,
        'engines_available': nc_available_engines(),
        'resolution': [scene.render.resolution_x, scene.render.resolution_y],
        'resolution_percentage': scene.render.resolution_percentage,
        'file_format': scene.render.image_settings.file_format,
        'filepath': scene.render.filepath,
        'frame_start': scene.frame_start,
        'frame_end': scene.frame_end,
        'frame_current': scene.frame_current,
        'fps': scene.render.fps,
        'camera': scene.camera.name if scene.camera else '',
        'cycles_samples': getattr(getattr(scene, 'cycles', None), 'samples', 0),
        'cycles_device': getattr(getattr(scene, 'cycles', None), 'device', ''),
        'world': scene.world.name if scene.world else '',
    },
    'objects': objects,
    'materials': materials,
    'cameras': cameras,
    'lights': lights,
    'collections': [{'name': item.name,
                     'object_count': len(item.objects),
                     'children': [child.name for child in item.children]}
                    for item in data.collections],
    'counts': {
        'objects': len(data.objects),
        'meshes': len(data.meshes),
        'materials': len(data.materials),
        'cameras': len(cameras),
        'lights': len(lights),
        'collections': len(data.collections),
        'images': len(data.images),
        'actions': len(data.actions),
        'vertices': vertex_total,
        'faces': face_total,
    },
}
print('NC_RESULT ' + json.dumps(report, default=str))
'''

#: glTF dışa aktarma — argümanlar 5.2 operatör RNA'sından doğrulandı.
_EXPORT_GLTF_SCRIPT = '''
# ── glTF 2.0 (.glb) dışa aktarma ──
target = {target}
module = getattr(bpy.ops, 'export_scene', None)
if module is None or not hasattr(module, 'gltf'):
    raise RuntimeError('operator yok: bpy.ops.export_scene.gltf - io_scene_gltf2 '
                       'eklentisi etkin degil')
options = {{
    'filepath': target,
    'export_format': 'GLB',
    'use_selection': False,
    'export_apply': True,
    'export_yup': True,
    'export_materials': 'EXPORT',
    'export_animations': False,
}}
try:
    bpy.ops.export_scene.gltf(**options)
except TypeError as exc:
    # Bazi surumlerde ek parametreler yok: yalnizca zorunlu olanlarla dene
    bpy.ops.export_scene.gltf(filepath=target, export_format='GLB')
if not os.path.isfile(target) or os.path.getsize(target) == 0:
    raise RuntimeError('disa aktarma dosyasi olusmadi veya bos: ' + str(target))
print('NC_RESULT ' + json.dumps({{
    'file': target,
    'size': os.path.getsize(target),
    'format': 'GLB',
    'objects': [obj.name for obj in bpy.context.scene.collection.all_objects],
    'blender_version': bpy.app.version_string,
}}, default=str))
'''

#: FBX dışa aktarma — `export_format`, `export_selected`, `use_animations` YOK.
_EXPORT_FBX_SCRIPT = '''
# ── FBX dışa aktarma ──
target = {target}
module = getattr(bpy.ops, 'export_scene', None)
if module is None or not hasattr(module, 'fbx'):
    raise RuntimeError('operator yok: bpy.ops.export_scene.fbx - io_scene_fbx '
                       'eklentisi etkin degil')
options = {{
    'filepath': target,
    'use_selection': False,
    'path_mode': 'COPY',
    'embed_textures': False,
    'bake_space_transform': False,
    'apply_scale_options': 'FBX_SCALE_NONE',
    'axis_forward': '-Z',
    'axis_up': 'Y',
}}
try:
    bpy.ops.export_scene.fbx(**options)
except TypeError as exc:
    bpy.ops.export_scene.fbx(filepath=target)
if not os.path.isfile(target) or os.path.getsize(target) == 0:
    raise RuntimeError('disa aktarma dosyasi olusmadi veya bos: ' + str(target))
print('NC_RESULT ' + json.dumps({{
    'file': target,
    'size': os.path.getsize(target),
    'format': 'FBX',
    'objects': [obj.name for obj in bpy.context.scene.collection.all_objects],
    'blender_version': bpy.app.version_string,
}}, default=str))
'''


# ─── Modül Düzeyi Yardımcılar ───────────────────────────────────────────────

def _absolute(text: str, workspace: Optional[Path]) -> Path:
    """
    Metni mutlak `Path`'e çevirir (ortam değişkenleri ve `~` genişletilir).

    Args:
        text: Yol metni
        workspace: Göreli yolların çözüleceği kök

    Returns:
        Mutlak yol (çözümleme başarısızsa genişletilmiş aday)
    """
    candidate = Path(os.path.expandvars(os.path.expanduser(str(text))))
    if not candidate.is_absolute():
        candidate = (workspace or Path.cwd()) / candidate
    try:
        return candidate.resolve()
    except OSError:  # pragma: no cover
        return candidate


def _vector(value: Any, default: Tuple[float, ...]) -> Tuple[float, ...]:
    """
    Bir değeri sayı demetine çevirir (eksik/bozuk girişte varsayılana düşer).

    Args:
        value: Liste/demet/sayı veya None
        default: Varsayılan demet (uzunluk belirleyici)

    Returns:
        Varsayılanla aynı uzunlukta sayı demeti
    """
    if value is None:
        return tuple(float(item) for item in default)
    if isinstance(value, (int, float)):
        return (float(value),) * len(default)
    result: List[float] = []
    try:
        for index, item in enumerate(value):
            if index >= len(default):
                break
            result.append(float(item))
    except (TypeError, ValueError):
        return tuple(float(item) for item in default)
    while len(result) < len(default):
        result.append(float(default[len(result)]))
    return tuple(result)


def _color(value: Any, default: Tuple[float, ...]) -> Tuple[float, ...]:
    """
    RGBA rengi normalize eder (3 bileşen verilirse alfa 1.0 yapılır).

    Args:
        value: `[r, g, b]` veya `[r, g, b, a]`
        default: Varsayılan renk

    Returns:
        4 bileşenli demet
    """
    if value is None:
        return tuple(float(item) for item in default)
    try:
        items = [float(item) for item in value]
    except (TypeError, ValueError):
        return tuple(float(item) for item in default)
    if len(items) == 3:
        items.append(1.0)
    while len(items) < 4:
        items.append(1.0)
    return tuple(items[:4])


def _normalize_resolution(resolution: Any) -> Tuple[int, int]:
    """Çözünürlüğü (genişlik, yükseklik) tam sayı çiftine çevirir."""
    try:
        width, height = int(resolution[0]), int(resolution[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return DEFAULT_RESOLUTION
    return (max(4, width), max(4, height))


def _quote_argument(item: Any) -> str:
    """
    Kabuğa verilecek bir argümanı tırnaklar.

    PowerShell ve `cmd` çift tırnak kabul eder; yol sonundaki ters bölü ile
    çift tırnak birleşimi (`..."`) kaçış sorunu yarattığı için sondaki ters
    bölü `os.path.normpath` ile temizlenir.

    Args:
        item: Argüman

    Returns:
        Tırnaklanmış metin
    """
    text = str(item)
    if os.name == "nt":
        return '"' + text.replace('"', '\\"') + '"'
    return "'" + text.replace("'", "'\\''") + "'"


def _build_command(executable: str, arguments: Sequence[str],
                   environment: Optional[Dict[str, str]] = None) -> str:
    """
    Kabuğa verilecek komut metnini üretir.

    Ortam değişkenleri komutun başına kabuk sözdizimiyle eklenir; nedeni:
    `ShellIntegration.run` imzası `env` kabul etmez, ancak alt sürece değer
    aktarmak gerekir. Değer tırnaklanır, böylece `A; B` gibi bir değer komut
    enjeksiyonuna dönüşmez. `ShellIntegration` her komutu tek bir
    `powershell -Command` çağrısına çevirdiği için bu önekler alt süreç
    ortamına gerçekten geçer (bu makinede doğrulandı).

    Args:
        executable: `blender.exe` yolu
        arguments: Argüman listesi (sıra korunur)
        environment: Eklenecek ortam değişkenleri

    Returns:
        Kabuk komutu
    """
    prefix = ""
    for key, value in (environment or {}).items():
        if os.name == "nt":
            quoted = "'" + str(value).replace("'", "''") + "'"
            prefix += f"$env:{key}={quoted}; "
        else:
            quoted = "'" + str(value).replace("'", "'\\''") + "'"
            prefix += f"{key}={quoted} "
    joined = " ".join(_quote_argument(item) for item in arguments)
    return f"{prefix}& {_quote_argument(executable)} {joined}".strip()


def _kill_lingering_blender(arguments: Sequence[str]) -> List[int]:
    """
    Zaman aşımından sonra **yetim kalan** Blender süreçlerini sonlandırır.

    Neden gerekli? `ShellIntegration.run` zaman aşımında PowerShell sürecini
    sonlandırır, ancak `blender.exe` o PowerShell'in **çocuğudur** ve yetim
    kalır: arka planda çalışmaya devam eder ve kullanıcının makinesini meşgul
    eder. Bu proje kullanıcının bilgisayarını doyurmama kuralına tabidir.

    Güvenlik: **yalnızca** komut satırında bu çağrıya ait geçici betik dosyası
    (`--python <script.py>`) geçen süreçler sonlandırılır. Kullanıcının kendi
    Blender oturumları (farklı komut satırı) **etkilenmez**. Yetim süreç
    bulunamazsa hiçbir şey yapılmaz; bu fonksiyon asla istisna atmaz.

    Args:
        arguments: Çalıştırılan Blender argüman listesi

    Returns:
        Sonlandırılan süreç kimlikleri (boş liste = yetim yok)
    """
    if os.name != "nt":
        return []
    script = ""
    for index, item in enumerate(arguments):
        if str(item) == "--python" and index + 1 < len(arguments):
            script = str(arguments[index + 1])
            break
    if not script or "nc_" not in os.path.basename(script):
        return []
    killed: List[int] = []
    # WMI/CIM sorgusu ek derleme gerektirmez ve her Windows'ta bulunur: yalnızca
    # komut satırında BU çağrının geçici betiği geçen Blender süreçleri bulunur.
    query = ("Get-CimInstance Win32_Process -Filter \"Name='blender.exe'\" | "
             "Where-Object {{ $_.CommandLine -like '*{name}*' }} | "
             "ForEach-Object {{ $_.ProcessId }}").format(
                 name=os.path.basename(script))
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if not powershell:
        return []
    try:
        found = subprocess.run(
            [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", query],
            capture_output=True, text=True, timeout=30.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"Yetim Blender aranamadı: {exc}")
        return []
    for line in (found.stdout or "").splitlines():
        line = line.strip()
        if not line.isdigit():
            continue
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", line],
                           capture_output=True, text=True, timeout=15.0,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            killed.append(int(line))
        except (OSError, subprocess.SubprocessError):
            continue
    if killed:
        logger.warning("Zaman aşımından sonra yetim Blender süreçleri "
                       f"sonlandırıldı: {killed}")
    return killed


def _parse_markers(stdout: str, stderr: str) -> Tuple[Any, List[str]]:
    """
    Çıktıdaki `NC_RESULT` / `NC_ERROR` satırlarını ayrıştırır.

    Blender kendi bilgi satırlarını da basar ("Blend loaded", "Saved ..."), bu
    yüzden ayrıştırma satır bazlıdır ve JSON kesilmişse ilk geçerli JSON
    değerine geri düşer.

    Args:
        stdout: Süreç standart çıktısı
        stderr: Süreç hata çıktısı

    Returns:
        (result, error_lines)
    """
    result: Any = None
    errors: List[str] = []
    for raw_line in str(stdout or "").splitlines():
        line = raw_line.strip()
        if result is None and RESULT_MARKER in line:
            payload = line.split(RESULT_MARKER, 1)[1].strip()
            if payload:
                parsed = _parse_json(payload)
                result = parsed if parsed is not None else {"_raw": payload[:2000]}
        if line.startswith(ERROR_MARKER):
            errors.append(line[len(ERROR_MARKER):].strip())
    for raw_line in str(stderr or "").splitlines():
        line = raw_line.strip()
        if line.startswith(ERROR_MARKER):
            errors.append(line[len(ERROR_MARKER):].strip())
    return result, errors


def _parse_json(text: str) -> Any:
    """
    Metni JSON olarak çözer; kesilmiş/karışık çıktıda ilk değeri bulur.

    Args:
        text: JSON adayı

    Returns:
        Çözülen değer veya `None`
    """
    candidate = str(text or "").strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        pass
    decoder = json.JSONDecoder()
    for start, character in enumerate(candidate):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[start:])
            return value
        except ValueError:
            continue
    return None


def _last_traceback_line(text: str) -> str:
    """İzleme kaydındaki son anlamlı satırı (hata mesajını) bulur."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if re.match(r"^[A-Za-z_.]*(Error|Exception)\b", line):
            return line[:300]
    return lines[-1][:300] if lines else ""


def _skip_version_line(text: str) -> bool:
    """
    Satırın `--version` sürüm satırı **olmadığını** kaba biçimde anlar.

    Blender `--version` çıktısının ilk satırı sürüm satırıdır; ancak bazı
    kurulumlarda önce günlük/uyarı satırları basılabilir.

    Args:
        text: Satır

    Returns:
        Satır atlanmalı mı?
    """
    stripped = str(text or "").strip()
    if not stripped:
        return True
    return bool(re.match(r"^(\d{2}:\d{2}:\d{2}|Writing|Warning|Error|Read blend|"
                         r"Blender quit|Color management)", stripped))


def _first_version_line(text: str) -> str:
    """
    `--version` çıktısından sürüm metnini çıkarır.

    Blender ``Blender 5.2.2 LTS`` yazar; bu metot "Blender" önekini atar
    (ör. ``5.2.2 LTS``).

    Args:
        text: `--version` çıktısı

    Returns:
        Sürüm metni (bulunamazsa boş)
    """
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if _skip_version_line(stripped):
            continue
        if stripped.lower().startswith("blender"):
            return stripped[len("Blender"):].strip() or stripped
    return ""


def probe_blender(executable: str = "") -> Dict[str, Any]:
    """
    Blender kurulumunu **denetleyici kurmadan** sınar (yetenek keşfi).

    Args:
        executable: `blender.exe` yolu (boş = bilinen yollar denenir)

    Returns:
        {"available", "executable", "version", "searched"}
    """
    candidate = executable
    if not candidate:
        found = shutil.which("blender")
        if found and "launcher" not in os.path.basename(found).lower():
            candidate = found
        elif Path(DEFAULT_EXECUTABLE).exists():
            candidate = DEFAULT_EXECUTABLE
        elif os.name == "nt":
            for root in SEARCH_ROOTS:
                base = Path(root)
                if not base.is_dir():
                    continue
                try:
                    if (base / "blender.exe").exists():
                        candidate = str(base / "blender.exe")
                        break
                    matches = [item for item in sorted(base.glob("Blender*/blender.exe"))
                               if "launcher" not in item.name.lower()]
                    if matches:
                        candidate = str(matches[0])
                        break
                except OSError:
                    continue

    if not candidate or not Path(candidate).exists():
        return {"available": False, "executable": "", "version": "",
                "searched": list(SEARCH_ROOTS)}

    controller = BlenderController()
    controller._executable = candidate
    version = controller.version()
    return {"available": bool(version), "executable": candidate,
            "version": version, "searched": list(SEARCH_ROOTS)}


__all__ = [
    "BlenderController",
    "BlenderError",
    "probe_blender",
    "RESULT_MARKER",
    "ERROR_MARKER",
    "DEFAULT_EXECUTABLE",
    "DEFAULT_ENGINE",
    "DEFAULT_DEVICE",
    "KNOWN_ENGINES",
]
