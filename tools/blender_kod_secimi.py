"""
Blender Kod Seçimi Ölçümü
==========================
Yerel modelin **pratik** yeteneğini ölçer: doğru Blender kodunu seçebiliyor mu?

Neden bu test? Modelin akıcı metin üretemediği ölçüldü. Ama bir model
üretemese bile **doğru seçeneği tanıyabilir** — ve bu, ajan mimarisinde
işe yarar: kod önerileri arasından en uygununu elemek.

Test biçimi (çoktan seçmeli, uydurma yok):
    İstem : "Blender'da sahneyi temizlemek için hangi kod?"
    Adaylar: 4 gerçek kod parçası (1 doğru + 3 çeldirici)

Skor, modelin **kendi log olasılığıyla** hesaplanır
(`NeuroCoreModel.sequence_logprob`); kelime torbası sezgisi kullanılmaz.

Rastgele taban: %25. Anlamlı sonuç: belirgin biçimde üstü.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def _p(*parts: Any) -> None:
    """Tamponsuz yazdırır."""
    print(" ".join(str(part) for part in parts), flush=True)


#: Görev → (istem, doğru kod, çeldiriciler)
#:
#: Tüm kod parçaları **gerçek bpy API'sidir**; yanlış olanlar başka bir işi
#: yapar (sözdizimsel olarak geçerli). Böylece test yalnızca "kod gibi
#: görünüyor mu" değil, "doğru işi mi yapıyor" bilgisini ölçer.
GOREVLER: List[Dict[str, Any]] = [
    {
        "ad": "Sahneyi temizle",
        "istem": "Blender'da sahnedeki tüm nesneleri silmek için hangi kod kullanılır?\nKod:",
        "dogru": "bpy.ops.object.select_all(action='SELECT')\nbpy.ops.object.delete(use_global=False)",
        "celdirenler": [
            "bpy.ops.mesh.primitive_cube_add(size=2)",
            "bpy.ops.object.transform_apply(location=True, rotation=True)",
            "bpy.ops.export_scene.gltf(filepath='model.glb')",
        ],
    },
    {
        "ad": "Küp ekle",
        "istem": "Blender'da sahneye küp eklemek için hangi kod kullanılır?\nKod:",
        "dogru": "bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))",
        "celdirenler": [
            "bpy.ops.object.select_all(action='SELECT')",
            "bpy.ops.render.render(write_still=True)",
            "bpy.ops.object.light_add(type='SUN')",
        ],
    },
    {
        "ad": "Malzeme ata",
        "istem": "Blender'da bir nesneye kırmızı malzeme atamak için hangi kod?\nKod:",
        "dogru": ("malzeme = bpy.data.materials.new('Kirmizi')\n"
                  "nesne.data.materials.append(malzeme)"),
        "celdirenler": [
            "nesne.location = (1, 2, 3)",
            "bpy.context.scene.camera = bpy.context.active_object",
            "bpy.ops.object.delete(use_global=False)",
        ],
    },
    {
        "ad": "Kamera ekle",
        "istem": "Blender'da kamerayı sahneye eklemek ve etkin kamera yapmak için?\nKod:",
        "dogru": ("bpy.ops.object.camera_add(location=(7, -7, 5))\n"
                  "bpy.context.scene.camera = bpy.context.active_object"),
        "celdirenler": [
            "bpy.ops.mesh.primitive_uv_sphere_add(radius=1)",
            "sahne.render.engine = 'CYCLES'",
            "bpy.ops.object.select_all(action='SELECT')",
        ],
    },
    {
        "ad": "Render al",
        "istem": "Blender'da başsız (headless) render almak için hangi kod?\nKod:",
        "dogru": ("sahne.render.filepath = 'cikti.png'\n"
                  "bpy.ops.render.render(write_still=True)"),
        "celdirenler": [
            "bpy.ops.mesh.primitive_cube_add(size=1)",
            "bpy.ops.object.light_add(type='POINT')",
            "bpy.ops.object.select_all(action='DESELECT')",
        ],
    },
    {
        "ad": "GLB dışa aktar",
        "istem": "Blender'da sahneyi glTF (.glb) olarak dışa aktarmak için?\nKod:",
        "dogru": ("bpy.ops.export_scene.gltf(filepath='model.glb', "
                  "export_format='GLB')"),
        "celdirenler": [
            "bpy.ops.object.delete(use_global=False)",
            "bpy.ops.mesh.primitive_cylinder_add(vertices=32)",
            "bpy.ops.render.render(write_still=True)",
        ],
    },
    {
        "ad": "Işık ekle",
        "istem": "Blender'da sahneye güneş ışığı eklemek için hangi kod?\nKod:",
        "dogru": "bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))",
        "celdirenler": [
            "bpy.ops.object.camera_add(location=(7, -7, 5))",
            "bpy.ops.mesh.primitive_plane_add(size=20)",
            "bpy.ops.object.transform_apply(scale=True)",
        ],
    },
    {
        "ad": "Malzeme rengi",
        "istem": "Blender'da malzemenin temel rengini ayarlamak için hangi kod?\nKod:",
        "dogru": ("bsdf = malzeme.node_tree.nodes['Principled BSDF']\n"
                  "bsdf.inputs['Base Color'].default_value = (0.9, 0.15, 0.1, 1.0)"),
        "celdirenler": [
            "nesne.scale = (2, 1, 1)",
            "bpy.ops.object.select_all(action='SELECT')",
            "sahne.render.resolution_x = 640",
        ],
    },
    {
        "ad": "Küre ekle",
        "istem": "Blender'da yarıçapı 1 olan küre eklemek için hangi kod?\nKod:",
        "dogru": "bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=32)",
        "celdirenler": [
            "bpy.ops.mesh.primitive_cube_add(size=1)",
            "bpy.ops.object.camera_add()",
            "bpy.ops.export_scene.gltf(filepath='a.glb')",
        ],
    },
    {
        "ad": "Dönüşüm uygula",
        "istem": "Blender'da ölçek ve dönüşümü nesneye kalıcı uygulamak için?\nKod:",
        "dogru": "bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)",
        "celdirenler": [
            "bpy.ops.mesh.primitive_cone_add(vertices=32)",
            "bpy.ops.object.delete(use_global=False)",
            "bpy.context.scene.camera = bpy.context.active_object",
        ],
    },
]


def karistir(adaylar: List[str], tohum: int) -> List[str]:
    """
    Adayları **belirleyici** biçimde karıştırır.

    Neden? Doğru cevap hep aynı sırada olursa model sırayı öğrenip şans
    eseri yüksek skor alabilir. Sabit tohum, testin tekrarlanabilir olmasını
    sağlar.
    """
    import random

    kopya = list(adaylar)
    random.Random(tohum).shuffle(kopya)
    return kopya


def main() -> int:
    """Ölçümü çalıştırır."""
    from neurocore import NeuroCoreEngine

    _p("=" * 70)
    _p("  BLENDER KOD SECIMI OLCUMU (yerel model, kendi olasiliklari)")
    _p("=" * 70)

    engine = NeuroCoreEngine()
    engine.initialize()
    model = engine.model
    _p(f"  egitim adimi : {model.growth.total_training_steps}")
    _p(f"  sozluk       : {model.tokenizer.vocab_size}")
    _p("")

    dogru = 0
    toplam = 0
    detay: List[Dict[str, Any]] = []
    basladi = time.perf_counter()

    for index, gorev in enumerate(GOREVLER):
        adaylar = karistir([gorev["dogru"]] + list(gorev["celdirenler"]), index)
        skorlar: Dict[str, float] = {}
        for aday in adaylar:
            skor = model.sequence_logprob(gorev["istem"] + "\n" + aday,
                                          prompt=gorev["istem"] + "\n")
            skorlar[aday] = skor if skor is not None else float("-inf")
        secim = max(skorlar, key=skorlar.get)
        ok = secim == gorev["dogru"]
        dogru += ok
        toplam += 1
        detay.append({
            "ad": gorev["ad"], "dogru_mu": bool(ok),
            "secilen": secim[:60], "beklenen": gorev["dogru"][:60],
        })
        im = "OK " if ok else "   "
        _p(f"  {im} {gorev['ad']:16} | secilen: {secim.splitlines()[0][:52]}")

    sure = time.perf_counter() - basladi
    oran = dogru / max(1, toplam)
    _p("")
    _p("=" * 70)
    _p(f"  SONUC: {dogru}/{toplam} = %{round(oran * 100)}")
    _p(f"  Rastgele taban: %25  (4 aday)")
    _p(f"  Kazanc: {round(oran / 0.25, 2)}x rastgeleye gore")
    _p(f"  Sure: {sure:.0f}s")
    _p("=" * 70)

    rapor = {
        "dogru": dogru, "toplam": toplam, "oran": round(oran, 3),
        "rastgele_taban": 0.25, "kazanc": round(oran / 0.25, 2),
        "egitim_adimi": model.growth.total_training_steps,
        "detay": detay,
    }
    Path("_blender_kod_raporu.json").write_text(
        json.dumps(rapor, ensure_ascii=False, indent=1), encoding="utf-8")
    engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
