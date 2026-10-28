"""Araç uctan uca testi: LocalRankController gercekten kod seciyor mu?"""
import warnings, logging, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
def p(*a): print(" ".join(str(x) for x in a), flush=True)

from neurocore.agent.local_rank import LocalRankController
from neurocore import NeuroCoreEngine

p("=" * 70)
p("  ARAC TESTI: local.rank_candidates")
p("=" * 70)
eng = NeuroCoreEngine(); eng.initialize()
c = LocalRankController(); c.initialize(engine=eng)
p(f"  durum: {json.dumps(c.get_stats(), ensure_ascii=False)[:220]}")
p("")
p("=== ARAC TANIMLARI ===")
for td in c.tools():
    p(f"  {td.name:26} | guvenlik: {td.safety.value} | parametre: {len(td.parameters)}")
p("")

arac = {td.name: td for td in c.tools()}
p("=== GERCEK CAGRILAR ===")
testler = [
    ("Blender'da sahnedeki tüm nesneleri silmek için hangi kod?\nKod:",
     ["bpy.ops.object.select_all(action='SELECT')\nbpy.ops.object.delete(use_global=False)",
      "bpy.ops.mesh.primitive_cube_add(size=2)",
      "bpy.ops.export_scene.gltf(filepath='m.glb')"]),
    ("Blender'da kırmızı malzeme atamak için hangi kod?\nKod:",
     ["nesne.location = (1, 2, 3)",
      "malzeme = bpy.data.materials.new('Kirmizi')\nnesne.data.materials.append(malzeme)",
      "bpy.context.scene.camera = bpy.context.active_object"]),
    ("Python'da iki sayıyı toplayan fonksiyon?\nKod:",
     ["def topla(a, b):\n    return a + b",
      "import os\nos.remove('x')"]),
]
for istem, adaylar in testler:
    p(f"ISTEM: {istem.splitlines()[0][:60]}")
    r = arac["local.rank_candidates"].handler(prompt=istem, candidates=adaylar)
    if r.get("ok"):
        p(f"  en iyi #{r['best_index']} | guven: {r['confidence']} | fark: {r['margin']}")
        p(f"  secilen: {r['best_text'].splitlines()[0][:60]}")
    else:
        p(f"  HATA: {r.get('error')}")
    p("")
p(f"  sayaclar: {json.dumps(c.get_stats(), ensure_ascii=False)[:240]}")
c.shutdown(); eng.shutdown()
