"""Blender gorevi: 3 nesneli kompozisyon + malzeme + render + GLB."""
import warnings, logging, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
def p(*a): print(" ".join(str(x) for x in a), flush=True)

from neurocore.agent.blender import BlenderController

ws = Path("BlenderTest").resolve()
ws.mkdir(exist_ok=True)
b = BlenderController(); b.initialize(workspace=str(ws), shell=None)
p(f"Blender: {b.version()} | exe: {Path(b.executable()).name}")

KOD = '''
import bpy, math
from pathlib import Path
OUT = Path(OUT_STR)

# Sahneyi temizle
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

def malzeme(ad, renk, puruz=0.4):
    m = bpy.data.materials.new(ad)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (renk[0], renk[1], renk[2], 1.0)
    bsdf.inputs["Roughness"].default_value = puruz
    return m

# Zemin
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
zemin = bpy.context.active_object
zemin.name = "Zemin"
zemin.data.materials.append(malzeme("ZeminMal", (0.18, 0.18, 0.22), 0.85))

# Uc nesne
bpy.ops.mesh.primitive_cube_add(size=1.6, location=(-3, 0, 0.8))
kup = bpy.context.active_object
kup.name = "Kup"
kup.data.materials.append(malzeme("KirmiziMal", (0.85, 0.18, 0.14)))

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.9, segments=32, location=(0, 0, 0.9))
kure = bpy.context.active_object
kure.name = "Kure"
kure.data.materials.append(malzeme("MaviMal", (0.16, 0.5, 0.9)))

bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.7, depth=1.8, location=(3, 0, 0.9))
silindir = bpy.context.active_object
silindir.name = "Silindir"
silindir.data.materials.append(malzeme("YesilMal", (0.22, 0.7, 0.3)))

# Isik
bpy.ops.object.light_add(type="SUN", location=(6, -6, 10))
bpy.context.active_object.data.energy = 4.0

# Kamera
bpy.ops.object.camera_add(location=(9, -9, 6),
                          rotation=(math.radians(64), 0, math.radians(45)))
bpy.context.scene.camera = bpy.context.active_object

# Render ayarlari + render
sahne = bpy.context.scene
sahne.render.engine = "CYCLES"
sahne.cycles.device = "CPU"
sahne.cycles.samples = 24
sahne.render.resolution_x = 480
sahne.render.resolution_y = 320
sahne.render.filepath = str(OUT / "kompozisyon.png")
sahne.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)

# GLB disa aktar
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=str(OUT / "kompozisyon.glb"),
                          export_format="GLB", use_selection=True)

print("NC_NESNE_SAYISI", len([o for o in bpy.data.objects if o.type == "MESH"]))
print("NC_MALZEME_SAYISI", len(bpy.data.materials))
print("NC_TAMAM")
'''.replace("OUT_STR", repr(str(ws)))

p("")
p("=== BLENDER BETIGI CALISTIRILIYOR ===")
t0 = time.perf_counter()
r = b.run_script(KOD, timeout=600.0)
sure = time.perf_counter() - t0
p(f"  ok={r.get('ok')} | sure={sure:.1f}s | cikis kodu={r.get('returncode')}")
cikti = str(r.get("stdout") or "")
for satir in cikti.splitlines():
    if "NC_" in satir: p(f"  {satir.strip()[:90]}")
if r.get("errors"): p(f"  HATALAR: {str(r['errors'])[:300]}")
p("")
p("=== URETILEN DOSYALAR ===")
for f in sorted(ws.iterdir()):
    p(f"  {f.name:26} {f.stat().st_size:>12,} bayt")
p("")
p("=== NESNE DOGRULAMA (Blender'dan oku) ===")
obj = b.list_objects()
p(f"  nesneler: {json.dumps(obj.get('objects', obj), ensure_ascii=False)[:300]}")
b.shutdown()
