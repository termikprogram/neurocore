import warnings, logging, sys, json
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path.cwd()))
out = []
def p(*a): out.append(" ".join(str(x) for x in a))
from neurocore.neural.model import NeuroCoreModel
import inspect
p("=== sequence_score var mi? ===")
for name in ("sequence_score", "score_sequence", "perplexity", "log_likelihood"):
    p(f"  {name}: {hasattr(NeuroCoreModel, name)}")
p()
p("=== olcum betiginde kullanilan yol ===")
p("  (yoksa sezgisel anahtar kelime yedegi kullanildi -> %83 YANILTICI olabilir)")
p()
p("=== embed_text var mi? ===")
p(f"  embed_text: {hasattr(NeuroCoreModel, 'embed_text')}")
if hasattr(NeuroCoreModel, "embed_text"):
    p(f"  imza: {inspect.signature(NeuroCoreModel.embed_text)}")
p()
p("=== ANLAM TESTI: gomme benzerligi (kelime torbasi degil) ===")
from neurocore import NeuroCoreEngine
e = NeuroCoreEngine(); e.initialize()
m = e.model
cumleler = {
    "binary": "ikili sayı sistemi bit bayt",
    "blender": "blender bpy mesh malzeme render",
    "os": "işletim sistemi kernel süreç bellek",
    "libraries": "kütüphane numpy torch asyncio",
}
vecs = {}
for k, s in cumleler.items():
    try:
        v = m.embed_text(s)
        vecs[k] = v
        p(f"  {k:10}: gomme uretildi ({len(v) if v else 0} boyut)")
    except Exception as exc:
        p(f"  {k:10}: HATA {type(exc).__name__}: {str(exc)[:70]}")
if len(vecs) >= 2:
    import math
    def cos(a, b):
        d = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
        return d / (na*nb) if na and nb else 0.0
    p()
    p("  Benzerlik matrisi (yuksek = ayni alan):")
    ks = list(vecs)
    p("            " + "".join(f"{k:>10}" for k in ks))
    for a in ks:
        satir = "".join(f"{cos(vecs[a], vecs[b]):>10.3f}" for b in ks)
        p(f"    {a:8}{satir}")
e.shutdown()
open("_anlam.txt","w",encoding="utf-8").write("\n".join(out))
