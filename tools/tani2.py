import warnings, logging, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
def p(*a): print(" ".join(str(x) for x in a), flush=True)
from neurocore.knowledge import KnowledgeLayer
k = KnowledgeLayer(); k.initialize()
st = k.get_stats()
p("=== KASA ISTATISTIK ===")
p(f"  {json.dumps(st, ensure_ascii=False)[:400]}")
p()
rows = k.search("müfredat", limit=3)
p(f"=== 'müfredat' aramasi: {len(rows or [])} sonuc ===")
for r in (rows or [])[:3]:
    if isinstance(r, dict):
        p(f"  baslik: {str(r.get('title'))[:60]} | kind: {r.get('kind')} | kategori: {r.get('category')}")
    else:
        p(f"  {type(r).__name__}: {str(getattr(r,'title','?'))[:60]}")
p()
p("=== ILK 5 NOT (kasa geneli) ===")
try:
    recent = k.recent(limit=5)
    for n in (recent or []):
        t = n.get("title") if isinstance(n, dict) else getattr(n, "title", "?")
        kd = n.get("kind") if isinstance(n, dict) else getattr(n, "kind", "?")
        p(f"  [{str(kd):10}] {str(t)[:62]}")
except Exception as exc:
    p(f"  recent HATA: {exc}")
p()
p("=== build_dataset_from_vault NE ISTIYOR? ===")
import inspect
from neurocore.neural.trainer import Trainer
src = inspect.getsource(Trainer.build_dataset_from_vault)
for n, line in enumerate(src.splitlines(), 1):
    s = line.strip()
    if any(x in s for x in ("kind", "list_notes", "notes", "filter", "if ", "skip", "exclude", "max_notes", "for ")):
        p(f"  {n:4}: {s[:104]}")
k.shutdown()
