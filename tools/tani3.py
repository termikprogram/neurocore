import warnings, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
def p(*a): print(" ".join(str(x) for x in a), flush=True)
from neurocore import NeuroCoreEngine
e = NeuroCoreEngine(); e.initialize()
k = e.knowledge
p("=== KnowledgeLayer ALANLARI ===")
for n in ("vault", "index", "graph", "vault_stats"):
    obj = getattr(k, n, None)
    p(f"  {n:12}: {'VAR (' + type(obj).__name__ + ')' if obj is not None else 'yok'}")
p()
p("=== vault.iter_notes var mi? ===")
v = getattr(k, "vault", None)
p(f"  vault turu: {type(v).__name__ if v else 'YOK'}")
if v is not None:
    p(f"  iter_notes: {hasattr(v, 'iter_notes')}")
    p(f"  notes     : {hasattr(v, 'notes')}")
    p(f"  count     : {hasattr(v, 'count')}")
    try:
        notes = list(v.iter_notes(limit=5))
        p(f"  ilk 5 not : {len(notes)}")
        for note in notes[:3]:
            p(f"    - {str(note.title)[:62]} ({note.word_count} kelime)")
    except Exception as exc:
        p(f"  iter_notes HATA: {type(exc).__name__}: {exc}")
p()
p("=== DOGRU CAGRi: build_dataset_from_vault(vault) ===")
try:
    ds = e.trainer.build_dataset_from_vault(v, e.model.tokenizer)
    p(f"  ornek: {len(ds)} | token: {ds.stats().get('tokens')}")
except Exception as exc:
    p(f"  HATA: {type(exc).__name__}: {exc}")
e.shutdown()
