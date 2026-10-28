import warnings, logging, json, time, sys
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
from pathlib import Path
# Tamponsuz cikti
import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.stdout.reconfigure(line_buffering=True)
def p(*a): print(" ".join(str(x) for x in a), flush=True)

from neurocore import NeuroCoreEngine
from neurocore.neural.trainer import Trainer

p("=== MOTOR YUKLENIYOR ===")
t0 = time.perf_counter()
e = NeuroCoreEngine(); e.initialize()
p(f"  {time.perf_counter()-t0:.1f} sn | model steps: {e.model.growth.total_training_steps}")

p("")
p("=== KASA VERI KUMESI (kac ornek?) ===")
t0 = time.perf_counter()
try:
    ds = e.trainer.build_dataset_from_vault(e.knowledge, e.model.tokenizer)
    p(f"  {time.perf_counter()-t0:.1f} sn | ornek: {len(ds)} | token: {ds.stats().get('tokens')}")
    p(f"  stats: {json.dumps(ds.stats(), ensure_ascii=False)[:200]}")
except Exception as exc:
    p(f"  HATA: {type(exc).__name__}: {exc}")

p("")
p("=== DENEYIM VERI KUMESI ===")
t0 = time.perf_counter()
try:
    items = e.evolution.curriculum.build(e.evolution.experiences, max_items=300)
    pairs = e.evolution.curriculum.to_training_pairs(items)
    ds2 = e.trainer.build_dataset_from_experiences(pairs, e.model.tokenizer)
    p(f"  {time.perf_counter()-t0:.1f} sn | ornek: {len(ds2)}")
except Exception as exc:
    p(f"  HATA: {type(exc).__name__}: {exc}")

p("")
p("=== EGITIM PLANI (400 adim istegi) ===")
try:
    plan = e.trainer.plan_training(estimated_samples=300, model=e.model, steps=400, rationale="test")
    p(f"  adim: {plan.steps} | batch: {plan.batch_size} | seq: {plan.seq_len}")
    p(f"  lr: {plan.learning_rate} | accum: {plan.grad_accumulation}")
except Exception as exc:
    p(f"  HATA: {type(exc).__name__}: {exc}")
e.shutdown()
