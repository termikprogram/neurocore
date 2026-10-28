"""Uzun egitim: 873 ornekli kasa veri kumesi ile 400 adim."""
import warnings, logging, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
def p(*a): print(time.strftime("%H:%M:%S"), "|", " ".join(str(x) for x in a), flush=True)

from neurocore import NeuroCoreEngine

p("MOTOR YUKLENIYOR...")
t0 = time.perf_counter()
e = NeuroCoreEngine(); e.initialize()
p(f"hazir ({time.perf_counter()-t0:.1f}s) | model adimi: {e.model.growth.total_training_steps}")

p("KASA VERI KUMESI...")
t0 = time.perf_counter()
ds = e.trainer.build_dataset_from_vault(e.knowledge.vault, e.model.tokenizer)
p(f"  {len(ds)} ornek, {ds.stats().get('tokens')} token ({time.perf_counter()-t0:.1f}s)")

p("EGITIM BASLIYOR (400 adim)...")
t0 = time.perf_counter()
res = e.train(steps=400, from_vault=True, from_experiences=True,
              max_minutes=45.0, user_initiated=True)
p(f"  sure: {time.perf_counter()-t0:.1f}s")
p(f"  ok={res.get('ok')} adim={res.get('steps_completed')}/{res.get('steps_requested')}")
p(f"  kayip: {res.get('initial_loss')} -> {res.get('final_loss')}")
p(f"  token: {res.get('tokens_seen')} | neden: {res.get('stop_reason')}")
if res.get("error"): p(f"  HATA: {str(res['error'])[:150]}")

p("MODEL KAYDEDILIYOR...")
info = e.model.save()
p(f"  {json.dumps(info, ensure_ascii=False)[:150]}")
p(f"MODEL: {e.model.growth.total_training_steps} / 800 adim")
e.shutdown()
p("BITTI")
