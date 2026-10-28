"""Son adimlar: 800 esigini as (taze surec, temiz VRAM)."""
import warnings, logging, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
def p(*a): print(time.strftime("%H:%M:%S"), "|", " ".join(str(x) for x in a), flush=True)
from neurocore import NeuroCoreEngine

p("MOTOR YUKLENIYOR (temiz VRAM)...")
e = NeuroCoreEngine(); e.initialize()
p(f"  model adimi: {e.model.growth.total_training_steps}")

ds = e.trainer.build_dataset_from_vault(e.knowledge.vault, e.model.tokenizer)
p(f"VERI: {len(ds)} ornek")

# 40 adim: 800 esigini guvenle asar
for tur in range(1, 5):
    onceki = e.model.growth.total_training_steps
    if onceki >= 820:
        p(f"  yeterli ({onceki}) — duruluyor")
        break
    p(f"TUR {tur}: egitim (25 adim)...")
    t0 = time.perf_counter()
    res = e.train(steps=25, from_vault=True, from_experiences=True,
                  max_minutes=12.0, user_initiated=True)
    sonraki = e.model.growth.total_training_steps
    p(f"  {res.get('steps_completed')}/{res.get('steps_requested')} adim | "
      f"kayip {res.get('initial_loss'):.4f} -> {res.get('final_loss'):.4f} | "
      f"{time.perf_counter()-t0:.0f}s | neden={res.get('stop_reason')}")
    p(f"  MODEL: {sonraki} / 800")
    e.model.save()

son = e.model.growth.total_training_steps
p("")
p(f"=== SONUC: {son} adim | uretim esigi (800): {'ASILDI' if son >= 800 else 'henuz' } ===")
p(f"  can_generate: {e.model.can_generate}")
e.shutdown()
