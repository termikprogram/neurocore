import inspect, warnings, logging, sys
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path.cwd()))
from neurocore.neural.trainer import Trainer
src = inspect.getsource(Trainer.train)
lines = src.splitlines()
print("=== EGITIM DONGUSU (unscale / step / accum) ===")
for n, line in enumerate(lines, 1):
    if any(k in line for k in ("unscale_", "scaler", "optimizer.step", "zero_grad",
                               "grad_accumulation", "loss.backward", "reduce")):
        print(f"  {n:4}: {line.rstrip()[:112]}")
