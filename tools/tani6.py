import inspect, warnings, logging, sys, json
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path.cwd()))
from neurocore.neural.trainer import Trainer
src = inspect.getsource(Trainer.plan_training)
lines = src.splitlines()
print("=== plan_training: bellek tahmini blogu (satir 80-105) ===")
for n, line in enumerate(lines, 1):
    if 78 <= n <= 108:
        print(f"  {n:4}: {line.rstrip()[:112]}")
