import inspect, warnings, logging
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import sys
sys.path.insert(0, str(Path.cwd()))
from neurocore.neural.trainer import Trainer
src = inspect.getsource(Trainer.train)
print("=== Trainer.train: durdurma kosullari ===")
for n, line in enumerate(src.splitlines(), 1):
    s = line.strip()
    if any(k in s for k in ("stop_reason", "break", "kaynak", "stall", "wait", "if ", "max_minutes",
                            "steps_completed", "target_loss", "should_stall", "elapsed")):
        print(f"  {n:4}: {s[:112]}")
