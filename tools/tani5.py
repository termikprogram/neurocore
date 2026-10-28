import inspect, warnings, logging, sys
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path.cwd()))
from neurocore.neural.trainer import Trainer
src = inspect.getsource(Trainer.train)
lines = src.splitlines()
print("=== 'pause' ve 'kaynak_beklemesi' blogu ===")
for n, line in enumerate(lines, 1):
    if "kaynak_beklemesi" in line or "pause" in line:
        for k in range(max(0, n-6), min(n+14, len(lines))):
            print(f"  {k+1:4}: {lines[k].rstrip()[:110]}")
        print("  ---")
