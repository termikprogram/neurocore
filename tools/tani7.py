import inspect, warnings, logging, sys
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path.cwd()))
from neurocore.neural.trainer import Trainer
src = inspect.getsource(Trainer)
i = src.find("class _ResourceGuard")
if i < 0:
    i = src.find("_resource_state")
    print("guard sinifi bulunamadi; _resource_state:")
print(src[i:i+2400] if i >= 0 else "yok")
