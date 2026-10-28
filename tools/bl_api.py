import warnings, logging, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
def p(*a): print(" ".join(str(x) for x in a), flush=True)
from neurocore.multimodal import BlenderController
import inspect
p("=== BlenderController API ===")
for name in dir(BlenderController):
    if name.startswith("_") or not callable(getattr(BlenderController, name, None)):
        continue
    try:
        sig = inspect.signature(getattr(BlenderController, name))
        p(f"  {name}{sig}")
    except Exception:
        pass
