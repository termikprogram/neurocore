import warnings, logging, sys, inspect
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path.cwd()))
from neurocore.agent.blender import BlenderController
print("=== BlenderController (kullanilabilir metotlar) ===")
for name in sorted(dir(BlenderController)):
    if name.startswith("_"): continue
    obj = getattr(BlenderController, name, None)
    if not callable(obj): continue
    try:
        print(f"  {name}{inspect.signature(obj)}")
    except Exception:
        print(f"  {name}(...)")
