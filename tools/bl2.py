import warnings, logging, sys, inspect
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path.cwd()))
import neurocore.multimodal as mm
print("=== neurocore.multimodal icerigi ===")
print(" ", [n for n in dir(mm) if not n.startswith("_")][:30])
print()
import pkgutil
print("=== blender modulu var mi? ===")
for m in pkgutil.iter_modules(mm.__path__):
    print("  ", m.name)
