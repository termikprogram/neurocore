import inspect, warnings, logging, sys, json
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path.cwd()))
from neurocore.neural.trainer import Trainer
print("=== _resource_state() ===")
print(inspect.getsource(Trainer._resource_state))
