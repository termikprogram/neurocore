"""VRAM butcesi: hangi model boyutu egitilebilir?"""
import warnings, logging, sys, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
def p(*a): print(" ".join(str(x) for x in a), flush=True)

def vram():
    out = subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.free","--format=csv,noheader"],
                         capture_output=True, text=True, timeout=15).stdout.strip()
    return out

p("=" * 70)
p("  VRAM BUTCESI ANALIZI")
p("=" * 70)
p(f"  baslangic: {vram()}")

import torch
if not torch.cuda.is_available():
    p("  CUDA yok — olcum yapilamaz"); raise SystemExit(1)
free, total = torch.cuda.mem_get_info()
p(f"  torch: {free/1024**2:.0f} MB bos / {total/1024**2:.0f} MB toplam")

from neurocore.neural.transformer import TransformerConfig, TransformerBackbone
from neurocore.neural.trainer import Trainer
from neurocore.core.config import SystemConfig

p("")
p("=== MEVCUT MODEL (208M) ===")
cfg = SystemConfig()
t = Trainer()
t.initialize(cfg)
from neurocore import NeuroCoreEngine
e = NeuroCoreEngine(); e.initialize()
m = e.model
mem = m.memory_estimate()
p(f"  parametre     : {m.parameter_count():,}")
p(f"  cikarim VRAM  : {mem.get('inference_mb', 0):.0f} MB")
p(f"  egitim (adam) : {mem.get('training_adam_mb', 0):.0f} MB")
free, total = torch.cuda.mem_get_info()
p(f"  su anki kullanim: {(total-free)/1024**2:.0f} MB")
p("")
p("=== ARTMIS BOYUTLARIN TAHMINI ===")
p(f"  {'boyut':>22} | {'parametre':>14} | {'egitim RAM/VRAM':>16}")
temel = m.parameter_count()
# Kabaca: parametre artisi VRAM'i lineer artirir (adam: 4x param + aktivasyon)
for ad, kat in (("mevcut (8 katman, 4 uzman)", 1.0),
                ("+4 katman (12 katman)", 1.5),
                ("+8 katman (16 katman)", 2.0),
                ("12 uzman (mevcut katman)", 2.0),
                ("16 katman + 12 uzman", 3.5)):
    param = temel * kat
    egitim = param * 4 / 1024**2 * 2.2   # adam: 4 bayt x param x (grad+2 moment) ~ 2.2x
    p(f"  {ad:>22} | {param:>14,} | {egitim:>13.0f} MB")
p("")
p("  NOT: 1080 Ti'da AMP (fp16) acik; aktivasyon bellegi ayri hesaplanir.")
e.shutdown()
