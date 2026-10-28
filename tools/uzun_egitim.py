"""Uzun egitim: genisletilmis mufredat + tokenizer yenileme + cok adim."""
import warnings, logging, json, time, sys
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
from pathlib import Path

out = []
def p(*a):
    line = " ".join(str(x) for x in a)
    out.append(line)
    print(line, flush=True)

sys.path.insert(0, str(Path.cwd()))
from neurocore.neural.pipeline import CurriculumTrainer

p("=" * 64)
p("  GENISLETILMIS MUFREDATLA EGITIM (20 kalip)")
p("=" * 64)

t = CurriculumTrainer(patterns=20, vocab_size=8000)
t.initialize(workspace=str(Path.cwd()), data_path=str(Path.cwd() / ".neurocore"))

# 1) Hazirlik: tokenizer + gomme
p("")
p("--- 1) HAZIRLIK (tokenizer + gomme boyutlandirma) ---")
t0 = time.perf_counter()
r = t.prepare()
p(f"  sure: {time.perf_counter()-t0:.1f} sn")
for k in ("lessons","chars","vocab_before","vocab_after","rows_added",
          "tokens_after","chars_per_token_before","chars_per_token_after","gain"):
    p(f"  {k:26} {r.get(k)}")
p("")

# 2) Uzun egitim: 400 adim, tek tur
p("--- 2) EGITIM (400 adim) ---")
t0 = time.perf_counter()
ozet = t.run_rounds(rounds=1, steps_per_round=400, max_minutes=60.0,
                    skip_prepare=True, fresh_engine_per_round=False)
p(f"  sure: {time.perf_counter()-t0:.1f} sn")
for k in ("ok","rounds_ok","steps_total","loss_first","loss_last","model_steps","saved","vault_notes"):
    p(f"  {k:16} {ozet.get(k)}")
pr = ozet.get("progress") or {}
if pr.get("error"): p(f"  HATA: {pr['error'][:140]}")
p("")

# 3) Durum
g = json.loads(Path("models/neurocore/growth.json").read_text(encoding="utf-8"))
p(f"--- 3) MODEL: {g.get('total_training_steps')} / 800 adim ---")
p("=" * 64)
