import warnings, logging, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
def p(*a): print(" ".join(str(x) for x in a), flush=True)
from neurocore import NeuroCoreEngine

p("=" * 66)
p("  URETIM ESIGI VE KALITE OLCUMU")
p("=" * 66)
t0 = time.perf_counter()
e = NeuroCoreEngine(); e.initialize()
p(f"  motor: {time.perf_counter()-t0:.1f}s")
m = e.model
steps = m.growth.total_training_steps
p(f"  egitim adimi : {steps} / {m.MIN_GENERATION_STEPS}")
p(f"  can_generate : {m.can_generate}")
p(f"  tokenizer    : {m.tokenizer.vocab_size} sozluk")
p(f"  embedding    : {tuple(m.backbone.token_embedding.weight.shape)}")
p(f"  parametre    : {m.parameter_count():,}")
p(f"  olgunluk     : {json.dumps(m.generation_readiness(), ensure_ascii=False)[:160]}")
p("")
p("=== URETIM TESTI ===")
sorular = [
    "Blender ile 3B Modelleme — Malzeme oluşturma ve renklendirme nedir?",
    "İkili Sayı Sistemi — İkili (binary) sayı sistemi nedir?",
    "Blender ile 3B Modelleme — Sahneyi temizlemek ve nesne eklemek — örnek göster",
]
for soru in sorular:
    p("")
    p(f"  SORU: {soru[:70]}")
    t0 = time.perf_counter()
    try:
        uretim = m.generate(soru, max_new_tokens=90, temperature=0.7)
        metin = uretim.get("text", "") if isinstance(uretim, dict) else str(uretim)
        p(f"  SURE: {time.perf_counter()-t0:.1f}s")
        p(f"  CIKTI: {metin[:420]}")
    except Exception as exc:
        p(f"  HATA: {type(exc).__name__}: {exc}")
e.shutdown()
