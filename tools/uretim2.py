import warnings, logging, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
def p(*a): print(" ".join(str(x) for x in a), flush=True)
from neurocore import NeuroCoreEngine

e = NeuroCoreEngine(); e.initialize()
m = e.model
p("=" * 68)
p(f"  URETIM TESTI — model {m.growth.total_training_steps} adim, can_generate={m.can_generate}")
p("=" * 68)
sorular = [
    "Blender ile 3B Modelleme — Malzeme oluşturma ve renklendirme nedir?",
    "İkili Sayı Sistemi — İkili (binary) sayı sistemi nedir?",
    "Blender ile 3B Modelleme — Sahneyi temizlemek ve nesne eklemek — örnek göster",
    "İşletim Sistemi — Çekirdek (kernel) ve kullanıcı alanı nasıl çalışır?",
]
for soru in sorular:
    p("")
    p(f"SORU : {soru}")
    t0 = time.perf_counter()
    r = m.generate_text(soru, max_new_tokens=110, temperature=0.7)
    p(f"SURE : {time.perf_counter()-t0:.1f}s | yeni token: {r.get('new_tokens')} | durdu: {r.get('stopped_by')}")
    if r.get("quality_rejected"): p(f"  (kalite reddi: {r.get('error')})")
    metin = (r.get("text") or "").strip()
    p(f"CIKTI: {metin[:500] if metin else '(bos)'}")
e.shutdown()
