"""Gercek olcum: modelin KENDI olasiliklariyla katman tahmini (cloze)."""
import warnings, logging, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
def p(*a): print(" ".join(str(x) for x in a), flush=True)

from neurocore import NeuroCoreEngine
from neurocore.neural.curriculum import build_curriculum, LAYER_LABELS

e = NeuroCoreEngine(); e.initialize()
m = e.model
p("=" * 70)
p("  GERCEK OLCUM — modelin KENDI olasiliklari (kelime torbasi DEGIL)")
p("=" * 70)
p(f"  egitim adimi: {m.growth.total_training_steps} | sozluk: {m.tokenizer.vocab_size}")
p("")

# ONCE dogrulama: sequence_logprob dogru calisiyor mu?
test_istem = "Blender ile 3B Modelleme"
iyi = test_istem + " — bpy ile malzeme ve render"
kotu = test_istem + " — zzz qqq xxx"
s_iyi = m.sequence_logprob(iyi, prompt=test_istem)
s_kotu = m.sequence_logprob(kotu, prompt=test_istem)
p("=== YONTEM DOGRULAMASI ===")
p(f"  anlamli devam skoru : {s_iyi}")
p(f"  sacma devam skoru   : {s_kotu}")
p(f"  dogru siralama      : {s_iyi is not None and s_kotu is not None and s_iyi > s_kotu}")
p("")

# CLOZE: ders govdesinden katman tahmini (6 aday)
katmanlar = ["binary", "hardware", "os", "languages", "libraries", "blender"]
dersler = build_curriculum()
dogru = 0; toplam = 0; detay = []
t0 = time.perf_counter()
for ders in dersler[:12]:
    istem = f"Konu: {ders.body[:110]}\nBu dersin alani:"
    skorlar = {}
    for k in katmanlar:
        s = m.sequence_logprob(istem + " " + k, prompt=istem)
        skorlar[k] = s if s is not None else float("-inf")
    secim = max(skorlar, key=skorlar.get)
    ok = secim == ders.layer
    dogru += ok; toplam += 1
    detay.append({"beklenen": ders.layer, "secim": secim, "dogru": bool(ok),
                  "baslik": ders.title[:44]})
    im = "OK " if ok else "   "
    p(f"  {im} beklenen={ders.layer:10} secim={secim:10} {ders.title[:42]}")
p("")
p(f"=== SONUC: {dogru}/{toplam} (%{round(dogru/max(1,toplam)*100)}) | "
  f"rastgele taban %{round(100/6)} | {time.perf_counter()-t0:.0f}s ===")
json.dump({"dogru": dogru, "toplam": toplam, "detay": detay},
          open("_cloze.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
e.shutdown()
