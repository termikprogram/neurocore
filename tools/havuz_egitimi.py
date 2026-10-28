"""
Aşama 4 — Havuzla eğitim + gerçek doğrulama kaybı
==================================================
⚠ Neden bu betik?
----------------
Aşama 0 ölçümü (kanıtlandı):
```
eğitim verisi   : ~600 kasa notu (~1.300 örnek)   ← core/engine.py:2166
havuz           : 61.195 kayıt / 23,2M token      ← HİÇ KULLANILMADI
eğitim kaybı    : 0,083   (ezber)
gerçek kayıp    : 6,44    (havuzda, rastgele 8,45)
doğrulama kaybı : hiç ölçülmemiş
```

Bu betik, donmuş dilimlerle eğitir:
```
.neurocore/egitim_dilimi.jsonl      55.784 kayıt   (eğitim)
.neurocore/dogrulama_kumesi.jsonl    5.411 kayıt   (⚠ eğitime GİRMEZ)
```

⚠ Doğrulama kaybı artık **her `--eval` adımda** ölçülür ve raporlanır
(`trainer.train(validation=...)` — Aşama 4'te bağlandı).

Kullanım
--------
    python tools/havuz_egitimi.py --adim 300 --eval 100
    python tools/havuz_egitimi.py --adim 300 --eval 100 --deneme   (modeli kaydetmez)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

EGITIM = Path(".neurocore/egitim_dilimi.jsonl")
DOGRULAMA = Path(".neurocore/dogrulama_kumesi.jsonl")
#: ⚠ v2: içerik tekilleştirilmiş + istem belgenin kendi parçası
EGITIM_V2 = Path(".neurocore/egitim_dilimi_v2.jsonl")
DOGRULAMA_V2 = Path(".neurocore/dogrulama_kumesi_v2.jsonl")
RAPOR = Path("_havuz_egitimi.json")

#: ⚠ Rastgele referans: ln(vocab_size)
def rastgele_kayip(vocab: int) -> float:
    """Rastgele modelin token başına kaybı."""
    import math

    return round(math.log(max(2, vocab)), 3)


def _kayitlar(yol: Path, limit: int = 0) -> List[str]:
    """JSONL'den metinleri okur."""
    metinler: List[str] = []
    if not yol.exists():
        return metinler
    with yol.open(encoding="utf-8") as f:
        for satir in f:
            if limit and len(metinler) >= limit:
                break
            satir = satir.strip()
            if not satir:
                continue
            try:
                d = json.loads(satir)
            except json.JSONDecodeError:
                continue
            m = (d.get("metin") or d.get("text") or "").strip()
            if len(m) >= 40:
                metinler.append(m)
    return metinler


def _bol(metin: str) -> Tuple[str, str]:
    """
    Metni (istem, hedef) çiftine ayırır.

    ⚠ `TrainingDataset.add()` **boş istemi reddeder** (`trainer.py:363`).
    Bu yüzden ilk satır istem, kalanı hedef olur. Kodlanan metin yine
    `prompt + "\\n" + target` = orijinal belge.
    """
    m = metin.strip()
    if "\n" in m:
        ilk, geri = m.split("\n", 1)
        if len(ilk.strip()) >= 8 and len(geri.strip()) >= 16:
            return ilk.strip()[:200], geri.strip()
    return m[:120].strip(), m[120:].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Havuzla eğitim (doğrulamalı)")
    ap.add_argument("--adim", type=int, default=300)
    ap.add_argument("--eval", type=int, default=100,
                    help="Kaç adımda bir doğrulama ölçülsün")
    ap.add_argument("--ornek", type=int, default=20000,
                    help="Eğitimde kullanılacak en fazla örnek")
    ap.add_argument("--dakika", type=float, default=45.0)
    ap.add_argument("--deneme", action="store_true",
                    help="Modeli diske kaydetme")
    ap.add_argument("--dilim", default="v2", choices=("v1", "v2"),
                    help="v2 = tekilleştirilmiş + belge içi istem (varsayılan)")
    ap.add_argument("--seq", type=int, default=256,
                    help="Dizi uzunluğu (128 → ~2× hız, daha kısa bağlam)")
    ap.add_argument("--talimat", default=".neurocore/talimat_verisi.jsonl",
                    help="Talimat (soru→cevap) verisi; boş verilirse karıştırılmaz")
    ap.add_argument("--talimat-agirlik", type=float, default=3.0,
                    help="Talimat örneklerinin ağırlığı (az sayıda ama kritik)")
    ap.add_argument("--batch", type=int, default=0,
                    help="Toplu iş boyutu (0 = plan seçer). ⚠ TDR için küçült")
    ap.add_argument("--kaydet-her", type=int, default=0,
                    help="Kaç adımda bir checkpoint (0 = plan). Sık = güvenli")
    ap.add_argument("--amp-yok", action="store_true",
                    help="AMP (fp16) kapat — Pascal'da CUDA 'unknown error' "
                         "alınırsa fp32 ile devam et")
    args = ap.parse_args()

    print("=" * 74)
    print("  AŞAMA 4 — HAVUZLA EĞİTİM + DOĞRULAMA KAYBI")
    print("=" * 74)

    egitim_yolu = EGITIM_V2 if args.dilim == "v2" else EGITIM
    dogrulama_yolu = DOGRULAMA_V2 if args.dilim == "v2" else DOGRULAMA
    if args.dilim == "v2" and not egitim_yolu.exists():
        print("  ✗ v2 dilim yok — önce tools/havuz_hazirla.py")
        return 1

    metin_egitim = _kayitlar(egitim_yolu, limit=args.ornek)
    metin_val = _kayitlar(dogrulama_yolu, limit=1200)
    print(f"\n  dilim        : {args.dilim} ({egitim_yolu.name})")
    print(f"  eğitim metni : {len(metin_egitim):,}")
    print(f"  doğrulama    : {len(metin_val):,}  (⚠ eğitime girmiyor)")
    if not metin_egitim or not metin_val:
        print("  ✗ dilim dosyaları yok — önce tools/dogrulama_kur.py")
        return 1

    from neurocore import NeuroCoreEngine
    from neurocore.neural.trainer import TrainingDataset

    t0 = time.perf_counter()
    e = NeuroCoreEngine()
    e.initialize()
    m = e.model
    print(f"\n  model        : {m.growth.total_training_steps:,} adım · "
          f"{m.growth.total_tokens_seen:,} token")
    print(f"  yükleme      : {time.perf_counter() - t0:.1f} sn")

    rast = rastgele_kayip(m.model_config.vocab_size)
    print(f"  rastgele ref : {rast}  (ln {m.model_config.vocab_size})")

    # ── Veri kümesi ────────────────────────────────────────────────────
    t1 = time.perf_counter()
    seq_len = max(64, int(args.seq))
    # ⚠ v2: istem = belgenin kendi ilk parçası (başlık DEĞİL).
    # v1'de aynı başlık yüzlerce farklı devama gidiyordu → belirsiz.
    if args.dilim == "v2":
        from tools.havuz_hazirla import istem_hedef as _bol_v2

        _bol_fn = _bol_v2
    else:
        _bol_fn = _bol

    ds = TrainingDataset(m.tokenizer, seq_len,
                         max_samples=len(metin_egitim) + 16)
    for metin in metin_egitim:
        p, tg = _bol_fn(metin)
        ds.add(p, tg, weight=1.0, tags=(), difficulty=0.5)

    # ⚠ Talimat (soru→cevap) örnekleri — az sayıda ama **yüksek ağırlık**.
    # Ölçüm: havuz yalnız devam biçiminde; serbest soru %14,7 kalıyordu.
    #
    # ⚠ ÖLÇÜM DÜZELTMESİ: talimat karışınca **havuz** doğrulaması
    # yükselebilir (dağılım kayması; ölçüldü: %15). Bu yüzden talimattan
    # **ayrı** doğrulama ayrılır ve iki metrik ayrı raporlanır.
    n_talimat = 0
    talimat_val: List[Dict[str, str]] = []
    if args.talimat:
        tp = Path(args.talimat)
        if tp.exists():
            _ham: List[Dict[str, str]] = []
            with tp.open(encoding="utf-8") as f:
                for satir in f:
                    satir = satir.strip()
                    if not satir:
                        continue
                    try:
                        d = json.loads(satir)
                    except json.JSONDecodeError:
                        continue
                    soru = (d.get("soru") or "").strip()
                    cevap = (d.get("cevap") or "").strip()
                    if soru and cevap:
                        _ham.append({"soru": soru, "cevap": cevap})
            # ⚠ Sabit tohumla %10 ayrılır (her koşuda aynı küme)
            random.Random(11).shuffle(_ham)
            _kes = max(1, len(_ham) // 10)
            talimat_val = _ham[:_kes]
            for d in _ham[_kes:]:
                ds.add(d["soru"], d["cevap"],
                       weight=float(args.talimat_agirlik),
                       tags=("talimat",), difficulty=0.4)
                n_talimat += 1
    vd = TrainingDataset(m.tokenizer, seq_len,
                         max_samples=len(metin_val) + 16)
    for metin in metin_val:
        p, tg = _bol_fn(metin)
        vd.add(p, tg, weight=1.0, tags=(), difficulty=0.5)
    #: ⚠ Talimat doğrulama kümesi (ayrı metrik)
    td = TrainingDataset(m.tokenizer, seq_len,
                         max_samples=max(16, len(talimat_val) + 16))
    for d in talimat_val:
        td.add(d["soru"], d["cevap"], weight=1.0, tags=("talimat",))
    print(f"\n  ds           : {len(ds):,} örnek · "
          f"{ds.stats().get('tokens', 0):,} token")
    if n_talimat:
        print(f"  talimat       : {n_talimat:,} çift "
              f"(ağırlık {args.talimat_agirlik}) · "
              f"doğrulama {len(td):,}")
    print(f"  val ds       : {len(vd):,} örnek · "
          f"{vd.stats().get('tokens', 0):,} token")
    print(f"  hazırlık     : {time.perf_counter() - t1:.1f} sn")

    if ds.is_empty or vd.is_empty:
        print("  ✗ veri kümesi boş")
        e.shutdown()
        return 1

    # ── Plan ───────────────────────────────────────────────────────────
    plan = e.trainer.plan_training(
        estimated_samples=len(ds), model=m, steps=args.adim,
        rationale="Aşama 4 — havuzla eğitim (doğrulamalı)",
        # ⚠ Gerçek token sayısı verilir → doygunluk oranı **bu turun
        # hacmi / bu veri kümesi** olur (ömür boyu sayaç değil).
        data_tokens=int(ds.stats().get("tokens", 0) or 0),
    )
    plan.eval_every = max(1, args.eval)
    # ⚠ Planın dizi uzunluğu **veri kümesiyle** aynı olmalı; aksi halde
    # `tokens_per_step` ve birikim hesabı yanlış olur.
    plan.seq_len = seq_len
    # ⚠ ÖLÇÜLDÜ: AMP (fp16 GradScaler) ile ilk adımda
    #   `RuntimeError: CUDA error: unknown error` alındı (Pascal mimarisi).
    #   fp32 bu model boyutunda rahatça sığar (40M parametre).
    if args.amp_yok:
        plan.use_amp = False
    # ⚠ ÖLÇÜLDÜ: `nvlddmkm` Olay 153 (GPU sıfırlama) yoğun yükte oluşuyor.
    # Toplu işi küçültmek tek çekirdek süresini kısaltır → TDR riski düşer.
    if args.batch > 0:
        plan.batch_size = max(1, int(args.batch))
    if args.kaydet_her > 0:
        plan.save_every = max(1, int(args.kaydet_her))
    # ⚠ Küçük toplu işte etkin boyutu korumak için birikim artırılır
    if plan.batch_size < 16:
        plan.grad_accumulation = max(plan.grad_accumulation,
                                     max(1, 16 // plan.batch_size))
    print(f"\n  plan         : {plan.steps} adım · batch {plan.batch_size} "
          f"· accum {plan.grad_accumulation} · seq {plan.seq_len}")
    print(f"                 lr {plan.learning_rate:.2e} · "
          f"eval her {plan.eval_every} adım")

    # ── Eğitim ─────────────────────────────────────────────────────────
    # ⚠ İki hedef var (belge devamı + talimat). Eğitim sırasındaki
    # doğrulama **birleşik** kümedir; tur sonunda ikisi **ayrı** ölçülür.
    vd_all = TrainingDataset(m.tokenizer, seq_len,
                             max_samples=len(vd) + len(td) + 16)
    for s_ in vd.samples:
        vd_all.add(s_.prompt, s_.target, s_.weight, s_.tags, s_.difficulty)
    for s_ in td.samples:
        vd_all.add(s_.prompt, s_.target, s_.weight, s_.tags, s_.difficulty)

    print("\n── EĞİTİM ──")
    t2 = time.perf_counter()
    sonuc = e.trainer.train(
        m, ds, plan=plan,
        validation=vd_all if not vd_all.is_empty else None,
        max_minutes=args.dakika, user_initiated=True,
    )
    sure = time.perf_counter() - t2
    s = sonuc.to_dict()

    # ⚠ Ayrı ölçüm: belge vs talimat
    v_pool = e.trainer.evaluate(m, vd) if not vd.is_empty else {}
    v_tal = e.trainer.evaluate(m, td) if not td.is_empty else {}

    print(f"\n  adım         : {s.get('steps_completed')}/{plan.steps} "
          f"· {sure / 60:.1f} dk · durdu: {s.get('stop_reason')}")
    print(f"  EĞİTİM kaybı : {s.get('initial_loss')} → "
          f"{s.get('final_loss')}")
    print(f"  DOĞRULAMA    : {s.get('initial_val_loss')} → "
          f"{s.get('final_val_loss')}   ← birleşik")
    if v_pool:
        print(f"    belge       : kayıp {v_pool.get('loss')} · "
              f"ppl {v_pool.get('perplexity')}")
    if v_tal:
        print(f"    talimat     : kayıp {v_tal.get('loss')} · "
              f"ppl {v_tal.get('perplexity')}")

    vh = s.get("val_history") or []
    if vh:
        print("\n  doğrulama trendi:")
        ilk = vh[0].get("loss")
        for v in vh:
            k = v.get("loss")
            deg = ""
            if ilk and k is not None:
                deg = f" ({(k - ilk) / ilk * 100:+.1f}%)"
            print(f"    adım {v.get('step'):>4} · kayıp {k} · "
                  f"ppl {v.get('perplexity')}{deg}")

    # ── Yorum ──────────────────────────────────────────────────────────
    iv = s.get("initial_val_loss") or 0
    fv = s.get("final_val_loss") or 0
    print("\n" + "=" * 74)
    if iv > 0 and fv > 0:
        dusus = (iv - fv) / iv * 100
        print(f"  doğrulama düşüşü: %{dusus:.2f}")
        if dusus >= 3:
            karar = "ÖĞRENİYOR — doğrulama kaybı düşüyor"
        elif dusus > 0:
            karar = "AZ ÖĞRENİYOR — düşüş %3'ün altında"
        else:
            karar = "ÖĞRENMİYOR — doğrulama kaybı düşmedi"
        print(f"  KARAR: {karar}")
    else:
        karar = "doğrulama ölçülemedi"
        print(f"  ⚠ {karar}")
    print("=" * 74)

    # ── Kaydet ─────────────────────────────────────────────────────────
    if not args.deneme:
        try:
            m.save()
            print("  model kaydedildi ✓")
        except Exception as exc:
            print(f"  ⚠ model kaydedilemedi: {type(exc).__name__}: {exc}")
    else:
        print("  (deneme modu — model kaydedilmedi)")

    RAPOR.write_text(json.dumps({
        "adim": s.get("steps_completed"), "sure_dk": round(sure / 60, 1),
        "egitim_kaybi": [s.get("initial_loss"), s.get("final_loss")],
        "dogrulama_kaybi": [iv, fv],
        "val_history": vh, "karar": karar,
        "ornek": len(ds), "val_ornek": len(vd),
        "rastgele_ref": rast, "plan": s.get("plan"),
        "stop_reason": s.get("stop_reason"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  rapor → {RAPOR}")
    e.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
