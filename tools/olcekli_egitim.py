"""
Ölçekli Eğitim Hattı
=====================
`scale.py` ile üretilen **akıl yürütme verisiyle** modeli yeniden eğitir.

Akış:
    1. Ölçekli veri üret (bilgi + düşünme izleri)
    2. Tokenizer'ı bu veriyle yenile (sözlük büyür)
    3. Gömme katmanını yeniden boyutlandır
    4. Dersleri kasaya yaz (kalıcı bilgi)
    5. Uzun eğitim turu çalıştır
    6. **Düşünme ölçümü**: model akıl yürütme istemine adım adım yanıt
       verebiliyor mu?

Neden bu hat ayrı?
    `pipeline.py` müfredatı yönetir ama tek turda ~70 adım atar. Ölçek
    büyütme için **uzun, kesintisiz** tura ihtiyaç var; bu betik onu yapar ve
    ilerlemeyi tamponsuz yazar (uzun işlerde ilerleme görünmeli).
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from neurocore.neural.scale import generate_samples, scale_report  # noqa: E402


def _p(*parts: Any) -> None:
    """Tamponsuz yazdırır."""
    print(" ".join(str(part) for part in parts), flush=True)


def main() -> int:
    """Ölçekli eğitimi çalıştırır."""
    import argparse

    parser = argparse.ArgumentParser(description="Ölçekli akıl yürütme eğitimi")
    parser.add_argument("--adim", type=int, default=600, help="Eğitim adımı")
    parser.add_argument("--kitle", type=int, default=3, help="Hedef kitle sayısı")
    parser.add_argument("--dakika", type=float, default=90.0, help="Süre sınırı")
    parser.add_argument("--tokenizer-yenile", action="store_true",
                        help="Tokenizer'ı yeniden eğit (gömmeyi sıfırlar)")
    parser.add_argument("--olcum", action="store_true", help="Yalnızca ölçüm")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    from neurocore import NeuroCoreEngine

    _p("=" * 72)
    _p("  OLCekLI AKIL YURUTME EGITIMI")
    _p("=" * 72)

    motor = NeuroCoreEngine()
    motor.initialize()
    model = motor.model
    _p(f"  baslangic adimi : {model.growth.total_training_steps}")
    _p(f"  tokenizer       : {model.tokenizer.vocab_size}")

    # ── 1) Veri üret ──
    ornekler = generate_samples(audiences=args.kitle, knowledge_patterns=20)
    _p(f"  uretilen ornek  : {len(ornekler):,}")

    # ── 2) Tokenizer yenileme (isteğe bağlı) ──
    if args.tokenizer_yenile:
        from neurocore.neural.tokenizer import BPETokenizer

        docs = [f"{o.prompt}\n{o.target}" for o in ornekler]
        eski = int(model.tokenizer.vocab_size)
        _p(f"  tokenizer yenileniyor ({eski} → ...)")
        yeni = BPETokenizer()
        yeni.train(docs, vocab_size=8000, min_frequency=1, verbose=False)
        yeni.initialize(getattr(model, "_config", None), None)
        model.tokenizer = yeni
        satir = model.backbone.resize_vocab(int(yeni.vocab_size))
        model.model_config.vocab_size = int(yeni.vocab_size)
        _p(f"  tokenizer       : {eski} → {yeni.vocab_size} ({satir:+d} gomme satiri)")

    # ── 3) Doğrudan veri kümesi ──
    #
    # ⚠ Kasa yolu **kullanılmaz**: `build_dataset_from_vault` en fazla 500 not
    # alır ve her not yazımı gömme + sürümleme gerektirir (ölçüldü: 2.030 not
    # >7 dakika). Ölçekli eğitimde örnekler doğrudan veri kümesine yüklenir.
    from neurocore.neural.trainer import TrainingDataset

    dataset = TrainingDataset(model.tokenizer, seq_len=256)
    yuklenen = 0
    for ornek in ornekler:
        try:
            if dataset.add(ornek.prompt, ornek.target,
                           tags=[ornek.layer, ornek.kind],
                           difficulty=0.4 if ornek.kind == "bilgi" else 0.7):
                yuklenen += 1
        except Exception:
            pass
    _p(f"  veri kumesi     : {yuklenen:,}/{len(ornekler):,} ornek yuklendi")
    _p(f"  veri istatistik : {json.dumps(dataset.stats(), ensure_ascii=False)[:180]}")

    if args.olcum:
        motor.shutdown()
        return 0

    # ── 4) Eğitim ──
    _p("")
    _p(f"  EGITIM BASLIYOR ({args.adim} adim, sinir {args.dakika:.0f} dk)")
    _p("-" * 72)
    basladi = time.perf_counter()
    try:
        plan = motor.trainer.plan_training(
            estimated_samples=len(dataset), model=motor.model, steps=args.adim,
            rationale="ölçekli akıl yürütme",
        )
        _p(f"  plan: {plan.steps} adim | batch {plan.batch_size} | "
           f"seq {plan.seq_len} | accum {plan.grad_accumulation} | amp {plan.use_amp}")
        sonuc = motor.trainer.train(
            motor.model, dataset, plan=plan, max_minutes=args.dakika,
            user_initiated=True,
        ).to_dict()
    except Exception:
        _p("  EGITIM COKTU:")
        traceback.print_exc()
        motor.shutdown()
        return 1

    sure = time.perf_counter() - basladi
    _p("-" * 72)
    _p(f"  adim        : {sonuc.get('steps_completed')}/{sonuc.get('steps_requested')}")
    _p(f"  kayip       : {sonuc.get('initial_loss')} -> {sonuc.get('final_loss')}")
    _p(f"  token       : {sonuc.get('tokens_seen'):,}")
    _p(f"  neden       : {sonuc.get('stop_reason')}")
    _p(f"  sure        : {sure:.0f} sn")
    if sonuc.get("error"):
        _p(f"  HATA        : {str(sonuc['error'])[:160]}")

    motor.model.save()
    toplam = model.growth.total_training_steps
    _p(f"  MODEL       : {toplam} adim")

    # ── 5) Düşünme ölçümü ──
    _p("")
    _p("=" * 72)
    _p("  DUSUNME OLCUMU (akil yurutme istemi -> adim adim yanit)")
    _p("=" * 72)
    from neurocore.neural.reasoning import THINK_PREFIX

    testler = [
        f"{THINK_PREFIX} Blender'da küp eklemek için hangi yaklaşım daha uygun?",
        f"{THINK_PREFIX} Eğitim kaybı 0,24 ama üretim bozuk. Ne anlama geliyor?",
        f"{THINK_PREFIX} Blender betiği çalışmıyor, neden olabilir?",
        f"{THINK_PREFIX} Ağ istekleri yavaş — ne yapmalıyım?",
    ]
    for istem in testler:
        r = model.generate_text(istem, max_new_tokens=140, temperature=0.3)
        metin = (r.get("text") or "").strip()
        _p("")
        _p(f"  ISTEM : {istem[:78]}")
        _p(f"  CIKTI : {metin[:400] if metin else '(bos)'}")

    motor.shutdown()
    _p("")
    _p("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
