"""
Hata ayıklama: eğitim çöküşünün **tam izini** yakalar.

Eğitim döngüsü istisnaları kırpılmış mesajla raporluyor; kök nedeni bulmak
için aynı eğitimi kısa adımla çalıştırıp tam traceback yazdırır.
"""

from __future__ import annotations

import logging
import sys
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

# Tam iz için NeuroCore kaydedicisini DEBUG'a al
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr,
                    format="%(levelname)s %(name)s: %(message)s")

from neurocore import NeuroCoreEngine  # noqa: E402


def main() -> int:
    """Eğitimi çalıştırır ve tam izi yazdırır."""
    engine = NeuroCoreEngine()
    engine.initialize()
    print(f"motor hazir | model adimi: {engine.model.growth.total_training_steps}",
          flush=True)

    dataset = engine.trainer.build_dataset_from_vault(
        engine.knowledge.vault, engine.model.tokenizer)
    print(f"veri kumesi: {len(dataset)} ornek", flush=True)

    # Eğiticinin çöküşünü yakalamak için doğrudan çağır
    try:
        plan = engine.trainer.plan_training(
            estimated_samples=len(dataset), model=engine.model,
            steps=200, rationale="hata ayiklama",
        )
        print(f"plan: adim={plan.steps} batch={plan.batch_size} "
              f"seq={plan.seq_len} amp={plan.use_amp} "
              f"accum={plan.grad_accumulation}", flush=True)

        result = engine.trainer.train(
            engine.model, dataset, plan=plan, max_minutes=20.0,
            user_initiated=True,
        )
        print(f"sonuc: ok={result.steps_completed > 0} "
              f"adim={result.steps_completed}/{result.steps_requested} "
              f"kayip={result.initial_loss:.4f}->{result.final_loss:.4f} "
              f"neden={result.stop_reason}", flush=True)
    except Exception:
        print("\n=== TAM IZ ===", flush=True)
        traceback.print_exc()
        return 1

    engine.model.save()
    print(f"model: {engine.model.growth.total_training_steps} adim", flush=True)
    engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
