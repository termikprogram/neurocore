"""
Ölçekli Düşünme Eğitimi — Uzun Süreli Çalıştırıcı
==================================================
Modeli **akıl yürütme verisiyle** uzun süre eğitir; her turda kaydeder ve
ilerlemeyi diske yazar. Kesintiye dayanıklıdır: yeniden başlatıldığında
kaldığı yerden devam eder.

Neden ayrı çalıştırıcı?
    Tek bir `engine.train` çağrısı dakikalarla sınırlıdır ve süreç çökerse
    ilerleme kaydedilmez. Ölçek büyütme **saatler** sürer; bu yüzden:

        * Her tur ayrı `train()` çağrısı → ilerleme kaydedilir
        * Kaynak kapısı yüzünden tur atlanırsa **beklenir**, çökülmez
        * Model her turda diske yazılır → veri kaybı yok

Kullanım:
    python tools/uzun_dusunme.py --tur 20 --adim 150 --dakika 20
"""

from __future__ import annotations

import argparse
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


def _p(*parts: Any) -> None:
    """Tamponsuz yazdırır."""
    print(time.strftime("%H:%M:%S"), "|", " ".join(str(x) for x in parts),
          flush=True)


def main() -> int:
    """Uzun süreli düşünme eğitimini çalıştırır."""
    parser = argparse.ArgumentParser(description="Ölçekli düşünme eğitimi")
    parser.add_argument("--tur", type=int, default=20, help="Tur sayısı")
    parser.add_argument("--adim", type=int, default=150, help="Tur başına adım")
    parser.add_argument("--dakika", type=float, default=20.0,
                        help="Tur başına süre sınırı (dakika)")
    parser.add_argument("--kitle", type=int, default=3, help="Hedef kitle sayısı")
    parser.add_argument("--tokenizer-yenile", action="store_true",
                        help="Tokenizer'ı yenile (gömmeyi sıfırlar)")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)

    from neurocore import NeuroCoreEngine
    from neurocore.neural.scale import generate_samples
    from neurocore.neural.trainer import TrainingDataset

    _p("=" * 72)
    _p("  OLCekLI DUSUNME EGITIMI (uzun sureli)")
    _p("=" * 72)

    motor = NeuroCoreEngine()
    motor.initialize()
    model = motor.model
    _p(f"baslangic: {model.growth.total_training_steps} adim | "
       f"tokenizer {model.tokenizer.vocab_size}")

    # ── Veri (bir kez üretilir, turlar boyunca yeniden kullanılır) ──
    ornekler = generate_samples(audiences=args.kitle, knowledge_patterns=20)
    _p(f"veri: {len(ornekler):,} ornek uretildi")

    if args.tokenizer_yenile:
        from neurocore.neural.tokenizer import BPETokenizer

        docs = [f"{o.prompt}\n{o.target}" for o in ornekler]
        eski = int(model.tokenizer.vocab_size)
        yeni = BPETokenizer()
        yeni.train(docs, vocab_size=8000, min_frequency=1, verbose=False)
        yeni.initialize(getattr(model, "_config", None), None)
        model.tokenizer = yeni
        satir = model.backbone.resize_vocab(int(yeni.vocab_size))
        model.model_config.vocab_size = int(yeni.vocab_size)
        _p(f"tokenizer: {eski} -> {yeni.vocab_size} ({satir:+d} satir)")

    def veri_kumesi() -> Any:
        """Taze veri kümesi kurar (her turda yeniden karıştırma için)."""
        ds = TrainingDataset(model.tokenizer, seq_len=256)
        for ornek in ornekler:
            try:
                ds.add(ornek.prompt, ornek.target,
                       tags=[ornek.layer, ornek.kind],
                       difficulty=0.4 if ornek.kind == "bilgi" else 0.7)
            except Exception:
                pass
        return ds

    ds = veri_kumesi()
    _p(f"veri kumesi: {json.dumps(ds.stats(), ensure_ascii=False)[:150]}")

    gecmis: List[Dict[str, Any]] = []
    basladi = time.time()

    for tur in range(1, args.tur + 1):
        onceki = int(model.growth.total_training_steps)
        _p("")
        _p(f"--- TUR {tur}/{args.tur} (model {onceki} adim) ---")

        # Kaynak kapısı: uygun değilse bekle, çökme
        uygun, neden = True, ""
        try:
            uygun, neden = motor.resources_available(for_heavy_work=True)
        except Exception:
            pass
        if not uygun:
            _p(f"kaynaklar uygun degil: {neden[:70]} — 30 sn bekleniyor")
            for _ in range(6):
                time.sleep(30)
                uygun, neden = motor.resources_available(for_heavy_work=True)
                if uygun:
                    break
            if not uygun:
                _p("kaynaklar acilmadi — tur atlandi")
                gecmis.append({"tur": tur, "ok": False, "neden": neden[:80]})
                continue

        try:
            plan = motor.trainer.plan_training(
                estimated_samples=len(ds), model=model, steps=args.adim,
                rationale="uzun düşünme eğitimi",
            )
            t0 = time.perf_counter()
            sonuc = motor.trainer.train(
                model, ds, plan=plan, max_minutes=args.dakika,
                user_initiated=True,
            ).to_dict()
            sure = time.perf_counter() - t0
        except Exception:
            _p("TUR COKTU:")
            traceback.print_exc()
            gecmis.append({"tur": tur, "ok": False, "neden": "istisna"})
            continue

        simdiki = int(model.growth.total_training_steps)
        kayit = {
            "tur": tur,
            "adim": sonuc.get("steps_completed"),
            "istenen": sonuc.get("steps_requested"),
            "kayip_ilk": sonuc.get("initial_loss"),
            "kayip_son": sonuc.get("final_loss"),
            "token": sonuc.get("tokens_seen"),
            "neden": sonuc.get("stop_reason"),
            "sure_s": round(sure, 1),
            "model_adim": simdiki,
            "ok": bool(sonuc.get("steps_completed")),
        }
        gecmis.append(kayit)
        _p(f"  adim {kayit['adim']}/{kayit['istenen']} | "
           f"kayip {kayit['kayip_ilk']} -> {kayit['kayip_son']} | "
           f"{kayit['sure_s']}s | {kayit['neden']}")
        _p(f"  MODEL: {simdiki} adim")

        try:
            model.save()
        except Exception as exc:
            _p(f"  kayit hatasi: {exc}")

        # ── Tur arası VRAM boşaltma ──
        #
        # ⚠ Ölçülen tuzak: eğitim VRAM'i doldurur ve `shutdown()` tek başına
        # bırakmaz (1450 MiB'de kalır). Sonraki turun ön kontrolü
        # "VRAM %99 ≥ %90" görüp bekler. Bu yüzden her turdan sonra motor
        # kapatılır, çöp toplanır, CUDA önbelleği boşaltılır ve motor
        # yeniden yüklenir.
        try:
            model.save()
        except Exception:
            pass
        try:
            motor.shutdown()
        except Exception:
            pass
        model = None
        import gc as _gc

        _gc.collect()
        try:
            import torch as _t

            if _t.cuda.is_available():
                _t.cuda.empty_cache()
                _t.cuda.ipc_collect()
                _serbest, _toplam = _t.cuda.mem_get_info()
                _p(f"  VRAM bosaltildi: {( _toplam-_serbest)/1024**2:.0f} MB kullanimda")
        except Exception:
            pass
        time.sleep(2.0)
        motor = NeuroCoreEngine()
        motor.initialize()
        model = motor.model

        # İlerleme dosyası (kesintiye dayanıklı)
        Path(".neurocore/dusunme_egitimi.json").write_text(
            json.dumps({"gecmis": gecmis,
                        "model_adim": simdiki,
                        "tokenizer": int(model.tokenizer.vocab_size),
                        "toplam_sure_s": round(time.time() - basladi, 1),
                        "guncelleme": time.strftime("%H:%M:%S")},
                       ensure_ascii=False, indent=1), encoding="utf-8")

    # ── Özet ──
    _p("")
    _p("=" * 72)
    _p("  OZET")
    _p("=" * 72)
    basarili = [g for g in gecmis if g.get("ok")]
    _p(f"  tamamlanan tur : {len(basarili)}/{len(gecmis)}")
    _p(f"  toplam adim    : {sum(int(g.get('adim') or 0) for g in gecmis)}")
    _p(f"  model adimi    : {model.growth.total_training_steps}")
    _p(f"  gorulen token  : {model.growth.total_tokens_seen:,}")
    _p(f"  toplam sure    : {(time.time()-basladi)/60:.1f} dk")
    if basarili:
        _p(f"  kayip (ilk tur): {basarili[0].get('kayip_ilk')} -> "
           f"{basarili[-1].get('kayip_son')}")
    _p("=" * 72)
    motor.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
