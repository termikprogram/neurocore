"""
Model Sağlık Ölçümü — Eğitim mi Ezber mi?
==========================================
⚠ Ölçülen kritik sorun
----------------------
Eğitim 22 tur boyunca "kayıp 0,083 → 0,063, iyileşti" raporladı. Ama
**bağımsız ölçümde kayıp 4,99** çıktı. 79× fark.

Sebep: eğitim kaybı **ezberlenmiş partide** ölçülüyordu. Model aynı
veriyi 144 kez görmüştü — 0,063 "neredeyse mükemmel hatırlama"
demekti, "öğrenme" değil.

Bu araç iki ölçümü **ayırır**:
    * **EĞİTİM kaybı** — modelin gördüğü veri
    * **DOĞRULAMA kaybı** — modelin **görmediği** veri (held-out)

İkisi arasındaki fark **genelleme açığıdır**. Açık büyükse model
ezberliyor, öğrenmiyor.

Kullanım
--------
    python tools/model_saglik.py
    python tools/model_saglik.py --dogrulama-orani 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)


@dataclass
class SaglikRaporu:
    """Model sağlık ölçümü sonucu."""
    adim: int = 0
    token: int = 0
    parametre: int = 0
    vocab: int = 0
    egitim_kaybi: float = 0.0
    dogrulama_kaybi: float = 0.0
    rastgele_beklenti: float = 0.0
    egitim_ornek: int = 0
    dogrulama_ornek: int = 0
    egitim_adim: int = 0
    dogrulama_adim: int = 0
    egitim_neden: str = ""
    dogrulama_neden: str = ""
    acik: float = 0.0           # dogrulama - egitim
    acik_orani: float = 0.0     # dogrulama / egitim
    karar: str = ""
    oneriler: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adim": self.adim, "token": self.token,
            "parametre": self.parametre, "vocab": self.vocab,
            "egitim_kaybi": round(self.egitim_kaybi, 4),
            "dogrulama_kaybi": round(self.dogrulama_kaybi, 4),
            "rastgele_beklenti": round(self.rastgele_beklenti, 4),
            "egitim_ornek": self.egitim_ornek,
            "dogrulama_ornek": self.dogrulama_ornek,
            "acik": round(self.acik, 4),
            "acik_orani": round(self.acik_orani, 2),
            "karar": self.karar, "oneriler": self.oneriler,
        }


class SaglikOlcer:
    """Modelin ezber mi öğrenme mi yaptığını ölçer."""

    def __init__(self, *, dogrulama_orani: float = 0.20,
                 maks_not: int = 4000) -> None:
        self.dogrulama_orani = max(0.05, min(0.5, dogrulama_orani))
        self.maks_not = maks_not
        self._motor: Any = None

    def baglan(self) -> bool:
        """Motoru hazırlar."""
        try:
            from neurocore import NeuroCoreEngine

            self._motor = NeuroCoreEngine()
            self._motor.initialize()
            return True
        except Exception as exc:
            print(f"  motor başlatılamadı: {exc}")
            return False

    def _veri_bol(self) -> Tuple[Any, Any, int, int]:
        """
        Kasayı **eğitim** ve **doğrulama** olarak böler.

        ⚠ Kritik: doğrulama seti eğitime **sokulmaz**. Aksi hâlde
        ölçüm ezberi gizler (ölçülen hata).
        """
        import random

        motor = self._motor
        t = motor.trainer
        notlar = list(motor.knowledge.vault.iter_notes(limit=self.maks_not))
        # Tohumlu karıştırma — tekrarlanabilir
        rng = random.Random(1234)
        rng.shuffle(notlar)
        kesim = int(len(notlar) * (1.0 - self.dogrulama_orani))
        egitim_notlar = notlar[:kesim]
        dogrulama_notlar = notlar[kesim:]

        def kur(alt: List[Any]) -> Any:
            ds = t.build_dataset_from_vault(
                _ListeVault(alt), motor.model.tokenizer, seq_len=256,
                max_notes=len(alt) + 1)
            return ds

        return (kur(egitim_notlar), kur(dogrulama_notlar),
                len(egitim_notlar), len(dogrulama_notlar))

    def olc(self) -> SaglikRaporu:
        """Sağlık ölçümünü yapar."""
        import math

        m = self._motor
        t = m.trainer
        r = SaglikRaporu()
        # Modelin hazır olduğundan emin ol (`trainer.train` bunu kontrol
        # eder ve False ise `model_hazir_degil` ile 0 adım döner)
        if not getattr(m.model, "is_ready", False):
            try:
                m.model.initialize()
            except Exception as exc:
                print(f"  model hazırlanamadı: {str(exc)[:60]}")
        r.adim = int(getattr(m.model.growth, "total_training_steps", 0) or 0)
        r.token = int(getattr(m.model.growth, "total_tokens_seen", 0) or 0)
        r.parametre = int(m.model.parameter_count())
        r.vocab = int(getattr(m.model.tokenizer, "vocab_size", 0) or 0)
        r.rastgele_beklenti = math.log(max(2, r.vocab))

        egitim, dogrulama, ne, nd = self._veri_bol()

        # ── Eğitim kaybı ──
        # ⚠ `len(dataset)` bazı kurulumlarda 0 dönebiliyor; `samples`
        # özelliği daha güvenilir (ölçüldü).
        # ⚠ `dataset.samples` bir **liste** döndürüyor (sayı değil) —
        # `int()` çağrısı TypeError verir (ölçüldü). `len(dataset)`
        # güvenilirdir.
        def _adet(d: Any) -> int:
            try:
                return int(len(d))
            except Exception:
                try:
                    return int(len(getattr(d, "samples", []) or []))
                except Exception:
                    return 0

        n_eg = _adet(egitim)
        n_dg = _adet(dogrulama)
        r.egitim_ornek = n_eg
        r.dogrulama_ornek = n_dg
        if n_eg > 0:
            # ⚠ ÖLÇÜLEN HATA: burada `model=m` yazıyordu ama `m` **motor**
            # (`NeuroCoreEngine`), model değil. `trainer.train` `is_ready`
            # arar, motorda bulamaz ve `model_hazir_degil` ile **0 adım**
            # döner — ölçüm sessizce sıfır çıkıyordu.
            p = t.plan_training(estimated_samples=n_eg, steps=6,
                                model=m.model)
            s = t.train(model=m.model, dataset=egitim, plan=p,
                        user_initiated=True, save_checkpoints=False)
            # ⚠ ÖLÇÜLEN TUZAK: `s.initial_loss if s.initial_loss else ...`
            # yazımı **0,0 falsy** olduğu için zinciri bozuyordu ve eğitim
            # hiç çalışmadığında 0,0 raporluyordu ("ÖLÇÜLEMEDİ" yerine
            # sahte sıfır). Açık `is not None` kontrolü gerekir.
            if s.steps_completed and s.initial_loss is not None:
                r.egitim_kaybi = float(s.initial_loss)
            r.egitim_adim = int(s.steps_completed or 0)
            r.egitim_neden = str(s.stop_reason or "")

        # ── Doğrulama kaybı ──
        #
        # ⚠ `save_checkpoints=False` şart: doğrulama seti eğitime
        # sızmamalı. `user_initiated=False` de kaynak gözcüsünü yumuşatır.
        if n_dg > 0:
            p2 = t.plan_training(estimated_samples=n_dg, steps=4,
                                 model=m.model)
            for _ in range(2):
                p2.learning_rate = 0.0   # ağırlık değişmesin
            s2 = t.train(model=m.model, dataset=dogrulama, plan=p2,
                         user_initiated=True, save_checkpoints=False)
            if s2.steps_completed and s2.initial_loss is not None:
                r.dogrulama_kaybi = float(s2.initial_loss)
            r.dogrulama_adim = int(s2.steps_completed or 0)
            r.dogrulama_neden = str(s2.stop_reason or "")

        r.acik = r.dogrulama_kaybi - r.egitim_kaybi
        r.acik_orani = (r.dogrulama_kaybi / r.egitim_kaybi
                        if r.egitim_kaybi > 1e-9 else 0.0)
        self._karar(r)
        return r

    @staticmethod
    def _karar(r: SaglikRaporu) -> None:
        """Ölçüme göre karar ve öneri üretir."""
        rast = r.rastgele_beklenti
        if r.egitim_adim == 0 or r.dogrulama_adim == 0:
            r.karar = "OLCULEMEDI"
            r.oneriler.append(
                f"Eğitim çalışmadı (adım: eğitim={r.egitim_adim}, "
                f"doğrulama={r.dogrulama_adim})")
            if r.egitim_neden:
                r.oneriler.append(f"neden: {r.egitim_neden}")
            r.oneriler.append(
                "Kaynak gözcüsü duraklatmış olabilir — VRAM/RAM boşaltıp "
                "tekrar deneyin")
            return
        if r.egitim_kaybi <= 0 or r.dogrulama_kaybi <= 0:
            r.karar = "OLCULEMEDI"
            r.oneriler.append("Kayıp sıfır döndü — ölçüm yolu hatalı")
            return

        # 1) Model rastgeleden iyi mi?
        if r.dogrulama_kaybi >= rast * 0.95:
            r.karar = "OGRENMEMIS"
            r.oneriler.append(
                f"Doğrulama kaybı ({r.dogrulama_kaybi:.2f}) rastgele "
                f"beklentiye ({rast:.2f}) yakın — model hiçbir şey öğrenmemiş")
            r.oneriler.append("Modeli sıfırdan eğit; mevcut ağırlıklar değersiz")
            return

        # 2) Ezber mi?
        if r.acik_orani >= 3.0 and r.egitim_kaybi < 0.5:
            r.karar = "EZBERLIYOR"
            r.oneriler.append(
                f"Eğitim kaybı {r.egitim_kaybi:.4f} ama doğrulama "
                f"{r.dogrulama_kaybi:.4f} — {r.acik_orani:.1f}× açık")
            r.oneriler.append(
                "Model eğitim setini ezberlemiş, genelleme yapamıyor")
            r.oneriler.append(
                f"Veri/parametre oranı {r.token / max(1, r.parametre):.5f} "
                f"(sağlıklı: 20) — model **çok büyük** veya veri **çok az**")
            r.oneriler.append(
                "Çözüm: modeli küçült (211M → 20-30M) veya veriyi 10× artır")
            return

        # 3) Sağlıklı mı?
        if r.acik_orani < 1.5:
            r.karar = "OGRENIYOR"
            r.oneriler.append(
                f"Eğitim {r.egitim_kaybi:.4f} / doğrulama "
                f"{r.dogrulama_kaybi:.4f} — açık {r.acik_orani:.2f}×, makul")
            r.oneriler.append("Eğitime devam edilebilir")
            return

        r.karar = "KISMEN_EZBER"
        r.oneriler.append(
            f"Açık {r.acik_orani:.2f}× — orta düzey ezber")
        r.oneriler.append("Doğrulama kaybı izlenerek devam edilmeli")
        r.oneriler.append("Erken durdurma: doğrulama kaybı artarsa dur")


class _ListeVault:
    """`iter_notes` sağlayan basit kasa sarmalayıcı (veri bölme için)."""

    def __init__(self, notlar: List[Any]) -> None:
        self._notlar = notlar

    def iter_notes(self, limit: int = 1000) -> Any:
        for n in self._notlar[:limit]:
            yield n


def main() -> int:
    """Sağlık ölçümünü çalıştırır."""
    ap = argparse.ArgumentParser(description="Model sağlık ölçümü")
    ap.add_argument("--dogrulama-orani", type=float, default=0.20,
                    help="Doğrulama için ayrılacak oran")
    ap.add_argument("--maks-not", type=int, default=4000,
                    help="Kullanılacak en fazla not")
    args = ap.parse_args()

    print("=" * 78)
    print("  MODEL SAĞLIK ÖLÇÜMÜ — eğitim mi, ezber mi?")
    print("=" * 78)

    o = SaglikOlcer(dogrulama_orani=args.dogrulama_orani,
                    maks_not=args.maks_not)
    if not o.baglan():
        return 1
    r = o.olc()

    print(f"  adım          : {r.adim:,}")
    print(f"  token         : {r.token:,}")
    print(f"  parametre     : {r.parametre:,}")
    print(f"  vocab         : {r.vocab:,}")
    print()
    print(f"  eğitim örnek  : {r.egitim_ornek:,}")
    print(f"  doğrulama örn.: {r.dogrulama_ornek:,}")
    print()
    print(f"  EĞİTİM kaybı  : {r.egitim_kaybi:.4f}")
    print(f"  DOĞRULAMA     : {r.dogrulama_kaybi:.4f}")
    print(f"  rastgele      : {r.rastgele_beklenti:.4f}  (ln vocab)")
    print(f"  AÇIK          : {r.acik:.4f}  ({r.acik_orani:.2f}×)")
    print()
    print(f"  KARAR: {r.karar}")
    for x in r.oneriler:
        print(f"    - {x}")
    print()
    print(f"  veri/parametre: {r.token / max(1, r.parametre):.6f} "
          f"(sağlıklı 20, eksik "
          f"{20 / max(1e-9, r.token / max(1, r.parametre)):,.0f}×)")

    Path("_model_saglik.json").write_text(
        json.dumps(r.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"  rapor: _model_saglik.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
