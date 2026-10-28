"""
Ölçek Danışmanı — Veri mi, Model mi, Eğitim mi?
================================================
Yerel modelin **neden** akıcı üretemediğini ölçer ve ne yapılması
gerektiğini söyler. Tahmin yok; oranlar ve donanım sınırı ölçülür.

Neden gerekli?
    "Daha çok eğitelim" içgüdüsü yanlış olabilir. Ölçüm gösterdi:

        Model    : 208M parametre
        Veri     : 271.457 token
        Oran     : 0,0013 token/parametre

    Sağlıklı bir dil modeli için oran ~**20 token/parametre**'dir. Yani
    veri **~15.000× eksik**. Bu durumda:

        * Daha çok adım atmak → aynı veriyi tekrarlar → **ezberler**
        * Modeli büyütmek    → oranı **daha da kötüleştirir**
        * Doğru hamle        → **veri üretmek**

Danışman üç seçeneği karşılaştırır ve **ölçülen** sayılarla öneri verir.

Donanım sınırı (ölçüldü, GTX 1080 Ti / 11 GB):
    Eğitim ≈ 4 × parametre × 2,2 bayt
    → ~10 GB bütçede üst sınır **~740M parametre**

Kullanım:
    python tools/olcek_danismani.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

#: Sağlıklı token/parametre oranı (dil modeli literatürü ~20)
HEALTHY_RATIO = 20.0

#: Bu oranın altında model "veri açı" sayılır
MIN_ACCEPTABLE_RATIO = 1.0

#: Ölçülen donanım sınırı (1080 Ti, 11 GB, AMP)
MAX_TRAINABLE_PARAMS = 740_000_000

#: Eğitim belleği katsayısı: parametre × 4 bayt × 2,2 (grad + 2 moment)
BYTES_PER_PARAM = 4 * 2.2


def _p(*parts: Any) -> None:
    """Tamponsuz yazdırır."""
    print(" ".join(str(part) for part in parts), flush=True)


@dataclass
class ScaleVerdict:
    """Ölçek değerlendirmesinin sonucu."""
    params: int = 0
    tokens: int = 0
    samples: int = 0
    steps: int = 0
    loss: float = 0.0
    ratio: float = 0.0
    deficit: float = 0.0
    vram_needed_mb: float = 0.0
    vram_budget_mb: float = 0.0
    #: `data` | `model` | `train` | `balanced`
    bottleneck: str = ""
    advice: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "params": self.params, "tokens": self.tokens,
            "samples": self.samples, "steps": self.steps,
            "loss": round(self.loss, 4), "ratio": round(self.ratio, 6),
            "deficit_x": round(self.deficit, 1),
            "vram_needed_mb": round(self.vram_needed_mb),
            "vram_budget_mb": round(self.vram_budget_mb),
            "bottleneck": self.bottleneck, "advice": self.advice,
            "actions": self.actions,
        }


def training_vram_mb(params: int) -> float:
    """Verilen parametre sayısının eğitim belleği ihtiyacı (MB)."""
    return params * BYTES_PER_PARAM / 1024**2


def max_params_for_budget(budget_mb: float) -> int:
    """Bütçeye sığan en büyük parametre sayısı."""
    return int(budget_mb * 1024**2 / BYTES_PER_PARAM)


def assess(*, params: int, tokens: int, samples: int = 0, steps: int = 0,
           loss: float = 0.0,
           vram_budget_mb: float = 10_200.0) -> ScaleVerdict:
    """
    Ölçek durumunu değerlendirir ve **ne yapılması gerektiğini** söyler.

    Args:
        params: Modelin parametre sayısı
        tokens: Eğitim verisinin token sayısı
        samples: Örnek sayısı
        steps: Atılan eğitim adımı
        loss: Son kayıp
        vram_budget_mb: Kullanılabilir VRAM (MB)

    Returns:
        `ScaleVerdict` — darboğaz, gerekçe ve eylem listesi
    """
    sonuc = ScaleVerdict(params=params, tokens=tokens, samples=samples,
                         steps=steps, loss=loss,
                         vram_budget_mb=vram_budget_mb)
    if params <= 0:
        sonuc.bottleneck = "bilinmiyor"
        sonuc.advice = "parametre sayısı okunamadı"
        return sonuc

    # ── Oran ve eksik ──
    gerekli = params * HEALTHY_RATIO
    sonuc.ratio = tokens / params
    sonuc.deficit = gerekli / max(1, tokens)
    sonuc.vram_needed_mb = training_vram_mb(params)
    sonuc.actions = []

    # ── Darboğaz ──
    if sonuc.ratio >= HEALTHY_RATIO:
        sonuc.bottleneck = "dengeli"
        sonuc.advice = ("Veri/parametre oranı sağlıklı (≥20). Üretim kalitesi "
                        "yetersizse **modeli büyütmek** artık anlamlı.")
        sonuc.actions.append({
            "action": "model_buyut",
            "why": f"oran {sonuc.ratio:.1f} ≥ {HEALTHY_RATIO:.0f}",
            "max_params": min(MAX_TRAINABLE_PARAMS,
                              max_params_for_budget(vram_budget_mb)),
        })
    elif sonuc.ratio >= MIN_ACCEPTABLE_RATIO:
        sonuc.bottleneck = "train"
        sonuc.advice = ("Oran kabul edilebilir ama sağlıklı değil. Veri "
                        "üretmeye devam edip **eğitimi sürdürmek** doğru.")
        sonuc.actions.append({
            "action": "veri_uret_ve_egit",
            "why": f"oran {sonuc.ratio:.2f} < {HEALTHY_RATIO:.0f}",
            "missing_tokens": int(max(0, gerekli - tokens)),
        })
    else:
        # ⚠ Asıl durum: model veriye göre **çok büyük**
        sonuc.bottleneck = "data"
        sonuc.advice = (
            f"Model veriye göre çok büyük: {sonuc.ratio:.4f} token/parametre, "
            f"sağlıklı oran {HEALTHY_RATIO:.0f}. "
            f"~{sonuc.deficit:,.0f}× daha fazla veri gerekir. "
            "Daha çok adım atmak **aynı veriyi tekrarlar** → ezberler; "
            "modeli büyütmek oranı **daha da kötüleştirir**. "
            "Doğru hamle: **veri üretmek**."
        )
        sonuc.actions.append({
            "action": "veri_uret",
            "why": f"oran {sonuc.ratio:.4f} << {HEALTHY_RATIO:.0f}",
            "missing_tokens": int(max(0, gerekli - tokens)),
            "deficit_x": round(sonuc.deficit, 1),
        })
        # Alternatif: veriyle uyumlu model boyutu
        uygun = max(1, int(tokens / HEALTHY_RATIO))
        sonuc.actions.append({
            "action": "modeli_kucult_veya_veriyi_buyut",
            "why": "denge için iki yol var",
            "params_for_current_data": uygun,
            "current_params": params,
        })

    # ── Donanım denetimi ──
    if sonuc.vram_needed_mb > vram_budget_mb:
        sonuc.actions.append({
            "action": "donanim_yetersiz",
            "why": (f"eğitim {sonuc.vram_needed_mb:.0f} MB ister, "
                    f"bütçe {vram_budget_mb:.0f} MB"),
            "max_params": max_params_for_budget(vram_budget_mb),
        })
    return sonuc


def measure_live() -> Dict[str, Any]:
    """
    Gerçek sistemden ve kasadan ölçüm yapar.

    Model yoksa veya motor yüklenemezse bunu **açıkça** bildirir.
    """
    from neurocore.neural.scale import scale_report

    veri: Dict[str, Any] = {}
    try:
        veri = scale_report(audiences=3, knowledge_patterns=20)
    except Exception as exc:
        veri = {"error": str(exc)[:120]}

    model: Dict[str, Any] = {}
    try:
        from neurocore import NeuroCoreEngine

        motor = NeuroCoreEngine()
        motor.initialize()
        m = motor.model
        # NOT: `parameter_count()` yalnızca **etkin** (MoE seyrekliği
        # hesaba katılmış) sayıyı verir. Depolama/eğitim belleği için
        # **toplam** parametre gerekir: `sum(p.numel())`.
        toplam = int(sum(p.numel() for p in m.backbone.parameters()))
        model = {
            "params_active": int(m.parameter_count()),
            "params_total": toplam,
            "steps": int(m.growth.total_training_steps),
            "tokens_seen": int(m.growth.total_tokens_seen),
            "vocab": int(m.tokenizer.vocab_size),
            "experts": int(getattr(m.model_config, "num_experts", 0) or 0),
            "layers": int(getattr(m.model_config, "num_layers", 0) or 0),
            "can_generate": bool(m.can_generate),
        }
        motor.shutdown()
    except Exception as exc:
        model = {"error": str(exc)[:120]}
    return {"data": veri, "model": model}


def main() -> int:
    """Ölçümü yapar ve öneriyi yazdırır."""
    import argparse

    parser = argparse.ArgumentParser(description="Ölçek danışmanı")
    parser.add_argument("--params", type=int, default=0,
                        help="Parametre sayısı (0 = canlı ölç)")
    parser.add_argument("--tokens", type=int, default=0,
                        help="Veri token sayısı (0 = canlı ölç)")
    parser.add_argument("--json", action="store_true", help="JSON çıktı")
    args = parser.parse_args()

    _p("=" * 74)
    _p("  ÖLÇEK DANIŞMANI — VERİ Mİ, MODEL Mİ, EĞİTİM Mİ?")
    _p("=" * 74)

    if args.params and args.tokens:
        params, tokens = args.params, args.tokens
        kaynak = "elle verildi"
        samples = steps = 0
        loss = 0.0
    else:
        olcum = measure_live()
        model = olcum.get("model") or {}
        veri = olcum.get("data") or {}
        params = int(model.get("params_total") or 0)
        tokens = int(veri.get("tokens") or 0)
        samples = int(veri.get("samples") or 0)
        steps = int(model.get("steps") or 0)
        loss = float(model.get("loss") or 0.0)
        kaynak = "canlı ölçüm"
        _p(f"  model: {json.dumps(model, ensure_ascii=False)[:200]}")
        _p(f"  veri : {json.dumps({k: v for k, v in veri.items() if k != 'per_kind'}, ensure_ascii=False)[:200]}")
        _p("")

    if params <= 0 or tokens <= 0:
        _p("  HATA: parametre veya token sayısı ölçülemedi")
        return 1

    sonuc = assess(params=params, tokens=tokens, samples=samples,
                   steps=steps, loss=loss)

    _p(f"  Kaynak           : {kaynak}")
    _p(f"  Parametre (toplam): {sonuc.params:,}")
    _p(f"  Veri tokenu      : {sonuc.tokens:,}")
    _p(f"  Örnek            : {sonuc.samples:,}")
    _p(f"  Eğitim adımı     : {sonuc.steps:,}")
    _p("")
    _p(f"  ORAN             : {sonuc.ratio:.6f} token/parametre")
    _p(f"  Sağlıklı oran    : {HEALTHY_RATIO:.0f} token/parametre")
    _p(f"  EKSİK            : ~{sonuc.deficit:,.1f}× daha fazla veri")
    _p("")
    _p(f"  Eğitim VRAM      : {sonuc.vram_needed_mb:,.0f} MB "
       f"(bütçe {sonuc.vram_budget_mb:,.0f} MB)")
    _p(f"  Donanım sınırı   : ~{max_params_for_budget(sonuc.vram_budget_mb):,} parametre")
    _p("")
    _p(f"  DARBOĞAZ         : {sonuc.bottleneck.upper()}")
    _p(f"  ÖNERİ            : {sonuc.advice}")
    _p("")
    _p("  EYLEMLER:")
    for eylem in sonuc.actions:
        _p(f"    - {eylem.get('action')}: {eylem.get('why')}")
        for k, v in eylem.items():
            if k not in ("action", "why"):
                _p(f"        {k}: {v:,}" if isinstance(v, int) else f"        {k}: {v}")
    _p("=" * 74)

    if args.json:
        Path("_olcek_raporu.json").write_text(
            json.dumps(sonuc.to_dict(), ensure_ascii=False, indent=1),
            encoding="utf-8")
        _p("  rapor: _olcek_raporu.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
