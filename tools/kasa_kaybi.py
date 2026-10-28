"""
Aşama 0b — "0,083 nereden geliyor?" kesin kanıt
================================================
⚠ Hipotez
---------
`train_from_vault(..., max_notes=500)` → model **en fazla 500 kasa notu**
ile eğitildi (~1000 örnek). Havuzdaki **61.195 kayıt** hiç kullanılmadı.

Eğer doğruysa:
    kasa notlarındaki kayıp  ≈ 0,083  (eğiticinin bildirdiği)
    havuzdaki kayıp          ≈ 6,44   (Aşama 0 ölçümü)

Bu betik kasa veri kümesini kurar ve kaybı ölçer.

Kullanım
--------
    python tools/kasa_kaybi.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def main() -> int:
    print("=" * 74)
    print("  AŞAMA 0b — KASA KAYBI (0,083 hipotezi)")
    print("=" * 74)

    from neurocore import NeuroCoreEngine

    t0 = time.perf_counter()
    e = NeuroCoreEngine()
    e.initialize()
    m = e.model
    print(f"\n  model    : {m.growth.total_training_steps:,} adım · "
          f"{m.growth.total_tokens_seen:,} token")
    print(f"  yükleme  : {time.perf_counter() - t0:.1f} sn")

    # ── 1. Kasa kaç not? ────────────────────────────────────────────────
    # ⚠ Ölçüldü: `e.vault` YOK; kasa `e.knowledge` (KnowledgeLayer)
    # üzerinden erişilir → `e.knowledge.vault`
    katman = getattr(e, "knowledge", None)
    vault = getattr(katman, "vault", None) if katman is not None else None
    if vault is None:
        vault = getattr(e, "vault", None)
    toplam_not = 0
    if vault is not None and hasattr(vault, "iter_notes"):
        try:
            toplam_not = sum(1 for _ in vault.iter_notes(limit=200000))
        except Exception:
            toplam_not = -1
    print(f"  kasada   : {toplam_not:,} not")

    # ── 2. Eğitimin kullandığı veri kümesi (max_notes=500) ─────────────
    ds500 = e.trainer.build_dataset_from_vault(
        vault, m.tokenizer, max_notes=500)
    print(f"  kasa ds  : {len(ds500)} örnek (max_notes=500 — VARSAYILAN)")

    ds_tum = e.trainer.build_dataset_from_vault(
        vault, m.tokenizer, max_notes=max(1, toplam_not or 500))
    print(f"  tam kasa : {len(ds_tum)} örnek (tüm notlar)")

    # ── 3. Kayıp ölçümü ────────────────────────────────────────────────
    sys.path.insert(0, str(_KOK / "tools"))
    from tools.dogrulama_kur import kayip_olc

    def _metinler(ds: Any, adet: int = 80) -> List[str]:
        """Örnekleri tam metne çevir (prompt + target)."""
        out: List[str] = []
        for s in list(ds.samples)[:adet]:
            out.append(f"{s.prompt}{s.target}")
        return out

    print("\n── KAYIP ──")
    sonuc: Dict[str, Any] = {}
    for ad, ds in (("kasa_500", ds500), ("kasa_tum", ds_tum)):
        metinler = _metinler(ds)
        if not metinler:
            continue
        r = kayip_olc(m, metinler)
        sonuc[ad] = r
        print(f"  {ad:10} kayıp={r['kayip']:>8} · token={r['token']:>6} "
              f"· metin={r['metin']:>3}")

    # ⚠ Karşılaştırma: Aşama 0'ın havuz ölçümü
    eski = Path("_dogrulama_olcumu.json")
    if eski.exists():
        try:
            d = json.loads(eski.read_text(encoding="utf-8"))
            h = (d.get("gorulmus") or {}).get("kayip")
            if h is not None:
                sonuc["havuz"] = {"kayip": h}
                print(f"  {'havuz':10} kayıp={h:>8}  (Aşama 0 — 60 metin)")
        except Exception:
            pass

    # ── 4. Karar ───────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    k500 = sonuc.get("kasa_500", {}).get("kayip")
    havuz = sonuc.get("havuz", {}).get("kayip")
    egitici = 0.0834  # tur 1-19 loglarındaki tipik değer
    if k500 is not None:
        fark = abs(k500 - egitici)
        print(f"  eğiticinin bildirdiği : {egitici}")
        print(f"  kasa_500 kaybı        : {k500}   (fark {fark:.4f})")
        if havuz is not None:
            print(f"  havuz kaybı           : {havuz}")
        if fark < 0.15:
            print("  ✅ HİPOTEZ DOĞRULANDI — 0,083 kasa notlarının kaybı")
            print("     Model 500 notu ezberledi; havuz hiç kullanılmadı.")
            sonuc["karar"] = "kasa_500_dogrulandi"
        else:
            print("  ⚠ Hipotez ZAYIF — 0,083 başka bir yerden geliyor")
            sonuc["karar"] = "belirsiz"
    print("=" * 74)

    Path("_kasa_kaybi.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  rapor → _kasa_kaybi.json")
    e.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
