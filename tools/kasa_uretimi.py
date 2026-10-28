"""
Aşama 0c — Model kasa metnini ÜRETEBİLİYOR mu?
===============================================
⚠ Neden?
--------
Kasa kaybı 1,44 (perplexity ~4,2) → model kasa üslubunu **bilmesi**
gerekir. Ama serbest sorularda çöp üretiyor. Çelişki.

Bu betik kasa başlıklarını istem olarak verir ve üretimi kasa
metniyle karşılaştırır. Ayrıca model/tokenizer tutarlılığını denetler.

Kullanım
--------
    python tools/kasa_uretimi.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def kelime_isabeti(uretilen: str, gercek: str) -> Dict[str, Any]:
    """Üretilen metnin kaç kelimesi gerçek kasa metninde geçiyor?"""
    import re

    def kel(x: str) -> List[str]:
        return re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]{4,}", x.lower())

    u = kel(uretilen)
    g = set(kel(gercek))
    if not u:
        return {"kelime": 0, "isabet": 0, "oran": 0.0}
    isabet = sum(1 for w in u if w in g)
    return {"kelime": len(u), "isabet": isabet,
            "oran": round(isabet / len(u), 3)}


def main() -> int:
    print("=" * 74)
    print("  AŞAMA 0c — KASA METNİ ÜRETİMİ")
    print("=" * 74)

    from neurocore import NeuroCoreEngine

    e = NeuroCoreEngine()
    e.initialize()
    m = e.model

    # ── 1. Tutarlılık denetimi ─────────────────────────────────────────
    print("\n── TUTARLILIK ──")
    try:
        print(f"  model vocab      : {m.model_config.vocab_size}")
        print(f"  tokenizer vocab  : {m.tokenizer.vocab_size}")
        tutarli = m.model_config.vocab_size == m.tokenizer.vocab_size
        print(f"  eşleşme          : {'✓' if tutarli else '✗ UYUMSUZ'}")
    except Exception as exc:
        print(f"  ✗ okunamadı: {type(exc).__name__}: {exc}")

    # ── 2. Kasa notlarından istem ───────────────────────────────────────
    katman = getattr(e, "knowledge", None)
    vault = getattr(katman, "vault", None) if katman is not None else None
    notlar: List[Any] = []
    if vault is not None:
        try:
            for n in vault.iter_notes(limit=8):
                if n.body and n.word_count >= 20:
                    notlar.append(n)
                if len(notlar) >= 5:
                    break
        except Exception as exc:
            print(f"  kasa okunamadı: {type(exc).__name__}")

    print(f"\n── ÜRETİM (kasa başlıklarından) ── {len(notlar)} not")
    sonuc: List[Dict[str, Any]] = []
    toplam_u = 0
    toplam_i = 0
    for n in notlar:
        istem = f"# {n.title}\n"
        try:
            r = m.generate_text(istem, max_new_tokens=60, temperature=0.7)
        except Exception as exc:
            print(f"  ✗ {n.title[:40]}: {type(exc).__name__}")
            continue
        metin = (r.get("text") or "").strip()
        yeni = metin[len(istem):].strip() if metin.startswith(istem) else metin
        k = kelime_isabeti(yeni, n.body)
        toplam_u += k["kelime"]
        toplam_i += k["isabet"]
        print(f"\n  ── {n.title[:50]}")
        print(f"     kasa gövdesi : {n.body[:110]!r}")
        print(f"     model çıktısı: {yeni[:110]!r}")
        print(f"     kelime isabeti: {k['isabet']}/{k['kelime']} "
              f"(%{k['oran'] * 100:.0f}) · red={r.get('quality_rejected')}")
        sonuc.append({"baslik": n.title, "cikti": yeni, **k,
                      "red": bool(r.get("quality_rejected"))})

    oran = toplam_i / max(1, toplam_u)
    print("\n" + "=" * 74)
    print(f"  TOPLAM kelime isabeti: {toplam_i}/{toplam_u} = "
          f"%{oran * 100:.1f}")
    if oran >= 0.35:
        karar = "kasa üslubunu KISMEN öğrenmiş"
    else:
        karar = "kasa metnini ÜRETEMİYOR — kayıp 1,44'e rağmen"
    print(f"  KARAR: {karar}")
    print("=" * 74)

    Path("_kasa_uretimi.json").write_text(
        json.dumps({"oran": oran, "karar": karar, "ornek": sonuc},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("  rapor → _kasa_uretimi.json")
    e.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
