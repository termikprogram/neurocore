"""
Havuz Hazırlama v2 — Tekilleştirme + bilgilendirici istem
==========================================================
⚠ Neden v2? (Aşama 4 ÖLÇÜMÜ)
----------------------------
```
eğitim    : 55.784 kayıt → 26.110 benzersiz başlık  (%69 mükerrer)
            217× "# javascript · index.js"
            196× "# python · utils.py"
```

İki sorun vardı:
```
1. MÜKERRER İÇERİK : aynı belge birden çok kez
2. BELİRSİZ İSTEM  : istem yalnız başlıktı ("# python · utils.py")
                     → yüzlerce FARKLI devam aynı isteme bağlı
                     → model o konumda tutarlı bir şey öğrenemez
```

v2 çözümü
---------
```
1. İçerik hash'i (sha256) ile **tekilleştir**
2. İstem = belgenin **KENDİ ilk parçası** (başlık değil)
   → her istem benzersiz ve bilgilendirici
3. Bölme **başlık grubu** düzeyinde (kardeş belgeler aynı tarafa)
   → sızıntı yok
```

Kullanım
--------
    python tools/havuz_hazirla.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

HAVUZ = Path(".neurocore/egitim_havuzu.jsonl")
EGITIM_V2 = Path(".neurocore/egitim_dilimi_v2.jsonl")
DOGRULAMA_V2 = Path(".neurocore/dogrulama_kumesi_v2.jsonl")
RAPOR = Path("_havuz_hazirla.json")

VAL_ORAN = 0.08
#: ⚠ İstem uzunluğu: belgenin kendi ilk parçası
ISTEM_MIN = 120
ISTEM_MAX = 400


def _normal(metin: str) -> str:
    """Hash için normalleştirme (boşluk farkları tekilleştirmeyi bölmesin)."""
    return re.sub(r"\s+", " ", metin).strip().lower()


def _hash(metin: str) -> str:
    return hashlib.sha256(_normal(metin).encode("utf-8")).hexdigest()


def _baslik(metin: str) -> str:
    """Başlık (ilk satır) — bölme grubu anahtarı."""
    return metin.strip().split("\n", 1)[0][:120]


def istem_hedef(metin: str) -> Tuple[str, str]:
    """
    Belgeyi (istem, hedef) çiftine ayırır.

    ⚠ v1'de istem yalnız **başlık**tı; aynı başlık yüzlerce belgede
    geçtiği için istem ayırt edici değildi. v2'de istem belgenin
    **kendi ilk parçası**: satır sonuna denk gelen, ~%30 uzunlukta,
    en az `ISTEM_MIN` en çok `ISTEM_MAX` karakter.
    """
    m = metin.strip()
    if len(m) < ISTEM_MIN + 40:
        return m[:ISTEM_MIN], m[ISTEM_MIN:]

    hedef_uzun = max(ISTEM_MIN, min(ISTEM_MAX, int(len(m) * 0.30)))
    # ⚠ Satır sonuna hizala (kod yapısı bozulmasın)
    kes = m.rfind("\n", ISTEM_MIN, hedef_uzun)
    if kes < ISTEM_MIN:
        kes = hedef_uzun
    return m[:kes].strip(), m[kes:].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Havuz v2 hazırlama")
    ap.add_argument("--tohum", type=int, default=42)
    ap.add_argument("--val-oran", type=float, default=VAL_ORAN)
    args = ap.parse_args()

    print("=" * 74)
    print("  HAVUZ HAZIRLAMA v2 — tekilleştirme + bilgilendirici istem")
    print("=" * 74)

    ham: List[str] = []
    with HAVUZ.open(encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir:
                continue
            try:
                d = json.loads(satir)
            except json.JSONDecodeError:
                continue
            m = (d.get("metin") or "").strip()
            if m:
                ham.append(m)
    print(f"\n  ham kayıt        : {len(ham):,}")

    # ── Tekilleştirme ──────────────────────────────────────────────────
    tekil: Dict[str, str] = {}
    mukerrer = 0
    for m in ham:
        h = _hash(m)
        if h in tekil:
            mukerrer += 1
            continue
        tekil[h] = m
    print(f"  mükerrer (içerik): {mukerrer:,} "
          f"(%{100 * mukerrer // max(1, len(ham))})")
    print(f"  benzersiz belge  : {len(tekil):,}")

    # ── Kısa belgeleri at (devam eğitimi için yetersiz) ────────────────
    belgeler = [m for m in tekil.values() if len(m) >= ISTEM_MIN + 40]
    print(f"  kullanılabilir   : {len(belgeler):,} "
          f"(≥{ISTEM_MIN + 40} karakter)")

    # ── Başlık grubu düzeyinde bölme ───────────────────────────────────
    gruplar: Dict[str, List[str]] = {}
    for m in belgeler:
        gruplar.setdefault(_baslik(m), []).append(m)
    anahtarlar = sorted(gruplar)
    random.Random(args.tohum).shuffle(anahtarlar)
    kes = max(1, int(len(anahtarlar) * max(0.01, min(0.5, args.val_oran))))
    val_anahtar = set(anahtarlar[:kes])

    egitim: List[str] = []
    dogrulama: List[str] = []
    for a in anahtarlar:
        (dogrulama if a in val_anahtar else egitim).extend(gruplar[a])

    # ── Sızıntı denetimi ───────────────────────────────────────────────
    h_eg = {_hash(m) for m in egitim}
    h_dog = {_hash(m) for m in dogrulama}
    sizinti = h_eg & h_dog
    b_eg = {_baslik(m) for m in egitim}
    b_dog = {_baslik(m) for m in dogrulama}

    print(f"\n  eğitim           : {len(egitim):,} belge · "
          f"{len(b_eg):,} başlık grubu")
    print(f"  doğrulama        : {len(dogrulama):,} belge · "
          f"{len(b_dog):,} başlık grubu")
    print(f"  içerik sızıntısı : {len(sizinti)}  "
          f"{'✓' if not sizinti else '✗ HATA'}")
    print(f"  başlık kesişimi  : {len(b_eg & b_dog)}  "
          f"{'✓' if not (b_eg & b_dog) else '✗ HATA'}")

    # ── İstem/hedef örneği ─────────────────────────────────────────────
    print("\n  ── istem örnekleri ──")
    for m in dogrulama[:2]:
        p, t = istem_hedef(m)
        print(f"    istem ({len(p)} kr): {p[:70]!r}")
        print(f"    hedef ({len(t)} kr): {t[:70]!r}")

    # ── Yaz ────────────────────────────────────────────────────────────
    for yol, veri in ((EGITIM_V2, egitim), (DOGRULAMA_V2, dogrulama)):
        with yol.open("w", encoding="utf-8") as f:
            for m in veri:
                f.write(json.dumps({"metin": m}, ensure_ascii=False) + "\n")
        print(f"\n  yazıldı: {yol} ({len(veri):,})")

    ozet = {
        "ham": len(ham), "mukerrer": mukerrer, "tekil": len(tekil),
        "egitim": len(egitim), "dogrulama": len(dogrulama),
        "sizinti": len(sizinti), "baslik_kesisim": len(b_eg & b_dog),
        "istem_min": ISTEM_MIN, "istem_max": ISTEM_MAX,
    }
    RAPOR.write_text(json.dumps(ozet, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    print(f"  rapor → {RAPOR}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
