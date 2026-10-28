"""
Ölçümlü Eğitim — Eğitim ve Doğrulama **Ayrı Süreçlerde**
==========================================================
⚠ Ölçülen mimari sorun
----------------------
Doğrulama ölçümünü eğitimle **aynı süreçte** yapmak CUDA bağlamını
bozuyor:

    tur 1: 86 adım, eğitim 1,563 | doğrulama None
    tur 2: 0 adım  [CUDA error: unknown error]
    tur 3+: 0 adım [CUDA error: unknown error]

Denenen ve **işe yaramayan** çözümler:
    * `torch.cuda.empty_cache()` — çözmedi
    * Doğrulamayı `trainer.train()` yerine saf ileri geçişle yapmak
    * Veri setini cihaza önceden taşımak (VRAM'i doldurdu)
    * `CUDA_LAUNCH_BLOCKING=1`

**Kök sebep:** `backbone.eval()` ↔ `backbone.train()` hızlı geçişi ve
optimize edici durumu aynı bağlamda çakışıyor.

Doğru mimari
------------
    ┌─────────────────┐         ┌──────────────────┐
    │  EĞİTİM SÜRECİ  │         │ DOĞRULAMA SÜRECİ │
    │  (uzun, GPU)    │         │  (kısa, GPU)     │
    │  N tur eğitir   │         │  kayıp ölçer     │
    │  kaydeder       │         │  çıkar           │
    └────────┬────────┘         └────────▲─────────┘
             │ model.pt                  │
             └───────────────────────────┘
                  her X turda

Her doğrulama **temiz bir süreçte** çalışır — CUDA bağlamı bozulmaz.

Kullanım
--------
    python tools/olcumlu_egitim.py --tur 12 --adim 150 --olcum-arasi 3
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

ILERLEME = Path(".neurocore/olcumlu_egitim.json")


@dataclass
class Kayit:
    """Bir eğitim + ölçüm turunun kaydı."""
    no: int = 0
    adim: int = 0
    egitim_kaybi: Optional[float] = None
    dogrulama_kaybi: Optional[float] = None
    acik: float = 0.0
    acik_orani: float = 0.0
    egitim_s: float = 0.0
    olcum_s: float = 0.0
    not_: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "no": self.no, "adim": self.adim,
            "egitim_kaybi": self.egitim_kaybi,
            "dogrulama_kaybi": self.dogrulama_kaybi,
            "acik": round(self.acik, 4),
            "acik_orani": round(self.acik_orani, 3),
            "egitim_s": round(self.egitim_s, 1),
            "olcum_s": round(self.olcum_s, 1),
            "not": self.not_[:80],
        }


class OlcumluEgitim:
    """Eğitimi ve doğrulamayı **ayrı süreçlerde** çalıştırır."""

    def __init__(self, *, tur: int = 12, adim: int = 150,
                 olcum_arasi: int = 3, dakika: float = 18.0) -> None:
        self.tur_sayisi = max(1, tur)
        self.adim = max(5, adim)
        self.olcum_arasi = max(1, olcum_arasi)
        self.dakika = dakika
        self.kayitlar: List[Kayit] = []
        self._egitim_betik = _KOK / "tools" / "taze_egitim.py"
        self._olcum_betik = _KOK / "tools" / "model_saglik.py"

    # ─── Alt süreçler ───

    def _egit(self, adim: int) -> Dict[str, Any]:
        """Bir eğitim turunu **ayrı süreçte** çalıştırır."""
        komut = [sys.executable, "-u", "-X", "utf8",
                 str(self._egitim_betik), "--tek-tur",
                 "--adim", str(adim), "--dakika", str(self.dakika)]
        try:
            s = subprocess.run(komut, capture_output=True, text=True,
                               timeout=self.dakika * 60 + 900,
                               cwd=str(_KOK), encoding="utf-8",
                               errors="replace")
            for satir in (s.stdout or "").splitlines():
                if satir.startswith("NC_RESULT "):
                    return json.loads(satir[len("NC_RESULT "):])
        except Exception as exc:
            return {"hata": f"{type(exc).__name__}: {str(exc)[:60]}"}
        return {"hata": "sonuç satırı yok"}

    def _olc(self) -> Optional[Dict[str, Any]]:
        """
        Sağlık ölçümünü **ayrı süreçte** çalıştırır.

        ⚠ Ayrı süreç şart: aynı süreçte doğrulama CUDA bağlamını
        bozuyor (ölçüldü: turlar 2+ "CUDA error: unknown error").
        """
        komut = [sys.executable, "-u", "-X", "utf8",
                 str(self._olcum_betik), "--maks-not", "1500"]
        try:
            s = subprocess.run(komut, capture_output=True, text=True,
                               timeout=1800, cwd=str(_KOK),
                               encoding="utf-8", errors="replace")
            yol = _KOK / "_model_saglik.json"
            if yol.exists():
                return json.loads(yol.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"hata": f"{type(exc).__name__}: {str(exc)[:60]}"}
        return {"hata": "sağlık raporu üretilemedi"}

    # ─── Ana döngü ───

    def calistir(self) -> Dict[str, Any]:
        """Ölçümlü eğitimi çalıştırır."""
        print("=" * 78)
        print(f"  ÖLÇÜMLÜ EĞİTİM — {self.tur_sayisi} tur × {self.adim} adım")
        print(f"  doğrulama ölçümü: her {self.olcum_arasi} turda "
              f"(AYRI SÜREÇ)")
        print("=" * 78)

        for i in range(self.tur_sayisi):
            no = i + 1
            k = Kayit(no=no)

            # 1) Eğitim (ayrı süreç)
            t0 = time.perf_counter()
            v = self._egit(self.adim)
            k.egitim_s = time.perf_counter() - t0
            k.adim = int(v.get("steps_completed") or 0)
            k.egitim_kaybi = v.get("final_loss")
            if v.get("hata"):
                k.not_ = str(v["hata"])[:60]

            # 2) Ölçüm (ayrı süreç, periyodik)
            if no % self.olcum_arasi == 0 or no == self.tur_sayisi:
                t1 = time.perf_counter()
                r = self._olc()
                k.olcum_s = time.perf_counter() - t1
                if r and not r.get("hata"):
                    k.dogrulama_kaybi = r.get("dogrulama_kaybi")
                    k.acik = float(r.get("acik") or 0.0)
                    k.acik_orani = float(r.get("acik_orani") or 0.0)

            self.kayitlar.append(k)
            self._kaydet()

            ek = ""
            if k.not_:
                ek = f" [{k.not_[:36]}]"
            dg = ("—" if k.dogrulama_kaybi is None
                  else f"{k.dogrulama_kaybi:.4f}")
            print(f"  tur {no:3} | {k.adim:3} adım | "
                  f"eğitim {k.egitim_kaybi if k.egitim_kaybi is None else round(k.egitim_kaybi, 4)} | "
                  f"doğrulama {dg} | açık {k.acik_orani:.2f}× | "
                  f"{k.egitim_s / 60:.1f}+{k.olcum_s / 60:.1f} dk{ek}", flush=True)

        return self.ozet()

    def _kaydet(self) -> None:
        try:
            ILERLEME.write_text(json.dumps({
                "guncelleme": datetime.now().isoformat(timespec="seconds"),
                "kayitlar": [k.to_dict() for k in self.kayitlar],
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass

    def ozet(self) -> Dict[str, Any]:
        """Sonucu özetler."""
        eg = [k.egitim_kaybi for k in self.kayitlar if k.egitim_kaybi]
        dg = [k.dogrulama_kaybi for k in self.kayitlar
              if k.dogrulama_kaybi is not None]
        print()
        print("=" * 78)
        print("  ÖZET")
        print("=" * 78)
        print(f"  tur          : {len(self.kayitlar)}")
        print(f"  toplam adım  : {sum(k.adim for k in self.kayitlar):,}")
        if eg:
            print(f"  eğitim kaybı : {eg[0]:.4f} → {eg[-1]:.4f}")
        if len(dg) >= 2:
            print(f"  doğrulama    : {dg[0]:.4f} → {dg[-1]:.4f}")
            fark = dg[-1] - dg[0]
            print(f"  değişim      : {fark:+.4f} "
                  f"({'İYİLEŞTİ' if fark < 0 else 'KÖTÜLEŞTİ'})")
        elif dg:
            print(f"  doğrulama    : {dg[0]:.4f} (tek ölçüm)")
        return {
            "tur": len(self.kayitlar),
            "toplam_adim": sum(k.adim for k in self.kayitlar),
            "egitim_ilk": eg[0] if eg else None,
            "egitim_son": eg[-1] if eg else None,
            "dogrulama_ilk": dg[0] if dg else None,
            "dogrulama_son": dg[-1] if dg else None,
        }


def main() -> int:
    """Ölçümlü eğitimi başlatır."""
    ap = argparse.ArgumentParser(description="Ölçümlü eğitim")
    ap.add_argument("--tur", type=int, default=12)
    ap.add_argument("--adim", type=int, default=150)
    ap.add_argument("--olcum-arasi", type=int, default=3,
                    help="Kaç turda bir doğrulama ölçülsün")
    ap.add_argument("--dakika", type=float, default=18.0)
    args = ap.parse_args()

    e = OlcumluEgitim(tur=args.tur, adim=args.adim,
                      olcum_arasi=args.olcum_arasi, dakika=args.dakika)
    try:
        sonuc = e.calistir()
    except KeyboardInterrupt:
        print("\n  kesildi")
        sonuc = e.ozet()
    Path("_olcumlu_ozet.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
