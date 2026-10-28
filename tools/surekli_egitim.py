"""
Sürekli Eğitim Koşucusu
========================
Modeli **kesintisiz** eğitir — tur tur, hata olsa bile devam eder.

Neden ayrı koşucu?
    `taze_egitim.py` tek bir parti çalıştırır ve biter. Uzun eğitim
    (saatler) için:

        * Bir tur çökerse **durmamalı**, sonrakine geçmeli
        * Uygulama açılırsa (VRAM kapılırsa) **bekleyip** devam etmeli
        * İlerleme **kaydedilmeli** ki kesintiden sonra sürdürülebilsin
        * Toplam süre **hedeflenmeli** (ör. 8 saat)

Bu koşucu bunların hepsini yapar ve sonunda **özet rapor** verir.

Kullanım
--------
    python tools/surekli_egitim.py --saat 8 --adim 200
    python tools/surekli_egitim.py --saat 2 --adim 150 --kod-agirlikli
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: İlerleme dosyası (kesintiden sonra sürdürmek için)
ILERLEME = Path(".neurocore/surekli_egitim.json")

#: Tur çıktısını yakalayan işaret
ISARET = "NC_RESULT "


@dataclass
class Tur:
    """Tek eğitim turunun kaydı."""
    no: int = 0
    adim: int = 0
    istenen: int = 0
    kayip_ilk: Optional[float] = None
    kayip_son: Optional[float] = None
    token: Optional[int] = None
    sure_s: float = 0.0
    geri_alindi: bool = False
    hata: str = ""

    def iyilesti(self) -> bool:
        """Kayıp düştü mü?"""
        if self.kayip_ilk is None or self.kayip_son is None:
            return False
        return float(self.kayip_son) < float(self.kayip_ilk)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "no": self.no, "adim": self.adim, "istenen": self.istenen,
            "kayip_ilk": self.kayip_ilk, "kayip_son": self.kayip_son,
            "token": self.token, "sure_s": round(self.sure_s, 1),
            "geri_alindi": self.geri_alindi, "hata": self.hata[:120],
        }


class SurekliEgitim:
    """Kesintisiz eğitim koşucusu."""

    def __init__(self, *, saat: float = 8.0, adim: int = 200,
                 dakika: float = 25.0, bekleme: float = 120.0) -> None:
        self.hedef_s = max(60.0, saat * 3600.0)
        self.adim = max(10, adim)
        self.dakika = dakika
        self.bekleme = bekleme
        self.turlar: List[Tur] = []
        self.baslangic = time.time()
        self._betik = _KOK / "tools" / "taze_egitim.py"

    # ─── Yardımcılar ───

    def gecen_s(self) -> float:
        return time.time() - self.baslangic

    def kalan_s(self) -> float:
        return max(0.0, self.hedef_s - self.gecen_s())

    @staticmethod
    def _vram_bos() -> bool:
        """VRAM eğitime yetiyor mu? (kendi modelimiz ~1,8 GB ister)"""
        try:
            import torch

            if not torch.cuda.is_available():
                return True
            free, total = torch.cuda.mem_get_info()
            return (free / 1024**2) > 2600.0
        except Exception:
            return True

    def _tur_calistir(self, no: int) -> Tur:
        """Bir eğitim turunu **ayrı süreçte** çalıştırır."""
        t = Tur(no=no, istenen=self.adim)
        t0 = time.perf_counter()
        komut = [sys.executable, "-u", "-X", "utf8", str(self._betik),
                 "--tek-tur", "--adim", str(self.adim),
                 "--dakika", str(self.dakika)]
        try:
            s = subprocess.run(komut, capture_output=True, text=True,
                               timeout=self.dakika * 60 + 900,
                               cwd=str(_KOK), encoding="utf-8",
                               errors="replace")
            cikti = (s.stdout or "") + (s.stderr or "")
        except subprocess.TimeoutExpired:
            t.hata = "tur zaman aşımı"
            t.sure_s = time.perf_counter() - t0
            return t
        except Exception as exc:
            t.hata = f"{type(exc).__name__}: {exc}"
            t.sure_s = time.perf_counter() - t0
            return t

        t.sure_s = time.perf_counter() - t0
        for satir in cikti.splitlines():
            if satir.startswith(ISARET):
                try:
                    v = json.loads(satir[len(ISARET):])
                except Exception:
                    continue
                t.adim = int(v.get("steps_completed") or 0)
                t.kayip_ilk = v.get("initial_loss")
                t.kayip_son = v.get("final_loss")
                t.token = v.get("tokens_seen")
                t.geri_alindi = bool(v.get("rolled_back"))
                if not t.adim:
                    t.hata = str(v.get("stop_reason") or "0 adım")
                break
        else:
            t.hata = "sonuç satırı yok"
        return t

    def kaydet(self) -> None:
        """İlerlemeyi diske yazar."""
        try:
            ILERLEME.write_text(json.dumps({
                "hedef_saat": round(self.hedef_s / 3600.0, 2),
                "gecen_s": round(self.gecen_s(), 1),
                "turlar": [t.to_dict() for t in self.turlar],
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass

    # ─── Ana döngü ───

    def calistir(self) -> Dict[str, Any]:
        """Hedef süre boyunca tur tur eğitir."""
        print("=" * 78)
        print(f"  SÜREKLİ EĞİTİM — hedef {self.hedef_s / 3600:.1f} saat, "
              f"tur başına {self.adim} adım")
        print("=" * 78)
        no = 0
        bos_bekleme = 0
        while self.kalan_s() > 60.0:
            # VRAM kapısı: uygulama açıksa bekle
            if not self._vram_bos():
                bos_bekleme += 1
                print(f"  VRAM yetersiz — {self.bekleme:.0f} sn bekleniyor "
                      f"({bos_bekleme}. kez)", flush=True)
                time.sleep(self.bekleme)
                continue

            no += 1
            t = self._tur_calistir(no)
            self.turlar.append(t)
            self.kaydet()

            durum = "GERİ ALINDI" if t.geri_alindi else (
                "iyileşti" if t.iyilesti() else "ıraksadı")
            kayip = (f"{t.kayip_ilk} → {t.kayip_son}"
                     if t.kayip_ilk is not None else "—")
            print(f"  tur {no:3} | {t.adim:3}/{self.adim} adım | "
                  f"{t.sure_s / 60:5.1f} dk | kayıp {kayip} | {durum}"
                  f"{' | ' + t.hata[:40] if t.hata else ''}", flush=True)

            # Üst üste hata varsa uzun bekle
            son = self.turlar[-5:]
            if len(son) == 5 and all(x.adim == 0 for x in son):
                print("  5 tur üst üste başarısız — 10 dk bekleniyor",
                      flush=True)
                time.sleep(600)

        return self.ozet()

    def ozet(self) -> Dict[str, Any]:
        """Toplam sonucu özetler."""
        toplam_adim = sum(t.adim for t in self.turlar)
        iyi = sum(1 for t in self.turlar if t.iyilesti())
        geri = sum(1 for t in self.turlar if t.geri_alindi)
        hata = sum(1 for t in self.turlar if t.adim == 0)
        son = [t.kayip_son for t in self.turlar
               if t.kayip_son is not None and not t.geri_alindi]
        ilk = [t.kayip_ilk for t in self.turlar if t.kayip_ilk is not None]

        print()
        print("=" * 78)
        print("  ÖZET")
        print("=" * 78)
        print(f"  süre          : {self.gecen_s() / 3600:.2f} saat")
        print(f"  tur           : {len(self.turlar)}")
        print(f"  toplam adım   : {toplam_adim:,}")
        print(f"  iyileşen tur  : {iyi}")
        print(f"  geri alınan   : {geri}")
        print(f"  başarısız tur : {hata}")
        if ilk:
            print(f"  ilk kayıp     : {ilk[0]:.4f}")
        if son:
            print(f"  son kayıp     : {son[-1]:.4f}")
        print("=" * 78)
        return {
            "sure_s": round(self.gecen_s(), 1),
            "tur": len(self.turlar), "toplam_adim": toplam_adim,
            "iyilesen": iyi, "geri_alinan": geri, "basarisiz": hata,
            "ilk_kayip": ilk[0] if ilk else None,
            "son_kayip": son[-1] if son else None,
            "turlar": [t.to_dict() for t in self.turlar],
        }


def main() -> int:
    """Sürekli eğitimi başlatır."""
    ap = argparse.ArgumentParser(description="Sürekli eğitim koşucusu")
    ap.add_argument("--saat", type=float, default=8.0, help="Hedef süre")
    ap.add_argument("--adim", type=int, default=200, help="Tur başına adım")
    ap.add_argument("--dakika", type=float, default=25.0,
                    help="Tur başına en fazla dakika")
    ap.add_argument("--bekleme", type=float, default=120.0,
                    help="VRAM beklerken bekleme (sn)")
    args = ap.parse_args()

    k = SurekliEgitim(saat=args.saat, adim=args.adim,
                      dakika=args.dakika, bekleme=args.bekleme)
    try:
        sonuc = k.calistir()
    except KeyboardInterrupt:
        print("\n  kesildi — özet:")
        sonuc = k.ozet()
    Path("_surekli_egitim_ozet.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  rapor: _surekli_egitim_ozet.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
