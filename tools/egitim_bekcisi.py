"""
Sürekli Eğitim Bekçisi — Kesintisiz Öğrenme
============================================
Modeli **sürekli** eğitir: bir koşucu biterse yenisini başlatır.

Neden ayrı bekçi?
    `surekli_egitim.py` bir hedef süre kadar çalışır ve **biter**.
    Kullanıcı "sürekli eğitmeyi unutma" dedi — bu yüzden:

        * Koşucu bitince **yenisi başlar**
        * Bilgisayar yeniden başlarsa **kaldığı yerden devam eder**
        * Her koşucu turu biter, ilerleme kaydedilir
        * Toplam geçmiş tek dosyada birikir (`_egitim_gecmisi.json`)

Kullanım
--------
    # Kesintisiz (bilgisayar kapanana kadar)
    python tools/egitim_bekcisi.py --devamli

    # Toplam 8 saat
    python tools/egitim_bekcisi.py --toplam-saat 8

    # Tek parti (varsayılan: 2 saat)
    python tools/egitim_bekcisi.py
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

#: Toplam geçmiş dosyası (partiler arası birikir)
GECMIS = Path(".neurocore/egitim_gecmisi.json")

#: Bekçi durumu (kesintiden sürdürmek için)
DURUM = Path(".neurocore/egitim_bekcisi.json")


@dataclass
class Parti:
    """Bir eğitim partisinin kaydı."""
    no: int = 0
    baslangic: str = ""
    sure_dk: float = 0.0
    tur: int = 0
    toplam_adim: int = 0
    iyilesen: int = 0
    geri_alinan: int = 0
    ilk_kayip: Optional[float] = None
    son_kayip: Optional[float] = None
    hata: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "no": self.no, "baslangic": self.baslangic,
            "sure_dk": round(self.sure_dk, 1), "tur": self.tur,
            "toplam_adim": self.toplam_adim, "iyilesen": self.iyilesen,
            "geri_alinan": self.geri_alinan, "ilk_kayip": self.ilk_kayip,
            "son_kayip": self.son_kayip, "hata": self.hata[:120],
        }


class EgitimBekcisi:
    """Eğitimi kesintisiz sürdürür."""

    def __init__(self, *, parti_saat: float = 2.0,
                 toplam_saat: Optional[float] = None,
                 adim: int = 200, dakika: float = 18.0) -> None:
        self.parti_s = max(300.0, parti_saat * 3600.0)
        self.toplam_s = toplam_saat * 3600.0 if toplam_saat else None
        self.adim = adim
        self.dakika = dakika
        self.partiler: List[Parti] = []
        self.baslangic = time.time()
        self._betik = _KOK / "tools" / "surekli_egitim.py"
        self._gecmis = self._yukle()

    # ─── Geçmiş ───

    def _yukle(self) -> Dict[str, Any]:
        """Önceki eğitim geçmişini yükler."""
        if GECMIS.exists():
            try:
                return json.loads(GECMIS.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"partiler": [], "toplam_adim": 0, "baslangic": ""}

    def _kaydet(self, parti: Parti) -> None:
        """Partiyi geçmişe ekler."""
        try:
            g = self._gecmis
            g.setdefault("partiler", []).append(parti.to_dict())
            g["toplam_adim"] = int(g.get("toplam_adim", 0)) + parti.toplam_adim
            g["son_guncelleme"] = datetime.now().isoformat(timespec="seconds")
            if not g.get("baslangic"):
                g["baslangic"] = datetime.now().isoformat(timespec="seconds")
            GECMIS.write_text(json.dumps(g, ensure_ascii=False, indent=1),
                              encoding="utf-8")
            self._gecmis = g
        except Exception:
            pass
        try:
            DURUM.write_text(json.dumps({
                "son_parti": parti.no,
                "gecen_dk": round((time.time() - self.baslangic) / 60, 1),
                "partiler": len(self.partiler),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass

    def gecen_s(self) -> float:
        return time.time() - self.baslangic

    def devam_etmeli(self) -> bool:
        """Bir parti daha çalışmalı mı?"""
        if self.toplam_s is not None and self.gecen_s() >= self.toplam_s:
            return False
        return True

    # ─── Parti ───

    def parti_calistir(self, no: int) -> Parti:
        """Bir eğitim partisini çalıştırır (alt süreç)."""
        p = Parti(no=no, baslangic=datetime.now().isoformat(timespec="seconds"))
        kalan = self.parti_s
        if self.toplam_s is not None:
            kalan = min(kalan, max(0.0, self.toplam_s - self.gecen_s()))
        saat = kalan / 3600.0
        t0 = time.perf_counter()

        komut = [sys.executable, "-u", "-X", "utf8", str(self._betik),
                 "--saat", f"{saat:.3f}", "--adim", str(self.adim),
                 "--dakika", str(self.dakika)]
        try:
            s = subprocess.run(komut, capture_output=True, text=True,
                               timeout=kalan + 600, cwd=str(_KOK),
                               encoding="utf-8", errors="replace")
            cikti = (s.stdout or "") + (s.stderr or "")
        except subprocess.TimeoutExpired:
            p.hata = "parti zaman aşımı"
            p.sure_dk = (time.perf_counter() - t0) / 60
            return p
        except Exception as exc:
            p.hata = f"{type(exc).__name__}: {exc}"
            p.sure_dk = (time.perf_counter() - t0) / 60
            return p

        p.sure_dk = (time.perf_counter() - t0) / 60
        # Özet dosyasından sonucu oku
        ozet = _KOK / "_surekli_egitim_ozet.json"
        if ozet.exists():
            try:
                v = json.loads(ozet.read_text(encoding="utf-8"))
                p.tur = int(v.get("tur") or 0)
                p.toplam_adim = int(v.get("toplam_adim") or 0)
                p.iyilesen = int(v.get("iyilesen") or 0)
                p.geri_alinan = int(v.get("geri_alinan") or 0)
                p.ilk_kayip = v.get("ilk_kayip")
                p.son_kayip = v.get("son_kayip")
            except Exception as exc:
                p.hata = f"özet okunamadı: {str(exc)[:60]}"
        else:
            p.hata = "özet dosyası yok"
        return p

    # ─── Ana döngü ───

    def calistir(self) -> Dict[str, Any]:
        """Partiler hâlinde kesintisiz eğitir."""
        g = self._gecmis
        print("=" * 78)
        print("  EĞİTİM BEKÇİSİ — kesintisiz öğrenme")
        print("=" * 78)
        onceki = len(g.get("partiler") or [])
        if onceki:
            print(f"  önceki geçmiş: {onceki} parti, "
                  f"{g.get('toplam_adim', 0):,} adım")
        print(f"  parti süresi : {self.parti_s / 3600:.1f} saat")
        print(f"  toplam hedef : "
              f"{'sınırsız' if self.toplam_s is None else f'{self.toplam_s / 3600:.1f} saat'}")
        print()

        no = onceki
        while self.devam_etmeli():
            no += 1
            print(f"  ── parti {no} başlıyor "
                  f"({datetime.now().strftime('%H:%M:%S')}) ──", flush=True)
            p = self.parti_calistir(no)
            self.partiler.append(p)
            self._kaydet(p)

            print(f"  parti {no}: {p.tur} tur | {p.toplam_adim:,} adım | "
                  f"{p.sure_dk:.1f} dk | iyileşen {p.iyilesen} | "
                  f"geri {p.geri_alinan} | kayıp {p.ilk_kayip} → {p.son_kayip}"
                  f"{' | ' + p.hata[:50] if p.hata else ''}", flush=True)

            # Ardışık boş partiler → bekle
            son = self.partiler[-3:]
            if len(son) == 3 and all(x.toplam_adim == 0 for x in son):
                print("  3 parti üst üste 0 adım — 10 dk bekleniyor", flush=True)
                time.sleep(600)

        return self.ozet()

    def ozet(self) -> Dict[str, Any]:
        """Toplam sonucu özetler."""
        adim = sum(p.toplam_adim for p in self.partiler)
        tur = sum(p.tur for p in self.partiler)
        iyi = sum(p.iyilesen for p in self.partiler)
        geri = sum(p.geri_alinan for p in self.partiler)
        son = [p.son_kayip for p in self.partiler if p.son_kayip]

        print()
        print("=" * 78)
        print("  BEKÇİ ÖZETİ")
        print("=" * 78)
        print(f"  bu oturum : {self.gecen_s() / 3600:.2f} saat, "
              f"{len(self.partiler)} parti")
        print(f"  adım      : {adim:,} ({tur} tur)")
        print(f"  iyileşen  : {iyi} | geri alınan: {geri}")
        if son:
            print(f"  son kayıp : {son[-1]:.4f}")
        g = self._gecmis
        print(f"  TÜM GEÇMİŞ: {len(g.get('partiler') or [])} parti, "
              f"{g.get('toplam_adim', 0):,} adım")
        print("=" * 78)

        # Model durumu
        try:
            gr = json.loads((_KOK / "models/neurocore/growth.json")
                            .read_text(encoding="utf-8"))
            print(f"  MODEL: {gr.get('total_training_steps'):,} adım | "
                  f"{gr.get('total_tokens_seen'):,} token")
        except Exception:
            pass
        return {"parti": len(self.partiler), "adim": adim,
                "iyilesen": iyi, "geri_alinan": geri}


def main() -> int:
    """Bekçiyi başlatır."""
    ap = argparse.ArgumentParser(description="Sürekli eğitim bekçisi")
    ap.add_argument("--parti-saat", type=float, default=2.0,
                    help="Parti başına saat")
    ap.add_argument("--toplam-saat", type=float, default=8.0,
                    help="Toplam hedef saat (0 = sınırsız)")
    ap.add_argument("--devamli", action="store_true",
                    help="Sınırsız çalış (bilgisayar kapanana kadar)")
    ap.add_argument("--adim", type=int, default=200, help="Tur başına adım")
    ap.add_argument("--dakika", type=float, default=18.0,
                    help="Tur başına en fazla dakika")
    args = ap.parse_args()

    toplam = None if args.devamli else (
        args.toplam_saat if args.toplam_saat > 0 else None)
    b = EgitimBekcisi(parti_saat=args.parti_saat, toplam_saat=toplam,
                      adim=args.adim, dakika=args.dakika)
    try:
        sonuc = b.calistir()
    except KeyboardInterrupt:
        print("\n  kesildi — özet:")
        sonuc = b.ozet()
    (_KOK / "_bekci_ozet.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
