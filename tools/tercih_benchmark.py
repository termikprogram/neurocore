"""
Tercih Benchmark'ı — Eğitilen Veriyle Eşleşen Ölçüm
====================================================
⚠ Ölçülen sorun
---------------
`tools/karar_benchmark.py` **teknik bilgi** soruları sorar:
    "Godot'ta karakter hareketi hangi fonksiyonda yapılmalı?"
    "Erosion'da deposit oranı ne olmalı?"

Ama `tools/tercih_verisi_uret.py` **tercih/estetik yargı** öğretti:
    "Fantasy arazi için hangi kaya tonu daha iyi?"
    "Hangi kamera açısı daha uygun?"

**Ölçüm uyuşmazlığı:** Öğretilen beceri ile ölçülen beceri farklı.
Sonuç: 377 tercih örneği eklendi, benchmark **hiç değişmedi** (66,7%).

Bu araç **tercih verisiyle eşleşen** bir benchmark üretir:
    * Kasada `tercih` kategorisindeki notlardan sorular
    * Model `iyi` ve `kotu` seçenekleri ayırt edebiliyor mu?
    * **Ters çevirme testi:** seçenekler yer değiştirilince de doğru mu?

Ters çevirme testi kritiktir: model "iyi" seçeneği **konumdan** değil
**içerikten** tanımalı.

Kullanım
--------
    python tools/tercih_benchmark.py
    python tools/tercih_benchmark.py --adet 40 --ters-test
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)


@dataclass
class Sonuc:
    """Tercih benchmark sonucu."""
    toplam: int = 0
    dogru: int = 0
    dogruluk: float = 0.0
    ters_dogru: int = 0
    ters_toplam: int = 0
    ters_dogruluk: float = 0.0
    ortalama_guven: float = 0.0
    sure_s: float = 0.0
    detaylar: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "toplam": self.toplam, "dogru": self.dogru,
            "dogruluk": round(self.dogruluk, 4),
            "ters_dogru": self.ters_dogru,
            "ters_toplam": self.ters_toplam,
            "ters_dogruluk": round(self.ters_dogruluk, 4),
            "ortalama_guven": round(self.ortalama_guven, 4),
            "sure_s": round(self.sure_s, 1),
            "detaylar": self.detaylar[:30],
        }


class TercihBenchmark:
    """Tercih verisiyle eşleşen benchmark."""

    def __init__(self, *, adet: int = 30, ters_test: bool = True,
                 tohum: int = 7) -> None:
        self.adet = max(5, adet)
        self.ters_test = ters_test
        self.tohum = tohum
        self._model: Any = None
        self._motor: Any = None
        self.sorular: List[Dict[str, str]] = []

    def baglan(self) -> bool:
        """Modeli yükler ve kasayı okur."""
        try:
            from neurocore import NeuroCoreEngine

            self._motor = NeuroCoreEngine()
            self._motor.initialize()
            self._model = self._motor.model
        except Exception as exc:
            print(f"  model yüklenemedi: {exc}")
            return False
        return self._sorulari_yukle()

    def _sorulari_yukle(self) -> bool:
        """
        `tercih` kategorisindeki notlardan sorular üretir.

        Not biçimi:
            **Soru:** ...
            **Daha iyi:** ...
            **Daha kötü:** ...
            **Gerekçe:** ...
        """
        try:
            notlar = list(self._motor.knowledge.vault.iter_notes(limit=200000))
        except Exception as exc:
            print(f"  kasa okunamadı: {exc}")
            return False

        tercih = [n for n in notlar
                  if (getattr(n, "category", "") or "") == "tercih"]
        rng = random.Random(self.tohum)
        rng.shuffle(tercih)

        for n in tercih:
            govde = getattr(n, "body", "") or ""
            soru = self._alan(govde, "Soru")
            iyi = self._alan(govde, "Daha iyi")
            kotu = self._alan(govde, "Daha kötü")
            if soru and iyi and kotu and len(iyi) > 5 and len(kotu) > 5:
                self.sorular.append({"soru": soru, "iyi": iyi, "kotu": kotu})
            if len(self.sorular) >= self.adet:
                break
        print(f"  tercih sorusu: {len(self.sorular)} "
              f"(kasadaki tercih notu: {len(tercih)})")
        return len(self.sorular) >= 5

    @staticmethod
    def _alan(govde: str, ad: str) -> str:
        """Markdown gövdesinden `**Ad:** değer` çıkarır."""
        m = re.search(rf"\*\*{re.escape(ad)}:\*\*\s*(.+)", govde)
        return m.group(1).strip() if m else ""

    def _skorla(self, soru: str, adaylar: Sequence[str]) -> List[float]:
        """
        Adayları puanlar (uzunluk normalize edilmiş).

        ⚠ Kısa cevap önyargısı: uzunluk normalizasyonu olmadan model
        "en kısa aday"ı seçer.
        """
        skorlar: List[float] = []
        for aday in adaylar:
            try:
                ham = self._model.sequence_logprob(f"{soru}\nCevap: {aday}")
                deger = float(ham) if ham is not None else -99.0
            except Exception:
                deger = -99.0
            skorlar.append(deger / max(1, len(aday)) * 20.0)
        return skorlar

    def calistir(self) -> Sonuc:
        """Benchmark'ı çalıştırır."""
        s = Sonuc()
        t0 = time.perf_counter()
        rng = random.Random(self.tohum + 1)
        guvenler: List[float] = []

        for kayit in self.sorular:
            soru, iyi, kotu = kayit["soru"], kayit["iyi"], kayit["kotu"]

            # ── Normal test ──
            adaylar = [iyi, kotu]
            rng.shuffle(adaylar)
            iyi_idx = adaylar.index(iyi)
            skorlar = self._skorla(soru, adaylar)
            if skorlar:
                secilen = max(range(len(skorlar)), key=lambda i: skorlar[i])
                ok = secilen == iyi_idx
                if ok:
                    s.dogru += 1
                s.toplam += 1
                sirali = sorted(skorlar, reverse=True)
                g = ((sirali[0] - sirali[1]) / max(1e-6, abs(sirali[0]))
                     if len(sirali) > 1 else 0.0)
                guvenler.append(max(0.0, min(1.0, g)))
                s.detaylar.append({
                    "soru": soru[:44], "basarili": ok,
                    "guven": round(g, 3), "ters": False,
                })

            # ── Ters çevirme testi ──
            #
            # ⚠ Kritik: model "iyi" seçeneği **konumdan** değil
            # **içerikten** tanımalı. Seçenekler ters çevrilince de
            # doğruyu bulmalı.
            if self.ters_test:
                ters = [kotu, iyi]
                rng.shuffle(ters)
                t_iyi = ters.index(iyi)
                t_skor = self._skorla(soru, ters)
                if t_skor:
                    t_sec = max(range(len(t_skor)), key=lambda i: t_skor[i])
                    t_ok = t_sec == t_iyi
                    if t_ok:
                        s.ters_dogru += 1
                    s.ters_toplam += 1
                    s.detaylar.append({
                        "soru": soru[:44], "basarili": t_ok,
                        "guven": 0.0, "ters": True,
                    })

        s.sure_s = time.perf_counter() - t0
        s.dogruluk = s.dogru / max(1, s.toplam)
        s.ters_dogruluk = s.ters_dogru / max(1, s.ters_toplam)
        s.ortalama_guven = (sum(guvenler) / len(guvenler)
                            if guvenler else 0.0)
        return s


def main() -> int:
    """Benchmark'ı çalıştırır."""
    ap = argparse.ArgumentParser(description="Tercih benchmark'ı")
    ap.add_argument("--adet", type=int, default=30, help="Soru sayısı")
    ap.add_argument("--ters-test", action="store_true",
                    help="Seçenekleri ters çevirip de dene")
    args = ap.parse_args()

    print("=" * 78)
    print("  TERCİH BENCHMARK'I — eğitilen veriyle eşleşen ölçüm")
    print("=" * 78)

    b = TercihBenchmark(adet=args.adet, ters_test=args.ters_test)
    if not b.baglan():
        return 1
    print()
    s = b.calistir()

    print(f"  DOĞRULUK       : {s.dogruluk * 100:.1f}%  ({s.dogru}/{s.toplam})")
    print(f"  ortalama güven : {s.ortalama_guven:.3f}")
    if s.ters_toplam:
        print(f"  TERS ÇEVİRME   : {s.ters_dogruluk * 100:.1f}%  "
              f"({s.ters_dogru}/{s.ters_toplam})")
    print(f"  süre           : {s.sure_s:.1f} sn")
    print()
    print("  DETAY (ilk 10):")
    for d in s.detaylar[:10]:
        im = "+" if d["basarili"] else " "
        tur = "ters" if d.get("ters") else "düz "
        print(f"  {im} [{tur}] {d['soru'][:50]}")

    # Yorum
    print()
    rast = 50.0
    if s.dogruluk * 100 >= 75:
        print(f"  ✓ Model tercih ayırt edebiliyor "
              f"(%{s.dogruluk * 100:.0f} > rastgele %{rast:.0f})")
    elif s.dogruluk * 100 >= 60:
        print(f"  ~ Model kısmen ayırt ediyor (%{s.dogruluk * 100:.0f})")
    else:
        print(f"  ✗ Model tercih ayırt edemiyor "
              f"(%{s.dogruluk * 100:.0f} ≈ rastgele %{rast:.0f})")

    if s.ters_toplam and abs(s.ters_dogruluk - s.dogruluk) > 0.25:
        print("  ⚠ Ters çevirme ile fark büyük — model **konuma** bakıyor "
              "olabilir")

    Path("_tercih_benchmark.json").write_text(
        json.dumps(s.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"  rapor: _tercih_benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
