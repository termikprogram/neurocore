"""
Karar Kalitesi Benchmark'ı — myAi'nin Ölçülmüş Gücü
=====================================================
⚠ Ölçülen gerçek
----------------
myAi **kod üretemiyor** (ölçüldü: 1/10). Ama **seçim yapabiliyor**
(ölçüldü: 5/5, sıra önyargısı kontrolünde 8/8).

Üretim döngüsü bu güce dayanıyor: myAi hangi paleti, hangi seviye
türünü, hangi tuş düzenini seçeceğine karar veriyor.

Bu araç o kararı **ölçer**:
    * Doğruluk oranı (doğru aday seçildi mi)
    * **Sıra önyargısı** (hep aynı konumu mu seçiyor — sahte başarı)
    * Güven dağılımı (skorlar ne kadar ayrışıyor)

Sıra önyargısı testi kritiktir: model doğru cevabı **her zaman ilk
sıraya** koyarsak %100 "başarılı" görünür. Gerçek öğrenme için adaylar
karıştırılır ve model yine doğrusunu bulmalıdır.

Kullanım
--------
    python tools/karar_benchmark.py
    python tools/karar_benchmark.py --sira-testi
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)


#: Soru bankası: (soru, doğru cevap, yanlış cevaplar)
SORULAR: Sequence[Tuple[str, str, Sequence[str]]] = (
    ("Fantasy terrain icin hangi renk paleti daha uygun?",
     "doygun yesil cimen, sicak bej kum, mor-gri kaya, parlak kar",
     ["gri tonlamali gercekci renkler, dusuk doygunluk",
      "neon pembe ve siyah, yuksek kontrast",
      "tek renk duz yesil, golge yok"]),
    ("Lowpoly bir agac modelinde govde kac kenarli olmali?",
     "6 kenarli silindir, dusuk poligon",
     ["64 kenarli silindir, puruzsuz yuzey",
      "12 kenarli silindir, orta detay",
      "kure govde, silindir yok"]),
    ("Godot'ta karakter hareketi hangi fonksiyonda yapilmali?",
     "_physics_process icinde move_and_slide ile",
     ["_process icinde position degistirerek",
      "_ready icinde bir kez hesaplayarak",
      "_input icinde dogrudan konum atayarak"]),
    ("Arazi dokularinda moire (dalga deseni) nasil onlenir?",
     "UV olcegini dusurup Linear filtreleme kullanarak",
     ["UV olcegini artirip Closest filtreleme kullanarak",
      "doku cozunurlugunu 16x16 yaparak",
      "dokuyu tamamen kaldirarak"]),
    ("Hydraulic erosion'da deposit (biriktirme) orani ne olmali?",
     "dusuk, yaklasik 0.03 ve tortu ile sinirli",
     ["yuksek, yaklasik 0.5 ve sinirsiz",
      "sifir, hic biriktirme yapilmaz",
      "1.0, tum tortu biriktirilir"]),
    ("Godot'ta master server'in gorevi nedir?",
     "kimlik dogrulama, eslestirme ve bilet uretme",
     ["oyun fizigini hesaplamak",
      "istemci grafiklerini render etmek",
      "yerel kayit dosyalarini yazmak"]),
    ("Bir mesh'te karsilikli yuzlerin gorunmesi icin ne gerekir?",
     "CULL_DISABLED veya cift tarafli malzeme",
     ["daha fazla vertex eklemek",
      "normalleri ters cevirmek",
      "dokuyu saydam yapmak"]),
    ("Labirent uretiminde geri izleme algoritmasi ne uretir?",
     "tek cozumlu, dallanan labirent",
     ["cok cozumlu, dongusel labirent",
      "cozumsuz, kapali labirent",
      "duz koridor, dallanma yok"]),
    ("Karakter iskeletinde kemik sayisi lowpoly icin kac olmali?",
     "15-25 kemik yeterli",
     ["100+ kemik gerekli",
      "2-3 kemik yeterli",
      "kemik kullanilmamali"]),
    ("Skeleton3D'de mesh nasil kemige baglanir?",
     "BoneAttachment3D ile bone_idx atanarak",
     ["mesh dogrudan kemigin cocugu yapilarak",
      "mesh ile kemik ayni isimlendirilerek",
      "baglanmaz, ayri hareket eder"]),
    ("Godot'ta hile onleme icin istemci ne gondermeli?",
     "yalniz girdi; sonucu sunucu hesaplamali",
     ["hesaplanmis konum ve hasar miktari",
      "kendi saglik degeri",
      "sunucu ayarlari"]),
    ("Vertex color ile katman karistirmak neden verimli?",
     "doku ve ek draw call gerektirmez",
     ["daha yuksek cozunurluk verir",
      "isik hesaplamasini hizlandirir",
      "otomatik UV uretir"]),
    ("Godot'ta bir dugumun cocuklarina nasil erisilir?",
     "get_node veya $Yol/Dugum ile",
     ["get_child_by_name ile",
      "find_node_recursive ile",
      "dugumler dogrudan erisilemez"]),
    ("Arazi yukseklik haritasi uretirken hangi gurultu kullanilir?",
     "Perlin veya Simplex fraktal gurultu",
     ["tamamen rastgele degerler",
      "tek renkli duz deger",
      "sinus dalgasi"]),
    ("Godot'ta 10000 cimi nasil tek draw call'da gosteririz?",
     "MultiMeshInstance3D kullanarak",
     ["10000 ayri MeshInstance3D ekleyerek",
      "tek buyuk kutu kullanarak",
      "dokuyu tekrarlayarak"]),
)


@dataclass
class Sonuc:
    """Benchmark sonucu."""
    toplam: int = 0
    dogru: int = 0
    dogruluk: float = 0.0
    ortalama_guven: float = 0.0
    sira_dogruluk: float = 0.0
    sira_testi: bool = False
    sure_s: float = 0.0
    hatalar: List[str] = field(default_factory=list)
    detaylar: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "toplam": self.toplam, "dogru": self.dogru,
            "dogruluk": round(self.dogruluk, 4),
            "ortalama_guven": round(self.ortalama_guven, 4),
            "sira_dogruluk": round(self.sira_dogruluk, 4),
            "sira_testi": self.sira_testi,
            "sure_s": round(self.sure_s, 1),
            "hatalar": self.hatalar[:5], "detaylar": self.detaylar,
        }


class KararBenchmark:
    """myAi'nin seçim kalitesini ölçer."""

    def __init__(self, *, sira_testi: bool = False,
                 tohum: int = 42) -> None:
        self.sira_testi = sira_testi
        self.tohum = tohum
        self._model: Any = None

    def baglan(self) -> bool:
        """Yerel modeli yükler."""
        try:
            from neurocore import NeuroCoreEngine

            motor = NeuroCoreEngine()
            motor.initialize()
            self._model = motor.model
            return True
        except Exception as exc:
            print(f"  model yüklenemedi: {exc}")
            return False

    def _skorla(self, soru: str, adaylar: Sequence[str]) -> List[float]:
        """
        Adayları puanlar (uzunluk normalize edilmiş).

        ⚠ Ölçülen önyargı: model **kısa** cevabı yüksek puanlıyor
        (uzunluk normalizasyonu olmadan karar "en kısa aday"a düşer).
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
        s = Sonuc(sira_testi=self.sira_testi)
        t0 = time.perf_counter()
        rng = random.Random(self.tohum)
        guvenler: List[float] = []
        sira_dogru = 0

        for soru, dogru, yanlislar in SORULAR:
            adaylar = [dogru] + list(yanlislar)
            if self.sira_testi:
                rng.shuffle(adaylar)
            dogru_idx = adaylar.index(dogru)

            skorlar = self._skorla(soru, adaylar)
            if not skorlar:
                s.hatalar.append(soru[:40])
                continue

            secilen = max(range(len(skorlar)), key=lambda i: skorlar[i])
            ok = secilen == dogru_idx
            if ok:
                s.dogru += 1
                sira_dogru += 1
            s.toplam += 1

            # Güven: en iyi ile ikinci arasındaki bağıl fark
            sirali = sorted(skorlar, reverse=True)
            guven = ((sirali[0] - sirali[1]) / max(1e-6, abs(sirali[0]))
                     if len(sirali) > 1 else 0.0)
            guvenler.append(max(0.0, min(1.0, guven)))

            s.detaylar.append({
                "soru": soru[:44],
                "dogru_konum": dogru_idx,
                "secilen_konum": secilen,
                "basarili": ok,
                "guven": round(guven, 3),
            })

        s.sure_s = time.perf_counter() - t0
        s.dogruluk = s.dogru / max(1, s.toplam)
        s.ortalama_guven = (sum(guvenler) / len(guvenler)
                            if guvenler else 0.0)
        s.sira_dogruluk = sira_dogru / max(1, s.toplam)
        return s


def main() -> int:
    """Benchmark'ı çalıştırır."""
    ap = argparse.ArgumentParser(description="Karar kalitesi benchmark'ı")
    ap.add_argument("--sira-testi", action="store_true",
                    help="Adayları karıştır (sıra önyargısını ele verir)")
    args = ap.parse_args()

    print("=" * 78)
    print("  KARAR KALİTESİ BENCHMARK'I — myAi'nin ölçülmüş gücü")
    print("=" * 78)

    b = KararBenchmark(sira_testi=args.sira_testi)
    if not b.baglan():
        return 1
    print(f"  soru sayısı   : {len(SORULAR)}")
    print(f"  sıra testi    : {'AÇIK (adaylar karışık)' if args.sira_testi else 'kapalı'}")
    print()
    s = b.calistir()

    print(f"  DOĞRULUK      : {s.dogruluk * 100:.1f}%  "
          f"({s.dogru}/{s.toplam})")
    print(f"  ortalama güven: {s.ortalama_guven:.3f}")
    print(f"  süre          : {s.sure_s:.1f} sn")
    print()
    print("  DETAY (ilk 8):")
    for d in s.detaylar[:8]:
        im = "+" if d["basarili"] else " "
        print(f"  {im} [{d['guven']:.2f}] doğru@{d['dogru_konum']} "
              f"seçilen@{d['secilen_konum']} — {d['soru'][:46]}")

    # Yorum
    print()
    if s.dogruluk >= 0.75:
        print("  ✓ Model seçim yapabiliyor (üretim döngüsünde kullanılabilir)")
    elif s.dogruluk >= 0.5:
        print("  ~ Model kısmen seçebiliyor (güven düşük)")
    else:
        print("  ✗ Model seçim yapamıyor — karar katmanı güvenilmez")

    if s.ortalama_guven < 0.05:
        print("  ⚠ Güven çok düşük: skorlar neredeyse eşit — seçim "
              "neredeyse rastgele")

    Path("_karar_benchmark.json").write_text(
        json.dumps(s.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"  rapor: _karar_benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
