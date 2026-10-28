"""
Yerel Model Karar Motoru — Fantasy Terrain Texture
===================================================
İş bölümü
---------
Ölçüm gösterdi ki yerel model (myAi) **iki işten birini** yapabiliyor:

    ✅ Seçim / sınıflandırma  → verilen adaylar arasından doğrusunu seçer
                                (ölçüldü: 5/5, sıra önyargısı yok)
    ❌ Serbest üretim         → Blender betiği yazamaz
                                (ölçüldü: iki denemede de geçersiz Python)

Bu yüzden doğru mimari **iş bölümüdür**:

    Yerel model  →  **karar verir**  (hangi palet, hangi katman oranı,
                                     hangi doku karakteri, hangi eşikler)
    Bulut model  →  **kodu yazar**   (Blender Python betiği)
    NeuroCore    →  **yürütür ve doğrular**

Bu modül **karar katmanıdır**: yerel modelden yapılandırılmış seçimler alır
ve bunları Blender betiğinin parametrelerine çevirir.

Kullanım
--------
    from tools.fantasy_karar import FantasyKarar

    karar = FantasyKarar()
    secim = karar.palet_sec()          # yerel model seçer
    print(secim)                        # {'secim': ..., 'guven': ...}
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Betik doğrudan çalıştırıldığında proje kökünü yola ekle
# (aksi hâlde `import neurocore` başarısız olur)
_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logger = logging.getLogger("fantasy_karar")


# ─── Fantasy Paletleri ──────────────────────────────────────────────────────

#: Her palet: katman → (R, G, B). Fantasy türü **doygun ve sıcak** olur.
PALETLER: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "buyulu_orman": {
        "kum": (0.86, 0.78, 0.55),      # sıcak altın kum
        "cimen": (0.24, 0.62, 0.28),    # doygun zümrüt çimen
        "kaya": (0.48, 0.42, 0.58),     # mor-gri büyülü kaya
        "kar": (0.98, 0.99, 1.00),      # parlak kar
    },
    "ejderha_diyari": {
        "kum": (0.72, 0.50, 0.32),      # kızıl toprak
        "cimen": (0.38, 0.52, 0.20),    # zeytin yeşili
        "kaya": (0.36, 0.30, 0.34),     # koyu volkanik
        "kar": (0.94, 0.88, 0.82),      # kül karışık kar
    },
    "peri_ulkesi": {
        "kum": (0.94, 0.86, 0.80),      # pembe-bej
        "cimen": (0.52, 0.82, 0.44),    # açık nane yeşili
        "kaya": (0.70, 0.62, 0.82),     # lavanta kaya
        "kar": (1.00, 0.98, 1.00),
    },
    "kuzey_efsanesi": {
        "kum": (0.68, 0.70, 0.74),      # buzul çakıl
        "cimen": (0.30, 0.48, 0.38),    # soğuk çam yeşili
        "kaya": (0.42, 0.48, 0.56),     # mavi-gri kaya
        "kar": (0.96, 0.98, 1.00),
    },
    "col_batakligi": {
        "kum": (0.56, 0.48, 0.34),      # çamur
        "cimen": (0.34, 0.44, 0.22),    # bataklık yeşili
        "kaya": (0.40, 0.36, 0.32),
        "kar": (0.88, 0.90, 0.86),
    },
}

#: Doku karakterleri
DOKU_KARAKTERI: Dict[str, Dict[str, float]] = {
    "puruzlu":   {"gurultu": 0.18, "cizgi": 0.0,  "benek": 0.00},
    "benekli":   {"gurultu": 0.10, "cizgi": 0.0,  "benek": 0.14},
    "katmanli":  {"gurultu": 0.08, "cizgi": 0.45, "benek": 0.00},
    "kristalli": {"gurultu": 0.06, "cizgi": 0.20, "benek": 0.22},
    "yumusak":   {"gurultu": 0.06, "cizgi": 0.0,  "benek": 0.00},
}

#: Kar sınırı profilleri (yükseklik eşiği)
KAR_PROFILI: Dict[str, Tuple[float, float]] = {
    "alcak":  (0.52, 0.66),     # kar erken başlar (karlı dünya)
    "orta":   (0.70, 0.84),     # dengeli
    "yuksek": (0.84, 0.94),     # kar yalnız zirvede
}


@dataclass
class Secim:
    """Yerel modelin bir kararı."""
    soru: str = ""
    secim: str = ""
    guven: float = 0.0          # 0..1 arası skor farkı
    adaylar: List[str] = field(default_factory=list)
    skorlar: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "soru": self.soru, "secim": self.secim,
            "guven": round(self.guven, 4), "adaylar": self.adaylar,
            "skorlar": [round(s, 4) for s in self.skorlar],
        }


class FantasyKarar:
    """
    Yerel modelden **yapılandırılmış kararlar** alır.

    Yöntem: her soru için adaylar verilir, yerel model her adaya bir
    log-olasılık skoru verir, **en yüksek** seçilir. Skor **uzunluğa göre
    normalize** edilir — ölçüldü: model uzun cevapları sistematik olarak
    düşük puanlıyor, normalizasyon olmadan karar yanıltıcı olur.

    Güven = (en iyi − ikinci) / |en iyi| — kararın ne kadar net olduğu.
    """

    def __init__(self, model: Any = None) -> None:
        self._model = model

    # ─── Model ───

    def baglan(self) -> Any:
        """Yerel modeli yükler (yoksa None)."""
        if self._model is not None:
            return self._model
        try:
            from neurocore import NeuroCoreEngine

            motor = NeuroCoreEngine()
            motor.initialize()
            self._model = motor.model
            logger.info("Yerel model bağlandı")
        except Exception as exc:
            logger.warning(f"Yerel model yüklenemedi: {exc}")
            self._model = None
        return self._model

    def _skorla(self, soru: str, adaylar: Sequence[str]) -> List[float]:
        """
        Adayları puanlar (uzunluk normalize edilmiş).

        ⚠ Ölçülen önyargı: model **kısa** cevabı yüksek puanlıyor
        (`-7.68` kısa vs `-7.96` uzun). Normalizasyon olmadan karar
        "en kısa aday"a düşer — içerik değil uzunluk kazanır.
        """
        m = self.baglan()
        if m is None:
            return [0.0] * len(adaylar)
        skorlar: List[float] = []
        for aday in adaylar:
            try:
                ham = m.sequence_logprob(f"{soru}\nCevap: {aday}")
                deger = float(ham) if ham is not None else -99.0
            except Exception as exc:
                logger.debug(f"Skor alınamadı: {exc}")
                deger = -99.0
            # Uzunluk normalizasyonu
            skorlar.append(deger / max(1, len(aday)) * 20.0)
        return skorlar

    def _sec(self, soru: str, adaylar: Sequence[str],
             anahtarlar: Optional[Sequence[str]] = None) -> Secim:
        """Adaylar arasından seçer."""
        skorlar = self._skorla(soru, adaylar)
        if not skorlar:
            return Secim(soru=soru)
        en_iyi = max(range(len(skorlar)), key=lambda i: skorlar[i])
        sirali = sorted(skorlar, reverse=True)
        fark = (sirali[0] - sirali[1]) / max(1e-6, abs(sirali[0])) \
            if len(sirali) > 1 else 1.0
        anahtar = (list(anahtarlar) if anahtarlar else list(adaylar))
        return Secim(
            soru=soru,
            secim=str(anahtar[en_iyi]),
            guven=max(0.0, min(1.0, fark)),
            adaylar=[str(a) for a in adaylar],
            skorlar=skorlar,
        )

    # ─── Kararlar ───

    def palet_sec(self) -> Secim:
        """Hangi fantasy paleti kullanılsın?"""
        adaylar = [
            "buyulu_orman: doygun zumrut cimen, sicak altin kum, "
            "mor-gri buyulu kaya, parlak kar",
            "ejderha_diyari: kizil toprak, zeytin yesili, koyu volkanik kaya",
            "peri_ulkesi: nane yesili cimen, pembe-bej kum, lavanta kaya",
            "kuzey_efsanesi: buzul cakil, soguk cam yesili, mavi-gri kaya",
            "col_batakligi: camur, bataklik yesili, koyu kaya",
        ]
        return self._sec(
            "Eglenceli bir fantasy oyunu icin hangi arazi renk paleti "
            "en uygun? Canli ve masalsi olmali.",
            adaylar, list(PALETLER.keys()))

    def doku_karakteri_sec(self) -> Secim:
        """Dokuların karakteri ne olsun?"""
        adaylar = [
            "puruzlu: belirgin gurultu, dogal ve canli yuzey",
            "benekli: koyu benekler, cimen ve toprakta doku hissi",
            "katmanli: yatay cizgiler, tortul kaya gorunumu",
            "kristalli: benek + hafif cizgi, buyulu kristal hissi",
            "yumusak: az gurultu, sade ve temiz yuzey",
        ]
        return self._sec(
            "Fantasy arazi dokulari nasil bir karakter tasimali?",
            adaylar, list(DOKU_KARAKTERI.keys()))

    def kar_profili_sec(self) -> Secim:
        """Kar hangi yükseklikte başlasın?"""
        adaylar = [
            "alcak: kar alcak tepelerde baslar, karlı masal dunyasi",
            "orta: kar orta yukseklikte baslar, dengeli dort mevsim",
            "yuksek: kar yalniz zirvelerde, yesil agirlikli arazi",
        ]
        return self._sec(
            "Fantasy arazide kar hangi yukseklikte baslamali?",
            adaylar, list(KAR_PROFILI.keys()))

    def katman_orani_sec(self) -> Secim:
        """Baskın katman hangisi olsun?"""
        adaylar = [
            "cimen_baskin: yesil ovalar genis, kaya az",
            "kaya_baskin: sert ve kayalik arazi, yesil az",
            "dengeli: cimen ve kaya esit agirlikta",
        ]
        return self._sec(
            "Fantasy arazide hangi katman baskin olmali?",
            adaylar, ["cimen_baskin", "kaya_baskin", "dengeli"])

    def tum_kararlar(self) -> Dict[str, Any]:
        """Tüm kararları alır ve Blender parametrelerine çevirir."""
        palet = self.palet_sec()
        doku = self.doku_karakteri_sec()
        kar = self.kar_profili_sec()
        oran = self.katman_orani_sec()

        renkler = PALETLER.get(palet.secim, PALETLER["buyulu_orman"])
        doku_p = DOKU_KARAKTERI.get(doku.secim, DOKU_KARAKTERI["puruzlu"])
        kar_p = KAR_PROFILI.get(kar.secim, KAR_PROFILI["orta"])

        esik = {
            "cimen_baskin": {"kum": (0.20, 0.28), "kaya": (0.52, 0.72)},
            "kaya_baskin": {"kum": (0.08, 0.14), "kaya": (0.28, 0.46)},
            "dengeli": {"kum": (0.14, 0.22), "kaya": (0.42, 0.62)},
        }.get(oran.secim, {"kum": (0.14, 0.22), "kaya": (0.42, 0.62)})

        return {
            "palet": palet.to_dict(),
            "doku": doku.to_dict(),
            "kar": kar.to_dict(),
            "oran": oran.to_dict(),
            "parametreler": {
                "renkler": {k: list(v) for k, v in renkler.items()},
                "doku": doku_p,
                "kar_esigi": list(kar_p),
                "kum_esigi": list(esik["kum"]),
                "kaya_esigi": list(esik["kaya"]),
            },
        }


if __name__ == "__main__":
    import json

    k = FantasyKarar()
    sonuc = k.tum_kararlar()
    print("=" * 74)
    print("  YEREL MODELIN KARARLARI")
    print("=" * 74)
    for anahtar in ("palet", "doku", "kar", "oran"):
        s = sonuc[anahtar]
        print(f"  {anahtar:8} : {s['secim']:14} (guven {s['guven']:.2f})")
    print()
    print("  BLENDER PARAMETRELERI:")
    print(json.dumps(sonuc["parametreler"], indent=2, ensure_ascii=False))
