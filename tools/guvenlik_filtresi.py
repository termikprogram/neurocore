"""
myAi Güvenlik Filtresi — Aday Değerleri Doğrular
==================================================
⚠ Ölçülen gerçek (tur 14)
-------------------------
myAi **saçma seçenekleri eliyor**, ince ayrım yapamıyor:

    OK [+2,39] Hangi tuş düzeni?    ← yanlışlar: "tuş yok", "100 tuş"
    OK [+1,67] Seviye türü?         ← yanlışlar: "düz boş alan"
    OK [+0,95] HUD gösterilsin mi?  ← yanlışlar: "oyuncu canını bilemez"
    OK [+0,79] Fantasy paleti?      ← yanlışlar: "neon pembe"
       [     ] Labirent boyutu?     ← 21 vs 2 vs 501
       [     ] Karakter hızı?       ← 3.5 vs 5.0 vs 7.5 (hepsi makul)

**Doğru kullanım:** myAi bir **güvenlik filtresidir** — saçma/tehlikeli
aday değerleri eler. İnce ayarlar **varsayılan değerlerden** gelir.

Mimari
------
    ┌──────────────────────────────────────────┐
    │  1. Varsayılan değer + makul aralık      │
    │     (şablondan — güvenilir)              │
    ├──────────────────────────────────────────┤
    │  2. myAi GÜVENLİK FİLTRESİ               │
    │     Aşırı/absürt adayları ELE            │
    │     (ölçülen güç: 4/7, hepsi absürt)     │
    ├──────────────────────────────────────────┤
    │  3. Godot doğrulaması                    │
    │     Üret, çalıştır, hata varsa dur       │
    └──────────────────────────────────────────┘

⚠ CUDA/VRAM notu: myAi modeli yükler. Eğitim çalışırken **çakışır**.
Bu araç ayrı süreçte çalışacak şekilde tasarlandı (`--alt-surec`).

Kullanım
--------
    # Doğrudan (model bellekte)
    python tools/guvenlik_filtresi.py --deger "labirent_boyut" 501

    # Alt süreç olarak (üretim döngüsünden)
    from tools.guvenlik_filtresi import filtrele_alt_surec
    sonuc = filtrele_alt_surec("labirent_boyut", [21, 2, 501])
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)


#: Üretim parametreleri — makul aralıklar ve sorular
#:
#: `varsayilan`: güvenilir başlangıç değeri (şablondan)
#: `aralik`: (min, max) — dışındakiler **absürt** sayılır
#: `soru`: myAi'ye sorulacak soru
#: `birim`: değer biçimi (sayı, metin)
PARAMETRELER: Dict[str, Dict[str, Any]] = {
    "labirent_boyut": {
        "varsayilan": 21,
        "aralik": (7, 61),
        "soru": "Labirent kaç hücre genişliğinde olsun?",
        "birim": "hucre",
    },
    "karakter_hizi": {
        "varsayilan": 5.0,
        "aralik": (1.0, 20.0),
        "soru": "Karakterin yürüme hızı kaç birim/saniye olsun?",
        "birim": "birim/sn",
    },
    "agac_sayisi": {
        "varsayilan": 40,
        "aralik": (5, 500),
        "soru": "Sahneye kaç ağaç yerleştirilsin?",
        "birim": "agac",
    },
    "kaya_sayisi": {
        "varsayilan": 25,
        "aralik": (3, 300),
        "soru": "Sahneye kaç kaya yerleştirilsin?",
        "birim": "kaya",
    },
    "arena_yaricap": {
        "varsayilan": 22.0,
        "aralik": (8.0, 80.0),
        "soru": "Arena yarıçapı kaç birim olsun?",
        "birim": "birim",
    },
    "magara_tunel": {
        # ⚠ Test yakaladı: `Kararlar` varsayılanı 10, filtre 12 idi.
        # Tutarlılık için `Kararlar` esas alınır (üretim onu kullanır).
        "varsayilan": 10,
        "aralik": (4, 40),
        "soru": "Mağara kaç tünel noktası içersin?",
        "birim": "nokta",
    },
    "yukseklik": {
        "varsayilan": 6.0,
        "aralik": (0.5, 30.0),
        "soru": "Arazi yükseklik ölçeği ne olsun?",
        "birim": "birim",
    },
}


@dataclass
class FiltreSonuc:
    """Güvenlik filtresi sonucu."""
    parametre: str = ""
    deger: Any = None
    guvenli: bool = True
    neden: str = ""
    skor: float = 0.0
    myAi_kullanildi: bool = False
    varsayilan: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parametre": self.parametre, "deger": self.deger,
            "guvenli": self.guvenli, "neden": self.neden[:100],
            "skor": round(self.skor, 3),
            "myAi_kullanildi": self.myAi_kullanildi,
            "varsayilan": self.varsayilan,
        }


def aralik_kontrol(parametre: str, deger: Any) -> Tuple[bool, str]:
    """
    **Aralık kontrolü** — myAi'den bağımsız, her zaman çalışır.

    ⚠ Bu birincil savunmadır: absürt değerler burada yakalanır.
    myAi **ikincil** savunmadır (myAi kararsızsa varsayılan kullanılır).
    """
    p = PARAMETRELER.get(parametre)
    if not p:
        return True, "bilinmeyen parametre"
    alt, ust = p["aralik"]
    try:
        d = float(deger)
    except (TypeError, ValueError):
        return False, f"sayısal değil: {deger!r}"
    if d < alt:
        return False, f"çok küçük ({d} < {alt})"
    if d > ust:
        return False, f"çok büyük ({d} > {ust})"
    return True, ""


class GuvenlikFiltresi:
    """Aday parametre değerlerini doğrular."""

    def __init__(self) -> None:
        self._model: Any = None

    def baglan(self) -> bool:
        """myAi modelini yükler."""
        try:
            from neurocore import NeuroCoreEngine

            motor = NeuroCoreEngine()
            motor.initialize()
            self._model = motor.model
            return True
        except Exception:
            return False

    def _myAi_skor(self, parametre: str, adaylar: Sequence[Any]) -> List[float]:
        """
        myAi'nin adaylara verdiği skorlar (uzunluk normalize).

        ⚠ Ölçülen önyargı: model **kısa** cevabı yüksek puanlıyor;
        normalizasyon olmadan karar "en kısa aday"a düşer.
        """
        if self._model is None:
            return []
        p = PARAMETRELER.get(parametre) or {}
        soru = p.get("soru") or f"{parametre} değeri ne olsun?"
        birim = p.get("birim") or ""
        skorlar: List[float] = []
        for a in adaylar:
            metin = f"{a} {birim}".strip()
            try:
                ham = self._model.sequence_logprob(f"{soru}\nCevap: {metin}")
                deger = float(ham) if ham is not None else -99.0
            except Exception:
                deger = -99.0
            skorlar.append(deger / max(1, len(metin)) * 20.0)
        return skorlar

    def filtrele(self, parametre: str,
                 adaylar: Sequence[Any]) -> FiltreSonuc:
        """
        Adaylar arasından **güvenli** olanı seçer.

        Akış:
            1. **Aralık kontrolü** — absürt adayları ele (myAi'siz, güvenilir)
            2. Kalan adaylar içinde myAi **en iyisini** seçer
            3. myAi kararsızsa (skorlar yakın) → **varsayılan** kullanılır
        """
        p = PARAMETRELER.get(parametre) or {}
        s = FiltreSonuc(parametre=parametre,
                        varsayilan=p.get("varsayilan"))

        # ── 1) Aralık kontrolü ──
        guvenli_adaylar: List[Any] = []
        for a in adaylar:
            ok, neden = aralik_kontrol(parametre, a)
            if ok:
                guvenli_adaylar.append(a)
        if not guvenli_adaylar:
            s.guvenli = False
            s.deger = p.get("varsayilan")
            s.neden = "tüm adaylar aralık dışı — varsayılan kullanıldı"
            return s

        # Tek aday kaldıysa myAi'ye sormaya gerek yok
        if len(guvenli_adaylar) == 1:
            s.deger = guvenli_adaylar[0]
            s.neden = "tek güvenli aday (aralık kontrolü)"
            return s

        # ── 2) myAi seçimi ──
        skorlar = self._myAi_skor(parametre, guvenli_adaylar)
        if not skorlar:
            s.deger = p.get("varsayilan")
            s.neden = "myAi yok — varsayılan"
            return s

        s.myAi_kullanildi = True
        en_iyi = max(range(len(skorlar)), key=lambda i: skorlar[i])
        fark = 0.0
        if len(skorlar) > 1:
            sirali = sorted(skorlar, reverse=True)
            fark = abs(sirali[0] - sirali[1]) / max(1e-6, abs(sirali[0]))
        s.skor = fark

        # ── 3) Kararsızlık kontrolü ──
        #
        # ⚠ Ölçülen gerçek (tur 13): makul adaylar arasında model
        # **rastgele** seçiyor (%50). Skorlar yakınsa myAi'ye güvenmek
        # yerine **varsayılan** kullanmak daha güvenlidir.
        ESIK = 0.05
        if fark < ESIK:
            s.deger = p.get("varsayilan")
            s.neden = (f"myAi kararsız (fark {fark:.3f} < {ESIK}) — "
                       f"varsayılan kullanıldı")
            return s

        s.deger = guvenli_adaylar[en_iyi]
        s.neden = f"myAi seçti (fark {fark:.3f})"
        return s


# ─── Alt süreç arayüzü ───
#
# ⚠ Neden alt süreç: myAi modeli ~1,8 GB VRAM ister. Eğitim çalışırken
# aynı süreçte yüklemek CUDA bağlamını bozar (ölçüldü, tur 7).

def filtrele_alt_surec(parametre: str, adaylar: Sequence[Any],
                       zaman_asimi: float = 120.0) -> Dict[str, Any]:
    """
    Filtreyi **ayrı süreçte** çalıştırır (VRAM çakışmasını önler).

    Returns:
        `FiltreSonuc.to_dict()` veya hata sözlüğü
    """
    # Önce aralık kontrolü — myAi gerekmez, her zaman güvenilir
    p = PARAMETRELER.get(parametre) or {}
    guvenli = [a for a in adaylar if aralik_kontrol(parametre, a)[0]]
    if len(guvenli) == 1:
        return FiltreSonuc(parametre=parametre, deger=guvenli[0],
                           neden="tek güvenli aday").to_dict()
    if not guvenli:
        return FiltreSonuc(parametre=parametre, guvenli=False,
                           deger=p.get("varsayilan"),
                           neden="tüm adaylar aralık dışı").to_dict()

    komut = [sys.executable, "-u", "-X", "utf8",
             str(Path(__file__).resolve()),
             "--parametre", parametre,
             "--adaylar", json.dumps(list(adaylar))]
    try:
        s = subprocess.run(komut, capture_output=True, text=True,
                           timeout=zaman_asimi, cwd=str(_KOK),
                           encoding="utf-8", errors="replace")
        for satir in (s.stdout or "").splitlines():
            if satir.startswith("SONUC "):
                return json.loads(satir[len("SONUC "):])
    except Exception as exc:
        return {"parametre": parametre, "hata": f"{type(exc).__name__}",
                "deger": p.get("varsayilan"), "guvenli": True,
                "neden": "alt süreç başarısız — varsayılan"}
    return {"parametre": parametre, "deger": p.get("varsayilan"),
            "guvenli": True, "neden": "sonuç alınamadı — varsayılan"}


def main() -> int:
    """Tek parametre filtresini çalıştırır."""
    ap = argparse.ArgumentParser(description="myAi güvenlik filtresi")
    ap.add_argument("--parametre", required=True)
    ap.add_argument("--adaylar", required=True,
                    help="JSON dizi, ör. '[21, 2, 501]'")
    ap.add_argument("--sessiz", action="store_true")
    args = ap.parse_args()

    try:
        adaylar = json.loads(args.adaylar)
    except Exception as exc:
        print(f"SONUC {json.dumps({'hata': str(exc)})}")
        return 1

    g = GuvenlikFiltresi()
    if not g.baglan() and not args.sessiz:
        print("  uyarı: myAi yüklenemedi — yalnız aralık kontrolü",
              file=sys.stderr)
    s = g.filtrele(args.parametre, adaylar)
    print(f"SONUC {json.dumps(s.to_dict(), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
