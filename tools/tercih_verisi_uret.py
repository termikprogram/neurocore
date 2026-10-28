"""
Tercih Verisi Üretici — Karar Katmanını Güçlendirir
====================================================
⚠ Ölçülen sorun
---------------
`tools/karar_benchmark.py` ile myAi'nin karar kalitesi ölçüldü:

    DOĞRULUK      : 66,7%  (10/15)   ← rastgeleden iyi ama zayıf
    ortalama güven: 0,177            ← skorlar birbirine yakın

**Kök neden:** Model **teknik bilgi** ile eğitildi ("yüz bütçesi nedir",
"erozyon formülü") ama **tercih/estetik yargı** örnekleri görmedi
("hangi doku karakteri daha eğlenceli ve **neden**").

Seçim yapmak için modelin şunu öğrenmesi gerekir:
    * Adaylar arasındaki **farkı** ayırt etmek
    * **Neden** birinin daha iyi olduğunu bilmek
    * Tercih kalıplarını (doygunluk, okunabilirlik, ölçek uyumu) tanımak

Bu araç o boşluğu doldurur: öğretmen modelle **gerekçeli tercih
örnekleri** üretir.

Yöntem
------
    1. **Tercih alanları** — hangi kararlar için veri gerekli
    2. **Öğretmene toplu soru** — "N tercih çifti üret, her biri
       gerekçeli" (JSON)
    3. **Doğrulama** — tercih ifadesi + gerekçe dolu olmalı
    4. **Kasaya yazma** — `tercih` kategorisi
    5. **Tekrarı at**

Kullanım
--------
    python tools/tercih_verisi_uret.py --tur 40 --parti 12
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Tercih alanları — (konu, kapsam)
ALANLAR: Sequence[Tuple[str, str]] = (
    ("arazi renk paleti",
     "fantasy/gerçekçi arazi paletleri: doygunluk, sıcaklık, kaya tonu, "
     "kar parlaklığı, kum rengi. Hangisi daha eğlenceli ve neden"),
    ("doku karakteri",
     "doku stilleri: pürüzlü, benekli, katmanlı, kristalli, yumuşak. "
     "Hangisi hangi yüzeyde daha iyi ve neden"),
    ("lowpoly yüz bütçesi",
     "farklı nesneler için poligon sayısı: kaya, ağaç, karakter, ev, "
     "mobilya, çiçek. Mobil vs PC. Neden o sayı"),
    ("ağaç ve bitki formu",
     "gövde kenar sayısı, taç katmanı, dal açısı, yaprak yoğunluğu. "
     "Silüet okunabilirliği ve neden"),
    ("ışık ve gölge",
     "güneş açısı, enerji, gölge yumuşaklığı, ortam ışığı. Sahne ruhu "
     "ve okunabilirlik"),
    ("seviye ölçeği",
     "labirent hücre boyutu, duvar yüksekliği, koridor genişliği, "
     "karakter boyutu. Oynanabilirlik ve neden"),
    ("kamera açısı",
     "üçüncü şahıs, izometrik, tepeden, omuz üstü. Hangi oyun türünde "
     "hangisi ve neden"),
    ("tuş atama düzeni",
     "WASD vs ok tuşları, zıplama tuşu, etkileşim tuşu, envanter. "
     "Ergonomi ve alışkanlık"),
    ("karakter paleti",
     "kahraman, büyücü, savaşçı, goblin, iskelet renkleri. Okunabilirlik "
     "ve tür beklentisi"),
    ("malzeme pürüzlülüğü",
     "roughness/metallic değerleri: taş, ahşap, metal, buz, su, çimen. "
     "Gerçekçilik vs stilize"),
    ("seviye zorluğu",
     "düşman sayısı, hasar, sağlık, labirent boyutu. Öğrenme eğrisi"),
    ("animasyon hızı",
     "yürüme, koşma, saldırı süreleri. Hissiyat ve tepkisellik"),
    ("parçacık ve efekt",
     "meşale alevi, su dalgası, toz, büyü efekti. Hangi yoğunluk ve neden"),
    ("arayüz yoğunluğu",
     "HUD öğeleri, envanter ızgarası, yazı boyutu. Okunabilirlik"),
    ("sunucu mimarisi",
     "master/proxy/oyun sunucusu ayrımı, yetki modeli. Güvenlik ve "
     "gecikme dengesi"),
    ("senkronizasyon yöntemi",
     "reliable vs unreliable, tick hızı, interpolasyon. Tutarlılık vs "
     "gecikme"),
)

#: İstem — yer tutucu + replace (⚠ `str.format` JSON'daki `{` ile bozulur)
ISTEM = """Sen deneyimli bir oyun geliştiricisi ve teknik eğitmensin. Bir dil modeline **tercih yargısı** öğretiyorsun.

KONU: __KONU__
KAPSAM: __KAPSAM__

__ADET__ adet **tercih çifti** üret. Her çift şu alanları içersin:

- `soru`: bir tercih sorusu (örn. "Fantasy arazi için hangi kaya tonu daha iyi?")
- `iyi`: **daha iyi** olan seçenek, somut ve kısa
- `kotu`: **daha kötü** olan seçenek, somut ve kısa
- `gerekce`: `iyi` neden daha iyi — **teknik ve somut**, **en az 60 karakter**, 1-2 TAM cümle (yarım bırakma)
- `etiket`: kısa etiket (tek kelime, örn. "palet")

KURALLAR:
1. `iyi` ve `kotu` **aynı uzunlukta** olsun (model uzunluk önyargısı
   öğrenmesin — ölçüldü: model kısa cevabı yüksek puanlıyor).
2. `gerekce` teknik olmalı: "daha güzel" değil, "doygunluk düşük olduğu
   için uzak mesafede gri görünür".
3. Seçenekler **somut** olsun (renk değeri, sayı, yöntem adı).
4. Çiftler **birbirinden farklı** olsun.
5. Yalnızca JSON dizi döndür, açıklama yazma.

ÇIKTI BİÇİMİ (yalnızca bu JSON):
[{"soru": "...", "iyi": "...", "kotu": "...", "gerekce": "...", "etiket": "..."}]

Şimdi __ADET__ çift üret."""


@dataclass
class UretimSonuc:
    """Bir üretim turunun sonucu."""
    tur: int = 0
    alan: str = ""
    istenen: int = 0
    gelen: int = 0
    gecerli: int = 0
    yeni: int = 0
    sure_s: float = 0.0
    hata: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tur": self.tur, "alan": self.alan[:36],
            "istenen": self.istenen, "gelen": self.gelen,
            "gecerli": self.gecerli, "yeni": self.yeni,
            "sure_s": round(self.sure_s, 1), "hata": self.hata[:100],
        }


def _json_dizisi(metin: str) -> List[Dict[str, Any]]:
    """
    Öğretmen yanıtından JSON dizisini çıkarır — **kesik yanıta dayanıklı**.

    ⚠ Ölçülen sorun: öğretmen `finish_reason="length"` ile yanıt veriyor,
    dizi ortada kesiliyor; `json.loads` çözemez ve **0 örnek** döner.
    """
    metin = (metin or "").strip()
    if not metin:
        return []
    blok = re.search(r"```(?:json)?\s*(.+?)```", metin, re.DOTALL)
    if blok:
        metin = blok.group(1).strip()

    bas = metin.find("[")
    son = metin.rfind("]")
    if bas >= 0 and son > bas:
        try:
            veri = json.loads(metin[bas:son + 1])
            if isinstance(veri, list) and veri:
                return [x for x in veri if isinstance(x, dict)]
        except Exception:
            pass

    # Nesne nesne ayıkla
    govde = metin[bas + 1:] if bas >= 0 else metin
    kayitlar: List[Dict[str, Any]] = []
    derinlik = 0
    baslangic = -1
    tirnak = False
    kacis = False
    for i, ch in enumerate(govde):
        if kacis:
            kacis = False
            continue
        if ch == "\\":
            kacis = True
            continue
        if ch == '"':
            tirnak = not tirnak
            continue
        if tirnak:
            continue
        if ch == "{":
            if derinlik == 0:
                baslangic = i
            derinlik += 1
        elif ch == "}":
            derinlik -= 1
            if derinlik == 0 and baslangic >= 0:
                try:
                    kayitlar.append(json.loads(govde[baslangic:i + 1]))
                except Exception:
                    pass
                baslangic = -1
    return [x for x in kayitlar if isinstance(x, dict)]


def _gecerli_mi(k: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Tercih çifti geçerli mi?

    ⚠ Uzunluk dengesi kritik: `iyi` ve `kotu` uzunlukları yakın olmalı,
    aksi hâlde model **uzunluk önyargısı** öğrenir (ölçüldü: model kısa
    cevabı yüksek puanlıyor).
    """
    soru = str(k.get("soru") or "").strip()
    iyi = str(k.get("iyi") or "").strip()
    kotu = str(k.get("kotu") or "").strip()
    gerekce = str(k.get("gerekce") or "").strip()
    if not all((soru, iyi, kotu, gerekce)):
        return False, "eksik alan"
    if len(iyi) < 6 or len(kotu) < 6:
        return False, "seçenek çok kısa"
    # ⚠ Ölçülen sorun: öğretmen gerekçeyi **yarım** bırakıyordu
    # ("...patlama anında büyük ad" — 95 karakterde kesik). Eşik 25 → 60.
    if len(gerekce) < 60:
        return False, f"gerekçe çok kısa ({len(gerekce)} < 60)"
    # Cümle sonu işareti olmalı (yarım cümle tespiti)
    if gerekce.rstrip()[-1:] not in ".!?":
        return False, "gerekçe yarım cümle"
    if len(iyi) > 200 or len(kotu) > 200:
        return False, "seçenek çok uzun"
    # Uzunluk dengesi: oran 2×'i geçmesin
    oran = max(len(iyi), len(kotu)) / max(1, min(len(iyi), len(kotu)))
    if oran > 2.0:
        return False, f"uzunluk dengesiz ({oran:.1f}×)"
    if iyi.lower() == kotu.lower():
        return False, "seçenekler aynı"
    return True, ""


class TercihUretici:
    """Öğretmen modelle gerekçeli tercih örneği üretir."""

    def __init__(self, *, alanlar: Optional[Sequence[Tuple[str, str]]] = None,
                 parti: int = 12, veri_yolu: str = ".neurocore") -> None:
        self.alanlar = list(alanlar) if alanlar else list(ALANLAR)
        self.parti = max(4, min(25, parti))
        self.veri_yolu = veri_yolu
        self._gorulen: Set[str] = set()
        self._llm: Any = None
        self._kasa: Any = None
        self.sonuclar: List[UretimSonuc] = []

    def baglan(self) -> bool:
        """Öğretmeni ve kasayı hazırlar."""
        try:
            from neurocore.agent import LLMRouter

            self._llm = LLMRouter()
            self._llm.initialize(data_path=self.veri_yolu)
        except Exception as exc:
            print(f"  öğretmen bağlanamadı: {exc}")
            return False
        try:
            from neurocore.knowledge import KnowledgeLayer

            self._kasa = KnowledgeLayer()
            self._kasa.initialize()
        except Exception as exc:
            print(f"  kasa bağlanamadı: {exc}")
            return False
        self._gorulen = self._mevcut()
        print(f"  mevcut tercih örneği: {len(self._gorulen)}")
        return True

    def _mevcut(self) -> Set[str]:
        """Kasada zaten olan tercih soruları."""
        gorulen: Set[str] = set()
        if self._kasa is None:
            return gorulen
        try:
            for n in self._kasa.vault.iter_notes(limit=200000):
                if (getattr(n, "category", "") or "") == "tercih":
                    t = (getattr(n, "title", "") or "").replace("Tercih: ", "")
                    if t:
                        gorulen.add(t.strip()[:80])
        except Exception:
            pass
        return gorulen

    def tur(self, tur_no: int, alan_idx: int) -> UretimSonuc:
        """Bir üretim turu çalıştırır."""
        alan, kapsam = self.alanlar[alan_idx % len(self.alanlar)]
        s = UretimSonuc(tur=tur_no, alan=alan, istenen=self.parti)
        t0 = time.perf_counter()
        kayitlar = self._iste(alan, kapsam, self.parti)
        s.gelen = len(kayitlar)
        for k in kayitlar:
            ok, _ = _gecerli_mi(k)
            if not ok:
                continue
            s.gecerli += 1
            anahtar = str(k.get("soru") or "").strip()[:80]
            if anahtar in self._gorulen:
                continue
            if self._yaz(k, alan):
                self._gorulen.add(anahtar)
                s.yeni += 1
        if not kayitlar:
            s.hata = "yanıt yok veya JSON çözülemedi"
        s.sure_s = time.perf_counter() - t0
        self.sonuclar.append(s)
        return s

    def _iste(self, alan: str, kapsam: str,
              adet: int) -> List[Dict[str, Any]]:
        """Öğretmenden tercih çiftleri ister."""
        if self._llm is None:
            return []
        from neurocore.agent.llm import LLMMessage

        istem = (ISTEM.replace("__KONU__", alan)
                 .replace("__KAPSAM__", kapsam)
                 .replace("__ADET__", str(adet)))
        try:
            if hasattr(self._llm, "chat_with_budget"):
                y = self._llm.chat_with_budget([LLMMessage("user", istem)],
                                               temperature=0.8)
            else:
                y = self._llm.chat([LLMMessage("user", istem)],
                                   max_tokens=6000, temperature=0.8)
        except Exception:
            return []
        if not getattr(y, "ok", False):
            return []
        return _json_dizisi(getattr(y, "text", "") or "")

    def _yaz(self, k: Dict[str, Any], alan: str) -> bool:
        """
        Tercih örneğini kasaya yazar.

        ⚠ Biçim kritik: eğitim verisi olarak kullanılacak. Hem **iyi** hem
        **kötü** seçenek yazılır ki model ikisini ayırt etmeyi öğrensin.
        """
        if self._kasa is None:
            return False
        try:
            from neurocore.knowledge.vault import Note

            soru = str(k.get("soru") or "").strip()
            iyi = str(k.get("iyi") or "").strip()
            kotu = str(k.get("kotu") or "").strip()
            gerekce = str(k.get("gerekce") or "").strip()
            etiket = str(k.get("etiket") or "tercih").strip()[:20]

            govde = (f"## Tercih\n\n"
                     f"**Soru:** {soru}\n\n"
                     f"**Daha iyi:** {iyi}\n\n"
                     f"**Daha kötü:** {kotu}\n\n"
                     f"**Gerekçe:** {gerekce}\n\n"
                     f"**Alan:** {alan}\n")
            simdi = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            n = Note(
                id="", title=f"Tercih: {soru[:70]}", body=govde,
                kind="concept", category="tercih",
                tags=["tercih", "karar", etiket],
                confidence=0.85, importance=0.75,
                created=simdi, updated=simdi, version=1,
                sources=["tercih_verisi_uret"], links=[], backlinks=[],
                embedding_model="", embedding=None, path="", extra={})
            self._kasa.vault.write_note(n, touch=False, emit=False)
            return True
        except Exception:
            return False

    def kaydet(self) -> None:
        """İlerlemeyi kaydeder."""
        try:
            Path(".neurocore/tercih_ilerleme.json").write_text(
                json.dumps({
                    "tur": len(self.sonuclar),
                    "ornek": len(self._gorulen),
                    "sonuclar": [s.to_dict() for s in self.sonuclar],
                }, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass


def main() -> int:
    """Tercih verisi üretir."""
    ap = argparse.ArgumentParser(description="Tercih verisi üretici")
    ap.add_argument("--tur", type=int, default=30, help="Üretim turu")
    ap.add_argument("--parti", type=int, default=12, help="Tur başına çift")
    ap.add_argument("--alan", default="", help="Yalnızca bu alan (alt dizge)")
    ap.add_argument("--veri", default=".neurocore", help="Veri yolu")
    args = ap.parse_args()

    print("=" * 78)
    print("  TERCİH VERİSİ ÜRETİCİ — karar katmanını güçlendirir")
    print("=" * 78)

    alanlar = list(ALANLAR)
    if args.alan:
        k = args.alan.lower()
        alanlar = [x for x in alanlar if k in x[0].lower()
                   or k in x[1].lower()]
        if not alanlar:
            print(f"  alan bulunamadı: {args.alan}")
            return 1
        print(f"  filtre: {len(alanlar)} alan")

    t = TercihUretici(alanlar=alanlar, parti=args.parti,
                      veri_yolu=args.veri)
    if not t.baglan():
        return 1

    toplam_yeni = 0
    for i in range(args.tur):
        s = t.tur(i + 1, i)
        toplam_yeni += s.yeni
        im = "+" if s.yeni > 0 else " "
        print(f"  {im} tur {s.tur:3}/{args.tur} | {s.alan[:30]:32} "
              f"gelen={s.gelen:3} geçerli={s.gecerli:3} YENİ={s.yeni:3} "
              f"({s.sure_s:.0f}s) {s.hata[:30]}", flush=True)
        t.kaydet()

    print()
    print(f"  TOPLAM: yeni={toplam_yeni}  "
          f"kasadaki tercih={len(t._gorulen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
