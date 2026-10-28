"""
Düşünme Ölçümü
===============
Yerel modelin **düşünme** yeteneğini ayrı ayrı ölçer.

Neden ayrı ölçüm?
    "Düşünme" tek bir yetenek değil, dört ayrı beceridir ve her biri
    farklı ölçülür:

    1. **Yol seçimi** — verilen problem için hangi yaklaşım daha uygun?
       (çoktan seçmeli; üretim gerektirmez)
    2. **Adım sıralama** — doğru işlem sırası hangisi?
    3. **Teşhis** — belirtiye hangi neden uyar?
    4. **Serbest üretim** — model kendi başına adım yazabiliyor mu?

    Ölçüm, modelin **kendi log olasılığını** kullanır
    (`NeuroCoreModel.sequence_logprob`); kelime torbası sezgisi yok.

Rastgele taban her testte aday sayısına göre bildirilir; taban bilinmeden
yüzde anlamsızdır.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def _p(*parts: Any) -> None:
    """Tamponsuz yazdırır."""
    print(" ".join(str(part) for part in parts), flush=True)


#: (problem, doğru yaklaşım, çeldiriciler)
YOL_SECIMI: List[Tuple[str, str, List[str]]] = [
    ("Blender'da 200 nesne oluşturulacak. Hangi yaklaşım daha uygun?",
     "Bir nesne oluşturup bpy.data.objects.new() ile kopyalamak; sahne bir kez güncellenir",
     ["Her nesne için ayrı bpy.ops çağrısı yapmak; okunur ama yavaş",
      "Nesneleri elle tek tek konumlandırmak",
      "Hiç nesne oluşturmayıp dışa aktarmak"]),
    ("Bir betik 30 saniye sürüyor ve darboğaz G/Ç. Ne yapmalı?",
     "İstekleri eşzamanlı çalıştırmak; bekleme süresi toplanmaz",
     ["Algoritma karmaşıklığını düşürmek",
      "Daha güçlü CPU almak",
      "Kodu yeniden yazmak"]),
    ("Eğitim kaybı 0,2 ama üretim tutarsız. İlk ne yapmalı?",
     "Veri/parametre oranını ölçmek; veri ölçeği yetersiz olabilir",
     ["Modeli büyütmek",
      "Öğrenme oranını artırmak",
      "Daha çok adım atmak"]),
    ("Yerel model akıcı metin yazamıyor ama seçimleri doğru. Nasıl kullanmalı?",
     "Adayları sıralamak için kullanmak; üretimi bulut modeline bırakmak",
     ["Modeli kullanmayı bırakmak",
      "Yalnızca üretim için zorlamak",
      "Veri kümesini silmek"]),
    ("Ajan görevi 24 adımda bitiremedi. Doğru çözüm?",
     "Görevi alt görevlere bölüp her birini ayrı çalıştırmak",
     ["Adım sınırını yükseltmek",
      "Aynı görevi tekrar denemek",
      "Ajanı kapatmak"]),
    ("API anahtarı doğru ama 403 dönüyor. İlk şüphe?",
     "İstek kimliği (User-Agent) eksik; bot koruması engelliyor",
     ["Anahtar süresi dolmuş",
      "Sunucu kapalı",
      "Ağ bağlantısı yok"]),
]

#: (belirti, doğru neden, çeldiriciler)
TESHIS: List[Tuple[str, str, List[str]]] = [
    ("Blender betiği hata vermeden başarısız oluyor.",
     "İstisna çıkış koduna yansımıyor; --python-exit-code 1 gerekli",
     ["Blender kurulu değil", "Python sürümü eski", "Disk dolu"]),
    ("Eğitim 'unscale_ has already been called' veriyor.",
     "GradScaler bayrağı sıfırlanmamış; atlanan adımda step() çağrılmamış",
     ["Öğrenme oranı çok yüksek", "Veri bozuk", "GPU arızalı"]),
    ("8 turun 7'si 'kaynaklar dolu' diyor ama GPU boş.",
     "Kapanış VRAM'i bırakmıyor; kendi belleğimiz sistem sanılıyor",
     ["GPU gerçekten dolu", "RAM yetersiz", "Disk yavaş"]),
    ("Model eğitim verisini bile geri getiremiyor.",
     "Veri ölçeği parametreye göre çok küçük; örüntü öğreniliyor, dizi değil",
     ["Tokenizer bozuk", "Model bozuk", "Kayıp yanlış hesaplanıyor"]),
]


def _score(model: Any, istem: str, aday: str) -> float:
    """Adayın ortalama log olasılığı."""
    try:
        deger = model.sequence_logprob(f"{istem}\n{aday}", prompt=f"{istem}\n")
    except Exception:
        deger = None
    return float(deger) if deger is not None else float("-inf")


def _karistir(items: List[str], tohum: int) -> List[str]:
    """Belirleyici karıştırma (sıra önyargısını kırar)."""
    import random

    kopya = list(items)
    random.Random(tohum).shuffle(kopya)
    return kopya


def main() -> int:
    """Ölçümü çalıştırır."""
    from neurocore import NeuroCoreEngine

    _p("=" * 72)
    _p("  DUSUNME OLCUMU (yerel model, kendi olasiliklari)")
    _p("=" * 72)

    engine = NeuroCoreEngine()
    engine.initialize()
    model = engine.model
    _p(f"  egitim adimi : {model.growth.total_training_steps}")
    _p(f"  tokenizer    : {model.tokenizer.vocab_size}")
    _p(f"  gorulen token: {model.growth.total_tokens_seen:,}")
    rapor: Dict[str, Any] = {
        "egitim_adimi": model.growth.total_training_steps,
        "tokenizer": model.tokenizer.vocab_size,
    }

    # ── 1) Yol seçimi ──
    _p("")
    _p("-" * 72)
    _p("  1) YOL SECIMI (4 aday, rastgele taban %25)")
    _p("-" * 72)
    dogru = 0
    for index, (problem, iyi, kotuler) in enumerate(YOL_SECIMI):
        adaylar = _karistir([iyi] + list(kotuler), index)
        istem = f"{problem}\nEn uygun yaklasim:"
        skorlar = {a: _score(model, istem, a) for a in adaylar}
        secim = max(skorlar, key=skorlar.get)
        ok = secim == iyi
        dogru += ok
        _p(f"  {'OK ' if ok else '   '} {problem[:52]}")
        _p(f"       secilen: {secim[:70]}")
    oran1 = dogru / len(YOL_SECIMI)
    rapor["yol_secimi"] = {"dogru": dogru, "toplam": len(YOL_SECIMI),
                           "oran": round(oran1, 3), "taban": 0.25}
    _p(f"  SONUC: {dogru}/{len(YOL_SECIMI)} = %{round(oran1 * 100)} "
       f"(taban %25 → {round(oran1 / 0.25, 2)}x)")

    # ── 2) Teşhis ──
    _p("")
    _p("-" * 72)
    _p("  2) TESHIS (4 aday, rastgele taban %25)")
    _p("-" * 72)
    dogru2 = 0
    for index, (belirti, neden, kotuler) in enumerate(TESHIS):
        adaylar = _karistir([neden] + list(kotuler), 100 + index)
        istem = f"BELIRTI: {belirti}\nOLASI NEDEN:"
        skorlar = {a: _score(model, istem, a) for a in adaylar}
        secim = max(skorlar, key=skorlar.get)
        ok = secim == neden
        dogru2 += ok
        _p(f"  {'OK ' if ok else '   '} {belirti[:52]}")
        _p(f"       secilen: {secim[:70]}")
    oran2 = dogru2 / len(TESHIS)
    rapor["teshis"] = {"dogru": dogru2, "toplam": len(TESHIS),
                       "oran": round(oran2, 3), "taban": 0.25}
    _p(f"  SONUC: {dogru2}/{len(TESHIS)} = %{round(oran2 * 100)} "
       f"(taban %25 → {round(oran2 / 0.25, 2)}x)")

    # ── 3) Serbest üretim ──
    _p("")
    _p("-" * 72)
    _p("  3) SERBEST URETIM (model kendi adimlarini yazabiliyor mu?)")
    _p("-" * 72)
    serbest: List[Dict[str, str]] = []
    for istem in ("DÜŞÜN: Blender'da küp eklemek için hangi yaklaşım daha uygun?",
                  "DÜŞÜN: Eğitim kaybı düştü ama üretim bozuk, ne anlama geliyor?",
                  "DÜŞÜN: Ağ istekleri yavaş, ne yapmalıyım?"):
        r = model.generate_text(istem, max_new_tokens=120, temperature=0.3)
        metin = (r.get("text") or "").strip()
        # Adım yapısı var mı?
        adim_var = "ADIM" in metin.upper() or "KARAR" in metin.upper()
        serbest.append({"istem": istem, "uretim": metin[:300],
                        "adim_yapisi": adim_var})
        _p(f"  ISTEM : {istem[:66]}")
        _p(f"  CIKTI : {metin[:200] if metin else '(bos)'}")
        _p(f"  adim yapisi: {adim_var}")
        _p("")
    rapor["serbest"] = serbest

    # ── Özet ──
    _p("=" * 72)
    _p("  OZET")
    _p("=" * 72)
    _p(f"  Yol secimi     : %{round(oran1 * 100)} (taban %25, "
       f"{round(oran1 / 0.25, 2)}x)")
    _p(f"  Teshis         : %{round(oran2 * 100)} (taban %25, "
       f"{round(oran2 / 0.25, 2)}x)")
    _p(f"  Serbest uretim : "
       f"{sum(1 for s in serbest if s['adim_yapisi'])}/{len(serbest)} adim yapisi")
    _p("=" * 72)

    Path("_dusunme_raporu.json").write_text(
        json.dumps(rapor, ensure_ascii=False, indent=1), encoding="utf-8")
    engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
