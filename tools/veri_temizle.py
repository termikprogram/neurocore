"""
Eğitim Verisi Temizleyici — Tekrar Filtresi
============================================
⚠ Ölçülen sorun (tur 25)
------------------------
Model sonsuz tekrar üretiyor:
```
'# 3.3.3.3.3.1\\n\\n# 3.3.3.3.3.2\\n...'      (sayaç)
'func malzeme_uret_2 / _3 / _4 ...'          (fonksiyon tekrarı)
'mesh.vertices.append((0,0,0)) × N'          (dizi tekrarı)
```

**Kök neden:** Kasada `kod_uretim` (sentetik, 2.291 not) baskın ve bu
notlar öğretmen modelin ürettiği, **tekrar içeren** içerik taşıyor.
Model bu kalıbı **öğrendi**.

Çözüm: **Eğitim verisini temizle**
-----------------------------------
    1. Her notu **tekrar tespitinden** geçir
    2. Tekrar içerenleri **kes** veya **at**
    3. Temiz veriyle **yeniden ince ayar**
    4. Etkiyi **ölç**

⚠ Ölçülen tuzak: Veri temizliği **aşırı agresif** olursa veri kaybı
büyük olur. Ölçüm: kaç not kesildi, kaç not atıldı.

Kullanım
--------
    python tools/veri_temizle.py --analiz      # temizliği ölç
    python tools/veri_temizle.py --temizle     # temiz veri üret
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

TEMIZ_VERI = Path(".neurocore/smollm_veri_temiz.jsonl")
HAM_VERI = Path(".neurocore/smollm_veri_v2.jsonl")


#: Blender/Godot kod kalıpları — tekrar şüphelisi
KOD_KATEGORI = ("kod_uretim", "kod_ogrenme", "kod_ornek",
                "blender_api", "api_referans")

#: Kod notları kaç kez tekrarlanır (kod ağırlığı)
KOD_TEKRAR = 3


def tekrar_skoru(metin: str) -> Tuple[float, str]:
    """
    Metnin **tekrar oranını** ölçer.

    ⚠ Ölçülen 3 tekrar kalıbı:
        1. **Sayaç**: `# 3.3.3.3.3.1`, `= 3`, `= 4`, `= 5`
        2. **Son ek**: `func ad_2`, `func ad_3`, `func ad_4`
        3. **Dizi**: aynı satır N kez

    Returns:
        (oran 0..1, neden) — oran yüksekse metin tekrarlı
    """
    satirlar = [s.strip() for s in metin.splitlines() if s.strip()]
    if len(satirlar) < 6:
        return 0.0, ""

    # 1) Normalize edilmiş satır tekrarı
    norm = [re.sub(r"\d+", "#", s) for s in satirlar]
    from collections import Counter

    sayac = Counter(norm)
    en_cok = sayac.most_common(1)[0][1] if sayac else 0
    oran_satir = en_cok / len(norm)

    # 2) Sayaç deseni: ardışık satırlar aynı ama sayı artıyor
    sayac_deseni = 0
    for i in range(1, len(norm)):
        if norm[i] == norm[i - 1]:
            sayac_deseni += 1
    oran_sayac = sayac_deseni / len(norm)

    # 3) Son ek deseni: `ad_2`, `ad_3` — sayı sonda
    son_ek = 0
    for s in satirlar:
        if re.search(r"_\d+\s*[\(\s]", s) or re.search(r"_\d+$", s):
            son_ek += 1
    oran_son_ek = son_ek / len(satirlar)

    # ⚠ ÖLÇÜLEN YANLIŞ POZİTİF: `(?:\d+\.){3,}` deseni normal metinde
    # de eşleşiyor (sürüm numaraları, IP'ler, liste işaretleri) ve
    # sağlıklı notları "tekrarlı" sayıyordu (ölçüldü: 0,5 skorlu
    # notların çoğu normal). Kural **kaldırıldı**.

    oran = max(oran_satir, oran_sayac, oran_son_ek)
    nedenler: List[str] = []
    if oran_satir >= 0.50:
        nedenler.append(f"satır tekrarı %{oran_satir * 100:.0f}")
    if oran_sayac >= 0.50:
        nedenler.append(f"sayı artışı %{oran_sayac * 100:.0f}")
    if oran_son_ek >= 0.50:
        nedenler.append(f"son ek tekrarı %{oran_son_ek * 100:.0f}")
    return oran, ", ".join(nedenler)


def metni_temizle(metin: str, esik: int = 6) -> Tuple[str, str]:
    """
    Metni **tekrar başladığı yerde keser**.

    ⚠ ÖLÇÜLEN TUZAK: eşik 3 çok agresifti — **2.634 not kesildi (%47)**
    ve ortalama tekrar oranı **arttı** (0,129 → 0,152). Sebep: geçerli
    kodda aynı satır 3 kez tekrarlanır:
        `st.add_vertex(a)` × 3      (üçgen kurma)
        `add_child(mi)` × 3          (çoklu nesne)
        `pass` × 3                   (boş blok)

    Gerçek **kaçak döngü** çok daha fazla tekrar eder (10-50×). Eşik
    6'ya çıkarıldı — yalnız **gerçek** tekrarı yakalar.

    Args:
        metin: Ham metin
        esik: Aynı satır kaç kez tekrarlanınca kesilsin

    Returns:
        (temiz_metin, neden) — neden boşsa kesilmedi
    """
    satirlar = metin.splitlines()
    if len(satirlar) < esik + 2:
        return metin, ""
    sayac: Dict[str, int] = {}
    for i, s in enumerate(satirlar):
        anahtar = re.sub(r"\d+", "#", s.strip())
        anahtar = re.sub(r"\s+", " ", anahtar)
        if len(anahtar) < 6:
            continue
        sayac[anahtar] = sayac.get(anahtar, 0) + 1
        if sayac[anahtar] >= esik:
            return "\n".join(satirlar[:i + 1]), \
                f"satır {esik}× tekrar (satır {i + 1})"
    # Sayaç deseni
    norm = [re.sub(r"\d+", "#", s.strip()) for s in satirlar]
    for i in range(esik, len(norm)):
        pencere = norm[i - esik:i]
        if len(set(pencere)) == 1 and len(pencere[0]) > 6:
            return "\n".join(satirlar[:i - esik + 1]), \
                f"sayaç döngüsü (satır {i + 1})"
    return metin, ""


@dataclass
class TemizlikRaporu:
    """Veri temizliği raporu."""
    toplam: int = 0
    kesilen: int = 0
    atilan: int = 0
    temiz: int = 0
    ortalama_once: float = 0.0
    ortalama_sonra: float = 0.0
    nedenler: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "toplam": self.toplam, "kesilen": self.kesilen,
            "atilan": self.atilan, "temiz": self.temiz,
            "ortalama_tekrar_once": round(self.ortalama_once, 4),
            "ortalama_tekrar_sonra": round(self.ortalama_sonra, 4),
            "nedenler": dict(sorted(self.nedenler.items(),
                                    key=lambda x: -x[1])[:8]),
        }


def analiz(maks_not: int = 8000) -> TemizlikRaporu:
    """Kasayı analiz eder — ne kadar tekrar var?"""
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return TemizlikRaporu()

    r = TemizlikRaporu()
    toplam_once = 0.0
    toplam_sonra = 0.0
    for n in k.vault.iter_notes(limit=maks_not):
        govde = (getattr(n, "body", "") or "").strip()
        if len(govde) < 40:
            continue
        r.toplam += 1
        oran, neden = tekrar_skoru(govde)
        toplam_once += oran
        temiz, kes_neden = metni_temizle(govde)
        oran2, _ = tekrar_skoru(temiz)
        toplam_sonra += oran2
        if kes_neden:
            r.kesilen += 1
            anahtar = kes_neden.split(" (")[0]
            r.nedenler[anahtar] = r.nedenler.get(anahtar, 0) + 1
            if len(temiz) < 60:
                r.atilan += 1
        else:
            r.temiz += 1
    k.shutdown()

    if r.toplam:
        r.ortalama_once = toplam_once / r.toplam
        r.ortalama_sonra = toplam_sonra / r.toplam
    return r


def temizle(maks_not: int = 8000, kod_tekrar: int = KOD_TEKRAR) -> int:
    """
    Temizlenmiş eğitim verisi üretir.

    ⚠ Kod notları `kod_tekrar` kez tekrarlanır (kod ağırlığı). Ama
    **tekrar içeren notlar önce kesilir** — aksi hâlde tekrar kalıbı
    öğretilir.
    """
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return 0

    sayi = 0
    kesilen = 0
    atilan = 0
    with TEMIZ_VERI.open("w", encoding="utf-8") as f:
        for n in k.vault.iter_notes(limit=maks_not):
            govde = (getattr(n, "body", "") or "").strip()
            baslik = (getattr(n, "title", "") or "").strip()
            if len(govde) < 40:
                continue
            # ── Tekrarı kes ──
            temiz, neden = metni_temizle(govde)
            if neden:
                kesilen += 1
                if len(temiz) < 80:
                    atilan += 1
                    continue
                govde = temiz
            kat = getattr(n, "category", "") or ""
            metin = f"# {baslik}\n\n{govde}\n\n"
            tekrar = kod_tekrar if kat in KOD_KATEGORI else 1
            for _ in range(tekrar):
                f.write(json.dumps({"metin": metin, "kategori": kat},
                                   ensure_ascii=False) + "\n")
                sayi += 1
    k.shutdown()
    print(f"  temiz veri: {sayi} satır · {kesilen} not kesildi · "
          f"{atilan} not atıldı → {TEMIZ_VERI}")
    return sayi


def main() -> int:
    """Veri temizliği işlemlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="Eğitim verisi temizleyici")
    ap.add_argument("--analiz", action="store_true")
    ap.add_argument("--temizle", action="store_true")
    ap.add_argument("--maks-not", type=int, default=8000)
    args = ap.parse_args()

    print("=" * 78)
    print("  EĞİTİM VERİSİ TEMİZLEYİCİ — tekrar filtresi")
    print("=" * 78)

    sonuc: Dict[str, Any] = {}
    if args.analiz or not args.temizle:
        r = analiz(maks_not=args.maks_not)
        print(f"  not: {r.toplam}")
        print(f"  temiz: {r.temiz}  kesilen: {r.kesilen}  atılan: {r.atilan}")
        print(f"  ortalama tekrar oranı: {r.ortalama_once:.3f} → "
              f"{r.ortalama_sonra:.3f}")
        print(f"  nedenler: {json.dumps(r.nedenler, ensure_ascii=False)}")
        sonuc["analiz"] = r.to_dict()
    if args.temizle:
        n = temizle(maks_not=args.maks_not)
        sonuc["temiz_satir"] = n

    Path("_veri_temizlik.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
