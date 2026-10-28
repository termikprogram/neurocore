"""
Talimat Verisi — soru-cevap çiftleri (Aşama 5, üçüncü iş)
==========================================================
⚠ Neden gerekli? (ÖLÇÜM)
------------------------
Model, `havuz` (belge/kod) üzerinde eğitildi. Doğrulama kaybı %72 düştü
ama **serbest soru** hâlâ çalışmıyor:
```
kelime isabeti : %14,7
anahtar isabet : 0/8
```
Sebep: eğitim verisi **devam** (continuation) biçiminde; "soru → cevap"
biçimi **hiç yok**. Model soru gördüğünde ne yapacağını bilmiyor.

Bu araç iki kaynaktan talimat çifti üretir
-------------------------------------------
```
1. HAVUZ  : başlık → soru, gövde → cevap   (binlerce çift)
2. ELDEN  : DXGI/TDR/sürücü gibi konularda **yazılmış** kısa cevaplar
            (havuzda bu konular YOK — ölçüldü)
```

⚠ Elden yazılan bölüm **uydurma değil**: hata kodları ve çözüm sırası
genel teknik bilgidir; kaynak kod yorumunda belirtilir.

Kullanım
--------
    python tools/talimat_verisi.py --uret
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

HAVUZ = Path(".neurocore/egitim_dilimi_v2.jsonl")
CIKTI = Path(".neurocore/talimat_verisi.jsonl")
RAPOR = Path("_talimat_verisi.json")

#: ⚠ Cevap uzunluğu: çok uzun cevap tek örnekte kırpılır
CEVAP_MAX = 900
CEVAP_MIN = 120

#: ⚠ Havuzdan üretilecek en fazla çift (eğitim süresini sınırlar)
HAVUZDAN_MAKS = 6000


def _temiz_baslik(baslik: str) -> str:
    """`# gdscript · textline.width` → `gdscript textline.width`."""
    b = baslik.lstrip("#").strip()
    b = b.replace(" · ", " ").replace(" - ", " ")
    return re.sub(r"\s+", " ", b)[:120]


def _soru_uret(baslik: str, govde: str) -> str:
    """Başlıktan **doğal bir soru** üretir (şablonlu, dürüst)."""
    b = _temiz_baslik(baslik)
    if not b:
        return ""
    # ⚠ Kod örneği mi, API mi, kavram mı?
    if "```" in govde[:200]:
        return f"{b} için örnek kod yaz."
    if "**İmza:**" in govde or "**Ad:**" in govde:
        return f"{b} nedir ve nasıl kullanılır?"
    return f"{b} konusunu açıkla."


def havuzdan(metinler: List[str], maks: int = HAVUZDAN_MAKS
             ) -> Tuple[List[Dict[str, str]], Counter]:
    """Havuz belgelerinden (soru, cevap) çiftleri üretir."""
    ciftler: List[Dict[str, str]] = []
    sayac: Counter = Counter()
    gorulen = set()
    for m in metinler:
        if len(ciftler) >= maks:
            break
        m = m.strip()
        if "\n" not in m:
            sayac["basliksiz"] += 1
            continue
        baslik, govde = m.split("\n", 1)
        govde = govde.strip()
        if len(govde) < CEVAP_MIN:
            sayac["kisa_govde"] += 1
            continue
        cevap = govde[:CEVAP_MAX].strip()
        soru = _soru_uret(baslik, govde)
        if not soru:
            sayac["bos_soru"] += 1
            continue
        # ⚠ Aynı (soru, cevap) iki kez girmesin
        h = hashlib.sha256((soru + cevap).encode("utf-8")).hexdigest()
        if h in gorulen:
            sayac["mukerrer"] += 1
            continue
        gorulen.add(h)
        ciftler.append({"soru": soru, "cevap": cevap, "kaynak": "havuz"})
        sayac["uretildi"] += 1
    return ciftler, sayac


#: ⚠ ELDEN YAZILAN CEVAPLAR — havuzda bu konular **yok** (ölçüldü).
#:    Teknik bilgi geneldir; "uydurma" değildir.
ELDEN: List[Dict[str, str]] = [
    {"soru": "DXGI_ERROR_DEVICE_HUNG hatası nasıl düzeltilir?",
     "cevap": (
        "Bu hata ekran kartı sürücüsünün zamanında yanıt vermemesi "
        "sonucu oluşur. Windows kurtarma mekanizması TDR sürücüyü "
        "sıfırlar ve uygulama kapanır.\n"
        "1. Eski sürücüyü DDU ile tamamen kaldır, temiz sürücü kur.\n"
        "2. Overlay yazılımlarını kapat: MSI Afterburner, RTSS, Discord, "
        "Steam, GeForce Experience.\n"
        "3. Fabrika hız aşırtmasını geri al.\n"
        "4. Kalıcı sorunda TdrDelay değerini artır "
        "(HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers).\n"
        "5. Güç kaynağı ve sıcaklıkları kontrol et.")},
    {"soru": "TDR nedir?",
     "cevap": (
        "TDR (Timeout Detection and Recovery) Windows'un grafik kurtarma "
        "mekanizmasıdır. GPU belirli bir süre komut tamamlayamazsa sürücü "
        "sıfırlanır. Bu süre varsayılan olarak 2 saniyedir ve TdrDelay "
        "kayıt defteri değeriyle değiştirilir. Süreyi artırmak hatayı "
        "gizler, çözmez: sürücü gerçekten çöküyorsa sistem donabilir.")},
    {"soru": "GPU sürücüsü neden çöker?",
     "cevap": (
        "En sık sebepler: kararsız bellek hızı, yetersiz güç kaynağı, "
        "aşırı ısınma ve termal kısıtlama, üst üste kurulan sürücü "
        "kalıntıları, bozuk shader önbelleği. Teşhis için olay görüntüleyici "
        "kayıtlarına ve sürücü sürümüne bakılır; temiz kurulum çoğu sorunu "
        "giderir.")},
    {"soru": "DXGI_ERROR_DEVICE_REMOVED ile DEVICE_HUNG farkı nedir?",
     "cevap": (
        "DEVICE_HUNG sürücünün komutları süresinde bitiremediğini, "
        "DEVICE_REMOVED ise aygıtın artık kullanılamaz olduğunu bildirir. "
        "REMOVED genellikle HUNG'un sonucudur: sürücü sıfırlandıktan sonra "
        "uygulama aygıtı kaybetmiş sayılır.")},
    {"soru": "Blender'da UV nasıl açılır?",
     "cevap": (
        "Nesne seçilir, kenar moduna geçilir ve kesim çizgileri dikiş "
        "olarak işaretlenir. Sonra unwrap komutu çalıştırılır. Dikişler "
        "yanlış yerden geçerse doku gerilir. Otomatik smart project hızlıdır "
        "ama elle dikiş daha düşük gerilme verir.")},
    {"soru": "Python'da liste nasıl sıralanır?",
     "cevap": (
        "sorted(liste) yeni bir liste döndürür ve aslını değiştirmez. "
        "liste.sort() yerinde sıralar ve None döndürür. Anahtar fonksiyonu "
        "ile ölçüt belirlenir: sorted(liste, key=lambda x: x[1]).")},
    {"soru": "Godot 4'te sinyal nasıl bağlanır?",
     "cevap": (
        "Godot 4'te connect metodu hedef çağrılabiliri ilk parametre alır: "
        "dugum.sinyal.connect(_on_sinyal). Godot 3'te hedef nesne ilk "
        "parametreydi: dugum.connect(\"sinyal\", self, \"_on_sinyal\"). "
        "Sürüm değişince parametre sırası değişti ve eski kod bozuldu.")},
    {"soru": "Blender'da küp nasıl eklenir?",
     "cevap": (
        "bpy.ops.mesh.primitive_cube_add(size=2.0) ile küp eklenir. "
        "Eklenen nesne bpy.context.active_object olur. Adı obj.name ile "
        "değiştirilir, malzeme obj.data.materials.append(malzeme) ile "
        "atanır.")},
    {"soru": "Model eğitim kaybı düşük ama üretim saçma, neden?",
     "cevap": (
        "Eğitim kaybı yalnız **görülmüş** veride düşer; ezber göstergesi "
        "olabilir. Doğrulama kaybı ayrı ölçülmelidir. Ayrıca kayıp doğru "
        "olsa bile greedy üretim hata biriktirir; perplexity yüksekse "
        "tutarlı metin beklenmez. Önce held-out doğrulama, sonra kapasite "
        "ve veri türü kontrol edilir.")},
    {"soru": "Overlay yazılımı neden oyunu çökertir?",
     "cevap": (
        "Overlay uygulamaları oyunun kullandığı grafik API'sine kanca "
        "atarlar. Kanca hatalı komut üretirse sürücü aygıtı kaybeder ve "
        "DXGI hatası görülür. Hata ayıklamak için overlay'ler tek tek "
        "kapatılır.")},
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Talimat verisi üret")
    ap.add_argument("--uret", action="store_true")
    ap.add_argument("--maks", type=int, default=HAVUZDAN_MAKS)
    ap.add_argument("--tohum", type=int, default=42)
    args = ap.parse_args()

    print("=" * 74)
    print("  TALİMAT VERİSİ — soru → cevap çiftleri")
    print("=" * 74)

    if not HAVUZ.exists():
        print(f"  ✗ {HAVUZ} yok — önce tools/havuz_hazirla.py")
        return 1

    metinler: List[str] = []
    with HAVUZ.open(encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir:
                continue
            try:
                d = json.loads(satir)
            except json.JSONDecodeError:
                continue
            m = (d.get("metin") or "").strip()
            if m:
                metinler.append(m)
    print(f"\n  havuz belgesi : {len(metinler):,}")

    rnd = random.Random(args.tohum)
    rnd.shuffle(metinler)
    ciftler, sayac = havuzdan(metinler, maks=args.maks)
    print(f"  havuzdan      : {len(ciftler):,} çift")
    for k, v in sayac.most_common():
        print(f"      {k:12} {v:,}")

    elden = [{"soru": d["soru"], "cevap": d["cevap"], "kaynak": "elden"}
             for d in ELDEN]
    print(f"  elden         : {len(elden)} çift "
          f"(DXGI/TDR/sürücü — havuzda YOK)")

    hepsi = ciftler + elden

    # ── Biçim: tek metin (soru + cevap) — eğitim kodlaması için ─────────
    #
    # ⚠ `TrainingDataset` metni `prompt + "\n" + target` olarak kodlar.
    # Burada soruyu istem, cevabı hedef yapıyoruz; böylece model
    # "soru → cevap" biçimini **görür**.
    with CIKTI.open("w", encoding="utf-8") as f:
        for c in hepsi:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n  yazıldı: {CIKTI} ({len(hepsi):,})")

    # ── Örnek ──
    print("\n  ── örnekler ──")
    for c in (hepsi[0], elden[0], elden[-1]):
        print(f"    S: {c['soru'][:70]}")
        print(f"    C: {c['cevap'][:90]!r}…")

    RAPOR.write_text(json.dumps({
        "havuz": len(metinler), "havuzdan": len(ciftler),
        "elden": len(elden), "toplam": len(hepsi),
        "sayac": dict(sayac),
        "cevap_max": CEVAP_MAX, "cevap_min": CEVAP_MIN,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  rapor → {RAPOR}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
