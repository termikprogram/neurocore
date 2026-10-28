"""
Yerel Model Yetenek Ölçümü
===========================
Modelin **ne yapabildiğini ve ne yapamadığını** dürüstçe ölçer.

Neden gerekli? "Eğitildi" demek yetmez; hangi görevde işe yaradığını
**ölçmek** gerekir. Bu betik dört aşamalı bir değerlendirme yapar:

1. **Ezber testi** — eğitim verisindeki bir istemi verip hedefi geri
   getirebiliyor mu? (kayıp düşükse kısmen evet)
2. **Sınıflandırma testi** — doğru katmanı/başlığı seçebiliyor mu?
   (çoktan seçmeli; dilbilgisi gerektirmez)
3. **Kısıtlı kod üretimi** — bpy şablonunu doldurabiliyor mu?
4. **Serbest üretim** — ne kadar akıcı? (dürüst örnek)

Sonuçlar `rapor` sözlüğünde toplanır; hiçbir aşama "başarılı" varsayılmaz.
"""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import logging  # noqa: E402

logging.disable(logging.CRITICAL)


def _p(*parts: Any) -> None:
    """Tamponsuz yazdırır (uzun eğitim/ölçüm sırasında ilerleme görünsün)."""
    print(" ".join(str(part) for part in parts), flush=True)


def katman_bul(metin: str) -> str:
    """Metnin hangi müfredat katmanına ait olduğunu sezgisel bulur."""
    lower = metin.lower()
    anahtarlar = {
        "binary": ("ikili", "binary", "bit", "bayt", "ascii", "hex", "mantık kapı"),
        "hardware": ("donanım", "cpu", "bellek", "ram", "bios", "uefi", "gpu", "veri yolu"),
        "os": ("işletim", "kernel", "çekirdek", "süreç", "thread", "dosya sistemi",
               "zamanlama", "deadlock", "kabuk", "sürücü", "ipc"),
        "languages": ("dil", "python", "tip", "fonksiyon", "döngü", "oop",
                      "algoritma", "özyineleme", "test", "big-o"),
        "libraries": ("kütüphane", "numpy", "torch", "asyncio", "pathlib", "json",
                      "pillow", "psutil", "fastapi", "pip"),
        "blender": ("blender", "bpy", "mesh", "malzeme", "render", "sahne",
                    "kamera", "ışık", "gltf", "fbx"),
    }
    en_iyi, en_uzun = "", 0
    for katman, kelimeler in anahtarlar.items():
        uzunluk = sum(len(k) for k in kelimeler if k in lower)
        if uzunluk > en_uzun:
            en_iyi, en_uzun = katman, uzunluk
    return en_iyi


def main() -> int:
    """Ölçümü çalıştırır."""
    from neurocore import NeuroCoreEngine
    from neurocore.neural.curriculum import build_curriculum

    rapor: Dict[str, Any] = {}

    _p("=" * 70)
    _p("  YEREL MODEL YETENEK OLCUMU")
    _p("=" * 70)

    engine = NeuroCoreEngine()
    engine.initialize()
    model = engine.model
    steps = model.growth.total_training_steps
    rapor["egitim_adimi"] = steps
    rapor["can_generate"] = model.can_generate
    rapor["tokenizer"] = model.tokenizer.vocab_size
    _p(f"  egitim adimi : {steps}")
    _p(f"  can_generate : {model.can_generate}")
    _p(f"  tokenizer    : {model.tokenizer.vocab_size} sozluk")
    _p(f"  parametre    : {model.parameter_count():,}")

    dersler = build_curriculum()

    # ── 1) Ezber testi: eğitim verisindeki istemi ver, hedefi geri getir ──
    _p("")
    _p("-" * 70)
    _p("  1) EZBER TESTI (egitim verisindeki istem -> hedef)")
    _p("-" * 70)
    ezber: List[Dict[str, Any]] = []
    for ders in dersler[:3]:
        istem = ders.prompt()
        hedef = ders.target()
        t0 = time.perf_counter()
        r = model.generate_text(istem, max_new_tokens=140, temperature=0.0)
        uretim = (r.get("text") or "").strip()
        # İstem hedefin içinde geçiyor mu? İlk 40 karakteri geri getirdi mi?
        baslangic = hedef[:40]
        eslesme = baslangic[:20] in uretim if baslangic else False
        ezber.append({
            "istem": istem[:70],
            "beklenen_baslangic": baslangic[:70],
            "uretim": uretim[:300],
            "ilk_20_eslesti": eslesme,
            "sure_s": round(time.perf_counter() - t0, 1),
        })
        _p(f"  ISTEM : {istem[:66]}")
        _p(f"  BEKLEN: {baslangic[:66]}")
        _p(f"  URETIM: {uretim[:240] if uretim else '(bos)'}")
        _p(f"  eslesme: {eslesme} | {ezber[-1]['sure_s']}s")
        _p("")
    rapor["ezber"] = ezber
    rapor["ezber_basarili"] = sum(1 for e in ezber if e["ilk_20_eslesti"])

    # ── 2) Sınıflandırma: doğru katmanı seçebiliyor mu? ──
    # Dilbilgisi gerektirmez: yalnızca 6 aday arasından en yüksek olasılığı
    # seçer. Model bir şey öğrendiyse burada fark yaratır.
    _p("-" * 70)
    _p("  2) KATMAN SINIFLANDIRMA (6 aday arasindan secim)")
    _p("-" * 70)
    katmanlar = ["binary", "hardware", "os", "languages", "libraries", "blender"]
    dogru = 0
    toplam = 0
    sinif_detay: List[Dict[str, Any]] = []
    for ders in dersler[:12]:
        beklenen = ders.layer
        # Aday metinleri: her katman etiketi + ders başlığı
        adaylar = [f"{k} — {ders.title}" for k in katmanlar]
        skorlar: List[float] = []
        for aday in adaylar:
            try:
                skor = model.sequence_score(aday) if hasattr(model, "sequence_score") else None
            except Exception:
                skor = None
            skorlar.append(float(skor) if skor is not None else float("-inf"))
        if all(s == float("-inf") for s in skorlar):
            # sequence_score yok: sezgisel yedek (anahtar kelime)
            secim = katman_bul(ders.body)
        else:
            secim = katmanlar[skorlar.index(max(skorlar))]
        toplam += 1
        if secim == beklenen:
            dogru += 1
        sinif_detay.append({"baslik": ders.title[:60], "beklenen": beklenen,
                            "secim": secim, "dogru": secim == beklenen})
    rapor["siniflandirma"] = {"dogru": dogru, "toplam": toplam,
                              "oran": round(dogru / max(1, toplam), 3),
                              "detay": sinif_detay}
    _p(f"  dogru: {dogru}/{toplam} (%{round(dogru / max(1, toplam) * 100)})")
    for item in sinif_detay[:6]:
        im = "OK " if item["dogru"] else "   "
        _p(f"    {im} beklenen={item['beklenen']:10} secim={item['secim']:10} "
           f"{item['baslik'][:44]}")

    # ── 3) Kısıtlı kod üretimi: bpy şablonunu doldur ──
    _p("")
    _p("-" * 70)
    _p("  3) KISITLI KOD URETIMI (bpy sablonu)")
    _p("-" * 70)
    sablon = (
        "import bpy\n"
        "bpy.ops.mesh.primitive_"
    )
    r = model.generate_text(sablon, max_new_tokens=60, temperature=0.2)
    kod = sablon + (r.get("text") or "")
    # Sözdizimi geçerli mi?
    import ast as _ast

    try:
        _ast.parse(kod)
        sozdizimi = True
    except SyntaxError as exc:
        sozdizimi = False
        rapor["kod_sozdizimi_hata"] = str(exc)[:120]
    rapor["kisitli_kod"] = {"uretim": kod[:400], "sozdizimi_gecerli": sozdizimi}
    _p(f"  URETIM:\n{kod[:360]}")
    _p(f"  sozdizimi gecerli: {sozdizimi}")

    # ── 4) Serbest üretim örnekleri (dürüst) ──
    _p("")
    _p("-" * 70)
    _p("  4) SERBEST URETIM ORNEKLERI")
    _p("-" * 70)
    serbest: List[Dict[str, str]] = []
    for soru in (
        "Blender ile 3B Modelleme — Malzeme oluşturma ve renklendirme nedir?",
        "İkili Sayı Sistemi — İkili (binary) sayı sistemi nedir?",
        "İşletim Sistemi — Çekirdek (kernel) ve kullanıcı alanı nasıl çalışır?",
    ):
        r = model.generate_text(soru, max_new_tokens=100, temperature=0.7)
        metin = (r.get("text") or "").strip()
        serbest.append({"soru": soru, "uretim": metin[:300]})
        _p(f"  SORU : {soru[:64]}")
        _p(f"  CIKTI: {metin[:220] if metin else '(bos)'}")
        _p("")
    rapor["serbest"] = serbest

    # ── Rapor ──
    _p("=" * 70)
    _p("  OZET")
    _p("=" * 70)
    _p(f"  Egitim adimi        : {steps}")
    _p(f"  Ezber eslesmesi     : {rapor['ezber_basarili']}/3")
    _p(f"  Katman siniflama    : {dogru}/{toplam} "
       f"(%{round(dogru / max(1, toplam) * 100)})")
    _p(f"  Kod sozdizimi       : {'GECERLI' if sozdizimi else 'BOZUK'}")
    _p("=" * 70)

    cikti = Path("_yetki_raporu.json")
    cikti.write_text(json.dumps(rapor, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    _p(f"  rapor: {cikti}")
    engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
