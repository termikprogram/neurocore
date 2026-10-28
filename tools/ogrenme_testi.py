"""
"Bilmiyorsa Öğrenir mi?" — Uçtan Uca Test
==========================================
Ajanın bilmediği bir konuyla karşılaştığında **gerçekten öğrenip
öğrenmediğini** ölçer. Sahte bileşen yok; gerçek öğretmen (API) ve gerçek
bilgi kasası kullanılır.

Akış:
    1. Kasa durumu ölçülür (konu hakkında ne biliyor?)
    2. `learn.topic` çağrılır — gerçek öğretmen, gerçek kaynak indirme
    3. Öğrenme sonucu doğrulanır (cevap, kaynak adresleri)
    4. Kasa **tekrar** sorgulanır — bilgi gerçekten yazıldı mı?
    5. İkinci çağrıda öğrenmeye gerek kalmadığı doğrulanır (kasadan gelir)

Bu, kullanıcının sorusunun kanıtıdır: **"bir iş verildiğinde bilmiyorsa
öğrenerek yapacak mı?"**

Kullanım:
    python tools/ogrenme_testi.py
    python tools/ogrenme_testi.py --konu "Godot 4.8 sinyal sistemi"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

#: Bilinmesi muhtemel olmayan, **doğrulanabilir** konular.
#:
#: Seçim ölçütü: resmî dokümantasyonu olan, kaynağı bulunabilir konular.
#: Uydurma kaynak üretilirse test bunu yakalar (adres indirilemez).
VARSAYILAN_KONULAR: List[str] = [
    "Godot 4.8'de CharacterBody3D ile hareket ve yerçekimi nasıl uygulanır",
    "Blender 5'te Geometry Nodes ile prosedürel dağılım nasıl yapılır",
]


def _p(*parts: Any) -> None:
    """Tamponsuz yazdırır."""
    print(" ".join(str(part) for part in parts), flush=True)


def _bolum(metin: str) -> None:
    """Bölüm başlığı."""
    _p("")
    _p("-" * 74)
    _p(f"  {metin}")
    _p("-" * 74)


def main() -> int:
    """Öğrenme akışını uçtan uca test eder."""
    parser = argparse.ArgumentParser(description="Öğrenme testi")
    parser.add_argument("--konu", default="", help="Öğrenilecek konu")
    parser.add_argument("--ikinci", action="store_true",
                        help="İkinci turda kasadan gelmesini de doğrula")
    args = parser.parse_args()

    from neurocore import NeuroCoreEngine
    from neurocore.agent import LLMRouter
    from neurocore.agent.knowledge_work import KnowledgeWorkflow
    from neurocore.agent.study import SelfStudy

    konular = [args.konu] if args.konu else list(VARSAYILAN_KONULAR)

    _p("=" * 74)
    _p("  BİLMİYORSA ÖĞRENİR Mİ? — UÇTAN UCA TEST")
    _p("=" * 74)

    motor = NeuroCoreEngine()
    motor.initialize()
    kasa = motor.knowledge
    _p(f"  kasa hazır      : {kasa is not None}")

    # ── Öğretmeni kur (gerçek API) ──
    #
    # ⚠ Motor bir LLM yönlendiricisi **taşımaz** (mimari gereği ajan katmanı
    # onu kurar). Bu yüzden ölçüm aracı kendi yönlendiricisini kurar;
    # aksi hâlde `_teacher_name()` "yok" döner ve öğrenme hiç denenmez.
    yonlendirici = LLMRouter()
    yonlendirici.initialize(data_path=str(Path.cwd() / ".neurocore"))
    sohbet = SelfStudy()
    sohbet.initialize(llm=yonlendirici, knowledge=kasa,
                      recorder=None, workspace=str(Path.cwd()))
    ogretmen = sohbet._teacher_name()
    _p(f"  öğretmen        : {ogretmen}")
    if not ogretmen or ogretmen == "yok":
        _p("  UYARI: öğretmen yok — öğrenme yapılamaz (API anahtarı gerekli)")

    akis = KnowledgeWorkflow()
    akis.initialize(study=sohbet, knowledge=kasa)

    rapor: Dict[str, Any] = {"ogretmen": ogretmen, "konular": []}

    for konu in konular:
        _bolum(f"KONU: {konu[:64]}")

        # ── 1) Kasa durumu (öncesi) ──
        onceki = akis._search_vault(konu, limit=5)
        _p(f"  1) Kasa (önce)  : {len(onceki)} sonuç")
        _p(f"     → {'biliyor' if len(onceki) >= 2 else 'BİLMİYOR'}")

        # ── 2) Öğren ──
        _p("  2) Öğreniliyor (öğretmen + kaynak indirme)...")
        t0 = time.perf_counter()
        kayit = akis.learn(konu)
        sure = time.perf_counter() - t0
        _p(f"     süre          : {sure:.1f} sn")
        _p(f"     başarılı      : {kayit.ok}")
        _p(f"     soruyu üreten : {kayit.question_source or '—'}")
        if kayit.question:
            _p(f"     soru          : {kayit.question[:90]}")
        if kayit.answer:
            _p(f"     cevap         : {kayit.answer[:200]}")
        if kayit.sources:
            _p(f"     kaynaklar     : {len(kayit.sources)}")
            for url in kayit.sources[:3]:
                _p(f"       - {url[:80]}")
        if kayit.error:
            _p(f"     HATA          : {kayit.error[:120]}")
        if kayit.note_title:
            _p(f"     kasa notu     : {kayit.note_title[:70]}")

        # ── 3) Kasa durumu (sonrası) ──
        sonraki = akis._search_vault(konu, limit=5)
        _p(f"  3) Kasa (sonra) : {len(sonraki)} sonuç")
        yazildi = len(sonraki) > len(onceki)
        _p(f"     → {'KASAYA YAZILDI' if yazildi else 'yazılmadı'}")

        # ── 4) İkinci soruş: kasadan mı geliyor? ──
        ikinci_kaynak = ""
        if args.ikinci or kayit.ok:
            yanit = akis.ask(konu, allow_learn=False)
            ikinci_kaynak = yanit["source"]
            _p(f"  4) Tekrar soruş : kaynak={ikinci_kaynak} "
               f"({'kasadan geldi' if ikinci_kaynak == 'vault' else 'kasada yok'})")

        rapor["konular"].append({
            "konu": konu[:120],
            "kasa_once": len(onceki), "kasa_sonra": len(sonraki),
            "ogrenildi": kayit.ok, "sure_s": round(sure, 1),
            "kaynak_sayisi": len(kayit.sources),
            "soru_kaynagi": kayit.question_source,
            "ikinci_kaynak": ikinci_kaynak,
            "hata": kayit.error[:200],
        })

    # ── Özet ──
    _bolum("ÖZET")
    basarili = [k for k in rapor["konular"] if k["ogrenildi"]]
    _p(f"  öğrenilen konu   : {len(basarili)}/{len(rapor['konular'])}")
    if basarili:
        toplam_kaynak = sum(k["kaynak_sayisi"] for k in basarili)
        _p(f"  toplam kaynak    : {toplam_kaynak}")
        ort = sum(k["sure_s"] for k in basarili) / len(basarili)
        _p(f"  ortalama süre    : {ort:.1f} sn/konu")
        yerel = sum(1 for k in basarili if k["soru_kaynagi"] == "local")
        _p(f"  yerel model soru : {yerel}/{len(basarili)}")
        kasada = sum(1 for k in basarili if k["ikinci_kaynak"] == "vault")
        _p(f"  tekrar kasadan   : {kasada}/{len(basarili)}")
    _p(f"  sayaçlar         : {json.dumps(akis.get_stats(), ensure_ascii=False)[:200]}")
    _p("=" * 74)

    Path("_ogrenme_raporu.json").write_text(
        json.dumps(rapor, ensure_ascii=False, indent=1), encoding="utf-8")
    _p("  rapor: _ogrenme_raporu.json")
    motor.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
