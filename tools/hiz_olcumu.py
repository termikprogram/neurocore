"""
Hız Ölçümü (Benchmark)
=======================
Sistemin **gerçek hızını** ölçer: parçalama, paralellik, kaynak bütçesi,
kuyruk davranışı ve eğitim hızı.

Neden gerekli?
    "Hızımızı görelim" sorusunun cevabı tahminle verilmez. Bu betik beş
    ölçüm yapar ve **ölçülen** sayıları raporlar:

        1. **Bütçe**      — sisteme göre kaç parça paralel?
        2. **Paralellik** — bağımsız iş gerçekten hızlanıyor mu?
        3. **Kuyruk**     — kaynak yetmezse ne oluyor?
        4. **Günlük**     — kayıt ve devam edebilme çalışıyor mu?
        5. **Eğitim**     — adım/dakika (gerçek model varsa)

Kullanım:
    python tools/hiz_olcumu.py            # tüm ölçümler
    python tools/hiz_olcumu.py --hizli    # kısa sürüm
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def _p(*parts: Any) -> None:
    """Tamponsuz yazdırır."""
    print(" ".join(str(part) for part in parts), flush=True)


def _baslik(metin: str) -> None:
    """Bölüm başlığı."""
    _p("")
    _p("-" * 74)
    _p(f"  {metin}")
    _p("-" * 74)


#: Ölçülecek sistem profilleri
PROFILLER: Dict[str, Dict[str, Any]] = {
    "KÜÇÜK  (4 GB, %10 dolu)": dict(
        cpu_percent=10, ram_percent=10, vram_percent=10,
        ram_total_mb=4096, vram_total_mb=4096, cpu_cores=4),
    "ORTA   (32 GB, %50 dolu)": dict(
        cpu_percent=50, ram_percent=50, vram_percent=50,
        ram_total_mb=32768, vram_total_mb=16384, cpu_cores=16),
    "BÜYÜK  (32 GB, %10 dolu)": dict(
        cpu_percent=10, ram_percent=20, vram_percent=10,
        ram_total_mb=32768, vram_total_mb=11264, cpu_cores=24),
    "YOĞUN  (4 GB, %85 dolu)": dict(
        cpu_percent=85, ram_percent=85, vram_percent=85,
        ram_total_mb=4096, vram_total_mb=4096, cpu_cores=4),
}


def olcum_butce() -> Dict[str, Any]:
    """1) Bütçe: sisteme göre kaç parça paralel?"""
    from neurocore.core.governor import ResourceGovernor, ResourceSnapshot

    _baslik("1) KAYNAK BÜTÇESİ — SİSTEME GÖRE OTOMATİK UYUM")
    gov = ResourceGovernor()
    _p(f"  {'Sistem':28} {'Bütçe':>8} {'Parça RAM':>11} {'Paralel':>8} {'Kısıt':>7}")
    sonuc: Dict[str, Any] = {}
    for ad, alanlar in PROFILLER.items():
        snap = ResourceSnapshot(**alanlar)
        butce = gov.plan(snap)
        _p(f"  {ad:28} %{gov.spendable_percent(snap.ram_percent):5.1f} "
           f"{butce.ram_mb:8.0f} MB {butce.parallel:8} {butce.limiting:>7}")
        sonuc[ad] = butce.to_dict()
    return sonuc


def olcum_paralellik(*, adetler: List[int], gecikme: float = 0.2) -> Dict[str, Any]:
    """2) Paralellik: bağımsız iş hızlanıyor mu?"""
    from neurocore.core.governor import ResourceSnapshot
    from neurocore.core.work import WorkEngine, WorkStep

    _baslik("2) PARALELLİK — BAĞIMSIZ İŞ HIZLANIYOR MU?")
    snap = ResourceSnapshot(**PROFILLER["BÜYÜK  (32 GB, %10 dolu)"])
    _p(f"  Ölçüm: {gecikme:.2f} sn bekleme/iş (G/Ç benzeri), büyük sistem")
    _p(f"  {'Adım':>6} {'Parça':>7} {'Süre':>9} {'Seri olsaydı':>13} {'Hızlanma':>10}")
    sonuc: Dict[str, Any] = {}
    for adet in adetler:
        kok = tempfile.mkdtemp(prefix="nc_bm_")
        try:
            motor = WorkEngine(workspace=kok)
            adimlar = [WorkStep(id=f"j{i}", title=f"İş {i}", kind="io",
                                needs={"ram_mb": 20.0}) for i in range(adet)]
            motor.plan("benchmark", adimlar, snapshot=snap)

            def isle(adim: WorkStep) -> str:
                time.sleep(gecikme)
                return f"ok:{adim.id}"

            motor.register("io", isle)
            t0 = time.perf_counter()
            rapor = motor.run(snapshot=snap)
            sure = time.perf_counter() - t0
            seri = adet * gecikme
            hizlanma = seri / sure if sure > 0 else 0.0
            _p(f"  {adet:6} {len(motor.chunks):7} {sure:8.2f}s {seri:12.1f}s "
               f"{hizlanma:9.1f}x")
            sonuc[str(adet)] = {
                "chunks": len(motor.chunks), "duration_s": round(sure, 3),
                "serial_s": round(seri, 3), "speedup": round(hizlanma, 2),
                "steps_done": rapor["steps_done"],
            }
        finally:
            shutil.rmtree(kok, ignore_errors=True)
    return sonuc


def olcum_kuyruk() -> Dict[str, Any]:
    """3) Kuyruk: kaynak yetmezse ne oluyor?"""
    from neurocore.core.governor import ResourceSnapshot
    from neurocore.core.work import WorkEngine, WorkStep

    _baslik("3) KUYRUK — KAYNAK YETMEZSE")
    yogun = ResourceSnapshot(**PROFILLER["YOĞUN  (4 GB, %85 dolu)"])
    kok = tempfile.mkdtemp(prefix="nc_bq_")
    sonuc: Dict[str, Any] = {}
    try:
        motor = WorkEngine(workspace=kok)
        motor.plan("karma iş", [
            WorkStep(id="hafif", title="Hafif iş (20 MB)", kind="k",
                     needs={"ram_mb": 20.0}),
            WorkStep(id="orta", title="Orta iş (150 MB)", kind="k",
                     needs={"ram_mb": 150.0}),
            WorkStep(id="agir", title="Ağır iş (2 GB)", kind="k",
                     needs={"ram_mb": 2000.0}),
        ], snapshot=yogun)
        cagri: List[str] = []
        motor.register("k", lambda a: cagri.append(a.id) or "ok")
        butce = motor.governor.plan(yogun)
        _p(f"  Bütçe        : {butce.parallel} paralel, parça RAM {butce.ram_mb:.0f} MB")
        t0 = time.perf_counter()
        rapor = motor.run_until_done(snapshot=yogun, max_wait_s=1.5, poll_s=0.5)
        sure = time.perf_counter() - t0
        _p(f"  Çalışan      : {cagri} ({len(cagri)} adım)")
        _p(f"  Kuyrukta     : {len(rapor['still_blocked'])} parça")
        for x in rapor["still_blocked"]:
            _p(f"    {x['id']}: {x['reason'][:66]}")
        _p(f"  Süre         : {sure:.2f}s (bekleme {rapor['waited_s']}s, "
           f"{rapor['rounds']} tur)")
        _p(f"  Ağır iş çalıştı mı: {'EVET (HATA!)' if 'agir' in cagri else 'hayır (doğru)'}")
        sonuc = {
            "budget_parallel": butce.parallel,
            "budget_ram_mb": round(butce.ram_mb, 1),
            "ran": cagri,
            "deferred": len(rapor["still_blocked"]),
            "waited_s": rapor["waited_s"],
            "heavy_deferred": "agir" not in cagri,
        }
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    return sonuc


def olcum_gunluk() -> Dict[str, Any]:
    """4) Günlük: kayıt ve devam edebilme."""
    from neurocore.core.governor import ResourceSnapshot
    from neurocore.core.work import JOURNAL_FILE, WorkEngine, WorkJournal, WorkStep

    _baslik("4) GÜNLÜK — KAYIT VE DEVAM EDEBİLME")
    snap = ResourceSnapshot(**PROFILLER["BÜYÜK  (32 GB, %10 dolu)"])
    kok = Path(tempfile.mkdtemp(prefix="nc_bj_"))
    sonuc: Dict[str, Any] = {}
    try:
        motor = WorkEngine(workspace=str(kok))
        motor.plan("günlük testi",
                   [WorkStep(id=f"g{i}", title=f"Adım {i}", kind="k",
                             needs={"ram_mb": 10.0}) for i in range(20)],
                   snapshot=snap)
        motor.register("k", lambda a: "ok")
        t0 = time.perf_counter()
        rapor = motor.run(snapshot=snap)
        yazma_s = time.perf_counter() - t0
        yol = kok / JOURNAL_FILE
        boyut = yol.stat().st_size if yol.exists() else 0
        kayitlar = WorkJournal(yol).load()
        _p(f"  20 adım yazıldı : {yazma_s:.3f}s")
        _p(f"  Günlük boyutu   : {boyut:,} bayt")
        _p(f"  Günlük kaydı    : {len(kayitlar)} satır")

        t0 = time.perf_counter()
        yeni = WorkEngine(workspace=str(kok))
        durum = yeni.resume_state()
        okuma_s = time.perf_counter() - t0
        _p(f"  Devam okuması   : {okuma_s:.3f}s | "
           f"{durum['completed_count']} adım tamamlanmış")
        sonuc = {
            "steps": rapor["steps"], "journal_bytes": boyut,
            "journal_records": len(kayitlar),
            "write_s": round(yazma_s, 3), "resume_read_s": round(okuma_s, 4),
            "resumed_completed": durum["completed_count"],
        }
    finally:
        shutil.rmtree(kok, ignore_errors=True)
    return sonuc


def olcum_egitim(*, adim: int = 30, dakika: float = 6.0) -> Dict[str, Any]:
    """
    5) Eğitim hızı — **gerçek** model varsa.

    Model/GPU yoksa atlanır ve bu **açıkça** bildirilir (uydurma yok).
    """
    from neurocore.neural.training_work import TrainingWorkflow

    _baslik("5) EĞİTİM HIZI — GERÇEK MODEL")
    try:
        from neurocore import NeuroCoreEngine

        motor = NeuroCoreEngine()
        motor.initialize()
    except Exception as exc:
        _p(f"  ATLANDI: motor yüklenemedi ({exc})")
        return {"skipped": True, "reason": str(exc)[:120]}

    try:
        model = motor.model
        baslangic = int(model.growth.total_training_steps)
        _p(f"  Başlangıç adımı : {baslangic}")
        akis = TrainingWorkflow(workspace=".neurocore")
        akis.initialize(
            train_fn=motor.train,
            model_steps_fn=lambda: int(model.growth.total_training_steps))
        akis.plan_rounds(rounds=1, steps_per_round=adim,
                         max_minutes=dakika, objective="hız ölçümü")
        t0 = time.perf_counter()
        rapor = akis.run(max_wait_s=120.0, poll_s=10.0)
        sure = time.perf_counter() - t0
        hiz = akis.speed_report()
        _p(f"  Süre            : {sure:.1f}s")
        _p(f"  Adım            : {hiz['steps_total']}")
        _p(f"  ADIM/DAKİKA     : {hiz['steps_per_minute']:.2f}")
        _p(f"  Saniye/adım     : {hiz['seconds_per_step']:.3f}")
        _p(f"  Token           : {hiz['tokens_total']:,}")
        _p(f"  Kayıp           : {hiz['loss_first']} → {hiz['loss_last']}")
        _p(f"  Model adımı     : {hiz['model_steps']}")
        return hiz
    except Exception as exc:
        _p(f"  HATA: {type(exc).__name__}: {str(exc)[:160]}")
        return {"error": str(exc)[:200]}
    finally:
        try:
            motor.shutdown()
        except Exception:
            pass


def main() -> int:
    """Tüm ölçümleri çalıştırır."""
    parser = argparse.ArgumentParser(description="Hız ölçümü")
    parser.add_argument("--hizli", action="store_true", help="Kısa sürüm")
    parser.add_argument("--egitim", action="store_true",
                        help="Eğitim hızını da ölç (yavaş, GPU kullanır)")
    parser.add_argument("--adim", type=int, default=30,
                        help="Eğitim ölçümü adım sayısı")
    args = parser.parse_args()

    _p("=" * 74)
    _p("  HIZ ÖLÇÜMÜ — PARÇALI İŞ MOTORU")
    _p("=" * 74)
    basladi = time.perf_counter()
    rapor: Dict[str, Any] = {}

    rapor["butce"] = olcum_butce()
    adetler = [8, 16] if args.hizli else [8, 16, 24, 32]
    rapor["paralellik"] = olcum_paralellik(adetler=adetler)
    rapor["kuyruk"] = olcum_kuyruk()
    rapor["gunluk"] = olcum_gunluk()
    if args.egitim:
        rapor["egitim"] = olcum_egitim(adim=args.adim)

    _baslik("ÖZET")
    par = rapor["paralellik"]
    if par:
        en_iyi = max(par.items(), key=lambda x: x[1]["speedup"])
        _p(f"  En iyi hızlanma : {en_iyi[1]['speedup']:.1f}x "
           f"({en_iyi[0]} iş, {en_iyi[1]['chunks']} parça)")
    kuy = rapor["kuyruk"]
    _p(f"  Kuyruk koruması : "
       f"{'çalışıyor' if kuy.get('heavy_deferred') else 'BAŞARISIZ'}")
    gun = rapor["gunluk"]
    _p(f"  Günlük kaydı    : {gun.get('journal_records')} satır, "
       f"{gun.get('journal_bytes', 0):,} bayt")
    _p(f"  Devam okuması   : {gun.get('resume_read_s')}s")
    if rapor.get("egitim"):
        eg = rapor["egitim"]
        if "steps_per_minute" in eg:
            _p(f"  Eğitim hızı     : {eg['steps_per_minute']:.2f} adım/dakika")
    _p(f"  Toplam süre     : {time.perf_counter() - basladi:.1f}s")
    _p("=" * 74)

    Path("_hiz_raporu.json").write_text(
        json.dumps(rapor, ensure_ascii=False, indent=1), encoding="utf-8")
    _p("  rapor: _hiz_raporu.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
