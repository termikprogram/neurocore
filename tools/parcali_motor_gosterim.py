"""
Parçalı İş Motoru — Gösterim
=============================
Kullanıcının senaryosunu baştan sona çalıştırır ve ölçülen sonuçları yazar.

Senaryo: "8.000.000 adımlı bir proje verildi. GPU, CPU ve RAM'de kullanacağı
alana göre işleri parçalara bölecek, fazladan bölemediği adımları sıraya
alacak. Nerede kaldığını, ne yaptığını kayıt altına alacak."

Bu betik üç sistemi karşılaştırır:

    * KÜÇÜK  : 4 GB VRAM, 4 GB RAM, 4 çekirdek, %10 dolu
    * ORTA   : 16 GB VRAM, 32 GB RAM, 16 çekirdek, %50 dolu
    * YOĞUN  : 4 GB VRAM, 4 GB RAM, 4 çekirdek, %85 dolu

Kullanım:
    python tools/parcali_motor_gosterim.py
"""

from __future__ import annotations

import logging
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


#: Karşılaştırılacak sistem profilleri
PROFILLER = {
    "KÜÇÜK (4 GB, %10 dolu)": dict(
        cpu_percent=10, ram_percent=10, vram_percent=10,
        ram_total_mb=4096, vram_total_mb=4096, cpu_cores=4),
    "ORTA (32 GB, %50 dolu)": dict(
        cpu_percent=50, ram_percent=50, vram_percent=50,
        ram_total_mb=32768, vram_total_mb=16384, cpu_cores=16),
    "YOĞUN (4 GB, %85 dolu)": dict(
        cpu_percent=85, ram_percent=85, vram_percent=85,
        ram_total_mb=4096, vram_total_mb=4096, cpu_cores=4),
}


def main() -> int:
    """Gösterimi çalıştırır."""
    from neurocore.core.governor import ResourceGovernor, ResourceSnapshot
    from neurocore.core.work import JOURNAL_FILE, WorkEngine, WorkStep, WorkJournal

    _p("=" * 76)
    _p("  PARÇALI İŞ MOTORU — 8.000.000 ADIM SENARYOSU")
    _p("=" * 76)

    kok = Path(tempfile.mkdtemp(prefix="nc_demo_"))
    # 8M adımı temsil eden plan: 48 modül, her biri 166.667 alt adım
    TOPLAM_ADIM = 8_000_000
    MODUL = 48
    alt_adim = TOPLAM_ADIM // MODUL

    adimlar: List[WorkStep] = []
    for i in range(MODUL):
        # Bağımlılık: her 8 modülde bir zincir kırılır (paralellik doğsun)
        dep = [f"m{i-1:03d}"] if i % 8 else []
        adimlar.append(WorkStep(
            id=f"m{i:03d}", title=f"Modül {i} ({alt_adim:,} alt adım)",
            kind="code",
            needs={"cpu": 2.0, "ram_mb": 120.0},
            depends_on=dep,
            payload={"alt_adim": alt_adim},
            estimate_s=1.0,
        ))

    _p(f"  Proje        : {MODUL} modül × {alt_adim:,} alt adım")
    _p(f"  Toplam adım  : {TOPLAM_ADIM:,}")
    _p(f"  Bağımlılık   : her 8 modülde bir zincir kırılır (paralellik)")
    _p("")

    # ── 1) Bütçe karşılaştırması ──
    _p("-" * 76)
    _p("  1) KAYNAK BÜTÇESİ — SİSTEME GÖRE OTOMATİK UYUM")
    _p("-" * 76)
    gov = ResourceGovernor()
    _p(f"  {'Sistem':26} {'Bütçe':>8} {'Parça':>16} {'Paralel':>8} {'Kısıt':>6}")
    for ad, alanlar in PROFILLER.items():
        snap = ResourceSnapshot(**alanlar)
        butce = gov.plan(snap)
        _p(f"  {ad:26} %{gov.spendable_percent(snap.cpu_percent):5.1f} "
           f"{butce.ram_mb:9.0f} MB {butce.parallel:8} {butce.limiting:>6}")
    _p("")

    # ── 2) Küçük sistemde tam akış ──
    _p("-" * 76)
    _p("  2) TAM AKIŞ — KÜÇÜK SİSTEMDE (4 GB, %10 dolu)")
    _p("-" * 76)
    snap = ResourceSnapshot(**PROFILLER["KÜÇÜK (4 GB, %10 dolu)"])
    motor = WorkEngine(workspace=str(kok), chunk_ratio=0.05, chunk_steps=6)

    parcalar = motor.plan("8M adımlı proje", adimlar, snapshot=snap)
    butce = motor.governor.plan(snap)
    _p(f"  Plan         : {len(adimlar)} adım → {len(parcalar)} parça")
    _p(f"  Bütçe        : {butce.parallel} paralel, parça tavanı "
       f"CPU %{butce.cpu_percent:.1f} / RAM {butce.ram_mb:.0f} MB")
    _p(f"  Kısıtlayan   : {butce.limiting}")
    _p("")

    calisan = {"n": 0, "adim": 0}
    kilit = __import__("threading").Lock()

    def isle(adim: WorkStep) -> str:
        """Modül üretimini taklit eder (G/Ç bekler)."""
        with kilit:
            calisan["n"] += 1
            calisan["adim"] += adim.payload.get("alt_adim", 0)
        time.sleep(0.05)
        return f"{adim.payload.get('alt_adim', 0):,} alt adım üretildi"

    motor.register("code", isle)
    t0 = time.perf_counter()
    rapor = motor.run(snapshot=snap)
    sure = time.perf_counter() - t0

    _p(f"  Sonuç        : {rapor['steps_done']}/{rapor['steps']} adım bitti, "
       f"{rapor['steps_failed']} hata")
    _p(f"  Kuyrukta     : {rapor['chunks_blocked']} parça")
    _p(f"  Süre         : {sure:.2f} sn")
    _p(f"  Üretilen     : {calisan['adim']:,} alt adım")
    _p("")

    # ── 3) Günlük ve algoritma ──
    _p("-" * 76)
    _p("  3) KAYIT — NEREDE KALDI, NE YAPTI")
    _p("-" * 76)
    kayitlar = WorkJournal(kok / JOURNAL_FILE).load()
    turler: Dict[str, int] = {}
    for kayit in kayitlar:
        turler[kayit.get("olay", "?")] = turler.get(kayit.get("olay", "?"), 0) + 1
    _p(f"  Günlük kaydı : {len(kayitlar)} satır ({kok.name}/{JOURNAL_FILE})")
    for tur, adet in sorted(turler.items()):
        _p(f"    {tur:12} {adet}")
    _p("")

    yeni = WorkEngine(workspace=str(kok))
    durum = yeni.resume_state()
    _p(f"  Devam durumu : {durum['completed_count']} adım tamamlanmış")
    _p(f"  Amaç         : {durum['objective'][:60]}")
    _p("")

    _p("  ALGORİTMA (ilk 10 satır):")
    for satir in motor.algorithm(limit=4).splitlines():
        _p(f"    {satir}")
    _p("")

    # ── 4) Yoğun sistemde kuyruklama ──
    _p("-" * 76)
    _p("  4) YOĞUN SİSTEMDE KUYRUKLAMA (4 GB, %85 dolu)")
    _p("-" * 76)
    yogun = ResourceSnapshot(**PROFILLER["YOĞUN (4 GB, %85 dolu)"])
    agir = [
        WorkStep(id="hafif", title="Hafif iş", kind="code",
                 needs={"cpu": 1.0, "ram_mb": 20.0}),
        WorkStep(id="agir", title="Ağır iş (2 GB)", kind="code",
                 needs={"cpu": 1.0, "ram_mb": 2000.0}),
    ]
    motor2 = WorkEngine(workspace=str(kok))
    motor2.plan("yoğun", agir, snapshot=yogun)
    cagri = {"n": 0}
    motor2.register("code",
                    lambda a: cagri.__setitem__("n", cagri["n"] + 1) or "ok")
    b2 = motor2.governor.plan(yogun)
    r2 = motor2.run(snapshot=yogun)
    _p(f"  Bütçe        : {b2.parallel} paralel, RAM {b2.ram_mb:.0f} MB")
    _p(f"  Çalışan      : {cagri['n']} adım (hafif olan)")
    _p(f"  Kuyrukta     : {r2['chunks_blocked']} parça (ağır olan)")
    for parca in motor2.chunks:
        _p(f"    {parca.id} [{parca.status:16}] {parca.reason[:58]}")
    _p("")
    _p("=" * 76)

    import shutil

    shutil.rmtree(kok, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
