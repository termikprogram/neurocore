"""
Taze Süreçli Eğitim — Her Tur Ayrı Python Süreci
=================================================
Eğitim turlarını **ayrı süreçlerde** çalıştırır; böylece her tur temiz VRAM
ile başlar ve tam hızda koşar.

Neden gerekli? Ölçüldü: aynı süreçte turlar tekrarlandığında eğitim
VRAM'i bırakmıyor ve hız **13× düşüyor**:

    tur 1 (temiz süreç) : 13,92 adım/dk
    tur 2 (VRAM dolu)   :  2,24 adım/dk
    tur 3               : 10,26 adım/dk
    tur 4+              :  1,05 → 0,87 adım/dk

Süreç başına bir tur çalıştırmak CUDA bağlamını sıfırlar; hız korunur.
Model her tur sonunda diske yazıldığı için **kayıp olmaz**.

Kullanım:
    python tools/taze_egitim.py --tur 12 --adim 150
    python tools/taze_egitim.py --durum
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Tur başına çalıştırılan alt betik (kendini çağırır)
SINGLE_FLAG = "--tek-tur"

#: İlerleme dosyası
PROGRESS = ".neurocore/taze_egitim.json"


def _p(*parts: Any) -> None:
    """Tamponsuz yazdırır."""
    print(time.strftime("%H:%M:%S"), "|", " ".join(str(x) for x in parts),
          flush=True)


def tek_tur(adim: int, dakika: float) -> int:
    """
    **Tek bir eğitim turunu** kendi sürecinde çalıştırır.

    Sonucu JSON olarak stdout'a yazar (üst süreç okur).
    """
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    from neurocore import NeuroCoreEngine

    motor = NeuroCoreEngine()
    motor.initialize()
    model = motor.model
    oncesi = int(model.growth.total_training_steps)

    # ── Iraksama koruması: tur öncesi ağırlıkları sakla ──
    #
    # ⚠ Ölçüldü: ezberlenmiş veride eğitim kaybı **artırabiliyor**
    # (0,39 → 0,56). Böyle bir turun ağırlıklarını kaydetmek modeli
    # bozar. Bu yüzden tur öncesi durum saklanır; kayıp belirgin artarsa
    # **geri alınır** ve tur başarısız sayılır.
    import shutil
    import tempfile

    yedek = Path(tempfile.mkdtemp(prefix="nc_tur_")) / "model_yedek"
    geri_alindi = False
    try:
        shutil.copytree(model._model_dir, yedek)
    except Exception:
        yedek = None

    t0 = time.perf_counter()
    sonuc = motor.train(steps=adim, from_vault=True, from_experiences=True,
                        max_minutes=dakika, user_initiated=True)
    sure = time.perf_counter() - t0

    # ── Kayıp arttıysa geri al ──
    ilk = sonuc.get("initial_loss")
    son = sonuc.get("final_loss")
    if ilk is not None and son is not None and float(ilk) > 0:
        artis = (float(son) - float(ilk)) / float(ilk)
        # Eşik %25: toplu (batch) gürültüsü %10-20 arası dalgalanma yaratır;
        # çok sıkı eşik **gerçek ilerlemeyi** de geri alırdı (ölçüldü:
        # %10 eşiği iyileşen turları kesiyordu).
        if artis > 0.25 and yedek is not None and Path(yedek).exists():
            try:
                shutil.rmtree(model._model_dir)
                shutil.copytree(yedek, model._model_dir)
                geri_alindi = True
                sonuc["stop_reason"] = "iraksama_geri_alindi"
            except Exception:
                pass
    if yedek is not None:
        shutil.rmtree(Path(yedek).parent, ignore_errors=True)

    if not geri_alindi:
        model.save()
    sonrasi = int(model.growth.total_training_steps)
    motor.shutdown()

    cikti = {
        "steps_requested": int(sonuc.get("steps_requested") or adim),
        "steps_completed": int(sonuc.get("steps_completed") or 0),
        "initial_loss": sonuc.get("initial_loss"),
        "final_loss": sonuc.get("final_loss"),
        "tokens_seen": sonuc.get("tokens_seen"),
        "stop_reason": sonuc.get("stop_reason"),
        "model_steps": sonrasi,
        "model_steps_before": oncesi,
        "duration_s": round(sure, 1),
        "rolled_back": geri_alindi,
    }
    # Yalnızca JSON satırı yaz (üst süreç ayrıştırır)
    print("NC_RESULT " + json.dumps(cikti, ensure_ascii=False), flush=True)
    return 0


def durum_yaz(gecmis: List[Dict[str, Any]], toplam_sure: float) -> None:
    """İlerlemeyi diske yazar (kesintiye dayanıklı)."""
    yol = ROOT / PROGRESS
    try:
        yol.parent.mkdir(parents=True, exist_ok=True)
        yol.write_text(json.dumps({
            "gecmis": gecmis,
            "toplam_sure_s": round(toplam_sure, 1),
            "guncelleme": time.strftime("%H:%M:%S"),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as exc:
        _p(f"ilerleme yazılamadı: {exc}")


def main() -> int:
    """Turları ayrı süreçlerde çalıştırır."""
    parser = argparse.ArgumentParser(description="Taze süreçli eğitim")
    parser.add_argument("--tur", type=int, default=12, help="Tur sayısı")
    parser.add_argument("--adim", type=int, default=150, help="Tur başına adım")
    parser.add_argument("--dakika", type=float, default=18.0,
                        help="Tur başına süre sınırı")
    parser.add_argument("--durum", action="store_true", help="Durumu yazdır")
    parser.add_argument(SINGLE_FLAG, action="store_true",
                        help="(iç kullanım) tek tur çalıştır")
    args = parser.parse_args()

    if args.durum:
        yol = ROOT / PROGRESS
        if not yol.exists():
            print("İlerleme yok — henüz eğitim başlatılmadı.")
            return 1
        veri = json.loads(yol.read_text(encoding="utf-8"))
        print("=" * 70)
        print("  TAZE SÜREÇLİ EĞİTİM — DURUM")
        print("=" * 70)
        for kayit in veri.get("gecmis", []):
            adim = kayit.get("steps_completed") or 0
            sure = kayit.get("duration_s") or 0
            hiz = adim / (sure / 60) if sure else 0
            print(f"  tur {kayit.get('tur'):2}: {adim:3}/{kayit.get('adim')} adım | "
                  f"{sure:6.1f}s | {hiz:5.2f} adım/dk | {kayit.get('neden')}")
        print(f"  toplam: {veri.get('toplam_sure_s', 0)/60:.1f} dk")
        print("=" * 70)
        return 0

    if args.tek_tur:
        return tek_tur(args.adim, args.dakika)

    # ── Üst süreç: her turu ayrı süreçte çalıştır ──
    gecmis: List[Dict[str, Any]] = []
    basladi = time.time()
    _p("=" * 70)
    _p(f"  TAZE SÜREÇLİ EĞİTİM — {args.tur} tur × {args.adim} adım")
    _p("  (her tur ayrı Python süreci: temiz VRAM, tam hız)")
    _p("=" * 70)

    for index in range(1, args.tur + 1):
        komut = [sys.executable, "-u", str(Path(__file__).resolve()),
                 SINGLE_FLAG, "--adim", str(args.adim),
                 "--dakika", str(args.dakika)]
        t0 = time.time()
        try:
            surec = subprocess.run(komut, cwd=str(ROOT), capture_output=True,
                                   text=True, encoding="utf-8",
                                   errors="replace", timeout=args.dakika * 60 + 600)
        except subprocess.TimeoutExpired:
            _p(f"TUR {index}: ZAMAN AŞIMI (alt süreç yanıt vermedi)")
            gecmis.append({"tur": index, "ok": False, "neden": "zaman_asimi"})
            durum_yaz(gecmis, time.time() - basladi)
            continue
        sure = time.time() - t0

        # Alt süreçten JSON sonucu ayıkla
        veri: Dict[str, Any] = {}
        for satir in (surec.stdout or "").splitlines():
            if satir.startswith("NC_RESULT "):
                try:
                    veri = json.loads(satir[len("NC_RESULT "):])
                except json.JSONDecodeError:
                    pass
        if not veri:
            _p(f"TUR {index}: sonuç okunamadı (çıkış kodu {surec.returncode})")
            hata = (surec.stderr or "")[-300:]
            gecmis.append({"tur": index, "ok": False, "neden": "sonuc_yok",
                           "cikis": surec.returncode, "stderr": hata})
            durum_yaz(gecmis, time.time() - basladi)
            continue

        adim = veri.get("steps_completed") or 0
        hiz = adim / (sure / 60) if sure else 0
        kayit = {
            "tur": index, "adim": args.adim, "ok": adim > 0,
            "steps_completed": adim,
            "kayip_ilk": veri.get("initial_loss"),
            "kayip_son": veri.get("final_loss"),
            "token": veri.get("tokens_seen"),
            "neden": veri.get("stop_reason"),
            "sure_s": round(sure, 1), "hiz_adim_dk": round(hiz, 2),
            "model_adim": veri.get("model_steps"),
        }
        gecmis.append(kayit)
        _p(f"TUR {index:2}/{args.tur}: {adim:3} adım | {sure:6.1f}s | "
           f"{hiz:5.2f} adım/dk | kayıp {veri.get('initial_loss')} → "
           f"{veri.get('final_loss')} | model {veri.get('model_steps')}")
        durum_yaz(gecmis, time.time() - basladi)

    # ── Özet ──
    _p("")
    _p("=" * 70)
    _p("  ÖZET")
    _p("=" * 70)
    basarili = [k for k in gecmis if k.get("ok")]
    if basarili:
        toplam_adim = sum(k["steps_completed"] for k in basarili)
        toplam_sure = sum(k["sure_s"] for k in basarili)
        _p(f"  tamamlanan tur : {len(basarili)}/{args.tur}")
        _p(f"  toplam adım    : {toplam_adim}")
        _p(f"  toplam süre    : {toplam_sure/60:.1f} dk")
        _p(f"  ORTALAMA HIZ   : {toplam_adim/(toplam_sure/60):.2f} adım/dk")
        _p(f"  son model adımı: {basarili[-1].get('model_adim')}")
        hizlar = [k["hiz_adim_dk"] for k in basarili]
        _p(f"  hız aralığı    : {min(hizlar):.2f} – {max(hizlar):.2f} adım/dk")
    _p(f"  toplam süre    : {(time.time()-basladi)/60:.1f} dk")
    _p("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
