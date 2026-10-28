"""
Müfredat Eğitimi — Bağımsız Çalıştırıcı
========================================
Yerel modeli müfredatla eğitir; arka planda uzun süre çalışabilir.

Kullanım:
    python tools/mufredat_egit.py                    # hazırlık + 8 tur
    python tools/mufredat_egit.py --tur 12 --dakika 8
    python tools/mufredat_egit.py --durum            # ilerlemeyi yazdır

İlerleme: `.neurocore/curriculum_progress.json`
Günlük  : `.neurocore/logs/neurocore.log`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Proje kökünü yola ekle (betik tools/ içinden çalıştırılır)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def durum(veri: str) -> int:
    """İlerleme dosyasını yazdırır."""
    yol = Path(veri) / "curriculum_progress.json"
    if not yol.exists():
        print("İlerleme dosyası yok — henüz eğitim başlatılmadı.")
        return 1
    data = json.loads(yol.read_text(encoding="utf-8"))
    print("=" * 62)
    print("  MÜFREDAT EĞİTİMİ — DURUM")
    print("=" * 62)
    for key, label in (
        ("stage", "Aşama"), ("lessons", "Ders"), ("vocab_before", "Sözlük (önce)"),
        ("vocab_after", "Sözlük (sonra)"), ("tokens", "Token"),
        ("rounds", "Tamamlanan tur"), ("steps_done", "Bu oturumda adım"),
        ("loss_first", "Kayıp (ilk)"), ("loss_last", "Kayıp (son)"),
        ("total_model_steps", "Model toplam adım"),
        ("elapsed_seconds", "Geçen süre (sn)"), ("finished", "Bitti"),
    ):
        if key in data:
            print(f"  {label:22}: {data[key]}")
    if data.get("error"):
        print(f"  HATA                  : {data['error']}")
    for note in (data.get("notes") or [])[-4:]:
        print(f"  not: {note}")
    print("=" * 62)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Müfredat eğitim hattı")
    parser.add_argument("--tur", type=int, default=8, help="Eğitim turu sayısı")
    parser.add_argument("--dakika", type=float, default=6.0,
                        help="Tur başına süre sınırı (dakika)")
    parser.add_argument("--sozluk", type=int, default=8000, help="Hedef sözlük")
    parser.add_argument("--kalip", type=int, default=8, help="Ders başına soru kalıbı")
    parser.add_argument("--adim", type=int, default=0,
                        help="Tur başına adım (0 = otomatik plan; otomatik plan "
                             "küçük veri kümesinde 10'a yuvarlanır)")
    parser.add_argument("--veri", default=str(ROOT / ".neurocore"),
                        help="Veri dizini")
    parser.add_argument("--durum", action="store_true", help="Yalnızca durumu yazdır")
    parser.add_argument("--hazirlik-atla", action="store_true",
                        help="Tokenizer adımını atla")
    args = parser.parse_args()

    if args.durum:
        return durum(args.veri)

    from neurocore.neural.pipeline import CurriculumTrainer

    trainer = CurriculumTrainer(patterns=args.kalip, vocab_size=args.sozluk)
    trainer.initialize(workspace=str(ROOT), data_path=args.veri)
    basladi = time.time()
    try:
        ozet = trainer.run_rounds(
            rounds=args.tur, max_minutes=args.dakika,
            steps_per_round=(args.adim or None),
            skip_prepare=args.hazirlik_atla,
        )
    except KeyboardInterrupt:
        print("\nKullanıcı durdurdu.")
        return 130
    finally:
        trainer.shutdown()

    print("=" * 62)
    print("  MÜFREDAT EĞİTİMİ BİTTİ")
    print("=" * 62)
    for key, label in (
        ("ok", "Başarılı"), ("rounds_ok", "Başarılı tur"),
        ("lessons", "Ders"), ("vault_notes", "Kasaya yazılan"),
        ("steps_total", "Bu oturumda adım"), ("loss_first", "Kayıp (ilk)"),
        ("loss_last", "Kayıp (son)"), ("model_steps", "Model toplam adım"),
        ("saved", "Kaydedildi"),
    ):
        if key in ozet:
            print(f"  {label:22}: {ozet[key]}")
    ilerleme = ozet.get("progress") or {}
    if ilerleme.get("vocab_after"):
        print(f"  {'Sözlük':22}: {ilerleme.get('vocab_before')} → "
              f"{ilerleme.get('vocab_after')}")
    if ilerleme.get("error"):
        print(f"  {'HATA':22}: {ilerleme['error'][:120]}")
    print(f"  {'Süre':22}: {time.time() - basladi:.0f} sn")
    print("=" * 62)
    return 0 if ozet.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
