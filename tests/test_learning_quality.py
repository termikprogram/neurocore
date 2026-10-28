"""
Öğrenme kalite kuralları — 4 kuralın testleri
==============================================
Ölçülen sorun (2026-10-28, gerçek kasa):

```
bugün yazılan 22 notun 11'i : "Çözüm: onay veriyorum", "Çözüm: ekle",
                              "Çözüm: 1", "Çözüm: g0" … güven 0.925–0.990
kasanın %48,5'i            : kind=skill (ajanın kendi gürültüsü)
deneyim→eğitim             : 5 deneyim → 2 adım (toplamın %0,015'i)
```

Kurallar
--------
```
1. NOT SÜZGECİ : görev < 20 karakter veya tümü onay sözü → not YAZILMAZ
2. BAŞLIK      : sistem istemi ("SORU:", "Sen bir …") ayıklanır
3. GÜVEN       : tek adımlık göreve ≥0.9 verilmez; adım × başarı ile kalibre
4. EĞİTİM      : 50 deneyim / en az 20 adım · kasa: kind ∈ {reference,qa},
                 güven ≥ 0.6
```
"""

from __future__ import annotations

import tempfile
import types
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest import mock

from neurocore.agent.learning import (
    MIN_NEW_EXPERIENCES,
    MIN_TRAIN_STEPS,
    NOT_MIN_UZUNLUK,
    LearningRecorder,
)
from neurocore.neural.trainer import Trainer


# ─── Sahteler ───────────────────────────────────────────────────────────────

@dataclass
class _Adim:
    kind: str
    tool: str = ""
    ok: bool = True
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Kosu:
    task: str = ""
    answer: str = ""
    ok: bool = True
    mode: str = "tools"
    provider: str = "Gemini"
    model: str = "gemini-3.5-flash"
    tools_used: List[str] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)
    commands_run: List[str] = field(default_factory=list)
    steps: List[Any] = field(default_factory=list)
    duration_ms: float = 1200.0
    reason: str = "tamamlandı"
    error: str = ""
    cancelled: bool = False


def _kayit(task: str = "sayi.py olustur ve calistir",
           adim: int = 3, basarili: int = 3) -> Any:
    """`LearningRecord` üretir (gerçek sınıf, doldurulmuş alanlar)."""
    from neurocore.agent.learning import record_from_run

    kosu = _Kosu(
        task=task,
        answer="yapıldı",
        ok=True,
        tools_used=["code.create_file"] * adim,
        steps=[_Adim("tool", "code.create_file", i < basarili)
               for i in range(adim)],
    )
    kayit = record_from_run(kosu)
    return kayit


# ─── 1) NOT SÜZGECİ ─────────────────────────────────────────────────────────

class TestNotSuzgeci(unittest.TestCase):
    """Kısa ve onay sözü görevler kasa notu üretmemeli."""

    def test_kisa_gorev_yazilmaz(self) -> None:
        for gorev in ("ekle", "devam et", "onay veriyorum", "1", "tamam"):
            kayit = _kayit(gorev)
            yazilsin, sebep = LearningRecorder._not_yazilabilir(kayit)
            self.assertFalse(yazilsin, f"{gorev!r} not üretmemeliydi")
            self.assertTrue(sebep)

    def test_onay_sozu_uzun_olsa_da_yazilmaz(self) -> None:
        """⚠ "onaylıyorum" gibi tek sözcük uzun yazılsa da gürültüdür."""
        kayit = _kayit("onayliyorum onay veriyorum tamam")
        yazilsin, _ = LearningRecorder._not_yazilabilir(kayit)
        self.assertFalse(yazilsin)

    def test_gercek_gorev_yazilir(self) -> None:
        kayit = _kayit("sayi.py olustur ve calistir")
        yazilsin, sebep = LearningRecorder._not_yazilabilir(kayit)
        self.assertTrue(yazilsin, sebep)

    def test_kayit_sirasinda_suzgec_calisir(self) -> None:
        """`record()` not yazmamalı, sayacı artırmalı."""
        rec = LearningRecorder()
        with mock.patch.object(rec, "_write_note") as yaz:
            rec.record(_Kosu(task="onay veriyorum", ok=True,
                             tools_used=["code.read_file"],
                             steps=[_Adim("tool", "code.read_file", True)]))
            self.assertFalse(yaz.called, "not yazılmamalıydı")
        self.assertEqual(rec.get_stats()["notes_skipped"], 1)

    def test_uzun_gorevde_not_yazilir(self) -> None:
        rec = LearningRecorder()
        with mock.patch.object(rec, "_write_note") as yaz:
            rec.record(_Kosu(task="sayi.py olustur ve calistir", ok=True,
                             tools_used=["code.create_file"],
                             steps=[_Adim("tool", "code.create_file", True)]))
            self.assertTrue(yaz.called, "not yazılmalıydı")


# ─── 2) BAŞLIK ──────────────────────────────────────────────────────────────

class TestBaslik(unittest.TestCase):
    """Sistem istemi başlığa sızmamalı."""

    def test_soru_oneki_ayiklanir(self) -> None:
        oz = LearningRecorder._gorev_ozu(
            "Aşağıdaki soruyu elinden geldiğince cevapla. "
            "SORU: DXGI hatası nedir?")
        self.assertEqual(oz, "DXGI hatası nedir?")

    def test_sen_bir_oneki_ayiklanir(self) -> None:
        oz = LearningRecorder._gorev_ozu(
            "Sen bir yazılım geliştiricisisin. Kasadaki notu özetle.")
        self.assertIn("Kasadaki notu özetle", oz)
        self.assertNotIn("Sen bir yazılım", oz)

    def test_baslik_ozden_uretilir(self) -> None:
        kayit = _kayit("Aşağıdaki soruyu cevapla. SORU: godot sinyal bağla")
        baslik = LearningRecorder._note_title(kayit)
        self.assertNotIn("Aşağıdaki soruyu", baslik)
        self.assertIn("godot sinyal bağla", baslik)


# ─── 3) GÜVEN ───────────────────────────────────────────────────────────────

class TestGuven(unittest.TestCase):
    """Tek adımlık iş 'çözüm' sayılmamalı."""

    def test_tek_adim_dusuk_guven(self) -> None:
        kayit = _kayit("onay veriyorum", adim=1, basarili=1)
        kayit.reward = 0.85
        guven = LearningRecorder._not_guven(kayit)
        self.assertLessEqual(guven, 0.6, f"tek adım güveni {guven}")

    def test_cok_adim_daha_yuksek(self) -> None:
        tek = _kayit("gerçek görev metni burada", adim=1, basarili=1)
        cok = _kayit("gerçek görev metni burada", adim=10, basarili=10)
        tek.reward = cok.reward = 0.85
        self.assertGreater(LearningRecorder._not_guven(cok),
                           LearningRecorder._not_guven(tek))

    def test_basarisiz_adim_guveni_dusurur(self) -> None:
        iyi = _kayit("gerçek görev metni burada", adim=6, basarili=6)
        kotu = _kayit("gerçek görev metni burada", adim=6, basarili=1)
        iyi.reward = kotu.reward = 0.85
        self.assertGreater(LearningRecorder._not_guven(iyi),
                           LearningRecorder._not_guven(kotu))

    def test_ust_sinir(self) -> None:
        kayit = _kayit("gerçek görev", adim=20, basarili=20)
        kayit.reward = 1.0
        self.assertLessEqual(LearningRecorder._not_guven(kayit), 0.9)


# ─── 4) EĞİTİM EŞİKLERİ ─────────────────────────────────────────────────────

class TestEgitimEsikleri(unittest.TestCase):
    """50 deneyim · en az 20 adım · kasa süzgeci."""

    def test_varsayilan_deneyim_esigi(self) -> None:
        self.assertEqual(LearningRecorder().min_new_experiences,
                         MIN_NEW_EXPERIENCES)
        self.assertGreaterEqual(MIN_NEW_EXPERIENCES, 50)

    def test_en_az_adim_sabit(self) -> None:
        self.assertGreaterEqual(MIN_TRAIN_STEPS, 20)

    def test_train_now_en_az_adim_gonderir(self) -> None:
        """`steps=None` → motora en az 20 adım gitmeli (eskiden 2)."""
        rec = LearningRecorder()
        yakalanan: Dict[str, Any] = {}

        class _Motor:
            def train(self, **kw: Any) -> Dict[str, Any]:
                yakalanan.update(kw)
                return {"ok": True, "steps_completed": kw.get("steps") or 0}

        rec._engine = _Motor()                       # type: ignore[assignment]
        with mock.patch.object(rec, "_ensure_engine", return_value=(True, "")), \
                mock.patch.object(rec, "_flush_observations", return_value=None):
            sonuc = rec.train_now(force=True)
        self.assertEqual(yakalanan.get("steps"), MIN_TRAIN_STEPS,
                         f"gönderilen adım: {yakalanan.get('steps')}")
        self.assertTrue(sonuc.get("ok"))

    def test_kasa_tur_suzgeci(self) -> None:
        """Eğitim yalnız `reference`/`qa` türünü kullanmalı."""
        self.assertEqual(tuple(Trainer.EGITIM_NOTE_TURLERI),
                         ("reference", "qa"))
        self.assertGreaterEqual(Trainer.EGITIM_NOTE_MIN_GUVEN, 0.6)

    def test_kasa_suzgeci_veri_uretmez(self) -> None:
        """`skill` türü ve düşük güvenli notlar veri kümesine girmemeli."""
        from neurocore.neural.trainer import TrainingDataset

        class _Tok:
            vocab_size = 128

            def encode(self, text: str) -> List[int]:
                return [ord(c) % 128 for c in str(text)][:64] or [1]

            def decode(self, ids: List[int]) -> str:
                return "".join(chr(i) for i in ids)

        notlar = [
            types.SimpleNamespace(title="iyi", body="bu bir referans notudur ve yeterince uzundur",
                                  word_count=8, confidence=0.9, category="api_referans",
                                  tags=[], kind="reference", summary="özet"),
            types.SimpleNamespace(title="gurultu", body="onay veriyorum onay veriyorum onay",
                                  word_count=5, confidence=0.99, category="learned",
                                  tags=[], kind="skill", summary="özet"),
            types.SimpleNamespace(title="dusuk", body="düşük güvenli bir not gövdesi burada",
                                  word_count=6, confidence=0.3, category="api_referans",
                                  tags=[], kind="reference", summary="özet"),
        ]

        class _Kasa:
            def iter_notes(self, limit: int = 0) -> Any:
                return iter(notlar)

        egitici = Trainer()
        ds = egitici.build_dataset_from_vault(
            _Kasa(), _Tok(), seq_len=16, max_notes=10)
        self.assertIsInstance(ds, TrainingDataset)
        # ⚠ Yalnız 1 not (reference + güven 0.9) girmeli
        self.assertEqual(len(ds), 1, f"giren örnek: {len(ds)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
