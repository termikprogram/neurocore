"""
Doğrulamalı Eğitim — Gerçek Genelleme Takibi
=============================================
⚠ Ölçülen sorun
---------------
Eğitim 22 tur boyunca "kayıp 0,083" raporladı; bağımsız ölçüm **4,99**
çıktı. Sebep: eğitim kaybı **ezberlenmiş partide** ölçülüyordu.

`tools/model_saglik.py` bu boşluğu kapattı ama **elle** çalıştırılıyor.
Bu araç doğrulamayı **eğitimin içine** bağlar:

    Her tur sonunda:
        eğitim kaybı   (modelin gördüğü veri)
        doğrulama kaybı (modelin GÖRMEDİĞİ %20)
        açık oranı     (doğrulama / eğitim)
    Karar:
        açık artıyorsa  → EZBER BAŞLADI, dur
        doğrulama iyileşiyorsa → devam

Yöntem
------
    1. Kasayı **tohumlu** olarak %80/%20 böl (tekrarlanabilir)
    2. Doğrulama setini **eğitime sokma**
    3. Her turda doğrulama kaybını ölç
    4. En iyi doğrulama kaybındaki ağırlıkları sakla
    5. Doğrulama kaybı N tur art arda artarsa **erken dur**

Kullanım
--------
    python tools/dogrulamali_egitim.py --tur 20 --adim 150
    python tools/dogrulamali_egitim.py --tur 8 --adim 200 --sabir 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: İlerleme dosyası
ILERLEME = Path(".neurocore/dogrulamali_egitim.json")


@dataclass
class Tur:
    """Bir eğitim turunun doğrulamalı kaydı."""
    no: int = 0
    adim: int = 0
    egitim_kaybi: Optional[float] = None
    dogrulama_kaybi: Optional[float] = None
    acik: float = 0.0
    acik_orani: float = 0.0
    sure_s: float = 0.0
    geri_alindi: bool = False
    en_iyi: bool = False
    not_: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "no": self.no, "adim": self.adim,
            "egitim_kaybi": self.egitim_kaybi,
            "dogrulama_kaybi": self.dogrulama_kaybi,
            "acik": round(self.acik, 4),
            "acik_orani": round(self.acik_orani, 3),
            "sure_s": round(self.sure_s, 1),
            "geri_alindi": self.geri_alindi, "en_iyi": self.en_iyi,
            "not": self.not_[:80],
        }


class DogrulamaliEgitim:
    """Doğrulama kaybını izleyerek eğitir."""

    def __init__(self, *, tur: int = 20, adim: int = 150,
                 dogrulama_orani: float = 0.20, sabir: int = 3,
                 maks_not: int = 6000) -> None:
        self.tur_sayisi = max(1, tur)
        self.adim = max(5, adim)
        self.dogrulama_orani = max(0.05, min(0.5, dogrulama_orani))
        self.sabir = max(1, sabir)
        self.maks_not = maks_not
        self._motor: Any = None
        self._egitim: Any = None
        self._dogrulama: Any = None
        self.turlar: List[Tur] = []
        self.en_iyi_dogrulama: Optional[float] = None
        self.en_iyi_agirliklar: Any = None
        self.en_iyi_tur: int = 0

    # ─── Kurulum ───

    def baglan(self) -> bool:
        """Motoru hazırlar ve veriyi böler."""
        try:
            from neurocore import NeuroCoreEngine

            self._motor = NeuroCoreEngine()
            self._motor.initialize()
        except Exception as exc:
            print(f"  motor başlatılamadı: {exc}")
            return False
        return self._bol()

    def _bol(self) -> bool:
        """
        Kasayı eğitim/doğrulama olarak böler.

        ⚠ Doğrulama seti eğitime **sokulmaz** — aksi hâlde ölçüm ezberi
        gizler (ölçülen hata: log 0,083 / gerçek 4,99).
        """
        import random

        m = self._motor
        t = m.trainer
        notlar = list(m.knowledge.vault.iter_notes(limit=self.maks_not))
        if len(notlar) < 20:
            print(f"  yetersiz not: {len(notlar)}")
            return False
        rng = random.Random(1234)
        rng.shuffle(notlar)
        kesim = int(len(notlar) * (1.0 - self.dogrulama_orani))
        eg, dg = notlar[:kesim], notlar[kesim:]

        class _LV:
            def __init__(self, n: List[Any]) -> None:
                self._n = n

            def iter_notes(self, limit: int = 1000) -> Any:
                for x in self._n[:limit]:
                    yield x

        self._egitim = t.build_dataset_from_vault(
            _LV(eg), m.model.tokenizer, seq_len=256, max_notes=len(eg) + 1)
        self._dogrulama = t.build_dataset_from_vault(
            _LV(dg), m.model.tokenizer, seq_len=256, max_notes=len(dg) + 1)
        print(f"  veri: eğitim {len(eg)} not → {len(self._egitim)} örnek | "
              f"doğrulama {len(dg)} not → {len(self._dogrulama)} örnek")
        return len(self._egitim) > 0 and len(self._dogrulama) > 0

    # ─── Ölçüm ───

    def _kayip(self, veri: Any, adim: int = 2) -> Optional[float]:
        """
        Bir veri setinde kayıp ölçer — **eğiticiyi kullanmadan**.

        ⚠ ÖLÇÜLEN SORUN: önceki sürüm doğrulamayı `trainer.train()` ile
        yapıyordu. Her turda 6 eğitim + 2 doğrulama = **8 çağrı** oluyordu
        ve 2-3 turdan sonra CUDA bağlamı bozuluyordu:

            CUDA error: unknown error

        `torch.cuda.empty_cache()` bunu **çözmedi** (ölçüldü).

        Doğru yol: doğrulama **saf ileri geçiştir** — optimize edici,
        gradyan, kaynak gözcüsü gerekmez. `backbone` doğrudan çağrılır.
        """
        try:
            import torch

            m = self._motor.model
            cihaz = next(m.backbone.parameters()).device
            m.backbone.eval()
            toplam = 0.0
            sayi = 0
            with torch.no_grad():
                for parti in veri.batches(2, shuffle=False, seed=7):
                    ids = parti["input_ids"].to(cihaz)
                    etiket = parti["labels"].to(cihaz)
                    if etiket.shape[1] < 2:
                        continue
                    cikti = m.backbone(ids)
                    # ⚠ ÖLÇÜLEN GERÇEK: backbone bir **sözlük** döndürüyor
                    #     {"logits": Tensor(B, T, V), "aux_loss": ..., "cache": None}
                    # `getattr(cikti, "logits")` bu yüzden **None** veriyordu
                    # (ölçüldü: "TypeError: 'NoneType' object is not
                    # subscriptable"). Anahtar erişimi gerekir.
                    logit = None
                    if isinstance(cikti, dict):
                        logit = cikti.get("logits")
                    elif hasattr(cikti, "logits"):
                        logit = cikti.logits
                    elif isinstance(cikti, (tuple, list)) and cikti:
                        logit = cikti[0]
                    if logit is None:
                        return None
                    # Kaydırmalı dil modeli kaybı
                    kaydirma = logit[:, :-1, :].contiguous()
                    hedef = etiket[:, 1:].contiguous()
                    kayip = torch.nn.functional.cross_entropy(
                        kaydirma.view(-1, kaydirma.size(-1)),
                        hedef.view(-1), ignore_index=-100)
                    toplam += float(kayip)
                    sayi += 1
                    if sayi >= 12:
                        break
            m.backbone.train()
            return toplam / sayi if sayi else None
        except Exception as exc:
            import traceback as _tb

            print(f"    ölçüm hatası: {type(exc).__name__}: {exc}")
            for satir in _tb.format_exc().splitlines()[-5:]:
                print(f"      {satir}")
            try:
                self._motor.model.backbone.train()
            except Exception:
                pass
            return None

    def _cihaza_tasi(self, veri: Any) -> int:
        """
        Veri setinin **önbelleğini** modelin cihazına taşır.

        ⚠ ÖLÇÜLEN HATA: `build_dataset_from_vault` partileri CPU'da
        üretiyor; model CUDA'da olduğu için ileri geçiş
        "Expected all tensors to be on the same device" veriyor.
        Küçük veride bu açık hata, büyük veride ise asenkron
        "illegal memory access" olarak görünüyor.

        Returns:
            Taşınan örnek sayısı (0 = taşıma yolu yok, sorun değil)
        """
        try:
            import torch

            cihaz = next(self._motor.model.backbone.parameters()).device
            if str(cihaz) == "cpu":
                return 0
            tasinan = 0
            # Veri setinin iç önbelleğini bul (uygulamaya göre değişir)
            for ad in ("_cache", "_samples", "samples", "_items", "items"):
                liste = getattr(veri, ad, None)
                if liste is None:
                    continue
                if callable(liste):
                    continue
                try:
                    for i, x in enumerate(liste):
                        liste[i] = self._tas(x, cihaz, torch)
                        tasinan += 1
                except Exception:
                    continue
            return tasinan
        except Exception:
            return 0

    @staticmethod
    def _tas(x: Any, cihaz: Any, torch: Any) -> Any:
        """Tek bir örneği/partiyi cihaza taşır."""
        if isinstance(x, dict):
            return {k: (v.to(cihaz) if hasattr(v, "to") else v)
                    for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)((v.to(cihaz) if hasattr(v, "to") else v)
                           for v in x)
        if hasattr(x, "to"):
            return x.to(cihaz)
        return x

    @staticmethod
    def _cuda_temizle() -> None:
        """CUDA önbelleğini boşaltır."""
        try:
            import torch as _t

            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass

    # ─── Tur ───

    def tur_calistir(self, no: int) -> Tur:
        """Bir eğitim turu + doğrulama ölçümü."""
        import copy

        t = self._motor.trainer
        k = Tur(no=no)
        t0 = time.perf_counter()

        # 1) Eğitim
        try:
            p = t.plan_training(estimated_samples=len(self._egitim),
                                steps=self.adim, model=self._motor.model)
            # ⚠ ÖLÇÜLEN HATA: veri setinin **tamamını** GPU'ya taşımak
            # VRAM'i dolduruyor (eğitim + AdamW durumları zaten yüklü) ve
            #     "CUDA error: unknown error"
            # veriyordu. Parti bazlı taşıma yeterli — eğitici kendi
            # partilerini cihaza taşır, doğrulama `_kayip` içinde taşınır.
            s = t.train(model=self._motor.model, dataset=self._egitim,
                        plan=p, user_initiated=True, save_checkpoints=False)
            k.adim = int(s.steps_completed or 0)
            if s.initial_loss is not None:
                k.egitim_kaybi = float(s.final_loss
                                       if s.final_loss is not None
                                       else s.initial_loss)
            if not k.adim:
                k.not_ = f"eğitim çalışmadı: {s.stop_reason}"
        except Exception as exc:
            k.not_ = f"eğitim hatası: {type(exc).__name__}"
            k.sure_s = time.perf_counter() - t0
            return k

        # 2) Doğrulama
        dk = self._kayip(self._dogrulama, adim=2)
        k.dogrulama_kaybi = dk
        if k.egitim_kaybi is not None and dk is not None:
            k.acik = dk - k.egitim_kaybi
            k.acik_orani = dk / k.egitim_kaybi if k.egitim_kaybi > 1e-9 else 0.0

        # 3) En iyi doğrulama kaybını sakla
        if dk is not None:
            if self.en_iyi_dogrulama is None or dk < self.en_iyi_dogrulama:
                self.en_iyi_dogrulama = dk
                self.en_iyi_tur = no
                k.en_iyi = True
                try:
                    self.en_iyi_agirliklar = copy.deepcopy(
                        {a: b.detach().cpu().clone() for a, b in
                         self._motor.model.backbone.state_dict().items()})
                except Exception as exc:
                    k.not_ += f" | ağırlık saklanamadı: {str(exc)[:30]}"

        # Tur sonunda CUDA önbelleğini boşalt (bağlam kararlılığı)
        try:
            import torch as _t2

            if _t2.cuda.is_available():
                _t2.cuda.empty_cache()
                _t2.cuda.synchronize()
        except Exception:
            pass
        k.sure_s = time.perf_counter() - t0
        return k

    # ─── Ana döngü ───

    def calistir(self) -> Dict[str, Any]:
        """Doğrulamalı eğitimi çalıştırır."""
        if self._motor is None:
            return {}
        print("=" * 78)
        print(f"  DOĞRULAMALI EĞİTİM — {self.tur_sayisi} tur × {self.adim} adım")
        print(f"  doğrulama oranı %{self.dogrulama_orani * 100:.0f}, "
              f"sabır {self.sabir}")
        print("=" * 78)

        artis = 0
        for i in range(self.tur_sayisi):
            k = self.tur_calistir(i + 1)
            self.turlar.append(k)
            self._kaydet()

            ek = ""
            if k.en_iyi:
                ek = " EN IYI"
            if k.not_:
                ek += f" [{k.not_[:40]}]"
            print(f"  tur {k.no:3} | {k.adim:3} adım | {k.sure_s / 60:5.1f} dk | "
                  f"eğitim {k.egitim_kaybi if k.egitim_kaybi is None else round(k.egitim_kaybi, 4)} | "
                  f"doğrulama {k.dogrulama_kaybi if k.dogrulama_kaybi is None else round(k.dogrulama_kaybi, 4)} | "
                  f"açık {k.acik_orani:.2f}×{ek}", flush=True)

            # Erken durma: doğrulama üst üste artıyor
            if len(self.turlar) >= 2:
                onceki = self.turlar[-2].dogrulama_kaybi
                simdi = k.dogrulama_kaybi
                if onceki is not None and simdi is not None and simdi > onceki:
                    artis += 1
                else:
                    artis = 0
                if artis >= self.sabir:
                    print(f"  ⚠ doğrulama kaybı {artis} tur üst üste arttı "
                          f"— erken durma (ezber başladı)", flush=True)
                    break

        # En iyi ağırlıklara dön
        if self.en_iyi_agirliklar is not None:
            try:
                cihaz = next(self._motor.model.backbone.parameters()).device
                self._motor.model.backbone.load_state_dict(
                    {a: b.to(cihaz) for a, b in self.en_iyi_agirliklar.items()})
                print(f"  ✓ en iyi doğrulama ağırlıklarına dönüldü "
                      f"(tur {self.en_iyi_tur}, kayıp {self.en_iyi_dogrulama:.4f})")
            except Exception as exc:
                print(f"  ⚠ en iyi ağırlıklara dönülemedi: {str(exc)[:50]}")
        return self.ozet()

    def _kaydet(self) -> None:
        try:
            ILERLEME.write_text(json.dumps({
                "guncelleme": datetime.now().isoformat(timespec="seconds"),
                "en_iyi_dogrulama": self.en_iyi_dogrulama,
                "en_iyi_tur": self.en_iyi_tur,
                "turlar": [t.to_dict() for t in self.turlar],
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass

    def ozet(self) -> Dict[str, Any]:
        """Sonucu özetler."""
        eg = [t.egitim_kaybi for t in self.turlar if t.egitim_kaybi]
        dg = [t.dogrulama_kaybi for t in self.turlar if t.dogrulama_kaybi]
        print()
        print("=" * 78)
        print("  ÖZET")
        print("=" * 78)
        print(f"  tur           : {len(self.turlar)}")
        print(f"  toplam adım   : {sum(t.adim for t in self.turlar):,}")
        if eg:
            print(f"  eğitim kaybı  : {eg[0]:.4f} → {eg[-1]:.4f}")
        if dg:
            print(f"  doğrulama     : {dg[0]:.4f} → {dg[-1]:.4f}")
            fark = dg[-1] - dg[0]
            print(f"  doğrulama değişimi: {fark:+.4f} "
                  f"({'İYİLEŞTİ' if fark < 0 else 'KÖTÜLEŞTİ'})")
        print(f"  en iyi doğrulama: {self.en_iyi_dogrulama} (tur {self.en_iyi_tur})")
        return {
            "tur": len(self.turlar),
            "toplam_adim": sum(t.adim for t in self.turlar),
            "egitim_ilk": eg[0] if eg else None,
            "egitim_son": eg[-1] if eg else None,
            "dogrulama_ilk": dg[0] if dg else None,
            "dogrulama_son": dg[-1] if dg else None,
            "en_iyi_dogrulama": self.en_iyi_dogrulama,
            "en_iyi_tur": self.en_iyi_tur,
        }


def main() -> int:
    """Doğrulamalı eğitimi başlatır."""
    ap = argparse.ArgumentParser(description="Doğrulamalı eğitim")
    ap.add_argument("--tur", type=int, default=20, help="Tur sayısı")
    ap.add_argument("--adim", type=int, default=150, help="Tur başına adım")
    ap.add_argument("--dogrulama-orani", type=float, default=0.20)
    ap.add_argument("--sabir", type=int, default=3,
                    help="Doğrulama kaç tur artarsa durulsun")
    ap.add_argument("--maks-not", type=int, default=6000)
    args = ap.parse_args()

    e = DogrulamaliEgitim(tur=args.tur, adim=args.adim,
                          dogrulama_orani=args.dogrulama_orani,
                          sabir=args.sabir, maks_not=args.maks_not)
    if not e.baglan():
        return 1
    try:
        sonuc = e.calistir()
    except KeyboardInterrupt:
        print("\n  kesildi")
        sonuc = e.ozet()
    Path("_dogrulamali_ozet.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  rapor: _dogrulamali_ozet.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
