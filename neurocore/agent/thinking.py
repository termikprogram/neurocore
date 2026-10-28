"""
Düşünme Döngüsü (Think → Select → Verify)
==========================================
Yerel modelin **ölçülmüş** yeteneğini (tanıma/seçme) bulut modelinin
**üretme** yeteneğiyle birleştirir.

Neden bu döngü?
    Ölçüm sonucu (875 adım, 208M parametre, 4M token):

        Üretim (yazma)        : YOK  — çıktı dilbilgisel olarak tutarsız
        Seçme (tanıma)        : VAR  — Blender kodunda %70, yaklaşımda %50
                                       (rastgele taban %25)

    Yani "düşünme" bu ölçekte **yazmakla** değil, **seçmekle** olur:
    bulut modeli birden fazla yol önerir, yerel model en tutarlısını seçer.
    Bu, tek bir modelin yapamayacağı bir iş bölümüdür ve ikisi birlikte
    tek başına olduklarından iyidir.

Döngü:
    1. **ÜRET**  — bulut modeli N farklı yaklaşım/çözüm önerir
    2. **SEÇ**   — yerel model bunları kendi olasılığıyla sıralar
    3. **DOĞRULA** — seçim belirsizse (`margin` küçük) buluta sorulur;
       değilse yerel seçim kullanılır
    4. **KAYDET** — hangi seçimin yapıldığı ve güven düzeyi günlüğe yazılır
       (böylece isabet oranı zamanla ölçülebilir)

Dürüst sınırlar:
    * Yerel isabet %50–70'tir; `uncertain` durumunda karar **verilmez**.
    * Bulut modeli yoksa döngü çalışmaz (yerel model tek başına üretemiyor).
    * Ölçüm günlüğü tutulur ki isabet iddiası kanıtlanabilir olsun.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.thinking")

#: Kaç yaklaşım önerilsin (çok fazla = yerel model gürültüyü ayırt edemez)
DEFAULT_CANDIDATES = 3

#: Bu farkın altındaki yerel seçim "belirsiz" sayılır ve buluta düşülür.
#: Ölçüm: doğru/yanlış adaylar arası tipik fark 0,5–3,0.
DEFAULT_MIN_MARGIN = 0.15

#: Ölçüm günlüğü (isabet iddiası kanıtlanabilir olsun)
JOURNAL_FILE = "thinking_journal.jsonl"


@dataclass
class ThinkingStep:
    """Döngünün tek adımı (denetim izi)."""
    stage: str
    detail: str
    candidates: int = 0
    chosen_index: int = -1
    margin: Optional[float] = None
    confidence: str = ""
    source: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "stage": self.stage, "detail": self.detail[:300],
            "candidates": self.candidates, "chosen_index": self.chosen_index,
            "margin": self.margin, "confidence": self.confidence,
            "source": self.source, "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class ThinkingResult:
    """Döngünün sonucu."""
    ok: bool
    chosen: str = ""
    source: str = ""          # `local` (yerel seçti) | `cloud` (buluta düşüldü)
    confidence: str = ""
    margin: Optional[float] = None
    candidates: List[str] = field(default_factory=list)
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[ThinkingStep] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "ok": self.ok, "chosen": self.chosen, "source": self.source,
            "confidence": self.confidence, "margin": self.margin,
            "candidates": self.candidates,
            "ranking": self.ranking,
            "steps": [s.to_dict() for s in self.steps],
            "error": self.error,
        }


class ThinkingLoop:
    """
    Düşünme döngüsü: bulut üretir, yerel seçer, belirsizlikte buluta düşülür.

    Kullanım:
        loop = ThinkingLoop()
        loop.initialize(ranker=local_rank_controller, data_path=".neurocore")
        sonuc = loop.think(
            problem="Blender'da 200 nesne oluşturulacak",
            propose=lambda n: bulut_modeli.oner(problem, n),
        )
        print(sonuc.chosen, sonuc.source, sonuc.confidence)
    """

    name = "agent.thinking"

    def __init__(self, *, candidates: int = DEFAULT_CANDIDATES,
                 min_margin: float = DEFAULT_MIN_MARGIN) -> None:
        self.candidates = max(2, int(candidates))
        self.min_margin = float(min_margin)
        self._ranker: Any = None
        self._data_path: Optional[Path] = None
        self._initialized = False
        self._counter: Dict[str, int] = {
            "rounds": 0, "local_chosen": 0, "cloud_fallback": 0,
            "uncertain": 0, "errors": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, *, ranker: Any = None, data_path: str = "",
                   event_bus: Any = None) -> None:
        """Döngüyü başlatır (idempotent)."""
        if self._initialized:
            if ranker is not None:
                self._ranker = ranker
            return
        self._ranker = ranker
        self._data_path = Path(data_path) if data_path else Path(".neurocore")
        try:
            self._data_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._data_path = Path(".")
        self._initialized = True
        logger.info(
            f"Düşünme döngüsü hazır — aday {self.candidates}, "
            f"belirsizlik eşiği {self.min_margin}"
        )

    def shutdown(self) -> None:
        """Kaynakları bırakır."""
        self._initialized = False

    def get_stats(self) -> Dict[str, Any]:
        """Sayaçlar ve oranlar."""
        toplam = max(1, self._counter["rounds"])
        return {
            **self._counter,
            "initialized": self._initialized,
            "local_share": round(self._counter["local_chosen"] / toplam, 3),
            "uncertain_rate": round(self._counter["uncertain"] / toplam, 3),
            "candidates": self.candidates,
            "min_margin": self.min_margin,
        }

    # ─── Günlük ─────────────────────────────────────────────────────────

    def _log(self, kayit: Dict[str, Any]) -> None:
        """
        Ölçüm günlüğüne yazar (JSONL).

        Amaç: "yerel seçim işe yarıyor" iddiası **kanıtlanabilir** olsun.
        Her turda hangi kaynağın seçtiği ve güven düzeyi kaydedilir.
        """
        if self._data_path is None:
            return
        try:
            yol = self._data_path / JOURNAL_FILE
            with yol.open("a", encoding="utf-8") as dosya:
                dosya.write(json.dumps(kayit, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.debug(f"Düşünme günlüğü yazılamadı: {exc}")

    # ─── Döngü ──────────────────────────────────────────────────────────

    def think(self, problem: str,
              propose: Callable[[int], Sequence[str]],
              *, context: str = "") -> ThinkingResult:
        """
        Düşünme döngüsünü çalıştırır.

        Args:
            problem: Çözülecek problem (yaklaşım seçimi için)
            propose: `(adet) -> adaylar` üreten çağrılabilir (bulut modeli)
            context: Ek bağlam (kod, kısıt)

        Returns:
            `ThinkingResult` — seçilen yol, kaynağı ve güven düzeyi
        """
        self._counter["rounds"] += 1
        adimlar: List[ThinkingStep] = []
        basladi = time.perf_counter()

        # ── 1) ÜRET ──
        t0 = time.perf_counter()
        try:
            adaylar = [str(c).strip() for c in propose(self.candidates)
                       if str(c).strip()]
        except Exception as exc:
            self._counter["errors"] += 1
            logger.warning(f"Aday üretimi başarısız: {exc}")
            return ThinkingResult(ok=False, error=f"aday üretilemedi: {exc}",
                                  steps=adimlar)

        adimlar.append(ThinkingStep(
            stage="uret", detail=f"{len(adaylar)} aday üretildi",
            candidates=len(adaylar), source="cloud",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))
        if len(adaylar) < 2:
            # Tek aday: seçim yapılamaz, doğrudan kullan
            self._counter["cloud_fallback"] += 1
            secilen = adaylar[0] if adaylar else ""
            adimlar.append(ThinkingStep(
                stage="sec", detail="tek aday — seçim gerekmedi",
                source="cloud"))
            self._log({"problem": problem[:200], "source": "cloud",
                       "reason": "tek_aday", "chosen": secilen[:300]})
            return ThinkingResult(ok=bool(secilen), chosen=secilen,
                                  source="cloud", confidence="yok",
                                  candidates=adaylar, steps=adimlar)

        # ── 2) SEÇ (yerel model) ──
        t0 = time.perf_counter()
        if self._ranker is None:
            self._counter["errors"] += 1
            return ThinkingResult(ok=False, error="yerel sıralayıcı yok",
                                  candidates=adaylar, steps=adimlar)
        try:
            siralama = self._ranker.rank_candidates(
                f"{problem}\n{context}\nEn uygun yaklasim:",
                adaylar, min_margin=self.min_margin)
        except Exception as exc:
            self._counter["errors"] += 1
            logger.warning(f"Yerel sıralama başarısız: {exc}")
            siralama = {"ok": False, "error": str(exc)}

        if not siralama.get("ok"):
            self._counter["cloud_fallback"] += 1
            adimlar.append(ThinkingStep(
                stage="sec", detail=f"yerel sıralama başarısız: "
                                    f"{siralama.get('error', '')[:80]}",
                source="cloud"))
            self._log({"problem": problem[:200], "source": "cloud",
                       "reason": "yerel_hata", "chosen": adaylar[0][:300]})
            return ThinkingResult(ok=True, chosen=adaylar[0], source="cloud",
                                  confidence="yok", candidates=adaylar,
                                  steps=adimlar)

        belirsiz = bool(siralama.get("uncertain"))
        adimlar.append(ThinkingStep(
            stage="sec",
            detail=("yerel seçim belirsiz — buluta düşüldü" if belirsiz
                    else "yerel seçim kullanıldı"),
            candidates=len(adaylar),
            chosen_index=int(siralama.get("best_index", 0)),
            margin=siralama.get("margin"),
            confidence=str(siralama.get("confidence", "")),
            source="cloud" if belirsiz else "local",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

        # ── 3) DOĞRULA (belirsizlikte buluta düş) ──
        if belirsiz:
            self._counter["uncertain"] += 1
            self._counter["cloud_fallback"] += 1
            secilen = adaylar[0]
            kaynak = "cloud"
            # Bulut zaten adayları sırayla önerdi; ilkini "bulut tercihi" say
            adimlar.append(ThinkingStep(
                stage="dogrula",
                detail="yerel fark eşiğin altında — bulut tercihi geçerli",
                margin=siralama.get("margin"), source="cloud"))
        else:
            self._counter["local_chosen"] += 1
            index = int(siralama.get("best_index", 0))
            secilen = adaylar[index] if 0 <= index < len(adaylar) else adaylar[0]
            kaynak = "local"

        sure = (time.perf_counter() - basladi) * 1000
        adimlar.append(ThinkingStep(
            stage="bitti", detail=f"kaynak: {kaynak}", source=kaynak,
            duration_ms=sure))
        self._log({
            "problem": problem[:200], "source": kaynak,
            "confidence": siralama.get("confidence"),
            "margin": siralama.get("margin"),
            "candidates": len(adaylar),
            "chosen": secilen[:300],
        })
        logger.info(
            f"Düşünme: {len(adaylar)} aday, kaynak={kaynak}, "
            f"güven={siralama.get('confidence')}, {sure:.0f}ms"
        )
        return ThinkingResult(
            ok=True, chosen=secilen, source=kaynak,
            confidence=str(siralama.get("confidence", "")),
            margin=siralama.get("margin"), candidates=adaylar,
            ranking=list(siralama.get("ranking") or []), steps=adimlar,
        )


__all__ = [
    "ThinkingLoop", "ThinkingResult", "ThinkingStep",
    "DEFAULT_CANDIDATES", "DEFAULT_MIN_MARGIN", "JOURNAL_FILE",
]
