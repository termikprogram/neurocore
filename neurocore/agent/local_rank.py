"""
Yerel Model ile Aday Sıralama
==============================
Yerel modelin **ölçülmüş** yeteneğini işe yarar bir araca dönüştürür.

Neden bu araç?
    Ölçüm sonucu (809 adım, 208M parametre):

    * **Akıcı metin üretimi: YOK** — çıktı dilbilgisel olarak tutarsız
    * **Çoktan seçmeli tanıma: VAR** — Blender kod seçiminde **%70**
      (rastgele taban %25 → 2,8× kazanç)
    * **Anlam sıralaması: VAR** — anlamlı devam (−7,15) saçmadan (−8,87)
      daha yüksek olasılık alıyor

    Yani model "yazamıyor" ama "tanıyor". Bu, ajan mimarisinde gerçek bir
    işe yarar: bulut modelinin ürettiği **aday kodları yerel model süzebilir**.
    Böylece iki model birlikte çalışır — kullanıcının istediği
    "eklediğimiz API modellerinden öğrenme" döngüsünün uygulanabilir parçası.

Nasıl çalışır?
    Her aday, istemle birlikte modele verilir ve **yalnızca devam kısmının**
    ortalama log olasılığı hesaplanır (`NeuroCoreModel.sequence_logprob`).
    En yüksek skorlu aday "en tutarlı" sayılır. Kelime torbası sezgisi
    **kullanılmaz** — skor doğrudan modelin kendi olasılığıdır.

Dürüst sınırlar:
    * Sıralama **kesin doğru** değildir; %70 isabetlidir. Bu yüzden sonuç
      "öneri" olarak sunulur, kesin karar olarak değil.
    * Model eğitim verisinin dışındaki alanlarda güvenilmez.
    * `min_margin` eşiğinin altındaki farklar "belirsiz" işaretlenir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from neurocore.agent.tools import Safety, ToolDef
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.local_rank")

#: En az bu kadar aday verilmelidir (tek adayın sıralaması anlamsız)
MIN_CANDIDATES = 2

#: Bu farkın altındaki birincilik "belirsiz" sayılır (log olasılık birimi).
#: Ölçüm: doğru/yanlış adaylar arasındaki tipik fark 0,5–3,0 arasındadır.
DEFAULT_MIN_MARGIN = 0.05

#: Aday başına en büyük karakter (model bağlamı 2048 token)
MAX_CANDIDATE_CHARS = 4000


@dataclass
class RankedCandidate:
    """Sıralanmış tek aday."""
    index: int
    text: str
    score: float
    #: Birinciyle arasındaki fark
    margin: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "index": self.index,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "preview": self.text[:200],
        }


def _as_list(candidates: Any) -> List[str]:
    """
    Gelen adayları listeye çevirir.

    Model bazen liste yerine JSON **metni** veya `---` ile ayrılmış tek bir
    metin gönderir; ikisi de kabul edilir. Aksi hâlde araç, modelin doğal
    çıktısı yüzünden sürekli hata verirdi.
    """
    if isinstance(candidates, (list, tuple)):
        return [str(item) for item in candidates]
    if isinstance(candidates, str):
        import json as _json

        try:
            parsed = _json.loads(candidates)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        # `---` ayırıcısı veya boş satırla ayrılmış bloklar
        if "\n---\n" in candidates:
            return [part for part in candidates.split("\n---\n") if part.strip()]
        return [candidates] if candidates.strip() else []
    return []


class LocalRankController:
    """
    Yerel modelle aday sıralama aracı.

    Kullanım:
        c = LocalRankController()
        c.initialize(engine=engine)
        sonuc = c.rank_candidates("Blender'da küp ekle:\nKod:", [a, b, c])
    """

    name = "agent.local_rank"

    def __init__(self) -> None:
        self._engine: Any = None
        self._initialized = False
        self._counter: Dict[str, int] = {
            "calls": 0, "candidates_scored": 0, "errors": 0,
            "uncertain": 0, "model_unavailable": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None,
                   *, engine: Any = None) -> None:
        """Denetleyiciyi başlatır (idempotent)."""
        if self._initialized:
            if engine is not None:
                self._engine = engine
            return
        self._engine = engine
        self._initialized = True
        logger.info("Yerel sıralama denetleyicisi hazır")

    def shutdown(self) -> None:
        """Kaynakları bırakır."""
        self._initialized = False

    def get_stats(self) -> Dict[str, Any]:
        """Sayaçlar ve model durumu."""
        model = self._model()
        ready = False
        steps = 0
        if model is not None:
            try:
                ready = bool(model.can_generate)
                steps = int(model.growth.total_training_steps)
            except Exception:
                pass
        return {
            **self._counter,
            "initialized": self._initialized,
            "model_ready": ready,
            "training_steps": steps,
        }

    def _model(self) -> Any:
        """
        Yerel modeli döndürür; yoksa **tembel** yüklemeyi dener.

        Neden tembel? Masaüstü uygulaması motoru her zaman elinde tutmaz
        (208M parametre ≈ 6 sn yükleme). Sıralama aracı yalnızca
        çağrıldığında yüklenir; böylece uygulama açılışı yavaşlamaz.
        """
        if self._engine is not None:
            return getattr(self._engine, "model", None)
        # Tembel yükleme: yalnızca ilk kullanımda
        try:
            from neurocore import NeuroCoreEngine

            engine = NeuroCoreEngine()
            engine.initialize()
            self._engine = engine
            logger.info("Yerel model sıralama için yüklendi (tembel)")
            return getattr(engine, "model", None)
        except Exception as exc:
            logger.warning(f"Yerel model yüklenemedi: {exc}")
            return None

    def attach_engine(self, engine: Any) -> None:
        """Var olan motoru bağlar (yeni yükleme yapmaz)."""
        if engine is not None:
            self._engine = engine

    # ─── Sıralama ───────────────────────────────────────────────────────

    def rank_candidates(self, prompt: str,
                        candidates: Sequence[str],
                        *, min_margin: float = DEFAULT_MIN_MARGIN,
                        top_k: int = 0) -> Dict[str, Any]:
        """
        Adayları yerel modelin olasılığına göre sıralar.

        Args:
            prompt: Ortak istem (ör. "Blender'da küp ekle:\\nKod:")
            candidates: Aday metinler (en az 2)
            min_margin: Birinciliğin belirsiz sayılacağı fark eşiği
            top_k: Yalnızca ilk K adayı döndür (0 = hepsi)

        Returns:
            {"ok", "ranking", "best_index", "confidence", "uncertain", ...}
        """
        self._counter["calls"] += 1

        temiz = [str(c)[:MAX_CANDIDATE_CHARS] for c in candidates
                 if str(c).strip()]
        if len(temiz) < MIN_CANDIDATES:
            return {"ok": False,
                    "error": f"en az {MIN_CANDIDATES} aday gerekir",
                    "received": len(temiz)}

        model = self._model()
        if model is None:
            self._counter["model_unavailable"] += 1
            return {"ok": False, "error": "yerel model yüklü değil"}

        basladi = time.perf_counter()
        skorlar: List[float] = []
        for aday in temiz:
            try:
                skor = model.sequence_logprob(f"{prompt}\n{aday}", prompt=prompt)
            except Exception as exc:
                logger.debug(f"Aday skorlanamadı: {exc}")
                skor = None
            skorlar.append(float(skor) if skor is not None else float("-inf"))
            self._counter["candidates_scored"] += 1

        if all(s == float("-inf") for s in skorlar):
            self._counter["errors"] += 1
            return {"ok": False, "error": "hiçbir aday skorlanamadı"}

        sirali = sorted(range(len(temiz)), key=lambda i: skorlar[i], reverse=True)
        en_iyi = skorlar[sirali[0]]
        ikinci = skorlar[sirali[1]] if len(sirali) > 1 else float("-inf")
        fark = (en_iyi - ikinci) if ikinci != float("-inf") else float("inf")
        belirsiz = fark < float(min_margin)
        if belirsiz:
            self._counter["uncertain"] += 1

        sonuc: List[RankedCandidate] = []
        for sira, index in enumerate(sirali):
            sonuc.append(RankedCandidate(
                index=index, text=temiz[index], score=skorlar[index],
                margin=skorlar[index] - en_iyi,
            ))
        if top_k > 0:
            sonuc = sonuc[:top_k]

        sure = (time.perf_counter() - basladi) * 1000
        logger.info(
            f"Yerel sıralama: {len(temiz)} aday, en iyi #{sirali[0]} "
            f"(fark {fark:.3f}{', BELİRSİZ' if belirsiz else ''}), "
            f"{sure:.0f}ms"
        )
        return {
            "ok": True,
            "best_index": sirali[0],
            "best_text": temiz[sirali[0]],
            "margin": round(fark, 4) if fark != float("inf") else None,
            "uncertain": belirsiz,
            "confidence": ("belirsiz" if belirsiz
                           else ("yüksek" if fark >= 0.5 else "orta")),
            "candidates": len(temiz),
            "ranking": [item.to_dict() for item in sonuc],
            "duration_ms": round(sure, 1),
            "model_steps": int(getattr(model.growth, "total_training_steps", 0) or 0),
        }

    # ─── Araçlar ────────────────────────────────────────────────────────

    def tools(self) -> List[ToolDef]:
        """Ajan için araç tanımları."""
        from neurocore.agent.tools import ToolParameter as P

        def handler(prompt: str = "", candidates: Any = None,
                    min_margin: float = DEFAULT_MIN_MARGIN,
                    top_k: int = 0, **_: Any) -> Dict[str, Any]:
            """Araç gövdesi."""
            return self.rank_candidates(prompt, _as_list(candidates),
                                        min_margin=float(min_margin),
                                        top_k=int(top_k))

        def approaches_handler(problem: str = "", approaches: Any = None,
                               min_margin: float = DEFAULT_MIN_MARGIN,
                               **_: Any) -> Dict[str, Any]:
            """
            Yaklaşım/çözüm yollarını sıralar.

            Ölçülen isabet: yol seçiminde %50, teşhiste %50 (taban %25).
            Bu, kod seçiminden (%70) daha zayıftır; sonuç **öneri** olarak
            kullanılmalı ve `uncertain` ise karar verilmemelidir.
            """
            return self.rank_candidates(
                f"{problem}\nEn uygun yaklasim:", _as_list(approaches),
                min_margin=float(min_margin))

        return [
            ToolDef(
                name="local.rank_candidates",
                description=(
                    "Yerel modelin kendi olasılığıyla aday metinleri/kodları "
                    "sıralar. Bulut modelinin ürettiği birden fazla çözüm "
                    "arasından en tutarlısını seçmek için kullanılır. "
                    "ÖLÇÜLEN İSABET: Blender kod seçiminde %70 (rastgele %25). "
                    "Sonucu 'öneri' olarak değerlendirin; `uncertain` true ise "
                    "fark belirsizdir ve karar vermeyin."
                ),
                handler=handler,
                parameters=[
                    P("prompt", required=True,
                      description="Ortak istem (ör. 'Blender\\'da küp ekle:\\nKod:')"),
                    P("candidates", "array", "Aday metinler (en az 2)",
                      required=True),
                    P("min_margin", "number",
                      "Bu farkın altı belirsiz sayılır", default=DEFAULT_MIN_MARGIN),
                    P("top_k", "integer", "Yalnızca ilk K aday (0 = hepsi)",
                      default=0),
                ],
                safety=Safety.READ,
                category="local",
                returns="sıralama, en iyi aday, güven düzeyi",
                tags=["yerel", "sıralama", "seçim", "kod"],
            ),
            ToolDef(
                name="local.rank_approaches",
                description=(
                    "Bir problem için önerilen **yaklaşımları** yerel modelle "
                    "sıralar ('hangisi daha etkili?'). Düşünme/planlama "
                    "aşamasında, bulut modelinin sunduğu çözüm yolları "
                    "arasından seçim yapmak için kullanılır. "
                    "ÖLÇÜLEN İSABET: %50 (rastgele %25). Kod seçiminden "
                    "zayıftır — yalnızca daraltma amaçlı kullanın."
                ),
                handler=approaches_handler,
                parameters=[
                    P("problem", required=True, description="Problem tanımı"),
                    P("approaches", "array", "Aday yaklaşımlar (en az 2)",
                      required=True),
                    P("min_margin", "number", "Belirsizlik eşiği",
                      default=DEFAULT_MIN_MARGIN),
                ],
                safety=Safety.READ,
                category="local",
                returns="sıralama, en uygun yaklaşım, güven düzeyi",
                tags=["yerel", "düşünme", "planlama", "seçim"],
            ),
            ToolDef(
                name="local.model_status",
                description=(
                    "Yerel modelin durumunu bildirir: eğitim adımı, üretim "
                    "yapabilir mi, sıralama sayaçları. Yerel modeli kullanmadan "
                    "ÖNCE çağırın."
                ),
                handler=lambda **_: self.get_stats(),
                parameters=[],
                safety=Safety.READ,
                category="local",
                returns="model durumu ve sayaçlar",
                tags=["yerel", "durum"],
            ),
        ]


__all__ = ["LocalRankController", "RankedCandidate",
           "MIN_CANDIDATES", "DEFAULT_MIN_MARGIN"]
