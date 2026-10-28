"""
Düşünme İş Akışı — Düşünmeyi Adımlara Böler
=============================================
Bir düşünme turunu **üç ayrı adıma** böler: üret → seç → doğrula. Böylece
her aşama ayrı ayrı kaydedilir, ölçülür ve gerektiğinde **paralel** çalışır.

Neden bölünür?
    `ThinkingLoop.think()` tek bir çağrıdır: adayları üretir, sıralar,
    karar verir. Uzun bir projede bu çağrı hem yavaş hem **opak** olur:

        * Hangi aşama yavaş? (üretim mi, sıralama mı?)
        * Aday üretimi başarısız mı oldu, yoksa sıralama mı belirsiz kaldı?
        * Kesinti olursa hangi aşamadan devam edilir?

    Adımlara bölmek bunların hepsini görünür kılar ve **paralellik** açar:
    birden fazla bağımsız problem için aday üretimi eşzamanlı yapılabilir.

Üç adım:

    1. **ÜRET**    (cloud)  — bulut modeli N aday önerir
    2. **SEÇ**     (local)  — yerel model adayları sıralar
    3. **DOĞRULA** (cloud)  — belirsizse buluta düşülür; değilse onaylanır

Kaynak ihtiyacı: üretim ve doğrulama **ağ bekler** (hafif CPU, az RAM);
sıralama **yerel modeli çalıştırır** (GPU/CPU). Bu ayrım, motorun doğru
parçaları kuyruğa almasını sağlar.

Ölçülen değer (karar için):
    * Yerel seçim isabeti: Blender kodunda %70, yaklaşımda %50
    * Yerel model yoksa döngü **buluta düşer** ve bunu açıkça bildirir
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from neurocore.core.governor import ResourceSnapshot
from neurocore.core.work import WorkEngine, WorkStep
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.thinking_work")

#: Aşama kaynak ihtiyaçları (ölçüme dayalı)
STAGE_RESOURCES: Dict[str, Dict[str, float]] = {
    # Bulut çağrısı: ağ bekler, CPU/GPU neredeyse boş
    "uret": {"cpu": 0.5, "ram_mb": 40.0},
    # Yerel model çıkarımı: GPU/CPU kullanır
    "sec": {"cpu": 8.0, "ram_mb": 900.0, "vram_mb": 900.0},
    # Doğrulama: bulut çağrısı (genelde kısa)
    "dogrula": {"cpu": 0.5, "ram_mb": 40.0},
}

#: Aşama tahmini süreleri (saniye)
STAGE_ESTIMATES: Dict[str, float] = {
    "uret": 8.0,      # bulut modeli yanıtı
    "sec": 2.0,       # yerel sıralama (ölçüldü: 3 ms–2 sn)
    "dogrula": 0.2,   # karar (ağ çağrısı yok, mantık)
}


@dataclass
class ThinkingTrace:
    """Bir düşünme turunun aşama aşama kaydı."""
    problem: str = ""
    candidates: List[str] = field(default_factory=list)
    chosen: str = ""
    source: str = ""
    confidence: str = ""
    margin: Optional[float] = None
    stage_times: Dict[str, float] = field(default_factory=dict)
    error: str = ""

    @property
    def total_s(self) -> float:
        """Tüm aşamaların toplam süresi."""
        return sum(self.stage_times.values())

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "problem": self.problem[:200],
            "candidates": len(self.candidates),
            "chosen": self.chosen[:300],
            "source": self.source,
            "confidence": self.confidence,
            "margin": self.margin,
            "stage_times": {k: round(v, 3) for k, v in self.stage_times.items()},
            "total_s": round(self.total_s, 3),
            "error": self.error[:200],
        }


class ThinkingWorkflow:
    """
    Düşünmeyi parçalı motora bağlar.

    Kullanım:
        akis = ThinkingWorkflow(workspace=".neurocore")
        akis.initialize(thinking=thinking_loop,
                        propose=lambda problem, n: cloud.propose(problem, n))
        akis.plan_problems(["problem A", "problem B"])
        rapor = akis.run()
        for iz in akis.traces():
            print(iz.chosen, iz.source, iz.total_s)
    """

    name = "agent.thinking_work"

    def __init__(self, *, workspace: str = "",
                 chunk_ratio: float = 0.5) -> None:
        #: Parça payı: her düşünme parçası bütçenin yarısını kullanır →
        #: iki problem paralel düşünülebilir.
        self.engine = WorkEngine(workspace=workspace,
                                 chunk_ratio=chunk_ratio, chunk_steps=3)
        self._thinking: Any = None
        self._propose: Optional[Callable[[str, int], Sequence[str]]] = None
        self._candidates: int = 3
        self._traces: List[ThinkingTrace] = []
        self._state: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        self._counter: Dict[str, int] = {
            "produces": 0, "selects": 0, "verifies": 0, "cloud_fallbacks": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, *, thinking: Any = None,
                   propose: Optional[Callable[[str, int], Sequence[str]]] = None,
                   candidates: int = 3) -> None:
        """
        İş akışını başlatır.

        Args:
            thinking: `ThinkingLoop` (yerel seçim için)
            propose: `(problem, adet) -> adaylar` üreten bulut çağrısı
            candidates: Kaç aday önerilsin
        """
        if thinking is not None:
            self._thinking = thinking
        if propose is not None:
            self._propose = propose
        self._candidates = max(2, int(candidates))
        if self._initialized:
            return
        self.engine.register("think_uret", self._run_produce)
        self.engine.register("think_sec", self._run_select)
        self.engine.register("think_dogrula", self._run_verify)
        self._initialized = True
        logger.info(
            f"Düşünme iş akışı hazır — aday {self._candidates}, "
            f"3 aşama (üret/seç/doğrula)"
        )

    def shutdown(self) -> None:
        """Kaynakları bırakır."""
        self._initialized = False

    # ─── Planlama ───────────────────────────────────────────────────────

    def plan_problems(self, problems: Sequence[str], *,
                      context: str = "",
                      objective: str = "") -> List[WorkStep]:
        """
        Her problem için üç aşamalı zincir kurar.

        Proble **bağımsız** problemler paralel düşünülebilir; bu yüzden
        aralarında bağımlılık **kurulmaz**. Aşamalar kendi içinde sıralıdır
        (üret → seç → doğrula).

        Args:
            problems: Düşünülecek problemler
            context: Ortak bağlam (kod, kısıt)
            objective: Amaç metni

        Returns:
            `WorkStep` listesi
        """
        adimlar: List[WorkStep] = []
        for pindex, problem in enumerate(problems):
            anahtar = f"p{pindex:03d}"
            self._state[anahtar] = {"problem": problem, "context": context,
                                    "candidates": [], "rank": None,
                                    "chosen": "", "source": "",
                                    "confidence": "", "margin": None,
                                    "stage_times": {}, "error": ""}
            önceki: Optional[str] = None
            for asama, tur in (("uret", "think_uret"), ("sec", "think_sec"),
                               ("dogrula", "think_dogrula")):
                kimlik = f"{anahtar}_{asama}"
                adimlar.append(WorkStep(
                    id=kimlik,
                    title=f"{problem[:48]} — {asama}",
                    kind=tur,
                    needs=dict(STAGE_RESOURCES[asama]),
                    depends_on=[önceki] if önceki else [],
                    payload={"key": anahtar, "stage": asama,
                             "problem": problem, "context": context},
                    estimate_s=STAGE_ESTIMATES[asama],
                ))
                önceki = kimlik
        amaç = objective or f"{len(problems)} problem için düşünme"
        self.engine.plan(amaç, adimlar)
        logger.info(f"Düşünme planı: {len(problems)} problem × 3 aşama "
                    f"= {len(adimlar)} adım")
        return adimlar

    # ─── Aşamalar ───────────────────────────────────────────────────────

    def _run_produce(self, step: WorkStep) -> str:
        """Aşama 1: adayları üret (bulut modeli)."""
        anahtar = str(step.payload.get("key"))
        durum = self._state.setdefault(anahtar, {})
        problem = str(step.payload.get("problem") or "")
        self._counter["produces"] += 1
        basladi = time.perf_counter()
        try:
            if self._propose is None:
                raise RuntimeError("aday üretici bağlı değil")
            adaylar = [str(c).strip() for c in
                       self._propose(problem, self._candidates)
                       if str(c).strip()]
        except Exception as exc:
            durum["error"] = f"üretim hatası: {exc}"
            durum.setdefault("stage_times", {})["uret"] = (
                time.perf_counter() - basladi)
            raise
        durum["candidates"] = adaylar
        durum.setdefault("stage_times", {})["uret"] = time.perf_counter() - basladi
        return f"{len(adaylar)} aday üretildi"

    def _run_select(self, step: WorkStep) -> str:
        """Aşama 2: yerel model adayları sıralar."""
        anahtar = str(step.payload.get("key"))
        durum = self._state.setdefault(anahtar, {})
        adaylar = list(durum.get("candidates") or [])
        self._counter["selects"] += 1
        basladi = time.perf_counter()
        if len(adaylar) < 2 or self._thinking is None:
            # Seçim yapılamıyor: ilk aday kullanılır, **açıkça** bildirilir
            durum["chosen"] = adaylar[0] if adaylar else ""
            durum["source"] = "cloud"
            durum["confidence"] = "yok"
            durum.setdefault("stage_times", {})["sec"] = (
                time.perf_counter() - basladi)
            return ("seçim atlandı (tek aday veya yerel model yok) — "
                    f"ilk aday: {durum['chosen'][:200]}")

        problem = str(step.payload.get("problem") or "")
        baglam = str(step.payload.get("context") or "")
        try:
            sonuc = self._thinking.think(
                f"{problem}\n{baglam}\nEn uygun yaklasim:", 
                lambda n: adaylar[:n])
        except Exception as exc:
            durum["error"] = f"seçim hatası: {exc}"
            durum["chosen"] = adaylar[0]
            durum["source"] = "cloud"
            durum.setdefault("stage_times", {})["sec"] = (
                time.perf_counter() - basladi)
            raise
        durum["chosen"] = sonuc.chosen
        durum["source"] = sonuc.source
        durum["confidence"] = sonuc.confidence
        durum["margin"] = sonuc.margin
        durum["rank"] = sonuc.ranking
        durum.setdefault("stage_times", {})["sec"] = time.perf_counter() - basladi
        return (f"[{sonuc.source}/{sonuc.confidence}] "
                f"{sonuc.chosen[:220]}")

    def _run_verify(self, step: WorkStep) -> str:
        """
        Aşama 3: doğrula.

        Yerel seçim **belirginse** onaylanır; belirsizse veya yerel model
        yoksa bulut tercihi geçerli olur ve bu durum kaydedilir.
        """
        anahtar = str(step.payload.get("key"))
        durum = self._state.setdefault(anahtar, {})
        self._counter["verifies"] += 1
        basladi = time.perf_counter()
        kaynak = str(durum.get("source") or "")
        if kaynak == "local":
            karar = "yerel seçim onaylandı"
        else:
            self._counter["cloud_fallbacks"] += 1
            karar = "yerel seçim kullanılmadı — bulut tercihi geçerli"
        durum.setdefault("stage_times", {})["dogrula"] = (
            time.perf_counter() - basladi)

        iz = ThinkingTrace(
            problem=str(durum.get("problem") or ""),
            candidates=list(durum.get("candidates") or []),
            chosen=str(durum.get("chosen") or ""),
            source=kaynak, confidence=str(durum.get("confidence") or ""),
            margin=durum.get("margin"),
            stage_times=dict(durum.get("stage_times") or {}),
            error=str(durum.get("error") or ""),
        )
        self._traces.append(iz)
        return f"{karar} | {iz.chosen[:200]}"

    # ─── Yürütme ve Raporlama ───────────────────────────────────────────

    def run(self, *, snapshot: Optional[ResourceSnapshot] = None,
            max_wait_s: float = 600.0, poll_s: float = 15.0) -> Dict[str, Any]:
        """
        Planlanan düşünme adımlarını çalıştırır.

        ⚠ `max_wait_s` **test edilebilir olmalı**: sınırsız bekleme, yerel
        modelin VRAM'i yetmediği bir sistemde testi askıda bırakır.
        """
        rapor = self.engine.run_until_done(
            snapshot=snapshot, max_wait_s=max_wait_s, poll_s=poll_s)
        rapor["thinking"] = [iz.to_dict() for iz in self._traces]
        rapor["stages"] = self.stage_report()
        return rapor

    def stage_report(self) -> Dict[str, Any]:
        """
        Aşama başına süre ölçümü — **nerede yavaş** sorusunun cevabı.
        """
        toplam: Dict[str, float] = {}
        sayim: Dict[str, int] = {}
        for iz in self._traces:
            for asama, sure in (iz.stage_times or {}).items():
                toplam[asama] = toplam.get(asama, 0.0) + float(sure)
                sayim[asama] = sayim.get(asama, 0) + 1
        return {
            "stages": {
                asama: {
                    "calls": sayim.get(asama, 0),
                    "total_s": round(toplam.get(asama, 0.0), 3),
                    "mean_s": round(toplam.get(asama, 0.0)
                                    / max(1, sayim.get(asama, 0)), 3),
                }
                for asama in ("uret", "sec", "dogrula")
            },
            "local_share": round(
                sum(1 for iz in self._traces if iz.source == "local")
                / max(1, len(self._traces)), 3),
            "counter": dict(self._counter),
        }

    def traces(self) -> List[ThinkingTrace]:
        """Tamamlanan düşünme izleri."""
        return list(self._traces)

    def get_stats(self) -> Dict[str, Any]:
        """Sayaçlar ve aşama raporu."""
        return {
            "initialized": self._initialized,
            "traces": len(self._traces),
            "report": self.stage_report(),
            "engine": self.engine.get_stats(),
        }


__all__ = [
    "ThinkingWorkflow", "ThinkingTrace",
    "STAGE_RESOURCES", "STAGE_ESTIMATES",
]
