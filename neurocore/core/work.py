"""
Parçalı İş Motoru — Planla, Böl, Sırala, Yürüt, Kaydet
======================================================
Büyük bir işi **IDM mantığıyla** parçalara böler, kaynak bütçesine göre
paralel çalıştırır, sığmayanı **kuyruğa** alır ve her adımı **günlüğe**
yazar.

Neden bu motor?
    Kullanıcı isteği: "8.000.000 adımlı bir proje verildi diyelim; GPU, CPU
    ve RAM'de kullanacağı alana göre işleri parçalara bölecek, fazladan
    bölemediği adımları sıraya alacak. Nerede kaldığını, ne yaptığını
    kayıt altına alacak."

    Tek bir `for` döngüsüyle 8M adım çalıştırmak:
        * kaynağı doyurur (bilgisayar kilitlenir),
        * kesintiye uğrarsa **baştan başlar**,
        * nerede kalındığı bilinmez,
        * paralelleştirilemez.

    Bu motor dört sorunu birden çözer:

        1. **Bölme**    — iş, kaynak bütçesine sığan parçalara ayrılır
        2. **Sıralama** — sığmayan parçalar kuyruğa girer, kaynak açılınca çıkar
        3. **Dayanıklılık** — her parça bitince günlüğe yazılır; kesintiden
           sonra **kaldığı yerden** devam eder
        4. **İzlenebilirlik** — hangi adımda ne yapıldığı, ne kadar sürdüğü,
           hangi kaynağı kullandığı kayıtlıdır; sonunda **algoritma** olarak
           yeniden üretilebilir

Akış:

    PLANLA  → adımları çıkar, bağımlılıkları bağla, kaynak ihtiyacını yaz
    BÖL     → bağımsız adımları gruplayıp **parça** yap
    ÖLÇ     → kaynak yönetişiminden bütçe al (kaç parça paralel?)
    YÜRÜT   → hazır parçaları çalıştır, biteni kaydet, bağımlıları aç
    KUYRUK  → sığmayan/bağımlılığı gelmemiş parçalar bekler
    KAYDET  → günlük: adım, süre, kaynak, sonuç

Günlük aynı zamanda **algoritmanın kendisidir**: `algorithm()` metodu
yapılan işi adım adım, süreleriyle birlikte yeniden üretir.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from neurocore.core.governor import (
    ChunkBudget,
    GovernorPolicy,
    ResourceGovernor,
    ResourceSnapshot,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("core.work")

#: Parça durumları
PENDING = "bekliyor"
READY = "hazır"
RUNNING = "çalışıyor"
DONE = "bitti"
FAILED = "başarısız"
SKIPPED = "atlandı"
BLOCKED = "kaynak bekliyor"
#: Dış girdi bekleniyor (kullanıcı onayı gibi).
#:
#: Neden ayrı durum? Onay bekleyen adımı "başarısız" saymak, toplu işlerde
#: her yazma adımının çökmüş görünmesine yol açıyordu. İş **bekliyor**,
#: bozulmamış; kullanıcı onaylayınca kaldığı yerden devam eder.
WAITING = "onay bekliyor"


class WorkWaiting(Exception):
    """
    Adım **dış girdi** bekliyor (başarısız değil).

    Bir işleyici bunu fırlatarak "bu adım şu an yapılamaz ama bozulmadı"
    der. Motor adımı `WAITING` durumuna alır ve gerekçeyi kaydeder.
    """

    def __init__(self, reason: str = "") -> None:
        super().__init__(reason or "dış girdi bekleniyor")
        self.reason = reason or "dış girdi bekleniyor"

#: Günlük dosyası (parça tamamlandıkça eklenir — kesintiye dayanıklı)
JOURNAL_FILE = "work_journal.jsonl"


@dataclass
class WorkStep:
    """
    Tek bir iş adımı.

    `needs` alanı kaynak ihtiyacını taşır: `{"cpu": 5.0, "ram_mb": 200,
    "vram_mb": 0}`. Yönetişim bu değere göre parçanın sığıp sığmadığına
    karar verir.
    """
    id: str
    title: str
    kind: str = "generic"
    needs: Dict[str, float] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    #: Tahmini süre (saniye) — planlama ve sıralama için
    estimate_s: float = 1.0
    status: str = PENDING
    chunk_id: str = ""
    started: float = 0.0
    finished: float = 0.0
    result: str = ""
    error: str = ""

    @property
    def duration_s(self) -> float:
        """Gerçekleşen süre."""
        if self.started and self.finished:
            return max(0.0, self.finished - self.started)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "id": self.id, "title": self.title, "kind": self.kind,
            "needs": dict(self.needs), "depends_on": list(self.depends_on),
            "estimate_s": self.estimate_s, "status": self.status,
            "chunk_id": self.chunk_id,
            "duration_s": round(self.duration_s, 3),
            "result": self.result[:400], "error": self.error[:300],
        }


@dataclass
class WorkChunk:
    """Birlikte çalıştırılacak adım grubu."""
    id: str
    steps: List[WorkStep] = field(default_factory=list)
    needs: Dict[str, float] = field(default_factory=dict)
    status: str = PENDING
    reason: str = ""
    started: float = 0.0
    finished: float = 0.0

    @property
    def duration_s(self) -> float:
        """Parçanın süresi."""
        if self.started and self.finished:
            return max(0.0, self.finished - self.started)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "id": self.id, "steps": [s.id for s in self.steps],
            "needs": {k: round(v, 2) for k, v in self.needs.items()},
            "status": self.status, "reason": self.reason[:200],
            "duration_s": round(self.duration_s, 3),
        }


class WorkJournal:
    """
    Parça günlüğü (JSONL, eklemeli).

    Neden JSONL? Kesinti anında dosya bozulmaz: her satır bağımsız bir
    kayıttır, yarım kalan son satır atlanır. Tek bir JSON dosyası yazma
    sırasında kesilirse **tümü** kaybolurdu.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._memory: List[Dict[str, Any]] = []

    def append(self, kayit: Dict[str, Any]) -> None:
        """Bir kaydı günlüğe ekler."""
        with self._lock:
            self._memory.append(kayit)
            if self.path is None:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as dosya:
                    dosya.write(json.dumps(kayit, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.debug(f"Günlük yazılamadı: {exc}")

    def records(self) -> List[Dict[str, Any]]:
        """Bellekteki kayıtlar."""
        with self._lock:
            return list(self._memory)

    def load(self) -> List[Dict[str, Any]]:
        """
        Günlüğü diskten okur (yarım satırlar atlanır).

        Kesintiden sonra "nerede kaldım" sorusunun cevabı buradan gelir.
        """
        if self.path is None or not self.path.exists():
            return []
        kayitlar: List[Dict[str, Any]] = []
        try:
            for satir in self.path.read_text(encoding="utf-8").splitlines():
                satir = satir.strip()
                if not satir:
                    continue
                try:
                    kayitlar.append(json.loads(satir))
                except json.JSONDecodeError:
                    # Yarım yazılmış satır — atla, kalanı oku
                    continue
        except OSError as exc:
            logger.debug(f"Günlük okunamadı: {exc}")
        return kayitlar


class WorkEngine:
    """
    Parçalı iş motoru.

    Kullanım:
        engine = WorkEngine(workspace=".neurocore")
        engine.plan("Proje açıklaması", steps=[...])
        engine.register("build", build_fn)
        rapor = engine.run()
        print(engine.algorithm())
    """

    name = "core.work"

    def __init__(self, *, workspace: str = "",
                 policy: Optional[GovernorPolicy] = None,
                 chunk_steps: int = 8,
                 chunk_ratio: float = 0.0) -> None:
        # Parça payı **yönetişim politikasıyla aynı** olmalı; aksi hâlde
        # planlayıcının kurduğu parça boyu ile hesaplanan eşzamanlılık
        # çelişir (iri parçalar birleşip paralelliği yok eder).
        politika = policy or GovernorPolicy()
        if chunk_ratio:
            politika.chunk_ratio = max(0.02, min(1.0, float(chunk_ratio)))
        self.governor = ResourceGovernor(politika)
        #: Parça başına en fazla adım (kaynak yetse de parça şişmesin)
        self.chunk_steps = max(1, int(chunk_steps))
        #: Yönetişimle aynı parça payı — bkz. `GovernorPolicy.chunk_ratio`
        self.chunk_ratio = politika.chunk_ratio
        self.steps: List[WorkStep] = []
        self.chunks: List[WorkChunk] = []
        self._handlers: Dict[str, Callable[[WorkStep], Any]] = {}
        self._journal = WorkJournal(
            (Path(workspace) / JOURNAL_FILE) if workspace else None)
        self._lock = threading.RLock()
        self._objective = ""
        self._counter: Dict[str, int] = {
            "runs": 0, "steps_done": 0, "steps_failed": 0,
            "chunks_done": 0, "chunks_deferred": 0,
        }

    # ─── Planlama ───────────────────────────────────────────────────────

    def plan(self, objective: str,
             steps: Sequence[WorkStep],
             *, snapshot: Optional[ResourceSnapshot] = None) -> List[WorkChunk]:
        """
        İşi planlar: adımları alır, **kaynağa sığan parçalara** böler.

        Bölme mantığı (IDM benzeri, kaynak farkında):
            1. Adımlar bağımlılıklarına göre **katmanlara** ayrılır
               (aynı katmandaki adımlar birbirinden bağımsızdır).
            2. Her katmanda adımlar sırayla taranır ve **bütçeye sığdıkça**
               aynı parçaya eklenir.
            3. Bütçeyi aşacak adım **yeni parça** başlatır; tek başına da
               sığmıyorsa kendi parçasına konur ve çalışma anında
               **kuyruğa** alınır (kaynak açılınca çalışır).

        Neden adım sayısına göre değil kaynağa göre?
            Ölçülen hata: parçalar sabit `chunk_steps` ile kurulduğunda
            (ör. 5 adım × 400 MB = 2000 MB) küçük bir sistemin parça bütçesi
            (205 MB) hiçbir parçaya yetmiyor ve **tek adım bile
            çalışmıyordu**. Parça sınırı kaynaktan türemelidir.

        Args:
            objective: Projenin amacı (günlüğe yazılır)
            steps: İş adımları
            snapshot: Kaynak durumu (None = canlı oku)

        Returns:
            Parça listesi
        """
        with self._lock:
            self._objective = objective
            # ⚠ Adımlar **yeniden kurulurken** durumları korunmalı.
            # `append_plan` tamamlanmış adımları `DONE` bırakır; `plan()`
            # bunları sıfırlarsa tamamlanan iş **tekrar çalışır** (ölçüldü:
            # bir adım iki kez koştu). Yalnızca yeni adımlar `PENDING`
            # başlar.
            yeni_adimlar: List[WorkStep] = []
            for adim in steps:
                if adim.status in (DONE, SKIPPED):
                    yeni_adimlar.append(adim)
                    continue
                adim.status = PENDING
                adim.started = 0.0
                adim.finished = 0.0
                adim.chunk_id = ""
                yeni_adimlar.append(adim)
            self.steps = yeni_adimlar
            self.chunks = []

            anlik = snapshot or ResourceSnapshot.live()
            butce = self.governor.plan(anlik)
            katmanlar = self._layers(self.steps)
            # Tavan her katmanda o katmanın en büyük adımına göre uyarlanır
            tavan = self._chunk_ceiling(
                butce, [a for k in katmanlar for a in k])
            logger.info(
                f"Parça tavanı: CPU %{tavan.get('cpu', 0):.1f} · "
                f"RAM {tavan.get('ram_mb', 0):.0f} MB · "
                f"VRAM {tavan.get('vram_mb', 0):.0f} MB"
            )

            # Hedef parça sayısı: `1 / chunk_ratio` (parça boyu ile tutarlı).
            # Her katman bu sayıya **bölünür**; böylece bağımsız adımlar
            # ayrı parçalara düşer ve eşzamanlı çalışabilir.
            hedef_parca = max(1, int(round(1.0 / max(1e-6, self.chunk_ratio))))
            hedef_parca = min(hedef_parca, max(1, butce.parallel))

            sira = 0
            for seviye, katman in enumerate(katmanlar):
                # ── Katmanı eşit parçalara böl ──
                #
                # Hedef parça sayısı `1 / chunk_ratio`; böylece **parça boyu
                # ile eşzamanlılık hedefi tutarlı** olur.
                #
                # ⚠ Küçük dilimleri "birleştirme" tuzağı: parçaları tavan
                # altında kaldıkça birleştirmek paralelliği **yok eder**
                # (ölçüldü: 8 bağımsız adım tek parçaya düşüp seri çalıştı,
                # 1,64 sn — hiç hızlanma yok). Küçük parça, kaynak nezaketi
                # açısından zararsızdır; paralelliği korumak esas amaçtır.
                #
                # `chunk_steps` yalnızca **üst sınır** olarak uygulanır
                # (kaynak bolsa bile parça şişmesin).
                ulasilabilir = min(max(1, hedef_parca), len(katman))
                ust_sinir = max(1, self.chunk_steps)
                ulasilabilir = max(1, min(ulasilabilir,
                                          max(1, len(katman) // 1)))
                parca_boyu = max(1, min(
                    ust_sinir,
                    -(-len(katman) // ulasilabilir),      # tavan bölme
                ))

                for bas in range(0, len(katman), parca_boyu):
                    dilim = list(katman[bas:bas + parca_boyu])
                    ihtiyac = self._sum_needs(dilim)
                    asiyor = not self._within(ihtiyac, tavan)
                    self.chunks.append(self._make_chunk(
                        sira, dilim, ihtiyac, seviye,
                        not_=("parça tavanı aşıyor — kuyruğa alınabilir"
                              if asiyor else "")))
                    sira += 1

            for parca in self.chunks:
                for adim in parca.steps:
                    adim.chunk_id = parca.id

        logger.info(
            f"Plan hazır: {len(self.steps)} adım → {len(self.chunks)} parça "
            f"({len(katmanlar)} katman, parça tavanı CPU %{tavan.get('cpu', 0):.1f}"
            f"/RAM {tavan.get('ram_mb', 0):.0f}MB/"
            f"VRAM {tavan.get('vram_mb', 0):.0f}MB)"
        )
        return self.chunks

    def _chunk_ceiling(self, budget: ChunkBudget,
                       steps: Sequence[WorkStep] = ()) -> Dict[str, float]:
        """
        Parça başına kaynak tavanı — **en büyük adıma göre uyarlanır**.

        Temel kural: tavan = bütçe × `chunk_ratio`; böylece bütçe birkaç
        parçayı besler ve paralellik doğar.

        ⚠ Uyarlama şart: tek bir adım sabit dilimden büyükse (ör. 400 MB'lık
        adım, 205 MB'lık dilim) hiçbir parça sığmaz ve **tek adım bile
        çalışmaz** — ölçüldü. Bu yüzden tavan, verilen adımların en büyük
        ihtiyacına kadar **genişletilir**; genişletme toplam bütçeyi aşmaz.

        Args:
            budget: Yönetişimden gelen parça bütçesi
            steps: Katmandaki adımlar (en büyüğü tavanı belirler)

        Returns:
            {"cpu", "ram_mb", "vram_mb"} tavanı
        """
        pay = max(0.05, min(1.0, float(self.chunk_ratio)))
        tavan = {
            "cpu": max(1.0, budget.cpu_percent * pay),
            "ram_mb": max(16.0, budget.ram_mb * pay),
            "vram_mb": max(0.0, budget.vram_mb * pay),
        }
        tam = {
            "cpu": max(1.0, budget.cpu_percent),
            "ram_mb": max(16.0, budget.ram_mb),
            "vram_mb": max(0.0, budget.vram_mb),
        }
        for adim in steps:
            for kaynak, deger in (adim.needs or {}).items():
                gerekli = float(deger)
                if gerekli > tavan.get(kaynak, 0.0):
                    tavan[kaynak] = min(gerekli, tam.get(kaynak, gerekli))
        return tavan

    @staticmethod
    def _merge_needs(base: Dict[str, float],
                     extra: Dict[str, float]) -> Dict[str, float]:
        """İki kaynak ihtiyacını toplar."""
        birlesik = dict(base)
        for kaynak, deger in (extra or {}).items():
            birlesik[kaynak] = birlesik.get(kaynak, 0.0) + float(deger)
        return birlesik

    @staticmethod
    def _within(needs: Dict[str, float], tavan: Dict[str, float]) -> bool:
        """İhtiyaç tavanın içinde mi?"""
        for kaynak, deger in (needs or {}).items():
            if float(deger) > float(tavan.get(kaynak, 0.0)):
                return False
        return True

    def _make_chunk(self, sira: int, steps: Sequence[WorkStep],
                    needs: Dict[str, float], seviye: int,
                    not_: str = "") -> WorkChunk:
        """Parça nesnesi kurar."""
        parca = WorkChunk(id=f"c{sira:05d}", steps=list(steps),
                          needs=dict(needs))
        parca.reason = not_ or f"bağımlılık katmanı {seviye}"
        return parca

    def _layers(self, steps: Sequence[WorkStep]) -> List[List[WorkStep]]:
        """
        Adımları bağımlılık katmanlarına ayırır.

        Örnek: A → B → C zinciri 3 katman; A ve B bağımsızsa 1 katman
        (ikisi paralel çalışabilir).
        """
        kalan = {adim.id: adim for adim in steps}
        biten: set = set()
        katmanlar: List[List[WorkStep]] = []
        # Güvenlik: döngüsel bağımlılıkta sonsuz döngüye girmemek için sınır
        for _ in range(len(steps) + 2):
            if not kalan:
                break
            hazir = [adim for adim in kalan.values()
                     if all(dep in biten or dep not in kalan
                            for dep in adim.depends_on)]
            if not hazir:
                # Kalan adımlar birbirine bağlı (döngü): tek katmana koy
                katmanlar.append(list(kalan.values()))
                break
            katmanlar.append(hazir)
            for adim in hazir:
                biten.add(adim.id)
                kalan.pop(adim.id, None)
        return katmanlar

    def _sum_needs(self, steps: Sequence[WorkStep]) -> Dict[str, float]:
        """Parçanın toplam kaynak ihtiyacı."""
        toplam: Dict[str, float] = {}
        for adim in steps:
            for kaynak, deger in (adim.needs or {}).items():
                toplam[kaynak] = toplam.get(kaynak, 0.0) + float(deger)
        return toplam

    # ─── Yürütme ────────────────────────────────────────────────────────

    def register(self, kind: str, handler: Callable[[WorkStep], Any]) -> None:
        """Adım türü için işleyici kaydeder."""
        self._handlers[str(kind)] = handler

    def run(self, *, snapshot: Optional[ResourceSnapshot] = None,
            max_chunks: int = 0,
            on_event: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Planı yürütür: hazır parçaları çalıştırır, sığmayanı kuyruğa alır.

        Args:
            snapshot: Kaynak durumu (None = her turda canlı oku)
            max_chunks: En fazla kaç parça çalışsın (0 = hepsi)
            on_event: Her parça sonrası bilgilendirme

        Returns:
            Özet rapor
        """
        self._counter["runs"] += 1
        basladi = time.time()
        kuyruk = [p for p in self.chunks if p.status in (PENDING, BLOCKED)]
        calisan = 0

        self._journal.append({
            "olay": "başladı", "amaç": self._objective,
            "adım": len(self.steps), "parça": len(self.chunks),
            "bağımlılık_katmanı": len({p.reason for p in self.chunks}),
            "zaman": time.time(),
        })

        while kuyruk:
            if max_chunks and calisan >= max_chunks:
                break
            # Kaynak bütçesi (her turda taze — sistem değişebilir)
            anlik = snapshot or ResourceSnapshot.live()
            butce = self.governor.plan(anlik)

            # ── Hazır parçaları seç ──
            hazir = [p for p in kuyruk if self._deps_ok(p)]
            if not hazir:
                # Bağımlılığı gelmemiş; kalanlar bekliyor
                for parca in kuyruk:
                    parca.status = BLOCKED
                    parca.reason = "bağımlılık bekleniyor"
                break

            # Kaynağa sığanları çalıştır (en fazla `parallel` kadar)
            sigar = [p for p in hazir
                     if self.governor.fits(p.needs, butce, anlik)]
            if not sigar:
                # Hiçbiri sığmıyor → kuyruğa geri koy, kaynak bekle
                self._counter["chunks_deferred"] += len(hazir)
                for parca in hazir:
                    parca.status = BLOCKED
                    parca.reason = (
                        f"kaynağa sığmıyor (ihtiyaç {parca.needs}, "
                        f"bütçe CPU %{butce.cpu_percent:.1f}/"
                        f"RAM {butce.ram_mb:.0f}MB/VRAM {butce.vram_mb:.0f}MB)"
                    )
                    self._journal.append({
                        "olay": "ertelendi", "parça": parca.id,
                        "ihtiyaç": parca.needs, "bütçe": butce.to_dict(),
                        "zaman": time.time(),
                    })
                logger.info(
                    f"{len(hazir)} parça kaynağa sığmadı — kuyrukta bekliyor "
                    f"({butce.limiting} kısıtlıyor)")
                break

            # ── Sığanları **paralel** çalıştır ──
            #
            # IDM'in özü budur: boştaki kapasiteyi besleyen kadar parça aynı
            # anda çalışır. `butce.parallel` kaç parçanın kaynağa sığdığını
            # söyler; o kadarını eşzamanlı başlatırız.
            #
            # İş parçacıkları tercih edilir çünkü adımlar çoğunlukla G/Ç,
            # ağ veya **harici süreç** bekler; bu bekleme sırasında GIL
            # serbest kalır ve paralellik gerçekten kazanç sağlar. Saf Python
            # aritmetiği yoğun işlerde kazanç sınırlıdır (GIL), ama kaynak
            # nezaketi açısından yine zarar vermez.
            parti = sigar[:max(1, butce.parallel)]
            if max_chunks:
                parti = parti[:max(0, max_chunks - calisan)]
            if not parti:
                break
            self._run_batch(parti, on_event=on_event, budget=butce)
            for parca in parti:
                kuyruk.remove(parca)
                calisan += 1

        # ── Kalan parçaları işaretle ──
        for parca in kuyruk:
            if parca.status == PENDING:
                parca.status = BLOCKED
                parca.reason = parca.reason or "sıra bekliyor"

        rapor = self.report()
        rapor["duration_s"] = round(time.time() - basladi, 2)
        self._journal.append({
            "olay": "bitti", "özet": {
                "adım_bitti": rapor["steps_done"],
                "adım_başarısız": rapor["steps_failed"],
                "parça_bitti": rapor["chunks_done"],
                "parça_bekleyen": rapor["chunks_blocked"],
            },
            "süre_s": rapor["duration_s"], "zaman": time.time(),
        })
        return rapor

    def _deps_ok(self, chunk: WorkChunk) -> bool:
        """
        Parçanın **dış** bağımlılıkları tamamlandı mı?

        ⚠ Yalnızca parça dışındaki bağımlılıklar beklenir. Parça içi sıra
        zaten `_run_chunk` tarafından adım adım korunur; iç bağımlılıkları
        da beklemek parçayı **hiç çalışmaz** hâle getirir.

        Ölçülen hata: `[s000] → [s001 → s002 → s003]` zincirinde parça
        içindeki `s001`, `s000`'ı beklediği için parça "bağımlılık
        bekleniyor" diye işaretleniyor ve 40 adımın **hiçbiri**
        çalışmıyordu.
        """
        icerdekiler = {adim.id for adim in chunk.steps}
        for adim in chunk.steps:
            for dep in adim.depends_on:
                if dep in icerdekiler:
                    continue          # parça içi — sıralı yürütme hâlleder
                hedef = next((s for s in self.steps if s.id == dep), None)
                if hedef is not None and hedef.status not in (DONE, SKIPPED):
                    return False
        return True

    def _run_batch(self, chunks: Sequence[WorkChunk], *,
                   on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
                   budget: Optional[ChunkBudget] = None) -> None:
        """
        Parçaları **eşzamanlı** çalıştırır.

        `butce.parallel` kadar iş parçacığı açılır; her parça kendi
        parçacığında yürür ve bitince günlüğe yazılır. Bir parça çökse bile
        diğerleri etkilenmez (hata parça içinde yakalanır).

        Args:
            chunks: Çalıştırılacak parçalar
            on_event: Parça sonrası bilgilendirme
            budget: Kullanılan bütçe (günlük için)
        """
        import threading

        # Tek parça varsa iş parçacığı açmaya gerek yok
        if len(chunks) == 1:
            self._run_chunk(chunks[0])
            if on_event is not None:
                try:
                    on_event({"parça": chunks[0].to_dict(),
                              "bütçe": (budget.to_dict() if budget else {})})
                except Exception:
                    pass
            return

        hatalar: List[BaseException] = []

        def kos(parca: WorkChunk) -> None:
            """Tek parçayı çalıştırır (iş parçacığı gövdesi)."""
            try:
                self._run_chunk(parca)
            except BaseException as exc:   # noqa: BLE001 - parçayı izole et
                hatalar.append(exc)
                parca.status = FAILED
                parca.reason = f"parça çöktü: {exc}"
                logger.warning(f"Parça çöktü ({parca.id}): {exc}")

        isler = [threading.Thread(target=kos, args=(p,), daemon=True,
                                  name=f"nc-chunk-{p.id}") for p in chunks]
        for is_ in isler:
            is_.start()
        for is_ in isler:
            is_.join()
        for parca in chunks:
            if on_event is not None:
                try:
                    on_event({"parça": parca.to_dict(),
                              "bütçe": (budget.to_dict() if budget else {})})
                except Exception:
                    pass

    def _run_chunk(self, chunk: WorkChunk) -> None:
        """Bir parçayı çalıştırır ve günlüğe yazar."""
        # Tamamlanmış parça yeniden çalıştırılmaz (idempotentlik)
        if chunk.status == DONE and all(a.status in (DONE, SKIPPED)
                                        for a in chunk.steps):
            return
        chunk.status = RUNNING
        chunk.started = time.time()
        for adim in chunk.steps:
            self._run_step(adim)
        chunk.finished = time.time()
        # Bekleyen adım parçayı **başarısız** yapmaz; parça "bekliyor" olur
        bekleyen = any(a.status == WAITING for a in chunk.steps)
        basarili = all(a.status in (DONE, SKIPPED, WAITING) for a in chunk.steps)
        if bekleyen:
            chunk.status = WAITING
        else:
            chunk.status = DONE if basarili else FAILED
        if basarili:
            with self._lock:
                self._counter["chunks_done"] += 1
        self._journal.append({
            "olay": "parça", "parça": chunk.id, "durum": chunk.status,
            "adımlar": [a.id for a in chunk.steps],
            "süre_s": round(chunk.duration_s, 3),
            "ihtiyaç": chunk.needs,
            "zaman": time.time(),
        })

    def _run_step(self, step: WorkStep) -> None:
        """
        Tek adımı çalıştırır; hatayı yutar ve kaydeder.

        **Idempotent:** zaten tamamlanmış (`bitti`) veya atlanmış adım
        yeniden çalıştırılmaz. Bu, `run()` çağrısının birden fazla kez
        yapılmasını (kuyruk ısrarı, `append_plan` sonrası yeniden çalıştırma)
        güvenli kılar. Aksi hâlde bir dosyayı iki kez yazan veya bir süreci
        iki kez başlatan adımlar **istenmeyen yan etki** üretirdi.
        """
        if step.status in (DONE, SKIPPED):
            return
        step.status = RUNNING
        step.started = time.time()
        isleyici = self._handlers.get(step.kind)
        try:
            if isleyici is None:
                # İşleyici yok: adım başarılı sayılır (kuru çalıştırma)
                step.result = "işleyici yok — kuru çalıştırma"
                step.status = DONE
            else:
                cikti = isleyici(step)
                step.result = ("" if cikti is None else str(cikti))[:2000]
                step.status = DONE
            with self._lock:
                self._counter["steps_done"] += 1
        except WorkWaiting as exc:
            # Dış girdi bekleniyor — **başarısızlık değil**
            step.status = WAITING
            step.result = exc.reason[:500]
            step.error = ""
            with self._lock:
                self._counter["steps_waiting"] = (
                    self._counter.get("steps_waiting", 0) + 1)
            logger.info(f"Adım bekliyor ({step.id}): {exc.reason[:100]}")
        except Exception as exc:
            step.error = f"{type(exc).__name__}: {exc}"
            step.status = FAILED
            with self._lock:
                self._counter["steps_failed"] += 1
            logger.warning(f"Adım başarısız ({step.id}): {step.error[:120]}")
        step.finished = time.time()
        self._journal.append({
            "olay": "adım", "adım": step.id, "başlık": step.title[:120],
            "tür": step.kind, "durum": step.status,
            "süre_s": round(step.duration_s, 3),
            "sonuç": step.result[:300], "hata": step.error[:200],
            "zaman": time.time(),
        })

    # ─── Raporlama ve Algoritma ─────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Durum özeti."""
        sayim: Dict[str, int] = {}
        for adim in self.steps:
            sayim[adim.status] = sayim.get(adim.status, 0) + 1
        return {
            "objective": self._objective[:200],
            "steps": len(self.steps),
            "chunks": len(self.chunks),
            "steps_done": sayim.get(DONE, 0),
            "steps_failed": sayim.get(FAILED, 0),
            "steps_pending": sayim.get(PENDING, 0),
            "chunks_done": sum(1 for c in self.chunks if c.status == DONE),
            "chunks_blocked": sum(1 for c in self.chunks
                                  if c.status in (BLOCKED, PENDING)),
            "status_counts": sayim,
            "journal_records": len(self._journal.records()),
            "counter": dict(self._counter),
        }

    def algorithm(self, *, limit: int = 0) -> str:
        """
        Yapılan işi **algoritma** olarak yeniden üretir.

        Kullanıcı isteği: "en son yapılan projenin algoritması ile detaylı
        bilgi sahibi olur." Bu metot, planı ve gerçekleşen süreleri birleştirip
        okunabilir bir yöntem listesi üretir.

        Args:
            limit: En fazla kaç adım yazılsın (0 = hepsi)

        Returns:
            Adım adım algoritma metni
        """
        satirlar: List[str] = []
        satirlar.append(f"AMAÇ: {self._objective}")
        satirlar.append(f"ÖLÇEK: {len(self.steps)} adım, {len(self.chunks)} parça")
        satirlar.append("")
        adimlar = self.steps if not limit else self.steps[:limit]
        for index, adim in enumerate(adimlar, start=1):
            bagim = (f" (bağımlı: {', '.join(adim.depends_on)})"
                     if adim.depends_on else "")
            satirlar.append(
                f"{index:4}. [{adim.kind}] {adim.title}{bagim}"
            )
            if adim.needs:
                ihtiyac = ", ".join(f"{k}={v:g}" for k, v in adim.needs.items())
                satirlar.append(f"      kaynak: {ihtiyac}")
            if adim.status == DONE:
                satirlar.append(
                    f"      sonuç: başarılı ({adim.duration_s:.2f} sn)")
            elif adim.status == FAILED:
                satirlar.append(f"      sonuç: HATA — {adim.error[:100]}")
            else:
                satirlar.append(f"      sonuç: {adim.status}")
        if limit and len(self.steps) > limit:
            satirlar.append(f"… ({len(self.steps) - limit} adım daha)")
        return "\n".join(satirlar)

    def resume_state(self) -> Dict[str, Any]:
        """
        Kesintiden sonra kaldığı yeri bildirir.

        Günlükten tamamlanmış adımları okur; planlayıcı bunları atlayarak
        devam edebilir.
        """
        kayitlar = self._journal.load()
        biten: set = set()
        for kayit in kayitlar:
            if kayit.get("olay") == "adım" and kayit.get("durum") == DONE:
                biten.add(str(kayit.get("adım")))
        return {
            "journal_records": len(kayitlar),
            "completed_steps": sorted(biten),
            "completed_count": len(biten),
            "objective": next((k.get("amaç", "") for k in kayitlar
                               if k.get("olay") == "başladı"), ""),
        }

    def run_until_done(self, *, snapshot: Optional[ResourceSnapshot] = None,
                       max_wait_s: float = 3600.0,
                       poll_s: float = 20.0,
                       on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
                       stop_event: Optional[threading.Event] = None
                       ) -> Dict[str, Any]:
        """
        **Kaynak açılana kadar bekleyip** kalan parçaları çalıştırır.

        Neden gerekli? `run()` tek geçiştir: kaynak yetmezse parçalar
        `kaynak bekliyor` durumunda kalır ve kullanıcı elle tekrar çağırmak
        zorunda kalır. Ölçüldü: 8 turluk bir işte turların 7'si kaynak
        yüzünden atlanmıştı. Bu metot kuyruğu **kendiliğinden** eritir.

        Bekleme mantığı:
            * Her turda kaynak yeniden ölçülür (`snapshot` verilmediyse).
            * Hiçbir parça ilerlemiyorsa `poll_s` kadar beklenir.
            * `max_wait_s` aşılırsa durur ve kalanı **raporlar** (sessizce
              pes etmez).

        Args:
            snapshot: Sabit kaynak profili (None = her turda canlı ölç)
            max_wait_s: Toplam bekleme sınırı (saniye)
            poll_s: İlerleme olmayınca bekleme aralığı
            on_event: Parça sonrası bilgilendirme
            stop_event: Dışarıdan durdurma (`threading.Event`)

        Returns:
            Özet rapor (`waited_s`, `still_blocked` alanlarıyla)
        """
        basladi = time.time()
        beklenen = 0.0
        turlar = 0
        son_ilerleme = time.time()
        onceki_biten = -1

        while True:
            turlar += 1
            if stop_event is not None and stop_event.is_set():
                break
            rapor = self.run(snapshot=snapshot, on_event=on_event)
            biten = rapor["steps_done"] + rapor["steps_failed"]

            # Tüm adımlar bitti mi?
            if biten >= rapor["steps"]:
                break
            # Kaynak bekleyen parça yoksa (bağımlılık sorunu) dur
            bekleyen = [p for p in self.chunks if p.status == BLOCKED
                        and "kaynağa sığmıyor" in (p.reason or "")]
            if not bekleyen:
                break

            # İlerleme var mı?
            if biten == onceki_biten:
                gecen = time.time() - son_ilerleme
                if gecen + poll_s > max_wait_s:
                    logger.warning(
                        f"Kaynak {max_wait_s:.0f} sn içinde açılmadı — "
                        f"{len(bekleyen)} parça kuyrukta kalıyor"
                    )
                    break
                time.sleep(poll_s)
                beklenen += poll_s
            else:
                onceki_biten = biten
                son_ilerleme = time.time()

        rapor = self.report()
        rapor["waited_s"] = round(beklenen, 1)
        rapor["rounds"] = turlar
        rapor["total_s"] = round(time.time() - basladi, 1)
        rapor["still_blocked"] = [
            {"id": p.id, "reason": p.reason[:160],
             "needs": {k: round(v, 1) for k, v in p.needs.items()}}
            for p in self.chunks
            if p.status in (BLOCKED, PENDING)
        ]
        self._journal.append({
            "olay": "kuyruk_bitti",
            "tur": turlar, "bekleme_s": rapor["waited_s"],
            "kalan_parça": len(rapor["still_blocked"]),
            "zaman": time.time(),
        })
        return rapor

    def append_plan(self, objective: str,
                    steps: Sequence[WorkStep]) -> List[WorkChunk]:
        """
        Var olan plana **yeni adımlar** ekler.

        Neden? Uzun işler parça parça planlanır: kullanıcı yeni bir aşama
        istediğinde tüm planı yeniden kurmak, tamamlanmış adımları
        kaybettirir. Bu metot yalnızca **ekler**; tamamlanmış adımlar
        korunur.

        Args:
            objective: Amaç (boşsa eskisi korunur)
            steps: Eklenecek adımlar

        Returns:
            Güncellenmiş parça listesi
        """
        with self._lock:
            tamamlanan = [a for a in self.steps
                          if a.status in (DONE, SKIPPED)]
            mevcut_id = {a.id for a in self.steps}
            kalan_yeni = [a for a in steps if a.id not in mevcut_id]
            if not kalan_yeni:
                return self.chunks
            hepsi = tamamlanan + [
                a for a in self.steps if a.status not in (DONE, SKIPPED)
            ] + kalan_yeni
            amac = objective or self._objective
        logger.info(
            f"Plana {len(kalan_yeni)} adım eklendi "
            f"({len(tamamlanan)} tamamlanmış korundu)"
        )
        return self.plan(amac, hepsi)

    def deferred_summary(self) -> Dict[str, Any]:
        """
        Kuyrukta bekleyenlerin özeti — kullanıcıya neyin beklediğini söyler.
        """
        bekleyen = [p for p in self.chunks if p.status in (BLOCKED, PENDING)]
        if not bekleyen:
            return {"count": 0, "steps": 0, "reason": "", "reasons": {}}
        gerekceler: Dict[str, int] = {}
        for parca in bekleyen:
            anahtar = (parca.reason or "bilinmiyor")[:80]
            gerekceler[anahtar] = gerekceler.get(anahtar, 0) + 1
        en_sik = max(gerekceler.items(), key=lambda x: x[1])
        return {
            "count": len(bekleyen),
            "steps": sum(len(p.steps) for p in bekleyen),
            "reason": en_sik[0],
            "reasons": gerekceler,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Sayaçlar ve yönetişim durumu."""
        return {
            **self._counter,
            "steps": len(self.steps),
            "chunks": len(self.chunks),
            "governor": self.governor.get_stats(),
            "parallel_now": self.governor.plan().parallel,
        }


__all__ = [
    "WorkEngine", "WorkStep", "WorkChunk", "WorkJournal",
    "JOURNAL_FILE", "PENDING", "READY", "RUNNING", "DONE", "FAILED",
    "SKIPPED", "BLOCKED", "WAITING", "WorkWaiting",
]
