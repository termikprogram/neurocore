"""
Ajan Köprüsü — Görevleri Parçalı Motora Bağlar
===============================================
Ajanın işlerini `WorkStep`'e çevirir; böylece uzun görevler
**parçalanır**, **kuyruğa** alınır ve **kesintiden sonra** devam eder.

Neden gerekli?
    Ajan döngüsü tek bir `run()` çağrısında en fazla `max_steps` (varsayılan
    24) adım atar ve süre sınırı vardır. Uzun bir görev (ör. "8 milyon adımlı
    projeyi üret") bu sınırı aşar ve **sıfırdan** başlamak gerekir.

    Parçalı motor ise işi böler, kaynağa sığdıkça çalıştırır ve her adımı
    günlüğe yazar. İkisini birleştirmek şu kazançları verir:

        * **Uzun görev** tek oturumda bitmese bile ilerleme kaydedilir
        * **Kaynak nezaketi** — ağır adımlar kuyruğa alınır, sistem kilitlenmez
        * **Paralellik** — bağımsız adımlar eşzamanlı çalışır
        * **İzlenebilirlik** — hangi araç, hangi parametre, ne sonuç, kaç sn

Kullanım:

    bridge = AgentBridge(workspace=".neurocore")
    bridge.initialize(loop=agent_loop, thinking=thinking_loop)
    adimlar = bridge.steps_for("Proje üret", plan=[
        {"title": "Analiz", "tool": "code.workspace_map", "args": {}},
        {"title": "Yaz", "tool": "code.create_file",
         "args": {"path": "a.py", "content": "..."}, "needs_previous": True},
    ])
    bridge.engine.plan("Proje üret", adimlar)
    rapor = bridge.engine.run()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from neurocore.core.governor import ResourceSnapshot
from neurocore.core.work import WorkEngine, WorkStep
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.bridge")

#: Araç türüne göre kaynak tahmini.
#:
#: Değerler **ölçüme** dayanır: kod yazma hafif, test/derleme orta, eğitim ve
#: render ağır. Yanlış tahmin ölümcül değildir (motor yine kuyruğa alır) ama
#: doğru tahmin paralelliği artırır.
TOOL_RESOURCES: Dict[str, Dict[str, float]] = {
    "code.read_file": {"cpu": 0.5, "ram_mb": 20.0},
    "code.list_directory": {"cpu": 0.5, "ram_mb": 10.0},
    "code.workspace_map": {"cpu": 1.0, "ram_mb": 40.0},
    "code.create_file": {"cpu": 1.0, "ram_mb": 30.0},
    "code.edit_file": {"cpu": 1.0, "ram_mb": 30.0},
    "code.search": {"cpu": 1.5, "ram_mb": 60.0},
    "code.run_tests": {"cpu": 6.0, "ram_mb": 300.0},
    "shell.run": {"cpu": 6.0, "ram_mb": 300.0},
    "shell.powershell": {"cpu": 6.0, "ram_mb": 300.0},
    "vision.screenshot": {"cpu": 2.0, "ram_mb": 120.0, "vram_mb": 60.0},
    "godot.run_script": {"cpu": 8.0, "ram_mb": 500.0},
    "blender.run_script": {"cpu": 10.0, "ram_mb": 700.0, "vram_mb": 200.0},
    "blender.render": {"cpu": 40.0, "ram_mb": 1500.0, "vram_mb": 800.0},
}

#: Bilinmeyen araç için güvenli varsayılan
DEFAULT_RESOURCES: Dict[str, float] = {"cpu": 2.0, "ram_mb": 100.0}

#: Araç türüne göre tahmini süre (saniye)
TOOL_ESTIMATES: Dict[str, float] = {
    "code.read_file": 0.5, "code.list_directory": 0.5,
    "code.workspace_map": 1.5, "code.create_file": 1.0,
    "code.edit_file": 1.0, "code.search": 3.0, "code.run_tests": 30.0,
    "shell.run": 10.0, "vision.screenshot": 3.0,
    "blender.render": 120.0, "blender.run_script": 20.0,
}


@dataclass
class PlannedStep:
    """Ajan için planlanan tek iş."""
    title: str
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    kind: str = "tool"
    #: Önceki adım tamamlanmadan başlamasın
    needs_previous: bool = False
    needs: Dict[str, float] = field(default_factory=dict)
    estimate_s: float = 0.0
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "title": self.title, "tool": self.tool, "kind": self.kind,
            "needs": dict(self.needs), "estimate_s": self.estimate_s,
            "depends_on": list(self.depends_on),
        }


def resources_for_tool(tool: str) -> Dict[str, float]:
    """
    Aracın kaynak ihtiyacını tahmin eder.

    Eşleşme sırası: tam ad → nokta öneki → varsayılan. Böylece yeni araçlar
    eklendiğinde motor yine çalışır (yalnızca tahmin kaba olur).
    """
    ad = str(tool or "")
    if ad in TOOL_RESOURCES:
        return dict(TOOL_RESOURCES[ad])
    onek = ad.split(".")[0] if "." in ad else ad
    for anahtar, deger in TOOL_RESOURCES.items():
        if anahtar.startswith(f"{onek}."):
            return dict(deger)
    return dict(DEFAULT_RESOURCES)


def estimate_for_tool(tool: str) -> float:
    """Aracın tahmini süresi (saniye)."""
    ad = str(tool or "")
    if ad in TOOL_ESTIMATES:
        return TOOL_ESTIMATES[ad]
    return 2.0


class AgentBridge:
    """
    Ajan ile parçalı iş motoru arasındaki köprü.

    Kullanım:
        bridge = AgentBridge(workspace=".neurocore")
        bridge.initialize(loop=loop, thinking=thinking)
        bridge.plan_steps("Görev", [...])        # WorkStep listesi üretir
        rapor = bridge.run()
    """

    name = "agent.bridge"

    def __init__(self, *, workspace: str = "", chunk_ratio: float = 0.05,
                 chunk_steps: int = 6) -> None:
        self.engine = WorkEngine(workspace=workspace,
                                 chunk_ratio=chunk_ratio,
                                 chunk_steps=chunk_steps)
        self._loop: Any = None
        self._thinking: Any = None
        self._tool_runner: Optional[Callable[[str, Dict[str, Any]], Any]] = None
        self._initialized = False
        #: Onay bekleyen adımlar: {adım_id: {"tool","args","reason","at"}}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}
        #: Onaylanan adımlar (yeniden çalıştırılacak)
        self._approved: set = set()
        #: Onay reddedilen adımlar (atlanacak)
        self._denied: set = set()
        #: Onay beklerken en fazla kaç saniye
        self.approval_wait_s: float = 300.0
        #: İç kullanım: `_call_tool` ham `ToolResult` döndürsün mü?
        self._raw_result_expected = False
        self._workspace = workspace
        self._counter: Dict[str, int] = {
            "plans": 0, "steps_planned": 0, "tool_calls": 0, "tool_errors": 0,
            "thinking_calls": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, *, loop: Any = None, thinking: Any = None,
                   tool_runner: Optional[Callable[[str, Dict[str, Any]], Any]] = None
                   ) -> None:
        """
        Köprüyü başlatır.

        Args:
            loop: Ajan döngüsü (araç kayıt defteri taşır)
            thinking: Düşünme döngüsü (yaklaşım seçimi için)
            tool_runner: Özel araç çalıştırıcı (None = kayıt defterinden bul)
        """
        # ⚠ Erken dönüşte bile `loop`/`thinking` **güncellenmeli**.
        #
        # Ölçülen hata: köprü uygulama açılışında `loop=None` ile kurulur
        # (ajan döngüsü henüz yoktur), sonra döngü hazır olunca yeniden
        # bağlanır. Eski kod "zaten kurulu" diyip erken dönüyordu ve araç
        # kayıt defteri **hiç bağlanmıyordu** — her adım "araç kayıt defteri
        # yok" diyordu.
        if loop is not None:
            self._loop = loop
        if thinking is not None:
            self._thinking = thinking
        if tool_runner is not None:
            self._tool_runner = tool_runner
        if self._initialized:
            return
        # Motorun tür işleyicileri
        self.engine.register("tool", self._run_tool_step)
        self.engine.register("think", self._run_think_step)
        self.engine.register("generic", lambda adim: "atlandı (işleyici yok)")
        self._initialized = True
        logger.info(
            f"Ajan köprüsü hazır — araç kaynak tahmini "
            f"{len(TOOL_RESOURCES)} tür için tanımlı"
            f"{' (döngüye bağlı)' if self._loop is not None else ' (döngü bekliyor)'}"
        )

    def shutdown(self) -> None:
        """Kaynakları bırakır."""
        self._loop = None
        self._thinking = None

    def get_stats(self) -> Dict[str, Any]:
        """Sayaçlar."""
        return {
            **self._counter,
            "engine": self.engine.get_stats(),
        }

    # ─── Planlama ───────────────────────────────────────────────────────

    def plan_steps(self, objective: str,
                   plan: Sequence[Dict[str, Any]]) -> List[WorkStep]:
        """
        Sözlük planını `WorkStep` listesine çevirir.

        Plan biçimi:
            [{"title": "Analiz", "tool": "code.workspace_map", "args": {}},
             {"title": "Yaz", "tool": "code.create_file",
              "args": {...}, "needs_previous": True},
             {"title": "Seç", "kind": "think",
              "candidates": [...]}]

        `needs_previous=True` olan adım, bir öncekine bağımlı olur (zincir);
        aksi hâlde bağımsız sayılır ve **paralel** çalışabilir.

        Args:
            objective: Görevin amacı (günlüğe yazılır)
            plan: Adım sözlükleri

        Returns:
            `WorkStep` listesi
        """
        self._counter["plans"] += 1
        adimlar: List[WorkStep] = []
        onceki: Optional[str] = None

        for index, ham in enumerate(plan):
            kimlik = str(ham.get("id") or f"t{index:03d}")
            arac = str(ham.get("tool") or "")
            tur = str(ham.get("kind") or ("think" if ham.get("candidates")
                                          else "tool"))
            ihtiyac = ham.get("needs") or resources_for_tool(arac)
            tahmin = float(ham.get("estimate_s")
                           or estimate_for_tool(arac))

            bagim: List[str] = list(ham.get("depends_on") or [])
            if ham.get("needs_previous") and onceki:
                bagim.append(onceki)

            adim = WorkStep(
                id=kimlik,
                title=str(ham.get("title") or f"Adım {index + 1}"),
                kind=tur,
                needs=dict(ihtiyac),
                depends_on=bagim,
                payload={
                    "tool": arac,
                    "args": dict(ham.get("args") or {}),
                    "candidates": list(ham.get("candidates") or []),
                    "prompt": str(ham.get("prompt") or ""),
                },
                estimate_s=tahmin,
            )
            adimlar.append(adim)
            onceki = kimlik

        self._counter["steps_planned"] += len(adimlar)
        self.engine.plan(objective, adimlar)
        return adimlar

    # ─── Yürütme ────────────────────────────────────────────────────────

    def _run_tool_step(self, step: WorkStep) -> str:
        """
        Araç adımını çalıştırır.

        Araç, ajan döngüsünün kayıt defterinden bulunur. Sonuç kısa bir
        metne indirgenir (günlüğe yazılacak).
        """
        arac = str(step.payload.get("tool") or "")
        argumanlar = dict(step.payload.get("args") or {})
        if not arac:
            return "araç belirtilmemiş — atlandı"

        # ── Kullanıcı reddettiyse atla ──
        if step.id in self._denied:
            self._denied.discard(step.id)
            return "kullanıcı reddetti — adım atlandı"

        self._counter["tool_calls"] += 1
        try:
            sonuc = self._call_tool(arac, argumanlar)
        except Exception as exc:
            # Onay gerekiyorsa **başarısızlık değil**, bekleme durumudur
            if self._needs_approval(exc):
                return self._await_approval(step, arac, argumanlar, exc)
            self._counter["tool_errors"] += 1
            raise RuntimeError(f"{arac}: {exc}") from exc
        return self._summarize(sonuc)

    @staticmethod
    def _needs_approval(exc: BaseException) -> bool:
        """İstisna onay gereksiniminden mi kaynaklandı?"""
        metin = str(exc).lower()
        return ("onay gerekir" in metin or "onay gerekiyor" in metin
                or "approval" in metin or "denetimli modunda" in metin)

    def _await_approval(self, step: WorkStep, tool: str,
                        args: Dict[str, Any], exc: BaseException) -> str:
        """
        Onay bekler; onay gelirse adımı **yeniden çalıştırır**.

        ⚠ Neden başarısızlık değil? Denetimli modda yazma/çalıştırma
        araçları onay ister. Onay bekleyen adımı "başarısız" saymak, toplu
        işlerde **her yazma adımının** çökmüş görünmesine yol açıyordu
        (ölçüldü: 3 adımlık planın 2'si başarısız). Oysa iş **bekliyor**,
        bozulmuş değil.

        Bekleme sırasında adım `kaynak bekliyor` durumunda kalır ve
        kullanıcı onayladığında kaldığı yerden devam eder.
        """
        self._pending_approvals[step.id] = {
            "tool": tool, "args": args, "reason": str(exc)[:200],
            "at": time.time(), "title": step.title,
        }
        self._counter["approval_waits"] = self._counter.get("approval_waits", 0) + 1
        logger.info(
            f"Adım onay bekliyor: {step.title[:50]} ({tool}) — "
            f"/api/work/approve ile onaylayın"
        )

        # Kısa süre bekle (kullanıcı arayüzden onaylayabilir)
        bitis = time.time() + self.approval_wait_s
        while time.time() < bitis:
            if step.id in self._denied:
                self._denied.discard(step.id)
                self._pending_approvals.pop(step.id, None)
                return "kullanıcı reddetti — adım atlandı"
            if step.id in self._approved:
                self._approved.discard(step.id)
                self._pending_approvals.pop(step.id, None)
                self._counter["approvals_used"] = (
                    self._counter.get("approvals_used", 0) + 1)
                logger.info(f"Onay alındı, adım yeniden çalışıyor: {tool}")
                # ⚠ Yeniden çağrı **ham** yapılır ve sonuç doğrudan açılır.
                # `_call_tool`'u tekrar çağırmak, kayıt defteri onayı yeni
                # tükettiği için **yine onay isteyip sonsuz döngü** kurardı
                # (ölçüldü: süreç askıda kaldı).
                try:
                    sonuc = self._call_tool_raw(tool, args)
                except Exception as exc:
                    raise RuntimeError(f"{tool}: {exc}") from exc
                return f"[onaylandı] {self._summarize(self._unwrap(sonuc))}"
            time.sleep(0.5)

        # Süre aşıldı: adım **bekliyor** olarak kalır (başarısız DEĞİL).
        # `WorkWaiting` fırlatmak, motorun adımı `onay bekliyor` durumuna
        # almasını sağlar; böylece toplu iş "çöktü" görünmez.
        self._counter["approvals_pending"] = len(self._pending_approvals)
        from neurocore.core.work import WorkWaiting

        raise WorkWaiting(
            f"{tool}: onay bekleniyor — /api/work/approve ile onaylayıp "
            f"işi sürdürebilirsiniz")

    def approve(self, step_id: str) -> bool:
        """Bekleyen adımı onaylar (iş kaldığı yerden devam eder)."""
        if step_id not in self._pending_approvals:
            return False
        self._approved.add(step_id)
        logger.info(f"Onay kaydedildi: {step_id}")
        return True

    def deny(self, step_id: str) -> bool:
        """Bekleyen adımı reddeder (adım atlanır)."""
        if step_id not in self._pending_approvals:
            return False
        self._denied.add(step_id)
        logger.info(f"Red kaydedildi: {step_id}")
        return True

    def pending_approvals(self) -> List[Dict[str, Any]]:
        """Onay bekleyen adımlar (arayüz gösterir)."""
        return [
            {"step_id": kimlik, **{k: v for k, v in bilgi.items()
                                   if k != "at"},
             "waited_s": round(time.time() - float(bilgi.get("at", 0)), 1)}
            for kimlik, bilgi in self._pending_approvals.items()
        ]

    def _call_tool(self, tool: str, args: Dict[str, Any]) -> Any:
        """
        Aracı kayıt defterinden bulup çağırır.

        **`registry.call()` kullanılır** — `handler(**args)` doğrudan
        çağırmak yerine. Neden? `call()` şunları uygular:

            * özerklik politikası (okuma/yazma/çalıştırma sınırı)
            * onay akışı (`needs_approval`, bilet üretimi)
            * denetim günlüğü ve `ToolResult` sarmalama

        Doğrudan `handler` çağırmak bu güvenlik katmanlarını **atlar**.
        """
        runner = getattr(self, "_tool_runner", None)
        if callable(runner):
            return runner(tool, args)

        # ⚠ Ajan döngüsü kayıt defterini `_registry` (alt çizgili) altında
        # tutar; eski kod yalnızca `registry` arıyordu ve **her adım**
        # "araç kayıt defteri yok" diyordu.
        kayit = self._registry()
        if kayit is None:
            return f"araç kayıt defteri yok — {tool} çağrılmadı"

        # Tercih edilen yol: politikayı uygulayan `call`
        cagir = getattr(kayit, "call", None)
        if callable(cagir):
            sonuc = cagir(tool, dict(args))
            return self._unwrap(sonuc)

        # Yedek: `get` ile tanımı bul, işleyiciyi çağır
        al = getattr(kayit, "get", None)
        if callable(al):
            tanim = al(tool)
            isleyici = getattr(tanim, "handler", None)
            if callable(isleyici):
                return isleyici(**args)
        return f"araç bulunamadı: {tool}"

    def _call_tool_raw(self, tool: str, args: Dict[str, Any]) -> Any:
        """
        Aracı **sarmalamadan** çağırır (`ToolResult` döner).

        Onay sonrası yeniden çalıştırmada kullanılır: sonucu `_unwrap` ile
        kendimiz açarız, böylece onay denetimi **tekrar tetiklenmez**.
        """
        kayit = self._registry()
        if kayit is None:
            return f"araç kayıt defteri yok — {tool} çağrılmadı"
        cagir = getattr(kayit, "call", None)
        if callable(cagir):
            return cagir(tool, dict(args))
        al = getattr(kayit, "get", None)
        if callable(al):
            tanim = al(tool)
            isleyici = getattr(tanim, "handler", None)
            if callable(isleyici):
                return isleyici(**args)
        return f"araç bulunamadı: {tool}"

    def _registry(self) -> Any:
        """Ajan döngüsünün araç kayıt defterini bulur."""
        for ad in ("registry", "_registry"):
            kayit = getattr(self._loop, ad, None)
            if kayit is not None:
                return kayit
        return None

    @staticmethod
    def _unwrap(result: Any) -> Any:
        """
        `ToolResult` sarmalını açar.

        Araç sonucu `{"ok": …, "data": …, "error": …}` biçiminde gelir;
        günlüğe okunur bir metin yazmak için içeriği çıkarılır.

        ⚠ **Onay gereksinimi** burada yakalanır: `needs_approval=True` ise
        istisna fırlatılır ki `_run_tool_step` onu **bekleme** olarak
        işleyebilsin. Sözlük biçiminde de (`{"ok": False, "error": …,
        "needs_approval": True}`) aynı denetim yapılır; aksi hâlde onay
        bilgisi kaybolur ve adım yanlışlıkla "başarısız" görünürdü.
        """
        def _al(anahtar: str) -> Any:
            """Nesne ya da sözlükten alan okur."""
            deger = getattr(result, anahtar, None)
            if deger is None and isinstance(result, dict):
                deger = result.get(anahtar)
            return deger

        onay = bool(_al("needs_approval"))
        engel = bool(_al("blocked"))
        hata = _al("error")
        gerekce = _al("reason") or hata or "onay gerekir"
        if onay or (engel and hata):
            raise RuntimeError(f"onay gerekir — {str(gerekce)[:200]}")

        veri = _al("data")
        basarili = _al("ok")
        if basarili is False and hata:
            raise RuntimeError(str(hata)[:300])
        if veri is not None:
            return veri
        return result

    @staticmethod
    def _summarize(result: Any) -> str:
        """Sonucu günlük için kısa metne indirger."""
        if result is None:
            return "tamam"
        if isinstance(result, str):
            return result[:600]
        if isinstance(result, dict):
            for anahtar in ("text", "output", "stdout", "result", "message"):
                if anahtar in result and result[anahtar]:
                    return str(result[anahtar])[:600]
            return str(result)[:600]
        return str(result)[:600]

    def _run_think_step(self, step: WorkStep) -> str:
        """
        Düşünme adımını çalıştırır (yerel model adayları sıralar).

        Düşünme döngüsü yoksa aday listesinin ilki kullanılır ve bu durum
        sonuca **açıkça** yazılır (sessizce yanlış sonuç vermemek için).
        """
        adaylar = list(step.payload.get("candidates") or [])
        istem = str(step.payload.get("prompt") or step.title)
        if not adaylar:
            return "aday yok — düşünme atlandı"
        if self._thinking is None or len(adaylar) < 2:
            return f"düşünme yok, ilk aday: {adaylar[0][:300]}"
        self._counter["thinking_calls"] += 1
        try:
            sonuc = self._thinking.think(
                istem, lambda n: adaylar[:n])
        except Exception as exc:
            logger.warning(f"Düşünme çalışmadı: {exc}")
            return f"düşünme hatası, ilk aday: {adaylar[0][:300]}"
        if not sonuc.ok:
            return f"düşünme başarısız, ilk aday: {adaylar[0][:300]}"
        return (f"[{sonuc.source}/{sonuc.confidence}] "
                f"{sonuc.chosen[:500]}")

    # ─── Toplu Çalıştırma ───────────────────────────────────────────────

    def run(self, *, snapshot: Optional[ResourceSnapshot] = None,
            max_chunks: int = 0, max_wait_s: float = 0.0,
            poll_s: float = 15.0) -> Dict[str, Any]:
        """
        Planlanan işi çalıştırır.

        Args:
            snapshot: Kaynak durumu (None = canlı oku)
            max_chunks: En fazla kaç parça (0 = hepsi)
            max_wait_s: >0 ise kaynak/onay için bekler (`run_until_done`)
            poll_s: Bekleme aralığı

        Returns:
            Özet rapor (motorun raporu + köprü sayaçları)
        """
        if max_wait_s > 0:
            rapor = self.engine.run_until_done(
                snapshot=snapshot, max_wait_s=max_wait_s, poll_s=poll_s)
        else:
            rapor = self.engine.run(snapshot=snapshot, max_chunks=max_chunks)
        rapor["bridge"] = dict(self._counter)
        rapor["approvals"] = self.pending_approvals()
        return rapor

    def algorithm(self, **kwargs: Any) -> str:
        """Yapılan işi algoritma olarak döndürür."""
        return self.engine.algorithm(**kwargs)

    def resume_state(self) -> Dict[str, Any]:
        """Kesintiden sonra kaldığı yeri bildirir."""
        return self.engine.resume_state()

    def pending_steps(self) -> List[WorkStep]:
        """
        Henüz tamamlanmamış adımlar.

        Kesintiden sonra **kalan** işi yeniden planlamak için kullanılır.
        """
        return [a for a in self.engine.steps
                if a.status not in ("bitti", "atlandı")]

    def resume(self, *, snapshot: Optional[ResourceSnapshot] = None) -> Dict[str, Any]:
        """
        Kalan adımlarla **yeni bir motor** kurup devam eder.

        Neden yeni motor? Günlük diskte kalıcıdır ama bellek durumu kaybolur
        (süreç yeniden başlar). Tamamlanan adımlar günlükten okunur, kalanlar
        yeniden planlanır.
        """
        durum = self.resume_state()
        biten = set(durum.get("completed_steps") or [])
        kalan = [a for a in self.engine.steps if a.id not in biten]
        if not kalan:
            return {"ok": True, "resumed": False,
                    "message": "tüm adımlar zaten tamamlanmış"}
        amaç = self.engine._objective or "devam"
        yeni = WorkEngine(workspace=self._workspace,
                          chunk_ratio=self.engine.chunk_ratio,
                          chunk_steps=self.engine.chunk_steps)
        yeni.register("tool", self._run_tool_step)
        yeni.register("think", self._run_think_step)
        yeni.register("generic", lambda adim: "atlandı")
        # Bağımlılıkları tamamlanmış say (biten adımlar artık yok)
        for adim in kalan:
            adim.depends_on = [d for d in adim.depends_on if d not in biten]
            adim.status = "bekliyor"
        yeni.plan(amaç, kalan)
        self.engine = yeni
        self._counter["steps_planned"] += len(kalan)
        logger.info(f"Devam: {len(biten)} adım atlandı, {len(kalan)} adım kaldı")
        rapor = yeni.run(snapshot=snapshot)
        rapor["resumed"] = True
        rapor["skipped_steps"] = len(biten)
        return rapor


__all__ = [
    "AgentBridge", "PlannedStep", "resources_for_tool", "estimate_for_tool",
    "TOOL_RESOURCES", "DEFAULT_RESOURCES", "TOOL_ESTIMATES",
]
