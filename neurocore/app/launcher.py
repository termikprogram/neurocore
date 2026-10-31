"""
NeuroCore Masaüstü Uygulaması — Başlatıcı
=========================================
Ajanı kurar, yerel sunucuyu başlatır ve masaüstü penceresini açar.

Neden Edge `--app` modu?
    Bu makinede Node.js yok; bu yüzden Electron tabanlı bir kabuk kurulamaz.
    Windows 11'de **WebView2 çalışma zamanı ve Edge zaten kurulu** ve Edge'in
    `--app=<url>` bayrağı, adres çubuğu ve sekme çubuğu olmayan **gerçek bir
    uygulama penceresi** açar. Böylece sıfır ek kurulum ile masaüstü
    deneyimi elde edilir. Kurulum gerektiren bir kabuk istenirse
    `--browser` parametresiyle varsayılan tarayıcıya düşülebilir.

Kullanım::

    python -m neurocore.app                       # varsayılan: mevcut dizin
    python -m neurocore.app --workspace E:/proje  # belirli bir proje
    python -m neurocore.app --autonomy trusted    # başlangıç özerklik seviyesi
    python -m neurocore.app --port 8791 --no-window   # yalnızca sunucu
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading


def _utf8_konsol() -> None:
    """
    ⚠ Windows konsolu varsayılan olarak cp1254/cp1252 olabilir; kutu çizim
    karakterleri (─, ·, ı) UnicodeEncodeError ile uygulamayı çökertiyordu
    (ölçüldü: temiz kurulumda `python -m neurocore.app` → charmap hatası).
    stdout/stderr'i UTF-8'e çevirir; konsol yoksa (pythonw) sessizce geçer.
    """
    for akis in (sys.stdout, sys.stderr):
        try:
            akis.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    except Exception:
        pass

import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("app")

#: Varsayılan port (doluysa otomatik olarak bir sonraki boş port seçilir)
DEFAULT_PORT = 8791

#: Pencere boyutu (Edge `--window-size`)
DEFAULT_WINDOW = "1500,950"

#: Edge hata ayıklama portu (yalnızca 127.0.0.1; arayüz DOM'unu incelemek için)
DEFAULT_DEBUG_PORT = 9222


class DesktopApp:
    """
    Masaüstü uygulamasının tamamı: ajan + sunucu + pencere.

    Kullanım:
        app = DesktopApp(workspace="E:/proje", autonomy="supervised")
        app.setup()          # ajan katmanlarını kur
        app.serve(blocking=True)
    """

    def __init__(self, workspace: Optional[str] = None,
                 autonomy: str = "supervised",
                 port: int = DEFAULT_PORT,
                 host: str = "127.0.0.1",
                 open_window: bool = True,
                 browser: bool = False) -> None:
        self.workspace = str(Path(workspace or os.getcwd()).expanduser().resolve())
        self.autonomy = autonomy
        self.port = port
        self.host = host
        # DİKKAT: öznitelik adı `open_window` OLAMAZ — `open_window()` metodunu
        # gölgeler ve `self.open_window()` çağrısı
        # "TypeError: 'bool' object is not callable" ile çöker.
        self.open_window_on_start = open_window
        self.use_default_browser = browser
        #: Arayüz hata ayıklama portu (Edge DevTools protokolü, yalnızca yerel)
        self.debug_port = DEFAULT_DEBUG_PORT

        self.state: Any = None
        self.registry: Any = None
        self.router: Any = None
        self.loop: Any = None
        self.server: Any = None
        self.app: Any = None
        self._window_process: Optional[subprocess.Popen] = None
        self._setup_done = False
        self._started_at = time.time()
        #: Öğrenme kaydedicisi (kurulumda oluşturulur)
        self.learner: Any = None
        #: Kendi kendini geliştirme döngüsü (kurulumda oluşturulur)
        self.study: Any = None

    # ─── Kurulum ────────────────────────────────────────────────────────

    def setup(self) -> Dict[str, Any]:
        """
        Ajan katmanlarını kurar ve araçları kaydeder.

        Returns:
            {"ok", "tools", "modules", "llm", "summary"}
        """
        if self._setup_done:
            return {"ok": True, "note": "zaten kurulu"}

        from neurocore.app.state import AppState
        from neurocore.core.event_bus import EventBus
        from neurocore.agent import (
            AgentLoop,
            AutonomyLevel,
            AutonomyPolicy,
            CodingAgent,
            LLMRouter,
            ShellIntegration,
            ToolContext,
            ToolRegistry,
        )

        self.state = AppState()
        # ⚠ MCP gibi geç çalışan uçlar launcher'a erişsin
        self.state._launcher = self
        self.state.workspace = self.workspace

        # ── Olay bus'ı (denetim ve telemetri için) ──
        event_bus = EventBus()
        event_bus.start() if hasattr(event_bus, "start") else None
        self.state.event_bus = event_bus

        # ── Veri dizini ──
        data_path = Path(self.workspace) / ".neurocore"
        try:
            data_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            data_path = Path.home() / ".neurocore"
            data_path.mkdir(parents=True, exist_ok=True)
        # ⚠ Ortak havuz kimliği/anahtarı burada tutulur
        self.data_path = data_path

        # ── Özerklik politikası ──
        try:
            level = AutonomyLevel(self.autonomy)
        except ValueError:
            level = AutonomyLevel.SUPERVISED
        policy = AutonomyPolicy(level=level)

        # ── Araç kayıt defteri ──
        self.registry = ToolRegistry()
        self.registry.initialize(
            event_bus=event_bus, policy=policy,
            context=ToolContext(workspace=self.workspace, event_bus=event_bus,
                                autonomy=policy),
        )

        # ── Kodlama ajanı ──
        try:
            from neurocore.agent import CodingAgent as _Coding

            self.state.coding = _Coding()
            self.state.coding.initialize(workspace=self.workspace, event_bus=event_bus)
            self.registry.register_many(self.state.coding.tools())
        except Exception as exc:
            logger.error(f"Kodlama ajanı kurulamadı: {exc}")

        # ── Kabuk katmanı ──
        try:
            from neurocore.agent import ShellIntegration as _Shell

            self.state.shell = _Shell()
            self.state.shell.initialize(workspace=self.workspace, event_bus=event_bus)
            self.registry.register_many(self.state.shell.tools())
        except Exception as exc:
            logger.error(f"Kabuk katmanı kurulamadı: {exc}")

        # ── Görsel doğrulama ──
        try:
            from neurocore.vision import DesktopVision

            self.state.vision = DesktopVision()
            self.state.vision.initialize(None, event_bus)
        except Exception as exc:
            logger.error(f"Görsel doğrulama kurulamadı: {exc}")

        # ── Godot / Blender (isteğe bağlı) ──
        for module_name, class_name, tool_attr in (
            ("godot", "GodotController", "state.godot"),
            ("blender", "BlenderController", "state.blender"),
        ):
            try:
                module = __import__(f"neurocore.agent.{module_name}",
                                    fromlist=[class_name])
                controller_class = getattr(module, class_name)
                controller = controller_class()
                controller.initialize(workspace=self.workspace, event_bus=event_bus,
                                      shell=self.state.shell)
                setattr(self.state, module_name, controller)
                self.registry.register_many(controller.tools())
                logger.info(f"{class_name} yüklendi")
            except Exception as exc:
                logger.warning(f"{class_name} yüklenemedi: {exc}")

        # ── Görsel araçları kayıt defterine ekle ──
        if self.state.vision is not None:
            self._register_vision_tools()

        # ── Yerel model ile aday sıralama ──
        # Ölçüldü: yerel model akıcı metin üretemiyor ama **tanıyor**
        # (Blender kod seçiminde %70, yaklaşımda %50; rastgele %25). Bu araç
        # o yeteneği bulut modelinin ürettiği adayları süzmek için kullanır.
        try:
            from neurocore.agent.local_rank import LocalRankController

            self.state.local_rank = LocalRankController()
            self.state.local_rank.initialize(event_bus=event_bus, engine=None)
            self.registry.register_many(self.state.local_rank.tools())
            logger.info("Yerel sıralama araçları kaydedildi")
        except Exception as exc:
            logger.warning(f"Yerel sıralama kurulamadı: {exc}")

        # ── Düşünme döngüsü ──
        # Bulut üretir → yerel seçer → belirsizse buluta düşülür. İkisi
        # birlikte, tek başına olduklarından iyidir.
        try:
            from neurocore.agent.thinking import ThinkingLoop

            self.state.thinking = ThinkingLoop()
            self.state.thinking.initialize(
                ranker=self.state.local_rank, data_path=str(data_path))
            logger.info("Düşünme döngüsü bağlandı")
        except Exception as exc:
            logger.warning(f"Düşünme döngüsü kurulamadı: {exc}")

        # ── Ajan köprüsü (parçalı iş motoru) ──
        # Ajan görevleri `WorkStep`'e çevrilir: parçalanır, kaynağa göre
        # kuyruğa alınır, günlüğe yazılır ve kesintiden sonra sürdürülür.
        try:
            from neurocore.agent.bridge import AgentBridge

            self.state.bridge = AgentBridge(workspace=str(data_path))
            self.state.bridge.initialize(
                loop=self.loop, thinking=self.state.thinking)
            logger.info("Ajan köprüsü (parçalı motor) bağlandı")
        except Exception as exc:
            logger.warning(f"Ajan köprüsü kurulamadı: {exc}")

        # ── Dil modeli ──
        try:
            from neurocore.agent import LLMRouter as _Router

            self.router = _Router()
            self.router.initialize(data_path=str(data_path))
        except Exception as exc:
            logger.error(f"Dil modeli katmanı kurulamadı: {exc}")
            self.router = None

        # ── Sohbet deposu ──
        # Her sohbet kendi çalışma alanına bağlanabilir; soldan klasör
        # seçildiğinde o sohbet oraya bağlanır, seçilmezse **genel sohbet
        # modu** (proje bağlamı olmadan) çalışır.
        try:
            from neurocore.app.chats import ChatStore

            self.chats = ChatStore()
            self.chats.initialize(data_path=str(data_path))
            self.state.chats = self.chats
            logger.info("Sohbet deposu bağlandı")
        except Exception as exc:
            logger.error(f"Sohbet deposu kurulamadı: {exc}")

        # ── Bilgi iş akışı (ajan bilmediğinde öğrenir) ──
        #
        # Ölçülen boşluk: `SelfStudy` (öğretmen–öğrenci döngüsü) ve
        # `LearningRecorder` vardı ama ajana **bağlı değildi**. Ajanın 65
        # aracı arasında `knowledge.*` (arama) vardı, `learn.*` yoktu — yani
        # bilmediği bir konuda tahmin yürütüyor ya da "bilmiyorum" diyordu.
        # Bu blok üç araç ekler: `knowledge.ask`, `learn.topic`, `learn.status`.
        try:
            from neurocore.agent.knowledge_work import KnowledgeWorkflow

            self.state.knowledge_work = KnowledgeWorkflow()
            self.state.knowledge_work.initialize(
                study=getattr(self.state, "study", None),
                knowledge=getattr(self.state, "knowledge", None))
            self.registry.register_many(self.state.knowledge_work.tools())
            logger.info("Bilgi iş akışı bağlandı — bilmiyorsa öğrenir")
        except Exception as exc:
            logger.warning(f"Bilgi iş akışı kurulamadı: {exc}")

        # ── Ajan döngüsü ──
        from neurocore.agent import AgentLoop as _Loop

        self.loop = _Loop()
        self.loop.initialize(
            self.registry, self.router, workspace=self.workspace,
            event_bus=event_bus,
            on_step=lambda step: self.state.events.publish(
                "agent.step", step.to_dict(max_result=2000)
            ),
            approval_callback=self._approval_callback,
        )

        # ── Düşünme döngüsünü ajana bağla (Aşama 5) ──
        #
        # ⚠ ÖLÇÜLEN EKSİK: `ThinkingLoop` yalnızca **köprüye** (parçalı iş
        # motoru) veriliyordu; sohbet/görev yolunda (`AgentLoop.run`) hiç
        # kullanılmıyordu. Sonuç: ajan planı **tek** çağrıyla alıyordu,
        # yerel modelin seçme yeteneğinden yararlanılmıyordu.
        # Burada bağlanır: bulut N farklı plan önerir, yerel model seçer.
        try:
            _thinker = getattr(self.state, "thinking", None)
            self.loop.attach_thinker(_thinker)
            logger.info(
                "Düşünme döngüsü ajana bağlandı — plan seçimi yerel modelle"
                if _thinker is not None else
                "Düşünme döngüsü yok — plan tek çağrıyla alınacak")
        except Exception as exc:
            logger.warning(f"Düşünme ajana bağlanamadı: {exc}")

        # ── Ajan köprüsünü **döngüye** bağla ──
        #
        # ⚠ Sıra önemli: köprü, araç çağrılarını ajan döngüsünün **araç kayıt
        # defterinden** bulur. Köprü döngüden önce kurulduğu için başlangıçta
        # `loop=None` kalır; burada gerçek döngü bağlanır. Aksi hâlde hiçbir
        # araç çağrılamaz ve tüm adımlar "araç kayıt defteri yok" derdi.
        kopru = getattr(self.state, "bridge", None)
        if kopru is not None:
            try:
                kopru.initialize(loop=self.loop,
                                 thinking=getattr(self.state, "thinking", None))
                logger.info(
                    "Ajan köprüsü ajan döngüsüne bağlandı "
                    f"({len(self.registry.names())} araç erişilebilir)")
            except Exception as exc:
                logger.warning(f"Köprü döngüye bağlanamadı: {exc}")

        # ── İptal zinciri ──
        # "Durdur" basıldığında yalnızca ajan bayrağı kalkmaz; çalışan kabuk
        # komutları da öldürülür. Aksi hâlde 600 saniyelik bir komut iptalden
        # sonra da çalışmaya devam ediyordu.
        #
        # Dikkat: kabuk katmanı `self.state.shell` altındadır (`self.shell`
        # diye bir alan yoktur — yanlış ad, kancanın sessizce bağlanmamasına
        # yol açmıştı).
        shell = getattr(self.state, "shell", None)
        if shell is not None:
            try:
                self.loop.add_cancel_hook(shell.on_cancel)
                logger.info("İptal zinciri bağlandı: Durdur → çalışan komutları öldür")
            except Exception as exc:
                logger.error(f"İptal zinciri bağlanamadı: {exc}")
        else:
            logger.error(
                "Kabuk katmanı yok — 'Durdur' çalışan komutları öldüremeyecek"
            )

        # ⚠ MCP: dış uygulamalar (Blender, Godot) stdio üzerinden bağlanır.
        #    Yönetici tembeldir: süreç yalnız araç istendiğinde başlar.
        try:
            from neurocore.agent.mcp import MCPYonetici

            self.mcp = MCPYonetici(Path(self.data_path) / "mcp.json")
            self.state.mcp = self.mcp
            self._mcp_arac_kaydet()
            logger.info(f"MCP yöneticisi hazır — "
                        f"{self.mcp.durum()['sunucu_sayisi']} sunucu")
        except Exception as exc:
            logger.warning(f"MCP yöneticisi kurulamadı: {exc}")
            self.mcp = None

        # ── Öğrenme kaydedicisi ──
        # Ajanın yaptığı işi deneyime, kasaya ve eğitime bağlar. Bu bağlama
        # olmadan ajan saatlerce çalışsa bile yerel model ilerlemezdi.
        try:
            from neurocore.agent.learning import LearningRecorder

            self.learner = LearningRecorder()
            self.learner.initialize(workspace=self.workspace,
                                    data_path=str(data_path), event_bus=event_bus)
            # Arka plan işçisi: kaynak uygunsa ve yeterli yeni deneyim
            # birikmişse eğitimi kendiliğinden başlatır. Kaynak nezaketi
            # kapısı her turda yeniden denetlenir.
            self.learner.start_worker(interval=180.0)
            self.state.learner = self.learner
            self.loop.set_run_listener(self._on_agent_run)
            logger.info("Öğrenme kaydedicisi bağlandı ve arka plan işçisi başladı")
        except Exception as exc:
            logger.error(f"Öğrenme kaydedicisi kurulamadı: {exc}")

        # ── Kendi kendini geliştirme (öğretmen–öğrenci döngüsü) ──
        # Yerel model, bulut modelini **öğretmen** olarak kullanır: soru üretir,
        # öğretmenin önerdiği kaynakları **indirip doğrular**, kaynağa dayalı
        # cevap alır, yerel modelin eksiğini yazdırır ve kasaya not bırakır.
        # Bu bağ olmadan öğrenme yalnızca kullanıcının görevlerine bağlı kalır.
        try:
            from neurocore.agent.study import SelfStudy

            self.study = SelfStudy()
            self.study.initialize(
                llm=self.router,
                knowledge=self.learner.knowledge_layer() if self.learner else None,
                recorder=self.learner,
                workspace=self.workspace,
                event_bus=event_bus,
            )
            self.state.study = self.study
            # Arka plan: 20 dakikada bir tur (API kotasını ve diski ölçülü
            # kullanmak için; kullanıcı elle de başlatabilir)
            self.study.start_worker(interval=1200.0)
            logger.info("Kendi kendini geliştirme bağlandı")
        except Exception as exc:
            logger.error(f"Kendi kendini geliştirme kurulamadı: {exc}")

        self.state.registry = self.registry
        self.state.router = self.router
        self.state.loop = self.loop

        self._setup_done = True
        summary = {
            "ok": True,
            "tools": len(self.registry),
            "categories": self.registry.categories(),
            "autonomy": policy.level.label,
            "agent_mode": self.loop.mode,
            "llm_available": bool(self.router and self.router.available),
            "vision": self.state.vision is not None,
            "godot": self.state.godot is not None,
            "blender": self.state.blender is not None,
            "workspace": self.workspace,
        }
        logger.info(
            f"Uygulama hazır — {summary['tools']} araç, özerklik: "
            f"{summary['autonomy']}, ajan modu: {summary['agent_mode']}, "
            f"model: {'var' if summary['llm_available'] else 'yok (kural tabanlı)'}"
        )
        return summary

    def _register_vision_tools(self) -> None:
        """Görsel doğrulama araçlarını kayıt defterine ekler."""
        from neurocore.agent.tools import Safety, ToolContext, ToolDef, ToolParameter

        vision = self.state.vision

        def screenshot(ctx: ToolContext, window: str = "", tag: str = "agent",
                       max_side: int = 1280) -> Dict[str, Any]:
            return vision.screenshot(window=window or None, save=True, tag=tag,
                                     max_side=max_side)

        def list_windows(ctx: ToolContext, query: str = "",
                         min_size: int = 100) -> Dict[str, Any]:
            windows = vision.list_windows(min_size=min_size, query=query)
            return {"count": len(windows), "windows": windows}

        def describe_screen(ctx: ToolContext, question: str = "",
                            window: str = "") -> Dict[str, Any]:
            return vision.describe(question=question, window=window or None)

        def verify_app(ctx: ToolContext, title: str, expect_content: bool = True,
                       timeout: float = 15.0) -> Dict[str, Any]:
            return vision.verify_application(title, expect_content=expect_content,
                                             timeout=timeout)

        def ui_text(ctx: ToolContext, query: str) -> Dict[str, Any]:
            return vision.window_text(query)

        def compare(ctx: ToolContext, baseline: str = "") -> Dict[str, Any]:
            if baseline:
                return vision.check_baseline(baseline)
            return {"ok": False, "error": "referans adı gerekli"}

        def set_baseline(ctx: ToolContext, name: str, window: str = "") -> Dict[str, Any]:
            return {"ok": vision.set_baseline(name, window=window or None)}

        definitions = [
            ToolDef(
                name="vision.screenshot",
                description="Ekran görüntüsü alır ve analiz eder: boş mu, "
                            "parlaklık, kontrast, baskın renkler. Bir uygulamanın "
                            "gerçekten bir şey gösterdiğini doğrulamak için kullanın.",
                handler=screenshot, safety=Safety.READ, category="vision",
                parameters=[
                    ToolParameter("window", description="Pencere başlığı (boş = tüm ekran)"),
                    ToolParameter("tag", description="Dosya etiketi"),
                    ToolParameter("max_side", "integer", "En uzun kenar", default=1280),
                ],
                returns="görüntü yolu ve analiz", tags=["görsel", "ekran"],
            ),
            ToolDef(
                name="vision.list_windows",
                description="Açık pencereleri listeler (başlık, süreç, konum, boyut).",
                handler=list_windows, safety=Safety.READ, category="vision",
                parameters=[ToolParameter("query", description="Arama metni"),
                            ToolParameter("min_size", "integer", "Minimum boyut",
                                          default=100)],
                returns="pencere listesi", tags=["görsel", "pencere"],
            ),
            ToolDef(
                name="vision.describe",
                description="Ekranı veya bir pencereyi betimler: sağlık durumu, "
                            "OCR ile okunan metin. 'Ekranda ne var?' sorusunu yanıtlar.",
                handler=describe_screen, safety=Safety.READ, category="vision",
                parameters=[ToolParameter("question", description="Bağlam sorusu"),
                            ToolParameter("window", description="Pencere başlığı")],
                returns="betimleme ve OCR metni", tags=["görsel", "ocr"],
            ),
            ToolDef(
                name="vision.verify_app",
                description="Bir uygulamanın açıldığını ve BOŞ OLMAYAN bir pencere "
                            "gösterdiğini doğrular. Süreç başlamış olsa bile pencere "
                            "gelmemiş olabilir; bu araç gerçeği söyler.",
                handler=verify_app, safety=Safety.READ, category="vision",
                parameters=[ToolParameter("title", required=True,
                                          description="Pencere başlığı parçası"),
                            ToolParameter("expect_content", "boolean",
                                          "İçerik beklensin mi", default=True),
                            ToolParameter("timeout", "number", "Saniye", default=15.0)],
                returns="doğrulama sonucu", tags=["görsel", "doğrulama"],
            ),
            ToolDef(
                name="vision.ui_text",
                description="Bir pencerenin arayüz metinlerini okur (buton "
                            "etiketleri, hata mesajları). Ekran görüntüsü almadan "
                            "arayüzün ne dediğini öğrenir.",
                handler=ui_text, safety=Safety.READ, category="vision",
                parameters=[ToolParameter("query", required=True,
                                          description="Pencere başlığı")],
                returns="arayüz metinleri", tags=["görsel", "ui"],
            ),
            ToolDef(
                name="vision.set_baseline",
                description="Mevcut ekranı referans olarak kaydeder (regresyon testi).",
                handler=set_baseline, safety=Safety.READ, category="vision",
                parameters=[ToolParameter("name", required=True, description="Referans adı"),
                            ToolParameter("window", description="Pencere başlığı")],
                returns="başarı durumu", tags=["görsel", "regresyon"],
            ),
            ToolDef(
                name="vision.compare_baseline",
                description="Mevcut ekranı kayıtlı referansla karşılaştırır: arayüz "
                            "değişti mi, neresi değişti?",
                handler=compare, safety=Safety.READ, category="vision",
                parameters=[ToolParameter("baseline", required=True,
                                          description="Referans adı")],
                returns="fark sonucu", tags=["görsel", "regresyon"],
            ),
        ]
        self.registry.register_many(definitions)

    def _on_agent_run(self, run: Any) -> None:
        """
        Her ajan çalışması bittiğinde çağrılır: işi **öğrenmeye** çevirir.

        Kaydedici yoksa sessizce atlanır (öğrenme isteğe bağlıdır).
        """
        learner = getattr(self, "learner", None)
        if learner is None:
            return
        try:
            record = learner.record(run)
            self.state.events.publish("learning.recorded", {
                "task": record.task[:160], "reward": record.reward,
                "ok": record.ok, "instructive": record.instructive,
                "at": record.at, "tools": record.tools_used,
            })
        except Exception as exc:
            logger.debug(f"Ajan çalışması kaydedilemedi: {exc}")

    def _approval_callback(self, tool: str, args: Dict[str, Any], safety: str,
                           reason: str, summary: str) -> Tuple[bool, bool]:
        """
        Ajan onay istediğinde çağrılır (iş parçacığında).

        Arayüze bir olay yayar ve kullanıcının yanıtını bekler. Arayüz bağlı
        değilse zaman aşımı sonunda **reddedilir**.
        """
        from neurocore.agent.tools import SAFETY_LABELS, Safety

        try:
            safety_enum = Safety(safety)
            label = SAFETY_LABELS.get(safety_enum, safety)
        except ValueError:
            label = safety

        return self.state.approvals.request(tool, args, safety, label, reason, summary)

    # ─── Sunucu ─────────────────────────────────────────────────────────

    def find_port(self, start: Optional[int] = None, attempts: int = 20) -> int:
        """
        Boş bir port bulur.

        ÖNEMLİ: kullanılan portu **raporlar**. Aynı anda birden fazla
        NeuroCore penceresi açılabildiği için sabit port çakışır. Ancak
        sessizce başka porta geçmek tehlikelidir: pencere her zaman
        `http://127.0.0.1:<port>/` adresini açtığı için, eski bir örnek
        8791'i tutuyorsa yeni örnek 8792'ye geçer ve pencere **eski sunucuya**
        bağlanır. Kullanıcı yeni araçları görmez ve eski örneğin takılı
        kalmış onay istekleriyle karşılaşır.
        """
        base = start or self.port
        for offset in range(attempts):
            candidate = base + offset
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind((self.host, candidate))
                    if offset:
                        logger.warning(
                            f"{base} portu dolu — {candidate} kullanılacak. "
                            f"Eski bir NeuroCore örneği çalışıyor olabilir; "
                            f"kapatmak için: Kisayol ile 'Sunuculari-Kapat.bat'"
                        )
                    return candidate
                except OSError:
                    continue
        raise RuntimeError(f"boş port bulunamadı ({base}..{base + attempts - 1})")

    def _existing_instance(self, port: int) -> Optional[Dict[str, Any]]:
        """
        Bu portta çalışan bir NeuroCore örneği var mı?

        Returns:
            Bulunursa durum sözlüğü, yoksa None
        """
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"http://{self.host}:{port}/api/state", timeout=2.5
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None
        # Başka bir uygulama aynı portu kullanıyor olabilir; imzayı doğrula
        if "workspace" in payload and "tools" in payload and "autonomy" in payload:
            return payload
        return None

    def ensure_single_instance(self, reuse: bool = True) -> Dict[str, Any]:
        """
        Aynı portta çalışan bir örnek varsa onu **kullanır**.

        Neden gerekli? Kullanıcı `Run.bat`'a iki kez basarsa iki sunucu ve iki
        pencere açılır; ikincisi farklı bir porta geçer ve pencere eski
        sunucuya bağlanır. Bu, "yeni sürüm çalışmıyor" gibi görünen kafa
        karıştırıcı bir durum yaratır. Bunun yerine mevcut örnek yeniden
        kullanılır ve pencere ona odaklanır.

        Args:
            reuse: True ise mevcut örnek kullanılır

        Returns:
            {"reused": bool, "port": int, "state": dict}
        """
        state = self._existing_instance(self.port)
        if state is None:
            return {"reused": False, "port": self.port, "state": {}}

        if not reuse:
            raise RuntimeError(
                f"{self.host}:{self.port} üzerinde zaten bir NeuroCore çalışıyor"
            )

        logger.info(
            f"Mevcut NeuroCore örneği bulundu ({self.host}:{self.port}) — "
            "yeni örnek başlatılmadı, pencere ona bağlanacak"
        )
        return {"reused": True, "port": self.port, "state": state}

    def url(self) -> str:
        """Arayüz adresi."""
        return f"http://{self.host}:{self.port}/"

    def serve(self, blocking: bool = True) -> Optional[threading.Thread]:
        """
        Sunucuyu başlatır.

        Args:
            blocking: True ise süreç sunucu kapanana kadar bloklar

        Returns:
            blocking=False ise sunucu iş parçacığı
        """
        if not self._setup_done:
            self.setup()

        import uvicorn

        from neurocore.app.server import create_app

        self.port = self.find_port(self.port)
        self.app = create_app(self.state, static_dir=Path(__file__).parent / "web")

        # `/api/sunucu-kapat` uç noktası bu geri çağrımı kullanır: arayüzden
        # veya başlatıcı betiğinden gelen kapatma isteği sunucuyu durdurur.
        self.state.shutdown_callback = self._request_stop

        config = uvicorn.Config(
            self.app, host=self.host, port=self.port,
            log_level="warning", access_log=False, loop="asyncio",
        )
        self.server = uvicorn.Server(config)

        # ⚠ Ortak havuz: port kesinleştikten sonra tünel + kayıt + heartbeat
        self._havuz_baslat()

        logger.info(f"Sunucu başlıyor: {self.url()}")

        if blocking:
            self.server.run()
            return None

        thread = threading.Thread(target=self.server.run, name="NeuroCoreServer",
                                  daemon=True)
        thread.start()
        self._wait_until_ready()
        return thread

    def _mcp_arac_kaydet(self) -> None:
        """
        MCP araçlarını kayıt defterine ekler: `mcp.<sunucu>.kullan`.

        ⚠ Sunucuya **bağlanmadan** kaydedilir; araç şeması ilk çağrıda
        sunucudan istenir. Böylece başlangıç yavaşlamaz ve sunucu kapalıysa
        ajan yalnız o aracı "bağlanamadı" diye görür.

        ⚠ `register()` imzası: `register(ToolDef, replace=...)` — ilk yazımda
        (isim, fn, category=...) denendiği için sessizce başarısız olmuştu.
        """
        if self.registry is None or self.mcp is None:
            return
        from neurocore.agent.tools import (
            Safety, ToolDef, ToolParameter, ToolResult,
        )

        for bilgi in (self.mcp.durum().get("sunucular") or []):
            sunucu = str(bilgi.get("ad") or "")
            if not sunucu:
                continue
            isim = f"mcp.{sunucu}.kullan"

            def _calistir(*_args: Any, _s: str = sunucu, **kwargs: Any) -> Any:
                paket = dict(kwargs or {})
                if _args and isinstance(_args[0], dict):
                    paket = {**_args[0], **paket}
                arac = str(paket.get("arac") or "")
                arguments = paket.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        import json as _json

                        arguments = _json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if not arac:
                    return ToolResult(ok=False, tool=f"mcp.{_s}.kullan",
                                      error="`arac` alanı gerekli")
                sonuc = self.mcp.cagir(f"mcp.{_s}.{arac}", arguments)
                return ToolResult(
                    ok=bool(sonuc.get("ok")),
                    tool=f"mcp.{_s}.kullan",
                    data={"metin": sonuc.get("metin", "")},
                    error="" if sonuc.get("ok") else str(
                        sonuc.get("error") or sonuc.get("metin") or "")[:300],
                )

            tanim = ToolDef(
                name=isim,
                description=(f"MCP sunucusu `{sunucu}` üzerinden araç çağırır. "
                             f"Önce o sunucunun araçlarından birini seç "
                             f"(ör. `arac`), sonra `arguments` sözlüğünü ver."),
                handler=_calistir,
                parameters=[
                    ToolParameter(name="arac", type="string",
                                  description="MCP aracının adı", required=True),
                    ToolParameter(name="arguments", type="object",
                                  description="Araca verilecek parametreler",
                                  required=False, default={}),
                ],
                safety=Safety.EXECUTE,
                category="mcp",
                returns="MCP sunucusunun metin yanıtı",
            )
            try:
                self.registry.register(tanim, replace=True)
                logger.info(f"MCP aracı kaydedildi: {isim}")
            except Exception as exc:
                logger.warning(f"MCP aracı kaydedilemedi ({isim}): {exc}")

    def _havuz_baslat(self) -> None:
        """
        Ortak havuzu başlatır (adım 2): ters tünel + kayıt + heartbeat.

        ⚠ Neden burada? Sunucu portu (`self.port`) ancak `start()` içinde
        kesinleşir; tünel o porta yönlendirilir. `NC_PEER=0` ile kapatılır.

        ⚠ Sessiz yutma yok: kurulamazsa uyarı yazılır ve sayaç artar.
        """
        import os

        if (os.environ.get("NC_PEER") or "1").strip().lower() in (
                "0", "false", "hayir", "hayır", "off"):
            logger.info("Ortak havuz kapalı (NC_PEER=0)")
            return
        try:
            from neurocore.net.peer_pool import PeerPool

            havuz = PeerPool()
            havuz.initialize(data_path=str(getattr(self, "data_path",
                                                   ".neurocore")))
            sonuc = havuz.baslat(
                port=int(self.port),
                caps=["code", "blender", "arastirma", "egitim"],
                tunel=True,
            )
            self.state.peer_pool = havuz
            # ⚠ Adım 3: ajan döngüsüne bağla — izinli okuma araçları
            # önce komşuda denenir, yanıt yoksa yerelde çalışır.
            try:
                self.loop.attach_peer(havuz)
            except Exception as exc:
                logger.warning(f"Havuz ajana bağlanamadı: {exc}")
            kayit = sonuc.get("kayit") or {}
            logger.info(
                f"Ortak havuz: tünel={sonuc.get('tunel')} · "
                f"kayıt={'ok' if kayit.get('ok') else 'yok'} · "
                f"ip={kayit.get('ip')}")
            if not sonuc.get("ok"):
                logger.warning(f"Ortak havuz eksik: {sonuc.get('hata')}")
        except Exception as exc:
            logger.warning(
                f"Ortak havuz kurulamadı: {type(exc).__name__}: {exc}")

    def _wait_until_ready(self, timeout: float = 20.0) -> bool:
        """Sunucunun yanıt vermesini bekler."""
        import urllib.error
        import urllib.request

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{self.url()}api/state", timeout=1.5):
                    return True
            except (urllib.error.URLError, OSError):
                time.sleep(0.15)
        logger.warning(f"Sunucu {timeout:.0f}s içinde yanıt vermedi")
        return False

    # ─── Pencere ────────────────────────────────────────────────────────

    def open_window(self) -> bool:
        """
        Masaüstü penceresini açar (Edge `--app` modu).

        Returns:
            Pencere açıldıysa True
        """
        url = self.url()

        if self.use_default_browser:
            webbrowser.open(url)
            return True

        edge = self._find_edge()
        if edge:
            arguments = [
                edge,
                f"--app={url}",
                f"--window-size={DEFAULT_WINDOW}",
                "--window-position=80,60",
                # Uygulama penceresi hissi: tarayıcı chrome'u yok,
                # ayrı profil (kullanıcının normal oturumunu karıştırmaz)
                f"--user-data-dir={self._edge_profile_dir()}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-features=Translate,MediaRouter",
                "--disable-background-mode",
                # Hata ayıklama: arayüzün canlı DOM'unu incelemeyi mümkün kılar
                # (yalnızca yerel, rastgele port). Sorun teşhisinde belirleyici.
                f"--remote-debugging-port={self.debug_port}",
                "--remote-allow-origins=*",
            ]
            try:
                self._window_process = subprocess.Popen(
                    arguments, close_fds=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                logger.info(f"Masaüstü penceresi açıldı (Edge --app): {url}")
                return True
            except OSError as exc:
                logger.warning(f"Edge penceresi açılamadı: {exc}")

        logger.info("Edge bulunamadı — varsayılan tarayıcıya düşülüyor")
        webbrowser.open(url)
        return True

    def _edge_profile_dir(self) -> str:
        """
        Edge için ayrı profil dizini.

        Kullanıcının kendi tarayıcı profili **kullanılmaz**; aksi hâlde
        NeuroCore penceresi kullanıcının açık sekmelerini devralır veya
        kapanınca onları kapatır. Ayrı profil, uygulamanın bağımsız
        davranmasını sağlar.
        """
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "NeuroCore" / "ui-profile"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            base = Path.home() / ".neurocore-ui"
            base.mkdir(parents=True, exist_ok=True)
        return str(base)

    @staticmethod
    def _find_edge() -> str:
        """Edge çalıştırılabilirinin yolunu bulur."""
        found = shutil.which("msedge")
        if found:
            return found
        candidates = [
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                         "Microsoft", "Edge", "Application", "msedge.exe"),
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                         "Microsoft", "Edge", "Application", "msedge.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""),
                         "Microsoft", "Edge", "Application", "msedge.exe"),
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return ""

    def close_window(self) -> None:
        """Açılan uygulama penceresini kapatır."""
        if self._window_process is None:
            return
        try:
            self._window_process.terminate()
        except Exception as exc:
            logger.debug(f"Pencere kapatılamadı: {exc}")
        self._window_process = None

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def _request_stop(self) -> None:
        """
        Sunucudan temiz kapanma ister.

        `uvicorn.Server.run()` çalışırken bu bayrağı görür, açık bağlantıları
        kapatır ve döngüden çıkar; böylece `run()` geri döner ve süreç
        normal biçimde sonlanır.
        """
        if self.server is not None:
            self.server.should_exit = True
            logger.info("Kapanma istendi — sunucu durduruluyor")

    def shutdown(self) -> Dict[str, Any]:
        """Tüm katmanları kapatır."""
        report: Dict[str, Any] = {"ok": True, "closed": []}
        self.close_window()

        for name, component in (
            ("study", getattr(self, "study", None)),
            ("learner", getattr(self, "learner", None)),
            ("loop", self.loop), ("codding", self.state.coding if self.state else None),
            ("shell", self.state.shell if self.state else None),
            ("vision", self.state.vision if self.state else None),
            ("godot", self.state.godot if self.state else None),
            ("blender", self.state.blender if self.state else None),
            # ⚠ Ortak havuz: kaydı siler + ters tüneli kapatır (adım 2)
            ("peer_pool", getattr(self.state, "peer_pool", None)
             if self.state else None),
            ("registry", self.registry),
            ("router", self.router),
        ):
            if component is None:
                continue
            try:
                if hasattr(component, "shutdown"):
                    component.shutdown()
                    report["closed"].append(name)
            except Exception as exc:
                logger.debug(f"{name} kapatılamadı: {exc}")

        if self.server is not None:
            try:
                self.server.should_exit = True
                report["closed"].append("server")
            except Exception:
                pass

        report["uptime_seconds"] = round(time.time() - self._started_at, 1)
        logger.info(f"Uygulama kapatıldı ({report['uptime_seconds']}s)")
        return report

    def run(self) -> int:
        """
        Uygulamayı çalıştırır (bloklayıcı).

        Returns:
            Çıkış kodu
        """
        try:
            summary = self.setup()
        except Exception as exc:
            print(f"\n  Kurulum başarısız: {exc}\n", file=sys.stderr)
            logger.error(f"Kurulum hatası: {exc}", exc_info=True)
            return 1

        print()
        print("  NeuroCore Masaüstü")
        print("  " + "─" * 56)
        print(f"  Çalışma alanı : {summary['workspace']}")
        print(f"  Araç sayısı   : {summary['tools']} ({summary['categories']})")
        print(f"  Özerklik      : {summary['autonomy']}")
        print(f"  Ajan modu     : {summary['agent_mode']}")
        print(f"  Dil modeli    : {'hazır' if summary['llm_available'] else 'YOK — kural tabanlı mod'}")
        print(f"  Görsel        : {'var' if summary['vision'] else 'yok'}"
              f" · Godot: {'var' if summary['godot'] else 'yok'}"
              f" · Blender: {'var' if summary['blender'] else 'yok'}")
        print()

        # Kapanışta temizlik
        def on_signal(_signum: int, _frame: Any) -> None:
            print("\n  Kapatılıyor…")
            self.shutdown()
            raise SystemExit(0)

        for signal_name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, signal_name):
                try:
                    signal.signal(getattr(signal, signal_name), on_signal)
                except (ValueError, OSError):
                    pass

        # ── Tek örnek denetimi ──
        # Kullanıcı başlatıcıya iki kez basarsa iki sunucu açılır ve ikincisi
        # farklı bir porta geçer; pencere ise her zaman aynı adresi açtığı için
        # **eski** sunucuya bağlanır. Bu, "yeni sürüm çalışmıyor" gibi görünen
        # ve eski örneğin takılı onaylarını gösteren kafa karıştırıcı bir
        # durum yaratır. Bunun yerine mevcut örnek yeniden kullanılır.
        existing = self.ensure_single_instance()
        if existing["reused"]:
            print("  Zaten çalışan bir NeuroCore örneği bulundu.")
            print(f"  Pencere ona bağlanıyor: {self.url()}")
            print("  (Kapatmak için: Sunuculari-Kapat.bat)\n")
            if self.open_window_on_start:
                self.open_window()
                # Pencere açık kalsın diye süreç ayakta bekletilir; aksi hâlde
                # başlatıcı kapanınca pencere de kapanır.
                try:
                    while True:
                        time.sleep(5.0)
                except KeyboardInterrupt:
                    pass
            return 0

        self.port = self.find_port(self.port)
        if self.open_window_on_start:
            def open_later() -> None:
                """Sunucu hazır olduktan sonra pencereyi açar."""
                try:
                    if self._wait_until_ready(timeout=25.0):
                        self.open_window()
                    else:
                        print(f"\n  Sunucu açılamadı. Elle deneyin: {self.url()}\n",
                              file=sys.stderr)
                except Exception as exc:
                    # Pencere açılamazsa uygulama yine de çalışmalı; kullanıcı
                    # adresi tarayıcıya yapıştırabilir.
                    logger.error(f"Pencere açılamadı: {exc}")
                    print(f"\n  Pencere açılamadı ({exc}). "
                          f"Tarayıcıda açın: {self.url()}\n", file=sys.stderr)

            threading.Thread(target=open_later, name="NeuroCoreWindow",
                             daemon=True).start()

        print(f"  Arayüz        : {self.url()}")
        print("  Kapatmak için bu pencerede Ctrl+C\n")

        try:
            self.serve(blocking=True)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return 0


# ─── Komut Satırı ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Komut satırı ayrıştırıcısı."""
    parser = argparse.ArgumentParser(
        prog="python -m neurocore.app",
        description="NeuroCore masaüstü uygulaması — kodlama, kabuk ve uygulama "
                    "kontrolü yapabilen yerel yapay zeka ajanı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  python -m neurocore.app\n"
            "  python -m neurocore.app --workspace E:/PROJELER/oyunum\n"
            "  python -m neurocore.app --autonomy readonly --no-window\n"
            "  python -m neurocore.app --browser --port 9000\n"
        ),
    )
    parser.add_argument("--workspace", "-w", default="",
                        help="Çalışma alanı (varsayılan: geçerli dizin)")
    parser.add_argument("--autonomy", "-a", default="supervised",
                        choices=["readonly", "supervised", "trusted", "yolo"],
                        help="Başlangıç özerklik seviyesi (varsayılan: supervised)")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT,
                        help=f"Sunucu portu (varsayılan: {DEFAULT_PORT}; doluysa "
                             "sonraki boş port seçilir)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Dinlenecek adres (varsayılan: 127.0.0.1)")
    parser.add_argument("--no-window", action="store_true",
                        help="Pencere açma; yalnızca sunucuyu çalıştır")
    parser.add_argument("--browser", action="store_true",
                        help="Edge --app yerine varsayılan tarayıcıyı kullan")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Günlük seviyesi (varsayılan: info)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Uygulama giriş noktası."""
    _utf8_konsol()
    arguments = build_parser().parse_args(argv)

    workspace = arguments.workspace or os.getcwd()
    if not Path(workspace).is_dir():
        print(f"  Hata: çalışma alanı yok: {workspace}", file=sys.stderr)
        return 2

    # ── Günlükleme ──
    # Konsolsuz çalıştırıldığında (`pythonw.exe` ile, yani çift tıklanan
    # başlatıcıda) `sys.stdout` **None** olur ve konsola yazan her günlük
    # çağrısı sessizce kaybolur. Bu yüzden:
    #   * konsol yoksa konsol günlüğü kapatılır,
    #   * dosya günlüğü **her zaman** açılır ve çalışma alanının altına yazılır.
    # Böylece bir şey ters giderse kullanıcı `.neurocore/logs/neurocore.log`
    # dosyasından sebebi görebilir.
    from neurocore.utils.logger import NeuroCoreLogger

    has_console = sys.stdout is not None and sys.stderr is not None
    log_dir = Path(workspace) / ".neurocore" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_dir = Path.home() / ".neurocore" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

    NeuroCoreLogger.initialize(
        log_dir=str(log_dir),
        level=arguments.log_level.upper(),
        console=has_console,
        file=True,
        force=True,
    )

    if not has_console:
        # `pythonw` altında `print` çağrıları hata verir; sessizce yut.
        class _NullStream:
            """Konsol yokken `print` hedefi (yazılanı yok sayar)."""

            def write(self, _data: str) -> int:
                return 0

            def flush(self) -> None:
                return None

            def isatty(self) -> bool:
                return False

        sys.stdout = _NullStream()   # type: ignore[assignment]
        sys.stderr = _NullStream()   # type: ignore[assignment]

    app = DesktopApp(
        workspace=workspace,
        autonomy=arguments.autonomy,
        port=arguments.port,
        host=arguments.host,
        open_window=not arguments.no_window,
        browser=arguments.browser,
    )
    return app.run()


__all__ = ["DesktopApp", "main", "build_parser", "DEFAULT_PORT"]
