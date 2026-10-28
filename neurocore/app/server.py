"""
NeuroCore Masaüstü Uygulaması — FastAPI Sunucusu
=================================================
Ajanı arayüze bağlayan HTTP + WebSocket uç noktaları.

Tüm uç noktalar **yalnızca 127.0.0.1** üzerinde dinlenir; dışarıya açılmaz.

Uç noktalar::

    GET  /                     → arayüz (index.html)
    GET  /api/state            → genel durum (özerklik, model, araç sayısı)
    POST /api/chat             → görev çalıştır (asenkron, hemen döner)
    POST /api/cancel           → çalışan görevi iptal et
    GET  /api/runs             → son çalışmaların özeti
    GET  /api/tools            → araç listesi
    POST /api/tool             → tek bir aracı doğrudan çağır
    GET  /api/autonomy         → özerklik seviyesi ve seçenekleri
    POST /api/autonomy         → seviyeyi değiştir
    POST /api/approve          → onay isteğini yanıtla
    GET  /api/approvals        → bekleyen onaylar
    GET  /api/settings         → dil modeli sağlayıcıları
    POST /api/settings/llm     → API anahtarı kaydet
    POST /api/settings/llm/test→ bağlantıyı sına
    POST /api/settings/workspace → çalışma alanını değiştir
    GET  /api/vision/...       → ekran görüntüsü, pencere listesi, analiz
    GET  /api/health           → sağlık raporu
    WS   /ws                   → canlı olay akışı

ÖNEMLİ — neden bu dosyada `from __future__ import annotations` YOK:
    O ifade tüm tip ipuçlarını **string**'e çevirir. FastAPI uç nokta
    imzalarını çalışma zamanında inceler; `request: Request` ifadesini
    çözemeyince `Request` nesnesini **zorunlu sorgu parametresi** sanır ve
    her istek `422 Unprocessable Entity` ile reddedilir
    (`{"loc":["query","request"],"msg":"Field required"}`). Bu yüzden bu
    modülde anotasyonlar gerçek nesne olarak kalmalıdır.
"""

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neurocore.app.state import HAS_FASTAPI, AppState, PendingApproval
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("app.server")


def _language_of(suffix: str) -> str:
    """Dosya uzantısından dil etiketi (arayüzde simge için)."""
    return {
        ".py": "python", ".js": "javascript", ".mjs": "javascript",
        ".cjs": "javascript", ".ts": "typescript", ".tsx": "typescript",
        ".jsx": "javascript", ".html": "html", ".htm": "html",
        ".css": "css", ".scss": "css", ".json": "json", ".md": "markdown",
        ".txt": "text", ".yml": "yaml", ".yaml": "yaml", ".toml": "toml",
        ".ini": "ini", ".cfg": "ini", ".sh": "shell", ".ps1": "powershell",
        ".bat": "batch", ".cmd": "batch", ".gd": "gdscript",
        ".cs": "csharp", ".cpp": "cpp", ".c": "c", ".h": "c",
        ".rs": "rust", ".go": "go", ".java": "java", ".rb": "ruby",
        ".sql": "sql", ".xml": "xml", ".svg": "svg", ".blend": "blender",
        ".tscn": "godot", ".tres": "godot", ".godot": "godot",
    }.get((suffix or "").lower(), "")


def apply_workspace(state: AppState, path: str) -> List[str]:
    """
    Çalışma alanını **tüm** bileşenlere uygular.

    Kodlama ajanı, kabuk katmanı, Godot ve Blender denetleyicileri aynı köke
    yönlendirilir; aksi hâlde ajan bir projede okurken başka bir projede
    yazardı.

    Args:
        state: Uygulama durumu
        path: Yeni kök dizin (mutlak, doğrulanmış)

    Returns:
        Güncellenen bileşenlerin adları
    """
    state.workspace = path
    switched: List[str] = []
    for component in (state.coding, state.shell, state.godot,
                      state.blender, state.vision):
        if component is None:
            continue
        try:
            if hasattr(component, "set_workspace"):
                component.set_workspace(path)
                switched.append(type(component).__name__)
            elif hasattr(component, "_workspace"):
                component._workspace = Path(path)
                switched.append(type(component).__name__)
        except Exception as exc:
            logger.warning(
                f"Çalışma alanı güncellenemedi ({type(component).__name__}): {exc}"
            )
    logger.info(f"Çalışma alanı: {path} ({', '.join(switched) or 'bileşen yok'})")
    state.events.publish("settings.workspace_changed", {
        "workspace": path, "components": switched,
    })
    return switched


#: ⚠ Uzaktan (komşu düğümden) çalıştırılabilir araçlar — **yalnız okuma**.
#:
#: Neden bu liste var? Ters tünel sunucunun `127.0.0.1`'ine bağlanır ve
#: relay isteği sunucu üzerinden gelir; **kaynak IP güvenilir değildir**.
#: Bu yüzden IP ile sınırlama yerine iki katman kullanılır:
#:   1. Paylaşılan anahtar (`ortak_token`)
#:   2. Bu **izin listesi** — `shell.run`, `code.run_command`, dosya yazma
#:      gibi araçlar uzaktan ASLA çalışmaz.
#: ⚠ Tek doğru kaynak: `neurocore.net.peer_pool.IZINLI_ARACLAR`.
#: Ajan döngüsü de **aynı** listeyi kullanır; ayrı yazılsaydı bir tarafta
#: izinli olan araç diğerinde reddedilirdi.
from neurocore.net.peer_pool import IZINLI_ARACLAR as PEER_IZINLI_ARACLAR


async def _hizli_oku(fn: Any, varsayilan: Any, *, zaman_asimi: float = 2.0) -> Any:
    """
    Kilit tutabilecek okumayı **ayrı iş parçacığında** ve **zaman aşımıyla** yapar.

    ⚠ ÖLÇÜLDÜ: ajan bir aracı çalıştırırken `registry` kilidini tutuyor;
    `/api/state` doğrudan `registry.get_stats()` çağırdığı için **tüm istek
    bloklanıyordu** (görev sırasında 6 sn'lik istemci zaman aşımı aşıldı,
    arayüz dondu). Artık kilit beklense bile yalnız bu okuma bekler; olay
    döngüsü ve diğer uçlar yanıt vermeye devam eder.
    """
    import asyncio

    try:
        return await asyncio.wait_for(asyncio.to_thread(fn), timeout=zaman_asimi)
    except asyncio.TimeoutError:
        return varsayilan
    except Exception:
        return varsayilan


def peer_job_isle(state: Any, payload: Any) -> Dict[str, Any]:
    """
    `/peer/job` mantığı — **HTTP'den bağımsız** (test edilebilir).

    Args:
        state: `AppState` (üzerinde `registry` aranır)
        payload: `{"token", "tool", "args"}`

    Returns:
        `{"ok", "sonuc"?, "hata"?, "ms"?, "arac"?}`
    """
    if not isinstance(payload, dict):
        return {"ok": False, "hata": "gövde sözlük değil"}

    # ── 1) Paylaşılan anahtar ──
    try:
        from neurocore.net.peer_pool import ortak_token

        beklenen = ortak_token()
    except Exception:                              # pragma: no cover
        beklenen = ""
    gelen = str(payload.get("token") or "")
    # ⚠ ÖLÇÜLEN RİSK: küme anahtarı yoksa eski kod doğrulamayı **atlıyordu**
    # → uç, ağdan gelen herkese açık kalıyordu. Artık anahtar yoksa
    # istek reddedilir (havuz başlatılmamış = uzak iş kabul edilmez).
    if not beklenen:
        return {"ok": False,
                "hata": "yetkisiz: küme anahtarı yok (havuz başlatılmamış)"}
    import hmac

    if not hmac.compare_digest(gelen, beklenen):
        return {"ok": False, "hata": "yetkisiz: token uyuşmuyor"}

    # ── 2) Araç adı ve izin ──
    arac = str(payload.get("tool") or "").strip()
    if not arac:
        return {"ok": False, "hata": "araç adı yok"}
    if arac not in PEER_IZINLI_ARACLAR:
        return {"ok": False,
                "hata": f"araç uzaktan çalıştırılamaz: {arac}"}

    kayit = getattr(state, "registry", None)
    if kayit is None:
        return {"ok": False, "hata": "araç kayıt defteri yok"}
    try:
        if not kayit.has(arac):
            return {"ok": False, "hata": f"araç kayıtlı değil: {arac}"}
    except Exception:
        pass

    args = payload.get("args") or {}
    if not isinstance(args, dict):
        return {"ok": False, "hata": "args sözlük değil"}

    # ── 3) Çalıştır ──
    basladi = time.perf_counter()
    try:
        sonuc = kayit.call(arac, args)
    except Exception as exc:
        return {"ok": False, "hata": f"{type(exc).__name__}: {exc}"[:200]}
    sure = round((time.perf_counter() - basladi) * 1000, 1)

    if getattr(sonuc, "needs_approval", False):
        return {"ok": False, "ms": sure, "arac": arac,
                "hata": "onay gerektiriyor — uzak düğümde çalıştırılamaz"}
    if not getattr(sonuc, "ok", False):
        return {"ok": False, "ms": sure, "arac": arac,
                "hata": str(getattr(sonuc, "error", "") or "başarısız")[:200]}
    # ⚠ `to_dict` çıktıyı JSON-güvenli ve sınırlı üretir
    try:
        veri = sonuc.to_dict(max_chars=20000)
    except Exception:
        veri = {"sonuc": str(getattr(sonuc, "data", ""))[:4000]}
    return {"ok": True, "ms": sure, "arac": arac, "sonuc": veri}


def create_app(state: AppState, static_dir: Optional[Path] = None) -> Any:
    """
    FastAPI uygulamasını oluşturur.

    Args:
        state: Paylaşılan uygulama durumu
        static_dir: Arayüz dosyalarının dizini (`app/web`)

    Returns:
        FastAPI uygulaması

    Raises:
        RuntimeError: FastAPI kurulu değilse
    """
    if not HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI kurulu değil — `pip install fastapi uvicorn` gerekir"
        )

    from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

    app = FastAPI(
        title="NeuroCore Masaüstü",
        description="Donanım-farkında, çok-modlu, kendi kendini büyütebilen yazılım ajanı",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    web_root = static_dir or (Path(__file__).parent / "web")

    # Onay isteklerini **bir kez** olay akışına bağla. Bu bağlama WebSocket
    # içinde yapılırsa, ajan bir onay istediğinde hiç arayüz bağlı değilse
    # istek sessizce kaybolur; bağlama burada yapıldığında ise olay akışına
    # yazılır ve sonradan bağlanan arayüz bekleyen onayları görür.
    state.approvals.add_listener(
        lambda approval: state.events.publish(
            "agent.approval_requested", approval.to_dict()
        )
    )

    # Onay **çözümlendiğinde** (onay/ret/zaman aşımı/iptal) arayüze haber ver.
    # Bu olmadan, zaman aşımına uğramış bir istek modalda takılı kalır:
    # kullanıcı butona basar, sunucu "onay bulunamadı" der ve modal kapanmaz.
    state.approvals.add_resolution_listener(
        lambda approval, approved, how: state.events.publish(
            "agent.approval_resolved",
            {"id": approval.id, "tool": approval.tool,
             "approved": approved, "how": how, "at": time.time()},
        )
    )

    # ─── Yardımcılar ────────────────────────────────────────────────────

    def _check_origin(request: Request) -> None:
        """
        İsteğin yerel arayüzden geldiğini doğrular.

        Sunucu yalnızca 127.0.0.1 dinlediği için dış erişim zaten mümkün
        değildir; bu denetim, kullanıcının tarayıcısındaki başka bir sayfanın
        (CSRF) yerel sunucuya istek atmasını engeller.
        """
        origin = request.headers.get("origin") or request.headers.get("referer") or ""
        if not origin:
            return
        allowed = ("127.0.0.1", "localhost", "[::1]")
        if not any(host in origin for host in allowed):
            logger.warning(f"Reddedilen origin: {origin}")
            raise HTTPException(status_code=403, detail="yalnızca yerel arayüz erişebilir")

    def _json(data: Any, status: int = 200) -> JSONResponse:
        return JSONResponse(content=data, status_code=status)

    # ─── Arayüz ─────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index() -> Any:
        """Arayüzü döner."""
        page = web_root / "index.html"
        if not page.exists():
            return HTMLResponse(
                "<h1>NeuroCore</h1><p>Arayüz dosyası bulunamadı: "
                f"<code>{page}</code></p>",
                status_code=500,
            )
        content = page.read_text(encoding="utf-8")
        # Önbellek kırıcı: Edge, `--app` penceresinde statik dosyaları önbelleğe
        # alır. Sunucu yeniden başladığında pencerede **eski** JavaScript kalır
        # ve yeni mantık (ör. onay senkronizasyonu) çalışmaz. Sürüm damgası
        # bunu engeller.
        stamp = f"{_asset_stamp()}"
        content = content.replace('/app.js"', f'/app.js?v={stamp}"')
        content = content.replace('/style.css"', f'/style.css?v={stamp}"')
        return HTMLResponse(content, headers={"Cache-Control": "no-store"})

    def _asset_stamp() -> str:
        """Arayüz dosyalarının son değişiklik zamanına dayalı sürüm damgası."""
        latest = 0.0
        for name in ("app.js", "style.css", "index.html"):
            candidate = web_root / name
            try:
                latest = max(latest, candidate.stat().st_mtime)
            except OSError:
                continue
        return str(int(latest))

    @app.get("/app.js")
    async def app_js() -> Any:
        """Arayüz betiği."""
        target = web_root / "app.js"
        if not target.exists():
            raise HTTPException(status_code=404, detail="app.js yok")
        return FileResponse(
            target, media_type="application/javascript; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/style.css")
    async def style_css() -> Any:
        """Arayüz stilleri."""
        target = web_root / "style.css"
        if not target.exists():
            raise HTTPException(status_code=404, detail="style.css yok")
        return FileResponse(
            target, media_type="text/css; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    # ─── Durum ──────────────────────────────────────────────────────────

    @app.get("/api/learning")
    async def get_learning(request: Request) -> Any:
        """
        Öğrenme durumu: kaç deneyim birikti, model ne durumda, eğitim hazır mı?

        Arayüzdeki **Öğrenme** paneli bunu kullanır; kullanıcı yerel modelin
        gerçekten gelişip gelişmediğini buradan görür.
        """
        _check_origin(request)
        if state.learner is None:
            return _json({"ok": False, "error": "öğrenme kaydedicisi yok"})
        # ⚠ Eğitim sürerken `status()` model/kilicinde bekleyebilir
        status = await _hizli_oku(state.learner.status, None, zaman_asimi=3.0)
        if status is None:
            return _json({"ok": False,
                          "error": "öğrenme durumu şu an okunamadı (meşgul)"},
                         status=503)
        status["ok"] = True
        status["recent"] = state.learner.records(limit=12)
        return _json(status)

    @app.post("/api/learning/train")
    async def post_learning_train(request: Request,
                                  payload: Dict[str, Any] = Body(default={})) -> Any:
        """
        Yerel modeli **hemen** eğitir (kullanıcı istediğinde).

        Eğitim uzun sürebilir; istek bloklanmasın diye ayrı iş parçacığında
        çalışır ve sonuç `learning.training_finished` olayıyla bildirilir.
        """
        _check_origin(request)
        if state.learner is None:
            raise HTTPException(status_code=503, detail="öğrenme kaydedicisi yok")
        if state.learner.is_training:
            return _json({"ok": False, "error": "eğitim zaten çalışıyor"}, status=409)

        options = payload or {}
        minutes = float(options.get("max_minutes") or 4.0)
        steps = options.get("steps")

        def run_training() -> None:
            """Eğitimi arka planda yürütür."""
            try:
                state.learner.train_now(max_minutes=minutes, steps=steps, force=True)
            except Exception as exc:
                logger.error(f"Eğitim hatası: {exc}", exc_info=True)
                state.events.publish("learning.training_finished", {
                    "ok": False, "error": f"{type(exc).__name__}: {exc}",
                })

        threading.Thread(target=run_training, name="NeuroCoreTrain",
                         daemon=True).start()
        return _json({"ok": True, "started": True,
                      "max_minutes": minutes, "steps": steps,
                      "message": "eğitim başlatıldı — sonuç öğrenme panelinde görünecek"})

    @app.post("/api/learning/feedback")
    async def post_learning_feedback(request: Request,
                                     payload: Dict[str, Any] = Body(...)) -> Any:
        """
        Son çalışmaya **kullanıcı puanı** verir (−1..1).

        En değerli öğrenme sinyalidir: model doğrudan düzeltilir. Arayüzdeki
        👍 / 👎 düğmeleri bunu çağırır.
        """
        _check_origin(request)
        if state.learner is None:
            raise HTTPException(status_code=503, detail="öğrenme kaydedicisi yok")
        reward = payload.get("reward")
        if reward is None:
            raise HTTPException(status_code=400, detail="reward gerekli")
        try:
            value = max(-1.0, min(1.0, float(reward)))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="reward sayı olmalı") from exc

        when = payload.get("at")
        result = state.learner.feedback(
            float(when) if when is not None else time.time(),
            value, note=str(payload.get("note") or ""),
        )
        return _json(result, status=200 if result.get("ok") else 404)

    @app.get("/api/state")
    async def get_state(request: Request) -> Any:
        """Genel durum: çalışma alanı, özerklik, model, araç sayıları."""
        _check_origin(request)
        # ⚠ Kilit tutabilecek okumalar: iş parçacığı + 2 sn zaman aşımı
        # ⚠ ÖLÇÜLDÜ: ilk `router.status()` 6,5 sn sürebiliyor; 2 sn bütçe
        # onu kesiyordu → UI "model yok" gösteriyordu (regresyon). Sağlayıcı
        # durumu için daha geniş bütçe, araç istatistikleri için dar bütçe.
        autonomy = (await _hizli_oku(state.registry.policy.to_dict, {},
                                     zaman_asimi=3.0)
                    if state.registry else {})
        llm = (await _hizli_oku(state.router.status, {}, zaman_asimi=8.0)
               if state.router else {})
        tools = (await _hizli_oku(state.registry.get_stats, {},
                                  zaman_asimi=2.5)
                 if state.registry else {})
        return _json({
            "ok": True,
            "workspace": state.workspace,
            "workspace_name": Path(state.workspace).name,
            "busy": state.is_busy(),
            "busy_since": state.busy_since,
            "uptime_seconds": round(state.uptime(), 1),
            "autonomy": autonomy,
            "llm": {
                "available": llm.get("available", False),
                "active": llm.get("active", ""),
                "model": llm.get("active_model", ""),
                "supports_tools": llm.get("supports_tools", False),
                "note": llm.get("note", ""),
                "stats": llm.get("stats", {}),
            },
            "agent_mode": state.loop.mode if state.loop else "rules",
            "tools": {
                "total": tools.get("tools", 0),
                "by_category": tools.get("by_category", {}),
                "calls": tools.get("calls", 0),
                "blocked": tools.get("blocked", 0),
                "failed": tools.get("failed", 0),
            },
            "approvals": state.approvals.status(),
            "events": state.events.status(),
            "modules": _module_status(),
            "last_run": state.last_run,
        })

    def _module_status() -> Dict[str, Any]:
        """Yetenek modüllerinin durumu (Godot/Blender/görsel)."""
        try:
            from neurocore.agent import available_modules
            modules = available_modules()["modules"]
        except Exception:
            modules = {}
        return {
            "coding": state.coding is not None,
            "shell": state.shell is not None,
            "vision": state.vision is not None,
            "godot": state.godot is not None,
            "blender": state.blender is not None,
            "details": modules,
        }

    @app.get("/api/health")
    async def get_health(request: Request) -> Any:
        """Sağlık raporu: hangi bileşen hazır, ne eksik?"""
        _check_origin(request)
        issues: List[str] = []
        capabilities: Dict[str, bool] = {}

        if state.router is None or not state.router.available:
            issues.append(
                "Dil modeli yapılandırılmadı — ajan kural tabanlı modda çalışır. "
                "Ayarlar'dan bir API anahtarı girin."
            )
        capabilities["llm"] = bool(state.router and state.router.available)
        capabilities["tools"] = bool(state.registry and len(state.registry) > 0)
        capabilities["coding"] = state.coding is not None
        capabilities["shell"] = state.shell is not None
        capabilities["vision"] = state.vision is not None
        capabilities["godot"] = state.godot is not None
        capabilities["blender"] = state.blender is not None

        if state.vision is not None:
            try:
                vision_health = state.vision.health()
                capabilities["ocr"] = bool(
                    vision_health.get("capabilities", {}).get("ocr")
                )
                capabilities["ui_text"] = bool(
                    vision_health.get("capabilities", {}).get("ui_text")
                )
            except Exception:
                pass

        if state.shell is not None:
            try:
                versions = state.shell.toolchain_versions()
                for name in ("git", "node"):
                    if not versions.get(name, {}).get("available"):
                        issues.append(f"{name} kurulu değil — ilgili yetenekler sınırlı")
            except Exception:
                pass

        return _json({
            "ok": not issues or capabilities["llm"],
            "capabilities": capabilities,
            "issues": issues,
            "workspace": state.workspace,
            "uptime_seconds": round(state.uptime(), 1),
        })

    # ─── Sohbet / Görev ─────────────────────────────────────────────────

    @app.post("/api/chat")
    async def post_chat(request: Request, payload: Dict[str, Any] = Body(...)) -> Any:
        """
        Görevi başlatır ve **hemen** döner.

        Sonuç WebSocket üzerinden akar (`agent.step`, `agent.run_finished`).
        Böylece uzun görevler arayüzü bloklamaz.
        """
        _check_origin(request)
        message = str(payload.get("message") or payload.get("task") or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="mesaj boş olamaz")
        if state.loop is None:
            raise HTTPException(status_code=503, detail="ajan döngüsü hazır değil")
        if state.is_busy():
            return _json({
                "ok": False,
                "error": "ajan şu anda çalışıyor — bekleyin veya iptal edin",
                "busy_since": state.busy_since,
            }, status=409)

        session_id = str(payload.get("session_id") or "default")
        # ⚠ 24 adım yetmiyordu (bkz. AgentLoop.VARSAYILAN_ADIM)
        max_steps = int(payload.get("max_steps") or 32)
        max_seconds = float(payload.get("max_seconds") or 900.0)
        plan_first = bool(payload.get("plan_first", False))

        # ── Sohbet bağlamı ──
        # Mesajı sohbet deposuna yaz: soldaki listede görünür ve sohbetin
        # kendi çalışma alanı korunur. Sohbet yoksa **otomatik oluşturulur**
        # ki kullanıcı elle "yeni sohbet" demek zorunda kalmasın.
        if state.chats is not None:
            if not state.chats.get(session_id):
                chat = state.chats.create(
                    workspace=str(payload.get("workspace") or "").strip()
                )
                session_id = chat.id
                state.active_chat = chat.id
            state.chats.append(session_id, "user", message)
            chat = state.chats.get(session_id)
            # Sohbetin klasörü varsa çalışma alanı ona uysun (sohbet bazlı
            # çalışma alanı kuralı)
            if chat is not None and chat.workspace \
                    and state.workspace != chat.workspace:
                apply_workspace(state, chat.workspace)
            state.events.publish("chats.updated", {"id": session_id})

        history = _session_history(session_id)
        state.sessions.setdefault(session_id, []).append(
            {"role": "user", "content": message}
        )

        state.set_busy(True)
        state.events.publish("agent.run_started", {
            "message": message, "session_id": session_id,
            "max_steps": max_steps, "plan_first": plan_first,
            "mode": state.loop.mode,
            "chat_id": session_id,
            "workspace": state.workspace,
            "workspace_name": Path(state.workspace).name if state.workspace else "",
        })

        thread = threading.Thread(
            target=_run_agent_sync,
            args=(message, session_id, max_steps, max_seconds, history, plan_first),
            name="NeuroCoreAgentRun",
            daemon=True,
        )
        thread.start()

        return _json({
            "ok": True,
            "started": True,
            "session_id": session_id,
            "mode": state.loop.mode,
            "message": message,
        })

    def _session_history(session_id: str) -> List[Any]:
        """Oturum geçmişini LLM mesajlarına çevirir."""
        from neurocore.agent.llm import LLMMessage

        entries = state.sessions.get(session_id, [])[-8:]
        return [LLMMessage(entry["role"], entry["content"]) for entry in entries]

    def _run_agent_sync(message: str, session_id: str, max_steps: int,
                        max_seconds: float, history: List[Any],
                        plan_first: bool) -> None:
        """
        Ajanı iş parçacığında çalıştırır (WebSocket olayları yayar).

        Neden iş parçacığı? `AgentLoop.run` senkron ve bloklayıcıdır (araçlar
        alt süreç çalıştırır, kullanıcı onayı bekler). FastAPI'nin olay
        döngüsünü bloklamak arayüzü tamamen dondururdu.
        """
        started = time.perf_counter()
        try:
            # ── Çalışma alanını **çalışma anında** garanti et ──
            # Sohbetin klasörü varsa araçlar o klasörde çalışmalıdır. Arayüz
            # ile sunucu arasındaki bir yarış (sohbet değiştirme, sayfa
            # yenileme) alanı başka bir köke kaydırabilir; bu yüzden görev
            # başlamadan hemen önce yeniden uygulanır.
            if state.chats is not None:
                chat = state.chats.get(session_id)
                if chat is not None and chat.workspace \
                        and state.workspace != chat.workspace:
                    logger.info(
                        f"Çalışma alanı sohbete göre düzeltiliyor: "
                        f"{state.workspace} → {chat.workspace}"
                    )
                    apply_workspace(state, chat.workspace)

            project_context = _project_context()
            run = state.loop.run(
                message, max_steps=max_steps, max_seconds=max_seconds,
                history=history, project_context=project_context,
                plan_first=plan_first,
            )
            payload = run.to_dict(include_steps=True)
            payload["duration_ms"] = (time.perf_counter() - started) * 1000
            state.last_run = {
                "task": message, "answer": run.answer, "ok": run.ok,
                "mode": run.mode, "steps": len(run.steps),
                "duration_ms": payload["duration_ms"],
                "reason": run.reason,
                "tools_used": run.tools_used,
                "files_changed": run.files_changed,
                "at": time.time(),
            }
            state.sessions.setdefault(session_id, []).append(
                {"role": "assistant", "content": run.answer}
            )
            if len(state.sessions.get(session_id, [])) > 40:
                state.sessions[session_id] = state.sessions[session_id][-40:]
            # Yanıtı da sohbete yaz (geçmiş yeniden açıldığında görünsün)
            if state.chats is not None and state.chats.get(session_id):
                state.chats.append(session_id, "assistant", run.answer or "")
            payload["chat_id"] = session_id
            state.events.publish("agent.run_finished", payload)
        except Exception as exc:
            logger.error(f"Ajan çalışması çöktü: {exc}", exc_info=True)
            state.events.publish("agent.run_finished", {
                "ok": False, "task": message, "answer": "",
                "error": f"{type(exc).__name__}: {exc}",
                "reason": "beklenmeyen hata", "steps": [],
            })
        finally:
            state.set_busy(False)

    def _project_context() -> str:
        """
        Ajana verilecek kısa proje özeti.

        Her çalışmada tam harita çıkarmak pahalıdır; bu yüzden yalnızca
        tür, dosya sayısı, diller ve anahtar dosyalar özetlenir.
        """
        if state.coding is None:
            return ""
        try:
            info = state.coding.workspace_map(max_files=60, max_depth=3)
            return (
                f"Tür: {info.get('project_type')} ({info.get('project_label')})\n"
                f"Dosya: {info.get('file_count')} ({info.get('size_mb')} MB)\n"
                f"Diller: {json.dumps(info.get('languages', {}), ensure_ascii=False)}\n"
                f"Anahtar dosyalar: {', '.join(info.get('key_files', [])[:8])}\n"
                f"Giriş noktaları: {', '.join(info.get('entry_points', [])[:6])}"
            )
        except Exception as exc:
            logger.debug(f"Proje bağlamı üretilemedi: {exc}")
            return ""

    @app.get("/api/study")
    async def get_study(request: Request) -> Any:
        """
        Kendi kendini geliştirme durumu: öğretmen model, işlenen konular,
        son öğrenme turları.
        """
        _check_origin(request)
        if state.study is None:
            return _json({"ok": False, "error": "kendi kendine öğrenme kapalı"})
        try:
            status = state.study.status()
        except Exception as exc:
            return _json({"ok": False, "error": str(exc)}, status=500)
        status["ok"] = True
        return _json(status)

    @app.post("/api/study/run")
    async def post_study_run(request: Request,
                             payload: Dict[str, Any] = Body(default={})) -> Any:
        """
        **Şimdi bir öğrenme turu** çalıştırır (kullanıcı istediğinde).

        Tur uzun sürebilir (soru üretimi + kaynak indirme + cevap), bu yüzden
        ayrı iş parçacığında çalışır; sonuç `study.round_finished` olayıyla
        bildirilir.
        """
        _check_origin(request)
        if state.study is None:
            raise HTTPException(status_code=503, detail="kendi kendine öğrenme kapalı")
        if state.study.is_studying:
            return _json({"ok": False, "error": "öğrenme turu zaten çalışıyor"},
                         status=409)

        topic = str((payload or {}).get("topic") or "").strip()

        def run_round() -> None:
            """Turu arka planda yürütür."""
            try:
                state.study.study(topic)
            except Exception as exc:
                logger.error(f"Öğrenme turu hatası: {exc}", exc_info=True)
                state.events.publish("study.round_finished", {
                    "ok": False, "error": f"{type(exc).__name__}: {exc}",
                })

        threading.Thread(target=run_round, name="NeuroCoreStudy",
                         daemon=True).start()
        return _json({"ok": True, "started": True, "topic": topic,
                      "message": "öğrenme turu başlatıldı"})

    @app.post("/api/study/topics")
    async def post_study_topics(request: Request,
                                payload: Dict[str, Any] = Body(...)) -> Any:
        """Öğrenilecek konuları ayarlar (öğretmenin odaklanacağı alanlar)."""
        _check_origin(request)
        if state.study is None:
            raise HTTPException(status_code=503, detail="kendi kendine öğrenme kapalı")
        raw = payload.get("topics")
        if isinstance(raw, str):
            topics = [line.strip() for line in raw.splitlines() if line.strip()]
        elif isinstance(raw, list):
            topics = [str(item).strip() for item in raw if str(item).strip()]
        else:
            raise HTTPException(status_code=400, detail="topics liste veya metin olmalı")
        state.study._topics = topics[:50]
        logger.info(f"Öğrenme konuları ayarlandı: {len(topics)} konu")
        return _json({"ok": True, "topics": state.study._topics})

    @app.get("/api/work/approvals")
    async def work_approvals(request: Request) -> Any:
        """
        Onay bekleyen adımlar.

        Denetimli modda yazma/çalıştırma araçları onay ister. Arayüz bunu
        gösterir; kullanıcı onaylayınca iş **kaldığı yerden** devam eder.
        """
        _check_origin(request)
        bridge = getattr(state, "bridge", None)
        if bridge is None:
            return _json({"ok": False, "error": "parçalı motor kurulu değil"})
        bekleyen = bridge.pending_approvals()
        return _json({"ok": True, "pending": bekleyen, "count": len(bekleyen)})

    @app.post("/api/work/approve")
    async def work_approve(request: Request) -> Any:
        """
        Onay bekleyen adımı **onaylar** veya **reddeder**.

        Gövde: `{"step_id": "...", "allow": true}`

        Onaylanan adım yeniden çalıştırılır; reddedilen adım atlanır ve
        iş **durma** noktasından devam eder (baştan başlamaz).
        """
        _check_origin(request)
        bridge = getattr(state, "bridge", None)
        if bridge is None:
            return _json({"ok": False, "error": "parçalı motor kurulu değil"})
        try:
            body = await request.json()
        except Exception:
            return _json({"ok": False, "error": "geçersiz JSON"})
        kimlik = str(body.get("step_id") or "")
        if not kimlik:
            return _json({"ok": False, "error": "step_id gerekli"})
        izin = bool(body.get("allow", True))
        basarili = (bridge.approve(kimlik) if izin else bridge.deny(kimlik))
        return _json({
            "ok": basarili,
            "step_id": kimlik,
            "action": "onaylandı" if izin else "reddedildi",
            "error": "" if basarili else "bu adım onay beklemiyor",
            "pending": len(bridge.pending_approvals()),
        })

    @app.get("/api/work/status")
    async def work_status(request: Request) -> Any:
        """
        Parçalı iş motorunun durumu.

        Arayüz bunu kullanarak "hangi adımda, kaç parça, ne kadar kuyrukta"
        bilgisini gösterir. Kullanıcı uzun bir işin nerede olduğunu görür.
        """
        _check_origin(request)
        bridge = getattr(state, "bridge", None)
        if bridge is None:
            return _json({"ok": False, "error": "parçalı motor kurulu değil"})
        motor = bridge.engine
        return _json({
            "ok": True,
            "report": motor.report(),
            "deferred": motor.deferred_summary(),
            "governor": motor.governor.plan().to_dict(),
            "bridge": bridge.get_stats(),
            "approvals": bridge.pending_approvals(),
            "steps": [a.to_dict() for a in motor.steps[-40:]],
            "chunks": [c.to_dict() for c in motor.chunks[-40:]],
        })

    @app.post("/api/work/plan")
    async def work_plan(request: Request) -> Any:
        """
        Bir görevi **planlayıp** çalıştırır (parçalı).

        Gövde:
            {"objective": "...", "plan": [{"title", "tool", "args", ...}],
             "run": true, "max_chunks": 0}

        `plan` adımları `AgentBridge.plan_steps` ile `WorkStep`'e çevrilir:
        kaynak ihtiyacı araç türünden tahmin edilir, `needs_previous` olan
        adımlar zincirlenir, bağımsızlar **paralel** çalışır.
        """
        _check_origin(request)
        bridge = getattr(state, "bridge", None)
        if bridge is None:
            return _json({"ok": False, "error": "parçalı motor kurulu değil"})
        try:
            body = await request.json()
        except Exception:
            return _json({"ok": False, "error": "geçersiz JSON"})
        amaç = str(body.get("objective") or "görev")
        plan = body.get("plan") or []
        if not isinstance(plan, list) or not plan:
            return _json({"ok": False, "error": "plan listesi gerekli"})

        def _iş() -> None:
            """Ağır iş arka planda (istek bloklanmasın)."""
            try:
                import threading

                def kos() -> None:
                    try:
                        bridge.plan_steps(amaç, plan)
                        if body.get("run", True):
                            bridge.run(max_chunks=int(body.get("max_chunks") or 0))
                    except Exception as exc:
                        logger.error(f"Parçalı iş çöktü: {exc}", exc_info=True)

                threading.Thread(target=kos, name="nc-work-plan",
                                 daemon=True).start()
            except Exception as exc:
                logger.error(f"Parçalı iş başlatılamadı: {exc}")

        _iş()
        # Kullanıcıya **neyin** beklendiğini açıkça söyle: Denetimli modda
        # yazma/çalıştırma araçları onay ister ve onay verilmezse adım
        # "başarısız" olur. Bu, sessizce yanlış sonuç vermekten iyidir ama
        # kullanıcı nedenini bilmelidir.
        yazan = sorted({str(a.get("tool") or "") for a in plan
                        if str(a.get("tool") or "").startswith(("code.create",
                                                                "code.edit",
                                                                "code.delete",
                                                                "shell.",
                                                                "godot.",
                                                                "blender."))})
        özerklik = ""
        try:
            özerklik = str(state.autonomy.label)
        except Exception:
            özerklik = "bilinmiyor"
        return _json({
            "ok": True, "planned": len(plan),
            "message": "plan kuruldu ve arka planda çalışıyor",
            "autonomy": özerklik,
            "approval_may_be_needed": yazan,
            "note": ("Denetimli/Güvenli modda yazma ve çalıştırma araçları "
                     "onay gerektirir; onay verilmezse o adımlar başarısız "
                     "olur. Onay gerektirmeyen çalıştırma için özerkliği "
                     "'Güvenilir' yapın veya araçları önceden onaylayın."
                     ) if yazan else "",
        })

    @app.post("/api/work/resume")
    async def work_resume(request: Request) -> Any:
        """
        Kesintiden sonra **kalan** adımlarla devam eder.

        Günlük (`work_journal.jsonl`) okunur; tamamlanan adımlar atlanır,
        kalanlar yeniden planlanır ve çalıştırılır.
        """
        _check_origin(request)
        bridge = getattr(state, "bridge", None)
        if bridge is None:
            return _json({"ok": False, "error": "parçalı motor kurulu değil"})
        durum = bridge.resume_state()
        try:
            import threading

            def kos() -> None:
                try:
                    bridge.resume()
                except Exception as exc:
                    logger.error(f"Devam çöktü: {exc}", exc_info=True)

            threading.Thread(target=kos, name="nc-work-resume",
                             daemon=True).start()
        except Exception as exc:
            return _json({"ok": False, "error": str(exc)})
        return _json({"ok": True, "journal": durum,
                      "message": "kalan adımlarla devam ediliyor"})

    @app.get("/api/work/algorithm")
    async def work_algorithm(request: Request, limit: int = 0) -> Any:
        """
        Yapılan işi **algoritma** olarak döndürür (adım adım, süreleriyle).
        """
        _check_origin(request)
        bridge = getattr(state, "bridge", None)
        if bridge is None:
            return _json({"ok": False, "error": "parçalı motor kurulu değil"})
        return _json({"ok": True,
                      "algorithm": bridge.algorithm(limit=int(limit or 0))})

    @app.get("/api/processes")
    async def get_processes(request: Request) -> Any:
        """
        Şu anda çalışan alt süreçler (komut, süre, PID).

        Arayüz bunu kullanarak "çalışıyor" göstergesini çizer; kullanıcı
        hangi komutun çalıştığını görür ve tek tek durdurabilir.
        """
        _check_origin(request)
        running: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {}
        if state.shell is not None:
            try:
                running = state.shell.running_processes()
                stats = state.shell.get_stats()
            except Exception as exc:
                logger.debug(f"Çalışan süreçler alınamadı: {exc}")
        return _json({
            "ok": True,
            "running": running,
            "count": len(running),
            "cancel_requested": bool(stats.get("cancel_requested", False)),
            "killed": int(stats.get("killed", 0) or 0),
            "session_alive": bool(stats.get("session_alive", False)),
        })

    @app.post("/api/processes/kill")
    async def post_processes_kill(request: Request,
                                  payload: Dict[str, Any] = Body(default={})) -> Any:
        """
        Çalışan alt süreçleri sonlandırır.

        `payload.pid` verilirse yalnızca o süreç; verilmezse **hepsi**
        (alt süreç ağacıyla) öldürülür.
        """
        _check_origin(request)
        if state.shell is None:
            raise HTTPException(status_code=503, detail="kabuk katmanı yok")
        pid = (payload or {}).get("pid")
        if pid:
            try:
                target = int(pid)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="pid sayı olmalı") from exc
            killed = 0
            for entry in state.shell.running_processes():
                if int(entry.get("pid") or 0) == target:
                    state.shell._kill_pid_tree(target)
                    state.shell._unregister_process(target)
                    killed = 1
                    break
            state.events.publish("shell.processes_killed",
                                 {"count": killed, "pid": target})
            return _json({"ok": True, "killed": killed, "pid": target})
        killed = state.shell.kill_all("kullanıcı isteği")
        state.events.publish("shell.processes_killed", {"count": killed})
        return _json({"ok": True, "killed": killed})

    @app.post("/api/cancel")
    async def post_cancel(request: Request) -> Any:
        """
        Çalışan görevi iptal eder **ve çalışan komutları öldürür**.

        Yalnızca bayrak kaldırmak yetmez: o anda çalışan bir komut (ör. 60
        saniyelik bir sunucu) iptalden sonra da devam ederdi. `loop.cancel()`
        iptal geri çağrımlarını tetikler; bunlar alt süreçleri sonlandırır.
        """
        _check_origin(request)
        if state.loop is None:
            return _json({"ok": False, "error": "ajan hazır değil"})
        before = 0
        if state.shell is not None:
            try:
                before = len(state.shell.running_processes())
            except Exception:
                before = 0
        state.loop.cancel()
        released = state.approvals.resolve_all(approved=False)
        after = 0
        if state.shell is not None:
            try:
                after = len(state.shell.running_processes())
            except Exception:
                after = 0
        killed = max(0, before - after)
        state.events.publish("agent.cancelled", {
            "pending_approvals": released, "killed_processes": killed,
        })
        return _json({"ok": True, "cancelled": True,
                      "released_approvals": released,
                      "killed_processes": killed})

    @app.get("/api/runs")
    async def get_runs(request: Request) -> Any:
        """Son çalışmanın özeti."""
        _check_origin(request)
        return _json({"ok": True, "last_run": state.last_run})

    # ─── Sohbetler ──────────────────────────────────────────────────────

    @app.get("/api/chats")
    async def get_chats(request: Request) -> Any:
        """
        Sohbetleri listeler (son güncellenene göre).

        Her sohbetin bağlı olduğu **çalışma alanı** da döner; arayüz soldaki
        listede klasör adını gösterir.
        """
        _check_origin(request)
        if state.chats is None:
            return _json({"ok": False, "chats": [], "error": "sohbet deposu yok"})
        return _json({"ok": True, "chats": state.chats.list(limit=100),
                      "stats": state.chats.get_stats(),
                      "active": state.active_chat})

    @app.post("/api/chats")
    async def post_chats(request: Request,
                         payload: Dict[str, Any] = Body(default={})) -> Any:
        """
        Yeni sohbet oluşturur.

        `workspace` verilirse sohbet o klasöre bağlanır ve çalışma alanı
        **hemen** o klasöre geçer. Verilmezse genel sohbet modu çalışır.
        """
        _check_origin(request)
        if state.chats is None:
            raise HTTPException(status_code=503, detail="sohbet deposu yok")

        raw_workspace = str((payload or {}).get("workspace") or "").strip()
        workspace = ""
        if raw_workspace:
            target = Path(os.path.expandvars(os.path.expanduser(raw_workspace)))
            if not target.is_dir():
                return _json({"ok": False, "error": f"dizin yok: {target}"},
                             status=400)
            workspace = str(target.resolve())

        chat = state.chats.create(
            title=str((payload or {}).get("title") or "").strip(),
            workspace=workspace,
        )
        state.active_chat = chat.id
        switched: List[str] = []
        if workspace:
            switched = apply_workspace(state, workspace)
        state.events.publish("chats.created", chat.to_dict())
        return _json({"ok": True, "chat": chat.to_dict(),
                      "updated_components": switched})

    @app.get("/api/chats/{chat_id}")
    async def get_chat(request: Request, chat_id: str) -> Any:
        """Tek sohbeti mesajlarıyla döndürür."""
        _check_origin(request)
        if state.chats is None:
            raise HTTPException(status_code=503, detail="sohbet deposu yok")
        chat = state.chats.get(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="sohbet yok")
        return _json({"ok": True, "chat": chat.to_dict(with_messages=True,
                                                       message_limit=120)})

    @app.post("/api/chats/{chat_id}/activate")
    async def post_chat_activate(request: Request, chat_id: str) -> Any:
        """
        Sohbeti **etkin** yapar ve çalışma alanını ona göre ayarlar.

        Sohbete klasör bağlıysa çalışma alanı o klasöre geçer; bağlı değilse
        genel çalışma alanına dönülür (genel sohbet modu).
        """
        _check_origin(request)
        if state.chats is None:
            raise HTTPException(status_code=503, detail="sohbet deposu yok")
        chat = state.chats.get(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="sohbet yok")

        state.active_chat = chat.id
        switched: List[str] = []
        if chat.workspace:
            switched = apply_workspace(state, chat.workspace)
        else:
            # Genel mod: proje köküne dön
            root = state.general_workspace or state.workspace
            if root:
                switched = apply_workspace(state, root)
        state.events.publish("chats.activated", {
            "id": chat.id, "workspace": state.workspace,
            "general": not chat.workspace,
        })
        return _json({"ok": True, "chat": chat.to_dict(with_messages=True,
                                                       message_limit=120),
                      "workspace": state.workspace,
                      "general": not chat.workspace,
                      "updated_components": switched})

    @app.post("/api/chats/{chat_id}/workspace")
    async def post_chat_workspace(request: Request, chat_id: str,
                                  payload: Dict[str, Any] = Body(...)) -> Any:
        """
        Sohbetin çalışma alanını değiştirir.

        Boş `path` → sohbet **genel moda** alınır (klasör bağlantısı kaldırılır).
        """
        _check_origin(request)
        if state.chats is None:
            raise HTTPException(status_code=503, detail="sohbet deposu yok")
        chat = state.chats.get(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="sohbet yok")

        raw_path = str(payload.get("path") or "").strip()
        switched: List[str] = []
        if raw_path:
            target = Path(os.path.expandvars(os.path.expanduser(raw_path)))
            if not target.is_dir():
                return _json({"ok": False, "error": f"dizin yok: {target}"},
                             status=400)
            chat = state.chats.set_workspace(chat_id, str(target.resolve()))
            # Etkin sohbetse çalışma alanı hemen değişsin
            if state.active_chat == chat_id:
                switched = apply_workspace(state, chat.workspace)
        else:
            chat = state.chats.set_workspace(chat_id, "")
            if state.active_chat == chat_id:
                root = state.general_workspace or state.workspace
                if root:
                    switched = apply_workspace(state, root)

        state.events.publish("chats.workspace_changed", {
            "id": chat_id, "workspace": (chat.workspace if chat else ""),
        })
        return _json({"ok": True, "chat": chat.to_dict() if chat else {},
                      "workspace": state.workspace,
                      "updated_components": switched})

    @app.post("/api/chats/{chat_id}/rename")
    async def post_chat_rename(request: Request, chat_id: str,
                               payload: Dict[str, Any] = Body(...)) -> Any:
        """Sohbet başlığını değiştirir."""
        _check_origin(request)
        if state.chats is None:
            raise HTTPException(status_code=503, detail="sohbet deposu yok")
        title = str(payload.get("title") or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="başlık gerekli")
        chat = state.chats.rename(chat_id, title)
        if chat is None:
            raise HTTPException(status_code=404, detail="sohbet yok")
        state.events.publish("chats.renamed", {"id": chat_id, "title": chat.title})
        return _json({"ok": True, "chat": chat.to_dict()})

    @app.post("/api/chats/{chat_id}/clear")
    async def post_chat_clear(request: Request, chat_id: str) -> Any:
        """Sohbetin mesajlarını temizler (klasör bağlı kalır)."""
        _check_origin(request)
        if state.chats is None:
            raise HTTPException(status_code=503, detail="sohbet deposu yok")
        chat = state.chats.clear(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="sohbet yok")
        state.sessions.pop(chat_id, None)
        state.events.publish("chats.cleared", {"id": chat_id})
        return _json({"ok": True, "chat": chat.to_dict()})

    @app.delete("/api/chats/{chat_id}")
    async def delete_chat(request: Request, chat_id: str) -> Any:
        """Sohbeti siler."""
        _check_origin(request)
        if state.chats is None:
            raise HTTPException(status_code=503, detail="sohbet deposu yok")
        if not state.chats.delete(chat_id):
            raise HTTPException(status_code=404, detail="sohbet yok")
        state.sessions.pop(chat_id, None)
        if state.active_chat == chat_id:
            state.active_chat = ""
        state.events.publish("chats.deleted", {"id": chat_id})
        return _json({"ok": True, "deleted": chat_id})

    # ─── Araçlar ────────────────────────────────────────────────────────

    @app.get("/api/tools")
    async def get_tools(request: Request, category: str = "",
                        query: str = "") -> Any:
        """Kayıtlı araçları listeler."""
        _check_origin(request)
        if state.registry is None:
            return _json({"ok": False, "tools": [], "error": "kayıt defteri yok"})
        return _json({
            "ok": True,
            "tools": state.registry.list_tools(
                category=category or None, query=query
            ),
            "categories": state.registry.categories(),
            "policy": state.registry.policy.to_dict(),
        })

    @app.post("/api/tool")
    async def post_tool(request: Request, payload: Dict[str, Any] = Body(...)) -> Any:
        """
        Tek bir aracı doğrudan çağırır (ajan döngüsü olmadan).

        Arayüzdeki "araçları dene" paneli bunu kullanır; ayrıca bir görevi
        tek adımda çalıştırmak için de yararlıdır.

        Onay gerekiyorsa istek **bloklanmaz**: çağrı arka planda bekletilir,
        yanıt `needs_approval=True` ile hemen döner ve kullanıcı karar
        verdikten sonra sonuç `tool.executed` olayıyla arayüze gönderilir.
        Böylece bir onay beklerken HTTP isteği zaman aşımına uğramaz.
        """
        _check_origin(request)
        if state.registry is None:
            raise HTTPException(status_code=503, detail="kayıt defteri yok")
        name = str(payload.get("tool") or "").strip()
        args = payload.get("args") or {}
        if not name:
            raise HTTPException(status_code=400, detail="araç adı gerekli")
        if not isinstance(args, dict):
            raise HTTPException(status_code=400, detail="args sözlük olmalı")

        result = state.registry.call(name, args)

        if result.needs_approval:
            ticket = f"tool_{int(time.time() * 1000)}"
            thread = threading.Thread(
                target=_await_approval_and_run,
                args=(ticket, name, args, result.reason),
                name="NeuroCoreToolApproval",
                daemon=True,
            )
            thread.start()
            return _json({
                "ok": False,
                "needs_approval": True,
                "blocked": True,
                "ticket": ticket,
                "tool": name,
                "safety": result.safety,
                "safety_label": _safety_label(result.safety),
                "reason": result.reason,
                "message": "onay bekleniyor — karar verildiğinde sonuç "
                           "tool.executed olayıyla bildirilecek",
            })

        return _json({
            "ok": result.ok,
            "result": result.to_dict(),
            "render": result.render(max_chars=8000),
        })

    def _await_approval_and_run(ticket: str, name: str, args: Dict[str, Any],
                                reason: str) -> None:
        """
        Onay bekler, karar gelince aracı çalıştırır ve sonucu yayınlar.

        Neden ayrı iş parçacığı? HTTP isteği onay boyunca açık kalamaz
        (kullanıcı dakikalarca düşünebilir ve istemci zaman aşımına uğrar).
        """
        from neurocore.agent.tools import describe_tool_call

        safety = "execute"
        tool = state.registry.get(name) if state.registry else None
        if tool is not None:
            safety = tool.safety.value

        approved, remember = state.approvals.request(
            name, args, safety, _safety_label(safety), reason,
            # Ajan döngüsüyle **aynı** açıklama üretilir: kullanıcı, işi
            # ajan mı yoksa panel mi başlattı, farklı metin görmemeli.
            describe_tool_call(name, args),
        )

        payload: Dict[str, Any] = {
            "ticket": ticket, "tool": name, "args": args,
            "approved": approved, "remember": remember, "at": time.time(),
        }
        if not approved:
            payload.update({"ok": False, "denied": True,
                            "error": "kullanıcı onay vermedi"})
        else:
            state.registry.approve(name, remember=remember, consume=True)
            outcome = state.registry.call(name, args)
            payload.update({
                "ok": outcome.ok,
                "result": outcome.to_dict(),
                "render": outcome.render(max_chars=8000),
            })

        state.store_tool_result(ticket, payload)
        state.events.publish("tool.executed", payload)
        logger.info(
            f"Araç sonucu yayınlandı: {name} → "
            f"{'onaylandı' if approved else 'reddedildi'}"
        )

    def _safety_label(safety: str) -> str:
        """Güvenlik sınıfının Türkçe etiketi."""
        from neurocore.agent.tools import SAFETY_LABELS, Safety

        try:
            return SAFETY_LABELS.get(Safety(safety), safety)
        except ValueError:
            return safety

    @app.get("/api/tool/result")
    async def get_tool_result(request: Request, key: str) -> Any:
        """Onay sonrası tamamlanan bir araç çağrısının sonucunu döndürür."""
        _check_origin(request)
        stored = state.get_tool_result(key)
        if stored is None:
            return _json({"ok": False, "error": "sonuç bulunamadı"}, status=404)
        return _json({"ok": True, "result": stored})

    # ─── Özerklik ───────────────────────────────────────────────────────

    @app.get("/api/autonomy")
    async def get_autonomy(request: Request) -> Any:
        """Özerklik seviyelerini ve aktif olanı döner."""
        _check_origin(request)
        if state.registry is None:
            raise HTTPException(status_code=503, detail="kayıt defteri yok")
        from neurocore.agent.tools import AutonomyPolicy

        return _json({
            "ok": True,
            "levels": AutonomyPolicy.levels(),
            "current": state.registry.policy.to_dict(),
            "audit": state.registry.audit(limit=40),
        })

    @app.post("/api/autonomy")
    async def post_autonomy(request: Request, payload: Dict[str, Any] = Body(...)) -> Any:
        """
        Özerklik seviyesini değiştirir.

        **Bu, kullanıcının en önemli denetim aracıdır.** Seviye düşürüldüğünde
        bekleyen onaylar iptal edilir; yükseltildiğinde mevcut onaylar geçersiz
        sayılır (kullanıcı yeni seviyeyi bilinçli seçmiş olur).
        """
        _check_origin(request)
        if state.registry is None:
            raise HTTPException(status_code=503, detail="kayıt defteri yok")
        level = str(payload.get("level") or "").strip()
        if not level:
            raise HTTPException(status_code=400, detail="seviye gerekli")

        released = state.approvals.resolve_all(approved=False)
        try:
            info = state.registry.set_level(level)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Ek politika ayarları
        policy = state.registry.policy
        if "dry_run" in payload:
            policy.dry_run = bool(payload["dry_run"])
        if "allow" in payload and isinstance(payload["allow"], list):
            policy.allow = [str(item) for item in payload["allow"]]
        if "deny" in payload and isinstance(payload["deny"], list):
            policy.deny = [str(item) for item in payload["deny"]]

        logger.info(f"Özerklik seviyesi arayüzden değişti: {info['label']}")
        state.events.publish("agent.autonomy_changed", info)
        return _json({"ok": True, "autonomy": info,
                      "released_approvals": released})

    # ─── Onay ───────────────────────────────────────────────────────────

    @app.post("/api/shutdown")
    async def post_shutdown(request: Request) -> Any:
        """
        Uygulamayı **temiz biçimde** kapatır (başlatıcı betiği kullanır).

        Neden gerekli? `Run.bat` çift tıklandığında birden fazla örnek
        birikebilir ve hiçbiri diğerinden haberdar olmaz. Temiz bir kapatma
        uç noktası, kullanıcının tek komutla hepsini kapatmasını sağlar;
        süreç öldürmek (taskkill) ise kabuk oturumlarını ve geçici dosyaları
        yarım bırakır.

        Bu uç nokta yalnızca yerel arayüzden çağrılabilir (origin denetimi).
        """
        _check_origin(request)
        logger.info("Kapatma isteği alındı (API)")

        # Bekleyen onayları reddet: yoksa iş parçacıkları zaman aşımını bekler
        released = state.approvals.resolve_all(approved=False)
        if state.loop is not None:
            state.loop.cancel()

        def stop_soon() -> None:
            """Yanıtın gitmesini bekleyip sunucuyu durdurur."""
            time.sleep(0.6)
            try:
                if state.shutdown_callback is not None:
                    state.shutdown_callback()
            except Exception as exc:
                logger.debug(f"Kapatma geri çağrımı hata verdi: {exc}")

        threading.Thread(target=stop_soon, name="NeuroCoreShutdown",
                         daemon=True).start()
        return _json({"ok": True, "shutting_down": True,
                      "released_approvals": released})

    @app.get("/api/approvals")
    async def get_approvals(request: Request) -> Any:
        """Bekleyen onay istekleri."""
        _check_origin(request)
        return _json({"ok": True, "pending": state.approvals.pending(),
                      "status": state.approvals.status()})

    @app.post("/api/approve")
    async def post_approve(request: Request, payload: Dict[str, Any] = Body(...)) -> Any:
        """Onay isteğini yanıtlar."""
        _check_origin(request)
        approval_id = str(payload.get("id") or "").strip()
        approved = bool(payload.get("approved", False))
        remember = bool(payload.get("remember", False))

        if approval_id == "*":
            count = state.approvals.resolve_all(approved=approved, remember=remember)
            return _json({"ok": True, "resolved": count})

        if not approval_id:
            raise HTTPException(status_code=400, detail="onay kimliği gerekli")
        result = state.approvals.resolve(approval_id, approved, remember)
        return _json(result, status=200 if result.get("ok") else 404)

    # ─── Ayarlar ────────────────────────────────────────────────────────

    @app.get("/api/settings")
    async def get_settings(request: Request) -> Any:
        """Dil modeli ve çalışma alanı ayarları."""
        _check_origin(request)
        llm = state.router.status() if state.router else {"available": False}
        return _json({
            "ok": True,
            "llm": llm,
            "workspace": state.workspace,
            "provider_catalog": _provider_catalog(),
        })

    def _provider_catalog() -> List[Dict[str, Any]]:
        """Arayüzde gösterilecek sağlayıcı kataloğu (anahtar gereksinimleri)."""
        from neurocore.agent.llm import KNOWN_SERVICES, SUGGESTED_MODELS

        catalog: List[Dict[str, Any]] = []
        for key, spec in KNOWN_SERVICES.items():
            catalog.append({
                "name": key,
                "label": spec.get("label", key),
                "base_url": spec.get("base_url", ""),
                "default_model": spec.get("model", ""),
                # Model adları sık değişir; kullanıcıya güncel öneriler sunulur
                # ki eski bir ad yüzünden 404 almasın.
                "suggested_models": SUGGESTED_MODELS.get(key, []),
                "env_var": spec.get("env", ""),
                "needs_key": bool(spec.get("env")),
                "local": "127.0.0.1" in spec.get("base_url", ""),
                "help": _provider_help(key),
            })
        return catalog

    def _provider_help(key: str) -> str:
        """Sağlayıcı için kısa yardım metni."""
        helps = {
            "deepseek": "platform.deepseek.com adresinden API anahtarı alın. "
                        "Ucuz ve araç çağrısını destekler — önerilen.",
            "gemini": "aistudio.google.com/apikey adresinden ücretsiz anahtar alın. "
                      "Google'ın OpenAI uyumlu uç noktası kullanılır; anahtar "
                      "'AIza' ile başlar ve araç çağrısı desteklenir.",
            "openai": "platform.openai.com adresinden anahtar alın.",
            "groq": "console.groq.com adresinden ücretsiz anahtar alın; çok hızlı. "
                    "Araç çağrısı için 'openai/gpt-oss-20b' önerilir "
                    "('allam-2-7b' araç çağrısını desteklemez).",
            "openrouter": "openrouter.ai adresinden anahtar alın; tek anahtarla "
                          "birçok model.",
            "together": "api.together.xyz adresinden anahtar alın.",
            "lmstudio": "LM Studio'yu açıp yerel sunucuyu başlatın (port 1234). "
                        "Anahtar gerekmez.",
            "vllm": "vLLM sunucusunu 8000 portunda başlatın. Anahtar gerekmez.",
            "llamacpp": "llama.cpp sunucusunu 8080 portunda başlatın. Anahtar gerekmez.",
        }
        return helps.get(key, "")

    @app.post("/api/settings/llm")
    async def post_settings_llm(request: Request,
                                payload: Dict[str, Any] = Body(...)) -> Any:
        """
        API anahtarını kaydeder.

        Anahtar **düz metin olarak yanıta konmaz**; yalnızca kaydedildiği
        bildirilir.
        """
        _check_origin(request)
        if state.router is None:
            raise HTTPException(status_code=503, detail="dil modeli yönlendiricisi yok")
        provider = str(payload.get("provider") or "").strip()
        api_key = str(payload.get("api_key") or "").strip()
        model = str(payload.get("model") or "").strip()
        base_url = str(payload.get("base_url") or "").strip()

        if not provider:
            raise HTTPException(status_code=400, detail="sağlayıcı gerekli")
        if len(api_key) > 400:
            raise HTTPException(status_code=400, detail="anahtar çok uzun")

        result = state.router.save_credentials(
            provider, api_key=api_key, model=model, base_url=base_url
        )
        status = result.get("status", {})
        # Anahtarın kendisini asla geri gönderme
        for item in status.get("providers", []):
            item.pop("api_key", None)
        state.events.publish("settings.llm_updated", {
            "provider": provider, "available": status.get("available", False),
        })
        return _json({"ok": result.get("ok", False),
                      "error": result.get("error", ""),
                      "llm": status})

    @app.post("/api/settings/llm/test")
    async def post_settings_llm_test(request: Request,
                                     payload: Dict[str, Any] = Body(default={})) -> Any:
        """Sağlayıcı bağlantısını gerçek bir istekle sınar."""
        _check_origin(request)
        if state.router is None:
            raise HTTPException(status_code=503, detail="dil modeli yönlendiricisi yok")
        provider = str((payload or {}).get("provider") or "").strip()
        result = await asyncio.to_thread(state.router.test, provider)
        return _json(result)

    @app.post("/api/settings/llm/clear")
    async def post_settings_llm_clear(request: Request,
                                      payload: Dict[str, Any] = Body(default={})) -> Any:
        """Kayıtlı anahtarları siler."""
        _check_origin(request)
        if state.router is None:
            raise HTTPException(status_code=503, detail="yönlendirici yok")
        provider = str((payload or {}).get("provider") or "").strip()
        result = state.router.clear_credentials(provider)
        return _json(result)

    @app.post("/api/settings/workspace")
    async def post_settings_workspace(request: Request,
                                      payload: Dict[str, Any] = Body(...)) -> Any:
        """
        **Genel** çalışma alanını değiştirir.

        Not: Sohbet bazlı çalışma alanı için `/api/chats/{id}/workspace`
        kullanılır. Bu uç nokta, sohbeti olmayan (genel) oturumun kökünü
        değiştirir.
        """
        _check_origin(request)
        path = str(payload.get("path") or "").strip()
        if not path:
            raise HTTPException(status_code=400, detail="yol gerekli")

        target = Path(os.path.expandvars(os.path.expanduser(path)))
        if not target.is_dir():
            return _json({"ok": False, "error": f"dizin yok: {target}"}, status=400)

        switched = apply_workspace(state, str(target.resolve()))
        return _json({"ok": True, "workspace": state.workspace,
                      "updated_components": switched})

    @app.post("/api/settings/mkdir")
    async def post_settings_mkdir(request: Request,
                                  payload: Dict[str, Any] = Body(...)) -> Any:
        """
        Klasör oluşturur (dizin seçici için).

        DSH'de **yeni klasör** işlemi dizin seçici dialogunda yapılır; dosya
        ağacı yalnızca listeler. Aynı model burada da uygulanır: seçilen
        klasörün **altında** yeni klasör açılır ve oluşturulan klasör seçilir.

        Args:
            payload: `parent` (üst dizin) ve `name` (klasör adı)
        """
        _check_origin(request)
        parent = str(payload.get("parent") or "").strip()
        name = str(payload.get("name") or "").strip()

        if not name:
            raise HTTPException(status_code=400, detail="klasör adı gerekli")
        # Ad güvenliği: yol ayırıcı ve üst dizin kaçışı yasak
        if any(char in name for char in ("\\", "/", ":", "*", "?", '"', "<", ">", "|")):
            return _json({"ok": False,
                          "error": "klasör adında \\ / : * ? \" < > | kullanılamaz"},
                         status=400)
        if name in (".", ".."):
            return _json({"ok": False, "error": "geçersiz klasör adı"}, status=400)

        base = Path(os.path.expandvars(os.path.expanduser(parent))) if parent \
            else Path(state.workspace)
        if not base.is_dir():
            return _json({"ok": False, "error": f"üst dizin yok: {base}"}, status=400)

        target = base / name
        if target.exists():
            return _json({"ok": False, "error": f"zaten var: {target.name}"},
                         status=409)
        try:
            target.mkdir(parents=False)
        except OSError as exc:
            return _json({"ok": False, "error": f"oluşturulamadı: {exc}"}, status=400)

        logger.info(f"Klasör oluşturuldu: {target}")
        state.events.publish("settings.directory_created", {"path": str(target)})
        return _json({"ok": True, "path": str(target), "name": target.name})

    @app.get("/api/workspace/tree")
    async def get_workspace_tree(request: Request, path: str = "",
                                 session_id: str = "") -> Any:
        """
        Çalışma alanı dosya ağacı — **tek seviye** listeler.

        DSH'nin `workspaceFiles.list()` davranışı: ağaç seviye seviye okunur
        ve **oturumun çalışma alanına** köklenir. Tüm ağacı bir seferde
        döndürmek büyük projelerde yavaş olurdu.

        Args:
            path: Mutlak dizin yolu (boş = kök)
            session_id: Oturum kimliği (kök belirlemek için)
        """
        _check_origin(request)
        if state.coding is None:
            raise HTTPException(status_code=503, detail="kodlama ajanı yok")

        # Kök: sohbetin çalışma alanı, yoksa genel çalışma alanı
        root = state.general_workspace or state.workspace
        if session_id and state.chats is not None:
            chat = state.chats.get(session_id)
            if chat is not None and chat.workspace:
                root = chat.workspace

        if path:
            # Yol güvenliği: ajanın kendi çözücüsü kullanılır (kum havuzu)
            try:
                target = state.coding.resolve(path)
            except Exception as exc:
                return _json({"ok": False, "error": f"yol reddedildi: {exc}"},
                             status=400)
        else:
            target = Path(root)
            if not target.is_dir():
                return _json({"ok": False, "error": f"kök dizin yok: {target}"},
                             status=404)

        if not target.is_dir():
            return _json({"ok": False, "error": "dizin değil"}, status=400)

        entries: List[Dict[str, Any]] = []
        truncated = False
        try:
            items = sorted(
                target.iterdir(),
                key=lambda item: (not item.is_dir(), item.name.lower()),
            )
        except OSError as exc:
            return _json({"ok": False, "error": str(exc)}, status=400)

        for item in items:
            if len(entries) >= 400:
                truncated = True
                break
            try:
                is_directory = item.is_dir()
            except OSError:
                continue
            entry: Dict[str, Any] = {
                "name": item.name,
                "path": str(item),
                "type": "directory" if is_directory else "file",
            }
            if not is_directory:
                try:
                    entry["size"] = item.stat().st_size
                except OSError:
                    entry["size"] = 0
                entry["language"] = _language_of(item.suffix)
            entries.append(entry)

        return _json({
            "ok": True, "root": str(root), "path": str(target),
            "entries": entries, "truncated": truncated,
        })

    @app.get("/api/workspace/file")
    async def get_workspace_file(request: Request, path: str = "",
                                max_bytes: int = 200_000) -> Any:
        """
        Dosya içeriğini okur (önizleme için).

        Kum havuzu kuralları geçerlidir; çalışma alanı dışındaki yollar
        reddedilir. Metin olmayan dosyalar için `binary: True` döner.
        """
        _check_origin(request)
        if state.coding is None:
            raise HTTPException(status_code=503, detail="kodlama ajanı yok")
        if not path:
            raise HTTPException(status_code=400, detail="yol gerekli")
        try:
            target = state.coding.resolve(path)
        except Exception as exc:
            return _json({"ok": False, "error": f"yol reddedildi: {exc}"},
                         status=400)
        if not target.is_file():
            return _json({"ok": False, "error": "dosya değil"}, status=400)
        try:
            raw = target.read_bytes()[:max(1, min(int(max_bytes), 2_000_000))]
        except OSError as exc:
            return _json({"ok": False, "error": str(exc)}, status=400)

        binary = b"\x00" in raw[:4096]
        return _json({
            "ok": True, "path": str(target), "name": target.name,
            "size": target.stat().st_size, "binary": binary,
            "language": _language_of(target.suffix),
            "text": "" if binary else raw.decode("utf-8", "replace"),
        })

    @app.get("/api/settings/browse")
    async def get_settings_browse(request: Request, path: str = "") -> Any:
        """
        Dizinleri listeler (çalışma alanı seçici için).

        Yalnızca **dizin adları** döner; dosya içeriği okunmaz.
        """
        _check_origin(request)
        target = Path(os.path.expandvars(os.path.expanduser(path))) if path \
            else Path(state.workspace)
        if not target.is_dir():
            target = Path(state.workspace)
        entries: List[Dict[str, Any]] = []
        try:
            for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if item.name.startswith(".") and item.name not in (".git",):
                    continue
                try:
                    is_directory = item.is_dir()
                except OSError:
                    continue
                if not is_directory:
                    continue
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "has_project": any(
                        (item / marker).exists() for marker in (
                            "project.godot", "package.json", "pyproject.toml",
                            "Cargo.toml", "go.mod", "setup.py", "*.blend",
                        )
                    ),
                })
                if len(entries) >= 200:
                    break
        except OSError as exc:
            return _json({"ok": False, "error": str(exc)}, status=400)

        parent = str(target.parent) if target.parent != target else ""
        return _json({"ok": True, "path": str(target), "parent": parent,
                      "entries": entries})

    # ─── Görsel Doğrulama ───────────────────────────────────────────────

    @app.post("/api/vision/screenshot")
    async def post_vision_screenshot(request: Request,
                                     payload: Dict[str, Any] = Body(default={})) -> Any:
        """Ekran görüntüsü alır ve analiz eder."""
        _check_origin(request)
        if state.vision is None:
            raise HTTPException(status_code=503, detail="görsel doğrulama hazır değil")
        options = payload or {}
        result = await asyncio.to_thread(
            state.vision.screenshot,
            window=options.get("window") or None,
            region=tuple(options["region"]) if options.get("region") else None,
            save=True,
            tag=str(options.get("tag") or "ui"),
            max_side=int(options.get("max_side") or 1280),
        )
        return _json(result)

    @app.get("/api/vision/windows")
    async def get_vision_windows(request: Request, query: str = "",
                                 min_size: int = 100) -> Any:
        """Açık pencereleri listeler."""
        _check_origin(request)
        if state.vision is None:
            raise HTTPException(status_code=503, detail="görsel doğrulama hazır değil")
        windows = await asyncio.to_thread(
            state.vision.list_windows, True, min_size, 120, query
        )
        return _json({"ok": True, "windows": windows, "count": len(windows)})

    @app.get("/api/vision/foreground")
    async def get_vision_foreground(request: Request) -> Any:
        """Ön plandaki pencere."""
        _check_origin(request)
        if state.vision is None:
            raise HTTPException(status_code=503, detail="görsel doğrulama hazır değil")
        window = await asyncio.to_thread(state.vision.foreground_window)
        return _json({"ok": True, "window": window})

    @app.post("/api/vision/verify")
    async def post_vision_verify(request: Request,
                                 payload: Dict[str, Any] = Body(...)) -> Any:
        """
        Bir uygulamanın açıldığını ve içerik gösterdiğini doğrular.

        Ajan "Godot açıldı mı?" sorusunu bu uçla yanıtlar: süreç başlamış olsa
        bile pencere gelmemiş veya boş olabilir.
        """
        _check_origin(request)
        if state.vision is None:
            raise HTTPException(status_code=503, detail="görsel doğrulama hazır değil")
        title = str(payload.get("title") or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="pencere başlığı gerekli")
        result = await asyncio.to_thread(
            state.vision.verify_application, title,
            expect_content=bool(payload.get("expect_content", True)),
            timeout=float(payload.get("timeout") or 15.0),
        )
        return _json(result)

    @app.get("/api/vision/shot")
    async def get_vision_shot(request: Request, path: str) -> Any:
        """Kaydedilmiş bir ekran görüntüsünü döner (arayüzde göstermek için)."""
        _check_origin(request)
        target = Path(os.path.expandvars(os.path.expanduser(path)))
        if not target.is_file() or target.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise HTTPException(status_code=404, detail="görüntü bulunamadı")
        # Yalnızca ekran görüntüsü dizininden okumaya izin ver
        if state.vision is not None and state.vision._shot_dir is not None:
            try:
                target.resolve().relative_to(Path(state.vision._shot_dir).resolve())
            except ValueError:
                raise HTTPException(status_code=403, detail="izin verilmeyen dizin")
        return FileResponse(target, media_type="image/png")

    # ─── WebSocket ──────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """
        Canlı olay akışı.

        Ajan adımları, onay istekleri ve durum değişiklikleri buradan akar.
        Bağlantı koptuğunda abonelik temizlenir; bekleyen onaylar zaman
        aşımıyla reddedilir.
        """
        await websocket.accept()
        queue_ = state.events.subscribe()
        state.events.bind_loop(asyncio.get_running_loop())

        try:
            await websocket.send_json({
                "type": "connected",
                "data": {
                    "workspace": state.workspace,
                    "autonomy": state.registry.policy.to_dict() if state.registry else {},
                    "busy": state.is_busy(),
                    "mode": state.loop.mode if state.loop else "rules",
                },
                "at": time.time(),
            })

            # Bekleyen onayları hemen gönder (sayfa yenilendiyse)
            for approval in state.approvals.pending():
                await websocket.send_json({
                    "type": "agent.approval_requested", "data": approval,
                    "at": time.time(),
                })

            async def pump() -> None:
                """Kuyruktaki olayları istemciye gönderir."""
                while True:
                    event = await queue_.get()
                    await websocket.send_json(event)

            pump_task = asyncio.create_task(pump())

            # İstemciden gelen mesajları dinle (ping/iptal)
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = str(message.get("type") or "")
                if kind == "ping":
                    await websocket.send_json({"type": "pong", "at": time.time()})
                elif kind == "cancel":
                    if state.loop is not None:
                        state.loop.cancel()
                    state.approvals.resolve_all(approved=False)
                elif kind == "approve":
                    state.approvals.resolve(
                        str(message.get("id") or ""),
                        bool(message.get("approved", False)),
                        bool(message.get("remember", False)),
                    )
        except WebSocketDisconnect:
            logger.debug("WebSocket istemcisi ayrıldı")
        except Exception as exc:
            logger.debug(f"WebSocket hatası: {exc}")
        finally:
            try:
                pump_task.cancel()
            except (NameError, UnboundLocalError):
                pass
            state.events.unsubscribe(queue_)

    # ══ ⚠ ORTAK HAVUZ: uzak düğümden iş alma ══════════════════════════
    #
    # Ölçüldü: ters tünel sunucunun `127.0.0.1`'ine bağlanır ve relay
    # isteği sunucu üzerinden gelir → kaynak IP güvenilmez. Bu yüzden
    # kimlik doğrulama **paylaşılan anahtar** ile yapılır; araç erişimi
    # `PEER_IZINLI_ARACLAR` (yalnız okuma) ile sınırlıdır.

    @app.get("/peer/info")
    async def peer_info() -> Any:
        """Komşu düğümler için kimlik/yetenek bilgisi (kimlik gerekmez)."""
        araclar = []
        kayit = getattr(state, "registry", None)
        for ad in PEER_IZINLI_ARACLAR:
            try:
                if kayit is not None and kayit.has(ad):
                    araclar.append(ad)
            except Exception:
                continue
        return {
            "ok": True,
            "surum": "neurocore",
            "arac_sayisi": len(kayit) if kayit is not None else 0,
            "izinli_araclar": araclar,
            "isinma": True,
        }

    @app.post("/peer/job")
    async def peer_job(request: Request) -> Any:
        """Uzak düğümden gelen işi çalıştırır (yalnız okuma araçları)."""
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"ok": False, "hata": "geçersiz JSON"},
                                status_code=400)
        yanit = peer_job_isle(state, payload)
        # ⚠ Yetkisiz istek 403, diğerleri 200 (çağıran ayırt edebilsin)
        if not yanit.get("ok") and "yetkisiz" in str(yanit.get("hata", "")):
            return JSONResponse(yanit, status_code=403)
        return JSONResponse(yanit)

    return app


__all__ = ["create_app"]
