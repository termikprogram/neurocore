"""
NeuroCore Masaüstü Uygulaması — Backend
========================================
Ajanı arayüze bağlayan HTTP + WebSocket sunucusu.

Mimari::

    Edge --app penceresi (arayüz)
            │  HTTP + WebSocket (yalnızca 127.0.0.1)
            ▼
    FastAPI (bu modül)
            │
            ├─ /api/chat        → AgentLoop (asenkron, iş parçacığında)
            ├─ /api/approve     → onay bekleyen çağrıları yanıtlar
            ├─ /api/tools       → araç listesi
            ├─ /api/autonomy    → özerklik seviyesi (salt okunur/denetimli/...)
            ├─ /api/settings    → dil modeli anahtarları
            ├─ /api/vision      → ekran görüntüsü ve analiz
            └─ WebSocket /ws    → canlı olay akışı (adımlar, onay istekleri)

Neden ayrı bir sunucu süreci değil de aynı süreç?
    Ajan durumu (araç kayıt defteri, kabuk oturumu, onay bekleyen çağrılar)
    tek bir süreçte yaşamalıdır. Ayrı süreç, durumu IPC ile paylaşmayı ve
    kabuk oturumunu yönetmeyi gereksiz yere karmaşıklaştırırdı.

Güvenlik:
    * Yalnızca `127.0.0.1` dinlenir — dışarıya açılmaz.
    * Tarayıcıdan gelen istekler `Origin` denetiminden geçer.
    * Onay bekleyen çağrılar zaman aşımına uğrar (kullanıcı yanıt vermezse
      **reddedilir**, sessizce çalıştırılmaz).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("app.server")

try:
    from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False
    FastAPI = None  # type: ignore


# ─── Oturum ve Onay Yönetimi ────────────────────────────────────────────────

@dataclass
class PendingApproval:
    """
    Kullanıcı yanıtı bekleyen bir araç çağrısı.

    Yaşam döngüsü: ajan onay ister → `PendingApproval` oluşur ve arayüze
    gönderilir → kullanıcı onaylar/reddeder → `event` set edilir → ajan devam
    eder. Kullanıcı hiç yanıt vermezse `timeout` sonrası **reddedilir**.
    """
    id: str
    tool: str
    args: Dict[str, Any]
    safety: str
    safety_label: str
    reason: str
    summary: str
    at: float = field(default_factory=time.time)
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False
    remember: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "id": self.id, "tool": self.tool, "args": self.args,
            "safety": self.safety, "safety_label": self.safety_label,
            "reason": self.reason, "summary": self.summary, "at": self.at,
        }

    def decide(self, approved: bool, remember: bool = False) -> None:
        """Kararı kaydeder ve bekleyen ajanı uyandırır."""
        self.approved = approved
        self.remember = remember
        self.event.set()


class ApprovalBroker:
    """
    Onay isteklerini arayüz ile ajan arasında taşır.

    Ajan iş parçacığında çalışır, arayüz asenkron olduğu için `threading.Event`
    ile köprü kurulur. Arayüz bağlı değilse (veya yanıt vermezse) istek
    **reddedilir** — güvenli varsayılan budur.
    """

    def __init__(self, timeout: float = 180.0) -> None:
        self._pending: Dict[str, PendingApproval] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._timeout = timeout
        self._listeners: List[Callable[[PendingApproval], None]] = []
        #: Onay çözümlendiğinde (onay/ret/zaman aşımı) çağrılır. Arayüz bunu
        #: kullanarak modalı **kapatır**; aksi hâlde zaman aşımına uğramış bir
        #: istek ekranda takılı kalır ve kullanıcı yanıtlayamaz.
        self._resolution_listeners: List[Callable[[PendingApproval, bool, str], None]] = []
        self.stats = {"requested": 0, "approved": 0, "denied": 0, "timed_out": 0}

    def add_listener(self, listener: Callable[[PendingApproval], None]) -> None:
        """Yeni onay isteği geldiğinde çağrılacak dinleyici."""
        self._listeners.append(listener)

    def add_resolution_listener(
        self, listener: Callable[[PendingApproval, bool, str], None]
    ) -> None:
        """
        Onay çözümlendiğinde çağrılacak dinleyici.

        Args:
            listener: `(approval, approved, how)` — `how` şunlardan biri:
                `"kullanıcı"`, `"zaman aşımı"`, `"iptal"`, `"kapatma"`
        """
        self._resolution_listeners.append(listener)

    def _notify_resolution(self, approval: PendingApproval, approved: bool,
                           how: str) -> None:
        """Çözümleme dinleyicilerini bilgilendirir (hata yutulur)."""
        for listener in self._resolution_listeners:
            try:
                listener(approval, approved, how)
            except Exception as exc:
                logger.debug(f"Çözümleme dinleyicisi hata verdi: {exc}")

    def request(self, tool: str, args: Dict[str, Any], safety: str,
                safety_label: str, reason: str, summary: str) -> tuple:
        """
        Onay ister ve **bloklar** (ajan iş parçacığında çağrılır).

        Returns:
            (onaylandı, hatırla)
        """
        with self._lock:
            self._counter += 1
            approval = PendingApproval(
                id=f"ap_{int(time.time() * 1000)}_{self._counter}",
                tool=tool, args=args, safety=safety, safety_label=safety_label,
                reason=reason, summary=summary,
            )
            self._pending[approval.id] = approval
        self.stats["requested"] += 1

        for listener in self._listeners:
            try:
                listener(approval)
            except Exception as exc:
                logger.debug(f"Onay dinleyicisi hata verdi: {exc}")

        logger.info(f"Onay bekleniyor: {tool} ({safety}) — {summary[:80]}")
        decided = approval.event.wait(timeout=self._timeout)

        with self._lock:
            self._pending.pop(approval.id, None)

        if not decided:
            self.stats["timed_out"] += 1
            logger.warning(
                f"Onay zaman aşımı ({self._timeout:.0f}s): {tool} — reddedildi"
            )
            self._notify_resolution(approval, False, "zaman aşımı")
            return (False, False)

        if approval.approved:
            self.stats["approved"] += 1
        else:
            self.stats["denied"] += 1
        self._notify_resolution(approval, approval.approved, "kullanıcı")
        return (approval.approved, approval.remember)

    def resolve(self, approval_id: str, approved: bool,
                remember: bool = False) -> Dict[str, Any]:
        """Arayüzden gelen kararı uygular."""
        with self._lock:
            approval = self._pending.get(approval_id)
        if approval is None:
            return {"ok": False, "error": "onay isteği bulunamadı (zaman aşımı?)"}
        approval.decide(approved, remember)
        logger.info(
            f"Onay kararı: {approval.tool} → "
            f"{'ONAY' if approved else 'RET'}{' (hatırla)' if remember else ''}"
        )
        return {"ok": True, "id": approval_id, "approved": approved,
                "remember": remember, "tool": approval.tool}

    def resolve_all(self, approved: bool, remember: bool = False,
                    how: str = "iptal") -> int:
        """
        Bekleyen tüm onayları tek seferde yanıtlar.

        Args:
            approved: Onay mı, ret mi?
            remember: Oturum boyunca hatırlansın mı?
            how: Çözümleme nedeni (`iptal`, `kapatma`, `arayüz`)
        """
        with self._lock:
            pending = list(self._pending.values())
        for approval in pending:
            approval.decide(approved, remember)
            self._notify_resolution(approval, approved, how)
        if pending:
            logger.info(f"{len(pending)} bekleyen onay kapatıldı ({how})")
        return len(pending)

    def pending(self) -> List[Dict[str, Any]]:
        """Bekleyen onay istekleri."""
        with self._lock:
            return [approval.to_dict() for approval in self._pending.values()]

    def status(self) -> Dict[str, Any]:
        """Broker durumu."""
        with self._lock:
            count = len(self._pending)
        return {**self.stats, "pending": count, "timeout_seconds": self._timeout}


# ─── Olay Yayını (WebSocket) ────────────────────────────────────────────────

class EventStream:
    """
    Ajan olaylarını bağlı arayüzlere yayar.

    Ajan iş parçacığında üretilen olaylar asenkron WebSocket'lere
    `queue.Queue` üzerinden aktarılır; böylece iki dünya birbirini bloklamaz.
    Yavaş bir istemci, ajanın çalışmasını yavaşlatmaz (kuyruk sınırlıdır ve
    dolduğunda en eski olay düşürülür).
    """

    def __init__(self, max_queue: int = 2000) -> None:
        self._subscribers: List[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._max = max_queue
        self._dropped = 0
        self._sent = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Ana asyncio döngüsünü bağlar (iş parçacığından yayın için)."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        """Yeni bir abone kuyruğu oluşturur."""
        queue_: asyncio.Queue = asyncio.Queue(maxsize=self._max)
        with self._lock:
            self._subscribers.append(queue_)
        return queue_

    def unsubscribe(self, queue_: asyncio.Queue) -> None:
        """Aboneliği kaldırır."""
        with self._lock:
            if queue_ in self._subscribers:
                self._subscribers.remove(queue_)

    def publish(self, event_type: str, data: Any) -> None:
        """
        Olayı tüm abonelere yayar (her iş parçacığından güvenli).

        Args:
            event_type: Olay türü (ör. `agent.step`)
            data: Olay verisi
        """
        payload = {"type": event_type, "data": data, "at": time.time()}
        with self._lock:
            subscribers = list(self._subscribers)
            loop = self._loop

        if not subscribers:
            return

        for queue_ in subscribers:
            try:
                queue_.put_nowait(payload)
                self._sent += 1
            except asyncio.QueueFull:
                # Yavaş istemci: en eski olayı düşür ve yenisini koy
                try:
                    queue_.get_nowait()
                    queue_.put_nowait(payload)
                except Exception:
                    pass
                self._dropped += 1

    def subscriber_count(self) -> int:
        """Bağlı istemci sayısı."""
        with self._lock:
            return len(self._subscribers)

    def status(self) -> Dict[str, Any]:
        """Akış durumu."""
        return {
            "subscribers": self.subscriber_count(),
            "sent": self._sent,
            "dropped": self._dropped,
            "bound_loop": self._loop is not None,
        }


# ─── Uygulama Durumu ────────────────────────────────────────────────────────

class AppState:
    """
    Uygulamanın tüm çalışma durumu.

    Tek bir yerde tutulur ki HTTP uç noktaları ve ajan döngüsü aynı nesnelere
    erişsin.
    """

    def __init__(self) -> None:
        self.workspace: str = str(Path.cwd())
        self.registry: Any = None
        self.router: Any = None
        self.loop: Any = None
        self.coding: Any = None
        self.shell: Any = None
        self.vision: Any = None
        self.godot: Any = None
        self.blender: Any = None
        #: Yerel model ile aday sıralama (kod seçimi)
        self.local_rank: Any = None
        #: Düşünme döngüsü (bulut üretir, yerel seçer)
        self.thinking: Any = None
        #: Ajan köprüsü (parçalı iş motoru)
        self.bridge: Any = None
        #: Bilgi iş akışı (ajan bilmediğinde öğrenir)
        self.knowledge_work: Any = None
        #: Öğrenme kaydedicisi (deneyim + kasa + eğitim köprüsü)
        self.learner: Any = None
        #: MCP yöneticisi (Blender/Godot vb. dış uygulama bağlantıları)
        self.mcp: Any = None
        #: Kendi kendini geliştirme döngüsü (öğretmen–öğrenci)
        self.study: Any = None
        #: Sohbet deposu (her sohbetin kendi çalışma alanı vardır)
        self.chats: Any = None
        #: Etkin sohbet kimliği ("" = yok)
        self.active_chat: str = ""
        #: Klasör seçilmemişken kullanılan **genel** çalışma alanı.
        #:
        #: Bir sohbetten çıkıp genel moda dönüldüğünde buraya dönülür; aksi
        #: hâlde ajan önceki projenin kökünde kalırdı.
        self.general_workspace: str = str(Path.cwd())
        self.engine: Any = None
        self.event_bus: Any = None
        self.approvals = ApprovalBroker()
        self.events = EventStream()
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.busy: bool = False
        self.busy_since: float = 0.0
        self.last_run: Optional[Dict[str, Any]] = None
        #: Sonuçları sonradan okunabilen araç çağrıları (onay sonrası tamamlanan)
        self.tool_results: Dict[str, Dict[str, Any]] = {}
        #: Sunucuyu durdurmak için geri çağrım (launcher tarafından bağlanır)
        self.shutdown_callback: Optional[Callable[[], None]] = None
        self._lock = threading.RLock()
        self.started_at = time.time()

    def store_tool_result(self, key: str, payload: Dict[str, Any],
                          limit: int = 200) -> None:
        """
        Araç sonucunu saklar (onay sonrası tamamlanan çağrılar için).

        Arayüz, bir aracı çalıştırdığında onay gerekiyorsa yanıt **henüz
        hazır değildir**. Sonuç buraya yazılır ve WebSocket olayıyla birlikte
        arayüze bildirilir; ayrıca kimliği bilen istemci sonradan okuyabilir.
        """
        with self._lock:
            self.tool_results[key] = payload
            if len(self.tool_results) > limit:
                for stale in sorted(self.tool_results,
                                    key=lambda item: self.tool_results[item].get("at", 0)
                                    )[:len(self.tool_results) - limit]:
                    self.tool_results.pop(stale, None)

    def get_tool_result(self, key: str) -> Optional[Dict[str, Any]]:
        """Saklanan araç sonucunu döndürür."""
        with self._lock:
            return self.tool_results.get(key)

    def set_busy(self, busy: bool) -> None:
        """Çalışma durumunu ayarlar."""
        with self._lock:
            self.busy = busy
            self.busy_since = time.time() if busy else 0.0

    def is_busy(self) -> bool:
        """Ajan çalışıyor mu?"""
        with self._lock:
            return self.busy

    def uptime(self) -> float:
        """Uygulama çalışma süresi (saniye)."""
        return time.time() - self.started_at


__all__ = [
    "HAS_FASTAPI",
    "ApprovalBroker",
    "PendingApproval",
    "EventStream",
    "AppState",
    "create_app",
]
