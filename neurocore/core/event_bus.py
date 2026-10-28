"""
NeuroCore Olay Yönetim Sistemi (EventBus)
==========================================
Katmanlar arası asenkron iletişim için publish/subscribe pattern.
Öncelik seviyeleri, filtreleme, geçmiş tutma ve wildcard abonelik desteği.
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
import fnmatch
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union
from uuid import uuid4


# ─── Olay Öncelik Seviyeleri ────────────────────────────────────────────────

class EventPriority(IntEnum):
    """Olay öncelik seviyeleri — düşük değer = yüksek öncelik."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


# ─── Olay Veri Yapısı ──────────────────────────────────────────────────────

@dataclass
class Event:
    """
    Sistem olayı veri yapısı.
    
    Attributes:
        event_type: Olay tipi (ör: 'hardware.ram_warning', 'training.complete')
        data: Olay verisi (herhangi bir tip)
        source: Olayı yayınlayan modül/katman adı
        priority: Olay önceliği
        tags: Olay etiketleri (filtreleme için)
        event_id: Benzersiz olay kimliği (otomatik)
        timestamp: Oluşturulma zamanı (otomatik)
    """
    event_type: str
    data: Any = None
    source: str = "unknown"
    priority: EventPriority = EventPriority.NORMAL
    tags: Set[str] = field(default_factory=set)
    event_id: str = field(default_factory=lambda: str(uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    def matches_pattern(self, pattern: str) -> bool:
        """Wildcard pattern ile eşleşme kontrolü (ör: 'hardware.*')."""
        return fnmatch.fnmatch(self.event_type, pattern)


# ─── Abonelik Kaydı ────────────────────────────────────────────────────────

@dataclass
class Subscription:
    """Olay aboneliği kaydı."""
    subscription_id: str
    pattern: str
    handler: Union[Callable, Coroutine]
    is_async: bool
    priority_filter: Optional[EventPriority] = None
    tag_filter: Optional[Set[str]] = None
    source_filter: Optional[str] = None
    max_calls: Optional[int] = None  # None = sınırsız
    call_count: int = 0
    active: bool = True

    def should_handle(self, event: Event) -> bool:
        """Bu aboneliğin olayı işleyip işlemeyeceğini kontrol eder."""
        if not self.active:
            return False
        if self.max_calls is not None and self.call_count >= self.max_calls:
            self.active = False
            return False
        if not event.matches_pattern(self.pattern):
            return False
        if self.priority_filter is not None and event.priority > self.priority_filter:
            return False
        if self.tag_filter and not self.tag_filter.intersection(event.tags):
            return False
        if self.source_filter and event.source != self.source_filter:
            return False
        return True


# ─── Ana EventBus Sınıfı ───────────────────────────────────────────────────

class EventBus:
    """
    Asenkron olay yönetim sistemi.
    
    Tüm katmanlar arası iletişim bu bus üzerinden gerçekleşir.
    Publish/Subscribe pattern ile loose coupling sağlar.
    
    Kullanım:
        bus = EventBus()
        
        # Senkron handler
        bus.subscribe('hardware.*', on_hardware_event)
        
        # Asenkron handler
        bus.subscribe('training.complete', async_handler, is_async=True)
        
        # Olay yayınla
        bus.publish(Event(event_type='hardware.ram_warning', data={'usage': 85}))
        
        # Asenkron yayınla
        await bus.publish_async(Event(event_type='training.complete', data=result))
    """

    def __init__(self, history_size: int = 1000):
        """
        Args:
            history_size: Saklanacak maksimum olay geçmişi sayısı
        """
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._all_subscriptions: List[Subscription] = []
        self._history: deque = deque(maxlen=history_size)
        self._lock = threading.RLock()
        self._stats: Dict[str, int] = defaultdict(int)  # event_type -> publish count
        self._logger = logging.getLogger("neurocore.event_bus")
        self._paused = False

    # ─── Abonelik Yönetimi ──────────────────────────────────────────────

    def subscribe(
        self,
        pattern: str,
        handler: Callable,
        is_async: bool = False,
        priority_filter: Optional[EventPriority] = None,
        tag_filter: Optional[Set[str]] = None,
        source_filter: Optional[str] = None,
        max_calls: Optional[int] = None,
    ) -> str:
        """
        Bir olay pattern'ine abone ol.
        
        Args:
            pattern: Olay pattern'i (wildcard destekli: 'hardware.*', '*')
            handler: Olay handler fonksiyonu (Event -> None)
            is_async: Handler async mi?
            priority_filter: Sadece bu ve üstü öncelikli olayları al
            tag_filter: Sadece bu etiketlere sahip olayları al
            source_filter: Sadece bu kaynaktan gelen olayları al
            max_calls: Maksimum çağrı sayısı (None = sınırsız)
            
        Returns:
            Abonelik ID'si (unsubscribe için kullanılır)
        """
        sub_id = f"sub_{uuid4().hex[:12]}"
        
        subscription = Subscription(
            subscription_id=sub_id,
            pattern=pattern,
            handler=handler,
            is_async=is_async,
            priority_filter=priority_filter,
            tag_filter=tag_filter,
            source_filter=source_filter,
            max_calls=max_calls,
        )
        
        with self._lock:
            self._subscriptions[pattern].append(subscription)
            self._all_subscriptions.append(subscription)
        
        self._logger.debug(f"Abonelik eklendi: {pattern} -> {handler.__name__} [{sub_id}]")
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Aboneliği iptal et.
        
        Args:
            subscription_id: İptal edilecek abonelik ID'si
            
        Returns:
            Başarılı ise True
        """
        with self._lock:
            for pattern, subs in self._subscriptions.items():
                for sub in subs:
                    if sub.subscription_id == subscription_id:
                        sub.active = False
                        subs.remove(sub)
                        self._all_subscriptions.remove(sub)
                        self._logger.debug(f"Abonelik iptal edildi: [{subscription_id}]")
                        return True
        return False

    def unsubscribe_all(self, pattern: Optional[str] = None) -> int:
        """
        Tüm veya belirli pattern'deki abonelikleri iptal et.
        
        Returns:
            İptal edilen abonelik sayısı
        """
        with self._lock:
            if pattern:
                count = len(self._subscriptions.get(pattern, []))
                for sub in self._subscriptions.get(pattern, []):
                    sub.active = False
                    if sub in self._all_subscriptions:
                        self._all_subscriptions.remove(sub)
                self._subscriptions[pattern] = []
                return count
            else:
                count = len(self._all_subscriptions)
                for sub in self._all_subscriptions:
                    sub.active = False
                self._subscriptions.clear()
                self._all_subscriptions.clear()
                return count

    # ─── Olay Yayınlama ────────────────────────────────────────────────

    def publish(self, event: Event) -> int:
        """
        Senkron olay yayınla. Tüm uygun handler'lar çağrılır.
        
        Args:
            event: Yayınlanacak olay
            
        Returns:
            Çağrılan handler sayısı
        """
        if self._paused:
            return 0
            
        with self._lock:
            self._history.append(event)
            self._stats[event.event_type] += 1
            handlers_called = 0
            
            # Tüm abonelikleri kontrol et
            for subscription in list(self._all_subscriptions):
                if subscription.should_handle(event):
                    try:
                        if subscription.is_async:
                            # Async handler'ı uygun loop'ta çalıştır
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(subscription.handler(event))
                            except RuntimeError:
                                # Event loop yoksa yeni bir tane oluştur
                                asyncio.run(subscription.handler(event))
                        else:
                            subscription.handler(event)
                        subscription.call_count += 1
                        handlers_called += 1
                    except Exception as e:
                        self._logger.error(
                            f"Olay handler hatası: {subscription.pattern} -> "
                            f"{subscription.handler.__name__}: {e}"
                        )
            
            return handlers_called

    async def publish_async(self, event: Event) -> int:
        """
        Asenkron olay yayınla. Async handler'lar await edilir.
        
        Args:
            event: Yayınlanacak olay
            
        Returns:
            Çağrılan handler sayısı
        """
        if self._paused:
            return 0
            
        with self._lock:
            self._history.append(event)
            self._stats[event.event_type] += 1
            matching = [
                s for s in list(self._all_subscriptions)
                if s.should_handle(event)
            ]
        
        handlers_called = 0
        for subscription in matching:
            try:
                if subscription.is_async:
                    await subscription.handler(event)
                else:
                    subscription.handler(event)
                subscription.call_count += 1
                handlers_called += 1
            except Exception as e:
                self._logger.error(
                    f"Async olay handler hatası: {subscription.pattern} -> "
                    f"{subscription.handler.__name__}: {e}"
                )
        
        return handlers_called

    # ─── Hızlı Yayınlama Yardımcıları ──────────────────────────────────

    def emit(self, event_type: str, data: Any = None, source: str = "unknown",
             priority: EventPriority = EventPriority.NORMAL) -> int:
        """Kısa yoldan olay yayınla."""
        return self.publish(Event(
            event_type=event_type, data=data,
            source=source, priority=priority
        ))

    def emit_warning(self, event_type: str, data: Any = None, source: str = "unknown") -> int:
        """Uyarı öncelikli olay yayınla."""
        return self.publish(Event(
            event_type=event_type, data=data,
            source=source, priority=EventPriority.HIGH
        ))

    def emit_critical(self, event_type: str, data: Any = None, source: str = "unknown") -> int:
        """Kritik öncelikli olay yayınla."""
        return self.publish(Event(
            event_type=event_type, data=data,
            source=source, priority=EventPriority.CRITICAL
        ))

    # ─── Geçmiş ve İstatistikler ───────────────────────────────────────

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
        since: Optional[float] = None,
    ) -> List[Event]:
        """
        Olay geçmişini sorgula.
        
        Args:
            event_type: Filtrelenecek olay tipi (wildcard destekli)
            limit: Maksimum sonuç sayısı
            since: Bu timestamp'ten sonraki olaylar
            
        Returns:
            Eşleşen olayların listesi (en yeniden en eskiye)
        """
        with self._lock:
            events = list(self._history)
        
        # Filtrele
        if event_type:
            events = [e for e in events if e.matches_pattern(event_type)]
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        # En yeniler önce, limit uygula
        return list(reversed(events))[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Olay istatistiklerini döndürür."""
        with self._lock:
            return {
                "total_events": sum(self._stats.values()),
                "event_counts": dict(self._stats),
                "active_subscriptions": len([s for s in self._all_subscriptions if s.active]),
                "total_subscriptions": len(self._all_subscriptions),
                "history_size": len(self._history),
                "paused": self._paused,
            }

    # ─── Kontrol ────────────────────────────────────────────────────────

    def pause(self) -> None:
        """Olay yayınlamayı duraklat."""
        self._paused = True
        self._logger.info("EventBus duraklatıldı")

    def resume(self) -> None:
        """Olay yayınlamayı devam ettir."""
        self._paused = False
        self._logger.info("EventBus devam ediyor")

    def clear_history(self) -> None:
        """Olay geçmişini temizle."""
        with self._lock:
            self._history.clear()
            self._stats.clear()

    def reset(self) -> None:
        """EventBus'ı tamamen sıfırla."""
        with self._lock:
            self.unsubscribe_all()
            self.clear_history()
            self._paused = False
