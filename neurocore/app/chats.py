"""
Sohbet Deposu
==============
Her sohbeti **kendi çalışma alanıyla** saklar.

Neden gerekli? Çalışma alanı tek bir genel ayardı; bir projede çalışırken
başka bir projeye geçince önceki sohbetin bağlamı karışıyordu. Artık her
sohbet kendi klasörüne bağlıdır:

    Sohbet A → E:/projeler/site      (site üzerinde çalışır)
    Sohbet B → E:/projeler/oyun      (oyun üzerinde çalışır)
    Sohbet C → (klasör yok)          → **genel sohbet modu**

Genel sohbet modunda ajan yine çalışır ama proje bağlamı olmadan; dosya
araçları genel çalışma alanına düşer.

Depolama: `<data>/.neurocore/chats.json` — süreç yeniden başlasa da sohbetler
ve bağlı klasörleri korunur.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("app.chats")

#: Sohbet başlığının en büyük uzunluğu
TITLE_LIMIT = 64

#: Saklanan en büyük sohbet sayısı (dosya şişmesin)
MAX_CHATS = 200

#: Sohbet başına saklanan en büyük mesaj sayısı
MAX_MESSAGES = 200


def make_title(text: str, fallback: str = "Yeni sohbet") -> str:
    """
    İlk mesajdan sohbet başlığı üretir.

    Args:
        text: Kullanıcının ilk mesajı
        fallback: Metin boşsa kullanılacak başlık

    Returns:
        Tek satır, kısaltılmış başlık
    """
    clean = " ".join((text or "").split())
    if not clean:
        return fallback
    if len(clean) <= TITLE_LIMIT:
        return clean
    return clean[:TITLE_LIMIT].rsplit(" ", 1)[0] + "…"


@dataclass
class Chat:
    """Tek bir sohbet."""
    id: str
    title: str = "Yeni sohbet"
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    #: Bağlı çalışma alanı ("" = genel sohbet modu)
    workspace: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)

    @property
    def workspace_name(self) -> str:
        """Klasör adı (arayüzde gösterilir)."""
        if not self.workspace:
            return ""
        return Path(self.workspace).name or self.workspace

    def to_dict(self, *, with_messages: bool = False,
                message_limit: int = 60) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        data: Dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            "workspace": self.workspace,
            "workspace_name": self.workspace_name,
            "general": not self.workspace,
            "message_count": len(self.messages),
        }
        if with_messages:
            data["messages"] = self.messages[-message_limit:]
        return data


class ChatStore:
    """
    Sohbetleri diskte saklar ve çalışma alanı eşlemesini yönetir.

    Kullanım:
        store = ChatStore()
        store.initialize(data_path=".neurocore")
        chat = store.create(workspace="E:/projeler/site")
        store.append(chat.id, "user", "sayfa yap")
        store.set_workspace(chat.id, "E:/projeler/baska")
    """

    name = "app.chats"

    def __init__(self) -> None:
        self._chats: Dict[str, Chat] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        self._file: Optional[Path] = None
        self._initialized = False

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, data_path: str = "") -> None:
        """Depoyu başlatır ve kayıtlı sohbetleri okur."""
        if self._initialized:
            return
        base = Path(data_path) if data_path else Path(".neurocore")
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"Sohbet dizini oluşturulamadı: {exc}")
        self._file = base / "chats.json"
        self._load()
        self._initialized = True
        logger.info(
            f"Sohbet deposu hazır — {len(self._chats)} sohbet "
            f"({sum(1 for c in self._chats.values() if c.workspace)} tanesi "
            "klasöre bağlı)"
        )

    def shutdown(self) -> None:
        """Yazar ve kapatır."""
        self._save()
        self._initialized = False

    def _load(self) -> None:
        """Kayıtlı sohbetleri okur."""
        if self._file is None or not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Sohbet dosyası okunamadı: {exc}")
            return
        items = raw.get("chats") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            chat = Chat(
                id=str(item["id"]),
                title=str(item.get("title") or "Yeni sohbet"),
                created=float(item.get("created") or time.time()),
                updated=float(item.get("updated") or time.time()),
                workspace=str(item.get("workspace") or ""),
                messages=[
                    {"role": str(m.get("role", "user")),
                     "content": str(m.get("content", ""))}
                    for m in (item.get("messages") or [])
                    if isinstance(m, dict)
                ][-MAX_MESSAGES:],
            )
            self._chats[chat.id] = chat
            self._order.append(chat.id)

    def _save(self) -> None:
        """Diske yazar (atomik değiştirme)."""
        if self._file is None:
            return
        with self._lock:
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "chats": [self._chats[cid].to_dict(with_messages=True,
                                                   message_limit=MAX_MESSAGES)
                          for cid in self._order if cid in self._chats],
            }
        try:
            temporary = self._file.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            temporary.replace(self._file)
        except OSError as exc:
            logger.warning(f"Sohbetler kaydedilemedi: {exc}")

    # ─── Sorgular ───────────────────────────────────────────────────────

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Sohbetleri **son güncellenene göre** listeler."""
        with self._lock:
            chats = [self._chats[cid] for cid in self._order
                     if cid in self._chats]
        chats.sort(key=lambda item: item.updated, reverse=True)
        return [chat.to_dict() for chat in chats[:limit]]

    def get(self, chat_id: str) -> Optional[Chat]:
        """Sohbeti döndürür."""
        with self._lock:
            return self._chats.get(chat_id)

    def history(self, chat_id: str, limit: int = 40) -> List[Dict[str, str]]:
        """Sohbet geçmişi (en yeni `limit` mesaj)."""
        chat = self.get(chat_id)
        if chat is None:
            return []
        return list(chat.messages[-limit:])

    # ─── Değişiklikler ──────────────────────────────────────────────────

    def create(self, *, title: str = "", workspace: str = "") -> Chat:
        """Yeni sohbet oluşturur (en başa eklenir)."""
        chat = Chat(
            id=uuid.uuid4().hex[:12],
            title=title or "Yeni sohbet",
            workspace=workspace or "",
        )
        with self._lock:
            self._chats[chat.id] = chat
            self._order.insert(0, chat.id)
            # Sınırsız büyümeyi önle
            while len(self._order) > MAX_CHATS:
                removed = self._order.pop()
                self._chats.pop(removed, None)
        self._save()
        logger.info(
            f"Sohbet oluşturuldu: {chat.id} ({chat.title[:40]!r}, "
            f"klasör: {chat.workspace_name or 'genel'})"
        )
        return chat

    def append(self, chat_id: str, role: str, content: str) -> Optional[Chat]:
        """
        Sohbete mesaj ekler.

        İlk kullanıcı mesajı geldiğinde başlık otomatik üretilir.
        """
        chat = self.get(chat_id)
        if chat is None:
            return None
        with self._lock:
            chat.messages.append({"role": role, "content": content})
            if len(chat.messages) > MAX_MESSAGES:
                chat.messages = chat.messages[-MAX_MESSAGES:]
            if role == "user" and chat.title == "Yeni sohbet":
                chat.title = make_title(content)
            chat.updated = time.time()
            if chat_id in self._order:
                self._order.remove(chat_id)
            self._order.insert(0, chat_id)
        return chat

    def set_workspace(self, chat_id: str, workspace: str) -> Optional[Chat]:
        """Sohbetin çalışma alanını değiştirir ("" = genel mod)."""
        chat = self.get(chat_id)
        if chat is None:
            return None
        chat.workspace = workspace or ""
        chat.updated = time.time()
        self._save()
        logger.info(
            f"Sohbet {chat_id} çalışma alanı: "
            f"{chat.workspace_name or 'genel sohbet'}"
        )
        return chat

    def rename(self, chat_id: str, title: str) -> Optional[Chat]:
        """Başlığı değiştirir."""
        chat = self.get(chat_id)
        if chat is None:
            return None
        chat.title = make_title(title, fallback=chat.title)
        chat.updated = time.time()
        self._save()
        return chat

    def delete(self, chat_id: str) -> bool:
        """Sohbeti siler."""
        with self._lock:
            if chat_id not in self._chats:
                return False
            self._chats.pop(chat_id, None)
            if chat_id in self._order:
                self._order.remove(chat_id)
        self._save()
        logger.info(f"Sohbet silindi: {chat_id}")
        return True

    def clear(self, chat_id: str) -> Optional[Chat]:
        """Sohbetin mesajlarını temizler (klasör bağlı kalır)."""
        chat = self.get(chat_id)
        if chat is None:
            return None
        chat.messages = []
        chat.title = "Yeni sohbet"
        chat.updated = time.time()
        self._save()
        return chat

    # ─── Yardımcılar ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """İstatistikler."""
        with self._lock:
            chats = list(self._chats.values())
        return {
            "chats": len(chats),
            "with_workspace": sum(1 for chat in chats if chat.workspace),
            "general": sum(1 for chat in chats if not chat.workspace),
            "messages": sum(len(chat.messages) for chat in chats),
            "file": str(self._file or ""),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ChatStore chats={len(self._chats)}>"


__all__ = ["ChatStore", "Chat", "make_title", "TITLE_LIMIT", "MAX_CHATS"]
