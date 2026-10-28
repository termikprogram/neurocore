"""
NeuroCore İşletim Sistemi — Denetim Kaydı (Audit Log)
=====================================================
OS katmanının yaptığı **her eylem** insan-okunur bir markdown günlüğüne
yazılır: ``<kasa veya data>/70_journal/os_audit.md``.

Özellikler:
    - Obsidian uyumlu başlık blokları (``## [zaman] EYLEM hedef``).
    - Bellekte halka tampon (``deque(maxlen=2000)``) — dosya olmasa da çalışır.
    - Dosya yazımı **asla istisna fırlatmaz**; başarısızlık loglanır.
    - `rotate()` ile günlük dosyası son N kayda indirgenir.

Kullanım:
    audit = AuditLog()
    audit.initialize(config, event_bus)
    audit.record("file.write", "notlar/x.md", ok=True, duration_ms=3.1)
    for entry in audit.tail(10):
        print(entry.to_markdown())
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("os.audit")

#: Bellekte tutulan maksimum kayıt sayısı
RING_BUFFER_SIZE = 2000

#: Markdown blok ayırıcı (``## [zaman] …``)
_BLOCK_SPLIT = re.compile(r"(?m)^(?=##\s*\[)")


# ─── Denetim Kaydı ─────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    """
    Tek bir denetim kaydı.

    Attributes:
        timestamp: Unix zaman damgası.
        action: Eylem adı (ör: ``file.write``, ``shell.run``).
        target: Eylemin hedefi (yol, komut, pid …).
        ok: Eylem başarılı mı?
        detail: Ek açıklama (tek satıra indirgenir).
        source: Kaydı üreten alt sistem.
        duration_ms: Eylem süresi (ms).
        approved: Eylem kullanıcı/onay mekanizması tarafından onaylandı mı?
    """

    timestamp: float
    action: str
    target: str
    ok: bool
    detail: str = ""
    source: str = "os"
    duration_ms: float = 0.0
    approved: bool = False

    def to_markdown(self) -> str:
        """Kaydı Obsidian'da okunabilir markdown bloğuna çevirir."""
        zaman = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        durum = "✅ başarılı" if self.ok else "❌ başarısız"
        onay = "evet" if self.approved else "hayır"

        satirlar = [f"## [{zaman}] {str(self.action).upper()} {self._tek_satir(self.target, 300)}"]
        satirlar.append(f"- **durum**: {durum}")
        satirlar.append(
            f"- **kaynak**: `{self.source}` · **süre**: {self.duration_ms:.1f} ms · "
            f"**onay**: {onay}"
        )
        if self.detail:
            satirlar.append(f"- **detay**: {self._tek_satir(self.detail, 800)}")
        return "\n".join(satirlar)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dict."""
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "target": self.target,
            "ok": self.ok,
            "detail": self.detail,
            "source": self.source,
            "duration_ms": self.duration_ms,
            "approved": self.approved,
        }

    @staticmethod
    def _tek_satir(text: str, limit: int) -> str:
        """Çok satırlı metni markdown bloğunu bozmayacak tek satıra indirir."""
        tek = " ⏎ ".join(str(text).splitlines()) if text else ""
        if len(tek) > limit:
            tek = tek[: limit - 1] + "…"
        return tek


# ─── Denetim Günlüğü ───────────────────────────────────────────────────────

class AuditLog:
    """
    Markdown tabanlı denetim günlüğü.

    Günlük dosyası yolu verilmezse `initialize()` sırasında şu sırayla
    belirlenir:
        ``<knowledge.vault_path>/70_journal/os_audit.md`` →
        ``<data_path>/70_journal/os_audit.md``

    Dosya yazımı başarısız olsa bile kayıt bellekte tutulur ve sistem çalışmaya
    devam eder.
    """

    name = "os.audit"

    def __init__(self, path: Optional[str] = None) -> None:
        """
        Args:
            path: Günlük dosyası yolu. None ise `initialize()` içinde çözülür.
        """
        self._path: Optional[str] = str(path) if path else None
        self._entries: Deque[AuditEntry] = deque(maxlen=RING_BUFFER_SIZE)
        self._initialized: bool = False
        self._config: Any = None
        self._event_bus: Any = None

        # Sayaçlar
        self._total: int = 0
        self._ok_count: int = 0
        self._fail_count: int = 0
        self._write_errors: int = 0
        self._rotations: int = 0
        self._last_error: str = ""
        self._last_record_ts: float = 0.0

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Günlüğü başlatır; varsayılan yolu çözer ve dizini oluşturur.

        İdempotenttir.
        """
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus

        if not self._path:
            self._path = self._default_path(config)

        if self._path:
            try:
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self._write_errors += 1
                self._last_error = str(exc)
                logger.warning("Denetim günlüğü dizini oluşturulamadı: %s", exc)

        self._initialized = True
        logger.info("Denetim günlüğü hazır: %s", self._path or "(yalnızca bellek)")

    def shutdown(self) -> None:
        """Günlüğü kapatır (idempotent). Bellekteki kayıtlar korunur."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info(
            "Denetim günlüğü kapatıldı — toplam kayıt: %d (%d başarısız)",
            self._total,
            self._fail_count,
        )

    def _default_path(self, config: Any) -> Optional[str]:
        """Varsayılan günlük yolunu config'ten türetir."""
        vault = None
        if config is not None:
            knowledge = getattr(config, "knowledge", None)
            vault = getattr(knowledge, "vault_path", None) if knowledge is not None else None

        base_path = getattr(config, "base_path", None) if config is not None else None
        if not base_path:
            base_path = str(Path(__file__).resolve().parents[2])

        data_path = getattr(config, "data_path", None) if config is not None else None
        if not data_path:
            data_path = os.path.join(str(base_path), "data")

        kok = vault or data_path
        return os.path.join(str(kok), "70_journal", "os_audit.md")

    @property
    def path(self) -> Optional[str]:
        """Günlük dosyası yolu (None ise yalnızca bellek)."""
        return self._path

    # ─── Kayıt ──────────────────────────────────────────────────────────

    def record(
        self,
        action: str,
        target: str,
        ok: bool = True,
        detail: str = "",
        duration_ms: float = 0.0,
        approved: bool = False,
        source: str = "os",
    ) -> AuditEntry:
        """
        Yeni bir denetim kaydı ekler.

        Args:
            action: Eylem adı (ör: ``shell.run``).
            target: Hedef (yol, komut, pid).
            ok: Başarılı mı?
            detail: Ek açıklama.
            duration_ms: Süre (ms).
            approved: Onaylandı mı?
            source: Kaynak alt sistem.

        Returns:
            Oluşturulan `AuditEntry` (dosya yazımı başarısız olsa bile döner).
        """
        entry = AuditEntry(
            timestamp=time.time(),
            action=str(action),
            target=str(target),
            ok=bool(ok),
            detail=str(detail or ""),
            source=str(source),
            duration_ms=float(duration_ms or 0.0),
            approved=bool(approved),
        )

        self._entries.append(entry)
        self._total += 1
        self._last_record_ts = entry.timestamp
        if entry.ok:
            self._ok_count += 1
        else:
            self._fail_count += 1

        self._write_entry(entry)

        if self._event_bus:
            try:
                self._event_bus.emit(
                    "os.audit_recorded",
                    {
                        "action": entry.action,
                        "target": entry.target,
                        "ok": entry.ok,
                        "duration_ms": entry.duration_ms,
                    },
                    source="os.audit",
                )
            except Exception as exc:
                logger.debug("Denetim olayı yayınlanamadı: %s", exc)

        return entry

    def _write_entry(self, entry: AuditEntry) -> bool:
        """Kaydı günlük dosyasına ekler; asla istisna fırlatmaz."""
        if not self._path:
            return False
        try:
            blok = entry.to_markdown()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(blok + "\n\n")
            return True
        except Exception as exc:
            self._write_errors += 1
            self._last_error = str(exc)
            logger.warning("Denetim kaydı yazılamadı (%s): %s", self._path, exc)
            return False

    # ─── Sorgu ──────────────────────────────────────────────────────────

    def tail(self, limit: int = 50) -> List[AuditEntry]:
        """
        Son `limit` kaydı **en yeniden en eskiye** döndürür.
        """
        if limit <= 0:
            return []
        entries = list(self._entries)
        return list(reversed(entries[-limit:]))

    def search(self, needle: str, limit: int = 50) -> List[AuditEntry]:
        """
        Metin aramasını eylem/hedef/detay alanlarında yapar.

        Returns:
            Eşleşen kayıtlar (en yeniden en eskiye).
        """
        if not needle:
            return []
        anahtar = str(needle).lower()
        bulunan: List[AuditEntry] = []
        for entry in reversed(self._entries):
            if (
                anahtar in entry.action.lower()
                or anahtar in entry.target.lower()
                or anahtar in entry.detail.lower()
            ):
                bulunan.append(entry)
                if len(bulunan) >= max(1, limit):
                    break
        return bulunan

    def entries(self) -> List[AuditEntry]:
        """Bellekteki tüm kayıtları (eskiden yeniye) döndürür."""
        return list(self._entries)

    # ─── Bakım ──────────────────────────────────────────────────────────

    def rotate(self, max_entries: int = 5000) -> int:
        """
        Günlük dosyasını son `max_entries` kayda indirger.

        Args:
            max_entries: Dosyada kalacak maksimum kayıt bloğu sayısı.

        Returns:
            Silinen kayıt sayısı (hata durumunda 0 — istisna fırlatılmaz).
        """
        if not self._path or max_entries <= 0:
            return 0

        try:
            dosya = Path(self._path)
            if not dosya.exists():
                return 0

            metin = dosya.read_text(encoding="utf-8", errors="replace")
            parcalar = _BLOCK_SPLIT.split(metin)

            baslik = ""
            bloklar: List[str] = []
            for parca in parcalar:
                if parca.startswith("## ["):
                    bloklar.append(parca)
                elif not bloklar:
                    baslik = parca

            if len(bloklar) <= max_entries:
                return 0

            kalan = bloklar[-max_entries:]
            silinen = len(bloklar) - len(kalan)
            yeni_icerik = (baslik or "") + "\n".join(kalan)

            gecici = dosya.with_suffix(dosya.suffix + ".tmp")
            gecici.write_text(yeni_icerik, encoding="utf-8")
            os.replace(str(gecici), str(dosya))

            self._rotations += 1
            logger.info("Denetim günlüğü döndürüldü: %d kayıt arşivden çıkarıldı", silinen)

            if self._event_bus:
                try:
                    self._event_bus.emit(
                        "os.audit_rotated",
                        {"removed": silinen, "kept": len(kalan), "path": self._path},
                        source="os.audit",
                    )
                except Exception as exc:
                    logger.debug("Rotasyon olayı yayınlanamadı: %s", exc)

            return silinen
        except Exception as exc:
            self._write_errors += 1
            self._last_error = str(exc)
            logger.warning("Denetim günlüğü döndürülemedi: %s", exc)
            return 0

    def clear(self) -> int:
        """Bellekteki kayıtları temizler (dosyaya dokunmaz)."""
        adet = len(self._entries)
        self._entries.clear()
        return adet

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir günlük metrikleri."""
        boyut = 0
        var = False
        if self._path:
            try:
                dosya = Path(self._path)
                var = dosya.exists()
                boyut = dosya.stat().st_size if var else 0
            except Exception:
                var = False
                boyut = 0

        return {
            "name": self.name,
            "initialized": self._initialized,
            "path": self._path,
            "file_exists": var,
            "file_size_bytes": boyut,
            "in_memory_entries": len(self._entries),
            "ring_buffer_size": RING_BUFFER_SIZE,
            "total_recorded": self._total,
            "ok_count": self._ok_count,
            "fail_count": self._fail_count,
            "write_errors": self._write_errors,
            "rotations": self._rotations,
            "last_record_ts": self._last_record_ts,
            "last_error": self._last_error,
        }


__all__ = ["AuditLog", "AuditEntry", "RING_BUFFER_SIZE"]
