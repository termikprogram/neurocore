"""
NeuroCore İşletim Sistemi — Dosya Sistemi Araçları
==================================================
L5 (OS) katmanının **dosya G/Ç** cephesi. Tüm yollar `Sandbox` üzerinden
doğrulanır; politika ihlali (`SandboxViolation`) asla yutulmaz, çağırana
iletilir.

Özellikler:
    - Okuma/yazma/ekleme, dizin listeleme, ad ve içerik arama
    - Taşıma/kopyalama (üzerine yazma denetimli)
    - `confirm=True` olmadan **asla silmez**; varsayılan olarak Geri Dönüşüm
      Kutusu'na gönderir (ctypes `SHFileOperationW`), kalıcı silme yalnızca
      `use_trash=False` verildiğinde yapılır
    - Atomik yazma (geçici dosya + `os.replace`)
    - `watchdog` varsa dizin izleme (yoksa uyarı verip no-op)

Kullanım:
    fs = FileSystemTools()
    fs.initialize(config, event_bus)
    fs.write_text("notlar/x.md", "# Merhaba")
    for kayit in fs.list_dir("notlar"):
        print(kayit["path"])
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from neurocore.os.sandbox import Sandbox, SandboxViolation
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("os.filesystem")


# ─── İstisna ───────────────────────────────────────────────────────────────

class FileSystemError(Exception):
    """Dosya sistemi işlemi başarısız oldu (sandbox ihlali hariç)."""


# ─── Ana Dosya Sistemi Sınıfı ──────────────────────────────────────────────

class FileSystemTools:
    """
    Sandbox denetimli dosya sistemi araçları.

    `sandbox` özniteliği dışarıdan atanabilir (``fs.sandbox = sandbox``) ya da
    `attach_sandbox()` ile bağlanabilir. Hiçbiri verilmezse modül kendi
    varsayılan `Sandbox` örneğini tembel olarak oluşturur → tek başına da
    (testlerde) çalışır.
    """

    name = "os.filesystem"

    def __init__(self) -> None:
        self.sandbox: Optional[Sandbox] = None
        self.audit: Any = None
        self._initialized: bool = False
        self._config: Any = None
        self._event_bus: Any = None
        self._counters: Dict[str, int] = defaultdict(int)
        self._last_error: str = ""
        self._watchers: Dict[str, Dict[str, Any]] = {}
        self._observer: Any = None

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Bağımlılıkları bağlar; sandbox yoksa varsayılanını kurar (idempotent)."""
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus

        if self.sandbox is None:
            kok = getattr(config, "base_path", None) if config is not None else None
            sandbox = Sandbox(str(kok) if kok else None)
            sandbox.initialize(config, event_bus)
            self.sandbox = sandbox

        self._initialized = True
        logger.info("Dosya sistemi araçları hazır — kök: %s", self.sandbox.root)

    def shutdown(self) -> None:
        """İzleyicileri durdurur ve durumu serbest bırakır (idempotent)."""
        for watch_id in list(self._watchers):
            try:
                self.unwatch(watch_id)
            except Exception as exc:
                logger.debug("İzleyici kapatılamadı (%s): %s", watch_id, exc)
        self._watchers.clear()
        self._stop_observer()
        self._initialized = False
        logger.info("Dosya sistemi araçları kapatıldı")

    def attach_sandbox(self, sandbox: Sandbox) -> Sandbox:
        """
        Paylaşılan sandbox örneğini bağlar.

        Args:
            sandbox: OSLayer'ın sahip olduğu tek sandbox örneği.

        Returns:
            Bağlanan sandbox.
        """
        self.sandbox = sandbox
        logger.debug("Dosya sistemi araçlarına sandbox bağlandı: %s", sandbox.root)
        return sandbox

    def attach_audit(self, audit: Any) -> Any:
        """Denetim günlüğünü bağlar (opsiyonel)."""
        self.audit = audit
        return audit

    def _require_sandbox(self) -> Sandbox:
        """Sandbox'ı döndürür; yoksa tembel olarak oluşturur."""
        if self.sandbox is None:
            sandbox = Sandbox()
            sandbox.initialize(self._config, self._event_bus)
            self.sandbox = sandbox
        return self.sandbox

    def _resolve(self, path: Any, must_exist: bool = False, write: bool = False) -> Path:
        """
        Yolu sandbox üzerinden doğrular.

        `SandboxViolation` bilinçli olarak yakalanmaz → çağırana iletilir.
        """
        sandbox = self._require_sandbox()
        if write:
            cozulen = sandbox.resolve_write(path, must_exist=must_exist)
        else:
            cozulen = sandbox.resolve_path(path, must_exist=must_exist)
        self._counters["path_resolved"] += 1
        return cozulen

    def _dry_run(self) -> bool:
        """Sandbox kuru çalıştırma modunda mı?"""
        try:
            return bool(self._require_sandbox().policy.dry_run)
        except Exception:
            return False

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Olay yayınlar (event_bus yoksa sessizce atlar)."""
        if not self._event_bus:
            return
        try:
            self._event_bus.emit(event, data, source=self.name)
        except Exception as exc:
            logger.debug("Olay yayınlanamadı (%s): %s", event, exc)

    def _record(self, action: str, target: str, ok: bool = True, detail: str = "") -> None:
        """Denetim günlüğüne kayıt düşer (audit yoksa no-op)."""
        if not self.audit:
            return
        try:
            self.audit.record(action, target, ok=ok, detail=detail, source=self.name)
        except Exception as exc:
            logger.debug("Denetim kaydı yazılamadı: %s", exc)

    # ─── Okuma ──────────────────────────────────────────────────────────

    def read_text(self, path: Any, max_bytes: int = 2_000_000, encoding: str = "utf-8") -> str:
        """
        Metin dosyası okur.

        Args:
            path: Dosya yolu (sandbox denetiminden geçer).
            max_bytes: Okunacak maksimum bayt sayısı (kesme sessizce yapılır).
            encoding: Kodlama; hatalı baytlar ``errors="replace"`` ile değiştirilir.

        Returns:
            Dosya içeriği.

        Raises:
            SandboxViolation: Yol politikaya uymuyorsa.
            FileSystemError: Dosya yoksa veya okunamıyorsa.
        """
        try:
            hedef = self._resolve(path, must_exist=True)
        except SandboxViolation:
            raise
        except FileNotFoundError as exc:
            self._fail(str(exc))
            raise FileSystemError(str(exc)) from exc
        except Exception as exc:
            self._fail(f"Yol çözümlenemedi: {exc}")
            raise FileSystemError(f"Yol çözümlenemedi: {exc}") from exc

        try:
            with open(hedef, "rb") as f:
                ham = f.read(int(max_bytes))
            self._counters["read_text"] += 1
            return ham.decode(encoding, errors="replace")
        except SandboxViolation:
            raise
        except Exception as exc:
            self._fail(f"Okuma hatası ({hedef}): {exc}")
            raise FileSystemError(f"Okuma hatası ({hedef}): {exc}") from exc

    def read_bytes(self, path: Any, max_bytes: Optional[int] = None) -> bytes:
        """Bayt dizisi olarak dosya okur."""
        try:
            hedef = self._resolve(path, must_exist=True)
            with open(hedef, "rb") as f:
                ham = f.read(int(max_bytes)) if max_bytes else f.read()
            self._counters["read_bytes"] += 1
            return ham
        except SandboxViolation:
            raise
        except FileNotFoundError as exc:
            self._fail(str(exc))
            raise FileSystemError(str(exc)) from exc
        except Exception as exc:
            self._fail(f"Bayt okuma hatası: {exc}")
            raise FileSystemError(f"Bayt okuma hatası: {exc}") from exc

    # ─── Yazma ──────────────────────────────────────────────────────────

    def write_text(
        self,
        path: Any,
        content: str,
        create_dirs: bool = True,
        backup: bool = False,
    ) -> str:
        """
        Metni atomik olarak yazar (geçici dosya + `os.replace`).

        Args:
            path: Hedef yol.
            content: Yazılacak metin.
            create_dirs: Üst dizinleri oluştur.
            backup: True ise mevcut dosya ``<ad>.bak`` olarak yedeklenir.

        Returns:
            Yazılan dosyanın yolu.
        """
        hedef = self._resolve(path, write=True)  # SandboxViolation yükselir

        if self._dry_run():
            logger.info("[kuru çalıştırma] Yazma atlandı: %s", hedef)
            self._record("file.write", str(hedef), ok=True, detail="kuru çalıştırma")
            return str(hedef)

        try:
            if create_dirs:
                hedef.parent.mkdir(parents=True, exist_ok=True)

            if backup and hedef.exists():
                yedek = hedef.with_name(hedef.name + ".bak")
                shutil.copy2(hedef, yedek)

            self._atomic_write_path(hedef, str(content))
            self._counters["write_text"] += 1

            bilgi = {"path": str(hedef), "bytes": len(str(content).encode("utf-8")), "backup": bool(backup)}
            self._emit("os.file_written", bilgi)
            self._record("file.write", str(hedef), ok=True, detail=f"{bilgi['bytes']} bayt")
            return str(hedef)
        except SandboxViolation:
            raise
        except Exception as exc:
            self._fail(f"Yazma hatası ({hedef}): {exc}")
            self._record("file.write", str(hedef), ok=False, detail=str(exc))
            raise FileSystemError(f"Yazma hatası ({hedef}): {exc}") from exc

    def write_bytes(self, path: Any, data: bytes, create_dirs: bool = True) -> str:
        """Bayt dizisini diske yazar."""
        hedef = self._resolve(path, write=True)

        if self._dry_run():
            logger.info("[kuru çalıştırma] Bayt yazma atlandı: %s", hedef)
            return str(hedef)

        try:
            if create_dirs:
                hedef.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write_path(hedef, bytes(data))
            self._counters["write_bytes"] += 1
            self._emit("os.file_written", {"path": str(hedef), "bytes": len(data)})
            self._record("file.write_bytes", str(hedef), ok=True, detail=f"{len(data)} bayt")
            return str(hedef)
        except SandboxViolation:
            raise
        except Exception as exc:
            self._fail(f"Bayt yazma hatası ({hedef}): {exc}")
            raise FileSystemError(f"Bayt yazma hatası ({hedef}): {exc}") from exc

    def append_text(self, path: Any, content: str) -> str:
        """Dosyanın sonuna metin ekler (dosya yoksa oluşturur)."""
        hedef = self._resolve(path, write=True)

        if self._dry_run():
            logger.info("[kuru çalıştırma] Ekleme atlandı: %s", hedef)
            return str(hedef)

        try:
            hedef.parent.mkdir(parents=True, exist_ok=True)
            with open(hedef, "a", encoding="utf-8") as f:
                f.write(str(content))
            self._counters["append_text"] += 1
            self._emit("os.file_written", {"path": str(hedef), "bytes": len(str(content).encode("utf-8")), "mode": "append"})
            self._record("file.append", str(hedef), ok=True)
            return str(hedef)
        except SandboxViolation:
            raise
        except Exception as exc:
            self._fail(f"Ekleme hatası ({hedef}): {exc}")
            raise FileSystemError(f"Ekleme hatası ({hedef}): {exc}") from exc

    def atomic_write(self, path: Any, content: str) -> str:
        """
        Geçici dosya + `os.replace` ile atomik yazma.

        Returns:
            Yazılan dosyanın yolu.
        """
        hedef = self._resolve(path, write=True)

        if self._dry_run():
            logger.info("[kuru çalıştırma] Atomik yazma atlandı: %s", hedef)
            return str(hedef)

        try:
            hedef.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write_path(hedef, str(content))
            self._counters["atomic_write"] += 1
            self._emit("os.file_written", {"path": str(hedef), "bytes": len(str(content).encode("utf-8")), "atomic": True})
            self._record("file.atomic_write", str(hedef), ok=True)
            return str(hedef)
        except SandboxViolation:
            raise
        except Exception as exc:
            self._fail(f"Atomik yazma hatası ({hedef}): {exc}")
            raise FileSystemError(f"Atomik yazma hatası ({hedef}): {exc}") from exc

    def _atomic_write_path(self, hedef: Path, icerik: Any) -> None:
        """Düşük seviyeli atomik yazma (yol doğrulaması yapılmış varsayılır)."""
        mod = "wb" if isinstance(icerik, (bytes, bytearray)) else "w"
        gecici = None
        try:
            fd, gecici = tempfile.mkstemp(prefix=".nc_tmp_", dir=str(hedef.parent))
            with os.fdopen(fd, mod, encoding=None if mod == "wb" else "utf-8") as f:
                f.write(icerik)
            os.replace(gecici, str(hedef))
            gecici = None
        finally:
            if gecici and os.path.exists(gecici):
                try:
                    os.unlink(gecici)
                except Exception:
                    pass

    # ─── Listeleme ve Arama ─────────────────────────────────────────────

    def list_dir(
        self,
        path: Any,
        recursive: bool = False,
        max_items: int = 5000,
        pattern: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dizin içeriğini listeler.

        Args:
            path: Dizin yolu.
            recursive: Alt dizinlere in.
            max_items: Maksimum kayıt sayısı.
            pattern: Dosya adı deseni (fnmatch).

        Returns:
            ``[{"path", "name", "is_dir", "size", "mtime"}, …]``
        """
        kok = self._resolve(path, must_exist=True)
        kayitlar: List[Dict[str, Any]] = []

        try:
            if recursive:
                yineleyici = os.walk(kok)
                for dizin, alt_dizinler, dosyalar in yineleyici:
                    for ad in list(alt_dizinler) + list(dosyalar):
                        if len(kayitlar) >= max_items:
                            break
                        if pattern and not fnmatch.fnmatch(ad, pattern):
                            continue
                        kayit = self._entry_info(Path(dizin) / ad)
                        if kayit:
                            kayitlar.append(kayit)
                    if len(kayitlar) >= max_items:
                        break
            else:
                for girdi in sorted(kok.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                    if len(kayitlar) >= max_items:
                        break
                    if pattern and not fnmatch.fnmatch(girdi.name, pattern):
                        continue
                    kayit = self._entry_info(girdi)
                    if kayit:
                        kayitlar.append(kayit)
        except SandboxViolation:
            raise
        except Exception as exc:
            self._fail(f"Listeleme hatası ({kok}): {exc}")
            raise FileSystemError(f"Listeleme hatası ({kok}): {exc}") from exc

        self._counters["list_dir"] += 1
        return kayitlar

    def _entry_info(self, yol: Path) -> Optional[Dict[str, Any]]:
        """Tek bir dizin girdisinin meta bilgisi (hata halinde None)."""
        try:
            stat = yol.stat()
            return {
                "path": str(yol),
                "name": yol.name,
                "is_dir": yol.is_dir(),
                "size": int(stat.st_size),
                "mtime": float(stat.st_mtime),
            }
        except Exception:
            return None

    def search(
        self,
        root: Any,
        pattern: str,
        content_regex: Optional[str] = None,
        max_results: int = 200,
        max_depth: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Ad desenine (ve isteğe bağlı içerik regex'ine) uyan dosyaları arar.

        Args:
            root: Arama kökü.
            pattern: Dosya adı deseni (ör: ``*.py``).
            content_regex: Dosya içeriğinde aranacak regex.
            max_results: Maksimum sonuç.
            max_depth: Maksimum dizin derinliği.

        Returns:
            ``[{"path", "name", "size", "mtime", "line", "match"}, …]``
        """
        kok = self._resolve(root, must_exist=True)
        derleyici = None
        if content_regex:
            try:
                derleyici = re.compile(content_regex)
            except re.error as exc:
                raise FileSystemError(f"Geçersiz içerik regex'i: {exc}") from exc

        sonuclar: List[Dict[str, Any]] = []
        kok_derinlik = len(kok.parts)

        try:
            for dizin, alt_dizinler, dosyalar in os.walk(kok):
                derinlik = len(Path(dizin).parts) - kok_derinlik
                if derinlik >= max_depth:
                    alt_dizinler[:] = []
                for ad in dosyalar:
                    if len(sonuclar) >= max_results:
                        break
                    if not fnmatch.fnmatch(ad, pattern):
                        continue
                    tam = Path(dizin) / ad
                    kayit = self._entry_info(tam)
                    if not kayit:
                        continue

                    if derleyici is not None:
                        eslesme = self._grep_file(tam, derleyici)
                        if eslesme is None:
                            continue
                        kayit["line"] = eslesme[0]
                        kayit["match"] = eslesme[1]

                    sonuclar.append(kayit)
                if len(sonuclar) >= max_results:
                    break
        except SandboxViolation:
            raise
        except Exception as exc:
            self._fail(f"Arama hatası ({kok}): {exc}")
            raise FileSystemError(f"Arama hatası ({kok}): {exc}") from exc

        self._counters["search"] += 1
        return sonuclar

    def _grep_file(self, yol: Path, derleyici: "re.Pattern[str]", maks_bayt: int = 2_000_000):
        """Dosyada regex arar; bulursa ``(satır_no, satır_metni)`` döndürür."""
        try:
            if yol.stat().st_size > maks_bayt * 4:
                return None
            with open(yol, "r", encoding="utf-8", errors="replace") as f:
                for numara, satir in enumerate(f, 1):
                    if derleyici.search(satir):
                        return numara, satir.strip()[:300]
        except Exception:
            return None
        return None

    def find_by_name(self, root: Any, pattern: str, max_results: int = 200) -> List[str]:
        """
        Ad desenine uyan dosya yollarını döndürür.

        Returns:
            Eşleşen yolların listesi (en fazla `max_results`).
        """
        return [kayit["path"] for kayit in self.search(root, pattern, max_results=max_results)]

    # ─── Taşıma / Kopyalama / Silme ─────────────────────────────────────

    def move(self, src: Any, dst: Any, overwrite: bool = False) -> str:
        """Dosya/dizin taşır. Hedef varsa ve `overwrite=False` ise hata verir."""
        kaynak = self._resolve(src, must_exist=True)
        hedef = self._resolve(dst, write=True)

        if str(kaynak) == str(hedef):
            return str(hedef)

        if hedef.exists() and not overwrite:
            raise FileSystemError(f"Hedef zaten var (overwrite=False): {hedef}")

        if self._dry_run():
            logger.info("[kuru çalıştırma] Taşıma atlandı: %s -> %s", kaynak, hedef)
            return str(hedef)

        try:
            hedef.parent.mkdir(parents=True, exist_ok=True)
            if hedef.is_dir() and not overwrite:
                raise FileSystemError(f"Hedef dizin dolu: {hedef}")
            if hedef.exists() and overwrite:
                if hedef.is_dir():
                    shutil.rmtree(hedef)
                else:
                    hedef.unlink()
            shutil.move(str(kaynak), str(hedef))
            self._counters["move"] += 1
            self._emit("os.file_moved", {"src": str(kaynak), "dst": str(hedef)})
            self._record("file.move", f"{kaynak} -> {hedef}", ok=True)
            return str(hedef)
        except SandboxViolation:
            raise
        except FileSystemError:
            raise
        except Exception as exc:
            self._fail(f"Taşıma hatası ({kaynak} -> {hedef}): {exc}")
            self._record("file.move", f"{kaynak} -> {hedef}", ok=False, detail=str(exc))
            raise FileSystemError(f"Taşıma hatası: {exc}") from exc

    def copy(self, src: Any, dst: Any, overwrite: bool = False) -> str:
        """Dosya/dizin kopyalar."""
        kaynak = self._resolve(src, must_exist=True)
        hedef = self._resolve(dst, write=True)

        if hedef.exists() and not overwrite:
            raise FileSystemError(f"Hedef zaten var (overwrite=False): {hedef}")

        if self._dry_run():
            logger.info("[kuru çalıştırma] Kopyalama atlandı: %s -> %s", kaynak, hedef)
            return str(hedef)

        try:
            hedef.parent.mkdir(parents=True, exist_ok=True)
            if kaynak.is_dir():
                if hedef.exists() and overwrite:
                    shutil.rmtree(hedef)
                shutil.copytree(str(kaynak), str(hedef), dirs_exist_ok=bool(overwrite))
            else:
                shutil.copy2(str(kaynak), str(hedef))
            self._counters["copy"] += 1
            self._emit("os.file_copied", {"src": str(kaynak), "dst": str(hedef)})
            self._record("file.copy", f"{kaynak} -> {hedef}", ok=True)
            return str(hedef)
        except SandboxViolation:
            raise
        except FileSystemError:
            raise
        except Exception as exc:
            self._fail(f"Kopyalama hatası ({kaynak} -> {hedef}): {exc}")
            raise FileSystemError(f"Kopyalama hatası: {exc}") from exc

    def delete(self, path: Any, confirm: bool = False, use_trash: bool = True) -> Dict[str, Any]:
        """
        Dosya/dizin siler.

        Güvenlik kuralları:
            - `confirm=True` verilmedikçe **hiçbir şey silinmez**.
            - `use_trash=True` (varsayılan) → Geri Dönüşüm Kutusu'na gönderir;
              bu başarısız olursa **kalıcı silmeye düşmez**, hata döndürür.
            - Kalıcı silme yalnızca `use_trash=False` ile açıkça istenir.
            - Sandbox `dry_run` modundaysa hiçbir şey silinmez.

        Returns:
            ``{"ok", "path", "deleted", "method", "reason", "dry_run"}``
        """
        sonuc: Dict[str, Any] = {
            "ok": False,
            "path": str(path),
            "deleted": False,
            "method": "",
            "reason": "",
            "dry_run": False,
        }

        if not confirm:
            sonuc["reason"] = "Silme reddedildi: confirm=True gerekli"
            logger.warning("Silme reddedildi (confirm yok): %s", path)
            self._record("file.delete", str(path), ok=False, detail="confirm yok")
            return sonuc

        hedef = self._resolve(path, write=True)  # SandboxViolation yükselir
        sonuc["path"] = str(hedef)

        if not hedef.exists():
            sonuc["reason"] = f"Yol bulunamadı: {hedef}"
            return sonuc

        if self._dry_run():
            sonuc.update({"ok": True, "dry_run": True, "reason": "kuru çalıştırma — silinmedi"})
            logger.info("[kuru çalıştırma] Silme atlandı: %s", hedef)
            return sonuc

        try:
            if use_trash:
                basarili = self._send_to_trash(hedef)
                if not basarili:
                    sonuc["reason"] = (
                        "Geri Dönüşüm Kutusu'na gönderilemedi (kalıcı silme yapılmadı). "
                        "Kalıcı silmek için use_trash=False kullanın."
                    )
                    logger.warning("Geri dönüşüm kutusu başarısız: %s", hedef)
                    self._record("file.delete", str(hedef), ok=False, detail="geri dönüşüm kutusu hatası")
                    return sonuc
                sonuc["method"] = "recycle_bin"
            else:
                if hedef.is_dir():
                    shutil.rmtree(hedef)
                else:
                    hedef.unlink()
                sonuc["method"] = "permanent"

            sonuc["ok"] = True
            sonuc["deleted"] = True
            self._counters["delete"] += 1
            self._emit("os.file_deleted", {"path": str(hedef), "method": sonuc["method"]})
            self._record("file.delete", str(hedef), ok=True, detail=sonuc["method"])
            return sonuc
        except SandboxViolation:
            raise
        except Exception as exc:
            sonuc["reason"] = str(exc)
            self._fail(f"Silme hatası ({hedef}): {exc}")
            self._record("file.delete", str(hedef), ok=False, detail=str(exc))
            return sonuc

    def _send_to_trash(self, hedef: Path) -> bool:
        """
        Windows'ta dosyayı Geri Dönüşüm Kutusu'na gönderir.

        `SHFileOperationW` + ``FOF_ALLOWUNDO``. Windows dışında veya hata
        durumunda False döner (kalıcı silmeye **düşmez**).
        """
        if sys.platform != "win32":
            logger.warning("Geri dönüşüm kutusu yalnızca Windows'ta desteklenir")
            return False

        try:
            import ctypes
            from ctypes import wintypes

            class _SHFILEOPSTRUCTW(ctypes.Structure):
                _fields_ = (
                    ("hwnd", wintypes.HWND),
                    ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR),
                    ("fFlags", ctypes.c_uint16),
                    ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p),
                    ("lpszProgressTitle", wintypes.LPCWSTR),
                )

            FO_DELETE = 3
            FOF_SILENT = 0x0004
            FOF_NOCONFIRMATION = 0x0010
            FOF_ALLOWUNDO = 0x0040
            FOF_NOERRORUI = 0x0400

            islem = _SHFILEOPSTRUCTW()
            islem.hwnd = None
            islem.wFunc = FO_DELETE
            islem.pFrom = str(hedef) + "\0\0"  # çift null ile sonlandırılmalı
            islem.pTo = None
            islem.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI

            sonuc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(islem))
            return sonuc == 0 and not islem.fAnyOperationsAborted
        except Exception as exc:
            logger.warning("SHFileOperationW başarısız (%s): %s", hedef, exc)
            return False

    # ─── Yardımcılar ────────────────────────────────────────────────────

    def hash_file(self, path: Any, algorithm: str = "sha256") -> str:
        """
        Dosyanın karma (hash) değerini hesaplar.

        Args:
            path: Dosya yolu.
            algorithm: ``hashlib`` algoritma adı (sha256, md5, blake2b, …).

        Returns:
            Onaltılık karma değeri.
        """
        hedef = self._resolve(path, must_exist=True)
        try:
            ozet = hashlib.new(algorithm)
        except Exception as exc:
            raise FileSystemError(f"Geçersiz karma algoritması: {algorithm}") from exc

        try:
            with open(hedef, "rb") as f:
                for parca in iter(lambda: f.read(1024 * 1024), b""):
                    ozet.update(parca)
            self._counters["hash_file"] += 1
            return ozet.hexdigest()
        except SandboxViolation:
            raise
        except Exception as exc:
            self._fail(f"Karma hesaplanamadı ({hedef}): {exc}")
            raise FileSystemError(f"Karma hesaplanamadı: {exc}") from exc

    def disk_usage(self, path: Any) -> Dict[str, Any]:
        """Disk kullanım bilgisi (bayt ve GB)."""
        hedef = self._resolve(path)
        hedef_str = str(hedef) if hedef.exists() else str(self._require_sandbox().root)
        try:
            kullanim = shutil.disk_usage(hedef_str)
            gb = 1024 ** 3
            self._counters["disk_usage"] += 1
            return {
                "path": hedef_str,
                "total_bytes": int(kullanim.total),
                "used_bytes": int(kullanim.used),
                "free_bytes": int(kullanim.free),
                "total_gb": round(kullanim.total / gb, 2),
                "used_gb": round(kullanim.used / gb, 2),
                "free_gb": round(kullanim.free / gb, 2),
                "percent": round(float(kullanim.used) / float(kullanim.total or 1) * 100, 1),
            }
        except SandboxViolation:
            raise
        except Exception as exc:
            self._fail(f"Disk kullanımı alınamadı ({hedef_str}): {exc}")
            raise FileSystemError(f"Disk kullanımı alınamadı: {exc}") from exc

    def ensure_dir(self, path: Any) -> str:
        """Dizini (ve üst dizinlerini) oluşturur; yolu döndürür."""
        hedef = self._resolve(path, write=True)
        if self._dry_run():
            logger.info("[kuru çalıştırma] Dizin oluşturma atlandı: %s", hedef)
            return str(hedef)
        try:
            hedef.mkdir(parents=True, exist_ok=True)
            self._counters["ensure_dir"] += 1
            return str(hedef)
        except SandboxViolation:
            raise
        except Exception as exc:
            self._fail(f"Dizin oluşturulamadı ({hedef}): {exc}")
            raise FileSystemError(f"Dizin oluşturulamadı: {exc}") from exc

    # ─── İzleme (watchdog) ──────────────────────────────────────────────

    def watch(self, path: Any, callback: Callable[[Dict[str, Any]], Any], recursive: bool = True) -> str:
        """
        Dizini `watchdog` ile izlemeye başlar.

        Callback'e şu biçimde bir sözlük iletilir::

            {"event": "created|modified|deleted|moved", "path": ...,
             "is_dir": bool, "dest": str}

        Args:
            path: İzlenecek dizin.
            callback: Olay alıcısı (istisnaları yutulur, sistemi düşürmez).
            recursive: Alt dizinleri de izle.

        Returns:
            İzleme kimliği; `watchdog` yoksa boş string (uyarı loglanır).
        """
        hedef = self._resolve(path, must_exist=True)

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except Exception as exc:
            logger.warning("watchdog bulunamadı — dosya izleme devre dışı: %s", exc)
            return ""

        tools = self

        class _Handler(FileSystemEventHandler):
            """watchdog olaylarını NeuroCore sözlüğüne çevirir."""

            def _bildir(self, tur: str, event: Any) -> None:
                veri = {
                    "event": tur,
                    "path": str(getattr(event, "src_path", "")),
                    "is_dir": bool(getattr(event, "is_directory", False)),
                    "dest": str(getattr(event, "dest_path", "") or ""),
                }
                try:
                    callback(veri)
                except Exception as hata:
                    logger.warning("İzleme callback hatası: %s", hata)

            def on_created(self, event):  # noqa: D102 - watchdog arayüzü
                self._bildir("created", event)

            def on_modified(self, event):  # noqa: D102
                self._bildir("modified", event)

            def on_deleted(self, event):  # noqa: D102
                self._bildir("deleted", event)

            def on_moved(self, event):  # noqa: D102
                self._bildir("moved", event)

        try:
            if self._observer is None:
                self._observer = Observer()
                self._observer.daemon = True
                self._observer.start()

            izleme = self._observer.schedule(_Handler(), str(hedef), recursive=bool(recursive))
            watch_id = f"watch_{uuid.uuid4().hex[:8]}"
            self._watchers[watch_id] = {"watch": izleme, "path": str(hedef), "recursive": bool(recursive)}
            self._counters["watch"] += 1
            logger.info("Dizin izleniyor: %s (id=%s)", hedef, watch_id)

            if self._event_bus:
                try:
                    self._event_bus.emit(
                        "os.watch_started",
                        {"watch_id": watch_id, "path": str(hedef), "recursive": bool(recursive)},
                        source=self.name,
                    )
                except Exception:
                    pass

            return watch_id
        except SandboxViolation:
            raise
        except Exception as exc:
            logger.warning("Dizin izlenemedi (%s): %s", hedef, exc)
            return ""

    def unwatch(self, watch_id: str) -> bool:
        """
        İzlemeyi durdurur.

        Returns:
            Başarılı ise True.
        """
        kayit = self._watchers.pop(watch_id, None)
        if not kayit:
            return False
        try:
            if self._observer is not None:
                self._observer.unschedule(kayit["watch"])
        except Exception as exc:
            logger.debug("İzleme kaldırılamadı (%s): %s", watch_id, exc)

        if not self._watchers:
            self._stop_observer()

        if self._event_bus:
            try:
                self._event_bus.emit("os.watch_stopped", {"watch_id": watch_id}, source=self.name)
            except Exception:
                pass
        return True

    def _stop_observer(self) -> None:
        """watchdog Observer'ı durdurur."""
        if self._observer is None:
            return
        try:
            self._observer.stop()
            self._observer.join(timeout=2.0)
        except Exception as exc:
            logger.debug("Observer durdurulamadı: %s", exc)
        self._observer = None

    # ─── Hata ve İstatistik ─────────────────────────────────────────────

    def _fail(self, mesaj: str) -> None:
        """Hata sayacını artırır ve loglar."""
        self._counters["errors"] += 1
        self._last_error = mesaj
        logger.warning(mesaj)

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dosya sistemi metrikleri."""
        sandbox_stats: Dict[str, Any] = {}
        try:
            sandbox_stats = self._require_sandbox().get_stats()
        except Exception:
            sandbox_stats = {}

        return {
            "name": self.name,
            "initialized": self._initialized,
            "sandbox_root": str(self.sandbox.root) if self.sandbox else "",
            "sandbox_initialized": bool(getattr(self.sandbox, "_initialized", False)),
            "dry_run": self._dry_run(),
            "operations": dict(self._counters),
            "watchers": len(self._watchers),
            "watchdog_active": self._observer is not None,
            "last_error": self._last_error,
            "sandbox_violations": sandbox_stats.get("violations", 0),
            "total_bytes_written_ops": self._counters.get("write_text", 0)
            + self._counters.get("write_bytes", 0)
            + self._counters.get("append_text", 0),
            "timestamp": time.time(),
        }


__all__ = ["FileSystemTools", "FileSystemError"]
