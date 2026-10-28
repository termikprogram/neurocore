"""
NeuroCore İşletim Sistemi — Sandbox (Güvenlik Kafesi)
=====================================================
L5 (OS) katmanının **güvenlik çekirdeği**. Dosya yollarını izinli köklere
hapsetmek, yasaklı desenleri engellemek ve yıkıcı komutları tespit etmek
için kullanılır.

Tasarım kuralları:
    - **Bağımlılıksız**: Yalnızca standart kütüphane kullanılır; ``psutil``,
      ``winreg`` gibi ağır/opsiyonel modüller bu modüle hiç girmez.
    - **Karar verir, çalıştırmaz**: Sandbox hiçbir G/Ç yapmaz; yalnızca
      "izinli mi?" sorusunu yanıtlar. Gerçek eylem `filesystem/shell/process`
      modüllerinde gerçekleşir.
    - **İki arayüz**: `resolve_path` ihlalde `SandboxViolation` fırlatır,
      `check_path` / `check_command` ise fırlatmadan ``(ok, sebep)`` döner.

Kullanım:
    sandbox = Sandbox()
    sandbox.initialize(config, event_bus)

    yol = sandbox.resolve_path("notlar/x.md")          # -> Path
    ok, sebep = sandbox.check_path("C:/Windows/x.dll")  # -> (False, "...")
    ok, sebep = sandbox.is_dangerous("format C:")       # -> (True, "...")
"""

from __future__ import annotations

import fnmatch
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("os.sandbox")


# ─── Varsayılan Politikalar ────────────────────────────────────────────────

DEFAULT_DENIED_PATTERNS: List[str] = [
    "*.sys",
    "*.dll",
    r"C:\Windows\**",
    "**/System32/**",
    "**/.ssh/**",
    "**/id_rsa*",
    "**/.env",
    "**/ntuser.dat*",
    "**/SAM",
    "**/SECURITY",
]

#: Yıkıcı komut imzaları: (regex, Türkçe gerekçe)
_DANGEROUS_RULES: List[Tuple[str, str]] = [
    (r"(^|[;&|(]\s*)format(\.com)?(\s|$)", "Disk biçimlendirme komutu (format)"),
    (r"\brd(\.exe)?\b[^&|]*\s/s\b", "Özyinelemeli dizin silme (rd /s)"),
    (r"\brmdir(\.exe)?\b[^&|]*\s/s\b", "Özyinelemeli dizin silme (rmdir /s)"),
    (
        r"\bdel(\.exe)?\b[^&|]*(/s\b|/f\b[^&|]*/q\b|/q\b[^&|]*/f\b)",
        "Toplu ve geri alınamaz silme (del /f /s /q)",
    ),
    (
        r"\b(remove-item|ri|rm)\b[^&|]*-(recurse|r)\b",
        "Özyinelemeli silme (Remove-Item -Recurse)",
    ),
    (r"\bshutdown(\.exe)?\b", "Sistem kapatma (shutdown)"),
    (r"\bstop-computer\b", "Sistem kapatma (Stop-Computer)"),
    (r"\brestart-computer\b", "Sistem yeniden başlatma (Restart-Computer)"),
    (r"\bremove-computer\b", "Makineyi etki alanından çıkarma (Remove-Computer)"),
    (r"\bdiskpart(\.exe)?\b", "Disk bölümleme aracı (diskpart)"),
    (r"\bbcdedit(\.exe)?\b", "Önyükleme yapılandırması değişikliği (bcdedit)"),
    (r"\bcipher(\.exe)?\b[^&|]*\s/w\b", "Boş alan üzerine yazma (cipher /w)"),
    (r"\bvssadmin(\.exe)?\b[^&|]*\bdelete\b", "Gölge kopya silme (vssadmin delete)"),
    (
        r"\breg(\.exe)?\b\s+delete\b[^&|]*\b(hklm|hkey_local_machine)\b",
        "Kayıt defteri silme (reg delete HKLM)",
    ),
    (r"\btakeown(\.exe)?\b", "Dosya sahipliğini ele geçirme (takeown)"),
    (r"\bicacls(\.exe)?\b[^&|]*\s/grant\b", "Erişim izni değiştirme (icacls /grant)"),
    (r"\bmkfs(\.\w+)?\b", "Dosya sistemi oluşturma (mkfs)"),
    (r"\bdd\b[^&|]*\bif=[^&|]*\bof=/dev/", "Ham diske yazma (dd of=/dev/...)"),
    (r">\s*/dev/(sd|hd|nvme|disk)", "Ham diske yönlendirme (> /dev/sd...)"),
    (r"\b(halt|poweroff)\b", "Sistemi durdurma (halt/poweroff)"),
    (
        r"\bnet\s+user\b[^&|]*/delete\b",
        "Kullanıcı hesabı silme (net user /delete)",
    ),
    (r"\bchmod\b[^&|]*\s-R\s+777\s+/\s*$", "Kök dizinde izinleri açma (chmod -R 777 /)"),
]

#: Fork bomb imzaları (boşluklar temizlenmiş komut üzerinde aranır)
_FORK_BOMB_SIGNATURES: Tuple[str, ...] = (
    ":(){:|:&};:",
    ":(){:|:&};",
    ":(){ :|:& };:".replace(" ", ""),
)

_DANGEROUS_COMPILED: List[Tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in _DANGEROUS_RULES
]


# ─── İstisna ───────────────────────────────────────────────────────────────

class SandboxViolation(Exception):
    """Sandbox politikası ihlali — yol veya komut izinli değil."""


# ─── Politika Veri Yapısı ──────────────────────────────────────────────────

@dataclass
class SandboxPolicy:
    """
    Sandbox politikası.

    Attributes:
        allowed_paths: İzinli kök dizinler. Boşsa yalnızca çalışma alanı kökü
            ve geçici dizin izinlidir.
        denied_patterns: Yasaklı yol desenleri (fnmatch/glob).
        allow_shell: Kabuk komutu çalıştırma izni.
        allow_write: Dosya yazma/silme/taşıma izni.
        allow_process_kill: Süreç sonlandırma izni.
        allow_registry_write: Kayıt defterine yazma izni (Windows).
        max_command_seconds: Tek komut için üst süre sınırı (saniye).
        dry_run: True ise eylemler gerçekten çalıştırılmaz, yalnızca raporlanır.
    """

    allowed_paths: List[str] = field(default_factory=list)
    denied_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_DENIED_PATTERNS))
    allow_shell: bool = True
    allow_write: bool = True
    allow_process_kill: bool = False
    allow_registry_write: bool = False
    max_command_seconds: float = 300.0
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Politikayı JSON-serileştirilebilir dict'e çevirir."""
        return asdict(self)

    def copy(self, **changes: Any) -> "SandboxPolicy":
        """Politikanın değiştirilmiş bir kopyasını üretir."""
        return replace(self, **changes)


# ─── Ana Sandbox Sınıfı ────────────────────────────────────────────────────

class Sandbox:
    """
    Yol ve komut güvenlik denetleyicisi.

    Kurallar:
        - Bir yol, izinli köklerden birinin altında değilse reddedilir.
        - Bir yol, yasaklı desenlerden biriyle eşleşirse reddedilir.
        - Yıkıcı komutlar (`format`, `rd /s`, `Remove-Item -Recurse`, …)
          `is_dangerous` tarafından işaretlenir.

    Kullanım:
        sandbox = Sandbox()
        sandbox.initialize()
        sandbox.configure(SandboxPolicy(allow_shell=False, dry_run=True))
    """

    name = "os.sandbox"

    def __init__(self, root: Optional[str] = None) -> None:
        """
        Args:
            root: Çalışma alanı kökü. None ise NeuroCore proje kökü kullanılır.
        """
        self._project_root: Path = Path(
            root if root else Path(__file__).resolve().parents[2]
        )
        self._root: Path = self._project_root
        self._temp_root: Path = Path(tempfile.gettempdir())
        self._policy: SandboxPolicy = SandboxPolicy()
        self._allowed_roots: List[Path] = []
        self._initialized: bool = False
        self._config: Any = None
        self._event_bus: Any = None

        # Sayaçlar
        self._checks: int = 0
        self._resolved: int = 0
        self._violations: int = 0
        self._commands_checked: int = 0
        self._commands_blocked: int = 0
        self._last_violation: str = ""

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Sandbox'ı başlatır; politika köklerini çözer.

        İdempotenttir: ikinci çağrı no-op'tur.
        """
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus

        base_path = getattr(config, "base_path", None) if config is not None else None
        if base_path:
            self._root = Path(str(base_path))

        self.configure(self._default_policy())
        self._initialized = True

        logger.info(
            "Sandbox hazır — kök: %s | izinli kök sayısı: %d | yasaklı desen: %d",
            self._root,
            len(self._allowed_roots),
            len(self._policy.denied_patterns),
        )

    def shutdown(self) -> None:
        """Sandbox'ı kapatır (idempotent). Politika korunur, sayaçlar sıfırlanmaz."""
        if not self._initialized:
            return
        self._initialized = False
        self._allowed_roots = []
        logger.info("Sandbox kapatıldı (toplam ihlal: %d)", self._violations)

    # ─── Politika ───────────────────────────────────────────────────────

    def _default_policy(self) -> SandboxPolicy:
        """Varsayılan politikayı üretir: proje kökü + geçici dizin."""
        return SandboxPolicy(
            allowed_paths=[str(self._root), str(self._temp_root)],
            denied_patterns=list(DEFAULT_DENIED_PATTERNS),
        )

    def configure(self, policy: SandboxPolicy) -> None:
        """
        Politikayı uygular ve izinli kökleri normalize eder.

        Args:
            policy: Uygulanacak politika. `allowed_paths` boşsa proje kökü +
                geçici dizin kullanılır.
        """
        if policy is None:
            policy = self._default_policy()

        if not policy.allowed_paths:
            policy = policy.copy(allowed_paths=[str(self._root), str(self._temp_root)])

        self._policy = policy
        self._allowed_roots = []
        for raw in policy.allowed_paths:
            try:
                self._allowed_roots.append(self._normalize_root(raw))
            except Exception as exc:  # bozuk kök kaydı sistemi düşürmesin
                logger.warning("İzinli kök çözümlenemedi: %s (%s)", raw, exc)

        if not self._allowed_roots:
            self._allowed_roots = [self._project_root]

        if self._event_bus:
            try:
                self._event_bus.emit(
                    "os.sandbox_configured",
                    {
                        "allowed_paths": [str(p) for p in self._allowed_roots],
                        "denied_patterns": len(policy.denied_patterns),
                        "allow_shell": policy.allow_shell,
                        "allow_write": policy.allow_write,
                        "dry_run": policy.dry_run,
                    },
                    source="os.sandbox",
                )
            except Exception as exc:
                logger.debug("Sandbox olayı yayınlanamadı: %s", exc)

    def _normalize_root(self, raw: str) -> Path:
        """Bir kök yolu mutlak + normalize biçime çevirir."""
        text = os.path.expandvars(os.path.expanduser(str(raw)))
        path = Path(text)
        if not path.is_absolute():
            path = self._project_root / path
        return Path(os.path.normpath(str(path)))

    @property
    def policy(self) -> SandboxPolicy:
        """Aktif politika."""
        return self._policy

    @property
    def root(self) -> Path:
        """Çalışma alanı kökü."""
        return self._root

    @property
    def allowed_roots(self) -> List[Path]:
        """Normalize edilmiş izinli kökler."""
        return list(self._allowed_roots)

    def __repr__(self) -> str:  # pragma: no cover - tanı amaçlı
        return (
            f"<Sandbox root={self._root} izinli={len(self._allowed_roots)} "
            f"ihlal={self._violations}>"
        )

    # ─── Yol Denetimi ───────────────────────────────────────────────────

    def resolve_path(self, path: str, must_exist: bool = False) -> Path:
        """
        Yolu normalize eder, doğrular ve mutlak `Path` döndürür.

        ``~`` ve ``%VAR%``/``$VAR`` genişletilir; göreli yollar çalışma alanı
        köküne göre çözülür.

        Args:
            path: Çözümlenecek yol.
            must_exist: True ise dosya/dizin yoksa `FileNotFoundError` fırlatılır.

        Returns:
            Mutlak, normalize edilmiş `Path`.

        Raises:
            SandboxViolation: Yol izinli köklerin dışındaysa veya yasaklı
                bir desenle eşleşiyorsa (ya da yol bozuksa).
            FileNotFoundError: `must_exist=True` ve yol mevcut değilse.
        """
        if path is None or str(path).strip() == "":
            raise self._violation("Boş yol verildi", "")

        try:
            text = str(path).strip()
            expanded = os.path.expandvars(os.path.expanduser(text))
            if expanded.startswith("~"):
                expanded = os.path.expanduser(expanded)

            candidate = Path(expanded)
            if not candidate.is_absolute():
                candidate = self._root / candidate
            resolved = Path(os.path.normpath(str(candidate)))
        except SandboxViolation:
            raise
        except Exception as exc:
            raise self._violation(f"Yol çözümlenemedi: {path} ({exc})", str(path)) from exc

        self._checks += 1

        hit = self._match_denied(resolved)
        if hit:
            raise self._violation(f"Yasaklı yol deseniyle eşleşti ({hit}): {resolved}", str(resolved))

        if not self._is_allowed(resolved):
            izinli = ", ".join(str(p) for p in self._allowed_roots)
            raise self._violation(
                f"Yol izinli köklerin dışında: {resolved} (izinli: {izinli})",
                str(resolved),
            )

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Yol bulunamadı: {resolved}")

        self._resolved += 1
        return resolved

    def resolve_write(self, path: str, must_exist: bool = False) -> Path:
        """
        Yazma işlemleri için yol çözer ve `allow_write` bayrağını denetler.

        `resolve_path` + yazma izni kontrolünün birleşimidir.

        Raises:
            SandboxViolation: `allow_write=False` ise veya yol politikaya uymuyorsa.
        """
        if not self._policy.allow_write:
            raise self._violation("Yazma izni kapalı (allow_write=False)", str(path))
        return self.resolve_path(path, must_exist=must_exist)

    def check_path(self, path: str, write: bool = False) -> Tuple[bool, str]:
        """
        Fırlatmayan yol denetimi.

        Args:
            path: Denetlenecek yol.
            write: True ise yazma izni de denetlenir.

        Returns:
            ``(ok, sebep)`` — ok True ise sebep "izinli" olur.
        """
        try:
            if write:
                self.resolve_write(path)
            else:
                self.resolve_path(path)
            return True, "izinli"
        except SandboxViolation as exc:
            return False, str(exc)
        except FileNotFoundError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"Yol denetlenemedi: {exc}"

    def _is_allowed(self, path: Path) -> bool:
        """Yol izinli köklerden birinin altında mı?"""
        target = os.path.normcase(str(path))
        for root in self._allowed_roots:
            root_str = os.path.normcase(str(root))
            if target == root_str:
                return True
            if target.startswith(root_str.rstrip("\\/") + os.sep):
                return True
        return False

    def _match_denied(self, path: Path) -> str:
        """
        Yolu yasaklı desenlerle karşılaştırır.

        Returns:
            Eşleşen desen (yoksa boş string).
        """
        native = str(path)
        forward = native.replace("\\", "/")
        candidates = [native, forward, native.lower(), forward.lower()]

        for pattern in self._policy.denied_patterns:
            if not pattern:
                continue
            variants = {pattern, pattern.replace("\\", "/"), pattern.replace("/", "\\")}
            for variant in variants:
                for candidate in candidates:
                    try:
                        if fnmatch.fnmatch(candidate, variant):
                            return pattern
                    except Exception:
                        continue
                # "…/**" ile biten desenler için dizinin kendisi de yasak
                stripped = variant.rstrip("*")
                if stripped.endswith(("/", "\\")):
                    base = stripped.rstrip("/\\").lower()
                    if base and (forward.lower() == base or native.lower() == base):
                        return pattern
        return ""

    # ─── Komut Denetimi ─────────────────────────────────────────────────

    def is_dangerous(self, command: str) -> Tuple[bool, str]:
        """
        Komutun yıkıcı olup olmadığını tespit eder.

        Büyük/küçük harf duyarsızdır, fazladan boşluk ve bayraklara karşı
        toleranslıdır.

        Args:
            command: Denetlenecek komut satırı.

        Returns:
            ``(True, gerekçe)`` yıkıcıysa, aksi halde ``(False, "temiz")``.
        """
        if not command or not str(command).strip():
            return False, "temiz"

        text = " ".join(str(command).split()).lower()

        compact = re.sub(r"\s+", "", text)
        for signature in _FORK_BOMB_SIGNATURES:
            if signature and signature in compact:
                return True, "Fork bombası deseni tespit edildi"

        if re.search(r"\w+\(\)\s*\{[^}]*\|[^}]*&[^}]*\}\s*;\s*\w+", text):
            return True, "Fork bombası deseni tespit edildi"

        for pattern, reason in _DANGEROUS_COMPILED:
            if pattern.search(text):
                return True, reason

        return False, "temiz"

    def check_command(self, command: str) -> Tuple[bool, str]:
        """
        Komutu kabuk izni + yıkıcılık açısından denetler (fırlatmaz).

        Returns:
            ``(ok, sebep)`` — ok True ise sebep "izinli" olur.
        """
        self._commands_checked += 1

        if not self._policy.allow_shell:
            self._commands_blocked += 1
            return False, "Kabuk çalıştırma izni kapalı (allow_shell=False)"

        dangerous, reason = self.is_dangerous(command)
        if dangerous:
            self._commands_blocked += 1
            logger.warning("Yıkıcı komut engellendi: %s (%s)", command, reason)
            if self._event_bus:
                try:
                    self._event_bus.emit(
                        "os.command_blocked",
                        {"command": str(command)[:200], "reason": reason},
                        source="os.sandbox",
                    )
                except Exception as exc:
                    logger.debug("Engelleme olayı yayınlanamadı: %s", exc)
            return False, reason

        return True, "izinli"

    def check_registry_write(self) -> Tuple[bool, str]:
        """Kayıt defterine yazma iznini denetler (fırlatmaz)."""
        if not self._policy.allow_registry_write:
            return False, "Kayıt defterine yazma izni kapalı (allow_registry_write=False)"
        return True, "izinli"

    # ─── İhlal Kaydı ────────────────────────────────────────────────────

    def _violation(self, reason: str, target: str) -> SandboxViolation:
        """İhlali sayar, loglar, olay yayınlar ve istisna nesnesi döndürür."""
        self._violations += 1
        self._last_violation = reason
        logger.warning("Sandbox ihlali: %s", reason)
        if self._event_bus:
            try:
                self._event_bus.emit(
                    "os.sandbox_violation",
                    {"reason": reason, "target": str(target)[:200], "count": self._violations},
                    source="os.sandbox",
                )
            except Exception as exc:
                logger.debug("İhlal olayı yayınlanamadı: %s", exc)
        return SandboxViolation(reason)

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sandbox metrikleri."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "root": str(self._root),
            "allowed_roots": [str(p) for p in self._allowed_roots],
            "denied_patterns": list(self._policy.denied_patterns),
            "allow_shell": self._policy.allow_shell,
            "allow_write": self._policy.allow_write,
            "allow_process_kill": self._policy.allow_process_kill,
            "allow_registry_write": self._policy.allow_registry_write,
            "max_command_seconds": self._policy.max_command_seconds,
            "dry_run": self._policy.dry_run,
            "path_checks": self._checks,
            "paths_resolved": self._resolved,
            "violations": self._violations,
            "last_violation": self._last_violation,
            "commands_checked": self._commands_checked,
            "commands_blocked": self._commands_blocked,
        }


__all__ = [
    "Sandbox",
    "SandboxPolicy",
    "SandboxViolation",
    "DEFAULT_DENIED_PATTERNS",
]
