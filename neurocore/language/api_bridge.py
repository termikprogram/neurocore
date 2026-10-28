"""
NeuroCore API Köprüsü (Dil Katmanı)
====================================
Uzak LLM/çeviri servislerine gerçek HTTP çağrıları yapan köprü.

Sağlayıcılar:
    - openai    : POST https://api.openai.com/v1/chat/completions
    - anthropic : POST https://api.anthropic.com/v1/messages
    - deepl     : POST https://api-free.deepl.com/v2/translate (form verisi)
    - custom    : OpenAI uyumlu herhangi bir uç nokta

Tasarım:
    - Senkron yol **yalnızca stdlib** (`urllib.request`) kullanır → sıfır
      zorunlu bağımlılık.
    - `aiohttp` varsa `atranslate` / `acomplete` asenkron sarmalayıcıları
      da sunulur; yoksa sessizce başarısız sonuç döner.
    - API anahtarları **yalnızca ortam değişkenlerinden** okunur ve asla
      loglanmaz (maskelenir: `sk-…abcd`).

Kullanım:
    bridge = ApiBridge()
    bridge.initialize(config, event_bus)
    if bridge.available_providers():
        sonuc = bridge.translate("merhaba", "tr", "en")
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from neurocore.utils.logger import NeuroCoreLogger
from neurocore.language.dictionary import LANGUAGE_LABELS, normalize_language_code

logger = NeuroCoreLogger.get_logger("language.api")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: Çeviri sonucu üretirken kullanılan sağlayıcı kimlikleri
PROVIDER_KINDS: Tuple[str, ...] = ("openai", "anthropic", "deepl", "custom")

#: Anahtar arama sırası: NEUROCORE_* önce, sonra genel değişken
API_KEY_ENV_VARS: Dict[str, Tuple[str, ...]] = {
    "openai": ("NEUROCORE_OPENAI_API_KEY", "OPENAI_API_KEY"),
    "anthropic": ("NEUROCORE_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    "deepl": ("NEUROCORE_DEEPL_API_KEY", "DEEPL_API_KEY"),
}

#: Bağlantı ölçümü için hedefler (TCP connect)
CONNECTIVITY_TARGETS: Tuple[Tuple[str, int], ...] = (
    ("1.1.1.1", 443),
    ("api.openai.com", 443),
)

#: Bağlantı sonucunun önbellekte kalma süresi (saniye)
CONNECTIVITY_TTL_SEC: float = 60.0

#: OpenAI sohbet tamamlama uç noktası
OPENAI_URL: str = "https://api.openai.com/v1/chat/completions"
OPENAI_DEFAULT_MODEL: str = "gpt-4o-mini"
#: Anthropic mesaj uç noktası
ANTHROPIC_URL: str = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION: str = "2023-06-01"
ANTHROPIC_DEFAULT_MODEL: str = "claude-3-5-haiku-20241022"
#: DeepL çeviri uç noktası
DEEPL_URL: str = "https://api-free.deepl.com/v2/translate"
#: DeepL hedef dil eşlemesi (bölgesel varyant zorunlu)
DEEPL_LANG_MAP: Dict[str, str] = {
    "en": "EN-US", "pt": "PT-PT", "zh": "ZH-HANS",
}


# ─── Anahtar Yardımcıları ──────────────────────────────────────────────────

def mask_key(secret: str) -> str:
    """
    Gizli anahtarı loglanabilir biçime maskeler.

    Örnekler:
        'sk-proj-abcdefgh1234' → 'sk-…1234'
        'abcdef'               → '…cdef'
        ''                     → ''

    Args:
        secret: Ham anahtar

    Returns:
        Maskelenmiş gösterim (asla tam anahtar içermez)
    """
    if not secret:
        return ""
    text = str(secret)
    tail = text[-4:] if len(text) > 4 else text
    prefix = "sk-" if text.startswith("sk-") else ""
    return f"{prefix}…{tail}"


def resolve_api_key(env_names: Tuple[str, ...]) -> str:
    """
    Ortam değişkenlerinden anahtarı bulur (ilk dolu olan).

    Args:
        env_names: Denenecek değişken adları (öncelik sırasıyla)

    Returns:
        Anahtar veya boş string
    """
    for name in env_names:
        value = os.environ.get(name, "")
        if value and str(value).strip():
            return str(value).strip()
    return ""


# ─── Sağlayıcı Tanımı ──────────────────────────────────────────────────────

@dataclass
class ApiProvider:
    """
    Tek bir API sağlayıcı tanımı.

    Attributes:
        name: Sağlayıcı kimliği ('openai', 'anthropic', ...)
        kind: Sağlayıcı türü ('openai' | 'anthropic' | 'deepl' | 'custom')
        base_url: Uç nokta URL'si
        model: Model kimliği (LLM sağlayıcıları için)
        api_key_env: Virgülle ayrılmış ortam değişkeni adları
        enabled: Sağlayıcı etkin mi
        timeout: Varsayılan zaman aşımı (saniye)
        note: Serbest açıklama
    """
    name: str
    kind: str
    base_url: str
    model: str = ""
    api_key_env: str = ""
    enabled: bool = True
    timeout: float = 20.0
    note: str = ""

    def key_env_names(self) -> Tuple[str, ...]:
        """Anahtar için denenecek ortam değişkeni adları."""
        names: List[str] = []
        if self.api_key_env:
            names.extend(part.strip() for part in str(self.api_key_env).split(",") if part.strip())
        names.extend(API_KEY_ENV_VARS.get(self.kind, ()))
        # Yinelenenleri sırayı koruyarak kaldır
        seen: set = set()
        unique: List[str] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        return tuple(unique)

    def api_key(self) -> str:
        """Sağlayıcının API anahtarını ortamdan okur (yoksa '')."""
        return resolve_api_key(self.key_env_names())

    def has_key(self) -> bool:
        """Ortamda geçerli bir anahtar var mı?"""
        return bool(self.api_key())

    def masked_key(self) -> str:
        """Maskelenmiş anahtar gösterimi (loglama için güvenli)."""
        return mask_key(self.api_key())

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlayıcı özeti (anahtar İÇERMEZ)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "enabled": self.enabled,
            "has_key": self.has_key(),
            "masked_key": self.masked_key(),
            "timeout": self.timeout,
        }


# ─── Ana Köprü Sınıfı ──────────────────────────────────────────────────────

class ApiBridge:
    """
    Uzak API sağlayıcılarına erişim köprüsü.

    Özellikler:
        - Sağlayıcı kaydı + anahtar varlığına göre otomatik seçim
        - Gerçek HTTP (stdlib urllib), zaman aşımlı, hata toleranslı
        - `config.language.api_fallback` False ise hiç ağ çağrısı yapılmaz
        - `connectivity()` sonucu 60 sn önbelleklenir; başarısız kontrol
          60 sn boyunca tekrarlanmaz

    Kullanım:
        bridge = ApiBridge()
        bridge.initialize(config, event_bus)
        sonuc = bridge.translate("merhaba", "tr", "en")
        cevap = bridge.complete("Merhaba de", max_tokens=64)
    """

    name: str = "language.api"

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — ağ erişimi yok."""
        self._initialized: bool = False
        self._config = None
        self._event_bus = None
        self._providers: Dict[str, ApiProvider] = {}
        self._order: List[str] = []
        self._api_fallback: bool = True
        self._connectivity: Dict[str, Any] = {}
        self._connectivity_ts: float = 0.0
        self._lock = threading.RLock()
        self._stats: Dict[str, int] = {
            "translate_calls": 0,
            "translate_success": 0,
            "translate_failed": 0,
            "translate_disabled": 0,
            "complete_calls": 0,
            "complete_success": 0,
            "complete_failed": 0,
            "http_requests": 0,
            "http_errors": 0,
            "connectivity_checks": 0,
            "connectivity_skips": 0,
            "events": 0,
        }
        self._last_error: str = ""
        self._last_provider: str = ""

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config=None, event_bus=None) -> None:
        """
        Köprüyü başlat ve varsayılan sağlayıcıları kaydet.

        İdempotenttir: ikinci çağrı no-op'tur.

        Args:
            config: SystemConfig benzeri nesne (None → varsayılanlar)
            event_bus: Olay yolu (None → olay yayınlanmaz)
        """
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus

        language_cfg = getattr(config, "language", None) if config is not None else None
        fallback = getattr(language_cfg, "api_fallback", True)
        self._api_fallback = bool(fallback) if fallback is not None else True

        self._register_defaults(config)

        self._initialized = True
        with self._lock:
            available = self._available_names_locked()
        logger.info(
            f"API köprüsü hazır: {len(self._providers)} sağlayıcı, "
            f"anahtarlı: {available or 'yok'}, fallback: {self._api_fallback}"
        )
        if self._event_bus:
            self._stats["events"] += 1
            self._event_bus.emit(
                "language.api_ready",
                {
                    "providers": list(self._providers),
                    "available": available,
                    "api_fallback": self._api_fallback,
                },
                source=self.name,
            )

    def shutdown(self) -> None:
        """Kaynakları serbest bırak. İdempotenttir."""
        if not self._initialized:
            return
        with self._lock:
            self._connectivity = {}
            self._connectivity_ts = 0.0
        self._initialized = False
        logger.info("API köprüsü kapatıldı")

    def _register_defaults(self, config=None) -> None:
        """Varsayılan sağlayıcıları (openai, anthropic, deepl) kaydeder."""
        language_cfg = getattr(config, "language", None) if config is not None else None
        configured = getattr(language_cfg, "api_providers", None) if language_cfg is not None else None
        if not isinstance(configured, (list, tuple)) or not configured:
            configured = ["openai", "anthropic"]

        defaults: Dict[str, ApiProvider] = {
            "openai": ApiProvider(
                name="openai", kind="openai", base_url=OPENAI_URL,
                model=OPENAI_DEFAULT_MODEL,
                api_key_env="NEUROCORE_OPENAI_API_KEY,OPENAI_API_KEY",
                timeout=20.0, note="OpenAI sohbet tamamlama",
            ),
            "anthropic": ApiProvider(
                name="anthropic", kind="anthropic", base_url=ANTHROPIC_URL,
                model=ANTHROPIC_DEFAULT_MODEL,
                api_key_env="NEUROCORE_ANTHROPIC_API_KEY,ANTHROPIC_API_KEY",
                timeout=30.0, note="Anthropic mesajlar API'si",
            ),
            "deepl": ApiProvider(
                name="deepl", kind="deepl", base_url=DEEPL_URL,
                model="",
                api_key_env="NEUROCORE_DEEPL_API_KEY,DEEPL_API_KEY",
                timeout=20.0, note="DeepL çeviri servisi",
            ),
        }

        for name in configured:
            key = str(name).strip().lower()
            if not key:
                continue
            provider = defaults.get(key)
            if provider is None:
                continue
            with self._lock:
                if provider.name not in self._providers:
                    self._providers[provider.name] = provider
                    self._order.append(provider.name)

        # DeepL her zaman kayıtlıdır (anahtarı varsa kullanılabilir)
        if "deepl" not in self._providers:
            self.register(defaults["deepl"])

    # ─── Kayıt ─────────────────────────────────────────────────────────

    def register(self, provider: ApiProvider) -> None:
        """
        Yeni bir sağlayıcı kaydeder (varsa günceller).

        Args:
            provider: Kaydedilecek sağlayıcı
        """
        if not isinstance(provider, ApiProvider) or not provider.name:
            logger.debug("Geçersiz API sağlayıcı kaydı atlandı")
            return
        name = str(provider.name).strip().lower()
        provider.name = name
        if not provider.kind:
            provider.kind = "custom"
        with self._lock:
            if name not in self._providers:
                self._order.append(name)
            self._providers[name] = provider
        logger.debug(
            f"API sağlayıcı kaydedildi: {name} ({provider.kind}), "
            f"anahtar: {'var ' + provider.masked_key() if provider.has_key() else 'yok'}"
        )

    def register_custom(
        self,
        name: str,
        base_url: str,
        model: str = "",
        api_key_env: str = "",
    ) -> None:
        """OpenAI uyumlu özel bir uç noktayı kaydeder."""
        self.register(ApiProvider(
            name=name, kind="custom", base_url=base_url, model=model,
            api_key_env=api_key_env or name.upper() + "_API_KEY",
        ))

    def providers(self) -> List[Dict[str, Any]]:
        """Kayıtlı tüm sağlayıcıların (maskelenmiş) özetleri."""
        with self._lock:
            return [self._providers[name].to_dict() for name in self._order if name in self._providers]

    def available_providers(self) -> List[str]:
        """
        Etkin ve anahtarı olan sağlayıcı adları.

        Returns:
            Sağlayıcı adları (kayıt sırasına göre)
        """
        with self._lock:
            return self._available_names_locked()

    def _available_names_locked(self) -> List[str]:
        """Kilit altında çağrılması gereken kullanılabilirlik hesabı."""
        names: List[str] = []
        for name in self._order:
            provider = self._providers.get(name)
            if provider is None or not provider.enabled:
                continue
            if provider.has_key():
                names.append(name)
        return names

    def _resolve_provider(self, provider: Optional[str] = None) -> Optional[ApiProvider]:
        """
        Kullanılacak sağlayıcıyı seçer.

        Args:
            provider: İstenen sağlayıcı adı (None → anahtarı olan ilk sağlayıcı)

        Returns:
            ApiProvider veya None (kullanılabilir sağlayıcı yok)
        """
        with self._lock:
            if provider:
                candidate = self._providers.get(str(provider).strip().lower())
                if candidate is None or not candidate.enabled or not candidate.has_key():
                    return None
                return candidate
            available = self._available_names_locked()
            if not available:
                return None
            return self._providers.get(available[0])

    # ─── Çeviri ────────────────────────────────────────────────────────

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        provider: Optional[str] = None,
        timeout: float = 20.0,
    ) -> Any:
        """
        Metni API sağlayıcısıyla çevirir.

        `config.language.api_fallback` False ise **hiç ağ çağrısı yapılmaz**;
        `warnings=["api_devre_disi"]` içeren başarısız sonuç döner.

        Args:
            text: Kaynak metin
            source_lang: Kaynak dil kodu
            target_lang: Hedef dil kodu
            provider: Belirli sağlayıcı adı (None → otomatik)
            timeout: Zaman aşımı (saniye)

        Returns:
            TranslationResult
        """
        from neurocore.language.translator import TranslationResult, make_failed_result

        self._stats["translate_calls"] += 1
        src = normalize_language_code(source_lang) or "tr"
        dst = normalize_language_code(target_lang) or "en"

        if not text or not str(text).strip():
            self._stats["translate_failed"] += 1
            return make_failed_result(text, src, dst, provider="api", warnings=["bos_metin"])

        if not self._api_fallback:
            self._stats["translate_disabled"] += 1
            logger.debug("API çevirisi devre dışı (api_fallback=False) — ağ çağrısı yok")
            return make_failed_result(
                text, src, dst, provider="api", warnings=["api_devre_disi"],
            )

        selected = self._resolve_provider(provider)
        if selected is None:
            self._stats["translate_failed"] += 1
            wanted = provider or "otomatik"
            return make_failed_result(
                text, src, dst, provider="api",
                warnings=[f"api_saglayici_yok: {wanted}"],
            )

        # Anahtarı olmayan sağlayıcı → ağ çağrısı yok
        if not selected.has_key():
            self._stats["translate_failed"] += 1
            return make_failed_result(
                text, src, dst, provider="api",
                warnings=["api_anahtari_yok"],
            )

        try:
            if selected.kind == "deepl":
                translated = self._deepl_translate(selected, str(text), src, dst, timeout)
            else:
                translated = self._llm_translate(selected, str(text), src, dst, timeout)
        except Exception as exc:
            self._stats["translate_failed"] += 1
            self._last_error = str(exc)
            logger.warning(f"API çevirisi başarısız ({selected.name}): {exc}")
            return make_failed_result(
                text, src, dst, provider="api",
                warnings=[f"api_hatasi: {exc}"],
            )

        if not translated:
            self._stats["translate_failed"] += 1
            return make_failed_result(
                text, src, dst, provider="api",
                warnings=["api_bos_yanit"],
            )

        self._stats["translate_success"] += 1
        self._last_provider = selected.name
        return TranslationResult(
            text=translated,
            source_lang=src,
            target_lang=dst,
            provider="api",
            confidence=0.8,
            warnings=[f"saglayici:{selected.name}"],
        )

    def _llm_translate(
        self,
        provider: ApiProvider,
        text: str,
        src: str,
        dst: str,
        timeout: float,
    ) -> str:
        """LLM tabanlı sağlayıcı (OpenAI/Anthropic/custom) ile çeviri yapar."""
        src_name = LANGUAGE_LABELS.get(src, src)
        dst_name = LANGUAGE_LABELS.get(dst, dst)
        prompt = (
            f"Translate the following text from {src_name} ({src}) to {dst_name} ({dst}). "
            "Preserve technical terms, numbers, markdown and formatting. "
            "Return ONLY the translation, without explanations or quotes.\n\n"
            f"{text}"
        )

        if provider.kind == "anthropic":
            system = (
                "You are a professional translator. Output only the translated text."
            )
            payload = {
                "model": provider.model or ANTHROPIC_DEFAULT_MODEL,
                "max_tokens": max(256, min(4096, len(text) * 3 + 256)),
                "temperature": 0.2,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }
            headers = {
                "x-api-key": provider.api_key(),
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            }
            data = self._post_json(provider.base_url, payload, headers, timeout)
            blocks = data.get("content", []) if isinstance(data, dict) else []
            parts = [
                str(block.get("text", ""))
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(part for part in parts if part).strip()

        # openai + custom (OpenAI uyumlu)
        payload = {
            "model": provider.model or OPENAI_DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a professional translator. Output only the translated text."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {provider.api_key()}",
            "Content-Type": "application/json",
        }
        data = self._post_json(provider.base_url, payload, headers, timeout)
        return self._extract_openai_text(data)

    def _deepl_translate(
        self,
        provider: ApiProvider,
        text: str,
        src: str,
        dst: str,
        timeout: float,
    ) -> str:
        """DeepL çeviri uç noktasına form verisiyle istek atar."""
        target_code = DEEPL_LANG_MAP.get(dst, dst.upper())
        form: List[Tuple[str, str]] = [("text", text), ("target_lang", target_code)]
        if src and src != "auto":
            form.append(("source_lang", DEEPL_LANG_MAP.get(src, src.upper()).split("-")[0]))

        body = urllib.parse.urlencode(form).encode("utf-8")
        headers = {
            "Authorization": f"DeepL-Auth-Key {provider.api_key()}",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "NeuroCore/0.1",
        }
        data = self._post_json(provider.base_url, None, headers, timeout, raw_body=body)
        translations = data.get("translations", []) if isinstance(data, dict) else []
        if translations and isinstance(translations[0], dict):
            return str(translations[0].get("text", "")).strip()
        return ""

    @staticmethod
    def _extract_openai_text(data: Any) -> str:
        """OpenAI yanıtından metni çıkarır."""
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices", [])
        if not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message", {})
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, list):
                # Çok parçalı içerik (vision vb.)
                return "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                ).strip()
            return str(content or "").strip()
        return str(first.get("text", "")).strip()

    # ─── Genel Tamamlama ───────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        provider: Optional[str] = None,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Sistemin geri kalanı için genel amaçlı metin tamamlama.

        Args:
            prompt: Kullanıcı istemi
            provider: Belirli sağlayıcı adı (None → otomatik)
            system: Sistem istemi
            max_tokens: Üretilecek en fazla jeton
            temperature: Örnekleme sıcaklığı
            timeout: Zaman aşımı (saniye)

        Returns:
            {"ok": bool, "text": str, "provider": str, "error": str}
        """
        self._stats["complete_calls"] += 1

        if not prompt or not str(prompt).strip():
            self._stats["complete_failed"] += 1
            return {"ok": False, "text": "", "provider": provider or "", "error": "bos_istem"}

        if not self._api_fallback:
            self._stats["complete_failed"] += 1
            return {"ok": False, "text": "", "provider": provider or "", "error": "api_devre_disi"}

        selected = self._resolve_provider(provider)
        if selected is None:
            self._stats["complete_failed"] += 1
            return {
                "ok": False, "text": "", "provider": provider or "",
                "error": "api_saglayici_yok",
            }
        if not selected.has_key():
            self._stats["complete_failed"] += 1
            return {
                "ok": False, "text": "", "provider": selected.name,
                "error": "api_anahtari_yok",
            }
        if selected.kind == "deepl":
            self._stats["complete_failed"] += 1
            return {
                "ok": False, "text": "", "provider": selected.name,
                "error": "saglayici_tamamlama_desteklemiyor",
            }

        try:
            if selected.kind == "anthropic":
                payload: Dict[str, Any] = {
                    "model": selected.model or ANTHROPIC_DEFAULT_MODEL,
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "messages": [{"role": "user", "content": str(prompt)}],
                }
                if system:
                    payload["system"] = str(system)
                headers = {
                    "x-api-key": selected.api_key(),
                    "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                }
                data = self._post_json(selected.base_url, payload, headers, timeout)
                blocks = data.get("content", []) if isinstance(data, dict) else []
                text = "\n".join(
                    str(block.get("text", ""))
                    for block in blocks
                    if isinstance(block, dict) and block.get("type") == "text"
                ).strip()
            else:
                messages: List[Dict[str, str]] = []
                if system:
                    messages.append({"role": "system", "content": str(system)})
                messages.append({"role": "user", "content": str(prompt)})
                payload = {
                    "model": selected.model or OPENAI_DEFAULT_MODEL,
                    "messages": messages,
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                }
                headers = {
                    "Authorization": f"Bearer {selected.api_key()}",
                    "Content-Type": "application/json",
                }
                data = self._post_json(selected.base_url, payload, headers, timeout)
                text = self._extract_openai_text(data)
        except Exception as exc:
            self._stats["complete_failed"] += 1
            self._last_error = str(exc)
            logger.warning(f"API tamamlama başarısız ({selected.name}): {exc}")
            return {"ok": False, "text": "", "provider": selected.name, "error": str(exc)}

        if not text:
            self._stats["complete_failed"] += 1
            return {"ok": False, "text": "", "provider": selected.name, "error": "api_bos_yanit"}

        self._stats["complete_success"] += 1
        self._last_provider = selected.name
        return {"ok": True, "text": text, "provider": selected.name, "error": ""}

    # ─── HTTP Katmanı (stdlib) ─────────────────────────────────────────

    def _build_opener(self):
        """Proxy ortam değişkenlerini onurlandıran bir urllib opener üretir."""
        proxy = (
            os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        )
        if proxy:
            handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            return urllib.request.build_opener(handler)
        return urllib.request.build_opener()

    def _post_json(
        self,
        url: str,
        payload: Optional[Dict[str, Any]],
        headers: Dict[str, str],
        timeout: float,
        raw_body: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        JSON (veya hazır gövde) ile POST atar ve JSON yanıtı döndürür.

        Ağ erişimi burada gerçekleşir. Tüm hatalar istisna olarak yükselir;
        çağıran taraf yakalar.

        Args:
            url: Uç nokta
            payload: JSON gövdesi (None → raw_body kullanılır)
            headers: HTTP başlıkları (anahtar içerir, loglanmaz)
            timeout: Zaman aşımı (saniye)
            raw_body: Hazır gövde (form verisi için)

        Returns:
            Ayrıştırılmış JSON sözlüğü
        """
        body = raw_body if raw_body is not None else json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        for key, value in (headers or {}).items():
            if value:
                request.add_header(key, value)
        if raw_body is None:
            request.add_header("Content-Type", "application/json")
        request.add_header("User-Agent", "NeuroCore/0.1")

        effective_timeout = float(timeout) if timeout and timeout > 0 else 20.0
        self._stats["http_requests"] += 1
        logger.debug(f"HTTP POST -> {url} (zaman aşımı {effective_timeout:.0f}s)")

        try:
            opener = self._build_opener()
            with opener.open(request, timeout=effective_timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            self._stats["http_errors"] += 1
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # pragma: no cover - savunma
                detail = ""
            raise RuntimeError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            self._stats["http_errors"] += 1
            raise RuntimeError(f"baglanti_hatasi: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            self._stats["http_errors"] += 1
            raise RuntimeError("zaman_asimi") from exc

        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            self._stats["http_errors"] += 1
            raise RuntimeError(f"gecersiz_json: {raw[:200]}") from exc

    # ─── Bağlantı Durumu ───────────────────────────────────────────────

    def connectivity(self, timeout: float = 4.0) -> Dict[str, Any]:
        """
        Ağ erişimini TCP bağlantısıyla ölçer (sonuç 60 sn önbelleklenir).

        Son kontrol başarısızsa 60 saniye boyunca ağa tekrar dokunulmaz.

        Args:
            timeout: Tek bir hedef için zaman aşımı (saniye)

        Returns:
            {"online": bool, "checked": [urls], "latency_ms": float}
        """
        now = time.time()
        with self._lock:
            cached = dict(self._connectivity)
            age = now - self._connectivity_ts

        if cached and age < CONNECTIVITY_TTL_SEC:
            self._stats["connectivity_skips"] += 1
            return cached

        self._stats["connectivity_checks"] += 1
        checked: List[str] = []
        latency_ms = 0.0
        online = False

        for host, port in CONNECTIVITY_TARGETS:
            checked.append(f"{host}:{port}")
            started = time.perf_counter()
            try:
                with socket.create_connection((host, port), timeout=float(timeout)):
                    elapsed = (time.perf_counter() - started) * 1000.0
                    latency_ms = elapsed
                    online = True
                    break
            except Exception:
                continue

        snapshot = {
            "online": bool(online),
            "checked": checked,
            "latency_ms": round(float(latency_ms), 2),
            "ts": round(now, 3),
        }
        with self._lock:
            self._connectivity = snapshot
            self._connectivity_ts = now

        if self._event_bus and not online:
            self._stats["events"] += 1
            self._event_bus.emit(
                "language.offline",
                {"checked": checked, "timeout": timeout},
                source=self.name,
            )
        return snapshot

    def is_online(self, timeout: float = 4.0) -> bool:
        """Ağ erişimi var mı? (connectivity() önbelleğini kullanır)"""
        return bool(self.connectivity(timeout).get("online", False))

    # ─── Asenkron Sarmalayıcılar (aiohttp, isteğe bağlı) ───────────────

    async def atranslate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        provider: Optional[str] = None,
        timeout: float = 20.0,
    ) -> Any:
        """
        `aiohttp` ile asenkron çeviri.

        `aiohttp` yoksa veya istek başarısız olursa başarısız bir
        TranslationResult döner (istisna yükseltmez).
        """
        from neurocore.language.translator import make_failed_result

        src = normalize_language_code(source_lang) or "tr"
        dst = normalize_language_code(target_lang) or "en"

        if not self._api_fallback:
            return make_failed_result(text, src, dst, provider="api", warnings=["api_devre_disi"])

        selected = self._resolve_provider(provider)
        if selected is None or not selected.has_key():
            return make_failed_result(text, src, dst, provider="api", warnings=["api_saglayici_yok"])

        try:
            import aiohttp
        except Exception:
            logger.debug("aiohttp yok — asenkron çeviri senkron yola düşürüldü")
            return self.translate(text, src, dst, provider=provider, timeout=timeout)

        if selected.kind == "deepl":
            return self.translate(text, src, dst, provider=provider, timeout=timeout)

        src_name = LANGUAGE_LABELS.get(src, src)
        dst_name = LANGUAGE_LABELS.get(dst, dst)
        prompt = (
            f"Translate the following text from {src_name} ({src}) to {dst_name} ({dst}). "
            "Return ONLY the translation.\n\n" + str(text)
        )

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=float(timeout))) as session:
                if selected.kind == "anthropic":
                    url = selected.base_url or ANTHROPIC_URL
                    headers = {
                        "x-api-key": selected.api_key(),
                        "anthropic-version": ANTHROPIC_VERSION,
                        "content-type": "application/json",
                    }
                    payload = {
                        "model": selected.model or ANTHROPIC_DEFAULT_MODEL,
                        "max_tokens": 2048,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                else:
                    url = selected.base_url or OPENAI_URL
                    headers = {
                        "Authorization": f"Bearer {selected.api_key()}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": selected.model or OPENAI_DEFAULT_MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a professional translator. Output only the translation."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                    }

                self._stats["http_requests"] += 1
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:300]
                        self._stats["http_errors"] += 1
                        raise RuntimeError(f"HTTP {response.status}: {detail}")
                    data = await response.json()
        except Exception as exc:
            self._stats["translate_failed"] += 1
            self._last_error = str(exc)
            logger.warning(f"Asenkron API çevirisi başarısız ({selected.name}): {exc}")
            return make_failed_result(text, src, dst, provider="api", warnings=[f"api_hatasi: {exc}"])

        from neurocore.language.translator import TranslationResult
        translated = (
            self._extract_openai_text(data) if selected.kind != "anthropic"
            else "\n".join(
                str(block.get("text", ""))
                for block in (data.get("content", []) if isinstance(data, dict) else [])
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
        )
        if not translated:
            self._stats["translate_failed"] += 1
            return make_failed_result(text, src, dst, provider="api", warnings=["api_bos_yanit"])

        self._stats["translate_success"] += 1
        return TranslationResult(
            text=translated, source_lang=src, target_lang=dst,
            provider="api", confidence=0.8, warnings=[f"saglayici:{selected.name}"],
        )

    async def acomplete(
        self,
        prompt: str,
        provider: Optional[str] = None,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        `aiohttp` ile asenkron metin tamamlama.

        `aiohttp` yoksa senkron yola düşer. Asla istisna yükseltmez.
        """
        try:
            import aiohttp
        except Exception:
            return self.complete(
                prompt, provider=provider, system=system,
                max_tokens=max_tokens, temperature=temperature, timeout=timeout,
            )

        if not self._api_fallback:
            return {"ok": False, "text": "", "provider": provider or "", "error": "api_devre_disi"}

        selected = self._resolve_provider(provider)
        if selected is None or not selected.has_key() or selected.kind == "deepl":
            return {
                "ok": False, "text": "", "provider": provider or "",
                "error": "api_saglayici_yok",
            }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=float(timeout))) as session:
                if selected.kind == "anthropic":
                    url = selected.base_url or ANTHROPIC_URL
                    headers = {
                        "x-api-key": selected.api_key(),
                        "anthropic-version": ANTHROPIC_VERSION,
                        "content-type": "application/json",
                    }
                    payload: Dict[str, Any] = {
                        "model": selected.model or ANTHROPIC_DEFAULT_MODEL,
                        "max_tokens": int(max_tokens),
                        "temperature": float(temperature),
                        "messages": [{"role": "user", "content": str(prompt)}],
                    }
                    if system:
                        payload["system"] = str(system)
                else:
                    url = selected.base_url or OPENAI_URL
                    headers = {
                        "Authorization": f"Bearer {selected.api_key()}",
                        "Content-Type": "application/json",
                    }
                    messages: List[Dict[str, str]] = []
                    if system:
                        messages.append({"role": "system", "content": str(system)})
                    messages.append({"role": "user", "content": str(prompt)})
                    payload = {
                        "model": selected.model or OPENAI_DEFAULT_MODEL,
                        "messages": messages,
                        "max_tokens": int(max_tokens),
                        "temperature": float(temperature),
                    }

                self._stats["http_requests"] += 1
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:300]
                        self._stats["http_errors"] += 1
                        return {
                            "ok": False, "text": "", "provider": selected.name,
                            "error": f"HTTP {response.status}: {detail}",
                        }
                    data = await response.json()
        except Exception as exc:
            self._stats["complete_failed"] += 1
            return {"ok": False, "text": "", "provider": selected.name, "error": str(exc)}

        text = (
            self._extract_openai_text(data) if selected.kind != "anthropic"
            else "\n".join(
                str(block.get("text", ""))
                for block in (data.get("content", []) if isinstance(data, dict) else [])
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
        )
        if not text:
            self._stats["complete_failed"] += 1
            return {"ok": False, "text": "", "provider": selected.name, "error": "api_bos_yanit"}

        self._stats["complete_success"] += 1
        return {"ok": True, "text": text, "provider": selected.name, "error": ""}

    # ─── İstatistikler ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        with self._lock:
            providers = [self._providers[name].to_dict() for name in self._order if name in self._providers]
            available = self._available_names_locked()
            connectivity = dict(self._connectivity)
            connectivity_age = (
                round(time.time() - self._connectivity_ts, 1) if self._connectivity_ts else None
            )
        return {
            "name": self.name,
            "initialized": self._initialized,
            "api_fallback": self._api_fallback,
            "providers": providers,
            "available": available,
            "last_provider": self._last_provider,
            "last_error": self._last_error,
            "connectivity": connectivity,
            "connectivity_age_sec": connectivity_age,
            **self._stats,
        }
