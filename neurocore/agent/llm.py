"""
NeuroCore Dil Modeli Sağlayıcıları
===================================
Ajanın "düşünme" katmanı: hangi model çağrılır, nasıl çağrılır, yoksa ne olur?

Neden ayrı katman?
    Ajan döngüsü bir modele ihtiyaç duyar ama **hangi** modele bağlı olmamalıdır.
    Bu makinede durum şu: DSH oturum jetonu platform API'si için geçerli değil,
    Ollama/LM Studio kurulu değil, NeuroCore'un kendi transformer'ı (208M) ise
    güvenilir araç çağrısı yapamayacak kadar küçük. Bu yüzden sağlayıcılar
    **takılıp çıkarılabilir** olmalı ve hiçbiri yoksa sistem çalışmaya devam
    etmelidir (kural tabanlı planlayıcı).

Sağlayıcı türleri:
    * `OpenAICompatibleProvider` — `/v1/chat/completions` konuşan her servis:
      DeepSeek, OpenAI, Groq, Together, OpenRouter, LM Studio, vLLM, llama.cpp
    * `OllamaProvider` — Ollama'nın `/api/chat` uç noktası
    * `NeuroCoreProvider` — kendi modelimiz (deneysel; araç çağrısı için
      yalnızca JSON istemi üzerinden, güvenilirliği düşük)
    * `EchoProvider` — sağlayıcı yokken test/kapalı çalışma için

Anahtar keşfi sırası (ilk bulunan kazanır):
    1. `config.llm.api_key`
    2. Ortam değişkenleri: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`,
       `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`,
       `NC_LLM_API_KEY`
    3. `data/llm_credentials.json` (arayüzden kaydedilir — dosya izinleri
       kısıtlanır)
    4. Yerel sunucu (`http://127.0.0.1:11434` veya `:1234`) — anahtarsız

Hiçbiri yoksa `available=False` döner ve ajan kural tabanlı moda düşer.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.llm")

#: Yerel sunucu erişilebilirlik önbelleği: "host:port" → (zaman, açık mı)
_PORT_CACHE: Dict[str, Tuple[float, bool]] = {}
_PORT_CACHE_TTL = 10.0

#: HTTP isteklerinde kullanılan uygulama kimliği.
#:
#: ZORUNLUDUR: `urllib` varsayılan olarak `Python-urllib/3.x` gönderir ve
#: Cloudflare korumalı sağlayıcılar (Groq, bazı OpenRouter uçları) bunu bot
#: sayıp isteği `HTTP 403 (error code: 1010)` ile reddeder. Ayrıca bazı
#: sağlayıcılar boş User-Agent'ı da engeller.
USER_AGENT = "NeuroCore/0.1 (+local agent; https://localhost)"


def tcp_reachable(url: str, timeout: float = 1.5) -> bool:
    """
    Bir adresin TCP kapısı açık mı? (sunucu gerçekten dinliyor mu?)

    Neden gerekli: Yerel sağlayıcılar (`127.0.0.1:1234` gibi) anahtar
    istemez; bu yüzden "anahtar var mı" kontrolü onları **her zaman
    kullanılabilir** gösterir. Oysa sunucu çalışmıyorsa ajan her çağrıda
    bağlantı hatası alır ve gereksiz yere yedek sağlayıcıları dener.
    Bu fonksiyon 1.5 saniyelik bir TCP denemesiyle gerçeği söyler ve
    sonucu 10 saniye önbelleğe alır (her çağrıda yavaşlamasın).
    """
    try:
        parsed = urllib.parse.urlparse(url if "://" in url else f"http://{url}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except (ValueError, AttributeError):
        return False

    key = f"{host}:{port}"
    cached = _PORT_CACHE.get(key)
    now = time.time()
    if cached is not None and now - cached[0] < _PORT_CACHE_TTL:
        return cached[1]

    reachable = False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            reachable = True
    except OSError:
        reachable = False
    _PORT_CACHE[key] = (now, reachable)
    return reachable


class LLMError(Exception):
    """Dil modeli çağrısı başarısız."""


# ─── Veri Yapıları ──────────────────────────────────────────────────────────

@dataclass
class LLMMessage:
    """
    Sohbet mesajı (OpenAI biçimiyle uyumlu).

    Attributes:
        role: system | user | assistant | tool
        content: Metin içeriği
        name: Araç adı (yalnızca `tool` rolünde)
        tool_call_id: Yanıtlanan araç çağrısının kimliği
        tool_calls: Asistanın araç çağrıları
        raw: Sağlayıcıya **birebir** gönderilecek ham mesaj. Verilirse
            `to_dict()` bunu kullanır; sağlayıcıya özgü alanların
            (ör. Gemini `thought_signature`) kaybolmasını engeller.
    """
    role: str                       # system | user | assistant | tool
    content: str = ""
    name: str = ""
    tool_call_id: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """API biçimine çevirir (boş alanlar atlanır)."""
        if self.raw:
            # Ham mesajı olduğu gibi gönder; yalnızca rol garanti edilir.
            message = dict(self.raw)
            message.setdefault("role", self.role)
            return message
        data: Dict[str, Any] = {"role": self.role}
        if self.content:
            data["content"] = self.content
        elif self.role == "assistant" and self.tool_calls:
            data["content"] = None
        if self.name:
            data["name"] = self.name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        return data


@dataclass
class LLMResponse:
    """Model yanıtı."""
    ok: bool = False
    text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    #: Sağlayıcının döndürdüğü **ham** asistan mesajı.
    #:
    #: Neden saklanır? Bazı sağlayıcılar araç çağrısına ek alanlar koyar ve
    #: bunların **birebir** geri gönderilmesini şart koşar. Örnek: Gemini,
    #: `thought_signature` alanını `extra_content` içinde döndürür ve
    #: gönderilmezse `HTTP 400 Function call is missing a thought_signature`
    #: hatası verir. Mesajı yeniden inşa etmek bu alanları kaybettirir.
    raw_message: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    finish_reason: str = ""
    error: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """Model araç çağırmak istedi mi?"""
        return bool(self.tool_calls)

    def assistant_message(self) -> Dict[str, Any]:
        """
        Geçmişe eklenecek asistan mesajı.

        Ham mesaj varsa **olduğu gibi** kullanılır (sağlayıcıya özgü alanlar
        korunsun); yoksa `tool_calls` üzerinden yeniden inşa edilir.
        """
        if self.raw_message:
            message = dict(self.raw_message)
            message.setdefault("role", "assistant")
            return message
        return {
            "role": "assistant",
            "content": self.text or None,
            "tool_calls": list(self.tool_calls),
        } if self.tool_calls else {"role": "assistant", "content": self.text or ""}

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "ok": self.ok, "text": self.text,
            "tool_calls": self.tool_calls, "model": self.model,
            "provider": self.provider,
            "tokens": {"prompt": self.prompt_tokens,
                       "completion": self.completion_tokens,
                       "total": self.total_tokens},
            "duration_ms": round(self.duration_ms, 1),
            "finish_reason": self.finish_reason, "error": self.error,
        }


# ─── Sağlayıcı Arayüzü ──────────────────────────────────────────────────────

class LLMProvider:
    """
    Sağlayıcı taban sınıfı.

    Alt sınıflar `chat()` ve `health()` uygular. `supports_tools` özelliği,
    ajanın şema tabanlı araç çağrısı mı yoksa JSON istemi mi kullanacağını
    belirler.
    """

    name = "base"
    supports_tools = False

    def __init__(self, model: str = "", api_key: str = "", base_url: str = "",
                 temperature: float = 0.2, max_tokens: int = 8192,
                 timeout: float = 180.0) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        # Not: 8192 varsayılanı bilinçlidir. "Reasoning" modelleri (gpt-oss,
        # deepseek-reasoner) düşünme token'larını **aynı** bütçeden harcar;
        # düşük bir sınırda (ör. 4096) yanıt tamamen boş dönebilir ve ajan
        # bunu "model yanıt vermiyor" sanır.
        self.max_tokens = max_tokens
        self.timeout = timeout
        #: Bu sağlayıcıda kullanılamadığı **doğrulanmış** model adları. Aynı
        #: modeli her istekte yeniden denemek kullanıcıyı dakikalarca
        #: bekletir; bu küme sayesinde doğrudan çalışan yedeğe geçilir.
        self._dead_models: set = set()
        self._counter = {"calls": 0, "errors": 0, "prompt_tokens": 0,
                         "completion_tokens": 0, "total_ms": 0.0}

    @property
    def available(self) -> bool:
        """Sağlayıcı kullanılabilir mi?"""
        return False

    def chat(self, messages: Sequence[LLMMessage],
             tools: Optional[Sequence[Dict[str, Any]]] = None,
             **kwargs: Any) -> LLMResponse:
        """Mesajları gönderir ve yanıtı döndürür."""
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        """Sağlayıcı sağlık raporu."""
        return {
            "provider": self.name, "available": self.available,
            "model": self.model, "base_url": self.base_url,
            "supports_tools": self.supports_tools,
            "has_api_key": bool(self.api_key),
            "stats": self.get_stats(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Çağrı istatistikleri."""
        return {
            **self._counter,
            "avg_ms": round(self._counter["total_ms"] / self._counter["calls"], 1)
            if self._counter["calls"] else 0.0,
        }

    @staticmethod
    def _is_thought_signature_error(error: str) -> bool:
        """Hata, eksik/geçersiz düşünme imzasından mı kaynaklanıyor?"""
        text = (error or "").lower()
        return "thought_signature" in text or "thought signature" in text

    def _normalize_payload(self, payload: Dict[str, Any]) -> int:
        """
        Sağlayıcıya özgü alanları isteğe uygun hâle getirir.

        **DeepSeek düşünme modu (`reasoning_content`):** Derin düşünme
        modelleri, araç çağrısı içeren asistan mesajlarında
        `reasoning_content` alanının **aynen geri gönderilmesini** bekler.
        Alan yoksa istek reddedilir:

            HTTP 400: The `reasoning_content` in the thinking mode must be
            passed back to the API.

        ⚠ Bu hata **sağlayıcı değişiminde** kaçınılmaz olarak ortaya çıkar:
        Groq/Gemini'den gelen asistan mesajlarında bu alan yoktur; zincir
        DeepSeek'e düştüğünde istek geçersiz olur. Ölçüldü: bu yüzden ajan
        üç sağlayıcının **hepsi** başarısız olunca görevi bırakıyordu.

        Çözüm: alan yoksa **boş dize** eklenir. DeepSeek boş değeri kabul
        eder (ölçüldü ve doğrulandı) — mevcut düşünme metni varsa
        **dokunulmaz**.

        Returns:
            Eklenen alan sayısı
        """
        eklenen = 0
        for message in payload.get("messages", []):
            if message.get("role") != "assistant":
                continue
            if not message.get("tool_calls"):
                continue
            if "reasoning_content" not in message:
                message["reasoning_content"] = ""
                eklenen += 1
        return eklenen

    @staticmethod
    def _strip_thought_signatures(payload: Dict[str, Any]) -> int:
        """
        İstekteki asistan araç çağrılarından imza alanlarını temizler.

        Neden gerekli? Gemini, araç çağrılarına bir **düşünme imzası**
        (`extra_content.google.thought_signature`) ekler ve sonraki istekte
        bunun aynen gönderilmesini bekler. Ancak imza konuşma bağlamına
        bağlıdır: geçmiş bir turdan geliyorsa, model değiştiyse veya bağlam
        kısaldıysa geçersiz olur ve istek
        `HTTP 400 Function call is missing a thought_signature` ile reddedilir.
        Bu durumda imzaları temizleyip yeniden denemek, görevi kurtarır —
        çağrı geçmişi (hangi araçların çalıştığı) korunur, yalnızca imza atılır.

        Returns:
            Temizlenen imza sayısı
        """
        stripped = 0
        for message in payload.get("messages", []):
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and "extra_content" in call:
                    call.pop("extra_content", None)
                    stripped += 1
        return stripped

    @staticmethod
    def _drop_last_tool_turn(payload: Dict[str, Any]) -> bool:
        """
        Sondaki asistan araç turunu ve sonuçlarını istekten çıkarır.

        İmza temizliği yetmezse başvurulan son çare: model, o ana kadar
        yaptıklarını özetleyerek devam eder. Araç etkileri zaten uygulanmıştır
        (dosyalar yazıldı, komutlar çalıştı); yalnızca konuşma bağlamından
        düşer.

        Returns:
            Bir tur çıkarıldıysa True
        """
        messages = payload.get("messages", [])
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "assistant" and messages[index].get("tool_calls"):
                del messages[index:]
                return True
        return False

    def _post_json(self, url: str, payload: Dict[str, Any],
                   headers: Dict[str, str]) -> Tuple[bool, Dict[str, Any], str]:
        """
        JSON POST isteği yapar.

        `urllib` kullanılır — `requests` bağımlılığı yoktur ve bu çağrı
        ajanın kritik yolunda değildir.

        ÖNEMLİ — User-Agent zorunludur:
            Sağlayıcıların çoğu (Groq, Cloudflare korumalı API'ler) `urllib`'in
            varsayılan `Python-urllib/3.x` kimliğini **bot** sayıp isteği
            `HTTP 403 (error code: 1010)` ile reddeder. Bu, "anahtar geçersiz"
            gibi görünen ama aslında tarayıcı kimliği isteyen bir hatadır ve
            teşhisi zordur. Gerçek bir uygulama kimliği göndermek sorunu
            çözer.

        Returns:
            (başarılı, yanıt_gövdesi, hata_metni)
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json; charset=utf-8")
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", USER_AGENT)
        for key, value in headers.items():
            request.add_header(key, value)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
            return True, json.loads(raw), ""
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            hint = ""
            if exc.code == 401:
                hint = (" — API anahtarı geçersiz. Ayarlar'dan yeni bir anahtar "
                        "girin veya ortam değişkeni tanımlayın")
            elif exc.code == 402:
                hint = " — bakiye yetersiz"
            elif exc.code == 429:
                hint = " — hız sınırı; biraz bekleyip tekrar deneyin"
            elif exc.code in (500, 502, 503):
                hint = " — sağlayıcı geçici olarak kullanılamıyor"
            return False, {}, f"HTTP {exc.code}{hint}: {detail}"
        except urllib.error.URLError as exc:
            return False, {}, f"bağlantı hatası: {exc.reason}"
        except (TimeoutError, OSError) as exc:
            return False, {}, f"ağ hatası: {exc}"
        except json.JSONDecodeError as exc:
            return False, {}, f"geçersiz JSON yanıtı: {exc}"

    def _repair_tool_messages(self, payload: Dict[str, Any]) -> int:
        """
        Yanıtsız kalan araç çağrılarını onarır.

        ⚠ Ölçülen hata: Sağlayıcı, `tool_calls` içeren bir asistan
        mesajından sonra **her** `tool_call_id` için bir `tool` mesajı
        bekler. Ajan bir turda **birden fazla** araç çağırdığında (yeni
        model `deepseek-chat` paralel çağrı yapıyor) ve tur yarıda
        kaldığında (bağlam taşması, hata, iptal) bazı çağrılar yanıtsız
        kalır. Sonraki istek reddedilir:

            HTTP 400: An assistant message with 'tool_calls' must be
            followed by tool messages responding to each 'tool_call_id'.
            (insufficient tool messages following tool_calls message)

        ⚠ Bu durum **kalıcıdır**: bozuk geçmiş temizlenmediği sürece her
        sonraki istek de reddedilir — oturum tamamen kilitlenir.

        Çözüm: eksik her `tool_call_id` için **sentetik** bir `tool`
        yanıtı eklenir. Böylece:

            * İstek geçerli olur
            * Çağrı geçmişi **korunur** (model ne denediğini görür)
            * Model, aracın çalışmadığını bilir ve **tekrar dener**

        Returns:
            Eklenen sentetik yanıt sayısı
        """
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return 0

        eklenen = 0
        i = 0
        while i < len(messages):
            mesaj = messages[i]
            if not isinstance(mesaj, dict) or mesaj.get("role") != "assistant":
                i += 1
                continue
            cagrilar = mesaj.get("tool_calls") or []
            if not cagrilar:
                i += 1
                continue

            # Bu asistan mesajını izleyen tool mesajlarını topla
            yanitlanan: set = set()
            j = i + 1
            while j < len(messages):
                sonraki = messages[j]
                if not isinstance(sonraki, dict):
                    break
                if sonraki.get("role") != "tool":
                    break
                yanitlanan.add(str(sonraki.get("tool_call_id") or ""))
                j += 1

            # Eksik olanlar için sentetik yanıt ekle
            eksikler = []
            for cagri in cagrilar:
                if not isinstance(cagri, dict):
                    continue
                kimlik = str(cagri.get("id") or "")
                if kimlik and kimlik not in yanitlanan:
                    eksikler.append(kimlik)
            if eksikler:
                yeni = [
                    {"role": "tool", "tool_call_id": kimlik,
                     "content": ("[araç yanıtı yok — önceki tur tamamlanmadan "
                                 "kesildi; gerekirse tekrar çağır]")}
                    for kimlik in eksikler
                ]
                messages[j:j] = yeni
                eklenen += len(yeni)
                i = j + len(yeni)
                continue
            i = j

        if eklenen:
            logger.info(
                f"Araç geçmişi onarıldı: {eklenen} eksik yanıt tamamlandı "
                f"(sağlayıcı her tool_call_id için yanıt bekler)"
            )
        return eklenen

    def _record(self, response: LLMResponse) -> None:
        """İstatistikleri günceller."""
        self._counter["calls"] += 1
        self._counter["total_ms"] += response.duration_ms
        self._counter["prompt_tokens"] += response.prompt_tokens
        self._counter["completion_tokens"] += response.completion_tokens
        if not response.ok:
            self._counter["errors"] += 1
            # Son hata saklanır: kota denetimi (`provider_quota_exhausted`)
            # ve arayüzdeki "kota doldu" uyarısı buna bakar.
            self._counter["last_error"] = (response.error or "")[:300]
            self._counter["last_error_at"] = time.time()
        else:
            self._counter["last_error"] = ""

    @staticmethod
    def _is_permanently_unavailable(error: str) -> bool:
        """
        Model **kalıcı olarak** kullanılamıyor mu (kota değil)?

        Ayrım kritiktir:
            * Kalıcı (`404 no longer available`) → modeli değiştir ve kaydet.
            * Geçici (`429 kota doldu`) → **modeli değiştirme**. Kullanıcının
              açıkça seçtiği modeli geçici bir kota yüzünden ezmek yanlıştır;
              kota yarın sıfırlanır ama kullanıcının seçimi kaybolur.
        """
        text = (error or "").lower()
        if "429" in text or "quota" in text or "exceeded" in text \
                or "rate limit" in text or "high demand" in text or "503" in text:
            return False
        return any(marker in text for marker in
                   ("404", "no longer available", "does not exist",
                    "not found", "deprecated", "unsupported model"))

    def _alternative_model(self, response_body: Dict[str, Any],
                           error: str) -> str:
        """
        Kullanılabilecek alternatif model adını bulur.

        Sıra:
            1. Sağlayıcının hata mesajında **önerdiği** ad
               (Google: "use models/gemini-3.8-flash") — ama daha önce
               başarısız olmuşsa atlanır
            2. Bu sağlayıcı için tanımlı yedek zincirindeki sıradaki
               **ölü olmayan** model
            3. Boş (alternatif yok)
        """
        # 1) Sağlayıcının önerisi
        match = _SUGGESTED_MODEL_RE.search(error or "")
        if match:
            suggested = match.group(1)
            if suggested != self.model and suggested not in self._dead_models:
                return suggested

        # 2) Yedek zinciri (ölü modelleri atla)
        chain = SUGGESTED_MODELS.get(self.service, [])
        for candidate in chain:
            if candidate != self.model and candidate not in self._dead_models:
                return candidate
        return ""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} model={self.model} available={self.available}>"


# ─── OpenAI Uyumlu Sağlayıcı ────────────────────────────────────────────────

#: Bilinen OpenAI-uyumlu servisler (varsayılan adres ve model)
#:
#: Model adları **sağlayıcının güncel kataloğundan** seçilmelidir; eskimiş bir
#: ad, anahtar geçerli olsa bile `HTTP 404 (model does not exist)` verir ve
#: bu, "bağlantı kurulamıyor" gibi görünür.
KNOWN_SERVICES: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        # Model seçimi — ÖLÇÜLDÜ (2026-10-07):
        #
        #   deepseek-chat     : düşünme token'ı **0** → bütçe tamamen
        #                       görünür yanıta kalır. Araç çağrısında
        #                       güvenilir. **VARSAYILAN.**
        #   deepseek-flash    : 2.885 düşünme token'ı harcıyor; büyük
        #                       bağlamda bütçeyi bitirip **boş yanıt**
        #                       dönebiliyor.
        #   deepseek-v4-pro   : 4.560 düşünme token'ı, görünür içerik 0.
        #                       Araç ağırlıklı görevler için **uygun değil**.
        #
        # ⚠ Ölçülen hata: `deepseek-flash` ile ajan "model 3 kez boş yanıt
        # verdi (bitiş nedeni: length, token: 8192)" diyerek durdu. Sebep
        # düşünme token'larının 8192 bütçesini tüketmesiydi.
        "model": "deepseek-chat",
        "env": "DEEPSEEK_API_KEY",
        "label": "DeepSeek",
    },
    "gemini": {
        # Google, OpenAI uyumlu bir uç nokta sunar; bu sayede aynı istemci
        # kodu değişmeden çalışır ve araç çağrısı desteklenir.
        # Not: `gemini-2.0-flash` Google tarafından kapatıldı; güncel model
        # adları `SUGGESTED_MODELS` içinde yedek zinciriyle listelenir.
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.6-flash",
        "env": "GEMINI_API_KEY",
        "label": "Google Gemini",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env": "OPENAI_API_KEY",
        "label": "OpenAI",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        # Not: `llama-3.3-70b-versatile` Groq kataloğundan kaldırıldı.
        # Güncel araç-çağrısı destekleyen modeller listelenmiştir.
        "model": "openai/gpt-oss-20b",
        "env": "GROQ_API_KEY",
        "label": "Groq",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
        "env": "OPENROUTER_API_KEY",
        "label": "OpenRouter",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "model": "deepseek-ai/DeepSeek-V3",
        "env": "TOGETHER_API_KEY",
        "label": "Together AI",
    },
    "lmstudio": {
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "local-model",
        "env": "",
        "label": "LM Studio (yerel)",
    },
    "vllm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "local-model",
        "env": "",
        "label": "vLLM (yerel)",
    },
    "llamacpp": {
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "local-model",
        "env": "",
        "label": "llama.cpp sunucusu (yerel)",
    },
}

#: Sağlayıcı başına önerilen modeller (arayüzde liste olarak gösterilir).
#:
#: İlk sıradaki model varsayılan kabul edilir; sonrakiler **yedek** zinciridir.
#: Sağlayıcılar modelleri sık değiştirir (Google, `gemini-2.0-flash` için
#: "artık kullanılamıyor, `gemini-3.8-flash` kullanın" mesajı döndürüyor) ve
#: yeni modeller yoğunluk nedeniyle geçici olarak `503` verebilir. Yedek
#: zinciri, tek bir model adına bağımlılığı ortadan kaldırır.
SUGGESTED_MODELS: Dict[str, List[str]] = {
    "gemini": [
        # Sıra ölçülerek belirlendi: `3.8-flash` yeni olduğu için sık sık
        # `503 (yoğunluk)` veriyor, `3.6-flash` ise günlük ücretsiz kotayı
        # çabuk dolduruyor. Bu yüzden önce **istikrarlı çalışan** modeller
        # denenir; yeni modeller yedekte kalır.
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
    ],
    "groq": [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    ],
    "deepseek": ["deepseek-chat", "deepseek-v4-pro", "deepseek-flash"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    "openrouter": [
        "deepseek/deepseek-chat",
        "google/gemini-2.5-flash",
        "meta-llama/llama-3.3-70b-instruct",
    ],
    "together": ["deepseek-ai/DeepSeek-V3", "meta-llama/Llama-3.3-70B-Instruct-Turbo"],
}

#: Kullanımdan kaldırılmış model adı → yerine kullanılacak ad.
#:
#: Sağlayıcılar modellerini kaldırdığında kayıtlı kimlik dosyasındaki eski ad
#: `HTTP 404 (model does not exist)` hatası verir. Kullanıcı bunu "anahtar
#: geçersiz" sanır. Bu eşleme, kayıtlı adı sessizce güncel olanla değiştirir.
DEPRECATED_MODELS: Dict[str, str] = {
    # Groq, Llama 3.3 serisini kataloğundan kaldırdı
    "llama-3.3-70b-versatile": "openai/gpt-oss-20b",
    "llama-3.3-70b-specdec": "openai/gpt-oss-20b",
    "llama-3.1-70b-versatile": "openai/gpt-oss-20b",
    "llama-3.1-8b-instant": "openai/gpt-oss-20b",
    "mixtral-8x7b-32768": "openai/gpt-oss-20b",
    "gemma2-9b-it": "openai/gpt-oss-20b",
    # Gemini: 2.0 ve 1.5 serisi kapatıldı (Google'ın kendi uyarı mesajı)
    "gemini-2.0-flash": "gemini-3.6-flash",
    "gemini-2.0-flash-lite": "gemini-3.6-flash",
    "gemini-1.5-flash": "gemini-3.6-flash",
    "gemini-1.5-pro": "gemini-3.6-flash",
    "gemini-pro": "gemini-3.6-flash",
    "gemini-2.5-pro": "gemini-3.6-flash",
    "gemini-2.5-flash-lite": "gemini-2.5-flash",
    # DeepSeek modelleri — ÖLÇÜLDÜ (2026-10-07), araç ağırlıklı iş için:
    #
    #   deepseek-chat     : düşünme token'ı **0** → 8192 bütçesinin tamamı
    #                       görünür yanıt ve araç çağrısına kalır. Hızlı
    #                       (7,8 sn) ve güvenilir. **TERCİH EDİLEN.**
    #   deepseek-flash    : 2.885 düşünme token'ı harcar; bağlam büyüyünce
    #                       bütçeyi bitirip **boş yanıt** döner.
    #   deepseek-v4-pro   : 4.560 düşünme token'ı, görünür içerik 0 →
    #                       araç ağırlıklı görevlerde **kullanılamaz**.
    #
    # ⚠ Ölçülen hata: ajan `deepseek-flash` ile "model 3 kez boş yanıt
    # verdi (bitiş nedeni: length, token: 8192)" diyerek durdu. Kök neden
    # düşünme token'larının bütçeyi tüketmesiydi.
    #
    # ⚠ `deepseek-chat`'i buraya "deprecated" diye yazmak, kullanıcının
    # bilinçli seçimini **sessizce geri çeviriyordu** (ölçüldü: ayar
    # kaydedilmesine rağmen model `deepseek-flash`'e dönüyordu).
    "deepseek-v3": "deepseek-chat",
    "deepseek-v3.1": "deepseek-chat",
    "deepseek-v3.2": "deepseek-chat",
    "deepseek-coder": "deepseek-chat",
    "deepseek-flash": "deepseek-chat",
    "deepseek-reasoner": "deepseek-v4-pro",
    "deepseek-r1": "deepseek-v4-pro",
}

#: Yanıt gövdesinden önerilen model adını çıkaran kalıp.
#:
#: Google, kullanımdan kaldırılan bir model için şu mesajı döndürür:
#: `"... is no longer available. Please update your code to use
#: models/gemini-3.8-flash ..."` — bu adı otomatik yakalamak, kullanıcıya
#: "model bulunamadı" demekten çok daha yararlıdır.
_SUGGESTED_MODEL_RE = re.compile(
    r"use\s+models?/([A-Za-z0-9._\-]+)", re.IGNORECASE
)

#: Yalnızca varlığı eskimiş sayılan adlar (sağlayıcının varsayılanına düşer)
_STALE_MODEL_NAMES: Dict[str, str] = {}

class OpenAICompatibleProvider(LLMProvider):
    """
    `/v1/chat/completions` konuşan her servis için sağlayıcı.

    Araç çağrısı (function calling) desteklenir; bu, ajanın **güvenilir**
    biçimde araç seçmesini sağlar. Model araç çağrısını desteklemiyorsa
    ajan JSON istemi moduna düşer (bkz. `agent.loop`).
    """

    name = "openai_compatible"
    supports_tools = True

    def __init__(self, service: str = "deepseek", model: str = "",
                 api_key: str = "", base_url: str = "", **kwargs: Any) -> None:
        spec = KNOWN_SERVICES.get(service, {})
        self.service = service
        self.label = spec.get("label", service)
        super().__init__(
            model=model or spec.get("model", ""),
            api_key=api_key or (os.environ.get(spec.get("env", ""), "")
                                if spec.get("env") else ""),
            base_url=base_url or spec.get("base_url", ""),
            **kwargs,
        )

    @property
    def available(self) -> bool:
        """
        Sağlayıcı kullanılabilir mi?

        Bulut servisleri için API anahtarı yeterlidir. **Yerel** servisler
        (LM Studio, vLLM, llama.cpp) anahtar istemez; bu yüzden gerçekten
        dinleyip dinlemedikleri bir TCP denemesiyle doğrulanır — aksi hâlde
        ajan çalışmayan bir sunucuyu "hazır" sanıp her turda bağlantı hatası
        alırdı.
        """
        if not self.base_url:
            return False
        if self.api_key:
            return True
        parsed_host = urllib.parse.urlparse(self.base_url).hostname or ""
        if parsed_host in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
            return tcp_reachable(self.base_url)
        # Uzak adres ama anahtar yok: kullanılamaz say
        return False

    def chat(self, messages: Sequence[LLMMessage],
             tools: Optional[Sequence[Dict[str, Any]]] = None,
             **kwargs: Any) -> LLMResponse:
        """Sohbet tamamlama isteği gönderir."""
        started = time.perf_counter()
        response = LLMResponse(provider=self.label or self.name, model=self.model)

        if not self.available:
            response.error = (
                f"{self.label} kullanılamıyor — API anahtarı yok"
                + (f" ({KNOWN_SERVICES.get(self.service, {}).get('env', '')})"
                   if KNOWN_SERVICES.get(self.service, {}).get("env") else "")
            )
            return response

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Sağlayıcıya özgü alanları isteğe uygun hâle getir
        # (DeepSeek düşünme modu `reasoning_content` bekler)
        self._normalize_payload(payload)
        # Yanıtsız kalan araç çağrılarını tamamla (sağlayıcı her
        # `tool_call_id` için bir `tool` mesajı bekler; eksikse istek
        # reddedilir ve **oturum kilitlenir**)
        self._repair_tool_messages(payload)

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        ok, body, error = self._post_json(
            f"{self.base_url}/chat/completions", payload, headers
        )
        response.duration_ms = (time.perf_counter() - started) * 1000

        # ── Model kullanımdan kalktı / geçici olarak kullanılamıyor ──
        # Sağlayıcılar modelleri **kalıcı olarak** kapatır
        # (`404 no longer available`). Kullanıcıya hata göstermek yerine yedek
        # zincirdeki sıradaki model denenir; sağlayıcı kendi önerdiği adı
        # mesajda belirtiyorsa o ad önceliklidir.
        #
        # Kota/yoğunluk (`429`/`503`) buraya **girmez**: geçici bir durum
        # yüzünden kullanıcının açıkça seçtiği modeli değiştirmek yanlıştır.
        if not ok and self._is_permanently_unavailable(error):
            alternative = self._alternative_model(body, error)
            # Kullanılamayan modelleri hatırla: aynı modeli her istekte
            # yeniden denemek, kullanıcıyı dakikalarca bekletir.
            if self.model:
                self._dead_models.add(self.model)
            if alternative and alternative != self.model:
                logger.warning(
                    f"{self.label}: '{self.model}' kullanılamıyor "
                    f"({error[:60]}) — '{alternative}' deneniyor"
                )
                previous_model = self.model
                self.model = alternative
                payload["model"] = alternative
                ok, body, error = self._post_json(
                    f"{self.base_url}/chat/completions", payload, headers
                )
                if not ok:
                    # Deneme başarısız: orijinal adı koru ki sonraki çağrı
                    # yine yedek zincirini denesin.
                    self.model = previous_model

        # ── Düşünme imzası hatası: temizleyip yeniden dene ──
        # Gemini'nin imzası konuşma bağlamına bağlıdır; geçmiş bir turdan
        # geldiğinde veya model değiştiğinde geçersiz olur. İmzaları
        # temizlemek çağrı geçmişini korur, yalnızca imzayı atar.
        if not ok and self._is_thought_signature_error(error):
            stripped = self._strip_thought_signatures(payload)
            if stripped:
                logger.warning(
                    f"{self.label}: düşünme imzası reddedildi — {stripped} imza "
                    "temizlenip yeniden denenecek"
                )
                ok, body, error = self._post_json(
                    f"{self.base_url}/chat/completions", payload, headers
                )

        # ── Hâlâ reddediliyorsa: son araç turunu düşür ──
        if not ok and self._is_thought_signature_error(error):
            if self._drop_last_tool_turn(payload):
                logger.warning(
                    f"{self.label}: imza sorunu sürüyor — son araç turu "
                    "bağlamdan çıkarılıp yeniden denenecek"
                )
                ok, body, error = self._post_json(
                    f"{self.base_url}/chat/completions", payload, headers
                )

        response.model = response.model or self.model
        if not ok:
            response.error = error
            self._record(response)
            logger.warning(f"{self.label} çağrısı başarısız: {error[:200]}")
            return response

        try:
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            response.text = (message.get("content") or "").strip()
            response.tool_calls = list(message.get("tool_calls") or [])
            # Ham mesajı sakla: sağlayıcıya özgü alanlar (Gemini'nin
            # `thought_signature`'ı gibi) geri gönderilirken korunmalı.
            response.raw_message = dict(message)
            response.finish_reason = choice.get("finish_reason", "")
            response.model = body.get("model", self.model)
            usage = body.get("usage") or {}
            response.prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            response.completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            response.total_tokens = int(usage.get("total_tokens", 0) or 0)
            response.raw = body
            response.ok = True
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            response.error = f"yanıt ayrıştırılamadı: {exc}"
            logger.warning(f"{self.label} yanıtı ayrıştırılamadı: {exc}")

        self._record(response)
        return response

    def health(self) -> Dict[str, Any]:
        """Sağlayıcı sağlığı (gerçek bir istek yapmaz — hızlı kontrol)."""
        info = super().health()
        info.update({
            "service": self.service,
            "label": self.label,
            "env_var": KNOWN_SERVICES.get(self.service, {}).get("env", ""),
        })
        return info


# ─── Ollama Sağlayıcı ───────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """
    Ollama yerel sunucusu (`/api/chat`).

    Araç çağrısı Ollama'da model bazlıdır; destekleyen modellerde çalışır,
    aksi hâlde ajan JSON istemi moduna düşer.
    """

    name = "ollama"
    supports_tools = True

    #: Sunucu erişilebilirlik önbelleği (zaman, sonuç) — sınıf düzeyinde
    _probe_cache: Optional[Tuple[float, bool]] = None

    def __init__(self, model: str = "llama3.1", base_url: str = "http://127.0.0.1:11434",
                 **kwargs: Any) -> None:
        super().__init__(model=model, base_url=base_url, **kwargs)

    @property
    def available(self) -> bool:
        """Sunucu ayakta mı? (hızlı, önbellekli deneme)"""
        return self._probe()

    def _probe(self) -> bool:
        """Sunucuya kısa bir istek atar (5 sn önbellekli)."""
        cached = self._probe_cache
        now = time.time()
        if cached is not None and now - cached[0] < 5.0:
            return cached[1]
        result = False
        try:
            request = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(request, timeout=2.0) as response:
                result = response.status == 200
        except Exception:
            result = False
        self._probe_cache = (now, result)
        return result

    def list_models(self) -> List[str]:
        """Sunucudaki modeller."""
        try:
            request = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(request, timeout=5.0) as response:
                body = json.loads(response.read().decode("utf-8", "replace"))
            return [item.get("name", "") for item in body.get("models", [])]
        except Exception:
            return []

    def chat(self, messages: Sequence[LLMMessage],
             tools: Optional[Sequence[Dict[str, Any]]] = None,
             **kwargs: Any) -> LLMResponse:
        """Ollama sohbet isteği."""
        started = time.perf_counter()
        response = LLMResponse(provider=self.name, model=self.model)

        if not self._probe():
            response.error = (
                f"Ollama sunucusu yanıt vermiyor ({self.base_url}) — "
                "`ollama serve` çalışıyor mu?"
            )
            return response

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        if tools:
            payload["tools"] = list(tools)

        ok, body, error = self._post_json(f"{self.base_url}/api/chat", payload, {})
        response.duration_ms = (time.perf_counter() - started) * 1000

        if not ok:
            response.error = error
            self._record(response)
            return response

        try:
            message = body.get("message") or {}
            response.text = (message.get("content") or "").strip()
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                response.tool_calls.append({
                    "id": call.get("id") or f"call_{len(response.tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": json.dumps(function.get("arguments") or {},
                                                ensure_ascii=False)
                        if not isinstance(function.get("arguments"), str)
                        else function.get("arguments", "{}"),
                    },
                })
            response.prompt_tokens = int(body.get("prompt_eval_count", 0) or 0)
            response.completion_tokens = int(body.get("eval_count", 0) or 0)
            response.total_tokens = response.prompt_tokens + response.completion_tokens
            response.finish_reason = body.get("done_reason", "")
            response.ok = True
        except (KeyError, TypeError, ValueError) as exc:
            response.error = f"yanıt ayrıştırılamadı: {exc}"

        self._record(response)
        return response

    def health(self) -> Dict[str, Any]:
        """Ollama sağlığı ve model listesi."""
        info = super().health()
        info["models"] = self.list_models()[:20]
        return info


# ─── NeuroCore Kendi Modeli ─────────────────────────────────────────────────

class NeuroCoreProvider(LLMProvider):
    """
    NeuroCore'un kendi transformer'ını kullanır (deneysel).

    Dürüst uyarı: 208M parametreli, bu makinede eğitilmiş bir model
    **güvenilir araç çağrısı yapamaz**. Bu sağlayıcı yalnızca:
        * çevrimdışı/tamamen yerel çalışma isteği için
        * model büyüdükçe (v0.2+) gerçek kullanım için
    vardır. Ajan döngüsü, bu sağlayıcıyla araç çağrısı başarısız olursa
    kural tabanlı planlayıcıya düşer.
    """

    name = "neurocore"
    supports_tools = False

    def __init__(self, engine: Any = None, model: str = "neurocore-local",
                 **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._engine = engine

    @property
    def available(self) -> bool:
        """Motor bağlı ve üretim yapabiliyor mu?"""
        if self._engine is None:
            return False
        try:
            model = getattr(self._engine, "model", None)
            return bool(model is not None and getattr(model, "can_generate", False))
        except Exception:
            return False

    def attach(self, engine: Any) -> None:
        """Motoru bağlar."""
        self._engine = engine

    def chat(self, messages: Sequence[LLMMessage],
             tools: Optional[Sequence[LLMMessage]] = None,
             **kwargs: Any) -> LLMResponse:
        """Kendi modelimizle metin üretir (araç şeması desteklenmez)."""
        started = time.perf_counter()
        response = LLMResponse(provider=self.name, model=self.model)

        if not self.available:
            response.error = (
                "NeuroCore modeli henüz üretim yapamıyor (yeterli eğitim adımı yok)"
            )
            return response

        prompt = "\n".join(
            f"{message.role}: {message.content}" for message in messages if message.content
        )
        try:
            text = self._engine.model.generate_text(prompt, max_new_tokens=512)
            response.text = (text or "").strip()
            response.ok = bool(response.text)
            if not response.ok:
                response.error = "model boş yanıt üretti"
        except Exception as exc:
            response.error = f"üretim hatası: {exc}"

        response.duration_ms = (time.perf_counter() - started) * 1000
        self._record(response)
        return response

    def health(self) -> Dict[str, Any]:
        """Motor durumu."""
        info = super().health()
        if self._engine is not None:
            try:
                info["readiness"] = self._engine.model.generation_readiness()
            except Exception:
                pass
        info["note"] = (
            "kendi modelimiz araç çağrısı yapamaz; ajan kural tabanlı moda düşer"
        )
        return info


# ─── Sağlayıcı Yöneticisi ───────────────────────────────────────────────────

class LLMRouter:
    """
    Sağlayıcıları yönetir: keşif, seçim, yedekleme, anahtar kaydetme.

    Kullanım:
        router = LLMRouter()
        router.initialize(config=config, data_path="data")
        print(router.status())

        response = router.chat([LLMMessage("user", "merhaba")])
        if not response.ok:
            print("sağlayıcı yok:", response.error)

        router.save_credentials("deepseek", "sk-...")   # arayüzden
    """

    name = "agent.llm"

    #: Denenecek hizmet sırası (anahtar bulunan ilk hizmet kazanır)
    DISCOVERY_ORDER: Tuple[str, ...] = (
        "deepseek", "gemini", "groq", "openai", "openrouter", "together",
        "lmstudio", "vllm", "llamacpp",
    )

    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}
        self._active: Optional[LLMProvider] = None
        self._preferred: str = ""
        self._config: Any = None
        self._data_path: Optional[Path] = None
        self._credential_file: Optional[Path] = None
        self._initialized = False
        self._last_error = ""
        self._history: List[Dict[str, Any]] = []
        #: Son okunan kimlik dosyası (her `status()` çağrısında disk okumamak için)
        self._stored_credentials: Dict[str, Any] = {}
        self._stored_keys: List[str] = []
        #: Kullanımdan kalkmış model adı güncellendi mi? (diske yazılacak)
        self._credentials_migrated = False

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, data_path: str = "",
                   engine: Any = None) -> None:
        """
        Sağlayıcıları keşfeder ve aktif olanı seçer.

        Args:
            config: SystemConfig (isteğe bağlı `llm` bölümü okunur)
            data_path: Kimlik bilgisi dosyasının dizini
            engine: NeuroCore motoru (yerel sağlayıcı için)
        """
        if self._initialized:
            return
        self._config = config

        base = data_path or getattr(config, "data_path", "") or "data"
        self._data_path = Path(base)
        try:
            self._data_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._data_path = Path.cwd()
        self._credential_file = self._data_path / "llm_credentials.json"

        stored = self._load_credentials()
        self._stored_credentials = stored
        self._stored_keys = sorted(stored.get("keys", {}).keys())
        llm_config = getattr(config, "llm", None)

        # Tercih: config > kayıtlı dosya > ortam değişkeni
        preferred = ""
        if llm_config is not None:
            preferred = str(getattr(llm_config, "provider", "") or "")
        self._preferred = preferred or str(stored.get("provider", "") or "")

        for service in self.DISCOVERY_ORDER:
            spec = KNOWN_SERVICES[service]
            api_key = ""
            if llm_config is not None and \
                    str(getattr(llm_config, "provider", "")) == service:
                api_key = str(getattr(llm_config, "api_key", "") or "")
            if not api_key:
                api_key = str(stored.get("keys", {}).get(service, "") or "")
            if not api_key and spec.get("env"):
                api_key = os.environ.get(spec["env"], "")

            base_url = ""
            if llm_config is not None and \
                    str(getattr(llm_config, "provider", "")) == service:
                base_url = str(getattr(llm_config, "base_url", "") or "")
            if not base_url:
                base_url = str(stored.get("base_urls", {}).get(service, "") or "")

            model = ""
            if llm_config is not None and \
                    str(getattr(llm_config, "provider", "")) == service:
                model = str(getattr(llm_config, "model", "") or "")
            if not model:
                model = str(stored.get("models", {}).get(service, "") or "")
            # Kullanımdan kalkmış model adlarını güncelle: sağlayıcılar
            # modellerini kaldırır ve eski ad `HTTP 404 (model does not exist)`
            # verir — bu, "anahtar geçersiz" gibi görünen yanıltıcı bir hatadır.
            replacement = DEPRECATED_MODELS.get(model, "")
            if replacement:
                logger.warning(
                    f"{service}: '{model}' artık kullanılamıyor — "
                    f"'{replacement}' kullanılacak"
                )
                model = replacement
                stored.setdefault("models", {})[service] = replacement
                self._credentials_migrated = True

            provider = OpenAICompatibleProvider(
                service=service, model=model, api_key=api_key, base_url=base_url,
                temperature=float(getattr(llm_config, "temperature", 0.2) or 0.2),
                max_tokens=int(getattr(llm_config, "max_tokens", 8192) or 8192),
                timeout=float(getattr(llm_config, "timeout", 180.0) or 180.0),
            )
            self._providers[service] = provider

        # Yerel model (her zaman kayıtlı, kullanılabilirliği koşullu)
        ollama = OllamaProvider(
            model=str(stored.get("models", {}).get("ollama", "") or "llama3.1"),
            base_url=str(stored.get("base_urls", {}).get("ollama", "")
                         or "http://127.0.0.1:11434"),
        )
        self._providers["ollama"] = ollama

        local = NeuroCoreProvider(engine=engine)
        self._providers["neurocore"] = local

        self._initialized = True
        self._stored_credentials = stored
        self._stored_keys = sorted(stored.get("keys", {}).keys())
        # Model göçü yapıldıysa diske yaz: aksi hâlde her açılışta aynı uyarı
        # tekrarlanır ve kullanıcı nedenini göremez.
        if self._credentials_migrated:
            self._persist_credentials(stored)
        self._active = self._select()

        # Model adını **şimdi** doğrula: yeni bir model yoğunluk nedeniyle
        # geçici olarak `503` verebilir ve bunu ilk görev sırasında keşfetmek,
        # kullanıcıya yarıda kesilmiş bir çalışma gösterir. Başlangıçta
        # doğrulamak, sağlam bir modelle başlamayı garanti eder.
        self._validate_active_model()
        if self._active is not None:
            logger.info(
                f"Dil modeli sağlayıcısı: {self._active.name} "
                f"({self._active.model}) — araç çağrısı: "
                f"{'var' if self._active.supports_tools else 'yok'}"
            )
        else:
            logger.warning(
                "Dil modeli sağlayıcısı bulunamadı — ajan kural tabanlı modda "
                "çalışacak. Ayarlar'dan bir API anahtarı girin veya Ollama başlatın."
            )

    def _validate_active_model(self) -> None:
        """
        Aktif sağlayıcının model adını **hafif bir istekle** doğrular.

        Neden gerekli? Sağlayıcılar modelleri sık değiştirir ve yeni modeller
        yoğunluk nedeniyle geçici olarak `503` döndürür. Bunu ilk görev
        sırasında keşfetmek, kullanıcıya yarıda kesilmiş bir çalışma gösterir.
        Başlangıçta doğrulayıp çalışan bir modele geçmek bu sorunu ortadan
        kaldırır.

        Doğrulama başarısız olursa **sessizce** yedek zincir denenir; hiçbiri
        çalışmazsa mevcut model korunur (kullanıcı gerçek hatayı görsün).
        """
        provider = self._active
        if provider is None or not hasattr(provider, "service"):
            return
        if not provider.available:
            return

        chain = SUGGESTED_MODELS.get(getattr(provider, "service", ""), [])
        if not chain:
            return

        original = provider.model
        candidates = [original] + [name for name in chain if name != original]

        for index, candidate in enumerate(candidates):
            provider.model = candidate
            try:
                response = provider.chat(
                    [LLMMessage("user", "ping")], max_tokens=16, temperature=0.0
                )
            except Exception as exc:
                logger.debug(f"Model doğrulaması hata verdi ({candidate}): {exc}")
                continue

            if response.ok:
                if candidate != original:
                    logger.info(
                        f"Model doğrulandı: '{candidate}' "
                        f"(önceki '{original}' artık kullanılamıyor)"
                    )
                    self._remember_model(getattr(provider, "service", ""), candidate)
                else:
                    logger.info(f"Model doğrulandı: '{candidate}'")
                return

            # Kalıcı değil (kota/yoğunluk) → kullanıcının seçimini KORU
            if not provider._is_permanently_unavailable(response.error):
                logger.warning(
                    f"Model '{candidate}' geçici olarak kullanılamıyor "
                    f"({response.error[:70]}) — seçim korunuyor, kotayı "
                    "doldurmasın diye doğrulama durduruldu"
                )
                provider.model = original
                return

            logger.warning(
                f"Model '{candidate}' artık kullanılamıyor: {response.error[:80]}"
            )
            provider._dead_models.add(candidate)

        provider.model = original

    def _remember_model(self, service: str, model: str) -> None:
        """Çalıştığı doğrulanan modeli kimlik dosyasına yazar."""
        if not service:
            return
        stored = self._stored_credentials or self._load_credentials()
        stored.setdefault("models", {})[service] = model
        if self._persist_credentials(stored):
            self._stored_credentials = stored
            logger.info(f"Çalışan model kaydedildi: {service} → {model}")

    def chat_with_budget(self, messages: Sequence["LLMMessage"],
                         *, tools: Optional[List[Dict[str, Any]]] = None,
                         provider: str = "",
                         token_ladder: Sequence[int] = (1024, 3072, 8192),
                         **kwargs: Any) -> "LLMResponse":
        """
        Çağrıyı **artan token bütçesiyle** yapar.

        Neden gerekli? "Düşünme" modelleri (deepseek-flash, deepseek-reasoner,
        gpt-oss) düşünme token'larını **aynı** bütçeden harcar. Küçük bir
        sınırda yanıt tamamen boş döner ve `finish_reason` `"length"` olur —
        ölçüldü: 700 token'lık sınırda `deepseek-flash` 913 token harcayıp
        **hiç metin üretmedi**. Bu metot, boş yanıt aldığında bütçeyi artırıp
        yeniden dener.

        Args:
            messages: Sohbet mesajları
            tools: Araç şemaları
            provider: Sağlayıcı adı (boş = aktif)
            token_ladder: Denenecek token sınırları (artan)
            **kwargs: `chat()` için ek argümanlar

        Returns:
            En başarılı `LLMResponse` (hepsi boşsa sonuncusu)
        """
        last: Optional[LLMResponse] = None
        for index, budget in enumerate(token_ladder):
            response = self.chat(messages, tools=tools, provider=provider,
                                 max_tokens=int(budget), **kwargs)
            last = response
            if not response.ok:
                return response
            if response.text or response.tool_calls:
                return response
            # Boş yanıt: bütçe düşünmede tükendi mi?
            if response.finish_reason != "length" or index == len(token_ladder) - 1:
                return response
            logger.info(
                f"Yanıt boş (bitiş: length, {response.completion_tokens} token) — "
                f"bütçe {budget} → {token_ladder[index + 1]} olarak artırılıyor"
            )
        return last if last is not None else LLMResponse(
            ok=False, error="hiçbir bütçe denenemedi"
        )

    def failover(self, reason: str = "") -> Optional[str]:
        """
        Aktif sağlayıcı kullanılamazsa **çalışan başka bir sağlayıcıya** geçer.

        Neden gerekli? Ücretsiz katmanların günlük kotası dolar (`HTTP 429`)
        ve kullanıcı elle geçiş yapana kadar ajan hiçbir iş yapamaz. Birden
        fazla anahtar varsa en hızlı çözüm otomatik geçiştir.

        Args:
            reason: Geçiş gerekçesi (günlüğe yazılır)

        Returns:
            Geçilen sağlayıcının adı; geçiş yapılamadıysa None
        """
        current = getattr(self._active, "service", "") if self._active else ""
        # Aynı sağlayıcıya iki kez geçmeye çalışma (sonsuz döngü koruması)
        tried = getattr(self, "_failover_tried", set())
        tried.add(current)
        self._failover_tried = tried

        for name in self.DISCOVERY_ORDER:
            if name in tried:
                continue
            provider = self._providers.get(name)
            if provider is None:
                continue
            try:
                if not provider.available:
                    continue
            except Exception:
                continue
            logger.warning(
                f"Sağlayıcı değiştiriliyor: {current or 'yok'} → {name} "
                f"({(reason or 'kullanılamıyor')[:90]})"
            )
            self._preferred = name
            self._active = provider
            # Tercihi kalıcı hâle getir (yeniden başlatmada da geçerli olsun)
            stored = self._stored_credentials or self._load_credentials()
            stored["provider"] = name
            if self._persist_credentials(stored):
                self._stored_credentials = stored
            self._emit_provider_change(current, name, reason)
            return name
        logger.warning("Yedek sağlayıcı bulunamadı — hiçbiri kullanılabilir değil")
        return None

    def _emit_provider_change(self, old: str, new: str, reason: str) -> None:
        """Sağlayıcı değişimini olay veri yoluna bildirir."""
        bus = getattr(self, "_event_bus", None)
        if bus is None:
            return
        try:
            bus.emit("llm.failover", {
                "from": old, "to": new, "reason": (reason or "")[:200],
                "model": getattr(self._active, "model", ""),
            }, source="agent.llm")
        except Exception as exc:
            logger.debug(f"Geçiş olayı yayınlanamadı: {exc}")

    def provider_quota_exhausted(self) -> bool:
        """
        Aktif sağlayıcı **kota/yapılandırma** nedeniyle kullanılamaz mı?

        Son çağrıların hata oranına ve son hata metnine bakar. Kota
        (`429`/`quota`) veya kimlik hatası varsa True döner; bu durumda
        başka bir sağlayıcıya geçmek anlamlıdır.
        """
        provider = self._active
        if provider is None:
            return True
        try:
            if not provider.available:
                return True
        except Exception:
            return True

        # Dikkat: sağlayıcının istatistik metodu `get_stats()`'tır
        # (`stats()` değil) — yanlış ad, kotanın hiç görülmemesine yol açtı.
        stats: Dict[str, Any] = {}
        for attribute in ("get_stats", "stats"):
            method = getattr(provider, attribute, None)
            if callable(method):
                try:
                    stats = method() or {}
                except Exception:
                    stats = {}
                break

        calls = int(stats.get("calls", 0) or 0)
        errors = int(stats.get("errors", 0) or 0)
        last_error = str(stats.get("last_error", "") or "").lower()

        # Kota hatasının **türüne** bakmadan da karar verilebilir: bir
        # sağlayıcı birkaç kez denendi ve **hepsi** başarısız olduysa
        # pratikte ölüdür (kota, kimlik, ağ — fark etmez). Hata metnini
        # tanımaya çalışmak kırılgandır; sayı yalan söylemez.
        if calls >= 2 and errors >= calls:
            return True

        # Kota/yoğunluk hatası → başka sağlayıcıya geçmek anlamlıdır
        if any(marker in last_error for marker in
               ("429", "quota", "exceeded", "rate limit", "503",
                "high demand", "overloaded", "capacity")):
            return True
        # Kimlik hatası → anahtar geçersiz, geçiş de çözebilir
        if any(marker in last_error for marker in
               ("401", "403", "invalid api key", "authentication", "unauthorized")):
            return True
        return False

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır."""
        self._initialized = False
        logger.info(f"Dil modeli yönlendiricisi kapatıldı ({self._stats()['calls']} çağrı)")

    def _select(self) -> Optional[LLMProvider]:
        """
        Aktif sağlayıcıyı seçer.

        Sıra: tercih edilen (kullanılabilirse) → keşif sırası → yerel Ollama →
        NeuroCore → hiçbiri.
        """
        candidates: List[str] = []
        if self._preferred:
            candidates.append(self._preferred)
        candidates.extend(self.DISCOVERY_ORDER)
        candidates.append("ollama")
        candidates.append("neurocore")

        seen: set = set()
        for name in candidates:
            if name in seen:
                continue
            seen.add(name)
            provider = self._providers.get(name)
            if provider is None:
                continue
            try:
                if provider.available:
                    return provider
            except Exception as exc:
                logger.debug(f"Sağlayıcı kontrolü başarısız ({name}): {exc}")
        return None

    # ─── Durum ──────────────────────────────────────────────────────────

    @property
    def active(self) -> Optional[LLMProvider]:
        """Aktif sağlayıcı (None = hiçbiri)."""
        return self._active

    @property
    def available(self) -> bool:
        """Herhangi bir sağlayıcı kullanılabilir mi?"""
        return self._active is not None and self._active.available

    @property
    def supports_tools(self) -> bool:
        """Aktif sağlayıcı şema tabanlı araç çağrısı destekliyor mu?"""
        return bool(self._active and self._active.supports_tools)

    def set_provider(self, name: str, model: str = "") -> Dict[str, Any]:
        """
        Aktif sağlayıcıyı değiştirir (arayüzden).

        Args:
            name: Sağlayıcı adı (`deepseek`, `openai`, `ollama`, ...)
            model: Model adı (boş = varsayılan)

        Returns:
            Yeni durum sözlüğü
        """
        provider = self._providers.get(name)
        if provider is None:
            return {"ok": False, "error": f"bilinmeyen sağlayıcı: {name}",
                    "available": [key for key in self._providers]}
        if model:
            provider.model = model
        if not provider.available:
            return {"ok": False,
                    "error": f"{name} kullanılamıyor (anahtar veya sunucu yok)",
                    "status": self.status()}
        self._active = provider
        self._preferred = name
        logger.info(f"Aktif dil modeli: {name} ({provider.model})")
        return {"ok": True, "status": self.status()}

    def status(self) -> Dict[str, Any]:
        """Tam durum raporu (arayüz için)."""
        providers: List[Dict[str, Any]] = []
        for name, provider in self._providers.items():
            try:
                available = provider.available
            except Exception:
                available = False
            spec = KNOWN_SERVICES.get(name, {})
            providers.append({
                "name": name,
                "label": getattr(provider, "label", spec.get("label", name)),
                "model": provider.model,
                "available": available,
                "active": provider is self._active,
                "supports_tools": provider.supports_tools,
                "local": any(host in provider.base_url for host in
                             ("127.0.0.1", "localhost", "0.0.0.0", "::1")),
                "env_var": spec.get("env", ""),
                "requires_key": bool(spec.get("env")),
                "base_url": provider.base_url,
            })
        return {
            "available": self.available,
            "active": getattr(self._active, "label", "") or (
                self._active.name if self._active else ""),
            "active_key": self._active.name if self._active else "",
            "active_model": self._active.model if self._active else "",
            "supports_tools": self.supports_tools,
            "preferred": self._preferred,
            "providers": providers,
            "credential_file": str(self._credential_file or ""),
            "has_stored_credentials": bool(self._stored_keys),
            "last_error": self._last_error,
            "stats": self._stats(),
            "note": (
                "" if self.available else
                "Hiçbir dil modeli yapılandırılmadı. Ajan kural tabanlı modda "
                "çalışır: komutları ve dosya işlemlerini yapabilir ama serbest "
                "biçimde akıl yürütemez. Ayarlar'dan API anahtarı girin."
            ),
        }

    def _stats(self) -> Dict[str, Any]:
        """Tüm sağlayıcıların toplam istatistiği."""
        calls = errors = 0
        prompt_tokens = completion_tokens = 0
        total_ms = 0.0
        for provider in self._providers.values():
            counter = provider._counter
            calls += counter["calls"]
            errors += counter["errors"]
            prompt_tokens += counter["prompt_tokens"]
            completion_tokens += counter["completion_tokens"]
            total_ms += counter["total_ms"]
        return {
            "calls": calls, "errors": errors,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_ms": round(total_ms, 1),
        }

    # ─── Çağrı ──────────────────────────────────────────────────────────

    def chat(self, messages: Sequence[LLMMessage],
             tools: Optional[Sequence[Dict[str, Any]]] = None,
             provider: str = "", **kwargs: Any) -> LLMResponse:
        """
        Aktif sağlayıcıyla sohbet eder; başarısız olursa yedeğe geçer.

        Args:
            messages: Mesaj geçmişi
            tools: Araç şemaları (destekleyen sağlayıcılar için)
            provider: Belirli bir sağlayıcıyı zorla
            **kwargs: `temperature`, `max_tokens` geçersiz kılmaları

        Returns:
            `LLMResponse`
        """
        if provider:
            chosen = self._providers.get(provider)
        else:
            chosen = self._active

        if chosen is None:
            response = LLMResponse(
                ok=False, provider="none",
                error=self._last_error or (
                    "yapılandırılmış dil modeli yok — Ayarlar'dan API anahtarı "
                    "girin veya Ollama başlatın"
                ),
            )
            # Başarısız çağrı da **geçmişe** yazılır: kullanıcı "neden yanıt
            # gelmiyor?" sorusunun yanıtını çağrı kaydında görebilmeli.
            self._last_error = response.error
            self._history.append({
                "at": time.time(), "provider": "none", "model": "",
                "ok": False, "duration_ms": 0.0, "tokens": 0,
                "error": response.error[:200],
            })
            if len(self._history) > 200:
                self._history = self._history[-200:]
            return response

        response = chosen.chat(messages, tools=tools, **kwargs)

        # Yedek: birincil sağlayıcı çöktüyse sıradakini dene (yalnızca ağ/kota
        # hatalarında; içerik hatalarında değil)
        if not response.ok and not provider:
            fallback_error = response.error
            is_transient = any(marker in fallback_error for marker in
                               ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503",
                                "bağlantı hatası", "ağ hatası", "zaman aşımı"))
            if is_transient:
                for name, candidate in self._providers.items():
                    if candidate is chosen or not candidate.available:
                        continue
                    logger.info(f"Yedek sağlayıcı deneniyor: {name}")
                    alternate = candidate.chat(messages, tools=tools, **kwargs)
                    if alternate.ok:
                        alternate.error = ""
                        response = alternate
                        break

        if not response.ok:
            self._last_error = response.error

        self._history.append({
            "at": time.time(), "provider": response.provider,
            "model": response.model, "ok": response.ok,
            "duration_ms": response.duration_ms,
            "tokens": response.total_tokens,
            "error": response.error[:200],
        })
        if len(self._history) > 200:
            self._history = self._history[-200:]
        return response

    def call_history(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Son model çağrıları (şeffaflık için)."""
        return self._history[-limit:]

    def health(self) -> Dict[str, Any]:
        """Sağlayıcı sağlığı (arayüz için)."""
        status = self.status()
        status["providers_detail"] = [
            provider.health() for provider in self._providers.values()
        ]
        return status

    # ─── Kimlik Bilgileri ───────────────────────────────────────────────

    def _persist_credentials(self, data: Dict[str, Any]) -> bool:
        """
        Kimlik bilgilerini diske yazar (izinleri kısıtlayarak).

        Args:
            data: Yazılacak sözlük

        Returns:
            Başarılı ise True
        """
        if self._credential_file is None:
            return False
        try:
            self._credential_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._restrict_file(self._credential_file)
            return True
        except OSError as exc:
            logger.warning(f"Kimlik bilgileri yazılamadı: {exc}")
            return False

    def _load_credentials(self) -> Dict[str, Any]:
        """Kayıtlı kimlik bilgilerini okur."""
        if self._credential_file is None or not self._credential_file.exists():
            return {}
        try:
            data = json.loads(self._credential_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug(f"Kimlik dosyası okunamadı: {exc}")
            return {}

    def _restrict_file(self, path: Path) -> str:
        """
        Kimlik dosyasını yalnızca sahibine kısıtlar.

        Windows'ta `os.chmod` yalnızca **okuma** bayrağını anlamlı biçimde
        değiştirir; POSIX izin bitleri %666 görünmeye devam eder. Bu yüzden
        gerçek koruma `icacls` ile devralmayı kaldırıp yalnızca geçerli
        kullanıcıya izin vermekle sağlanır. Bu başarısız olursa (kısıtlı
        ortam, ilke engeli) sessizce dosya izniyle yetinilir — kritik değil.

        Returns:
            Uygulanan yöntem: `"acl"`, `"chmod"` veya `"yok"`
        """
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

        if os.name != "nt":
            return "chmod"

        try:
            user = os.environ.get("USERNAME", "")
            if not user:
                return "chmod"
            result = subprocess.run(
                ["icacls", str(path), "/inheritance:r",
                 "/grant:r", f"{user}:(R,W)"],
                capture_output=True, text=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return "acl" if result.returncode == 0 else "chmod"
        except (OSError, subprocess.SubprocessError):
            return "chmod"

    def save_credentials(self, service: str, api_key: str = "",
                         model: str = "", base_url: str = "") -> Dict[str, Any]:
        """
        API anahtarını diske kaydeder (arayüzden girilir).

        Dosya yalnızca kullanıcıya okunur/yazılır olacak şekilde
        kısıtlanmaya çalışılır (`chmod 600`; Windows'ta etkisi sınırlıdır).

        Args:
            service: Sağlayıcı adı
            api_key: API anahtarı (boş = silinmez, yalnızca diğer alanlar)
            model: Varsayılan model
            base_url: Özel adres

        Returns:
            {"ok", "service", "status"}
        """
        if self._credential_file is None:
            return {"ok": False, "error": "kimlik dosyası yolu yok"}

        data = self._load_credentials()
        data.setdefault("keys", {})
        data.setdefault("models", {})
        data.setdefault("base_urls", {})

        if api_key:
            data["keys"][service] = api_key.strip()
        if model:
            data["models"][service] = model.strip()
        if base_url:
            data["base_urls"][service] = base_url.strip()
        data["provider"] = service
        data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        try:
            self._credential_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._restrict_file(self._credential_file)
        except OSError as exc:
            return {"ok": False, "error": f"yazma hatası: {exc}"}

        # Sağlayıcıyı yeniden yapılandır
        provider = self._providers.get(service)
        if provider is not None:
            if api_key:
                provider.api_key = api_key.strip()
            if model:
                provider.model = model.strip()
            if base_url:
                provider.base_url = base_url.strip().rstrip("/")

        self._preferred = service
        if provider is not None and provider.available:
            self._active = provider

        self._stored_credentials = data
        self._stored_keys = sorted(data.get("keys", {}).keys())
        logger.info(f"Kimlik bilgisi kaydedildi: {service}")
        return {"ok": True, "service": service, "status": self.status()}

    def clear_credentials(self, service: str = "") -> Dict[str, Any]:
        """Kayıtlı anahtarları siler."""
        if self._credential_file is None:
            return {"ok": False, "error": "kimlik dosyası yolu yok"}
        data = self._load_credentials()
        if service:
            data.get("keys", {}).pop(service, None)
            data.get("models", {}).pop(service, None)
            data.get("base_urls", {}).pop(service, None)
            provider = self._providers.get(service)
            if provider is not None:
                provider.api_key = ""
        else:
            data["keys"] = {}
            data["models"] = {}
            data["base_urls"] = {}
            for provider in self._providers.values():
                provider.api_key = ""
        try:
            self._credential_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        self._stored_credentials = data
        self._stored_keys = sorted(data.get("keys", {}).keys())
        self._active = self._select()
        return {"ok": True, "status": self.status()}

    def test(self, service: str = "") -> Dict[str, Any]:
        """
        Bir sağlayıcıyı **gerçek** bir istekle sınar.

        Arayüzdeki "Bağlantıyı test et" düğmesi bunu çağırır; yapılandırmanın
        doğru olduğunu kullanıcıya kanıtlar.

        Not: `max_tokens` bilinçli olarak yüksek tutulur. "Reasoning"
        modelleri (gpt-oss, deepseek-reasoner) düşünme token'larını aynı
        bütçeden harcar; küçük bir sınırda yanıt **boş** döner ve test
        başarısız görünür — oysa bağlantı çalışıyordur.
        """
        provider = self._providers.get(service) if service else self._active
        if provider is None:
            return {"ok": False, "error": "sağlayıcı yok"}
        if not provider.available:
            return {"ok": False,
                    "error": f"{provider.name} kullanılamıyor (anahtar/sunucu yok)"}

        started = time.perf_counter()
        response = provider.chat(
            [LLMMessage("user", "Yalnızca şu iki kelimeyi yaz: baglanti tamam")],
            max_tokens=1024, temperature=0.0,
        )
        reply = (response.text or "").strip()
        # Bağlantı kuruldu ama metin boş: model düşünme bütçesini tüketti veya
        # yalnızca araç çağrısı üretti. Bu bir bağlantı hatası **değildir**.
        note = ""
        if response.ok and not reply:
            if response.tool_calls:
                reply = "(model araç çağrısı üretti — bağlantı çalışıyor)"
            elif response.finish_reason == "length":
                note = ("yanıt boş: model düşünme token'ları için ayrılan "
                        "bütçeyi tüketti. Bağlantı çalışıyor; daha büyük bir "
                        "model veya daha yüksek token sınırı deneyin.")
                reply = "(boş yanıt — düşünme bütçesi tükendi)"
            else:
                note = "yanıt boş döndü; bağlantı kuruldu ama model metin üretmedi"
                reply = "(boş yanıt)"

        return {
            "ok": response.ok,
            "provider": provider.name,
            "label": getattr(provider, "label", provider.name),
            "model": response.model or provider.model,
            "reply": reply[:160],
            "note": note,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "tokens": response.total_tokens,
            "finish_reason": response.finish_reason,
            "error": response.error,
        }

    def available_providers(self) -> List[Dict[str, Any]]:
        """Kullanılabilir sağlayıcıların kısa listesi."""
        return [item for item in self.status()["providers"] if item["available"]]

    def attach_engine(self, engine: Any) -> None:
        """NeuroCore motorunu yerel sağlayıcıya bağlar."""
        provider = self._providers.get("neurocore")
        if isinstance(provider, NeuroCoreProvider):
            provider.attach(engine)
            if self._active is None:
                self._active = self._select()

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<LLMRouter active={self._active.name if self._active else 'yok'} "
                f"available={self.available}>")


__all__ = [
    "LLMMessage",
    "LLMResponse",
    "LLMProvider",
    "LLMError",
    "OpenAICompatibleProvider",
    "OllamaProvider",
    "NeuroCoreProvider",
    "LLMRouter",
    "KNOWN_SERVICES",
]
