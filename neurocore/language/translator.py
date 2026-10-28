"""
NeuroCore Çeviri Motoru
========================
Çok sağlayıcılı çeviri altyapısı: kimlik (identity), sözlük, yerel sinir
ağı (MarianMT) ve API çeviricileri; LRU önbellek ve sıralı zincir (cascade).

Tasarım ilkeleri:
    - `CascadeTranslator` **asla hata fırlatmaz**; her sağlayıcı çağrısı
      try/except içindedir ve başarısızlık `warnings` listesine yazılır.
    - Ağır bağımlılıklar (`torch`, `transformers`) fonksiyon içinde import
      edilir → paket yoksa sistem yine de ayağa kalkar.
    - Terim sözlüğü (glossary) sonuçlara ön-geçiş olarak uygulanır.

Zincir sırası:
    identity → cache → terminology → neural → dictionary → api → passthrough

Kullanım:
    cascade = CascadeTranslator(store, cache, bridge)
    cascade.initialize(config, event_bus)
    result = cascade.translate("yapay sinir ağı", "tr", "en")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

from neurocore.utils.logger import NeuroCoreLogger
from neurocore.language.dictionary import TerminologyStore, normalize_language_code

logger = NeuroCoreLogger.get_logger("language.translator")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: Sözlük çevirisinde kabul edilen en düşük kapsama (altında güven < 0.3)
DICTIONARY_MIN_COVERAGE: float = 0.25
#: Zincirin başarılı sayılması için gereken en düşük kapsama
CASCADE_ACCEPT_COVERAGE: float = 0.30
#: Önbellek dosyası adı
CACHE_FILENAME: str = "translation_cache.json"
#: MarianMT model önbellek dizini (base_path altında)
MODEL_CACHE_DIRNAME: str = "models/translations"

#: Kelime ayırma deseni (harf + rakam + iç kesme işareti)
_WORD_RE: Pattern[str] = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
#: Türkçe basit gövde indirgeme ekleri (en uzundan kısaya)
_TR_SUFFIXES: Tuple[str, ...] = (
    "larından", "lerinden", "larında", "lerinde", "lardan", "lerden",
    "ların", "lerin", "lara", "lere", "ları", "leri", "dan", "den",
    "tan", "ten", "lar", "ler", "da", "de", "ta", "te", "ın", "in",
    "un", "ün", "ı", "i", "u", "ü", "a", "e",
)

#: Çeviri istemi şablonu (API sağlayıcıları için)
TRANSLATION_INSTRUCTION: str = (
    "You are a professional translator. Translate the user text from {src_name} "
    "to {dst_name}. Preserve technical terms, numbers, markdown and formatting. "
    "Return ONLY the translation, without explanations or quotes."
)


# ─── Çeviri Sonucu ─────────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    """
    Tek bir çeviri işleminin sonucu.

    Attributes:
        text: Çevrilmiş metin (başarısızsa kaynak metin)
        source_lang: Kaynak dil kodu
        target_lang: Hedef dil kodu
        provider: Sonucu üreten sağlayıcı ('identity', 'neural', 'api', ...)
        confidence: 0.0-1.0 güven
        cached: Sonuç önbellekten mi geldi
        glossary_hits: Uygulanan terim eşleşmeleri {kaynak: hedef}
        warnings: Uyarı kodları/mesajları (ör: 'ceviri_yapilamadi')
    """
    text: str
    source_lang: str
    target_lang: str
    provider: str
    confidence: float = 0.0
    cached: bool = False
    glossary_hits: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Çeviri anlamlı biçimde üretildi mi?"""
        return bool(self.text) and self.provider not in ("failed", "none") and self.confidence > 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sözlük döndür."""
        return {
            "text": self.text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "provider": self.provider,
            "confidence": round(float(self.confidence), 4),
            "cached": bool(self.cached),
            "glossary_hits": dict(self.glossary_hits),
            "warnings": list(self.warnings),
            "ok": self.ok,
        }


def _normalize_for_spans(text: str) -> str:
    """
    Aralık eşleştirmesi için metni sadeleştirir.

    Sözlükteki **kanonik normalizasyonun** aynısını kullanır (İ/ı → i) ki
    terim desenleri ile taranan metin aynı biçime gelsin. Dönüşüm karakter
    sayısını korur → `re.finditer` konumları özgün metinle hizalı kalır.
    """
    if not text:
        return ""
    from neurocore.language.dictionary import _normalize_term
    return _normalize_term(text)


def make_failed_result(
    text: str,
    source_lang: str,
    target_lang: str,
    provider: str = "failed",
    warnings: Optional[List[str]] = None,
    confidence: float = 0.0,
    glossary_hits: Optional[Dict[str, str]] = None,
) -> TranslationResult:
    """Başarısız/geri düşüş (fallback) sonucu üretir."""
    return TranslationResult(
        text=text,
        source_lang=normalize_language_code(source_lang) or source_lang,
        target_lang=normalize_language_code(target_lang) or target_lang,
        provider=provider,
        confidence=confidence,
        glossary_hits=dict(glossary_hits or {}),
        warnings=list(warnings or []),
    )


# ─── Çeviri Önbelleği ──────────────────────────────────────────────────────

class TranslationCache:
    """
    LRU çeviri önbelleği (isteğe bağlı disk kalıcılığı).

    Anahtar: blake2b(f"{src}|{dst}|{text}")
    Kapasite aşıldığında en eski (en az kullanılan) kayıt atılır.

    Kullanım:
        cache = TranslationCache()
        cache.initialize(config, event_bus)
        hit = cache.get("merhaba", "tr", "en")
    """

    name: str = "language.cache"

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — I/O yok."""
        self._initialized: bool = False
        self._config = None
        self._event_bus = None
        self._store: "OrderedDict[str, TranslationResult]" = OrderedDict()
        self._max_size: int = 5000
        self._path: Optional[Path] = None
        self._lock = threading.RLock()
        self._stats: Dict[str, int] = {
            "hits": 0, "misses": 0, "puts": 0, "evictions": 0,
            "clears": 0, "saves": 0, "loads": 0, "load_skipped": 0, "errors": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config=None, event_bus=None) -> None:
        """Önbelleği başlat ve diskten yükle. İdempotenttir."""
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus

        language_cfg = getattr(config, "language", None) if config is not None else None
        size = getattr(language_cfg, "translation_cache_size", None)
        try:
            self._max_size = max(1, int(size)) if size else 5000
        except (TypeError, ValueError):
            self._max_size = 5000

        self._path = self._resolve_path()
        loaded = self.load()
        self._initialized = True

        logger.info(
            f"Çeviri önbelleği hazır: kapasite {self._max_size}, "
            f"yüklenen {loaded}, dosya: {self._path}"
        )
        if self._event_bus:
            self._event_bus.emit(
                "language.cache_ready",
                {"max_size": self._max_size, "loaded": loaded, "path": str(self._path or "")},
                source=self.name,
            )

    def shutdown(self) -> None:
        """Önbelleği kapat (diske yazmaz — `save()` ayrı çağrılır). İdempotenttir."""
        if not self._initialized:
            return
        with self._lock:
            self._store.clear()
        self._initialized = False
        logger.info("Çeviri önbelleği kapatıldı")

    def _resolve_path(self) -> Optional[Path]:
        """Önbellek dosyası yolunu çözer (<data_path>/translation_cache.json)."""
        config = self._config
        data_path = getattr(config, "data_path", None) if config is not None else None
        if not data_path:
            base_path = getattr(config, "base_path", None) if config is not None else None
            if not base_path:
                base_path = str(Path(__file__).resolve().parents[2])
            data_path = str(Path(str(base_path)) / "data")
        try:
            return Path(str(data_path)) / CACHE_FILENAME
        except Exception:  # pragma: no cover - savunma
            return None

    # ─── Anahtar ───────────────────────────────────────────────────────

    @staticmethod
    def key(text: str, src: str, dst: str) -> str:
        """Önbellek anahtarı: blake2b(f'{src}|{dst}|{text}')."""
        src_code = normalize_language_code(src) or str(src).lower()
        dst_code = normalize_language_code(dst) or str(dst).lower()
        raw = f"{src_code}|{dst_code}|{text if text is not None else ''}"
        return hashlib.blake2b(raw.encode("utf-8", errors="replace"), digest_size=16).hexdigest()

    # ─── Erişim ────────────────────────────────────────────────────────

    def get(self, text: str, src: str, dst: str) -> Optional[TranslationResult]:
        """
        Önbellekten çeviri okur (LRU: erişilen kayıt sona taşınır).

        Args:
            text: Kaynak metin
            src: Kaynak dil kodu
            dst: Hedef dil kodu

        Returns:
            Kopyalanmış TranslationResult (cached=True) veya None
        """
        key = self.key(text, src, dst)
        with self._lock:
            result = self._store.get(key)
            if result is None:
                self._stats["misses"] += 1
                return None
            self._store.move_to_end(key)
            self._stats["hits"] += 1

        return TranslationResult(
            text=result.text,
            source_lang=result.source_lang,
            target_lang=result.target_lang,
            provider=result.provider,
            confidence=result.confidence,
            cached=True,
            glossary_hits=dict(result.glossary_hits),
            warnings=list(result.warnings),
        )

    def put(self, result: TranslationResult, source_text: Optional[str] = None) -> None:
        """
        Çeviri sonucunu önbelleğe yazar (yalnızca anlamlı sonuçlar).

        Anahtar kaynak metne dayandığı için `source_text` verilmelidir.
        Verilmezse sonuç önbelleğe alınamaz; bu durum sessizce sayılır.

        Args:
            result: Yazılacak sonuç
            source_text: Çevrilen özgün metin (önerilir)
        """
        if source_text:
            self.put_for(source_text, result)
            return

        # Kaynak metin bilinmiyor → anahtar üretilemez, yalnızca sayaç artırılır
        if isinstance(result, TranslationResult) and result.ok:
            self._stats["missing_source"] = self._stats.get("missing_source", 0) + 1

    def put_for(self, source_text: str, result: TranslationResult) -> None:
        """
        Kaynak metni açıkça vererek sonucu önbelleğe yazar.

        Args:
            source_text: Çevrilen özgün metin (anahtar için)
            result: Çeviri sonucu
        """
        if not source_text or not isinstance(result, TranslationResult) or not result.ok:
            return

        key = self.key(source_text, result.source_lang, result.target_lang)
        stored = TranslationResult(
            text=result.text,
            source_lang=result.source_lang,
            target_lang=result.target_lang,
            provider=result.provider,
            confidence=result.confidence,
            cached=False,
            glossary_hits=dict(result.glossary_hits),
            warnings=list(result.warnings),
        )
        with self._lock:
            self._store[key] = stored
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)
                self._stats["evictions"] += 1
        self._stats["puts"] += 1

    def clear(self) -> None:
        """Önbelleği tamamen boşaltır."""
        with self._lock:
            self._store.clear()
        self._stats["clears"] += 1
        logger.debug("Çeviri önbelleği temizlendi")

    def size(self) -> int:
        """Önbellekteki kayıt sayısı."""
        with self._lock:
            return len(self._store)

    # ─── Kalıcılık ─────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """
        Önbelleği diske yazar.

        Args:
            path: Dosya yolu (None → varsayılan)

        Returns:
            Yazılan dosya yolu (hata → boş string)
        """
        target = Path(path) if path else self._path
        if target is None:
            return ""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                items = list(self._store.items())[-self._max_size:]
            payload = {
                "version": 1,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "items": [
                    {
                        "k": key,
                        "src": r.source_lang,
                        "dst": r.target_lang,
                        "result": r.text,
                        "provider": r.provider,
                        "confidence": r.confidence,
                        "warnings": list(r.warnings),
                    }
                    for key, r in items
                ],
            }
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)
        except Exception as exc:
            self._stats["errors"] += 1
            logger.warning(f"Çeviri önbelleği kaydedilemedi ({target}): {exc}")
            return ""

        self._stats["saves"] += 1
        return str(target)

    def load(self, path: Optional[str] = None) -> int:
        """
        Önbelleği diskten yükler (bozuk dosyaya toleranslı).

        Args:
            path: Dosya yolu (None → varsayılan)

        Returns:
            Yüklenen kayıt sayısı
        """
        target = Path(path) if path else self._path
        if target is None or not target.exists():
            return 0
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            self._stats["errors"] += 1
            logger.warning(f"Çeviri önbelleği okunamadı ({target}): {exc}")
            return 0

        items = payload.get("items", []) if isinstance(payload, dict) else []
        loaded = 0
        with self._lock:
            for item in items:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("k", ""))
                text = item.get("result")
                if not key or not isinstance(text, str) or not text:
                    continue
                src = str(item.get("src", ""))
                dst = str(item.get("dst", ""))
                self._store[key] = TranslationResult(
                    text=text, source_lang=src, target_lang=dst,
                    provider=str(item.get("provider", "cache")),
                    confidence=float(item.get("confidence", 0.5) or 0.5),
                    cached=True,
                    warnings=list(item.get("warnings", []) or []),
                )
                loaded += 1
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)
                self._stats["evictions"] += 1
        self._stats["loads"] += 1
        if loaded:
            logger.info(f"Çeviri önbelleği yüklendi: {loaded} kayıt <- {target}")
        return loaded

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        with self._lock:
            size = len(self._store)
        total_lookups = self._stats["hits"] + self._stats["misses"]
        return {
            "name": self.name,
            "initialized": self._initialized,
            "size": size,
            "max_size": self._max_size,
            "path": str(self._path or ""),
            "hit_rate": round(self._stats["hits"] / total_lookups, 4) if total_lookups else 0.0,
            **self._stats,
        }


# ─── Çevirici Taban Sınıfı ─────────────────────────────────────────────────

class BaseTranslator(ABC):
    """
    Tüm çeviricilerin ortak arayüzü (Katman Sözleşmesi).

    Alt sınıflar `available()` ve `translate()` uygulamak zorundadır.
    """

    name: str = "base"

    def __init__(self, config=None) -> None:
        """Yalnızca durum kurulumu — I/O yok."""
        self._config = config
        self._event_bus = None
        self._initialized: bool = False
        self._stats: Dict[str, int] = {
            "calls": 0, "success": 0, "failure": 0, "skipped": 0, "errors": 0,
        }

    def initialize(self, config=None, event_bus=None) -> None:
        """Çeviriciyi başlat. İdempotenttir: ikinci çağrı no-op'tur."""
        if self._initialized:
            return
        if config is not None:
            self._config = config
        self._event_bus = event_bus
        self._initialized = True
        logger.debug(f"Çevirici hazır: {self.name}")

    def shutdown(self) -> None:
        """Kaynakları serbest bırak. İdempotenttir."""
        self._initialized = False

    @abstractmethod
    def available(self) -> bool:
        """Bu çevirici şu anda kullanılabilir mi?"""
        raise NotImplementedError

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str, **kw) -> TranslationResult:
        """Metni çevirir."""
        raise NotImplementedError

    def supports(self, source_lang: str, target_lang: str) -> bool:
        """
        Dil çiftinin desteklenip desteklenmediğini söyler.

        Varsayılan: farklı diller arası çeviri desteklenir.
        """
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        return bool(src and dst and src != dst)

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "available": self._safe_available(),
            **self._stats,
        }

    def _safe_available(self) -> bool:
        """`available()` çağrısını hata-fırlatmadan sarmalar."""
        try:
            return bool(self.available())
        except Exception as exc:  # pragma: no cover - savunma
            logger.debug(f"{self.name}.available() hata verdi: {exc}")
            return False

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Olay yayınlar (bus yoksa sessiz geçer)."""
        if self._event_bus:
            try:
                self._event_bus.emit(f"language.{event}", data, source=self.name)
            except Exception as exc:  # pragma: no cover - savunma
                logger.debug(f"Olay yayınlanamadı ({event}): {exc}")


# ─── Kimlik Çevirici ───────────────────────────────────────────────────────

class IdentityTranslator(BaseTranslator):
    """
    Aynı dil çifti için geçiş (passthrough) çevirici.

    tr→tr isteğinde metni değiştirmeden, güven 1.0 ile döndürür.
    """

    name: str = "identity"

    def __init__(self, config=None) -> None:
        super().__init__(config)

    def available(self) -> bool:
        """Kimlik çevirici her zaman kullanılabilir."""
        return True

    def supports(self, source_lang: str, target_lang: str) -> bool:
        """Yalnızca kaynak == hedef olduğunda destekler."""
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        return bool(src and dst and src == dst)

    def translate(self, text: str, source_lang: str, target_lang: str, **kw) -> TranslationResult:
        """Metni olduğu gibi döndürür (güven 1.0)."""
        self._stats["calls"] += 1
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        if src != dst:
            self._stats["skipped"] += 1
            return make_failed_result(
                text, src, dst, provider=self.name,
                warnings=["identity_dil_uyusmuyor"],
            )
        self._stats["success"] += 1
        return TranslationResult(
            text=text, source_lang=src, target_lang=dst,
            provider=self.name, confidence=1.0,
        )


# ─── Sözlük Çevirici ───────────────────────────────────────────────────────

class DictionaryTranslator(BaseTranslator):
    """
    Terim sözlüğü tabanlı kelime-kelime çevirici.

    - Bilinmeyen kelimeler **değiştirilmez** ve `warnings` içinde raporlanır.
    - `confidence` = kapsama oranı (çevrilebilen kelime / toplam kelime).
    - Kapsama 0.25 altındaysa güven 0.3'ün altına düşürülür.
    """

    name: str = "dictionary"

    def __init__(self, store: Optional[TerminologyStore] = None, config=None) -> None:
        super().__init__(config)
        self._store = store if store is not None else TerminologyStore()
        self._coverage_sum: float = 0.0
        self._last_coverage: float = 0.0
        self._target_stopwords: Dict[str, set] = {}

    def initialize(self, config=None, event_bus=None) -> None:
        """Çeviriciyi ve (gerekirse) terim deposunu başlatır. İdempotenttir."""
        if self._initialized:
            return
        super().initialize(config, event_bus)
        if not getattr(self._store, "_initialized", False):
            try:
                self._store.initialize(config, event_bus)
            except Exception as exc:
                logger.warning(f"Sözlük çevirici: terim deposu başlatılamadı: {exc}")

    @property
    def store(self) -> TerminologyStore:
        """Kullanılan terim deposu."""
        return self._store

    def available(self) -> bool:
        """Depoda en az bir kayıt varsa kullanılabilir."""
        try:
            return len(self._store) > 0
        except Exception:  # pragma: no cover - savunma
            return False

    def supports(self, source_lang: str, target_lang: str) -> bool:
        """Depoda bu dil çifti için kayıt varsa destekler."""
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        if not src or not dst or src == dst:
            return False
        try:
            return bool(self._store.vocabulary(src, dst))
        except Exception:  # pragma: no cover - savunma
            return False

    def translate(self, text: str, source_lang: str, target_lang: str, **kw) -> TranslationResult:
        """
        Metni sözlükten kelime kelime çevirir.

        Tek geçişte çalışır:
            1. Çok kelimeli terimler en uzun eşleşme önceliğiyle, özgün
               metnin karakter aralıkları üzerinden yerleştirilir.
            2. Terim dışında kalan kelimeler sözlükten tek tek çevrilir.

        Bilinmeyen kelimeler **değiştirilmez**, özgün hâliyle kalır ve
        `warnings` içinde raporlanır. `confidence` = kapsama oranı
        (çevrilen özgün karakter / toplam kelime karakteri).
        """
        self._stats["calls"] += 1
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)

        if not text or not text.strip():
            self._stats["failure"] += 1
            return make_failed_result(text, src, dst, provider=self.name,
                                      warnings=["bos_metin"])

        try:
            # 1) Çok kelimeli terim eşleşmeleri (çakışmasız, uzunluk öncelikli)
            phrase_matches = self._match_phrases(text, src, dst)
            applied: Dict[str, str] = {
                original: translation for _start, _end, original, translation in phrase_matches
            }

            # 2) Tek geçiş: terimler ve kelimeler aynı taramada yerleştirilir
            pieces: List[str] = []
            unknown: List[str] = []
            unknown_seen: set = set()
            cursor = 0
            total_chars = 0
            translated_chars = 0

            for match in _WORD_RE.finditer(text):
                token = match.group(0)
                total_chars += len(token)

                covering = next(
                    (item for item in phrase_matches
                     if item[0] <= match.start() and match.end() <= item[1]),
                    None,
                )

                if covering is not None:
                    # Terim bölgesi: ilk kelimede tüm terim bir kerede yazılır
                    start, end, original, translation = covering
                    if match.start() == start:
                        pieces.append(text[cursor:start])
                        pieces.append(translation)
                        cursor = end
                    translated_chars += len(token)
                    continue

                entry = self._store.lookup(token, src, dst)
                if not (entry and entry.get("translation")):
                    # Türkçe eklerini atarak gövde adayını dene
                    variant = self._singular_candidate(token, src)
                    if variant:
                        entry = self._store.lookup(variant, src, dst)

                if entry and entry.get("translation"):
                    replacement = str(entry["translation"])
                    applied.setdefault(token, replacement)
                    translated_chars += len(token)
                else:
                    replacement = token
                    lowered = token.lower()
                    if lowered not in unknown_seen:
                        unknown_seen.add(lowered)
                        unknown.append(token)

                pieces.append(text[cursor:match.start()])
                pieces.append(replacement)
                cursor = match.end()

            pieces.append(text[cursor:])
            result_text = re.sub(r"[ \t]{2,}", " ", "".join(pieces)).strip()
            if not result_text:
                result_text = text

            coverage = (translated_chars / total_chars) if total_chars else 0.0
            confidence = coverage
            if coverage < DICTIONARY_MIN_COVERAGE:
                confidence = min(confidence, 0.29)

            warnings: List[str] = []
            if unknown:
                warnings.append(
                    "bilinmeyen_kelimeler: " + ", ".join(unknown[:8])
                    + ("..." if len(unknown) > 8 else "")
                )
            if coverage < DICTIONARY_MIN_COVERAGE:
                warnings.append("sozluk_kapsamasi_dusuk")

            self._coverage_sum += coverage
            self._last_coverage = coverage
            self._stats["success"] += 1

            return TranslationResult(
                text=result_text,
                source_lang=src,
                target_lang=dst,
                provider=self.name,
                confidence=round(float(confidence), 4),
                glossary_hits=applied,
                warnings=warnings,
            )
        except Exception as exc:
            self._stats["errors"] += 1
            logger.warning(f"Sözlük çevirisi başarısız: {exc}")
            return make_failed_result(
                text, src, dst, provider=self.name,
                warnings=[f"sozluk_hatasi: {exc}"],
            )

    @staticmethod
    def _singular_candidate(token: str, src: str) -> str:
        """Türkçe için basit gövde adayı üretir (çoğul/hâl eklerini atar)."""
        if src != "tr" or len(token) <= 4:
            return ""
        lowered = token.lower()
        for suffix in _TR_SUFFIXES:
            if lowered.endswith(suffix) and len(lowered) - len(suffix) >= 3:
                return token[: len(token) - len(suffix)]
        return ""

    def _match_phrases(
        self,
        text: str,
        src: str,
        dst: str,
    ) -> List[Tuple[int, int, str, str]]:
        """
        Metindeki çok kelimeli terimleri bulur.

        Aralıklar **özgün metnin** karakter koordinatlarındadır; eşleşmeler
        uzunluk önceliğiyle seçilir ve birbirleriyle çakışmaz.

        Returns:
            [(başlangıç, bitiş, özgün_terim, çeviri)] listesi (konuma göre sıralı)
        """
        matches: List[Tuple[int, int, str, str]] = []
        try:
            lowered = _normalize_for_spans(text)
            for _canonical, pattern in self._store._pattern_index(src, dst):
                for match in pattern.finditer(lowered):
                    original = text[match.start():match.end()]
                    entry = self._store.lookup(original, src, dst)
                    if entry and entry.get("translation"):
                        matches.append((
                            match.start(), match.end(), original, str(entry["translation"]),
                        ))
                        break
        except Exception as exc:  # pragma: no cover - savunma
            logger.debug(f"Terim eşleşmeleri bulunamadı: {exc}")
            return []

        if not matches:
            return []

        # Çakışan eşleşmeleri uzunluk önceliğiyle çöz
        matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        accepted: List[Tuple[int, int, str, str]] = []
        for item in matches:
            if any(item[0] < acc[1] and acc[0] < item[1] for acc in accepted):
                continue
            accepted.append(item)
        accepted.sort(key=lambda item: item[0])
        return accepted

    def _phrase_spans(self, text: str, src: str, dst: str) -> List[Tuple[int, int]]:
        """
        Terim eşleşmelerinin özgün metindeki karakter aralıkları.

        `translate()` artık `_match_phrases()` kullanır; bu yardımcı
        teşhis/uyumluluk amaçlı korunmuştur ve istatistik sayaçlarını
        değiştirmez.

        Args:
            text: Özgün metin
            src: Kaynak dil kodu
            dst: Hedef dil kodu

        Returns:
            [(başlangıç, bitiş)] listesi (özgün metin koordinatlarında)
        """
        return [
            (start, end)
            for start, end, _original, _translation in self._match_phrases(text, src, dst)
        ]

    @staticmethod
    def _overlaps(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
        """Aralık, verilen bölgelerden biriyle kesişiyor mu?"""
        for span_start, span_end in spans:
            if start < span_end and span_start < end:
                return True
        return False

    def _looks_like_target_language(self, token: str, dst: str) -> bool:
        """
        Kelime hedef dilin durak kelimelerinden biri mi? (ör: 'and', 'the')

        Terim geçişinden sonra hedef dilde kalan bağlaçları tanımak için
        kullanılır; durak kelime listeleri tembel yüklenir ve önbelleklenir.
        """
        if not token or not dst:
            return False
        if dst not in self._target_stopwords:
            words: set = set()
            try:
                from neurocore.language.detector import STOPWORDS
                words = {str(word).lower() for word in STOPWORDS.get(dst, [])}
            except Exception:  # pragma: no cover - savunma
                words = set()
            self._target_stopwords[dst] = words
        return token.lower() in self._target_stopwords[dst]

    @staticmethod
    def _rebuild(intermediate: str, replacements: List[str]) -> str:
        """
        Çevrilen kelimeleri özgün noktalama/boşluk düzeniyle birleştirir.

        Not: `translate()` tek geçişli yerleştirme kullanır; bu yardımcı
        dışarıdan çağrılabilecek eski biçimlerle uyumluluk için korunmuştur.
        """
        pieces: List[str] = []
        cursor = 0
        index = 0
        for match in _WORD_RE.finditer(intermediate):
            if index >= len(replacements):
                break
            pieces.append(intermediate[cursor:match.start()])
            pieces.append(replacements[index])
            cursor = match.end()
            index += 1
        pieces.append(intermediate[cursor:])
        return "".join(pieces).strip()

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        stats = super().get_stats()
        calls = max(1, self._stats["success"])
        stats.update({
            "glossary_entries": len(self._store) if self._store else 0,
            "last_coverage": round(float(self._last_coverage), 4),
            "avg_coverage": round(float(self._coverage_sum / calls), 4),
        })
        return stats


# ─── Sinir Ağı Çevirici (MarianMT) ─────────────────────────────────────────

class NeuralTranslator(BaseTranslator):
    """
    Yerel MarianMT (Helsinki-NLP/opus-mt-*) tabanlı çevirici.

    Davranış:
        - `transformers` + `torch` yoksa → `available()` False
        - `config.language.use_local_model` False ise → `available()` False
        - Model yalnızca yerel önbellekte varsa yüklenir; indirme yalnızca
          `NEUROCORE_ALLOW_DOWNLOAD=1` ortam değişkeniyle yapılır.
        - Model dizini: <base_path>/models/translations/<src>-<dst>/

    Kullanım:
        neural = NeuralTranslator()
        neural.initialize(config, event_bus)
        if neural.available():
            sonuc = neural.translate("merhaba", "tr", "en")
    """

    name: str = "neural"
    #: MarianMT dil çifti için alternatif model kimlikleri
    MODEL_TEMPLATE: str = "Helsinki-NLP/opus-mt-{src}-{dst}"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self._models: Dict[str, Tuple[Any, Any]] = {}
        self._model_lock = threading.RLock()
        self._torch_available: Optional[bool] = None
        self._transformers_available: Optional[bool] = None
        self._last_error: str = ""
        self._last_load_ms: float = 0.0
        self._base_path: str = ""
        self._use_local_model: bool = True
        self._stats.update({"model_loads": 0, "cache_hits": 0, "segments": 0})

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config=None, event_bus=None) -> None:
        """Çeviriciyi başlat (model YÜKLENMEZ). İdempotenttir."""
        if self._initialized:
            return
        super().initialize(config, event_bus)

        language_cfg = getattr(config, "language", None) if config is not None else None
        use_local = getattr(language_cfg, "use_local_model", True)
        self._use_local_model = bool(use_local) if use_local is not None else True

        base_path = getattr(config, "base_path", None) if config is not None else None
        if not base_path:
            base_path = str(Path(__file__).resolve().parents[2])
        self._base_path = str(base_path)

        self._torch_available = self._module_importable("torch")
        self._transformers_available = self._module_importable("transformers")

        logger.info(
            f"Sinir ağı çevirici hazır (torch: {self._torch_available}, "
            f"transformers: {self._transformers_available}, "
            f"yerel model: {self._use_local_model})"
        )

    def shutdown(self) -> None:
        """Yüklü modelleri bellekten düşürür. İdempotenttir."""
        if not self._initialized:
            return
        with self._model_lock:
            count = len(self._models)
            self._models.clear()
        self._initialized = False
        logger.info(f"Sinir ağı çevirici kapatıldı ({count} model düşürüldü)")

    @staticmethod
    def _module_importable(name: str) -> bool:
        """Bir modülün import edilebilir olup olmadığını (yüklemeden) sınar."""
        import importlib.util
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    # ─── Kullanılabilirlik ─────────────────────────────────────────────

    def model_dir(self, source_lang: str, target_lang: str) -> Path:
        """Yerel model dizini: <base_path>/models/translations/<src>-<dst>/."""
        src = normalize_language_code(source_lang) or "src"
        dst = normalize_language_code(target_lang) or "dst"
        return Path(self._base_path or ".") / MODEL_CACHE_DIRNAME / f"{src}-{dst}"

    def is_cached(self, source_lang: str, target_lang: str) -> bool:
        """Model yerel önbellekte (ağ erişimi gerektirmeden) var mı?"""
        try:
            directory = self.model_dir(source_lang, target_lang)
            return directory.is_dir() and any(directory.iterdir())
        except Exception:  # pragma: no cover - savunma
            return False

    def _download_allowed(self) -> bool:
        """Model indirmeye izin var mı? (yalnızca NEUROCORE_ALLOW_DOWNLOAD=1)."""
        return str(os.environ.get("NEUROCORE_ALLOW_DOWNLOAD", "")).strip() == "1"

    def can_translate(self, source_lang: str, target_lang: str) -> bool:
        """
        Bu dil çifti için çeviri yapılabilir mi?

        Ağ erişimi YAPMAZ: yalnızca yerel dizin varlığına ve indirme
        iznine bakar.
        """
        if not self._use_local_model:
            return False
        if not (self._torch_available and self._transformers_available):
            return False
        if self.is_cached(source_lang, target_lang):
            return True
        return self._download_allowed()

    def available(self) -> bool:
        """Herhangi bir çeviri yapılabiliyor mu? (ağ erişimi yok)"""
        if not self._use_local_model:
            return False
        if not (self._torch_available and self._transformers_available):
            return False
        with self._model_lock:
            if self._models:
                return True
        try:
            base = Path(self._base_path or ".") / MODEL_CACHE_DIRNAME
            if base.is_dir() and any(base.iterdir()):
                return True
        except Exception:  # pragma: no cover - savunma
            pass
        return self._download_allowed()

    def supports(self, source_lang: str, target_lang: str) -> bool:
        """MarianMT dil çifti destekleniyor mu?"""
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        if not src or not dst or src == dst:
            return False
        return self.can_translate(src, dst)

    # ─── Model Yükleme ─────────────────────────────────────────────────

    def _load_model(self, source_lang: str, target_lang: str) -> Optional[Tuple[Any, Any]]:
        """
        MarianMT modelini tembel olarak yükler (ve önbelleğe alır).

        Returns:
            (tokenizer, model) ikilisi veya None
        """
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        pair = f"{src}-{dst}"

        with self._model_lock:
            cached = self._models.get(pair)
            if cached is not None:
                self._stats["cache_hits"] += 1
                return cached

        if not self.can_translate(src, dst):
            self._last_error = "model_yerel_degil"
            logger.debug(f"MarianMT modeli yok ve indirme kapalı: {pair}")
            return None

        started = time.perf_counter()
        try:
            import torch  # noqa: F401  (fonksiyon içi ağır import)
            from transformers import MarianMTModel, MarianTokenizer
        except Exception as exc:
            self._last_error = f"bagimlilik_eksik: {exc}"
            logger.warning(f"Sinir ağı çeviri bağımlılıkları eksik: {exc}")
            return None

        directory = self.model_dir(src, dst)
        source = str(directory) if self.is_cached(src, dst) else self.MODEL_TEMPLATE.format(src=src, dst=dst)

        try:
            logger.info(f"MarianMT modeli yükleniyor: {source}")
            tokenizer = MarianTokenizer.from_pretrained(source)
            model = MarianMTModel.from_pretrained(source)
            model.eval()
            if self.is_cached(src, dst):
                try:
                    tokenizer.save_pretrained(str(directory))
                    model.save_pretrained(str(directory))
                except Exception as exc:  # pragma: no cover - savunma
                    logger.debug(f"Model yerel dizine yazılamadı: {exc}")
        except Exception as exc:
            self._last_error = f"model_yuklenemedi: {exc}"
            self._stats["errors"] += 1
            logger.warning(f"MarianMT modeli yüklenemedi ({pair}): {exc}")
            return None

        with self._model_lock:
            self._models[pair] = (tokenizer, model)
        self._stats["model_loads"] += 1
        self._last_load_ms = (time.perf_counter() - started) * 1000.0
        self._last_error = ""
        logger.info(f"MarianMT modeli hazır: {pair} ({self._last_load_ms:.0f}ms)")
        self._emit("neural_model_loaded", {"pair": pair, "load_ms": round(self._last_load_ms, 1)})
        return tokenizer, model

    # ─── Çeviri ────────────────────────────────────────────────────────

    def translate(self, text: str, source_lang: str, target_lang: str, **kw) -> TranslationResult:
        """
        Metni yerel MarianMT modeliyle çevirir.

        Kwargs:
            max_length (int): Üretilecek en fazla jeton (varsayılan 512)
            num_beams (int): Kiriş sayısı (varsayılan 4)
        """
        self._stats["calls"] += 1
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)

        if not text or not text.strip():
            self._stats["failure"] += 1
            return make_failed_result(text, src, dst, provider=self.name,
                                      warnings=["bos_metin"])

        if not self._use_local_model:
            self._stats["skipped"] += 1
            return make_failed_result(text, src, dst, provider=self.name,
                                      warnings=["yerel_model_kapali"])

        max_length = int(kw.get("max_length", 512) or 512)
        num_beams = int(kw.get("num_beams", 4) or 4)

        try:
            loaded = self._load_model(src, dst)
            if loaded is None:
                self._stats["failure"] += 1
                return make_failed_result(
                    text, src, dst, provider=self.name,
                    warnings=[f"neural_kullanilamiyor: {self._last_error or 'bilinmiyor'}"],
                )
            tokenizer, model = loaded

            import torch

            segments = self._split_segments(text, max_length)
            outputs: List[str] = []
            with torch.no_grad():
                for segment in segments:
                    batch = tokenizer(
                        [segment], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length,
                    )
                    generated = model.generate(
                        **batch, max_length=max_length, num_beams=num_beams,
                        early_stopping=True,
                    )
                    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                    outputs.append(decoded[0] if decoded else segment)

            self._stats["segments"] += len(segments)
            self._stats["success"] += 1
            self._last_error = ""

            return TranslationResult(
                text=" ".join(part.strip() for part in outputs if part is not None).strip(),
                source_lang=src,
                target_lang=dst,
                provider=self.name,
                confidence=0.85,
                warnings=[] if len(segments) == 1 else [f"{len(segments)}_parca_birlestirildi"],
            )
        except Exception as exc:
            self._stats["errors"] += 1
            self._last_error = str(exc)
            logger.warning(f"Sinir ağı çevirisi başarısız ({src}->{dst}): {exc}")
            return make_failed_result(
                text, src, dst, provider=self.name,
                warnings=[f"neural_hatasi: {exc}"],
            )

    @staticmethod
    def _split_segments(text: str, max_length: int) -> List[str]:
        """
        Uzun metni modelin bağlamına sığacak parçalara böler.

        Cümle sınırlarına saygı gösterir; zorunlu kalırsa kelime sınırından böler.
        """
        limit = max(64, int(max_length * 3))  # kaba karakter bütçesi
        if len(text) <= limit:
            return [text]

        sentences = re.split(r"(?<=[.!?…])\s+", text)
        segments: List[str] = []
        current = ""
        for sentence in sentences:
            if not sentence:
                continue
            if current and len(current) + len(sentence) + 1 > limit:
                segments.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
            while len(current) > limit:
                cut = current.rfind(" ", 0, limit)
                if cut <= 0:
                    cut = limit
                segments.append(current[:cut].strip())
                current = current[cut:].strip()
        if current.strip():
            segments.append(current.strip())
        return segments or [text]

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        stats = super().get_stats()
        with self._model_lock:
            loaded = sorted(self._models.keys())
        stats.update({
            "torch": bool(self._torch_available),
            "transformers": bool(self._transformers_available),
            "use_local_model": self._use_local_model,
            "download_allowed": self._download_allowed(),
            "loaded_models": loaded,
            "model_dir": str(Path(self._base_path or ".") / MODEL_CACHE_DIRNAME),
            "last_load_ms": round(float(self._last_load_ms), 1),
            "last_error": self._last_error,
        })
        return stats


# ─── API Çevirici ──────────────────────────────────────────────────────────

class ApiTranslator(BaseTranslator):
    """
    `ApiBridge` üzerinden uzak çeviri servislerini kullanan çevirici.

    Çevrimdışıysa veya anahtar yoksa `available()` False döner ve ağ
    çağrısı yapılmaz.
    """

    name: str = "api"

    def __init__(self, bridge=None, config=None) -> None:
        super().__init__(config)
        self._bridge = bridge

    def initialize(self, config=None, event_bus=None) -> None:
        """Köprüyü bağlar ve başlatır. İdempotenttir."""
        if self._initialized:
            return
        super().initialize(config, event_bus)
        if self._bridge is None:
            try:
                from neurocore.language.api_bridge import ApiBridge
                self._bridge = ApiBridge()
            except Exception as exc:
                logger.warning(f"API köprüsü oluşturulamadı: {exc}")
                return
        if not getattr(self._bridge, "_initialized", False):
            try:
                self._bridge.initialize(config, event_bus)
            except Exception as exc:
                logger.warning(f"API köprüsü başlatılamadı: {exc}")

    @property
    def bridge(self):
        """Kullanılan API köprüsü (None olabilir)."""
        return self._bridge

    def available(self) -> bool:
        """Anahtarı olan ve etkin en az bir sağlayıcı var mı?"""
        if self._bridge is None:
            return False
        try:
            return bool(self._bridge.available_providers())
        except Exception as exc:  # pragma: no cover - savunma
            logger.debug(f"API kullanılabilirlik kontrolü başarısız: {exc}")
            return False

    def supports(self, source_lang: str, target_lang: str) -> bool:
        """Farklı diller ve mevcut bir sağlayıcı varsa destekler."""
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        return bool(src and dst and src != dst and self.available())

    def translate(self, text: str, source_lang: str, target_lang: str, **kw) -> TranslationResult:
        """
        Metni API sağlayıcısıyla çevirir.

        Kwargs:
            provider (str): Belirli sağlayıcı adı
            timeout (float): Saniye cinsinden zaman aşımı
        """
        self._stats["calls"] += 1
        if self._bridge is None:
            self._stats["skipped"] += 1
            return make_failed_result(text, source_lang, target_lang, provider=self.name,
                                      warnings=["api_koprusu_yok"])
        try:
            result = self._bridge.translate(
                text, source_lang, target_lang,
                provider=kw.get("provider"),
                timeout=float(kw.get("timeout", 20.0) or 20.0),
            )
        except Exception as exc:
            self._stats["errors"] += 1
            return make_failed_result(
                text, source_lang, target_lang, provider=self.name,
                warnings=[f"api_hatasi: {exc}"],
            )

        if result.ok:
            self._stats["success"] += 1
        else:
            self._stats["failure"] += 1
        return result


# ─── Zincir (Cascade) Çevirici ─────────────────────────────────────────────

class CascadeTranslator(BaseTranslator):
    """
    Sıralı çok sağlayıcılı çeviri zinciri.

    Sıra:
        identity → cache → terminology (sonuç sonrası geçiş) → neural →
        dictionary → api → passthrough (başarısız)

    Garantiler:
        - **Asla istisna fırlatmaz**; her sağlayıcı try/except içindedir ve
          hata `warnings` listesine eklenir.
        - Her zaman bir `TranslationResult` döner.
        - `language.translated` / `language.translation_failed` olayları yayınlanır.

    Kullanım:
        cascade = CascadeTranslator(store, cache, bridge)
        cascade.initialize(config, event_bus)
        sonuc = cascade.translate("yapay sinir ağı", "tr", "en")
    """

    name: str = "cascade"

    def __init__(
        self,
        store: Optional[TerminologyStore] = None,
        cache: Optional[TranslationCache] = None,
        bridge=None,
        translators: Optional[List[BaseTranslator]] = None,
        config=None,
    ) -> None:
        """Yalnızca durum kurulumu — alt çeviriciler burada kurulur, I/O yok."""
        super().__init__(config)
        self._store = store if store is not None else TerminologyStore()
        self._cache = cache if cache is not None else TranslationCache()

        if bridge is None:
            try:
                from neurocore.language.api_bridge import ApiBridge
                bridge = ApiBridge()
            except Exception as exc:  # pragma: no cover - savunma
                logger.debug(f"API köprüsü tembel oluşturulamadı: {exc}")
                bridge = None
        self._bridge = bridge

        self._identity = IdentityTranslator(config)
        self._dictionary = DictionaryTranslator(self._store, config)
        self._neural = NeuralTranslator(config)
        self._api = ApiTranslator(bridge, config)

        self._chain: List[BaseTranslator] = list(translators) if translators else [
            self._identity, self._neural, self._dictionary, self._api,
        ]

        self._stats.update({
            "identity_hits": 0, "cache_hits": 0, "neural_hits": 0,
            "dictionary_hits": 0, "api_hits": 0, "passthrough": 0,
        })

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config=None, event_bus=None) -> None:
        """Zinciri ve tüm alt bileşenleri başlatır. İdempotenttir."""
        if self._initialized:
            return
        super().initialize(config, event_bus)

        if not getattr(self._store, "_initialized", False):
            try:
                self._store.initialize(config, event_bus)
            except Exception as exc:
                logger.warning(f"Zincir: terim deposu başlatılamadı: {exc}")
        if not getattr(self._cache, "_initialized", False):
            try:
                self._cache.initialize(config, event_bus)
            except Exception as exc:
                logger.warning(f"Zincir: önbellek başlatılamadı: {exc}")

        for translator in self._chain:
            try:
                translator.initialize(config, event_bus)
            except Exception as exc:
                logger.warning(f"Zincir: {translator.name} başlatılamadı: {exc}")

        logger.info(
            f"Çeviri zinciri hazır: {' -> '.join(t.name for t in self._chain)}"
        )

    def shutdown(self) -> None:
        """Zinciri ve alt bileşenleri kapatır. İdempotenttir."""
        if not self._initialized:
            return
        for translator in self._chain:
            try:
                translator.shutdown()
            except Exception as exc:  # pragma: no cover - savunma
                logger.debug(f"{translator.name} kapatma hatası: {exc}")
        try:
            self._cache.shutdown()
        except Exception as exc:  # pragma: no cover - savunma
            logger.debug(f"Önbellek kapatma hatası: {exc}")
        self._initialized = False
        logger.info("Çeviri zinciri kapatıldı")

    # ─── Erişimciler ───────────────────────────────────────────────────

    @property
    def store(self) -> TerminologyStore:
        """Terim deposu."""
        return self._store

    @property
    def cache(self) -> TranslationCache:
        """Çeviri önbelleği."""
        return self._cache

    @property
    def bridge(self):
        """API köprüsü (None olabilir)."""
        return self._bridge

    @property
    def neural(self) -> NeuralTranslator:
        """Yerel sinir ağı çevirici."""
        return self._neural

    @property
    def translators(self) -> List[BaseTranslator]:
        """Zincirdeki çeviriciler."""
        return list(self._chain)

    def available(self) -> bool:
        """Zincirde en az bir gerçek çevirici kullanılabilir mi?"""
        return any(t._safe_available() for t in self._chain)

    def supports(self, source_lang: str, target_lang: str) -> bool:
        """Kimlik dahil her dil çifti için True (fallback her zaman var)."""
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        return bool(src and dst)

    # ─── Çeviri ────────────────────────────────────────────────────────

    def translate(self, text: str, source_lang: str, target_lang: str, **kw) -> TranslationResult:
        """
        Metni zincirden geçirerek çevirir. Asla istisna fırlatmaz.

        Args:
            text: Kaynak metin
            source_lang: Kaynak dil kodu ('' → tr varsayılanı denenmez, 'tr' kabul edilir)
            target_lang: Hedef dil kodu

        Kwargs:
            use_api (bool): False → API sağlayıcıları atlanır
            provider (str): Belirli API sağlayıcısı

        Returns:
            TranslationResult (her koşulda)
        """
        self._stats["calls"] += 1
        src = normalize_language_code(source_lang) or str(source_lang or "").strip().lower()
        dst = normalize_language_code(target_lang) or str(target_lang or "").strip().lower()
        use_api = bool(kw.get("use_api", True))
        warnings: List[str] = []

        try:
            raw_text = "" if text is None else str(text)

            # 1) Kimlik — aynı dil
            if src and dst and src == dst:
                self._stats["identity_hits"] += 1
                self._stats["success"] += 1
                result = TranslationResult(
                    text=raw_text, source_lang=src, target_lang=dst,
                    provider="identity", confidence=1.0,
                )
                self._emit_translated(result, cached=False)
                return result

            if not raw_text.strip():
                self._stats["failure"] += 1
                result = make_failed_result(
                    raw_text, src, dst, provider="failed",
                    warnings=["bos_metin"],
                )
                self._emit_failed(result, "bos_metin")
                return result

            # 2) Önbellek
            cache_hit = self._cache_get(raw_text, src, dst)
            if cache_hit is not None:
                self._stats["cache_hits"] += 1
                self._stats["success"] += 1
                glossary_hits = self._apply_terminology(cache_hit, raw_text, src, dst, warnings)
                self._emit_translated(cache_hit, cached=True)
                if glossary_hits:
                    cache_hit.glossary_hits = {**cache_hit.glossary_hits, **glossary_hits}
                return cache_hit

            # 3) Sağlayıcı zinciri (neural → dictionary → api)
            for translator in self._chain:
                if translator is self._identity:
                    continue
                if translator is self._api and not use_api:
                    warnings.append("api_atlandi")
                    continue
                try:
                    if not translator._safe_available():
                        continue
                    if not translator.supports(src, dst):
                        continue
                except Exception as exc:
                    warnings.append(f"{translator.name}_kontrol_hatasi: {exc}")
                    continue

                try:
                    candidate = translator.translate(raw_text, src, dst, **kw)
                except Exception as exc:
                    self._stats["errors"] += 1
                    warnings.append(f"{translator.name}_hatasi: {exc}")
                    logger.debug(f"{translator.name} çeviri hatası: {exc}")
                    continue

                if not isinstance(candidate, TranslationResult):
                    warnings.append(f"{translator.name}_gecersiz_sonuc")
                    continue

                warnings.extend(candidate.warnings)

                # Sözlük sonucu için kapsama eşiği: düşükse sonraki sağlayıcıya geç
                if candidate.provider == "dictionary" and candidate.confidence < CASCADE_ACCEPT_COVERAGE:
                    self._stats["skipped"] += 1
                    continue

                if candidate.ok or candidate.provider == "dictionary":
                    if candidate.provider == "neural":
                        self._stats["neural_hits"] += 1
                    elif candidate.provider == "dictionary":
                        self._stats["dictionary_hits"] += 1
                    elif candidate.provider == "api":
                        self._stats["api_hits"] += 1

                    if candidate.provider != "dictionary":
                        glossary_hits = self._apply_terminology(candidate, raw_text, src, dst, warnings)
                        candidate.glossary_hits = {**candidate.glossary_hits, **glossary_hits}

                    candidate.warnings = list(dict.fromkeys(warnings))
                    self._cache_put(raw_text, candidate)
                    self._stats["success"] += 1
                    self._emit_translated(candidate, cached=False)
                    return candidate

            # 4) Geri düşüş — kaynak metin işaretlenerek döndürülür
            self._stats["passthrough"] += 1
            self._stats["failure"] += 1
            warnings.append("ceviri_yapilamadi")
            result = make_failed_result(
                raw_text, src, dst, provider="failed", warnings=warnings,
            )
            self._emit_failed(result, "ceviri_yapilamadi")
            return result

        except Exception as exc:  # Son savunma hattı — zincir asla çökmez
            self._stats["errors"] += 1
            self._stats["failure"] += 1
            logger.warning(f"Çeviri zinciri beklenmeyen hata: {exc}")
            fallback = make_failed_result(
                "" if text is None else str(text), src, dst, provider="failed",
                warnings=warnings + [f"zincir_hatasi: {exc}", "ceviri_yapilamadi"],
            )
            self._emit_failed(fallback, "zincir_hatasi")
            return fallback

    # ─── İç Yardımcılar ────────────────────────────────────────────────

    def _cache_get(self, text: str, src: str, dst: str) -> Optional[TranslationResult]:
        """Önbellek okumasını hata-fırlatmadan yapar."""
        try:
            return self._cache.get(text, src, dst)
        except Exception as exc:  # pragma: no cover - savunma
            logger.debug(f"Önbellek okuma hatası: {exc}")
            return None

    def _cache_put(self, source_text: str, result: TranslationResult) -> None:
        """Önbellek yazmasını hata-fırlatmadan yapar."""
        try:
            self._cache.put_for(source_text, result)
        except Exception as exc:  # pragma: no cover - savunma
            logger.debug(f"Önbellek yazma hatası: {exc}")

    def _apply_terminology(
        self,
        result: TranslationResult,
        original_text: str,
        src: str,
        dst: str,
        warnings: List[str],
    ) -> Dict[str, str]:
        """
        Terim sözlüğü ön-geçişini sonuca uygular.

        Çıktı metninde kaynak dilde kalmış bilinen terimler hedef dile çevrilir
        (ör: neural çıktısında 'gradient' → 'gradyan' tersi yönde tutarlılık).

        Returns:
            Uygulanan ek terim eşleşmeleri
        """
        try:
            if not result.text:
                return {}
            new_text, applied = self._store.apply_phrase(result.text, src, dst)
            if applied:
                result.text = new_text
                # Özgün metinde de geçen terimler kapsama sinyali olarak sayılır
                if src and dst:
                    _ = self._store.lookup_phrase(original_text, src, dst)
            return applied
        except Exception as exc:
            warnings.append(f"terim_uygulama_hatasi: {exc}")
            return {}

    def _emit_translated(self, result: TranslationResult, cached: bool) -> None:
        """`language.translated` olayı yayınlar."""
        self._emit("translated", {
            "source_lang": result.source_lang,
            "target_lang": result.target_lang,
            "provider": result.provider,
            "confidence": round(float(result.confidence), 4),
            "cached": bool(cached),
            "glossary_hits": len(result.glossary_hits),
            "warnings": result.warnings[:5],
            "length": len(result.text or ""),
        })

    def _emit_failed(self, result: TranslationResult, reason: str) -> None:
        """`language.translation_failed` olayı yayınlar."""
        self._emit("translation_failed", {
            "source_lang": result.source_lang,
            "target_lang": result.target_lang,
            "reason": reason,
            "warnings": result.warnings[:5],
        })

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        stats = super().get_stats()
        try:
            cache_stats = self._cache.get_stats()
        except Exception:  # pragma: no cover - savunma
            cache_stats = {}
        try:
            store_entries = len(self._store)
        except Exception:  # pragma: no cover - savunma
            store_entries = 0
        stats.update({
            "chain": [t.name for t in self._chain],
            "glossary_entries": store_entries,
            "cache_size": cache_stats.get("size", 0),
            "providers": {
                t.name: t._safe_available() for t in self._chain
            },
        })
        return stats
