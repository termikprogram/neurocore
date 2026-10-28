"""
NeuroCore Dil ve Çeviri Katmanı (L6)
=====================================
Dil algılama, çok sağlayıcılı çeviri, terim sözlüğü, çeviri önbelleği ve
API köprüsünü tek bir katman cephesinde birleştirir.

Yetkinlikler:
    - `detect()`               → dil algılama (yazı sistemi + durak kelime)
    - `translate()`            → tekil çeviri (zincir: kimlik→önbellek→neural→sözlük→api)
    - `translate_batch()`      → toplu çeviri
    - `to_internal()`          → metni iç dile (tr) normalize etme
    - `keywords()`             → durak kelime filtresi + TF ağırlıklı anahtar kelimeler
    - `summarize_extractive()` → cümle puanlamalı özet (Türkçe kısaltma duyarlı)
    - `glossary()`/`add_term()`→ terim sözlüğü erişimi
    - `normalize()`            → NFC, boşluk sadeleştirme, Türkçe i/İ düzeltmesi
    - `health()`               → ağ erişimi YAPMAYAN sağlık raporu (`deep=True` hariç)

Kullanım:
    layer = LanguageLayer()
    layer.initialize(config, event_bus)
    print(layer.detect("Merhaba dünya"))
    print(layer.translate("yapay sinir ağı", "en"))
    layer.shutdown()
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Sequence, Set, Tuple

from neurocore.utils.logger import NeuroCoreLogger
from neurocore.language.api_bridge import ApiBridge
from neurocore.language.detector import (
    LATIN_LANGUAGES,
    LanguageDetector,
    LanguageGuess,
    guess_encoding,
    is_probably_code,
)
from neurocore.language.dictionary import (
    LANGUAGE_LABELS,
    TerminologyStore,
    normalize_language_code,
)
from neurocore.language.translator import (
    CASCADE_ACCEPT_COVERAGE,
    BaseTranslator,
    CascadeTranslator,
    TranslationCache,
    TranslationResult,
)

logger = NeuroCoreLogger.get_logger("language.layer")


# ─── Metin Desenleri ───────────────────────────────────────────────────────

#: Kelime deseni (harf + rakam + iç kesme işareti)
_WORD_RE: Pattern[str] = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
#: Cümle sonu deseni
_SENTENCE_END_RE: Pattern[str] = re.compile(r"([.!?…]+[\"'”’)\]]*\s+)")
#: Cümle ayırıcı (kısa yol)
_SENTENCE_SPLIT_RE: Pattern[str] = re.compile(r"(?<=[.!?…])\s+")
#: Korunacak kısaltmalar (Türkçe + yaygın Latince kısaltmalar)
_ABBREVIATIONS: Tuple[str, ...] = (
    "vb", "vs", "bkz", "örn", "ör", "dr", "prof", "doç", "av", "no", "sf",
    "yy", "mah", "cad", "sok", "apt", "tel", "fak", "üniv", "böl", "vb",
    "e.g", "i.e", "etc", "cf", "al", "fig", "eq", "mr", "mrs", "ms", "st",
)
#: Kısaltma/sayı sonrası noktayı geçici olarak saklayan yer tutucu
_DOT_PLACEHOLDER: str = "\u0001"
#: Sıfır genişlikli ve görünmez karakterler
_ZERO_WIDTH_RE: Pattern[str] = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
#: Boşluk sadeleştirme
_WHITESPACE_RE: Pattern[str] = re.compile(r"[ \t\u00a0\u2000-\u200a]+")
#: Çok satırlı boşluk
_MULTI_NEWLINE_RE: Pattern[str] = re.compile(r"\n{3,}")
#: Türkçe basit gövde indirgeme ekleri (uzundan kısaya sıralı)
_TR_STEM_SUFFIXES: Tuple[str, ...] = (
    "larında", "lerinde", "larından", "lerinden", "lardan", "lerden",
    "ların", "lerin", "lara", "lere", "ları", "leri", "dan", "den",
    "lar", "ler", "da", "de", "ın", "in", "un", "ün", "ı", "i", "u", "ü",
)
#: Özet puanlamasında bu uzunluğun altındaki cümleler cezalandırılır
_MIN_SENTENCE_CHARS: int = 12


# ─── Katman Cephesi ────────────────────────────────────────────────────────

class LanguageLayer:
    """
    Dil ve çeviri katmanı cephesi (L6).

    Alt bileşenleri (algılayıcı, terim deposu, önbellek, API köprüsü,
    çeviri zinciri) tek bir sözleşme altında toplar. `initialize()` tüm
    bağımlılıkları bağlar; ağır iş yapmaz, ağ erişimi denemez.

    Kullanım:
        layer = LanguageLayer()
        layer.initialize(None, None)          # config/event_bus isteğe bağlı
        assert layer.detect("Merhaba")["code"] == "tr"
    """

    name: str = "language"

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — I/O ve ağ erişimi yok."""
        self._initialized: bool = False
        self._config = None
        self._event_bus = None

        self._detector = LanguageDetector()
        self._terminology = TerminologyStore()
        self._cache = TranslationCache()
        self._bridge = ApiBridge()
        self._translator = CascadeTranslator(
            store=self._terminology,
            cache=self._cache,
            bridge=self._bridge,
        )

        self._default_language: str = "tr"
        self._supported_languages: List[str] = ["tr", "en", "de", "fr", "es"]
        self._auto_save_glossary: bool = False

        self._stats: Dict[str, int] = {
            "detect_calls": 0,
            "translate_calls": 0,
            "translate_success": 0,
            "translate_failed": 0,
            "batch_calls": 0,
            "batch_items": 0,
            "to_internal_calls": 0,
            "to_internal_translated": 0,
            "keyword_calls": 0,
            "summary_calls": 0,
            "normalize_calls": 0,
            "sentence_splits": 0,
            "glossary_calls": 0,
            "term_additions": 0,
            "health_checks": 0,
            "deep_health_checks": 0,
            "events": 0,
        }
        self._seed_report: Dict[str, Any] = {}

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config=None, event_bus=None) -> None:
        """
        Katmanı başlat: alt bileşenleri bağla, varsayılan sözlüğü tohumla.

        İdempotenttir: ikinci çağrı no-op'tur. `config is None` güvenlidir
        (tüm varsayılanlar kullanılır).

        Args:
            config: SystemConfig benzeri nesne (None → varsayılanlar)
            event_bus: Olay yolu (None → olay yayınlanmaz)
        """
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus

        language_cfg = getattr(config, "language", None) if config is not None else None
        default_lang = getattr(language_cfg, "default_language", None) if language_cfg is not None else None
        self._default_language = normalize_language_code(default_lang or "tr") or "tr"

        supported = getattr(language_cfg, "supported_languages", None) if language_cfg is not None else None
        if isinstance(supported, (list, tuple)) and supported:
            self._supported_languages = [
                normalize_language_code(code) for code in supported if str(code).strip()
            ]
        else:
            self._supported_languages = ["tr", "en", "de", "fr", "es"]

        # Alt bileşenler
        self._detector.initialize(config, event_bus)
        self._terminology.initialize(config, event_bus)
        self._cache.initialize(config, event_bus)
        self._bridge.initialize(config, event_bus)
        self._translator.initialize(config, event_bus)

        # Sözlük boşsa küratörlü tr↔en teknik sözlüğü tohumla
        seeded = 0
        if len(self._terminology) == 0:
            try:
                seeded = self._terminology.seed_default_bilingual()
                stats = self._terminology.get_stats()
                self._seed_report = {
                    "pairs": seeded,
                    "entries": stats.get("entries", 0),
                    "domains": stats.get("domains", {}),
                }
            except Exception as exc:
                logger.warning(f"Varsayılan sözlük tohumlanamadı: {exc}")

        self._initialized = True
        logger.info(
            f"Dil katmanı hazır: iç dil '{self._default_language}', "
            f"desteklenen {len(self._supported_languages)} dil, "
            f"sözlük {len(self._terminology)} kayıt, "
            f"API sağlayıcıları {self._bridge.available_providers() or 'yok'}"
        )
        self._emit("language.ready", {
            "default_language": self._default_language,
            "supported_languages": self._supported_languages,
            "glossary_entries": len(self._terminology),
            "seeded_pairs": seeded,
            "providers": self._bridge.available_providers(),
        })

    def shutdown(self) -> None:
        """
        Katmanı kapat: alt bileşenleri serbest bırak. İdempotenttir.

        Çeviri önbelleği diske yazılmaya çalışılır (hata toleranslı).
        """
        if not self._initialized:
            return

        try:
            self._cache.save()
        except Exception as exc:  # pragma: no cover - savunma
            logger.debug(f"Önbellek kapatmada kaydedilemedi: {exc}")

        for component in (self._translator, self._bridge, self._cache,
                          self._terminology, self._detector):
            try:
                component.shutdown()
            except Exception as exc:  # pragma: no cover - savunma
                logger.debug(f"Bileşen kapatma hatası ({component.name}): {exc}")

        self._initialized = False
        logger.info("Dil katmanı kapatıldı")

    # ─── Özellikler ────────────────────────────────────────────────────

    @property
    def detector(self) -> LanguageDetector:
        """Dil algılayıcı."""
        return self._detector

    @property
    def terminology(self) -> TerminologyStore:
        """Terim deposu (sözlük)."""
        return self._terminology

    @property
    def cache(self) -> TranslationCache:
        """Çeviri önbelleği."""
        return self._cache

    @property
    def bridge(self) -> ApiBridge:
        """API köprüsü."""
        return self._bridge

    @property
    def translator(self) -> CascadeTranslator:
        """Çeviri zinciri."""
        return self._translator

    @property
    def internal_language(self) -> str:
        """Sistemin iç dili (config.language.default_language)."""
        return self._default_language

    # ─── Dil Algılama ──────────────────────────────────────────────────

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Metnin dilini algılar.

        Args:
            text: Analiz edilecek metin

        Returns:
            LanguageGuess.to_dict() + {'script': str, 'supported': bool,
            'is_code': bool, 'length': int, 'words': int}
        """
        self._stats["detect_calls"] += 1
        try:
            guess: LanguageGuess = self._detector.detect(text)
        except Exception as exc:
            logger.warning(f"Dil algılama hatası: {exc}")
            guess = LanguageGuess("unknown", 0.0, {}, "unknown")

        raw = "" if text is None else str(text)
        payload = guess.to_dict()
        payload.update({
            "script": self._detector.script_of(raw),
            "supported": self._detector.is_supported(guess.code) if guess.code != "unknown" else False,
            "is_code": is_probably_code(raw),
            "length": len(raw),
            "words": len(self.tokenize_words(raw)),
            "internal_language": self._default_language,
        })
        return payload

    # ─── Çeviri ────────────────────────────────────────────────────────

    def translate(
        self,
        text: str,
        target_lang: Optional[str] = None,
        source_lang: Optional[str] = None,
        use_api: bool = True,
    ) -> Dict[str, Any]:
        """
        Metni hedef dile çevirir.

        Kaynak dil verilmezse otomatik algılanır; algılanamazsa iç dil
        varsayılır. Sonuç her koşulda bir sözlüktür (istisna fırlatılmaz).

        Args:
            text: Kaynak metin
            target_lang: Hedef dil kodu (None → config iç dili)
            source_lang: Kaynak dil kodu (None → otomatik algıla)
            use_api: False → API sağlayıcıları atlanır

        Returns:
            TranslationResult.to_dict() + {'detected': {...}, 'auto_detected': bool}
        """
        self._stats["translate_calls"] += 1
        raw_text = "" if text is None else str(text)
        target = normalize_language_code(target_lang or self._default_language) or self._default_language

        detected_payload: Optional[Dict[str, Any]] = None
        auto_detected = False
        source = normalize_language_code(source_lang) if source_lang else ""

        if not source:
            if not raw_text.strip():
                source = self._default_language
            else:
                try:
                    guess = self._detector.detect(raw_text)
                except Exception:  # pragma: no cover - savunma
                    guess = LanguageGuess("unknown", 0.0, {}, "unknown")
                detected_payload = guess.to_dict()
                detected_payload["script"] = self._detector.script_of(raw_text)
                auto_detected = True
                if guess.code and guess.code != "unknown" and guess.confidence > 0.0:
                    source = guess.code
                else:
                    # Algılanamadı → iç dile göre karar ver
                    source = self._default_language

        try:
            result: TranslationResult = self._translator.translate(
                raw_text, source, target, use_api=use_api,
            )
        except Exception as exc:  # Son savunma — zincir zaten korumalı
            self._stats["translate_failed"] += 1
            logger.warning(f"Çeviri hatası: {exc}")
            from neurocore.language.translator import make_failed_result
            result = make_failed_result(
                raw_text, source, target, provider="failed",
                warnings=[f"katman_hatasi: {exc}", "ceviri_yapilamadi"],
            )

        payload = result.to_dict()
        payload.update({
            "detected": detected_payload,
            "auto_detected": auto_detected,
            "requested_target": target,
            "internal_language": self._default_language,
            "was_translated": bool(result.text.strip() and result.text.strip() != raw_text.strip()),
        })

        if result.ok:
            self._stats["translate_success"] += 1
        else:
            self._stats["translate_failed"] += 1
        return payload

    def translate_batch(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: Optional[str] = None,
        use_api: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Birden çok metni sırayla çevirir (önbellek tekrarları hızlandırır).

        Args:
            texts: Metin listesi
            target_lang: Hedef dil kodu
            source_lang: Kaynak dil kodu (None → her metin için otomatik)
            use_api: False → API sağlayıcıları atlanır

        Returns:
            Her metin için bir sonuç sözlüğü (aynı sıradadır)
        """
        self._stats["batch_calls"] += 1
        items: Sequence[Any] = texts if isinstance(texts, (list, tuple)) else [texts]
        results: List[Dict[str, Any]] = []

        # Kaynak dil verilmediyse tüm yığın için tek kez algıla (tutarlılık + hız)
        batch_source = normalize_language_code(source_lang) if source_lang else ""
        if not batch_source and items:
            try:
                joined = " ".join(str(item) for item in items if item)
                guess = self._detector.detect(joined)
                if guess.code and guess.code != "unknown":
                    batch_source = guess.code
            except Exception:  # pragma: no cover - savunma
                batch_source = ""

        for item in items:
            self._stats["batch_items"] += 1
            try:
                results.append(self.translate(
                    item, target_lang, batch_source or None, use_api=use_api,
                ))
            except Exception as exc:  # pragma: no cover - savunma
                from neurocore.language.translator import make_failed_result
                failed = make_failed_result(
                    "" if item is None else str(item), batch_source or "unknown",
                    normalize_language_code(target_lang) or target_lang,
                    provider="failed", warnings=[f"yigin_hatasi: {exc}", "ceviri_yapilamadi"],
                )
                results.append(failed.to_dict())
        return results

    def to_internal(self, text: str) -> Dict[str, Any]:
        """
        Metni iç dile (varsayılan `tr`) normalize eder.

        Metin zaten iç dilde ise değiştirilmez.

        Args:
            text: Kaynak metin

        Returns:
            {'text': str, 'original': str, 'source_lang': str,
             'target_lang': str, 'translated': bool, 'detected': dict|None,
             'confidence': float, 'provider': str}
        """
        self._stats["to_internal_calls"] += 1
        raw_text = "" if text is None else str(text)
        target = self._default_language

        detected_payload: Optional[Dict[str, Any]] = None
        try:
            guess = self._detector.detect(raw_text)
            detected_payload = guess.to_dict()
            detected_payload["script"] = self._detector.script_of(raw_text)
            source = guess.code if guess.code and guess.code != "unknown" else target
        except Exception:  # pragma: no cover - savunma
            guess = LanguageGuess("unknown", 0.0, {}, "unknown")
            source = target

        if source == target or not raw_text.strip():
            return {
                "text": raw_text,
                "original": raw_text,
                "source_lang": source,
                "target_lang": target,
                "translated": False,
                "detected": detected_payload,
                "confidence": 1.0 if source == target and raw_text.strip() else 0.0,
                "provider": "identity",
                "warnings": [],
            }

        result = self.translate(raw_text, target, source, use_api=True)
        if result.get("ok") and result.get("was_translated"):
            self._stats["to_internal_translated"] += 1
        return {
            "text": result.get("text", raw_text),
            "original": raw_text,
            "source_lang": result.get("source_lang", source),
            "target_lang": target,
            "translated": bool(result.get("was_translated")),
            "detected": detected_payload,
            "confidence": float(result.get("confidence", 0.0) or 0.0),
            "provider": str(result.get("provider", "failed")),
            "warnings": list(result.get("warnings", []) or []),
        }

    # ─── Sözlük ────────────────────────────────────────────────────────

    def glossary(self, source_lang: str, target_lang: str, limit: int = 200) -> Dict[str, str]:
        """
        Bir dil çiftinin terim sözlüğünü döndürür.

        Args:
            source_lang: Kaynak dil kodu
            target_lang: Hedef dil kodu
            limit: En fazla terim sayısı

        Returns:
            {terim: çeviri} sözlüğü (güvene göre sıralı)
        """
        self._stats["glossary_calls"] += 1
        try:
            rows = self._terminology.terms_for(source_lang, target_lang, limit=limit)
        except Exception as exc:
            logger.warning(f"Sözlük okunamadı ({source_lang}->{target_lang}): {exc}")
            return {}
        return {str(row["term"]): str(row["translation"]) for row in rows}

    def add_term(
        self,
        term: str,
        translation: str,
        source_lang: str = "tr",
        target_lang: str = "en",
        domain: str = "user",
    ) -> Dict[str, Any]:
        """
        Sözlüğe yeni terim ekler ve (mümkünse) diske yazar.

        Args:
            term: Kaynak dildeki terim
            translation: Hedef dildeki karşılığı
            source_lang: Kaynak dil kodu
            target_lang: Hedef dil kodu
            domain: Alan etiketi

        Returns:
            {'ok': bool, 'entry': dict|None, 'saved': str, 'error': str}
        """
        src = normalize_language_code(source_lang) or "tr"
        dst = normalize_language_code(target_lang) or "en"
        try:
            self._terminology.add(
                term, translation, src, dst, domain=domain or "user", confidence=0.9,
            )
        except Exception as exc:
            logger.warning(f"Terim eklenemedi ({term!r}): {exc}")
            return {"ok": False, "entry": None, "saved": "", "error": str(exc)}

        self._stats["term_additions"] += 1
        entry = self._terminology.lookup(term, src, dst)
        saved = ""
        if self._auto_save_glossary:
            try:
                saved = self._terminology.save()
            except Exception as exc:  # pragma: no cover - savunma
                logger.debug(f"Sözlük kaydedilemedi: {exc}")

        self._emit("language.term_added", {
            "term": term, "translation": translation,
            "src": src, "dst": dst, "domain": domain,
        })
        return {"ok": True, "entry": entry, "saved": saved, "error": ""}

    def save_glossary(self) -> str:
        """Sözlüğü diske yazar ve dosya yolunu döndürür."""
        try:
            return self._terminology.save()
        except Exception as exc:
            logger.warning(f"Sözlük kaydedilemedi: {exc}")
            return ""

    # ─── Anahtar Kelimeler ─────────────────────────────────────────────

    def keywords(self, text: str, top_k: int = 12) -> List[str]:
        """
        Metnin anahtar kelimelerini çıkarır.

        Yaklaşım: algılanan dilin durak kelimeleri (ve İngilizce) çıkarılır,
        kalan kelimeler TF ile ağırlıklandırılır, uzun kelimeler hafifçe
        öne alınır. Türkçe için basit gövde indirgeme uygulanır.

        Args:
            text: Analiz edilecek metin
            top_k: Döndürülecek en fazla kelime

        Returns:
            Anahtar kelimeler (ağırlığa göre azalan)
        """
        self._stats["keyword_calls"] += 1
        if not text or not str(text).strip():
            return []
        raw = str(text)

        try:
            guess = self._detector.detect(raw)
            language = guess.code if guess.code != "unknown" else ""
        except Exception:  # pragma: no cover - savunma
            language = ""

        stopwords: Set[str] = set(self._detector.stopword_set("en"))
        if language:
            stopwords |= set(self._detector.stopword_set(language))
        else:
            for code in LATIN_LANGUAGES:
                stopwords |= set(self._detector.stopword_set(code))
        stopwords |= {"bir", "bu", "ve", "ile", "için"}

        trimmed: List[str] = []
        for token in self.tokenize_words(raw):
            if len(token) < 2:
                continue
            trimmed.append(token)

        # Gövde indirgeme (yalnızca gruplama için; çıktı özgün biçimdir)
        stem_map: Dict[str, str] = {}
        display: Dict[str, Counter] = {}
        for token in trimmed:
            lowered = token.lower()
            if lowered in stopwords:
                continue
            stem = self._stem_tr(token) if language == "tr" else lowered
            if len(stem) < 2:
                continue
            stem_map.setdefault(stem, token)
            display.setdefault(stem, Counter())[token] += 1

        if not display:
            return []

        weights: Dict[str, float] = {}
        for stem, forms in display.items():
            total = sum(forms.values())
            weights[stem] = total * (1.0 + 0.08 * min(6, len(stem)))
        ranked = sorted(weights, key=lambda s: (-weights[s], s))
        return [stem_map[stem] for stem in ranked[:max(1, int(top_k))]]

    # ─── Cümle İşlemleri ───────────────────────────────────────────────

    def sentence_split(self, text: str) -> List[str]:
        """
        Metni cümlelere ayırır (Türkçe kısaltma ve ondalık sayı duyarlı).

        "vb.", "vs.", "bkz.", "Dr." gibi kısaltmalar ve "3.14" gibi ondalık
        sayılar cümle sonu sayılmaz.

        Args:
            text: Metin

        Returns:
            Boş olmayan cümleler (özgün sırada)
        """
        self._stats["sentence_splits"] += 1
        if not text or not str(text).strip():
            return []

        raw = str(text).strip()
        protected = raw

        # 1) Ondalık sayılar: "3.14" → nokta korunur
        protected = re.sub(r"(?<=\d)\.(?=\d)", _DOT_PLACEHOLDER, protected)

        # 2) Kısaltmalar: "vb." / "Dr." / "e.g." → nokta korunur
        abbrev_pattern = r"(?<![^\W\d_])(?:" + "|".join(
            re.escape(item) for item in sorted(set(_ABBREVIATIONS), key=len, reverse=True)
        ) + r")\.(?=\s|$)"
        protected = re.sub(
            abbrev_pattern,
            lambda match: match.group(0)[:-1] + _DOT_PLACEHOLDER,
            protected,
            flags=re.IGNORECASE | re.UNICODE,
        )

        # 3) Cümle sonlarında böl
        parts = _SENTENCE_SPLIT_RE.split(protected)

        sentences: List[str] = []
        for part in parts:
            restored = part.replace(_DOT_PLACEHOLDER, ".").strip()
            if restored:
                sentences.append(restored)
        return sentences

    def tokenize_words(self, text: str) -> List[str]:
        """
        Metni kelimelere ayırır (rakam/noktalama atılır, iç kesme korunur).

        Args:
            text: Metin

        Returns:
            Kelime listesi (özgün sırada)
        """
        if not text:
            return []
        return [match.group(0) for match in _WORD_RE.finditer(str(text))]

    def summarize_extractive(self, text: str, max_sentences: int = 3) -> str:
        """
        Cümle puanlamalı (extractive) özet üretir.

        Puan = kelime frekansı toplamı × konum avantajı × uzunluk cezası.
        Seçilen cümleler **özgün sırada** birleştirilir.

        Args:
            text: Özetlenecek metin
            max_sentences: En fazla cümle sayısı

        Returns:
            Özet metni (girdi boşsa boş string)
        """
        self._stats["summary_calls"] += 1
        if not text or not str(text).strip():
            return ""

        sentences = self.sentence_split(text)
        if not sentences:
            return ""
        limit = max(1, int(max_sentences))
        if len(sentences) <= limit:
            return " ".join(sentences)

        try:
            guess = self._detector.detect(str(text))
            language = guess.code if guess.code != "unknown" else ""
        except Exception:  # pragma: no cover - savunma
            language = ""

        stopwords: Set[str] = set(self._detector.stopword_set("en"))
        if language:
            stopwords |= set(self._detector.stopword_set(language))
        else:
            for code in LATIN_LANGUAGES:
                stopwords |= set(self._detector.stopword_set(code))

        # Belge geneli kelime frekansları
        frequencies: Counter = Counter()
        sentence_tokens: List[List[str]] = []
        for sentence in sentences:
            tokens = [
                token.lower() for token in self.tokenize_words(sentence)
                if len(token) > 2 and token.lower() not in stopwords
            ]
            sentence_tokens.append(tokens)
            frequencies.update(tokens)

        if not frequencies:
            return " ".join(sentences[:limit])

        max_freq = max(frequencies.values())
        scored: List[Tuple[float, int, str]] = []
        total = len(sentences)

        for index, sentence in enumerate(sentences):
            tokens = sentence_tokens[index]
            if tokens:
                base = sum(frequencies[token] / max_freq for token in tokens) / math.sqrt(len(tokens))
            else:
                base = 0.0

            # Konum avantajı: ilk cümleler hafifçe öne çıkar
            position_bonus = 1.0 + 0.20 * (1.0 - (index / max(1, total - 1)))
            # Uzunluk cezası: çok kısa cümleler özet taşımaz
            length_penalty = 1.0 if len(sentence) >= _MIN_SENTENCE_CHARS else 0.55

            scored.append((base * position_bonus * length_penalty, index, sentence))

        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = scored[:limit]
        chosen.sort(key=lambda item: item[1])  # özgün sıraya dön
        return " ".join(item[2] for item in chosen).strip()

    # ─── Normalizasyon ─────────────────────────────────────────────────

    def normalize(self, text: str, lower: bool = False) -> str:
        """
        Metni normalize eder.

        İşlemler: NFC, sıfır genişlikli karakter temizliği, boşluk
        sadeleştirme, satır sonu düzenleme, (isteğe bağlı) Türkçe
        noktalı/noktasız i düzeltmeli küçük harfe çevirme.

        Args:
            text: Metin
            lower: True → Türkçe farkındalı küçük harfe çevir

        Returns:
            Normalize edilmiş metin
        """
        self._stats["normalize_calls"] += 1
        if text is None:
            return ""

        raw = str(text)
        try:
            normalized = unicodedata.normalize("NFC", raw)
        except (TypeError, ValueError):  # pragma: no cover - savunma
            normalized = raw

        normalized = _ZERO_WIDTH_RE.sub("", normalized)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = _WHITESPACE_RE.sub(" ", normalized)
        normalized = re.sub(r" *\n *", "\n", normalized)
        normalized = _MULTI_NEWLINE_RE.sub("\n\n", normalized).strip()

        if lower:
            normalized = self._turkish_lower(normalized)
        return normalized

    @staticmethod
    def _turkish_lower(text: str) -> str:
        """
        Türkçe farkındalı küçük harfe çevirir.

        'I' → 'ı', 'İ' → 'i' eşlemesi korunur; diğer karakterler standart
        Unicode küçük harfe iner.
        """
        return text.replace("I", "\u0131").replace("\u0130", "i").lower()

    @staticmethod
    def _turkish_upper(text: str) -> str:
        """
        Türkçe farkındalı büyük harfe çevirir.

        'i' → 'İ', 'ı' → 'I' eşlemesi korunur.
        """
        return text.replace("i", "\u0130").replace("\u0131", "I").upper()

    @staticmethod
    def _stem_tr(word: str) -> str:
        """
        Türkçe için basit gövde indirgeme (yalnızca gruplama amaçlı).

        Ekler: -lar/-ler, -in/-ın, -da/-de, -den/-dan, -ı/-i/-u/-ü
        Kök en az 3 karakter kalmalıdır.
        """
        lowered = word.lower()
        for suffix in _TR_STEM_SUFFIXES:
            if lowered.endswith(suffix) and len(lowered) - len(suffix) >= 3:
                return lowered[: len(lowered) - len(suffix)]
        return lowered

    # ─── Sağlık ────────────────────────────────────────────────────────

    def health(self, deep: bool = False) -> Dict[str, Any]:
        """
        Katman sağlık raporu.

        **Varsayılan olarak ağ erişimi YAPMAZ.** `deep=True` verildiğinde
        `bridge.connectivity()` çağrılır (60 sn önbellekli) ve gerçek
        çevrimiçi durumu raporlanır.

        Args:
            deep: True → ağ bağlantısını da ölç

        Returns:
            {'ok': bool, 'providers': [...], 'neural_available': bool,
             'glossary_size': int, 'online': bool, 'issues': [...], ...}
        """
        self._stats["health_checks"] += 1
        issues: List[str] = []

        if not self._initialized:
            issues.append("katman_baslatilmadi")

        components = {
            "detector": self._detector,
            "terminology": self._terminology,
            "cache": self._cache,
            "bridge": self._bridge,
            "translator": self._translator,
        }
        component_state = {
            key: bool(getattr(component, "_initialized", False))
            for key, component in components.items()
        }
        for key, ready in component_state.items():
            if not ready:
                issues.append(f"{key}_hazir_degil")

        try:
            providers = self._bridge.available_providers()
        except Exception as exc:  # pragma: no cover - savunma
            providers = []
            issues.append(f"saglayici_kontrol_hatasi: {exc}")

        try:
            neural_available = bool(self._translator.neural._safe_available())
        except Exception:  # pragma: no cover - savunma
            neural_available = False

        try:
            glossary_size = len(self._terminology)
        except Exception:  # pragma: no cover - savunma
            glossary_size = 0

        if glossary_size == 0:
            issues.append("sozluk_bos")
        if not providers:
            issues.append("api_saglayici_yok")

        online: Optional[bool] = None
        connectivity: Dict[str, Any] = {}
        if deep:
            self._stats["deep_health_checks"] += 1
            try:
                connectivity = self._bridge.connectivity()
                online = bool(connectivity.get("online", False))
                if not online:
                    issues.append("cevrimdisi")
            except Exception as exc:  # pragma: no cover - savunma
                issues.append(f"baglanti_kontrol_hatasi: {exc}")
                online = False

        try:
            cache_size = self._cache.size()
        except Exception:  # pragma: no cover - savunma
            cache_size = 0

        # Zincir en azından kimlik + sözlük ile kısmi çeviri yapabilir
        functional = bool(component_state.get("translator", False))
        ok = self._initialized and component_state.get("detector", False) and functional

        return {
            "ok": bool(ok),
            "name": self.name,
            "initialized": self._initialized,
            "deep": bool(deep),
            "default_language": self._default_language,
            "supported_languages": list(self._supported_languages),
            "providers": providers,
            "neural_available": neural_available,
            "glossary_size": glossary_size,
            "cache_size": cache_size,
            "online": online,
            "connectivity": connectivity,
            "components": component_state,
            "issues": issues,
        }

    # ─── İç Yardımcılar ────────────────────────────────────────────────

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Olay yayınlar (bus yoksa sessiz geçer)."""
        if not self._event_bus:
            return
        try:
            self._stats["events"] += 1
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover - savunma
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def languages(self) -> List[Dict[str, str]]:
        """Desteklenen dillerin kod + etiket listesi."""
        codes = list(dict.fromkeys(
            self._supported_languages + sorted(self._detector.profile_summary())
        ))
        return [
            {"code": code, "label": LANGUAGE_LABELS.get(code, code)}
            for code in codes
        ]

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        def _safe(factory, default):
            try:
                return factory()
            except Exception:  # pragma: no cover - savunma
                return default

        terminology = _safe(self._terminology.get_stats, {})
        cache_stats = _safe(self._cache.get_stats, {})
        translator_stats = _safe(self._translator.get_stats, {})
        detector_stats = _safe(self._detector.get_stats, {})
        bridge_stats = _safe(self._bridge.get_stats, {})

        return {
            "name": self.name,
            "initialized": self._initialized,
            "default_language": self._default_language,
            "supported_languages": list(self._supported_languages),
            "glossary_entries": terminology.get("entries", 0),
            "glossary_domains": terminology.get("domains", {}),
            "glossary_pairs": terminology.get("pairs", {}),
            "glossary_path": terminology.get("path", ""),
            "cache": {
                "size": cache_stats.get("size", 0),
                "max_size": cache_stats.get("max_size", 0),
                "hit_rate": cache_stats.get("hit_rate", 0.0),
                "hits": cache_stats.get("hits", 0),
                "misses": cache_stats.get("misses", 0),
            },
            "translator": {
                "chain": translator_stats.get("chain", []),
                "providers": translator_stats.get("providers", {}),
                "calls": translator_stats.get("calls", 0),
                "success": translator_stats.get("success", 0),
                "failure": translator_stats.get("failure", 0),
                "identity_hits": translator_stats.get("identity_hits", 0),
                "cache_hits": translator_stats.get("cache_hits", 0),
                "neural_hits": translator_stats.get("neural_hits", 0),
                "dictionary_hits": translator_stats.get("dictionary_hits", 0),
                "api_hits": translator_stats.get("api_hits", 0),
                "passthrough": translator_stats.get("passthrough", 0),
            },
            "detector": {
                "profiles": detector_stats.get("profiles", 0),
                "total_stopwords": detector_stats.get("total_stopwords", 0),
                "last_guess": detector_stats.get("last_guess"),
                "detect_calls": detector_stats.get("detect_calls", 0),
            },
            "api": {
                "available": bridge_stats.get("available", []),
                "api_fallback": bridge_stats.get("api_fallback", True),
                "http_requests": bridge_stats.get("http_requests", 0),
                "http_errors": bridge_stats.get("http_errors", 0),
                "last_provider": bridge_stats.get("last_provider", ""),
            },
            "seed": self._seed_report,
            **self._stats,
        }
