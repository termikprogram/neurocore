"""
NeuroCore Bilgi Sentezleyici (L8 · Evolution)
===============================================
Birden çok notu birleştirip **daha yüksek seviyeli** yeni bir not üretir:

    Özet → Ana Noktalar → Detay → Çelişkiler → Kaynaklar → İlişkili

Ayrıca gerçek çelişki tespiti yapar:

    - Aynı konu hakkında **sayısal** iddiaların uyuşmaması
      ("GPU sıcaklığı 80 dereceyi aşarsa..." ile "...95 dereceyi aşarsa...")
    - Açık olumsuzlama çiftleri ("X artırır" ↔ "X azaltır")
    - Karşıt kelime çiftleri (artır/azalt, yükselt/düşür, always/never, ...)

Notlar düz dict olarak gelir; **girdi asla değiştirilmez**. Çıktı Obsidian
uyumlu markdown gövdesi ve `[[id]]` bağlantılarıdır. Torch kullanılmaz;
benzerlik ölçümleri token/kosinüs tabanlıdır.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("evolution.synthesizer")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: Markdown bölüm başlıkları
SECTION_SUMMARY = "## Özet"
SECTION_POINTS = "## Ana Noktalar"
SECTION_DETAIL = "## Detay"
SECTION_CONTRADICTIONS = "## Çelişkiler"
SECTION_SOURCES = "## Kaynaklar"
SECTION_RELATED = "## İlişkili"

#: Sentez için gereken en az not sayısı
MIN_NOTES = 2

#: Güven üst sınırı
CONFIDENCE_CAP = 0.9

#: Birleştirmeye değer eşiği
MERGE_WORTHY = 0.75

#: Görmezden gelinecek çok kısa kelimeler
_MIN_TERM_LEN = 3

_WORD_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü_]+")
_WS_RE = re.compile(r"\s+")
_SENT_RE = re.compile(r"(?<=[.!?…])\s+")
_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
_MARKDOWN_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
_HEADING_STRIP_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)


# ─── Karşıtlık Sözlükleri ──────────────────────────────────────────────────

#: Karşıt fiil/sıfat çiftleri (stem biçiminde)
OPPOSITE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("artır", "azalt"),
    ("yükselt", "düşür"),
    ("hızlı", "yavaş"),
    ("her zaman", "asla"),
    ("always", "never"),
    ("increase", "decrease"),
    ("enable", "disable"),
    ("aç", "kapat"),
    ("etkinleştir", "devre dışı"),
    ("iyileştir", "kötüleştir"),
    ("güçlendir", "zayıflat"),
    ("before", "after"),
    ("önce", "sonra"),
    ("allow", "deny"),
    ("izin ver", "yasakla"),
)

#: Açık olumsuzlama farkının anlamlı olduğu çiftler (eylem tabanlı olanlar)
NEGATION_SENSITIVE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("artır", "azalt"),
    ("yükselt", "düşür"),
    ("increase", "decrease"),
    ("enable", "disable"),
    ("etkinleştir", "devre dışı"),
)

#: Olumsuzlama işaretleri
NEGATION_MARKERS: Tuple[str, ...] = ("değil", "yok", "asla", "not", "never", "no")

#: Bilinen birimler (ham → kanonik)
_UNIT_MAP: Dict[str, str] = {
    "derece": "derece", "degrees": "derece", "degree": "derece", "°c": "derece", "c": "derece",
    "%": "yuzde", "yüzde": "yuzde", "percent": "yuzde", "percentage": "yuzde",
    "gb": "gb", "mb": "mb", "kb": "kb", "tb": "tb", "gib": "gb", "mib": "mb",
    "ghz": "ghz", "mhz": "mhz", "hz": "hz",
    "watt": "watt", "w": "watt", "kw": "kw",
    "saniye": "saniye", "second": "saniye", "seconds": "saniye", "sn": "saniye", "sec": "saniye",
    "dakika": "dakika", "minute": "dakika", "minutes": "dakika", "dk": "dakika", "min": "dakika",
    "saat": "saat", "hour": "saat", "hours": "saat", "h": "saat",
    "gün": "gun", "gun": "gun", "day": "gun", "days": "gun",
    "yıl": "yil", "yil": "yil", "year": "yil", "years": "yil", "sene": "yil",
    "ms": "ms", "milisaniye": "ms", "millisecond": "ms",
    "ns": "ns", "µs": "us", "us": "us",
    "kat": "kat", "x": "kat", "times": "kat",
    "bit": "bit", "byte": "byte", "b": "byte",
    "kwh": "kwh", "wh": "wh", "joule": "joule", "j": "joule",
    "volt": "volt", "v": "volt", "amper": "amper", "a": "amper",
    "rpm": "rpm", "fps": "fps", "dpi": "dpi", "ppi": "ppi",
    "çekirdek": "cekirdek", "core": "cekirdek", "cores": "cekirdek",
    "token": "token", "parametre": "parametre", "parameter": "parametre",
    "katman": "katman", "layer": "katman", "layers": "katman",
    "uzman": "uzman", "expert": "uzman", "experts": "uzman",
}

#: Türkçe gövde son ekleri (çelişki tespitinde kaba normalleştirme)
_TR_SUFFIXES: Tuple[str, ...] = (
    "larında", "lerinde", "larındaki", "lerindeki", "ması", "mesi", "dığı",
    "diği", "duğu", "düğü", "larak", "lardan", "lerden", "ların", "lerin",
    "lara", "lere", "ları", "leri", "daki", "deki", "dan", "den", "tan",
    "ten", "lar", "ler", "dır", "dir", "dur", "dür", "tır", "tir", "sı",
    "si", "su", "sü", "ın", "in", "un", "ün", "ı", "i", "u", "ü", "da",
    "de", "ta", "te", "a", "e",
)

#: Türkçe yapım ekleri (fiil → edilgen/ettirgen biçimler: "artırılmalıdır")
_DERIVATIONAL_SUFFIXES: Tuple[str, ...] = (
    "ılmalıdır", "ilmelidir", "ulmalıdır", "ülmelidir", "malıdır", "melidir",
    "ılmalı", "ilmelidir", "ılır", "ilir", "ulur", "ülür", "ıyor", "iyor",
    "uyor", "üyor", "acak", "ecek", "malı", "meli", "arak", "erek", "dır",
    "dir", "mış", "miş", "muş", "müş", "ıldı", "ildi",
)


# ─── Metin Yardımcıları ────────────────────────────────────────────────────

def _norm_word(text: Any) -> str:
    """Kelimeyi küçük harfe indirir (Türkçe uyumlu, aksanlar korunur)."""
    return unicodedata.normalize("NFKC", str(text or "")).lower()


def _fold(text: Any) -> str:
    """Aksanları sadeleştirir: 'sıcaklığı' → 'sicakligi' (karşılaştırma için)."""
    lowered = _norm_word(text)
    table = str.maketrans({
        "ı": "i", "İ": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c",
        "â": "a", "î": "i", "û": "u",
    })
    return lowered.translate(table)


def stem(token: str) -> str:
    """Kaba Türkçe gövde indirgemesi (yalnızca karşılaştırma amaçlı)."""
    word = _fold(token)
    if len(word) <= 4:
        return word
    for suffix in _TR_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)]
    return word


def _tokens(text: Any) -> List[str]:
    """Kelime listesi."""
    return _WORD_RE.findall(str(text or ""))


def _terms(text: Any, min_len: int = _MIN_TERM_LEN) -> List[str]:
    """Gövde indirgenmiş içerik terimleri."""
    return [stem(t) for t in _tokens(text) if len(t) >= min_len]


def _term_set(text: Any, min_len: int = _MIN_TERM_LEN) -> Set[str]:
    """Gövde indirgenmiş terim kümesi."""
    return set(_terms(text, min_len))


def _word_stems(text: Any, min_len: int = 4) -> Set[str]:
    """
    Kelime bazlı gövde kümesi (karşıt fiil eşleştirmesi için).

    Türkçe kelimeler üretken ekler alır ("artır" → "artırılmalıdır"); bu yüzden
    yalnızca gövde indirgemesi yetmez. Ek temizliğinden sonra kalan biçimler
    kümesi döndürülür ve karşılaştırmalarda **önek eşleşmesi** kullanılır.
    """
    stems: Set[str] = set()
    for token in _tokens(text):
        if len(token) < min_len:
            continue
        folded = _fold(token)
        stems.add(stem(folded))
        trimmed = folded
        for suffix in _DERIVATIONAL_SUFFIXES:
            if trimmed.endswith(suffix) and len(trimmed) - len(suffix) >= 3:
                trimmed = trimmed[: len(trimmed) - len(suffix)]
                break
        stems.add(stem(trimmed))
    return {s for s in stems if len(s) >= 3}


def _matches_stem(needle_stem: str, stems: Set[str], min_prefix: int = 4) -> bool:
    """`needle_stem` kümedeki bir gövdeyle eşleşiyor mu (birebir ya da önek)?"""
    if not needle_stem:
        return False
    if needle_stem in stems:
        return True
    if len(needle_stem) < min_prefix:
        return False
    return any(candidate.startswith(needle_stem) for candidate in stems)


def _has_negation(text: Any) -> bool:
    """Metinde açık olumsuzluk var mı (kelime ya da Türkçe olumsuz fiil eki)?"""
    lowered = _norm_word(text)
    padded = f" {lowered} "
    if any(f" {marker} " in padded for marker in NEGATION_MARKERS):
        return True
    return bool(re.search(r"\b\w{3,}(?:maz|mez|madı|medi|mıyor|miyor|muyor|müyor)\b", lowered))


def _sentences(text: Any) -> List[str]:
    """Metni cümlelere böler; markdown başlıkları ve boş satırlar atılır."""
    cleaned = _MARKDOWN_HEADING_RE.sub("", str(text or ""))
    cleaned = cleaned.replace("\r", " ")
    parts = [p.strip() for p in _SENT_RE.split(_WS_RE.sub(" ", cleaned))]
    return [p for p in parts if len(p) > 2]


def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    """Jaccard benzerliği."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _cosine_terms(a: Set[str], b: Set[str]) -> float:
    """Terim kümeleri arasında kosinüs benzerliği (ikili ağırlık)."""
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    return overlap / ((len(a) ** 0.5) * (len(b) ** 0.5))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Değeri [low, high] aralığına sıkıştır."""
    return max(low, min(high, float(value)))


def _note_field(note: Any, key: str, default: Any = None) -> Any:
    """Not dict'inden güvenli alan okuma."""
    if isinstance(note, dict):
        value = note.get(key, default)
        return default if value is None else value
    return default


def _note_id(note: Any, index: int = 0) -> str:
    """Not kimliği (yoksa deterministik sentetik kimlik)."""
    value = _note_field(note, "id")
    if value:
        return str(value)
    title = str(_note_field(note, "title", "") or "")
    if not title:
        return f"note_{index}"
    digest = hashlib.blake2b(title.encode("utf-8"), digest_size=4).hexdigest()
    return f"note_{digest}"


def _note_tag_set(note: Any) -> Set[str]:
    """Notun etiket kümesi (küçük harf)."""
    tags = _note_field(note, "tags", []) or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    return {_norm_word(t) for t in tags if str(t).strip()}


def _clean_body(note: Any) -> str:
    """Not gövdesini markdown başlıklarından arındırır."""
    body = str(_note_field(note, "body", "") or "")
    return _HEADING_STRIP_RE.sub("", body).strip()


# ─── Sayısal İddia Ayrıştırma ──────────────────────────────────────────────

def _normalize_unit(raw: str) -> Optional[str]:
    """Ham birim metnini kanonik biçime çevirir (bilinmiyorsa None)."""
    token = _norm_word(raw).strip().strip(".,;:")
    if not token:
        return None
    if token in _UNIT_MAP:
        return _UNIT_MAP[token]
    folded = _fold(token)
    if folded in _UNIT_MAP:
        return _UNIT_MAP[folded]
    if folded.startswith("%") or folded.endswith("%"):
        return "yuzde"
    for prefix in ("derece", "saniye", "dakika", "saat", "yil", "gun", "kat"):
        if folded.startswith(prefix):
            return prefix
    return None


def _parse_value(raw: str) -> Optional[float]:
    """Sayı token'ını float'a çevirir ('1.5', '1,5' desteklenir)."""
    token = str(raw or "").strip()
    if not token:
        return None
    if "," in token and "." in token:
        token = token.replace(".", "").replace(",", ".")
    elif "," in token:
        token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _numeric_assertions(text: Any) -> List[Tuple[str, float, str, List[str], str]]:
    """
    Metinden sayısal iddiaları çıkarır.

    Returns:
        `(özne, değer, kanonik_birim, bağlam_terimleri, cümle)` demetleri.
        `bağlam_terimleri`, sayının bulunduğu cümlede sayıdan **önce** geçen
        içerik terimleridir; özne karşılaştırmasında kullanılır.
    """
    assertions: List[Tuple[str, float, str, List[str], str]] = []

    for sentence in _sentences(text):
        words = _tokens(sentence)
        prefixes: List[str] = []
        for position, word in enumerate(words):
            if _NUMBER_RE.match(word):
                value = _parse_value(word)
                if value is None:
                    continue
                unit: Optional[str] = None
                for offset in (1, 2):
                    if position + offset < len(words):
                        unit = _normalize_unit(words[position + offset])
                        if unit:
                            break
                if not unit:
                    continue
                context = prefixes[-4:]
                subject = " ".join(context) or sentence[:48]
                assertions.append((subject, value, unit, list(context), sentence))
            else:
                stripped = _norm_word(word)
                if len(stripped) >= _MIN_TERM_LEN or stripped in ("x", "gb", "mb"):
                    prefixes.append(stem(stripped))

    return assertions


def _subject_overlap(a: Sequence[str], b: Sequence[str]) -> List[str]:
    """İki özne bağlamının paylaştığı terimler."""
    return sorted(set(a) & set(b))


# ─── Sentez Sonucu ─────────────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    """
    Sentezlenmiş bilgi notu.

    Attributes:
        title: Not başlığı
        body: Obsidian uyumlu markdown gövdesi
        tags: Etiketler
        confidence: Güven (0..1)
        sources: Kaynak referansları
        kind: Not türü (`concept`, `synthesis`, `comparison`, ...)
        rationale: Neden bu not üretildi
        contradictions: Tespit edilen çelişki açıklamaları
    """

    title: str
    body: str
    tags: List[str]
    confidence: float
    sources: List[str]
    kind: str = "concept"
    rationale: str = ""
    contradictions: List[str] = field(default_factory=list)

    def to_note_payload(self) -> Dict[str, Any]:
        """Kasa not yüküne (payload) çevirir."""
        return {
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "confidence": round(float(self.confidence), 4),
            "sources": list(self.sources),
            "kind": self.kind,
            "metadata": {
                "synthesized": True,
                "rationale": self.rationale,
                "contradiction_count": len(self.contradictions),
                "source_count": len(self.sources),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dict."""
        return {
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "confidence": round(float(self.confidence), 4),
            "sources": list(self.sources),
            "kind": self.kind,
            "rationale": self.rationale,
            "contradictions": list(self.contradictions),
        }


# ─── Bilgi Sentezleyici ────────────────────────────────────────────────────

class KnowledgeSynthesizer:
    """
    Notlardan yeni, daha yüksek seviyeli bilgi üretir.

    Kullanım:
        syn = KnowledgeSynthesizer()
        syn.initialize(config, event_bus)
        result = syn.synthesize(notes, topic="GPU Termal Yönetimi")
        if result:
            payload = result.to_note_payload()
    """

    name = "evolution.synthesizer"

    def __init__(self) -> None:
        self._config: Any = None
        self._event_bus: Any = None
        self._counters: Dict[str, float] = {
            "syntheses": 0.0,
            "skipped": 0.0,
            "contradictions": 0.0,
            "bodies": 0.0,
        }
        self._last_result: Dict[str, Any] = {}
        self._initialized: bool = False

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Sentezleyiciyi bağla. İdempotenttir."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._initialized = True
        logger.info("Bilgi sentezleyici hazır")

    def shutdown(self) -> None:
        """Kaynakları serbest bırak. İdempotenttir."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info("Bilgi sentezleyici kapatıldı")

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Olay yayınla (bus yoksa atlanır)."""
        if self._event_bus:
            try:
                self._event_bus.emit(event_type, data, source=self.name)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Olay yayınlanamadı ({event_type}): {exc}")

    # ─── Ana Sentez ────────────────────────────────────────────────────

    def synthesize(
        self,
        notes: List[dict],
        topic: str = "",
        kind: str = "concept",
    ) -> Optional[SynthesisResult]:
        """
        Notları birleştirip yeni bir sentez notu üretir.

        Args:
            notes: Kasa not yükleri (dict listesi) — **değiştirilmez**
            topic: Sentez konusu (boşsa en sık etiketten türetilir)
            kind: Not türü

        Returns:
            `SynthesisResult` ya da kullanılabilir not yoksa None
        """
        usable = [n for n in (notes or []) if isinstance(n, dict) and _clean_body(n).strip()]
        if len(usable) < MIN_NOTES:
            self._counters["skipped"] += 1.0
            logger.info(
                f"Sentez atlandı: en az {MIN_NOTES} kullanılabilir not gerekli "
                f"(gelen: {len(usable)})"
            )
            return None

        try:
            contradictions = self.find_contradictions(usable)
            points = self.extract_key_points(usable, max_points=8)
            topic_title = str(topic or "").strip() or self._derive_topic(usable)
            tags = self._collect_tags(usable)
            sources: List[str] = []
            confidences: List[float] = []

            for index, note in enumerate(usable):
                confidences.append(_clamp(float(_note_field(note, "confidence", 0.5) or 0.5)))
                for source in (_note_field(note, "sources", []) or []):
                    text = str(source)
                    if text and text not in sources:
                        sources.append(text)
                if not _note_field(note, "sources", []):
                    sources.append(f"note://{_note_id(note, index)}")

            # Kapsam faktörü: birleşik ana noktaların ne kadarı temsil ediliyor?
            union_terms: Set[str] = set()
            for note in usable:
                union_terms |= _term_set(f"{_note_field(note, 'title', '')} {_clean_body(note)}")
            covered_terms: Set[str] = set()
            for point in points:
                covered_terms |= _term_set(point)
            coverage = _jaccard(union_terms, covered_terms) if union_terms else 0.0
            coverage_factor = _clamp(0.5 + coverage * 2.5, 0.5, 1.0)

            mean_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            max_confidence = max(confidences) if confidences else 0.5
            confidence = min(
                min(mean_confidence * coverage_factor, CONFIDENCE_CAP),
                max_confidence + 0.1,
            )
            confidence = _clamp(confidence, 0.0, CONFIDENCE_CAP)

            conflict_texts = [
                f"{c.get('a')} ↔ {c.get('b')}: {c.get('reason')}" for c in contradictions
            ]
            body = self.build_note_body(
                topic_title,
                points,
                sources,
                contradictions=contradictions,
            )

            rationale = (
                f"{len(usable)} not birleştirildi; kapsam faktörü {coverage_factor:.2f}, "
                f"ortalama kaynak güveni {mean_confidence:.2f}"
            )
            if contradictions:
                rationale += f", {len(contradictions)} çelişki işaretlendi"

            result = SynthesisResult(
                title=topic_title,
                body=body,
                tags=tags,
                confidence=confidence,
                sources=sources,
                kind=str(kind or "concept"),
                rationale=rationale,
                contradictions=conflict_texts,
            )

            self._counters["syntheses"] += 1.0
            self._counters["contradictions"] += float(len(contradictions))
            self._last_result = result.to_dict()

            self._emit("evolution.note_synthesized", {
                "title": result.title,
                "confidence": round(confidence, 4),
                "source_count": len(usable),
                "contradictions": len(contradictions),
                "kind": result.kind,
            })
            logger.info(
                f"Sentez tamam: '{result.title}' · {len(usable)} kaynak · "
                f"güven={confidence:.2f} · çelişki={len(contradictions)}"
            )
            return result

        except Exception as exc:
            self._counters["skipped"] += 1.0
            logger.warning(f"Sentez hatası: {exc}")
            return None

    def _derive_topic(self, notes: Sequence[dict]) -> str:
        """Konu verilmediyse en sık etiketten başlık türetir."""
        counts: Dict[str, int] = {}
        for note in notes:
            for tag in _note_tag_set(note):
                counts[tag] = counts.get(tag, 0) + 1
        if counts:
            top = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            return top.title()
        title = str(_note_field(notes[0], "title", "") or "Sentez").strip()
        return title or "Sentez"

    def _collect_tags(self, notes: Sequence[dict]) -> List[str]:
        """Notlardan birleşik etiket listesi (sıklığa göre)."""
        counts: Dict[str, int] = {}
        for note in notes:
            for tag in _note_tag_set(note):
                counts[tag] = counts.get(tag, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [tag for tag, _ in ordered[:10]]

    # ─── Özet ve Ana Noktalar ──────────────────────────────────────────

    def summarize(self, notes: List[dict], max_sentences: int = 5) -> str:
        """
        Notlardan çıkarımsal (extractive) özet üretir.

        Cümle puanı = diğer notlarla terim örtüşmesi (merkezilik) + uzunluk
        uygunluğu + not başı olma bonusu. Ağırlıklı `confidence` ile çarpılır.
        """
        limit = max(1, int(max_sentences))
        candidates: List[Tuple[str, Set[str], float]] = []

        for note in (notes or []):
            if not isinstance(note, dict):
                continue
            weight = _clamp(float(_note_field(note, "confidence", 0.5) or 0.5), 0.0, 1.0)
            for position, sentence in enumerate(_sentences(_clean_body(note))):
                length = len(_tokens(sentence))
                if length < 4:
                    continue
                position_bonus = 0.15 if position == 0 else 0.0
                candidates.append((sentence, _term_set(sentence), weight + position_bonus))

        if not candidates:
            return ""

        # Merkezilik: her cümlenin diğerleriyle ortalama örtüşmesi
        scored: List[Tuple[float, str]] = []
        for index, (sentence, terms, weight) in enumerate(candidates):
            overlap = 0.0
            comparisons = 0
            for other_index, (_, other_terms, _) in enumerate(candidates):
                if other_index == index:
                    continue
                overlap += _jaccard(terms, other_terms)
                comparisons += 1
            centrality = overlap / comparisons if comparisons else 0.0
            length = len(_tokens(sentence))
            length_factor = 1.0 if 8 <= length <= 40 else 0.6
            score = (0.55 * centrality + 0.25 * weight + 0.20 * length_factor) * (0.5 + weight)
            scored.append((score, sentence))

        scored.sort(key=lambda pair: -pair[0])

        chosen: List[str] = []
        seen: Set[str] = set()
        for _, sentence in scored:
            fingerprint = " ".join(sorted(_term_set(sentence)))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            chosen.append(sentence.strip())
            if len(chosen) >= limit:
                break

        return " ".join(chosen)

    def extract_key_points(self, notes: List[dict], max_points: int = 8) -> List[str]:
        """
        Ana noktaları çıkarır (cümle bazlı, tekrarsız).

        Nokta seçimi: not güveni + cümle merkeziliği + sayı/koşul içerme
        (somut bilgi) bonusu.
        """
        limit = max(1, int(max_points))
        points: List[str] = []
        seen: Set[str] = set()

        candidates: List[Tuple[float, str]] = []
        for note in (notes or []):
            if not isinstance(note, dict):
                continue
            confidence = _clamp(float(_note_field(note, "confidence", 0.5) or 0.5), 0.0, 1.0)
            for sentence in _sentences(_clean_body(note)):
                tokens = _tokens(sentence)
                if len(tokens) < 5:
                    continue
                concreteness = 0.0
                if re.search(r"\d", sentence):
                    concreteness += 0.15
                if any(marker in _norm_word(sentence) for marker in ("eğer", "ise", "aşarsa", "gerekiyorsa", "if ")):
                    concreteness += 0.10
                length_factor = 1.0 if 6 <= len(tokens) <= 45 else 0.7
                score = 0.5 * confidence + 0.3 * length_factor + concreteness
                candidates.append((score, sentence.strip()))

        candidates.sort(key=lambda pair: -pair[0])
        for _, sentence in candidates:
            fingerprint = " ".join(sorted(_term_set(sentence)))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            points.append(sentence if sentence.endswith((".", "!", "?", "…")) else sentence + ".")
            if len(points) >= limit:
                break

        return points

    # ─── Çelişki Tespiti ───────────────────────────────────────────────

    def find_contradictions(self, notes: List[dict]) -> List[dict]:
        """
        Notlar arası çelişkileri bulur.

        Üç yöntem birleştirilir:

            1. **Sayısal çelişki**: aynı birimde, paylaşılan özneye sahip ve
               değerleri %10'dan fazla ayrışan iddialar.
            2. **Açık olumsuzlama**: bir notta "X artırır", diğerinde
               "X artırmaz".
            3. **Karşıt kelime çiftleri**: artır/azalt, yükselt/düşür, always/never,
               her zaman/asla, increase/decrease, enable/disable, ...

        Returns:
            `[{"a": id, "b": id, "reason": str, "tokens": [...]}]`
        """
        results: List[dict] = []
        usable = [n for n in (notes or []) if isinstance(n, dict)]

        try:
            for i in range(len(usable)):
                for j in range(i + 1, len(usable)):
                    note_a, note_b = usable[i], usable[j]
                    id_a, id_b = _note_id(note_a, i), _note_id(note_b, j)
                    body_a = f"{_note_field(note_a, 'title', '')} {_clean_body(note_a)}"
                    body_b = f"{_note_field(note_b, 'title', '')} {_clean_body(note_b)}"

                    # 1) Sayısal çelişkiler
                    for finding in self._numeric_conflicts(body_a, body_b):
                        results.append({
                            "a": id_a,
                            "b": id_b,
                            "reason": finding["reason"],
                            "tokens": finding["tokens"],
                        })

                    # 2) Karşıt kelime / olumsuzlama çelişkileri
                    for finding in self._opposite_conflicts(body_a, body_b):
                        results.append({
                            "a": id_a,
                            "b": id_b,
                            "reason": finding["reason"],
                            "tokens": finding["tokens"],
                        })
        except Exception as exc:
            logger.warning(f"Çelişki tespitinde hata: {exc}")

        if results:
            self._emit("evolution.contradiction_found", {
                "count": len(results),
                "pairs": [f"{r['a']}↔{r['b']}" for r in results[:5]],
            })
        return results

    def _numeric_conflicts(self, body_a: str, body_b: str) -> List[dict]:
        """Aynı özne + aynı birim + farklı değer → sayısal çelişki."""
        findings: List[dict] = []
        assertions_a = _numeric_assertions(body_a)
        assertions_b = _numeric_assertions(body_b)

        for subject_a, value_a, unit_a, context_a, sentence_a in assertions_a:
            for subject_b, value_b, unit_b, context_b, sentence_b in assertions_b:
                if unit_a != unit_b:
                    continue
                shared = _subject_overlap(context_a, context_b)
                if not shared:
                    continue
                denominator = max(abs(value_a), abs(value_b), 1e-9)
                if abs(value_a - value_b) / denominator <= 0.10:
                    continue  # aynı değer → çelişki değil, uyum
                findings.append({
                    "reason": (
                        f"'{' '.join(shared)}' konusunda aynı birimde ({unit_a}) farklı değerler: "
                        f"{value_a:g} vs {value_b:g}"
                    ),
                    "tokens": shared + [unit_a],
                })
        # Aynı çifti tek kez raporla
        return self._dedupe_findings(findings)

    def _opposite_conflicts(self, body_a: str, body_b: str) -> List[dict]:
        """Karşıt fiil çiftleri ve olumsuzlama farkları."""
        findings: List[dict] = []
        terms_a = _term_set(body_a)
        terms_b = _term_set(body_b)
        stems_a = _word_stems(body_a)
        stems_b = _word_stems(body_b)

        for positive, negative in OPPOSITE_PAIRS:
            pos_stem = stem(positive.split()[0])
            neg_stem = stem(negative.split()[0])
            a_has_pos = _matches_stem(pos_stem, stems_a)
            a_has_neg = _matches_stem(neg_stem, stems_a)
            b_has_pos = _matches_stem(pos_stem, stems_b)
            b_has_neg = _matches_stem(neg_stem, stems_b)
            if not ((a_has_pos and b_has_neg) or (a_has_neg and b_has_pos)):
                continue

            shared = sorted((terms_a & terms_b) - {pos_stem, neg_stem})
            if len(shared) < 1:
                continue
            findings.append({
                "reason": (
                    f"Karşıt ifadeler: '{positive}' ↔ '{negative}' "
                    f"({' '.join(shared[:4])} konusunda)"
                ),
                "tokens": [positive, negative] + shared[:4],
            })

        # Açık olumsuzlama: bir tarafta olumsuz, diğerinde değil
        for positive, negative in NEGATION_SENSITIVE_PAIRS:
            pos_stem = stem(positive.split()[0])
            if not (_matches_stem(pos_stem, stems_a) and _matches_stem(pos_stem, stems_b)):
                continue
            neg_a = _has_negation(body_a)
            neg_b = _has_negation(body_b)
            if neg_a == neg_b:
                continue
            shared = sorted((terms_a & terms_b) - {pos_stem})
            if not shared:
                continue
            findings.append({
                "reason": (
                    f"Olumsuzlama farkı: '{positive}' bir notta olumlu, diğerinde olumsuz "
                    f"({' '.join(shared[:4])})"
                ),
                "tokens": [positive] + shared[:4],
            })

        return self._dedupe_findings(findings)

    @staticmethod
    def _dedupe_findings(findings: List[dict]) -> List[dict]:
        """Aynı gerekçe + token kümesini tekilleştirir."""
        unique: List[dict] = []
        seen: Set[Tuple[str, Tuple[str, ...]]] = set()
        for finding in findings:
            key = (str(finding.get("reason")), tuple(sorted(finding.get("tokens") or [])))
            if key in seen:
                continue
            seen.add(key)
            unique.append(finding)
        return unique

    # ─── Gövde Üretimi ─────────────────────────────────────────────────

    def build_note_body(
        self,
        topic: str,
        points: List[str],
        sources: List[str],
        contradictions: Optional[List[dict]] = None,
    ) -> str:
        """
        Obsidian uyumlu markdown gövdesi üretir.

        Bölümler: `## Özet`, `## Ana Noktalar`, `## Detay`, `## Çelişkiler`
        (yalnızca varsa, `[[id]]` bağlantılarıyla), `## Kaynaklar`,
        `## İlişkili` (`[[id]]` bağlantıları).
        """
        self._counters["bodies"] += 1.0
        title = str(topic or "").strip() or "Sentez"
        point_list = [str(p).strip() for p in (points or []) if str(p).strip()]
        conflicts = list(contradictions or [])

        lines: List[str] = []

        # Özet
        summary_source = " ".join(point_list[:3]) if point_list else f"{title} hakkında derlenmiş bilgi."
        lines.append(SECTION_SUMMARY)
        lines.append(summary_source.strip())
        lines.append("")

        # Ana Noktalar
        lines.append(SECTION_POINTS)
        if point_list:
            for point in point_list:
                lines.append(f"- {point}")
        else:
            lines.append("- (Ana nokta çıkarılamadı)")
        lines.append("")

        # Detay
        lines.append(SECTION_DETAIL)
        lines.append(
            f"Bu not, **{title}** konusundaki {len(point_list)} ana noktanın "
            f"{len(sources)} kaynaktan derlenmesiyle üretilmiştir."
        )
        for index, point in enumerate(point_list, start=1):
            lines.append("")
            lines.append(f"{index}. {point}")
        lines.append("")

        # Çelişkiler (yalnızca varsa)
        if conflicts:
            lines.append(SECTION_CONTRADICTIONS)
            for conflict in conflicts:
                note_a = str(conflict.get("a", "?"))
                note_b = str(conflict.get("b", "?"))
                reason = str(conflict.get("reason", "gerekçe belirtilmedi"))
                lines.append(f"- [[{note_a}]] ↔ [[{note_b}]] — {reason}")
            lines.append("")

        # Kaynaklar
        lines.append(SECTION_SOURCES)
        if sources:
            for source in sources:
                lines.append(f"- `{source}`")
        else:
            lines.append("- (kaynak belirtilmedi)")
        lines.append("")

        # İlişkili
        lines.append(SECTION_RELATED)
        related: List[str] = []
        if conflicts:
            for conflict in conflicts:
                for key in ("a", "b"):
                    value = str(conflict.get(key, "")).strip()
                    if value and value not in related:
                        related.append(value)
        for source in sources:
            identifier = str(source)
            if identifier.startswith("note://"):
                identifier = identifier[len("note://"):]
                if identifier and identifier not in related:
                    related.append(identifier)
        if related:
            for identifier in related:
                lines.append(f"- [[{identifier}]]")
        else:
            lines.append("- (ilişkili not bulunamadı)")
        lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    # ─── Bağlantı Önerisi ve Birleştirme Değeri ────────────────────────

    def propose_links(
        self,
        notes: List[dict],
        candidate_titles: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Notlar arasında bağlantı önerir.

        Notun başlık + gövde terimleri, diğer notların başlıklarıyla
        karşılaştırılır; yeterli örtüşme varsa öneri üretilir.

        Returns:
            `[{"source": id, "target_title": str, "reason": str, "strength": float}]`
        """
        usable = [n for n in (notes or []) if isinstance(n, dict)]
        suggestions: List[dict] = []
        if not usable:
            return suggestions

        # Aday başlık havuzu
        titles: List[Tuple[str, str]] = []  # (id, title)
        for index, note in enumerate(usable):
            title = str(_note_field(note, "title", "") or "").strip()
            if title:
                titles.append((_note_id(note, index), title))
        for title in (candidate_titles or []):
            text = str(title or "").strip()
            if text and all(existing != text for _, existing in titles):
                titles.append((f"candidate::{text}", text))

        for index, note in enumerate(usable):
            source_id = _note_id(note, index)
            note_title = str(_note_field(note, "title", "") or "").strip().lower()
            body_terms = _term_set(f"{_note_field(note, 'title', '')} {_clean_body(note)}")
            if not body_terms:
                continue

            for target_id, target_title in titles:
                if target_id == source_id:
                    continue
                if target_title.strip().lower() == note_title:
                    continue
                title_terms = _term_set(target_title)
                if not title_terms:
                    continue
                overlap = body_terms & title_terms
                if not overlap:
                    continue
                strength = _clamp(
                    _cosine_terms(body_terms, title_terms) * 0.7 + min(1.0, len(overlap) / 3.0) * 0.3
                )
                if strength < 0.20:
                    continue
                suggestions.append({
                    "source": source_id,
                    "target_title": target_title,
                    "reason": f"'{target_title}' başlığı ile örtüşen terimler: {', '.join(sorted(overlap)[:5])}",
                    "strength": round(strength, 4),
                })

        suggestions.sort(key=lambda item: (-item["strength"], item["source"]))
        return suggestions[: max(20, len(usable) * 3)]

    def score_merge_worthiness(self, a: dict, b: dict) -> float:
        """
        İki notun birleştirmeye değer olup olmadığını puanlar (0..1).

        Bileşenler: başlık benzerliği (%40), etiket Jaccard'ı (%20),
        gövde terim örtüşmesi (%40). `> 0.75` → "bunları birleştir".
        """
        try:
            note_a = a if isinstance(a, dict) else {}
            note_b = b if isinstance(b, dict) else {}

            title_a = _term_set(_note_field(note_a, "title", ""))
            title_b = _term_set(_note_field(note_b, "title", ""))
            title_similarity = max(_jaccard(title_a, title_b), _cosine_terms(title_a, title_b))

            tag_similarity = _jaccard(_note_tag_set(note_a), _note_tag_set(note_b))

            body_a = _term_set(_clean_body(note_a))
            body_b = _term_set(_clean_body(note_b))
            body_similarity = max(_jaccard(body_a, body_b), _cosine_terms(body_a, body_b))

            score = (
                0.40 * title_similarity
                + 0.20 * tag_similarity
                + 0.40 * body_similarity
            )
            return round(_clamp(score), 4)
        except Exception as exc:
            logger.debug(f"Birleştirme puanı hesaplanamadı: {exc}")
            return 0.0

    # ─── İstatistikler ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "syntheses": int(self._counters["syntheses"]),
            "skipped": int(self._counters["skipped"]),
            "contradictions_found": int(self._counters["contradictions"]),
            "bodies_built": int(self._counters["bodies"]),
            "min_notes": MIN_NOTES,
            "merge_worthy_threshold": MERGE_WORTHY,
            "last_result": {
                "title": self._last_result.get("title", ""),
                "confidence": self._last_result.get("confidence", 0.0),
            },
        }
