"""
NeuroCore Öz-Değerlendirme Modülü (L8 · Evolution)
===================================================
Üretilen bir yanıtın kalitesini **açıklanabilir** bileşenlere ayırarak puanlar:

    relevance (0.30) · coverage (0.22) · coherence (0.18) · grounding (0.20)
    · calibration (0.10)

Hiçbir LLM veya torch kullanılmaz; tüm puanlar TF/token/istatistik tabanlıdır ve
`QualityReport.components` içinde tek tek raporlanır. Bu, evrim döngüsünün
"neden bu yanıt zayıf?" sorusunu cevaplayabilmesi için şarttır.

Bileşenler:
    relevance   : Soru içerik kelimeleri ile yanıt kelimeleri arasında TF örtüşmesi,
                  bigram örtüşmesi, ilk cümlede anahtar kelime bonusu ve
                  aşırı kısa yanıt cezası.
    coverage    : Soru içerik terimlerinin yanıtta (birebir veya gövde eşleşmesiyle)
                  görülme oranı + çoklu istek sorularında listeleme bonusu.
    coherence   : Cümle sayısı/uzunluk bandı, tekrar eden cümle, sarkan bağlaç,
                  ardışık aynı başlangıç, aşırı BÜYÜK HARF, paragraf yapısı.
    grounding   : Yanıt cümlelerinin bağlam öğeleriyle anlamlı token örtüşmesi oranı.
    calibration : Belirsizlik seviyesi ile yanıtın kesinlik dili uyumu.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("evolution.evaluator")


# ─── Bileşen Ağırlıkları ve Eşikler ────────────────────────────────────────

WEIGHTS: Dict[str, float] = {
    "relevance": 0.30,
    "coverage": 0.22,
    "coherence": 0.18,
    "grounding": 0.20,
    "calibration": 0.10,
}

#: Verdikt eşikleri (alt sınır → etiket)
VERDICT_THRESHOLDS: List[Tuple[float, str]] = [
    (0.78, "excellent"),
    (0.60, "good"),
    (0.45, "acceptable"),
    (0.30, "weak"),
    (0.00, "poor"),
]

#: Bağlam verilmediğinde grounding için nötr puan
GROUNDING_NO_CONTEXT = 0.5


# ─── Stopword Listesi (tr + en) ────────────────────────────────────────────

STOPWORDS: Set[str] = {
    # Türkçe
    "ve", "veya", "ile", "için", "gibi", "kadar", "ama", "fakat", "ancak", "ki",
    "de", "da", "mi", "mı", "mu", "mü", "ise", "eğer", "şey", "şeyler", "bu",
    "şu", "o", "bir", "bazı", "her", "hiç", "çok", "daha", "en", "az", "kendi",
    "ben", "sen", "biz", "siz", "onlar", "benim", "senin", "bizim", "sizin",
    "olarak", "olduğu", "olduğunu", "olan", "olur", "olabilir", "dir", "dır",
    "dur", "dür", "tır", "tir", "tur", "tür", "ise", "ne", "neden", "niçin",
    "nasıl", "nerede", "hangi", "kim", "kaç", "zaman", "sonra", "önce", "göre",
    "üzere", "hakkında", "ile", "arasında", "içinde", "üzerinde", "yani",
    "böyle", "şöyle", "artık", "hâlâ", "hala", "sadece", "yalnız", "bile",
    "tüm", "bütün", "başka", "diğer", "ayrıca", "dolayı", "ötürü", "karşın",
    "rağmen", "hem", "ya", "ya da", "değil", "olması", "oldukça", "genelde",
    "genellikle", "mutlaka", "belki", "acaba",
    # İngilizce
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "have", "has", "had", "having", "will",
    "would", "shall", "should", "can", "could", "may", "might", "must", "of",
    "to", "in", "on", "at", "by", "for", "with", "from", "as", "it", "its",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
    "our", "your", "my", "me", "us", "not", "no", "so", "such", "which",
    "who", "whom", "what", "when", "where", "why", "how", "there", "here",
    "about", "into", "over", "under", "between", "also", "very", "more",
    "most", "some", "any", "all", "each", "every", "both", "few", "many",
    "much", "other", "only", "just", "up", "down", "out", "off", "again",
}


# ─── Kelime ve Cümle Yardımcıları ──────────────────────────────────────────

_WORD_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü_]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n{2,}")
_LIST_RE = re.compile(r"^\s*(?:[-*•‣]|\d+[.)])\s+", re.MULTILINE)
_WS_RE = re.compile(r"\s+")

#: Türkçe gövde (stem) son ekleri — uzundan kısaya denenir
_TR_SUFFIXES: Tuple[str, ...] = (
    "larında", "lerinde", "larındaki", "lerindeki", "ması", "mesi", "masını",
    "mesini", "dıklarında", "diklerinde", "larak", "lererek", "lardan",
    "lerden", "ların", "lerin", "lara", "lere", "ları", "leri", "lardaki",
    "lerdeki", "daki", "deki", "taki", "teki", "dan", "den", "tan", "ten",
    "lar", "ler", "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "sı", "si", "su", "sü", "ın", "in", "un", "ün", "ı", "i", "u", "ü",
    "da", "de", "ta", "te", "ya", "ye", "a", "e",
)

#: Soru kalıpları (soru sorulduğunu anlamak için)
_QUESTION_MARKERS: Tuple[str, ...] = (
    "neden", "niçin", "niye", "nasıl", "ne şekilde", "ne zaman", "nerede",
    "hangi", "kaç", "ne kadar", "mi", "mı", "mu", "mü", "why", "how", "when",
    "where", "which", "what", "who", "?",
)

#: Listeleme isteği işaretleri
_ENUM_MARKERS: Tuple[str, ...] = (
    "ve", "listele", "listesi", "say", "sayısı", "maddeler", "adımlar",
    "adım adım", "sırala", "çeşitleri", "türleri", "yolları", "yöntemleri",
    "and", "list", "enumerate", "steps", "types", "ways", "methods",
)

#: Sarkan bağlaçlar (cümle başında kalırsa akış bozulur)
_DANGLING_CONNECTORS: Set[str] = {
    "ve", "veya", "ama", "fakat", "ancak", "çünkü", "yani", "ayrıca", "ile",
    "ki", "hem", "öte", "and", "or", "but", "because", "so", "also", "however",
    "therefore", "thus",
}

#: Belirsizlik/çekince ifadeleri
_HEDGE_WORDS: Tuple[str, ...] = (
    "muhtemelen", "olabilir", "sanırım", "sanıyorum", "galiba", "belki",
    "tahmin", "tahminen", "yaklaşık", "görünüyor", "gibi görünüyor",
    "olduğu düşünülüyor", "varsayılabilir", "olması muhtemel", "eğiliminde",
    "genellikle", "çoğunlukla", "bazı durumlarda", "kesin değil", "emin değilim",
    "probably", "possibly", "perhaps", "maybe", "likely", "unlikely",
    "may", "might", "could", "seems", "appears", "suggests", "roughly",
    "approximately", "around", "tends to", "in some cases", "not certain",
)

#: Kesinlik ifadeleri
_ASSERTIVE_WORDS: Tuple[str, ...] = (
    "kesinlikle", "mutlaka", "her zaman", "asla", "hiçbir", "şüphesiz",
    "tartışmasız", "kesin", "definitely", "certainly", "always", "never",
    "without doubt", "undoubtedly", "must",
)

#: Olumsuzluk işaretleri
_NEGATION_MARKERS: Tuple[str, ...] = ("değil", "yok", "asla", "not", "never", "no")


def _tokens(text: Any) -> List[str]:
    """Metni küçük harfli kelimelere ayırır."""
    return _WORD_RE.findall(str(text or "").lower())


def _content_terms(text: Any, min_len: int = 3) -> List[str]:
    """Stopword'süz içerik terimleri (sıra korunur, tekrarlar kalır)."""
    return [t for t in _tokens(text) if len(t) >= min_len and t not in STOPWORDS]


def _stem(token: str) -> str:
    """
    Basit kural tabanlı Türkçe gövde indirgemesi.

    Mükemmel bir morfoloji değil; yalnızca "sıcaklığı" ↔ "sıcaklık" gibi
    çekim farklarını kapatacak kadar yeterlidir.
    """
    word = str(token or "")
    if len(word) <= 4:
        return word
    for suffix in _TR_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)]
    return word


def _terms_set(text: Any, stem: bool = True) -> Set[str]:
    """İçerik terimleri kümesi (isteğe bağlı gövde indirgemeli)."""
    return {_stem(t) for t in _content_terms(text)} if stem else set(_content_terms(text))


def _bigrams(tokens: List[str]) -> Set[Tuple[str, str]]:
    """Ardışık kelime ikilileri."""
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _sentences(text: Any) -> List[str]:
    """Metni cümlelere böler (boş parçalar atılır)."""
    raw = _WS_RE.sub(" ", str(text or "").replace("\r", "")).strip()
    if not raw:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(raw)]
    return [p for p in parts if p]


def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    """Jaccard benzerliği (iki küme de boşsa 0)."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Değeri [low, high] aralığına sıkıştır."""
    return max(low, min(high, float(value)))


def _f1(overlap: float, query_mass: float, answer_mass: float) -> float:
    """Örtüşme kütlesinden F1 benzerliği üretir."""
    if overlap <= 0.0 or query_mass <= 0.0 or answer_mass <= 0.0:
        return 0.0
    precision = overlap / answer_mass
    recall = overlap / query_mass
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _token_f1(reference: str, candidate: str) -> float:
    """İki metin arasında gövde-indirgenmiş token F1 skoru."""
    ref = _content_terms(reference)
    cand = _content_terms(candidate)
    if not ref or not cand:
        return 0.0
    ref_counter: Dict[str, int] = {}
    for token in ref:
        key = _stem(token)
        ref_counter[key] = ref_counter.get(key, 0) + 1
    cand_counter: Dict[str, int] = {}
    for token in cand:
        key = _stem(token)
        cand_counter[key] = cand_counter.get(key, 0) + 1
    overlap = sum(min(count, cand_counter.get(key, 0)) for key, count in ref_counter.items())
    return _f1(float(overlap), float(len(ref)), float(len(cand)))


def _detect_negation(text: Any) -> bool:
    """Metinde olumsuzluk var mı (kelime ya da Türkçe '-ma/-me' eki)."""
    lowered = str(text or "").lower()
    padded = f" {lowered} "
    if any(f" {marker} " in padded for marker in _NEGATION_MARKERS):
        return True
    return bool(re.search(r"\b\w{3,}(?:ma|me)(?:dı|di|z|r|sı|si)?\b", lowered))


# ─── Kalite Raporu ─────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """
    Tek bir yanıtın açıklanabilir kalite raporu.

    Attributes:
        score: Nihai puan (0..1)
        components: Bileşen bazlı puanlar (`relevance`, `coverage`, ...)
        issues: Tespit edilen sorun kodları (Türkçe/ASCII)
        suggestions: İyileştirme önerileri
        verdict: `poor | weak | acceptable | good | excellent`
    """

    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    verdict: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dict."""
        return {
            "score": round(float(self.score), 4),
            "components": {k: round(float(v), 4) for k, v in self.components.items()},
            "issues": list(self.issues),
            "suggestions": list(self.suggestions),
            "verdict": self.verdict,
        }


# ─── Öz-Değerlendirici ─────────────────────────────────────────────────────

class SelfEvaluator:
    """
    Kural tabanlı, açıklanabilir yanıt değerlendiricisi.

    Kullanım:
        ev = SelfEvaluator()
        ev.initialize(config, event_bus)
        report = ev.evaluate("GPU neden ısınır?", "...", context=["kn_1"])
        print(report.score, report.verdict, report.issues)
    """

    name = "evolution.evaluator"

    def __init__(self) -> None:
        self._config: Any = None
        self._event_bus: Any = None
        self._counters: Dict[str, float] = {
            "evaluations": 0.0,
            "errors": 0.0,
            "score_sum": 0.0,
            "comparisons": 0.0,
        }
        self._last_report: Optional[Dict[str, Any]] = None
        self._initialized: bool = False

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Değerlendiriciyi bağla. İdempotenttir."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._initialized = True
        logger.info("Öz-değerlendirici hazır (ağırlıklar: " +
                    ", ".join(f"{k}={v:.2f}" for k, v in WEIGHTS.items()) + ")")

    def shutdown(self) -> None:
        """Kaynakları serbest bırak. İdempotenttir."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info("Öz-değerlendirici kapatıldı")

    # ─── Ana Değerlendirme ──────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        answer: str,
        context: Optional[List[str]] = None,
        uncertainty: float = 0.5,
        latency_ms: float = 0.0,
        expected: Optional[str] = None,
    ) -> QualityReport:
        """
        Yanıtı değerlendirir ve her zaman bir rapor döndürür.

        Args:
            query: Kullanıcı sorgusu
            answer: Değerlendirilecek yanıt
            context: Kullanılan bağlam parçaları (not metinleri / alıntılar)
            uncertainty: Modelin bildirdiği belirsizlik (0..1)
            latency_ms: Yanıt gecikmesi (ms) — rapora gecikme uyarısı ekler
            expected: Bilinen doğru/ideal yanıt (verilirse `similarity` bileşeni eklenir)

        Returns:
            `QualityReport` — iç hata durumunda bile geçerli bir rapor
        """
        context_items = [str(c) for c in (context or [])]
        issues: List[str] = []
        suggestions: List[str] = []

        # Boş girdi: değerlendirilecek bir şey yok → nötr rapor
        if not str(query or "").strip() or not str(answer or "").strip():
            self._counters["errors"] += 1.0
            return QualityReport(
                score=0.5,
                components={},
                issues=["degerlendirme_hatasi: bos_girdi"],
                suggestions=["Sorgu ve yanıt dolu olmalı"],
                verdict=self._verdict_of(0.5),
            )

        try:
            components: Dict[str, float] = {
                "relevance": _clamp(self.score_relevance(query, answer)),
                "coverage": _clamp(self.score_coverage(query, answer, context_items)),
                "coherence": _clamp(self.score_coherence(answer)),
                "grounding": _clamp(self.score_grounding(answer, context_items)),
                "calibration": _clamp(self.score_calibration(answer, uncertainty)),
            }

            score = sum(WEIGHTS[key] * components[key] for key in WEIGHTS)

            if expected:
                similarity = _clamp(_token_f1(expected, answer))
                components["similarity"] = similarity
                score = 0.5 * score + 0.5 * similarity

            score = _clamp(score)
            verdict = self._verdict_of(score)
            issues, suggestions = self._diagnose(components, query, answer, context_items)

            if latency_ms and float(latency_ms) > 5000.0:
                issues.append("yuksek_gecikme")
                suggestions.append("Yanıt süresi 5 sn üzerinde; önbellek veya daha küçük model düşünün")

            report = QualityReport(
                score=score,
                components=components,
                issues=issues,
                suggestions=suggestions,
                verdict=verdict,
            )

            self._counters["evaluations"] += 1.0
            self._counters["score_sum"] += score
            self._last_report = report.to_dict()

            if self._event_bus:
                try:
                    self._event_bus.emit(
                        "evolution.quality_evaluated",
                        {"score": round(score, 4), "verdict": verdict,
                         "components": report.to_dict()["components"],
                         "query": str(query)[:120]},
                        source=self.name,
                    )
                except Exception:
                    pass

            return report

        except Exception as exc:  # Graceful degradation: asla çökme
            self._counters["errors"] += 1.0
            logger.warning(f"Değerlendirme hatası: {exc}")
            return QualityReport(
                score=0.5,
                components={},
                issues=[f"degerlendirme_hatasi: {exc}"],
                suggestions=[],
                verdict=self._verdict_of(0.5),
            )

    # ─── Bileşen 1: İlgi (Relevance) ────────────────────────────────────

    def score_relevance(self, query: str, answer: str) -> float:
        """
        Soru ile yanıt arasındaki TF tabanlı ilgi skoru.

        - İçerik kelimeleri üzerinden F1 (gövde indirgemeli)
        - Bigram örtüşmesi (kelime sırası bilgisi)
        - Yanıtın ilk cümlesinde soru anahtar kelimesi bonusu
        - Soruya göre aşırı kısa yanıt cezası
        """
        query_terms = _content_terms(query)
        answer_terms = _content_terms(answer)
        if not query_terms or not answer_terms:
            return 0.0

        query_counter: Dict[str, int] = {}
        for token in query_terms:
            key = _stem(token)
            query_counter[key] = query_counter.get(key, 0) + 1
        answer_counter: Dict[str, int] = {}
        for token in answer_terms:
            key = _stem(token)
            answer_counter[key] = answer_counter.get(key, 0) + 1

        overlap = sum(min(count, answer_counter.get(key, 0)) for key, count in query_counter.items())
        unigram_f1 = _f1(float(overlap), float(len(query_terms)), float(len(answer_terms)))

        query_bigrams = _bigrams([_stem(t) for t in query_terms])
        answer_bigrams = _bigrams([_stem(t) for t in answer_terms])
        if query_bigrams and answer_bigrams:
            bigram_overlap = len(query_bigrams & answer_bigrams)
            bigram_f1 = _f1(float(bigram_overlap), float(len(query_bigrams)), float(len(answer_bigrams)))
        else:
            bigram_f1 = 0.0

        # İlk cümlede anahtar kelime bonusu
        sentences = _sentences(answer)
        first_sentence_terms = _terms_set(sentences[0]) if sentences else set()
        query_keys = set(query_counter)
        first_hit = len(first_sentence_terms & query_keys)
        first_bonus = min(0.15, 0.075 * first_hit) if first_hit else 0.0

        score = 0.70 * unigram_f1 + 0.30 * bigram_f1 + first_bonus

        # Aşırı kısa yanıt cezası: soru ne kadar zenginse yanıt da o kadar uzun olmalı
        query_len = len(query_terms)
        answer_len = len(answer_terms)
        if answer_len < max(3, int(query_len * 0.6)):
            score *= 0.6
        elif answer_len < max(5, query_len):
            score *= 0.85

        return _clamp(score)

    # ─── Bileşen 2: Kapsam (Coverage) ───────────────────────────────────

    def score_coverage(self, query: str, answer: str, context: Optional[List[str]] = None) -> float:
        """
        Sorudaki içerik terimlerinin yanıtta karşılanma oranı.

        - Birebir eşleşme veya gövde eşleşmesi kabul edilir
        - Soru çoklu sonuç istiyorsa (`ve`/`and`/`listele`/`say`) listeleme bonusu
        """
        query_terms = _content_terms(query)
        if not query_terms:
            return 0.0

        distinct: List[str] = []
        for term in query_terms:
            key = _stem(term)
            if key not in distinct:
                distinct.append(key)
        if not distinct:
            return 0.0

        lowered_answer = str(answer or "").lower()
        answer_keys = _terms_set(answer)

        covered = 0
        for key in distinct:
            if key in answer_keys:
                covered += 1
            elif len(key) >= 4 and key in lowered_answer:
                # Gövde eşleşmesi: "sıcaklık" → "sıcaklığı"
                covered += 1

        score = covered / len(distinct)

        # Listeleme bonusu
        lowered_query = str(query or "").lower()
        wants_enumeration = (
            len(distinct) >= 3
            and any(marker in lowered_query for marker in _ENUM_MARKERS)
        ) or len(re.findall(r"\bve\b|\band\b|,", lowered_query)) >= 2

        if wants_enumeration:
            bullet_count = len(_LIST_RE.findall(str(answer or "")))
            if bullet_count >= 3:
                score = min(1.0, score + 0.12)
            elif bullet_count >= 1:
                score = min(1.0, score + 0.06)

        return _clamp(score)

    # ─── Bileşen 3: Tutarlılık (Coherence) ──────────────────────────────

    def score_coherence(self, answer: str) -> float:
        """
        Yanıtın biçimsel tutarlılığı.

        Cümle sayısı/uzunluk bandı, tekrar eden cümleler, sarkan bağlaçlar,
        ardışık aynı başlangıçlar, aşırı büyük harf ve paragraf yapısı ölçülür.
        """
        text = str(answer or "").strip()
        if not text:
            return 0.0

        sentences = _sentences(text)
        if not sentences:
            return 0.0

        penalty = 0.0
        bonus = 0.0

        # 1) Uzunluk bandı (8–35 kelime ideal)
        lengths = [len(_tokens(s)) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        if 8.0 <= mean_len <= 35.0:
            bonus += 0.10
        elif mean_len < 5.0:
            penalty += 0.30
        elif mean_len < 8.0:
            penalty += 0.15
        elif mean_len > 50.0:
            penalty += 0.20
        elif mean_len > 35.0:
            penalty += 0.10

        # 2) Cümle sayısı
        if len(sentences) == 1 and lengths[0] > 40:
            penalty += 0.10

        # 3) Tekrar eden cümleler (Jaccard > 0.8)
        repeated = 0
        sentence_sets = [_terms_set(s) for s in sentences]
        for i in range(len(sentence_sets)):
            for j in range(i + 1, len(sentence_sets)):
                if sentence_sets[i] and _jaccard(sentence_sets[i], sentence_sets[j]) > 0.8:
                    repeated += 1
        if repeated:
            penalty += min(0.35, 0.15 * repeated)

        # 4) Sarkan bağlaçlar (cümle başında)
        dangling = 0
        for sentence in sentences:
            words = _tokens(sentence)
            if words and words[0] in _DANGLING_CONNECTORS:
                dangling += 1
        if dangling:
            penalty += min(0.20, 0.08 * dangling)

        # 5) Ardışık aynı başlangıç (>2 kez)
        openers = [(_tokens(s)[0] if _tokens(s) else "") for s in sentences]
        streak = 1
        for i in range(1, len(openers)):
            if openers[i] and openers[i] == openers[i - 1]:
                streak += 1
                if streak > 2:
                    penalty += 0.06
            else:
                streak = 1

        # 6) Aşırı BÜYÜK HARF
        letters = [c for c in text if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio > 0.6 and len(letters) > 40:
                penalty += 0.15
            elif upper_ratio > 0.4 and len(letters) > 40:
                penalty += 0.06

        # 7) Paragraf yapısı
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) >= 2 and len(sentences) >= 4:
            bonus += 0.10
        elif len(sentences) >= 6 and len(paragraphs) == 1:
            penalty += 0.06

        return _clamp(0.75 + bonus - penalty)

    # ─── Bileşen 4: Dayanaklılık (Grounding) ────────────────────────────

    def score_grounding(self, answer: str, context: Optional[List[str]] = None) -> float:
        """
        Yanıtın bağlama dayanma oranı.

        Bir yanıt cümlesi, herhangi bir bağlam öğesiyle **en az 2 içerik token'ı**
        paylaşıyorsa "dayanaklı" sayılır. Bağlam verilmemişse 0.5 (nötr) döner ve
        `evaluate` tarafında `baglam_yok` sorunu işaretlenir.
        """
        items = [str(c) for c in (context or [])]
        if not items:
            return GROUNDING_NO_CONTEXT

        sentences = _sentences(answer)
        if not sentences:
            return 0.0

        context_sets = [_terms_set(item) for item in items]
        context_sets = [s for s in context_sets if s]
        if not context_sets:
            return GROUNDING_NO_CONTEXT

        grounded = 0
        for sentence in sentences:
            sentence_terms = _terms_set(sentence)
            if not sentence_terms:
                continue
            if any(len(sentence_terms & ctx) >= 2 for ctx in context_sets):
                grounded += 1

        return _clamp(grounded / len(sentences))

    # ─── Bileşen 5: Kalibrasyon (Calibration) ───────────────────────────

    def score_calibration(self, answer: str, uncertainty: float) -> float:
        """
        Belirsizlik ile dil kesinliğinin uyumu.

        - Yüksek belirsizlik (>0.6) + kesin dil → ceza
        - Yüksek belirsizlik + çekince dili → ödül
        - Düşük belirsizlik + kesin dil → ödül
        - Düşük belirsizlik + aşırı çekince → hafif ceza
        """
        text = str(answer or "").lower()
        if not text.strip():
            return 0.0

        try:
            unc = _clamp(float(uncertainty))
        except (TypeError, ValueError):
            unc = 0.5

        hedge_count = sum(1 for marker in _HEDGE_WORDS if marker in text)
        assertive_count = sum(1 for marker in _ASSERTIVE_WORDS if marker in text)
        words = max(1, len(_tokens(text)))
        hedge_density = min(1.0, (hedge_count * 12.0) / words)
        assertive_density = min(1.0, (assertive_count * 12.0) / words)

        score = 0.75
        if unc > 0.6:
            if hedge_count == 0:
                score -= 0.40
            else:
                score += 0.15 * min(1.0, hedge_density * 2.0)
            if assertive_density > 0.0:
                score -= 0.15 * assertive_density
        elif unc < 0.4:
            if hedge_density > 0.4:
                score -= 0.20
            else:
                score += 0.15
            if assertive_density > 0.0:
                score += 0.05
        else:
            score += 0.05 if hedge_count else 0.0

        return _clamp(score)

    # ─── Karşılaştırma ─────────────────────────────────────────────────

    def compare(self, a: str, b: str, query: str) -> Dict[str, Any]:
        """
        İki yanıtı aynı sorgu için karşılaştırır.

        Args:
            a: Birinci yanıt
            b: İkinci yanıt
            query: Ortak sorgu

        Returns:
            `{"winner": "a|b|tie", "delta": float, "reasons": [...], "a": {...}, "b": {...}}`
        """
        self._counters["comparisons"] += 1.0
        try:
            report_a = self.evaluate(query, a)
            report_b = self.evaluate(query, b)
        except Exception as exc:
            logger.warning(f"Karşılaştırma hatası: {exc}")
            return {"winner": "tie", "delta": 0.0, "reasons": [f"karsilastirma_hatasi: {exc}"],
                    "a": {}, "b": {}}

        delta = float(report_a.score) - float(report_b.score)
        reasons: List[str] = []

        if abs(delta) < 0.02:
            winner = "tie"
            reasons.append("Puanlar neredeyse eşit (fark < 0.02)")
        else:
            winner = "a" if delta > 0 else "b"

        for key in WEIGHTS:
            left = float(report_a.components.get(key, 0.0))
            right = float(report_b.components.get(key, 0.0))
            if abs(left - right) >= 0.15:
                better = "A" if left > right else "B"
                reasons.append(f"{key}: {better} daha iyi ({left:.2f} vs {right:.2f})")

        if not reasons:
            reasons.append("Belirgin bir bileşen farkı yok")

        return {
            "winner": winner,
            "delta": round(delta, 4),
            "reasons": reasons,
            "a": report_a.to_dict(),
            "b": report_b.to_dict(),
        }

    # ─── Yardımcılar ───────────────────────────────────────────────────

    @staticmethod
    def _verdict_of(score: float) -> str:
        """Puanı verdikt etiketine çevirir."""
        for threshold, label in VERDICT_THRESHOLDS:
            if score >= threshold:
                return label
        return "poor"

    def _diagnose(
        self,
        components: Dict[str, float],
        query: str,
        answer: str,
        context: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Bileşen puanlarından sorun ve öneri listesi üretir."""
        issues: List[str] = []
        suggestions: List[str] = []

        if components.get("relevance", 0.0) < 0.4:
            issues.append("zayif_ilgi")
            suggestions.append("Yanıt sorunun anahtar kavramlarını doğrudan ele almalı")
        if components.get("coverage", 0.0) < 0.5:
            issues.append("eksik_kapsam")
            suggestions.append("Sorudaki terimlerin tamamı yanıtta karşılanmalı")
        if components.get("coherence", 0.0) < 0.5:
            issues.append("tutarlilik_sorunu")
            suggestions.append("Cümleleri kısaltın, tekrarları ve sarkan bağlaçları kaldırın")
        if not context:
            issues.append("baglam_yok")
            suggestions.append("Yanıtı kasada bulunan notlara dayandırın")
        elif components.get("grounding", 0.0) < 0.5:
            issues.append("zayif_dayanak")
            suggestions.append("İddiaları bağlamdaki kaynaklarla destekleyin")
        if components.get("calibration", 0.0) < 0.4:
            issues.append("kalibrasyon_uyumsuz")
            suggestions.append("Belirsizlik yüksekse çekince dili kullanın ('muhtemelen', 'olabilir')")
        if len(_tokens(answer)) < 10:
            issues.append("cok_kisa_yanit")

        return issues, suggestions

    # ─── İstatistikler ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        evaluations = self._counters["evaluations"]
        return {
            "name": self.name,
            "initialized": self._initialized,
            "evaluations": int(evaluations),
            "errors": int(self._counters["errors"]),
            "comparisons": int(self._counters["comparisons"]),
            "mean_score": round(self._counters["score_sum"] / evaluations, 4) if evaluations else 0.0,
            "weights": dict(WEIGHTS),
            "last_report": self._last_report or {},
        }
