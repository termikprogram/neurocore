"""
NeuroCore Dil Algılama Modülü
==============================
Durak kelime (stopword) profilleri, karakter/ek ipuçları ve yazı sistemi
(script) kapısı kullanarak metnin dilini ağır bağımlılık olmadan algılar.

Sinyaller:
    1. Durak kelime profili  — dile özgü fonksiyon kelimeleri (ana sinyal)
    2. Karakter/ek ipuçları  — ığüşıöç, ß, ñ, ç, sch/ung/keit, tion/eaux ...
    3. Yazı sistemi kapısı   — kiril/arap/ÇHC yazıları adayları daraltır
    4. Uzunluk belirsizliği  — kısa metinde güven 0.45 ile sınırlanır

Kullanım:
    detector = LanguageDetector()
    detector.initialize()
    guess = detector.detect("Merhaba, bugün hava çok güzel.")
    print(guess.code, guess.confidence)
"""

from __future__ import annotations

import codecs
import locale
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("language.detector")


# ─── Varsayılan Diller ve Yazı Sistemleri ──────────────────────────────────

DEFAULT_LANGUAGES: List[str] = [
    "tr", "en", "de", "fr", "es", "it", "pt", "nl", "ru", "ar", "zh", "ja", "ko",
]

#: Latın alfabesini kullanan diller (durak kelime profili karşılaştırılabilir)
LATIN_LANGUAGES: List[str] = ["tr", "en", "de", "fr", "es", "it", "pt", "nl"]

#: Yazı sistemi → o yazıyı kullanan aday diller
SCRIPT_LANGUAGES: Dict[str, List[str]] = {
    "cyrillic": ["ru"],
    "arabic": ["ar"],
    "han": ["zh"],
    "hiragana": ["ja"],
    "katakana": ["ja"],
    "hangul": ["ko"],
}

#: Geçerli yazı sistemi etiketleri (script_of çıktıları)
SCRIPT_NAMES: List[str] = [
    "latin", "cyrillic", "greek", "arabic", "han", "hangul",
    "hiragana", "katakana", "devanagari", "unknown", "mixed",
]


# ─── Durak Kelime Profilleri ───────────────────────────────────────────────
#
# Her dil için 40+ gerçek fonksiyon kelimesi. Bu sözlük modül içinde
# gömülüdür → hiçbir dosya/ağ bağımlılığı yoktur.

STOPWORDS: Dict[str, List[str]] = {
    # ── Türkçe ──
    "tr": [
        "ve", "bir", "bu", "şu", "o", "da", "de", "ki", "mi", "mı", "mu", "mü",
        "için", "ile", "ama", "fakat", "ancak", "çok", "daha", "en", "gibi",
        "kadar", "sonra", "önce", "göre", "her", "hem", "veya", "ya", "eğer",
        "ise", "olarak", "olan", "olduğu", "var", "yok", "değil", "ben", "sen",
        "biz", "siz", "onlar", "benim", "senin", "bizim", "sizin", "onun",
        "ne", "nasıl", "neden", "niçin", "nerede", "hangi", "kim", "şey",
        "zaman", "gün", "yıl", "de", "değil", "üzerinde", "arasında", "tüm",
        "bütün", "bazı", "hiç", "yine", "artık", "şimdi", "burada", "orada",
    ],
    # ── İngilizce ──
    "en": [
        "the", "a", "an", "and", "or", "but", "if", "then", "than", "of",
        "to", "in", "on", "at", "by", "for", "with", "without", "from",
        "into", "about", "as", "is", "are", "was", "were", "be", "been",
        "being", "am", "do", "does", "did", "have", "has", "had", "will",
        "would", "can", "could", "should", "may", "might", "must", "this",
        "that", "these", "those", "it", "its", "he", "she", "they", "them",
        "we", "you", "i", "my", "your", "his", "her", "our", "their", "not",
        "no", "yes", "very", "more", "most", "also", "there", "here", "when",
        "where", "which", "who", "what", "how", "why", "all", "any", "some",
        "such", "only", "just", "so", "up", "out", "over", "after", "before",
    ],
    # ── Almanca ──
    "de": [
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
        "einem", "einer", "eines", "und", "oder", "aber", "wenn", "dann",
        "als", "wie", "wo", "was", "wer", "warum", "weil", "dass", "ob",
        "ist", "sind", "war", "waren", "sein", "wird", "werden", "wurde",
        "haben", "hat", "hatte", "kann", "können", "muss", "soll", "will",
        "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "dich", "uns",
        "nicht", "kein", "keine", "mit", "ohne", "für", "von", "zu", "zum",
        "zur", "auf", "in", "im", "an", "am", "bei", "nach", "vor", "über",
        "unter", "durch", "gegen", "auch", "nur", "noch", "schon", "sehr",
        "mehr", "so", "hier", "dort", "jetzt", "immer", "alle", "etwas",
    ],
    # ── Fransızca ──
    "fr": [
        "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "mais",
        "donc", "or", "ni", "car", "que", "qui", "quoi", "dont", "où", "quand",
        "comment", "pourquoi", "est", "sont", "était", "étaient", "être",
        "été", "a", "ont", "avait", "avoir", "sera", "seront", "peut",
        "peuvent", "doit", "doivent", "je", "tu", "il", "elle", "on", "nous",
        "vous", "ils", "elles", "me", "te", "se", "mon", "ton", "son", "notre",
        "votre", "leur", "ne", "pas", "plus", "moins", "très", "aussi",
        "avec", "sans", "pour", "par", "sur", "sous", "dans", "entre", "vers",
        "chez", "ce", "cet", "cette", "ces", "son", "tout", "tous", "toute",
        "comme", "alors", "ainsi", "encore", "déjà", "ici", "là", "y", "en",
    ],
    # ── İspanyolca ──
    "es": [
        "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero",
        "sino", "porque", "pues", "que", "quien", "quienes", "cual", "cuando",
        "como", "donde", "es", "son", "era", "eran", "fue", "ser", "sido",
        "está", "están", "estaba", "estar", "ha", "han", "había", "haber",
        "tiene", "tienen", "puede", "debe", "yo", "tú", "él", "ella",
        "nosotros", "vosotros", "ellos", "ellas", "me", "te", "se", "mi", "tu",
        "su", "nuestro", "vuestro", "no", "sí", "más", "menos", "muy",
        "también", "con", "sin", "para", "por", "en", "sobre", "bajo", "entre",
        "hacia", "hasta", "desde", "este", "esta", "esto", "ese", "esa",
        "aquel", "todo", "todos", "toda", "algo", "nada", "ya", "aquí", "allí",
    ],
    # ── İtalyanca ──
    "it": [
        "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "e", "ed", "o",
        "od", "ma", "però", "perché", "poiché", "che", "chi", "cui", "quale",
        "quando", "come", "dove", "è", "sono", "era", "erano", "essere",
        "stato", "ha", "hanno", "aveva", "avere", "può", "possono", "deve",
        "io", "tu", "lui", "lei", "noi", "voi", "loro", "mi", "ti", "si", "mio",
        "tuo", "suo", "nostro", "vostro", "non", "più", "meno", "molto",
        "anche", "con", "senza", "per", "tra", "fra", "su", "in", "nel",
        "nella", "questo", "questa", "quello", "quella", "tutto", "tutti",
        "già", "ancora", "qui", "qua", "là", "come", "cioè", "quindi",
    ],
    # ── Portekizce ──
    "pt": [
        "o", "a", "os", "as", "um", "uma", "uns", "umas", "e", "ou", "mas",
        "porém", "porque", "pois", "que", "quem", "qual", "quando", "como",
        "onde", "é", "são", "era", "eram", "ser", "sido", "está", "estão",
        "estava", "estar", "tem", "têm", "tinha", "ter", "pode", "podem",
        "deve", "eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas", "me",
        "te", "se", "meu", "teu", "seu", "nosso", "vosso", "não", "sim",
        "mais", "menos", "muito", "também", "com", "sem", "para", "por", "em",
        "sobre", "entre", "até", "desde", "este", "esta", "isso", "esse",
        "essa", "aquele", "todo", "todos", "toda", "algo", "nada", "já",
        "ainda", "aqui", "ali", "então", "assim",
    ],
    # ── Felemenkçe ──
    "nl": [
        "de", "het", "een", "en", "of", "maar", "want", "omdat", "dat", "die",
        "wie", "wat", "wanneer", "waar", "hoe", "waarom", "is", "zijn", "was",
        "waren", "worden", "wordt", "heeft", "hebben", "had", "kan", "kunnen",
        "moet", "moeten", "wil", "ik", "jij", "hij", "zij", "wij", "jullie",
        "mij", "mijn", "jouw", "ons", "niet", "geen", "met", "zonder", "voor",
        "van", "naar", "op", "in", "aan", "bij", "over", "onder", "tussen",
        "ook", "nog", "al", "heel", "meer", "minder", "zo", "hier", "daar",
        "nu", "dan", "als", "deze", "dit", "alle", "iets", "niets",
    ],
    # ── Rusça ──
    "ru": [
        "и", "в", "не", "что", "он", "на", "я", "с", "как", "а", "то", "все",
        "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
        "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
        "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг",
        "ли", "если", "уже", "или", "ни", "быть", "был", "была", "были",
        "есть", "это", "этот", "эта", "эти", "для", "при", "чтобы", "мы",
        "они", "их", "его", "себя", "кто", "где", "почему", "чем", "очень",
    ],
    # ── Arapça ──
    "ar": [
        "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "التي",
        "الذي", "الذين", "كان", "كانت", "يكون", "تكون", "هو", "هي", "هم",
        "نحن", "أنا", "أنت", "لا", "ما", "لم", "لن", "إن", "أن", "قد", "كل",
        "بعض", "بين", "قبل", "بعد", "حتى", "إذا", "أو", "و", "ثم", "لكن",
        "عندما", "حيث", "كيف", "لماذا", "أين", "ماذا", "جدا", "أيضا", "هناك",
    ],
    # ── Çince ──
    "zh": [
        "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
        "没有", "看", "好", "自己", "这", "他", "她", "它", "我们", "他们",
        "这个", "那个", "什么", "怎么", "为什么", "哪里", "可以", "因为",
        "所以", "但是", "而且", "如果", "虽然", "还是", "已经", "非常",
    ],
    # ── Japonca ──
    "ja": [
        "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ",
        "ある", "いる", "も", "する", "から", "な", "こと", "として", "い",
        "や", "れる", "など", "なっ", "ない", "この", "ため", "その", "あっ",
        "よう", "また", "もの", "という", "あり", "まで", "られ", "なる",
        "へ", "か", "だ", "これ", "それ", "あれ", "です", "ます", "ました",
    ],
    # ── Korece ──
    "ko": [
        "이", "그", "저", "것", "수", "있다", "없다", "하다", "되다", "이다",
        "아니다", "및", "등", "때", "곳", "사람", "우리", "저는", "제가",
        "그리고", "그러나", "하지만", "그래서", "왜냐하면", "또한", "매우",
        "아주", "정말", "너무", "에서", "에게", "으로", "로", "와", "과",
        "은", "는", "을", "를", "에", "의", "도", "만", "부터", "까지",
    ],
}


# ─── Karakter / Ek İpuçları ────────────────────────────────────────────────

CHAR_HINTS: Dict[str, str] = {
    "tr": "ğüşıöçĞÜŞİÖÇ",
    "en": "",
    "de": "äöüßÄÖÜ",
    "fr": "éèêëàâçîïôûùüÿœÉÈÊÀÂÇÎÏÔÛÙ",
    "es": "ñ¿¡áéíóúüÑÁÉÍÓÚ",
    "it": "àèéìòùÀÈÉÌÒÙ",
    "pt": "ãõçáâêéíóôúàüÃÕÇÁÂÊÉÍÓÔÚ",
    "nl": "ijĳéëï",
    "ru": "",
    "ar": "",
    "zh": "",
    "ja": "",
    "ko": "",
}

#: Dile özgü ek/parça ipuçları — (ipucu, ağırlık) çiftleri
SUFFIX_HINTS: Dict[str, List[Tuple[str, float]]] = {
    "tr": [
        ("lar", 0.55), ("ler", 0.55), ("lık", 0.42), ("lik", 0.42),
        ("siz", 0.38), ("sız", 0.38), ("dir", 0.40), ("dır", 0.40),
        ("mek", 0.50), ("mak", 0.50), ("ma", 0.22), ("me", 0.22),
        ("yor", 0.62), ("acak", 0.42), ("ecek", 0.42), ("mış", 0.40),
    ],
    "en": [
        ("tion", 0.55), ("ing", 0.55), ("ness", 0.55), ("ment", 0.50),
        ("ould", 0.62), ("ight", 0.42), ("the", 0.42), ("ly", 0.26),
    ],
    "de": [
        ("sch", 0.58), ("ung", 0.62), ("keit", 0.68), ("heit", 0.66),
        ("chen", 0.55), ("ich", 0.42), ("eit", 0.50), ("lich", 0.58),
        ("zu", 0.24), ("ei", 0.22),
    ],
    "fr": [
        ("qu", 0.52), ("tion", 0.55), ("eaux", 0.70), ("eau", 0.52),
        ("ent", 0.36), ("ais", 0.46), ("ez", 0.38), ("oux", 0.50),
        ("ille", 0.50), ("ment", 0.40),
    ],
    "es": [
        ("ción", 0.68), ("ll", 0.42), ("ñ", 0.62), ("ado", 0.46),
        ("ido", 0.46), ("iendo", 0.55), ("iento", 0.50), ("dad", 0.48),
        ("á", 0.30), ("qué", 0.62),
    ],
    "it": [
        ("zione", 0.68), ("gli", 0.66), ("che", 0.42), ("mente", 0.50),
        ("ato", 0.36), ("ito", 0.36), ("ggio", 0.55), ("zione", 0.68),
    ],
    "pt": [
        ("ção", 0.68), ("ões", 0.62), ("não", 0.68), ("mente", 0.48),
        ("eiro", 0.46), ("inha", 0.44), ("lh", 0.46), ("muito", 0.55),
    ],
    "nl": [
        ("ij", 0.66), ("sch", 0.46), ("lijk", 0.62), ("heid", 0.60),
        ("aan", 0.32), ("een", 0.46), ("voor", 0.50), ("niet", 0.60),
    ],
    "ru": [
        ("что", 0.66), ("ого", 0.60), ("его", 0.55), ("ность", 0.68),
        ("ться", 0.70), ("ский", 0.62), ("ать", 0.55), ("ить", 0.55),
    ],
    "ar": [
        ("ال", 0.60), ("من", 0.55), ("في", 0.55), ("ية", 0.55),
        ("ون", 0.50), ("ات", 0.50), ("هذا", 0.65),
    ],
    "zh": [("的", 0.60), ("我们", 0.50), ("这个", 0.50), ("可以", 0.45)],
    "ja": [("です", 0.60), ("ます", 0.60), ("こと", 0.50), ("として", 0.45)],
    "ko": [("니다", 0.60), ("습니다", 0.62), ("하다", 0.50), ("에서", 0.45)],
}

#: Kod-benzeri metin tespiti için referans doğal dil kelimeleri
CODE_REFERENCE_WORDS: frozenset = frozenset({
    "ve", "bir", "için", "ile", "bu", "da", "de", "gibi", "çok", "daha",
    "the", "and", "for", "with", "this", "that", "from", "not", "are", "was",
    "der", "die", "das", "und", "ist", "nicht", "mit", "auf",
    "le", "la", "les", "des", "est", "une", "dans", "pour",
    "el", "los", "las", "una", "que", "con", "para", "como",
})


# ─── Tahmin Veri Yapısı ────────────────────────────────────────────────────

@dataclass
class LanguageGuess:
    """
    Tek bir dil tahmini.

    Attributes:
        code: ISO 639-1 dil kodu ('tr', 'en', ...) veya 'unknown'
        confidence: 0.0-1.0 güven (normalize edilmiş en yüksek skor)
        scores: Tüm adayların normalize edilmiş skorları (toplam ≈ 1.0)
        method: 'heuristic' | 'heuristic_short' | 'script' | 'unknown'
    """
    code: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    method: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sözlük döndür."""
        return {
            "code": self.code,
            "confidence": round(float(self.confidence), 4),
            "scores": {k: round(float(v), 4) for k, v in self.scores.items()},
            "method": self.method,
        }

    def __str__(self) -> str:  # pragma: no cover - kolaylık
        return f"{self.code} ({self.confidence:.2f}, {self.method})"


# ─── Unicode Yardımcıları ──────────────────────────────────────────────────

_WORD_RE: Pattern[str] = re.compile(r"[^\W\d_]+", re.UNICODE)
_SCRIPT_CHAR_CACHE: Dict[str, Optional[str]] = {}


def _char_script(ch: str) -> Optional[str]:
    """
    Tek bir karakterin yazı sistemini döndürür (önbellekli).

    Returns:
        'latin' | 'cyrillic' | ... | None (ilgisiz karakter: rakam, boşluk)
    """
    cached = _SCRIPT_CHAR_CACHE.get(ch)
    if cached is not None or ch in _SCRIPT_CHAR_CACHE:
        return cached

    script: Optional[str] = None
    if ch.isalpha():
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = ""
        if "LATIN" in name:
            script = "latin"
        elif "CYRILLIC" in name:
            script = "cyrillic"
        elif "GREEK" in name:
            script = "greek"
        elif "ARABIC" in name:
            script = "arabic"
        elif "HIRAGANA" in name:
            script = "hiragana"
        elif "KATAKANA" in name:
            script = "katakana"
        elif "HANGUL" in name:
            script = "hangul"
        elif "CJK" in name or "IDEOGRAPH" in name:
            script = "han"
        elif "DEVANAGARI" in name:
            script = "devanagari"

    if len(_SCRIPT_CHAR_CACHE) < 8192:
        _SCRIPT_CHAR_CACHE[ch] = script
    return script


def script_of(text: str) -> str:
    """
    Metnin baskın yazı sistemini ('script') belirler.

    Args:
        text: İncelenecek metin

    Returns:
        'latin' | 'cyrillic' | 'greek' | 'arabic' | 'han' | 'hangul' |
        'hiragana' | 'katakana' | 'devanagari' | 'mixed' | 'unknown'
    """
    if not text:
        return "unknown"

    counts: Counter = Counter()
    for ch in text:
        if ch.isspace() or ch.isdigit():
            continue
        script = _char_script(ch)
        if script:
            counts[script] += 1

    if not counts:
        return "unknown"

    if len(counts) == 1:
        return next(iter(counts))

    # Birden fazla yazı sistemi: toplamın %85'inden fazlası baskınsa onu döndür
    total = sum(counts.values())
    top_script, top_count = counts.most_common(1)[0]
    if top_count / total >= 0.85:
        return top_script

    # Latin + tek bir CJK yazısı (Japonca: kanji + kana) → baskın olanı seç
    non_latin = {k: v for k, v in counts.items() if k != "latin"}
    if "latin" not in counts and set(non_latin) <= {"han", "hiragana", "katakana"}:
        kana = counts.get("hiragana", 0) + counts.get("katakana", 0)
        if kana > 0:
            return "hiragana" if counts.get("hiragana", 0) >= counts.get("katakana", 0) else "katakana"
        return "han"

    return "mixed"


def _normalize_for_matching(text: str) -> str:
    """Metni küçük harfe indirger, Unicode'u NFC'ye getirir, boşlukları sadeleştirir."""
    lowered = text.lower().replace("\u0130", "i").replace("\u0131", "i")
    try:
        normalized = unicodedata.normalize("NFC", lowered)
    except (TypeError, ValueError):  # pragma: no cover - savunma
        normalized = lowered
    return re.sub(r"\s+", " ", normalized).strip()


def is_probably_code(text: str) -> bool:
    """
    Metnin doğal dil değil, kaynak kodu olma olasılığını sezgisel ölçer.

    Heuristikler:
        - Süslü parantez/noktalı virgül yoğunluğu
        - Girinti satırlarının oranı
        - Kod anahtar kelimeleri (def, return, class, import, const, ...)
        - Doğal dil referans kelimesi yokluğu

    Args:
        text: İncelenecek metin

    Returns:
        True → büyük olasılıkla kod
    """
    if not text or len(text.strip()) < 8:
        return False

    # Kısa örnekte zayıf sinyal → doğal dil kabul et
    if len(text.strip()) < 24:
        return False

    # Doğal dillerin çoğu Latin/Kiril/Arap değilse kod kararı verme
    script = script_of(text)
    if script in ("han", "hiragana", "katakana", "hangul", "arabic", "devanagari"):
        return False

    lines = text.splitlines() or [text]
    total_chars = max(1, len(text))

    brace_score = (text.count("{") + text.count("}") + text.count(";")) * 12.0 / total_chars
    paren_score = (text.count("(") + text.count(")")) * 4.0 / total_chars
    symbol_ratio = (
        text.count("=") + text.count(">") + text.count("<") + text.count("|")
        + text.count("#") + text.count("$")
    ) / total_chars
    indent_ratio = sum(1 for ln in lines if ln[:1] in (" ", "\t")) / max(1, len(lines))

    code_word_score = 0.0
    lowered = text.lower()
    for kw in ("def ", "return ", "class ", "import ", "function ", "const ",
               "self.", "print(", "();", " == ", " != ", "});"):
        if kw in lowered:
            code_word_score += 0.12

    words = _WORD_RE.findall(lowered)
    natural_hits = sum(1 for w in words if w in CODE_REFERENCE_WORDS)
    natural_ratio = natural_hits / max(1, len(words))

    score = (
        0.30 * min(1.0, brace_score)
        + 0.18 * min(1.0, paren_score)
        + 0.22 * min(1.0, symbol_ratio * 4.0)
        + 0.22 * indent_ratio
        + min(0.45, code_word_score)
        - 0.35 * min(1.0, natural_ratio * 8.0)
    )

    return score >= 0.34


def guess_encoding(data: bytes) -> str:
    """
    Byte dizisinin kodlamasını tahmin eder.

    Sıra: BOM → katı UTF-8 → locale varsayılanı → cp1252 → latin-1 → utf-8

    Args:
        data: Ham byte dizisi

    Returns:
        Python codec adı ('utf-8', 'utf-16-le', 'cp1252', ...)
    """
    if not data:
        return "utf-8"

    # 1) BOM tespiti
    if data.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if data.startswith(codecs.BOM_UTF32_LE) or data.startswith(codecs.BOM_UTF32_BE):
        return "utf-32"
    if data.startswith(codecs.BOM_UTF16_LE):
        return "utf-16"
    if data.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"

    # 2) Geçerli UTF-8 mi?
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # 3) Yerel tercih edilen kodlama
    try:
        preferred = locale.getpreferredencoding(False) or ""
    except Exception:  # pragma: no cover - savunma
        preferred = ""
    if preferred:
        try:
            data.decode(preferred)
            return preferred.lower()
        except (UnicodeDecodeError, LookupError):
            pass

    # 4) Windows/Latin yaygın kodlamaları
    for candidate in ("cp1252", "cp1254", "latin-1"):
        try:
            data.decode(candidate)
            return candidate
        except (UnicodeDecodeError, LookupError):
            continue

    return "utf-8"


# ─── Ana Dil Algılayıcı ───────────────────────────────────────────────────

class LanguageDetector:
    """
    Sezgisel dil algılayıcı.

    Durak kelime profilleri + karakter/ek ipuçları + yazı sistemi kapısı
    ile 13 dili gerçek olarak puanlar. Ağır bağımlılık yoktur; tüm veri
    modül içinde gömülüdür.

    Kullanım:
        detector = LanguageDetector()
        detector.initialize(config, event_bus)
        guess = detector.detect("Merhaba dünya")
        for g in detector.detect_all("Hello world", top_k=3):
            print(g.code, g.confidence)
    """

    name: str = "language.detector"

    #: Bu uzunluğun altındaki metinlerde güven sınırlanır
    SHORT_TEXT_CHARS: int = 12
    #: Bu token sayısının altındaki metinlerde güven sınırlanır
    SHORT_TEXT_WORDS: int = 3
    #: Kısa metinlerde izin verilen en yüksek güven
    SHORT_CONFIDENCE_CAP: float = 0.45

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — I/O yok."""
        self._initialized: bool = False
        self._config = None
        self._event_bus = None
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._supported: List[str] = list(DEFAULT_LANGUAGES)
        self._stats: Dict[str, int] = {
            "detect_calls": 0,
            "detect_all_calls": 0,
            "unknown_results": 0,
            "short_results": 0,
            "script_results": 0,
            "encoding_guesses": 0,
            "code_checks": 0,
            "guess_events": 0,
        }
        self._last_guess: Optional[str] = None

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config=None, event_bus=None) -> None:
        """
        Algılayıcıyı başlat. İdempotenttir: ikinci çağrı no-op'tur.

        Args:
            config: SystemConfig benzeri nesne (None → varsayılanlar)
            event_bus: Olay yolu (None → olay yayınlanmaz)
        """
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus

        supported = getattr(getattr(config, "language", None), "supported_languages", None)
        if isinstance(supported, (list, tuple)) and supported:
            self._supported = [str(code).strip().lower() for code in supported if str(code).strip()]
        else:
            self._supported = list(DEFAULT_LANGUAGES)

        self._load_default_profiles()
        self._initialized = True

        logger.info(
            f"Dil algılayıcı hazır: {len(self._profiles)} profil "
            f"({', '.join(sorted(self._profiles))})"
        )
        if self._event_bus:
            self._event_bus.emit(
                "language.detector_ready",
                {"profiles": sorted(self._profiles), "supported": self._supported},
                source=self.name,
            )

    def shutdown(self) -> None:
        """Kaynakları serbest bırak. İdempotenttir."""
        if not self._initialized:
            return
        self._profiles.clear()
        self._initialized = False
        logger.info("Dil algılayıcı kapatıldı")

    # ─── Profil Yönetimi ───────────────────────────────────────────────

    def _load_default_profiles(self) -> None:
        """Gömülü durak kelime + karakter ipucu profillerini yükler."""
        for code, words in STOPWORDS.items():
            self._profiles[code] = {
                "stopwords": {_normalize_for_matching(w) for w in words if w},
                "chars": {c for c in CHAR_HINTS.get(code, "") if c},
                "suffixes": list(SUFFIX_HINTS.get(code, [])),
                "words_raw": list(words),
            }

    def add_profile(self, code: str, stopwords: List[str], char_hints: str = "") -> None:
        """
        Yeni bir dil profili ekler (veya mevcut profili zenginleştirir).

        Args:
            code: Dil kodu ('tr', 'en', ...)
            stopwords: Dile özgü durak kelimeler
            char_hints: Dile özgü karakterler (ör: 'ğüşıöç')
        """
        normalized = self.normalize_code(code)
        if not normalized:
            return

        profile = self._profiles.setdefault(normalized, {
            "stopwords": set(), "chars": set(), "suffixes": [], "words_raw": [],
        })
        added = 0
        for word in stopwords or []:
            token = _normalize_for_matching(str(word))
            if token and token not in profile["stopwords"]:
                profile["stopwords"].add(token)
                profile["words_raw"].append(str(word))
                added += 1
        if char_hints:
            for ch in str(char_hints):
                if ch not in profile["chars"]:
                    profile["chars"].add(ch)
        if normalized not in self._supported:
            self._supported.append(normalized)

        logger.debug(f"Dil profili güncellendi: {normalized} (+{added} durak kelime)")

    # ─── Kod Normalizasyonu ────────────────────────────────────────────

    def normalize_code(self, code: str) -> str:
        """
        Dil kodunu ISO 639-1 iki harfli biçime indirger.

        Örnekler:
            'tr-TR' → 'tr', 'ENG' → 'en', 'deu' → 'de', 'tur' → 'tr',
            'fra' → 'fr', 'spa' → 'es', 'zh-Hans' → 'zh'

        Args:
            code: Ham dil kodu

        Returns:
            İki harfli kod veya boş string
        """
        if not code:
            return ""
        raw = str(code).strip().lower().replace("_", "-")
        if not raw:
            return ""
        base = raw.split("-")[0].split(".")[0]
        if len(base) == 2:
            return base
        from neurocore.language.dictionary import normalize_language_code
        return normalize_language_code(base)

    def is_supported(self, code: str) -> bool:
        """Verilen dil kodunun desteklenip desteklenmediğini söyler."""
        normalized = self.normalize_code(code)
        if not normalized:
            return False
        if normalized in self._supported:
            return True
        return normalized in self._profiles

    # ─── Yazı Sistemi ──────────────────────────────────────────────────

    def script_of(self, text: str) -> str:
        """Metnin baskın yazı sistemini döndürür (bkz. modül fonksiyonu)."""
        return script_of(text)

    # ─── Algılama ──────────────────────────────────────────────────────

    def detect(self, text: str) -> LanguageGuess:
        """
        Metnin dilini algılar.

        Args:
            text: Analiz edilecek metin

        Returns:
            En olası LanguageGuess; hiçbir sinyal yoksa
            LanguageGuess('unknown', 0.0, {}, 'unknown')
        """
        self._stats["detect_calls"] += 1

        if not text or not str(text).strip():
            self._stats["unknown_results"] += 1
            return LanguageGuess("unknown", 0.0, {}, "unknown")

        text = str(text)
        candidates = self._build_candidates(text)
        script = script_of(text)

        if not candidates:
            self._stats["unknown_results"] += 1
            return LanguageGuess("unknown", 0.0, {}, "unknown")

        words = self._tokenize(text)
        short = len(text.strip()) < self.SHORT_TEXT_CHARS or len(words) < self.SHORT_TEXT_WORDS
        method = "heuristic_short" if short else ("script" if script not in ("latin", "unknown") else "heuristic")

        raw_scores: Dict[str, float] = {}
        for code in candidates:
            score = self._score_language(text, words, code, script)
            if score > 0.0:
                raw_scores[code] = score

        # Durak kelime sinyali hiç yoksa ve metin çok kısaysa → bilinmiyor
        if not raw_scores:
            self._stats["unknown_results"] += 1
            if self._event_bus:
                self._stats["guess_events"] += 1
                self._event_bus.emit(
                    "language.detected",
                    {"code": "unknown", "confidence": 0.0, "method": "unknown", "length": len(text)},
                    source=self.name,
                )
            return LanguageGuess("unknown", 0.0, {}, "unknown")

        normalized = self._normalize_scores(raw_scores)
        best_code = max(normalized, key=lambda c: normalized[c])
        confidence = normalized[best_code]

        if short:
            confidence = min(confidence, self.SHORT_CONFIDENCE_CAP)
            self._stats["short_results"] += 1
        if script not in ("latin", "unknown"):
            self._stats["script_results"] += 1

        guess = LanguageGuess(
            code=best_code,
            confidence=round(float(confidence), 4),
            scores=normalized,
            method=method,
        )
        self._last_guess = best_code

        if self._event_bus:
            self._stats["guess_events"] += 1
            self._event_bus.emit(
                "language.detected",
                {
                    "code": guess.code,
                    "confidence": guess.confidence,
                    "method": guess.method,
                    "script": script,
                    "length": len(text),
                },
                source=self.name,
            )

        logger.debug(
            f"Dil algılandı: {guess.code} ({guess.confidence:.2f}, {guess.method}, "
            f"{len(words)} kelime)"
        )
        return guess

    def detect_all(self, text: str, top_k: int = 5) -> List[LanguageGuess]:
        """
        Metin için aday dilleri skor sırasına göre döndürür.

        Args:
            text: Analiz edilecek metin
            top_k: Döndürülecek maksimum aday sayısı

        Returns:
            LanguageGuess listesi (en yüksek skordan düşüğe)
        """
        self._stats["detect_all_calls"] += 1

        if not text or not str(text).strip():
            return []

        text = str(text)
        candidates = self._build_candidates(text)
        if not candidates:
            return []

        words = self._tokenize(text)
        text = str(text)
        script = script_of(text)
        short = len(text.strip()) < self.SHORT_TEXT_CHARS or len(words) < self.SHORT_TEXT_WORDS

        raw_scores: Dict[str, float] = {}
        for code in candidates:
            score = self._score_language(text, words, code, script)
            if score > 0.0:
                raw_scores[code] = score

        if not raw_scores:
            return []

        normalized = self._normalize_scores(raw_scores)
        results: List[LanguageGuess] = []
        for code in sorted(normalized, key=lambda c: normalized[c], reverse=True):
            confidence = normalized[code]
            if short:
                confidence = min(confidence, self.SHORT_CONFIDENCE_CAP)
            results.append(LanguageGuess(
                code=code,
                confidence=round(float(confidence), 4),
                scores=normalized,
                method="heuristic_short" if short else (
                    "script" if script not in ("latin", "unknown") else "heuristic"
                ),
            ))
        return results[:max(1, int(top_k))]

    # ─── Puanlama ──────────────────────────────────────────────────────

    def _build_candidates(self, text: str) -> List[str]:
        """Yazı sistemi kapısına göre aday dil listesini daraltır."""
        script = script_of(text)

        if script in SCRIPT_LANGUAGES:
            allowed = [c for c in SCRIPT_LANGUAGES[script] if c in self._profiles]
            if allowed:
                return allowed
        elif script == "mixed":
            counted: Counter = Counter()
            for ch in text:
                s = _char_script(ch)
                if s:
                    counted[s] += 1
            # Japonca karışık yazı: kanji + kana
            if set(counted) <= {"han", "hiragana", "katakana", "latin"}:
                if counted.get("hiragana", 0) or counted.get("katakana", 0):
                    return [c for c in ("ja", "zh") if c in self._profiles]
            return [c for c in LATIN_LANGUAGES if c in self._profiles]

        return [c for c in LATIN_LANGUAGES if c in self._profiles]

    def _score_language(self, text: str, words: List[str], code: str, script: str) -> float:
        """
        Tek bir dil için ham skor hesaplar.

        Bileşenler:
            - Durak kelime isabetleri (frekans ağırlıklı)
            - Karakter ipuçları
            - Ek/parça ipuçları
        """
        profile = self._profiles.get(code)
        if not profile:
            return 0.0

        stopwords = profile["stopwords"]
        score = 0.0

        # 1) Durak kelime sinyali — ana sinyal
        if stopwords:
            freq: Counter = Counter()
            for word in words:
                token = _normalize_for_matching(word)
                if token in stopwords:
                    freq[token] += 1
            if freq:
                for count in freq.values():
                    score += 1.0 + 0.5 * math.log(1.0 + count)

        # 2) Karakter ipuçları
        chars = profile["chars"]
        if chars:
            lowered = text.lower()
            char_hits = sum(1 for ch in chars if ch in lowered)
            if char_hits:
                score += min(2.5, char_hits * 0.35)
            # Türkçe noktalı/noktasız i ayrımı güçlü bir sinyaldir
            if code == "tr" and ("ı" in lowered or "İ" in text):
                score += 0.35

        # 3) Ek/parça ipuçları
        lowered_text = _normalize_for_matching(text)
        for hint, weight in profile["suffixes"]:
            if hint in lowered_text:
                score += weight

        # Latin dışı yazılarda durak kelime eşleşmesi zayıfsa skoru düşür
        if script in ("cyrillic", "arabic", "han", "hiragana", "katakana", "hangul"):
            if score <= 0.0:
                return 0.0

        return max(0.0, score)

    @staticmethod
    def _normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Ham skorları toplamı 1.0 olacak biçimde normalize eder."""
        if not raw_scores:
            return {}
        max_score = max(raw_scores.values())
        if max_score <= 0.0:
            return {}
        # Adaylar arasındaki ayrımı keskinleştir (üst skor öne çıksın)
        boosted = {code: (value / max_score) ** 2.5 for code, value in raw_scores.items()}
        total = sum(boosted.values())
        if total <= 0.0:
            return {}
        normalized = {code: value / total for code, value in boosted.items()}
        return {code: round(float(value), 4) for code, value in normalized.items()}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Metni harf tabanlı kelimelere ayırır (rakam/noktalama atılır)."""
        if not text:
            return []
        return [match.group(0) for match in _WORD_RE.finditer(text)]

    # ─── Yardımcılar ───────────────────────────────────────────────────

    def stopwords_for(self, code: str) -> List[str]:
        """
        Bir dilin durak kelime listesini döndürür.

        Args:
            code: Dil kodu

        Returns:
            Durak kelimeler (yoksa boş liste); önce en sık kullanılanlar
        """
        normalized = self.normalize_code(code)
        profile = self._profiles.get(normalized)
        if not profile:
            return []
        words = list(profile.get("words_raw", []))
        if words:
            return words
        return sorted(profile["stopwords"])

    def stopword_set(self, code: str) -> set:
        """Dilin normalize edilmiş durak kelime kümesini döndürür."""
        normalized = self.normalize_code(code)
        profile = self._profiles.get(normalized)
        if not profile:
            return set()
        return set(profile["stopwords"])

    def encoding(self, data: bytes) -> str:
        """Byte dizisinin kodlamasını tahmin eder (istatistikleri günceller)."""
        self._stats["encoding_guesses"] += 1
        return guess_encoding(data)

    def is_code(self, text: str) -> bool:
        """Metnin kaynak kodu olup olmadığını sezgisel ölçer."""
        self._stats["code_checks"] += 1
        return is_probably_code(text)

    def profile_summary(self) -> Dict[str, Any]:
        """Yüklü profillerin özetini döndürür (teşhis için)."""
        return {
            code: {
                "stopwords": len(profile["stopwords"]),
                "chars": len(profile["chars"]),
                "suffix_hints": len(profile["suffixes"]),
            }
            for code, profile in sorted(self._profiles.items())
        }

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "profiles": len(self._profiles),
            "languages": sorted(self._profiles),
            "supported": list(self._supported),
            "total_stopwords": sum(len(p["stopwords"]) for p in self._profiles.values()),
            "last_guess": self._last_guess,
            **self._stats,
        }
