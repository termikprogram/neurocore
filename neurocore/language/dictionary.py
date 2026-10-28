"""
NeuroCore Terim Sözlüğü Modülü
===============================
İki dilli terim deposu (terminology store): JSON kalıcılığı, en-uzun-eşleşme
öncelikli çok kelimeli terim taraması, ifade uygulama ve kasa (vault)
bağlama desteği.

Varsayılan diller: Türkçe ↔ İngilizce teknik sözlük (AI/ML, donanım,
bilim, genel). Sözlük dosyası varsayılan olarak:
    <data_path>/terminology.json
    (data_path yoksa) <vault_path>/_index/terminology.json

Kullanım:
    store = TerminologyStore()
    store.initialize(config, event_bus)
    store.seed_default_bilingual()
    entry = store.lookup("gradyan", "tr", "en")
    new_text, applied = store.apply_phrase("yapay sinir ağı", "tr", "en")
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("language.dictionary")


# ─── Dil Kodu Normalizasyonu ───────────────────────────────────────────────

#: ISO 639-2/T ve 639-2/B kodları → ISO 639-1
ISO3_TO_ISO1: Dict[str, str] = {
    # Türkçe
    "tur": "tr",
    # İngilizce
    "eng": "en",
    # Almanca
    "deu": "de", "ger": "de",
    # Fransızca
    "fra": "fr", "fre": "fr",
    # İspanyolca
    "spa": "es",
    # İtalyanca
    "ita": "it",
    # Portekizce
    "por": "pt",
    # Felemenkçe
    "nld": "nl", "dut": "nl",
    # Rusça
    "rus": "ru",
    # Arapça
    "ara": "ar",
    # Çince
    "zho": "zh", "chi": "zh",
    # Japonca
    "jpn": "ja",
    # Korece
    "kor": "ko",
    # Yunanca / Latince / diğer yaygın
    "ell": "el", "gre": "el",
    "lat": "la",
    "heb": "he",
    "hin": "hi",
    "fas": "fa", "per": "fa",
    "ukr": "uk",
    "pol": "pl",
    "swe": "sv",
    "dan": "da",
    "nor": "no",
    "fin": "fi",
    "ces": "cs", "cze": "cs",
    "ron": "ro", "rum": "ro",
    "hun": "hu",
    "bul": "bg",
}

#: Dil kodundan insan-okur etiket
LANGUAGE_LABELS: Dict[str, str] = {
    "tr": "Türkçe", "en": "İngilizce", "de": "Almanca", "fr": "Fransızca",
    "es": "İspanyolca", "it": "İtalyanca", "pt": "Portekizce", "nl": "Felemenkçe",
    "ru": "Rusça", "ar": "Arapça", "zh": "Çince", "ja": "Japonca",
    "ko": "Korece", "el": "Yunanca", "la": "Latince",
}

#: Varsayılan sözlük dosya adı
DEFAULT_TERMINOLOGY_FILENAME: str = "terminology.json"

#: Sözlük dosya şema sürümü
TERMINOLOGY_SCHEMA_VERSION: int = 1


def normalize_language_code(code: str) -> str:
    """
    Dil kodunu ISO 639-1 iki harfli biçime indirger.

    Örnekler:
        'tr-TR' → 'tr', 'ENG' → 'en', 'deu' → 'de', 'tur' → 'tr',
        'fra' → 'fr', 'spa' → 'es', 'zh-Hans' → 'zh'

    Args:
        code: Ham dil kodu

    Returns:
        İki harfli kod; tanınmazsa küçük harfli taban kod (veya '')
    """
    if not code:
        return ""
    raw = str(code).strip().lower().replace("_", "-")
    if not raw:
        return ""
    base = raw.split("-")[0].split(".")[0].strip()
    if not base:
        return ""
    if len(base) == 2:
        return base
    if len(base) == 3 and base in ISO3_TO_ISO1:
        return ISO3_TO_ISO1[base]
    return base


def _key(term: str, src: str, dst: str) -> Tuple[str, str, str]:
    """Dizin anahtarı üretir: (küçük harf terim, kaynak dil, hedef dil)."""
    return (_normalize_term(term), normalize_language_code(src), normalize_language_code(dst))


def _normalize_term(term: str) -> str:
    """
    Terimi ve taranan metni eşleştirme için sadeleştirir.

    **Kanonik normalizasyon** — hem sözlük anahtarları hem de taranan metin
    bu fonksiyondan geçmelidir, aksi hâlde Türkçe noktalı/noktasız i
    (İ U+0130, ı U+0131) nedeniyle eşleşmeler kaçar.

    Dönüşümler karakter sayısını **korur** (1 karakter → 1 karakter), böylece
    elde edilen konumlar özgün metinle hizalı kalır:
        'İ' → 'i', 'ı' → 'i', NFC, küçük harf, tekil boşluk.
    """
    if not term:
        return ""
    lowered = str(term).replace("\u0130", "i").lower()
    try:
        lowered = unicodedata.normalize("NFC", lowered)
    except (TypeError, ValueError):  # pragma: no cover - savunma
        pass
    return re.sub(r"\s+", " ", lowered).strip()


def word_boundary_pattern(term: str, flags: int = re.UNICODE) -> Pattern[str]:
    """
    Bir terim için kelime sınırına saygılı, derlenmiş arama deseni üretir.

    Desen yapısı: `(?<![^\\W\\d_])TERİM(?![^\\W\\d_])`

    Not: `[^\\W\\d_]` sınıfına nicelik verilmez. Aksi hâlde Türkçe noktasız
    'ı' (U+0131) gibi karakterlerde Python'un büyük/küçük harf katlaması
    sıfır genişlikli eşleşmeye yol açar ve desen **hiçbir zaman** eşleşmez.

    Args:
        term: Aranacak terim (özgün biçim)
        flags: `re` modül bayrakları

    Returns:
        Derlenmiş regex deseni
    """
    return re.compile(
        r"(?<![^\W\d_])" + re.escape(_normalize_term(term)) + r"(?![^\W\d_])",
        flags,
    )


# ─── Terim Kaydı ───────────────────────────────────────────────────────────

@dataclass
class TerminologyEntry:
    """
    Tek bir iki dilli terim kaydı.

    Attributes:
        term: Kaynak dildeki terim
        translation: Hedef dildeki karşılığı
        src: Kaynak dil kodu ('tr')
        dst: Hedef dil kodu ('en')
        domain: Alan etiketi ('ai', 'hardware', 'science', 'general', ...)
        confidence: 0.0-1.0 güven
        note: Serbest not / kaynak açıklaması
        hits: Bu terimin kaç kez kullanıldığı (istatistik)
    """
    term: str
    translation: str
    src: str
    dst: str
    domain: str = "general"
    confidence: float = 0.8
    note: str = ""
    hits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """JSON (dosya şeması) biçimine çevirir."""
        return {
            "term": self.term,
            "translation": self.translation,
            "src": self.src,
            "dst": self.dst,
            "domain": self.domain,
            "confidence": round(float(self.confidence), 4),
            "note": self.note,
        }

    def to_public_dict(self) -> Dict[str, Any]:
        """Dışa dönük (API) sözlük gösterimi."""
        data = self.to_dict()
        data["hits"] = self.hits
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TerminologyEntry":
        """Dosyadan/sözlükten kayıt üretir; eksik alanlara toleranslıdır."""
        return cls(
            term=str(data.get("term", "")).strip(),
            translation=str(data.get("translation", "")).strip(),
            src=normalize_language_code(str(data.get("src", ""))),
            dst=normalize_language_code(str(data.get("dst", ""))),
            domain=str(data.get("domain", "general") or "general"),
            confidence=float(data.get("confidence", 0.8) or 0.8),
            note=str(data.get("note", "") or ""),
        )


# ─── Varsayılan İki Dilli Teknik Sözlük (tr ↔ en) ──────────────────────────
#
# Gerçek, doğru karşılıklar. Alan etiketleri: 'ai', 'hardware', 'science',
# 'general'. Kullanıcı istekleri ve mimari doküman terimleriyle uyumludur.

DEFAULT_TR_EN_GLOSSARY: List[Tuple[str, str, str]] = [
    # ── Yapay zeka / makine öğrenmesi ──
    ("yapay zeka", "artificial intelligence", "ai"),
    ("makine öğrenmesi", "machine learning", "ai"),
    ("derin öğrenme", "deep learning", "ai"),
    ("yapay sinir ağı", "neural network", "ai"),
    ("sinir ağı", "neural network", "ai"),
    ("pekiştirmeli öğrenme", "reinforcement learning", "ai"),
    ("denetimli öğrenme", "supervised learning", "ai"),
    ("denetimsiz öğrenme", "unsupervised learning", "ai"),
    ("gömme", "embedding", "ai"),
    ("dikkat", "attention", "ai"),
    ("öz dikkat", "self-attention", "ai"),
    ("eğitim", "training", "ai"),
    ("çıkarım", "inference", "ai"),
    ("niceleme", "quantization", "ai"),
    ("aşırı öğrenme", "overfitting", "ai"),
    ("geri yayılım", "backpropagation", "ai"),
    ("kayıp fonksiyonu", "loss function", "ai"),
    ("öğrenme oranı", "learning rate", "ai"),
    ("gradyan", "gradient", "ai"),
    ("gradyan inişi", "gradient descent", "ai"),
    ("toplu iş", "batch", "ai"),
    ("bellek bankası", "memory bank", "ai"),
    ("uzman karışımı", "mixture of experts", "ai"),
    ("bilgi grafı", "knowledge graph", "ai"),
    ("sürümleme", "versioning", "ai"),
    ("özet", "summary", "ai"),
    ("kaynak", "source", "ai"),
    ("güven", "confidence", "ai"),
    ("bağlam", "context", "ai"),
    ("istem", "prompt", "ai"),
    ("jeton", "token", "ai"),
    ("jetonlaştırma", "tokenization", "ai"),
    ("belirteç", "token", "ai"),
    ("bağlantı", "link", "ai"),
    ("not", "note", "ai"),
    ("arama", "search", "ai"),
    ("kasa", "vault", "ai"),
    ("gömme boyutu", "embedding dimension", "ai"),
    ("dönüştürücü", "transformer", "ai"),
    ("kodlayıcı", "encoder", "ai"),
    ("kod çözücü", "decoder", "ai"),
    ("katman", "layer", "ai"),
    ("ağırlık", "weight", "ai"),
    ("etkinleştirme fonksiyonu", "activation function", "ai"),
    ("kapsülleme", "embedding", "ai"),
    ("sinir ağı", "neural network", "ai"),
    ("yapay sinir ağları", "neural networks", "ai"),
    ("veri kümesi", "dataset", "ai"),
    ("eğitim verisi", "training data", "ai"),
    ("doğrulama", "validation", "ai"),
    ("test kümesi", "test set", "ai"),
    ("genelleme", "generalization", "ai"),
    ("aşırı uyum", "overfitting", "ai"),
    ("yetersiz öğrenme", "underfitting", "ai"),
    ("düzenlileştirme", "regularization", "ai"),
    ("kayıp", "loss", "ai"),
    ("geri yayma", "backpropagation", "ai"),
    ("devir", "epoch", "ai"),
    ("yineleme", "iteration", "ai"),
    ("öğrenme hızı", "learning rate", "ai"),
    ("kümeleme", "clustering", "ai"),
    ("sınıflandırma", "classification", "ai"),
    ("bağlanım", "regression", "ai"),
    ("regresyon", "regression", "ai"),
    ("öznitelik", "feature", "ai"),
    ("etiket", "label", "ai"),
    ("model", "model", "ai"),
    ("model ağırlıkları", "model weights", "ai"),
    ("parametre", "parameter", "ai"),
    ("hiperparametre", "hyperparameter", "ai"),
    ("mimari", "architecture", "ai"),
    ("kılavuzlu ince ayar", "supervised fine-tuning", "ai"),
    ("ince ayar", "fine-tuning", "ai"),
    ("insan geri bildirimiyle pekiştirmeli öğrenme", "reinforcement learning from human feedback", "ai"),
    ("kendini dikkat", "self-attention", "ai"),
    ("çok başlı dikkat", "multi-head attention", "ai"),
    ("konum kodlaması", "positional encoding", "ai"),
    ("sözcük dağarcığı", "vocabulary", "ai"),
    ("gizli durum", "hidden state", "ai"),
    ("kayan pencere", "sliding window", "ai"),
    ("kontrol noktası", "checkpoint", "ai"),
    ("dağıtım", "deployment", "ai"),
    ("uç cihaz", "edge device", "ai"),
    ("damıtma", "distillation", "ai"),
    ("budama", "pruning", "ai"),
    ("seyreklik", "sparsity", "ai"),
    ("karışıklık matrisi", "confusion matrix", "ai"),
    ("kesinlik", "precision", "ai"),
    ("duyarlılık", "recall", "ai"),
    ("f1 skoru", "f1 score", "ai"),
    ("kosinüs benzerliği", "cosine similarity", "ai"),
    ("vektör", "vector", "ai"),
    ("vektör veritabanı", "vector database", "ai"),
    ("geri getirme", "retrieval", "ai"),
    ("üretim", "generation", "ai"),
    ("büyük dil modeli", "large language model", "ai"),
    ("doğal dil işleme", "natural language processing", "ai"),
    ("bilgisayarlı görü", "computer vision", "ai"),
    ("konuşma tanıma", "speech recognition", "ai"),
    ("görüntü işleme", "image processing", "ai"),
    ("sınıflandırıcı", "classifier", "ai"),
    ("eğitim döngüsü", "training loop", "ai"),
    ("kayıp eğrisi", "loss curve", "ai"),
    ("gömme katmanı", "embedding layer", "ai"),
    ("rastgele başlatma", "random initialization", "ai"),
    ("rastgelelik", "randomness", "ai"),
    ("gizli katman", "hidden layer", "ai"),
    ("çıktı katmanı", "output layer", "ai"),
    ("girdi katmanı", "input layer", "ai"),
    ("etkinleştirme", "activation", "ai"),
    ("yönlendirici", "router", "ai"),
    ("uzman", "expert", "ai"),
    ("belirsizlik", "uncertainty", "ai"),
    # ── Donanım / sistem ──
    ("işlemci", "processor", "hardware"),
    ("işlemci çekirdeği", "processor core", "hardware"),
    ("çekirdek", "core", "hardware"),
    ("iş parçacığı", "thread", "hardware"),
    ("önbellek", "cache", "hardware"),
    ("sanal bellek", "virtual memory", "hardware"),
    ("işletim sistemi", "operating system", "hardware"),
    ("süreç", "process", "hardware"),
    ("dosya sistemi", "filesystem", "hardware"),
    ("güç kaynağı", "power supply", "hardware"),
    ("soğutma", "cooling", "hardware"),
    ("sıcaklık", "temperature", "hardware"),
    ("darboğaz", "bottleneck", "hardware"),
    ("gecikme", "latency", "hardware"),
    ("verim", "throughput", "hardware"),
    ("bellek", "memory", "hardware"),
    ("bellek kullanımı", "memory usage", "hardware"),
    ("depolama", "storage", "hardware"),
    ("disk", "disk", "hardware"),
    ("anakart", "motherboard", "hardware"),
    ("ekran kartı", "graphics card", "hardware"),
    ("güç tüketimi", "power consumption", "hardware"),
    ("klavye", "keyboard", "hardware"),
    ("fan", "fan", "hardware"),
    ("termal yönetim", "thermal management", "hardware"),
    ("işlemci kullanımı", "cpu usage", "hardware"),
    ("bellek sızıntısı", "memory leak", "hardware"),
    ("çekirdek sayısı", "core count", "hardware"),
    ("sanal makine", "virtual machine", "hardware"),
    ("kapsayıcı", "container", "hardware"),
    ("ağ", "network", "hardware"),
    ("bant genişliği", "bandwidth", "hardware"),
    ("önbellek isabeti", "cache hit", "hardware"),
    ("yazılım", "software", "hardware"),
    ("donanım", "hardware", "hardware"),
    ("sürücü", "driver", "hardware"),
    ("yapılandırma", "configuration", "hardware"),
    ("kıyaslama", "benchmark", "hardware"),
    ("optimizasyon", "optimization", "hardware"),
    # ── Bilim / akademik ──
    ("veri", "data", "science"),
    ("bilgi", "information", "science"),
    ("bilgi tabanı", "knowledge base", "science"),
    ("analiz", "analysis", "science"),
    ("yöntem", "method", "science"),
    ("sonuç", "result", "science"),
    ("sonuçlar", "results", "science"),
    ("deney", "experiment", "science"),
    ("hipotez", "hypothesis", "science"),
    ("teori", "theory", "science"),
    ("kuram", "theory", "science"),
    ("gözlem", "observation", "science"),
    ("ölçüm", "measurement", "science"),
    ("nicel", "quantitative", "science"),
    ("nitel", "qualitative", "science"),
    ("istatistik", "statistics", "science"),
    ("olasılık", "probability", "science"),
    ("ortalama", "mean", "science"),
    ("standart sapma", "standard deviation", "science"),
    ("örneklem", "sample", "science"),
    ("değişken", "variable", "science"),
    ("kanıt", "evidence", "science"),
    ("literatür", "literature", "science"),
    ("atıf", "citation", "science"),
    ("kaynakça", "bibliography", "science"),
    ("akademik", "academic", "science"),
    ("araştırma", "research", "science"),
    ("geliştirme", "development", "science"),
    ("mühendislik", "engineering", "science"),
    ("matematik", "mathematics", "science"),
    ("fizik", "physics", "science"),
    ("kimya", "chemistry", "science"),
    ("biyoloji", "biology", "science"),
    ("tıp", "medicine", "science"),
    ("enerji", "energy", "science"),
    ("kuvvet", "force", "science"),
    ("hız", "velocity", "science"),
    ("kütle", "mass", "science"),
    ("yoğunluk", "density", "science"),
    ("basınç", "pressure", "science"),
    ("hacim", "volume", "science"),
    ("algoritma", "algorithm", "science"),
    ("karmaşıklık", "complexity", "science"),
    ("hesaplama", "computation", "science"),
    ("benzetim", "simulation", "science"),
    ("modelleme", "modeling", "science"),
    ("görselleştirme", "visualization", "science"),
    ("nicelik", "quantity", "science"),
    ("yapı", "structure", "science"),
    ("süreç yönetimi", "process management", "science"),
    # ── Genel ──
    ("gün", "day", "general"),
    ("zaman", "time", "general"),
    ("saat", "hour", "general"),
    ("dakika", "minute", "general"),
    ("hafta", "week", "general"),
    ("ay", "month", "general"),
    ("yıl", "year", "general"),
    ("tarih", "date", "general"),
    ("dosya", "file", "general"),
    ("klasör", "folder", "general"),
    ("dizin", "directory", "general"),
    ("adres", "address", "general"),
    ("sayfa", "page", "general"),
    ("kitap", "book", "general"),
    ("makale", "article", "general"),
    ("yazı", "text", "general"),
    ("metin", "text", "general"),
    ("sözcük", "word", "general"),
    ("kelime", "word", "general"),
    ("cümle", "sentence", "general"),
    ("dil", "language", "general"),
    ("çeviri", "translation", "general"),
    ("sözlük", "dictionary", "general"),
    ("anlam", "meaning", "general"),
    ("örnek", "example", "general"),
    ("liste", "list", "general"),
    ("tablo", "table", "general"),
    ("şema", "schema", "general"),
    ("anahtar", "key", "general"),
    ("değer", "value", "general"),
    ("işlev", "function", "general"),
    ("işlevsellik", "functionality", "general"),
    ("yöntem çağrısı", "method call", "general"),
    ("hata", "error", "general"),
    ("uyarı", "warning", "general"),
    ("başarı", "success", "general"),
    ("başarısızlık", "failure", "general"),
    ("deneme", "attempt", "general"),
    ("sınır", "limit", "general"),
    ("ayar", "setting", "general"),
    ("seçenek", "option", "general"),
    ("kullanıcı", "user", "general"),
    ("sistem", "system", "general"),
    ("proje", "project", "general"),
    ("görev", "task", "general"),
    ("hedef", "target", "general"),
    ("amaç", "purpose", "general"),
    ("plan", "plan", "general"),
    ("adım", "step", "general"),
    ("süre", "duration", "general"),
    ("sonuç raporu", "result report", "general"),
    ("rapor", "report", "general"),
    ("güncelleme", "update", "general"),
    ("değişiklik", "change", "general"),
    ("düzeltme", "fix", "general"),
    ("iyileştirme", "improvement", "general"),
    ("gelişme", "progress", "general"),
    ("başlangıç", "start", "general"),
    ("bitiş", "end", "general"),
    ("orta", "middle", "general"),
    ("büyük", "large", "general"),
    ("küçük", "small", "general"),
    ("hızlı", "fast", "general"),
    ("yavaş", "slow", "general"),
    ("kolay", "easy", "general"),
    ("zor", "difficult", "general"),
    ("önemli", "important", "general"),
    ("gerekli", "required", "general"),
    ("yeterli", "sufficient", "general"),
    ("eksik", "missing", "general"),
    ("tam", "complete", "general"),
    ("boş", "empty", "general"),
    ("dolu", "full", "general"),
    ("yeni", "new", "general"),
    ("eski", "old", "general"),
    ("aynı", "same", "general"),
    ("farklı", "different", "general"),
    ("benzer", "similar", "general"),
    ("doğru", "correct", "general"),
    ("yanlış", "wrong", "general"),
    ("güvenli", "safe", "general"),
    ("risk", "risk", "general"),
    ("sorumluluk", "responsibility", "general"),
    ("disiplin", "discipline", "general"),
    ("çıktı", "output", "general"),
    ("girdi", "input", "general"),
    ("kişi", "person", "general"),
    ("insan", "human", "general"),
    ("toplum", "society", "general"),
    ("şirket", "company", "general"),
    ("ekip", "team", "general"),
    ("grup", "group", "general"),
    ("topluluk", "community", "general"),
    ("makine", "machine", "general"),
    ("cihaz", "device", "general"),
    ("araç", "tool", "general"),
    ("parça", "part", "general"),
    ("numara", "number", "general"),
    ("sayı", "number", "general"),
    ("miktar", "amount", "general"),
    ("oran", "rate", "general"),
    ("yüzde", "percent", "general"),
    ("durum", "status", "general"),
    ("aşama", "phase", "general"),
    ("seviye", "level", "general"),
    ("kalite", "quality", "general"),
    ("performans", "performance", "general"),
    ("kapasite", "capacity", "general"),
    ("hacim ölçüsü", "volume measure", "general"),
    ("yemek", "food", "general"),
    ("su", "water", "general"),
    ("hava", "air", "general"),
    ("ev", "house", "general"),
    ("şehir", "city", "general"),
    ("ülke", "country", "general"),
    ("dünya", "world", "general"),
    ("yaşam", "life", "general"),
    ("sağlık", "health", "general"),
    ("eğitim sistemi", "education system", "general"),
    ("okul", "school", "general"),
    ("öğrenci", "student", "general"),
    ("öğretmen", "teacher", "general"),
    ("para", "money", "general"),
    ("fiyat", "price", "general"),
    ("pazar", "market", "general"),
    ("ticaret", "trade", "general"),
    ("hizmet", "service", "general"),
    ("ürün", "product", "general"),
    ("müşteri", "customer", "general"),
]


# ─── Ana Terim Deposu ──────────────────────────────────────────────────────

class TerminologyStore:
    """
    İki dilli terim deposu.

    Özellikler:
        - JSON kalıcılığı (şema sürümlü, bozuk dosyaya toleranslı)
        - Bellek içi dizin: {(terim_küçük, src, dst): TerminologyEntry}
        - En uzun eşleşme öncelikli, kelime sınırına saygılı ifade taraması
        - Varsayılan tr↔en teknik sözlük tohumlama
        - Kasa (vault) notlarından güvenli terim bağlama

    Kullanım:
        store = TerminologyStore()
        store.initialize(config, event_bus)
        store.seed_default_bilingual()
        store.lookup_phrase("yapay sinir ağı ile", "tr", "en")
    """

    name: str = "language.dictionary"

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — I/O yok."""
        self._initialized: bool = False
        self._config = None
        self._event_bus = None
        self._entries: Dict[Tuple[str, str, str], TerminologyEntry] = {}
        self._index_dirty: bool = True
        self._phrase_index: Dict[Tuple[str, str], List[Tuple[str, Pattern[str]]]] = {}
        self._lock = threading.RLock()
        self._default_path: Optional[Path] = None
        self._loaded_from: str = ""
        self._stats: Dict[str, int] = {
            "added": 0,
            "updated": 0,
            "lookups": 0,
            "lookup_hits": 0,
            "phrase_lookups": 0,
            "phrase_matches": 0,
            "applied_terms": 0,
            "seeded": 0,
            "imported": 0,
            "vault_terms": 0,
            "saves": 0,
            "loads": 0,
            "errors": 0,
            "events": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config=None, event_bus=None) -> None:
        """
        Terim deposunu başlat. İdempotenttir: ikinci çağrı no-op'tur.

        Args:
            config: SystemConfig benzeri nesne (None → varsayılanlar)
            event_bus: Olay yolu (None → olay yayınlanmaz)
        """
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus
        self._default_path = self._resolve_default_path()

        loaded = 0
        if self._default_path and self._default_path.exists():
            loaded = self.load(str(self._default_path))

        self._initialized = True
        logger.info(
            f"Terim deposu hazır: {len(self._entries)} kayıt "
            f"(dosya: {self._default_path}, yüklenen: {loaded})"
        )
        if self._event_bus:
            self._stats["events"] += 1
            self._event_bus.emit(
                "language.dictionary_ready",
                {"entries": len(self._entries), "path": str(self._default_path or "")},
                source=self.name,
            )

    def shutdown(self) -> None:
        """Kaynakları serbest bırak. İdempotenttir (veri diske yazılmaz)."""
        if not self._initialized:
            return
        with self._lock:
            self._phrase_index.clear()
            self._index_dirty = True
        self._initialized = False
        logger.info(f"Terim deposu kapatıldı ({len(self._entries)} kayıt bellekte kaldı)")

    # ─── Yol Çözümleme ─────────────────────────────────────────────────

    def _resolve_default_path(self) -> Path:
        """
        Varsayılan sözlük dosyası yolunu belirler.

        Sıra: <data_path>/terminology.json → <vault_path>/_index/terminology.json
              → <base_path>/data/terminology.json
        """
        config = self._config

        data_path = getattr(config, "data_path", None) if config is not None else None
        if data_path:
            return Path(str(data_path)) / DEFAULT_TERMINOLOGY_FILENAME

        knowledge = getattr(config, "knowledge", None) if config is not None else None
        vault_path = getattr(knowledge, "vault_path", "") if knowledge is not None else ""
        if vault_path:
            return Path(str(vault_path)) / "_index" / DEFAULT_TERMINOLOGY_FILENAME

        base_path = getattr(config, "base_path", None) if config is not None else None
        if not base_path:
            base_path = str(Path(__file__).resolve().parents[2])
        return Path(str(base_path)) / "data" / DEFAULT_TERMINOLOGY_FILENAME

    def path(self) -> str:
        """Aktif sözlük dosyası yolunu döndürür."""
        if self._default_path is None:
            self._default_path = self._resolve_default_path()
        return str(self._default_path)

    # ─── Kalıcılık ─────────────────────────────────────────────────────

    def load(self, path: Optional[str] = None) -> int:
        """
        Sözlüğü JSON dosyasından yükler.

        Bozuk/eksik dosya sistemi çökertmez; hata loglanır ve 0 döner.

        Args:
            path: Dosya yolu (None → varsayılan yol)

        Returns:
            Yüklenen kayıt sayısı
        """
        target = Path(path) if path else Path(self.path())
        if not target.exists():
            logger.debug(f"Sözlük dosyası yok, atlanıyor: {target}")
            return 0

        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            self._stats["errors"] += 1
            logger.warning(f"Sözlük dosyası okunamadı ({target}): {exc}")
            return 0

        entries_raw: Sequence[Any] = []
        if isinstance(raw, dict):
            entries_raw = raw.get("entries", []) or []
        elif isinstance(raw, list):
            entries_raw = raw

        loaded = 0
        with self._lock:
            for item in entries_raw:
                if not isinstance(item, dict):
                    continue
                entry = TerminologyEntry.from_dict(item)
                if not entry.term or not entry.translation or not entry.src or not entry.dst:
                    continue
                self._entries[_key(entry.term, entry.src, entry.dst)] = entry
                loaded += 1
            self._index_dirty = True

        self._loaded_from = str(target)
        self._stats["loads"] += 1
        logger.info(f"Terim deposu yüklendi: {loaded} kayıt <- {target}")
        if self._event_bus:
            self._stats["events"] += 1
            self._event_bus.emit(
                "language.dictionary_loaded",
                {"entries": loaded, "path": str(target)},
                source=self.name,
            )
        return loaded

    def save(self, path: Optional[str] = None) -> str:
        """
        Sözlüğü JSON dosyasına yazar.

        Args:
            path: Dosya yolu (None → varsayılan yol)

        Returns:
            Yazılan dosya yolu (hata durumunda boş string)
        """
        target = Path(path) if path else Path(self.path())
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": TERMINOLOGY_SCHEMA_VERSION,
                "updated": datetime.now().isoformat(timespec="seconds"),
                "entries": [entry.to_dict() for entry in self._iter_entries()],
            }
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(target)
        except Exception as exc:
            self._stats["errors"] += 1
            logger.warning(f"Sözlük kaydedilemedi ({target}): {exc}")
            return ""

        self._stats["saves"] += 1
        if self._event_bus:
            self._stats["events"] += 1
            self._event_bus.emit(
                "language.dictionary_saved",
                {"entries": len(self._entries), "path": str(target)},
                source=self.name,
            )
        return str(target)

    def _iter_entries(self) -> List[TerminologyEntry]:
        """Kayıtları deterministik sırayla döndürür."""
        with self._lock:
            return sorted(
                self._entries.values(),
                key=lambda e: (e.src, e.dst, e.domain, e.term.lower()),
            )

    # ─── Kayıt Ekleme / Sorgu ──────────────────────────────────────────

    def add(
        self,
        term: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        domain: str = "general",
        confidence: float = 0.8,
        source_note: str = "",
    ) -> None:
        """
        Depoya terim ekler (varsa günceller).

        Args:
            term: Kaynak dildeki terim
            translation: Hedef dildeki karşılığı
            source_lang: Kaynak dil kodu
            target_lang: Hedef dil kodu
            domain: Alan etiketi
            confidence: 0.0-1.0 güven
            source_note: Not / kaynak açıklaması
        """
        term = (term or "").strip()
        translation = (translation or "").strip()
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        if not term or not translation or not src or not dst:
            logger.debug(f"Geçersiz terim kaydı atlandı: {term!r} -> {translation!r} ({src}->{dst})")
            return

        key = _key(term, src, dst)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                # Kullanıcı/elle eklenen kayıtlar varsayılan tohumları ezebilir
                if existing.confidence >= 0.98 and confidence < 0.98:
                    return
                if (existing.translation == translation and existing.domain == domain
                        and abs(existing.confidence - confidence) < 1e-6):
                    return
                existing.translation = translation
                existing.domain = domain
                existing.confidence = float(confidence)
                if source_note:
                    existing.note = source_note
                self._stats["updated"] += 1
                self._index_dirty = True
                return
            self._entries[key] = TerminologyEntry(
                term=term,
                translation=translation,
                src=src,
                dst=dst,
                domain=domain or "general",
                confidence=float(confidence),
                note=source_note or "",
            )
            self._index_dirty = True
        self._stats["added"] += 1

    def add_entry(self, entry: TerminologyEntry) -> None:
        """TerminologyEntry nesnesini doğrudan ekler."""
        self.add(
            entry.term, entry.translation, entry.src, entry.dst,
            domain=entry.domain, confidence=entry.confidence, source_note=entry.note,
        )

    def vocabulary_has(self, term: str, source_lang: str, target_lang: str) -> bool:
        """
        Terim bu dil çiftinde kayıtlı mı? (istatistik sayaçlarını değiştirmez)

        Args:
            term: Sorgulanan terim
            source_lang: Kaynak dil kodu
            target_lang: Hedef dil kodu

        Returns:
            True → kayıt var
        """
        key = _key(term, normalize_language_code(source_lang), normalize_language_code(target_lang))
        if not key[0]:
            return False
        with self._lock:
            return key in self._entries

    def lookup(self, term: str, source_lang: str, target_lang: str) -> Optional[Dict[str, Any]]:
        """
        Tek bir terimin karşılığını arar.

        Args:
            term: Aranacak terim
            source_lang: Kaynak dil kodu
            target_lang: Hedef dil kodu

        Returns:
            Kayıt sözlüğü ({'term','translation','src','dst','domain',
            'confidence','note','hits'}) veya None
        """
        self._stats["lookups"] += 1
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        key = _key(term, src, dst)
        if not key[0]:
            return None

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.hits += 1
            data = entry.to_public_dict()

        self._stats["lookup_hits"] += 1
        return data

    def lookup_phrase(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_terms: int = 8,
    ) -> Dict[str, str]:
        """
        Metin içindeki bilinen terimleri (çok kelimeli dahil) bulur.

        En uzun eşleşme önceliklidir: "yapay sinir ağı" varken "sinir ağı"
        ayrıca eşleşmez.

        Args:
            text: Taranacak metin
            source_lang: Kaynak dil kodu
            target_lang: Hedef dil kodu
            max_terms: Döndürülecek en fazla terim sayısı

        Returns:
            {kaynak_terim: hedef_terim} sözlüğü
        """
        self._stats["phrase_lookups"] += 1
        if not text:
            return {}

        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        pattern_entries = self._pattern_index(src, dst)
        if not pattern_entries:
            return {}

        lowered = _normalize_term(text)
        found: List[Tuple[int, int, str, str]] = []

        for _canonical, pattern in pattern_entries:
            for match in pattern.finditer(lowered):
                original = text[match.start():match.end()]
                entry = self._entries.get(_key(original, src, dst))
                if entry is None:
                    continue
                found.append((match.start(), match.end(), original, entry.translation))
                break  # her terim için en fazla bir eşleşme yeterli

        # Çakışan (iç içe geçen) eşleşmeleri uzunluk önceliğiyle çöz
        found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        accepted: List[Tuple[int, int, str, str]] = []
        for item in found:
            if any(item[0] < acc[1] and acc[0] < item[1] for acc in accepted):
                continue
            accepted.append(item)

        result: Dict[str, str] = {}
        for _start, _end, original, translation in accepted:
            if original in result:
                continue
            result[original] = translation
            if len(result) >= max(1, int(max_terms)):
                break

        if result:
            self._stats["phrase_matches"] += len(result)
        return result

    def apply_phrase(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        keep_original_in_parens: bool = False,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Metindeki bilinen terimleri hedef dile çevirir.

        Args:
            text: Kaynak metin
            source_lang: Kaynak dil kodu
            target_lang: Hedef dil kodu
            keep_original_in_parens: True → 'çeviri (özgün)' biçiminde yaz

        Returns:
            (yeni_metin, uygulanan_eşleşmeler) ikilisi
        """
        if not text:
            return text, {}

        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        pattern_entries = self._pattern_index(src, dst)
        if not pattern_entries:
            return text, {}

        lowered = _normalize_term(text)
        matches: List[Tuple[int, int, str, str]] = []

        for _canonical, pattern in pattern_entries:
            for match in pattern.finditer(lowered):
                original = text[match.start():match.end()]
                entry = self._entries.get(_key(original, src, dst))
                if entry is None:
                    continue
                replacement = entry.translation
                if keep_original_in_parens:
                    replacement = f"{replacement} ({original})"
                matches.append((match.start(), match.end(), original, replacement))
                break

        if not matches:
            return text, {}

        matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        accepted: List[Tuple[int, int, str, str]] = []
        for item in matches:
            if any(item[0] < acc[1] and acc[0] < item[1] for acc in accepted):
                continue
            accepted.append(item)

        pieces: List[str] = []
        applied: Dict[str, str] = {}
        cursor = 0
        for start, end, original, replacement in accepted:
            if start < cursor:
                continue
            pieces.append(text[cursor:start])
            pieces.append(replacement)
            cursor = end
            if original not in applied:
                applied[original] = replacement
        pieces.append(text[cursor:])

        new_text = re.sub(r"[ \t]{2,}", " ", "".join(pieces)).strip()
        if applied:
            self._stats["applied_terms"] += len(applied)
            for original in applied:
                key = _key(original, src, dst)
                entry = self._entries.get(key)
                if entry is not None:
                    entry.hits += 1
        return new_text, applied

    def terms_for(
        self,
        source_lang: str,
        target_lang: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Bir dil çiftinin tüm terimlerini döndürür.

        Args:
            source_lang: Kaynak dil kodu
            target_lang: Hedef dil kodu
            limit: En fazla kayıt sayısı

        Returns:
            Kayıt listesi (güvene göre azalan)
        """
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        with self._lock:
            rows = [
                entry.to_public_dict()
                for entry in self._entries.values()
                if entry.src == src and entry.dst == dst
            ]
        rows.sort(key=lambda row: (-float(row.get("confidence", 0.0)), row.get("term", "")))
        return rows[:max(1, int(limit))]

    def vocabulary(self, source_lang: str, target_lang: str) -> Dict[str, str]:
        """Terim çiftini düz sözlük olarak döndürür ({terim: çeviri})."""
        src = normalize_language_code(source_lang)
        dst = normalize_language_code(target_lang)
        with self._lock:
            return {
                entry.term: entry.translation
                for entry in self._entries.values()
                if entry.src == src and entry.dst == dst
            }

    def export_dict(self, source_lang: str, target_lang: str) -> Dict[str, str]:
        """Bir dil çiftini düz sözlük olarak dışa aktarır."""
        return dict(sorted(self.vocabulary(source_lang, target_lang).items()))

    def import_pairs(
        self,
        pairs: List[Tuple[str, str]],
        source_lang: str,
        target_lang: str,
        domain: str = "imported",
    ) -> int:
        """
        (terim, çeviri) çiftlerini toplu içe aktarır.

        Args:
            pairs: (kaynak terim, hedef terim) ikilileri
            source_lang: Kaynak dil kodu
            target_lang: Hedef dil kodu
            domain: Alan etiketi

        Returns:
            Eklenen kayıt sayısı (kopyalar sayılmaz)
        """
        added = 0
        with self._lock:
            for pair in pairs or []:
                try:
                    term, translation = pair[0], pair[1]
                except (TypeError, IndexError, KeyError):
                    continue
                before = len(self._entries)
                self.add(
                    str(term), str(translation), source_lang, target_lang,
                    domain=domain, confidence=0.8,
                )
                if len(self._entries) > before:
                    added += 1
        self._stats["imported"] += added
        logger.debug(f"İçe aktarma: {added} yeni terim ({source_lang}->{target_lang}, {domain})")
        return added

    def stats_by_domain(self) -> Dict[str, int]:
        """Alan etiketine göre kayıt sayıları."""
        counter: Counter = Counter()
        with self._lock:
            for entry in self._entries.values():
                counter[entry.domain or "general"] += 1
        return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))

    # ─── Tohumlama ─────────────────────────────────────────────────────

    def seed_default_bilingual(self) -> int:
        """
        Küratörlü tr↔en teknik sözlüğü tohumlar (çift yönlü).

        Hem tr→en hem en→tr kayıtları eklenir. Kullanıcı tarafından
        düzenlenmiş (confidence ≥ 0.98) kayıtlar korunur.

        Returns:
            Eklenen/güncellenen kayıt sayısı
        """
        pairs = [(tr, en) for tr, en, _domain in DEFAULT_TR_EN_GLOSSARY]
        domains = {tr: domain for tr, _en, domain in DEFAULT_TR_EN_GLOSSARY}

        before = len(self._entries)
        with self._lock:
            for term, translation in pairs:
                domain = domains.get(term, "general")
                self.add(term, translation, "tr", "en", domain=domain, confidence=0.95)
                self.add(translation, term, "en", "tr", domain=domain, confidence=0.95)
        added = max(0, len(self._entries) - before)
        self._stats["seeded"] += added

        if self._event_bus:
            self._stats["events"] += 1
            self._event_bus.emit(
                "language.glossary_seeded",
                {"pairs": len(pairs), "entries": len(self._entries), "added": added},
                source=self.name,
            )
        logger.info(
            f"Varsayılan sözlük tohumlandı: {len(pairs)} çift, "
            f"depo toplam {len(self._entries)} kayıt (+{added})"
        )
        return added

    # ─── Kasa Bağlama ──────────────────────────────────────────────────

    def auto_link_vault(self, vault_dir: Optional[str] = None) -> int:
        """
        Kasa dizinindeki markdown notlarından terim adayları çıkarır.

        Frontmatter `tags` ve `## ` başlıkları aday terim sayılır ve yalnızca
        kendi kendine çeviri (src == dst) olarak, zaten yoksa eklenir.
        Kasa dosyaları **asla değiştirilmez** — sadece okunur.

        Args:
            vault_dir: Kasa kök dizini (None → config.knowledge.vault_path)

        Returns:
            Eklenen terim sayısı
        """
        directory = vault_dir
        if not directory:
            knowledge = getattr(self._config, "knowledge", None) if self._config is not None else None
            directory = getattr(knowledge, "vault_path", "") if knowledge is not None else ""
        if not directory:
            logger.debug("auto_link_vault: kasa dizini verilmedi, atlanıyor")
            return 0

        root = Path(str(directory))
        if not root.is_dir():
            logger.debug(f"auto_link_vault: dizin yok, atlanıyor ({root})")
            return 0

        added = 0
        scanned = 0
        try:
            md_files = sorted(root.rglob("*.md"))
        except Exception as exc:
            self._stats["errors"] += 1
            logger.warning(f"auto_link_vault: dizin taranamadı ({root}): {exc}")
            return 0

        for md_file in md_files[:500]:
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            scanned += 1
            candidates = self._extract_terms_from_note(content)
            for term in candidates:
                key: Tuple[str, str, str] = (_normalize_term(term), "tr", "tr")
                with self._lock:
                    if key in self._entries:
                        continue
                self.add(term, term, "tr", "tr", domain="vault", confidence=0.7,
                         source_note=f"vault:{md_file.name}")
                added += 1

        self._stats["vault_terms"] += added
        logger.info(
            f"Kasa bağlama: {scanned} not tarandı, {added} yeni terim eklendi ({root})"
        )
        if self._event_bus and added:
            self._stats["events"] += 1
            self._event_bus.emit(
                "language.vault_linked",
                {"notes": scanned, "added": added, "vault": str(root)},
                source=self.name,
            )
        return added

    @staticmethod
    def _extract_terms_from_note(content: str) -> List[str]:
        """
        Bir markdown notundan terim adayları çıkarır.

        Kaynaklar: frontmatter `tags: [a, b]` / `tags: a, b` ve `## Başlık`.
        Kasa dosyası değiştirilmez.
        """
        terms: List[str] = []

        # Frontmatter etiketleri
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", content, re.DOTALL)
        if fm_match:
            block = fm_match.group(1)
            tags_match = re.search(r"^\s*tags\s*:\s*(.+)$", block, re.MULTILINE)
            if tags_match:
                raw = tags_match.group(1).strip()
                raw = raw.strip("[]")
                for part in re.split(r"[,\u060C]", raw):
                    candidate = part.strip().strip("'\"-").strip()
                    if candidate:
                        terms.append(candidate)
                # Liste biçiminde çok satırlı etiketler
                for extra in re.findall(r"^\s*-\s*(.+)$", block, re.MULTILINE):
                    candidate = extra.strip().strip("'\"").strip()
                    if candidate and candidate not in terms:
                        terms.append(candidate)

        # '## ' başlıkları
        for heading in re.findall(r"^#{2,3}\s+(.+?)\s*$", content, re.MULTILINE):
            candidate = heading.strip().strip("#").strip()
            if candidate:
                terms.append(candidate)

        # Filtre: 2-40 karakter, en fazla 4 kelime, harf içermeli, gürültü olmasın
        filtered: List[str] = []
        for term in terms:
            clean = re.sub(r"\s+", " ", term).strip()
            if not (2 <= len(clean) <= 40):
                continue
            if len(clean.split()) > 4:
                continue
            if not re.search(r"[^\W\d_]", clean, re.UNICODE):
                continue
            if clean.lower() in {"özet", "detay", "ilişkiler", "iliskiler", "tags", "notlar"}:
                continue
            if clean not in filtered:
                filtered.append(clean)
        return filtered

    # ─── İç Yardımcılar ────────────────────────────────────────────────

    def _pattern_index(self, src: str, dst: str) -> List[Tuple[str, Pattern[str]]]:
        """
        (src, dst) çifti için derlenmiş, uzunluğa göre azalan eşleşme desenleri.

        Tembel kurulur ve yeni kayıt eklendiğinde geçersiz kılınır.
        """
        pair = (src, dst)
        with self._lock:
            if not self._index_dirty:
                cached = self._phrase_index.get(pair)
                if cached is not None:
                    return cached

            # Tüm çiftleri yeniden kur (kayıt kümesi değişmiş olabilir)
            grouped: Dict[Tuple[str, str], List[str]] = {}
            for entry in self._entries.values():
                grouped.setdefault((entry.src, entry.dst), []).append(entry.term)

            self._phrase_index = {}
            for pair_key, terms in grouped.items():
                unique = sorted({term for term in terms if term}, key=lambda t: (-len(t), t.lower()))
                compiled: List[Tuple[str, Pattern[str]]] = []
                for term in unique:
                    normalized = _normalize_term(term)
                    if not normalized:
                        continue
                    compiled.append((term, word_boundary_pattern(term)))
                self._phrase_index[pair_key] = compiled
            self._index_dirty = False
            return self._phrase_index.get(pair, [])

    def __len__(self) -> int:
        """Depodaki kayıt sayısı."""
        return len(self._entries)

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        with self._lock:
            total = len(self._entries)
            domains = Counter(entry.domain or "general" for entry in self._entries.values())
            pairs = Counter(f"{entry.src}->{entry.dst}" for entry in self._entries.values())
        return {
            "name": self.name,
            "initialized": self._initialized,
            "entries": total,
            "pairs": dict(sorted(pairs.items())),
            "domains": dict(sorted(domains.items())),
            "path": self.path(),
            "loaded_from": self._loaded_from,
            "cached_pairs": len(self._phrase_index),
            **{k: v for k, v in self._stats.items() if k != "events"},
            "events": self._stats["events"],
        }
