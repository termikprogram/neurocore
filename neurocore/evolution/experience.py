"""
NeuroCore Deneyim Tamponu (L8 · Evolution)
===========================================
Her etkileşim bir **deneyim**dir: sorgu, yanıt, kullanılan bağlam, geri bildirim,
kendi kendini değerlendirme puanı, belirsizlik, gecikme ve donanım anlık görüntüsü.

Bu modül üç işi yapar:

1. `Experience` — tek bir etkileşimin değişmez kaydı (öncelikli örnekleme için
   `priority` ağırlığı üretir).
2. `ExperienceBuffer` — diske **append-only JSONL** (`experience.jsonl`) yazan,
   bellekte `deque` tutan ve eşlik eden `experience_vectors.npy` matrisini
   senkron tutan tampon.
3. `hash_embed` — torch **gerektirmeyen**, tamamen deterministik yerel gömme
   (blake2b imzalı unigram + bigram). Böylece anlamsal geri çağırma, ağır bir
   model yüklenmeden de çalışır (zarif bozulma ilkesi).

Not: Bu katman diğer NeuroCore katmanlarını **import etmez**. Yalnızca stdlib +
numpy kullanır.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("evolution.experience")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: JSONL dosya adı (append-only, satır başına bir deneyim)
EXPERIENCE_FILE = "experience.jsonl"

#: Gömme matrisi dosya adı (float32, (N, embedding_dim))
VECTOR_FILE = "experience_vectors.npy"

#: Tampon meta verisi (geriye dönük uyumluluk için tutulan indeks dosyası)
INDEX_FILE = "index.npz"

#: `data_path` hiçbir yerden çözülemezse kullanılacak göreli dizin
DEFAULT_DATA_PATH = "data/experience"

#: Alt dizin adı: kök veri yolu verildiğinde kayıtlar buraya yazılır
EXPERIENCE_DIR = "experience"

#: Gömme boyutu alt/üst sınırları
MIN_EMBED_DIM = 16
MAX_EMBED_DIM = 4096

#: Kelime parçalama deseni (Türkçe karakterler dahil)
_WORD_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü_]+")


# ─── Geri Bildirim Sinyalleri ──────────────────────────────────────────────

class FeedbackSignal(str, Enum):
    """
    Deneyime ilişkin geri bildirim türü.

    `str` tabanlı enum olduğu için JSON'a doğrudan serileşir ve
    `FeedbackSignal("explicit_positive")` biçiminde geri yüklenebilir.
    """
    EXPLICIT_POSITIVE = "explicit_positive"   # Kullanıcı açıkça onayladı
    EXPLICIT_NEGATIVE = "explicit_negative"   # Kullanıcı açıkça reddetti
    IMPLICIT_POSITIVE = "implicit_positive"   # Davranışsal onay (kabul, devam)
    IMPLICIT_NEGATIVE = "implicit_negative"   # Davranışsal ret (tekrar soru, iptal)
    NEUTRAL = "neutral"                       # Nötr
    CORRECTION = "correction"                 # Kullanıcı doğrusunu verdi
    NONE = "none"                             # Henüz geri bildirim yok


#: Sinyal → varsayılan ödül eşlemesi
SIGNAL_REWARD: Dict[FeedbackSignal, float] = {
    FeedbackSignal.EXPLICIT_POSITIVE: 1.0,
    FeedbackSignal.IMPLICIT_POSITIVE: 0.5,
    FeedbackSignal.NEUTRAL: 0.0,
    FeedbackSignal.IMPLICIT_NEGATIVE: -0.5,
    FeedbackSignal.EXPLICIT_NEGATIVE: -1.0,
    FeedbackSignal.CORRECTION: -0.3,
    FeedbackSignal.NONE: 0.0,
}


# ─── Gömme Boyutu Çözümleme ────────────────────────────────────────────────

def report_embed_dim(raw: Any, fallback: int = 768) -> int:
    """
    Herhangi bir değeri güvenli bir gömme boyutuna indirger.

    Deneyim vektörleri farklı donanım profillerinde farklı boyutlarda
    üretilebilir; bu yardımcı, bozuk/eksik değerlerde sistemi düşürmeden
    makul bir boyut döndürür.
    """
    try:
        dim = int(raw)
    except (TypeError, ValueError):
        dim = int(fallback)
    if dim < MIN_EMBED_DIM:
        dim = MIN_EMBED_DIM
    if dim > MAX_EMBED_DIM:
        dim = MAX_EMBED_DIM
    return dim


def resolve_data_path(spec: Any, default_relative: str) -> Path:
    """
    Veri yolu tanımını mutlak `Path`'e çevirir.

    `spec` mutlak bir yol olabilir, proje köküne göreli bir yol olabilir
    (`"data/experience"`) ya da hiç verilmemiş olabilir. Göreli yollar
    çalışma dizinine değil, proje köküne göre çözülür.
    """
    project_root = Path(__file__).resolve().parents[2]
    raw = str(spec or default_relative).strip() or default_relative
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return project_root / raw


def to_relative_display(path: Path) -> str:
    """Log mesajlarında kullanılacak kısa, göreli yol gösterimi."""
    try:
        root = Path(__file__).resolve().parents[2]
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path)


def _norm_text(text: Any) -> str:
    """Metni NFKC ile normalize eder, küçük harfe indirir (Türkçe uyumlu)."""
    raw = str(text or "")
    if not raw:
        return ""
    return unicodedata.normalize("NFKC", raw).lower()


def _words(text: Any) -> List[str]:
    """Metni kelimelere ayırır (Türkçe karakterler korunur)."""
    return _WORD_RE.findall(_norm_text(text))


def _as_float(value: Any, default: float = 0.0) -> float:
    """Güvenli float dönüşümü (None/NaN → default)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:  # NaN kontrolü
        return float(default)
    return out


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Değeri [low, high] aralığına sıkıştır."""
    return max(low, min(high, float(value)))


# ─── Deterministik Yerel Gömme ─────────────────────────────────────────────

def hash_embed(text: str, dim: int) -> "np.ndarray":
    """
    Deterministik, eğitim gerektirmeyen yerel metin gömmesi.

    Yöntem (torch yok, `hash()` yok):
        - Metin unigram + bigram özelliklerine ayrılır.
        - Her özellik `blake2b` ile 8 byte'a özetlenir → indeks + işaret.
        - İşaretli toplam L2-normalize edilir (kosinüs = nokta çarpımı).

    Aynı metin her zaman aynı vektörü üretir; bu yüzden farklı oturumlarda
    yazılan `experience_vectors.npy` tekrar üretilebilir.

    Args:
        text: Gömülecek metin
        dim: Vektör boyutu (16..4096 aralığına sıkıştırılır)

    Returns:
        float32 `(dim,)` L2-normalize vektör (boş metinde sıfır vektör)
    """
    size = report_embed_dim(dim)
    vec = np.zeros(size, dtype=np.float32)

    tokens = _words(text)
    if not tokens:
        # Sıfır vektör: kosinüs benzerliği 0 döner, çökme olmaz.
        return vec

    features: List[str] = list(tokens)
    features.extend(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))

    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % size
        sign = 1.0 if digest[4] & 1 else -1.0
        # Uzun özellikler biraz daha ağır basar (bigram vurgusu)
        vec[index] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


# ─── Deneyim Kaydı ─────────────────────────────────────────────────────────

@dataclass
class Experience:
    """
    Tek bir etkileşimin tam kaydı.

    Attributes:
        id: Benzersiz deneyim kimliği (`exp_<zaman>_<rastgele>`)
        timestamp: Unix zaman damgası
        query: Kullanıcı sorgusu
        answer: Üretilen yanıt
        context_used: Kullanılan bağlam referansları (not id'si / kaynak)
        feedback: Geri bildirim sinyali
        reward: Ödül (-1..1)
        quality: Öz-değerlendirme puanı (0..1)
        uncertainty: Model belirsizliği (0..1)
        latency_ms: Yanıt gecikmesi (ms)
        hardware: Donanım anlık görüntüsü (profil/cihaz/ram_free_mb/vram_free_mb)
        strategy: İzlenen yol (planlayıcı şablonu / rota)
        tags: Etiketler
        metadata: Serbest ek veri
    """

    id: str
    timestamp: float
    query: str
    answer: str
    context_used: List[str] = field(default_factory=list)
    feedback: FeedbackSignal = FeedbackSignal.NONE
    reward: float = 0.0
    quality: float = 0.0
    uncertainty: float = 0.5
    latency_ms: float = 0.0
    hardware: Dict[str, Any] = field(default_factory=dict)
    strategy: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ─── Serileştirme ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Deneyimi JSON-serileştirilebilir dict'e çevirir."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "query": self.query,
            "answer": self.answer,
            "context_used": list(self.context_used),
            "feedback": self.feedback.value if isinstance(self.feedback, FeedbackSignal)
                        else str(self.feedback),
            "reward": self.reward,
            "quality": self.quality,
            "uncertainty": self.uncertainty,
            "latency_ms": self.latency_ms,
            "hardware": dict(self.hardware),
            "strategy": self.strategy,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experience":
        """
        Dict'ten deneyim kurar. Eksik/bozuk alanlar güvenli varsayılana düşer.

        Args:
            d: `to_dict()` çıktısı ya da benzeri bir sözlük
        """
        data = d if isinstance(d, dict) else {}

        try:
            feedback = FeedbackSignal(str(data.get("feedback", FeedbackSignal.NONE.value)))
        except ValueError:
            feedback = FeedbackSignal.NONE

        context = data.get("context_used") or []
        tags = data.get("tags") or []

        return cls(
            id=str(data.get("id") or Experience.new_id()),
            timestamp=_as_float(data.get("timestamp"), time.time()),
            query=str(data.get("query") or ""),
            answer=str(data.get("answer") or ""),
            context_used=[str(c) for c in context] if isinstance(context, (list, tuple)) else [],
            feedback=feedback,
            reward=_clamp(_as_float(data.get("reward"), 0.0), -1.0, 1.0),
            quality=_clamp(_as_float(data.get("quality"), 0.0), 0.0, 1.0),
            uncertainty=_clamp(_as_float(data.get("uncertainty"), 0.5), 0.0, 1.0),
            latency_ms=max(0.0, _as_float(data.get("latency_ms"), 0.0)),
            hardware=dict(data.get("hardware") or {}),
            strategy=str(data.get("strategy") or ""),
            tags=[str(t) for t in tags] if isinstance(tags, (list, tuple)) else [],
            metadata=dict(data.get("metadata") or {}),
        )

    # ─── Yardımcılar ────────────────────────────────────────────────────

    @staticmethod
    def new_id() -> str:
        """Zamana dayalı, çakışma olasılığı düşük deneyim kimliği üretir."""
        return f"exp_{int(time.time() * 1000)}_{os.urandom(4).hex()}"

    @property
    def priority(self) -> float:
        """
        Öncelikli örnekleme ağırlığı (0..1).

        İki bağımsız sinyal birleşir:

        - `|reward|` → geri bildirimin **şiddeti**
        - `(1 - quality)` → modelin bu örnekte **ne kadar öğrenmeye aç** olduğu

        Zamandan **bağımsızdır** (recency taşımaz); böylece eski ama öğretici
        deneyimler tampondan düşene kadar yenileriyle eşit şansa sahip olur.
        Açık düzeltmeler (`CORRECTION`) öğretici oldukları için sabit küçük bir
        taban bonus alır.
        """
        signal_weight = (1.0 - self.quality) if self.quality > 0.0 else 0.5
        base = abs(float(self.reward)) * signal_weight
        if self.feedback == FeedbackSignal.CORRECTION:
            base += 0.25
        elif self.feedback in (FeedbackSignal.EXPLICIT_NEGATIVE, FeedbackSignal.EXPLICIT_POSITIVE):
            base += 0.10
        return _clamp(base, 0.0, 1.0)

    def tokens_estimate(self) -> int:
        """
        Sorgu + yanıt için kaba token tahmini (~0.75 kelime/token).

        Gerçek tokenizer bu katmanda bulunmadığı için çarpımsal bir tahmin
        yeterlidir; eğitim bütçesi hesaplarında kullanılır.
        """
        words = len(str(self.query).split()) + len(str(self.answer).split())
        return int(words / 0.75) if words else 0

    def age_seconds(self, now: Optional[float] = None) -> float:
        """Deneyimin yaşı (saniye)."""
        return max(0.0, (now if now is not None else time.time()) - self.timestamp)


# ─── Deneyim Tamponu ───────────────────────────────────────────────────────

class ExperienceBuffer:
    """
    Append-only deneyim tamponu.

    Depolama düzeni (`<data_path>/experience/`):
        experience.jsonl          → satır başına bir JSON deneyim (append-only)
        experience_vectors.npy    → float32 `(N, embedding_dim)` gömme matrisi
        index.npz                 → tampon meta verisi (boyut, sayı, zaman)

    Dayanıklılık:
        - Bozuk JSON satırları **atlanır**, sayılır (`_corrupt_lines`) ve
          uyarı loglanır; tampon asla çökmez.
        - Gömme boyutu değişirse vektör matrisi yeniden üretilir.
        - `initialize()` idempotenttir.
    """

    name = "evolution.experience"

    def __init__(self) -> None:
        # Konfigürasyon (initialize ile doldurulur)
        self._config: Any = None
        self._event_bus: Any = None

        # Yollar
        self._data_path: str = ""
        self._dir: Optional[Path] = None
        self._file: Optional[Path] = None
        self._vector_file: Optional[Path] = None
        self._index_file: Optional[Path] = None

        # Ayarlar
        self._embedding_dim: int = 768
        self._buffer_size: int = 10000

        # Bellek içi durum
        self._buffer: deque = deque(maxlen=self._buffer_size)
        self._index: Dict[str, Experience] = {}
        self._counters: Dict[str, float] = {
            "added": 0.0,
            "duplicates": 0.0,
            "corrupt_lines": 0.0,
            "feedbacks": 0.0,
        }
        self._vectors: Optional[np.ndarray] = None
        self._initialized: bool = False
        self._last_error: str = ""

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Tamponu bağla, diskten geri yükle ve gömme matrisini yeniden kur.

        İdempotenttir: ikinci çağrı no-op'tur.
        """
        if self._initialized:
            return

        self._config = config
        self._event_bus = event_bus

        self._buffer_size = self._resolve_buffer_size(config)
        self._buffer = deque(maxlen=self._buffer_size)
        self._embedding_dim = report_embed_dim(self._resolve_embedding_dim(config))
        self._data_path = self._resolve_data_path(config)

        self._dir = resolve_data_path(self._data_path, DEFAULT_DATA_PATH)
        self._file = self._dir / EXPERIENCE_FILE
        self._vector_file = self._dir / VECTOR_FILE
        self._index_file = self._dir / INDEX_FILE

        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - dosya sistemi hatası
            self._last_error = str(exc)
            logger.error(f"Deneyim dizini oluşturulamadı: {exc}")

        loaded = self._load_jsonl()
        self._rebuild_vectors(loaded)

        self._initialized = True
        logger.info(
            f"Deneyim tamponu hazır: {len(self._buffer)} kayıt · "
            f"dim={self._embedding_dim} · dizin={to_relative_display(self._dir)}"
        )
        self._emit("evolution.experience_ready", {
            "count": len(self._buffer),
            "embedding_dim": self._embedding_dim,
            "dir": str(self._dir),
        })

    def shutdown(self) -> None:
        """
        Kaynakları serbest bırak. İdempotenttir.

        JSONL zaten append-only olarak diske yazıldığı için ek yazma gerekmez;
        yalnızca vektör matrisi son hâliyle diske bırakılır.
        """
        if not self._initialized:
            return

        try:
            self._save_vectors()
        except Exception as exc:
            logger.warning(f"Gömme matrisi kapatma sırasında yazılamadı: {exc}")

        self._initialized = False
        logger.info(f"Deneyim tamponu kapatıldı ({len(self._buffer)} kayıt bellekte kaldı)")

    # ─── Konfigürasyon Çözümleme ────────────────────────────────────────

    @staticmethod
    def _resolve_buffer_size(config: Any) -> int:
        """`config.evolution.learning_buffer_size` değerini çöz (yoksa 10000)."""
        try:
            value = int(getattr(getattr(config, "evolution", None), "learning_buffer_size", 10000))
        except (TypeError, ValueError):
            value = 10000
        return max(1, value)

    @staticmethod
    def _resolve_embedding_dim(config: Any) -> int:
        """`config.neural.embedding_dim` değerini çöz (yoksa 768)."""
        return report_embed_dim(getattr(getattr(config, "neural", None), "embedding_dim", 768))

    @staticmethod
    def _resolve_data_path(config: Any) -> str:
        """
        Deneyim dizinini çöz: `config.data_path` → `<base_path>/data/experience`
        → proje kökündeki `config.yaml` → `data/experience`.

        Verilen kök yolun altına `experience/` alt dizini eklenir; böylece
        katmanın diğer bileşenleri (günlük, büyüme durumu) aynı kökü paylaşır.
        """
        explicit = getattr(config, "data_path", None) if config is not None else None
        if explicit:
            return str(Path(str(explicit)) / EXPERIENCE_DIR)

        base = getattr(config, "base_path", None) if config is not None else None
        if base:
            return str(Path(str(base)) / "data" / EXPERIENCE_DIR)

        # Konfigürasyon yok → proje kökündeki config.yaml varsa oku
        try:
            project_root = Path(__file__).resolve().parents[2]
            cfg_file = project_root / "config.yaml"
            if cfg_file.exists():
                import yaml  # ağır değil, ama yalnızca gerektiğinde yüklenir

                data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
                value = data.get("data_path")
                if value:
                    return str(Path(str(value)) / EXPERIENCE_DIR)
                if data.get("base_path"):
                    return str(Path(str(data["base_path"])) / "data" / EXPERIENCE_DIR)
        except Exception:
            pass

        return DEFAULT_DATA_PATH

    # ─── Disk G/Ç ───────────────────────────────────────────────────────

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Olay yayınla (event_bus yoksa sessizce atlanır)."""
        if self._event_bus:
            try:
                self._event_bus.emit(event_type, data, source=self.name)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Olay yayınlanamadı ({event_type}): {exc}")

    def _load_jsonl(self) -> List[Experience]:
        """
        JSONL dosyasını okuyup tamponu doldurur.

        Bozuk satırlar atlanır ve sayılır; hiçbir durumda istisna yükseltilmez.

        Returns:
            Diske yazılmış sırayla okunan deneyimler (eskiden yeniye)
        """
        loaded: List[Experience] = []
        if not self._file or not self._file.exists():
            return loaded

        corrupt = 0
        try:
            with open(self._file, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        corrupt += 1
                        logger.warning(f"Bozuk deneyim satırı atlandı (satır {line_no}): {exc}")
                        continue
                    if not isinstance(record, dict):
                        corrupt += 1
                        logger.warning(f"Bozuk deneyim satırı atlandı (satır {line_no}): nesne değil")
                        continue
                    loaded.append(Experience.from_dict(record))
        except Exception as exc:  # pragma: no cover
            self._last_error = str(exc)
            logger.error(f"Deneyim dosyası okunamadı: {exc}")

        self._counters["corrupt_lines"] = float(corrupt)

        # deque(maxlen) en yeni kayıtları tutar
        for experience in loaded:
            self._buffer.append(experience)
            self._index[experience.id] = experience

        if corrupt:
            logger.warning(f"{corrupt} bozuk deneyim satırı yok sayıldı")
        return loaded

    def _append_jsonl(self, experience: Experience) -> None:
        """Tek deneyimi JSONL dosyasına ekler (append-only)."""
        if not self._file:
            return
        line = json.dumps(experience.to_dict(), ensure_ascii=False)
        with open(self._file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _rebuild_vectors(self, experiences: Optional[Iterable[Experience]] = None) -> None:
        """
        Gömme matrisini yeniden üretir (deterministik `hash_embed` ile).

        Diskteki matris varsa ve boyutları uyuyorsa doğrudan yüklenir; aksi
        hâlde tüm deneyimler yeniden gömülür.
        """
        items = list(experiences) if experiences is not None else list(self._buffer)

        if not items:
            self._vectors = np.zeros((0, self._embedding_dim), dtype=np.float32)
            return

        # 1) Diskteki matrisi deneyelim
        if self._vector_file and self._vector_file.exists():
            try:
                cached = np.load(self._vector_file)
                if cached.ndim == 2 and cached.shape == (len(items), self._embedding_dim):
                    self._vectors = cached.astype(np.float32, copy=False)
                    return
            except Exception as exc:
                logger.warning(f"Gömme matrisi okunamadı, yeniden üretiliyor: {exc}")

        # 2) Yeniden göm
        matrix = np.zeros((len(items), self._embedding_dim), dtype=np.float32)
        for row, experience in enumerate(items):
            matrix[row] = hash_embed(f"{experience.query}\n{experience.answer}", self._embedding_dim)
        self._vectors = matrix
        self._save_vectors()

    def _save_vectors(self) -> None:
        """Gömme matrisini `.npy` olarak diske yazar (atomik değil, hızlı yol)."""
        if self._vectors is None or not self._vector_file:
            return
        try:
            self._vector_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(self._vector_file, self._vectors)
        except Exception as exc:
            logger.warning(f"Gömme matrisi yazılamadı: {exc}")

    def _save_index(self) -> None:
        """Tampon meta verisini `index.npz` olarak yazar."""
        if not self._index_file:
            return
        try:
            np.savez_compressed(
                self._index_file,
                count=np.array([len(self._buffer)], dtype=np.int64),
                embedding_dim=np.array([self._embedding_dim], dtype=np.int64),
                updated_at=np.array([time.time()], dtype=np.float64),
            )
        except Exception as exc:
            logger.debug(f"İndeks dosyası yazılamadı: {exc}")

    # ─── Yazma ─────────────────────────────────────────────────────────

    def add(self, experience: Experience) -> Experience:
        """
        Deneyimi tampona ekler: bellekte `deque`, diskte JSONL, matriste vektör.

        Args:
            experience: Eklenecek deneyim (id boşsa üretilir)

        Returns:
            Eklenen deneyim (id garanti edilmiş hâlde)
        """
        if not isinstance(experience, Experience):
            raise TypeError("Beklenen Experience nesnesi")

        if not experience.id:
            experience.id = Experience.new_id()
        if not experience.timestamp:
            experience.timestamp = time.time()

        # deque(maxlen) taşan kayıtları otomatik düşürür
        self._buffer.append(experience)
        self._index[experience.id] = experience
        self._counters["added"] += 1.0

        try:
            self._append_jsonl(experience)
        except Exception as exc:
            self._last_error = str(exc)
            logger.error(f"Deneyim diske yazılamadı: {exc}")

        self._append_vector(experience)
        self._save_index()

        self._emit("evolution.experience_added", {
            "id": experience.id,
            "tags": list(experience.tags),
            "quality": experience.quality,
            "buffer_size": len(self._buffer),
        })
        return experience

    def _append_vector(self, experience: Experience) -> None:
        """Yeni deneyimin vektörünü matrise ekler."""
        vec = hash_embed(f"{experience.query}\n{experience.answer}", self._embedding_dim)
        vec = vec.reshape(1, -1)
        if self._vectors is None or self._vectors.shape[0] == 0:
            self._vectors = vec
        elif self._vectors.shape[1] != self._embedding_dim:
            self._rebuild_vectors()
            self._vectors = np.vstack([self._vectors, vec])
        else:
            self._vectors = np.vstack([self._vectors, vec])

        # Matris tampon boyutundan büyükse (deque taşması) baştan hizala
        if self._vectors.shape[0] > len(self._buffer):
            self._rebuild_vectors()

    def log(self, query: str, answer: str, **kw: Any) -> Experience:
        """
        Kısayol: deneyim kur ve tampona ekle.

        Args:
            query: Kullanıcı sorgusu
            answer: Üretilen yanıt
            **kw: `Experience` alanları (context_used, feedback, reward, ...)

        Returns:
            Eklenen `Experience`
        """
        known = {
            "context_used", "feedback", "reward", "quality", "uncertainty",
            "latency_ms", "hardware", "strategy", "tags", "metadata", "timestamp", "id",
        }
        extra = {k: v for k, v in kw.items() if k not in known}
        fields = {k: v for k, v in kw.items() if k in known}

        metadata = dict(fields.pop("metadata", {}) or {})
        if extra:
            metadata.update(extra)

        feedback = fields.pop("feedback", FeedbackSignal.NONE)
        if isinstance(feedback, str):
            try:
                feedback = FeedbackSignal(feedback)
            except ValueError:
                feedback = FeedbackSignal.NONE

        experience = Experience(
            id=str(fields.pop("id", "") or Experience.new_id()),
            timestamp=_as_float(fields.pop("timestamp", time.time()), time.time()),
            query=str(query or ""),
            answer=str(answer or ""),
            context_used=list(fields.pop("context_used", []) or []),
            feedback=feedback,
            reward=_clamp(_as_float(fields.pop("reward", 0.0)), -1.0, 1.0),
            quality=_clamp(_as_float(fields.pop("quality", 0.0)), 0.0, 1.0),
            uncertainty=_clamp(_as_float(fields.pop("uncertainty", 0.5)), 0.0, 1.0),
            latency_ms=max(0.0, _as_float(fields.pop("latency_ms", 0.0))),
            hardware=dict(fields.pop("hardware", {}) or {}),
            strategy=str(fields.pop("strategy", "") or ""),
            tags=[str(t) for t in (fields.pop("tags", []) or [])],
            metadata=metadata,
        )
        return self.add(experience)

    def record_feedback(
        self,
        experience_id: str,
        feedback: FeedbackSignal,
        reward: Optional[float] = None,
        note: str = "",
    ) -> bool:
        """
        Var olan deneyime geri bildirim yazar.

        Deneyim kayıtları append-only olduğundan güncelleme, **yeni bir satır**
        olarak yazılan bir "yama" kaydıyla temsil edilir; bellekteki nesne
        yerinde güncellenir (aynı id korunur).

        Args:
            experience_id: Hedef deneyim kimliği
            feedback: Geri bildirim sinyali
            reward: Açık ödül (None → sinyalin varsayılanı)
            note: Serbest not

        Returns:
            Güncelleme başarılıysa True
        """
        experience = self._index.get(str(experience_id))
        if experience is None:
            logger.warning(f"Geri bildirim için deneyim bulunamadı: {experience_id}")
            return False

        if isinstance(feedback, str):
            try:
                feedback = FeedbackSignal(feedback)
            except ValueError:
                logger.warning(f"Bilinmeyen geri bildirim sinyali: {feedback}")
                return False

        experience.feedback = feedback
        experience.reward = _clamp(
            _as_float(reward, SIGNAL_REWARD.get(feedback, 0.0)), -1.0, 1.0
        )
        if note:
            experience.metadata["feedback_note"] = str(note)
            experience.metadata["feedback_note_at"] = time.time()
        self._counters["feedbacks"] += 1.0

        # Append-only yama satırı
        try:
            if self._file:
                payload = experience.to_dict()
                payload["_patch"] = True
                with open(self._file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"Geri bildirim diske yazılamadı: {exc}")

        self._emit("evolution.feedback_recorded", {
            "id": experience.id,
            "feedback": feedback.value,
            "reward": experience.reward,
        })
        return True

    # ─── Okuma ─────────────────────────────────────────────────────────

    def get(self, experience_id: str) -> Optional[Experience]:
        """Kimliğe göre deneyim döndürür (yoksa None)."""
        return self._index.get(str(experience_id))

    def recent(self, limit: int = 50) -> List[Experience]:
        """En yeni `limit` deneyimi (yeniden eskiye) döndürür."""
        if limit <= 0:
            return []
        items = list(self._buffer)
        return list(reversed(items))[:limit]

    def __len__(self) -> int:
        """Tampondaki deneyim sayısı."""
        return len(self._buffer)

    def __iter__(self):
        """Tampondaki deneyimler üzerinde (eskiden yeniye) gezinme."""
        return iter(list(self._buffer))

    def sample(self, n: int, prioritized: bool = True, seed: Optional[int] = None) -> List[Experience]:
        """
        Tampondan örnek çeker.

        Args:
            n: İstenen örnek sayısı
            prioritized: True → `Experience.priority` ağırlıklı örnekleme
            seed: Tekrarlanabilirlik için rastgelelik tohumu

        Returns:
            Örneklenen deneyimler (tampon boşsa boş liste)
        """
        items = list(self._buffer)
        if not items or n <= 0:
            return []
        if n >= len(items):
            return items

        rng = random.Random(seed)

        if not prioritized:
            return rng.sample(items, n)

        weights = [max(1e-6, float(e.priority)) for e in items]
        picked: List[Experience] = []
        pool = list(range(len(items)))
        pool_weights = list(weights)
        for _ in range(n):
            index = rng.choices(pool, weights=pool_weights, k=1)[0]
            position = pool.index(index)
            picked.append(items[index])
            pool.pop(position)
            pool_weights.pop(position)
        return picked

    def sample_batch(self, batch_size: int, embedding_dim: Optional[int] = None) -> Dict[str, Any]:
        """
        Eğitim için öncelikli örneklenmiş batch üretir.

        Args:
            batch_size: Batch büyüklüğü
            embedding_dim: İstenen gömme boyutu (yalnızca meta veri olarak raporlanır;
                vektörler `hash_embed` ile bu boyutta üretilir)

        Returns:
            `{"inputs", "targets", "rewards", "weights", ...}` — `weights`
            kayıp ağırlıklandırması için ortalaması 1.0 olacak şekilde normalize
            edilmiş, [0.25, 4.0] aralığına sıkıştırılmıştır.
        """
        dim = report_embed_dim(embedding_dim or self._embedding_dim)
        chosen = self.sample(batch_size, prioritized=True)

        priorities = [max(1e-6, float(e.priority)) for e in chosen]
        if priorities:
            mean_priority = sum(priorities) / len(priorities)
            weights = [_clamp(p / mean_priority, 0.25, 4.0) for p in priorities]
        else:
            weights = []

        return {
            "inputs": [e.query for e in chosen],
            "targets": [e.answer for e in chosen],
            "rewards": [float(e.reward) for e in chosen],
            "weights": weights,
            "ids": [e.id for e in chosen],
            "priorities": priorities,
            "qualities": [float(e.quality) for e in chosen],
            "tags": [list(e.tags) for e in chosen],
            "embedding_dim": dim,
            "count": len(chosen),
        }

    def high_value(self, threshold: float = 0.3, limit: int = 200) -> List[Experience]:
        """Ödülü eşiğin üzerinde olan yüksek değerli deneyimler (yeniden eskiye)."""
        items = [e for e in reversed(self._buffer) if float(e.reward) >= float(threshold)]
        return items[:limit]

    def low_quality(self, threshold: float = 0.45, limit: int = 200) -> List[Experience]:
        """
        Kalitesi eşiğin altında kalan deneyimler (en kötüden iyiye).

        Yalnızca **değerlendirilmiş** kayıtlar dikkate alınır (quality > 0).
        """
        items = [
            e for e in self._buffer
            if 0.0 < float(e.quality) < float(threshold)
        ]
        items.sort(key=lambda e: (float(e.quality), -e.timestamp))
        return items[:limit]

    def corrections(self, limit: int = 200) -> List[Experience]:
        """Kullanıcı düzeltmeleri (en yeniden eskiye)."""
        items = [e for e in reversed(self._buffer) if e.feedback == FeedbackSignal.CORRECTION]
        return items[:limit]

    def by_tag(self, tag: str, limit: int = 200) -> List[Experience]:
        """Etikete göre deneyimler (en yeniden eskiye, büyük/küçük harf duyarsız)."""
        needle = str(tag or "").strip().lower()
        if not needle:
            return []
        items = [
            e for e in reversed(self._buffer)
            if any(str(t).strip().lower() == needle for t in e.tags)
        ]
        return items[:limit]

    def similar(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.15,
    ) -> List[Tuple[Experience, float]]:
        """
        Anlamsal geri çağırma: sorguya en benzeyen deneyimler (torch yok).

        Kosinüs benzerliği, L2-normalize edilmiş `hash_embed` matrisi üzerinde
        nokta çarpımına indirgenir.

        Args:
            query: Aranan metin
            k: Döndürülecek en fazla deneyim sayısı
            min_score: Alt benzerlik eşiği

        Returns:
            `(Experience, skor)` çiftleri, skora göre azalan sırada
        """
        if self._vectors is None or self._vectors.shape[0] == 0 or not self._buffer:
            return []

        items = list(self._buffer)
        matrix = self._vectors
        # deque taşması sonrası matris ile tampon hizasını garanti et
        if matrix.shape[0] != len(items):
            self._rebuild_vectors(items)
            matrix = self._vectors if self._vectors is not None else np.zeros((0, self._embedding_dim), np.float32)
            if matrix.shape[0] != len(items):
                return []

        vector = hash_embed(query, self._embedding_dim)
        scores = matrix @ vector

        order = np.argsort(-scores)
        results: List[Tuple[Experience, float]] = []
        for row in order[: max(k * 3, k)]:
            score = float(scores[int(row)])
            if score < float(min_score):
                break
            results.append((items[int(row)], score))
            if len(results) >= k:
                break
        return results

    # ─── İstatistik ve Bakım ───────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """
        Tampon istatistikleri.

        Returns:
            count, ortalama ödül/kalite, geri bildirim ve etiket histogramları,
            en eski/en yeni zaman damgası, diskteki bayt sayısı.
        """
        items = list(self._buffer)
        count = len(items)

        rewards = [float(e.reward) for e in items]
        qualities = [float(e.quality) for e in items if float(e.quality) > 0.0]
        feedback_hist = Counter(e.feedback.value for e in items)
        tag_hist: Counter = Counter()
        for experience in items:
            for tag in experience.tags:
                tag_hist[str(tag)] += 1

        timestamps = [float(e.timestamp) for e in items if e.timestamp]
        disk_bytes = 0
        try:
            if self._file and self._file.exists():
                disk_bytes += self._file.stat().st_size
            if self._vector_file and self._vector_file.exists():
                disk_bytes += self._vector_file.stat().st_size
        except Exception:
            pass

        return {
            "count": count,
            "capacity": self._buffer_size,
            "usage": round(count / self._buffer_size, 4) if self._buffer_size else 0.0,
            "mean_reward": round(sum(rewards) / count, 4) if count else 0.0,
            "mean_quality": round(sum(qualities) / len(qualities), 4) if qualities else 0.0,
            "evaluated": len(qualities),
            "feedback_histogram": dict(feedback_hist),
            "tag_histogram": dict(tag_hist.most_common(20)),
            "oldest": min(timestamps) if timestamps else 0.0,
            "newest": max(timestamps) if timestamps else 0.0,
            "bytes_on_disk": disk_bytes,
            "embedding_dim": self._embedding_dim,
            "vector_rows": int(self._vectors.shape[0]) if self._vectors is not None else 0,
            "corrupt_lines": int(self._counters["corrupt_lines"]),
            "dir": str(self._dir) if self._dir else "",
            "initialized": self._initialized,
        }

    def compact(self, keep: Optional[int] = None) -> int:
        """
        JSONL dosyasını sıkıştırır: en yeni `keep` kaydı tutar, `(query, answer)`
        çiftine göre yinelenenleri düşürür.

        Args:
            keep: Tutulacak kayıt sayısı (None → `learning_buffer_size`)

        Returns:
            Kaldırılan kayıt sayısı
        """
        limit = int(keep) if keep is not None else self._buffer_size
        limit = max(1, limit)

        items = list(self._buffer)
        original = len(items)

        # 1) Yinelenenleri düşür (en yeni kazanır), 2) en yeni `limit` kaydı tut
        seen: set = set()
        kept_reversed: List[Experience] = []
        for experience in reversed(items):
            digest = hashlib.blake2b(
                f"{experience.query}\u0000{experience.answer}".encode("utf-8"),
                digest_size=16,
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            kept_reversed.append(experience)
            if len(kept_reversed) >= limit:
                break

        kept = list(reversed(kept_reversed))
        removed = original - len(kept)

        if removed <= 0 and len(items) <= limit:
            logger.debug("Sıkıştırma gerekmedi")
            return 0

        # Dosyayı atomik olarak yeniden yaz
        if self._file:
            tmp = self._file.with_suffix(".jsonl.tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as handle:
                    for experience in kept:
                        handle.write(json.dumps(experience.to_dict(), ensure_ascii=False) + "\n")
                os.replace(tmp, self._file)
            except Exception as exc:
                self._last_error = str(exc)
                logger.error(f"Sıkıştırma başarısız: {exc}")
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
                return 0

        self._buffer = deque(kept, maxlen=self._buffer_size)
        self._index = {e.id: e for e in kept}
        self._rebuild_vectors(kept)
        self._save_index()

        logger.info(f"Deneyim tamponu sıkıştırıldı: {removed} kayıt kaldırıldı, {len(kept)} kaldı")
        self._emit("evolution.buffer_compacted", {"removed": removed, "kept": len(kept)})
        return removed

    def clear(self) -> int:
        """
        Tamponu ve diskteki dosyaları temizler.

        Returns:
            Silinen kayıt sayısı
        """
        removed = len(self._buffer)
        self._buffer.clear()
        self._index.clear()
        self._vectors = np.zeros((0, self._embedding_dim), dtype=np.float32)

        for path in (self._file, self._vector_file, self._index_file):
            try:
                if path and path.exists():
                    path.unlink()
            except Exception as exc:
                logger.warning(f"Dosya silinemedi ({path}): {exc}")

        logger.info(f"Deneyim tamponu temizlendi ({removed} kayıt)")
        self._emit("evolution.buffer_cleared", {"removed": removed})
        return removed

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        stats = self.stats()
        stats.update({
            "name": self.name,
            "added_total": int(self._counters["added"]),
            "feedbacks_total": int(self._counters["feedbacks"]),
            "last_error": self._last_error,
        })
        return stats
