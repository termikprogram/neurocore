"""
NeuroCore Müfredat Oluşturucu (L8 · Evolution)
================================================
Deneyim tamponundan **artan zorlukta** eğitim seti üretir (curriculum learning).

Karışım stratejisi (`build`):

    %35 düzeltmeler      → en yüksek ağırlık (kullanıcı doğrusunu söyledi)
    %25 yüksek değerli   → başarılı örnekler (ağırlık 1.0)
    %25 düşük kaliteli   → etiketlerine hedeflenmiş başarısızlıklar
    %15 keşif/yeni       → embedding matrisinde en "benzersiz" örnekler

Keşif dilimi sıralaması `config.evolution.expansion_strategy` ile belirlenir:
    depth_first      → en sık etiket kümesini sömür
    breadth_first    → etiketler arasında sırayla gez
    curiosity_driven → embedding yeniliği en yüksek olanları seç

Aynı prompt (normalize edilmiş hash) iki kez sete girmez. Bu modül yalnızca
stdlib + numpy kullanır ve `ExperienceBuffer`'a yalnızca **duck typing** ile
bağlanır (tip ipucu `TYPE_CHECKING` altında).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from neurocore.utils.logger import NeuroCoreLogger

if TYPE_CHECKING:  # pragma: no cover - yalnızca tip denetimi
    from neurocore.evolution.experience import ExperienceBuffer

logger = NeuroCoreLogger.get_logger("evolution.curriculum")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: `build()` karışım oranları
MIX_CORRECTIONS = 0.35
MIX_HIGH_VALUE = 0.25
MIX_LOW_QUALITY = 0.25
MIX_EXPLORATORY = 0.15

#: Düzeltme örneklerinin taban ağırlığı
WEIGHT_CORRECTION = 2.0
WEIGHT_HIGH_VALUE = 1.0
WEIGHT_LOW_QUALITY = 1.5
WEIGHT_EXPLORATORY = 0.8

#: Soru biçimi eşlemeleri (anlam KORUNUR, yalnızca biçim değişir)
_QUESTION_FORMS: Tuple[Tuple[str, str], ...] = (
    ("neden", "niçin"),
    ("niçin", "neden"),
    ("nasıl", "ne şekilde"),
    ("ne şekilde", "nasıl"),
    ("ne zaman", "hangi vakitte"),
    ("nerede", "hangi konumda"),
    ("hangi", "ne tür"),
    ("what", "which"),
    ("which", "what"),
    ("why", "for what reason"),
    ("how", "in what way"),
)

#: Önek iskeleleri (prompt'a eklenir, anlam değişmez)
_PREFIX_SCAFFOLDS: Tuple[str, ...] = (
    "Adım adım açıkla: ",
    "Kısaca: ",
    "Maddeler halinde: ",
    "Örnek vererek açıkla: ",
)

#: Sonek iskeleleri
_SUFFIX_SCAFFOLDS: Tuple[str, ...] = (
    " Kısaca özetle.",
    " Adım adım açıkla.",
    " Maddeler halinde yaz.",
)

_WORD_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü_]+")
_WS_RE = re.compile(r"\s+")
_COMPLEXITY_MARKERS: Tuple[str, ...] = (
    "neden", "niçin", "nasıl", "karşılaştır", "fark", "adım", "hesapla",
    "optimiz", "mimari", "trade-off", "analiz", "why", "how", "compare",
)


# ─── Yardımcılar ───────────────────────────────────────────────────────────

def _normalize(text: Any) -> str:
    """Prompt karşılaştırması için normalize edilmiş metin."""
    raw = unicodedata.normalize("NFKC", str(text or "")).lower()
    return _WS_RE.sub(" ", raw).strip()


def _hash_of(text: Any) -> str:
    """Normalize edilmiş metnin kısa blake2b özeti."""
    return hashlib.blake2b(_normalize(text).encode("utf-8"), digest_size=12).hexdigest()


def _words(text: Any) -> List[str]:
    """Kelime listesi."""
    return _WORD_RE.findall(str(text or ""))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Değeri [low, high] aralığına sıkıştır."""
    return max(low, min(high, float(value)))


# ─── Müfredat Öğesi ────────────────────────────────────────────────────────

@dataclass
class CurriculumItem:
    """
    Tek bir eğitim öğesi.

    Attributes:
        prompt: Modelin göreceği girdi
        target: Hedeflenen çıktı ("" → "bundan kaçın" anlamına gelir)
        weight: Kayıp ağırlığı
        difficulty: 0 (kolay) .. 1 (zor)
        source: Öğenin kaynağı (correction | high_value | low_quality | exploratory | augment)
        tags: İlgili etiketler
    """

    prompt: str
    target: str
    weight: float = 1.0
    difficulty: float = 0.5
    source: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dict."""
        return {
            "prompt": self.prompt,
            "target": self.target,
            "weight": round(float(self.weight), 4),
            "difficulty": round(float(self.difficulty), 4),
            "source": self.source,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CurriculumItem":
        """Dict'ten öğe kurar."""
        data = d if isinstance(d, dict) else {}
        return cls(
            prompt=str(data.get("prompt") or ""),
            target=str(data.get("target") or ""),
            weight=float(data.get("weight", 1.0) or 1.0),
            difficulty=float(data.get("difficulty", 0.5) or 0.5),
            source=str(data.get("source") or ""),
            tags=[str(t) for t in (data.get("tags") or [])],
        )

    @property
    def key(self) -> str:
        """Tekilleştirme anahtarı (normalize prompt hash'i)."""
        return _hash_of(self.prompt)


# ─── Müfredat Oluşturucu ───────────────────────────────────────────────────

class CurriculumBuilder:
    """
    Deneyimlerden artan zorlukta eğitim seti üretir.

    Kullanım:
        builder = CurriculumBuilder()
        builder.initialize(config, event_bus)
        items = builder.build(buffer, max_items=500)
        pairs = builder.to_training_pairs(builder.order(items))
    """

    name = "evolution.curriculum"

    def __init__(self) -> None:
        self._config: Any = None
        self._event_bus: Any = None
        self._strategy: str = "curiosity_driven"
        self._counters: Dict[str, float] = {
            "builds": 0.0,
            "items": 0.0,
            "duplicates": 0.0,
            "augmentations": 0.0,
        }
        self._last_sources: Dict[str, int] = {}
        self._initialized: bool = False

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Konfigürasyonu bağla. İdempotenttir."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._strategy = str(
            getattr(getattr(config, "evolution", None), "expansion_strategy", "curiosity_driven")
            or "curiosity_driven"
        ).strip().lower()
        if self._strategy not in ("depth_first", "breadth_first", "curiosity_driven"):
            logger.warning(f"Bilinmeyen genişleme stratejisi '{self._strategy}', varsayılana düşülüyor")
            self._strategy = "curiosity_driven"
        self._initialized = True
        logger.info(f"Müfredat oluşturucu hazır (keşif stratejisi: {self._strategy})")

    def shutdown(self) -> None:
        """Kaynakları serbest bırak. İdempotenttir."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info("Müfredat oluşturucu kapatıldı")

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Olay yayınla (bus yoksa atlanır)."""
        if self._event_bus:
            try:
                self._event_bus.emit(event_type, data, source=self.name)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Olay yayınlanamadı ({event_type}): {exc}")

    # ─── Zorluk Ölçümü ─────────────────────────────────────────────────

    def difficulty_of(self, query: str, answer: str, quality: float) -> float:
        """
        Bir öğenin zorluğunu tahmin eder (0..1).

        Bileşenler: düşük kalite (öğrenilecek çok), uzun yanıt, çok kelimeli
        soru ve karmaşıklık işaretleri (`neden`, `nasıl`, `karşılaştır`, ...).
        """
        try:
            qual = _clamp(float(quality))
        except (TypeError, ValueError):
            qual = 0.5

        quality_gap = 1.0 - qual

        answer_words = len(_words(answer))
        query_words = len(_words(query))
        length_factor = min(1.0, (answer_words / 120.0) + (query_words / 40.0) * 0.5)

        lowered = str(query or "").lower()
        marker_hits = sum(1 for marker in _COMPLEXITY_MARKERS if marker in lowered)
        marker_factor = min(1.0, marker_hits * 0.4)

        if "?" in str(query or "") and re.search(r"\b(ve|and)\b", lowered):
            marker_factor = min(1.0, marker_factor + 0.2)

        difficulty = 0.55 * quality_gap + 0.30 * length_factor + 0.15 * marker_factor
        return round(_clamp(difficulty), 4)

    # ─── Ana Kurulum ───────────────────────────────────────────────────

    def build(
        self,
        buffer: "ExperienceBuffer",
        max_items: int = 500,
        focus: Optional[List[str]] = None,
    ) -> List[CurriculumItem]:
        """
        Karışım stratejisine göre müfredat kurar.

        Args:
            buffer: `ExperienceBuffer` benzeri nesne
            max_items: Üretilecek en fazla öğe sayısı
            focus: Yalnızca bu etiketlere odaklan (None → tümü)

        Returns:
            Tekilleştirilmiş `CurriculumItem` listesi
        """
        self._counters["builds"] += 1.0
        limit = max(1, int(max_items))
        focus_tags = {str(f).strip().lower() for f in (focus or []) if str(f).strip()}

        quota_corrections = max(1, int(limit * MIX_CORRECTIONS))
        quota_high = max(1, int(limit * MIX_HIGH_VALUE))
        quota_low = max(1, int(limit * MIX_LOW_QUALITY))
        quota_explore = max(1, limit - quota_corrections - quota_high - quota_low)

        items: List[CurriculumItem] = []
        seen: set = set()
        sources: Dict[str, int] = {}

        def _append(item: Optional[CurriculumItem], source: str) -> bool:
            """Tekilleştirerek listeye ekler."""
            if item is None or not item.prompt.strip():
                return False
            key = item.key
            if key in seen:
                self._counters["duplicates"] += 1.0
                return False
            if focus_tags and not (focus_tags & {t.lower() for t in item.tags}):
                return False
            seen.add(key)
            item.source = item.source or source
            items.append(item)
            sources[source] = sources.get(source, 0) + 1
            return True

        try:
            # 1) %35 — düzeltmeler (en yüksek ağırlık)
            for item in self.from_corrections(buffer, max_items=quota_corrections * 2):
                if not _append(item, "correction"):
                    continue
                if sum(sources.get(k, 0) for k in ("correction",)) >= quota_corrections:
                    break

            # 2) %25 — yüksek değerli başarılar
            for item in self.from_high_value(buffer, max_items=quota_high * 2):
                if not _append(item, "high_value"):
                    continue
                if sources.get("high_value", 0) >= quota_high:
                    break

            # 3) %25 — düşük kaliteli başarısızlıklar (etiketlerine hedefli)
            for item in self._from_low_quality(buffer, max_items=quota_low * 2):
                if not _append(item, "low_quality"):
                    continue
                if sources.get("low_quality", 0) >= quota_low:
                    break

            # 4) %15 — keşif / yeni (daha önce seçilen prompt'lar hariç)
            for item in self._exploratory(buffer, max_items=quota_explore * 2, exclude=seen):
                if not _append(item, "exploratory"):
                    continue
                if sources.get("exploratory", 0) >= quota_explore:
                    break

        except Exception as exc:
            logger.warning(f"Müfredat kurulumunda hata: {exc}")

        ordered = self.order(items)[:limit]
        self._counters["items"] += float(len(ordered))
        self._last_sources = sources

        self._emit("evolution.curriculum_built", {
            "count": len(ordered),
            "sources": sources,
            "strategy": self._strategy,
        })
        logger.info(
            f"Müfredat kuruldu: {len(ordered)} öğe "
            f"({', '.join(f'{k}={v}' for k, v in sorted(sources.items())) or 'boş'})"
        )
        return ordered

    # ─── Kaynak 1: Düzeltmeler ─────────────────────────────────────────

    def from_corrections(self, buffer: "ExperienceBuffer", max_items: int = 200) -> List[CurriculumItem]:
        """
        Kullanıcı düzeltmelerinden öğeler üretir.

        Düzeltmede kullanıcı doğru cevabı işaret ettiği için hedef, deneyimin
        yanıtıdır; ağırlık en yüksektir.
        """
        items: List[CurriculumItem] = []
        try:
            records = buffer.corrections(limit=max_items)
        except Exception as exc:
            logger.warning(f"Düzeltmeler okunamadı: {exc}")
            return items

        for experience in records:
            note = ""
            try:
                note = str((experience.metadata or {}).get("feedback_note") or "")
            except Exception:
                note = ""
            target = note.strip() or str(experience.answer or "")
            items.append(CurriculumItem(
                prompt=str(experience.query or ""),
                target=target,
                weight=WEIGHT_CORRECTION,
                difficulty=self.difficulty_of(experience.query, target, experience.quality),
                source="correction",
                tags=list(experience.tags),
            ))
        return items

    # ─── Kaynak 2: Yüksek Değerli ──────────────────────────────────────

    def from_high_value(self, buffer: "ExperienceBuffer", max_items: int = 200) -> List[CurriculumItem]:
        """Yüksek ödüllü başarılı deneyimlerden öğeler üretir (ağırlık 1.0)."""
        items: List[CurriculumItem] = []
        try:
            records = buffer.high_value(threshold=0.3, limit=max_items)
        except Exception as exc:
            logger.warning(f"Yüksek değerli deneyimler okunamadı: {exc}")
            return items

        for experience in records:
            items.append(CurriculumItem(
                prompt=str(experience.query or ""),
                target=str(experience.answer or ""),
                weight=WEIGHT_HIGH_VALUE,
                difficulty=self.difficulty_of(
                    experience.query, experience.answer, max(0.7, experience.quality)
                ),
                source="high_value",
                tags=list(experience.tags),
            ))
        return items

    # ─── Kaynak 3: Düşük Kaliteli ──────────────────────────────────────

    def _from_low_quality(
        self,
        buffer: "ExperienceBuffer",
        max_items: int = 200,
    ) -> List[CurriculumItem]:
        """
        Düşük kaliteli yanıtlardan **düzeltici hedefli** öğeler üretir.

        Başarısızlıklar etiketlerine göre hedeflenir: hedef metin, o etiketlerde
        istenen davranışı tarif eder ("" hedefi "öğren ki bundan kaçınasın"
        anlamına gelir, ancak kullanılabilir bir düzeltici hedef de üretilir).
        """
        items: List[CurriculumItem] = []
        try:
            records = buffer.low_quality(threshold=0.45, limit=max_items)
        except Exception as exc:
            logger.warning(f"Düşük kaliteli deneyimler okunamadı: {exc}")
            return items

        for experience in records:
            tags = [str(t) for t in (experience.tags or [])]
            tag_text = ", ".join(tags) if tags else "genel"
            if experience.feedback.value == "correction" and str(experience.answer or "").strip():
                target = str(experience.answer)
            else:
                target = (
                    f"Bu soruya {tag_text} kapsamında somut, ölçülebilir ve doğrulanabilir "
                    "bilgiyle yanıt ver; genel ifadelerden kaçın ve belirsizliği açıkça belirt."
                )
            items.append(CurriculumItem(
                prompt=str(experience.query or ""),
                target=target,
                weight=WEIGHT_LOW_QUALITY,
                difficulty=self.difficulty_of(experience.query, experience.answer, experience.quality),
                source="low_quality",
                tags=tags,
            ))
        return items

    # ─── Kaynak 4: Keşif ───────────────────────────────────────────────

    def _exploratory(
        self,
        buffer: "ExperienceBuffer",
        max_items: int = 200,
        exclude: Optional[set] = None,
    ) -> List[CurriculumItem]:
        """
        Keşif dilimi: en "yeni" örnekleri stratejiye göre sıralar.

        - `curiosity_driven`: embedding matrisinde diğerlerine en az benzeyenler
        - `depth_first`     : en sık etiket kümesini önce sömür
        - `breadth_first`   : etiketler arasında sırayla gez (round-robin)

        Args:
            buffer: `ExperienceBuffer` benzeri nesne
            max_items: Üretilecek en fazla öğe
            exclude: Daha önce başka dilimlerde kullanılmış prompt anahtarları
        """
        try:
            records = list(buffer.recent(limit=10000))
        except Exception as exc:
            logger.warning(f"Keşif için deneyimler okunamadı: {exc}")
            return []

        if not records:
            return []

        excluded = set(exclude or set())
        if excluded:
            filtered = [e for e in records if _hash_of(e.query) not in excluded]
            if filtered:
                records = filtered

        strategy = self._strategy
        if strategy == "depth_first":
            ordered = self._order_depth_first(records)
        elif strategy == "breadth_first":
            ordered = self._order_breadth_first(records)
        else:
            ordered = self._order_curiosity_driven(records, buffer)

        items: List[CurriculumItem] = []
        for experience in ordered[:max_items]:
            items.append(CurriculumItem(
                prompt=str(experience.query or ""),
                target=str(experience.answer or ""),
                weight=WEIGHT_EXPLORATORY,
                difficulty=self.difficulty_of(experience.query, experience.answer, experience.quality),
                source="exploratory",
                tags=list(experience.tags),
            ))
        return items

    @staticmethod
    def _tag_key(experience: Any) -> str:
        """Deneyimin birincil etiketi (yoksa 'genel')."""
        tags = [str(t).strip().lower() for t in (getattr(experience, "tags", None) or []) if str(t).strip()]
        return tags[0] if tags else "genel"

    def _order_depth_first(self, records: Sequence[Any]) -> List[Any]:
        """En sık etiket kümesini önce tüketen sıralama."""
        counts: Dict[str, int] = {}
        for experience in records:
            counts[self._tag_key(experience)] = counts.get(self._tag_key(experience), 0) + 1
        if not counts:
            return list(records)
        dominant = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        primary = [e for e in records if self._tag_key(e) == dominant]
        rest = [e for e in records if self._tag_key(e) != dominant]
        return primary + rest

    def _order_breadth_first(self, records: Sequence[Any]) -> List[Any]:
        """Etiketler arasında sırayla gezen (round-robin) sıralama."""
        buckets: Dict[str, List[Any]] = {}
        order: List[str] = []
        for experience in records:
            key = self._tag_key(experience)
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(experience)

        ordered: List[Any] = []
        index = 0
        while True:
            added = False
            for key in order:
                bucket = buckets[key]
                if index < len(bucket):
                    ordered.append(bucket[index])
                    added = True
            if not added:
                break
            index += 1
        return ordered

    def _order_curiosity_driven(self, records: Sequence[Any], buffer: Any) -> List[Any]:
        """Embedding yeniliğine göre sıralama (matris yoksa yeniden-eskiye düşer)."""
        matrix = getattr(buffer, "_vectors", None)
        items = list(records)
        if matrix is None or getattr(matrix, "ndim", 0) != 2 or matrix.shape[0] != len(items):
            return items

        try:
            normalized = np.asarray(matrix, dtype=np.float32)
            # Nokta çarpım matrisi (vektörler L2-normalize olduğu için kosinüs)
            similarity = normalized @ normalized.T
            np.fill_diagonal(similarity, -1.0)
            mean_similarity = similarity.mean(axis=1)
            order = np.argsort(mean_similarity)  # en düşük ortalama benzerlik = en yeni
            return [items[int(i)] for i in order]
        except Exception as exc:
            logger.debug(f"Yenilik sıralaması yapılamadı: {exc}")
            return items

    # ─── Sıralama ──────────────────────────────────────────────────────

    def order(self, items: List[CurriculumItem]) -> List[CurriculumItem]:
        """
        Artan zorlukta sıralar (curriculum learning); eşitlikte ağırlık azalan.

        Args:
            items: Sıralanacak öğeler

        Returns:
            Yeni sıralanmış liste (girdi değiştirilmez)
        """
        return sorted(
            list(items or []),
            key=lambda item: (round(float(item.difficulty), 4), -float(item.weight), item.prompt),
        )

    # ─── Artırma (Augmentation) ────────────────────────────────────────

    def augment(self, item: CurriculumItem, n: int = 2, seed: Optional[int] = None) -> List[CurriculumItem]:
        """
        Deterministik parafrazlar üretir — **anlam asla değişmez**.

        Dönüşümler:
            1. Soru biçimi değişimi (`neden→niçin`, `nasıl→ne şekilde`, `what→which`)
            2. Önek iskelesi ("Adım adım açıkla: ...", "Kısaca: ...")
            3. Sonek iskelesi ("... Kısaca özetle.")
            4. Yanıt-hedef yeniden sıralama (aynı bilgi, farklı sıra)

        Args:
            item: Kaynak öğe
            n: Üretilecek en fazla varyant sayısı
            seed: Tekrarlanabilirlik için tohum

        Returns:
            Varyant listesi (kaynak öğe dahil edilmez)
        """
        count = max(0, int(n))
        if count == 0 or not isinstance(item, CurriculumItem):
            return []

        rng = random.Random(seed)
        variants: List[CurriculumItem] = []

        # 1) Soru biçimi değişimi
        swapped = self._swap_question_form(item.prompt)
        if swapped and _normalize(swapped) != _normalize(item.prompt):
            variants.append(self._variant(item, prompt=swapped, tag="form"))

        # 2) Önek iskeleleri
        for prefix in _PREFIX_SCAFFOLDS:
            if len(variants) >= count:
                break
            prompt = f"{prefix}{item.prompt}"
            if _normalize(prompt) == _normalize(item.prompt):
                continue
            variants.append(self._variant(item, prompt=prompt, tag="prefix"))

        # 3) Sonek iskeleleri
        for suffix in _SUFFIX_SCAFFOLDS:
            if len(variants) >= count:
                break
            prompt = f"{item.prompt}{suffix}"
            variants.append(self._variant(item, prompt=prompt, tag="suffix"))

        # 4) Yanıt-hedef yeniden sıralama
        if len(variants) < count:
            reordered = self._reorder_target(item.target)
            if reordered and _normalize(reordered) != _normalize(item.target):
                variants.append(self._variant(item, target=reordered, tag="order"))

        rng.shuffle(variants)
        chosen = variants[:count]
        self._counters["augmentations"] += float(len(chosen))
        return chosen

    @staticmethod
    def _variant(
        item: CurriculumItem,
        prompt: Optional[str] = None,
        target: Optional[str] = None,
        tag: str = "",
    ) -> CurriculumItem:
        """Kaynak öğeden türetilmiş varyant üretir."""
        tags = list(item.tags)
        if tag and tag not in tags:
            tags.append(f"aug:{tag}")
        return CurriculumItem(
            prompt=prompt if prompt is not None else item.prompt,
            target=target if target is not None else item.target,
            weight=float(item.weight) * 0.9,  # türev örnek biraz daha hafif
            difficulty=float(item.difficulty),
            source="augment" if not item.source else f"{item.source}:augment",
            tags=tags,
        )

    @staticmethod
    def _swap_question_form(prompt: str) -> str:
        """Soru kalıbını eşdeğeriyle değiştirir (anlam korunur)."""
        text = str(prompt or "")
        lowered = text.lower()
        for source, target in _QUESTION_FORMS:
            if source in lowered:
                index = lowered.find(source)
                replaced = text[:index] + target + text[index + len(source):]
                # Yalnızca ilk eşleşme değiştirilir (deterministik)
                return replaced
        return text

    @staticmethod
    def _reorder_target(target: str) -> str:
        """Hedef metnin cümle sırasını döndürür (aynı bilgi, farklı sıra)."""
        text = str(target or "").strip()
        if not text:
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
        if len(sentences) < 2:
            return text
        return " ".join(sentences[1:] + sentences[:1])

    # ─── Çıktı Üretimi ─────────────────────────────────────────────────

    def to_training_pairs(self, items: List[CurriculumItem]) -> Dict[str, Any]:
        """
        Öğeleri paralel listelere açar.

        Returns:
            `{"prompts", "targets", "weights", "difficulties"}` + yardımcı alanlar
        """
        sequence = list(items or [])
        return {
            "prompts": [item.prompt for item in sequence],
            "targets": [item.target for item in sequence],
            "weights": [round(float(item.weight), 4) for item in sequence],
            "difficulties": [round(float(item.difficulty), 4) for item in sequence],
            "sources": [item.source for item in sequence],
            "tags": [list(item.tags) for item in sequence],
            "count": len(sequence),
        }

    def export(self, items: List[CurriculumItem], path: str) -> str:
        """
        Öğeleri JSONL olarak **atomik** yazar.

        Args:
            items: Dışa aktarılacak öğeler
            path: Hedef dosya yolu

        Returns:
            Yazılan dosyanın yolu
        """
        target = Path(str(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")

        with open(tmp, "w", encoding="utf-8") as handle:
            for item in list(items or []):
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp, target)
        logger.info(f"Müfredat dışa aktarıldı: {len(list(items or []))} öğe → {target}")
        self._emit("evolution.curriculum_exported", {
            "count": len(list(items or [])),
            "path": str(target),
        })
        return str(target)

    # ─── İstatistikler ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "strategy": self._strategy,
            "builds": int(self._counters["builds"]),
            "items_total": int(self._counters["items"]),
            "duplicates_dropped": int(self._counters["duplicates"]),
            "augmentations": int(self._counters["augmentations"]),
            "last_sources": dict(self._last_sources),
            "mix": {
                "corrections": MIX_CORRECTIONS,
                "high_value": MIX_HIGH_VALUE,
                "low_quality": MIX_LOW_QUALITY,
                "exploratory": MIX_EXPLORATORY,
            },
        }
