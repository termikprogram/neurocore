"""
NeuroCore Bellek Bankası (Memory Bank)
=======================================
Modelin bağlam dışından bilgi geri çağırmasını sağlayan ilişkisel bellek.

Ne işe yarar?
    Ağırlıklar yalnızca dil yapısını taşır; **olgu bilgisi kasada** (Obsidian
    vault) yaşar. Bellek bankası bu iki dünyayı birleştirir: geçmiş
    etkileşimlerden çıkarılan anahtar-değer çiftleri saklanır ve yeni sorgu
    geldiğinde benzer anahtarlar geri çağrılır.

Tasarım:
    * **Anahtar** — omurganın gizli durumundan (veya hash gömmeden) türetilen
      vektör; **değer** — ham bağlam metni + meta veri.
    * **Geri çağırma** — kosinüs benzerliği + tazelik + kullanım sıklığı
      ağırlıklı puanlama. Tek bir "en iyi" eşleşme değil, **çeşitlendirilmiş**
      bir küme döner (MMR).
    * **Yazma politikası** — her şeyi ezberlemek zararlıdır: yalnızca yeterince
      yeni (surprise) veya yeterince önemli (utility) bilgi yazılır.
    * **Kapasite** — sabit yuva sayısı; dolduğunda fayda/fiyat oranı en düşük
      kayıt atılır (LRU değil, **fayda tabanlı** tahliye).
    * **Kalıcılık** — NumPy `.npz` + JSON yan dosyası; atomik yazma.

Bu modül torch'a **bağlı değildir**; numpy veya saf Python ile çalışır, böylece
donanım profili ne olursa olsun bellekte tutarlılık sağlanır.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.memory")

try:  # pragma: no cover
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False


# ─── Yardımcılar ────────────────────────────────────────────────────────────

def _normalize(vector: Sequence[float]) -> List[float]:
    """Vektörü L2-normalize eder."""
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 1e-12:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    """İç çarpım (normalize vektörlerde kosinüs)."""
    length = min(len(a), len(b))
    return sum(float(a[index]) * float(b[index]) for index in range(length))


# ─── Kayıt ──────────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """
    Bellekteki tek bir kayıt.

    Attributes:
        key: Arama vektörü (normalize)
        value: Saklanan metin/bağlam
        importance: 0..1 — yazma anındaki fayda tahmini
        source: Kaynak etiketi (`vault:kn_x`, `os:system_report`, `user`)
        tags: Arama filtresi için etiketler
        meta: Ek veri (JSON-serileştirilebilir olmalı)
        created_at / last_used: Zaman damgaları
        uses: Kaç kez geri çağrıldı
        success: Geri çağrıldığı yanıtların ortalama kalitesi (0..1)
    """
    key: List[float]
    value: str
    importance: float = 0.5
    source: str = "unknown"
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    uses: int = 0
    success: float = 0.5

    @property
    def age_days(self) -> float:
        """Kaydın yaşı (gün)."""
        return max(0.0, (time.time() - self.created_at) / 86400.0)

    @property
    def idle_days(self) -> float:
        """Son kullanımdan bu yana geçen süre (gün)."""
        return max(0.0, (time.time() - self.last_used) / 86400.0)

    def utility(self, now: Optional[float] = None) -> float:
        """
        Kaydın fayda puanı — tahliye kararının temeli.

        Fayda = önem × başarı × kullanım kanıtı × tazelik.
        Kullanım kanıtı logaritmiktir: 100 kez kullanılmak 1 kez
        kullanılmaktan 10 kat değerli değildir.
        """
        current = now or time.time()
        recency = 0.5 ** (max(0.0, (current - self.last_used) / 86400.0) / 30.0)
        usage_proof = 1.0 + math.log1p(max(0, self.uses))
        return float(self.importance * (0.3 + 0.7 * self.success) * usage_proof * recency)

    def to_dict(self, include_key: bool = False) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        data = {
            "value": self.value,
            "importance": round(float(self.importance), 4),
            "source": self.source,
            "tags": list(self.tags),
            "meta": dict(self.meta),
            "created_at": self.created_at,
            "last_used": self.last_used,
            "uses": int(self.uses),
            "success": round(float(self.success), 4),
        }
        if include_key:
            data["key"] = list(self.key)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        """Sözlükten kayıt üretir."""
        return cls(
            key=[float(value) for value in (data.get("key") or [])],
            value=str(data.get("value", "")),
            importance=float(data.get("importance", 0.5) or 0.5),
            source=str(data.get("source", "unknown")),
            tags=[str(tag) for tag in (data.get("tags") or [])],
            meta=dict(data.get("meta") or {}),
            created_at=float(data.get("created_at", time.time()) or time.time()),
            last_used=float(data.get("last_used", time.time()) or time.time()),
            uses=int(data.get("uses", 0) or 0),
            success=float(data.get("success", 0.5) or 0.5),
        )


@dataclass
class RecallResult:
    """Geri çağırma sonucu."""
    record: MemoryRecord
    score: float
    similarity: float
    recency: float
    utility: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {
            "value": self.record.value,
            "source": self.record.source,
            "tags": list(self.record.tags),
            "score": round(self.score, 5),
            "similarity": round(self.similarity, 5),
            "recency": round(self.recency, 5),
            "utility": round(self.utility, 5),
            "uses": self.record.uses,
            "meta": dict(self.record.meta),
        }


@dataclass
class WriteDecision:
    """Yazma kararının gerekçesi (şeffaflık için)."""
    write: bool
    reason: str
    novelty: float = 0.0
    importance: float = 0.0
    evicted: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        return {"write": self.write, "reason": self.reason,
                "novelty": round(self.novelty, 5),
                "importance": round(self.importance, 5),
                "evicted": self.evicted}


# ─── Bellek Bankası ─────────────────────────────────────────────────────────

class MemoryBank:
    """
    İlişkisel bellek bankası (retrieval-augmented memory).

    Kullanım:
        bank = MemoryBank()
        bank.initialize(config, event_bus)
        bank.attach_embedder(embed.embed)

        bank.remember("GPU sıcaklığı tozdan yükselir.", source="vault:kn_1",
                      tags=["gpu"], importance=0.8)
        hits = bank.recall("GPU neden ısınır?", top_k=3)
        for hit in hits:
            print(hit.score, hit.record.value)

        bank.feedback(record_key, quality=0.9)     # kullanım kalitesi
        bank.save()                                 # kalıcı hâle getir
    """

    name = "neural.memory"

    def __init__(self) -> None:
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._lock = threading.RLock()

        self._records: "OrderedDict[str, MemoryRecord]" = OrderedDict()
        self._embed_fn: Optional[Callable[[str], Sequence[float]]] = None
        self._embed_dim = 768

        # Kapasite ve politika
        self.capacity = 512
        self.novelty_threshold = 0.72      # bu benzerliğin üstü "zaten biliyorum"
        self.importance_threshold = 0.35   # bunun altındaki önemsiz bilgi yazılmaz
        self.eviction_margin = 0.9         # yeni kayıt, en zayıf kaydın bu katından değerli olmalı
        self.tag_normalize = True

        # Kalıcılık
        self._store_dir: Optional[Path] = None
        self._dirty = False

        self._counter = {
            "writes": 0, "rejected": 0, "evictions": 0, "recalls": 0,
            "hits": 0, "misses": 0, "feedback": 0, "loads": 0, "saves": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None,
                   store_dir: Optional[str] = None) -> None:
        """
        Bellek bankasını başlatır ve diske kaydedilmiş kayıtları yükler.

        Args:
            config: SystemConfig
            event_bus: EventBus
            store_dir: Kalıcılık dizini (None ise config'ten türetilir)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus

        neural = getattr(config, "neural", None)
        self._embed_dim = int(getattr(neural, "embedding_dim", 768) or 768)
        self.capacity = max(64, int(getattr(neural, "memory_bank_size", 512) or 512))

        if store_dir:
            self._store_dir = Path(store_dir)
        else:
            base = getattr(config, "data_path", None)
            if not base:
                base_path = getattr(config, "base_path", None) or os.getcwd()
                base = str(Path(base_path) / "data")
            self._store_dir = Path(base) / "memory"

        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"Bellek dizini oluşturulamadı ({self._store_dir}): {exc}")
            self._store_dir = None

        self._initialized = True
        loaded = self.load()
        logger.info(
            f"Bellek bankası hazır — kapasite {self.capacity}, "
            f"{len(self._records)} kayıt yüklendi"
        )

    def shutdown(self) -> None:
        """Kayıtları diske yazıp kapatır."""
        if not self._initialized:
            return
        try:
            if self._dirty:
                self.save()
        except Exception as exc:
            logger.warning(f"Bellek kaydedilemedi: {exc}")
        self._initialized = False
        logger.info(
            f"Bellek bankası kapatıldı ({len(self._records)} kayıt, "
            f"{self._counter['hits']}/{self._counter['recalls']} isabet)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Bellek istatistikleri."""
        recalls = self._counter["recalls"]
        sources: Dict[str, int] = {}
        for record in self._records.values():
            sources[record.source.split(":")[0]] = sources.get(record.source.split(":")[0], 0) + 1
        return {
            **self._counter,
            "size": len(self._records),
            "capacity": self.capacity,
            "utilization": round(len(self._records) / max(1, self.capacity), 4),
            "hit_rate": round(self._counter["hits"] / recalls, 4) if recalls else 0.0,
            "embed_dim": self._embed_dim,
            "embedder_attached": self._embed_fn is not None,
            "mean_importance": round(
                sum(record.importance for record in self._records.values())
                / max(1, len(self._records)), 4
            ),
            "sources": sources,
            "store_dir": str(self._store_dir or ""),
        }

    @property
    def size(self) -> int:
        """Kayıt sayısı."""
        return len(self._records)

    @property
    def is_full(self) -> bool:
        """Kapasite doldu mu?"""
        return len(self._records) >= self.capacity

    def attach_embedder(self, embed_fn: Callable[[str], Sequence[float]]) -> None:
        """
        Gömme fonksiyonunu bağlar (nöral katman tarafından enjekte edilir).

        Bağlanmazsa anahtar olarak basit bir hash vektörü kullanılır —
        banka yine de çalışır, yalnızca geri çağırma kalitesi düşer.
        """
        self._embed_fn = embed_fn
        logger.info("Bellek bankasına gömme fonksiyonu bağlandı")

    # ─── Anahtar Üretimi ────────────────────────────────────────────────

    def _make_key(self, text: str) -> List[float]:
        """Metinden arama anahtarı üretir (normalize vektör)."""
        if self._embed_fn is not None:
            try:
                vector = self._embed_fn(text)
                if vector:
                    return _normalize(list(vector))
            except Exception as exc:
                logger.debug(f"Gömme başarısız, hash'e düşülüyor: {exc}")
        from neurocore.neural.embedding import hash_embed
        return hash_embed(text, dim=self._embed_dim, salt="neurocore-memory")

    # ─── Yazma ──────────────────────────────────────────────────────────

    def should_write(self, key: Sequence[float], importance: float,
                     source: str = "") -> WriteDecision:
        """
        Yazma kararını verir (yenilik + önem).

        Kural:
            * Banka boşsa → yaz (her zaman bir çekirdek gerekir).
            * En benzer kayıt çok benzerse (`novelty_threshold`) → yazma,
              onun yerine o kaydı güncelle (tekrar ezberleme).
            * Önem eşiğin altındaysa → yazma.
            * Banka doluysa ve yeni kayıt en zayıf kayıttan değersizse → yazma.

        Returns:
            WriteDecision (gerekçesiyle)
        """
        if not self._records:
            return WriteDecision(True, "banka boş — çekirdek kayıt", novelty=1.0,
                                 importance=float(importance))

        best_similarity = 0.0
        best_id = ""
        for record_id, record in self._records.items():
            similarity = _dot(key, record.key)
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = record_id

        novelty = max(0.0, 1.0 - best_similarity)

        if best_similarity >= self.novelty_threshold:
            return WriteDecision(False, f"zaten biliyorum (benzerlik {best_similarity:.2f})",
                                 novelty=novelty, importance=float(importance))

        if importance < self.importance_threshold:
            return WriteDecision(False, f"önem eşiğin altında ({importance:.2f})",
                                 novelty=novelty, importance=float(importance))

        if self.is_full:
            weakest_id, weakest = self._weakest()
            new_utility = float(importance) * (1.0 + novelty)
            if weakest is not None and new_utility < weakest.utility() * self.eviction_margin:
                return WriteDecision(
                    False,
                    f"bellek dolu ve yeni kayıt en zayıftan değersiz "
                    f"({new_utility:.2f} < {weakest.utility():.2f})",
                    novelty=novelty, importance=float(importance),
                )
            return WriteDecision(True, "bellek dolu — en zayıf kayıt tahliye edilecek",
                                 novelty=novelty, importance=float(importance),
                                 evicted=weakest_id)

        return WriteDecision(True, f"yeni bilgi (yenilik {novelty:.2f})",
                             novelty=novelty, importance=float(importance))

    def remember(
        self,
        text: str,
        *,
        source: str = "unknown",
        tags: Optional[Iterable[str]] = None,
        importance: float = 0.5,
        meta: Optional[Dict[str, Any]] = None,
        force: bool = False,
        key_text: Optional[str] = None,
    ) -> Optional[str]:
        """
        Bankaya yeni bilgi yazar (yazma politikası uygulanır).

        Args:
            text: Saklanacak bağlam metni
            source: Kaynak etiketi (`vault:kn_x`, `os:snapshot`, `user`)
            tags: Etiketler
            importance: 0..1 önem
            meta: Ek veri
            force: Yazma politikasını atla
            key_text: Anahtar için farklı bir metin (None = `text`)

        Returns:
            Kayıt kimliği veya yazılmadıysa None
        """
        if not text or not text.strip():
            return None

        key = self._make_key(key_text or text)
        decision = WriteDecision(True, "zorlandı", importance=float(importance))
        if not force:
            decision = self.should_write(key, importance, source)
            if not decision.write:
                self._counter["rejected"] += 1
                logger.debug(f"Bellek yazımı reddedildi: {decision.reason}")
                self._emit("neural.memory_rejected", decision.to_dict())
                return None

        # Benzer kayıt varsa güncelle (değeri uzat, kullanımı pekiştir)
        existing_id = self._find_near_duplicate(key)
        if existing_id and not force:
            record = self._records[existing_id]
            if text.strip() not in record.value:
                record.value = f"{record.value}\n\n{text.strip()}"
            record.importance = max(record.importance, float(importance))
            record.last_used = time.time()
            if tags:
                for tag in self._normalize_tags(tags):
                    if tag not in record.tags:
                        record.tags.append(tag)
            self._records.move_to_end(existing_id)
            self._dirty = True
            self._emit("neural.memory_updated", {"id": existing_id, "source": source})
            return existing_id

        if decision.evicted:
            self._evict(decision.evicted)

        record_id = self._new_id()
        record = MemoryRecord(
            key=key,
            value=text.strip(),
            importance=float(min(1.0, max(0.0, importance))),
            source=source,
            tags=self._normalize_tags(tags or []),
            meta=dict(meta or {}),
        )
        with self._lock:
            self._records[record_id] = record
            while len(self._records) > self.capacity:
                self._evict(self._weakest()[0])

        self._counter["writes"] += 1
        self._dirty = True
        self._emit("neural.memory_written", {
            "id": record_id, "source": source, "importance": record.importance,
            "size": len(self._records), "reason": decision.reason,
        })
        return record_id

    def forget(self, record_id: str) -> bool:
        """Tek kaydı siler."""
        with self._lock:
            if record_id in self._records:
                self._records.pop(record_id)
                self._dirty = True
                return True
        return False

    def clear(self, source_prefix: Optional[str] = None) -> int:
        """
        Kayıtları temizler.

        Args:
            source_prefix: Yalnızca bu önekle başlayan kaynakları sil
                (None = hepsi)

        Returns:
            Silinen kayıt sayısı
        """
        with self._lock:
            if source_prefix is None:
                count = len(self._records)
                self._records.clear()
            else:
                to_remove = [
                    record_id for record_id, record in self._records.items()
                    if record.source.startswith(source_prefix)
                ]
                for record_id in to_remove:
                    self._records.pop(record_id, None)
                count = len(to_remove)
        if count:
            self._dirty = True
        return count

    def _new_id(self) -> str:
        """Benzersiz kayıt kimliği."""
        import uuid
        return f"mem_{uuid.uuid4().hex[:10]}"

    def _normalize_tags(self, tags: Iterable[str]) -> List[str]:
        """Etiketleri normalize eder."""
        if not self.tag_normalize:
            return [str(tag) for tag in tags]
        try:
            from neurocore.knowledge.note import normalize_tags
            return normalize_tags(tags)
        except Exception:
            return [str(tag).strip().lower() for tag in tags if str(tag).strip()]

    def _find_near_duplicate(self, key: Sequence[float]) -> Optional[str]:
        """Neredeyse aynı anahtara sahip kaydı bulur."""
        for record_id, record in self._records.items():
            if _dot(key, record.key) >= (self.novelty_threshold + 0.05):
                return record_id
        return None

    def _weakest(self) -> Tuple[str, Optional[MemoryRecord]]:
        """Fayda puanı en düşük kaydı bulur."""
        if not self._records:
            return "", None
        now = time.time()
        weakest_id = min(self._records, key=lambda rid: self._records[rid].utility(now))
        return weakest_id, self._records[weakest_id]

    def _evict(self, record_id: str) -> bool:
        """Kaydı tahliye eder."""
        with self._lock:
            record = self._records.pop(record_id, None)
        if record is None:
            return False
        self._counter["evictions"] += 1
        self._dirty = True
        self._emit("neural.memory_evicted", {
            "id": record_id, "source": record.source,
            "uses": record.uses, "importance": record.importance,
        })
        logger.debug(f"Bellek kaydı tahliye edildi: {record_id} ({record.source})")
        return True

    # ─── Geri Çağırma ───────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: int = 5,
        *,
        min_score: float = 0.25,
        tags: Optional[Iterable[str]] = None,
        source_prefix: Optional[str] = None,
        exclude_source_prefixes: Optional[Iterable[str]] = None,
        diversify: bool = True,
        mark_used: bool = True,
        similarity_weight: float = 0.7,
        recency_weight: float = 0.15,
        utility_weight: float = 0.15,
    ) -> List[RecallResult]:
        """
        Sorguya en uygun kayıtları geri çağırır.

        Puan = benzerlik × w1 + tazelik × w2 + fayda × w3

        Args:
            query: Sorgu metni
            top_k: Döndürülecek kayıt sayısı
            min_score: Minimum puan eşiği
            tags: Yalnızca bu etiketlerden birini taşıyan kayıtlar
            source_prefix: Yalnızca bu kaynak önekiyle başlayanlar
            exclude_source_prefixes: Bu öneklerle başlayan kaynakları atla
                (ör. `("query:",)` → sistemin kendi yanıtlarını dışla ve
                öz besleme döngüsünü önle)
            diversify: MMR ile çeşitlilik uygula
            mark_used: Kullanım sayacını artır
            similarity_weight / recency_weight / utility_weight: Puan ağırlıkları

        Returns:
            RecallResult listesi (puana göre azalan)
        """
        self._counter["recalls"] += 1
        if not self._records or not query.strip():
            self._counter["misses"] += 1
            return []

        key = self._make_key(query)
        now = time.time()
        wanted_tags = set(self._normalize_tags(tags)) if tags else set()
        excluded_prefixes = tuple(str(p) for p in (exclude_source_prefixes or ()))

        scored: List[RecallResult] = []
        for record_id, record in self._records.items():
            if wanted_tags and not wanted_tags.intersection(record.tags):
                continue
            if source_prefix and not record.source.startswith(source_prefix):
                continue
            if excluded_prefixes and record.source.startswith(excluded_prefixes):
                continue
            if not record.key or len(record.key) != len(key):
                continue

            similarity = _dot(key, record.key)
            if similarity <= 0.0:
                continue
            recency = 0.5 ** (max(0.0, (now - record.last_used) / 86400.0) / 30.0)
            utility = min(2.0, record.utility(now)) / 2.0

            score = (
                similarity_weight * similarity
                + recency_weight * recency
                + utility_weight * utility
            )
            if score >= min_score:
                scored.append(RecallResult(record=record, score=score,
                                           similarity=similarity, recency=recency,
                                           utility=utility))

        scored.sort(key=lambda item: -item.score)
        if diversify and len(scored) > 2:
            scored = self._mmr(scored, top_k)
        selected = scored[:top_k]

        if mark_used:
            for result in selected:
                result.record.uses += 1
                result.record.last_used = now
            if selected:
                self._dirty = True

        if selected:
            self._counter["hits"] += 1
        else:
            self._counter["misses"] += 1

        self._emit("neural.memory_recalled", {
            "query": query[:120], "hits": len(selected),
            "top_score": round(selected[0].score, 4) if selected else 0.0,
        })
        return selected

    @staticmethod
    def _mmr(results: List[RecallResult], top_k: int,
             lambda_param: float = 0.75) -> List[RecallResult]:
        """
        MMR ile çeşitlendirme — birbirinin kopyası kayıtları eleme.

        Kayıt değerlerinin kelime örtüşmesi yedek benzerlik ölçüsüdür.
        """
        if len(results) <= 2:
            return results
        selected = [results[0]]
        remaining = results[1:]

        def text_similarity(a: RecallResult, b: RecallResult) -> float:
            words_a = set(a.record.value.lower().split()[:200])
            words_b = set(b.record.value.lower().split()[:200])
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / len(words_a | words_b)

        while remaining and len(selected) < max(top_k, 2):
            best_index = 0
            best_value = -1e9
            for index, candidate in enumerate(remaining):
                redundancy = max(text_similarity(candidate, chosen) for chosen in selected)
                value = lambda_param * candidate.score - (1 - lambda_param) * redundancy
                if value > best_value:
                    best_value = value
                    best_index = index
            selected.append(remaining.pop(best_index))
        return selected

    def recall_text(self, query: str, top_k: int = 5,
                    separator: str = "\n---\n", **kwargs: Any) -> str:
        """
        Geri çağrılan kayıtları tek bir bağlam metnine birleştirir.

        Nöral katmana doğrudan beslenebilecek biçimde, her kayıt kaynağı ve
        puanıyla etiketlenir.
        """
        results = self.recall(query, top_k=top_k, **kwargs)
        if not results:
            return ""
        blocks = [
            f"[bellek {index + 1} · kaynak: {result.record.source} · "
            f"puan: {result.score:.2f}]\n{result.record.value}"
            for index, result in enumerate(results)
        ]
        return separator.join(blocks)

    def associate(self, record_id: str, quality: float) -> bool:
        """
        Bir kaydın işe yararlığını geri bildirimle günceller.

        Args:
            record_id: Kayıt kimliği
            quality: 0..1 — kaydın yanıta katkısı

        Returns:
            Başarılı ise True
        """
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                return False
            # Üstel hareketli ortalama: geçmiş performans yavaş güncellenir
            alpha = 0.3
            record.success = float(min(1.0, max(0.0,
                (1 - alpha) * record.success + alpha * quality)))
            if quality > 0.7:
                record.importance = min(1.0, record.importance + 0.05)
            elif quality < 0.3:
                record.importance = max(0.05, record.importance - 0.1)
        self._counter["feedback"] += 1
        self._dirty = True
        return True

    def feedback_by_source(self, source: str, quality: float) -> int:
        """Bir kaynağa ait tüm kayıtların kalitesini günceller."""
        updated = 0
        for record_id, record in list(self._records.items()):
            if record.source == source:
                if self.associate(record_id, quality):
                    updated += 1
        return updated

    def consolidate(self, min_similarity: float = 0.86,
                    max_merges: int = 50) -> Dict[str, Any]:
        """
        Belleği sıkıştırır: çok benzeyen kayıtları birleştirir.

        Uzun oturumlarda biriken yığını azaltır ve geri çağırma kalitesini
        artırır (aynı bilginin beş kopyası ilk 5 sonucu doldurur).

        Args:
            min_similarity: Birleştirme eşiği
            max_merges: Tek turda en fazla kaç birleştirme

        Returns:
            {"merged": int, "before": int, "after": int}
        """
        before = len(self._records)
        merged = 0
        ids = list(self._records.keys())

        for index in range(len(ids)):
            if merged >= max_merges:
                break
            first_id = ids[index]
            first = self._records.get(first_id)
            if first is None:
                continue
            for other_index in range(index + 1, len(ids)):
                other_id = ids[other_index]
                other = self._records.get(other_id)
                if other is None:
                    continue
                if _dot(first.key, other.key) < min_similarity:
                    continue
                # Değerleri birleştir (kısa olan uzun olana eklenir)
                if other.value.strip() and other.value.strip() not in first.value:
                    first.value = f"{first.value}\n\n{other.value.strip()}"
                first.importance = max(first.importance, other.importance)
                first.uses += other.uses
                first.success = (first.success + other.success) / 2.0
                for tag in other.tags:
                    if tag not in first.tags:
                        first.tags.append(tag)
                self._records.pop(other_id, None)
                merged += 1
                if merged >= max_merges:
                    break

        if merged:
            self._dirty = True
            self._emit("neural.memory_consolidated", {
                "merged": merged, "before": before, "after": len(self._records),
            })
            logger.info(
                f"Bellek sıkıştırıldı: {merged} kayıt birleştirildi "
                f"({before} → {len(self._records)})"
            )
        return {"merged": merged, "before": before, "after": len(self._records)}

    def prune(self, max_age_days: float = 90.0, min_utility: float = 0.05) -> int:
        """
        Yaşlı ve faydasız kayıtları temizler.

        Args:
            max_age_days: Bu yaştan eski kayıtlar aday
            min_utility: Bu faydanın altındaki kayıtlar silinir

        Returns:
            Silinen kayıt sayısı
        """
        now = time.time()
        doomed: List[str] = []
        for record_id, record in self._records.items():
            if record.idle_days >= max_age_days and record.utility(now) < min_utility:
                doomed.append(record_id)
        for record_id in doomed:
            self._records.pop(record_id, None)
        if doomed:
            self._dirty = True
            logger.info(f"Bellekten {len(doomed)} yaşlı kayıt temizlendi")
        return len(doomed)

    # ─── Kalıcılık ──────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """
        Belleği diske kaydeder.

        Vektörler `.npz` (veya numpy yoksa JSON), meta veriler JSON olarak
        yazılır. Yazma atomiktir (geçici dosya + `os.replace`).

        Returns:
            Yazılan meta dosyasının yolu
        """
        if self._store_dir is None and path is None:
            logger.debug("Bellek dizini yok — kayıt atlandı")
            return ""
        target_dir = Path(path).parent if path else self._store_dir
        assert target_dir is not None
        target_dir.mkdir(parents=True, exist_ok=True)
        meta_path = Path(path) if path else (target_dir / "memory.json")
        vector_path = meta_path.with_suffix(".npz")

        records = list(self._records.items())
        meta_payload = {
            "version": 1,
            "embed_dim": self._embed_dim,
            "capacity": self.capacity,
            "count": len(records),
            "records": [
                {"id": record_id, **record.to_dict(include_key=False)}
                for record_id, record in records
            ],
        }

        try:
            if HAS_NUMPY and records:
                matrix = np.asarray([record.key for _, record in records], dtype=np.float32)
                temp_vector = vector_path.with_name(vector_path.name + ".tmp")
                np.savez_compressed(temp_vector, keys=matrix)
                os.replace(str(temp_vector) + ".npz" if not str(temp_vector).endswith(".npz")
                           else temp_vector, vector_path)
            else:
                meta_payload["keys"] = [record.key for _, record in records]

            temp_meta = meta_path.with_name(meta_path.name + ".tmp")
            temp_meta.write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            os.replace(temp_meta, meta_path)
        except (OSError, ValueError) as exc:
            logger.error(f"Bellek kaydedilemedi: {exc}")
            return ""

        self._dirty = False
        self._counter["saves"] += 1
        logger.debug(f"Bellek kaydedildi: {meta_path} ({len(records)} kayıt)")
        return str(meta_path)

    def load(self, path: Optional[str] = None) -> int:
        """
        Belleği diskten yükler.

        Bozuk dosyalar sessizce atlanır (uyarı loglanır) — sistem çalışmaya
        devam eder.

        Returns:
            Yüklenen kayıt sayısı
        """
        if self._store_dir is None and path is None:
            return 0
        meta_path = Path(path) if path else (self._store_dir / "memory.json")  # type: ignore[union-attr]
        if not meta_path.exists():
            return 0

        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Bellek dosyası okunamadı ({meta_path}): {exc}")
            return 0

        raw_records = payload.get("records") or []
        keys: List[List[float]] = []

        if HAS_NUMPY:
            vector_path = meta_path.with_suffix(".npz")
            if vector_path.exists():
                try:
                    with np.load(vector_path) as data:
                        keys = data["keys"].astype(np.float32).tolist()
                except (OSError, ValueError, KeyError) as exc:
                    logger.warning(f"Bellek vektörleri okunamadı ({vector_path}): {exc}")
                    keys = []
        if not keys:
            keys = [[float(value) for value in key] for key in (payload.get("keys") or [])]

        self._embed_dim = int(payload.get("embed_dim", self._embed_dim) or self._embed_dim)
        loaded = 0
        with self._lock:
            for index, raw in enumerate(raw_records):
                if index < len(keys):
                    raw = {**raw, "key": keys[index]}
                record = MemoryRecord.from_dict(raw)
                # Anahtar yoksa metinden yeniden üret (geriye dönük uyum)
                if not record.key:
                    record.key = self._make_key(record.value)
                record_id = str(raw.get("id") or self._new_id())
                self._records[record_id] = record
                loaded += 1
            # Kapasite aşılmışsa en zayıfları at
            while len(self._records) > self.capacity:
                self._evict(self._weakest()[0])

        self._counter["loads"] += 1
        if loaded:
            logger.info(f"Bellek yüklendi: {meta_path} ({loaded} kayıt)")
        return loaded

    def export(self, path: str, include_vectors: bool = False) -> str:
        """
        Belleği okunabilir JSON olarak dışa aktarır (inceleme/paylaşım için).

        Args:
            path: Hedef dosya
            include_vectors: Anahtar vektörleri de yazılsın mı?

        Returns:
            Yazılan dosya yolu
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "exported_at": time.time(),
            "count": len(self._records),
            "stats": self.get_stats(),
            "records": [
                {"id": record_id, "utility": round(record.utility(), 4),
                 **record.to_dict(include_key=include_vectors)}
                for record_id, record in self._records.items()
            ],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(target)

    # ─── Raporlama ──────────────────────────────────────────────────────

    def top_records(self, limit: int = 10, by: str = "utility") -> List[Dict[str, Any]]:
        """
        En değerli kayıtları listeler.

        Args:
            limit: Kaç kayıt
            by: `utility` | `uses` | `importance` | `recent`
        """
        keys = {
            "utility": lambda item: -item[1].utility(),
            "uses": lambda item: -item[1].uses,
            "importance": lambda item: -item[1].importance,
            "recent": lambda item: -item[1].last_used,
        }
        sorter = keys.get(by, keys["utility"])
        ordered = sorted(self._records.items(), key=sorter)[:limit]
        return [
            {
                "id": record_id,
                "source": record.source,
                "tags": list(record.tags),
                "importance": round(record.importance, 4),
                "uses": record.uses,
                "utility": round(record.utility(), 4),
                "age_days": round(record.age_days, 2),
                "preview": record.value[:160].replace("\n", " "),
            }
            for record_id, record in ordered
        ]

    def has_source(self, source: str) -> bool:
        """Bu kaynak etiketiyle kayıt var mı?"""
        return any(record.source == source for record in self._records.values())

    def sources(self) -> Dict[str, int]:
        """Kaynak etiketi → kayıt sayısı."""
        counts: Dict[str, int] = {}
        for record in self._records.values():
            counts[record.source] = counts.get(record.source, 0) + 1
        return counts

    def health(self) -> Dict[str, Any]:
        """Bellek bankası sağlık raporu."""
        issues: List[str] = []
        if not self._initialized:
            issues.append("banka başlatılmadı")
        if self._embed_fn is None:
            issues.append("gömme fonksiyonu bağlı değil — hash anahtarları kullanılıyor")
        if self.size == 0:
            issues.append("banka boş")
        utilization = self.size / max(1, self.capacity)
        if utilization > 0.95:
            issues.append("bellek dolmak üzere — konsolidasyon önerilir")
        return {
            "ok": self._initialized and self.size > 0,
            "size": self.size,
            "capacity": self.capacity,
            "utilization": round(utilization, 4),
            "issues": issues,
            "stats": self.get_stats(),
        }

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<MemoryBank size={self.size}/{self.capacity} dim={self._embed_dim}>"


__all__ = [
    "MemoryBank",
    "MemoryRecord",
    "RecallResult",
    "WriteDecision",
]
