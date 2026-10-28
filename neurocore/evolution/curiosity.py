"""
NeuroCore Merak Motoru (L8 · Evolution)
========================================
Bilgi kasasındaki **boşlukları, çelişkileri ve bakımsız alanları** bulur ve
bunları önceliklendirilmiş araştırma görevlerine çevirir.

Merak türleri ve taban ağırlıkları:

    contradiction    0.95  → çelişkili bilgi en acil müdahale
    gap              0.80  → kendi notu olmayan ama bahsedilen konu
    unexplored_cluster 0.70 → bağlantılı ama hiç sentezlenmemiş küme
    shallow          0.55  → çok kısa, doyurucu olmayan not
    orphan           0.40  → hiçbir bağlantısı olmayan not
    stale            0.35  → eski ve güveni düşük konu

Öncelik = tür ağırlığı + kanıt katkısı + konu uzunluğu/önemi − tekrar cezası.
Union-find, in-degree sayımı ve önceliklendirme tamamen bu dosyada uygulanır;
dış bağımlılık yoktur.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("evolution.curiosity")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: Tür bazlı taban ağırlıklar
KIND_WEIGHTS: Dict[str, float] = {
    "contradiction": 0.95,
    "gap": 0.80,
    "unexplored_cluster": 0.70,
    "shallow": 0.55,
    "orphan": 0.40,
    "stale": 0.35,
}

#: Tür bazlı beklenen not sayısı (araştırma planı için)
EXPECTED_NOTES: Dict[str, int] = {
    "contradiction": 2,
    "gap": 2,
    "unexplored_cluster": 3,
    "shallow": 2,
    "orphan": 1,
    "stale": 1,
}

#: Belirsizlik işaretleri
UNCERTAIN_MARKERS: Tuple[str, ...] = ("todo", "belirsiz", "emin değil", "??", "fixme", "eksik")

#: Düşük güven eşiği
LOW_CONFIDENCE = 0.4

#: İçerik terimi asgari uzunluğu
_MIN_TERM_LEN = 3

_WORD_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü_]+")
_WS_RE = re.compile(r"\s+")
_SENT_RE = re.compile(r"(?<=[.!?…])\s+")
_HEADING_STRIP_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")


# ─── Yardımcılar ───────────────────────────────────────────────────────────

def _norm(text: Any) -> str:
    """Küçük harfli, tek boşluklu metin."""
    return _WS_RE.sub(" ", str(text or "").lower()).strip()


def _words(text: Any) -> List[str]:
    """Kelime listesi."""
    return _WORD_RE.findall(str(text or ""))


def _terms(text: Any, min_len: int = _MIN_TERM_LEN) -> Set[str]:
    """İçerik terimi kümesi."""
    return {w.lower() for w in _words(text) if len(w) >= min_len}


def _clean_body(note: Any) -> str:
    """Not gövdesini başlıklardan arındırır."""
    if not isinstance(note, dict):
        return ""
    return _HEADING_STRIP_RE.sub("", str(note.get("body") or "")).strip()


def _note_id(note: Any, index: int = 0) -> str:
    """Not kimliği."""
    if isinstance(note, dict) and note.get("id"):
        return str(note["id"])
    return f"note_{index}"


def _tag_set(note: Any) -> Set[str]:
    """Notun etiket kümesi (normalize)."""
    if not isinstance(note, dict):
        return set()
    tags = note.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    return {_norm(t).replace(" ", "-") for t in tags if str(t).strip()}


def _confidence(note: Any) -> float:
    """Notun güven değeri (0..1)."""
    try:
        value = float((note or {}).get("confidence", 0.5))
    except (TypeError, ValueError, AttributeError):
        return 0.5
    return max(0.0, min(1.0, value))


def _updated_at(note: Any) -> float:
    """Notun güncellenme zamanı (yoksa 0)."""
    if not isinstance(note, dict):
        return 0.0
    for key in ("updated", "updated_at", "modified", "created"):
        value = note.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Değeri [low, high] aralığına sıkıştır."""
    return max(low, min(high, float(value)))


# ─── Union-Find ────────────────────────────────────────────────────────────

class _UnionFind:
    """Bağlantılı bileşenleri bulmak için ağırlıklı union-find."""

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def add(self, item: str) -> None:
        """Öğeyi kümeye ekler (idempotent)."""
        if item not in self._parent:
            self._parent[item] = item
            self._rank[item] = 0

    def find(self, item: str) -> str:
        """Yol sıkıştırmalı kök bulma."""
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # yol sıkıştırma
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        """İki kümeyi birleştirir (rank bazlı)."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1

    def components(self) -> Dict[str, List[str]]:
        """Kök → üyeler sözlüğü."""
        groups: Dict[str, List[str]] = {}
        for item in list(self._parent):
            groups.setdefault(self.find(item), []).append(item)
        return groups


# ─── Merak Öğesi ───────────────────────────────────────────────────────────

@dataclass
class CuriosityItem:
    """
    Keşfedilmeyi bekleyen tek bir merak konusu.

    Attributes:
        topic: Konu başlığı
        kind: `gap | contradiction | stale | orphan | shallow | unexplored_cluster`
        reason: Neden merak edilmeli (Türkçe, insan okunur)
        priority: 0..1 öncelik
        evidence: Kanıt referansları (not id'leri, alıntılar)
        suggested_query: Önerilen araştırma sorgusu
        tags: İlgili etiketler
    """

    topic: str
    kind: str
    reason: str
    priority: float
    evidence: List[str] = field(default_factory=list)
    suggested_query: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dict."""
        return {
            "topic": self.topic,
            "kind": self.kind,
            "reason": self.reason,
            "priority": round(float(self.priority), 4),
            "evidence": list(self.evidence),
            "suggested_query": self.suggested_query or f"{self.topic} nedir ve nasıl çalışır?",
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CuriosityItem":
        """Dict'ten öğe kurar."""
        data = d if isinstance(d, dict) else {}
        return cls(
            topic=str(data.get("topic") or ""),
            kind=str(data.get("kind") or "gap"),
            reason=str(data.get("reason") or ""),
            priority=float(data.get("priority", 0.0) or 0.0),
            evidence=[str(e) for e in (data.get("evidence") or [])],
            suggested_query=str(data.get("suggested_query") or ""),
            tags=[str(t) for t in (data.get("tags") or [])],
        )


# ─── Merak Motoru ──────────────────────────────────────────────────────────

class CuriosityEngine:
    """
    Bilgi grafını tarayıp araştırma görevleri üretir.

    Kullanım:
        engine = CuriosityEngine()
        engine.initialize(config, event_bus)
        items = engine.scan(notes, graph)
        plan = engine.next_research_plan(items, top_k=5)
    """

    name = "evolution.curiosity"

    def __init__(self) -> None:
        self._config: Any = None
        self._event_bus: Any = None
        self._max_items: int = 25
        self._counters: Dict[str, float] = {
            "scans": 0.0,
            "items": 0.0,
            "contradictions": 0.0,
        }
        self._last_kinds: Dict[str, int] = {}
        self._initialized: bool = False

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Motoru bağla. İdempotenttir."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        try:
            self._max_items = int(
                getattr(self._config, "curiosity_max_items", 25) or 25
            ) if config is not None else 25
        except (TypeError, ValueError):
            self._max_items = 25
        self._initialized = True
        logger.info("Merak motoru hazır")

    def shutdown(self) -> None:
        """Kaynakları serbest bırak. İdempotenttir."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info("Merak motoru kapatıldı")

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Olay yayınla (bus yoksa atlanır)."""
        if self._event_bus:
            try:
                self._event_bus.emit(event_type, data, source=self.name)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Olay yayınlanamadı ({event_type}): {exc}")

    # ─── Ana Tarama ────────────────────────────────────────────────────

    def scan(
        self,
        notes: List[dict],
        graph: Optional[Dict[str, List[str]]] = None,
        max_items: int = 25,
    ) -> List[CuriosityItem]:
        """
        Tüm merak türlerini tarar, önceliğe göre sıralar ve tekilleştirir.

        Args:
            notes: Kasa not yükleri
            graph: `{kaynak_id: [hedef_id, ...]}` bağlantı haritası
            max_items: Döndürülecek en fazla öğe

        Returns:
            Azalan öncelik sırasında `CuriosityItem` listesi
        """
        self._counters["scans"] += 1.0
        usable = [n for n in (notes or []) if isinstance(n, dict)]
        edges = graph if isinstance(graph, dict) else {}

        collected: List[CuriosityItem] = []

        for producer in (
            lambda: self.knowledge_gaps(usable),
            lambda: self.contradictions(usable, self._extract_contradictions(usable)),
            lambda: self.stale_topics(usable),
            lambda: self.orphans(usable, edges),
            lambda: self.shallow_notes(usable),
            lambda: self.unexplored_clusters(usable, edges),
        ):
            try:
                collected.extend(producer())
            except Exception as exc:
                logger.warning(f"Merak tarayıcı hatası: {exc}")

        ordered = self._rank_and_dedupe(collected)[: max(1, int(max_items))]

        kinds: Dict[str, int] = {}
        for item in ordered:
            kinds[item.kind] = kinds.get(item.kind, 0) + 1
        self._last_kinds = kinds
        self._counters["items"] += float(len(ordered))

        self._emit("evolution.curiosity_scanned", {
            "items": len(ordered),
            "kinds": kinds,
            "note_count": len(usable),
        })
        logger.info(
            f"Merak taraması: {len(ordered)} konu "
            f"({', '.join(f'{k}={v}' for k, v in sorted(kinds.items())) or 'temiz'})"
        )
        return ordered

    def _extract_contradictions(self, notes: Sequence[dict]) -> List[dict]:
        """
        Çelişkileri kendi içinde tespit eder (sentetik bağımlılık olmadan).

        `KnowledgeSynthesizer` enjekte edilmediği için hafif bir karşıt-kelime
        taraması yapılır; `layer.py` gerçek sentezleyici çıktısını geçirir.
        """
        findings: List[dict] = []
        try:
            from neurocore.evolution.synthesizer import (  # yerel import: döngü yok
                KnowledgeSynthesizer,
            )

            findings = KnowledgeSynthesizer().find_contradictions(list(notes))
        except Exception as exc:
            logger.debug(f"Yerel çelişki taraması yapılamadı: {exc}")
        return findings

    # ─── 1) Bilgi Boşlukları ───────────────────────────────────────────

    def knowledge_gaps(self, notes: List[dict]) -> List[CuriosityItem]:
        """
        Bilgi boşluklarını bulur:

        - Bahsedilen ama **kendi notu olmayan** etiketler
        - Düşük güvenli (<0.4) ama önemli (yüksek in-degree) notlar
        - `TODO` / `?` / `belirsiz` işaretli notlar
        - Gövdesinde soru sorulup yanıtlanmayan konular
        """
        items: List[CuriosityItem] = []
        usable = [n for n in (notes or []) if isinstance(n, dict)]
        if not usable:
            return items

        # Etiket → kaç notta geçiyor / kaç notun birincil etiketi
        tag_frequency: Dict[str, int] = {}
        tags_with_own_note: Set[str] = set()
        title_terms: Set[str] = set()

        in_degree: Dict[str, int] = {}
        for index, note in enumerate(usable):
            note_id = _note_id(note, index)
            in_degree.setdefault(note_id, 0)
            title_terms |= _terms(str(note.get("title") or ""))
            for tag in _tag_set(note):
                tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

        for index, note in enumerate(usable):
            title = _norm(note.get("title") or "")
            for tag in _tag_set(note):
                if tag and (tag in title or tag.replace("-", " ") in title):
                    tags_with_own_note.add(tag)

        # Graf referanslarından in-degree
        for note in usable:
            body = _clean_body(note)
            for match in _WIKILINK_RE.findall(body):
                target = _norm(match).strip()
                if target:
                    in_degree[target] = in_degree.get(target, 0) + 1

        # (a) Kendi notu olmayan etiketler
        for tag, frequency in sorted(tag_frequency.items(), key=lambda kv: (-kv[1], kv[0])):
            if tag in tags_with_own_note:
                continue
            items.append(CuriosityItem(
                topic=tag.replace("-", " "),
                kind="gap",
                reason=(
                    f"'{tag}' etiketi {frequency} notta geçiyor ama kendine ait bir notu yok"
                ),
                priority=0.0,
                evidence=[f"tag:{tag}"],
                suggested_query=f"{tag.replace('-', ' ')} nedir ve nasıl çalışır?",
                tags=[tag],
            ))

        # (b) Düşük güvenli ama önemli notlar
        for index, note in enumerate(usable):
            confidence = _confidence(note)
            if confidence >= LOW_CONFIDENCE:
                continue
            note_id = _note_id(note, index)
            importance = in_degree.get(note_id, 0)
            if importance <= 0:
                continue
            title = str(note.get("title") or "").strip() or note_id
            items.append(CuriosityItem(
                topic=title,
                kind="gap",
                reason=(
                    f"'{title}' notu {importance} kez referans veriliyor ama güveni düşük "
                    f"({confidence:.2f})"
                ),
                priority=0.0,
                evidence=[note_id, f"confidence={confidence:.2f}", f"in_degree={importance}"],
                suggested_query=f"{title} hakkında güvenilir kaynaklarla doğrulama yap",
                tags=sorted(_tag_set(note)),
            ))

        # (c) Belirsizlik işaretleri
        for index, note in enumerate(usable):
            body = _clean_body(note)
            lowered = _norm(body)
            hits = [marker for marker in UNCERTAIN_MARKERS if marker in lowered]
            if not hits:
                continue
            title = str(note.get("title") or "").strip() or _note_id(note, index)
            items.append(CuriosityItem(
                topic=title,
                kind="gap",
                reason=f"'{title}' notu tamamlanmamış işaret taşıyor: {', '.join(hits)}",
                priority=0.0,
                evidence=[_note_id(note, index)] + hits,
                suggested_query=f"{title} konusunu tamamla ve belirsizlikleri gider",
                tags=sorted(_tag_set(note)),
            ))

        # (d) Gövdede yanıtsız sorular
        for index, note in enumerate(usable):
            body = _clean_body(note)
            title = str(note.get("title") or "").strip() or _note_id(note, index)
            for sentence in _SENT_RE.split(_WS_RE.sub(" ", body)):
                text = sentence.strip()
                if not text.endswith("?"):
                    continue
                question = text.lstrip("-*•0123456789. )").strip()
                if len(_words(question)) < 3:
                    continue
                items.append(CuriosityItem(
                    topic=question[:120],
                    kind="gap",
                    reason=f"'{title}' notunda yanıtlanmamış soru var",
                    priority=0.0,
                    evidence=[_note_id(note, index), question[:160]],
                    suggested_query=question[:200],
                    tags=sorted(_tag_set(note)),
                ))
        return items

    # ─── 2) Çelişkiler ─────────────────────────────────────────────────

    def contradictions(self, notes: List[dict], contradictions: List[dict]) -> List[CuriosityItem]:
        """
        Verilen çelişki kayıtlarını merak öğelerine çevirir.

        Çelişki kaydı yoksa sessizce boş liste döner (motor yine de çalışır).
        """
        items: List[CuriosityItem] = []
        usable = [n for n in (notes or []) if isinstance(n, dict)]
        index_of: Dict[str, dict] = {}
        for index, note in enumerate(usable):
            index_of[_note_id(note, index)] = note

        for conflict in (contradictions or []):
            if not isinstance(conflict, dict):
                continue
            a = str(conflict.get("a") or "")
            b = str(conflict.get("b") or "")
            reason = str(conflict.get("reason") or "çelişki")
            tokens = [str(t) for t in (conflict.get("tokens") or [])]

            title_a = str((index_of.get(a) or {}).get("title") or a)
            title_b = str((index_of.get(b) or {}).get("title") or b)
            topic = tokens[0] if tokens else f"{title_a} ↔ {title_b}"

            self._counters["contradictions"] += 1.0
            items.append(CuriosityItem(
                topic=str(topic),
                kind="contradiction",
                reason=f"'{title_a}' ile '{title_b}' çelişiyor: {reason}",
                priority=0.0,
                evidence=[a, b] + tokens[:4],
                suggested_query=f"{title_a} ve {title_b} çelişkisini kaynaklarla çöz",
                tags=sorted({t for t in tokens if len(t) > 2})[:6],
            ))
        return items

    # ─── 3) Eskimiş Konular ────────────────────────────────────────────

    def stale_topics(self, notes: List[dict], max_age_days: float = 90.0) -> List[CuriosityItem]:
        """
        Eski güncellenme zamanı **ve** düşük güveni birleştiren konular.

        Yalnızca iki koşul birlikte sağlandığında işaretlenir; aksi hâlde her
        eski not gürültü üretirdi.
        """
        items: List[CuriosityItem] = []
        now = time.time()

        for index, note in enumerate(notes or []):
            if not isinstance(note, dict):
                continue
            updated = _updated_at(note)
            if updated <= 0:
                continue
            age_days = (now - updated) / 86400.0
            confidence = _confidence(note)
            if age_days <= max_age_days or confidence >= LOW_CONFIDENCE:
                continue
            title = str(note.get("title") or "").strip() or _note_id(note, index)
            items.append(CuriosityItem(
                topic=title,
                kind="stale",
                reason=(
                    f"'{title}' {age_days:.0f} gündür güncellenmedi ve güveni düşük "
                    f"({confidence:.2f})"
                ),
                priority=0.0,
                evidence=[_note_id(note, index), f"age_days={age_days:.0f}",
                          f"confidence={confidence:.2f}"],
                suggested_query=f"{title} bilgisi hâlâ geçerli mi? Güncel kaynaklarla doğrula",
                tags=sorted(_tag_set(note)),
            ))
        return items

    # ─── 4) Öksüz Notlar ───────────────────────────────────────────────

    def orphans(self, notes: List[dict], graph: Optional[Dict[str, List[str]]] = None) -> List[CuriosityItem]:
        """
        Ne gelen ne giden bağlantısı olan notlar.

        Bu notlar grafta görünmez; bağlanmadıkça bilgi olarak erişilemez.
        """
        items: List[CuriosityItem] = []
        usable = [n for n in (notes or []) if isinstance(n, dict)]
        if not usable:
            return items

        edges = graph if isinstance(graph, dict) else {}
        incoming: Dict[str, int] = {}
        outgoing: Dict[str, int] = {}

        for source, targets in edges.items():
            outgoing[_norm(source)] = len(targets or [])
            for target in (targets or []):
                key = _norm(target)
                incoming[key] = incoming.get(key, 0) + 1

        for index, note in enumerate(usable):
            note_id = _note_id(note, index)
            key = _norm(note_id)
            title_key = _norm(note.get("title") or "")
            linked_out = outgoing.get(key, 0) or outgoing.get(title_key, 0)
            linked_in = incoming.get(key, 0) or incoming.get(title_key, 0)

            # Gövdedeki [[...]] bağlantıları da giden kenar sayılır
            if not linked_out:
                linked_out = len(_WIKILINK_RE.findall(_clean_body(note)))

            if linked_in or linked_out:
                continue
            title = str(note.get("title") or "").strip() or note_id
            items.append(CuriosityItem(
                topic=title,
                kind="orphan",
                reason=f"'{title}' notunun hiçbir gelen/giden bağlantısı yok",
                priority=0.0,
                evidence=[note_id],
                suggested_query=f"{title} hangi konularla ilişkili? Bağlantı kur",
                tags=sorted(_tag_set(note)),
            ))
        return items

    # ─── 5) Sığ Notlar ─────────────────────────────────────────────────

    def shallow_notes(self, notes: List[dict], min_words: int = 40) -> List[CuriosityItem]:
        """Çok kısa (doyurucu olmayan) notları işaretler."""
        items: List[CuriosityItem] = []
        threshold = max(1, int(min_words))

        for index, note in enumerate(notes or []):
            if not isinstance(note, dict):
                continue
            body = _clean_body(note)
            word_count = len(_words(body))
            if word_count >= threshold:
                continue
            title = str(note.get("title") or "").strip() or _note_id(note, index)
            items.append(CuriosityItem(
                topic=title,
                kind="shallow",
                reason=f"'{title}' yalnızca {word_count} kelime içeriyor (eşik: {threshold})",
                priority=0.0,
                evidence=[_note_id(note, index), f"words={word_count}"],
                suggested_query=f"{title} konusunu detaylandır: örnekler, sayılar ve koşullar ekle",
                tags=sorted(_tag_set(note)),
            ))
        return items

    # ─── 6) Keşfedilmemiş Kümeler ──────────────────────────────────────

    def unexplored_clusters(
        self,
        notes: List[dict],
        graph: Optional[Dict[str, List[str]]] = None,
    ) -> List[CuriosityItem]:
        """
        Bağlantı grafiğinde >2 notlu, hiç **sentezlenmemiş** bileşenleri bulur.

        Union-find ile bağlantılı bileşenler çıkarılır; bir kümenin baskın
        etiketi herhangi bir notun başlığında geçmiyorsa küme "keşfedilmemiş"
        sayılır ve üst seviye bir sentez notu önerilir.
        """
        items: List[CuriosityItem] = []
        usable = [n for n in (notes or []) if isinstance(n, dict)]
        edges = graph if isinstance(graph, dict) else {}
        if not usable:
            return items

        id_of: Dict[str, str] = {}
        note_of: Dict[str, dict] = {}
        for index, note in enumerate(usable):
            note_id = _note_id(note, index)
            id_of[note_id] = note_id
            note_of[note_id] = note
            title_key = _norm(note.get("title") or "")
            if title_key:
                id_of[title_key] = note_id

        union = _UnionFind()
        for note_id in id_of.values():
            union.add(note_id)

        has_edges = False
        for source, targets in edges.items():
            source_id = id_of.get(_norm(source)) or id_of.get(str(source))
            if not source_id:
                continue
            for target in (targets or []):
                target_id = id_of.get(_norm(target)) or id_of.get(str(target))
                if not target_id or target_id == source_id:
                    continue
                union.union(source_id, target_id)
                has_edges = True

        if not has_edges:
            return items

        # Mevcut sentez notlarının kapsadığı başlıklar
        covered_titles = " | ".join(
            _norm(note.get("title") or "") for note in usable if isinstance(note, dict)
        )

        for _, members in union.components().items():
            if len(members) <= 2:
                continue
            tag_counts: Dict[str, int] = {}
            for member in members:
                for tag in _tag_set(note_of.get(member, {})):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if not tag_counts:
                continue
            dominant = max(tag_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            label = dominant.replace("-", " ")
            if label and label in covered_titles:
                continue  # zaten sentezlenmiş

            member_titles = [
                str(note_of.get(member, {}).get("title") or member) for member in members[:4]
            ]
            items.append(CuriosityItem(
                topic=label,
                kind="unexplored_cluster",
                reason=(
                    f"'{label}' çevresinde {len(members)} not bağlantılı ama üst seviye "
                    "sentez notu yok"
                ),
                priority=0.0,
                evidence=list(members[:6]) + member_titles[:2],
                suggested_query=f"{label} konusundaki notları birleştirip sentez notu üret",
                tags=sorted(tag_counts)[:6],
            ))
        return items

    # ─── Sıralama ve Tekilleştirme ─────────────────────────────────────

    def _rank_and_dedupe(self, items: Sequence[CuriosityItem]) -> List[CuriosityItem]:
        """
        Öncelik hesaplar, konuya göre tekilleştirir ve azalan sırada döndürür.

        Öncelik = tür ağırlığı + 0.05 × kanıt sayısı + konu uzunluğu katkısı,
        [0, 1] aralığına sıkıştırılır. Aynı konu birden çok kez görülürse en
        yüksek öncelikli kayıt tutulur ve kanıtları birleştirilir.
        """
        best: Dict[str, CuriosityItem] = {}
        repeats: Dict[str, int] = {}

        for item in items:
            if not isinstance(item, CuriosityItem) or not str(item.topic).strip():
                continue
            key = _norm(item.topic)
            weight = KIND_WEIGHTS.get(item.kind, 0.3)
            evidence_bonus = min(0.12, 0.04 * len(item.evidence or []))
            topic_bonus = min(0.06, len(_words(item.topic)) * 0.015)
            item.priority = round(_clamp(weight + evidence_bonus + topic_bonus), 4)

            if key not in best:
                best[key] = item
                repeats[key] = 1
                continue

            repeats[key] += 1
            current = best[key]
            if item.priority > current.priority:
                merged = list(current.evidence or []) + list(item.evidence or [])
                item.evidence = merged[:8]
                best[key] = item
            else:
                current.evidence = list(current.evidence or []) + list(item.evidence or [])
                current.evidence = current.evidence[:8]

        ordered: List[CuriosityItem] = []
        for key, item in best.items():
            # Aynı konu birden çok türde görüldüyse hafif öncelik artışı
            if repeats.get(key, 1) > 1:
                item.priority = round(_clamp(item.priority + 0.03), 4)
            ordered.append(item)

        ordered.sort(key=lambda entry: (-entry.priority, entry.kind, entry.topic))
        return ordered

    # ─── Araştırma Planı ───────────────────────────────────────────────

    def next_research_plan(self, items: List[CuriosityItem], top_k: int = 5) -> List[dict]:
        """
        Merak öğelerini sıralı araştırma görevlerine çevirir.

        Returns:
            `[{"topic", "query", "priority", "kind", "expected_notes"}]`
        """
        limit = max(1, int(top_k))
        ordered = sorted(
            [i for i in (items or []) if isinstance(i, CuriosityItem)],
            key=lambda entry: (-entry.priority, entry.topic),
        )

        plan: List[dict] = []
        seen: Set[str] = set()
        for item in ordered:
            key = _norm(item.topic)
            if key in seen:
                continue
            seen.add(key)
            plan.append({
                "topic": item.topic,
                "query": item.suggested_query or f"{item.topic} nedir?",
                "priority": round(float(item.priority), 4),
                "kind": item.kind,
                "expected_notes": EXPECTED_NOTES.get(item.kind, 1),
            })
            if len(plan) >= limit:
                break

        self._emit("evolution.research_planned", {"tasks": len(plan)})
        return plan

    # ─── İstatistikler ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "scans": int(self._counters["scans"]),
            "items_found": int(self._counters["items"]),
            "contradictions_found": int(self._counters["contradictions"]),
            "last_kinds": dict(self._last_kinds),
            "kind_weights": dict(KIND_WEIGHTS),
        }
