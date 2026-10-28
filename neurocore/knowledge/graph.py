"""
NeuroCore Bilgi Grafı
======================
Notlar arası ilişkileri yöneten, analiz eden ve görselleştiren graf katmanı.

Yetenekler:
    * **İlişki türleri** — `links` (yazarın kurduğu), `backlink`, `similar`,
      `contradicts`, `derived_from`, `supports`, `part_of`, `related`
    * **Merkezilik** — PageRank, derece, arasındalık (betweenness), yakınlık
    * **Topluluk tespiti** — etiket yayılımı (label propagation)
    * **Kopuk parçalar** — bağlantısız bileşenler (union-find)
    * **Gezinti** — komşu genişletme, en kısa yol, alt graf çıkarma
    * **Anomali** — yetim notlar, kırık bağlantılar, zayıf köprüler
    * **Dışa aktarma** — JSON, Mermaid, DOT, Markdown

networkx kuruluysa ağır analizler (betweenness vb.) onunla yapılır; değilse
saf Python yedekleri devreye girer — sistem her koşulda çalışır.
"""

from __future__ import annotations

import heapq
import json
import math
import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("knowledge.graph")


# ─── İlişki Türleri ─────────────────────────────────────────────────────────

#: Desteklenen kenar ilişki türleri ve yönlü olup olmadıkları
RELATION_TYPES: Dict[str, bool] = {
    "links": True,          # [[wiki]] bağlantısı (yönlü)
    "backlink": True,       # ters yön (türetilmiş, saklanmaz)
    "similar": False,       # içerik benzerliği (yönsüz)
    "related": False,       # genel ilişki (yönsüz)
    "contradicts": False,   # çelişki (yönsüz)
    "supports": True,       # destekler (yönlü)
    "derived_from": True,   # türetilmiş (yönlü)
    "part_of": True,        # parçası (yönlü)
    "merged_into": True,    # birleştirildi (yönlü)
    "synthesized": True,    # sentez kaynağı (yönlü)
}

#: Kenar ağırlığı varsayılanları
RELATION_WEIGHTS: Dict[str, float] = {
    "links": 1.0,
    "backlink": 1.0,
    "similar": 0.5,
    "related": 0.6,
    "contradicts": 1.2,
    "supports": 0.9,
    "derived_from": 1.0,
    "part_of": 0.8,
    "merged_into": 1.0,
    "synthesized": 0.9,
}


class GraphError(Exception):
    """Graf işlem hatası."""


# ─── Graf Veri Yapıları ─────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """Graf düğümü (bir not)."""
    id: str
    title: str = ""
    kind: str = "concept"
    category: str = "learned"
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.5
    importance: float = 0.5
    word_count: int = 0
    updated: float = 0.0
    path: str = ""
    archived: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "kind": self.kind,
            "category": self.category, "tags": list(self.tags),
            "confidence": round(self.confidence, 4),
            "importance": round(self.importance, 4),
            "word_count": self.word_count,
            "updated": self.updated, "path": self.path,
            "archived": self.archived,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            kind=str(data.get("kind", "concept")),
            category=str(data.get("category", "learned")),
            tags=list(data.get("tags") or []),
            confidence=float(data.get("confidence", 0.5) or 0.5),
            importance=float(data.get("importance", 0.5) or 0.5),
            word_count=int(data.get("word_count", 0) or 0),
            updated=float(data.get("updated", 0.0) or 0.0),
            path=str(data.get("path", "")),
            archived=bool(data.get("archived", False)),
        )


@dataclass
class GraphEdge:
    """Graf kenarı (iki not arası ilişki)."""
    src: str
    dst: str
    relation: str = "links"
    weight: float = 1.0
    reason: str = ""
    unresolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src, "dst": self.dst, "relation": self.relation,
            "weight": round(self.weight, 4), "reason": self.reason,
            "unresolved": self.unresolved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        return cls(
            src=str(data.get("src", "")),
            dst=str(data.get("dst", "")),
            relation=str(data.get("relation", "links")),
            weight=float(data.get("weight", 1.0) or 1.0),
            reason=str(data.get("reason", "")),
            unresolved=bool(data.get("unresolved", False)),
        )


@dataclass
class GraphMetrics:
    """Graf geneli metrikler."""
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    mean_degree: float = 0.0
    max_degree: int = 0
    component_count: int = 0
    largest_component: int = 0
    isolated_count: int = 0
    broken_links: int = 0
    cycle_count: int = 0
    relation_counts: Dict[str, int] = field(default_factory=dict)
    top_hubs: List[Tuple[str, float]] = field(default_factory=list)
    top_authorities: List[Tuple[str, float]] = field(default_factory=list)
    community_count: int = 0
    tag_clusters: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "density": round(self.density, 5),
            "mean_degree": round(self.mean_degree, 3),
            "max_degree": self.max_degree,
            "component_count": self.component_count,
            "largest_component": self.largest_component,
            "isolated_count": self.isolated_count,
            "broken_links": self.broken_links,
            "cycle_count": self.cycle_count,
            "relation_counts": dict(self.relation_counts),
            "top_hubs": [(nid, round(s, 4)) for nid, s in self.top_hubs],
            "top_authorities": [(nid, round(s, 4)) for nid, s in self.top_authorities],
            "community_count": self.community_count,
            "tag_clusters": dict(list(self.tag_clusters.items())[:20]),
        }


# ─── Ana Graf Sınıfı ────────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    Notlar arası ilişki grafı.

    Kullanım:
        graph = KnowledgeGraph()
        graph.initialize(config, event_bus)

        graph.rebuild(index)                       # indeksten/notlardan kur
        graph.add_edge(GraphEdge("kn_a", "kn_b", "supports"))

        related = graph.neighbors("kn_a", depth=2)
        metrics = graph.metrics()
        print(graph.to_mermaid(focus="kn_a"))

    Graf, `KnowledgeIndex`'ten bağımsız olarak da kullanılabilir; `rebuild`
    hem `KnowledgeIndex` hem `KnowledgeVault` hem de düz not listesi kabul eder.
    """

    name = "knowledge.graph"

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[Tuple[str, str, str], GraphEdge] = {}
        self._out: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        self._in: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        self._lock = threading.RLock()
        self._config = None
        self._event_bus = None
        self._initialized = False

        # Analiz önbellekleri (graf değiştiğinde geçersiz kılınır)
        self._cache: Dict[str, Any] = {}
        self._pagerank_cache: Optional[Dict[str, float]] = None

        self._counter = {
            "rebuilds": 0, "edge_adds": 0, "edge_removes": 0,
            "traversals": 0, "analyses": 0, "broken_links": 0,
        }

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Graf katmanını başlat (idempotent)."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._initialized = True
        logger.info("Bilgi grafı hazır")

    def shutdown(self) -> None:
        """Kaynakları serbest bırak."""
        self._initialized = False
        logger.info(
            f"Bilgi grafı kapatıldı ({len(self._nodes)} düğüm, {len(self._edges)} kenar)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Graf istatistikleri."""
        return {
            **self._counter,
            "initialized": self._initialized,
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "relations": self.relation_counts(),
        }

    # ─── Düğüm ve Kenar Yönetimi ────────────────────────────────────────

    def add_node(self, node: GraphNode) -> GraphNode:
        """Düğüm ekler veya günceller."""
        with self._lock:
            self._nodes[node.id] = node
            self._invalidate()
        return node

    def add_nodes(self, nodes: Iterable[GraphNode]) -> int:
        """Çok sayıda düğüm ekler."""
        count = 0
        with self._lock:
            for node in nodes:
                self._nodes[node.id] = node
                count += 1
            self._invalidate()
        return count

    def add_edge(
        self,
        src: str,
        dst: str,
        relation: str = "links",
        weight: Optional[float] = None,
        reason: str = "",
        *,
        create_missing: bool = True,
    ) -> GraphEdge:
        """
        Kenar ekler (varsa ağırlığı günceller).

        Args:
            src: Kaynak not kimliği
            dst: Hedef not kimliği
            relation: İlişki türü (RELATION_TYPES)
            weight: Ağırlık (None ise ilişki türüne göre varsayılan)
            reason: İnsan okur gerekçe
            create_missing: Eksik düğümler için yer tutucu oluştur

        Returns:
            Eklenen/güncellenen kenar
        """
        if not src or not dst:
            raise GraphError("Kenar için src ve dst zorunludur")
        if src == dst:
            raise GraphError("Öz-döngü eklenemez")

        resolved_relation = relation if relation in RELATION_TYPES else "related"
        resolved_weight = (
            float(weight) if weight is not None
            else RELATION_WEIGHTS.get(resolved_relation, 0.6)
        )

        with self._lock:
            if create_missing:
                for node_id, label in ((src, "kaynak"), (dst, "hedef")):
                    if node_id not in self._nodes:
                        self._nodes[node_id] = GraphNode(
                            id=node_id,
                            title=f"<eksik {label}: {node_id}>",
                            kind="meta",
                            category="meta",
                        )
            dst_known = dst in self._nodes
            edge = GraphEdge(
                src=src, dst=dst, relation=resolved_relation,
                weight=resolved_weight, reason=reason,
                unresolved=not dst_known,
            )
            if edge.unresolved:
                self._counter["broken_links"] += 1
            key = (src, dst, resolved_relation)
            self._edges[key] = edge
            self._out[src].add(key)
            self._in[dst].add(key)
            self._invalidate()
            self._counter["edge_adds"] += 1
        return edge

    def add_edges(self, edges: Iterable[GraphEdge], create_missing: bool = True) -> int:
        """Çok sayıda kenar ekler."""
        count = 0
        for edge in edges:
            try:
                self.add_edge(
                    edge.src, edge.dst, edge.relation, edge.weight,
                    edge.reason, create_missing=create_missing,
                )
                count += 1
            except GraphError:
                continue
        return count

    def remove_edge(self, src: str, dst: str, relation: Optional[str] = None) -> int:
        """Kenar(lar)ı siler; silinen sayısını döndürür."""
        with self._lock:
            keys = [
                key for key in list(self._edges.keys())
                if key[0] == src and key[1] == dst and (relation is None or key[2] == relation)
            ]
            for key in keys:
                self._edges.pop(key, None)
                self._out.get(src, set()).discard(key)
                self._in.get(dst, set()).discard(key)
            if keys:
                self._invalidate()
                self._counter["edge_removes"] += len(keys)
            return len(keys)

    def remove_node(self, node_id: str) -> int:
        """Düğümü ve tüm ilişkili kenarları siler."""
        with self._lock:
            if node_id not in self._nodes:
                return 0
            keys = list(self._out.get(node_id, set())) + list(self._in.get(node_id, set()))
            for key in keys:
                self._edges.pop(key, None)
                self._out.get(key[0], set()).discard(key)
                self._in.get(key[1], set()).discard(key)
            self._out.pop(node_id, None)
            self._in.pop(node_id, None)
            self._nodes.pop(node_id, None)
            self._invalidate()
            return len(keys)

    def has_node(self, node_id: str) -> bool:
        """Düğüm var mı?"""
        return node_id in self._nodes

    def node(self, node_id: str) -> Optional[GraphNode]:
        """Düğümü döndürür."""
        return self._nodes.get(node_id)

    def title_of(self, node_id: str) -> str:
        """Düğüm başlığını (yoksa kimliği) döndürür."""
        node = self._nodes.get(node_id)
        return node.title if node and node.title else node_id

    @property
    def node_ids(self) -> List[str]:
        """Tüm düğüm kimlikleri."""
        return list(self._nodes.keys())

    @property
    def edges(self) -> List[GraphEdge]:
        """Tüm kenarlar."""
        return list(self._edges.values())

    @property
    def node_count(self) -> int:
        """Düğüm sayısı."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Kenar sayısı."""
        return len(self._edges)

    # ─── Graf Kurulumu ──────────────────────────────────────────────────

    def rebuild(self, source: Any, *, derive_similarity: bool = False,
                similarity_threshold: float = 0.72, embed_fn: Any = None) -> Dict[str, Any]:
        """
        Grafı bir kaynaktan tamamen yeniden kurar.

        Args:
            source: `KnowledgeIndex`, `KnowledgeVault` veya not listesi
                (her not `id`,`title`,`links`,`tags` alanlarını içermeli)
            derive_similarity: Benzerlik kenarları türet (etiket + başlık örtüşmesi)
            similarity_threshold: Benzerlik eşiği
            embed_fn: (metin) -> vektör; verilirse benzerlik gömme ile hesaplanır

        Returns:
            {"nodes": int, "edges": int, "similar": int, "broken": int}
        """
        notes = self._load_source_notes(source)
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._out.clear()
            self._in.clear()
            self._invalidate()
            self._counter["broken_links"] = 0
            self._counter["rebuilds"] += 1

        known: Set[str] = set()
        for raw in notes:
            node = self._note_to_node(raw)
            if not node.id:
                continue
            known.add(node.id)
            with self._lock:
                self._nodes[node.id] = node

        edge_count = 0
        broken = 0
        for raw in notes:
            src = str(raw.get("id") or "")
            if not src:
                continue
            # Açık `links` alanı
            for dst in (raw.get("links") or []):
                dst = str(dst)
                if not dst or dst == src:
                    continue
                if dst not in known:
                    broken += 1
                try:
                    self.add_edge(src, dst, "links", create_missing=True)
                    edge_count += 1
                except GraphError:
                    continue
            # Gövdedeki [[wiki]] bağlantıları (id veya başlık biçiminde)
            body = str(raw.get("body") or "")
            if body:
                for target, _label in _extract_wikilinks(body):
                    dst = target if target in known else self._resolve_by_title(target, known, notes)
                    if not dst or dst == src:
                        continue
                    try:
                        self.add_edge(src, dst, "links",
                                      reason="gövde bağlantısı", create_missing=True)
                        edge_count += 1
                    except GraphError:
                        continue
            # Yapısal ilişkiler (frontmatter'dan)
            for relation in ("supports", "contradicts", "derived_from", "part_of",
                             "merged_into", "related"):
                for dst in _as_list(raw.get(relation)):
                    if dst and dst != src:
                        try:
                            self.add_edge(src, str(dst), relation, create_missing=True)
                            edge_count += 1
                        except GraphError:
                            continue

        similar = 0
        if derive_similarity:
            similar = self.derive_similarity_edges(
                threshold=similarity_threshold, embed_fn=embed_fn
            )

        self._counter["broken_links"] = broken
        self._invalidate()
        result = {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "similar": similar,
            "broken": broken,
        }
        self._emit("knowledge.graph_rebuilt", result)
        logger.info(
            f"Graf kuruldu: {result['nodes']} düğüm, {result['edges']} kenar "
            f"({similar} benzerlik, {broken} kırık bağlantı)"
        )
        return result

    @staticmethod
    def _load_source_notes(source: Any) -> List[Dict[str, Any]]:
        """Kaynağı (indeks/kasa/liste) not sözlükleri listesine çevirir."""
        if source is None:
            return []
        if isinstance(source, (list, tuple)):
            return [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in source]
        if hasattr(source, "iter_notes"):
            return [note.to_dict() for note in source.iter_notes()]
        if hasattr(source, "load_all"):
            return [note.to_dict() if hasattr(note, "to_dict") else dict(note)
                    for note in source.load_all()]
        raise GraphError("Graf kaynağı anlaşılamadı (indeks, kasa veya not listesi bekleniyor)")

    @staticmethod
    def _note_to_node(raw: Dict[str, Any]) -> GraphNode:
        """Not sözlüğünü graf düğümüne çevirir."""
        updated = raw.get("updated")
        if isinstance(updated, str):
            from neurocore.knowledge.note import parse_timestamp
            updated = parse_timestamp(updated)
        return GraphNode(
            id=str(raw.get("id", "")),
            title=str(raw.get("title", "")),
            kind=str(raw.get("kind", "concept")),
            category=str(raw.get("category", "learned")),
            tags=[str(t) for t in (raw.get("tags") or [])],
            confidence=float(raw.get("confidence", 0.5) or 0.5),
            importance=float(raw.get("importance", 0.5) or 0.5),
            word_count=int(raw.get("word_count", 0) or 0),
            updated=float(updated or 0.0),
            path=str(raw.get("path", "")),
            archived=bool(raw.get("archived", False)),
        )

    @staticmethod
    def _resolve_by_title(target: str, known: Set[str], notes: List[Dict[str, Any]]) -> str:
        """`[[Başlık]]` biçimindeki bağlantıyı not kimliğine çözer."""
        from neurocore.knowledge.note import normalize_title_for_compare
        needle = normalize_title_for_compare(target)
        if not needle:
            return ""
        best_id = ""
        best_score = 0.0
        for raw in notes:
            note_id = str(raw.get("id") or "")
            if note_id not in known:
                continue
            candidate = normalize_title_for_compare(str(raw.get("title") or ""))
            if candidate == needle:
                return note_id
            if candidate and (needle in candidate or candidate in needle):
                overlap = len(set(needle.split()) & set(candidate.split()))
                if overlap > best_score:
                    best_score = overlap
                    best_id = note_id
        return best_id

    # ─── Benzerlik Kenarları ────────────────────────────────────────────

    def derive_similarity_edges(self, threshold: float = 0.72,
                                max_per_node: int = 5,
                                embed_fn: Any = None) -> int:
        """
        Etiket ve başlık örtüşmesinden (veya gömme vektörlerinden) benzerlik
        kenarları türetir.

        Args:
            threshold: Benzerlik eşiği (0..1)
            max_per_node: Düğüm başına en fazla kaç benzerlik kenarı
            embed_fn: (metin) -> vektör; verilirse kosinüs benzerliği kullanılır

        Returns:
            Eklenen kenar sayısı
        """
        nodes = list(self._nodes.values())
        if len(nodes) < 2:
            return 0

        # Etiket bloğu indeksi: ortak etiket yoksa karşılaştırmaya gerek yok
        by_tag: Dict[str, Set[str]] = defaultdict(set)
        for node in nodes:
            for tag in node.tags:
                by_tag[tag].add(node.id)

        pairs: Set[Tuple[str, str]] = set()
        for tag, members in by_tag.items():
            if len(members) < 2 or len(members) > 200:
                continue
            ordered = sorted(members)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    pairs.add((ordered[i], ordered[j]))
        # Ortak etiketi olmayan ama başlığı benzeyen notlar için kısmi tarama
        if len(nodes) <= 400:
            ordered_ids = sorted(node.id for node in nodes)
            for i in range(len(ordered_ids)):
                for j in range(i + 1, len(ordered_ids)):
                    pairs.add((ordered_ids[i], ordered_ids[j]))

        vectors: Dict[str, List[float]] = {}
        if embed_fn is not None:
            for node in nodes:
                try:
                    vectors[node.id] = list(embed_fn(f"{node.title} {' '.join(node.tags)}"))
                except Exception as exc:
                    logger.debug(f"Gömme üretilemedi ({node.id}): {exc}")

        added = 0
        per_node: Dict[str, int] = defaultdict(int)
        scored: List[Tuple[float, str, str]] = []

        for a_id, b_id in pairs:
            node_a = self._nodes.get(a_id)
            node_b = self._nodes.get(b_id)
            if not node_a or not node_b:
                continue
            if vectors and a_id in vectors and b_id in vectors:
                score = _cosine(vectors[a_id], vectors[b_id])
            else:
                score = self._structural_similarity(node_a, node_b)
            if score >= threshold:
                scored.append((score, a_id, b_id))

        scored.sort(key=lambda item: -item[0])
        for score, a_id, b_id in scored:
            if per_node[a_id] >= max_per_node or per_node[b_id] >= max_per_node:
                continue
            try:
                self.add_edge(a_id, b_id, "similar", weight=round(score, 4),
                              reason=f"yapısal benzerlik {score:.2f}",
                              create_missing=False)
                per_node[a_id] += 1
                per_node[b_id] += 1
                added += 1
            except GraphError:
                continue
        return added

    @staticmethod
    def _structural_similarity(a: GraphNode, b: GraphNode) -> float:
        """İki notun yapısal benzerliği: etiket Jaccard + başlık örtüşmesi."""
        from neurocore.knowledge.note import title_similarity
        tags_a, tags_b = set(a.tags), set(b.tags)
        tag_score = (len(tags_a & tags_b) / len(tags_a | tags_b)) if (tags_a | tags_b) else 0.0
        title_score = title_similarity(a.title, b.title)
        kind_bonus = 0.1 if a.kind == b.kind else 0.0
        return min(1.0, 0.5 * tag_score + 0.45 * title_score + kind_bonus)

    # ─── Gezinti ────────────────────────────────────────────────────────

    def neighbors(self, node_id: str, depth: int = 1, direction: str = "both",
                  relations: Optional[Iterable[str]] = None) -> List[str]:
        """
        Genişlik-öncelikli komşu genişletme.

        Args:
            node_id: Başlangıç düğümü
            depth: Kaç katman genişletilecek (1 = doğrudan komşular)
            direction: `out`, `in` veya `both`
            relations: Yalnızca bu ilişki türleri izlenir

        Returns:
            Başlangıç hariç, mesafeye göre sıralı düğüm kimlikleri
        """
        self._counter["traversals"] += 1
        allowed = set(relations) if relations else None
        visited: Set[str] = {node_id}
        order: List[str] = []
        queue: deque = deque([(node_id, 0)])

        while queue:
            current, level = queue.popleft()
            if level >= depth:
                continue
            for neighbor in self._direct_neighbors(current, direction, allowed):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                order.append(neighbor)
                queue.append((neighbor, level + 1))
        return order

    def _direct_neighbors(self, node_id: str, direction: str,
                          allowed: Optional[Set[str]]) -> List[str]:
        """Tek katman komşular."""
        result: List[str] = []
        if direction in ("out", "both"):
            for key in self._out.get(node_id, ()):  # type: ignore[arg-type]
                if allowed and key[2] not in allowed:
                    continue
                result.append(key[1])
        if direction in ("in", "both"):
            for key in self._in.get(node_id, ()):  # type: ignore[arg-type]
                if allowed and key[2] not in allowed:
                    continue
                result.append(key[0])
        return result

    def subgraph(self, node_ids: Iterable[str], include_external: bool = False) -> "KnowledgeGraph":
        """Verilen düğümlerden alt graf oluşturur."""
        wanted = {str(i) for i in node_ids}
        result = KnowledgeGraph()
        result.initialize(self._config, self._event_bus)
        for node_id in wanted:
            node = self._nodes.get(node_id)
            if node:
                result.add_node(GraphNode(**node.to_dict()))
        for edge in self._edges.values():
            in_src = edge.src in wanted
            in_dst = edge.dst in wanted
            if in_src and in_dst:
                result.add_edge(edge.src, edge.dst, edge.relation, edge.weight,
                                edge.reason, create_missing=True)
            elif include_external and (in_src or in_dst):
                result.add_edge(edge.src, edge.dst, edge.relation, edge.weight,
                                edge.reason, create_missing=True)
        return result

    def expand_context(self, node_ids: Sequence[str], depth: int = 1,
                       relations: Optional[Iterable[str]] = None) -> List[str]:
        """
        Bir not kümesini komşularıyla genişletir (RAG bağlamı için).

        Girdi kimlikleri korunur, yeniler mesafeye göre eklenir.
        """
        ordered: List[str] = [nid for nid in node_ids if nid in self._nodes]
        seen: Set[str] = set(ordered)
        for node_id in list(ordered):
            for neighbor in self.neighbors(node_id, depth=depth, relations=relations):
                if neighbor not in seen and neighbor in self._nodes:
                    seen.add(neighbor)
                    ordered.append(neighbor)
        return ordered

    def shortest_path(self, src: str, dst: str, max_depth: int = 6,
                      relations: Optional[Iterable[str]] = None) -> List[str]:
        """
        İki not arasındaki en kısa yolu bulur (BFS).

        Returns:
            Yol üzerindeki düğüm kimlikleri (src ve dst dahil); yol yoksa []
        """
        if src == dst:
            return [src] if src in self._nodes else []
        if src not in self._nodes or dst not in self._nodes:
            return []

        allowed = set(relations) if relations else None
        previous: Dict[str, str] = {src: ""}
        queue: deque = deque([(src, 0)])
        while queue:
            current, level = queue.popleft()
            if level >= max_depth:
                continue
            for neighbor in self._direct_neighbors(current, "both", allowed):
                if neighbor in previous:
                    continue
                previous[neighbor] = current
                if neighbor == dst:
                    path = [dst]
                    while path[-1] != src:
                        path.append(previous[path[-1]])
                    return list(reversed(path))
                queue.append((neighbor, level + 1))
        return []

    def common_neighbors(self, a: str, b: str) -> List[str]:
        """İki notun ortak komşuları (öneri motoru için)."""
        set_a = set(self._direct_neighbors(a, "both", None))
        set_b = set(self._direct_neighbors(b, "both", None))
        return sorted(set_a & set_b)

    # ─── Analiz ─────────────────────────────────────────────────────────

    def _invalidate(self) -> None:
        """Analiz önbelleklerini geçersiz kılar."""
        self._cache.clear()
        self._pagerank_cache = None

    def pagerank(self, damping: float = 0.85, iterations: int = 30,
                 tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Ağırlıklı PageRank hesaplar (önem merkeziliği).

        Giden bağlantısı olmayan düğümlerin ağırlığı tüm düğümlere dağıtılır
        (dangling node düzeltmesi).

        Returns:
            {node_id: skor} — toplam 1.0
        """
        if self._pagerank_cache is not None:
            return self._pagerank_cache
        self._counter["analyses"] += 1

        nodes = list(self._nodes.keys())
        n = len(nodes)
        if n == 0:
            return {}
        if n == 1:
            result = {nodes[0]: 1.0}
            self._pagerank_cache = result
            return result

        index = {node_id: i for i, node_id in enumerate(nodes)}
        out_weight = [0.0] * n
        out_links: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

        for edge in self._edges.values():
            if edge.src not in index or edge.dst not in index:
                continue
            i, j = index[edge.src], index[edge.dst]
            out_links[i].append((j, edge.weight))
            out_weight[i] += edge.weight

        rank = [1.0 / n] * n
        for _ in range(iterations):
            new_rank = [(1.0 - damping) / n] * n
            dangling = 0.0
            for i in range(n):
                if out_weight[i] <= 0.0:
                    dangling += rank[i]
                    continue
                factor = damping * rank[i] / out_weight[i]
                for j, weight in out_links[i]:
                    new_rank[j] += factor * weight
            if dangling > 0.0:
                share = damping * dangling / n
                for i in range(n):
                    new_rank[i] += share
            delta = sum(abs(new_rank[i] - rank[i]) for i in range(n))
            rank = new_rank
            if delta < tolerance:
                break

        total = sum(rank) or 1.0
        result = {nodes[i]: rank[i] / total for i in range(n)}
        self._pagerank_cache = result
        return result

    def degree_centrality(self) -> Dict[str, float]:
        """Derece merkeziliği (giren + çıkan, ağırlıklı)."""
        result: Dict[str, float] = {}
        max_possible = max(1, len(self._nodes) - 1)
        for node_id in self._nodes:
            total = 0.0
            for key in self._out.get(node_id, ()):  # type: ignore[arg-type]
                total += self._edges[key].weight
            for key in self._in.get(node_id, ()):  # type: ignore[arg-type]
                total += self._edges[key].weight
            result[node_id] = min(1.0, total / max_possible)
        return result

    def in_degree(self) -> Dict[str, int]:
        """Giren kenar sayısı (kaç not bu nota bağlanıyor)."""
        return {node_id: len(self._in.get(node_id, ())) for node_id in self._nodes}  # type: ignore[arg-type]

    def out_degree(self) -> Dict[str, int]:
        """Çıkan kenar sayısı."""
        return {node_id: len(self._out.get(node_id, ())) for node_id in self._nodes}  # type: ignore[arg-type]

    def betweenness_centrality(self, sample: Optional[int] = None) -> Dict[str, float]:
        """
        Arasındalık merkeziliği — "köprü" notları bulur.

        networkx varsa onu kullanır (daha hızlı ve doğru); yoksa Brandes
        algoritmasının ağırlıksız saf Python uygulamasına düşer.

        Args:
            sample: Örneklem düğüm sayısı (büyük graflarda yaklaşık hesap)
        """
        self._counter["analyses"] += 1
        try:
            import networkx as nx  # type: ignore
            graph = self.to_networkx()
            if sample and sample < graph.number_of_nodes():
                result = nx.betweenness_centrality(graph, k=sample, weight="weight")
            else:
                result = nx.betweenness_centrality(graph, weight="weight")
            return {k: float(v) for k, v in result.items()}
        except ImportError:
            logger.debug("networkx yok — arasındalık saf Python ile hesaplanıyor")
            return self._brandes()

    def _brandes(self) -> Dict[str, float]:
        """Brandes algoritması (ağırlıksız, saf Python)."""
        nodes = list(self._nodes.keys())
        centrality = {node_id: 0.0 for node_id in nodes}
        adjacency = {node_id: self._direct_neighbors(node_id, "both", None) for node_id in nodes}

        for source in nodes:
            stack: List[str] = []
            predecessors: Dict[str, List[str]] = {node_id: [] for node_id in nodes}
            sigma = {node_id: 0.0 for node_id in nodes}
            sigma[source] = 1.0
            distance = {node_id: -1 for node_id in nodes}
            distance[source] = 0
            queue: deque = deque([source])

            while queue:
                vertex = queue.popleft()
                stack.append(vertex)
                for neighbor in adjacency.get(vertex, []):
                    if distance[neighbor] < 0:
                        distance[neighbor] = distance[vertex] + 1
                        queue.append(neighbor)
                    if distance[neighbor] == distance[vertex] + 1:
                        sigma[neighbor] += sigma[vertex]
                        predecessors[neighbor].append(vertex)

            delta = {node_id: 0.0 for node_id in nodes}
            while stack:
                vertex = stack.pop()
                for predecessor in predecessors.get(vertex, []):
                    if sigma[vertex] > 0:
                        delta[predecessor] += (sigma[predecessor] / sigma[vertex]) * (1.0 + delta[vertex])
                if vertex != source:
                    centrality[vertex] += delta[vertex]

        # Yönsüz grafta her yol iki kez sayılır
        scale = 1.0 / max(1.0, (len(nodes) - 1) * (len(nodes) - 2))
        return {node_id: value * scale for node_id, value in centrality.items()}

    def components(self) -> List[List[str]]:
        """
        Bağlantılı bileşenler (union-find, yönsüz kabul).

        Returns:
            Büyükten küçüğe sıralı bileşen listesi
        """
        parent: Dict[str, str] = {node_id: node_id for node_id in self._nodes}

        def find(x: str) -> str:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(a: str, b: str) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for edge in self._edges.values():
            if edge.src in parent and edge.dst in parent:
                union(edge.src, edge.dst)

        groups: Dict[str, List[str]] = defaultdict(list)
        for node_id in self._nodes:
            groups[find(node_id)].append(node_id)
        return sorted(groups.values(), key=len, reverse=True)

    def communities(self, max_iterations: int = 30, seed: int = 42) -> Dict[str, int]:
        """
        Etiket yayılımı ile topluluk tespiti (deterministik).

        Returns:
            {node_id: topluluk_index}
        """
        self._counter["analyses"] += 1
        nodes = sorted(self._nodes.keys())
        if not nodes:
            return {}

        adjacency = {node_id: self._direct_neighbors(node_id, "both", None) for node_id in nodes}
        labels = {node_id: node_id for node_id in nodes}

        for _ in range(max_iterations):
            changed = False
            for node_id in nodes:
                neighbors = adjacency.get(node_id, [])
                if not neighbors:
                    continue
                counts: Dict[str, float] = defaultdict(float)
                for neighbor in neighbors:
                    counts[labels[neighbor]] += 1.0
                if not counts:
                    continue
                # En sık etiket; eşitlikte sözlük sırası ile determinist seçim
                best_label = max(sorted(counts.keys()), key=lambda label: counts[label])
                if labels[node_id] != best_label:
                    labels[node_id] = best_label
                    changed = True
            if not changed:
                break

        # Etiketleri 0..k aralığına sıkıştır
        unique = {label: index for index, label in enumerate(sorted(set(labels.values())))}
        return {node_id: unique[label] for node_id, label in labels.items()}

    def find_isolated(self, include_archived: bool = False) -> List[str]:
        """
        Yetim notlar: hiç giren veya çıkan bağlantısı olmayan düğümler.

        `related`/`similar` gibi türetilmiş kenarlar sayılmaz — yalnızca
        yazarın kurduğu `links` ilişkisi değerlendirilir.
        """
        isolated: List[str] = []
        for node_id, node in self._nodes.items():
            if node.archived and not include_archived:
                continue
            has_link = False
            for key in self._out.get(node_id, ()):  # type: ignore[arg-type]
                if key[2] == "links":
                    has_link = True
                    break
            if not has_link:
                for key in self._in.get(node_id, ()):  # type: ignore[arg-type]
                    if key[2] == "links":
                        has_link = True
                        break
            if not has_link:
                isolated.append(node_id)
        return isolated

    def find_broken_links(self) -> List[Tuple[str, str]]:
        """Hedefi var olmayan bağlantılar: [(kaynak, hedef)]."""
        return [
            (edge.src, edge.dst) for edge in self._edges.values()
            if edge.unresolved or edge.dst not in self._nodes
        ]

    def relation_counts(self) -> Dict[str, int]:
        """İlişki türüne göre kenar sayıları."""
        counts: Dict[str, int] = defaultdict(int)
        for edge in self._edges.values():
            counts[edge.relation] += 1
        return dict(counts)

    def detect_cycles(self, max_cycles: int = 50) -> List[List[str]]:
        """
        `supports`/`derived_from`/`part_of` kenarlarında döngü arar
        (kavramsal hiyerarşide döngü hatadır).

        Returns:
            Döngü yolları listesi
        """
        hierarchy_relations = {"supports", "derived_from", "part_of", "merged_into"}
        adjacency: Dict[str, List[str]] = defaultdict(list)
        for edge in self._edges.values():
            if edge.relation in hierarchy_relations:
                adjacency[edge.src].append(edge.dst)

        cycles: List[List[str]] = []
        state: Dict[str, int] = {}   # 0 = ziyaret edilmedi, 1 = yığında, 2 = bitti
        stack: List[str] = []

        def visit(node_id: str) -> None:
            if len(cycles) >= max_cycles:
                return
            state[node_id] = 1
            stack.append(node_id)
            for neighbor in adjacency.get(node_id, []):
                if state.get(neighbor, 0) == 1:
                    index = stack.index(neighbor) if neighbor in stack else 0
                    cycles.append(stack[index:] + [neighbor])
                elif state.get(neighbor, 0) == 0:
                    visit(neighbor)
            stack.pop()
            state[node_id] = 2

        for node_id in list(self._nodes.keys()):
            if state.get(node_id, 0) == 0:
                visit(node_id)
        return cycles

    def metrics(self, top_n: int = 10) -> GraphMetrics:
        """
        Graf geneli metrikleri hesaplar (PageRank dahil).

        Args:
            top_n: Kaç merkez düğüm raporlanacak
        """
        self._counter["analyses"] += 1
        metrics = GraphMetrics()
        metrics.node_count = len(self._nodes)
        metrics.edge_count = len(self._edges)

        n = metrics.node_count
        if n > 1:
            metrics.density = metrics.edge_count / (n * (n - 1))

        degrees = [
            len(self._out.get(node_id, ())) + len(self._in.get(node_id, ()))  # type: ignore[arg-type]
            for node_id in self._nodes
        ]
        if degrees:
            metrics.mean_degree = sum(degrees) / len(degrees)
            metrics.max_degree = max(degrees)

        components = self.components()
        metrics.component_count = len(components)
        metrics.largest_component = len(components[0]) if components else 0
        metrics.isolated_count = len(self.find_isolated())
        metrics.broken_links = len(self.find_broken_links())
        metrics.relation_counts = self.relation_counts()
        metrics.cycle_count = len(self.detect_cycles(max_cycles=20))

        pagerank = self.pagerank()
        ranked = sorted(pagerank.items(), key=lambda item: -item[1])
        metrics.top_hubs = [(node_id, score) for node_id, score in ranked[:top_n]]

        in_deg = self.in_degree()
        metrics.top_authorities = sorted(
            in_deg.items(), key=lambda item: -item[1]
        )[:top_n]

        communities = self.communities()
        metrics.community_count = len(set(communities.values())) if communities else 0

        tag_counts: Dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            for tag in node.tags:
                tag_counts[tag] += 1
        metrics.tag_clusters = dict(
            sorted(tag_counts.items(), key=lambda item: -item[1])[:30]
        )
        self._cache["metrics"] = metrics
        return metrics

    # ─── Öneri ve Analiz Yardımcıları ───────────────────────────────────

    def suggest_links(self, threshold: float = 0.6, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Kurulması yararlı olacak bağlantıları önerir.

        Ölçüt: ortak komşu sayısı + etiket örtüşmesi + başlık benzerliği,
        zaten bağlı olan çiftler hariç.
        """
        suggestions: List[Dict[str, Any]] = []
        nodes = list(self._nodes.values())
        existing: Set[Tuple[str, str]] = set()
        for edge in self._edges.values():
            existing.add((edge.src, edge.dst))
            existing.add((edge.dst, edge.src))

        by_tag: Dict[str, List[str]] = defaultdict(list)
        for node in nodes:
            for tag in node.tags:
                by_tag[tag].append(node.id)

        seen_pairs: Set[Tuple[str, str]] = set()
        for tag, members in by_tag.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = sorted((members[i], members[j]))
                    if (a, b) in seen_pairs or (a, b) in existing:
                        continue
                    seen_pairs.add((a, b))
                    if len(seen_pairs) > 4000:
                        break
                    score = self._structural_similarity(self._nodes[a], self._nodes[b])
                    common = len(self.common_neighbors(a, b))
                    total = score + 0.08 * common
                    if total >= threshold:
                        suggestions.append({
                            "a": a, "b": b,
                            "title_a": self.title_of(a), "title_b": self.title_of(b),
                            "score": round(min(1.0, total), 4),
                            "shared_tags": sorted(set(self._nodes[a].tags) & set(self._nodes[b].tags)),
                            "common_neighbors": common,
                        })
        suggestions.sort(key=lambda item: -item["score"])
        return suggestions[:limit]

    def hub_notes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """En merkezi notlar (PageRank) — bilgi haritasının ana düğümleri."""
        pagerank = self.pagerank()
        in_deg = self.in_degree()
        out_deg = self.out_degree()
        ranked = sorted(pagerank.items(), key=lambda item: -item[1])[:limit]
        return [
            {
                "id": node_id,
                "title": self.title_of(node_id),
                "pagerank": round(score, 5),
                "in_degree": in_deg.get(node_id, 0),
                "out_degree": out_deg.get(node_id, 0),
                "tags": self._nodes[node_id].tags if node_id in self._nodes else [],
                "confidence": self._nodes[node_id].confidence if node_id in self._nodes else 0.0,
            }
            for node_id, score in ranked
        ]

    def bridge_notes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Köprü notları (arasındalık) — farklı kümeleri birleştiren notlar."""
        centrality = self.betweenness_centrality()
        ranked = sorted(centrality.items(), key=lambda item: -item[1])[:limit]
        return [
            {"id": node_id, "title": self.title_of(node_id), "betweenness": round(score, 6)}
            for node_id, score in ranked
        ]

    def neighborhood_report(self, node_id: str, depth: int = 1) -> Dict[str, Any]:
        """Bir notun komşuluk raporu (bağlam kartı)."""
        if node_id not in self._nodes:
            return {"found": False, "id": node_id}
        node = self._nodes[node_id]
        related = self.neighbors(node_id, depth=depth)
        return {
            "found": True,
            "id": node_id,
            "title": node.title,
            "kind": node.kind,
            "tags": node.tags,
            "confidence": node.confidence,
            "outgoing": [self.title_of(edge.dst) for edge in self._edges.values()
                         if edge.src == node_id][:20],
            "incoming": [self.title_of(edge.src) for edge in self._edges.values()
                         if edge.dst == node_id][:20],
            "neighborhood_size": len(related),
            "neighbors": [self.title_of(nid) for nid in related[:30]],
            "pagerank": round(self.pagerank().get(node_id, 0.0), 6),
        }

    # ─── Dışa Aktarma ───────────────────────────────────────────────────

    def to_networkx(self) -> Any:
        """
        networkx DiGraph'e çevirir (kurulu değilse GraphError).

        Kenar öznitelikleri `relation`, `weight`, `reason`, `unresolved`.
        Düğüm öznitelikleri GraphNode alanlarıdır.
        """
        try:
            import networkx as nx  # type: ignore
        except ImportError as exc:
            raise GraphError("networkx kurulu değil — to_networkx kullanılamaz") from exc

        graph = nx.DiGraph()
        for node in self._nodes.values():
            graph.add_node(node.id, **node.to_dict())
        for edge in self._edges.values():
            graph.add_edge(
                edge.src, edge.dst,
                relation=edge.relation, weight=edge.weight,
                reason=edge.reason, unresolved=edge.unresolved,
            )
        return graph

    def to_dict(self) -> Dict[str, Any]:
        """Grafı JSON-serileştirilebilir sözlüğe çevirir."""
        return {
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges.values()],
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "relations": self.relation_counts(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Sözlükten graf kurar."""
        graph = cls()
        graph.initialize(None, None)
        for raw in data.get("nodes") or []:
            graph.add_node(GraphNode.from_dict(raw))
        for raw in data.get("edges") or []:
            edge = GraphEdge.from_dict(raw)
            try:
                graph.add_edge(edge.src, edge.dst, edge.relation, edge.weight,
                               edge.reason, create_missing=True)
            except GraphError:
                continue
        return graph

    def save(self, path: str) -> str:
        """Grafı JSON dosyasına kaydeder (atomik)."""
        import os
        from pathlib import Path as _Path
        from neurocore.knowledge.note import now_iso
        target = _Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"generated_at": now_iso(), **self.to_dict()}
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, target)
        logger.info(f"Graf kaydedildi: {target} ({len(self._nodes)} düğüm)")
        return str(target)

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """JSON dosyasından graf yükler."""
        from pathlib import Path as _Path
        data = json.loads(_Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_mermaid(self, focus: Optional[str] = None, max_nodes: int = 60,
                   direction: str = "TD") -> str:
        """
        Mermaid akış diyagramı üretir (Obsidian uyumlu).

        Args:
            focus: Yalnızca bu düğümün komşuluğunu çiz
            max_nodes: Maksimum düğüm sayısı
            direction: `TD`, `LR`, `BT`, `RL`
        """
        if focus and focus in self._nodes:
            selected = [focus] + self.neighbors(focus, depth=2)
        else:
            pagerank = self.pagerank()
            selected = [nid for nid, _ in sorted(pagerank.items(), key=lambda i: -i[1])]
        selected = selected[:max_nodes]
        selected_set = set(selected)

        lines = [f"graph {direction}"]
        for node_id in selected:
            node = self._nodes.get(node_id)
            if not node:
                continue
            label = _mermaid_safe(node.title or node_id)
            shape_open, shape_close = _mermaid_shape(node.kind)
            lines.append(f'    {_mermaid_id(node_id)}{shape_open}"{label}"{shape_close}')

        style_by_relation = {
            "links": "-->",
            "supports": "==>",
            "contradicts": "-.->",
            "derived_from": "-->",
            "part_of": "-->",
            "merged_into": "==>",
            "similar": "-.-",
            "related": "---",
        }
        for edge in self._edges.values():
            if edge.src not in selected_set or edge.dst not in selected_set:
                continue
            arrow = style_by_relation.get(edge.relation, "-->")
            label = f"|{edge.relation}|" if edge.relation not in ("links", "related") else ""
            lines.append(
                f"    {_mermaid_id(edge.src)} {arrow}{label} {_mermaid_id(edge.dst)}"
            )

        lines.append("    classDef lowConf fill:#3a2a2a,stroke:#a33")
        low_conf = [nid for nid in selected
                    if self._nodes[nid].confidence < 0.4]
        if low_conf:
            lines.append("    class " + ",".join(_mermaid_id(n) for n in low_conf) + " lowConf")
        return "\n".join(lines)

    def to_dot(self, max_nodes: int = 200) -> str:
        """Graphviz DOT biçimi (harici görselleştirme araçları için)."""
        pagerank = self.pagerank()
        selected = [nid for nid, _ in sorted(pagerank.items(), key=lambda i: -i[1])][:max_nodes]
        selected_set = set(selected)
        lines = ["digraph NeuroCore {", '  rankdir=LR;', '  node [shape=box, style=rounded];']
        for node_id in selected:
            node = self._nodes[node_id]
            label = (node.title or node_id).replace('"', "'")
            size = 0.6 + 2.0 * pagerank.get(node_id, 0.0) * max(1, len(selected))
            lines.append(f'  "{node_id}" [label="{label}", width={size:.2f}];')
        for edge in self._edges.values():
            if edge.src in selected_set and edge.dst in selected_set:
                lines.append(
                    f'  "{edge.src}" -> "{edge.dst}" '
                    f'[label="{edge.relation}", weight={edge.weight:.2f}];'
                )
        lines.append("}")
        return "\n".join(lines)

    def to_markdown(self, title: str = "Bilgi Grafı") -> str:
        """
        Graf raporunu Obsidian notu biçiminde üretir.

        `[[wiki]]` bağlantıları içerdiği için Obsidian graf görünümünde
        doğrudan gezilebilir.
        """
        metrics = self.metrics()
        lines = [
            f"# {title}",
            "",
            f"_{len(self._nodes)} not · {len(self._edges)} ilişki · "
            f"{metrics.component_count} kopuk parça · "
            f"{metrics.isolated_count} yetim not_",
            "",
            "## Merkez Notlar (PageRank)",
            "",
            "| # | Not | PageRank | Giren | Çıkan | Etiketler |",
            "|---|---|---|---|---|---|",
        ]
        for index, hub in enumerate(self.hub_notes(12), start=1):
            lines.append(
                f"| {index} | [[{hub['id']}\\|{hub['title']}]] | {hub['pagerank']:.5f} | "
                f"{hub['in_degree']} | {hub['out_degree']} | {', '.join(hub['tags'][:4])} |"
            )

        lines += ["", "## İlişki Dağılımı", ""]
        for relation, count in sorted(metrics.relation_counts.items(), key=lambda i: -i[1]):
            lines.append(f"- `{relation}`: {count}")

        if metrics.tag_clusters:
            lines += ["", "## Etiket Kümeleri", ""]
            for tag, count in list(metrics.tag_clusters.items())[:20]:
                lines.append(f"- `{tag}`: {count} not")

        isolated = self.find_isolated()
        if isolated:
            lines += ["", "## Yetim Notlar", ""]
            for node_id in isolated[:25]:
                lines.append(f"- [[{node_id}\\|{self.title_of(node_id)}]]")

        broken = self.find_broken_links()
        if broken:
            lines += ["", "## Kırık Bağlantılar", ""]
            for src, dst in broken[:25]:
                lines.append(f"- `{self.title_of(src)}` → `{dst}` (hedef yok)")

        lines += ["", "## Mermaid Görünümü", "", "```mermaid",
                  self.to_mermaid(max_nodes=40), "```", ""]
        return "\n".join(lines)

    # ─── İç Yardımcılar ─────────────────────────────────────────────────

    def _emit(self, event_type: str, data: Any) -> None:
        """Olay yayınla."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(event_type, data, source=self.name)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Olay yayınlanamadı ({event_type}): {exc}")

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __repr__(self) -> str:  # pragma: no cover
        return f"<KnowledgeGraph nodes={len(self._nodes)} edges={len(self._edges)}>"


# ─── Modül Yardımcıları ─────────────────────────────────────────────────────

_WIKILINK_RE = re.compile(r"\[\[([^\[\]|]+?)(?:\|[^\[\]]+?)?\]\]")


def _extract_wikilinks(text: str) -> List[Tuple[str, str]]:
    """Gövdedeki `[[hedef]]` bağlantılarını çıkarır (hafif, bağımlılıksız)."""
    if not text:
        return []
    return [((m.group(1) or "").strip(), (m.group(1) or "").strip())
            for m in _WIKILINK_RE.finditer(text)]


def _as_list(value: Any) -> List[str]:
    """Değeri string listesine çevirir."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Kosinüs benzerliği (bağımlılıksız)."""
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = norm_a = norm_b = 0.0
    for i in range(length):
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _mermaid_id(node_id: str) -> str:
    """Mermaid güvenli düğüm kimliği."""
    return "n" + re.sub(r"[^A-Za-z0-9]", "", node_id)[:24]


def _mermaid_safe(text: str) -> str:
    """Mermaid etiketi için metni temizler."""
    return re.sub(r'["\n\r<>]', " ", text)[:60].strip() or "?"


def _mermaid_shape(kind: str) -> Tuple[str, str]:
    """Not türüne göre Mermaid şekli: (açılış, kapanış)."""
    shapes = {
        "concept": ("[", "]"),
        "entity": ("((", "))"),
        "source": ("[/", "/]"),
        "project": ("[[", "]]"),
        "session": (">", "]"),
        "skill": ("{{", "}}"),
        "journal": ("[", "]"),
        "meta": ("((", "))"),
    }
    return shapes.get(kind, ("[", "]"))


__all__ = [
    "KnowledgeGraph",
    "GraphNode",
    "GraphEdge",
    "GraphMetrics",
    "GraphError",
    "RELATION_TYPES",
    "RELATION_WEIGHTS",
]
