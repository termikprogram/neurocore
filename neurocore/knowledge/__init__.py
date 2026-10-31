"""
NeuroCore Bilgi Katmanı (L2)
=============================
Obsidian-tarzı bilgi motoru: kasa, indeks, graf, arama, sürümleme.

Alt modüller:
    note        — Not veri modeli, frontmatter, bağlantılar
    vault       — Kasa dizin düzeni ve dosya G/Ç
    index       — SQLite + FTS5 + vektör indeksi
    graph       — Bilgi grafı, merkezilik, topluluklar
    search      — Hibrit arama (BM25 + vektör + graf) ve bağlam üretimi
    versioning  — Sürüm anlık görüntüleri ve geri alma
    layer       — Katman cephesi (dış API)
"""

from neurocore.knowledge.note import (
    NOTE_CATEGORIES,
    NOTE_KINDS,
    Note,
    cosine_similarity,
    make_note,
    merge_notes,
    new_note_id,
    normalize_tag,
    normalize_tags,
    note_diff,
    parse_timestamp,
    slugify,
    strip_markdown,
    title_similarity,
)
from neurocore.knowledge.vault import (
    KnowledgeVault,
    VaultError,
    VaultNotInitialized,
    VaultStats,
)
from neurocore.knowledge.index import (
    DatabaseUnavailable,
    KnowledgeIndex,
    blob_to_vector,
    fold_for_search,
    vector_to_blob,
)
from neurocore.knowledge.graph import (
    RELATION_TYPES,
    RELATION_WEIGHTS,
    GraphEdge,
    GraphError,
    GraphMetrics,
    GraphNode,
    KnowledgeGraph,
)
from neurocore.knowledge.search import (
    ContextBuilder,
    HybridSearch,
    QueryAnalysis,
    QueryAnalyzer,
    RerankWeights,
    SearchError,
    SearchResult,
)
from neurocore.knowledge.versioning import (
    DiffSummary,
    NoteVersioner,
    VersionRecord,
    VersioningError,
)
from neurocore.knowledge.layer import IngestResult, KnowledgeLayer, MaintenanceReport

__all__ = [
    # Cephe
    "KnowledgeLayer",
    "IngestResult",
    "MaintenanceReport",
    # Not modeli
    "Note",
    "NOTE_KINDS",
    "NOTE_CATEGORIES",
    "make_note",
    "merge_notes",
    "note_diff",
    "new_note_id",
    "normalize_tag",
    "normalize_tags",
    "slugify",
    "strip_markdown",
    "title_similarity",
    "cosine_similarity",
    "parse_timestamp",
    # Kasa
    "KnowledgeVault",
    "VaultError",
    "VaultNotInitialized",
    "VaultStats",
    # İndeks
    "KnowledgeIndex",
    "DatabaseUnavailable",
    "vector_to_blob",
    "blob_to_vector",
    "fold_for_search",
    # Graf
    "KnowledgeGraph",
    "GraphNode",
    "GraphEdge",
    "GraphMetrics",
    "GraphError",
    "RELATION_TYPES",
    "RELATION_WEIGHTS",
    # Arama
    "HybridSearch",
    "SearchResult",
    "SearchError",
    "QueryAnalyzer",
    "QueryAnalysis",
    "RerankWeights",
    "ContextBuilder",
    # Sürümleme
    "NoteVersioner",
    "VersionRecord",
    "DiffSummary",
    "VersioningError",
]
