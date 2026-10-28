"""
Bilgi katmanı testleri: not modeli, kasa, indeks, graf, sürümleme, cephe.

Bu testler sistemin **kalbini** doğrular: bilginin yazılması, bulunması,
ilişkilendirilmesi ve sürümlenmesi. Hepsi geçici dizinlerde çalışır.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.helpers import assert_json_safe, isolated_config

from neurocore.knowledge.note import (
    Note,
    cosine_similarity,
    extract_wikilinks,
    make_note,
    merge_notes,
    normalize_tags,
    slugify,
    split_frontmatter,
    strip_markdown,
    title_similarity,
)
from neurocore.knowledge.vault import KnowledgeVault
from neurocore.knowledge.index import KnowledgeIndex, blob_to_vector, vector_to_blob
from neurocore.knowledge.graph import KnowledgeGraph
from neurocore.knowledge.search import HybridSearch, QueryAnalyzer, ContextBuilder
from neurocore.knowledge.versioning import NoteVersioner
from neurocore.knowledge.layer import KnowledgeLayer


class TestNoteModel(unittest.TestCase):
    """Not veri modeli: frontmatter, bağlantılar, etiketler, serileştirme."""

    def test_frontmatter_roundtrip(self) -> None:
        """Not → markdown → not dönüşümü tüm alanları korur."""
        note = make_note(
            "GPU Termal Yönetimi",
            "## Özet\nGPU sıcaklığı tozdan etkilenir.\n\n## Detay\nFan hızı önemlidir.",
            tags=["GPU", "termal yönetim", "#donanim"],
            confidence=0.82,
            importance=0.6,
            sources=["os://nvidia-smi"],
        )
        markdown = note.to_markdown()
        self.assertTrue(markdown.startswith("---"), "frontmatter ile başlamalı")

        restored = Note.from_markdown(markdown)
        self.assertEqual(restored.title, note.title)
        self.assertEqual(restored.tags, note.tags)
        self.assertAlmostEqual(restored.confidence, note.confidence, places=3)
        self.assertEqual(restored.sources, note.sources)
        self.assertEqual(restored.kind, note.kind)
        self.assertEqual(restored.id, note.id)

    def test_tags_are_normalized(self) -> None:
        """Türkçe karakterler katlanır, boşluklar tireye döner, tekrar atılır."""
        tags = normalize_tags(["Yapay Zeka", "#GPU", "yapay zeka", "Termal Yönetim"])
        self.assertIn("yapay-zeka", tags)
        self.assertIn("gpu", tags)
        self.assertIn("termal-yonetim", tags)
        self.assertEqual(len(tags), len(set(tags)), "tekrar olmamalı")

    def test_wikilinks_extracted(self) -> None:
        """`[[hedef|etiket]]` biçimli bağlantılar çıkarılır."""
        body = "Bkz [[kn_abc12345]] ve [[Termal Macun|macun notu]]."
        links = extract_wikilinks(body)
        targets = [target for target, _label in links]
        self.assertIn("kn_abc12345", targets)
        self.assertIn("Termal Macun", targets)

    def test_inline_tags_not_confused_with_links(self) -> None:
        """Bağlantı içindeki `#` etiket sayılmaz."""
        note = Note.from_markdown(
            "---\nid: kn_11111111\ntitle: T\n---\n"
            "Metin [[kn_abc12345|#başlık]] ve #gercek-etiket"
        )
        self.assertIn("gercek-etiket", note.tags)
        self.assertNotIn("baslik", note.tags)

    def test_missing_frontmatter_still_loads(self) -> None:
        """Frontmatter yoksa not yine de okunur (kasa asla kırılmaz)."""
        note = Note.from_markdown("# Başlıksız Not\n\nGövde metni.")
        self.assertTrue(note.id)
        self.assertEqual(note.title, "Başlıksız Not")
        self.assertIn("Gövde", note.body)

    def test_slugify_turkish(self) -> None:
        """Slug Türkçe karakterleri ASCII'ye katlar."""
        self.assertEqual(slugify("GPU Termal Yönetimi — Nedir?"), "gpu-termal-yonetimi-nedir")

    def test_merge_notes_combines_content(self) -> None:
        """İki not birleştiğinde etiketler, kaynaklar ve gövde toplanır."""
        first = make_note("Termal Yönetim", "Birinci gövde içeriği.", tags=["termal"])
        second = make_note("Termal Bakım", "İkinci gövde içeriği.", tags=["bakim"],
                           sources=["file://x"])
        merged = merge_notes(first, second)
        self.assertIn("bakim", merged.tags)
        self.assertIn("İkinci gövde", merged.body)
        self.assertIn(second.id, merged.links)
        self.assertEqual(merged.extra.get("merged_from")[0]["id"], second.id)

    def test_strip_markdown_removes_formatting(self) -> None:
        """Arama için markdown işaretleri temizlenir."""
        text = strip_markdown("## Başlık\n**kalın** ve [[link|görünen]] metin `kod`")
        self.assertNotIn("##", text)
        self.assertNotIn("**", text)
        self.assertIn("görünen", text)

    def test_title_similarity_ordering(self) -> None:
        """Benzer başlıklar yüksek, ilgisizler düşük puan alır."""
        high = title_similarity("GPU Termal Yönetimi", "GPU Termal Yönetim")
        low = title_similarity("GPU Termal Yönetimi", "Yapay Sinir Ağları")
        self.assertGreater(high, low)
        self.assertGreater(high, 0.7)

    def test_cosine_similarity_bounds(self) -> None:
        """Kosinüs benzerliği sınırları doğru."""
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0, places=6)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0, places=6)
        self.assertEqual(cosine_similarity([], [1, 0]), 0.0)


class TestVault(unittest.TestCase):
    """Kasa: dizin düzeni, yazma/okuma, bütünlük, ekler."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_vault_")
        self.vault = KnowledgeVault(self.root)
        self.vault.initialize(None, None)

    def tearDown(self) -> None:
        self.vault.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_layout_created(self) -> None:
        """Kasa dizin düzeni ve README oluşturulur."""
        for directory in ("00_meta", "10_concepts", "_index", "_attachments", "90_archive"):
            self.assertTrue((Path(self.root) / directory).is_dir(), f"{directory} yok")
        self.assertTrue((Path(self.root) / "README.md").exists())

    def test_write_read_roundtrip(self) -> None:
        """Not yazılır, kimlikle okunur, gövde korunur."""
        note = self.vault.create_note("Test Notu", "## Özet\nİçerik burada.",
                                      tags=["test"], confidence=0.7)
        self.assertTrue(note.path.endswith(".md"))
        restored = self.vault.read_note(note.id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.title, "Test Notu")
        self.assertIn("İçerik burada", restored.body)

    def test_category_routing(self) -> None:
        """Not türüne göre doğru dizine yazılır."""
        note = self.vault.create_note("Yetenek", "İçerik", kind="skill")
        self.assertTrue(note.path.startswith("60_skills/"), note.path)

    def test_title_change_moves_file(self) -> None:
        """Başlık değişince dosya adı güncellenir, eski dosya silinir."""
        note = self.vault.create_note("Eski Başlık", "İçerik")
        old_path = Path(self.root) / note.path
        updated = self.vault.update_note(note.id, title="Yeni Başlık")
        new_path = Path(self.root) / updated.path
        self.assertTrue(new_path.exists())
        self.assertFalse(old_path.exists())
        self.assertIn("yeni-baslik", updated.path)

    def test_search_by_title_and_content(self) -> None:
        """Başlık ve içerik aramaları doğru sonuç verir."""
        self.vault.create_note("GPU Termal", "Fan hızı önemlidir.")
        self.vault.create_note("CPU Yükü", "Çekirdek kullanımı izlenir.")
        title_hits = [note.title for note, _ in self.vault.search_by_title("termal")]
        self.assertIn("GPU Termal", title_hits)
        content_hits = [note.title for note, _ in self.vault.search_by_content("çekirdek")]
        self.assertIn("CPU Yükü", content_hits)

    def test_archive_moves_note(self) -> None:
        """Arşivleme notu `90_archive` altına taşır."""
        note = self.vault.create_note("Arşivlenecek", "İçerik")
        target = self.vault.archive_note(note.id, reason="test")
        self.assertIsNotNone(target)
        self.assertTrue(target.startswith("90_archive/"))
        archived = self.vault.read_note(note.id)
        self.assertTrue(archived.extra.get("archived"))

    def test_integrity_check_detects_missing_frontmatter(self) -> None:
        """Frontmatter'sız dosya bütünlük denetiminde bildirilir."""
        broken = Path(self.root) / "10_concepts" / "bozuk.md"
        broken.write_text("# Frontmatter yok\n\nGövde.", encoding="utf-8")
        result = self.vault.integrity_check(fix=False)
        kinds = {issue["kind"] for issue in result["issues"]}
        self.assertIn("missing_frontmatter", kinds)

    def test_integrity_fix_repairs(self) -> None:
        """`fix=True` bozuk frontmatter'ı onarır."""
        broken = Path(self.root) / "10_concepts" / "onarilacak.md"
        broken.write_text("# Onarılacak\n\nGövde metni yeterince uzun olsun.", encoding="utf-8")
        result = self.vault.integrity_check(fix=True)
        self.assertGreater(result["fixed"], 0)
        text = broken.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---"), "frontmatter eklenmeli")

    def test_path_escape_rejected(self) -> None:
        """Kasa dışına çıkan yollar reddedilir."""
        from neurocore.knowledge.vault import VaultError

        with self.assertRaises(VaultError):
            self.vault._resolve_within_vault("../../gizli.txt")

    def test_attachment_storage(self) -> None:
        """Ekler kasaya yazılır ve geri okunur."""
        data = b"\x89PNG\r\n\x1a\n" + b"0" * 64
        path = self.vault.save_attachment(data, "test görsel.png", subdir="png",
                                          note_id="kn_abc12345")
        self.assertIn("_attachments/png/", path.replace("\\", "/"))
        self.assertEqual(self.vault.read_attachment(path), data)

    def test_duplicate_detection(self) -> None:
        """Neredeyse aynı notlar tekrar olarak bulunur."""
        body = "GPU sıcaklığı 80 dereceyi aşarsa fan hızı artırılmalıdır. " * 5
        self.vault.create_note("Termal Yönetim", body, tags=["termal"])
        self.vault.create_note("Termal Yönetim", body, tags=["termal"])
        duplicates = self.vault.find_duplicates(threshold=0.8)
        self.assertTrue(duplicates, "tekrar bulunmalı")
        self.assertGreaterEqual(duplicates[0]["score"], 0.8)

    def test_vault_stats(self) -> None:
        """Kasa istatistikleri doğru sayar."""
        self.vault.create_note("A", "içerik bir", tags=["x"])
        self.vault.create_note("B", "içerik iki", tags=["x", "y"])
        stats = self.vault.vault_stats()
        self.assertGreaterEqual(stats.total_notes, 3)  # README dahil
        self.assertIn("x", stats.tag_histogram)
        assert_json_safe(self, stats.to_dict(), "vault_stats")


class TestIndex(unittest.TestCase):
    """SQLite indeksi: FTS araması, vektörler, kenarlar, güncelleme."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_index_")
        self.vault = KnowledgeVault(self.root)
        self.vault.initialize(None, None)
        self.index = KnowledgeIndex(str(Path(self.root) / "_index" / "test.db"))
        self.index.initialize(None, None)
        for title, body, tags in (
            ("GPU Termal Yönetimi", "GPU sıcaklığı 80 dereceyi aşarsa fan hızı artırılır.", ["gpu", "termal"]),
            ("CPU Sıcaklık İzleme", "CPU sıcaklığı 90 dereceyi aşmamalıdır.", ["cpu", "termal"]),
            ("RAM Optimizasyonu", "Boş RAM miktarına göre önbellek ayarlanır.", ["ram"]),
        ):
            self.vault.create_note(title, body, tags=tags, confidence=0.7)
        self.index.rebuild(self.vault)

    def tearDown(self) -> None:
        self.index.close()
        self.vault.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_fts5_available(self) -> None:
        """Bu ortamda FTS5 kullanılabilir olmalı (yoksa uyarı)."""
        # FTS5 yoksa sistem LIKE yedeğine düşer; test bunu tolere eder.
        self.assertIsInstance(self.index.has_fts5, bool)

    def test_turkish_diacritic_search(self) -> None:
        """Türkçe diakritikli ve diakritiksiz sorgular aynı sonucu verir."""
        with_diacritics = self.index.search_fts("sıcaklık", limit=5)
        without = self.index.search_fts("sicaklik", limit=5)
        self.assertTrue(with_diacritics, "diakritikli sorgu sonuç vermeli")
        self.assertTrue(without, "diakritiksiz sorgu sonuç vermeli")
        self.assertEqual(
            {hit["title"] for hit in with_diacritics},
            {hit["title"] for hit in without},
        )

    def test_multiword_query_recall(self) -> None:
        """Çok kelimeli sorgu, kelimelerin dağıldığı notları da bulur."""
        hits = self.index.search_fts("GPU termal sıcaklık", limit=5)
        self.assertTrue(hits)
        self.assertIn("GPU Termal Yönetimi", [hit["title"] for hit in hits])

    def test_irrelevant_query_returns_nothing(self) -> None:
        """Alakasız sorgu boş döner."""
        hits = self.index.search_fts("kuantum kromodinamik plazma", limit=5)
        self.assertEqual(hits, [])

    def test_update_reflected_in_search(self) -> None:
        """Güncellenen not yeni içeriğiyle bulunur."""
        note = self.vault.read_note(
            [n.id for n in self.vault.iter_notes() if n.title == "RAM Optimizasyonu"][0]
        )
        note.body += "\n\nSanal bellek kullanımı da izlenmelidir."
        self.vault.save_note(note)
        self.index.upsert_note(note)
        hits = self.index.search_fts("sanal bellek", limit=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["title"], "RAM Optimizasyonu")

    def test_delete_removes_from_search(self) -> None:
        """Silinen not aramada görünmez."""
        note_id = [n.id for n in self.vault.iter_notes() if n.title == "RAM Optimizasyonu"][0]
        self.index.delete_note(note_id)
        hits = self.index.search_fts("önbellek önbellek", limit=5)
        self.assertNotIn(note_id, [hit["id"] for hit in hits])

    def test_filters(self) -> None:
        """Tür ve etiket filtreleri çalışır."""
        by_tag = self.index.search_fts("termal", limit=10, tags=["cpu"])
        self.assertTrue(all("cpu" in hit["tags"] for hit in by_tag))
        excluded = self.index.search_fts("termal", limit=10, exclude_kinds=["concept"])
        self.assertEqual(excluded, [], "tüm notlar concept → hepsi dışlanmalı")

    def test_vector_storage_roundtrip(self) -> None:
        """Gömme vektörleri BLOB olarak saklanır ve geri okunur."""
        vector = [0.1, -0.2, 0.3, 0.4]
        blob = vector_to_blob(vector)
        restored = blob_to_vector(blob)
        self.assertEqual(len(restored), 4)
        for expected, actual in zip(vector, restored):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_similar_notes(self) -> None:
        """Vektör benzerliği en yakın notu bulur."""
        notes = list(self.vault.iter_notes())
        target = next(note for note in notes if note.title.startswith("GPU"))
        self.index.store_embedding(target.id, [1.0] + [0.0] * 31, model="test")
        for note in notes:
            if note.id != target.id:
                self.index.store_embedding(note.id, [0.0] * 31 + [1.0], model="test")
        hits = self.index.similar_notes([1.0] + [0.0] * 31, limit=1)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["id"], target.id)

    def test_edge_management(self) -> None:
        """Kenarlar eklenir, sorgulanır ve silinir."""
        notes = list(self.vault.iter_notes())
        src, dst = notes[0].id, notes[1].id
        self.assertTrue(self.index.add_edge(src, dst, "supports", 0.9, "test"))
        edges = self.index.edges(relation="supports")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["src"], src)
        self.assertEqual(self.index.remove_edge(src, dst, "supports"), 1)
        self.assertEqual(self.index.edges(relation="supports"), [])

    def test_stats_json_safe(self) -> None:
        """İndeks istatistikleri JSON-güvenli."""
        assert_json_safe(self, self.index.get_stats(), "index_stats")


class TestGraph(unittest.TestCase):
    """Bilgi grafı: düğümler, kenarlar, merkezilik, topluluklar."""

    def setUp(self) -> None:
        self.graph = KnowledgeGraph()
        self.graph.initialize(None, None)
        notes = [
            {"id": "kn_a", "title": "GPU Termal", "tags": ["gpu", "termal"],
             "body": "Bkz [[CPU Sıcaklık]].", "confidence": 0.8, "links": []},
            {"id": "kn_b", "title": "CPU Sıcaklık", "tags": ["cpu", "termal"],
             "body": "Bkz [[GPU Termal]].", "confidence": 0.7, "links": []},
            {"id": "kn_c", "title": "Termal Macun", "tags": ["termal"],
             "body": "Bağımsız not.", "confidence": 0.6, "links": ["kn_a"]},
            {"id": "kn_d", "title": "Yapay Sinir Ağı", "tags": ["yapay-zeka"],
             "body": "İzole not.", "confidence": 0.5, "links": []},
        ]
        self.graph.rebuild(notes)

    def tearDown(self) -> None:
        self.graph.shutdown()

    def test_nodes_and_edges(self) -> None:
        """Gövdedeki bağlantılar kenara dönüşür."""
        self.assertEqual(self.graph.node_count, 4)
        self.assertGreaterEqual(self.graph.edge_count, 3)

    def test_pagerank_identifies_hub(self) -> None:
        """En çok bağlanan not en yüksek PageRank'ı alır."""
        ranks = self.graph.pagerank()
        self.assertAlmostEqual(sum(ranks.values()), 1.0, places=4)
        self.assertEqual(max(ranks, key=ranks.get), "kn_a")

    def test_neighbors_traversal(self) -> None:
        """Komşu genişletme doğru çalışır."""
        neighbors = self.graph.neighbors("kn_c", depth=1)
        self.assertIn("kn_a", neighbors)

    def test_shortest_path(self) -> None:
        """İki not arasındaki yol bulunur."""
        path = self.graph.shortest_path("kn_c", "kn_b")
        self.assertEqual(path[0], "kn_c")
        self.assertEqual(path[-1], "kn_b")
        self.assertIn("kn_a", path)

    def test_isolated_detection(self) -> None:
        """Bağlantısız not yetim olarak bulunur."""
        isolated = self.graph.find_isolated()
        self.assertIn("kn_d", isolated)

    def test_components_and_communities(self) -> None:
        """Bileşen ve topluluk sayıları tutarlı."""
        components = self.graph.components()
        self.assertGreaterEqual(len(components), 2)
        communities = self.graph.communities()
        self.assertEqual(len(communities), self.graph.node_count)

    def test_metrics_json_safe(self) -> None:
        """Graf metrikleri JSON-güvenli."""
        assert_json_safe(self, self.graph.metrics().to_dict(), "graph_metrics")

    def test_self_loop_rejected(self) -> None:
        """Öz-döngü reddedilir."""
        from neurocore.knowledge.graph import GraphError

        with self.assertRaises(GraphError):
            self.graph.add_edge("kn_a", "kn_a", "links")

    def test_mermaid_export(self) -> None:
        """Mermaid çıktısı geçerli biçimde üretilir."""
        mermaid = self.graph.to_mermaid(max_nodes=10)
        self.assertTrue(mermaid.startswith("graph TD"))
        # Düğüm kimlikleri Mermaid-güvenli biçimde sadeleştirilir (kn_a → kna)
        self.assertIn("kna", mermaid)
        self.assertIn("-->", mermaid)
        self.assertIn("GPU Termal", mermaid)


class TestSearch(unittest.TestCase):
    """Hibrit arama: sorgu analizi, füzyon, bağlam üretimi."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_search_")
        self.config = isolated_config(self.root)
        self.layer = KnowledgeLayer()
        self.layer.initialize(self.config, None)
        for title, body, tags in (
            ("GPU Termal Yönetimi", "GPU sıcaklığı 80 dereceyi aşarsa fan hızı artırılır.", ["gpu", "termal"]),
            ("Termal Macun Yenileme", "Termal macun iki yılda bir yenilenmelidir.", ["termal", "bakim"]),
            ("Yapay Sinir Ağı", "Gradyan inişi ile eğitilir.", ["yapay-zeka"]),
        ):
            self.layer.write(title, body, tags=tags, confidence=0.75)

    def tearDown(self) -> None:
        self.layer.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_query_analyzer(self) -> None:
        """Sorgu analizi anahtar kelime ve niyet çıkarır."""
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("GPU sıcaklığı neden yükselir?")
        self.assertIn("gpu", analysis.all_words)
        self.assertTrue(analysis.is_question)
        self.assertIn("causal", analysis.intents)

    def test_hybrid_search_finds_relevant(self) -> None:
        """Hibrit arama ilgili notu üstte döndürür."""
        results = self.layer.search("GPU sıcaklığı neden yükselir?", limit=3)
        self.assertTrue(results)
        self.assertEqual(results[0].title, "GPU Termal Yönetimi")

    def test_search_irrelevant_empty(self) -> None:
        """Alakasız sorgu boş döner."""
        results = self.layer.search("kuantum kromodinamik", limit=3)
        self.assertEqual(results, [])

    def test_context_builder_budget(self) -> None:
        """Bağlam üretici karakter bütçesini aşmaz."""
        results = self.layer.search("termal", limit=5, include_body=True)
        builder = ContextBuilder(max_chars=200)
        context = builder.build(results, query="termal")
        self.assertLessEqual(len(context["text"]), 400,
                             "bütçe aşılmamalı (kaynak listesi payı dahil)")
        self.assertIn("confidence", context)

    def test_answer_material_excludes_generated(self) -> None:
        """`qa`/`meta` türündeki notlar yanıt bağlamına girmez."""
        self.layer.write("Sentetik Yanıt", "Bu not yanıt bağlamına girmemeli.",
                         kind="qa", confidence=0.9)
        material = self.layer.answer_material("sentetik yanıt bağlam", limit=5)
        titles = [result["title"] for result in material["results"]]
        self.assertNotIn("Sentetik Yanıt", titles)

    def test_related_notes(self) -> None:
        """İlişkili not önerisi çalışır."""
        gpu_note = [n for n in self.layer.vault.iter_notes()
                    if n.title.startswith("GPU")][0]
        related = self.layer.related(gpu_note.id, limit=3)
        self.assertIsInstance(related, list)


class TestVersioning(unittest.TestCase):
    """Sürümleme: anlık görüntü, geçmiş, fark, geri alma, budama."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_version_")
        self.versioner = NoteVersioner()
        self.versioner.initialize(None, None, root=str(Path(self.root) / "versions"))

    def tearDown(self) -> None:
        self.versioner.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_snapshot_increments_version(self) -> None:
        """Her değişiklik yeni sürüm oluşturur."""
        note = make_note("Test", "birinci gövde")
        first = self.versioner.snapshot(note, reason="ilk")
        self.assertEqual(first.version, 1)
        note.body += " ikinci"
        second = self.versioner.snapshot(note, reason="ikinci")
        self.assertEqual(second.version, 2)
        self.assertEqual(len(self.versioner.history(note.id)), 2)

    def test_unchanged_content_skipped(self) -> None:
        """Değişmeyen içerik için sürüm yazılmaz."""
        note = make_note("Test", "gövde")
        self.versioner.snapshot(note, reason="ilk")
        self.assertIsNone(self.versioner.snapshot(note, reason="ayni"))

    def test_restore_returns_old_content(self) -> None:
        """Geri alma eski gövdeyi getirir ve yeni sürüm oluşturur."""
        note = make_note("Test", "orijinal gövde")
        self.versioner.snapshot(note, reason="ilk")
        note.body = "değiştirilmiş gövde"
        self.versioner.snapshot(note, reason="ikinci")

        restored = self.versioner.restore(note.id, 1)
        self.assertIsNotNone(restored)
        self.assertIn("orijinal", restored.body)
        self.assertEqual(len(self.versioner.history(note.id)), 3)

    def test_diff_reports_changes(self) -> None:
        """Sürüm farkı değişen alanları bildirir."""
        note = make_note("Test", "kısa")
        self.versioner.snapshot(note, reason="ilk")
        note.body = "çok daha uzun bir gövde metni"
        note.confidence = 0.9
        self.versioner.snapshot(note, reason="ikinci")

        diff = self.versioner.diff(note.id, 1)
        self.assertIsNotNone(diff)
        self.assertIn("body", diff.changed_fields)
        self.assertNotEqual(diff.words_delta, 0)

    def test_prune_respects_pinned(self) -> None:
        """Budama sabitlenmiş sürümü korur."""
        note = make_note("Test", "v1")
        self.versioner.snapshot(note, reason="v1")
        self.versioner.pin(note.id, 1, True)
        for index in range(2, 8):
            note.body = f"gövde {index}"
            self.versioner.snapshot(note, reason=f"v{index}")

        self.versioner.prune(note.id, max_versions=3)
        versions = [record.version for record in self.versioner.history(note.id)]
        self.assertIn(1, versions, "sabitlenmiş sürüm korunmalı")
        self.assertLessEqual(len(versions), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
