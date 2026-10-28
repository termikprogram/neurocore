"""
Nöral katman testleri: tokenizer, dikkat, MoE, bellek, gömme, büyüme, eğitim.

Bu testler modelin **matematiksel doğruluğunu** ve **büyüme nötrlüğünü**
doğrular — sistemin "kendini büyütebilme" iddiasının temeli budur.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.helpers import assert_json_safe, assert_vector, isolated_config

from neurocore.neural.tokenizer import BPETokenizer, SimpleTokenizer
from neurocore.neural.embedding import (
    EmbeddingEngine,
    cosine_similarity,
    hash_embed,
    l2_normalize,
    tokenize,
)
from neurocore.neural.memory import MemoryBank
from neurocore.neural.growth import GrowthAction, GrowthController, GrowthMetrics
from neurocore.neural.transformer import TransformerConfig, config_from_system

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


class TestTokenizer(unittest.TestCase):
    """Byte-level BPE tokenizer: eğitim, kodlama, kalıcılık."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = [
            "Yapay sinir ağları gradyan inişi ile eğitilir.",
            "GPU sıcaklığı termal macun ile düşürülür.",
            "Bilgi kasası Obsidian uyumlu markdown dosyaları saklar.",
            "Türkçe karakterler: ığşçöüİĞŞÇÖÜ",
            "Bellek bankası anahtar-değer çiftleri tutar.",
        ] * 20
        cls.tokenizer = BPETokenizer()
        cls.tokenizer.initialize(None, None)
        cls.tokenizer.train(cls.corpus, vocab_size=600)

    def test_vocab_grew(self) -> None:
        """Eğitim sözlüğü büyütür (bayt düzeyinden fazlası)."""
        self.assertGreater(self.tokenizer.vocab_size, 300)
        self.assertGreater(len(self.tokenizer.merges), 20)

    def test_roundtrip_preserves_text(self) -> None:
        """Kodlama → çözme metni birebir korur (Türkçe dahil)."""
        for text in ("Yapay sinir ağları", "GPU sıcaklığı ığşçöü", "Merhaba dünya! 123", ""):
            with self.subTest(text=text):
                ids = self.tokenizer.encode(text)
                self.assertEqual(self.tokenizer.decode(ids), text)

    def test_special_tokens(self) -> None:
        """Özel token'lar tek id olarak kodlanır."""
        ids = self.tokenizer.encode("<|user|>Merhaba<|assistant|>")
        self.assertEqual(ids[0], self.tokenizer.vocab["<|user|>"])
        user_token = "<|user|>"
        self.assertIn(self.tokenizer.token_id(user_token), ids)

    def test_never_produces_unknown(self) -> None:
        """Byte-level BPE her metni kodlar — `<unk>` üretmez."""
        ids = self.tokenizer.encode("日本語テキスト żółć")
        unk = self.tokenizer.unk_id
        # Özel token olmadığı için unk görünmemeli (bayt yedeği devrede)
        self.assertNotIn(unk, ids)
        self.assertEqual(self.tokenizer.decode(ids), "日本語テキスト żółć")

    def test_truncation(self) -> None:
        """`max_length` sınırı uygulanır."""
        ids = self.tokenizer.encode("uzun bir metin " * 50, max_length=10)
        self.assertEqual(len(ids), 10)

    def test_batch_padding(self) -> None:
        """Toplu kodlama eşit uzunlukta diziler üretir."""
        batch = self.tokenizer.encode_batch(["kısa", "çok daha uzun bir metin"], padding=True)
        lengths = {len(row) for row in batch["input_ids"]}
        self.assertEqual(len(lengths), 1, "tüm diziler aynı uzunlukta olmalı")

    def test_save_load_roundtrip(self) -> None:
        """Kaydedilen tokenizer aynı kodlamayı üretir."""
        root = tempfile.mkdtemp(prefix="nc_tok_")
        try:
            path = self.tokenizer.save(str(Path(root) / "tokenizer.json"))
            restored = BPETokenizer.load(path)
            self.assertEqual(restored.vocab_size, self.tokenizer.vocab_size)
            text = "Yapay sinir ağları"
            self.assertEqual(restored.encode(text), self.tokenizer.encode(text))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_simple_tokenizer_interface(self) -> None:
        """Basit tokenizer aynı arayüzü sunar."""
        tokenizer = SimpleTokenizer()
        tokenizer.initialize()
        tokenizer.train(self.corpus, vocab_size=300)
        ids = tokenizer.encode("Yapay sinir ağları")
        self.assertTrue(ids)
        self.assertIn("Yapay", tokenizer.decode(ids))
        self.assertTrue(tokenizer.get_stats())


class TestEmbedding(unittest.TestCase):
    """Gömme motoru: hash stratejisi, determinizm, benzerlik, IDF."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = EmbeddingEngine()
        cls.engine.initialize(None, None)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.shutdown()

    def test_hash_embed_deterministic(self) -> None:
        """Aynı metin her zaman aynı vektörü verir."""
        first = hash_embed("merhaba dünya", dim=256)
        second = hash_embed("merhaba dünya", dim=256)
        self.assertEqual(first, second)

    def test_hash_embed_normalized(self) -> None:
        """Hash gömme L2-normalize edilir."""
        assert_vector(self, hash_embed("test metni", dim=128), dim=128)

    def test_empty_text_zero_vector(self) -> None:
        """Boş metin sıfır vektör verir (hata değil)."""
        vector = hash_embed("", dim=64)
        self.assertEqual(len(vector), 64)
        self.assertTrue(all(value == 0.0 for value in vector))

    def test_turkish_stemming_improves_similarity(self) -> None:
        """Ek kırpma, çekimli biçimlerin benzerliğini artırır."""
        similar = cosine_similarity(
            hash_embed("sıcaklık", dim=512), hash_embed("sıcaklığı", dim=512)
        )
        unrelated = cosine_similarity(
            hash_embed("sıcaklık", dim=512), hash_embed("uzay gemisi", dim=512)
        )
        self.assertGreater(similar, unrelated)
        # Kısa kelimelerde hash gömme gürültülüdür; beklenen alt sınır mütevazı
        self.assertGreater(similar, 0.15)

    def test_similarity_self_is_one(self) -> None:
        """Metnin kendisiyle benzerliği 1'dir."""
        vector = hash_embed("GPU termal yönetimi", dim=384)
        self.assertAlmostEqual(cosine_similarity(vector, vector), 1.0, places=6)

    def test_engine_embed_dim(self) -> None:
        """Motor doğru boyutta vektör üretir."""
        assert_vector(self, self.engine.embed("test"), dim=self.engine.dim)

    def test_cache_hit(self) -> None:
        """Önbellek aynı metni yeniden hesaplamaz."""
        self.engine.clear_cache()
        self.engine.embed("önbellek testi")
        before = self.engine.get_stats()["cache"]["hits"]
        self.engine.embed("önbellek testi")
        after = self.engine.get_stats()["cache"]["hits"]
        self.assertGreater(after, before)

    def test_fit_idf(self) -> None:
        """IDF öğrenimi terim sözlüğü üretir."""
        count = self.engine.fit_idf([
            "GPU sıcaklık yönetimi önemlidir",
            "CPU sıcaklık izleme gereklidir",
            "termal macun yenileme zamanı",
        ])
        self.assertGreater(count, 5)
        self.assertTrue(self.engine.keywords("GPU sıcaklık yönetimi"))

    def test_deduplicate(self) -> None:
        """Neredeyse aynı metinler tekrar olarak işaretlenir."""
        texts = ["GPU sıcaklığı yükseliyor", "GPU sıcaklığı yükseliyor", "Tamamen farklı konu"]
        duplicates = self.engine.deduplicate(texts, threshold=0.95)
        self.assertIn(1, duplicates)

    def test_stats_json_safe(self) -> None:
        """Gömme istatistikleri JSON-güvenli."""
        assert_json_safe(self, self.engine.get_stats(), "embedding_stats")


class TestMemoryBank(unittest.TestCase):
    """Bellek bankası: yazma politikası, geri çağırma, kalıcılık."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_memory_")
        self.embedding = EmbeddingEngine()
        self.embedding.initialize(None, None)
        self.bank = MemoryBank()
        self.bank.initialize(None, None, store_dir=self.root)
        self.bank.attach_embedder(self.embedding.embed)

    def tearDown(self) -> None:
        self.bank.shutdown()
        self.embedding.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_remember_and_recall(self) -> None:
        """Yazılan bilgi geri çağrılır."""
        record_id = self.bank.remember(
            "GPU sıcaklığı 80 dereceyi aşarsa fan hızı artırılmalıdır.",
            source="vault:kn_1", tags=["gpu"], importance=0.8,
        )
        self.assertIsNotNone(record_id)
        results = self.bank.recall("GPU sıcaklık", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0].record.source, "vault:kn_1")

    def test_duplicate_rejected(self) -> None:
        """Aynı bilgi iki kez yazılmaz."""
        text = "Termal macun iki yılda bir yenilenmelidir."
        self.assertIsNotNone(self.bank.remember(text, source="a", importance=0.7))
        self.assertIsNone(self.bank.remember(text, source="b", importance=0.9))

    def test_low_importance_rejected(self) -> None:
        """Önemsiz bilgi yazılmaz."""
        self.bank.remember("Önemli bilgi burada yer alıyor.", importance=0.8)
        self.assertIsNone(
            self.bank.remember("Tamamen önemsiz bir detay.", importance=0.05)
        )

    def test_recall_excludes_generated(self) -> None:
        """`query:` kaynaklı kayıtlar dışlanabilir (öz besleme önleme)."""
        self.bank.remember("Sistemin kendi ürettiği yanıt metni.",
                           source="query:answer", importance=0.9)
        self.bank.remember("Kasada bulunan gerçek bilgi metni.",
                           source="vault:kn_2", importance=0.9)
        excluded = self.bank.recall("metin", top_k=5,
                                    exclude_source_prefixes=("query:",))
        sources = [result.record.source for result in excluded]
        self.assertNotIn("query:answer", sources)

    def test_persistence_roundtrip(self) -> None:
        """Banka kapatılıp açıldığında kayıtlar korunur."""
        self.bank.remember("Kalıcılık testi için önemli bilgi.", source="vault:kn_3",
                           importance=0.8)
        self.bank.save()

        restored = MemoryBank()
        restored.initialize(None, None, store_dir=self.root)
        restored.attach_embedder(self.embedding.embed)
        self.assertGreater(restored.size, 0)
        self.assertTrue(restored.recall("kalıcılık", top_k=1))
        restored.shutdown()

    def test_consolidate(self) -> None:
        """Benzer kayıtlar birleştirilir."""
        self.bank.novelty_threshold = 0.99  # tekrar engelini geçici olarak kaldır
        self.bank.remember("GPU sıcaklık yönetimi önemlidir.", source="a",
                           importance=0.7, force=True)
        self.bank.remember("GPU sıcaklık yönetimi önemlidir!", source="b",
                           importance=0.7, force=True)
        result = self.bank.consolidate(min_similarity=0.8)
        self.assertGreaterEqual(result["merged"], 1)

    def test_associate_feedback(self) -> None:
        """Geri bildirim kaydın başarı puanını günceller."""
        record_id = self.bank.remember("Geri bildirim testi metni.", source="a",
                                       importance=0.7)
        before = self.bank._records[record_id].success
        self.assertTrue(self.bank.associate(record_id, 0.95))
        self.assertGreater(self.bank._records[record_id].success, before)

    def test_stats_json_safe(self) -> None:
        """Bellek istatistikleri JSON-güvenli."""
        assert_json_safe(self, self.bank.get_stats(), "memory_stats")


class TestGrowthPolicy(unittest.TestCase):
    """Büyüme politikası: karar mantığı ve maliyet."""

    def setUp(self) -> None:
        self.controller = GrowthController()
        self.controller.initialize(None, None)

    def tearDown(self) -> None:
        self.controller.shutdown()

    def test_grow_when_capacity_insufficient(self) -> None:
        """Yüksek entropi + düşük güven + yeterli veri → uzman büyüt."""
        metrics = GrowthMetrics(
            expert_entropy=0.95, routing_confidence=0.1,
            num_experts=4, max_experts=12, buffer_size=500, min_samples=100,
            expert_usage=[0.25] * 4, memory_pressure="NORMAL",
        )
        decision = self.controller.decide(metrics)
        self.assertEqual(decision.action, GrowthAction.GROW_EXPERTS)
        self.assertGreater(decision.confidence, 0.4)

    def test_noop_when_healthy(self) -> None:
        """Sağlıklı modelde büyüme yapılmaz."""
        metrics = GrowthMetrics(
            expert_entropy=0.3, routing_confidence=0.8,
            num_experts=4, max_experts=12, buffer_size=500,
            mean_quality=0.85, expert_usage=[0.25] * 4,
        )
        decision = self.controller.decide(metrics)
        self.assertEqual(decision.action, GrowthAction.NOOP)

    def test_freeze_idle_expert(self) -> None:
        """Kullanılmayan uzman dondurulur."""
        metrics = GrowthMetrics(
            expert_entropy=0.4, routing_confidence=0.7,
            num_experts=5, max_experts=12, buffer_size=500,
            expert_usage=[0.4, 0.35, 0.24, 0.001, 0.009],
        )
        decision = self.controller.decide(metrics)
        self.assertIn(decision.action, (GrowthAction.FREEZE_EXPERT,
                                        GrowthAction.PRUNE_EXPERT))

    def test_critical_pressure_prunes(self) -> None:
        """Kritik bellek baskısında kaynak serbest bırakılır."""
        metrics = GrowthMetrics(
            expert_entropy=0.9, routing_confidence=0.1,
            num_experts=8, max_experts=12, buffer_size=500,
            expert_usage=[0.4, 0.3, 0.2, 0.05, 0.02, 0.01, 0.01, 0.01],
            memory_pressure="CRITICAL",
        )
        decision = self.controller.decide(metrics)
        self.assertIn(decision.action, (GrowthAction.PRUNE_EXPERT,
                                        GrowthAction.QUANTIZE))

    def test_noop_at_max_experts(self) -> None:
        """Uzman sınırında büyüme yapılmaz."""
        metrics = GrowthMetrics(
            expert_entropy=0.99, routing_confidence=0.05,
            num_experts=12, max_experts=12, buffer_size=1000,
            expert_usage=[1 / 12] * 12,
        )
        decision = self.controller.decide(metrics)
        self.assertNotEqual(decision.action, GrowthAction.GROW_EXPERTS)

    def test_training_plan(self) -> None:
        """Eğitim planı önerisi üretilir."""
        metrics = GrowthMetrics(buffer_size=400, mean_quality=0.3, min_samples=100)
        plan = self.controller.next_training_plan(metrics)
        self.assertTrue(plan["should_train"])
        self.assertGreater(plan["steps"], 0)
        self.assertGreater(plan["lr"], 0)

    def test_stats_json_safe(self) -> None:
        """Büyüme istatistikleri JSON-güvenli."""
        assert_json_safe(self, self.controller.get_stats(), "growth_stats")


class TestTransformerConfig(unittest.TestCase):
    """Omurga yapılandırması: profil eşlemesi ve bellek tahmini."""

    def test_config_from_system(self) -> None:
        """Sistem konfigürasyonu omurga yapılandırmasına çevrilir."""
        config = isolated_config()
        transformer = config_from_system(config)
        self.assertEqual(transformer.embed_dim, config.neural.embedding_dim)
        self.assertEqual(transformer.num_layers, config.neural.num_transformer_layers)
        self.assertEqual(transformer.embed_dim % transformer.num_heads, 0)

    def test_budget_overrides(self) -> None:
        """Kaynak bütçesi mimariyi küçültür."""
        config = isolated_config()
        # Büyük bir taban kullan (test yapılandırması zaten küçültülmüş olabilir)
        config.neural.embedding_dim = 768
        config.neural.num_transformer_layers = 8
        config.neural.num_attention_heads = 12
        base = config_from_system(config)
        reduced = config_from_system(config, budget={
            "embedding_dim": 128, "num_transformer_layers": 2,
            "num_attention_heads": 4, "memory_pressure": "CRITICAL",
            "gradient_checkpointing": True,
        })
        self.assertLess(reduced.embed_dim, base.embed_dim)
        self.assertLess(reduced.num_layers, base.num_layers)
        self.assertTrue(reduced.gradient_checkpointing)
        self.assertEqual(reduced.top_k_experts, 1)

    def test_memory_estimate(self) -> None:
        """Bellek tahmini makul değerler üretir."""
        from neurocore.neural.transformer import estimate_backbone_memory

        config = TransformerConfig(vocab_size=2000, embed_dim=128, num_layers=2, num_heads=4)
        estimate = estimate_backbone_memory(config, batch_size=4, seq_len=128)
        self.assertGreater(estimate["parameters"], 0)
        self.assertGreater(estimate["weights_mb"], 0)
        self.assertIn("training_mb", estimate)
        assert_json_safe(self, estimate, "memory_estimate")

    def test_head_adjustment(self) -> None:
        """Bölünmeyen baş sayısı sessizce düzeltilir."""
        config = TransformerConfig(vocab_size=100, embed_dim=100, num_layers=2, num_heads=7)
        self.assertEqual(config.embed_dim % config.num_heads, 0)


@unittest.skipUnless(HAS_TORCH, "torch kurulu değil")
class TestAttentionAndMoE(unittest.TestCase):
    """Dikkat mekanizması ve MoE: sayısal doğruluk ve nötr büyüme."""

    def test_kv_cache_matches_full_forward(self) -> None:
        """KV önbellekli artımlı çıkarım tam ileri geçişle aynı sonucu verir."""
        from neurocore.neural.attention import AttentionConfig, KVCache, MultiHeadAttention

        config = AttentionConfig(embed_dim=64, num_heads=4, dropout=0.0, max_seq_len=64)
        attention = MultiHeadAttention(config)
        attention.eval()
        x = torch.randn(2, 12, 64)

        with torch.no_grad():
            full, _ = attention(x)
            cache = KVCache(max_len=64)
            stepwise = torch.cat(
                [attention(x[:, index:index + 1, :], cache=cache, use_cache=True)[0]
                 for index in range(12)],
                dim=1,
            )
        self.assertTrue(torch.allclose(full, stepwise, atol=1e-4))

    def test_no_nan_with_padding_mask(self) -> None:
        """Tamamen dolgu olan satırlar NaN üretmez."""
        from neurocore.neural.attention import AttentionConfig, MultiHeadAttention

        config = AttentionConfig(embed_dim=32, num_heads=2, dropout=0.0, max_seq_len=32)
        attention = MultiHeadAttention(config)
        attention.eval()
        x = torch.randn(1, 8, 32)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]])
        with torch.no_grad():
            output, _ = attention(x, attention_mask=mask)
        self.assertTrue(torch.isfinite(output).all(), "çıktı sonlu olmalı")

    def test_moe_growth_is_neutral(self) -> None:
        """Yeni uzman eklemek model çıktısını değiştirmez (nötr büyüme)."""
        from neurocore.neural.moe import MixtureOfExperts, MoEConfig

        config = MoEConfig(embed_dim=64, num_experts=3, max_experts=8, top_k=2,
                           expert_hidden_dim=96)
        moe = MixtureOfExperts(config)
        moe.eval()
        x = torch.randn(2, 8, 64)

        with torch.no_grad():
            before = moe(x).clone()
            added = moe.grow(2, zero_init=True)
            after = moe(x)
        self.assertEqual(added, 2)
        delta = float((before - after).abs().max())
        self.assertLess(delta, 1e-4, f"büyüme nötr olmalı (delta={delta})")

    def test_moe_balance_loss_finite(self) -> None:
        """Dengeleme kaybı sonlu ve pozitif."""
        from neurocore.neural.moe import MixtureOfExperts, MoEConfig

        moe = MixtureOfExperts(MoEConfig(embed_dim=32, num_experts=4, top_k=2,
                                         expert_hidden_dim=48))
        x = torch.randn(2, 8, 32)
        output, aux_loss, stats = moe(x, return_stats=True)
        self.assertTrue(torch.isfinite(aux_loss))
        self.assertGreaterEqual(float(aux_loss), 0.0)
        self.assertEqual(len(stats.expert_usage), 4)
        # `expert_usage` atama oranıdır: her belirteç top_k=2 uzmana atanır,
        # bu yüzden toplamı top_k'dır. Göreli pay toplamı 1.0 olmalı.
        self.assertAlmostEqual(sum(stats.expert_usage), 2.0, places=3)
        self.assertAlmostEqual(sum(stats.expert_share()), 1.0, places=5)
        for share in stats.expert_share():
            self.assertGreaterEqual(share, 0.0)
            self.assertLessEqual(share, 1.0)

    def test_expert_pruning_keeps_model_usable(self) -> None:
        """Uzman budama sonrası model çalışmaya devam eder."""
        from neurocore.neural.moe import MixtureOfExperts, MoEConfig

        moe = MixtureOfExperts(MoEConfig(embed_dim=32, num_experts=4, max_experts=6,
                                         top_k=2, expert_hidden_dim=48))
        x = torch.randn(1, 4, 32)
        self.assertTrue(moe.prune_expert(1))
        self.assertEqual(moe.num_experts, 3)
        output = moe(x)
        self.assertEqual(tuple(output.shape), (1, 4, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_generation_produces_tokens(self) -> None:
        """Üretim istenen sayıda token ekler."""
        from neurocore.neural.transformer import TransformerBackbone, TransformerConfig

        config = TransformerConfig(vocab_size=256, embed_dim=64, num_layers=1,
                                   num_heads=4, max_seq_len=64, dropout=0.0,
                                   num_experts=2, max_experts=4)
        model = TransformerBackbone(config)
        model.eval()
        prompt = torch.randint(0, 256, (1, 4))
        generated = model.generate(prompt, max_new_tokens=8, temperature=0.0)
        self.assertEqual(generated.shape[1], 12)


class TestTrainerDataset(unittest.TestCase):
    """Eğitim veri kümesi: kodlama, toplu iş, bölme."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = BPETokenizer()
        cls.tokenizer.initialize(None, None)
        cls.tokenizer.train(
            ["Yapay sinir ağları gradyan inişi ile eğitilir."] * 30
            + ["GPU sıcaklığı termal macun ile düşürülür."] * 30,
            vocab_size=400,
        )

    def setUp(self) -> None:
        from neurocore.neural.trainer import TrainingDataset

        self.dataset = TrainingDataset(self.tokenizer, seq_len=64)
        for index in range(10):
            self.dataset.add(f"Soru {index} nedir?", f"Bu {index}. yanıt metnidir.", weight=1.0)

    def test_add_and_encode(self) -> None:
        """Örnekler eklenir ve kodlanır."""
        self.assertEqual(len(self.dataset), 10)
        stats = self.dataset.stats()
        self.assertEqual(stats["encoded"], 10)
        self.assertGreater(stats["tokens"], 0)

    def test_batches_shapes(self) -> None:
        """Toplu işler doğru biçimde üretilir."""
        batches = list(self.dataset.batches(4))
        self.assertGreater(len(batches), 0)
        first = batches[0]
        if HAS_TORCH:
            self.assertEqual(first["input_ids"].shape[0], 4)
            self.assertEqual(first["labels"].shape, first["input_ids"].shape)
            self.assertEqual(first["attention_mask"].shape, first["input_ids"].shape)
            self.assertTrue(torch.all(first["labels"] <= self.tokenizer.vocab_size))

    def test_padding_labels_masked(self) -> None:
        """Dolgu konumları kayıptan hariç tutulur (-100)."""
        batches = list(self.dataset.batches(8))
        if HAS_TORCH:
            labels = batches[0]["labels"]
            mask = batches[0]["attention_mask"]
            padded_positions = mask == 0
            if padded_positions.any():
                self.assertTrue(
                    torch.all(labels[padded_positions] == -100),
                    "dolgu etiketleri -100 olmalı",
                )

    def test_split(self) -> None:
        """Veri kümesi eğitim/doğrulama olarak bölünür."""
        train, valid = self.dataset.split(validation_ratio=0.3)
        self.assertEqual(len(train) + len(valid), 10)
        self.assertGreater(len(valid), 0)

    def test_from_batch(self) -> None:
        """Evrim katmanı paketinden veri yüklenir."""
        from neurocore.neural.trainer import TrainingDataset

        dataset = TrainingDataset(self.tokenizer, seq_len=64)
        added = dataset.from_batch({
            "prompts": ["soru bir", "soru iki"],
            "targets": ["yanıt bir", "yanıt iki"],
            "weights": [1.0, 2.0],
        })
        self.assertEqual(added, 2)
        self.assertEqual(dataset.samples[1].weight, 2.0)

    def test_save_load(self) -> None:
        """Veri kümesi JSONL olarak kaydedilip yüklenir."""
        from neurocore.neural.trainer import TrainingDataset

        root = tempfile.mkdtemp(prefix="nc_ds_")
        try:
            path = self.dataset.save(str(Path(root) / "dataset.jsonl"))
            restored = TrainingDataset.load(path, self.tokenizer, seq_len=64)
            self.assertEqual(len(restored), len(self.dataset))
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestValidationWiring(unittest.TestCase):
    """
    Doğrulama kaybı bağlantısı — **regresyon koruması**.

    ⚠ Ölçülen hata (Aşama 0):
        `train(validation=...)` parametresi kabul ediliyordu ama
        **eğitim döngüsünde hiç kullanılmıyordu**. Bu yüzden eğitim
        kaybı 0,083'e inerken gerçek genelleme 6,44 (rastgele 8,45)
        kaldı ve **hiç fark edilmedi**.

        Doğrulama artık: başlangıçta bir kez + her `eval_every` adımda
        ölçülür, `val_history`'ye yazılır ve `TrainingResult`'a geçer.
    """

    def test_result_alanlari(self) -> None:
        """`TrainingResult` doğrulama alanlarını taşımalı."""
        from neurocore.neural.trainer import TrainingResult

        r = TrainingResult()
        for ad in ("initial_val_loss", "final_val_loss",
                   "best_val_loss", "val_history"):
            self.assertTrue(hasattr(r, ad), f"{ad} eksik")
        self.assertEqual(r.val_history, [])

    def test_to_dict_dogrulama(self) -> None:
        """`to_dict()` doğrulama değerlerini JSON-güvenli vermeli."""
        from tests.helpers import assert_json_safe

        from neurocore.neural.trainer import TrainingResult

        r = TrainingResult(initial_val_loss=6.5, final_val_loss=5.4,
                           best_val_loss=5.2)
        r.val_history.append({"step": 0, "loss": 6.5,
                              "perplexity": 665.1})
        d = r.to_dict()
        self.assertIn("initial_val_loss", d)
        self.assertIn("final_val_loss", d)
        self.assertIn("best_val_loss", d)
        self.assertIn("val_history", d)
        self.assertAlmostEqual(d["initial_val_loss"], 6.5, places=4)
        self.assertAlmostEqual(d["final_val_loss"], 5.4, places=4)
        assert_json_safe(self, d)

    def test_train_dongusunde_kullanilir(self) -> None:
        """
        ⚠ Asıl korunacak şey: `validation` **döngüde** kullanılmalı.
        Yalnız imzada durması bu hatanın ta kendisiydi.
        """
        import inspect

        from neurocore.neural.trainer import Trainer

        kaynak = inspect.getsource(Trainer.train)
        self.assertIn("eval_every", kaynak,
                      "eval_every döngüde yok — doğrulama ölü")
        self.assertIn("validation is not None", kaynak)
        self.assertIn("val_history", kaynak)
        self.assertIn("neural.validation", kaynak,
                      "doğrulama olayı yayınlanmıyor")

    def test_evaluate_var(self) -> None:
        """`evaluate()` doğrulama kaybını üretmeli (kayıp + ppl)."""
        import inspect

        from neurocore.neural.trainer import Trainer

        src = inspect.getsource(Trainer.evaluate)
        self.assertIn("no_grad", src)
        self.assertIn("perplexity", src)

    def test_plan_eval_every_alani(self) -> None:
        """Plan `eval_every` taşımalı (varsayılan 0 = kapalı)."""
        from neurocore.neural.trainer import TrainingPlan

        p = TrainingPlan()
        self.assertEqual(p.eval_every, 0)
        self.assertIn("eval_every", p.to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
