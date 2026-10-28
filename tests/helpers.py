"""
NeuroCore test paketi ortak yardımcıları.

Bu modül, testlerin gerçek sistemi kurmasını sağlar:

    * Geçici bir kök dizinde izole konfigürasyon (`isolated_config`)
    * Kurulmuş bir `NeuroCoreEngine` (fixture benzeri bağlam yöneticisi)
    * Küçük ama gerçek bir kasa içeriği (`seed_vault`)
    * Testlerde tekrar eden doğrulamalar için yardımcılar

Tasarım kararı: Testler **gerçek** bileşenleri kullanır (mock değil), ancak
geçici dizinler sayesinde birbirini etkilemez ve geliştiricinin kasasını
kirletmez.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# NVIDIA sürücü sorgusu testlerde de uyarı üretmesin
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore")

from neurocore.core.config import HardwareProfile, SystemConfig
from neurocore.core.engine import NeuroCoreEngine


# ─── Kasa İçeriği ───────────────────────────────────────────────────────────

#: Testlerde kullanılan gerçekçi Türkçe notlar (kasa dolgunluğu için)
SAMPLE_NOTES: List[Dict[str, Any]] = [
    {
        "title": "GPU Termal Yönetimi",
        "body": (
            "GPU sıcaklığı yük altında 80-85 derece aralığında çalışmalıdır. "
            "90 derecenin üzeri throttling başlatır ve uzun vadede donanıma zarar verir.\n\n"
            "Başlıca nedenler: toz birikmesi fanı tıkar, termal macun 2-3 yılda "
            "kurur, yetersiz hava akışı sıcak havayı hapseder."
        ),
        "tags": ["donanim", "gpu", "termal"],
        "confidence": 0.82,
        "kind": "concept",
    },
    {
        "title": "Termal Macun Yenileme",
        "body": (
            "Termal macun iki yılda bir yenilenmelidir. Yoğun kullanımda bu süre "
            "bir yıla inebilir. Uygulama öncesi yüzey izopropil alkolle temizlenir.\n\n"
            "Aşırı macun uygulamak ısı transferini azaltır; bezelye tanesi kadar "
            "yeterlidir."
        ),
        "tags": ["termal", "bakim"],
        "confidence": 0.75,
        "kind": "skill",
    },
    {
        "title": "CPU Sıcaklık İzleme",
        "body": (
            "CPU sıcaklığı yük altında 90 dereceyi aşmamalıdır. Soğutucu montajı ve "
            "termal macun durumu düzenli kontrol edilmelidir."
        ),
        "tags": ["donanim", "cpu", "termal"],
        "confidence": 0.7,
        "kind": "concept",
    },
    {
        "title": "Yapay Sinir Ağı Temelleri",
        "body": (
            "Yapay sinir ağları gradyan inişi ve geri yayılım algoritması ile "
            "eğitilir. Kayıp fonksiyonu, öğrenme oranı ve toplu iş boyutu temel "
            "hiperparametrelerdir."
        ),
        "tags": ["yapay-zeka", "sinir-agi"],
        "confidence": 0.8,
        "kind": "concept",
    },
    {
        "title": "Bilgi Kasası Yapısı",
        "body": (
            "Kasa Obsidian uyumlu markdown dosyalarından oluşur. Her notun "
            "frontmatter bölümünde kimlik, başlık, etiketler ve güven alanları "
            "bulunur. Bağlantılar çift köşeli parantez ile kurulur."
        ),
        "tags": ["bilgi-kasasi", "obsidian"],
        "confidence": 0.85,
        "kind": "concept",
    },
]


# ─── İzole Konfigürasyon ────────────────────────────────────────────────────

def isolated_config(root: Optional[str] = None, profile: str = "low") -> SystemConfig:
    """
    Geçici kökte izole bir `SystemConfig` üretir.

    Dizinler (`vault`, `data`, `models`) **hemen** oluşturulur; böylece
    testler katman kurulumunu beklemeden dosya sistemine güvenebilir.

    Args:
        root: Kök dizin (None = yeni geçici dizin)
        profile: Donanım profili (`low` = testlerde hızlı kurulum)

    Returns:
        Yapılandırılmış SystemConfig
    """
    base = Path(root) if root else Path(tempfile.mkdtemp(prefix="neurocore_test_"))
    base.mkdir(parents=True, exist_ok=True)

    config = SystemConfig()
    config.base_path = str(base)
    config.knowledge.vault_path = str(base / "knowledge_base")
    config.data_path = str(base / "data")
    config.models_path = str(base / "models")
    config.debug_mode = False

    # Dizinleri hemen oluştur (katmanlar tembel oluştursa da testler beklemesin)
    for directory in (config.knowledge.vault_path, config.data_path, config.models_path):
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    try:
        config.apply_hardware_profile(HardwareProfile(profile))
    except ValueError:
        config.apply_hardware_profile(HardwareProfile.LOW)

    # Testlerde hız için küçük model
    config.neural.num_transformer_layers = 2
    config.neural.embedding_dim = 128
    config.neural.num_attention_heads = 4
    config.neural.num_initial_experts = 3
    config.neural.max_experts = 6
    config.multimodal.vocab_size = 1024
    config.multimodal.max_text_length = 256
    config.max_batch_size = 8
    config.num_workers = 2
    return config


@contextmanager
def engine_context(profile: str = "low", seed: bool = True,
                   start: bool = False) -> Iterator[NeuroCoreEngine]:
    """
    Geçici kökte kurulmuş bir motor sağlar ve sonunda temizler.

    Kullanım:
        with engine_context() as engine:
            result = engine.query("...")

    Args:
        profile: Donanım profili
        seed: Kasa örnek notlarla doldurulsun mu?
        start: Arka plan döngüleri başlatılsın mı?
    """
    config = isolated_config(profile=profile)
    engine = NeuroCoreEngine(config)
    try:
        engine.initialize()
        if seed:
            seed_vault(engine)
        if start:
            engine.start(maintenance_interval=600)
        yield engine
    finally:
        try:
            engine.shutdown(save=False)
        except Exception:
            pass
        shutil.rmtree(config.base_path, ignore_errors=True)


def seed_vault(engine: NeuroCoreEngine, notes: Optional[List[Dict[str, Any]]] = None) -> int:
    """
    Kasaya örnek notları yazar.

    Returns:
        Yazılan not sayısı
    """
    if engine.knowledge is None:
        return 0
    written = 0
    for note in (notes or SAMPLE_NOTES):
        try:
            engine.knowledge.write(
                title=note["title"],
                body=note["body"],
                tags=note.get("tags"),
                confidence=note.get("confidence", 0.6),
                kind=note.get("kind", "concept"),
                reason="test verisi",
                author="test",
            )
            written += 1
        except Exception:
            continue
    return written


# ─── Doğrulama Yardımcıları ─────────────────────────────────────────────────

def assert_json_safe(test_case: Any, payload: Any, name: str = "payload") -> None:
    """
    Bir yapının JSON'a serileştirilebildiğini doğrular.

    Tüm katmanların `get_stats()` sözleşmesi bunu gerektirir; bu yardımcı
    arayüzlerin JSON-güvenli kalmasını garanti eder.
    """
    try:
        json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        test_case.fail(f"{name} JSON-serileştirilebilir değil: {exc}")


def assert_vector(test_case: Any, vector: Any, dim: Optional[int] = None,
                  name: str = "vector") -> None:
    """Bir gömme vektörünün geçerliliğini doğrular (boyut, sonluluk, normalizasyon)."""
    import math

    test_case.assertIsInstance(vector, list, f"{name} liste olmalı")
    test_case.assertTrue(len(vector) > 0, f"{name} boş olmamalı")
    if dim is not None:
        test_case.assertEqual(len(vector), dim,
                              f"{name} boyutu {dim} olmalı, {len(vector)} bulundu")
    for value in vector:
        test_case.assertTrue(math.isfinite(float(value)),
                             f"{name} sonlu değerler içermeli")
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    test_case.assertAlmostEqual(norm, 1.0, places=3,
                                msg=f"{name} L2-normalize olmalı (norm={norm:.4f})")


def summarize_engine(engine: NeuroCoreEngine) -> Dict[str, Any]:
    """Testlerde log/teşhis için motorun kısa özeti."""
    return {
        "state": engine.state.value,
        "profile": engine.config.hardware_profile,
        "device": engine.config.device,
        "notes": len(engine.knowledge.vault._path_index) if engine.knowledge else 0,
        "params": engine.model.parameter_count() if engine.model else 0,
        "memory": engine.memory.size if engine.memory else 0,
        "errors": dict(engine._init_errors),
    }


__all__ = [
    "SAMPLE_NOTES",
    "isolated_config",
    "engine_context",
    "seed_vault",
    "assert_json_safe",
    "assert_vector",
    "summarize_engine",
]
