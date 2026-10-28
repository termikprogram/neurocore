"""
NeuroCore Çok-Modlu Füzyon Katmanı (L4)
========================================
Farklı modalitelerden gelen `EncodedChunk` vektörlerini tek bir ortak temsile
(`FusedRepresentation`) birleştirir.

Stratejiler:
    - **attention** : numpy ile gerçek tek-kafalı cross-modal dikkat
      (Q/K/V projeksiyonları, softmax(Q·Kᵀ/√d), ağırlıklı V toplamı)
    - **early**     : tüm vektörleri birleştirip öğrenilmiş-benzeri sabit
      projeksiyonla `embedding_dim`'e indirgeme
    - **late**      : güven × modalite önseli ile ağırlıklı ortalama
    - **concat**    : birleştirip deterministik katlama/tekrar ile hizalama

Tüm projeksiyonlar sabit tohumlu (`salt="neurocore-fusion"`) olduğundan sonuç
süreçten bağımsız **deterministiktir**. `torch` gerekmez (nn.Module benzeri
sade bir sınıftır).

Örnek:
    fusion = FusionLayer()
    fusion.initialize(config, bus)
    fused = fusion.fuse([text_chunk, image_chunk], strategy="attention")
    print(fused.weights)   # {'text': 0.61, 'image': 0.39}
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from neurocore.multimodal.types import (
    DEFAULT_EMBEDDING_DIM,
    EncodedChunk,
    FusedRepresentation,
    Modality,
    attach_native,
    cosine_similarity,
    l2_normalize,
    projection_matrix,
)
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("multimodal.fusion")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: Füzyon projeksiyonlarının tohumu ve tuzu (deterministik — değiştirilmemeli).
FUSION_SEED = 7
FUSION_SALT = "neurocore-fusion"

#: Desteklenen füzyon stratejileri.
FUSION_STRATEGIES: Tuple[str, ...] = ("attention", "early", "late", "concat")

#: Modalite önsel ağırlıkları (late/early/concat stratejilerinde kullanılır).
DEFAULT_MODALITY_PRIORS: Dict[str, float] = {
    Modality.TEXT.value: 1.00,
    Modality.IMAGE.value: 0.90,
    Modality.AUDIO.value: 0.85,
    Modality.TABLE.value: 0.80,
    Modality.UNKNOWN.value: 0.50,
}


# ─── Ana Füzyon Sınıfı ─────────────────────────────────────────────────────

class FusionLayer:
    """
    Modalite vektörlerini birleştiren füzyon katmanı (nn.Module benzeri düz sınıf).

    `torch` bağımlılığı yoktur; `__call__`, `to()`, `train()`, `eval()` gibi
    modül benzeri yardımcılar API uyumu için sağlanmıştır.
    """

    name = "multimodal.fusion"

    def __init__(self) -> None:
        """Yalnızca durum kurulumu — G/Ç veya ağır import yok."""
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._embedding_dim = DEFAULT_EMBEDDING_DIM
        self._default_strategy = "attention"
        self._priors: Dict[str, float] = dict(DEFAULT_MODALITY_PRIORS)
        self._matrix_cache: Dict[Tuple[str, int, int], "np.ndarray"] = {}
        self._training = False
        self._stats: Dict[str, Any] = {
            "fuse_calls": 0,
            "empty_fuses": 0,
            "chunks_fused": 0,
            "single_chunk": 0,
            "errors": 0,
        }
        self._strategy_counts: Dict[str, int] = {s: 0 for s in FUSION_STRATEGIES}

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """
        Füzyon katmanını yapılandırır (idempotent).

        Args:
            config: `SystemConfig` (`embedding_dim`, `fusion_strategy` okunur)
            event_bus: Olay yolu (opsiyonel)
        """
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._embedding_dim = self._read_embedding_dim(config)
        strategy = self._read_strategy(config)
        if strategy not in FUSION_STRATEGIES:
            logger.warning("Bilinmeyen füzyon stratejisi '%s' — 'attention' kullanılacak",
                           strategy)
            strategy = "attention"
        self._default_strategy = strategy
        self._initialized = True
        logger.info("Füzyon katmanı hazır (dim=%d, strateji=%s)",
                    self._embedding_dim, self._default_strategy)
        if self._event_bus:
            self._event_bus.emit(
                "multimodal.fusion_ready",
                {"dim": self._embedding_dim, "strategy": self._default_strategy,
                 "strategies": list(FUSION_STRATEGIES)},
                source=self.name,
            )

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır (idempotent)."""
        if not self._initialized:
            return
        self._matrix_cache.clear()
        self._initialized = False
        logger.info("Füzyon katmanı kapatıldı")

    # ─── Modül Benzeri Yardımcılar ──────────────────────────────────────

    def __call__(self, chunks: List[EncodedChunk],
                 strategy: Optional[str] = None) -> FusedRepresentation:
        """`fuse()` için kısayol (nn.Module benzeri çağrı)."""
        return self.fuse(chunks, strategy=strategy)

    def to(self, device: Any) -> "FusionLayer":
        """Cihaz taşıma taklidi — numpy tabanlı olduğu için kendini döndürür."""
        logger.debug("Füzyon katmanı saf numpy çalışır; cihaz '%s' yok sayıldı", device)
        return self

    def train(self, mode: bool = True) -> "FusionLayer":
        """Eğitim modu bayrağı (parametre olmadığı için yalnızca durum tutar)."""
        self._training = bool(mode)
        return self

    def eval(self) -> "FusionLayer":
        """Değerlendirme moduna alır."""
        return self.train(False)

    def parameters(self) -> List[Any]:
        """Öğrenilmiş parametre yok — boş liste (API uyumu)."""
        return []

    # ─── Ana Füzyon ─────────────────────────────────────────────────────

    def fuse(self, chunks: List[EncodedChunk],
             strategy: Optional[str] = None) -> FusedRepresentation:
        """
        Chunk listesini seçilen stratejiyle tek temsile birleştirir.

        Args:
            chunks: `EncodedChunk` listesi (None/boş öğeler atlanır)
            strategy: "attention" | "early" | "late" | "concat"
                (None → config.multimodal.fusion_strategy)

        Returns:
            `FusedRepresentation`; liste boşsa sıfır vektör + Türkçe uyarı logu
        """
        self._stats["fuse_calls"] += 1
        selected = strategy or self._default_strategy
        if selected not in FUSION_STRATEGIES:
            logger.warning("Bilinmeyen strateji '%s' — 'attention' kullanılacak", selected)
            selected = "attention"

        valid = self._valid_chunks(chunks)
        if not valid:
            self._stats["empty_fuses"] += 1
            logger.warning("Füzyon için geçerli chunk yok — sıfır vektör döndürülüyor")
            return FusedRepresentation(
                vector=[0.0] * self._embedding_dim,
                modalities=[],
                weights={},
                strategy=selected,
                parts={},
                meta={"empty": True, "n_chunks": 0, "embedding_dim": self._embedding_dim},
            )

        self._stats["chunks_fused"] += len(valid)
        self._strategy_counts[selected] = self._strategy_counts.get(selected, 0) + 1
        if len(valid) == 1:
            self._stats["single_chunk"] += 1
            return self._single(valid[0], selected)

        try:
            if selected == "attention":
                fused = self.attention_fuse(valid)
            elif selected == "early":
                fused = self.early_fuse(valid)
            elif selected == "late":
                fused = self.late_fuse(valid)
            else:
                fused = self.concat_fuse(valid)
        except Exception as exc:
            self._stats["errors"] += 1
            logger.error("Füzyon hatası (%s): %s — late fusion'a düşülüyor", selected, exc)
            fused = self.late_fuse(valid)

        if self._event_bus:
            self._event_bus.emit(
                "multimodal.fusion_applied",
                {"strategy": fused.strategy, "n_chunks": len(valid),
                 "weights": fused.weights,
                 "modalities": [m.value for m in fused.modalities]},
                source=self.name,
            )
        return fused

    def attention_fuse(self, chunks: List[EncodedChunk]) -> FusedRepresentation:
        """
        Gerçek tek-kafalı cross-modal dikkat ile füzyon (saf numpy).

        Adımlar:
            1. Her chunk `embedding_dim`'e hizalanır ve X (n, d) matrisi kurulur.
            2. Paylaşılan sabit projeksiyonlarla Q = X·Wq, K = X·Wk, V = X·Wv.
            3. Skorlar = Q·Kᵀ/√d + 0.5·log(g)  (g = güven × modalite önseli).
            4. A = softmax(skorlar, eksen=1) → satır ortalaması dikkat kütlesi.
            5. Çıktı = (kütle · (g ⊙ V)) · Wo → L2 normalize.

        Args:
            chunks: En az iki geçerli chunk

        Returns:
            `FusedRepresentation`; `weights` softmax dikkat kütlesidir (toplam 1)
        """
        matrix, names, gains = self._prepare(chunks)
        dim = matrix.shape[1]

        query = matrix @ self._matrix(dim, dim, "q")
        key = matrix @ self._matrix(dim, dim, "k")
        value = matrix @ self._matrix(dim, dim, "v")

        scores = (query @ key.T) / math.sqrt(max(1, dim))
        scores = scores + 0.5 * np.log(np.clip(gains, 1e-6, None))[None, :]

        attention = self._softmax(scores, axis=1)
        mass = attention.mean(axis=0)
        total = float(mass.sum())
        mass = mass / total if total > 0 else np.full_like(mass, 1.0 / mass.size)

        output = mass @ (value * gains[:, None])
        output = output @ self._matrix(dim, dim, "o")
        vector = l2_normalize(output.tolist())

        weights = self._weights_from_vector(mass, names)
        return FusedRepresentation(
            vector=vector,
            modalities=self._modalities(chunks),
            weights=weights,
            strategy="attention",
            parts=self._parts(chunks, names),
            meta=self._meta(chunks, {
                "attention_entropy": round(float(-np.sum(
                    np.where(mass > 0, mass * np.log(mass + 1e-12), 0.0))), 6),
                "gains": {name: round(float(gain), 6)
                          for name, gain in zip(names, gains)},
                "head_dim": int(dim),
            }),
        )

    def early_fuse(self, chunks: List[EncodedChunk]) -> FusedRepresentation:
        """
        Erken füzyon: vektörleri birleştirip sabit projeksiyonla `embedding_dim`'e indirger.

        Args:
            chunks: En az iki geçerli chunk

        Returns:
            `FusedRepresentation` (ağırlıklar güven × önsel ile normalize edilir)
        """
        matrix, names, gains = self._prepare(chunks)
        concatenated = matrix.ravel()
        in_dim = int(concatenated.shape[0])
        out_dim = self._embedding_dim
        projected = concatenated @ self._matrix(in_dim, out_dim, "early")
        vector = l2_normalize(projected.tolist())
        weights = self._weights_from_vector(gains / max(1e-9, float(gains.sum())), names)
        return FusedRepresentation(
            vector=vector,
            modalities=self._modalities(chunks),
            weights=weights,
            strategy="early",
            parts=self._parts(chunks, names),
            meta=self._meta(chunks, {
                "concatenated_dim": in_dim,
                "projection": f"({in_dim}, {out_dim})",
            }),
        )

    def late_fuse(self, chunks: List[EncodedChunk]) -> FusedRepresentation:
        """
        Geç füzyon: güven × modalite önseli ile ağırlıklı ortalama.

        Args:
            chunks: En az iki geçerli chunk

        Returns:
            `FusedRepresentation` (ağırlıklar toplamı 1)
        """
        matrix, names, gains = self._prepare(chunks)
        total = float(gains.sum())
        weights_vector = gains / total if total > 1e-9 else np.full_like(
            gains, 1.0 / gains.size)
        combined = weights_vector @ matrix
        vector = l2_normalize(combined.tolist())
        return FusedRepresentation(
            vector=vector,
            modalities=self._modalities(chunks),
            weights=self._weights_from_vector(weights_vector, names),
            strategy="late",
            parts=self._parts(chunks, names),
            meta=self._meta(chunks, {"weighted_by": "confidence*modality_prior"}),
        )

    def concat_fuse(self, chunks: List[EncodedChunk]) -> FusedRepresentation:
        """
        Basit birleştirme: vektörleri uç uca ekleyip `align_dim` ile hizalar.

        `early_fuse`'dan farkı: rastgele projeksiyon yerine deterministik
        katlama/tekrar kullanır (parametresiz, kayıpsız-hizalama).

        Args:
            chunks: En az iki geçerli chunk

        Returns:
            `FusedRepresentation`
        """
        matrix, names, gains = self._prepare(chunks)
        concatenated = matrix.ravel().tolist()
        aligned = self.align_dim(concatenated, self._embedding_dim)
        weights = gains / max(1e-9, float(gains.sum()))
        return FusedRepresentation(
            vector=l2_normalize(aligned),
            modalities=self._modalities(chunks),
            weights=self._weights_from_vector(weights, names),
            strategy="concat",
            parts=self._parts(chunks, names),
            meta=self._meta(chunks, {
                "concatenated_dim": len(concatenated),
                "alignment": "fold_and_tile",
            }),
        )

    # ─── Boyut Hizalama ─────────────────────────────────────────────────

    def align_dim(self, vector: Sequence[float], target_dim: int) -> List[float]:
        """
        Vektörü `target_dim` boyutuna deterministik olarak hizalar.

        - Eşitse: L2-normalize kopya
        - Uzunsa: fazlalık başa **katlanır** (`v[i % target] += v[i]`)
        - Kısaysa: vektör hedefe kadar **döngüsel tekrarla** doldurulur
          (sıfır dolgudan daha fazla bilgi korur)

        Args:
            vector: Girdi vektörü
            target_dim: Hedef boyut

        Returns:
            Uzunluğu `target_dim` olan L2-normalize liste
        """
        target = int(target_dim)
        if target <= 0:
            raise ValueError("target_dim pozitif olmalı")
        values = [float(v) for v in (vector or [])]
        if not values:
            return [0.0] * target
        if len(values) == target:
            return l2_normalize(values)

        folded = [0.0] * target
        for index, value in enumerate(values):
            folded[index % target] += value
        if len(values) < target:
            repeats = math.ceil(target / len(values))
            for repeat in range(1, repeats):
                for index, value in enumerate(values):
                    position = index + repeat * len(values)
                    if position >= target:
                        break
                    folded[position] += value / math.sqrt(repeats)
        return l2_normalize(folded)

    # ─── Modalite Önseli ────────────────────────────────────────────────

    def set_modality_prior(self, modality: Modality, weight: float) -> None:
        """
        Bir modalitenin füzyon önsel ağırlığını ayarlar.

        Args:
            modality: `Modality` veya string ("text", "image", ...)
            weight: 0..10 arası ağırlık (kırpılır)
        """
        key = Modality.from_value(modality).value
        self._priors[key] = float(np.clip(float(weight), 0.0, 10.0))
        logger.debug("Modalite önseli güncellendi: %s = %.3f", key, self._priors[key])

    def get_modality_prior(self, modality: Modality) -> float:
        """
        Bir modalitenin önsel ağırlığını döndürür.

        Args:
            modality: `Modality` veya string

        Returns:
            Önsel ağırlık (tanımsızsa 1.0)
        """
        key = Modality.from_value(modality).value
        return float(self._priors.get(key, 1.0))

    def get_priors(self) -> Dict[str, float]:
        """Tüm modalite önsel ağırlıklarının kopyasını döndürür."""
        return dict(self._priors)

    # ─── Benzerlik ──────────────────────────────────────────────────────

    def similarity(self, a: FusedRepresentation, b: FusedRepresentation) -> float:
        """
        İki füzyon temsili arasında kosinüs benzerliği hesaplar.

        Args:
            a: Birinci temsil
            b: İkinci temsil

        Returns:
            [-1, 1] arası benzerlik (uyumsuz boyutta 0.0)
        """
        if a is None or b is None:
            return 0.0
        if len(a.vector) != len(b.vector):
            logger.warning("Füzyon benzerliği için boyutlar farklı: %d vs %d",
                           len(a.vector), len(b.vector))
            return 0.0
        return float(cosine_similarity(a.vector, b.vector))

    # ─── İç Yardımcılar ─────────────────────────────────────────────────

    def _valid_chunks(self, chunks: Optional[Sequence[EncodedChunk]]) -> List[EncodedChunk]:
        """Boş/geçersiz chunk'ları ayıklar."""
        valid: List[EncodedChunk] = []
        for chunk in (chunks or []):
            if chunk is None:
                continue
            if not isinstance(chunk, EncodedChunk):
                logger.warning("Füzyona EncodedChunk olmayan öğe verildi: %s",
                               type(chunk).__name__)
                continue
            if not chunk.vector:
                logger.warning("Boş vektörlü chunk atlandı (%s)", chunk.modality.value)
                continue
            valid.append(chunk)
        return valid

    def _prepare(self, chunks: Sequence[EncodedChunk]
                 ) -> Tuple["np.ndarray", List[str], "np.ndarray"]:
        """
        Chunk'ları (n, d) matrisine, isimlere ve kazanç vektörüne çevirir.

        Args:
            chunks: Geçerli chunk listesi

        Returns:
            (matris, isimler, kazançlar) — kazanç = güven × modalite önseli
        """
        aligned = [np.asarray(self.align_dim(chunk.vector, self._embedding_dim),
                              dtype=np.float64) for chunk in chunks]
        matrix = np.vstack(aligned)
        gains = np.asarray([
            max(1e-3, float(chunk.confidence)) * max(1e-3, self.get_modality_prior(
                chunk.modality))
            for chunk in chunks
        ], dtype=np.float64)
        return matrix, self._unique_names(chunks), gains

    @staticmethod
    def _unique_names(chunks: Sequence[EncodedChunk]) -> List[str]:
        """Aynı modalite birden fazla kez geldiyse '#2', '#3' ekleriyle adlandırır."""
        seen: Dict[str, int] = {}
        names: List[str] = []
        for chunk in chunks:
            base = chunk.modality.value
            seen[base] = seen.get(base, 0) + 1
            names.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
        return names

    @staticmethod
    def _modalities(chunks: Sequence[EncodedChunk]) -> List[Modality]:
        """Chunk'ların modalite listesi."""
        return [chunk.modality for chunk in chunks]

    @staticmethod
    def _weights_from_vector(values: "np.ndarray", names: Sequence[str]
                             ) -> Dict[str, float]:
        """
        Vektörü isim→ağırlık sözlüğüne çevirir.

        Aynı modalite birden fazla kez geldiyse ağırlıkları toplanır, ardından
        toplam **tam olarak 1.0** olacak şekilde normalize edilir (yuvarlama
        artığı en büyük ağırlığa eklenir; böylece kayan nokta hatası birikmez).

        Args:
            values: Ağırlık vektörü (chunk sırasıyla)
            names: Chunk adları ("text", "image#2", ...)

        Returns:
            {modalite: ağırlık} sözlüğü (toplam 1.0)
        """
        merged: Dict[str, float] = {}
        for name, value in zip(names, values.tolist()):
            base = str(name).split("#")[0]
            merged[base] = merged.get(base, 0.0) + float(value)
        total = sum(merged.values())
        if total > 1e-9:
            merged = {key: value / total for key, value in merged.items()}
        rounded = {key: round(float(value), 6) for key, value in merged.items()}
        if rounded:
            residual = 1.0 - sum(rounded.values())
            if residual:
                largest = max(rounded, key=lambda key: rounded[key])
                rounded[largest] = round(rounded[largest] + residual, 9)
        return rounded

    def _parts(self, chunks: Sequence[EncodedChunk], names: Sequence[str]
               ) -> Dict[str, List[float]]:
        """Her chunk'ın hizalanmış vektörünü isim→vektör sözlüğü olarak döndürür."""
        parts: Dict[str, List[float]] = {}
        for chunk, name in zip(chunks, names):
            parts[name] = self.align_dim(chunk.vector, self._embedding_dim)
        return parts

    @staticmethod
    def _meta(chunks: Sequence[EncodedChunk], extra: Dict[str, Any]) -> Dict[str, Any]:
        """Füzyon meta bilgisini kurar (güvenler, boyutlar, ek alanlar)."""
        meta: Dict[str, Any] = {
            "n_chunks": len(chunks),
            "confidences": [round(float(chunk.confidence), 6) for chunk in chunks],
            "input_dims": [len(chunk.vector) for chunk in chunks],
            "modalities": [chunk.modality.value for chunk in chunks],
        }
        meta.update(extra)
        return meta

    def _single(self, chunk: EncodedChunk, strategy: str) -> FusedRepresentation:
        """
        Tek chunk durumu: tüm stratejiler için vektör aynen döner, ağırlık 1.0.

        Args:
            chunk: Tek geçerli chunk
            strategy: İstenen strateji (bilgi amaçlı meta'ya yazılır)

        Returns:
            `FusedRepresentation` (weights = {modalite: 1.0})
        """
        vector = self.align_dim(chunk.vector, self._embedding_dim)
        name = chunk.modality.value
        meta = {
            "n_chunks": 1,
            "single_chunk": True,
            "requested_strategy": strategy,
            "confidence": round(float(chunk.confidence), 6),
            "confidence_gate": round(
                max(1e-3, float(chunk.confidence))
                * max(1e-3, self.get_modality_prior(chunk.modality)), 6),
            "embedding_dim": self._embedding_dim,
        }
        attach_native(meta, vector)
        return FusedRepresentation(
            vector=vector,
            modalities=[chunk.modality],
            weights={name: 1.0},
            strategy=strategy,
            parts={name: vector},
            meta=meta,
        )

    @staticmethod
    def _softmax(scores: "np.ndarray", axis: int = 1) -> "np.ndarray":
        """
        Sayısal olarak kararlı softmax.

        Args:
            scores: Ham skor matrisi
            axis: Normalizasyon ekseni

        Returns:
            Olasılık matrisi (satırlar toplamı 1)
        """
        shifted = scores - np.max(scores, axis=axis, keepdims=True)
        exponentials = np.exp(shifted)
        totals = np.sum(exponentials, axis=axis, keepdims=True)
        return exponentials / np.maximum(totals, 1e-12)

    def _matrix(self, in_dim: int, out_dim: int, kind: str) -> "np.ndarray":
        """
        Füzyon projeksiyon matrisini (önbellekli) döndürür.

        Args:
            in_dim: Giriş boyutu
            out_dim: Çıkış boyutu
            kind: "q" | "k" | "v" | "o" | "early" (tuz ayrımı)

        Returns:
            Şekil (in_dim, out_dim) float64 matris
        """
        key = (kind, int(in_dim), int(out_dim))
        matrix = self._matrix_cache.get(key)
        if matrix is None:
            matrix = projection_matrix(int(in_dim), int(out_dim), FUSION_SEED,
                                       f"{FUSION_SALT}-{kind}")
            self._matrix_cache[key] = matrix
        return matrix

    @staticmethod
    def _read_embedding_dim(config: Any) -> int:
        """`config.neural.embedding_dim` değerini güvenle okur."""
        if config is None:
            return DEFAULT_EMBEDDING_DIM
        try:
            return int(config.neural.embedding_dim)
        except Exception:
            return DEFAULT_EMBEDDING_DIM

    @staticmethod
    def _read_strategy(config: Any) -> str:
        """`config.multimodal.fusion_strategy` değerini güvenle okur."""
        if config is None:
            return "attention"
        try:
            return str(config.multimodal.fusion_strategy or "attention").strip().lower()
        except Exception:
            return "attention"

    # ─── İstatistikler ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        JSON-serileştirilebilir istatistikleri döndürür.

        Returns:
            Füzyon sayaçları, strateji dağılımı, önsel ağırlıklar, önbellek
        """
        calls = max(1, self._stats["fuse_calls"])
        return {
            "name": self.name,
            "initialized": self._initialized,
            "embedding_dim": self._embedding_dim,
            "default_strategy": self._default_strategy,
            "strategies": list(FUSION_STRATEGIES),
            "training": self._training,
            "modality_priors": self.get_priors(),
            "counters": dict(self._stats),
            "strategy_distribution": dict(self._strategy_counts),
            "avg_chunks_per_fuse": round(self._stats["chunks_fused"] / calls, 2),
            "matrix_cache_entries": len(self._matrix_cache),
            "fusion_seed": FUSION_SEED,
            "fusion_salt": FUSION_SALT,
        }


__all__ = ["FusionLayer", "FUSION_STRATEGIES", "DEFAULT_MODALITY_PRIORS",
           "FUSION_SEED", "FUSION_SALT"]
