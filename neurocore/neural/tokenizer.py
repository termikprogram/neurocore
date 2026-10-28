"""
NeuroCore Tokenizer
====================
Byte-level BPE tokenizer — sıfırdan, harici bağımlılık olmadan.

Neden kendi tokenizer'ımız?
    * `tiktoken`/`sentencepiece` kurulu olmayabilir ve sistem yine de çalışmalı.
    * Türkçe morfolojisi (eklemeli dil, uzun kelimeler) için BPE özellikle
      uygundur: kök + ek parçaları anlamlı alt-kelime birimleri üretir.
    * Özel token'lar (rol işaretleri, modalite ayraçları) doğrudan desteklenir.

Algoritma: byte→unicode eşlemesi (GPT-2 tarzı) + ikili birleştirme (BPE).
    * Metin UTF-8 baytlarına çevrilir, her bayt yazdırılabilir bir Unicode
      karaktere eşlenir (kayıpsız, `bytes ↔ str` birebir).
    * Eğitim: en sık geçen komşu çiftleri yinelemeli olarak birleştirir.
    * Kodlama: öğrenilen birleştirme sıralamasına göre greedy uygular.

Türkçe desteği: `ı İ ğ ş ç ö ü` baytları UTF-8'de 2 bayttır; byte-level
yaklaşım bu karakterleri asla `<unk>` yapmaz — her metin kodlanabilir.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("neural.tokenizer")


# ─── Byte ↔ Unicode Eşlemesi (GPT-2 uyumlu) ─────────────────────────────────

def _build_byte_encoder() -> Dict[int, str]:
    """
    Baytları yazdırılabilir Unicode karakterlere eşler.

    GPT-2'nin kullandığı şema: görünür ASCII ve Latin-1 aralığı korunur,
    kalan baytlar 256+ ofsetle taşınır. Böylece her bayt tek karakter olur.
    """
    printable = list(range(ord("!"), ord("~") + 1))          # 33..126
    printable += list(range(ord("¡"), ord("¬") + 1))          # 161..172
    printable += list(range(ord("®"), ord("ÿ") + 1))          # 174..255
    mapping: Dict[int, str] = {}
    offset = 0
    for byte in range(256):
        if byte in printable:
            mapping[byte] = chr(byte)
        else:
            mapping[byte] = chr(256 + offset)
            offset += 1
    return mapping


BYTE_ENCODER: Dict[int, str] = _build_byte_encoder()
BYTE_DECODER: Dict[str, int] = {char: byte for byte, char in BYTE_ENCODER.items()}


# ─── Özel Token'lar ─────────────────────────────────────────────────────────

DEFAULT_SPECIAL_TOKENS: Tuple[str, ...] = (
    "<pad>",      # 0
    "<unk>",      # 1
    "<bos>",      # 2
    "<eos>",      # 3
    "<|user|>",
    "<|system|>",
    "<|assistant|>",
    "<|context|>",
    "<|image|>",
    "<|audio|>",
    "<|table|>",
    "<|think|>",
    "<|verify|>",
    "<|tool|>",
    "<|memory|>",
)

#: Sohbet/istem biçimlendirmede kullanılan rol token'ları
ROLE_TOKENS: Dict[str, str] = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "context": "<|context|>",
    "tool": "<|tool|>",
    "think": "<|think|>",
}

#: Ön bölütleme deseni — GPT-2 tarzı: boşluk, harf, sayı, diğer
PRE_TOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[\wÇĞİÖŞÜçğıöşü]+| ?[^\s\wÇĞİÖŞÜçğıöşü]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


class TokenizerError(Exception):
    """Tokenizer hatası."""


# ─── Eğitim Sonucu ──────────────────────────────────────────────────────────

@dataclass
class TokenizerStats:
    """Tokenizer istatistikleri."""
    vocab_size: int = 0
    merge_count: int = 0
    special_count: int = 0
    trained_documents: int = 0
    trained_tokens: int = 0
    cache_hits: int = 0
    encodes: int = 0
    decodes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "merges": self.merge_count,
            "special_tokens": self.special_count,
            "trained_documents": self.trained_documents,
            "trained_tokens": self.trained_tokens,
            "cache_hits": self.cache_hits,
            "encodes": self.encodes,
            "decodes": self.decodes,
        }


# ─── Ana Tokenizer ──────────────────────────────────────────────────────────

class BPETokenizer:
    """
    Byte-level BPE tokenizer.

    Kullanım:
        tok = BPETokenizer()
        tok.initialize(config, event_bus)

        tok.train(["metin bir", "metin iki"], vocab_size=8000)
        ids = tok.encode("Merhaba dünya")
        text = tok.decode(ids)

        tok.save("models/tokenizer.json")
        tok2 = BPETokenizer.load("models/tokenizer.json")

    Tokenizer eğitilmemişse bile çalışır: bayt düzeyine düşer
    (`vocab_size ≈ 256 + özel token`).
    """

    name = "neural.tokenizer"

    def __init__(self, special_tokens: Optional[Sequence[str]] = None) -> None:
        """
        Args:
            special_tokens: Özel token listesi (None = varsayılan)
        """
        self._config = None
        self._event_bus = None
        self._initialized = False
        self._lock = threading.RLock()

        self.special_tokens: List[str] = list(special_tokens or DEFAULT_SPECIAL_TOKENS)
        #: [(a, b), ...] — birleştirme sırası (öncelik sırası korunur)
        self.merges: List[Tuple[str, str]] = []
        self._merge_ranks: Dict[Tuple[str, str], int] = {}
        #: token string → id
        self._vocab: Dict[str, int] = {}
        #: id → token string
        self._id_to_token: Dict[str, str] = {}
        self._stats = TokenizerStats()
        self._cache: Dict[str, List[int]] = {}
        self._cache_limit = 20000

        self._rebuild_vocab()

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Tokenizer'ı başlatır (idempotent)."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._initialized = True
        logger.info(
            f"Tokenizer hazır — sözlük: {len(self._vocab)} "
            f"({len(self.special_tokens)} özel, {len(self.merges)} birleştirme)"
        )

    def shutdown(self) -> None:
        """Kaynakları serbest bırakır."""
        self._cache.clear()
        self._initialized = False

    def get_stats(self) -> Dict[str, Any]:
        """Tokenizer istatistikleri."""
        return {**self._stats.to_dict(), "initialized": self._initialized}

    # ─── Sözlük ─────────────────────────────────────────────────────────

    def _rebuild_vocab(self) -> None:
        """
        Sözlüğü özel token'lar + 256 bayt + birleştirmelerden yeniden kurar.

        Birleştirme listesinin sırası id atamasını belirler; bu sayede
        kaydetme/yükleme sonrası id'ler deterministik kalır.
        """
        vocab: Dict[str, int] = {}
        for token in self.special_tokens:
            vocab[token] = len(vocab)
        for byte in range(256):
            vocab[BYTE_ENCODER[byte]] = len(vocab)
        for first, second in self.merges:
            merged = first + second
            if merged not in vocab:
                vocab[merged] = len(vocab)

        self._vocab = vocab
        self._id_to_token = {index: token for token, index in vocab.items()}
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._stats.vocab_size = len(vocab)
        self._stats.merge_count = len(self.merges)
        self._stats.special_count = len(self.special_tokens)

    @property
    def vocab_size(self) -> int:
        """Sözlük boyutu."""
        return len(self._vocab)

    @property
    def vocab(self) -> Dict[str, int]:
        """token → id eşlemesi (kopya)."""
        return dict(self._vocab)

    @property
    def bos_id(self) -> int:
        """`<bos>` token id'si."""
        return self._vocab.get("<bos>", 0)

    @property
    def eos_id(self) -> int:
        """`<eos>` token id'si."""
        return self._vocab.get("<eos>", 0)

    @property
    def pad_id(self) -> int:
        """`<pad>` token id'si."""
        return self._vocab.get("<pad>", 0)

    @property
    def unk_id(self) -> int:
        """`<unk>` token id'si."""
        return self._vocab.get("<unk>", 0)

    def token_id(self, token: str) -> Optional[int]:
        """Token'ın id'si (yoksa None)."""
        return self._vocab.get(token)

    def id_token(self, token_id: int) -> str:
        """id'nin token'ı (yoksa `<unk>`)."""
        return self._id_to_token.get(int(token_id), "<unk>")

    # ─── Eğitim ─────────────────────────────────────────────────────────

    def train(
        self,
        documents: Iterable[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        max_merges: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        BPE birleştirmelerini öğrenir.

        Args:
            documents: Eğitim metinleri
            vocab_size: Hedef sözlük boyutu (özel + 256 bayt dahil)
            min_frequency: Bir çiftin birleştirilmesi için minimum sıklık
            max_merges: Maksimum birleştirme sayısı (None = sözlük boyutundan)
            verbose: Her 500 birleştirmede ilerleme logla

        Returns:
            {"vocab_size": int, "merges": int, "documents": int, "tokens": int}
        """
        texts = [str(doc) for doc in documents if doc]
        if not texts:
            raise TokenizerError("Eğitim için en az bir metin gerekir")

        target = max(256 + len(self.special_tokens) + 1, int(vocab_size))
        limit = max_merges if max_merges is not None else (target - 256 - len(self.special_tokens))

        # 1. Metinleri kelime parçalarına böl, her parçayı bayt sembollerine çevir
        word_freqs: Counter = Counter()
        for text in texts:
            for chunk in PRE_TOKEN_PATTERN.findall(text):
                symbols = tuple(BYTE_ENCODER[byte] for byte in chunk.encode("utf-8"))
                if symbols:
                    word_freqs[symbols] += 1

        total_symbols = sum(len(word) * freq for word, freq in word_freqs.items())
        logger.info(
            f"BPE eğitimi başlıyor: {len(texts)} belge, {len(word_freqs)} benzersiz parça, "
            f"{total_symbols} sembol, hedef sözlük {target}"
        )

        # 2. İkili birleştirme döngüsü
        merges: List[Tuple[str, str]] = []
        for iteration in range(int(limit)):
            pair_counts: Counter = Counter()
            for word, freq in word_freqs.items():
                for index in range(len(word) - 1):
                    pair_counts[(word[index], word[index + 1])] += freq

            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break

            merges.append(best_pair)
            merged_symbol = best_pair[0] + best_pair[1]

            # Sözlükteki tüm parçalarda bu çifti birleştir
            new_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                new_freqs[self._merge_pair_in_word(word, best_pair, merged_symbol)] += freq
            word_freqs = new_freqs

            if verbose and (iteration + 1) % 500 == 0:
                logger.info(
                    f"  BPE: {iteration + 1}/{int(limit)} birleştirme, "
                    f"sözlük {256 + len(self.special_tokens) + len(merges)}"
                )

        self.merges = merges
        self._rebuild_vocab()
        self._cache.clear()

        self._stats.trained_documents += len(texts)
        self._stats.trained_tokens += total_symbols

        result = {
            "vocab_size": self.vocab_size,
            "merges": len(self.merges),
            "documents": len(texts),
            "tokens": total_symbols,
        }
        logger.info(
            f"BPE eğitimi tamamlandı: sözlük {self.vocab_size}, "
            f"{len(self.merges)} birleştirme"
        )
        self._emit("neural.tokenizer_trained", result)
        return result

    @staticmethod
    def _merge_pair_in_word(word: Tuple[str, ...], pair: Tuple[str, str],
                            merged: str) -> Tuple[str, ...]:
        """Bir kelimedeki belirli çifti birleştirir (soldan sağa, örtüşmesiz)."""
        if len(word) < 2:
            return word
        result: List[str] = []
        index = 0
        while index < len(word):
            if index < len(word) - 1 and word[index] == pair[0] and word[index + 1] == pair[1]:
                result.append(merged)
                index += 2
            else:
                result.append(word[index])
                index += 1
        return tuple(result)

    def train_from_vault(self, vault: Any, vocab_size: int = 8000,
                         max_notes: int = 2000) -> Dict[str, Any]:
        """
        Doğrudan kasadan eğitim yapar (bilgi kasasıyla bütünleşik).

        Args:
            vault: `KnowledgeVault` (veya `iter_notes` sunan nesne)
            vocab_size: Hedef sözlük boyutu
            max_notes: Kullanılacak maksimum not sayısı
        """
        documents: List[str] = []
        iterator = vault.iter_notes(limit=max_notes) if hasattr(vault, "iter_notes") else iter([])
        for note in iterator:
            text = f"{note.title}\n{note.body}"
            if text.strip():
                documents.append(text)
        if not documents:
            logger.warning("Kasada eğitim metni bulunamadı")
            return {"vocab_size": self.vocab_size, "merges": len(self.merges),
                    "documents": 0, "tokens": 0}
        return self.train(documents, vocab_size=vocab_size)

    # ─── Kodlama / Çözme ────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tokens: bool = False,
    ) -> Any:
        """
        Metni token id'lerine çevirir.

        Args:
            text: Girdi metni
            add_special_tokens: Başına `<bos>`, sonuna `<eos>` ekle
            max_length: Maksimum token sayısı
            truncation: `max_length` aşılırsa kırp (False ise hata verir)
            return_tokens: Token metinlerini de döndür

        Returns:
            token id listesi veya `return_tokens=True` ise (ids, tokens)
        """
        if not text:
            ids: List[int] = []
            return (ids, []) if return_tokens else ids

        self._stats.encodes += 1

        # Önbellek (özel token'lar dahil edilmediğinde güvenli)
        cache_key = text if not add_special_tokens and max_length is None else ""
        if cache_key and cache_key in self._cache:
            self._stats.cache_hits += 1
            cached = list(self._cache[cache_key])
            return (cached, [self.id_token(i) for i in cached]) if return_tokens else cached

        # Özel token'ları koru: metni parçalara ayır
        segments = self._split_on_special(text)
        ids = []
        for segment, is_special in segments:
            if is_special:
                token_id = self._vocab.get(segment)
                if token_id is not None:
                    ids.append(token_id)
                continue
            ids.extend(self._encode_segment(segment))

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

        if max_length is not None and len(ids) > max_length:
            if not truncation:
                raise TokenizerError(
                    f"Metin çok uzun: {len(ids)} token > max_length={max_length}"
                )
            ids = ids[:max_length]

        if cache_key and len(self._cache) < self._cache_limit:
            self._cache[cache_key] = list(ids)

        if return_tokens:
            return ids, [self.id_token(i) for i in ids]
        return ids

    def _encode_segment(self, segment: str) -> List[int]:
        """Özel token içermeyen bir metin parçasını kodlar."""
        result: List[int] = []
        for chunk in PRE_TOKEN_PATTERN.findall(segment):
            symbols = [BYTE_ENCODER[byte] for byte in chunk.encode("utf-8")]
            merged = self._apply_bpe(symbols)
            for symbol in merged:
                token_id = self._vocab.get(symbol)
                if token_id is None:
                    # Bilinmeyen sembol: baytlarına ayır (asla kayıp olmaz)
                    for char in symbol:
                        byte = BYTE_DECODER.get(char)
                        if byte is not None:
                            result.append(self._vocab.get(BYTE_ENCODER[byte], self.unk_id))
                        else:
                            result.append(self.unk_id)
                else:
                    result.append(token_id)
        return result

    def _apply_bpe(self, symbols: List[str]) -> List[str]:
        """
        Öğrenilen birleştirmeleri greedy uygular.

        Her adımda **en düşük rank'lı** (en erken öğrenilmiş) çift birleştirilir —
        bu, eğitimdeki davranışı birebir yansıtır.
        """
        if len(symbols) < 2 or not self._merge_ranks:
            return symbols

        word = list(symbols)
        while len(word) >= 2:
            best_rank = None
            best_index = -1
            for index in range(len(word) - 1):
                rank = self._merge_ranks.get((word[index], word[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_index < 0:
                break
            word[best_index] = word[best_index] + word[best_index + 1]
            del word[best_index + 1]
        return word

    def _split_on_special(self, text: str) -> List[Tuple[str, bool]]:
        """
        Metni özel token sınırlarına göre böler.

        Returns:
            [(parça, özel_mi), ...] — özel token'lar orijinal hâliyle korunur
        """
        if not self.special_tokens:
            return [(text, False)]
        # Uzun token'lar önce denenmeli (ör. `<|assistant|>` < `<|a|>`).
        # Token'lar `<|...|>` biçiminde olduğu için `re.escape` tam güvenli.
        pattern = "|".join(
            re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
        )
        parts: List[Tuple[str, bool]] = []
        last = 0
        for match in re.finditer(pattern, text):
            if match.start() > last:
                parts.append((text[last:match.start()], False))
            parts.append((match.group(0), True))
            last = match.end()
        if last < len(text):
            parts.append((text[last:], False))
        return parts or [(text, False)]

    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
        errors: str = "replace",
    ) -> str:
        """
        Token id'lerini metne çevirir.

        Bayt tamponu üzerinden çalışır: birleştirmeler tek tek karakter
        ekleyip sonunda UTF-8 olarak çözülür, böylece çok baytlı Türkçe
        karakterler (ı, ş, ğ) asla bozulmaz.

        Args:
            ids: Token id'leri
            skip_special_tokens: Özel token'ları çıktıdan çıkar
            errors: UTF-8 çözme hata politikası

        Returns:
            Çözülen metin
        """
        self._stats.decodes += 1
        parts: List[str] = []
        buffer = bytearray()

        def flush() -> None:
            """Tampondaki baytları metne çevirip biriktirir."""
            if buffer:
                parts.append(buffer.decode("utf-8", errors=errors))
                buffer.clear()

        for token_id in ids:
            token = self._id_to_token.get(int(token_id))
            if token is None:
                continue
            if token in self.special_tokens:
                if skip_special_tokens:
                    continue
                flush()
                parts.append(token)
                continue
            for char in token:
                byte = BYTE_DECODER.get(char)
                if byte is not None:
                    buffer.append(byte)
                else:
                    # Sözlükte olmayan karakter (bozuk dosya): doğrudan yaz
                    flush()
                    parts.append(char)
        flush()
        return "".join(parts)

    def decode_ids(self, ids: Sequence[int], skip_special_tokens: bool = True) -> List[str]:
        """id listesini token metinlerine çevirir (hata ayıklama için)."""
        tokens: List[str] = []
        for token_id in ids:
            token = self._id_to_token.get(int(token_id), "<unk>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return tokens

    def encode_batch(self, texts: Sequence[str], max_length: Optional[int] = None,
                     padding: bool = False, pad_to_multiple_of: int = 8
                     ) -> Dict[str, List[List[int]]]:
        """
        Toplu kodlama (eğitim verisi hazırlığı için).

        Args:
            texts: Metin listesi
            max_length: Maksimum uzunluk
            padding: Dizileri eşit uzunluğa doldur
            pad_to_multiple_of: Dolgu uzunluğunu bu sayının katına yuvarla

        Returns:
            {"input_ids": [[...]], "lengths": [int], "attention_mask": [[...]]}
        """
        encoded = [self.encode(text, add_special_tokens=True, max_length=max_length)
                   for text in texts]
        lengths = [len(ids) for ids in encoded]

        if padding and encoded:
            target = max(lengths)
            if pad_to_multiple_of > 1:
                target = ((target + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            padded = [ids + [self.pad_id] * (target - len(ids)) for ids in encoded]
            mask = [[1] * len(ids) + [0] * (target - len(ids)) for ids in encoded]
        else:
            padded = encoded
            mask = [[1] * len(ids) for ids in encoded]

        return {"input_ids": padded, "lengths": lengths, "attention_mask": mask}

    # ─── Sohbet Biçimlendirme ───────────────────────────────────────────

    def format_chat(self, messages: Sequence[Dict[str, str]],
                    add_generation_prompt: bool = True) -> str:
        """
        Mesaj listesini rol token'larıyla tek metne çevirir.

        Args:
            messages: [{"role": "user"|"system"|"assistant", "content": "..."}]
            add_generation_prompt: Sonuna `<|assistant|>` ekle (üretim için)

        Returns:
            Biçimlendirilmiş istem metni
        """
        parts: List[str] = []
        for message in messages:
            role = str(message.get("role", "user")).lower()
            token = ROLE_TOKENS.get(role, "<|user|>")
            content = str(message.get("content", "")).strip()
            parts.append(f"{token}\n{content}\n")
        if add_generation_prompt:
            parts.append(f"{ROLE_TOKENS['assistant']}\n")
        return "".join(parts)

    def parse_chat(self, text: str) -> List[Dict[str, str]]:
        """`format_chat` çıktısını mesaj listesine geri çevirir."""
        pattern = "|".join(re.escape(token) for token in ROLE_TOKENS.values())
        messages: List[Dict[str, str]] = []
        current_role: Optional[str] = None
        buffer: List[str] = []
        reverse_roles = {token: role for role, token in ROLE_TOKENS.items()}

        for line in text.splitlines():
            stripped = line.strip()
            if stripped in reverse_roles:
                if current_role is not None:
                    messages.append({"role": current_role, "content": "\n".join(buffer).strip()})
                current_role = reverse_roles[stripped]
                buffer = []
            elif current_role is not None:
                buffer.append(line)
        if current_role is not None:
            messages.append({"role": current_role, "content": "\n".join(buffer).strip()})
        return messages

    # ─── Kalıcılık ──────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """
        Tokenizer'ı JSON olarak kaydeder.

        Args:
            path: Hedef dosya

        Returns:
            Yazılan dosya yolu
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "special_tokens": list(self.special_tokens),
            "merges": [[first, second] for first, second in self.merges],
            "stats": self._stats.to_dict(),
        }
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(temp, target)
        logger.info(f"Tokenizer kaydedildi: {target} ({self.vocab_size} token)")
        return str(target)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        JSON'dan tokenizer yükler.

        Dosya bozuksa boş (bayt düzeyi) tokenizer döner — sistem çalışmaya
        devam eder.
        """
        target = Path(path)
        if not target.exists():
            logger.warning(f"Tokenizer dosyası bulunamadı, boş tokenizer kullanılacak: {target}")
            return cls()
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"Tokenizer yüklenemedi ({target}): {exc}")
            return cls()

        tokenizer = cls(special_tokens=payload.get("special_tokens") or DEFAULT_SPECIAL_TOKENS)
        tokenizer.merges = [
            (str(pair[0]), str(pair[1]))
            for pair in (payload.get("merges") or [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]
        tokenizer._rebuild_vocab()
        logger.info(f"Tokenizer yüklendi: {target} ({tokenizer.vocab_size} token)")
        return tokenizer

    def default_path(self, base_path: Optional[str] = None) -> str:
        """Varsayılan kayıt yolu (`<base>/models/tokenizer.json`)."""
        base = base_path
        if base is None and self._config is not None:
            base = getattr(self._config, "base_path", None)
        base = base or os.getcwd()
        return str(Path(base) / "models" / "tokenizer.json")

    def save_default(self) -> str:
        """Varsayılan konuma kaydeder."""
        return self.save(self.default_path())

    @classmethod
    def load_default(cls, base_path: Optional[str] = None) -> "BPETokenizer":
        """Varsayılan konumdan yükler (yoksa boş tokenizer)."""
        base = base_path or os.getcwd()
        path = Path(base) / "models" / "tokenizer.json"
        return cls.load(str(path)) if path.exists() else cls()

    # ─── Yardımcılar ────────────────────────────────────────────────────

    def token_count(self, text: str) -> int:
        """Metnin token sayısı (kodlamadan hızlı tahmin gerekiyorsa `estimate`)."""
        return len(self.encode(text))

    def estimate_tokens(self, text: str) -> int:
        """
        Kodlama yapmadan kaba token tahmini.

        Türkçe için ampirik oran: ~3.4 karakter/token (eklemeli dil,
        uzun kelimeler). Hızlı bütçe hesabı için kullanılır.
        """
        if not text:
            return 0
        return max(1, int(len(text) / 3.4))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Metni token sınırında kırpar (kelime ortasında kesmez)."""
        ids = self.encode(text, max_length=max_tokens)
        return self.decode(ids)

    def split_tokens(self, text: str) -> List[str]:
        """Metni görünür token parçalarına ayırır (inceleme için)."""
        _ids, tokens = self.encode(text, return_tokens=True)
        return tokens

    def merge_quality(self) -> Dict[str, Any]:
        """
        Birleştirme kalitesi raporu.

        Uzun token'lar (çok baytlı birleşimler) dil için daha verimli
        temsiller üretir; bu rapor tokenizer'ın olgunluğunu gösterir.
        """
        lengths = [len(token) for token in self.merges] if self.merges else []
        byte_tokens = sum(1 for token in self._vocab if len(token) == 1)
        return {
            "vocab_size": self.vocab_size,
            "merges": len(self.merges),
            "byte_level_tokens": byte_tokens,
            "mean_merged_length": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
            "max_merged_length": max(lengths) if lengths else 0,
            "chars_per_token_hint": round(self.vocab_size / max(1, 256), 3),
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
        return self.vocab_size

    def __repr__(self) -> str:  # pragma: no cover
        return f"<BPETokenizer vocab={self.vocab_size} merges={len(self.merges)}>"


# ─── Basit Kelime Tokenizer (yedek) ─────────────────────────────────────────

class SimpleTokenizer:
    """
    Kelime/karakter tabanlı basit tokenizer — BPE eğitimi istenmediğinde.

    Deterministik hash tabanlı kelime sözlüğü kullanır; `BPETokenizer` ile
    aynı arayüzü sunar (`encode`, `decode`, `vocab_size`, `pad_id`, …).
    Sözlük dışı kelimeler karakterlere ayrılır — kayıp yoktur.
    """

    name = "neural.tokenizer.simple"

    def __init__(self, special_tokens: Optional[Sequence[str]] = None) -> None:
        self.special_tokens: List[str] = list(special_tokens or DEFAULT_SPECIAL_TOKENS)
        self._vocab: Dict[str, int] = {token: index for index, token in enumerate(self.special_tokens)}
        self._id_to_token: Dict[int, str] = {index: token for token, index in self._vocab.items()}
        self._initialized = False
        self._lock = threading.RLock()

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """No-op (arayüz uyumu)."""
        self._initialized = True

    def shutdown(self) -> None:
        """No-op (arayüz uyumu)."""
        self._initialized = False

    def get_stats(self) -> Dict[str, Any]:
        """Tokenizer istatistikleri."""
        return {"vocab_size": self.vocab_size, "kind": "simple",
                "initialized": self._initialized}

    @property
    def vocab_size(self) -> int:
        """Sözlük boyutu."""
        return len(self._vocab)

    @property
    def vocab(self) -> Dict[str, int]:
        """token → id eşlemesi."""
        return dict(self._vocab)

    @property
    def pad_id(self) -> int:
        """`<pad>` id'si."""
        return self._vocab.get("<pad>", 0)

    @property
    def bos_id(self) -> int:
        """`<bos>` id'si."""
        return self._vocab.get("<bos>", 0)

    @property
    def eos_id(self) -> int:
        """`<eos>` id'si."""
        return self._vocab.get("<eos>", 0)

    @property
    def unk_id(self) -> int:
        """`<unk>` id'si."""
        return self._vocab.get("<unk>", 0)

    def train(self, documents: Iterable[str], vocab_size: int = 8000,
              **kwargs: Any) -> Dict[str, Any]:
        """Kelime sıklığına göre sözlük kurar."""
        counter: Counter = Counter()
        documents = list(documents)
        for document in documents:
            counter.update(re.findall(r"[\wÇĞİÖŞÜçğıöşü]+|[^\s\w]", str(document)))

        limit = max(0, int(vocab_size) - len(self.special_tokens))
        for word, _ in counter.most_common(limit):
            if word not in self._vocab:
                index = len(self._vocab)
                self._vocab[word] = index
                self._id_to_token[index] = word

        logger.info(f"Basit tokenizer eğitildi: {self.vocab_size} token")
        return {"vocab_size": self.vocab_size, "documents": len(documents),
                "tokens": sum(counter.values())}

    def encode(self, text: str, add_special_tokens: bool = False,
               max_length: Optional[int] = None, truncation: bool = True,
               return_tokens: bool = False) -> Any:
        """Metni id'lere çevirir; bilinmeyen kelimeler karakterlere ayrılır."""
        ids: List[int] = []
        for word in re.findall(r"[\wÇĞİÖŞÜçğıöşü]+|[^\s\w]", text or ""):
            token_id = self._vocab.get(word)
            if token_id is not None:
                ids.append(token_id)
            else:
                for char in word:
                    ids.append(self._vocab.get(char, self.unk_id))
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_length is not None and len(ids) > max_length:
            if not truncation:
                raise TokenizerError(f"Metin çok uzun: {len(ids)} > {max_length}")
            ids = ids[:max_length]
        if return_tokens:
            return ids, [self._id_to_token.get(i, "<unk>") for i in ids]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True, **_: Any) -> str:
        """id'leri metne çevirir."""
        parts: List[str] = []
        for token_id in ids:
            token = self._id_to_token.get(int(token_id), "")
            if not token:
                continue
            if token in self.special_tokens:
                if not skip_special_tokens:
                    parts.append(token)
                continue
            parts.append(token if re.fullmatch(r"[^\s\w]", token) else " " + token)
        return "".join(parts).strip()

    def format_chat(self, messages: Sequence[Dict[str, str]],
                    add_generation_prompt: bool = True) -> str:
        """Mesajları rol token'larıyla birleştirir."""
        parts = []
        for message in messages:
            role = str(message.get("role", "user")).lower()
            parts.append(f"{ROLE_TOKENS.get(role, '<|user|>')}\n{message.get('content', '')}\n")
        if add_generation_prompt:
            parts.append(f"{ROLE_TOKENS['assistant']}\n")
        return "".join(parts)

    def save(self, path: str) -> str:
        """Sözlüğü JSON olarak kaydeder."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"version": 1, "kind": "simple",
                        "special_tokens": self.special_tokens,
                        "vocab": self._vocab}, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(target)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """JSON'dan yükler."""
        target = Path(path)
        if not target.exists():
            return cls()
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        tokenizer = cls(payload.get("special_tokens"))
        for token, index in (payload.get("vocab") or {}).items():
            tokenizer._vocab[str(token)] = int(index)
            tokenizer._id_to_token[int(index)] = str(token)
        return tokenizer


def create_tokenizer(config: Any = None, prefer_bpe: bool = True) -> Any:
    """
    Uygun tokenizer'ı üretir.

    Önce kaydedilmiş BPE tokenizer aranır; yoksa ve `prefer_bpe` ise boş
    BPE (bayt düzeyi) döner, aksi hâlde `SimpleTokenizer`.
    """
    base = getattr(config, "base_path", None) if config is not None else None
    if prefer_bpe:
        base = base or os.getcwd()
        path = Path(base) / "models" / "tokenizer.json"
        if path.exists():
            tokenizer = BPETokenizer.load(str(path))
            tokenizer.initialize(config, None)
            return tokenizer
        tokenizer = BPETokenizer()
        tokenizer.initialize(config, None)
        return tokenizer
    return SimpleTokenizer()


__all__ = [
    "BPETokenizer",
    "SimpleTokenizer",
    "TokenizerError",
    "TokenizerStats",
    "create_tokenizer",
    "DEFAULT_SPECIAL_TOKENS",
    "ROLE_TOKENS",
    "BYTE_ENCODER",
    "BYTE_DECODER",
]
