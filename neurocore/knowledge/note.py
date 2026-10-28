"""
NeuroCore Not Modeli
=====================
Obsidian uyumlu markdown notlarının veri modeli, frontmatter ayrıştırma,
bağlantı çıkarımı, etiket normalizasyonu ve serileştirme.

Not biçimi (kasada saklanan gerçek dosya):

    ---
    id: kn_7f3a91c2
    title: "GPU Termal Yönetimi"
    kind: concept
    tags: [donanim, gpu, termal]
    confidence: 0.78
    ---

    ## Özet
    ...

Tasarım kararları:
    * Frontmatter **her zaman YAML** olarak yazılır (Obsidian uyumlu).
    * Okuma tarafı YAML bulunamazsa satır bazlı basit ayrıştırıcıya düşer
      (PyYAML kurulu olmasa bile kasa okunabilir kalır).
    * `[[wiki bağlantıları]]` gövdeden otomatik çıkarılır; hem id hem başlık
      biçimini destekler.
    * Bilinmeyen frontmatter alanları `extra` sözlüğünde korunur — kullanıcı
      kendi alanlarını ekleyebilir ve kaybolmaz.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("knowledge.note")


# ─── Sabitler ───────────────────────────────────────────────────────────────

FRONTMATTER_DELIMITER = "---"

#: Not türleri (kind)
NOTE_KINDS: Tuple[str, ...] = (
    "concept", "entity", "source", "project", "session",
    "skill", "journal", "meta", "research", "summary",
)

#: Kasada kullanılan üst kategoriler (dizin eşlemesi)
NOTE_CATEGORIES: Tuple[str, ...] = (
    "system", "learned", "research", "meta", "user",
    "skill", "session", "journal", "project", "source", "concept", "entity",
)

_ID_RE = re.compile(r"^kn_[0-9a-f]{8}$")
_WIKILINK_RE = re.compile(r"\[\[([^\[\]|]+?)(?:\|([^\[\]]+?))?\]\]")
_MD_LINK_RE = re.compile(r"(?<!!)\[([^\[\]]+)\]\(([^()\s]+)\)")
#: Wiki/markdown bağlantı blokları — satır içi etiket ararken bu bölgeler atlanır
_LINK_BLOCK_RE = re.compile(r"\[\[[^\[\]]*\]\]|\[[^\[\]]*\]\([^()\s]*\)")
_TAG_INLINE_RE = re.compile(r"(?<![\w/])#([A-Za-zÇĞİÖŞÜçğıöşü0-9_][\w\-/ÇĞİÖŞÜçğıöşü]*)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|belirsiz|bilinmiyor|\?\?\?)\b", re.IGNORECASE)

#: Türkçe karakter katlama tablosu (etiket/slug normalizasyonu için)
_TR_FOLD = str.maketrans({
    "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ı": "i", "İ": "i",
    "ö": "o", "Ö": "o", "ş": "s", "Ş": "s", "ü": "u", "Ü": "u",
    "â": "a", "Â": "a", "î": "i", "Î": "i", "û": "u", "Û": "u",
})


# ─── Yardımcı Fonksiyonlar ──────────────────────────────────────────────────

def now_iso() -> str:
    """Yerel saat diliminde ISO-8601 zaman damgası."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def utc_timestamp() -> float:
    """POSIX zaman damgası (float)."""
    return datetime.now(timezone.utc).timestamp()


def parse_timestamp(value: Any) -> float:
    """
    Frontmatter'dan gelen çeşitli zaman biçimlerini float'a çevirir.

    Desteklenen: ISO-8601 string, unix timestamp (int/float), datetime nesnesi.
    Ayrıştırılamazsa 0.0 döner.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        # Saniye cinsinden makul aralık
        return float(value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        # Sadece sayı ise unix timestamp kabul et
        try:
            return float(text)
        except ValueError:
            pass
        candidate = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                return 0.0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return 0.0


def iso_from_timestamp(ts: float) -> str:
    """Float zaman damgasını yerel ISO-8601 metnine çevirir."""
    if not ts:
        return now_iso()
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def fold_turkish(text: str) -> str:
    """Türkçe karakterleri ASCII karşılığına katlar (arama ve slug için)."""
    return text.translate(_TR_FOLD)


def slugify(text: str, max_length: int = 80) -> str:
    """
    Başlıktan dosya adı üretir: küçük harf, Türkçe katlanmış, tiresi ayraç.

    >>> slugify("GPU Termal Yönetimi — Nedir?")
    'gpu-termal-yonetimi-nedir'
    """
    folded = fold_turkish(text).lower()
    folded = re.sub(r"[^a-z0-9]+", "-", folded)
    folded = folded.strip("-")
    if len(folded) > max_length:
        folded = folded[:max_length].rstrip("-")
    return folded or "not"


def normalize_tag(tag: str) -> str:
    """
    Etiketi tek biçime getirir: küçük harf, boşluk→tire, `#` yok, Türkçe katlanmış.

    >>> normalize_tag("  #Yapay Zeka ")
    'yapay-zeka'
    """
    if not tag:
        return ""
    cleaned = str(tag).strip().lstrip("#").strip()
    cleaned = fold_turkish(cleaned).lower()
    cleaned = re.sub(r"[\s_]+", "-", cleaned)
    cleaned = re.sub(r"[^a-z0-9\-/\.]", "", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned


def normalize_tags(tags: Iterable[Any]) -> List[str]:
    """Etiket listesini normalize eder, tekrarları atar, sırayı korur."""
    seen: set = set()
    result: List[str] = []
    for raw in tags or []:
        for part in str(raw).split(","):
            tag = normalize_tag(part)
            if tag and tag not in seen:
                seen.add(tag)
                result.append(tag)
    return result


def new_note_id() -> str:
    """Benzersiz not kimliği üretir (`kn_` + 8 hex)."""
    import uuid
    return f"kn_{uuid.uuid4().hex[:8]}"


def is_valid_note_id(value: str) -> bool:
    """`kn_xxxxxxxx` biçiminde mi?"""
    return bool(value) and bool(_ID_RE.match(value))


def content_hash(*parts: str) -> str:
    """İçerik parmak izi (blake2b, 16 hex)."""
    hasher = hashlib.blake2b(digest_size=8)
    for part in parts:
        hasher.update((part or "").encode("utf-8", errors="replace"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()


# ─── Basit YAML Ayrıştırıcı (PyYAML yedeği) ─────────────────────────────────

def _coerce_scalar(raw: str) -> Any:
    """YAML skaler değerini Python tipine çevirir."""
    text = raw.strip()
    if not text:
        return ""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("true", "yes", "on", "evet"):
        return True
    if lowered in ("false", "no", "off", "hayir", "hayır"):
        return False
    if lowered in ("null", "none", "~", ""):
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", text):
            return float(text)
    except ValueError:
        pass
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(part) for part in _split_inline_list(inner)]
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        result: Dict[str, Any] = {}
        if not inner:
            return result
        for part in _split_inline_list(inner):
            if ":" in part:
                key, _, val = part.partition(":")
                result[key.strip().strip("'\"")] = _coerce_scalar(val)
        return result
    return text


def _split_inline_list(inner: str) -> List[str]:
    """`[a, b, "c, d"]` içeriğini tırnak farkındalıklı olarak böler."""
    parts: List[str] = []
    buffer: List[str] = []
    quote: Optional[str] = None
    depth = 0
    for char in inner:
        if quote:
            buffer.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            buffer.append(char)
        elif char in "[{":
            depth += 1
            buffer.append(char)
        elif char in "]}":
            depth -= 1
            buffer.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
    if buffer:
        parts.append("".join(buffer).strip())
    return [part for part in parts if part != ""]


def parse_frontmatter_fallback(text: str) -> Dict[str, Any]:
    """
    PyYAML olmadan frontmatter ayrıştırıcı.

    Basit `key: value` satırları ile `- item` listelerini destekler.
    Karmaşık YAML (iç içe sözlükler) desteklenmez; o durumda PyYAML kullanılır.
    """
    data: Dict[str, Any] = {}
    current_key: Optional[str] = None
    current_list: Optional[List[Any]] = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        stripped = raw_line.strip()

        # Liste öğesi
        if stripped.startswith("- ") and current_key:
            if current_list is None:
                current_list = []
                data[current_key] = current_list
            current_list.append(_coerce_scalar(stripped[2:]))
            continue

        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            current_key = key
            if value == "":
                data[key] = None
                current_list = None
            else:
                data[key] = _coerce_scalar(value)
                current_list = data[key] if isinstance(data[key], list) else None
    return data


def split_frontmatter(text: str) -> Tuple[Dict[str, Any], str, bool]:
    """
    Markdown metnini (frontmatter, gövde, yaml_kullanıldı) üçlüsüne ayırır.

    Frontmatter yoksa `({}, tüm_metin, False)` döner.
    """
    if not text:
        return {}, "", False

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.lstrip().startswith(FRONTMATTER_DELIMITER):
        return {}, normalized, False

    # Baştaki boş satırları atlayıp ilk delimitleri bul
    start_index = normalized.index(FRONTMATTER_DELIMITER)
    after_open = normalized[start_index + len(FRONTMATTER_DELIMITER):]
    # Kapanış delimitleri: satır başında ---
    close_match = re.search(r"^---\s*$", after_open, re.MULTILINE)
    if not close_match:
        return {}, normalized, False

    raw_yaml = after_open[: close_match.start()]
    body = after_open[close_match.end():].lstrip("\n")

    data: Dict[str, Any] = {}
    used_yaml = False
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(raw_yaml)
        if isinstance(loaded, dict):
            data = {str(k): v for k, v in loaded.items()}
            used_yaml = True
        elif loaded is None:
            used_yaml = True
    except ImportError:
        pass
    except Exception as exc:  # bozuk YAML
        logger.warning(f"Frontmatter YAML ayrıştırılamadı, yedek ayrıştırıcı kullanılıyor: {exc}")

    if not used_yaml:
        data = parse_frontmatter_fallback(raw_yaml)

    return data, body, used_yaml


def dump_frontmatter(data: Dict[str, Any]) -> str:
    """Frontmatter sözlüğünü YAML metnine çevirir (Obsidian uyumlu)."""
    if not data:
        return ""
    try:
        import yaml  # type: ignore
        dumped = yaml.dump(
            data, allow_unicode=True, default_flow_style=None,
            sort_keys=False, width=1000,
        )
        return dumped.rstrip("\n")
    except Exception:
        lines: List[str] = []
        for key, value in data.items():
            lines.append(f"{key}: {_dump_scalar(value)}")
        return "\n".join(lines)


def _dump_scalar(value: Any) -> str:
    """Tek bir değeri YAML skaler metnine çevirir."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_dump_scalar(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, dict):
        inner = ", ".join(f"{k}: {_dump_scalar(v)}" for k, v in value.items())
        return f"{{{inner}}}"
    text = str(value)
    if text == "" or re.search(r"[:#\[\]{},&*!|>'\"%@`\n]", text) or text.strip() != text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


# ─── Gövde Analizi ──────────────────────────────────────────────────────────

def extract_wikilinks(body: str) -> List[Tuple[str, str]]:
    """
    Gövdedeki `[[hedef]]` ve `[[hedef|görünen]]` bağlantılarını çıkarır.

    Returns:
        (hedef, görünen_ad) çiftleri; görünen ad yoksa hedef ile aynıdır.
    """
    if not body:
        return []
    results: List[Tuple[str, str]] = []
    for match in _WIKILINK_RE.finditer(body):
        target = (match.group(1) or "").strip()
        label = (match.group(2) or target).strip()
        if target:
            results.append((target, label))
    return results


def extract_markdown_links(body: str) -> List[Tuple[str, str]]:
    """Gövdedeki `[ad](hedef)` bağlantılarını çıkarır."""
    if not body:
        return []
    return [(m.group(1).strip(), m.group(2).strip()) for m in _MD_LINK_RE.finditer(body)]


def extract_inline_tags(body: str) -> List[str]:
    """
    Gövdedeki `#etiket` biçimli satır içi etiketleri çıkarır.

    Kod blokları ve `[[wiki bağlantısı]]` / `[ad](hedef)` bölgeleri atlanır;
    aksi halde `[[#başlık]]` gibi Obsidian başlık bağlantıları yanlışlıkla
    etiket sayılırdı.
    """
    if not body:
        return []
    without_code = _CODE_FENCE_RE.sub(" ", body)
    without_links = _LINK_BLOCK_RE.sub(" ", without_code)
    return [normalize_tag(m.group(1)) for m in _TAG_INLINE_RE.finditer(without_links)]


def extract_headings(body: str) -> List[Dict[str, Any]]:
    """Başlıkları seviye ve metinle birlikte döndürür."""
    if not body:
        return []
    return [
        {"level": len(m.group(1)), "text": m.group(2).strip()}
        for m in _HEADING_RE.finditer(body)
    ]


def extract_section(body: str, heading: str) -> str:
    """Belirli bir başlığın altındaki bölümü döndürür (başlık hariç)."""
    if not body:
        return ""
    pattern = re.compile(
        rf"^#{{1,6}}\s+{re.escape(heading)}\s*$(.*?)(?=^#{{1,6}}\s|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(body)
    return match.group(1).strip() if match else ""


def extract_bullets(body: str) -> List[str]:
    """Gövdedeki madde işaretli ve numaralı liste öğelerini döndürür."""
    if not body:
        return []
    items: List[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if re.match(r"^([-*+]|\d+[.)])\s+\S", stripped):
            items.append(re.sub(r"^([-*+]|\d+[.)])\s+", "", stripped))
    return items


def strip_markdown(body: str) -> str:
    """Arama/gömme için markdown işaretlerini temizler (kod blokları korunur)."""
    if not body:
        return ""
    text = _CODE_FENCE_RE.sub(lambda m: m.group(0)[3:-3], body)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = _WIKILINK_RE.sub(lambda m: (m.group(2) or m.group(1)), text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─── Not Veri Modeli ────────────────────────────────────────────────────────

@dataclass
class Note:
    """
    Tek bir bilgi notu.

    `body` markdown gövdesidir (frontmatter hariç). `extra`, tanınmayan
    frontmatter alanlarını korur; böylece kullanıcı eklediği özel alanları
    kaybetmez.
    """
    # Kimlik ve içerik
    id: str = field(default_factory=new_note_id)
    title: str = ""
    body: str = ""

    # Sınıflandırma
    kind: str = "concept"
    category: str = "learned"
    tags: List[str] = field(default_factory=list)

    # Güven ve önem
    confidence: float = 0.5
    importance: float = 0.5

    # Zaman
    created: str = field(default_factory=now_iso)
    updated: str = field(default_factory=now_iso)
    version: int = 1

    # Kaynak ve ilişki
    sources: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)          # not id'leri
    backlinks: List[str] = field(default_factory=list)      # çalışma zamanında doldurulur

    # Gömme
    embedding_model: str = ""
    embedding: Optional[List[float]] = None

    # Dosya sistemi
    path: str = ""

    # Korunan bilinmeyen alanlar
    extra: Dict[str, Any] = field(default_factory=dict)

    # ─── Türetilmiş Özellikler ──────────────────────────────────────────

    @property
    def word_count(self) -> int:
        """Gövdedeki kelime sayısı."""
        return len(self.body.split())

    @property
    def char_count(self) -> int:
        """Gövde karakter sayısı."""
        return len(self.body)

    @property
    def created_ts(self) -> float:
        """Oluşturulma zamanı (float)."""
        return parse_timestamp(self.created)

    @property
    def updated_ts(self) -> float:
        """Son güncelleme zamanı (float)."""
        return parse_timestamp(self.updated)

    @property
    def age_days(self) -> float:
        """Son güncellemeden bu yana geçen gün sayısı."""
        delta = utc_timestamp() - self.updated_ts
        return max(0.0, delta / 86400.0)

    @property
    def summary(self) -> str:
        """
        Kısa özet: `## Özet` bölümü varsa onu, yoksa ilk paragrafı döndürür.
        """
        section = extract_section(self.body, "Özet") or extract_section(self.body, "Ozet")
        if section:
            return section[:500]
        plain = strip_markdown(self.body)
        if not plain:
            return ""
        first_paragraph = plain.split("\n\n")[0]
        return first_paragraph[:500]

    @property
    def excerpt(self) -> str:
        """Tek satırlık kısa tanıtım (log ve listeler için)."""
        text = self.summary.replace("\n", " ").strip()
        return text[:120] + ("…" if len(text) > 120 else "")

    @property
    def has_todos(self) -> bool:
        """Gövdede TODO/belirsiz işareti var mı?"""
        return bool(_TODO_RE.search(self.body))

    @property
    def is_shallow(self) -> bool:
        """İçerik çok mu sığ (kelime sayısı düşük)?"""
        return self.word_count < 40

    @property
    def is_low_confidence(self) -> bool:
        """Güven eşiğin altında mı?"""
        return self.confidence < 0.4

    @property
    def filename(self) -> str:
        """Kasa içindeki dosya adı."""
        if self.path:
            return Path(self.path).name
        return f"{self.id}.md"

    @property
    def title_slug(self) -> str:
        """Başlıktan türetilmiş slug."""
        return slugify(self.title)

    @property
    def wikilinks(self) -> List[Tuple[str, str]]:
        """Gövdedeki wiki bağlantıları."""
        return extract_wikilinks(self.body)

    @property
    def headings(self) -> List[Dict[str, Any]]:
        """Gövde başlıkları."""
        return extract_headings(self.body)

    # ─── Mutasyon ──────────────────────────────────────────────────────

    def touch(self) -> None:
        """Güncelleme zamanını şimdiye çeker ve sürümü artırır."""
        self.updated = now_iso()
        self.version = max(1, int(self.version or 1)) + 1

    def set_links(self, link_ids: Sequence[str]) -> None:
        """Bağlantıları tekilleştirerek ayarlar (kendine bağlantıyı atar)."""
        seen: set = set()
        cleaned: List[str] = []
        for link in link_ids:
            if not link or link == self.id or link in seen:
                continue
            seen.add(link)
            cleaned.append(link)
        self.links = cleaned

    def merge_tags(self, extra_tags: Iterable[str]) -> None:
        """Yeni etiketleri normalize ederek ekler."""
        self.tags = normalize_tags(list(self.tags) + list(extra_tags or []))

    def clamp_scores(self) -> None:
        """Güven ve önem değerlerini 0..1 aralığına sıkıştırır."""
        self.confidence = float(min(1.0, max(0.0, self.confidence)))
        self.importance = float(min(1.0, max(0.0, self.importance)))

    def decay_confidence(self, factor: float = 0.9) -> float:
        """
        Güveni kademeli düşürür (çelişki veya eskime durumunda).

        Args:
            factor: Çarpım katsayısı (0..1)

        Returns:
            Yeni güven değeri
        """
        self.confidence = float(min(1.0, max(0.0, self.confidence * factor)))
        self.touch()
        return self.confidence

    def reinforce(self, amount: float = 0.05) -> float:
        """
        Güveni pekiştirir (doğrulama veya tekrar kullanım sonrası).

        Args:
            amount: Eklenecek miktar (0..1)

        Returns:
            Yeni güven değeri
        """
        self.confidence = float(min(1.0, max(0.0, self.confidence + amount)))
        return self.confidence

    # ─── Serileştirme ──────────────────────────────────────────────────

    def to_frontmatter(self) -> Dict[str, Any]:
        """
        Frontmatter sözlüğünü üretir.

        Obsidian uyumluluğu için alan sırası korunur; `extra` alanları
        en sona eklenir.
        """
        data: Dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "category": self.category,
            "tags": list(self.tags),
            "confidence": round(float(self.confidence), 3),
            "importance": round(float(self.importance), 3),
            "created": self.created,
            "updated": self.updated,
            "version": int(self.version),
        }
        if self.sources:
            data["sources"] = list(self.sources)
        if self.links:
            data["links"] = list(self.links)
        if self.embedding_model:
            data["embedding_model"] = self.embedding_model

        # Bilinmeyen alanları koru (çekirdek alanlarla çakışmayanlar)
        for key, value in (self.extra or {}).items():
            if key not in data:
                data[key] = value
        return data

    def to_markdown(self, include_frontmatter: bool = True) -> str:
        """
        Notu kasa dosyası metnine dönüştürür.

        Args:
            include_frontmatter: Frontmatter eklensin mi?

        Returns:
            UTF-8 markdown metni (sonunda tek yeni satır)
        """
        parts: List[str] = []
        if include_frontmatter:
            front = dump_frontmatter(self.to_frontmatter())
            if front:
                parts.append(f"{FRONTMATTER_DELIMITER}\n{front}\n{FRONTMATTER_DELIMITER}")
        body = (self.body or "").rstrip()
        if body:
            parts.append(body)
        return "\n\n".join(parts) + "\n"

    @classmethod
    def from_markdown(
        cls,
        text: str,
        path: Optional[str] = None,
        default_id: Optional[str] = None,
    ) -> "Note":
        """
        Markdown metninden not oluşturur.

        Frontmatter eksik/bozuksa makul varsayılanlar üretilir ve not
        **yine de** okunur (kasa asla okunamaz hale gelmez).

        Args:
            text: Dosya içeriği
            path: Notun kasa içindeki yolu
            default_id: Frontmatter'da id yoksa kullanılacak kimlik

        Returns:
            Oluşturulan Note
        """
        data, body, _ = split_frontmatter(text)

        # Başlık: frontmatter > ilk H1 > dosya adı
        title = str(data.get("title") or "").strip()
        if not title:
            h1 = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
            if h1:
                title = h1.group(1).strip()
        if not title and path:
            title = Path(path).stem

        note_id = str(data.get("id") or "").strip() or (default_id or new_note_id())

        known_keys = {
            "id", "title", "kind", "category", "tags", "confidence", "importance",
            "created", "updated", "version", "sources", "links", "embedding_model",
            "backlinks",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        note = cls(
            id=note_id,
            title=title or note_id,
            body=body.strip(),
            kind=str(data.get("kind") or "concept").strip() or "concept",
            category=str(data.get("category") or "learned").strip() or "learned",
            tags=normalize_tags(data.get("tags") or []),
            confidence=_safe_float(data.get("confidence"), 0.5),
            importance=_safe_float(data.get("importance"), 0.5),
            created=str(data.get("created") or now_iso()),
            updated=str(data.get("updated") or data.get("created") or now_iso()),
            version=int(_safe_float(data.get("version"), 1)),
            sources=_as_str_list(data.get("sources")),
            links=_as_str_list(data.get("links")),
            embedding_model=str(data.get("embedding_model") or ""),
            path=path or "",
            extra=extra,
        )
        note.clamp_scores()

        # Gövdedeki satır içi etiketleri birleştir
        inline = extract_inline_tags(body)
        if inline:
            note.merge_tags(inline)
        return note

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Note":
        """Sözlükten not oluşturur (katmanlar arası taşıma biçimi)."""
        if not isinstance(data, dict):
            raise TypeError("Note.from_dict sözlük bekler")
        known_keys = {
            "id", "title", "body", "kind", "category", "tags", "confidence",
            "importance", "created", "updated", "version", "sources", "links",
            "backlinks", "embedding_model", "embedding", "path",
        }
        extra = dict(data.get("extra") or {})
        extra.update({k: v for k, v in data.items() if k not in known_keys})
        note = cls(
            id=str(data.get("id") or new_note_id()),
            title=str(data.get("title") or ""),
            body=str(data.get("body") or ""),
            kind=str(data.get("kind") or "concept"),
            category=str(data.get("category") or "learned"),
            tags=normalize_tags(data.get("tags") or []),
            confidence=_safe_float(data.get("confidence"), 0.5),
            importance=_safe_float(data.get("importance"), 0.5),
            created=str(data.get("created") or now_iso()),
            updated=str(data.get("updated") or now_iso()),
            version=int(_safe_float(data.get("version"), 1)),
            sources=_as_str_list(data.get("sources")),
            links=_as_str_list(data.get("links")),
            backlinks=_as_str_list(data.get("backlinks")),
            embedding_model=str(data.get("embedding_model") or ""),
            embedding=list(data["embedding"]) if isinstance(data.get("embedding"), (list, tuple)) else None,
            path=str(data.get("path") or ""),
            extra=extra,
        )
        note.clamp_scores()
        return note

    def to_dict(self, include_embedding: bool = False, include_body: bool = True) -> Dict[str, Any]:
        """
        Notu JSON-serileştirilebilir sözlüğe çevirir.

        Args:
            include_embedding: Gömme vektörü dahil edilsin mi? (büyük veri)
            include_body: Gövde dahil edilsin mi?
        """
        result: Dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "category": self.category,
            "tags": list(self.tags),
            "confidence": round(float(self.confidence), 4),
            "importance": round(float(self.importance), 4),
            "created": self.created,
            "updated": self.updated,
            "version": int(self.version),
            "sources": list(self.sources),
            "links": list(self.links),
            "backlinks": list(self.backlinks),
            "embedding_model": self.embedding_model,
            "path": self.path,
            "word_count": self.word_count,
            "excerpt": self.excerpt,
            "summary": self.summary,
            "has_todos": self.has_todos,
        }
        if include_body:
            result["body"] = self.body
        if include_embedding and self.embedding is not None:
            result["embedding"] = list(self.embedding)
        if self.extra:
            result["extra"] = dict(self.extra)
        return result

    # ─── Karşılaştırma ─────────────────────────────────────────────────

    def identity_hash(self) -> str:
        """Notun kimlik parmak izi (yol + başlık)."""
        return content_hash(self.id, self.title)

    def content_fingerprint(self) -> str:
        """İçerik parmak izi — yalnızca başlık ve gövdeye bağlı."""
        return content_hash(normalize_title_for_compare(self.title), strip_markdown(self.body))

    def is_same_content(self, other: "Note", threshold: float = 0.999) -> bool:
        """İki not aynı içerikte mi?"""
        if self.content_fingerprint() == other.content_fingerprint():
            return True
        return title_similarity(self.title, other.title) >= threshold and \
            jaccard(set(self.body.split()), set(other.body.split())) >= threshold

    def __repr__(self) -> str:  # pragma: no cover - hata ayıklama
        return (
            f"<Note {self.id} '{self.title[:40]}' kind={self.kind} "
            f"v{self.version} conf={self.confidence:.2f} words={self.word_count}>"
        )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Note):
            return self.id == other.id
        return NotImplemented


# ─── Karşılaştırma Yardımcıları ─────────────────────────────────────────────

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Güvenli float dönüşümü."""
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str_list(value: Any) -> List[str]:
    """Değeri string listesine çevirir (virgüllü metin, liste veya tekil)."""
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        if "," in value:
            return [part.strip() for part in value.split(",") if part.strip()]
        return [value.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def normalize_title_for_compare(title: str) -> str:
    """Başlığı karşılaştırma için sadeleştirir: küçük harf, alfanümerik, katlanmış."""
    folded = fold_turkish(title or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def jaccard(a: set, b: set) -> float:
    """Jaccard benzerliği (0..1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def title_similarity(a: str, b: str) -> float:
    """
    İki başlık arasındaki benzerlik (0..1).

    Kelime Jaccard'ı ile karakter trigram Dice katsayısının ortalaması;
    kısa başlıklarda daha kararlı çalışır.
    """
    na, nb = normalize_title_for_compare(a), normalize_title_for_compare(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    token_score = jaccard(set(na.split()), set(nb.split()))

    def trigrams(text: str) -> set:
        padded = f"  {text} "
        return {padded[i:i + 3] for i in range(len(padded) - 2)}

    ta, tb = trigrams(na), trigrams(nb)
    dice = (2 * len(ta & tb) / (len(ta) + len(tb))) if (ta or tb) else 0.0

    return round(0.45 * token_score + 0.55 * dice, 4)


def body_similarity(a: str, b: str, max_tokens: int = 4000) -> float:
    """
    İki gövde arasındaki kelime kümesi benzerliği (0..1).

    Çok büyük gövdelerde ilk `max_tokens` kelime kullanılır (performans).
    """
    ta = set(strip_markdown(a).lower().split()[:max_tokens])
    tb = set(strip_markdown(b).lower().split()[:max_tokens])
    return jaccard(ta, tb)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Kosinüs benzerliği (-1..1). Farklı boyutlarda kısa olan kadar karşılaştırır.
    """
    if a is None or b is None:
        return 0.0
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(length):
        va = float(a[i])
        vb = float(b[i])
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ─── Not Fabrikası ──────────────────────────────────────────────────────────

def make_note(
    title: str,
    body: str = "",
    *,
    kind: str = "concept",
    category: str = "learned",
    tags: Optional[Iterable[str]] = None,
    confidence: float = 0.5,
    importance: float = 0.5,
    sources: Optional[Iterable[str]] = None,
    links: Optional[Iterable[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    note_id: Optional[str] = None,
) -> Note:
    """
    Doğru varsayılanlarla hızlı not üretimi.

    Gövde verilmezse başlıktan iskelet bir gövde oluşturulur; böylece
    Obsidian'da açıldığında boş görünmez.
    """
    if not title or not str(title).strip():
        raise ValueError("Not başlığı zorunludur")
    clean_title = str(title).strip()
    clean_body = (body or "").strip()
    if not clean_body:
        clean_body = (
            f"## Özet\n{clean_title} hakkında henüz derlenmiş bilgi yok.\n\n"
            "## Detay\n_(boş)_\n"
        )
    note = Note(
        id=note_id or new_note_id(),
        title=clean_title,
        body=clean_body,
        kind=kind,
        category=category,
        tags=normalize_tags(tags or []),
        confidence=float(confidence),
        importance=float(importance),
        sources=_as_str_list(list(sources or [])),
        links=_as_str_list(list(links or [])),
        extra=dict(extra or {}),
    )
    note.set_links(note.links)
    note.clamp_scores()
    return note


def merge_notes(primary: Note, secondary: Note, new_title: Optional[str] = None) -> Note:
    """
    İki notu tek notta birleştirir (birincil not korunur, kimliği değişmez).

    * Gövde: birincil `## Özet` + ikincil gövde `## Birleştirilen: <başlık>` başlığı altında.
    * Etiketler, kaynaklar ve bağlantılar birleştirilir.
    * Güven: ağırlıklı ortalama (kelime sayısına göre).
    * Çelişki bilgisi `extra["merged_from"]` altında saklanır.
    """
    total_words = max(1, primary.word_count + secondary.word_count)
    blended = (
        primary.confidence * primary.word_count
        + secondary.confidence * secondary.word_count
    ) / total_words

    merged_body = (
        f"{primary.body.rstrip()}\n\n"
        f"## Birleştirilen: {secondary.title}\n"
        f"_{secondary.id} · güven {secondary.confidence:.2f} · v{secondary.version}_\n\n"
        f"{secondary.body.strip()}\n"
    )

    primary.title = (new_title or primary.title).strip()
    primary.body = merged_body
    primary.merge_tags(secondary.tags)
    primary.confidence = float(min(1.0, max(0.0, blended)))
    primary.importance = float(max(primary.importance, secondary.importance))
    primary.sources = list(dict.fromkeys(primary.sources + secondary.sources))
    primary.set_links(primary.links + secondary.links + [secondary.id])

    merged_from = list(primary.extra.get("merged_from") or [])
    merged_from.append({
        "id": secondary.id,
        "title": secondary.title,
        "version": secondary.version,
        "confidence": secondary.confidence,
        "merged_at": now_iso(),
    })
    primary.extra["merged_from"] = merged_from
    primary.touch()
    return primary


def note_diff(old: Note, new: Note) -> Dict[str, Any]:
    """
    İki not sürümü arasındaki farkı çıkarır.

    Returns:
        {"changed": [...], "old": {...}, "new": {...}, "body_delta_words": int}
    """
    fields = ["title", "kind", "category", "tags", "confidence", "importance",
              "sources", "links", "body"]
    changed: List[str] = []
    old_values: Dict[str, Any] = {}
    new_values: Dict[str, Any] = {}

    for name in fields:
        old_value = getattr(old, name, None)
        new_value = getattr(new, name, None)
        if isinstance(old_value, list):
            old_value = list(old_value)
        if isinstance(new_value, list):
            new_value = list(new_value)
        if old_value != new_value:
            changed.append(name)
            old_values[name] = old_value if name != "body" else (old.body or "")[:400]
            new_values[name] = new_value if name != "body" else (new.body or "")[:400]

    return {
        "changed": changed,
        "old": old_values,
        "new": new_values,
        "body_delta_words": new.word_count - old.word_count,
    }


__all__ = [
    "Note",
    "NOTE_KINDS",
    "NOTE_CATEGORIES",
    "make_note",
    "merge_notes",
    "note_diff",
    "new_note_id",
    "is_valid_note_id",
    "slugify",
    "fold_turkish",
    "normalize_tag",
    "normalize_tags",
    "now_iso",
    "utc_timestamp",
    "iso_from_timestamp",
    "parse_timestamp",
    "content_hash",
    "split_frontmatter",
    "parse_frontmatter_fallback",
    "dump_frontmatter",
    "extract_wikilinks",
    "extract_markdown_links",
    "extract_inline_tags",
    "extract_headings",
    "extract_section",
    "extract_bullets",
    "strip_markdown",
    "title_similarity",
    "body_similarity",
    "cosine_similarity",
    "jaccard",
    "normalize_title_for_compare",
]
