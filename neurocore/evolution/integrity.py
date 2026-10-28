"""
NeuroCore Bütünlük Bekçisi (L8 · Evolution)
=============================================
Bilgi kasasının **tutarlılığını** denetler ve onarım planları üretir:

    contradiction · duplicate · orphan · broken_link · stale
    · low_confidence · inconsistent_tag · missing_source · corrupt

Yinelenen not tespiti gerçektir: önce başlıklarda token-kümesi Jaccard'ı ile
hızlı eleme, sonra gövde metinlerinin `_fingerprint` (blake2b imzalı shingle,
L2-normalize, 256 boyut) vektörleri arasında kosinüs benzerliği karşılaştırılır.
Torch kullanılmaz.

Ayrıca çelişki raporlarını denetim bulgusuna çevirir, karantina planı ve
**geri alma (rollback) planı** üretir; sağlık puanı ağırlıklı ciddiyet kütlesini
not sayısına göre normalize eder.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("evolution.integrity")


# ─── Sabitler ──────────────────────────────────────────────────────────────

#: Varsayılan yinelenme eşiği (kosinüs)
DUPLICATE_THRESHOLD = 0.82

#: Varsayılan eskime eşiği (gün)
STALE_DAYS = 180.0

#: Parmak izi boyutu
FINGERPRINT_DIM = 256

#: Tür bazlı ciddiyet ağırlıkları (sağlık puanı için)
KIND_SEVERITY_WEIGHT: Dict[str, float] = {
    "corrupt": 1.0,
    "contradiction": 0.9,
    "broken_link": 0.6,
    "duplicate": 0.6,
    "low_confidence": 0.5,
    "inconsistent_tag": 0.3,
    "missing_source": 0.4,
    "orphan": 0.25,
    "stale": 0.35,
}

#: Arşive taşınacak türler
_ARCHIVE_KINDS = {"low_confidence", "stale", "corrupt"}

#: Düzeltme gerektiren türler
_FIX_KINDS = {"broken_link", "inconsistent_tag", "missing_source", "orphan", "duplicate"}

#: İnceleme gerektiren türler
_REVIEW_KINDS = {"contradiction", "corrupt"}

#: Geçerli etiket deseni (küçük harf, rakam, tire)
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*$")

#: Geçerli not kimliği deseni
_ID_RE = re.compile(r"^[A-Za-z0-9_\-:.]{2,80}$")

_WORD_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü_]+")
_WS_RE = re.compile(r"\s+")
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")
_HEADING_STRIP_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)

#: Zaman damgası sınırları
_MIN_VALID_TS = 978307200.0    # 2001-01-01
_MAX_VALID_TS = 4102444800.0   # 2100-01-01

#: Türkçe karakter katlama tablosu
_FOLD_TABLE = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c",
    "â": "a", "î": "i", "û": "u",
})


# ─── Metin ve Vektör Yardımcıları ──────────────────────────────────────────

def _norm(text: Any) -> str:
    """Küçük harfli, tek boşluklu metin."""
    return _WS_RE.sub(" ", str(text or "").lower()).strip()


def _fold(text: Any) -> str:
    """Türkçe karakterleri ASCII karşılığına katlar."""
    return unicodedata.normalize("NFKC", str(text or "")).lower().translate(_FOLD_TABLE)


def _words(text: Any) -> List[str]:
    """Kelime listesi."""
    return _WORD_RE.findall(str(text or ""))


def _slugify_tag(tag: Any) -> str:
    """Etiketi kanonik biçime çevirir: küçük harf, boşluk→tire, Türkçe katlanmış."""
    text = _fold(str(tag or "")).strip()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"[^a-z0-9\-]", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def _clean_body(note: Any) -> str:
    """Not gövdesini markdown başlıklarından arındırır."""
    if not isinstance(note, dict):
        return ""
    return _HEADING_STRIP_RE.sub("", str(note.get("body") or "")).strip()


def _note_id(note: Any, index: int = 0) -> str:
    """Not kimliği (yoksa sentetik)."""
    if isinstance(note, dict) and note.get("id"):
        return str(note["id"])
    return f"note_{index}"


def _tag_list(note: Any) -> List[str]:
    """Notun etiket listesi (ham biçim korunur)."""
    if not isinstance(note, dict):
        return []
    tags = note.get("tags") or []
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return [str(t) for t in tags if str(t).strip()]


def _confidence(note: Any) -> Optional[float]:
    """Notun güven değeri (okunamıyorsa None)."""
    if not isinstance(note, dict) or note.get("confidence") is None:
        return None
    try:
        return float(note.get("confidence"))
    except (TypeError, ValueError):
        return None


def _timestamp(note: Any, keys: Sequence[str] = ("updated", "updated_at", "modified", "created")) -> Optional[float]:
    """Notun zaman damgası (bulunamazsa None)."""
    if not isinstance(note, dict):
        return None
    for key in keys:
        value = note.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _title_terms(note: Any) -> Set[str]:
    """Başlık token kümesi (hızlı eleme için)."""
    if not isinstance(note, dict):
        return set()
    return {w.lower() for w in _words(note.get("title") or "") if len(w) >= 2}


def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    """Jaccard benzerliği."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Değeri [low, high] aralığına sıkıştır."""
    return max(low, min(high, float(value)))


def _pairs(n: int) -> Iterable[Tuple[int, int]]:
    """`(i, j)` indeks çiftlerini üretir (i < j)."""
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


# ─── Parmak İzi ────────────────────────────────────────────────────────────

def _fingerprint(text: str, dim: int = FINGERPRINT_DIM) -> "np.ndarray":
    """
    Metnin deterministik parmak izi vektörü (L2-normalize, float32).

    Yöntem: unigram + bigram + trigram shingle'ları `blake2b` ile indeks ve
    işarete çevrilir (imzalı rastgele projeksiyon). Aynı metin her zaman aynı
    vektörü üretir; kosinüs benzerliği nokta çarpımına indirgenir.

    Args:
        text: Metin
        dim: Vektör boyutu

    Returns:
        `(dim,)` float32 vektör (boş metinde sıfır vektör)
    """
    size = max(16, int(dim))
    vector = np.zeros(size, dtype=np.float32)

    tokens = [w.lower() for w in _words(text)]
    if not tokens:
        return vector

    shingles: List[str] = list(tokens)
    for ngram in (2, 3):
        for i in range(len(tokens) - ngram + 1):
            shingles.append(" ".join(tokens[i:i + ngram]))

    for shingle in shingles:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % size
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign

    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


# ─── Bütünlük Bulgusu ──────────────────────────────────────────────────────

@dataclass
class IntegrityIssue:
    """
    Tek bir bütünlük sorunu.

    Attributes:
        kind: `contradiction | duplicate | orphan | broken_link | stale |
            low_confidence | inconsistent_tag | missing_source | corrupt`
        severity: 0..1 ciddiyet
        subject: Not kimliği ya da yolu
        detail: İnsan okunur açıklama
        suggestion: Önerilen düzeltme
        evidence: Kanıt referansları
    """

    kind: str
    severity: float
    subject: str
    detail: str
    suggestion: str = ""
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir dict."""
        return {
            "kind": self.kind,
            "severity": round(float(self.severity), 4),
            "subject": self.subject,
            "detail": self.detail,
            "suggestion": self.suggestion,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IntegrityIssue":
        """Dict'ten bulgu kurar."""
        data = d if isinstance(d, dict) else {}
        return cls(
            kind=str(data.get("kind") or "corrupt"),
            severity=_clamp(data.get("severity", 0.5) or 0.0),
            subject=str(data.get("subject") or ""),
            detail=str(data.get("detail") or ""),
            suggestion=str(data.get("suggestion") or ""),
            evidence=[str(e) for e in (data.get("evidence") or [])],
        )


# ─── Bütünlük Bekçisi ──────────────────────────────────────────────────────

class IntegrityGuard:
    """
    Bilgi kasası bütünlük denetleyicisi.

    Kullanım:
        guard = IntegrityGuard()
        guard.initialize(config, event_bus)
        issues = guard.audit(notes, graph, contradictions)
        plan = guard.quarantine_plan(issues)
        score = guard.health_score(issues, len(notes))
    """

    name = "evolution.integrity"

    def __init__(self) -> None:
        self._config: Any = None
        self._event_bus: Any = None
        self._counters: Dict[str, float] = {
            "audits": 0.0,
            "issues": 0.0,
            "audited_notes": 0.0,
        }
        self._last_kinds: Dict[str, int] = {}
        self._last_health: float = 1.0
        self._initialized: bool = False

    # ─── Yaşam Döngüsü ──────────────────────────────────────────────────

    def initialize(self, config: Any = None, event_bus: Any = None) -> None:
        """Bekçiyi bağla. İdempotenttir."""
        if self._initialized:
            return
        self._config = config
        self._event_bus = event_bus
        self._initialized = True
        logger.info("Bütünlük bekçisi hazır")

    def shutdown(self) -> None:
        """Kaynakları serbest bırak. İdempotenttir."""
        if not self._initialized:
            return
        self._initialized = False
        logger.info("Bütünlük bekçisi kapatıldı")

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Olay yayınla (bus yoksa atlanır)."""
        if self._event_bus:
            try:
                self._event_bus.emit(event_type, data, source=self.name)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Olay yayınlanamadı ({event_type}): {exc}")

    # ─── Ana Denetim ───────────────────────────────────────────────────

    def audit(
        self,
        notes: List[dict],
        graph: Optional[Dict[str, List[str]]] = None,
        contradictions: Optional[List[dict]] = None,
    ) -> List[IntegrityIssue]:
        """
        Tüm bütünlük kontrollerini çalıştırır.

        Args:
            notes: Kasa not yükleri
            graph: `{kaynak_id: [hedef_id, ...]}` bağlantı haritası
            contradictions: Çelişki raporları (`synthesizer.find_contradictions`)

        Returns:
            Ciddiyete göre azalan sırada `IntegrityIssue` listesi
        """
        self._counters["audits"] += 1.0
        usable = [n for n in (notes or []) if isinstance(n, dict)]
        self._counters["audited_notes"] += float(len(usable))

        issues: List[IntegrityIssue] = []

        for producer in (
            lambda: self.check_duplicates(usable),
            lambda: self.check_broken_links(usable, graph),
            lambda: self.check_metadata(usable),
            lambda: self.check_staleness(usable),
            lambda: self._from_contradictions(usable, contradictions or []),
        ):
            try:
                issues.extend(producer())
            except Exception as exc:
                logger.warning(f"Bütünlük denetim adımı başarısız: {exc}")

        # Aynı bulguyu tekrarlamayalım
        unique: List[IntegrityIssue] = []
        seen: Set[Tuple[str, str, str]] = set()
        for issue in issues:
            key = (issue.kind, issue.subject, issue.detail)
            if key in seen:
                continue
            seen.add(key)
            unique.append(issue)

        unique.sort(key=lambda item: (-float(item.severity), item.kind, item.subject))

        kinds: Dict[str, int] = {}
        for issue in unique:
            kinds[issue.kind] = kinds.get(issue.kind, 0) + 1
        self._last_kinds = kinds
        self._counters["issues"] += float(len(unique))

        health = self.health_score(unique, len(usable))
        self._last_health = health

        self._emit("evolution.integrity_audited", {
            "issues": len(unique),
            "kinds": kinds,
            "note_count": len(usable),
            "health": health,
        })
        logger.info(
            f"Bütünlük denetimi: {len(unique)} bulgu, sağlık={health:.2f} "
            f"({', '.join(f'{k}={v}' for k, v in sorted(kinds.items())) or 'temiz'})"
        )
        return unique

    # ─── 1) Yinelenenler ───────────────────────────────────────────────

    def check_duplicates(self, notes: List[dict], threshold: float = DUPLICATE_THRESHOLD) -> List[IntegrityIssue]:
        """
        Yinelenen notları bulur.

        İki aşamalı: başlıklarda token-kümesi Jaccard'ı (hızlı eleme) → gövde
        parmak izlerinde kosinüs benzerliği. Eşiğin üzerindeki çiftler
        "birleştir" önerisiyle raporlanır.
        """
        issues: List[IntegrityIssue] = []
        usable = [n for n in (notes or []) if isinstance(n, dict)]
        if len(usable) < 2:
            return issues

        threshold_value = float(threshold or DUPLICATE_THRESHOLD)
        title_sets = [_title_terms(note) for note in usable]
        bodies = [_clean_body(note) for note in usable]

        fingerprints = np.zeros((len(usable), FINGERPRINT_DIM), dtype=np.float32)
        for index, body in enumerate(bodies):
            fingerprints[index] = _fingerprint(body)

        for i, j in _pairs(len(usable)):
            body_i, body_j = bodies[i].strip(), bodies[j].strip()
            if not body_i or not body_j:
                continue

            # Hızlı eleme: başlıklar tamamen farklıysa gövdeye bakma
            title_similarity = _jaccard(title_sets[i], title_sets[j])
            if title_similarity < 0.34 and title_similarity > 0.0:
                continue

            cosine = float(np.dot(fingerprints[i], fingerprints[j]))
            if cosine < threshold_value:
                continue

            severity = _clamp(0.45 + (cosine - threshold_value) * 1.5, 0.45, 0.95)
            id_i, id_j = _note_id(usable[i], i), _note_id(usable[j], j)
            issues.append(IntegrityIssue(
                kind="duplicate",
                severity=round(severity, 4),
                subject=id_i,
                detail=(
                    f"'{usable[i].get('title') or id_i}' ile '{usable[j].get('title') or id_j}' "
                    f"neredeyse aynı (kosinüs={cosine:.3f}, başlık Jaccard={title_similarity:.2f})"
                ),
                suggestion="İki notu birleştir; ikincisini arşive taşı ve bağlantıları güncelle",
                evidence=[id_i, id_j, f"cosine={cosine:.3f}"],
            ))
        return issues

    # ─── 2) Kırık Bağlantılar ──────────────────────────────────────────

    def check_broken_links(
        self,
        notes: List[dict],
        graph: Optional[Dict[str, List[str]]] = None,
    ) -> List[IntegrityIssue]:
        """
        Hedefi var olmayan bağlantıları bulur.

        Hem verilen graf kenarları hem de gövdedeki `[[...]]` bağlantıları
        denetlenir.
        """
        issues: List[IntegrityIssue] = []
        usable = [n for n in (notes or []) if isinstance(n, dict)]
        if not usable:
            return issues

        endpoints = self._valid_endpoints(usable)

        reported: Set[Tuple[str, str]] = set()
        edges = graph if isinstance(graph, dict) else {}

        for source, targets in edges.items():
            source_key = _norm(source)
            for target in (targets or []):
                target_key = _norm(target)
                if not target_key or target_key in endpoints:
                    continue
                key = (source_key, target_key)
                if key in reported:
                    continue
                reported.add(key)
                issues.append(IntegrityIssue(
                    kind="broken_link",
                    severity=0.6,
                    subject=str(source),
                    detail=f"'{source}' notu var olmayan '{target}' notuna bağlanıyor",
                    suggestion="Bağlantıyı kaldır ya da hedef notu oluştur",
                    evidence=[str(source), str(target)],
                ))

        for index, note in enumerate(usable):
            note_id = _note_id(note, index)
            for match in _WIKILINK_RE.findall(_clean_body(note)):
                target_key = _norm(match)
                if not target_key or target_key in endpoints:
                    continue
                key = (_norm(note_id), target_key)
                if key in reported:
                    continue
                reported.add(key)
                issues.append(IntegrityIssue(
                    kind="broken_link",
                    severity=0.5,
                    subject=note_id,
                    detail=f"Gövdedeki [[{match}]] bağlantısının hedefi bulunamadı",
                    suggestion="Bağlantıyı düzelt ya da eksik notu oluştur",
                    evidence=[note_id, str(match)],
                ))
        return issues

    def _valid_endpoints(self, notes: Sequence[dict]) -> Set[str]:
        """Geçerli bağlantı hedefleri (id, katlanmış id, başlık, katlanmış başlık)."""
        endpoints: Set[str] = set()
        for index, note in enumerate(notes):
            note_id = _note_id(note, index)
            title = str(note.get("title") or "").strip()
            for value in (note_id, title, note.get("path")):
                if not value:
                    continue
                text = str(value)
                endpoints.add(_norm(text))
                endpoints.add(_norm(_fold(text)))
                endpoints.add(_norm(text.replace(" ", "-")))
        return endpoints

    # ─── 3) Üst Veri (Metadata) ────────────────────────────────────────

    def check_metadata(self, notes: List[dict]) -> List[IntegrityIssue]:
        """
        Frontmatter/üst veri sorunlarını bulur:

        eksik/geçersiz alanlar, 0..1 dışı güven, gelecekte/geçersiz zaman
        damgası, etiket normalizasyonu (küçük harf, boşluksuz, Türkçe katlanmış),
        yinelenen etiketler, `title == id`, boş gövde.
        """
        issues: List[IntegrityIssue] = []
        now = time.time()

        for index, note in enumerate(notes or []):
            if not isinstance(note, dict):
                continue
            note_id = _note_id(note, index)

            # (a) Boş gövde
            body = _clean_body(note)
            if not body:
                issues.append(IntegrityIssue(
                    kind="corrupt",
                    severity=0.7,
                    subject=note_id,
                    detail="Not gövdesi boş",
                    suggestion="Notu sil, arşive taşı ya da içerik ekle",
                    evidence=[note_id],
                ))

            # (b) Eksik kimlik / başlık
            if not note.get("id"):
                issues.append(IntegrityIssue(
                    kind="corrupt",
                    severity=0.5,
                    subject=note_id,
                    detail="Not kimliği (id) eksik",
                    suggestion="Frontmatter'a benzersiz bir id ekle",
                    evidence=[note_id],
                ))
            elif not _ID_RE.match(str(note.get("id"))):
                issues.append(IntegrityIssue(
                    kind="corrupt",
                    severity=0.35,
                    subject=note_id,
                    detail=f"Not kimliği biçimi geçersiz: {note.get('id')!r}",
                    suggestion="Kimliği harf/rakam/tire kullanarak yeniden üret",
                    evidence=[note_id],
                ))

            title = str(note.get("title") or "").strip()
            if not title:
                issues.append(IntegrityIssue(
                    kind="corrupt",
                    severity=0.4,
                    subject=note_id,
                    detail="Not başlığı eksik",
                    suggestion="Anlamlı bir başlık ekle",
                    evidence=[note_id],
                ))
            elif note.get("id") and _norm(title) == _norm(note.get("id")):
                issues.append(IntegrityIssue(
                    kind="inconsistent_tag",
                    severity=0.2,
                    subject=note_id,
                    detail="Başlık, not kimliğiyle aynı",
                    suggestion="İnsan okunur bir başlık yaz",
                    evidence=[note_id],
                ))

            # (c) Güven aralığı
            confidence = _confidence(note)
            if note.get("confidence") is None:
                issues.append(IntegrityIssue(
                    kind="low_confidence",
                    severity=0.3,
                    subject=note_id,
                    detail="Güven (confidence) alanı yok",
                    suggestion="Varsayılan güven değeri ata (ör. 0.5)",
                    evidence=[note_id],
                ))
            elif confidence is not None and not (0.0 <= confidence <= 1.0):
                issues.append(IntegrityIssue(
                    kind="low_confidence",
                    severity=0.5,
                    subject=note_id,
                    detail=f"Güven değeri 0..1 aralığı dışında: {confidence}",
                    suggestion="Güven değerini 0..1 aralığına sıkıştır",
                    evidence=[note_id, f"confidence={confidence}"],
                ))
            elif confidence is not None and confidence < 0.4:
                issues.append(IntegrityIssue(
                    kind="low_confidence",
                    severity=round(_clamp(0.55 - confidence, 0.15, 0.55), 4),
                    subject=note_id,
                    detail=f"Güven değeri düşük: {confidence:.2f}",
                    suggestion="Notu doğrula, kaynak ekle ya da güveni düşürüp işaretle",
                    evidence=[note_id, f"confidence={confidence:.2f}"],
                ))

            # (d) Zaman damgaları
            for key in ("created", "updated"):
                value = note.get(key)
                if value is None:
                    continue
                try:
                    stamp = float(value)
                except (TypeError, ValueError):
                    issues.append(IntegrityIssue(
                        kind="corrupt",
                        severity=0.4,
                        subject=note_id,
                        detail=f"'{key}' zaman damgası sayı değil: {value!r}",
                        suggestion="ISO-8601 ya da Unix zaman damgası kullan",
                        evidence=[note_id, key],
                    ))
                    continue
                if stamp <= 0:
                    continue
                if stamp > now + 86400.0:
                    issues.append(IntegrityIssue(
                        kind="corrupt",
                        severity=0.45,
                        subject=note_id,
                        detail=f"'{key}' zaman damgası gelecekte: {stamp:.0f}",
                        suggestion="Saat dilimi/dönüşüm hatasını düzelt",
                        evidence=[note_id, key, f"ts={stamp:.0f}"],
                    ))
                elif stamp < _MIN_VALID_TS or stamp > _MAX_VALID_TS:
                    issues.append(IntegrityIssue(
                        kind="corrupt",
                        severity=0.4,
                        subject=note_id,
                        detail=f"'{key}' zaman damgası geçersiz aralıkta: {stamp:.0f}",
                        suggestion="Zaman damgasını gerçek bir tarihle değiştir",
                        evidence=[note_id, key, f"ts={stamp:.0f}"],
                    ))

            # (e) Etiketler
            raw_tags = _tag_list(note)
            normalized: List[str] = []
            malformed: List[str] = []
            for tag in raw_tags:
                canonical = _slugify_tag(tag)
                if not canonical:
                    malformed.append(str(tag))
                    continue
                if not _TAG_RE.match(canonical) or canonical != str(tag):
                    malformed.append(str(tag))
                if canonical not in normalized:
                    normalized.append(canonical)

            if malformed:
                issues.append(IntegrityIssue(
                    kind="inconsistent_tag",
                    severity=0.25,
                    subject=note_id,
                    detail=f"Normalize edilmemiş etiketler: {', '.join(malformed[:5])}",
                    suggestion="Etiketleri küçük harf, tiresi ve Türkçe karakterlerden arındırılmış yaz",
                    evidence=[note_id] + malformed[:5],
                ))
            if len(normalized) != len(raw_tags) and not malformed:
                issues.append(IntegrityIssue(
                    kind="inconsistent_tag",
                    severity=0.2,
                    subject=note_id,
                    detail="Yinelenen etiketler var",
                    suggestion="Etiket listesini tekilleştir",
                    evidence=[note_id] + normalized[:5],
                ))
            if not raw_tags:
                issues.append(IntegrityIssue(
                    kind="inconsistent_tag",
                    severity=0.15,
                    subject=note_id,
                    detail="Not etiketsiz; arama ve kümeleme zayıflar",
                    suggestion="En az bir konu etiketi ekle",
                    evidence=[note_id],
                ))

            # (f) Kaynak
            sources = note.get("sources") or []
            if isinstance(sources, str):
                sources = [sources] if sources.strip() else []
            if not sources:
                issues.append(IntegrityIssue(
                    kind="missing_source",
                    severity=0.3,
                    subject=note_id,
                    detail="Notun kaynağı yok",
                    suggestion="Kaynak referansı ekle (dosya, komut çıktısı, URL)",
                    evidence=[note_id],
                ))
        return issues

    # ─── 4) Eskime ─────────────────────────────────────────────────────

    def check_staleness(self, notes: List[dict], max_age_days: float = STALE_DAYS) -> List[IntegrityIssue]:
        """
        Eski notları işaretler. Ciddiyet yaşla artar, yüksek güvenle azalır.
        """
        issues: List[IntegrityIssue] = []
        now = time.time()
        threshold_days = float(max_age_days or STALE_DAYS)

        for index, note in enumerate(notes or []):
            if not isinstance(note, dict):
                continue
            stamp = _timestamp(note)
            if stamp is None or stamp <= 0:
                continue
            age_days = (now - stamp) / 86400.0
            if age_days <= threshold_days:
                continue
            confidence = _confidence(note)
            confidence = 0.5 if confidence is None else _clamp(confidence)
            severity = _clamp(0.25 + 0.02 * (age_days - threshold_days) / 30.0, 0.25, 0.7)
            severity = _clamp(severity - 0.15 * confidence, 0.15, 0.7)
            note_id = _note_id(note, index)
            issues.append(IntegrityIssue(
                kind="stale",
                severity=round(severity, 4),
                subject=note_id,
                detail=(
                    f"Not {age_days:.0f} gündür güncellenmedi (eşik: {threshold_days:.0f} gün, "
                    f"güven: {confidence:.2f})"
                ),
                suggestion="Bilgiyi doğrula; geçersizse arşive taşı, geçerliyse güncelle",
                evidence=[note_id, f"age_days={age_days:.0f}"],
            ))
        return issues

    # ─── 5) Çelişkiler ─────────────────────────────────────────────────

    def _from_contradictions(self, notes: Sequence[dict], contradictions: List[dict]) -> List[IntegrityIssue]:
        """Çelişki raporlarını bütünlük bulgusuna çevirir."""
        issues: List[IntegrityIssue] = []
        index_of: Dict[str, dict] = {}
        for index, note in enumerate(notes):
            index_of[_note_id(note, index)] = note

        for conflict in (contradictions or []):
            if not isinstance(conflict, dict):
                continue
            a = str(conflict.get("a") or "")
            b = str(conflict.get("b") or "")
            if not a and not b:
                continue
            reason = str(conflict.get("reason") or "gerekçe belirtilmedi")
            subject = a or b
            issues.append(IntegrityIssue(
                kind="contradiction",
                severity=0.9,
                subject=subject,
                detail=f"'{a}' ile '{b}' çelişiyor: {reason}",
                suggestion=(
                    "İki notu `contradicts` kenarıyla bağla, güveni düşür ve "
                    "hangisinin doğru olduğunu araştır"
                ),
                evidence=[a, b] + [str(t) for t in (conflict.get("tokens") or [])][:4],
            ))
        return issues

    # ─── 6) Karantina Planı ────────────────────────────────────────────

    def quarantine_plan(self, issues: List[IntegrityIssue]) -> Dict[str, Any]:
        """
        Bulguları eylem planına çevirir.

        Returns:
            `{"to_archive": [...], "to_merge": [(a, b)], "to_fix": [...],
              "to_review": [...], "summary": {...}}`
        """
        to_archive: List[str] = []
        to_merge: List[Tuple[str, str]] = []
        to_fix: List[str] = []
        to_review: List[str] = []

        for issue in (issues or []):
            if not isinstance(issue, IntegrityIssue):
                continue
            if issue.kind == "duplicate":
                pair = self._duplicate_pair(issue)
                if pair and pair not in to_merge:
                    to_merge.append(pair)
                continue
            if issue.kind in _ARCHIVE_KINDS and float(issue.severity) >= 0.5:
                if issue.subject not in to_archive:
                    to_archive.append(issue.subject)
                continue
            if issue.kind in _FIX_KINDS:
                if issue.subject not in to_fix:
                    to_fix.append(issue.subject)
                continue
            if issue.kind in _REVIEW_KINDS:
                if issue.subject not in to_review:
                    to_review.append(issue.subject)
                continue
            if issue.subject not in to_fix:
                to_fix.append(issue.subject)

        plan = {
            "to_archive": to_archive,
            "to_merge": to_merge,
            "to_fix": to_fix,
            "to_review": to_review,
            "summary": {
                "archive": len(to_archive),
                "merge": len(to_merge),
                "fix": len(to_fix),
                "review": len(to_review),
                "total_issues": len(list(issues or [])),
            },
        }
        self._emit("evolution.quarantine_planned", dict(plan["summary"]))
        return plan

    @staticmethod
    def _duplicate_pair(issue: IntegrityIssue) -> Optional[Tuple[str, str]]:
        """Yinelenme bulgusundan (a, b) çiftini çıkarır."""
        evidence = [str(e) for e in (issue.evidence or [])]
        if len(evidence) >= 2 and evidence[0] and evidence[1]:
            return evidence[0], evidence[1]
        if issue.subject:
            return issue.subject, issue.subject
        return None

    # ─── 7) Geri Alma Planı ────────────────────────────────────────────

    def rollback_plan(self, applied: List[dict], failed_index: int) -> List[dict]:
        """
        Kısmen uygulanmış bir toplu işlem için ters işlemleri üretir.

        `applied[:failed_index]` başarıyla uygulanmış kabul edilir; ters
        işlemler **ters sırada** döndürülür.

        Args:
            applied: Uygulanan işlem kayıtları
            failed_index: Başarısız olan işlemin indeksi

        Returns:
            `[{"op", "target", ...}]` — uygulanacak ters işlemler
        """
        operations = list(applied or [])
        try:
            boundary = int(failed_index)
        except (TypeError, ValueError):
            boundary = len(operations)
        boundary = max(0, min(boundary, len(operations)))

        rollback: List[dict] = []
        for record in reversed(operations[:boundary]):
            if not isinstance(record, dict):
                continue
            op = str(record.get("op") or record.get("action") or "").strip().lower()
            target = str(record.get("target") or record.get("note_id") or record.get("id") or "")

            if op in ("upsert_note", "create_note", "create", "upsert"):
                rollback.append({
                    "op": "archive_note",
                    "target": target,
                    "reason": "geri alma: oluşturulan not kaldırılıyor",
                    "restore": record.get("previous"),
                })
            elif op in ("update_note", "update"):
                rollback.append({
                    "op": "restore_note",
                    "target": target,
                    "body": record.get("previous_body", ""),
                    "confidence": record.get("previous_confidence"),
                    "reason": "geri alma: önceki içerik geri yükleniyor",
                })
            elif op in ("delete_note", "archive_note", "archive"):
                rollback.append({
                    "op": "restore_note",
                    "target": target,
                    "restore": record.get("previous"),
                    "reason": "geri alma: arşivlenen not geri getiriliyor",
                })
            elif op in ("link_notes", "link"):
                rollback.append({
                    "op": "unlink_notes",
                    "target": target,
                    "dst": str(record.get("dst") or record.get("target_id") or ""),
                    "reason": "geri alma: kurulan bağlantı kaldırılıyor",
                })
            elif op in ("unlink_notes", "unlink"):
                rollback.append({
                    "op": "link_notes",
                    "target": target,
                    "dst": str(record.get("dst") or record.get("target_id") or ""),
                    "reason": "geri alma: kaldırılan bağlantı geri kuruluyor",
                })
            elif op in ("merge_notes", "merge"):
                rollback.append({
                    "op": "split_merge",
                    "target": target,
                    "merged_into": str(record.get("merged_into") or ""),
                    "sources": list(record.get("sources") or []),
                    "reason": "geri alma: birleştirme ayrılıyor",
                })
            elif op in ("tag_add", "add_tag"):
                rollback.append({
                    "op": "tag_remove",
                    "target": target,
                    "tag": str(record.get("tag") or ""),
                    "reason": "geri alma: eklenen etiket kaldırılıyor",
                })
            else:
                rollback.append({
                    "op": "review",
                    "target": target,
                    "reason": f"geri alma: bilinmeyen işlem '{op}' elle incelenmeli",
                })

        self._emit("evolution.rollback_planned", {
            "operations": len(rollback),
            "failed_index": boundary,
        })
        return rollback

    # ─── 8) Sağlık Puanı ───────────────────────────────────────────────

    def health_score(self, issues: List[IntegrityIssue], note_count: int) -> float:
        """
        Kasanın sağlık puanı (0..1).

        Ağırlıklı ciddiyet kütlesi not sayısına göre normalize edilir:
        `1.0 - Σ(tür_ağırlığı × ciddiyet) / max(1, not_sayısı)`.
        """
        try:
            count = max(0, int(note_count))
        except (TypeError, ValueError):
            count = 0

        mass = 0.0
        for issue in (issues or []):
            if not isinstance(issue, IntegrityIssue):
                continue
            weight = KIND_SEVERITY_WEIGHT.get(issue.kind, 0.3)
            mass += weight * _clamp(issue.severity)

        if count <= 0:
            return 1.0 if mass <= 0.0 else 0.0
        return round(_clamp(1.0 - mass / count), 4)

    # ─── İstatistikler ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """JSON-serileştirilebilir sağlık/performans metrikleri."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "audits": int(self._counters["audits"]),
            "issues_found": int(self._counters["issues"]),
            "notes_audited": int(self._counters["audited_notes"]),
            "last_kinds": dict(self._last_kinds),
            "last_health": self._last_health,
            "severity_weights": dict(KIND_SEVERITY_WEIGHT),
            "duplicate_threshold": DUPLICATE_THRESHOLD,
            "fingerprint_dim": FINGERPRINT_DIM,
        }
