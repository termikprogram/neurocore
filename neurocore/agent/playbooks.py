"""
Playbook — başarılı iş akışlarını **prosedür** olarak saklama ve geri çağırma
============================================================================
Amaç: ajan aynı tür görevi her seferinde sıfırdan planlamasın.

Nasıl çalışır
-------------
```
başarılı çalışma (ödül ≥ 0.6, en az 2 araç)
   → araç dizisi + hedef → .neurocore/playbooks.jsonl  (eklemeli)
sonraki görev
   → hedef benzerliği (Jaccard ≥ 0.35) → en iyi 1 playbook
   → plan aşamasında **ipucu** olarak mesaja eklenir
```

⚠ Bu bir "ezber" değildir: playbook **yol gösterir**, zorunlu kılmaz.
Model farklı yol seçebilir; playbook yalnız "bu iş daha önce şöyle çözüldü"
bilgisini taşır.

Dosya biçimi (JSONL — kesinti güvenli):

```json
{"id": "pb_ab12cd34", "goal": "kılıç modeli oluştur", "adimlar":
 [{"tool": "asset.uret", "args": {"kategori": "silah"}},
  {"tool": "asset.onizleme"}], "odul": 0.85, "kullanim": 0, "ts": 1793...}
```
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.playbooks")

#: Varsayılan dosya (git dışı)
DOSYA = "playbooks.jsonl"

#: Playbook'a dönüşmesi için gereken en düşük ödül
MIN_ODUL = 0.6

#: En az kaç araç adımı (tek adımlık işler playbook değildir)
MIN_ADIM = 2

#: En fazla kaç adım saklanır (bağlam sınırı)
MAX_ADIM = 8

#: Benzerlik eşiği (Jaccard)
MIN_BENZERLIK = 0.35

#: Saklanacak en fazla playbook (eskiler atılır)
MAX_PLAYBOOK = 200

#: Türkçe/İngilizce işlev sözcükleri — benzerlikte sayılmaz
_STOP = frozenset("""
bir bu şu o ve ile için gibi çok daha en de da ki mi mı mu mü the a an of to
in on for with and or is are be create make add use using please lütfen
""".split())


def _kelimeler(metin: str) -> set:
    """Hedef metnini anlamlı sözcüklere indirger."""
    kucuk = str(metin or "").lower()
    parcalar = re.findall(r"[a-zçğıöşü0-9_.]+", kucuk)
    return {p for p in parcalar if p not in _STOP and len(p) > 1}


def _eslesen(x: set, y: set) -> int:
    """
    Önek eşleşmesiyle kesişim sayısı.

    ⚠ Ölçüldü: düz küme kesişimi Türkçe eklerde çöküyor —
    "ağaç" ≠ "ağaçları" → benzerlik 0.20 (eşik 0.35 altında). Bu yüzden
    kısa sözcük uzun sözcüğün **önekiyse** eşleşmiş sayılır (en az 4 harf).
    """
    sayi = 0
    for a in x:
        for b in y:
            if a == b:
                sayi += 1
                break
            kisa, uzun = (a, b) if len(a) <= len(b) else (b, a)
            if len(kisa) >= 4 and uzun.startswith(kisa):
                sayi += 1
                break
    return sayi


def benzerlik(a: str, b: str) -> float:
    """İki hedef metni arasındaki Jaccard benzerliği (önek duyarlı, 0..1)."""
    x, y = _kelimeler(a), _kelimeler(b)
    if not x or not y:
        return 0.0
    kesisim = _eslesen(x, y)
    birlesim = len(x) + len(y) - kesisim
    return kesisim / birlesim if birlesim else 0.0


@dataclass
class Playbook:
    """Bir başarılı iş akışının özeti."""

    goal: str
    adimlar: List[Dict[str, Any]] = field(default_factory=list)
    odul: float = 0.0
    kullanim: int = 0
    ts: float = field(default_factory=time.time)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            ham = f"{self.goal}|{json.dumps(self.adimlar, sort_keys=True)}"
            self.id = "pb_" + hashlib.sha256(
                ham.encode("utf-8")).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "goal": self.goal, "adimlar": self.adimlar[:MAX_ADIM],
            "odul": round(float(self.odul), 3), "kullanim": int(self.kullanim),
            "ts": float(self.ts),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Playbook":
        return cls(
            goal=str(d.get("goal") or "")[:300],
            adimlar=list(d.get("adimlar") or [])[:MAX_ADIM],
            odul=float(d.get("odul") or 0.0),
            kullanim=int(d.get("kullanim") or 0),
            ts=float(d.get("ts") or 0.0),
            id=str(d.get("id") or ""),
        )

    def ipucu(self) -> str:
        """Plana eklenecek kısa ipucu metni."""
        satirlar = [f"{i + 1}. {a.get('tool')}"
                    + (f" ({a.get('args')})" if a.get("args") else "")
                    for i, a in enumerate(self.adimlar[:MAX_ADIM])]
        return (f"Daha önce benzer bir görev şöyle çözüldü "
                f"(ödül {self.odul:.2f}, {self.kullanim} kez kullanıldı):\n"
                + "\n".join(satirlar)
                + "\nİstersen bu yolu izle; daha iyisini biliyorsan farklı git.")


# ─── Depo ───────────────────────────────────────────────────────────────────

class PlaybookStore:
    """
    Playbook deposu (JSONL, eklemeli).

    ⚠ Neden ayrı dosya? Kasa markdown'dır ve insan içindir; playbook
    makine içindir ve her görevde okunur. Karıştırmak aramayı yavaşlatır.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else Path(".neurocore") / DOSYA
        self._cache: Optional[List[Playbook]] = None

    # ── okuma ──
    def hepsi(self, tazele: bool = False) -> List[Playbook]:
        if self._cache is not None and not tazele:
            return self._cache
        kayitlar: List[Playbook] = []
        if self.path.exists():
            for satir in self.path.read_text(encoding="utf-8",
                                             errors="ignore").splitlines():
                satir = satir.strip()
                if not satir:
                    continue
                try:
                    kayitlar.append(Playbook.from_dict(json.loads(satir)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue          # bozuk satır atlanır (kesinti güvenli)
        self._cache = kayitlar
        return kayitlar

    def bul(self, goal: str, *, esik: float = MIN_BENZERLIK,
            adet: int = 1) -> List[Tuple[Playbook, float]]:
        """Hedefe en benzer playbook'lar (benzerlik, ödül sırasıyla)."""
        adaylar = []
        for pb in self.hepsi():
            skor = benzerlik(goal, pb.goal)
            if skor >= esik:
                adaylar.append((pb, skor))
        adaylar.sort(key=lambda x: (x[1], x[0].odul), reverse=True)
        return adaylar[:max(0, adet)]

    # ── yazma ──
    def kaydet(self, goal: str, adimlar: Sequence[Dict[str, Any]],
               odul: float) -> Optional[Playbook]:
        """
        Başarılı akışı kaydeder (uygunsa).

        Returns:
            Kaydedilen `Playbook` veya `None` (eşik altı / tekrar).
        """
        temiz = [a for a in adimlar if a.get("tool")][:MAX_ADIM]
        if float(odul) < MIN_ODUL or len(temiz) < MIN_ADIM:
            return None
        pb = Playbook(goal=str(goal)[:300], adimlar=temiz,
                      odul=min(1.0, max(0.0, float(odul))))
        mevcut = self.hepsi(tazele=True)
        for eski in mevcut:
            if eski.id == pb.id:
                eski.kullanim += 1        # aynı akış tekrar kullanıldı
                eski.odul = max(eski.odul, pb.odul)
                self._yaz(mevcut)
                return eski
        mevcut.append(pb)
        # ⚠ Depo büyümesin: en eski/düşük ödüllü atılır
        if len(mevcut) > MAX_PLAYBOOK:
            mevcut.sort(key=lambda p: (p.odul, p.ts), reverse=True)
            mevcut = mevcut[:MAX_PLAYBOOK]
        self._yaz(mevcut)
        logger.info(f"Playbook kaydedildi: {pb.goal[:50]} · "
                    f"{len(pb.adimlar)} adım · ödül {pb.odul:.2f}")
        return pb

    def kullanim_artir(self, pb_id: str) -> None:
        """Playbook kullanıldığında sayacı artırır (geri besleme)."""
        mevcut = self.hepsi(tazele=True)
        degisti = False
        for pb in mevcut:
            if pb.id == pb_id:
                pb.kullanim += 1
                degisti = True
                break
        if degisti:
            self._yaz(mevcut)

    def _yaz(self, kayitlar: Iterable[Playbook]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            satirlar = [json.dumps(pb.to_dict(), ensure_ascii=False)
                        for pb in kayitlar]
            # ⚠ Önce geçici dosya, sonra taşı: yarım yazım depoyu bozmasın
            gecici = self.path.with_suffix(".tmp")
            gecici.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
            gecici.replace(self.path)
            self._cache = list(kayitlar)
        except OSError as exc:
            logger.warning(f"Playbook yazılamadı: {exc}")

    def istatistik(self) -> Dict[str, Any]:
        kayitlar = self.hepsi(tazele=True)
        return {
            "playbook": len(kayitlar),
            "ortalama_adim": round(
                sum(len(p.adimlar) for p in kayitlar) / max(1, len(kayitlar)), 2),
            "ortalama_odul": round(
                sum(p.odul for p in kayitlar) / max(1, len(kayitlar)), 3),
            "toplam_kullanim": sum(p.kullanim for p in kayitlar),
            "dosya": str(self.path),
        }
