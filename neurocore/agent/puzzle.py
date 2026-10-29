"""
Puzzle — iç içe fazlar, alt fazlar ve walkthrough'lar
=====================================================
Kullanıcının istediği yapı: bir görev **fazlara** ayrılır, fazlar kendi
içinde alt fazlara bölünebilir, her fazın başında bir **walkthrough**
(durum + netleştirme soruları) vardır ve kullanıcı "şu kısmı değiştir"
diyebilir. Yapı keyfî derinliktedir.

Neden "Puzzle"? Tek parça (piece) hem işi hem kabı temsil eder; her parça
kendi içinde daha küçük parçalara ayrılır ve sonunda tek resim oluşur.

```text
Puzzle (görev)
└─ Faz 1  "Keşif"                     [tamam]
   ├─ Walkthrough (sorular + cevap)
   ├─ Adım 1.1  proje haritası         [tamam]
   └─ Faz 1.2  "Bağımlılıklar"          [tamam]
      └─ Adım 1.2.1
└─ Faz 2  "Uygulama"                  [aktif]
   └─ ...
```

Sınırlar (MUST)
---------------
```
· derinlik ≤ MAX_DERINLIK (4) · toplam parça ≤ MAX_PARCA (64)
· walkthrough soruları ≤ 5 · bir parçanın alt parçası ≤ 7
· LLM çözemezse tek fazlık düz plan üretilir (akış bozulmaz)
· parça kimliği yol tabanlıdır: "p1", "p1.2", "p1.2.3" (revizyonda sabit)
```
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurocore.agent.cognition import _json_cikar
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.puzzle")

MAX_DERINLIK = 4
MAX_PARCA = 64
MAX_ALT = 7
MAX_SORU = 5
PLAN_TOKEN = 1400

#: Parça türleri
FAZ = "faz"
ADIM = "adim"
WALKTHROUGH = "walkthrough"
TURLER = (FAZ, ADIM, WALKTHROUGH)

#: Durumlar
BEKLIYOR, AKTIF, TAMAM, REVIZE = "bekliyor", "aktif", "tamam", "revize"
DURUMLAR = (BEKLIYOR, AKTIF, TAMAM, REVIZE)


@dataclass
class Parca:
    """Puzzle'ın tek düğümü: faz, adım veya walkthrough."""

    id: str
    baslik: str
    tur: str = FAZ
    aciklama: str = ""
    durum: str = BEKLIYOR
    olcut: str = ""                       # "tamam" sayılma ölçütü
    sorular: List[str] = field(default_factory=list)
    cevap: str = ""
    not_: str = ""                        # kullanıcı revizyon notu
    cocuklar: List["Parca"] = field(default_factory=list)
    derinlik: int = 0

    # ── gezinme ──
    def bul(self, kid: str) -> Optional["Parca"]:
        if self.id == kid:
            return self
        for c in self.cocuklar:
            bulunan = c.bul(kid)
            if bulunan is not None:
                return bulunan
        return None

    def duz(self) -> List["Parca"]:
        """Ön sıralı düz liste."""
        liste = [self]
        for c in self.cocuklar:
            liste.extend(c.duz())
        return liste

    def yaprak(self) -> Optional["Parca"]:
        """İlk tamamlanmamış yaprak (yürütülecek iş)."""
        if not self.cocuklar:
            return None if self.durum == TAMAM else self
        for c in self.cocuklar:
            aday = c.yaprak()
            if aday is not None:
                return aday
        return None

    # ── serileştirme ──
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "baslik": self.baslik, "tur": self.tur,
            "aciklama": self.aciklama, "durum": self.durum,
            "olcut": self.olcut, "sorular": list(self.sorular),
            "cevap": self.cevap, "not": self.not_, "derinlik": self.derinlik,
            "cocuklar": [c.to_dict() for c in self.cocuklar],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], derinlik: int = 0) -> "Parca":
        return cls(
            id=str(d.get("id") or "p1"),
            baslik=str(d.get("baslik") or "")[:200],
            tur=str(d.get("tur") or FAZ) if d.get("tur") in TURLER else FAZ,
            aciklama=str(d.get("aciklama") or "")[:600],
            durum=(str(d.get("durum")) if d.get("durum") in DURUMLAR
                   else BEKLIYOR),
            olcut=str(d.get("olcut") or "")[:300],
            sorular=[str(s)[:200] for s in (d.get("sorular") or [])][:MAX_SORU],
            cevap=str(d.get("cevap") or "")[:2000],
            not_=str(d.get("not") or "")[:1000],
            derinlik=derinlik,
            cocuklar=[Parca.from_dict(c, derinlik + 1)
                      for c in (d.get("cocuklar") or [])][:MAX_ALT],
        )


class Puzzle:
    """Bir görevin iç içe plan ağacı."""

    def __init__(self, gorev: str = "", kok: Optional[Parca] = None) -> None:
        self.gorev = str(gorev)[:600]
        self.kok = kok
        self.aktif_id = kok.id if kok is not None else ""
        self.olusturma = time.time()
        self.revizyon = 0
        #: ⚠ A) Walkthrough'da **duruldu mu** (kullanıcı cevabı bekleniyor)
        self.bekliyor = False
        #: B) Çalışma sırasında kaç kez dinamik alt faz açıldı
        self.genisletme = 0

    # ── sorgular ──
    def parcalar(self) -> List[Parca]:
        return self.kok.duz() if self.kok is not None else []

    def sayi(self) -> int:
        return len(self.parcalar())

    def bul(self, kid: str) -> Optional[Parca]:
        return self.kok.bul(kid) if self.kok is not None else None

    def aktif(self) -> Optional[Parca]:
        return self.bul(self.aktif_id) if self.aktif_id else None

    def tamamlanan(self) -> int:
        return sum(1 for p in self.parcalar() if p.durum == TAMAM)

    def tamamlandi(self) -> bool:
        return bool(self.kok is not None
                    and self.kok.durum == TAMAM
                    and self.kok.yaprak() is None)

    # ── ilerleme ──
    def aktif_yaprak(self) -> Optional[Parca]:
        """Şu an yürütülmesi gereken yaprak."""
        aktif = self.aktif()
        if aktif is not None:
            y = aktif.yaprak() or aktif
            return y
        return self.kok.yaprak() if self.kok is not None else None

    def ilerle(self) -> Optional[Parca]:
        """
        Aktif yaprağı **tamam** işaretler ve sıradaki yaprağı etkinleştirir.

        ⚠ Faz bitince üst fazın tüm çocukları bittiyse faz da tamamlanır.
        """
        yaprak = self.aktif_yaprak()
        if yaprak is not None:
            yaprak.durum = TAMAM
            ust = self._ust(yaprak.id)
            while ust is not None:
                if all(c.durum == TAMAM for c in ust.cocuklar):
                    ust.durum = TAMAM
                    ust = self._ust(ust.id)
                else:
                    break
        sonraki = self.kok.yaprak() if self.kok is not None else None
        if sonraki is None:
            if self.kok is not None:
                self.kok.durum = TAMAM
            self.aktif_id = ""
            return None
        sonraki.durum = AKTIF
        self.aktif_id = sonraki.id
        return sonraki

    def _ust(self, kid: str) -> Optional[Parca]:
        """Parçanın üst parçasını bulur (yol tabanlı kimlik)."""
        if "." not in kid:
            return None
        ust_id = kid.rsplit(".", 1)[0]
        return self.bul(ust_id)

    def bekleyen_sorular(self) -> List[Dict[str, Any]]:
        """
        Cevap bekleyen walkthrough'lar (aktif yol üzerinde).

        ⚠ A) Ajan bunları görünce **durur**; kullanıcı cevaplayınca devam eder.
        """
        yol: List[Parca] = []
        aktif = self.aktif()
        if aktif is not None:
            for parca in self.parcalar():
                if aktif.id == parca.id or aktif.id.startswith(parca.id + "."):
                    yol.append(parca)
        adaylar = yol or self.parcalar()
        sonuc: List[Dict[str, Any]] = []
        for parca in adaylar:
            if parca.sorular and not parca.cevap:
                sonuc.append({
                    "id": parca.id, "baslik": parca.baslik,
                    "sorular": list(parca.sorular),
                })
        return sonuc[:3]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gorev": self.gorev,
            "kok": self.kok.to_dict() if self.kok is not None else None,
            "aktif_id": self.aktif_id,
            "olusturma": self.olusturma,
            "revizyon": self.revizyon,
            "sayi": self.sayi(),
            "tamamlanan": self.tamamlanan(),
            "tamamlandi": self.tamamlandi(),
            "bekliyor": self.bekliyor,
            "genisletme": self.genisletme,
            "sorular": self.bekleyen_sorular(),
        }

    def revize(self, kid: str, not_: str) -> Optional[Parca]:
        """Kullanıcı bir parçayı değiştirmek istediğinde işaretler."""
        parca = self.bul(kid)
        if parca is None:
            return None
        parca.not_ = str(not_)[:1000]
        parca.durum = REVIZE
        self.revizyon += 1
        self.aktif_id = parca.id
        logger.info(f"Puzzle revizyonu: {parca.id} — {parca.not_[:60]}")
        return parca

    def cevapla(self, kid: str, cevap: str) -> Optional[Parca]:
        """Walkthrough sorusuna cevap yazar."""
        parca = self.bul(kid)
        if parca is None:
            return None
        parca.cevap = str(cevap)[:2000]
        if parca.durum == BEKLIYOR:
            parca.durum = AKTIF
        return parca

    # ── özet ──
    def ozet(self, sinir: int = 1500) -> str:
        """Modele verilecek kısa plan metni."""
        if self.kok is None:
            return ""
        satirlar = [f"PUZZLE: {self.gorev[:160]}",
                    f"(parça {self.tamamlanan()}/{self.sayi()} tamam)"]
        for parca in self.parcalar():
            isaret = {"tamam": "✓", "aktif": "▶", "revize": "✎"}.get(
                parca.durum, "·")
            girinti = "  " * parca.derinlik
            satirlar.append(f"{girinti}{isaret} [{parca.id}] {parca.baslik}"
                            + (f" — {parca.olcut}" if parca.olcut else ""))
            if parca.not_:
                satirlar.append(f"{girinti}   ✎ kullanıcı notu: {parca.not_}")
            for soru in parca.sorular[:3]:
                durum = "cevaplandı" if parca.cevap else "SORU"
                satirlar.append(f"{girinti}   ? ({durum}) {soru}")
        return "\n".join(satirlar)[:sinir]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Puzzle":
        p = cls(gorev=str(d.get("gorev") or ""))
        kok = d.get("kok")
        p.kok = Parca.from_dict(kok) if isinstance(kok, dict) else None
        p.aktif_id = str(d.get("aktif_id") or "")
        p.olusturma = float(d.get("olusturma") or time.time())
        p.revizyon = int(d.get("revizyon") or 0)
        return p

    # ── C) kasa (Obsidian markdown) ──
    def kasaya_yaz(self, kasa: Path) -> Optional[Path]:
        """
        Tamamlanan (veya kullanıcı onaylı) planı kasaya **markdown** yazar.

        ⚠ Ölçüt: yalnız `tamamlanan >= 1` ise yazılır; boş plan kasayı kirletmez.
        Kasa notu `kind: reference`, `tags: [puzzle, plan]` taşır; böylece
        eğitim süzgecinden geçer (MUST 14).
        """
        if self.kok is None or self.tamamlanan() < 1:
            return None
        import re as _re

        slug = _re.sub(r"[^a-z0-9]+", "-",
                       self.gorev.lower().replace("ı", "i")
                       .replace("ş", "s").replace("ğ", "g")
                       .replace("ü", "u").replace("ö", "o")
                       .replace("ç", "c")).strip("-")[:60] or "plan"
        hedef = Path(kasa) / "10_concepts" / f"puzzle-{slug}.md"
        satirlar = [
            "---",
            f'title: "Puzzle: {self.gorev[:70]}"',
            "kind: reference",
            "tags: [puzzle, plan, walkthrough]",
            "confidence: 0.7",
            f"created: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
            f"parca: {self.sayi()}",
            f"tamamlanan: {self.tamamlanan()}",
            "---",
            "",
            f"# Puzzle — {self.gorev[:120]}",
            "",
            "## Fazlar",
            "",
        ]
        for parca in self.parcalar():
            if parca.derinlik == 0:
                continue
            girinti = "  " * max(0, parca.derinlik - 1)
            isaret = "x" if parca.durum == TAMAM else " "
            satirlar.append(f"{girinti}- [{isaret}] **{parca.baslik}**"
                            + (f" — _{parca.olcut}_" if parca.olcut else ""))
            for soru in parca.sorular:
                satirlar.append(f"{girinti}  - soru: {soru}")
            if parca.cevap:
                satirlar.append(f"{girinti}  - cevap: {parca.cevap}")
            if parca.not_:
                satirlar.append(f"{girinti}  - revizyon: {parca.not_}")
        try:
            hedef.parent.mkdir(parents=True, exist_ok=True)
            hedef.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
            logger.info(f"Puzzle kasaya yazıldı: {hedef}")
            return hedef
        except OSError as exc:
            logger.warning(f"Puzzle kasaya yazılamadı: {exc}")
            return None

    # ── kalıcılık ──
    def kaydet(self, yol: Path) -> bool:
        try:
            yol.parent.mkdir(parents=True, exist_ok=True)
            yol.write_text(json.dumps(self.to_dict(), ensure_ascii=False,
                                      indent=1), encoding="utf-8")
            return True
        except OSError as exc:
            logger.warning(f"Puzzle kaydedilemedi: {exc}")
            return False

    @classmethod
    def yukle(cls, yol: Path) -> Optional["Puzzle"]:
        try:
            veri = json.loads(Path(yol).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cls.from_dict(veri) if isinstance(veri, dict) else None


_ISTEM = """Sen bir **planlayıcısın**. Görevi iç içe fazlara böl.

Kurallar:
- En fazla {max_faz} üst faz üret; her faz en fazla {max_alt} alt parçaya
  ayrılabilir (istersen `alt` alanını kullan, istemezsen boş bırak).
- Her faz için 1-3 **netleştirme sorusu** yaz (`walkthrough`): kullanıcı
  bunları cevaplayınca iş daha doğru yapılır. Sorular kısa ve somut olsun.
- Her parçaya **ölçüt** (`olcut`) yaz: bu parça ne zaman "tamam" sayılır?
- Türkçe yaz. Yalnızca JSON döndür:

{{"hedef": "tek cümlelik hedef",
  "parcalar": [
    {{"baslik": "Faz 1 — keşif",
      "aciklama": "ne yapılacak",
      "olcut": "hangi kanıt tamam sayılır",
      "walkthrough": ["soru 1", "soru 2"],
      "alt": [{{"baslik": "1.1 ...", "olcut": "..."}}]}}]}}

GÖREV: {gorev}
"""


class PuzzlePlanner:
    """Görevi iç içe Puzzle ağacına çevirir (tek LLM çağrısı + düzeltme)."""

    def __init__(self, router: Any, *, max_faz: int = 4) -> None:
        self.router = router
        self.max_faz = max(1, min(int(max_faz or 4), MAX_ALT))

    def planla(self, gorev: str) -> Puzzle:
        """Görevden ağaç üretir; LLM yoksa/başarısızsa düz plana düşer."""
        gorev = str(gorev or "").strip()
        if not gorev:
            return Puzzle()
        if self.router is None:
            return self._duz_plan(gorev)
        istem = _ISTEM.format(gorev=gorev[:1200], max_faz=self.max_faz,
                              max_alt=MAX_ALT)
        try:
            from neurocore.agent.llm import LLMMessage

            yanit = self.router.chat([LLMMessage("user", istem)],
                                     max_tokens=PLAN_TOKEN, temperature=0.0)
            if not getattr(yanit, "ok", False):
                logger.warning("Puzzle planı: model yanıtı başarısız — "
                               f"{str(getattr(yanit, 'error', ''))[:160]}")
                return self._duz_plan(gorev)
            veri = _json_cikar(getattr(yanit, "text", "") or "")
        except Exception as exc:
            # ⚠ Teşhis için görünür: plan alınamazsa düz plana düşer
            logger.warning(f"Puzzle planı alınamadı: "
                           f"{type(exc).__name__}: {exc}")
            return self._duz_plan(gorev)

        kok = self._agac(veri, gorev)
        if kok is None:
            logger.warning(f"Puzzle: JSON ağacı kurulamadı "
                           f"(tip={type(veri).__name__}, "
                           f"anahtar={list(veri)[:4] if isinstance(veri, dict) else '-'})")
            return self._duz_plan(gorev)
        return self._etkin(Puzzle(gorev=gorev, kok=kok))

    @staticmethod
    def _etkin(puzzle: Puzzle) -> Puzzle:
        """İlk yaprağı **aktif** yapar (yürütme buradan başlar)."""
        if puzzle.kok is None:
            return puzzle
        yaprak = puzzle.kok.yaprak()
        if yaprak is not None:
            yaprak.durum = AKTIF
            puzzle.aktif_id = yaprak.id
        return puzzle

    # ── ağaç kurucular ──
    def _agac(self, veri: Any, gorev: str) -> Optional[Parca]:
        if not isinstance(veri, dict):
            return None
        ham = veri.get("parcalar") or veri.get("fazlar") or []
        if not isinstance(ham, list) or not ham:
            return None
        kok = Parca(id="p0", baslik=str(veri.get("hedef") or gorev)[:200],
                    tur=FAZ, derinlik=0, durum=AKTIF)
        toplam = 0
        for i, oge in enumerate(ham[:self.max_faz], start=1):
            parca = self._parca(oge, f"p{i}", derinlik=1)
            if parca is None:
                continue
            kok.cocuklar.append(parca)
            toplam += len(parca.duz())
            if toplam > MAX_PARCA:
                break
        if not kok.cocuklar:
            return None
        return kok

    def _parca(self, oge: Any, kid: str, derinlik: int) -> Optional[Parca]:
        if derinlik > MAX_DERINLIK:
            return None
        if isinstance(oge, str):
            return Parca(id=kid, baslik=oge[:200], tur=FAZ,
                         derinlik=derinlik, durum=BEKLIYOR)
        if not isinstance(oge, dict):
            return None
        sorular = oge.get("walkthrough") or oge.get("sorular") or []
        parca = Parca(
            id=kid,
            baslik=str(oge.get("baslik") or oge.get("title") or "")[:200],
            tur=FAZ,
            aciklama=str(oge.get("aciklama") or oge.get("description") or "")[:600],
            olcut=str(oge.get("olcut") or oge.get("check") or "")[:300],
            sorular=[str(s)[:200] for s in sorular][:MAX_SORU],
            derinlik=derinlik,
        )
        if not parca.baslik:
            return None
        altlar = oge.get("alt") or oge.get("children") or []
        if isinstance(altlar, list):
            for j, alt in enumerate(altlar[:MAX_ALT], start=1):
                cocuk = self._parca(alt, f"{kid}.{j}", derinlik + 1)
                if cocuk is not None:
                    parca.cocuklar.append(cocuk)
        return parca

    _GENISLET_ISTEM = """Bir plan parçası çok kaba kaldı; onu **alt parçalara** böl.

Parça: {baslik}
Açıklama: {aciklama}
Ölçüt: {olcut}

En fazla {max_alt} alt parça üret; her birine `olcut` yaz. Yalnızca JSON:
{{"alt": [{{"baslik": "...", "olcut": "..."}}, ...]}}
"""

    def genislet(self, parca: "Parca", *, max_alt: int = 4) -> List["Parca"]:
        """
        ⚠ B) Çalışma sırasında **dinamik alt faz** açar.

        Kaba kalan bir yaprağı (ör. "Aşama 2") daha küçük parçalara böler.
        Hata durumunda boş liste döner — akış bozulmaz.
        """
        if self.router is None or parca is None:
            return []
        if parca.cocuklar or parca.derinlik >= MAX_DERINLIK:
            return []
        istem = self._GENISLET_ISTEM.format(
            baslik=parca.baslik[:160], aciklama=parca.aciklama[:300],
            olcut=parca.olcut[:200], max_alt=max(1, min(max_alt, MAX_ALT)))
        try:
            from neurocore.agent.llm import LLMMessage

            yanit = self.router.chat([LLMMessage("user", istem)],
                                     max_tokens=700, temperature=0.0)
            if not getattr(yanit, "ok", False):
                return []
            veri = _json_cikar(getattr(yanit, "text", "") or "")
        except Exception as exc:
            logger.debug(f"Puzzle genişletilemedi: {exc}")
            return []
        ham = (veri or {}).get("alt") if isinstance(veri, dict) else None
        if not isinstance(ham, list):
            return []
        yeni: List[Parca] = []
        for j, oge in enumerate(ham[:max_alt], start=1):
            cocuk = self._parca(oge, f"{parca.id}.{j}", parca.derinlik + 1)
            if cocuk is not None:
                yeni.append(cocuk)
        if yeni:
            parca.cocuklar.extend(yeni)
            logger.info(f"Puzzle genişletildi: {parca.id} → {len(yeni)} alt parça")
        return yeni

    def _duz_plan(self, gorev: str) -> Puzzle:
        """LLM yoksa: tek faz, üç adım (akış bozulmaz)."""
        kok = Parca(id="p0", baslik=gorev[:120], tur=FAZ, durum=AKTIF)
        for i, baslik in enumerate(("Durumu incele", "İşi yap",
                                    "Sonucu doğrula"), start=1):
            kok.cocuklar.append(Parca(id=f"p{i}", baslik=baslik, tur=ADIM,
                                      derinlik=1))
        kok.cocuklar[0].durum = AKTIF
        return Puzzle(gorev=gorev, kok=kok)
