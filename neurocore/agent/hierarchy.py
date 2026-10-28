"""
Hiyerarşik Planlama — hedef → alt hedefler → adımlar
=====================================================
Düz plan ("1, 2, 3 yap") karmaşık görevlerde dağılır: model adımları
karıştırır, bağımlılıkları kaçırır. Hiyerarşi bunu iki düzeye böler:

```text
HEDEF   kılıç modeli üret, animasyon ekle, 4 açıdan önizleme al
 ├─ ALT HEDEF 1  varlık üret        → asset.uret(kategori=silah)
 │                                  → asset.katalog(dogrula=True)
 ├─ ALT HEDEF 2  animasyon          → asset.animasyon(tur=kilic_savurma)
 └─ ALT HEDEF 3  doğrulama          → asset.onizleme(aci=4)
```

Tasarım sınırları (MUST)
------------------------
```
· en fazla 4 alt hedef · alt hedef başına en fazla 3 adım
· araç adları **verilen listeden** seçilir (uydurma araç yok)
· LLM yanıtı çözülemezse hiyerarşi **boş** döner → ajan düz plana düşer
· hiçbir adım otomatik çalıştırılmaz: plan yalnız **yönerge**dir
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from neurocore.agent.cognition import _json_cikar
from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.hierarchy")

#: Sınırlar
MAX_ALT = 4
MAX_ADIM_PER_ALT = 3
PLAN_TOKEN = 900


@dataclass
class Adim:
    """Alt hedefin tek adımı."""

    tool: str = ""
    args: str = ""
    neden: str = ""

    def metin(self) -> str:
        parca = f"`{self.tool}`"
        if self.args:
            parca += f" ({self.args})"
        if self.neden:
            parca += f" — {self.neden}"
        return parca


@dataclass
class AltHedef:
    """Tek alt hedef ve adımları."""

    baslik: str = ""
    adimlar: List[Adim] = field(default_factory=list)
    dogrulama: str = ""          # bu alt hedef nasıl doğrulanır

    def metin(self, sira: int) -> str:
        satirlar = [f"{sira}. {self.baslik}"]
        for a in self.adimlar:
            satirlar.append(f"   - {a.metin()}")
        if self.dogrulama:
            satirlar.append(f"   ✓ doğrulama: {self.dogrulama}")
        return "\n".join(satirlar)


@dataclass
class Hiyerarsi:
    """Bir görevin iki düzeyli planı."""

    hedef: str = ""
    alt_hedefler: List[AltHedef] = field(default_factory=list)
    ok: bool = False
    error: str = ""
    raw: str = ""

    def ozet(self, sinir: int = 1400) -> str:
        """Mesaja eklenecek plan metni."""
        if not self.alt_hedefler:
            return ""
        parcalar = [f"HEDEF: {self.hedef[:200]}", "ALT HEDEFLER:"]
        for i, alt in enumerate(self.alt_hedefler, 1):
            parcalar.append(alt.metin(i))
        return "\n".join(parcalar)[:sinir]

    def araclar(self) -> List[str]:
        return [a.tool for alt in self.alt_hedefler for a in alt.adimlar
                if a.tool]

    @classmethod
    def coz(cls, metin: str) -> "Hiyerarsi":
        """Modelin JSON yanıtını ayrıştırır (dayanıklı)."""
        veri = _json_cikar(metin)
        if not isinstance(veri, dict):
            return cls(ok=False, error="json çözülemedi", raw=str(metin)[:400])
        ham = veri.get("subgoals") or veri.get("alt_hedefler") or []
        if not isinstance(ham, list):
            return cls(ok=False, error="subgoals liste değil")
        altlar: List[AltHedef] = []
        for öğe in ham[:MAX_ALT]:
            if isinstance(öğe, str):
                altlar.append(AltHedef(baslik=öğe[:160]))
                continue
            if not isinstance(öğe, dict):
                continue
            adimlar: List[Adim] = []
            for a in (öğe.get("steps") or öğe.get("adimlar") or [])[:MAX_ADIM_PER_ALT]:
                if isinstance(a, str):
                    adimlar.append(Adim(tool=a[:80]))
                    continue
                if not isinstance(a, dict):
                    continue
                adimlar.append(Adim(
                    tool=str(a.get("tool") or "")[:80],
                    args=str(a.get("args") or "")[:160],
                    neden=str(a.get("why") or a.get("neden") or "")[:160],
                ))
            altlar.append(AltHedef(
                baslik=str(öğe.get("title") or öğe.get("baslik") or "")[:200],
                adimlar=[a for a in adimlar if a.tool],
                dogrulama=str(öğe.get("check") or öğe.get("dogrulama") or "")[:200],
            ))
        altlar = [a for a in altlar if a.baslik or a.adimlar]
        if not altlar:
            return cls(ok=False, error="alt hedef yok", raw=str(metin)[:400])
        return cls(
            hedef=str(veri.get("goal") or veri.get("hedef") or "")[:300],
            alt_hedefler=altlar, ok=True, raw=str(metin)[:600],
        )


_ISTEM = """Sen bir **planlayıcısın**. Görevi 2-4 alt hedefe böl ve her alt
hedef için en fazla 3 araç adımı öner. Yalnızca JSON döndür:

{{"goal": "kısa hedef",
  "subgoals": [
    {{"title": "alt hedef",
      "steps": [{{"tool": "araç.adı", "args": "kısa argüman", "why": "gerekçe"}}],
      "check": "bu alt hedef nasıl doğrulanır"}}]}}

Kurallar:
- `tool` YALNIZCA verilen listeden seçilir; listede yoksa o adımı yazma.
- Alt hedefler sıralı ve bağımlılıkları doğru olmalı.
- Her alt hedefin bir `check` (doğrulama) alanı olmalı.
- Görev kısaysa 2 alt hedef yeterli; şişirme.

GÖREV: {gorev}

KULLANILABİLİR ARAÇLAR: {araclar}
"""


def planla(router: Any, gorev: str, *, araclar: Sequence[str],
           max_token: int = PLAN_TOKEN) -> Hiyerarsi:
    """
    İki düzeyli plan üretir (tek LLM çağrısı).

    ⚠ Her hata durumunda **boş** hiyerarşi döner; ajan düz plana düşer.
    """
    if router is None or not str(gorev or "").strip():
        return Hiyerarsi(hedef=str(gorev)[:300], ok=False, error="router/görev yok")
    liste = ", ".join(list(araclar)[:40]) or "(liste boş)"
    istem = _ISTEM.format(gorev=str(gorev)[:800], araclar=liste)
    try:
        from neurocore.agent.llm import LLMMessage

        yanit = router.chat([LLMMessage("user", istem)], max_tokens=max_token,
                            temperature=0.0)
    except Exception as exc:
        logger.debug(f"Hiyerarşi çağrısı başarısız: {type(exc).__name__}: {exc}")
        return Hiyerarsi(hedef=str(gorev)[:300], ok=False,
                         error=f"{type(exc).__name__}: {exc}")
    if not getattr(yanit, "ok", False):
        return Hiyerarsi(hedef=str(gorev)[:300], ok=False,
                         error=str(getattr(yanit, "error", ""))[:200])
    sonuc = Hiyerarsi.coz(getattr(yanit, "text", "") or "")
    if not sonuc.hedef:
        sonuc.hedef = str(gorev)[:300]
    # ⚠ Liste dışı araç adları atılır (halüsinasyonu plana sokmayız)
    izinli = set(araclar)
    if izinli:
        for alt in sonuc.alt_hedefler:
            alt.adimlar = [a for a in alt.adimlar if a.tool in izinli]
        sonuc.alt_hedefler = [a for a in sonuc.alt_hedefler
                              if a.baslik or a.adimlar]
        if not sonuc.alt_hedefler:
            sonuc.ok = False
            sonuc.error = "geçerli araç içeren alt hedef yok"
    return sonuc
