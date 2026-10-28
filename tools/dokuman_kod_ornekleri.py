"""
Doküman Kod Örneği Çıkarıcısı
==============================
`tools/blender_dokuman.py` API **imzalarını** topladı (1.375 kayıt).
Bu araç doküman sayfalarındaki **gerçek kod örneklerini** çıkarır.

Neden ayrı araç?
    Doküman iki tür veri taşır:
        1. **API imzaları** — "hangi fonksiyon var, parametreleri ne"
           (`blender_dokuman.py` bunu topladı)
        2. **Kod örnekleri** — "gerçek kullanım nasıl görünür"
           (bu araç)

    İkincisi **kod tamamlama** için kritik: model imzayı bilse bile
    kullanım kalıbını görmeli.

Yöntem
------
    1. **Geniş sayfa listesi** — örnek kod içeren sayfalar
    2. **HTML indir** (önbellekli)
    3. **`<code>`/`<pre>` bloklarını çıkar** — `bpy`/`import` içerenler
    4. **Çok satırlıları böl** — önek/devam çiftleri
    5. **Kasaya yaz** — `kod_ornek` kategorisi

Kullanım
--------
    python tools/dokuman_kod_ornekleri.py
    python tools/dokuman_kod_ornekleri.py --min-satir 2
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import logging
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Örnek kod içeren doküman sayfaları
SAYFALAR: Sequence[str] = (
    # Eğitimler ve örnekler
    "info_quickstart.html",
    "info_overview.html",
    "info_tutorial_addon.html",
    "info_gotchas.html",
    "info_best_practice.html",
    # API kullanım örnekleri
    "bpy.ops.mesh.html",
    "bpy.ops.object.html",
    "bpy.ops.armature.html",
    "bpy.ops.pose.html",
    "bpy.ops.render.html",
    "bpy.ops.wm.html",
    "bpy.types.Mesh.html",
    "bpy.types.Object.html",
    "bpy.types.Material.html",
    "bpy.types.Image.html",
    "bpy.types.NodeTree.html",
    "bpy.types.Armature.html",
    "bpy.types.Scene.html",
    "bpy.types.bpy_struct.html",
    "bpy.types.bpy_prop_collection.html",
    "mathutils.html",
    "mathutils.Vector.html",
    "mathutils.Matrix.html",
    "mathutils.Quaternion.html",
    "mathutils.Euler.html",
    "bpy.utils.html",
    "bpy.path.html",
    "bpy.app.html",
    "gpu.html",
    "blf.html",
    "bgl.html",
    "aud.html",
    "bpy.context.html",
    "bpy.props.html",
    "bpy.msgbus.html",
    "bpy.types.Operator.html",
    "bpy.types.Panel.html",
    "bpy.types.PropertyGroup.html",
    "bpy.types.Keymap.html",
    "bpy.types.Window.html",
    "bpy.types.UILayout.html",
)

KOK_URL = "https://docs.blender.org/api/current/"
ONBELLEK = Path(".neurocore/dokuman_onbellek")


@dataclass
class KodOrnegi:
    """Çıkarılmış kod örneği."""
    onek: str = ""
    devam: str = ""
    kaynak: str = ""
    tam: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"onek": self.onek[:200], "devam": self.devam[:200],
                "kaynak": self.kaynak}


def _temizle(metin: str) -> str:
    """HTML etiketlerini temizler."""
    metin = re.sub(r"<[^>]+>", "", metin)
    for a, b in (("&gt;", ">"), ("&lt;", "<"), ("&quot;", '"'),
                 ("&#39;", "'"), ("&amp;", "&"), ("&nbsp;", " "),
                 ("\u00b6", ""), ("\u00a0", " ")):
        metin = metin.replace(a, b)
    return metin


def kod_bloklari(ham: str) -> List[str]:
    """
    Doküman sayfasındaki **kod bloklarını** çıkarır.

    ⚠ Ölçülen format: Blender dokümanı Sphinx kullanır. Kod blokları
    `<div class="highlight-python">` içinde `<pre>` taşır. Bazıları
    yalnız `<code>` etiketi kullanır.
    """
    bloklar: List[str] = []
    # 1) Sphinx highlight blokları
    for m in re.finditer(
            r'<div class="highlight[^"]*"[^>]*>(.*?)</div>', ham, re.DOTALL):
        icerik = m.group(1)
        pre = re.search(r"<pre[^>]*>(.*?)</pre>", icerik, re.DOTALL)
        if pre:
            bloklar.append(_temizle(pre.group(1)))
    # 2) Düz `<pre>` blokları
    for m in re.finditer(r"<pre[^>]*>(.*?)</pre>", ham, re.DOTALL):
        bloklar.append(_temizle(m.group(1)))
    # 3) `<code>` blokları (kısa satır içi örnekler)
    for m in re.finditer(r"<code[^>]*>(.*?)</code>", ham, re.DOTALL):
        bloklar.append(_temizle(m.group(1)))

    # Tekrarları at + anlamlı olanları tut
    gorulen: Set[str] = set()
    sonuc: List[str] = []
    for b in bloklar:
        t = b.strip()
        if not t or t in gorulen:
            continue
        if len(t) < 12 or len(t) > 4000:
            continue
        # Kod gibi görünmeli
        if not any(k in t for k in ("bpy", "import ", "mathutils",
                                    "=", "def ", "class ")):
            continue
        gorulen.add(t)
        sonuc.append(t)
    return sonuc


def ornek_ciftleri(kod: str, min_satir: int = 2) -> List[KodOrnegi]:
    """
    Çok satırlı koddan **önek/devam çiftleri** üretir.

    ⚠ Kritik: devam **yeni satırdan** başlar (müfredat dersinden ders:
    aynı satıra eklemek sözdizimini bozar).
    """
    satirlar = [s for s in kod.splitlines() if s.strip()]
    if len(satirlar) < min_satir + 1:
        return []
    ciftler: List[KodOrnegi] = []
    # 2-4 satırlık önekler
    for n in (2, 3, 4):
        if len(satirlar) <= n:
            continue
        onek = "\n".join(satirlar[:n])
        devam = "\n".join(satirlar[n:n + 2])
        if not devam.strip():
            continue
        ciftler.append(KodOrnegi(onek=onek, devam=devam,
                                 tam="\n".join(satirlar)))
    return ciftler


def _gecerli_mi(onek: str, devam: str) -> bool:
    """
    Önek + devam geçerli Python mu?

    ⚠ Ölçülen tuzak: kod parçaları **yarımdır** (açık blok/parantez).
    Eksik blok `pass` ile kapatılıp denenir.
    """
    import ast

    tam = onek.rstrip("\n") + "\n" + devam
    for deneme in (tam,):
        try:
            ast.parse(deneme)
            return True
        except SyntaxError:
            pass
    # Açık bloğu kapat
    satirlar = tam.splitlines()
    mevcut = len(satirlar[-1]) - len(satirlar[-1].lstrip()) if satirlar else 0
    acik = 0
    for s in satirlar:
        t = s.split("#")[0]
        acik += t.count("(") - t.count(")")
        acik += t.count("[") - t.count("]")
        acik += t.count("{") - t.count("}")
    ek = ")" * max(0, acik)
    try:
        ast.parse(tam.rstrip() + ek + "\n" + " " * (mevcut + 4) + "pass")
        return True
    except SyntaxError:
        return False


class DokumanOrnek:
    """Doküman sayfalarından kod örneği çıkarır."""

    def __init__(self) -> None:
        self._kasa: Any = None
        self._gorulen: Set[str] = set()
        self.toplam_blok = 0
        self.toplam_cift = 0

    def baglan(self) -> bool:
        """Kasayı hazırlar."""
        try:
            from neurocore.knowledge import KnowledgeLayer

            self._kasa = KnowledgeLayer()
            self._kasa.initialize()
        except Exception as exc:
            print(f"  kasa bağlanamadı: {exc}")
            return False
        self._gorulen = self._mevcut()
        print(f"  mevcut kod örneği: {len(self._gorulen)}")
        return True

    def _mevcut(self) -> Set[str]:
        """Kasada zaten olan örnek anahtarları."""
        gorulen: Set[str] = set()
        if self._kasa is None:
            return gorulen
        try:
            for n in self._kasa.vault.iter_notes(limit=300000):
                if (getattr(n, "category", "") or "") == "kod_ornek":
                    t = (getattr(n, "title", "") or "")
                    if t:
                        gorulen.add(t.strip()[:80])
        except Exception:
            pass
        return gorulen

    def _indir(self, yol: str) -> Optional[str]:
        """Sayfayı indirir (önbellekli)."""
        ONBELLEK.mkdir(parents=True, exist_ok=True)
        yerel = ONBELLEK / yol.replace("/", "_")
        if yerel.exists() and yerel.stat().st_size > 1000:
            return yerel.read_text(encoding="utf-8", errors="replace")
        try:
            req = urllib.request.Request(
                KOK_URL + yol, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                ham = r.read().decode("utf-8", errors="replace")
            yerel.write_text(ham, encoding="utf-8")
            return ham
        except Exception:
            return None

    def sayfa_isle(self, yol: str) -> Tuple[int, int]:
        """Bir sayfadan örnek çıkarır ve yazar."""
        ham = self._indir(yol)
        if not ham:
            return 0, 0
        bloklar = kod_bloklari(ham)
        self.toplam_blok += len(bloklar)
        yazilan = 0
        for blok in bloklar:
            for cift in ornek_ciftleri(blok):
                self.toplam_cift += 1
                if not _gecerli_mi(cift.onek, cift.devam):
                    continue
                anahtar = cift.onek.strip()[:80]
                if anahtar in self._gorulen:
                    continue
                cift.kaynak = yol
                if self._yaz(cift):
                    self._gorulen.add(anahtar)
                    yazilan += 1
        return len(bloklar), yazilan

    def _yaz(self, o: KodOrnegi) -> bool:
        """Kod örneğini kasaya yazar."""
        if self._kasa is None:
            return False
        try:
            from neurocore.knowledge.vault import Note

            govde = (f"## Kod Örneği (resmî doküman)\n\n"
                     f"**Önek:**\n```python\n{o.onek}\n```\n\n"
                     f"**Devam:**\n```python\n{o.devam}\n```\n\n"
                     f"**Tam kod:**\n```python\n{o.tam[:1200]}\n```\n\n"
                     f"**Kaynak:** {o.kaynak}\n")
            simdi = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            n = Note(
                id="", title=f"KodÖrnek: {o.onek.splitlines()[-1][:60]}",
                body=govde, kind="skill", category="kod_ornek",
                tags=["kod", "ornek", "blender", "dokuman"],
                confidence=0.9, importance=0.7,
                created=simdi, updated=simdi, version=1,
                sources=[f"docs.blender.org/{o.kaynak}"],
                links=[], backlinks=[], embedding_model="",
                embedding=None, path="", extra={})
            self._kasa.vault.write_note(n, touch=False, emit=False)
            return True
        except Exception:
            return False


def main() -> int:
    """Doküman kod örneklerini çıkarır."""
    ap = argparse.ArgumentParser(description="Doküman kod örneği çıkarıcı")
    ap.add_argument("--min-satir", type=int, default=2)
    ap.add_argument("--liste", action="store_true")
    args = ap.parse_args()

    if args.liste:
        for s in SAYFALAR:
            print(f"  {s}")
        return 0

    print("=" * 78)
    print("  DOKÜMAN KOD ÖRNEĞİ ÇIKARICI")
    print("=" * 78)

    d = DokumanOrnek()
    if not d.baglan():
        return 1

    toplam_yeni = 0
    basarili = 0
    for yol in SAYFALAR:
        t0 = time.perf_counter()
        blok, yeni = d.sayfa_isle(yol)
        toplam_yeni += yeni
        if blok:
            basarili += 1
        im = "+" if yeni else " "
        print(f"  {im} {yol[:38]:40} blok={blok:3} YENİ={yeni:4} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)

    print()
    print(f"  TOPLAM: sayfa={basarili}/{len(SAYFALAR)} "
          f"blok={d.toplam_blok} çift={d.toplam_cift} "
          f"YENİ={toplam_yeni} kasada={len(d._gorulen)}")
    Path("_kod_ornek_ozet.json").write_text(json.dumps({
        "sayfa": basarili, "blok": d.toplam_blok,
        "cift": d.toplam_cift, "yeni": toplam_yeni,
        "kasada": len(d._gorulen),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
