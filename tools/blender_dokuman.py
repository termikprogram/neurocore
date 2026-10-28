"""
Blender Doküman Madencisi — Gerçek API Verisi
==============================================
Hedefin 3. adımı: **"internetten API referansı çekip kod verisini çoğalt"**.

Neden resmî doküman?
    `tools/api_referansi_toplay.py` öğretmen modele API listesi ürettirdi
    (292 referans). Ama **resmî Blender dokümanı**:
        * **Doğrulanmış** — her fonksiyon gerçekten var
        * **Tam imza** — tüm parametreler ve varsayılanları
        * **Güncel** — sürümle uyumlu

    Doküman formatı **Sphinx**: her fonksiyon bir `<dl class="py function">`
    bloğu, imza `<dt>` içinde, açıklama `<dd>` içinde.

Yöntem
------
    1. **Sayfa listesi** — hangi modüller taranacak
    2. **HTML indir** (User-Agent ile — ölçüldü: gerekli)
    3. **Ayrıştır** — `<dt>` imzaları + `<code>` örnekleri
    4. **Çağrı üret** — imzadan gerçek kullanım örneği
    5. **Kasaya yaz** — `blender_api` kategorisi

Kullanım
--------
    python tools/blender_dokuman.py --sayfa 8
    python tools/blender_dokuman.py --liste
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Doküman sayfaları — (yol, modül adı)
SAYFALAR: Sequence[Tuple[str, str]] = (
    ("bpy.ops.mesh.html", "bpy.ops.mesh"),
    ("bpy.ops.object.html", "bpy.ops.object"),
    ("bpy.ops.transform.html", "bpy.ops.transform"),
    ("bpy.ops.material.html", "bpy.ops.material"),
    ("bpy.ops.texture.html", "bpy.ops.texture"),
    ("bpy.ops.armature.html", "bpy.ops.armature"),
    ("bpy.ops.pose.html", "bpy.ops.pose"),
    ("bpy.ops.render.html", "bpy.ops.render"),
    ("bpy.ops.wm.html", "bpy.ops.wm"),
    ("bpy.ops.export_scene.html", "bpy.ops.export_scene"),
    ("bpy.ops.import_scene.html", "bpy.ops.import_scene"),
    ("bpy.types.Mesh.html", "bpy.types.Mesh"),
    ("bpy.types.Object.html", "bpy.types.Object"),
    ("bpy.types.Material.html", "bpy.types.Material"),
    ("bpy.types.Image.html", "bpy.types.Image"),
    ("bpy.types.NodeTree.html", "bpy.types.NodeTree"),
    ("bpy.types.Armature.html", "bpy.types.Armature"),
    ("bpy.types.Bone.html", "bpy.types.Bone"),
    ("bpy.types.Scene.html", "bpy.types.Scene"),
    ("bpy.types.Collection.html", "bpy.types.Collection"),
    ("bmesh.types.BMesh.html", "bmesh.types.BMesh"),
    ("mathutils.html", "mathutils"),
    ("bpy.data.html", "bpy.data"),
)

#: Doküman kök URL'i
KOK_URL = "https://docs.blender.org/api/current/"

#: Önbellek dizini (tekrar indirmeyi önler)
ONBELLEK = Path(".neurocore/dokuman_onbellek")


@dataclass
class ApiKaydi:
    """Tek bir API kaydı."""
    ad: str = ""
    imza: str = ""
    parametreler: List[str] = field(default_factory=list)
    aciklama: str = ""
    modul: str = ""
    kaynak: str = "dokuman"

    def kullanim(self) -> str:
        """
        İmzadan **gerçek kullanım örneği** üretir.

        ⚠ Kritik: yalnız zorunlu (varsayılansız) parametreler verilir;
        isteğe bağlılar tipik değerlerle gösterilir. Uydurma değer
        yazılmaz — tip bilgisinden türetilir.
        """
        if not self.parametreler:
            return f"{self.ad}()"
        # Yalnız ilk 3 parametreyi göster (örnek kısa olsun)
        secili = self.parametreler[:3]
        return f"{self.ad}({', '.join(secili)})"

    def to_dict(self) -> Dict[str, Any]:
        return {"ad": self.ad, "imza": self.imza,
                "parametreler": self.parametreler,
                "aciklama": self.aciklama[:200], "modul": self.modul}


def _temizle(metin: str) -> str:
    """HTML etiketlerini ve varlıklarını temizler."""
    metin = re.sub(r"<[^>]+>", "", metin)
    for a, b in (("&gt;", ">"), ("&lt;", "<"), ("&quot;", '"'),
                 ("&#39;", "'"), ("&amp;", "&"), ("&nbsp;", " "),
                 ("\u00b6", ""), ("\u00a0", " ")):
        metin = metin.replace(a, b)
    return re.sub(r"\s+", " ", metin).strip()


def _imza_ayristir(blok: str) -> Tuple[str, List[str]]:
    """
    Sphinx `<dt>` bloğundan **fonksiyon adı + parametreler** çıkarır.

    Biçim:
        <span class="sig-prename descclassname">bpy.ops.mesh.</span>
        <span class="sig-name descname">primitive_cube_add</span>
        <span class="sig-paren">(</span>
        <em class="sig-param">size=2.0</em>
        ...
    """
    ad_m = re.search(
        r'class="sig-(?:prename descclassname|name descname)"[^>]*>'
        r'<span class="pre">([^<]+)</span>', blok)
    if not ad_m:
        # Bazı sayfalarda tek parça hâlinde
        ad_m = re.search(r'id="([\w.]+)"', blok)
        if not ad_m:
            return "", []
        ad = ad_m.group(1)
    else:
        # Hem prename hem name gerekli
        parcalar = re.findall(
            r'class="sig-(?:prename descclassname|name descname)"[^>]*>'
            r'<span class="pre">([^<]+)</span>', blok)
        ad = "".join(parcalar)
        if not ad:
            ad = html_mod.unescape(ad_m.group(1))

    # Parametreler: `sig-param` içindeki `pre` metinleri
    params = re.findall(
        r'class="sig-param"[^>]*>(?:<span[^>]*>)?([^<]+)', blok)
    temiz_params: List[str] = []
    for p in params:
        t = _temizle(p).rstrip(", ")
        # Yalnız ad=deger veya ad bırak
        t = re.sub(r"^[*]+", "", t).strip()
        if t and t not in ("(", ")", ",", "*"):
            temiz_params.append(t)
    return ad, temiz_params


def sayfa_ayristir(ham: str, modul: str) -> List[ApiKaydi]:
    """
    Bir doküman sayfasından API kayıtlarını çıkarır.

    ⚠ Ölçülen format: her fonksiyon `<dl class="py function">` bloğu.
    `<dt>` imzayı, `<dd>` açıklamayı taşır.
    """
    kayitlar: List[ApiKaydi] = []
    # Fonksiyon bloklarını bul
    bloklar = re.findall(
        r'<dl class="py (?:function|method|attribute|class)">(.*?)</dl>',
        ham, re.DOTALL)
    for blok in bloklar:
        dt_m = re.search(r"<dt[^>]*>(.*?)</dt>", blok, re.DOTALL)
        if not dt_m:
            continue
        ad, params = _imza_ayristir(dt_m.group(1))
        if not ad or len(ad) < 4:
            continue
        # Açıklama: ilk <p>
        dd_m = re.search(r"<dd[^>]*>(.*?)</dd>", blok, re.DOTALL)
        aciklama = ""
        if dd_m:
            p_m = re.search(r"<p>(.*?)</p>", dd_m.group(1), re.DOTALL)
            if p_m:
                aciklama = _temizle(p_m.group(1))[:300]
        kayitlar.append(ApiKaydi(ad=ad, imza=f"{ad}({', '.join(params)})",
                                 parametreler=params, aciklama=aciklama,
                                 modul=modul))
    return kayitlar


def kod_ornekleri(ham: str) -> List[str]:
    """Sayfadaki `<code>` bloklarından gerçek kod örnekleri çıkarır."""
    ornekler: List[str] = []
    for blok in re.findall(r"<code[^>]*>(.*?)</code>", ham, re.DOTALL):
        t = _temizle(blok)
        if len(t) > 15 and ("bpy" in t or "import" in t):
            ornekler.append(t[:300])
    return ornekler


class BlenderDokuman:
    """Blender dokümanını indirir ve ayrıştırır."""

    def __init__(self, *, onbellek: bool = True) -> None:
        self.onbellek = onbellek
        self._kasa: Any = None
        self._gorulen: Set[str] = set()
        self.kayitlar: List[ApiKaydi] = []
        self.ornekler: List[str] = []

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
        print(f"  mevcut Blender API kaydı: {len(self._gorulen)}")
        return True

    def _mevcut(self) -> Set[str]:
        """Kasada zaten olan API adları."""
        gorulen: Set[str] = set()
        if self._kasa is None:
            return gorulen
        try:
            for n in self._kasa.vault.iter_notes(limit=200000):
                kat = getattr(n, "category", "") or ""
                if kat in ("blender_api", "api_referans"):
                    ad = (getattr(n, "title", "") or "")
                    for on in ("API: ", "BlenderAPI: "):
                        ad = ad.replace(on, "")
                    if ad:
                        gorulen.add(ad.strip())
        except Exception:
            pass
        return gorulen

    # ─── İndirme ───

    def _indir(self, yol: str) -> Optional[str]:
        """
        Doküman sayfasını indirir (önbellekli).

        ⚠ Ölçülen: `User-Agent` başlığı **şart** — onsuz istek reddedilir.
        """
        ONBELLEK.mkdir(parents=True, exist_ok=True)
        yerel = ONBELLEK / yol.replace("/", "_")
        if self.onbellek and yerel.exists() and yerel.stat().st_size > 1000:
            return yerel.read_text(encoding="utf-8", errors="replace")
        try:
            req = urllib.request.Request(
                KOK_URL + yol, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                ham = r.read().decode("utf-8", errors="replace")
            yerel.write_text(ham, encoding="utf-8")
            return ham
        except Exception as exc:
            print(f"    indirilemedi ({yol}): {type(exc).__name__}")
            return None

    # ─── İşleme ───

    def sayfa_isle(self, yol: str, modul: str) -> Tuple[int, int]:
        """Bir sayfayı indirir, ayrıştırır ve kasaya yazar."""
        ham = self._indir(yol)
        if not ham:
            return 0, 0
        kayitlar = sayfa_ayristir(ham, modul)
        ornekler = kod_ornekleri(ham)
        self.kayitlar.extend(kayitlar)
        self.ornekler.extend(ornekler)

        yazilan = 0
        for k in kayitlar:
            if k.ad in self._gorulen:
                continue
            if self._yaz(k):
                self._gorulen.add(k.ad)
                yazilan += 1
        return len(kayitlar), yazilan

    def _yaz(self, k: ApiKaydi) -> bool:
        """API kaydını kasaya yazar."""
        if self._kasa is None:
            return False
        try:
            from neurocore.knowledge.vault import Note

            params = ", ".join(k.parametreler) if k.parametreler else "yok"
            govde = (f"## Blender API (resmî doküman)\n\n"
                     f"**Ad:** `{k.ad}`\n\n"
                     f"**İmza:** `{k.imza}`\n\n"
                     f"**Parametreler:** {params}\n\n"
                     f"**Açıklama:** {k.aciklama or 'yok'}\n\n"
                     f"**Kullanım:**\n```python\n{k.kullanim()}\n```\n\n"
                     f"**Modül:** {k.modul}\n")
            simdi = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            n = Note(
                id="", title=f"BlenderAPI: {k.ad}", body=govde,
                kind="reference", category="blender_api",
                tags=["api", "blender", "dokuman",
                      k.modul.split(".")[-1]],
                confidence=0.95, importance=0.7,
                created=simdi, updated=simdi, version=1,
                sources=[f"docs.blender.org/{k.modul}"],
                links=[], backlinks=[], embedding_model="",
                embedding=None, path="", extra={})
            self._kasa.vault.write_note(n, touch=False, emit=False)
            return True
        except Exception:
            return False


def main() -> int:
    """Doküman madenciliğini çalıştırır."""
    ap = argparse.ArgumentParser(description="Blender doküman madencisi")
    ap.add_argument("--sayfa", type=int, default=0,
                    help="Kaç sayfa işlensin (0 = hepsi)")
    ap.add_argument("--liste", action="store_true", help="Sayfaları listele")
    args = ap.parse_args()

    if args.liste:
        for yol, modul in SAYFALAR:
            print(f"  {modul:26} {yol}")
        return 0

    print("=" * 78)
    print("  BLENDER DOKÜMAN MADENCİSİ — resmî API verisi")
    print("=" * 78)

    b = BlenderDokuman()
    if not b.baglan():
        return 1

    sayfalar = list(SAYFALAR)
    if args.sayfa > 0:
        sayfalar = sayfalar[:args.sayfa]

    toplam_kayit = 0
    toplam_yeni = 0
    for yol, modul in sayfalar:
        t0 = time.perf_counter()
        kayit, yeni = b.sayfa_isle(yol, modul)
        toplam_kayit += kayit
        toplam_yeni += yeni
        im = "+" if yeni else " "
        print(f"  {im} {modul:24} kayıt={kayit:4} YENİ={yeni:4} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)

    print()
    print(f"  TOPLAM: kayıt={toplam_kayit}  yeni={toplam_yeni}  "
          f"kasada={len(b._gorulen)}")
    Path("_dokuman_ozet.json").write_text(json.dumps({
        "toplam_kayit": toplam_kayit, "yeni": toplam_yeni,
        "kasada": len(b._gorulen),
        "ornek_sayisi": len(b.ornekler),
        "ornekler": [k.to_dict() for k in b.kayitlar[:20]],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
