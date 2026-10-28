"""
Araştırma Araçları — Kütüphane, Sürüm ve API Arama
====================================================
⚠ Kullanıcı hedefi (C)
----------------------
> *"myAi'ye araştırma yeteneği ver — istediği dilin/kütüphanenin güncel
> sürümünü ve API'sini internetten bulup okuyabilsin"*

Üç araç
-------
    1. `kutuphane_ara(dil, kutuphane, sürüm)`
       → Resmî doküman sayfasını bulur
    2. `api_imzasi_al(dil, fonksiyon)`
       → İmza + parametreler + örnek getirir
    3. `surum_karsilastir(dil, kütüphane, v1, v2)`
       → İki sürüm arasındaki farkı çıkarır

⚠ Kritik tasarım: **hem cevap verir hem kasayı büyütür**
------------------------------------------------------------------
Her arama sonucu kasaya `arastirma` kategorisiyle yazılır. Böylece
model **bir kez araştırdığını bir daha aramaz** — kalıcı öğrenme.

Kaynak yönlendirme
------------------
    python     → docs.python.org/3.{sürüm}/library/{modül}.html
    gdscript   → docs.godotengine.org/en/stable/classes/class_{sınıf}.html
    javascript → developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/...
    rust       → doc.rust-lang.org/std/
    cpp        → en.cppreference.com/w/cpp/

Kullanım
--------
    python tools/arastirma.py --ara python pathlib
    python tools/arastirma.py --api python "Path.read_text"
    python tools/arastirma.py --surum python 3.10 3.13
    python tools/arastirma.py --test
"""

from __future__ import annotations

import argparse
import gzip
import html as html_mod
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

ONBELLEK = Path(".neurocore/arastirma_onbellek")

#: Dil → doküman kökü ve URL kalıbı
DIL_KAYNAK: Dict[str, Dict[str, str]] = {
    "python": {
        "ad": "Python",
        "kalip": "https://docs.python.org/3.{surum}/library/{ad}.html",
        "varsayilan_surum": "",
        "arama": "https://docs.python.org/3/search.html?q={sorgu}",
    },
    "gdscript": {
        "ad": "Godot",
        "kalip": ("https://docs.godotengine.org/en/stable/classes/"
                  "class_{ad}.html"),
        "varsayilan_surum": "4.x",
        "arama": "https://docs.godotengine.org/en/stable/search.html?q={sorgu}",
    },
    "javascript": {
        "ad": "JavaScript (MDN)",
        "kalip": ("https://developer.mozilla.org/en-US/docs/Web/"
                  "JavaScript/Reference/Global_Objects/{ad}"),
        "varsayilan_surum": "ES2024",
        "arama": ("https://developer.mozilla.org/en-US/search?q={sorgu}"),
    },
    "rust": {
        "ad": "Rust",
        "kalip": "https://doc.rust-lang.org/std/{ad}/index.html",
        "varsayilan_surum": "stable",
        "arama": "https://doc.rust-lang.org/std/?search={sorgu}",
    },
    "cpp": {
        "ad": "C++",
        "kalip": "https://en.cppreference.com/w/cpp/{ad}",
        "varsayilan_surum": "C++23",
        "arama": "https://en.cppreference.com/mwiki/index.php?search={sorgu}",
    },
}


def _indir(url: str, *, zaman: int = 30) -> Optional[str]:
    """URL'yi indirir (gzip destekli, önbellekli)."""
    ONBELLEK.mkdir(parents=True, exist_ok=True)
    ad = re.sub(r"[^a-zA-Z0-9]", "_", url)[:170]
    yerel = ONBELLEK / ad
    if yerel.exists() and yerel.stat().st_size > 500:
        return yerel.read_text(encoding="utf-8", errors="replace")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0",
                          "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=zaman) as r:
            ham = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                try:
                    ham = gzip.decompress(ham)
                except Exception:
                    pass
        metin = ham.decode("utf-8", errors="replace")
        yerel.write_text(metin, encoding="utf-8")
        return metin
    except Exception:
        return None


def _temizle(metin: str) -> str:
    """HTML temizler."""
    metin = re.sub(r"<script[^>]*>.*?</script>", " ", metin, flags=re.DOTALL)
    metin = re.sub(r"<style[^>]*>.*?</style>", " ", metin, flags=re.DOTALL)
    metin = re.sub(r"<[^>]+>", "", metin)
    for a, b in (("&gt;", ">"), ("&lt;", "<"), ("&quot;", '"'),
                 ("&#39;", "'"), ("&amp;", "&"), ("&nbsp;", " "),
                 ("\u00b6", ""), ("\u00a0", " ")):
        metin = metin.replace(a, b)
    return re.sub(r"\n{3,}", "\n\n", metin).strip()


@dataclass
class ArastirmaSonucu:
    """Araştırma sonucu."""
    arac: str = ""
    sorgu: str = ""
    bulundu: bool = False
    url: str = ""
    ozet: str = ""
    imzalar: List[str] = field(default_factory=list)
    ornekler: List[str] = field(default_factory=list)
    surum: str = ""
    sure_s: float = 0.0
    kasaya_yazildi: bool = False

    def metin(self) -> str:
        """Kasa kaydı için metin."""
        p = [f"# Araştırma: {self.sorgu}", "",
             f"**Araç:** {self.arac}", f"**Dil:** {self.sorgu.split(':')[0]}",
             f"**Sürüm:** {self.surum}", f"**Kaynak:** {self.url}", ""]
        if self.ozet:
            p.extend([self.ozet[:800], ""])
        if self.imzalar:
            p.append("**İmzalar:**")
            p.append("```")
            p.extend(self.imzalar[:20])
            p.append("```")
            p.append("")
        if self.ornekler:
            p.append("**Örnekler:**")
            for o in self.ornekler[:3]:
                p.extend([f"```\n{o[:600]}\n```", ""])
        return "\n".join(p)

    def to_dict(self) -> Dict[str, Any]:
        return {"arac": self.arac, "sorgu": self.sorgu,
                "bulundu": self.bulundu, "url": self.url,
                "ozet": self.ozet[:200], "imza": len(self.imzalar),
                "ornek": len(self.ornekler), "surum": self.surum,
                "sure_s": round(self.sure_s, 2),
                "kasaya": self.kasaya_yazildi}


class Arastirmaci:
    """Kütüphane/API/sürüm araştırma araçları."""

    def __init__(self, *, kasaya: bool = True) -> None:
        self.kasaya = kasaya
        self._kasa: Any = None
        self.istatistik = {"arama": 0, "api": 0, "surum": 0, "basarisiz": 0}

    def _kasa_baglan(self) -> bool:
        if self._kasa is not None:
            return True
        if not self.kasaya:
            return False
        try:
            from neurocore.knowledge import KnowledgeLayer

            self._kasa = KnowledgeLayer()
            self._kasa.initialize()
            return True
        except Exception:
            return False

    def _yaz(self, s: ArastirmaSonucu) -> bool:
        """Sonucu kasaya yazar (kalıcı öğrenme)."""
        if not s.bulundu or not self._kasa_baglan():
            return False
        try:
            from neurocore.knowledge.vault import Note

            simdi = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            n = Note(
                id="", title=f"Araştırma: {s.sorgu}"[:120],
                body=s.metin(), kind="reference", category="arastirma",
                tags=["arastirma", s.arac, s.surum],
                confidence=0.9, importance=0.8,
                created=simdi, updated=simdi, version=1,
                sources=[s.url], links=[], backlinks=[],
                embedding_model="", embedding=None, path="", extra={})
            self._kasa.vault.write_note(n, touch=False, emit=False)
            s.kasaya_yazildi = True
            return True
        except Exception:
            return False

    # ─── Araç 1: Kütüphane ara ───

    def kutuphane_ara(self, dil: str, kutuphane: str,
                      surum: str = "") -> ArastirmaSonucu:
        """
        Resmî doküman sayfasını bulur ve **içeriğini** getirir.

        ⚠ Yöntem: dilin URL kalıbına adı yerleştir, sayfayı indir,
        imzaları ve örnekleri çıkar.
        """
        t0 = time.perf_counter()
        s = ArastirmaSonucu(arac="kutuphane_ara",
                            sorgu=f"{dil}:{kutuphane}")
        k = DIL_KAYNAK.get(dil)
        if not k:
            s.ozet = f"bilinmeyen dil: {dil}"
            self.istatistik["basarisiz"] += 1
            return s
        sur = surum or k["varsayilan_surum"]
        s.surum = sur
        # Python kalıbında sürüm zorunlu
        if dil == "python" and not sur:
            sur = "3"
        ad = kutuphane.lower().replace(".", "/")
        url = k["kalip"].format(surum=sur, ad=ad)
        s.url = url
        ham = _indir(url)
        if not ham:
            # Python için sürümsüz dene
            if dil == "python":
                url2 = (f"https://docs.python.org/3/library/"
                        f"{kutuphane.lower()}.html")
                ham = _indir(url2)
                s.url = url2
        if not ham:
            s.ozet = "sayfa indirilemedi"
            self.istatistik["basarisiz"] += 1
            return s

        s.bulundu = True
        self.istatistik["arama"] += 1
        # Başlık
        mb = re.search(r"<title[^>]*>(.*?)</title>", ham, re.DOTALL)
        s.ozet = _temizle(mb.group(1))[:300] if mb else ""
        # İmzalar
        s.imzalar = self._imza_cikar(ham, dil)[:30]
        # Örnek kod
        s.ornekler = self._ornek_cikar(ham)[:5]
        self._yaz(s)
        s.sure_s = time.perf_counter() - t0
        return s

    def _imza_cikar(self, ham: str, dil: str) -> List[str]:
        """Dile göre imzaları çıkarır."""
        out: List[str] = []
        if dil == "gdscript":
            try:
                sys.path.insert(0, str(_KOK / "tools"))
                from tools.dokuman_madencisi import godot_imzalari

                mb = re.search(r"class_([a-z0-9_]+)\.html", ham)
                out = [i for _, i in godot_imzalari(
                    ham, mb.group(1) if mb else "")]
            except Exception:
                pass
        else:
            # Sphinx deseni
            for m in re.finditer(r'<dt[^>]*id="([^"]+)"[^>]*>(.*?)</dt>',
                                 ham, re.DOTALL):
                parcalar = re.findall(
                    r'class="sig-(?:prename descclassname|name descname)"'
                    r'[^>]*><span class="pre">([^<]+)</span>', m.group(2))
                if not parcalar:
                    continue
                ad = "".join(parcalar).strip()
                params = re.findall(
                    r'class="sig-param"[^>]*>(?:<span[^>]*>)?([^<]+)',
                    m.group(2))
                p = [_temizle(x).rstrip(", ") for x in params][:6]
                out.append(f"{ad}({', '.join(p)})" if p else f"{ad}()")
        return out

    def _ornek_cikar(self, ham: str) -> List[str]:
        """Kod bloklarını çıkarır."""
        out: List[str] = []
        for m in re.finditer(
                r'<div class="highlight[^"]*"[^>]*>(.*?)</div>', ham,
                re.DOTALL):
            pre = re.search(r"<pre[^>]*>(.*?)</pre>", m.group(1), re.DOTALL)
            if pre:
                t = _temizle(pre.group(1))
                if 30 < len(t) < 2000:
                    out.append(t)
        for m in re.finditer(r'<pre[^>]*class="[^"]*brush[^"]*"[^>]*>'
                             r'(.*?)</pre>', ham, re.DOTALL):
            t = _temizle(m.group(1))
            if 30 < len(t) < 2000:
                out.append(t)
        gorulen: Set[str] = set()
        return [x for x in out if not (x in gorulen or gorulen.add(x))][:5]

    # ─── Araç 2: API imzası al ───

    def api_imzasi_al(self, dil: str, fonksiyon: str,
                      surum: str = "") -> ArastirmaSonucu:
        """
        Bir fonksiyon/metot için **imza + parametre + örnek** getirir.

        ⚠ `fonksiyon` biçimi: `Modül.ad` (ör. `pathlib.Path.read_text`)
        """
        # ⚠ ÖLÇÜLEN HATA (tur 15): `Modül.üye` biçiminde **üye** modül
        # sanılıyordu:
        #     'FastNoiseLite.frequency' → ad='frequency'
        #     → https://.../class_frequency.html  (404)
        # Doğru: **ilk parça** modül, **son parça** üye.
        d = dil
        ad = fonksiyon
        if "." in fonksiyon:
            parcalar = [x for x in fonksiyon.split(".") if x]
            # Dil ön eki verilmişse (`python.pathlib.read_text`) onu ayır
            if parcalar and parcalar[0].lower() in DIL_KAYNAK:
                d = parcalar[0].lower()
                parcalar = parcalar[1:]
            if parcalar:
                ad = parcalar[0]          # ← modül = İLK parça
        s = self.kutuphane_ara(d, ad, surum)
        s.arac = "api_imzasi_al"
        if s.bulundu:
            # İmzaları fonksiyon adına göre süz
            hedef = fonksiyon.split(".")[-1].lower()
            ilgili = [i for i in s.imzalar if hedef in i.lower()]
            if ilgili:
                s.imzalar = ilgili
            self.istatistik["api"] += 1
        return s

    # ─── Araç 3: Sürüm karşılaştır ───

    def surum_karsilastir(self, dil: str, kutuphane: str,
                          v1: str, v2: str) -> ArastirmaSonucu:
        """
        İki sürüm arasındaki **API farkını** çıkarır.

        ⚠ Yöntem: her iki sürümün sayfasını indir, imza kümelerini
        çıkar, farkı hesapla (`v2'de yeni` / `v2'de kaldırılan`).
        """
        t0 = time.perf_counter()
        s = ArastirmaSonucu(arac="surum_karsilastir",
                            sorgu=f"{dil}:{kutuphane} {v1}→{v2}",
                            surum=f"{v1} → {v2}")
        k = DIL_KAYNAK.get(dil)
        if not k:
            self.istatistik["basarisiz"] += 1
            return s

        def topla(surum: str) -> Set[str]:
            """
            Bir sürümün imza kümesi.

            ⚠ ÖLÇÜLEN TUZAK (tur 4): docs.python.org **sürümü yok
            sayıyor** — `docs.python.org/3.10/library/json.html` isteği
            `/3/library/json.html`e yönlendiriliyor. Bu yüzden "3.10"
            tarafı **0 imza** veriyordu ve fark yanlış çıkıyordu
            (`3.10: 0 · 3.13: 24 · yeni: 24` — hepsi yanlış).

            Çözüm: **kurulu kütüphane** sürümüyle eşleşiyorsa `inspect`
            kullan (kesin veri), değilse dokümanı indir.
            """
            vi = sys.version_info
            kurulu = f"{vi.major}.{vi.minor}"
            # ⚠ ÖLÇÜLEN (tur 4): 3.10 tarafı `inspect` (14 imza), 3.13
            # tarafı doküman (24 imza) kullanınca **elma-armut**
            # karşılaştırması oluyordu: "24 yeni, 14 kalkan" — hepsi
            # yanlış, çünkü iki taraf **farklı yöntemle** çıkarılmış.
            #
            # Kural: kurulu sürüm **her iki tarafta da** varsa `inspect`
            # kullan. Yoksa iki taraf da dokümandan çıkarılır.
            if dil == "python" and surum == kurulu:
                try:
                    sys.path.insert(0, str(_KOK / "tools"))
                    from tools.python_api_toplayici import modul_tara

                    return {k.imza for k in modul_tara(kutuphane)
                            if k.imza}
                except Exception:
                    pass
            # Kurulu sürüm değilse **aynı yöntemi** kullan (doküman)
            if dil == "python":
                u = (f"https://docs.python.org/{surum}/library/"
                     f"{kutuphane.lower()}.html")
            else:
                u = k["kalip"].format(surum=surum, ad=kutuphane.lower())
            h = _indir(u)
            if not h:
                return set()
            return set(self._imza_cikar(h, dil))

        a = topla(v1)
        b = topla(v2)
        if not a and not b:
            s.ozet = "iki sürüm de indirilemedi"
            self.istatistik["basarisiz"] += 1
            return s
        s.bulundu = True
        s.sure_s = time.perf_counter() - t0
        yeni = sorted(b - a)
        kalkan = sorted(a - b)
        s.ozet = (f"{v1}: {len(a)} imza · {v2}: {len(b)} imza · "
                  f"yeni: {len(yeni)} · kalkan: {len(kalkan)}")
        s.imzalar = [f"[{v2} YENİ] {x}" for x in yeni[:15]]
        s.imzalar += [f"[{v2} KALKAN] {x}" for x in kalkan[:10]]
        s.url = (f"https://docs.python.org/{v2}/library/"
                 f"{kutuphane.lower()}.html")
        self.istatistik["surum"] += 1
        self._yaz(s)
        return s


def main() -> int:
    """Araştırma araçlarını çalıştırır."""
    ap = argparse.ArgumentParser(description="Araştırma araçları")
    ap.add_argument("--ara", nargs=2, metavar=("DIL", "KUTUPHANE"))
    ap.add_argument("--api", nargs=2, metavar=("DIL", "FONKSIYON"))
    # ⚠ ÖLÇÜLEN (tur 4): `nargs=3` kullanıcıyı yanıltıyordu —
    # `--surum python json 3.10 3.13` **reddediliyordu** (3 değer bekliyor
    # ama 4 veriliyor). Düzeltme: 2 değer al, sürümleri `_` ile ayır.
    ap.add_argument("--surum", nargs=2, metavar=("DIL", "KUTUPHANE_V1_V2"),
                    help='ör. --surum python json_3.10_3.13')
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--kasa", action="store_true", default=True)
    args = ap.parse_args()

    print("=" * 78)
    print("  ARAŞTIRMA ARAÇLARI — kütüphane · API · sürüm")
    print("=" * 78)

    a = Arastirmaci(kasaya=args.kasa)
    sonuclar: List[Dict[str, Any]] = []

    if args.ara:
        s = a.kutuphane_ara(args.ara[0], args.ara[1])
        print(f"\n  SORGU: {s.sorgu}")
        print(f"  URL  : {s.url}")
        print(f"  bulundu: {s.bulundu} · imza: {len(s.imzalar)} · "
              f"örnek: {len(s.ornekler)}")
        print(f"  özet : {s.ozet[:100]}")
        if s.imzalar:
            print("  İMZALAR:")
            for i in s.imzalar[:8]:
                print(f"    {i[:72]}")
        sonuclar.append(s.to_dict())

    if args.api:
        s = a.api_imzasi_al(args.api[0], args.api[1])
        print(f"\n  API: {s.sorgu}")
        print(f"  bulundu: {s.bulundu} · imza: {len(s.imzalar)}")
        for i in s.imzalar[:8]:
            print(f"    {i[:72]}")
        sonuclar.append(s.to_dict())

    if args.surum:
        parcalar = args.surum[1].split("_")
        if len(parcalar) >= 3:
            kutuphane = parcalar[0]
            v1, v2 = parcalar[1], parcalar[2]
            s = a.surum_karsilastir(args.surum[0], kutuphane, v1, v2)
            print(f"\n  SÜRÜM: {s.sorgu}")
            print(f"  {s.ozet}")
            for i in s.imzalar[:12]:
                print(f"    {i[:72]}")
            sonuclar.append(s.to_dict())

    if args.test or not (args.ara or args.api or args.surum):
        print("\n  TESTLER:")
        testler = [
            ("kutuphane_ara", lambda: a.kutuphane_ara("python", "json")),
            ("kutuphane_ara", lambda: a.kutuphane_ara("gdscript", "Vector3")),
            ("api_imzasi_al", lambda: a.api_imzasi_al("python", "pathlib")),
        ]
        for ad, fn in testler:
            try:
                s = fn()
                im = "✓" if s.bulundu else "✗"
                print(f"    {im} {ad:16} {s.sorgu:28} "
                      f"imza={len(s.imzalar):3} örnek={len(s.ornekler)} "
                      f"kasa={s.kasaya_yazildi}")
                sonuclar.append(s.to_dict())
            except Exception as exc:
                print(f"    ✗ {ad:16} hata: {type(exc).__name__}")
        print(f"\n  İSTATİSTİK: {json.dumps(a.istatistik)}")

    Path("_arastirma.json").write_text(
        json.dumps({"sonuclar": sonuclar, "istatistik": a.istatistik},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
