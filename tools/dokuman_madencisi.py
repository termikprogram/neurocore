"""
Çok Dilli Doküman Madencisi — Sürüm Etiketli Kod Verisi
=========================================================
⚠ Kullanıcı hedefi (A)
----------------------
> *"her dilin kütüphanesi ve sürümüne kadar geniş, **çok dilli ve çok
> sürümlü** kod verisi topla"*

**Ölçülen erişim (tur 2):**

| Kaynak | Boyut | Durum |
|---|---|---|
| Python 3.10 | 16,7 KB | ✅ HTTP 200 |
| Python 3.13 | 19,7 KB | ✅ HTTP 200 |
| Godot 4.x | 1,3 MB | ✅ HTTP 200 |
| MDN JavaScript | 203,9 KB | ✅ HTTP 200 |
| Rust std | 58,2 KB | ✅ HTTP 200 |

Yöntem
------
    1. **Doküman kaynağı tanımı** — kök URL + dil + sürüm
    2. **Bağlantı keşfi** — API sayfalarını bul (aynı alan adı)
    3. **İndir** (önbellekli, gzip destekli)
    4. **Ayrıştır** — kod blokları + imzalar + açıklama
    5. **Sürüm etiketiyle** yaz

Desteklenen formatlar
---------------------
    * **Sphinx** (Python, Godot, Rust) — `<dl class="py function">`
    * **MDN** — `<pre class="brush: js">` + başlık yapısı

Kullanım
--------
    python tools/dokuman_madencisi.py --liste
    python tools/dokuman_madencisi.py --kesif python
    python tools/dokuman_madencisi.py --topla python --maks 400
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

ONBELLEK = Path(".neurocore/dokuman_onbellek2")


@dataclass
class Kaynak:
    """Bir doküman kaynağı."""
    anahtar: str
    ad: str
    dil: str
    surum: str
    kok: str
    #: Yalnız bu önekle başlayan bağlantılar izlenir
    onemli: str = ""
    #: Ayrıştırıcı türü
    bicim: str = "sphinx"
    #: Doğrudan URL kalıbı — `{m}` modül adıyla değişir.
    #:
    #: ⚠ ÖLÇÜLEN (tur 3): Python doküman dizini (`library/index.html`)
    #: **JavaScript ile** yükleniyor — statik HTML'de bağlantı **yok**
    #: (313 href'in yalnız 4'ü "library" içeriyor). Bağlantı keşfi
    #: bu yüzden **0 sayfa** buldu.
    #: Çözüm: modül adından **doğrudan URL** üret.
    url_kalip: str = ""


#: Doküman kaynakları — çok dilli, çok sürümlü
KAYNAKLAR: Sequence[Kaynak] = (
    Kaynak("python310", "Python 3.10", "python", "3.10",
           "https://docs.python.org/3.10/", "3.10/library/"),
    Kaynak("python313", "Python 3.13", "python", "3.13",
           "https://docs.python.org/3.13/", "3.13/library/",
           url_kalip="https://docs.python.org/3.13/library/{m}.html"),
    # ⚠ tur 7: Python **eğitim** ve **referans** bölümleri de eklendi
    # (dil öğretisi + sözdizimi kuralları).
    Kaynak("python313_egitim", "Python 3.13 Eğitim", "python", "3.13",
           "https://docs.python.org/3.13/tutorial/", "tutorial/"),
    Kaynak("godot", "Godot 4.x", "gdscript", "4.x",
           "https://docs.godotengine.org/en/stable/", "classes/"),
    Kaynak("rust", "Rust std", "rust", "stable",
           "https://doc.rust-lang.org/std/", "std/"),
    # ⚠ ÖLÇÜLEN (tur 3): `onemli="Reference/"` filtresi CSS ve HTML
    # sayfalarını da getiriyordu. Filtre **JavaScript/Reference/** olarak
    # daraltıldı — yalnız JS API'si.
    Kaynak("js_mdn", "JavaScript MDN", "javascript", "ES2024",
           "https://developer.mozilla.org/en-US/docs/Web/JavaScript/",
           "JavaScript/Reference/", "mdn"),
    # ⚠ ÖLÇÜLEN (tur 4): `onemli="cpp/"` filtresi **0 sayfa** buldu.
    # Sebep: cppreference bağlantıları `/cpp/...` (başta **eğik çizgi**).
    # Filtre düzeltildi.
    Kaynak("cpp", "C++ cppreference", "cpp", "C++23",
           "https://en.cppreference.com/w/cpp", "/cpp/"),
)

#: Anahtar → kaynak
KAYNAK_HARITA: Dict[str, Kaynak] = {k.anahtar: k for k in KAYNAKLAR}


def _indir(url: str, *, zaman: int = 30) -> Optional[str]:
    """URL'yi indirir (gzip destekli, önbellekli)."""
    ONBELLEK.mkdir(parents=True, exist_ok=True)
    ad = re.sub(r"[^a-zA-Z0-9]", "_", url)[:180]
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
    """HTML etiketlerini temizler."""
    metin = re.sub(r"<script[^>]*>.*?</script>", " ", metin, flags=re.DOTALL)
    metin = re.sub(r"<style[^>]*>.*?</style>", " ", metin, flags=re.DOTALL)
    metin = re.sub(r"<[^>]+>", "", metin)
    for a, b in (("&gt;", ">"), ("&lt;", "<"), ("&quot;", '"'),
                 ("&#39;", "'"), ("&amp;", "&"), ("&nbsp;", " "),
                 ("\u00b6", ""), ("\u00a0", " ")):
        metin = metin.replace(a, b)
    return re.sub(r"\n{3,}", "\n\n", metin).strip()


def baglanti_bul(ham: str, kok: str, onemli: str) -> List[str]:
    """Sayfadaki iç bağlantıları bulur."""
    bulunan: Set[str] = set()
    for m in re.finditer(r'href="([^"#?]+)"', ham):
        h = m.group(1)
        if h.startswith(("http://", "https://", "mailto:", "javascript:")):
            continue
        # ⚠ ÖLÇÜLEN (tur 3): MDN bağlantıları **uzantısız**
        # (`/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array`).
        # İlk sürüm yalnız `.html` ve `/` kabul ediyordu → **0 sayfa**
        # bulundu. Artık uzantısız yollar da kabul edilir.
        if h.endswith((".css", ".js", ".png", ".jpg", ".svg", ".ico",
                       ".json", ".xml", ".txt", ".pdf")):
            continue
        if "?" in h or "=" in h:
            continue
        tam = urllib.parse.urljoin(kok, h)
        if urllib.parse.urlparse(tam).netloc != urllib.parse.urlparse(
                kok).netloc:
            continue
        if onemli and onemli not in tam:
            continue
        bulunan.add(tam)
    return sorted(bulunan)


def kod_bloklari(ham: str, bicim: str = "sphinx") -> List[str]:
    """Sayfadaki kod bloklarını çıkarır."""
    bloklar: List[str] = []
    # Sphinx: <div class="highlight-*"> ... <pre> ... </pre>
    for m in re.finditer(
            r'<div class="highlight[^"]*"[^>]*>(.*?)</div>', ham,
            re.DOTALL):
        pre = re.search(r"<pre[^>]*>(.*?)</pre>", m.group(1), re.DOTALL)
        if pre:
            bloklar.append(_temizle(pre.group(1)))
    # MDN: <pre class="brush: ...">
    for m in re.finditer(r'<pre[^>]*class="[^"]*brush[^"]*"[^>]*>(.*?)</pre>',
                         ham, re.DOTALL):
        bloklar.append(_temizle(m.group(1)))
    # Düz <pre>
    for m in re.finditer(r"<pre[^>]*>(.*?)</pre>", ham, re.DOTALL):
        bloklar.append(_temizle(m.group(1)))

    # Temizle: kısa, tekrarlı ve kod gibi olmayanları at
    gorulen: Set[str] = set()
    sonuc: List[str] = []
    for b in bloklar:
        t = b.strip()
        if len(t) < 25 or len(t) > 3000:
            continue
        if t in gorulen:
            continue
        if not any(k in t for k in ("(", "=", "import", "func ", "fn ",
                                    "def ", "let ", "const ", "var ",
                                    "class ", "return", "#include")):
            continue
        gorulen.add(t)
        sonuc.append(t)
    return sonuc


def godot_uyeleri(ham: str) -> Tuple[List[str], List[str], List[str]]:
    """
    **Godot** sınıf sayfasından üyeleri çıkarır.

    ⚠ ÖLÇÜLEN (tur 3): Godot **Sphinx kullanmıyor** —
        `<dt` sayısı: **0** · `sig-name`: **0** · `py method`: **0**
    Bu yüzden genel Sphinx ayrıştırıcısı **0 imza** buldu (yalnız 664
    kod örneği toplandı).

    Gerçek yapı (ölçüldü):
        `id="class-vector3-method-normalized"` — 48 metot
        `id="class-vector3-property-x"`       — özellikler
        `id="class-vector3-constant-zero"`    — sabitler

    Returns:
        (metotlar, özellikler, sabitler) — ad listeleri
    """
    metotlar = list(dict.fromkeys(re.findall(
        r'id="class-[a-z0-9_]+-method-([a-z0-9_]+)"', ham)))
    ozellikler = list(dict.fromkeys(re.findall(
        r'id="class-[a-z0-9_]+-property-([a-z0-9_]+)"', ham)))
    sabitler = list(dict.fromkeys(re.findall(
        r'id="class-[a-z0-9_]+-constant-([a-z0-9_]+)"', ham)))
    return metotlar, ozellikler, sabitler


def godot_imzalari(ham: str, sinif: str) -> List[Tuple[str, str]]:
    """
    Godot metot imzalarını çıkarır.

    ⚠ ÖLÇÜLEN GERÇEK YAPI (tur 3, `<tr>` tablo varsayımı **yanlıştı**):
        `<p class="classref-method" id="class-vector3-method-normalized">`
        `<a ...><span>Vector3</span></a>`
        ` <strong>normalized</strong>() <abbr ...>const</abbr>`

    Yani: **dönüş tipi** bağlantı → `strong` metot adı → `()` → `const`.
    """
    sonuc: List[Tuple[str, str]] = []
    desen = re.compile(
        r'<p class="classref-method" id="class-[a-z0-9_]+-method-'
        r'([a-z0-9_]+)">(.*?)</p>', re.DOTALL)
    for m in desen.finditer(ham):
        ad = m.group(1)
        govde = m.group(2)
        # Dönüş tipi: ilk bağlantının metni
        donus = ""
        db = re.search(r'<a[^>]*>(.*?)</a>', govde, re.DOTALL)
        if db:
            donus = _temizle(db.group(1)).strip()[:36]
        # Parametreler: `()` içi
        pm = re.search(r"\(([^)]*)\)", govde)
        params = _temizle(pm.group(1)).strip() if pm else ""
        params = re.sub(r"\s+", " ", params)
        imza = (f"{ad}({params}) -> {donus}" if donus
                else f"{ad}({params})")
        sonuc.append((f"{sinif}.{ad}", imza))
    # Özellikler
    for m in re.finditer(
            r'<p class="classref-property" id="class-[a-z0-9_]+-property-'
            r'([a-z0-9_]+)">(.*?)</p>', ham, re.DOTALL):
        ad = m.group(1)
        govde = m.group(2)
        db = re.search(r'<a[^>]*>(.*?)</a>', govde, re.DOTALL)
        tip = _temizle(db.group(1)).strip()[:36] if db else ""
        sonuc.append((f"{sinif}.{ad}", f"{ad} : {tip}"))
    # Sabitler
    for m in re.finditer(
            r'<p class="classref-constant" id="class-[a-z0-9_]+-constant-'
            r'([a-z0-9_]+)">(.*?)</p>', ham, re.DOTALL):
        ad = m.group(1)
        govde = _temizle(m.group(2))[:80]
        sonuc.append((f"{sinif}.{ad}", govde))
    return list(dict.fromkeys(sonuc))[:120]


def imzalar(ham: str, bicim: str = "sphinx") -> List[Tuple[str, str]]:
    """
    Sphinx imzalarını çıkarır — (ad, imza).

    ⚠ Ölçülen format (Blender dokümanıyla aynı):
        `<span class="sig-prename descclassname">modül.</span>`
        `<span class="sig-name descname">fonksiyon</span>`
    """
    sonuc: List[Tuple[str, str]] = []
    for m in re.finditer(r'<dt[^>]*id="([^"]+)"[^>]*>(.*?)</dt>', ham,
                         re.DOTALL):
        kimlik = m.group(1)
        govde = m.group(2)
        parcalar = re.findall(
            r'class="sig-(?:prename descclassname|name descname)"[^>]*>'
            r'<span class="pre">([^<]+)</span>', govde)
        if not parcalar:
            continue
        tam = "".join(parcalar).strip()
        params = re.findall(
            r'class="sig-param"[^>]*>(?:<span[^>]*>)?([^<]+)', govde)
        p_temiz = [_temizle(p).rstrip(", ") for p in params][:8]
        imza = f"{tam}({', '.join(p_temiz)})" if p_temiz else f"{tam}()"
        sonuc.append((tam, imza))
    return sonuc


def baslik_al(ham: str) -> str:
    """Sayfa başlığı."""
    m = re.search(r"<title[^>]*>(.*?)</title>", ham, re.DOTALL)
    if m:
        return _temizle(m.group(1))[:120]
    m = re.search(r"<h1[^>]*>(.*?)</h1>", ham, re.DOTALL)
    return _temizle(m.group(1))[:120] if m else ""


def sayfa_isle(ham: str, url: str, kaynak: Kaynak) -> List[Dict[str, Any]]:
    """Bir sayfadan kayıtlar üretir."""
    baslik = baslik_al(ham) or url.rsplit("/", 2)[-2]
    kayitlar: List[Dict[str, Any]] = []

    # 1) İmzalar
    #
    # ⚠ ÖLÇÜLEN (tur 3): Godot **Sphinx kullanmıyor** (`<dt` = 0).
    # Genel ayrıştırıcı 0 imza buldu. Dile özel ayrıştırıcı gerekir.
    if kaynak.dil == "gdscript":
        sinif = ""
        mb = re.search(r"class_([a-z0-9_]+)\.html", url)
        if mb:
            sinif = mb.group(1)
        imza_listesi = godot_imzalari(ham, sinif)
    else:
        imza_listesi = imzalar(ham, kaynak.bicim)
    for ad, imza in imza_listesi[:40]:
        metin = (f"# {kaynak.dil} · {ad}\n\n"
                 f"**Dil:** {kaynak.dil}\n"
                 f"**Sürüm:** {kaynak.surum}\n"
                 f"**İmza:** `{imza}`\n"
                 f"**Kaynak:** {baslik}\n\n"
                 f"```\n{imza}\n```\n")
        kayitlar.append({"metin": metin, "kategori": f"{kaynak.dil}_api",
                         "dil": kaynak.dil, "surum": kaynak.surum,
                         "tur": "imza", "url": url})

    # 2) Kod blokları
    for blok in kod_bloklari(ham, kaynak.bicim)[:20]:
        metin = (f"# {kaynak.dil} örnek · {baslik}\n\n"
                 f"**Dil:** {kaynak.dil}\n"
                 f"**Sürüm:** {kaynak.surum}\n\n"
                 f"```{kaynak.dil}\n{blok}\n```\n")
        kayitlar.append({"metin": metin, "kategori": f"{kaynak.dil}_ornek",
                         "dil": kaynak.dil, "surum": kaynak.surum,
                         "tur": "ornek", "url": url})
    return kayitlar


def modul_listesi() -> List[str]:
    """
    Standart kütüphane modül adları.

    ⚠ `sys.stdlib_module_names` Python 3.10+ ile gelir — bu ölçüm
    ortamında **mevcut**.
    """
    try:
        return sorted(m for m in sys.stdlib_module_names
                      if not m.startswith("_"))
    except Exception:
        return ["os", "sys", "json", "re", "math", "collections",
                "itertools", "functools", "pathlib", "typing",
                "asyncio", "datetime", "logging", "unittest"]


def kesif(kaynak: Kaynak, *, maks: int = 200,
          seviye: int = 2) -> List[str]:
    """
    Kaynağın API sayfalarını keşfeder.

    ⚠ Üç yöntem:
        1. **Doğrudan kalıp** (`url_kalip`) — modül adından URL üret.
           Python dokümanı için **şart** (JS ile yüklenen dizin).
        2. **Tek seviye** bağlantı keşfi — ana sayfadan iç bağlantılar.
        3. **Çok seviyeli** keşif (`seviye=2`) — bulunan sayfalardan da
           bağlantı izle.

    ⚠ ÖLÇÜLEN (tur 5): tek seviye keşif **yetersiz** — cppreference
    yalnız **172** sayfa verdi (ana sayfa yalnız kategori bağlantıları
    taşıyor). Çok seviyeli keşif bunu katlar.
    """
    if kaynak.url_kalip:
        moduller = modul_listesi()
        return [kaynak.url_kalip.format(m=m) for m in moduller][:maks]

    bulunan: Set[str] = set()
    islenecek: List[str] = [kaynak.kok]
    gorulen_sayfa: Set[str] = set()

    for _ in range(max(1, seviye)):
        yeni: List[str] = []
        for u in islenecek:
            if u in gorulen_sayfa or len(bulunan) >= maks:
                continue
            gorulen_sayfa.add(u)
            ham = _indir(u)
            if not ham:
                continue
            for b in baglanti_bul(ham, u, kaynak.onemli):
                if b in bulunan:
                    continue
                bulunan.add(b)
                yeni.append(b)
                if len(bulunan) >= maks:
                    break
        if not yeni or len(bulunan) >= maks:
            break
        # Sonraki seviyede ilk N sayfayı tara (bütçe sınırı)
        islenecek = yeni[:60]
    return sorted(bulunan)[:maks]


def topla(kaynak: Kaynak, *, maks_sayfa: int = 100,
          hedef: Path) -> Dict[str, Any]:
    """Bir kaynaktan veri toplar."""
    sayfalar = kesif(kaynak, maks=maks_sayfa)
    print(f"  {kaynak.ad}: {len(sayfalar)} sayfa bulundu")
    hedef.parent.mkdir(parents=True, exist_ok=True)
    sayi = 0
    karakter = 0
    islenen = 0
    with hedef.open("w", encoding="utf-8") as f:
        for i, url in enumerate(sayfalar, 1):
            ham = _indir(url)
            if not ham:
                continue
            kayitlar = sayfa_isle(ham, url, kaynak)
            if not kayitlar:
                continue
            islenen += 1
            for k in kayitlar:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")
                sayi += 1
                karakter += len(k["metin"])
            if i % 25 == 0:
                print(f"    {i}/{len(sayfalar)} sayfa · {sayi} kayıt · "
                      f"~{karakter // 3500:,}k token", flush=True)
    return {"kaynak": kaynak.anahtar, "ad": kaynak.ad,
            "sayfa": islenen, "kayit": sayi, "karakter": karakter,
            "token_tahmini": karakter // 3.5.__int__() if False else
            int(karakter / 3.5), "dosya": str(hedef)}


def main() -> int:
    """Doküman madenciliğini çalıştırır."""
    ap = argparse.ArgumentParser(description="Çok dilli doküman madencisi")
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--kesif", default="", help="Kaynak anahtarı")
    ap.add_argument("--topla", default="", help="Kaynak anahtarı")
    ap.add_argument("--maks", type=int, default=100, help="Azami sayfa")
    args = ap.parse_args()

    if args.liste:
        for k in KAYNAKLAR:
            print(f"  {k.anahtar:12} {k.ad:22} {k.dil:12} "
                  f"sürüm={k.surum:8} {k.kok}")
        print(f"\n  TOPLAM: {len(KAYNAKLAR)} kaynak")
        return 0

    print("=" * 78)
    print("  ÇOK DİLLİ DOKÜMAN MADENCİSİ — sürüm etiketli")
    print("=" * 78)

    sonuc: Dict[str, Any] = {}
    if args.kesif:
        k = KAYNAK_HARITA.get(args.kesif)
        if not k:
            print(f"  bilinmeyen kaynak: {args.kesif}")
            return 1
        print(f"\n  {k.ad} keşfi...")
        sayfalar = kesif(k, maks=args.maks)
        print(f"  {len(sayfalar)} sayfa:")
        for u in sayfalar[:20]:
            print(f"    {u}")
        sonuc["kesif"] = {"kaynak": k.anahtar, "sayfa": len(sayfalar)}

    if args.topla:
        k = KAYNAK_HARITA.get(args.topla)
        if not k:
            print(f"  bilinmeyen kaynak: {args.topla}")
            return 1
        print()
        r = topla(k, maks_sayfa=args.maks,
                  hedef=Path(f".neurocore/dokuman_{k.anahtar}.jsonl"))
        print(f"\n  TOPLANDI: {r['sayfa']} sayfa · {r['kayit']:,} kayıt · "
              f"~{r['token_tahmini']:,} token")
        print(f"  dosya: {r['dosya']}")
        sonuc["toplama"] = r

    Path("_dokuman_madencisi.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
