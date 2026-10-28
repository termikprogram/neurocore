"""
NeuroCore Ağ ve Araştırma Araçları
===================================
İnternetten **okunabilir metin** çekme ve araştırma yardımcıları.

Neden ayrı modül? Ajan katmanı yerel dosyalarla çalışır; kendi kendini
geliştirmek için **dış bilgi** gerekir. Bu modül, öğretmen modelin
(gemini/deepseek/groq) gördüğü ham HTML'i değil, **temizlenmiş metni**
okumasını sağlar — böylece öğrenilen bilgi gerçek kaynağa dayanır.

Güvenlik:
    * Yalnızca `http`/`https` şemaları kabul edilir.
    * Yerel/özel ağ adresleri (`localhost`, `127.0.0.1`, `192.168.*`,
      `10.*`, `*.local`) **reddedilir** — SSRF koruması.
    * Yanıt boyutu sınırlanır (varsayılan 512 KB).
    * Özel IP aralıklarına yönlendiren yönlendirmeler izlenmez.
"""

from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.web")

#: HTTP isteklerinde kullanılan kimlik (bazı siteler boş kimliği reddeder)
USER_AGENT = "Mozilla/5.0 (compatible; NeuroCore/0.1; +local research agent)"

#: İndirilecek en büyük yanıt (bayt)
MAX_BYTES = 512 * 1024

#: Varsayılan zaman aşımı (saniye)
DEFAULT_TIMEOUT = 25.0

#: Özel/yerel ağ adresleri (SSRF koruması)
_BLOCKED_HOSTS = ("localhost", "127.", "0.0.0.0", "::1", "[::1]")


@dataclass
class WebPage:
    """İndirilen ve temizlenmiş bir sayfa."""
    url: str = ""
    final_url: str = ""
    title: str = ""
    text: str = ""
    links: List[str] = field(default_factory=list)
    status: int = 0
    content_type: str = ""
    truncated: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        """Başarılı mı?"""
        return self.status == 200 and bool(self.text) and not self.error

    def to_dict(self, max_chars: int = 20000) -> Dict[str, Any]:
        """JSON-serileştirilebilir hâl."""
        text = self.text
        truncated = self.truncated
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True
        return {
            "ok": self.ok, "url": self.url, "final_url": self.final_url,
            "title": self.title, "text": text, "links": self.links[:40],
            "status": self.status, "content_type": self.content_type,
            "truncated": truncated, "error": self.error,
        }


def is_public_url(url: str) -> Tuple[bool, str]:
    """
    Adresin **genel** internete ait olduğunu doğrular.

    Neden? Ajanın `http://localhost:8791/api/...` gibi iç adreslere istek
    atması (SSRF) engellenmelidir; ayrıca yerel ağ taraması yapılmamalıdır.

    Returns:
        (izinli mi, gerekçe)
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False, "adres çözümlenemedi"
    if parsed.scheme not in ("http", "https"):
        return False, f"yalnızca http/https desteklenir (verilen: {parsed.scheme or 'yok'})"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False, "adres boş"
    if any(host.startswith(item) for item in _BLOCKED_HOSTS):
        return False, "yerel adresler reddedilir"
    if host.endswith(".local") or host.endswith(".internal"):
        return False, "yerel alan adları reddedilir"
    # Sayısal IP ise özel aralıkta mı?
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True, ""      # alan adı — DNS çözümlemesi ayrıca yapılır
    if address.is_private or address.is_loopback or address.is_link_local \
            or address.is_reserved:
        return False, f"özel ağ adresi reddedilir ({host})"
    return True, ""


def _resolve_is_public(host: str) -> Tuple[bool, str]:
    """Alan adının çözümlendiği IP'nin genel olduğunu doğrular."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return False, f"alan adı çözümlenemedi: {exc}"
    for info in infos:
        raw = info[4][0]
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if address.is_private or address.is_loopback or address.is_link_local:
            return False, f"alan adı özel adrese çözümleniyor ({raw})"
    return True, ""


# ─── HTML Temizleme ─────────────────────────────────────────────────────────

#: İçeriği olmayan bloklar (tamamen atılır)
_DROP_BLOCKS = re.compile(
    r"<(script|style|noscript|svg|iframe|head|nav|footer|form)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)

#: Ana içerik adayları (varsa yalnızca bunlar kullanılır)
_MAIN_BLOCKS = re.compile(
    r"<(article|main)\b[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL
)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_LINK = re.compile(r"""<a\b[^>]*href=["']([^"'#]+)["']""", re.IGNORECASE)
_TITLE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def html_to_text(markup: str) -> str:
    """
    HTML'i okunabilir **düz metne** çevirir.

    Öğretmen modele ham HTML vermek hem pahalı hem gürültülüdür; bu fonksiyon
    menü/script/stil bloklarını atar, ana içeriği (varsa `article`/`main`)
    önceler ve boşlukları düzenler.
    """
    if not markup:
        return ""
    # Ana içerik varsa onu kullan (menü ve kenar çubuklarından kurtulur)
    main = _MAIN_BLOCKS.search(markup)
    body = main.group(2) if main else markup

    body = _DROP_BLOCKS.sub(" ", body)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"</(p|div|li|h[1-6]|tr|section)>", "\n", body, flags=re.IGNORECASE)
    body = _TAG.sub(" ", body)
    body = html.unescape(body)
    body = _WS.sub(" ", body)
    lines = [line.strip() for line in body.splitlines()]
    text = "\n".join(line for line in lines if line)
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


def extract_title(markup: str) -> str:
    """Sayfa başlığını çıkarır."""
    match = _TITLE.search(markup or "")
    if not match:
        return ""
    return html.unescape(_TAG.sub("", match.group(1))).strip()[:200]


def extract_links(markup: str, base_url: str = "") -> List[str]:
    """Sayfadaki bağlantıları **mutlak** adreslere çevirir."""
    links: List[str] = []
    for raw in _LINK.findall(markup or ""):
        absolute = urllib.parse.urljoin(base_url, raw.strip()) if base_url else raw.strip()
        if absolute.startswith(("http://", "https://")) and absolute not in links:
            links.append(absolute)
    return links


# ─── İndirme ────────────────────────────────────────────────────────────────

def fetch(url: str, *, timeout: float = DEFAULT_TIMEOUT,
          max_bytes: int = MAX_BYTES, allow_private: bool = False) -> WebPage:
    """
    Bir web sayfasını indirir ve **düz metne** çevirir.

    Args:
        url: Adres (http/https)
        timeout: Zaman aşımı (saniye)
        max_bytes: En büyük yanıt boyutu
        allow_private: Yerel adreslere izin ver (test için; varsayılan kapalı)

    Returns:
        `WebPage`
    """
    page = WebPage(url=url)
    if not allow_private:
        allowed, reason = is_public_url(url)
        if not allowed:
            page.error = reason
            return page
        host = urllib.parse.urlparse(url).hostname or ""
        try:
            ipaddress.ip_address(host)
        except ValueError:
            public, why = _resolve_is_public(host)
            if not public:
                page.error = why
                return page

    request = urllib.request.Request(url, method="GET")
    request.add_header("User-Agent", USER_AGENT)
    request.add_header("Accept", "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5")
    request.add_header("Accept-Language", "tr,en;q=0.8")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            page.status = getattr(response, "status", 200)
            page.final_url = response.geturl()
            page.content_type = response.headers.get("Content-Type", "")
            raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raw = raw[:max_bytes]
                page.truncated = True
    except urllib.error.HTTPError as exc:
        page.status = exc.code
        page.error = f"HTTP {exc.code}: {exc.reason}"
        return page
    except urllib.error.URLError as exc:
        page.error = f"bağlantı hatası: {exc.reason}"
        return page
    except (TimeoutError, socket.timeout):
        page.error = f"zaman aşımı ({timeout:.0f}s)"
        return page
    except Exception as exc:
        page.error = f"{type(exc).__name__}: {exc}"
        return page

    charset = "utf-8"
    match = re.search(r"charset=([\w-]+)", page.content_type, re.IGNORECASE)
    if match:
        charset = match.group(1)
    markup = raw.decode(charset, "replace")

    if "html" in page.content_type.lower() or "<html" in markup[:2000].lower():
        page.title = extract_title(markup)
        page.text = html_to_text(markup)
        page.links = extract_links(markup, page.final_url or url)
    else:
        # Düz metin / JSON: olduğu gibi al
        page.text = markup.strip()
    return page


#: DDG Lite sonuç bloğu: bağlantı + başlık
_DDG_LINK = re.compile(
    r"""<a[^>]+href=["']([^"']+)["'][^>]*class=['"]result-link['"][^>]*>(.*?)</a>""",
    re.IGNORECASE | re.DOTALL,
)

#: DDG Lite özet (snippet) hücresi
_DDG_SNIPPET = re.compile(
    r"""<td[^>]*class=['"]result-snippet['"][^>]*>(.*?)</td>""",
    re.IGNORECASE | re.DOTALL,
)

#: DDG yönlendirme bağlantısındaki gerçek hedef
_UDDG = re.compile(r"[?&]uddg=([^&]+)")


def _decode_ddg_url(href: str) -> str:
    """
    DuckDuckGo yönlendirme adresinden **gerçek** hedefi çıkarır.

    DDG bağlantıları `//duckduckgo.com/l/?uddg=<yüzde-kodlu adres>` biçiminde
    gelir ve `//` ile başlar (şemasız). Bu fonksiyon yüzde kodunu çözer ve
    `https:` şemasını ekler.
    """
    if not href:
        return ""
    href = html.unescape(href.strip())
    if href.startswith("//"):
        href = "https:" + href
    match = _UDDG.search(href)
    if match:
        target = urllib.parse.unquote(match.group(1))
        if target.startswith(("http://", "https://")):
            return target
    return href if href.startswith(("http://", "https://")) else ""


#: Son arama zamanı (nazik aralık için)
_LAST_SEARCH = {"at": 0.0}

#: İki arama arası en az süre (saniye). Arama motorları hızlı ardışık
#: isteklerde bot korumasını devreye sokar; ölçüldü: DuckDuckGo Lite ilk
#: sorguya gerçek sonuç, hemen ardından gelen sorgulara boş yanıt veriyor.
MIN_SEARCH_INTERVAL = 3.0


def _search_github(query: str, *, limit: int,
                   timeout: float) -> List[Dict[str, str]]:
    """
    GitHub API ile **resmî kaynak/kod** arar (yüksek güvenilirlik).

    Neden? Arama motorları bot koruması uygular ve kırılgandır (ölçüldü:
    DuckDuckGo kalıcı engelledi, Bing sorguyu yok sayıp hep aynı sonucu
    döndürdü). GitHub API'si ise:

        * Kimlik doğrulaması istemez (anonim kota: 10 istek/dakika)
        * Bot koruması yoktur, JSON döner
        * **Teknik sorular için en alakalı sonucu** verir: resmî depo,
          kaynak dosya, doküman

    Sorgudaki **ayırt edici** terimler (uzun ve özel olanlar) aranır;
    "nedir", "nasıl" gibi kelimeler gürültü yaptığı için atılır.
    """
    terimler = [t for t in re.findall(r"[0-9A-Za-z_çğıöşüÇĞİÖŞÜ]{4,}",
                                      query.lower())
                if t not in _SEARCH_STOP]
    if not terimler:
        return []
    # En uzun (en özel) en fazla 3 terim → sorgu daralsın, sonuç isabetli olsun
    terimler = sorted(set(terimler), key=len, reverse=True)[:3]
    sorgu = " ".join(terimler)

    sonuclar: List[Dict[str, str]] = []
    # Yalnızca **depo** araması: kod araması kimlik doğrulaması ister
    # (ölçüldü: `HTTP 401 Requires authentication`) ve anonim kullanımda
    # her zaman boş döner.
    for tur, yol in (
        ("depo", "https://api.github.com/search/repositories?q="
                 f"{urllib.parse.quote_plus(sorgu)}&per_page={limit}"),
    ):
        if len(sonuclar) >= limit:
            break
        try:
            ham = _fetch_markup(yol, timeout=timeout)
            veri = json.loads(ham) if ham else {}
        except Exception as exc:
            logger.debug(f"GitHub {tur} araması başarısız: {exc}")
            continue
        for kayit in (veri.get("items") or []):
            if tur == "depo":
                ad = str(kayit.get("full_name") or "")
                aciklama = str(kayit.get("description") or "")
            else:
                ad = str(kayit.get("name") or "")
                depo = (kayit.get("repository") or {}).get("full_name") or ""
                aciklama = f"{depo} — {kayit.get('path') or ''}"
            url = str(kayit.get("html_url") or "")
            if not url:
                continue
            sonuclar.append({"title": ad[:120], "url": url,
                             "snippet": aciklama[:200]})
            if len(sonuclar) >= limit:
                break
    return sonuclar


#: Arama sorgusunda yok sayılan kısa/genel terimler
_SEARCH_STOP = frozenset({
    "nedir", "nasil", "nasıl", "nicin", "niçin", "hangi", "ile", "icin",
    "için", "bir", "bu", "ve", "veya", "ama", "daha", "cok", "çok",
    "the", "and", "for", "with", "how", "what", "why", "does", "use",
    "using", "kullan", "kullanim", "kullanım", "arasindaki", "arasındaki",
    "farki", "farkı", "acikla", "açıkla", "ornek", "örnek",
})


def _resolve_bing_url(url: str) -> str:
    """
    Bing'in yönlendirme adresini gerçek hedefe çevirir.

    ⚠ Ölçüldü: Bing sonuç bağlantıları doğrudan hedef değil, izleme
    yönlendirmesidir:

        https://www.bing.com/ck/a?!&&p=…&u=a1aHR0cHM6Ly9kb2NzLmdvZG90…&ntn=3

    Gerçek adres `u=` parametresinde **base64** (URL-güvenli, `a1` öneki)
    olarak durur. Çözülmezse `bing.com` filtresine takılır ve **tüm**
    sonuçlar elenır — ölçüldü: 10 sonuç bulundu, 0 tanesi kaldı.
    """
    import base64

    try:
        sorgu = urllib.parse.urlparse(url).query
        deger = urllib.parse.parse_qs(sorgu).get("u", [""])[0]
        if not deger:
            return url
        # Bing `a1` öneki ekler ve base64'ü URL-güvenli biçimde yazar
        ham = deger[2:] if deger.startswith("a1") else deger
        ham = ham.replace("-", "+").replace("_", "/")
        ham += "=" * (-len(ham) % 4)
        cozulmus = base64.b64decode(ham).decode("utf-8", errors="replace")
        if cozulmus.startswith(("http://", "https://")):
            return cozulmus
    except Exception:
        pass
    return url


def _search_bing(query: str, *, limit: int,
                 timeout: float) -> List[Dict[str, str]]:
    """
    Bing üzerinden arama yapar (yedek motor).

    Neden gerekli? Ölçüldü: DuckDuckGo, aynı IP'den yoğun kullanım sonrası
    **kalıcı olarak** boş sonuç döndürmeye başlıyor (15-19 sn gecikme,
    0 sonuç) ve bu durum 90 sn beklemekle geçmiyor. Tek motora bağlı
    kalmak arama yeteneğini kırılgan yapar.
    """
    address = ("https://www.bing.com/search?q="
               + urllib.parse.quote_plus(query) + "&count=20")
    markup = _fetch_markup(address, timeout=timeout)
    if not markup:
        return []

    sonuclar: List[Dict[str, str]] = []
    # Bing sonuç bağlantıları: <h2><a href="URL">Başlık</a></h2>
    desen = re.compile(
        r'<h2[^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL)
    for eslesme in desen.finditer(markup):
        url = html.unescape(eslesme.group(1))
        baslik = html.unescape(re.sub(r"<[^>]+>", "", eslesme.group(2))).strip()
        if not baslik or not url:
            continue
        # Bing izleme yönlendirmesini gerçek hedefe çevir
        if "bing.com/ck/" in url:
            url = _resolve_bing_url(url)
        # Bing'in kendi alan adlarını ve reklamlarını atla
        if any(k in url for k in ("bing.com", "go.microsoft.com",
                                  "microsofttranslator")):
            continue
        if any(x["url"] == url for x in sonuclar):
            continue
        sonuclar.append({"title": baslik, "url": url, "snippet": ""})
        if len(sonuclar) >= limit:
            break
    return sonuclar


def _search_wikipedia(query: str, *, limit: int,
                      timeout: float) -> List[Dict[str, str]]:
    """
    Wikipedia API üzerinden arama yapar (son yedek).

    Bot koruması yoktur ve API'si açıktır — ancak kapsamı ansiklopediktir
    (kod örnekleri için zayıf). Bu yüzden son sırada denenir.
    """
    address = ("https://tr.wikipedia.org/w/api.php?action=query&list=search"
               "&format=json&srlimit=" + str(max(1, limit))
               + "&srsearch=" + urllib.parse.quote_plus(query))
    markup = _fetch_markup(address, timeout=timeout)
    if not markup:
        return []
    try:
        veri = json.loads(markup)
    except Exception:
        return []
    sonuclar: List[Dict[str, str]] = []
    for kayit in ((veri.get("query") or {}).get("search") or []):
        baslik = str(kayit.get("title") or "").strip()
        if not baslik:
            continue
        sonuclar.append({
            "title": baslik,
            "url": ("https://tr.wikipedia.org/wiki/"
                    + urllib.parse.quote(baslik.replace(" ", "_"))),
            "snippet": re.sub(r"<[^>]+>", "",
                              str(kayit.get("snippet") or "")).strip(),
        })
        if len(sonuclar) >= limit:
            break
    return sonuclar


def search_web(query: str, *, limit: int = 5,
               timeout: float = DEFAULT_TIMEOUT,
               retries: int = 1) -> List[Dict[str, str]]:
    """
    Web araması yapar ve sonuç listesini döndürür.

    **Motor zinciri** kullanılır — biri engellenirse diğeri devreye girer:

        1. **DuckDuckGo Lite** — API anahtarı gerektirmez, en iyi kapsam
        2. **Bing** — DDG engellediğinde (ölçüldü: kalıcı olabiliyor)
        3. **Wikipedia API** — bot koruması yok, ama kapsamı ansiklopedik

    ⚠ Ölçülen sorun: tek motora bağlı kalmak kırılgan. DuckDuckGo, yoğun
    kullanım sonrası **kalıcı olarak** boş sonuç döndürdü (15-19 sn
    gecikme, 0 sonuç) ve bu 90 sn beklemekle geçmedi. Bu durumda ajan
    "arama çalışmıyor" diyerek araştırmayı bırakıyordu.

    Args:
        query: Arama sorgusu
        limit: En fazla sonuç
        timeout: Zaman aşımı
        retries: Boş yanıtta yeniden deneme sayısı

    Returns:
        [{"title", "url", "snippet"}, …]
    """
    if not query.strip():
        return []

    # ── 1) DuckDuckGo Lite (birincil) ──
    address = ("https://lite.duckduckgo.com/lite/?q="
               + urllib.parse.quote_plus(query))
    for attempt in range(max(1, retries + 1)):
        # Nazik aralık: son aramadan bu yana yeterli süre geçsin
        elapsed = time.time() - _LAST_SEARCH["at"]
        if elapsed < MIN_SEARCH_INTERVAL:
            time.sleep(MIN_SEARCH_INTERVAL - elapsed)
        _LAST_SEARCH["at"] = time.time()

        markup = _fetch_markup(address, timeout=timeout)
        results = _parse_ddg_lite(markup, limit=limit)
        if results:
            logger.info(f"Arama (ddg): {query[:50]!r} → {len(results)} sonuç")
            return results
        # ⚠ Bekleme yapmadan yedek motora geçilir. Ölçüldü: DDG kalıcı
        # olarak engellediğinde her deneme ~16 sn sürüyor ve zincir
        # 40+ saniye bekleyip yine boş dönüyordu. Yedekler **hızlı**
        # (0,3-0,6 sn), bu yüzden önce onlar denenir.
        break

    # ── 2) Yedek motorlar ──
    # GitHub önce denenir: bot koruması yok, teknik sorularda isabetli
    for motor, islev in (("github", _search_github),
                         ("bing", _search_bing),
                         ("wikipedia", _search_wikipedia)):
        try:
            sonuc = islev(query, limit=limit, timeout=timeout)
        except Exception as exc:
            logger.debug(f"{motor} araması başarısız: {exc}")
            sonuc = []
        if sonuc:
            logger.info(
                f"Arama ({motor}): {query[:50]!r} → {len(sonuc)} sonuç "
                f"(birincil motor boş döndü)"
            )
            return sonuc

    logger.warning(f"Arama sonuçsuz (tüm motorlar): {query[:60]!r}")
    return []


def _fetch_markup(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Ham HTML indirir (arama sonucu ayrıştırması için)."""
    request = urllib.request.Request(url, method="GET")
    request.add_header("User-Agent", USER_AGENT)
    request.add_header("Accept", "text/html")
    request.add_header("Accept-Language", "tr,en;q=0.8")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(400 * 1024).decode("utf-8", "replace")
    except Exception as exc:
        logger.debug(f"Ham HTML alınamadı: {exc}")
        return ""


def _parse_ddg_lite(markup: str, *, limit: int = 5) -> List[Dict[str, str]]:
    """DuckDuckGo Lite HTML'inden başlık/adres/özet çıkarır."""
    if not markup:
        return []
    links = _DDG_LINK.findall(markup)
    snippets = [_clean_fragment(item) for item in _DDG_SNIPPET.findall(markup)]

    results: List[Dict[str, str]] = []
    for index, (href, label) in enumerate(links):
        target = _decode_ddg_url(href)
        if not target or "duckduckgo.com" in target:
            continue
        title = _clean_fragment(label)
        snippet = snippets[index] if index < len(snippets) else ""
        results.append({
            "title": title[:200],
            "url": target,
            "snippet": snippet[:400],
        })
        if len(results) >= limit:
            break
    return results


def _clean_fragment(fragment: str) -> str:
    """HTML parçasını düz metne çevirir (etiketleri atar)."""
    text = re.sub(r"<[^>]+>", " ", fragment or "")
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()


#: Öğretmen modele kaynak adresi önermesi için verilen istem.
#:
#: Neden arama motoru yerine bu? Ölçüldü: DuckDuckGo (her iki uç nokta),
#: Mojeek ve Searx bot koruması uyguluyor — ajan araştırma yapamıyordu.
#: Oysa dil modelleri **resmî dokümantasyon adreslerini** güvenilir biçimde
#: bilir (`docs.python.org/3/library/asyncio.html` gibi). Adresi modelden
#: alıp sayfayı **kendimiz indirip doğrularız**; böylece bilgi uydurma değil,
#: gerçek kaynağa dayanır.
SOURCE_PROMPT = """Aşağıdaki teknik soru için, cevabı içeren **resmî ve güvenilir**
kaynak adreslerini listele.

Soru: {question}

Kurallar:
- Yalnızca gerçekten var olduğundan emin olduğun adresleri yaz.
- Öncelik: resmî dokümantasyon, standart kuruluşları, birinci elden kaynaklar.
- Blog veya forum yerine dokümantasyon tercih et.
- En fazla 4 adres.

Yanıtı SADECE JSON dizisi olarak ver, başka metin yazma:
["https://...", "https://..."]"""


def propose_sources(question: str, llm: Any, *, limit: int = 4) -> List[str]:
    """
    Öğretmen modelden **kaynak adresi** önerileri alır.

    Args:
        question: Araştırılacak soru
        llm: `LLMRouter` benzeri nesne (`chat` metodu olmalı)
        limit: En fazla adres

    Returns:
        Doğrulanmış (erişilebilir) adresler
    """
    if llm is None or not question.strip():
        return []
    try:
        from neurocore.agent.llm import LLMMessage

        request = [LLMMessage("user", SOURCE_PROMPT.format(question=question[:600]))]
        # `chat_with_budget` kullanılır: düşünme modelleri (deepseek-flash)
        # küçük token sınırında yanıtı tamamen düşünmede harcayıp **boş**
        # döndürür; bu metot bütçeyi artırıp yeniden dener.
        if hasattr(llm, "chat_with_budget"):
            response = llm.chat_with_budget(request, temperature=0.2)
        else:
            response = llm.chat(request, max_tokens=2048, temperature=0.2)
    except Exception as exc:
        logger.debug(f"Kaynak önerisi alınamadı: {exc}")
        return []
    if not response.ok:
        logger.debug(f"Kaynak önerisi başarısız: {response.error[:80]}")
        return []

    text = response.text or ""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    candidates: List[str] = []
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                candidates = [str(item).strip() for item in parsed]
        except json.JSONDecodeError:
            pass
    if not candidates:
        # JSON çözülemezse adresleri metinden ayıkla
        candidates = re.findall(r"https?://[^\s\"'<>)\]]+", text)

    verified: List[str] = []
    for url in candidates:
        url = url.rstrip(".,;)")
        if not url.startswith(("http://", "https://")):
            continue
        allowed, _ = is_public_url(url)
        if not allowed or url in verified:
            continue
        verified.append(url)
        if len(verified) >= limit:
            break
    logger.info(f"Kaynak önerisi: {question[:40]!r} → {len(verified)} adres")
    return verified


__all__ = [
    "WebPage", "fetch", "search_web", "html_to_text", "extract_links",
    "extract_title", "is_public_url", "propose_sources",
    "USER_AGENT", "MAX_BYTES", "SOURCE_PROMPT",
]
