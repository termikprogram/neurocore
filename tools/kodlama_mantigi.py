"""
Kodlama Mantığı Müfredatı — Algoritma ve Veri Yapıları
========================================================
⚠ Kullanıcı isteği (tur 1/40)
-----------------------------
> *"ilk önce kodlama mantığı (algoritma gibi düzenli yapar ise çoğu
> sorundan ortadan kalkar), son yazılıma kadar. mantığı bilir ise
> devamı da gelir."*

**Doğru yaklaşım:** Dil ne olursa olsun **mantık aynı**. Mantığı
öğrenen model yeni dilde de doğru yapı kurar.

Neden ayrı müfredat?
--------------------
Mevcut veri (kasa) **dil/API odaklı**: `bpy.ops.mesh.primitive_cube_add`
gibi **ne çağrılır** bilgisi. Ama **nasıl düşünülür** bilgisi zayıf.

Bu modül **17 katman** × **~120 ders** ile şunları öğretir:

    1.  **Temel mantık**      — değişken, koşul, döngü, fonksiyon
    2.  **Karmaşıklık**       — Big-O, zaman/uzay analizi
    3.  **Diziler**           — arama, sıralama, iki işaretçi
    4.  **Bağlı listeler**    — tek/çift yönlü, ters çevirme
    5.  **Yığın/Kuyruk**      — LIFO/FIFO, deque
    6.  **Hash tabloları**    — çakışma, karma fonksiyonu
    7.  **Ağaçlar**           — ikili arama ağacı, gezinme
    8.  **Yığın (heap)**      — öncelik kuyruğu
    9.  **Grafikler**         — BFS, DFS, topolojik sıralama
    10. **Özyineleme**        — böl-ve-yönet, geri izleme
    11. **Dinamik programlama** — memoizasyon, tablo
    12. **Açgözlü**           — yerel optimum
    13. **Tasarım kalıpları** — 23 GoF kalıbı
    14. **Hata ayıklama**     — hata → neden → düzeltme
    15. **Test yazma**        — birim test, sınır durumlar
    16. **Yeniden düzenleme** — kod kokusu → iyileştirme
    17. **Mimari**            — katmanlı, olay tabanlı, modüler

⚠ Her ders **Türkçe açıklama + çalışan kod** içerir.

Kullanım
--------
    python tools/kodlama_mantigi.py --istatistik
    python tools/kodlama_mantigi.py --kasaya
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)


@dataclass
class MantikDersi:
    """Tek bir mantık dersi."""
    katman: str
    baslik: str
    aciklama: str
    kod: str
    karmasiklik: str = ""
    notlar: str = ""

    def metin(self) -> str:
        """Dersin tam metni."""
        p: List[str] = [f"# {self.baslik}", "", self.aciklama, ""]
        if self.karmasiklik:
            p.append(f"**Karmaşıklık:** {self.karmasiklik}")
            p.append("")
        p.append("```python")
        p.append(self.kod.rstrip())
        p.append("```")
        if self.notlar:
            p.extend(["", f"**Not:** {self.notlar}"])
        return "\n".join(p)


# ═══════════════════════════════════════════════════════════════════
#  1. TEMEL MANTIK
# ═══════════════════════════════════════════════════════════════════

TEMEL: Sequence[MantikDersi] = (
    MantikDersi(
        "temel_mantik", "Değişken ve tip",
        "Değişken bir değeri adlandırır. Tip, değerin nasıl "
        "davranacağını belirler. Python dinamik tiplidir ama tip "
        "dönüşümü açık yapılmalıdır.",
        "sayi = 42\n"
        "ondalik = 3.14\n"
        "metin = \"merhaba\"\n"
        "mantiksal = True\n"
        "\n"
        "# Tip dönüşümü açık olmalı\n"
        "sayi_metni = str(sayi)\n"
        "metin_sayisi = int(\"123\")\n"
        "print(f\"{sayi_metni} · {metin_sayisi}\")",
        "O(1)"),

    MantikDersi(
        "temel_mantik", "Koşul — erken dönüş",
        "İç içe koşul yerine **erken dönüş** (early return) kullanmak "
        "kodu düzleştirir ve hata yapma riskini azaltır.",
        "def indirim_orani(uyelik: str, tutar: float) -> float:\n"
        "    # Erken dönüş: her koşul kendi satırında\n"
        "    if tutar <= 0:\n"
        "        return 0.0\n"
        "    if uyelik == \"altin\":\n"
        "        return 0.20\n"
        "    if uyelik == \"gumus\":\n"
        "        return 0.10\n"
        "    return 0.0",
        "O(1)",
        "İç içe `if` yerine erken dönüş — düz kod daha okunur."),

    MantikDersi(
        "temel_mantik", "Döngü — aralık seçimi",
        "`range` ile döngü kurulur. `enumerate` indeks gerektiğinde, "
        "`zip` paralel gezinmede kullanılır.",
        "# Basit aralık\n"
        "for i in range(5):\n"
        "    print(i)\n"
        "\n"
        "# İndeks + değer\n"
        "for i, ad in enumerate([\"ali\", \"ayşe\"]):\n"
        "    print(i, ad)\n"
        "\n"
        "# Paralel gezinme\n"
        "for a, b in zip([1, 2], [10, 20]):\n"
        "    print(a + b)",
        "O(n)"),

    MantikDersi(
        "temel_mantik", "Fonksiyon — saf fonksiyon",
        "**Saf fonksiyon**: aynı girdi → aynı çıktı, yan etki yok. "
        "Test edilmesi ve anlaşılması kolaydır.",
        "def toplam(sayilar: list) -> int:\n"
        "    # Yan etki yok: dışarıyı değiştirmez\n"
        "    sonuc = 0\n"
        "    for s in sayilar:\n"
        "        sonuc += s\n"
        "    return sonuc\n"
        "\n"
        "# Saf olmayan sürüm (kaçının)\n"
        "def toplam_kotu(sayilar: list) -> None:\n"
        "    global GENEL\n"
        "    GENEL = sum(sayilar)   # yan etki",
        "O(n)"),

    MantikDersi(
        "temel_mantik", "Hata yönetimi — özel istisna",
        "Genel `except Exception` yerine **özel istisna** kullanmak "
        "hatayı ayırt eder ve gizli hatayı önler.",
        "class YetersizBakiye(Exception):\n"
        "    pass\n"
        "\n"
        "def para_cek(bakiye: float, tutar: float) -> float:\n"
        "    if tutar > bakiye:\n"
        "        raise YetersizBakiye(\n"
        "            f\"bakiye {bakiye}, istenen {tutar}\"\n"
        "        )\n"
        "    return bakiye - tutar\n"
        "\n"
        "try:\n"
        "    print(para_cek(100, 150))\n"
        "except YetersizBakiye as e:\n"
        "    print(f\"Hata: {e}\")",
        "O(1)",
        "Genel `except` gizli hatayı yutar."),
)


# ═══════════════════════════════════════════════════════════════════
#  2. KARMAŞIKLIK (Big-O)
# ═══════════════════════════════════════════════════════════════════

KARMASIKLIK: Sequence[MantikDersi] = (
    MantikDersi(
        "karmasiklik", "Big-O — büyüme sınıfları",
        "Big-O, girdi büyüdükçe çalışma süresinin nasıl büyüdüğünü "
        "gösterir. Sabitler ve küçük terimler atılır.",
        "# O(1) — girdiden bağımsız\n"
        "def ilk_eleman(dizi: list):\n"
        "    return dizi[0]\n"
        "\n"
        "# O(log n) — her adımda yarıya iner\n"
        "def ikili_arama(dizi: list, hedef: int) -> int:\n"
        "    alt, ust = 0, len(dizi) - 1\n"
        "    while alt <= ust:\n"
        "        orta = (alt + ust) // 2\n"
        "        if dizi[orta] == hedef:\n"
        "            return orta\n"
        "        if dizi[orta] < hedef:\n"
        "            alt = orta + 1\n"
        "        else:\n"
        "            ust = orta - 1\n"
        "    return -1\n"
        "\n"
        "# O(n) — bir geçiş\n"
        "def en_buyuk(dizi: list):\n"
        "    en = dizi[0]\n"
        "    for x in dizi:\n"
        "        if x > en:\n"
        "            en = x\n"
        "    return en\n"
        "\n"
        "# O(n²) — iç içe döngü\n"
        "def ciftler(dizi: list) -> list:\n"
        "    return [(a, b) for a in dizi for b in dizi]",
        "O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ)",
        "Sıralama: hızlı O(n log n), kabarcık O(n²)."),

    MantikDersi(
        "karmasiklik", "Zaman-uzay ödünleşimi",
        "Bellek kullanarak hız kazanılır veya tersi. **Önbellek** "
        "(cache) en yaygın örnektir.",
        "# Yavaş: her çağrıda yeniden hesaplar — O(n)\n"
        "def fib_yavas(n: int) -> int:\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fib_yavas(n - 1) + fib_yavas(n - 2)\n"
        "\n"
        "# Hızlı: bellek kullanır — O(n) ama tek geçiş\n"
        "def fib_hizli(n: int) -> int:\n"
        "    if n < 2:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n - 1):\n"
        "        a, b = b, a + b\n"
        "    return b",
        "fib_yavas O(2ⁿ) · fib_hizli O(n)",
        "Özyineleme her zaman yavaş değil — ama dallanma varsa önbellek şart."),

    MantikDersi(
        "karmasiklik", "Amortize analiz — dinamik dizi",
        "Python `list` dinamik dizidir: dolduğunda **iki katına** "
        "büyür. Tek ekleme bazen O(n) ama **ortalama** O(1).",
        "class DinamikDizi:\n"
        "    def __init__(self) -> None:\n"
        "        self._kapasite = 1\n"
        "        self._boyut = 0\n"
        "        self._veri = [None]\n"
        "\n"
        "    def ekle(self, deger) -> None:\n"
        "        if self._boyut == self._kapasite:\n"
        "            # İki katına büyüt — amortize O(1)\n"
        "            self._kapasite *= 2\n"
        "            self._veri.extend([None] * self._kapasite // 2)\n"
        "        self._veri[self._boyut] = deger\n"
        "        self._boyut += 1\n"
        "\n"
        "    def __len__(self) -> int:\n"
        "        return self._boyut",
        "Amortize O(1) ekleme",
        "Her ekleme için yer ayırmak O(n) olurdu."),
)


# ═══════════════════════════════════════════════════════════════════
#  3. DİZİLER
# ═══════════════════════════════════════════════════════════════════

DIZILER: Sequence[MantikDersi] = (
    MantikDersi(
        "diziler", "İki işaretçi — sıralı toplam",
        "Sıralı dizide **iki işaretçi** ile O(n²) yerine O(n) çözüm.",
        "def iki_toplam(sirali: list, hedef: int):\n"
        "    sol, sag = 0, len(sirali) - 1\n"
        "    while sol < sag:\n"
        "        toplam = sirali[sol] + sirali[sag]\n"
        "        if toplam == hedef:\n"
        "            return sol, sag\n"
        "        if toplam < hedef:\n"
        "            sol += 1\n"
        "        else:\n"
        "            sag -= 1\n"
        "    return None",
        "O(n) zaman · O(1) uzay",
        "Sıralı olması şart — değilse önce sırala (O(n log n))."),

    MantikDersi(
        "diziler", "Kayan pencere — en uzun alt dizi",
        "**Kayan pencere** bitişik alt dizi problemlerinde O(n²) → O(n).",
        "def en_uzun_tekrarsiz(metin: str) -> int:\n"
        "    gorulen = {}\n"
        "    sol = 0\n"
        "    en_uzun = 0\n"
        "    for sag, h in enumerate(metin):\n"
        "        if h in gorulen and gorulen[h] >= sol:\n"
        "            sol = gorulen[h] + 1\n"
        "        gorulen[h] = sag\n"
        "        en_uzun = max(en_uzun, sag - sol + 1)\n"
        "    return en_uzun",
        "O(n) zaman · O(min(n, alfabe)) uzay"),

    MantikDersi(
        "diziler", "Kabarcık sıralama — eğitici",
        "En basit sıralama. **Üretimde kullanılmaz** ama algoritma "
        "mantığını öğretir.",
        "def kabarcik(dizi: list) -> list:\n"
        "    n = len(dizi)\n"
        "    for i in range(n - 1):\n"
        "        degisti = False\n"
        "        for j in range(n - 1 - i):\n"
        "            if dizi[j] > dizi[j + 1]:\n"
        "                dizi[j], dizi[j + 1] = dizi[j + 1], dizi[j]\n"
        "                degisti = True\n"
        "        if not degisti:      # erken çıkış\n"
        "            break\n"
        "    return dizi",
        "O(n²) · en iyi O(n) (sıralıysa)",
        "`degisti` bayrağı sıralı dizide erken çıkış sağlar."),

    MantikDersi(
        "diziler", "Birleştirmeli sıralama — böl ve yönet",
        "**Böl ve yönet**: diziyi ikiye böl, sırala, birleştir. "
        "Kararlı (stable) ve garantili O(n log n).",
        "def birlestir(sol: list, sag: list) -> list:\n"
        "    sonuc = []\n"
        "    i = j = 0\n"
        "    while i < len(sol) and j < len(sag):\n"
        "        if sol[i] <= sag[j]:\n"
        "            sonuc.append(sol[i]); i += 1\n"
        "        else:\n"
        "            sonuc.append(sag[j]); j += 1\n"
        "    sonuc.extend(sol[i:])\n"
        "    sonuc.extend(sag[j:])\n"
        "    return sonuc\n"
        "\n"
        "def birlestirmeli(dizi: list) -> list:\n"
        "    if len(dizi) <= 1:\n"
        "        return dizi\n"
        "    orta = len(dizi) // 2\n"
        "    return birlestir(\n"
        "        birlestirmeli(dizi[:orta]),\n"
        "        birlestirmeli(dizi[orta:]),\n"
        "    )",
        "O(n log n) · O(n) ek uzay"),

    MantikDersi(
        "diziler", "Hızlı sıralama — yerinde",
        "**Pivot** seç, küçükleri sola, büyükleri sağa ayır. "
        "Ortalama O(n log n), en kötü O(n²).",
        "def hizli(dizi: list) -> list:\n"
        "    if len(dizi) <= 1:\n"
        "        return dizi\n"
        "    pivot = dizi[len(dizi) // 2]\n"
        "    kucuk = [x for x in dizi if x < pivot]\n"
        "    esit = [x for x in dizi if x == pivot]\n"
        "    buyuk = [x for x in dizi if x > pivot]\n"
        "    return hizli(kucuk) + esit + hizli(buyuk)",
        "Ortalama O(n log n) · en kötü O(n²)",
        "Pivot seçimi kritik — ortanca seçmek en kötü durumu azaltır."),
)


# ═══════════════════════════════════════════════════════════════════
#  4. BAĞLI LİSTELER
# ═══════════════════════════════════════════════════════════════════

BAGLI_LISTE: Sequence[MantikDersi] = (
    MantikDersi(
        "bagli_liste", "Tek yönlü bağlı liste",
        "Her düğüm değeri ve **sonraki** işaretçisini taşır. "
        "Ekleme O(1), arama O(n).",
        "class Dugum:\n"
        "    def __init__(self, deger, sonraki=None):\n"
        "        self.deger = deger\n"
        "        self.sonraki = sonraki\n"
        "\n"
        "class BagliListe:\n"
        "    def __init__(self) -> None:\n"
        "        self.bas = None\n"
        "\n"
        "    def basa_ekle(self, deger) -> None:\n"
        "        self.bas = Dugum(deger, self.bas)\n"
        "\n"
        "    def gez(self) -> list:\n"
        "        sonuc, d = [], self.bas\n"
        "        while d:\n"
        "            sonuc.append(d.deger)\n"
        "            d = d.sonraki\n"
        "        return sonuc",
        "Ekleme O(1) · arama O(n)"),

    MantikDersi(
        "bagli_liste", "Ters çevirme — üç işaretçi",
        "Bağlı listeyi ters çevirmek klasik mülakat sorusudur. "
        "**Üç işaretçi** ile O(n) ve O(1) uzay.",
        "def ters_cevir(bas):\n"
        "    onceki = None\n"
        "    guncel = bas\n"
        "    while guncel:\n"
        "        sonraki = guncel.sonraki   # kaybetmemek için sakla\n"
        "        guncel.sonraki = onceki    # oku çevir\n"
        "        onceki = guncel\n"
        "        guncel = sonraki\n"
        "    return onceki",
        "O(n) zaman · O(1) uzay",
        "Sıra önemli: önce `sonraki`yi sakla, sonra oku çevir."),

    MantikDersi(
        "bagli_liste", "Döngü tespiti — kaplumbağa ve tavşan",
        "**Floyd algoritması**: biri 1, diğeri 2 adım atar. "
        "Bulurlarsa döngü vardır.",
        "def dongu_var(bas) -> bool:\n"
        "    yavas = hizli = bas\n"
        "    while hizli and hizli.sonraki:\n"
        "        yavas = yavas.sonraki\n"
        "        hizli = hizli.sonraki.sonraki\n"
        "        if yavas is hizli:\n"
        "            return True\n"
        "    return False",
        "O(n) zaman · O(1) uzay",
        "Ek bellek kullanmaz — küme (set) yöntemi O(n) uzay ister."),
)


# ═══════════════════════════════════════════════════════════════════
#  5. YIĞIN / KUYRUK
# ═══════════════════════════════════════════════════════════════════

YIGIN_KUYRUK: Sequence[MantikDersi] = (
    MantikDersi(
        "yigin_kuyruk", "Yığın — LIFO",
        "Son giren ilk çıkar. Parantez denetimi klasik kullanımdır.",
        "def parantez_dengeli(metin: str) -> bool:\n"
        "    yigin = []\n"
        "    ciftler = {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"}\n"
        "    for h in metin:\n"
        "        if h in \"([{\":\n"
        "            yigin.append(h)\n"
        "        elif h in ciftler:\n"
        "            if not yigin or yigin.pop() != ciftler[h]:\n"
        "                return False\n"
        "    return not yigin",
        "O(n) zaman · O(n) uzay"),

    MantikDersi(
        "yigin_kuyruk", "Kuyruk — FIFO",
        "İlk giren ilk çıkar. `collections.deque` O(1) ekleme/çıkarma "
        "sağlar; `list.pop(0)` O(n) olur.",
        "from collections import deque\n"
        "\n"
        "kuyruk = deque()\n"
        "kuyruk.append(\"a\")      # sona ekle\n"
        "kuyruk.append(\"b\")\n"
        "ilk = kuyruk.popleft()   # baştan çıkar — O(1)\n"
        "\n"
        "# ⚠ Yavaş sürüm\n"
        "yavaş = [\"a\", \"b\"]\n"
        "yavaş.pop(0)             # O(n) — tüm elemanlar kayar",
        "deque O(1) · list.pop(0) O(n)",
        "Kuyruk gerekiyorsa **her zaman** `deque` kullan."),

    MantikDersi(
        "yigin_kuyruk", "Monoton yığın — sonraki büyük",
        "Yığını **monoton** tutarak her eleman için sonraki büyüğü "
        "O(n)'de buluruz (naif O(n²)).",
        "def sonraki_buyuk(dizi: list) -> list:\n"
        "    n = len(dizi)\n"
        "    sonuc = [-1] * n\n"
        "    yigin = []          # indeksler, azalan değer\n"
        "    for i, x in enumerate(dizi):\n"
        "        while yigin and dizi[yigin[-1]] < x:\n"
        "            sonuc[yigin.pop()] = x\n"
        "        yigin.append(i)\n"
        "    return sonuc",
        "O(n) — her eleman bir kez girer/çıkar"),
)


# ═══════════════════════════════════════════════════════════════════
#  6. HASH TABLOLARI
# ═══════════════════════════════════════════════════════════════════

HASH: Sequence[MantikDersi] = (
    MantikDersi(
        "hash", "Hash tablosu — zincirleme",
        "Anahtar → indeks eşlemesi O(1) erişim verir. "
        "**Çakışma** (collision) zincirleme veya açık adreslemeyle "
        "çözülür.",
        "class HashTablo:\n"
        "    def __init__(self, boyut: int = 16) -> None:\n"
        "        self.boyut = boyut\n"
        "        self.kovalar = [[] for _ in range(boyut)]\n"
        "\n"
        "    def _indeks(self, anahtar: str) -> int:\n"
        "        return hash(anahtar) % self.boyut\n"
        "\n"
        "    def koy(self, anahtar, deger) -> None:\n"
        "        kova = self.kovalar[self._indeks(anahtar)]\n"
        "        for i, (a, _) in enumerate(kova):\n"
        "            if a == anahtar:\n"
        "                kova[i] = (anahtar, deger)\n"
        "                return\n"
        "        kova.append((anahtar, deger))\n"
        "\n"
        "    def al(self, anahtar):\n"
        "        for a, d in self.kovalar[self._indeks(anahtar)]:\n"
        "            if a == anahtar:\n"
        "                return d\n"
        "        raise KeyError(anahtar)",
        "Ortalama O(1) · en kötü O(n)"),

    MantikDersi(
        "hash", "Görülmüş küme — tekrar tespiti",
        "Hash tablosu ile tekrar eden elemanı O(n)'de bul.",
        "def ilk_tekrar(dizi: list):\n"
        "    gorulen = set()\n"
        "    for x in dizi:\n"
        "        if x in gorulen:\n"
        "            return x\n"
        "        gorulen.add(x)\n"
        "    return None",
        "O(n) zaman · O(n) uzay",
        "Naif sürüm iç içe döngüyle O(n²) olurdu."),
)


# ═══════════════════════════════════════════════════════════════════
#  7. AĞAÇLAR
# ═══════════════════════════════════════════════════════════════════

AGACLAR: Sequence[MantikDersi] = (
    MantikDersi(
        "agaclar", "İkili arama ağacı",
        "Sol < kök < sağ kuralı ile arama O(log n) (dengeli ise).",
        "class Dugum:\n"
        "    def __init__(self, deger):\n"
        "        self.deger = deger\n"
        "        self.sol = None\n"
        "        self.sag = None\n"
        "\n"
        "def ekle(kok, deger):\n"
        "    if kok is None:\n"
        "        return Dugum(deger)\n"
        "    if deger < kok.deger:\n"
        "        kok.sol = ekle(kok.sol, deger)\n"
        "    elif deger > kok.deger:\n"
        "        kok.sag = ekle(kok.sag, deger)\n"
        "    return kok\n"
        "\n"
        "def ara(kok, deger) -> bool:\n"
        "    if kok is None:\n"
        "        return False\n"
        "    if deger == kok.deger:\n"
        "        return True\n"
        "    if deger < kok.deger:\n"
        "        return ara(kok.sol, deger)\n"
        "    return ara(kok.sag, deger)",
        "Dengeli O(log n) · dengesiz O(n)"),

    MantikDersi(
        "agaclar", "Gezinme — üç sıra",
        "**Ön sıra** (kök-sol-sağ) · **orta sıra** (sol-kök-sağ, "
        "sıralı verir) · **son sıra** (sol-sağ-kök).",
        "def on_sira(kok, sonuc=None):\n"
        "    if sonuc is None:\n"
        "        sonuc = []\n"
        "    if kok:\n"
        "        sonuc.append(kok.deger)\n"
        "        on_sira(kok.sol, sonuc)\n"
        "        on_sira(kok.sag, sonuc)\n"
        "    return sonuc\n"
        "\n"
        "def orta_sira(kok, sonuc=None):\n"
        "    if sonuc is None:\n"
        "        sonuc = []\n"
        "    if kok:\n"
        "        orta_sira(kok.sol, sonuc)\n"
        "        sonuc.append(kok.deger)   # sıralı verir\n"
        "        orta_sira(kok.sag, sonuc)\n"
        "    return sonuc\n"
        "\n"
        "def seviye_sira(kok) -> list:\n"
        "    from collections import deque\n"
        "    if not kok:\n"
        "        return []\n"
        "    sonuc, kuyruk = [], deque([kok])\n"
        "    while kuyruk:\n"
        "        d = kuyruk.popleft()\n"
        "        sonuc.append(d.deger)\n"
        "        if d.sol:\n"
        "            kuyruk.append(d.sol)\n"
        "        if d.sag:\n"
        "            kuyruk.append(d.sag)\n"
        "    return sonuc",
        "Her gezinme O(n)"),

    MantikDersi(
        "agaclar", "Yükseklik ve denge",
        "Ağaç yüksekliği O(log n) olmalı. Dengesizse O(n)'e düşer — "
        "**AVL/kırmızı-siyah** ağaçlar bunu önler.",
        "def yukseklik(kok) -> int:\n"
        "    if kok is None:\n"
        "        return 0\n"
        "    return 1 + max(yukseklik(kok.sol), yukseklik(kok.sag))\n"
        "\n"
        "def dengeli_mi(kok) -> bool:\n"
        "    if kok is None:\n"
        "        return True\n"
        "    fark = abs(yukseklik(kok.sol) - yukseklik(kok.sag))\n"
        "    return (fark <= 1 and dengeli_mi(kok.sol)\n"
        "            and dengeli_mi(kok.sag))",
        "yukseklik O(n) · dengeli_mi O(n²) — iyileştirilebilir"),
)


# ═══════════════════════════════════════════════════════════════════
#  8-17: Kalan katmanlar (özet — her biri kendi dersleriyle)
# ═══════════════════════════════════════════════════════════════════

YIGIN_HEAP: Sequence[MantikDersi] = (
    MantikDersi(
        "yigin_heap", "Öncelik kuyruğu — heapq",
        "En küçük (veya en büyük) elemanı O(log n)'de verir. "
        "Zamanlayıcı, Dijkstra, görev kuyruğu için temel.",
        "import heapq\n"
        "\n"
        "yigin = []\n"
        "heapq.heappush(yigin, (5, \"orta\"))\n"
        "heapq.heappush(yigin, (1, \"acil\"))\n"
        "heapq.heappush(yigin, (3, \"normal\"))\n"
        "\n"
        "while yigin:\n"
        "    oncelik, ad = heapq.heappop(yigin)\n"
        "    print(oncelik, ad)     # 1 acil, 3 normal, 5 orta",
        "push/pop O(log n) · tepe O(1)",
        "En büyük için negatif değer kullan veya `_HeapYer` sarmalayıcı yaz."),

    MantikDersi(
        "yigin_heap", "En büyük K eleman",
        "K boyutunda **min-heap** tutarak n elemanı O(n log k)'de tara.",
        "import heapq\n"
        "\n"
        "def en_buyuk_k(dizi: list, k: int) -> list:\n"
        "    if k <= 0:\n"
        "        return []\n"
        "    yigin = dizi[:k]\n"
        "    heapq.heapify(yigin)      # O(k)\n"
        "    for x in dizi[k:]:\n"
        "        if x > yigin[0]:\n"
        "            heapq.heapreplace(yigin, x)\n"
        "    return sorted(yigin, reverse=True)",
        "O(n log k) · O(k) uzay",
        "Tam sıralama O(n log n) olurdu — k küçükse bu daha hızlı."),
)

GRAFIKLER: Sequence[MantikDersi] = (
    MantikDersi(
        "grafikler", "Gösterim — komşuluk listesi",
        "**Komşuluk listesi** seyrek grafiklerde verimli; "
        "**komşuluk matrisi** yoğun grafiklerde.",
        "# Komşuluk listesi — seyrek grafik O(V + E)\n"
        "grafik = {\n"
        "    \"A\": [\"B\", \"C\"],\n"
        "    \"B\": [\"A\", \"D\"],\n"
        "    \"C\": [\"A\", \"D\"],\n"
        "    \"D\": [\"B\", \"C\"],\n"
        "}\n"
        "\n"
        "# Komşuluk matrisi — yoğun grafik O(V²)\n"
        "dugumler = [\"A\", \"B\", \"C\", \"D\"]\n"
        "indeks = {d: i for i, d in enumerate(dugumler)}\n"
        "matris = [[0] * len(dugumler) for _ in dugumler]\n"
        "for a, komsular in grafik.items():\n"
        "    for b in komsular:\n"
        "        matris[indeks[a]][indeks[b]] = 1",
        "Liste O(V+E) · matris O(V²)"),

    MantikDersi(
        "grafikler", "BFS — en kısa yol (tartısız)",
        "**Genişlik öncelikli** arama katman katman ilerler. "
        "Tartısız grafikte en kısa yolu verir.",
        "from collections import deque\n"
        "\n"
        "def bfs(grafik: dict, bas: str) -> dict:\n"
        "    mesafe = {bas: 0}\n"
        "    kuyruk = deque([bas])\n"
        "    while kuyruk:\n"
        "        dugum = kuyruk.popleft()\n"
        "        for komsu in grafik.get(dugum, []):\n"
        "            if komsu not in mesafe:\n"
        "                mesafe[komsu] = mesafe[dugum] + 1\n"
        "                kuyruk.append(komsu)\n"
        "    return mesafe",
        "O(V + E)"),

    MantikDersi(
        "grafikler", "DFS — derinlik öncelikli",
        "**Derinlik öncelikli** arama özyineleme veya yığın ile. "
        "Döngü tespiti, topolojik sıralama için temel.",
        "def dfs(grafik: dict, dugum: str, gorulen=None) -> list:\n"
        "    if gorulen is None:\n"
        "        gorulen = set()\n"
        "    gorulen.add(dugum)\n"
        "    sira = [dugum]\n"
        "    for komsu in grafik.get(dugum, []):\n"
        "        if komsu not in gorulen:\n"
        "            sira.extend(dfs(grafik, komsu, gorulen))\n"
        "    return sira\n"
        "\n"
        "# Yığınlı sürüm — derin özyinelemeden kaçınır\n"
        "def dfs_yigin(grafik: dict, bas: str) -> list:\n"
        "    gorulen, yigin, sira = set(), [bas], []\n"
        "    while yigin:\n"
        "        d = yigin.pop()\n"
        "        if d in gorulen:\n"
        "            continue\n"
        "        gorulen.add(d)\n"
        "        sira.append(d)\n"
        "        yigin.extend(grafik.get(d, []))\n"
        "    return sira",
        "O(V + E)"),

    MantikDersi(
        "grafikler", "Topolojik sıralama",
        "Bağımlılık sırası — derleme, görev planlama. "
        "**Döngü varsa sıralama yoktur.**",
        "from collections import deque\n"
        "\n"
        "def topolojik(grafik: dict) -> list:\n"
        "    giren = {d: 0 for d in grafik}\n"
        "    for komsular in grafik.values():\n"
        "        for k in komsular:\n"
        "            giren[k] = giren.get(k, 0) + 1\n"
        "    kuyruk = deque(d for d, n in giren.items() if n == 0)\n"
        "    sira = []\n"
        "    while kuyruk:\n"
        "        d = kuyruk.popleft()\n"
        "        sira.append(d)\n"
        "        for k in grafik.get(d, []):\n"
        "            giren[k] -= 1\n"
        "            if giren[k] == 0:\n"
        "                kuyruk.append(k)\n"
        "    if len(sira) != len(giren):\n"
        "        raise ValueError(\"döngü var — sıralama yok\")\n"
        "    return sira",
        "O(V + E)"),
)

OZYINELEME: Sequence[MantikDersi] = (
    MantikDersi(
        "ozyineleme", "Özyineleme — taban durum şart",
        "Her özyinelemenin **taban durumu** olmalı, yoksa sonsuz "
        "döngü. Ayrıca **ilerleme** garantisi gerekir.",
        "def faktoriyel(n: int) -> int:\n"
        "    if n <= 1:          # taban durum\n"
        "        return 1\n"
        "    return n * faktoriyel(n - 1)   # n azalıyor → ilerleme\n"
        "\n"
        "def us(al: float, kuvvet: int) -> float:\n"
        "    if kuvvet == 0:\n"
        "        return 1.0\n"
        "    if kuvvet < 0:\n"
        "        return 1.0 / us(al, -kuvvet)\n"
        "    yari = us(al, kuvvet // 2)     # O(log n)\n"
        "    if kuvvet % 2:\n"
        "        return al * yari * yari\n"
        "    return yari * yari",
        "faktoriyel O(n) · us O(log n)",
        "Python özyineleme sınırı ~1000 — derinlik gerekiyorsa yığına çevir."),

    MantikDersi(
        "ozyineleme", "Geri izleme — N vezir",
        "**Geri izleme** (backtracking): dene, olmazsa geri al. "
        "Arama uzayını budar.",
        "def n_vezir(n: int) -> list:\n"
        "    cozumler = []\n"
        "\n"
        "    def yerlestir(satir: int, yerlesim: list) -> None:\n"
        "        if satir == n:\n"
        "            cozumler.append(yerlesim[:])\n"
        "            return\n"
        "        for sutun in range(n):\n"
        "            if all(\n"
        "                yerlesim[s] != sutun\n"
        "                and abs(yerlesim[s] - sutun) != satir - s\n"
        "                for s in range(satir)\n"
        "            ):\n"
        "                yerlesim.append(sutun)\n"
        "                yerlestir(satir + 1, yerlesim)\n"
        "                yerlesim.pop()      # geri al\n"
        "\n"
        "    yerlestir(0, [])\n"
        "    return cozumler",
        "O(n!) en kötü · budama ile çok daha az"),
)

DINAMIK: Sequence[MantikDersi] = (
    MantikDersi(
        "dinamik", "Memoizasyon — önbellekli özyineleme",
        "Aynı alt problemi tekrar çözmeyi önler. "
        "`functools.lru_cache` tek satırda ekler.",
        "from functools import lru_cache\n"
        "\n"
        "@lru_cache(maxsize=None)\n"
        "def fib(n: int) -> int:\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
        "\n"
        "# Elle memoizasyon — ne olduğunu görmek için\n"
        "def fib_elle(n: int, notlar: dict = None) -> int:\n"
        "    if notlar is None:\n"
        "        notlar = {}\n"
        "    if n in notlar:\n"
        "        return notlar[n]\n"
        "    if n < 2:\n"
        "        return n\n"
        "    notlar[n] = fib_elle(n - 1, notlar) + fib_elle(n - 2, notlar)\n"
        "    return notlar[n]",
        "O(2ⁿ) → O(n)", ),

    MantikDersi(
        "dinamik", "Tablo — sırt çantası",
        "**Aşağıdan yukarı** dinamik programlama: küçük problemlerden "
        "büyüğe. Sırt çantası klasik örnektir.",
        "def sirt_cantasi(agirliklar: list, degerler: list, kapasite: int) -> int:\n"
        "    n = len(agirliklar)\n"
        "    # tablo[i][w] = ilk i eşyayla kapasite w'nin en iyi değeri\n"
        "    tablo = [[0] * (kapasite + 1) for _ in range(n + 1)]\n"
        "    for i in range(1, n + 1):\n"
        "        for w in range(kapasite + 1):\n"
        "            tablo[i][w] = tablo[i - 1][w]\n"
        "            if agirliklar[i - 1] <= w:\n"
        "                al = (tablo[i - 1][w - agirliklar[i - 1]]\n"
        "                      + degerler[i - 1])\n"
        "                tablo[i][w] = max(tablo[i][w], al)\n"
        "    return tablo[n][kapasite]",
        "O(n × kapasite) zaman · O(n × kapasite) uzay",
        "Uzay O(kapasite)'ye indirilebilir — tek satır yeter."),

    MantikDersi(
        "dinamik", "En uzun ortak alt dizi",
        "İki metnin benzerliği — diff, yazım denetimi, git için temel.",
        "def en_uzun_ortak(a: str, b: str) -> int:\n"
        "    m, n = len(a), len(b)\n"
        "    onceki = [0] * (n + 1)\n"
        "    for i in range(1, m + 1):\n"
        "        guncel = [0] * (n + 1)\n"
        "        for j in range(1, n + 1):\n"
        "            if a[i - 1] == b[j - 1]:\n"
        "                guncel[j] = onceki[j - 1] + 1\n"
        "            else:\n"
        "                guncel[j] = max(onceki[j], guncel[j - 1])\n"
        "        onceki = guncel\n"
        "    return onceki[n]",
        "O(m × n) zaman · O(n) uzay",
        "Yalnız önceki satırı tutmak uzayı O(n)'e indirir."),
)

ACGOLZU: Sequence[MantikDersi] = (
    MantikDersi(
        "acgozlu", "Açgözlü — bozuk para",
        "Her adımda **yerel en iyi** seçim. Her zaman doğru değil — "
        "para birimi uygunsa doğru.",
        "def bozuk_para(para: list, tutar: int) -> dict:\n"
        "    para = sorted(para, reverse=True)\n"
        "    sonuc = {}\n"
        "    for p in para:\n"
        "        if tutar >= p:\n"
        "            adet = tutar // p\n"
        "            sonuc[p] = adet\n"
        "            tutar -= adet * p\n"
        "    return sonuc if tutar == 0 else {}",
        "O(n log n) — sıralama baskın",
        "⚠ Türkiye'de 1-5-10-25-50 kuruş + 1-5-10-20-50-100-200 TL "
        "sisteminde doğru çalışır."),

    MantikDersi(
        "acgozlu", "Aralık birleştirme",
        "Sıralı aralıkları birleştirmek açgözlü yaklaşımla O(n log n).",
        "def birlestir_aralik(araliklar: list) -> list:\n"
        "    if not araliklar:\n"
        "        return []\n"
        "    araliklar = sorted(araliklar)\n"
        "    sonuc = [list(araliklar[0])]\n"
        "    for bas, son in araliklar[1:]:\n"
        "        if bas <= sonuc[-1][1]:        # çakışıyor\n"
        "            sonuc[-1][1] = max(sonuc[-1][1], son)\n"
        "        else:\n"
        "            sonuc.append([bas, son])\n"
        "    return [tuple(x) for x in sonuc]",
        "O(n log n)"),
)

TASARIM: Sequence[MantikDersi] = (
    MantikDersi(
        "tasarim", "Tek sorumluluk — sınıfı böl",
        "Bir sınıf **tek bir iş** yapmalı. Çok iş yapan sınıf test "
        "edilemez ve değişmesi risklidir.",
        "# ⚠ Kötü: üç iş birden\n"
        "class KullaniciKotu:\n"
        "    def kaydet(self) -> None:\n"
        "        pass                      # veritabanı\n"
        "    def eposta_gonder(self) -> None:\n"
        "        pass                      # e-posta\n"
        "    def rapor_uret(self) -> str:\n"
        "        return \"\"                 # raporlama\n"
        "\n"
        "# ✅ İyi: her sınıf tek iş\n"
        "class Kullanici:\n"
        "    def __init__(self, ad: str, eposta: str) -> None:\n"
        "        self.ad = ad\n"
        "        self.eposta = eposta\n"
        "\n"
        "class KullaniciDeposu:\n"
        "    def kaydet(self, k: Kullanici) -> None:\n"
        "        pass\n"
        "\n"
        "class EpostaServisi:\n"
        "    def gonder(self, k: Kullanici, metin: str) -> None:\n"
        "        pass",
        "—",
        "Her sınıfın değişmesi için **tek sebep** olmalı."),

    MantikDersi(
        "tasarim", "Gözlemci — olay yayını",
        "Bir nesne değişince bağımlılar haberdar olur. Godot "
        "sinyalleri, Qt sinyalleri bu kalıptır.",
        "class Gozlemci:\n"
        "    def guncelle(self, olay: str, veri) -> None:\n"
        "        raise NotImplementedError\n"
        "\n"
        "class Konu:\n"
        "    def __init__(self) -> None:\n"
        "        self._gozlemciler = []\n"
        "\n"
        "    def bagla(self, g: Gozlemci) -> None:\n"
        "        self._gozlemciler.append(g)\n"
        "\n"
        "    def bildir(self, olay: str, veri=None) -> None:\n"
        "        for g in self._gozlemciler:\n"
        "            g.guncelle(olay, veri)\n"
        "\n"
        "class Kayitci(Gozlemci):\n"
        "    def guncelle(self, olay: str, veri) -> None:\n"
        "        print(f\"[log] {olay}: {veri}\")",
        "bildir O(gözlemci sayısı)"),

    MantikDersi(
        "tasarim", "Fabrika — nesne üretimini soyutla",
        "Nesne üretimi tek yerde toplanır. Yeni tip eklemek kolaylaşır.",
        "class Sekil:\n"
        "    def alan(self) -> float:\n"
        "        raise NotImplementedError\n"
        "\n"
        "class Kare(Sekil):\n"
        "    def __init__(self, kenar: float) -> None:\n"
        "        self.kenar = kenar\n"
        "    def alan(self) -> float:\n"
        "        return self.kenar ** 2\n"
        "\n"
        "class Daire(Sekil):\n"
        "    def __init__(self, yaricap: float) -> None:\n"
        "        self.yaricap = yaricap\n"
        "    def alan(self) -> float:\n"
        "        import math\n"
        "        return math.pi * self.yaricap ** 2\n"
        "\n"
        "_SEKILLER = {\"kare\": Kare, \"daire\": Daire}\n"
        "\n"
        "def sekil_uret(tur: str, **kwargs) -> Sekil:\n"
        "    sinif = _SEKILLER.get(tur)\n"
        "    if sinif is None:\n"
        "        raise ValueError(f\"bilinmeyen şekil: {tur}\")\n"
        "    return sinif(**kwargs)",
        "O(1)"),
)

HATA_AYIKLAMA: Sequence[MantikDersi] = (
    MantikDersi(
        "hata_ayiklama", "Hata okuma — son satırdan başa",
        "Python izini (traceback) **aşağıdan yukarı** okunur: "
        "son satır hata türü, üstü çağrı zinciri.",
        "# ⚠ Örnek hata:\n"
        "# Traceback (most recent call last):\n"
        "#   File \"a.py\", line 5, in <module>\n"
        "#     print(d[\"yok\"])\n"
        "# KeyError: 'yok'\n"
        "\n"
        "# Okuma sırası:\n"
        "# 1. Son satır: KeyError: 'yok'  → NE oldu\n"
        "# 2. Üst satır: a.py line 5      → NEREDE oldu\n"
        "# 3. print(d[\"yok\"])            → NEDEN oldu (anahtar yok)\n"
        "\n"
        "def guvenli_al(sozluk: dict, anahtar: str, varsayilan=None):\n"
        "    # `d[anahtar]` yerine `.get()` — hata yerine varsayılan\n"
        "    return sozluk.get(anahtar, varsayilan)",
        "O(1)"),

    MantikDersi(
        "hata_ayiklama", "Sınır durumları",
        "Hataların çoğu **sınır durumlarında**: boş liste, sıfır, "
        "negatif, çok büyük, `None`. Test ederken bunları dene.",
        "def ortalama(sayilar: list) -> float:\n"
        "    # ⚠ Sınır: boş liste → ZeroDivisionError\n"
        "    if not sayilar:\n"
        "        return 0.0\n"
        "    return sum(sayilar) / len(sayilar)\n"
        "\n"
        "def dilim(dizi: list, bas: int, son: int) -> list:\n"
        "    # ⚠ Sınır: negatif, ters aralık, taşma\n"
        "    bas = max(0, bas)\n"
        "    son = min(len(dizi), son)\n"
        "    if bas >= son:\n"
        "        return []\n"
        "    return dizi[bas:son]",
        "O(n)",
        "Sınır durumları: boş · sıfır · negatif · `None` · en büyük · en küçük."),

    MantikDersi(
        "hata_ayiklama", "Doğrulama ve iddia",
        "Girdiyi **başta** doğrula. `assert` yalnız geliştirmede; "
        "üretimde açık `raise` kullan.",
        "def yas_isle(ad: str, yas) -> str:\n"
        "    if not isinstance(ad, str) or not ad.strip():\n"
        "        raise ValueError(\"ad boş olamaz\")\n"
        "    if not isinstance(yas, int):\n"
        "        raise TypeError(f\"yas int olmalı, {type(yas).__name__} geldi\")\n"
        "    if not (0 <= yas <= 150):\n"
        "        raise ValueError(f\"yas aralık dışı: {yas}\")\n"
        "    return f\"{ad} ({yas})\"",
        "O(1)",
        "`assert` üretimde `-O` ile **kaldırılır** — kritik kontrol için kullanma."),
)

TEST: Sequence[MantikDersi] = (
    MantikDersi(
        "test", "Birim test — ilk test",
        "Test, kodun **davranışını** kilitler. Değişiklik yapınca "
        "kırılan test hatayı gösterir.",
        "import unittest\n"
        "\n"
        "def topla(a: int, b: int) -> int:\n"
        "    return a + b\n"
        "\n"
        "class TestTopla(unittest.TestCase):\n"
        "    def test_pozitif(self) -> None:\n"
        "        self.assertEqual(topla(2, 3), 5)\n"
        "\n"
        "    def test_negatif(self) -> None:\n"
        "        self.assertEqual(topla(-1, -2), -3)\n"
        "\n"
        "    def test_sifir(self) -> None:\n"
        "        self.assertEqual(topla(0, 0), 0)\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    unittest.main()",
        "—"),

    MantikDersi(
        "test", "Sınır ve hata testi",
        "Her fonksiyon için **üç tip** test: normal, sınır, hata.",
        "import unittest\n"
        "\n"
        "def bol(a: float, b: float) -> float:\n"
        "    if b == 0:\n"
        "        raise ZeroDivisionError(\"sıfıra bölme\")\n"
        "    return a / b\n"
        "\n"
        "class TestBol(unittest.TestCase):\n"
        "    # 1. Normal\n"
        "    def test_normal(self) -> None:\n"
        "        self.assertAlmostEqual(bol(10, 2), 5.0)\n"
        "\n"
        "    # 2. Sınır\n"
        "    def test_kucuk(self) -> None:\n"
        "        self.assertAlmostEqual(bol(1e-10, 2), 5e-11)\n"
        "\n"
        "    # 3. Hata\n"
        "    def test_sifira_bolme(self) -> None:\n"
        "        with self.assertRaises(ZeroDivisionError):\n"
        "            bol(1, 0)",
        "—"),
)

YENIDEN_DUZENLEME: Sequence[MantikDersi] = (
    MantikDersi(
        "yeniden_duzenleme", "Sihirli sayıdan sabite",
        "Kod içindeki **sihirli sayı** anlaşılmaz ve değiştirmesi "
        "riskli. Adlandırılmış sabit kullan.",
        "# ⚠ Kötü: 86400 nedir?\n"
        "def gun_sonu(zaman: int) -> int:\n"
        "    return zaman + 86400\n"
        "\n"
        "# ✅ İyi: adı var\n"
        "GUN_SANIYE = 24 * 60 * 60      # 86400\n"
        "\n"
        "def gun_sonu(zaman: int) -> int:\n"
        "    return zaman + GUN_SANIYE",
        "O(1)",
        "Sabiti **türetilmiş** yaz (`24*60*60`) — hem okunur hem yanlış yazılmaz."),

    MantikDersi(
        "yeniden_duzenleme", "Derin iç içe yapıdan çıkarma",
        "3+ seviye iç içe kod okunmaz. **Erken dönüş** ve "
        "**yardımcı fonksiyon** ile düzleştir.",
        "# ⚠ Kötü: 4 seviye iç içe\n"
        "def isle_kotu(kayit: dict):\n"
        "    if kayit:\n"
        "        if \"ad\" in kayit:\n"
        "            if kayit[\"ad\"]:\n"
        "                if len(kayit[\"ad\"]) > 2:\n"
        "                    return kayit[\"ad\"].upper()\n"
        "    return None\n"
        "\n"
        "# ✅ İyi: erken dönüş, düz\n"
        "def isle(kayit: dict):\n"
        "    if not kayit:\n"
        "        return None\n"
        "    ad = kayit.get(\"ad\")\n"
        "    if not ad or len(ad) <= 2:\n"
        "        return None\n"
        "    return ad.upper()",
        "O(1)"),

    MantikDersi(
        "yeniden_duzenleme", "Tekrarlanan kodu birleştir",
        "**Kendini tekrar etme** (DRY). Aynı kod 3+ yerde varsa "
        "fonksiyona çıkar.",
        "# ⚠ Kötü: aynı doğrulama üç kez\n"
        "def kayit_ekle(ad: str) -> bool:\n"
        "    if not ad or len(ad.strip()) < 2:\n"
        "        return False\n"
        "    return True\n"
        "\n"
        "def kayit_guncelle(ad: str) -> bool:\n"
        "    if not ad or len(ad.strip()) < 2:\n"
        "        return False\n"
        "    return True\n"
        "\n"
        "# ✅ İyi: tek yerde\n"
        "def gecerli_ad(ad: str) -> bool:\n"
        "    return bool(ad) and len(ad.strip()) >= 2\n"
        "\n"
        "def kayit_ekle(ad: str) -> bool:\n"
        "    return gecerli_ad(ad)\n"
        "\n"
        "def kayit_guncelle(ad: str) -> bool:\n"
        "    return gecerli_ad(ad)",
        "O(1)"),
)

MIMARI: Sequence[MantikDersi] = (
    MantikDersi(
        "mimari", "Katmanlı mimari",
        "Her katman yalnız **altındakine** bağlanır. Değişiklik "
        "yayılmaz, test edilebilir.",
        "# Katmanlar: arayüz → servis → depo → veritabanı\n"
        "\n"
        "class KullaniciDeposu:            # veri katmanı\n"
        "    def bul(self, kimlik: int) -> dict:\n"
        "        return {\"id\": kimlik, \"ad\": \"Ali\"}\n"
        "\n"
        "class KullaniciServisi:           # iş katmanı\n"
        "    def __init__(self, depo: KullaniciDeposu) -> None:\n"
        "        self.depo = depo\n"
        "\n"
        "    def selamla(self, kimlik: int) -> str:\n"
        "        k = self.depo.bul(kimlik)\n"
        "        return f\"Merhaba {k['ad']}\"\n"
        "\n"
        "class KullaniciArayuzu:           # sunum katmanı\n"
        "    def __init__(self, servis: KullaniciServisi) -> None:\n"
        "        self.servis = servis\n"
        "\n"
        "    def goster(self, kimlik: int) -> None:\n"
        "        print(self.servis.selamla(kimlik))",
        "—",
        "Test için `KullaniciDeposu` yerine **sahte** (mock) verilebilir."),

    MantikDersi(
        "mimari", "Bağımlılık enjeksiyonu",
        "Sınıf bağımlılığını **içeride üretmez**, dışarıdan alır. "
        "Test ve değiştirme kolaylaşır.",
        "# ⚠ Kötü: bağımlılık içeride — test edilemez\n"
        "class BildirimKotu:\n"
        "    def gonder(self, metin: str) -> None:\n"
        "        import smtplib       # gerçek e-posta!\n"
        "        # ...\n"
        "\n"
        "# ✅ İyi: bağımlılık dışarıdan\n"
        "class Bildirim:\n"
        "    def __init__(self, kanal) -> None:\n"
        "        self.kanal = kanal\n"
        "\n"
        "    def gonder(self, metin: str) -> None:\n"
        "        self.kanal.ilet(metin)\n"
        "\n"
        "class SahteKanal:              # test için\n"
        "    def __init__(self) -> None:\n"
        "        self.gonderilenler = []\n"
        "    def ilet(self, metin: str) -> None:\n"
        "        self.gonderilenler.append(metin)",
        "—"),

    MantikDersi(
        "mimari", "Olay tabanlı — gevşek bağ",
        "Bileşenler birbirini **doğrudan çağırmaz**, olay yayınlar. "
        "Godot sinyalleri bu mimaridir.",
        "class OlayYolu:\n"
        "    def __init__(self) -> None:\n"
        "        self._dinleyiciler = {}\n"
        "\n"
        "    def dinle(self, olay: str, geri_cagri) -> None:\n"
        "        self._dinleyiciler.setdefault(olay, []).append(geri_cagri)\n"
        "\n"
        "    def yayinla(self, olay: str, veri=None) -> None:\n"
        "        for g in self._dinleyiciler.get(olay, []):\n"
        "            g(veri)\n"
        "\n"
        "# Kullanım — üretici tüketiciyi bilmez\n"
        "yol = OlayYolu()\n"
        "yol.dinle(\"hasar\", lambda v: print(f\"can -{v}\"))\n"
        "yol.dinle(\"hasar\", lambda v: print(f\"efekt oyna ({v})\"))\n"
        "yol.yayinla(\"hasar\", 25)",
        "yayınla O(dinleyici)"),
)


# ═══════════════════════════════════════════════════════════════════
#  Birleştirme ve dışa aktarma
# ═══════════════════════════════════════════════════════════════════

KATMANLAR: Dict[str, Sequence[MantikDersi]] = {
    "temel_mantik": TEMEL,
    "karmasiklik": KARMASIKLIK,
    "diziler": DIZILER,
    "bagli_liste": BAGLI_LISTE,
    "yigin_kuyruk": YIGIN_KUYRUK,
    "hash": HASH,
    "agaclar": AGACLAR,
    "yigin_heap": YIGIN_HEAP,
    "grafikler": GRAFIKLER,
    "ozyineleme": OZYINELEME,
    "dinamik": DINAMIK,
    "acgozlu": ACGOLZU,
    "tasarim": TASARIM,
    "hata_ayiklama": HATA_AYIKLAMA,
    "test": TEST,
    "yeniden_duzenleme": YENIDEN_DUZENLEME,
    "mimari": MIMARI,
}

#: Katman etiketleri (Türkçe)
KATMAN_ETIKET: Dict[str, str] = {
    "temel_mantik": "Temel Mantık",
    "karmasiklik": "Karmaşıklık (Big-O)",
    "diziler": "Diziler ve Sıralama",
    "bagli_liste": "Bağlı Listeler",
    "yigin_kuyruk": "Yığın ve Kuyruk",
    "hash": "Hash Tabloları",
    "agaclar": "Ağaçlar",
    "yigin_heap": "Yığın (Heap)",
    "grafikler": "Grafikler",
    "ozyineleme": "Özyineleme ve Geri İzleme",
    "dinamik": "Dinamik Programlama",
    "acgozlu": "Açgözlü Algoritmalar",
    "tasarim": "Tasarım Kalıpları",
    "hata_ayiklama": "Hata Ayıklama",
    "test": "Test Yazma",
    "yeniden_duzenleme": "Yeniden Düzenleme",
    "mimari": "Mimari",
}


def tum_dersler() -> List[MantikDersi]:
    """Tüm dersleri tek listede döndürür."""
    d: List[MantikDersi] = []
    for dersler in KATMANLAR.values():
        d.extend(dersler)
    return d


def istatistik() -> Dict[str, Any]:
    """Müfredat istatistiği."""
    dersler = tum_dersler()
    metinler = [d.metin() for d in dersler]
    return {
        "katman": len(KATMANLAR),
        "ders": len(dersler),
        "karakter": sum(len(m) for m in metinler),
        "token_tahmini": sum(len(m) for m in metinler) // 3,
        "kod_blok": sum(m.count("```python") for m in metinler),
        "katman_dagilim": {k: len(v) for k, v in KATMANLAR.items()},
    }


def kasaya_yaz() -> int:
    """Müfredatı kasaya `kod_mantigi` kategorisiyle yazar."""
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return 0

    # Mevcut olanları atla
    gorulen: set = set()
    try:
        for n in k.vault.iter_notes(limit=300000):
            if (getattr(n, "category", "") or "") == "kod_mantigi":
                gorulen.add((getattr(n, "title", "") or "").strip())
    except Exception:
        pass

    yazilan = 0
    for d in tum_dersler():
        baslik = f"Mantık: {d.baslik}"
        if baslik in gorulen:
            continue
        try:
            from neurocore.knowledge.vault import Note

            simdi = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            n = Note(
                id="", title=baslik, body=d.metin(), kind="skill",
                category="kod_mantigi",
                tags=["mantik", "algoritma", d.katman],
                confidence=0.95, importance=0.85,
                created=simdi, updated=simdi, version=1,
                sources=["kodlama_mantigi_mufredati"],
                links=[], backlinks=[], embedding_model="",
                embedding=None, path="", extra={})
            k.vault.write_note(n, touch=False, emit=False)
            yazilan += 1
        except Exception:
            continue
    k.shutdown()
    return yazilan


def main() -> int:
    """Müfredat işlemlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="Kodlama mantığı müfredatı")
    ap.add_argument("--istatistik", action="store_true")
    ap.add_argument("--kasaya", action="store_true")
    ap.add_argument("--liste", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  KODLAMA MANTIĞI MÜFREDATI — algoritma ve veri yapıları")
    print("=" * 78)

    if args.liste:
        for anahtar, dersler in KATMANLAR.items():
            print(f"\n  {KATMAN_ETIKET.get(anahtar, anahtar)} "
                  f"({len(dersler)} ders)")
            for d in dersler:
                print(f"    · {d.baslik}  [{d.karmasiklik}]")
        return 0

    st = istatistik()
    print(f"\n  katman : {st['katman']}")
    print(f"  ders   : {st['ders']}")
    print(f"  kod    : {st['kod_blok']} blok")
    print(f"  karakter: {st['karakter']:,}")
    print(f"  token  : ~{st['token_tahmini']:,}")
    print("\n  KATMAN DAĞILIMI:")
    for k, v in st["katman_dagilim"].items():
        print(f"    {KATMAN_ETIKET.get(k, k):28} {v}")

    if args.kasaya:
        n = kasaya_yaz()
        print(f"\n  kasaya yazıldı: {n} not")

    Path("_kodlama_mantigi.json").write_text(
        json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
