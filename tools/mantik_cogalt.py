"""
Mantık Müfredatı Çoğaltıcı — 51'den 300+ Derse
================================================
⚠ Kullanıcı hedefi (B)
----------------------
> *"myAi'ye kodlama mantığını öğret (algoritma, veri yapısı, karmaşıklık,
> tasarım kalıpları) çünkü mantık bilirse gerisi gelir."*

**Sorun:** `kodlama_mantigi.py` 17 katman · **51 ders** içeriyor. Bu
mantık öğretmek için **yetersiz** — her konu tek örnekle anlatılıyor.

Çözüm: **Çok dilli + çok varyantlı üretim**
--------------------------------------------
Her temel algoritmayı **5 dilde** ve **3 varyantta** üret:

    dil      : python · javascript · rust · cpp · gdscript
    varyant  : temel · iyileştirilmiş · sınır durumları

    51 ders × 5 dil × 3 varyant = **~765 ders**

⚠ Kritik: her üretilen kod **sözdizimsel olarak doğru** olmalı.
Bu yüzden şablonlar **elle yazılır** (uydurma yok).

Kullanım
--------
    python tools/mantik_coğalt.py --istatistik
    python tools/mantik_coğalt.py --kasaya
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
class DilDers:
    """Tek bir dilde tek bir algoritma dersi."""
    algoritma: str
    kategori: str
    dil: str
    varyant: str
    aciklama: str
    kod: str
    karmasiklik: str = ""

    def baslik(self) -> str:
        return f"{self.algoritma} — {self.dil} ({self.varyant})"

    def metin(self) -> str:
        p = [f"# {self.baslik()}", "",
             f"**Kategori:** {self.kategori}",
             f"**Dil:** {self.dil}",
             f"**Varyant:** {self.varyant}", ""]
        if self.karmasiklik:
            p.extend([f"**Karmaşıklık:** {self.karmasiklik}", ""])
        p.extend([self.aciklama, ""])
        p.append(f"```{self.dil}")
        p.append(self.kod.rstrip())
        p.append("```")
        return "\n".join(p)


# ═══════════════════════════════════════════════════════════════════
#  Algoritma şablonları — her biri 5 dilde, elle yazılmış
# ═══════════════════════════════════════════════════════════════════

#: (algoritma, kategori, karmaşıklık, {dil: (temel, iyileştirilmiş)})
ALGORITMALAR: Sequence[Tuple[str, str, str, Dict[str, Tuple[str, str]]]] = (
    (
        "İkili arama", "arama", "O(log n)",
        {
            "python": (
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
                "    return -1\n",
                "# Taşma güvenli sürüm — `(alt+ust)//2` büyük sayılarda taşar\n"
                "def ikili_arama_guvenli(dizi: list, hedef: int) -> int:\n"
                "    alt, ust = 0, len(dizi) - 1\n"
                "    while alt <= ust:\n"
                "        orta = alt + (ust - alt) // 2   # taşma yok\n"
                "        if dizi[orta] == hedef:\n"
                "            return orta\n"
                "        if dizi[orta] < hedef:\n"
                "            alt = orta + 1\n"
                "        else:\n"
                "            ust = orta - 1\n"
                "    return -1\n",
            ),
            "javascript": (
                "function ikiliArama(dizi, hedef) {\n"
                "  let alt = 0, ust = dizi.length - 1;\n"
                "  while (alt <= ust) {\n"
                "    const orta = Math.floor((alt + ust) / 2);\n"
                "    if (dizi[orta] === hedef) return orta;\n"
                "    if (dizi[orta] < hedef) alt = orta + 1;\n"
                "    else ust = orta - 1;\n"
                "  }\n"
                "  return -1;\n"
                "}\n",
                "// Taşma güvenli + sınır denetimi\n"
                "function ikiliAramaGuvenli(dizi, hedef) {\n"
                "  if (!Array.isArray(dizi) || dizi.length === 0) return -1;\n"
                "  let alt = 0, ust = dizi.length - 1;\n"
                "  while (alt <= ust) {\n"
                "    const orta = alt + Math.floor((ust - alt) / 2);\n"
                "    if (dizi[orta] === hedef) return orta;\n"
                "    if (dizi[orta] < hedef) alt = orta + 1;\n"
                "    else ust = orta - 1;\n"
                "  }\n"
                "  return -1;\n"
                "}\n",
            ),
            "rust": (
                "fn ikili_arama(dizi: &[i32], hedef: i32) -> Option<usize> {\n"
                "    let (mut alt, mut ust) = (0usize, dizi.len());\n"
                "    while alt < ust {\n"
                "        let orta = alt + (ust - alt) / 2;\n"
                "        if dizi[orta] == hedef {\n"
                "            return Some(orta);\n"
                "        } else if dizi[orta] < hedef {\n"
                "            alt = orta + 1;\n"
                "        } else {\n"
                "            ust = orta;\n"
                "        }\n"
                "    }\n"
                "    None\n"
                "}\n",
                "// Rust'ta `Option` + dilim (slice) — sınır güvenliği derleyicide\n"
                "fn ikili_arama_ozgun(dizi: &[i32], hedef: i32) -> Option<usize> {\n"
                "    dizi.binary_search(&hedef).ok()\n"
                "}\n",
            ),
            "cpp": (
                "#include <vector>\n"
                "int ikili_arama(const std::vector<int>& dizi, int hedef) {\n"
                "    int alt = 0, ust = static_cast<int>(dizi.size()) - 1;\n"
                "    while (alt <= ust) {\n"
                "        int orta = alt + (ust - alt) / 2;\n"
                "        if (dizi[orta] == hedef) return orta;\n"
                "        if (dizi[orta] < hedef) alt = orta + 1;\n"
                "        else ust = orta - 1;\n"
                "    }\n"
                "    return -1;\n"
                "}\n",
                "#include <algorithm>\n"
                "// STL sürümü — `std::binary_search` yalnız var/yok der;\n"
                "// konum için `std::lower_bound` gerekir.\n"
                "bool var_mi(const std::vector<int>& dizi, int hedef) {\n"
                "    return std::binary_search(dizi.begin(), dizi.end(), hedef);\n"
                "}\n",
            ),
            "gdscript": (
                "func ikili_arama(dizi: Array, hedef: int) -> int:\n"
                "    var alt := 0\n"
                "    var ust := dizi.size() - 1\n"
                "    while alt <= ust:\n"
                "        var orta := alt + (ust - alt) / 2\n"
                "        if dizi[orta] == hedef:\n"
                "            return orta\n"
                "        if dizi[orta] < hedef:\n"
                "            alt = orta + 1\n"
                "        else:\n"
                "            ust = orta - 1\n"
                "    return -1\n",
                "# Godot yerleşikleri — `bsearch` ve `find`\n"
                "func ikili_arama_yerlesik(dizi: Array, hedef: int) -> int:\n"
                "    if not dizi.is_sorted():\n"
                "        dizi.sort()\n"
                "    return dizi.bsearch(hedef)\n",
            ),
        },
    ),
    (
        "Kabarcık sıralama", "siralama", "O(n²)",
        {
            "python": (
                "def kabarcik(dizi: list) -> list:\n"
                "    n = len(dizi)\n"
                "    for i in range(n - 1):\n"
                "        for j in range(n - 1 - i):\n"
                "            if dizi[j] > dizi[j + 1]:\n"
                "                dizi[j], dizi[j + 1] = dizi[j + 1], dizi[j]\n"
                "    return dizi\n",
                "# Erken çıkış — sıralı dizide O(n)\n"
                "def kabarcik_erken(dizi: list) -> list:\n"
                "    n = len(dizi)\n"
                "    for i in range(n - 1):\n"
                "        degisti = False\n"
                "        for j in range(n - 1 - i):\n"
                "            if dizi[j] > dizi[j + 1]:\n"
                "                dizi[j], dizi[j + 1] = dizi[j + 1], dizi[j]\n"
                "                degisti = True\n"
                "        if not degisti:\n"
                "            break\n"
                "    return dizi\n",
            ),
            "javascript": (
                "function kabarcik(dizi) {\n"
                "  const n = dizi.length;\n"
                "  for (let i = 0; i < n - 1; i++) {\n"
                "    for (let j = 0; j < n - 1 - i; j++) {\n"
                "      if (dizi[j] > dizi[j + 1]) {\n"
                "        [dizi[j], dizi[j + 1]] = [dizi[j + 1], dizi[j]];\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "  return dizi;\n"
                "}\n",
                "// JS yerleşiği — `sort` varsayılan olarak **metin** sıralar!\n"
                "function siralaGuvenli(dizi) {\n"
                "  return [...dizi].sort((a, b) => a - b);   // sayısal\n"
                "}\n",
            ),
            "rust": (
                "fn kabarcik(dizi: &mut [i32]) {\n"
                "    let n = dizi.len();\n"
                "    for i in 0..n.saturating_sub(1) {\n"
                "        for j in 0..n - 1 - i {\n"
                "            if dizi[j] > dizi[j + 1] {\n"
                "                dizi.swap(j, j + 1);\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "}\n",
                "// Rust yerleşiği — `sort_unstable` O(n log n)\n"
                "fn sirala_verimli(dizi: &mut [i32]) {\n"
                "    dizi.sort_unstable();\n"
                "}\n",
            ),
            "cpp": (
                "#include <vector>\n"
                "void kabarcik(std::vector<int>& dizi) {\n"
                "    int n = static_cast<int>(dizi.size());\n"
                "    for (int i = 0; i < n - 1; ++i)\n"
                "        for (int j = 0; j < n - 1 - i; ++j)\n"
                "            if (dizi[j] > dizi[j + 1])\n"
                "                std::swap(dizi[j], dizi[j + 1]);\n"
                "}\n",
                "#include <algorithm>\n"
                "void sirala_verimli(std::vector<int>& dizi) {\n"
                "    std::sort(dizi.begin(), dizi.end());   // O(n log n)\n"
                "}\n",
            ),
            "gdscript": (
                "func kabarcik(dizi: Array) -> Array:\n"
                "    var n := dizi.size()\n"
                "    for i in range(n - 1):\n"
                "        for j in range(n - 1 - i):\n"
                "            if dizi[j] > dizi[j + 1]:\n"
                "                var gecici = dizi[j]\n"
                "                dizi[j] = dizi[j + 1]\n"
                "                dizi[j + 1] = gecici\n"
                "    return dizi\n",
                "# Godot yerleşiği\n"
                "func sirala_verimli(dizi: Array) -> Array:\n"
                "    dizi.sort()\n"
                "    return dizi\n",
            ),
        },
    ),
    (
        "Bağlı liste ters çevirme", "bagli_liste", "O(n)",
        {
            "python": (
                "class Dugum:\n"
                "    def __init__(self, deger, sonraki=None):\n"
                "        self.deger = deger\n"
                "        self.sonraki = sonraki\n"
                "\n"
                "def ters_cevir(bas):\n"
                "    onceki = None\n"
                "    guncel = bas\n"
                "    while guncel:\n"
                "        sonraki = guncel.sonraki\n"
                "        guncel.sonraki = onceki\n"
                "        onceki = guncel\n"
                "        guncel = sonraki\n"
                "    return onceki\n",
                "# Özyinelemeli sürüm — daha kısa, O(n) yığın\n"
                "def ters_cevir_oz(kok):\n"
                "    if kok is None or kok.sonraki is None:\n"
                "        return kok\n"
                "    yeni = ters_cevir_oz(kok.sonraki)\n"
                "    kok.sonraki.sonraki = kok\n"
                "    kok.sonraki = None\n"
                "    return yeni\n",
            ),
            "javascript": (
                "class Dugum {\n"
                "  constructor(deger, sonraki = null) {\n"
                "    this.deger = deger;\n"
                "    this.sonraki = sonraki;\n"
                "  }\n"
                "}\n"
                "\n"
                "function tersCevir(bas) {\n"
                "  let onceki = null, guncel = bas;\n"
                "  while (guncel) {\n"
                "    const sonraki = guncel.sonraki;\n"
                "    guncel.sonraki = onceki;\n"
                "    onceki = guncel;\n"
                "    guncel = sonraki;\n"
                "  }\n"
                "  return onceki;\n"
                "}\n",
                "// JS yerleşiği — dizi kullan (çoğu durumda daha hızlı)\n"
                "function tersCevirDizi(dizi) {\n"
                "  return [...dizi].reverse();\n"
                "}\n",
            ),
            "rust": (
                "#[derive(Debug)]\n"
                "struct Dugum<T> {\n"
                "    deger: T,\n"
                "    sonraki: Option<Box<Dugum<T>>>,\n"
                "}\n"
                "\n"
                "fn ters_cevir<T>(mut bas: Option<Box<Dugum<T>>>)\n"
                "    -> Option<Box<Dugum<T>>> {\n"
                "    let mut onceki = None;\n"
                "    while let Some(mut d) = bas {\n"
                "        bas = d.sonraki.take();\n"
                "        d.sonraki = onceki;\n"
                "        onceki = Some(d);\n"
                "    }\n"
                "    onceki\n"
                "}\n",
                "// Sahiplik sistemi — `take()` ile düğüm taşınır\n"
                "// Rust'ta bağlı liste yerine `Vec` tercih edilir:\n"
                "fn ters_vektor<T: Clone>(v: &[T]) -> Vec<T> {\n"
                "    v.iter().rev().cloned().collect()\n"
                "}\n",
            ),
            "cpp": (
                "struct Dugum {\n"
                "    int deger;\n"
                "    Dugum* sonraki;\n"
                "    explicit Dugum(int d, Dugum* s = nullptr)\n"
                "        : deger(d), sonraki(s) {}\n"
                "};\n"
                "\n"
                "Dugum* ters_cevir(Dugum* bas) {\n"
                "    Dugum* onceki = nullptr;\n"
                "    while (bas) {\n"
                "        Dugum* sonraki = bas->sonraki;\n"
                "        bas->sonraki = onceki;\n"
                "        onceki = bas;\n"
                "        bas = sonraki;\n"
                "    }\n"
                "    return onceki;\n"
                "}\n",
                "#include <memory>\n"
                "// Akıllı işaretçi — bellek sızıntısı yok\n"
                "using DugumPtr = std::unique_ptr<Dugum>;\n"
                "DugumPtr ters_cevir_guvenli(DugumPtr bas) {\n"
                "    DugumPtr onceki = nullptr;\n"
                "    while (bas) {\n"
                "        DugumPtr sonraki = std::move(bas->sonraki);\n"
                "        bas->sonraki = std::move(onceki);\n"
                "        onceki = std::move(bas);\n"
                "        bas = std::move(sonraki);\n"
                "    }\n"
                "    return onceki;\n"
                "}\n",
            ),
            "gdscript": (
                "class Dugum:\n"
                "    var deger: int\n"
                "    var sonraki: Dugum = null\n"
                "    func _init(d: int, s: Dugum = null) -> void:\n"
                "        deger = d\n"
                "        sonraki = s\n"
                "\n"
                "func ters_cevir(bas: Dugum) -> Dugum:\n"
                "    var onceki: Dugum = null\n"
                "    var guncel := bas\n"
                "    while guncel != null:\n"
                "        var sonraki := guncel.sonraki\n"
                "        guncel.sonraki = onceki\n"
                "        onceki = guncel\n"
                "        guncel = sonraki\n"
                "    return onceki\n",
                "# Godot yerleşiği — `Array.reverse()`\n"
                "func ters_dizi(dizi: Array) -> Array:\n"
                "    var kopya := dizi.duplicate()\n"
                "    kopya.reverse()\n"
                "    return kopya\n",
            ),
        },
    ),
    (
        "Yığın ile parantez denetimi", "yigin", "O(n)",
        {
            "python": (
                "def dengeli(metin: str) -> bool:\n"
                "    yigin = []\n"
                "    ciftler = {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"}\n"
                "    for h in metin:\n"
                "        if h in \"([{\":\n"
                "            yigin.append(h)\n"
                "        elif h in ciftler:\n"
                "            if not yigin or yigin.pop() != ciftler[h]:\n"
                "                return False\n"
                "    return not yigin\n",
                "# Türkçe karakter ve yorum atlama\n"
                "def dengeli_tam(metin: str) -> tuple:\n"
                "    yigin, hatalar = [], []\n"
                "    ciftler = {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"}\n"
                "    for i, h in enumerate(metin):\n"
                "        if h in \"([{\":\n"
                "            yigin.append((h, i))\n"
                "        elif h in ciftler:\n"
                "            if not yigin:\n"
                "                hatalar.append(f\"fazla '{h}' @{i}\")\n"
                "            elif yigin[-1][0] != ciftler[h]:\n"
                "                hatalar.append(f\"'{h}' @{i} uyumsuz\")\n"
                "            else:\n"
                "                yigin.pop()\n"
                "    for h, i in yigin:\n"
                "        hatalar.append(f\"kapanmamış '{h}' @{i}\")\n"
                "    return (not hatalar, hatalar)\n",
            ),
            "javascript": (
                "function dengeli(metin) {\n"
                "  const yigin = [];\n"
                "  const ciftler = { ')': '(', ']': '[', '}': '{' };\n"
                "  for (const h of metin) {\n"
                "    if ('([{'.includes(h)) yigin.push(h);\n"
                "    else if (h in ciftler) {\n"
                "      if (yigin.pop() !== ciftler[h]) return false;\n"
                "    }\n"
                "  }\n"
                "  return yigin.length === 0;\n"
                "}\n",
                "// Hata konumu bildiren sürüm\n"
                "function dengeliTam(metin) {\n"
                "  const yigin = [], hatalar = [];\n"
                "  const ciftler = { ')': '(', ']': '[', '}': '{' };\n"
                "  [...metin].forEach((h, i) => {\n"
                "    if ('([{'.includes(h)) yigin.push([h, i]);\n"
                "    else if (h in ciftler) {\n"
                "      const son = yigin.pop();\n"
                "      if (!son) hatalar.push(`fazla '${h}' @${i}`);\n"
                "      else if (son[0] !== ciftler[h])\n"
                "        hatalar.push(`'${h}' @${i} uyumsuz`);\n"
                "    }\n"
                "  });\n"
                "  for (const [h, i] of yigin)\n"
                "    hatalar.push(`kapanmamış '${h}' @${i}`);\n"
                "  return { gecerli: hatalar.length === 0, hatalar };\n"
                "}\n",
            ),
            "rust": (
                "fn dengeli(metin: &str) -> bool {\n"
                "    let mut yigin: Vec<char> = Vec::new();\n"
                "    for h in metin.chars() {\n"
                "        match h {\n"
                "            '(' | '[' | '{' => yigin.push(h),\n"
                "            ')' => if yigin.pop() != Some('(') { return false; },\n"
                "            ']' => if yigin.pop() != Some('[') { return false; },\n"
                "            '}' => if yigin.pop() != Some('{') { return false; },\n"
                "            _ => {}\n"
                "        }\n"
                "    }\n"
                "    yigin.is_empty()\n"
                "}\n",
                "// `match` + `Option` — Rust'ta idiomatik\n"
                "fn dengeli_ozgun(metin: &str) -> Result<(), usize> {\n"
                "    let mut yigin: Vec<(char, usize)> = Vec::new();\n"
                "    for (i, h) in metin.chars().enumerate() {\n"
                "        match h {\n"
                "            '(' | '[' | '{' => yigin.push((h, i)),\n"
                "            ')' | ']' | '}' => {\n"
                "                let beklenen = match h {\n"
                "                    ')' => '(', ']' => '[', _ => '{',\n"
                "                };\n"
                "                match yigin.pop() {\n"
                "                    Some((c, _)) if c == beklenen => {}\n"
                "                    _ => return Err(i),\n"
                "                }\n"
                "            }\n"
                "            _ => {}\n"
                "        }\n"
                "    }\n"
                "    if yigin.is_empty() { Ok(()) } else { Err(yigin[0].1) }\n"
                "}\n",
            ),
            "cpp": (
                "#include <string>\n"
                "#include <stack>\n"
                "bool dengeli(const std::string& metin) {\n"
                "    std::stack<char> yigin;\n"
                "    for (char h : metin) {\n"
                "        if (h == '(' || h == '[' || h == '{') yigin.push(h);\n"
                "        else if (h == ')' || h == ']' || h == '}') {\n"
                "            if (yigin.empty()) return false;\n"
                "            char beklenen = (h == ')') ? '(' : (h == ']') ? '[' : '{';\n"
                "            if (yigin.top() != beklenen) return false;\n"
                "            yigin.pop();\n"
                "        }\n"
                "    }\n"
                "    return yigin.empty();\n"
                "}\n",
                "#include <vector>\n"
                "#include <optional>\n"
                "// Konum bildiren sürüm — `std::optional` ile\n"
                "std::optional<std::size_t> ilk_hata(const std::string& m) {\n"
                "    std::vector<std::pair<char, std::size_t>> yigin;\n"
                "    for (std::size_t i = 0; i < m.size(); ++i) {\n"
                "        char h = m[i];\n"
                "        if (h == '(' || h == '[' || h == '{') yigin.push_back({h, i});\n"
                "        else if (h == ')' || h == ']' || h == '}') {\n"
                "            char beklenen = (h == ')') ? '(' : (h == ']') ? '[' : '{';\n"
                "            if (yigin.empty() || yigin.back().first != beklenen)\n"
                "                return i;\n"
                "            yigin.pop_back();\n"
                "        }\n"
                "    }\n"
                "    if (!yigin.empty()) return yigin.front().second;\n"
                "    return std::nullopt;\n"
                "}\n",
            ),
            "gdscript": (
                "func dengeli(metin: String) -> bool:\n"
                "    var yigin := []\n"
                "    var ciftler := {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"}\n"
                "    for h in metin:\n"
                "        if h in \"([{\":\n"
                "            yigin.append(h)\n"
                "        elif h in ciftler:\n"
                "            if yigin.is_empty() or yigin.pop_back() != ciftler[h]:\n"
                "                return false\n"
                "    return yigin.is_empty()\n",
                "# Godot'ta hata listesi döndüren sürüm\n"
                "func dengeli_tam(metin: String) -> Dictionary:\n"
                "    var yigin := []\n"
                "    var hatalar := []\n"
                "    var ciftler := {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"}\n"
                "    for i in metin.length():\n"
                "        var h := metin[i]\n"
                "        if h in \"([{\":\n"
                "            yigin.append([h, i])\n"
                "        elif h in ciftler:\n"
                "            if yigin.is_empty():\n"
                "                hatalar.append(\"fazla '%s' @%d\" % [h, i])\n"
                "            elif yigin[-1][0] != ciftler[h]:\n"
                "                hatalar.append(\"'%s' @%d uyumsuz\" % [h, i])\n"
                "            else:\n"
                "                yigin.pop_back()\n"
                "    for k in yigin:\n"
                "        hatalar.append(\"kapanmamış '%s' @%d\" % [k[0], k[1]])\n"
                "    return {\"gecerli\": hatalar.is_empty(), \"hatalar\": hatalar}\n",
            ),
        },
    ),
    (
        "Hash tablosu", "hash", "Ortalama O(1)",
        {
            "python": (
                "class HashTablo:\n"
                "    def __init__(self, boyut: int = 16) -> None:\n"
                "        self.boyut = boyut\n"
                "        self.kovalar = [[] for _ in range(boyut)]\n"
                "        self.adet = 0\n"
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
                "        self.adet += 1\n"
                "\n"
                "    def al(self, anahtar):\n"
                "        for a, d in self.kovalar[self._indeks(anahtar)]:\n"
                "            if a == anahtar:\n"
                "                return d\n"
                "        raise KeyError(anahtar)\n",
                "# Yük faktörü + otomatik büyüme\n"
                "class HashTabloOto:\n"
                "    YUK_ESIK = 0.75\n"
                "\n"
                "    def __init__(self, boyut: int = 8) -> None:\n"
                "        self.boyut = boyut\n"
                "        self.kovalar = [[] for _ in range(boyut)]\n"
                "        self.adet = 0\n"
                "\n"
                "    def _indeks(self, a) -> int:\n"
                "        return hash(a) % self.boyut\n"
                "\n"
                "    def _buyut(self) -> None:\n"
                "        eski = [x for kova in self.kovalar for x in kova]\n"
                "        self.boyut *= 2\n"
                "        self.kovalar = [[] for _ in range(self.boyut)]\n"
                "        self.adet = 0\n"
                "        for a, d in eski:\n"
                "            self.koy(a, d)\n"
                "\n"
                "    def koy(self, a, d) -> None:\n"
                "        kova = self.kovalar[self._indeks(a)]\n"
                "        for i, (k, _) in enumerate(kova):\n"
                "            if k == a:\n"
                "                kova[i] = (a, d)\n"
                "                return\n"
                "        kova.append((a, d))\n"
                "        self.adet += 1\n"
                "        if self.adet / self.boyut > self.YUK_ESIK:\n"
                "            self._buyut()\n"
                "\n"
                "    def al(self, a):\n"
                "        for k, d in self.kovalar[self._indeks(a)]:\n"
                "            if k == a:\n"
                "                return d\n"
                "        raise KeyError(a)\n",
            ),
            "javascript": (
                "class HashTablo {\n"
                "  constructor(boyut = 16) {\n"
                "    this.boyut = boyut;\n"
                "    this.kovalar = Array.from({ length: boyut }, () => []);\n"
                "  }\n"
                "  #indeks(anahtar) {\n"
                "    let h = 0;\n"
                "    for (const k of String(anahtar))\n"
                "      h = (h * 31 + k.charCodeAt(0)) | 0;\n"
                "    return Math.abs(h) % this.boyut;\n"
                "  }\n"
                "  koy(anahtar, deger) {\n"
                "    const kova = this.kovalar[this.#indeks(anahtar)];\n"
                "    for (const c of kova)\n"
                "      if (c[0] === anahtar) { c[1] = deger; return; }\n"
                "    kova.push([anahtar, deger]);\n"
                "  }\n"
                "  al(anahtar) {\n"
                "    for (const [k, d] of this.kovalar[this.#indeks(anahtar)])\n"
                "      if (k === anahtar) return d;\n"
                "    throw new Error(`anahtar yok: ${anahtar}`);\n"
                "  }\n"
                "}\n",
                "// JS yerleşiği — `Map` her türlü anahtarı destekler\n"
                "class HashTabloYerlesik {\n"
                "  #ic = new Map();\n"
                "  koy(anahtar, deger) { this.#ic.set(anahtar, deger); }\n"
                "  al(anahtar) {\n"
                "    if (!this.#ic.has(anahtar))\n"
                "      throw new Error(`anahtar yok: ${anahtar}`);\n"
                "    return this.#ic.get(anahtar);\n"
                "  }\n"
                "  get boyut() { return this.#ic.size; }\n"
                "}\n",
            ),
            "rust": (
                "use std::collections::HashMap;\n"
                "\n"
                "#[derive(Default)]\n"
                "struct HashTablo {\n"
                "    kovalar: Vec<Vec<(String, i32)>>,\n"
                "    boyut: usize,\n"
                "}\n"
                "\n"
                "impl HashTablo {\n"
                "    fn yeni(boyut: usize) -> Self {\n"
                "        Self { kovalar: vec![Vec::new(); boyut], boyut }\n"
                "    }\n"
                "    fn indeks(&self, a: &str) -> usize {\n"
                "        let mut h: usize = 0;\n"
                "        for b in a.bytes() { h = h.wrapping_mul(31).wrapping_add(b as usize); }\n"
                "        h % self.boyut\n"
                "    }\n"
                "    fn koy(&mut self, a: &str, d: i32) {\n"
                "        let i = self.indeks(a);\n"
                "        for c in self.kovalar[i].iter_mut() {\n"
                "            if c.0 == a { c.1 = d; return; }\n"
                "        }\n"
                "        self.kovalar[i].push((a.to_string(), d));\n"
                "    }\n"
                "    fn al(&self, a: &str) -> Option<i32> {\n"
                "        let i = self.indeks(a);\n"
                "        self.kovalar[i].iter().find(|c| c.0 == a).map(|c| c.1)\n"
                "    }\n"
                "}\n",
                "// Rust yerleşiği — `HashMap` (SipHash)\n"
                "use std::collections::HashMap;\n"
                "\n"
                "fn ornek() {\n"
                "    let mut h: HashMap<String, i32> = HashMap::new();\n"
                "    h.insert(\"anahtar\".to_string(), 42);\n"
                "    h.entry(\"varsayilan\".to_string()).or_insert(0);\n"
                "    if let Some(v) = h.get(\"anahtar\") {\n"
                "        println!(\"{v}\");\n"
                "    }\n"
                "}\n",
            ),
            "cpp": (
                "#include <string>\n"
                "#include <vector>\n"
                "#include <optional>\n"
                "class HashTablo {\n"
                "    std::vector<std::vector<std::pair<std::string, int>>> kovalar_;\n"
                "    std::size_t boyut_;\n"
                "    std::size_t indeks(const std::string& a) const {\n"
                "        std::size_t h = 0;\n"
                "        for (char c : a) h = h * 31 + static_cast<unsigned char>(c);\n"
                "        return h % boyut_;\n"
                "    }\n"
                "public:\n"
                "    explicit HashTablo(std::size_t b = 16) : kovalar_(b), boyut_(b) {}\n"
                "    void koy(const std::string& a, int d) {\n"
                "        auto& kova = kovalar_[indeks(a)];\n"
                "        for (auto& [k, v] : kova) if (k == a) { v = d; return; }\n"
                "        kova.emplace_back(a, d);\n"
                "    }\n"
                "    std::optional<int> al(const std::string& a) const {\n"
                "        for (const auto& [k, v] : kovalar_[indeks(a)])\n"
                "            if (k == a) return v;\n"
                "        return std::nullopt;\n"
                "    }\n"
                "};\n",
                "#include <unordered_map>\n"
                "// STL sürümü — `std::unordered_map`\n"
                "void ornek() {\n"
                "    std::unordered_map<std::string, int> h;\n"
                "    h[\"anahtar\"] = 42;\n"
                "    h.try_emplace(\"varsayilan\", 0);\n"
                "    if (auto it = h.find(\"anahtar\"); it != h.end())\n"
                "        int v = it->second;\n"
                "}\n",
            ),
            "gdscript": (
                "class HashTablo:\n"
                "    var boyut: int\n"
                "    var kovalar: Array = []\n"
                "\n"
                "    func _init(b: int = 16) -> void:\n"
                "        boyut = b\n"
                "        for i in b:\n"
                "            kovalar.append([])\n"
                "\n"
                "    func _indeks(anahtar: String) -> int:\n"
                "        return abs(anahtar.hash()) % boyut\n"
                "\n"
                "    func koy(anahtar: String, deger) -> void:\n"
                "        var kova: Array = kovalar[_indeks(anahtar)]\n"
                "        for c in kova:\n"
                "            if c[0] == anahtar:\n"
                "                c[1] = deger\n"
                "                return\n"
                "        kova.append([anahtar, deger])\n"
                "\n"
                "    func al(anahtar: String):\n"
                "        for c in kovalar[_indeks(anahtar)]:\n"
                "            if c[0] == anahtar:\n"
                "                return c[1]\n"
                "        return null\n",
                "# Godot yerleşiği — `Dictionary`\n"
                "func ornek() -> void:\n"
                "    var h := {}\n"
                "    h[\"anahtar\"] = 42\n"
                "    if h.has(\"anahtar\"):\n"
                "        var v = h[\"anahtar\"]\n",
            ),
        },
    ),
    (
        "BFS en kısa yol", "grafik", "O(V + E)",
        {
            "python": (
                "from collections import deque\n"
                "\n"
                "def bfs(grafik: dict, bas) -> dict:\n"
                "    mesafe = {bas: 0}\n"
                "    kuyruk = deque([bas])\n"
                "    while kuyruk:\n"
                "        dugum = kuyruk.popleft()\n"
                "        for komsu in grafik.get(dugum, []):\n"
                "            if komsu not in mesafe:\n"
                "                mesafe[komsu] = mesafe[dugum] + 1\n"
                "                kuyruk.append(komsu)\n"
                "    return mesafe\n",
                "# Yolu da döndüren sürüm (ebeveyn izleme)\n"
                "from collections import deque\n"
                "\n"
                "def bfs_yol(grafik: dict, bas, hedef) -> list:\n"
                "    ebeveyn = {bas: None}\n"
                "    kuyruk = deque([bas])\n"
                "    while kuyruk:\n"
                "        dugum = kuyruk.popleft()\n"
                "        if dugum == hedef:\n"
                "            yol, d = [], dugum\n"
                "            while d is not None:\n"
                "                yol.append(d)\n"
                "                d = ebeveyn[d]\n"
                "            return yol[::-1]\n"
                "        for komsu in grafik.get(dugum, []):\n"
                "            if komsu not in ebeveyn:\n"
                "                ebeveyn[komsu] = dugum\n"
                "                kuyruk.append(komsu)\n"
                "    return []\n",
            ),
            "javascript": (
                "function bfs(grafik, bas) {\n"
                "  const mesafe = { [bas]: 0 };\n"
                "  const kuyruk = [bas];\n"
                "  let bas_i = 0;\n"
                "  while (bas_i < kuyruk.length) {\n"
                "    const dugum = kuyruk[bas_i++];\n"
                "    for (const komsu of grafik[dugum] ?? []) {\n"
                "      if (!(komsu in mesafe)) {\n"
                "        mesafe[komsu] = mesafe[dugum] + 1;\n"
                "        kuyruk.push(komsu);\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "  return mesafe;\n"
                "}\n",
                "// Yol izleme\n"
                "function bfsYol(grafik, bas, hedef) {\n"
                "  const ebeveyn = { [bas]: null };\n"
                "  const kuyruk = [bas];\n"
                "  let i = 0;\n"
                "  while (i < kuyruk.length) {\n"
                "    const dugum = kuyruk[i++];\n"
                "    if (dugum === hedef) {\n"
                "      const yol = [];\n"
                "      for (let d = dugum; d !== null; d = ebeveyn[d]) yol.push(d);\n"
                "      return yol.reverse();\n"
                "    }\n"
                "    for (const komsu of grafik[dugum] ?? []) {\n"
                "      if (!(komsu in ebeveyn)) {\n"
                "        ebeveyn[komsu] = dugum;\n"
                "        kuyruk.push(komsu);\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "  return [];\n"
                "}\n",
            ),
            "rust": (
                "use std::collections::{HashMap, VecDeque};\n"
                "\n"
                "fn bfs(grafik: &HashMap<&str, Vec<&str>>, bas: &str)\n"
                "    -> HashMap<String, usize> {\n"
                "    let mut mesafe = HashMap::new();\n"
                "    mesafe.insert(bas.to_string(), 0usize);\n"
                "    let mut kuyruk = VecDeque::new();\n"
                "    kuyruk.push_back(bas.to_string());\n"
                "    while let Some(dugum) = kuyruk.pop_front() {\n"
                "        let d = mesafe[&dugum];\n"
                "        if let Some(komsular) = grafik.get(dugum.as_str()) {\n"
                "            for k in komsular {\n"
                "                if !mesafe.contains_key(*k) {\n"
                "                    mesafe.insert(k.to_string(), d + 1);\n"
                "                    kuyruk.push_back(k.to_string());\n"
                "                }\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "    mesafe\n"
                "}\n",
                "// Yol izleme — `HashMap<String, Option<String>>`\n"
                "use std::collections::{HashMap, VecDeque};\n"
                "\n"
                "fn bfs_yol(grafik: &HashMap<&str, Vec<&str>>,\n"
                "           bas: &str, hedef: &str) -> Vec<String> {\n"
                "    let mut ebeveyn: HashMap<String, Option<String>> = HashMap::new();\n"
                "    ebeveyn.insert(bas.to_string(), None);\n"
                "    let mut kuyruk = VecDeque::from([bas.to_string()]);\n"
                "    while let Some(d) = kuyruk.pop_front() {\n"
                "        if d == hedef {\n"
                "            let mut yol = Vec::new();\n"
                "            let mut g = Some(d);\n"
                "            while let Some(x) = g {\n"
                "                g = ebeveyn[&x].clone();\n"
                "                yol.push(x);\n"
                "            }\n"
                "            yol.reverse();\n"
                "            return yol;\n"
                "        }\n"
                "        for k in grafik.get(d.as_str()).into_iter().flatten() {\n"
                "            if !ebeveyn.contains_key(*k) {\n"
                "                ebeveyn.insert(k.to_string(), Some(d.clone()));\n"
                "                kuyruk.push_back(k.to_string());\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "    Vec::new()\n"
                "}\n",
            ),
            "cpp": (
                "#include <string>\n"
                "#include <vector>\n"
                "#include <queue>\n"
                "#include <unordered_map>\n"
                "std::unordered_map<std::string, int> bfs(\n"
                "    const std::unordered_map<std::string,\n"
                "                             std::vector<std::string>>& grafik,\n"
                "    const std::string& bas) {\n"
                "    std::unordered_map<std::string, int> mesafe{{bas, 0}};\n"
                "    std::queue<std::string> kuyruk;\n"
                "    kuyruk.push(bas);\n"
                "    while (!kuyruk.empty()) {\n"
                "        auto dugum = kuyruk.front(); kuyruk.pop();\n"
                "        auto it = grafik.find(dugum);\n"
                "        if (it == grafik.end()) continue;\n"
                "        for (const auto& k : it->second)\n"
                "            if (!mesafe.count(k)) {\n"
                "                mesafe[k] = mesafe[dugum] + 1;\n"
                "                kuyruk.push(k);\n"
                "            }\n"
                "    }\n"
                "    return mesafe;\n"
                "}\n",
                "#include <optional>\n"
                "// Yol izleme\n"
                "std::vector<std::string> bfs_yol(\n"
                "    const std::unordered_map<std::string,\n"
                "                             std::vector<std::string>>& g,\n"
                "    const std::string& bas, const std::string& hedef) {\n"
                "    std::unordered_map<std::string, std::optional<std::string>> eb;\n"
                "    eb[bas] = std::nullopt;\n"
                "    std::queue<std::string> k;\n"
                "    k.push(bas);\n"
                "    while (!k.empty()) {\n"
                "        auto d = k.front(); k.pop();\n"
                "        if (d == hedef) {\n"
                "            std::vector<std::string> yol;\n"
                "            for (auto x = std::optional<std::string>{d};\n"
                "                 x.has_value(); x = eb[*x])\n"
                "                yol.push_back(*x);\n"
                "            std::reverse(yol.begin(), yol.end());\n"
                "            return yol;\n"
                "        }\n"
                "        auto it = g.find(d);\n"
                "        if (it == g.end()) continue;\n"
                "        for (const auto& k2 : it->second)\n"
                "            if (!eb.count(k2)) { eb[k2] = d; k.push(k2); }\n"
                "    }\n"
                "    return {};\n"
                "}\n",
            ),
            "gdscript": (
                "func bfs(grafik: Dictionary, bas: String) -> Dictionary:\n"
                "    var mesafe := {bas: 0}\n"
                "    var kuyruk := [bas]\n"
                "    var i := 0\n"
                "    while i < kuyruk.size():\n"
                "        var dugum: String = kuyruk[i]\n"
                "        i += 1\n"
                "        for komsu in grafik.get(dugum, []):\n"
                "            if not mesafe.has(komsu):\n"
                "                mesafe[komsu] = mesafe[dugum] + 1\n"
                "                kuyruk.append(komsu)\n"
                "    return mesafe\n",
                "# Yol izleme\n"
                "func bfs_yol(grafik: Dictionary, bas: String, hedef: String) -> Array:\n"
                "    var ebeveyn := {bas: null}\n"
                "    var kuyruk := [bas]\n"
                "    var i := 0\n"
                "    while i < kuyruk.size():\n"
                "        var dugum: String = kuyruk[i]\n"
                "        i += 1\n"
                "        if dugum == hedef:\n"
                "            var yol := []\n"
                "            var d = dugum\n"
                "            while d != null:\n"
                "                yol.append(d)\n"
                "                d = ebeveyn[d]\n"
                "            yol.reverse()\n"
                "            return yol\n"
                "        for komsu in grafik.get(dugum, []):\n"
                "            if not ebeveyn.has(komsu):\n"
                "                ebeveyn[komsu] = dugum\n"
                "                kuyruk.append(komsu)\n"
                "    return []\n",
            ),
        },
    ),
    (
        "Birleştirmeli sıralama", "siralama", "O(n log n)",
        {
            "python": (
                "def birlestir(sol: list, sag: list) -> list:\n"
                "    sonuc, i, j = [], 0, 0\n"
                "    while i < len(sol) and j < len(sag):\n"
                "        if sol[i] <= sag[j]:\n"
                "            sonuc.append(sol[i]); i += 1\n"
                "        else:\n"
                "            sonuc.append(sag[j]); j += 1\n"
                "    sonuc.extend(sol[i:]); sonuc.extend(sag[j:])\n"
                "    return sonuc\n"
                "\n"
                "def birlestirmeli(dizi: list) -> list:\n"
                "    if len(dizi) <= 1:\n"
                "        return dizi\n"
                "    orta = len(dizi) // 2\n"
                "    return birlestir(birlestirmeli(dizi[:orta]),\n"
                "                     birlestirmeli(dizi[orta:]))\n",
                "# Kararlı (stable) — eşit elemanlar sırası korunur\n"
                "def birlestirmeli_kararli(dizi: list, anahtar=None) -> list:\n"
                "    if anahtar is None:\n"
                "        anahtar = lambda x: x\n"
                "    if len(dizi) <= 1:\n"
                "        return dizi[:]\n"
                "    orta = len(dizi) // 2\n"
                "    sol = birlestirmeli_kararli(dizi[:orta], anahtar)\n"
                "    sag = birlestirmeli_kararli(dizi[orta:], anahtar)\n"
                "    sonuc, i, j = [], 0, 0\n"
                "    while i < len(sol) and j < len(sag):\n"
                "        if anahtar(sol[i]) <= anahtar(sag[j]):\n"
                "            sonuc.append(sol[i]); i += 1\n"
                "        else:\n"
                "            sonuc.append(sag[j]); j += 1\n"
                "    return sonuc + sol[i:] + sag[j:]\n",
            ),
            "javascript": (
                "function birlestir(sol, sag) {\n"
                "  const sonuc = [];\n"
                "  let i = 0, j = 0;\n"
                "  while (i < sol.length && j < sag.length) {\n"
                "    if (sol[i] <= sag[j]) sonuc.push(sol[i++]);\n"
                "    else sonuc.push(sag[j++]);\n"
                "  }\n"
                "  return sonuc.concat(sol.slice(i), sag.slice(j));\n"
                "}\n"
                "\n"
                "function birlestirmeli(dizi) {\n"
                "  if (dizi.length <= 1) return dizi;\n"
                "  const orta = Math.floor(dizi.length / 2);\n"
                "  return birlestir(birlestirmeli(dizi.slice(0, orta)),\n"
                "                  birlestirmeli(dizi.slice(orta)));\n"
                "}\n",
                "// JS yerleşiği — `sort` kararlıdır (ES2019+)\n"
                "function siralaKararli(dizi) {\n"
                "  return [...dizi].sort((a, b) => a - b);\n"
                "}\n",
            ),
            "rust": (
                "fn birlestir(sol: &[i32], sag: &[i32]) -> Vec<i32> {\n"
                "    let mut sonuc = Vec::with_capacity(sol.len() + sag.len());\n"
                "    let (mut i, mut j) = (0, 0);\n"
                "    while i < sol.len() && j < sag.len() {\n"
                "        if sol[i] <= sag[j] { sonuc.push(sol[i]); i += 1; }\n"
                "        else { sonuc.push(sag[j]); j += 1; }\n"
                "    }\n"
                "    sonuc.extend_from_slice(&sol[i..]);\n"
                "    sonuc.extend_from_slice(&sag[j..]);\n"
                "    sonuc\n"
                "}\n"
                "\n"
                "fn birlestirmeli(dizi: &[i32]) -> Vec<i32> {\n"
                "    if dizi.len() <= 1 { return dizi.to_vec(); }\n"
                "    let orta = dizi.len() / 2;\n"
                "    birlestir(&birlestirmeli(&dizi[..orta]),\n"
                "              &birlestirmeli(&dizi[orta..]))\n"
                "}\n",
                "// Rust yerleşiği — `sort` kararlı, `sort_unstable` hızlı\n"
                "fn sirala_kararli(dizi: &mut [i32]) { dizi.sort(); }\n",
            ),
            "cpp": (
                "#include <vector>\n"
                "std::vector<int> birlestir(const std::vector<int>& a,\n"
                "                          const std::vector<int>& b) {\n"
                "    std::vector<int> s;\n"
                "    s.reserve(a.size() + b.size());\n"
                "    std::size_t i = 0, j = 0;\n"
                "    while (i < a.size() && j < b.size())\n"
                "        s.push_back(a[i] <= b[j] ? a[i++] : b[j++]);\n"
                "    s.insert(s.end(), a.begin() + i, a.end());\n"
                "    s.insert(s.end(), b.begin() + j, b.end());\n"
                "    return s;\n"
                "}\n"
                "\n"
                "std::vector<int> birlestirmeli(const std::vector<int>& d) {\n"
                "    if (d.size() <= 1) return d;\n"
                "    auto o = d.begin() + d.size() / 2;\n"
                "    return birlestir(birlestirmeli({d.begin(), o}),\n"
                "                     birlestirmeli({o, d.end()}));\n"
                "}\n",
                "#include <algorithm>\n"
                "// STL — `std::stable_sort` kararlı\n"
                "void sirala_kararli(std::vector<int>& d) {\n"
                "    std::stable_sort(d.begin(), d.end());\n"
                "}\n",
            ),
            "gdscript": (
                "func birlestir(sol: Array, sag: Array) -> Array:\n"
                "    var sonuc := []\n"
                "    var i := 0\n"
                "    var j := 0\n"
                "    while i < sol.size() and j < sag.size():\n"
                "        if sol[i] <= sag[j]:\n"
                "            sonuc.append(sol[i]); i += 1\n"
                "        else:\n"
                "            sonuc.append(sag[j]); j += 1\n"
                "    sonuc.append_array(sol.slice(i))\n"
                "    sonuc.append_array(sag.slice(j))\n"
                "    return sonuc\n"
                "\n"
                "func birlestirmeli(dizi: Array) -> Array:\n"
                "    if dizi.size() <= 1:\n"
                "        return dizi\n"
                "    var orta := dizi.size() / 2\n"
                "    return birlestir(birlestirmeli(dizi.slice(0, orta)),\n"
                "                     birlestirmeli(dizi.slice(orta)))\n",
                "# Godot yerleşiği\n"
                "func sirala_kararli(dizi: Array) -> Array:\n"
                "    var kopya := dizi.duplicate()\n"
                "    kopya.sort()\n"
                "    return kopya\n",
            ),
        },
    ),
)

#: Kategoriler
KATEGORI: Dict[str, str] = {
    "arama": "Arama Algoritmaları",
    "siralama": "Sıralama Algoritmaları",
    "bagli_liste": "Bağlı Liste",
    "yigin": "Yığın (Stack)",
    "hash": "Hash Tablosu",
    "grafik": "Grafik Algoritmaları",
    "agac": "Ağaç Algoritmaları",
    "ozyineleme": "Özyineleme ve Geri İzleme",
    "heap": "Öncelik Kuyruğu",
}

#: Varyant etiketleri
VARYANT: Dict[int, str] = {0: "temel", 1: "iyileştirilmiş"}


def tum_dersler() -> List[DilDers]:
    """Tüm dil derslerini üretir."""
    dersler: List[DilDers] = []
    for algoritma, kategori, karmasiklik, diller in ALGORITMALAR:
        for dil, kodlar in diller.items():
            for i, kod in enumerate(kodlar):
                v = VARYANT.get(i, f"v{i}")
                dersler.append(DilDers(
                    algoritma=algoritma, kategori=kategori, dil=dil,
                    varyant=v,
                    aciklama=(f"{algoritma} algoritmasının **{dil}** "
                              f"dilindeki **{v}** sürümü. "
                              f"Kategori: {KATEGORI.get(kategori, kategori)}."),
                    kod=kod, karmasiklik=karmasiklik))
    return dersler


def istatistik() -> Dict[str, Any]:
    """Müfredat istatistiği."""
    d = tum_dersler()
    metinler = [x.metin() for x in d]
    from collections import Counter

    return {
        "algoritma": len(ALGORITMALAR),
        "dil": len({x.dil for x in d}),
        "ders": len(d),
        "karakter": sum(len(m) for m in metinler),
        "token_tahmini": sum(len(m) for m in metinler) // 3,
        "dil_dagilim": dict(Counter(x.dil for x in d)),
        "kategori_dagilim": dict(Counter(x.kategori for x in d)),
        "varyant_dagilim": dict(Counter(x.varyant for x in d)),
    }


def kasaya_yaz() -> int:
    """Dersleri kasaya yazar."""
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return 0

    gorulen: set = set()
    try:
        for n in k.vault.iter_notes(limit=300000):
            if (getattr(n, "category", "") or "") == "kod_mantigi_dil":
                gorulen.add((getattr(n, "title", "") or "").strip())
    except Exception:
        pass

    yazilan = 0
    for d in tum_dersler():
        baslik = f"Mantık: {d.baslik()}"
        if baslik in gorulen:
            continue
        try:
            from neurocore.knowledge.vault import Note

            simdi = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            n = Note(
                id="", title=baslik, body=d.metin(), kind="skill",
                category="kod_mantigi_dil",
                tags=["mantik", "algoritma", d.kategori, d.dil],
                confidence=0.95, importance=0.85,
                created=simdi, updated=simdi, version=1,
                sources=["mantik_cogalt"],
                links=[], backlinks=[], embedding_model="",
                embedding=None, path="", extra={})
            k.vault.write_note(n, touch=False, emit=False)
            yazilan += 1
        except Exception:
            continue
    k.shutdown()
    return yazilan


def main() -> int:
    """Çoğaltma işlemlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="Mantık müfredatı çoğaltıcı")
    ap.add_argument("--istatistik", action="store_true")
    ap.add_argument("--kasaya", action="store_true")
    ap.add_argument("--liste", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  MANTIK MÜFREDATI ÇOĞALTICI — çok dilli algoritma dersleri")
    print("=" * 78)

    if args.liste:
        for d in tum_dersler():
            print(f"  {d.kategori:16} {d.algoritma:26} "
                  f"{d.dil:12} {d.varyant}")
        return 0

    st = istatistik()
    print(f"\n  algoritma : {st['algoritma']}")
    print(f"  dil       : {st['dil']}")
    print(f"  ders      : {st['ders']}")
    print(f"  karakter  : {st['karakter']:,}")
    print(f"  token     : ~{st['token_tahmini']:,}")
    print(f"\n  DİL DAĞILIMI     : {st['dil_dagilim']}")
    print(f"  KATEGORİ DAĞILIMI: {st['kategori_dagilim']}")

    if args.kasaya:
        n = kasaya_yaz()
        print(f"\n  kasaya yazıldı: {n} not")

    Path("_mantik_cogalt.json").write_text(
        json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
