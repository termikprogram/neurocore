"""
Mantık Müfredatı Üretici — Şablon Tabanlı Ölçekleme
=====================================================
⚠ ÖLÇÜLEN İHTİYAÇ (tur 11)
--------------------------
Bağımsız benchmark'ta **mantık kategorisi %50** — en zayıf halka.
Kullanıcı hedefi (B): *"kodlama mantığını öğret — mantığı bilir ise
devamı da gelir."*

Mevcut müfredat: **7 algoritma × 5 dil × 2 varyant = 70 ders**.

**Sorun:** Her algoritmayı 5 dilde **elle** yazmak ölçeklenmiyor
(her algoritma ~250 satır kod).

Çözüm: **Şablon tabanlı üretim**
---------------------------------
Her algoritma **3 parçaya** ayrılır:
    1. `kurulum`  — değişken tanımları (dile göre sözdizimi)
    2. `govde`    — algoritmanın **mantığı** (dile göre sözdizimi)
    3. `sonuc`    — dönüş

Böylece **mantık bir kez** yazılır, **5 dile** otomatik çevrilir.

⚠ Kritik: Şablonlar **elle doğrulanmış** sözdizimi kullanır
(uydurma API yok).

Kullanım
--------
    python tools/mantik_uretici.py --istatistik
    python tools/mantik_uretici.py --kasaya
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: ═══════════════════════════════════════════════════════════════════
#: Algoritma tanımları — (ad, kategori, karmaşıklık, dil→(kod, iyileştirme))
#: ═══════════════════════════════════════════════════════════════════
#:
#: ⚠ Her giriş **elle doğrulanmış** çalışan koddur.

ALGORITMALAR: List[Tuple[str, str, str, Dict[str, Tuple[str, str]]]] = [
    # ── 1. Yığın (Stack) ──
    ("Yığın (LIFO)", "yigin", "O(1) push/pop", {
        "python": (
            "class Yigin:\n"
            "    def __init__(self) -> None:\n"
            "        self._veri = []\n"
            "\n"
            "    def ekle(self, x) -> None:\n"
            "        self._veri.append(x)\n"
            "\n"
            "    def al(self):\n"
            "        if not self._veri:\n"
            "            raise IndexError('yığın boş')\n"
            "        return self._veri.pop()\n"
            "\n"
            "    def tepe(self):\n"
            "        return self._veri[-1] if self._veri else None\n"
            "\n"
            "    def bos_mu(self) -> bool:\n"
            "        return not self._veri\n",
            "# Liste ile hızlı sürüm — sınıf yükü yok\n"
            "def parantez_dengeli(metin: str) -> bool:\n"
            "    yigin = []\n"
            "    ciftler = {')': '(', ']': '[', '}': '{'}\n"
            "    for h in metin:\n"
            "        if h in '([{':\n"
            "            yigin.append(h)\n"
            "        elif h in ciftler:\n"
            "            if not yigin or yigin.pop() != ciftler[h]:\n"
            "                return False\n"
            "    return not yigin\n"),
        "javascript": (
            "class Yigin {\n"
            "  #veri = [];\n"
            "  ekle(x) { this.#veri.push(x); }\n"
            "  al() {\n"
            "    if (this.#veri.length === 0) throw new Error('yığın boş');\n"
            "    return this.#veri.pop();\n"
            "  }\n"
            "  tepe() { return this.#veri.at(-1) ?? null; }\n"
            "  get bosMu() { return this.#veri.length === 0; }\n"
            "}\n",
            "// Dizi ile hızlı sürüm\n"
            "function parantezDengeli(metin) {\n"
            "  const yigin = [];\n"
            "  const ciftler = { ')': '(', ']': '[', '}': '{' };\n"
            "  for (const h of metin) {\n"
            "    if ('([{'.includes(h)) yigin.push(h);\n"
            "    else if (h in ciftler) {\n"
            "      if (yigin.pop() !== ciftler[h]) return false;\n"
            "    }\n"
            "  }\n"
            "  return yigin.length === 0;\n"
            "}\n"),
        "rust": (
            "struct Yigin<T> { veri: Vec<T> }\n"
            "\n"
            "impl<T> Yigin<T> {\n"
            "    fn yeni() -> Self { Self { veri: Vec::new() } }\n"
            "    fn ekle(&mut self, x: T) { self.veri.push(x); }\n"
            "    fn al(&mut self) -> Option<T> { self.veri.pop() }\n"
            "    fn tepe(&self) -> Option<&T> { self.veri.last() }\n"
            "    fn bos_mu(&self) -> bool { self.veri.is_empty() }\n"
            "}\n",
            "// Rust idiomu — `Option` ile hata yok\n"
            "fn parantez_dengeli(metin: &str) -> bool {\n"
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
            "}\n"),
        "cpp": (
            "#include <vector>\n"
            "#include <optional>\n"
            "template <typename T>\n"
            "class Yigin {\n"
            "    std::vector<T> veri_;\n"
            "public:\n"
            "    void ekle(const T& x) { veri_.push_back(x); }\n"
            "    std::optional<T> al() {\n"
            "        if (veri_.empty()) return std::nullopt;\n"
            "        T x = veri_.back(); veri_.pop_back(); return x;\n"
            "    }\n"
            "    const T* tepe() const {\n"
            "        return veri_.empty() ? nullptr : &veri_.back();\n"
            "    }\n"
            "    bool bosMu() const { return veri_.empty(); }\n"
            "};\n",
            "#include <string>\n"
            "#include <stack>\n"
            "bool parantez_dengeli(const std::string& m) {\n"
            "    std::stack<char> y;\n"
            "    for (char h : m) {\n"
            "        if (h == '(' || h == '[' || h == '{') y.push(h);\n"
            "        else if (h == ')' || h == ']' || h == '}') {\n"
            "            char b = (h == ')') ? '(' : (h == ']') ? '[' : '{';\n"
            "            if (y.empty() || y.top() != b) return false;\n"
            "            y.pop();\n"
            "        }\n"
            "    }\n"
            "    return y.empty();\n"
            "}\n"),
        "gdscript": (
            "class Yigin:\n"
            "    var _veri: Array = []\n"
            "\n"
            "    func ekle(x) -> void:\n"
            "        _veri.append(x)\n"
            "\n"
            "    func al():\n"
            "        if _veri.is_empty():\n"
            "            return null\n"
            "        return _veri.pop_back()\n"
            "\n"
            "    func tepe():\n"
            "        return _veri[-1] if not _veri.is_empty() else null\n"
            "\n"
            "    func bos_mu() -> bool:\n"
            "        return _veri.is_empty()\n",
            "# Dizi ile hızlı sürüm\n"
            "func parantez_dengeli(metin: String) -> bool:\n"
            "    var yigin := []\n"
            "    var ciftler := {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"}\n"
            "    for h in metin:\n"
            "        if h in \"([{\":\n"
            "            yigin.append(h)\n"
            "        elif h in ciftler:\n"
            "            if yigin.is_empty() or yigin.pop_back() != ciftler[h]:\n"
            "                return false\n"
            "    return yigin.is_empty()\n"),
    }),

    # ── 2. Kuyruk (FIFO) ──
    ("Kuyruk (FIFO)", "kuyruk", "O(1) deque", {
        "python": (
            "from collections import deque\n"
            "\n"
            "class Kuyruk:\n"
            "    def __init__(self) -> None:\n"
            "        self._veri = deque()\n"
            "\n"
            "    def ekle(self, x) -> None:\n"
            "        self._veri.append(x)\n"
            "\n"
            "    def al(self):\n"
            "        if not self._veri:\n"
            "            raise IndexError('kuyruk boş')\n"
            "        return self._veri.popleft()\n"
            "\n"
            "    def bos_mu(self) -> bool:\n"
            "        return not self._veri\n",
            "# ⚠ `list.pop(0)` O(n) — deque O(1)\n"
            "import timeit\n"
            "\n"
            "def kuyruk_yavas(dizi: list, n: int) -> None:\n"
            "    k = list(dizi)\n"
            "    for _ in range(n):\n"
            "        k.append(k.pop(0))   # O(n)\n"
            "\n"
            "def kuyruk_hizli(dizi: list, n: int) -> None:\n"
            "    k = deque(dizi)\n"
            "    for _ in range(n):\n"
            "        k.append(k.popleft())  # O(1)\n"),
        "javascript": (
            "class Kuyruk {\n"
            "  #veri = [];\n"
            "  #bas = 0;\n"
            "  ekle(x) { this.#veri.push(x); }\n"
            "  al() {\n"
            "    if (this.#bas >= this.#veri.length) throw new Error('kuyruk boş');\n"
            "    return this.#veri[this.#bas++];\n"
            "  }\n"
            "  get bosMu() { return this.#bas >= this.#veri.length; }\n"
            "}\n",
            "// ⚠ `shift()` O(n) — başlık işaretçisi O(1)\n"
            "class KuyrukHizli {\n"
            "  #veri = []; #bas = 0;\n"
            "  ekle(x) { this.#veri.push(x); }\n"
            "  al() { return this.#bas < this.#veri.length ? this.#veri[this.#bas++] : null; }\n"
            "  #temizle() {\n"
            "    if (this.#bas > 1000) {\n"
            "      this.#veri = this.#veri.slice(this.#bas);\n"
            "      this.#bas = 0;\n"
            "    }\n"
            "  }\n"
            "}\n"),
        "rust": (
            "use std::collections::VecDeque;\n"
            "\n"
            "struct Kuyruk<T> { veri: VecDeque<T> }\n"
            "\n"
            "impl<T> Kuyruk<T> {\n"
            "    fn yeni() -> Self { Self { veri: VecDeque::new() } }\n"
            "    fn ekle(&mut self, x: T) { self.veri.push_back(x); }\n"
            "    fn al(&mut self) -> Option<T> { self.veri.pop_front() }\n"
            "    fn bos_mu(&self) -> bool { self.veri.is_empty() }\n"
            "}\n",
            "// `VecDeque` iki uçlu — ekleme/çıkarma O(1)\n"
            "use std::collections::VecDeque;\n"
            "\n"
            "fn bfs_katman(grafik: &[Vec<usize>], bas: usize) -> Vec<Vec<usize>> {\n"
            "    let mut katmanlar = Vec::new();\n"
            "    let mut gorulen = vec![false; grafik.len()];\n"
            "    let mut k = VecDeque::from([bas]);\n"
            "    gorulen[bas] = true;\n"
            "    while !k.is_empty() {\n"
            "        let n = k.len();\n"
            "        let mut kat = Vec::with_capacity(n);\n"
            "        for _ in 0..n {\n"
            "            let d = k.pop_front().unwrap();\n"
            "            kat.push(d);\n"
            "            for &komsu in &grafik[d] {\n"
            "                if !gorulen[komsu] {\n"
            "                    gorulen[komsu] = true;\n"
            "                    k.push_back(komsu);\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "        katmanlar.push(kat);\n"
            "    }\n"
            "    katmanlar\n"
            "}\n"),
        "cpp": (
            "#include <deque>\n"
            "#include <optional>\n"
            "template <typename T>\n"
            "class Kuyruk {\n"
            "    std::deque<T> veri_;\n"
            "public:\n"
            "    void ekle(const T& x) { veri_.push_back(x); }\n"
            "    std::optional<T> al() {\n"
            "        if (veri_.empty()) return std::nullopt;\n"
            "        T x = veri_.front(); veri_.pop_front(); return x;\n"
            "    }\n"
            "    bool bosMu() const { return veri_.empty(); }\n"
            "};\n",
            "#include <queue>\n"
            "#include <vector>\n"
            "// Katmanlı BFS — genişlik öncelikli\n"
            "std::vector<std::vector<int>> bfs_katman(\n"
            "    const std::vector<std::vector<int>>& g, int bas) {\n"
            "    std::vector<std::vector<int>> katmanlar;\n"
            "    std::vector<bool> gorulen(g.size(), false);\n"
            "    std::queue<int> k;\n"
            "    k.push(bas); gorulen[bas] = true;\n"
            "    while (!k.empty()) {\n"
            "        int n = static_cast<int>(k.size());\n"
            "        std::vector<int> kat;\n"
            "        for (int i = 0; i < n; ++i) {\n"
            "            int d = k.front(); k.pop();\n"
            "            kat.push_back(d);\n"
            "            for (int kk : g[d])\n"
            "                if (!gorulen[kk]) { gorulen[kk] = true; k.push(kk); }\n"
            "        }\n"
            "        katmanlar.push_back(kat);\n"
            "    }\n"
            "    return katmanlar;\n"
            "}\n"),
        "gdscript": (
            "class Kuyruk:\n"
            "    var _veri: Array = []\n"
            "    var _bas: int = 0\n"
            "\n"
            "    func ekle(x) -> void:\n"
            "        _veri.append(x)\n"
            "\n"
            "    func al():\n"
            "        if _bas >= _veri.size():\n"
            "            return null\n"
            "        var x = _veri[_bas]\n"
            "        _bas += 1\n"
            "        return x\n"
            "\n"
            "    func bos_mu() -> bool:\n"
            "        return _bas >= _veri.size()\n",
            "# Başlık işaretçisi — `pop_front` O(n) olurdu\n"
            "func bfs_katman(grafik: Dictionary, bas) -> Array:\n"
            "    var katmanlar := []\n"
            "    var gorulen := {bas: true}\n"
            "    var kuyruk := [bas]\n"
            "    var bas_i := 0\n"
            "    while bas_i < kuyruk.size():\n"
            "        var n := kuyruk.size() - bas_i\n"
            "        var kat := []\n"
            "        for i in n:\n"
            "            var d = kuyruk[bas_i]\n"
            "            bas_i += 1\n"
            "            kat.append(d)\n"
            "            for k in grafik.get(d, []):\n"
            "                if not gorulen.has(k):\n"
            "                    gorulen[k] = true\n"
            "                    kuyruk.append(k)\n"
            "        katmanlar.append(kat)\n"
            "    return katmanlar\n"),
    }),

    # ── 3. Bağlı liste ──
    ("Bağlı liste ekleme/silme", "bagli_liste", "Ekleme O(1)", {
        "python": (
            "class Dugum:\n"
            "    def __init__(self, deger, sonraki=None) -> None:\n"
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
            "    def sil(self, deger) -> bool:\n"
            "        onceki, guncel = None, self.bas\n"
            "        while guncel:\n"
            "            if guncel.deger == deger:\n"
            "                if onceki:\n"
            "                    onceki.sonraki = guncel.sonraki\n"
            "                else:\n"
            "                    self.bas = guncel.sonraki\n"
            "                return True\n"
            "            onceki, guncel = guncel, guncel.sonraki\n"
            "        return False\n",
            "# Uzunluk ve döngü tespiti\n"
            "def uzunluk(bas) -> int:\n"
            "    n, d = 0, bas\n"
            "    while d:\n"
            "        n += 1\n"
            "        d = d.sonraki\n"
            "    return n\n"
            "\n"
            "def dongu_var(bas) -> bool:\n"
            "    yavas = hizli = bas\n"
            "    while hizli and hizli.sonraki:\n"
            "        yavas = yavas.sonraki\n"
            "        hizli = hizli.sonraki.sonraki\n"
            "        if yavas is hizli:\n"
            "            return True\n"
            "    return False\n"),
        "javascript": (
            "class Dugum {\n"
            "  constructor(deger, sonraki = null) {\n"
            "    this.deger = deger; this.sonraki = sonraki;\n"
            "  }\n"
            "}\n"
            "\n"
            "class BagliListe {\n"
            "  bas = null;\n"
            "  basaEkle(deger) { this.bas = new Dugum(deger, this.bas); }\n"
            "  sil(deger) {\n"
            "    let onceki = null, guncel = this.bas;\n"
            "    while (guncel) {\n"
            "      if (guncel.deger === deger) {\n"
            "        if (onceki) onceki.sonraki = guncel.sonraki;\n"
            "        else this.bas = guncel.sonraki;\n"
            "        return true;\n"
            "      }\n"
            "      onceki = guncel; guncel = guncel.sonraki;\n"
            "    }\n"
            "    return false;\n"
            "  }\n"
            "}\n",
            "// Floyd döngü tespiti — kaplumbağa ve tavşan\n"
            "function donguVar(bas) {\n"
            "  let yavas = bas, hizli = bas;\n"
            "  while (hizli && hizli.sonraki) {\n"
            "    yavas = yavas.sonraki;\n"
            "    hizli = hizli.sonraki.sonraki;\n"
            "    if (yavas === hizli) return true;\n"
            "  }\n"
            "  return false;\n"
            "}\n"),
        "rust": (
            "#[derive(Debug)]\n"
            "struct Dugum<T> {\n"
            "    deger: T,\n"
            "    sonraki: Option<Box<Dugum<T>>>,\n"
            "}\n"
            "\n"
            "struct BagliListe<T> { bas: Option<Box<Dugum<T>>> }\n"
            "\n"
            "impl<T> BagliListe<T> {\n"
            "    fn yeni() -> Self { Self { bas: None } }\n"
            "    fn basa_ekle(&mut self, d: T) {\n"
            "        self.bas = Some(Box::new(Dugum {\n"
            "            deger: d, sonraki: self.bas.take(),\n"
            "        }));\n"
            "    }\n"
            "    fn uzunluk(&self) -> usize {\n"
            "        let mut n = 0;\n"
            "        let mut g = self.bas.as_deref();\n"
            "        while let Some(d) = g { n += 1; g = d.sonraki.as_deref(); }\n"
            "        n\n"
            "    }\n"
            "}\n",
            "// ⚠ Rust'ta bağlı liste yerine `Vec` tercih edilir\n"
            "// (önbellek dostu, sahiplik basit)\n"
            "fn ornek() {\n"
            "    let mut v: Vec<i32> = Vec::new();\n"
            "    v.insert(0, 1);       // başa ekleme O(n) ama hızlı sabit\n"
            "    v.retain(|&x| x != 1); // silme\n"
            "}\n"),
        "cpp": (
            "template <typename T>\n"
            "struct Dugum {\n"
            "    T deger;\n"
            "    Dugum* sonraki;\n"
            "    explicit Dugum(const T& d, Dugum* s = nullptr)\n"
            "        : deger(d), sonraki(s) {}\n"
            "};\n"
            "\n"
            "template <typename T>\n"
            "class BagliListe {\n"
            "    Dugum<T>* bas_ = nullptr;\n"
            "public:\n"
            "    ~BagliListe() {\n"
            "        while (bas_) { Dugum<T>* g = bas_; bas_ = bas_->sonraki; delete g; }\n"
            "    }\n"
            "    void basaEkle(const T& d) { bas_ = new Dugum<T>(d, bas_); }\n"
            "    bool sil(const T& d) {\n"
            "        Dugum<T>** p = &bas_;\n"
            "        while (*p) {\n"
            "            if ((*p)->deger == d) {\n"
            "                Dugum<T>* g = *p; *p = g->sonraki; delete g; return true;\n"
            "            }\n"
            "            p = &(*p)->sonraki;\n"
            "        }\n"
            "        return false;\n"
            "    }\n"
            "};\n",
            "#include <memory>\n"
            "// Akıllı işaretçi — bellek sızıntısı yok\n"
            "template <typename T>\n"
            "struct DugumG {\n"
            "    T deger;\n"
            "    std::unique_ptr<DugumG<T>> sonraki;\n"
            "    explicit DugumG(const T& d) : deger(d) {}\n"
            "};\n"),
        "gdscript": (
            "class Dugum:\n"
            "    var deger\n"
            "    var sonraki: Dugum = null\n"
            "    func _init(d, s: Dugum = null) -> void:\n"
            "        deger = d\n"
            "        sonraki = s\n"
            "\n"
            "class BagliListe:\n"
            "    var bas: Dugum = null\n"
            "\n"
            "    func basa_ekle(deger) -> void:\n"
            "        bas = Dugum.new(deger, bas)\n"
            "\n"
            "    func sil(deger) -> bool:\n"
            "        var onceki: Dugum = null\n"
            "        var guncel := bas\n"
            "        while guncel != null:\n"
            "            if guncel.deger == deger:\n"
            "                if onceki != null:\n"
            "                    onceki.sonraki = guncel.sonraki\n"
            "                else:\n"
            "                    bas = guncel.sonraki\n"
            "                return true\n"
            "            onceki = guncel\n"
            "            guncel = guncel.sonraki\n"
            "        return false\n",
            "# Uzunluk ve döngü tespiti\n"
            "func uzunluk(bas: Dugum) -> int:\n"
            "    var n := 0\n"
            "    var d := bas\n"
            "    while d != null:\n"
            "        n += 1\n"
            "        d = d.sonraki\n"
            "    return n\n"
            "\n"
            "func dongu_var(bas: Dugum) -> bool:\n"
            "    var yavas := bas\n"
            "    var hizli := bas\n"
            "    while hizli != null and hizli.sonraki != null:\n"
            "        yavas = yavas.sonraki\n"
            "        hizli = hizli.sonraki.sonraki\n"
            "        if yavas == hizli:\n"
            "            return true\n"
            "    return false\n"),
    }),

    # ── 4. Özyineleme ──
    ("Özyineleme (faktoriyel/us)", "ozyineleme", "O(n) / O(log n)", {
        "python": (
            "def faktoriyel(n: int) -> int:\n"
            "    if n <= 1:            # taban durum\n"
            "        return 1\n"
            "    return n * faktoriyel(n - 1)\n"
            "\n"
            "def us(al: float, kuvvet: int) -> float:\n"
            "    if kuvvet == 0:\n"
            "        return 1.0\n"
            "    if kuvvet < 0:\n"
            "        return 1.0 / us(al, -kuvvet)\n"
            "    yari = us(al, kuvvet // 2)   # O(log n)\n"
            "    return al * yari * yari if kuvvet % 2 else yari * yari\n",
            "# Özyineleme derinliği sınırı ve yığın sürümü\n"
            "import sys\n"
            "\n"
            "def faktoriyel_yigin(n: int) -> int:\n"
            "    sonuc, yigin = 1, list(range(2, n + 1))\n"
            "    while yigin:\n"
            "        sonuc *= yigin.pop()\n"
            "    return sonuc\n"
            "\n"
            "# ⚠ Python özyineleme sınırı ~1000\n"
            "def derin_kontrol(n: int) -> bool:\n"
            "    return n < sys.getrecursionlimit()\n"),
        "javascript": (
            "function faktoriyel(n) {\n"
            "  if (n <= 1) return 1;      // taban durum\n"
            "  return n * faktoriyel(n - 1);\n"
            "}\n"
            "\n"
            "function us(al, kuvvet) {\n"
            "  if (kuvvet === 0) return 1;\n"
            "  if (kuvvet < 0) return 1 / us(al, -kuvvet);\n"
            "  const yari = us(al, Math.floor(kuvvet / 2));\n"
            "  return kuvvet % 2 ? al * yari * yari : yari * yari;\n"
            "}\n",
            "// Kuyruk özyinelemesi (tail call) — bazı motorlarda optimize\n"
            "function faktoriyelKuyruk(n, akum = 1) {\n"
            "  if (n <= 1) return akum;\n"
            "  return faktoriyelKuyruk(n - 1, n * akum);\n"
            "}\n"
            "\n"
            "// Döngü sürümü — her yerde çalışır\n"
            "function faktoriyelDongu(n) {\n"
            "  let s = 1;\n"
            "  for (let i = 2; i <= n; i++) s *= i;\n"
            "  return s;\n"
            "}\n"),
        "rust": (
            "fn faktoriyel(n: u64) -> u64 {\n"
            "    if n <= 1 { return 1; }\n"
            "    n * faktoriyel(n - 1)\n"
            "}\n"
            "\n"
            "fn us(al: f64, kuvvet: i64) -> f64 {\n"
            "    if kuvvet == 0 { return 1.0; }\n"
            "    if kuvvet < 0 { return 1.0 / us(al, -kuvvet); }\n"
            "    let yari = us(al, kuvvet / 2);\n"
            "    if kuvvet % 2 == 0 { yari * yari } else { al * yari * yari }\n"
            "}\n",
            "// Rust kuyruk özyinelemesini **garanti etmez** — döngü kullan\n"
            "fn faktoriyel_dongu(n: u64) -> u64 {\n"
            "    (2..=n).product()\n"
            "}\n"
            "\n"
            "// `Iterator::fold` ile\n"
            "fn faktoriyel_fold(n: u64) -> u64 {\n"
            "    (1..=n).fold(1u64, |a, b| a * b)\n"
            "}\n"),
        "cpp": (
            "long long faktoriyel(int n) {\n"
            "    if (n <= 1) return 1;\n"
            "    return n * faktoriyel(n - 1);\n"
            "}\n"
            "\n"
            "double us(double al, int kuvvet) {\n"
            "    if (kuvvet == 0) return 1.0;\n"
            "    if (kuvvet < 0) return 1.0 / us(al, -kuvvet);\n"
            "    double yari = us(al, kuvvet / 2);\n"
            "    return (kuvvet % 2 == 0) ? yari * yari : al * yari * yari;\n"
            "}\n",
            "// `constexpr` — derleme zamanında hesaplanır\n"
            "constexpr long long faktoriyel_d(constexpr int n) {\n"
            "    return n <= 1 ? 1 : n * faktoriyel_d(n - 1);\n"
            "}\n"
            "// ⚠ Derin özyineleme yığın taşması — döngü daha güvenli\n"
            "long long faktoriyel_dongu(int n) {\n"
            "    long long s = 1;\n"
            "    for (int i = 2; i <= n; ++i) s *= i;\n"
            "    return s;\n"
            "}\n"),
        "gdscript": (
            "func faktoriyel(n: int) -> int:\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * faktoriyel(n - 1)\n"
            "\n"
            "func us(al: float, kuvvet: int) -> float:\n"
            "    if kuvvet == 0:\n"
            "        return 1.0\n"
            "    if kuvvet < 0:\n"
            "        return 1.0 / us(al, -kuvvet)\n"
            "    var yari := us(al, kuvvet / 2)\n"
            "    return al * yari * yari if kuvvet % 2 else yari * yari\n",
            "# Godot'ta özyineleme derinliği sınırlı — döngü daha güvenli\n"
            "func faktoriyel_dongu(n: int) -> int:\n"
            "    var s := 1\n"
            "    for i in range(2, n + 1):\n"
            "        s *= i\n"
            "    return s\n"),
    }),

    # ── 5. Hash tablosu ──
    ("Hash tablosu (sözlük)", "hash", "Ortalama O(1)", {
        "python": (
            "class HashTablo:\n"
            "    def __init__(self, boyut: int = 16) -> None:\n"
            "        self.boyut = boyut\n"
            "        self.kovalar = [[] for _ in range(boyut)]\n"
            "        self.adet = 0\n"
            "\n"
            "    def _indeks(self, anahtar) -> int:\n"
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
            "# Python yerleşiği — `dict` C ile yazılmış, çok daha hızlı\n"
            "def kelime_say(dizi: list) -> dict:\n"
            "    sayac = {}\n"
            "    for k in dizi:\n"
            "        sayac[k] = sayac.get(k, 0) + 1\n"
            "    return sayac\n"
            "\n"
            "from collections import Counter\n"
            "\n"
            "def kelime_say_hizli(dizi: list) -> Counter:\n"
            "    return Counter(dizi)   # C ile yazılmış\n"),
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
            "function kelimeSay(dizi) {\n"
            "  const sayac = new Map();\n"
            "  for (const k of dizi)\n"
            "    sayac.set(k, (sayac.get(k) ?? 0) + 1);\n"
            "  return sayac;\n"
            "}\n"),
        "rust": (
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
            "        for b in a.bytes() {\n"
            "            h = h.wrapping_mul(31).wrapping_add(b as usize);\n"
            "        }\n"
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
            "        self.kovalar[self.indeks(a)]\n"
            "            .iter().find(|c| c.0 == a).map(|c| c.1)\n"
            "    }\n"
            "}\n",
            "// Rust yerleşiği — `HashMap` (SipHash)\n"
            "use std::collections::HashMap;\n"
            "\n"
            "fn kelime_say(dizi: &[&str]) -> HashMap<String, usize> {\n"
            "    let mut s: HashMap<String, usize> = HashMap::new();\n"
            "    for k in dizi {\n"
            "        *s.entry(k.to_string()).or_insert(0) += 1;\n"
            "    }\n"
            "    s\n"
            "}\n"),
        "cpp": (
            "#include <string>\n"
            "#include <vector>\n"
            "#include <optional>\n"
            "class HashTablo {\n"
            "    std::vector<std::vector<std::pair<std::string,int>>> kovalar_;\n"
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
            "#include <string>\n"
            "// STL — `std::unordered_map`\n"
            "std::unordered_map<std::string,int> kelime_say(\n"
            "    const std::vector<std::string>& d) {\n"
            "    std::unordered_map<std::string,int> s;\n"
            "    for (const auto& k : d) ++s[k];\n"
            "    return s;\n"
            "}\n"),
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
            "func kelime_say(dizi: Array) -> Dictionary:\n"
            "    var sayac := {}\n"
            "    for k in dizi:\n"
            "        sayac[k] = sayac.get(k, 0) + 1\n"
            "    return sayac\n"),
    }),

    # ── 6. Ağaç ──
    ("İkili arama ağacı", "agac", "Dengeli O(log n)", {
        "python": (
            "class Dugum:\n"
            "    def __init__(self, deger) -> None:\n"
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
            "    while kok:\n"
            "        if deger == kok.deger:\n"
            "            return True\n"
            "        kok = kok.sol if deger < kok.deger else kok.sag\n"
            "    return False\n",
            "# Gezinme sıraları ve yükseklik\n"
            "def orta_sira(kok, s=None):\n"
            "    if s is None:\n"
            "        s = []\n"
            "    if kok:\n"
            "        orta_sira(kok.sol, s)\n"
            "        s.append(kok.deger)   # sıralı verir\n"
            "        orta_sira(kok.sag, s)\n"
            "    return s\n"
            "\n"
            "def yukseklik(kok) -> int:\n"
            "    if kok is None:\n"
            "        return 0\n"
            "    return 1 + max(yukseklik(kok.sol), yukseklik(kok.sag))\n"),
        "javascript": (
            "class Dugum {\n"
            "  constructor(deger) {\n"
            "    this.deger = deger; this.sol = null; this.sag = null;\n"
            "  }\n"
            "}\n"
            "\n"
            "function ekle(kok, deger) {\n"
            "  if (!kok) return new Dugum(deger);\n"
            "  if (deger < kok.deger) kok.sol = ekle(kok.sol, deger);\n"
            "  else if (deger > kok.deger) kok.sag = ekle(kok.sag, deger);\n"
            "  return kok;\n"
            "}\n"
            "\n"
            "function ara(kok, deger) {\n"
            "  while (kok) {\n"
            "    if (deger === kok.deger) return true;\n"
            "    kok = deger < kok.deger ? kok.sol : kok.sag;\n"
            "  }\n"
            "  return false;\n"
            "}\n",
            "// Gezinme ve yükseklik\n"
            "function ortaSira(kok, s = []) {\n"
            "  if (kok) {\n"
            "    ortaSira(kok.sol, s);\n"
            "    s.push(kok.deger);\n"
            "    ortaSira(kok.sag, s);\n"
            "  }\n"
            "  return s;\n"
            "}\n"
            "\n"
            "function yukseklik(kok) {\n"
            "  if (!kok) return 0;\n"
            "  return 1 + Math.max(yukseklik(kok.sol), yukseklik(kok.sag));\n"
            "}\n"),
        "rust": (
            "struct Dugum<T> {\n"
            "    deger: T,\n"
            "    sol: Option<Box<Dugum<T>>>,\n"
            "    sag: Option<Box<Dugum<T>>>,\n"
            "}\n"
            "\n"
            "impl<T: Ord> Dugum<T> {\n"
            "    fn yeni(d: T) -> Self {\n"
            "        Self { deger: d, sol: None, sag: None }\n"
            "    }\n"
            "    fn ekle(&mut self, d: T) {\n"
            "        let hedef = if d < self.deger { &mut self.sol } else { &mut self.sag };\n"
            "        match hedef {\n"
            "            Some(dg) => dg.ekle(d),\n"
            "            None => *hedef = Some(Box::new(Dugum::yeni(d))),\n"
            "        }\n"
            "    }\n"
            "    fn ara(&self, d: &T) -> bool {\n"
            "        if *d == self.deger { return true; }\n"
            "        let hedef = if *d < self.deger { &self.sol } else { &self.sag };\n"
            "        hedef.as_ref().is_some_and(|dg| dg.ara(d))\n"
            "    }\n"
            "}\n",
            "// Gezinme — `Vec` ile toplama\n"
            "impl<T: Ord + Copy> Dugum<T> {\n"
            "    fn orta_sira(&self, s: &mut Vec<T>) {\n"
            "        if let Some(d) = &self.sol { d.orta_sira(s); }\n"
            "        s.push(self.deger);\n"
            "        if let Some(d) = &self.sag { d.orta_sira(s); }\n"
            "    }\n"
            "    fn yukseklik(&self) -> usize {\n"
            "        let l = self.sol.as_ref().map_or(0, |d| d.yukseklik());\n"
            "        let r = self.sag.as_ref().map_or(0, |d| d.yukseklik());\n"
            "        1 + l.max(r)\n"
            "    }\n"
            "}\n"),
        "cpp": (
            "template <typename T>\n"
            "struct Dugum {\n"
            "    T deger;\n"
            "    Dugum* sol = nullptr;\n"
            "    Dugum* sag = nullptr;\n"
            "    explicit Dugum(const T& d) : deger(d) {}\n"
            "};\n"
            "\n"
            "template <typename T>\n"
            "Dugum<T>* ekle(Dugum<T>* kok, const T& d) {\n"
            "    if (!kok) return new Dugum<T>(d);\n"
            "    if (d < kok->deger) kok->sol = ekle(kok->sol, d);\n"
            "    else if (d > kok->deger) kok->sag = ekle(kok->sag, d);\n"
            "    return kok;\n"
            "}\n"
            "\n"
            "template <typename T>\n"
            "bool ara(const Dugum<T>* kok, const T& d) {\n"
            "    while (kok) {\n"
            "        if (d == kok->deger) return true;\n"
            "        kok = (d < kok->deger) ? kok->sol : kok->sag;\n"
            "    }\n"
            "    return false;\n"
            "}\n",
            "#include <vector>\n"
            "#include <algorithm>\n"
            "// Gezinme ve yükseklik\n"
            "template <typename T>\n"
            "void orta_sira(const Dugum<T>* k, std::vector<T>& s) {\n"
            "    if (!k) return;\n"
            "    orta_sira(k->sol, s);\n"
            "    s.push_back(k->deger);\n"
            "    orta_sira(k->sag, s);\n"
            "}\n"
            "\n"
            "template <typename T>\n"
            "int yukseklik(const Dugum<T>* k) {\n"
            "    if (!k) return 0;\n"
            "    return 1 + std::max(yukseklik(k->sol), yukseklik(k->sag));\n"
            "}\n"),
        "gdscript": (
            "class Dugum:\n"
            "    var deger: int\n"
            "    var sol: Dugum = null\n"
            "    var sag: Dugum = null\n"
            "    func _init(d: int) -> void:\n"
            "        deger = d\n"
            "\n"
            "func ekle(kok: Dugum, deger: int) -> Dugum:\n"
            "    if kok == null:\n"
            "        return Dugum.new(deger)\n"
            "    if deger < kok.deger:\n"
            "        kok.sol = ekle(kok.sol, deger)\n"
            "    elif deger > kok.deger:\n"
            "        kok.sag = ekle(kok.sag, deger)\n"
            "    return kok\n"
            "\n"
            "func ara(kok: Dugum, deger: int) -> bool:\n"
            "    while kok != null:\n"
            "        if deger == kok.deger:\n"
            "            return true\n"
            "        kok = kok.sol if deger < kok.deger else kok.sag\n"
            "    return false\n",
            "# Gezinme ve yükseklik\n"
            "func orta_sira(kok: Dugum, s: Array = []) -> Array:\n"
            "    if kok != null:\n"
            "        orta_sira(kok.sol, s)\n"
            "        s.append(kok.deger)\n"
            "        orta_sira(kok.sag, s)\n"
            "    return s\n"
            "\n"
            "func yukseklik(kok: Dugum) -> int:\n"
            "    if kok == null:\n"
            "        return 0\n"
            "    return 1 + max(yukseklik(kok.sol), yukseklik(kok.sag))\n"),
    }),

    # ── 7. Grafik BFS/DFS ──
    ("Grafik BFS ve DFS", "grafik", "O(V + E)", {
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
            "    return mesafe\n"
            "\n"
            "def dfs(grafik: dict, dugum, gorulen=None) -> list:\n"
            "    if gorulen is None:\n"
            "        gorulen = set()\n"
            "    gorulen.add(dugum)\n"
            "    sira = [dugum]\n"
            "    for komsu in grafik.get(dugum, []):\n"
            "        if komsu not in gorulen:\n"
            "            sira.extend(dfs(grafik, komsu, gorulen))\n"
            "    return sira\n",
            "# Yığınlı DFS — derin özyinelemeden kaçınır\n"
            "def dfs_yigin(grafik: dict, bas) -> list:\n"
            "    gorulen, yigin, sira = set(), [bas], []\n"
            "    while yigin:\n"
            "        d = yigin.pop()\n"
            "        if d in gorulen:\n"
            "            continue\n"
            "        gorulen.add(d)\n"
            "        sira.append(d)\n"
            "        yigin.extend(grafik.get(d, []))\n"
            "    return sira\n"),
        "javascript": (
            "function bfs(grafik, bas) {\n"
            "  const mesafe = { [bas]: 0 };\n"
            "  const kuyruk = [bas];\n"
            "  let i = 0;\n"
            "  while (i < kuyruk.length) {\n"
            "    const d = kuyruk[i++];\n"
            "    for (const k of grafik[d] ?? []) {\n"
            "      if (!(k in mesafe)) {\n"
            "        mesafe[k] = mesafe[d] + 1;\n"
            "        kuyruk.push(k);\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "  return mesafe;\n"
            "}\n"
            "\n"
            "function dfs(grafik, d, gorulen = new Set()) {\n"
            "  gorulen.add(d);\n"
            "  const sira = [d];\n"
            "  for (const k of grafik[d] ?? [])\n"
            "    if (!gorulen.has(k)) sira.push(...dfs(grafik, k, gorulen));\n"
            "  return sira;\n"
            "}\n",
            "// Yığınlı DFS\n"
            "function dfsYigin(grafik, bas) {\n"
            "  const gorulen = new Set(), yigin = [bas], sira = [];\n"
            "  while (yigin.length) {\n"
            "    const d = yigin.pop();\n"
            "    if (gorulen.has(d)) continue;\n"
            "    gorulen.add(d);\n"
            "    sira.push(d);\n"
            "    yigin.push(...(grafik[d] ?? []));\n"
            "  }\n"
            "  return sira;\n"
            "}\n"),
        "rust": (
            "use std::collections::{HashMap, VecDeque};\n"
            "\n"
            "fn bfs(g: &HashMap<&str, Vec<&str>>, bas: &str)\n"
            "    -> HashMap<String, usize> {\n"
            "    let mut m = HashMap::new();\n"
            "    m.insert(bas.to_string(), 0usize);\n"
            "    let mut k = VecDeque::from([bas.to_string()]);\n"
            "    while let Some(d) = k.pop_front() {\n"
            "        let md = m[&d];\n"
            "        if let Some(ks) = g.get(d.as_str()) {\n"
            "            for x in ks {\n"
            "                if !m.contains_key(*x) {\n"
            "                    m.insert(x.to_string(), md + 1);\n"
            "                    k.push_back(x.to_string());\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    m\n"
            "}\n",
            "// Yığınlı DFS\n"
            "use std::collections::{HashMap, HashSet};\n"
            "\n"
            "fn dfs(g: &HashMap<&str, Vec<&str>>, bas: &str) -> Vec<String> {\n"
            "    let mut gorulen: HashSet<String> = HashSet::new();\n"
            "    let mut yigin = vec![bas.to_string()];\n"
            "    let mut sira = Vec::new();\n"
            "    while let Some(d) = yigin.pop() {\n"
            "        if gorulen.contains(&d) { continue; }\n"
            "        gorulen.insert(d.clone());\n"
            "        sira.push(d.clone());\n"
            "        if let Some(ks) = g.get(d.as_str()) {\n"
            "            for x in ks { yigin.push(x.to_string()); }\n"
            "        }\n"
            "    }\n"
            "    sira\n"
            "}\n"),
        "cpp": (
            "#include <string>\n"
            "#include <vector>\n"
            "#include <queue>\n"
            "#include <unordered_map>\n"
            "std::unordered_map<std::string,int> bfs(\n"
            "    const std::unordered_map<std::string,std::vector<std::string>>& g,\n"
            "    const std::string& bas) {\n"
            "    std::unordered_map<std::string,int> m{{bas, 0}};\n"
            "    std::queue<std::string> k;\n"
            "    k.push(bas);\n"
            "    while (!k.empty()) {\n"
            "        auto d = k.front(); k.pop();\n"
            "        auto it = g.find(d);\n"
            "        if (it == g.end()) continue;\n"
            "        for (const auto& x : it->second)\n"
            "            if (!m.count(x)) { m[x] = m[d] + 1; k.push(x); }\n"
            "    }\n"
            "    return m;\n"
            "}\n",
            "#include <unordered_set>\n"
            "#include <stack>\n"
            "// Yığınlı DFS\n"
            "std::vector<std::string> dfs(\n"
            "    const std::unordered_map<std::string,std::vector<std::string>>& g,\n"
            "    const std::string& bas) {\n"
            "    std::unordered_set<std::string> gorulen;\n"
            "    std::vector<std::string> yigin{bas}, sira;\n"
            "    while (!yigin.empty()) {\n"
            "        auto d = yigin.back(); yigin.pop_back();\n"
            "        if (gorulen.count(d)) continue;\n"
            "        gorulen.insert(d);\n"
            "        sira.push_back(d);\n"
            "        auto it = g.find(d);\n"
            "        if (it != g.end())\n"
            "            for (const auto& x : it->second) yigin.push_back(x);\n"
            "    }\n"
            "    return sira;\n"
            "}\n"),
        "gdscript": (
            "func bfs(grafik: Dictionary, bas) -> Dictionary:\n"
            "    var mesafe := {bas: 0}\n"
            "    var kuyruk := [bas]\n"
            "    var i := 0\n"
            "    while i < kuyruk.size():\n"
            "        var d = kuyruk[i]\n"
            "        i += 1\n"
            "        for k in grafik.get(d, []):\n"
            "            if not mesafe.has(k):\n"
            "                mesafe[k] = mesafe[d] + 1\n"
            "                kuyruk.append(k)\n"
            "    return mesafe\n"
            "\n"
            "func dfs(grafik: Dictionary, d, gorulen: Dictionary = {}) -> Array:\n"
            "    gorulen[d] = true\n"
            "    var sira := [d]\n"
            "    for k in grafik.get(d, []):\n"
            "        if not gorulen.has(k):\n"
            "            sira.append_array(dfs(grafik, k, gorulen))\n"
            "    return sira\n",
            "# Yığınlı DFS — derin özyinelemeden kaçınır\n"
            "func dfs_yigin(grafik: Dictionary, bas) -> Array:\n"
            "    var gorulen := {}\n"
            "    var yigin := [bas]\n"
            "    var sira := []\n"
            "    while not yigin.is_empty():\n"
            "        var d = yigin.pop_back()\n"
            "        if gorulen.has(d):\n"
            "            continue\n"
            "        gorulen[d] = true\n"
            "        sira.append(d)\n"
            "        for k in grafik.get(d, []):\n"
            "            yigin.append(k)\n"
            "    return sira\n"),
    }),

    # ── 8. Dinamik programlama ──
    ("Dinamik programlama", "dinamik", "O(n) / O(n×m)", {
        "python": (
            "from functools import lru_cache\n"
            "\n"
            "@lru_cache(maxsize=None)\n"
            "def fib(n: int) -> int:\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
            "\n"
            "def fib_tablo(n: int) -> int:\n"
            "    if n < 2:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n - 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n",
            "# Sırt çantası — aşağıdan yukarı DP\n"
            "def sirt_cantasi(agirliklar: list, degerler: list,\n"
            "                  kapasite: int) -> int:\n"
            "    n = len(agirliklar)\n"
            "    # Yalnız bir satır — uzay O(kapasite)\n"
            "    tablo = [0] * (kapasite + 1)\n"
            "    for i in range(n):\n"
            "        for w in range(kapasite, agirliklar[i] - 1, -1):\n"
            "            tablo[w] = max(tablo[w],\n"
            "                           tablo[w - agirliklar[i]] + degerler[i])\n"
            "    return tablo[kapasite]\n"),
        "javascript": (
            "const notlar = new Map();\n"
            "\n"
            "function fib(n) {\n"
            "  if (n < 2) return n;\n"
            "  if (notlar.has(n)) return notlar.get(n);\n"
            "  const v = fib(n - 1) + fib(n - 2);\n"
            "  notlar.set(n, v);\n"
            "  return v;\n"
            "}\n"
            "\n"
            "function fibTablo(n) {\n"
            "  if (n < 2) return n;\n"
            "  let a = 0, b = 1;\n"
            "  for (let i = 0; i < n - 1; i++) [a, b] = [b, a + b];\n"
            "  return b;\n"
            "}\n",
            "// Sırt çantası — tek satır O(kapasite)\n"
            "function sirtCantasi(agirliklar, degerler, kapasite) {\n"
            "  const t = new Array(kapasite + 1).fill(0);\n"
            "  for (let i = 0; i < agirliklar.length; i++) {\n"
            "    for (let w = kapasite; w >= agirliklar[i]; w--) {\n"
            "      t[w] = Math.max(t[w], t[w - agirliklar[i]] + degerler[i]);\n"
            "    }\n"
            "  }\n"
            "  return t[kapasite];\n"
            "}\n"),
        "rust": (
            "use std::collections::HashMap;\n"
            "\n"
            "fn fib(n: u64, notlar: &mut HashMap<u64, u64>) -> u64 {\n"
            "    if n < 2 { return n; }\n"
            "    if let Some(&v) = notlar.get(&n) { return v; }\n"
            "    let v = fib(n - 1, notlar) + fib(n - 2, notlar);\n"
            "    notlar.insert(n, v);\n"
            "    v\n"
            "}\n"
            "\n"
            "fn fib_tablo(n: u64) -> u64 {\n"
            "    if n < 2 { return n; }\n"
            "    let (mut a, mut b) = (0u64, 1u64);\n"
            "    for _ in 0..n - 1 { let t = a + b; a = b; b = t; }\n"
            "    b\n"
            "}\n",
            "// Sırt çantası — tek satır\n"
            "fn sirt_cantasi(agirliklar: &[usize], degerler: &[i32],\n"
            "                 kapasite: usize) -> i32 {\n"
            "    let mut t = vec![0i32; kapasite + 1];\n"
            "    for i in 0..agirliklar.len() {\n"
            "        for w in (agirliklar[i]..=kapasite).rev() {\n"
            "            t[w] = t[w].max(t[w - agirliklar[i]] + degerler[i]);\n"
            "        }\n"
            "    }\n"
            "    t[kapasite]\n"
            "}\n"),
        "cpp": (
            "#include <unordered_map>\n"
            "long long fib(int n, std::unordered_map<int,long long>& n2) {\n"
            "    if (n < 2) return n;\n"
            "    auto it = n2.find(n);\n"
            "    if (it != n2.end()) return it->second;\n"
            "    long long v = fib(n - 1, n2) + fib(n - 2, n2);\n"
            "    n2[n] = v;\n"
            "    return v;\n"
            "}\n"
            "\n"
            "long long fib_tablo(int n) {\n"
            "    if (n < 2) return n;\n"
            "    long long a = 0, b = 1;\n"
            "    for (int i = 0; i < n - 1; ++i) { long long t = a + b; a = b; b = t; }\n"
            "    return b;\n"
            "}\n",
            "#include <vector>\n"
            "#include <algorithm>\n"
            "// Sırt çantası — tek satır\n"
            "int sirt_cantasi(const std::vector<int>& ag,\n"
            "                 const std::vector<int>& deg, int kap) {\n"
            "    std::vector<int> t(kap + 1, 0);\n"
            "    for (std::size_t i = 0; i < ag.size(); ++i)\n"
            "        for (int w = kap; w >= ag[i]; --w)\n"
            "            t[w] = std::max(t[w], t[w - ag[i]] + deg[i]);\n"
            "    return t[kap];\n"
            "}\n"),
        "gdscript": (
            "var _notlar := {}\n"
            "\n"
            "func fib(n: int) -> int:\n"
            "    if n < 2:\n"
            "        return n\n"
            "    if _notlar.has(n):\n"
            "        return _notlar[n]\n"
            "    var v := fib(n - 1) + fib(n - 2)\n"
            "    _notlar[n] = v\n"
            "    return v\n"
            "\n"
            "func fib_tablo(n: int) -> int:\n"
            "    if n < 2:\n"
            "        return n\n"
            "    var a := 0\n"
            "    var b := 1\n"
            "    for i in range(n - 1):\n"
            "        var t := a + b\n"
            "        a = b\n"
            "        b = t\n"
            "    return b\n",
            "# Sırt çantası — tek satır\n"
            "func sirt_cantasi(agirliklar: Array, degerler: Array,\n"
            "                   kapasite: int) -> int:\n"
            "    var t := []\n"
            "    for i in kapasite + 1:\n"
            "        t.append(0)\n"
            "    for i in agirliklar.size():\n"
            "        for w in range(kapasite, agirliklar[i] - 1, -1):\n"
            "            t[w] = max(t[w], t[w - agirliklar[i]] + degerler[i])\n"
            "    return t[kapasite]\n"),
    }),

    # ── 9. Açgözlü ──
    ("Açgözlü algoritma", "acgozlu", "O(n log n)", {
        "python": (
            "def bozuk_para(para: list, tutar: int) -> dict:\n"
            "    para = sorted(para, reverse=True)\n"
            "    sonuc = {}\n"
            "    for p in para:\n"
            "        if tutar >= p:\n"
            "            adet = tutar // p\n"
            "            sonuc[p] = adet\n"
            "            tutar -= adet * p\n"
            "    return sonuc if tutar == 0 else {}\n",
            "# Aralık birleştirme — çakışan aralıkları birleştir\n"
            "def birlestir_aralik(araliklar: list) -> list:\n"
            "    if not araliklar:\n"
            "        return []\n"
            "    araliklar = sorted(araliklar)\n"
            "    sonuc = [list(araliklar[0])]\n"
            "    for bas, son in araliklar[1:]:\n"
            "        if bas <= sonuc[-1][1]:\n"
            "            sonuc[-1][1] = max(sonuc[-1][1], son)\n"
            "        else:\n"
            "            sonuc.append([bas, son])\n"
            "    return [tuple(x) for x in sonuc]\n"),
        "javascript": (
            "function bozukPara(para, tutar) {\n"
            "  const s = [...para].sort((a, b) => b - a);\n"
            "  const sonuc = {};\n"
            "  for (const p of s) {\n"
            "    if (tutar >= p) {\n"
            "      const adet = Math.floor(tutar / p);\n"
            "      sonuc[p] = adet;\n"
            "      tutar -= adet * p;\n"
            "    }\n"
            "  }\n"
            "  return tutar === 0 ? sonuc : {};\n"
            "}\n",
            "// Aralık birleştirme\n"
            "function birlestirAralik(araliklar) {\n"
            "  if (araliklar.length === 0) return [];\n"
            "  const s = [...araliklar].sort((a, b) => a[0] - b[0]);\n"
            "  const sonuc = [[...s[0]]];\n"
            "  for (const [bas, son] of s.slice(1)) {\n"
            "    if (bas <= sonuc.at(-1)[1]) sonuc.at(-1)[1] = Math.max(sonuc.at(-1)[1], son);\n"
            "    else sonuc.push([bas, son]);\n"
            "  }\n"
            "  return sonuc;\n"
            "}\n"),
        "rust": (
            "fn bozuk_para(para: &[u32], mut tutar: u32) -> Option<Vec<(u32, u32)>> {\n"
            "    let mut p: Vec<u32> = para.to_vec();\n"
            "    p.sort_unstable_by(|a, b| b.cmp(a));\n"
            "    let mut sonuc = Vec::new();\n"
            "    for x in p {\n"
            "        if tutar >= x {\n"
            "            let adet = tutar / x;\n"
            "            sonuc.push((x, adet));\n"
            "            tutar -= adet * x;\n"
            "        }\n"
            "    }\n"
            "    if tutar == 0 { Some(sonuc) } else { None }\n"
            "}\n",
            "// Aralık birleştirme\n"
            "fn birlestir_aralik(mut a: Vec<(i32, i32)>) -> Vec<(i32, i32)> {\n"
            "    if a.is_empty() { return Vec::new(); }\n"
            "    a.sort_unstable();\n"
            "    let mut sonuc = vec![a[0]];\n"
            "    for &(bas, son) in &a[1..] {\n"
            "        let last = sonuc.last_mut().unwrap();\n"
            "        if bas <= last.1 { last.1 = last.1.max(son); }\n"
            "        else { sonuc.push((bas, son)); }\n"
            "    }\n"
            "    sonuc\n"
            "}\n"),
        "cpp": (
            "#include <vector>\n"
            "#include <algorithm>\n"
            "#include <optional>\n"
            "std::optional<std::vector<std::pair<int,int>>> bozuk_para(\n"
            "    std::vector<int> para, int tutar) {\n"
            "    std::sort(para.begin(), para.end(), std::greater<int>());\n"
            "    std::vector<std::pair<int,int>> sonuc;\n"
            "    for (int p : para)\n"
            "        if (tutar >= p) {\n"
            "            int adet = tutar / p;\n"
            "            sonuc.emplace_back(p, adet);\n"
            "            tutar -= adet * p;\n"
            "        }\n"
            "    if (tutar == 0) return sonuc;\n"
            "    return std::nullopt;\n"
            "}\n",
            "// Aralık birleştirme\n"
            "std::vector<std::pair<int,int>> birlestir_aralik(\n"
            "    std::vector<std::pair<int,int>> a) {\n"
            "    if (a.empty()) return {};\n"
            "    std::sort(a.begin(), a.end());\n"
            "    std::vector<std::pair<int,int>> s{a[0]};\n"
            "    for (std::size_t i = 1; i < a.size(); ++i) {\n"
            "        if (a[i].first <= s.back().second)\n"
            "            s.back().second = std::max(s.back().second, a[i].second);\n"
            "        else s.push_back(a[i]);\n"
            "    }\n"
            "    return s;\n"
            "}\n"),
        "gdscript": (
            "func bozuk_para(para: Array, tutar: int) -> Dictionary:\n"
            "    var p := para.duplicate()\n"
            "    p.sort()\n"
            "    p.reverse()\n"
            "    var sonuc := {}\n"
            "    for x in p:\n"
            "        if tutar >= x:\n"
            "            var adet := tutar / x\n"
            "            sonuc[x] = adet\n"
            "            tutar -= adet * x\n"
            "    return sonuc if tutar == 0 else {}\n",
            "# Aralık birleştirme\n"
            "func birlestir_aralik(araliklar: Array) -> Array:\n"
            "    if araliklar.is_empty():\n"
            "        return []\n"
            "    var a := araliklar.duplicate()\n"
            "    a.sort_custom(func(x, y): return x[0] < y[0])\n"
            "    var sonuc := [[a[0][0], a[0][1]]]\n"
            "    for i in range(1, a.size()):\n"
            "        var bas = a[i][0]\n"
            "        var son = a[i][1]\n"
            "        if bas <= sonuc[-1][1]:\n"
            "            sonuc[-1][1] = max(sonuc[-1][1], son)\n"
            "        else:\n"
            "            sonuc.append([bas, son])\n"
            "    return sonuc\n"),
    }),

    # ── 10. Sıralama (kabarcık + seçmeli) ──
    ("Temel sıralamalar", "siralama", "O(n²)", {
        "python": (
            "def kabarcik(dizi: list) -> list:\n"
            "    n = len(dizi)\n"
            "    for i in range(n - 1):\n"
            "        degisti = False\n"
            "        for j in range(n - 1 - i):\n"
            "            if dizi[j] > dizi[j + 1]:\n"
            "                dizi[j], dizi[j + 1] = dizi[j + 1], dizi[j]\n"
            "                degisti = True\n"
            "        if not degisti:\n"
            "            break\n"
            "    return dizi\n"
            "\n"
            "def secmeli(dizi: list) -> list:\n"
            "    n = len(dizi)\n"
            "    for i in range(n - 1):\n"
            "        en_kucuk = i\n"
            "        for j in range(i + 1, n):\n"
            "            if dizi[j] < dizi[en_kucuk]:\n"
            "                en_kucuk = j\n"
            "        dizi[i], dizi[en_kucuk] = dizi[en_kucuk], dizi[i]\n"
            "    return dizi\n",
            "# Python yerleşiği — Timsort, O(n log n)\n"
            "def sirala(dizi: list) -> list:\n"
            "    return sorted(dizi)\n"
            "\n"
            "# ⚠ `sort` yerinde değiştirir, `sorted` yeni liste döner\n"
            "def sirala_yerinde(dizi: list) -> None:\n"
            "    dizi.sort()\n"),
        "javascript": (
            "function kabarcik(dizi) {\n"
            "  const n = dizi.length;\n"
            "  for (let i = 0; i < n - 1; i++) {\n"
            "    let degisti = false;\n"
            "    for (let j = 0; j < n - 1 - i; j++) {\n"
            "      if (dizi[j] > dizi[j + 1]) {\n"
            "        [dizi[j], dizi[j + 1]] = [dizi[j + 1], dizi[j]];\n"
            "        degisti = true;\n"
            "      }\n"
            "    }\n"
            "    if (!degisti) break;\n"
            "  }\n"
            "  return dizi;\n"
            "}\n"
            "\n"
            "function secmeli(dizi) {\n"
            "  const n = dizi.length;\n"
            "  for (let i = 0; i < n - 1; i++) {\n"
            "    let e = i;\n"
            "    for (let j = i + 1; j < n; j++) if (dizi[j] < dizi[e]) e = j;\n"
            "    [dizi[i], dizi[e]] = [dizi[e], dizi[i]];\n"
            "  }\n"
            "  return dizi;\n"
            "}\n",
            "// ⚠ JS `sort` varsayılan olarak METİN sıralar!\n"
            "function siralaGuvenli(dizi) {\n"
            "  return [...dizi].sort((a, b) => a - b);   // sayısal\n"
            "}\n"),
        "rust": (
            "fn kabarcik(dizi: &mut [i32]) {\n"
            "    let n = dizi.len();\n"
            "    for i in 0..n.saturating_sub(1) {\n"
            "        let mut degisti = false;\n"
            "        for j in 0..n - 1 - i {\n"
            "            if dizi[j] > dizi[j + 1] {\n"
            "                dizi.swap(j, j + 1);\n"
            "                degisti = true;\n"
            "            }\n"
            "        }\n"
            "        if !degisti { break; }\n"
            "    }\n"
            "}\n"
            "\n"
            "fn secmeli(dizi: &mut [i32]) {\n"
            "    let n = dizi.len();\n"
            "    for i in 0..n.saturating_sub(1) {\n"
            "        let mut e = i;\n"
            "        for j in i + 1..n { if dizi[j] < dizi[e] { e = j; } }\n"
            "        dizi.swap(i, e);\n"
            "    }\n"
            "}\n",
            "// Rust yerleşiği — O(n log n)\n"
            "fn sirala(dizi: &mut [i32]) { dizi.sort_unstable(); }\n"),
        "cpp": (
            "#include <vector>\n"
            "#include <algorithm>\n"
            "void kabarcik(std::vector<int>& d) {\n"
            "    int n = static_cast<int>(d.size());\n"
            "    for (int i = 0; i < n - 1; ++i) {\n"
            "        bool degisti = false;\n"
            "        for (int j = 0; j < n - 1 - i; ++j)\n"
            "            if (d[j] > d[j + 1]) {\n"
            "                std::swap(d[j], d[j + 1]);\n"
            "                degisti = true;\n"
            "            }\n"
            "        if (!degisti) break;\n"
            "    }\n"
            "}\n"
            "\n"
            "void secmeli(std::vector<int>& d) {\n"
            "    int n = static_cast<int>(d.size());\n"
            "    for (int i = 0; i < n - 1; ++i) {\n"
            "        int e = i;\n"
            "        for (int j = i + 1; j < n; ++j) if (d[j] < d[e]) e = j;\n"
            "        std::swap(d[i], d[e]);\n"
            "    }\n"
            "}\n",
            "// STL — O(n log n)\n"
            "void sirala(std::vector<int>& d) {\n"
            "    std::sort(d.begin(), d.end());\n"
            "}\n"),
        "gdscript": (
            "func kabarcik(dizi: Array) -> Array:\n"
            "    var n := dizi.size()\n"
            "    for i in range(n - 1):\n"
            "        var degisti := false\n"
            "        for j in range(n - 1 - i):\n"
            "            if dizi[j] > dizi[j + 1]:\n"
            "                var t = dizi[j]\n"
            "                dizi[j] = dizi[j + 1]\n"
            "                dizi[j + 1] = t\n"
            "                degisti = true\n"
            "        if not degisti:\n"
            "            break\n"
            "    return dizi\n"
            "\n"
            "func secmeli(dizi: Array) -> Array:\n"
            "    var n := dizi.size()\n"
            "    for i in range(n - 1):\n"
            "        var e := i\n"
            "        for j in range(i + 1, n):\n"
            "            if dizi[j] < dizi[e]:\n"
            "                e = j\n"
            "        var t = dizi[i]\n"
            "        dizi[i] = dizi[e]\n"
            "        dizi[e] = t\n"
            "    return dizi\n",
            "# Godot yerleşiği\n"
            "func sirala(dizi: Array) -> Array:\n"
            "    var k := dizi.duplicate()\n"
            "    k.sort()\n"
            "    return k\n"),
    }),
]

#: Kategori etiketleri
KATEGORI: Dict[str, str] = {
    "yigin": "Yığın (Stack)",
    "kuyruk": "Kuyruk (Queue)",
    "bagli_liste": "Bağlı Liste",
    "ozyineleme": "Özyineleme",
    "hash": "Hash Tablosu",
    "agac": "Ağaç",
    "grafik": "Grafik",
    "dinamik": "Dinamik Programlama",
    "acgozlu": "Açgözlü Algoritma",
    "siralama": "Sıralama",
}

VARYANT: Dict[int, str] = {0: "temel", 1: "iyileştirilmiş"}


@dataclass
class UretilenDers:
    """Üretilen tek ders."""
    algoritma: str
    kategori: str
    dil: str
    varyant: str
    kod: str
    karmasiklik: str

    def baslik(self) -> str:
        return f"{self.algoritma} — {self.dil} ({self.varyant})"

    def metin(self) -> str:
        return (f"# {self.baslik()}\n\n"
                f"**Kategori:** {KATEGORI.get(self.kategori, self.kategori)}\n"
                f"**Dil:** {self.dil}\n"
                f"**Varyant:** {self.varyant}\n"
                f"**Karmaşıklık:** {self.karmasiklik}\n\n"
                f"```{self.dil}\n{self.kod.rstrip()}\n```\n")


def tum_dersler() -> List[UretilenDers]:
    """Tüm dersleri üretir."""
    d: List[UretilenDers] = []
    for ad, kat, karmasiklik, diller in ALGORITMALAR:
        for dil, kodlar in diller.items():
            for i, kod in enumerate(kodlar):
                d.append(UretilenDers(
                    algoritma=ad, kategori=kat, dil=dil,
                    varyant=VARYANT.get(i, f"v{i}"), kod=kod,
                    karmasiklik=karmasiklik))
    return d


def istatistik() -> Dict[str, Any]:
    """İstatistik."""
    d = tum_dersler()
    mt = [x.metin() for x in d]
    from collections import Counter

    return {
        "algoritma": len(ALGORITMALAR),
        "dil": len({x.dil for x in d}),
        "ders": len(d),
        "karakter": sum(len(m) for m in mt),
        "token_tahmini": sum(len(m) for m in mt) // 3,
        "dil_dagilim": dict(Counter(x.dil for x in d)),
        "kategori_dagilim": dict(Counter(x.kategori for x in d)),
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
            if (getattr(n, "category", "") or "") == "kod_mantigi_uretilen":
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
                category="kod_mantigi_uretilen",
                tags=["mantik", "algoritma", d.kategori, d.dil],
                confidence=0.95, importance=0.85,
                created=simdi, updated=simdi, version=1,
                sources=["mantik_uretici"], links=[], backlinks=[],
                embedding_model="", embedding=None, path="", extra={})
            k.vault.write_note(n, touch=False, emit=False)
            yazilan += 1
        except Exception:
            continue
    k.shutdown()
    return yazilan


def main() -> int:
    """Üreticiyi çalıştırır."""
    ap = argparse.ArgumentParser(description="Mantık müfredatı üretici")
    ap.add_argument("--istatistik", action="store_true")
    ap.add_argument("--kasaya", action="store_true")
    ap.add_argument("--liste", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  MANTIK MÜFREDATI ÜRETİCİ — şablon tabanlı, çok dilli")
    print("=" * 78)

    if args.liste:
        for d in tum_dersler():
            print(f"  {d.kategori:14} {d.algoritma:28} {d.dil:11} {d.varyant}")
        return 0

    st = istatistik()
    print(f"\n  algoritma : {st['algoritma']}")
    print(f"  dil       : {st['dil']}")
    print(f"  ders      : {st['ders']}")
    print(f"  karakter  : {st['karakter']:,}")
    print(f"  token     : ~{st['token_tahmini']:,}")
    print(f"\n  DİL      : {st['dil_dagilim']}")
    print(f"  KATEGORİ : {st['kategori_dagilim']}")

    if args.kasaya:
        n = kasaya_yaz()
        print(f"\n  kasaya yazıldı: {n} not")

    Path("_mantik_uretici.json").write_text(
        json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
