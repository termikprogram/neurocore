"""
Godot Derleme Testi — Üretilen Kod Gerçekten Çalışıyor mu?
============================================================
⚠ Neden gerekli
---------------
Tur 29'a kadar kod üretimi **yalnız metin** olarak doğrulandı:
    * Kısıtlar (dil kilidi, API sözlüğü)
    * Tekrar kesme
    * Yazım düzeltme
    * Yapısal denetim (parantez, `:`)

**Ama hiçbiri Godot'un derleyicisini kullanmadı.** Geçerli görünen
kod Godot'ta **parse hatası** verebilir.

Bu araç: üretilen her kod parçasını **gerçek bir Godot projesine**
koyar, `--headless` ile derler ve hataları toplar.

Yöntem
------
    1. Üretilen kodları `kod_uretici` ile al
    2. Her birini ayrı bir `.gd` dosyasına yaz
    3. Godot `--headless --check-only` ile derle
    4. Hataları **sınıflandır** (parse / tip / çalışma zamanı)
    5. Geçme oranını ölç

Kullanım
--------
    python tools/godot_derleme_testi.py
    python tools/godot_derleme_testi.py --adet 5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

GODOT = r"E:\AppsTools\Godot_v4.8-dev7_win64\Godot_v4.8-dev7_win64_console.exe"

#: Test projesi kökü
PROJE = Path("TestProjects/derleme_testi")

#: Her test için gereken önek — üretilen parça **eksik** olabilir
#: (örneğin yalnız fonksiyon gövdesi). Bu önek tamamlar.
ONEKLER: Dict[str, str] = {
    "duvar": "extends Node3D\n\n",
    "arazi": "extends Node3D\n\n",
    "malzeme": "extends Node3D\n\n",
    "karakter": "",
    "agac": "extends Node3D\n\n",
}


@dataclass
class DerlemeSonuc:
    """Bir kod parçasının derleme sonucu."""
    ad: str = ""
    gecti: bool = False
    hata: str = ""
    hata_turu: str = ""
    satir: int = 0
    kod: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ad": self.ad, "gecti": self.gecti,
                "hata": self.hata[:150], "hata_turu": self.hata_turu,
                "satir": self.satir, "kod": self.kod[:200]}


def _hata_sinifla(cikti: str) -> Tuple[str, str, int]:
    """
    Godot çıktısından **hata türü + mesaj + satır** çıkarır.

    ⚠ Ölçülen Godot hata biçimleri:
        `Parse Error: ...`          → sözdizimi
        `SCRIPT ERROR: ...`         → çalışma zamanı
        `Invalid call. Nonexistent function ...` → API hatası
        `res://x.gd:12 - Parse Error: ...`       → satır bilgili
    """
    for satir in cikti.splitlines():
        s = satir.strip()
        if "Parse Error" in s or "Parser Error" in s:
            m = re.search(r":(\d+)\s*-\s*Parse Error:?\s*(.*)", s)
            if m:
                return "parse", m.group(2)[:140], int(m.group(1))
            m2 = re.search(r"Parse Error:?\s*(.*)", s)
            return "parse", (m2.group(1)[:140] if m2 else s[:140]), 0
        if "Nonexistent function" in s or "Invalid call" in s:
            return "api", s[:140], 0
        if "SCRIPT ERROR" in s:
            m3 = re.search(r":(\d+)\s*-\s*(.*)", s)
            if m3:
                return "runtime", m3.group(2)[:140], int(m3.group(1))
            return "runtime", s[:140], 0
    return "", "", 0


def _girinti_duzelt(kod: str) -> str:
    """
    **Ortak girintiyi kaldırır, gövde girintisini korur.**

    ⚠ ÖLÇÜLEN TUZAK: `lstrip()` ile satır satır girinti silmek
    Godot'ta `Expected indented block after function declaration`
    hatası veriyor (ölçüldü: 0/5 derlendi).

    Neden? `func uret():` satırı sıfır girintiye inerken **gövdesi**
    de sıfıra indi → Godot gövdeyi göremedi.

    Doğru yöntem: **en küçük** girintiyi (boş olmayan satırlar arasında)
    bul ve **her satırdan** o kadar kaldır. Böylece:
        * `func` sıfır girintiye iner
        * gövde kendi **göreli** girintisini korur
    """
    satirlar = kod.splitlines()
    girintiler = [len(s) - len(s.lstrip()) for s in satirlar if s.strip()]
    if not girintiler:
        return kod
    en_az = min(girintiler)
    if en_az == 0:
        return kod
    return "\n".join(
        (s[en_az:] if len(s) >= en_az and s.strip() else s)
        for s in satirlar)


def proje_kur(hedef: Path) -> None:
    """Minimal Godot projesi kurar (derleme için gerekli)."""
    hedef.mkdir(parents=True, exist_ok=True)
    (hedef / "project.godot").write_text(
        '[application]\nconfig/name="derleme_testi"\n'
        'run/main_scene="res://ana.tscn"\n'
        '[rendering]\nrenderer/rendering_method="gl_compatibility"\n',
        encoding="utf-8")
    (hedef / "ana.tscn").write_text(
        '[gd_scene format=3]\n\n[node name="Ana" type="Node3D"]\n',
        encoding="utf-8")
    (hedef / "scripts").mkdir(exist_ok=True)


def derle(hedef: Path, script: str, saniye: int = 60) -> Tuple[bool, str]:
    """
    Bir GDScript dosyasını Godot ile derler.

    ⚠ `--check-only` **yalnız sözdizimi** denetler, sahneyi çalıştırmaz.
    Bu, hızlı ve güvenli (GPU kullanmaz).
    """
    if not Path(GODOT).exists():
        return False, "godot yok"
    yol = hedef / "scripts" / f"{script}.gd"
    try:
        s = subprocess.run(
            [GODOT, "--headless", "--path", str(hedef),
             "--check-only", "--script", f"res://scripts/{script}.gd"],
            capture_output=True, text=True, timeout=saniye,
            encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, "zaman aşımı"
    except Exception as exc:
        return False, f"{type(exc).__name__}"
    cikti = (s.stdout or "") + (s.stderr or "")
    return s.returncode == 0, cikti


def main() -> int:
    """Godot derleme testini çalıştırır."""
    ap = argparse.ArgumentParser(description="Godot derleme testi")
    ap.add_argument("--adet", type=int, default=5,
                    help="Kaç parça test edilsin")
    args = ap.parse_args()

    print("=" * 78)
    print("  GODOT DERLEME TESTİ — üretilen kod gerçekten derleniyor mu?")
    print("=" * 78)

    if not Path(GODOT).exists():
        print(f"  Godot bulunamadı: {GODOT}")
        return 1

    # Projeyi kur
    if PROJE.exists():
        shutil.rmtree(PROJE, ignore_errors=True)
    proje_kur(PROJE)
    print(f"  proje kuruldu: {PROJE}")

    # Kod üret
    try:
        from tools.kod_uretici import KodUretici, GODOT_ISTEMLERI
    except Exception as exc:
        print(f"  üretici yüklenemedi: {exc}")
        return 1

    u = KodUretici()
    if not u.yukle():
        return 1

    sonuclar: List[DerlemeSonuc] = []
    anahtarlar = list(GODOT_ISTEMLERI)[:args.adet]
    print(f"\n  {len(anahtarlar)} parça test ediliyor...\n")

    for i, a in enumerate(anahtarlar, 1):
        s = u.uret_veya_sablon(a, tur="godot")
        # Önek ekle (üretilen parça eksik olabilir)
        onek = ONEKLER.get(a, "extends Node3D\n\n")
        kod = (onek + s.kod) if not s.kod.startswith("extends") else s.kod

        # ⚠ GİRDİNTİ DÜZELTME — ölçülen tuzak:
        # İlk sürüm `func`/`var` satırlarını `lstrip()` ediyordu; bu,
        # **gövde girintisini de** sildiği için Godot
        # "Expected indented block after function declaration" veriyordu
        # (ölçüldü: 0/5 derlendi).
        #
        # Doğru yöntem: **tüm bloğu** ortak girintiden kurtar (dedent),
        # gövde girintisini **koru**.
        kod = _girinti_duzelt(kod)

        (PROJE / "scripts" / f"t{i}_{a}.gd").write_text(kod,
                                                        encoding="utf-8")
        gecti, cikti = derle(PROJE, f"t{i}_{a}")
        tur, mesaj, satir = _hata_sinifla(cikti)
        r = DerlemeSonuc(ad=a, gecti=gecti, hata=mesaj if not gecti else "",
                         hata_turu=tur, satir=satir, kod=kod)
        sonuclar.append(r)
        im = "✓" if gecti else "✗"
        kaynak = "M" if s.kaynak == "model" else "S"
        print(f"  {im} [{kaynak}] {a:10} ", end="")
        if gecti:
            print("DERLENDİ")
        else:
            print(f"{tur:8} satır {satir:3} · {mesaj[:56]}", flush=True)

    gecen = sum(1 for r in sonuclar if r.gecti)
    print()
    print("-" * 78)
    print(f"  SONUÇ: {gecen}/{len(sonuclar)} derlendi "
          f"(%{gecen / max(1, len(sonuclar)) * 100:.0f})")

    # Hata türü dağılımı
    from collections import Counter

    turler = Counter(r.hata_turu for r in sonuclar if not r.gecti)
    if turler:
        print(f"  hata türleri: {dict(turler)}")

    Path("_godot_derleme.json").write_text(json.dumps({
        "gecen": gecen, "toplam": len(sonuclar),
        "oran": gecen / max(1, len(sonuclar)),
        "hata_turleri": dict(turler),
        "sonuclar": [r.to_dict() for r in sonuclar],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  rapor: _godot_derleme.json")
    return 0 if gecen == len(sonuclar) else 1


if __name__ == "__main__":
    raise SystemExit(main())
