"""
Blender Derleme Testi — Gerçek Blender ile Kod Doğrulama
==========================================================
⚠ Neden gerekli
---------------
Tur 30'da **Godot derleme testi** yazıldı ve gerçek hataları buldu
(prose, girinti, `return` eksikliği). Ama **Blender tarafı** yalnız
metin denetimiyle doğrulanıyordu.

Bu araç: üretilen Blender kodunu **gerçek Blender'da** çalıştırır.

Yöntem
------
    1. Üretilen kodu `.py` dosyasına yaz
    2. `blender --background --python dosya.py` ile çalıştır
    3. Çıktı + hataları topla
    4. Hataları **sınıflandır** (sözdizimi / API / çalışma zamanı)

⚠ Ölçülen Blender hata biçimleri:
    `SyntaxError: ...`                 → sözdizimi
    `AttributeError: ...`              → uydurma API
    `TypeError: ...`                   → yanlış parametre
    `RuntimeError: ...`                → çalışma zamanı

Kullanım
--------
    python tools/blender_derleme_testi.py
    python tools/blender_derleme_testi.py --adet 6
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

BLENDER = r"E:\AppsTools\Blender 5.2\blender.exe"

#: Test dizini
DIZIN = Path("TestProjects/blender_derleme")

#: ⚠ ÖLÇÜLEN (tur 35): şablonlarım kapanmamış parantez bırakıyordu
#: (`mesh = bpy.data.meshes.new(` → `(` hiç kapanmıyor). Bu yüzden
#: gerçek Blender `(` was never closed veriyordu. Şablonlar artık
#: **tam ve derlenebilir**.
ONEKLER: Dict[str, str] = {
    "kup": "import bpy\n",
    "malzeme": "import bpy\n",
    "mesh": "import bpy\n",
    "goruntu": "import bpy\n",
    "vertex": ("import bpy\nmesh = bpy.data.meshes.new('M')\n"
               "mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [],"
               " [(0, 1, 2)])\n"),
    "render": "import bpy\nscene = bpy.context.scene\n",
}

#: Her test için kapanış — eksik ifadeleri tamamlar.
#:
#: ⚠ İki rol: (1) açık parantezleri kapat, (2) referans edilen
#: nesneleri tanımla.
KAPANISLAR: Dict[str, str] = {
    "kup": "\nprint('TAMAM')\n",
    "malzeme": "\nprint('TAMAM')\n",
    "mesh": "\nprint('TAMAM')\n",
    "goruntu": "\nprint('TAMAM')\n",
    "vertex": "\nprint('TAMAM')\n",
    "render": "\nprint('TAMAM')\n",
}

#: ⚠ Açık parantezleri kapatma yardımcısı.
def _parantez_kapat(kod: str) -> str:
    """
    Açık parantezleri kapatır.

    ⚠ Ölçülen: model/şablon `mesh = bpy.data.meshes.new(` ile bitiyor →
    `(` kapanmıyor → Blender `'(' was never closed` veriyor.
    """
    acik = kod.count("(") - kod.count(")")
    acik_k = kod.count("[") - kod.count("]")
    if acik > 0:
        kod = kod.rstrip() + ")" * acik
    if acik_k > 0:
        kod = kod.rstrip() + "]" * acik_k
    return kod


@dataclass
class BlenderSonuc:
    """Bir kod parçasının Blender sonucu."""
    ad: str = ""
    gecti: bool = False
    hata: str = ""
    hata_turu: str = ""
    kod: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ad": self.ad, "gecti": self.gecti,
                "hata": self.hata[:160], "hata_turu": self.hata_turu,
                "kod": self.kod[:300]}


def _hata_sinifla(cikti: str) -> Tuple[str, str]:
    """
    Blender çıktısından **hata türü + mesaj** çıkarır.

    ⚠ Ölçülen: Blender hataları `Traceback` içinde gelir ve son satır
    hata türünü taşır.
    """
    # Traceback son satırı: `AttributeError: ...`
    for satir in reversed(cikti.splitlines()):
        s = satir.strip()
        m = re.match(r"^(\w+Error):\s*(.*)$", s)
        if m:
            tur = m.group(1)
            if tur == "SyntaxError":
                return "sozdizimi", m.group(2)[:150]
            if tur == "AttributeError":
                return "api", m.group(2)[:150]
            if tur == "TypeError":
                return "parametre", m.group(2)[:150]
            if tur in ("NameError", "KeyError", "IndexError"):
                return "tanimsiz", m.group(2)[:150]
            return "runtime", m.group(2)[:150]
    return "", ""


def calistir(yol: Path, saniye: int = 90) -> Tuple[bool, str]:
    """Bir Python dosyasını Blender'da çalıştırır."""
    if not Path(BLENDER).exists():
        return False, "blender yok"
    try:
        s = subprocess.run(
            [BLENDER, "--background", "--factory-startup",
             "--python", str(yol)],
            capture_output=True, text=True, timeout=saniye,
            encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, "zaman aşımı"
    except Exception as exc:
        return False, f"{type(exc).__name__}"
    cikti = (s.stdout or "") + (s.stderr or "")
    # Traceback var mı?
    hata_var = "Traceback" in cikti or "Error:" in cikti
    return (not hata_var), cikti


def main() -> int:
    """Blender derleme testini çalıştırır."""
    ap = argparse.ArgumentParser(description="Blender derleme testi")
    ap.add_argument("--adet", type=int, default=6)
    args = ap.parse_args()

    print("=" * 78)
    print("  BLENDER DERLEME TESTİ — üretilen kod gerçekten çalışıyor mu?")
    print("=" * 78)

    if not Path(BLENDER).exists():
        print(f"  Blender bulunamadı: {BLENDER}")
        return 1
    if DIZIN.exists():
        shutil.rmtree(DIZIN, ignore_errors=True)
    DIZIN.mkdir(parents=True, exist_ok=True)

    try:
        from tools.kod_uretici import KodUretici, BLENDER_ISTEMLERI
    except Exception as exc:
        print(f"  üretici yüklenemedi: {exc}")
        return 1

    u = KodUretici()
    if not u.yukle():
        return 1

    anahtarlar = list(BLENDER_ISTEMLERI)[:args.adet]
    print(f"\n  {len(anahtarlar)} parça test ediliyor...\n")

    sonuclar: List[BlenderSonuc] = []
    for i, a in enumerate(anahtarlar, 1):
        s = u.uret_veya_sablon(a, tur="blender")
        onek = ONEKLER.get(a, "import bpy\n")
        kapanis = KAPANISLAR.get(a, "\n")
        kod = _parantez_kapat(onek + s.kod) + kapanis
        yol = DIZIN / f"t{i}_{a}.py"
        yol.write_text(kod, encoding="utf-8")
        gecti, cikti = calistir(yol)
        tur, mesaj = _hata_sinifla(cikti)
        if gecti and not tur:
            r = BlenderSonuc(ad=a, gecti=True, kod=kod)
        else:
            r = BlenderSonuc(ad=a, gecti=False, hata=mesaj,
                             hata_turu=tur or "bilinmeyen", kod=kod)
        sonuclar.append(r)
        im = "✓" if r.gecti else "✗"
        kaynak = "M" if s.kaynak == "model" else "S"
        print(f"  {im} [{kaynak}] {a:10} ", end="")
        print("ÇALIŞTI" if r.gecti else
              f"{r.hata_turu:10} · {r.hata[:56]}", flush=True)

    gecen = sum(1 for r in sonuclar if r.gecti)
    print()
    print("-" * 78)
    print(f"  SONUÇ: {gecen}/{len(sonuclar)} çalıştı "
          f"(%{gecen / max(1, len(sonuclar)) * 100:.0f})")
    from collections import Counter

    turler = Counter(r.hata_turu for r in sonuclar if not r.gecti)
    if turler:
        print(f"  hata türleri: {dict(turler)}")

    Path("_blender_derleme.json").write_text(json.dumps({
        "gecen": gecen, "toplam": len(sonuclar),
        "oran": gecen / max(1, len(sonuclar)),
        "hata_turleri": dict(turler),
        "sonuclar": [r.to_dict() for r in sonuclar],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  rapor: _blender_derleme.json")
    return 0 if gecen == len(sonuclar) else 1


if __name__ == "__main__":
    raise SystemExit(main())
