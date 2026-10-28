"""
Gerçek Godot 4 GDScript Korpusu — Doğrulayıcı ile Filtrelenmiş
================================================================
⚠ ÖLÇÜLEN DURUM (tur 15-22)
---------------------------
```
Yerel .gd dosya         : 3.516 (34 MB)  ← Godot 3/4 karışık
Godot 3 sızıntısı       : titowar_kod %35,5
Doküman kod bloğu       : yalnız 116 blok (çoğu `@export` örneği)
```

**Sorun:** Gerçek Godot 4 kodu **az**; var olan da Godot 3 ile karışık.
Yanlış kod öğretmek, eksik koddan **zararlı** (tur 17 dersi).

Çözüm: **Godot 4 derleyicisiyle doğrula**
------------------------------------------
Godot'un **gerçek ayrıştırıcısı** (`--check-only`) ile her bloğu
doğrula. Yalnız **derlenen** kod korpuse girer.

⚠ Neden bu güvenilir?
    Godot 3 sözdizimi Godot 4'te **derlenmez** → otomatik ayıklanır.
    Ölçüldü: `SetNoiseScale(` → `Parse Error: Function not found`.

Kullanım
--------
    python tools/godot4_korpus.py --tara
    python tools/godot4_korpus.py --dogrula --maks 200
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Godot 4.8 geliştirme sürümü (konsol)
GODOT = Path(r"E:\AppsTools\Godot_v4.8-dev7_win64"
             r"\Godot_v4.8-dev7_win64_console.exe")

#: Kaynak kökler
#: ⚠ tur 41: Kendi **Godot 4.8** projelerimiz — hepsi gerçek derleyici
#: ile doğrulanmış ortamlarda çalışıyor (TitoWar **Godot 3** olduğu
#: için kök taramasında eleniyordu).
KOKLER: Sequence[str] = (
    r"E:\PROJELER\myAi\TestProjects",
    r"E:\PROJELER\mobile-test",
    r"E:\PROJELER",
)

#: Atlanacak dizinler
ATLA = re.compile(r"node_modules|\\\.git\\|__pycache__|addons/")

#: ⚠ Godot 3 göstergeleri — hiç doğrulanmadan atılır (hız için)
G3 = re.compile(
    r"\.set_[a-z_]+\s*\(|\.get_[a-z_]+\s*\(\s*\)|\byield\s*\(|"
    r"\bKinematicBody|\bSpatialMaterial\b|\bColor3\s*\(|"
    r"\bPool(Byte|Int|Real|String|Vector2|Vector3|Color)Array\b|"
    r"\bambientColor\b|\bdiffuseColor\b|\bOpenSimplexNoise\b|"
    r"\bDirectory\.new|\bFile\.new")

#: Godot 4 göstergeleri — biri olmalı
G4 = re.compile(
    r"\bNode3D\b|\bCharacterBody3D\b|\bMeshInstance3D\b|\bCamera3D\b|"
    r"\bArea3D\b|\bRigidBody3D\b|\bStaticBody3D\b|\bMarker3D\b|"
    r"\bStandardMaterial3D\b|\bFastNoiseLite\b|\bPacked\w+Array\b|"
    r"\bDirAccess\b|\bFileAccess\b|\bTime\.get_ticks|\bawait\s|"
    r"\binstantiate\s*\(|@export|@onready|@rpc|\bGPUParticles\w*\b|"
    r"\bTween\b|\bMultiMeshInstance3D\b|\bTexture2D\b|\bNode2D\b")

#: Çıktı
CIKTI = Path(".neurocore/godot4_korpus.jsonl")


@dataclass
class Tarama:
    """Tarama istatistiği."""
    dosya: int = 0
    g3_atilan: int = 0
    g4_disi: int = 0
    dogrulanan: int = 0
    derlenen: int = 0
    hata: int = 0
    karakter: int = 0
    hata_tipi: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _godot_dogrula(kod: str, zaman: int = 60) -> Tuple[bool, str]:
    """Godot'un gerçek ayrıştırıcısıyla doğrular."""
    if not GODOT.exists():
        return (False, "godot yok")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "project.godot").write_text(
            '[application]\nconfig/name="d"\n'
            '[rendering]\nrenderer/rendering_method="mobile"\n',
            encoding="utf-8")
        h = d / "t.gd"
        h.write_text(kod, encoding="utf-8")
        return _godot_calistir(d, h, zaman)


def _godot_calistir(d: Path, h: Path, zaman: int) -> Tuple[bool, str]:
    """`--check-only` çalıştırır."""
    try:
        p = subprocess.run(
            [str(GODOT), "--headless", "--path", str(d),
             "--check-only", "--script", str(h)],
            capture_output=True, text=True, timeout=zaman,
            encoding="utf-8", errors="replace")
        out = (p.stdout or "") + (p.stderr or "")
        hata = [s for s in out.splitlines()
                if "error" in s.lower() or "Parse Error" in s]
        return (not hata, hata[0][:110] if hata else "")
    except subprocess.TimeoutExpired:
        return (False, "zaman aşımı")
    except Exception as exc:
        return (False, f"{type(exc).__name__}")


def _proje_baglaminda_dogrula(dosya: Path, zaman: int = 90
                              ) -> Tuple[bool, str]:
    """
    Dosyayı **kendi proje dizininde** doğrular.

    ⚠ tur 24: Ölçüldü ki izole doğrulama `class_name` ve `preload`
    bağımlılıklarını çözemiyor:
        `Could not find base class ""`    24 kez
        `Could not resolve super class`    4 kez

    Çözüm: dosyanın **dizinini** geçici projeye kopyala (en çok
    60 dosya), `project.godot` ekle, gerçek dosyayı derle.
    Böylece kardeş sınıflar ve `preload` yolları çözülür.
    """
    if not GODOT.exists():
        return (False, "godot yok")
    kaynak_dizin = dosya.parent
    try:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "project.godot").write_text(
                '[application]\nconfig/name="p"\n'
                '[rendering]\nrenderer/rendering_method="mobile"\n',
                encoding="utf-8")
            # ⚠ Kardeş `.gd` ve `.tscn` dosyalarını kopyala (sınırlı)
            sayi = 0
            for f in kaynak_dizin.iterdir():
                if sayi >= 60:
                    break
                if f.is_file() and f.suffix in (".gd", ".tscn", ".tres",
                                                ".import"):
                    try:
                        (d / f.name).write_bytes(f.read_bytes())
                        sayi += 1
                    except Exception:
                        continue
            hedef = d / dosya.name
            if not hedef.exists():
                return (False, "kopyalanamadı")
            return _godot_calistir(d, hedef, zaman)
    except Exception as exc:
        return (False, f"{type(exc).__name__}")


def _bloklar(kod: str, *, maks_satir: int = 90) -> List[str]:
    """
    GDScript dosyasını **derlenebilir bloklara** böler.

    ⚠ Yöntem: `extends` satırını koru, gövdeyi fonksiyon sınırlarında
    kes. Godot her bloğu bağımsız derler.
    """
    satirlar = kod.splitlines()
    if not satirlar:
        return []
    # `extends` / `class_name` başlığını bul
    baslik: List[str] = []
    for s in satirlar[:12]:
        t = s.strip()
        if t.startswith("extends") or t.startswith("@tool"):
            baslik.append(s)
        elif t.startswith("class_name"):
            # ⚠ `class_name` blok içinde **tekrar tanımlanamaz** →
            # yalnız `extends` kısmını al.
            continue
        elif t and not t.startswith("#"):
            break
    # ⚠ tur 23: sınıf **üye değişkenlerini** başlığa ekle — aksi hâlde
    # fonksiyon içindeki `hiz`, `can` gibi adlar "not declared" hatası
    # veriyordu (178 hata).
    uyeler: List[str] = []
    for s in satirlar:
        t = s.strip()
        if not t or t.startswith("#"):
            continue
        # `var x := ...` / `var x: Tip = ...` / `const X = ...` / `@export var`
        if re.match(r"^(var|const)\s+[A-Za-z_]\w*", t) or \
                re.match(r"^@\w+(\([^)]*\))?\s*var\s+[A-Za-z_]\w*", t) or \
                re.match(r"^enum\s+\w+", t) or \
                re.match(r"^signal\s+\w+", t) or \
                re.match(r"^class\s+\w+", t):
            # ⚠ tur 24: ÇOK SATIRLI üye tanımı — parantez/köşeli parantez
            # dengesi kapanana kadar devam et. Ölçüldü (tur 23): tek satır
            # alınınca 9 blok "Expected indented block" hatası veriyordu.
            idx = satirlar.index(s)
            parca = [t]
            denge = (t.count("(") - t.count(")")
                     + t.count("[") - t.count("]")
                     + t.count("{") - t.count("}"))
            if denge > 0:
                for ek in satirlar[idx + 1:idx + 9]:
                    parca.append(ek.rstrip())
                    denge += (ek.count("(") - ek.count(")")
                              + ek.count("[") - ek.count("]")
                              + ek.count("{") - ek.count("}"))
                    if denge <= 0:
                        break
            uyeler.append("\n".join(parca))
            if len(uyeler) >= 40:
                break
        elif re.match(r"^func\s+", t):
            break
    baslik = ["extends Node"] if not baslik else baslik
    baslik = baslik + uyeler
    out: List[str] = []
    # Fonksiyon sınırlarında blok üret
    i = 0
    while i < len(satirlar):
        if re.match(r"^\s*func\s+\w+", satirlar[i]):
            blok = list(baslik) + [""]
            j = i
            while j < len(satirlar) and j - i < maks_satir:
                blok.append(satirlar[j])
                # Sonraki fonksiyon başlıyorsa dur
                if j > i and re.match(r"^\s*func\s+\w+", satirlar[j]):
                    blok.pop()
                    break
                j += 1
            metin = "\n".join(blok).rstrip()
            if len(metin) >= 80:
                out.append(metin)
            i = j
        else:
            i += 1
    # Fonksiyon yoksa tüm dosya (başlıkla)
    if not out:
        metin = "\n".join(baslik + [""] + satirlar).strip()
        if len(metin) >= 80:
            out.append(metin[:6000])
    return out


def tara(*, dogrula: bool = False, maks: int = 0) -> Tarama:
    """Yerel `.gd` dosyalarını tarar ve doğrular."""
    t = Tarama()
    adaylar: List[str] = []
    for kok in KOKLER:
        p = Path(kok)
        if not p.exists():
            continue
        for f in p.rglob("*.gd"):
            if ATLA.search(str(f)) or f.stat().st_size > 200_000:
                continue
            try:
                kod = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            t.dosya += 1
            if G3.search(kod):
                t.g3_atilan += 1
                continue
            if not G4.search(kod):
                t.g4_disi += 1
                continue
            adaylar.extend(_bloklar(kod))
            if maks and len(adaylar) >= maks * 3:
                break

    print(f"  dosya         : {t.dosya:,}")
    print(f"  atılan (G3)   : {t.g3_atilan:,}")
    print(f"  atılan (G4 yok): {t.g4_disi:,}")
    print(f"  aday blok     : {len(adaylar):,}")

    if not dogrula:
        return t

    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    gorulen: set = set()
    with CIKTI.open("w", encoding="utf-8") as g:
        for i, blok in enumerate(adaylar):
            if maks and t.derlenen >= maks:
                break
            a = blok[:110]
            if a in gorulen:
                continue
            gorulen.add(a)
            t.dogrulanan += 1
            ok, hata = _godot_dogrula(blok)
            if ok:
                t.derlenen += 1
                t.karakter += len(blok)
                g.write(json.dumps(
                    {"metin": blok + "\n", "kategori": "godot4_korpus",
                     "dogrulandi": True}, ensure_ascii=False) + "\n")
            else:
                t.hata += 1
                anahtar = re.sub(r'"[^"]*"', '""', hata)[:70]
                t.hata_tipi[anahtar] = t.hata_tipi.get(anahtar, 0) + 1
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(adaylar)} · derlenen "
                      f"{t.derlenen} · hata {t.hata}", flush=True)
    return t


def main() -> int:
    """Tarama/doğrulamayı çalıştırır."""
    ap = argparse.ArgumentParser(description="Godot 4 korpus doğrulayıcı")
    ap.add_argument("--tara", action="store_true")
    ap.add_argument("--dogrula", action="store_true")
    ap.add_argument("--maks", type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print("  GERÇEK GODOT 4 KORPUSU — derleyici doğrulamalı")
    print("=" * 78)
    print(f"  Godot: {'VAR' if GODOT.exists() else 'YOK'}")

    t = tara(dogrula=args.dogrula, maks=args.maks)
    print(f"\n  doğrulanan    : {t.dogrulanan:,}")
    print(f"  DERLENEN      : {t.derlenen:,}")
    print(f"  hata          : {t.hata:,}")
    print(f"  karakter      : {t.karakter:,}")
    print(f"  token         : ~{t.karakter // 3:,}")
    if t.dogrulanan:
        print(f"  başarı oranı  : "
              f"%{t.derlenen / max(1, t.dogrulanan) * 100:.0f}")
    if t.hata_tipi:
        print("\n  HATA TİPLERİ:")
        for h, n in sorted(t.hata_tipi.items(), key=lambda x: -x[1])[:8]:
            print(f"    {n:4}x {h}")
    print(f"\n  dosya: {CIKTI}")

    Path("_godot4_korpus.json").write_text(
        json.dumps(t.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
