"""
Godot 4 Doküman Kod Bloklarını Retrieval İndeksine Çevirir
============================================================
⚠ ÖLÇÜLEN İHTİYAÇ (tur 21)
--------------------------
Retrieval bağlamı token dağılımını **kaydırdı**:
```
BAĞLAMSIZ : 'ambientColor = Color3(...)'   ← Godot 3
BAĞLAMLI  : 'ambient_color = Color(...)'   ← Godot 4'e kaydı  ✓
```
**Ama derleme hâlâ 0/4** — çünkü bağlam yalnız **imza** içeriyor:
```
#   BaseMaterial3D.backlight : Color
#   FastNoiseLite.frequency : float
```

İmza **"ne var"** der; **"nasıl kullanılır"** demez.

Çözüm: **doküman kod bloklarını** indeksle
-------------------------------------------
`dokuman_godot.jsonl` içinde **gerçek GDScript kod blokları** var
(Godot dokümanından çıkarıldı). Bunlar **kullanım** gösterir:
```gdscript
var noise = FastNoiseLite.new()
noise.noise_type = FastNoiseLite.TYPE_PERLIN
noise.frequency = 0.05
```

Kullanım
--------
    python tools/godot4_ornek_indeks.py --istatistik
    python tools/godot4_ornek_indeks.py --uret
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

KAYNAK = Path(".neurocore/dokuman_godot.jsonl")
CIKTI = Path(".neurocore/godot4_ornekler.jsonl")

#: GDScript kod bloğu
BLOK = re.compile(r"```(?:gdscript|gd)?\s*\n(.*?)```", re.DOTALL)

#: ⚠ Godot 3 göstergeleri — bu bloklar **alınmaz** (tur 16-17 dersi)
G3 = re.compile(
    r"\.set_[a-z_]+\s*\(|\.get_[a-z_]+\s*\(\s*\)|\byield\s*\(|"
    r"\bKinematicBody|\bSpatial\b|\bSpatialMaterial\b|\bColor3\s*\(|"
    r"\bPool(Byte|Int|Real|String|Vector2|Vector3|Color)Array\b|"
    r"\bambientColor\b|\bdiffuseColor\b|\bOpenSimplexNoise\b|"
    r"\bDirectory\b|\bFile\.new\s*\(|\bMeshInstance\b(?!3D)|"
    r"\bCamera\b(?!3D)|\bPosition3D\b|\.instance\s*\(\s*\)")

#: Godot 4 göstergeleri — blok **bunlardan birini** içermeli
G4 = re.compile(
    r"\bNode3D\b|\bCharacterBody3D\b|\bMeshInstance3D\b|\bCamera3D\b|"
    r"\bArea3D\b|\bRigidBody3D\b|\bStaticBody3D\b|\bMarker3D\b|"
    r"\bStandardMaterial3D\b|\bFastNoiseLite\b|\bPacked\w+Array\b|"
    r"\bDirAccess\b|\bFileAccess\b|\bTime\.get_ticks|\bawait\s|"
    r"\binstantiate\s*\(|@export|@onready|@rpc|"
    r"\bGPUParticles\w*\b|\bMultiMeshInstance3D\b|\bTween\b")


@dataclass
class Istatistik:
    """Üretim istatistiği."""
    kaynak: int = 0
    blok: int = 0
    g3_atilan: int = 0
    g4_disi_atilan: int = 0
    kisa_atilan: int = 0
    kayit: int = 0
    karakter: int = 0
    sinif_dagilim: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()
             if k != "sinif_dagilim"}
        d["en_cok"] = sorted(self.sinif_dagilim.items(),
                             key=lambda x: -x[1])[:12]
        return d


#: URL'den sınıf adı (`class_fastnoiselite.html`)
URL_SINIF = re.compile(r"class[_-]([a-z0-9_%]+)\.html")

#: Kaynak başlığından sınıf adı
BASLIK_SINIF = re.compile(r"\*\*Kaynak:\*\*\s*([^&\n|]+?)\s*(?:&mdash;|—|\||$)")


def _anahtar_kelimeler(kod: str) -> List[str]:
    """Kod bloğundan arama anahtarları çıkarır."""
    k: List[str] = []
    # Sınıf adları (`X.new()`, `X.` erişimi)
    for m in re.finditer(r"\b([A-Z]\w{2,})\s*\.", kod):
        k.append(m.group(1).lower())
    for m in re.finditer(r"\b([A-Z]\w{2,})\.new\s*\(", kod):
        k.append(m.group(1).lower())
    # Fonksiyon/özellik adları (`x.frequency`)
    for m in re.finditer(r"\.([a-z_]\w{3,})", kod):
        k.append(m.group(1))
    # `func _ready` gibi
    for m in re.finditer(r"\bfunc\s+(\w+)", kod):
        k.append(m.group(1).lower())
    return list(dict.fromkeys(k))[:40]


def uret() -> Istatistik:
    """Kod bloklarını indeksler."""
    s = Istatistik()
    if not KAYNAK.exists():
        print(f"  kaynak yok: {KAYNAK}")
        return s

    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    gorulen: set = set()
    with KAYNAK.open(encoding="utf-8") as f, \
            CIKTI.open("w", encoding="utf-8") as g:
        for satir in f:
            try:
                d = json.loads(satir)
            except Exception:
                continue
            s.kaynak += 1
            metin = d.get("metin", "")
            url = d.get("url", "") or ""
            # Sınıf adı: kaynak başlığı > URL
            mb = BASLIK_SINIF.search(metin)
            sinif = mb.group(1).strip() if mb else ""
            if not sinif:
                mu = URL_SINIF.search(url)
                sinif = mu.group(1) if mu else ""
            for m in BLOK.finditer(metin):
                kod = m.group(1).strip()
                s.blok += 1
                if len(kod) < 60:
                    s.kisa_atilan += 1
                    continue
                # ⚠ Godot 3 kalıntısı → AT
                if G3.search(kod):
                    s.g3_atilan += 1
                    continue
                # ⚠ Godot 4 göstergesi yok → AT
                if not G4.search(kod):
                    s.g4_disi_atilan += 1
                    continue
                anahtar = kod[:120]
                if anahtar in gorulen:
                    continue
                gorulen.add(anahtar)
                kw = _anahtar_kelimeler(kod)
                g.write(json.dumps(
                    {"kod": kod, "sinif": sinif, "url": url,
                     "kelimeler": kw, "uzunluk": len(kod)},
                    ensure_ascii=False) + "\n")
                s.kayit += 1
                s.karakter += len(kod)
                if sinif:
                    s.sinif_dagilim[sinif] = (
                        s.sinif_dagilim.get(sinif, 0) + 1)
    return s


def main() -> int:
    """Üretimi çalıştırır."""
    ap = argparse.ArgumentParser(description="Godot 4 örnek indeksi")
    ap.add_argument("--istatistik", action="store_true")
    ap.add_argument("--uret", action="store_true")
    ap.add_argument("--ornek", type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print("  GODOT 4 DOKÜMAN KOD BLOKLARI → RETRIEVAL İNDEKSİ")
    print("=" * 78)
    print(f"  kaynak: {KAYNAK}")
    print(f"  ⚠ Godot 3 kalıntılı bloklar ATILIR (tur 16-17 dersi)")

    s = uret()
    print(f"\n  kaynak kayıt    : {s.kaynak:,}")
    print(f"  kod bloğu       : {s.blok:,}")
    print(f"  atılan (kısa)   : {s.kisa_atilan:,}")
    print(f"  atılan (Godot3) : {s.g3_atilan:,}  ← sızıntı önlendi")
    print(f"  atılan (G4 yok) : {s.g4_disi_atilan:,}")
    print(f"  İNDEKSLENEN     : {s.kayit:,}")
    print(f"  karakter        : {s.karakter:,}")
    print(f"  dosya           : {CIKTI}")
    print(f"\n  en çok örnek: {sorted(s.sinif_dagilim.items(), key=lambda x: -x[1])[:8]}")

    if args.ornek and CIKTI.exists():
        print("\n  ÖRNEK BLOKLAR:")
        with CIKTI.open(encoding="utf-8") as f:
            for i, satir in enumerate(f):
                if i >= args.ornek:
                    break
                d = json.loads(satir)
                print(f"\n    [{d['sinif'] or '?'}] ({d['uzunluk']} karakter)")
                for ln in d["kod"].splitlines()[:6]:
                    print(f"      {ln}")

    Path("_godot4_ornek.json").write_text(
        json.dumps(s.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
