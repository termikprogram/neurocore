"""
Gerçek Godot Kodu Madencisi — TitoWar Projesinden
===================================================
⚠ Neden bu kaynak?
------------------
Şimdiye kadar kod verisi **öğretmen modelden** üretildi (`kod_uretim`)
veya **resmî dokümandan** çekildi (`blender_api`, `kod_ornek`).

**Üçüncü ve en değerli kaynak:** çalışan, **gerçek** Godot projesi.

```
E:\\PROJELER\\TitoWar : 1.384 .gd dosyası · 13,3 MB
```

| Kaynak | Boyut | Nitelik |
|---|---|---|
| Öğretmen üretimi | 2.291 not | ⚠️ Uydurma olabilir |
| Resmî doküman | 1.375 not | ✅ Doğrulanmış API |
| **TitoWar (gerçek proje)** | **13,3 MB** | ✅ **Çalışan kod** |

**Kritik:** TitoWar kodu **Godot 4.8'de çalışıyor** — sözdizimi,
API kullanımı, kalıplar **gerçek**. Model bu veriyi öğrenirse
**çalışan kod** üretir.

Yöntem
------
    1. `.gd` dosyalarını bul
    2. **Tekrar eden dosyaları** at (TitoWar'da aynı dosya 3 kez var)
    3. Çok büyük dosyaları **böl** (213 KB tek dosya → örnek değil)
    4. **Fonksiyon bloklarını** çıkar (`func ...` → sonu)
    5. Kasaya yaz — `godot_gercek` kategorisi

Kullanım
--------
    python tools/titowar_kod.py --analiz
    python tools/titowar_kod.py --topla --maks 3000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Gerçek Godot projesi kökü
KAYNAK = Path(r"E:\PROJELER\TitoWar")

#: Fonksiyon başlangıcı
FUNC_DESEN = re.compile(r"^(func\s+\w+|@\w+.*)$", re.MULTILINE)

#: Tek örnek için azami satır
MAKS_SATIR = 60


@dataclass
class MadenRaporu:
    """Madencilik raporu."""
    dosya: int = 0
    tekil_dosya: int = 0
    ornek: int = 0
    atlanan_kisa: int = 0
    atlanan_uzun: int = 0
    satir: int = 0
    token_tahmini: int = 0
    kategoriler: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dosya": self.dosya, "tekil_dosya": self.tekil_dosya,
            "ornek": self.ornek, "atlanan_kisa": self.atlanan_kisa,
            "atlanan_uzun": self.atlanan_uzun, "satir": self.satir,
            "token_tahmini": self.token_tahmini,
            "kategoriler": self.kategoriler,
        }


def _dosya_ozeti(yol: Path) -> str:
    """Dosya içeriğinin özeti (tekrar tespiti için)."""
    try:
        h = hashlib.md5()
        h.update(yol.read_bytes())
        return h.hexdigest()
    except Exception:
        return ""


def _fonksiyonlar(icerik: str) -> List[str]:
    """
    Bir `.gd` dosyasından **fonksiyon bloklarını** çıkarır.

    ⚠ Ölçülen: TitoWar'da 213 KB'lık tek dosyalar var — bunlar örnek
    olarak **kullanılamaz** (model tek örnekte 60 satırdan fazlasını
    öğrenemez). Fonksiyonlara bölmek gerekir.
    """
    satirlar = icerik.splitlines()
    bloklar: List[str] = []
    baslangic: Optional[int] = None
    for i, s in enumerate(satirlar):
        # Fonksiyon başlangıcı (girintisiz `func`)
        if re.match(r"^func\s+\w+", s):
            if baslangic is not None:
                bloklar.append("\n".join(satirlar[baslangic:i]).rstrip())
            baslangic = i
        # Çok uzun bloğu kes
        if (baslangic is not None
                and i - baslangic >= MAKS_SATIR):
            bloklar.append("\n".join(satirlar[baslangic:i + 1]).rstrip())
            baslangic = None
    if baslangic is not None:
        bloklar.append("\n".join(satirlar[baslangic:]).rstrip())
    return [b for b in bloklar if b.strip()]


def _kategori(yol: Path) -> str:
    """Dosya yolundan **konu kategorisi** çıkarır."""
    p = str(yol).lower()
    for anahtar, kat in (
        ("player", "karakter"), ("character", "karakter"),
        ("npc", "karakter"), ("enemy", "dusman"), ("monster", "dusman"),
        ("inventory", "envanter"), ("item", "esya"),
        ("ui", "arayuz"), ("menu", "arayuz"), ("hud", "arayuz"),
        ("network", "ag"), ("server", "ag"), ("client", "ag"),
        ("terrain", "arazi"), ("world", "dunya"), ("map", "dunya"),
        ("spell", "buyu"), ("skill", "yetenek"), ("combat", "savas"),
        ("quest", "gorev"), ("dialog", "diyalog"),
        ("save", "kayit"), ("config", "ayar"), ("input", "girdi"),
    ):
        if anahtar in p:
            return kat
    return "genel"


def analiz(maks: int = 100000) -> MadenRaporu:
    """Kaynağı analiz eder."""
    r = MadenRaporu()
    if not KAYNAK.exists():
        print(f"  kaynak yok: {KAYNAK}")
        return r
    gorulen: Set[str] = set()
    for yol in sorted(KAYNAK.rglob("*.gd"))[:maks]:
        r.dosya += 1
        ozet = _dosya_ozeti(yol)
        if ozet in gorulen:
            continue
        gorulen.add(ozet)
        r.tekil_dosya += 1
        try:
            icerik = yol.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for b in _fonksiyonlar(icerik):
            satir = len(b.splitlines())
            if satir < 3:
                r.atlanan_kisa += 1
                continue
            r.ornek += 1
            r.satir += satir
            r.token_tahmini += len(b) // 3
            kat = _kategori(yol)
            r.kategoriler[kat] = r.kategoriler.get(kat, 0) + 1
    return r


def topla(maks: int = 3000, *, hedef: Path = Path(
        ".neurocore/titowar_kod.jsonl")) -> MadenRaporu:
    """
    Gerçek Godot kodu toplar ve kasaya/dosyaya yazar.

    ⚠ Yorum satırları **temizlenmez** — gerçek kodda yorumlar
    yapısal bilgi taşır. Ama `kod_disi_var` denetiminden geçirilir
    (İngilizce düz yazı ayıklanır).
    """
    r = MadenRaporu()
    if not KAYNAK.exists():
        print(f"  kaynak yok: {KAYNAK}")
        return r

    # Prose ayıklayıcı
    try:
        from tools.kod_son_isleme import SonIsleme

        son = SonIsleme()
    except Exception:
        son = None

    gorulen_dosya: Set[str] = set()
    gorulen_blok: Set[str] = set()
    hedef.parent.mkdir(parents=True, exist_ok=True)

    with hedef.open("w", encoding="utf-8") as f:
        for yol in sorted(KAYNAK.rglob("*.gd")):
            if r.ornek >= maks:
                break
            r.dosya += 1
            ozet = _dosya_ozeti(yol)
            if ozet in gorulen_dosya:
                continue
            gorulen_dosya.add(ozet)
            r.tekil_dosya += 1
            try:
                icerik = yol.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            kat = _kategori(yol)
            for b in _fonksiyonlar(icerik):
                if r.ornek >= maks:
                    break
                satir = len(b.splitlines())
                if satir < 3:
                    r.atlanan_kisa += 1
                    continue
                if len(b) < 60 or len(b) > 4000:
                    r.atlanan_uzun += 1
                    continue
                anahtar = b.strip()[:100]
                if anahtar in gorulen_blok:
                    continue
                # ⚠ tur 34: `arayuz`/`genel` de alınıyor — modelin
                # Godot kalıplarını görmesi için. Prose denetimi
                # **gevşetildi** (yalnız çok belirgin prose atılır).
                if son is not None:
                    d, _ = son.kod_disi_var(b)
                    if d and len(b.splitlines()) < 8:
                        continue
                gorulen_blok.add(anahtar)
                r.ornek += 1
                r.satir += satir
                r.token_tahmini += len(b) // 3
                r.kategoriler[kat] = r.kategoriler.get(kat, 0) + 1
                f.write(json.dumps({
                    "metin": b + "\n\n",
                    "kategori": "godot_gercek",
                    "alt_kategori": kat,
                    "kaynak": str(yol.relative_to(KAYNAK))[:120],
                }, ensure_ascii=False) + "\n")
    return r


def main() -> int:
    """Madencilik işlemlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="Gerçek Godot kodu madencisi")
    ap.add_argument("--analiz", action="store_true")
    ap.add_argument("--topla", action="store_true")
    ap.add_argument("--maks", type=int, default=3000)
    args = ap.parse_args()

    print("=" * 78)
    print("  GERÇEK GODOT KODU MADENCİSİ — TitoWar projesinden")
    print("=" * 78)
    print(f"  kaynak: {KAYNAK}")

    sonuc: Dict[str, Any] = {}
    if args.analiz or not args.topla:
        a = analiz()
        print(f"\n  dosya: {a.dosya}  tekil: {a.tekil_dosya}")
        print(f"  fonksiyon örneği: {a.ornek}")
        print(f"  satır: {a.satir:,}  token: ~{a.token_tahmini:,}")
        print(f"  atlanan (kısa): {a.atlanan_kisa}  "
              f"(uzun): {a.atlanan_uzun}")
        print(f"  kategoriler: {json.dumps(a.kategoriler, ensure_ascii=False)}")
        sonuc["analiz"] = a.to_dict()
    if args.topla:
        r = topla(maks=args.maks)
        print(f"\n  TOPLANDI: {r.ornek} örnek · {r.satir:,} satır · "
              f"~{r.token_tahmini:,} token")
        print(f"  kategoriler: {json.dumps(r.kategoriler, ensure_ascii=False)}")
        print(f"  dosya: .neurocore/titowar_kod.jsonl")
        sonuc["toplama"] = r.to_dict()

    Path("_titowar_kod.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
