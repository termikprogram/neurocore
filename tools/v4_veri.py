"""
v4 Veri Üretici — Prose'u Yoruma Çevir (Bağlamı Koru)
=======================================================
⚠ ÖLÇÜLEN DERS (tur 31)
-----------------------
```
v2 (karışık veri) : kod tamamlama 8/14 · Godot derleme 5/5 · kayıp 1,0289
v3 (saf kod)      : kod tamamlama 8/14 · Godot derleme 1/5 · kayıp 1,5696
```

**Prose'u tamamen atmak zarar verdi:**
    1. Veri hacmi **%61 düştü** (2,4M → 937k token)
    2. **Bağlam kaybı** — `**İmza:**` ve `**Kullanım:**` alanları API'nin
       *ne olduğunu* öğretiyordu
    3. Anlamsız kelime üretimi başladı
       (`labirene_ureter_dubere_uretan_ureti_labirne`)

Doğru yaklaşım: **dönüştür, atma**
----------------------------------
    ÖNCE (prose — model bunu kod yerine üretiyor):
        '**Açıklama:** Sahneye Bézier eğrisinden oluşan bir daire ekler.'

    SONRA (kod yorumu — model kodu öğrenir, prose değil):
        '# Sahneye Bézier eğrisinden oluşan bir daire ekler'

Yapı
----
    ┌────────────────────────────────────────┐
    │ ## Blender API (resmî doküman)          │  ← başlık at
    │ **Ad:** `primitive_cube_add`            │  ← koda çevir
    │ **İmza:** `...`                         │
    │ **Kullanım:**                           │
    │ ```python                               │
    │ bpy.ops.mesh.primitive_cube_add()       │  ← KORU (kod)
    │ ```                                     │
    └────────────────────────────────────────┘

    →
    # API: primitive_cube_add
    # İmza: bpy.ops.mesh.primitive_cube_add(size=2.0)
    bpy.ops.mesh.primitive_cube_add()

Kullanım
--------
    python tools/v4_veri.py --uret
    python tools/v4_veri.py --ornek
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

V4_VERI = Path(".neurocore/v4_veri.jsonl")

#: Kod kategorileri
KOD_KATEGORI = ("kod_uretim", "kod_ogrenme", "kod_ornek",
                "blender_api", "api_referans")

#: Kod bloğu deseni
KOD_BLOK = re.compile(r"```(?:python|gdscript|py|gd)?\s*\n(.*?)```",
                      re.DOTALL)

#: Yapılandırılmış alanlar → yorum önekine eşleme
ALAN_ESLEME: Sequence[Tuple[str, str]] = (
    ("Ad", "API"),
    ("İmza", "İmza"),
    ("Imza", "İmza"),
    ("Parametreler", "Parametreler"),
    ("Açıklama", "Ne yapar"),
    ("Aciklama", "Ne yapar"),
    ("Konu", "Konu"),
    ("Önek", "Önek"),
    ("Onek", "Önek"),
    ("Devam", "Devam"),
    ("Kullanım", "Kullanım"),
    ("Kullanim", "Kullanım"),
    ("Modül", "Modül"),
    ("Modul", "Modül"),
    ("Kaynak", "Kaynak"),
)

#: Alan satırını yakalayan desen
ALAN_DESEN = re.compile(
    r"^\s*\*\*(Ad|İmza|Imza|Parametreler|Açıklama|Aciklama|Konu|"
    r"Önek|Onek|Devam|Kullanım|Kullanim|Modül|Modul|Kaynak|"
    r"Tam kod):?\*\*\s*(.*)$",
    re.MULTILINE)


@dataclass
class V4Rapor:
    """v4 veri üretim raporu."""
    not_toplam: int = 0
    kod_blok: int = 0
    alan_yorum: int = 0
    satir: int = 0
    token_tahmini: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "not_toplam": self.not_toplam, "kod_blok": self.kod_blok,
            "alan_yorum": self.alan_yorum, "satir": self.satir,
            "token_tahmini": self.token_tahmini,
        }


def _yorum_satiri(etiket: str, deger: str) -> str:
    """
    Yapılandırılmış alanı **kod yorumuna** çevirir.

    ⚠ Ölçülen: `**Açıklama:** ...` satırı prose olarak öğreniliyordu.
    Yorum öneki (`#`) eklenince model bunu **kod bağlamı** olarak görür
    ama **kod olarak üretmez** (yorum satırı üretmek için `#` gerekir,
    model bunu nadiren yapar).
    """
    d = deger.strip().strip("`").strip()
    if not d:
        return ""
    # Çok uzun değerleri kısalt
    if len(d) > 160:
        d = d[:157] + "..."
    # Çok satırlı değerleri birleştir
    d = re.sub(r"\s+", " ", d)
    # ⚠ Yer tutucu (`<<<KOD>>>`) yorum satırına girmemeli — kod
    # blokları zaten ayrıca ekleniyor. Yalnız yer tutucuysa atla.
    if d.strip() == "<<<KOD>>>":
        return ""
    return f"# {etiket}: {d}"


def notu_v4_yap(govde: str) -> str:
    """
    Bir notu **v4 biçimine** çevirir.

    Akış:
        1. Kod bloklarını bul → koru
        2. Yapılandırılmış alanları → `# Etiket: değer` yorumu yap
        3. Markdown başlıkları → `# Başlık` yorumu
        4. Kod dışı serbest prose → **at** (yalnız yorumlar + kod kalır)
    """
    # ── 1) Kod bloklarını çıkar ──
    bloklar = [b.strip() for b in KOD_BLOK.findall(govde)
               if len(b.strip()) > 15]
    # Kod bloklarını metinden çıkar (tekrar işlenmesin)
    kalan = KOD_BLOK.sub("\n<<<KOD>>>\n", govde)

    # ── 2) Alanları yoruma çevir ──
    basliklar: List[str] = []
    for m in ALAN_DESEN.finditer(kalan):
        etiket = m.group(1)
        deger = m.group(2)
        tr = dict(ALAN_ESLEME).get(etiket, etiket)
        y = _yorum_satiri(tr, deger)
        if y:
            basliklar.append(y)

    # ── 3) Markdown başlıkları ──
    for m in re.finditer(r"^\s*#{1,6}\s+(.+)$", kalan, re.MULTILINE):
        t = m.group(1).strip()
        # Zaten alan satırıysa atla
        if any(t.startswith(f"**{a}") for a, _ in ALAN_ESLEME):
            continue
        if len(t) < 80:
            basliklar.append(f"# {t}")

    # ── 4) Birleştir: yorumlar + kod ──
    parcalar: List[str] = []
    if basliklar:
        # Tekrarları at, sırayı koru
        gorulen: set = set()
        benzersiz: List[str] = []
        for b in basliklar:
            if b not in gorulen:
                gorulen.add(b)
                benzersiz.append(b)
        parcalar.append("\n".join(benzersiz[:8]))
    if bloklar:
        parcalar.append("\n\n".join(bloklar))

    if not parcalar:
        return ""
    return "\n\n".join(parcalar).strip()


def uret(maks_not: int = 8000, *, kod_tekrar: int = 3) -> V4Rapor:
    """v4 verisini üretir."""
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return V4Rapor()

    r = V4Rapor()
    with V4_VERI.open("w", encoding="utf-8") as f:
        for n in k.vault.iter_notes(limit=maks_not):
            govde = (getattr(n, "body", "") or "").strip()
            kat = getattr(n, "category", "") or ""
            if len(govde) < 40:
                continue
            r.not_toplam += 1
            v4 = notu_v4_yap(govde)
            if not v4 or len(v4) < 30:
                continue
            r.kod_blok += 1
            r.alan_yorum += v4.count("\n#") + (1 if v4.startswith("#") else 0)
            r.satir += len(v4.splitlines())
            r.token_tahmini += len(v4) // 3
            tekrar = kod_tekrar if kat in KOD_KATEGORI else 1
            for _ in range(tekrar):
                f.write(json.dumps({"metin": v4 + "\n\n", "kategori": kat},
                                   ensure_ascii=False) + "\n")
    k.shutdown()
    return r


def ornek_goster(maks: int = 3) -> None:
    """Örnek dönüşümleri gösterir."""
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return
    sayi = 0
    for n in k.vault.iter_notes(limit=8000):
        if sayi >= maks:
            break
        kat = getattr(n, "category", "") or ""
        if kat not in ("blender_api", "kod_ornek"):
            continue
        govde = (getattr(n, "body", "") or "").strip()
        if len(govde) < 100:
            continue
        sayi += 1
        print(f"\n{'=' * 74}")
        print(f"  ÖRNEK {sayi} — {kat} · {getattr(n, 'title', '')[:50]}")
        print(f"{'=' * 74}")
        print("  --- ÖNCE ---")
        for s in govde.splitlines()[:10]:
            print(f"    {s[:72]}")
        v4 = notu_v4_yap(govde)
        print("  --- SONRA (v4) ---")
        for s in v4.splitlines()[:10]:
            print(f"    {s[:72]}")
    k.shutdown()


def main() -> int:
    """v4 veri işlemlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="v4 veri üretici")
    ap.add_argument("--uret", action="store_true")
    ap.add_argument("--ornek", action="store_true")
    ap.add_argument("--maks-not", type=int, default=8000)
    args = ap.parse_args()

    print("=" * 78)
    print("  v4 VERİ — prose'u yoruma çevir (bağlamı koru)")
    print("=" * 78)

    sonuc: Dict[str, Any] = {}
    if args.ornek:
        ornek_goster()
    if args.uret:
        r = uret(maks_not=args.maks_not)
        print(f"\n  not: {r.not_toplam}")
        print(f"  dönüştürülen: {r.kod_blok}")
        print(f"  yorum satırı: {r.alan_yorum}")
        print(f"  toplam satır: {r.satir}")
        print(f"  token tahmini: ~{r.token_tahmini:,}")
        print(f"  dosya: {V4_VERI}")
        sonuc["uretim"] = r.to_dict()

    Path("_v4_veri.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
