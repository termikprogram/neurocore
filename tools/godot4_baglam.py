"""
Godot 4 API Bağlam Enjektörü (retrieval) — tur 21
====================================================
⚠ ÖLÇÜLEN DURUM (tur 20)
------------------------
Hedefe yönelik ince ayar **çalışmadı**:
```
frequency      p=0.00003  |  Seed         p=0.14160   ← YANLIŞ 4.700× olası
albedo_color   p=0.00000  |  ambientColor p=0.61328   ← YANLIŞ BASKIN
```

**Sebep:** Temel model ~18T token görmüş; bizim 750k token'imiz
**1:24.000.000** oranında. Godot 3 bilgisi **24 katmanın tamamına**
yayılmış, biz yalnız **2 katmanı** değiştirebiliyoruz.

Ayrıca daha çok katman eğitmek **CUDA hatası** veriyor:
```
--dondur 0.6 (10 katman, 285M) : adım 1000'de CUDA error
--dondur 0.9 ( 2 katman, 181M) : 2.000 adım hatasız ✓
```

Çözüm: **Model yazmasın, sorsun** — retrieval
----------------------------------------------
Modelin **ağırlığını** değiştirmek yerine **bağlamını** değiştir:

```
SORGU   : "Godot 4 FastNoiseLite materials"
        ↓
İNDEKS  : 2.003 imza · 558 sınıf (dokümandan, %100 doğru)
        ↓
BAĞLAM  : "Godot 4 API referansı:
           - FastNoiseLite.frequency : float
           - StandardMaterial3D.albedo_color : Color
           ..."
        ↓
MODEL   : bağlamı kullanarak kod yazar
```

⚠ Neden bu çalışır? Modele **"frequency"** token'ını **veriyoruz**;
üretmesini beklemiyoruz.

Kullanım
--------
    python tools/godot4_baglam.py --istatistik
    python tools/godot4_baglam.py --sorgu "FastNoiseLite malzeme"
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

#: Godot 4 API indeksi (tur 18'de dokümandan üretildi)
INDEKS = Path(".neurocore/godot4_api.jsonl")

#: ⚠ Godot 3 → 4 uyarı tablosu (bağlama eklenir)
G3_UYARI: Sequence[Tuple[str, str]] = (
    ("ambientColor", "albedo_color"), ("diffuseColor", "albedo_color"),
    ("albedoColor", "albedo_color"), ("Color3(", "Color("),
    ("SpatialMaterial", "StandardMaterial3D"), ("Spatial", "Node3D"),
    ("KinematicBody", "CharacterBody3D"),
    ("MeshInstance\n", "MeshInstance3D"), ("Camera\n", "Camera3D"),
    ("Area\n", "Area3D"), ("yield(", "await ("), (".instance()",
                                                   ".instantiate()"),
    ("OS.get_ticks_msec", "Time.get_ticks_msec"),
    ("PoolByteArray", "PackedByteArray"), ("PoolIntArray", "PackedInt32Array"),
    ("PoolStringArray", "PackedStringArray"),
    ("PoolVector3Array", "PackedVector3Array"),
    ("OpenSimplexNoise", "FastNoiseLite"), ("SetNoiseScale(", "noise_scale = "),
    ("set_", ""), ("Directory", "DirAccess"), ("File.new()", "FileAccess.open"),
)


@dataclass
class Indeks:
    """Godot 4 API indeksi."""
    uyeler: List[Tuple[str, str, str]] = field(default_factory=list)
    siniflar: Dict[str, List[Tuple[str, str, str]]] = field(
        default_factory=dict)
    yuklendi: bool = False

    def yukle(self, yol: Path = INDEKS) -> bool:
        """İndeksi yükler."""
        if self.yuklendi or not yol.exists():
            return self.yuklendi
        with yol.open(encoding="utf-8") as f:
            for satir in f:
                try:
                    d = json.loads(satir)
                except Exception:
                    continue
                sinif = d.get("sinif", "")
                uye = d.get("uye", "")
                if not sinif or not uye:
                    continue
                mt = re.search(r"\*\*Tür:\*\*\s*(\w+)(?:\s*→\s*(\w+))?",
                               d.get("metin", ""))
                tur = mt.group(1) if mt else "?"
                tip = (mt.group(2) or "") if mt else ""
                self.siniflar.setdefault(sinif, []).append((uye, tip, tur))
                self.uyeler.append((sinif, uye, tip))
        self.yuklendi = True
        return True

    def istatistik(self) -> Dict[str, Any]:
        """İndeks istatistiği."""
        self.yukle()
        return {"sinif": len(self.siniflar), "uye": len(self.uyeler)}


#: ─── Retrieval ────────────────────────────────────────────────────────

def _puan(sinif: str, uye: str, tip: str, kelimeler: Sequence[str]) -> int:
    """Bir üyenin sorguya uygunluk puanı."""
    p = 0
    sl, ul, tl = sinif.lower(), uye.lower(), tip.lower()
    for k in kelimeler:
        if k == sl:
            p += 10
        elif k in sl:
            p += 5
        if k == ul:
            p += 8
        elif k in ul:
            p += 4
        if k == tl:
            p += 2
    return p


def baglam_uret(sorgu: str, *, sinir: int = 24,
                indeks: Indeks = None) -> str:
    """
    Sorguya göre **Godot 4 API bağlamı** üretir.

    ⚠ Çıktı modele **doğrudan** enjekte edilir; modelin API adını
    **üretmesi beklenmez**.
    """
    idx = indeks if indeks is not None else Indeks()
    idx.yukle()
    kelimeler = [k for k in re.split(r"[^a-z0-9_]+", sorgu.lower())
                 if len(k) > 2]
    # ⚠ tur 21 ÖLÇÜMÜ: aynı (sınıf, üye, tip) **tekrar ediyordu**
    # (tur 18 verisi her imza için 2 kayıt üretmişti). Tekilleştir.
    puanli: List[Tuple[int, str, str, str]] = []
    gorulen: set = set()
    for sinif, uye, tip in idx.uyeler:
        anahtar = (sinif, uye, tip)
        if anahtar in gorulen:
            continue
        gorulen.add(anahtar)
        p = _puan(sinif, uye, tip, kelimeler)
        if p > 0:
            puanli.append((p, sinif, uye, tip))
    # ⚠ Sınıf adı sorguda geçmiyorsa (ör. "renk") **hiç** sonuç
    # çıkmıyordu. Türkçe eş anlamlıları da dene.
    if not puanli:
        ek = {"renk": "color", "malzeme": "material", "hareket": "move",
              "gürültü": "noise", "bekleme": "wait", "zaman": "time",
              "kamera": "camera", "ışık": "light", "ses": "audio",
              "doku": "texture", "animasyon": "animation",
              "fizik": "physics", "girdi": "input"}
        ek_kelimeler = [ek.get(k, k) for k in kelimeler]
        for sinif, uye, tip in idx.uyeler:
            if _puan(sinif, uye, tip, ek_kelimeler) > 0:
                puanli.append((1, sinif, uye, tip))
    puanli.sort(key=lambda x: (-x[0], x[1], x[2]))
    secili = puanli[:sinir]

    satirlar = [
        "# Godot 4 API REFERANSI (dokümandan, %100 doğru)",
        "# ⚠ Godot 4 adlarını kullan — Godot 3 adları GEÇERSİZ:",
    ]
    for g3, g4 in G3_UYARI[:12]:
        if g4:
            satirlar.append(f"#   {g3!r} → {g4!r}")
    satirlar.append("#")
    for _, sinif, uye, tip in secili:
        if tip:
            satirlar.append(f"#   {sinif}.{uye} : {tip}")
        else:
            satirlar.append(f"#   {sinif}.{uye}")
    # Seçili sınıfların diğer üyeleri (küçük örneklem)
    if secili:
        ana = secili[0][1]
        digerleri = [u for u, _, _ in idx.siniflar.get(ana, [])][:14]
        satirlar.append("#")
        satirlar.append(f"# {ana} üyeleri: {', '.join(digerleri)}")
    # ⚠ tur 22: GERÇEK KOD ÖRNEKLERİ (Godot derleyicisiyle doğrulanmış)
    ornekler = _ornek_getir(kelimeler, sinir=3)
    if ornekler:
        satirlar.append("#")
        satirlar.append("# Godot 4 KULLANIM ÖRNEKLERİ "
                        "(derleyici ile doğrulanmış):")
        for kod in ornekler:
            satirlar.append("```gdscript")
            satirlar.extend(kod.splitlines()[:16])
            satirlar.append("```")
    return "\n".join(satirlar) + "\n"


#: ⚠ Gerçek kod örnekleri (doğrulanmış korpus + doküman blokları)
ORNEK_DOSYALARI = (".neurocore/godot4_korpus.jsonl",
                   ".neurocore/godot4_ornekler.jsonl")

_ORNEK_ONBELLEK: List[Dict[str, Any]] = []


def _ornek_getir(kelimeler: Sequence[str], *, sinir: int = 3
                 ) -> List[str]:
    """Sorguya en uygun **gerçek kod örneklerini** getirir."""
    global _ORNEK_ONBELLEK
    if not _ORNEK_ONBELLEK:
        for yol in ORNEK_DOSYALARI:
            p = Path(yol)
            if not p.exists():
                continue
            with p.open(encoding="utf-8") as f:
                for satir in f:
                    try:
                        d = json.loads(satir)
                    except Exception:
                        continue
                    kod = d.get("kod") or d.get("metin", "")
                    if len(kod) >= 80:
                        _ORNEK_ONBELLEK.append(
                            {"kod": kod,
                             "kelimeler": d.get("kelimeler", [])})
    puanli: List[Tuple[int, str]] = []
    for o in _ORNEK_ONBELLEK:
        p = sum(2 for k in kelimeler
                if k in o["kelimeler"] or k in o["kod"].lower())
        if p > 0:
            puanli.append((p, o["kod"]))
    puanli.sort(key=lambda x: -x[0])
    return [k for _, k in puanli[:sinir]]


def kod_uret(sorgu: str, onek: str, *, sinir: int = 24,
             maks_token: int = 70, model: str = "models/qwen_temiz2"
             ) -> Tuple[str, str]:
    """
    Bağlam enjekte edilmiş kod üretir.

    Dönüş: (bağlam, üretilen kod)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    baglam = baglam_uret(sorgu, sinir=sinir)
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(cihaz)
    mdl.eval()
    tam = baglam + "\n" + onek
    ids = tok(tam, return_tensors="pt")["input_ids"].to(cihaz)
    with torch.no_grad():
        c = mdl.generate(ids, max_new_tokens=maks_token, do_sample=False,
                         pad_token_id=tok.eos_token_id,
                         no_repeat_ngram_size=3)
    u = tok.decode(c[0][ids.shape[1]:], skip_special_tokens=True)
    return (baglam, onek + u)


def main() -> int:
    """Bağlam üreticisini çalıştırır."""
    ap = argparse.ArgumentParser(description="Godot 4 bağlam enjektörü")
    ap.add_argument("--istatistik", action="store_true")
    ap.add_argument("--sorgu", default="")
    ap.add_argument("--sinir", type=int, default=24)
    args = ap.parse_args()

    print("=" * 78)
    print("  GODOT 4 API BAĞLAM ENJEKTÖRÜ (retrieval)")
    print("=" * 78)

    idx = Indeks()
    st = idx.istatistik()
    print(f"  indeks: {INDEKS}")
    print(f"  sınıf : {st['sinif']:,} · üye: {st['uye']:,}")

    if args.sorgu:
        print(f"\n  SORGU: {args.sorgu!r}\n")
        print(baglam_uret(args.sorgu, sinir=args.sinir))
    else:
        print("\n  ÖRNEK SORGULAR:")
        for q in ("FastNoiseLite gürültü", "StandardMaterial3D renk",
                  "CharacterBody3D hareket", "Tween animasyon",
                  "Timer bekleme"):
            b = baglam_uret(q, sinir=6)
            api = [s for s in b.splitlines() if s.startswith("#   ")][-6:]
            print(f"\n  {q!r}:")
            for s in api:
                print(f"    {s[4:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
