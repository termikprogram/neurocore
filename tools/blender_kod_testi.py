"""
Hedef F kanıtı — myAi **kendi başına** Blender kodu yazabiliyor mu?
====================================================================
⚠ Soru: 54 derslik müfredat + 76 tamamlama çifti **işe yaradı mı**?

Ölçüm
-----
```
1. Modelden **yeni** bir Blender görevi istenir (müfredatta YOK)
2. Üretilen kod Blender'da **çalıştırılır**
3. Nesne/malzeme **ölçülür** (disk'ten, iddiadan değil)
```

⚠ Görevler müfredattaki örneklerden **farklı** seçilir:
```
· bevel (pah) — müfredatta `modifier` var ama bevel örneği yok
· emission (ışıma) — malzeme dersinde yalnız BSDF var
· lattice (kafes) — hiç geçmiyor
```

Kullanım
--------
    python tools/blender_kod_testi.py --uret
    python tools/blender_kod_testi.py --calistir
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

BLENDER = Path(r"E:\AppsTools\Blender 5.2\blender.exe")

#: ⚠ Görevler — müfredatta **örneği olmayan** konular
GOREVLER: List[Dict[str, Any]] = [
    {
        "ad": "bevel_kup",
        "istek": (
            "Blender Python: 2x2x2 boyutunda bir küp oluştur, sahnedeki "
            "diğer her şeyi sil, küpe 0.15 genişliğinde ve 3 segmentli "
            "BEVEL modifier ekle ve modifier'ı UYGULA (apply). Sonra "
            "objenin adını 'Pahli_Kup' yap."
        ),
        "beklenen": {"nesne": "Pahli_Kup", "yuz_min": 50},
    },
    {
        "ad": "isima_malzeme",
        "istek": (
            "Blender Python: Yeni bir malzeme oluştur, adı 'Isik_M'. "
            "Principled BSDF'in 'Emission Color' değerini kırmızı "
            "(1, 0, 0, 1) ve 'Emission Strength' değerini 5.0 yap. "
            "Malzemeyi aktif objeye ata."
        ),
        "beklenen": {"malzeme": "Isik_M", "emission": 5.0},
    },
    {
        "ad": "kafes_deform",
        "istek": (
            "Blender Python: Bir plane (düzlem) oluştur, ona KAFES "
            "(Lattice) modifier ekle. Kafes nesnesinin X ölçeğini 2.0 "
            "yap ve kafesin ilk noktasını (point) X yönünde 0.5 "
            "kaydır."
        ),
        "beklenen": {"lattice": True},
    },
]

#: ⚠ Blender tarafında çalıştırma + ölçüm
CALISTIR = r'''
import bpy, json, sys

try:
    exec(compile(KOD_V, "<uretilen>", "exec"), {"bpy": bpy, "__name__": "__main__"})
    olcum = {"calisti": True}
except Exception as exc:
    print("K" + json.dumps({"calisti": False,
        "hata": f"{type(exc).__name__}: {str(exc)[:150]}"}))
    raise SystemExit(0)

# ⚠ Ölçüm: diskten değil, sahneden
olcum["nesneler"] = [o.name for o in bpy.data.objects if o.type == "MESH"]
olcum["malzemeler"] = [m.name for m in bpy.data.materials]
olcum["modifier"] = {}
for o in bpy.data.objects:
    if o.modifiers:
        olcum["modifier"][o.name] = [m.type for m in o.modifiers]
# ⚠ Yüz sayısı (bevel kontrolü)
olcum["yuz"] = {}
for o in bpy.data.objects:
    if o.type == "MESH":
        olcum["yuz"][o.name] = len(o.data.polygons)
# ⚠ Emission kontrolü
olcum["emission"] = {}
for m in bpy.data.materials:
    if not m.use_nodes:
        continue
    for n in m.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            try:
                e = n.inputs["Emission Strength"].default_value
                olcum["emission"][m.name] = round(float(e), 3)
            except Exception:
                pass
# ⚠ Lattice kontrolü
olcum["lattice"] = [o.name for o in bpy.data.objects if o.type == "LATTICE"]
print("K" + json.dumps(olcum, ensure_ascii=False))
'''


def _kod_temizle(metin: str) -> str:
    """Model çıktısından kodu ayıklar."""
    if not metin:
        return ""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", metin, re.S)
    if m:
        return m.group(1).strip()
    # ⚠ Kod bloğu yoksa: `import bpy` ile başlayan kısmı al
    i = metin.find("import bpy")
    if i >= 0:
        return metin[i:].strip()
    return metin.strip()


def kod_uret(gorev: Dict[str, Any]) -> Dict[str, Any]:
    """Modelden Blender kodu ister."""
    _kok = str(_KOK)
    if _kok not in sys.path:
        sys.path.insert(0, _kok)
    from neurocore.agent.coding import CodingAgent

    a = CodingAgent()
    # ⚠ `kod_yaz` aracı varsa onu kullan
    for ad in ("kod_yaz", "kod_uret", "yaz"):
        f = getattr(a, ad, None)
        if f is None:
            continue
        try:
            r = f(gorev["istek"])
            if isinstance(r, dict):
                kod = r.get("kod") or r.get("metin") or ""
                if kod:
                    return {"basarili": True, "kod": _kod_temizle(kod),
                            "ham": str(kod)[:400]}
            elif isinstance(r, str) and r:
                return {"basarili": True, "kod": _kod_temizle(r),
                        "ham": r[:400]}
        except Exception as exc:
            return {"basarili": False, "hata": f"{ad}: {type(exc).__name__}"}
    return {"basarili": False, "hata": "kod üretim aracı yok"}


def calistir(kod: str, zaman: int = 240) -> Dict[str, Any]:
    """Kodu Blender'da çalıştırır ve ölçer."""
    import subprocess
    import tempfile

    if not BLENDER.exists():
        return {"hata": "blender yok"}
    if not kod.strip():
        return {"hata": "kod boş"}
    tam = f"KOD_V = {kod!r}\n" + CALISTIR
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "k.py"
        f.write_text(tam, encoding="utf-8")
        try:
            p = subprocess.run(
                [str(BLENDER), "--background", "--factory-startup",
                 "--python", str(f)],
                capture_output=True, text=True, timeout=zaman,
                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return {"hata": "zaman aşımı"}
    out = (p.stdout or "") + (p.stderr or "")
    for satir in out.splitlines():
        if "K{" in satir:
            try:
                return json.loads(satir[satir.index("K{") + 1:])
            except Exception:
                pass
    iz = [x.strip() for x in out.splitlines()
          if "Error" in x or "error" in x]
    return {"hata": "; ".join(iz[:2])[:170] or "çıktı yok"}


def puanla(gorev: Dict[str, Any], olcum: Dict[str, Any]) -> Dict[str, Any]:
    """Beklenen ile ölçümü karşılaştırır."""
    b = gorev.get("beklenen", {})
    ok: List[str] = []
    kt: List[str] = []
    nesneler = olcum.get("nesneler") or []
    if "nesne" in b:
        if b["nesne"] in nesneler:
            ok.append(f"nesne {b['nesne']} ✓")
        else:
            kt.append(f"nesne {b['nesne']} yok (var: {nesneler})")
    if "yuz_min" in b:
        y = (olcum.get("yuz") or {}).get(b.get("nesne", ""), 0)
        if y >= b["yuz_min"]:
            ok.append(f"yüz {y} ≥ {b['yuz_min']} ✓")
        else:
            kt.append(f"yüz {y} < {b['yuz_min']}")
    if "malzeme" in b:
        if b["malzeme"] in (olcum.get("malzemeler") or []):
            ok.append(f"malzeme {b['malzeme']} ✓")
        else:
            kt.append(f"malzeme {b['malzeme']} yok")
    if "emission" in b:
        e = (olcum.get("emission") or {}).get(b.get("malzeme", ""), 0)
        if abs(e - b["emission"]) < 0.01:
            ok.append(f"emission {e} ✓")
        else:
            kt.append(f"emission {e} ≠ {b['emission']}")
    if b.get("lattice"):
        if olcum.get("lattice"):
            ok.append(f"lattice {olcum['lattice']} ✓")
        else:
            kt.append("lattice nesnesi yok")
    return {"basarili": not kt, "ok": ok, "eksik": kt}


def main() -> int:
    """Kod üretimi + çalıştırma testi."""
    import argparse

    ap = argparse.ArgumentParser(description="Blender kod üretim testi")
    ap.add_argument("--gorev", default="", help="Tek görev adı")
    ap.add_argument("--kod-dosya", default="",
                    help="Kod dosyasından oku (model atlanır)")
    args = ap.parse_args()

    print("=" * 78)
    print("  HEDEF F KANITI — myAi KENDİ BAŞINA BLENDER KODU YAZABİLİYOR MU?")
    print("=" * 78)

    gorevler = ([g for g in GOREVLER if g["ad"] == args.gorev]
                if args.gorev else GOREVLER)
    sonuc: Dict[str, Any] = {"gorev": [], "basarili": 0}

    for g in gorevler:
        print(f"\n  ── {g['ad']} " + "─" * (54 - len(g["ad"])))
        t0 = time.perf_counter()
        if args.kod_dosya:
            kod = Path(args.kod_dosya).read_text(encoding="utf-8")
            uretim = {"basarili": True, "kod": kod, "ham": "(dosyadan)"}
        else:
            uretim = kod_uret(g)
        if not uretim.get("basarili"):
            print(f"     ✗ üretim: {uretim.get('hata')}")
            sonuc["gorev"].append({"ad": g["ad"], "basarili": False,
                                   "hata": uretim.get("hata")})
            continue
        sure_uret = time.perf_counter() - t0
        n_kod = len(uretim["kod"].splitlines())
        print(f"     kod: {n_kod} satır · {sure_uret:.1f} sn")
        o = calistir(uretim["kod"])
        if o.get("hata"):
            # ⚠ Çalışmadıysa hata mesajını göster
            print(f"     ✗ çalıştırma: {o['hata'][:100]}")
            sonuc["gorev"].append({"ad": g["ad"], "basarili": False,
                                   "hata": o["hata"], "kod": uretim["kod"]})
            continue
        p = puanla(g, o)
        im = "✓" if p["basarili"] else "~"
        print(f"     {im} ölçüm: {', '.join(p['ok']) or '-'}")
        if p["eksik"]:
            print(f"       eksik: {', '.join(p['eksik'])}")
        if p["basarili"]:
            sonuc["basarili"] += 1
        sonuc["gorev"].append({"ad": g["ad"], "basarili": p["basarili"],
                               "ok": p["ok"], "eksik": p["eksik"],
                               "kod": uretim["kod"], "olcum": o})

    n = len(gorevler)
    print("\n" + "=" * 78)
    print(f"  SONUÇ: {sonuc['basarili']}/{n} görev başarılı")
    print("=" * 78)
    Path("_blender_kod_testi.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if sonuc["basarili"] == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
