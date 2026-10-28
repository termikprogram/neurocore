"""
Bağımsız Benchmark — Ezber Değil, Genelleme Ölçümü
====================================================
⚠ ÖLÇÜLEN SORUN (tur 9)
-----------------------
40 soruluk benchmark'ta modeller:
```
qwen_coder_myai (v5, 2,1M token) : %70  ← EN İYİ
qwen_havuz2     (9,5M token)     : %65
qwen_karisim    (4,2M token)     : %62
qwen_alan       (12,5M token)    : %62
```

**Şüphe:** Daha fazla veriyle eğitilen modeller **daha kötü** —
bu, benchmark'ın **v5 eğitim verisiyle örtüşmesinden** olabilir
(model **ezberliyor**, genelleme yapmıyor).

Çözüm: **Bağımsız test seti**
------------------------------
    1. **Hiçbir modelin görmediği** kaynaklardan soru üret
       (türetilmiş, elle yazılmış, farklı API'ler)
    2. **Sızıntı denetimi** — her sorunun öneki eğitim verisinde
       var mı diye **ara**; varsa **at**
    3. Ölç ve karşılaştır

Sızıntı denetimi neden şart?
----------------------------
Eğitim verisi 23,2M token ve **yerel proje kodu** içeriyor. Bir soru
öneki (`import bpy\nbpy.ops.mesh.primitive_`) eğitimde **birebir**
geçiyorsa model onu **ezberlemiş** olabilir → ölçüm **yanıltıcı**.

Kullanım
--------
    python tools/bagimsiz_benchmark.py --sizinti
    python tools/bagimsiz_benchmark.py --kiyasla
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: ⚠ Bağımsız sorular — **türetilmiş**, eğitim verisinden uzak
#:
#: Tasarım ilkeleri:
#:     * Önekler **kısa ve genel** (tek API yolu değil)
#:     * Yanıtlar **mantıksal** (isim ezberi değil)
#:     * 3 dil: python · gdscript · javascript
#:     * Kendi alanımız (Godot) + genel (algoritma, veri yapısı)
SORULAR: Sequence[Tuple[str, Sequence[str], str, str]] = (
    # ── MANTIK (algoritma düşüncesi) ──
    ("def ikili_arama(dizi, hedef):\n    alt, ust = 0, len(dizi) - 1\n"
     "    while alt <= ust:\n        orta = (alt + ust) // 2\n"
     "        if dizi[orta] == hedef:\n            return ",
     ("orta", "alt", "-1"), "mantik", "python"),
    ("def faktoriyel(n):\n    if n <= 1:\n        return ",
     ("1",), "mantik", "python"),
    ("def en_buyuk(dizi):\n    en = dizi[0]\n    for x in dizi:\n"
     "        if x > en:\n            en = ",
     ("x",), "mantik", "python"),
    ("def ters_cevir(bas):\n    onceki = None\n    guncel = bas\n"
     "    while guncel:\n        sonraki = guncel.sonraki\n"
     "        guncel.sonraki = ",
     ("onceki", "None"), "mantik", "python"),
    ("def dengeli(metin):\n    yigin = []\n    for h in metin:\n"
     "        if h in '([{':\n            yigin.append(h)\n"
     "        elif h in ')]}':\n            if not yigin:\n                return ",
     ("False",), "mantik", "python"),
    ("def fib(n, notlar=None):\n    if notlar is None:\n        notlar = {}\n"
     "    if n in notlar:\n        return notlar[n]\n"
     "    if n < 2:\n        return ",
     ("n", "1", "0"), "mantik", "python"),
    ("import heapq\n\ndef en_buyuk_k(dizi, k):\n    yigin = dizi[:k]\n"
     "    heapq.heapify(yigin)\n    for x in dizi[k:]:\n"
     "        if x > yigin[0]:\n            heapq.",
     ("heapreplace", "heappush"), "mantik", "python"),
    ("def bfs(grafik, bas):\n    mesafe = {bas: 0}\n    kuyruk = [bas]\n"
     "    while kuyruk:\n        dugum = kuyruk.pop(0)\n"
     "        for komsu in grafik.get(dugum, []):\n"
     "            if komsu not in mesafe:\n                mesafe[komsu] = ",
     ("mesafe[dugum] + 1", "1"), "mantik", "python"),

    # ── MANTIK (algoritma düşüncesi — genişletildi) ──
    ("def ikili_arama(dizi, hedef):\n    alt, ust = 0, len(dizi) - 1\n"
     "    while alt <= ust:\n        orta = alt + (ust - alt) // 2\n"
     "        if dizi[orta] == hedef:\n            return orta\n"
     "        if dizi[orta] < hedef:\n            alt = ",
     ("orta + 1",), "mantik", "python"),
    ("def asal_mi(n):\n    if n < 2:\n        return False\n"
     "    i = 2\n    while i * i <= n:\n        if n % i == 0:\n"
     "            return ",
     ("False",), "mantik", "python"),
    ("def gcd(a, b):\n    while b:\n        a, b = b, ",
     ("a % b",), "mantik", "python"),
    ("def ters_metin(m):\n    return m[",
     ("::-1",), "mantik", "python"),
    ("def palindrom(m):\n    m = m.lower()\n    return m == ",
     ("m[::-1]",), "mantik", "python"),
    ("def en_kucuk(dizi):\n    en = dizi[0]\n    for x in dizi[1:]:\n"
     "        if x < en:\n            en = ",
     ("x",), "mantik", "python"),
    ("def toplam_dongu(n):\n    s = 0\n    for i in range(1, n + 1):\n"
     "        s += ",
     ("i",), "mantik", "python"),
    ("def sayi_ters(n):\n    s = 0\n    while n > 0:\n"
     "        s = s * 10 + n % 10\n        n //= ",
     ("10",), "mantik", "python"),
    ("def us_hizli(a, n):\n    if n == 0:\n        return 1\n"
     "    y = us_hizli(a, n // 2)\n    if n % 2 == 0:\n        return "
     "y * y\n    return ",
     ("a * y * y", "y * y * a"), "mantik", "python"),
    ("def kume_mi(dizi):\n    return len(dizi) == len(",
     ("set(dizi)", "set("), "mantik", "python"),
    ("def ikinci_en_buyuk(dizi):\n    a = b = float('-inf')\n"
     "    for x in dizi:\n        if x > a:\n            b, a = a, x\n"
     "        elif x > b and x != a:\n            b = ",
     ("x",), "mantik", "python"),
    ("def ardısık_mi(dizi):\n    dizi = sorted(set(dizi))\n"
     "    for i in range(1, len(dizi)):\n"
     "        if dizi[i] - dizi[i-1] != 1:\n            return ",
     ("False",), "mantik", "python"),
    # ── GODOT (kendi alanımız, ama farklı API'ler) ──
    ("extends Node2D\n\nfunc _ready() -> void:\n    var dizi := [3, 1, 2]\n"
     "    dizi.",
     ("sort",), "godot", "gdscript"),
    ("extends Node3D\n\nfunc _hazirla() -> void:\n"
     "    var kutu := BoxMesh.new()\n    kutu.size = ",
     ("Vector3",), "godot", "gdscript"),
    ("extends CharacterBody3D\n\nconst HIZ := 5.0\n\n"
     "func _physics_process(delta: float) -> void:\n"
     "    velocity.x = Input.get_axis(\"sol\", \"sag\") * ",
     ("HIZ",), "godot", "gdscript"),
    ("extends Node\n\nfunc _hazir() -> void:\n"
     "    var d := {\"a\": 1}\n    d.",
     ("has", "get", "erase", "size", "keys"), "godot", "gdscript"),
    ("extends Node3D\n\nfunc _olustur() -> void:\n"
     "    var mi := MeshInstance3D.new()\n    mi.",
     ("mesh", "position", "name", "material_override"),
     "godot", "gdscript"),
    ("extends Node\n\nsignal can_bitti\n\nfunc _hasar_al(n: int) -> void:\n"
     "    can -= n\n    if can <= 0:\n        can_bitti.",
     ("emit",), "godot", "gdscript"),
    ("extends Node3D\n\nfunc _kamera_kur() -> void:\n"
     "    var k := Camera3D.new()\n    add_child(k)\n    k.",
     ("look_at", "position", "current", "fov"), "godot", "gdscript"),
    ("extends Node\n\nfunc _kaydet(yol: String) -> void:\n"
     "    var f := FileAccess.open(yol, FileAccess.",
     ("WRITE", "READ"), "godot", "gdscript"),
    ("extends Node3D\n\nfunc _arazi() -> void:\n"
     "    var n := FastNoiseLite.new()\n    n.",
     ("frequency", "fractal_octaves", "noise_type", "seed"),
     "godot", "gdscript"),
    ("extends Node\n\nfunc _girdi() -> Vector2:\n"
     "    return Input.get_vector(",
     ("\"sol\"", "'sol'", "\"a\""), "godot", "gdscript"),
    ("extends Node3D\n\nfunc _malzeme() -> StandardMaterial3D:\n"
     "    var m := StandardMaterial3D.new()\n    m.",
     ("albedo_color", "roughness", "metallic", "emission"),
     "godot", "gdscript"),
    ("extends Node\n\nvar _sayac := 0\n\nfunc _process(delta: float) -> void:\n"
     "    _sayac += 1\n    if _sayac % 60 == 0:\n        ",
     ("print", "if", "_sayac", "pass"), "godot", "gdscript"),

    # ── BLENDER (kendi alanımız, farklı API'ler) ──
    ("import bpy\nfor ob in bpy.data.objects:\n    if ob.type == ",
     ("'MESH'", '"MESH"', "'CAMERA'"), "blender", "python"),
    ("import bpy\nbpy.ops.object.select_all(action=",
     ("'SELECT'", "'DESELECT'", "'TOGGLE'"), "blender", "python"),
    ("import mathutils\nv = mathutils.Vector((1, 2, 3))\nv.",
     ("normalized", "length", "x", "dot", "cross"), "blender", "python"),
    ("import bpy\nscene = bpy.context.scene\nscene.render.",
     ("engine", "resolution_x", "filepath", "fps"), "blender", "python"),
    ("import bpy\nmat = bpy.data.materials.new(name='M')\nmat.",
     ("use_nodes", "diffuse_color", "node_tree", "name"),
     "blender", "python"),
    ("import bpy\nmesh = bpy.data.meshes.new(name='M')\n"
     "mesh.from_pydata(",
     ("[", "verts", "("), "blender", "python"),
    ("import bpy\nimg = bpy.data.images.new(name='D', ",
     ("width", "256", "height"), "blender", "python"),
    ("import bpy\nob = bpy.context.object\nob.rotation_euler",
     (".", "=", "["), "blender", "python"),
    ("import bpy\nbpy.ops.mesh.primitive_cylinder_add(vertices=",
     ("32", "8", "6", "16"), "blender", "python"),
    ("import bpy\nfor v in mesh.vertices:\n    v.co.",
     ("x", "y", "z"), "blender", "python"),

    # ── JAVASCRIPT (genel, farklı dil) ──
    ("function ikiliArama(dizi, hedef) {\n"
     "  let alt = 0, ust = dizi.length - 1;\n"
     "  while (alt <= ust) {\n"
     "    const orta = Math.floor((alt + ust) / 2);\n"
     "    if (dizi[orta] === hedef) return ",
     ("orta",), "js", "javascript"),
    ("const dizi = [3, 1, 2];\ndizi.sort(",
     ("(a, b) => a - b", "function", "("), "js", "javascript"),
    ("function toplam(dizi) {\n  return dizi.reduce(",
     ("(a, b) => a + b", "function", "("), "js", "javascript"),
    ("const sozluk = new Map();\nsozluk.",
     ("set", "get", "has", "delete"), "js", "javascript"),
    ("async function veriAl(url) {\n  const c = await fetch(",
     ("url", "'", "\""), "js", "javascript"),
    ("class Dugum {\n  constructor(deger) {\n    this.deger = deger;\n"
     "    this.sonraki = ",
     ("null",), "js", "javascript"),
    ("const kareler = dizi.map(",
     ("x => x * x", "(x)", "function"), "js", "javascript"),
    ("function faktoriyel(n) {\n  if (n <= 1) return ",
     ("1",), "js", "javascript"),
)


@dataclass
class SizintiSonuc:
    """Sızıntı denetimi sonucu."""
    onek: str = ""
    sizinti: bool = False
    kaynak: str = ""
    eslesme: str = ""


def _egitim_metni(yollar: Optional[Sequence[str]] = None) -> str:
    """
    Eğitim verisini **tek metin** olarak yükler (sızıntı denetimi için).

    ⚠ Bellek: 23M token ≈ 80 MB metin. Tek seferde yüklenebilir.
    """
    if yollar is None:
        yollar = [
            ".neurocore/v6_veri.jsonl", ".neurocore/yerel_kod.jsonl",
            ".neurocore/karisim_verisi.jsonl",
            ".neurocore/dokuman_python313.jsonl",
            ".neurocore/dokuman_godot.jsonl",
            ".neurocore/dokuman_cpp.jsonl",
            ".neurocore/python_api.jsonl",
        ]
    parcalar: List[str] = []
    for y in yollar:
        p = Path(y)
        if not p.exists():
            continue
        try:
            with p.open(encoding="utf-8") as f:
                for satir in f:
                    try:
                        parcalar.append(json.loads(satir)["metin"])
                    except Exception:
                        continue
        except Exception:
            continue
    return "\n".join(parcalar)


def sizinti_denetle(*, egitim: Optional[str] = None,
                    min_uzunluk: int = 40) -> List[SizintiSonuc]:
    """
    Soru önekleri eğitim verisinde **birebir** var mı?

    ⚠ Yöntem: önekin **son 60 karakteri** eğitim metninde aranır.
    Bulunursa o soru **ezberlenebilir** demektir.
    """
    metin = egitim if egitim is not None else _egitim_metni()
    print(f"  eğitim metni: {len(metin):,} karakter")
    sonuclar: List[SizintiSonuc] = []
    for onek, _, kat, dil in SORULAR:
        s = SizintiSonuc(onek=onek[-50:].replace("\n", "\\n"))
        # Son 60 karakteri ara (tam eşleşme)
        parca = onek[-60:]
        if len(parca) < min_uzunluk:
            parca = onek
        if parca in metin:
            s.sizinti = True
            s.eslesme = parca[:40].replace("\n", "\\n")
            s.kaynak = "birebir"
        sonuclar.append(s)
    return sonuclar


def olc(yol: str, *, maks_token: int = 32) -> Dict[str, Any]:
    """Bağımsız benchmark'ı çalıştırır."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not Path(yol).exists():
        return {"hata": f"model yok: {yol}"}
    tok = AutoTokenizer.from_pretrained(yol)
    mdl = AutoModelForCausalLM.from_pretrained(yol)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(cihaz)
    mdl.eval()

    try:
        from tools.kod_son_isleme import SonIsleme

        son = SonIsleme()
    except Exception:
        son = None

    # ⚠ SIZINTI DENETİMİ (tur 10): 38 sorudan **11'i** eğitim verisinde
    # birebir geçiyor. Sızan sorular **ezberlenebilir** → ölçüm yanıltıcı.
    # Yalnız **temiz** sorular ölçülür.
    sizan: Set[str] = set()
    try:
        for s in sizinti_denetle():
            if s.sizinti:
                sizan.add(s.onek)
    except Exception:
        pass

    dogru = 0
    prose = 0
    kat_d: Dict[str, List[int]] = {}
    detay: List[Dict[str, Any]] = []
    atlanan = 0
    for onek, kabul, kat, dil in SORULAR:
        if onek[-50:].replace("\n", "\\n") in sizan:
            atlanan += 1
            continue
        try:
            ids = tok(onek, return_tensors="pt")["input_ids"].to(cihaz)
            with torch.no_grad():
                c = mdl.generate(ids, max_new_tokens=maks_token,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id,
                                 no_repeat_ngram_size=3)
            u = tok.decode(c[0][ids.shape[1]:], skip_special_tokens=True)
        except Exception as exc:
            u = f"HATA {type(exc).__name__}"
        ok = any(k.lower() in u.lower() for k in kabul)
        if ok:
            dogru += 1
        if son is not None:
            p, _ = son.kod_disi_var(u)
            if p:
                prose += 1
        d = kat_d.setdefault(kat, [0, 0])
        d[1] += 1
        if ok:
            d[0] += 1
        detay.append({"onek": onek[-40:].replace("\n", "\\n"),
                      "uretilen": u[:60], "dogru": ok, "kat": kat,
                      "dil": dil})
    return {"model": yol, "dogru": dogru, "toplam": len(SORULAR),
            "oran": dogru / max(1, len(SORULAR)), "prose": prose,
            "kategori": kat_d, "detay": detay}


def yazdir(m: Dict[str, Any]) -> None:
    """Sonucu yazdırır."""
    if "hata" in m:
        print(f"  {m['hata']}")
        return
    print(f"  {m['model']}")
    print(f"    BAĞIMSIZ: {m['dogru']}/{m['toplam']} "
          f"(%{m['oran'] * 100:.0f}) · prose {m['prose']}")
    for k, d in m["kategori"].items():
        print(f"      {k:10} {d[0]:2}/{d[1]:2} "
              f"(%{d[0] / max(1, d[1]) * 100:.0f})")


def main() -> int:
    """Bağımsız benchmark'ı çalıştırır."""
    ap = argparse.ArgumentParser(description="Bağımsız benchmark")
    ap.add_argument("--sizinti", action="store_true")
    ap.add_argument("--kiyasla", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--liste", action="store_true")
    args = ap.parse_args()

    if args.liste:
        for onek, kabul, kat, dil in SORULAR:
            print(f"  [{kat:8}][{dil:11}] "
                  f"{onek.splitlines()[-1][:46]!r} → {kabul[:3]}")
        print(f"\n  TOPLAM: {len(SORULAR)} bağımsız soru")
        return 0

    print("=" * 78)
    print("  BAĞIMSIZ BENCHMARK — ezber değil, genelleme ölçümü")
    print("=" * 78)
    print(f"  {len(SORULAR)} soru · 4 kategori · 3 dil")
    print()

    sonuc: Dict[str, Any] = {}
    if args.sizinti:
        print("  SIZINTI DENETİMİ:")
        sonuclar = sizinti_denetle()
        sizan = [s for s in sonuclar if s.sizinti]
        print(f"    sızan soru: {len(sizan)}/{len(sonuclar)}")
        for s in sizan[:8]:
            print(f"      ⚠ {s.onek!r}")
        sonuc["sizinti"] = {"sizan": len(sizan),
                            "toplam": len(sonuclar),
                            "ornekler": [s.onek for s in sizan[:10]]}

    if args.kiyasla:
        print("\n  MODEL KARŞILAŞTIRMASI (bağımsız):")
        for ad in ("qwen_coder_myai", "qwen_havuz2", "qwen_karisim",
                   "qwen_alan", "qwen_mantik", "qwen_tm",
                   "qwen_temiz2", "qwen_t18", "qwen_g4a", "qwen_g4b", "qwen_g4c", "qwen_coder_15b"):
            yol = f"models/{ad}"
            if not Path(yol).exists():
                continue
            m = olc(yol)
            sonuc[ad] = {k: v for k, v in m.items() if k != "detay"}
            yazdir(m)
            print()
    elif args.model:
        m = olc(args.model)
        sonuc["olcum"] = {k: v for k, v in m.items() if k != "detay"}
        yazdir(m)
        print("\n  YANLIŞLAR:")
        for d in m.get("detay", []):
            if not d["dogru"]:
                print(f"    [{d['kat']:8}] {d['onek']!r}")
                print(f"        → {d['uretilen'][:56]!r}")

    Path("_bagimsiz_benchmark.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
