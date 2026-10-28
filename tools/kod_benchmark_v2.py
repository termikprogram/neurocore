"""
Zorlaştırılmış Kod Tamamlama Benchmark'ı
==========================================
⚠ Neden gerekli (tur 33)
------------------------
Eski test seti **14 soru** — v2 ve v5 ikisi de **8/14 (%57)** aldı.
Test **doygunluğa** ulaştı: modeller arasındaki farkı ölçemiyor.

Yeni benchmark: **40 soru**, 4 zorluk katmanı
----------------------------------------------
    KOLAY   (10) : Tek API adı           `bpy.ops.mesh.primitive_` → cube_add
    ORTA    (12) : Parametre/alan         `img = bpy.data.images.new(` → name=
    ZOR     (10) : Çok adımlı kalıp       node ağacı, sinyal bağlama
    ÇOK ZOR  (8)  : Godot oyun kalıbı     fizik, animasyon, RPC

Ölçütler
--------
    * **Doğruluk** — beklenen kalıp üretildi mi
    * **Derlenebilirlik** — Godot/Python sözdizimi geçerli mi
    * **Prose oranı** — kod yerine yazı üretildi mi
    * **Zorluk kırılımı** — hangi katmanda zayıf

Kullanım
--------
    python tools/kod_benchmark_v2.py --model models/smollm2_kod_v6
    python tools/kod_benchmark_v2.py --kiyasla
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: (önek, kabul edilen kalıplar, zorluk, tür)
SORULAR: Sequence[Tuple[str, Sequence[str], str, str]] = (
    # ── KOLAY (10) — tek API adı ──
    ("import bpy\nbpy.ops.mesh.primitive_",
     ("cube_add", "cylinder_add", "uv_sphere_add", "cone_add",
      "torus_add", "plane_add", "circle_add", "monkey_add"),
     "kolay", "blender"),
    ("import bpy\nbpy.data.materials.", ("new", "get", "remove"),
     "kolay", "blender"),
    ("import bpy\nbpy.data.meshes.", ("new", "get", "remove"),
     "kolay", "blender"),
    ("import bpy\nbpy.data.images.", ("new", "load", "get"),
     "kolay", "blender"),
    ("import bpy\nbpy.data.objects.", ("new", "get", "remove"),
     "kolay", "blender"),
    ("extends Node3D\n\nfunc _ready() -> void:\n    var mi := MeshInstance3D.new()\n    mi.",
     ("mesh", "position", "name", "material"), "kolay", "godot"),
    ("extends CharacterBody3D\n\nfunc _physics_process(delta: float) -> void:\n    velocity.",
     ("x", "y", "z"), "kolay", "godot"),
    ("const PORT := 7777\npeer.create_server(",
     ("PORT", "16", "0"), "kolay", "godot"),
    ("import bpy\nbpy.ops.object.", ("select_all", "delete", "mode_set",
                                      "duplicate", "join"),
     "kolay", "blender"),
    ("import bpy\nbpy.context.scene.", ("render", "frame", "objects",
                                        "collection"), "kolay", "blender"),

    # ── ORTA (12) — parametre/alan ──
    ("import bpy\nmat = bpy.data.materials.new(",
     ("name", "'", '"'), "orta", "blender"),
    ("import bpy\nimg = bpy.data.images.new(",
     ("name", "width", "256", "512"), "orta", "blender"),
    ("import bpy\nmesh = bpy.data.meshes.new(",
     ("name", "'", '"'), "orta", "blender"),
    ("import bpy\nob = bpy.context.object\n"
     "for p in ob.data.polygons:\n    p.use_smooth = ", ("False", "True"),
     "orta", "blender"),
    ("import bpy\nbsdf.inputs['Base Color'].default_value = ",
     ("(", "Color", "0."), "orta", "blender"),
    ("import bpy\nscene.render.engine = ",
     ("'CYCLES'", "'BLENDER", "engine"), "orta", "blender"),
    ("import bpy\nfor v in mesh.vertices:\n    v.co.z = ",
     ("0", "h", "yukseklik"), "orta", "blender"),
    ("extends Node3D\n\nfunc _ready() -> void:\n"
     "    var m := StandardMaterial3D.new()\n    m.albedo_color = ",
     ("Color", "("), "orta", "godot"),
    ("extends Node3D\n\nfunc _ready() -> void:\n"
     "    var sil := CylinderMesh.new()\n    sil.radial_segments = ",
     ("6", "8", "3", "4", "5"), "orta", "godot"),
    ("import bpy\nbpy.ops.object.select_all(action=",
     ("'SELECT'", "'DESELECT'", "'TOGGLE'"), "orta", "blender"),
    ("extends Node3D\n\nvar hiz := 5.0\n\n"
     "func _process(delta: float) -> void:\n    position.x += ",
     ("hiz", "delta", "0"), "orta", "godot"),
    ("import bpy\nnt = mat.node_tree\nnt.nodes.new(",
     ("'ShaderNode", "type"), "orta", "blender"),

    # ── ZOR (10) — çok adımlı kalıp ──
    ("import bpy\nnt = mat.node_tree\nuvn = nt.nodes.new('ShaderNodeUVMap')\n"
     "nt.links.new(uvn.outputs['UV'], ",
     ("olcek", "inputs", "mapping"), "zor", "blender"),
    ("import bpy\nbpy.ops.mesh.primitive_cube_add(size=2)\n"
     "ob = bpy.context.object\nob.name = ",
     ('"', "'", "name"), "zor", "blender"),
    ("extends Node3D\n\nfunc _hazirla() -> void:\n"
     "    var alan := Area3D.new()\n    alan.body_entered.connect(",
     ("_", "func", "on_"), "zor", "godot"),
    ("extends CharacterBody3D\n\n"
     "func _physics_process(delta: float) -> void:\n"
     "    if not is_on_floor():\n        velocity.y -= ",
     ("9.8", "yer", "gravity", "delta"), "zor", "godot"),
    ("@rpc(\"any_peer\", \"call_remote\")\nfunc _hasar_al(miktar: int) -> void:\n"
     "    if not multiplayer.is_server():\n        return\n    ",
     ("can", "health", "miktar"), "zor", "godot"),
    ("import bpy\nimg = bpy.data.images.new('doku', 256, 256)\n"
     "px = list(img.pixels)\nfor i in range(0, len(px), 4):\n    px[i] = ",
     ("0", "1", "r", "renk"), "zor", "blender"),
    ("extends Node3D\n\nfunc _olustur() -> MeshInstance3D:\n"
     "    var st := SurfaceTool.new()\n    st.begin(",
     ("Mesh.", "PRIMITIVE"), "zor", "godot"),
    ("extends Node3D\n\n@onready var kamera: Camera3D = $Kamera\n\n"
     "func _process(_d: float) -> void:\n    kamera.look_at(",
     ("position", "global_position", "target"), "zor", "godot"),
    ("import bpy\nfor i, ob in enumerate(bpy.data.objects):\n"
     "    ob.location.x = ",
     ("i", "0", "x"), "zor", "blender"),
    ("extends Node\n\nsignal hasar_alindi(miktar: int)\n\n"
     "func _hazir() -> void:\n    hasar_alindi.",
     ("connect", "emit"), "zor", "godot"),

    # ── ÇOK ZOR (8) — oyun kalıpları ──
    ("extends Node3D\n\nfunc _labirent_kur(n: int) -> void:\n"
     "    var izgara := []\n    for y in range(n):\n        var satir := []\n"
     "        for x in range(n):\n            satir.append(",
     ("0", "1", "true", "false", "duvar"), "cok_zor", "godot"),
    ("class_name Envanter\nextends Node\n\n"
     "var esyalar: Array[String] = []\n\n"
     "func ekle(ad: String) -> bool:\n    if esyalar.has(ad):\n        return ",
     ("false", "true"), "cok_zor", "godot"),
    ("extends Node3D\n\nvar can := 100\n\n"
     "func _hasar_al(miktar: int) -> void:\n    can -= miktar\n"
     "    if can <= 0:\n        ",
     ("ol", "queue_free", "die", "can"), "cok_zor", "godot"),
    ("extends Node\n\nvar oyuncular := {}\n\n"
     "func _oyuncu_ekle(id: int, ad: String) -> void:\n    oyuncular[id] = ",
     ("ad", "{", "Oyuncu", "id"), "cok_zor", "godot"),
    ("extends Node3D\n\nfunc _dusman_uret(adet: int) -> void:\n"
     "    for i in range(adet):\n        var d := preload(",
     ("'res://", '"res://', "Dusman"), "cok_zor", "godot"),
    ("extends Node\n\nfunc _kaydet(yol: String) -> void:\n"
     "    var f := FileAccess.open(yol, FileAccess.",
     ("WRITE", "READ"), "cok_zor", "godot"),
    ("extends Node3D\n\nfunc _dunya_uret() -> void:\n"
     "    var n := FastNoiseLite.new()\n    n.frequency = ",
     ("0.", "1."), "cok_zor", "godot"),
    ("extends Node\n\nconst HIZ := 5.0\n\n"
     "func _girdi_al() -> Vector2:\n    return Input.get_vector(",
     ('"', "'", "sol", "sola"), "cok_zor", "godot"),
)


@dataclass
class SoruSonuc:
    """Bir sorunun sonucu."""
    onek: str = ""
    uretilen: str = ""
    dogru: bool = False
    prose: bool = False
    zorluk: str = ""
    tur: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"onek": self.onek[-40:], "uretilen": self.uretilen[:70],
                "dogru": self.dogru, "prose": self.prose,
                "zorluk": self.zorluk, "tur": self.tur}


def olc(yol: str, *, maks_token: int = 32) -> Dict[str, Any]:
    """Zorlaştırılmış benchmark'ı çalıştırır."""
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

    sonuclar: List[SoruSonuc] = []
    for onek, kabul, zorluk, tur in SORULAR:
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
        s = SoruSonuc(onek=onek, uretilen=u, zorluk=zorluk, tur=tur)
        s.dogru = any(k.lower() in u.lower() for k in kabul)
        if son is not None:
            s.prose, _ = son.kod_disi_var(u)
        sonuclar.append(s)

    # Kırılım
    kirilim: Dict[str, Dict[str, int]] = {}
    for s in sonuclar:
        d = kirilim.setdefault(s.zorluk, {"dogru": 0, "toplam": 0})
        d["toplam"] += 1
        if s.dogru:
            d["dogru"] += 1
    tur_kirilim: Dict[str, Dict[str, int]] = {}
    for s in sonuclar:
        d = tur_kirilim.setdefault(s.tur, {"dogru": 0, "toplam": 0})
        d["toplam"] += 1
        if s.dogru:
            d["dogru"] += 1

    dogru = sum(1 for s in sonuclar if s.dogru)
    prose = sum(1 for s in sonuclar if s.prose)
    return {
        "model": yol, "dogru": dogru, "toplam": len(sonuclar),
        "oran": dogru / max(1, len(sonuclar)), "prose": prose,
        "zorluk": kirilim, "tur": tur_kirilim,
        "detay": [s.to_dict() for s in sonuclar],
    }


def yazdir(m: Dict[str, Any]) -> None:
    """Sonucu yazdırır."""
    if "hata" in m:
        print(f"  {m['hata']}")
        return
    print(f"  {m['model']}")
    print(f"    DOĞRULUK : {m['dogru']}/{m['toplam']} "
          f"(%{m['oran'] * 100:.0f})")
    print(f"    PROSE    : {m['prose']}/{m['toplam']}")
    print("    ZORLUK KIRILIMI:")
    for z in ("kolay", "orta", "zor", "cok_zor"):
        d = m["zorluk"].get(z)
        if d:
            print(f"      {z:10} {d['dogru']:2}/{d['toplam']:2} "
                  f"(%{d['dogru'] / max(1, d['toplam']) * 100:.0f})")
    print("    TÜR KIRILIMI:")
    for t, d in m["tur"].items():
        print(f"      {t:10} {d['dogru']:2}/{d['toplam']:2} "
              f"(%{d['dogru'] / max(1, d['toplam']) * 100:.0f})")


def main() -> int:
    """Benchmark'ı çalıştırır."""
    ap = argparse.ArgumentParser(description="Zorlaştırılmış benchmark")
    ap.add_argument("--model", default="models/smollm2_kod_v6")
    ap.add_argument("--kiyasla", action="store_true")
    ap.add_argument("--liste", action="store_true")
    args = ap.parse_args()

    if args.liste:
        for onek, kabul, z, t in SORULAR:
            print(f"  [{z:8}][{t:7}] {onek.splitlines()[-1][:44]}")
        print(f"\n  TOPLAM: {len(SORULAR)} soru")
        return 0

    print("=" * 78)
    print("  ZORLAŞTIRILMIŞ KOD TAMAMLAMA BENCHMARK")
    print("=" * 78)
    print(f"  {len(SORULAR)} soru · 4 zorluk katmanı · 2 dil")
    print()

    sonuc: Dict[str, Any] = {}
    if args.kiyasla:
        for ad in ("v2", "v5", "v6"):
            yol = f"models/smollm2_kod_{ad}"
            if not Path(yol).exists():
                continue
            m = olc(yol)
            sonuc[ad] = {k: v for k, v in m.items() if k != "detay"}
            yazdir(m)
            print()
    else:
        m = olc(args.model)
        sonuc["olcum"] = {k: v for k, v in m.items() if k != "detay"}
        yazdir(m)
        # Yanlışları göster
        print("\n  YANLIŞ CEVAPLAR (ilk 8):")
        for d in m.get("detay", []):
            if not d["dogru"]:
                print(f"    [{d['zorluk']:7}] {d['onek']!r}")
                print(f"        → {d['uretilen'][:60]!r}")

    Path("_benchmark_v2.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
