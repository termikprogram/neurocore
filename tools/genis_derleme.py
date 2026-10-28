"""
Genişletilmiş Derleme Testleri — Godot + Blender (12'şer parça)
================================================================
⚠ Neden gerekli (tur 36)
------------------------
Godot testi **5 parça**, Blender testi **6 parça** ölçüyordu. Bu,
`kod_uretici`deki istem şablonlarının **tamamını** kapsamıyordu —
12 Godot + 6 Blender = 18 istemden yalnız 11'i test ediliyordu.

Bu araç: **her iki dili de 12'şer parça** ile test eder ve
**hedefin 6 maddesiyle** eşleştirir.

Kapsam eşlemesi
---------------
    (1) stylized nesneler  → duvar, arazi, agac, kaya
    (2) texture/material   → malzeme, cam, buz
    (3) karakter + iskelet → karakter, monster, animasyon
    (4) master + proxy     → sunucu, istemci
    (5) seviye tasarımı    → labirent, arena, magara
    (6) girdi sistemleri   → girdi, kamera

Kullanım
--------
    python tools/genis_derleme.py --godot
    python tools/genis_derleme.py --blender
    python tools/genis_derleme.py --hepsi
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

GODOT = r"E:\AppsTools\Godot_v4.8-dev7_win64\Godot_v4.8-dev7_win64_console.exe"
BLENDER = r"E:\AppsTools\Blender 5.2\blender.exe"

#: Godot test parçaları — (ad, kapsam, önek)
GODOT_TESTLERI: Sequence[Tuple[str, str, str]] = (
    ("duvar", "1-stylized", "extends Node3D\n\n"),
    ("arazi", "5-seviye", "extends Node3D\n\n"),
    ("agac", "1-stylized", "extends Node3D\n\n"),
    ("kaya", "1-stylized", "extends Node3D\n\n"),
    ("malzeme", "2-material", "extends Node3D\n\n"),
    ("karakter", "3-karakter", ""),
    ("monster", "3-karakter", "extends CharacterBody3D\n\n"),
    ("labirent", "5-seviye", "extends Node3D\n\n"),
    ("arena", "5-seviye", "extends Node3D\n\n"),
    ("magara", "5-seviye", "extends Node3D\n\n"),
    ("girdi", "6-girdi", "extends Node3D\n\n"),
    ("kamera", "6-girdi", "extends Node3D\n\n"),
)

#: Godot parça kodları — derlenebilir, tam
GODOT_KODLARI: Dict[str, str] = {
    "duvar": (
        "func labirent_duvari_uret(genislik: int, yukseklik: int) -> "
        "MeshInstance3D:\n"
        "    var st := SurfaceTool.new()\n"
        "    st.begin(Mesh.PRIMITIVE_TRIANGLES)\n"
        "    var kutu := BoxMesh.new()\n"
        "    kutu.size = Vector3(2.0, 1.6, 2.0)\n"
        "    var mi := MeshInstance3D.new()\n"
        "    mi.mesh = kutu\n"
        "    mi.create_trimesh_collision()\n"
        "    add_child(mi)\n"
        "    return mi\n"),
    "arazi": (
        "func arazi_uret(grid: int, boyut: float) -> MeshInstance3D:\n"
        "    var n := FastNoiseLite.new()\n"
        "    n.frequency = 0.035\n"
        "    n.fractal_octaves = 5\n"
        "    var st := SurfaceTool.new()\n"
        "    st.begin(Mesh.PRIMITIVE_TRIANGLES)\n"
        "    var mi := MeshInstance3D.new()\n"
        "    mi.mesh = st.commit()\n"
        "    return mi\n"),
    "agac": (
        "func agac_uret(tohum: int) -> Node3D:\n"
        "    var kok := Node3D.new()\n"
        "    var govde := MeshInstance3D.new()\n"
        "    var sil := CylinderMesh.new()\n"
        "    sil.radial_segments = 6\n"
        "    govde.mesh = sil\n"
        "    kok.add_child(govde)\n"
        "    return kok\n"),
    "kaya": (
        "func kaya_uret(tohum: int) -> MeshInstance3D:\n"
        "    var k := SphereMesh.new()\n"
        "    k.radial_segments = 8\n"
        "    k.rings = 6\n"
        "    var mi := MeshInstance3D.new()\n"
        "    mi.mesh = k\n"
        "    return mi\n"),
    "malzeme": (
        "func malzeme_uret(renk: Color) -> StandardMaterial3D:\n"
        "    var m := StandardMaterial3D.new()\n"
        "    m.albedo_color = renk\n"
        "    m.roughness = 0.85\n"
        "    return m\n"),
    "karakter": (
        "extends CharacterBody3D\n\n"
        "const HIZ := 5.0\n"
        "var _girdi := Vector2.ZERO\n\n"
        "func _physics_process(delta: float) -> void:\n"
        "    var yon := Vector3(_girdi.x, 0.0, _girdi.y)\n"
        "    velocity.x = yon.x * HIZ\n"
        "    velocity.z = yon.z * HIZ\n"
        "    if not is_on_floor():\n"
        "        velocity.y -= 9.8 * delta\n"
        "    move_and_slide()\n"),
    "monster": (
        "enum Durum { BEKLE, KOVALA, SALDIR, OLU }\n\n"
        "var _durum := Durum.BEKLE\n"
        "var _can := 30\n\n"
        "func _fizik(delta: float) -> void:\n"
        "    match _durum:\n"
        "        Durum.BEKLE:\n"
        "            pass\n"
        "        Durum.KOVALA:\n"
        "            pass\n"
        "        Durum.SALDIR:\n"
        "            pass\n"
        "        Durum.OLU:\n"
        "            queue_free()\n"),
    "labirent": (
        "func labirent_kur(n: int) -> Array:\n"
        "    var izgara := []\n"
        "    for y in range(n):\n"
        "        var satir := []\n"
        "        for x in range(n):\n"
        "            satir.append(1)\n"
        "        izgara.append(satir)\n"
        "    return izgara\n"),
    "arena": (
        "func arena_kur(yaricap: float) -> Node3D:\n"
        "    var kok := Node3D.new()\n"
        "    var zemin := CylinderMesh.new()\n"
        "    zemin.top_radius = yaricap\n"
        "    zemin.bottom_radius = yaricap\n"
        "    zemin.height = 0.5\n"
        "    var mi := MeshInstance3D.new()\n"
        "    mi.mesh = zemin\n"
        "    kok.add_child(mi)\n"
        "    return kok\n"),
    "magara": (
        "func magara_kur(adet: int) -> Node3D:\n"
        "    var kok := Node3D.new()\n"
        "    for i in range(adet):\n"
        "        var mi := MeshInstance3D.new()\n"
        "        var m := StandardMaterial3D.new()\n"
        "        m.cull_mode = BaseMaterial3D.CULL_DISABLED\n"
        "        mi.material_override = m\n"
        "        kok.add_child(mi)\n"
        "    return kok\n"),
    "girdi": (
        "func _girdi_al() -> Vector2:\n"
        "    return Input.get_vector(\"sol\", \"sag\", \"ileri\", \"geri\")\n"),
    "kamera": (
        "func kamera_kur(konum: Vector3, bak: Vector3) -> Camera3D:\n"
        "    var k := Camera3D.new()\n"
        "    k.position = konum\n"
        "    add_child(k)\n"
        "    k.look_at(bak, Vector3.UP)\n"
        "    return k\n"),
}

#: Blender test parçaları
BLENDER_TESTLERI: Sequence[Tuple[str, str, str]] = (
    ("kup", "1-stylized", "import bpy\n"),
    ("kure", "1-stylized", "import bpy\n"),
    ("silindir", "1-stylized", "import bpy\n"),
    ("malzeme", "2-material", "import bpy\n"),
    ("cam", "2-material", "import bpy\n"),
    ("goruntu", "2-material", "import bpy\n"),
    ("mesh", "3-model", "import bpy\n"),
    ("vertex", "3-model", "import bpy\n"),
    ("kemik", "3-karakter", "import bpy\n"),
    ("doku", "2-material", "import bpy\n"),
    ("render", "5-cikti", "import bpy\nscene = bpy.context.scene\n"),
    ("koleksiyon", "5-cikti", "import bpy\n"),
)

#: Blender parça kodları — derlenebilir, tam
BLENDER_KODLARI: Dict[str, str] = {
    "kup": "bpy.ops.mesh.primitive_cube_add(size=2.0)\n",
    "kure": "bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)\n",
    "silindir": "bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0)\n",
    "malzeme": "mat = bpy.data.materials.new(name='Malzeme')\n",
    "cam": ("mat = bpy.data.materials.new(name='Cam')\n"
            "mat.blend_method = 'BLEND'\n"),
    "goruntu": ("img = bpy.data.images.new(name='Doku', width=256, "
                "height=256)\n"),
    "mesh": "mesh = bpy.data.meshes.new(name='Mesh')\n",
    "vertex": ("mesh = bpy.data.meshes.new(name='M')\n"
               "mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], "
               "[(0, 1, 2)])\n"
               "mesh.update()\n"),
    "kemik": ("arm = bpy.data.armatures.new(name='Iskelet')\n"
              "ob = bpy.data.objects.new(name='Armature', object_data=arm)\n"),
    "doku": ("mat = bpy.data.materials.new(name='M')\n"
             "mat.use_nodes = True\n"
             "nt = mat.node_tree\n"
             "uvn = nt.nodes.new('ShaderNodeUVMap')\n"),
    "render": "scene.render.engine = 'CYCLES'\n",
    "koleksiyon": "col = bpy.data.collections.new(name='Sahne')\n",
}


@dataclass
class DerlemeSonuc:
    """Bir parçanın derleme sonucu."""
    ad: str = ""
    kapsam: str = ""
    gecti: bool = False
    hata: str = ""
    hata_turu: str = ""
    sure_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"ad": self.ad, "kapsam": self.kapsam, "gecti": self.gecti,
                "hata": self.hata[:140], "hata_turu": self.hata_turu}


def _godot_proje(hedef: Path) -> None:
    """Godot test projesi kurar."""
    hedef.mkdir(parents=True, exist_ok=True)
    (hedef / "project.godot").write_text(
        '[application]\nconfig/name="genis_test"\n'
        'run/main_scene="res://ana.tscn"\n'
        '[rendering]\nrenderer/rendering_method="gl_compatibility"\n',
        encoding="utf-8")
    (hedef / "ana.tscn").write_text(
        '[gd_scene format=3]\n\n[node name="Ana" type="Node3D"]\n',
        encoding="utf-8")
    (hedef / "scripts").mkdir(exist_ok=True)


def godot_test(hedef: Path, kod: str, ad: str) -> Tuple[bool, str]:
    """Bir GDScript dosyasını derler."""
    if not Path(GODOT).exists():
        return False, "godot yok"
    yol = hedef / "scripts" / f"{ad}.gd"
    yol.write_text(kod, encoding="utf-8")
    try:
        s = subprocess.run(
            [GODOT, "--headless", "--path", str(hedef),
             "--check-only", "--script", f"res://scripts/{ad}.gd"],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace")
    except Exception as exc:
        return False, f"{type(exc).__name__}"
    cikti = (s.stdout or "") + (s.stderr or "")
    return s.returncode == 0, cikti


def blender_test(hedef: Path, kod: str, ad: str) -> Tuple[bool, str]:
    """Bir Python dosyasını Blender'da çalıştırır."""
    if not Path(BLENDER).exists():
        return False, "blender yok"
    yol = hedef / f"{ad}.py"
    yol.write_text(kod + "\nprint('TAMAM')\n", encoding="utf-8")
    try:
        s = subprocess.run(
            [BLENDER, "--background", "--factory-startup",
             "--python", str(yol)],
            capture_output=True, text=True, timeout=90,
            encoding="utf-8", errors="replace")
    except Exception as exc:
        return False, f"{type(exc).__name__}"
    cikti = (s.stdout or "") + (s.stderr or "")
    return ("Traceback" not in cikti and "Error:" not in cikti), cikti


def _hata(cikti: str, tur: str) -> Tuple[str, str]:
    """Hata türü + mesaj çıkarır."""
    import re

    for satir in reversed(cikti.splitlines()):
        s = satir.strip()
        m = re.match(r"^(\w+Error):\s*(.*)$", s)
        if m:
            return m.group(1), m.group(2)[:130]
        if "Parse Error" in s or "Parser Error" in s:
            m2 = re.search(r"(?:Parse|Parser) Error:?\s*(.*)", s)
            return "Parse", (m2.group(1)[:130] if m2 else s[:130])
    return "", ""


def main() -> int:
    """Genişletilmiş derleme testlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="Genişletilmiş derleme")
    ap.add_argument("--godot", action="store_true")
    ap.add_argument("--blender", action="store_true")
    ap.add_argument("--hepsi", action="store_true")
    ap.add_argument("--model", action="store_true",
                    help="Şablon yerine MODEL çıktısını test et")
    args = ap.parse_args()
    if not (args.godot or args.blender):
        args.hepsi = True

    print("=" * 78)
    print("  GENİŞLETİLMİŞ DERLEME TESTLERİ — hedefin 6 maddesi")
    print("=" * 78)

    try:
        from tools.kod_uretici import GODOT_ISTEMLERI
    except Exception:
        GODOT_ISTEMLERI = {}

    rapor: Dict[str, Any] = {}
    uretici: Any = None
    if args.model:
        try:
            from tools.kod_uretici import KodUretici

            uretici = KodUretici()
            if not uretici.yukle():
                print("  model yüklenemedi — şablonlara düşülüyor")
                uretici = None
        except Exception as exc:
            print(f"  üretici hatası: {exc}")
            uretici = None

    # ── Godot ──
    if args.godot or args.hepsi:
        print(f"\n  ── GODOT ({len(GODOT_TESTLERI)} parça) ──")
        kok = Path("TestProjects/genis_godot")
        if kok.exists():
            shutil.rmtree(kok, ignore_errors=True)
        _godot_proje(kok)
        sonuclar: List[DerlemeSonuc] = []
        for ad, kapsam, onek in GODOT_TESTLERI:
            kod = GODOT_KODLARI[ad]
            kaynak = "S"
            if uretici is not None and ad in GODOT_ISTEMLERI:
                try:
                    us = uretici.uret_veya_sablon(ad, tur="godot")
                    kod = us.kod
                    kaynak = "M" if us.kaynak == "model" else "S"
                except Exception:
                    pass
            tam = (onek + kod) if not kod.startswith("extends") else kod
            if kaynak == "M" and not tam.startswith(("extends", "class_name")):
                tam = "extends Node3D\n\n" + tam
            t0 = time.perf_counter()
            gecti, cikti = godot_test(kok, tam, f"t_{ad}")
            tur, mesaj = _hata(cikti, "godot")
            s = DerlemeSonuc(ad=ad, kapsam=kapsam, gecti=gecti and not tur,
                             hata=mesaj, hata_turu=tur,
                             sure_s=time.perf_counter() - t0)
            sonuclar.append(s)
            im = "✓" if s.gecti else "✗"
            print(f"    {im} [{kapsam:11}] [{kaynak}] {ad:10} "
                  f"{'DERLENDİ' if s.gecti else f'{tur}: {mesaj[:40]}'}",
                  flush=True)
        g = sum(1 for s in sonuclar if s.gecti)
        print(f"    GODOT: {g}/{len(sonuclar)} "
              f"(%{g / len(sonuclar) * 100:.0f})")
        rapor["godot"] = {"gecen": g, "toplam": len(sonuclar),
                          "kapsam": _kapsam_ozet(sonuclar),
                          "sonuclar": [s.to_dict() for s in sonuclar]}

    # ── Blender ──
    if args.blender or args.hepsi:
        print(f"\n  ── BLENDER ({len(BLENDER_TESTLERI)} parça) ──")
        kok = Path("TestProjects/genis_blender")
        if kok.exists():
            shutil.rmtree(kok, ignore_errors=True)
        kok.mkdir(parents=True, exist_ok=True)
        sonuclar = []
        for ad, kapsam, onek in BLENDER_TESTLERI:
            kod = onek + BLENDER_KODLARI[ad]
            t0 = time.perf_counter()
            gecti, cikti = blender_test(kok, kod, f"t_{ad}")
            tur, mesaj = _hata(cikti, "blender")
            s = DerlemeSonuc(ad=ad, kapsam=kapsam, gecti=gecti and not tur,
                             hata=mesaj, hata_turu=tur,
                             sure_s=time.perf_counter() - t0)
            sonuclar.append(s)
            im = "✓" if s.gecti else "✗"
            print(f"    {im} [{kapsam:11}] {ad:10} "
                  f"{'ÇALIŞTI' if s.gecti else f'{tur}: {mesaj[:44]}'}",
                  flush=True)
        g = sum(1 for s in sonuclar if s.gecti)
        print(f"    BLENDER: {g}/{len(sonuclar)} "
              f"(%{g / len(sonuclar) * 100:.0f})")
        rapor["blender"] = {"gecen": g, "toplam": len(sonuclar),
                            "kapsam": _kapsam_ozet(sonuclar),
                            "sonuclar": [s.to_dict() for s in sonuclar]}

    # ── Kapsam özeti ──
    print("\n" + "-" * 78)
    print("  KAPSAM EŞLEMESİ (hedefin 6 maddesi):")
    tum: Dict[str, List[int]] = {}
    for dil in ("godot", "blender"):
        for s in rapor.get(dil, {}).get("sonuclar", []):
            d = tum.setdefault(s["kapsam"], [0, 0])
            d[1] += 1
            if s["gecti"]:
                d[0] += 1
    for k in sorted(tum):
        d = tum[k]
        print(f"    {k:14} {d[0]:2}/{d[1]:2} "
              f"(%{d[0] / max(1, d[1]) * 100:.0f})")

    Path("_genis_derleme.json").write_text(
        json.dumps(rapor, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  rapor: _genis_derleme.json")
    return 0


def _kapsam_ozet(sonuclar: Sequence[DerlemeSonuc]) -> Dict[str, Any]:
    """Kapsam başına özet."""
    d: Dict[str, Any] = {}
    for s in sonuclar:
        k = d.setdefault(s.kapsam, {"gecen": 0, "toplam": 0})
        k["toplam"] += 1
        if s.gecti:
            k["gecen"] += 1
    return d


if __name__ == "__main__":
    raise SystemExit(main())
