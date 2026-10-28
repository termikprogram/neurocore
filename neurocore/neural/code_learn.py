"""
Kod Öğrenme Seti — Blender Python Tamamlama
============================================
Yerel modele **kod yazmayı** öğretir.

Neden ayrı veri kümesi?
    Ölçüm gösterdi: mevcut model **kod tamamlayamıyor** (0/6):

        Girdi : "import bpy\\nbpy.ops.mesh.primitive_"
        Çıktı : "(tonnadodirb. (er 4 sela.2, s 1Er3.75,derincc"

    Sebep: müfredat **bilgi metni** (açıklama + örnek blok) hâlinde
    öğretildi. Model "erozyon nedir" sorusunu cevaplayabiliyor ama
    `bpy.ops.mesh.primitive_` sonrasını **üretemiyor** — çünkü hiç
    "yarım kodu tamamla" görevi görmedi.

    Bu modül o boşluğu kapatır: **önek → devam** çiftleri üretir.

Yöntem
------
    1. Müfredattaki 20+ Blender kod bloğu → satır satır önek/devam çifti
    2. Blender API kalıpları (blok blok, gerçek kullanım)
    3. Kısa, net, **tamamlanabilir** parçalar (tek satır - 4 satır)

⚠ Uydurma API yok: tüm çağrılar gerçek Blender API'sidir ve
`ast.parse` ile doğrulanır.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class KodOrnegi:
    """Tek bir kod tamamlama örneği."""
    onek: str = ""          # verilen kısım
    devam: str = ""         # modelin üretmesi gereken
    etiket: str = "blender"
    kaynak: str = ""
    #: Devam **yeni satırdan** mı başlar? (müfredat örneklerinde evet)
    yeni_satir: bool = False

    def egitim_metni(self) -> str:
        """Eğitim çifti (istem → hedef)."""
        return f"KOD:\n{self.onek}\nDEVAM:\n{self.devam}"

    def tam_metin(self) -> str:
        """Önek + devam birleşimi (doğrulama ve üretim karşılaştırması)."""
        return birlestir(self.onek, self.devam, self.yeni_satir)

    def to_dict(self) -> Dict[str, Any]:
        return {"onek": self.onek, "devam": self.devam,
                "etiket": self.etiket, "kaynak": self.kaynak,
                "yeni_satir": self.yeni_satir}


# ─── Blender API Kalıpları ──────────────────────────────────────────────────
#
# Her kalıp: (önek, devam). Önek **yarım** bırakılır ki model tamamlamayı
# öğrensin. Hepsi gerçek API ve geçerli Python.

KALIPLAR: Sequence[Tuple[str, str, str]] = (
    # ── Mesh primitifleri ──
    ("import bpy\nbpy.ops.mesh.primitive_cube_add(",
     "size=2.0, location=(0, 0, 0))", "primitive"),
    ("import bpy\nbpy.ops.mesh.primitive_cylinder_add(",
     "vertices=6, radius=0.15, depth=2.0)", "primitive"),
    ("import bpy\nbpy.ops.mesh.primitive_cone_add(",
     "vertices=6, radius1=0.8, depth=1.6)", "primitive"),
    ("import bpy\nbpy.ops.mesh.primitive_uv_sphere_add(",
     "segments=8, ring_count=6, radius=0.5)", "primitive"),
    ("import bpy\nbpy.ops.mesh.primitive_ico_sphere_add(",
     "subdivisions=1, radius=0.5)", "primitive"),
    ("import bpy\nbpy.ops.mesh.primitive_grid_add(",
     "x_subdivisions=80, y_subdivisions=80, size=40.0)", "primitive"),
    ("import bpy\nbpy.ops.mesh.primitive_plane_add(",
     "size=20.0, location=(0, 0, 0))", "primitive"),
    # ── Nesne erişimi ──
    ("import bpy\nob = bpy.context.object\n",
     "print(len(ob.data.vertices), len(ob.data.polygons))", "nesne"),
    ("import bpy\nob = bpy.data.objects[",
     "'Terrain']", "nesne"),
    ("import bpy\nob = bpy.context.object\nob.name = ",
     "'Govde'", "nesne"),
    # ── Lowpoly gölgeleme ──
    ("import bpy\nob = bpy.context.object\nfor p in ob.data.polygons:\n    p.use_smooth = ",
     "False", "lowpoly"),
    ("import bpy\nob = bpy.context.object\nucgen = sum(len(p.vertices) - 2 for p in ",
     "ob.data.polygons)", "lowpoly"),
    # ── Yüz azaltma ──
    ("import bpy\nob = bpy.context.object\nmod = ob.modifiers.new('Dec', 'DECIMATE')\nmod.ratio = ",
     "0.5\nbpy.ops.object.modifier_apply(modifier=mod.name)", "decimate"),
    # ── Vertex color ──
    ("import bpy\nob = bpy.context.object\nmesh = ob.data\nca = mesh.color_attributes.new(",
     "name='Col', type='FLOAT_COLOR', domain='POINT')", "vertexcolor"),
    ("import bpy\nca = mesh.color_attributes['Col']\nfor i in range(len(ca.data)):\n    ca.data[i].color = ",
     "(0.25, 0.5, 0.2, 1.0)", "vertexcolor"),
    # ── Malzeme ──
    ("import bpy\nmat = bpy.data.materials.new(",
     "name='TerrainMat')", "material"),
    ("import bpy\nmat = bpy.data.materials.new('M')\nmat.use_nodes = True\nnt = ",
     "mat.node_tree", "material"),
    ("import bpy\nbsdf = nt.nodes.get(",
     "'Principled BSDF')", "material"),
    ("import bpy\nbsdf.inputs['Base Color'].default_value = ",
     "(0.3, 0.55, 0.22, 1.0)", "material"),
    ("import bpy\nbsdf.inputs['Roughness'].default_value = ",
     "0.9", "material"),
    # ── Texture düğümü (KRTİK: UV bağlama) ──
    ("import bpy\nd = nt.nodes.new('ShaderNodeTexImage')\nd.image = ",
     "img", "texture"),
    ("import bpy\nuvn = nt.nodes.new('ShaderNodeUVMap')\nuvn.uv_map = ",
     "'UVMap'", "texture"),
    ("import bpy\nolcek = nt.nodes.new('ShaderNodeMapping')\nolcek.inputs['Scale'].default_value = ",
     "(10.0, 10.0, 1.0)", "texture"),
    ("import bpy\nnt.links.new(uvn.outputs['UV'], ",
     "olcek.inputs['Vector'])", "texture"),
    ("import bpy\nnt.links.new(olcek.outputs['Vector'], ",
     "d.inputs['Vector'])", "texture"),
    ("import bpy\nimg = bpy.data.images.new(",
     "name='tex', width=256, height=256)", "texture"),
    ("import bpy\nimg.pixels.foreach_set(",
     "px.ravel())", "texture"),
    # ── UV ──
    ("import bpy\nuv = mesh.uv_layers.new(",
     "name='UVMap')", "uv"),
    ("import bpy\nfor poly in mesh.polygons:\n    for li in poly.loop_indices:\n        uv.data[li].uv = ",
     "((vi % GRID) / (GRID - 1), (vi // GRID) / (GRID - 1))", "uv"),
    # ── Işık / kamera / render ──
    ("import bpy\nbpy.ops.object.light_add(type='SUN', ",
     "location=(30, -25, 40))", "sahne"),
    ("import bpy\ngunes = bpy.context.object\ngunes.data.energy = ",
     "4.0", "sahne"),
    ("import bpy\ngunes.rotation_euler = (math.radians(50), 0, ",
     "math.radians(35))", "sahne"),
    ("import bpy\nbpy.ops.object.camera_add(",
     "location=(40, -40, 30))", "sahne"),
    ("import bpy\ncam = bpy.context.object\ncam.data.type = ",
     "'ORTHO'", "sahne"),
    ("import bpy\ncam.data.ortho_scale = ",
     "84.0", "sahne"),
    ("import bpy\nscene = bpy.context.scene\nscene.render.engine = ",
     "'CYCLES'", "render"),
    ("import bpy\nscene.cycles.device = ",
     "'CPU'", "render"),
    ("import bpy\nscene.cycles.samples = ",
     "48", "render"),
    ("import bpy\nscene.render.resolution_x = ",
     "900", "render"),
    ("import bpy\nscene.render.filepath = ",
     "'render_01.png'\nbpy.ops.render.render(write_still=True)", "render"),
    ("import bpy\nbpy.ops.wm.save_as_mainfile(",
     "filepath='terrain.blend')", "kayit"),
    # ── Gürültü / matematik ──
    ("import math\ndef smoothstep(a, b, x):\n    t = max(0.0, min(1.0, (x - a) / (b - a)))\n    return ",
     "t * t * (3.0 - 2.0 * t)", "matematik"),
    ("import random\ndef agac_parametreleri(tohum):\n    r = random.Random(tohum)\n    return ",
     "{'boy': r.uniform(1.5, 4.0), 'tac': r.uniform(0.6, 1.4)}", "matematik"),
    ("import math\nh = ((n + 1) / 2) ** ",
     "2.0", "matematik"),
    ("import math\nh += (1.0 - abs(n2 * 2.0 - 1.0)) * ",
     "0.45", "matematik"),
    ("import numpy as np\nyy, xx = np.mgrid[",
     "0:TS, 0:TS].astype(np.float32)", "matematik"),
    ("import numpy as np\npx[:, :, 0] = np.clip(",
     "temel[0] + alan, 0, 1)", "matematik"),
    # ── Erozyon ──
    ("heights[k] -= kes\nheights[en_iyi] += ",
     "min(DEPOSIT, kes * 0.25)", "erozyon"),
    ("for tur in range(EROZYON_TUR):\n    su = ",
     "[0.02] * N", "erozyon"),
    ("sira = sorted(range(N), key=lambda k: heights[k], ",
     "reverse=True)", "erozyon"),
    ("if en_iyi >= 0 and en_dus > 1e-7:\n    tasi = su[k] * ",
     "0.5", "erozyon"),
    # ── Katman ağırlıkları ──
    ("w_kum = 1.0 - smoothstep(",
     "0.14, 0.22, t)", "katman"),
    ("w_kar = smoothstep(0.70, 0.84, t) * smoothstep(",
     "0.50, 0.72, egim)", "katman"),
    ("w_kaya = max(smoothstep(0.42, 0.62, t) * (1.0 - w_kar),\n              ",
     "1.0 - smoothstep(0.58, 0.80, egim))", "katman"),
    ("w_cimen = max(0.0, 1.0 - ",
     "w_kum - w_kaya - w_kar)", "katman"),
    ("normaller[vi] = nv.", "normalized()", "katman"),
    # ── Dosya / klasör ──
    ("import os\nos.makedirs(",
     "OUT, exist_ok=True)", "dosya"),
    ("import os\nimg.filepath_raw = os.path.join(",
     "OUT, ad + '.png')", "dosya"),
    # ── Hata ayıklama ──
    ("print(f'{len(mesh.vertices)} vertex, ",
     "{len(mesh.polygons)} yuz')", "debug"),
    ("assert len(mesh.polygons) > 0, ",
     "'mesh bos!'", "debug"),
)


def kalip_ornekleri() -> List[KodOrnegi]:
    """API kalıplarından kod örnekleri üretir."""
    ornekler: List[KodOrnegi] = []
    for onek, devam, etiket in KALIPLAR:
        ornekler.append(KodOrnegi(onek=onek, devam=devam,
                                  etiket=etiket, kaynak="kalip"))
    return ornekler


def mufredat_ornekleri() -> List[KodOrnegi]:
    """
    Müfredattaki gerçek kod bloklarından **satır satır** örnek üretir.

    Her kod bloğu için: ilk N satır önek, sonraki satır devam. Böylece
    model gerçek dosya bağlamında tamamlamayı öğrenir.
    """
    from neurocore.neural.curriculum import build_curriculum

    ornekler: List[KodOrnegi] = []
    for ders in build_curriculum():
        kod = (ders.code or "").strip()
        if not kod or kod.startswith("#"):
            continue
        try:
            ast.parse(kod)
        except SyntaxError:
            continue
        satirlar = kod.splitlines()
        # 2..4 satırlık önekler
        for n in (2, 3, 4):
            if len(satirlar) <= n:
                continue
            onek = "\n".join(satirlar[:n])
            devam = "\n".join(satirlar[n:n + 2])
            ornekler.append(KodOrnegi(onek=onek, devam=devam,
                                      etiket=ders.layer, yeni_satir=True,
                                      kaynak=f"mufredat:{ders.title[:40]}"))
    return ornekler


def tum_kod_ornekleri() -> List[KodOrnegi]:
    """Kalıp + müfredat örneklerini birleştirir (tekrarları atar)."""
    gorulen: set = set()
    sonuc: List[KodOrnegi] = []
    for o in kalip_ornekleri() + mufredat_ornekleri():
        anahtar = (o.onek, o.devam)
        if anahtar in gorulen:
            continue
        gorulen.add(anahtar)
        sonuc.append(o)
    return sonuc


def birlestir(onek: str, devam: str, yeni_satir: bool = False) -> str:
    """
    Önek ile devamı **geçerli Python** olacak şekilde birleştirir.

    ⚠ Ölçülen sorun: doğrulayıcı devamı her zaman yeni satıra koyuyordu.
    Oysa kalıpların çoğu **aynı satırı tamamlar**:

        önek : "ob.name = "
        devam: "'Govde'"
        → doğru birleşim: "ob.name = 'Govde'"   (tek satır)
        → yanlış birleşim: "ob.name = \\n'Govde'"  (SyntaxError)

    Bu yüzden: devam **girintili** başlıyorsa (yeni blok) yeni satıra,
    aksi hâlde son satıra eklenir.
    """
    if not onek:
        return devam
    if not devam:
        return onek
    ilk = devam.splitlines()[0] if devam else ""
    yeni = (yeni_satir
            or devam.startswith(("\n", " ", "\t"))
            or ilk.lstrip() != ilk
            or onek.rstrip().endswith(":"))
    if yeni:
        return onek.rstrip("\n") + "\n" + devam
    return onek + devam


def _eksik_blok_kapat(kod: str) -> str:
    """
    Açık kalan blokları `pass` ile kapatır (test amaçlı).

    ⚠ Neden gerekli: kod parçaları doğal olarak **yarımdır**:

        "for item in sorted(kok.iterdir()):"
        → `ast.parse` başarısız: "expected an indented block"

    Bu **doğru kod**dur; gövdesi bir sonraki parçadadır. Girinti
    seviyesini otomatik bulup `pass` ekleriz.
    """
    satirlar = kod.rstrip().splitlines()
    if not satirlar:
        return kod
    # Son satırın girintisi + 4 (yeni blok seviyesi)
    son = satirlar[-1]
    mevcut = len(son) - len(son.lstrip())
    # Açık parantez sayısına göre kapanış ekle
    acik = 0
    for s in satirlar:
        temiz = s.split("#")[0]
        acik += temiz.count("(") - temiz.count(")")
        acik += temiz.count("[") - temiz.count("]")
        acik += temiz.count("{") - temiz.count("}")
    ek = ""
    if acik > 0:
        ek = ")" * acik
    return kod.rstrip() + ek + "\n" + " " * (mevcut + 4) + "pass"


def _gecerli_mi(kod: str) -> bool:
    """Parça bağlamla geçerli mi (eksik blok kapatılarak denenir)."""
    if not kod.strip():
        return False
    for deneme in (kod, _eksik_blok_kapat(kod)):
        try:
            ast.parse(deneme)
            return True
        except SyntaxError:
            continue
    return False


def dogrula() -> Dict[str, Any]:
    """
    Kod örneklerini doğrular.

    Ölçüt: **birleşim** geçerli Python olmalı (eksik blok kapatılarak
    denenir). Önek tek başına yarım olabilir — devam onu tamamlar.

    ⚠ Kritik: modele **bozuk kod** öğretmek ona yanlış kalıp öğretir.
    Geçersiz örnekler ayıklanır.
    """
    gecerli: List[KodOrnegi] = []
    gecersiz: List[Dict[str, str]] = []
    for o in tum_kod_ornekleri():
        tam = o.tam_metin()
        if _gecerli_mi(tam):
            gecerli.append(o)
            continue
        # Devam yeni bir blok başlatıyorsa (önek ":" ile bitiyor)
        # girintiyi koruyarak tekrar dene
        if o.onek.rstrip().endswith(":") and _gecerli_mi(
                o.onek.rstrip() + "\n    " + o.devam.lstrip()):
            gecerli.append(o)
            continue
        try:
            ast.parse(tam)
            gecerli.append(o)
        except SyntaxError as exc:
            gecersiz.append({"onek": o.onek[:70], "devam": o.devam[:50],
                             "hata": str(exc)[:70]})
    return {
        "toplam": len(gecerli) + len(gecersiz),
        "gecerli": len(gecerli),
        "gecersiz": len(gecersiz),
        "oran": len(gecerli) / max(1, len(gecerli) + len(gecersiz)),
        "hatalar": gecersiz[:10],
        "ornekler": gecerli,
    }


if __name__ == "__main__":
    import json

    r = dogrula()
    print("=" * 74)
    print("  KOD ÖĞRENME SETİ")
    print("=" * 74)
    print(f"  toplam örnek : {r['toplam']}")
    print(f"  geçerli      : {r['gecerli']}")
    print(f"  geçersiz     : {r['gecersiz']}")
    print(f"  oran         : %{r['oran'] * 100:.1f}")
    if r["hatalar"]:
        print("\n  GEÇERSİZ ÖRNEKLER:")
        for h in r["hatalar"]:
            print(f"    önek : {h['onek']!r}")
            print(f"    devam: {h['devam']!r}")
            print(f"    hata : {h['hata']}")
    # Etiket dağılımı
    from collections import Counter
    dagilim = Counter(o.etiket for o in r["ornekler"])
    print("\n  ETİKET DAĞILIMI:")
    for k, v in dagilim.most_common():
        print(f"    {k:16} {v:3}")
    print("\n  ÖRNEK ÇİFTLER:")
    for o in r["ornekler"][:3]:
        print(f"    ÖNEK : {o.onek[:60]!r}")
        print(f"    DEVAM: {o.devam[:60]!r}")
        print()
