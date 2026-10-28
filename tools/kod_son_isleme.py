"""
Kod Son İşleme — Yazım Düzeltmesi + Uydurma API Filtresi
==========================================================
⚠ Ölçülen 2 hata kalıbı (tur 28)
--------------------------------
**1. Nokta → alt çizgi (yazım hatası)**
```
model:   bpy_data.meshes.new()          ← YANLIŞ
doğru:   bpy.data.meshes.new()
```
Model **doğru API'yi düşünüyor** ama yanlış yazıyor. Bu bir **yazım**
sorunu — kural tabanlı düzeltme güvenli.

**2. Uydurma API**
```
model:   bpy.ops.scene.add_object(...)   ← Blender'da yok
model:   b3.b3_box_add_to_mesh(...)      ← anlamsız
```

Çözüm: **son işleme katmanı**
-----------------------------
    ┌──────────────────────────────────────┐
    │ 1. Model çıktısı                     │
    ├──────────────────────────────────────┤
    │ 2. kod_disi_kes   (açıklama metni)   │
    ├──────────────────────────────────────┤
    │ 3. tekrar_kes     (döngü)            │
    ├──────────────────────────────────────┤
    │ 4. YAZIM DÜZELT   (bu araç)          │
    │    bpy_data → bpy.data               │
    ├──────────────────────────────────────┤
    │ 5. UYDURMA FİLTRE (bu araç)          │
    │    sözlükte olmayan bpy.ops reddet   │
    ├──────────────────────────────────────┤
    │ 6. Kalite kapısı → geçmezse şablon   │
    └──────────────────────────────────────┘

Kullanım
--------
    python tools/kod_son_isleme.py --test
    python tools/kod_son_isleme.py --duzelt "bpy_data.meshes.new()"
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

SOZLUK_YOLU = Path(".neurocore/api_sozlugu.json")

#: Yazım düzeltmeleri — modelin yaptığı **ölçülmüş** hatalar
#:
#: ⚠ Kural: yalnız **kesin** düzeltmeler. Şüpheli olan eklenmez.
YAZIM_DUZELTME: Sequence[Tuple[str, str]] = (
    # Nokta → alt çizgi (en sık hata)
    ("bpy_data", "bpy.data"),
    ("bpy_ops", "bpy.ops"),
    ("bpy_types", "bpy.types"),
    ("bpy_context", "bpy.context"),
    ("bpy_props", "bpy.props"),
    ("bpy_utils", "bpy.utils"),
    ("bpy_path", "bpy.path"),
    ("bpy_app", "bpy.app"),
    ("bpy_msgbus", "bpy.msgbus"),
    ("mathutils_", "mathutils."),
    ("bmesh_", "bmesh."),
    # Çift nokta
    ("bpy..", "bpy."),
    ("..data", ".data"),
    # Yaygın harf hataları
    ("bpy.objct", "bpy.object"),
    ("bpy.data.material.new", "bpy.data.materials.new"),
    ("bpy.data.mesh.new", "bpy.data.meshes.new"),
    ("bpy.data.image.new", "bpy.data.images.new"),
    ("bpy.data.object.new", "bpy.data.objects.new"),
)

#: Uydurma API kalıpları — sözlükte olmasa da **kesin** yanlış olanlar
UYDURMA_DESEN: Sequence[str] = (
    r"\bb\d\.b\d_",              # `b3.b3_box_add_to_mesh` — anlamsız
    r"\bbpy\.ops\.scene\.add_object\b",
    r"\bbpy\.ops\.select_all_with_",
    r"\bnew_from_object_data\b",
    r"\bbpy_object\.new_",
    r"\bbpy_data\.layers\.",
    r"\bscene\.renderer\.",      # `scene.render.engine` doğru
    r"\bmesh2\.vertices",
    r"\bv_copy\.copy\(",
)


@dataclass
class IslemSonuc:
    """Son işleme sonucu."""
    kod: str = ""
    duzeltme: List[str] = field(default_factory=list)
    uydurma: List[str] = field(default_factory=list)
    gecerli: bool = True
    neden: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"duzeltme": self.duzeltme[:6], "uydurma": self.uydurma[:4],
                "gecerli": self.gecerli, "neden": self.neden[:100],
                "kod": self.kod[:300]}


class SonIsleme:
    """Kod çıktısını düzeltir ve doğrular."""

    def __init__(self) -> None:
        self.adlar: Set[str] = set()
        if SOZLUK_YOLU.exists():
            try:
                d = json.loads(SOZLUK_YOLU.read_text(encoding="utf-8"))
                self.adlar = set(d.get("adlar") or [])
            except Exception:
                pass
        # Godot API'leri (sözlükte yok — bpy sözlüğü)
        self.godot_onek = ("MeshInstance3D", "SurfaceTool", "BoxMesh",
                           "CylinderMesh", "SphereMesh", "PrismMesh",
                           "StandardMaterial3D", "FastNoiseLite",
                           "CharacterBody3D", "Node3D", "Vector3",
                           "Transform3D", "AnimationPlayer", "Skeleton3D",
                           "BoneAttachment3D", "OmniLight3D", "Area3D",
                           "CollisionShape3D", "CapsuleShape3D",
                           "BoxShape3D", "SphereShape3D", "Time", "Input",
                           "Engine", "OS", "FileAccess", "JSON",
                           "PackedByteArray", "Color", "Basis", "Quaternion")

    # ─── Yazım düzeltmesi ───

    def yazim_duzelt(self, kod: str) -> Tuple[str, List[str]]:
        """
        **Ölçülmüş** yazım hatalarını düzeltir.

        ⚠ Yalnız kesin kurallar — şüpheli düzeltme yapılmaz. Her
        değişiklik **kaydedilir** (izlenebilirlik).
        """
        yapilan: List[str] = []
        for yanlis, dogru in YAZIM_DUZELTME:
            if yanlis in kod:
                kod = kod.replace(yanlis, dogru)
                yapilan.append(f"{yanlis} → {dogru}")
        return kod, yapilan

    # ─── Uydurma API filtresi ───

    def uydurma_bul(self, kod: str) -> List[str]:
        """
        **Kesin yanlış** API kalıplarını bulur.

        ⚠ Neden "kesin"? Sözlük eksik olduğu için (tur 24'te ölçüldü)
        genel allowlist zarar veriyordu. Burada yalnız **anlamsız**
        kalıplar (`b3.b3_...`) denetlenir.
        """
        bulunan: List[str] = []
        for desen in UYDURMA_DESEN:
            for m in re.finditer(desen, kod):
                bulunan.append(m.group(0)[:40])
        return list(dict.fromkeys(bulunan))[:5]

    # ─── Godot doğrulama ───

    def godot_gecerli(self, kod: str) -> Tuple[bool, str]:
        """
        Godot GDScript çıktısı **yapısal** olarak geçerli mi?

        ⚠ Denetlenen (Godot headless olmadan yapılabilen kontroller):
            * Parantez/köşeli parantez dengesi
            * `func` satırlarında `:` var mı
            * Python kalıntısı (`{ }` blok, `def `)
            * Tire içeren tanımlayıcı (`move_and-slide` — geçersiz)
        """
        # Parantez dengesi
        for ac, kapa, ad in (("(", ")", "parantez"),
                             ("[", "]", "köşeli"),
                             ("{", "}", "süslü")):
            if kod.count(ac) != kod.count(kapa):
                return False, f"{ad} dengesiz ({kod.count(ac)}≠{kod.count(kapa)})"
        # `func` satırlarında iki nokta
        for satir in kod.splitlines():
            s = satir.strip()
            if s.startswith("func ") and not s.rstrip().endswith(":"):
                if "->" in s or "(" in s:
                    return False, f"func satırında ':' yok: {s[:40]}"
        # Python kalıntısı
        if re.search(r"^\s*def\s+\w+\s*\(", kod, re.M):
            return False, "Python `def` kalıntısı"
        if re.search(r"\bfunc\s+\w+[^\n]*\)\s*\{", kod):
            return False, "Godot'ta `{` blok yok (Python/C kalıntısı)"
        # Tire içeren tanımlayıcı
        tir = re.findall(r"\b([a-z_]\w*-[\w-]+)\s*\(", kod)
        if tir:
            return False, f"tire içeren ad: {tir[0][:30]}"
        return True, ""

    # ─── Kod dışı metin denetimi ───

    #: İngilizce işlev sözcükleri — düz yazı göstergesi
    PROSE_KELIME = (
        "the", "a", "an", "this", "that", "these", "those", "with",
        "and", "or", "but", "is", "are", "was", "were", "will",
        "can", "should", "would", "you", "your", "it", "its",
        "above", "below", "following", "however", "therefore",
        "creates", "returns", "defines", "represents", "means",
        "note", "here", "there", "when", "then", "which", "where",
        "for", "of", "to", "be", "been", "has", "have", "do", "does",
        "make", "makes", "use", "used", "using", "we", "our", "they",
        "from", "at", "on", "by", "so", "if", "any", "all", "some",
        # ⚠ tur 39: Godot istemlerinde görülen prose kalıpları
        "first", "second", "third", "step", "case", "example", "used",
        "such", "also", "more", "most", "other", "another", "each",
        "same", "different", "well", "just", "only", "even", "both",
    )

    #: Godot anahtar sözcükleri — **kod göstergesi**.
    #:
    #: ⚠ ÖLÇÜLEN TUZAK: ilk sürümde `is`, `in`, `as`, `or`, `and`
    #: vardı. Ama bunlar **İngilizce işlev sözcükleri** de:
    #:     `'The first step is to create a surface tool.'`
    #: Bu satırda `is` geçtiği için **kod sayıldı** ve düz yazı tespiti
    #: çalışmadı. Yalnız **İngilizce'de görülmeyen** anahtar sözcükler
    #: bırakıldı.
    KOD_ANAHTAR = (
        "func", "var", "const", "class_name", "extends", "signal",
        "enum", "elif", "await", "yield", "static", "null", "void",
        "onready", "export", "print", "self", "true", "false",
        "Vector3", "Color", "String", "Array", "Dictionary",
        "MeshInstance3D", "SurfaceTool", "BoxMesh", "CylinderMesh",
    )

    def kod_disi_var(self, kod: str) -> Tuple[bool, str]:
        """
        Kod gövdesinde **düz yazı (prose)** var mı?

        ⚠ ÖLÇÜLEN SORUN: Godot gerçek derlemede şunu verdi:
            `Unexpected identifier "The" in class body.`
        Çünkü model kod yerine **İngilizce açıklama** üretiyor:
            `'return m\\n\\nThe above code creates a standard material...'`

        ⚠ ÖLÇÜLEN TUZAK (ilk sürüm): kural "`.` içeren satır koddur"
        diyordu. Ama **cümle sonu noktası** da `.` — bu yüzden gerçek
        düz yazı **kod sayıldı** ve ret çalışmadı.

        Doğru ölçüt — **iki sinyal birlikte**:
            1. Satırda **İngilizce işlev sözcüğü** var (`the`, `with`…)
            2. Satırda **Godot anahtar sözcüğü** YOK
            3. Kod operatörü (`:=`, `->`, `==`) YOK
        """
        for satir in kod.splitlines():
            s = satir.strip()
            # ⚠ tur 39: eşik 20 → 14 karakter. Model kısa prose
            # üretiyordu (`'In this case...'`) ve eşiğin altında kalıyordu.
            if len(s) < 14:
                continue
            kelimeler = re.findall(r"[A-Za-z']+", s.lower())
            if len(kelimeler) < 3:
                continue
            # Kod operatörleri → kod satırı
            if any(op in s for op in (":=", "->", "==", "!=", "+=",
                                      "-=", "*=", "/=", "&&", "||")):
                continue
            # Atama veya çağrı → kod satırı
            if "=" in s or "(" in s or ";" in s:
                continue
            # Godot anahtar sözcüğü → kod satırı
            if any(k in kelimeler for k in self.KOD_ANAHTAR):
                continue
            # İngilizce işlev sözcüğü → düz yazı
            prose = [k for k in kelimeler if k in self.PROSE_KELIME]
            if len(prose) >= 2:
                return True, f"kod dışı metin: {s[:50]!r}"
        return False, ""

    #: Zorunlu parametreleri olan Blender API'leri.
    #:
    #: ⚠ ÖLÇÜLEN (tur 35): gerçek Blender'da şu hatalar alındı:
    #:     `BlendDataMeshes.new(): required parameter "name" not specified`
    #:     `BlendDataImages.new(): required parameter "width" not specified`
    #:     `bpy_struct: attribute "vertices" from "Mesh" is read-only`
    #:     `positional argument follows keyword argument`
    #: Bunlar **yapısal** hatalar — derleme öncesi yakalanabilir.
    ZORUNLU_PARAM: Dict[str, Tuple[str, ...]] = {
        "bpy.data.meshes.new": ("name",),
        "bpy.data.materials.new": ("name",),
        "bpy.data.images.new": ("name", "width"),
        "bpy.data.objects.new": ("name",),
        "bpy.data.cameras.new": ("name",),
        "bpy.data.lights.new": ("name",),
        "bpy.data.collections.new": ("name",),
        "bpy.data.textures.new": ("name",),
    }

    #: Salt-okunur Blender koleksiyonları — **atama** yapılırsa hata.
    #: (Metot çağrısı `mesh.vertices.append(...)` geçerlidir.)
    SALT_OKUNUR = (
        "mesh.vertices", "mesh.polygons", "mesh.edges", "mesh.loops",
        "img.pixels", "scene.objects",
    )

    def blender_yapisal(self, kod: str) -> Tuple[bool, str]:
        """
        **Blender yapısal** denetimi — derleme öncesi.

        ⚠ ÖLÇÜLEN 4 hata kalıbı (gerçek Blender, 0/6 çalıştı):
            1. Zorunlu parametre eksik (`meshes.new()` → `name` gerekli)
            2. Salt-okunur koleksiyona atama (`mesh.vertices = ...`)
            3. Anahtar sözcükten sonra konumsal argüman
            4. Kapanmamış parantez
        """
        # ── 1) Zorunlu parametreler ──
        for api, params in self.ZORUNLU_PARAM.items():
            for m in re.finditer(re.escape(api) + r"\s*\(([^)]*)", kod):
                icerik = m.group(1)
                for p in params:
                    if f"{p}=" not in icerik:
                        return False, f"{api}(): '{p}' parametresi yok"
        # ── 2) Salt-okunur koleksiyona atama ──
        for alan in self.SALT_OKUNUR:
            if re.search(re.escape(alan) + r"\s*=(?!=)", kod):
                return False, f"salt-okunur alana atama: {alan}"
        # ── 3) Anahtar sözcükten sonra konumsal argüman ──
        for m in re.finditer(r"\(([^()]*)\)", kod):
            icerik = m.group(1)
            if "=" not in icerik:
                continue
            parcalar = [p.strip() for p in icerik.split(",") if p.strip()]
            gorulen_kw = False
            for p in parcalar:
                if re.match(r"^\w+\s*=(?!=)", p):
                    gorulen_kw = True
                elif gorulen_kw and not p.startswith(("*", ")")):
                    return False, ("anahtar sözcükten sonra konumsal "
                                   f"argüman: {p[:30]}")
        # ── 4) Geçersiz modül yolu ──
        #
        # ⚠ ÖLÇÜLEN (tur 35): model `bpy.mesh.vertices` üretti →
        # gerçek Blender: `module 'bpy' has no attribute 'mesh'`.
        # `bpy`'nin geçerli üst alanları **API sözlüğünden** bilinir.
        if self.adlar:
            gecerli: Set[str] = set()
            for a in self.adlar:
                if a.startswith("bpy."):
                    parca = a.split(".")
                    if len(parca) >= 2:
                        gecerli.add(parca[1])
            if gecerli:
                for m in re.finditer(r"\bbpy\.(\w+)", kod):
                    alan = m.group(1)
                    if alan not in gecerli:
                        return False, f"geçersiz bpy alanı: bpy.{alan}"
        # ── 5) String enum yerine sayı ──
        #
        # ⚠ ÖLÇÜLEN (tur 36): `scene.render.engine = 3` →
        # gerçek Blender: `expected a string enum, not int`.
        # Blender enum alanları **string** ister ('CYCLES', 'BLENDER_EEVEE').
        ENUM_ALAN = ("scene.render.engine", "image.file_format",
                     "ob.mode", "material.blend_method",
                     "scene.render.film_transparent")
        for alan in ENUM_ALAN:
            if re.search(re.escape(alan) + r"\s*=\s*\d", kod):
                return False, f"enum alanına sayı: {alan}"
        # ── 6) Kapanmamış parantez ──
        if kod.count("(") != kod.count(")"):
            return False, (f"parantez dengesiz "
                           f"({kod.count('(')}≠{kod.count(')')})")
        return True, ""

    # ─── Ana işlem ───

    def isle(self, kod: str, *, tur: str = "blender") -> IslemSonuc:
        """Tüm son işleme adımlarını uygular."""
        s = IslemSonuc()
        kod, s.duzeltme = self.yazim_duzelt(kod)
        s.uydurma = self.uydurma_bul(kod)
        if s.uydurma:
            s.gecerli = False
            s.neden = f"uydurma API: {s.uydurma[:2]}"
            s.kod = kod
            return s
        # ── Kod dışı metin (düz yazı) denetimi ──
        #
        # ⚠ Ölçülen: Godot derlemesi `Unexpected identifier "The"` verdi.
        # Kesme yetmediği için **reddetme** eklendi.
        disi, neden = self.kod_disi_var(kod)
        if disi:
            s.gecerli = False
            s.neden = neden
            s.kod = kod
            return s
        if tur == "godot":
            ok, neden = self.godot_gecerli(kod)
            if not ok:
                s.gecerli = False
                s.neden = neden
                s.kod = kod
                return s
        else:
            # ── Blender yapısal denetimi ──
            #
            # ⚠ Ölçülen (tur 35): gerçek Blender'da 0/6 çalıştı.
            # Sebepler: zorunlu parametre eksikliği, salt-okunur atama,
            # argüman sırası. Hepsi **derleme öncesi** yakalanabilir.
            ok, neden = self.blender_yapisal(kod)
            if not ok:
                s.gecerli = False
                s.neden = neden
                s.kod = kod
                return s
        s.kod = kod
        s.neden = "geçerli"
        return s


def main() -> int:
    """Son işleme testlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="Kod son işleme")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--duzelt", default="", help="Metni düzelt ve göster")
    args = ap.parse_args()

    print("=" * 78)
    print("  KOD SON İŞLEME — yazım düzeltmesi + uydurma filtresi")
    print("=" * 78)

    s = SonIsleme()
    print(f"  sözlük: {len(s.adlar)} API adı")
    sonuc: Dict[str, Any] = {}

    if args.duzelt:
        r = s.isle(args.duzelt)
        print(f"\n  GİRDİ  : {args.duzelt[:70]!r}")
        print(f"  ÇIKTI  : {r.kod[:70]!r}")
        print(f"  düzeltme: {r.duzeltme}")
        print(f"  geçerli : {r.gecerli} ({r.neden[:40]})")
        return 0

    if args.test:
        testler: List[Tuple[str, str, bool]] = [
            # (kod, tür, beklenen_gecerli)
            # ⚠ tur 35: `meshes.new()` zorunlu `name` parametresi ister;
            # yazım düzeltmesinden sonra da **reddedilir** (doğru davranış).
            ("bpy_data.meshes.new(name='M')", "blender", True),
            ("bpy_ops.mesh.primitive_cube_add()", "blender", True),
            ("bpy.data.materials.new(name='M')", "blender", True),
            ("b3.b3_box_add_to_mesh()", "blender", False),
            ("bpy.ops.scene.add_object()", "blender", False),
            ("scene.renderer.engine = 3", "blender", False),
            ("v_copy.copy(v)", "blender", False),
            # Godot
            ("func uret() -> void:\n    pass\n", "godot", True),
            ("func uret() -> void\n    pass\n", "godot", False),
            ("func move_and-slide() -> void:\n    pass\n", "godot", False),
            ("func uret() -> void: {\n    pass\n", "godot", False),
            ("mesh.vertices.append((0, 0, 0))\n", "godot", True),
        ]
        print("\n  SON İŞLEME TESTLERİ:")
        gecen = 0
        for kod, tur, beklenen in testler:
            r = s.isle(kod, tur=tur)
            ok = r.gecerli == beklenen
            if ok:
                gecen += 1
            im = "✓" if ok else "✗"
            ek = f" [{', '.join(r.duzeltme)}]" if r.duzeltme else ""
            print(f"    {im} [{tur:7}] {kod.splitlines()[0][:38]:40} "
                  f"geçerli={r.gecerli}{ek}")
            sonuc.setdefault("testler", []).append(
                {"kod": kod[:60], "tur": tur, "beklenen": beklenen,
                 "gelen": r.gecerli, "neden": r.neden})
        print(f"\n  TEST SONUCU: {gecen}/{len(testler)}")
        sonuc["test_gecen"] = gecen
        sonuc["test_toplam"] = len(testler)

    Path("_son_isleme.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
