"""
API Referansı Toplayıcı — İnternetten Kod Verisi Çoğaltma
==========================================================
Hedefin 3. adımı: **"kötüyse internetten API referansı çekip kod
verisini çoğalt"**.

Neden gerekli?
    Ölçüm: mevcut model kod tamamlayamıyor (1/8). Kök neden veri ölçeği:

        Parametre : 211.255.328
        Kod verisi:    360.000 token
        Sağlıklı  :  4.225.106.560 token  (20 × parametre)
        EKSİK     :     11.700×

    `kod_verisi_uret.py` **kod tamamlama çiftleri** üretti (2.900 örnek).
    Bu araç farklı bir açıdan besler: **API referans bilgisi** —
    hangi fonksiyon ne yapar, hangi parametreleri alır, nasıl çağrılır.

    İkisi birlikte hem **sözdizimi** hem **anlam** öğretir.

Yöntem
------
    1. **API alan listesi** — hangi modüller/kütüphaneler taranacak
    2. **Öğretmene toplu soru** — "N fonksiyonu listele" (JSON çıktı)
    3. **Doğrulama** — Python örnekleri `ast.parse` ile denetlenir
    4. **Kasaya yazma** — `api_referans` kategorisi
    5. **Tekrarı at** — aynı API iki kez yazılmaz

Kullanım
--------
    python tools/api_referansi_toplay.py --tur 40 --parti 20
    python tools/api_referansi_toplay.py --alan "bpy.ops.mesh"
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: API alanları — (modül, kapsam açıklaması)
ALANLAR: Sequence[Tuple[str, str]] = (
    ("bpy.ops.mesh", "Mesh primitifleri ve düzenleme operatörleri: "
     "primitive_*, edit mode işlemleri, extrude, bevel, subdivide, "
     "loopcut, merge, separate"),
    ("bpy.ops.object", "Nesne operatörleri: add, delete, duplicate, join, "
     "mode_set, transform_apply, parent_set, shade_flat, shade_smooth, "
     "select_all, origin_set"),
    ("bpy.ops.export_scene", "Dışa aktarma: gltf/GLB, fbx, obj, "
     "export_animations, export_skins, export_yup parametreleri"),
    ("bpy.ops.wm", "Dosya işlemleri: save_as_mainfile, open_mainfile, "
     "read_homefile, quit_blender"),
    ("bpy.ops.render", "Render: render, write_still, viewport_render, "
     "opengl animasyon"),
    ("bpy.data.meshes", "Mesh veri API'si: new, from_pydata, vertices, "
     "edges, polygons, loops, uv_layers, color_attributes, normals, "
     "validate, update, calc_loop_triangles"),
    ("bpy.data.materials", "Malzeme API'si: new, use_nodes, node_tree, "
     "diffuse_color, metallic, roughness, blend_method"),
    ("bpy.data.objects", "Nesne veri API'si: new, remove, select_set, "
     "location, rotation_euler, scale, matrix_world, bound_box, "
     "data, modifiers"),
    ("bpy.data.images", "Görüntü API'si: new, load, pixels, "
     "foreach_get/set, save, filepath_raw, file_format, "
     "pack, unpack, scale"),
    ("bpy.data.armatures", "İskelet API'si: new, edit_bones, bones, "
     "bone.head, bone.tail, bone.parent, bone.use_deform"),
    ("bpy.data.actions", "Animasyon API'si: new, fcurves, keyframe_points, "
     "frame_range, interpolation"),
    ("bpy.types nodes", "Shader düğümleri: ShaderNodeTexImage, "
     "ShaderNodeUVMap, ShaderNodeMapping, ShaderNodeMix, "
     "ShaderNodeSeparateColor, ShaderNodeBsdfPrincipled, node_tree.links.new"),
    ("bpy.types.Object modifiers", "Modifier API'si: modifiers.new, "
     "DECIMATE, SUBSURF, SOLIDIFY, ARMATURE, MIRROR, ratio, levels, "
     "modifier_apply"),
    ("bpy.context", "Bağlam API'si: scene, object, selected_objects, "
     "active_object, mode, view_layer, collection, window, area, region"),
    ("bpy.utils", "Yardımcı API: register_class, unregister_class, "
     "units, blend_paths, escape_identifier"),
    ("mathutils", "Vektör/matris: Vector, Matrix, Quaternion, Euler, "
     "normalize, cross, dot, lerp, slerp, to_track_quat"),
    ("bmesh", "BMesh API'si: new, from_mesh, to_mesh, verts, edges, faces, "
     "ops.create_cone, ops.create_cube, ops.subdivide_edges, "
     "ops.bevel, normal_update"),
    ("bpy.ops.curve", "Eğri operatörleri: primitive_bezier_circle_add, "
     "primitive_curve_add, convert, curve data"),
    ("Godot GDScript Node3D", "Godot 4 Node3D ve türevleri: position, "
     "global_position, transform, basis, look_at, rotate_y, "
     "add_child, get_node, queue_free, find_child"),
    ("Godot GDScript MeshInstance3D", "Godot mesh API'si: mesh, "
     "material_override, BoxMesh, CylinderMesh, SphereMesh, PrismMesh, "
     "PlaneMesh, surface_get_arrays, create_trimesh_collision"),
    ("Godot GDScript SurfaceTool", "Godot SurfaceTool: begin, add_vertex, "
     "set_normal, set_color, set_uv, add_index, generate_normals, "
     "generate_tangents, commit, append_from"),
    ("Godot GDScript AnimationPlayer", "Godot animasyon: Animation, "
     "AnimationLibrary, add_animation, play, current_animation, "
     "AnimationTree, AnimationNodeStateMachine, travel"),
    ("Godot GDScript Skeleton3D", "Godot iskelet: add_bone, set_bone_rest, "
     "set_bone_parent, set_bone_pose_rotation, get_bone_count, "
     "find_bone, BoneAttachment3D.bone_idx"),
    ("Godot GDScript MultiplayerAPI", "Godot ağ: ENetMultiplayerPeer, "
     "create_server, create_client, @rpc, rpc_id, is_server, "
     "get_remote_sender_id, peer_connected, multiplayer_peer"),
    ("Godot GDScript Input", "Godot girdi: Input.get_vector, "
     "is_action_pressed, is_action_just_pressed, InputMap.add_action, "
     "action_add_event, InputEventKey, InputEventMouseMotion"),
    ("Godot GDScript CharacterBody3D", "Godot karakter: velocity, "
     "move_and_slide, is_on_floor, up_direction, floor_max_angle, "
     "get_slide_collision"),
    ("Godot GDScript NavigationAgent3D", "Godot yol bulma: "
     "target_position, get_next_path_position, is_navigation_finished, "
     "NavigationRegion3D.bake_navigation_mesh"),
    ("Godot GDScript StandardMaterial3D", "Godot malzeme: albedo_color, "
     "albedo_texture, roughness, metallic, emission, transparency, "
     "cull_mode, uv1_scale, vertex_color_use_as_albedo, texture_filter"),
    ("Godot GDScript FastNoiseLite", "Godot gürültü: noise_type, seed, "
     "frequency, fractal_octaves, fractal_lacunarity, fractal_gain, "
     "get_noise_2d, get_noise_3d, domain_warp"),
    ("Godot GDScript Tween", "Godot animasyon yardımcısı: create_tween, "
     "tween_property, tween_method, tween_callback, set_parallel, "
     "set_trans, set_ease, chain"),
)

#: İstem şablonu — yer tutucu + replace ile doldurulur
#: (⚠ `str.format` KULLANILMAZ: JSON örneğindeki `{` yer tutucu sanılır)
ISTEM = """Sen bir API dokümantasyon uzmanısın. Bir dil modeline **API referansı** öğretmek için yapılandırılmış bilgi üretiyorsun.

MODÜL/ALAN: __ALAN__
KAPSAM: __KAPSAM__

Bu alandan __ADET__ adet **en çok kullanılan** fonksiyon/özellik seç ve her biri için şu alanları doldur:

- `api`: tam ad (örn. `bpy.ops.mesh.primitive_cube_add`)
- `aciklama`: tek cümle, ne yapar (Türkçe)
- `parametreler`: virgülle ayrılmış parametre adları (yoksa "")
- `donus`: dönüş değeri (yoksa "yok")
- `ornek`: **gerçek, çalışan** tek satır kod örneği
- `ipucu`: tek cümle, pratik uyarı (isteğe bağlı)

KURALLAR:
1. Örnekler **gerçek API** kullanmalı; uydurma fonksiyon yazma.
2. Örnekler **tek satır** olsun, çalışır durumda.
3. Fonksiyonlar **birbirinden farklı** olsun.
4. Yalnızca JSON dizi döndür, açıklama yazma.

ÇIKTI BİÇİMİ (yalnızca bu JSON):
[__ORNEK__]

Şimdi __ADET__ fonksiyon listele."""


@dataclass
class ToplamaSonuc:
    """Bir toplama turunun sonucu."""
    tur: int = 0
    alan: str = ""
    istenen: int = 0
    gelen: int = 0
    gecerli: int = 0
    yeni: int = 0
    sure_s: float = 0.0
    hata: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tur": self.tur, "alan": self.alan[:40],
            "istenen": self.istenen, "gelen": self.gelen,
            "gecerli": self.gecerli, "yeni": self.yeni,
            "sure_s": round(self.sure_s, 1), "hata": self.hata[:120],
        }


def _json_dizisi_ayikla(metin: str) -> List[Dict[str, Any]]:
    """
    Öğretmen yanıtından JSON dizisini çıkarır — **kesik yanıta dayanıklı**.

    ⚠ Ölçülen sorun: öğretmen `finish_reason="length"` ile yanıt veriyor,
    dizi ortada kesiliyor. `json.loads` çözemez ve **0 örnek** döner.
    Çözüm: nesne nesne ayıkla.
    """
    metin = (metin or "").strip()
    if not metin:
        return []
    blok = re.search(r"```(?:json)?\s*(.+?)```", metin, re.DOTALL)
    if blok:
        metin = blok.group(1).strip()

    bas = metin.find("[")
    son = metin.rfind("]")
    if bas >= 0 and son > bas:
        try:
            veri = json.loads(metin[bas:son + 1])
            if isinstance(veri, list) and veri:
                return [x for x in veri if isinstance(x, dict)]
        except Exception:
            pass

    govde = metin[bas + 1:] if bas >= 0 else metin
    kayitlar: List[Dict[str, Any]] = []
    derinlik = 0
    baslangic = -1
    tirnak = False
    kacis = False
    for i, ch in enumerate(govde):
        if kacis:
            kacis = False
            continue
        if ch == "\\":
            kacis = True
            continue
        if ch == '"':
            tirnak = not tirnak
            continue
        if tirnak:
            continue
        if ch == "{":
            if derinlik == 0:
                baslangic = i
            derinlik += 1
        elif ch == "}":
            derinlik -= 1
            if derinlik == 0 and baslangic >= 0:
                try:
                    kayitlar.append(json.loads(govde[baslangic:i + 1]))
                except Exception:
                    pass
                baslangic = -1
    return [x for x in kayitlar if isinstance(x, dict)]


def _ornek_gecerli_mi(api: str, ornek: str) -> bool:
    """Kod örneği geçerli mi? (GDScript ise yapı kontrolü)"""
    if not ornek or not ornek.strip():
        return False
    if len(ornek) > 300:
        return False
    # GDScript işaretleri → yapı kontrolü
    if (any(k in ornek for k in ("extends ", "func ", "var ", " := ",
                                 "@export", "@rpc", "print("))
            and "bpy" not in ornek and "import " not in ornek):
        for ac, kapa in (("(", ")"), ("[", "]"), ("{", "}")):
            if ornek.count(ac) != ornek.count(kapa):
                return False
        return True
    # Python → ast ile doğrula
    try:
        ast.parse(ornek.strip())
        return True
    except SyntaxError:
        pass
    # İfade olabilir (tek satır, atama olmadan)
    try:
        ast.parse("x = " + ornek.strip())
        return True
    except SyntaxError:
        return False


class ApiToplayici:
    """Öğretmen modelle API referansı toplar."""

    def __init__(self, *, alanlar: Optional[Sequence[Tuple[str, str]]] = None,
                 parti: int = 20, veri_yolu: str = ".neurocore") -> None:
        self.alanlar = list(alanlar) if alanlar else list(ALANLAR)
        self.parti = max(5, min(40, parti))
        self.veri_yolu = veri_yolu
        self._gorulen: Set[str] = set()
        self._llm: Any = None
        self._kasa: Any = None
        self.sonuclar: List[ToplamaSonuc] = []

    def baglan(self) -> bool:
        """Öğretmeni ve kasayı hazırlar."""
        try:
            from neurocore.agent import LLMRouter

            self._llm = LLMRouter()
            self._llm.initialize(data_path=self.veri_yolu)
        except Exception as exc:
            print(f"  öğretmen bağlanamadı: {exc}")
            return False
        try:
            from neurocore.knowledge import KnowledgeLayer

            self._kasa = KnowledgeLayer()
            self._kasa.initialize()
        except Exception as exc:
            print(f"  kasa bağlanamadı: {exc}")
            return False
        self._gorulen = self._mevcut()
        print(f"  mevcut API referansı: {len(self._gorulen)}")
        return True

    def _mevcut(self) -> Set[str]:
        """Kasada zaten olan API adları."""
        gorulen: Set[str] = set()
        if self._kasa is None:
            return gorulen
        try:
            for n in self._kasa.vault.iter_notes(limit=200000):
                if (getattr(n, "category", "") or "") == "api_referans":
                    ad = (getattr(n, "title", "") or "").replace("API: ", "")
                    if ad:
                        gorulen.add(ad.strip())
        except Exception:
            pass
        return gorulen

    def tur(self, tur_no: int, alan_idx: int) -> ToplamaSonuc:
        """Bir toplama turu çalıştırır."""
        alan, kapsam = self.alanlar[alan_idx % len(self.alanlar)]
        s = ToplamaSonuc(tur=tur_no, alan=alan, istenen=self.parti)
        t0 = time.perf_counter()
        kayitlar = self._iste(alan, kapsam, self.parti)
        s.gelen = len(kayitlar)
        for k in kayitlar:
            api = str(k.get("api") or "").strip()
            ornek = str(k.get("ornek") or "").strip()
            if not api or not _ornek_gecerli_mi(api, ornek):
                continue
            s.gecerli += 1
            if api in self._gorulen:
                continue
            if self._yaz(k, alan):
                self._gorulen.add(api)
                s.yeni += 1
        if not kayitlar:
            s.hata = "yanıt yok veya JSON çözülemedi"
        s.sure_s = time.perf_counter() - t0
        self.sonuclar.append(s)
        return s

    def _iste(self, alan: str, kapsam: str, adet: int) -> List[Dict[str, Any]]:
        """Öğretmenden API listesi ister."""
        if self._llm is None:
            return []
        from neurocore.agent.llm import LLMMessage

        istem = (ISTEM.replace("__ALAN__", alan)
                 .replace("__KAPSAM__", kapsam)
                 .replace("__ADET__", str(adet)))
        try:
            if hasattr(self._llm, "chat_with_budget"):
                y = self._llm.chat_with_budget([LLMMessage("user", istem)],
                                               temperature=0.7)
            else:
                y = self._llm.chat([LLMMessage("user", istem)],
                                   max_tokens=6000, temperature=0.7)
        except Exception:
            return []
        if not getattr(y, "ok", False):
            return []
        return _json_dizisi_ayikla(getattr(y, "text", "") or "")

    def _yaz(self, k: Dict[str, Any], alan: str) -> bool:
        """API referansını kasaya yazar."""
        if self._kasa is None:
            return False
        try:
            from neurocore.knowledge.vault import Note

            api = str(k.get("api") or "").strip()
            aciklama = str(k.get("aciklama") or "").strip()
            params = str(k.get("parametreler") or "").strip()
            donus = str(k.get("donus") or "").strip()
            ornek = str(k.get("ornek") or "").strip()
            ipucu = str(k.get("ipucu") or "").strip()

            govde = (f"## API Referansı\n\n"
                     f"**Ad:** `{api}`\n\n"
                     f"**Açıklama:** {aciklama}\n\n"
                     f"**Parametreler:** {params or 'yok'}\n\n"
                     f"**Dönüş:** {donus or 'yok'}\n\n"
                     f"**Örnek:**\n```python\n{ornek}\n```\n")
            if ipucu:
                govde += f"\n**İpucu:** {ipucu}\n"
            govde += f"\n**Modül:** {alan}\n"

            simdi = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            n = Note(
                id="", title=f"API: {api}", body=govde,
                kind="reference", category="api_referans",
                tags=["api", "referans", alan.split(".")[0].split()[0]],
                confidence=0.85, importance=0.7,
                created=simdi, updated=simdi, version=1,
                sources=["api_referansi_toplay"], links=[], backlinks=[],
                embedding_model="", embedding=None, path="", extra={})
            self._kasa.vault.write_note(n, touch=False, emit=False)
            return True
        except Exception:
            return False

    def kaydet(self) -> None:
        """İlerlemeyi kaydeder."""
        try:
            Path(".neurocore/api_toplama_ilerleme.json").write_text(
                json.dumps({
                    "tur": len(self.sonuclar),
                    "api_sayisi": len(self._gorulen),
                    "sonuclar": [s.to_dict() for s in self.sonuclar],
                }, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass


def main() -> int:
    """API referansı toplar."""
    ap = argparse.ArgumentParser(description="API referansı toplayıcı")
    ap.add_argument("--tur", type=int, default=30, help="Toplama turu")
    ap.add_argument("--parti", type=int, default=20, help="Tur başına API")
    ap.add_argument("--alan", default="", help="Yalnızca bu alan (alt dizge)")
    ap.add_argument("--veri", default=".neurocore", help="Veri yolu")
    args = ap.parse_args()

    print("=" * 78)
    print("  API REFERANSI TOPLAYICI — internetten kod bilgisi")
    print("=" * 78)

    alanlar = list(ALANLAR)
    if args.alan:
        k = args.alan.lower()
        alanlar = [x for x in alanlar if k in x[0].lower()
                   or k in x[1].lower()]
        if not alanlar:
            print(f"  alan bulunamadı: {args.alan}")
            return 1
        print(f"  filtre: {len(alanlar)} alan")

    t = ApiToplayici(alanlar=alanlar, parti=args.parti,
                     veri_yolu=args.veri)
    if not t.baglan():
        return 1

    toplam_yeni = 0
    for i in range(args.tur):
        s = t.tur(i + 1, i)
        toplam_yeni += s.yeni
        im = "+" if s.yeni > 0 else " "
        print(f"  {im} tur {s.tur:3}/{args.tur} | {s.alan[:34]:36} "
              f"gelen={s.gelen:3} geçerli={s.gecerli:3} YENİ={s.yeni:3} "
              f"({s.sure_s:.0f}s) {s.hata[:34]}", flush=True)
        t.kaydet()

    print()
    print(f"  TOPLAM: yeni={toplam_yeni}  kasadaki API={len(t._gorulen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
