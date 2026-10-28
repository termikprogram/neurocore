"""
Kod Verisi Üretici — Öğretmen Modelle Sentetik Örnek
=====================================================
Yerel modelin kod öğrenmesi için **toplu** kod tamamlama örneği üretir.

Neden gerekli?
    Ölçüm gösterdi: 196 kod notu (18,5k token) **yetmiyor**.

        Parametre : 211.255.328
        Kod verisi:    18.500 token
        Gereken   : 4.225.106.560 token  (20 × parametre)
        EKSİK     : 227.000×

    Eğitim turu ıraksadı (kayıp 0,084 → 0,121) ve geri alındı — çünkü
    öğrenilecek sinyal verinin içinde yok.

    Elle örnek yazmak bu açığı kapatmaz. **Öğretmen modelden toplu
    üretim** gerekir: her çağrıda 20-30 örnek, doğrulama ile ayıklama.

Yöntem
------
    1. **Konu listesi** — hangi API alanları kapsanacak
    2. **Toplu istek** — öğretmen modele "N örnek üret" (JSON çıktı)
    3. **Doğrulama** — her örnek `ast.parse` ile denetlenir
    4. **Kasaya yazma** — yalnızca geçerli örnekler
    5. **Tekrarları at** — aynı önek+devam iki kez yazılmaz

Kullanım
--------
    python tools/kod_verisi_uret.py --tur 40 --parti 25
    python tools/kod_verisi_uret.py --konu "bpy.ops.mesh" --tur 10
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

#: Kod üretim konuları — API alanı → açıklama
KONULAR: Sequence[Tuple[str, str]] = (
    ("bpy.ops.mesh primitive ekleme",
     "bpy.ops.mesh.primitive_* çağrıları: cube, cylinder, cone, uv_sphere, "
     "ico_sphere, grid, plane, torus, monkey. Parametreler: size, vertices, "
     "radius, depth, segments, ring_count, location, rotation"),
    ("mesh veri erişimi ve değiştirme",
     "bpy.context.object.data üzerinden vertices, polygons, edges, "
     "normals, uv_layers, color_attributes erişimi ve değiştirme"),
    ("yüz sayısı ve lowpoly azaltma",
     "yüz/köşe sayma, Decimate modifier, Limited Dissolve, üçgen sayısı "
     "hesabı, shade flat/smooth ayarları"),
    ("vertex color ve köşe renklendirme",
     "color_attributes.new, domain POINT/CORNER, FLOAT_COLOR/BYTE_COLOR, "
     "renk atama, Vertex Color düğümü ile malzemeye bağlama"),
    ("material ve Principled BSDF",
     "bpy.data.materials.new, use_nodes, node_tree, Principled BSDF "
     "girişleri: Base Color, Metallic, Roughness, Emission, Alpha"),
    ("texture ve UV bağlama",
     "bpy.data.images.new, pixels.foreach_set, ShaderNodeTexImage, "
     "ShaderNodeUVMap, ShaderNodeMapping, links.new ile UV bağlama"),
    ("mesh düğüm zinciri (shader nodes)",
     "ShaderNodeMix (RGBA), ShaderNodeSeparateColor, links.new, "
     "inputs[6]/inputs[7] A/B renkleri, inputs[0] Factor, outputs[2] Result"),
    ("arazi ve yükseklik haritası",
     "grid mesh üretimi, from_pydata, yükseklik atama, normalize, "
     "üs profili, ridged gürültü, fraktal toplam"),
    ("hidrolik erozyon",
     "yağmur damlası, 8 komşu akış, aşındırma, tortu biriktirme, "
     "deposit oranı, yineleme döngüsü"),
    ("ışık ve kamera kurulumu",
     "light_add SUN/POINT/SPOT/AREA, enerji, rotation_euler, "
     "camera_add, ORTHO/PERSP, ortho_scale, look_at"),
    ("render ayarları",
     "scene.render.engine CYCLES/EEVEE, cycles.device CPU/GPU, samples, "
     "resolution_x/y, filepath, render(write_still=True)"),
    ("dosya kaydetme ve dışa aktarma",
     "wm.save_as_mainfile, export_scene.gltf (GLB, animations, skins), "
     "export_scene.fbx, os.path.join, makedirs"),
    ("armature ve iskelet",
     "armature_add, edit_bones.new, kemik hiyerarşisi (parent), "
     "kemik uçları (head/tail), pose mode, iskelet adlandırma"),
    ("iskelet ağırlıklandırma (skinning)",
     "parent_set ARMATURE_AUTO, vertex_group, modifier ARMATURE, "
     "manuel ağırlık atama, deform kemikleri"),
    ("animasyon ve keyframe",
     "pose kemik döndürme, keyframe_insert, action, frame_set, "
     "NLA, animasyon eğrisi, döngü ayarı"),
    ("Godot temel sahne ve düğüm",
     "extends Node3D, _ready, _process, _physics_process, "
     "get_node, @onready, add_child, queue_free"),
    ("Godot sinyal sistemi",
     "signal tanımı, emit, connect, has_signal, "
     "yerleşik sinyaller (pressed, body_entered, timeout)"),
    ("Godot girdi ve WASD",
     "Input.get_vector, is_action_pressed, is_action_just_pressed, "
     "InputMap.add_action, action_add_event, InputEventKey"),
    ("Godot CharacterBody3D hareket",
     "velocity, move_and_slide, is_on_floor, yerçekimi, zıplama, "
     "move_toward, transform.basis"),
    ("Godot mesh üretimi",
     "ArrayMesh, SurfaceTool.begin/add_vertex/generate_normals/commit, "
     "BoxMesh, CylinderMesh, SphereMesh, radial_segments"),
    ("Godot material",
     "StandardMaterial3D, albedo_color, roughness, metallic, "
     "transparency, cull_mode, vertex_color_use_as_albedo"),
    ("Godot seviye ve labirent",
     "geri izleme algoritması, ızgara, duvar hücresi, "
     "create_trimesh_collision, NavigationRegion3D, bake_navigation_mesh"),
    ("Godot RPC ve çok oyunculu",
     "@rpc işaretleyicisi, rpc_id, multiplayer.is_server, "
     "ENetMultiplayerPeer, create_server/create_client, "
     "get_remote_sender_id doğrulaması"),
    ("Godot arayüz",
     "Control, VBoxContainer, HBoxContainer, GridContainer, anchor preset, "
     "ProgressBar, Label, Button, CanvasLayer"),
    ("Godot ses",
     "AudioStreamPlayer, AudioStreamPlayer3D, unit_size, max_distance, "
     "AudioServer bus, linear_to_db, max_polyphony"),
    ("Python matematik ve yardımcılar",
     "smoothstep, lerp, clamp, normalize, vektör işlemleri, "
     "tohum ile rastgele üretim, numpy dizi işlemleri"),
)

#: Üretim istemi — öğretmene gönderilir
ISTEM = """Sen bir kod öğretmenisin. Görevin, bir dil modeline **kod tamamlama** öğretmek için örnek çiftleri üretmek.

KONU: __KONU__
KAPSAM: __KAPSAM__

__ADET__ adet kod tamamlama örneği üret. Her örnek:
- `onek`: yarım kalmış kod parçası (1-4 satır). Modelin devamını üretmesi gerekecek.
- `devam`: öneği TAMAMLAYAN kod (1-3 satır).
- `etiket`: konu için kısa etiket (tek kelime, örn. "primitive")

KURALLAR:
1. `onek` + `devam` birleşince **geçerli Python/GDScript** olmalı.
2. Önek **anlamlı bir yerde** kesilmeli: parantez içi, `= ` sonrası, `:` sonrası, satır ortası.
3. `devam` **kısa** olsun (1-3 satır) — uzun bloklar öğretmek için uygun değil.
4. Kod **gerçek API** kullanmalı; uydurma fonksiyon yazma.
5. Örnekler **birbirinden farklı** olsun; aynı kalıbı tekrarlama.
6. Yalnızca JSON dizi döndür, açıklama yazma.

ÇIKTI BİÇİMİ (yalnızca bu JSON):
[{"onek": "...", "devam": "...", "etiket": "..."}, ...]

Şimdi __ADET__ örnek üret."""


@dataclass
class UretimSonuc:
    """Bir üretim turunun sonucu."""
    tur: int = 0
    konu: str = ""
    istenen: int = 0
    gelen: int = 0
    gecerli: int = 0
    yeni: int = 0
    sure_s: float = 0.0
    hata: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tur": self.tur, "konu": self.konu[:44],
            "istenen": self.istenen, "gelen": self.gelen,
            "gecerli": self.gecerli, "yeni": self.yeni,
            "sure_s": round(self.sure_s, 1), "hata": self.hata[:120],
        }


def _gecerli_mi(onek: str, devam: str) -> bool:
    """
    Önek + devam geçerli Python mu?

    ⚠ GDScript örnekleri Python değildir; onlar için yalnızca **yapı**
    kontrol edilir (parantez dengesi, boş olmama).
    """
    if not onek.strip() or not devam.strip():
        return False
    if len(onek) > 400 or len(devam) > 300:
        return False
    tam = onek + devam
    # GDScript işaretleri varsa yapı kontrolü
    if any(k in tam for k in ("extends ", "func _", "@export", "@rpc",
                              "var ", " := ")):
        # Parantez dengesi
        for ac, kapa in (("(", ")"), ("[", "]"), ("{", "}")):
            if tam.count(ac) != tam.count(kapa):
                return False
        return True
    try:
        ast.parse(tam if not onek.rstrip().endswith(":") else
                  onek + "\n    " + devam.lstrip())
        return True
    except SyntaxError:
        pass
    # Açık blok/parantez olabilir
    for ek in ("pass", ")", "))", "]"):
        try:
            ast.parse(onek + ek + "\n" + devam)
            return True
        except SyntaxError:
            continue
    return False


def _kayitlari_sec(kayitlar: Sequence[Any]) -> List[Dict[str, str]]:
    """Ham kayıtları doğrulanmış örnek sözlüklerine çevirir."""
    sonuc: List[Dict[str, str]] = []
    for kayit in kayitlar:
        if not isinstance(kayit, dict):
            continue
        onek = str(kayit.get("onek") or "").strip()
        devam = str(kayit.get("devam") or "").strip()
        etiket = str(kayit.get("etiket") or "kod").strip()[:24]
        if onek and devam:
            sonuc.append({"onek": onek, "devam": devam, "etiket": etiket})
    return sonuc


def _onerileri_ayikla(metin: str) -> List[Dict[str, str]]:
    """
    Öğretmen yanıtından örnekleri çıkarır — **kesik JSON'a dayanıklı**.

    ⚠ Ölçülen sorun: öğretmen `finish_reason="length"` ile yanıt veriyor,
    yani JSON dizisi **ortada kesiliyor**:

        [{"onek": "...", "devam": "..."}, {"onek": "...", "devam":

    `json.loads` bunu çözemez ve **0 örnek** döner. Oysa kesilen yanıtta
    onlarca **tam** nesne vardır (ölçüldü: 2419 karakterlik yanıt,
    sıfır örnek).

    Çözüm: (1) tam diziyi dene, (2) olmazsa **nesne nesne** ayıkla,
    (3) yarım kalan son nesneyi kapatmayı dene.
    """
    metin = (metin or "").strip()
    if not metin:
        return []
    blok = re.search(r"```(?:json)?\s*(.+?)```", metin, re.DOTALL)
    if blok:
        metin = blok.group(1).strip()

    bas = metin.find("[")
    son = metin.rfind("]")

    # ── 1) Tam dizi ──
    if bas >= 0 and son > bas:
        try:
            veri = json.loads(metin[bas:son + 1])
            if isinstance(veri, list):
                secili = _kayitlari_sec(veri)
                if secili:
                    return secili
        except Exception:
            pass

    # ── 2) Nesne nesne ayıkla (kesik yanıt) ──
    govde = metin[bas + 1:] if bas >= 0 else metin
    kayitlar: List[Any] = []
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

    # ── 3) Yarım kalan son nesneyi kapatmayı dene ──
    if baslangic >= 0:
        parca = govde[baslangic:]
        for ek in ('"}', '"}}', "}", '"]}'):
            try:
                kayitlar.append(json.loads(parca + ek))
                break
            except Exception:
                continue
    return _kayitlari_sec(kayitlar)


class KodVerisiUretici:
    """Öğretmen modelle toplu kod örneği üretir."""

    def __init__(self, *, konular: Optional[Sequence[Tuple[str, str]]] = None,
                 parti: int = 25, veri_yolu: str = ".neurocore") -> None:
        self.konular = list(konular) if konular else list(KONULAR)
        self.parti = max(5, min(40, parti))
        self.veri_yolu = veri_yolu
        self._gorulen: Set[Tuple[str, str]] = set()
        self._llm: Any = None
        self._kasa: Any = None
        self.sonuclar: List[UretimSonuc] = []
        self._ilerleme = Path(".neurocore/kod_verisi_ilerleme.json")

    # ─── Kurulum ───

    def baglan(self) -> bool:
        """Öğretmen modeli ve kasayı hazırlar."""
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
        self._gorulen = self._mevcut_onekler()
        print(f"  mevcut benzersiz önek: {len(self._gorulen)}")
        return True

    def _mevcut_onekler(self) -> Set[Tuple[str, str]]:
        """Kasada zaten olan önek+devam çiftleri (tekrarı önler)."""
        gorulen: Set[Tuple[str, str]] = set()
        if self._kasa is None:
            return gorulen
        try:
            for not_ in self._kasa.vault.iter_notes(limit=100000):
                kat = getattr(not_, "category", "") or ""
                if kat != "kod_uretim":
                    continue
                govde = getattr(not_, "body", "") or ""
                onek = self._ara(govde, "**Önek:**")
                devam = self._ara(govde, "**Devam:**")
                if onek and devam:
                    gorulen.add((onek, devam))
        except Exception as exc:
            print(f"  mevcut örnekler okunamadı: {exc}")
        return gorulen

    @staticmethod
    def _ara(metin: str, baslik: str) -> str:
        """Markdown gövdesinden kod bloğu çıkarır."""
        i = metin.find(baslik)
        if i < 0:
            return ""
        j = metin.find("```", i)
        if j < 0:
            return ""
        k = metin.find("\n", j)
        l = metin.find("```", k)
        if k < 0 or l < 0:
            return ""
        return metin[k + 1:l].strip()

    # ─── Üretim ───

    def _iste(self, konu: str, kapsam: str, adet: int) -> List[Dict[str, str]]:
        """Öğretmenden örnek ister."""
        if self._llm is None:
            return []
        from neurocore.agent.llm import LLMMessage

        # ⚠ `str.format` KULLANILMAZ: istem JSON örneği içeriyor ve
        # `{` karakterleri yer tutucu sanılıp `KeyError` verir —
        # bu yüzden öğretmen hiç yanıt vermiyordu (ölçüldü).
        istem = (ISTEM.replace("__KONU__", konu)
                      .replace("__KAPSAM__", kapsam)
                      .replace("__ADET__", str(adet)))
        try:
            if hasattr(self._llm, "chat_with_budget"):
                y = self._llm.chat_with_budget(
                    [LLMMessage("user", istem)], temperature=0.85)
            else:
                y = self._llm.chat([LLMMessage("user", istem)],
                                   max_tokens=6000, temperature=0.85)
        except Exception as exc:
            return []
        if not getattr(y, "ok", False):
            return []
        return _onerileri_ayikla(getattr(y, "text", "") or "")

    def tur(self, tur_no: int, konu_idx: int) -> UretimSonuc:
        """Bir üretim turu çalıştırır."""
        konu, kapsam = self.konular[konu_idx % len(self.konular)]
        s = UretimSonuc(tur=tur_no, konu=konu, istenen=self.parti)
        t0 = time.perf_counter()
        oneriler = self._iste(konu, kapsam, self.parti)
        s.gelen = len(oneriler)
        for o in oneriler:
            if not _gecerli_mi(o["onek"], o["devam"]):
                continue
            s.gecerli += 1
            anahtar = (o["onek"], o["devam"])
            if anahtar in self._gorulen:
                continue
            if self._yaz(o["onek"], o["devam"], o["etiket"], konu):
                self._gorulen.add(anahtar)
                s.yeni += 1
        if not oneriler:
            s.hata = "öğretmen yanıt vermedi veya JSON çözülemedi"
        s.sure_s = time.perf_counter() - t0
        self.sonuclar.append(s)
        return s

    def _yaz(self, onek: str, devam: str, etiket: str, konu: str) -> bool:
        """Örneği kasaya yazar."""
        if self._kasa is None:
            return False
        try:
            from neurocore.knowledge.vault import Note

            govde = (
                f"## Kod Tamamlama (üretilmiş)\n\n"
                f"**Konu:** {konu}\n\n"
                f"**Önek:**\n```python\n{onek}\n```\n\n"
                f"**Devam:**\n```python\n{devam}\n```\n\n"
                f"**Tam kod:**\n```python\n{onek}{devam}\n```\n"
            )
            simdi = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            n = Note(
                id="", title=f"KodU: {etiket} - {onek.splitlines()[-1][:36]}",
                body=govde, kind="skill", category="kod_uretim",
                tags=["kod", "tamamlama", "uretilmis", etiket],
                confidence=0.8, importance=0.6,
                created=simdi, updated=simdi, version=1,
                sources=["kod_verisi_uret"], links=[], backlinks=[],
                embedding_model="", embedding=None, path="", extra={})
            self._kasa.vault.write_note(n, touch=False, emit=False)
            return True
        except Exception:
            return False

    def kaydet(self) -> None:
        """İlerlemeyi diske yazar (devam edilebilir)."""
        try:
            self._ilerleme.write_text(json.dumps(
                {"tur": len(self.sonuclar),
                 "gorulen": len(self._gorulen),
                 "sonuclar": [s.to_dict() for s in self.sonuclar]},
                ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass


def main() -> int:
    """Toplu kod verisi üretir."""
    ap = argparse.ArgumentParser(description="Kod verisi üretici")
    ap.add_argument("--tur", type=int, default=20, help="Üretim turu")
    ap.add_argument("--parti", type=int, default=25, help="Tur başına örnek")
    ap.add_argument("--konu", default="", help="Yalnızca bu konu (alt dizge)")
    ap.add_argument("--veri", default=".neurocore", help="Veri yolu")
    args = ap.parse_args()

    print("=" * 76)
    print("  KOD VERİSİ ÜRETİCİ — öğretmen modelle toplu örnek")
    print("=" * 76)

    konular = list(KONULAR)
    if args.konu:
        k = args.konu.lower()
        konular = [x for x in konular if k in x[0].lower()]
        if not konular:
            print(f"  konu bulunamadı: {args.konu}")
            return 1

    u = KodVerisiUretici(konular=konular, parti=args.parti,
                         veri_yolu=args.veri)
    if not u.baglan():
        return 1

    toplam_yeni = 0
    toplam_gecerli = 0
    for i in range(args.tur):
        s = u.tur(i + 1, i)
        toplam_yeni += s.yeni
        toplam_gecerli += s.gecerli
        im = "+" if s.yeni > 0 else " "
        print(f"  {im} tur {s.tur:3}/{args.tur} | {s.konu[:34]:36} "
              f"gelen={s.gelen:3} geçerli={s.gecerli:3} YENİ={s.yeni:3} "
              f"({s.sure_s:.0f}s) {s.hata[:40]}")
        u.kaydet()

    print()
    print(f"  TOPLAM: geçerli={toplam_gecerli}  yeni={toplam_yeni}  "
          f"kasadaki benzersiz önek={len(u._gorulen)}")
    print(f"  ilerleme: {u._ilerleme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
