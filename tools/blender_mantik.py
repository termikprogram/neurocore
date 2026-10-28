"""
Blender Mantık Müfredatı — "Neden" Öğretimi
=============================================
⚠ Kullanıcı hedefi (F)
----------------------
*"eğer mantığını öğrenirse, nasıl yapacağı konusunda kendini
eğitebilir. blender eğitimi eksik ise tamamlayalım, blender a tam
hakimiyet sağlayalım."*

⚠ Neden "mantık"?
-----------------
Blender API'si **800+ operatör** ve **yüzlerce** özellik içerir.
Hepsini ezberlemek imkânsız. Ama **mantığını** bilen:
```
· Yeni bir operatör görünce **tahmin** edebilir
· Hata mesajından **kök nedeni** çıkarabilir
· Kendi çözümünü **türetebilir**
```

⚠ Ölçülen eksik (tur başı taraması)
-----------------------------------
```
Kasa: 1.375 Blender notu
  animation : 20 geçiş   ← ÇOK ZAYIF
  uv        : 27 geçiş   ← ÇOK ZAYIF
  armature  : 690 geçiş  ✓
  export    : 1.210 geçiş ✓
```

Bu müfredat **eksik alanlara** odaklanır + **mantığı** öğretir.

Bölümler
--------
    1. Koordinat ve uzay mantığı
    2. Nesne / mesh / veri ayrımı
    3. Modifier yığını mantığı
    4. Bağlam (`bpy.context`) mantığı
    5. Operatör vs veri erişimi
    6. Kemik uzayı ve weight mantığı
    7. UV ve dikiş mantığı
    8. Malzeme düğüm mantığı
    9. Animasyon ve fcurve mantığı
    10. Sürüm farkları ve hata ayıklama

Kullanım
--------
    python tools/blender_mantik.py --istatistik
    python tools/blender_mantik.py --kasaya
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: ═══════════════════════════════════════════════════════════════════
#: MÜFREDAT — her katman: (ad, açıklama, dersler)
#: ═══════════════════════════════════════════════════════════════════
#:
#: ⚠ Her ders **"neden"** sorusunu yanıtlar, yalnız "nasıl" değil.
MÜFREDAT: Sequence[Tuple[str, str, Sequence[Tuple[str, str, str]]]] = (

    ("01_uzay", "Koordinat ve uzay mantığı", (

        ("Blender'ın eksen takımı neden Z-yukarı?",
"Blender **Z-yukarı** (sağ el kuralı) kullanır; Unity/Godot "
         "Y-yukarı'dır. Bu yüzden GLTF'e aktarımda **eksen dönüşümü** "
         "olur: `+Y_gltf = +Z_blender`.\n\n"
         "⚠ Pratik sonuç: Blender'da karakter **Z ekseninde** ayakta "
         "durur; oyunda **Y ekseninde** durur.",
         "bpy.ops.export_scene.gltf(export_yup=True)  # varsayılan"),

        ("Yerel (local) vs dünya (world) koordinatı",
"Her nesnenin `location/rotation/scale`'i **yerel**dir "
         "(parent'a göre). `matrix_world` **dünya** uzayındadır.\n\n"
         "```python\n"
         "ob.location          # yerel (parent'a göre)\n"
         "ob.matrix_world      # dünya (sahne köküne göre)\n"
         "ob.matrix_world.translation  # dünya konumu (Vector)\n"
         "```\n"
         "⚠ Ölçülen tuzak: `ob.location` ile dünya konumunu okumak "
         "**yanlış** — parent varsa farklıdır.",
         "dunya = ob.matrix_world.translation"),

        ("Ölçek (scale) neden uygulanmalı?",
"`ob.scale` **nesne** dönüşümüdür; mesh verisi değişmez. "
         "Bevel/modifier'lar **mesh** uzayında çalışır → ölçek "
         "uygulanmazsa kalınlıklar **bozuk** çıkar.\n\n"
         "```python\n"
         "bpy.ops.object.transform_apply(scale=True)\n"
         "```\n"
         "⚠ Ölçüldü: bevel genişliği 0.012 iken ölçek 4 ise gerçek "
         "kalınlık 0.048 olur — beklenmeyen sonuç.",
         "bpy.ops.object.transform_apply(location=False,\n"
         "                               rotation=False, scale=True)"),

    )),

    ("02_veri", "Nesne / mesh / veri ayrımı", (

        ("Nesne (Object) vs Veri (Mesh) neden ayrı?",
"Blender'da **nesne** bir kapsayıcıdır (konum, parent, "
         "modifier); **mesh** ise saf geometridir. İki nesne **aynı "
         "mesh'i** paylaşabilir.\n\n"
         "```python\n"
         "ob = bpy.data.objects['Kup']   # nesne\n"
         "me = ob.data                    # mesh verisi\n"
         "ob2 = ob.copy(); ob2.data = ob.data  # mesh PAYLAŞILIR\n"
         "```\n"
         "⚠ Sonuç: `me.vertices` değiştirince **iki nesne de** "
         "değişir. Kopyalamak için `me.copy()`.",
         "me2 = ob.data.copy()  # bağımsız kopya"),

        ("Neden `bpy.data` ile `bpy.context` farklı?",
"`bpy.data` **tüm** veri bloklarını tutar (sahne dışı dahil); "
         "`bpy.context` **şu an** etkin olanı verir.\n\n"
         "```python\n"
         "len(bpy.data.meshes)      # sahnedeki + sahnede OLMAYAN\n"
         "bpy.context.active_object # yalnız etkin\n"
         "```\n"
         "⚠ Ölçülen tuzak: silinmiş nesnenin mesh'i `bpy.data`'da "
         "**kalır** → dosya şişer. `bpy.ops.outliner.orphans_purge()`.",
         "bpy.ops.outliner.orphans_purge(do_recursive=True)"),

        ("Normaller neden dışa bakmalı?",
"Yüz normali **hangi tarafın dış** olduğunu söyler. Ters "
         "normalli mesh: ışık yanlış, GLTF'te **görünmez**.\n\n"
         "```python\n"
         "bpy.ops.mesh.normals_make_consistent(inside=False)\n"
         "```\n"
         "⚠ Ölçüldü: parçaları `join` ile birleştirince normaller "
         "karışabilir → her birleştirmeden sonra düzelt.",
         "bpy.ops.mesh.normals_make_consistent(inside=False)"),

    )),

    ("03_modifier", "Modifier yığını mantığı", (

        ("Modifier sırası neden önemli?",
"Modifier'lar **yukarıdan aşağıya** uygulanır. `Subsurf → "
         "Bevel` ile `Bevel → Subsurf` **tamamen farklı** sonuç verir.\n\n"
         "```python\n"
         "# Subsurf ÖNCE → bevel yumuşak yüzeyde çalışır\n"
         "# Bevel ÖNCE  → subsurf bevel'i yumuşatır (istenmez)\n"
         "```\n"
         "⚠ Kural: **şekillendirme → detay → yumuşatma** sırası.",
         "# modifier listesi sırası = uygulama sırası"),

        ("`modifier_apply` neden gerekli?",
"Modifier **canlı** bir hesaplamadır; `me.vertices` "
         "**değişmez**. Vertex'lere erişmek için uygulamak şart.\n\n"
         "```python\n"
         "m = ob.modifiers.new('AltBolme', 'SUBSURF')\n"
         "m.levels = 3\n"
         "# ⚠ Bu noktada me.vertices HÂLÂ eski!\n"
         "bpy.ops.object.modifier_apply(modifier=m.name)\n"
         "# Artık me.vertices yeni\n"
         "```\n"
         "⚠ Ölçüldü: `me.vertices` sayısı uygulamadan **değişmiyor**.",
         "bpy.ops.object.modifier_apply(modifier=m.name)"),

        ("`SIMPLE` vs `CATMULL_CLARK` subdivision",
"`SIMPLE` yüzeyi **düz** böler (hacim korunur); "
         "`CATMULL_CLARK` **yuvarlar** (hacim küçülür).\n\n"
         "```python\n"
         "m.subdivision_type = 'SIMPLE'         # kaya, mimari\n"
         "m.subdivision_type = 'CATMULL_CLARK'  # organik\n"
         "```\n"
         "⚠ Kaya için `SIMPLE` doğru — `CATMULL_CLARK` kayayı "
         "**sabun** yapar.",
         "m.subdivision_type = 'SIMPLE'"),

    )),

    ("04_baglam", "Bağlam (bpy.context) mantığı", (

        ("Operatörler neden `context` ister?",
"`bpy.ops.*` operatörleri **etkin nesne + mod** üzerinde "
         "çalışır. Bağlam yanlışsa: `RuntimeError: context is "
         "incorrect`.\n\n"
         "```python\n"
         "bpy.context.view_layer.objects.active = ob   # etkin yap\n"
         "ob.select_set(True)                          # seç\n"
         "bpy.ops.object.mode_set(mode='EDIT')          # mod\n"
         "```\n"
         "⚠ Sıra önemli: **etkin → seç → mod**.",
         "bpy.context.view_layer.objects.active = ob"),

        ("`EDIT` vs `OBJECT` modu — hangi veri erişilebilir?",
"```python\n"
         "# OBJECT modunda:\n"
         "ob.data.vertices          # ✓ okunur\n"
         "# EDIT modunda:\n"
         "ob.data.vertices          # ⚠ güncel OLMAYABİLİR\n"
         "bmesh.from_edit_mesh(ob.data)   # ✓ doğru yol\n"
         "```\n"
         "⚠ Ölçülen tuzak: EDIT modunda `me.vertices` **eski** "
         "sayıyı verir.",
         "bm = bmesh.from_edit_mesh(ob.data)"),

        ("Headless (`--background`) bağlam farkı",
"`--background`'da **pencere yok** → bazı operatörler "
         "`context.window` ister ve **başarısız** olur.\n\n"
         "```python\n"
         "# ⚠ Headless'ta çalışmayabilir:\n"
         "bpy.ops.object.mode_set(mode='EDIT')\n"
         "# ✓ Alternatif: veri API'si\n"
         "for v in ob.data.vertices: v.co.z *= 0.7\n"
         "```\n"
         "⚠ Ölçüldü: bizim betikler `--background` ile çalışıyor; "
         "`mode_set` **çalışıyor** ama `uv.smart_project` için "
         "**seçim** şart.",
         "# headless'ta veri API'sini tercih et"),

    )),

    ("05_operator", "Operatör vs veri erişimi", (

        ("Ne zaman `bpy.ops`, ne zaman `bpy.data`?",
"```python\n"
         "# bpy.ops  → BAĞLAM ister, geri alma (undo) kaydeder\n"
         "# bpy.data → bağlamsız, HIZLI, toplu işlem\n"
         "```\n"
         "⚠ Ölçüm: 10.000 vertex'i döngüde `bpy.ops` ile işlemek "
         "**çok yavaş**; `bpy.data` ile **anlık**.\n\n"
         "Kural: **tek nesne + bağlam** → ops · **toplu/aritmetik** → "
         "veri.",
         "for v in me.vertices: v.co *= 0.5   # hızlı"),

        ("`from_pydata` — mesh'i sıfırdan kurma",
"Operatör kullanmadan mesh üretmenin yolu:\n\n"
         "```python\n"
         "me = bpy.data.meshes.new('M')\n"
         "me.from_pydata(verts, [], faces)   # köşe + yüz listesi\n"
         "me.update()                        # ⚠ ŞART\n"
         "ob = bpy.data.objects.new('O', me)\n"
         "bpy.context.collection.objects.link(ob)   # ⚠ sahneye ekle\n"
         "```\n"
         "⚠ İki tuzak: `update()` unutulursa normaller boş; "
         "`link()` unutulursa nesne görünmez.",
         "me.update()  # normal hesaplansın"),

        ("`bmesh` ne zaman gerekir?",
"`bmesh`, mesh'in **yarı-veri** yapısıdır: kenar/yüz "
         "komşulukları, bevel, inset gibi **topolojik** işlemler.\n\n"
         "```python\n"
         "bm = bmesh.new()\n"
         "bm.from_mesh(me)\n"
         "bmesh.ops.bevel(bm, geom=bm.edges[:], offset=0.01)\n"
         "bm.to_mesh(me); bm.free()   # ⚠ free() şart\n"
         "```\n"
         "⚠ `bmesh` düz `me.vertices`'ten **güçlü** ama karmaşık — "
         "basit işler için gereksiz.",
         "bm.free()  # bellek sızıntısını önler"),

    )),

    ("06_kemik", "Kemik uzayı ve weight mantığı", (

        ("Kemik neden `head`/`tail` ile tanımlanır?",
"Kemik bir **vektördür**: başı (head) ve sonu (tail). "
         "Dönme ekseni **head** etrafındadır.\n\n"
         "```python\n"
         "b = arm.edit_bones.new('omurga')\n"
         "b.head = Vector((0, 0, 0))    # baş\n"
         "b.tail = Vector((0, 0, 1))    # son → +Z yönünde\n"
         "```\n"
         "⚠ Ölçüldü: kemik `head`→`tail` yönü **yerel Y ekseni** "
         "olur (Blender geleneği).",
         "bone.head = Vector((0,0,0)); bone.tail = Vector((0,0,1))"),

        ("`use_connect` ne zaman doğru?",
"```python\n"
         "child.use_connect = True   # head, parent.tail'e KAYNAR\n"
         "child.use_connect = False  # bağımsız konum\n"
         "```\n"
         "⚠ Ölçüldü: omurga zincirinde **True** (kopmaz), bacaklarda "
         "**False** (kalça ile arasında boşluk olabilir).\n\n"
         "`use_connect=True` iken `head` **değiştirilemez**.",
         "b.use_connect = ad in ('omurga','boyun','kafa')"),

        ("Automatic weight nasıl çalışır?",
"`parent_set(type='ARMATURE_AUTO')` = **kemik ısı haritası** "
         "(heat map) algoritması:\n"
         "```\n"
         "1. Her vertex için kemiklere uzaklık hesaplanır\n"
         "2. Isı yayılımı (heat diffusion) ile ağırlık dağıtılır\n"
         "3. Normalize edilir (toplam = 1)\n"
         "```\n"
         "⚠ Ölçüldü: kemikler mesh **içinde** değilse weight "
         "**başarısız** olur (`Bone Heat Weighting: failed`).\n"
         "Çözüm: kemikleri mesh hacminin **içine** yerleştir.",
         "bpy.ops.object.parent_set(type='ARMATURE_AUTO')"),

        ("Ağırlık doğrulaması neden şart?",
"`ARMATURE_AUTO` **sessizce başarısız** olabilir. "
         "Doğrulama:\n\n"
         "```python\n"
         "bagli = sum(1 for v in me.vertices if len(v.groups) > 0)\n"
         "oran = bagli / len(me.vertices)\n"
         "# oran < 0.9 ise weight BOZUK\n"
         "```\n"
         "⚠ Ölçüldü: timsah 1078/1078 = **oran 1.0** — tüm vertex'ler "
         "bağlı. Oran düşükse animasyon **parçalanır**.",
         "oran = bagli / len(me.vertices)"),

    )),

    ("07_uv", "UV ve dikiş mantığı", (

        ("UV neden gerekli?",
"3B yüzeyi 2B dokuya eşlemek için. UV olmadan doku "
         "**görünmez**.\n\n"
         "```python\n"
         "len(me.uv_layers)     # 0 ise doku yok\n"
         "me.uv_layers.active.data[li].uv   # döngü (loop) bazında\n"
         "```\n"
         "⚠ Ölçüldü: UV **döngü** (loop) bazındadır — vertex "
         "bazında **değil**. Bir vertex'in birden çok UV'si olabilir "
         "(dikiş kenarında).",
         "me.uv_layers.active.data[loop_index].uv"),

        ("`smart_project` vs `unwrap` — ne zaman hangisi?",
"```python\n"
         "# smart_project: otomatik, açı tabanlı, HIZLI\n"
         "bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)\n"
         "# unwrap: DİKİŞ (seam) gerektirir, KONTROLLÜ\n"
         "bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)\n"
         "```\n"
         "⚠ Fark: `smart_project` **her kenarı** potansiyel dikiş "
         "sayar → çok ada (island); `unwrap` **senin** dikişlerini "
         "kullanır → az ada, daha iyi.",
         "bpy.ops.uv.smart_project(angle_limit=1.15)"),

        ("Dikiş (seam) nereden geçmeli?",
"Dikiş **görünmeyen** yerlere konur:\n"
         "```\n"
         "· Silindir  → arka dikey kenar\n"
         "· Küp       → bir köşe\n"
         "· Karakter  → iç bacak, kol altı, sırt ortası\n"
         "· Kaya      → önemsiz (dokusu düz)\n"
         "```\n"
         "⚠ Mantık: dikiş **gerilme** (stretch) yaratır; gerilme "
         "görünür yerde olmamalı.",
         "bpy.ops.mesh.mark_seam(clear=False)"),

    )),

    ("08_malzeme", "Malzeme düğüm mantığı", (

        ("Principled BSDF neden varsayılan?",
"Tek düğümde **tüm** PBR kanalları: Base Color, Metallic, "
         "Roughness, Normal, Emission. GLTF aktarımı **yalnız** "
         "bunu tam destekler.\n\n"
         "```python\n"
         "bsdf = mat.node_tree.nodes['Principled BSDF']\n"
         "bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)\n"
         "```\n"
         "⚠ Blender sürümleri **giriş adlarını** değiştirir:\n"
         "`Specular` → `Specular IOR Level` (4.x). Güvenli okuma:\n"
         "`if 'X' in bsdf.inputs:`",
         "if 'Specular IOR Level' in bsdf.inputs: ..."),

        ("Prosedürel doku vs görüntü doku",
"```python\n"
         "# Prosedürel — dosya yok, parametrik, sonsuz çözünürlük\n"
         "tex = nt.nodes.new('ShaderNodeTexNoise')\n"
         "# Görüntü — dosya gerekir, UV gerekir\n"
         "img = nt.nodes.new('ShaderNodeTexImage'); img.image = bpy.data.images.load(p)\n"
         "```\n"
         "⚠ Ölçüldü: bizim kaliteli kaya **prosedürel** (dosya yok, "
         "UV'ye bağımlı değil). Görüntü doku **UV şart** kılar.",
         "tex = nt.nodes.new('ShaderNodeTexNoise')"),

        ("Bump vs Displacement — hangisi ne zaman?",
"```\n"
         "Bump        : sadece NORMAL'i değiştirir → geometri aynı\n"
         "Displacement: geometriyi DEĞİŞTİRİR → poly gerekir\n"
         "```\n"
         "⚠ Ölçüldü: bizim kayada `ShaderNodeBump` kullandık — "
         "vertex sayısı **değişmedi** (965). Displacement "
         "kullansaydık çok daha fazla vertex gerekirdi.",
         "bump = nt.nodes.new('ShaderNodeBump')"),

    )),

    ("09_animasyon", "Animasyon ve fcurve mantığı", (

        ("Keyframe neden `data_path` ile yazılır?",
"Animasyon **veri yolu** (data path) + indeks ile adreslenir:\n\n"
         "```python\n"
         "k.rotation_euler = (0.5, 0, 0)\n"
         "k.keyframe_insert(data_path='rotation_euler', frame=1)\n"
         "```\n"
         "⚠ Pose kemiğinde yol: `pose.bones[\"ad\"].rotation_euler` — "
         "bu yüzden **kemik adı** animasyonun parçasıdır.",
         "k.keyframe_insert(data_path='rotation_euler', frame=1)"),

        ("Blender 5.x'te Action yapısı **değişti**",
"```python\n"
         "# ⚠ Blender 5.x: `Action.fcurves` KALDIRILDI\n"
         "a.fcurves              # AttributeError\n"
         "# ✓ Yeni: layers → strips → channelbags → fcurves\n"
         "for kat in a.layers:\n"
         "    for strip in kat.strips:\n"
         "        for cb in strip.channelbags:\n"
         "            for fc in cb.fcurves: ...\n"
         "```\n"
         "⚠ Ölçüldü (tur 1): bu değişiklik bizi **durdu** — hata "
         "mesajı 'no attribute fcurves' idi.\n"
         "Ders: **sürüm** kontrolü şart, eski örnekler çalışmaz.",
         "for kat in a.layers: ...  # Blender 5.x"),

        ("Neden kemik ADI animasyonu bağlar?",
"```python\n"
         "k = pb.get('on_ayak_l')          # ad ile bul\n"
         "k.rotation_euler = (...)\n"
         "k.keyframe_insert(...)\n"
         "```\n"
         "⚠ Sonuç: kemik adı değişirse animasyon **kırılır**. "
         "Bu yüzden her armature'e **kendi** animasyonu gerekir — "
         "genel animasyon **uygulanamaz**.",
         "k = pb.get('on_ayak_l')  # ad zorunlu"),

        ("Ne kadar kemik animasyonlu olmalı?",
"Timsah ölçümü:\n"
         "```\n"
         "toplam kemik        : 12\n"
         "animasyonlu kemik   :  8  (%67)\n"
         "sabit               : kok, omurga, boyun, cene\n"
         "```\n"
         "⚠ Mantık: **sabit kemik de animasyona dahildir** — "
         "değeri 0 olduğu için keyframe gerekmez. %100 doldurmak "
         "dosyayı şişirir.",
         "# yalnız DEĞİŞEN kemiklere keyframe"),

    )),

    ("10_veri_api", "Veri API'si vs operatör", (

        ("`select_all` veri katmanını DEĞİŞTİRMEZ",
"⚠ Ölçüldü (tur 3): `bpy.ops.mesh.select_all("
         "action=\"DESELECT\")` **veri katmanına yansımıyor**.\n\n"
         "```\n"
         "baslangic_use_seam : 0\n"
         "select_all DESELECT\n"
         "edit_secili        : 72   ← HÂLÂ 72!\n"
         "object_use_seam    : 72   ← mark_seam HEPSİNİ işaretledi\n"
         "```\n"
         "**Sonuç:** `mark_seam` beklenen 25 yerine **72** kenarı "
         "dikiş yaptı.\n\n"
         "✅ Çözüm — operatör yerine **veri API'si**:\n"
         "```python\n"
         "for e in me.edges:\n"
         "    if kosul(e): e.use_seam = True   # mod gerekmez\n"
         "```\n"
         "⚠ Doğrulandı: 48 kapak + 6 dikey = **54** kenar (beklenen).",
         "e.use_seam = True   # veri özelliği"),

        ("Operatör ↔ veri ayrımı: `unwrap` mod ister",
"⚠ Ölçüldü (tur 3): veri API'si **OBJECT** modunda çalışır ama "
         "`uv.unwrap` **EDIT** modu ister.\n\n"
         "```\n"
         "RuntimeError: Operator bpy.ops.mesh.select_all.poll() failed,\n"
         "              context is incorrect\n"
         "```\n"
         "✅ Sıra:\n"
         "```python\n"
         "# 1) OBJECT modunda veri API'siyle dikiş\n"
         "for e in me.edges:\n"
         "    if kosul(e): e.use_seam = True\n"
         "# 2) SONRA EDIT moduna geç\n"
         "bpy.ops.object.mode_set(mode=\"EDIT\")\n"
         "bpy.ops.mesh.select_all(action=\"SELECT\")\n"
         "bpy.ops.uv.unwrap(method=\"ANGLE_BASED\", margin=0.02)\n"
         "```",
         "bpy.ops.object.mode_set(mode=\"EDIT\")  # unwrap için ŞART"),

        ("Kemik eşleştirme: **isim** değil **rol**",
"⚠ Ölçüldü (tur 4): iki yaratıkta aynı işlev farklı adla "
         "olabilir.\n\n"
         "```\n"
         "on_ayak_l  ≡  front_leg_l  ≡  bacak_on_sol\n"
         "kafa       ≡  head\n"
         "kuyruk_1   ≡  tail_1\n"
         "```\n"
         "**Ölçüm (aynı animasyon, 3 farklı armature):**\n"
         "```\n"
         "A (aynı adlar)    : rol=1.00  isim=1.00  → UYGULANIR\n"
         "B (eş anlamlı)    : rol=1.00  isim=0.00  → UYGULANIR ✓\n"
         "D (kuş, bacak yok): rol=0.42  eksik_rol=[on_ayak_l,...]\n"
         "                    → UYARLAMA GEREK\n"
         "```\n"
         "✅ Kriter: **animasyonlu bir rol EKSİK mi?**\n"
         "Eksikse uygulanamaz (kanal hedefsiz kalır). Hiyerarşi farkı "
         "**uyarı**dır, engel değil.",
         "rol = rol_bul(kemik)   # isim değil, İŞLEV"),

    )),

    ("11_fcurve", "Keyframe ve fcurve mekaniği", (

        ("`keyframe_insert` neden GÜVENİLMEZ? (Blender 5.x)",
"⚠ Ölçüldü (tur 4) — **en sinsi tuzak**:\n\n"
         "```python\n"
         "kb.rotation_euler = [0.55, 0.0, 0.0]\n"
         "kb.keyframe_insert(data_path=\"rotation_euler\", frame=1)\n"
         "print(kb.rotation_euler[0])   # 0.55 ✓ (yazıldı!)\n"
         "# AMA fcurve'de: 0.0  ✗\n"
         "```\n\n"
         "**Ölçüm (24 kanal):**\n"
         "```\n"
         "idx2 (kuyruk) → 0.22  ✓ tutuyor\n"
         "idx0 (bacak)  → 0.0   ✗ SIFIRLANDI\n"
         "idx1 (kafa)   → 0.0   ✗ SIFIRLANDI\n"
         "DOLU: 24 kanaldan yalnız 3\n"
         "```\n\n"
         "**Neden?** Blender 5.x **katmanlı Action**'ında "
         "`keyframe_insert` slot/channelbag bağını kendi kurar ve keyfi "
         "`array_index`leri güvenilir yazmaz.\n\n"
         "✅ **Çözüm — fcurve'ü DOĞRUDAN kur:**\n"
         "```python\n"
         "act  = bpy.data.actions.new(ad)\n"
         "kat  = act.layers.new(\"Katman\")\n"
         "st   = kat.strips.new(type=\"KEYFRAME\")\n"
         "slot = act.slots.new(id_type=\"OBJECT\", name=arm.name)\n"
         "cb   = st.channelbag(slot, ensure=True)\n"
         "fc   = cb.fcurves.new(data_path=tam, index=idx)\n"
         "fc.keyframe_points.add(len(noktalar))\n"
         "for i, (kare, deger) in enumerate(noktalar):\n"
         "    kp = fc.keyframe_points[i]\n"
         "    kp.co = (float(kare), float(deger))\n"
         "fc.update()\n"
         "```\n"
         "**Doğrulandı:** dolu kanal 3 → **8** (kaynakla eşit), "
         "kaydedildi ve yeniden açıldı.",
         "fc = cb.fcurves.new(data_path=tam, index=idx)"),

        ("`fcurve.new()` neden kanal BAŞINA bir kez?",
"⚠ Ölçüldü (tur 8) — **en sinsi sessiz hata**:\n\n"
         "```python\n"
         "# ⚠ YANLIŞ: her karede yeni (idx, tek-nokta) çifti\n"
         "def kanal_ekle(kemik, idx, noktalar):\n"
         "    KANALLAR.setdefault(kemik, []).append((idx, noktalar))\n"
         "for kare in range(25):\n"
         "    kanal_ekle(\"ust_bacak_l\", 0, [(kare, deger)])   # 25 kez!\n"
         "```\n\n"
         "Sonra aynı kanal **25 kez** açılmaya çalışılır:\n"
         "```python\n"
         "for idx, noktalar in kanallar:            # 25 tur\n"
         "    fc = cb.fcurves.new(data_path=tam, index=idx)\n"
         "    # 1. → ✓ başarılı (1 nokta, deger=0)\n"
         "    # 2-25. → Blender hata verir (fcurve zaten var)\n"
         "    except Exception:\n"
         "        continue        # ← SESSİZCE YUTULDU\n"
         "```\n\n"
         "**Ölçüm:**\n"
         "```\n"
         "üretim ölçümü : fcurve=350  keyframe=350   ← İYİ GÖRÜNÜYOR\n"
         "diskten       : fcurve= 14  kf=14  DOLU=0  ← HEPSİ SIFIR\n"
         "                kare=[1.0, 1.0]            ← animasyon YOK\n"
         "```\n\n"
         "✅ **Çözüm — kanal başına BİRİKTİR, sonra BİR KEZ kur:**\n"
         "```python\n"
         "#: {kemik: {eksen: [(kare, deger), ...]}}\n"
         "def kanal_ekle(kemik, idx, kare, deger):\n"
         "    KANALLAR.setdefault(kemik, {}).setdefault(idx, []).append(\n"
         "        (kare, deger))\n"
         "\n"
         "for kemik in sorted(KANALLAR.keys()):\n"
         "    for idx in sorted(KANALLAR[kemik].keys()):\n"
         "        noktalar = sorted(KANALLAR[kemik][idx])\n"
         "        fc = cb.fcurves.new(data_path=tam, index=idx)\n"
         "        fc.keyframe_points.add(len(noktalar))\n"
         "        for i, (kare, deger) in enumerate(noktalar):\n"
         "            fc.keyframe_points[i].co = (float(kare), float(deger))\n"
         "        fc.update()\n"
         "```\n\n"
         "⚠ **Ve sessiz `except` YASAK** — hata toplanıp raporlanmalı:\n"
         "```python\n"
         "_fc_hata = []\n"
         "except Exception as exc:\n"
         "    _fc_hata.append(f\"{tam}[{idx}]: {type(exc).__name__}\")\n"
         "```\n"
         "Doğrulandı: DOLU kanal **0 → 14/14**, kare **[1,25]**.",
         "KANALLAR.setdefault(kemik, {}).setdefault(idx, []).append(...)"),

    )),

    ("12_kalicilik", "Kalıcılık — kaydetme ve birikme", (

        ("`Action.use_fake_user` neden şart?",
"⚠ Ölçüldü (tur 9): Blender, **fake user**'ı olmayan ve hiçbir "
         "yere bağlı olmayan action'ları dosyayı kaydederken **siler**.\n\n"
         "```\n"
         "Üretim  : 5 animasyon\n"
         "Diskten : 1 action (bekleme)     ← 4'ü KAYBOLDU\n"
         "```\n\n"
         "✅ Çözüm:\n"
         "```python\n"
         "act = bpy.data.actions.new(ad)\n"
         "act.use_fake_user = True      # ← dosyada KALIR\n"
         "```\n\n"
         "⚠ **Ek tuzak:** betik birden çok kez çalışınca Blender aynı "
         "ada `.001`, `.002` ekler → ölçüm: **20 action** (4 tekrar × 5). "
         "Çözüm — önce eskisini sil:\n"
         "```python\n"
         "for _a in list(bpy.data.actions):\n"
         "    if _a.name == ad or _a.name.startswith(ad + \".\"):\n"
         "        _a.use_fake_user = False\n"
         "        bpy.data.actions.remove(_a, do_unlink=True)\n"
         "```",
         "act.use_fake_user = True   # dosyada kalması için ŞART"),

        ("Düğümler neden BİRİKİR ve nasıl önlenir?",
"⚠ Ölçüldü (tur 11-12): aynı varlığa 4 kez doku uygulanınca "
         "düğüm sayısı **14 → 16 → 18 → 20** oldu.\n\n"
         "**Sebep:** her uygulama **ekliyordu**, değiştirmiyordu.\n\n"
         "✅ **Çözüm — uygulamadan ÖNCE temizle:**\n"
         "```python\n"
         "# ⚠ BSDF ve Output KORUNUR, doku düğümleri silinir\n"
         "_KORU = {\"ShaderNodeBsdfPrincipled\",\n"
         "         \"ShaderNodeOutputMaterial\", \"ShaderNodeMixShader\"}\n"
         "for _n in list(nt.nodes):\n"
         "    if _n.bl_idname in _KORU:\n"
         "        continue\n"
         "    nt.nodes.remove(_n)\n"
         "```\n\n"
         "**Doğrulandı:** 4 kez uygulama → **sabit 4 düğüm** · "
         "farklı dokular → yine **4** (değiştirme çalışıyor).\n\n"
         "⚠ Kural: **idempotent** olmayan üretim, tekrarda bozar.",
         "for _n in list(nt.nodes): nt.nodes.remove(_n)"),

        ("Boş liste **hata değildir**",
"⚠ Ölçüldü (tur 9, 12, 13) — **üç kez** aynı tuzak:\n\n"
         "```python\n"
         "olcum = {\"yazilan\": [...], \"hata\": [], \"kemik\": 12}\n"
         "#                                 ↑ HER ZAMAN var (boş liste)\n"
         "\n"
         "if \"hata\" in d:        # ✗ YANLIŞ — boş liste de True\n"
         "    print(f\"✗ {d['hata'][:70]}\")\n"
         "# → başarılı üretim \"✗ []\" diye raporlandı\n"
         "```\n\n"
         "✅ **Doğru ölçüt: hata DOLU mu?**\n"
         "```python\n"
         "_h = d.get(\"hata\")\n"
         "if isinstance(_h, str) and _h:      # gerçek hata\n"
         "    ...\n"
         "if isinstance(_h, list) and _h:     # birikmiş hatalar\n"
         "    ...\n"
         "```\n\n"
         "⚠ **Ölçüm sözlüğü ile hata alanı aynı yerde olmamalı** — "
         "karışıklığın kaynağı bu.",
         "_h = d.get(\"hata\"); if isinstance(_h, str) and _h: ..."),

    )),

    ("13_dogrulama", "Doğrulama — diskten ölç, görsel bak", (

        ("Neden diskten YENİDEN AÇIP ölçmek şart?",
"⚠ Ölçüldü (tur 6, 8, 9, 12) — **dört kez** aynı desen:\n\n"
         "```\n"
         "tur 6 : üretim 'fcurve=350, kf=350'  → diskte 14 fcurve, DOLU=0\n"
         "        sebep: keyframe_insert keyfi array_index'i yazmıyor\n"
         "tur 8 : üretim 'fcurve=350'          → diskte 14, DOLU=0\n"
         "        sebep: kanal başına 25 fcurve + sessiz except\n"
         "tur 9 : üretim '5 action'            → diskte 1 action\n"
         "        sebep: fake user yok → Blender sildi\n"
         "tur 9 : kütüphane '5 animasyon'      → 1 kayıtlı\n"
         "        sebep: yalnız aktif action okunuyordu\n"
         "```\n\n"
         "**Ortak desen:**\n"
         "```\n"
         "ÜRETİM ÖLÇÜMÜ İYİ GÖRÜNÜYOR  →  DİSK FARKLI SÖYLÜYOR\n"
         "```\n\n"
         "✅ **Kural:** `kaydet → YENİDEN AÇ → ölç`\n"
         "```python\n"
         "bpy.ops.wm.save_as_mainfile(filepath=yol)\n"
         "bpy.ops.wm.open_mainfile(filepath=yol)   # ← YENİDEN AÇ\n"
         "# ŞİMDİ ölç\n"
         "```\n"
         "⚠ Blender betiğinin `print`'ine **asla** güvenme.",
         "bpy.ops.wm.open_mainfile(filepath=yol)  # doğrulama için"),

        ("NLA nedir, ne zaman gerekir?",
"⚠ Ölçüldü (tur 12-13): ayrı action'lar **bağımsız** çalışır → "
         "ani sıçrama.\n\n"
         "```\n"
         "olum bittiğinde : alt_bacak_l = 1.45 rad\n"
         "sonraki anim    : 0.0'dan başlar      ← SIÇRAMA\n"
         "```\n\n"
         "✅ **İki parçalı çözüm:**\n"
         "```python\n"
         "# 1) Geçiş action'ı — smoothstep enterpolasyon (3t² − 2t³)\n"
         "for i in range(gkare + 1):\n"
         "    t = i / gkare\n"
         "    s = t * t * (3.0 - 2.0 * t)     # yumuşak giriş/çıkış\n"
         "    deger = v1 + (v2 - v1) * s\n"
         "\n"
         "# 2) NLA'ya SIRALI yerleştir (üst üste binmesin)\n"
         "iz = arm.animation_data.nla_tracks.new()\n"
         "serit = iz.strips.new(ad, int(sira), a)\n"
         "serit.frame_start = float(sira)\n"
         "serit.frame_end = float(sira) + (ksn - ks)\n"
         "sira = int(serit.frame_end) + 1\n"
         "```\n\n"
         "⚠ **Neden smoothstep?** Doğrusal geçiş başlangıç/bitişte "
         "**ani hız değişimi** yaratır; smoothstep yumuşatır.\n\n"
         "**Ölçüm:** 14 geçiş (7 GERÇEK) · 23 NLA izi.",
         "s = t * t * (3.0 - 2.0 * t)   # smoothstep"),

        ("Görsel doğrulama neden ŞART?",
"⚠ Ölçüldü (tur 14): tur 1-13 boyunca varlıklar yalnız "
         "**sayıyla** doğrulandı:\n"
         "```\n"
         "vertex=2012 · yüz=2270 · kemik=19 · weight=1.0 · UV=1\n"
         "```\n"
         "**Ama sayılar doğru olsa da model bozuk görünebilir:**\n"
         "```\n"
         "· Ters normaller  → model siyah / içi dışı\n"
         "· Yanlış ölçek    → orantısız parçalar\n"
         "· Eksik parça     → sayı doğru, şekil yanlış\n"
         "· Yanlış malzeme  → renk uyuşmazlığı\n"
         "```\n\n"
         "✅ **4 açıdan render:**\n"
         "```python\n"
         "for ad, yatay, yuk in ((\"on\", -90, 0.10), (\"yan\", 0, 0.10),\n"
         "                       (\"uc_dort\", -45, 0.28), (\"ust\", -45, 1.05)):\n"
         "    kamera.location = hedef + yon * mesafe\n"
         "    bpy.ops.render.render(write_still=True)\n"
         "```\n\n"
         "⚠ **Kamera hedefi dünya uzayından** hesaplanmalı — "
         "`ob.location` parent'lı nesnede yanıltıcıdır.\n\n"
         "**Yan bulgu:** önizleme sayesinde `insan_kaliteli` varlığında "
         "`deri` dokusunun uygulandığı **görsel olarak** doğrulandı.",
         "bpy.ops.render.render(write_still=True)"),

        ("Sessiz başarısızlık nasıl yakalanır?",
"Blender bazı hataları **sessizce** yutar:\n"
         "```python\n"
         "# weight başarısız olsa da parent_set hata vermez!\n"
         "bpy.ops.object.parent_set(type='ARMATURE_AUTO')\n"
         "# → DOĞRULAMA şart:\n"
         "bagli = sum(1 for v in me.vertices if len(v.groups) > 0)\n"
         "```\n"
         "⚠ Kural: **her üretimden sonra diske yaz, yeniden aç, "
         "ölç**. `print`'e güvenme.",
         "# üret → kaydet → YENİDEN AÇ → doğrula"),

    )),

    # ══════════════════════════════════════════════════════════════════
    # ⚠ tur 20: ÖLÇÜM METODOLOJİSİ — kod yazmaktan daha kritik
    # ══════════════════════════════════════════════════════════════════
    ("14_olcum", "Ölçüm metodolojisi — doğru ölç, doğru yorumla", (
        ("Yanlış şeyi ölçmek yanlış sonuç verir",
         "⚠ Ölçüldü (tur 18-19): **3 tur** boyunca ölçüm aracının kendisi "
         "yanlıştı.\n\n"
         "```\n"
         "tur 18: `pose_bone.tail` ölçüldü\n"
         "        → rotation_euler 0 → 0.48 → -0.48 DEĞİŞİYOR\n"
         "        → tail SABİT [-0.105, 0.0, 0.46]\n"
         "        → 'animasyon çalışmıyor' sanıldı\n"
         "```\n\n"
         "**Gerçek sebep:** `pose_bone.tail` **REST pozunu** döndürür; "
         "pozisyon değişmez, yalnız `rotation_euler` değişir.\n\n"
         "**İkinci hata (tur 19):** diz kemiği ölçülürken **boyun/kafa** "
         "grubuna bağlı vertex seçildi:\n"
         "```\n"
         "vertex 200 → grupları [['boyun', 0.129], ['kafa', 0.871]]\n"
         "→ diz dönerken SABİT kalması NORMAL!\n"
         "```\n\n"
         "✅ **Kural:** Ölçtüğün şeyin, test ettiğin şeyle **ilişkili** "
         "olduğunu doğrula:\n"
         "```python\n"
         "# ⚠ Doğru: O KEMİĞE bağlı vertex'i seç\n"
         "_vg = {g.index: g.name for g in mesh.vertex_groups}\n"
         "for _v in mesh.data.vertices:\n"
         "    _en = max(_v.groups, key=lambda g: g.weight)\n"
         "    if _en.weight < 0.5:\n"
         "        continue\n"
         "    _ad = _vg.get(_en.group, '')\n"
         "    if _ad and _ad not in _secili:\n"
         "        _secili[_ad] = _v.index\n"
         "```",
         "_en = max(_v.groups, key=lambda g: g.weight)"),
        ("Örnekleme döngü noktalarına düşebilir",
         "⚠ Ölçüldü (tur 18): `yurume` 24 kare · `sin(2πt)` döngüsü.\n\n"
         "```python\n"
         "# ⚠ YANLIŞ: 3 kare örnekleme\n"
         "adim = (ksn - ks) / 2\n"
         "# yurume: kare 1, 13, 25\n"
         "# sin(2π·0) = 0 · sin(2π·0.5) = 0 · sin(2π·1) = 0\n"
         "# → HEPSİ SIFIR → sahte 'HAREKET_YOK'\n"
         "```\n\n"
         "✅ **Çözüm: yeterli örnek + yarım adım kaydırma**\n"
         "```python\n"
         "_n = max(7, KARE_SAYISI)      # ⚠ en az 7\n"
         "_kaydir = 0.5                 # ⚠ döngü tepelerini yakala\n"
         "for i in range(_n):\n"
         "    t = (i + _kaydir) / _n\n"
         "    kareler.append(_KS + int(t * _toplam_kare))\n"
         "```\n\n"
         "⚠ **Genel kural:** Periyodik sinyali örneklerken örnek sayısı "
         "**periyodun tam böleni** olmamalı.",
         "_kaydir = 0.5   # döngü tepelerini yakalar"),
        ("Ölçüm aracının önbelleği yanıltır",
         "⚠ Ölçüldü (tur 18): her render çağrısı **yeni** PNG ekliyordu "
         "→ klasörde karışık kareler:\n"
         "```\n"
         "1. çağrı : kare_001, kare_007, kare_013, kare_019, kare_025\n"
         "2. çağrı : kare_001, kare_004, ..., kare_030      (7 kare)\n"
         "→ karşılaştırma YANLIŞ (9 kare karışık)\n"
         "```\n\n"
         "✅ **Çözüm: çıktı klasörünü TEMİZLE**\n"
         "```python\n"
         "if os.path.isdir(CIKTI):\n"
         "    for _f in os.listdir(CIKTI):\n"
         "        if _f.endswith('.png'):\n"
         "            os.remove(os.path.join(CIKTI, _f))\n"
         "```\n\n"
         "⚠ Kural: **üretim çıktısı birikimli ise, ölçüm de birikimli "
         "olur** → yanlış.",
         "os.remove(os.path.join(CIKTI, _f))  # önce temizle"),
        ("`pose_bone` rest pozunda kalır — manuel yaz",
         "⚠ Ölçüldü (tur 18-19): Blender'ın **action değerlendirmesi "
         "render'a yansımıyor**.\n\n"
         "```\n"
         "PNG dosyaları : MD5 FARKLI   ✓\n"
         "Piksel verisi : 0/589824     ✗  ← İÇERİK AYNI\n"
         "```\n\n"
         "**Ama deformasyon ÇALIŞIYOR** — yalnız manuel yazımla:\n"
         "```\n"
         "kare  1: diz vertex = [-0.105, 0.00000, 0.38800]\n"
         "kare 15: diz vertex = [-0.105, 0.05493, 0.41345]   ← 55 mm ✓\n"
         "kare 30: diz vertex = [-0.105, 0.07146, 0.45123]   ← 71 mm ✓\n"
         "```\n\n"
         "✅ **Çözüm: fcurve değerlerini DOĞRUDAN pose bone'a yaz**\n"
         "```python\n"
         "def _poz_yaz(kare):\n"
         "    for _yol, _eksenler in _fc_gruplar.items():\n"
         "        if 'pose.bones[' not in _yol:\n"
         "            continue\n"
         "        _kemik = _yol.split('pose.bones[')[1].split(']')[0]\n"
         "        _pb = pb.get(_kemik)\n"
         "        _ozellik = _yol.split('.')[-1]\n"
         "        # ⚠ rotation_euler için mod uyumu ŞART\n"
         "        if _ozellik == 'rotation_euler' and \\\n"
         "                _pb.rotation_mode == 'QUATERNION':\n"
         "            _pb.rotation_mode = 'XYZ'\n"
         "        _mevcut = list(getattr(_pb, _ozellik))\n"
         "        for _e, _f in _eksenler.items():\n"
         "            _mevcut[_e] = float(_f.evaluate(kare))\n"
         "        setattr(_pb, _ozellik, _mevcut)\n"
         "```\n"
         "**Sonuç:** piksel farkı **%0 → %6,56**.\n\n"
         "⚠ **Ders:** 'Çalışmıyor' ≠ 'Yanlış yoldan çağırıyorum'. "
         "Alternatif yol denemeden **sınır ilan etmemeli**.",
         "setattr(_pb, _ozellik, _mevcut)  # manuel poz"),
        ("Tip karışıklığı geç bulunur",
         "⚠ Ölçüldü (tur 19): aynı sözlükte iki farklı tip:\n"
         "```python\n"
         "hareket = {}\n"
         "hareket['diz'] = [0.0, 0.48, 0.0]        # radyan (fcurve)\n"
         "hareket['el']  = [(0.1, 0.2, 0.3), ...]  # 3B nokta (metre)\n"
         "→ TypeError: '>' not supported between instances of\n"
         "             'tuple' and 'float'\n"
         "```\n\n"
         "✅ **Çözüm: AYRI sözlükler + açık tip belgesi**\n"
         "```python\n"
         "hareket = {}   # kemik → [fcurve değerleri] (RADYAN)\n"
         "deform = {}    # kemik → [(x, y, z), ...]   (METRE)\n"
         "```\n\n"
         "⚠ Ve **karar hangi ölçüte dayanıyor** yazılı olmalı:\n"
         "```python\n"
         "#: ⚠ Karar **deformasyona** dayanır (varsa); yoksa fcurve'ye\n"
         "_en_buyuk = _en_deform if _en_deform > 0.0 else _en_fcurve\n"
         "```",
         "hareket = {}  # radyan · deform = {}  # metre"),
        ("Her girdi varyantını test et",
         "⚠ Ölçüldü (tur 21-22): `--anim` bayrağı **2 tur** boyunca "
         "\"çalışmıyor\" göründü. Gerçek sebep: **dosya seçimi**.\n\n"
         "```\n"
         "_blend_bul(\"insan_orta_orta\")\n"
         "  → insan_kaliteli_kaliteli/varlik.blend    ← YANLIŞ\n"
         "_blend_bul(\"insan\")\n"
         "  → insan_kaliteli_kaliteli/varlik.blend    ← DOĞRU (tesadüf)\n"
         "```\n\n"
         "**Neden yanlış teşhis kondu:**\n"
         "```\n"
         "· `_blend_bul(\"insan\")` test edildi → DOĞRU döndü\n"
         "· ⚠ AMA `_blend_bul(\"insan_orta_orta\")` test EDİLMEDİ\n"
         "· Doğrudan betik testi çalışıyordu (dosya yolu elle veriliyordu)\n"
         "```\n\n"
         "✅ **Kural: her girdi varyantını test et.**\n"
         "```python\n"
         "# ⚠ YETERSİZ: tek girdi\n"
         "assert _blend_bul(\"insan\") is not None\n"
         "\n"
         "# ✅ YETERLİ: tüm varyantlar + AYRIŞMA kontrolü\n"
         "adlar = [\"insan_orta_orta\", \"insan_lowpoly_lowpoly\",\n"
         "         \"insan_kaliteli_kaliteli\"]\n"
         "yollar = {a: str(_blend_bul(a)) for a in adlar}\n"
         "assert len(set(yollar.values())) == len(yollar), (\n"
         "    f\"aynı dosya birden çok ad için: {yollar}\")\n"
         "```\n\n"
         "⚠ **Özellikle:** aynı öneki paylaşan adlar (`insan_*`) "
         "**ayrışmalı**.",
         "assert len(set(yollar.values())) == len(yollar)  # ayrışma"),
        ("Sayaç ile karar aynı kaynaktan olmalı",
         "⚠ Ölçüldü (tur 23): çıktı **çelişkiliydi**:\n"
         "```\n"
         "✓ blok  NORMAL  en_buyuk=0.82398  kemik=0/19\n"
         "        ↑ karar DOĞRU              ↑ sayaç YANLIŞ\n"
         "\n"
         "en deforme :\n"
         "  el_l  0.82398     ← liste DOLU!\n"
         "  el_r  0.82015\n"
         "```\n\n"
         "**Kök neden:** karar `deform` sözlüğüne dayanıyordu ama sayaç "
         "`hareket` sözlüğünden okunuyordu:\n"
         "```python\n"
         "olcum[\"hareketli_kemik\"] = sum(\n"
         "    1 for v in olcum[\"hareket\"].values() if v > 0.001)\n"
         "                  ↑ fcurve (radyan) — çoğu kemik SABİT\n"
         "                  ↑ ama karar `deform` (metre) sözlüğünden!\n"
         "```\n\n"
         "✅ **Çözüm:** sayaç, kararın dayandığı **aynı** sözlükten:\n"
         "```python\n"
         "#: ⚠ Karar **deformasyona** dayanıyor\n"
         "_hareketli = sum(1 for v in _deform_s.values() if v > 0.002)\n"
         "olcum[\"hareketli_kemik\"] = _hareketli\n"
         "```\n\n"
         "**Sonuç:** `kemik=0/19` → **`kemik=15/19`**.\n\n"
         "⚠ **Genel kural:** Bir ölçüm sonucu üretiyorsan, "
         "**türetilmiş tüm sayaçları** o ölçümün **aynı kaynağından** "
         "hesapla. Aksi halde \"çalışıyor ama 0\" gibi **çelişkili** "
         "çıktı verir.",
         "sum(1 for v in _deform_s.values() if v > 0.002)  # aynı kaynak"),
        ("Üretim aracı **yolu bildirmeli**",
         "⚠ Ölçüldü (tur 25): `asset.uret` başarı bildirdi ama zincir "
         "koptu.\n\n"
         "```\n"
         "asset.uret(urun=\"timsah\")\n"
         "  → basarili  True\n"
         "  → vertex    1078      ✓\n"
         "  → kemik     12        ✓\n"
         "  → klasor    (BOŞ)     ✗\n"
         "\n"
         "asset.animasyon    → ✗ \"armature yok\"\n"
         "asset.gecis        → ✗ \"armature yok\"\n"
         "asset.onizleme     → ✗ \"varlık bulunamadı: \"\n"
         "```\n\n"
         "**⚠ Ve dosya diske YAZILIYORDU** (ölçüldü: 3 → 4 timsah). "
         "Yani üretim çalışıyordu, **yol bildirimi** kopuktu.\n\n"
         "**Kök neden:** dataclass'ın `to_dict()` çıktısında `klasor` "
         "**yok**:\n"
         "```python\n"
         "sd = _s.to_dict()\n"
         "# → {\"ad\", \"basarili\", \"olcum\", \"sure_s\", \"hata\",\n"
         "#    \"blend\", \"glb\"}\n"
         "#     ↑ \"blend\" ÜST düzeyde, `olcum` içinde DEĞİL\n"
         "d = dict(sd.get(\"olcum\") or {})   # ⚠ yalnız olcum\n"
         "d.setdefault(\"klasor\", \"\")      # ⚠ SESSİZCE BOŞ ekler\n"
         "```\n\n"
         "✅ **Çözüm — yolu hesapla:**\n"
         "```python\n"
         "# ⚠ `klasor` = `blend` dosyasının ÜST DİZİNİ\n"
         "_blend_yolu = str(_sd.get(\"blend\") or \"\")\n"
         "if _blend_yolu:\n"
         "    d[\"klasor\"] = str(Path(_blend_yolu).parent)\n"
         "    d[\"blend\"] = _blend_yolu\n"
         "else:\n"
         "    d[\"klasor\"] = \"\"\n"
         "```\n\n"
         "⚠ **Ve `setdefault(\"x\", \"\")` tehlikelidir** — hata "
         "vermez, sessizce boş koyar. Değeri **hesapla** ya da "
         "**hata ver**.\n\n"
         "⚠ **Genel kural:** Bir üretim aracı, sonucu **nerede** "
         "bıraktığını **bildirmeli**. Bildirmezse zincir kopar.",
         "d[\"klasor\"] = str(Path(_blend_yolu).parent)  # yolu BİLDİR"),
        ("Beş araçta aynı hata → ortak yardımcı",
         "⚠ Ölçüldü (tur 25): **5 araçta aynı** yol hatası vardı.\n\n"
         "```\n"
         "anim_uret · anim_gecis · varlik_onizleme · doku_uygula ·\n"
         "anim_onizleme\n"
         "\n"
         "hepsi: Path(\"Varliklar\") / hedef / \"varlik.blend\"\n"
         "\n"
         "hedef = \"Varliklar\\yaratik\\timsah_0007\"\n"
         "→ Varliklar\\Varliklar\\yaratik\\...   ✗ ÇİFT\n"
         "```\n\n"
         "⚠ **Ve ID ile arama yapanlar bulamıyordu:**\n"
         "```\n"
         "asset.gecis    → \"varlık bulunamadı: Varliklar\\yaratik\\timsah_0007\"\n"
         "asset.onizleme → aynı\n"
         "```\n\n"
         "✅ **Çözüm — TEK doğru kaynak:**\n"
         "```python\n"
         "# tools/varlik_yolu.py — 6 aşamalı çözümleme\n"
         "1. Tam `.blend` yolu\n"
         "2. Tam klasör yolu (Varliklar içeriyor)\n"
         "3. Göreli `kategori/ad`\n"
         "4. Yalnız ad → kategori klasörleri\n"
         "5. Katalog → **puanlı** eşleştirme (tur 22 dersi)\n"
         "6. Derin arama (son çare)\n"
         "```\n\n"
         "```python\n"
         "def _ortak_blend_bul(hedef: str):\n"
         "    from tools.varlik_yolu import blend_bul\n"
         "    return blend_bul(hedef)\n"
         "```\n\n"
         "⚠ **Kural:** Aynı hata **birden çok** dosyada görülüyorsa, "
         "her birini ayrı düzeltmek yerine **ortak yardımcı** yaz.\n\n"
         "**Ve her girdi varyantını test et** (tur 22 dersi):\n"
         "```\n"
         "✓ timsah_0007 · timsah_0001 · insan · insan_orta_orta\n"
         "✓ Varliklar/yaratik/timsah_0007 · yaratik/timsah_0007\n"
         "✓ Varliklar/yaratik/timsah_0007/varlik.blend\n"
         "```",
         "_o = _ortak_blend_bul(hedef)  # 5 araç tek kaynak"),
        ("Bayrağı değil **sonucu** doğrula",
         "⚠ Ölçüldü (tur 25-26): uçtan uca test `basarili` bayrağına "
         "baktı ve **yanlış** sonuç verdi.\n\n"
         "```\n"
         "✓ doku=- düğüm=0        ← basarili=True ama DOKU YOK\n"
         "```\n\n"
         "**Kök neden:** test `a.doku(hedef, tur=\"deri\", uygula=True)` "
         "çağırdı. Ama araç **üç modlu**:\n"
         "```python\n"
         "def doku(sorgu=\"\", hedef=\"\", doku=\"\", dosya=\"\", ...):\n"
         "    #   ↑ İLK parametre `sorgu` (arama terimi), hedef DEĞİL\n"
         "```\n"
         "→ `hedef` **arama terimi** olarak yorumlandı, doku "
         "uygulanmadı.\n\n"
         "✅ **Çözüm — mod ayrımını doğru kullan:**\n"
         "```python\n"
         "# 1. ARA + SUN\n"
         "r = a.doku(sorgu=\"deri\")\n"
         "\n"
         "# 2. PROSEDÜREL UYGULA\n"
         "r = a.doku(hedef=\"Varliklar/yaratik/timsah_0007\", doku=\"deri\")\n"
         "# → {\"dugum\": 4, \"baglanti\": 3, \"uv\": 1}   ✓ GERÇEK SONUÇ\n"
         "\n"
         "# 3. GÖRÜNTÜ UYGULA\n"
         "r = a.doku(hedef=..., dosya=\"doku.png\", kanal=\"renk\")\n"
         "```\n\n"
         "✅ **Ve testi şöyle yaz:**\n"
         "```python\n"
         "# ⚠ YETERSİZ: bayrağa bak\n"
         "ok = bool(r.get(\"basarili\"))\n"
         "\n"
         "# ✅ YETERLİ: ASIL çıktıya bak\n"
         "ok = (bool(r.get(\"basarili\"))\n"
         "      and r.get(\"dugum\", 0) >= 3\n"
         "      and r.get(\"baglanti\", 0) >= 2)\n"
         "```\n\n"
         "⚠ **Genel kural:** `basarili` bayrağı **yetmez** — aracın "
         "**asıl çıktısı** (düğüm, kemik, dosya, ölçüm) kontrol "
         "edilmeli.\n\n"
         "⚠ **Ve test yazarken araç çıktısı ÖNCE yazdırılmalı** — "
         "varsayılan anahtar adı kullanmak **sahte hata** üretir "
         "(tur 25: `animasyon=0` ← aslında 13).",
         "ok = r.get(\"dugum\", 0) >= 3 and r.get(\"baglanti\", 0) >= 2"),
        ("Ne zaman 'sınır' ilan edilir?",
         "⚠ Ölçüldü (tur 18-19): tur 18'de 'Blender render'ı "
         "güncellemiyor' diye **sınır ilan edildi** — yanlıştı.\n\n"
         "**İki hata birlikteydi:**\n"
         "```\n"
         "1. Yanlış vertex ölçülüyordu (boyun/kafa, diz değil)\n"
         "2. Manuel poz yazımı denenmemişti\n"
         "```\n\n"
         "✅ **Sınır ilan etmeden önce kontrol listesi:**\n"
         "```\n"
         "□ Ölçtüğüm şey, test ettiğim şeyle ilişkili mi?\n"
         "□ Örnekleme yeterli mi (periyot böleni değil)?\n"
         "□ Önbellek/birikim karışıklık yaratıyor mu?\n"
         "□ ALTERNATİF YOL denendi mi?\n"
         "□ Hata mesajı gerçekten ne diyor?\n"
         "□ İki hata üst üste binmiş olabilir mi?\n"
         "```\n\n"
         "⚠ **Kural:** Sınır, **birden çok bağımsız yol** denendikten "
         "sonra ilan edilir.\n\n"
         "**Doğrulanmış sınır örneği (tur 9):** `keyframe_insert` "
         "keyfi `array_index`'i yazmıyor — 4 kez ölçüldü, 2 alternatif "
         "denendi (slot bağlama, farklı api), sonra sınır kabul edildi.",
         "# Sınır ilan etmeden önce alternatif yol DENE"),
    )),

    ("15_surum", "Sürüm farkları ve hata ayıklama", (

        ("API sürümler arası neden değişir?",
"Blender 2.7x → 2.8x → 3.x → 4.x → 5.x: her sürümde "
         "**yeniden adlandırma**.\n\n"
         "```python\n"
         "# 2.7x          → 2.8x+\n"
         "'Spatial'       → 'Node3D'        (Godot gibi!)\n"
         "'Blender Game'  → kaldırıldı\n"
         "# 3.x           → 4.x\n"
         "'Specular'      → 'Specular IOR Level'\n"
         "# 4.x           → 5.x\n"
         "Action.fcurves   → layers/channelbags\n"
         "```\n"
         "⚠ Ders: **`bpy.app.version`** kontrol et, örnek koda "
         "körü körüne güvenme.",
         "v = bpy.app.version   # (5, 2, 2)"),

        ("Hata mesajından kök neden çıkarma",
"```\n"
         "'context is incorrect'\n"
         "  → etkin nesne/mod yanlış → active + select + mode_set\n"
         "'Cannot open file'\n"
         "  → klasör YOK → mkdir(parents=True)\n"
         "'Bone Heat Weighting: failed'\n"
         "  → kemik mesh İÇİNDE değil\n"
         "'Attempting to unscale FP16 gradients'\n"
         "  → (torch) hassasiyet uyumsuz\n"
         "'no attribute fcurves'\n"
         "  → Blender 5.x API değişikliği\n"
         "```\n"
         "⚠ Ölçüldü (tur 1): 4 hatanın **hepsi** bu yolla çözüldü.",
         "# hata mesajını OKU, tahmin etme"),

    )),

)


@dataclass
class MantikDersi:
    """Tek mantık dersi."""
    katman: str = ""
    baslik: str = ""
    mantik: str = ""
    kod: str = ""

    def metin(self) -> str:
        return (f"# Blender mantığı — {self.baslik}\n\n"
                f"**Katman:** {self.katman}\n\n"
                f"## Neden?\n\n{self.mantik}\n\n"
                f"## Kod\n\n```python\n{self.kod}\n```\n")


def tum_dersler() -> List[MantikDersi]:
    """Tüm dersleri üretir."""
    out: List[MantikDersi] = []
    for kat_ad, kat_aciklama, dersler in MÜFREDAT:
        for baslik, mantik, kod in dersler:
            out.append(MantikDersi(katman=f"{kat_ad} — {kat_aciklama}",
                                   baslik=baslik, mantik=mantik, kod=kod))
    return out


def istatistik() -> Dict[str, object]:
    """Müfredat istatistiği."""
    d = tum_dersler()
    mt = [x.metin() for x in d]
    return {
        "katman": len(MÜFREDAT),
        "ders": len(d),
        "karakter": sum(len(m) for m in mt),
        "token_tahmini": sum(len(m) for m in mt) // 3,
        "katman_dagilim": {k[0]: len(k[2]) for k in MÜFREDAT},
    }


def kasaya_yaz() -> int:
    """Dersleri kasaya yazar."""
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return 0
    gorulen: set = set()
    try:
        for n in k.vault.iter_notes(limit=300000):
            if (getattr(n, "category", "") or "") == "blender_mantik":
                gorulen.add((getattr(n, "title", "") or "").strip())
    except Exception:
        pass
    yazilan = 0
    for d in tum_dersler():
        baslik = f"Blender mantığı: {d.baslik}"
        if baslik in gorulen:
            continue
        try:
            from neurocore.knowledge.vault import Note

            simdi = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            n = Note(
                id="", title=baslik, body=d.metin(), kind="skill",
                category="blender_mantik",
                tags=["blender", "mantik", d.katman.split("_")[0]],
                confidence=0.95, importance=0.9,
                created=simdi, updated=simdi, version=1,
                sources=["blender_mantik"], links=[], backlinks=[],
                embedding_model="", embedding=None, path="", extra={})
            k.vault.write_note(n, touch=False, emit=False)
            yazilan += 1
        except Exception:
            continue
    k.shutdown()
    return yazilan


def main() -> int:
    """Müfredatı çalıştırır."""
    ap = argparse.ArgumentParser(description="Blender mantık müfredatı")
    ap.add_argument("--istatistik", action="store_true")
    ap.add_argument("--kasaya", action="store_true")
    ap.add_argument("--liste", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  BLENDER MANTIK MÜFREDATI — 'neden' öğretimi")
    print("=" * 78)

    if args.liste:
        for d in tum_dersler():
            print(f"  [{d.katman.split(' — ')[0]}] {d.baslik}")
        return 0

    st = istatistik()
    print(f"\n  katman    : {st['katman']}")
    print(f"  ders      : {st['ders']}")
    print(f"  karakter  : {st['karakter']:,}")
    print(f"  token     : ~{st['token_tahmini']:,}")
    print(f"\n  KATMAN DAĞILIMI:")
    for k, v in st["katman_dagilim"].items():
        print(f"    {k:14} {v} ders")

    if args.kasaya:
        n = kasaya_yaz()
        print(f"\n  kasaya yazıldı: {n} not")

    Path("_blender_mantik.json").write_text(
        json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
