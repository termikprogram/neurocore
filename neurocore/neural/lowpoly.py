"""
Prosedürel 3B İçerik Müfredatı (Lowpoly)
=========================================
Yerel modelin **lowpoly varlık üretimi** ve **arazi (terrain) sistemleri**
öğrenmesi için katmanlı içerik.

Neden ayrı modül?
    Ölçüm gösterdi: yerel model bu alanda **şans seviyesinde** kaldı —
    anlamlı bir cevap, saçma bir cevaba karşı kaybetti (`-8.05` vs `-8.00`).
    Model "lowpoly" kavramını, ağaç/çiçek yapısını, noise tabanlı arazi
    üretimini, material katmanlarını ve su/yol işleme kurallarını
    **bilmiyor**. Bu modül o boşluğu kapatır.

    Alan özellikle uygun: kural tabanlı, ölçülebilir ve **doğrulanabilir**.
    "Yüz sayısı az olmalı" gibi kararlar tartışmasızdır — öğretmesi kolay,
    öğrenilmesi test edilebilir.

Katmanlar:

    lowpoly_temel   — Lowpoly felsefesi, yüz bütçesi, silüet, stilizasyon
    lowpoly_bitki   — Ağaç ve çiçek modelleme (gövde, dal, yaprak, taç)
    lowpoly_arazi   — Heightmap, noise, erozyon, grid mesh, LOD
    lowpoly_doku    — Material katmanları, ağırlık haritaları, blend
    lowpoly_detay   — Su, yol, dağ, biome ve iyileştirmeler

⚠ İçerik **gerçek Blender API'sine** ve **yerleşik prosedürel tekniklere**
dayanır: Perlin/Simplex noise, hidrolik erozyon, ağırlık haritası, vb.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from neurocore.neural.curriculum import Lesson

#: Prosedürel içerik katmanları (sıra öğretim sırasıdır)
LOWPOLY_LAYERS: Sequence[str] = (
    "lowpoly_temel", "lowpoly_bitki", "lowpoly_arazi",
    "lowpoly_doku", "lowpoly_detay",
)

#: Katman etiketleri
LOWPOLY_LABELS: Dict[str, str] = {
    "lowpoly_temel": "Lowpoly Temelleri",
    "lowpoly_bitki": "Ağaç ve Çiçek Modelleme",
    "lowpoly_arazi": "Arazi (Terrain) Üretimi",
    "lowpoly_doku": "Material Katmanları (Texture)",
    "lowpoly_detay": "Su, Yol ve İyileştirmeler",
}


# ─── Katman 1: Lowpoly Temelleri ────────────────────────────────────────────

def _temel_lessons() -> List[Lesson]:
    """Lowpoly felsefesi ve yüz bütçesi."""
    return [
        Lesson("lowpoly_temel", "Lowpoly nedir, neden tercih edilir?",
               "**Lowpoly**, düşük poligon sayısıyla (genelde 100–2000 yüz) "
               "stilize edilmiş 3B modelleme yaklaşımıdır. Tercih nedenleri: "
               "**performans** (GPU/CPU yükü azalır, mobilde kritik), "
               "**dosya boyutu** (ağ üzerinden hızlı iner), "
               "**sanatsal stil** (keskin ve temiz görünüm), "
               "**üretim hızı** (az detay = hızlı modelleme). "
               "Önemli ilke: **silüet** her şeydir — model uzaktan "
               "tanınmıyorsa poligon sayısı önemsizdir.",
               ["lowpoly", "performans", "silüet"]),
        Lesson("lowpoly_temel", "Yüz bütçesi nasıl belirlenir?",
               "Yüz bütçesi **kullanım amacına** göre belirlenir. Tipik "
               "değerler: mobil oyunda **karakter 1.500–5.000 yüz**, "
               "**props (eşya) 100–800 yüz**, **ağaç 300–1.500 yüz**, "
               "**çiçek/çalı 20–150 yüz**. Kural: ekranda kaç nesne "
               "görünecek × yüz sayısı = toplam bütçe. 60 FPS mobil hedefi "
               "için kare başına **~100.000–300.000 yüz** sınırı vardır. "
               "Ağaç ormanda 500 kez görünüyorsa 1.000 yüz → 500.000 yüz "
               "eder; bu yüzden ağaç **300 yüzü geçmemeli**.",
               ["yüz bütçesi", "mobil", "optimizasyon"]),
        Lesson("lowpoly_temel", "Silüet ve okunabilirlik",
               "İyi lowpoly model **uzaktan tanınır**. Teknikler: "
               "**abartılı oranlar** (büyük taç, kalın gövde), "
               "**net kırılmalar** (yumuşak geçiş yerine köşe), "
               "**az sayıda büyük şekil** (10 küçük yaprak yerine 3 büyük "
               "kütle). Test yöntemi: modeli **tek renge** boyayıp "
               "silüete bakın — ne olduğu anlaşılıyorsa tasarım doğrudur. "
               "Anlaşılmıyorsa detay eklemek değil, **şekli sadeleştirmek** "
               "gerekir.",
               ["silüet", "okunabilirlik", "tasarım"]),
        Lesson("lowpoly_temel", "Düşük poligonlu şekiller nasıl üretilir?",
               "Temel yaklaşım: **az kenarlı primitifler** kullan. "
               "Silindir 6–8 kenar (ağaç gövdesi), küre 8×6 dilim "
               "(taç kütlesi), koni 6–8 kenar (yaprak kümesi), küp "
               "(kutu/gövde). **Alt bölme (subdivision) KULLANMA** — "
               "lowpoly'nin amacına aykırıdır. Yüz azaltmak için "
               "**Decimate** modifier (`ratio` 0.3–0.5), düzeltmek için "
               "**Limited Dissolve** (`angle_limit` 5–15°). "
               "**Shade Flat** kullan; smooth shading kenarları yumuşatıp "
               "lowpoly görünümünü bozar.",
               ["silindir", "decimate", "shade flat"], code=(
                   "import bpy\n"
                   "\n"
                   "# 6 kenarli silindir = agac govdesi\n"
                   "bpy.ops.mesh.primitive_cylinder_add(\n"
                   "    vertices=6, radius=0.15, depth=2.0, location=(0, 0, 1))\n"
                   "govde = bpy.context.object\n"
                   "\n"
                   "# 6 kenarli koni = yaprak kumesi\n"
                   "bpy.ops.mesh.primitive_cone_add(\n"
                   "    vertices=6, radius1=0.8, depth=1.6, location=(0, 0, 2.6))\n"
                   "tac = bpy.context.object\n"
                   "\n"
                   "# Lowpoly gorunumu icin duz golgeleme (onemli!)\n"
                   "for ob in (govde, tac):\n"
                   "    for p in ob.data.polygons:\n"
                   "        p.use_smooth = False\n"
                   "\n"
                   "print('govde yuz:', len(govde.data.polygons),\n"
                   "      'tac yuz:', len(tac.data.polygons))")),
        Lesson("lowpoly_temel", "Poligon sayısı nasıl ölçülür ve azaltılır?",
               "Ölçüm: `len(obj.data.polygons)` yüz sayısı, "
               "`len(obj.data.vertices)` köşe sayısı. Azaltma yolları: "
               "**Decimate** modifier (collapse modu, `ratio`), "
               "**Limited Dissolve** (düz yüzeyleri birleştirir), "
               "**Remesh** (voxel/blocks, stilize görünüm için), "
               "manuel temizlik. ⚠ **Üçgen vs dörtgen**: oyun motorları "
               "üçgen sayar; dörtgen 2 üçgen demektir. "
               "`triangles = sum(len(p.vertices) - 2 for p in polygons)`.",
               ["decimate", "limited dissolve", "üçgen"], code=(
                   "import bpy\n"
                   "\n"
                   "ob = bpy.context.object\n"
                   "yuz = len(ob.data.polygons)\n"
                   "kose = len(ob.data.vertices)\n"
                   "# Oyun motoru ucgen sayar: n kenarli yuz = (n-2) ucgen\n"
                   "ucgen = sum(len(p.vertices) - 2 for p in ob.data.polygons)\n"
                   "print(f'yuz={yuz} kose={kose} ucgen={ucgen}')\n"
                   "\n"
                   "# Decimate ile azalt (yuzde 50)\n"
                   "mod = ob.modifiers.new('Dec', 'DECIMATE')\n"
                   "mod.ratio = 0.5\n"
                   "bpy.ops.object.modifier_apply(modifier=mod.name)\n"
                   "print('sonra:', sum(len(p.vertices) - 2\n"
                   "                    for p in ob.data.polygons), 'ucgen')")),
        Lesson("lowpoly_temel", "Malzeme sayısı ve doku atlası",
               "Her **malzeme slotu ayrı draw call** demektir; mobilde "
               "maliyetlidir. Kural: **bir nesne = bir malzeme**. Birden "
               "fazla renk gerekiyorsa: **vertex color** (köşe rengi) "
               "kullan — ekstra doku ve draw call gerekmez, "
               "`Principled BSDF` → `Color Attribute` düğümü ile bağlanır. "
               "Alternatif: **doku atlası** (tek resimde tüm renkler, "
               "UV ile seçilir). Lowpoly'de vertex color en pratik yoldur.",
               ["vertex color", "draw call", "atlas"], code=(
                   "import bpy\n"
                   "\n"
                   "ob = bpy.context.object\n"
                   "# Kose rengi katmani ekle\n"
                   "if not ob.data.color_attributes:\n"
                   "    ob.data.color_attributes.new(\n"
                   "        name='Renk', type='BYTE_COLOR', domain='CORNER')\n"
                   "katman = ob.data.color_attributes['Renk']\n"
                   "for i in range(len(katman.data)):\n"
                   "    katman.data[i].color = (0.25, 0.5, 0.2, 1.0)  # yesil\n"
                   "\n"
                   "# Malzemeye bagla\n"
                   "mat = ob.data.materials[0] if ob.data.materials else \\\n"
                   "    ob.data.materials.append(bpy.data.materials.new('M'))\n"
                   "mat = ob.data.materials[0]\n"
                   "mat.use_nodes = True\n"
                   "dugumler = mat.node_tree.nodes\n"
                   "bsdf = dugumler.get('Principled BSDF')\n"
                   "renk = dugumler.new('ShaderNodeVertexColor')\n"
                   "renk.layer_name = 'Renk'\n"
                   "mat.node_tree.links.new(renk.outputs['Color'],\n"
                   "                        bsdf.inputs['Base Color'])")),
        Lesson("lowpoly_temel", "Prosedürel üretim neden tercih edilir?",
               "Elle modellemek yerine **kodla üretmek** şu avantajları "
               "verir: **varyasyon** (aynı koddan 1000 farklı ağaç), "
               "**tutarlılık** (hepsi aynı stil), "
               "**ölçek** (elle 1000 ağaç modellenemez), "
               "**parametre kontrolü** (boy, renk, yoğunluk ayarlanabilir), "
               "**tekrar üretilebilirlik** (tohum ile aynı sonuç). "
               "Blender'da iki yol: **Python betiği** (tam kontrol) veya "
               "**Geometry Nodes** (düğüm tabanlı, görsel).",
               ["prosedürel", "varyasyon", "geometry nodes"]),
        Lesson("lowpoly_temel", "Tohum (seed) ile tekrarlanabilir üretim",
               "Prosedürel üretimde **rastgelelik kontrollü** olmalıdır. "
               "`random.seed(42)` aynı sonucu üretir; bu hata ayıklama ve "
               "tutarlılık için kritiktir. Her nesneye **farklı ama sabit** "
               "tohum** vermek (ör. `seed = hash(obje_adi)`) hem çeşitlilik "
               "hem tekrarlanabilirlik sağlar. **Sahne üretiminde** bu "
               "olmadan her çalıştırmada farklı dünya çıkar ve hata "
               "ayıklanamaz.",
               ["seed", "rastgelelik", "tekrarlanabilir"], code=(
                   "import random\n"
                   "\n"
                   "# Ayni tohum -> ayni sonuc (tekrarlanabilir)\n"
                   "def agac_parametreleri(tohum: int) -> dict:\n"
                   "    r = random.Random(tohum)\n"
                   "    return {\n"
                   "        'boy': r.uniform(1.5, 4.0),\n"
                   "        'govde_yaricap': r.uniform(0.08, 0.2),\n"
                   "        'tac_yaricap': r.uniform(0.6, 1.4),\n"
                   "        'yaprak_rengi': (r.uniform(0.1, 0.3),\n"
                   "                         r.uniform(0.4, 0.7),\n"
                   "                         r.uniform(0.1, 0.3), 1.0),\n"
                   "    }\n"
                   "\n"
                   "print(agac_parametreleri(42))\n"
                   "print(agac_parametreleri(42))   # ayni cikti\n"
                   "print(agac_parametreleri(43))   # farkli agac")),
    ]


# ─── Katman 2: Ağaç ve Çiçek ────────────────────────────────────────────────

def _bitki_lessons() -> List[Lesson]:
    """Ağaç, çiçek ve bitki modelleme."""
    return [
        Lesson("lowpoly_bitki", "Lowpoly ağaç yapısı: gövde, dal, taç",
               "Ağaç üç parçadan kurulur: **gövde** (6–8 kenarlı silindir, "
               "tabana doğru kalınlaşır), **dal** (gövdeden açıyla çıkan "
               "daha ince silindirler, 2–4 adet), **taç** (yaprak kütlesi; "
               "2–3 koni veya 1 az dilimli küre). Yüz bütçesi dağılımı: "
               "gövde ~50, dallar ~100, taç ~200 → **toplam 350 yüz**. "
               "İpucu: taç için **koni kullanmak** küreden daha az yüz "
               "harcar ve daha stilize görünür.",
               ["gövde", "dal", "taç"], code=(
                   "import bpy, math\n"
                   "\n"
                   "def lowpoly_agac(boy=3.0, tohum=1):\n"
                   "    # Govde: 6 kenarli, tabana dogru kalin\n"
                   "    bpy.ops.mesh.primitive_cylinder_add(\n"
                   "        vertices=6, radius=boy * 0.05, depth=boy * 0.6,\n"
                   "        location=(0, 0, boy * 0.3))\n"
                   "    govde = bpy.context.object\n"
                   "    govde.name = 'Govde'\n"
                   "    # Tabani kalinlastir (lowpoly agac karakteri)\n"
                   "    for v in govde.data.vertices:\n"
                   "        if v.co.z < 0:\n"
                   "            v.co.x *= 1.8\n"
                   "            v.co.y *= 1.8\n"
                   "    # Tac: 2 koni (az yuz, stilize)\n"
                   "    koniler = []\n"
                   "    for i, (z, r) in enumerate(((0.75, 0.42), (1.0, 0.30))):\n"
                   "        bpy.ops.mesh.primitive_cone_add(\n"
                   "            vertices=6, radius1=boy * r, depth=boy * 0.45,\n"
                   "            location=(0, 0, boy * z))\n"
                   "        koniler.append(bpy.context.object)\n"
                   "    return govde, koniler\n"
                   "\n"
                   "g, k = lowpoly_agac(3.0)\n"
                   "print('govde yuz:', len(g.data.polygons),\n"
                   "      'tac yuz:', sum(len(x.data.polygons) for x in k))")),
        Lesson("lowpoly_bitki", "Farklı ağaç türleri nasıl çeşitlenir?",
               "Çeşitlilik **parametrelerle** üretilir, yeni model "
               "yazmadan: **iğne yapraklı** (dar uzun koni, koyu yeşil, "
               "dikey), **geniş yapraklı** (geniş küre/koni, açık yeşil), "
               "**palmiye** (uzun ince gövde, tepede yelpaze), "
               "**çalı** (gövdesiz, 3 küçük küre), **kuru ağaç** "
               "(dallar açık, taç yok). Biyom-özel: kurak bölgede küçük "
               "ve seyrek, ormanda uzun ve sık. Aynı kodu farklı "
               "parametrelerle çağırmak **kod tekrarını önler**.",
               ["iğne yapraklı", "palmiye", "biyom"]),
        Lesson("lowpoly_bitki", "Lowpoly çiçek nasıl modellenir?",
               "Çiçek **çok küçük** olduğu için yüz bütçesi 20–60 "
               "arasıdır. Yapı: **sap** (3 kenarlı silindir veya tek "
               "dörtgen şerit), **taç yapraklar** (4–8 adet düz "
               "dörtgen/üçgen, hafif açılı), **orta** (küçük küre veya "
               "daire). Alternatif teknik: **billboard** (her yöne dönen "
               "tek dörtgen) — en ucuz yöntem, uzaktan fark edilmez. "
               "Çiçekler genelde **kümeler** halinde serpilir, tek tek "
               "değil.",
               ["çiçek", "taç yaprak", "billboard"], code=(
                   "import bpy, math\n"
                   "\n"
                   "def lowpoly_cicek(yaprak=6, boy=0.3):\n"
                   "    # Sap: 3 kenarli ince silindir\n"
                   "    bpy.ops.mesh.primitive_cylinder_add(\n"
                   "        vertices=3, radius=boy * 0.04, depth=boy * 0.7,\n"
                   "        location=(0, 0, boy * 0.35))\n"
                   "    sap = bpy.context.object\n"
                   "    # Tac yapraklar: duz dortgenler, hafif acili\n"
                   "    for i in range(yaprak):\n"
                   "        aci = (2 * math.pi / yaprak) * i\n"
                   "        bpy.ops.mesh.primitive_plane_add(\n"
                   "            size=boy * 0.35,\n"
                   "            location=(math.cos(aci) * boy * 0.18,\n"
                   "                      math.sin(aci) * boy * 0.18,\n"
                   "                      boy * 0.75))\n"
                   "        p = bpy.context.object\n"
                   "        p.rotation_euler = (math.radians(-25), 0, aci)\n"
                   "    return sap\n"
                   "\n"
                   "lowpoly_cicek(6, 0.3)\n"
                   "print('cicek olusturuldu')")),
        Lesson("lowpoly_bitki", "Yaprak kütlesi: küre mi koni mi?",
               "Karar **istenen stile** göre verilir. **Koni** (6–8 kenar): "
               "en az yüz (12–16), keskin silüet, iğne yapraklı ve stilize "
               "ağaçlar için ideal. **İkosahedron** (`subdivisions=1`, "
               "20 yüz): organik görünüm, geniş yapraklı için iyi, "
               "**en verimli küre benzeri**. **UV küre** "
               "(`segments=8, ring_count=6`): 96 yüz, yumuşak ama pahalı — "
               "lowpoly'de genelde **gereksiz**. **Küp** (12 üçgen): "
               "en ucuz, blok stil için. Sıralama: küp < ikosahedron < "
               "koni < UV küre.",
               ["ikosahedron", "koni", "yüz karşılaştırma"], code=(
                   "import bpy\n"
                   "\n"
                   "# Ayni boyutta farkli tac sekilleri - yuz sayilarini karsilastir\n"
                   "bpy.ops.mesh.primitive_cube_add(size=1)\n"
                   "kup = sum(len(p.vertices) - 2\n"
                   "          for p in bpy.context.object.data.polygons)\n"
                   "bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.5)\n"
                   "iko = sum(len(p.vertices) - 2\n"
                   "          for p in bpy.context.object.data.polygons)\n"
                   "bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=0.5, depth=1)\n"
                   "koni = sum(len(p.vertices) - 2\n"
                   "           for p in bpy.context.object.data.polygons)\n"
                   "bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=6, radius=0.5)\n"
                   "kure = sum(len(p.vertices) - 2\n"
                   "           for p in bpy.context.object.data.polygons)\n"
                   "print(f'kup={kup} ikosahedron={iko} koni={koni} uv_kure={kure}')")),
        Lesson("lowpoly_bitki", "Ağaçları sahneye dağıtma (scatter)",
               "Tek ağaç yeterli değil; **orman** gerekir. Yöntem: "
               "**ızgara + gürültü** (grid noktalarına rastgele kaydırma), "
               "**Poisson disk** (birbirine çok yakın olmasın — en doğal), "
               "**yoğunluk haritası** (ormanın ortasında sık, kenarda "
               "seyrek). Performans için **MultiMesh/instancing** "
               "kullanılır: aynı mesh yüzlerce kez çizilir, sahne ağacında "
               "tek düğüm olur. Blender'da `Geometry Nodes` → "
               "`Distribute Points on Faces` + `Instance on Points`.",
               ["scatter", "poisson", "instancing"], code=(
                   "import bpy, math, random\n"
                   "\n"
                   "def orman_serp(adet=50, alan=20.0, min_mesafe=1.5, tohum=7):\n"
                   "    \"\"\"Poisson-benzeri: birbirine cok yaklasmayan dagitim.\"\"\"\n"
                   "    r = random.Random(tohum)\n"
                   "    noktalar = []\n"
                   "    deneme = 0\n"
                   "    while len(noktalar) < adet and deneme < adet * 40:\n"
                   "        deneme += 1\n"
                   "        x = r.uniform(-alan / 2, alan / 2)\n"
                   "        y = r.uniform(-alan / 2, alan / 2)\n"
                   "        if all((x - a) ** 2 + (y - b) ** 2 >= min_mesafe ** 2\n"
                   "               for a, b in noktalar):\n"
                   "            noktalar.append((x, y))\n"
                   "    return noktalar\n"
                   "\n"
                   "n = orman_serp(50, 20.0, 1.5, 7)\n"
                   "print(f'{len(n)} agac noktasi uretildi')\n"
                   "print('ilk 3:', [(round(a,1), round(b,1)) for a, b in n[:3]])")),
        Lesson("lowpoly_bitki", "Mevsim ve renk varyasyonu",
               "Aynı ağaç modeli farklı **renklerle** farklı mevsim verir: "
               "**ilkbahar** (açık yeşil, pembe çiçek), **yaz** (koyu "
               "yeşil), **sonbahar** (turuncu/kırmızı/sarı), **kış** "
               "(taç yok, kahverengi dallar). Renk **vertex color** ile "
               "değiştirilir — model yeniden üretilmez. Bu, tek mesh + "
               "4 varyant demektir; bellek ve üretim maliyeti düşer. "
               "Renk geçişi için `ColorRamp` düğümü kullanılabilir.",
               ["mevsim", "vertex color", "varyant"]),
        Lesson("lowpoly_bitki", "Çim ve yer örtüsü nasıl yapılır?",
               "Çim **tek tek modellenmez** — milyonlarca gerekir. "
               "Yöntemler: **doku** (en ucuz; zemine çim dokusu), "
               "**billboard kartlar** (birkaç bin dörtgen, kameraya döner), "
               "**kümeler** (5–10 bıçak birleşik tek mesh, instancing ile "
               "binlerce), **shader tabanlı** (Geometri Shader ile GPU'da "
               "üretim, en performanslı ama karmaşık). Lowpoly oyunda "
               "genelde **doku + az sayıda küme** kullanılır.",
               ["çim", "billboard", "instancing"]),
    ]


# ─── Katman 3: Arazi ────────────────────────────────────────────────────────

def _arazi_lessons() -> List[Lesson]:
    """Heightmap, noise, erozyon, grid mesh."""
    return [
        Lesson("lowpoly_arazi", "Arazi (terrain) nedir, nasıl temsil edilir?",
               "**Arazi**, oyun dünyasının yüzeyidir. Temsil yolları: "
               "**heightmap** (gri tonlu resim; parlaklık = yükseklik — "
               "en yaygın), **üçgen ağ (mesh)** (her köşe elle/otomatik "
               "konumlu), **voxel** (küpler; Minecraft tarzı), "
               "**prosedürel** (her sorguda matematiksel fonksiyon). "
               "Heightmap avantajı: **tek resim = tüm dünya**, küçük "
               "dosya, kolay düzenleme. 16-bit PNG kullanılır (8-bit "
               "basamaklı görünür).",
               ["heightmap", "voxel", "temsil"]),
        Lesson("lowpoly_arazi", "Perlin ve Simplex noise nedir?",
               "**Noise**, rastgele ama **komşu değerleri uyumlu** üreten "
               "fonksiyondur (rastgele sayıdan farkı: yumuşak geçiş). "
               "**Perlin** (1983) ve **Simplex** (2001, daha hızlı ve "
               "yön bağımsız) en yaygınlarıdır. **Fraktal (fBm)**: birden "
               "fazla frekansı toplama — `yükseklik = Σ(genlik_i × "
               "noise(frekans_i × x))`, her adımda `genlik /= 2`, "
               "`frekans *= 2`. Bu, **doğal görünüm** verir: büyük "
               "tepeler + küçük pürüzler. Tek frekans = yapay görünüm.",
               ["perlin", "simplex", "fbm"], code=(
                   "import math, random\n"
                   "\n"
                   "def noise1d(x: float, tohum: int = 0) -> float:\n"
                   "    \"\"\"Basit deger gurultusu (Perlin'in sadelesmis hali).\"\"\"\n"
                   "    i = math.floor(x)\n"
                   "    f = x - i\n"
                   "    # Yumusak gecis (smoothstep)\n"
                   "    f = f * f * (3 - 2 * f)\n"
                   "    r = random.Random(tohum)\n"
                   "    def nokta(k):\n"
                   "        r.seed(tohum * 7919 + k)\n"
                   "        return r.uniform(-1, 1)\n"
                   "    return nokta(i) * (1 - f) + nokta(i + 1) * f\n"
                   "\n"
                   "def fbm(x: float, oktav: int = 4, tohum: int = 0) -> float:\n"
                   "    \"\"\"Fraktal: birden fazla frekansi topla.\"\"\"\n"
                   "    toplam, genlik, frekans = 0.0, 1.0, 1.0\n"
                   "    for o in range(oktav):\n"
                   "        toplam += genlik * noise1d(x * frekans, tohum + o)\n"
                   "        genlik *= 0.5\n"
                   "        frekans *= 2.0\n"
                   "    return toplam\n"
                   "\n"
                   "print([round(fbm(x / 10.0, 4), 3) for x in range(6)])")),
        Lesson("lowpoly_arazi", "Grid mesh'ten arazi üretimi (Blender)",
               "Adımlar: **1)** grid oluştur (`primitive_grid_add`, "
               "`x_subdivisions` = çözünürlük), **2)** her köşenin Z "
               "değerini heightmap/noise'tan ata, **3)** normalleri "
               "yeniden hesapla, **4)** gölgeleme ayarla. Çözünürlük "
               "dengesi: 128×128 grid = 32.768 yüz (kabul edilebilir), "
               "512×512 = 524.288 yüz (ağırdır → **LOD** gerekir). "
               "Lowpoly arazi için **düşük çözünürlük + shade flat** "
               "stilize görünüm verir.",
               ["grid", "çözünürlük", "lod"], code=(
                   "import bpy, math, random\n"
                   "\n"
                   "def basit_noise(x, y, tohum=0):\n"
                   "    r = random.Random(int(x * 1000) * 73856093 ^\n"
                   "                       int(y * 1000) * 19349663 ^ tohum)\n"
                   "    return r.uniform(-1, 1)\n"
                   "\n"
                   "def arazi_uret(boyut=20.0, bolum=48, yukseklik=3.0, tohum=5):\n"
                   "    bpy.ops.mesh.primitive_grid_add(\n"
                   "        x_subdivisions=bolum, y_subdivisions=bolum, size=boyut)\n"
                   "    arazi = bpy.context.object\n"
                   "    arazi.name = 'Arazi'\n"
                   "    olcek = bolum / boyut\n"
                   "    for v in arazi.data.vertices:\n"
                   "        # Fraktal benzeri: iki frekans toplami\n"
                   "        d = (basit_noise(v.co.x * olcek, v.co.y * olcek, tohum)\n"
                   "             + 0.5 * basit_noise(v.co.x * olcek * 2,\n"
                   "                                 v.co.y * olcek * 2, tohum + 1))\n"
                   "        v.co.z = d * yukseklik\n"
                   "    # Lowpoly: duz golgeleme + normalleri yenile\n"
                   "    for p in arazi.data.polygons:\n"
                   "        p.use_smooth = False\n"
                   "    return arazi\n"
                   "\n"
                   "a = arazi_uret(20.0, 48, 3.0, 5)\n"
                   "print('arazi yuz:', len(a.data.polygons))")),
        Lesson("lowpoly_arazi", "Erozyon: araziyi doğal göstermek",
               "Saf noise **yapay** görünür (her yerde aynı pürüz). "
               "**Erozyon** doğal şekiller ekler: **hidrolik** (su akışı "
               "vadi açar, tortuyu ovanın altına taşır), **termal** "
               "(dik yamaçlar yığılır, eğim sınırı). Basit hidrolik "
               "algoritma: her noktaya yağmur düşer → en dik komşuya akar "
               "→ yüksekliği aşındırır → taşıdığı tortuyu eğim azalınca "
               "bırakır. Sonuç: **vadiler, nehir yatakları, ovalar** — "
               "yani gerçek arazi görünümü.",
               ["erozyon", "hidrolik", "vadi"], code=(
                   "import bpy\n"
                   "\n"
                   "def hidrolik_erozyon(arazi, tur=8, yagmur=0.02, asinma=0.3):\n"
                   "    \"\"\"Basit hidrolik erozyon: su asagi akar, toprak tasir.\"\"\"\n"
                   "    verts = arazi.data.vertices\n"
                   "    # Komsuluk: grid oldugunu varsayarak indeks ile bul\n"
                   "    n = int(len(verts) ** 0.5)\n"
                   "    yukseklik = [v.co.z for v in verts]\n"
                   "    for _ in range(tur):\n"
                   "        yeni = list(yukseklik)\n"
                   "        for i in range(len(verts)):\n"
                   "            satir, sutun = divmod(i, n)\n"
                   "            komsular = []\n"
                   "            for dr, ds in ((-1,0),(1,0),(0,-1),(0,1)):\n"
                   "                r2, s2 = satir + dr, sutun + ds\n"
                   "                if 0 <= r2 < n and 0 <= s2 < n:\n"
                   "                    komsular.append(r2 * n + s2)\n"
                   "            if not komsular:\n"
                   "                continue\n"
                   "            # En alcak komsuya toprak tasi\n"
                   "            en_alcak = min(komsular, key=lambda k: yukseklik[k])\n"
                   "            fark = yukseklik[i] - yukseklik[en_alcak]\n"
                   "            if fark > 0:\n"
                   "                tasi = min(fark * 0.5, yagmur) * asinma\n"
                   "                yeni[i] -= tasi\n"
                   "                yeni[en_alcak] += tasi\n"
                   "        yukseklik = yeni\n"
                   "    for v, z in zip(verts, yukseklik):\n"
                   "        v.co.z = z\n"
                   "    return arazi")),
        Lesson("lowpoly_arazi", "Yükseklik ölçekleme ve dağ sınırı",
               "Ham noise **-1..1** arasındadır; ölçeklenmezse arazi "
               "neredeyse düzdür. Ölçekleme: `z = noise × yukseklik_olcegi`. "
               "Dağ oluşturmak için **üs alma** (power) kullanılır: "
               "`z = (noise + 1) / 2` (0..1), sonra `z ** 2.5` → düşük "
               "alanlar düz kalır, yüksek alanlar **dağ** olur. Ters "
               "yönde `z ** 0.5` yayla/plato verir. Bu basit dönüşüm "
               "**biyom karakterini** belirler: düz ova, tepelik, "
               "dağlık, kayalık.",
               ["yükseklik ölçeği", "üs alma", "biyom"], code=(
                   "def profil_uygula(yukseklik01: float, us: float) -> float:\n"
                   "    \"\"\"0..1 araligindaki yuksekligi profille sekillendir.\"\"\"\n"
                   "    return yukseklik01 ** us\n"
                   "\n"
                   "# Ayni ham deger, farkli profiller\n"
                   "ham = 0.55\n"
                   "for ad, us in (('ova (duz)', 0.5), ('tepelik', 1.0),\n"
                   "               ('daglik', 2.5), ('kayalik', 4.0)):\n"
                   "    print(f'{ad:12} us={us} -> {profil_uygula(ham, us):.3f}')\n"
                   "\n"
                   "# Daglik profilde alcak alanlar duz kalir:\n"
                   "for h in (0.1, 0.3, 0.5, 0.7, 0.9):\n"
                   "    print(f'  ham={h} -> daglik={profil_uygula(h, 2.5):.3f}')")),
        Lesson("lowpoly_arazi", "LOD: uzak araziyi basitleştirme",
               "**LOD** (Level of Detail): kameraya yakın bölge yüksek "
               "çözünürlük, uzak bölge düşük. Böylece **görsel kalite "
               "korunur, maliyet düşer**. Uygulama: arazi **parçalara "
               "(chunk)** bölünür (ör. 64×64 birim), her parça kamera "
               "mesafesine göre 4 seviyeden biriyle çizilir. "
               "**Yarık (crack) sorunu**: farklı çözünürlükteki komşu "
               "parçaların kenarları uyuşmaz → kenar dikişi (stitching) "
               "veya aynı kenar çözünürlüğü gerekir.",
               ["lod", "chunk", "crack"]),
        Lesson("lowpoly_arazi", "Biome: araziye karakter verme",
               "**Biyom**, iklim ve yüksekliğe göre bitki/toprak "
               "topluluğudur. Basit kural tablosu: **deniz** (0–0.2), "
               "**kumsal** (0.2–0.25), **ova/çayır** (0.25–0.45), "
               "**orman** (0.45–0.65), **tepe/kaya** (0.65–0.8), "
               "**kar** (0.8–1.0). Bir başka eksen **nem**: kurak ↔ "
               "tropikal. İki eksen (yükseklik × nem) **biyom matrisi** "
               "verir. Her biyomun kendi **material katmanı** ve "
               "**bitki seti** olur.",
               ["biome", "yükseklik", "nem"], code=(
                   "# Yukseklik ve neme gore biyom secimi\n"
                   "BIYOMLAR = [\n"
                   "    (0.00, 0.20, 'deniz'),\n"
                   "    (0.20, 0.25, 'kumsal'),\n"
                   "    (0.25, 0.45, 'cayir'),\n"
                   "    (0.45, 0.65, 'orman'),\n"
                   "    (0.65, 0.80, 'kayalik'),\n"
                   "    (0.80, 1.01, 'kar'),\n"
                   "]\n"
                   "\n"
                   "def biyom_sec(yukseklik01: float, nem01: float) -> str:\n"
                   "    for alt, ust, ad in BIYOMLAR:\n"
                   "        if alt <= yukseklik01 < ust:\n"
                   "            if ad == 'cayir' and nem01 > 0.7:\n"
                   "                return 'bataklik'\n"
                   "            if ad == 'cayir' and nem01 < 0.3:\n"
                   "                return 'bozkir'\n"
                   "            return ad\n"
                   "    return 'bilinmiyor'\n"
                   "\n"
                   "for h, n in ((0.1, 0.5), (0.35, 0.8), (0.35, 0.1), (0.9, 0.5)):\n"
                   "    print(f'h={h} nem={n} -> {biyom_sec(h, n)}')")),
    ]


# ─── Katman 4: Material Katmanları ──────────────────────────────────────────

def _doku_lessons() -> List[Lesson]:
    """Ağırlık haritaları ve katman karıştırma."""
    return [
        Lesson("lowpoly_doku", "Terrain material katmanları nedir?",
               "Tek bir doku tüm araziye yetmez (kum, çimen, kaya, kar "
               "farklı). **Katmanlama**: her yüzeye bir doku, üst üste "
               "**ağırlık haritası** ile karıştırılır. Ağırlık haritası "
               "**0–1** arası değer taşır: 0 = bu katman yok, 1 = tam "
               "bu katman. Tipik arazi shader'ı **4 katman** destekler "
               "(tek RGBA ağırlık dokusu). Daha fazlası **ikinci ağırlık "
               "dokusu** veya **doku dizisi (texture array)** gerektirir.",
               ["katman", "ağırlık haritası", "shader"]),
        Lesson("lowpoly_doku", "Yükseklik tabanlı katman geçişi",
               "En basit ve en yaygın kural: **yüksekliğe göre katman**. "
               "Kum deniz seviyesinde, çimen ortada, kaya yüksekte, kar "
               "tepelerde. Geçiş **yumuşak** olmalıdır (keskin sınır "
               "yapay görünür) — `smoothstep(alt, üst, yükseklik)` "
               "fonksiyonu kullanılır. `smoothstep`, 0 ile 1 arasında "
               "S eğrisi üretir: sınırın iki yanında yumuşak karışım.",
               ["yükseklik", "smoothstep", "geçiş"], code=(
                   "def smoothstep(alt: float, ust: float, x: float) -> float:\n"
                   "    \"\"\"S egrisi: alt'ta 0, ust'te 1, arasi yumusak.\"\"\"\n"
                   "    if ust == alt:\n"
                   "        return 0.0 if x < alt else 1.0\n"
                   "    t = max(0.0, min(1.0, (x - alt) / (ust - alt)))\n"
                   "    return t * t * (3 - 2 * t)\n"
                   "\n"
                   "# Yukseklige gore kum -> cimen -> kaya -> kar agirliklari\n"
                   "def katman_agirliklari(h: float) -> dict:\n"
                   "    kum = 1.0 - smoothstep(0.20, 0.28, h)\n"
                   "    kar = smoothstep(0.78, 0.86, h)\n"
                   "    kaya = smoothstep(0.62, 0.72, h) * (1.0 - kar)\n"
                   "    cimen = max(0.0, 1.0 - kum - kaya - kar)\n"
                   "    return {'kum': round(kum, 3), 'cimen': round(cimen, 3),\n"
                   "            'kaya': round(kaya, 3), 'kar': round(kar, 3)}\n"
                   "\n"
                   "for h in (0.1, 0.3, 0.5, 0.7, 0.9):\n"
                   "    print(f'h={h}: {katman_agirliklari(h)}')")),
        Lesson("lowpoly_doku", "Eğim tabanlı katman geçişi",
               "Yükseklik tek başına yetmez: **dik yamaçta** çimen "
               "tutmaz, kaya olur. **Eğim (slope)** = yüzey normalinin "
               "dikeyle açısı. `normal.z` 1 ise düz, 0 ise dikey duvar. "
               "Kural: `kaya_agirligi = 1 - smoothstep(0.6, 0.85, "
               "normal.z)` → dik yerlerde kaya. Gerçekçi görünüm için "
               "**yükseklik VE eğim birlikte** kullanılır: düz yüksek "
               "yer karlı, dik yüksek yer kayalı.",
               ["eğim", "normal", "kaya"], code=(
                   "def egim_agirligi(normal_z: float) -> float:\n"
                   "    \"\"\"Dik yamac -> kaya. normal_z: 1 duz, 0 dikey.\"\"\"\n"
                   "    if normal_z >= 0.85:\n"
                   "        return 0.0\n"
                   "    if normal_z <= 0.60:\n"
                   "        return 1.0\n"
                   "    t = (0.85 - normal_z) / (0.85 - 0.60)\n"
                   "    return t * t * (3 - 2 * t)\n"
                   "\n"
                   "# Birlestirilmis kural\n"
                   "def gercekci_katman(h: float, normal_z: float) -> dict:\n"
                   "    kaya_egim = egim_agirligi(normal_z)\n"
                   "    kar = max(0.0, (h - 0.78) / 0.22) * (1.0 - kaya_egim)\n"
                   "    kaya = max(kaya_egim, (1.0 - kar) * max(0.0, (h - 0.6) / 0.3))\n"
                   "    cimen = max(0.0, 1.0 - kar - kaya)\n"
                   "    return {'cimen': round(cimen, 3), 'kaya': round(kaya, 3),\n"
                   "            'kar': round(kar, 3)}\n"
                   "\n"
                   "print('duz tepe  :', gercekci_katman(0.85, 0.98))\n"
                   "print('dik tepe  :', gercekci_katman(0.85, 0.55))\n"
                   "print('duz ova   :', gercekci_katman(0.30, 0.99))")),
        Lesson("lowpoly_doku", "Katman karıştırma matematiği",
               "Ağırlıklar **toplamı 1** olmalıdır, aksi hâlde renk "
               "kayar. Normalizasyon: `w_i = a_i / Σa`. Karışım: "
               "`renk = Σ(w_i × doku_i)`. Shader'da bu **Mix** düğümleri "
               "zinciri veya **lerp** ile yapılır. Sıra önemlidir: "
               "genelde **taban → ara → üst** şeklinde üst üste "
               "bindirilir (`lerp(taban, ara, w1)` sonra "
               "`lerp(sonuç, üst, w2)`). Doku tekrarı (tiling) için "
               "UV ölçeği ayarlanır.",
               ["normalizasyon", "lerp", "tiling"], code=(
                   "def normalize(agirliklar: dict) -> dict:\n"
                   "    \"\"\"Agirliklarin toplami 1 olsun (renk kaymasin).\"\"\"\n"
                   "    toplam = sum(agirliklar.values())\n"
                   "    if toplam <= 0:\n"
                   "        return {k: 0.0 for k in agirliklar}\n"
                   "    return {k: v / toplam for k, v in agirliklar.items()}\n"
                   "\n"
                   "def karistir(renkler: dict, agirliklar: dict) -> tuple:\n"
                   "    \"\"\"Agirlikli renk ortalamasi (shader'daki Mix karsiligi).\"\"\"\n"
                   "    w = normalize(agirliklar)\n"
                   "    r = sum(renkler[k][0] * w[k] for k in renkler)\n"
                   "    g = sum(renkler[k][1] * w[k] for k in renkler)\n"
                   "    b = sum(renkler[k][2] * w[k] for k in renkler)\n"
                   "    return (round(r, 3), round(g, 3), round(b, 3))\n"
                   "\n"
                   "renkler = {'kum': (0.76, 0.70, 0.50), 'cimen': (0.30, 0.55, 0.22),\n"
                   "           'kaya': (0.45, 0.43, 0.40), 'kar': (0.95, 0.96, 0.98)}\n"
                   "print(karistir(renkler, {'kum': 0.2, 'cimen': 1.0, 'kaya': 0.3, 'kar': 0.0}))")),
        Lesson("lowpoly_doku", "Doku bozulmasını önleme (tiling)",
               "Aynı doku geniş alana yayılınca **tekrar deseni** "
               "görünür (belli olur). Çözümler: **doku harmanlama** "
               "(iki ölçekte aynı dokuyu karıştır), **detay dokusu** "
               "(ikinci ince doku), **rastgele döndürme** (her chunk "
               "farklı açı), **makro varyasyon** (geniş ölçekli renk "
               "değişimi), **triplanar eşleme** (dik yüzeylerde UV "
               "kaymasını önler — üç eksende izdüşüm).",
               ["tiling", "triplanar", "detay"]),
        Lesson("lowpoly_doku", "Vertex color ile ucuz katmanlama",
               "Doku ve shader maliyeti istemiyorsan **köşe rengi** "
               "kullan: her köşeye renk ata, GPU arada **interpolasyon** "
               "yapsın. Avantaj: **sıfır doku belleği**, sıfır ek draw "
               "call, mobil için ideal. Dezavantaj: **düşük çözünürlük** "
               "(köşe başına bir renk; detay için çok köşe gerekir). "
               "Lowpoly stilde bu genelde **yeterlidir** ve hatta "
               "istenen görünümdür.",
               ["vertex color", "interpolasyon", "mobil"]),
    ]


# ─── Katman 5: Su, Yol ve İyileştirmeler ────────────────────────────────────

def _detay_lessons() -> List[Lesson]:
    """Su, yol, dağ ve arazi iyileştirmeleri."""
    return [
        Lesson("lowpoly_detay", "Su nasıl eklenir?",
               "Su eklemenin üç katmanı vardır: **1) Yatak** — araziyi su "
               "seviyesinin altına indir (`z < su_seviyesi`), vadi/nehir "
               "oluştur. **2) Yüzey** — su seviyesinde **düz bir düzlem** "
               "(plane) koy; araziyi keser, doğal göl/deniz oluşur. "
               "**3) Kıyı** — su kenarında **nemli toprak/kum** katmanı ve "
               "hafif eğim. Su yüzeyi için saydam mavi malzeme "
               "(`Transmission` veya `Alpha`), hafif dalga için vertex "
               "kaydırma (`Sine` dalgası) kullanılır.",
               ["su", "yatak", "kıyı"], code=(
                   "import bpy, math\n"
                   "\n"
                   "def su_ekle(arazi, su_seviyesi=0.2, boyut=20.0):\n"
                   "    # 1) Araziyi su seviyesinin altina indir (gol yatagi)\n"
                   "    for v in arazi.data.vertices:\n"
                   "        if v.co.z < su_seviyesi:\n"
                   "            # Yatagi biraz daha derinlestir\n"
                   "            v.co.z = su_seviyesi - (su_seviyesi - v.co.z) * 1.4\n"
                   "    # 2) Su yuzeyi: duz duzlem\n"
                   "    bpy.ops.mesh.primitive_plane_add(size=boyut * 1.2,\n"
                   "                                     location=(0, 0, su_seviyesi))\n"
                   "    su = bpy.context.object\n"
                   "    su.name = 'Su'\n"
                   "    # 3) Saydam mavi malzeme\n"
                   "    mat = bpy.data.materials.new('M_Su')\n"
                   "    mat.use_nodes = True\n"
                   "    bsdf = mat.node_tree.nodes.get('Principled BSDF')\n"
                   "    bsdf.inputs['Base Color'].default_value = (0.1, 0.3, 0.55, 1)\n"
                   "    if 'Alpha' in bsdf.inputs:\n"
                   "        bsdf.inputs['Alpha'].default_value = 0.75\n"
                   "    mat.blend_method = 'BLEND'\n"
                   "    su.data.materials.append(mat)\n"
                   "    return su\n"
                   "\n"
                   "print('su fonksiyonu hazir')")),
        Lesson("lowpoly_detay", "Nehir ve göl nasıl oluşur?",
               "**Göl**: kapalı çukur (yerel minimum). Erozyon "
               "algoritması doğal çukurlar üretir; `z < su_seviyesi` olan "
               "kapalı bölgeler göl olur. **Nehir**: yüksekten alçağa "
               "**sürekli iniş yolu**. Algoritma: her noktadan en dik "
               "komşuya in, ta ki su seviyesine ulaşana kadar — bu "
               "**akış yolu** nehirdir. Aynı yolu **aşındırarak** (derin "
               "kazıp genişleterek) gerçek nehir yatağı elde edilir. "
               "Nehir material katmanı: **çakıl/ıslak toprak**.",
               ["göl", "nehir", "akış yolu"], code=(
                   "def akis_yonu(yukseklik, n, i):\n"
                   "    \"\"\"Bir noktadan en dik inis yonunu bul.\"\"\"\n"
                   "    satir, sutun = divmod(i, n)\n"
                   "    en_iyi, en_dik = None, 0.0\n"
                   "    for dr, ds in ((-1,0),(1,0),(0,-1),(0,1)):\n"
                   "        r2, s2 = satir + dr, sutun + ds\n"
                   "        if 0 <= r2 < n and 0 <= s2 < n:\n"
                   "            k = r2 * n + s2\n"
                   "            fark = yukseklik[i] - yukseklik[k]\n"
                   "            if fark > en_dik:\n"
                   "                en_dik, en_iyi = fark, k\n"
                   "    return en_iyi\n"
                   "\n"
                   "def nehir_bul(yukseklik, n, baslangic):\n"
                   "    \"\"\"Yuksekten alcaga inen akis yolunu izle.\"\"\"\n"
                   "    yol, mevcut, adim = [baslangic], baslangic, 0\n"
                   "    while adim < n * n:\n"
                   "        sonraki = akis_yonu(yukseklik, n, mevcut)\n"
                   "        if sonraki is None or sonraki in yol:\n"
                   "            break            # cukur veya dongu -> dur\n"
                   "        yol.append(sonraki)\n"
                   "        mevcut, adim = sonraki, adim + 1\n"
                   "    return yol\n"
                   "\n"
                   "print('nehir fonksiyonu hazir')")),
        Lesson("lowpoly_detay", "Yol nasıl yapılır?",
               "Yol, araziye **uyan** bir şerittir. Adımlar: **1)** iki "
               "nokta arasında **A\\*** veya **Dijkstra** ile en az maliyetli "
               "yol (maliyet = yükseklik farkı; dik yerden kaçınır), "
               "**2)** yol boyunca noktaları **düzleştir** (yol düz "
               "olmalı, araziyi kesmez), **3)** yol genişliğini "
               "**ağırlık haritasına** yaz, **4)** yol material katmanı "
               "ata (toprak/çakıl). Yol kenarına hafif **şev** (eğim) "
               "eklemek doğal görünür.",
               ["yol", "a*", "düzleştirme"], code=(
                   "import heapq\n"
                   "\n"
                   "def yol_bul(yukseklik, n, baslangic, hedef):\n"
                   "    \"\"\"A*: dik yerlerden kacinan en az maliyetli yol.\"\"\"\n"
                   "    baslangic, hedef = tuple(baslangic), tuple(hedef)\n"
                   "    kuyruk = [(0.0, baslangic)]\n"
                   "    onceki = {baslangic: None}\n"
                   "    maliyet = {baslangic: 0.0}\n"
                   "    while kuyruk:\n"
                   "        _, mevcut = heapq.heappop(kuyruk)\n"
                   "        if mevcut == hedef:\n"
                   "            yol = []\n"
                   "            while mevcut is not None:\n"
                   "                yol.append(mevcut)\n"
                   "                mevcut = onceki[mevcut]\n"
                   "            return yol[::-1]\n"
                   "        r, s = mevcut\n"
                   "        for dr, ds in ((-1,0),(1,0),(0,-1),(0,1)):\n"
                   "            r2, s2 = r + dr, s + ds\n"
                   "            if not (0 <= r2 < n and 0 <= s2 < n):\n"
                   "                continue\n"
                   "            # Maliyet: yukseklik farki (dik yerden kacin)\n"
                   "            fark = abs(yukseklik[r2 * n + s2] -\n"
                   "                       yukseklik[r * n + s])\n"
                   "            yeni = maliyet[mevcut] + 1.0 + fark * 8.0\n"
                   "            if (r2, s2) not in maliyet or yeni < maliyet[(r2, s2)]:\n"
                   "                maliyet[(r2, s2)] = yeni\n"
                   "                onceki[(r2, s2)] = mevcut\n"
                   "                tahmin = abs(r2 - hedef[0]) + abs(s2 - hedef[1])\n"
                   "                heapq.heappush(kuyruk, (yeni + tahmin, (r2, s2)))\n"
                   "    return []\n"
                   "\n"
                   "print('yol fonksiyonu hazir')")),
        Lesson("lowpoly_detay", "Dağlık bölge nasıl oluşturulur?",
               "Dağ, **yüksek genlik + düşük frekans** ile başlar: geniş "
               "ve yüksek kütle. Sonra **erozyon** ile sırtlar ve vadiler "
               "eklenir (dağlar düz koni değildir). Ek teknikler: "
               "**ridged noise** (`1 - |noise|` — keskin sırtlar üretir, "
               "dağ için ideal), **domain warping** (noise koordinatını "
               "başka noise ile kaydırma — çok doğal şekiller), "
               "**kar sınırı** (belirli yükseklik üstü beyaz katman), "
               "**kaya dokusu** dik yamaçlarda (eğim kuralı).",
               ["dağ", "ridged noise", "domain warping"], code=(
                   "import math, random\n"
                   "\n"
                   "def ridged_noise(x: float, tohum: int = 0) -> float:\n"
                   "    \"\"\"Keskin sirtlar uretir - daglar icin ideal.\"\"\"\n"
                   "    r = random.Random(int(x * 100) * 7919 + tohum)\n"
                   "    return 1.0 - abs(r.uniform(-1, 1))\n"
                   "\n"
                   "def dag_profili(ham01: float, keskinlik: float = 2.5) -> float:\n"
                   "    \"\"\"Yuksek alanlari keskinlestir, alcaklari duzlestir.\"\"\"\n"
                   "    return ham01 ** keskinlik\n"
                   "\n"
                   "# Karsilastirma: normal noise vs ridged\n"
                   "print('normal profil :', [round(dag_profili(h, 1.0), 2)\n"
                   "                           for h in (0.2, 0.5, 0.8)])\n"
                   "print('dag profili   :', [round(dag_profili(h, 2.5), 2)\n"
                   "                           for h in (0.2, 0.5, 0.8)])\n"
                   "print('ridged ornek  :', [round(ridged_noise(x), 2)\n"
                   "                           for x in range(5)])")),
        Lesson("lowpoly_detay", "Kıyı şeridi ve geçiş bölgeleri",
               "Keskin su/arazi sınırı yapay görünür. **Kıyı şeridi** "
               "katmanları: **su altı** (koyu kum), **ıslak kum** "
               "(su seviyesinin ±5 cm'si, koyu ve parlak), **kuru kum** "
               "(açık), **çimen** (yumuşak geçiş). Ek olarak **köpük** "
               "(su kenarında beyaz şerit) ve **hafif eğim** (su altı "
               "yüzeyi yumuşak) doğallık katar. Kural: geçiş "
               "**yüksekliğe göre smoothstep** ile yapılır, keskin "
               "sınırla değil.",
               ["kıyı", "ıslak kum", "köpük"]),
        Lesson("lowpoly_detay", "Arazi iyileştirmeleri: kontrol listesi",
               "Kaliteli arazi için kontrol listesi: **1) Ölçek** — "
               "yükseklik/genişlik oranı gerçekçi mi (dağ ~1:5, tepe "
               "~1:20), **2) Erozyon** — vadiler var mı, yoksa yapay mı, "
               "**3) Su** — akış yönü mantıklı mı (yukarıdan aşağı), "
               "**4) Katmanlar** — yükseklik+eğim birlikte kullanılmış mı, "
               "**5) Geçişler** — keskin sınır var mı (olmamalı), "
               "**6) Yol** — araziye uyuyor mu, dik geçiyor mu, "
               "**7) Yoğunluk** — bitki örtüsü biyoma uygun mu, "
               "**8) LOD** — uzakta basitleşiyor mu, "
               "**9) Yüz bütçesi** — hedefin altında mı.",
               ["kontrol listesi", "kalite", "ölçek"]),
        Lesson("lowpoly_detay", "Aşamalı iyileştirme: basitten karmaşığa",
               "Arazi **tek seferde** yapılmaz; aşamalı geliştirilir: "
               "**Aşama 1** düz zemin + tek renk. **Aşama 2** noise ile "
               "yükseklik. **Aşama 3** erozyon + dağ profili. "
               "**Aşama 4** material katmanları (yükseklik). "
               "**Aşama 5** eğim tabanlı katmanlar. **Aşama 6** su + "
               "nehir. **Aşama 7** yol + bitki örtüsü. **Aşama 8** LOD + "
               "optimizasyon. Her aşama **çalışır durumda** bırakılır — "
               "böylece bir sorun çıktığında hangi aşamada olduğu bilinir.",
               ["aşamalı", "iterasyon", "geliştirme"]),
        Lesson("lowpoly_detay", "Sahne kurulumu: ışık, kamera, render",
               "Üretilen varlığı **görebilmek** için sahne gerekir: "
               "**güneş** (`SUN`, `energy` 3–5, açı 45° — düz ışık "
               "detayı gizler), **gökyüzü** (`world` rengi açık mavi), "
               "**kamera** (`ORTHO` izometrik görünüm lowpoly için "
               "idealdir; `PERSP` gerçekçi), **zemin** (gölge alıcı). "
               "Render: `CYCLES` (kaliteli, yavaş) veya `BLENDER_EEVEE` "
               "(hızlı, oyun benzeri). Lowpoly sunumunda **EEVEE + "
               "flat shading** çoğu zaman yeterlidir.",
               ["ışık", "kamera", "render"], code=(
                   "import bpy, math\n"
                   "\n"
                   "def sahne_kur(iso=True):\n"
                   "    # Işik: gunes, 45 derece\n"
                   "    bpy.ops.object.light_add(type='SUN', location=(5, -5, 8))\n"
                   "    gunes = bpy.context.object\n"
                   "    gunes.data.energy = 4.0\n"
                   "    gunes.rotation_euler = (math.radians(50), 0,\n"
                   "                            math.radians(35))\n"
                   "    # Kamera\n"
                   "    bpy.ops.object.camera_add(location=(9, -9, 7))\n"
                   "    kamera = bpy.context.object\n"
                   "    if iso:\n"
                   "        kamera.data.type = 'ORTHO'\n"
                   "        kamera.data.ortho_scale = 14.0\n"
                   "    kamera.rotation_euler = (math.radians(60), 0,\n"
                   "                             math.radians(45))\n"
                   "    bpy.context.scene.camera = kamera\n"
                   "    # Gokyuzu rengi\n"
                   "    dunya = bpy.context.scene.world\n"
                   "    if dunya:\n"
                   "        dunya.use_nodes = True\n"
                   "        arka = dunya.node_tree.nodes.get('Background')\n"
                   "        if arka:\n"
                   "            arka.inputs['Color'].default_value = (0.55, 0.72, 0.9, 1)\n"
                   "    return gunes, kamera\n"
                   "\n"
                   "print('sahne fonksiyonu hazir')")),
    ]


# ─── Müfredat ───────────────────────────────────────────────────────────────

def build_lowpoly_curriculum(
        layers: Sequence[str] | None = None) -> List[Lesson]:
    """
    Lowpoly/prosedürel içerik müfredatını üretir.

    Args:
        layers: Yalnızca bu katmanlar (None = hepsi, sıralı)

    Returns:
        Ders listesi
    """
    builders = {
        "lowpoly_temel": _temel_lessons,
        "lowpoly_bitki": _bitki_lessons,
        "lowpoly_arazi": _arazi_lessons,
        "lowpoly_doku": _doku_lessons,
        "lowpoly_detay": _detay_lessons,
    }
    istenen = list(layers) if layers else list(LOWPOLY_LAYERS)
    dersler: List[Lesson] = []
    for ad in istenen:
        uretici = builders.get(ad)
        if uretici is not None:
            dersler.extend(uretici())
    return dersler


def lowpoly_stats(layers: Sequence[str] | None = None) -> Dict[str, object]:
    """Lowpoly müfredatı istatistikleri."""
    dersler = build_lowpoly_curriculum(layers)
    per_layer: Dict[str, Dict[str, int]] = {}
    for ders in dersler:
        kayit = per_layer.setdefault(ders.layer,
                                     {"lessons": 0, "chars": 0, "code": 0})
        kayit["lessons"] += 1
        kayit["chars"] += len(ders.as_document())
        kayit["code"] += 1 if ders.code.strip() else 0
    return {
        "total_lessons": len(dersler),
        "total_chars": sum(item["chars"] for item in per_layer.values()),
        "per_layer": per_layer,
        "layers": [LOWPOLY_LABELS.get(a, a)
                   for a in (layers or LOWPOLY_LAYERS)],
    }


__all__ = [
    "LOWPOLY_LAYERS", "LOWPOLY_LABELS", "build_lowpoly_curriculum",
    "lowpoly_stats",
]
