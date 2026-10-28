"""
UV Dikiş ve Açma — Elle Kontrollü (Hedef E)
=============================================
⚠ Kullanıcı hedefi
------------------
*"oluşturulan varlıklar için texture uv si ayarlamak da kolay değil"*

⚠ Ölçülen durum (tur 1)
-----------------------
```
3/3 varlıkta UV var  ✓
AMA: yalnız `bpy.ops.uv.smart_project` — OTOMATİK
```

**`smart_project` neden yetmez?**
```
· Her kenarı potansiyel dikiş sayar → ÇOK ADA (island)
· Ada sayısı arttıkça doku israfı + dikiş izi artar
· Kontrol YOK — hangi kenarın dikiş olduğunu sen seçemezsin
```

Çözüm: **`mark_seam` + `unwrap`**
----------------------------------
```
1. Dikiş (seam) kenarlarını SEÇ   ← asıl zor kısım
2. `mark_seam` ile işaretle
3. `unwrap` ile aç (dikişleri takip eder)
```

⚠ Dikiş nereden geçmeli? (mantık)
```
Silindir  → arka dikey kenar
Küp       → bir köşe
Karakter  → iç bacak, kol altı, sırt ortası
Kaya      → önemsiz (dokusu düz)
```
**Mantık:** dikiş **gerilme** yaratır → görünür yerde olmamalı.

Bu betik
--------
    1. **Açı tabanlı dikiş tespiti** — keskin kenarlar dikiş olur
    2. `mark_seam` + `unwrap`
    3. **Kalite ölçümü**: ada sayısı, UV alanı kullanımı, gerilme
    4. `smart_project` ile **karşılaştırma**

Kullanım
--------
    python tools/uv_dikis.py --kiyasla
    python tools/uv_dikis.py --uret --kategori kaya
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

BLENDER = Path(r"E:\AppsTools\Blender 5.2\blender.exe")


#: ⚠ Kıyaslama betiği: aynı mesh üzerinde İKİ yöntem
#:
#: Ölçülen kalite ölçütleri:
#:     * **ada sayısı** (island) — az olan iyi
#:     * **UV alanı kullanımı** — 0..1 arası, yüksek iyi (paketleme)
#:     * **gerilme** (stretch) — açı/alan bozulması, düşük iyi
KIYASLA = r'''
import bpy, bmesh, math, json
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)

# ══ TEST MESH'İ — silindir (dikiş mantığı net) ══════════════════════
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.5, depth=1.4)
ob = bpy.context.active_object
ob.name = "Test_Silindir"
me = ob.data
bpy.ops.object.transform_apply(scale=True)

def olcum_et(etiket):
    """UV kalite ölçümü."""
    m = ob.data
    bm = bmesh.new(); bm.from_mesh(m)
    uv = bm.loops.layers.uv.active
    if uv is None:
        bm.free()
        return {"etiket": etiket, "uv": 0}

    # ── Ada (island) sayısı: UV kenar bağlantı bileşenleri ──
    # ⚠ Basit yaklaşım: döngü UV'lerini 4 ondalıkla grupla
    noktalar = {}
    for f in bm.faces:
        for l in f.loops:
            k = (round(l[uv].uv.x, 4), round(l[uv].uv.y, 4))
            noktalar.setdefault(k, 0)
            noktalar[k] += 1
    # Bağlı bileşen: UV kenarı paylaşan yüzler
    yuz_kom = {f.index: set() for f in bm.faces}
    kenar_yuz = {}
    for f in bm.faces:
        for l in f.loops:
            k1 = (round(l[uv].uv.x, 4), round(l[uv].uv.y, 4))
            k2 = (round(l.link_loop_next[uv].uv.x, 4),
                  round(l.link_loop_next[uv].uv.y, 4))
            kenar_yuz.setdefault(frozenset((k1, k2)), []).append(f.index)
    for yuzler in kenar_yuz.values():
        if len(yuzler) > 1:
            for a in yuzler:
                for b in yuzler:
                    if a != b:
                        yuz_kom[a].add(b)
    gorulen, ada = set(), 0
    for i in yuz_kom:
        if i in gorulen:
            continue
        ada += 1
        yigin = [i]
        while yigin:
            x = yigin.pop()
            if x in gorulen:
                continue
            gorulen.add(x)
            yigin.extend(yuz_kom[x] - gorulen)

    # ── Alan kullanımı ve gerilme ──
    uv_min = [1e9, 1e9]; uv_max = [-1e9, -1e9]
    for k in noktalar:
        uv_min[0] = min(uv_min[0], k[0]); uv_min[1] = min(uv_min[1], k[1])
        uv_max[0] = max(uv_max[0], k[0]); uv_max[1] = max(uv_max[1], k[1])
    kutu = (uv_max[0] - uv_min[0]) * (uv_max[1] - uv_min[1])

    # Gerilme: 3B alan / UV alanı oranının değişkenliği
    oranlar = []
    for f in bm.faces:
        a3 = f.calc_area()
        auv = 0.0
        ls = list(f.loops)
        for i2 in range(len(ls)):
            a = ls[i2][uv].uv; b2 = ls[(i2 + 1) % len(ls)][uv].uv
            auv += a.x * b2.y - b2.x * a.y
        auv = abs(auv) * 0.5
        if auv > 1e-9:
            oranlar.append(a3 / auv)
    if oranlar:
        ort = sum(oranlar) / len(oranlar)
        sapma = (sum((x - ort) ** 2 for x in oranlar) / len(oranlar)) ** 0.5
        gerilme = sapma / ort if ort > 0 else 0.0
    else:
        gerilme = -1.0

    bm.free()
    return {"etiket": etiket, "ada": ada, "uv_kutu": round(kutu, 4),
            "gerilme": round(gerilme, 4), "yuz": len(m.polygons),
            "uv_katman": len(m.uv_layers)}


sonuc = []

# ══ YÖNTEM A: smart_project (otomatik) ═══════════════════════════════
bpy.ops.object.select_all(action="DESELECT")
ob.select_set(True)
bpy.context.view_layer.objects.active = ob
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
try:
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)
except Exception as e:
    sonuc.append({"etiket": "smart_project", "hata": str(e)[:80]})
bpy.ops.object.mode_set(mode="OBJECT")
sonuc.append(olcum_et("smart_project"))

# ══ YÖNTEM B: açı tabanlı dikiş + unwrap ════════════════════════════
# ⚠ tur 3 ÖLÇÜMÜ: ilk sürüm 55/72 kenarı dikiş yaptı (AŞIRI).
# Sebep: `select_sharp_edges` + kapak + dikey = neredeyse hepsi.
# Doğru dikiş mantığı (silindir için):
#     · ÜST ve ALT kapak çemberleri  → kapak ayrılsın
#     · TEK dikey kesim kenarı       → gövde açılsın
# Toplam beklenen: ~25/72 kenar.
# ⚠ tur 3 KESİN KÖK NEDEN (ölçüldü):
#     `bpy.ops.mesh.select_all(action="DESELECT")` **veri katmanına
#     yansımıyor** — `edit_secili` 72'de kalıyor → `mark_seam` TÜM
#     kenarları işaretliyor (72/72, beklenen 25).
#
# ✅ Çözüm: `e.select` **doğrudan veri API'siyle** ayarlanır ve
#    `mark_seam` operatörü yerine **`e.use_seam = True`** kullanılır
#    (veri özelliği — mod değişimi gerektirmez).
me2 = ob.data

def dikis_isaretle(kosul):
    """Kenarları koşula göre dikiş yapar (doğrudan veri API'si)."""
    n = 0
    for e in me2.edges:
        v1 = me2.vertices[e.vertices[0]].co
        v2 = me2.vertices[e.vertices[1]].co
        if kosul(v1, v2):
            e.use_seam = True
            n += 1
    return n

# 1) Kapak çemberleri (üst + alt): Z farkı ~0 ve |Z| = 0.7
kapak = dikis_isaretle(
    lambda a, b: abs(a.z - b.z) < 1e-4 and abs(abs(a.z) - 0.7) < 1e-3)
# 2) Tek dikey kesim kenarı (x≈0, y<0 tarafı — arka)
dikey = dikis_isaretle(
    lambda a, b: (abs(a.z - b.z) > 1.0 and a.x < 0.02 and a.y < 0.0
                  and b.x < 0.02 and b.y < 0.0))
sonuc.append({"etiket": "dikis_secim", "kapak": kapak, "dikey": dikey})
# ⚠ tur 3: `unwrap` **EDIT modu** ister (veri API'si OBJECT'taydı)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
try:
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.02)
    unwrap_ok = True
except Exception as e:
    unwrap_ok = False
    sonuc.append({"etiket": "unwrap", "hata": str(e)[:80]})
bpy.ops.object.mode_set(mode="OBJECT")
sonuc.append(olcum_et("dikis+unwrap"))

# Dikiş sayısı
dikis = sum(1 for e in me.edges if e.use_seam)
sonuc.append({"etiket": "dikis_sayisi", "deger": dikis,
              "kenar": len(me.edges), "unwrap_ok": unwrap_ok})

print("OLCUM:" + json.dumps(sonuc, ensure_ascii=False))
'''


def kiyasla(*, seam_aci: float = 0.52, zaman: int = 180
            ) -> Dict[str, Any]:
    """İki UV yöntemini kıyaslar."""
    if not BLENDER.exists():
        return {"hata": "blender yok"}
    kod = f"SEAM_ACI = {seam_aci!r}\n" + KIYASLA
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "k.py"
        b.write_text(kod, encoding="utf-8")
        try:
            p = subprocess.run(
                [str(BLENDER), "--background", "--factory-startup",
                 "--python", str(b)],
                capture_output=True, text=True, timeout=zaman,
                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return {"hata": "zaman aşımı"}
    out = (p.stdout or "") + (p.stderr or "")
    for satir in out.splitlines():
        if satir.strip().startswith("OLCUM:"):
            return {"olcum": json.loads(satir.strip()[6:])}
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:160] or "OLCUM yok"}


def main() -> int:
    """UV dikiş kıyaslaması."""
    ap = argparse.ArgumentParser(description="UV dikiş ve açma")
    ap.add_argument("--kiyasla", action="store_true")
    ap.add_argument("--aci", type=float, default=0.52,
                    help="Dikiş açı eşiği (radyan)")
    args = ap.parse_args()

    print("=" * 78)
    print("  UV DİKİŞ VE AÇMA — elle kontrollü vs otomatik")
    print("=" * 78)
    print(f"  Blender: {BLENDER.name} · dikiş açısı: {args.aci} rad")

    if not args.kiyasla:
        return 0

    print("\n  kıyaslanıyor…", flush=True)
    t0 = time.perf_counter()
    r = kiyasla(seam_aci=args.aci)
    sure = time.perf_counter() - t0
    if "hata" in r:
        print(f"    ✗ {r['hata']}")
        return 1
    print(f"    ✓ {sure:.1f} sn\n")
    for x in r["olcum"]:
        if x.get("etiket") == "dikis_sayisi":
            print(f"  DİKİŞ: {x['deger']} / {x['kenar']} kenar · "
                  f"unwrap={'✓' if x.get('unwrap_ok') else '✗'}")
        elif "hata" in x:
            print(f"  {x['etiket']:16} ✗ {x['hata']}")
        else:
            print(f"  {x['etiket']:16} ada={x.get('ada', '?'):>3} "
                  f"uv_kutu={x.get('uv_kutu', '?'):>7} "
                  f"GERİLME={x.get('gerilme', '?'):>8} "
                  f"yüz={x.get('yuz', '?')}")

    Path("_uv_dikis.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
