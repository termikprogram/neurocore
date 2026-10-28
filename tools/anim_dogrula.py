"""
Animasyon Doğrulama — Diskten Yeniden Açıp Ölç
================================================
⚠ Neden ayrı doğrulama?
-----------------------
Tur 6, 8 ve 9'da **üç kez** şu desen görüldü:
```
üretim ölçümü : "5 animasyon üretildi, 350 keyframe"   ✓ İYİ
diskten       : 1 action, 14 fcurve, DOLU=0             ✗ GERÇEK
```

**Sebep çeşitliydi:**
```
tur 6 : `keyframe_insert` keyfi array_index'i yazmıyor
tur 8 : kanal başına 25 fcurve açılmaya çalışılıyor
tur 9 : kullanılmayan action'lar kaydederken SİLİNİYOR
        (fake user yok → Blender temizler)
```

**Tek güvenilir yöntem:** kaydet → **yeniden aç** → ölç.

Bu betik tüm armature'lü varlıkları tarar ve her action'ı ölçer.

Kullanım
--------
    python tools/anim_dogrula.py --tara
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

#: ⚠ Diskten action ölçümü
OLCUM = r'''
import bpy, json

bpy.ops.wm.open_mainfile(filepath=BLEND_YOLU)

o = {"armature": 0, "kemik": 0, "mesh": 0, "vg": 0, "vertex": 0,
     "agirlikli": 0, "uv": 0, "actions": [], "fake_user": 0}

for ob in bpy.data.objects:
    if ob.type == "ARMATURE":
        o["armature"] += 1
        o["kemik"] = len(ob.data.bones)
    elif ob.type == "MESH" and not o["mesh"]:
        me = ob.data
        o["mesh"] = 1
        o["vertex"] = len(me.vertices)
        o["vg"] = len(ob.vertex_groups)
        o["agirlikli"] = sum(1 for v in me.vertices if len(v.groups) > 0)
        o["uv"] = len(me.uv_layers)

for a in bpy.data.actions:
    if getattr(a, "use_fake_user", False):
        o["fake_user"] += 1
    fc = 0
    kf = 0
    dolu = 0
    bos = 0
    for kat in a.layers:
        for st in kat.strips:
            for cb in st.channelbags:
                for f in cb.fcurves:
                    fc += 1
                    n = len(f.keyframe_points)
                    kf += n
                    try:
                        e = max(abs(float(x.co[1])) for x in f.keyframe_points)
                    except ValueError:
                        e = 0.0
                    if e > 1e-6:
                        dolu += 1
                    else:
                        bos += 1
    o["actions"].append({
        "ad": a.name,
        "kare": [round(float(a.frame_range[0]), 1),
                 round(float(a.frame_range[1]), 1)],
        "fcurve": fc, "kf": kf, "dolu": dolu, "bos": bos,
        "dongu": bool(getattr(a, "use_cyclic", False)),
        "fake_user": bool(getattr(a, "use_fake_user", False)),
    })

o["action_sayisi"] = len(o["actions"])
print("V" + json.dumps(o, ensure_ascii=False))
'''


def dogrula(blend: Path, *, zaman: int = 300) -> Dict[str, Any]:
    """Tek `.blend` dosyasını diskten doğrular."""
    if not BLENDER.exists():
        return {"hata": "blender yok"}
    if not blend.exists():
        return {"hata": "dosya yok"}
    kod = f'BLEND_YOLU = r"{blend.resolve()}"\n' + OLCUM
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "d.py"
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
        if "V{" in satir:
            try:
                return json.loads(satir[satir.index("V{") + 1:])
            except Exception:
                pass
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:160] or "çıktı yok"}


def tara() -> Dict[str, Any]:
    """Tüm armature'lü varlıkları doğrular."""
    from tools.varlik_katalogu import listele

    sonuc: Dict[str, Any] = {"varlik": [], "toplam_action": 0,
                             "dolu_action": 0, "hata": 0}
    for v in listele():
        if not v.get("armature_var"):
            continue
        bl = Path(v.get("blend") or "")
        if not bl.exists():
            continue
        d = dogrula(bl)
        if "hata" in d:
            sonuc["hata"] += 1
            sonuc["varlik"].append({"id": v.get("id"), "hata": d["hata"]})
            continue
        d["id"] = v.get("id")
        d["blend"] = str(bl)
        sonuc["varlik"].append(d)
        sonuc["toplam_action"] += d.get("action_sayisi", 0)
        # ⚠ "Dolu" action: en az bir kanalı sıfırdan farklı VE kf > 10
        for a in d.get("actions", []):
            if a["dolu"] > 0 and a["kf"] > 10:
                sonuc["dolu_action"] += 1
    return sonuc


def main() -> int:
    """Doğrulama."""
    ap = argparse.ArgumentParser(description="Animasyon doğrulama")
    ap.add_argument("--tara", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  ANİMASYON DOĞRULAMA — diskten yeniden açıp ölç")
    print("=" * 78)

    if not args.tara:
        return 0

    t0 = time.perf_counter()
    s = tara()
    sure = time.perf_counter() - t0

    print(f"\n  {len(s['varlik'])} varlık · {sure:.1f} sn")
    for v in s["varlik"]:
        if "hata" in v:
            print(f"\n  ✗ {v['id']}: {v['hata'][:60]}")
            continue
        print(f"\n  ══ {v['id']} ══")
        print(f"    armature={v['armature']} kemik={v['kemik']} "
              f"vertex={v['vertex']} grup={v['vg']} "
              f"ağırlıklı={v['agirlikli']} uv={v['uv']}")
        print(f"    action={v['action_sayisi']} "
              f"(fake_user={v['fake_user']})")
        print(f"    {'ad':12} {'kare':>12} {'fcurve':>6} {'kf':>5} "
              f"{'DOLU':>4} {'BOŞ':>4} döngü")
        for a in sorted(v["actions"], key=lambda x: x["ad"]):
            bayrak = "✓" if a["dolu"] > 0 and a["kf"] > 10 else "✗"
            print(f"    {bayrak} {a['ad']:10} "
                  f"[{a['kare'][0]:>4},{a['kare'][1]:>4}] "
                  f"{a['fcurve']:>6} {a['kf']:>5} {a['dolu']:>4} "
                  f"{a['bos']:>4} {a['dongu']}")

    print(f"\n  {'=' * 74}")
    print(f"  TOPLAM ACTION : {s['toplam_action']}")
    print(f"  DOLU ACTION   : {s['dolu_action']}")
    print(f"  HATALI VARLIK : {s['hata']}")
    print(f"  {'=' * 74}")

    Path("_anim_dogrulama.json").write_text(
        json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
