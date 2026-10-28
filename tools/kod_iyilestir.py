"""
Kod Üretimi İyileştirme — Tekrar Cezası + Kod Ağırlığı
=======================================================
⚠ Ölçülen sorun (tur 22)
------------------------
SmolLM2 ince ayar ile kod tamamlama **1/14 → 8/14** yükseldi. Ama
başarısız 6/14'ün çoğu **tekrar döngüsü**:

    "bpy.ops.mesh.primitive_"  → "extrude_polygon_polygon_polygon_polygon_..."
    "nt.links.new(...)"        → "bpy.link.link_type=bpy.link.link_type.link..."

Ve **konu dışı** karışma (SmolLM2 çok dilli):
    "p.use_smooth = "          → "# テクスチャを取得"

Çözümler
--------
    1. **Tekrar cezası** — `repetition_penalty=1.3` üretimde
    2. **Sıcaklık** — `temperature=0.2` (deterministik ama esnek)
    3. **Kod ağırlığı** — kod notları eğitimde öne çıksın
    4. **Daha fazla adım** — 1500 → 8000

Kullanım
--------
    python tools/kod_iyilestir.py --egit --adim 8000
    python tools/kod_iyilestir.py --olc
    python tools/kod_iyilestir.py --karsilastir
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

TEMEL_MODEL = "models/smollm2_135m"
KOD_MODEL = "models/smollm2_kod"
KOD_MODEL_V2 = "models/smollm2_kod_v6"
#: ⚠ tur 31: saf kod verisi (prose temizlenmiş) kullanılıyor
VERI = Path(".neurocore/v6_veri.jsonl")
VERI_YEDEK = Path(".neurocore/smollm_veri_v2.jsonl")

#: Kod kategorileri — ağırlıklandırılacak
KOD_KATEGORI = ("kod_uretim", "kod_ogrenme", "kod_ornek",
                "blender_api", "api_referans")

#: Ağırlık: kod notları kaç kez tekrarlansın
KOD_TEKRAR = 3


def veri_uret_v2(maks_not: int = 8000) -> int:
    """
    **Kod ağırlıklı** eğitim verisi üretir.

    ⚠ Ölçülen sorun: v1 veri tüm notları eşit aldı; kod notları
    (%76) sayıca çok ama `learned`/`tercih` notları da karıştı.
    Yeni sürüm: kod notları **3× tekrarlanır**, diğerleri 1×.
    """
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return 0

    sayi = 0
    kod_sayi = 0
    with VERI.open("w", encoding="utf-8") as f:
        for n in k.vault.iter_notes(limit=maks_not):
            govde = (getattr(n, "body", "") or "").strip()
            baslik = (getattr(n, "title", "") or "").strip()
            if len(govde) < 40:
                continue
            kat = getattr(n, "category", "") or ""
            metin = f"# {baslik}\n\n{govde}\n\n"
            tekrar = KOD_TEKRAR if kat in KOD_KATEGORI else 1
            for _ in range(tekrar):
                f.write(json.dumps({"metin": metin, "kategori": kat},
                                   ensure_ascii=False) + "\n")
                sayi += 1
            if kat in KOD_KATEGORI:
                kod_sayi += 1
    k.shutdown()
    print(f"  veri v2: {sayi} satır ({kod_sayi} kod notu × {KOD_TEKRAR}) "
          f"→ {VERI}")
    return sayi


def egit(adim: int = 8000, lr: float = 3e-5, parti: int = 2,
         dizi: int = 512) -> Dict[str, Any]:
    """
    Uzun ince ayar — kod ağırlıklı veriyle.

    ⚠ Parti boyutu 2: 135M model + 512 dizi için ~1,5 GB VRAM.
    """
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    s: Dict[str, Any] = {}
    t0 = time.perf_counter()
    if not VERI.exists():
        veri_uret_v2()

    print(f"  model yükleniyor: {TEMEL_MODEL}")
    tok = AutoTokenizer.from_pretrained(TEMEL_MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(TEMEL_MODEL)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(cihaz)
    mdl.train()
    print(f"  cihaz: {cihaz}")

    # Veriyi token'la
    metinler: List[str] = []
    with VERI.open(encoding="utf-8") as f:
        for satir in f:
            try:
                metinler.append(json.loads(satir)["metin"])
            except Exception:
                continue
    print(f"  {len(metinler)} satır okundu")

    tokenler: List[Any] = []
    toplam = 0
    for m in metinler:
        t = tok(m, return_tensors="pt", truncation=True, max_length=dizi)
        if t["input_ids"].shape[1] < 16:
            continue
        tokenler.append(t["input_ids"][0])
        toplam += t["input_ids"].shape[1]
    print(f"  {len(tokenler)} örnek token'landı · {toplam:,} token")
    s["ornek"] = len(tokenler)
    s["token"] = toplam

    iy = torch.optim.AdamW(mdl.parameters(), lr=lr)
    rng = random.Random(42)
    kayiplar: List[float] = []
    for adim_no in range(1, adim + 1):
        idx = rng.randrange(len(tokenler))
        ids = tokenler[idx][:dizi].unsqueeze(0).to(cihaz)
        if ids.shape[1] < 8:
            continue
        cikti = mdl(input_ids=ids, labels=ids)
        cikti.loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        iy.step()
        iy.zero_grad(set_to_none=True)
        kayiplar.append(float(cikti.loss))
        if adim_no % 500 == 0:
            ort = sum(kayiplar[-500:]) / min(500, len(kayiplar))
            print(f"    adım {adim_no}/{adim} · kayıp {ort:.4f} · "
                  f"{(time.perf_counter() - t0) / 60:.1f} dk", flush=True)

    s["adim"] = len(kayiplar)
    s["kayip_ilk"] = kayiplar[0] if kayiplar else None
    s["kayip_son"] = (sum(kayiplar[-200:]) / min(200, len(kayiplar))
                      if kayiplar else None)
    s["sure_s"] = time.perf_counter() - t0

    hedef = Path(KOD_MODEL_V2)
    hedef.mkdir(parents=True, exist_ok=True)
    mdl.save_pretrained(str(hedef))
    tok.save_pretrained(str(hedef))
    print(f"  kaydedildi: {hedef}")
    s["kaydedildi"] = str(hedef)
    return s


#: Kod tamamlama test seti (14 test — tur 22 ile aynı, karşılaştırılabilir)
TESTLER: List[Tuple[str, List[str]]] = [
    ("import bpy\nbpy.ops.mesh.primitive_",
     ["cube_add", "cylinder_add", "uv_sphere_add", "cone_add", "torus_add"]),
    ("import bpy\nmat = bpy.data.materials.new(", ["name", "'"]),
    ("import bpy\nbsdf.inputs['Base Color'].default_value = ",
     ["(", "Color", "0."]),
    ("extends Node3D\n\nfunc _ready() -> void:\n"
     "    var mi := MeshInstance3D.new()\n    mi.mesh = ",
     ["Box", "Cylinder", "Sphere"]),
    ("import bpy\nfor v in mesh.vertices:\n    v.co.z = ", ["0", "h"]),
    ("import bpy\nnt.links.new(uvn.outputs['UV'], ", ["olcek", "inputs"]),
    ("extends CharacterBody3D\n\nfunc _physics_process(delta: float) -> void:\n"
     "    velocity.x = ", ["0", "yon", "giris"]),
    ("import bpy\nob = bpy.context.object\n"
     "for p in ob.data.polygons:\n    p.use_smooth = ", ["False"]),
    ("import bpy\nimg = bpy.data.images.new(", ["name", "width", "256"]),
    ("const PORT := 7777\npeer.create_server(", ["PORT", "16"]),
    ("import bpy\nmesh = bpy.data.meshes.new(", ["name", "'"]),
    ("import bpy\nscene.render.engine = ", ["'CYCLES'", "'BLENDER", "engine"]),
    ("var t := Transform3D()\nt.origin = ", ["Vector3", "konum"]),
    ("import bpy\nbpy.ops.object.select_all(action=", ["'SELECT'", "'DESELECT'"]),
]


def olc(yol: str, *, tekrar_cezasi: float = 1.0,
        sicaklik: float = 0.0) -> Dict[str, Any]:
    """
    Kod tamamlama ölçümü.

    Args:
        yol: Model dizini
        tekrar_cezasi: 1.0 = kapalı, 1.3 = açık (tekrar bastırma)
        sicaklik: 0.0 = deterministik (greedy), 0.2 = hafif rastgele
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not Path(yol).exists():
        return {"hata": f"model yok: {yol}"}
    tok = AutoTokenizer.from_pretrained(yol)
    mdl = AutoModelForCausalLM.from_pretrained(yol)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(cihaz)
    mdl.eval()

    ok = 0
    detay: List[Dict[str, Any]] = []
    for onek, kabul in TESTLER:
        try:
            ids = tok(onek, return_tensors="pt")["input_ids"].to(cihaz)
            uret = {"max_new_tokens": 24,
                    "pad_token_id": tok.eos_token_id,
                    "repetition_penalty": tekrar_cezasi}
            if sicaklik > 0:
                uret.update({"do_sample": True, "temperature": sicaklik,
                             "top_p": 0.95})
            else:
                uret["do_sample"] = False
            with torch.no_grad():
                c = mdl.generate(ids, **uret)
            u = tok.decode(c[0][ids.shape[1]:], skip_special_tokens=True)
        except Exception as exc:
            u = f"HATA {type(exc).__name__}"
        basarili = any(k.lower() in u.lower() for k in kabul)
        if basarili:
            ok += 1
        detay.append({"onek": onek.splitlines()[-1][:40],
                      "uretilen": u[:70], "basarili": basarili})
    return {"model": yol, "basarili": ok, "toplam": len(TESTLER),
            "oran": ok / len(TESTLER), "tekrar_cezasi": tekrar_cezasi,
            "sicaklik": sicaklik, "detay": detay}


def main() -> int:
    """İyileştirme işlemlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="Kod üretimi iyileştirme")
    ap.add_argument("--veri", action="store_true")
    ap.add_argument("--egit", action="store_true")
    ap.add_argument("--olc", action="store_true")
    ap.add_argument("--karsilastir", action="store_true",
                    help="Tekrar cezası açık/kapalı karşılaştır")
    ap.add_argument("--adim", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    print("=" * 78)
    print("  KOD ÜRETİMİ İYİLEŞTİRME — tekrar cezası + kod ağırlığı")
    print("=" * 78)

    sonuc: Dict[str, Any] = {}
    if args.veri or not VERI.exists():
        veri_uret_v2()
    if args.egit:
        e = egit(adim=args.adim, lr=args.lr)
        sonuc["egitim"] = e
        print(f"  EĞİTİM: {e.get('adim')} adım · "
              f"{e.get('kayip_ilk'):.4f} → {e.get('kayip_son'):.4f} · "
              f"{e.get('sure_s', 0) / 60:.1f} dk")
    if args.olc:
        yol = KOD_MODEL_V2 if Path(KOD_MODEL_V2).exists() else KOD_MODEL
        m = olc(yol)
        sonuc["olcum"] = m
        print(f"\n  ÖLÇÜM ({yol}): {m.get('basarili')}/{m.get('toplam')}")
        for d in m.get("detay", []):
            im = "OK " if d["basarili"] else "   "
            print(f"    {im}{d['onek']!r} -> {d['uretilen']!r}")
    if args.karsilastir:
        yol = KOD_MODEL_V2 if Path(KOD_MODEL_V2).exists() else KOD_MODEL
        print(f"\n  KARŞILAŞTIRMA ({yol})")
        for ceza, sic in ((1.0, 0.0), (1.3, 0.0), (1.3, 0.2)):
            m = olc(yol, tekrar_cezasi=ceza, sicaklik=sic)
            anahtar = f"ceza{ceza}_sic{sic}"
            sonuc[anahtar] = {"basarili": m.get("basarili"),
                              "toplam": m.get("toplam"),
                              "oran": m.get("oran")}
            print(f"    ceza={ceza} sıcaklık={sic}: "
                  f"{m.get('basarili')}/{m.get('toplam')} "
                  f"(%{(m.get('oran') or 0) * 100:.0f})", flush=True)

    Path("_kod_iyilestir.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
