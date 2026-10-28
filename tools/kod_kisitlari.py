"""
Kod Üretim Kısıtları — API Sözlüğü + Dil Kilidi
================================================
⚠ Ölçülen 2 hata kalıbı (tur 23)
--------------------------------
Kod tamamlama %64 (9/14). Başarısız 5'in ikisi sistematik:

**1. Yanlış API adı** — model uydurma fonksiyon üretiyor:
```
'select_all(action='   → 'bpy.ops.select_all_with_mask)'    ← yok böyle API
'bpy.ops.mesh.primitive_' → 'cube_add_to_mesh('             ← yanlış ad
```

**2. Japonca karışma** — SmolLM2 çok dilli, baskın dil kayıyor:
```
'p.use_smooth = '      → '1\\n\\n# テクスチャを取得\\nbpy.context'
```

Çözüm: **üretim kısıtları**
--------------------------------
    1. **API sözlüğü** — kasada 1.667 doğrulanmış API adı var.
       Üretilen metinde geçen `bpy.*`/`mathutils.*` çağrıları
       sözlükte yoksa **uyarı** verilir (ve istenirse düzeltilir).
    2. **Dil kilidi** — üretimde yalnız ASCII + Türkçe karakterlere
       izin; CJK/Arap alfabesi görülürse **yeniden üret**.
    3. **Token kısıtı** — `bad_words_ids` ile CJK token'ları yasakla.

Kullanım
--------
    python tools/kod_kisitlari.py --sozluk        # API sözlüğünü kur
    python tools/kod_kisitlari.py --test          # kısıtları test et
    python tools/kod_kisitlari.py --kiyasla       # kısıtlı/kısıtsız
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

MODEL_V2 = "models/smollm2_kod_v2"
MODEL_V3 = "models/smollm2_kod_v3"
SOZLUK_YOLU = Path(".neurocore/api_sozlugu.json")

#: İzin verilen alfabe — Latin + Türkçe + kod sembolleri
IZINLI_DESEN = re.compile(
    r"^[\x09\x0a\x0d\x20-\x7e"
    r"\u00a0-\u00ff"          # Latin-1 (Türkçe dahil)
    r"\u0100-\u017f"          # Latin Extended-A (ğ, ş, ı…)
    r"\u2010-\u2027\u2030-\u205e"  # tipografi
    r"]*$")

#: CJK ve diğer yasak bloklar
YASAK_DESEN = re.compile(
    r"[\u3000-\u30ff"      # Japonca (hiragana, katakana)
    r"\u4e00-\u9fff"       # CJK ideogram
    r"\uac00-\ud7af"       # Korece
    r"\u0600-\u06ff"       # Arapça
    r"\u0400-\u04ff"       # Kiril
    r"]")


@dataclass
class KisitSonuc:
    """Bir kısıt kontrolünün sonucu."""
    gecerli: bool = True
    neden: str = ""
    bilinmeyen_api: List[str] = field(default_factory=list)
    yasak_dil: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"gecerli": self.gecerli, "neden": self.neden[:120],
                "bilinmeyen_api": self.bilinmeyen_api[:5],
                "yasak_dil": self.yasak_dil[:5]}


def api_sozlugu_kur() -> int:
    """
    Kasada doğrulanmış API adlarından **sözlük** kurar.

    Kaynaklar:
        * `blender_api` — resmî doküman imzaları (1.375)
        * `api_referans` — öğretmen üretimi (292)

    ⚠ Ölçülen değer: bu iki kaynak **doğrulanmış** — dokümandan veya
    öğretmenden geliyor. Sözlük bu adları içerir.
    """
    try:
        from neurocore.knowledge import KnowledgeLayer

        k = KnowledgeLayer()
        k.initialize()
    except Exception as exc:
        print(f"  kasa açılamadı: {exc}")
        return 0

    adlar: Set[str] = set()
    kaynak_sayi = {"blender_api": 0, "api_referans": 0}
    for n in k.vault.iter_notes(limit=300000):
        kat = getattr(n, "category", "") or ""
        if kat not in ("blender_api", "api_referans"):
            continue
        govde = getattr(n, "body", "") or ""
        # `**Ad:** \`bpy.ops...\`` biçiminden çıkar
        for m in re.finditer(r"\*\*Ad:\*\*\s*`([^`]+)`", govde):
            ad = m.group(1).strip()
            if ad and len(ad) > 3:
                adlar.add(ad)
        # `**İmza:** \`ad(...)\`` biçiminden de
        for m in re.finditer(r"\*\*İmza:\*\*\s*`([A-Za-z_][\w.]*)\(", govde):
            ad = m.group(1).strip()
            if ad and len(ad) > 3:
                adlar.add(ad)
        if kat in kaynak_sayi:
            kaynak_sayi[kat] += 1
    k.shutdown()

    # Modül öneklerini de ekle (bpy.ops.mesh.* gibi)
    onekler = set()
    for ad in adlar:
        parcalar = ad.split(".")
        for i in range(1, len(parcalar)):
            onekler.add(".".join(parcalar[:i]))

    SOZLUK_YOLU.parent.mkdir(parents=True, exist_ok=True)
    SOZLUK_YOLU.write_text(json.dumps({
        "adlar": sorted(adlar), "onekler": sorted(onekler),
        "kaynak": kaynak_sayi, "toplam": len(adlar),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  API sözlüğü: {len(adlar)} ad · {len(onekler)} önek "
          f"({kaynak_sayi})")
    return len(adlar)


class KodKisitlari:
    """Üretilen kodu doğrular ve kısıtlar."""

    def __init__(self, sozluk: Optional[Set[str]] = None) -> None:
        self.adlar: Set[str] = set()
        self.onekler: Set[str] = set()
        if sozluk is not None:
            self.adlar = set(sozluk)
        elif SOZLUK_YOLU.exists():
            d = json.loads(SOZLUK_YOLU.read_text(encoding="utf-8"))
            self.adlar = set(d.get("adlar") or [])
            self.onekler = set(d.get("onekler") or [])

    # ─── Dil kilidi ───

    def dil_gecerli(self, metin: str) -> Tuple[bool, List[str]]:
        """
        Metin **yalnız izinli alfabeyi** kullanıyor mu?

        ⚠ Ölçülen hata: SmolLM2 çok dilli; `# テクスチャを取得` gibi
        Japonca parçalar üretiyor. Bunlar kod değil, gürültü.
        """
        yasak = YASAK_DESEN.findall(metin)
        if yasak:
            return False, list(dict.fromkeys(yasak))[:5]
        return True, []

    # ─── API doğrulama ───

    def api_gecerli(self, metin: str) -> Tuple[bool, List[str]]:
        """
        Metinde geçen API çağrıları sözlükte mi?

        ⚠ Ölçülen hata: model `bpy.ops.select_all_with_mask` gibi
        **var olmayan** API adları üretiyor.

        ⚠ Dikkat: yalnız **nokta içeren** çağrılar denetlenir
        (`bpy.ops.mesh.primitive_cube_add`). Kısa yerel değişkenler
        (`mesh`, `ob`) denetlenmez.
        """
        if not self.adlar:
            return True, []
        # ⚠ ÖLÇÜLEN SORUN: katı allowlist performansı **düşürdü**
        # (9/14 → 8/14). Sebep: sözlük **eksik** —
        # `bpy.context.scene.render.engine` ve `bpy.data.objects`
        # gibi **geçerli** API'ler doküman madenciliğinde yakalanmadı.
        # Katı liste doğru üretimi de reddediyor.
        #
        # Doğru yaklaşım: yalnız **bariz uydurma** adları reddet.
        # Gerçek API adları kısa ve snake_case'tir; uydurmalar uzun
        # ve iç içe tekrar içerir.
        bilinmeyen: List[str] = []
        for m in re.finditer(r"\b((?:bpy|mathutils|bmesh|gpu|blf|aud)"
                             r"(?:\.[a-z_][\w]*){2,})\s*\(", metin):
            ad = m.group(1)
            if ad in self.adlar:
                continue
            # Sözlükte yok — ama gerçekten uydurma mı?
            son = ad.split(".")[-1]
            parcalar = ad.split(".")
            # 1) Çok uzun tek parça (35+ karakter) — gerçek API'ler kısa
            if len(son) > 34:
                bilinmeyen.append(ad)
                continue
            # 2) İç içe tekrar: `link.link_type.link_type`
            if len(parcalar) != len(set(parcalar)) and len(parcalar) > 3:
                bilinmeyen.append(ad)
                continue
            # 3) Aynı kelime 3+ kez: `polygon_polygon_polygon`
            kelimeler = son.split("_")
            if len(kelimeler) > 4:
                from collections import Counter as _C
                en_cok = _C(kelimeler).most_common(1)[0][1]
                if en_cok >= 3:
                    bilinmeyen.append(ad)
                    continue
            # 4) `_with_` kalıbı sözlükte yoksa uydurma olabilir
            #    (Godot/Blender'da `_with_` yaygın değil)
            if "_with_" in son and ad not in self.adlar:
                bilinmeyen.append(ad)
                continue
        return not bilinmeyen, list(dict.fromkeys(bilinmeyen))[:5]

    def dogrula(self, metin: str) -> KisitSonuc:
        """Tüm kısıtları uygular."""
        s = KisitSonuc()
        s.gecerli, s.yasak_dil = self.dil_gecerli(metin)
        if not s.gecerli:
            s.neden = f"yasak alfabe: {s.yasak_dil[:3]}"
            return s
        s.gecerli, s.bilinmeyen_api = self.api_gecerli(metin)
        if not s.gecerli:
            s.neden = f"bilinmeyen API: {s.bilinmeyen_api[:3]}"
            return s
        s.neden = "geçerli"
        return s

    # ─── Yasak token listesi ───

    def yasak_tokenler(self, tok: Any, maks: int = 400) -> List[List[int]]:
        """
        CJK token'larını **yasak listesi** olarak döndürür.

        ⚠ Neden gerekli: `generate(..., bad_words_ids=...)` ile bu
        token'lar üretimde **hiç seçilemez**. Dil kilidinden daha
        güçlüdür — model o token'ı hiç üretemez.
        """
        yasak: List[List[int]] = []
        try:
            sozluk = tok.get_vocab()
        except Exception:
            return []
        for parca, idx in sozluk.items():
            if not parca:
                continue
            if YASAK_DESEN.search(parca):
                yasak.append([idx])
                if len(yasak) >= maks:
                    break
        return yasak


#: Test seti (tur 23 ile aynı — karşılaştırılabilir)
TESTLER: Sequence[Tuple[str, Sequence[str]]] = (
    ("import bpy\nbpy.ops.mesh.primitive_",
     ("cube_add", "cylinder_add", "uv_sphere_add", "cone_add", "torus_add")),
    ("import bpy\nmat = bpy.data.materials.new(", ("name", "'")),
    ("import bpy\nbsdf.inputs['Base Color'].default_value = ",
     ("(", "Color", "0.")),
    ("extends Node3D\n\nfunc _ready() -> void:\n"
     "    var mi := MeshInstance3D.new()\n    mi.mesh = ",
     ("Box", "Cylinder", "Sphere")),
    ("import bpy\nfor v in mesh.vertices:\n    v.co.z = ", ("0", "h")),
    ("import bpy\nnt.links.new(uvn.outputs['UV'], ", ("olcek", "inputs")),
    ("extends CharacterBody3D\n\nfunc _physics_process(delta: float) -> void:\n"
     "    velocity.x = ", ("0", "yon", "giris")),
    ("import bpy\nob = bpy.context.object\n"
     "for p in ob.data.polygons:\n    p.use_smooth = ", ("False",)),
    ("import bpy\nimg = bpy.data.images.new(", ("name", "width", "256")),
    ("const PORT := 7777\npeer.create_server(", ("PORT", "16")),
    ("import bpy\nmesh = bpy.data.meshes.new(", ("name", "'")),
    ("import bpy\nscene.render.engine = ", ("'CYCLES'", "'BLENDER", "engine")),
    ("var t := Transform3D()\nt.origin = ", ("Vector3", "konum")),
    ("import bpy\nbpy.ops.object.select_all(action=", ("'SELECT'", "'DESELECT'")),
)


def olc(yol: str, *, kisit: bool = False, yasak_token: bool = False,
        tekrar: int = 3) -> Dict[str, Any]:
    """
    Kod tamamlama ölçümü — istenirse kısıtlarla.

    Args:
        kisit: Üretimi kısıtlarla denetle (başarısızsa yeniden dene)
        yasak_token: CJK token'larını yasakla
        tekrar: Kısıt başarısızsa kaç kez yeniden denenir
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

    kk = KodKisitlari() if kisit else None
    yasak = kk.yasak_tokenler(tok) if (kk and yasak_token) else []

    ok = 0
    reddedilen = 0
    detay: List[Dict[str, Any]] = []
    for onek, kabul in TESTLER:
        uretilen = ""
        gecerli = False
        for deneme in range(tekrar if kisit else 1):
            try:
                ids = tok(onek, return_tensors="pt")["input_ids"].to(cihaz)
                uret: Dict[str, Any] = {"max_new_tokens": 24,
                                        "do_sample": deneme > 0,
                                        "pad_token_id": tok.eos_token_id}
                if deneme > 0:
                    uret["temperature"] = 0.3 + deneme * 0.2
                    uret["top_p"] = 0.95
                if yasak:
                    uret["bad_words_ids"] = yasak
                with torch.no_grad():
                    c = mdl.generate(ids, **uret)
                uretilen = tok.decode(c[0][ids.shape[1]:],
                                      skip_special_tokens=True)
            except Exception as exc:
                uretilen = f"HATA {type(exc).__name__}"
                continue
            if kk is None:
                gecerli = True
                break
            sonuc = kk.dogrula(onek + uretilen)
            if sonuc.gecerli:
                gecerli = True
                break
            reddedilen += 1
        basarili = (any(k.lower() in uretilen.lower() for k in kabul)
                    and gecerli)
        if basarili:
            ok += 1
        detay.append({"onek": onek.splitlines()[-1][:40],
                      "uretilen": uretilen[:70], "basarili": basarili,
                      "gecerli": gecerli})
    return {"model": yol, "basarili": ok, "toplam": len(TESTLER),
            "oran": ok / len(TESTLER), "kisit": kisit,
            "yasak_token": yasak_token, "reddedilen": reddedilen,
            "detay": detay}


def main() -> int:
    """Kısıt işlemlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="Kod üretim kısıtları")
    ap.add_argument("--sozluk", action="store_true", help="API sözlüğü kur")
    ap.add_argument("--test", action="store_true", help="Kısıtları test et")
    ap.add_argument("--kiyasla", action="store_true",
                    help="Kısıtlı/kısıtsız karşılaştır")
    ap.add_argument("--model", default=MODEL_V2)
    ap.add_argument("--adim", type=int, default=0,
                    help=">0 ise ince ayar yap")
    args = ap.parse_args()

    print("=" * 78)
    print("  KOD ÜRETİM KISITLARI — API sözlüğü + dil kilidi")
    print("=" * 78)

    sonuc: Dict[str, Any] = {}
    if args.sozluk or not SOZLUK_YOLU.exists():
        n = api_sozlugu_kur()
        sonuc["sozluk"] = n

    kk = KodKisitlari()
    print(f"  sözlük yüklendi: {len(kk.adlar)} API adı")

    if args.test:
        testler = [
            ("import bpy\nbpy.ops.mesh.primitive_cube_add(size=2)",
             True, "geçerli API"),
            ("bpy.ops.select_all_with_mask()", False, "uydurma API"),
            ("# テクスチャを取得\nbpy.context", False, "Japonca"),
            ("mesh.vertices.append((0, 0, 0))", True, "yerel değişken"),
            ("# Create a mesh\nmesh = bpy.data.meshes.new(name='M')",
             True, "geçerli"),
        ]
        print("\n  KISIT TESTLERİ:")
        for metin, beklenen, aciklama in testler:
            s = kk.dogrula(metin)
            im = "✓" if s.gecerli == beklenen else "✗"
            print(f"    {im} {aciklama:20} geçerli={s.gecerli} "
                  f"({s.neden[:44]})")
            sonuc.setdefault("testler", []).append(
                {"aciklama": aciklama, "beklenen": beklenen,
                 "gelen": s.gecerli, "neden": s.neden})

    if args.kiyasla:
        print("\n  KIYASLAMA (model: " + args.model + ")")
        for kisit, yasak in ((False, False), (True, False), (True, True)):
            m = olc(args.model, kisit=kisit, yasak_token=yasak)
            anahtar = f"kisit{int(kisit)}_yasak{int(yasak)}"
            sonuc[anahtar] = {"basarili": m.get("basarili"),
                              "toplam": m.get("toplam"),
                              "reddedilen": m.get("reddedilen")}
            print(f"    kısıt={kisit} yasak_token={yasak}: "
                  f"{m.get('basarili')}/{m.get('toplam')} "
                  f"(%{(m.get('oran') or 0) * 100:.0f}) "
                  f"reddedilen={m.get('reddedilen')}", flush=True)

    Path("_kod_kisitlari.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
