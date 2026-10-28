"""
Kod Üretici — SmolLM2 Tabanlı Gerçek Kod Üretimi
==================================================
⚠ Ölçülen durum
---------------
```
myAi (211M, sıfırdan)     : kod tamamlama  1/14 (%7)
SmolLM2-v2 (135M, ince ayar): kod tamamlama  9/14 (%64)
```

**Hedef:** SmolLM2'yi üretim döngüsüne bağlamak — myAi karar verir,
SmolLM2 **kod yazar**, Godot doğrular.

Neden kod yazsın?
-----------------
Şablon kod çalışıyor ama **sabit**. Gerçek kod üretimi:
    * Kullanıcı isteğine göre **özel** kod
    * Yeni nesne türleri (şablonda olmayan)
    * Variasyonlar (farklı boyut, renk, davranış)

Mimari
------
    ┌──────────────────────────────────────────────┐
    │ 1. myAi KARAR KATMANI                        │
    │    palet, seviye, tuş, HUD + güvenlik filtresi│
    ├──────────────────────────────────────────────┤
    │ 2. SmolLM2 KOD ÜRETİCİ (bu araç)             │
    │    • istem → kod                              │
    │    • dil kilidi (CJK yasak)                   │
    │    • API sözlüğü (uydurma reddi)              │
    │    • tekrar deneme (kısıt reddederse)         │
    ├──────────────────────────────────────────────┤
    │ 3. ŞABLON (yedek)                            │
    │    kısıt 3 denemede geçmezse devreye girer    │
    ├──────────────────────────────────────────────┤
    │ 4. GODOT DOĞRULAMASI                         │
    │    --headless → exit=0                       │
    └──────────────────────────────────────────────┘

Kullanım
--------
    python tools/kod_uretici.py --istem "Labirent duvarı ekle" --nesne duvar
    python tools/kod_uretici.py --test           # içerik testleri
    python tools/kod_uretici.py --kip           # script üret
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Model dizini (sırayla denenir)
#: ⚠ tur 31 ÖLÇÜMÜ: `v3` (saf kod verisi) prose üretmiyor AMA Godot
#: derlemesi 1/5 (v2: 5/5). Sebep: saf kod verisi %61 daha az token
#: (937k vs 2,4M) ve **bağlam kaybı** (API açıklamaları atıldı).
#: Karar: **v2 birincil**, v3 yedek.
#: ⚠ tur 9 ÖLÇÜMÜ: 4 model karşılaştırıldı (40 soruluk benchmark):
#:   qwen_coder_myai (v5 verisi, 2,1M token) : %70  ← EN İYİ
#:   qwen_havuz2     (9,5M token)            : %65
#:   qwen_havuz      (9,5M token)            : %62
#:   qwen_karisim    (4,2M, %70 alan)        : %62
#:   qwen_alan       (9,5M+3M)               : %62
#: Ders: **alan verisi baskın** ve **yeterli tekrar** kritik.
#: ⚠ tur 10 ÖLÇÜMÜ — **bağımsız** benchmark (sızıntı temizlenmiş):
#:   Doygun 40s (ezber eğilimli)  vs  Bağımsız 38s (genelleme)
#:     qwen_coder_myai : %70  ·  %47   ← doygun testte iyi, genellemede zayıf
#:     qwen_havuz2     : %65  ·  %50   ← GENELLEMEDE EN İYİ, Godot %73
#:     qwen_alan       : %62  ·  %50
#:     qwen_karisim    : %62  ·  %47
#: Karar: **qwen_havuz2 birincil** (genelleme + Godot alt katmanı en iyi).
#: ⚠ tur 13 ÖLÇÜMÜ — 50 soruluk **bağımsız** benchmark:
#:   qwen_havuz2 : 26/50 (%52) · mantik 8/12 · godot 8/11  ← EN İYİ toplam
#:   qwen_tm     : 25/50 (%50) · mantik 8/12 · godot 7/11 · js 5/7
#:                 (tamamlama çiftleri: belge biçiminden +%9 mantık)
#:   qwen_mantik : 24/50 (%48) · mantik 7/12  (belge 6× — gerileme)
#:   qwen_coder  : 24/50 (%48) · godot 5/11
#: Karar: **qwen_havuz2 birincil**, qwen_tm ikincil.
MODELLER = ("models/qwen_havuz2", "models/qwen_tm",
            "models/qwen_coder_myai", "models/qwen_mantik",
            "models/smollm2_kod_v5")

#: Nesne türleri için istem şablonları (Godot GDScript)
#: ⚠ ÖLÇÜLEN SORUN: istemler **yorum satırı** içeriyordu ve model bu
#: yorumları **sürdürüyordu** (`# Create a mesh` ×N). İstemler sadeleştirildi:
#: yalnız **kod** verilir, açıklayıcı yorum verilmez.
GODOT_ISTEMLERI: Dict[str, str] = {
    "duvar": (
        "func labirent_duvari_uret(genislik: int, yukseklik: int) -> MeshInstance3D:\n"
        "    var st := SurfaceTool.new()\n"
        "    st.begin(Mesh.PRIMITIVE_TRIANGLES)\n"
        "    var kutu := BoxMesh.new()\n"
        "    kutu.size = Vector3(2.0, 1.6, 2.0)\n"
        "    var mi := MeshInstance3D.new()\n"
        # ⚠ ÖLÇÜLEN: dönüş tipi `MeshInstance3D` ama `return` yoktu;
        # Godot derlemesi "Not all code paths return a value" verdi.
        "    return mi\n"),
    "arazi": (
        "func arazi_uret(grid: int, boyut: float) -> MeshInstance3D:\n"
        "    var n := FastNoiseLite.new()\n"
        "    n.frequency = 0.035\n"
        "    n.fractal_octaves = 5\n"
        "    var st := SurfaceTool.new()\n"
        "    st.begin(Mesh.PRIMITIVE_TRIANGLES)\n"
        "    var mi := MeshInstance3D.new()\n"
        "    mi.mesh = st.commit()\n"
        "    return mi\n"),
    "malzeme": (
        "func malzeme_uret(renk: Color) -> StandardMaterial3D:\n"
        "    var m := StandardMaterial3D.new()\n"
        "    m.albedo_color = renk\n"
        "    m.roughness = 0.85\n"
        "    return m\n"),
    "karakter": (
        # ⚠ ÖLÇÜLEN: model `_physics_update()` uyduruyordu ve
        # `velocity` tanımsızdı. Şablon artık **tam** ve derlenebilir.
        "extends CharacterBody3D\n\n"
        "const HIZ := 5.0\n"
        "var _girdi := Vector2.ZERO\n\n"
        "func _physics_process(delta: float) -> void:\n"
        "    var yon := Vector3(_girdi.x, 0.0, _girdi.y)\n"
        "    velocity.x = yon.x * HIZ\n"
        "    velocity.z = yon.z * HIZ\n"
        "    if not is_on_floor():\n"
        "        velocity.y -= 9.8 * delta\n"
        "    move_and_slide()\n"),
    "agac": (
        "func agac_uret(tohum: int) -> Node3D:\n"
        "    var govde := MeshInstance3D.new()\n"
        "    var sil := CylinderMesh.new()\n"
        "    sil.radial_segments = 6\n"
        "    govde.mesh = sil\n"
        "    return govde\n"),
}

#: Blender Python istem şablonları
#: ⚠ ÖLÇÜLEN (tur 35): istemler `mesh = bpy.data.meshes.new(` gibi
#: **yarım** ifadelerdi → gerçek Blender'da `'(' was never closed`.
#: Şimdi istem öneki + **tamamlayıcı** tutulur: önek modele verilir,
#: tamamlayıcı şablon olarak kullanılır.
BLENDER_TAMAMLAYICI: Dict[str, str] = {
    "kup": "size=2.0)\n",
    "malzeme": "name='Malzeme')\n",
    "mesh": "name='Mesh')\n",
    "goruntu": "name='Doku', width=256, height=256)\n",
    "vertex": "0.0\n",
    "render": "'CYCLES'\n",
}

BLENDER_ISTEMLERI: Dict[str, str] = {
    "kup": ("import bpy\nbpy.ops.mesh.primitive_cube_add(",
            ("cube_add",)),
    "malzeme": ("import bpy\nmat = bpy.data.materials.new(", ("name", "'")),
    "mesh": ("import bpy\nmesh = bpy.data.meshes.new(", ("name", "'")),
    "goruntu": ("import bpy\nimg = bpy.data.images.new(",
                ("name", "width", "256")),
    "vertex": ("import bpy\nfor v in mesh.vertices:\n    v.co.z = ", ("0", "h")),
    "render": ("import bpy\nscene.render.engine = ",
               ("'CYCLES'", "'BLENDER", "engine")),
}


@dataclass
class UretimSonuc:
    """Kod üretim sonucu."""
    istem: str = ""
    kod: str = ""
    gecerli: bool = False
    neden: str = ""
    deneme: int = 0
    sure_s: float = 0.0
    kaynak: str = ""      # "model" | "sablon"

    def to_dict(self) -> Dict[str, Any]:
        return {"istem": self.istem[:60], "kod": self.kod[:400],
                "gecerli": self.gecerli, "neden": self.neden[:80],
                "deneme": self.deneme, "kaynak": self.kaynak,
                "sure_s": round(self.sure_s, 2)}


def tekrar_kes(kod: str, esik: int = 3) -> Tuple[str, str]:
    """
    **Tekrar döngüsünü keser.**

    ⚠ ÖLÇÜLEN SORUN: model sonsuz tekrar üretiyor:
        `# 3.3.3.3.3.1\n\n# 3.3.3.3.3.2\n...`
        `func malzeme_uret_2 / _3 / _4 ...`
        `scene.render.engine.scene_id = 3, 4, 5, 6 ...`
        `mesh.vertices.append((0,0,0)) × N`

    Kısıt doğrulayıcısı bunu **yakalamıyordu** çünkü her satır tek başına
    geçerli. Çözüm: aynı satır (veya sayı dışında aynı satır) `esik` kez
    tekrarlanırsa **kes**.

    Returns:
        (kesilmiş_kod, neden) — neden boşsa kesilmedi
    """
    satirlar = kod.splitlines()
    if len(satirlar) < esik:
        return kod, ""
    sayac: Dict[str, int] = {}
    for i, s in enumerate(satirlar):
        # Normalize: sayıları ve son ekleri kırp
        anahtar = re.sub(r"\d+", "#", s.strip())
        anahtar = re.sub(r"\s+", " ", anahtar)
        if len(anahtar) < 6:
            continue
        sayac[anahtar] = sayac.get(anahtar, 0) + 1
        if sayac[anahtar] >= esik:
            return "\n".join(satirlar[:i + 1]), \
                f"tekrar döngüsü (aynı satır {esik}×)"
    # Ardışık sayı artışı: `= 3`, `= 4`, `= 5` ...
    if len(satirlar) >= esik + 1:
        son = [re.sub(r"\d+", "#", x.strip()) for x in satirlar[-esik - 1:]]
        if len(set(son)) == 1:
            return "\n".join(satirlar[:-esik]), "sayaç döngüsü"
    return kod, ""


#: Kod dışı metin kalıpları — model bazen kod yerine açıklama üretiyor
KOD_DISI = (
    "The above code", "This code", "This function", "In this example",
    "Note that", "Explanation:", "Here is", "Here's", "As you can see",
    "This will", "You can", "This creates", "The following",
)


def kod_disi_kes(kod: str) -> Tuple[str, str]:
    """
    **Kod dışı metni keser.**

    ⚠ ÖLÇÜLEN SORUN: model bazen kod yerine İngilizce açıklama üretiyor:
        `'return m\n\nThe above code creates a '`
        `'...\n\n# Create a mesh\nmesh = bpy.context.object.new_...'`

    Çözüm: bilinen açıklama kalıplarını ara ve **oradan kes**.

    Returns:
        (temiz_kod, neden) — neden boşsa kesilmedi
    """
    for kalip in KOD_DISI:
        # ── Tam eşleşme ──
        i = kod.find(kalip)
        if i > 0:
            # Kalıptan önceki son satır sonuna kadar al
            kesim = kod.rfind("\n", 0, i)
            if kesim > 20:
                return kod[:kesim], f"kod dışı metin: {kalip!r}"
        # ── Kısmi eşleşme (token sınırında kesilmiş) ──
        #
        # ⚠ ÖLÇÜLEN SORUN: çıktı `max_new_tokens` sınırında kesilince
        # kalıbın **tamamı** görünmüyor:
        #     `'return m\\n\\nThe above co'`  ← "code" eksik
        # `find("The above code")` → -1 döner, kesme **çalışmaz**.
        # Çözüm: kalıbın **uzun bir önekini** de ara.
        for uzunluk in range(len(kalip) - 1, 9, -1):
            onek = kalip[:uzunluk]
            j = kod.find(onek)
            if j > 0:
                kesim = kod.rfind("\n", 0, j)
                if kesim > 20:
                    return kod[:kesim], \
                        f"kod dışı metin (kısmi): {onek!r}"
    return kod, ""


class KodUretici:
    """SmolLM2 ile kod üretir (kısıtlı)."""

    def __init__(self, *, model: Optional[str] = None,
                 maks_deneme: int = 3) -> None:
        self._model: Any = None
        self._tok: Any = None
        self._cihaz: str = "cpu"
        self._kisit: Any = None
        self._yasak: List[List[int]] = []
        self.maks_deneme = max(1, maks_deneme)
        # ── Son işleme (yazım düzeltme + uydurma filtresi) ──
        try:
            from tools.kod_son_isleme import SonIsleme

            self._son = SonIsleme()
        except Exception:
            self._son = None
        self._model_yolu = model or self._model_bul()
        self.istatistik = {"model": 0, "sablon": 0, "reddedilen": 0,
                           "tekrar_kesildi": 0, "kod_disi_kesildi": 0,
                           "yazim_duzeltildi": 0}

    @staticmethod
    def _model_bul() -> str:
        for m in MODELLER:
            if Path(m).exists():
                return m
        return ""

    # ─── Yükleme ───

    def yukle(self) -> bool:
        """Modeli ve kısıtları yükler."""
        if not self._model_yolu:
            print("  model bulunamadı")
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self._model_yolu)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_yolu)
            self._cihaz = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._cihaz)
            self._model.eval()
        except Exception as exc:
            print(f"  model yüklenemedi: {type(exc).__name__}")
            return False
        # Kısıtlar
        try:
            from tools.kod_kisitlari import KodKisitlari

            self._kisit = KodKisitlari()
            self._yasak = self._kisit.yasak_tokenler(self._tok)
        except Exception:
            self._kisit = None
        print(f"  model: {self._model_yolu} ({self._cihaz}) · "
              f"kısıt: {len(self._kisit.adlar) if self._kisit else 0} API · "
              f"yasak token: {len(self._yasak)}")
        return True

    # ─── Üretim ───

    def uret(self, istem: str, *, maks_token: int = 80,
             sicaklik: float = 0.0, tur: str = "blender") -> UretimSonuc:
        """
        İstemden kod üretir — kısıtlarla.

        ⚠ Akış:
            1. Üret (ilk deneme: greedy/deterministik)
            2. Kısıtları denetle (dil kilidi + API sözlüğü)
            3. Reddedilirse **yeniden dene** (artan sıcaklık)
            4. Maks deneme aşılırsa **geçersiz** döner (çağıran şablona düşer)
        """
        import torch

        s = UretimSonuc(istem=istem)
        t0 = time.perf_counter()
        if self._model is None:
            s.neden = "model yüklü değil"
            s.sure_s = time.perf_counter() - t0
            return s

        for deneme in range(self.maks_deneme):
            s.deneme = deneme + 1
            try:
                ids = self._tok(istem, return_tensors="pt")["input_ids"].to(
                    self._cihaz)
                uret: Dict[str, Any] = {
                    "max_new_tokens": maks_token,
                    "pad_token_id": self._tok.eos_token_id,
                    "do_sample": deneme > 0 or sicaklik > 0,
                }
                if deneme > 0 or sicaklik > 0:
                    uret["temperature"] = max(0.2, sicaklik + deneme * 0.25)
                    uret["top_p"] = 0.95
                if self._yasak:
                    uret["bad_words_ids"] = self._yasak
                # ⚠ ÖLÇÜLEN ÇÖZÜM: `no_repeat_ngram_size` tekrar döngüsünü
                # **üretim sırasında** engeller. Tekrar cezası zararlıydı
                # (57% → 43%) çünkü geçerli tekrarı da bastırıyordu;
                # n-gram yasağı yalnız **birebir** 4'lü tekrarı engeller.
                # ⚠ Ölçülen: 4'lü yasak sayaç desenini (`# 3.3.3\n# 4.0.3`)
                # yakalamıyor. 3'lü daha güçlü ama geçerli kalıpları
                # bozabilir. Dengeli değer: 3.
                uret["no_repeat_ngram_size"] = 3
                with torch.no_grad():
                    c = self._model.generate(ids, **uret)
                uretilen = self._tok.decode(
                    c[0][ids.shape[1]:], skip_special_tokens=True)
            except Exception as exc:
                s.neden = f"üretim hatası: {type(exc).__name__}"
                continue

            # ── Kod dışı metni kes ──
            #
            # ⚠ Ölçülen sorun: model bazen kod yerine açıklama üretiyor
            # ("The above code creates a ..."). Bu kod değil, gürültü.
            uretilen, disi_nedeni = kod_disi_kes(uretilen)
            if disi_nedeni:
                self.istatistik["kod_disi_kesildi"] = (
                    self.istatistik.get("kod_disi_kesildi", 0) + 1)

            # ── Tekrar döngüsünü kes ──
            #
            # ⚠ Ölçülen sorun: kısıt doğrulayıcısı tekrarı yakalamıyordu
            # (her satır tek başına geçerli). Ayrı tespit gerekir.
            uretilen, tekrar_nedeni = tekrar_kes(uretilen)

            # ── Son işleme: yazım düzeltme + uydurma filtresi ──
            #
            # ⚠ Ölçülen 2 hata: (1) `bpy_data` ← nokta yerine alt çizgi,
            # (2) uydurma API (`b3.b3_box_add_to_mesh`). İkisi de burada
            # düzeltilir/reddedilir.
            if self._son is not None:
                si = self._son.isle(uretilen, tur=tur)
                if si.duzeltme:
                    self.istatistik["yazim_duzeltildi"] = (
                        self.istatistik.get("yazim_duzeltildi", 0) + 1)
                if not si.gecerli:
                    self.istatistik["reddedilen"] += 1
                    s.neden = si.neden
                    continue
                uretilen = si.kod
            if tekrar_nedeni:
                self.istatistik["tekrar_kesildi"] = (
                    self.istatistik.get("tekrar_kesildi", 0) + 1)

            if self._kisit is not None:
                sonuc = self._kisit.dogrula(istem + uretilen)
                if not sonuc.gecerli:
                    self.istatistik["reddedilen"] += 1
                    s.neden = sonuc.neden
                    continue
            # ── Ek kalite kapısı: çok kısa mı? ──
            if len(uretilen.strip()) < 8:
                self.istatistik["reddedilen"] += 1
                s.neden = f"çıktı çok kısa ({len(uretilen.strip())} karakter)"
                continue
            s.kod = istem + uretilen
            s.gecerli = True
            s.neden = "geçerli"
            s.kaynak = "model"
            break

        s.sure_s = time.perf_counter() - t0
        if s.gecerli:
            self.istatistik["model"] += 1
        return s

    def uret_veya_sablon(self, anahtar: str, *,
                         tur: str = "godot") -> UretimSonuc:
        """
        Model üretir; **kısıt geçmezse şablona düşer**.

        ⚠ Bu, üretim hattının güvenlik ağıdır: kullanıcı **her zaman**
        çalışan kod alır.
        """
        kaynak = GODOT_ISTEMLERI if tur == "godot" else BLENDER_ISTEMLERI
        sablon = kaynak.get(anahtar)
        if sablon is None:
            return UretimSonuc(istem=anahtar, neden="bilinmeyen anahtar")

        if tur == "godot":
            istem = sablon
        else:
            istem = sablon[0]

        s = self.uret(istem, tur=tur)
        if s.gecerli:
            return s
        # Şablona düş
        if tur == "godot":
            s.kod = sablon
        else:
            # ⚠ Ölçülen: yarım istem + ilk kabul edilen değer =
            # derlenebilir kod. `BLENDER_TAMAMLAYICI` bunu sağlar.
            tamam = BLENDER_TAMAMLAYICI.get(anahtar, "")
            s.kod = sablon[0] + (tamam or sablon[1][0])
        s.gecerli = True
        s.kaynak = "sablon"
        s.neden = f"şablon (model: {s.neden[:40]})"
        self.istatistik["sablon"] += 1
        return s


def main() -> int:
    """Kod üreticiyi çalıştırır."""
    ap = argparse.ArgumentParser(description="Kod üretici (SmolLM2)")
    ap.add_argument("--istem", default="", help="Ham istem")
    ap.add_argument("--nesne", default="", help="Şablon anahtarı")
    ap.add_argument("--tur", default="godot", choices=("godot", "blender"))
    ap.add_argument("--test", action="store_true", help="İçerik testleri")
    ap.add_argument("--maks-token", type=int, default=80)
    args = ap.parse_args()

    print("=" * 78)
    print("  KOD ÜRETİCİ — SmolLM2 + kısıtlar + şablon yedeği")
    print("=" * 78)

    u = KodUretici()
    if not u.yukle():
        return 1

    sonuclar: List[Dict[str, Any]] = []

    if args.istem:
        s = u.uret(args.istem, maks_token=args.maks_token)
        print(f"\n  İSTEM: {args.istem[:60]!r}")
        print(f"  KOD  : {s.kod[:200]!r}")
        print(f"  geçerli={s.gecerli} kaynak={s.kaynak} neden={s.neden[:50]}")
        sonuclar.append(s.to_dict())

    if args.nesne:
        s = u.uret_veya_sablon(args.nesne, tur=args.tur)
        print(f"\n  NESNE: {args.nesne} ({args.tur})")
        print(f"  kaynak={s.kaynak} geçerli={s.gecerli}")
        print(f"  KOD:\n{s.kod[:400]}")
        sonuclar.append(s.to_dict())

    if args.test:
        print("\n  İÇERİK TESTLERİ:")
        for tur, anahtarlar in (("godot", list(GODOT_ISTEMLERI)),
                                ("blender", list(BLENDER_ISTEMLERI))):
            for a in anahtarlar:
                s = u.uret_veya_sablon(a, tur=tur)
                im = "M" if s.kaynak == "model" else "S"
                print(f"    [{im}] {tur}/{a:12} geçerli={s.gecerli} "
                      f"deneme={s.deneme} {s.neden[:36]}")
                sonuclar.append(s.to_dict())

        print(f"\n  İSTATİSTİK: model={u.istatistik['model']} "
              f"şablon={u.istatistik['sablon']} "
              f"reddedilen={u.istatistik['reddedilen']}")
        toplam = u.istatistik["model"] + u.istatistik["sablon"]
        if toplam:
            print(f"  Model başarı oranı: "
                  f"%{u.istatistik['model'] / toplam * 100:.0f}")

    Path("_kod_uretici.json").write_text(
        json.dumps({"sonuclar": sonuclar, "istatistik": u.istatistik},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
