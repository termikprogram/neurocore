"""
Godot 4 API'sinden Doğrudan Eğitim Verisi Üretir
==================================================
⚠ ÖLÇÜLEN DURUM (tur 15-18)
---------------------------
```
qwen_havuz2 (temel)      : derlenen 0/4 · Godot3 kirli 2/4
qwen_temiz2 (temiz veri) : derlenen 1/4 · Godot3 kirli 1/4
qwen_t18 (uzun eğitim)   : benchmark 24/50 (%48)
```

Model hâlâ **Godot 4 API'lerini bilmiyor** (`ambientColor`, `Color3`,
`set_`) — çünkü eğitim verisi ağırlıklı **Godot 3**.

Çözüm: **Dokümandan doğrudan Godot 4 verisi üret**
---------------------------------------------------
`dokuman_godot.jsonl` içinde **2.008 imza** var (gerçek Godot 4
dokümanından çıkarıldı). Bu imzalar **kesin doğru**.

Her imzadan **kod tamamlama örneği** üret:
```
İMZA   : FastNoiseLite.frequency : float
ÖRNEK  : "var n := FastNoiseLite.new()\nn.frequency = " → "0.01"
```

⚠ Neden bu veri değerli?
    * **%100 Godot 4** — Godot 3 kalıntısı **yok**
    * **Gerçek API adları** — `albedo_color`, `noise_scale`, `seed`
    * **Sınıf + üye** ilişkisi doğru
    * **Sınırsız çoğaltılabilir** (imza → 3 varyant)

Kullanım
--------
    python tools/godot4_veri.py --istatistik
    python tools/godot4_veri.py --uret
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Kaynak — Godot dokümanından çıkarılmış imzalar
KAYNAK = Path(".neurocore/dokuman_godot.jsonl")

#: Çıktı
CIKTI = Path(".neurocore/godot4_api.jsonl")

#: ⚠ Türe göre örnek değer (Godot 4 tipleri)
TIP_DEGER: Dict[str, str] = {
    "int": "1", "float": "0.5", "bool": "true", "String": '"metin"',
    "Vector2": "Vector2(1, 2)", "Vector3": "Vector3(1, 2, 3)",
    "Vector2i": "Vector2i(1, 2)", "Vector3i": "Vector3i(1, 2, 3)",
    "Color": "Color(1, 0, 0)", "NodePath": '^"../Dugum"',
    "Array": "[]", "Dictionary": "{}", "PackedByteArray": "PackedByteArray()",
    "PackedInt32Array": "PackedInt32Array()",
    "PackedFloat32Array": "PackedFloat32Array()",
    "PackedStringArray": "PackedStringArray()",
    "PackedVector2Array": "PackedVector2Array()",
    "PackedVector3Array": "PackedVector3Array()",
    "Transform2D": "Transform2D()", "Transform3D": "Transform3D()",
    "Basis": "Basis()", "Quaternion": "Quaternion()",
    "Rect2": "Rect2()", "RID": "RID()", "Callable": "Callable()",
    "Signal": "Signal()", "Object": "null", "void": "",
}

#: ⚠ Bu sınıflar Godot 4'te **var**; dönüşümde kullanılmaz
GODOT4_SINIF = re.compile(
    r"^(Node|Node2D|Node3D|CharacterBody2D|CharacterBody3D|Area2D|Area3D|"
    r"RigidBody2D|RigidBody3D|StaticBody2D|StaticBody3D|MeshInstance3D|"
    r"Camera2D|Camera3D|Sprite2D|Sprite3D|Label|Button|Control|"
    r"StandardMaterial3D|FastNoiseLite|Tween|Timer|RayCast3D|"
    r"CollisionShape3D|AnimationPlayer|AudioStreamPlayer|"
    r"MultiMeshInstance3D|GPUParticles3D|WorldEnvironment|"
    r"Resource|SceneTree|Input|Engine|Time|FileAccess|DirAccess|"
    r"Vector2|Vector3|Color|Transform3D|Basis|Rect2|Dictionary|Array)$")


#: ⚠ Godot sınıf adı çevirici (tur 18)
#:
#: Ölçülen hata: `.title()` kullanınca `AABB` → `Aabb`,
#: `AStar2D` → `Astar2D` oldu. Godot **PascalCase + kısaltma** kullanır.
#:
#: Kurallar:
#:     * `2d`/`3d` son eki → `2D`/`3D`
#:     * Bilinen kısaltmalar korunur (`aabb`, `aes`, `csg`, `gpu`, …)
#:     * Diğer parçalar baş harfi büyük
_BILINEN: Dict[str, str] = {
    "aabb": "AABB", "aes": "AES", "csg": "CSG", "gpu": "GPU",
    "cpu": "CPU", "hdr": "HDR", "glsl": "GLSL", "ip": "IP",
    "os": "OS", "ui": "UI", "url": "URL", "rpc": "RPC",
    "sdfgi": "SDFGI", "ssao": "SSAO", "ssil": "SSIL", "sse": "SSE",
    "astar": "AStar", "http": "HTTP", "tcp": "TCP", "udp": "UDP",
    "xml": "XML", "json": "JSON", "svg": "SVG", "png": "PNG",
    "jpg": "JPG", "ktx": "KTX", "dds": "DDS", "exr": "EXR",
    "hsv": "HSV", "rgb": "RGB", "rgba": "RGBA", "api": "API",
    "id": "ID", "vr": "VR", "xr": "XR", "ar": "AR", "ik": "IK",
    "nav": "Nav", "dns": "DNS", "tls": "TLS", "ssl": "SSL",
}


def sinif_adi(ham: str) -> str:
    """
    URL parçasından **doğru** Godot sınıf adını üretir.

    `aabb` → `AABB` · `astar2d` → `AStar2D` · `animatedsprite2d`
    → `AnimatedSprite2D` · `aescontext` → `AESContext`
    """
    if not ham:
        return ""
    s = ham.strip().lower()
    # `2d` / `3d` son eki
    son = ""
    if s.endswith("2d"):
        s, son = s[:-2], "2D"
    elif s.endswith("3d"):
        s, son = s[:-3], "3D"
    # Bilinen kısaltma tam eşleşme
    if s in _BILINEN:
        return _BILINEN[s] + son
    # Bilinen kısaltma + kalan (aescontext → AESContext)
    for kisa, uzun in sorted(_BILINEN.items(), key=lambda x: -len(x[0])):
        if s.startswith(kisa) and len(s) > len(kisa):
            kalan = s[len(kisa):]
            return uzun + kalan[0].upper() + kalan[1:] + son
    # Alt çizgi varsa parçala
    if "_" in ham:
        return "".join(x[:1].upper() + x[1:] for x in ham.split("_")) + son
    return s[:1].upper() + s[1:] + son


#: ⚠ DOĞRU sınıf adı kaynağı (tur 18): doküman sayfasının **başlığı**.
#:
#: `**Kaynak:** FastNoiseLite &mdash; Godot Engine (stable) ...`
#:                ^^^^^^^^^^^^^ tam sınıf adı
#:
#: Ölçüldü: URL slug'ından tahmin yalnız **6/13** doğruydu
#: (`node3d` → `Nod3D`). Başlık ise **kesin**.
KAYNAK_SINIF = re.compile(
    r"\*\*Kaynak:\*\*\s*([^&\n|]+?)\s*(?:&mdash;|—|\||$)")

#: ⚠ Niteliksiz **özellik** imzaları: `PI = 3.14` · `seed : int`
OZELLIK2 = re.compile(r"^([A-Za-z_]\w*)\s*(?::|=)\s*(.+?)\s*$")

#: ⚠ Niteliksiz **tipli özellik**: `frequency : float`
OZELLIK3 = re.compile(r"^([A-Za-z_]\w*)\s*:\s*([A-Za-z_][\w.]*)\s*$")

#: ⚠ Niteliksiz **metot** imzaları: `char(code: int) -> String`
METOT3 = re.compile(r"^([A-Za-z_]\w*)\(([^)]*)\)\s*(?:->\s*(\w+))?")

@dataclass
class UretimIstatistik:
    """Üretim istatistiği."""
    kaynak_kayit: int = 0
    imza: int = 0
    ozellik: int = 0
    metot: int = 0
    uretilen: int = 0
    karakter: int = 0
    sinif_sayisi: int = 0
    ornek_sinif: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


#: ⚠ ÖLÇÜLEN YAPI (tur 18): `dokuman_godot.jsonl` kayıtları
#: `{"dil","kategori","metin","surum","tur","url"}` biçiminde —
#: imza **metin içinde** `**İmza:** \`...\`` olarak duruyor, ayrı
#: bir `imzalar` alanı **yok**.
IMZA_ALANI = re.compile(r"\*\*İmza:\*\*\s*`([^`]+)`")

#: İmza desenleri — `sinif.uye` biçiminde **nitelikli** olanlar
#:   "FastNoiseLite.frequency : float"    → özellik
#:   "Vector3.normalized() -> Vector3"    → metot
OZELLIK = re.compile(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*:\s*(\w+)")
METOT2 = re.compile(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)\(([^)]*)\)"
                    r"\s*(?:->\s*(\w+))?")
METOT = re.compile(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(")

#: ⚠ URL'den sınıf adı çıkarma (`class_fastnoiselite.html`)
URL_SINIF = re.compile(r"class[_-]([a-z0-9_]+)\.html")


def _ornekler(sinif: str, uye: str, tip: str, metot_mu: bool,
              parametreler: str = "") -> List[str]:
    """Bir üye için kod tamamlama örnekleri üretir."""
    ornekler: List[str] = []
    if metot_mu:
        # Metot çağrısı
        args = ""
        if parametreler.strip():
            parcalar = []
            for p in parametreler.split(","):
                p = p.strip()
                if not p or ":" not in p:
                    continue
                ptip = p.split(":")[-1].strip().split("=")[0].strip()
                parcalar.append(TIP_DEGER.get(ptip, "null"))
            args = ", ".join(parcalar)
        d = TIP_DEGER.get(tip, "")
        if tip in ("void", "") or not tip:
            onek = (f"extends Node3D\n\nfunc _hazirla() -> void:\n"
                    f"    var d := {sinif}.new()\n    d.{uye}(")
            devam = f"{args})\n"
        else:
            onek = (f"extends Node3D\n\nfunc _hazirla() -> {tip}:\n"
                    f"    var d := {sinif}.new()\n    return d.{uye}(")
            devam = f"{args})\n"
        ornekler.append(onek + devam)
    else:
        # Özellik atama / okuma
        deger = TIP_DEGER.get(tip, "null")
        if deger:
            ornekler.append(
                f"extends Node3D\n\nfunc _hazirla() -> void:\n"
                f"    var d := {sinif}.new()\n    d.{uye} = {deger}\n")
        ornekler.append(
            f"extends Node3D\n\nfunc _hazirla() -> {tip or 'void'}:\n"
            f"    var d := {sinif}.new()\n    return d.{uye}\n")
    return ornekler


def _kod_blogu(metin: str) -> List[str]:
    """Bir kayıttaki ```gdscript bloklarını çıkarır."""
    return re.findall(r"```gdscript\s*\n(.*?)```", metin, re.DOTALL)


def uret() -> UretimIstatistik:
    """İmzalardan eğitim verisi üretir."""
    s = UretimIstatistik()
    if not KAYNAK.exists():
        print(f"  kaynak yok: {KAYNAK}")
        return s

    siniflar: Dict[str, List[Tuple[str, str, bool, str]]] = {}
    with KAYNAK.open(encoding="utf-8") as f:
        for satir in f:
            try:
                d = json.loads(satir)
            except Exception:
                continue
            s.kaynak_kayit += 1
            metin = d.get("metin", "")
            # ⚠ Sınıf adı URL'den veya başlıktan
            url = d.get("url", "") or ""
            # ⚠ Öncelik: kaynak başlığı (kesin) > URL slug (tahmin)
            mb = KAYNAK_SINIF.search(metin)
            if mb:
                varsayilan_sinif = mb.group(1).strip()
            else:
                mu = URL_SINIF.search(url)
                varsayilan_sinif = sinif_adi(mu.group(1)) if mu else ""
            for ham in IMZA_ALANI.findall(metin):
                imza = ham.strip()
                if len(imza) < 4:
                    continue
                m = OZELLIK.match(imza)
                if m:
                    sinif, uye, tip = m.group(1), m.group(2), m.group(3)
                    siniflar.setdefault(sinif, []).append(
                        (uye, tip, False, ""))
                    s.ozellik += 1
                    s.imza += 1
                    continue
                m = METOT2.match(imza)
                if m:
                    sinif, uye = m.group(1), m.group(2)
                    pars = m.group(3) or ""
                    tip = m.group(4) or ""
                    siniflar.setdefault(sinif, []).append(
                        (uye, tip, True, pars))
                    s.metot += 1
                    s.imza += 1
                    continue
                if not varsayilan_sinif:
                    continue
                # ⚠ Niteliksiz **metot**: `char(code: int) -> String`
                m = METOT3.match(imza)
                if m and "(" in imza:
                    siniflar.setdefault(varsayilan_sinif, []).append(
                        (m.group(1), m.group(3) or "", True, m.group(2) or ""))
                    s.metot += 1
                    s.imza += 1
                    continue
                # ⚠ Niteliksiz **tipli özellik**: `frequency : float`
                m = OZELLIK3.match(imza)
                if m:
                    siniflar.setdefault(varsayilan_sinif, []).append(
                        (m.group(1), m.group(2), False, ""))
                    s.ozellik += 1
                    s.imza += 1
                    continue
                # ⚠ Niteliksiz **değerli özellik**: `PI = 3.14`
                m = OZELLIK2.match(imza)
                if m:
                    ad, deger = m.group(1), m.group(2).strip()
                    # ⚠ `= deger` biçiminde tip bilinmiyor; değerden çıkar
                    tip = ""
                    if deger and deger[0].isdigit():
                        tip = "float" if "." in deger else "int"
                    elif deger.startswith('"'):
                        tip = "String"
                    elif deger in ("true", "false"):
                        tip = "bool"
                    elif re.match(r"^[A-Z]\w*$", deger):
                        tip = deger
                    siniflar.setdefault(varsayilan_sinif, []).append(
                        (ad, tip, False, ""))
                    s.ozellik += 1
                    s.imza += 1

    s.sinif_sayisi = len(siniflar)
    s.ornek_sinif = sorted(siniflar)[:12]

    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    gorulen: set = set()
    with CIKTI.open("w", encoding="utf-8") as f:
        for sinif in sorted(siniflar):
            uyeler = siniflar[sinif]
            for uye, tip, metot_mu, pars in uyeler:
                for kod in _ornekler(sinif, uye, tip, metot_mu, pars):
                    anahtar = kod[-60:]
                    if anahtar in gorulen:
                        continue
                    gorulen.add(anahtar)
                    kayit = (f"# Godot 4 API — {sinif}.{uye}\n\n"
                             f"**Sınıf:** {sinif}\n"
                             f"**Üye:** {uye}\n"
                             f"**Tür:** {'metot' if metot_mu else 'özellik'}"
                             f"{' → ' + tip if tip else ''}\n\n"
                             f"```gdscript\n{kod}```\n")
                    f.write(json.dumps(
                        {"metin": kayit, "kategori": "godot4_api",
                         "sinif": sinif, "uye": uye}, ensure_ascii=False)
                        + "\n")
                    s.uretilen += 1
                    s.karakter += len(kayit)
    return s


def main() -> int:
    """Üretimi çalıştırır."""
    ap = argparse.ArgumentParser(description="Godot 4 API verisi")
    ap.add_argument("--istatistik", action="store_true")
    ap.add_argument("--uret", action="store_true")
    ap.add_argument("--ornek", type=int, default=8)
    args = ap.parse_args()

    print("=" * 78)
    print("  GODOT 4 API → EĞİTİM VERİSİ")
    print("=" * 78)
    print(f"  kaynak: {KAYNAK}")

    s = uret()
    print(f"\n  kaynak kayıt : {s.kaynak_kayit:,}")
    print(f"  imza         : {s.imza:,} "
          f"(özellik {s.ozellik:,} · metot {s.metot:,})")
    print(f"  sınıf        : {s.sinif_sayisi:,}")
    print(f"  ÜRETİLEN     : {s.uretilen:,} örnek")
    print(f"  karakter     : {s.karakter:,}")
    print(f"  token        : ~{s.karakter // 3:,}")
    print(f"  dosya        : {CIKTI}")
    print(f"\n  örnek sınıflar: {', '.join(s.ornek_sinif)}")

    # İlk örnekleri göster
    if CIKTI.exists():
        print("\n  ÜRETİLEN ÖRNEKLER:")
        with CIKTI.open(encoding="utf-8") as f:
            for i, satir in enumerate(f):
                if i >= args.ornek:
                    break
                d = json.loads(satir)
                kod = d["metin"].split("```gdscript\n")[-1].rstrip("`\n")
                print(f"    [{d['sinif']}.{d['uye']}]")
                for ln in kod.splitlines()[:4]:
                    print(f"      {ln}")

    Path("_godot4_veri.json").write_text(
        json.dumps(s.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
