"""
Derleme-Güdümlü Onarım Döngüsü
================================
⚠ ÖLÇÜLEN SORUN (tur 25-28)
---------------------------
Model Godot 4 kodu üretiyor ama **çok satırlı sözdizimi** hataları
kalıyor:
```gdscript
m.albedo_color = Color(1, 0, 0)      ← DOĞRU
m.normal_scale = 0.5                 ← DOĞRU
m.normal_map = Texture("res://...")  ← `Texture` Godot 4'te yok
```
```
z.wait_time = 1.0                    ← DOĞRU
z.connect("timeout", self, "_x")     ← Godot 3 imzası
```

**Gözlem:** Üyeler doğru, **sözdizimi** hatalı. Tek seferlik üretim
yetmiyor.

Çözüm: **Derleme-güdümlü onarım**
----------------------------------
```
1. ÜRET   : model + retrieval bağlamı ile kod yazar
2. DERLE  : gerçek Godot derleyicisi doğrular
3. HATA   : hata mesajı **bağlama geri beslenir**
4. TEKRAR : model hatayı görüp düzeltir (en çok N tur)
5. KAYIT  : derlenen ilk sürüm korunur
```

⚠️ Neden çalışabilir?
    Godot hata mesajları **açık**:
        `Parse Error: Function "set_material()" not found`
        `Parse Error: Identifier "NOISE_TYPE_TERRAIN" not declared`
    Model bu ipucunu kullanıp **hedefli** düzeltebilir.

Kullanım
--------
    python tools/derleme_onarim.py --calistir
    python tools/derleme_onarim.py --calistir --tur 4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

GODOT = Path(r"E:\AppsTools\Godot_v4.8-dev7_win64"
             r"\Godot_v4.8-dev7_win64_console.exe")

#: Test görevleri
GOREV: Sequence[Tuple[str, str]] = (
    ("FastNoiseLite noise gürültü arazi",
     "extends Node3D\n\nfunc _ready() -> void:\n"
     "    var n := FastNoiseLite.new()\n    n."),
    ("StandardMaterial3D material malzeme renk",
     "extends MeshInstance3D\n\nfunc _ready() -> void:\n"
     "    var m := StandardMaterial3D.new()\n    m."),
    ("Timer wait bekleme zamanlayıcı",
     "extends Node\n\nfunc _ready() -> void:\n"
     "    var z := Timer.new()\n    z."),
    ("Tween animasyon geçiş",
     "extends Node\n\nfunc _ready() -> void:\n"
     "    var t := create_tween()\n    t."),
)


@dataclass
class OnarimSonuc:
    """Tek görevin onarım sonucu."""
    gorev: str = ""
    tur: int = 0
    derlendi: bool = False
    kod: str = ""
    hatalar: List[str] = field(default_factory=list)
    satir: int = 0
    eksik_tanim: bool = False
    eksik_ad: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"gorev": self.gorev, "tur": self.tur,
                "derlendi": self.derlendi,
                "satir": self.satir,
                "eksik_tanim": self.eksik_tanim,
                "eksik_ad": self.eksik_ad,
                "hata_sayisi": len(self.hatalar),
                "hatalar": self.hatalar[:4]}


#: ─── Derleme ──────────────────────────────────────────────────────────

def derle(kod: str, zaman: int = 90) -> Tuple[bool, str]:
    """
    Godot'un **gerçek** ayrıştırıcısıyla doğrular.

    ⚠ Godot 3 sözdizimi Godot 4'te derlenmez → otomatik ayıklanır.
    """
    if not GODOT.exists():
        return (False, "godot yok")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "project.godot").write_text(
            '[application]\nconfig/name="t"\n'
            '[rendering]\nrenderer/rendering_method="mobile"\n',
            encoding="utf-8")
        h = d / "t.gd"
        h.write_text(kod, encoding="utf-8")
        try:
            p = subprocess.run(
                [str(GODOT), "--headless", "--path", str(d),
                 "--check-only", "--script", str(h)],
                capture_output=True, text=True, timeout=zaman,
                encoding="utf-8", errors="replace")
            out = (p.stdout or "") + (p.stderr or "")
            # ⚠ tur 33 ÖLÇÜMÜ: hata mesajı **iki satır**:
            #     `SCRIPT ERROR: Parse Error: <mesaj>`
            #     `at: GDScript::reload (res://t.gd:4)`   ← SATIR NO
            # İlk sürüm yalnız ilk satırı döndürüyordu → satır numarası
            # **hiç kullanılmadı** (`satir=0`). Çözüm: bir sonraki
            # `at:` satırını da ekle.
            ss = out.splitlines()
            for i, s in enumerate(ss):
                if "error" in s.lower() or "Parse Error" in s:
                    ek = ""
                    if i + 1 < len(ss) and "at:" in ss[i + 1]:
                        ek = " " + ss[i + 1].strip()
                    return (False, s.strip() + ek)
            return (True, "")
        except subprocess.TimeoutExpired:
            return (False, "zaman aşımı")
        except Exception as exc:
            return (False, f"{type(exc).__name__}")


def hata_ozu(hata: str) -> str:
    """Godot hata mesajından **özü** çıkarır."""
    m = re.search(r"Parse Error:\s*(.+?)(?:;|$)", hata)
    return m.group(1).strip() if m else hata[:120]


#: ─── Onarım ───────────────────────────────────────────────────────────

#: ⚠ Godot 4'te **geçersiz** olan kalıplar — ölçülerek doğrulandı.
#:
#: `connect(self, "_x")` → `Invalid argument for "connect"`
#: `t.` (sarkık)         → `Expected identifier after "."`
G4_YASAK = (
    "# ⚠ Godot 4 YASAKLARI (ölçüldü, derlenmez):",
    '#   connect(self, "_metot")   → connect(_metot)   [Godot 4 callable]',
    "#   t.                         → t.<üye> yaz, sarkık bırakma",
    "#   yield(...)                → await ...",
    "#   KinematicBody              → CharacterBody3D",
    "#   .SetX(...)                 → .x = ...",
    "#   ambientColor               → albedo_color",
    "#   Color3(...)                → Color(...)",
    "#   Texture(...)               → load(...)",
)


#: ⚠ tur 34 ÖLÇÜMÜ — iki kök neden:
#:   1. `m.emission_intensity =`  → **değer yok** (üretim kesildi)
#:   2. `t.` + satır sonu         → **tekrar döngüsü**
#:
#: Çözüm: isteme **gerçek değer örnekleri** ekle (pozitif, somut).
#: ⚠ Yasak listesi DEĞİL — tur 32'de onun gerilettiği ölçüldü.
DEGER_ORNEK = (
    "",
    "# Gerçek Godot 4 kullanım örnekleri (değerler dahil):",
    "#   m.albedo_color = Color(1, 0, 0)",
    "#   m.roughness = 0.5",
    "#   m.metallic = 0.0",
    "#   m.emission_enabled = true",
    "#   n.frequency = 0.03",
    "#   n.seed = 12345",
    "#   z.wait_time = 1.0",
    "#   z.one_shot = true",
    '#   t.tween_property(self, "position", Vector2(100, 0), 1.0)',
    '#   t.tween_property(self, "modulate", Color.RED, 0.5)',
    "#   t.set_loops(3)",
    "",
    "# ⚠ Her atamanın SAĞINDA değer olmalı; satırı yarım bırakma.",
)


def _baglam(sorgu: str, *, sinir: int = 20) -> str:
    """Godot 4 API bağlamı (tur 21-23)."""
    # ⚠ tur 32 ÖLÇÜMÜ: yasak listesi eklemek **GERİLETTİ**
    # (tur 31: 1/4 → tur 32: 0/4; FastNoiseLite bile kayboldu).
    # Çok kısıtlayıcı istem modeli kilitliyor → liste kaldırıldı.
    # ⚠ tur 34 ÖLÇÜMÜ: 15 satırlık **değer örnekleri listesi geriletti**
    # (tur 33: 2/4 → tur 34: 0/4). Bağlamı şişirmek modeli kilitliyor.
    # → Yalnız **tek satırlık** ipucu (asgari müdahale).
    try:
        from tools.godot4_baglam import baglam_uret

        return baglam_uret(sorgu, sinir=sinir)
    except Exception:
        return ""


def _onarim_istemi(kod: str, hata: str, tur: int) -> str:
    """
    Onarım istemi üretir.

    ⚠ tur 30 ÖLÇÜMÜ — iki yaklaşım denendi:

    **A) Yalnız gövde göster** (`_govde_ayikla`):
        `# Yalnızca gövde satırlarını düzelt:\n{govde}`
        → Model **ilk satırı kaybetti** (` wait_time = 1.0` düştü),
          önekteki `z.` **sarkık** kaldı:
              Parse Error: Expected identifier after "." for attribute access

    **B) Tam kodu göster, sonucu öneke göre kes** (SEÇİLEN):
        Model **tam bağlamı** görür → doğru düzeltir; biz çıktıdan
        yalnız önekten **sonrasını** alırız.

    Neden B? Model `z.` ile başlayan bir satırı **bitirebilmek** için
    bağlamı görmeli. A'da bağlam yoktu, ilk satırı attı.
    """
    oz = hata_ozu(hata)
    return (f"# Godot 4 derleme hatası: {oz}\n"
            f"# Aşağıdaki kodu düzelt. TAM kodu yaz "
            f"(extends/func satırları DAHİL):\n"
            f"{kod}\n"
            f"# Düzeltilmiş tam kod:\n")


def _govde_ayikla(kod: str, onek: str) -> str:
    """Koddan öneği ayıklar — yalnız **gövde** döner."""
    if kod.startswith(onek):
        return kod[len(onek):]
    i = kod.find("func ")
    if i >= 0:
        j = kod.find("\n", i)
        return kod[j + 1:] if j > 0 else ""
    return kod


#: ⚠ Onarım isteminin **son işaretçisi** (tur 35).
#:
#: Model bu işaretten **sonra** tam kodu yazar; önek istemin
#: **başında** olduğu için çıktıda görünmez.
TAM_KOD_ISARET = "# Düzeltilmiş TAM kod:"


def _tam_koddan_govde(uretilen: str, onek: str) -> str:
    """
    Modelin ürettiği **tam koddan** gövdeyi çıkarır.

    ⚠ Model tam kodu yazdığı için öneği **kendisi** içerir; biz
    öneği bulup sonrasını alırız. Bulunamazsa `func` satırından
    sonrasını alırız.
    """
    u = re.sub(r"```\w*\n?", "", uretilen).replace("```", "")
    # ⚠ tur 35: Onarım isteminde önek **çıktıda yoktur** (istemin
    # başında kalır). Model `# Düzeltilmiş TAM kod:` işaretinden
    # sonra tam kodu yazar → oradan kes.
    if TAM_KOD_ISARET in u:
        u = u.split(TAM_KOD_ISARET, 1)[1]
    # Önek birebir geçiyorsa kes
    if onek in u:
        return u.split(onek, 1)[1]
    # ⚠ tur 38-39 ÖLÇÜMÜ: "son eşleşme" düzeltmesi **GERİLETTİ**
    # (tur 37: 3/4 → tur 38: 2/4; Tween de kayboldu).
    # Eski davranış (ilk eşleşmeden kes) daha iyi sonuç veriyor.
    # Not: `z.` başka yerde de geçebiliyor ama pratikte kazanç sağlıyor.
    son = onek.rstrip().splitlines()[-1].strip() if onek.strip() else ""
    if son and son in u:
        return u.split(son, 1)[1]
    # `func` satırından sonrasını al
    m = re.search(r"^\s*func\s+\w+\s*\([^)]*\)[^:]*:\s*$", u,
                  re.MULTILINE)
    if m:
        return u[m.end():]
    return u


#: ⚠ Godot satır numarasını verir (tur 31 ÖLÇÜMÜ):
#:     `at: GDScript::reload (res://t.gd:4)`
SATIR_DESEN = re.compile(r"res://[^:)]+:(\d+)")


def hata_satiri(hata: str, kod: str) -> Tuple[int, str]:
    """
    Hata mesajından **(satır no, satır metni)** çıkarır.

    ⚠ Ölçüldü (tur 31): Godot her hatada satır numarası veriyor →
    tüm kodu değil, **yalnız hatalı satırı** düzelttirebiliriz.
    """
    m = SATIR_DESEN.search(hata)
    if not m:
        return (0, "")
    no = int(m.group(1))
    satirlar = kod.splitlines()
    if 1 <= no <= len(satirlar):
        # ⚠ Çevre satırları da ver (bağlam)
        bas = max(0, no - 3)
        son = min(len(satirlar), no + 2)
        return (no, "\n".join(f"{i + 1:3}| {satirlar[i]}"
                               for i in range(bas, son)))
    return (no, "")


def _hedefli_istem(kod: str, hata: str, satir: int, cevre: str,
                   *, eksik: bool = False) -> str:
    """Satır-hedefli onarım istemi."""
    oz = hata_ozu(hata)
    if eksik:
        # ⚠ `Identifier not declared` → modele **tanım eklet**
        return (f"# Godot 4 hatası: {oz}\n"
                f"# Eksik tanımı EKLE (fonksiyon/değişken).\n"
                f"# Mevcut kod:\n{kod}\n"
                f"# Düzeltilmiş TAM kod (eksik tanım DAHİL):\n")
    # ⚠ tur 34 ÖLÇÜMÜ: isteme eklenen **her** ipucu geriletti:
    #     15 satırlık değer listesi : 2/4 → 0/4
    #     tek satırlık ipucu        : 2/4 → 1/4
    # → İstem tur 33'teki **asgari** hâline döndürüldü.
    return (f"# Godot 4 hatası (satır {satir}): {oz}\n"
            f"# Hatalı bölge:\n{cevre}\n"
            f"# Yalnız bu bölgeyi düzelt. TAM kodu yaz:\n"
            f"{kod}\n"
            f"# Düzeltilmiş TAM kod:\n")


#: ⚠ Eksik tanım hataları — onarım döngüsü tek başına çözemez (tur 30)
EKSIK_DESEN = re.compile(
    r"not declared in the current scope|"
    r"Identifier \"[^\"]+\" not declared|"
    r"Function \"[^\"]+\" not found")


#: ⚠ tur 36 ÖLÇÜMÜ — üretilen kodda **üç normalleştirme** gerekli.
#:
#: Kaynak: model token düzeyinde analiz edildi.
#:
#: 1. **Sarkan nokta + girintili devam**:
#:        `t.` + `\n` + `        tween_property(...)`
#:      → Godot: `Expected identifier after "." for attribute access.`
#:      Godot girintiyi anlamlı sayar; satır birleştirilmeli.
#:
#: 2. **`f` son eki**: `100.0f` → `100.0`
#:      → Godot: `Invalid numeric notation.` (C/Python alışkanlığı)
#:
#: 3. **Girintili nokta devamı**: `\n        .start()`
#:      → Godot: `Expected statement, found "Indent" instead.`
#:
#: ⚠ Bu normalleştirmeler **yalnız sözdizimsel** — API adlarına dokunmaz
#: (tur 17/22/25/29/32/34 dersi: anlamsal dönüşüm geriletti).
NORMAL_1 = re.compile(r"\.\s*\n\s+([A-Za-z_])")
NORMAL_2 = re.compile(r"(\d+\.\d+)f\b")
NORMAL_3 = re.compile(r"\n\s+\.(\w+)")


#: ⚠ tur 37: **değersiz atama** kalıbı — `m.clearcoat =` gibi.
#:
#: Ölçüldü: model sonsuz özellik listesi üretiyor ve son satır
#: değersiz kalıyor → `Expected an expression after "="`.
YARIM_ATAMA = re.compile(r"^\s*[\w.\[\]]+\s*=\s*$")


def normalestir(kod: str) -> str:
    """Üretilen kodu **sözdizimsel** olarak normalleştirir."""
    # ⚠ tur 41 ÖLÇÜMÜ: model kod **çiti** üretiyor ve satır içi temizlik
    # yetersiz kalıyordu:
    #     5| '    ```gdscript'
    #     7| '    ```'
    #   → `Parse Error: Expected statement, found "`" instead.`
    # Çözüm: çit **satırlarını** (baştaki boşluk ne olursa olsun) at.
    kod = "\n".join(x for x in kod.splitlines()
                    if not x.strip().startswith("```"))
    k = NORMAL_1.sub(r".\1", kod)
    k = NORMAL_2.sub(r"\1", k)
    k = NORMAL_3.sub(r".\1", k)
    # ⚠ tur 37 ÖLÇÜMÜ: model bazen **dosyanın tamamını tekrar** ediyor:
    #     1| extends Node              ← 1. kopya
    #     ...
    #    13| extends Node              ← 2. kopya başlıyor
    #   → "Identifier ... not declared" (ilk kopyadaki tanım karışıyor)
    # Çözüm: **ikinci** `extends` satırından sonrasını at.
    ss = k.splitlines()
    for j in range(1, len(ss)):
        if re.match(r"^\s*extends\s+\w+", ss[j]):
            ss = ss[:j]
            break
    k = "\n".join(ss)

    # ⚠ tur 37: değersiz/yarım atama satırlarını at.
    # Ölçüldü: `m.emission_intensity =` → "Expected an expression".
    # Bu satırlar **silmek** derlemeyi sağlıyor (tur 36 dersi:
    # yalnız sözdizimsel, API adlarına dokunmaz).
    satirlar = [x for x in k.splitlines()
                if not YARIM_ATAMA.match(x)]
    # ⚠ tur 37 ÖLÇÜMÜ: model `func _x():` başlığını yazıp gövdeyi
    # yarım bırakabiliyor (dosya tekrarından kesilince):
    #     22| 'func _on_timeout()'
    #   → "Expected end of statement after expression"
    # Çözüm: gövdesi olmayan son `func` satırına **`    pass`** ekle.
    for j in range(len(satirlar) - 1, -1, -1):
        t = satirlar[j].strip()
        if not t:
            continue
        if t.startswith("func ") or re.match(r"^func\s+\w+", t):
            # Gövde var mı? (girintili satır)
            if j + 1 >= len(satirlar) or not satirlar[j + 1].startswith(
                    (" ", "\t")):
                # ⚠ Başlık `()` içermiyorsa tamamla
                if "(" not in t:
                    satirlar[j] = t + "() -> void:"
                elif not t.rstrip().endswith(":"):
                    satirlar[j] = t.rstrip() + ":"
                satirlar.insert(j + 1, "    pass")
            break
        break
    k = "\n".join(satirlar)
    if not k.endswith("\n"):
        k += "\n"
    return k


def onar(model_yolu: str, sorgu: str, onek: str, *, tur: int = 4,
         maks_token: int = 90, tohum: int = 42) -> OnarimSonuc:
    """
    Derleme-güdümlü onarım döngüsü.

    Her turda: üret → derle → hata varsa hata ile yeniden üret.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    s = OnarimSonuc(gorev=sorgu)
    tok = AutoTokenizer.from_pretrained(model_yolu)
    mdl = AutoModelForCausalLM.from_pretrained(model_yolu,
                                               dtype=torch.float32)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(cihaz)
    mdl.eval()

    baglam = _baglam(sorgu)
    kod = ""
    # ⚠ tur 31: hata durumu **döngüden önce** sıfırlanır ve her turda
    # güncellenir. Ölçülen hata: `s.eksik_tanim` yalnız `else` dalında
    # atanıyordu → `elif` bir tur **gecikmeli** okuyordu.
    s.eksik_tanim = False
    s.eksik_ad = ""
    for i in range(1, tur + 1):
        s.tur = i
        if i == 1:
            istem = baglam + "\n" + onek
        elif s.eksik_tanim and s.eksik_ad:
            # ⚠ tur 31: EKSİK TANIM — fonksiyon gövdesini ürettir.
            # Ölçüldü: model tam-kod isteminde `connect`i Godot 3'e
            # **geri çeviriyor**; ama `func ad() -> void:` önekiyle
            # doğru gövdeyi üretiyor.
            istem = (kod + f"\n# Eksik fonksiyonu TAM olarak yaz:\n"
                     f"func {s.eksik_ad}() -> void:\n")
        else:
            # ⚠ tur 31: SATIR HEDEFLİ onarım (Godot satır numarası verir)
            _h = s.hatalar[-1]
            _no, _cevre = hata_satiri(_h, kod)
            s.satir = _no
            istem = onek + "\n" + _hedefli_istem(
                kod, _h, _no, _cevre, eksik=s.eksik_tanim)
        ids = tok(istem, return_tensors="pt",
                  truncation=True, max_length=1024)["input_ids"].to(cihaz)
        with torch.no_grad():
            # ⚠ tur 30: DETERMİNİSTİK — tur 29'da do_sample gürültü ekledi
            c = mdl.generate(ids, max_new_tokens=maks_token,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
        u = tok.decode(c[0][ids.shape[1]:], skip_special_tokens=True)
        # ⚠ tur 30: model **tam kodu** yazar; gövdeyi öneğe göre ayıkla
        if s.eksik_tanim and s.eksik_ad and i > 1:
            # ⚠ Eksik tanım modunda çıktı **gövde**dir; koda ekle
            _yeni = "\n" + "\n".join(
                x for x in u.splitlines()
                if not re.match(r"^\s*(extends|func)\b", x))
            kod = kod + f"\n\nfunc {s.eksik_ad}() -> void:" + _yeni
        else:
            g = _tam_koddan_govde(u, onek)
            # ⚠ tur 41 ÖLÇÜMÜ: bu filtre **ZARARLI** — gövdedeki
            # `func _on_timeout() -> void:` tanımını da atıyordu →
            # `Identifier "_on_timeout" not declared`.
            # Dosya tekrarı zaten `normalestir()` içinde (ikinci
            # `extends`'ten kesme) çözülüyor; yani filtre gereksiz.
            kod = onek + g
        # ⚠ tur 36: ÜRETİLEN KODU NORMALLEŞTİR (yalnız sözdizimi).
        # Ölçüldü: 3 normalleştirme Tween'i derlenebilir yaptı.
        kod = normalestir(kod)

        # ⚠ tur 33: BOŞ GÖVDE denetimi — TEK DEĞİŞKEN olarak ölçülüyor.
        # Ölçüldü (Godot 4): model boş gövde üretince önekteki `t.`
        # **sarkık** kalıyor → `Expected identifier after "."`.
        _bos = (not g.strip()) and not (
            s.eksik_tanim and s.eksik_ad and i > 1)
        if _bos:
            ok, hata = (False, 'Parse Error: bos govde')
        else:
            ok, hata = derle(kod)
        if ok:
            s.derlendi = True
            s.kod = kod
            break
        s.hatalar.append(hata)
        # ⚠ tur 31: eksik-tanım durumu **her turda** güncellenir.
        # Ölçülen hata: yalnız `else` dalında atanıyordu → bir tur
        # gecikmeli okunuyordu ve başarılı tur kaydedilmiyordu.
        s.eksik_tanim = bool(EKSIK_DESEN.search(hata))
        _em = re.search(r'Identifier "([^"]+)" not declared'
                        r'|Function "([^"]+)" not found', hata)
        s.eksik_ad = ((_em.group(1) or _em.group(2)) if _em else "")
        s.satir = hata_satiri(hata, kod)[0]
    if not s.kod:
        s.kod = kod
    del mdl
    torch.cuda.empty_cache()
    return s


def main() -> int:
    """Onarım döngüsünü çalıştırır."""
    ap = argparse.ArgumentParser(description="Derleme-güdümlü onarım")
    ap.add_argument("--model", default="models/qwen_coder_15b")
    ap.add_argument("--tur", type=int, default=4)
    ap.add_argument("--calistir", action="store_true")
    ap.add_argument("--gorev", type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print("  DERLEME-GÜDÜMLÜ ONARIM DÖNGÜSÜ")
    print("=" * 78)
    print(f"  model: {args.model}")
    print(f"  tur  : {args.tur} · görev: {len(GOREV)}")
    print(f"  Godot: {'VAR' if GODOT.exists() else 'YOK'}")

    if not args.calistir:
        return 0

    gorevler = (GOREV[args.gorev - 1:args.gorev] if args.gorev
                else GOREV)
    sonuclar: List[OnarimSonuc] = []
    for sorgu, onek in gorevler:
        print(f"\n  [{sorgu}]")
        s = onar(args.model, sorgu, onek, tur=args.tur)
        sonuclar.append(s)
        print(f"    tur {s.tur} · derlendi={s.derlendi}")
        for h in s.hatalar[:3]:
            print(f"      ✗ {hata_ozu(h)[:66]}")
        if s.derlendi:
            print(f"      ✓ DERLENDİ (tur {s.tur})")
            for ln in s.kod.splitlines()[:8]:
                print(f"        {ln}")

    n = len(sonuclar)
    d = sum(1 for x in sonuclar if x.derlendi)
    ort = sum(x.tur for x in sonuclar) / max(1, n)
    print(f"\n  {'=' * 74}")
    print(f"  DERLENEN: {d}/{n}  ·  ortalama tur: {ort:.1f}")
    print(f"  {'=' * 74}")
    Path("_derleme_onarim.json").write_text(
        json.dumps({"sonuclar": [s.to_dict() for s in sonuclar],
                    "derlenen": d, "toplam": n},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())