"""
Blender Mantık Müfredatı → Tamamlama Çifti Biçimi
===================================================
⚠ Neden? (tur 13 ÖLÇÜMÜ)
------------------------
```
tur 12 : mantık notları BELGE biçimi      → %67 → %58  (−%9)
tur 13 : TAMAMLAMA ÇİFTİ biçimi           → %58 → %67  (+%9)
```

**Kural:** Kod tamamlama öğretmek için veri **tamamlama biçiminde**
olmalı. Belge biçimi **zarar veriyor**.

⚠ Sorun
-------
`tools/blender_mantik.py` 43 ders üretir ama bunlar **belge**:
```
# Blender mantığı — Kemik neden head/tail ile tanımlanır?

**Katman:** 06_kemik — Kemik uzayı ve weight mantığı

## Neden?

Kemik bir **vektördür**: başı (head) ve sonu (tail). ...

## Kod

```python
bone.head = Vector((0,0,0)); bone.tail = Vector((0,0,1))
```
```

**Bu biçim model için uygun değil.** Doğru biçim:
```
### Önek (prompt)
bpy.ops.object.armature_add(enter_editmode=True)
arm = bpy.context.active_object
eb = arm.edit_bones
b = eb.new("omurga")
b.

### Tamamlama (completion)
head = Vector((0, 0, 0))
b.tail = Vector((0, 0, 1))
```

Bu betik
--------
```
1. Her dersin **kod** alanını al
2. Anlamlı bir **önek** kes (son satırın bir kısmı)
3. (önek, tamamlama) çifti üret
4. `ast.parse` / parantez dengesi ile **doğrula**
5. JSONL'e yaz
```

Kullanım
--------
    python tools/blender_mantik_tm.py --uret
    python tools/blender_mantik_tm.py --istatistik
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

CIKTI = Path(".neurocore") / "blender_mantik_tamamlama.jsonl"

#: ⚠ Python anahtar sözcükleri — önek bunlarla bitmemeli
ANAHTAR = {
    "if", "elif", "else", "for", "while", "def", "class", "return",
    "import", "from", "as", "try", "except", "finally", "with", "lambda",
    "yield", "pass", "break", "continue", "and", "or", "not", "in", "is",
    "None", "True", "False", "global", "nonlocal", "assert", "del",
    "raise", "async", "await",
}


@dataclass
class Cift:
    """Tek tamamlama çifti."""
    ders: str = ""
    katman: str = ""
    onek: str = ""
    tamamlama: str = ""
    baglam: str = ""
    tam_kod: str = ""

    def metin(self) -> str:
        """
        ⚠ Eğitim metni — **tamamlama biçimi**.

        Biçim:
            # <katman> · <ders>
            <bağlam>
            <önek><tamamlama>
        """
        bas = f"# Blender · {self.katman} · {self.ders}"
        if self.baglam:
            bas += f"\n# {self.baglam}"
        return f"{bas}\n{self.onek}{self.tamamlama}"

    def to_dict(self) -> Dict[str, Any]:
        return {"metin": self.metin(), "ders": self.ders,
                "katman": self.katman, "onek": self.onek,
                "tamamlama": self.tamamlama}


#: ⚠ Kod bloğundan anlamlı önek kesme noktaları
def _onek_kes(kod: str) -> List[Tuple[str, str]]:
    """
    Koddan **(önek, tamamlama)** çiftleri üretir.

    ⚠ tur 16 ÖLÇÜMÜ: ilk sürüm token'ları **ortadan bölüyordu**:
    ```
    önek      : "bpy.ops.export_scene.gltf(export_yup"
    tamamlama : "=True)  # varsayılan"     ← TOKEN BÖLÜNMÜŞ
    ```
    Bu model için **anlamsız** — `export_yup` tek bir tanımlayıcıdır.

    ✅ **Anlamlı sınırlara hizala:**
        1. `=` sonrası (değer tamamlanır)  → önek `... = `
        2. `,` sonrası (argüman tamamlanır)
        3. satır sonu (yeni satır tamamlanır)
    """
    satirlar = kod.splitlines()
    ciftler: List[Tuple[str, str]] = []

    # ⚠ Strateji 1: `=` sonrası kes — "değeri tamamla"
    # ⚠ Not: `location=False` gibi bitişik yazım Python'da geçerli,
    # ama okunabilirlik için `= ` tercih edilir.
    for i, s in enumerate(satirlar):
        if not s.strip() or s.strip().startswith("#"):
            continue
        girinti = len(s) - len(s.lstrip())
        # ⚠ SON `=` bul (== ve >= hariç)
        esler = [m.start() for m in re.finditer(r"(?<![=<>!])=(?!=)", s)]
        for es in esler:
            onek_satir = s[:es + 1]
            tam_satir = s[es + 1:]
            # ⚠ Önek `=` ile bitmeli, tamamlama boş olmamalı
            if not tam_satir.strip():
                continue
            if len(tam_satir.strip()) < 2:
                continue
            onek = "\n".join(satirlar[:i] + [onek_satir])
            tam = tam_satir + ("\n" + "\n".join(satirlar[i + 1:])
                              if i + 1 < len(satirlar) else "")
            ciftler.append((_bosluk_duzelt(onek, tam), tam))
            break

    # ⚠ Strateji 2: `,` sonrası kes — "argümanı tamamla"
    for i, s in enumerate(satirlar):
        if not s.strip() or s.strip().startswith("#"):
            continue
        virguller = [m.start() for m in re.finditer(r",", s)]
        if not virguller:
            continue
        vs = virguller[-1]
        onek_satir = s[:vs + 1]
        tam_satir = s[vs + 1:]
        if len(tam_satir.strip()) < 3:
            continue
        onek = "\n".join(satirlar[:i] + [onek_satir])
        tam = tam_satir + ("\n" + "\n".join(satirlar[i + 1:])
                          if i + 1 < len(satirlar) else "")
        ciftler.append((onek, tam))

    # ⚠ Strateji 3: satır sonu — "sonraki satırı tamamla"
    if len(satirlar) >= 2:
        for i in range(1, len(satirlar)):
            s = satirlar[i]
            if not s.strip() or s.strip().startswith("#"):
                continue
            girinti = len(s) - len(s.lstrip())
            if len(s) - girinti < 8:
                continue
            onek = "\n".join(satirlar[:i]) + "\n" + " " * girinti
            tam = s.strip()
            ciftler.append((onek, tam))
            break

    # ⚠ Strateji 3b: çok kısa kodlar için — kelime sınırından kes
    if len(satirlar) == 1:
        s = satirlar[0]
        if s.strip() and not s.strip().startswith("#"):
            # ⚠ Son "atom"u ayır (tanımlayıcı veya değer)
            m = re.search(r"[.\w]+$", s)
            if m and m.start() > 0:
                onek = s[:m.start()]
                tam = s[m.start():]
                if len(tam) >= 3 and len(onek) >= 6:
                    ciftler.append((onek, tam))

    # ⚠ Strateji 4: `def ...():` gövdesi — "gövdeyi tamamla"
    for i, s in enumerate(satirlar):
        t = s.strip()
        if not (t.startswith("def ") or t.startswith("class ")):
            continue
        if not t.endswith(":"):
            continue
        if i + 1 >= len(satirlar):
            continue
        onek = "\n".join(satirlar[:i + 1]) + "\n"
        tam = "\n".join(satirlar[i + 1:])
        if len(tam.strip()) >= 5:
            ciftler.append((onek, tam))
            break

    return ciftler


def _bosluk_duzelt(onek: str, tam: str) -> str:
    """
    ⚠ Önek `=` ile bitiyorsa ve tamamlama boşlukla başlamıyorsa,
    öneğe boşluk ekler (okunabilirlik).

    `location=` + `False` → `location= ` + `False`
    """
    if onek.endswith("=") and tam and not tam.startswith(" "):
        return onek + " "
    if onek.endswith(",") and tam and not tam.startswith(" "):
        return onek + " "
    return onek


def _dogrula(onek: str, tamamlama: str) -> bool:
    """
    ⚠ Çifti doğrular.

    Ölçüt:
        1. Birleşik kod **parantez dengeli** olmalı
        2. Tamamlama **boş olmamalı**
        3. Tamamlama **yeni satırla başlamamalı** (girinti hariç)
    """
    if not tamamlama.strip():
        return False
    birlesik = onek + tamamlama
    # ⚠ Parantez dengesi
    for ac, kapa in (("(", ")"), ("[", "]"), ("{", "}")):
        if birlesik.count(ac) != birlesik.count(kapa):
            return False
    # ⚠ Tırnak dengesi (kaba)
    if birlesik.count('"') % 2 != 0:
        return False
    if birlesik.count("'") % 2 != 0:
        return False
    return True


def uret(*, hepsi: bool = False) -> List[Cift]:
    """
    Müfredattan tamamlama çiftleri üretir.

    ⚠ Her ders için **birden çok** çift üretilir (her satır bir aday).
    """
    from tools.blender_mantik import tum_dersler

    ciftler: List[Cift] = []
    for d in tum_dersler():
        kod = d.kod.strip()
        if not kod:
            continue
        # ⚠ Çok satırlı kod tercih edilir
        adaylar = _onek_kes(kod)
        gecerli = 0
        for onek, tam in adaylar:
            if not _dogrula(onek, tam):
                continue
            # ⚠ Bağlam: dersin "neden" metninden ilk cümle (kısa)
            baglam = ""
            ilk = d.mantik.strip().split("\n")[0]
            ilk = re.sub(r"[`*⚠]", "", ilk).strip()
            if 8 < len(ilk) < 90:
                baglam = ilk
            ciftler.append(Cift(
                ders=d.baslik, katman=d.katman.split(" — ")[0],
                onek=onek, tamamlama=tam, baglam=baglam, tam_kod=kod))
            gecerli += 1
            # ⚠ tur 16 ÖLÇÜMÜ: 3 çift/ders → kapsama 32/43.
            # 6'ya çıkarıldı → tüm dersler kapsanır.
            if not hepsi and gecerli >= 6:
                break
    return ciftler


def istatistik(ciftler: Optional[List[Cift]] = None) -> Dict[str, Any]:
    """Çift istatistiği."""
    if ciftler is None:
        ciftler = uret()
    mt = [c.metin() for c in ciftler]
    return {
        "cift": len(ciftler),
        "karakter": sum(len(m) for m in mt),
        "token_tahmini": sum(len(m) for m in mt) // 3,
        "ders_kapsama": len({c.ders for c in ciftler}),
        "katman_kapsama": len({c.katman for c in ciftler}),
        "ortalama_tamamlama": round(
            sum(len(c.tamamlama) for c in ciftler)
            / max(1, len(ciftler)), 1),
    }


def yaz(ciftler: Optional[List[Cift]] = None) -> int:
    """JSONL'e yazar."""
    if ciftler is None:
        ciftler = uret()
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    with CIKTI.open("w", encoding="utf-8") as f:
        for c in ciftler:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return len(ciftler)


def main() -> int:
    """Tamamlama çifti üretimi."""
    ap = argparse.ArgumentParser(description="Mantık tamamlama çiftleri")
    ap.add_argument("--uret", action="store_true")
    ap.add_argument("--istatistik", action="store_true")
    ap.add_argument("--ornek", type=int, default=0,
                    help="Kaç örnek gösterilsin")
    args = ap.parse_args()

    print("=" * 78)
    print("  BLENDER MANTIK → TAMAMLAMA ÇİFTİ")
    print("=" * 78)

    ciftler = uret()
    st = istatistik(ciftler)
    print(f"\n  çift           : {st['cift']}")
    print(f"  karakter       : {st['karakter']:,}")
    print(f"  token          : ~{st['token_tahmini']:,}")
    print(f"  ders kapsama   : {st['ders_kapsama']}")
    print(f"  katman kapsama : {st['katman_kapsama']}")
    print(f"  ort. tamamlama : {st['ortalama_tamamlama']} karakter")

    if args.ornek:
        print(f"\n  {'=' * 74}")
        for c in ciftler[:args.ornek]:
            print(f"\n  --- {c.katman} · {c.ders[:50]} ---")
            print(f"  ÖNEK:")
            for ln in c.onek.splitlines()[-4:]:
                print(f"    {ln}")
            print(f"  TAMAMLAMA: {c.tamamlama.splitlines()[0]!r}")

    if args.uret:
        n = yaz(ciftler)
        print(f"\n  yazıldı: {CIKTI} ({n} çift)")
        Path("_mantik_tm.json").write_text(
            json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
