"""
SmolLM2 İnce Ayar — Küçük Model, Kod Odaklı
============================================
⚠ Ölçülen sorun
---------------
```
Model   : 211.255.328 parametre
Veri    :    543.958 token
Oran    :     0,00257 token/parametre  (sağlıklı: 20)
EKSİK   :  7.780×
```

211M parametreli, **sıfırdan** eğitilmiş model 543k token'ı akıcı koda
çeviremiyor (ölçüldü: kod tamamlama 1/14).

Çözüm: **küçük ama ön-eğitimli** model
--------------------------------------
`HuggingFaceTB/SmolLM2-135M`:
    * **135M** parametre (211M yerine — veri/parametre oranı **iyileşir**)
    * **2 trilyon token** ön-eğitim (kod dahil)
    * Türkçe + İngilizce + kod karışık veriyle eğitilmiş

**Kritik fark:** Sıfırdan eğitim yerine **ince ayar**. Model zaten dili
ve kodu biliyor; biz ona **kendi verimizi** (5.660 not) öğretiyoruz.

Yeni oran
---------
```
SmolLM2: 135.000.000 parametre / 543.958 token = 0,00403
İyileşme: 211M'e göre 1,57×
Ama asıl kazanç: **2T token ön-eğitim** — oran hesabı anlamsız hale gelir
```

Kullanım
--------
    python tools/smollm_ince_ayar.py --indir
    python tools/smollm_ince_ayar.py --egit --adim 2000
    python tools/smollm_ince_ayar.py --olc
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Model kimliği
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

#: Yerel model dizini
MODEL_DIR = Path("models/smollm2_135m")

#: Eğitim verisi
VERI = Path(".neurocore/smollm_veri.jsonl")


@dataclass
class EgitimSonuc:
    """İnce ayar sonucu."""
    adim: int = 0
    kayip_ilk: Optional[float] = None
    kayip_son: Optional[float] = None
    sure_s: float = 0.0
    ornek: int = 0
    token: int = 0
    hata: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adim": self.adim, "kayip_ilk": self.kayip_ilk,
            "kayip_son": self.kayip_son, "sure_s": round(self.sure_s, 1),
            "ornek": self.ornek, "token": self.token, "hata": self.hata[:100],
        }


class SmolLM:
    """SmolLM2 indirme, ince ayar ve ölçüm."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tok: Any = None

    # ─── İndirme ───

    def indir(self) -> bool:
        """Modeli ve tokenizer'ı Hugging Face'ten indirir."""
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            print(f"  huggingface_hub yok: {exc}")
            return False

        if MODEL_DIR.exists() and (MODEL_DIR / "model.safetensors").exists():
            print(f"  model zaten var: {MODEL_DIR}")
            return True

        print(f"  indiriliyor: {MODEL_ID}")
        try:
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=str(MODEL_DIR),
                allow_patterns=["*.json", "*.safetensors", "*.txt"])
        except Exception as exc:
            print(f"  indirme hatası: {type(exc).__name__}: {str(exc)[:80]}")
            return False
        boyut = sum(f.stat().st_size for f in MODEL_DIR.rglob("*")
                    if f.is_file())
        print(f"  indirildi: {boyut / 1024**2:.0f} MB")
        return True

    def yukle(self, egitim: bool = False) -> bool:
        """Modeli belleğe yükler."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            print(f"  transformers yok: {exc}")
            return False
        if not MODEL_DIR.exists():
            print("  model dizini yok — önce --indir")
            return False
        try:
            self._tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
            self._model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR))
            if egitim:
                cihaz = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = self._model.to(cihaz)
                self._model.train()
                print(f"  model yüklendi (eğitim, {cihaz})")
            else:
                self._model.eval()
                print("  model yüklendi (değerlendirme)")
            return True
        except Exception as exc:
            print(f"  yükleme hatası: {type(exc).__name__}: {str(exc)[:80]}")
            return False

    # ─── Veri ───

    def veri_uret(self, maks_not: int = 8000) -> int:
        """
        Kasadaki notları **metin** hâline getirir.

        Biçim: `# Başlık\\n\\nGövde\\n\\n`
        Bu, SmolLM2'nin ön-eğitim biçimine yakındır.
        """
        try:
            from neurocore.knowledge import KnowledgeLayer

            k = KnowledgeLayer()
            k.initialize()
        except Exception as exc:
            print(f"  kasa açılamadı: {exc}")
            return 0

        sayi = 0
        with VERI.open("w", encoding="utf-8") as f:
            for n in k.vault.iter_notes(limit=maks_not):
                govde = (getattr(n, "body", "") or "").strip()
                baslik = (getattr(n, "title", "") or "").strip()
                if len(govde) < 40:
                    continue
                kayit = {"metin": f"# {baslik}\n\n{govde}\n\n",
                         "kategori": getattr(n, "category", "") or ""}
                f.write(json.dumps(kayit, ensure_ascii=False) + "\n")
                sayi += 1
        k.shutdown()
        print(f"  veri üretildi: {sayi} örnek → {VERI}")
        return sayi

    # ─── Eğitim ───

    def egit(self, adim: int = 2000, parti: int = 4, dizi: int = 512,
             lr: float = 5e-5) -> EgitimSonuc:
        """
        İnce ayar yapar.

        ⚠ Ölçülen tuzak: sıfırdan eğitimden farklı olarak **düşük öğrenme
        oranı** gerekir (5e-5). Ön-eğitimli ağırlıkları bozmamak için.
        """
        import torch

        s = EgitimSonuc()
        t0 = time.perf_counter()
        if self._model is None and not self.yukle(egitim=True):
            s.hata = "model yüklenemedi"
            return s
        if not VERI.exists():
            s.hata = "veri yok — önce --veri"
            return s

        # Veriyi oku
        metinler: List[str] = []
        with VERI.open(encoding="utf-8") as f:
            for satir in f:
                try:
                    metinler.append(json.loads(satir)["metin"])
                except Exception:
                    continue
        if not metinler:
            s.hata = "veri boş"
            return s
        s.ornek = len(metinler)

        # Token'la
        cihaz = next(self._model.parameters()).device
        tokenler: List[Any] = []
        toplam_tok = 0
        for m in metinler:
            t = self._tok(m, return_tensors="pt", truncation=True,
                          max_length=dizi)
            if t["input_ids"].shape[1] < 16:
                continue
            tokenler.append(t["input_ids"][0])
            toplam_tok += t["input_ids"].shape[1]
        if not tokenler:
            s.hata = "token üretilemedi"
            return s
        s.token = toplam_tok
        print(f"  {len(tokenler)} örnek, {toplam_tok:,} token")

        iy = torch.optim.AdamW(self._model.parameters(), lr=lr)
        self._model.train()
        kayiplar: List[float] = []
        import random

        rng = random.Random(42)
        for adim_no in range(1, adim + 1):
            # Rastgele bir örnek seç (basit yaklaşım — tam parti yerine)
            idx = rng.randrange(len(tokenler))
            ids = tokenler[idx][:dizi].unsqueeze(0).to(cihaz)
            if ids.shape[1] < 8:
                continue
            cikti = self._model(input_ids=ids, labels=ids)
            kayip = cikti.loss
            kayip.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            iy.step()
            iy.zero_grad(set_to_none=True)
            kayiplar.append(float(kayip))
            if adim_no == 1:
                s.kayip_ilk = float(kayip)
            if adim_no % 100 == 0:
                ort = sum(kayiplar[-100:]) / min(100, len(kayiplar))
                print(f"    adım {adim_no}/{adim} · kayıp {ort:.4f} · "
                      f"{(time.perf_counter() - t0) / 60:.1f} dk", flush=True)

        s.adim = len(kayiplar)
        s.kayip_son = (sum(kayiplar[-50:]) / min(50, len(kayiplar))
                       if kayiplar else None)
        s.sure_s = time.perf_counter() - t0
        return s

    def kaydet(self, yol: Optional[Path] = None) -> bool:
        """İnce ayarlı modeli kaydeder."""
        hedef = yol or Path("models/smollm2_kod")
        try:
            hedef.mkdir(parents=True, exist_ok=True)
            self._model.save_pretrained(str(hedef))
            self._tok.save_pretrained(str(hedef))
            print(f"  kaydedildi: {hedef}")
            return True
        except Exception as exc:
            print(f"  kayıt hatası: {type(exc).__name__}")
            return False

    # ─── Ölçüm ───

    def olc(self) -> Dict[str, Any]:
        """Kod tamamlama testi (aynı test seti)."""
        import torch

        if self._model is None and not self.yukle():
            return {"hata": "model yüklenemedi"}
        cihaz = next(self._model.parameters()).device
        testler = [
            ("import bpy\nbpy.ops.mesh.primitive_",
             ["cube_add", "cylinder_add", "uv_sphere_add"]),
            ("import bpy\nmat = bpy.data.materials.new(", ["name", "'"]),
            ("extends Node3D\n\nfunc _ready() -> void:\n    var mi := MeshInstance3D.new()\n    mi.mesh = ",
             ["Box", "Cylinder"]),
            ("import bpy\nfor v in mesh.vertices:\n    v.co.z = ",
             ["0", "h"]),
            ("import bpy\nob = bpy.context.object\nfor p in ob.data.polygons:\n    p.use_smooth = ",
             ["False"]),
            ("const PORT := 7777\npeer.create_server(", ["PORT", "16"]),
        ]
        ok = 0
        detay = []
        for onek, kabul in testler:
            try:
                ids = self._tok(onek, return_tensors="pt")["input_ids"].to(cihaz)
                with torch.no_grad():
                    cikti = self._model.generate(
                        ids, max_new_tokens=16, do_sample=False,
                        pad_token_id=self._tok.eos_token_id)
                uretilen = self._tok.decode(
                    cikti[0][ids.shape[1]:], skip_special_tokens=True)
            except Exception as exc:
                uretilen = f"HATA {type(exc).__name__}"
            basarili = any(k.lower() in uretilen.lower() for k in kabul)
            if basarili:
                ok += 1
            detay.append({"onek": onek.splitlines()[-1][:40],
                          "uretilen": uretilen[:60], "basarili": basarili})
        return {"basarili": ok, "toplam": len(testler),
                "oran": ok / len(testler), "detay": detay}


def main() -> int:
    """SmolLM2 işlemlerini çalıştırır."""
    ap = argparse.ArgumentParser(description="SmolLM2 ince ayar")
    ap.add_argument("--indir", action="store_true")
    ap.add_argument("--veri", action="store_true")
    ap.add_argument("--egit", action="store_true")
    ap.add_argument("--olc", action="store_true")
    ap.add_argument("--adim", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--maks-not", type=int, default=8000)
    args = ap.parse_args()

    print("=" * 78)
    print("  SmolLM2-135M — küçük, ön-eğitimli, kod odaklı ince ayar")
    print("=" * 78)

    s = SmolLM()
    sonuc: Dict[str, Any] = {}

    if args.indir or not MODEL_DIR.exists():
        if not s.indir():
            return 1
    if args.veri or not VERI.exists():
        s.veri_uret(maks_not=args.maks_not)
    if args.egit:
        r = s.egit(adim=args.adim, lr=args.lr)
        sonuc["egitim"] = r.to_dict()
        print(f"  EĞİTİM: {r.adim} adım · {r.kayip_ilk} → {r.kayip_son} · "
              f"{r.sure_s / 60:.1f} dk")
        if r.adim:
            s.kaydet()
    if args.olc:
        if s._model is None:
            s.yukle()
        m = s.olc()
        sonuc["olcum"] = m
        print(f"  ÖLÇÜM: {m.get('basarili')}/{m.get('toplam')}")
        for d in m.get("detay", [])[:6]:
            im = "OK " if d["basarili"] else "   "
            print(f"    {im}{d['onek']!r} -> {d['uretilen']!r}")

    Path("_smollm_sonuc.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
