"""
Animasyon Kütüphanesi — Çoklu Animasyon Üretici
=================================================
⚠ Kullanıcı hedefi (D)
----------------------
*"başarılı animler olursa onları o modele göre kayıt da tutma, çünkü
sürekli denemeler ile uygun animi oluşturmak kolay olmayacak."*

⚠ Mevcut durum (tur 8 sonu)
---------------------------
```
Kütüphane: 2 animasyon
  Yuruyus        (timsah · 12 kemik · 600 kf)
  Insan_Yuruyus  (insan  · 19 kemik · 350 kf)
```
**Her varlık için TEK animasyon** — mimari çoklu destekliyor ama
içerik yok.

Bu betik
--------
Bir armature'e **5 animasyon** ekler:
```
bekleme  (idle)     → hafif nefes/salınım, döngü
yurume   (walk)     → normal yürüyüş, döngü
kosma    (run)      → hızlı, büyük adım, öne eğik
saldiri  (attack)   → tek seferlik, kol/pençe savurma
olum     (death)    → tek seferlik, yere yığılma
```

⚠ Kritik tasarım: **döngü vs tek seferlik**
--------------------------------------------
```
Döngü (bekleme/yurume/kosma):
    ilk kare == son kare  → kesintisiz tekrar

Tek seferlik (saldiri/olum):
    ilk ≠ son             → `use_cyclic` KAPALI
    son kare "durur"      → animasyon biter
```

Blender 5.x notu (tur 6-8 ölçümleri):
```
· fcurve'leri DOĞRUDAN kur (keyframe_insert indeksi yazmıyor)
· kanal BAŞINA tek fcurve (biriktir, sonra kur)
· slot + channelbag ZORUNLU
```

Kullanım
--------
    python tools/anim_uret.py --hedef yaratik/timsah_0001 --tumu
    python tools/anim_uret.py --hedef karakter/insan_orta_orta --tumu
    python tools/anim_uret.py --liste
"""

from __future__ import annotations

import argparse
import inspect
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


#: ⚠ Animasyon tarifleri — **kemik rolüne** göre yazılır.
#:
#: Rol tabanlı yazım sayesinde aynı tarif **farklı kemik adlarına**
#: uygulanabilir (tur 4 dersi: `on_ayak_l` ≡ `front_leg_l`).
#:
#: Her tarif: (ad, döngü, kare, {rol: [(eksen, işlev)]})
#:   işlev(t, faz) → radyan
TARIFLER: Dict[str, Dict[str, Any]] = {

    # ── BEKLEME (idle) — hafif nefes ─────────────────────────────────
    "bekleme": {
        "dongu": True,
        "kare": 32,
        "aciklama": "Yerinde duruş — nefes + hafif salınım",
        "kanallar": {
            "omurga": [(1, lambda t, p: 0.022 * _sin(t))],
            "gogus": [(1, lambda t, p: 0.018 * _sin(t + 0.6))],
            "boyun": [(1, lambda t, p: -0.014 * _sin(t + 0.4))],
            "kafa": [(1, lambda t, p: -0.012 * _sin(t + 0.3)),
                     (2, lambda t, p: 0.020 * _sin(t * 0.5))],
            "on_ayak_l": [(2, lambda t, p: 0.010 * _sin(t + p))],
            "on_ayak_r": [(2, lambda t, p: 0.010 * _sin(t + p))],
            "arka_ayak_l": [(2, lambda t, p: 0.010 * _sin(t + p))],
            "arka_ayak_r": [(2, lambda t, p: 0.010 * _sin(t + p))],
            "ust_bacak_l": [(2, lambda t, p: 0.010 * _sin(t + p))],
            "ust_bacak_r": [(2, lambda t, p: 0.010 * _sin(t + p))],
            "ust_kol_l": [(0, lambda t, p: 0.030 * _sin(t + 0.5))],
            "ust_kol_r": [(0, lambda t, p: 0.030 * _sin(t + 0.5))],
            "alt_kol_l": [(0, lambda t, p: 0.020 * _sin(t + 1.0))],
            "alt_kol_r": [(0, lambda t, p: 0.020 * _sin(t + 1.0))],
            "kuyruk_1": [(2, lambda t, p: 0.040 * _sin(t * 0.7))],
            "kuyruk_2": [(2, lambda t, p: 0.055 * _sin(t * 0.7 + 0.4))],
            "kuyruk_3": [(2, lambda t, p: 0.070 * _sin(t * 0.7 + 0.8))],
            "kanat_l": [(2, lambda t, p: 0.050 * _sin(t * 0.8))],
            "kanat_r": [(2, lambda t, p: -0.050 * _sin(t * 0.8))],
        },
    },

    # ── YÜRÜME (walk) ────────────────────────────────────────────────
    "yurume": {
        "dongu": True,
        "kare": 24,
        "aciklama": "Normal yürüyüş",
        "kanallar": {
            # ── Dört ayaklı (timsah) ──
            "on_ayak_l": [(0, lambda t, p: _sin(t) * 0.55)],
            "on_ayak_r": [(0, lambda t, p: -_sin(t) * 0.55)],
            "arka_ayak_l": [(0, lambda t, p: -_sin(t) * 0.50)],
            "arka_ayak_r": [(0, lambda t, p: _sin(t) * 0.50)],
            # ── İki ayaklı (insan) ──
            "ust_bacak_l": [(0, lambda t, p: _sin(t) * 0.48)],
            "ust_bacak_r": [(0, lambda t, p: -_sin(t) * 0.48)],
            "alt_bacak_l": [(0, lambda t, p: max(0.0, -_sin(t)) * 0.85)],
            "alt_bacak_r": [(0, lambda t, p: max(0.0, _sin(t)) * 0.85)],
            "ayak_l": [(0, lambda t, p: -_sin(t) * 0.22)],
            "ayak_r": [(0, lambda t, p: _sin(t) * 0.22)],
            # ── Kollar (bacaklara zıt) ──
            "ust_kol_l": [(0, lambda t, p: -_sin(t) * 0.42)],
            "ust_kol_r": [(0, lambda t, p: _sin(t) * 0.42)],
            "alt_kol_l": [(0, lambda t, p: abs(_sin(t)) * 0.22)],
            "alt_kol_r": [(0, lambda t, p: abs(_sin(t)) * 0.22)],
            # ── Ortak gövde ──
            "kok": [(2, lambda t, p: _sin(t) * 0.035)],
            "omurga": [(2, lambda t, p: -_sin(t) * 0.030)],
            "gogus": [(2, lambda t, p: _sin(t) * 0.025)],
            "kafa": [(0, lambda t, p: _sin(t) * 0.09),
                     (2, lambda t, p: -_sin(t) * 0.020)],
            "kuyruk_1": [(2, lambda t, p: _sin(t) * 0.22)],
            "kuyruk_2": [(2, lambda t, p: _sin(t + 0.5) * 0.30)],
            "kuyruk_3": [(2, lambda t, p: _sin(t + 1.0) * 0.38)],
        },
    },

    # ── KOŞMA (run) — büyük adım, öne eğik, hızlı ────────────────────
    "kosma": {
        "dongu": True,
        "kare": 18,
        "aciklama": "Hızlı koşu — büyük adım, öne eğik gövde",
        "kanallar": {
            "on_ayak_l": [(0, lambda t, p: _sin(t) * 0.95)],
            "on_ayak_r": [(0, lambda t, p: -_sin(t) * 0.95)],
            "arka_ayak_l": [(0, lambda t, p: -_sin(t) * 0.85)],
            "arka_ayak_r": [(0, lambda t, p: _sin(t) * 0.85)],
            "ust_bacak_l": [(0, lambda t, p: _sin(t) * 0.85)],
            "ust_bacak_r": [(0, lambda t, p: -_sin(t) * 0.85)],
            "alt_bacak_l": [(0, lambda t, p: max(0.0, -_sin(t)) * 1.35)],
            "alt_bacak_r": [(0, lambda t, p: max(0.0, _sin(t)) * 1.35)],
            "ayak_l": [(0, lambda t, p: -_sin(t) * 0.40)],
            "ayak_r": [(0, lambda t, p: _sin(t) * 0.40)],
            "ust_kol_l": [(0, lambda t, p: -_sin(t) * 0.80)],
            "ust_kol_r": [(0, lambda t, p: _sin(t) * 0.80)],
            "alt_kol_l": [(0, lambda t, p: 0.55 + abs(_sin(t)) * 0.35)],
            "alt_kol_r": [(0, lambda t, p: 0.55 + abs(_sin(t)) * 0.35)],
            # ⚠ Koşuda gövde ÖNE eğik (sabit + salınım)
            "kok": [(0, lambda t, p: 0.12),
                    (2, lambda t, p: _sin(t) * 0.055)],
            "omurga": [(0, lambda t, p: 0.10),
                       (2, lambda t, p: -_sin(t) * 0.045)],
            "gogus": [(0, lambda t, p: 0.08),
                      (2, lambda t, p: _sin(t) * 0.035)],
            "kafa": [(0, lambda t, p: -0.18 + _sin(t) * 0.14)],
            "kuyruk_1": [(0, lambda t, p: 0.15),
                         (2, lambda t, p: _sin(t) * 0.35)],
            "kuyruk_2": [(0, lambda t, p: 0.12),
                         (2, lambda t, p: _sin(t + 0.5) * 0.45)],
            "kuyruk_3": [(0, lambda t, p: 0.10),
                         (2, lambda t, p: _sin(t + 1.0) * 0.55)],
        },
    },

    # ── SALDIRI (attack) — tek seferlik ──────────────────────────────
    "saldiri": {
        "dongu": False,
        "kare": 20,
        "aciklama": "Saldırı — geri çekil, sonra savur",
        "kanallar": {
            # ⚠ Zamanlama: 0-0.35 hazırlık, 0.35-0.60 vuruş, sonra dönüş
            "on_ayak_l": [(0, lambda t, p: _vurus(t) * 1.25)],
            "on_ayak_r": [(0, lambda t, p: _vurus(t) * 1.25)],
            "arka_ayak_l": [(0, lambda t, p: _vurus(t) * -0.35)],
            "arka_ayak_r": [(0, lambda t, p: _vurus(t) * -0.35)],
            "ust_kol_l": [(0, lambda t, p: _vurus(t) * 1.60)],
            "ust_kol_r": [(0, lambda t, p: _vurus(t) * 1.60)],
            "alt_kol_l": [(0, lambda t, p: max(0.0, _vurus(t)) * 0.90)],
            "alt_kol_r": [(0, lambda t, p: max(0.0, _vurus(t)) * 0.90)],
            "omurga": [(0, lambda t, p: _vurus(t) * 0.30)],
            "gogus": [(0, lambda t, p: _vurus(t) * 0.35)],
            "boyun": [(0, lambda t, p: _vurus(t) * 0.28)],
            "kafa": [(0, lambda t, p: _vurus(t) * 0.45)],
            # ⚠ Çene açılması (varsa)
            "cene": [(0, lambda t, p: max(0.0, _vurus(t)) * 0.55)],
            "jaw": [(0, lambda t, p: max(0.0, _vurus(t)) * 0.55)],
            "kuyruk_1": [(2, lambda t, p: -_vurus(t) * 0.40)],
            "kuyruk_2": [(2, lambda t, p: -_vurus(t) * 0.50)],
            "kuyruk_3": [(2, lambda t, p: -_vurus(t) * 0.60)],
        },
    },

    # ── ÖLÜM (death) — tek seferlik, yere yığılma ────────────────────
    "olum": {
        "dongu": False,
        "kare": 30,
        "aciklama": "Ölüm — dizler çöker, gövde yana devrilir",
        "kanallar": {
            # ⚠ `_olum(t)` 0→1 monoton: geri dönmez
            "kok": [(1, lambda t, p: _olum(t) * 0.55),
                    (2, lambda t, p: _olum(t) * 0.80)],
            "omurga": [(1, lambda t, p: _olum(t) * 0.45),
                       (2, lambda t, p: _olum(t) * 0.35)],
            "gogus": [(1, lambda t, p: _olum(t) * 0.40),
                      (2, lambda t, p: _olum(t) * 0.25)],
            "boyun": [(1, lambda t, p: _olum(t) * 0.35)],
            "kafa": [(1, lambda t, p: _olum(t) * 0.60),
                     (2, lambda t, p: _olum(t) * 0.30)],
            "ust_bacak_l": [(0, lambda t, p: -_olum(t) * 1.05)],
            "ust_bacak_r": [(0, lambda t, p: -_olum(t) * 0.95)],
            "alt_bacak_l": [(0, lambda t, p: _olum(t) * 1.45)],
            "alt_bacak_r": [(0, lambda t, p: _olum(t) * 1.35)],
            "ust_kol_l": [(0, lambda t, p: _olum(t) * 0.75),
                          (2, lambda t, p: -_olum(t) * 0.50)],
            "ust_kol_r": [(0, lambda t, p: _olum(t) * 0.65),
                          (2, lambda t, p: _olum(t) * 0.50)],
            "alt_kol_l": [(0, lambda t, p: _olum(t) * 0.55)],
            "alt_kol_r": [(0, lambda t, p: _olum(t) * 0.45)],
            "on_ayak_l": [(0, lambda t, p: _olum(t) * 0.90)],
            "on_ayak_r": [(0, lambda t, p: _olum(t) * 0.85)],
            "arka_ayak_l": [(0, lambda t, p: -_olum(t) * 0.70)],
            "arka_ayak_r": [(0, lambda t, p: -_olum(t) * 0.65)],
            "kuyruk_1": [(2, lambda t, p: _olum(t) * 0.35)],
            "kuyruk_2": [(2, lambda t, p: _olum(t) * 0.45)],
            "kuyruk_3": [(2, lambda t, p: _olum(t) * 0.55)],
        },
    },

    # ── ZIPLAMA (jump) — tek seferlik ────────────────────────────────
    "ziplama": {
        "dongu": False,
        "kare": 24,
        "aciklama": "Zıplama — çömel, havalan, in",
        "kanallar": {
            # ⚠ Bacaklar: çömelmede bükülür, havada uzanır
            "ust_bacak_l": [(0, lambda t, p: -_zipla(t) * 0.75)],
            "ust_bacak_r": [(0, lambda t, p: -_zipla(t) * 0.72)],
            "alt_bacak_l": [(0, lambda t, p: _zipla(t) * 1.15)],
            "alt_bacak_r": [(0, lambda t, p: _zipla(t) * 1.10)],
            "ayak_l": [(0, lambda t, p: -_zipla(t) * 0.35)],
            "ayak_r": [(0, lambda t, p: -_zipla(t) * 0.33)],
            "on_ayak_l": [(0, lambda t, p: -_zipla(t) * 0.80)],
            "on_ayak_r": [(0, lambda t, p: -_zipla(t) * 0.78)],
            "arka_ayak_l": [(0, lambda t, p: -_zipla(t) * 0.70)],
            "arka_ayak_r": [(0, lambda t, p: -_zipla(t) * 0.68)],
            # ⚠ Kollar: havalanırken yukarı
            "ust_kol_l": [(0, lambda t, p: -_zipla(t) * 1.25)],
            "ust_kol_r": [(0, lambda t, p: -_zipla(t) * 1.25)],
            "alt_kol_l": [(0, lambda t, p: abs(_zipla(t)) * 0.55)],
            "alt_kol_r": [(0, lambda t, p: abs(_zipla(t)) * 0.55)],
            # ⚠ Govde: havalanirken geriye, comelirken one
            "kok": [(0, lambda t, p: _zipla(t) * 0.28)],
            "omurga": [(0, lambda t, p: -_zipla(t) * 0.22)],
            "gogus": [(0, lambda t, p: -_zipla(t) * 0.18)],
            "kafa": [(0, lambda t, p: _zipla(t) * 0.30)],
            # ⚠ Havada kuyruk dengeler
            "kuyruk_1": [(0, lambda t, p: _zipla(t) * 0.35)],
            "kuyruk_2": [(0, lambda t, p: _zipla(t) * 0.45)],
            "kuyruk_3": [(0, lambda t, p: _zipla(t) * 0.55)],
        },
    },

    # ── DÜŞME (fall) — tek seferlik ──────────────────────────────────
    "dusme": {
        "dongu": False,
        "kare": 20,
        "aciklama": "Düşme — hızlanan, kollar yukarı",
        "kanallar": {
            "ust_bacak_l": [(0, lambda t, p: _dusme(t) * 0.55)],
            "ust_bacak_r": [(0, lambda t, p: _dusme(t) * 0.50)],
            "alt_bacak_l": [(0, lambda t, p: _dusme(t) * 0.85)],
            "alt_bacak_r": [(0, lambda t, p: _dusme(t) * 0.80)],
            "ayak_l": [(0, lambda t, p: -_dusme(t) * 0.45)],
            "ayak_r": [(0, lambda t, p: -_dusme(t) * 0.42)],
            "ust_kol_l": [(0, lambda t, p: -_dusme(t) * 1.50)],
            "ust_kol_r": [(0, lambda t, p: -_dusme(t) * 1.50)],
            "alt_kol_l": [(0, lambda t, p: _dusme(t) * 0.70)],
            "alt_kol_r": [(0, lambda t, p: _dusme(t) * 0.70)],
            "kok": [(0, lambda t, p: -_dusme(t) * 0.30)],
            "omurga": [(0, lambda t, p: -_dusme(t) * 0.25)],
            "kafa": [(0, lambda t, p: _dusme(t) * 0.40)],
            "kuyruk_1": [(0, lambda t, p: -_dusme(t) * 0.40)],
            "kuyruk_2": [(0, lambda t, p: -_dusme(t) * 0.50)],
            "kuyruk_3": [(0, lambda t, p: -_dusme(t) * 0.60)],
        },
    },

    # ── OTURMA (sit) — tek seferlik (kalıcı poz) ─────────────────────
    "oturma": {
        "dongu": False,
        "kare": 22,
        "aciklama": "Oturma — dizler bükülür, gövde alçalır",
        "kanallar": {
            "ust_bacak_l": [(0, lambda t, p: -_oturma(t) * 1.35)],
            "ust_bacak_r": [(0, lambda t, p: -_oturma(t) * 1.32)],
            "alt_bacak_l": [(0, lambda t, p: _oturma(t) * 1.55)],
            "alt_bacak_r": [(0, lambda t, p: _oturma(t) * 1.52)],
            "ayak_l": [(0, lambda t, p: -_oturma(t) * 0.20)],
            "ayak_r": [(0, lambda t, p: -_oturma(t) * 0.19)],
            "on_ayak_l": [(0, lambda t, p: _oturma(t) * 0.90)],
            "on_ayak_r": [(0, lambda t, p: _oturma(t) * 0.88)],
            "arka_ayak_l": [(0, lambda t, p: -_oturma(t) * 0.95)],
            "arka_ayak_r": [(0, lambda t, p: -_oturma(t) * 0.92)],
            "ust_kol_l": [(0, lambda t, p: _oturma(t) * 0.35)],
            "ust_kol_r": [(0, lambda t, p: _oturma(t) * 0.35)],
            "alt_kol_l": [(0, lambda t, p: _oturma(t) * 0.65)],
            "alt_kol_r": [(0, lambda t, p: _oturma(t) * 0.65)],
            "kok": [(0, lambda t, p: _oturma(t) * 0.45)],
            "omurga": [(0, lambda t, p: _oturma(t) * 0.18)],
            "gogus": [(0, lambda t, p: -_oturma(t) * 0.12)],
            "kafa": [(0, lambda t, p: -_oturma(t) * 0.22)],
            "kuyruk_1": [(0, lambda t, p: _oturma(t) * 0.30)],
            "kuyruk_2": [(0, lambda t, p: _oturma(t) * 0.40)],
            "kuyruk_3": [(0, lambda t, p: _oturma(t) * 0.50)],
        },
    },

    # ── BLOK (block) — tek seferlik, savunma pozu ────────────────────
    "blok": {
        "dongu": False,
        "kare": 16,
        "aciklama": "Blok — kollar önde çapraz, gövde kapanır",
        "kanallar": {
            # ⚠ Kollar göğüs önünde çapraz (X pozisyonu)
            "ust_kol_l": [(0, lambda t, p: _oturma(t) * 1.30),
                          (2, lambda t, p: _oturma(t) * 0.45)],
            "ust_kol_r": [(0, lambda t, p: _oturma(t) * 1.30),
                          (2, lambda t, p: -_oturma(t) * 0.45)],
            "alt_kol_l": [(0, lambda t, p: _oturma(t) * 1.55)],
            "alt_kol_r": [(0, lambda t, p: _oturma(t) * 1.55)],
            "omuz_l": [(2, lambda t, p: _oturma(t) * 0.25)],
            "omuz_r": [(2, lambda t, p: -_oturma(t) * 0.25)],
            # ⚠ Gövde kapanır, dizler hafif bükülür
            "kok": [(0, lambda t, p: _oturma(t) * 0.22)],
            "omurga": [(0, lambda t, p: _oturma(t) * 0.30)],
            "gogus": [(0, lambda t, p: _oturma(t) * 0.20)],
            "boyun": [(0, lambda t, p: _oturma(t) * 0.15)],
            "kafa": [(0, lambda t, p: _oturma(t) * 0.25)],
            "ust_bacak_l": [(0, lambda t, p: -_oturma(t) * 0.35)],
            "ust_bacak_r": [(0, lambda t, p: -_oturma(t) * 0.33)],
            "alt_bacak_l": [(0, lambda t, p: _oturma(t) * 0.45)],
            "alt_bacak_r": [(0, lambda t, p: _oturma(t) * 0.43)],
        },
    },
    # ── YÜZME (swim) — döngü ────────────────────────────────────────
    "yuzme": {
        "dongu": True,
        "kare": 28,
        "aciklama": "Yüzme — kollar/kuyruk dalgalanır, gövde yatay",
        "kanallar": {
            # ⚠ Gövde YATAY (0.9 rad öne eğik) + dalgalanma
            "kok": [(0, lambda t, p: 0.85),
                    (2, lambda t, p: _yuzme(t) * 0.10)],
            "omurga": [(0, lambda t, p: 0.15),
                       (2, lambda t, p: _yuzme(t + 0.15) * 0.14)],
            "gogus": [(0, lambda t, p: 0.10),
                      (2, lambda t, p: _yuzme(t + 0.30) * 0.12)],
            "boyun": [(0, lambda t, p: -0.35)],
            "kafa": [(0, lambda t, p: -0.45),
                     (2, lambda t, p: _yuzme(t + 0.45) * 0.10)],
            # ⚠ Kollar: kürek çekme (büyük çevrim)
            "ust_kol_l": [(0, lambda t, p: -1.20 + _yuzme(t) * 0.85)],
            "ust_kol_r": [(0, lambda t, p: -1.20 + _yuzme(t + 0.5) * 0.85)],
            "alt_kol_l": [(0, lambda t, p: 0.55 + abs(_yuzme(t)) * 0.45)],
            "alt_kol_r": [(0, lambda t, p: 0.55 + abs(_yuzme(t + 0.5)) * 0.45)],
            # ⚠ Bacaklar: yavaş çırpma
            "ust_bacak_l": [(0, lambda t, p: 0.15 + _yuzme(t + 0.25) * 0.35)],
            "ust_bacak_r": [(0, lambda t, p: 0.15 + _yuzme(t + 0.75) * 0.35)],
            "alt_bacak_l": [(0, lambda t, p: 0.30 + abs(_yuzme(t + 0.25)) * 0.40)],
            "alt_bacak_r": [(0, lambda t, p: 0.30 + abs(_yuzme(t + 0.75)) * 0.40)],
            # ⚠ Dört ayaklı (timsah) — kuyruk ana itici
            "on_ayak_l": [(0, lambda t, p: _yuzme(t) * 0.65)],
            "on_ayak_r": [(0, lambda t, p: _yuzme(t + 0.5) * 0.65)],
            "arka_ayak_l": [(0, lambda t, p: _yuzme(t + 0.25) * 0.55)],
            "arka_ayak_r": [(0, lambda t, p: _yuzme(t + 0.75) * 0.55)],
            "kuyruk_1": [(2, lambda t, p: _yuzme(t) * 0.55)],
            "kuyruk_2": [(2, lambda t, p: _yuzme(t + 0.2) * 0.70)],
            "kuyruk_3": [(2, lambda t, p: _yuzme(t + 0.4) * 0.85)],
        },
    },

    # ── TIRMANMA (climb) — döngü ─────────────────────────────────────
    "tirmanma": {
        "dongu": True,
        "kare": 26,
        "aciklama": "Tırmanma — dönüşümlü kol/bacak, gövde duvara dönük",
        "kanallar": {
            # ⚠ Gövde öne eğik (duvara yapışık)
            "kok": [(0, lambda t, p: 0.30)],
            "omurga": [(0, lambda t, p: 0.18),
                       (2, lambda t, p: _tirmanma(t) * 0.12)],
            "gogus": [(0, lambda t, p: 0.12)],
            "boyun": [(0, lambda t, p: -0.30)],
            "kafa": [(0, lambda t, p: -0.40),
                     (2, lambda t, p: _tirmanma(t + 0.25) * 0.15)],
            # ⚠ Kollar: DÖNÜŞÜMLÜ yukarı uzanma (zıt faz)
            "ust_kol_l": [(0, lambda t, p: -1.55 + _tirmanma(t) * 0.95)],
            "ust_kol_r": [(0, lambda t, p: -1.55 + _tirmanma(t + 0.5) * 0.95)],
            "alt_kol_l": [(0, lambda t, p: 0.70 - _tirmanma(t) * 0.55)],
            "alt_kol_r": [(0, lambda t, p: 0.70 - _tirmanma(t + 0.5) * 0.55)],
            "el_l": [(0, lambda t, p: _tirmanma(t) * 0.35)],
            "el_r": [(0, lambda t, p: _tirmanma(t + 0.5) * 0.35)],
            # ⚠ Bacaklar: kollara ZIT (dört nokta temas)
            "ust_bacak_l": [(0, lambda t, p: 0.55 + _tirmanma(t + 0.5) * 0.60)],
            "ust_bacak_r": [(0, lambda t, p: 0.55 + _tirmanma(t) * 0.60)],
            "alt_bacak_l": [(0, lambda t, p: 0.85 - _tirmanma(t + 0.5) * 0.45)],
            "alt_bacak_r": [(0, lambda t, p: 0.85 - _tirmanma(t) * 0.45)],
            "ayak_l": [(0, lambda t, p: -_tirmanma(t + 0.5) * 0.30)],
            "ayak_r": [(0, lambda t, p: -_tirmanma(t) * 0.30)],
            # ⚠ Dört ayaklı — dönüşümlü pençe
            "on_ayak_l": [(0, lambda t, p: -0.95 + _tirmanma(t) * 0.90)],
            "on_ayak_r": [(0, lambda t, p: -0.95 + _tirmanma(t + 0.5) * 0.90)],
            "arka_ayak_l": [(0, lambda t, p: -0.85 + _tirmanma(t + 0.5) * 0.85)],
            "arka_ayak_r": [(0, lambda t, p: -0.85 + _tirmanma(t) * 0.85)],
            "kuyruk_1": [(0, lambda t, p: 0.25),
                         (2, lambda t, p: _tirmanma(t) * 0.30)],
            "kuyruk_2": [(2, lambda t, p: _tirmanma(t + 0.2) * 0.40)],
            "kuyruk_3": [(2, lambda t, p: _tirmanma(t + 0.4) * 0.50)],
        },
    },

    # ── HASAR ALMA (hurt) — tek seferlik ────────────────────────────
    "hasar": {
        "dongu": False,
        "kare": 18,
        "aciklama": "Hasar — geri tepme + sarsıntı + toparlanma",
        "kanallar": {
            "kok": [(0, lambda t, p: _hasar(t) * 0.45)],
            "omurga": [(0, lambda t, p: _hasar(t) * 0.60)],
            "gogus": [(0, lambda t, p: _hasar(t) * 0.50)],
            "boyun": [(0, lambda t, p: _hasar(t) * 0.40)],
            "kafa": [(0, lambda t, p: _hasar(t) * 0.75)],
            # ⚠ Kollar: savunmasız geri
            "ust_kol_l": [(0, lambda t, p: -_hasar(t) * 0.75),
                          (2, lambda t, p: _hasar(t) * 0.30)],
            "ust_kol_r": [(0, lambda t, p: -_hasar(t) * 0.75),
                          (2, lambda t, p: -_hasar(t) * 0.30)],
            "alt_kol_l": [(0, lambda t, p: abs(_hasar(t)) * 0.65)],
            "alt_kol_r": [(0, lambda t, p: abs(_hasar(t)) * 0.65)],
            # ⚠ Bacaklar: geri adım
            "ust_bacak_l": [(0, lambda t, p: -_hasar(t) * 0.55)],
            "ust_bacak_r": [(0, lambda t, p: -_hasar(t) * 0.45)],
            "alt_bacak_l": [(0, lambda t, p: abs(_hasar(t)) * 0.70)],
            "alt_bacak_r": [(0, lambda t, p: abs(_hasar(t)) * 0.60)],
            # ⚠ Dört ayaklı
            "on_ayak_l": [(0, lambda t, p: -_hasar(t) * 0.70)],
            "on_ayak_r": [(0, lambda t, p: -_hasar(t) * 0.65)],
            "arka_ayak_l": [(0, lambda t, p: _hasar(t) * 0.50)],
            "arka_ayak_r": [(0, lambda t, p: _hasar(t) * 0.45)],
            "kuyruk_1": [(2, lambda t, p: -_hasar(t) * 0.35)],
            "kuyruk_2": [(2, lambda t, p: -_hasar(t) * 0.45)],
            "kuyruk_3": [(2, lambda t, p: -_hasar(t) * 0.55)],
        },
    },

    # ── DİRİLME (revive) — tek seferlik, ölümün TERSİ ───────────────
    "dirilme": {
        "dongu": False,
        "kare": 32,
        "aciklama": "Dirilme — yerde başlar, ayağa kalkar",
        "kanallar": {
            # ⚠ `_dirilme(t)` = 1 → 0 (azalan)
            "kok": [(1, lambda t, p: _dirilme(t) * 0.55),
                    (2, lambda t, p: _dirilme(t) * 0.80)],
            "omurga": [(1, lambda t, p: _dirilme(t) * 0.45),
                       (2, lambda t, p: _dirilme(t) * 0.35)],
            "gogus": [(1, lambda t, p: _dirilme(t) * 0.40),
                      (2, lambda t, p: _dirilme(t) * 0.25)],
            "boyun": [(1, lambda t, p: _dirilme(t) * 0.35)],
            "kafa": [(1, lambda t, p: _dirilme(t) * 0.60),
                     (2, lambda t, p: _dirilme(t) * 0.30)],
            "ust_bacak_l": [(0, lambda t, p: -_dirilme(t) * 1.05)],
            "ust_bacak_r": [(0, lambda t, p: -_dirilme(t) * 0.95)],
            "alt_bacak_l": [(0, lambda t, p: _dirilme(t) * 1.45)],
            "alt_bacak_r": [(0, lambda t, p: _dirilme(t) * 1.35)],
            "ust_kol_l": [(0, lambda t, p: _dirilme(t) * 0.75),
                          (2, lambda t, p: -_dirilme(t) * 0.50)],
            "ust_kol_r": [(0, lambda t, p: _dirilme(t) * 0.65),
                          (2, lambda t, p: _dirilme(t) * 0.50)],
            "alt_kol_l": [(0, lambda t, p: _dirilme(t) * 0.55)],
            "alt_kol_r": [(0, lambda t, p: _dirilme(t) * 0.45)],
            "on_ayak_l": [(0, lambda t, p: _dirilme(t) * 0.90)],
            "on_ayak_r": [(0, lambda t, p: _dirilme(t) * 0.85)],
            "arka_ayak_l": [(0, lambda t, p: -_dirilme(t) * 0.70)],
            "arka_ayak_r": [(0, lambda t, p: -_dirilme(t) * 0.65)],
            "kuyruk_1": [(2, lambda t, p: _dirilme(t) * 0.35)],
            "kuyruk_2": [(2, lambda t, p: _dirilme(t) * 0.45)],
            "kuyruk_3": [(2, lambda t, p: _dirilme(t) * 0.55)],
        },
    },
}


#: ⚠ Betik üretiminde kullanılan yardımcı işlevler
YARDIMCI = r'''
import math

def _sin(t):
    """Tam çevrim sinüs (2π t)."""
    return math.sin(t * 2.0 * math.pi)

def _vurus(t):
    """
    Saldırı zamanlaması — 0 → 1 → 0 (tek tepe).

    ⚠ 0-0.35 hazırlık (geri), 0.35-0.60 vuruş (ileri), 0.60-1 dönüş
    """
    if t < 0.35:
        return -math.sin(t / 0.35 * math.pi * 0.5) * 0.55
    if t < 0.60:
        u = (t - 0.35) / 0.25
        return -0.55 + u * 1.55
    u = (t - 0.60) / 0.40
    return 1.0 * (1.0 - u) ** 1.6

def _zipla(t):
    """
    Zıplama — 0 → 1 → 0 (tek tepe, havada kalma).

    ⚠ 0-0.2 çömelme (negatif), 0.2-0.5 yükselme, 0.5-1 iniş
    """
    if t < 0.18:
        return -(t / 0.18) * 0.45
    if t < 0.55:
        u = (t - 0.18) / 0.37
        return -0.45 + (u ** 0.6) * 1.0
    u = (t - 0.55) / 0.45
    return 1.0 - (u ** 1.4) * 1.45


def _dusme(t):
    """Düşme — 0 → 1 hızlanan (yerçekimi)."""
    return t ** 1.8


def _oturma(t):
    """Oturma — 0 → 1 yumuşak (smoothstep)."""
    return t * t * (3.0 - 2.0 * t)


def _yuzme(t):
    """
    Yüzme — tam çevrim (kollar/kuyruk dalgalanır).

    ⚠ `sin` ile aynı ama **genlik modülasyonlu** (su direnci).
    """
    return math.sin(t * 2.0 * math.pi) * (0.75 + 0.25 * math.sin(
        t * 4.0 * math.pi))


def _tirmanma(t):
    """
    Tırmanma — testere dişi (bir el yukarı, sonra diğeri).

    ⚠ 0-0.5 sol kol, 0.5-1 sağ kol — **simetrik olmayan**.
    """
    if t < 0.5:
        return math.sin(t * 2.0 * math.pi) * 1.0
    return math.sin((t - 0.5) * 2.0 * math.pi) * 1.0


def _hasar(t):
    """
    Hasar alma — 0 → 1 → geri (sarsıntı, sonra toparlanma).

    ⚠ 0-0.15 ani geri tepme, 0.15-0.5 sarsıntı, 0.5-1 toparlanma
    """
    if t < 0.15:
        return -(t / 0.15) * 0.85
    if t < 0.5:
        u = (t - 0.15) / 0.35
        return -0.85 * (1.0 - u) + 0.25 * math.sin(u * 6.0 * math.pi)
    u = (t - 0.5) / 0.5
    return 0.25 * (1.0 - u) ** 2


def _dirilme(t):
    """
    Dirilme — 1 → 0 (ölümden geri).

    ⚠ `_olum`un TERSİ: yerde başlar, ayağa kalkar.
    """
    return 1.0 - _olum(t)


def _olum(t):
    """Ölüm — 0 → 1 monoton (yumuşak başlangıç, hızlı çöküş)."""
    if t < 0.25:
        return (t / 0.25) ** 3 * 0.15
    u = (t - 0.25) / 0.75
    return 0.15 + 0.85 * (1.0 - (1.0 - u) ** 2.2)
'''


#: ⚠ Blender betiği — animasyonları armature'e yazar
URETICI = r'''
import bpy, json, math

bpy.ops.wm.open_mainfile(filepath=HEDEF_BLEND)

# ⚠ Yardımcı işlevler (YARDIMCI) çağrı tarafında **önce** eklenir

# ⚠ Hedef armature
arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if arm is None:
    print("S" + json.dumps({"hata": "armature yok"}))
    raise SystemExit(0)

pb = arm.pose.bones
MEVCUT_KEMIKLER = [b.name for b in arm.data.bones]

# ⚠ ROL eşleştirmesi (kemik adı farklı olsa da çalışsın)
ROLLER = {
    "kok": ("kok", "root", "kök"),
    "omurga": ("omurga", "spine", "govde", "body", "pelvis"),
    "gogus": ("gogus", "göğüs", "chest", "toraks"),
    "boyun": ("boyun", "neck"),
    "kafa": ("kafa", "head", "kelle"),
    "cene": ("cene", "çene", "jaw", "mouth"),
    "on_ayak_l": ("on_ayak_l", "front_leg_l", "bacak_on_sol", "on_bacak_l",
                  "ayak_on_l"),
    "on_ayak_r": ("on_ayak_r", "front_leg_r", "bacak_on_sag", "on_bacak_r",
                  "ayak_on_r"),
    "arka_ayak_l": ("arka_ayak_l", "back_leg_l", "bacak_arka_sol",
                    "arka_bacak_l", "ayak_arka_l"),
    "arka_ayak_r": ("arka_ayak_r", "back_leg_r", "bacak_arka_sag",
                    "arka_bacak_r", "ayak_arka_r"),
    "kuyruk_1": ("kuyruk_1", "kuyruk1", "tail_1", "tail1"),
    "kuyruk_2": ("kuyruk_2", "kuyruk2", "tail_2", "tail2"),
    "kuyruk_3": ("kuyruk_3", "kuyruk3", "tail_3", "tail3"),
    "ust_bacak_l": ("ust_bacak_l", "thigh_l", "uyluk_l", "bacak_ust_l"),
    "ust_bacak_r": ("ust_bacak_r", "thigh_r", "uyluk_r", "bacak_ust_r"),
    "alt_bacak_l": ("alt_bacak_l", "shin_l", "baldir_l", "bacak_alt_l"),
    "alt_bacak_r": ("alt_bacak_r", "shin_r", "baldir_r", "bacak_alt_r"),
    "ayak_l": ("ayak_l", "foot_l", "foot_l_bone"),
    "ayak_r": ("ayak_r", "foot_r", "foot_r_bone"),
    "omuz_l": ("omuz_l", "shoulder_l", "clavicle_l"),
    "omuz_r": ("omuz_r", "shoulder_r", "clavicle_r"),
    "ust_kol_l": ("ust_kol_l", "upper_arm_l", "arm_l", "kol_ust_l"),
    "ust_kol_r": ("ust_kol_r", "upper_arm_r", "arm_r", "kol_ust_r"),
    "alt_kol_l": ("alt_kol_l", "forearm_l", "lower_arm_l", "kol_alt_l"),
    "alt_kol_r": ("alt_kol_r", "forearm_r", "lower_arm_r", "kol_alt_r"),
    "el_l": ("el_l", "hand_l", "el_sol"),
    "el_r": ("el_r", "hand_r", "el_sag"),
    "kanat_l": ("kanat_l", "wing_l", "sol_kanat"),
    "kanat_r": ("kanat_r", "wing_r", "sag_kanat"),
}

def rol_esle(rol):
    """Rol → hedef armature'teki kemik adı."""
    for es in ROLLER.get(rol, (rol,)):
        if es in MEVCUT_KEMIKLER:
            return es
    return None

olcum = {"yazilan": [], "atlanan": [], "hata": [], "kemik": len(MEVCUT_KEMIKLER)}

# ⚠ TARIFLER'i eval ile kur — lambda'lar YARDIMCI işlevlerini görsün
TARIFLER_V = eval(TARIFLER_KAYNAK, globals())   # noqa: S307

for ANIM_AD, TARIF in TARIFLER_V.items():
    DONGU = TARIF["dongu"]
    KARE_SAYISI = TARIF["kare"]
    KANAL_TARIF = TARIF["kanallar"]

    # ⚠ Kanalları rol → gerçek kemik eşlemesiyle kur
    #: {kemik: {eksen: [(kare, deger), ...]}}
    kanallar: dict = {}
    for rol, tanimlar in KANAL_TARIF.items():
        kemik = rol_esle(rol)
        if kemik is None:
            continue
        for eksen, islev in tanimlar:
            for kare in range(KARE_SAYISI + 1):
                t = kare / KARE_SAYISI
                try:
                    deger = float(islev(t, 0.35))
                except Exception as exc:
                    olcum["hata"].append(f"{ANIM_AD}/{rol}: {type(exc).__name__}")
                    deger = 0.0
                kanallar.setdefault(kemik, {}).setdefault(eksen, []).append(
                    (kare + 1, deger))

    if not kanallar:
        olcum["atlanan"].append(f"{ANIM_AD}: eşleşen kemik yok")
        continue

    # ⚠ tur 9 ÖLÇÜMÜ: betik birden çok kez çalışınca Blender aynı ada
    # `.001`, `.002` ekler → ölçüm: timsah'ta **20 action** (4 tekrar × 5).
    # Çözüm: aynı adlı (ve türevli) action'lar **önce silinir**.
    for _a in list(bpy.data.actions):
        if _a.name == ANIM_AD or _a.name.startswith(ANIM_AD + "."):
            try:
                _a.use_fake_user = False
            except Exception:
                pass
            try:
                bpy.data.actions.remove(_a, do_unlink=True)
            except Exception:
                pass

    # ⚠ Action oluştur (Blender 5.x katmanlı API)
    if not arm.animation_data:
        arm.animation_data_create()
    act = bpy.data.actions.new(ANIM_AD)
    # ⚠ tur 9 KESİN KÖK NEDEN: Blender kaydederken **kullanılmayan
    # action'ları siler** (fake user yoksa). Ölçüm: 5 animasyon
    # üretildi ama diskte **1** kaldı (`bekleme`).
    # → Her action'a **fake user** verilir, böylece dosyada kalır.
    act.use_fake_user = True
    # ⚠ Tek seferlik animasyonda döngü KAPALI
    try:
        act.use_cyclic = bool(DONGU)
    except Exception:
        pass
    kat = act.layers.new("Katman")
    st = kat.strips.new(type="KEYFRAME")
    slot = act.slots.new(id_type="OBJECT", name=arm.name)
    cb = st.channelbag(slot, ensure=True)

    # ⚠ fcurve'leri DOĞRUDAN kur — kanal BAŞINA bir kez
    yazilan_kf = 0
    for kemik in sorted(kanallar.keys()):
        for eksen in sorted(kanallar[kemik].keys()):
            noktalar = sorted(kanallar[kemik][eksen])
            tam = f'pose.bones["{kemik}"].rotation_euler'
            try:
                fc = cb.fcurves.new(data_path=tam, index=eksen)
            except Exception as exc:
                olcum["hata"].append(
                    f"{ANIM_AD}/{kemik}[{eksen}]: {type(exc).__name__}")
                continue
            fc.keyframe_points.add(len(noktalar))
            for i, (kare, deger) in enumerate(noktalar):
                kp = fc.keyframe_points[i]
                kp.co = (float(kare), float(deger))
                kp.interpolation = "BEZIER"
            fc.update()
            yazilan_kf += len(noktalar)

    olcum["yazilan"].append({
        "ad": ANIM_AD, "dongu": DONGU, "kare": KARE_SAYISI,
        "kemik": len(kanallar),
        "fcurve": sum(len(v) for v in kanallar.values()),
        "keyframe": yazilan_kf, "aciklama": TARIF["aciklama"]})

# ⚠ Sahneyi ilk animasyona ayarla
if bpy.data.actions:
    arm.animation_data.action = bpy.data.actions[0]
bpy.ops.wm.save_as_mainfile(filepath=HEDEF_BLEND)
olcum["tum_action"] = [a.name for a in bpy.data.actions]
print("S" + json.dumps(olcum, ensure_ascii=False))
'''


def _sin(t: float) -> float:
    """Tam çevrim sinüs."""
    import math

    return math.sin(t * 2.0 * math.pi)


#: ⚠ Python tarafında test için (Blender tarifiyle aynı)
def _vurus(t: float) -> float:
    """Saldırı zamanlaması."""
    import math

    if t < 0.35:
        return -math.sin(t / 0.35 * math.pi * 0.5) * 0.55
    if t < 0.60:
        return -0.55 + (t - 0.35) / 0.25 * 1.55
    return 1.0 * (1.0 - (t - 0.60) / 0.40) ** 1.6


def _olum(t: float) -> float:
    """Ölüm eğrisi."""
    if t < 0.25:
        return (t / 0.25) ** 3 * 0.15
    u = (t - 0.25) / 0.75
    return 0.15 + 0.85 * (1.0 - (1.0 - u) ** 2.2)


# ══ ⚠ tur 25: ORTAK yol çözümleyici ═══════════════════════════════════
# Neden: 5 araçta **aynı** yol hatası vardı —
#   `Path("Varliklar") / hedef` → hedef zaten `Varliklar\...` içeriyorsa
#   **çift** birleşiyordu (`Varliklar\Varliklar\...`).
# Ve ID ile arama yapanlar bulamıyordu.
def _ortak_blend_bul(hedef: str):
    """Ortak çözümleyici (tam yol · klasör · göreli · ad · katalog)."""
    try:
        from tools.varlik_yolu import blend_bul

        return blend_bul(hedef)
    except Exception:
        return None


def uret(hedef: str, *, zaman: int = 420) -> Dict[str, Any]:
    """
    Hedef varlığa **tüm** animasyonları ekler.

    ⚠ `hedef` üç biçimde gelebilir:
        1. tam yol     : `Varliklar/karakter/insan_orta_orta/varlik.blend`
        2. kategori/ad : `karakter/insan_orta_orta`
        3. yalnız ad   : `insan_orta_orta`, `timsah_0001`
    """
    if not BLENDER.exists():
        return {"hata": "blender yok"}
    # ⚠ tur 25: ortak çözümleyici (çift `Varliklar\` önlendi)
    _o = _ortak_blend_bul(hedef)
    if _o is not None:
        blend = _o
    else:
        _p = Path(hedef)
        blend = _p if _p.suffix == ".blend" else (
            Path("Varliklar") / hedef / "varlik.blend")
    if not blend.exists():
        # ⚠ Yalnız ad verilmişse: her kategori altında dene
        _parca = hedef.split("/")[-1]
        for _kat in ("kaya", "bitki", "esya", "mimari", "yaratik",
                     "karakter"):
            _aday = Path("Varliklar") / _kat / _parca / "varlik.blend"
            if _aday.exists():
                blend = _aday
                break
    if not blend.exists():
        # Katalogdan dene (ad'da kalite eki olabilir)
        from tools.varlik_katalogu import listele

        kok = hedef.split("/")[-1]
        for v in listele():
            vid = str(v.get("id") or "")
            vad = str(v.get("ad") or "")
            _b = Path(v.get("blend") or "")
            if not _b.exists():
                continue
            if (vad == kok or vid.startswith(kok)
                    or kok.startswith(vad) or vad.startswith(kok)):
                blend = _b
                break
    if not blend.exists():
        return {"hata": f"blend yok: {blend}"}

    # ⚠ TARIFLER'i Blender'a **JSON olarak** gönderemeyiz (lambda var).
    # Bu yüzden betiği Python tarafında kurarız.
    # ⚠ YARDIMCI işlevler ÖNCE tanımlanır, sonra TARIFLER eval edilir
    kaynak = _tarifler_kaynak()
    kod = (f'HEDEF_BLEND = r"{blend.resolve()}"\n'
           + YARDIMCI
           + f"\nTARIFLER_KAYNAK = {kaynak!r}\n"
           + URETICI)

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "a.py"
        b.write_text(kod, encoding="utf-8")
        try:
            p = subprocess.run(
                [str(BLENDER), "--background", "--factory-startup",
                 "--python", str(b)],
                capture_output=True, text=True, timeout=zaman,
                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return {"hata": "zaman aşımı"}
    sure = time.perf_counter() - t0
    out = (p.stdout or "") + (p.stderr or "")
    for satir in out.splitlines():
        if "S{" in satir:
            try:
                d = json.loads(satir.strip()[1:])
                d["sure_s"] = round(sure, 2)
                d["hedef"] = hedef
                return d
            except Exception:
                pass
    iz = [x.strip() for x in out.splitlines() if "Error" in x]
    return {"hata": "; ".join(iz[:2])[:180] or "çıktı yok"}


def _tarifler_kaynak() -> str:
    """
    `TARIFLER` tanımını **kaynak dosyadan** çıkarır.

    ⚠ tur 9 ÖLÇÜMÜ: lambda'yı `inspect.getsource` ile ayıklamak
    **kırılgan** — `getsource` lambda'nın tüm satırını döner
    (`"omurga": [(1, lambda t, p: ...))],`) ve ayıklama heuristiği
    `(1, lambda t),` gibi **kesik** kod üretti.

    ✅ Sağlam yol: sözlük tanımını dosyadan **olduğu gibi** al ve
    Blender tarafında `eval` ile kur (lambda'lar zaten metindir).
    """
    yol = Path(__file__)
    ham = yol.read_text(encoding="utf-8")
    bas = ham.index("TARIFLER: Dict[str, Dict[str, Any]] = {")
    # ⚠ Sözlük `\n}\n` ile kapanır — ondan sonrasını kes
    son = ham.index("\n}\n", bas) + 2
    return ham[bas + len("TARIFLER: Dict[str, Dict[str, Any]] = "):son]


def main() -> int:
    """Animasyon üretimi."""
    ap = argparse.ArgumentParser(description="Çoklu animasyon üretici")
    ap.add_argument("--hedef", default="",
                    help="kategori/varlik (ör. yaratik/timsah_0001)")
    ap.add_argument("--tumu", action="store_true",
                    help="Tüm armature'lü varlıklara uygula")
    ap.add_argument("--liste", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  ÇOKLU ANİMASYON ÜRETİCİ")
    print("=" * 78)

    if args.liste:
        print(f"\n  {len(TARIFLER)} ANİMASYON TARİFİ:")
        for ad, t in TARIFLER.items():
            d = "döngü" if t["dongu"] else "tek seferlik"
            print(f"    {ad:10} {t['kare']:>3} kare · {d:14} "
                  f"{len(t['kanallar'])} rol · {t['aciklama']}")
        return 0

    hedefler: List[str] = []
    if args.tumu:
        # ⚠ Katalogdan armature'lü varlıkları bul.
        # Yol `kategori/ad` ile KURULMAZ (ad'da kalite eki olabilir) —
        # katalogdaki **gerçek `blend` yolu** kullanılır.
        from tools.varlik_katalogu import listele

        for v in listele():
            if not v.get("armature_var"):
                continue
            bl = Path(v.get("blend") or "")
            if not bl.exists():
                bl = (Path("Varliklar") / str(v.get("kategori", ""))
                      / str(v.get("ad", "")) / "varlik.blend")
            if bl.exists():
                hedefler.append(str(bl))
            else:
                print(f"  ⚠ bulunamadı: {v.get('id')}")
    elif args.hedef:
        hedefler = [args.hedef]

    if not hedefler:
        ap.print_help()
        return 0

    print(f"  {len(TARIFLER)} tarif · {len(hedefler)} hedef")
    sonuclar = []
    for h in hedefler:
        print(f"\n  [{h}]", flush=True)
        d = uret(h)
        sonuclar.append(d)
        # ⚠ tur 9 ÖLÇÜMÜ: `if "hata" in d:` YANLIŞ — ölçüm sözlüğünde
        # `"hata": []` her zaman bulunur, boş olsa bile. Bu yüzden
        # başarılı üretim "✗ []" diye raporlanıyordu.
        # Doğru ölçüt: hata **dolu** mu?
        _hata = d.get("hata")
        _dolu = bool(_hata) if isinstance(_hata, (list, str)) else False
        if _dolu:
            print(f"    ✗ {str(_hata)[:70]}")
            continue
        print(f"    ✓ {d['sure_s']} sn · {len(d.get('yazilan', []))} animasyon"
              f" · kemik={d.get('kemik')}")
        for y in d.get("yazilan", []):
            dongu = "döngü" if y["dongu"] else "tek  "
            print(f"      {y['ad']:10} {dongu} {y['kare']:>3} kare · "
                  f"kemik={y['kemik']:>2} fcurve={y['fcurve']:>2} "
                  f"kf={y['keyframe']:>4}")
        if d.get("atlanan"):
            print(f"      atlanan: {d['atlanan'][:3]}")
        # ⚠ Boş liste yanıltıcı mesaj üretiyordu → uzunluk kontrolü
        _h = d.get("hata") or []
        if isinstance(_h, list) and _h:
            print(f"      ⚠ {len(_h)} hata: {_h[:3]}")
        elif isinstance(_h, str) and _h:
            print(f"      ⚠ {_h[:80]}")

    Path("_anim_uretim.json").write_text(
        json.dumps({"sonuclar": sonuclar}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
