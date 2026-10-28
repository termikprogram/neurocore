"""
Uçtan uca varlık üretim zinciri testi (tur 25).

⚠ Amaç: Tüm 7 Blender aracını **tek bir gerçek senaryoda** çalıştırıp
her adımın çıktısını **ölçmek**.

Senaryo
-------
```
1. asset.uret       → "timsah" canavar (12 kemik, armature + weight)
2. asset.animasyon  → 13 tarif ekle
3. asset.gecis      → 24 geçiş + NLA
4. asset.doku       → doku ara + uygula
5. (UV)             → varlik_uret içinde otomatik
6. asset.anim_onizleme → hareket doğrula
7. asset.onizleme   → 4 açı render
8. asset.katalog    → kayıt + meta
```

⚠ Her adımda **ölçüm** alınır ve rapor edilir.
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

#: ⚠ Kök dizin (bu dosya `tools/` içinde)
_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def _bas(no: int, ad: str) -> float:
    print(f"\n  ── {no}. {ad} " + "─" * (52 - len(ad)))
    return time.perf_counter()


def _son(t0: float, ozet: str) -> None:
    print(f"     ✓ {ozet}  ({time.perf_counter() - t0:.1f} sn)")


def main() -> int:
    """Uçtan uca zincir."""
    print("=" * 78)
    print("  UÇTAN UCA VARLIK ÜRETİM ZİNCİRİ")
    print("=" * 78)

    from neurocore.agent.coding import CodingAgent

    a = CodingAgent()
    sonuc: Dict[str, Any] = {"adim": [], "basarili": 0, "hata": []}

    def kaydet(ad: str, ok: bool, detay: Any) -> None:
        sonuc["adim"].append({"ad": ad, "basarili": ok, "detay": detay})
        if ok:
            sonuc["basarili"] += 1
        else:
            sonuc["hata"].append(ad)

    # ══ 1. VARLIK ÜRET ═══════════════════════════════════════════════
    t = _bas(1, "asset.uret — timsah canavar (kaliteli)")
    r = a.uret(urun="timsah", kalite="kaliteli", tohum=7)
    if not r.get("basarili"):
        print(f"     ✗ {str(r.get('hata'))[:100]}")
        kaydet("uret", False, r.get("hata"))
        return 1
    # ⚠ Gerçek anahtarlar: vertex · yuz · uv · malzeme · armature ·
    #   kemik · weight_orani · animasyon · boyut · klasor
    vertex = r.get("vertex", 0)
    yuz = r.get("yuz", 0)
    kemik = r.get("kemik", 0)
    oran = r.get("weight_orani", 0)
    klasor = r.get("klasor", "")
    _son(t, f"vertex={vertex} yüz={yuz} kemik={kemik} "
            f"ağırlık={oran} uv={r.get('uv')} klasör={klasor}")
    kaydet("uret", True, {"vertex": vertex, "kemik": kemik,
                         "weight_orani": oran, "klasor": str(klasor)})

    # ══ 2. ANİMASYON ═════════════════════════════════════════════════
    t = _bas(2, "asset.animasyon — 13 tarif")
    # ⚠ `klasor` = Varliklar/<kategori>/<ad>_<kalite>
    hedef = str(r.get("klasor") or "")
    r2 = a.animasyon(hedef)
    if not r2.get("basarili"):
        print(f"     ✗ {str(r2.get('hata'))[:100]}")
        kaydet("animasyon", False, r2.get("hata"))
    else:
        _a = r2.get("animasyonlar") or r2.get("yazilan") or []
        n = len(_a)
        adlar = [x.get("ad") for x in _a]
        _son(t, f"animasyon={n} · {', '.join(adlar[:5])}…")
        kaydet("animasyon", True, {"sayi": n})

    # ══ 3. GEÇİŞ ═════════════════════════════════════════════════════
    t = _bas(3, "asset.gecis — 24 geçiş + NLA")
    r3 = a.gecis(hedef)
    if not r3.get("basarili"):
        print(f"     ✗ {str(r3.get('hata'))[:100]}")
        kaydet("gecis", False, r3.get("hata"))
    else:
        # ⚠ Gerçek anahtarlar: gecis_sayisi · gercek_gecis · nla_iz
        _g = r3.get("gecis_sayisi", 0)
        _gr = r3.get("gercek_gecis", 0)
        _atl = len(r3.get("atlanan", []))
        _son(t, f"geçiş={_g} gerçek={_gr} nla_iz={r3.get('nla_iz', 0)} "
                f"atlanan={_atl}")
        kaydet("gecis", True, {"gecis": _g, "gercek": _gr})

    # ══ 4. DOKU ══════════════════════════════════════════════════════
    # ⚠ tur 26: `asset.doku` ÜÇ MODLU:
    #     `sorgu`           → ara + puanla + sun
    #     `hedef` + `doku`  → prosedürel doku UYGULA
    #     `hedef` + `dosya` → görüntü doku UYGULA
    # ⚠ Önceki test `a.doku(hedef, tur="deri")` çağırıyordu →
    # `hedef` **sorgu** olarak yorumlandı, doku uygulanmadı.
    t = _bas(4, "asset.doku — prosedürel 'deri' dokusu UYGULA")
    r4a = a.doku(sorgu="deri")
    _ara = len(r4a.get("prosedurel") or [])
    r4 = a.doku(hedef=hedef, doku="deri")
    if not r4.get("basarili"):
        print(f"     ~ {str(r4.get('hata') or r4.get('uyari'))[:90]}")
        kaydet("doku", False, r4.get("hata"))
    else:
        _d = (r4.get("uygulanan") or r4.get("doku")
              or r4.get("secili") or "-")
        # ⚠ Gerçek anahtarlar: dugum · baglanti · uv (dugum_sayisi değil)
        _dug = r4.get("dugum") or r4.get("dugum_sayisi") or 0
        _bag = r4.get("baglanti", 0)
        _son(t, f"arama={_ara} doku={_d} düğüm={_dug} bağlantı={_bag} "
                f"uv={r4.get('uv', 0)}")
        # ⚠ Bayrak değil SONUÇ: düğüm + bağlantı olmalı
        _ok = bool(r4.get("basarili")) and _dug >= 3 and _bag >= 2
        kaydet("doku", _ok, {"doku": _d, "dugum": _dug, "baglanti": _bag})

    # ══ 5. ANİMASYON DOĞRULAMA ═══════════════════════════════════════
    t = _bas(5, "asset.anim_onizleme — hareket doğrula")
    r5 = a.anim_onizleme(hedef)
    if not r5.get("basarili"):
        print(f"     ~ {str(r5.get('hata'))[:90]}")
        kaydet("anim_onizleme", False, r5.get("hata"))
    else:
        _son(t, f"toplam={r5.get('toplam')} normal={r5.get('normal')} "
                f"düz={r5.get('gecis_duz')} hata={r5.get('hareketsiz')}")
        kaydet("anim_onizleme", True, {"normal": r5.get("normal")})

    # ══ 6. ÖNİZLEME ══════════════════════════════════════════════════
    t = _bas(6, "asset.onizleme — 4 açı render")
    r6 = a.onizleme(hedef)
    if not r6.get("basarili"):
        print(f"     ✗ {str(r6.get('hata'))[:100]}")
        kaydet("onizleme", False, r6.get("hata"))
    else:
        _son(t, f"açı={r6.get('aci') or r6.get('basarili')} "
                f"süre={r6.get('sure_s')} sn")
        kaydet("onizleme", True, {"basarili": r6.get("basarili")})

    # ══ 7. KATALOG ═══════════════════════════════════════════════════
    t = _bas(7, "asset.katalog — kayıt + meta")
    r7 = a.katalog()
    if not r7.get("basarili"):
        print(f"     ✗ {str(r7.get('hata'))[:100]}")
        kaydet("katalog", False, r7.get("hata"))
    else:
        s = r7.get("istatistik", {})
        _son(t, f"toplam={r7.get('toplam')} "
                f"armature={s.get('armaturelu')} anim={s.get('animasyonlu')} "
                f"önizleme={s.get('onizlemeli')} "
                f"anim_önizleme={s.get('anim_onizlemeli')}")
        kaydet("katalog", True, s)

    # ══ ÖZET ═════════════════════════════════════════════════════════
    n = len(sonuc["adim"])
    print("\n" + "=" * 78)
    print(f"  SONUÇ: {sonuc['basarili']}/{n} adım başarılı")
    if sonuc["hata"]:
        print(f"  ⚠ HATA: {sonuc['hata']}")
    print("=" * 78)

    import json

    Path("_uctan_uca.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if sonuc["basarili"] == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
