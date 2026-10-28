"""
Nihai Doğrulama — Hedefin Tüm Bileşenleri
==========================================
Tek komutla üretir, Godot ile doğrular ve rapor eder.

Doğruladığı bileşenler (hedefin 6 maddesi):
    1. Stylized lowpoly nesneler (12 çeşit)
    2. Texture / material sistemleri
    3. Karakter + monster + iskelet + animasyon
    4. Master server + proxy server ağ mimarisi
    5. Seviye tasarımı (labirent / arena / mağara)
    6. Girdi sistemleri (WASD, tuş atama)

Kullanım
--------
    python tools/nihai_dogrulama.py
    python tools/nihai_dogrulama.py --render
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

GODOT = r"E:\AppsTools\Godot_v4.8-dev7_win64\Godot_v4.8-dev7_win64_console.exe"


@dataclass
class Bilesen:
    """Bir hedef bileşeninin doğrulama sonucu."""
    ad: str = ""
    durum: bool = False
    kanit: str = ""
    detay: str = ""


@dataclass
class Rapor:
    """Nihai doğrulama raporu."""
    bilesenler: List[Bilesen] = field(default_factory=list)
    modlar: Dict[str, bool] = field(default_factory=dict)
    godot_hatalari: List[str] = field(default_factory=list)
    sure_s: float = 0.0
    render: List[str] = field(default_factory=list)

    def basarili(self) -> int:
        return sum(1 for b in self.bilesenler if b.durum)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "basarili": self.basarili(),
            "toplam": len(self.bilesenler),
            "bilesenler": [{"ad": b.ad, "durum": b.durum,
                            "kanit": b.kanit, "detay": b.detay[:120]}
                           for b in self.bilesenler],
            "modlar": self.modlar,
            "godot_hatalari": self.godot_hatalari[:6],
            "sure_s": round(self.sure_s, 1),
            "render": self.render,
        }


def godot_calistir(proje: Path, saniye: int = 120) -> Dict[str, Any]:
    """Godot ile projeyi başsız yükler."""
    if not Path(GODOT).exists():
        return {"atlandi": True}
    try:
        s = subprocess.run(
            [GODOT, "--headless", "--path", str(proje),
             "--quit-after", "150"],
            capture_output=True, text=True, timeout=saniye,
            encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"hata": f"{type(exc).__name__}"}
    cikti = (s.stdout or "") + (s.stderr or "")
    hatalar = [x.strip()[:130] for x in cikti.splitlines()
               if "ERROR" in x or "SCRIPT ERROR" in x]
    return {"exit": s.returncode, "hatalar": hatalar, "cikti": cikti}


def render_al(proje: Path, saniye: int = 120) -> List[str]:
    """Sahneyi PNG'ye çeker."""
    if not Path(GODOT).exists():
        return []
    try:
        subprocess.run(
            [GODOT, "--path", str(proje), "--script", "res://render.gd",
             "--resolution", "1024x768"],
            capture_output=True, text=True, timeout=saniye,
            encoding="utf-8", errors="replace")
    except Exception:
        return []
    return sorted(p.name for p in proje.glob("render_*.png"))


def main() -> int:
    """Nihai doğrulamayı çalıştırır."""
    ap = argparse.ArgumentParser(description="Nihai doğrulama")
    ap.add_argument("--render", action="store_true",
                    help="Render de al (GPU kullanır)")
    ap.add_argument("--cikti", default="TestProjects/nihai")
    args = ap.parse_args()

    t0 = time.perf_counter()
    r = Rapor()
    kok = Path(args.cikti)
    if kok.exists():
        shutil.rmtree(kok, ignore_errors=True)

    print("=" * 78)
    print("  NİHAİ DOĞRULAMA — hedefin tüm bileşenleri")
    print("=" * 78)

    # ── Modül kontrolleri ──
    from tools.godot_oyun_uret import (OyunUretici, Kararlar,
                                       NESNE_KATALOGU, PALETLER)
    from tools.godot_karakter_uret import (karakter_script, KEMIKLER,
                                           PARÇALAR, KARAKTER_PALETLERI)
    from tools.godot_ag_uret import ag_scriptleri
    from tools.guvenlik_filtresi import PARAMETRELER

    # 1) Stylized nesneler
    beklenen = ["kaya", "agac", "cali", "ev", "kapi", "pencere",
                "masa", "sandalye", "ucan_kaya", "buz", "sandik", "mesale"]
    eksik = [x for x in beklenen if x not in NESNE_KATALOGU]
    r.bilesenler.append(Bilesen(
        "1. Stylized lowpoly nesneler", not eksik,
        f"{len(NESNE_KATALOGU)} nesne",
        f"eksik: {eksik}" if eksik else "hepsi var"))

    # 2) Texture / material — dünya scriptinde kontrol
    u = OyunUretici(kok / "labirent")
    k = Kararlar(sahne_adi="nihai")
    k.nesneler = list(NESNE_KATALOGU.keys())
    u.uret(k)
    dunya = (kok / "labirent" / "scripts" / "dunya.gd").read_text(
        encoding="utf-8")
    mat_ozellik = ["StandardMaterial3D", "albedo_color", "roughness",
                   "vertex_color_use_as_albedo", "TRANSPARENCY_ALPHA",
                   "cull_mode", "material_override"]
    eksik_mat = [x for x in mat_ozellik if x not in dunya]
    r.bilesenler.append(Bilesen(
        "2. Texture / material", not eksik_mat,
        f"{len(mat_ozellik) - len(eksik_mat)}/{len(mat_ozellik)} özellik",
        f"eksik: {eksik_mat}" if eksik_mat else "hepsi var"))

    # 3) Karakter + iskelet + animasyon
    kh = (kok / "labirent" / "scripts" / "kahraman.gd").read_text(
        encoding="utf-8")
    gb = (kok / "labirent" / "scripts" / "goblin.gd").read_text(
        encoding="utf-8")
    kemik_ok = len(KEMIKLER) >= 12 and "Skeleton3D.new()" in kh
    bag_ok = "BoneAttachment3D.new()" in kh
    mon_ok = "enum Durum" in gb and "KOVALA" in gb
    anim_ok = all(a in kh for a in ("bekle", "yuru", "kos", "saldir",
                                    "hasar", "ol"))
    r.bilesenler.append(Bilesen(
        "3. Karakter + iskelet + animasyon",
        kemik_ok and bag_ok and mon_ok and anim_ok,
        f"{len(KEMIKLER)} kemik, {len(PARÇALAR)} parça, 6 animasyon",
        f"kemik={kemik_ok} bağ={bag_ok} monster={mon_ok} anim={anim_ok}"))

    # 4) Master + proxy server
    s = ag_scriptleri()
    ag_ok = all(x in s for x in ("master.gd", "proxy.gd",
                                 "oyun_sunucu.gd", "istemci.gd"))
    guv_ok = ("bilet_dogrula" in s.get("master.gd", "")
              and "@rpc" in s.get("oyun_sunucu.gd", "")
              and "get_remote_sender_id" in s.get("oyun_sunucu.gd", ""))
    r.bilesenler.append(Bilesen(
        "4. Master + proxy server", ag_ok and guv_ok,
        f"{len(s)} script",
        f"bileşenler={ag_ok} güvenlik={guv_ok}"))

    # 5-7) Seviye modları
    for ad, ayar in (("labirent", {"labirent": True}),
                     ("arena", {"arena": True}),
                     ("magara", {"magara": True})):
        ux = OyunUretici(kok / ad)
        kx = Kararlar(sahne_adi=ad)
        kx.labirent = kx.arena = kx.magara = False
        kx.nesneler = list(NESNE_KATALOGU.keys())
        for a, d in ayar.items():
            setattr(kx, a, d)
        ux.uret(kx)
        g = godot_calistir(kok / ad)
        ok = (not g.get("hata") and not g.get("hatalar")
              and g.get("exit") == 0)
        r.modlar[ad] = ok
        if g.get("hatalar"):
            r.godot_hatalari.extend(g["hatalar"][:2])

    r.bilesenler.append(Bilesen(
        "5. Seviye: labirent / arena / mağara",
        all(r.modlar.values()),
        f"{sum(r.modlar.values())}/3 mod Godot doğruladı",
        json.dumps(r.modlar)))

    # 6) Girdi sistemleri
    pg = (kok / "labirent" / "project.godot").read_text(encoding="utf-8")
    eylemler = pg.count("deadzone")
    girdi_ok = eylemler >= 4 and all(
        x in pg for x in ("ileri", "geri", "sol", "sag", "zipla"))
    r.bilesenler.append(Bilesen(
        "6. Girdi: WASD + tuş atama", girdi_ok,
        f"{eylemler} eylem", "hepsi var" if girdi_ok else "eksik"))

    # 7) Güvenlik filtresi
    r.bilesenler.append(Bilesen(
        "7. myAi güvenlik filtresi", len(PARAMETRELER) >= 5,
        f"{len(PARAMETRELER)} parametre", "aralık kontrolü aktif"))

    # ── Render ──
    if args.render:
        print("\n  Render alınıyor...")
        r.render = render_al(kok / "labirent")

    r.sure_s = time.perf_counter() - t0

    # ── Rapor ──
    print()
    print("-" * 78)
    for b in r.bilesenler:
        im = "✓" if b.durum else "✗"
        print(f"  {im} {b.ad:40} {b.kanit:28} {b.detay[:24]}")
    print("-" * 78)
    print(f"  SONUÇ: {r.basarili()}/{len(r.bilesenler)} bileşen doğrulandı")
    print(f"  süre : {r.sure_s:.1f} sn")
    if r.godot_hatalari:
        print(f"  Godot hataları: {len(r.godot_hatalari)}")
        for h in r.godot_hatalari[:3]:
            print(f"    {h[:110]}")
    else:
        print("  Godot hataları: 0")
    if r.render:
        print(f"  render: {', '.join(r.render)}")

    Path("_nihai_dogrulama.json").write_text(
        json.dumps(r.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("  rapor: _nihai_dogrulama.json")
    return 0 if r.basarili() == len(r.bilesenler) else 1


if __name__ == "__main__":
    raise SystemExit(main())
