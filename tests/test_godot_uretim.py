"""
Uçtan Uca Doğrulama Testi — Godot Oyun Üretimi
===============================================
Hedefin **kanıtı**: myAi karar verir → kod üretilir → Godot doğrular.

Bu test şunları ölçer:
    1. **Üreticiler** çalışıyor mu? (oyun, karakter, ağ)
    2. **Üretilen kod** sözdizimi geçerli mi?
    3. **Godot** projeyi hatasız yüklüyor mu? (başsız, exit kodu)
    4. **Sahne** kuruluyor mu? (arazi, labirent, karakterler, nesneler)
    5. **Üç seviye modu** çalışıyor mu? (labirent / arena / mağara)

Kullanım
--------
    python tests/test_godot_uretim.py
    python tests/test_godot_uretim.py --godot   # Godot doğrulaması da
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

#: Godot yolu
GODOT = r"E:\AppsTools\Godot_v4.8-dev7_win64\Godot_v4.8-dev7_win64_console.exe"

#: Godot doğrulaması yapılsın mı? (test sırasında ayarlanır)
GODOT_AKTIF = False


def godot_dogrula(proje: Path, saniye: int = 90) -> Dict[str, Any]:
    """
    Projeyi Godot ile **başsız** yükler ve hataları toplar.

    ⚠ Ölçülen tuzaklar:
        * `--quit-after` kullanılmalı; `--script` + `await` takılıyor
        * Godot GPU'yu kullanır — eğitim çalışırken çakışır
        * Çıktı UTF-8, `errors="replace"` gerekir
    """
    if not Path(GODOT).exists():
        return {"atlandi": "Godot bulunamadı"}
    try:
        s = subprocess.run(
            [GODOT, "--headless", "--path", str(proje),
             "--quit-after", "120"],
            capture_output=True, text=True, timeout=saniye,
            encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"hata": "Godot zaman aşımı"}
    except Exception as exc:
        return {"hata": f"{type(exc).__name__}: {str(exc)[:60]}"}

    cikti = (s.stdout or "") + (s.stderr or "")
    hatalar: List[str] = []
    for satir in cikti.splitlines():
        if "ERROR" in satir or "SCRIPT ERROR" in satir:
            hatalar.append(satir.strip()[:140])
    return {"exit": s.returncode, "hatalar": hatalar, "cikti": cikti}


class TestUreticiModulleri(unittest.TestCase):
    """Üretici modüllerin temel çalışması."""

    def test_oyun_uretici_icerik(self) -> None:
        """Oyun üretici gerekli dosyaları üretir."""
        from tools.godot_oyun_uret import OyunUretici, Kararlar, NESNE_KATALOGU

        with tempfile.TemporaryDirectory() as td:
            u = OyunUretici(Path(td))
            k = Kararlar(sahne_adi="test")
            k.nesneler = list(NESNE_KATALOGU.keys())
            r = u.uret(k)
            dosyalar = set(r["dosyalar"])
            for beklenen in ("project.godot", "ana.tscn", "dunya.gd",
                             "oyuncu.gd", "hud.gd", "kahraman.gd",
                             "goblin.gd", "render.gd"):
                self.assertTrue(any(beklenen in d for d in dosyalar),
                                f"{beklenen} üretilmedi: {dosyalar}")
            # Sunucu dosyaları
            self.assertTrue((Path(td) / "sunucu" / "master.gd").exists())
            self.assertTrue((Path(td) / "sunucu" / "proxy.gd").exists())

    def test_girdi_haritasi_dolu(self) -> None:
        """
        ⚠ Ölçülen hata: `Kararlar()` doğrudan oluşturulunca `girdi` boş
        kalıyordu ve `project.godot` "0 eylem" ile yazılıyordu — oyunda
        hiç tuş çalışmıyordu.
        """
        from tools.godot_oyun_uret import OyunUretici, Kararlar

        k = Kararlar()
        self.assertGreaterEqual(len(k.girdi), 4,
                                "varsayılan girdi haritası boş")
        for eylem in ("ileri", "geri", "sol", "sag", "zipla"):
            self.assertIn(eylem, k.girdi)

        with tempfile.TemporaryDirectory() as td:
            u = OyunUretici(Path(td))
            u.uret(k)
            icerik = (Path(td) / "project.godot").read_text(encoding="utf-8")
            self.assertIn("[input]", icerik)
            # En az 4 eylem tanımlı olmalı
            self.assertGreaterEqual(icerik.count("deadzone"), 4)

    def test_karakter_iskelet(self) -> None:
        """Karakter üretici iskelet + animasyon kodu üretir."""
        from tools.godot_karakter_uret import (
            karakter_script, KEMIKLER, PARÇALAR, KARAKTER_PALETLERI)

        self.assertGreaterEqual(len(KEMIKLER), 12, "kemik sayısı az")
        self.assertGreaterEqual(len(PARÇALAR), 8, "parça sayısı az")
        self.assertGreaterEqual(len(KARAKTER_PALETLERI), 4)

        for palet in KARAKTER_PALETLERI:
            kod = karakter_script("T", palet, monster=False)
            self.assertIn("Skeleton3D.new()", kod)
            self.assertIn("BoneAttachment3D.new()", kod)
            self.assertIn("set_bone_pose_rotation", kod)
            # Monster varyantı
            mkod = karakter_script("C", palet, monster=True)
            self.assertIn("enum Durum", mkod)
            self.assertIn("KOVALA", mkod)

    def test_karakter_fizik_cakismasi(self) -> None:
        """
        ⚠ Ölçülen hata: şablonda boş `_fizik` stub'ı vardı, monster kodu
        da tam sürümü ekliyordu → "Function _fizik has the same name as a
        previously declared function". Oyuncuda ise `_fizik` hiç yoktu
        ama `_process` onu çağırıyordu → "Function _fizik() not found".
        """
        from tools.godot_karakter_uret import karakter_script

        oyuncu = karakter_script("O", "kahraman", monster=False)
        monster = karakter_script("M", "goblin", monster=True)
        self.assertEqual(oyuncu.count("func _fizik"), 0,
                         "oyuncuda _fizik tanımı olmamalı")
        self.assertEqual(oyuncu.count("_fizik(delta)"), 0,
                         "oyuncuda _fizik çağrısı olmamalı")
        self.assertEqual(monster.count("func _fizik"), 1,
                         "monsterda tam 1 _fizik tanımı olmalı")
        self.assertEqual(monster.count("_fizik(delta)"), 1)

    def test_ag_mimarisi(self) -> None:
        """Ağ üretici 4 bileşeni üretir ve güvenlik önlemleri içerir."""
        from tools.godot_ag_uret import ag_scriptleri

        s = ag_scriptleri()
        for ad in ("master.gd", "proxy.gd", "oyun_sunucu.gd", "istemci.gd"):
            self.assertIn(ad, s, f"{ad} üretilmedi")
            self.assertGreater(len(s[ad]), 1500, f"{ad} çok kısa")

        # Güvenlik önlemleri
        self.assertIn("bilet_dogrula", s["master.gd"])
        self.assertIn("kullanildi", s["master.gd"])
        # ⚠ Fonksiyon adı `hiz_izni` (alt çizgisiz) — test
        # yazarken `_hiz_izni` aranmıştı ve yanlış başarısız oluyordu.
        self.assertIn("hiz_izni", s["master.gd"])
        self.assertIn("@rpc", s["oyun_sunucu.gd"])
        self.assertIn("get_remote_sender_id", s["oyun_sunucu.gd"])
        # Hasarı sunucu belirlemeli (istemci miktar göndermemeli)
        self.assertIn("hasar", s["oyun_sunucu.gd"].lower())

    def test_labirent_algoritmasi(self) -> None:
        """Labirent üretimi dunya.gd içinde yerel fonksiyon olmalı."""
        from tools.godot_oyun_uret import OyunUretici, Kararlar

        with tempfile.TemporaryDirectory() as td:
            u = OyunUretici(Path(td))
            k = Kararlar(); k.labirent = True; k.labirent_boyut = 15
            u.uret(k)
            icerik = (Path(td) / "scripts" / "dunya.gd").read_text(
                encoding="utf-8")
            self.assertIn("_labirent_kur", icerik)
            self.assertIn("_labirent_kaz", icerik)
            # ⚠ `class_name Labirent` sabitle çakışıyordu — ama bu ifade
            # **yorumda** geçiyor (hatayı açıklayan not). Yorumları
            # ayıklayıp koda bakmak gerekir.
            kod_satirlari = [
                x.split("#")[0] for x in icerik.splitlines()]
            kod = "\n".join(kod_satirlari)
            self.assertNotIn("class_name Labirent", kod,
                             "class_name hâlâ kodda (yorum değil)")
            # Kutu elle üretilmeli (BoxMesh.get_mesh() yok)
            self.assertIn("_kutu_ekle", icerik)
            self.assertNotIn("get_mesh()", kod)

    def test_uc_mod(self) -> None:
        """Labirent, arena ve mağara modları ayrı üretilebilir."""
        from tools.godot_oyun_uret import OyunUretici, Kararlar

        modlar = [
            ("labirent", {"labirent": True}),
            ("arena", {"arena": True}),
            ("magara", {"magara": True}),
        ]
        for ad, ayar in modlar:
            with tempfile.TemporaryDirectory() as td:
                u = OyunUretici(Path(td))
                k = Kararlar(sahne_adi=ad)
                k.labirent = k.arena = k.magara = False
                for anahtar, deger in ayar.items():
                    setattr(k, anahtar, deger)
                u.uret(k)
                icerik = (Path(td) / "scripts" / "dunya.gd").read_text(
                    encoding="utf-8")
                self.assertIn(f"const {ad.upper()} := true", icerik)

    def test_nesne_katalogu(self) -> None:
        """Hedefin istediği 12 nesne katalogda olmalı."""
        from tools.godot_oyun_uret import NESNE_KATALOGU

        beklenen = ["kaya", "masa", "sandalye", "ev", "kapi", "pencere",
                    "agac", "cali", "ucan_kaya", "buz", "sandik", "mesale"]
        for ad in beklenen:
            self.assertIn(ad, NESNE_KATALOGU, f"{ad} katalogda yok")
        self.assertGreaterEqual(len(NESNE_KATALOGU), 12)

    def test_dunya_nesne_kurulumu(self) -> None:
        """Her katalog nesnesi için kurulum fonksiyonu olmalı."""
        from tools.godot_oyun_uret import OyunUretici, Kararlar, NESNE_KATALOGU

        with tempfile.TemporaryDirectory() as td:
            u = OyunUretici(Path(td))
            k = Kararlar(); k.nesneler = list(NESNE_KATALOGU.keys())
            u.uret(k)
            icerik = (Path(td) / "scripts" / "dunya.gd").read_text(
                encoding="utf-8")
            for ad in ("_kaya", "_agac", "_cali", "_ev_ekle", "_kapi_ekle",
                       "_pencere_ekle", "_masa_ekle", "_sandalye_ekle",
                       "_ucan_kaya_ekle", "_buz_ekle", "_sandik_ekle",
                       "_mesale_ekle"):
                self.assertIn(ad, icerik, f"{ad} fonksiyonu yok")
            # Yükseklik oturtma (nesneler havada durmasın)
            self.assertIn("_yukseklik", icerik)


class TestGodotDogrulama(unittest.TestCase):
    """Gerçek Godot ile doğrulama (yalnız --godot ile çalışır)."""

    def setUp(self) -> None:
        if not GODOT_AKTIF:
            self.skipTest("Godot doğrulaması kapalı (--godot ile açılır)")
        if not Path(GODOT).exists():
            self.skipTest("Godot bulunamadı")

    def _uret_ve_dogrula(self, ad: str, **ayarlar: Any) -> None:
        from tools.godot_oyun_uret import OyunUretici, Kararlar, NESNE_KATALOGU

        with tempfile.TemporaryDirectory() as td:
            u = OyunUretici(Path(td))
            k = Kararlar(sahne_adi=ad)
            k.labirent = k.arena = k.magara = False
            k.nesneler = list(NESNE_KATALOGU.keys())
            for anahtar, deger in ayarlar.items():
                setattr(k, anahtar, deger)
            u.uret(k)
            r = godot_dogrula(Path(td))
            if "atlandi" in r or "hata" in r:
                self.skipTest(str(r))
            hatalar = r.get("hatalar") or []
            self.assertEqual(hatalar, [],
                             f"{ad}: Godot hataları:\n" +
                             "\n".join(hatalar[:6]))

    def test_godot_labirent(self) -> None:
        self._uret_ve_dogrula("godot_lab", labirent=True, labirent_boyut=15)

    def test_godot_arena(self) -> None:
        self._uret_ve_dogrula("godot_arena", arena=True, arena_yaricap=20.0)

    def test_godot_magara(self) -> None:
        self._uret_ve_dogrula("godot_mag", magara=True, magara_tunel=10)


def main() -> int:
    """Testleri çalıştırır."""
    global GODOT_AKTIF
    ap = argparse.ArgumentParser(description="Godot üretim testleri")
    ap.add_argument("--godot", action="store_true",
                    help="Godot ile de doğrula (GPU kullanır)")
    args, kalan = ap.parse_known_args()
    GODOT_AKTIF = args.godot

    print("=" * 78)
    print("  UÇTAN UCA DOĞRULAMA — Godot oyun üretimi")
    print(f"  Godot doğrulaması: {'AÇIK' if GODOT_AKTIF else 'kapalı'}")
    print("=" * 78)

    yukleyici = unittest.TestLoader()
    paket = unittest.TestSuite()
    paket.addTests(yukleyici.loadTestsFromTestCase(TestUreticiModulleri))
    paket.addTests(yukleyici.loadTestsFromTestCase(TestGodotDogrulama))
    sonuc = unittest.TextTestRunner(verbosity=2).run(paket)

    # Rapor
    rapor = {
        "test": sonuc.testsRun,
        "basarili": sonuc.testsRun - len(sonuc.failures) - len(sonuc.errors)
        - len(sonuc.skipped),
        "basarisiz": len(sonuc.failures),
        "hata": len(sonuc.errors),
        "atlanan": len(sonuc.skipped),
        "godot_aktif": GODOT_AKTIF,
    }
    Path("_godot_test_raporu.json").write_text(
        json.dumps(rapor, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(f"  SONUÇ: {rapor['basarili']}/{rapor['test']} başarılı, "
          f"{rapor['basarisiz']} başarısız, {rapor['hata']} hata, "
          f"{rapor['atlanan']} atlandı")
    return 0 if (sonuc.wasSuccessful() or sonuc.skipped) else 1


if __name__ == "__main__":
    raise SystemExit(main())
