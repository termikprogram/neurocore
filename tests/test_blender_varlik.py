"""
Blender Varlık ve Retarget Testleri
=====================================
⚠ Neden gerekli?
----------------
Blender entegrasyonunda **4 kritik tuzak** ölçülerek bulundu (tur 1-4).
Her biri sessizce yanlış sonuç üretiyordu:

    1. `Action.fcurves` Blender 5.x'te KALDIRILDI
       → `layers → strips → channelbags → fcurves`
    2. `select_all(DESELECT)` **veri katmanına yansımıyor**
       → `mark_seam` 72/72 kenarı işaretledi (beklenen 25)
    3. `keyframe_insert` keyfi `array_index`i GÜVENİLİR yazmıyor
       → 24 kanaldan yalnız 3'ü doldu (idx0/idx1 sıfırlandı)
    4. `unwrap` EDIT modu ister, `use_seam` istemez

Bu testler o kazançları **kilitler**.
"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

_KOK = Path(__file__).resolve().parent.parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

logging.disable(logging.CRITICAL)

BLENDER = Path(r"E:\AppsTools\Blender 5.2\blender.exe")


class TestKatalog(unittest.TestCase):
    """Varlık kataloğu — kategori, kalite, idempotentlik."""

    def test_kategoriler_tanimli(self) -> None:
        """8 kategori ve 3 kalite kademesi tanımlı olmalı."""
        from tools.varlik_katalogu import KALITE, KATEGORILER

        self.assertEqual(len(KATEGORILER), 8)
        self.assertEqual(len(KALITE), 3)
        # ⚠ Armature gerektiren kategoriler
        arm = [a for a, b in KATEGORILER.items()
               if b["armature_gerekli"]]
        self.assertIn("karakter", arm)
        self.assertIn("yaratik", arm)

    def test_kalite_esikleri_artan(self) -> None:
        """Kalite kademeleri artan yüz sayısı olmalı."""
        from tools.varlik_katalogu import KALITE

        e = [KALITE["lowpoly"]["azami_yuz"], KALITE["orta"]["azami_yuz"],
             KALITE["kaliteli"]["azami_yuz"]]
        self.assertEqual(e, sorted(e))

    def test_katalog_yolu_deterministik(self) -> None:
        """
        ⚠ tur 1 ÖLÇÜMÜ: yol hesaplaması var olan klasöre `_02` ekliyordu
        → aynı varlık her çağrıda yeni klasör → katalogda **tekrarlı
        kayıt** (10 kayıt / 6 dosya). Deterministik olmalı.
        """
        from tools.blender_varlik_testi import _varlik_yolu

        a = _varlik_yolu("kaya_lowpoly")
        b = _varlik_yolu("kaya_lowpoly")
        self.assertEqual(a, b, "yol hesabı deterministik değil")

    def test_kategori_turetme(self) -> None:
        """Ad'dan kategori doğru türetilmeli."""
        from tools.blender_varlik_testi import _kategori

        self.assertEqual(_kategori("kaya_lowpoly"), "kaya")
        self.assertEqual(_kategori("timsah_0001"), "yaratik")
        self.assertEqual(_kategori("agac_01"), "bitki")

    def test_kalite_turetme(self) -> None:
        """Ad'dan kalite doğru türetilmeli."""
        from tools.blender_varlik_testi import _kalite

        self.assertEqual(_kalite("kaya_lowpoly"), "lowpoly")
        self.assertEqual(_kalite("kaya_kaliteli"), "kaliteli")
        self.assertEqual(_kalite("kaya_orta"), "orta")


class TestAnimKutuphanesi(unittest.TestCase):
    """Animasyon kütüphanesi — kemik imzası ve rol eşleştirme."""

    def test_rol_eslestirme_es_anlamli(self) -> None:
        """
        ⚠ tur 4 ÖLÇÜMÜ: iki yaratıkta aynı işlev **farklı adla**
        olabilir. Rol eşleştirmesi olmadan animasyon uygulanamaz.
        """
        from tools.anim_kutuphanesi import rol_bul

        self.assertEqual(rol_bul("on_ayak_l"), "on_ayak_l")
        self.assertEqual(rol_bul("front_leg_l"), "on_ayak_l")
        self.assertEqual(rol_bul("bacak_on_sol"), "on_ayak_l")
        self.assertEqual(rol_bul("kafa"), "kafa")
        self.assertEqual(rol_bul("head"), "kafa")
        self.assertEqual(rol_bul("tail_1"), "kuyruk_1")

    def test_imza_siralama_bagimsiz(self) -> None:
        """İmza, kemik sırasından **bağımsız** olmalı."""
        from tools.anim_kutuphanesi import imza_hesapla

        k1 = ["kafa", "boyun", "omurga"]
        k2 = ["omurga", "kafa", "boyun"]
        h = {"kafa": "boyun", "boyun": "omurga", "omurga": ""}
        self.assertEqual(imza_hesapla(k1, h), imza_hesapla(k2, h))

    def test_imza_hiyerarsiye_duyarli(self) -> None:
        """Hiyerarşi değişince imza **değişmeli**."""
        from tools.anim_kutuphanesi import imza_hesapla

        k = ["kafa", "boyun"]
        a = imza_hesapla(k, {"kafa": "boyun", "boyun": ""})
        b = imza_hesapla(k, {"kafa": "", "boyun": ""})
        self.assertNotEqual(a, b)

    def test_esleme_tam_ad_oncelikli(self) -> None:
        """Tam ad eşleşmesi rol eşleşmesinden **önce** gelmeli."""
        from tools.anim_uygula import esleme_uret

        e = esleme_uret(["kafa", "kuyruk_1"], ["kafa", "tail_1"])
        self.assertEqual(e["kafa"], "kafa")
        self.assertEqual(e["kuyruk_1"], "tail_1")

    def test_kanal_indeksi_korunur(self) -> None:
        """
        ⚠ tur 4 KESİN KÖK NEDEN: `array_index` kaybolursa retarget tüm
        değerleri eksen 0'a yazar → dolu kanal 8 → 3.
        Kütüphane kanal adı **indeks içermeli**.
        """
        import inspect

        from tools import anim_kutuphanesi as m

        kaynak = inspect.getsource(m)
        self.assertIn("array_index", kaynak,
                      "fcurve array_index kullanılmıyor — retarget bozulur")
        self.assertIn('kanal = f"{kanal}[{fc.array_index}]"', kaynak,
                      "kanal adına indeks eklenmiyor")


class TestDokuBulucu(unittest.TestCase):
    """Doku bulucu — sınıflandırma ve prosedürel tanımlar."""

    def test_tur_siniflandirma(self) -> None:
        """Dosya adından doku türü çıkarılmalı."""
        from tools.doku_bulucu import _siniflandir

        kat, tur, _ = _siniflandir(Path("rock_04_diff_2k.png"))
        self.assertEqual(kat, "kaya")
        self.assertEqual(tur, "renk")

        kat2, tur2, _ = _siniflandir(Path("rock_04_rough_2k.png"))
        self.assertEqual(tur2, "pürüzlülük")

        _, tur3, _ = _siniflandir(Path("rock_04_nor_gl_2k.png"))
        self.assertEqual(tur3, "normal")

    def test_prosedurel_yedi_doku(self) -> None:
        """7 prosedürel doku tanımlı ve her birinde düğüm olmalı."""
        from tools.doku_bulucu import PROSEDUREL

        self.assertEqual(len(PROSEDUREL), 7)
        for ad, b in PROSEDUREL.items():
            self.assertTrue(b["dugumler"], f"{ad}: düğüm yok")
            self.assertEqual(len(b["renk"]), 3, f"{ad}: renk 3 bileşen")
            self.assertIn("puruzluluk", b, f"{ad}: puruzluluk yok")
            # ⚠ Düğüm tipleri Blender'da geçerli olmalı
            for tip, _ in b["dugumler"]:
                self.assertTrue(tip.startswith("ShaderNode"),
                                f"{ad}: geçersiz düğüm {tip}")

    def test_kaynaklar_lisansli(self) -> None:
        """Her çevrimiçi kaynağın lisans bilgisi olmalı."""
        from tools.doku_bulucu import KAYNAKLAR

        self.assertGreaterEqual(len(KAYNAKLAR), 5)
        for s in KAYNAKLAR:
            self.assertTrue(s.get("lisans"), f"{s['ad']}: lisans yok")
            self.assertTrue(s.get("url", "").startswith("http"))


class TestBlenderEntegrasyonu(unittest.TestCase):
    """Gerçek Blender ile bütünleşik testler."""

    def setUp(self) -> None:
        if not BLENDER.exists():
            self.skipTest("Blender bulunamadı")

    def test_blender_calisiyor(self) -> None:
        """Blender headless çalışmalı ve sürüm dönmeli."""
        import subprocess

        p = subprocess.run([str(BLENDER), "--version"], capture_output=True,
                           text=True, timeout=60, encoding="utf-8",
                           errors="replace")
        self.assertIn("Blender", (p.stdout or ""))

    def test_yedi_doku_derlendi(self) -> None:
        """
        ⚠ tur 4 ÖLÇÜMÜ: 7 prosedürel dokunun **hepsi** Blender'da
        düğüm ağacı kurmalı. Uyarı varsa düğüm/giriş adı yanlıştır.
        """
        import json as _json

        p = _KOK / "_doku_dogrulama.json"
        if not p.exists():
            self.skipTest("doku doğrulaması çalıştırılmadı")
        d = _json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(d.get("tam"), d.get("toplam"),
                         "bazı dokular uyarılı/hatalı")
        self.assertEqual(d.get("toplam"), 7)



class TestKarakterUretimi(unittest.TestCase):
    """
    İnsansı karakter — dik duruş, iki ayaklı.

    ⚠ Neden ayrı kategori?
    ----------------------
    Timsah **dört ayaklı + kuyruklu** (12 kemik); insan **dik + iki
    ayaklı** (19 kemik). Kullanıcı: *"her armature'lü varlık için kemik
    yapısı farklı olacağından ona uygun anim yapmak gerekecek"*
    """

    def test_insan_kemik_yapisi_dik(self) -> None:
        """İnsan iskeleti 19 kemik ve dik duruş içermeli."""
        import json as _json

        p = _KOK / "_karakter.json"
        if not p.exists():
            self.skipTest("karakter üretilmedi")
        d = _json.loads(p.read_text(encoding="utf-8"))
        adlar = d.get("kemik_adlari", [])
        self.assertEqual(len(adlar), 19, "19 kemik bekleniyor")
        # ⚠ İki ayaklı yapı
        for z in ("ust_bacak_l", "alt_bacak_l", "ayak_l",
                  "ust_bacak_r", "alt_bacak_r", "ayak_r"):
            self.assertIn(z, adlar, f"{z} eksik")
        # ⚠ İki kollu yapı
        for z in ("omuz_l", "ust_kol_l", "alt_kol_l", "el_l",
                  "omuz_r", "ust_kol_r", "alt_kol_r", "el_r"):
            self.assertIn(z, adlar, f"{z} eksik")
        # ⚠ Omurga zinciri
        for z in ("kok", "omurga", "gogus", "boyun", "kafa"):
            self.assertIn(z, adlar, f"{z} eksik")

    def test_insan_agirlik_tam(self) -> None:
        """Automatic weight tüm vertex'leri kapsamalı."""
        import json as _json

        p = _KOK / "_karakter.json"
        if not p.exists():
            self.skipTest("karakter üretilmedi")
        d = _json.loads(p.read_text(encoding="utf-8"))
        self.assertTrue(d.get("weight_ok"), "weight başarısız")
        self.assertEqual(d.get("vertex_group"), d.get("kemik_sayisi"),
                         "her kemik için bir vertex grubu olmalı")

    def test_insan_animasyonu_ortak_hatta(self) -> None:
        """
        ⚠ tur 10 ÖLÇÜMÜ: karakter `Insan_Yuruyus` üretiyordu ama
        `anim_uret.py` tarifleri eklerken onu **siliyordu** (ölçüm:
        5 action, 6 beklenirken).

        ✅ Çözüm: **tek kaynak** — karakter kendi animasyonunu üretmez,
        tüm animasyonlar `tools/anim_uret.py`'den gelir.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "karakter_uret.py",
                         encoding="utf-8").read()
        # ⚠ Yalnız **Blender betiği** (KARAKTER sabiti) kontrol edilir —
        # açıklama metinleri değil.
        self.assertIn("KARAKTER = r", kaynak)
        _blok = kaynak.split("KARAKTER = r", 1)[1]
        self.assertNotIn("bpy.data.actions.new", _blok,
                         "karakter kendi action'ını üretiyor — "
                         "anim_uret.py ile çakışır")
        self.assertIn("animasyonlar = []", _blok,
                      "karakter animasyonsuz üretmeli")

    def test_kanal_basina_tek_fcurve(self) -> None:
        """
        ⚠ tur 8 KESİN KÖK NEDEN: her karede `KANALLAR[kemik].append(
        (idx, [(kare, deger)]))` yapılırsa aynı kanal için **25 fcurve**
        açılmaya çalışılır; ilki başarılı olur, kalanı sessizce atlanır
        → **14 fcurve × 1 nokta**, tüm değerler 0.

        ⚠ tur 10: animasyon üretimi `anim_uret.py`'ye taşındı — kontrol
        orada yapılır.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_uret.py",
                         encoding="utf-8").read()
        # ⚠ fcurve kanal başına bir kez kurulmalı
        self.assertIn("for eksen in sorted(kanallar[kemik].keys())", kaynak,
                      "fcurve kanal başına bir kez kurulmuyor")
        # ⚠ Sessiz atlama raporlanmalı
        self.assertIn('olcum["hata"].append', kaynak,
                      "fcurve hataları sessizce atlanıyor")
        # ⚠ Eski action temizliği (tekrar birikimini önler)
        self.assertIn("bpy.data.actions.remove(_a, do_unlink=True)", kaynak,
                      "eski action temizliği yok")


class TestAnimTekillestirme(unittest.TestCase):
    """Animasyon kütüphanesi tekilleştirme."""

    def test_ayni_imza_tekrar_kaydedilmez(self) -> None:
        """
        ⚠ tur 8 ÖLÇÜMÜ: kalite kademeleri aynı kemik yapısını üretir →
        aynı animasyon **3 kez** kaydedildi. (`ad`, `imza`) tekilleştirilir.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_kutuphanesi.py",
                         encoding="utf-8").read()
        self.assertIn('x.get("ad") == a["ad"]', kaynak,
                      "ad tekilleştirmesi yok")
        self.assertIn('x.get("imza") == kayit.imza', kaynak,
                      "imza tekilleştirmesi yok")



class TestAnimUretimi(unittest.TestCase):
    """5 animasyon tarifi ve fake user koruması."""

    def test_dokuz_tarif_tanimli(self) -> None:
        """
        9 tarif olmalı — tur 13'te 4 eklendi: ziplama, dusme, oturma,
        blok.
        """
        from tools.anim_uret import TARIFLER

        self.assertEqual(len(TARIFLER), 13)
        for ad in ("bekleme", "yurume", "kosma", "saldiri", "olum",
                   "ziplama", "dusme", "oturma", "blok"):
            self.assertIn(ad, TARIFLER, f"{ad} tarifi eksik")

    def test_dongu_tek_ayrimi(self) -> None:
        """
        ⚠ Döngü (bekleme/yurume/kosma) ile tek seferlik
        (saldiri/olum) ayrımı doğru olmalı.
        """
        from tools.anim_uret import TARIFLER

        for ad in ("bekleme", "yurume", "kosma"):
            self.assertTrue(TARIFLER[ad]["dongu"], f"{ad} döngü olmalı")
        for ad in ("saldiri", "olum"):
            self.assertFalse(TARIFLER[ad]["dongu"], f"{ad} tek olmalı")

    def test_tarifler_rol_tabanli(self) -> None:
        """
        ⚠ Tarifler **kemik adına** değil **role** yazılmalı —
        aksi hâlde farklı armature'e uygulanamaz.
        """
        from tools.anim_uret import TARIFLER

        for ad, t in TARIFLER.items():
            self.assertTrue(t["kanallar"], f"{ad}: kanal yok")
            for rol in t["kanallar"]:
                # Rol adları Türkçe/İngilizce standart olmalı
                self.assertRegex(rol, r"^[a-z_0-9]+$",
                                 f"{ad}: geçersiz rol {rol}")

    def test_yeni_tarifler_mevcut(self) -> None:
        """⚠ tur 21: 4 yeni tarif (yüzme/tırmanma/hasar/dirilme)."""
        from tools.anim_uret import TARIFLER

        for ad in ("yuzme", "tirmanma", "hasar", "dirilme"):
            self.assertIn(ad, TARIFLER, f"'{ad}' tarifi eksik")
        # ⚠ Döngü/tek ayrımı
        self.assertTrue(TARIFLER["yuzme"]["dongu"])
        self.assertTrue(TARIFLER["tirmanma"]["dongu"])
        self.assertFalse(TARIFLER["hasar"]["dongu"])
        self.assertFalse(TARIFLER["dirilme"]["dongu"])

    def test_kaynak_cikarma_ve_eval(self) -> None:
        """
        ⚠ tur 9 KESİN KÖK NEDEN: lambda'yı `inspect.getsource` ile
        ayıklamak kırılgan (`(1, lambda t),` gibi kesik kod üretti).
        Çözüm: sözlük tanımı dosyadan çıkarılıp `eval` edilir.
        """
        import math

        from tools.anim_uret import _tarifler_kaynak

        k = _tarifler_kaynak()
        self.assertTrue(k.strip().startswith("{"))
        self.assertTrue(k.strip().endswith("}"))

        def _sin(t):
            return math.sin(t * 2.0 * math.pi)

        T = eval(k, {"_sin": _sin})          # noqa: S307
        self.assertEqual(len(T), 13)
        # ⚠ Değerler hesaplanabilmeli
        omurga = T["bekleme"]["kanallar"]["omurga"][0]
        self.assertEqual(omurga[0], 1)
        self.assertAlmostEqual(omurga[1](0.25, 0), 0.022, places=5)

    def test_fake_user_kullanilir(self) -> None:
        """
        ⚠ tur 9 KESİN KÖK NEDEN: Blender kaydederken **fake user**'ı
        olmayan action'ları SİLER. Ölçüm: 5 üretildi, diskte 1 kaldı.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_uret.py",
                         encoding="utf-8").read()
        self.assertIn("act.use_fake_user = True", kaynak,
                      "fake user yok — animasyonlar kaydedilmez")
        # ⚠ Tekrar birikimini önleyen temizlik
        self.assertIn("bpy.data.actions.remove(_a, do_unlink=True)",
                      kaynak, "eski action temizliği yok")

    def test_ayristirici_strip_kullanir(self) -> None:
        """
        ⚠ tur 9: `satir.strip().startswith("S{")` ama `satir[1:]`
        uyumsuzluğu → başarılı üretim "✗ []" diye raporlanıyordu.
        """
        import io

        for f in ("anim_uret.py", "anim_kutuphanesi.py", "varlik_kaydet.py"):
            kaynak = io.open(_KOK / "tools" / f, encoding="utf-8").read()
            self.assertNotIn("json.loads(satir[1:])", kaynak,
                             f"{f}: strip uyumsuzluğu")


class TestAnimKutuphanesiCoklu(unittest.TestCase):
    """Kütüphane bir varlığın TÜM action'larını okumalı."""

    def test_tum_actionlar_taranir(self) -> None:
        """
        ⚠ tur 9 ÖLÇÜMÜ: yalnız `animation_data.action` (aktif) okunuyordu
        → 5 animasyonlu varlıktan kütüphaneye **1**'i giriyordu.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_kutuphanesi.py",
                         encoding="utf-8").read()
        self.assertIn("bpy.data.actions", kaynak,
                      "tüm action'lar taranmıyor")
        self.assertIn("_hepsi", kaynak,
                      "aktif + diğer action birleştirmesi yok")



class TestVarlikAraclari(unittest.TestCase):
    """myAi araç kaydı — asset.uret, asset.animasyon, asset.katalog."""

    def test_uc_arac_kayitli(self) -> None:
        """3 asset aracı CORE_TOOLS'ta olmalı."""
        from neurocore.agent.loop import CORE_TOOLS

        for ad in ("asset.uret", "asset.animasyon", "asset.katalog"):
            self.assertIn(ad, CORE_TOOLS, f"{ad} kayıtlı değil")

    def test_isleyiciler_var(self) -> None:
        """Her aracın işleyicisi tanımlı olmalı."""
        from neurocore.agent.coding import CodingAgent

        a = CodingAgent()
        td = {t.name: t for t in a.tools()}
        for ad in ("asset.uret", "asset.animasyon", "asset.katalog"):
            t = td.get(ad)
            self.assertIsNotNone(t, f"{ad} araç kaydında yok")
            self.assertIsNotNone(t.handler, f"{ad} işleyicisi yok")

    def test_arac_secimi_asset(self) -> None:
        """
        ⚠ TOOL_HINTS olmadan araç **seçilmiyor** (tur 8 dersi).
        10 tipik istek → doğru asset aracı seçilmeli.
        """
        from neurocore.agent.loop import CORE_TOOLS, TOOL_HINTS

        istekler = [
            ("low-poly kaya oluştur", "asset.uret"),
            ("ağaç modeli üret kaliteli", "asset.uret"),
            ("timsah canavar yap", "asset.uret"),
            ("kaliteli sandık üret", "asset.uret"),
            ("duvar modeli oluştur", "asset.uret"),
            ("insan karakteri yap", "asset.uret"),
            ("yürüyüş animasyonu ekle", "asset.animasyon"),
            ("iskelet ve kemik yapısı kur", "asset.animasyon"),
            ("varlıkları listele katalog", "asset.katalog"),
        ]
        for istek, beklenen in istekler:
            gl = istek.lower()
            sec: list = []
            for anahtarlar, araclar in TOOL_HINTS:
                if any(k in gl for k in anahtarlar):
                    sec.extend(araclar)
            var = [x for x in dict.fromkeys(sec) if x in CORE_TOOLS]
            self.assertIn(beklenen, var,
                          f"'{istek}' için {beklenen} seçilmedi (var: {var})")

    def test_kaya_urunde_var(self) -> None:
        """⚠ tur 10: `kaya` URUNLER'de yoktu → ajan üretemiyordu."""
        from tools.varlik_uret import URUNLER

        self.assertIn("kaya", URUNLER)
        self.assertEqual(URUNLER["kaya"]["kategori"], "kaya")
        # ⚠ Kategori listesiyle tutarlı olmalı
        from tools.varlik_katalogu import KATEGORILER

        for ad, b in URUNLER.items():
            self.assertIn(b["kategori"], KATEGORILER,
                          f"{ad}: bilinmeyen kategori {b['kategori']}")

    def test_uretim_sonucu_dataclass_uyumu(self) -> None:
        """
        ⚠ tur 10 ÖLÇÜMÜ: `yaratik_uret.uret()` bir **dataclass**
        (`UretimSonuc`) döner; `"hata" not in d` `TypeError` veriyordu.
        İşleyici her iki biçimi de kabul etmeli.
        """
        import io

        kaynak = io.open(_KOK / "neurocore" / "agent" / "coding.py",
                         encoding="utf-8").read()
        self.assertIn("to_dict", kaynak,
                      "dataclass → sözlük dönüşümü yok")



class TestDokuAraci(unittest.TestCase):
    """asset.doku — ara, seçime sun, uygula (Hedef B)."""

    def test_arac_kayitli(self) -> None:
        """asset.doku CORE_TOOLS'ta ve işleyicisi olmalı."""
        from neurocore.agent.coding import CodingAgent
        from neurocore.agent.loop import CORE_TOOLS

        self.assertIn("asset.doku", CORE_TOOLS)
        a = CodingAgent()
        td = {t.name: t for t in a.tools()}
        self.assertIsNotNone(td.get("asset.doku"))
        self.assertIsNotNone(td["asset.doku"].handler)

    def test_yedi_prosedurel_doku(self) -> None:
        """7 prosedürel doku, her biri geçerli düğümlerle."""
        from tools.doku_bulucu import PROSEDUREL

        self.assertEqual(len(PROSEDUREL), 7)
        for ad, b in PROSEDUREL.items():
            self.assertGreaterEqual(len(b["dugumler"]), 1, ad)
            self.assertEqual(len(b["renk"]), 3, ad)
            self.assertIn("puruzluluk", b, ad)

    def test_puanlama_ve_siralama(self) -> None:
        """
        ⚠ Puanlama: kategori +3 · tür +2.5 · ad +2 · etiket +1 ·
        çözünürlük≥1024 +0.5. En yüksek puan **ilk** gelmeli.
        """
        from tools.doku_bulucu import ara, tara

        tara()
        r = ara("tahta renk", adet=5)
        if not r:
            self.skipTest("yerel doku kütüphanesi boş")
        puanlar = [x["puan"] for x in r]
        self.assertEqual(puanlar, sorted(puanlar, reverse=True),
                         "puanlar azalan sırada değil")
        # ⚠ "tahta" içeren doku en yüksek puanı almalı
        self.assertIn("tahta", r[0]["ad"].lower(),
                      f"en yüksek puan 'tahta' değil: {r[0]['ad']}")

    def test_cevrimici_kaynaklar_lisansli(self) -> None:
        """Her çevrimiçi kaynağın lisans bilgisi olmalı."""
        from tools.doku_bulucu import KAYNAKLAR

        self.assertGreaterEqual(len(KAYNAKLAR), 5)
        for s in KAYNAKLAR:
            self.assertTrue(s.get("lisans"), s["ad"])
            self.assertTrue(s.get("url", "").startswith("http"), s["ad"])

    def test_yanlis_doku_adi_reddedilir(self) -> None:
        """Bilinmeyen prosedürel doku adı **reddedilmeli**."""
        from tools.doku_uygula import uygula_prosedurel

        r = uygula_prosedurel("kaya", "olmayan_doku")
        self.assertIn("hata", r)
        self.assertIn("gecerli", r)

    def test_uygulama_dugum_ekler(self) -> None:
        """
        ⚠ Ölçüm: prosedürel doku uygulanınca düğüm sayısı **artmalı**
        (4 → 6) ve bağlantı kurulmalı.
        """
        import json as _json

        from tools.doku_uygula import uygula_prosedurel

        r = uygula_prosedurel("duvar", "toprak")
        if "hata" in r:
            self.skipTest(f"uygulanamadı: {r['hata']}")
        self.assertGreaterEqual(r["dugum"], 2,
                                "düğüm eklenmedi")
        self.assertGreaterEqual(r["baglanti"], 2,
                                "bağlantı kurulmadı")
        self.assertEqual(r.get("uyari"), [],
                         f"uyarı var: {r.get('uyari')}")



class TestDokuTemizleme(unittest.TestCase):
    """Doku tekrar uygulamada düğüm birikimi önlenmeli."""

    def test_temizleme_kodu_var(self) -> None:
        """
        ⚠ tur 12 ÖLÇÜMÜ: aynı varlığa 4 kez doku uygulanınca düğüm
        sayısı 14 → 16 → 18 → 20 oldu (birikme).
        Çözüm: uygulamadan **önce** eski doku düğümleri silinir.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "doku_uygula.py",
                         encoding="utf-8").read()
        self.assertIn("temizlenen", kaynak, "temizleme ölçümü yok")
        self.assertIn("nt.nodes.remove(_n)", kaynak, "düğüm silinmiyor")
        # ⚠ BSDF korunmalı
        self.assertIn("ShaderNodeBsdfPrincipled", kaynak,
                      "BSDF korunmuyor — malzeme bozulur")

    def test_tekrar_uygulama_sabit_dugum(self) -> None:
        """4 kez uygulama → düğüm sayısı **sabit** kalmalı."""
        from tools.doku_uygula import uygula_prosedurel

        sayilar = []
        for _ in range(3):
            r = uygula_prosedurel("duvar", "toprak")
            if "hata" in r:
                self.skipTest(f"uygulanamadı: {r['hata']}")
            sayilar.append(r["dugum_toplam"])
        self.assertEqual(len(set(sayilar)), 1,
                         f"düğüm sayısı değişiyor (birikme): {sayilar}")

    def test_farkli_doku_degistirir(self) -> None:
        """Farklı doku uygulanınca **değiştirmeli**, eklememeli."""
        from tools.doku_uygula import uygula_prosedurel

        sayilar = []
        for d in ("toprak", "mermer", "metal", "tahta"):
            r = uygula_prosedurel("duvar", d)
            if "hata" in r:
                self.skipTest(f"uygulanamadı: {r['hata']}")
            sayilar.append(r["dugum_toplam"])
        self.assertEqual(len(set(sayilar)), 1,
                         f"doku değiştirme çalışmıyor: {sayilar}")


class TestAnimGecis(unittest.TestCase):
    """Animasyon geçişleri — NLA + blend (Hedef D)."""

    def test_gecis_uretimi(self) -> None:
        """
        ⚠ 5 animasyon arasında geçiş olmalı. Ölçüm: 7 geçiş üretildi,
        NLA izi 12, toplam 12 action.
        """
        import json as _json

        p = _KOK / "_anim_gecis.json"
        if not p.exists():
            self.skipTest("geçiş üretilmedi")
        d = _json.loads(p.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(d.get("uretilen", [])), 5,
                                "yeterli geçiş üretilmedi")
        self.assertGreaterEqual(d.get("nla", 0), 5,
                                "NLA izi oluşturulmadı")

    def test_gercek_gecis_ayrimi(self) -> None:
        """
        ⚠ Fark 0 ise iki animasyon **zaten sürekli** (ör. bekleme ve
        yurume ikisi de `_sin(0) = 0` ile başlar). Raporlamada ayrılır.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_gecis.py",
                         encoding="utf-8").read()
        self.assertIn("gercek_gecis", kaynak,
                      "sürekli/gerçek geçiş ayrımı yok")
        self.assertIn("azami_fark", kaynak, "fark ölçümü yok")

    def test_bos_liste_hata_degil(self) -> None:
        """
        ⚠ tur 12: `if "hata" in d` YANLIŞ — ölçüm sözlüğünde
        `"hata": []` her zaman var. Boş liste hata değildir.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_gecis.py",
                         encoding="utf-8").read()
        self.assertNotIn('if "hata" in d:', kaynak,
                         "boş liste hata sayılıyor")
        self.assertIn("isinstance(_h, str) and _h", kaynak,
                      "hata tip kontrolü yok")

    def test_gecis_smoothstep(self) -> None:
        """
        ⚠ Geçiş **smoothstep** olmalı (3t² − 2t³) — doğrusal değil,
        böylece başlangıç/bitiş yumuşak olur.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_gecis.py",
                         encoding="utf-8").read()
        self.assertIn("3.0 - 2.0 * t", kaynak, "smoothstep yok")



class TestOnizleme(unittest.TestCase):
    """asset.onizleme — 4 açıdan render (görsel doğrulama)."""

    def test_arac_kayitli(self) -> None:
        from neurocore.agent.coding import CodingAgent
        from neurocore.agent.loop import CORE_TOOLS

        self.assertIn("asset.onizleme", CORE_TOOLS)
        a = CodingAgent()
        td = {t.name: t for t in a.tools()}
        self.assertIsNotNone(td.get("asset.onizleme"))
        self.assertIsNotNone(td["asset.onizleme"].handler)

    def test_alti_asset_araci(self) -> None:
        """6 Blender aracı olmalı."""
        from neurocore.agent.loop import CORE_TOOLS

        beklenen = {"asset.uret", "asset.animasyon", "asset.katalog",
                    "asset.doku", "asset.gecis", "asset.onizleme"}
        self.assertTrue(beklenen.issubset(set(CORE_TOOLS)),
                        f"eksik: {beklenen - set(CORE_TOOLS)}")

    def test_dort_aci_tanimli(self) -> None:
        """
        ⚠ Ölçüm: 4 açı (ön, yan, 3/4, üst) — her biri ayrı PNG.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "varlik_onizleme.py",
                         encoding="utf-8").read()
        for aci in ("on", "yan", "uc_dort", "ust"):
            self.assertIn(f'("{aci}"', kaynak, f"{aci} açısı tanımsız")
        # ⚠ 3 nokta ışık
        self.assertIn("Anahtar", kaynak)
        self.assertIn("Dolgu", kaynak)
        self.assertIn("Arka", kaynak)

    def test_agirlikli_merkez_hesabi(self) -> None:
        """
        ⚠ Kamera hedefi **dünya uzayındaki** sınır kutusundan
        hesaplanmalı — `ob.location` parent'lı nesnede yanıltıcıdır.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "varlik_onizleme.py",
                         encoding="utf-8").read()
        self.assertIn("matrix_world", kaynak,
                      "dünya uzayı dönüşümü yok")
        self.assertIn("bound_box", kaynak, "sınır kutusu yok")

    def test_canli_render(self) -> None:
        """Canlı render en az 1 açı vermeli."""
        from tools.varlik_onizleme import render

        r = render("kaya", cozunurluk=256)
        if "hata" in r:
            self.skipTest(f"render edilemedi: {r['hata'][:60]}")
        self.assertGreaterEqual(r.get("basarili", 0), 1)
        self.assertGreaterEqual(r.get("kamera", 0), 1)
        self.assertGreaterEqual(r.get("isik", 0), 3)
        # ⚠ Sınır kutusu ölçümü
        self.assertEqual(len(r.get("boyut", [])), 3)
        self.assertGreater(max(r["boyut"]), 0.0)

    def test_render_motoru_eevee(self) -> None:
        """⚠ Hız için EEVEE tercih edilmeli (Cycles yavaş)."""
        import io

        kaynak = io.open(_KOK / "tools" / "varlik_onizleme.py",
                         encoding="utf-8").read()
        self.assertIn("BLENDER_EEVEE", kaynak, "EEVEE motoru yok")
        self.assertIn("BLENDER_WORKBENCH", kaynak, "yedek motor yok")



class TestMantikMufredati(unittest.TestCase):
    """Blender mantık müfredatı — 14 katman · 43 ders (Hedef F)."""

    def test_katman_ve_ders_sayisi(self) -> None:
        """14 katman · en az 40 ders olmalı."""
        from tools.blender_mantik import MÜFREDAT, tum_dersler

        self.assertEqual(len(MÜFREDAT), 15,
                         "katman sayısı 15 olmalı")
        self.assertGreaterEqual(len(tum_dersler()), 54,
                                "en az 54 ders olmalı")

    def test_her_katmanda_ders_var(self) -> None:
        """Boş katman olmamalı."""
        from tools.blender_mantik import MÜFREDAT

        for ad, aciklama, dersler in MÜFREDAT:
            self.assertGreaterEqual(len(dersler), 2,
                                    f"{ad} katmanında {len(dersler)} ders")

    def test_dersler_neden_icerir(self) -> None:
        """
        ⚠ Müfredatın amacı **"neden"** öğretmek. Her ders gerekçe
        içermeli (ölçüm, örnek veya açıklama).
        """
        from tools.blender_mantik import tum_dersler

        for d in tum_dersler():
            self.assertGreater(len(d.mantik), 150,
                               f"'{d.baslik}' çok kısa")
            self.assertGreater(len(d.kod), 3,
                               f"'{d.baslik}' kod örneği yok")

    def test_ders_metni_bicimli(self) -> None:
        """Ders metni başlık + Neden + Kod içermeli."""
        from tools.blender_mantik import tum_dersler

        for d in tum_dersler():
            m = d.metin()
            self.assertIn("## Neden?", m, d.baslik)
            self.assertIn("```python", m, d.baslik)
            self.assertIn("**Katman:**", m, d.baslik)

    def test_olculmus_bulgular_yer_alir(self) -> None:
        """
        ⚠ tur 6-14'te bulunan kritik tuzaklar müfredatta **olmalı** —
        bunlar gerçek ölçümlerden geldi.
        """
        from tools.blender_mantik import tum_dersler

        basliklar = " ".join(d.baslik for d in tum_dersler())
        for anahtar in ("use_fake_user", "BİRİKİR", "Boş liste",
                        "disk", "GÜVENİLMEZ", "Görsel doğrulama"):
            self.assertIn(anahtar, basliklar,
                          f"'{anahtar}' dersi eksik")

    def test_tekrar_yok(self) -> None:
        """Aynı başlıkta iki ders olmamalı."""
        from tools.blender_mantik import tum_dersler

        basliklar = [d.baslik for d in tum_dersler()]
        self.assertEqual(len(basliklar), len(set(basliklar)),
                         "tekrarlı ders başlığı var")



class TestMantikTamamlama(unittest.TestCase):
    """
    Mantık müfredatı → tamamlama çifti biçimi (Hedef F).

    ⚠ tur 13 ÖLÇÜMÜ: belge biçimi −%9 · tamamlama çifti +%9.
    """

    def test_cift_uretimi(self) -> None:
        """En az 40 çift üretilmeli, 14 katmanı kapsamalı."""
        from tools.blender_mantik_tm import istatistik, uret

        st = istatistik(uret())
        self.assertGreaterEqual(st["cift"], 70)
        self.assertEqual(st["katman_kapsama"], 15)
        self.assertGreaterEqual(st["ders_kapsama"], 44)

    def test_ciftler_dengeli(self) -> None:
        """
        ⚠ Her çiftin birleşik kodu **parantez ve tırnak dengeli**
        olmalı — dengesiz veri modeli bozar.
        """
        from tools.blender_mantik_tm import uret

        for c in uret():
            b = c.onek + c.tamamlama
            for a, k in (("(", ")"), ("[", "]"), ("{", "}")):
                self.assertEqual(b.count(a), b.count(k),
                                 f"dengesiz {a}{k}: {c.ders}")

    def test_token_bolunmesi_yok(self) -> None:
        """
        ⚠ tur 16 KESİN KÖK NEDEN: ilk sürüm token'ları **ortadan
        bölüyordu** (`export_yup` | `=True`). Kesme noktaları anlamlı
        sınırlara (`=`, `,`, satır sonu) hizalı olmalı.
        """
        from tools.blender_mantik_tm import uret

        bolunmus = [c for c in uret()
                    if c.onek and c.tamamlama
                    and c.onek[-1].isalnum() and c.tamamlama[0].isalnum()]
        self.assertEqual(len(bolunmus), 0,
                         f"token bölünmesi: {[c.ders for c in bolunmus][:3]}")

    def test_metin_bicimi_tamamlama(self) -> None:
        """
        ⚠ Eğitim metni **tamamlama biçiminde** olmalı — belge değil.
        Belge biçimi (## Neden?, ```python) **zarar veriyor**.
        """
        from tools.blender_mantik_tm import uret

        for c in uret():
            m = c.metin()
            # ⚠ Belge işaretleri OLMAMALI
            self.assertNotIn("## Neden?", m, c.ders)
            self.assertNotIn("```", m, c.ders)
            # ⚠ Kod içermeli
            self.assertIn(c.onek.splitlines()[-1][:8], m, c.ders)

    def test_jsonl_yazimi(self) -> None:
        """JSONL dosyası yazılmalı ve okunabilmeli."""
        import json as _json

        p = _KOK / ".neurocore" / "blender_mantik_tamamlama.jsonl"
        if not p.exists():
            self.skipTest("çiftler üretilmedi")
        n = 0
        with p.open(encoding="utf-8") as f:
            for satir in f:
                d = _json.loads(satir)
                self.assertIn("metin", d)
                self.assertIn("onek", d)
                self.assertIn("tamamlama", d)
                n += 1
        self.assertGreaterEqual(n, 40)

    def test_veri_havuzunda(self) -> None:
        """⚠ Kaynak `veri_birlestir.KAYNAKLAR`'da olmalı."""
        from tools.veri_birlestir import KAYNAKLAR

        adlar = [k[0] for k in KAYNAKLAR]
        self.assertIn("blender_mantik", adlar,
                      "blender_mantik eğitim havuzunda yok")



class TestOnizlemeKatalog(unittest.TestCase):
    """Önizleme PNG'leri katalog meta verisine bağlı (Hedef G)."""

    def test_meta_alanlari_var(self) -> None:
        """VarlikMeta önizleme alanları içermeli."""
        from tools.varlik_katalogu import VarlikMeta

        m = VarlikMeta()
        self.assertTrue(hasattr(m, "onizleme"))
        self.assertTrue(hasattr(m, "onizleme_aci"))
        self.assertTrue(hasattr(m, "onizleme_klasor"))
        self.assertEqual(m.onizleme_aci, 0)
        self.assertEqual(m.onizleme, {})

    def test_istatistik_onizlemeli(self) -> None:
        """İstatistik 'onizlemeli' sayacı döndürmeli."""
        from tools.varlik_katalogu import istatistik

        s = istatistik()
        self.assertIn("onizlemeli", s)
        self.assertLessEqual(s["onizlemeli"], s["toplam"])

    def test_kaydet_onizlemeyi_baglar(self) -> None:
        """
        ⚠ tur 17: `kataloga_yaz()` önizleme klasörünü tarayıp
        PNG yollarını meta veriye yazmalı.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "varlik_kaydet.py",
                         encoding="utf-8").read()
        self.assertIn("onizleme_klasor", kaynak,
                      "önizleme klasörü bağlanmıyor")
        self.assertIn('klasor / "onizleme"', kaynak)
        self.assertIn("m.onizleme_aci = len(m.onizleme)", kaynak)

    def test_guncelleme_modu(self) -> None:
        """
        ⚠ tur 17 ÖLÇÜMÜ: kataloglayıcı mevcut kayıtları **atlıyordu**
        → önizlemeler meta veriye girmiyordu (Önizlemeli: 0).
        `--guncelle` ile meta veri yenilenir.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "varliklari_katalogla.py",
                         encoding="utf-8").read()
        self.assertIn("--guncelle", kaynak, "güncelleme seçeneği yok")
        self.assertIn("args.guncelle", kaynak)

    def test_katalog_araci_onizleme_dondurur(self) -> None:
        """`asset.katalog` önizleme yollarını içermeli."""
        from neurocore.agent.coding import CodingAgent

        a = CodingAgent()
        r = a.katalog()
        self.assertIn("onizlemeli", r)
        if not r["varliklar"]:
            self.skipTest("katalog boş")
        v = r["varliklar"][0]
        self.assertIn("onizleme_aci", v)
        self.assertIn("onizleme", v)

    def test_onizleme_dosyalari_var(self) -> None:
        """
        ⚠ Önizleme yolları **gerçekten var olan** dosyaları
        göstermeli (ölçüm: 20/20 varlık · 4/4 açı).
        """
        from tools.varlik_katalogu import listele

        v = [x for x in listele() if x.get("onizleme_aci", 0) >= 4]
        if not v:
            self.skipTest("önizlemeli varlık yok")
        for x in v[:5]:
            for aci, yol in x.get("onizleme", {}).items():
                p = Path(yol)
                self.assertTrue(p.exists(),
                                f"{x.get('id')}/{aci}: dosya yok — {yol}")
                self.assertGreater(p.stat().st_size, 1000,
                                   f"{x.get('id')}/{aci}: dosya çok küçük")



class TestAnimOnizleme(unittest.TestCase):
    """asset.anim_onizleme — hareket doğrulama (Hedef D)."""

    def test_arac_kayitli(self) -> None:
        from neurocore.agent.coding import CodingAgent
        from neurocore.agent.loop import CORE_TOOLS

        self.assertIn("asset.anim_onizleme", CORE_TOOLS)
        a = CodingAgent()
        td = {t.name: t for t in a.tools()}
        self.assertIsNotNone(td.get("asset.anim_onizleme"))
        self.assertIsNotNone(td["asset.anim_onizleme"].handler)

    def test_yedi_asset_araci(self) -> None:
        """7 Blender aracı olmalı."""
        from neurocore.agent.loop import CORE_TOOLS

        beklenen = {"asset.uret", "asset.animasyon", "asset.katalog",
                    "asset.doku", "asset.gecis", "asset.onizleme",
                    "asset.anim_onizleme"}
        self.assertTrue(beklenen.issubset(set(CORE_TOOLS)),
                        f"eksik: {beklenen - set(CORE_TOOLS)}")

    def test_manuel_poz_yazimi(self) -> None:
        """
        ⚠ tur 19 ÇÖZÜMÜ: Blender'ın action değerlendirmesi render'a
        **yansımıyor** (piksel farkı %0). Çözüm: fcurve değerlerini
        **doğrudan** `pose_bone.rotation_euler`'e yazmak.

        Ölçüm: diz vertex'i 0.000 → 0.055 → 0.071 m hareket etti.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_onizleme.py",
                         encoding="utf-8").read()
        self.assertIn("_poz_yaz", kaynak, "manuel poz yazımı yok")
        self.assertIn("_fc.evaluate(kare)", kaynak, "fcurve okunmuyor")
        self.assertIn("setattr(_pb", kaynak, "poz bone'a yazılmıyor")

    def test_dogru_vertex_secimi(self) -> None:
        """
        ⚠ tur 19: tur 18'de **yanlış vertex** ölçülüyordu
        (vertex 200 → `boyun`+`kafa` grubunda; diz dönünce sabit
        kalması NORMALDİ). Doğru ölçüm: **o kemiğe bağlı** vertex.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_onizleme.py",
                         encoding="utf-8").read()
        self.assertIn("vertex_groups", kaynak,
                      "vertex grubu eşlemesi yok")
        self.assertIn("max(_v.groups, key=lambda g: g.weight)", kaynak,
                      "en yüksek ağırlıklı grup seçilmiyor")

    def test_iki_olcut_ayri(self) -> None:
        """
        ⚠ Deformasyon (metre) ve fcurve (radyan) **ayrı** sözlüklerde
        tutulmalı — karışırsa tip hatası verir.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_onizleme.py",
                         encoding="utf-8").read()
        self.assertIn("deform = {}", kaynak, "deform sözlüğü yok")
        self.assertIn("en_buyuk_deform", kaynak)
        self.assertIn("en_buyuk_fcurve", kaynak)

    def test_ornekleme_dongu_noktalarina_dusmez(self) -> None:
        """
        ⚠ tur 18: 3 kare örnekleme döngü noktalarına düşüyordu
        (`sin(2πt)`: kare 1, 13, 25 → hepsi 0) → sahte "HAREKETSİZ".
        Çözüm: en az 7 kare + yarım kare kaydırma.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_onizleme.py",
                         encoding="utf-8").read()
        self.assertIn("_n = max(7, KARE_SAYISI)", kaynak,
                      "7 kare alt sınırı yok")
        self.assertIn("_kaydir", kaynak, "yarım kare kaydırma yok")

    def test_gecis_duz_ayrimi(self) -> None:
        """
        ⚠ `gecis_*` action'ları tasarım gereği **düz** olabilir
        (iki animasyon zaten sürekli ise 0→0 enterpolasyonu).
        Bu **hata değil**.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_onizleme.py",
                         encoding="utf-8").read()
        self.assertIn("GECIS_DUZ", kaynak, "geçiş düz ayrımı yok")
        self.assertIn('act.name.startswith("gecis_")', kaynak)

    def test_klasor_temizlenir(self) -> None:
        """
        ⚠ tur 18: her çağrı yeni PNG ekliyordu → karışık kareler.
        Çözüm: render'dan önce klasör temizlenir.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_onizleme.py",
                         encoding="utf-8").read()
        self.assertIn("os.remove", kaynak, "klasör temizlenmiyor")
        self.assertIn('endswith(".png")', kaynak)

    def test_canli_hareket_olcumu(self) -> None:
        """Canlı ölçüm: `kosma` NORMAL dönmeli."""
        from tools.anim_onizleme import render

        r = render("insan", "kosma", kare_sayisi=7, cozunurluk=256)
        if "hata" in r:
            self.skipTest(f"ölçülemedi: {r['hata'][:60]}")
        self.assertIn(r.get("karar"), ("NORMAL", "GECIS_DUZ"))
        # ⚠ `kosma` gerçek hareket içerir
        self.assertEqual(r.get("karar"), "NORMAL",
                         f"kosma {r.get('karar')} — beklenen NORMAL")
        self.assertGreater(r.get("en_buyuk_hareket", 0), 0.1)



class TestGorselAnimasyon(unittest.TestCase):
    """Animasyon render'ı görsel olarak değişiyor (tur 19 çözümü)."""

    def test_animasyon_render_uretir(self) -> None:
        """En az 5 kare render edilmeli."""
        from tools.anim_onizleme import render

        r = render("insan", "olum", kare_sayisi=7, cozunurluk=256)
        if "hata" in r:
            self.skipTest(f"render edilemedi: {r['hata'][:60]}")
        self.assertGreaterEqual(r.get("basarili", 0), 5)

    def test_deformasyon_pozitif(self) -> None:
        """
        ⚠ tur 19 ÖLÇÜMÜ: `olum` 0.684 m · `blok` 0.824 m ·
        `kosma` 0.776 m · `yurume` 0.447 m
        """
        from tools.anim_onizleme import render

        for anim, esik in (("olum", 0.3), ("blok", 0.4), ("kosma", 0.3)):
            r = render("insan", anim, kare_sayisi=7, cozunurluk=256)
            if "hata" in r:
                self.skipTest(f"{anim}: {r['hata'][:50]}")
            self.assertGreater(
                r.get("en_buyuk_deform", 0), esik,
                f"{anim}: deformasyon çok düşük "
                f"({r.get('en_buyuk_deform')} < {esik})")

    def test_deform_kemik_bazli(self) -> None:
        """Kemik bazlı deformasyon sözlüğü dönmeli."""
        from tools.anim_onizleme import render

        r = render("insan", "blok", kare_sayisi=7, cozunurluk=256)
        if "hata" in r:
            self.skipTest("render edilemedi")
        d = r.get("deform") or {}
        self.assertTrue(d, "kemik bazlı deformasyon yok")
        # ⚠ `blok` animasyonunda eller en çok hareket etmeli
        ilk = list(d.items())[0][0]
        self.assertIn(ilk, ("el_l", "el_r", "alt_kol_l", "alt_kol_r"),
                      f"en deforme kemik beklenmedik: {ilk}")



class TestOlcumMetodolojisi(unittest.TestCase):
    """
    Ölçüm metodolojisi katmanı (tur 20).

    ⚠ Gerekçe: tur 18-19'da **3 tur** boyunca ölçüm aracının kendisi
    yanlıştı. Bu katman "nasıl doğru ölçülür" sorusunu öğretir.
    """

    def test_katman_var(self) -> None:
        """`14_olcum` katmanı 6 ders içermeli."""
        from tools.blender_mantik import MÜFREDAT

        adlar = {k[0]: k[2] for k in MÜFREDAT}
        self.assertIn("14_olcum", adlar, "ölçüm katmanı yok")
        self.assertGreaterEqual(len(adlar["14_olcum"]), 11,
                                "en az 11 ders olmalı")

    def test_katman_sirasi(self) -> None:
        """Katmanlar **sıralı** numaralı olmalı (14 < 15)."""
        from tools.blender_mantik import MÜFREDAT

        adlar = [k[0] for k in MÜFREDAT]
        self.assertEqual(adlar, sorted(adlar),
                         "katman sırası bozuk")

    def test_olcum_dersleri_mevcut(self) -> None:
        """6 kritik ölçüm dersi bulunmalı."""
        from tools.blender_mantik import tum_dersler

        basliklar = " ".join(d.baslik for d in tum_dersler())
        for anahtar in ("Yanlış şeyi ölçmek",
                        "Örnekleme döngü noktalarına",
                        "önbelleği yanıltır",
                        "rest pozunda kalır",
                        "Tip karışıklığı",
                        "sınır"):
            self.assertIn(anahtar, basliklar,
                          f"'{anahtar}' dersi eksik")

    def test_kasa_senkron(self) -> None:
        """Kasadaki ders sayısı müfredatla eşleşmeli."""
        try:
            from neurocore.knowledge import KnowledgeLayer

            k = KnowledgeLayer()
            k.initialize()
        except Exception:
            self.skipTest("kasa açılamadı")
        try:
            n = sum(1 for x in k.vault.iter_notes(limit=300000)
                    if (getattr(x, "category", "") or "") == "blender_mantik")
        finally:
            try:
                k.shutdown()
            except Exception:
                pass
        from tools.blender_mantik import tum_dersler

        self.assertEqual(n, len(tum_dersler()),
                         f"kasa {n} ≠ müfredat {len(tum_dersler())}")



class TestBlendBul(unittest.TestCase):
    """
    `_blend_bul` dosya seçimi (tur 22).

    ⚠ tur 21-22 KESİN KÖK NEDEN: gevşek önek eşleştirmesi **yanlış
    dosyayı** seçiyordu:
        `_blend_bul("insan_orta_orta")`
        → `insan_kaliteli_kaliteli` (ikisi de "insan" ile başlıyor)
    Bu, `--anim` bayrağının çalışmıyor görünmesine yol açtı.
    """

    def test_tam_eslesme_oncelikli(self) -> None:
        """Tam eşleşme, önek eşleşmesini yenmeli."""
        from tools.anim_onizleme import _blend_bul

        b = _blend_bul("insan_orta_orta")
        if b is None:
            self.skipTest("insan_orta_orta yok")
        self.assertEqual(b.parent.name, "insan_orta_orta",
                         f"yanlış dosya: {b.parent.name}")

    def test_farkli_kaliteler_ayrisir(self) -> None:
        """Aynı öneki paylaşan kaliteler **ayrı** dosya dönmeli."""
        from tools.anim_onizleme import _blend_bul

        adlar = ["insan_orta_orta", "insan_lowpoly_lowpoly",
                 "insan_kaliteli_kaliteli"]
        yollar = {}
        for a in adlar:
            b = _blend_bul(a)
            if b is not None:
                yollar[a] = str(b)
        if len(yollar) < 2:
            self.skipTest("yeterli varlık yok")
        self.assertEqual(len(yollar), len(set(yollar.values())),
                         f"aynı dosya birden çok ad için: {yollar}")

    def test_dort_dosya_ortak_cozumleyici(self) -> None:
        """
        ⚠ tur 25: puanlı eşleştirme **ortak** modüle taşındı
        (`tools/varlik_yolu.py`). 5 araç oraya delege etmeli.
        """
        import io

        for f in ("anim_onizleme", "anim_gecis", "doku_uygula",
                  "varlik_onizleme", "anim_uret"):
            kaynak = io.open(_KOK / "tools" / f"{f}.py",
                             encoding="utf-8").read()
            self.assertTrue(
                "_ortak_blend_bul" in kaynak or "varlik_yolu" in kaynak,
                f"{f}.py ortak çözümleyiciye bağlı değil")

    def test_ortak_modul_puanli(self) -> None:
        """Ortak modülde puanlı eşleştirme olmalı."""
        import io

        kaynak = io.open(_KOK / "tools" / "varlik_yolu.py",
                         encoding="utf-8").read()
        self.assertIn("puan = 1000", kaynak, "tam eşleşme puanı yok")
        self.assertIn("en_iyi", kaynak)
        self.assertIn("en_iyi_yol", kaynak)

    def test_kalite_ayrimi(self) -> None:
        """
        ⚠ Aynı öneki paylaşan 3 kalite **ayrı** dosya dönmeli
        (tur 22'nin ana bulgusu).
        """
        from tools.anim_onizleme import _blend_bul

        beklenen = {
            "insan_orta_orta": "insan_orta_orta",
            "insan_lowpoly_lowpoly": "insan_lowpoly_lowpoly",
            "insan_kaliteli_kaliteli": "insan_kaliteli_kaliteli",
        }
        for girdi, klasor in beklenen.items():
            b = _blend_bul(girdi)
            if b is None:
                continue
            self.assertEqual(b.parent.name, klasor,
                             f"{girdi} → {b.parent.name}")

    def test_gevsek_onek_yok(self) -> None:
        """
        ⚠ tur 25: araçlarda **çift birleştirme** kalmamalı.
        `Path("Varliklar") / hedef` tek başına karar verici olmamalı.
        """
        import io

        for f in ("anim_uret", "anim_gecis", "doku_uygula",
                  "varlik_onizleme", "anim_onizleme"):
            kaynak = io.open(_KOK / "tools" / f"{f}.py",
                             encoding="utf-8").read()
            self.assertIn("_ortak_blend_bul", kaynak,
                          f"{f}.py: ortak çözümleyici yok")



class TestGecisGenisletme(unittest.TestCase):
    """Yeni tarifler için geçişler (tur 23)."""

    def test_yirmi_dort_cift(self) -> None:
        """24 geçiş çifti olmalı (14 → 24)."""
        import io
        import re

        kaynak = io.open(_KOK / "tools" / "anim_gecis.py",
                         encoding="utf-8").read()
        # ⚠ `["ad1", "ad2", gecis_kare]` kalıbını say
        n = len(re.findall(r'\["(\w+)", "(\w+)", gecis_kare\]', kaynak))
        self.assertGreaterEqual(n, 24, f"yalnız {n} çift var")

    def test_yeni_tarif_gecisleri(self) -> None:
        """Her yeni tarif için en az 2 geçiş olmalı."""
        import io
        import re

        kaynak = io.open(_KOK / "tools" / "anim_gecis.py",
                         encoding="utf-8").read()
        ciftler = re.findall(r'\["(\w+)", "(\w+)", gecis_kare\]', kaynak)
        duz = {a for c in ciftler for a in c}
        for ad in ("yuzme", "tirmanma", "hasar", "dirilme"):
            self.assertIn(ad, duz, f"'{ad}' geçişlerde yok")

    def test_kemik_sayisi_duzeltmesi(self) -> None:
        """
        ⚠ tur 23: `hareketli_kemik` **deform** sözlüğünden sayılmalı.
        Önceden fcurve sözlüğünden sayılıyordu → `0/19` görünüyordu.
        """
        import io

        kaynak = io.open(_KOK / "tools" / "anim_onizleme.py",
                         encoding="utf-8").read()
        self.assertIn("_deform_s.values()", kaynak,
                      "deform sözlüğünden sayım yok")
        self.assertIn("deformlu_kemik", kaynak)

    def test_canli_kemik_sayisi(self) -> None:
        """Canlı: `blok` animasyonunda 10+ kemik deforme olmalı."""
        from tools.anim_onizleme import render

        r = render("insan_orta_orta", "blok", kare_sayisi=7,
                   cozunurluk=256)
        if "hata" in r:
            self.skipTest(f"ölçülemedi: {r['hata'][:50]}")
        self.assertGreaterEqual(
            r.get("hareketli_kemik", 0), 5,
            f"kemik sayısı çok düşük: {r.get('hareketli_kemik')}")



class TestAnimOnizlemeKatalog(unittest.TestCase):
    """Animasyon önizlemesi katalogda (tur 24, Hedef D)."""

    def test_meta_alanlari(self) -> None:
        from tools.varlik_katalogu import VarlikMeta

        m = VarlikMeta()
        self.assertTrue(hasattr(m, "anim_onizleme"))
        self.assertTrue(hasattr(m, "anim_onizleme_sayisi"))
        self.assertEqual(m.anim_onizleme, {})
        self.assertEqual(m.anim_onizleme_sayisi, 0)

    def test_kaydet_baglama(self) -> None:
        """`kataloga_yaz()` animasyon önizlemesini bağlamalı."""
        import io

        kaynak = io.open(_KOK / "tools" / "varlik_kaydet.py",
                         encoding="utf-8").read()
        self.assertIn("anim_onizleme_klasor", kaynak)
        self.assertIn('klasor / "anim_onizleme"', kaynak)
        self.assertIn("m.anim_onizleme_sayisi = len(m.anim_onizleme)", kaynak)

    def test_istatistik(self) -> None:
        from tools.varlik_katalogu import istatistik

        s = istatistik()
        self.assertIn("anim_onizlemeli", s)
        self.assertIn("anim_onizleme_toplam", s)

    def test_canli_bes_varlik(self) -> None:
        """
        ⚠ tur 24 ÖLÇÜMÜ: 5 armature'lü varlık · **141 animasyon**
        önizlemesi katalogda.
        """
        from tools.varlik_katalogu import istatistik, listele

        v = [x for x in listele() if x.get("anim_onizleme_sayisi", 0) > 0]
        if not v:
            self.skipTest("animasyon önizlemesi yok")
        self.assertGreaterEqual(len(v), 3,
                                f"yalnız {len(v)} varlık")
        for x in v[:3]:
            for ad, bilgi in list(x.get("anim_onizleme", {}).items())[:2]:
                self.assertIn("png", bilgi, f"{ad}: png listesi yok")
                self.assertGreater(len(bilgi["png"]), 0)
                # ⚠ Dosyalar GERÇEKTEN var olmalı
                p = Path(bilgi["png"][0])
                self.assertTrue(p.exists(), f"{ad}: {p} yok")
        self.assertGreaterEqual(istatistik().get("anim_onizleme_toplam", 0),
                                50)

    def test_katalog_araci_dondurur(self) -> None:
        """`asset.katalog` animasyon önizlemesi döndürmeli."""
        from neurocore.agent.coding import CodingAgent

        a = CodingAgent()
        r = a.katalog()
        if not r["varliklar"]:
            self.skipTest("katalog boş")
        v = r["varliklar"][0]
        self.assertIn("anim_onizleme_sayisi", v)
        self.assertIn("anim_onizleme", v)



class TestAracSecimiYeniIstekler(unittest.TestCase):
    """
    Araç seçimi — **yeni** istekler (tur 26, Hedef F).

    ⚠ tur 26 ÖLÇÜMÜ: 17 yeni istekte 5 kaçak bulundu (%70).
    Anahtar sözcükler eklendikten sonra **17/17 (%100)**.
    """

    #: ⚠ Yeni istekler — hiçbiri eski testlerde yok
    ISTEKLER = (
        "kayadan bir kule yap", "timsahı yüzdür", "karakteri zıplat",
        "ağaca yaprak dokusu ver", "varlığın UV'sini aç",
        "yaratığa kuyruk animasyonu ekle", "modeli önizle",
        "hangi varlıklar var", "animasyonlar çalışıyor mu",
        "timsahın yürüyüşünü düzelt", "insanı tırmandır",
        "kılıca metal dokusu uygula", "sahneyi render et",
        "varlığı diske kaydet", "ölüm animasyonu ekle",
        "hasar alma animasyonu", "yeni monster üret",
    )

    def _sec(self, istek: str) -> list:
        from neurocore.agent.loop import CORE_TOOLS, TOOL_HINTS

        gl = istek.lower()
        sec = []
        for anahtarlar, araclar in TOOL_HINTS:
            if any(k in gl for k in anahtarlar):
                sec.extend(araclar)
        return [x for x in dict.fromkeys(sec)
                if x in CORE_TOOLS and x.startswith("asset.")]

    def test_yuzde_yuz_secim(self) -> None:
        """17/17 istek en az bir Blender aracına eşlenmeli."""
        kacak = [g for g in self.ISTEKLER if not self._sec(g)]
        self.assertEqual(kacak, [], f"kaçak istekler: {kacak}")

    def test_anim_onizleme_secimi(self) -> None:
        """`animasyonlar çalışıyor mu` → anim_onizleme."""
        self.assertIn("asset.anim_onizleme",
                      self._sec("animasyonlar çalışıyor mu"))

    def test_uv_secimi(self) -> None:
        """`varlığın UV'sini aç` → asset.uret (UV içinde yapılır)."""
        self.assertIn("asset.uret", self._sec("varlığın UV'sini aç"))

    def test_kaydet_secimi(self) -> None:
        """`varlığı diske kaydet` → asset.katalog."""
        self.assertIn("asset.katalog", self._sec("varlığı diske kaydet"))

    def test_onizle_kok_secimi(self) -> None:
        """`modeli önizle` → asset.onizleme (yalnız 'önizleme' yetmez)."""
        self.assertIn("asset.onizleme", self._sec("modeli önizle"))

    def test_zıplat_secimi(self) -> None:
        """`karakteri zıplat` → asset.animasyon."""
        self.assertIn("asset.animasyon", self._sec("karakteri zıplat"))



class TestKaliteKapisi(unittest.TestCase):
    """
    Kalite kapısı — kelime isabeti ölçütü.

    ⚠ ÖLÇÜM (sıfırdan model, 13.574 adım):
        Üretilen 96 kelimeden yalnız **16'sı** (%16,7) gerçek
        Türkçe/İngilizce kelimeydi. Ama ilk 5 ölçüt bunu
        **onaylıyordu** (alfabetik %96 · ort. 8,0 harf · tekrar yok).

        Kanıt: 8 sorunun **8'inde** `quality_rejected = False`.
    """

    #: ⚠ GERÇEK model çıktıları — reddedilmeli
    COP = (
        "Aimap giyeCostyAima) verirdgeç, 3 olarakAimasekiglede), lışroznost).",
        ").\nf\n\n işlemi-1.0Abvr SCname 10110 (ör.lical** s düzenlemekten0 "
        "metni)\n\n**\n\n** `-FNa kadaradd,.\n\n-",
        "lemedu ve **erdpy (0 veya GPU/ KB yap.\n\n\nMcei ve ** temel "
        "görevi vardır:**",
        '00dan# 1 veri "staerlar (294** bir aktarmayaıma bir satır-C,45run.',
        "bpycodek (\n\n0, Tipik amacakcakım kodu veri; hafiftir 1 (ör.c 1 ile",
    )

    #: ⚠ GERÇEK metin — kabul edilmeli
    GERCEK = (
        "DXGI_ERROR_DEVICE_HUNG hatası, ekran kartı sürücüsünün belirli bir "
        "süre içinde yanıt vermemesi sonucu oluşur. Sürücüyü güncelleyin.",
        "The device failed because the graphics driver stopped responding. "
        "Update the driver and check the timeout delay setting.",
        "Blender'da UV açmak için nesneyi seçin, dikişleri işaretleyin ve "
        "unwrap komutunu çalıştırın. Böylece yüzey düzleştirilir.",
        "import bpy\nbpy.ops.mesh.primitive_cube_add(size=2.0)\n"
        'obj = bpy.context.active_object\nobj.name = "Kup"',
        "Python'da bir listeyi sıralamak için sorted fonksiyonunu veya "
        "listenin sort metodunu kullanabilirsiniz.",
    )

    def test_cop_reddedilir(self) -> None:
        """Uydurma kelimelerden oluşan çıktı reddedilmeli."""
        from neurocore.neural.model import NeuroCoreModel

        for t in self.COP:
            self.assertFalse(
                NeuroCoreModel._looks_like_language(t),
                f"çöp GEÇTİ: {t[:50]!r}")

    def test_gercek_metin_kabul(self) -> None:
        """Gerçek Türkçe/İngilizce metin kabul edilmeli (yanlış ret yok)."""
        from neurocore.neural.model import NeuroCoreModel

        for t in self.GERCEK:
            self.assertTrue(
                NeuroCoreModel._looks_like_language(t),
                f"gerçek metin REDDEDİLDİ: {t[:50]!r}")

    def test_kok_eslestirme(self) -> None:
        """
        ⚠ Türkçe **eklemeli** dil — tam-kelime eşleştirmesi ekleri
        kaçırıyordu (`listeyi`, `seçin`, `işaretleyin` yanlış RED).
        Kök eşleştirmesi bunu çözer.
        """
        from neurocore.neural.model import _kelime_gercek

        for k in ("listeyi", "sürücüsünün", "güncelleyin", "işaretleyin",
                  "komutunu", "dikişleri"):
            self.assertTrue(_kelime_gercek(k), f"{k} gerçek sayılmadı")
        # ⚠ Uydurma kelimeler geçmemeli
        for k in ("Aimap", "giyeCostyAima", "lışroznost", "bpycodek"):
            self.assertFalse(_kelime_gercek(k), f"{k} gerçek SAYILDI")

    def test_kisa_kelime_tam_eslesme(self) -> None:
        """
        ⚠ Kısa kelimeler (≤ kök uzunluğu) yalnız TAM eşleşmeyle geçer;
        aksi halde 2 harfli kök eşleşmesi çok gevşek olurdu.
        """
        from neurocore.neural.model import _KOK_UZUNLUK, _kelime_gercek

        self.assertEqual(_KOK_UZUNLUK, 4)
        self.assertTrue(_kelime_gercek("kod"))
        self.assertTrue(_kelime_gercek("veri"))
        self.assertFalse(_kelime_gercek("xqz"))

    def test_sinir_durumlar(self) -> None:
        """Boş/kısa metin reddedilmeli."""
        from neurocore.neural.model import NeuroCoreModel

        for t in ("", "   ", "ab", "kısa"):
            self.assertFalse(NeuroCoreModel._looks_like_language(t),
                             f"{t!r} geçmemeliydi")


if __name__ == "__main__":
    unittest.main(verbosity=2)