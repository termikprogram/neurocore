"""
Ağ Müfredatı Testleri
======================
`neurocore.neural.network` — ağ yapısını temelden uzmanlığa öğretir.

Neden bu testler önemli?
    Ağ içeriği **teknik doğruluk** gerektirir: yanlış port numarası, yanlış
    protokol davranışı veya uydurma API adı modele **yanlış bilgi** öğretir.
    Bu testler:

        * her katmanın kapsandığını,
        * içeriğin boş olmadığını,
        * kod örneklerinin geçerli olduğunu,
        * **kritik teknik gerçeklerin** (port numaraları, protokol
          davranışları) yerinde olduğunu,
        * müfredata doğru bağlandığını

        doğrular.
"""

from __future__ import annotations

import ast
import unittest
from typing import Dict, List

from tests.helpers import assert_json_safe

from neurocore.neural.curriculum import (
    ALL_LAYERS,
    LAYER_LABELS,
    NETWORK_LAYER_NAMES,
    build_curriculum,
)
from neurocore.neural.network import (
    NETWORK_LABELS,
    NETWORK_LAYERS,
    build_network_curriculum,
    network_stats,
)


class TestNetworkCurriculumStructure(unittest.TestCase):
    """Yapı: katmanlar, dersler, kod."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.lessons = build_network_curriculum()

    def test_all_layers_covered(self) -> None:
        """Altı ağ katmanının hepsi ders içermeli."""
        mevcut = {ders.layer for ders in self.lessons}
        for katman in NETWORK_LAYERS:
            self.assertIn(katman, mevcut, f"{katman} katmanı boş")

    def test_each_layer_has_enough(self) -> None:
        """Her katman anlamlı sayıda ders içerir."""
        sayim: Dict[str, int] = {}
        for ders in self.lessons:
            sayim[ders.layer] = sayim.get(ders.layer, 0) + 1
        for katman, adet in sayim.items():
            self.assertGreaterEqual(adet, 8,
                                    f"{katman} çok az ders içeriyor")

    def test_no_empty_content(self) -> None:
        for ders in self.lessons:
            self.assertTrue(ders.title.strip())
            self.assertGreater(len(ders.body.strip()), 100,
                               f"gövde çok kısa: {ders.title}")

    def test_titles_unique(self) -> None:
        basliklar = [d.title for d in self.lessons]
        self.assertEqual(len(basliklar), len(set(basliklar)),
                         "aynı başlık birden fazla")

    def test_prompt_is_labeled(self) -> None:
        """Soru, katman etiketiyle başlar."""
        for ders in self.lessons:
            self.assertTrue(ders.prompt().startswith(ders.label))

    def test_code_examples_valid_python(self) -> None:
        """
        Kod örnekleri **geçerli Python** olmalı.

        Bozuk kod öğretmek modele yanlış kalıp öğretir. Yorum blokları
        (ASCII diyagramlar, karar tabloları) ve kabuk komutları atlanır —
        onlar Python değildir.
        """
        gecerli = 0
        for ders in self.lessons:
            govde = ders.code.strip()
            if not govde:
                continue
            # Yorum bloğu (diyagram/tablo) veya kabuk komutu → atla
            if govde.startswith("#"):
                continue
            if any(im in govde[:60] for im in
                   ("netstat", "arp -a", "route print", "godot --headless",
                    "tracert", "nslookup")):
                continue
            try:
                ast.parse(govde)
                gecerli += 1
            except SyntaxError as exc:
                self.fail(f"geçersiz kod — {ders.title}: {exc}")
        self.assertGreaterEqual(gecerli, 3,
                                "yeterli Python örneği test edilmedi")

    def test_covers_code_examples(self) -> None:
        """En az birkaç katmanda kod örneği olmalı."""
        kodlu = {d.layer for d in self.lessons if d.code.strip()}
        self.assertGreaterEqual(len(kodlu), 4,
                                f"kod örneği olan katman az: {kodlu}")

    def test_stats_shape(self) -> None:
        stats = network_stats()
        self.assertEqual(stats["total_lessons"], len(self.lessons))
        self.assertGreater(stats["total_chars"], 20_000)
        assert_json_safe(self, stats, "network_stats")

    def test_layer_filter(self) -> None:
        """Belirli katman istenebilmeli."""
        sadece = build_network_curriculum(["net_godot"])
        self.assertTrue(sadece)
        self.assertTrue(all(d.layer == "net_godot" for d in sadece))

    def test_labels_match_curriculum(self) -> None:
        """Etiketler `curriculum.LAYER_LABELS` ile aynı olmalı."""
        for katman in NETWORK_LAYERS:
            self.assertEqual(NETWORK_LABELS[katman],
                             LAYER_LABELS[katman])


class TestTechnicalAccuracy(unittest.TestCase):
    """
    Teknik doğruluk — **yanlış bilgi öğretmemek** için.

    Bu testler gerçek protokol değerlerini kilitler. Bir değer yanlış
    yazılırsa model yanlış öğrenir ve üretimde hata yapar.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.lessons = build_network_curriculum()
        cls.metin = " ".join(d.title + " " + d.body + " " + d.code
                             for d in cls.lessons).lower()

    def test_port_numbers_correct(self) -> None:
        """Kritik port numaraları doğru olmalı."""
        for port, ad in (("22", "ssh"), ("53", "dns"), ("80", "http"),
                         ("443", "https"), ("3306", "mysql"),
                         ("5432", "postgresql")):
            self.assertIn(port, self.metin, f"{ad} portu ({port}) eksik")

    def test_tcp_handshake_described(self) -> None:
        """TCP üçlü el sıkışma doğru anlatılmalı."""
        self.assertIn("syn", self.metin)
        self.assertIn("ack", self.metin)

    def test_udp_unreliable_stated(self) -> None:
        """UDP'nin güvenilmez olduğu açıkça yazmalı."""
        self.assertTrue("bağlantısız" in self.metin
                        or "baglantisiz" in self.metin)
        self.assertTrue("onay yok" in self.metin
                        or "yeniden iletim yok" in self.metin)

    def test_osi_has_seven_layers(self) -> None:
        """OSI modeli 7 katman olarak anlatılmalı."""
        osi = [d for d in self.lessons if "osi" in d.title.lower()]
        self.assertTrue(osi)
        govde = " ".join(d.body for d in osi).lower()
        self.assertIn("7", govde)
        for katman in ("fiziksel", "taşıma", "uygulama"):
            self.assertIn(katman, govde)

    def test_mtu_value_correct(self) -> None:
        """Ethernet MTU değeri 1500 olmalı."""
        self.assertIn("1500", self.metin)

    def test_proxy_forward_vs_reverse(self) -> None:
        """İleri ve ters proxy ayrımı doğru anlatılmalı."""
        self.assertTrue("i̇leri proxy" in self.metin
                        or "ileri proxy" in self.metin)
        self.assertIn("ters proxy", self.metin)

    def test_cap_theorem_stated(self) -> None:
        """CAP teoremi üç harfiyle geçmeli."""
        cap = [d for d in self.lessons if "cap" in d.body.lower()]
        self.assertTrue(cap)
        govde = " ".join(d.body for d in cap).lower()
        for harf in ("tutarlılık", "erişilebilirlik", "bölünme"):
            self.assertIn(harf, govde)

    def test_dedicated_vs_listen(self) -> None:
        """Dedicated ve listen sunucu ayrımı yerinde."""
        self.assertIn("dedicated", self.metin)
        self.assertIn("listen server", self.metin)
        self.assertIn("authoritative", self.metin)

    def test_godot_api_names_correct(self) -> None:
        """Godot API adları **gerçek** olmalı (uydurma değil)."""
        for api in ("multiplayer.multiplayer_peer", "peer_connected",
                    "create_server", "create_client",
                    "get_remote_sender_id", "is_server",
                    "multiplayer.get_peers", "set_multiplayer_authority"):
            self.assertIn(api.lower(), self.metin, f"{api} eksik")

    def test_godot_peer_types_correct(self) -> None:
        """Godot peer türleri doğru yazılmalı."""
        for tur in ("enetmultiplayerpeer", "websocketmultiplayerpeer"):
            self.assertIn(tur, self.metin)

    def test_rpc_annotation_modes(self) -> None:
        """RPC işaretleyici modları doğru olmalı."""
        for mod in ("any_peer", "call_remote", "reliable",
                    "unreliable_ordered"):
            self.assertIn(mod, self.metin)

    def test_quic_and_http3_linked(self) -> None:
        """QUIC'in UDP üzerinde olduğu ve HTTP/3 ile ilişkisi yazmalı."""
        quic = [d for d in self.lessons if "quic" in d.title.lower()]
        self.assertTrue(quic)
        govde = " ".join(d.body for d in quic).lower()
        self.assertIn("udp", govde)
        self.assertIn("http/3", govde)

    def test_stun_turn_explained(self) -> None:
        """STUN/TURN NAT geçişi anlatılmalı."""
        self.assertIn("stun", self.metin)
        self.assertIn("turn", self.metin)
        self.assertIn("hole punching", self.metin)

    def test_tick_rate_mentioned(self) -> None:
        """Tick oranı ve sabit adım kavramı geçmeli."""
        self.assertIn("tick", self.metin)
        self.assertIn("1/60", self.metin)

    def test_lag_compensation_rewind(self) -> None:
        """Gecikme telafisinin geri sarma yöntemi anlatılmalı."""
        lag = [d for d in self.lessons
               if "lag compensation" in d.title.lower()]
        self.assertTrue(lag)
        self.assertIn("rewind", " ".join(d.body for d in lag).lower())

    def test_security_rules_present(self) -> None:
        """İstemciye güvenmeme kuralı açıkça yazmalı."""
        self.assertTrue("güvenme" in self.metin or "guvenme" in self.metin)
        self.assertIn("is_server()", self.metin)


class TestNetworkIntegration(unittest.TestCase):
    """Müfredata entegrasyon — ağ dersleri eğitimde kullanılmalı."""

    def test_network_layers_in_all_layers(self) -> None:
        for katman in NETWORK_LAYER_NAMES:
            self.assertIn(katman, ALL_LAYERS)

    def test_build_curriculum_includes_network(self) -> None:
        """
        `build_curriculum()` ağ derslerini **içermeli**.

        Aksi hâlde tokenizer, kasa ve eğitim hattı ağ içeriğini görmez ve
        model ağ öğrenmez.
        """
        tum = build_curriculum()
        katmanlar = {d.layer for d in tum}
        for katman in NETWORK_LAYER_NAMES:
            self.assertIn(katman, katmanlar,
                          f"{katman} birleşik müfredatta yok")

    def test_curriculum_grew(self) -> None:
        """Birleşik müfredat eskisinden **büyük** olmalı."""
        tum = build_curriculum()
        self.assertGreaterEqual(len(tum), 110,
                                f"birleşik müfredat beklenenden az: {len(tum)}")
        ag = [d for d in tum if d.layer in NETWORK_LAYER_NAMES]
        self.assertGreaterEqual(len(ag), 50)

    def test_labels_registered(self) -> None:
        """Ağ katmanlarının etiketi `LAYER_LABELS`'ta olmalı."""
        for katman in NETWORK_LAYER_NAMES:
            self.assertIn(katman, LAYER_LABELS)

    def test_no_duplicate_titles_across_curriculum(self) -> None:
        """Birleşik müfredatta başlık tekrarı olmamalı."""
        tum = build_curriculum()
        basliklar = [d.title for d in tum]
        tekrar = [b for b in set(basliklar) if basliklar.count(b) > 1]
        self.assertEqual(tekrar, [], f"tekrar eden başlıklar: {tekrar}")

    def test_network_lessons_have_code_or_detail(self) -> None:
        """Her ağ dersi ya kod ya ayrıntılı gövde içermeli."""
        for ders in build_network_curriculum():
            self.assertTrue(
                ders.code.strip() or len(ders.body) > 150,
                f"ne kod ne ayrıntı: {ders.title}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
