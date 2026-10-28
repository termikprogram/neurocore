"""
Sohbet Bazlı Çalışma Alanı Testleri
====================================
Her sohbetin **kendi çalışma alanı** olmasını doğrular.

Kural:
    * Sohbetin klasörü varsa çalışma alanı **odur**.
    * Klasör yoksa **genel sohbet** modu (proje bağlamı olmadan).
    * Ayarlar'daki genel çalışma alanı yalnızca klasörsüz sohbetler için.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import assert_json_safe

from neurocore.app.chats import MAX_CHATS, Chat, ChatStore, make_title


class TestMakeTitle(unittest.TestCase):
    """Başlık üretimi (ilk mesajdan)."""

    def test_short_text_unchanged(self) -> None:
        self.assertEqual(make_title("kısa mesaj"), "kısa mesaj")

    def test_long_text_truncated(self) -> None:
        title = make_title("kelime " * 40)
        self.assertLessEqual(len(title), 66)

    def test_whitespace_collapsed(self) -> None:
        self.assertEqual(make_title("  çok   boşluklu  "), "çok boşluklu")

    def test_newlines_removed(self) -> None:
        """Başlık tek satır olmalı (arayüz bozulmasın)."""
        self.assertNotIn("\n", make_title("birinci satır\nikinci satır"))

    def test_empty_uses_fallback(self) -> None:
        self.assertEqual(make_title(""), "Yeni sohbet")
        self.assertEqual(make_title("   ", fallback="Boş"), "Boş")

    def test_turkish_characters_preserved(self) -> None:
        self.assertEqual(make_title("çğışöü testi"), "çğışöü testi")


class TestChatStore(unittest.TestCase):
    """Depo: oluşturma, mesaj, klasör, kalıcılık."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_chats_")
        self.store = ChatStore()
        self.store.initialize(data_path=self.root)

    def tearDown(self) -> None:
        import shutil

        self.store.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_create_with_workspace(self) -> None:
        chat = self.store.create(workspace=r"E:\projeler\site")
        self.assertEqual(chat.workspace, r"E:\projeler\site")
        self.assertEqual(chat.workspace_name, "site")
        self.assertFalse(chat.to_dict()["general"])

    def test_create_without_workspace_is_general(self) -> None:
        """Klasör verilmezse sohbet **genel modda** olur."""
        chat = self.store.create()
        self.assertEqual(chat.workspace, "")
        self.assertTrue(chat.to_dict()["general"])
        self.assertEqual(chat.workspace_name, "")

    def test_workspace_name_from_path(self) -> None:
        for path, expected in ((r"E:\a\b\proje", "proje"),
                               ("/home/user/site", "site"),
                               (r"C:\tek", "tek")):
            chat = Chat(id="x", workspace=path)
            self.assertEqual(chat.workspace_name, expected)

    def test_append_and_autotitle(self) -> None:
        """İlk kullanıcı mesajı başlığı üretir."""
        chat = self.store.create()
        self.store.append(chat.id, "user", "bir web sayfası yap")
        self.assertEqual(self.store.get(chat.id).title, "bir web sayfası yap")

    def test_title_not_overwritten(self) -> None:
        """Sonraki mesajlar başlığı değiştirmez."""
        chat = self.store.create()
        self.store.append(chat.id, "user", "ilk mesaj")
        self.store.append(chat.id, "user", "ikinci mesaj")
        self.assertEqual(self.store.get(chat.id).title, "ilk mesaj")

    def test_history_order(self) -> None:
        chat = self.store.create()
        self.store.append(chat.id, "user", "soru")
        self.store.append(chat.id, "assistant", "cevap")
        history = self.store.history(chat.id)
        self.assertEqual([item["role"] for item in history], ["user", "assistant"])

    def test_set_workspace_switch_and_clear(self) -> None:
        """Klasör bağlanıp kaldırılabilir."""
        chat = self.store.create()
        self.store.set_workspace(chat.id, r"E:\proje")
        self.assertEqual(self.store.get(chat.id).workspace_name, "proje")
        self.store.set_workspace(chat.id, "")
        self.assertEqual(self.store.get(chat.id).workspace, "")
        self.assertTrue(self.store.get(chat.id).to_dict()["general"])

    def test_persistence_across_restart(self) -> None:
        """
        Sohbetler ve **klasör bağlantıları** yeniden başlatmada korunur.

        Bu olmadan her açılışta tüm sohbetler kaybolurdu.
        """
        chat = self.store.create(workspace=r"E:\projeler\site")
        self.store.append(chat.id, "user", "kalıcı mı?")
        self.store.append(chat.id, "assistant", "evet")
        self.store.shutdown()

        fresh = ChatStore()
        fresh.initialize(data_path=self.root)
        try:
            loaded = fresh.get(chat.id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.workspace, r"E:\projeler\site")
            self.assertEqual(loaded.title, "kalıcı mı?")
            self.assertEqual(len(loaded.messages), 2)
        finally:
            fresh.shutdown()

    def test_list_sorted_by_updated(self) -> None:
        """En son güncellenen sohbet başta olur."""
        first = self.store.create(title="ilk")
        second = self.store.create(title="ikinci")
        self.store.append(first.id, "user", "yeni mesaj")
        titles = [item["title"] for item in self.store.list()]
        self.assertEqual(titles[0], "ilk")

    def test_delete(self) -> None:
        chat = self.store.create()
        self.assertTrue(self.store.delete(chat.id))
        self.assertIsNone(self.store.get(chat.id))
        self.assertFalse(self.store.delete(chat.id))

    def test_clear_keeps_workspace(self) -> None:
        """Mesaj temizliği klasör bağlantısını **bozmaz**."""
        chat = self.store.create(workspace=r"E:\proje")
        self.store.append(chat.id, "user", "mesaj")
        self.store.clear(chat.id)
        cleared = self.store.get(chat.id)
        self.assertEqual(len(cleared.messages), 0)
        self.assertEqual(cleared.workspace, r"E:\proje")
        self.assertEqual(cleared.title, "Yeni sohbet")

    def test_rename(self) -> None:
        chat = self.store.create()
        self.store.rename(chat.id, "Yeni Ad")
        self.assertEqual(self.store.get(chat.id).title, "Yeni Ad")

    def test_unknown_id_returns_none(self) -> None:
        self.assertIsNone(self.store.get("yok"))
        self.assertIsNone(self.store.append("yok", "user", "x"))
        self.assertIsNone(self.store.set_workspace("yok", r"E:\a"))
        self.assertEqual(self.store.history("yok"), [])

    def test_max_chats_enforced(self) -> None:
        """Sınırsız büyüme engellenir (dosya şişmesin)."""
        for index in range(MAX_CHATS + 12):
            self.store.create(title=f"sohbet {index}")
        self.assertLessEqual(len(self.store.list(limit=10_000)), MAX_CHATS)

    def test_message_limit_enforced(self) -> None:
        """Sohbet başına mesaj sayısı sınırlanır."""
        chat = self.store.create()
        for index in range(260):
            self.store.append(chat.id, "user", f"mesaj {index}")
        self.assertLessEqual(len(self.store.get(chat.id).messages), 200)

    def test_stats(self) -> None:
        self.store.create(workspace=r"E:\a")
        self.store.create()
        stats = self.store.get_stats()
        self.assertEqual(stats["chats"], 2)
        self.assertEqual(stats["with_workspace"], 1)
        self.assertEqual(stats["general"], 1)
        assert_json_safe(self, stats, "chat_stats")

    def test_chat_dict_json_safe(self) -> None:
        chat = self.store.create(workspace=r"E:\proje")
        self.store.append(chat.id, "user", "soru")
        data = chat.to_dict(with_messages=True)
        self.assertIn("messages", data)
        self.assertIn("workspace_name", data)
        assert_json_safe(self, data, "chat")

    def test_corrupt_file_tolerated(self) -> None:
        """Bozuk sohbet dosyası açılışı engellemez."""
        path = Path(self.root, "chats.json")
        path.write_text("{bozuk json", encoding="utf-8")
        fresh = ChatStore()
        fresh.initialize(data_path=self.root)
        try:
            self.assertEqual(fresh.list(), [])
        finally:
            fresh.shutdown()

    def test_partial_records_tolerated(self) -> None:
        """Eksik alanlı kayıtlar atlanır, sağlamlar yüklenir."""
        path = Path(self.root, "chats.json")
        path.write_text(json.dumps({"chats": [
            {"id": "iyi", "title": "Sağlam", "workspace": r"E:\p"},
            {"title": "kimliksiz"},
            "metin",
        ]}), encoding="utf-8")
        fresh = ChatStore()
        fresh.initialize(data_path=self.root)
        try:
            self.assertEqual(len(fresh.list()), 1)
            self.assertEqual(fresh.get("iyi").workspace, r"E:\p")
        finally:
            fresh.shutdown()


class TestPathResolution(unittest.TestCase):
    """
    Yol çözümlemesi: `""` ve `"."` **çalışma alanı kökü** olmalı.

    Ölçülen hata: `TestProjects` çalışma alanındayken `path="."` gönderen
    `code.list_directory`, uygulamanın başlatıldığı dizini listeliyordu —
    çünkü `"."` süreç CWD'sine çözülüyordu.
    """

    def setUp(self) -> None:
        import warnings

        warnings.filterwarnings("ignore")
        from neurocore.agent.coding import CodingAgent

        self.root = Path(tempfile.mkdtemp(prefix="nc_resolve_"))
        self.workspace = self.root / "proje"
        self.workspace.mkdir()
        (self.workspace / "iceride.txt").write_text("x", encoding="utf-8")
        self.agent = CodingAgent()
        self.agent.initialize(workspace=str(self.workspace))

    def tearDown(self) -> None:
        import shutil

        self.agent.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_empty_path_is_workspace_root(self) -> None:
        self.assertEqual(self.agent.resolve(""), self.workspace.resolve())

    def test_dot_path_is_workspace_root(self) -> None:
        self.assertEqual(self.agent.resolve("."), self.workspace.resolve())

    def test_relative_stays_inside(self) -> None:
        target = self.agent.resolve("iceride.txt")
        self.assertTrue(str(target).startswith(str(self.workspace.resolve())))

    def test_workspace_switch_affects_resolution(self) -> None:
        """Klasör değişince aynı göreli yol yeni köke çözülür."""
        other = self.root / "baska"
        other.mkdir()
        (other / "iceride.txt").write_text("y", encoding="utf-8")
        self.agent.set_workspace(str(other))
        self.assertEqual(self.agent.resolve("."), other.resolve())
        self.assertEqual(self.agent.resolve("iceride.txt"),
                         (other / "iceride.txt").resolve())

    def test_list_directory_uses_workspace(self) -> None:
        """`list_directory` çalışma alanını listeler, CWD'yi değil."""
        result = self.agent.list_directory("")
        self.assertIn("iceride.txt",
                      [item["name"] for item in result.get("items", [])])


class TestWorkspaceTreeHelpers(unittest.TestCase):
    """
    Dosya ağacı yardımcıları (DSH `workspaceFiles.list` modeli).

    Ağaç **oturumun çalışma alanına** köklenir ve **salt okunur**; mutasyon
    dizin seçicide (klasör oluşturma) ve ajan araçlarındadır.
    """

    def test_language_mapping(self) -> None:
        """Uzantıdan dil etiketi çıkarılır (arayüz simgesi için)."""
        from neurocore.app.server import _language_of

        self.assertEqual(_language_of(".py"), "python")
        self.assertEqual(_language_of(".PY"), "python")
        self.assertEqual(_language_of(".gd"), "gdscript")
        self.assertEqual(_language_of(".tscn"), "godot")
        self.assertEqual(_language_of(".md"), "markdown")
        self.assertEqual(_language_of(""), "")
        self.assertEqual(_language_of(".bilinmeyen"), "")

    def test_folder_name_validation(self) -> None:
        """
        Klasör adı doğrulaması: yol ayırıcı ve kaçış karakterleri reddedilir.

        Bu kural **klasör kaçışını** engeller: `..` veya `a/b` ile çalışma
        alanı dışına klasör açılamamalıdır.
        """
        forbidden = ["a/b", "a\\b", "..", ".", "C:", "a:b", "a*b", 'a"b',
                     "a<b", "a>b", "a|b", "a?b"]
        invalid_chars = set("\\/:*?\"<>|")
        for name in forbidden:
            has_bad_char = any(char in name for char in invalid_chars)
            is_dot = name in (".", "..")
            self.assertTrue(has_bad_char or is_dot,
                            f"{name!r} reddedilmeliydi")

    def test_folder_creation_inside_workspace(self) -> None:
        """Geçerli ad ile klasör gerçekten oluşturulur."""
        import shutil

        root = Path(tempfile.mkdtemp(prefix="nc_mkdir_"))
        try:
            target = root / "yeni_klasor"
            target.mkdir(parents=False)
            self.assertTrue(target.is_dir())
            # Aynı adı tekrar oluşturmak `FileExistsError` vermeli
            with self.assertRaises(FileExistsError):
                target.mkdir(parents=False)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
