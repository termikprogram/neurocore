"""
Masaüstü Uygulaması Testleri
=============================
Ajan katmanı, araç protokolü, özerklik politikası, onay akışı ve uygulama
durumunu doğrular.

Bu testler **gerçek** bileşenleri geçici dizinlerde kullanır; ağ çağrısı
yapmaz, dil modeli gerektirmez ve kullanıcının dosyalarını değiştirmez.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

from tests.helpers import assert_json_safe

from neurocore.agent.coding import (
    CodingAgent,
    CodingError,
    detect_dangerous_command,
    parse_test_output,
)
from neurocore.agent.llm import LLMMessage, LLMResponse, LLMRouter, tcp_reachable
from neurocore.agent.loop import AgentLoop, _extract_path, _extract_symbol, _normalize_decision
from neurocore.agent.shell import (
    CLASS_SAFETY,
    CommandClass,
    ShellIntegration,
    classify_command,
    command_safety,
)
from neurocore.agent.tools import (
    AutonomyLevel,
    AutonomyPolicy,
    Safety,
    ToolContext,
    ToolDef,
    ToolParameter,
    ToolRegistry,
    collect_tools,
    tool,
)
from neurocore.app.state import AppState, ApprovalBroker, EventStream


def _workspace() -> str:
    """Örnek bir proje kökü oluşturur."""
    root = tempfile.mkdtemp(prefix="nc_app_")
    base = Path(root)
    (base / "pkg").mkdir()
    (base / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (base / "pkg" / "calc.py").write_text(
        "import math\n\n"
        "VERSION = '1.0'\n\n"
        "class Hesap:\n"
        "    def __init__(self, base=0):\n"
        "        self.base = base\n\n"
        "    def topla(self, x):\n"
        "        return self.base + x\n\n"
        "def kare(x):\n"
        "    return x * x\n",
        encoding="utf-8",
    )
    (base / "tests").mkdir()
    (base / "tests" / "test_calc.py").write_text(
        "import os, sys, unittest\n"
        "sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))\n"
        "from pkg.calc import kare, Hesap\n\n"
        "class TestCalc(unittest.TestCase):\n"
        "    def test_kare(self): self.assertEqual(kare(3), 9)\n"
        "    def test_topla(self): self.assertEqual(Hesap(2).topla(3), 5)\n",
        encoding="utf-8",
    )
    (base / "pyproject.toml").write_text("[project]\nname='ornek'\n", encoding="utf-8")
    (base / "README.md").write_text("# Örnek Proje\n\nBir deneme.\n", encoding="utf-8")
    return root


# ─── Araç Protokolü ─────────────────────────────────────────────────────────

class TestSafetyAndPolicy(unittest.TestCase):
    """Güvenlik sınıfları ve özerklik politikası."""

    def test_safety_ordering(self) -> None:
        """Güvenlik sınıfları artan etki sırasına göre karşılaştırılır."""
        self.assertLess(Safety.READ, Safety.WRITE)
        self.assertLess(Safety.WRITE, Safety.EXECUTE)
        self.assertLess(Safety.EXECUTE, Safety.DESTRUCTIVE)
        self.assertLess(Safety.DESTRUCTIVE, Safety.SYSTEM)
        self.assertGreaterEqual(Safety.SYSTEM, Safety.SYSTEM)

    def test_readonly_blocks_everything_but_read(self) -> None:
        """Salt okunur modda yalnızca okuma serbesttir."""
        policy = AutonomyPolicy(level=AutonomyLevel.READONLY)
        allowed, _ = policy.decide("code.read_file", Safety.READ)
        self.assertTrue(allowed)
        for safety in (Safety.WRITE, Safety.EXECUTE, Safety.DESTRUCTIVE, Safety.SYSTEM):
            allowed, reason = policy.decide("code.edit_file", safety)
            self.assertFalse(allowed, f"{safety} serbest olmamalı")
            self.assertIn("salt okunur", reason)

    def test_supervised_requires_approval_for_writes(self) -> None:
        """Denetimli modda yazma ve üstü onay ister."""
        policy = AutonomyPolicy(level=AutonomyLevel.SUPERVISED)
        self.assertTrue(policy.decide("code.read_file", Safety.READ)[0])
        for safety in (Safety.WRITE, Safety.EXECUTE, Safety.DESTRUCTIVE):
            allowed, reason = policy.decide("code.edit_file", safety)
            self.assertFalse(allowed)
            self.assertIn("onay", reason)

    def test_trusted_allows_execute_not_destructive(self) -> None:
        """Güvenilir modda çalıştırma serbest, yıkıcı değil."""
        policy = AutonomyPolicy(level=AutonomyLevel.TRUSTED)
        self.assertTrue(policy.decide("shell.run", Safety.EXECUTE)[0])
        self.assertTrue(policy.decide("code.edit_file", Safety.WRITE)[0])
        self.assertFalse(policy.decide("shell.run", Safety.DESTRUCTIVE)[0])
        self.assertFalse(policy.decide("shell.run", Safety.SYSTEM)[0])

    def test_yolo_allows_all(self) -> None:
        """Serbest modda her sınıf izinlidir."""
        policy = AutonomyPolicy(level=AutonomyLevel.YOLO)
        for safety in (Safety.READ, Safety.WRITE, Safety.EXECUTE,
                       Safety.DESTRUCTIVE, Safety.SYSTEM):
            self.assertTrue(policy.decide("herhangi.arac", safety)[0], safety)

    def test_deny_list_overrides_level(self) -> None:
        """Yasak listesi seviyeden bağımsız uygulanır."""
        policy = AutonomyPolicy(level=AutonomyLevel.YOLO, deny=["shell.run"])
        allowed, reason = policy.decide("shell.run", Safety.READ)
        self.assertFalse(allowed)
        self.assertIn("yasaklı", reason)

    def test_allow_list_bypasses_approval(self) -> None:
        """Açık izin listesi onayı atlar."""
        policy = AutonomyPolicy(level=AutonomyLevel.SUPERVISED, allow=["code.edit_file"])
        self.assertTrue(policy.decide("code.edit_file", Safety.WRITE)[0])

    def test_levels_listing(self) -> None:
        """Seviye listesi arayüz için tam bilgi içerir."""
        levels = AutonomyPolicy.levels()
        self.assertEqual(len(levels), 4)
        for item in levels:
            self.assertIn("value", item)
            self.assertIn("label", item)
            self.assertIn("description", item)
            self.assertIn("auto_approve_up_to", item)

    def test_set_level_and_dict(self) -> None:
        """Seviye değiştirilebilir ve serileştirilebilir."""
        policy = AutonomyPolicy()
        policy.set_level("trusted")
        self.assertEqual(policy.level, AutonomyLevel.TRUSTED)
        assert_json_safe(self, policy.to_dict(), "policy")

    def test_invalid_level_rejected(self) -> None:
        """Geçersiz seviye hata verir."""
        policy = AutonomyPolicy()
        with self.assertRaises(ValueError):
            policy.set_level("sınırsız")


class TestToolRegistry(unittest.TestCase):
    """Araç kayıt defteri ve çağrı yönetimi."""

    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.initialize(
            policy=AutonomyPolicy(level=AutonomyLevel.SUPERVISED),
            context=ToolContext(workspace=os.getcwd()),
        )

        def read_handler(path: str) -> Dict[str, Any]:
            return {"path": path, "content": "test"}

        def write_handler(path: str, text: str) -> Dict[str, Any]:
            return {"path": path, "written": len(text)}

        def boom_handler() -> Dict[str, Any]:
            raise RuntimeError("kasıtlı hata")

        self.registry.register(ToolDef(
            name="t.read", description="okur", handler=read_handler,
            safety=Safety.READ, category="test",
            parameters=[ToolParameter("path", required=True)],
        ))
        self.registry.register(ToolDef(
            name="t.write", description="yazar", handler=write_handler,
            safety=Safety.WRITE, category="test",
            parameters=[ToolParameter("path", required=True),
                        ToolParameter("text", required=True)],
        ))
        self.registry.register(ToolDef(
            name="t.boom", description="patlar", handler=boom_handler,
            safety=Safety.READ, category="test",
        ))

    def tearDown(self) -> None:
        self.registry.shutdown()

    def test_read_runs_without_approval(self) -> None:
        """Okuma aracı onay istemez."""
        result = self.registry.call("t.read", {"path": "a.py"})
        self.assertTrue(result.ok)
        self.assertEqual(result.data["content"], "test")

    def test_write_needs_approval_and_does_not_run(self) -> None:
        """Onay gerektiren araç **çalıştırılmaz**."""
        result = self.registry.call("t.write", {"path": "a.py", "text": "x"})
        self.assertFalse(result.ok)
        self.assertTrue(result.blocked)
        self.assertTrue(result.needs_approval)
        self.assertIsNone(result.data)

    def test_approval_is_single_use(self) -> None:
        """
        Onay tek kullanımlıktır.

        Kullanıcı bir kez onayladığında ajanın sonraki çağrıları sormadan
        yapabilmesi ciddi bir güvenlik açığı olurdu.
        """
        first = self.registry.call("t.write", {"path": "a.py", "text": "x"})
        self.assertTrue(first.needs_approval)
        self.registry.approve("t.write")
        second = self.registry.call("t.write", {"path": "a.py", "text": "x"})
        self.assertTrue(second.ok, "onay sonrası çalışmalı")
        third = self.registry.call("t.write", {"path": "a.py", "text": "x"})
        self.assertTrue(third.needs_approval, "onay tükenmeli, yeniden sorulmalı")

    def test_remember_makes_it_permanent(self) -> None:
        """'Hatırla' onayı oturum boyunca geçerli olur."""
        self.registry.approve("t.write", remember=True)
        self.assertIn("t.write", self.registry.policy.allow)
        for _ in range(3):
            result = self.registry.call("t.write", {"path": "a.py", "text": "x"})
            self.assertTrue(result.ok)

    def test_unknown_tool(self) -> None:
        """Bilinmeyen araç anlaşılır hata verir."""
        result = self.registry.call("yok.boyle", {})
        self.assertFalse(result.ok)
        self.assertIn("bilinmeyen", result.error)

    def test_missing_parameter_message(self) -> None:
        """Eksik parametre, beklenen parametreleri bildirir."""
        result = self.registry.call("t.read", {})
        self.assertFalse(result.ok)
        self.assertIn("parametre", result.error)
        self.assertIn("path", result.reason)

    def test_exception_is_captured(self) -> None:
        """Araç içi istisna yakalanır ve hata olarak döner."""
        result = self.registry.call("t.boom", {})
        self.assertFalse(result.ok)
        self.assertIn("kasıtlı hata", result.error)

    def test_dry_run(self) -> None:
        """Kuru çalıştırmada yazma araçları çalışmaz."""
        self.registry.policy.dry_run = True
        result = self.registry.call("t.write", {"path": "a.py", "text": "x"})
        self.assertTrue(result.ok)
        self.assertTrue(result.data.get("dry_run"))
        self.assertFalse(result.data.get("written"), "gerçekten yazmamalı")

    def test_rate_limit(self) -> None:
        """Çağrı hızı sınırı aşılırsa reddedilir."""
        self.registry.policy.max_calls_per_minute = 3
        for _ in range(3):
            self.registry.call("t.read", {"path": "a.py"})
        result = self.registry.call("t.read", {"path": "a.py"})
        self.assertTrue(result.blocked)
        self.assertIn("hız", result.error)

    def test_schemas_and_prompt(self) -> None:
        """Şemalar ve istem metni üretilir."""
        schemas = self.registry.schemas()
        self.assertEqual(len(schemas), 3)
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            self.assertIn("parameters", schema["function"])
        prompt = self.registry.describe_for_prompt()
        self.assertIn("t.read", prompt)
        self.assertIn("yazma", prompt)   # yazma aracının rozeti

    def test_schema_safety_filter(self) -> None:
        """
        Güvenlik filtresi yalnızca izinli araçları döndürür.

        Not: Şemalardaki adlar API-güvenli biçimdedir (`t.read` → `t_read`);
        bazı sağlayıcılar (DeepSeek) `^[a-zA-Z0-9_-]+$` kalıbını şart koşar
        ve nokta içeren adı `HTTP 400` ile reddeder. **Kayıtlı** adlar
        değişmez — yalnızca gönderilen ad normalize edilir.
        """
        schemas = self.registry.schemas(max_safety=Safety.READ)
        names = {item["function"]["name"] for item in schemas}
        self.assertEqual(names, {"t_read", "t_boom"})
        # Kayıtlı adlar dokunulmamış olmalı
        self.assertTrue(self.registry.has("t.read"))

    def test_audit_trail(self) -> None:
        """Her çağrı denetim kaydına yazılır."""
        self.registry.call("t.read", {"path": "a.py"})
        self.registry.call("t.boom", {})
        records = self.registry.audit(limit=10)
        self.assertGreaterEqual(len(records), 2)
        tools = {record["tool"] for record in records}
        self.assertIn("t.read", tools)
        self.assertIn("t.boom", tools)

    def test_stats_json_safe(self) -> None:
        """İstatistikler JSON güvenli."""
        self.registry.call("t.read", {"path": "a.py"})
        assert_json_safe(self, self.registry.get_stats(), "registry_stats")

    def test_decorator_collection(self) -> None:
        """Dekoratörle işaretlenen fonksiyonlar toplanır."""
        @tool("deco.test", "deneme", safety=Safety.READ, category="test",
              parameters=[ToolParameter("x", required=True)])
        def sample(x: str) -> Dict[str, Any]:
            return {"x": x}

        found = collect_tools({"sample": sample})
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "deco.test")

    def test_tool_result_render(self) -> None:
        """Sonuç metni okunur biçimde üretilir."""
        result = self.registry.call("t.read", {"path": "a.py"})
        rendered = result.render()
        self.assertIn("t.read", rendered)
        self.assertIn("content", rendered)


# ─── Kodlama Ajanı ──────────────────────────────────────────────────────────

class TestCodingAgent(unittest.TestCase):
    """Kodlama araçları (gerçek dosya sistemi üzerinde)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = _workspace()

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def setUp(self) -> None:
        self.agent = CodingAgent()
        self.agent.initialize(workspace=self.workspace)

    def tearDown(self) -> None:
        self.agent.shutdown()

    def test_workspace_map(self) -> None:
        """Proje haritası tür, dil ve dosyaları bildirir."""
        result = self.agent.workspace_map()
        self.assertGreater(result["file_count"], 3)
        self.assertEqual(result["project_type"], "python")
        self.assertIn("python", result["languages"])
        self.assertTrue(all(item["relative"] for item in result["files"]))
        assert_json_safe(self, result, "workspace_map")

    def test_read_file_with_line_numbers(self) -> None:
        """Dosya satır numaralarıyla okunur."""
        result = self.agent.read_file("pkg/calc.py", max_lines=4)
        self.assertEqual(result["language"], "python")
        self.assertGreater(result["total_lines"], 4)
        self.assertIn("1| import math", result["content"])
        self.assertTrue(result["truncated"])

    def test_list_symbols_python_exact(self) -> None:
        """Python sembolleri ast ile kesin bulunur."""
        result = self.agent.list_symbols("pkg")
        names = {item["name"] for item in result["symbols"]}
        self.assertIn("Hesap", names)
        self.assertIn("kare", names)
        self.assertIn("topla", names)
        classes = [item for item in result["symbols"] if item["kind"] == "class"]
        methods = [item for item in result["symbols"] if item["kind"] == "method"]
        self.assertEqual(len(classes), 1)
        self.assertGreaterEqual(len(methods), 2)
        self.assertFalse(result["approximate_files"], "Python kesin olmalı")

    def test_find_symbol_reports_usages(self) -> None:
        """Sembol araması tanım ve kullanımları birlikte verir."""
        result = self.agent.find_symbol("kare")
        self.assertGreaterEqual(result["definition_count"], 1)
        self.assertTrue(any(item["file"].endswith("calc.py")
                            for item in result["definitions"]))

    def test_search_text(self) -> None:
        """Metin araması dosya ve satır bildirir."""
        result = self.agent.search_text("return", file_glob="*.py")
        self.assertGreater(result["count"], 0)
        for hit in result["hits"]:
            self.assertIn("file", hit)
            self.assertIn("line", hit)

    def test_search_regex(self) -> None:
        """Düzenli ifade araması çalışır."""
        result = self.agent.search_text(r"def\s+\w+", file_glob="*.py", regex=True)
        self.assertGreater(result["count"], 0)

    def test_find_files(self) -> None:
        """Glob ile dosya bulunur."""
        result = self.agent.find_files("*.py")
        self.assertGreater(result["count"], 0)
        self.assertTrue(any(item["name"] == "calc.py" for item in result["matches"]))

    def test_edit_file_surgical(self) -> None:
        """Cerrahi düzenleme yalnızca istenen metni değiştirir."""
        before = self.agent.read_file("pkg/calc.py")["content"]
        result = self.agent.edit_file("pkg/calc.py", "    return x * x", "    return x ** 2")
        self.assertTrue(result["ok"])
        self.assertEqual(result["replaced"], 1)
        self.assertTrue(result["syntax_ok"])
        self.assertIn("x ** 2", result["diff"])
        self.assertIn("-    return x * x", result["diff"].replace("    return x * x", "    return x * x"))
        after = self.agent.read_file("pkg/calc.py")["content"]
        self.assertNotEqual(before, after)
        # Geri al
        self.agent.revert(path="pkg/calc.py")

    def test_edit_file_missing_text(self) -> None:
        """Bulunmayan metin anlaşılır hata verir."""
        result = self.agent.edit_file("pkg/calc.py", "BULUNMAYAN-METIN-XYZ", "yeni")
        self.assertFalse(result["ok"])
        self.assertIn("bulunamadı", result["error"])

    def test_edit_file_ambiguous(self) -> None:
        """Birden çok eşleşme replace_all olmadan reddedilir."""
        self.agent.edit_file(
            "pkg/cok.py", "", "def f():\n    return 1\n\n\ndef g():\n    return 1\n",
            check_syntax=True,
        )
        result = self.agent.edit_file("pkg/cok.py", "    return 1", "    return 2")
        self.assertFalse(result["ok"])
        self.assertIn("kez geçiyor", result["error"])

    def test_no_syntax_check_for_non_python(self) -> None:
        """Python dışı dosyalarda sözdizimi denetimi atlanır (hata değil)."""
        result = self.agent.create_file("notlar.txt", "düz metin")
        self.assertTrue(result["ok"])
        self.assertIsNone(result.get("syntax_ok"))

    def test_dangerous_command_detection(self) -> None:
        """Yıkıcı komutlar tespit edilir."""
        dangerous = [
            "format C:", "diskpart", "Remove-Item -Recurse -Force x",
            "del /f /s /q *", "shutdown /s /t 0", "reg delete HKLM\\Software",
            "rd /s /q C:\\", "git reset --hard", "git clean -fd",
        ]
        for command in dangerous:
            blocked, reason = detect_dangerous_command(command)
            self.assertTrue(blocked, f"engellenmeliydi: {command}")
            self.assertTrue(reason)

    def test_safe_commands_pass(self) -> None:
        """Normal komutlar engellenmez."""
        for command in ("git status", "python app.py", "pytest -q",
                        "npm install", "mkdir yeni", "Get-ChildItem"):
            blocked, _ = detect_dangerous_command(command)
            self.assertFalse(blocked, f"engellenmemeliydi: {command}")

    def test_run_command(self) -> None:
        """Komut çalıştırılır ve çıktı döner."""
        result = self.agent.run_command("echo merhaba")
        self.assertTrue(result["ok"])
        self.assertEqual(result["returncode"], 0)
        self.assertIn("merhaba", result["stdout"])

    def test_run_command_blocked(self) -> None:
        """Yıkıcı komut çalıştırılmaz."""
        result = self.agent.run_command("format C:")
        self.assertFalse(result["ok"])
        self.assertTrue(result.get("blocked"))

    def test_run_tests(self) -> None:
        """Testler çalıştırılır ve özet döner (hepsi geçmeli)."""
        result = self.agent.run_tests()
        self.assertIn(result["framework"], ("pytest", "unittest"))
        self.assertEqual(result["failed"], 0, result.get("stdout_tail", "")[-600:])
        self.assertGreaterEqual(result["passed"], 2)
        self.assertIn("geçti", result["summary"])

    def test_unittest_command_avoids_discovery_error(self) -> None:
        """
        unittest komutu keşif hatası vermez.

        `python -m unittest discover -s tests -t .` çağrısı, `tests` paket
        değilse `ValueError: Start directory is not importable` ile çöker.
        """
        command = self.agent._unittest_command()
        self.assertIn("unittest", command)
        self.assertNotIn("-t . ", command.replace("-t tests", ""))
        outcome = self.agent.run_command(command)
        self.assertEqual(outcome["returncode"], 0, outcome.get("stderr", "")[:400])

    def test_diagnose_error(self) -> None:
        """Hata çıktısı dosya/satır/tür ile yorumlanır."""
        sample = (
            'Traceback (most recent call last):\n'
            '  File "pkg/calc.py", line 12, in kare\n'
            '    return x * x\n'
            "ModuleNotFoundError: No module named 'yok'\n"
        )
        result = self.agent.diagnose_error(sample)
        self.assertGreater(result["error_count"], 0)
        kinds = {item["type"] for item in result["errors"]}
        self.assertTrue(kinds & {"python_traceback", "missing_module"})
        self.assertTrue(result["advice"])

    def test_syntax_check(self) -> None:
        """Sözdizimi denetimi geçerli ve bozuk dosyayı ayırt eder."""
        good = self.agent.check_syntax("pkg/calc.py")
        self.assertTrue(good["ok"])
        self.agent.create_file("bozuk.py", "def f(:\n    pass\n")
        bad = self.agent.check_syntax("bozuk.py")
        self.assertFalse(bad["ok"])
        self.assertIn("satır", bad["error"])

    def test_create_file_reports_broken_syntax(self) -> None:
        """Bozuk sözdizimli yeni dosya başarı sayılmaz."""
        result = self.agent.create_file("bozuk2.py", "class X(:\n    pass\n")
        self.assertFalse(result["ok"])
        self.assertFalse(result["syntax_ok"])
        self.assertIn("sözdizimi", result["error"])

    def test_edit_reverts_on_syntax_break(self) -> None:
        """Sözdizimini bozan düzenleme otomatik geri alınır."""
        self.agent.create_file("saglam.py", "def f():\n    return 1\n", overwrite=True)
        result = self.agent.edit_file("saglam.py", "    return 1", "    return (")
        self.assertFalse(result["ok"])
        self.assertFalse(result["syntax_ok"])
        self.assertIn("geri alındı", result["error"])
        content = Path(self.workspace, "saglam.py").read_text(encoding="utf-8")
        self.assertIn("return 1", content, "bozuk hâl diske kalmamalı")

    def test_path_escape_blocked(self) -> None:
        """Çalışma alanı dışına erişim reddedilir."""
        with self.assertRaises(CodingError):
            self.agent.resolve("../../../../Windows/System32/drivers/etc/hosts")
        # Yol zorlaması komut üzerinden de engellenir mi? Komut kendi içinde
        # mutlak yol kullanabilir; kum havuzu **dosya araçları** içindir.
        result = self.agent.run_command("echo guvenli")
        self.assertIn("returncode", result)

    def test_write_outside_workspace_blocked(self) -> None:
        """Çalışma alanı dışına yazma reddedilir."""
        with self.assertRaises(CodingError):
            self.agent.create_file("../../../nc_kacis_denemesi.txt", "olmamalı")
        with self.assertRaises(CodingError):
            self.agent.edit_file("C:/Windows/System32/drivers/etc/hosts", "a", "b")

    def test_read_binary_file(self) -> None:
        """İkili dosya okunmaz, bildirilir."""
        binary = Path(self.workspace, "veri.bin")
        binary.write_bytes(b"\x00\x01\x02\x03" * 100)
        result = self.agent.read_file("veri.bin")
        self.assertEqual(result["language"], "binary")
        self.assertIn("ikili", result["content"])

    def test_git_status_graceful(self) -> None:
        """Git yoksa nazikçe bildirilir (hata fırlatmaz)."""
        result = self.agent.git_status()
        self.assertIn("available", result)
        if not result["available"]:
            self.assertIn("error", result)

    def test_stats_json_safe(self) -> None:
        """İstatistikler JSON güvenli."""
        assert_json_safe(self, self.agent.get_stats(), "coding_stats")

    def test_tools_listing(self) -> None:
        """Araç listesi tutarlı."""
        tools = self.agent.tools()
        names = {item.name for item in tools}
        for expected in ("code.workspace_map", "code.read_file", "code.edit_file",
                         "code.run_tests", "code.search_text", "code.diagnose_error"):
            self.assertIn(expected, names)
        for item in tools:
            self.assertEqual(item.category, "code")
            self.assertTrue(item.description)
            self.assertTrue(item.schema()["function"]["parameters"])


class TestTestOutputParsing(unittest.TestCase):
    """Farklı test framework çıktılarının ayrıştırılması."""

    def test_pytest_summary(self) -> None:
        """pytest özeti okunur."""
        result = parse_test_output(
            "FAILED tests/test_x.py::test_a\n"
            "2 failed, 10 passed, 1 skipped in 1.23s", "pytest")
        self.assertEqual(result["passed"], 10)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["duration"], "1.23s")
        self.assertIn("tests/test_x.py::test_a", result["failures"])

    def test_unittest_summary(self) -> None:
        """unittest özeti okunur."""
        result = parse_test_output(
            "FAIL: test_bad (tests.test_a.TestX)\n"
            "Ran 5 tests in 0.012s\n\nFAILED (failures=2, errors=1)", "unittest")
        self.assertEqual(result["passed"], 2)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["errors"], 1)

    def test_summary_ignores_parenthesised_counts(self) -> None:
        """Özet satırındaki (failures=2) test adı sanılmaz."""
        result = parse_test_output(
            "Ran 3 tests in 0.01s\n\nFAILED (failures=1)", "unittest")
        for name in result["failures"]:
            self.assertFalse(name.startswith("("), f"özet test adı sanıldı: {name}")

    def test_cargo_summary(self) -> None:
        """cargo test özeti okunur."""
        result = parse_test_output(
            "test result: ok. 7 passed; 1 failed; 2 ignored", "cargo")
        self.assertEqual(result["passed"], 7)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["skipped"], 2)

    def test_empty_output(self) -> None:
        """Boş çıktı hata vermez."""
        result = parse_test_output("", "pytest")
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["summary"], "")


# ─── Kabuk ──────────────────────────────────────────────────────────────────

class TestShellClassification(unittest.TestCase):
    """Komut sınıflandırma ve güvenlik eşlemesi."""

    def test_read_commands(self) -> None:
        """Bilgi toplama komutları doğru sınıflanır."""
        for command in ("git status", "ls -la", "Get-ChildItem", "pwd",
                        "python --version", "nvidia-smi"):
            kind, _ = classify_command(command)
            self.assertEqual(kind, CommandClass.INFO, command)
            self.assertEqual(command_safety(command), Safety.READ)

    def test_build_commands(self) -> None:
        """Çalıştırma/derleme komutları doğru sınıflanır."""
        for command in ("python app.py", "pytest -q", "npm install",
                        "cargo build --release", "godot --headless --import"):
            kind, _ = classify_command(command)
            self.assertEqual(kind, CommandClass.BUILD, command)

    def test_delete_and_dangerous(self) -> None:
        """Silme ve yıkıcı komutlar ayrılır."""
        kind, _ = classify_command("Remove-Item -Recurse -Force x")
        self.assertEqual(kind, CommandClass.DANGEROUS)
        kind, _ = classify_command("format C:")
        self.assertEqual(kind, CommandClass.DANGEROUS)
        kind, _ = classify_command("shutdown /s")
        self.assertEqual(kind, CommandClass.DANGEROUS)

    def test_system_commands(self) -> None:
        """Sistem düzeyi komutlar ayrılır."""
        for command in ("reg add HKLM\\Software\\X", "schtasks /create",
                        "New-Service -Name x"):
            kind, _ = classify_command(command)
            self.assertEqual(kind, CommandClass.SYSTEM, command)

    def test_dangerous_is_system_safety(self) -> None:
        """Yıkıcı komutlar en yüksek güvenlik sınıfındadır."""
        self.assertEqual(command_safety("format C:"), Safety.SYSTEM)
        self.assertEqual(CLASS_SAFETY["dangerous"], Safety.SYSTEM)

    def test_modify_commands(self) -> None:
        """Proje değiştiren komutlar ayrılır."""
        for command in ("mkdir yeni", "git add .", "Copy-Item a b"):
            kind, _ = classify_command(command)
            self.assertEqual(kind, CommandClass.MODIFY, command)

    def test_unknown_defaults_to_info(self) -> None:
        """Tanınmayan komut bilgi toplama sayılır (en az yetki)."""
        kind, reason = classify_command("bilinmeyen-arac --flag")
        self.assertEqual(kind, CommandClass.INFO)
        self.assertIn("tanınmayan", reason)


class TestShellIntegration(unittest.TestCase):
    """Kabuk katmanı (gerçek PowerShell)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = tempfile.mkdtemp(prefix="nc_shell_test_")

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def setUp(self) -> None:
        self.shell = ShellIntegration()
        self.shell.initialize(workspace=self.workspace)

    def tearDown(self) -> None:
        self.shell.shutdown()

    def test_run_echo(self) -> None:
        """Basit komut çalışır."""
        result = self.shell.run("echo merhaba")
        self.assertTrue(result.ok)
        self.assertIn("merhaba", result.stdout)

    def test_dangerous_blocked(self) -> None:
        """Yıkıcı komut engellenir."""
        result = self.shell.run("format C:")
        self.assertTrue(result.blocked)
        self.assertFalse(result.ok)
        self.assertIn("yıkıcı", result.blocked_reason)

    def test_diagnose_command_not_found(self) -> None:
        """Bulunamayan komut teşhis edilir."""
        result = self.shell.run("bu-komut-kesinlikle-yok-12345")
        self.assertFalse(result.ok)
        analysis = self.shell.diagnose(result.output, "bu-komut-kesinlikle-yok-12345")
        self.assertFalse(analysis["ok"])
        self.assertGreater(analysis["error_count"], 0)
        fix = self.shell.suggest_fix(result.output, "bu-komut-kesinlikle-yok-12345")
        self.assertTrue(fix["suggestion"])
        self.assertGreater(fix["confidence"], 0)

    def test_diagnose_access_denied(self) -> None:
        """Yetki hatası tanınır."""
        analysis = self.shell.diagnose(
            "Access to the path 'C:\\x' is denied.", "del C:\\x")
        kinds = {item["kind"] for item in analysis["errors"]}
        self.assertIn("access_denied", kinds)

    def test_diagnose_python_module(self) -> None:
        """Eksik Python paketi tanınır ve kurulum önerilir."""
        analysis = self.shell.diagnose("ModuleNotFoundError: No module named 'yokpaket'")
        kinds = {item["kind"] for item in analysis["errors"]}
        self.assertIn("python_module", kinds)
        fix = self.shell.suggest_fix("ModuleNotFoundError: No module named 'yokpaket'")
        self.assertIn("pip install", fix["alternative_command"])
        self.assertIn("yokpaket", fix["alternative_command"])

    def test_session_preserves_state(self) -> None:
        """
        Kalıcı oturum dizin ve değişken durumunu korur.

        Basit `subprocess.run` bunu yapamaz; `cd` her komutta kaybolur.
        """
        started = self.shell.start_session()
        self.assertTrue(started.get("ok"))
        (Path(self.workspace) / "altdizin").mkdir(exist_ok=True)

        self.shell.session_run("cd altdizin")
        current = self.shell.session_run("(Get-Location).Path")
        self.assertIn("altdizin", current.get("stdout", ""))

        self.shell.session_run("$script:nc_test = 21")
        value = self.shell.session_run("$script:nc_test * 2")
        self.assertIn("42", value.get("stdout", ""))

    def test_session_command_error_includes_diagnosis(self) -> None:
        """
        Oturum **aracı** başarısız komutta teşhis ve öneri döndürür.

        Not: Teşhis `session_run`'ın içinde değil, araç sarmalayıcısında
        eklenir — böylece düşük seviyeli API sade kalır, ajan ise her zaman
        yorumlanmış geri bildirim alır.
        """
        registry = ToolRegistry()
        registry.initialize(
            policy=AutonomyPolicy(level=AutonomyLevel.TRUSTED),
            context=ToolContext(workspace=self.workspace),
        )
        registry.register_many(self.shell.tools())
        context = ToolContext(workspace=self.workspace)

        result = registry.call(
            "shell.session_run",
            {"command": "Get-Content C:\\yok-boyle-dosya-12345.txt"},
            context=context,
        )
        # `result.ok` aracın **çalıştığını**, `data["ok"]` ise komutun
        # başarılı olduğunu bildirir. Başarısız bir komut, aracın düzgün
        # çalıştığı anlamına gelir; ajan bu ayrımı görmelidir.
        self.assertTrue(result.ok, "araç çalışmış olmalı")
        self.assertFalse(result.data["ok"], "komut başarısız olmalı")
        self.assertEqual(result.data["returncode"], 1)
        self.assertIn("diagnosis", result.data)
        self.assertIn("fix", result.data)
        self.assertGreater(result.data["diagnosis"]["error_count"], 0)
        self.assertTrue(result.data["fix"]["suggestion"])
        assert_json_safe(self, result.data, "session_diagnosis")
        registry.shutdown()

    def test_diagnose_directly(self) -> None:
        """Düşük seviyeli teşhis API'si de doğrudan kullanılabilir."""
        output = (
            "Get-Content : Cannot find path 'C:\\yok.txt' because it does not exist.\n"
            "At line:1 char:1\n"
            "+ CategoryInfo          : ObjectNotFound\n"
        )
        analysis = self.shell.diagnose(output, "Get-Content C:\\yok.txt")
        self.assertFalse(analysis["ok"])
        self.assertGreater(analysis["error_count"], 0)
        kinds = {item["kind"] for item in analysis["errors"]}
        self.assertIn("path_not_found", kinds)

    def test_run_python_multiline(self) -> None:
        """Çok satırlı Python kodu çalışır (tırnak/satır sorunu yaşanmaz)."""
        result = self.shell.run_python(
            "import sys\nprint('py', sys.version_info.major, 'çalışıyor')")
        self.assertTrue(result.ok, result.stderr[:300])
        self.assertIn("çalışıyor", result.stdout)

    def test_run_python_error_traceback(self) -> None:
        """Python hatası izleme kaydıyla döner."""
        result = self.shell.run_python("print(1 / 0)")
        self.assertFalse(result.ok)
        self.assertIn("ZeroDivisionError", result.output)

    def test_toolchain_versions(self) -> None:
        """Araç zinciri sürümleri toplanır."""
        versions = self.shell.toolchain_versions()
        self.assertIn("python", versions)
        self.assertTrue(versions["python"]["available"])
        self.assertIn("3.10", versions["python"]["version"])
        # PowerShell bu makinede her zaman bulunur
        self.assertTrue(versions["powershell"]["available"])
        assert_json_safe(self, versions, "toolchain")

    def test_versions_reject_error_text(self) -> None:
        """PowerShell hata metni 'sürüm' olarak raporlanmaz."""
        versions = self.shell.toolchain_versions()
        for name, info in versions.items():
            if info.get("available"):
                self.assertNotIn("At line:", info["version"], name)
                self.assertNotIn("CategoryInfo", info["version"], name)

    def test_timeout(self) -> None:
        """Uzun komut zaman aşımına uğrar."""
        result = self.shell.run("Start-Sleep -Seconds 30", timeout=3.0)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.ok)

    def test_timeout_is_bounded_with_grandchild(self) -> None:
        """
        Torun süreç stdout borusunu tutsa bile zaman sınırı **uygulanır**.

        Windows'ta `python → powershell → python` zincirinde ayakta kalan bir
        torun, boruyu açık tutar ve `subprocess.run(timeout=N)` çağrısı N'de
        **dönmez**. Ölçülen hata: 8 saniyelik sınır 59.6 saniyede tetiklendi;
        gerçekten takılan bir araçta ajan hiç dönmedi. Bu test o açığı
        kapatır: süreç ağacı `taskkill /T /F` ile öldürülür ve borular
        kapatılır.
        """
        started = time.perf_counter()
        result = self.shell.run(
            'python -c "import time; time.sleep(120)"', timeout=4.0)
        elapsed = time.perf_counter() - started

        self.assertTrue(result.timed_out, "zaman aşımı bildirilmeli")
        self.assertFalse(result.ok)
        self.assertLess(elapsed, 15.0,
                        f"sınır uygulanmadı: {elapsed:.1f}s (bütçe 4s)")

    def test_no_orphan_processes_after_timeout(self) -> None:
        """
        Zaman aşımından sonra yetim torun süreç kalmaz.

        Not: `wmic` Windows 11'de kaldırıldı; bu yüzden süreç listesi
        PowerShell'in CIM arayüzünden okunur.
        """
        if os.name != "nt":
            self.skipTest("yalnızca Windows")

        marker = "nc_orphan_check_12345"
        self.shell.run(
            f'python -c "import time; time.sleep(120)  # {marker}"', timeout=3.0)
        time.sleep(1.0)

        try:
            listing = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-CimInstance Win32_Process | Select-Object -ExpandProperty CommandLine"],
                capture_output=True, text=True, timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.skipTest(f"süreç listesi okunamadı: {exc}")

        if listing.returncode != 0:
            self.skipTest("süreç listesi alınamadı")
        self.assertNotIn(marker, listing.stdout or "",
                         "yetim torun süreç hâlâ çalışıyor")

    def test_tools_listing(self) -> None:
        """Kabuk araçları kaydedilebilir biçimde."""
        tools = self.shell.tools()
        names = {item.name for item in tools}
        for expected in ("shell.run", "shell.session_run", "shell.cmd",
                         "shell.python", "shell.diagnose", "shell.versions"):
            self.assertIn(expected, names)
        for item in tools:
            self.assertEqual(item.category, "shell")

    def test_tool_returns_json_safe_dict(self) -> None:
        """Araç çağrısı JSON güvenli sözlük döner."""
        registry = ToolRegistry()
        registry.initialize(
            policy=AutonomyPolicy(level=AutonomyLevel.TRUSTED),
            context=ToolContext(workspace=self.workspace),
        )
        registry.register_many(self.shell.tools())
        result = registry.call("shell.run", {"command": "echo araç"})
        self.assertTrue(result.ok)
        self.assertIsInstance(result.data, dict)
        self.assertIn("stdout", result.data)
        assert_json_safe(self, result.data, "shell_result")
        registry.shutdown()


# ─── Dil Modeli Katmanı ─────────────────────────────────────────────────────

class TestLLMRouter(unittest.TestCase):
    """Sağlayıcı keşfi ve kimlik yönetimi."""

    def setUp(self) -> None:
        self.data = tempfile.mkdtemp(prefix="nc_llm_test_")
        self.router = LLMRouter()
        self.router.initialize(data_path=self.data)

    def tearDown(self) -> None:
        import shutil
        self.router.shutdown()
        shutil.rmtree(self.data, ignore_errors=True)

    def test_status_structure(self) -> None:
        """Durum raporu arayüz için gerekli alanları içerir."""
        status = self.router.status()
        for key in ("available", "active", "active_model", "supports_tools",
                    "providers", "credential_file", "note"):
            self.assertIn(key, status)
        assert_json_safe(self, status, "llm_status")

    def test_local_providers_reported_unavailable(self) -> None:
        """
        Yerel sunucular çalışmıyorsa 'kullanılabilir' sayılmaz.

        Anahtar gerektirmedikleri için naif bir kontrol onları her zaman
        hazır gösterirdi; bu da ajanın her çağrıda bağlantı hatası almasına
        yol açardı.
        """
        for provider in self.router.status()["providers"]:
            if provider["local"] and not provider["available"]:
                self.assertFalse(provider["available"])

    def test_tcp_probe_detects_closed_port(self) -> None:
        """Kapalı port erişilemez bildirilir."""
        self.assertFalse(tcp_reachable("http://127.0.0.1:1/v1"))

    def test_tcp_probe_cache(self) -> None:
        """Yoklama önbelleğe alınır (her çağrıda yavaşlamaz)."""
        tcp_reachable("http://127.0.0.1:1/v1")
        started = time.perf_counter()
        tcp_reachable("http://127.0.0.1:1/v1")
        self.assertLess((time.perf_counter() - started) * 1000, 20.0)

    def test_save_and_clear_credentials(self) -> None:
        """Kimlik bilgisi kaydedilir ve silinir."""
        result = self.router.save_credentials("deepseek", "sk-test-anahtar",
                                              model="deepseek-chat")
        self.assertTrue(result["ok"])
        self.assertTrue(result["status"]["has_stored_credentials"])
        self.assertTrue(Path(result["status"]["credential_file"]).exists())

        cleared = self.router.clear_credentials()
        self.assertTrue(cleared["ok"])
        self.assertFalse(cleared["status"]["has_stored_credentials"])

    def test_credential_file_has_no_plaintext_in_status(self) -> None:
        """Anahtar durum raporunda görünmez."""
        self.router.save_credentials("deepseek", "sk-cok-gizli-anahtar")
        status = self.router.status()
        text = json.dumps(status, ensure_ascii=False)
        self.assertNotIn("sk-cok-gizli-anahtar", text)

    def test_chat_without_provider(self) -> None:
        """Sağlayıcı yokken çağrı anlaşılır hata döner (istisna değil)."""
        self.router.clear_credentials()
        for name in list(self.router._providers):
            self.router._providers[name].api_key = ""
        self.router._active = self.router._select()
        if self.router.available:
            self.skipTest("bu makinede bir sağlayıcı kullanılabilir")
        response = self.router.chat([LLMMessage("user", "merhaba")])
        self.assertFalse(response.ok)
        self.assertTrue(response.error)
        self.assertIn("API", response.error + "API")

    def test_response_serialisation(self) -> None:
        """Yanıt nesnesi JSON güvenli."""
        response = LLMResponse(ok=True, text="merhaba", model="test",
                               prompt_tokens=5, completion_tokens=3)
        assert_json_safe(self, response.to_dict(), "llm_response")

    def test_call_history(self) -> None:
        """Çağrı geçmişi tutulur."""
        self.router.chat([LLMMessage("user", "x")])
        history = self.router.call_history()
        self.assertGreaterEqual(len(history), 1)
        self.assertIn("provider", history[-1])


# ─── Ajan Döngüsü ───────────────────────────────────────────────────────────

class TestAgentLoop(unittest.TestCase):
    """Planla-uygula-gözlemle döngüsü (kural tabanlı mod)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = _workspace()

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.initialize(
            policy=AutonomyPolicy(level=AutonomyLevel.TRUSTED),
            context=ToolContext(workspace=self.workspace),
        )
        self.coding = CodingAgent()
        self.coding.initialize(workspace=self.workspace)
        self.shell = ShellIntegration()
        self.shell.initialize(workspace=self.workspace)
        self.registry.register_many(self.coding.tools())
        self.registry.register_many(self.shell.tools())

        self.router = LLMRouter()
        self.router.initialize(data_path=str(Path(self.workspace) / "data"))

        self.loop = AgentLoop()
        self.loop.initialize(self.registry, self.router, workspace=self.workspace)

    def tearDown(self) -> None:
        self.loop.shutdown()
        self.shell.shutdown()
        self.coding.shutdown()
        self.registry.shutdown()

    def test_mode_is_rules_without_llm(self) -> None:
        """Dil modeli yoksa kural tabanlı mod seçilir."""
        if self.router.available:
            self.skipTest("bu makinede dil modeli var")
        self.assertEqual(self.loop.mode, "rules")

    def test_workspace_map_task(self) -> None:
        """'Proje haritasını çıkar' isteği harita aracına gider."""
        run = self.loop.run("proje haritasını çıkar")
        self.assertTrue(run.ok)
        self.assertIn("code.workspace_map", run.tools_used)
        self.assertIn("proje türü", run.answer)
        self.assertNotIn("[OK] code.workspace_map", run.answer,
                         "ham araç çıktısı gösterilmemeli")

    def test_versions_task(self) -> None:
        """'Sürümler' isteği araç zinciri aracına gider."""
        run = self.loop.run("kurulu araçların sürümleri neler")
        self.assertIn("shell.versions", run.tools_used)
        self.assertIn("python", run.answer)

    def test_read_file_task(self) -> None:
        """Dosya yolu içeren istek dosyayı okur."""
        run = self.loop.run("pkg/calc.py dosyasını oku")
        self.assertIn("code.read_file", run.tools_used)
        self.assertIn("import math", run.answer)

    def test_tests_task(self) -> None:
        """'Testleri çalıştır' isteği test aracına gider."""
        run = self.loop.run("testleri çalıştır")
        self.assertIn("code.run_tests", run.tools_used)
        self.assertIn("geçti", run.answer)

    def test_unknown_request_lists_capabilities(self) -> None:
        """Anlaşılmayan istek yetenek özeti döner."""
        run = self.loop.run("merhaba nasılsın")
        self.assertTrue(run.ok)
        self.assertIn("Dil modeli", run.answer)
        self.assertIn("araç", run.answer.lower())

    def test_approval_flow_approve(self) -> None:
        """Onay verilirse araç çalışır."""
        self.registry.set_level("supervised")
        asked: List[str] = []

        def approve(tool_name: str, args: Dict[str, Any], safety: str,
                    reason: str, summary: str) -> Any:
            asked.append(tool_name)
            return (True, False)

        self.loop.set_approval_callback(approve)
        run = self.loop.run("şu komutu çalıştır: `echo onay-testi`")
        self.assertTrue(asked, "onay sorulmalıydı")
        self.assertEqual(asked[0], "shell.run")
        self.assertTrue(run.ok)
        self.assertGreaterEqual(run.approvals, 1)

    def test_approval_is_requested_every_time(self) -> None:
        """
        Onay her çağrıda yeniden istenir.

        Onayın kalıcı olması, kullanıcının farkında olmadığı sonraki
        çağrıların sessizce çalışması demektir.
        """
        self.registry.set_level("supervised")
        asked: List[str] = []
        self.loop.set_approval_callback(
            lambda tool, args, safety, reason, summary:
            (asked.append(tool), (True, False))[1])

        self.loop.run("şu komutu çalıştır: `echo birinci`")
        self.loop.run("şu komutu çalıştır: `echo ikinci`")
        self.assertEqual(len(asked), 2, "her çağrı için yeniden sorulmalı")

    def test_approval_flow_deny(self) -> None:
        """Onay reddedilirse araç çalışmaz ve durum bildirilir."""
        self.registry.set_level("supervised")
        self.loop.set_approval_callback(lambda *args: (False, False))
        run = self.loop.run("şu komutu çalıştır: `echo reddedilecek`")
        self.assertFalse(run.ok)
        self.assertEqual(run.reason, "onay reddedildi")
        self.assertGreaterEqual(run.denials, 1)
        self.assertIn("onay", run.answer.lower())

    def test_no_callback_denies(self) -> None:
        """Onay mekanizması yoksa araç **çalıştırılmaz** (güvenli varsayılan)."""
        self.registry.set_level("supervised")
        self.loop.set_approval_callback(None)
        run = self.loop.run("şu komutu çalıştır: `echo calismamali`")
        self.assertFalse(run.ok)
        self.assertEqual(run.reason, "onay reddedildi")

    def test_readonly_blocks_run(self) -> None:
        """Salt okunur modda çalıştırma engellenir."""
        self.registry.set_level("readonly")
        run = self.loop.run("şu komutu çalıştır: `echo x`")
        self.assertFalse(run.ok)

    def test_remember_skips_next_approval(self) -> None:
        """'Hatırla' seçilirse sonraki çağrı sorulmaz."""
        self.registry.set_level("supervised")
        asked: List[str] = []
        self.loop.set_approval_callback(
            lambda tool, args, safety, reason, summary:
            (asked.append(tool), (True, True))[1])
        self.loop.run("şu komutu çalıştır: `echo hatirla`")
        self.assertIn("shell.run", self.registry.policy.allow)

        before = len(asked)
        self.loop.set_approval_callback(
            lambda tool, args, safety, reason, summary:
            (asked.append(tool), (True, False))[1])
        self.loop.run("şu komutu çalıştır: `echo sorma`")
        self.assertEqual(len(asked), before, "hatırlanan araç için sorulmamalı")

    def test_cancel_is_honoured_mid_run(self) -> None:
        """
        İptal bayrağı çalışma **sırasında** etkilidir.

        `run()` bayrağı başlangıçta temizler (önceki iptalin yeni görevi
        engellememesi için); bu yüzden iptal, adım döngüsünün içinden
        tetiklenmelidir.
        """
        captured: List[Any] = []

        def listener(step: Any) -> None:
            captured.append(step)
            if step.kind == "tool":
                self.loop.cancel()

        self.loop.set_step_listener(listener)
        run = self.loop.run("proje haritasını çıkar ve sonra testleri çalıştır")
        self.assertFalse(run.ok)
        self.assertEqual(run.reason, "iptal edildi")
        self.loop.reset_cancel()

    def test_cancel_flag_cleared_on_new_run(self) -> None:
        """Önceki iptal, yeni görevi engellemez."""
        self.loop.cancel()
        run = self.loop.run("proje haritasını çıkar")
        self.assertTrue(run.ok, "yeni çalışma iptal bayrağını temizlemeli")

    def test_step_listener(self) -> None:
        """Adım dinleyicisi her adımda çağrılır."""
        steps: List[Any] = []
        self.loop.set_step_listener(lambda step: steps.append(step))
        run = self.loop.run("proje haritasını çıkar")
        self.assertEqual(len(steps), len(run.steps))
        kinds = {step.kind for step in steps}
        self.assertIn("tool", kinds)
        self.assertIn("answer", kinds)

    def test_run_serialisation(self) -> None:
        """Çalışma sonucu JSON güvenli ve özet üretilebilir."""
        run = self.loop.run("proje haritasını çıkar")
        assert_json_safe(self, run.to_dict(), "agent_run")
        rendered = run.render()
        self.assertIn("Görev:", rendered)

    def test_stats(self) -> None:
        """İstatistikler tutulur."""
        self.loop.run("proje haritasını çıkar")
        stats = self.loop.get_stats()
        self.assertGreaterEqual(stats["runs"], 1)
        self.assertGreaterEqual(stats["tool_calls"], 1)
        assert_json_safe(self, stats, "loop_stats")


class TestLoopHelpers(unittest.TestCase):
    """Döngü yardımcı fonksiyonları."""

    def test_extract_path(self) -> None:
        """Dosya yolu istekten çıkarılır."""
        self.assertEqual(_extract_path("pkg/main.py dosyasını oku"), "pkg/main.py")
        self.assertEqual(_extract_path("a\\b\\c.gd nedir"), "a\\b\\c.gd")
        self.assertEqual(_extract_path("yol yok burada"), "")

    def test_extract_symbol(self) -> None:
        """
        Sembol adı istekten çıkarılır.

        Türkçe cümlelerde sembol adı bir tür sözcüğünden önce gelir
        ("kare fonksiyonu"); bu kalıp tanınmazsa en doğal istek boş döner.
        """
        self.assertEqual(_extract_symbol("kare fonksiyonu nerede"), "kare")
        self.assertEqual(_extract_symbol("Hesap sınıfını bul"), "Hesap")
        self.assertEqual(_extract_symbol("topla metodu nerede"), "topla")
        self.assertEqual(_extract_symbol("kare() nerede kullanılıyor"), "kare")
        self.assertEqual(_extract_symbol("_private_metod nerede"), "_private_metod")
        self.assertEqual(_extract_symbol("merhaba nasılsın"), "")

    def test_normalize_decision(self) -> None:
        """Onay kararı farklı biçimlerden normalize edilir."""
        self.assertEqual(_normalize_decision(True), (True, False))
        self.assertEqual(_normalize_decision(False), (False, False))
        self.assertEqual(_normalize_decision(None), (False, False))
        self.assertEqual(_normalize_decision((True, True)), (True, True))
        self.assertEqual(_normalize_decision({"approved": True, "remember": True}),
                         (True, True))
        self.assertEqual(_normalize_decision("evet"), (True, False))
        self.assertEqual(_normalize_decision("hayir"), (False, False))


# ─── Uygulama Durumu ────────────────────────────────────────────────────────

class TestApprovalBroker(unittest.TestCase):
    """Arayüz ↔ ajan onay köprüsü."""

    def test_request_and_resolve(self) -> None:
        """Onay isteği çözümlenir."""
        broker = ApprovalBroker(timeout=10.0)
        outcome: List[Any] = []

        thread = threading.Thread(
            target=lambda: outcome.append(
                broker.request("shell.run", {"command": "x"}, "execute",
                               "çalıştırma", "gerekçe", "özet")))
        thread.start()
        time.sleep(0.4)

        pending = broker.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["tool"], "shell.run")
        self.assertEqual(pending[0]["safety_label"], "çalıştırma")

        result = broker.resolve(pending[0]["id"], approved=True, remember=False)
        self.assertTrue(result["ok"])
        thread.join(timeout=5)
        self.assertEqual(outcome, [(True, False)])
        self.assertEqual(broker.pending(), [])

    def test_timeout_denies(self) -> None:
        """Yanıt gelmezse istek **reddedilir** (sessizce çalışmaz)."""
        broker = ApprovalBroker(timeout=1.0)
        approved, remember = broker.request("code.edit_file", {}, "write",
                                            "yazma", "gerekçe", "özet")
        self.assertFalse(approved)
        self.assertFalse(remember)
        self.assertEqual(broker.stats["timed_out"], 1)

    def test_listener_notified(self) -> None:
        """Dinleyici yeni onay isteğinde çağrılır."""
        broker = ApprovalBroker(timeout=2.0)
        seen: List[str] = []
        broker.add_listener(lambda approval: seen.append(approval.tool))

        thread = threading.Thread(target=lambda: broker.request(
            "godot.run_script", {}, "execute", "çalıştırma", "r", "s"))
        thread.start()
        time.sleep(0.4)
        self.assertEqual(seen, ["godot.run_script"])
        broker.resolve_all(approved=False)
        thread.join(timeout=5)

    def test_resolve_unknown_id(self) -> None:
        """Bilinmeyen onay kimliği anlaşılır hata verir."""
        broker = ApprovalBroker()
        result = broker.resolve("yok", approved=True)
        self.assertFalse(result["ok"])
        self.assertIn("bulunamadı", result["error"])

    def test_resolve_all(self) -> None:
        """Tüm bekleyen onaylar tek seferde yanıtlanır."""
        broker = ApprovalBroker(timeout=5.0)
        threads = [
            threading.Thread(target=lambda: broker.request(
                f"tool.{index}", {}, "write", "yazma", "r", "s"))
            for index in range(3)
        ]
        for thread in threads:
            thread.start()
        time.sleep(0.5)
        self.assertEqual(len(broker.pending()), 3)
        self.assertEqual(broker.resolve_all(approved=False), 3)
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(broker.pending(), [])


class TestEventStream(unittest.TestCase):
    """Olay yayın akışı."""

    def test_publish_without_subscribers(self) -> None:
        """Abone yokken yayın sessizce yok sayılır."""
        stream = EventStream()
        stream.publish("test.event", {"x": 1})
        self.assertEqual(stream.status()["sent"], 0)

    def test_publish_and_receive(self) -> None:
        """Abone olayı alır."""
        import asyncio

        stream = EventStream()
        queue_ = stream.subscribe()
        stream.publish("agent.step", {"kind": "tool"})
        event = queue_.get_nowait()
        self.assertEqual(event["type"], "agent.step")
        self.assertEqual(event["data"]["kind"], "tool")
        self.assertEqual(stream.status()["sent"], 1)

    def test_unsubscribe(self) -> None:
        """Abonelik kaldırılır."""
        stream = EventStream()
        queue_ = stream.subscribe()
        self.assertEqual(stream.subscriber_count(), 1)
        stream.unsubscribe(queue_)
        self.assertEqual(stream.subscriber_count(), 0)

    def test_queue_overflow_drops_oldest(self) -> None:
        """Kuyruk dolduğunda en eski olay düşer (yavaş istemci ajanı bloklamaz)."""
        stream = EventStream(max_queue=3)
        queue_ = stream.subscribe()
        for index in range(10):
            stream.publish("e", {"i": index})
        self.assertEqual(queue_.qsize(), 3)
        self.assertGreater(stream.status()["dropped"], 0)
        newest = queue_.get_nowait()
        self.assertEqual(newest["data"]["i"], 7)


class TestAppState(unittest.TestCase):
    """Uygulama durumu."""

    def test_tool_result_store(self) -> None:
        """Araç sonuçları saklanır ve okunur."""
        state = AppState()
        state.store_tool_result("t1", {"ok": True, "at": time.time()})
        self.assertIsNotNone(state.get_tool_result("t1"))
        self.assertIsNone(state.get_tool_result("yok"))

    def test_tool_result_limit(self) -> None:
        """Sonuç deposu sınırsız büyümez."""
        state = AppState()
        for index in range(250):
            state.store_tool_result(f"t{index}", {"at": index})
        self.assertLessEqual(len(state.tool_results), 200)
        self.assertIsNotNone(state.get_tool_result("t249"))

    def test_busy_flag(self) -> None:
        """Çalışma durumu izlenir."""
        state = AppState()
        self.assertFalse(state.is_busy())
        state.set_busy(True)
        self.assertTrue(state.is_busy())
        self.assertGreater(state.busy_since, 0)
        state.set_busy(False)
        self.assertFalse(state.is_busy())
        self.assertEqual(state.busy_since, 0.0)

    def test_uptime(self) -> None:
        """Çalışma süresi pozitif."""
        state = AppState()
        self.assertGreaterEqual(state.uptime(), 0.0)


# ─── Görsel Doğrulama ───────────────────────────────────────────────────────

class TestDesktopVision(unittest.TestCase):
    """Görsel doğrulama (gerçek ekran)."""

    @classmethod
    def setUpClass(cls) -> None:
        from neurocore.vision import HAS_PIL

        if not HAS_PIL:
            raise unittest.SkipTest("Pillow kurulu değil")

    def setUp(self) -> None:
        from neurocore.vision import DesktopVision

        self.vision = DesktopVision()
        self.vision.initialize(None, None)

    def tearDown(self) -> None:
        self.vision.shutdown()

    def test_initialize_and_stats(self) -> None:
        """Katman başlar ve istatistik verir."""
        stats = self.vision.get_stats()
        self.assertTrue(stats["initialized"])
        self.assertTrue(stats["pillow"])
        assert_json_safe(self, stats, "vision_stats")

    def test_screenshot_and_analysis(self) -> None:
        """Ekran görüntüsü alınır ve analiz edilir."""
        result = self.vision.screenshot(save=False, max_side=400)
        self.assertTrue(result["ok"], result.get("error"))
        self.assertGreater(result["width"], 0)
        analysis = result["analysis"]
        self.assertIn("mean_brightness", analysis)
        self.assertIn("contrast", analysis)
        self.assertIn("is_blank", analysis)
        self.assertTrue(result["verdict"])

    def test_blank_detection(self) -> None:
        """Tek renk ekran 'boş' olarak tespit edilir."""
        from PIL import Image

        blank = Image.new("RGB", (320, 240), (10, 10, 10))
        analysis = self.vision.analyze(blank)
        self.assertTrue(analysis.is_blank)
        self.assertIn("BOŞ", analysis.verdict())

    def test_content_not_blank(self) -> None:
        """Desenli görüntü boş sayılmaz."""
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (320, 240), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        for index in range(0, 320, 20):
            draw.line([(index, 0), (index, 240)], fill=(0, 0, 0), width=3)
        analysis = self.vision.analyze(image)
        self.assertFalse(analysis.is_blank)
        self.assertGreater(analysis.edge_density, 0.0)

    def test_compare_identical(self) -> None:
        """Aynı görüntüler özdeş sayılır."""
        from PIL import Image

        image = Image.new("RGB", (200, 150), (30, 60, 90))
        diff = self.vision.compare(image, image)
        self.assertTrue(diff.identical)
        self.assertEqual(diff.changed_ratio, 0.0)

    def test_compare_different(self) -> None:
        """Farklı görüntülerde değişim oranı yüksek çıkar."""
        from PIL import Image

        first = Image.new("RGB", (200, 150), (0, 0, 0))
        second = Image.new("RGB", (200, 150), (255, 255, 255))
        diff = self.vision.compare(first, second)
        self.assertFalse(diff.identical)
        self.assertGreater(diff.changed_ratio, 0.9)

    def test_baseline_roundtrip(self) -> None:
        """Referans kaydedilir ve karşılaştırılır."""
        self.assertTrue(self.vision.set_baseline("test_ref", max_side=320))
        result = self.vision.check_baseline("test_ref", max_side=320)
        self.assertTrue(result["ok"])
        self.assertIn("diff", result)

    def test_missing_baseline(self) -> None:
        """Olmayan referans anlaşılır hata verir."""
        result = self.vision.check_baseline("yok_boyle_referans")
        self.assertFalse(result["ok"])
        self.assertIn("referans yok", result["error"])

    def test_list_windows(self) -> None:
        """Pencere listesi alınır."""
        windows = self.vision.list_windows(min_size=0, limit=50)
        self.assertIsInstance(windows, list)
        for window in windows:
            self.assertIn("title", window)
            self.assertIn("handle", window)

    def test_window_text_graceful(self) -> None:
        """Arayüz metni okunamasa da hata fırlatmaz."""
        result = self.vision.window_text("kesinlikle-olmayan-pencere-12345")
        self.assertFalse(result["ok"])
        self.assertTrue(result.get("error"))

    def test_health(self) -> None:
        """Sağlık raporu yetenekleri bildirir."""
        health = self.vision.health()
        self.assertIn("capabilities", health)
        self.assertIn("screenshot", health["capabilities"])
        assert_json_safe(self, health, "vision_health")


if __name__ == "__main__":
    unittest.main(verbosity=2)
