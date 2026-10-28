"""
Ajan Köprüsü Testleri
======================
`neurocore.agent.bridge` — ajan işlerini parçalı motora bağlar.

Neden bu testler önemli?
    Köprü, ajanın araç çağrılarını `WorkStep`'e çevirir. Yanlış kaynak
    tahmini veya bozuk bağımlılık zinciri, ajanı ya **kilitler** ya da
    **yanlış sırada** çalıştırır. Bu yüzden:

        * kaynak tahmininin makul olduğu,
        * `needs_previous` zincirinin doğru kurulduğu,
        * araç bulunamadığında çökmediği,
        * kesintiden sonra **kalan** işin doğru hesaplandığı
        * ve düşünme adımının sessizce yanlış sonuç vermediği

        doğrulanır.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import time
import unittest
from typing import Any, Dict, List

from tests.helpers import assert_json_safe

from neurocore.agent.bridge import (
    DEFAULT_RESOURCES,
    TOOL_RESOURCES,
    AgentBridge,
    PlannedStep,
    estimate_for_tool,
    resources_for_tool,
)
from neurocore.core.governor import ResourceSnapshot
from neurocore.core.work import DONE, BLOCKED, WorkStep


class FakeTool:
    """Sahte araç (kayıt defteri öğesi)."""

    def __init__(self, fn: Any) -> None:
        self.handler = fn


class FakeRegistry:
    """
    Sahte araç kayıt defteri.

    Gerçek `ToolRegistry` gibi **`call()`** sunar: köprü bu metodu tercih
    eder çünkü özerklik politikası ve onay akışını o uygular. Sahte kayıt
    defterinin de aynı arayüzü sunması gerekir; aksi hâlde test, gerçek
    kodun kullandığı yolu sınamaz.
    """

    def __init__(self, araclar: Dict[str, Any]) -> None:
        self._araclar = araclar
        self.calls: List[Dict[str, Any]] = []

    def names(self) -> List[str]:
        """Kayıtlı araç adları."""
        return sorted(self._araclar)

    def resolve(self, name: str) -> Any:
        """Aracı adıyla bulur."""
        fn = self._araclar.get(name)
        if fn is None:
            return None

        def sar(**kwargs: Any) -> Any:
            self.calls.append({"tool": name, "args": kwargs})
            return fn(**kwargs)

        return FakeTool(sar)

    def call(self, name: str, args: Dict[str, Any] | None = None,
             **kwargs: Any) -> Dict[str, Any]:
        """
        Aracı çağırır (gerçek kayıt defterindeki gibi sonuç sarmalar).

        ⚠ Önemli: sahte kayıt defteri, işleyici **onay gerektiğini**
        bildirirse (`{"needs_approval": True}` veya `ok: False`) bunu
        **üstteki `ok` alanına yansıtmalıdır**. Aksi hâlde köprü onay
        isteğini göremez ve adım yanlışlıkla başarılı sayılır — gerçek
        kayıt defterinde böyle bir davranış yoktur.
        """
        fn = self._araclar.get(name)
        if fn is None:
            return {"ok": False, "error": f"araç bulunamadı: {name}"}
        self.calls.append({"tool": name, "args": dict(args or {})})
        try:
            cikti = fn(**(args or {}))
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # İşleyici onay istiyorsa bu **üstte** görünmeli
        if isinstance(cikti, dict) and cikti.get("needs_approval"):
            return {"ok": False, "needs_approval": True,
                    "error": str(cikti.get("error") or "onay gerekir"),
                    "reason": str(cikti.get("reason") or "")}
        return {"ok": True, "data": cikti}


class FakeLoop:
    """Sahte ajan döngüsü."""

    def __init__(self, kayit: Any = None) -> None:
        self.registry = kayit


class FakeThinking:
    """Sahte düşünme döngüsü."""

    def __init__(self, *, chosen: str = "seçilen", source: str = "local",
                 confidence: str = "yüksek", fail: bool = False) -> None:
        self.chosen = chosen
        self.source = source
        self.confidence = confidence
        self.fail = fail
        self.calls = 0

    def think(self, problem: str, propose: Any, **kwargs: Any) -> Any:
        """Düşünme taklidi."""
        self.calls += 1
        from neurocore.agent.thinking import ThinkingResult

        if self.fail:
            return ThinkingResult(ok=False, error="kasıtlı")
        return ThinkingResult(ok=True, chosen=self.chosen, source=self.source,
                              confidence=self.confidence)


# ─── Kaynak Tahmini ─────────────────────────────────────────────────────────

class TestResourceEstimation(unittest.TestCase):
    """Araç kaynak tahmini."""

    def test_exact_match(self) -> None:
        r = resources_for_tool("code.run_tests")
        self.assertEqual(r, TOOL_RESOURCES["code.run_tests"])

    def test_prefix_fallback(self) -> None:
        """Tam eşleşme yoksa nokta öneki kullanılmalı."""
        r = resources_for_tool("code.yeni_arac")
        self.assertGreater(r.get("ram_mb", 0), 0)
        self.assertLessEqual(r.get("ram_mb", 0), 300.0)

    def test_unknown_tool_gets_default(self) -> None:
        r = resources_for_tool("hic.bilinmeyen")
        self.assertEqual(r, DEFAULT_RESOURCES)

    def test_empty_tool_is_safe(self) -> None:
        self.assertEqual(resources_for_tool(""), DEFAULT_RESOURCES)
        self.assertEqual(resources_for_tool(None), DEFAULT_RESOURCES)  # type: ignore[arg-type]

    def test_heavy_tools_are_heavier(self) -> None:
        """
        Ağır araçlar hafiflerden **belirgin biçimde** fazla kaynak istemeli.

        Aksi hâlde motor ağır işi kuyruğa almaz ve sistem kilitlenir.
        """
        hafif = resources_for_tool("code.read_file")
        agir = resources_for_tool("blender.render")
        self.assertGreater(agir["ram_mb"], hafif["ram_mb"] * 10)
        self.assertGreater(agir["cpu"], hafif["cpu"] * 10)

    def test_estimate_is_positive(self) -> None:
        for arac in ("code.read_file", "code.run_tests", "bilinmeyen"):
            self.assertGreater(estimate_for_tool(arac), 0.0)

    def test_estimates_ordered(self) -> None:
        """Render, testlerden; testler okumadan uzun sürmeli."""
        self.assertGreater(estimate_for_tool("blender.render"),
                           estimate_for_tool("code.run_tests"))
        self.assertGreater(estimate_for_tool("code.run_tests"),
                           estimate_for_tool("code.read_file"))

    def test_planned_step_dict_json_safe(self) -> None:
        adim = PlannedStep(title="x", tool="code.read_file",
                           needs={"ram_mb": 10.0})
        assert_json_safe(self, adim.to_dict(), "planned_step")


# ─── Planlama ───────────────────────────────────────────────────────────────

class TestBridgePlanning(unittest.TestCase):
    """Plan sözlüğünden `WorkStep` üretimi."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_bp_")
        self.snap = ResourceSnapshot(cpu_percent=10, ram_percent=10,
                                     vram_percent=10, ram_total_mb=32768,
                                     vram_total_mb=11264, cpu_cores=24)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _bridge(self) -> AgentBridge:
        br = AgentBridge(workspace=self.root)
        br.initialize()
        return br

    def test_steps_created(self) -> None:
        br = self._bridge()
        plan = [{"title": "A", "tool": "code.read_file", "args": {"path": "a"}},
                {"title": "B", "tool": "code.create_file", "args": {}}]
        adimlar = br.plan_steps("görev", plan)
        self.assertEqual(len(adimlar), 2)
        self.assertEqual(adimlar[0].payload["tool"], "code.read_file")

    def test_needs_previous_creates_chain(self) -> None:
        """`needs_previous` adımı bir öncekine bağlamalı (sıralı iş)."""
        br = self._bridge()
        plan = [
            {"title": "A", "tool": "code.read_file"},
            {"title": "B", "tool": "code.create_file", "needs_previous": True},
            {"title": "C", "tool": "code.create_file", "needs_previous": True},
        ]
        adimlar = br.plan_steps("zincir", plan)
        self.assertEqual(adimlar[0].depends_on, [])
        self.assertEqual(adimlar[1].depends_on, [adimlar[0].id])
        self.assertEqual(adimlar[2].depends_on, [adimlar[1].id])

    def test_independent_steps_have_no_deps(self) -> None:
        """`needs_previous` yoksa bağımsız kalmalı (paralel çalışsın)."""
        br = self._bridge()
        adimlar = br.plan_steps("paralel", [
            {"title": "A", "tool": "code.create_file"},
            {"title": "B", "tool": "code.create_file"},
            {"title": "C", "tool": "code.create_file"},
        ])
        for adim in adimlar:
            self.assertEqual(adim.depends_on, [])

    def test_explicit_depends_on_honoured(self) -> None:
        br = self._bridge()
        adimlar = br.plan_steps("açık", [
            {"id": "birinci", "title": "A", "tool": "code.read_file"},
            {"title": "B", "tool": "code.create_file",
             "depends_on": ["birinci"]},
        ])
        self.assertEqual(adimlar[1].depends_on, ["birinci"])

    def test_think_kind_inferred_from_candidates(self) -> None:
        """`candidates` varsa adım düşünme türü olmalı."""
        br = self._bridge()
        adimlar = br.plan_steps("düşün", [
            {"title": "Seç", "prompt": "hangisi?", "candidates": ["a", "b"]},
        ])
        self.assertEqual(adimlar[0].kind, "think")
        self.assertEqual(adimlar[0].payload["candidates"], ["a", "b"])

    def test_tool_step_kind(self) -> None:
        br = self._bridge()
        adimlar = br.plan_steps("araç", [{"title": "A", "tool": "code.read_file"}])
        self.assertEqual(adimlar[0].kind, "tool")

    def test_explicit_needs_override(self) -> None:
        """Plan kaynak ihtiyacı verebilir (tahmini geçersiz kılar)."""
        br = self._bridge()
        adimlar = br.plan_steps("özel", [
            {"title": "A", "tool": "code.read_file",
             "needs": {"ram_mb": 999.0}},
        ])
        self.assertEqual(adimlar[0].needs["ram_mb"], 999.0)

    def test_ids_generated_and_stable(self) -> None:
        br = self._bridge()
        adimlar = br.plan_steps("kimlik", [
            {"title": "A"}, {"title": "B"}, {"id": "ozel", "title": "C"},
        ])
        self.assertEqual(len({a.id for a in adimlar}), 3)
        self.assertIn("ozel", {a.id for a in adimlar})

    def test_engine_planned(self) -> None:
        br = self._bridge()
        br.plan_steps("motor", [{"title": "A", "tool": "code.read_file"}])
        self.assertEqual(len(br.engine.steps), 1)
        self.assertTrue(br.engine.chunks)


# ─── Yürütme ────────────────────────────────────────────────────────────────

class TestBridgeExecution(unittest.TestCase):
    """Araç çağrıları ve düşünme adımları."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_be_")
        self.snap = ResourceSnapshot(cpu_percent=10, ram_percent=10,
                                     vram_percent=10, ram_total_mb=32768,
                                     vram_total_mb=11264, cpu_cores=24)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _bridge(self, registry: Any = None,
                thinking: Any = None) -> AgentBridge:
        br = AgentBridge(workspace=self.root)
        br.initialize(loop=FakeLoop(registry), thinking=thinking)
        return br

    def test_tool_called_with_args(self) -> None:
        """Araç doğru parametrelerle çağrılmalı."""
        kayit = FakeRegistry({"code.read_file": lambda path="": f"içerik:{path}"})
        br = self._bridge(kayit)
        br.plan_steps("çağrı", [
            {"title": "Oku", "tool": "code.read_file", "args": {"path": "x.py"}},
        ])
        rapor = br.run(snapshot=self.snap)
        self.assertEqual(rapor["steps_done"], 1)
        self.assertEqual(kayit.calls[0]["tool"], "code.read_file")
        self.assertEqual(kayit.calls[0]["args"], {"path": "x.py"})

    def test_result_captured_in_step(self) -> None:
        kayit = FakeRegistry({"code.read_file": lambda **_: "SONUC_METNI"})
        br = self._bridge(kayit)
        br.plan_steps("sonuç", [{"title": "Oku", "tool": "code.read_file"}])
        br.run(snapshot=self.snap)
        self.assertIn("SONUC_METNI", br.engine.steps[0].result)

    def test_missing_registry_does_not_crash(self) -> None:
        """Kayıt defteri yoksa adım çökmemeli, bilgi dönmeli."""
        br = self._bridge(None)
        br.plan_steps("kayıtsız", [{"title": "A", "tool": "code.read_file"}])
        rapor = br.run(snapshot=self.snap)
        self.assertEqual(rapor["steps_done"], 1)
        self.assertIn("kayıt", br.engine.steps[0].result.lower())

    def test_unknown_tool_reported(self) -> None:
        """
        Bilinmeyen araç **başarısız adım** olmalı ve gerekçe görünmeli.

        Sessizce başarılı saymak kabul edilemez: plan yanlış araç adı
        içeriyorsa kullanıcı bunu görmeli.
        """
        kayit = FakeRegistry({})
        br = self._bridge(kayit)
        br.plan_steps("bilinmeyen", [{"title": "A", "tool": "yok.arac"}])
        rapor = br.run(snapshot=self.snap)
        self.assertEqual(rapor["steps_failed"], 1)
        self.assertIn("bulunamadı", br.engine.steps[0].error)

    def test_tool_exception_recorded_as_failure(self) -> None:
        """Araç istisnası adımı başarısız yapmalı, motoru çökertmemeli."""
        def patla(**_: Any) -> None:
            raise ValueError("araç bozuk")

        kayit = FakeRegistry({"code.read_file": patla})
        br = self._bridge(kayit)
        br.plan_steps("hata", [{"title": "A", "tool": "code.read_file"}])
        rapor = br.run(snapshot=self.snap)
        self.assertEqual(rapor["steps_failed"], 1)
        self.assertIn("araç bozuk", br.engine.steps[0].error)

    def test_think_step_uses_thinking_loop(self) -> None:
        """Düşünme adımı yerel seçimi kullanmalı."""
        thinking = FakeThinking(chosen="DOGRU_YOL", source="local",
                                confidence="yüksek")
        br = self._bridge(FakeRegistry({}), thinking)
        br.plan_steps("düşün", [
            {"title": "Seç", "prompt": "hangisi?", "candidates": ["a", "b"]},
        ])
        br.run(snapshot=self.snap)
        sonuc = br.engine.steps[0].result
        self.assertIn("DOGRU_YOL", sonuc)
        self.assertIn("local", sonuc)
        self.assertEqual(thinking.calls, 1)

    def test_think_step_without_loop_is_explicit(self) -> None:
        """Düşünme döngüsü yoksa **sessizce** yanlış yapmamalı."""
        br = self._bridge(FakeRegistry({}), None)
        br.plan_steps("düşün", [
            {"title": "Seç", "prompt": "p", "candidates": ["ilk", "ikinci"]},
        ])
        br.run(snapshot=self.snap)
        sonuc = br.engine.steps[0].result
        self.assertIn("düşünme yok", sonuc)
        self.assertIn("ilk", sonuc)

    def test_single_candidate_skips_thinking(self) -> None:
        thinking = FakeThinking()
        br = self._bridge(FakeRegistry({}), thinking)
        br.plan_steps("tek", [{"title": "S", "prompt": "p",
                               "candidates": ["tek_yol"]}])
        br.run(snapshot=self.snap)
        self.assertEqual(thinking.calls, 0)

    def test_thinking_failure_falls_back(self) -> None:
        thinking = FakeThinking(fail=True)
        br = self._bridge(FakeRegistry({}), thinking)
        br.plan_steps("bozuk", [{"title": "S", "prompt": "p",
                                 "candidates": ["yedek", "b"]}])
        br.run(snapshot=self.snap)
        self.assertIn("yedek", br.engine.steps[0].result)

    def test_heavy_step_deferred(self) -> None:
        """Ağır araç küçük sistemde **çalıştırılmamalı**."""
        yogun = ResourceSnapshot(cpu_percent=85, ram_percent=85,
                                 vram_percent=85, ram_total_mb=4096,
                                 vram_total_mb=4096, cpu_cores=4)
        kayit = FakeRegistry({"blender.render": lambda **_: "render bitti"})
        br = self._bridge(kayit)
        br.plan_steps("ağır", [{"title": "Render", "tool": "blender.render"}])
        rapor = br.run(snapshot=yogun)
        self.assertEqual(rapor["steps_done"], 0)
        self.assertEqual(len(kayit.calls), 0, "ağır iş çalıştırılmamalı")
        self.assertGreater(rapor["chunks_blocked"], 0)

    def test_report_json_safe(self) -> None:
        kayit = FakeRegistry({"code.read_file": lambda **_: "ok"})
        br = self._bridge(kayit)
        br.plan_steps("rapor", [{"title": "A", "tool": "code.read_file"}])
        assert_json_safe(self, br.run(snapshot=self.snap), "bridge_report")

    def test_stats_json_safe(self) -> None:
        br = self._bridge(FakeRegistry({}))
        assert_json_safe(self, br.get_stats(), "bridge_stats")


class TestApprovalFlow(unittest.TestCase):
    """
    Onay akışı — onay bekleyen adım **başarısız değil**, bekliyor.

    Ölçülen sorun: Denetimli modda yazma/çalıştırma araçları onay ister ve
    `registry.call()` `needs_approval` döndürür. Bu, "başarısız" sayıldığında
    toplu işlerde **her yazma adımı** çökmüş görünüyordu (ölçüldü: 3 adımlık
    planın 2'si başarısız). Oysa iş **bekliyor**, bozulmamış.
    """

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_apr_")
        self.snap = ResourceSnapshot(cpu_percent=10, ram_percent=10,
                                     vram_percent=10, ram_total_mb=32768,
                                     vram_total_mb=11264, cpu_cores=24)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _bridge(self, kayit: Any) -> AgentBridge:
        br = AgentBridge(workspace=self.root)
        br.initialize(loop=FakeLoop(kayit))
        return br

    def test_needs_approval_detection(self) -> None:
        """Onay istisnası tanınmalı, diğerleri tanınmamalı."""
        self.assertTrue(AgentBridge._needs_approval(
            RuntimeError("Denetimli modunda yazma işlemleri için onay gerekir")))
        self.assertTrue(AgentBridge._needs_approval(
            RuntimeError("approval required")))
        self.assertFalse(AgentBridge._needs_approval(
            RuntimeError("dosya bulunamadı")))
        self.assertFalse(AgentBridge._needs_approval(
            RuntimeError("Güvenilir modda çalıştırma izni yok")))

    def test_pending_step_marked_waiting_not_failed(self) -> None:
        """
        Onay gelmezse adım **`onay bekliyor`** durumunda kalmalı.

        ⚠ Bu, toplu işin bloke olmaması için kritiktir: adımı "başarısız"
        saymak, her yazma adımının çökmüş görünmesine yol açıyordu. İş
        **bekliyor**, bozulmamış.
        """
        from neurocore.core.work import WAITING

        def onay_iste(**_: Any) -> Dict[str, Any]:
            return {"ok": False, "needs_approval": True,
                    "error": "code.create_file: onay gerekir"}

        kayit = FakeRegistry({"code.create_file": onay_iste})
        br = self._bridge(kayit)
        br.approval_wait_s = 0.3
        br.plan_steps("onay", [
            {"title": "Yaz", "tool": "code.create_file", "args": {"path": "a"}},
        ])
        rapor = br.run(snapshot=self.snap, max_wait_s=1.0, poll_s=0.2)

        # Adım "başarısız" DEĞİL, "bekliyor" olmalı
        self.assertEqual(br.engine.steps[0].status, WAITING)
        self.assertEqual(rapor["steps_failed"], 0, "bekleme hata sayılmamalı")
        self.assertIn("t000", br._pending_approvals)
        self.assertIn("onay", br.engine.steps[0].result.lower())
        # Onay listesinde görünmeli
        self.assertEqual(len(br.pending_approvals()), 1)

    def test_approve_runs_the_step(self) -> None:
        """
        Onaylanan adım **yeniden çalıştırılmalı**.

        Bu, toplu işin bloke olmamasını sağlar: kullanıcı onaylar, iş
        kaldığı yerden devam eder.
        """
        cagri: List[Dict[str, Any]] = []

        def arac(**kwargs: Any) -> str:
            cagri.append(kwargs)
            return "yazıldı"

        kayit = FakeRegistry({"code.create_file": arac})
        br = self._bridge(kayit)
        # İlk çağrı onay ister gibi davran
        ilk = {"n": 0}
        asil_cagir = kayit.call

        def onayli_cagir(name: str, args: Any = None, **kw: Any) -> Any:
            ilk["n"] += 1
            if ilk["n"] == 1:
                return {"ok": False, "needs_approval": True,
                        "error": f"{name}: onay gerekir (denetimli)"}
            return asil_cagir(name, args, **kw)

        kayit.call = onayli_cagir      # type: ignore[assignment]
        br.approval_wait_s = 3.0
        br.plan_steps("onay", [
            {"title": "Yaz", "tool": "code.create_file", "args": {"path": "a"}},
        ])

        # Ayrı iş parçacığında onayla
        def onayla() -> None:
            time.sleep(0.5)
            br.approve("t000")

        threading.Thread(target=onayla, daemon=True).start()
        br.run(snapshot=self.snap, max_wait_s=1.0, poll_s=0.2)
        self.assertTrue(cagri, "onay sonrası araç çağrılmalı")
        self.assertIn("onaylandı", br.engine.steps[0].result)
        self.assertNotIn("t000", br._pending_approvals)

    def test_deny_skips_the_step(self) -> None:
        """Reddedilen adım atlanmalı ve iş durmamalı."""
        def onay_iste(**_: Any) -> Dict[str, Any]:
            return {"ok": False, "needs_approval": True, "error": "onay gerekir"}

        kayit = FakeRegistry({"code.create_file": onay_iste})
        br = self._bridge(kayit)
        br.approval_wait_s = 3.0
        br.plan_steps("red", [
            {"title": "Yaz", "tool": "code.create_file", "args": {}},
        ])

        def reddet() -> None:
            time.sleep(0.5)
            br.deny("t000")

        threading.Thread(target=reddet, daemon=True).start()
        br.run(snapshot=self.snap, max_wait_s=1.0, poll_s=0.2)
        self.assertIn("reddetti", br.engine.steps[0].result)
        # Red sonrası onay listesinden çıkmalı
        self.assertEqual(len(br.pending_approvals()), 0)

    def test_pending_approvals_list(self) -> None:
        """Arayüz için onay listesi dönmeli."""
        br = self._bridge(FakeRegistry({}))
        br._pending_approvals["x1"] = {
            "tool": "code.create_file", "args": {}, "reason": "onay",
            "at": time.time(), "title": "Yaz",
        }
        liste = br.pending_approvals()
        self.assertEqual(len(liste), 1)
        self.assertEqual(liste[0]["step_id"], "x1")
        self.assertIn("waited_s", liste[0])
        assert_json_safe(self, liste[0], "pending_approval")

    def test_approve_unknown_step_returns_false(self) -> None:
        br = self._bridge(FakeRegistry({}))
        self.assertFalse(br.approve("yok"))
        self.assertFalse(br.deny("yok"))

    def test_approval_counters(self) -> None:
        def onay_iste(**_: Any) -> Dict[str, Any]:
            return {"ok": False, "needs_approval": True, "error": "onay gerekir"}

        kayit = FakeRegistry({"code.create_file": onay_iste})
        br = self._bridge(kayit)
        br.approval_wait_s = 0.2
        br.plan_steps("sayaç", [
            {"title": "A", "tool": "code.create_file", "args": {}},
        ])
        br.run(snapshot=self.snap, max_wait_s=1.0, poll_s=0.2)
        self.assertGreaterEqual(br._counter.get("approval_waits", 0), 1)
        self.assertGreaterEqual(br._counter.get("steps_waiting", 0), 0)

    def test_normal_failure_still_raises(self) -> None:
        """Onay dışı hata **yine** başarısızlık olmalı."""
        def patla(**_: Any) -> None:
            raise ValueError("gerçek hata")

        kayit = FakeRegistry({"code.read_file": patla})
        br = self._bridge(kayit)
        br.plan_steps("hata", [{"title": "Oku", "tool": "code.read_file"}])
        rapor = br.run(snapshot=self.snap, max_wait_s=1.0, poll_s=0.2)
        self.assertEqual(rapor["steps_failed"], 1)
        self.assertEqual(len(br.pending_approvals()), 0)


# ─── Devam Edebilme ─────────────────────────────────────────────────────────

class TestBridgeResume(unittest.TestCase):
    """Kesintiden sonra kalan işi doğru hesaplama."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="nc_br2_")
        self.snap = ResourceSnapshot(cpu_percent=10, ram_percent=10,
                                     vram_percent=10, ram_total_mb=32768,
                                     vram_total_mb=11264, cpu_cores=24)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _bridge(self, kayit: Any) -> AgentBridge:
        br = AgentBridge(workspace=self.root)
        br.initialize(loop=FakeLoop(kayit))
        return br

    def test_pending_steps_after_run(self) -> None:
        """Çalıştırılamayan adımlar `pending` olarak görünmeli."""
        yogun = ResourceSnapshot(cpu_percent=90, ram_percent=90,
                                 vram_percent=90, ram_total_mb=4096,
                                 vram_total_mb=4096, cpu_cores=4)
        kayit = FakeRegistry({"code.read_file": lambda **_: "ok"})
        br = self._bridge(kayit)
        br.plan_steps("bekle", [
            {"title": "A", "tool": "code.read_file"},
            {"title": "Ağır", "tool": "blender.render"},
        ])
        br.run(snapshot=yogun)
        kalan = br.pending_steps()
        self.assertTrue(kalan)
        self.assertTrue(all(a.status != DONE for a in kalan))

    def test_resume_skips_completed(self) -> None:
        """Tamamlanan adımlar tekrar çalıştırılmamalı."""
        kayit = FakeRegistry({"code.read_file": lambda **_: "ok"})
        br = self._bridge(kayit)
        br.plan_steps("devam", [
            {"title": "A", "tool": "code.read_file"},
            {"title": "B", "tool": "code.read_file"},
        ])
        br.run(snapshot=self.snap)
        ilk_cagri = len(kayit.calls)
        self.assertEqual(ilk_cagri, 2)

        # Aynı köprüde devam: hepsi bitmiş → çalıştırma yok
        rapor = br.resume(snapshot=self.snap)
        self.assertFalse(rapor.get("resumed"))
        self.assertEqual(len(kayit.calls), ilk_cagri)
        self.assertIn("tamamlanmış", rapor.get("message", ""))

    def test_resume_state_reads_journal(self) -> None:
        kayit = FakeRegistry({"code.read_file": lambda **_: "ok"})
        br = self._bridge(kayit)
        br.plan_steps("günlük", [{"title": "A", "tool": "code.read_file"}])
        br.run(snapshot=self.snap)
        durum = br.resume_state()
        self.assertEqual(durum["completed_count"], 1)
        self.assertIn("günlük", durum["objective"])

    def test_resume_runs_remaining(self) -> None:
        """
        Kesintiden sonra **kalan** adımlar çalışmalı.

        Benzetim: yoğun sistemde ağır adım atlanır, sonra boş sistemde
        devam edilir.
        """
        yogun = ResourceSnapshot(cpu_percent=88, ram_percent=88,
                                 vram_percent=88, ram_total_mb=4096,
                                 vram_total_mb=4096, cpu_cores=4)
        kayit = FakeRegistry({
            "code.read_file": lambda **_: "hafif ok",
            "code.create_file": lambda **_: "yazıldı",
        })
        br = self._bridge(kayit)
        br.plan_steps("kısmi", [
            {"title": "Hafif", "tool": "code.read_file"},
            {"title": "Ağır", "tool": "blender.render"},
        ])
        br.run(snapshot=yogun)
        self.assertEqual(br.pending_steps()[0].title, "Ağır")

        rapor = br.resume(snapshot=self.snap)
        self.assertTrue(rapor.get("resumed"))
        self.assertEqual(rapor.get("skipped_steps"), 1)
        # Hafif adım tekrar çağrılmamalı
        self.assertEqual(sum(1 for c in kayit.calls
                             if c["tool"] == "code.read_file"), 1)

    def test_algorithm_after_resume(self) -> None:
        kayit = FakeRegistry({"code.read_file": lambda **_: "ok"})
        br = self._bridge(kayit)
        br.plan_steps("algoritma", [{"title": "A", "tool": "code.read_file"}])
        br.run(snapshot=self.snap)
        self.assertIn("algoritma", br.algorithm())

    def test_shutdown_clears_loop(self) -> None:
        br = self._bridge(FakeRegistry({}))
        br.shutdown()
        self.assertIsNone(br._loop)


if __name__ == "__main__":
    unittest.main(verbosity=2)
