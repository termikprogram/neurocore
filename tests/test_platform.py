"""
Platform yetenek katmanı testleri.

Amaç: RTX Spark (Windows/Arm64) ve AMD AI Halo (Linux/ROCm) senaryolarını
gerçek donanım olmadan doğrulamak — sahte modüllerle arka uç seçimi ve
platform davranışı test edilir.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from neurocore.platform import tespit
from neurocore.platform.tespit import (
    SistemBilgisi,
    _mimari_norm,
    gpu_bilgi,
    kabuk,
    ozellikler,
    sistem,
)


class TestMimari(unittest.TestCase):
    """Mimari normalizasyonu (Arm64 tanıma kritik)."""

    def test_x86_varyantlari(self) -> None:
        for ham in ("AMD64", "x86_64", "x64", "i386", "X86"):
            self.assertEqual(_mimari_norm(ham), "x86_64", ham)

    def test_arm_varyantlari(self) -> None:
        for ham in ("arm64", "aarch64", "ARMv8", "ARM64"):
            self.assertEqual(_mimari_norm(ham), "arm64", ham)

    def test_bilinmeyen(self) -> None:
        self.assertEqual(_mimari_norm("riscv64"), "riscv64")
        self.assertEqual(_mimari_norm(""), "unknown")


class TestSistem(unittest.TestCase):
    """sistem() gerçek ortamı doğru sınıflandırıyor mu."""

    def test_gercek_ortam(self) -> None:
        s = sistem()
        self.assertIn(s.os, ("windows", "linux", "macos", "other"))
        self.assertIn(s.arch, ("x86_64", "arm64", "unknown", "riscv64"))
        self.assertGreaterEqual(len(s.python.split(".")), 3)
        self.assertTrue(s.is_64bit)

    def test_sozluk(self) -> None:
        d = sistem().to_dict()
        for anahtar in ("os", "arch", "python", "is_64bit", "is_arm"):
            self.assertIn(anahtar, d)


class TestKabuk(unittest.TestCase):
    """kabuk() platforma göre doğru argv üretiyor mu."""

    def test_windows_powershell(self) -> None:
        sahte = SistemBilgisi(os="windows", arch="arm64", python="3.12.0")
        with mock.patch.object(tespit, "sistem", return_value=sahte):
            argv = kabuk("echo merhaba")
        self.assertEqual(argv[0], "powershell")
        self.assertIn("-Command", argv)
        self.assertEqual(argv[-1], "echo merhaba")

    def test_linux_bash(self) -> None:
        sahte = SistemBilgisi(os="linux", arch="x86_64", python="3.12.0")
        with mock.patch.object(tespit, "sistem", return_value=sahte):
            argv = kabuk("echo merhaba")
        self.assertEqual(argv[:3], ["bash", "-lc", "echo merhaba"])


class TestGpuArkaUc(unittest.TestCase):
    """Arka uç seçim sırası: nvml → amdsmi → rocm-smi → sysfs → torch → cpu."""

    def setUp(self) -> None:
        tespit.gpu_arka_ucu.cache_clear()
        tespit._GPU_ONBELLEK[0] = 0.0
        tespit._GPU_ONBELLEK[1] = None

    def tearDown(self) -> None:
        tespit.gpu_arka_ucu.cache_clear()

    def _sahte_uc(self, ad: str, veri):
        def _f():
            return veri
        return (ad, _f)

    def test_nvml_once(self) -> None:
        veri = {"name": "RTX", "available": True, "backend": "nvml"}
        ozel = [self._sahte_uc("nvml", veri), self._sahte_uc("amdsmi", veri)]
        with mock.patch.object(tespit, "_ARKA_UCLAR", ozel):
            self.assertEqual(tespit.gpu_arka_ucu(), "nvml")
            self.assertEqual(gpu_bilgi(yenile=True)["backend"], "nvml")

    def test_amdsmi_ikinci_sira(self) -> None:
        veri = {"name": "Radeon", "available": True, "backend": "amdsmi"}
        ozel = [self._sahte_uc("nvml", None), self._sahte_uc("amdsmi", veri)]
        with mock.patch.object(tespit, "_ARKA_UCLAR", ozel):
            self.assertEqual(tespit.gpu_arka_ucu(), "amdsmi")

    def test_gpu_yoksa_cpu(self) -> None:
        ozel = [self._sahte_uc("nvml", None), self._sahte_uc("amdsmi", None)]
        with mock.patch.object(tespit, "_ARKA_UCLAR", ozel):
            self.assertEqual(tespit.gpu_arka_ucu(), "cpu")
            self.assertIsNone(gpu_bilgi(yenile=True))

    def test_istisna_yutulur(self) -> None:
        def patlat():
            raise RuntimeError("sürücü yok")

        with mock.patch.object(tespit, "_ARKA_UCLAR", [("bozuk", patlat)]):
            self.assertEqual(tespit.gpu_arka_ucu(), "cpu")

    def test_bilgi_anahtarlari(self) -> None:
        veri = {
            "name": "X", "vram_total_mb": 1024, "vram_used_mb": 256,
            "vram_free_mb": 768, "vram_percent": 25.0, "gpu_utilization": 10,
            "memory_utilization": 5, "temperature_c": 50, "power_w": 100,
            "available": True, "backend": "nvml",
        }
        with mock.patch.object(tespit, "_ARKA_UCLAR", [self._sahte_uc("nvml", veri)]):
            bilgi = gpu_bilgi(yenile=True)
        for anahtar in ("name", "vram_total_mb", "vram_used_mb", "vram_percent",
                        "gpu_utilization", "temperature_c", "power_w", "backend"):
            self.assertIn(anahtar, bilgi)


class TestOzellikler(unittest.TestCase):
    """ozellikler() beklenen bayrakları üretiyor mu."""

    def test_bayraklar(self) -> None:
        o = ozellikler()
        for anahtar in ("os", "arch", "shell", "ui_automation", "gpu_backend",
                        "torch", "cuda", "rocm", "notes"):
            self.assertIn(anahtar, o)
        self.assertIn(o["shell"], ("powershell", "bash"))
        self.assertIsInstance(o["notes"], list)

    def test_linux_notu(self) -> None:
        sahte = SistemBilgisi(os="linux", arch="x86_64", python="3.12.0")
        with mock.patch.object(tespit, "sistem", return_value=sahte), \
             mock.patch.object(tespit, "gpu_arka_ucu", return_value="cpu"), \
             mock.patch.object(tespit, "ui_otomasyon", return_value=False):
            notlar = tespit._notlar(sahte, "cpu")
        self.assertTrue(any("amdsmi" in n or "rocm" in n for n in notlar))

    def test_windows_arm_notu(self) -> None:
        sahte = SistemBilgisi(os="windows", arch="arm64", python="3.12.0")
        notlar = tespit._notlar(sahte, "cpu")
        self.assertTrue(any("Arm" in n for n in notlar))

    def test_ui_otomasyon_linux_false(self) -> None:
        sahte = SistemBilgisi(os="linux", arch="x86_64", python="3.12.0")
        tespit.ui_otomasyon.cache_clear()
        with mock.patch.object(tespit, "sistem", return_value=sahte):
            self.assertFalse(tespit.ui_otomasyon())
        tespit.ui_otomasyon.cache_clear()

    def test_ui_otomasyon_windows_bayraksiz(self) -> None:
        """Windows'ta pywinauto yoksa False dönmeli (istisna değil)."""
        sahte = SistemBilgisi(os="windows", arch="arm64", python="3.12.0")
        tespit.ui_otomasyon.cache_clear()
        with mock.patch.object(tespit, "sistem", return_value=sahte):
            with mock.patch.dict(sys.modules, {"pywinauto": None}):
                self.assertFalse(tespit.ui_otomasyon())
        tespit.ui_otomasyon.cache_clear()


if __name__ == "__main__":
    unittest.main()
