"""
NeuroCore İşletim Sistemi Katmanı (L5)
======================================
Sistem envanteri · yetenek keşfi · dosya/süreç/kabuk eylemleri · sandbox ·
denetim kaydı.

Alt modüller:
    sandbox      — yol ve komut güvenlik politikası (bağımlılıksız)
    audit        — ``70_journal/os_audit.md`` markdown denetim günlüğü
    inventory    — OS/CPU/RAM/disk/GPU/ağ/ortam envanteri
    capabilities — araç ve özellik (torch, cuda, fts5, …) yoklaması
    filesystem   — sandbox denetimli dosya G/Ç, arama, izleme
    process      — süreç listeleme, başlatma, sonlandırma
    shell        — komut yürütme (PowerShell/Python kısayolları)
    layer        — `OSLayer` yüzü (hepsini tek çatı altında toplar)

Kullanım:
    from neurocore.os import OSLayer

    os_layer = OSLayer()
    os_layer.initialize(config, event_bus)
    print(os_layer.health())
    os_layer.shutdown()
"""

from __future__ import annotations

from neurocore.os.audit import AuditEntry, AuditLog
from neurocore.os.capabilities import CapabilityProbe, ToolInfo
from neurocore.os.filesystem import FileSystemError, FileSystemTools
from neurocore.os.inventory import SystemInventory, SystemInventoryReport
from neurocore.os.layer import OSLayer
from neurocore.os.process import ProcessManager
from neurocore.os.sandbox import Sandbox, SandboxPolicy, SandboxViolation
from neurocore.os.shell import CommandResult, ShellRunner

__all__ = [
    "OSLayer",
    "Sandbox",
    "SandboxPolicy",
    "SandboxViolation",
    "AuditLog",
    "AuditEntry",
    "SystemInventory",
    "SystemInventoryReport",
    "CapabilityProbe",
    "ToolInfo",
    "FileSystemTools",
    "FileSystemError",
    "ProcessManager",
    "ShellRunner",
    "CommandResult",
]
