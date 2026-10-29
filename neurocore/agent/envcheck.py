"""
Ortam ön denetimi — plan sonrası "ne kurulu, ne eksik?"
=======================================================
Kullanıcı sorusu: "Plan yaptıktan sonra işletim sistemine göre kullanacağı
araçları (yüklü mü?), gerekli paketleri/kütüphaneleri kontrol ediyor mu?
Eksikse kurulması gerektiğini söylüyor mu? (python kütüphanesi, npm, godot,
visual studio …)"

Bu modül, çalışma **planı kurulduktan hemen sonra** projeyi tarar:

```text
işletim sistemi  : windows / linux / macos + paket yöneticisi (winget/apt/brew)
araçlar          : python, pip, node, npm, godot (+ export şablonları),
                   blender, dotnet, cargo, git, uv
proje bağımlılığı: package.json · requirements.txt · pyproject.toml ·
                   project.godot · *.csproj · Cargo.toml
sonuç            : kurulu / eksik / sürüm uyuşmazlığı + OS'e uygun kurulum komutu
```

Tasarım kuralları
-----------------
```
· **Yalnız ölçer**, kendiliğinden kurmaz (kurulum kullanıcı onayı ister)
· Eksik yoksa tek satır: "ortam hazır" (modele gürültü yapmaz)
· Her eksik için **tek** kurulum komutu (OS'e uygun)
· 8 sn üst sınır: preflight görevi yavaşlatmaz
```
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.envcheck")

#: Tek araç sürüm sorgusu için üst sınır (sn)
SURUM_SURESI = 8.0

#: Bilinen araçların sürüm komutları
ARACLAR: Dict[str, Dict[str, Any]] = {
    "python": {"komut": "python", "surum": ["--version"],
               "winget": "Python.Python.3.12", "apt": "python3",
               "brew": "python3"},
    "pip": {"komut": "pip", "surum": ["--version"],
            "winget": None, "apt": "python3-pip", "brew": None},
    "node": {"komut": "node", "surum": ["--version"],
             "winget": "OpenJS.NodeJS.LTS", "apt": "nodejs", "brew": "node"},
    "npm": {"komut": "npm", "surum": ["--version"],
            "winget": "OpenJS.NodeJS.LTS", "apt": "npm", "brew": "node"},
    "godot": {"komut": "godot", "surum": ["--version"],
              "winget": "GodotEngine.Godot", "apt": "godot3", "brew": "godot"},
    "blender": {"komut": "blender", "surum": ["--version"],
                "winget": "BlenderFoundation.Blender", "apt": "blender",
                "brew": "blender"},
    "dotnet": {"komut": "dotnet", "surum": ["--version"],
               "winget": "Microsoft.DotNet.SDK.8", "apt": "dotnet-sdk-8.0",
               "brew": "dotnet"},
    "cargo": {"komut": "cargo", "surum": ["--version"],
              "winget": "Rustlang.Rustup", "apt": "cargo", "brew": "rust"},
    "git": {"komut": "git", "surum": ["--version"],
            "winget": "Git.Git", "apt": "git", "brew": "git"},
    "uv": {"komut": "uv", "surum": ["--version"],
           "winget": "astral-sh.uv", "apt": None,
           "brew": "uv"},
}

#: PATH'te olmayan ama yaygın kurulum yolları (ölçüldü: paketli araçlar)
GODOT_YOLLARI = (
    r"E:\AppsTools\Godot_v4.8-dev7_win64\Godot_v4.8-dev7_win64_console.exe",
    r"E:\AppsTools\Godot_v4.8-dev7_win64\Godot_v4.8-dev7_win64.exe",
    r"C:\Program Files\Godot\godot.exe",
)

BILINEN_YOLLAR: Dict[str, tuple] = {
    "node": (
        r"E:\AppsTools\DeepSeek Harness\resources\runtime\primary-runtime"
        r"\dependencies\node\bin\node.exe",
        r"C:\Program Files\nodejs\node.exe",
    ),
    "npm": (
        r"E:\AppsTools\DeepSeek Harness\resources\runtime\primary-runtime"
        r"\dependencies\node\bin\npm.cmd",
        r"C:\Program Files\nodejs\npm.cmd",
    ),
    "blender": (r"E:\AppsTools\Blender 5.2\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender\blender.exe"),
}

#: Windows'ta yaygın kod editörü / derleyici
EXTRA_WINDOWS = ("code", "msbuild", "cl")


def sistem_bilgisi() -> Dict[str, Any]:
    """İşletim sistemi + kullanılabilir paket yöneticileri."""
    ad = platform.system().lower()
    os_ad = ("windows" if ad.startswith("win")
             else "macos" if ad == "darwin" else "linux")
    paketler: List[str] = []
    if os_ad == "windows":
        for y in ("winget", "choco", "scoop"):
            if shutil.which(y):
                paketler.append(y)
    elif os_ad == "macos":
        if shutil.which("brew"):
            paketler.append("brew")
    else:
        for y in ("apt", "dnf", "pacman", "apk"):
            if shutil.which(y):
                paketler.append(y)
    return {"os": os_ad, "surum": platform.version()[:60],
            "paket_yoneticileri": paketler,
            "python": platform.python_version()}


def _surum(komut: str, args: List[str]) -> str:
    """Araç sürümünü okur (hata → boş)."""
    try:
        cikti = subprocess.run([komut] + list(args), capture_output=True,
                               text=True, timeout=SURUM_SURESI,
                               encoding="utf-8", errors="replace")
        metin = (cikti.stdout or cikti.stderr or "").strip().splitlines()
        return metin[0][:80] if metin else "var"
    except Exception:
        return ""


def arac_durumu(ad: str) -> Dict[str, Any]:
    """Bir aracın kurulu olup olmadığını ve sürümünü döndürür."""
    bilgi = ARACLAR.get(ad, {})
    komut = bilgi.get("komut", ad)
    yol = shutil.which(komut)
    if not yol and ad == "godot":
        for aday in GODOT_YOLLARI:
            if Path(aday).exists():
                yol = aday
                break
    if not yol:
        for aday in BILINEN_YOLLAR.get(ad, ()):
            if Path(aday).exists():
                yol = aday
                break
    if not yol:
        return {"ad": ad, "kurulu": False, "surum": "", "yol": ""}
    return {"ad": ad, "kurulu": True,
            "surum": _surum(yol, bilgi.get("surum", ["--version"])),
            "yol": yol, "path_disi": shutil.which(komut) is None}


def _kurulum_komutu(ad: str, sistem: Dict[str, Any]) -> str:
    """OS'e uygun **tek** kurulum komutu."""
    bilgi = ARACLAR.get(ad, {})
    pm = (sistem.get("paket_yoneticileri") or [""])[0]
    if not pm:
        return ""
    paket = bilgi.get(pm)
    if not paket:
        return ""
    if pm == "winget":
        return f"winget install --id {paket} -e"
    if pm == "choco":
        return f"choco install {paket} -y"
    if pm == "scoop":
        return f"scoop install {paket}"
    if pm == "brew":
        return f"brew install {paket}"
    if pm in ("apt", "dnf", "pacman", "apk"):
        bayrak = "-y" if pm in ("apt", "dnf") else "-S --noconfirm"
        return f"sudo {pm} install {paket} {bayrak}".strip()
    return ""


def _godot_sablonlari() -> Dict[str, Any]:
    """Godot export şablonları (web) kurulu mu?"""
    try:
        adaylar = [
            Path(os.environ.get("APPDATA", "")) / "Godot" / "export_templates",
            Path.home() / ".local" / "share" / "godot" / "export_templates",
            Path.home() / "Library" / "Application Support" / "Godot"
            / "export_templates",
        ]
        for kok in adaylar:
            if kok.is_dir():
                for surum in sorted(kok.iterdir(), reverse=True):
                    if (surum / "web_release.zip").exists():
                        return {"kurulu": True, "surum": surum.name}
    except Exception:
        pass
    return {"kurulu": False, "surum": ""}


def _python_paketleri(kok: Path) -> List[str]:
    """requirements.txt / pyproject.toml içindeki paket adları."""
    adlar: List[str] = []
    req = kok / "requirements.txt"
    if req.exists():
        try:
            for satir in req.read_text(encoding="utf-8",
                                       errors="ignore").splitlines():
                satir = satir.strip()
                if not satir or satir.startswith("#"):
                    continue
                ad = (satir.split("==")[0].split(">=")[0].split("~=")[0]
                      .split("[")[0].strip())
                if ad:
                    adlar.append(ad)
        except OSError:
            pass
    pyproj = kok / "pyproject.toml"
    if pyproj.exists() and not adlar:
        try:
            import re as _re

            metin = pyproj.read_text(encoding="utf-8", errors="ignore")
            blok = _re.search(r"dependencies\s*=\s*\[(.*?)\]", metin, _re.S)
            if blok:
                for parca in blok.group(1).split(","):
                    ad = parca.strip().strip('"').strip("'").split(">=")[0]
                    ad = ad.split("==")[0].split("[")[0].strip()
                    if ad:
                        adlar.append(ad)
        except OSError:
            pass
    return adlar[:20]


def _node_paketleri(kok: Path) -> List[str]:
    paket = kok / "package.json"
    if not paket.exists():
        return []
    try:
        veri = json.loads(paket.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    adlar = list((veri.get("dependencies") or {}))
    adlar += list((veri.get("devDependencies") or {}))
    return adlar[:25]


def denetle(kok: Path, gorev: str = "") -> Dict[str, Any]:
    """
    Projeyi ve görevi tarar; eksik araç/paketleri **OS'e uygun** komutla döndürür.

    Args:
        kok: Proje kökü (yoksa yalnız görev metninden çıkarım yapılır)
        gorev: Görev metni (ör. "godot ile oyun" → godot aranır)

    Returns:
        {"sistem", "araclar", "eksik": [{ad, tur, oneri, komut}],
         "uyarılar": [...], "hazir": bool}
    """
    kok = Path(kok)
    sistem = sistem_bilgisi()
    metin = f"{gorev} {kok.name}".lower()

    # 1) Hangi araçlar gerekli?
    gerekli: List[str] = []
    if (kok / "package.json").exists() or "node" in metin or "npm" in metin:
        gerekli += ["node", "npm"]
    if ((kok / "requirements.txt").exists() or (kok / "pyproject.toml").exists()
            or (kok / "main.py").exists() or "python" in metin):
        gerekli += ["python", "pip"]
    if (kok / "project.godot").exists() or "godot" in metin:
        gerekli.append("godot")
    if (kok / "Cargo.toml").exists() or "rust" in metin:
        gerekli.append("cargo")
    if any(kok.glob("*.csproj")) or any(kok.glob("*.sln")) or ".net" in metin:
        gerekli.append("dotnet")
    if "blender" in metin:
        gerekli.append("blender")
    if (kok / ".git").exists():
        gerekli.append("git")
    gerekli = list(dict.fromkeys(gerekli))

    # 2) Durumları
    araclar = [arac_durumu(ad) for ad in gerekli]
    eksik: List[Dict[str, Any]] = []
    uyarilar: List[str] = []
    for durum in araclar:
        if not durum["kurulu"]:
            eksik.append({
                "ad": durum["ad"], "tur": "araç",
                "oneri": f"{durum['ad']} kurulu değil",
                "komut": _kurulum_komutu(durum["ad"], sistem),
            })

    # 3) Godot export şablonları
    if "godot" in gerekli:
        sablon = _godot_sablonlari()
        if not sablon["kurulu"] and any((kok / "project.godot").exists()
                                        for _ in (0,)):
            eksik.append({
                "ad": "godot export şablonları", "tur": "şablon",
                "oneri": "web export şablonu yok (web_release.zip)",
                "komut": ("Godot editör → Editor → Manage Export Templates → "
                          "Download and Install"),
            })

    # 4) Proje bağımlılıkları
    if "node" in gerekli:
        bagimli = _node_paketleri(kok)
        if bagimli and not (kok / "node_modules").exists():
            eksik.append({
                "ad": f"{len(bagimli)} npm paketi", "tur": "paket",
                "oneri": ", ".join(bagimli[:6]) + ("…" if len(bagimli) > 6 else ""),
                "komut": "npm install",
            })
    if "python" in gerekli:
        for ad in _python_paketleri(kok):
            try:
                __import__(ad.replace("-", "_"))
            except Exception:
                eksik.append({
                    "ad": ad, "tur": "python paketi",
                    "oneri": f"{ad} içe aktarılamıyor",
                    "komut": f"{sys.executable} -m pip install {ad}",
                })

    # 5) Sürüm notları (Godot: PATH'te yoksa bilinen yol)
    if "godot" in gerekli:
        gd = next((a for a in araclar if a["ad"] == "godot"), None)
        if gd and gd["kurulu"] and not shutil.which("godot"):
            uyarilar.append("godot PATH'te yok; bilinen kurulum yolu kullanıldı")

    return {
        "sistem": sistem,
        "araclar": araclar,
        "eksik": eksik,
        "uyarilar": uyarilar,
        "hazir": not eksik,
        "kok": str(kok),
    }


def metin(rapor: Dict[str, Any], sinir: int = 900) -> str:
    """Modele verilecek kısa ön denetim metni."""
    sistem = rapor.get("sistem") or {}
    satirlar = [
        "ORTAM ÖN DENETİMİ",
        f"  işletim sistemi: {sistem.get('os')} · "
        f"paket yöneticisi: {', '.join(sistem.get('paket_yoneticileri') or []) or 'yok'}",
    ]
    kurulu = [f"{a['ad']} {a['surum'][:24]}" for a in rapor.get("araclar", [])
              if a.get("kurulu")]
    if kurulu:
        satirlar.append("  kurulu: " + " · ".join(kurulu))
    eksik = rapor.get("eksik") or []
    if not eksik:
        satirlar.append("  DURUM: hazır (eksik yok)")
    else:
        satirlar.append(f"  EKSİK ({len(eksik)}):")
        for e in eksik[:8]:
            satirlar.append(f"   - {e['ad']} ({e['tur']}): {e['oneri']}")
            if e.get("komut"):
                satirlar.append(f"     kurulum: {e['komut']}")
        satirlar.append(
            "  Kurulum GEREKİYORSA kullanıcıya sor; kendiliğinden kurma.")
    for u in (rapor.get("uyarilar") or [])[:3]:
        satirlar.append(f"  not: {u}")
    return "\n".join(satirlar)[:sinir]
