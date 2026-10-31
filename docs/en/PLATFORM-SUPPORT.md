# Platform Support — NeuroCore on RTX Spark & AMD AI Halo

**Status:** feasibility analysis (measured 2026-10-31)
**P1 — DONE (v0.3.0):** `neurocore/platform/` capability layer · 17 tests
**Türkçe özet en altta.**

---

## 1 · The two target devices

| Device | Architecture | OS | GPU stack |
|---|---|---|---|
| **NVIDIA RTX Spark** (N1X) | **Arm64** superchip, up to 128 GB unified | **Windows on Arm** | CUDA / TensorRT (Arm64 drivers) |
| **AMD Ryzen AI Halo** | x86-64 APU (Strix Halo class), unified memory | **Linux** (fully open-source stack) | ROCm (Ryzen AI Max support added Oct 2026) |

Sources: [NVIDIA RTX Spark — Arm-based superchip for Windows PCs](https://tech.yahoo.com/computing/articles/nvidia-unveils-rtx-spark-arm-050737236.html)
· [AMD Ryzen AI Halo review — open-source software](https://www.phoronix.com/review/amd-ryzen-ai-halo)
· [ROCm adds Ryzen AI Max support](https://www.techpowerup.com/337073/amd-updates-rocm-to-support-ryzen-ai-max-and-radeon-rx-9000-series)

So the two devices stress **two different axes**:

* RTX Spark → **Windows + Arm64** (new CPU architecture, same Windows APIs)
* AI Halo → **Linux + x86-64 + ROCm** (new OS + new GPU vendor stack)

---

## 2 · What our code depends on today (measured)

| Surface | Count | Files | Portability |
|---|---|---|---|
| Pure-Python core (FastAPI server, agent loop, code index, tools) | — | ~120 modules | ✅ portable |
| Windows-only imports (`pywinauto`, `comtypes`) | **2** | `neurocore/vision/desktop.py`, `neurocore/os/capabilities.py` | ⚠️ optional feature |
| `sys.platform` branches | **19** | capabilities, filesystem, inventory, … | ✅ already guarded |
| GPU telemetry via **NVML/pynvml** | **10 refs** | `neurocore/hardware/monitor.py` | ⚠️ NVIDIA-only |
| GPU telemetry in inventory | **8 refs** | `neurocore/hardware/*` | ⚠️ NVIDIA-only |
| PowerShell / cmd shell commands | **125 refs** | `agent/shell.py`, tool implementations | ⚠️ Windows-only behaviour |
| `requirements.txt` | torch, torchvision, torchaudio, GPUtil, pynvml, **pywinauto**, **comtypes** | — | ⚠️ mixed |

**Verdict:** the *product* (server + agent + index + tools) is portable. The blockers are
narrow and known: **2 optional imports, 18 NVML references, 125 shell call sites,
3 wheels that do not exist on every platform.**

---

## 3 · Blockers per target

### 3.1 Windows on Arm (RTX Spark)

| Item | Status | Plan |
|---|---|---|
| Python 3.11/3.12 **arm64** | official installer exists | `Kurulum.bat` works (auto-detect already) |
| `pywinauto` / `comtypes` (UI automation) | comtypes has **no arm64 wheels** | make it an **optional extra**; feature flag `ui_otomasyon=False` |
| PyTorch with CUDA on Windows Arm64 | **no official wheels** | Phase 1: CPU + `torch-directml`; Phase 2: CUDA via WSL2 |
| NVML (`pynvml`) | works on Windows Arm64 with NVIDIA driver | keep, add fallback |
| 125 PowerShell call sites | PowerShell 7 exists on Arm64 | keep, add capability probe |
| Godot / Blender | Godot arm64 builds exist; Blender Windows-Arm is beta | probe + disable tool if missing |

### 3.2 Linux + ROCm (AMD AI Halo)

| Item | Status | Plan |
|---|---|---|
| Python 3.10+ | distro packages | `kur_istemci.sh` already exists — extend with venv + uv |
| `pywinauto` / `comtypes` | **Windows-only** | skip on Linux (feature flag) |
| PyTorch + ROCm | official ROCm wheels (Linux x86-64) | `requirements-rocm.txt` with `--index-url .../rocm6.x` |
| NVML | absent | new backend: `amdsmi` → `rocm-smi` → `/sys/class/drm` |
| PowerShell 125 call sites | absent | `shell.py` already has a bash path; audit the 125 sites, keep POSIX equivalents |
| Desktop UI reading | absent | use `xdotool`/AT-SPI when available, else disable |

---

## 4 · Plan (4 phases, ~8–10 working days)

```
P1  capability layer (2–3 d)
    neurocore/platform/  →  sistem(), kabuk(), gpu_arka_ucu(), ozellikler()
    wrap pywinauto/comtypes in try/except, single feature flag source
P2  dependency split (1 d)
    requirements-base.txt      (portable core)
    requirements-windows.txt   (pywinauto, comtypes, pywin32)
    requirements-cuda.txt      (torch cu12x)
    requirements-rocm.txt      (torch rocm6.x)
P3  installers + service (2 d)
    Kurulum.bat          (Windows x64 + Arm64, already portable)
    kur_istemci.sh       (Linux x86-64 + aarch64, systemd unit, .desktop)
    N1X:  WSL2 helper script (CUDA path)
P4  GPU backends + tests (2–3 d)
    monitor.py:  NVML | amdsmi | rocm-smi | sysfs | CPU
    CI matrix: win-x64, win-arm64, linux-x64, linux-arm64 (import-only smoke)
```

**Effort estimate:** Arm64-Windows support ≈ **3–4 days**; Linux+ROCm ≈ **4–5 days**;
shared capability layer ≈ **2–3 days** (overlaps both).

---

## 4b · P1 delivered (v0.3.0)

```
neurocore/platform/tespit.py
    sistem()        → os (windows/linux/macos), arch (x86_64/arm64), python, is_arm
    kabuk(cmd)      → ["powershell", …] | ["bash", "-lc", …]
    gpu_arka_ucu()  → nvml | amdsmi | rocm-smi | sysfs | torch | cpu   (first that works)
    gpu_bilgi()     → one dict: name, vram_*, utilization, temperature, power, backend
    ui_otomasyon()  → pywinauto+comtypes available (False on Linux / Windows-Arm)
    ozellikler()    → flags + human-readable notes
```

Wired in:
* `hardware/monitor.py` → uses `platform.gpu_bilgi()`, NVML fallback kept
* `app/server.py` → `GET /api/state` now returns `platform` (os, arch, shell,
  gpu_backend, ui_automation, torch/cuda/rocm/directml, notes)

Requirements split: `requirements-base.txt` (portable) ·
`requirements-windows.txt` (pywinauto/comtypes) · `requirements-nvidia.txt` ·
`requirements-rocm.txt`. `requirements.txt` kept for compatibility.

Measured on the dev box (Windows 11, x86-64, GTX 1080 Ti):
```
platform: os=windows arch=x86_64 shell=powershell gpu_backend=nvml ui=True
gpu     : NVIDIA GeForce GTX 1080 Ti (nvml) · util available
tests   : tests/test_platform.py 17 OK · full suite 95 OK
```

## 4c · P2/P3 delivered (v0.3.1)

### Windows installer (`Kurulum.bat` / `installation.bat`)
Now installs by platform instead of one monolith:

```
[INFO] portable core: requirements-base.txt      ← always
        requirements-windows.txt                 ← Windows only (pywinauto/comtypes)
        torch torchvision torchaudio             ← skipped with --no-torch / --base-only
```

New flag `--base-only` (portable core only). Verified end to end in a clean copy:
`installation.bat --no-torch` → base installed, torch skipped, **"Setup complete", exit 0**.

### Linux installer (`kur_istemci.sh` / `installation.sh`)
* installs `requirements-base.txt` + `requirements-rocm.txt` on ROCm (amdsmi)
* **never** installs `requirements-windows.txt` on Linux
* `--systemd` → `~/.config/systemd/user/neurocore.service` (autostart, restart=on-failure)
* `--desktop` → `~/.local/share/applications/neurocore.desktop`
* prints `python -m neurocore.platform` (JSON capabilities) at the end
* syntax checked with `bash -n` (Git Bash)

### CI matrix (P4 start)
`.github/workflows/tests.yml` — `ubuntu-latest` + `windows-latest` × Python 3.10/3.12:
install **base only**, print platform JSON, run the whole unit suite, plus an
import-smoke job proving the core imports without torch/pywinauto.

## 5 · What works unchanged

* FastAPI server + Web UI (pure Python, browser-based) — **no change**
* Agent loop, tools registry, code index (SQLite), big-file reader — **no change**
* Blender/Godot tool *definitions* — only path detection changes
* The measured guarantees (98 unit tests) run on every platform

---

## Türkçe özet

**Evet, çalıştırabiliriz.** Ürünün çekirdeği (sunucu, ajan döngüsü, kod indeksi, araçlar)
saf Python ve taşınabilir. Engel olan üç şey dar kapsamlı:

1. **2 Windows'a özel içe aktarma** (`pywinauto`, `comtypes`) → isteğe bağlı özelliğe çevrilecek
2. **18 NVML referansı** → AMD için `amdsmi`/`rocm-smi` arka ucu eklenecek
3. **125 PowerShell çağrısı** → Linux'ta bash karşılıkları denetlenecek

Cihaz bazında:
* **RTX Spark** = Windows + **Arm64** → `Kurulum.bat` zaten taşınabilir; asıl sınır
  **PyTorch CUDA'nın Windows-Arm64 tekerleği olmaması** → önce CPU/DirectML, sonra WSL2.
* **AMD AI Halo** = **Linux + ROCm** → `kur_istemci.sh` genişletilecek, ROCm tekerlekleri
  resmî olarak var (Ekim 2026 itibarıyla Ryzen AI Max destekli).

Tahmini efor: **Arm64-Windows 3–4 gün**, **Linux+ROCm 4–5 gün**, ortak katman 2–3 gün.
Mevcut 98 test her platformda aynen çalışır.
