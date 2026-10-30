# Installation

**Version 0.2.1** · Windows 10/11 · Linux/macOS (beta) · Python 3.10+ · PyTorch

---

## 1 · Requirements

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| Windows | 10 (64-bit) | 11 | Linux/macOS supported by `kur_istemci.sh` |
| Python | 3.10 | **3.10.11** | 64-bit, "Add to PATH" |
| RAM | 8 GB | 16 GB+ | |
| Disk | 10 GB | 30 GB | checkpoints are large |
| GPU | — | NVIDIA 8 GB+ | runs without a GPU too |
| NVIDIA driver | — | ≥ 550 | for training/inference |

## 1b · Portable Python detection (no machine-specific paths)

`Kurulum.bat`, `Run*.bat` and `kur_istemci.ps1` **do not hardcode** a Python
path. They share one finder — `tools\py_bul.bat` — which tries, in order:

```
1. %NC_PYTHON%                      explicit override (full path to python.exe)
2. .venv\Scripts\python.exe        project virtual environment
3. py -3.13 … py -3.10, py -3       official Windows launcher
4. python / python3 on PATH
5. %LOCALAPPDATA%\Programs\Python\PythonXY\python.exe
6. %ProgramFiles%\PythonXY\python.exe  and  C:\PythonXY\python.exe
7. HKCU/HKLM\Software\Python\PythonCore registry (ExecutablePath)
```

Every candidate must report **3.10 or newer**; otherwise it is skipped with a warning.

```powershell
# different machine, Python in a custom folder:
set NC_PYTHON=C:\tools\Python312\python.exe
Kurulum.bat

# skip the virtualenv (install into the system interpreter):
Kurulum.bat --no-venv

# see the detection order:
Kurulum.bat --help
```

If nothing is found the installer prints the exact `winget` command:

```powershell
winget install --id Python.Python.3.12 -e
```

**Measured:** finder resolves `.venv\Scripts\python.exe` (3.10.11) in < 1 s;
`NC_PYTHON` override honoured first.

---

### GPU profiles (measured on this repository)

```
Pascal (GTX 1080 Ti, 11 GB)  : fp16 works · --amp-yok may be required · bf16 is emulated
Ampere (RTX 3060 Ti,  8 GB)  : native bf16 · fp16 AMP stable · --batch 8
Ada / Hopper (4090, A100, H100): native bf16, no GradScaler needed
AMD (ROCm, Linux)            : works through torch.cuda.* · dtype bf16 on gfx90a+
Apple Silicon / Intel        : CPU fallback (no MPS/XPU backend)
```

> The local model is 211M parameters. A 1.5B model does **not** fit in 8 GB VRAM.

---

## 2 · Quick install (one command)

### Windows

```powershell
git clone https://github.com/termikprogram/neurocore.git
cd neurocore
Set-ExecutionPolicy -Scope Process Bypass -Force
.\kur_istemci.ps1
```

### Linux / macOS

```bash
git clone https://github.com/termikprogram/neurocore.git
cd neurocore
bash kur_istemci.sh
```

### GPU detection (both scripts)

The installer **picks the PyTorch build from your hardware** — nothing is
hard-coded to a single CUDA version:

| Detected | Wheel |
|---|---|
| NVIDIA driver ≥ 550 | `cu124` |
| NVIDIA driver ≥ 525 | `cu121` |
| NVIDIA driver < 525 | `cu118` |
| AMD ROCm (`rocminfo` or `/opt/rocm`) | `rocm6.2` |
| AMD on Windows / Intel / no GPU | `cpu` |
| macOS | `cpu` |

Preview the decision without installing anything:

```powershell
.\kur_istemci.ps1 -SadeceGpu      # Windows
bash kur_istemci.sh --sadece-gpu  # Linux/macOS
```

Measured output:

```text
── 2b) GPU ALGILAMA ve PyTorch SEÇİMİ
  GPU        : NVIDIA GeForce GTX 1080 Ti
  sürücü     : 582.78
  PyTorch    : cu124
  gerekçe    : NVIDIA sürücü 582 → cu124
```

### What the installer does

```text
── 1b) SSH KEY   find a working key (NC_SSH_KEY → ~/.ssh/nc_<machine> → id_ed25519)
                 generate ed25519 if missing, restrict permissions
                 test against the server; if unauthorized print the exact
                 one-line command to authorize the PUBLIC key
                 persist NC_SSH_KEY (user environment)
── 2b) GPU       detect hardware, choose the PyTorch wheel (table above)
── 1)  PYTHON    require ≥ 3.10
── 2)  VENV      .venv
── 3)  TORCH     install the chosen build
── 4)  DEPS      requirements.txt
── 5)  CHECK     tools/kurulum_kontrol.py → READY / missing list
```

### Options

| Parameter (PowerShell) | Bash equivalent | Effect |
|---|---|---|
| `-SadeceAnahtar` | — | key step only |
| `-SadeceGpu` | `--sadece-gpu` | detection only |
| `-SadeceKontrol` | — | verification only |
| `-YetkiBekle` | — | show the authorize command, wait for Enter, retry |
| `-Torch cu121` | `--torch rocm6.2` | force a build |
| `-AnahtarDizin`, `-AnahtarAdi` | — | key location/name |
| `-AnahtarYenile` | — | generate a new key |
| `-VenvYok` | — | do not use a virtualenv |
| `-Sunucu user@host` | — | pool server (else `NC_SERVER` / `peer.json`) |

---

## 3 · Manual install

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
# NVIDIA (choose the wheel for your driver)
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# or CPU only
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tools\kurulum_kontrol.py
```

AMD on Linux:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

---

## 4 · Verify

```powershell
.venv\Scripts\python.exe tools\kurulum_kontrol.py            # 34–36 checks
.venv\Scripts\python.exe tools\kurulum_kontrol.py --kaydet   # also try pool registration
.venv\Scripts\python.exe tools\kabiliyet_olc.py              # 10 capability checks
```

Check sections:

```text
1) PYTHON      version, architecture
2) REPO        neurocore/, requirements.txt, tools/, tests/
3) PACKAGES    torch, fastapi, uvicorn, psutil, numpy, …
4) GPU         backend (CUDA/ROCm/CPU), name, VRAM, native bf16
5) RAM/DISK    thresholds
6) SSH/POOL    key, connection, IP as seen by the server, cluster token
7) TOOLS       28 core · 16 remotely allowed
8) APP         FastAPI, peer endpoint, __main__
```

Output: `RESULT: n/m · READY` or a list of what is missing (exit code 1).

---

## 5 · Packaging (moving to another machine)

```powershell
.\paketle.ps1
```

Measured: **12.0 MB** zip · 8,656 files · 5 folders.

```text
Included : neurocore · tools · tests · knowledge_base
           requirements.txt · kur_istemci.ps1 · BASLAT-istemci.bat · paketle.ps1
Excluded : models (23.6 GB) · data (6.5 GB) · .neurocore (2.3 GB)
           Varliklar · TestProjects · __pycache__ · .venv
```

> `.neurocore` is excluded **on purpose**: it holds the node identity
> (`node_id`). Copying it makes two machines look like the same node.

---

## 6 · Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `python` not found | PATH | reinstall Python 3.10 with "Add to PATH" |
| PowerShell script blocked | execution policy | `Set-ExecutionPolicy -Scope Process Bypass -Force` |
| `UNPROTECTED PRIVATE KEY FILE` | Windows key permissions | `icacls "<key>" /inheritance:r /grant:r "%USERNAME%:R"` |
| `Server connection ✗` | key not authorized | `.\kur_istemci.ps1 -SadeceAnahtar` then run the printed command |
| `torch` sees no CUDA | CPU wheel installed | `pip install torch --index-url .../cu124 --force-reinstall` |
| verification shows `bf16` • | emulation on Pascal | normal, informational |
| pool shows 1 node | duplicate `node_id` | delete `.neurocore\node_id` |
| not `READY`, packages `MISSING` | pip failure | `pip install -r requirements.txt` |

Log file: `.neurocore/logs/neurocore.log`
