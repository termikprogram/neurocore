# NeuroCore — English Quick Guide & Portability Matrix

**Version 0.1.0** · Windows-first · Python 3.10+ · PyTorch

NeuroCore is a local-first AI agent: it plans, uses tools (code, shell, Blender,
Godot), keeps a markdown knowledge vault, trains a small neural model, and can
share work with other NeuroCore machines over SSH.

> Turkish documentation is the primary one (README.md, docs/*.md). This file
> covers **what an English-speaking user needs** plus **which hardware works**.

---

## 1. Can an English-speaking user use it?

**Short answer: yes for the agent, limited for the local model.**

Everything the *agent* does runs on a **cloud LLM** (DeepSeek / Gemini / Groq),
which is multilingual. The **local 211M model** was trained on Turkish
documentation, so its English generation is weak.

Measured tokenizer cost (same tokenizer the model uses, vocab 4,679):

| Sample | Words | Tokens | Tokens/word |
|---|---|---|---|
| Turkish #1 | 18 | 45 | **2.50** |
| Turkish #2 | 14 | 27 | **1.93** |
| English #1 | 22 | 66 | **3.00** |
| English #2 | 18 | 56 | **3.11** |
| English #3 | 18 | 63 | **3.50** |

English costs **~20–40 % more tokens** than Turkish with this tokenizer. The
model still encodes/decodes English correctly (round-trip verified), but its
*statistics* are Turkish-heavy.

The agent's system prompt now contains an explicit language rule: **answer in
the user's language** (Turkish question → Turkish answer, English → English).

### What is translated vs not

| Item | English support |
|---|---|
| Agent responses (cloud LLM) | ✅ automatic (language rule) |
| Tool names, code, shell commands | ✅ language-neutral |
| Docs (README, docs/*.md) | ⚠️ Turkish (this file is the English entry point) |
| UI labels (`app/web`) | ⚠️ Turkish |
| Local model generation | ❌ weak (Turkish-trained, 4,679 vocab) |
| Knowledge vault notes | ⚠️ whatever language you and the agent write |

To make the local model better at English you would need to retrain the
tokenizer and model on an English/mixed corpus (see `tools/havuz_egitimi.py`).

---

## 2. Hardware portability

### 2.1 What the code actually requires

| Requirement | Where | Notes |
|---|---|---|
| Python 3.10+ | everywhere | 3.11/3.12 untested |
| PyTorch 2.x | neural layer | CPU wheel works, just slow |
| `torch.cuda` **or** CPU | model/trainer | CUDA-only code paths are guarded |
| `pynvml` | `hardware/monitor.py` | **optional** — import is inside `try/except`; missing NVML just disables GPU stats |
| `nvidia-smi` | `tools/shell.py` | optional info source |
| Blender 5.x | asset tools | only for Blender features |

Measured: with `torch.cuda.is_available()` forced to `False`, the resource
governor initialises fine and `model.py` moves the model to CPU. **No crash.**

### 2.2 GPU tiers

| GPU | VRAM | Backend | Status | Notes from measurements |
|---|---|---|---|---|
| GTX 1080 Ti (Pascal) | 11 GB | CUDA | ✅ tested | fp32/fp16 only; **`--batch 8 --amp-yok` needed** (driver TDR + fp16 underflow) |
| RTX 3060 Ti (Ampere) | 8 GB | CUDA | ✅ expected | native bf16, fp16 AMP stable, `--batch 8 --seq 128` |
| RTX 4090 / 5090 (Ada/Blackwell) | 24–32 GB | CUDA | ✅ expected | bf16 native; batch 32+ at seq 256 |
| A100 / H100 / H200 (AI) | 40–141 GB | CUDA | ✅ expected | single-GPU only (code uses device 0); no NCCL/multi-GPU |
| L40S / RTX 6000 Ada | 48 GB | CUDA | ✅ expected | same as Ada |
| AMD (ROCm, Linux) | 8–192 GB | ROCm | ⚠️ should work | PyTorch exposes ROCm through `torch.cuda.*`, so device detection works; **not tested here**; `nvidia-smi` path unused, `pynvml` guard needed (present) |
| AMD (Windows) | — | — | ❌ no | no ROCm on Windows; falls back to **CPU** |
| Apple Silicon (M1–M4) | shared | MPS | ❌ no | code has no MPS branch; falls back to CPU |
| Intel Arc / XPU | 8–16 GB | XPU | ❌ no | no XPU branch; CPU fallback |

### 2.3 What "CPU fallback" means in practice

The model is ~211M parameters (806 MB fp32). On CPU:
- inference works (slow: seconds per short completion),
- training is possible but ~10–50× slower than GPU,
- the **agent itself is unaffected** — cloud LLM does the thinking.

So on AMD-Windows, Apple Silicon or Intel GPUs the product still works as an
*agent + tools + vault*; only the local model becomes CPU-bound.

### 2.4 Multi-GPU

Not supported. `device` is a single string (`cuda` → device 0). For A100/H100
machines it uses the first GPU only.

---

## 3. Install (English)

```powershell
git clone https://github.com/termikprogram/neurocore.git
cd neurocore
Set-ExecutionPolicy -Scope Process Bypass -Force
.\kur_istemci.ps1            # Python + .venv + PyTorch CUDA + verify
```

Manual:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tools\kurulum_kontrol.py     # 36 checks → HAZIR
```

AMD (Linux, ROCm): install the ROCm PyTorch wheel instead:
```
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

CPU only:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 4. Run

```powershell
.\BASLAT-istemci.bat          # desktop app + shared pool
python -m neurocore.app --autonomy supervised
```

Ask in English; the agent answers in English. Example:

```
"Create a low-poly palm tree in Blender and render 4 preview angles"
"Read the config, list tools that need write permission, summarise risks"
```

## 5. Verify your machine

```powershell
python tools\kurulum_kontrol.py     # environment, GPU, VRAM, pool, tools
python tools\kabiliyet_olc.py       # 10 capability checks (offline, <1 s)
python tools\model_payi.py          # local vs cloud model usage
```

## 6. Known limits

- Windows-first: many tools assume PowerShell and Windows paths.
- Local model is Turkish-centric and small (0.51 tokens/parameter).
- No multi-GPU, no MPS/XPU backend.
- Driver stability on Pascal requires small batches (measured: TDR resets).
