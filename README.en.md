# NeuroCore

**A local-first AI agent for your own computer.** It plans, calls tools
(shell, code, Blender, Godot), keeps a markdown knowledge vault, trains a
small neural model in the background, and can share work with other
NeuroCore machines over SSH — never spending more than 90 % of the resources
you actually have free.

![NeuroCore main window](docs/images/app-main.png)

<sub>Real screenshot: DeepSeek provider active, 83 tools registered, 4,670 tokens used on the status bar.</sub>

---

## Table of contents

1. [What it does](#1-what-it-does)
2. [How it works](#2-how-it-works)
3. [Requirements](#3-requirements)
4. [Install (GPU-aware)](#4-install-gpu-aware)
5. [Add your API key](#5-add-your-api-key)
6. [First run](#6-first-run)
7. [Verify your machine](#7-verify-your-machine)
8. [Hardware support matrix](#8-hardware-support-matrix)
9. [Using it in English](#9-using-it-in-english)
10. [Output examples](#10-output-examples)
11. [Known limits](#11-known-limits)

---

## 1. What it does

| Capability | What you get |
|---|---|
| **Agent loop** | plan → tool call → observe → verify → answer (bounded, with reflection and recovery) |
| **Tools** | 83 registered: filesystem, shell, code editing, diagnosis, Blender, Godot, docs, vision, web |
| **Knowledge vault** | Obsidian-compatible markdown notes with frontmatter, searchable, used as retrieval source |
| **Local model** | 211M-parameter model trained from scratch on your own notes (0.51 tokens/param so far) |
| **Cognition layer** | error taxonomy, goal state, self-reflection, evidence verification (max 2 reflections, 1 correction) |
| **Learning** | successful work flows are stored as *playbooks* and suggested on similar tasks |
| **Resource governance** | `spendable = total × (1 − current) × 0.90`, hard ceiling 85 % |
| **Shared pool** | machines register over SSH and take dispatched jobs; local node takes over if a peer does not answer |

## 2. How it works

```text
                 ┌──────────────────────────────┐
   your task ───▶│  AgentLoop (agent/loop.py)   │
                 │  plan → act → verify         │
                 └───────┬───────────┬──────────┘
                         │           │
          tools (83) ────┘           └──── cloud LLM (DeepSeek / Gemini / Groq)
     files · shell · code ·                 │
     Blender · Godot · docs                 └─ if unavailable → failover chain
                         │
                 ┌───────▼──────────────┐
                 │  Cognition layer     │  error taxonomy · goal state
                 │  (agent/cognition)   │  reflection (≤2) · verification (≤1)
                 └───────┬──────────────┘
                         │
        ┌────────────────┼─────────────────┐
        ▼                ▼                 ▼
  knowledge vault   local model      playbooks
  (markdown notes)  (trainer,        (successful
                     growth.json)     tool sequences)
```

**The brain is a cloud LLM; the local model is trained, not used as the brain.**
That is why the agent works on weak hardware while the local model keeps
learning in the background.

## 3. Requirements

| Item | Minimum | Notes |
|---|---|---|
| OS | Windows 10/11 | Windows-first (PowerShell tools) |
| Python | 3.10 | 3.10.11 tested |
| GPU | any NVIDIA with 6 GB+, or CPU | see [matrix](#8-hardware-support-matrix) |
| Disk | 8 GB | PyTorch + model + vault |
| Accounts | one LLM API key | DeepSeek is the default and cheapest |

## 4. Install (GPU-aware)

The installer **detects your GPU and picks the matching PyTorch build**.
Nothing is hard-coded to CUDA 12.4 any more.

```powershell
git clone https://github.com/termikprogram/neurocore.git
cd neurocore
Set-ExecutionPolicy -Scope Process Bypass -Force
.\kur_istemci.ps1
```

### Check what it would pick (no installation)

```powershell
.\kur_istemci.ps1 -SadeceGpu
```

Measured output on a GTX 1080 Ti:

```text
── 2b) GPU ALGILAMA ve PyTorch SEÇİMİ
  GPU        : NVIDIA GeForce GTX 1080 Ti
  sürücü     : 582.78
  PyTorch    : cu124
  gerekçe    : NVIDIA sürücü 582 → cu124
```

### Detection rules

| Detected | PyTorch build |
|---|---|
| NVIDIA driver ≥ 550 | `cu124` |
| NVIDIA driver ≥ 525 | `cu121` |
| NVIDIA driver < 525 | `cu118` |
| AMD / Radeon on **Windows** | `cpu` (ROCm does not exist on Windows) |
| Intel GPU | `cpu` |
| No GPU | `cpu` |

Override any time:

```powershell
.\kur_istemci.ps1 -Torch cu121     # or cu124 / cu118 / cpu / auto (default)
```

### Manual install

```powershell
python -m venv .venv
# NVIDIA
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
# CPU only
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**AMD on Linux (ROCm)** — use the ROCm wheel, the code needs no changes
because PyTorch exposes ROCm through `torch.cuda.*`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

## 5. Add your API key

Three equivalent ways. The key is stored locally and **never returned** by
the API.

### 5.1 From the app (recommended)

1. Open the app (`.\BASLAT-istemci.bat` or `python -m neurocore.app`).
2. Click the **gear icon** (top right) → **Dil Modeli / Language model**.
3. Pick a provider, paste the key, press **Bağlantıyı sına** (test).
4. Press save. The status pill in the header turns into `DeepSeek · deepseek-chat`.

### 5.2 From a file

`.neurocore/llm_credentials.json` (git-ignored):

```json
{
  "keys": { "deepseek": "sk-..." },
  "models": { "deepseek": "deepseek-chat" },
  "base_urls": {},
  "provider": "deepseek",
  "updated_at": "2026-10-29T00:00:00"
}
```

### 5.3 From environment variables

| Provider | Variable |
|---|---|
| DeepSeek (default) | `DEEPSEEK_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Together | `TOGETHER_API_KEY` |

```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sk-...", "User")
```

Providers are tried in order with automatic failover; if one is rate-limited
the next one answers.

## 6. First run

```powershell
.\BASLAT-istemci.bat
```

The window opens at `http://127.0.0.1:8791`. Type a task. Autonomy levels:

| Level | Behaviour |
|---|---|
| `danışman` (advisor) | asks before anything that writes |
| **`denetimli` (supervised)** | default: writes/commands need approval, reads are free |
| `güvenilir` (trusted) | writes allowed, destructive commands still ask |
| `yolo` | everything except sandbox escape |

Example tasks that were actually run:

```text
"Create a low-poly palm tree in Blender and render 4 angles"     → 3 trees + 12 PNGs
"Build a trial blog site in WebTest: index.html, style.css, js"   → 9/9 checks passed
"Read the config, list tools needing write permission, summarise" → report
```

## 7. Verify your machine

```powershell
python tools\kurulum_kontrol.py     # 36 environment checks
python tools\kabiliyet_olc.py       # 10 capability checks, offline, <1 s
python tools\model_payi.py          # local vs cloud model usage
```

`kurulum_kontrol.py` now reports the backend explicitly:

```text
[✓] GPU backend    torch 2.6.0+cu124 · CUDA 12.4 (NVIDIA)
                   ↳ AMD/Linux: ROCm 6.x (AMD)  ·  no GPU: CPU
```

## 8. Hardware support matrix

| GPU | VRAM | Backend | Status | Notes (measured) |
|---|---|---|---|---|
| GTX 1080 Ti (Pascal) | 11 GB | CUDA | ✅ **tested** | fp16/fp32 only; `--batch 8 --amp-yok` (driver TDR + fp16 underflow) |
| RTX 3060 Ti (Ampere) | 8 GB | CUDA | ✅ expected | native bf16, `--batch 8 --seq 128` |
| RTX 4090 / 5090 | 24–32 GB | CUDA | ✅ expected | bf16 native, batch 32+ |
| A100 / H100 / H200 | 40–141 GB | CUDA | ✅ expected | **single GPU only** (device 0, no NCCL) |
| L40S / RTX 6000 Ada | 48 GB | CUDA | ✅ expected | same as Ada |
| AMD (ROCm, Linux) | 8–192 GB | ROCm | ⚠️ should work | ROCm is visible through `torch.cuda.*`; not tested in this repo |
| AMD (Windows) | — | — | ❌ | no ROCm → CPU fallback |
| Apple Silicon | shared | MPS | ❌ | no MPS branch → CPU |
| Intel Arc / XPU | 8–16 GB | XPU | ❌ | no XPU branch → CPU |

**CPU fallback is graceful** — measured: with `torch.cuda.is_available()`
forced to `False`, the resource governor and the model both initialise and
fall back to CPU. The *agent* is unaffected (cloud LLM does the thinking);
only the local training/inference becomes slow.

## 9. Using it in English

The agent answers in the language you write in — the system prompt now
contains an explicit language rule (before: Turkish prompt → Turkish answers
even for English questions).

**Measured tokenizer cost** (vocab 4,679 · 4,408 merges):

| Sample | Words | Tokens | Tokens/word |
|---|---|---|---|
| Turkish #1 | 18 | 45 | **2.50** |
| Turkish #2 | 14 | 27 | **1.93** |
| English #1 | 22 | 66 | **3.00** |
| English #2 | 18 | 56 | **3.11** |
| English #3 | 18 | 63 | **3.50** |

English costs ~20–40 % more tokens; round-trip encode/decode is correct in
both languages. The **local model** remains Turkish-centric (it was trained
on Turkish notes) — the cloud model handles English natively.

UI labels and the main documentation are Turkish; this file is the English
entry point. See also [docs/SPEC.md](docs/SPEC.md) for the technical contract.

## 10. Output examples

**Blender — three low-poly trees, four angles** (agent job, 302 s, 37 tool calls):

![Low-poly trees](docs/images/example-trees.png)

**Palm, rebuilt from the new fan recipe** (single-blade fronds, no joint gap):

![Palm](docs/images/example-palm.png)

## 11. Known limits

- **Windows-first**: many tools assume PowerShell and Windows paths.
- **Single GPU**: no multi-GPU / NCCL; big AI cards use one device.
- **No MPS/XPU** backend.
- **Pascal needs small batches**: `--batch 8 --amp-yok`, otherwise driver resets.
- **Local model is small and Turkish-centric** (0.51 tokens/param).
- **No bf16 auto-selection yet**: AMP uses fp16+scaler everywhere; A100/H100
  would be more stable with bf16.
- The UI is Turkish; only this document is English.

---

## License

Proprietary — see [LICENSE](LICENSE). Turkish documentation: [README.md](README.md).
