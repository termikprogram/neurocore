# Usage

**Version 0.1.0**

---

## 1 · Desktop application

```powershell
.\BASLAT-istemci.bat           # shared pool ON
python -m neurocore.app        # same thing
```

The browser opens at `http://127.0.0.1:<port>` (next free port if taken).

**The interface opens in English.** Switch languages from
**Settings → Language** (English / Türkçe); the change applies **immediately**
(no reload) and is remembered. `?lang=tr` / `?lang=en` overrides it for one visit.

### Options

```powershell
python -m neurocore.app --workspace E:/PROJECTS/mygame    # workspace
python -m neurocore.app --autonomy readonly               # read-only
python -m neurocore.app --no-window --port 9000           # server only
python -m neurocore.app --browser                         # default browser
python -m neurocore.app --log-level debug                 # verbose log
```

---

## 2 · Autonomy levels

| Level | Read | Write | Shell | Approval |
|---|---|---|---|---|
| `readonly` | ✓ | ✗ | ✗ | — |
| `supervised` | ✓ | ✓ | ✓ | **on every action** |
| `trusted` | ✓ | ✓ | ✓ | not required for low risk |
| `yolo` | ✓ | ✓ | ✓ | never asks |

> `yolo` is not recommended for production work.

---

## 3 · How the agent works

```text
task ──▶ plan (thinking: cloud proposes N candidates → local model ranks them)
     ──▶ tool selection (28 core tools + task-specific ones, budget 40)
     ──▶ [shared pool] allowed read-only tool is tried on a peer first
     ──▶ observe → next step (max 32 steps / 600 s)
     ──▶ answer + tools used + step log
```

- **Planning engages automatically** for longer tasks (≥ 60 characters).
- Every step is visible in the UI: `[remote:relay:NNNNN]` or `[local]`.
- Repeating the same tool + arguments 3 times breaks the loop (`REPEAT_LIMIT`).
- A **cognition layer** adds: error taxonomy, goal state, self-reflection
  (≤ 2 times), evidence verification (≤ 1 correction). Disable with `NC_COG=0`.

---

## 4 · Example tasks

```text
"Investigate the DXGI_ERROR_DEVICE_HUNG crash and propose a fix"
"Run the tests in this project and fix what breaks"
"Create a sword model, add an animation, render 4 preview angles"
"Summarise the Godot 4 signal notes from the vault"
"Read this file and list its functions"
```

For research-style tasks the agent uses `web.search` / `web.fetch` /
`knowledge.ask` on its own.

---

## 5 · Command-line tools

```powershell
python tools\kurulum_kontrol.py              # environment verification
python tools\kabiliyet_olc.py                # 10 capability checks (offline)
python tools\ag_havuzu.py --liste            # nodes in the pool
python tools\ag_havuzu.py --kaydet --port 8791
python tools\havuz_uctan_uca.py              # tunnel + register + relay measurement
python tools\ajan_is_paylasimi.py            # agent job sharing measurement
python tools\nat_testi.py                    # NAT / reverse tunnel test
```

### Training

```powershell
# prepare data (deduplicate + in-document prompts)
python tools\havuz_hazirla.py

# instruction (question → answer) data
python tools\talimat_verisi.py --uret

# train (validation loss is reported)
python tools\havuz_egitimi.py --adim 300 --eval 100 --ornek 20000 `
       --seq 128 --batch 8 --kaydet-her 100 --dakika 25
```

| Parameter | Meaning |
|---|---|
| `--seq` | sequence length (128 → roughly 2× faster) |
| `--batch` | batch size (8 for 8 GB VRAM) |
| `--kaydet-her` | checkpoint interval |
| `--amp-yok` | disable fp16 AMP (if Pascal raises CUDA errors) |
| `--talimat` | mix in question → answer data |
| `--dakika` | time limit (the resource governor tightens it further) |

> **Validation loss is the real metric**; training loss alone is misleading.
> The report shows document and instruction validation separately.

AMP dtype is chosen automatically: `bfloat16` on Ampere/Ada/Hopper and ROCm
gfx90a+, `float16` + GradScaler on Pascal, `float32` on CPU.

---

## 6 · Resource politeness

```text
spendable = total × (1 − current) × 0.90          hard ceiling 85 %
```

| Situation | Usable |
|---|---|
| system 50 % busy | 45 % of total |
| system 90 % free | 81 % → capped at 85 % |
| another process holds the GPU | training does not start (`resources_busy`) |

If resources are short the planner **shrinks batch size / resolution**; if it
cannot start it writes the reason. Monitor through the UI panel or
`neurocore.core.engine`.

---

## 7 · The vault (knowledge store)

```text
knowledge_base/            Obsidian-compatible markdown
.neurocore/index/          SQLite search index
```

- `knowledge.ask` → answer from the vault
- `learn.topic` → research a topic and write it to the vault
- `learn.status` → learning state

---

## 8 · Shared pool

Job sharing with NeuroCore instances on other machines:
**[SHARED-POOL.md](SHARED-POOL.md)**
