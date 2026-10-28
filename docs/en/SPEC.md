# NeuroCore — System Specification (English)

**Version 0.1.0** · normative source: [../SPEC.md](../SPEC.md) (Turkish)

This is the English rendering of the technical contract. It is written so that
an AI agent or a new engineer can rebuild the system without reading the code
line by line.

---

## 0 · How to read this document

- **MUST** = invariant. Violating it is a bug, not a preference.
- Numbers in tables are **measured**, not estimated; where a number came from a
  specific machine it is marked *(measured)*.
- `→` means "therefore / leads to".

---

## 1 · System summary

```text
NeuroCore = local software agent
            + markdown knowledge vault
            + small neural model trained from scratch
            + resource governor (never hogs the machine)
            + optional multi-machine pool
```

The **brain is a cloud LLM** (DeepSeek by default, with failover). The local
211M-parameter model is *trained* and used for **ranking/selection**, not as
the primary reasoner. This is a deliberate split: weak hardware still gets a
capable agent.

---

## 2 · Layer map and start-up order

```text
core/config.py        configuration, profiles, credentials path
core/governor.py      resource snapshot + spendable budget
core/engine.py        NeuroCoreEngine — wraps Trainer + Model
neural/model.py       transformer (211,255,328 params measured)
neural/trainer.py     training loop, AMP dtype selection, validation
agent/loop.py         AgentLoop — plan / act / verify
agent/cognition.py    error taxonomy, goal state, reflection, verification
agent/tools.py        ToolRegistry (83 tools), policy, approvals
agent/learning.py     experience recording → training trigger
agent/playbooks.py    successful flows stored as procedures (JSONL)
agent/hierarchy.py    goal → sub-goals → steps (single LLM call)
knowledge/            vault index (SQLite) + markdown notes
net/peer_pool.py      SSH reverse tunnel, registry, dispatch
app/server.py         FastAPI + WebSocket UI
```

Start-up order: config → governor → engine (lazy) → tool registry → loop →
server. Every layer's `initialize(config, event_bus)` **MUST** be idempotent
and every layer **MUST** expose `shutdown()` and `get_stats()`.

---

## 3 · Agent loop (`agent/loop.py`)

```text
run(task, max_steps=32, max_seconds=600, plan_first=auto)
  ├─ plan (auto when len(task) ≥ 60)         thinking.py → cloud N candidates
  │                                          → local model ranks → pick 1
  ├─ playbook hint (similarity ≥ 0.35)       "prefer an existing tool first"
  ├─ hierarchical plan (len(task) ≥ 80)       goal → ≤4 sub-goals → ≤3 steps
  ├─ loop while steps < max_steps and now < deadline:
  │    ├─ select tools (core 28 guaranteed + task-specific, budget 40)
  │    ├─ router.chat(tools=schemas)          cloud LLM
  │    ├─ repair messages                     drop orphan `tool` messages
  │    ├─ execute tool calls                  local or peer
  │    ├─ record experience                   learning.record()
  │    ├─ reflection gate (max 2, every 8 tool steps)
  │    └─ compress context (>24 messages → goal-state summary + last 10)
  └─ verification gate                        evidence check, ≤1 correction
```

Step kinds: `plan · think · tool · recovery · critique · verify · playbook ·
answer · error`.

**MUST 1** — a repeated (tool, args) pair 3 times breaks the loop
(`REPEAT_LIMIT`).
**MUST 2** — every step is recorded with index, kind, ok, text.
**MUST 3** — a run always ends with `run.ok`, `run.answer`, `run.reason`.

---

## 3.5 · Cognition layer (`agent/cognition.py`)

| Component | Purpose | Bound |
|---|---|---|
| Error taxonomy | 9 classes → per-class recovery instruction | 1 instruction per failure |
| Goal state | goal · facts (≤12) · attempts (≤16) · open questions | summary ≤1200 chars |
| Self-reflection | `continue / replan / answer / report` | ≤ 2 calls |
| Evidence verification | is the answer supported by tool output? | 1 check + 1 correction |

```text
tool result: ok  → goal.add_fact(one-line evidence)
             err → kind = classify(error); message += RECOVERY(kind)
every 8 tool steps, max 2: reflection → replan | answer | report | continue
final answer: evidence = tool outputs (≤2500 chars)
              EVIDENCE EMPTY → verification SKIPPED
              unsupported ∧ budget left → answer = correct(...)   # single call
```

**MUST 4** — reflection ≤ 2, correction ≤ 1.
**MUST 5** — no verification call when there is no evidence.
**MUST 6** — a router failure degrades silently; the loop is never blocked.
**MUST 7** — `NC_COG=0` restores the previous behaviour exactly.

---

## 4 · Tool system (`agent/tools.py`)

```python
registry.register(name, fn, category="read|write|execute", needs_approval=bool)
registry.call(name, args) → ToolResult(ok, tool, data, error, duration_ms)
```

- **83 tools** registered *(measured)*.
- Policy: `readonly · supervised · trusted · yolo`.
- `CORE_TOOLS` = 28, always sent to the model; the tool budget is **40** and
  the core set is **never truncated** (violating this silently removed
  `asset.animasyon` from 4/4 tasks).
- Peer jobs may only use `ALLOWED_TOOLS` (16, read-only).

**MUST 8** — a write tool MUST be refused in `readonly` mode.
**MUST 9** — approval-required tools MUST NOT run remotely.

---

## 5 · Thinking loop (`agent/thinking.py`)

```text
think(task) → cloud proposes up to 3 candidates
            → local model ranks by sequence log-probability
            → pick margin ≥ 0.15, else keep the cloud's first choice
```

Measured: local ranking 488 ms; thinking (including lazy load) 18.3 s.

---

## 6 · Resource governor (`core/governor.py`)

```text
spendable = total × (1 − current) × 0.90        hard ceiling 85 %
```

| Snapshot | Usable |
|---|---|
| 50 % used | 45 % of total |
| 90 % free | 81 % → capped 85 % |
| other process holds GPU | training refuses: `resources_busy` |

**MUST 10** — no training step starts above the 85 % ceiling.
**MUST 11** — if resources are short, shrink batch/resolution before refusing.

---

## 7 · Neural model (`neural/model.py`)

```text
params 211,255,328 (measured) · layers 8 · d_model 768 · heads 12 · head_dim 64
vocab 4,679 · MoE 5 experts top-2 · max_seq_len 2048 · fp32
weights 806 MB · Adam state 2,418 MB
```

**MUST 12** — every training run records `steps` and `tokens` into
`models/neurocore/growth.json`.

---

## 8 · Training algorithm (`neural/trainer.py`)

```text
plan_training(budget) → batch, seq, lr, steps, grad_accumulation, use_amp
AMP dtype: SM ≥ 8 or ROCm gfx90a+ → bfloat16 (no scaler)
           SM < 8                    → float16 + GradScaler
           CPU                       → float32
LR ladder: base × {3.0, 1.5, 0.6} then saturation × {1, 0.5, 0.2, 0.05}
Validation loss is reported every `--eval` steps; it is the real metric.
```

**MUST 13** — validation loss is reported separately for document and
instruction data.
**MUST 14** — a vault note is only used for training if
`kind ∈ {reference, qa}` and `confidence ≥ 0.6`.

---

## 9 · Learning loop (`agent/learning.py`)

Four quality rules (each was measured to reject noise):

```text
1. length < 20 chars OR approval-word ratio ≥ 0.6 → note rejected
2. single-step confidence ≤ 0.6
   confidence = base × steps × (0.7 + 0.3 × success), capped 0.9
3. system-prompt prefixes are never stored as knowledge
4. training needs ≥ 50 new experiences or ≥ 20 forced steps
```

**MUST 15** — a training run never starts with fewer than `MIN_TRAIN_STEPS`
(20) steps.

---

## 10 · Vault (`knowledge/`)

```text
knowledge_base/        Obsidian-compatible markdown + YAML frontmatter
Index   : kind, tags, confidence, created, source
Search  : SQLite FTS + embedding fallback
```

Vault statistics *(measured at one point)*: 6,061 notes, of which 2,937 (48.5 %)
were `kind: skill`; the strict training filter kept 1,803 (29.7 %).

**MUST 16** — every note has frontmatter with `kind` and `confidence`.

---

## 11 · Shared pool (`net/peer_pool.py`)

```text
register  → reverse tunnel + peers.jsonl (TTL 90 s), IP filled from $SSH_CLIENT
heartbeat → every 30 s
dispatch  → POST /peer/job {token, tool, args}
takeover  → no answer → local node runs the job
```

Measured: tunnel 6.3 s · relay 1.7–2.7 s · local takeover 2.7 s.

**MUST 17** — the SSH private key is never shared between clients.
**MUST 18** — the cluster token comes from the server, never derived from a key.
**MUST 19** — `/peer/job` rejects **403** when the token is missing or wrong.

---

## 12 · Work splitting (`agent/bridge.py`, `core/work.py`)

```text
goal → chunks sized by spendable budget → per-chunk checkpoint to work.json
interrupted run → resume from the last checkpoint, never restart
```

---

## 13 · Application API (`app/server.py`)

| Endpoint | Purpose |
|---|---|
| `GET /api/state` | workspace, autonomy, LLM, tool counts, last run |
| `POST /api/chat` | start a task (returns immediately, streams over WS) |
| `GET /api/runs` | last run summary |
| `POST /api/settings/llm` | save an API key (never echoed back) |
| `GET /api/learning` | learning status |
| `POST /api/learning/train` | train now |
| `GET/POST /peer/info`, `/peer/job` | pool endpoints |

**MUST 20** — every endpoint checks `Origin` when it is present (CSRF guard).
**MUST 21** — long-blocking reads run in a worker thread with a timeout
(measured: `registry` lock held during a tool call froze `/api/state` for
180+ s before this rule).

---

## 14 · Configuration

```text
NC_COG, NC_COG_REFLECT=2, NC_COG_EVERY=8, NC_COG_VERIFY=1
NC_HIER (0/1)          hierarchical planning
NC_PEER (0/1)          shared pool
NC_SERVER, NC_SSH_KEY  pool configuration
NC_PEER_TOKEN          pin the cluster token
NC_EGITIM_NOTE_TURLERI training note kinds
```

---

## 15 · Invariants (short list for an AI agent)

```text
MUST 1   repeat limit 3 breaks the loop
MUST 2   every step recorded
MUST 3   run ends with ok/answer/reason
MUST 4   reflection ≤ 2 · correction ≤ 1
MUST 5   no verification without evidence
MUST 6   cognition failures are silent
MUST 7   NC_COG=0 restores prior behaviour
MUST 8   readonly refuses writes
MUST 9   approval tools never run remotely
MUST 10  no training above 85 % usage
MUST 11  shrink before refusing
MUST 12  every run records steps + tokens
MUST 13  validation loss reported separately
MUST 14  training notes filtered by kind + confidence
MUST 15  training needs ≥ 20 steps
MUST 16  notes carry kind + confidence frontmatter
MUST 17  SSH private key never shared
MUST 18  token comes from the server
MUST 19  /peer/job closed without a valid token
MUST 20  Origin check on every endpoint
MUST 21  blocking reads run threaded with a timeout
MUST 22  playbooks only from successful (≥0.6) runs with ≥2 steps
MUST 23  hierarchy tool names must come from the provided list
MUST 24  context compression keeps the system prompt + last 10 messages
MUST 25  core tools are never truncated from the tool budget
MUST 26  messages are repaired before each chat call (no orphan `tool`)
MUST 27  documented bugs get a regression test
```

---

## 15.5 · Playbooks · hierarchy · context compression

| Component | File | Rule | Bound |
|---|---|---|---|
| Playbook | `agent/playbooks.py` | successful flows become procedures | reward ≥0.6, ≥2 steps, ≤8 steps, similarity ≥0.35, 200 records |
| Hierarchy | `agent/hierarchy.py` | goal → ≤4 sub-goals → ≤3 steps | one call, 900 tokens; failure → flat plan |
| Compression | `loop._baglami_sikistir` | old turns collapse into the goal summary | keep system + last 10 |
| Capability set | `tools/kabiliyet_olc.py` | 10 deterministic checks | <1 s, JSON score |

Measured pitfall: plain Jaccard on Turkish suffixes gave **0.20** for
"ağaç" ↔ "ağaçları" (below the 0.35 threshold). Prefix-aware matching raises
it to **0.50**.

---

## 16 · Failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA unknown error` | driver TDR (2 s default) | `--batch 8 --seq 128 --amp-yok` |
| `HTTP 400 tool_calls` | orphan `tool` message after compression | repair before each call |
| `/api/state` freezes | registry lock held by a tool | threaded read + timeout |
| peer token rejected | pool not started, empty token | start the pool or pin `NC_PEER_TOKEN` |
| empty model answer | reasoning tokens ate the budget | use `deepseek-chat` |
| two machines, one node | copied `.neurocore/node_id` | delete the file |

---

## 17 · Measurement protocol (reproducible)

```powershell
python tools\kurulum_kontrol.py     # environment
python tools\kabiliyet_olc.py       # 10 capability checks
python tools\ag_havuzu.py --liste   # pool
python tools\havuz_uctan_uca.py     # tunnel/relay
python -m unittest discover -s tests
```

Report both **before** and **after** numbers; never trust a script's own print
without re-reading the artifact from disk.

---

## 18 · Build history (8 stages)

```text
1 framework       core layers, config, logging, events
2 model           tokenizer, transformer, MoE, training loop
3 agent           loop, tools, policy, approvals
4 app             FastAPI + WebSocket UI, chats, workspaces
5 knowledge       vault, index, retrieval
6 network         SSH tunnel, pool, dispatch, takeover
7 reasoning       thinking, planning, playbooks, cognition
8 portability     GPU/ISO detection, English docs & UI, packaging
```

---

## 19 · Glossary

| Term | Meaning |
|---|---|
| **Vault** | markdown knowledge store (`knowledge_base/`) |
| **Playbook** | stored successful tool sequence |
| **Governor** | resource budget calculator |
| **Pool / peer** | another NeuroCore instance reachable over SSH |
| **Takeover** | local node runs a job the peer did not answer |
| **Spendable** | `total × (1 − current) × 0.90` |

---

## Appendix A · File map

```text
neurocore/core/       config, governor, engine, work
neurocore/neural/     tokenizer, model, trainer, curriculum, embedding
neurocore/agent/      loop, tools, cognition, thinking, learning, playbooks, hierarchy
neurocore/knowledge/  vault, index, retrieval
neurocore/net/        peer_pool, tunnel
neurocore/app/        server, launcher, web/
tools/                installers, benchmarks, asset generators
tests/                1000+ unit tests
docs/                 Turkish primary + en/ (this folder)
```

## Appendix B · Version

```text
0.1.0 · 2026-10-29 · Python 3.10.11 · torch 2.6.0+cu124
```
