# Architecture

```
neurocore/
  core/        engine, event bus, governor
  hardware/    monitor, allocator (politeness: 90% of free), profiler
  neural/      tokenizer, model, embedding, memory, trainer, growth
  knowledge/   vault (Obsidian-compatible), index (FTS5), search, graph
  agent/       loop, llm router, tools, cognition, puzzle, checkpoint,
               playbooks, bigfile, codeindex, refactor, handoff, envcheck
  app/         launcher, server (FastAPI), web UI, chats
  net/         peer pool (SSH discovery, job sharing, relay)
```

## Agent loop

```
task → rules? → plan (puzzle phases) → env preflight → tool loop
     → gates (tool-mandatory, item, evidence, refactor)
     → final answer → checkpoint → handoff files
```

## Resource politeness

`harcanabilir = total × (1 − used) × 0.90`, hard ceiling 85% for RAM/VRAM/CPU.

## Persistence

- `<project>/.neurocore/` — checkpoint, project card, puzzle, index DB
- `knowledge_base/` — vault notes (Obsidian front-matter)
- `data/` — memory bank, checkpoints

---

# Mimari (Türkçe)

```
core/        motor, olay veri yolu, governor
hardware/    izleme, kaynak tahsisi (boştakinin %90'ı), profiler
neural/      tokenizer, model, gömme, bellek, eğitici, büyüme
knowledge/   kasa (Obsidian uyumlu), indeks (FTS5), arama, graf
agent/       loop, llm yönlendirici, araçlar, biliş, puzzle, checkpoint,
             playbook, bigfile, codeindex, refactor, handoff, envcheck
app/         launcher, sunucu (FastAPI), web arayüzü, sohbetler
net/         eş havuzu (SSH keşif, iş paylaşımı, relay)
```

## Ajan döngüsü

```
görev → kurallar? → plan (puzzle fazları) → ortam ön denetimi → araç döngüsü
      → kapılar (araç zorunlu, madde, kanıt, refactor)
      → nihai yanıt → checkpoint → teslim dosyaları
```
