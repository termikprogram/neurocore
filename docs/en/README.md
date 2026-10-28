# NeuroCore — English documentation

Turkish is the primary documentation language; these files are the English
rendering for international users and for AI agents reading the repository.

| Document | What it covers |
|---|---|
| **[INSTALL.md](INSTALL.md)** | requirements, GPU-aware auto-install (Windows + Linux/macOS), manual install, verification, packaging, troubleshooting |
| **[USAGE.md](USAGE.md)** | running the app, autonomy levels, how the agent works, example tasks, CLI tools, training, resource politeness, vault |
| **[SHARED-POOL.md](SHARED-POOL.md)** | multi-machine job sharing: reverse tunnel, registry, cluster token, allowed tools, endpoints, limits |
| **[SPEC.md](SPEC.md)** | the technical contract: layer map, agent loop, cognition layer, tools, governor, training, vault, pool, 27 MUST invariants, failure modes, build history |

Quick links:

- Main README (English): [../../README.md](../../README.md)
- Türkçe README: [../../README.tr.md](../../README.tr.md)
- Normative Turkish spec: [../SPEC.md](../SPEC.md)
- Diagrams and screenshots: [../images](../images)

```text
Reading order for a new engineer:
  1. README.md            what it is and how to install it
  2. docs/en/SPEC.md      the contract you must not break
  3. docs/en/USAGE.md     day-to-day operation
  4. docs/en/SHARED-POOL.md  only if you run more than one machine
```
