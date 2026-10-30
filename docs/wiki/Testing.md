# Testing

```powershell
python -m unittest tests.test_bigfile tests.test_codeindex tests.test_codeast `
                  tests.test_refactor tests.test_toolresult tests.test_envcheck
```

| Suite | Count | Covers |
|---|---|---|
| `test_bigfile` | 16 | sparse index, windows, memory bound (<8 MB), stale index, search |
| `test_codeindex` | 18 | symbols, relations, incremental, deletion, `.gdignore`, file-path search |
| `test_codeast` | 20 | Python AST, GDScript parser, call graph, MRO, types, interactive map |
| `test_refactor` | 18 | loss detection, restore threshold, real `git checkout`, multi-language parity |
| `test_toolresult` | 10 | structure preservation above 12K chars |
| `test_envcheck` | 16 | OS detection, known paths, install commands, missing deps |

Total: **98 tests**.

## Project-level tests (per project)

The agent writes headless tests too, e.g. in TitoWarWeb:

```powershell
godot --headless --path . --script tests/test_movement_parity.gd
```

Note: headless tests do **not** run the player scene — that is why the
refactor guard measures `git diff --numstat` before trusting a green test run.

---

# Testler (Türkçe)

Toplam **98 test**. Ayrıca ajan proje içinde headless test yazıyor
(örn. `test_movement_parity.gd`).

Önemli not: headless testler **oyuncu sahnesini çalıştırmaz** — bu yüzden
refactor kapısı yeşil testlere güvenmeden önce `git diff --numstat` ölçer.
