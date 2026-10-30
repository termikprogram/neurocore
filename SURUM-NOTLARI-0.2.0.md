# NeuroCore v0.2.0 — big projects, code index, refactor guard

> Released **2026-10-30** · every claim below is **measured**, not estimated.

## 🇬🇧 English

### New: big files (10,000,000 lines)
`neurocore/agent/bigfile.py` — sparse line index + windowed read + streaming search.

| Operation | 10 M lines / 427 MB |
|---|---|
| index build | **8.2 s** · 0.6 MB RAM · 48 KB sidecar |
| random window (`start_line=9_999_500`) | **11 ms** · 232 KB RAM |
| streaming grep | 6.7 s (previously the file was **skipped** entirely) |

### New: project code index
`neurocore/agent/codeindex.py` — SQLite (+FTS5) with `sembol`, `iliski`, `cagri`,
`tip`, `kalitim`. AST-based for Python, structural parser for GDScript,
incremental (mtime/size), `.gdignore` aware.

| Operation | 2,000 files / 5.99 M lines | TitoWarWeb (3.7 GB) |
|---|---|---|
| full index | **39.8 s** · DB 13.4 MB | **7.3 s** · 497 files · 144,035 lines |
| incremental | **0.15 s** | — |
| `find_symbol` / `map` | **3–8 ms** | 19 ms / 4 ms |

### New tools (94 total)
`code.index`, `code.map`, `code.find_symbol`, `code.find_references`,
`code.callers`, `code.related`, `code.hierarchy`, `code.mro`, `code.types`,
`code.impact_map`, `code.outline`
* `code.impact_map` renders an **interactive** HTML+SVG call-graph (29 nodes, 9.6 KB, no CDN).

### New: refactor parity guard
`neurocore/agent/refactor.py` + loop gates:
* `git diff --numstat`: deletions > 150 and additions < deletions×0.60 → **code loss**
* `parite_kontrol()` across GDScript · Python · JS/TS · Rust · Go
* runs **before the final answer** *and* **at run end** (time-limit safe) with
  optional automatic `git checkout` restore
* Real catch: **706 deletions / 35 additions** → restored automatically.

### Fixed (tool pipeline)
* `ToolResult.to_dict()` turned `data` into a JSON **string** above 12K chars → the
  agent saw only metadata. Structure is preserved now.
* Tool messages used `render()` (raw JSON truncated to 6 KB) instead of the
  structured formatter.
* Context compression deleted read files → read-memory keeps 3 files in the summary.
* `tool_call` ids desynced between the assistant message and tool results →
  results were dropped as orphans.
* `search_text(path=<file>)` returned 0 results (treated a file as a directory).
* Content cap 14K → **36K** chars (`NC_ICERIK_SINIRI`); `read_file` default 400 → 800 lines.
* Gate isolation: a crashing gate no longer kills a run.

### Gates
* **Tool-mandatory gate** — task names `code.*` and no tool was called → answer rejected
  (measured: tool results in request **0 → 3**).
* **Item gate** — N-item tasks require evidence per item, counted from **messages**
  (plan/puzzle steps were inflating the count).
* **Evidence reminder** — if a symbol appears in tool output but the answer says "not found",
  the exact lines are quoted back for correction.

### Tests — 98 passing
`test_bigfile` 16 · `test_codeindex` 18 · `test_codeast` 20 · `test_refactor` 18 ·
`test_toolresult` 10 · `test_envcheck` 16

---

## 🇹🇷 Türkçe

### Yeni: büyük dosyalar (10.000.000 satır)
Seyrek satır indeksi + pencere okuma + akışlı arama.

| İşlem | 10 M satır / 427 MB |
|---|---|
| indeks kurulumu | **8,2 sn** · 0,6 MB RAM · 48 KB yan dosya |
| rastgele pencere (`start_line=9_999_500`) | **11 ms** · 232 KB RAM |
| akışlı arama | 6,7 sn (önceden dosya **tamamen atlanıyordu**) |

### Yeni: proje kod indeksi
SQLite (+FTS5): `sembol`, `iliski`, `cagri`, `tip`, `kalitim`. Python için AST,
GDScript için yapısal ayrıştırıcı, artımlı (mtime/size), `.gdignore` uyumlu.

| İşlem | 2.000 dosya / 5,99 M satır | TitoWarWeb (3,7 GB) |
|---|---|---|
| tam indeks | **39,8 sn** · DB 13,4 MB | **7,3 sn** · 497 dosya · 144.035 satır |
| artımlı | **0,15 sn** | — |
| `find_symbol` / `map` | **3–8 ms** | 19 ms / 4 ms |

### Yeni araçlar (toplam 94)
`code.index`, `code.map`, `code.find_symbol`, `code.find_references`,
`code.callers`, `code.related`, `code.hierarchy`, `code.mro`, `code.types`,
`code.impact_map`, `code.outline`
* `code.impact_map` **etkileşimli** HTML+SVG çağrı grafiği üretir (29 düğüm, 9,6 KB, CDN yok).

### Yeni: refactor parite kapısı
* `git diff --numstat`: silinen > 150 ve eklenen < silinen×0,60 → **kod kaybı**
* `parite_kontrol()`: GDScript · Python · JS/TS · Rust · Go
* **yanıttan önce** ve **koşu bitişinde** (süre sınırı dahil) çalışır;
  isteğe bağlı otomatik `git checkout`
* Gerçek yakalama: **706 silinen / 35 eklenen** → otomatik geri alındı.

### Düzeltilenler (araç boru hattı)
* `ToolResult.to_dict()` 12K karakter üstünde `data`yı JSON **string**'e çeviriyordu →
  ajan yalnız meta veri görüyordu. Yapı korunuyor.
* Araç mesajları `render()` (ham JSON, 6 KB) kullanıyordu → yapılandırılmış biçim.
* Bağlam sıkıştırması okunan dosyayı siliyordu → okuma belleği (3 dosya).
* `tool_call` kimlikleri eşleşmiyordu → sonuçlar yetim sayılıp siliniyordu.
* `search_text(path=<dosya>)` 0 sonuç dönüyordu (dosya dizin sanılıyordu).
* İçerik sınırı 14K → **36K** (`NC_ICERIK_SINIRI`); `read_file` 400 → 800 satır.
* Kapı hata yalıtımı: çöken kapı artık koşuyu öldürmüyor.

### Kapılar
* **Araç zorunlu**: görevde `code.*` geçiyor ve araç çağrılmamışsa yanıt reddedilir
  (ölçüm: istekte tool **0 → 3**).
* **Madde kapısı**: N maddeli görevde madde başına kanıt; kanıt **mesajlardan** sayılır.
* **Kanıt hatırlatma**: sembol araç çıktısında varken "yok" denirse satırlar alıntılanır.

### Testler — 98 geçiyor
`test_bigfile` 16 · `test_codeindex` 18 · `test_codeast` 20 · `test_refactor` 18 ·
`test_toolresult` 10 · `test_envcheck` 16

---

**Full changelog:** [CHANGELOG.md](https://github.com/termikprogram/neurocore/blob/main/CHANGELOG.md)
