# Project Index

`neurocore/agent/codeindex.py` runs on SQLite (+FTS5) inside `.neurocore/index/kod.db`.

## Schema

```
dosya   (path, mtime, size, satir, dil)
sembol  (path, satir, ad, tur)          class/func/const/signal/heading
iliski  (kaynak, hedef, tur)            import / include / require
cagri   (kaynak_sembol, hedef_ad, dosya, satir)
tip     (sembol, ad, tip, tur)          parameter / return / assignment
kalitim (alt, taban, dosya, satir)      class hierarchy
sembol_fts (FTS5)
```

## Parsing

| Language | Method |
|---|---|
| Python | real `ast` — docstrings never match, qualified names (`Class.method`), `ast.Call` for the call graph, annotations + `x = Cls()` for types |
| GDScript | structural scanner — `class_name/func/signal/const`, `preload`, calls inside functions |
| Others | line patterns (JS/TS, Rust, Go, C#, C/C++, markdown) |

`.gdignore` directories are skipped. Files above 8 MB are counted, not parsed.

## Measurements

| Workload | Full index | Incremental | Query |
|---|---|---|---|
| 2,000 files / 5.99 M lines | **39.8 s** · 13.4 MB | **0.15 s** | 3–8 ms |
| TitoWarWeb 3.7 GB | **7.3 s** · 497 files · 144,035 lines | — | 4–25 ms |

## Tools

| Tool | Purpose |
|---|---|
| `code.index` | build/update (incremental, `zorla=True` for full) |
| `code.map` | languages, biggest files, symbol density |
| `code.find_symbol` | definitions (FTS5) |
| `code.find_references` | who uses a name |
| `code.callers` | call graph: callers / callees |
| `code.related` | imports + users + symbols of a file |
| `code.hierarchy` / `code.mro` | class bases and chain |
| `code.types` | parameter/return/assignment types |
| `code.impact_map` | interactive HTML+SVG call graph |
| `code.outline` | structure map of one file |

---

# Proje İndeksi (Türkçe)

SQLite (+FTS5) veritabanı: `dosya`, `sembol`, `iliski`, `cagri`, `tip`, `kalitim`.

## Ayrıştırma

| Dil | Yöntem |
|---|---|
| Python | gerçek `ast` — docstring eşleşmez, nitelikli adlar, `ast.Call` ile çağrı grafiği, ek açıklamalar ve `x = Sinif()` ile tipler |
| GDScript | yapısal tarayıcı — `class_name/func/signal/const`, `preload`, fonksiyon içi çağrılar |
| Diğer | satır kalıpları (JS/TS, Rust, Go, C#, C/C++, markdown) |

`.gdignore` dizinleri atlanır; 8 MB üstü dosyalar yalnız satır sayılır.

## Ölçümler

| Yük | Tam indeks | Artımlı | Sorgu |
|---|---|---|---|
| 2.000 dosya / 5,99 M satır | **39,8 sn** · 13,4 MB | **0,15 sn** | 3–8 ms |
| TitoWarWeb 3,7 GB | **7,3 sn** · 497 dosya · 144.035 satır | — | 4–25 ms |
