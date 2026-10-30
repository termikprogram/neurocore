# NeuroCore Wiki

**English** · [Türkçe](Home-TR)

A local software agent that plans, writes code, runs commands, drives Godot/Blender,
shares jobs across machines — and, since **v0.2.0**, reads *very large* projects.

## Start here

| Page | What it covers |
|---|---|
| [Getting Started](Getting-Started) | install, first run, autonomy levels |
| [Architecture](Architecture) | layers, event bus, agent loop |
| [Big Files](Big-Files) | 10,000,000-line files: index, window, search |
| [Project Index](Project-Index) | symbols, imports, call graph, types, hierarchy |
| [Refactor Guard](Refactor-Guard) | parity gate, `git diff --numstat`, auto-restore |
| [Tool Pipeline](Tool-Pipeline) | how tool results reach the model (and how they used to vanish) |
| [Testing](Testing) | 98 tests, how to run them |
| [FAQ](FAQ) | common questions |

## Measured, not estimated

Every number in this wiki is measured on a real machine (Windows 11, Python 3.10.11).

```
10,000,000-line file   → index 8.2 s · random window 11 ms · RSS 232 KB
2,000-file project     → full index 39.8 s · DB 13.4 MB · query 3–8 ms
TitoWarWeb (3.7 GB)    → 497 files · 144,035 lines · indexed in 7.3 s
```

---

# NeuroCore Wiki (Türkçe)

**Türkçe** · [English](Home)

Planlayan, kod yazan, komut çalıştıran, Godot/Blender süren ve makineler arasında
iş paylaşan yerel yazılım ajanı — **v0.2.0** ile birlikte *çok büyük* projeleri de okur.

| Sayfa | İçerik |
|---|---|
| Sayfa (EN + TR) | İçerik |
|---|---|
| [Getting Started](Getting-Started) | kurulum, ilk çalıştırma, özerklik seviyeleri |
| [Architecture](Architecture) | katmanlar, olay veri yolu, ajan döngüsü |
| [Big Files](Big-Files) | 10.000.000 satır: indeks, pencere, arama |
| [Project Index](Project-Index) | semboller, import, çağrı grafiği, tipler |
| [Refactor Guard](Refactor-Guard) | parite kapısı, `numstat`, otomatik geri alma |
| [Tool Pipeline](Tool-Pipeline) | araç sonuçları modele nasıl ulaşır |
| [Testing](Testing) | 98 test, nasıl çalıştırılır |
| [FAQ](FAQ) | sık sorulanlar |
