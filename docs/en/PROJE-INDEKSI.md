# Büyük projelerde ilişkili kodu bulma

Tek dosya penceresi **tek dosya** içindir. Binlerce dosya ve milyonlarca
satırda gereken şey **kalıcı kod indeksi**: semboller + import grafiği + FTS5.

## Ölçüm (2000 dosya · 5.994.000 satır)

| İşlem | Süre | Not |
|---|---|---|
| İlk indeks kurulumu | **39,8 sn** | 82.000 sembol · 2.000 ilişki |
| Veritabanı boyutu | **13,4 MB** | `.neurocore/index/kod.db` |
| Artımlı (değişiklik yok) | **0,15 sn** | yalnız mtime karşılaştırması |
| Artımlı (1 dosya değişti) | **0,18 sn** | değişen dosya yeniden taranır |
| `find_symbol` | **3 ms** | FTS5 |
| `code.map` | **8 ms** | dil dağılımı + en büyük dosyalar |

Kendi projemizde (325 dosya · 173.030 satır): indeks **2,7 sn**, sorgular **1–2 ms**.

## Şema

```
dosya(path, mtime, size, satir, dil, sembol_sayi)
sembol( path, satir, ad, tur )          ← class/func/const/signal/baslik
iliski( kaynak, hedef, tur )            ← import / require / preload / using
sembol_fts (FTS5)
```

## Araçlar

| Araç | Ne yapar |
|---|---|
| `code.index` | kurar/günceller (artımlı; `zorla=True` tam tarama) |
| `code.map` | dil dağılımı, en büyük dosyalar, sembol yoğunluğu |
| `code.find_symbol` | proje genelinde tanım arar |
| `code.find_references` | adı kullanan dosyalar (import grafiği) |
| `code.related` | dosyanın import ettikleri + onu kullananlar + sembolleri |

## Ajan protokolü

```
code.index → code.map → code.find_symbol / code.related → code.read_file(pencere)
```

## Sınırlar

- 8 MB üstü dosyalar yalnız **satır sayılır** (sembol çıkarılmaz)
- `node_modules`, `.git`, `.godot`, `build`, `models` gibi dizinler atlanır
- İlişkiler regex tabanlıdır (derleyici düzeyinde değil); dinamik import'lar
  kaçabilir — şüphede `code.search_text` ile çapraz kontrol edin
- SQLite WAL; eşzamanlı okuma güvenli

Kaynak: `neurocore/agent/codeindex.py` · Testler: `tests/test_codeindex.py` (16)
