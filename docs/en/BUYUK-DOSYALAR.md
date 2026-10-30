# Büyük dosyalar — 10 milyon satır

`read_file` tüm dosyayı belleğe almaz. 2 MB üstü dosyalarda **seyrek satır
indeksi** kullanılır:

| İşlem | 10 M satır (427 MB) ölçümü |
|---|---|
| `indeks_kur` (ilk erişim) | 8,2 sn · 0,6 MB bellek · 48 KB yan dosya |
| `read_file(start_line=9_999_500, max_lines=100)` | **11 ms** · 232 KB bellek |
| `code.outline` | 2 ms (yapı haritası) |
| `search_text` (akışlı) | 6,7 sn · dosya atlanmıyor |

## Nasıl çalışır

1. **Seyrek indeks** — her 2000. satırın bayt ofseti `<dosya>.ncidx.json`
   içine yazılır (mtime+size ile doğrulanır, değişirse yeniden kurulur).
   Rastgele satıra `seek` ile atlanır; bellek sabit.
2. **Pencere okuma** — yalnız istenen aralık okunur (en fazla 2000 satır).
3. **Yapısal harita** — `code.outline`: `def/class/func/# başlık` satırları;
   ajan önce haritayı görür, sonra ilgili aralığı ister.
4. **Akışlı arama** — 4 MB üstü dosyalar artık atlanmaz; satır satır taranır.

## Ajan protokolü (sistem isteminde)

```
outline → read_file(start_line, max_lines) → edit_file → test
```

## Araçlar

| Araç | Açıklama |
|---|---|
| `code.read_file` | büyük dosyada otomatik indeksli pencere |
| `code.outline` | yapı haritası (satır numaralı) |
| `code.search_text` | büyük dosyada akışlı arama |

Kaynak: `neurocore/agent/bigfile.py` · Testler: `tests/test_bigfile.py` (16)
