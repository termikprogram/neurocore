# Big Files — 10,000,000 lines

`neurocore/agent/bigfile.py`

## The problem

`read_file` used `text.splitlines()` — a 10 M-line file is ~250 MB of text and
~1–2 GB of Python objects. `search_text` **skipped** any file above 4 MB.

## The solution

```
1) SPARSE LINE INDEX   every 2,000th line's byte offset → <file>.ncidx.json
2) WINDOW READ         seek + bounded read (≤2,000 lines), constant memory
3) STRUCTURE MAP       one streaming pass → symbols / headings
4) STREAMING SEARCH    line by line, bounded results
```

## Measurements (10,000,000 lines · 427 MB)

| Operation | Time | Memory |
|---|---|---|
| index build | **8.2 s** | **0.6 MB** (48 KB sidecar) |
| random window `start_line=9_999_500` | **11 ms** | 232 KB |
| deep window `start_line=5_000_000` | **1 ms** | — |
| streaming grep | 6.7 s | constant |
| `code.outline` | 2 ms | — |

## Usage

```python
from neurocore.agent import bigfile
bigfile.indeks_kur(path)                 # build/refresh index
bigfile.pencere(path, bas=9_999_500, adet=100)
bigfile.ara(path, "PATTERN", sinir=50)
bigfile.harita(path)                     # structure map
```

`code.read_file` switches to windowed mode automatically above 2 MB.

---

# Büyük Dosyalar — 10.000.000 satır (Türkçe)

## Sorun

`read_file` tüm dosyayı belleğe alıyordu; `search_text` 4 MB üstü dosyaları
**tamamen atlıyordu**.

## Çözüm

```
1) SEYREK SATIR İNDEKSİ  her 2000. satırın bayt ofseti → <dosya>.ncidx.json
2) PENCERE OKUMA         seek + sınırlı okuma (≤2000 satır), sabit bellek
3) YAPI HARİTASI         tek geçişte sembol/başlık çıkarımı
4) AKIŞLI ARAMA          satır satır, sınırlı sonuç
```

## Ölçümler (10.000.000 satır · 427 MB)

| İşlem | Süre | Bellek |
|---|---|---|
| indeks kurulumu | **8,2 sn** | **0,6 MB** (48 KB yan dosya) |
| rastgele pencere `start_line=9_999_500` | **11 ms** | 232 KB |
| derin pencere `start_line=5_000_000` | **1 ms** | — |
| akışlı arama | 6,7 sn | sabit |

`code.read_file` 2 MB üstünde otomatik pencere moduna geçer.
