# Refactor Guard

`neurocore/agent/refactor.py` + two loop gates.

## The incident

An agent was asked to split a 5,261-line controller into a separate movement class.
It reported success. The truth:

```
git diff --numstat : 706 deleted / 35 added
new file           : never grew (427 lines → 427 lines)
headless tests     : PASSED (the player scene never ran)
```

~671 lines of animation, camera and network code were gone.

## The rule

```
1) READ     code.outline, then ONE code.read_file(start_line, max_lines)  (≤800 lines)
2) ORDER    COPY → PASTE → VERIFY → only THEN DELETE
3) PARITY   every moved function must exist in the new file (find_symbol both sides)
4) MEASURE  git diff --numstat: deletions > 150 AND additions < deletions×0.60 → LOSS
5) TEST     parity/behaviour test must pass before finishing
```

## Gates

| Gate | When |
|---|---|
| `_refactor_denetle` | before the final answer |
| `_refactor_bitis_denetle` | at every run end (time-limit safe) |
| auto restore | `git checkout -- <file>` when ratio < 0.5 and `NC_OTOMATIK_GERI_AL=1` |

When loss is detected the run is marked `ok=False` with an explicit warning in the answer.

## Measurements

| State | numstat | Verdict |
|---|---|---|
| real incident | 706 / 35 | **LOSS=True** ✅ caught |
| after restore | 8 / 3 | LOSS=False |

`parite_kontrol()` supports GDScript, Python, JS/TS, Rust and Go.

---

# Refactor Koruma (Türkçe)

## Olay

Ajan 5.261 satırlık controller'ı bölmeyi "başardım" dedi. Gerçek:

```
git diff --numstat : 706 silinen / 35 eklenen
yeni dosya         : hiç büyümedi
headless testler   : GEÇTİ (oyuncu sahnesi çalışmadı)
```

## Kural

```
1) OKU      code.outline → TEK code.read_file(start_line, max_lines) (≤800 satır)
2) SIRA     KOPYALA → YAPIŞTIR → DOĞRULA → en SON SİL
3) PARİTE   taşınan her fonksiyon yeni dosyada olmalı
4) ÖLÇ      git diff --numstat: silinen > 150 ve eklenen < silinen×0,60 → KAYIP
5) TEST     parite testi geçmeden bitirme
```

## Kapılar

| Kapı | Ne zaman |
|---|---|
| `_refactor_denetle` | nihai yanıttan önce |
| `_refactor_bitis_denetle` | her koşu bitişinde (süre sınırı dahil) |
| otomatik geri alma | oran < 0,5 ve `NC_OTOMATIK_GERI_AL=1` |

Kayıp saptanırsa çalışma `ok=False` işaretlenir ve yanıta uyarı eklenir.
