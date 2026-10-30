# FAQ

**Does it read 10-million-line files?**
Yes. Sparse index + windowed reads: random window in **11 ms**, index build 8.2 s.

**Does it work on huge repositories?**
Yes. TitoWarWeb (3.7 GB) → 497 files / 144,035 lines indexed in **7.3 s**;
queries in 4–25 ms.

**Does it find related code?**
`code.callers`, `code.find_references`, `code.related`, `code.hierarchy`,
`code.mro`, `code.types`, `code.impact_map`.

**Can it lose my code during a refactor?**
It can — that is why the parity guard exists. It measures `git diff --numstat`,
marks the run failed and (by default) restores the file.

**Does it need internet?**
The model does (cloud LLM). The index, big-file reader and gates are local.

**Which model?**
DeepSeek `deepseek-chat` by default; Groq/Gemini/OpenRouter configurable.

---

# SSS (Türkçe)

**10 milyon satırlık dosyayı okur mu?** Evet — rastgele pencere **11 ms**.
**Çok büyük depoda çalışır mı?** Evet — 3,7 GB depo **7,3 sn**'de indekslenir.
**İlişkili kodu bulur mu?** `callers`, `find_references`, `related`, `hierarchy`, `mro`, `types`, `impact_map`.
**Refactor'da kod kaybeder mi?** Kaybedebilir; parite kapısı bunun için var — `numstat` ölçer, koşuyu başarısız işaretler ve dosyayı geri alır.
