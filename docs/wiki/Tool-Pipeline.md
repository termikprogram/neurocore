# Tool Pipeline

How a tool result becomes something the model can actually see.

```
tool → ToolResult → to_dict() → _summarize_result() → LLMMessage(role="tool") → provider
```

## Four failures we measured (and fixed)

| # | Bug | Symptom |
|---|---|---|
| 1 | `to_dict()` turned `data` into a JSON **string** above 12K chars | model saw only metadata: "no tool output" |
| 2 | tool messages used `render()` (raw JSON truncated to 6 KB) | a 20 KB file arrived as 6 KB of JSON |
| 3 | context compression dropped read files after 10 messages | agent "forgot" what it read |
| 4 | `tool_call` ids desynced from the assistant `raw_message` | results treated as orphans and deleted |

Also fixed: `search_text(path=<file>)` returned 0 results (file treated as a
directory), content cap 14K → 36K, `read_file` default 400 → 800 lines,
and gate-level error isolation (a crashing gate no longer kills a run).

## Gates that force evidence

| Gate | Trigger | Effect |
|---|---|---|
| tool-mandatory | task names `code.*`, no tool called | answer rejected, tool call demanded |
| item gate | 2+ numbered items, evidence < items | per-item evidence requested |
| evidence reminder | answer says "not found" while tool output contains it | exact lines quoted back |

Measured effect of the tool-mandatory gate: tool results in the request **0 → 3**.

## Debugging

```powershell
$env:NC_DUMP_REQ="1"   # .neurocore/last_request.json  (full message list)
$env:NC_DUMP_SEQ="1"   # .neurocore/last_requests.jsonl (every call)
$env:NC_DUMP="1"       # last 40 messages
```

---

# Araç Boru Hattı (Türkçe)

```
araç → ToolResult → to_dict() → _summarize_result() → LLMMessage(role="tool") → sağlayıcı
```

## Ölçülen dört hata

| # | Hata | Belirti |
|---|---|---|
| 1 | `to_dict()` 12K üstünde `data`yı JSON **string**'e çeviriyordu | model yalnız meta veri gördü: "araç çıktısı yok" |
| 2 | araç mesajları `render()` kullanıyordu (ham JSON, 6 KB) | 20 KB dosya 6 KB JSON olarak geldi |
| 3 | bağlam sıkıştırması okunan dosyayı 10 mesaj sonra siliyordu | ajan okuduğunu unuttu |
| 4 | `tool_call` kimlikleri eşleşmiyordu | sonuçlar yetim sayılıp silindi |

Ayrıca: `search_text(path=<dosya>)` 0 sonuç, içerik sınırı 14K → 36K,
`read_file` 400 → 800 satır, kapı hata yalıtımı.
