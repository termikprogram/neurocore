# Getting Started

## Requirements

- Python **3.10+** (tested 3.10.11)
- Node.js 18+ *(only for JS projects / `npm` tooling)*
- Godot 4.x *(only for Godot projects; export templates for web builds)*
- Git (recommended: enables refactor parity guard)

## Install

```powershell
git clone https://github.com/termikprogram/neurocore.git
cd neurocore
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m neurocore.app --port 8791
```

Open <http://127.0.0.1:8791>.

## First run

1. Pick a **workspace** (project folder) from the sidebar.
2. Describe a task in the chat box.
3. Watch **Activity** for tool calls; **Project** for the architecture card;
   **Puzzle** for the phase plan.

## Autonomy levels

| Level | Behaviour |
|---|---|
| `supervised` | asks before writes/commands |
| `yolo` | executes within sandbox limits |

```powershell
python -m neurocore.app --autonomy yolo
```

## Large projects

Before asking questions about a big repository:

```
code.index   → build/update the index (incremental)
code.map     → languages, biggest files
code.find_symbol("name") / code.callers("name")
```

---

# Kurulum (Türkçe)

## Gereksinimler

- Python **3.10+** (3.10.11 ile test edildi)
- Node.js 18+ *(yalnız JS projeleri / `npm` araçları için)*
- Godot 4.x *(yalnız Godot projeleri; web export için şablonlar)*
- Git (önerilir: refactor parite kapısı için)

## Kurulum

```powershell
git clone https://github.com/termikprogram/neurocore.git
cd neurocore
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m neurocore.app --port 8791
```

<http://127.0.0.1:8791> adresini açın.

## İlk çalıştırma

1. Kenar çubuğundan **çalışma alanı** (proje klasörü) seçin.
2. Sohbet kutusuna görevi yazın.
3. **Activity** araç çağrılarını, **Project** mimari kartını,
   **Puzzle** faz planını gösterir.

## Özerklik seviyeleri

| Seviye | Davranış |
|---|---|
| `supervised` | yazma/komut öncesi onay ister |
| `yolo` | sandbox sınırları içinde çalıştırır |

## Büyük projeler

Büyük bir depoda soru sormadan önce:

```
code.index   → indeksi kur/güncelle (artımlı)
code.map     → diller, en büyük dosyalar
code.find_symbol("ad") / code.callers("ad")
```
