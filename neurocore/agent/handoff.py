"""
Proje teslimi — başlatma betikleri ve sunum
===========================================
Kullanıcı sorusu: "Bize yaptığı projeyi kolay başlatabilmemiz için .bat/.ps1
dosyası oluşturdu mu? Başlangıçtaki plan/walkthrough güzelse, **proje sonu ve
sunum da güzel olmalı**."

Cevap: bu modül, çalışma bittiğinde projeye **elle dokunmadan** şunları ekler:

```text
baslat.ps1     PowerShell ile tek komut: bağımlılık + sunucu + tarayıcı
baslat.bat     çift tıkla çalışır (PowerShell betiğini çağırır)
SUNUM.md       teslim sunumu: ne yapıldı, mimari, çalıştırma, testler, sıradaki
sunum.html     tek dosyalık, karanlık temalı sunum sayfası (paylaşılabilir)
```

Kurallar (MUST)
---------------
```
· Var olan dosya **ezilmez** (kullanıcı/ajan yazdıysa dokunulmaz)
· Komutlar projeden türetilir: package.json scripts → npm start / npm test
  yoksa server.js → node server.js · main.py → python main.py
· Betikler yalnızca **çalıştırma** yapar; veri silmez, ağ dışına çıkmaz
```
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurocore.utils.logger import NeuroCoreLogger

logger = NeuroCoreLogger.get_logger("agent.handoff")

#: Elle yazılmış dosyaların üzerine yazma
KORUNAN = ("baslat.ps1", "baslat.bat", "SUNUM.md", "sunum.html")


# ─── komut keşfi ────────────────────────────────────────────────────────────

def komutlari_bul(kok: Path) -> Dict[str, Any]:
    """
    Projenin çalıştırma/test komutlarını **dosyalardan** türetir.

    Returns:
        {"calistir": [...], "test": [...], "kurulum": [...], "dil": "node|python|"}
    """
    kok = Path(kok)
    calistir: List[str] = []
    test: List[str] = []
    kurulum: List[str] = []
    dil = ""

    paket = kok / "package.json"
    if paket.exists():
        dil = "node"
        try:
            veri = json.loads(paket.read_text(encoding="utf-8"))
            betikler = veri.get("scripts") or {}
            if "start" in betikler:
                calistir.append("npm start")
            elif "dev" in betikler:
                calistir.append("npm run dev")
            if "test" in betikler:
                test.append("npm test")
            if (veri.get("dependencies") or veri.get("devDependencies")):
                kurulum.append("npm install")
        except (OSError, json.JSONDecodeError):
            pass

    if not calistir:
        for aday, komut in (("server.js", "node server.js"),
                            ("index.js", "node index.js"),
                            ("app.py", "python app.py"),
                            ("main.py", "python main.py"),
                            ("manage.py", "python manage.py runserver")):
            if (kok / aday).is_file():
                calistir.append(komut)
                dil = dil or ("node" if aday.endswith(".js") else "python")
                break

    if not test:
        if (kok / "tests").is_dir() or (kok / "test").is_dir():
            test.append("node --test" if dil == "node" else "python -m pytest")

    return {"calistir": calistir, "test": test, "kurulum": kurulum, "dil": dil}


# ─── betik üretimi ──────────────────────────────────────────────────────────

def _ps1(proje: str, komutlar: Dict[str, Any]) -> str:
    """PowerShell başlatma betiği (bağımlılık + sunucu + tarayıcı)."""
    kurulum = ""
    if komutlar.get("kurulum"):
        kurulum = (
            'if (-not (Test-Path "node_modules")) {\n'
            '  Write-Host "Bagimliliklar kuruluyor (npm install)..." -ForegroundColor Cyan\n'
            '  npm install\n'
            '}\n')
    calistir = (komutlar.get("calistir") or ["echo 'Calistirma komutu bulunamadi'"])[0]
    return f'''# {proje} — tek komutla baslat
# Kullanim:  .\\baslat.ps1          (PowerShell)
#            .\\baslat.bat          (cift tikla)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
Write-Host "=== {proje} ===" -ForegroundColor Cyan
{kurulum}
# Sunucuyu arka planda baslat ve tarayiciyi ac
Write-Host "Sunucu baslatiliyor: {calistir}" -ForegroundColor Cyan
$env:PORT = if ($env:PORT) {{ $env:PORT }} else {{ "3000" }}
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "{calistir}"
Start-Sleep -Seconds 3
Start-Process "http://localhost:$env:PORT"
Write-Host "Tarayicida aciliyor: http://localhost:$env:PORT" -ForegroundColor Green
Write-Host "Kapatmak icin acilan sunucu penceresini kapatin." -ForegroundColor DarkGray
'''


def _bat(proje: str) -> str:
    """Çift tıkla çalışan sarmalayıcı."""
    return f'''@echo off
rem {proje} — cift tikla baslat
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0baslat.ps1"
if errorlevel 1 pause
'''


# ─── sunum ──────────────────────────────────────────────────────────────────

def sunum_md(kart: Dict[str, Any], run_ozet: Optional[Dict[str, Any]] = None,
             komutlar: Optional[Dict[str, Any]] = None) -> str:
    """Teslim sunumu (markdown)."""
    kart = kart or {}
    komutlar = komutlar or {}
    mim = kart.get("mimari") or {}
    satirlar = [
        f"# {kart.get('proje', 'Proje')} — Teslim Sunumu",
        "",
        f"> {time.strftime('%Y-%m-%d %H:%M')} · NeuroCore tarafından üretildi",
        "",
        "## 1. Ne yapıldı?",
        "",
    ]
    if kart.get("gorev"):
        satirlar.append(f"- **Görev:** {kart['gorev'][:400]}")
    satirlar.append(f"- **Durum:** {kart.get('durum', '—')}")
    if kart.get("adim"):
        satirlar.append(f"- **Araç adımı:** {kart['adim']}")
    if run_ozet:
        if run_ozet.get("tests"):
            satirlar.append(f"- **Testler:** {run_ozet['tests']}")
        if run_ozet.get("duration"):
            satirlar.append(f"- **Süre:** {run_ozet['duration']}")

    satirlar += ["", "## 2. Mimari", ""]
    if mim.get("yigin"):
        satirlar.append(f"- **Yığın:** {', '.join(mim['yigin'])}")
    if mim.get("giris_noktalari"):
        satirlar.append(f"- **Giriş noktaları:** {', '.join(mim['giris_noktalari'])}")
    if mim.get("klasorler"):
        satirlar.append("- **Klasörler:** " + " · ".join(
            f"`{k['ad']}/` ({k['dosya']} dosya)" for k in mim["klasorler"][:10]))
    if mim.get("toplam_dosya"):
        satirlar.append(f"- **Toplam dosya:** {mim['toplam_dosya']}")

    satirlar += ["", "## 3. Nasıl çalıştırılır?", "", "```powershell",
                 ".\\baslat.bat        # ya da:  .\\baslat.ps1", "```", ""]
    if komutlar.get("kurulum"):
        satirlar += ["Elle:", "", "```powershell"]
        satirlar += komutlar["kurulum"] + (komutlar.get("calistir") or [])
        satirlar += ["```", ""]
    if komutlar.get("test"):
        satirlar += ["Testler:", "", "```powershell", *komutlar["test"], "```", ""]

    satirlar += ["## 4. Ne yapıldı (adımlar)", ""]
    for adim in (kart.get("dosyalar") or [])[:10]:
        satirlar.append(f"- `{adim}`")
    if not kart.get("dosyalar"):
        satirlar.append("- (değişen dosya kaydı yok)")

    satirlar += ["", "## 5. Sıradaki adımlar", "",
                 "- [ ] Oynanış/elle test",
                 "- [ ] README'yi güncelle (yeni uçlar, kurallar)",
                 "- [ ] Paketleme ve sürüm etiketi", ""]
    return "\n".join(satirlar)


def sunum_html(kart: Dict[str, Any], md_metin: str) -> str:
    """Tek dosyalık, karanlık temalı sunum sayfası (paylaşılabilir)."""
    kart = kart or {}
    mim = kart.get("mimari") or {}
    maddeler = "".join(
        f"<li><code>{k.get('ad')}/</code> <span>{k.get('dosya')} dosya</span></li>"
        for k in (mim.get("klasorler") or [])[:10])
    giris = "".join(f"<code>{g}</code> " for g in (mim.get("giris_noktalari") or []))
    return f'''<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<title>{kart.get("proje", "Proje")} — Sunum</title>
<style>
 body{{background:#0f1115;color:#e6e8ee;font:16px/1.6 system-ui,Segoe UI,sans-serif;
      margin:0;padding:48px}}
 h1{{font-size:34px;margin:0 0 6px}} .sub{{color:#9aa4b2;margin-bottom:28px}}
 .card{{background:#171a21;border:1px solid #2a2f3a;border-radius:14px;
       padding:20px 24px;margin:16px 0}}
 h2{{font-size:20px;margin:0 0 10px;color:#8ab4f8}}
 code{{font-family:Consolas,monospace;font-size:14px;background:#0f1115;
      border:1px solid #2a2f3a;border-radius:6px;padding:1px 6px}}
 ul{{margin:6px 0 0 18px}} li span{{color:#9aa4b2;font-size:13px}}
 .durum{{color:#7ee787;font-weight:600}}
 pre{{background:#0f1115;border:1px solid #2a2f3a;border-radius:10px;
     padding:12px 14px;overflow:auto;font-size:13px}}
</style></head><body>
<h1>{kart.get("proje", "Proje")}</h1>
<div class="sub">{time.strftime("%Y-%m-%d %H:%M")} · NeuroCore teslim sunumu ·
 <span class="durum">{kart.get("durum", "")}</span></div>

<div class="card"><h2>Ne yapıldı</h2>
 <p>{kart.get("gorev", "")[:400]}</p></div>

<div class="card"><h2>Mimari</h2>
 <p>Yığın: <code>{", ".join(mim.get("yigin") or []) or "—"}</code> ·
  Toplam dosya: <code>{mim.get("toplam_dosya", 0)}</code></p>
 <p>Giriş: {giris or "—"}</p>
 <ul>{maddeler}</ul></div>

<div class="card"><h2>Çalıştırma</h2>
 <pre>.\\baslat.bat     # çift tıkla
.\\baslat.ps1     # PowerShell</pre>
 <p>{" · ".join(kart.get("komutlar") or []) or ""}</p></div>

<div class="card"><h2>Ayrıntılı teslim notu</h2>
 <pre>{md_metin[:2500]}</pre></div>
</body></html>
'''


# ─── yazma ──────────────────────────────────────────────────────────────────

def yaz(kok: Path, kart: Dict[str, Any],
        run_ozet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Projeye başlatma betiklerini ve sunumu yazar.

    ⚠ Var olan dosyalar **ezilmez** (`KORUNAN` listesi).

    Returns:
        {"ok": bool, "yazilan": [...], "atlanan": [...]}
    """
    kok = Path(kok)
    if not kok.is_dir():
        return {"ok": False, "error": f"dizin yok: {kok}"}
    komutlar = komutlari_bul(kok)
    if not komutlar["calistir"]:
        return {"ok": False, "error": "çalıştırma komutu bulunamadı",
                "komutlar": komutlar}

    proje = kart.get("proje") or kok.name
    md = sunum_md(kart, run_ozet, komutlar)
    hedefler = {
        "baslat.ps1": _ps1(proje, komutlar),
        "baslat.bat": _bat(proje),
        "SUNUM.md": md,
        "sunum.html": sunum_html(kart, md),
    }
    yazilan: List[str] = []
    atlanan: List[str] = []
    for ad, icerik in hedefler.items():
        yol = kok / ad
        if yol.exists():
            atlanan.append(ad)
            continue
        try:
            yol.write_text(icerik, encoding="utf-8", newline="\n")
            yazilan.append(ad)
        except OSError as exc:
            logger.warning(f"{ad} yazılamadı: {exc}")
    if yazilan:
        logger.info(f"Proje teslimi yazıldı: {', '.join(yazilan)}")
    return {"ok": True, "yazilan": yazilan, "atlanan": atlanan,
            "komutlar": komutlar}
