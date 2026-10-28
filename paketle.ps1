# NeuroCore — İstemci Paketi Oluşturucu
# ======================================
# İkinci makineye taşınacak **gerekli** dosyaları tek zip'te toplar.
#
# Kullanım:
#     .\paketle.ps1
#     .\paketle.ps1 -Cikti "D:\neurocore-istemci.zip"
#
# ⚠ Dışarıda bırakılanlar (bilinçli):
#   models\        23,6 GB  → ağırlıklar makineye özgü, taşınmaz
#   data\           6,5 GB  → checkpoint'ler
#   .neurocore\     2,3 GB  → veri havuzu + DÜĞÜM KİMLİĞİ (kopyalanmamalı!)
#   Varliklar\      125 MB  → Blender çıktıları (isteğe bağlı)
#   TestProjects\    23 MB  → deneme projeleri
#   __pycache__, .venv, *.log, _*.json

[CmdletBinding()]
param(
    [string]$Cikti = "",
    [switch]$VarliklarDahil,
    [switch]$KnowledgeBaseDahil = $true
)

$ErrorActionPreference = "Continue"
$kok = $PSScriptRoot
Set-Location $kok

if (-not $Cikti) { $Cikti = Join-Path (Split-Path $kok -Parent) "neurocore-istemci.zip" }

Write-Host ("=" * 74)
Write-Host "  NEUROCORE — ISTEMCI PAKETI"
Write-Host ("=" * 74)
Write-Host "  kaynak : $kok"
Write-Host "  cikti  : $Cikti"

# ── Geçici kopya dizini ───────────────────────────────────────────────
$gecici = Join-Path $env:TEMP ("nc-paket-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $gecici -Force | Out-Null
Write-Host "  gecici : $gecici"

$dahil = @("neurocore", "tools", "tests", "raporlar")
if ($KnowledgeBaseDahil) { $dahil += "knowledge_base" }
if ($VarliklarDahil) { $dahil += "Varliklar" }

$kopyalanan = 0
foreach ($d in $dahil) {
    $kaynak = Join-Path $kok $d
    if (-not (Test-Path $kaynak)) { Write-Host "  - atlandi (yok): $d"; continue }
    $hedef = Join-Path $gecici $d
    robocopy $kaynak $hedef /E /NFL /NDL /NJH /NJS /NP /XD "__pycache__" ".git" ".venv" | Out-Null
    if ($LASTEXITCODE -ge 8) { Write-Host "  ! robocopy hata: $d ($LASTEXITCODE)" }
    $kopyalanan++
    Write-Host "  + $d"
}

# ── Kök dosyalar ──────────────────────────────────────────────────────
$kokDosyalar = @(
    "requirements.txt", "kur_istemci.ps1", "BASLAT-istemci.bat",
    "paketle.ps1", "README.md"
)
foreach ($f in $kokDosyalar) {
    $kaynak = Join-Path $kok $f
    if (Test-Path $kaynak) {
        Copy-Item $kaynak (Join-Path $gecici $f) -Force
        Write-Host "  + $f"
    }
}

# ── Zip ───────────────────────────────────────────────────────────────
if (Test-Path $Cikti) { Remove-Item $Cikti -Force }
Write-Host ""
Write-Host "  Sikistiriliyor..."
Compress-Archive -Path (Join-Path $gecici "*") -DestinationPath $Cikti -CompressionLevel Optimal
$boyut = (Get-Item $Cikti).Length / 1MB
Remove-Item $gecici -Recurse -Force -ErrorAction SilentlyContinue

# ── Özet ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host ("=" * 74)
Write-Host ("  PAKET HAZIR: {0}" -f $Cikti)
Write-Host ("  Boyut      : {0:N1} MB  ({1} klasor)" -f $boyut, $kopyalanan)
Write-Host ""
Write-Host "  Diger makinede:"
Write-Host "    1) zip'i ac"
Write-Host "    2) .\kur_istemci.ps1"
Write-Host "    3) .venv\Scripts\python.exe tools\kurulum_kontrol.py --kaydet"
Write-Host "    4) .\BASLAT-istemci.bat"
Write-Host ("=" * 74)
