# NeuroCore — İstemci Makine Kurulumu (tek komut)
# ================================================
# Yeni makinede (ör. RTX 3060 Ti) her şeyi yapar:
#
#   1. SSH ANAHTARI  — kendi anahtarını ÜRETİR, sunucuda dener,
#                      yetkili değilse **yapılacak komutu yazar**
#   2. Python denetimi (>= 3.10)
#   3. .venv + PyTorch (CUDA) + requirements.txt
#   4. tools/kurulum_kontrol.py ile doğrulama
#
# Kullanım:
#     .\kur_istemci.ps1
#     .\kur_istemci.ps1 -SadeceAnahtar            # yalnız anahtar adımı
#     .\kur_istemci.ps1 -SadeceKontrol            # yalnız doğrulama
#     .\kur_istemci.ps1 -YetkiBekle               # yetkilendirmeyi bekle
#     .\kur_istemci.ps1 -AnahtarDizin D:\keys     # anahtarı başka yere koy
#
# ⚠ Özel anahtar HİÇBİR makineyle paylaşılmaz. Sunucuya yalnız **açık**
#   anahtar eklenir. Küme anahtarı (token) sunucudan otomatik dağıtılır.

[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [ValidateSet("cu124", "cu121", "cu118")]
    [string]$Torch = "cu124",
    [string]$Sunucu = "",        # boşsa NC_SERVER / .neurocore/peer.json
    [string]$AnahtarDizin = "",
    [string]$AnahtarAdi = "",
    [switch]$VenvYok,
    [switch]$SadeceKontrol,
    [switch]$SadeceAnahtar,
    [switch]$YetkiBekle,
    [switch]$AnahtarYenile        # var olan anahtarı yok say, yenisini üret
)

# ⚠ "Stop" cmdlet hatalarında durur; ama python/pip uyarıyı stderr'e yazar
# (bu normaldir) — bu yüzden yerel komutlar için Continue.
$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

$kok = $PSScriptRoot
Set-Location $kok

function Adim($n, $metin) {
    Write-Host ""
    Write-Host "── $n) $metin" -ForegroundColor Cyan
}
function Mutlu($metin) { Write-Host "  [ok] $metin" -ForegroundColor Green }
function Uyari($metin) { Write-Host "  [!]  $metin" -ForegroundColor Yellow }
function Hata($metin)  { Write-Host "  [x]  $metin" -ForegroundColor Red }

# ⚠ Sunucu adresi: parametre → NC_SERVER → .neurocore/peer.json
if (-not $Sunucu) { $Sunucu = $env:NC_SERVER }
if (-not $Sunucu) {
    $ayarYolu = Join-Path $kok ".neurocore\peer.json"
    if (Test-Path $ayarYolu) {
        try { $Sunucu = (Get-Content $ayarYolu -Raw | ConvertFrom-Json).server } catch { }
    }
}
if (-not $Sunucu) {
    $Sunucu = "ubuntu@SUNUCU-ADRESI"
}

Write-Host ("=" * 74)
Write-Host "  NEUROCORE — ISTEMCI KURULUMU"
Write-Host ("=" * 74)
Write-Host "  klasor : $kok"
Write-Host "  sunucu : $Sunucu"
Write-Host "  torch  : $Torch"

# ══════════════════════════════════════════════════════════════════════
#  SSH ANAHTARI — üret · doğrula · (gerekirse) yetkilendirme talimatı
# ══════════════════════════════════════════════════════════════════════
function Test-SshAnahtar {
    param([string]$Yol)
    if ([string]::IsNullOrWhiteSpace($Yol) -or -not (Test-Path $Yol)) { return $false }
    $cikti = & ssh -i $Yol -o BatchMode=yes -o StrictHostKeyChecking=accept-new `
        -o IdentitiesOnly=yes -o ConnectTimeout=10 $Sunucu "echo NC_OK" 2>&1
    $metin = ($cikti | Out-String)
    return ($LASTEXITCODE -eq 0 -and $metin -match "NC_OK")
}

function Kisitla-Izin {
    param([string]$Yol)
    try {
        $kullanici = $env:USERNAME
        & icacls $Yol /inheritance:r /grant:r "$($kullanici):F" 2>&1 | Out-Null
    } catch { }
}

Adim "1b" "SSH ANAHTARI"

if (-not $AnahtarDizin) { $AnahtarDizin = Join-Path $env:USERPROFILE ".ssh" }
if (-not $AnahtarAdi) {
    # ⚠ Makineye özgü ad: proje klasörü kopyalansa bile anahtarlar çakışmaz
    $kisaAd = ($env:COMPUTERNAME -replace '[^A-Za-z0-9_-]', '')
    if (-not $kisaAd) { $kisaAd = "node" }
    $AnahtarAdi = "nc_$($kisaAd.ToLower())"
}
if (-not (Test-Path $AnahtarDizin)) {
    New-Item -ItemType Directory -Path $AnahtarDizin -Force | Out-Null
}
$ozel = Join-Path $AnahtarDizin $AnahtarAdi
$acik = "$ozel.pub"

# ── 1) Zaten çalışan bir anahtar var mı? ─────────────────────────────
$adaylar = @()
if ($env:NC_SSH_KEY) { $adaylar += $env:NC_SSH_KEY }
if (-not $AnahtarYenile) {
    # ⚠ Depoda sabit anahtar yolu YOK: yalnız ortam/ayar + üretilen anahtar
    $adaylar += @(
        $ozel,
        (Join-Path $env:USERPROFILE ".ssh\nc_node2"),
        (Join-Path $env:USERPROFILE ".ssh\id_ed25519")
    )
}

$calisan = ""
foreach ($a in $adaylar) {
    if ([string]::IsNullOrWhiteSpace($a)) { continue }
    Write-Host "  * deneniyor: $a"
    if (Test-SshAnahtar $a) { $calisan = $a; break }
}

if ($calisan) {
    Mutlu "Çalışan anahtar bulundu: $calisan"
    $ozel = $calisan
    $acik = "$calisan.pub"
    $env:NC_SSH_KEY = $calisan
    [Environment]::SetEnvironmentVariable("NC_SSH_KEY", $calisan, "User")
    Mutlu "NC_SSH_KEY kaydedildi (kalıcı, kullanıcı ortamı)"
} else {
    Write-Host "  * çalışan anahtar yok — bu makine için üretilecek"
    if (-not (Test-Path $ozel)) {
        Mutlu "anahtar üretiliyor: $ozel"
        # ⚠ `-N '""'` = boş parola. PowerShell 5.1 boş bir argümanı
        # düşürür → ssh-keygen parola sorar ve betik kilitlenir. Bu yüzden
        # iki tırnaklı biçim kullanılır.
        & ssh-keygen -t ed25519 -f $ozel -N '""' -C "neurocore-$($env:COMPUTERNAME)" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $ozel)) {
            Hata "anahtar üretilemedi (ssh-keygen); elle: ssh-keygen -t ed25519 -f `"$ozel`""
        } else {
            Mutlu "özel anahtar: $ozel"
        }
    } else {
        Uyari "anahtar zaten var, yeniden kullanılıyor: $ozel"
    }
    Kisitla-Izin $ozel
    $env:NC_SSH_KEY = $ozel
    [Environment]::SetEnvironmentVariable("NC_SSH_KEY", $ozel, "User")
    Mutlu "NC_SSH_KEY kaydedildi: $ozel"

    # ── Yetkilendirme gerekli ─────────────────────────────────────────
    Write-Host ""
    Uyari "bu makinenin anahtarı sunucuda YETKİLİ DEĞİL (beklenen: ilk kurulum)"
    if (Test-Path $acik) {
        Write-Host ""
        Write-Host "  AÇIK ANAHTAR (paylaşılabilir):" -ForegroundColor Cyan
        Write-Host ("  " + ((Get-Content $acik -Raw).Trim()))
        Write-Host ""
        Write-Host "  >> Sunucuya erişimi OLAN bir makinede şu komutu çalıştırın:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host ("    type `"$acik`" | ssh -i `"<orada çalışan anahtar>`" $Sunucu `"cat >> ~/.ssh/authorized_keys`"")
        Write-Host ""
        Write-Host "  (yalnız AÇIK anahtar eklenir; özel anahtar paylaşılmaz)" -ForegroundColor DarkGray
    }

    if ($YetkiBekle) {
        Write-Host ""
        Read-Host "  Yetkilendirmeyi yaptıktan sonra Enter'a basın"
        if (Test-SshAnahtar $ozel) {
            Mutlu "anahtar artık çalışıyor"
            $calisan = $ozel
        } else {
            Hata "hâlâ bağlanamıyor — yetkilendirme eksik"
        }
    }
}

if ($SadeceAnahtar) {
    Write-Host ""
    Write-Host ("=" * 74)
    Write-Host "  Anahtar adımı bitti."
    Write-Host "  NC_SSH_KEY = $env:NC_SSH_KEY"
    if ($calisan) { Write-Host "  DURUM      : çalışıyor (sunucuya bağlanabiliyor)" }
    else { Write-Host "  DURUM      : YETKİLENDİRME BEKLİYOR (yukarıdaki komut)" }
    Write-Host ("=" * 74)
    if ($calisan) { exit 0 } else { exit 1 }
}

# ── Yalnızca doğrulama modu ───────────────────────────────────────────
if ($SadeceKontrol) {
    $py0 = $PythonExe
    if (-not $py0 -or -not (Test-Path $py0)) {
        if (Test-Path ".venv\Scripts\python.exe") { $py0 = ".venv\Scripts\python.exe" }
        else { $py0 = "python" }
    }
    Write-Host ""
    Write-Host "  Doğrulama modu — kurulum yapılmayacak ($py0)"
    & $py0 (Join-Path $kok "tools\kurulum_kontrol.py")
    exit $LASTEXITCODE
}

# ══════════════════════════════════════════════════════════════════════
#  PYTHON
# ══════════════════════════════════════════════════════════════════════
Adim 1 "Python"
if (-not $PythonExe) {
    $pAdaylar = @()
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $pAdaylar += $cmd.Source }
    $pAdaylar += @(
        "E:\AppsTools\PYTHON\Python3.10\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Python310\python.exe",
        "C:\Program Files\Python310\python.exe"
    )
    foreach ($a in $pAdaylar) {
        if ($a -and (Test-Path $a)) { $PythonExe = $a; break }
    }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    Hata "Python bulunamadı"
    Write-Host "    Kur: https://www.python.org/downloads/release/python-31011/"
    Write-Host "    (kurulumda 'Add python.exe to PATH' işaretle)"
    exit 1
}
$surum = & $PythonExe -c "import sys;print('%d.%d.%d' % sys.version_info[:3])"
Mutlu "python : $PythonExe ($surum)"
if ([version]$surum -lt [version]"3.10") {
    Hata "Python >= 3.10 gerekli"
    exit 1
}

# ══════════════════════════════════════════════════════════════════════
#  SANAL ORTAM
# ══════════════════════════════════════════════════════════════════════
Adim 2 "Sanal ortam"
if ($VenvYok) {
    $py = $PythonExe
    Write-Host "  * .venv atlandı (sistem Python)"
} else {
    $venv = Join-Path $kok ".venv"
    if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
        Write-Host "  * oluşturuluyor: .venv"
        & $PythonExe -m venv $venv
        if ($LASTEXITCODE -ne 0) { Hata "venv kurulamadı"; exit 1 }
    } else {
        Write-Host "  * mevcut .venv kullanılacak"
    }
    $py = Join-Path $venv "Scripts\python.exe"
}
Mutlu "python : $py"

# ══════════════════════════════════════════════════════════════════════
#  PAKETLER
# ══════════════════════════════════════════════════════════════════════
Adim 3 "PyTorch (CUDA $Torch) — büyük indirme"
& $py -m pip install --upgrade pip --quiet
$torchIdx = "https://download.pytorch.org/whl/$Torch"
Write-Host "  * torch/torchvision kaynagi: $torchIdx"
& $py -m pip install torch torchvision --index-url $torchIdx
if ($LASTEXITCODE -ne 0) { Hata "PyTorch kurulamadı (ağ/disk?)"; exit 1 }

Adim 4 "Diğer bağımlılıklar"
& $py -m pip install -r (Join-Path $kok "requirements.txt")
if ($LASTEXITCODE -ne 0) { Uyari "bazı paketler kurulamadı — doğrulama gösterecek" }

# ══════════════════════════════════════════════════════════════════════
#  DOĞRULAMA
# ══════════════════════════════════════════════════════════════════════
Adim 5 "Doğrulama"
& $py (Join-Path $kok "tools\kurulum_kontrol.py")

Write-Host ""
Write-Host ("=" * 74)
Write-Host "  Kurulum bitti."
Write-Host "  Anahtar : $env:NC_SSH_KEY"
Write-Host "  Çalıştır: .\BASLAT-istemci.bat"
Write-Host "  Doğrula : $py tools\kurulum_kontrol.py --kaydet"
Write-Host ("=" * 74)
