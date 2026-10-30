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
    # ⚠ `auto`: GPU'yu algılar ve **doğru PyTorch derlemesini** seçer.
    [ValidateSet("auto", "cu124", "cu121", "cu118", "cpu")]
    [string]$Torch = "auto",
    [string]$Sunucu = "",        # boşsa NC_SERVER / .neurocore/peer.json
    [string]$AnahtarDizin = "",
    [string]$AnahtarAdi = "",
    [switch]$VenvYok,
    [switch]$SadeceKontrol,
    [switch]$SadeceAnahtar,
    [switch]$SadeceGpu,           # yalnız GPU algıla ve PyTorch seçimini yaz
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

# ── GPU'ya göre PyTorch derlemesi seç ────────────────────────────────
function Get-TorchSecim {
    param([string]$Istenen)
    $bilgi = @{ secim = "cpu"; surucu = ""; gpu = ""; neden = "" }

    # NVIDIA sürücüsü + kart adı
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        try {
            $surucu = (& $smi.Source --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
            $ad = (& $smi.Source --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
            if ($surucu) { $bilgi.surucu = $surucu.Trim() }
            if ($ad) { $bilgi.gpu = $ad.Trim() }
        } catch { }
    }
    # AMD / Intel
    if (-not $bilgi.gpu) {
        try {
            $vc = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                  Where-Object { $_.Name -notmatch 'Basic|Remote|Virtual' } |
                  Select-Object -First 1
            if ($vc) { $bilgi.gpu = $vc.Name }
        } catch { }
    }

    if ($Istenen -ne "auto") {
        $bilgi.secim = $Istenen
        $bilgi.neden = "kullanıcı seçimi"
        return $bilgi
    }

    if ($bilgi.surucu -match '^(\d+)') {
        $major = [int]$Matches[1]
        # ⚠ Eşleme: sürücü major → en yeni uyumlu CUDA derlemesi
        if ($major -ge 550)      { $bilgi.secim = "cu124" }
        elseif ($major -ge 525)  { $bilgi.secim = "cu121" }
        else                     { $bilgi.secim = "cu118" }
        $bilgi.neden = "NVIDIA sürücü $major → $($bilgi.secim)"
    }
    elseif ($bilgi.gpu -match 'AMD|Radeon') {
        # Windows'ta ROCm yok → CPU derlemesi (ajan çalışır, model yavaş)
        $bilgi.secim = "cpu"
        $bilgi.neden = "AMD GPU (Windows) — ROCm yok, CPU derlemesi"
    }
    elseif ($bilgi.gpu -match 'Intel') {
        $bilgi.secim = "cpu"
        $bilgi.neden = "Intel GPU — XPU desteği yok, CPU derlemesi"
    }
    else {
        $bilgi.secim = "cpu"
        $bilgi.neden = "GPU bulunamadı — CPU derlemesi"
    }
    return $bilgi
}

Adim "2b" "GPU ALGILAMA ve PyTorch SEÇİMİ"
if ($SadeceGpu) {
    $tb = Get-TorchSecim -Istenen $Torch
    Write-Host ""
    Write-Host ("=" * 74)
    Write-Host "  GPU        : $($tb.gpu)"
    Write-Host "  sürücü     : $($tb.surucu)"
    Write-Host "  PyTorch    : $($tb.secim)"
    Write-Host "  gerekçe    : $($tb.neden)"
    Write-Host ("=" * 74)
    exit 0
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
    # ⚠ Taşınabilir tespit: makineye özel yol YOK.
    # Sıra: NC_PYTHON > .venv > py launcher > PATH > ortak klasörler > registry
    $pAdaylar = @()
    if ($env:NC_PYTHON -and (Test-Path $env:NC_PYTHON)) { $pAdaylar += $env:NC_PYTHON }
    $venvPy = Join-Path (Get-Location) ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) { $pAdaylar += $venvPy }

    # py launcher (resmi Windows başlatıcısı) — en güvenilir yol
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($v in @("-3.13", "-3.12", "-3.11", "-3.10", "-3")) {
            try {
                $cikti = & py $v -c "import sys;print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $cikti) { $pAdaylar += "$cikti".Trim() }
            } catch { }
        }
    }

    foreach ($ad in @("python", "python3")) {
        $cmd = Get-Command $ad -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source) { $pAdaylar += $cmd.Source }
    }

    foreach ($v in @("313", "312", "311", "310")) {
        $pAdaylar += @(
            "$env:LOCALAPPDATA\Programs\Python\Python$v\python.exe",
            "$env:ProgramFiles\Python$v\python.exe",
            "C:\Python$v\python.exe"
        )
    }

    # Registry (PythonCore InstallPath)
    foreach ($kok in @("HKCU:\Software\Python\PythonCore",
                       "HKLM:\Software\Python\PythonCore",
                       "HKLM:\Software\WOW6432Node\Python\PythonCore")) {
        try {
            Get-ChildItem $kok -ErrorAction Stop | ForEach-Object {
                $ip = (Get-ItemProperty "$($_.PSPath)\InstallPath" -ErrorAction SilentlyContinue)
                if ($ip -and $ip.ExecutablePath) { $pAdaylar += $ip.ExecutablePath }
            }
        } catch { }
    }

    foreach ($a in ($pAdaylar | Select-Object -Unique)) {
        if (-not $a -or -not (Test-Path $a)) { continue }
        try {
            $v = & $a -c "import sys;print('%d.%d.%d' % sys.version_info[:3])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $v -and ([version]$v -ge [version]"3.10")) {
                $PythonExe = $a
                break
            }
        } catch { }
    }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    Hata "Python 3.10+ bulunamadı"
    Write-Host "    Kur    : winget install --id Python.Python.3.12 -e"
    Write-Host "    veya   : https://www.python.org/downloads/  ('Add python.exe to PATH')"
    Write-Host "    özel yol: `$env:NC_PYTHON = 'C:\yol\python.exe'"
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

$torchBilgi = Get-TorchSecim -Istenen $Torch
if ($torchBilgi.gpu)     { Mutlu "GPU       : $($torchBilgi.gpu)" }
if ($torchBilgi.surucu)  { Mutlu "sürücü    : $($torchBilgi.surucu)" }
Mutlu "PyTorch   : $($torchBilgi.secim)  ($($torchBilgi.neden))"
if ($torchBilgi.secim -eq "cpu") {
    Uyari "GPU hızlandırma YOK — ajan çalışır, yerel model CPU'da yavaş olur"
}
$Torch = $torchBilgi.secim

Adim 3 "PyTorch ($Torch) — büyük indirme"
& $py -m pip install --upgrade pip --quiet
# ⚠ CPU derlemesi ayrı dizinde; CUDA derlemeleri cuXXX altında
$torchIdx = if ($Torch -eq "cpu") { "https://download.pytorch.org/whl/cpu" } `
            else { "https://download.pytorch.org/whl/$Torch" }
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
