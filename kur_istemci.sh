#!/usr/bin/env bash
# =============================================================================
#  NeuroCore — Linux / macOS kurulum (GPU'ya göre PyTorch seçimi)
# =============================================================================
#  Kullanım:
#     bash kur_istemci.sh            # GPU otomatik algılanır
#     bash kur_istemci.sh --torch rocm6.2
#     bash kur_istemci.sh --torch cpu
#     bash kur_istemci.sh --sadece-gpu     # yalnız algıla, kurma
#     bash kur_istemci.sh --systemd        # systemd kullanıcı servisi kur
#     bash kur_istemci.sh --desktop        # .desktop kısayolu oluştur
#
#  Windows'ta `kur_istemci.ps1` kullanılır; bu betik POSIX sistemler içindir.
# =============================================================================
set -euo pipefail

TORCH="auto"
SADECE_GPU=0
VENV=".venv"
SYSTEMD=0
DESKTOP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --torch) TORCH="$2"; shift 2 ;;
    --sadece-gpu) SADECE_GPU=1; shift ;;
    --venv) VENV="$2"; shift 2 ;;
    --systemd) SYSTEMD=1; shift ;;
    --desktop) DESKTOP=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20
      exit 0 ;;
    *) echo "bilinmeyen argüman: $1" >&2; exit 2 ;;
  esac
done

baslik() { printf "\n\033[1m── %s\033[0m\n" "$1"; }
ok()     { printf "  \033[32m[ok]\033[0m %s\n" "$1"; }
uyari()  { printf "  \033[33m[!]\033[0m  %s\n" "$1"; }
hata()   { printf "  \033[31m[x]\033[0m %s\n" "$1" >&2; }

echo "==========================================================================="
echo "  NEUROCORE — KURULUM (Linux/macOS)"
echo "==========================================================================="
echo "  klasör : $(pwd)"
echo "  sistem : $(uname -srm)"
echo "  torch  : $TORCH"

# ─── GPU algılama ────────────────────────────────────────────────────────────
SECIM="cpu"; GPU_AD=""; SURUCU=""; NEDEN=""

if [[ "$TORCH" != "auto" ]]; then
  SECIM="$TORCH"; NEDEN="kullanıcı seçimi"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    SURUCU="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    GPU_AD="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r')"
    MAJOR="${SURUCU%%.*}"
    if   [[ "${MAJOR:-0}" -ge 550 ]]; then SECIM="cu124"
    elif [[ "${MAJOR:-0}" -ge 525 ]]; then SECIM="cu121"
    else SECIM="cu118"; fi
    NEDEN="NVIDIA sürücü ${MAJOR:-?} → $SECIM"
  elif command -v rocminfo >/dev/null 2>&1 || [[ -d /opt/rocm ]]; then
    # AMD: ROCm PyTorch `torch.cuda.*` üzerinden görünür, kod değişmez
    SECIM="rocm6.2"
    GPU_AD="$( (rocm-smi --showproductname 2>/dev/null | grep -m1 'Card series' || echo 'AMD (ROCm)') | tr -d '\r')"
    NEDEN="AMD ROCm algılandı → rocm6.2"
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    SECIM="cpu"; GPU_AD="Apple Silicon / Intel Mac"
    NEDEN="macOS — MPS dalı yok, CPU derlemesi"
  else
    SECIM="cpu"; NEDEN="GPU bulunamadı — CPU derlemesi"
  fi
fi

baslik "GPU ALGILAMA"
[[ -n "$GPU_AD" ]]  && ok "GPU     : $GPU_AD"
[[ -n "$SURUCU" ]]  && ok "sürücü  : $SURUCU"
ok "PyTorch : $SECIM"
ok "gerekçe : $NEDEN"
[[ "$SECIM" == "cpu" ]] && uyari "GPU hızlandırma YOK — ajan çalışır, yerel model CPU'da yavaş olur"

if [[ "$SADECE_GPU" == "1" ]]; then
  echo; exit 0
fi

# ─── Python ──────────────────────────────────────────────────────────────────
baslik "PYTHON"
PY=""
for aday in python3.12 python3.11 python3.10 python3; do
  if command -v "$aday" >/dev/null 2>&1; then PY="$(command -v "$aday")"; break; fi
done
[[ -z "$PY" ]] && { hata "python3 bulunamadı (3.10+ gerekli)"; exit 1; }
ok "python : $PY ($("$PY" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])'))"

# ─── Sanal ortam ─────────────────────────────────────────────────────────────
baslik "SANAL ORTAM"
[[ -d "$VENV" ]] || "$PY" -m venv "$VENV"
ok "venv    : $VENV/bin/python"

# ─── PyTorch ─────────────────────────────────────────────────────────────────
baslik "PYTORCH ($SECIM) — büyük indirme"
PIP="$VENV/bin/python -m pip"
$PIP install --upgrade pip --quiet
case "$SECIM" in
  cpu)      URL="https://download.pytorch.org/whl/cpu" ;;
  rocm*)    URL="https://download.pytorch.org/whl/$SECIM" ;;
  cu*)      URL="https://download.pytorch.org/whl/$SECIM" ;;
  *)        URL="https://download.pytorch.org/whl/cpu" ;;
esac
$PIP install torch torchvision --index-url "$URL"
ok "torch   : $("$VENV/bin/python" -c 'import torch;print(torch.__version__)')"
"$VENV/bin/python" - <<'PYEOK'
import torch, sys
hip = getattr(torch.version, "hip", None)
cuda = torch.version.cuda
if hip:
    print(f"  [ok] backend: ROCm {hip} (AMD)")
elif cuda:
    print(f"  [ok] backend: CUDA {cuda} (NVIDIA) · cihaz: {torch.cuda.get_device_name(0)}")
else:
    print("  [ok] backend: CPU")
try:
    major, _ = torch.cuda.get_device_capability(0)
    bf16 = "bf16 (donanım)" if major >= 8 else "fp16 (bf16 yok)"
    print(f"  [ok] AMP dtype: {bf16} · sm_{major}")
except Exception:
    print("  [ok] AMP dtype: fp32 (GPU yok)")
PYEOK

# ─── Bağımlılıklar (platforma göre) ─────────────────────────────────────────
baslik "BAĞIMLILIKLAR"
if [[ -f requirements-base.txt ]]; then
  $PIP install -r requirements-base.txt --quiet && ok "requirements-base.txt kuruldu"
else
  $PIP install -r requirements.txt --quiet && ok "requirements.txt kuruldu (eski dosya)"
fi
if [[ "$SECIM" == rocm* && -f requirements-rocm.txt ]]; then
  $PIP install -r requirements-rocm.txt --quiet || uyari "amdsmi kurulamadı (ROCm sürücüsü yok mu?)"
  ok "requirements-rocm.txt (amdsmi)"
fi
# Windows'a özel paketler (pywinauto/comtypes) Linux'ta KURULMAZ
[[ -f requirements-windows.txt ]] && ok "requirements-windows.txt atlandı (Windows'a özel)"

# ─── Doğrulama ──────────────────────────────────────────────────────────────
baslik "PLATFORM"
"$VENV/bin/python" -m neurocore.platform || uyari "platform bilgisi alınamadı"

baslik "DOĞRULAMA"
"$VENV/bin/python" tools/kurulum_kontrol.py || uyari "kurulum kontrolü uyarı verdi"
"$VENV/bin/python" tools/kabiliyet_olc.py || uyari "kabiliyet ölçümü tam geçmedi"

# ─── systemd kullanıcı servisi ──────────────────────────────────────────────
if [[ "$SYSTEMD" == "1" ]]; then
  baslik "SYSTEMD (kullanıcı servisi)"
  BIRIM_DIR="$HOME/.config/systemd/user"
  mkdir -p "$BIRIM_DIR"
  cat > "$BIRIM_DIR/neurocore.service" <<EOF
[Unit]
Description=NeuroCore local agent
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/$VENV/bin/python -m neurocore.app --no-window --port 8791 --autonomy yolo
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONUTF8=1

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable --now neurocore.service 2>/dev/null \
    && ok "servis: neurocore.service (systemctl --user status neurocore)" \
    || uyari "systemd yok/etkin değil — servis dosyası yazıldı: $BIRIM_DIR/neurocore.service"
fi

# ─── masaüstü kısayolu ──────────────────────────────────────────────────────
if [[ "$DESKTOP" == "1" ]]; then
  baslik "MASAÜSTÜ KISAYOLU"
  UYG="$HOME/.local/share/applications"
  mkdir -p "$UYG"
  cat > "$UYG/neurocore.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=NeuroCore
Comment=Local AI software agent
Exec=$(pwd)/$VENV/bin/python -m neurocore.app --port 8791
Path=$(pwd)
Icon=utilities-terminal
Terminal=false
Categories=Development;Utility;
EOF
  ok ".desktop: $UYG/neurocore.desktop"
fi

echo
echo "==========================================================================="
echo "  HAZIR — çalıştır:  $VENV/bin/python -m neurocore.app"
echo "  (Windows'ta: .\\BASLAT-istemci.bat)"
echo "==========================================================================="
