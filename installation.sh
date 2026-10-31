#!/usr/bin/env bash
# =============================================================================
#  NeuroCore — Linux/macOS kurulum (İngilizce adlı başlatıcı)
#  Kurulum / same installer as kur_istemci.sh — tüm argümanları ona iletir.
#
#     bash installation.sh                 # GPU otomatik algılanır
#     bash installation.sh --torch cpu     # yalnız CPU
#     bash installation.sh --systemd       # systemd kullanıcı servisi
#     bash installation.sh --desktop       # masaüstü kısayolu
#     bash installation.sh --sadece-gpu    # yalnız GPU tespiti
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f kur_istemci.sh ]]; then
  echo "[x] kur_istemci.sh bulunamadı — depoyu yeniden indirin." >&2
  exit 1
fi

exec bash kur_istemci.sh "$@"
