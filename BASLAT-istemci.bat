@echo off

chcp 65001 >nul
REM ============================================================
REM  NeuroCore - Istemci Makine Baslatici
REM ============================================================
REM  Ortak havuzu (komsu dugumlerle is paylasimi) ACIK baslatir.
REM  Kapatmak icin: NC_PEER=0 yap veya bu dosyada asagiyi degistir.
REM ============================================================
setlocal
cd /d "%~dp0"

REM --- Python secimi: once .venv, yoksa sistem ---
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY set "PY=python"

REM --- Ortak havuz acik (1) / kapali (0) ---
set "NC_PEER=1"

REM --- SSH anahtari (havuz icin). Farkli yoldaysa degistir. ---
REM ? Anahtar yolu ve sunucu: NC_SSH_KEY / NC_SERVER veya .neurocore\peer.json
REM    (kur_istemci.ps1 bunlari otomatik ayarlar)

echo.
echo   NeuroCore baslatiliyor...
echo   Python : %PY%
echo   Havuz  : NC_PEER=%NC_PEER%
echo.

"%PY%" -m neurocore.app %*

if errorlevel 1 (
  echo.
  echo   [!] Uygulama hata verdi. Kurulum kontrolu:
  echo       %PY% tools\kurulum_kontrol.py
  echo.
  pause
)
endlocal
