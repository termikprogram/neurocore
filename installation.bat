@echo off
REM ===========================================================================
REM  NeuroCore - Setup (English launcher)  /  Kurulum (Ingilizce adli baslatici)
REM
REM  Does exactly what Kurulum.bat does; it forwards every argument to it:
REM      installation.bat                 normal setup (creates .venv)
REM      installation.bat --no-venv       install into the system Python
REM      installation.bat --no-torch      skip torch/torchvision/torchaudio
REM      installation.bat --check         detect Python only (no install)
REM      installation.bat --help          show the detection order
REM
REM  Kurulum.bat ile ayni isi yapar; tum argumanlari ona iletir.
REM  Python is located automatically - no machine-specific paths.
REM ===========================================================================

setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

if not exist "%~dp0Kurulum.bat" (
    echo.
    echo   [ERROR] Kurulum.bat not found next to this file.
    echo   Download the repository again or restore Kurulum.bat.
    echo   ^(Bu dosyanin yaninda Kurulum.bat yok.^)
    echo.
    pause
    endlocal
    exit /b 1
)

call "%~dp0Kurulum.bat" %*
set "NC_KOD=%ERRORLEVEL%"
endlocal & exit /b %NC_KOD%
