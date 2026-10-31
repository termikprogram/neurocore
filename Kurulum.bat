@echo off
REM ===========================================================================
REM  NeuroCore - Setup (portable)
REM
REM  Works on any Windows machine: Python is located automatically
REM  (see tools\py_bul.bat). No machine-specific paths.
REM
REM  Usage:
REM      Kurulum.bat                normal setup  (creates .venv)
REM      Kurulum.bat --no-venv      install into the system Python
REM      set NC_PYTHON=C:\path\python.exe  -> explicit interpreter
REM
REM  The window stays open at the end so you can read the result.
REM ===========================================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title NeuroCore - Setup
cd /d "%~dp0"

echo.
echo   NeuroCore Setup
echo   ------------------------------------------------------------
echo.

REM --- Parse arguments -------------------------------------------------------
set "USE_VENV=1"
set "CHECK_ONLY=0"
for %%A in (%*) do (
    if /i "%%~A"=="--no-venv" set "USE_VENV=0"
    if /i "%%~A"=="--check" set "CHECK_ONLY=1"
    if /i "%%~A"=="--help" goto :help
    if /i "%%~A"=="-h" goto :help
)

REM --- 1) Find Python (portable) --------------------------------------------
echo   [1/4] Locating Python...
call "tools\py_bul.bat"
if not defined PYEXE goto :no_python

echo         Python : %PYVER%
if defined PYARGS (
    echo         via    : %PYEXE% %PYARGS%
) else (
    echo         via    : %PYEXE%
)
echo.

REM --- --check: yalnız tespit -------------------------------------------------
if "%CHECK_ONLY%"=="1" goto :check_only

REM --- 2) Virtual environment ------------------------------------------------
set "RUNPY=%PYEXE%"
set "RUNARGS=%PYARGS%"
if "%USE_VENV%"=="1" (
    if exist ".venv\Scripts\python.exe" (
        echo   [2/4] Using existing virtual environment: .venv
    ) else (
        echo   [2/4] Creating virtual environment ^(.venv^)...
        "%PYEXE%" %PYARGS% -m venv .venv
        if errorlevel 1 (
            echo   [WARN] Could not create .venv - falling back to the system Python.
            echo          You can also pass --no-venv to skip this step.
        )
    )
    if exist ".venv\Scripts\python.exe" (
        set "RUNPY=%CD%\.venv\Scripts\python.exe"
        set "RUNARGS="
    )
) else (
    echo   [2/4] Skipping virtual environment ^(--no-venv^)
)
echo.

REM --- 3) Packages -----------------------------------------------------------
echo   [3/4] Installing packages ^(this can take a few minutes^)...
echo.
"%RUNPY%" %RUNARGS% -m pip install --upgrade pip --disable-pip-version-check -q
if exist "requirements.txt" (
    "%RUNPY%" %RUNARGS% -m pip install -r requirements.txt --disable-pip-version-check
) else (
    echo   [NOTE] requirements.txt not found; installing core packages.
    "%RUNPY%" %RUNARGS% -m pip install fastapi uvicorn websockets psutil pywinauto comtypes Pillow pyyaml --disable-pip-version-check
)
echo.

echo   [4/4] Checking PyTorch ^(GPU support^)...
"%RUNPY%" %RUNARGS% -c "import torch; print('   torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo   [INFO] PyTorch is not installed. For an NVIDIA GPU install the CUDA build:
    echo.
    echo       "%RUNPY%" %RUNARGS% -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    echo.
    echo   The app also works on CPU ^(slower training^).
)
echo.

echo   ------------------------------------------------------------
echo   Verification
echo   ------------------------------------------------------------
"%RUNPY%" %RUNARGS% -c "import fastapi, uvicorn, websockets, psutil, PIL, yaml; print('   [OK] Core packages ready')" 2>nul
if errorlevel 1 (
    echo   [ERROR] Some packages could not be installed. Read the messages above.
    echo   Tip: retry with  "%RUNPY%" %RUNARGS% -m pip install -r requirements.txt
    pause
    exit /b 1
)
"%RUNPY%" %RUNARGS% -c "import pywinauto; print('   [OK] pywinauto (UI text reading) ready')" 2>nul
if errorlevel 1 echo   [INFO] pywinauto missing - UI text reading will be limited.
"%RUNPY%" %RUNARGS% -c "import torch; print('   [OK] torch ready')" 2>nul
if errorlevel 1 echo   [INFO] torch missing - training disabled, the agent still works.
echo.
echo   ------------------------------------------------------------
echo   Setup complete.
echo.
echo   Start the app with:  Run.bat
echo   (Python used: %RUNPY% %RUNARGS%)
echo.
pause
endlocal
exit /b 0

REM --- No Python found -------------------------------------------------------
:no_python
echo.
echo   [ERROR] Python 3.10+ was not found on this machine.
echo.
echo   Install it in one of these ways:
echo     1^) winget install --id Python.Python.3.12 -e
echo        ^(then reopen this window and run Kurulum.bat again^)
echo     2^) https://www.python.org/downloads/  - tick "Add python.exe to PATH"
echo.
echo   If Python is already installed in a custom folder, point to it directly:
echo       set NC_PYTHON=C:\path\to\python.exe
echo       Kurulum.bat
echo.
pause
endlocal
exit /b 1

:help
echo.
echo   Kurulum.bat  - NeuroCore setup
echo.
echo     Kurulum.bat               create .venv and install requirements
echo     Kurulum.bat --no-venv     install into the system Python
echo     set NC_PYTHON=...         use a specific python.exe
echo.
echo   Python 3.10+ is required. Detection order:
echo     NC_PYTHON ^> .venv ^> py launcher ^> PATH ^> common folders ^> registry
echo.
pause
endlocal
exit /b 0

REM --- Yalnız tespit modu (--check) -------------------------------------------
:check_only
echo   [OK] Python %PYVER% bulundu.
if defined PYARGS (
    echo        Komut : %PYEXE% %PYARGS%
) else (
    echo        Komut : %PYEXE%
)
"%PYEXE%" %PYARGS% -c "import sys;print('        Surum :', sys.version.split()[0], '/ 64-bit:', sys.maxsize > 2**32)"
echo.
echo   Kuruluma devam etmek icin:  Kurulum.bat
echo.
pause
endlocal
exit /b 0
