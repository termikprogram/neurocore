@echo off
REM ===========================================================================
REM  NeuroCore - Setup
REM
REM  Installs the required Python packages. Run once before the first start.
REM  The window stays open when finished so you can read the result.
REM ===========================================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title NeuroCore - Setup

cd /d "%~dp0"

echo.
echo   NeuroCore Setup
echo   ------------------------------------------------------------
echo.

REM --- Locate Python --------------------------------------------------------
set "PYEXE="
for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYEXE set "PYEXE=%%P"
)
if not defined PYEXE if exist "E:\AppsTools\PYTHON\Python3.10\python.exe" set "PYEXE=E:\AppsTools\PYTHON\Python3.10\python.exe"
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYEXE if exist "C:\Python312\python.exe" set "PYEXE=C:\Python312\python.exe"
if not defined PYEXE if exist "C:\Python311\python.exe" set "PYEXE=C:\Python311\python.exe"

if not defined PYEXE (
    echo   [ERROR] Python not found.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   and tick "Add python.exe to PATH" during setup.
    pause
    exit /b 1
)

echo   Python: !PYEXE!
"!PYEXE!" --version
echo.

echo   [1/3] Upgrading pip...
"!PYEXE!" -m pip install --upgrade pip --disable-pip-version-check -q
echo.

echo   [2/3] Installing requirements (this can take a few minutes)...
echo.
if exist "requirements.txt" (
    "!PYEXE!" -m pip install -r requirements.txt --disable-pip-version-check
) else (
    echo   [NOTE] requirements.txt not found; installing core packages.
    "!PYEXE!" -m pip install fastapi uvicorn websockets psutil pywinauto comtypes Pillow pyyaml --disable-pip-version-check
)
echo.

echo   [3/3] Checking PyTorch (GPU support)...
"!PYEXE!" -c "import torch; print('   torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo   [INFO] PyTorch is not installed. For an NVIDIA GPU install the CUDA build:
    echo.
    echo       "!PYEXE!" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    echo.
    echo   The app also works on CPU (slower training).
)
echo.

echo   ------------------------------------------------------------
echo   Verification
echo   ------------------------------------------------------------
"!PYEXE!" -c "import fastapi, uvicorn, websockets, psutil, PIL, yaml; print('   [OK] Core packages ready')" 2>nul
if errorlevel 1 (
    echo   [ERROR] Some packages could not be installed. Read the messages above.
    pause
    exit /b 1
)
"!PYEXE!" -c "import pywinauto; print('   [OK] pywinauto (UI text reading) ready')" 2>nul
if errorlevel 1 echo   [INFO] pywinauto missing - UI text reading will be limited.
"!PYEXE!" -c "import torch; print('   [OK] torch ready')" 2>nul
if errorlevel 1 echo   [INFO] torch missing - training disabled, the agent still works.
"!PYEXE!" -c "import pywinauto" >nul 2>&1
echo.
echo   Setup complete. Start the app with: Run.bat
echo.
pause
endlocal
exit /b 0
