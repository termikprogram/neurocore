@echo off
REM ===========================================================================
REM  NeuroCore Desktop - Launcher
REM
REM  Double-click this file. The app starts WITHOUT a console window
REM  (pythonw.exe is used).
REM
REM  If something goes wrong, check the log:
REM      .neurocore\logs\neurocore.log
REM
REM  To see the console output, use: Run-Debug.bat
REM ===========================================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title NeuroCore

cd /d "%~dp0"

echo.
echo   NeuroCore Desktop
echo   ------------------------------------------------------------
echo.

REM --- Locate the Python executable -----------------------------------------
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
    echo.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   and tick "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

REM --- Check required packages ---------------------------------------------
"!PYEXE!" -c "import fastapi, uvicorn, websockets, psutil" >nul 2>&1
if errorlevel 1 (
    echo   [MISSING] Some required packages are not installed.
    echo.
    set /p "ANSWER=   Run Kurulum.bat now? (Y/N): "
    if /i "!ANSWER!"=="Y" (
        call "%~dp0Kurulum.bat"
    ) else (
        echo.
        echo   Setup skipped. The app cannot start without its packages.
        pause
        exit /b 1
    )
)

REM --- Resolve pythonw.exe (console-free launcher) --------------------------
set "PYWEXE=!PYEXE:python.exe=pythonw.exe!"
if not exist "!PYWEXE!" set "PYWEXE="

echo   Python      : !PYEXE!
if defined PYWEXE (
    echo   Launcher    : !PYWEXE!  (no console window^)
) else (
    echo   Launcher    : !PYEXE!  (console will be minimised^)
)
echo   Workspace   : %CD%
echo   Interface   : http://127.0.0.1:8791/
echo.

REM --- Start ----------------------------------------------------------------
REM  The app is launched through PowerShell's Start-Process so it becomes a
REM  fully INDEPENDENT process: it survives this console window closing and it
REM  shows no console window of its own. A plain "start" keeps the child in
REM  the same console/job, so closing this window (or a launcher timeout)
REM  would take the app down with it.
set "NCARGS=-X utf8 -m neurocore.app %*"
if defined PYWEXE (
    set "NCRUN=!PYWEXE!"
) else (
    set "NCRUN=!PYEXE!"
)

powershell -NoProfile -Command ^
  "$p = Start-Process -FilePath '!NCRUN!' -ArgumentList '!NCARGS!' -WorkingDirectory '%CD%' -WindowStyle Hidden -PassThru;" ^
  "Write-Host ('   Started process ID: ' + $p.Id)"

echo.
echo   Starting... the window opens in a few seconds.
echo   To quit, close the NeuroCore window.
echo.
echo   Log: .neurocore\logs\neurocore.log
echo.

REM --- Wait briefly, then check whether the server answered ----------------
set "UP="
for /L %%I in (1,1,25) do (
    if not defined UP (
        powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8791/api/state' -TimeoutSec 3 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
        if not errorlevel 1 set "UP=1"
        if not defined UP timeout /t 2 /nobreak >nul 2>&1
    )
)

echo.
if defined UP (
    echo   [OK] NeuroCore is running.
) else (
    echo   [WARNING] The server is not responding yet.
    echo   The first start can take up to a minute.
    echo.
    echo   Still not working? Check the log file:
    echo       .neurocore\logs\neurocore.log
    echo   Or run Run-Debug.bat to see the error directly.
    echo.
    pause
)

endlocal
exit /b 0
