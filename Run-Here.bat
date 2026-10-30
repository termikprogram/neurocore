@echo off
REM ===========================================================================
REM  NeuroCore - Open THIS Folder
REM
REM  Put a shortcut to this file inside any project folder. NeuroCore uses
REM  that folder as its workspace, and the agent only works inside it.
REM
REM  Example: copy a shortcut into  E:\PROJECTS\mygame  and double-click it.
REM ===========================================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title NeuroCore - %CD%

set "WORKSPACE=%CD%"

echo.
echo   NeuroCore
echo   ------------------------------------------------------------
echo   Workspace : %WORKSPACE%
echo.

REM --- Locate Python (portable: tools\py_bul.bat) ---------------------------
REM  Tespit: NC_PYTHON > .venv > py launcher > PATH > ortak klasorler > registry
call "%~dp0tools\py_bul.bat"
if not defined PYEXE (
    echo   [ERROR] Python 3.10+ was not found on this machine.
    echo.
    echo   Install it:   winget install --id Python.Python.3.12 -e
    echo   or point to:  set NC_PYTHON=C:\path\to\python.exe
    echo.
    pause
    exit /b 1
)
if defined PYARGS (
    echo   Python : %PYVER%  ^(%PYEXE% %PYARGS%^)
) else (
    echo   Python : %PYVER%  ^(%PYEXE%^)
)

set "PYWEXE=!PYEXE:python.exe=pythonw.exe!"
if not exist "!PYWEXE!" set "PYWEXE="

echo   Starting...
REM  Independent process: survives this window closing, shows no console.
if defined PYWEXE (
    set "NCRUN=!PYWEXE!"
) else (
    set "NCRUN=!PYEXE!"
)
powershell -NoProfile -Command "$p = Start-Process -FilePath '!NCRUN!' -ArgumentList '-X utf8 -m neurocore.app --workspace \"%WORKSPACE%\" %*' -WorkingDirectory '%WORKSPACE%' -WindowStyle Hidden -PassThru; Write-Host ('   Started process ID: ' + $p.Id)"

set "UP="
for /L %%I in (1,1,20) do (
    if not defined UP (
        powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8791/api/state' -TimeoutSec 3 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
        if not errorlevel 1 set "UP=1"
        if not defined UP timeout /t 2 /nobreak >nul 2>&1
    )
)

echo.
if defined UP (
    echo   [OK] NeuroCore is running for: %WORKSPACE%
) else (
    echo   [WARNING] The server is not responding yet. The first start can take a minute.
    echo   Log: %WORKSPACE%\.neurocore\logs\neurocore.log
)
echo.
endlocal
exit /b 0
