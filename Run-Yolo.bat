@echo off
REM ===========================================================================
REM  NeuroCore - Free Mode (use with care)
REM
REM  Does NOT ask for approval on write, execute or destructive actions.
REM  Only the sandbox limits apply (paths outside the project, destructive
REM  command patterns). Useful for long tasks in projects you trust.
REM ===========================================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title NeuroCore - Free Mode

cd /d "%~dp0"

echo.
echo   ------------------------------------------------------------
echo    WARNING: FREE MODE
echo   ------------------------------------------------------------
echo    The agent will NOT ask for approval for write, execute
echo    or destructive actions. Only sandbox limits apply.
echo.
set /p "ANSWER=   Continue? (Y/N): "
if /i not "!ANSWER!"=="Y" (
    echo   Cancelled.
    timeout /t 2 >nul 2>&1
    exit /b 0
)

set "PYEXE="
for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYEXE set "PYEXE=%%P"
)
if not defined PYEXE if exist "E:\AppsTools\PYTHON\Python3.10\python.exe" set "PYEXE=E:\AppsTools\PYTHON\Python3.10\python.exe"
if not defined PYEXE if exist "C:\Python312\python.exe" set "PYEXE=C:\Python312\python.exe"
if not defined PYEXE if exist "C:\Python311\python.exe" set "PYEXE=C:\Python311\python.exe"

if not defined PYEXE (
    echo   [ERROR] Python not found.
    pause
    exit /b 1
)

set "PYWEXE=!PYEXE:python.exe=pythonw.exe!"
if not exist "!PYWEXE!" set "PYWEXE="

REM  Independent process: survives this window closing, shows no console.
if defined PYWEXE (
    set "NCRUN=!PYWEXE!"
) else (
    set "NCRUN=!PYEXE!"
)
powershell -NoProfile -Command "$p = Start-Process -FilePath '!NCRUN!' -ArgumentList '-X utf8 -m neurocore.app --autonomy yolo %*' -WorkingDirectory '%CD%' -WindowStyle Hidden -PassThru; Write-Host ('   Started process ID: ' + $p.Id)"

echo.
echo   Started in free mode.
timeout /t 4 >nul 2>&1
endlocal
exit /b 0
