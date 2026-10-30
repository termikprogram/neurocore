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
