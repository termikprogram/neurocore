@echo off
REM ===========================================================================
REM  NeuroCore - Debug Mode
REM
REM  Keeps the console visible and prints the full log.
REM  If the app does not start, THE REASON SHOWS UP HERE.
REM ===========================================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title NeuroCore - Debug

cd /d "%~dp0"

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

echo.
echo   NeuroCore - Debug Mode
echo   ------------------------------------------------------------
echo   Python: !PYEXE!
echo   Press Ctrl+C to stop.
echo.

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

"!PYEXE!" -X utf8 -m neurocore.app --log-level debug --no-window %*

echo.
echo   The application has stopped.
pause
endlocal
exit /b 0
