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
    pause
    exit /b 1
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
