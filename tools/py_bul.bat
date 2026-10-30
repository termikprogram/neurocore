@echo off
REM ===========================================================================
REM  NeuroCore - portable Python finder
REM
REM  Usage (from another .bat):
REM      call "%~dp0tools\py_bul.bat"
REM      if not defined PYEXE goto :no_python
REM      "%PYEXE%" %PYARGS% script.py
REM
REM  Sets:
REM      PYEXE   - interpreter (absolute path, or "py" launcher)
REM      PYARGS  - extra args for the launcher (e.g. "-3.12"), empty otherwise
REM      PYVER   - reported version string
REM
REM  Detection order (no machine-specific paths):
REM      1. %NC_PYTHON%            explicit override (full path to python.exe)
REM      2. .venv\Scripts\python.exe (project virtualenv)
REM      3. py -3.13 ... py -3.10  (official Windows launcher)
REM      4. py -3
REM      5. python / python3 on PATH
REM      6. %LOCALAPPDATA%\Programs\Python\PythonXY\python.exe
REM      7. %ProgramFiles%\PythonXY\python.exe  +  C:\PythonXY\python.exe
REM      8. HKCU/HKLM PythonCore registry (InstallPath)
REM  Every candidate must be >= 3.10.
REM ===========================================================================

setlocal EnableExtensions
set "PYEXE="
set "PYARGS="
set "PYVER="

REM --- 1) explicit override --------------------------------------------------
if defined NC_PYTHON (
    if exist "%NC_PYTHON%" (
        set "PYEXE=%NC_PYTHON%"
        goto :check
    )
    echo   [WARN] NC_PYTHON is set but not found: %NC_PYTHON%
)

REM --- 2) project virtualenv -------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    set "PYEXE=%CD%\.venv\Scripts\python.exe"
    goto :check
)

REM --- 3) py launcher, newest first -----------------------------------------
for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>&1
        if not errorlevel 1 (
            set "PYEXE=py"
            set "PYARGS=-%%V"
        )
    )
)

REM --- 4) py -3 --------------------------------------------------------------
if not defined PYEXE (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PYEXE=py"
        set "PYARGS=-3"
    )
)

REM --- 5) PATH ---------------------------------------------------------------
if not defined PYEXE (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYEXE set "PYEXE=%%P"
    )
)
if not defined PYEXE (
    for /f "delims=" %%P in ('where python3 2^>nul') do (
        if not defined PYEXE set "PYEXE=%%P"
    )
)

REM --- 6) per-user / per-machine install dirs --------------------------------
if not defined PYEXE (
    for %%V in (313 312 311 310) do (
        if not defined PYEXE (
            if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
                set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
            )
        )
    )
)
if not defined PYEXE (
    for %%V in (313 312 311 310) do (
        if not defined PYEXE (
            if exist "%ProgramFiles%\Python%%V\python.exe" (
                set "PYEXE=%ProgramFiles%\Python%%V\python.exe"
            )
        )
    )
)
if not defined PYEXE (
    for %%V in (313 312 311 310) do (
        if not defined PYEXE (
            if exist "C:\Python%%V\python.exe" set "PYEXE=C:\Python%%V\python.exe"
        )
    )
)

REM --- 7) registry (PythonCore InstallPath) ----------------------------------
if not defined PYEXE (
    for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore" /s /v ExecutablePath 2^>nul ^| findstr /i "python.exe"') do (
        if not defined PYEXE set "PYEXE=%%B"
    )
)
if not defined PYEXE (
    for /f "tokens=2,*" %%A in ('reg query "HKLM\Software\Python\PythonCore" /s /v ExecutablePath 2^>nul ^| findstr /i "python.exe"') do (
        if not defined PYEXE set "PYEXE=%%B"
    )
)

:check
if not defined PYEXE goto :notfound

REM --- version gate: >= 3.10 -------------------------------------------------
REM ⚠ `for /f` içinde tırnak kaçışı güvenilir değil; geçici dosya + errorlevel.
set "PYVER="
set "_ncver=%TEMP%\nc_pyver_%RANDOM%.txt"
"%PYEXE%" %PYARGS% -c "import sys;sys.stdout.write(sys.version.split()[0])" >"%_ncver%" 2>nul
if errorlevel 1 goto :verfail
set /p PYVER=<"%_ncver%" 2>nul
del "%_ncver%" >nul 2>&1
if not defined PYVER goto :notfound

"%PYEXE%" %PYARGS% -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if errorlevel 1 goto :oldpython

endlocal & set "PYEXE=%PYEXE%" & set "PYARGS=%PYARGS%" & set "PYVER=%PYVER%"
exit /b 0

:verfail
del "%_ncver%" >nul 2>&1
goto :notfound

:oldpython
echo   [WARN] Found Python %PYVER% at %PYEXE% - NeuroCore needs 3.10 or newer.
set "PYEXE="
set "PYARGS="
goto :notfound

:notfound
endlocal & set "PYEXE=" & set "PYARGS=" & set "PYVER="
exit /b 1
