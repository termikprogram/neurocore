@echo off
REM ==========================================================================
REM  Surekli Egitim - myAi modelini kesintisiz egitir
REM ==========================================================================
REM  Kullanim:
REM    Surekli-Egitim.bat              -> 8 saat egit
REM    Surekli-Egitim.bat devamli      -> bilgisayar kapanana kadar egit
REM
REM  Egitim sirasinda GPU gerektiren is yapmayin (oyun, render) -
REM  kaynak gozcusu egitimi duraklatir ve turlar kisalir.
REM
REM  Durdurmak icin: Ctrl+C veya bu pencereyi kapatin.
REM  Ilerleme: .neurocore\egitim_gecmisi.json
REM ==========================================================================

setlocal
cd /d "%~dp0"

REM --- Python (tasinabilir: tools\py_bul.bat) ------------------------------
call "%~dp0tools\py_bul.bat"
if not defined PYEXE (
    echo HATA: Python 3.10+ bulunamadi.
    echo   Kur    : winget install --id Python.Python.3.12 -e
    echo   Ozel   : set NC_PYTHON=C:\yol\python.exe
    pause
    exit /b 1
)
set "PYTHON=%PYEXE%"
set "PYTHON_ARGS=%PYARGS%"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

echo ==========================================================================
echo   SUREKLI EGITIM - myAi
echo ==========================================================================
echo.

if /i "%~1"=="devamli" (
    echo Mod: SINIRSIZ - bilgisayar kapanana kadar
    "%PYTHON%" %PYTHON_ARGS% -u -X utf8 tools\egitim_bekcisi.py --devamli --parti-saat 2
) else (
    echo Mod: 8 SAAT
    "%PYTHON%" %PYTHON_ARGS% -u -X utf8 tools\egitim_bekcisi.py --toplam-saat 8 --parti-saat 2
)

echo.
echo Egitim tamamlandi. Ozet: _bekci_ozet.json
pause
