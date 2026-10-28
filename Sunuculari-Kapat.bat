@echo off
REM ===========================================================================
REM  NeuroCore - Stop Running Instances
REM
REM  Closes every running NeuroCore instance cleanly.
REM  It first asks the API to shut down (shell sessions close properly), then
REM  cleans up leftovers that no longer respond.
REM ===========================================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title NeuroCore - Stop

echo.
echo   Stopping NeuroCore instances
echo   ------------------------------------------------------------
echo.

set "FOUND=0"

for /L %%P in (8791,1,8810) do (
    powershell -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:%%P/api/state' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 (
        echo   [FOUND] http://127.0.0.1:%%P/ - requesting shutdown...
        powershell -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:%%P/api/shutdown' -Method Post -ContentType 'application/json' -Body '{}' -TimeoutSec 5 | Out-Null } catch { }" >nul 2>&1
        set "FOUND=1"
    )
)

if "!FOUND!"=="1" (
    echo.
    echo   Shutdown requested. Waiting...
    timeout /t 4 /nobreak >nul 2>&1
)

echo.
echo   Checking for leftovers...

set "LEFT=0"

powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*NeuroCore*' -or $_.CommandLine -like '*ui-profile*' }; if ($p) { $p | ForEach-Object { Write-Host ('   [KILL] msedge (NeuroCore window) PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 set "LEFT=1"

powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { ($_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe') -and $_.CommandLine -like '*neurocore.app*' }; if ($p) { $p | ForEach-Object { Write-Host ('   [KILL] python (NeuroCore server) PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 set "LEFT=1"

REM Stale port lock: a pythonw process that no longer answers
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8791 -State Listen -ErrorAction SilentlyContinue; if ($c) { $c | ForEach-Object { $pr = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue; if ($pr -and ($pr.ProcessName -eq 'python' -or $pr.ProcessName -eq 'pythonw')) { Write-Host ('   [KILL] stale port 8791 held by PID ' + $pr.Id); Stop-Process -Id $pr.Id -Force -ErrorAction SilentlyContinue } }; exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 set "LEFT=1"

timeout /t 2 /nobreak >nul 2>&1

echo.
echo   ------------------------------------------------------------
if "!FOUND!!LEFT!"=="00" (
    echo   No running NeuroCore instance was found.
) else (
    echo   All NeuroCore instances have been stopped.
)
echo   ------------------------------------------------------------
echo.
timeout /t 3 /nobreak >nul 2>&1
endlocal
exit /b 0
