@echo off
REM ===========================================================================
REM  NeuroCore - Create Desktop Shortcut
REM
REM  Adds a "NeuroCore" shortcut to the Desktop and the Start Menu.
REM ===========================================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title NeuroCore - Create Shortcut

cd /d "%~dp0"

echo.
echo   Creating NeuroCore shortcut
echo   ------------------------------------------------------------
echo.

set "TARGET=%~dp0Run.bat"
set "WORKDIR=%~dp0"

set "PS1=%TEMP%\nc_shortcut.ps1"
> "%PS1%" echo $ws = New-Object -ComObject WScript.Shell
>>"%PS1%" echo $targets = @($ws.SpecialFolders('Desktop'), [Environment]::GetFolderPath('Programs'))
>>"%PS1%" echo foreach ($dir in $targets) {
>>"%PS1%" echo   $lnk = Join-Path $dir 'NeuroCore.lnk'
>>"%PS1%" echo   $s = $ws.CreateShortcut($lnk)
>>"%PS1%" echo   $s.TargetPath = '%TARGET%'
>>"%PS1%" echo   $s.WorkingDirectory = '%WORKDIR%'
>>"%PS1%" echo   $s.Description = 'NeuroCore - local software agent'
>>"%PS1%" echo   $s.WindowStyle = 7
>>"%PS1%" echo   $s.Save()
>>"%PS1%" echo   Write-Host ('   [OK] ' + $lnk)
>>"%PS1%" echo }

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%"
del "%PS1%" >nul 2>&1

echo.
echo   Done. Double-click the "NeuroCore" icon on your Desktop to start.
echo.
echo   ------------------------------------------------------------
echo   You can also create a shortcut for a specific project folder.
echo.
set /p "PROJECT=   Project folder path (leave empty to skip): "

if not "!PROJECT!"=="" (
    if exist "!PROJECT!\" (
        for %%A in ("!PROJECT!") do set "PNAME=%%~nxA"
        set "PS2=%TEMP%\nc_shortcut2.ps1"
        > "%PS2%" echo $ws = New-Object -ComObject WScript.Shell
        >>"%PS2%" echo $lnk = Join-Path $ws.SpecialFolders('Desktop') ('NeuroCore - !PNAME!.lnk')
        >>"%PS2%" echo $s = $ws.CreateShortcut($lnk)
        >>"%PS2%" echo $s.TargetPath = '%TARGET%'
        >>"%PS2%" echo $s.Arguments = '--workspace "!PROJECT!"'
        >>"%PS2%" echo $s.WorkingDirectory = '%WORKDIR%'
        >>"%PS2%" echo $s.Description = 'NeuroCore - !PNAME!'
        >>"%PS2%" echo $s.WindowStyle = 7
        >>"%PS2%" echo $s.Save()
        >>"%PS2%" echo Write-Host ('   [OK] ' + $lnk)
        powershell -NoProfile -ExecutionPolicy Bypass -File "!PS2!"
        del "!PS2!" >nul 2>&1
        echo.
        echo   This shortcut opens NeuroCore directly with that project.
    ) else (
        echo   [ERROR] Folder not found: !PROJECT!
    )
)

echo.
pause
endlocal
exit /b 0
