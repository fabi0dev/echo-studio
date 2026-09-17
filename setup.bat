@echo off
REM Echo Studio - one-time setup for Windows (double-click me).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
echo.
pause
