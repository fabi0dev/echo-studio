@echo off
REM Echo Studio - one-time setup for Windows (double-click me).
REM Optional backend: setup.bat auto|cu118|cu121|cu124|directml|cpu
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Torch "%~1"
echo.
pause
