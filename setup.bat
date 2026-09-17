@echo off
REM Echo Studio - one-time setup for Windows (double-click me).
REM Optional: pass a CUDA version, e.g.  setup.bat cu118   (cu118|cu121|cu124|cpu)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Cuda "%~1"
echo.
pause
