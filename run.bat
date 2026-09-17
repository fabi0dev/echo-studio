@echo off
REM Echo Studio - start the app on Windows (double-click me).
setlocal
cd /d "%~dp0"

REM First run? Install everything first.
if not exist ".venv\Scripts\python.exe" (
  echo Primeira execucao - rodando o setup...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
  if errorlevel 1 (
    echo.
    echo Setup falhou. Veja as mensagens acima.
    pause
    exit /b 1
  )
)

REM Make sure the model is present.
if not exist "models\chroma-q4.gguf" (
  echo Baixando o modelo Chroma Q4...
  ".venv\Scripts\python.exe" scripts\download_models.py
)

echo.
echo Iniciando Echo Studio em http://localhost:8000
echo (feche esta janela para parar o servidor)
REM Abre o navegador depois de alguns segundos, enquanto o servidor carrega.
start "" cmd /c "timeout /t 6 >nul & start "" http://localhost:8000"
".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
