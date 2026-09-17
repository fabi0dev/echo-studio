@echo off
REM Echo Studio - start the app on Windows (double-click me).
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_XET=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

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

if not exist "models\dreamshaper-8.safetensors" goto :download_models
goto :start

:download_models
echo Baixando os modelos (pode retomar se interromper)...
".venv\Scripts\python.exe" -u scripts\download_models.py
if errorlevel 1 (
  echo Download falhou. Rode run.bat de novo para retomar.
  pause
  exit /b 1
)

:start
if not exist "frontend\dist\index.html" (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo Instale Node.js para compilar a UI Vite.
    pause
    exit /b 1
  )
  echo Compilando o frontend...
  pushd frontend
  if not exist node_modules call npm install
  call npm run build
  if errorlevel 1 (
    popd
    echo Build do frontend falhou.
    pause
    exit /b 1
  )
  popd
)

echo.
echo Iniciando Echo Studio em http://localhost:8000
echo (feche esta janela para parar o servidor)
start "" cmd /c "timeout /t 6 >nul & start "" http://localhost:8000"
".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
