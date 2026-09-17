# Echo Studio — Windows setup.
# Creates a virtualenv, installs the right PyTorch build (CUDA if an NVIDIA GPU
# is detected, otherwise CPU), installs dependencies, and downloads the model.
# Run via setup.bat, or:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== Echo Studio setup (Windows) ===" -ForegroundColor Cyan

# --- Find Python -----------------------------------------------------------
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pyExe = "py"; $pyArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pyExe = "python"; $pyArgs = @()
} else {
    Write-Host "Python 3.10+ nao encontrado." -ForegroundColor Red
    Write-Host "Instale em https://www.python.org/downloads/ (marque 'Add python.exe to PATH')."
    exit 1
}

# --- Create venv -----------------------------------------------------------
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "-> criando ambiente virtual (.venv)" -ForegroundColor Yellow
    & $pyExe @pyArgs -m venv .venv
}

Write-Host "-> atualizando pip" -ForegroundColor Yellow
& $venvPy -m pip install --upgrade pip

# --- Detect GPU and install torch -----------------------------------------
$hasGpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasGpu) {
    Write-Host "-> GPU NVIDIA detectada: instalando PyTorch (CUDA 12.1)" -ForegroundColor Green
    & $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "-> Nenhuma GPU NVIDIA detectada: instalando PyTorch (CPU)" -ForegroundColor Yellow
    Write-Host "   (a geracao sera LENTA sem GPU)" -ForegroundColor Yellow
    & $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}

# --- Install the rest ------------------------------------------------------
Write-Host "-> instalando dependencias" -ForegroundColor Yellow
& $venvPy -m pip install -r requirements.txt

# --- Download the model ----------------------------------------------------
Write-Host "-> baixando o modelo Chroma Q4 (~5 GB, so na primeira vez)" -ForegroundColor Yellow
& $venvPy scripts\download_models.py

Write-Host ""
Write-Host "Setup concluido! Rode run.bat para iniciar." -ForegroundColor Cyan
