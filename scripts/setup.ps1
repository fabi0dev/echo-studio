param(
    # Which PyTorch build to install: cu118 | cu121 | cu124 | cpu | auto.
    # Omit to be asked interactively (or set $env:ECHO_CUDA).
    [string]$Cuda = ""
)

# Echo Studio — Windows setup.
# Creates a virtualenv, installs the chosen PyTorch build, installs
# dependencies, and downloads the model.
# Run via setup.bat, or:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Cuda cu121

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

# --- Decide which PyTorch build to install ---------------------------------
function Get-TorchIndex([string]$choice) {
    switch ($choice.ToLower()) {
        "cu118" { "https://download.pytorch.org/whl/cu118" }
        "cu121" { "https://download.pytorch.org/whl/cu121" }
        "cu124" { "https://download.pytorch.org/whl/cu124" }
        "cpu"   { "https://download.pytorch.org/whl/cpu" }
        default { $null }
    }
}

# Simple by default: auto-detect. Advanced override via -Cuda or $env:ECHO_CUDA
# (cu118 | cu121 | cu124 | cpu). No prompts.
$choice = $Cuda
if (-not $choice) { $choice = $env:ECHO_CUDA }
if ($choice -and $choice.ToLower() -eq "auto") { $choice = "" }

if (-not $choice) {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $choice = "cu121"  # sensible default for NVIDIA GPUs
    } else {
        Write-Host "-> Nenhuma GPU NVIDIA detectada." -ForegroundColor Yellow
        $choice = "cpu"
    }
}

$index = Get-TorchIndex $choice
if (-not $index) {
    Write-Host "Valor de CUDA invalido: '$choice' (use cu118|cu121|cu124|cpu)." -ForegroundColor Red
    exit 1
}

if ($choice -eq "cpu") {
    Write-Host "-> instalando PyTorch (CPU) — a geracao sera LENTA sem GPU" -ForegroundColor Yellow
} else {
    Write-Host "-> instalando PyTorch ($choice)" -ForegroundColor Green
}
& $venvPy -m pip install torch --index-url $index

# --- Install the rest ------------------------------------------------------
Write-Host "-> instalando dependencias" -ForegroundColor Yellow
& $venvPy -m pip install -r requirements.txt

# --- Download the model ----------------------------------------------------
Write-Host "-> baixando o modelo Chroma Q4 (~5 GB, so na primeira vez)" -ForegroundColor Yellow
& $venvPy scripts\download_models.py

Write-Host ""
Write-Host "Setup concluido! Rode run.bat para iniciar." -ForegroundColor Cyan
