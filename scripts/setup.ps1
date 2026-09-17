param(
    # Which PyTorch backend to install:
    # auto | cu118 | cu121 | cu124 | directml | cpu
    # Omit for auto-detect. $env:ECHO_TORCH (or legacy $env:ECHO_CUDA) also work.
    [string]$Torch = "",
    [string]$Cuda = ""
)

# Echo Studio — Windows setup.
# Creates a virtualenv, installs a PyTorch build that matches the GPU
# (CUDA, DirectML, or CPU), installs dependencies, and downloads the model.

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== Echo Studio setup (Windows) ===" -ForegroundColor Cyan

function Test-NvidiaSmi {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { return $true }
    $paths = @(
        "$env:SystemRoot\System32\nvidia-smi.exe",
        "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    )
    foreach ($p in $paths) {
        if (Test-Path $p) { return $true }
    }
    return $false
}

function Test-DirectXGpu {
    $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    foreach ($gpu in @($gpus)) {
        $name = [string]$gpu.Name
        if ($name -and $name -notmatch "Microsoft Basic Display") {
            return $true
        }
    }
    return $false
}

function Resolve-TorchBackend([string]$choice) {
    if ($choice) { return $choice.ToLowerInvariant() }
    if (Test-NvidiaSmi) { return "cu121" }
    if (Test-DirectXGpu) { return "directml" }
    return "cpu"
}

function Invoke-Pip {
    & $script:venvPy -m pip @args
    if ($LASTEXITCODE -ne 0) {
        throw "pip failed: pip $($args -join ' ')"
    }
}

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
$script:venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $script:venvPy)) {
    Write-Host "-> criando ambiente virtual (.venv)" -ForegroundColor Yellow
    & $pyExe @pyArgs -m venv .venv
}

Write-Host "-> atualizando pip" -ForegroundColor Yellow
Invoke-Pip install --upgrade pip

Write-Host "-> instalando dependencias" -ForegroundColor Yellow
Invoke-Pip install -r requirements.txt
# transformers 4.x vs huggingface_hub 1.x: install without resolver.
Invoke-Pip install "transformers>=4.44,<5" --no-deps
# hf-xet hangs at 0 bytes on some Windows machines; HTTP fallback is reliable.
& $script:venvPy -m pip uninstall -y hf-xet 2>$null

# Torch last so the matching backend is not overwritten by a default wheel.
$choice = $Torch
if (-not $choice) { $choice = $Cuda }
if (-not $choice) { $choice = $env:ECHO_TORCH }
if (-not $choice) { $choice = $env:ECHO_CUDA }
if ($choice -and $choice.ToLowerInvariant() -eq "auto") { $choice = "" }

$backend = Resolve-TorchBackend $choice

switch ($backend) {
    "cu118" { $index = "https://download.pytorch.org/whl/cu118" }
    "cu121" { $index = "https://download.pytorch.org/whl/cu121" }
    "cu124" { $index = "https://download.pytorch.org/whl/cu124" }
    "cpu" { $index = "https://download.pytorch.org/whl/cpu" }
    "directml" { $index = $null }
    default {
        Write-Host "Backend invalido: '$backend' (use auto|cu118|cu121|cu124|directml|cpu)." -ForegroundColor Red
        exit 1
    }
}

if ($backend -eq "directml") {
    Write-Host "-> GPU DirectX detectada: instalando PyTorch + DirectML (AMD/Intel/NVIDIA)" -ForegroundColor Green
    try {
        Invoke-Pip install torch-directml
    } catch {
        Write-Host "-> DirectML indisponivel neste Python; caindo para CPU." -ForegroundColor Yellow
        $backend = "cpu"
        $index = "https://download.pytorch.org/whl/cpu"
        Invoke-Pip install torch --index-url $index
    }
} elseif ($backend -eq "cpu") {
    Write-Host "-> nenhuma GPU utilizavel detectada: instalando PyTorch (CPU)" -ForegroundColor Yellow
    Invoke-Pip install torch --index-url $index
} else {
    Write-Host "-> instalando PyTorch ($backend)" -ForegroundColor Green
    Invoke-Pip install torch --index-url $index
}

# --- Download the model ----------------------------------------------------
Write-Host "-> baixando DreamShaper 8 (~2 GB, so na primeira vez)" -ForegroundColor Yellow
& $script:venvPy -u scripts\download_models.py
if ($LASTEXITCODE -ne 0) {
    throw "download_models.py failed"
}

Write-Host ""
Write-Host "Setup concluido! Rode run.bat para iniciar." -ForegroundColor Cyan
