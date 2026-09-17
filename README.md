# ✦ Echo Studio

A local, self-hosted image generator inspired by **Grok Imagine**, running
**DreamShaper 8** (Stable Diffusion 1.5) without a safety checker. Type a
prompt, pick an aspect ratio, hit **Gerar**, and everything is generated on
*your* machine — no API keys, nothing leaves the box.

> Built with a FastAPI backend (diffusers + SD 1.5) and a Vite + React
> TypeScript frontend. Layout: prompt bar on top, live progress, gallery grid below.

---

## ⚠️ Read this first

This repo is **ready to run**. DreamShaper 8 is an SD 1.5 checkpoint (~2 GB)
that fits 4–8 GB GPUs, including AMD via DirectML.

**Recommended target hardware**

| Setup | VRAM | Notes |
|-------|------|-------|
| **Windows + any DirectX 12 GPU** | **4 GB+** | AMD, Intel or NVIDIA. Setup installs DirectML (or CUDA if NVIDIA is present). |
| Linux + NVIDIA GPU | 4 GB+ | CUDA via `run.sh`. |
| Apple Silicon (M-series) | 8 GB+ unified | Works via `mps`. |
| CPU only (any OS) | — | Works, slower (tens of seconds per image). |

You need ~3 GB free disk for the checkpoint.

---

## 🚀 Quick start

You need **Python 3.10+** installed first. Everything else (virtualenv, the
right PyTorch build, dependencies and the ~2 GB model) is installed automatically.

### 🪟 Windows

1. Install Python from <https://www.python.org/downloads/> — **check "Add
   python.exe to PATH"** during install.
2. **Double-click `run.bat`.**

That's it. On the first run it detects your GPU, installs the matching PyTorch
(CUDA, DirectML for AMD/Intel, or CPU), installs everything, downloads
the model, then starts the app and opens your browser at
**http://localhost:8000**. Later runs just start the app.

> Prefer to install and start separately? Double-click **`setup.bat`** once,
> then **`run.bat`** whenever you want to use it.
>
> If Windows SmartScreen blocks the script, click *More info → Run anyway*, or
> run in PowerShell: `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1`.
>
> **Advanced (optional):** to force a backend, run `setup.bat directml`
> (or `cu121` / `cu118` / `cu124` / `cpu`). The default is auto-detect.

### 🍎 macOS / 🐧 Linux

```bash
./run.sh
```
Creates `.venv`, installs a matching torch build (CUDA if available, else
CPU/Apple-MPS), installs deps, downloads the model, and serves at
**http://localhost:8000**. To force a build: `ECHO_TORCH=cpu ./run.sh`.

<details>
<summary>Prefer manual steps? (any OS)</summary>

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# mac/linux: source .venv/bin/activate
pip install --upgrade pip
# Windows AMD/Intel: pip install torch-directml
# NVIDIA:            pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU / Apple:       pip install torch
pip install -r requirements.txt
python scripts/download_models.py            # downloads DreamShaper 8
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
</details>

### 🐳 Docker — NVIDIA (Linux / WSL2)
```bash
docker build -t echo-studio .
docker run --gpus all -p 8000:8000 \
  -v $PWD/models:/app/models -v $PWD/outputs:/app/outputs echo-studio
```

---

## 🎛️ Configuration

Copy `.env.example` → `.env` and tweak. Highlights:

| Var | Default | Meaning |
|-----|---------|---------|
| `ECHO_CHECKPOINT` | `models/dreamshaper-8.safetensors` | Local SD 1.5 checkpoint. |
| `ECHO_CHECKPOINT_URL` | Lykon DreamShaper 8 pruned | Where the downloader pulls the weights from. |
| `ECHO_DEVICE` | `auto` | `auto` → cuda → mps → DirectML → cpu. Force with `cuda`/`mps`/`directml`/`cpu`. |
| `ECHO_DML_DEVICE` | `auto` | DirectML GPU. `auto` prefers the dedicated card (RX 6600 over Vega iGPU). Index or name also work (`1`, `6600`). |
| `ECHO_DTYPE` | `auto` | `float32` on CPU/DirectML, `bfloat16` on CUDA/MPS. |
| `ECHO_CPU_OFFLOAD` | `true` | Offload modules to CPU between steps (CUDA, saves VRAM). |
| `ECHO_PRELOAD` | `false` | Load the model at boot instead of on first request. |

**Swapping the checkpoint:** point `ECHO_CHECKPOINT_URL` at any SD 1.5
`.safetensors` (Civitai/Hugging Face), delete `models/dreamshaper-8.safetensors`,
re-run the downloader. The CLIP safety checker is never loaded.

---

## 🧩 How it works

```
frontend/  (Vite + React + TS)  ──HTTP──►  FastAPI (backend/main.py)
                                        │
                                        ├─ jobs.py    background worker + progress
                                        └─ pipeline.py ImageEngine
                                                        │
              StableDiffusionPipeline.from_single_file( dreamshaper-8.safetensors )
                    └ CLIP text encoder + UNet + VAE  (safety checker off)
```

Dev UI: `cd frontend && npm run dev` (proxy `/api` → porta 8000).
Produção: FastAPI serve `frontend/dist`.

SD 1.5 at 512×512 fits AMD 8 GB cards via DirectML.

### API
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | model/device status, whether the checkpoint is present |
| `GET` | `/api/config` | default steps/guidance/limits for the UI |
| `POST` | `/api/generate` | `{prompt, negative_prompt, width, height, steps, guidance, seed, num_images}` → job |
| `GET` | `/api/jobs/{id}` | live progress (`queued`→`loading_model`→`running`→`done`) |
| `GET` | `/api/gallery` | recent outputs |
| `GET` | `/outputs/<file>` | generated PNGs (with prompt/seed embedded in metadata) |

Generation is serialized by a lock (one GPU = one job at a time); requests queue.

---

## ✨ UI features
- Prompt bar with **⌘/Ctrl + Enter** to generate.
- Aspect-ratio chips (1:1, 4:3, 3:2, 16:9, 2:3, 9:16) and 1/2/4 batch.
- Advanced panel: negative prompt, steps, guidance, fixed seed.
- Live per-step progress bar; shimmering placeholders while rendering.
- Gallery with lightbox, seed tags, and PNG download. Prompt + seed are saved
  into each PNG's metadata for reproducibility.

---

## 🩹 Troubleshooting
- **Windows: "python was not found"** → install Python from python.org and tick
  *Add python.exe to PATH*, then re-run `run.bat`. (The Microsoft Store alias can
  interfere — an actual python.org install is most reliable.)
- **Windows: script is blocked** → run `powershell -ExecutionPolicy Bypass -File
  scripts\setup.ps1`, or SmartScreen → *More info → Run anyway*.
- **"Checkpoint not found" / modelo ausente** → run `python scripts/download_models.py`
  (or on Windows, `.venv\Scripts\python scripts\download_models.py`).
- **CUDA / DirectML out of memory** → keep `ECHO_CPU_OFFLOAD=true`, lower resolution
  (try 512×512), reduce batch, or use a smaller side via `ECHO_MAX_SIDE`.
- **`from_single_file` import error** → `pip install -U diffusers omegaconf`
  (needs diffusers ≥ 0.40).
- **MPS errors on Mac** → set `ECHO_DTYPE=float32` (slower but robust), or
  `ECHO_DEVICE=cpu` as a fallback.
- **First generation is slow** → that's the one-time model load; set
  `ECHO_PRELOAD=true` to do it at startup instead.

---

## 📝 Notes & credits
- Grok Imagine also does *video*; Echo Studio is **text-to-image** (with img2img
  available). There is no CLIP safety checker.
- DreamShaper 8 by **Lykon**; inference via Hugging Face **diffusers**. Respect
  the model's license before publishing outputs.

Generated with [Claude Code](https://claude.com/claude-code).
