# ✦ Echo Studio

A local, self-hosted image generator inspired by **Grok Imagine**, running the
open **Chroma** diffusion model in its **Q4 (GGUF) quantized** form. Type a
prompt, pick an aspect ratio, hit **Imaginar**, and everything is generated on
*your* machine — no API keys, nothing leaves the box.

> Built with a FastAPI backend (diffusers + Chroma Q4 GGUF) and a zero-build
> vanilla-JS frontend. Layout: prompt bar on top, live progress, gallery grid below.

---

## ⚠️ Read this first

This repo is **ready to run but was intentionally NOT run / not downloaded on the
machine it was authored on.** Chroma is an ~8.9B-parameter model; even at Q4 it
needs a real GPU. Do the setup below on the target machine.

**Recommended target hardware**

| Setup | VRAM | Notes |
|-------|------|-------|
| **Windows + NVIDIA GPU** | **8–12 GB+** | Recommended. `run.bat` auto-installs the CUDA build. |
| Linux + NVIDIA GPU | 8–12 GB+ | Same, via `run.sh`. Q4 + `enable_model_cpu_offload` fits ~8 GB. |
| Apple Silicon (M-series) | 16 GB+ unified | Works via `mps`, slower; GGUF dequant on MPS can be finicky. |
| CPU only (any OS) | — | Technically runs, but a single image can take *many minutes*. |

You also need ~15–20 GB free disk (Q4 transformer ≈ 5 GB, T5 text encoder + VAE ≈ 10 GB).

---

## 🚀 Quick start

You need **Python 3.10+** installed first. Everything else (virtualenv, the
right PyTorch build, dependencies and the ~5 GB model) is installed automatically.

### 🪟 Windows

1. Install Python from <https://www.python.org/downloads/> — **check "Add
   python.exe to PATH"** during install.
2. **Double-click `run.bat`.**

That's it. On the first run it detects your GPU, installs the correct PyTorch
(CUDA if you have an NVIDIA GPU, otherwise CPU), installs everything, downloads
the model, then starts the app and opens your browser at
**http://localhost:8000**. Later runs just start the app.

> Prefer to install and start separately? Double-click **`setup.bat`** once,
> then **`run.bat`** whenever you want to use it.
>
> If Windows SmartScreen blocks the script, click *More info → Run anyway*, or
> run in PowerShell: `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1`.

### 🍎 macOS / 🐧 Linux

```bash
./run.sh
```
Creates `.venv`, installs the right torch build (CUDA if an NVIDIA GPU is
present, else CPU/Apple-MPS), installs deps, downloads the model, and serves at
**http://localhost:8000**.

<details>
<summary>Prefer manual steps? (any OS)</summary>

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# mac/linux: source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # NVIDIA; or plain "pip install torch" for CPU/Mac
pip install -r requirements.txt
python scripts/download_models.py            # downloads Chroma Q4 + warms base repo
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
| `ECHO_BASE_MODEL_ID` | `lodestones/Chroma1-HD` | Repo providing T5 encoder, VAE, tokenizer, scheduler. |
| `ECHO_CHROMA_GGUF` | `models/chroma-q4.gguf` | Local path to the Q4 transformer. |
| `ECHO_CHROMA_GGUF_URL` | silveroxides Q4_0 | Where the downloader pulls the GGUF from. |
| `ECHO_DEVICE` | `auto` | `auto` → cuda → mps → cpu. Force with `cuda`/`mps`/`cpu`. |
| `ECHO_DTYPE` | `auto` | `bfloat16` on GPU, `float32` on CPU. |
| `ECHO_CPU_OFFLOAD` | `true` | Offload modules to CPU between steps (CUDA, saves VRAM). |
| `ECHO_PRELOAD` | `false` | Load the model at boot instead of on first request. |

**Swapping the Chroma version/quant:** point `ECHO_CHROMA_GGUF_URL` at any file in
[silveroxides/Chroma-GGUF](https://huggingface.co/silveroxides/Chroma-GGUF)
(e.g. a `chroma-unlocked-vNN-Q4_0.gguf`), delete `models/chroma-q4.gguf`, re-run
the downloader. Larger quants (Q5/Q8) = better quality, more VRAM.

---

## 🧩 How it works

```
frontend/  (vanilla JS)  ──HTTP──►  FastAPI (backend/main.py)
                                        │
                                        ├─ jobs.py    background worker + progress
                                        └─ pipeline.py ChromaEngine
                                                        │
              ChromaTransformer2DModel.from_single_file( chroma-q4.gguf,
                    GGUFQuantizationConfig )   ← the Q4 quantized transformer
                                                        │
              ChromaPipeline.from_pretrained( Chroma1-HD, transformer=… )
                    └ T5-XXL text encoder + VAE + scheduler
```

The Q4 weights stay in low-memory `uint8` and are dequantized on the fly during
each forward pass, which is what keeps VRAM low.

### API
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | model/device status, whether the GGUF is present |
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
- **"Chroma GGUF model is missing"** → run `python scripts/download_models.py`
  (or on Windows, `.venv\Scripts\python scripts\download_models.py`).
- **CUDA out of memory** → keep `ECHO_CPU_OFFLOAD=true`, lower resolution
  (try 832×1216), reduce batch, or use a smaller side via `ECHO_MAX_SIDE`.
- **`GGUFQuantizationConfig` import error** → `pip install -U diffusers gguf`
  (needs diffusers ≥ 0.40).
- **MPS errors on Mac** → set `ECHO_DTYPE=float32` (slower but robust), or
  `ECHO_DEVICE=cpu` as a fallback.
- **First generation is slow** → that's the one-time model load; set
  `ECHO_PRELOAD=true` to do it at startup instead.

---

## 📝 Notes & credits
- Grok Imagine also does *video*; Chroma is a **text-to-image** model, so Echo
  Studio focuses on images (with img2img/inpainting pipelines available in
  diffusers if you want to extend it).
- Chroma by **lodestones**; Q4 GGUF conversions by **silveroxides**; inference via
  Hugging Face **diffusers**. Respect each model's license before publishing outputs.

Generated with [Claude Code](https://claude.com/claude-code).
