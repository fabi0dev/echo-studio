"""Echo Studio — FastAPI application.

Serves the web UI and a small JSON API around a local SD 1.5 engine.
Run with:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .jobs import jobs
from .paths import is_checkpoint_ready
from .pipeline import engine
from .schemas import GenerateRequest, HealthResponse, JobStatus

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
FRONTEND_DIR = FRONTEND_DIST if (FRONTEND_DIST / "index.html").is_file() else ROOT_DIR / "frontend"

app = FastAPI(title="Echo Studio", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    if settings.PRELOAD:
        # Load the model in the background so the server still answers /health.
        threading.Thread(target=_safe_preload, daemon=True).start()


def _safe_preload() -> None:
    try:
        engine.load()
    except Exception as exc:  # noqa: BLE001
        print(f"[echo] preload failed: {exc}")


# ----------------------------------------------------------------- API
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=engine.loaded,
        device=engine.device if engine.loaded else settings.DEVICE,
        dtype=engine.dtype_str if engine.loaded else settings.DTYPE,
        base_model=settings.MODEL_ID,
        gguf=settings.CHECKPOINT,
        gguf_present=engine.model_present(),
        base_ready=is_checkpoint_ready(),
    )


@app.get("/api/config")
def get_config() -> dict:
    return {
        "default_steps": settings.DEFAULT_STEPS,
        "max_steps": settings.MAX_STEPS,
        "default_guidance": settings.DEFAULT_GUIDANCE,
        "default_negative": settings.DEFAULT_NEGATIVE,
        "max_side": settings.MAX_SIDE,
        "max_batch": settings.MAX_BATCH,
    }


@app.post("/api/generate", response_model=JobStatus)
def generate(req: GenerateRequest) -> JobStatus:
    if not engine.model_present():
        raise HTTPException(
            status_code=503,
            detail=(
                "Checkpoint DreamShaper 8 ausente. Rode "
                "`python scripts/download_models.py` nesta máquina."
            ),
        )
    params = {
        "negative_prompt": req.negative_prompt,
        "width": req.width,
        "height": req.height,
        "steps": min(req.steps, settings.MAX_STEPS),
        "guidance": req.guidance,
        "seed": req.seed,
        "num_images": min(req.num_images, settings.MAX_BATCH),
        "init_image": req.init_image,
        "strength": req.strength,
    }
    job = jobs.submit(req.prompt.strip(), params)
    return _to_status(job)


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _to_status(job)


def _gallery_pngs() -> list[Path]:
    return sorted(
        settings.OUTPUT_DIR.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _safe_output_png(filename: str) -> Path:
    name = Path(filename).name
    if name != filename or not name.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="nome de arquivo inválido")
    if any(part in name for part in ("/", "\\", "..")):
        raise HTTPException(status_code=400, detail="nome de arquivo inválido")
    root = settings.OUTPUT_DIR.resolve()
    path = (root / name).resolve()
    if path.parent != root:
        raise HTTPException(status_code=400, detail="caminho inválido")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="imagem não encontrada")
    return path


@app.get("/api/gallery")
def gallery(limit: int = 60) -> dict:
    files = _gallery_pngs()[: max(1, min(limit, 200))]
    return {
        "images": [
            {"url": f"/outputs/{f.name}", "filename": f.name, **_read_png_meta(f)}
            for f in files
        ]
    }


@app.delete("/api/gallery")
def delete_gallery() -> dict:
    removed = 0
    for path in _gallery_pngs():
        path.unlink(missing_ok=True)
        removed += 1
    return {"ok": True, "removed": removed}


@app.delete("/api/gallery/{filename}")
def delete_image(filename: str) -> dict:
    path = _safe_output_png(filename)
    path.unlink()
    return {"ok": True, "filename": path.name}


def _read_png_meta(path) -> dict:
    """Pull the generation params we embedded when saving the PNG."""
    from PIL import Image

    def _num(val, cast):
        try:
            return cast(val)
        except (TypeError, ValueError):
            return None

    # Seed from filename as a fallback (…_<seed>.png).
    import re

    m = re.search(r"_(\d+)\.png$", path.name)
    meta = {
        "prompt": None,
        "negative_prompt": None,
        "seed": int(m.group(1)) if m else None,
        "steps": None,
        "guidance": None,
        "elapsed": None,
    }
    try:
        with Image.open(path) as im:
            t = getattr(im, "text", {}) or {}
        if t.get("prompt"):
            meta["prompt"] = t["prompt"]
        if t.get("negative_prompt"):
            meta["negative_prompt"] = t["negative_prompt"]
        if t.get("seed") is not None:
            meta["seed"] = _num(t["seed"], int) or meta["seed"]
        meta["steps"] = _num(t.get("steps"), int)
        meta["guidance"] = _num(t.get("guidance"), float)
        meta["elapsed"] = _num(t.get("elapsed"), float)
    except Exception:  # noqa: BLE001 - metadata is best-effort
        pass
    return meta


def _to_status(job) -> JobStatus:
    return JobStatus(
        id=job.id,
        state=job.state,
        progress=round(job.progress, 4),
        step=job.step,
        total_steps=job.total_steps,
        prompt=job.prompt,
        images=job.images,
        error=job.error,
        created_at=job.created_at,
        elapsed=round(job.elapsed, 2),
    )


# ------------------------------------------------------------- static files
# Generated images.
app.mount("/outputs", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="outputs")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# Frontend assets (css/js). Mounted last so it doesn't shadow /api or /outputs.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
