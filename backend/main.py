"""Echo Studio — FastAPI application.

Serves the web UI and a small JSON API around the Chroma Q4 engine.
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
from .pipeline import engine
from .schemas import GenerateRequest, HealthResponse, JobStatus

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

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
        base_model=settings.BASE_MODEL_ID,
        gguf=settings.CHROMA_GGUF,
        gguf_present=engine.gguf_present(),
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
    if not engine.gguf_present():
        raise HTTPException(
            status_code=503,
            detail=(
                "Chroma GGUF model is missing. Run "
                "`python scripts/download_models.py` on this machine first."
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
    }
    job = jobs.submit(req.prompt.strip(), params)
    return _to_status(job)


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _to_status(job)


@app.get("/api/gallery")
def gallery(limit: int = 60) -> dict:
    files = sorted(
        settings.OUTPUT_DIR.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    return {
        "images": [
            {"url": f"/outputs/{f.name}", "filename": f.name}
            for f in files
        ]
    }


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
