"""Runtime configuration for Echo Studio.

Everything is driven by environment variables so the same code runs on any
machine without edits. Copy `.env.example` to `.env` and adjust.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env sitting at the project root (one level above /backend), if present.
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _get(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


class Settings:
    """Central config object. Read once at import time."""

    # ----- Model sources ---------------------------------------------------
    # The base repo supplies the T5 text encoder, VAE, tokenizer and scheduler.
    # (from_pretrained on this repo loads a Diffusers-compatible Chroma.)
    BASE_MODEL_ID: str = _get("ECHO_BASE_MODEL_ID", "lodestones/Chroma1-HD")

    # The Q4 GGUF transformer. Point this at a local path (recommended) or a
    # remote HuggingFace blob URL. Default is the community Chroma-GGUF Q4_0.
    CHROMA_GGUF: str = _get(
        "ECHO_CHROMA_GGUF",
        # Filled in by scripts/download_models.py -> models/chroma-q4.gguf
        str(ROOT_DIR / "models" / "chroma-q4.gguf"),
    )
    # Fallback remote URL used by the downloader when the local file is missing.
    CHROMA_GGUF_URL: str = _get(
        "ECHO_CHROMA_GGUF_URL",
        "https://huggingface.co/silveroxides/Chroma-GGUF/resolve/main/"
        "Chroma1-HD/Chroma1-HD-Q4_0.gguf",
    )

    # ----- Hardware --------------------------------------------------------
    # "auto" -> cuda, else mps, else cpu. Force with "cuda" / "mps" / "cpu".
    DEVICE: str = _get("ECHO_DEVICE", "auto").lower()
    # "auto" | "bfloat16" | "float16" | "float32"
    DTYPE: str = _get("ECHO_DTYPE", "auto").lower()
    # Save VRAM by offloading modules to CPU between forward passes (CUDA only).
    CPU_OFFLOAD: bool = _get("ECHO_CPU_OFFLOAD", "true").lower() == "true"
    # Slice VAE decode to cut peak VRAM on large images.
    VAE_TILING: bool = _get("ECHO_VAE_TILING", "true").lower() == "true"

    # ----- Generation defaults / guardrails --------------------------------
    DEFAULT_STEPS: int = _get_int("ECHO_DEFAULT_STEPS", 28)
    MAX_STEPS: int = _get_int("ECHO_MAX_STEPS", 60)
    DEFAULT_GUIDANCE: float = float(_get("ECHO_DEFAULT_GUIDANCE", "3.0"))
    MAX_SIDE: int = _get_int("ECHO_MAX_SIDE", 1536)
    MAX_BATCH: int = _get_int("ECHO_MAX_BATCH", 4)
    DEFAULT_NEGATIVE: str = _get(
        "ECHO_DEFAULT_NEGATIVE",
        "low quality, ugly, unfinished, out of focus, deformed, disfigured, "
        "blurry, smudged, jpeg artifacts, watermark, text",
    )

    # ----- Server / storage ------------------------------------------------
    HOST: str = _get("ECHO_HOST", "0.0.0.0")
    PORT: int = _get_int("ECHO_PORT", 8000)
    OUTPUT_DIR: Path = Path(_get("ECHO_OUTPUT_DIR", str(ROOT_DIR / "outputs")))
    # Load the model the moment the server boots instead of on first request.
    PRELOAD: bool = _get("ECHO_PRELOAD", "false").lower() == "true"

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
