"""Runtime configuration for Echo Studio.

Everything is driven by environment variables so the same code runs on any
machine without edits. Copy `.env.example` to `.env` and adjust.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .paths import CHECKPOINT_PATH, CHECKPOINT_URL, DEFAULT_MODEL_ID, ROOT_DIR

load_dotenv(ROOT_DIR / ".env")


def _get(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    return _get(key, "true" if default else "false").lower() == "true"


def _resolve_checkpoint() -> str:
    configured = _get("ECHO_CHECKPOINT", str(CHECKPOINT_PATH))
    if configured.startswith(("http://", "https://")):
        return configured
    path = Path(configured)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return str(path)


def _output_dir() -> Path:
    configured = _get("ECHO_OUTPUT_DIR", str(ROOT_DIR / "outputs"))
    path = Path(configured)
    return path if path.is_absolute() else ROOT_DIR / path


class Settings:
    """Central config object. Read once at import time."""

    MODEL_ID: str = _get("ECHO_MODEL_ID", DEFAULT_MODEL_ID)
    CHECKPOINT: str = _resolve_checkpoint()
    CHECKPOINT_URL: str = _get("ECHO_CHECKPOINT_URL", CHECKPOINT_URL)

    DEVICE: str = _get("ECHO_DEVICE", "auto").lower()
    DTYPE: str = _get("ECHO_DTYPE", "auto").lower()
    CPU_OFFLOAD: bool = _get_bool("ECHO_CPU_OFFLOAD", True)
    VAE_TILING: bool = _get_bool("ECHO_VAE_TILING", True)
    CLIP_SKIP: int = _get_int("ECHO_CLIP_SKIP", 2)

    DEFAULT_STEPS: int = _get_int("ECHO_DEFAULT_STEPS", 20)
    MAX_STEPS: int = _get_int("ECHO_MAX_STEPS", 60)
    DEFAULT_GUIDANCE: float = float(_get("ECHO_DEFAULT_GUIDANCE", "7.0"))
    MAX_SIDE: int = _get_int("ECHO_MAX_SIDE", 768)
    MAX_BATCH: int = _get_int("ECHO_MAX_BATCH", 4)
    DEFAULT_NEGATIVE: str = _get(
        "ECHO_DEFAULT_NEGATIVE",
        "low quality, ugly, unfinished, out of focus, deformed, disfigured, "
        "blurry, smudged, jpeg artifacts, watermark, text",
    )

    HOST: str = _get("ECHO_HOST", "0.0.0.0")
    PORT: int = _get_int("ECHO_PORT", 8000)
    OUTPUT_DIR: Path = _output_dir()
    PRELOAD: bool = _get_bool("ECHO_PRELOAD", False)

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
