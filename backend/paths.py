"""Filesystem layout for Echo Studio models."""
from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "dreamshaper-8.safetensors"
DEFAULT_MODEL_ID = "Lykon/DreamShaper"
CHECKPOINT_URL = (
    "https://huggingface.co/Lykon/DreamShaper/resolve/main/"
    "DreamShaper_8_pruned.safetensors"
)
MIN_CHECKPOINT_BYTES = 1_000_000_000


def is_checkpoint_ready(path: Path | None = None) -> bool:
    target = path or CHECKPOINT_PATH
    return target.is_file() and target.stat().st_size >= MIN_CHECKPOINT_BYTES
