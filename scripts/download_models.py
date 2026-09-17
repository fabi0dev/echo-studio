#!/usr/bin/env python3
"""Download the Chroma Q4 GGUF transformer and warm the base repo cache.

Run this ONCE on the target machine before starting the server:

    python scripts/download_models.py

It fetches:
  * the Q4 GGUF transformer  -> models/chroma-q4.gguf
  * the base repo components (T5 text encoder, VAE, tokenizer, scheduler)
    into the standard HuggingFace cache, so the first generation is fast.

Override the sources with env vars (see .env.example):
  ECHO_CHROMA_GGUF_URL, ECHO_BASE_MODEL_ID
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TARGET = MODELS_DIR / "chroma-q4.gguf"

GGUF_URL = os.getenv(
    "ECHO_CHROMA_GGUF_URL",
    "https://huggingface.co/silveroxides/Chroma-GGUF/resolve/main/"
    "Chroma1-HD/Chroma1-HD-Q4_0.gguf",
)
BASE_MODEL_ID = os.getenv("ECHO_BASE_MODEL_ID", "lodestones/Chroma1-HD")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"✓ already present: {dest} ({human(dest.stat().st_size)})")
        return

    # Prefer huggingface_hub when available (resumable, uses HF_TOKEN, Xet).
    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        if "huggingface.co/" in url and "/resolve/" in url:
            repo_part, file_part = url.split("huggingface.co/")[1].split("/resolve/")
            repo_id = repo_part
            filename = file_part.split("/", 1)[1]  # strip the "main/" ref
            print(f"↓ hf_hub_download {repo_id} :: {filename}")
            cached = hf_hub_download(repo_id=repo_id, filename=filename)
            Path(dest).write_bytes(b"")  # touch
            os.remove(dest)
            os.symlink(cached, dest) if hasattr(os, "symlink") else _copy(cached, dest)
            print(f"✓ linked -> {dest}")
            return
    except Exception as exc:  # noqa: BLE001 - fall back to plain HTTP
        print(f"  (huggingface_hub unavailable/failed: {exc}; using plain download)")

    print(f"↓ downloading {url}")
    req = Request(url, headers={"User-Agent": "echo-studio/0.1"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(req) as resp, open(tmp, "wb") as fh:  # noqa: S310 - trusted HF host
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            read += len(chunk)
            if total:
                pct = read / total * 100
                sys.stdout.write(f"\r  {human(read)} / {human(total)} ({pct:4.1f}%)")
                sys.stdout.flush()
    print()
    tmp.rename(dest)
    print(f"✓ saved -> {dest} ({human(dest.stat().st_size)})")


def _copy(src: str, dest: Path) -> None:
    import shutil

    shutil.copyfile(src, dest)


def warm_base_repo() -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:
        print("• huggingface_hub not installed; base repo will download on first run.")
        return
    print(f"↓ warming base repo cache: {BASE_MODEL_ID}")
    try:
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            allow_patterns=[
                "*.json", "*.txt", "*.model",
                "text_encoder/*", "tokenizer/*", "vae/*", "scheduler/*",
            ],
        )
        print("✓ base repo cached (text encoder / vae / tokenizer / scheduler)")
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped base warm: {exc})")


if __name__ == "__main__":
    print("Echo Studio — model downloader\n" + "-" * 34)
    download(GGUF_URL, TARGET)
    warm_base_repo()
    print("\nDone. Start the server with:  ./run.sh   (or see README)")
