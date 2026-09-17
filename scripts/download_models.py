#!/usr/bin/env python3
"""Download the DreamShaper 8 SD 1.5 checkpoint.

Uses plain HTTP with resume. Hugging Face Xet/hub transfers hang at 0 bytes
on some Windows machines, so they are not used here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.paths import CHECKPOINT_PATH, CHECKPOINT_URL, MODELS_DIR  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODEL_URL = os.getenv("ECHO_CHECKPOINT_URL", CHECKPOINT_URL)


def human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000_000:
        print(f"already present: {dest} ({human(dest.stat().st_size)})")
        return

    tmp = dest.parent / f"{dest.name}.part"
    downloaded = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "echo-studio/0.2"}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
        print(f"resuming {dest.name} from {human(downloaded)}")
    else:
        print(f"downloading {dest.name}")

    req = Request(url, headers=headers)
    with urlopen(req) as resp, open(tmp, "ab" if downloaded else "wb") as fh:  # noqa: S310
        if resp.status == 200 and downloaded > 0:
            fh.seek(0)
            fh.truncate()
            downloaded = 0
        total_header = int(resp.headers.get("Content-Length", 0) or 0)
        total = downloaded + total_header if resp.status == 206 else (total_header or downloaded)
        read = downloaded
        last_pct = -1
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            read += len(chunk)
            if total:
                pct = int(read / total * 100)
                if pct != last_pct:
                    last_pct = pct
                    sys.stdout.write(f"\r  {human(read)} / {human(total)} ({pct:3d}%)")
                    sys.stdout.flush()
    print()
    os.replace(tmp, dest)
    print(f"saved -> {dest} ({human(dest.stat().st_size)})")


if __name__ == "__main__":
    print("Echo Studio - model downloader\n" + "-" * 34)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download(MODEL_URL, CHECKPOINT_PATH)
    print("\nDone. Start with run.bat (Windows) or ./run.sh")
