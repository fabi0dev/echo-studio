"""In-memory job manager: runs generations on a background worker thread."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, List, Optional

from PIL import Image

from .config import settings
from .pipeline import engine


@dataclass
class Job:
    id: str
    prompt: str
    params: dict
    state: str = "queued"          # queued|loading_model|running|done|error
    progress: float = 0.0
    step: int = 0
    total_steps: int = 0
    images: List[dict] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.created_at


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._queue: "Queue[str]" = Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # -------------------------------------------------------------- public
    def submit(self, prompt: str, params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], prompt=prompt, params=params,
                  total_steps=params["steps"] * params["num_images"])
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    # -------------------------------------------------------------- worker
    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._process(job)
            except Exception as exc:  # noqa: BLE001 - surfaced to client
                job.state = "error"
                job.error = str(exc)
                job.finished_at = time.time()
            finally:
                self._queue.task_done()

    def _process(self, job: Job) -> None:
        if not engine.loaded:
            job.state = "loading_model"
        else:
            job.state = "running"

        def on_step(done: int, total: int) -> None:
            job.state = "running"
            job.step = done
            job.total_steps = total
            job.progress = done / total if total else 0.0

        p = job.params
        init_image = self._decode_image(p.get("init_image"))
        results = engine.generate(
            prompt=job.prompt,
            negative_prompt=p.get("negative_prompt"),
            width=p["width"],
            height=p["height"],
            steps=p["steps"],
            guidance=p["guidance"],
            seed=p.get("seed"),
            num_images=p["num_images"],
            init_image=init_image,
            strength=p.get("strength", 0.65),
            on_step=on_step,
        )

        ts = time.strftime("%Y%m%d-%H%M%S")
        for idx, (image, seed) in enumerate(results):
            fname = f"{ts}_{job.id}_{idx}_{seed}.png"
            path = settings.OUTPUT_DIR / fname
            self._save_with_metadata(image, path, job, seed)
            job.images.append({
                "url": f"/outputs/{fname}",
                "seed": seed,
                "filename": fname,
            })

        job.progress = 1.0
        job.state = "done"
        job.finished_at = time.time()

    @staticmethod
    def _decode_image(data):
        """Decode a base64 (optionally data-URL) string into a PIL image."""
        if not data:
            return None
        import base64
        import io

        if "," in data and data.strip().startswith("data:"):
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data)
        return Image.open(io.BytesIO(raw))

    @staticmethod
    def _save_with_metadata(image: "Image.Image", path, job: Job, seed: int) -> None:
        from PIL import PngImagePlugin

        meta = PngImagePlugin.PngInfo()
        meta.add_text("prompt", job.prompt)
        meta.add_text("negative_prompt", str(job.params.get("negative_prompt") or ""))
        meta.add_text("seed", str(seed))
        meta.add_text("steps", str(job.params["steps"]))
        meta.add_text("guidance", str(job.params["guidance"]))
        meta.add_text("model", "Chroma Q4 (GGUF) via Echo Studio")
        image.save(path, format="PNG", pnginfo=meta)


jobs = JobManager()
