"""Chroma Q4 (GGUF) inference engine.

Wraps a diffusers ChromaPipeline whose transformer is loaded from a quantized
GGUF checkpoint. The heavy model is loaded lazily and guarded by a lock so a
single GPU serves one job at a time.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, List, Optional

from .config import settings

# torch/diffusers are imported lazily inside methods so the web server can boot
# (and answer /api/health) even on a box where the ML stack isn't installed yet.


def _resolve_device(requested: str):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(requested: str, device: str):
    import torch

    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    # auto
    if device == "cpu":
        return torch.float32
    return torch.bfloat16


class ChromaEngine:
    def __init__(self) -> None:
        self._pipe = None
        self._lock = threading.Lock()          # serializes generation
        self._load_lock = threading.Lock()     # serializes model loading
        self.device: str = "?"
        self.dtype_str: str = "?"
        self._loaded = False

    # ------------------------------------------------------------------ state
    @property
    def loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def gguf_is_local() -> bool:
        src = settings.CHROMA_GGUF
        return not src.startswith(("http://", "https://")) and Path(src).is_file()

    @staticmethod
    def gguf_present() -> bool:
        src = settings.CHROMA_GGUF
        if src.startswith(("http://", "https://")):
            return True  # remote; will be fetched on load
        return Path(src).is_file()

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return

            import torch
            from diffusers import (
                ChromaPipeline,
                ChromaTransformer2DModel,
                GGUFQuantizationConfig,
            )

            device = _resolve_device(settings.DEVICE)
            dtype = _resolve_dtype(settings.DTYPE, device)
            self.device = device
            self.dtype_str = str(dtype).replace("torch.", "")

            gguf_src = settings.CHROMA_GGUF
            if not gguf_src.startswith(("http://", "https://")) and not Path(gguf_src).is_file():
                raise FileNotFoundError(
                    f"Chroma GGUF not found at '{gguf_src}'. Run "
                    "`python scripts/download_models.py` or set ECHO_CHROMA_GGUF."
                )

            print(f"[echo] loading Chroma transformer (GGUF) from: {gguf_src}")
            transformer = ChromaTransformer2DModel.from_single_file(
                gguf_src,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )

            print(f"[echo] assembling ChromaPipeline from base: {settings.BASE_MODEL_ID}")
            pipe = ChromaPipeline.from_pretrained(
                settings.BASE_MODEL_ID,
                transformer=transformer,
                torch_dtype=dtype,
            )

            if device == "cuda" and settings.CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)

            if settings.VAE_TILING:
                try:
                    pipe.vae.enable_tiling()
                except Exception:  # noqa: BLE001 - optional optimization
                    pass

            self._pipe = pipe
            self._loaded = True
            print(f"[echo] model ready on {device} ({self.dtype_str})")

    # -------------------------------------------------------------- generate
    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: Optional[int],
        num_images: int,
        on_step: Optional[Callable[[int, int], None]] = None,
    ) -> List["object"]:
        """Run the diffusion loop. Returns a list of (PIL.Image, seed) tuples."""
        import torch

        self.load()

        # Snap to multiples of 16 (latent grid) and clamp to configured maximum.
        width = max(256, min(settings.MAX_SIDE, (width // 16) * 16))
        height = max(256, min(settings.MAX_SIDE, (height // 16) * 16))
        steps = max(1, min(settings.MAX_STEPS, steps))
        num_images = max(1, min(settings.MAX_BATCH, num_images))
        neg = negative_prompt if negative_prompt is not None else settings.DEFAULT_NEGATIVE

        results = []
        with self._lock:
            for i in range(num_images):
                img_seed = seed + i if seed is not None else int(torch.randint(0, 2**32 - 1, (1,)).item())
                generator = torch.Generator("cpu").manual_seed(img_seed)

                def _cb(_pipe, step_index, _timestep, cbk):
                    if on_step is not None:
                        # step_index is 0-based; report completed steps.
                        on_step(i * steps + step_index + 1, steps * num_images)
                    return cbk

                out = self._pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    num_images_per_prompt=1,
                    callback_on_step_end=_cb,
                )
                results.append((out.images[0], img_seed))
        return results


engine = ChromaEngine()
