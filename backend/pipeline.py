"""SD 1.5 inference engine (DreamShaper 8 by default).

Runs on CUDA, Apple MPS, DirectML (AMD/Intel on Windows) or CPU.
The CLIP safety checker is never loaded.
"""
from __future__ import annotations

import inspect
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional

from .config import settings
from .device import (
    describe_device,
    device_kind,
    resolve_dtype,
    resolve_offload_device,
    resolve_torch_device,
)
from .paths import is_checkpoint_ready


def _prepare_diffusers_import() -> None:
    """Make diffusers importable on torch 2.4.x (DirectML)."""
    import os

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import torch

    major, minor, *_ = (torch.__version__.split("+")[0].split(".") + ["0", "0"])[:2]
    if int(major) > 2 or (int(major) == 2 and int(minor) >= 5):
        _patch_sdpa_for_older_torch()
        return

    def _custom_op_noop(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func

        return wrap if fn is None else fn

    def _register_fake_noop(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        return wrap if fn is None else fn

    torch.library.custom_op = _custom_op_noop
    if hasattr(torch.library, "register_fake"):
        torch.library.register_fake = _register_fake_noop

    import sys
    from types import ModuleType

    if "torch.nn.attention.flex_attention" not in sys.modules:
        flex = ModuleType("torch.nn.attention.flex_attention")
        flex.BlockMask = type("BlockMask", (), {})
        flex.create_block_mask = lambda *args, **kwargs: None
        sys.modules["torch.nn.attention.flex_attention"] = flex

    import importlib.metadata as _metadata

    _orig_version = _metadata.version

    def _compat_version(name: str) -> str:
        if name in {"huggingface-hub", "huggingface_hub"}:
            installed = _orig_version(name)
            return "0.36.2" if installed.startswith("1.") else installed
        return _orig_version(name)

    _metadata.version = _compat_version
    _patch_sdpa_for_older_torch()


_SDPA_PATCHED = False


def _patch_sdpa_for_older_torch() -> None:
    """diffusers 0.40 passes enable_gqa; torch 2.4 SDPA does not accept it."""
    global _SDPA_PATCHED
    if _SDPA_PATCHED:
        return
    import torch
    import torch.nn.functional as F

    major, minor, *_ = (torch.__version__.split("+")[0].split(".") + ["0", "0"])[:2]
    if int(major) > 2 or (int(major) == 2 and int(minor) >= 5):
        _SDPA_PATCHED = True
        return

    original = F.scaled_dot_product_attention

    def scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        **_unused,
    ):
        if enable_gqa:
            q_heads = query.size(-3)
            kv_heads = key.size(-3)
            if q_heads != kv_heads:
                repeat = q_heads // kv_heads
                key = key.repeat_interleave(repeat, dim=-3)
                value = value.repeat_interleave(repeat, dim=-3)
        try:
            return original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
        except TypeError:
            return original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    F.scaled_dot_product_attention = scaled_dot_product_attention
    _SDPA_PATCHED = True


def _exc_text(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:
        return type(exc).__name__


def _to_device(value: Any, device: Any) -> Any:
    import torch

    if value is None:
        return None
    if not torch.is_tensor(value):
        return value
    if str(value.device) == str(device):
        return value
    # DirectML adapters cannot share tensors; hop through CPU.
    if device_kind(value.device) == "directml" and device_kind(device) == "directml":
        return value.cpu().to(device)
    return value.to(device)


def _has_params(module: Any) -> bool:
    getter = getattr(module, "parameters", None)
    if not callable(getter):
        return False
    try:
        next(getter())
        return True
    except (StopIteration, TypeError):
        return False


def _place_compute_modules(pipe: Any, device: Any) -> None:
    """Move UNet/CLIP onto the compute GPU. Leave the VAE where it is (CPU)."""
    components = getattr(pipe, "components", {}) or {}
    for name, module in components.items():
        if name == "vae" or module is None or not _has_params(module):
            continue
        module.to(device)


def _place_vae(pipe: Any, compute_device: Any) -> str:
    """Keep VAE on CPU under DirectML so the dGPU has room for UNet activations.

    Do not call ``vae.to()`` onto a second DirectML adapter: torch-directml
    deadlocks (and sometimes raises a garbled UnicodeDecodeError) when two
    GPUs share the process. ``vae.to(cpu)`` after a failed GPU move also
    deadlocks — leave weights on CPU and wrap decode/encode instead.
    """
    import torch

    vae = pipe.vae
    if device_kind(_module_device(vae)) != "cpu":
        vae.to("cpu")
    resolve_offload_device(compute_device)
    label = "cpu"

    try:
        vae.enable_slicing()
    except Exception:
        pass
    try:
        vae.enable_tiling()
    except Exception:
        pass

    vae_device = torch.device("cpu")
    original_decode = vae.decode
    original_encode = getattr(vae, "encode", None)

    def decode(latents, *args, **kwargs):
        out = original_decode(_to_device(latents, vae_device), *args, **kwargs)
        sample = out.sample if hasattr(out, "sample") else out[0] if isinstance(out, (tuple, list)) else out
        sample = _to_device(sample, "cpu")
        if hasattr(out, "sample"):
            out.sample = sample
            return out
        if isinstance(out, tuple):
            return (sample,) + out[1:]
        if isinstance(out, list):
            return [sample, *out[1:]]
        return sample

    vae.decode = decode
    if callable(original_encode):
        def encode(sample, *args, **kwargs):
            out = original_encode(_to_device(sample, vae_device), *args, **kwargs)
            dist = getattr(out, "latent_dist", None)
            if dist is not None:
                for attr in ("mean", "logvar", "std"):
                    tensor = getattr(dist, attr, None)
                    if tensor is not None and hasattr(tensor, "device"):
                        setattr(dist, attr, _to_device(tensor, compute_device))
            return out

        vae.encode = encode
    print(f"[echo] VAE on {label} (UNet on {describe_device(compute_device)})")
    return label


def _module_device(module: Any) -> Any:
    return next(module.parameters()).device


def _filter_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        accepted = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return kwargs
    if any(name == "kwargs" or name.startswith("**") for name in accepted):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in accepted}


class ImageEngine:
    def __init__(self) -> None:
        self._pipe = None
        self._img2img = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.device: str = "?"
        self.adapter: str = ""
        self.dtype_str: str = "?"
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def model_present() -> bool:
        src = settings.CHECKPOINT
        if src.startswith(("http://", "https://")):
            return True
        return is_checkpoint_ready(Path(src))

    def gguf_present(self) -> bool:
        return self.model_present()

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return

            ckpt = settings.CHECKPOINT
            if not ckpt.startswith(("http://", "https://")) and not is_checkpoint_ready(Path(ckpt)):
                raise FileNotFoundError(
                    f"Checkpoint not found at '{ckpt}'. Run "
                    "`python scripts/download_models.py` or set ECHO_CHECKPOINT."
                )

            _prepare_diffusers_import()
            import torch

            requested = resolve_torch_device(settings.DEVICE)
            dtype = resolve_dtype(settings.DTYPE, requested)
            pipe = self._build_pipe(ckpt, dtype)
            try:
                vae_label = self._place_pipe(pipe, requested, dtype)
                device = _module_device(pipe.unet)
            except Exception as exc:
                kind = device_kind(requested)
                if kind in {"cpu", "directml"}:
                    # DirectML already owns modules after a partial .to(); a
                    # second pipe.to(cpu) raises "resource deadlock would occur".
                    raise RuntimeError(
                        f"{kind} placement failed: {_exc_text(exc)}"
                    ) from exc
                print(f"[echo] {kind} failed ({_exc_text(exc)}); falling back to CPU")
                device = torch.device("cpu")
                dtype = torch.float32
                vae_label = self._place_pipe(pipe, device, dtype)

            self._pipe = pipe
            self.device = device_kind(device)
            self.adapter = describe_device(device)
            if vae_label and vae_label not in {self.adapter, "cpu"}:
                self.adapter = f"{self.adapter} · VAE {vae_label}"
            elif vae_label == "cpu" and self.device == "directml":
                self.adapter = f"{self.adapter} · VAE cpu"
            self.dtype_str = str(dtype).replace("torch.", "")
            self._loaded = True
            print(f"[echo] model ready on {self.adapter or self.device} ({self.dtype_str})")

    @staticmethod
    def _build_pipe(ckpt: str, dtype: Any):
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

        print(f"[echo] loading SD 1.5 checkpoint: {ckpt}")
        # Do not pass load_safety_checker at all: False still triggers the legacy
        # CompVis safety-checker download in diffusers 0.40.
        pipe = StableDiffusionPipeline.from_single_file(
            ckpt,
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )
        except TypeError:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
            )
        print("[echo] pipeline assembled (safety checker off)")
        return pipe

    @staticmethod
    def _place_pipe(pipe: Any, device: Any, dtype: Any) -> str:
        kind = device_kind(device)
        if kind == "cuda" and settings.CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        elif kind == "directml":
            _place_compute_modules(pipe, device)
        else:
            pipe.to(device)

        try:
            pipe.enable_attention_slicing("max" if kind == "directml" else "auto")
        except Exception:
            pass
        vae_label = ""
        if kind == "directml":
            vae_label = _place_vae(pipe, device)
        elif settings.VAE_TILING:
            for method in ("enable_vae_slicing", "enable_vae_tiling"):
                try:
                    getattr(pipe, method)()
                except Exception:
                    pass
        return vae_label

    def _img2img_pipe(self):
        if self._img2img is None:
            from diffusers import StableDiffusionImg2ImgPipeline

            self._img2img = StableDiffusionImg2ImgPipeline(**self._pipe.components)
            self._img2img.safety_checker = None
            self._img2img.requires_safety_checker = False
        return self._img2img

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
        init_image: Optional["object"] = None,
        strength: float = 0.65,
        on_step: Optional[Callable[[int, int], None]] = None,
    ) -> List["object"]:
        import time
        import torch

        already_loaded = self._loaded
        load_started = time.time()
        self.load()
        load_elapsed = 0.0 if already_loaded else time.time() - load_started
        if self._pipe is not None:
            self._pipe.set_progress_bar_config(disable=True)

        width = max(256, min(settings.MAX_SIDE, (width // 64) * 64))
        height = max(256, min(settings.MAX_SIDE, (height // 64) * 64))
        steps = max(1, min(settings.MAX_STEPS, steps))
        num_images = max(1, min(settings.MAX_BATCH, num_images))
        neg = negative_prompt if negative_prompt is not None else settings.DEFAULT_NEGATIVE

        is_img2img = init_image is not None
        if is_img2img:
            init_image = self._prepare_init_image(init_image, width, height)
            pipe = self._img2img_pipe()
            strength = max(0.05, min(1.0, strength))
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
        else:
            pipe = self._pipe

        results = []
        with self._lock:
            for i in range(num_images):
                img_seed = seed + i if seed is not None else int(torch.randint(0, 2**32 - 1, (1,)).item())
                generator = torch.Generator("cpu").manual_seed(img_seed)

                def _cb(_pipe, step_index, _timestep, cbk):
                    done = i * steps + step_index + 1
                    print(f"[echo] step {done}/{steps * num_images}")
                    if on_step is not None:
                        on_step(done, steps * num_images)
                    return cbk

                print(
                    f"[echo] generating {width}x{height} steps={steps} "
                    f"seed={img_seed} on {self.device}"
                )
                common = dict(
                    prompt=prompt,
                    negative_prompt=neg,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    num_images_per_prompt=1,
                    callback_on_step_end=_cb,
                    clip_skip=settings.CLIP_SKIP,
                )
                started = time.time()
                if is_img2img:
                    out = pipe(image=init_image, strength=strength, **_filter_kwargs(pipe.__call__, common))
                else:
                    out = pipe(**_filter_kwargs(pipe.__call__, common))
                elapsed = time.time() - started
                if i == 0:
                    elapsed += load_elapsed
                results.append((out.images[0], img_seed, elapsed))
        return results

    @staticmethod
    def _prepare_init_image(image, width: int, height: int):
        image = image.convert("RGB")
        if image.size != (width, height):
            from PIL import Image

            image = image.resize((width, height), Image.LANCZOS)
        return image


engine = ImageEngine()
