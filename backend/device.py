"""Hardware-agnostic device selection.

``ECHO_DEVICE=auto`` order: CUDA → Apple MPS → DirectML → CPU.
DirectML covers AMD/Intel/NVIDIA GPUs on Windows via DirectX 12.
"""
from __future__ import annotations

from typing import Any


def device_kind(device: Any) -> str:
    text = str(device).lower()
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("mps"):
        return "mps"
    if "privateuseone" in text or text in {"dml", "directml"}:
        return "directml"
    if text.startswith("cpu"):
        return "cpu"
    return text


def _directml_device() -> Any | None:
    try:
        import torch_directml
    except ImportError:
        return None
    try:
        if hasattr(torch_directml, "is_available") and not torch_directml.is_available():
            return None
        return torch_directml.device()
    except Exception:
        return None


def resolve_torch_device(requested: str) -> Any:
    import torch

    aliases = {"dml": "directml"}
    choice = aliases.get((requested or "auto").lower().strip(), (requested or "auto").lower().strip())

    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        dml = _directml_device()
        return dml if dml is not None else torch.device("cpu")

    if choice == "cuda":
        return torch.device("cuda")
    if choice == "mps":
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "directml":
        dml = _directml_device()
        if dml is None:
            raise RuntimeError(
                "ECHO_DEVICE=directml but torch-directml is missing. "
                "Re-run setup or set ECHO_DEVICE=cpu."
            )
        return dml
    return choice


def resolve_dtype(requested: str, device: Any) -> Any:
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    choice = (requested or "auto").lower()
    if choice in mapping:
        return mapping[choice]
    if device_kind(device) in {"cpu", "directml"}:
        return torch.float32
    return torch.bfloat16
