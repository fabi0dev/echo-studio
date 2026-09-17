"""Hardware-agnostic device selection.

``ECHO_DEVICE=auto`` order: CUDA → Apple MPS → DirectML → CPU.
DirectML covers AMD/Intel/NVIDIA GPUs on Windows via DirectX 12.
On hybrid laptops/desktops, adapter 0 is often the iGPU; we prefer the dGPU.
"""
from __future__ import annotations

import re
from typing import Any

from .config import settings


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


def _clean_adapter_name(name: str) -> str:
    return name.split("\x00", 1)[0].strip()


def _is_igpu_name(name: str) -> bool:
    text = name.lower()
    if "vega" in text:
        return True
    if re.search(r"radeon\(tm\) graphics", text) or re.search(r"radeon graphics$", text):
        return True
    return any(
        token in text
        for token in (
            "uhd",
            "iris",
            "hd graphics",
            "radeon 780m",
            "radeon 760m",
            "radeon 740m",
            "radeon 680m",
            "radeon 660m",
        )
    )


def _is_dgpu_name(name: str) -> bool:
    text = name.lower()
    if _is_igpu_name(name):
        return False
    if re.search(r"\brx\s*\d{3,4}\b", text):
        return True
    return any(
        token in text
        for token in ("rtx", "gtx", "geforce", "quadro", "arc a", "radeon pro", "radeon vii")
    )


def list_directml_adapters() -> list[tuple[int, str]]:
    try:
        import torch_directml
    except ImportError:
        return []
    try:
        if hasattr(torch_directml, "is_available") and not torch_directml.is_available():
            return []
        count = int(torch_directml.device_count())
    except Exception:
        return []
    adapters: list[tuple[int, str]] = []
    for index in range(count):
        try:
            name = _clean_adapter_name(str(torch_directml.device_name(index)))
        except Exception:
            name = f"adapter {index}"
        adapters.append((index, name))
    return adapters


def directml_index_of(device: Any) -> int | None:
    match = re.search(r":(\d+)", str(device))
    return int(match.group(1)) if match else None


def resolve_offload_device(compute_device: Any) -> Any:
    """Where the VAE should live so the compute GPU keeps VRAM for the UNet.

    torch-directml deadlocks if two adapters are used in the same process
    (``resource deadlock would occur`` / garbled native errors). The iGPU is
    listed, but VAE stays on CPU when the UNet is already on DirectML.
    """
    import torch

    if device_kind(compute_device) != "directml":
        return torch.device("cpu")
    adapters = list_directml_adapters()
    compute_index = directml_index_of(compute_device)
    extra = [name for index, name in adapters if index != compute_index]
    if extra:
        print(
            "[echo] extra GPU(s) present "
            f"({extra}) but DirectML cannot share a process; VAE stays on CPU"
        )
    return torch.device("cpu")


def describe_device(device: Any) -> str:
    kind = device_kind(device)
    if kind != "directml":
        return kind
    adapters = list_directml_adapters()
    text = str(device)
    match = re.search(r":(\d+)", text)
    if match:
        index = int(match.group(1))
        for adapter_index, name in adapters:
            if adapter_index == index:
                return f"directml:{index} {name}"
        return f"directml:{index}"
    if adapters:
        return f"directml:0 {adapters[0][1]}"
    return "directml"


def _pick_directml_index(requested: str) -> int:
    adapters = list_directml_adapters()
    if not adapters:
        return 0

    choice = (requested or "auto").strip().lower()
    if choice not in {"", "auto"}:
        if choice.isdigit():
            index = int(choice)
            if any(adapter_index == index for adapter_index, _ in adapters):
                return index
            raise RuntimeError(
                f"ECHO_DML_DEVICE={choice} is out of range. "
                f"Available: {adapters}"
            )
        for index, name in adapters:
            if choice in name.lower():
                return index
        raise RuntimeError(
            f"ECHO_DML_DEVICE={requested!r} did not match any adapter. "
            f"Available: {adapters}"
        )

    dedicated = [index for index, name in adapters if _is_dgpu_name(name)]
    if dedicated:
        return dedicated[0]
    discrete = [index for index, name in adapters if not _is_igpu_name(name)]
    if discrete:
        return discrete[-1]
    return adapters[-1][0]


def _directml_device() -> Any | None:
    try:
        import torch_directml
    except ImportError:
        return None
    try:
        if hasattr(torch_directml, "is_available") and not torch_directml.is_available():
            return None
        adapters = list_directml_adapters()
        if not adapters:
            return None
        index = _pick_directml_index(settings.DML_DEVICE)
        name = next((label for idx, label in adapters if idx == index), f"adapter {index}")
        print(f"[echo] DirectML adapters: {adapters}; using {index} {name}")
        return torch_directml.device(index)
    except Exception as exc:
        print(f"[echo] DirectML unavailable ({exc})")
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
