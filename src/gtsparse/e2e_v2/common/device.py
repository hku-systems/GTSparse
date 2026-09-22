from __future__ import annotations

import torch


def require_cuda_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")
    if resolved.type != "cuda":
        raise RuntimeError(f"expected a CUDA device, got {resolved}")
    return resolved


def resolve_runtime_dtype(name: str) -> torch.dtype:
    if str(name) == "fp16":
        return torch.float16
    if str(name) == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")
