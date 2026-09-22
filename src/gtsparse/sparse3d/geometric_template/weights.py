from __future__ import annotations

import torch

from .runtime import permute_weight_to_runtime_order as _permute_weight_to_runtime_order


def permute_weight_to_runtime_order(weight: torch.Tensor) -> torch.Tensor:
    return _permute_weight_to_runtime_order(weight)
