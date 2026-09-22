from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor

from .metadata import ensure_metadata
from .ops import BM, full_conv_tensor, inverse_conv_tensor, subm_conv_tensor
from .weights import permute_weight_to_runtime_order


def _normalize_3tuple(value) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    values = tuple(int(v) for v in value)
    if len(values) != 3:
        raise ValueError("expected an int or length-3 sequence")
    return values


class GeometricTemplateSubMConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        *,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = False,
        max_bm: int = BM,
        sorted: bool = False,
    ) -> None:
        super().__init__()
        if int(groups) != 1:
            raise ValueError("geometric template runtime currently requires groups=1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _normalize_3tuple(kernel_size)
        self.stride = _normalize_3tuple(stride)
        self.padding = _normalize_3tuple(padding)
        self.dilation = _normalize_3tuple(dilation)
        if self.kernel_size != (3, 3, 3):
            raise ValueError("geometric template runtime currently requires kernel_size=(3, 3, 3)")
        if self.stride != (1, 1, 1):
            raise ValueError("SubM conv currently requires stride=(1, 1, 1)")
        self.max_bm = int(max_bm)
        self.sorted = bool(sorted)
        self.weight = nn.Parameter(torch.empty(27, self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.runtime_weight_cache: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.runtime_weight_cache = None
        bound = 1.0 / math.sqrt(float(self.in_channels * 27))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _runtime_weight(self) -> torch.Tensor:
        cached = self.runtime_weight_cache
        if cached is None:
            cached = permute_weight_to_runtime_order(self.weight.contiguous())
            self.runtime_weight_cache = cached
        return cached

    def train(self, mode: bool = True):
        self.runtime_weight_cache = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self.runtime_weight_cache = None
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        logical_weight = self._runtime_weight()
        out = subm_conv_tensor(
            x,
            logical_weight,
            kernel_size=self.kernel_size,
            padding=self.padding,
            dilation=self.dilation,
            max_bm=self.max_bm,
            sorted=self.sorted,
        )
        if self.bias is not None:
            out.replace_feature_(out.features + self.bias.view(1, -1))
        return out


class GeometricTemplateSparseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        *,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = False,
        max_bm: int = BM,
        build_reverse: bool = True,
        sorted: bool = False,
    ) -> None:
        super().__init__()
        if int(groups) != 1:
            raise ValueError("geometric template runtime currently requires groups=1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _normalize_3tuple(kernel_size)
        self.stride = _normalize_3tuple(stride)
        self.padding = _normalize_3tuple(padding)
        self.dilation = _normalize_3tuple(dilation)
        if self.kernel_size != (3, 3, 3):
            raise ValueError("geometric template runtime currently requires kernel_size=(3, 3, 3)")
        self.max_bm = int(max_bm)
        self.build_reverse = bool(build_reverse)
        self.sorted = bool(sorted)
        self.weight = nn.Parameter(torch.empty(27, self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.runtime_weight_cache: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.runtime_weight_cache = None
        bound = 1.0 / math.sqrt(float(self.in_channels * 27))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _runtime_weight(self) -> torch.Tensor:
        cached = self.runtime_weight_cache
        if cached is None:
            cached = permute_weight_to_runtime_order(self.weight.contiguous())
            self.runtime_weight_cache = cached
        return cached

    def train(self, mode: bool = True):
        self.runtime_weight_cache = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self.runtime_weight_cache = None
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        logical_weight = self._runtime_weight()
        out = full_conv_tensor(
            x,
            logical_weight,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            max_bm=self.max_bm,
            build_reverse=self.build_reverse,
            sorted=self.sorted,
        )
        if self.bias is not None:
            out.replace_feature_(out.features + self.bias.view(1, -1))
        return out


class GeometricTemplateSparseInverseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        *,
        groups: int = 1,
        bias: bool = False,
        max_bm: int = BM,
    ) -> None:
        super().__init__()
        if int(groups) != 1:
            raise ValueError("geometric template runtime currently requires groups=1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _normalize_3tuple(kernel_size)
        if self.kernel_size != (3, 3, 3):
            raise ValueError("geometric template runtime currently requires kernel_size=(3, 3, 3)")
        self.max_bm = int(max_bm)
        self.weight = nn.Parameter(torch.empty(27, self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.runtime_weight_cache: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.runtime_weight_cache = None
        bound = 1.0 / math.sqrt(float(self.in_channels * 27))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _runtime_weight(self) -> torch.Tensor:
        cached = self.runtime_weight_cache
        if cached is None:
            cached = permute_weight_to_runtime_order(self.weight.contiguous())
            self.runtime_weight_cache = cached
        return cached

    def train(self, mode: bool = True):
        self.runtime_weight_cache = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self.runtime_weight_cache = None
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        logical_weight = self._runtime_weight()
        out = inverse_conv_tensor(x, logical_weight, max_bm=self.max_bm)
        if self.bias is not None:
            out.replace_feature_(out.features + self.bias.view(1, -1))
        return out


def _is_sparse_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            GeometricTemplateSubMConv3d,
            GeometricTemplateSparseConv3d,
            GeometricTemplateSparseInverseConv3d,
            GeometricTemplateSparseSequential,
        ),
    )


class GeometricTemplateSparseSequential(nn.Sequential):
    def forward(self, x):
        for module in self:
            if isinstance(x, GTSparseSparseConvTensor):
                if _is_sparse_module(module):
                    x = module(x)
                else:
                    x.replace_feature_(module(x.features))
            else:
                x = module(x)
        return x
