from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from gtsparse import _C
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor

from .metadata import GeometricTemplateMetadata


BM = 128


def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    return tuple(int(item) for item in value)


@dataclass(slots=True)
class Kernel9Runtime:
    out_rows: torch.Tensor
    input_rows_w1: torch.Tensor
    input_rows_w4: torch.Tensor
    input_rows_w7: torch.Tensor
    input_rows_w9: torch.Tensor
    template_ids: torch.Tensor
    input_row_offsets: torch.Tensor
    template_counts: torch.Tensor
    padded_counts: torch.Tensor
    out_coords: torch.Tensor
    coord_hashmap: torch.Tensor


def kernel9_conv(features, weight, runtime):
    fn = _C.gtsparse_kernel9_fp16_forward if features.dtype == torch.float16 else _C.gtsparse_kernel9_fp32_forward
    return fn(features, weight, runtime.out_rows, runtime.input_rows_w1, runtime.input_rows_w4,
              runtime.input_rows_w7, runtime.input_rows_w9, runtime.template_ids,
              runtime.input_row_offsets, runtime.out_coords.size(0))


class GeometricTemplateKernel9Conv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(3, 3, 1),
        *,
        stride=1,
        padding=(1, 1, 0),
        dilation=1,
        bias: bool = False,
        subm: bool = False,
        max_bm: int = BM,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.subm = bool(subm)
        self.transposed = False
        if self.kernel_size != (3, 3, 1):
            raise ValueError("kernel-9 specialization requires kernel_size=(3, 3, 1)")
        if self.in_channels % 32 != 0 or self.out_channels % 64 != 0:
            raise ValueError("kernel-9 setting requires Cin divisible by 32 and Cout divisible by 64")
        if self.subm and (self.stride != (1, 1, 1) or self.padding != (1, 1, 0)):
            raise ValueError("kernel-9 SubM requires stride=(1,1,1), padding=(1,1,0)")
        self.max_bm = int(max_bm)
        self.weight = nn.Parameter(torch.empty(9, self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.runtime_weight_cache: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.runtime_weight_cache = None
        bound = 1.0 / math.sqrt(float(self.in_channels * 9))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _runtime_weight(self) -> torch.Tensor:
        if self.runtime_weight_cache is None:
            self.runtime_weight_cache = self.weight.contiguous().view(9 * self.in_channels, self.out_channels)
        return self.runtime_weight_cache

    def train(self, mode: bool = True):
        self.runtime_weight_cache = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self.runtime_weight_cache = None
        return super()._load_from_state_dict(*args, **kwargs)

    def build_runtime(self, x: GTSparseSparseConvTensor):
        if self.subm:
            tensors = _C.gtsparse_kernel9_build_subm_runtime_from_coords(
                x.indices,
                self.dilation[0],
                self.dilation[1],
                self.max_bm,
                torch.empty(0, device=x.indices.device) if x.coord_hashmap is None else x.coord_hashmap,
            )
            return Kernel9Runtime(*tensors), tuple(x.spatial_shape)

        out_spatial = tuple(
            (int(size) + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
            for size, pad, dilation, kernel, stride in zip(
                x.spatial_shape,
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
            )
        )
        tensors = _C.gtsparse_kernel9_build_full_runtime_from_coords(
            x.indices,
            *out_spatial,
            *self.stride,
            *self.padding,
            self.dilation[0],
            self.dilation[1],
            self.max_bm,
            torch.empty(0, device=x.indices.device) if x.coord_hashmap is None else x.coord_hashmap,
        )
        return Kernel9Runtime(*tensors), out_spatial

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        runtime, out_spatial = self.build_runtime(x)
        out_features = kernel9_conv(x.features, self._runtime_weight(), runtime)
        if self.bias is not None:
            out_features = out_features + self.bias.view(1, -1)
        return x.replace_sparse(
            new_features=out_features,
            new_coords=runtime.out_coords,
            new_spatial_shape=out_spatial,
            new_batch_size=x.batch_size,
            coord_hashmap=runtime.coord_hashmap,
            metadata=GeometricTemplateMetadata(),
        )
